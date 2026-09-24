"""
Autoresearch pretraining script. Single-GPU, single-file.
Cherry-picked and simplified from nanochat.
Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import math
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from kernels import get_kernel
cap = torch.cuda.get_device_capability()
# varunneal's FA3 is Hopper only, use kernels-community on non-Hopper GPUs
repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
fa3 = get_kernel(repo).flash_attn_interface

from prepare import (MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, count_params,
                     make_dataloader, evaluate_bpb, report_efficiency_metrics,
                     TimeToTargetHarness)

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


# Where the value residual is SOURCED. The substrate gives (n_layer + 1) // 2 layers an owned
# nn.Embedding value table plus an input-dependent ve_gate, and selects those layers by PARITY,
# which at n_layer 7 is {0, 2, 4, 6}. This selects the same NUMBER of layers -- (n_layer + 1) // 2,
# four at n_layer 7 and four at n_layer 8 -- and places them at the two ENDS instead: {0, 1} and
# {5, 6} at n_layer 7. That is the arrangement of the lineage this file was cherry-picked and
# simplified from, where value embeddings sit on the first and last few blocks rather than on
# alternate blocks; the last layer is still always included, as the parent's docstring requires.
# Because the count, the ownership (one table and one gate per selected layer) and every per-layer
# shape are unchanged, all four declared instruments are unchanged BY CONSTRUCTION and are
# re-derivable here: 4 tables of 8192 * 512 = 16,777,216 params, 4 gates of 4 * 32 = 512 params,
# transformer_matrices = 7*(1,048,576 + 1,703,936) + 512 = 19,268,096, num_params_total =
# 4,194,304 + 16,777,216 + 4,194,304 + 19,268,096 + 14 = 44,433,934, dispatch matmul term =
# 6*(44,433,934 - 20,971,534) = 140,774,400, flash-attention tally = 12*4*128*5,376 = 33,030,144,
# counted total = 173,804,544, and the executed op set is the parent's -- four embedding gathers,
# four gate matmuls, four fused value adds, at different layer indices. r021's launch is the
# recorded reason this axis is worth one launch at all: changing WHICH layers CONSUME a value
# residual (all of them, sharing the four owned tables) moved that launch's recorded key by a whole
# probe interval, so the consumption pattern is not flat at rung scale. That is r021's measurement.
# Nothing here predicts this run's val_bpb at any probe, its recorded key, its per-update time or
# its allocator charge, in either direction.
def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (first and last layers, last included)."""
    n_ve = (n_layer + 1) // 2
    n_front = n_ve // 2
    return layer_idx < n_front or layer_idx >= n_layer - (n_ve - n_front)


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None
        # PER-HEAD output gate, and the mechanism of this card. The driver is the parent's -- the
        # same first ve_gate_channels = 32 channels of norm(x) the ve_gate reads, the same 2*sigmoid
        # map, the same zero init (see init_weights), the same position in the block -- and only the
        # TARGET's resolution moves: one (n_head, ve_gate_channels) = (4, 32) matrix per block, 128
        # params, 7 blocks = 896, against the parent's (512, 32) = 16,384 per block and 114,688. One
        # gate value now multiplies a whole head's head_dim = 128 channels (forward, below).
        # WHAT THIS DETERMINES, all re-derivable on CPU: per block 512*32 - 4*32 = 16,256 params
        # leave, so 7*16,256 = 113,792 leave the census; transformer params =
        # 7*(1,048,576 + 1,310,720 + 128) + 4*(4*32) = 16,515,968 + 512 = 16,516,480 and
        # num_params_total = 4,194,304 + 4,194,304 + 16,777,216 + 14 + 16,516,480 = 41,682,318
        # (8,649,858 under the 50,332,176 cap); the dispatch matmul term =
        # 6*(41,682,318 - 20,971,534) = 124,264,704 and the flash-attention tally is unchanged at
        # 12*4*128*2,688 = 16,515,072 (this diff passes no window_size and no new shape to
        # fa3.flash_attn_func), so flops_per_token_measured = 140,779,776 (98,298,624 under the
        # 239,078,400 ceiling). The gate's sigmoid and its multiply are elementwise and are not
        # counted by prepare.py's registry, as its metric description says. This shape also merges
        # the o_gates into the ve_gates' (4, 32) Muon group (11 matrices, multiplier
        # max(1, 4/32)**0.5 = 1.0), where the per-ELEMENT step is the parent's:
        # 0.02*2/sqrt(4*32) = 0.08*sqrt(32)/sqrt(512*32) for a polar-express output whose singular
        # values are ~1. Nothing here predicts this run's val_bpb at any probe, its recorded key,
        # which threshold it crosses on, its per-update time, its update rate or its allocator
        # charge, in either direction.
        self.o_gate = nn.Linear(self.ve_gate_channels, self.n_head, bias=False)

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        # Gate the attention output PER HEAD, conditioned on the block's own normed input x
        # (Block.forward passes norm(x)), read on the same first ve_gate_channels = 32 channels the
        # value-residual gate reads. The gate is applied on flash's own (B, T, n_head, head_dim)
        # layout BEFORE the view, so (B, T, n_head) values broadcast over head_dim = 128 and one
        # value scales a whole head; the view that follows is the parent's and still collapses
        # (n_head, head_dim) in that order, so no channel is reinterpreted. 2*sigmoid keeps the
        # neutral point at exactly 1.0 for a zero-initialised weight, so at init this line is the
        # identity on y bit for bit, exactly as the per-channel version was.
        y = y * (2 * torch.sigmoid(self.o_gate(x[..., :self.ve_gate_channels]))).unsqueeze(-1)
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, MLP_HIDDEN_DIM, bias=False)
        self.c_proj = nn.Linear(MLP_HIDDEN_DIM, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Value embeddings
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # Transformer blocks
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.attn.o_gate.weight)
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast embeddings to bf16
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=1024, device=None):
        # Truncated rotary spectrum. The head_dim // 2 = 64 channel pairs of a 128-dim head are
        # split in two: the first head_dim // 4 = 32 pairs rotate on a band running from
        # 1.0 rad/position down to 1/base rad/position, and the remaining 32 pairs are held at
        # exactly zero frequency, where cos = 1 and sin = 0 make apply_rotary_emb the identity
        # on them (y1 = x1 * 1 + x2 * 0 = x1, y2 = x1 * -0 + x2 * 1 = x2), so those channels
        # carry content that no position rotates. Shapes and dtypes are those of the geometric
        # spectrum this replaces: inv_freq still has head_dim // 2 entries and cos/sin are still
        # (1, seq_len, 1, head_dim // 2) bfloat16. This registers no parameter and reaches no
        # matmul and no flash-attention argument, so prepare.py's two counted terms are untouched.
        if device is None:
            device = self.transformer.wte.weight.device
        n_pair = head_dim // 2
        n_rot = head_dim // 4
        rotating = (1.0 / base) ** torch.linspace(0, 1, steps=n_rot, dtype=torch.float32, device=device)
        inv_freq = torch.cat([rotating, rotating.new_zeros(n_pair - n_rot)])
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        # Both tiers anchored on config.sequence_len so the two tiers move INDEPENDENTLY, carried
        # VERBATIM from candidates/r048-long-tier-1024/train.py lines 233-238: the incumbent
        # derives short_window from long_window, which is why r013-span-ladder-half could only
        # move both tiers together. LONG_WINDOW_DIVISOR = 2 -> long 1024 (r048's tier, carried),
        # LOCAL_WINDOW_DIVISOR = 16 -> short 128 (this card's tier).
        long_window = config.sequence_len // LONG_WINDOW_DIVISOR
        short_window = config.sequence_len // LOCAL_WINDOW_DIVISOR
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params))
        # Scale LR ∝ 1/√dmodel (tuned at 768 dim)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
        # CARRIED SETTLEMENT, executable text taken verbatim from
        # candidates/r066-granularity-keyed-step-size/train.py lines 346-353: batch_lr_scale keys
        # every group's step size to the token content of one logical update,
        # sqrt(TOTAL_BATCH_SIZE / LR_REFERENCE_BATCH_SIZE) = sqrt(65,536/262,144) = 0.5 exactly,
        # applied once after the groups are built and before initial_lr is recorded, so the training
        # loop's group["lr"] = group["initial_lr"] * lrm leaves the seconds-keyed envelope's SHAPE
        # untouched (WARMUP_RATIO, WARMDOWN_RATIO, FINAL_LR_FRAC, ANNEAL_SECONDS and
        # get_weight_decay are all untouched, so both 7 dp grids are the parent's) and only its
        # amplitude moves. The per-group print is carried with it so the BUILT program witnesses the
        # ten step sizes and the group table rather than this card arguing them.
        batch_lr_scale = (TOTAL_BATCH_SIZE / LR_REFERENCE_BATCH_SIZE) ** 0.5
        print(f"Scaling every group's LR by sqrt({TOTAL_BATCH_SIZE}/{LR_REFERENCE_BATCH_SIZE}) = {batch_lr_scale:.6f}")
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["lr"] = group["lr"] * batch_lr_scale
            group["initial_lr"] = group["lr"]
            print(f"  group {group['kind']:5s} n={len(group['params']):2d} "
                  f"shape={tuple(group['params'][0].shape)} lr={group['lr']:.10f}")
        return optimizer

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        # Rate carrier, carried VERBATIM from candidates/r048-long-tier-1024/train.py line 316
        # (itself r036-algebraic-softcap's line): same asymptote +/- softcap = +/- 15, same
        # f'(0) = 1 exactly, elementwise only. Carried so that
        # candidates/r048-long-tier-1024/train.py is this card's byte-matched control one literal
        # away and nothing is owed to separate a carrier.
        logits = logits * torch.rsqrt(1.0 + (logits * (1.0 / softcap)) ** 2)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
            return loss
        return logits

# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW, single GPU only)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    # Nesterov momentum
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    # Polar express orthogonalization
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    # NorMuon variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious weight decay + parameter update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(p, grad, state['exp_avg'], state['exp_avg_sq'],
                            self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                            self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim)
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
ASPECT_RATIO = 64       # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 128          # target head dimension for attention
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context

# MLP hidden width, in channels. 1664 = 13 * 128, ratio 3.25 of n_embd = 512, against the
# substrate's inherited 4 * config.n_embd = 2048. This is r020-mlp-width-1664's exact setting,
# re-applied unchanged, so r020 (this width, no autotuning) and r023 (this width, full
# max_autotune) are byte-matched controls on both sides of this card's one option key: the
# throughput split and the allocator split are each read against a sibling differing from this
# candidate by that key alone. It is carried because in-process benchmarking allocates while
# step 0's saved activations are live, and r020 recorded peak_vram_bytes 43,792,387,584 B at this
# width against the incumbent's 47,039,589,888 B -- those are r020's and r017's measurements, and
# NO claim is made here that either amount of room is enough for what this option's search
# allocates. Census and counted arithmetic at this width, all re-derivable: MLP c_fc 512*1664 +
# c_proj 1664*512 = 1,703,936 per block, so transformer_matrices = 8*(1,048,576 + 1,703,936) +
# 512 ve_gate params = 22,020,608 and num_params_total = 4,194,304 + 16,777,216 + 4,194,304 +
# 22,020,608 + 16 = 47,186,448, which is 3,145,728 under the 50,332,176 cap; counted total =
# 6*(47,186,448 - 20,971,536) + 12*4*128*5,632 = 157,289,472 + 34,603,008 = 191,892,480, which is
# 47,185,920 under the 239,078,400 ceiling. r020 measured this width's per-update quality cost at
# +0.004 to +0.008 bpb against the incumbent's width; that is r020's measurement and nothing is
# asserted here about this run.
# CARRIED SETTLEMENT, value taken from candidates/r063-mlp-width-1280-at-210/train.py: 1,280 =
# 10 * 128, ratio 2.5 of n_embd = 512. NOTE the inherited block above is written for 1,664 at DEPTH
# 8; on THESE bytes MLP c_fc is (1280, 512) and c_proj is (512, 1280), 1,310,720 params per block, so
# transformer params = 7*(1,048,576 + 1,310,720 + 16,384) + 4*(4*32) = 16,629,760 + 512 = 16,630,272
# and num_params_total = 4,194,304 + 4,194,304 + 16,777,216 + 14 + 16,630,272 = 41,796,110. It also
# re-keys c_fc's Muon group: setup_optimizer's max(1.0, shape[-2]/shape[-1])**0.5 gives
# 2.5**0.5 = 1.5811388 and effective lr 0.0632456 (7 dp) against the 3.25**0.5 = 1.8027756 and
# 0.0721110 of width 1,664 -- unchanged from r063's bytes, which is where this literal comes from.
MLP_HIDDEN_DIM = 1280

# Local attention span divisor: the sliding-window ("S") layers attend
# config.sequence_len // LOCAL_WINDOW_DIVISOR = 2048 // 8 = 256 keys to the left instead of
# the substrate's hard-coded half context, while the "L" layers (indices 3 and 7 under SSSL
# with the forced-long last layer) keep the full 2048 span. This is the exact setting
# r012-local-window-256 measured, re-applied unchanged: per-layer left spans
# (256, 256, 256, 2048, 256, 256, 256, 2048), span_sum 5,632. prepare.py's tally reads the
# window off the kernel call itself (kwargs.get("window_size", (-1, -1)), span =
# min(left, S)), so the counted attention term is 12 * n_head * head_dim * span_sum =
# 12 * 4 * 128 * 5,632 = 34,603,008 and the counted total is 176,163,840 + 34,603,008 =
# 210,766,848, which is 28,311,552 under the 239,078,400 ceiling. It is here because the
# anneal horizon below is keyed to a probe threshold and the number of updates that fit
# inside that threshold is a property of this setting: r012-local-window-256 recorded 735
# updates at the 270.0 s threshold against r009-rotary-band's 674. Those are those launches'
# measurements; nothing is claimed here about this run.
# The comment block above is written for r012's divisor 8 at DEPTH 8: on THESE bytes the divisor
# is 16 and the depth is 7, so the sliding-window ("S") layers attend
# config.sequence_len // LOCAL_WINDOW_DIVISOR = 2048 // 16 = 128 keys to the left while the two
# "L" layers (indices 3 and 6 under SSSL with the forced-long last layer) keep r048's
# config.sequence_len // LONG_WINDOW_DIVISOR = 2048 // 2 = 1024. Per-layer left spans become
# (128, 128, 128, 1024, 128, 128, 1024), span_sum 5*128 + 2*1024 = 2,688 against r048's 3,328.
# prepare.py reads the window off the kernel call itself (kwargs.get("window_size", (-1, -1)),
# span = min(left, S)), so the counted attention term is 12 * n_head * head_dim * span_sum =
# 12 * 4 * 128 * 2,688 = 16,515,072 and the counted total is 140,774,400 + 16,515,072 =
# 157,289,472, which is 81,788,928 under the 239,078,400 ceiling. This registers no parameter,
# changes no shape, no dtype and no tensor the file allocates: the only executed difference is the
# window_size kwarg five of the seven flash_attn_func calls run with. r012-local-window-256
# recorded -11.84% counted arithmetic bought +9.00% of clock on this tier at DEPTH 8 and a late
# matched-update cost of +0.0048 to +0.0061 bpb, and r048-long-tier-1024 recorded -7.24% bought
# +0.973% at +0.0012 bpb at its deciding probe; both are those launches' measurements, and nothing
# here claims anything about this run's per-update time, its update rate, its allocator charge or
# its quality at any probe, in either direction.
# THIS CARD'S ONE MECHANISM, and the only executable difference from incumbent/train.py:
# the short tier goes down one rung. GPT._compute_window_sizes reads short_window =
# config.sequence_len // LOCAL_WINDOW_DIVISOR = 2048 // 32 = 64 while long_window =
# config.sequence_len // LONG_WINDOW_DIVISOR = 2048 // 2 = 1024 is untouched, so under
# WINDOW_PATTERN "SSSL" at DEPTH 7 with the forced-long last layer the per-layer left spans
# become (64, 64, 64, 1024, 64, 64, 1024) and span_sum = 5*64 + 2*1024 = 2,368 against the
# parent's 2,688. WHAT THIS DETERMINES, all re-derivable on CPU: this registers no parameter
# and changes no channel count, dtype, tensor size or matmul shape, so num_params_total stays
# 41,682,318 and the dispatch matmul term stays 6*(41,682,318 - 20,971,534) = 124,264,704;
# prepare.py's flash tally is 12*B*T*h*d*min(left, S) per call over its own
# FLOPS_PROBE_ROWS = 8 rows of T = 1,024 (prepare.py:511, 545-557, 588-590), and at
# S = k.shape[1] = 1,024 both min(64, 1024) = 64 and min(1024, 1024) = 1024 hold, so the
# per-token attention term is 12*4*128*2,368 = 14,548,992 and flops_per_token_measured =
# 124,264,704 + 14,548,992 = 138,813,696 (100,264,704 under the 239,078,400 ceiling), equal to
# GPT.estimate_flops()'s analytic value because t = config.sequence_len = 2048 leaves both
# min(w, t) terms unchanged. TOTAL_BATCH_SIZE, DEVICE_BATCH_SIZE, TRAIN_SEQ_LEN,
# DATA_BUDGET_TOKENS, ANNEAL_SECONDS, every learning-rate literal, setup_optimizer's
# batch_lr_scale and the harness construction are untouched, so grad_accum_steps stays 1,
# tokens_per_step stays 65,536, training_data_tokens_available stays 631,241,817,
# target_val_bpb stays 1.05 and both 7 dp schedule grids are the parent's. WHAT ACTUALLY
# CHANGES: five of the seven flash_attn_func calls run with window_size = (64, 0) instead of
# (128, 0), and averaged over a 1,024-position row the reachable left context of those layers
# falls from (64*63/2 + 960*64 is the new one; the old is 128*127/2 + 896*128) 119.9375 to
# 61.96875 positions. Nothing here predicts this run's val_bpb at any probe, its recorded key,
# which probe threshold it crosses on, its per-update time, its update rate, its update supply
# or its allocator charge, in either direction.
LOCAL_WINDOW_DIVISOR = 32

# Long attention tier divisor, carried from candidates/r048-long-tier-1024/train.py line 521:
# the two "L" layers attend config.sequence_len // LONG_WINDOW_DIVISOR = 2048 // 2 = 1024 keys.
# Carried unchanged so that candidates/r048-long-tier-1024/train.py is this card's byte-matched
# control ONE LITERAL away (the LOCAL_WINDOW_DIVISOR line above), and so the tier this card moves
# is the short one alone.
LONG_WINDOW_DIVISOR = 2

# Optimization
# UPDATE GRANULARITY, the mechanism of this card. TOTAL_BATCH_SIZE is the token content of one
# logical optimizer update -- the count train.py hands prepare.TimeToTargetHarness.tick once per
# completed update -- and it has been 2**18 since r001-batch-updates halved it from the reference's
# 2**19 on a 12-block, width-768, 50,332,176-parameter model. The model this file now races is 7
# blocks at width 512 with 41,796,110 parameters and 141,462,528 counted FLOPs per token, and the
# token content of an update has never moved since. 2**17 = 131,072 tokens per update, with
# DEVICE_BATCH_SIZE 128 and TRAIN_SEQ_LEN 1024 below, keeps DEVICE_BATCH_SIZE * TRAIN_SEQ_LEN ==
# TOTAL_BATCH_SIZE, so grad_accum_steps stays 1, one microbatch is one whole update, and tick still
# receives exactly one synchronized duration and one token count per completed update with none of
# the update's work left outside the timed window.
# WHAT CANNOT MOVE. No parameter is registered and no channel count, dtype, window or model shape
# changes, so num_params_total stays 41,796,110 and flops_per_token_measured stays
# 6*(41,796,110 - 20,971,534) + 12*4*128*2,688 = 124,947,456 + 16,515,072 = 141,462,528;
# prepare.measure_flops_dispatch normalises by x[:FLOPS_PROBE_ROWS].numel() (prepare.py:588-590,
# 604), so the probe is per-token and batch-invariant at fixed T; DATA_BUDGET_TOKENS is untouched,
# so training_data_tokens_available stays 631,241,817; AUTORESEARCH_TARGET_VAL_BPB still reaches the
# frozen harness unchanged, so target_val_bpb stays 1.05; and prepare.evaluate_bpb still certifies
# at its own frozen EVAL_BATCH_SIZE x MAX_SEQ_LEN = 128 x 2048 with its own val loader, which
# train.py never touches.
# WHAT ACTUALLY CHANGES. Each optimizer update is computed from half as many tokens, so the same
# training clock is discretised into a different number of updates and each update's gradient is
# estimated from half the sample. setup_optimizer scales its learning rates by model width alone
# ((n_embd/768)**-0.5) and nothing in this file scales any learning rate by batch, while Muon's
# polar-express step is spectrally normalised per update, so at fixed LR the parameter movement made
# per TOKEN doubles. The frozen harness also holds its first 10 updates outside its clock, so the
# tokens those 10 updates carry for free halve. That is the mechanism, not a defect.
# Nothing here predicts this run's val_bpb at any probe, its recorded key, which probe threshold it
# crosses on, num_steps, total_tokens, per-update time, update rate or allocator charge, in either
# direction.
# THIRD RUNG ON THIS AXIS, and the mechanism of this card. The block above is written for 2**17 with
# DEVICE_BATCH_SIZE 128; on THESE bytes TOTAL_BATCH_SIZE = 2**16 = 65,536 tokens per logical update
# and DEVICE_BATCH_SIZE = 64 below, so DEVICE_BATCH_SIZE * TRAIN_SEQ_LEN = 64 * 1024 = 65,536 ==
# TOTAL_BATCH_SIZE, grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd = 1, one microbatch is
# one whole update, and prepare.TimeToTargetHarness.tick still receives exactly one synchronized
# duration and one token count (65,536) per completed update with none of that update's work outside
# the timed window. WHAT CANNOT MOVE: no parameter is registered and no channel count, dtype, window
# or model shape changes, so num_params_total stays 41,796,110 and flops_per_token_measured stays
# 6*(41,796,110 - 20,971,534) + 12*4*128*2,688 = 124,947,456 + 16,515,072 = 141,462,528;
# prepare.measure_flops_dispatch slices its own first FLOPS_PROBE_ROWS = 8 rows (prepare.py:511,
# 588-589) and normalises by their token count, 8 * 1,024 = 8,192, so the probe stays per-token and
# batch-invariant at fixed T; DATA_BUDGET_TOKENS is untouched, so training_data_tokens_available
# stays 631,241,817; AUTORESEARCH_TARGET_VAL_BPB still reaches the frozen harness, so target_val_bpb
# stays 1.05; ANNEAL_SECONDS, WARMUP_RATIO, WARMDOWN_RATIO and FINAL_LR_FRAC are untouched, so the
# seconds-keyed envelope is the parent's to 7 dp (lrm 1.0/1.0/1.0/0.8714286/0.6142857/0.3571429/0.1
# and Muon weight decay 0.1714286/0.1428571/0.1142857/0.0857143/0.0571429/0.0285714/0.0 at the
# 30 s ... 210 s thresholds); and prepare.evaluate_bpb still certifies at its own frozen
# EVAL_BATCH_SIZE x MAX_SEQ_LEN = 128 x 2048 with its own val loader, which train.py never touches.
# WHAT ACTUALLY CHANGES: each update is estimated from half as many tokens, so at fixed learning
# rates -- setup_optimizer scales by model width alone and nothing here scales by batch, while
# Muon's polar-express step is spectrally normalised per update -- the parameter movement made per
# TOKEN doubles again; the harness's ten free warmup updates carry half the tokens they carried; and
# get_muon_momentum keys its 300-update ramp from 0.85 to 0.95 on `step`, so that ramp now spans half
# the tokens it spanned on the parent. Those ride WITH the granularity, exactly as they rode with the
# parent's own halving, and are named in this card's risks rather than separated.
# Nothing here predicts this run's val_bpb at any probe, its recorded key, which probe threshold it
# crosses on, num_steps, total_tokens, per-update time, update rate, update supply or allocator
# charge, in either direction.
TOTAL_BATCH_SIZE = 2**16 # 65,536 tokens per optimizer step: exactly one 64 x 1024 microbatch, no accumulation
# Granularity reference for setup_optimizer's batch_lr_scale, carried with it from
# candidates/r066-granularity-keyed-step-size/train.py line 663: the token content of one logical
# update at which every learning-rate literal in this file was last measured. 2**18 = 262,144 stood
# from r001 through r063 (r044's UNEMBEDDING_LR 0.008 was measured at it, and EMBEDDING_LR,
# MATRIX_LR and SCALAR_LR carried through it unchanged) while r064 and r065 moved the granularity to
# 2**17 and then 2**16 with every literal left where it was. This is the denominator of that ratio
# and nothing else: it enters no shape, no dtype, no counted term and no census bucket, and it is
# read only inside setup_optimizer.
LR_REFERENCE_BATCH_SIZE = 2**18
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.008  # learning rate for lm_head (Adam); carried from r044-unembedding-lr-008
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of the anneal horizon for LR warmup
WARMDOWN_RATIO = 0.5    # fraction of the anneal horizon for LR warmdown
FINAL_LR_FRAC = 0.1     # LR floor as fraction of initial, held past the anneal horizon
# Anneal horizon, in TRAINING SECONDS on the harness clock (total_training_time, which
# skips the first 10 updates exactly as the harness does). The LR and weight-decay schedules
# are keyed to THIS, not to TIME_BUDGET. TIME_BUDGET remains the stop rule and the
# `remaining` log; this only changes what the anneal is stretched over. The frozen harness
# (prepare.TimeToTargetHarness.tick, prepare.PROBE_EVERY_SECONDS = 30.0, warmup_steps = 10,
# full_eval = True) accumulates each update's own duration and evaluates when that clock
# first reaches its next threshold, advancing the threshold by 30.0 s each time, so a
# reading can only be taken at a multiple of 30.0 s of this clock. The RECORDED instant is
# the clock after the update that crossed the threshold, i.e. it overshoots by up to one
# update duration (0.057-0.347 s observed across launches of this run), so the thresholds
# are the grid and the recorded values are not multiples of 30.
#
# 240.0 moves the horizon down one probe interval, from the 270.0 s threshold to the 240.0 s
# one, so the envelope completes ON the 240.0 s reading instead of standing mid-warmdown
# there. With WARMUP_RATIO 0.0, WARMDOWN_RATIO 0.5 and FINAL_LR_FRAC 0.1, get_lr_multiplier
# holds lrm = 1.0 through 120.0 s (progress 0.5) and then descends linearly to the 0.1 floor
# exactly at 240.0 s, and get_weight_decay = 0.2 * (1 - progress) reaches exactly 0.0 there.
# At the 30 s ... 240 s thresholds this envelope is lrm 1.0 / 1.0 / 1.0 / 1.0 / 0.775 / 0.55 /
# 0.325 / 0.1 with Muon weight decay 0.175 / 0.15 / 0.125 / 0.1 / 0.075 / 0.05 / 0.025 / 0.0.
# The horizon this replaces (270.0) is lrm 1.0 / 1.0 / 1.0 / 1.0 / 0.9 / 0.7 / 0.5 / 0.3 / 0.1
# with weight decay 0.177778 / 0.155556 / 0.133333 / 0.111111 / 0.088889 / 0.066667 /
# 0.044444 / 0.022222 / 0.0 at the 30 s ... 270 s thresholds, i.e. it stands at lrm 0.3 and
# weight decay 0.022222 at the 240.0 s reading and reaches its floor only at 270.0 s.
# Integrated LR-seconds delivered by the 240.0 s threshold therefore fall 203.25 -> 186.0
# (-8.49%): 135.0 s of hold plus 105.0 s descending 1.0 -> 0.3 (mean 0.65, 68.25) becomes
# 120.0 s of hold plus 120.0 s descending 1.0 -> 0.1 (mean 0.55, 66.0). All of the reduction
# is taken out of the second half of the envelope. Past the horizon `progress` clamps to 1.0,
# so the floor lrm 0.1 and weight decay 0.0 are held from 240.0 s onward instead of 270.0 s.
#
# Nothing is claimed here about where this run crosses, about what any probe of this run
# reads, about updates, tokens or steps, or about per-update time or allocator charge in
# either direction. Numbers from other launches are their measurements, not predictions about
# this one: r010-anneal-270 recorded that a horizon whose update content runs out long before
# the crossing gives the compressed envelope's gain back over the updates past it (its
# matched-update delta against its parent went -0.025250 at 520 updates and -0.022873 at 584,
# its own horizon, then -0.013981 at 649, -0.003932 at 715 and +0.001128 at 748, with
# post-horizon descent of 0.0000835 bpb per update), and r015-anneal270-span256 recorded the
# same shape one interval up on this same span (-0.015183 at its 240 s probe, which is 89% of
# its horizon in updates, decaying to -0.012758 at its own 270.0 s threshold, both against the
# byte-identical-span control r012-local-window-256 at matched update count). That is the
# shape this constant selects -- the gain is concentrated at and before the horizon and is
# repaid over the updates past it -- so it is worth a launch only when the crossing can land
# at or before the horizon, which is r010's precondition and is not asserted here.
# CARRIED SETTLEMENT, value taken from candidates/r062-cross-at-210/train.py: the seconds-keyed LR
# and weight-decay envelope completes at the 210.0 s probe threshold instead of the 240.0 s one.
# NOTE the inherited block above is written for 240.0. At WARMUP_RATIO 0.0, WARMDOWN_RATIO 0.5 and
# FINAL_LR_FRAC 0.1 this envelope reads, TO 7 DECIMAL PLACES, lrm 1.0 / 1.0 / 1.0 / 0.8714286 /
# 0.6142857 / 0.3571429 / 0.1 at the 30 s ... 210 s thresholds, with Muon weight decay
# 0.2*(1 - progress) = 0.1714286 / 0.1428571 / 0.1142857 / 0.0857143 / 0.0571429 / 0.0285714 / 0.0,
# and `progress` clamps to 1.0 past 210.0 s so the floor lrm 0.1 and weight decay 0.0 are held from
# there on. TIME_BUDGET remains the stop rule and the `remaining` log; this only changes what the
# anneal is stretched over.
# THIS CARD'S ONE MECHANISM, and the only executable difference from incumbent/train.py: the
# seconds-keyed LR and weight-decay envelope is re-keyed to the SIXTH probe threshold instead of the
# seventh, so it completes ON the 180.0 s reading rather than standing mid-warmdown there.
# prepare.TimeToTargetHarness advances next_probe_at by PROBE_EVERY_SECONDS = 30.0 from 30.0
# (prepare.py:463, 660-684) and holds its first 10 updates outside that clock, so 180.0 s is the
# sixth threshold of that grid and `progress = min(total_training_time / ANNEAL_SECONDS, 1.0)` now
# reaches 1.0 exactly there.
# WHAT THIS DETERMINES, all re-derivable on CPU: this registers no parameter and changes no channel
# count, dtype, window, tensor size or matmul shape, so num_params_total stays 4,194,304 + 4,194,304
# + 16,777,216 + 14 + 16,516,480 = 41,682,318 and flops_per_token_measured stays
# 6*(41,682,318 - 20,971,534) + 12*4*128*2,368 = 124,264,704 + 14,548,992 = 138,813,696 at span_sum
# 2,368. TOTAL_BATCH_SIZE, DEVICE_BATCH_SIZE, TRAIN_SEQ_LEN, DATA_BUDGET_TOKENS,
# LR_REFERENCE_BATCH_SIZE, every learning-rate literal, WARMUP_RATIO, WARMDOWN_RATIO,
# FINAL_LR_FRAC, TROUGH_SECONDS, setup_optimizer's batch_lr_scale = 0.5 and the harness construction
# are untouched, so grad_accum_steps stays 1, tokens_per_step stays 65,536,
# training_data_tokens_available stays 631,241,817, target_val_bpb stays 1.05 and every group's
# initial_lr is the parent's to 10 dp.
# WHAT ACTUALLY CHANGES: with WARMUP_RATIO 0.0, WARMDOWN_RATIO 0.5 and FINAL_LR_FRAC 0.1,
# get_lr_multiplier holds lrm = 1.0 through 90.0 s (progress 0.5) and then descends linearly to the
# 0.1 floor exactly at 180.0 s, reading TO 7 DECIMAL PLACES 1.0 / 1.0 / 1.0 / 0.7000000 /
# 0.4000000 / 0.1 at the 30 s ... 180 s thresholds, with Muon weight decay 0.2*(1 - progress) =
# 0.1666667 / 0.1333333 / 0.1000000 / 0.0666667 / 0.0333333 / 0.0; `progress` clamps to 1.0 past
# 180.0 s, so the floor lrm 0.1 and weight decay 0.0 are held from there on. The horizon this
# replaces (210.0) reads lrm 1.0 / 1.0 / 1.0 / 0.8714286 / 0.6142857 / 0.3571429 / 0.1 with decay
# 0.1714286 / 0.1428571 / 0.1142857 / 0.0857143 / 0.0571429 / 0.0285714 / 0.0 at the 30 s ... 210 s
# thresholds, i.e. it stands at lrm 0.3571429 and decay 0.0285714 at the 180.0 s reading. Integrated
# lrm-seconds delivered by 180.0 s therefore fall 155.8928571 -> 139.5 (-10.51%): 105.0 s of hold
# plus 75.0 s descending 1.0 -> 0.3571429 (mean 0.6785714, 50.8928571) becomes 90.0 s of hold plus
# 90.0 s descending 1.0 -> 0.1 (mean 0.55, 49.5). All of the reduction is taken out of the second
# half of the envelope. TIME_BUDGET remains the stop rule and the `remaining` log; this only changes
# what the anneal is stretched over.
# Quoted as other launches' measurements, with nothing inferred as fact about this one:
# r010-anneal-270 recorded that a horizon whose update content runs out long before the crossing
# gives the compressed envelope's gain back over the updates past it (matched-update deltas
# -0.025250 at 520 updates, -0.022873 at its own horizon 584, then -0.013981, -0.003932 and
# +0.001128 at 748, a post-horizon descent of 0.0000835 bpb per update); r015-anneal270-span256
# recorded -0.015183 at its 240 s probe, 89% of its horizon in updates, decaying to -0.012758 at its
# own 270.0 s threshold against a byte-identical-span control; and r052-anneal-210-regrid recorded
# 240.291 s on a re-key that did not by itself land the threshold it aimed at. Nothing here predicts
# this run's val_bpb at any probe, its recorded key, WHICH probe threshold it crosses on, its
# intra-tick overshoot, num_steps, total_tokens, per-update time, update rate, update supply or its
# allocator charge, in either direction.
ANNEAL_SECONDS = 180.0

# Probe-synchronised learning-rate trough. The frozen harness accumulates each update's own
# duration and evaluates only when that clock reaches its next threshold, advancing the
# threshold by prepare.PROBE_EVERY_SECONDS = 30.0 each time, so a reading is taken only at
# those instants and the step size that sets a reading is the step size of the updates just
# before it. TROUGH_SECONDS is the width of the ramp-down, measured in harness-clock seconds
# remaining before the next threshold; TROUGH_FLOOR is the fraction of the envelope that
# survives at the threshold itself. 6.0 s is about 15 updates at the parent's recorded 405 ms
# per update. Both are a dose, not a claim about where this run crosses: the trough is keyed
# to the distance to the NEXT threshold, whichever one that is, and it repeats at every one.
# Set to 0.0 in this candidate: get_probe_trough_multiplier's first branch returns 1.0 exactly
# when TROUGH_SECONDS <= 0.0, before any harness attribute is read, so the envelope this
# candidate runs is r005's bit for bit and the function, its call site and TROUGH_FLOOR are left
# in place unchanged. Recorded readings that motivate switching the phase term off, quoted as
# other launches' measurements with no inference asserted as fact here: at the 300 s probe r006
# (trough on) recorded val_bpb 1.053066 and r005 (trough off, same envelope) recorded 1.051509,
# and r006's last three probe intervals fell 0.015918 / 0.014174 / 0.011441 against r005's
# 0.018118 / 0.016031 / 0.012515.
TROUGH_SECONDS = 0.0
TROUGH_FLOOR = 0.15

# ---------------------------------------------------------------------------
# REPLICATE CONTROL, THIRD ATTEMPT, and the whole of this candidate's difference from
# incumbent/train.py. The EXECUTABLE text of this file is the incumbent's exactly and in full:
# strip every comment from this file and from incumbent/train.py and the two texts are identical,
# so no mechanism of the model, the attention implementation, the optimizer, the learning-rate
# schedule, the loss, the data path or the training loop differs by one token, and all four
# declared instruments are bit-identical BY CONSTRUCTION rather than by re-derivation. The only
# difference is this comment block, inserted at module level between TROUGH_FLOOR and the
# `# Model size` block, which touches no frozen-region anchor (`print("Parameter counts:")`,
# `report_efficiency_metrics(`, `TimeToTargetHarness(`, `harness.tick(`,
# `flops_per_token_measured = measure_flops_dispatch(`) and adds, removes, reorders and
# reindents no executable line.
#
# WHY A COMMENT AT ALL, AND WHY AT THIS SITE. A program identity is a hash over train.py,
# prepare.py, pyproject.toml and uv.lock, so byte-identical sources cannot be purchased twice.
# Two earlier replicates of these semantics reserved their bytes without yielding a measurement
# -- round 83's block, inserted before `# Model architecture`, was charged at launch position 78
# and lost to a scheduler preemption before delivery, and round 84's block, inserted before
# `# Optimization`, forfeited on an FA3 kernel fetch before the `Parameter counts:` witness
# printed -- so both of those sites and both of those texts are unavailable. This block is
# therefore new text at a third site, and both earlier insertion points are left untouched.
#
# WHAT THIS LAUNCH IS BOUGHT FOR, stated as the question and not as an expectation. Every level
# delta in this run's record is quoted against a launch-to-launch replicate spread that was
# CALIBRATED and never measured, and this run has since recorded an op-identical diff drawing
# -1.968% of update rate and a strictly work-ADDING diff drawing +1.685%, so the band those
# deltas are read against is demonstrably at least 3.65 pp of throughput wide. This row measures
# the draw of these exact semantics: one program, run twice, nothing else moved.
#
# THE LIVE CONFIGURATION AND ITS INSTRUMENTS, re-derived here because the inherited blocks below
# are written for earlier bases and their prose is stale (nothing executable reads any of it).
# DEPTH 7, ASPECT_RATIO 64 -> model_dim 512, HEAD_DIM 128 -> n_head 4, MLP_HIDDEN_DIM 1,280,
# LOCAL_WINDOW_DIVISOR 32 and LONG_WINDOW_DIVISOR 2 -> per-layer left spans
# (64, 64, 64, 1024, 64, 64, 1024) and span_sum 2,368, TOTAL_BATCH_SIZE 2**16 = 65,536 with
# DEVICE_BATCH_SIZE 64 x TRAIN_SEQ_LEN 1,024 so grad_accum_steps = 1, ANNEAL_SECONDS 180.0,
# TROUGH_SECONDS 0.0. Census: wte 4,194,304 + lm_head 4,194,304 + four value tables 16,777,216 +
# transformer 7*(1,048,576 + 1,310,720 + 128) + 4*128 = 16,516,480 + 14 scalars =
# num_params_total 41,682,318. Counted arithmetic: 6*(41,682,318 - 20,971,534) = 124,264,704 plus
# prepare.py's flash tally 12*4*128*2,368 = 14,548,992, so flops_per_token_measured =
# 138,813,696, equal to GPT.estimate_flops().
#
# Nothing here predicts this launch's val_bpb at any probe, its recorded key, which probe
# threshold it crosses on, its intra-tick overshoot, its gate margin, num_steps, total_tokens,
# its per-update time, its update rate, its update supply or its allocator charge, in either
# direction, and no relation -- equality, ordering, delta, ratio or bound -- is asserted between
# any reading of this row and any recorded reading of any other launch.
# ---------------------------------------------------------------------------

# Model size
DEPTH = 7               # number of transformer layers. Carried from r025/r026, and carried
# TOGETHER with the compile option below because the two only pay off together on this benchmark
# (see that comment block). base_dim = 7*64 = 448 rounds up to model_dim 512, so n_head stays 4 and
# head_dim 128 and only n_layer moves. NOTE the census and counted-arithmetic comments in the
# MLP_HIDDEN_DIM and LOCAL_WINDOW_DIVISOR blocks above are written for EIGHT blocks; at DEPTH 7 they
# read transformer_matrices = 7*(1,048,576 + 1,703,936) + 512 = 19,268,096, num_params_total =
# 4,194,304 + 16,777,216 + 4,194,304 + 19,268,096 + 14 = 44,433,934 (5,898,242 under the 50,332,176
# cap), per-layer left spans (256, 256, 256, 2048, 256, 256, 2048) with span_sum 5,376, and counted
# total = 6*(44,433,934 - 20,971,534) + 12*4*128*5,376 = 140,774,400 + 33,030,144 = 173,804,544
# (65,273,856 under the 239,078,400 ceiling).
# Second half of this card's one mechanism, and the half that keeps accumulation out of it: at
# TRAIN_SEQ_LEN 1024 below, DEVICE_BATCH_SIZE * TRAIN_SEQ_LEN = 128 * 1024 = 131,072 =
# TOTAL_BATCH_SIZE, so grad_accum_steps = 1 and the loader is asked for one microbatch per update.
# prepare.make_dataloader is parameterised in (B, T) by its own signature and allocates a
# row_buffer of (B, T + 1) = (128, 1,025) longs plus a pinned cpu_buffer and a device gpu_buffer of
# 2*B*T = 262,144 longs each (prepare.py:382-384).
# SUPPORTING CONSTANT of the mechanism above, and the half that keeps accumulation out of it. The
# DEPTH comment block above is written for 128 x 1024; on THESE bytes DEVICE_BATCH_SIZE = 64 at
# TRAIN_SEQ_LEN 1024, so DEVICE_BATCH_SIZE * TRAIN_SEQ_LEN = 65,536 = TOTAL_BATCH_SIZE and
# grad_accum_steps = 1. prepare.make_dataloader is parameterised in (B, T) by its own signature
# (prepare.py:330) and allocates row_buffer (B, T + 1) = (64, 1,025) = 65,600 longs plus a pinned
# cpu_buffer and a device gpu_buffer of 2*B*T = 131,072 longs each (prepare.py:382-384), with
# row_capacity = T + 1 = 1,025 unchanged, so one BOS is still laid per 1,024 positions and each
# row's packing geometry is the parent's; only the NUMBER of rows per update changes.
DEVICE_BATCH_SIZE = 64  # per-device batch size (reduce if OOM)
# Training microbatch SEQUENCE geometry, at fixed tokens per optimizer update. TOTAL_BATCH_SIZE
# stays 2**18 = 262,144 and DEVICE_BATCH_SIZE * TRAIN_SEQ_LEN = 64 * 4096 = 262,144, so
# grad_accum_steps stays 1 and prepare.TimeToTargetHarness.tick receives the same token count per
# completed update as the parent. This is the LOADER's packing geometry, not the model's:
# build_model_config still passes sequence_len=MAX_SEQ_LEN = 2048, so GPT._compute_window_sizes
# still reads long_window = 2048 // LONG_WINDOW_DIVISOR = 1024 and short_window =
# 2048 // LOCAL_WINDOW_DIVISOR = 128, the rotary table is still config.sequence_len * 10 = 20,480
# (>= 4096, so the assert T <= self.cos.size(1) in GPT.forward holds), and prepare.evaluate_bpb
# still certifies at its own frozen EVAL_BATCH_SIZE x MAX_SEQ_LEN = 128 x 2048 with its own val
# loader, which train.py never touches.
# WHY THE COUNTED INSTRUMENTS CANNOT MOVE. No parameter is registered and no channel count changes,
# so num_params_total stays 44,433,934. prepare.py's flash-attention tally is span = min(left, S)
# with S = k.shape[1] (prepare.py:550-557); at S = 4096, min(128, 4096) = 128 and
# min(1024, 4096) = 1024, so the per-layer left spans stay (128, 128, 128, 1024, 128, 128, 1024),
# span_sum stays 2,688, and flops_per_token_measured stays
# 6*(44,433,934 - 20,971,534) + 12*4*128*2,688 = 140,774,400 + 16,515,072 = 157,289,472.
# prepare.make_dataloader is parameterised in (B, T) by its own signature and allocates
# 2 * B * T longs for its pinned host buffer and its device buffer, so at B*T = 262,144 both are
# byte-identical to the parent's; _RESOLVED_TRAIN_BUDGET is assigned data_budget_tokens alone
# (prepare.py:349-353), so training_data_tokens_available stays 631,241,817 by construction.
# WHAT ACTUALLY CHANGES. row_capacity = T + 1 goes 2,049 -> 4,097, so best-fit packing crops fewer
# documents and lays one BOS per 4,096 positions instead of one per 2,048, and each position's
# reachable left context doubles in range: averaged over a row, sum_t min(t, window) / T takes the
# long tier from 768 to 896 positions and the short tier from 120 to 124. The token CONTENT of each
# update therefore differs from the parent's; this is the mechanism, not a defect.
# Nothing here predicts this run's val_bpb at any probe, its recorded key, its per-update time, its
# update rate or its allocator charge, in either direction.
# CARRIED SETTLEMENT, value taken from candidates/r062-cross-at-210/train.py line 665 and
# candidates/r063-mlp-width-1280-at-210/train.py: the loader packs rows of 1,024 positions. NOTE the
# inherited block above is written for the parent's 64 x 4096 geometry; on THESE bytes
# DEVICE_BATCH_SIZE is 128 and TRAIN_SEQ_LEN is 1,024, so row_capacity = T + 1 = 1,025, one BOS is
# laid per 1,024 positions, build_model_config still passes sequence_len=MAX_SEQ_LEN = 2048 so
# GPT._compute_window_sizes still reads long_window 1024 and short_window 128, the rotary table is
# still config.sequence_len * 10 = 20,480 (>= 1,024, so the assert T <= self.cos.size(1) in
# GPT.forward holds), and prepare.py's flash-attention tally reads span = min(left, S) at
# S = k.shape[1] = 1,024 (prepare.py:549-557), which leaves the per-layer left spans
# (128, 128, 128, 1024, 128, 128, 1024) and span_sum 2,688 exactly as the parent's.
TRAIN_SEQ_LEN = 1024

# Data budget: the number of DISTINCT training tokens this run may see, taken from the
# head of prepare.py's fixed global shuffle of the whole pool. This single integer is the
# entire data-efficiency surface -- there is no shard list to choose. 631,241,817 is the
# whole pool. Reducing it exposes fewer tokens of the same representative sample, and
# repetition follows automatically as consumption / budget.
DATA_BUDGET_TOKENS = 631_241_817

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

# Axis H only: set a positive target to stop at the first probe below it, or 0.0 for
# probe-only calibration, which records the quality-versus-time curve and never stops.
TARGET_VAL_BPB = float(os.environ.get("AUTORESEARCH_TARGET_VAL_BPB", "0") or 0)
PROBE_ONLY = os.environ.get("AUTORESEARCH_PROBE_CURVE", "") == "1"
MEASURE_ONLY_STEPS = int(os.environ.get("AUTORESEARCH_MEASURE_ONLY_STEPS", "0") or 0)

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
H100_BF16_PEAK_FLOPS = 989.5e12

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

def build_model_config(depth):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )

config = build_model_config(DEPTH)
print(f"Model config: {asdict(config)}")

with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()

param_counts = count_params(model)
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * TRAIN_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

optimizer = model.setup_optimizer(
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    scalar_lr=SCALAR_LR,
    adam_betas=ADAM_BETAS,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY,
)

# Kernel selection for this whole graph, chosen by measured time in process instead of by
# inductor's heuristics: options={"max_autotune": True} enables the Triton GEMM template search and
# benchmark_epilogue_fusion as well as the pointwise/reduction search. It is carried TOGETHER with
# DEPTH 7 because on this benchmark the two only pay off together: r023 measured this key at DEPTH 8
# with peak_vram_bytes 48,738,344,448 B, 1,539,367,936 B OVER the 47,198,976,512 B ceiling, while
# r026 measured it at DEPTH 7 with peak_vram_bytes 39,051,753,984 B -- r025's reading to the byte,
# i.e. 0 B of charge -- at +1.565% of update rate with counted instruments unmoved. Those are r023's,
# r025's and r026's measurements, and nothing here predicts this run's throughput, its allocator
# charge or any reading of any probe, in either direction. This option reaches only the inductor
# backend's kernel configuration: the module, every parameter, every shape and dtype, the optimizer,
# the schedule, the loss and the data path are untouched, and prepare.py's measure_flops_dispatch
# counts model._orig_mod, the uncompiled module, so the counted instruments cannot move.
# SECOND KEY IN THAT SAME OPTIONS DICT, and the whole of this candidate's executable difference
# from candidates/r049-short-tier-128/train.py: "coordinate_descent_tuning": True. torch.compile's
# options= dict sets torch._inductor.config keys directly, and this key is INDEPENDENT of
# max_autotune -- it defaults to False unless TORCHINDUCTOR_COORDINATE_DESCENT_TUNING is set in the
# environment, and this file sets only PYTORCH_ALLOC_CONF and HF_HUB_DISABLE_PROGRESS_BARS -- so on
# the parent's bytes the pass does not run. max_autotune benchmarks the candidate configs
# inductor's heuristics enumerate for each generated kernel and takes the best of that fixed list;
# this pass starts from that winner and walks one coordinate at a time (block size along the
# pointwise and reduction axes, num_warps, num_stages), keeping a step only when the measured time
# improves. WHERE IT CAN REACH: the graph's widest tensors are not its matmuls. The output head's
# activation is [128, 2048, 8192] = 2,147,483,648 elements, and the cap chain, the vocabulary-axis
# reduction and both of their backwards are generated pointwise/reduction kernels whose FORMULA
# (r046) and STORE WIDTH (r047) this run has already measured and whose launch geometry it never
# has. WHAT IT CANNOT REACH: the module, every parameter, every shape and dtype, the optimizer, the
# schedule, the loss and the data path are untouched, and prepare.py's measure_flops_dispatch
# counts model._orig_mod, the uncompiled module, so num_params_total stays 44,433,934 and
# flops_per_token_measured stays 6*(44,433,934 - 20,971,534) + 12*4*128*2,688 = 140,774,400 +
# 16,515,072 = 157,289,472, both by construction. A different launch geometry may reduce along the
# 8,192-wide vocabulary axis in a different order, so bitwise-identical logits are NOT claimed; the
# loss's fp32 accumulation and prepare.py's evaluate_bpb are unchanged. Quoted as other launches'
# measurements with nothing inferred as fact about this one: r024 took the pointwise/reduction
# search alone and was accepted, r026 measured the GEMM template search at DEPTH 7 at +1.565% of
# update rate with the counted instruments unmoved and peak_vram_bytes 39,051,753,984 B, and r023
# measured the same search at DEPTH 8 at 48,738,344,448 B, 1,539,367,936 B OVER the ceiling. This
# pass re-benchmarks kernels whose buffers max_autotune's existing search already allocates, which
# is a census of what it touches and not a bound on this row's allocator charge. The search runs at
# trace time, inside step 0, which the harness's ten-update warmup rule leaves outside its clock.
# Nothing here predicts this run's per-update time, its update rate, its allocator charge, its
# quality at any probe or its recorded key, in either direction.
model = torch.compile(model, dynamic=False, options={"max_autotune": True, "coordinate_descent_tuning": True})

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, TRAIN_SEQ_LEN, "train",
                              data_budget_tokens=DATA_BUDGET_TOKENS)
print(f"Data budget:    {DATA_BUDGET_TOKENS:,} distinct tokens from the shuffled pool")
harness = (TimeToTargetHarness(TARGET_VAL_BPB)
           if (TARGET_VAL_BPB > 0 or PROBE_ONLY) else None)
x, y, epoch = next(train_loader)  # prefetch first batch
# Dispatch-level FLOPs, measured HERE while memory still holds only the model and the
# optimizer state. Counted on the uncompiled module by the frozen probe.
from prepare import measure_flops_dispatch
flops_per_token_measured = measure_flops_dispatch(model, x, y)
print(f"FLOPs per token (measured at dispatch): {flops_per_token_measured:,}")

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")

# Schedules (all based on progress = training_time / TIME_BUDGET)

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)

def get_probe_trough_multiplier(harness):
    """Scale the LR envelope down into the next probe threshold and restore it after.

    Read-only use of the frozen harness: `next_probe_at` is the threshold it will test next
    and `training_seconds` is the clock it tests against. Nothing is assigned to the harness,
    no probe is added, moved, shortened or skipped, its cadence, warmup count and full-eval
    setting are untouched, and tick still receives exactly one call per completed update with
    that update's own synchronized duration and its own token count. Returns 1.0 when there
    is no harness (the probe-only / no-target path), which makes the schedule identical to
    the parent's there.
    """
    if harness is None or TROUGH_SECONDS <= 0.0:
        return 1.0
    seconds_to_probe = max(0.0, harness.next_probe_at - harness.training_seconds)
    ramp = min(1.0, seconds_to_probe / TROUGH_SECONDS)
    return TROUGH_FLOOR + (1.0 - TROUGH_FLOOR) * ramp

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
tokens_consumed = 0
step = 0

while True:
    torch.cuda.synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, epoch = next(train_loader)

    # Progress and schedules
    progress = min(total_training_time / ANNEAL_SECONDS, 1.0)
    lrm = get_lr_multiplier(progress) * get_probe_trough_multiplier(harness)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    optimizer.step()
    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()

    # Fast fail: abort if loss is exploding or NaN
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    # Read-only allocator high-water reads, twice, to attribute peak_vram_bytes between step 0's
    # transient compile/benchmarking buffers and the steady-state saved-activation set, which is
    # the attribution r023's row could not make. This is a pure read: max_memory_allocated()
    # resets nothing, torch.cuda.reset_peak_memory_stats() is NOT called anywhere in this file,
    # prepare.py's own read at report time is untouched, both lines print before the reporter so
    # the last METRICS_JSON line in stdout is still prepare.py's, and the reads sit outside the
    # timed window (after t1, before the next update's t0) so no duration passed to harness.tick
    # is affected. `step` here is the index of the update just completed.
    if step in (0, 20):
        print(f"\n[vram] step {step}: max_memory_allocated={torch.cuda.max_memory_allocated()} B", flush=True)

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / H100_BF16_PEAK_FLOPS
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

    # GC management (Python's GC causes ~500ms stalls)
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1
    tokens_consumed += TOTAL_BATCH_SIZE

    # Axis H: the frozen harness owns the clock, the probe cadence and the crossing.
    if harness is not None and harness.tick(model, tokenizer, dt, TOTAL_BATCH_SIZE):
        break

    if MEASURE_ONLY_STEPS and step >= MEASURE_ONLY_STEPS:
        break

    # Time's up — but only stop after warmup steps so we don't count compilation
    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r training log

total_tokens = tokens_consumed
t_end = time.time()

# Every score comes from the FROZEN prepare.py. train.py may change the model; it may
# not compute the number the model is judged on.
report_efficiency_metrics(
    model, tokenizer,
    num_steps=step,
    tokens_per_step=TOTAL_BATCH_SIZE,
    total_tokens=tokens_consumed,
    training_seconds=total_training_time,
    total_seconds=t_end - t_start,
    final_epoch=epoch,
    flops_measured=flops_per_token_measured,
    harness=harness,
)
print(f"depth:            {DEPTH}")
