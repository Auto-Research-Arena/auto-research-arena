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

from prepare import (MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader,
                     evaluate_bpb, report_efficiency_metrics, TimeToTargetHarness, count_params)

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


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


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
        # QK-norm sets both q and k to unit RMS per head, so the pre-scale dot product is
        # head_dim * cos(angle) with head_dim = 128. Scaling q here (not k) by
        # ATTN_LOGIT_SCALE is exactly equivalent to passing softmax_scale=0.15 to the kernel,
        # because the kernel's scaling is linear in q. Done by multiplication rather than by a
        # softmax_scale kwarg deliberately: fa3 is fetched by get_kernel at import time and
        # cannot be introspected on a login node, so a kwarg the build does not accept would
        # raise TypeError and forfeit the launch, whereas a multiply cannot fail.
        q, k = norm(q) * ATTN_LOGIT_SCALE, norm(k)

        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        # Depth-heterogeneous MLP width: the expansion is selected per shared block rather
        # than fixed for the whole model. Under SHARE_PERIOD 4 layer_idx is the block index
        # 0..3, and under WINDOW_PATTERN "SSLL" blocks 2 and 3 are the full-context ones,
        # so those two get the narrower MLP -- their share of the block's work is already
        # the more attention-dominated one.
        # The MLP hidden width is stated in CHANNELS rather than as an integer multiple of
        # n_embd. Nothing about the mechanism required the multiple to be an integer; the file
        # wrote it that way, and that made the smallest reachable step a full 1*d^2 per block.
        # Verified working at round 20, which recorded its predicted parameter count and its
        # predicted arithmetic exactly, including a fractional effective expansion.
        hidden = MLP_HIDDEN_BY_BLOCK[layer_idx % len(MLP_HIDDEN_BY_BLOCK)]
        self.c_fc = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config, layer_idx)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        # Cross-layer weight sharing with period SHARE_PERIOD: layer i is computed by
        # unique_blocks[i % SHARE_PERIOD], so n_layer layers of computation are carried by
        # SHARE_PERIOD distinct blocks. Depth of computation, and so FLOPs per token, is
        # unchanged; only the number of distinct parameter tensors falls. The period is 4
        # deliberately: WINDOW_PATTERN is length 4, so a shared block always sees the same
        # window; and 4 is even, so has_ve parity is preserved and the blocks built at
        # layer_idx 1 and 3 -- the ones holding a ve_gate -- always serve the VE layers.
        # Each of the n_layer positions keeps its own resid_lambdas and x0_lambdas below,
        # so the layers still differ in how they mix the residual and the x0 injection.
        assert config.n_layer % SHARE_PERIOD == 0
        unique_blocks = [Block(config, i) for i in range(SHARE_PERIOD)]
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([unique_blocks[i % SHARE_PERIOD] for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Value embeddings
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        # ONE shared value-embedding table for every VE layer. The layer-specific part of
        # the value-residual pathway is the per-layer, per-head input-dependent ve_gate,
        # which is untouched; four separate token->value maps were three redundant copies.
        self.value_embeds = nn.ModuleDict({
            "shared": nn.Embedding(config.vocab_size, kv_dim)
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

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
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
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
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
            ve = self.value_embeds["shared"](idx) if has_ve(i, self.config.n_layer) else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

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
ASPECT_RATIO = 48       # model_dim = depth * ASPECT_RATIO -> 384, num_heads 3 at HEAD_DIM 128:
                        # narrower residual trades capacity for optimizer updates in the fixed
                        # wall clock, since 12*L*d^2 is what the flops charge is levied on
HEAD_DIM = 128          # target head dimension for attention
ATTN_LOGIT_SCALE = 0.15 * HEAD_DIM**0.5   # = 1.6970562748477141. Multiplier applied to q
                        # AFTER QK-norm, which makes the effective attention logit scale
                        # 0.15 instead of the kernel's default head_dim**-0.5 = 0.08839.
                        # SECOND RUNG of the dial round 27 opened. Round 27 moved the target
                        # scale from the kernel default 0.08839 to 0.12, a factor of 1.358 in
                        # the logits; this moves it 0.12 -> 0.15, a factor of 1.250, so the
                        # two steps are comparable in log scale and the pair brackets the
                        # optimum if the response turns. 0.15 is a departure from the tuned
                        # speedrun constant 0.12 and that is the point: the substrate's value
                        # was inherited, not verified at this architecture, and round 27
                        # measured the first step in this direction as an improvement.
                        # Written as target_scale * sqrt(HEAD_DIM) so that the target scale
                        # 0.15 is the literal in the source and the multiplier is derived:
                        # the kernel multiplies q.k by head_dim**-0.5 internally, so scaling q
                        # by 0.15*sqrt(head_dim) yields exactly 0.15*q.k regardless of what
                        # the kernel's default is. self.head_dim == HEAD_DIM holds here because
                        # model_dim is built as num_heads * HEAD_DIM, so 384 // 3 == 128.
WINDOW_PATTERN = "SSSS" # sliding window pattern: L=full, S=half context. REVERSED from
                        # "SSLL": every layer the pattern controls now sees 1024 instead of
                        # 2048. _compute_window_sizes still forces window_sizes[-1] to the
                        # long window, so the effective contexts are seven layers at 1024 and
                        # the last at 2048, i.e. Sigma_S_l falls 12,288 -> 9,216. Length stays
                        # 4 so the SHARE_PERIOD blocks each keep a single fixed window.
                        # This SPENDS NO PARAMETERS AND BUYS WALL CLOCK: attention arithmetic
                        # falls by 14,155,776 FLOPs per token, 9.64 per cent of the total, and
                        # the 600 s TIME_BUDGET converts that directly into more steps and
                        # more tokens. Round 6 measured this dial in the OTHER direction and
                        # found context worth 0.074 prereg sd, i.e. nothing this instrument
                        # can see; round 30 measured tokens and found them worth about 0.40 sd
                        # per 29 per cent. Those two readings together are the whole argument.

# Optimization
TOTAL_BATCH_SIZE = 2**17 # ~131K tokens per optimizer step: twice the updates in the same
                         # 600 s at the same total tokens, still one micro-batch per update
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.02        # learning rate for matrix parameters (Muon). Halved: this is the
                        # only LR in the file that is NOT multiplied by dmodel_lr_scale, so it
                        # never followed the model from d=512 to d=384; and Muon's update
                        # magnitude is step-count independent, so total path length has grown
                        # with steps 753 -> 4161 at unchanged lr.
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# Model size
DEPTH = 8               # number of transformer layers
MLP_HIDDEN_BY_BLOCK = (1240, 1240, 944, 944)
                        # ROUND 75 -- THE FAR END OF THE SWEEP, WHICH IS WHERE THE TWO READINGS DIFFER.
                        # The wide pair FALLS 1264 -> 1240 while the narrow pair RISES 928 -> 944, so
                        # sum = 1240*2 + 944*2 = 2480 + 1888 = 4368, down 16 from 4384 -- one minimum step,
                        # since on a mirrored (a, a, b, b) profile under 8-alignment the sum moves only in
                        # multiples of 16. COMPOSITE and declared as one.
                        # num_params_total = 11,796,688 + 768*4368 = 15,151,312: -12,288 parameters,
                        # -0.081036 per cent, a cut of 69.897363 per cent against 50,332,176.
                        # flops_per_token_measured = 89,655,552 + 9216*4368 = 129,911,040.
                        # num_params_active = 15,151,312 - 6,064,912 = 9,086,400, from the ledger.
                        #
                        # THE QUESTION. Round 74 left two rival readings of the shape sweep at this count.
                        # READING A: the three drawn shapes -- 1264/920 at 1.0507476, 1280/904 at 1.0501509,
                        # 1288/896 at 1.0502591 -- trace a parabola whose minimum sits at wide 1281, a grid
                        # point already drawn, so the sweep is finished. READING B: the between-shape spread
                        # at this count is 0.000318 bpb while three IDENTICAL repeats of one profile span
                        # 0.00078485 bpb, so the shape effect is 2.5x SMALLER than the instrument's own
                        # repeatability and the three points are one point with noise. The readings disagree
                        # most at the FAR END of the sweep, away from the claimed vertex, and that is the
                        # cheapest place to separate them.
                        #
                        # WHY 1240 IS THE ONLY SHAPE LEFT THAT CAN ASK IT. Every other untried shape at this
                        # count carries an adverse measurement: 1248 is 0.217 sd worse (at 15,175,888) and
                        # 0.370 sd worse (at 15,139,024) than 1264; 1256 is 0.477 sd worse, which is 2.8
                        # repeat-sd and therefore resolvable; 1272 is 0.110 and 0.263 sd worse; 1296 would
                        # put the narrow pair at 888, below the round-72 knee -- the ONE shape difference on
                        # this arm that is resolvable, at 3.5 repeat-sd. Wide 1240 has never been used at
                        # any count on this arm.
                        #
                        # AND THE NARROW SIDE IS THE CHEAPEST WIDTH ON THE LADDER. Narrow 944 read 0.06893
                        # prereg sd at wide 1264, the shallowest rung of the seven-point narrow ladder, and
                        # it sits SEVEN minimum steps above the knee. Whatever this row measures, it will
                        # not be a capacity cliff.
                        #
                        # WHY THIS IS NOT A LOTTERY TICKET. Round 74 refused to spend the remaining rows on
                        # 31-per-cent draws at a count whose mean fails by half a repeat-sd, on the ground
                        # that a frontier assembled from the best of repeated draws at such a count is not a
                        # result -- the letter of the no-re-run rule does not reach a fresh shape, but its
                        # spirit does. This row is justified by the QUESTION above and not by its P(clear).
                        # If it lands near the other three, the arm has measured that shape at fixed count
                        # is unresolvable on this instrument, and that finding closes the search.
                        #
                        # WHERE THE ARM IS, IN THE INSTRUMENT'S OWN UNITS. The frontier clears its gate by
                        # +0.208 repeat-sd; the best row one step below it misses by -0.192. An identical
                        # re-draw of the frontier would FAIL about 42 per cent of the time. The arm is not
                        # approaching the gate, it is sitting on it, and has been since round 64. The
                        # pre-registered 0.0045534 remains the only error bar used to REPORT a margin; the
                        # measured repeat-sd is used only to decide what is worth a launch.
                        #
                        # MULTIPLICITY DECLARED. FOURTH attempt at the count 15,151,312, after cand-0074
                        # (1264, 0.190834 sd), cand-0081 (1280, 0.059782) and cand-0082 (1288, 0.083541).
                        #
                        # NO PREDICTION FOR THE OTHER METRICS. peak_vram_bytes: none -- four distinct values
                        # at one identical count settled that in round 70. decode_ms_nocache_4096: none.
                        # Throughput: no direction claimed; round 72 measured the fastest row of the
                        # neighbourhood as the worst of the arm.
                        #
                        # CONSTRAINTS. Exactly two distinct widths (1240 and 944), as the round-43 recompile
                        # limit requires. Both 8-aligned: 1240 = 8*155 and 944 = 8*118; round 55 refuted
                        # 4-alignment at 2.85 per cent of throughput. Tiles 10 + 8 = 18, the same as the
                        # frontier, with no boundary crossed on either pair. The profile
                        # (1240, 1240, 944, 944) has never been launched here.
                        #
                        # THE COMMENT FROM cand-0072 FOLLOWS. Both of its widths are superseded here and its
                        # "price of this dial as measured" paragraph is superseded by the round-64 through
                        # round-74 findings. It is left unedited, because it is the record of what was
                        # believed when that row was spent -- and that row is still the frontier.
                        # ROUND 64 -- THE ONLY UNTRIED POINT ON THE CHEAPEST MEASURED
                        # DIRECTION. The wide pair is HELD at the parent's 1264 and the narrow pair
                        # is cut by 24 each, so sum = 1264*2 + 928*2 = 2528 + 1856 = 4384, down 48
                        # from 4432 -- three minimum steps, since on a mirrored (a, a, b, b) profile
                        # under 8-alignment the sum moves only in multiples of 16.
                        # num_params_total = 11,796,688 + 768*4384 = 15,163,600, a strict improvement
                        # of the ranking key by 36,864 parameters, 0.242453 per cent, and 24,576
                        # below the best previous attempt's 15,188,176.
                        # flops_per_token_measured = 89,655,552 + 9216*4384 = 130,058,496.
                        # num_params_active = 15,163,600 - 6,064,912 = 9,098,688, since across this
                        # neighbourhood the ledger gives d(active)/d(total) = 1.0000 exactly.
                        #
                        # WHY THIS DIRECTION, AND NOT THE ONE THE LAST THREE ROWS USED. Round 63
                        # recorded that I had been selecting on profile skew -- an axis my own card
                        # priced at 0.0143 prereg sd, against this lineage's residual scatter of
                        # 0.0669 sd. A criterion 4.8x smaller than the noise it is used against
                        # cannot order rows, and it did not: of the three candidates at
                        # num_params_total 15,188,176 the BEST-placed on skew came second. So this
                        # ticket is placed on the only ordering in the five attempts at a scale
                        # comparable to the scatter:
                        #
                        #   cand-0067  wide 1264, narrow 944   +0.0689 sd   wide HELD
                        #   cand-0068  wide 1264, narrow 936   +0.0915 sd   wide HELD
                        #   cand-0071  wide 1272, narrow 936   +0.1786 sd   wide moved
                        #   cand-0070  wide 1248, narrow 952   +0.3085 sd   wide moved
                        #   cand-0069  wide 1256, narrow 952   +0.5456 sd   wide moved
                        #
                        # The two cheapest hold the wide pair and pay out of the narrow pair; all
                        # three that move the wide width cost more. That is two rows against three
                        # and it is NOT resolvable against a 0.0669 sd scatter -- it is the
                        # hypothesis this row is placed on, not a fact. It is at least consistent
                        # with round 62's finding that cand-0064 is a -0.2009 sd favourable draw on
                        # its own nine-rung ladder: if part of that residual belongs to wide = 1264
                        # rather than to that one launch, rows keeping 1264 inherit part of it and
                        # rows moving away forfeit it. This row keeps 1264 exactly.
                        #
                        # THE PRICE OF THIS DIAL, AS MEASURED. 12,288 parameters (narrow 952 -> 944)
                        # cost 0.068930 sd at cand-0067; 24,576 (952 -> 936) cost 0.091500 sd at
                        # cand-0068 -- the price grew 1.33x while the parameters doubled, sub-linear
                        # over the only interval measured. Whether it stays sub-linear at three
                        # steps is what this launch reports, and the card asserts neither answer.
                        # 36,864 on this dial has NEVER been measured: narrow 928 has never been
                        # launched at any wide value on this lineage, which holds 936, 944 and 952
                        # at wide 1264. Nor has num_params_total 15,163,600 been reached by any
                        # candidate on this arm -- unlike rounds 61-63 this row carries no
                        # multiplicity at all, it is the first attempt at its parameter count.
                        #
                        # WHY NOT A FOURTH SINGLE STEP. A row at narrow 940 would spend a charged
                        # launch interpolating between two rows whose separation, 0.023 sd, is a
                        # third of the scatter. The large step is the informative one.
                        #
                        # SKEW, FOR THE RECORD AND NOT AS THE REASON. 1264/928 = 1.362069, a
                        # displacement of 0.0092 from the confirmed 1.3529 optimum. Recorded because
                        # round 63 requires the number to be reported; the axis is worth 0.0143 sd
                        # and is a tie-breaker of last resort.
                        #
                        # CONSTRAINTS. Exactly two distinct widths (1264 and 928), as the round-43
                        # recompile-limit crash requires. Both 8-aligned: 1264 = 8*158 and
                        # 928 = 8*116; round 55 refuted 4-alignment at 2.85 per cent of throughput.
                        # Still TILE-MINIMAL: ceil(1264/128) = 10 and ceil(928/128) = 8, total
                        # 18 = ceil(2192/128), so no output tile is lost or gained.
                        #
                        # INHERITED COMMENT FROM cand-0064 FOLLOWS. It describes widths
                        # (1264, 1264, 952, 952): its wide value still holds here and its narrow
                        # value is superseded. Its closing paragraphs quote the 0.07-0.09 token
                        # price, the 0.4327 parameter price and the pod-offset margin argument, all
                        # of which the round-58 price audit and the round-62 ladder finding
                        # supersede; they are left unedited because they are the record of what was
                        # believed when that row was spent.
                        # THROUGHPUT-FUNDED DESCENT: cut the wide
                        # pair BACK ACROSS A MATMUL TILE BOUNDARY, so the same edit buys
                        # parameters and training tokens at once.
                        #
                        # Sum 1264*2 + 952*2 = 2528 + 1904 = 4432, down 48 from 4480.
                        # num_params_total = 11,796,688 + 768*4432 = 15,200,464, down 36,864
                        # parameters or 0.24193 per cent of the model.
                        # flops_per_token_measured = 89,655,552 + 9216*4432 = 130,500,864.
                        #
                        # THE MECHANISM IS A TILE COUNT, not arithmetic. On this run's own ladder
                        # at fixed narrow width 952, train_tokens_per_second reads 995,615 at wide
                        # 1272 and 995,495 at wide 1280, then drops to 984,064 at 1288 and stays
                        # in 984,263-989,899 for every wider value up to 1344. That is a ~1.16 per
                        # cent CLIFF between 1280 and 1288, far too large and too abrupt for the
                        # 0.11 per cent of arithmetic that separates them. With a 128-wide output
                        # tile, hidden 1280 is exactly 10 tiles and 1288 needs an eleventh that is
                        # 94 per cent empty. 1264 is 9.875 tiles, so it rounds to 10 and sits on
                        # the fast side of the same boundary.
                        #
                        # WHY THIS IS THE ONLY MOVE LEFT. Round 55 measured that this pod runs
                        # 1.586 per cent slower than the one every incumbent reading came from, and
                        # that tokens price at 0.07-0.09 prereg sd per per cent, so the node alone
                        # took 0.111-0.143 sd of gate margin from an incumbent that had 0.052-0.058.
                        # Margin has to be bought back before any dose can be afforded, the free
                        # way to buy it is a params-neutral realignment, and that is objective-
                        # neutral and refused because the retune lane is at 4 of 4. So the purchase
                        # has to travel WITH a cut, which is exactly what crossing the boundary
                        # downward does.
                        #
                        # 15,200,464 has NEVER been launched, and wide width 1264 has never been
                        # launched. Narrow 952 is HELD at the incumbent's value, which is what the
                        # predecessor's own nine-rung ladder cand-0049..cand-0057 did, so this is a
                        # single-dial move on the dial actually in use. The two rejected doses that
                        # are forbidden to me -- sum 4464 (cand-0050) and sum 4448 (cand-0049) --
                        # are both stepped OVER rather than landed on, and no lateral variant at
                        # either is used.
                        #
                        # SKEW. 1264/952 = 1.327731 against the parent's 1.352941, a fall of 0.0252
                        # that lands below the interior optimum round 48 placed at 1.3529. At the
                        # measured post-peak slope of 0.724 sd per unit that is a cost of 0.0182
                        # prereg sd, charged rather than ignored.
                        #
                        # STILL EXACTLY TWO DISTINCT WIDTHS (1264 and 952), as the round-43
                        # recompile-limit crash requires. Both are 8-aligned: 1264 = 8*158 and
                        # 952 = 8*119. Round 55 refuted 4-alignment at a cost of 2.85 per cent of
                        # throughput, so 8-alignment is held and will not be tested again.
                        # ORIGINAL COMMENT FROM cand-0047 FOLLOWS:
                        # THIRD parameter-neutral FLATTENING rung on
                        # the profile MAGNITUDE axis: 64 channels move off each wide block and
                        # onto each narrow block. Sum 1288*2 + 952*2 = 2576 + 1904 = 4480,
                        # identical to 1352*2 + 888*2 = 4480, so num_params_total stays
                        # 15,237,328 and flops_per_token stays 130,943,232.
                        # Rungs 1 and 2 of this exact operation gained 0.1326 and 0.0582 prereg
                        # sd; skew goes 1.5225 -> 1.3529, a step of 0.170 against their 0.235 and
                        # 0.200. Round 42 bracketed the optimum below skew 1.541 and far above
                        # 0.649, so 1.3529 should not overshoot. The 2:2 split and the wide-early
                        # order are BOTH held: round 42 closed the order axis at 0.3239 sd and
                        # round 46 closed the split-ratio axis at 0.3985 sd, so magnitude is the
                        # only degree of freedom of this profile still open. Exactly TWO distinct
                        # widths, as the round-43 recompile-limit crash requires. 1288 = 8*161 and
                        # 952 = 8*119, both 8-aligned; neither has ever been launched by this run.
                        #
                        # Sized small on purpose. The margin at the parent is 0.0932 prereg sd
                        # and the fitted price of a width move carries a FIXED charge of 0.047
                        # sd before a single channel comes off, so half the available margin is
                        # spent just by touching the dial. Expected-cut is nearly flat between
                        # 16 and 24 channels while the landing probability is much better at
                        # 16, and a structural ACCEPT refills the retune counter -- which is
                        # down to its last attempt -- while a breach leaves the arm able to do
                        # nothing but descend at an unchanged margin. That asymmetry decided
                        # the size at round 37 and it decides it again here.
                        #
                        # Still the WIDE pair: round 42 measured that wide-early is the correct
                        # direction (reversing it cost 0.3239 sd) and round 41 priced a wide cut
                        # 41 per cent cheaper than a narrow one. Cutting narrow would push the
                        # skew back UP, against a lever known to pay.
                        #
                        # 1352 = 8*169, so both wide matmuls stay 8-aligned, and the model still
                        # holds exactly TWO distinct MLP widths -- required, per the round-43
                        # crash, by the Muon-per-shape compile limit.
                        #
                        # PREVIOUS profile, kept for provenance:
                        # first parameter-reducing rung of this run that does NOT cut the
                        # narrow pair. 40 channels come off each wide block for -61,440
                        # parameters = -0.4009 per cent, taking num_params_total to
                        # 15,261,904 and flops_per_token to 131,238,144.
                        #
                        # Why the wide pair and not the narrow one: rounds 39 and 40 measured
                        # that a FLATTER profile is better, twice, for 0.1326 then 0.0582
                        # prereg sd. Cutting the narrow pair would push the skew back UP and
                        # spend against a lever that is known to pay. Cutting the WIDE pair
                        # descends the ranking key AND continues flattening: skew 1.586 ->
                        # 1.541. This is the first rung in the run whose parameter saving and
                        # whose quality lever point the SAME way.
                        #
                        # 1368 = 8*171, so both wide matmuls stay 8-aligned. Sum falls
                        # 4592 -> 4512; both metrics follow the exact calibrated relations
                        # params = 11,796,688 + 768*sum and flops = 89,655,552 + 9216*sum.
                        #
                        # PREVIOUS profile, kept for provenance:
                        # width-PROFILE dial opened at round 39, which gained 0.1326 prereg
                        # sd for zero parameters. Another 64 channels move out of each wide
                        # block and into each narrow block, continuing in the one direction
                        # this dial is known to reward. Skew falls 1.786 -> 1.586.
                        #
                        # The sum is the invariant that makes this free: 1408 + 1408 + 888 +
                        # 888 = 4592, identical to the parent and to its parent. MLP params
                        # are 2 * 384 * 4592 = 3,526,656 unchanged, so num_params_total holds
                        # at 15,323,344, and the FLOPs MLP term 12 * sum * d^2 holds too.
                        # 1408 = 8*176 and 888 = 8*111, so all four matmuls stay 8-aligned.
                        #
                        # PREVIOUS rung, kept for provenance:
                        # width PROFILE, not a rung down the channel dial. 64 channels move
                        # OUT of each of the two wide blocks and INTO each of the two narrow
                        # blocks. MLP parameters are 2 * n_embd * hidden per distinct block
                        # with no bias, so the MLP parameter count depends only on the SUM of
                        # this tuple: 2 * 384 * 4592 = 3,526,656 before and after. FLOPs per
                        # token likewise depend only on the sum. Both are byte-identical to
                        # the parent by construction, not by luck.
                        #
                        # Why this dial: the wide blocks have never been moved once in this
                        # run. Every one of the fourteen rungs spent so far cut the NARROW
                        # pair alone, so the wide:narrow ratio drifted from 1.00 to 2.02
                        # without a single measurement ever asking whether a skew that large
                        # is the right shape. A dial that a one-sided search pushed fourteen
                        # times is the dial most likely to sit past its own optimum.
                        #
                        # All four blocks carry equal compute: layer i uses block i %
                        # SHARE_PERIOD with DEPTH 8 and SHARE_PERIOD 4, so each distinct block
                        # runs exactly twice. "Narrow" is a label this arm imposed, not a
                        # structural role the substrate assigns.
                        #
                        # superseded comment, kept for provenance:
                        # blocks 2 and 3 -- the pair this dial has always moved -- go
                        # (round 37: 800 -> 760 hidden channels, for -61,440 parameters)
                        # = -0.3994 per cent. 760 = 8*95, so both matmul
                        # shapes stay 8-aligned. Blocks 0 and 1 stay at 1536 = 4*384.
SHARE_PERIOD = 4        # cross-layer weight sharing period: layer i uses block i % this,
                        # so DEPTH layers of compute are held in SHARE_PERIOD blocks
DEVICE_BATCH_SIZE = 64   # per-device batch size (reduce if OOM); 64 * 2048 = 2**17 keeps
                         # grad_accum_steps at exactly 1, which the assert below requires

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

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
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

model = torch.compile(model, dynamic=False)

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train",
                              data_budget_tokens=DATA_BUDGET_TOKENS)
print(f"Data budget:    {DATA_BUDGET_TOKENS:,} distinct tokens from the shuffled pool")
_time_cap = float(os.environ.get("AUTORESEARCH_TIME_CAP", "0") or 0)
harness = (TimeToTargetHarness(TARGET_VAL_BPB,
                               **({"cap": _time_cap} if _time_cap > 0 else {}))
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

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
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
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
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

    # Axis H: the frozen harness owns the clock, the probe cadence and the stop rule.
    if harness is not None and harness.tick(model, tokenizer, dt):
        break

    if MEASURE_ONLY_STEPS and step >= MEASURE_ONLY_STEPS:
        break

    # Time's up — but only stop after warmup steps so we don't count compilation
    if harness is None and step > 10 and total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r training log

total_tokens = step * TOTAL_BATCH_SIZE
t_end = time.time()

# Every score comes from the FROZEN prepare.py. train.py may change the model; it may
# not compute the number the model is judged on.
report_efficiency_metrics(
    model, tokenizer,
    num_steps=step,
    tokens_per_step=TOTAL_BATCH_SIZE,
    training_seconds=total_training_time,
    total_seconds=t_end - t_start,
    final_epoch=epoch,
    total_tokens=total_tokens,
    flops_measured=flops_per_token_measured,
    harness=harness,
)
print(f"depth:            {DEPTH}")
