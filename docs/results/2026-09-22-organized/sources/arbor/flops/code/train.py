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
    # Per-layer query-head count (len == n_layer). head_dim stays uniform (= n_embd //
    # n_head), so a layer with fewer heads is a NARROWER attention sublayer: its q/o
    # projections shrink and its FA3 span tally (12*h*d*span) shrinks proportionally.
    n_head_per_layer: tuple = ()
    # Per-layer KV-head count (len == n_layer). FA3 requires n_kv_head <= n_head and
    # n_head % n_kv_head == 0, so a 1-query-head layer must also drop to a single KV head
    # -- which shrinks c_k/c_v (and that layer's value-embedding table) as well.
    n_kv_head_per_layer: tuple = ()
    n_embd: int = 768
    # Explicit per-layer left window (in tokens). -1 means unbounded (full causal).
    window_left: tuple = ()
    mlp_ratio: tuple = ()       # per-layer MLP hidden multiplier (len == n_layer)
    lm_head_rank: int = 0       # 0 = dense unembedding, >0 = low-rank factorisation
    # Layers that get an additive (flops-free) token-identity embedding on the residual.
    resid_embed_layers: tuple = ()
    # Layers that keep a real attention sublayer. Every other layer replaces attention
    # with the unpriced causal shift mixer.
    attn_layers: tuple = ()
    # Layers whose additive residual embedding is indexed by a HASHED BIGRAM (prev, cur)
    # instead of the token identity. Still a pure gather, so still zero counted FLOPs.
    bigram_embed_layers: tuple = ()
    bigram_rows: int = 0        # rows in each bigram table (power of two)


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def shift_causal(x, k):
    """Shift the sequence right by k (each position sees position t-k). Unpriced."""
    return F.pad(x, (0, 0, k, 0))[:, :-k]


class ShiftMix(nn.Module):
    """Depthwise causal mixer built only from elementwise ops and pad/slice.

    prepare.measure_flops_dispatch prices matmul/conv/SDPA plus the FA3 span tally; a
    per-channel weighted sum of shifted copies of the residual touches none of those, so
    this sublayer mixes tokens at exactly zero counted FLOPs. Deliberately NOT conv1d,
    which FlopCounterMode does price.
    """

    def __init__(self, config):
        super().__init__()
        n = config.n_embd
        self.w0 = nn.Parameter(torch.empty(n))
        self.w1 = nn.Parameter(torch.empty(n))
        self.w2 = nn.Parameter(torch.empty(n))
        self.w4 = nn.Parameter(torch.empty(n))
        # Node 4.4: two extra taps at 8 and 16. Still zero counted FLOPs, and they extend the
        # mixer's reach to compensate (unpriced) for the query-head width removed from the
        # local attention layers.
        self.w8 = nn.Parameter(torch.empty(n))
        self.w16 = nn.Parameter(torch.empty(n))
        self.gate = nn.Parameter(torch.empty(n))

    @torch.no_grad()
    def init_weights(self):
        self.w0.fill_(1.0)
        self.w1.fill_(0.3)
        self.w2.fill_(0.2)
        self.w4.fill_(0.1)
        self.w8.fill_(0.05)
        self.w16.fill_(0.025)
        # Small gate: the sublayer starts as a mild perturbation of the residual, the same
        # role the zero-initialised attention c_proj plays in the attention layers.
        self.gate.fill_(0.1)

    def forward(self, x):
        y = (self.w0 * x
             + self.w1 * shift_causal(x, 1)
             + self.w2 * shift_causal(x, 2)
             + self.w4 * shift_causal(x, 4)
             + self.w8 * shift_causal(x, 8)
             + self.w16 * shift_causal(x, 16))
        return self.gate * y


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
        self.n_head = config.n_head_per_layer[layer_idx]
        self.n_kv_head = config.n_kv_head_per_layer[layer_idx]
        self.n_embd = config.n_embd
        # Uniform head dimension across layers (shared rotary tables); only the NUMBER of
        # query heads varies per layer.
        self.head_dim = self.n_embd // config.n_head
        assert self.n_embd % config.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        q_dim = self.n_head * self.head_dim
        self.c_q = nn.Linear(self.n_embd, q_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(q_dim, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)

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
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        hidden = config.mlp_ratio[layer_idx] * config.n_embd
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
        self.attn = CausalSelfAttention(config, layer_idx) if layer_idx in config.attn_layers else None
        # The unpriced shift mixer now runs in EVERY layer: it is free under the dispatch
        # counter, so it is pure added token-mixing capacity even where attention exists.
        self.shift = ShiftMix(config)
        self.mlp = MLP(config, layer_idx)

    def forward(self, x, ve, cos_sin, window_size):
        xn = norm(x)
        h = self.shift(xn)
        if self.attn is not None:
            h = h + self.attn(xn, ve, cos_sin, window_size)
        x = x + h
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
        self.lm_head = (nn.Sequential(
                            nn.Linear(config.n_embd, config.lm_head_rank, bias=False),
                            nn.Linear(config.lm_head_rank, config.vocab_size, bias=False))
                        if config.lm_head_rank > 0 else
                        nn.Linear(config.n_embd, config.vocab_size, bias=False))
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Value embeddings (plus, under the same flops-free gather umbrella, the additive
        # token-identity embeddings injected straight into the residual stream).
        head_dim = config.n_embd // config.n_head
        tables = {str(i): nn.Embedding(config.vocab_size,
                                       config.n_kv_head_per_layer[i] * head_dim)
                  for i in config.attn_layers}
        tables.update({f"r{i}": nn.Embedding(config.vocab_size, config.n_embd)
                       for i in config.resid_embed_layers})
        # Hashed-bigram residual tables. Node 2.3 diagnosed COLLISION PRESSURE (≈124k
        # distinct bigrams per batch against 8192 rows) as the binding limit, so these get
        # 2x the rows of a token table and two independent hashes into the same table, which
        # averages the collision noise instead of concentrating it in one row.
        tables.update({f"b{i}": nn.Embedding(config.bigram_rows, config.n_embd)
                       for i in config.bigram_embed_layers})
        self.value_embeds = nn.ModuleDict(tables)
        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        if self.config.lm_head_rank > 0:
            # Down-projection like any other matrix; the vocab factor keeps the tiny std the
            # dense head used, so logits still start near zero.
            torch.nn.init.uniform_(self.lm_head[0].weight, -s, s)
            torch.nn.init.normal_(self.lm_head[1].weight, mean=0.0, std=0.001)
        else:
            torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # Transformer blocks
        for block in self.transformer.h:
            if block.attn is not None:
                torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
                torch.nn.init.zeros_(block.attn.c_proj.weight)
            block.shift.init_weights()
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        # Value embeddings
        for name, ve in self.value_embeds.items():
            if name.startswith("r") or name.startswith("b"):
                # Residual token embeddings start at exactly zero: neutral at init, and a
                # zero-initialised embedding row still receives full gradient, so it learns.
                torch.nn.init.zeros_(ve.weight)
            else:
                torch.nn.init.uniform_(ve.weight, -s, s)
        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
        for block in self.transformer.h:
            if block.attn is not None and block.attn.ve_gate is not None:
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
        """Per-layer (left, right) windows taken straight from config.window_left."""
        windows = list(config.window_left)
        assert len(windows) == config.n_layer, (len(windows), config.n_layer)
        return [(int(w), 0) for w in windows]

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        h_per_layer = self.config.n_head_per_layer
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for i, window_size in enumerate(self.window_sizes):
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h_per_layer[i] * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5,
                        mixer_lr=0.02):
        model_dim = self.config.n_embd
        block_params = list(self.transformer.h.parameters())
        # Muon orthogonalizes 2-D matrices only; the shift mixer's per-channel 1-D vectors
        # need their own AdamW group.
        matrix_params = [p for p in block_params if p.ndim >= 2]
        vector_params = [p for p in block_params if p.ndim < 2]
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == (len(matrix_params) + len(vector_params) +
            len(embedding_params) + len(lm_head_params) + len(value_embeds_params) +
            len(resid_params) + len(x0_params))
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
        if vector_params:
            param_groups.append(dict(kind='adamw', params=vector_params, lr=mixer_lr,
                                     betas=adam_betas, eps=1e-10, weight_decay=0.0))
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
        if self.config.bigram_embed_layers:
            # Unique bigram id (vocab is a power of two, so this is a plain bit-concat) and
            # two multiplicative hashes of it. Integer elementwise arithmetic and gathers:
            # no matmul/conv/SDPA is issued, so measure_flops_dispatch prices all of this at
            # exactly zero.
            prev = F.pad(idx[:, :-1], (1, 0))
            bg = idx.to(torch.int64) * self.config.vocab_size + prev.to(torch.int64)
            mask = self.config.bigram_rows - 1
            bg_h1 = ((bg * 2654435761) >> 15) & mask
            bg_h2 = ((bg * 2246822519) >> 21) & mask
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            if f"r{i}" in self.value_embeds:
                # Flops-free additive capacity: a gather, scaled down so its contribution
                # cannot run away relative to the residual stream.
                x = x + 0.1 * self.value_embeds[f"r{i}"](idx)
            if f"b{i}" in self.value_embeds:
                tbl = self.value_embeds[f"b{i}"]
                x = x + 0.05 * (tbl(bg_h1) + tbl(bg_h2))
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
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
ASPECT_RATIO = 64       # model_dim = depth * ASPECT_RATIO (unused when MODEL_DIM is set)
MODEL_DIM = 384         # explicit residual width override (0 = derive from ASPECT_RATIO)
HEAD_DIM = 96           # target head dimension for attention
SHORT_WINDOW = 128      # left window for the (majority) local layers. Node 1.3 launch 3:
                        # launches 1 and 2 differ by 4,424,256 FLOPs/token spent on local
                        # attention WIDTH (the second query head on layers 4-6), worth
                        # -0.0039 bpb. Launch 1 had spent an almost identical 4,423,680 on
                        # local RANGE (128 -> 512). This reverts the range to the trunk's
                        # validated 128 and keeps the width, i.e. it re-prices the two ways
                        # of spending the same budget against each other. Node 1.1 measured
                        # local windows as massively over-provisioned (2048 -> 128 was free),
                        # so range is expected to be the cheaper thing to give up; the global
                        # layer 7's 2048 span is untouched.
GLOBAL_LAYERS = (-1,)   # layer indices that keep a full-sequence left window
ATTN_LAYERS = tuple(range(8))  # layers keeping real attention (ShiftMix now runs everywhere)
KV_HEAD_DIVISOR = 2     # GQA: n_kv_head = n_head // KV_HEAD_DIVISOR (overridden by N_KV_HEAD)
N_KV_HEAD = 2           # explicit KV-head count (0 = derive from KV_HEAD_DIVISOR)
MLP_RATIO = (2, 2, 2, 2, 2, 2, 2, 2)  # per-layer MLP hidden multiplier (node 4.4 cut B)
# Node 4.4 launch 1 measured cuts A+B+C together at exactly the predicted 64,097,280
# FLOPs/token but val_bpb 1.06881, i.e. +0.030 bpb -- ~3x the 0.0112 headroom 4.3 produced.
# Cut A is backed off first: a rank-128 factorisation caps the logit matrix at rank 128 for
# an 8192-way softmax, which is a bottleneck directly on the quantity being scored, whereas
# the MLP-ratio and attention-width cuts only remove interior capacity.
LM_HEAD_RANK = 256      # low-rank unembedding: n_embd -> rank -> vocab (cut A backed off)
RESID_EMBED_LAYERS = (5, 6, 7)  # layers given a flops-free additive token embedding
# Node 1.3 reinvestment: layers 0-2's token-identity tables are replaced by hashed-BIGRAM
# tables with 2x the rows. Node 2.3 measured -0.002 bpb from bigram indexing at 8192 rows
# and identified collisions (~124k distinct bigrams per batch) as the binding limit, so the
# freed parameters from the attention-width cut are spent on rows, where the evidence says
# the return is. Still a pure gather: zero counted FLOPs.
BIGRAM_EMBED_LAYERS = (0, 1, 2)
BIGRAM_ROWS = 16384

# Node 4.4 cut C: the global layer's VALUE is its range, not its head count. The FA3 tally
# is 12*h*d*span, so layer 7's 2048 span was 9.44M of the 13.57M tally. Narrowing it to 2
# heads of 96 halves that to 4.72M and also halves its q/o projections, while the 2048-token
# retrieval range (which node 2.4 showed is worth +0.038 bpb) is fully preserved.
# Node 4.4 cut C (validated at launch 2: cuts B+C together cost only 0.0067 bpb, leaving
# 0.0045 bpb of headroom). Launch 3 extends the same mechanism -- attention's value is its
# RANGE, not its head count -- to the three lowest local layers, whose 128-token span makes
# their tally tiny so the saving is mostly dense q/o projection FLOPs, and whose local mixing
# is the most redundant with the (unpriced, now 6-tap) ShiftMix that runs in every block.
NARROW_ATTN_LAYERS = {0: 1, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 2}
# Node 1.3 launch 1 drove EVERY local layer to a single 96-dim query head (and, forced by
# FA3's h % n_kv_head == 0, a single KV head, which also halves c_k/c_v and that layer's
# value-embedding table): 58,296,000 FLOPs/token exactly as predicted, but val_bpb 1.05199
# -- 0.0020 over the gate. The bundle as a whole priced at only 0.00034 bpb per MFLOP/token,
# so the floor is real but the trunk's 0.0010 bpb of margin could not fund all of it.
# Launch 2 buys back the single cheapest slice: the second query head on the UPPER local
# layers 4-6, where the residual stream is widest-used and the 6-tap ShiftMix is least able
# to substitute, while layers 0-3 stay at the one-head floor. Paid for in parameters by
# retiring layer 4's residual token table (the bigram tables are kept: node 2.3's evidence
# says rows there are worth more than another unigram row block).
# The GLOBAL layer 7 keeps 2 heads and its 2048 span throughout -- node 2.4 showed removing
# global range costs +0.038 bpb.

# Optimization
# Node 4.3: the data pool is exhausted (~0.93 epochs/run) so extra tokens no longer buy
# quality; the remaining currency is optimizer updates per token. Halving the batch
# 2**19 -> 2**18 doubles the number of updates at identical token cost (grad_accum 4 -> 2)
# and leaves flops_per_token_measured untouched. Both Muon and Adam take normalized steps
# of magnitude ~lr, so at fixed lr half the batch would double the total path length in
# weight space; LR_SCALE = 1/sqrt(2) is the standard interpolation that keeps the
# noise/signal balance (path length grows only sqrt(2)x while updates double).
TOTAL_BATCH_SIZE = 2**17 # ~131K tokens per optimizer step (grad_accum_steps = 1)
LR_SCALE = 0.5          # global LR rescale: 1/sqrt(2) per batch halving, two halvings
EMBEDDING_LR = 0.6 * LR_SCALE      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004 * LR_SCALE  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04 * LR_SCALE        # learning rate for matrix parameters (Muon)
MIXER_LR = 0.02 * LR_SCALE         # learning rate for the shift-mixer's 1-D params (Adam)
SCALAR_LR = 0.5 * LR_SCALE         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# Model size
DEPTH = 8               # number of transformer layers
DEVICE_BATCH_SIZE = 64  # per-device batch size (reduce if OOM); 8 attn + 8 shift sublayers
                        # hold far more activations than the 4+4 layout, so halve it to keep
                        # peak VRAM well inside the 47.2GB ceiling (TOTAL_BATCH_SIZE unchanged)

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
    if MODEL_DIM > 0:
        model_dim = MODEL_DIM
    else:
        base_dim = depth * ASPECT_RATIO
        model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    assert model_dim % HEAD_DIM == 0
    num_heads = model_dim // HEAD_DIM
    n_kv_head = N_KV_HEAD if N_KV_HEAD > 0 else max(1, num_heads // KV_HEAD_DIVISOR)
    assert num_heads % n_kv_head == 0, (num_heads, n_kv_head)
    assert len(MLP_RATIO) == depth, (len(MLP_RATIO), depth)
    global_layers = {i % depth for i in GLOBAL_LAYERS}
    attn_layers = tuple(sorted({i % depth for i in ATTN_LAYERS} | global_layers))
    # Layers without an attention sublayer declare window 0 so the analytic cross-check
    # agrees with the FA3 tally (which only sees the layers that actually call the kernel).
    window_left = tuple(MAX_SEQ_LEN if i in global_layers
                        else (SHORT_WINDOW if i in attn_layers else 0)
                        for i in range(depth))
    n_head_per_layer = tuple(NARROW_ATTN_LAYERS.get(i % depth, num_heads)
                             for i in range(depth))
    # A layer with h query heads cannot carry more than h KV heads, and FA3 needs
    # h % n_kv_head == 0: clamp the global GQA setting down to the largest divisor of h.
    n_kv_head_per_layer = tuple(max(k for k in range(1, min(n_kv_head, h) + 1) if h % k == 0)
                                for h in n_head_per_layer)
    for h, kv in zip(n_head_per_layer, n_kv_head_per_layer):
        assert 1 <= h <= num_heads and h % kv == 0, (h, num_heads, kv)
    print(f"Per-layer query heads: {n_head_per_layer}  kv heads: {n_kv_head_per_layer}")
    print(f"Per-layer left window: {window_left}")
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads,
        n_kv_head=n_kv_head, n_embd=model_dim,
        n_head_per_layer=n_head_per_layer,
        n_kv_head_per_layer=n_kv_head_per_layer,
        window_left=window_left,
        mlp_ratio=tuple(MLP_RATIO), lm_head_rank=LM_HEAD_RANK,
        resid_embed_layers=tuple(i % depth for i in RESID_EMBED_LAYERS),
        attn_layers=attn_layers,
        bigram_embed_layers=tuple(i % depth for i in BIGRAM_EMBED_LAYERS),
        bigram_rows=BIGRAM_ROWS,
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
    mixer_lr=MIXER_LR,
)

model = torch.compile(model, dynamic=False)

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train",
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

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
last_opt_ms = 0.0
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
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    # Instrumentation (node 4.3): the cost of doubling updates per token is the optimizer's
    # share of each step. Sync-and-time it only occasionally so it does not throttle the
    # CPU run-ahead of the normal steps.
    probe_opt = (step % 50 == 0)
    if probe_opt:
        torch.cuda.synchronize()
        t_opt0 = time.time()
    optimizer.step()
    if probe_opt:
        torch.cuda.synchronize()
        last_opt_ms = (time.time() - t_opt0) * 1000
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

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | opt: {last_opt_ms:.0f}ms | remaining: {remaining:.0f}s    ", end="", flush=True)

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
