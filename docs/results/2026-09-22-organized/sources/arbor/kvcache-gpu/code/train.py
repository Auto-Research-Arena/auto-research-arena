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
    # Window LENGTH is a trained hyperparameter, not a storage detail: with right-sized rings
    # the cache is exactly proportional to the summed window lengths, so `short_window` is the
    # single biggest lever left on kv_cache_bytes.
    short_window: int = 256
    # A GRADED all-short stack: the upper CLA group gets a longer (but still local) window,
    # which is the cheapest real quality lever left -- bytes grow linearly in the window while
    # a full-context 'L' layer would cost 2048 slots at once.
    med_window: int = 512
    mlp_mult: int = 5       # c_fc expansion; non-KV capacity, reinvested from the FLOPs slack


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


CLA_GROUP = 4   # cross-layer KV sharing: layers are grouped in fours, one k/v per group

# Node 3.10: the VE-table mixture is CUT back to one table per producer (the reference
# substrate's layout). Node 3.6 measured that the extra tables buy essentially no bpb, yet
# at 6 per producer they consumed 12.6M of the 46.4M parameters, 12 gathers + 12 AdamW
# states per step, and a 6-term gated sum in the v path. Those parameters are reallocated
# into the ONE capacity form that was measured to help (node 3.4: MLP_MULT 4->5 was
# bpb-positive) -- see MLP_MULT below. The previous blocker was peak_vram at 95% of its
# ceiling; the training-microbatch halving at node 3.9 dropped it to 48%, so the MLP's
# activation memory is now affordable.
VE_PER_PRODUCER = 1


def kv_producer(layer_idx):
    """Under CLA-4 the first layer of each group of four computes k/v; the rest reuse them."""
    return layer_idx % CLA_GROUP == 0


def kv_slot(layer_idx):
    """Index of the shared cache entry a layer reads/writes."""
    return layer_idx // CLA_GROUP


def has_ve(layer_idx, n_layer):
    """Value Embeddings live on the k/v-producing layer of every group.

    The value embedding feeds v, and under CLA-4 only producers own a v, so the ve tables sit
    on the producers (0,4). Each producer carries VE_PER_PRODUCER tables, each with its own
    input-dependent per-head gate: the sum of two ungated tables would collapse into one
    table, but independently gated ones do not, so the table count (and the parameter spend)
    can exceed the number of producers and still buy capacity.
    """
    return kv_producer(layer_idx)


def ve_keys(layer_idx, n_layer):
    """ModuleDict keys of the value-embedding tables owned by `layer_idx` (may be empty).

    Flat keys, one Embedding per key: prepare.py's analytic cross-check reads
    `ve.weight.numel()` off every value of `model.value_embeds`, so the dict must stay a flat
    dict of Embeddings rather than nesting a ModuleList per layer. ('.' is not a legal
    ModuleDict key, hence the underscore.)
    """
    if not has_ve(layer_idx, n_layer):
        return []
    return [f"{layer_idx}_{t}" for t in range(VE_PER_PRODUCER)]


# ---------------------------------------------------------------------------
# KV cache codec: int8 group quantization
#
# The cache is a storage format, not a computation. `forward` -- the branch val_bpb is read
# from -- never touches it, so quantizing it carries exactly zero quality risk to the trained
# model; the only thing it can move is how far the decode path drifts from `forward`
# (decode_tv_distance) and how long a step takes. Keys enter the cache post-rms-norm (unit
# rms per head, no outliers) and quantize almost perfectly; values are the looser tensor, so
# the scale group is a half head rather than a whole one -- 2 bf16 scales per (position,
# head) is 1.6% of the payload and buys outlier headroom.
# ---------------------------------------------------------------------------

# Scale metadata is NON-payload overhead: at group 16 it is 32 B on top of a 256 B int8
# payload per slot-position (11%). The groups are coarsened ASYMMETRICALLY (node 1.5), because
# the two tensors do not quantize alike: k enters the cache post-rms-norm (unit rms per head,
# no outliers), so its per-group absmax barely grows with the group size and it tolerates
# group 64 (2 scales per position per head = 4 B); v is the looser tensor and keeps a finer
# group 32 (8 B). Scale metadata per slot-position: 32 B -> 12 B, measured strictly better
# decode TV at node 1.5. This is an INFERENCE-SIDE format change only: `forward` never reads
# the cache, so the trained model and val_bpb are untouched.
KV_QUANT_GROUP_K = 64    # elements of head_dim sharing one scale, keys
KV_QUANT_GROUP_V = 32    # ... values
KV_QUANT_GROUP = KV_QUANT_GROUP_V   # default for callers that do not care
KV_QUANT_MAX = 127.0     # symmetric int8 range; -128 is never emitted
# SCALE DTYPE: fp16, not bf16. Identical 2 bytes, but 10 mantissa bits instead of 7, so the
# stored scale is quantized 8x more finely; bf16 scale rounding alone injected up to ~0.4%
# multiplicative error on EVERY reconstructed element, comparable to the int8 step error
# itself. Dynamic range is a non-issue: the scales are absmax/127 of post-rms-norm keys
# (unit rms, so O(1e-3..1e-2)) and of values (rms ~10-30), all far inside fp16's range, and
# the encoder clamps the scale away from fp16's subnormal floor before it divides.
KV_SCALE_DTYPE = torch.float16
KV_SCALE_MIN = 1e-7      # > fp16 min subnormal (5.96e-8): a stored scale is never 0
# STEP SIZE: raw absmax lets one outlier set the step for all 32-64 elements of the group, and
# it also fixes the code lattice with no regard for where the other elements fall. Rather than
# a blind shrink (which pays (1-c)*absmax of clipping error on the extreme element -- more
# than it saves on the rest for outlier-free groups), each group PICKS its own step from a
# small symmetric grid around absmax by minimising ITS OWN reconstruction SSE, scored on the
# fp16-rounded candidate so the winner is exactly what the cache reconstructs. c=1.0 is in the
# grid, so the chosen scale is never worse than plain absmax. Candidates above 1 are included:
# they clip nothing and sometimes align the lattice better with the bulk of the group (offline
# simulation: +-6% grid cuts per-element rms error ~7% on Gaussian keys and ~3% on
# heavy-tailed values, roughly 3x what shrink-only candidates buy).
# The search is VECTORIZED over the candidate axis -- one set of kernels regardless of grid
# size -- because kv_quantize also runs once per decode step inside the CUDA graph.
KV_SCALE_GRID_N = 33
KV_SCALE_GRID_STEP = 0.004      # c in 1 +- 0.064
_KV_SCALE_GRID = {}
_KV_FORMAT_REPORTED = False
_KV_ERROR_REPORTED = False


def _scale_grid(device):
    g = _KV_SCALE_GRID.get(device)
    if g is None:
        half = KV_SCALE_GRID_N // 2
        g = 1.0 + KV_SCALE_GRID_STEP * torch.arange(
            -half, KV_SCALE_GRID_N - half, device=device, dtype=torch.float32)
        _KV_SCALE_GRID[device] = g
    return g


def kv_quantize(x, group=KV_QUANT_GROUP):
    """bf16 (B,T,H,D) -> int8 payload (B,T,H,D) + fp16 scales (B,T,H*D/group).

    Symmetric per-group int8, with the step chosen per group from a grid around absmax to
    minimise that group's squared reconstruction error. The scale is rounded to fp16 *before*
    it divides (and each candidate is scored on its rounded value), so the code stored is the
    one the fp16 scale read back from the cache reconstructs -- encoder and decoder cannot
    disagree.
    """
    B, T, H, D = x.shape
    G = D // group
    xg = x.float().view(1, B, T, H, G, group)
    absmax = xg.abs().amax(dim=-1).clamp_min(1e-12)                    # (1,B,T,H,G)
    cand = _scale_grid(x.device).view(-1, 1, 1, 1, 1) / KV_QUANT_MAX
    # round-trip each candidate through fp16 so the scored scale is bit-exactly the stored one
    s = (absmax * cand).clamp_min(KV_SCALE_MIN).to(KV_SCALE_DTYPE).float()   # (C,B,T,H,G)
    codes = torch.round(xg / s.unsqueeze(-1)).clamp_(-KV_QUANT_MAX, KV_QUANT_MAX)
    sse = (codes * s.unsqueeze(-1) - xg).pow(2).sum(dim=-1)            # (C,B,T,H,G)
    pick = sse.argmin(dim=0, keepdim=True)
    scale = s.gather(0, pick).squeeze(0)                               # (B,T,H,G)
    # Re-derive the code from the winning (fp16-exact) scale: identical to that candidate's
    # row of `codes`, and cheaper than gathering a (C,...,group) tensor.
    code = torch.round(xg.squeeze(0) / scale.unsqueeze(-1))
    code = code.clamp_(-KV_QUANT_MAX, KV_QUANT_MAX).to(torch.int8)
    return code.view(B, T, H, D), scale.to(KV_SCALE_DTYPE).view(B, T, H * G)


def kv_dequantize(code, scale, group=KV_QUANT_GROUP):
    """int8 payload + fp16 scales -> bf16 (B,T,H,D).

    The product is formed in fp32 and rounded once to bf16: the reconstruction is then the
    nearest bf16 to the represented value, so none of int8's resolution is thrown away by a
    bf16 multiply.
    """
    B, T, H, D = code.shape
    G = D // group
    x = code.view(B, T, H, G, group).float() * scale.view(B, T, H, G, 1).float()
    return x.to(torch.bfloat16).view(B, T, H, D)


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


def apply_ve(attn, x, ve):
    """Gated sum of this layer's value-embedding tables, shaped like `v`.

    One function, called by both `compute_kv` and `_decode_body`, so the trained path and the
    cached path cannot drift apart. `ve` is the list of gathered tables (B,T,kv_dim). The
    2/n factor with zero-init gates makes the total initial ve contribution exactly 1.0 --
    identical to the single-table trunk -- so adding tables changes capacity, not init.
    """
    B, T = x.shape[0], x.shape[1]
    n = len(ve)
    gate = (2.0 / n) * torch.sigmoid(attn.ve_gate(x[..., :attn.ve_gate_channels]))
    gate = gate.view(B, T, n, attn.n_kv_head, 1)
    out = gate[:, :, 0] * ve[0].view(B, T, attn.n_kv_head, attn.head_dim)
    for t in range(1, n):
        out = out + gate[:, :, t] * ve[t].view(B, T, attn.n_kv_head, attn.head_dim)
    return out


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
        self.produces_kv = kv_producer(layer_idx)
        if self.produces_kv:
            self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
            self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.n_ve = len(ve_keys(layer_idx, config.n_layer))
        # One gate row per (table, kv head).
        self.ve_gate = (nn.Linear(self.ve_gate_channels, self.n_ve * self.n_kv_head, bias=False)
                        if (self.produces_kv and self.n_ve) else None)

    def compute_kv(self, x, ve, cos_sin):
        """Produce the k/v this layer and its group partners all attend to.

        k leaves here rotated and normalised, i.e. exactly as it is stored in the cache, so
        the shared cache entry is read identically by every member of the group.
        """
        B, T, C = x.size()
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embeddings with input-dependent gates per
        # (table, head). Several tables under separate gates is a rank-n_ve addition to v.
        if ve:
            v = v + apply_ve(self, x, ve)

        cos, sin = cos_sin
        k = norm(apply_rotary_emb(k, cos, sin))
        return k, v

    def forward(self, x, kv, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = cos_sin
        q = norm(apply_rotary_emb(q, cos, sin))
        k, v = kv

        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.mlp_mult * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.mlp_mult * config.n_embd, config.n_embd, bias=False)

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

    def forward(self, x, kv, ve, cos_sin, window_size):
        h = norm(x)
        if self.attn.produces_kv:
            kv = self.attn.compute_kv(h, ve, cos_sin)
        x = x + self.attn(h, kv, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x, kv


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
        # Value embeddings: a flat dict of tables, VE_PER_PRODUCER of them per k/v producer
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.ve_keys = {i: ve_keys(i, config.n_layer) for i in range(config.n_layer)}
        self.value_embeds = nn.ModuleDict({
            key: nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) for key in self.ve_keys[i]
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
            if block.attn.produces_kv:
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
        assert all(c in "SML" for c in pattern)
        long_window = config.sequence_len
        short_window = config.short_window
        med_window = config.med_window
        assert 0 < short_window <= med_window <= long_window
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0),
                          "M": (med_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # The substrate default force-set the LAST layer to full context. That is a
        # substrate-author default, not a task constraint: an ALL-SHORT stack is a legitimate
        # architecture (node 3.4 found shorter windows quality-neutral-to-positive), and
        # forcing one 2048-slot cache would be 4x this design's entire cache. Honour the
        # pattern as written.
        # CLA-4: all four members of a group read ONE cache entry, so they must mask it the
        # same way. Take the group's minimum window (a shorter window is a strict subset of a
        # longer one, so the shared entry is always read consistently). The default pattern
        # is chosen so no group actually disagrees.
        for start in range(0, config.n_layer - config.n_layer % CLA_GROUP, CLA_GROUP):
            group = window_sizes[start:start + CLA_GROUP]
            shared = min(w[0] for w in group)
            for offset in range(CLA_GROUP):
                window_sizes[start + offset] = (shared, 0)
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
        kv = None
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = [self.value_embeds[key](idx) for key in self.ve_keys[i]]
            x, kv = block(x, kv, ve, cos_sin, self.window_sizes[i])
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

    # -----------------------------------------------------------------------
    # The decode protocol. REQUIRED: prepare.py refuses to measure a model that
    # does not implement it. A candidate that wants no cache implements
    # `decode_step` as a full recompute of `forward` and pays for it in latency.
    # -----------------------------------------------------------------------

    def init_decode_state(self, batch, max_len, graph=True):
        """Preallocate every layer's cache, and keep the write position ON DEVICE.

        Both give the per-step call constant shapes and no Python-visible position: a lazy
        cache slot is a new shape mid-loop, and a Python `pos` int makes Dynamo guard on its
        value and trip `recompile_limit`. `seq` is int32, advanced with `add_()`, and read by
        the attention kernel as `cache_seqlens`.

        `graph=False` must run the step eagerly and capture nothing. It is how the instrument
        reads cache bytes without a graph's private pool in them, so a candidate that ignores
        the flag reports its own pool as cache and is charged for it.

        RIGHT-SIZED RINGS. A sliding-window layer can never read a slot older than its
        window, so a shared cache entry with window W needs exactly W+1 physical positions,
        not max_len: position p lands in slot p mod (W+1) and the entry it overwrites is
        p-(W+1), already out of reach. The hand-written decode attention addresses the ring
        directly (no FA3 page_table needed), so this costs nothing in bytes moved -- and it
        SAVES time, because the per-step dequantization now spans the ring rather than
        max_len. Because the ring length is exactly window+1, EVERY live ring slot is
        reachable once the ring has filled, so the sliding-window mask degenerates to "this
        residue has been written at all" (see `_decode_attend`).
        """
        assert max_len <= self.cos.size(1), f"max_len {max_len} exceeds rotary table"
        cfg = self.config
        head_dim = cfg.n_embd // cfg.n_head
        dev = self.transformer.wte.weight.device
        # CLA-2: one k/v pair per PAIR of layers, not per layer. The payload is int8 and the
        # scales are bf16, and both live in ONE byte arena (the scale region is a `view` of
        # its tail), so the allocator rounds a single block and the reading is the format's
        # size rather than the allocator's granularity times the number of buffers.
        n_slots = (cfg.n_layer + CLA_GROUP - 1) // CLA_GROUP
        n_groups = (head_dim // KV_QUANT_GROUP_K, head_dim // KV_QUANT_GROUP_V)
        # Both members of a CLA pair share one window (the pair minimum, set in
        # _compute_window_sizes), so the ring length is a property of the slot.
        rings = []
        for s in range(n_slots):
            window = self.window_sizes[s * CLA_GROUP][0]
            rings.append(max_len if window < 0 else min(window + 1, max_len))
        # int8 payload elements and bf16 scale elements, per buffer, per slot. k and v carry
        # different numbers of scales (different group sizes), so the scale region is sized
        # per tensor rather than per slot.
        per = [batch * r * cfg.n_kv_head * head_dim for r in rings]
        per_scale = [[batch * r * cfg.n_kv_head * g for g in n_groups] for r in rings]
        assert all(p % 2 == 0 for p in per)
        # bf16 arena; the payload regions are re-`view`ed as int8. Viewing a bf16 slice as a
        # SMALLER dtype is unconditionally legal, which the other direction is not. One
        # arena, so the allocator rounds once (512 B) instead of once per buffer. EVERY
        # auxiliary buffer lives in it too: `seq` (int32 view, 2 bf16 slots) and the ring-slot
        # id table (int16 view, 1 slot per ring position) were separate tiny tensors paying a
        # full 512 B allocator block each; inside the arena they cost their own bytes only.
        code_off, scale_off, cursor = [], [], 0
        for s in range(n_slots):
            code_off.append((cursor, cursor + per[s] // 2))     # k, v payload starts
            cursor += per[s]
        for s in range(n_slots):
            scale_off.append((cursor, cursor + per_scale[s][0]))   # k, v scale starts
            cursor += per_scale[s][0] + per_scale[s][1]
        seq_off = cursor
        cursor += 2 * batch                      # int32 write position per sequence
        pos_off = cursor
        cursor += max(rings)                     # int16 ring-slot ids
        arena = torch.zeros(cursor, dtype=torch.bfloat16, device=dev)
        _scale_grid(dev)   # materialise the codec's candidate grid OUTSIDE any graph capture
        analytic = arena.numel() * 2
        global _KV_FORMAT_REPORTED
        if not _KV_FORMAT_REPORTED:
            _KV_FORMAT_REPORTED = True
            scale_bytes = 2 * sum(p[0] + p[1] for p in per_scale)
            print(f"[kv-ring] max_len={max_len} rings={rings} "
                  f"slots_saved={1 - sum(rings) / (n_slots * max_len):.4f}")
            print(f"[kv-codec] int8 group_k={KV_QUANT_GROUP_K} group_v={KV_QUANT_GROUP_V} "
                  f"scale_dtype={KV_SCALE_DTYPE} grid={KV_SCALE_GRID_N}x{KV_SCALE_GRID_STEP} "
                  f"buffers={2 * n_slots} payload={2 * sum(per)}B scales={scale_bytes}B "
                  f"aux={2 * (2 * batch + max(rings))}B arena={analytic}B "
                  f"bytes_per_slot_position={analytic / sum(rings):.2f}")
        def code(slot, which):
            start = code_off[slot][which]
            return arena.narrow(0, start, per[slot] // 2).view(torch.int8).view(
                batch, rings[slot], cfg.n_kv_head, head_dim)
        def scale(slot, which):
            # Same 2 bytes as the bf16 arena element it aliases, 3 more mantissa bits.
            start = scale_off[slot][which]
            return arena.narrow(0, start, per_scale[slot][which]).view(KV_SCALE_DTYPE).view(
                batch, rings[slot], cfg.n_kv_head * n_groups[which])
        assert seq_off % 2 == 0, "int32 view of the arena must be 4 B aligned"
        seq = arena.narrow(0, seq_off, 2 * batch).view(torch.int32)
        pos_ids = arena.narrow(0, pos_off, max(rings)).view(torch.int16)
        pos_ids.copy_(torch.arange(max(rings), dtype=torch.int16, device=dev))
        return {
            "max_len": max_len,
            "graph_enabled": bool(graph),
            "seq": seq,
            "arena": arena,
            "rings": rings,
            # Ring-slot ids, not logical positions: only max(rings) of them are ever needed,
            # and int16 (rings are < 2**15) is the narrowest a sliding compare can hold.
            "pos_ids": pos_ids,
            "kc": [code(s, 0) for s in range(n_slots)],
            "vc": [code(s, 1) for s in range(n_slots)],
            "ks": [scale(s, 0) for s in range(n_slots)],
            "vs": [scale(s, 1) for s in range(n_slots)],
        }

    def reset_decode_state(self, state):
        """Return `state` to position 0 so the same buffers serve another request.

        A fresh state per request would invalidate an address-bound CUDA graph. Stale bytes
        above the position are unreachable -- attention reads only `cache_seqlens` keys -- so
        zeroing them would be timed work serving never does.
        """
        state["seq"].zero_()
        return state

    def _decode_body(self, idx, state, prefill):
        """`forward`'s body with the cache spliced in. `prefill` is a Python bool that is
        constant per variant, so the two branches are two graphs, not one per position."""
        B, Tn = idx.size()
        if prefill:
            cos, sin = self.cos[:, :Tn], self.sin[:, :Tn]
        else:
            seq_idx = state["seq"].to(torch.int64)
            cos = self.cos.index_select(1, seq_idx)
            sin = self.sin.index_select(1, seq_idx)

        x = norm(self.transformer.wte(idx))
        x0 = x
        k = v = None
        kv_deq = None                 # the pair's dequantized cache, dequantized once
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = [self.value_embeds[key](idx) for key in self.ve_keys[i]]
            attn = block.attn
            h = norm(x)
            q = attn.c_q(h).view(B, Tn, attn.n_head, attn.head_dim)
            q = norm(apply_rotary_emb(q, cos, sin))
            if attn.produces_kv:
                # Identical arithmetic to CausalSelfAttention.compute_kv: k is rotated and
                # normalised before it enters the cache, and the pair partner reads it as is.
                k = attn.c_k(h).view(B, Tn, attn.n_kv_head, attn.head_dim)
                v = attn.c_v(h).view(B, Tn, attn.n_kv_head, attn.head_dim)
                if ve:
                    v = v + apply_ve(attn, h, ve)
                k = norm(apply_rotary_emb(k, cos, sin))

            kc, vc = state["kc"][kv_slot(i)], state["vc"][kv_slot(i)]
            ks, vs = state["ks"][kv_slot(i)], state["vs"][kv_slot(i)]
            if prefill:
                # The prompt's own attention runs on the full-precision k,v, so prefill is
                # bit-identical to `forward`; the cache write is the quantized copy the
                # width-1 steps will read. The reference full-precision path exists only to
                # produce the quantized entries.
                if attn.produces_kv:
                    k_code, k_scale = kv_quantize(k, KV_QUANT_GROUP_K)
                    v_code, v_scale = kv_quantize(v, KV_QUANT_GROUP_V)
                    # Ring write. A prompt longer than the ring only needs its LAST `ring`
                    # positions kept: everything before that is outside the window of every
                    # step that follows, so writing it would be an overwrite of itself. The
                    # kept tail is `ring` consecutive positions, i.e. one write per residue,
                    # so index_copy_ sees no duplicate index.
                    ring = kc.size(1)
                    keep = min(Tn, ring)
                    if keep == ring and Tn > ring:
                        ridx = (torch.arange(Tn - keep, Tn, device=kc.device) % ring)
                        kc.index_copy_(1, ridx, k_code[:, Tn - keep:])
                        ks.index_copy_(1, ridx, k_scale[:, Tn - keep:])
                        vc.index_copy_(1, ridx, v_code[:, Tn - keep:])
                        vs.index_copy_(1, ridx, v_scale[:, Tn - keep:])
                    else:
                        kc[:, :Tn] = k_code
                        ks[:, :Tn] = k_scale
                        vc[:, :Tn] = v_code
                        vs[:, :Tn] = v_scale
                    global _KV_ERROR_REPORTED
                    if not _KV_ERROR_REPORTED:
                        # One-shot codec diagnostic, from the untimed cache-bytes probe.
                        for name, ref, code_, sc, g in (("k", k, k_code, k_scale, KV_QUANT_GROUP_K),
                                                        ("v", v, v_code, v_scale, KV_QUANT_GROUP_V)):
                            rec = kv_dequantize(code_, sc.view(B, Tn, -1), g)
                            err = (rec.float() - ref.float())
                            rms = ref.float().pow(2).mean().sqrt()
                            amax = ref.float().view(B, Tn, -1, g).abs().amax(-1).clamp_min(1e-12)
                            cc = (sc.float().view(B, Tn, -1) * KV_QUANT_MAX / amax)
                            print(f"[kv-codec] layer{i} {name} group={g}: rms={rms:.4f} "
                                  f"err_rms={err.pow(2).mean().sqrt():.6f} "
                                  f"err_max={err.abs().max():.6f} "
                                  f"scale_dtype={sc.dtype} shrink_mean={cc.mean():.4f} "
                                  f"shrink_frac_lt1={(cc < 0.999).float().mean():.4f}")
                        if i == 0:
                            _KV_ERROR_REPORTED = True
                y = fa3.flash_attn_func(q, k, v, causal=True,
                                        window_size=self.window_sizes[i])
            else:
                # An int8 cache cannot be handed to flash_attn_with_kvcache, so the step's
                # attention is written by hand: append (quantized) at the device-held
                # position, dequantize the slot, and score against it under an explicitly
                # built sliding-window causal mask. The decode path is not FLOPs-counted, so
                # this costs latency only. Shapes are constant per step and the position is
                # only ever a device tensor, so the CUDA graph captures it unchanged.
                if attn.produces_kv:
                    # Ring address of the write, entirely on device: `seq` never becomes a
                    # Python int, so the graph replays this unchanged.
                    pos = state["seq"].to(torch.int64).remainder(kc.size(1))
                    k_code, k_scale = kv_quantize(k, KV_QUANT_GROUP_K)
                    v_code, v_scale = kv_quantize(v, KV_QUANT_GROUP_V)
                    kc.index_copy_(1, pos, k_code)
                    ks.index_copy_(1, pos, k_scale)
                    vc.index_copy_(1, pos, v_code)
                    vs.index_copy_(1, pos, v_scale)
                    # One dequantization per pair: the consumer reads the same slot with the
                    # same (pair-minimum) window, so it reuses the producer's reconstruction.
                    # The ring is `window+1` long, so this dequantizes ONLY the reachable
                    # window instead of all max_len slots -- the step's dominant cost.
                    kv_deq = (kv_dequantize(kc, ks, KV_QUANT_GROUP_K),
                              kv_dequantize(vc, vs, KV_QUANT_GROUP_V))
                    # The step's OWN key/value are in hand at full precision, and the query
                    # attends to them with the largest single weight, so put the exact values
                    # back over their reconstruction: strictly closer to `forward`, and the
                    # index is the same device tensor the write used, so the graph is happy.
                    kv_deq[0].index_copy_(1, pos, k)
                    kv_deq[1].index_copy_(1, pos, v)
                y = self._decode_attend(q, kv_deq[0], kv_deq[1], state)
            x = x + attn.c_proj(y.contiguous().view(B, Tn, -1))
            x = x + block.mlp(norm(x))

        x = norm(x[:, -1:, :])
        softcap = 15
        logits = self.lm_head(x).float()
        return softcap * torch.tanh(logits / softcap)

    def _decode_attend(self, q, k, v, state):
        """One width-1 step's attention against the dequantized RING.

        `q` is (B,1,Hq,D) and the ring is (B,R,1,D) under MQA, so the whole step is two small
        batched matmuls: every query head scores against the single kv head.

        The sliding-window mask collapses. The ring is R = min(window+1, max_len) long, so
        ring slot r holds the LARGEST logical position j <= pos with j == r (mod R), and
        pos - j < R <= window+1 puts every live slot inside FA3's `causal=True,
        window_size=(window,0)` set (pos-window <= j <= pos) automatically -- a ring exactly
        the width of the window cannot hold an unreachable entry. The one thing left to
        exclude is a residue that has not been written yet, which is r > pos, so the mask is
        `r <= pos`. (When R == max_len, no wrap can happen at all and r IS the position, so
        the same expression is the plain causal mask.) This is the same "pre-window slots are
        provably unreachable" argument that made the paged version bit-exact, expressed as a
        modulo instead of a page table.

        Softmax runs in fp32 for the same reason FA3 keeps its accumulator there: a bf16
        score would perturb the distribution by more than the int8 storage does.
        """
        B, R, _, D = k.shape
        pos = state["seq"].to(torch.int64).view(B, 1)
        r = state["pos_ids"][:R].view(1, R)
        allowed = r <= pos
        scores = torch.einsum('bhd,bld->bhl', q[:, 0].float(), k[:, :, 0].float())
        scores = scores * (D ** -0.5)
        scores = scores.masked_fill(~allowed.unsqueeze(1), float('-inf'))
        probs = torch.softmax(scores, dim=-1)
        y = torch.einsum('bhl,bld->bhd', probs, v[:, :, 0].float())
        return y.to(q.dtype).unsqueeze(1)

    def decode_step(self, idx, state):
        """`logits, state = model.decode_step(idx, state)`, logits for the LAST position only.

        Width > 1 is the prefill and runs once per request; width 1 is the step, and it is
        the call the graph replays. `seq` is advanced on both paths, so the tensor is the
        single source of the position.
        """
        if idx.size(1) > 1:
            logits = self._decode_body(idx, state, prefill=True)
            state["seq"].add_(idx.size(1))
            return logits, state
        if not state.get("graph_enabled", True):
            logits = self._decode_body(idx, state, prefill=False)
            state["seq"].add_(1)
            return logits, state
        # The graph lives ON the state it was captured against, so its lifetime is exactly
        # that state's and there is no key to get wrong. A state whose capture failed holds
        # an uncaptured object and runs eager forever, which is slow and correct.
        graph = state.get("graph")
        if graph is None:
            graph = state["graph"] = _GraphedDecodeStep(self, state)
        if not graph.captured:
            logits = self._decode_body(idx, state, prefill=False)
            state["seq"].add_(1)
            return logits, state
        return graph.replay(idx), state


class _GraphedDecodeStep:
    """A manually captured `torch.cuda.CUDAGraph` over one cached decode step.

    Captured by hand, not by `torch.compile(mode="reduce-overhead")`: that mode copies graph
    inputs into a static pool, and `fa3.flash_attn_with_kvcache` appends k/v in place inside
    an opaque custom op, so inductor cannot see the mutation and the append lands in the pool
    copy instead of the real cache. Measured: fastest available configuration, 4.49 bits per
    token destroyed, no warning.

    Three constraints follow:

      1. The graph is bound to its state's addresses, so it is stored ON that state and
         cannot be reused across states.
      2. `seq.add_(1)` is captured after the attention reads it, so a replay attends,
         appends, then increments with no host round-trip.
      3. Warmup and capture execute their kernels, so `seq` is saved and restored around
         both; junk above `seq` is unreachable to `cache_seqlens`.

    Capture failure sets `captured` False and runs eager -- slow, correct, and visible.
    """

    WARMUP_REPLAYS = 3

    def __init__(self, model, state):
        self.model = model
        self.state = state
        self.captured = False
        self.reason = ""
        self.static_idx = torch.zeros(1, 1, dtype=torch.int64, device=state["seq"].device)
        self.static_logits = None
        self.graph = None
        if state["seq"].numel() != 1:
            self.reason = f"batch {state['seq'].numel()} != 1"
            return
        try:
            self._capture()
            self.captured = True
        except Exception as exc:            # noqa: BLE001 -- recorded, then paid for in ms
            self.reason = f"{type(exc).__name__}: {exc}"
            print(f"[decode-graph] capture failed, running eager: {self.reason}")
            self.graph, self.static_logits = None, None

    def _advance(self):
        logits = self.model._decode_body(self.static_idx, self.state, prefill=False)
        self.state["seq"].add_(1)
        return logits

    def _capture(self):
        seq0 = self.state["seq"].clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(self.WARMUP_REPLAYS):
                self.state["seq"].copy_(seq0)
                self._advance()
        torch.cuda.current_stream().wait_stream(stream)

        self.state["seq"].copy_(seq0)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_logits = self._advance()
        self.state["seq"].copy_(seq0)          # the capture itself executed one increment

    def replay(self, idx):
        self.static_idx.copy_(idx)
        self.graph.replay()
        return self.static_logits


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
N_KV_HEAD = 1           # KV heads shared by all query heads (1 = multi-query attention)
# CLA-4 groups layers (0..3) and (4..7) onto one cache entry each: 2 distinct KV caches for
# 8 layers. The pattern is GRADED all-short -- no layer is full-context, so the 2048-slot
# cache that was 73% of the trunk's bytes is gone -- but the upper group, which is where
# long-range recall is assembled, gets a 512 window instead of 256. Attempt 1 of this node
# ran the flat 256/256 version: it scored 150,528 B but val_bpb 1.05570, just over the <1.05
# gate (sibling node 3.5 showed all-short alone already costs +0.0202 at CLA-2). Doubling one
# group's window costs 256 slots, not 2048, and both groups still mask their shared entry
# consistently, so the min-window fallback never truncates.
WINDOW_PATTERN = "SSSSMMMM" # sliding window pattern: L=full, M=MED_WINDOW, S=SHORT_WINDOW
# The window LENGTHS are trained hyperparameters. With right-sized rings the cache is exactly
# sum over distinct (CLA) caches of window+1 slots: 257 + 513 = 770 slots against the trunk's
# 2819 and against sibling 3.5's 1028.
# Node 3.10 set these to [161,225]=386 slots. Node 1.8 leaves them EXACTLY there and changes
# only the inference-side storage format: bytes per slot-position 288 B -> 269.18 B (256 B
# int8 payload for 1 KV head x 128 dims + 4 B k scales at group 64 + 8 B v scales at group 32,
# plus the two in-arena auxiliary buffers amortized over the ring), analytic arena
# 111,168 -> 103,902 B with only allocator rounding on top. Node 1.6 showed cutting the
# windows further (128/192) alone breaks the <1.05 gate (1.05515) because window price is
# super-linear below ~160/224, so the windows are not touched here; and node 1.7 showed the
# MLP 6x -> 5x revert is what cost +0.00216 bpb, so MLP_MULT stays at 6.
# Node 3.14 spends two independently MEASURED bpb credits on slots. Credit 1 is node 3.13's
# schedule change (WARMDOWN_RATIO 0.5 -> 0.62, MATRIX_LR 0.0225 -> 0.021), measured at
# IDENTICAL bytes/FLOPs/params on this exact architecture: val_bpb 1.04955 -> 1.049082
# (+0.00047 of margin) with decode TV also improving 0.0267 -> 0.0224. Credit 2 is the trunk's
# existing 0.00036 of margin. Combined budget ~0.0009 bpb. Node 1.7's clean single-variable
# read prices window slots at ~0.0034 bpb per 64 ring slots at these sizes, i.e. ~0.000053 per
# slot. A 32-slot cut (160/224 -> 144/208) would cost ~0.0017 -- roughly TWICE the budget -- so
# it is not taken. A 16-slot cut costs ~0.00085, i.e. exactly the whole budget, leaving no
# margin against run-to-run noise (~0.0001-0.0002). This node takes a 12-slot cut
# (SHORT 160 -> 154, MED 224 -> 218; rings [155,219] = 374 slots, 12 fewer than 386): priced at
# ~0.00064 it leaves ~0.00028 of projected margin, and window price is reportedly super-linear
# below ~160/224 (node 1.6), which argues for the conservative rung. Projected val_bpb
# ~1.04972 (< 1.05) at analytic 100,674 B, ~100,708 B measured (-3.1% vs the trunk's 103,936).
# Node 3.22 re-prices this exact conversion against the slot price that node 3.21 MEASURED.
# Node 3.20 measured a clean, single-variable bpb credit from UNEMBEDDING_LR 0.0020 -> 0.0030 at
# bit-identical bytes/FLOPs/params/throughput: val_bpb 1.049939 -> 1.048177 (-0.00176), so the
# gate margin against 1.05 becomes 0.00182. Node 3.21 tried to convert that margin with a
# 24-slot cut (154/218 -> 142/206) and landed at val_bpb 1.050259, INELIGIBLE by 0.00026; the
# useful thing it produced is a MEASURED slot price in this regime: (1.050259 - 1.048177) / 24
# = 0.0000867 bpb per ring slot, 1.64x the 0.000053 linear estimate taken from node 1.7.
# Arithmetic for the eligible rung: budget 0.00182 bpb, minus ~0.0002 for the stacked-credit
# under-delivery that node 3.14 measured, leaves ~0.0016 spendable => 0.0016 / 0.0000867 ~= 18
# slots as the absolute ceiling. This node takes 12 slots (SHORT 154 -> 148, MED 218 -> 212,
# rings [149,213] = 362 slots, 12 fewer than 374): priced at 12 x 0.0000867 = 0.00104, which
# projects val_bpb ~1.04922 and leaves ~0.0006 of real margin against run-to-run noise
# (~0.0001-0.0002). The 16-slot option (0.00139, ~0.0002 margin) is deliberately NOT taken:
# this is the project's final launch and an ineligible smaller number is worth nothing.
# Analytic bytes: 362 x 269.18 + ~500 B allocator rounding ~= 97,960 B (-2.9% vs 100,864).
# Node 3.24 (final consolidation) takes a SMALL further cut against node 3.22's realized slot
# price. Trunk state: val_bpb 1.049286, so margin to the 1.05 gate is 0.000714. Node 3.22's
# realized price over its 12-slot conversion is 0.0000924 bpb/slot. Arithmetic:
#   8-slot cut  -> 8 x 0.0000924 = 0.00074 bpb, i.e. the ENTIRE margin (nothing left for the
#                  ~0.0002 stacked-credit under-delivery node 3.14 measured, and nothing for
#                  the unmeasured-magnitude EMBEDDING_LR move below, whose sign is favourable
#                  but whose size is a guess).
#   4-slot cut  -> 4 x 0.0000924 = 0.00037 bpb, projecting val_bpb ~1.04966 and leaving
#                  ~0.00034 of real margin plus whatever the LR move returns.
# This is the project's FINAL launch and an ineligible smaller number is worth nothing, so the
# 4-slot rung is taken: SHORT 148 -> 146, MED 212 -> 210, so rings [147,211] = 358 slots
# (4 fewer than 362). Analytic bytes
# 358 x 269.18 + ~400 B allocator rounding ~= 96,750 B (-1.1% vs 97,792).
SHORT_WINDOW = 146
MED_WINDOW = 210
# Node 3.10: the parameter budget freed by cutting VE_PER_PRODUCER 6 -> 1 (-10.5M params) is
# reinvested in MLP width, 5x -> 6x (+4.2M params, +25.2M FLOPs/token -> 212.3M against the
# 239.08M ceiling). 6x is chosen over 7x deliberately: 7x would land at 237.5M, i.e. spend
# essentially all the headroom, and FLOPs are charged TWICE -- against the ceiling and against
# the tokens that fit in the fixed 600 s clock -- so the last rung would very likely cost more
# in tokens trained than it buys in capacity. Net params 46.40M -> 40.11M, well under the
# 50.33M ceiling, and peak_vram has ~24 GB of slack since node 3.9 halved the microbatch.
MLP_MULT = 6

# Optimization
#
# The substrate's recipe was tuned for a much larger-per-step regime (4 KV heads, 1024
# windows, 4x MLP). For THIS 46M-param, 8-layer, all-local model a 2**19-token global batch
# is far above the critical batch size: under the fixed 600 s clock it buys nothing but
# costs half the optimizer steps (~885 instead of ~1770). Halving it to 2**18 (= exactly one
# DEVICE_BATCH_SIZE x MAX_SEQ_LEN microbatch, so grad_accum_steps becomes 1 and training
# throughput / peak_vram are unchanged) doubles the steps taken on the same token stream.
# LRs are rescaled for the smaller batch: sqrt(1/2) = 0.707 for the Adam groups (the usual
# small-batch scaling, which keeps per-token noise-to-signal constant), and a milder 0.75x
# for Muon, whose orthogonalized update is far less batch-size sensitive but now gets twice
# as many applications.
#
# Node 3.9 takes the SECOND rung: 2**18 -> 2**17. grad_accum_steps is already 1, so this
# means DEVICE_BATCH_SIZE 128 -> 64 (a 64x2048 = 131k-token microbatch, still far past the
# point where an A100/H100 GEMM is compute-bound at n_embd=512, so tok/sec should be
# essentially flat), doubling the optimizer steps again inside the same 600 s clock. The
# reported peak_vram is set by the FROZEN 128x2048 validation forward in prepare.py, so a
# smaller TRAINING microbatch cannot raise it (it may lower training-time peak, which is
# fine). LRs are rescaled by the same factors that worked at the last rung.
TOTAL_BATCH_SIZE = 2**17 # ~131K tokens per optimizer step (was 2**18)
EMBEDDING_LR = 0.30     # learning rate for token embeddings (Adam); 0.42 * sqrt(1/2).
                        # Node 3.24 attempt 1 MEASURED the downward move 0.30 -> 0.26: it cost
                        # +0.00031 bpb (1.049286 -> 1.049592) AND blew nopref decode TV from
                        # 0.03026 to 0.07681 (a single-step outlier; the mean stayed 0.0081),
                        # i.e. 3.23's "higher embedding LR lowers TV" holds in both directions
                        # and the lower rung is inadmissible. 0.30 is the optimum: the gradient
                        # is now measured on BOTH sides (0.42 cost +0.00052, 0.26 cost +0.00031).
UNEMBEDDING_LR = 0.0030 # learning rate for lm_head (Adam); node 3.20 measured -0.00176 bpb here
                        # at bit-identical bytes/FLOPs/params (was 0.0020). The gradient was still
                        # positive at this rung, but 0.0035 is left untested: the window cut below
                        # is sized against the MEASURED credit only.
MATRIX_LR = 0.021       # learning rate for matrix parameters (Muon); node 3.13, was 0.0225
SCALAR_LR = 0.25        # learning rate for per-layer scalars (Adam); 0.35 * sqrt(1/2)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.62   # fraction of time budget for LR warmdown (node 3.13, was 0.5)
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# Model size
DEPTH = 8               # number of transformer layers
DEVICE_BATCH_SIZE = 64  # per-device batch size; = TOTAL_BATCH_SIZE/MAX_SEQ_LEN, grad_accum 1

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
        n_layer=depth, n_head=num_heads, n_kv_head=N_KV_HEAD, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
        short_window=SHORT_WINDOW,
        med_window=MED_WINDOW,
        mlp_mult=MLP_MULT,
    )

config = build_model_config(DEPTH)
print(f"Model config: {asdict(config)}")
_kv_slots = (config.n_layer + CLA_GROUP - 1) // CLA_GROUP
_head_dim = config.n_embd // config.n_head
# Right-sized rings: a slot with window W needs W+1 physical positions, not MAX_SEQ_LEN.
_kv_rings = [min(_w[0] + 1, MAX_SEQ_LEN)
             for _w in [GPT._compute_window_sizes(None, config)[s * CLA_GROUP]
                        for s in range(_kv_slots)]]
_kv_bytes = (2 * sum(_kv_rings) * config.n_kv_head * _head_dim                # int8 payload
             + 2 * sum(_kv_rings) * config.n_kv_head                          # bf16 k scales
             * (_head_dim // KV_QUANT_GROUP_K)
             + 2 * sum(_kv_rings) * config.n_kv_head                          # bf16 v scales
             * (_head_dim // KV_QUANT_GROUP_V)
             + 2 * (2 + max(_kv_rings)))                                      # aux, in-arena
print(f"CLA group {CLA_GROUP}: {_kv_slots} distinct KV caches for {config.n_layer} layers")
print(f"KV ring lengths (max_len={MAX_SEQ_LEN}): {_kv_rings}")
print(f"Analytic KV cache storage (batch=1, max_len={MAX_SEQ_LEN}): {_kv_bytes:,} B")

with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()

param_counts = count_params(model)
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
print("CLA layout: " + ", ".join(
    f"L{i}{'(kv)' if kv_producer(i) else f'->slot{kv_slot(i)}'}"
    f"{'+ve x%d' % len(model.ve_keys[i]) if model.ve_keys[i] else ''}"
    f"@{model.window_sizes[i][0]}" for i in range(config.n_layer)))
print(f"MLP hidden: {model.transformer.h[0].mlp.c_fc.out_features} "
      f"({MLP_MULT}x n_embd), ve tables: {len(model.value_embeds)}")
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
