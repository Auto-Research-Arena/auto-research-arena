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

# The decode-attention sweep below compiles `_fused_attn_torch` once per
# (cache length, window, write-site) combination, which is more variants than dynamo's
# default cache_size_limit of 8: past the limit it stops compiling that code object
# ENTIRELY and the step runs as ~40 eager ops (measured: 238 -> 357 us/step on a cache
# length the sweep had not already compiled). Raised so any cache length the instrument
# asks for still gets a compiled step. Node 1.13 adds one graph per (split-K combination x
# reduction extent), so the ceiling is raised again: ~56 mono graphs are live by the end of
# the sweep, and a variant that silently stopped compiling would be timed as eager aten ops.
torch_dynamo_cache_limit = 256
import torch._dynamo
torch._dynamo.config.cache_size_limit = torch_dynamo_cache_limit
torch._dynamo.config.accumulated_cache_size_limit = 4 * torch_dynamo_cache_limit

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
    ffn_dim: int = 3072
    window_pattern: str = "SSSL"
    # Explicit value-embedding placement. None = the original alternating rule. A tuple of
    # layer indices lets the offline budget search decide HOW MANY FLOP-free value-embedding
    # tables to buy (they are excluded by name from the FLOPs proxy, so at low depth, where
    # the parameter ceiling binds and the FLOP ceiling does not, they are the only way to
    # spend leftover parameters at zero FLOP cost).
    ve_layers: tuple = None


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer, ve_layers=None):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    if ve_layers is not None:
        return layer_idx in ve_layers
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


# ---------------------------------------------------------------------------
# Fused decode helpers.
#
# The width-1 decode step is bound by the SERIAL COUNT of tiny device ops inside the
# captured CUDA graph, not by bandwidth (~60 MB of weight traffic per step). These four
# helpers are plain `torch.compile(dynamic=False)` (NOT reduce-overhead, so inductor never
# takes over graph capture or input pools) and collapse the per-layer elementwise chain
# into a handful of kernels. `fa3.flash_attn_with_kvcache` stays eager and OUTSIDE every
# compiled region because it appends k/v to the cache in place.
#
# They take pre-cast bf16 weight tensors as arguments: an fp32 parameter inside a compiled
# region would have inductor emit a cast kernel on EVERY call (autocast's weight cache only
# covers the eager dispatcher), which is exactly the kind of op we are removing.
# ---------------------------------------------------------------------------

@torch.compile(dynamic=False)
def _fused_prologue(idx, wte_w, cos_table, sin_table, seq):
    """Token embedding + norm, and the layer-INVARIANT rotary gather done once per step."""
    x = norm(F.embedding(idx, wte_w))
    pos = seq.to(torch.int64)
    return x, cos_table.index_select(1, pos), sin_table.index_select(1, pos)


def _skv(x, W, splits, out_dtype=None):
    """`y = x @ W.T` at width 1, as a SPLIT-K REDUCTION instead of a cuBLAS matmul.

    At m=1 cuBLAS picks a low-program-count `gemvx` tiling: measured, out-proj (1024x1024)
    spends 6.44us moving 2.1 MB = 326 GB/s, ~5x below the device, and the time did NOT move
    when node 2.7 halved the FFN matrices (6.51 -> 6.44 us) -- i.e. these kernels are
    occupancy/latency-bound, not byte-bound.

    Reshaping K into (splits, k_chunk) and reducing turns the projection into an inductor
    REDUCTION kernel whose output numel is N*splits instead of N, so the grid is a few
    thousand programs on the same bytes, and the adjacent norm / residual / relu-square work
    rides along as prologue/epilogue instead of bracketing an opaque library call. This is the
    same substitution node 1.10 measured working for the attention matmuls; it is NOT
    inductor's matmul TEMPLATE (node 1.5), which reproduces cuBLAS's tiling and therefore its
    occupancy problem.

    W is kept in its natural row-major (N, K) layout, so the reduction over `k_chunk` reads
    contiguously (node 1.10: the wrong layout was 1.8x SLOWER than the matmul).

    `splits == 0` is the cuBLAS path. `splits > 0` is the two-stage form (fp32 partials over
    k_chunk, then a reduction over the split axis); `splits < 0` is the one-shot form, a single
    reduction over both axes. Partials are always accumulated in fp32, which is what cuBLAS's
    bf16 gemv does too, so the result differs only by reduction order.
    """
    if out_dtype is None:
        out_dtype = x.dtype
    if splits == 0:
        y = F.linear(x, W)
        return y if y.dtype == out_dtype else y.to(out_dtype)
    n_split = abs(splits)
    N, K = W.shape
    kc = K // n_split
    assert kc * n_split == K, f"K={K} not divisible by splits={n_split}"
    lead = x.shape[:-1]
    xk = x.reshape(-1, 1, n_split, kc).float()
    wk = W.view(1, N, n_split, kc).float()
    if splits > 0:
        y = (wk * xk).sum(-1).sum(-1)
    else:
        y = (wk * xk).sum((-1, -2))
    return y.to(out_dtype).reshape(*lead, N)


def _qkv_gate_split(h, qkv_w, n_head, n_kv_head, head_dim, splits=0):
    """ONE GEMV producing q,k,v AND the value-embedding gate pre-activation.

    `_build_decode_weights` appends the `ve_gate` Linear's rows to the derived fused-QKV
    matrix, zero-padded from its 32 input channels to the full n_embd width. The gate reads
    the FIRST 32 channels of exactly the `h` the qkv projection reads, and the padded columns
    multiply by exact zeros, so the arithmetic is unchanged -- but the tiny 32->n_kv_head
    matmul kernel is gone from the serial decode chain. The parameters themselves are
    untouched (the derived matrix is a plain attribute, not a Parameter), so `forward`,
    the optimizer and `count_params` see the same model.
    """
    B, T = h.shape[0], h.shape[1]
    nqkv = (n_head + 2 * n_kv_head) * head_dim
    p = _skv(h, qkv_w, splits)
    # reshape, not view: `p[..., :nqkv]` is a last-dim slice of a wider row when the gate rows
    # are appended. The split is of a stride-1 dim so it IS a view here, but reshape cannot
    # raise and inductor lowers it to the same indexing.
    qkv = p[..., :nqkv].reshape(B, T, n_head + 2 * n_kv_head, head_dim)
    gate_pre = p[..., nqkv:] if p.shape[-1] > nqkv else None
    return qkv, gate_pre


@torch.compile(dynamic=False)
def _fused_pre_attn(x, x0, resid_l, x0_l, qkv_w, ve_w, idx, cos, sin,
                    n_head, n_kv_head, head_dim):
    """resid/x0 mix, norm, ONE fused qkv+gate GEMV, value-embedding gate, rotary, qk-norm."""
    x = resid_l * x + x0_l * x0
    h = norm(x)
    B, T = h.shape[0], h.shape[1]
    qkv, gate_pre = _qkv_gate_split(h, qkv_w, n_head, n_kv_head, head_dim)
    q = qkv[:, :, :n_head]
    k = qkv[:, :, n_head:n_head + n_kv_head]
    v = qkv[:, :, n_head + n_kv_head:]
    if ve_w is not None:
        ve = F.embedding(idx, ve_w).view(B, T, n_kv_head, head_dim)
        gate = 2 * torch.sigmoid(gate_pre)
        v = v + gate.unsqueeze(-1) * ve
    else:
        v = v.contiguous()
    q = norm(apply_rotary_emb(q, cos, sin))
    k = norm(apply_rotary_emb(k, cos, sin))
    return x, q, k, v


@torch.compile(dynamic=False)
def _fused_post_attn(x, y, proj_w, fc_w, mlp_proj_w):
    """attention out-projection + residual, then the whole MLP + residual."""
    B, T = x.shape[0], x.shape[1]
    x = x + F.linear(y.reshape(B, T, -1), proj_w)
    h = F.relu(F.linear(norm(x), fc_w)).square()
    return x + F.linear(h, mlp_proj_w)


@torch.compile(dynamic=False)
def _fused_attn_torch(x, q, k, v, kc, vc, seq, ar, proj_w, fc_w, mlp_proj_w,
                      window, n_head, n_kv_head, write):
    """Plain-torch width-1 attention over the FULL preallocated cache, + post-attn.

    Replaces `fa3.flash_attn_with_kvcache` on the step path. At one query over <=2049 keys
    the 514x128 reduction is trivially parallel, so a pair of tiny matmuls plus a softmax
    beats a decode kernel that FA3 confines to a single CUDA block.

    `write=True` does the k/v scatter here (one op, fused by inductor into the mask/score
    epilogue); `write=False` expects the caller to have done it eagerly. The mask is built
    on device from `seq`, so shapes are constant and there is no host sync:
    key j is visible iff  seq - window <= j <= seq, matching FA3's (window, 0) local mask
    exactly (on the 514-long no-prefill shape the window term is vacuous; on the 2049-long
    shape it is not). Softmax is accumulated in fp32.
    """
    B = x.shape[0]
    pos = seq.to(torch.int64)
    if write:
        kc.index_copy_(1, pos, k)
        vc.index_copy_(1, pos, v)
    L = ar.shape[0]
    p4 = pos.view(B, 1, 1, 1)
    a4 = ar.view(1, 1, 1, L)
    mask = a4 <= p4
    if window > 0:
        mask = mask & (a4 >= p4 - window)
    qh = q.transpose(1, 2)                                  # B, n_head, 1, hd
    kh = kc.transpose(1, 2)                                 # B, n_kv, L, hd
    vh = vc.transpose(1, 2)
    if n_kv_head != n_head:
        r = n_head // n_kv_head
        kh = kh.repeat_interleave(r, dim=1)
        vh = vh.repeat_interleave(r, dim=1)
    scale = qh.shape[-1] ** -0.5
    scores = torch.matmul(qh, kh.transpose(-1, -2)).float() * scale
    scores = torch.where(mask, scores, torch.full_like(scores, float("-inf")))
    p = torch.softmax(scores, dim=-1).to(vh.dtype)
    y = torch.matmul(p, vh).transpose(1, 2)                 # B, 1, n_head, hd
    x = x + F.linear(y.reshape(B, 1, -1), proj_w)
    h = F.relu(F.linear(norm(x), fc_w)).square()
    return x + F.linear(h, mlp_proj_w)


@torch.compile(dynamic=False)
def _fused_decode_step_mono(idx, wte_w, cos_t, sin_t, seq, ar, lm_w, resid_l, x0_l,
                            qkv_ws, ve_ws, proj_ws, fc_ws, mlp_proj_ws, kcs, vcs,
                            windows, n_head, n_kv_head, head_dim, head_major=False,
                            attn_reduce=False, extent=0, sk=(0, 0, 0, 0, 0)):
    """THE ENTIRE width-1 decode step as ONE compiled region.

    The short-cache (no-prefill) regime uses the plain-torch masked attention, which contains
    no opaque custom op, so nothing on this path has to stay eager: embed gather, rotary
    gather, per-layer resid/x0 mix, norm, fused qkv+gate GEMV, VE gate, rotary, qk-norm, k/v
    scatter, fp32 masked softmax, out-proj, MLP, final norm, lm_head and softcap all live in
    a single inductor graph. Splitting the same arithmetic across 2 regions per layer plus a
    prologue and a tail made inductor materialise every region boundary separately and leaked
    ~13 elementwise kernels per layer; at m=1 each kernel is a ~1.7us hardware floor, so the
    only lever is their COUNT. Arithmetic is op-for-op identical to the split version.

    `extent` (a python int, so dynamo bakes it in and every variant is its own graph) is how
    MANY cache slots the reduction/mask touches. The mask already kills every slot above the
    position, and a killed slot contributes exp(-inf) = exactly 0.0 to the softmax and
    0.0 * v to the output, so truncating the extent to any bound ABOVE the current position
    computes the same masked result while reading proportionally less cache and building a
    proportionally smaller mask. The caller is responsible for picking an extent that covers
    the position; `extent <= 0` means "the whole cache", the old behaviour.

    `sk` is a 5-tuple of python ints (also baked in per variant) selecting the SPLIT-K
    REDUCTION form of each weight projection -- (fused qkv+gate, out-proj, c_fc, c_proj,
    lm_head) -- see `_skv`. 0 keeps cuBLAS for that projection, so every combination is
    reachable and the sweep below can pick per-projection winners.
    """
    x = norm(F.embedding(idx, wte_w))
    pos = seq.to(torch.int64)
    cos = cos_t.index_select(1, pos)
    sin = sin_t.index_select(1, pos)
    x0 = x
    B = x.shape[0]
    full_L = ar.shape[0]
    L = full_L if extent <= 0 or extent >= full_L else extent
    if L != full_L:
        ar = ar[:L]
    p4 = pos.view(B, 1, 1, 1)
    a4 = ar.view(1, 1, 1, L)
    base_mask = a4 <= p4
    scale = head_dim ** -0.5
    for i in range(len(qkv_ws)):
        x = resid_l[i] * x + x0_l[i] * x0
        h = norm(x)
        qkv, gate_pre = _qkv_gate_split(h, qkv_ws[i], n_head, n_kv_head, head_dim, sk[0])
        q = qkv[:, :, :n_head]
        k = qkv[:, :, n_head:n_head + n_kv_head]
        v = qkv[:, :, n_head + n_kv_head:]
        if ve_ws[i] is not None:
            ve = F.embedding(idx, ve_ws[i]).view(B, 1, n_kv_head, head_dim)
            gate = 2 * torch.sigmoid(gate_pre)
            v = v + gate.unsqueeze(-1) * ve
        else:
            v = v.contiguous()
        q = norm(apply_rotary_emb(q, cos, sin))
        k = norm(apply_rotary_emb(k, cos, sin))
        kc, vc = kcs[i], vcs[i]
        window = windows[i]
        mask = base_mask if window <= 0 else (base_mask & (a4 >= p4 - window))
        qh = q.transpose(1, 2)
        if head_major:
            # Head-major cache (B, n_kv, max_len, head_dim): each head's K and V is a
            # CONTIGUOUS (max_len, head_dim) matrix, so the per-step score matmul is
            # q(1,hd) x K^T(hd,L) with lda = head_dim (the `gemv2T` shape) and the output
            # matmul is P(1,L) x V(L,hd) with lda = head_dim, instead of the strided
            # views the (B, max_len, n_kv, hd) FA3 layout forces on cuBLAS. Same
            # arithmetic; only the kernel cuBLAS selects changes.
            kc.index_copy_(2, pos, k.transpose(1, 2))
            vc.index_copy_(2, pos, v.transpose(1, 2))
            kh, vh = (kc, vc) if L == full_L else (kc[:, :, :L], vc[:, :, :L])
        else:
            kc.index_copy_(1, pos, k)
            vc.index_copy_(1, pos, v)
            kh = (kc if L == full_L else kc[:, :L]).transpose(1, 2)
            vh = (vc if L == full_L else vc[:, :L]).transpose(1, 2)
        gqa_r = n_head // n_kv_head
        if attn_reduce:
            # The SAME arithmetic as the two bmms, written as explicit broadcast
            # multiply + reduction so inductor owns both: at m=1 a "matmul" over
            # 8x514x128 is pure bandwidth (1 MB of cache per matmul), and cuBLAS spends
            # 9.5us (gemv2T) + 5.8us (cutlass align2) per layer on it because its m=1
            # kernels are latency- not bandwidth-bound. A triton reduction reads the same
            # 1 MB once and lets the mask/scale ride along as an epilogue.
            #
            # GQA is FREE in this form: the kv-head axis is BROADCAST over the group of
            # query heads that share it (an extra size-gqa_r axis), so no
            # `repeat_interleave` copy of the cache is ever materialised -- which is
            # exactly the cost that made node 2.6 reject GQA. Head order matches
            # repeat_interleave's (kv0 x r, kv1 x r, ...) because the group axis is the
            # minor one of the (n_kv_head, gqa_r) split of n_head.
            if gqa_r == 1:
                s = (qh.float() * kh.float()).sum(-1) * scale       # B, n_head, L
            else:
                qg = qh.reshape(B, n_kv_head, gqa_r, 1, head_dim)
                s = (qg.float() * kh.unsqueeze(2).float()).sum(-1).reshape(B, n_head, L) * scale
            s = torch.where(mask.view(B, 1, L), s, torch.full_like(s, float("-inf")))
            p = torch.softmax(s, dim=-1)
            if gqa_r == 1:
                y = (p.unsqueeze(-1) * vh.float()).sum(-2).to(x.dtype)   # B, n_head, hd
            else:
                pg = p.reshape(B, n_kv_head, gqa_r, L, 1)
                y = (pg * vh.unsqueeze(2).float()).sum(-2).to(x.dtype)   # B, n_kv, r, hd
            x = x + _skv(y.reshape(B, 1, -1), proj_ws[i], sk[1])
        else:
            if gqa_r != 1:
                kh = kh.repeat_interleave(gqa_r, dim=1)
                vh = vh.repeat_interleave(gqa_r, dim=1)
            scores = torch.matmul(qh, kh.transpose(-1, -2)).float() * scale
            scores = torch.where(mask, scores, torch.full_like(scores, float("-inf")))
            p = torch.softmax(scores, dim=-1).to(vh.dtype)
            y = torch.matmul(p, vh).transpose(1, 2)
            x = x + _skv(y.reshape(B, 1, -1), proj_ws[i], sk[1])
        hm = F.relu(_skv(norm(x), fc_ws[i], sk[2])).square()
        x = x + _skv(hm, mlp_proj_ws[i], sk[3])
    logits = _skv(norm(x), lm_w, sk[4], torch.float32)
    return 15 * torch.tanh(logits / 15)


# The width-1 attention configuration. `mode` is "fa3" or "torch"; `num_splits` is handed
# straight to FA3 (0 = its own heuristic); `nowindow` drops the explicit window_size, which
# is only legal when the cache is shorter than the window; `write` picks where the torch
# path's k/v scatter happens. Selected empirically in the diagnostics below, per cache-length
# regime, and frozen onto each state at init_decode_state time.
# `mono` routes the whole step through `_fused_decode_step_mono` (one compiled region).
# `hm` lays the k/v caches out head-major, (B, n_kv_head, max_len, head_dim), so the two
# per-layer attention matmuls are contiguous matrix-vector products rather than strided
# views. Only legal on the plain-torch path (FA3 requires the seq-major layout), so it is
# gated on `mode == "torch"` at init_decode_state time.
# `pad8` rounds the ALLOCATED slot count up to a multiple of 8 (514 -> 520) so the cache's
# leading dimension is alignment-friendly to the kernel that reads it. `blk` restricts the
# reduction to `ceil((pos+1)/blk)*blk` slots via one captured graph per block count.
_CFG_FA3_PINNED = {"mode": "fa3", "num_splits": 1, "nowindow": False, "write": True,
                   "mono": False, "hm": False, "pad8": False, "blk": 0}
_DECODE_CFG_SHORT = dict(_CFG_FA3_PINNED)
_DECODE_CFG_LONG = dict(_CFG_FA3_PINNED)
_DECODE_CFG_FORCE = None
_SHORT_REGIME_MAX_LEN = 1024
_MAX_EXTENTS = 4          # ceiling on captured graphs per state (see init_decode_state)


def _cfg_for_max_len(max_len):
    if _DECODE_CFG_FORCE is not None:
        return _DECODE_CFG_FORCE
    return _DECODE_CFG_SHORT if max_len <= _SHORT_REGIME_MAX_LEN else _DECODE_CFG_LONG


def _extent_for_pos(state):
    """Which of the state's FIXED reduction extents covers `state["pos"]`.

    Pure cost selection: every extent computes the same masked result (positions above the
    real device-side `seq` are masked to exactly -inf either way), so an over-large answer is
    merely slow and the last extent is always the whole cache. 0 means "the whole cache".
    """
    ex = state.get("extents")
    if not ex or len(ex) == 1:
        return 0
    k = state.get("pos", 0) // state["blk"]
    return ex[min(max(k, 0), len(ex) - 1)]


@torch.compile(dynamic=False)
def _fused_tail(x, lm_w):
    logits = F.linear(norm(x), lm_w).float()
    return 15 * torch.tanh(logits / 15)


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
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer, config.ve_layers) else None

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
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.ffn_dim, bias=False)
        self.c_proj = nn.Linear(config.ffn_dim, config.n_embd, bias=False)

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
            for i in range(config.n_layer) if has_ve(i, config.n_layer, config.ve_layers)
        })
        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        # Fused bf16 decode views of the same parameters, built lazily by
        # _build_decode_weights(). Plain attributes: not parameters, not buffers.
        self._dw = None
        self._lm_w = None
        self._ar = None

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
        """
        assert max_len <= self.cos.size(1), f"max_len {max_len} exceeds rotary table"
        self._build_decode_weights()
        cfg = self.config
        head_dim = cfg.n_embd // cfg.n_head
        dev = self.transformer.wte.weight.device
        kw = {"dtype": torch.bfloat16, "device": dev}
        # ALLOCATE a multiple-of-8 number of slots (514 -> 520). The cache leading dim is
        # what cuBLAS/cutlass inspects for alignment when it reads a (max_len, head_dim)
        # matrix, and 514 is not a multiple of 8, which is what forced the slower
        # `..._tn_align2` output kernel. The extra slots are pure padding: the mask is driven
        # by the real position (`a4 <= p4`), so they are unreachable, and the reduction extent
        # below is what decides how many of them are even read.
        acfg = _cfg_for_max_len(max_len)
        alloc_len = max_len if not acfg.get("pad8") else (max_len + 7) // 8 * 8
        assert alloc_len <= self.cos.size(1), f"padded {alloc_len} exceeds rotary table"
        # Key index vector for the plain-torch mask. Cached ON THE MODEL and keyed by length
        # so the cache-bytes probe (which builds a state twice and reads the delta) never
        # charges it: the second build allocates nothing.
        if self._ar is None:
            self._ar = {}
        if alloc_len not in self._ar:
            self._ar[alloc_len] = torch.arange(alloc_len, dtype=torch.int64, device=dev)
        # The cache LAYOUT is frozen onto the state alongside the attention config: the
        # plain-torch step reads the cache with ordinary matmuls, so on that path the
        # layout is free and head-major keeps both attention matmuls contiguous. FA3 (the
        # long-shape winner) requires (B, max_len, n_kv_head, head_dim), so it keeps it.
        # ONE layout per state either way -- never two copies of the cache.
        # Head-major is confined to the SHORT regime: that is where the plain-torch step
        # wins, and the long regime's prefill is FA3's (seq-major) by construction.
        hm = (bool(acfg.get("hm")) and acfg.get("mode") == "torch"
              and bool(acfg.get("mono")) and max_len <= _SHORT_REGIME_MAX_LEN)
        shape = ((batch, cfg.n_kv_head, alloc_len, head_dim) if hm else
                 (batch, alloc_len, cfg.n_kv_head, head_dim))
        # The FIXED set of reduction extents this state may run, one captured graph each.
        # extents[k] covers every position < extents[k]; the last one is the whole cache, so
        # every position is covered by construction. Every extent computes the SAME masked
        # result (see `_fused_decode_step_mono`), so which one runs is a pure cost choice.
        blk = int(acfg.get("blk") or 0) if acfg.get("mono") else 0
        if blk > 0 and blk < alloc_len:
            while True:
                extents = [min(alloc_len, (k + 1) * blk)
                           for k in range((alloc_len + blk - 1) // blk)]
                # A final block only a sliver wider than the previous boundary (520 vs 512)
                # would buy ONE position its own graph: fold it into the previous block.
                # `_extent_for_pos` clamps, and a clamp can only ENLARGE the extent, which is
                # always safe (an extent above the position computes the same thing).
                if len(extents) > 1 and extents[-1] - extents[-2] < blk // 2:
                    extents.pop(-2)
                # At most _MAX_EXTENTS graphs per state: each is its own inductor compile and
                # its own CUDA-graph pool, and the win from a finer block flattens out fast.
                if len(extents) <= _MAX_EXTENTS or blk >= alloc_len:
                    break
                blk *= 2
            extents = tuple(extents)
        else:
            blk, extents = 0, (alloc_len,)
        return {
            "max_len": max_len,
            "alloc_len": alloc_len,
            "graph_enabled": bool(graph),
            "acfg": acfg,
            "hm": hm,
            "extents": extents,
            "blk": blk,
            # Python-side mirror of `seq`, used for NOTHING but selecting which captured
            # graph replays (all of which compute the same result). `seq` remains the single
            # source of truth for every piece of arithmetic. Reset by reset_decode_state.
            "pos": 0,
            "ar": self._ar[alloc_len],
            "seq": torch.zeros(batch, dtype=torch.int32, device=dev),
            "kc": [torch.zeros(*shape, **kw) for _ in range(cfg.n_layer)],
            "vc": [torch.zeros(*shape, **kw) for _ in range(cfg.n_layer)],
        }

    def reset_decode_state(self, state):
        """Return `state` to position 0 so the same buffers serve another request.

        A fresh state per request would invalidate an address-bound CUDA graph. Stale bytes
        above the position are unreachable -- attention reads only `cache_seqlens` keys -- so
        zeroing them would be timed work serving never does.
        """
        state["seq"].zero_()
        state["pos"] = 0
        return state

    @torch.no_grad()
    def _build_decode_weights(self):
        """Cache the bf16 decode views of the SAME parameters `forward` uses.

        Three things happen here, once, off the timed path:
          * c_q/c_k/c_v are concatenated into one (n_head+2*n_kv_head)*head_dim x n_embd
            matrix, so a step issues 1 GEMV instead of 3. The `ve_gate` Linear (32 inputs ->
            n_kv_head) reads the first 32 channels of the SAME normalized h, so its rows are
            zero-padded to the full n_embd width and appended to that same matrix: one fewer
            matmul kernel in the serial chain at identical arithmetic (the padded columns
            multiply exact zeros). The parameters themselves are untouched -- `forward`, the
            optimizer and `count_params` still see exactly the Linears they always saw, so
            training is bit-identical to the reference and `num_params_total` cannot move.
          * every decode weight is pre-cast to bf16, which is precisely what autocast would
            produce; doing it here keeps cast kernels out of the compiled regions.
          * it is idempotent, so the cache-bytes probe (which builds a state twice and reads
            the allocation delta) never charges these bytes to the KV cache.
        """
        if self._dw is not None:
            return
        dw = []
        for i, block in enumerate(self.transformer.h):
            a = block.attn
            parts = [a.c_q.weight, a.c_k.weight, a.c_v.weight]
            if a.ve_gate is not None:
                gw = a.ve_gate.weight
                gpad = torch.zeros(gw.shape[0], self.config.n_embd,
                                   dtype=gw.dtype, device=gw.device)
                gpad[:, :a.ve_gate_channels] = gw
                parts.append(gpad)
            dw.append({
                "qkv": torch.cat(parts, 0).to(torch.bfloat16),
                "proj": a.c_proj.weight.to(torch.bfloat16),
                "fc": block.mlp.c_fc.weight.to(torch.bfloat16),
                "mlp_proj": block.mlp.c_proj.weight.to(torch.bfloat16),
                "ve": (self.value_embeds[str(i)].weight.to(torch.bfloat16)
                       if str(i) in self.value_embeds else None),
            })
        self._lm_w = self.lm_head.weight.to(torch.bfloat16)
        self._dw = dw
        # Per-tensor-kind lists, so the monolithic compiled step takes flat list args
        # (dynamo unrolls the layer loop over them) instead of a list of dicts.
        self._mono_w = {k: [d[k] for d in dw] for k in ("qkv", "ve", "proj", "fc", "mlp_proj")}

    def _decode_body(self, idx, state, prefill, extent=0):
        """`forward`'s body with the cache spliced in. `prefill` is a Python bool that is
        constant per variant, so the two branches are two graphs, not one per position.
        `extent` is the reduction bound (0 = whole cache), also constant per graph."""
        B, Tn = idx.size()
        dw = self._dw
        cfg = state.get("acfg", _CFG_FA3_PINNED)
        if not prefill and cfg.get("mono"):
            # ONE compiled region for the whole step. `windows` is a tuple of python ints
            # (constant for the life of a state, so dynamo bakes them in): -1 means the
            # window term is vacuous on this cache length, exactly as in the split path.
            a0 = self.transformer.h[0].attn
            windows = tuple(-1 if state["max_len"] <= w[0] else w[0] for w in self.window_sizes)
            # Per-projection split-K selection, baked in as python ints (one graph per combo).
            sk = tuple(int(cfg.get(k) or 0)
                       for k in ("sk_qkv", "sk_proj", "sk_fc", "sk_cproj", "sk_lm"))
            return _fused_decode_step_mono(
                idx, self.transformer.wte.weight, self.cos, self.sin,
                state["seq"], state["ar"], self._lm_w,
                self.resid_lambdas, self.x0_lambdas,
                self._mono_w["qkv"], self._mono_w["ve"], self._mono_w["proj"],
                self._mono_w["fc"], self._mono_w["mlp_proj"],
                state["kc"], state["vc"], windows,
                a0.n_head, a0.n_kv_head, a0.head_dim, bool(state.get("hm")),
                bool(cfg.get("red")), int(extent), sk)
        if prefill:
            cos, sin = self.cos[:, :Tn], self.sin[:, :Tn]
            x = norm(self.transformer.wte(idx))
        else:
            # Layer-invariant: the rotary gather used to run 8 times per step.
            x, cos, sin = _fused_prologue(idx, self.transformer.wte.weight,
                                          self.cos, self.sin, state["seq"])
        x0 = x
        for i, block in enumerate(self.transformer.h):
            attn = block.attn
            lw = dw[i]
            if prefill:
                x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
                h = norm(x)
                qkv, gate_pre = _qkv_gate_split(h, lw["qkv"], attn.n_head, attn.n_kv_head,
                                                attn.head_dim)
                q = qkv[:, :, :attn.n_head]
                k = qkv[:, :, attn.n_head:attn.n_head + attn.n_kv_head]
                v = qkv[:, :, attn.n_head + attn.n_kv_head:].contiguous()
                if lw["ve"] is not None:
                    ve = F.embedding(idx, lw["ve"]).view(B, Tn, attn.n_kv_head, attn.head_dim)
                    gate = 2 * torch.sigmoid(gate_pre)
                    v = v + gate.unsqueeze(-1) * ve
                q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
                q, k = norm(q), norm(k)
                kc, vc = state["kc"][i], state["vc"][i]
                if state.get("hm"):
                    # head-major state: FA3 still wants (B, T, n_kv, hd), and kc[:, :Tn]
                    # was only ever a copy of k, so feed k/v directly and fill the cache
                    # in the head-major layout the step path reads.
                    k = k.contiguous(); v = v.contiguous()
                    kc[:, :, :Tn] = k.transpose(1, 2)
                    vc[:, :, :Tn] = v.transpose(1, 2)
                    y = fa3.flash_attn_func(q, k, v, causal=True,
                                            window_size=self.window_sizes[i])
                else:
                    kc[:, :Tn] = k
                    vc[:, :Tn] = v
                    y = fa3.flash_attn_func(q, kc[:, :Tn], vc[:, :Tn], causal=True,
                                            window_size=self.window_sizes[i])
                x = x + F.linear(y.reshape(B, Tn, -1), lw["proj"])
                x = x + F.linear(F.relu(F.linear(norm(x), lw["fc"])).square(), lw["mlp_proj"])
            else:
                x, q, k, v = _fused_pre_attn(
                    x, x0, self.resid_lambdas[i], self.x0_lambdas[i],
                    lw["qkv"], lw["ve"], idx, cos, sin,
                    attn.n_head, attn.n_kv_head, attn.head_dim)
                # Appends k,v at cache_seqlens and attends over the whole preallocated
                # cache, so there is no `kc[:, start:end]` slice whose bounds depend on the
                # step number. `num_splits` used to be pinned to 1 because the op's fake
                # kernel refused to trace at the default 0 -- but this call is EAGER and
                # outside every compiled region, so nothing traces it any more and the split
                # count is free again. It is still a reduction-order choice, and the
                # agreement check in prepare.py has to pass with whatever is selected.
                # Kept EAGER and outside every compiled region: it mutates kc/vc in place.
                kc, vc = state["kc"][i], state["vc"][i]
                w = self.window_sizes[i]
                if cfg["mode"] == "torch":
                    if not cfg["write"]:
                        pos = state["seq"].to(torch.int64)
                        kc.index_copy_(1, pos, k)
                        vc.index_copy_(1, pos, v)
                    x = _fused_attn_torch(
                        x, q, k, v, kc, vc, state["seq"], state["ar"],
                        lw["proj"], lw["fc"], lw["mlp_proj"],
                        -1 if state["max_len"] <= w[0] else w[0],
                        attn.n_head, attn.n_kv_head, bool(cfg["write"]))
                    continue
                if cfg["nowindow"] and state["max_len"] <= w[0]:
                    w = (-1, -1)          # vacuous on this cache length: same math, no mask
                y = fa3.flash_attn_with_kvcache(q, kc, vc, k=k, v=v,
                                                cache_seqlens=state["seq"], causal=True,
                                                window_size=w,
                                                num_splits=cfg["num_splits"])
                x = _fused_post_attn(x, y, lw["proj"], lw["fc"], lw["mlp_proj"])

        if prefill:
            x = norm(x[:, -1:, :])
            softcap = 15
            logits = F.linear(x, self._lm_w).float()
            return softcap * torch.tanh(logits / softcap)
        return _fused_tail(x, self._lm_w)

    def decode_step(self, idx, state):
        """`logits, state = model.decode_step(idx, state)`, logits for the LAST position only.

        Width > 1 is the prefill and runs once per request; width 1 is the step, and it is
        the call the graph replays. `seq` is advanced on both paths, so the tensor is the
        single source of the position.
        """
        if idx.size(1) > 1:
            logits = self._decode_body(idx, state, prefill=True)
            state["seq"].add_(idx.size(1))
            state["pos"] = state.get("pos", 0) + idx.size(1)
            return logits, state
        if not state.get("graph_enabled", True):
            logits = self._decode_body(idx, state, prefill=False,
                                       extent=_extent_for_pos(state))
            state["seq"].add_(1)
            state["pos"] = state.get("pos", 0) + 1
            return logits, state
        # The graph lives ON the state it was captured against, so its lifetime is exactly
        # that state's and there is no key to get wrong. A state whose capture failed holds
        # an uncaptured object and runs eager forever, which is slow and correct.
        graph = state.get("graph")
        if graph is None:
            graph = state["graph"] = _GraphedDecodeStep(self, state)
        if not graph.captured:
            logits = self._decode_body(idx, state, prefill=False,
                                       extent=_extent_for_pos(state))
            state["seq"].add_(1)
            state["pos"] = state.get("pos", 0) + 1
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

    WARMUP_REPLAYS = 8
    LAST_STATUS = None

    def __init__(self, model, state):
        self.model = model
        self.state = state
        self.captured = False
        self.reason = ""
        self.static_idx = torch.zeros(1, 1, dtype=torch.int64, device=state["seq"].device)
        self.static_logits = None
        self.graph = None
        # One captured graph per reduction extent. `extents` is a fixed tuple decided at
        # init_decode_state time, so the SET of graphs is constant for the life of the state;
        # `replay` only picks which of them runs, and they all compute the same result.
        self.extents = tuple(state.get("extents") or (0,))
        if len(self.extents) == 1:
            self.extents = (0,)                 # 0 == whole cache, one graph, old behaviour
        self.graphs = []
        self.logits = []
        self.blk = int(state.get("blk") or 0)
        if state["seq"].numel() != 1:
            self.reason = f"batch {state['seq'].numel()} != 1"
            return
        try:
            self._capture()
            self.captured = True
        except Exception as exc:            # noqa: BLE001 -- recorded, then paid for in ms
            self.reason = f"{type(exc).__name__}: {exc}"
            self.graph, self.static_logits = None, None
            self.graphs, self.logits = [], []
        _GraphedDecodeStep.LAST_STATUS = (self.captured, self.reason)

    def _advance(self, extent=0):
        logits = self.model._decode_body(self.static_idx, self.state, prefill=False,
                                         extent=extent)
        self.state["seq"].add_(1)
        return logits

    def _capture(self):
        seq0 = self.state["seq"].clone()
        pos0 = self.state.get("pos", 0)
        for extent in self.extents:
            # Compile (and autotune) the fused helpers on the default stream FIRST: capturing
            # a graph while inductor is still generating kernels fails, and a failed capture
            # means eager forever.
            for _ in range(2):
                self.state["seq"].copy_(seq0)
                self._advance(extent)
            torch.cuda.synchronize()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(self.WARMUP_REPLAYS):
                    self.state["seq"].copy_(seq0)
                    self._advance(extent)
            torch.cuda.current_stream().wait_stream(stream)

            self.state["seq"].copy_(seq0)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                out = self._advance(extent)
            self.state["seq"].copy_(seq0)      # the capture itself executed one increment
            self.graphs.append(g)
            self.logits.append(out)
        self.state["pos"] = pos0
        # The whole-cache graph is the last one and is valid at every position.
        self.graph, self.static_logits = self.graphs[-1], self.logits[-1]

    def replay(self, idx):
        self.static_idx.copy_(idx)
        n = len(self.graphs)
        if n == 1:
            self.graphs[0].replay()
            self.state["pos"] = self.state.get("pos", 0) + 1
            return self.logits[0]
        p = self.state.get("pos", 0)
        k = p // self.blk
        if k >= n:
            k = n - 1
        self.graphs[k].replay()
        self.state["pos"] = p + 1
        return self.logits[k]


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

# Model architecture. Set EXPLICITLY: the depth-for-width rebalance taken to its floor,
# 2 serial layers x 1024 dim (8 heads x 128), after 4 x 768 / 6 x 640 / 8 x 512. At batch 1
# the decode step is latency-bound on the serial chain of device ops (measured: step time
# ~= #matmul kernels x ~6 us), so removing a layer removes 4 projections + 2 attention
# matmuls from that chain. The quality this costs is paid for out of node 3.1's 0.0444 bpb
# of slack under the 1.05 gate.
N_LAYER = 2
N_EMBD = 1024
HEAD_DIM = 128          # target head dimension for attention
# OFFLINE ARCHITECTURE SEARCH, RETARGETED (results/2.7-min-streamed-bytes/bytes_search.py).
#
# THE REGIME CHANGED. Nodes 1.9/1.10 removed the launch overhead and took attention off
# cuBLAS entirely; the surviving GEMM bucket is 84.0 us across exactly 9 projection kernels,
# which stream 2 bytes x (transformer_matrices + lm_head) = 67.1 MB per width-1 step, i.e.
# ~0.8-1.05 TB/s. Those kernels are now BANDWIDTH-BOUND ON WEIGHT BYTES, so the old tree
# lesson "widening is nearly free at batch 1" (true when every m=1 kernel cost a fixed ~6 us)
# is false here. The objective is therefore no longer "maximise num_params_total" (node 2.6)
# but:
#
#     MINIMISE   streamed_bytes = 2 bytes x (transformer_matrices + lm_head)
#     SUBJECT TO num_params_total ~= 50,332,176   and   analytic FLOPs <= 239,078,400
#
# `wte` and every value-embedding table are read as a SINGLE ROW (~2 KB) per decode step, so
# parameter mass parked in them is FREE at decode while the same mass in a matrix or in
# lm_head costs full bandwidth. At the parameter ceiling this closes to
#     streamed = 2 x (PARAM_CEIL - free_mass),   free_mass = 8192*n_embd + m*8192*kv_dim
# so the whole search collapses to MAXIMISING the free mass (m = number of VE tables).
#
# Consequences the sweep makes explicit:
#   * n_embd DOWN is COUNTERPRODUCTIVE: both wte and every VE table are 8192 x (width), so
#     narrowing shrinks the free pool and pushes mass back into matrices. 896 -> 56.6 MB,
#     768 -> 62.9 MB, 640 -> 69.2 MB, all WORSE than 1024's 50.3 MB.
#   * n_kv_head DOWN (GQA) is also COUNTERPRODUCTIVE here, for the same reason: a VE table is
#     8192 x n_kv_head x head_dim, so GQA shrinks the free pool faster than it shrinks the
#     qkv matrix (1024/kvh=4 -> 67.1 MB vs full MHA's 50.3 MB). The 2.6 objection to GQA
#     (repeat_interleave) is gone -- the reduction-form attention above now broadcasts over
#     kv groups -- but the axis loses on bytes anyway. Full MHA stays.
#   * ffn_dim DOWN is the whole win: it is the single biggest streamed item (2 layers x
#     2 x 1024 x 4096 x 2 B = 33.6 MB of the 67.1 MB). Halving 4096 -> 2048 frees 8.39M
#     parameters, which go into a SECOND value-embedding table (layer 0) at zero bytes and
#     zero FLOPs.
#   * n_layer stays 2 (depth 1 costs ~0.03 bpb we do not have; depth 3 is slower).
#
# CHOSEN: n_embd 1024, full MHA (8 x 128), ffn_dim 2048, VE tables on BOTH layers.
#   wte 8,388,608 + lm_head 8,388,608 + transformer_matrices 16,777,728
#   + value_embeds 16,777,216 (2 x 8192 x 1024) + scalars 4 = 50,332,164
#   = 100.000% of the 50,332,176 ceiling (12 spare -- tighter than the trunk's 268)
#   STREAMED = 2 x (16,777,728 + 8,388,608) = 50.33 MB, i.e. -25.0% vs the trunk's 67.11 MB
#   matrix share 33.3% of parameters (still a real transformer: ffn_dim = 2 x n_embd)
# The only row in the sweep with fewer bytes is 1152/ffn=426 (44.0 MB), rejected as
# degenerate: ffn_dim < n_embd/2 and a 25.0% matrix share.
FFN_DIM = 2048
N_KV_HEAD = 8            # full MHA: GQA would shrink the FLOP-free VE tables faster than it
                         # shrinks the qkv matrix, so it COSTS streamed bytes at this ceiling
# Moving 8.39M parameters out of the FFN and into a FLOP-free table also drops analytic FLOPs
# from 239,076,864 to 188,746,752, leaving 50.3M of FLOP headroom. FLOPs are irrelevant to
# the target metric, so spend the headroom on quality: "LL" gives layer 0 the full 2048-token
# span instead of 1024 (201,329,664 FLOPs, 84.2% of the ceiling). Span is parameter-free and
# cannot move the nopref metric -- that cache never exceeds 514 positions, below even the
# short window -- so this is free insurance against the val_bpb risk of the narrower FFN.
WINDOW_PATTERN = "LL"
VE_LAYERS = (0, 1)       # 16.78M FLOP-free, byte-free parameters (one table per layer)

# Optimization
# Node 3.1: the 600s clock is STEP-COUNT limited, not data limited. At TOTAL_BATCH_SIZE=2**19
# the run completed only ~846 optimizer updates while consuming ~0.70 epochs of the
# 631,241,817-token pool. 524,288 tokens/step is far above the critical batch size for a
# ~50M-parameter model, so per-update progress is already noise-saturated: halving the
# optimizer batch to 2**18 (= exactly one 128x2048 microbatch, grad_accum 1) buys ~2x the
# updates on the same data at the same wall clock and the same peak memory. None of these
# knobs appear anywhere in the decode path, so latency is unchanged by construction.
# LRs are deliberately left UNCHANGED: above the critical batch size the per-update step
# size is what it should be, and scaling it down by 1/sqrt(2) would give back exactly the
# progress the extra updates are meant to buy. A short LR warmup is added as cheap
# insurance against the noisier small-batch gradients in the first few updates.
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step (grad_accum = 1)
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.02     # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial
# Muon momentum warms 0.85 -> 0.95 over this many updates. Calibrated in step units for
# ~846 steps; the step count roughly doubles at TOTAL_BATCH_SIZE=2**18, so it is rescaled
# to keep the same fraction-of-run shape.
MUON_MOMENTUM_WARMUP_STEPS = 600

# Model size
DEPTH = N_LAYER         # number of transformer layers
DEVICE_BATCH_SIZE = 128  # per-device batch size (reduce if OOM)

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
    assert N_EMBD % HEAD_DIM == 0
    num_heads = N_EMBD // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=N_KV_HEAD, n_embd=N_EMBD,
        ffn_dim=FFN_DIM, window_pattern=WINDOW_PATTERN, ve_layers=VE_LAYERS,
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
    frac = min(step / MUON_MOMENTUM_WARMUP_STEPS, 1)
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
loss_trace = []  # (step, training_seconds, debiased smoothed loss, dt_ms) per update

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
    loss_trace.append((step, total_training_time, debiased_smooth_loss, dt * 1000))
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

# ---------------------------------------------------------------------------
# DIAGNOSTICS (never a METRICS_JSON line; nothing after the reporter)
# ---------------------------------------------------------------------------

def _training_diagnostics():
    print(f"[tdiag] TOTAL_BATCH_SIZE={TOTAL_BATCH_SIZE:,} "
          f"DEVICE_BATCH_SIZE={DEVICE_BATCH_SIZE} MAX_SEQ_LEN={MAX_SEQ_LEN} "
          f"grad_accum={grad_accum_steps}")
    print(f"[tdiag] optimizer updates: {step}   tokens consumed: {tokens_consumed:,} "
          f"(with repetition)   final epoch: {epoch}")
    print(f"[tdiag] epochs over the {DATA_BUDGET_TOKENS:,}-token pool: "
          f"{tokens_consumed / DATA_BUDGET_TOKENS:.3f}")
    print(f"[tdiag] counted training seconds: {total_training_time:.2f}s "
          f"(TIME_BUDGET {TIME_BUDGET}s); wall since start {t_end - t_start:.1f}s")
    if step > 11:
        timed = [e for e in loss_trace if e[0] > 10]
        ms = [e[3] for e in timed]
        ms_sorted = sorted(ms)
        med = ms_sorted[len(ms_sorted) // 2]
        print(f"[tdiag] ms/step: mean {sum(ms)/len(ms):.1f}  median {med:.1f}  "
              f"min {ms_sorted[0]:.1f}  max {ms_sorted[-1]:.1f}")
        print(f"[tdiag] tokens/sec (median step): {TOTAL_BATCH_SIZE/(med/1000):,.0f}")
    print("[tdiag] smoothed training-loss curve (10 evenly spaced points):")
    if loss_trace:
        n = len(loss_trace)
        for i in range(10):
            s, tsec, l, _ = loss_trace[min(n - 1, round(i * (n - 1) / 9))]
            print(f"[tdiag]   step {s:5d}  t {tsec:6.1f}s  lrm {get_lr_multiplier(min(tsec/TIME_BUDGET,1.0)):.3f}"
                  f"  smoothed_loss {l:.6f}")
        print(f"[tdiag] final smoothed training loss: {loss_trace[-1][2]:.6f} "
              f"(bits/token {loss_trace[-1][2]/math.log(2):.6f})")

try:
    _training_diagnostics()
except Exception as _exc:
    print(f"[tdiag] training diagnostics failed: {type(_exc).__name__}: {_exc}")


def _set_pos(state, n):
    """Diagnostics-only: jump a state to position `n`. `seq` IS the arithmetic; `pos` is the
    python-side mirror that only chooses which captured extent-graph replays, so a probe that
    pokes one must poke the other or it would be timing a variant the request never runs."""
    state["seq"].fill_(n)
    state["pos"] = n


def _kernel_cat(name):
    n = name.lower()
    if "flash" in n or "fwd_kernel" in n or "kvcache" in n or "splitkv" in n:
        return "flash_attn_with_kvcache"
    if "triton" in n:
        return "compiled (triton)"
    if any(s in n for s in ("gemm", "cutlass", "gemv", "cublas", "nt_align", "sm90")):
        return "GEMM/GEMV"
    return "other"


def _profile_eager_step(base, max_len, start_seq, tok, reps=3):
    """Per-kernel device time of ONE eager (uncaptured) step under the forced config."""
    from torch.profiler import profile, ProfilerActivity
    st = base.init_decode_state(batch=1, max_len=max_len, graph=False)
    _set_pos(st, start_seq)
    for _ in range(3):
        base.decode_step(tok, st)
    _set_pos(st, start_seq)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            base.decode_step(tok, st)
        torch.cuda.synchronize()

    def _dt(e):
        return getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0)
    kernels = [e for e in prof.key_averages() if _dt(e) > 0]
    total = sum(_dt(e) for e in kernels) / reps
    flash = [(e.key, e.count / reps, _dt(e) / max(1, e.count))
             for e in kernels if _kernel_cat(e.key) == "flash_attn_with_kvcache"]
    del st
    return total, flash, kernels, reps


_SWEEP_VARIANTS = [
    ("fa3 ns=1 (pinned ref)",  {"mode": "fa3", "num_splits": 1, "nowindow": False, "write": True}),
    ("fa3 ns=0 (heuristic)",   {"mode": "fa3", "num_splits": 0, "nowindow": False, "write": True}),
    ("fa3 ns=2",               {"mode": "fa3", "num_splits": 2, "nowindow": False, "write": True}),
    ("fa3 ns=4",               {"mode": "fa3", "num_splits": 4, "nowindow": False, "write": True}),
    ("fa3 ns=8",               {"mode": "fa3", "num_splits": 8, "nowindow": False, "write": True}),
    ("fa3 ns=0 no-window",     {"mode": "fa3", "num_splits": 0, "nowindow": True,  "write": True}),
    ("torch mask (write fused)", {"mode": "torch", "num_splits": 1, "nowindow": False, "write": True}),
    ("torch mask (write eager)", {"mode": "torch", "num_splits": 1, "nowindow": False, "write": False}),
    ("torch MONO (1 region)",  {"mode": "torch", "num_splits": 1, "nowindow": False,
                                "write": True, "mono": True}),
    ("torch MONO head-major kv", {"mode": "torch", "num_splits": 1, "nowindow": False,
                                  "write": True, "mono": True, "hm": True}),
    ("torch MONO hm+reduce attn", {"mode": "torch", "num_splits": 1, "nowindow": False,
                                   "write": True, "mono": True, "hm": True, "red": True}),
    ("torch MONO reduce attn",   {"mode": "torch", "num_splits": 1, "nowindow": False,
                                  "write": True, "mono": True, "red": True}),
    # --- node 1.11: size the state to the request, not to the rotary table.
    ("torch MONO hm+red pad8",    {"mode": "torch", "num_splits": 1, "nowindow": False,
                                   "write": True, "mono": True, "hm": True, "red": True,
                                   "pad8": True}),
    ("torch MONO hm+red pad8 blk256", {"mode": "torch", "num_splits": 1, "nowindow": False,
                                       "write": True, "mono": True, "hm": True, "red": True,
                                       "pad8": True, "blk": 256}),
    ("torch MONO hm+red pad8 blk128", {"mode": "torch", "num_splits": 1, "nowindow": False,
                                       "write": True, "mono": True, "hm": True, "red": True,
                                       "pad8": True, "blk": 128}),
]

# --- node 1.13: the weight projections as SPLIT-K REDUCTIONS instead of cuBLAS gemvx.
# Built on the node-1.11 winner (mono + head-major + reduction attn + pad8 + blk128), which is
# the only config where the whole step is one inductor region, so a reduction-form projection
# can fuse its neighbours in as prologue/epilogue. One key per projection, so the sweep sees
# each substitution ALONE (is this projection faster as a reduction?) and cumulatively (do the
# wins compose?); a losing variant is simply never selected and the score is unchanged.
_SK_BASE = {"mode": "torch", "num_splits": 1, "nowindow": False, "write": True,
            "mono": True, "hm": True, "red": True, "pad8": True, "blk": 128}


def _sk_cfg(label, **sk):
    return (f"MONO best + skv {label}", dict(_SK_BASE, **sk))


_SK_VARIANTS = [
    # one projection at a time: 1024x1024 out-proj, 2048x1024 c_fc, 1024x2048 c_proj,
    # 3080x1024 fused qkv+gate, 8192x1024 lm_head.
    _sk_cfg("proj=8", sk_proj=8),
    _sk_cfg("proj=16", sk_proj=16),
    _sk_cfg("proj=-8 (1-stage)", sk_proj=-8),
    _sk_cfg("fc=8", sk_fc=8),
    _sk_cfg("cproj=8", sk_cproj=8),
    _sk_cfg("cproj=16", sk_cproj=16),
    _sk_cfg("qkv=8", sk_qkv=8),
    _sk_cfg("lm=4", sk_lm=4),
    # cumulative
    _sk_cfg("proj/fc/cproj=8", sk_proj=8, sk_fc=8, sk_cproj=8),
    _sk_cfg("proj/fc/cproj/qkv=8", sk_proj=8, sk_fc=8, sk_cproj=8, sk_qkv=8),
    _sk_cfg("all 9 (+lm=4)", sk_proj=8, sk_fc=8, sk_cproj=8, sk_qkv=8, sk_lm=4),
]

TV_ACCEPT = 0.005          # a variant must agree with the pinned reference this closely
# The nopref regime IS the scored objective and its agreement gate has 0.05 of room
# against a decode path that already sits ~0.03 from `forward`, so a variant there may
# spend a little of that room on kernel selection. See _sweep_decode_attn.
TV_ACCEPT_NOPREF = 0.02


def _print_kernel_names(base, label, cfg, max_len, start_seq, tok):
    """FULL per-kernel name list for one step under `cfg` -- the fusion-leak evidence.

    The category histogram says "27 triton kernels" but not WHICH, and at m=1 every kernel is
    a ~1.7us (elementwise) / ~7.6us (matmul) floor, so the name list is the only way to see
    where inductor split a region it should have fused.
    """
    global _DECODE_CFG_FORCE
    prev = _DECODE_CFG_FORCE
    _DECODE_CFG_FORCE = cfg
    try:
        tot, _flash, kernels, reps = _profile_eager_step(base, max_len, start_seq, tok)

        def _dt(e):
            return getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0)
        cnt = {}
        for e in kernels:
            c = _kernel_cat(e.key)
            a = cnt.setdefault(c, [0.0, 0.0])
            a[0] += _dt(e) / reps
            a[1] += e.count / reps
        head = "  ".join(f"{c}={a[1]:.0f}k/{a[0]:.1f}us" for c, a in
                         sorted(cnt.items(), key=lambda kv: -kv[1][0]))
        n = sum(a[1] for a in cnt.values())
        print(f"[names] === {label}: {n:.0f} device ops, {tot:.1f}us total  [{head}]")
        for e in sorted(kernels, key=lambda e: -_dt(e)):
            print(f"[names]   {e.count/reps:4.1f}x {_dt(e)/max(1,e.count):7.2f}us "
                  f"{_kernel_cat(e.key):22s} {e.key}")
    except Exception as exc:                          # noqa: BLE001
        print(f"[names] {label}: failed {type(exc).__name__}: {exc}")
    finally:
        _DECODE_CFG_FORCE = prev
    gc.collect(); torch.cuda.empty_cache()


def _sweep_decode_attn(base, label, max_len, start_seq, steps, tv_accept=None, variants=None):
    """Time every width-1 attention variant on ONE real request shape, in one launch.

    Returns the winning config. `steps` graph replays from `start_seq` is exactly the decode
    half of the request being modelled, so the ms column is directly comparable to
    request_ms_median. Every variant is checked against the pinned-reference variant's own
    captured output, so a split-reduction or masking change that moves probabilities is
    disqualified here rather than in the scored agreement check.

    `tv_accept` is how far a variant may sit from that reference. It is a budget, not a
    correctness test: bf16 matmuls have no canonical reduction order, and the scored quantity
    is agreement with `forward`, which the decode path already misses by ~0.02-0.03 TV while
    the gate is 0.05. Measured: the one-region step differs from the split-region step by
    0.0062 TV purely because inductor feeds the score bmm a strided view instead of a
    materialised transpose, so cuBLAS picks a cutlass align2 kernel instead of gemv2T. The
    default is therefore deliberately tight, and it is relaxed ONLY for the nopref regime,
    which is the scored objective; the long regime moves only the request_ms_median tiebreak,
    so it is not worth any numeric risk against the decode_tv_distance_max gate.
    """
    global _DECODE_CFG_FORCE
    if tv_accept is None:
        tv_accept = TV_ACCEPT
    dev = base.transformer.wte.weight.device
    tok = torch.zeros(1, 1, dtype=torch.int64, device=dev)
    rows, ref_p = [], None
    print(f"[sweep] === {label}: max_len={max_len} start_seq={start_seq} steps={steps} "
          f"tv_accept={tv_accept} ===")
    for name, cfg in (variants if variants is not None else _SWEEP_VARIANTS):
        _DECODE_CFG_FORCE = cfg
        row = {"name": name, "cfg": cfg, "cap": False, "reason": "", "ms": float("inf"),
               "wall": float("nan"), "tv": float("nan"), "eager_us": float("nan"),
               "flash": []}
        st = g = None
        try:
            st = base.init_decode_state(batch=1, max_len=max_len, graph=True)
            for _ in range(3):
                base.decode_step(tok, st)
            row["cap"], row["reason"] = _GraphedDecodeStep.LAST_STATUS
            g = st.get("graph")
            if row["cap"] and g is not None:
                _set_pos(st, 0)
                for _ in range(8):
                    lg = g.replay(tok)
                p = torch.softmax(lg.float().view(-1), dim=-1).clone()
                if ref_p is None:
                    ref_p = p
                row["tv"] = float(0.5 * (p - ref_p).abs().sum())
                _set_pos(st, start_seq)
                for _ in range(20):
                    g.replay(tok)
                _set_pos(st, start_seq)
                ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
                torch.cuda.synchronize()
                t0 = time.time()
                ev0.record()
                for _ in range(steps):
                    g.replay(tok)
                ev1.record()
                torch.cuda.synchronize()
                row["ms"] = ev0.elapsed_time(ev1)
                row["wall"] = (time.time() - t0) * 1000.0
                eager_us, flash, _, _ = _profile_eager_step(base, max_len, start_seq, tok)
                row["eager_us"], row["flash"] = eager_us, flash
        except Exception as exc:                      # noqa: BLE001
            row["reason"] = f"{type(exc).__name__}: {exc}"
        del st, g
        rows.append(row)
        gc.collect(); torch.cuda.empty_cache()
    _DECODE_CFG_FORCE = None

    print(f"[sweep] {'variant':38s} {'cap':5s} {'dev ms':>8s} {'wall ms':>8s} "
          f"{'us/step':>8s} {'tv_vs_ref':>10s} {'eager us':>9s}  attn kernels")
    for r in rows:
        us = r["ms"] / steps * 1000.0 if r["ms"] < float("inf") else float("nan")
        fk = " ".join(f"{c:.1f}x{t:.2f}us" for _, c, t in r["flash"]) or "-"
        print(f"[sweep] {r['name']:38s} {str(r['cap']):5s} {r['ms']:8.2f} {r['wall']:8.2f} "
              f"{us:8.1f} {r['tv']:10.5f} {r['eager_us']:9.1f}  {fk}")
        if not r["cap"]:
            print(f"[sweep]   -> not captured / failed: {r['reason']!r}")
    ok = [r for r in rows if r["cap"] and r["ms"] < float("inf")
          and not (r["tv"] != r["tv"]) and r["tv"] <= tv_accept]
    best = min(ok, key=lambda r: r["ms"]) if ok else rows[0]
    ref = rows[0]
    print(f"[sweep] WINNER {label}: {best['name']} -> {best['cfg']}  "
          f"({best['ms']:.2f} ms vs pinned-ref {ref['ms']:.2f} ms, "
          f"{100 * (1 - best['ms'] / max(ref['ms'], 1e-9)):+.1f}%)")
    return dict(best["cfg"])


def _decode_diagnostics():
    base = model._orig_mod if hasattr(model, "_orig_mod") else model
    base.eval()
    steps = 200
    tok = torch.zeros(1, 1, dtype=torch.int64, device=device)
    print(f"[diag] config: n_layer={base.config.n_layer} n_embd={base.config.n_embd} "
          f"n_head={base.config.n_head} ffn_dim={base.config.ffn_dim}")
    print(f"[diag] windows: {[w[0] for w in base.window_sizes]}  "
          f"ve layers: {sorted(int(k) for k in base.value_embeds.keys())}")
    print(f"[diag] flops/token measured={flops_per_token_measured:,} "
          f"analytic={base.estimate_flops():,} ceiling=239,078,400")
    # The node-2.7 objective, measured on the actual model: every transformer matrix and
    # lm_head is read IN FULL once per width-1 step; wte and each value-embedding table are
    # single-row gathers (n_embd * 2 bytes each) and are effectively free.
    _pc = count_params(base)
    _streamed = 2 * (_pc["transformer_matrices"] + _pc["lm_head"])
    _gathered = 2 * base.config.n_embd * (1 + len(base.value_embeds))
    print(f"[diag] STREAMED decode bytes/step = 2 x (matrices {_pc['transformer_matrices']:,}"
          f" + lm_head {_pc['lm_head']:,}) = {_streamed:,} B ({_streamed/1e6:.2f} MB); "
          f"trunk was 67.11 MB -> {100*(_streamed/67108864 - 1):+.1f}%")
    print(f"[diag] row-gathered (free) bytes/step = {_gathered:,} B from "
          f"{_pc['wte'] + _pc['value_embeds']:,} parameters "
          f"({100*(_pc['wte']+_pc['value_embeds'])/_pc['total']:.1f}% of the total)")
    global _DECODE_CFG_SHORT, _DECODE_CFG_LONG
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        base._build_decode_weights()
        print(f"[diag] fused qkv weight shape: {tuple(base._dw[0]['qkv'].shape)} (layer0) "
              f"{tuple(base._dw[-1]['qkv'].shape)} (last, +ve_gate rows if any)")
        # BEFORE/AFTER fusion evidence: the full kernel-name list of the split-region torch
        # path (2 compiled regions per layer + prologue + tail) and of the monolithic one.
        _tok = torch.zeros(1, 1, dtype=torch.int64, device=device)
        _print_kernel_names(base, "BEFORE split regions (torch, write fused)",
                            {"mode": "torch", "num_splits": 1, "nowindow": False,
                             "write": True, "mono": False}, 514, 0, _tok)
        _print_kernel_names(base, "AFTER mono (1 compiled region)",
                            {"mode": "torch", "num_splits": 1, "nowindow": False,
                             "write": True, "mono": True}, 514, 0, _tok)
        _print_kernel_names(base, "AFTER mono + head-major kv cache",
                            {"mode": "torch", "num_splits": 1, "nowindow": False,
                             "write": True, "mono": True, "hm": True}, 514, 0, _tok)
        _print_kernel_names(base, "AFTER mono + head-major + reduction attn",
                            {"mode": "torch", "num_splits": 1, "nowindow": False,
                             "write": True, "mono": True, "hm": True, "red": True},
                            514, 0, _tok)
        # node 1.11: (a) the multiple-of-8 allocation (514 -> 520) and (b) the blocked
        # reduction extent. Both are visible in the kernel NAMES: (a) is where an `align2`
        # matmul would become an aligned variant, (b) is where the reduction/mask kernels
        # shrink. `_print_kernel_names` profiles at position `start_seq`, so the blk row is
        # the FIRST block's extent (128 of 520), i.e. what the early steps actually run.
        _print_kernel_names(base, "AFTER hm+red + pad8 (alloc 520)",
                            {"mode": "torch", "num_splits": 1, "nowindow": False,
                             "write": True, "mono": True, "hm": True, "red": True,
                             "pad8": True}, 514, 0, _tok)
        _print_kernel_names(base, "AFTER hm+red + pad8 + blk128 @pos0 (extent 128)",
                            {"mode": "torch", "num_splits": 1, "nowindow": False,
                             "write": True, "mono": True, "hm": True, "red": True,
                             "pad8": True, "blk": 128}, 514, 0, _tok)
        _print_kernel_names(base, "AFTER hm+red + pad8 + blk128 @pos400 (extent 520)",
                            {"mode": "torch", "num_splits": 1, "nowindow": False,
                             "write": True, "mono": True, "hm": True, "red": True,
                             "pad8": True, "blk": 128}, 514, 400, _tok)
        # node 1.13: the projections as split-K reductions. THIS is where `gemvx`/`ampere`
        # kernels either disappear from the name list or do not; per-kernel us in the list is
        # the direct cuBLAS-vs-reduction comparison on the same bytes.
        _print_kernel_names(base, "n1.13 skv proj/fc/cproj=8 (blk128 @pos0)",
                            dict(_SK_BASE, sk_proj=8, sk_fc=8, sk_cproj=8), 514, 0, _tok)
        _print_kernel_names(base, "n1.13 skv all 9 proj (+qkv=8, lm=4) (blk128 @pos0)",
                            dict(_SK_BASE, sk_proj=8, sk_fc=8, sk_cproj=8, sk_qkv=8,
                                 sk_lm=4), 514, 0, _tok)
        # The two shapes prepare.py actually probes: no-prefill (1+512, max_len 514) and
        # full-prefill (1536+512, max_len 2049). The plain-torch path reads the WHOLE cache,
        # so its cost scales with max_len while FA3's scales with the position -- the two
        # regimes can and do want different configurations.
        try:
            _DECODE_CFG_SHORT = _sweep_decode_attn(base, "nopref", 514, 0, 512,
                                                    tv_accept=TV_ACCEPT_NOPREF,
                                                    variants=_SWEEP_VARIANTS + _SK_VARIANTS)
        except Exception as exc:                       # noqa: BLE001
            print(f"[sweep] short sweep failed: {type(exc).__name__}: {exc}")
        try:
            _DECODE_CFG_LONG = _sweep_decode_attn(base, "prefill1536", 2049, 1536, 512)
        except Exception as exc:                       # noqa: BLE001
            print(f"[sweep] long sweep failed: {type(exc).__name__}: {exc}")
        print(f"[diag] SELECTED short(max_len<={_SHORT_REGIME_MAX_LEN}): {_DECODE_CFG_SHORT}")
        print(f"[diag] SELECTED long : {_DECODE_CFG_LONG}")
        # --- graphed path, selected config, on the nopref shape
        st = base.init_decode_state(batch=1, max_len=514, graph=True)
        print(f"[diag] selected short state: hm={st['hm']} "
              f"kc[0].shape={tuple(st['kc'][0].shape)} "
              f"kc[0].stride={st['kc'][0].stride()} "
              f"kv bytes={sum(t.numel() * t.element_size() for t in st['kc'] + st['vc']):,}")
        for _ in range(5):
            base.decode_step(tok, st)
        cap, reason = _GraphedDecodeStep.LAST_STATUS
        print(f"[diag] graph capture: captured={cap} reason={reason!r} "
              f"warmup_replays={_GraphedDecodeStep.WARMUP_REPLAYS}")
        print(f"[diag] selected short state: extents={st['extents']} blk={st['blk']} "
              f"alloc_len={st['alloc_len']} (requested {st['max_len']})")
        g = st.get("graph")
        if cap and g is not None:
            print(f"[diag] captured graphs: {len(g.graphs)} "
                  f"(one per extent {g.extents}), select blk={g.blk}")
            # Per-extent device time: what a step costs inside each block of the request.
            # Each graph is replayed DIRECTLY (bypassing the position dispatch) from seq=0, so
            # the only thing that varies between rows is how many cache slots are read; the
            # numbers these replays produce are meaningless, which is fine for a clock.
            for k in range(len(g.graphs)):
                _set_pos(st, 0)
                for _ in range(10):
                    g.graphs[k].replay()
                _set_pos(st, 0)
                ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
                torch.cuda.synchronize(); ev0.record()
                for _ in range(steps):
                    g.graphs[k].replay()
                ev1.record(); torch.cuda.synchronize()
                print(f"[diag]   extent {g.extents[k]:5d}: "
                      f"{ev0.elapsed_time(ev1)/steps*1000:7.1f} us/step (device)")
            base.reset_decode_state(st)
            # g is the _GraphedDecodeStep wrapper: replay() takes the next token.
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(steps):
                g.replay(tok)
            torch.cuda.synchronize()
            graph_us = (time.time() - t0) / steps * 1e6
            print(f"[diag] graph replay: {graph_us:.1f} us/step  -> "
                  f"{graph_us * 512 / 1000:.1f} ms for 512 steps")
            # device-side time of one replay, from CUDA events (excludes host launch)
            ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
            torch.cuda.synchronize()
            ev0.record()
            for _ in range(steps):
                g.replay(tok)
            ev1.record()
            torch.cuda.synchronize()
            print(f"[diag] graph replay device time: "
                  f"{ev0.elapsed_time(ev1) / steps * 1000:.1f} us/step")
        base.reset_decode_state(st)
        # --- eager path (graph=False), same shape
        st2 = base.init_decode_state(batch=1, max_len=514, graph=False)
        for _ in range(5):
            base.decode_step(tok, st2)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(steps):
            base.decode_step(tok, st2)
        torch.cuda.synchronize()
        eager_us = (time.time() - t0) / steps * 1e6
        print(f"[diag] eager step  : {eager_us:.1f} us/step")
        # --- how many device ops one step actually issues, and where the time sits
        base.reset_decode_state(st2)
        tot, _flash, kernels, reps = _profile_eager_step(base, 514, 0, tok)
        n_launch = sum(e.count for e in kernels) / reps

        def _dt(e):
            return getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0)
        print(f"[diag] device ops per eager step: {n_launch:.1f}  (top by count)")
        for e in sorted(kernels, key=lambda e: -e.count)[:14]:
            print(f"[diag]   {e.count/reps:5.1f}x {_dt(e)/max(1,e.count):7.2f}us "
                  f"{e.key[:70]}")
        # coarse breakdown: flash-attn kernels vs inductor-compiled regions vs GEMMs
        agg = {}
        for e in kernels:
            c = _kernel_cat(e.key)
            a = agg.setdefault(c, [0.0, 0])
            a[0] += _dt(e) / reps
            a[1] += e.count / reps
        tot = sum(v[0] for v in agg.values())
        print(f"[diag] eager step device time {tot:.1f}us across categories:")
        for c, (us, cnt) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
            print(f"[diag]   {c:26s} {us:7.1f}us ({100*us/max(tot,1e-9):4.1f}%) "
                  f"{cnt:5.1f} kernels/step")
        del st, st2, g
    from torch._dynamo.utils import counters as _dyn_counters
    gb = dict(_dyn_counters.get("graph_break", {}))
    print(f"[diag] dynamo graph_break counters ({len(gb)} distinct): "
          f"{sorted(gb.items(), key=lambda kv: -kv[1])[:8]}")
    print(f"[diag] dynamo cache_size_limit={torch._dynamo.config.cache_size_limit} "
          f"unique_graphs={_dyn_counters['stats'].get('unique_graphs')}")
    gc.collect(); torch.cuda.empty_cache()

try:
    _decode_diagnostics()
except Exception as _exc:
    print(f"[diag] diagnostics failed: {type(_exc).__name__}: {_exc}")

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

