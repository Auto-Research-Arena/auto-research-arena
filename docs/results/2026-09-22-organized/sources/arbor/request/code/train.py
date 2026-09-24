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

# The width-1 decode body is compiled with dynamic=False, so it retraces once per distinct
# cache shape (the probes use four), once per distinct split-K value the diagnostic tries AND
# once per matvec variant the sweep prices (the variant is a Python value Dynamo guards on).
# Measured: at 15 variants the limit of 64 was reached partway through the sweep, after which
# Dynamo abandoned this frame and every LATER cache shape ran eager -- the 512-slot no-prompt
# request went 55 -> 138 ms with no error anywhere. The limit only bounds how many traces are
# kept, and each is validated and timed before it can be pinned, so it is raised well past the
# grid size rather than trimmed to it.
for _limit_attr in ("recompile_limit", "cache_size_limit"):
    if hasattr(torch._dynamo.config, _limit_attr):
        setattr(torch._dynamo.config, _limit_attr, 512)

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
    mlp_ratio: float = 4.0
    # PARALLEL-BRANCH TOPOLOGY (node 3.5). The network is n_stages SERIAL stages; each stage
    # holds n_branch PARALLEL branches that all read the same normed stage input and whose
    # outputs are summed into the residual (GPT-J/PaLM parallel attn+MLP, generalised to k
    # branches). n_layer == n_stages * n_branch is the total number of attention+MLP blocks,
    # i.e. the capacity; n_stages is what the decode loop pays serially.
    n_stages: int = 12
    n_branch: int = 1
    # True: the branches' attention AND MLP both read the same norm(x) (full PaLM/GPT-J
    # parallel form, one sequential sublayer per stage). False: the MLP branches read
    # norm(x + attention), so a stage is TWO sequential sublayers -- the k branches inside
    # each sublayer are still parallel, so the single batched flash call and the merged
    # branch matvecs are unchanged.
    parallel_mlp: bool = True
    # Explicit value-embedding layer set. None = the default alternating rule. Depth is the
    # serial cost in the decode loop, so at reduced depth the freed parameter budget is spent
    # on extra VE layers, which are gathers: parameters at zero FLOPs and zero decode kernels.
    ve_layers: tuple = None


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def mv_reduce(x, w, dtype=None):
    """A batch-1 matvec written as a reduction instead of a gemm.

    At width 1 every `F.linear` in the decode body is a matrix-vector product, and cuBLAS
    serves those with gemv/skinny-gemm kernels that reach only ~200-800 GB/s on an H100 --
    the weight read is the whole cost, so the kernel is worth exactly the bandwidth it gets.
    Written as an explicit reduction, inductor emits its own (split) reduction kernel for it
    and picks the tiling from the shape. Same dot products, fp32 accumulation, so the
    difference from the gemm is bf16 rounding -- which is why the gate on this path is the
    distributional check in `_validate_width1`, not bit equality.

    `x` is cast to the weight dtype because the residual stream is fp32 (a bf16 activation
    times an fp32 per-layer lambda) and `F.linear`, which this replaces, is an autocast op
    that would have done exactly that cast. Without it the products are fp32 and the attention
    kernel is handed fp32 queries.
    """
    acc = w.dtype if dtype is None else dtype
    return (w * x.reshape(1, -1).to(w.dtype)).sum(-1, dtype=acc).view(1, 1, -1)


def quantize_rows(w):
    """Int8 weight-only quantisation of a decode matrix with per-OUTPUT-CHANNEL scales.

    `w` is [out, in]; every row gets its own absmax scale, so the quantisation grid adapts to
    each output channel and the dequantisation is a single vector multiply on the [out] result
    AFTER the fp32 reduction -- the scale never touches the inner loop. Returns (int8 rows,
    fp32 scales). Values, not bits, are preserved: this is the one approximation this node
    makes, and it is gated distributionally by `_validate_width1` (TV against the generic bf16
    body on a real prefilled cache), which demotes a site back to bf16 if it costs too much.
    """
    wf = w.detach().float()
    scale = (wf.abs().amax(dim=1) / 127.0).clamp_min(1e-12)
    q = torch.round(wf / scale[:, None]).clamp_(-127, 127).to(torch.int8)
    return q.contiguous(), scale.contiguous()


def mv_reduce_int8(x, wq, scale, dtype=None):
    """The batch-1 matvec over an int8 weight, as a reduction, dequantised in registers.

    Half the weight bytes of `mv_reduce` for the same dot products. Written as a reduction for
    exactly the reason that matters here: inductor fuses the int8 -> bf16 conversion into the
    reduction loop, so the int8 rows are loaded ONCE and the dequantised matrix is never
    materialised (a `F.linear(x, wq.to(bf16) * scale)` form would read int8 and then write AND
    read a bf16 copy, i.e. strictly more traffic than plain bf16). int8 magnitudes <= 127 are
    exactly representable in bf16, so the conversion itself is lossless; accumulation is fp32
    and the per-row scale is applied to the fp32 result.
    """
    acc = (wq.to(torch.bfloat16) * x.reshape(1, -1).to(torch.bfloat16)).sum(-1,
                                                                           dtype=torch.float32)
    out = acc * scale
    return out.to(torch.bfloat16 if dtype is None else dtype).view(1, 1, -1)


def mv_site(x, wb, wq, ws, form, dtype=None):
    """One decode matvec site. `form` is a Python int resolved at trace time:
    0 = `F.linear` (cuBLAS/autotuned triton over bf16), 1 = bf16 reduction, 2 = int8 reduction,
    3 = the hand-written int8 triton gemv below.
    """
    if form == 3:
        return mv_triton_int8(x, wq, ws, dtype)
    if form == 2:
        return mv_reduce_int8(x, wq, ws, dtype)
    if form == 1:
        return mv_reduce(x, wb, dtype)
    out = F.linear(x, wb)
    return out.float() if dtype == torch.float32 else out


# Option (a) of this node: an explicit triton gemv over the int8 weight. Inductor's reduction
# (form 2) already loads int8 and dequantises in registers, but its tiling is chosen by the
# generic reduction heuristics; this kernel fixes the tiling per shape so the row block and the
# k-loop are sized for a batch-1 weight stream. The trade-off it pays is fusion: a custom kernel
# cannot absorb the preceding rms-norm the way inductor's fused reduction does, so it is timed
# both in isolation (ledger) and inside the step (sweep), and only pinned if it wins there.
import triton                                                       # noqa: E402
import triton.language as tl                                        # noqa: E402


@triton.jit
def _int8_gemv_kernel(W, S, X, Y, M, K, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    rmask = rows < M
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        kmask = ks < K
        w = tl.load(W + rows[:, None] * K + ks[None, :],
                    mask=rmask[:, None] & kmask[None, :], other=0)
        xv = tl.load(X + ks, mask=kmask, other=0.0)
        acc += w.to(tl.float32) * xv.to(tl.float32)[None, :]
    out = tl.sum(acc, 1) * tl.load(S + rows, mask=rmask, other=0.0)
    tl.store(Y + rows, out, mask=rmask)


# Tiling per weight shape, filled in by `tune_int8_gemv` before anything is compiled or
# captured (so the winning constants are baked into the traced body and no autotuning ever
# happens inside a CUDA graph). Empty = use the heuristic.
_INT8_TILING = {}


def _int8_gemv_tiling(m, k):
    """Row block sized so the launch fills the device (>= ~2 waves of 108 SMs), k-loop capped."""
    tuned = _INT8_TILING.get((m, k))
    if tuned is not None:
        return tuned
    bm = 1
    for cand in (2, 4, 8, 16):
        if m // cand >= 216:
            bm = cand
    return bm, min(1024, triton.next_power_of_2(k)), 4


def mv_triton_int8(x, wq, scale, dtype=None):
    m, k = wq.shape
    bm, bk, warps = _int8_gemv_tiling(m, k)
    out_dtype = torch.float32 if dtype == torch.float32 else torch.bfloat16
    y = torch.empty((m,), dtype=out_dtype, device=wq.device)
    xf = x.reshape(-1).to(torch.bfloat16)
    _int8_gemv_kernel[(triton.cdiv(m, bm),)](wq, scale, xf, y, m, k,
                                             BLOCK_M=bm, BLOCK_K=bk, num_warps=warps)
    return y.view(1, 1, m)


# (name, bf16-reduction sites, gemm autotuning, INT8 sites, int8 engine) for the width-1 body.
# The engine is "reduce" (inductor fuses the int8 -> bf16 dequant into its reduction) or
# "triton" (the hand-tiled `_int8_gemv_kernel`). The sweep at the end of the file times every
# candidate on captured graphs and pins the fastest, so which kernel and which precision serves
# which matvec is measured on the machine, not asserted. int8 takes priority over the bf16
# reduction where a site appears in both sets.
W1_VARIANT_DEFAULT = ("linear", (), True, (), "reduce")
W1_AUTOTUNE_OPTIONS = {"max_autotune_gemm": True,
                       "max_autotune_gemm_backends": "ATEN,TRITON",
                       "coordinate_descent_tuning": False}

# How the width-1 step calls flash. The per-layer attention is THREE kernels in the trunk:
# the main kernel, the split-K combine, and `prepare_varlen_num_blocks` -- flash's own
# scheduler bookkeeping, launched because `cache_seqlens` puts the call on the varlen path and
# varlen implies dynamic split scheduling. The bookkeeping kernel does no arithmetic the step
# needs, and its input (a batch-1 sequence length) is known before the step runs, so it can be
# hoisted out of the step entirely: `get_scheduler_metadata` computes exactly that tensor once,
# and passing it as `scheduler_metadata` makes the C++ side skip the per-call launch.
#
#   "kvcache"       trunk: flash appends k/v itself, flash computes its own metadata.
#   "sched"         flash appends k/v, metadata hoisted out of the step.
#   "append"        we append k/v with `index_copy_` in the already-fused q/k/v epilogue and
#                   hand flash a cache with `cache_seqlens = seq + 1` and no k_new.
#   "append_sched"  both.
#
# Every mode is measured and distributionally validated in-run by `sweep_decode_attn`; a mode
# that fails either never reaches the scored probe.
DECODE_ATTN_MODE = "kvcache"



def has_ve(layer_idx, config):
    """Returns True if layer should have Value Embedding.

    Default: alternating, last always included. `config.ve_layers` overrides it with an
    explicit set.
    """
    if config.ve_layers is not None:
        return layer_idx in config.ve_layers
    return layer_idx % 2 == (config.n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


def rope_norm_slim(x, cos, sin):
    """Rotary + rms_norm on one pre-split view, same values as norm(apply_rotary_emb(x)).

    `x1 * (-sin) + x2 * cos` is rewritten as `x2 * cos - x1 * sin`: -sin is exact in any
    float format, so the two are bit-identical, one op shorter, and free of the intermediate
    negation. Under inductor the whole expression plus the norm is one kernel whose epilogue
    stores the concatenated result.
    """
    x1, x2 = x.chunk(2, dim=-1)
    y = torch.cat((x1 * cos + x2 * sin, x2 * cos - x1 * sin), dim=-1)
    return F.rms_norm(y, (y.size(-1),))


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
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config) else None

    def qkv(self, x, ve, cos_sin):
        """Projections, value residual and rotary/norm only. The flash call itself lives in
        `Stage`, which issues ONE call for all the branches of the stage."""
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
        return q, k, v


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = int(round(config.mlp_ratio * config.n_embd))
        self.c_fc = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Stage(nn.Module):
    """One SERIAL stage holding `n_branch` PARALLEL branches.

    Every branch's attention AND MLP read the same `norm(x)` (PaLM/GPT-J parallel form), and
    all branch outputs are summed into the residual:

        h = norm(x);  x_out = x + sum_b [ attn_b(h) + mlp_b(h) ]

    Nothing inside the stage depends on another branch's output, so the k attentions are ONE
    flash call with the branch axis folded into the batch axis, and the k qkv / MLP projections
    are single larger matvecs (the decode pack in `build_decode_weights` does exactly that).
    """

    def __init__(self, config, stage_idx):
        super().__init__()
        k = config.n_branch
        self.n_branch = k
        self.parallel_mlp = config.parallel_mlp
        self.stage_idx = stage_idx
        self.attn = nn.ModuleList([CausalSelfAttention(config, stage_idx * k + b)
                                   for b in range(k)])
        self.mlp = nn.ModuleList([MLP(config) for _ in range(k)])

    def forward(self, x, ve_list, cos_sin, window_size):
        B, T, C = x.size()
        h = norm(x)
        qs, ks, vs = [], [], []
        for b, attn in enumerate(self.attn):
            q, k, v = attn.qkv(h, ve_list[b], cos_sin)
            qs.append(q); ks.append(k); vs.append(v)
        # Branch axis folded into the batch axis: ONE flash call for the whole stage.
        q = torch.cat(qs, 0) if len(qs) > 1 else qs[0]
        k = torch.cat(ks, 0) if len(ks) > 1 else ks[0]
        v = torch.cat(vs, 0) if len(vs) > 1 else vs[0]
        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        a = None
        for b in range(self.n_branch):
            yb = y[b * B:(b + 1) * B].reshape(B, T, -1)
            o = self.attn[b].c_proj(yb)
            a = o if a is None else a + o
        if self.parallel_mlp:
            m = None
            for mlp in self.mlp:
                o = mlp(h)
                m = o if m is None else m + o
            return x + a + m
        x = x + a
        g = norm(x)
        m = None
        for mlp in self.mlp:
            o = mlp(g)
            m = o if m is None else m + o
        return x + m


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        # Split-K width for the cached-decode attention kernel. 1 = one CTA per (head, batch)
        # scanning every key serially; >1 partitions the key range across more CTAs and
        # reduces. Read by both the eager and the graph-captured step, so the two paths always
        # compute the same thing. Chosen empirically by the sweep at the end of this file.
        self.decode_num_splits = DECODE_NUM_SPLITS
        # Which flash entry configuration the width-1 step uses (see DECODE_ATTN_MODE), and the
        # hoisted scheduler metadata the "sched" modes hand it: one int32 tensor per layer,
        # rebuilt whenever the mode or num_splits changes and never inside the timed step.
        self.decode_attn_mode = DECODE_ATTN_MODE
        self._sched_md = {}
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Stage(config, i) for i in range(config.n_stages)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_stages))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_stages))
        # Value embeddings
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config)
        })
        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        # Width-1 decode body, compiled once per variant and shared by the eager and the
        # graphed path. 0 = not built yet, 1 = building, 2 = compiled and validated,
        # 3 = disabled. `_w1_variant` is (name, bf16-reduction sites, gemm-autotune, int8 sites)
        # and is pinned in-run by
        # the sweep at the end of this file, exactly as decode_num_splits is.
        self._w1_status = 0
        self._w1_reason = ""
        self._w1_building = False
        self._w1_variant = W1_VARIANT_DEFAULT
        self._w1_cache = {}
        # Objects that expose this model's `decode_step` to the instrument (the torch.compile
        # wrapper). The installed fast step is published on each of them so the probe's
        # `model.decode_step` lookup is a plain instance-dict hit.
        self._ds_hosts = []
        # Decode-time weight pack: plain attribute, never a Parameter or a buffer, so it is
        # invisible to count_params and to state_dict. Built lazily after training.
        self._dw = None
        # PREFILL program (node 1.6). The width-1536 prompt pass has its own
        # torch.compile(dynamic=False) callable over the PACKED bf16 decode weights, cached per
        # prompt width. `_pf_mode` is the switch the diagnostic flips to price the programs
        # compiled in the same run; `_pf_min_width` keeps the small widths that only the in-run
        # validation uses (2, 128) on the eager path, so they never cost a retrace.
        self._pf_cache = {}
        # "eager" = the generic prefill branch, "compiled" = the dedicated program,
        # "graph" = that program inside a manually captured CUDA graph. Pinned in-run by the
        # diagnostic at the end of this file, which prices all three on the trained model.
        self._pf_mode = "compiled"
        self._pf_min_width = 256
        self._pf_status = 0
        self._pf_reason = ""
        self._pf_building = False
        self._pf_val_tokens = None

    @torch.no_grad()
    def init_weights(self):
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # Transformer blocks
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for stage in self.transformer.h:
            for attn in stage.attn:
                torch.nn.init.uniform_(attn.c_q.weight, -s, s)
                torch.nn.init.uniform_(attn.c_k.weight, -s, s)
                torch.nn.init.uniform_(attn.c_v.weight, -s, s)
                torch.nn.init.zeros_(attn.c_proj.weight)
            for mlp in stage.mlp:
                torch.nn.init.uniform_(mlp.c_fc.weight, -s, s)
                torch.nn.init.zeros_(mlp.c_proj.weight)
        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
        for stage in self.transformer.h:
            for attn in stage.attn:
                if attn.ve_gate is not None:
                    torch.nn.init.zeros_(attn.ve_gate.weight)
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
        """One window per STAGE, not per attention block: the k branches of a stage are one
        flash call, so they share the call's window argument."""
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for stage_idx in range(config.n_stages):
            char = pattern[stage_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    @property
    def attn_windows(self):
        """One entry per attention BLOCK (stage window repeated over its branches). What the
        FLOP and kv-byte accounting is made of."""
        return [w for w in self.window_sizes for _ in range(self.config.n_branch)]

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
        for window_size in self.attn_windows:
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
        k = self.config.n_branch
        for i, stage in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve_list = [self.value_embeds[str(i * k + b)](idx)
                       if str(i * k + b) in self.value_embeds else None for b in range(k)]
            x = stage(x, ve_list, cos_sin, self.window_sizes[i])
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
        cfg = self.config
        head_dim = cfg.n_embd // cfg.n_head
        dev = self.transformer.wte.weight.device
        kw = {"dtype": torch.bfloat16, "device": dev}
        # ONE cache per STAGE, with the k branches folded into the leading (batch) axis in
        # branch-major order, so the stage's whole attention is a single flash call at batch=k
        # over contiguous memory.
        kb = cfg.n_branch * batch
        return {
            "max_len": max_len,
            "graph_enabled": bool(graph),
            "seq": torch.zeros(batch, dtype=torch.int32, device=dev),
            "kc": [torch.zeros(kb, max_len, cfg.n_kv_head, head_dim, **kw)
                   for _ in range(cfg.n_stages)],
            "vc": [torch.zeros(kb, max_len, cfg.n_kv_head, head_dim, **kw)
                   for _ in range(cfg.n_stages)],
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
        if not prefill and idx.size(0) == 1:
            # One body for width 1, reached identically from `decode_step` and from the graph
            # capture, so "compiled" is never a third computation. Falls back to the generic
            # body below if either the compile or the agreement check failed. Batch 1 only: the
            # branch axis of the packed body occupies the batch axis of every tensor.
            fn = self.decode_width1_callable()
            if fn is not None:
                return fn(idx, state["seq"], state["kc"], state["vc"],
                          self.sched_for(state["kc"][0].size(1)))
        if prefill and idx.size(0) == 1:
            # The prompt pass has its own compiled program, specialised to this width and
            # reading the same packed bf16 weights the step reads. None means "use the eager
            # body below" (compile raised, validation failed, or the width is too small to be
            # worth a graph).
            fn = self.prefill_callable(idx.size(1))
            if fn is not None:
                return fn(idx, state["kc"], state["vc"])
        B, Tn = idx.size()
        if prefill:
            cos, sin = self.cos[:, :Tn], self.sin[:, :Tn]
        else:
            seq_idx = state["seq"].to(torch.int64)
            cos = self.cos.index_select(1, seq_idx)
            sin = self.sin.index_select(1, seq_idx)

        x = norm(self.transformer.wte(idx))
        x0 = x
        kb = self.config.n_branch
        seqk = state["seq"].repeat(kb) if kb > 1 else state["seq"]
        for i, stage in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            h = norm(x)
            qs, ks, vs = [], [], []
            for b, attn in enumerate(stage.attn):
                gi = i * kb + b
                ve = self.value_embeds[str(gi)](idx) if str(gi) in self.value_embeds else None
                q, k, v = attn.qkv(h, ve, cos_sin=(cos, sin))
                qs.append(q); ks.append(k); vs.append(v)
            q = torch.cat(qs, 0) if kb > 1 else qs[0]
            k = torch.cat(ks, 0) if kb > 1 else ks[0]
            v = torch.cat(vs, 0) if kb > 1 else vs[0]

            kc, vc = state["kc"][i], state["vc"][i]
            if prefill:
                kc[:, :Tn] = k
                vc[:, :Tn] = v
                y = fa3.flash_attn_func(q, kc[:, :Tn], vc[:, :Tn], causal=True,
                                        window_size=self.window_sizes[i])
            else:
                # Appends k,v at cache_seqlens and attends over the whole preallocated cache,
                # so no slice bound depends on the step number.
                y = fa3.flash_attn_with_kvcache(q, kc, vc, k=k, v=v,
                                                cache_seqlens=seqk, causal=True,
                                                window_size=self.window_sizes[i],
                                                num_splits=self.decode_num_splits)
            acc = None
            for b in range(kb):
                yb = y[b * B:(b + 1) * B].reshape(B, Tn, -1)
                o = stage.attn[b].c_proj(yb)
                if self.config.parallel_mlp:
                    o = o + stage.mlp[b](h)
                acc = o if acc is None else acc + o
            x = x + acc
            if not self.config.parallel_mlp:
                g = norm(x)
                m = None
                for mlp in stage.mlp:
                    o = mlp(g)
                    m = o if m is None else m + o
                x = x + m

        x = norm(x[:, -1:, :])
        softcap = 15
        logits = self.lm_head(x).float()
        return softcap * torch.tanh(logits / softcap)

    # -----------------------------------------------------------------------
    # Width-1 decode: one hand-slimmed body, compiled once, issued by both the eager and
    # the graph-captured path.
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def build_decode_weights(self):
        """Pre-cast and pre-concatenate the weights the width-1 step reads. Same values.

        Two launch-count facts drive this. (1) Under autocast with grad mode off there is no
        weight-cast cache, so every step re-materialises a bf16 copy of every fp32 weight:
        ~50 extra kernels and ~300 MB of traffic per step, for a body whose whole job is one
        token. Casting once, after training, removes both -- the values are the same bf16
        numbers autocast would have produced. (2) q, k, v and the value-embedding gate all
        read the same `norm(x)`, so they are one gemv over a stacked weight instead of four;
        the gate rows are zero-padded from 32 to n_embd columns, and multiplying by an exact
        zero contributes exactly zero to a dot product, so that too is value-preserving.
        """
        cfg = self.config
        hd = cfg.n_embd // cfg.n_head
        kb = cfg.n_branch
        pack = []
        for i, stage in enumerate(self.transformer.h):
            attns, mlps = list(stage.attn), list(stage.mlp)
            # Rows grouped BY ROLE across branches (all q, then all k, then all v, then all
            # gates), so one slice of the single matvec output is already the branch-batched
            # tensor the flash call wants: no cat, no permute inside the step.
            rows = ([a.c_q.weight for a in attns] + [a.c_k.weight for a in attns] +
                    [a.c_v.weight for a in attns])
            has_gate = attns[0].ve_gate is not None
            assert all((a.ve_gate is not None) == has_gate for a in attns), \
                "branches of a stage must agree on value embeddings (one packed gate block)"
            if has_gate:
                for a in attns:
                    gate = a.ve_gate.weight.new_zeros((cfg.n_kv_head, cfg.n_embd))
                    gate[:, :a.ve_gate_channels] = a.ve_gate.weight
                    rows.append(gate)
            entry = {
                "qkvg": torch.cat(rows, 0).to(torch.bfloat16).contiguous(),
                # Out-projections concatenated along the INPUT axis: one matvec over the
                # stacked branch attention outputs IS the sum of the branch out-projections.
                "o": torch.cat([a.c_proj.weight for a in attns], 1).to(torch.bfloat16).contiguous(),
                "fc": torch.cat([m.c_fc.weight for m in mlps], 0).to(torch.bfloat16).contiguous(),
                "fc2": torch.cat([m.c_proj.weight for m in mlps], 1).to(torch.bfloat16).contiguous(),
            }
            if has_gate:
                # The branches' value embeddings concatenated on the feature axis: ONE gather
                # per stage instead of k.
                entry["ve"] = torch.cat(
                    [self.value_embeds[str(i * kb + b)].weight for b in range(kb)],
                    1).to(torch.bfloat16).contiguous()
            pack.append(entry)
        # cos and sin stacked so the per-step position gather is one kernel, not two.
        self._dw = {
            "layers": pack,
            "cs": torch.stack((self.cos, self.sin), 0).contiguous(),
            "lm_head": self.lm_head.weight.to(torch.bfloat16).contiguous(),
            "nh": cfg.n_head, "nkv": cfg.n_kv_head, "hd": hd, "qk": cfg.n_head * hd,
            "kv": cfg.n_kv_head * hd, "kb": kb,
        }
        # An int8 copy of every packed matrix, per-output-channel scaled. Still a pure function
        # of the trained parameters and still a plain attribute (not a Parameter), so it moves
        # no gate: forward(), training and val_bpb never see it. Built for every site so the
        # in-run sweep can price int8 against bf16 site by site; a site the sweep does not pin
        # simply never reads its int8 copy.
        for w in self._dw["layers"]:
            for name in ("qkvg", "o", "fc", "fc2"):
                q, s = quantize_rows(w[name])
                w[name + "_q"], w[name + "_s"] = q, s
        q, s = quantize_rows(self._dw["lm_head"])
        self._dw["lm_head_q"], self._dw["lm_head_s"] = q, s
        return self._dw

    @torch.no_grad()
    def build_sched_metadata(self, max_seqlen_k, ref_seqlen=None):
        """Hoist flash's per-call scheduler bookkeeping out of the step, once per config.

        `prepare_varlen_num_blocks` is 7.0 us of every 150 us step and computes one thing: how
        many key blocks (and how many of the requested splits) each batch element needs. At
        batch 1 that is a function of one sequence length. `get_scheduler_metadata` produces the
        same tensor on the host side; handing it back as `scheduler_metadata` sets
        `skip_scheduler_metadata_computation` in the C++ launcher, so the launch disappears.

        The tensor is built for ONE representative position of a given cache length and reused
        at every position, which is legal because it only carries a split count: the kernel
        partitions whatever the runtime `cache_seqlens` says across that many splits and the
        combine reads the same count, so a stale count costs load balance, not correctness.
        That is not asserted -- `_validate_width1` re-checks the next-token distribution on a
        real prefilled cache and the sweep re-times every position, so a mode whose metadata
        does not hold is never pinned.

        One set per cache length, because the two scored request shapes allocate different
        caches (2049 slots for the 1536-token prefill, 514 for the no-prompt shape) and the
        representative position of a 514-slot cache is not 1792.
        """
        cfg = self.config
        hd = cfg.n_embd // cfg.n_head
        dev = self.transformer.wte.weight.device
        max_k = int(max_seqlen_k)
        ref = int(max(1, min(max_k - 1 if ref_seqlen is None else ref_seqlen, max_k)))
        k_new = 0 if "append" in self.decode_attn_mode else 1
        kb = cfg.n_branch
        md = []
        for window in self.window_sizes:
            cache_seqlens = torch.full((kb,), ref, dtype=torch.int32, device=dev)
            md.append(fa3.get_scheduler_metadata(
                kb, 1, max_k, cfg.n_head, cfg.n_kv_head, hd, cache_seqlens,
                qkv_dtype=torch.bfloat16, max_seqlen_k_new=k_new, causal=True,
                window_size=window, num_splits=self.decode_num_splits))
        return md

    def sched_for(self, max_seqlen_k):
        """The hoisted metadata for a cache of this length, or None if the mode does not use it.

        Built outside the step and cached on the model, so the per-step cost is a dict hit on
        the eager path and exactly nothing on the replayed graph.
        """
        if "sched" not in self.decode_attn_mode:
            return None
        key = (self.decode_attn_mode, self.decode_num_splits, int(max_seqlen_k))
        md = self._sched_md.get(key)
        if md is None:
            md = self._sched_md[key] = self.build_sched_metadata(max_seqlen_k)
        return md

    def _decode_width1(self, idx, seq, kc_list, vc_list, sched=None):
        """The width-1 step body. Same math as `forward`, fewer aten ops and fewer launches.

        Value-preserving differences from the generic body: the position is a single position
        so `x[:, -1:]` is gone; rotary and the q/k norm are one expression on a pre-split view
        (`rope_norm_slim`); the residual lambdas fold into one `addcmul`; the value-residual
        gate is an `addcmul` on a `view`; q/k/v/gate are a single gemv over the pre-stacked
        weight; every weight is read in bf16 from the pack instead of being cast per step; the
        out-projection reads flash's output through `reshape`, so the `contiguous()` copy is
        paid only if the kernel ever returns a strided output.

        Takes the cache lists as arguments rather than the state dict so Dynamo guards on
        four tensors and two lists, not on a dict that also carries a graph object.
        """
        dw = self._dw
        nh, nkv, hd, qk, kv = dw["nh"], dw["nkv"], dw["hd"], dw["qk"], dw["kv"]
        B = idx.size(0)
        # Resolved once at trace time: `_w1_variant` is a Python value Dynamo guards on, so the
        # per-site choice of kernel is baked into the graph and costs nothing per step.
        sites = self._w1_variant[1] if B == 1 else ()
        i8 = self._w1_variant[3] if B == 1 else ()
        i8form = 3 if self._w1_variant[4] == "triton" else 2
        form = {n: (i8form if n in i8 else 1 if n in sites else 0)
                for n in ("qkvg", "o", "fc", "fc2", "head")}
        cs = dw["cs"].index_select(2, seq.to(torch.int64))
        cos, sin = cs[0], cs[1]
        # All three resolved at trace time from Python values Dynamo guards on, so the mode
        # costs nothing per step. `seq64`/`seqp1` are only materialised by the modes that need
        # them: the append writes at `seq`, and a cache we filled ourselves holds `seq + 1`
        # keys, which is exactly the length flash derives internally when it does the append.
        mode = self.decode_attn_mode
        own_append = "append" in mode
        if own_append:
            seq64 = seq.to(torch.int64)
            seqp1 = seq + 1

        x = norm(self.transformer.wte(idx))
        x0 = x
        kb = dw["kb"]
        # cache_seqlens for the branch-batched call: one contiguous int32 row per branch. One
        # tiny kernel for the whole step, not one per stage.
        seqk = seq.repeat(kb) if kb > 1 else seq
        if own_append:
            seqkp1 = seqk + 1 if kb > 1 else seqp1
        for i, stage in enumerate(self.transformer.h):
            w = dw["layers"][i]
            x = torch.addcmul(x * self.resid_lambdas[i], x0, self.x0_lambdas[i])
            h = norm(x)
            # ONE matvec for the whole stage: k branches' q, k, v (and ve gates) at once. The
            # pack groups rows by role, so each slice reshapes straight into the branch-batched
            # (k, 1, heads, head_dim) layout the single flash call consumes.
            qkvg = mv_site(h, w["qkvg"], w["qkvg_q"], w["qkvg_s"], form["qkvg"])
            oq, ok, ov = kb * qk, kb * qk + kb * kv, kb * qk + 2 * kb * kv
            q = rope_norm_slim(qkvg[..., :oq].reshape(kb, 1, nh, hd), cos, sin)
            k = rope_norm_slim(qkvg[..., oq:ok].reshape(kb, 1, nkv, hd), cos, sin)
            v = qkvg[..., ok:ov].reshape(kb, 1, nkv, hd)
            if "ve" in w:
                ve = F.embedding(idx.view(-1), w["ve"]).view(kb, 1, nkv, hd)
                gate = torch.sigmoid(qkvg[..., ov:])
                v = torch.addcmul(v, gate.reshape(kb, 1, nkv, 1), ve, value=2.0)
            # Appends k,v at cache_seqlens and attends over the whole preallocated cache, so
            # no slice bound depends on the step number. num_splits is read from the module
            # knob, which the sweep at the end of this file pins: the op's fake kernel refuses
            # to trace at the default num_splits=0, so an unpinned value is uncompilable.
            # `scheduler_metadata`, when present, is the hoisted bookkeeping tensor that removes
            # flash's `prepare_varlen_num_blocks` launch from the step.
            if own_append:
                # Our own append, in the same kernels that produced k and v: an in-place
                # index_copy_ at the device-side position, so the graph stays address-bound and
                # nothing here reads the step number on the host.
                kc_list[i].index_copy_(1, seq64, k)
                vc_list[i].index_copy_(1, seq64, v)
                y = fa3.flash_attn_with_kvcache(q, kc_list[i], vc_list[i],
                                                cache_seqlens=seqkp1, causal=True,
                                                window_size=self.window_sizes[i],
                                                num_splits=self.decode_num_splits,
                                                scheduler_metadata=None if sched is None
                                                else sched[i])
            else:
                y = fa3.flash_attn_with_kvcache(q, kc_list[i], vc_list[i], k=k, v=v,
                                                cache_seqlens=seqk, causal=True,
                                                window_size=self.window_sizes[i],
                                                num_splits=self.decode_num_splits,
                                                scheduler_metadata=None if sched is None
                                                else sched[i])
            # The k branches' out-projections are concatenated on the input axis and the k MLPs'
            # matrices are concatenated too, so the whole stage -- attention out + MLP, over all
            # branches -- is three matvecs whose SUM over branches the matmul does for free.
            # Nothing inside a sublayer depends on another branch's output, so there is no
            # serial dependence between branches; `parallel_mlp` decides whether the MLP
            # sublayer reads the stage input or the post-attention residual.
            a = mv_site(y.reshape(1, 1, -1), w["o"], w["o_q"], w["o_s"], form["o"])
            if self.config.parallel_mlp:
                m = mv_site(F.relu(mv_site(h, w["fc"], w["fc_q"], w["fc_s"],
                                           form["fc"])).square(),
                            w["fc2"], w["fc2_q"], w["fc2_s"], form["fc2"])
                x = x + a + m
            else:
                x = x + a
                x = x + mv_site(F.relu(mv_site(norm(x), w["fc"], w["fc_q"], w["fc_s"],
                                               form["fc"])).square(),
                                w["fc2"], w["fc2_q"], w["fc2_s"], form["fc2"])

        # The head lives inside the compiled region, so `norm -> unembed -> fp32 -> softcap` is
        # one matvec plus one fused pointwise kernel and there is no separate full-vocab fp32
        # copy: the matvec result is read once and the softcap epilogue writes the fp32 logits.
        # The reduction form accumulates straight into fp32, so it never materialises a bf16
        # full-vocab tensor at all.
        logits = mv_site(norm(x), dw["lm_head"], dw["lm_head_q"], dw["lm_head_s"],
                         form["head"], torch.float32)
        softcap = 15
        return softcap * torch.tanh(logits / softcap)

    def decode_width1_callable(self):
        """`_decode_width1` behind `torch.compile(dynamic=False)`, built once, lazily.

        Compiled rather than `mode="reduce-overhead"`-ed for the reason in
        `_GraphedDecodeStep`: that mode's static input pool swallows the in-place kv append.
        Plain inductor leaves the cache tensors alone, and that is checked, not assumed, by
        `_validate_width1` before the callable is handed out. Returns None -- meaning "use the
        generic body" -- if the compile raised, if the compiled and uncompiled fused bodies
        disagree, or if the fused body disagrees with the generic one. The reason is kept for
        the diagnostic print, so a demotion is visible in ms and in the log rather than silent.
        """
        if self._w1_building:
            return None                 # validating: the generic body is the reference
        variant = self._w1_variant
        # The attention mode and the split width are baked into the traced body, so they are
        # part of the identity of what was validated: a cache keyed on the matvec variant alone
        # would hand out a "validated" callable for an attention configuration nothing checked.
        key = (variant, self.decode_attn_mode, self.decode_num_splits)
        entry = self._w1_cache.get(key)
        if entry is None:
            self._w1_building = True
            try:
                if self._dw is None:
                    self.build_decode_weights()
                options = W1_AUTOTUNE_OPTIONS if variant[2] else None
                fn = torch.compile(self._decode_width1, dynamic=False, options=options)
                ok, detail = self._validate_width1(fn)
                entry = (fn if ok else None, detail)
            except Exception as exc:        # noqa: BLE001 -- recorded, then paid for in ms
                entry = (None, f"{type(exc).__name__}: {exc}")
            finally:
                self._w1_building = False
            self._w1_cache[key] = entry
        self._w1_reason = f"key={key} :: {entry[1]}"
        self._w1_status = 2 if entry[0] is not None else 3
        return entry[0]

    @torch.no_grad()
    def _validate_width1(self, fn):
        """Three paths, one scratch cache each, compared on next-token distributions.

        The paths are the generic body (the reference `forward` agrees with), the fused
        width-1 body, and its compiled form. Bit equality is the wrong test -- the fused gemv,
        the reassociated rotary and inductor's own scheduling all perturb bf16 rounding, and a
        bit-exact check silently disables a working compile over a 0.125 delta in a stored key.
        Total variation is the quantity the gate is made of. A lost kv append does not look
        like rounding: it drives TV towards 1 and leaves the appended row at zero. Both of
        those are caught; rounding is not.

        The scratch cache is produced by an actual prefill of real tokens, not by `randn`. A
        randn cache holds un-normed keys against a normed query, so the attention softmax is
        far more peaked than anything a real request produces and a 1-ulp bf16 difference
        anywhere flips which key wins. Measured at depth 5 / n_embd 384: on a randn cache the
        eager and the compiled form of the SAME fused body disagreed by TV 0.030 and the check
        disabled a correct compile, costing 227 kernels/step instead of ~60. On a prefilled
        cache the state is the one the gate measures on.
        Threshold: the scored gate is TV <= 0.05 per shape and the launch that pinned
        `append_sched` measured 0.0147 (prompted) / 0.0198 (no prompt) on the real probe, so the
        in-run check is deliberately tighter than the gate but not tighter than the bf16 noise it
        is measuring. At a 2-key context the SAME body eager vs compiled was seen at TV 0.031,
        i.e. a 0.03 cutoff rejects configurations that differ only in rounding (it threw away
        `append_sched`/32, the fastest split width in the isolated probe). 0.04 keeps a real
        margin below the gate; the failure mode that actually matters -- a lost kv append --
        drives TV towards 1 and leaves the appended row exactly zero, and both are still caught.
        """
        cfg = self.config
        dev = self.transformer.wte.weight.device
        steps, slots = 8, 160
        tv_tol = 0.04
        md = self.sched_for(slots)

        def tv_of(a, b):
            n = len(a)
            a = torch.cat(a, 0).float().view(n, -1)
            b = torch.cat(b, 0).float().view(n, -1)
            d = (torch.softmax(a, -1) - torch.softmax(b, -1)).abs().sum(-1)
            return float((0.5 * d).max())

        def round_at(pos, nsteps):
            """Three paths, one scratch cache each, from a real `pos`-token prefill."""
            toks = torch.randint(0, cfg.vocab_size, (pos + nsteps,), device=dev)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                state = self.init_decode_state(batch=1, max_len=slots, graph=False)
                self._decode_body(toks[:pos].view(1, pos), state, prefill=True)
            state["seq"].fill_(pos)
            kc, vc = [t.clone() for t in state["kc"]], [t.clone() for t in state["vc"]]
            kc2, vc2 = [t.clone() for t in state["kc"]], [t.clone() for t in state["vc"]]
            seq = torch.full((1,), pos, dtype=torch.int32, device=dev)
            seq2, seq3 = seq.clone(), state["seq"]
            gen_state = {"seq": seq3, "kc": state["kc"], "vc": state["vc"]}
            gen_logits, ref_logits, got_logits = [], [], []
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                for s in range(nsteps):
                    idx = toks[pos + s].view(1, 1)
                    gen_logits.append(self._decode_body(idx, gen_state, prefill=False))
                    seq3.add_(1)
                    ref_logits.append(self._decode_width1(idx, seq2, kc2, vc2, md))
                    seq2.add_(1)
                    got_logits.append(fn(idx, seq, kc, vc, md))
                    seq.add_(1)
            tv_fuse = tv_of(gen_logits, ref_logits)
            tv_comp = tv_of(ref_logits, got_logits)
            appended = min(kc[i][:, pos + nsteps - 1].abs().sum().item()
                           for i in range(cfg.n_stages))
            finite = bool(torch.cat(got_logits, 0).isfinite().all())
            ok = tv_fuse < tv_tol and tv_comp < tv_tol and appended > 0.0 and finite
            return ok, (f"pos={pos}: TV(generic,fused)={tv_fuse:.5f} "
                        f"TV(fused,compiled)={tv_comp:.5f} appended|sum|={appended:.3e} "
                        f"finite={finite}")

        # Two contexts in the same cache shape, so both are one traced graph. The long one is
        # the scored geometry; the short one is the no-prompt shape AND the only place a stale
        # hoisted split count could show up -- metadata built for a nearly full cache, used on a
        # context of three keys, is the worst case for splits that own no work.
        ok_long, detail_long = round_at(128, steps)
        ok_short, detail_short = round_at(2, 4)
        ok = ok_long and ok_short
        detail = (f"{detail_long} | {detail_short} -> "
                  f"{'compiled' if ok else 'DISABLED (generic body)'}")
        return ok, detail

    # -----------------------------------------------------------------------
    # Prefill: the prompt pass as its own compiled program (node 1.6).
    # -----------------------------------------------------------------------

    def _decode_prefill(self, idx, kc_list, vc_list):
        """The width-T prompt body, over the PACKED decode weights, written for one graph.

        Value-preserving differences from the generic prefill branch of `_decode_body`:

          * every matrix is read in bf16 from `self._dw` instead of being re-cast from the fp32
            parameter on every call -- under autocast with grad off there is no weight-cast
            cache, so the eager prefill materialises a bf16 copy of every weight per request;
          * q/k/v and the value-embedding gate of all branches are ONE gemm over the pre-stacked
            `qkvg` matrix (the gate rows are zero-padded, so the extra columns contribute exactly
            zero), and the branch out-projections and MLPs are single gemms over the
            input-axis-concatenated packs, which is the same sum the eager loop accumulates;
          * the branches' value embeddings are one gather over the feature-concatenated table;
          * the residual lambdas fold into one `addcmul`, and rotary+norm is `rope_norm_slim`;
          * attention reads `k`/`v` directly rather than the cache slice it just wrote them to
            (the same tensors), and the cache write is an ordinary slice assignment that
            inductor puts in the producing kernel's epilogue.

        int8 is deliberately NOT used here: at width 1536 these are gemms, not gemvs, so they
        are arithmetic-bound rather than weight-stream-bound and a dequantised operand would
        only add traffic. The bytes that matter at width 1 are not the bytes that matter here.
        """
        dw = self._dw
        nh, nkv, hd, qk, kv = dw["nh"], dw["nkv"], dw["hd"], dw["qk"], dw["kv"]
        kb = dw["kb"]
        T = idx.size(1)
        cs = dw["cs"][:, :, :T]
        cos, sin = cs[0], cs[1]
        x = norm(self.transformer.wte(idx))
        x0 = x
        for i in range(len(dw["layers"])):
            w = dw["layers"][i]
            x = torch.addcmul(x * self.resid_lambdas[i], x0, self.x0_lambdas[i])
            h = norm(x)
            qkvg = F.linear(h, w["qkvg"])
            oq, ok, ov = kb * qk, kb * qk + kb * kv, kb * qk + 2 * kb * kv
            # (T, branch, head, head_dim) -> (branch, T, head, head_dim): the branch axis is the
            # batch axis of the single flash call, exactly as at width 1.
            # `reshape`, not `view`: a slice of the last axis is strided, so the (T, branch)
            # merge is a copy the fused kernel writes anyway.
            q = qkvg[..., :oq].reshape(T, kb, nh, hd).transpose(0, 1)
            k = qkvg[..., oq:ok].reshape(T, kb, nkv, hd).transpose(0, 1)
            v = qkvg[..., ok:ov].reshape(T, kb, nkv, hd).transpose(0, 1)
            q = rope_norm_slim(q, cos, sin)
            k = rope_norm_slim(k, cos, sin)
            if "ve" in w:
                ve = F.embedding(idx.view(-1), w["ve"]).reshape(T, kb, nkv, hd).transpose(0, 1)
                gate = torch.sigmoid(qkvg[..., ov:].reshape(T, kb, nkv, 1).transpose(0, 1))
                v = torch.addcmul(v, gate, ve, value=2.0)
            else:
                v = v.contiguous()
            kc_list[i][:, :T] = k
            vc_list[i][:, :T] = v
            y = fa3.flash_attn_func(q, k, v, causal=True, window_size=self.window_sizes[i])
            # Branch-major on the feature axis, which is the axis `o` is concatenated on.
            yo = y.transpose(0, 1).reshape(1, T, kb * nh * hd)
            a = F.linear(yo, w["o"])
            if self.config.parallel_mlp:
                m = F.linear(F.relu(F.linear(h, w["fc"])).square(), w["fc2"])
                x = x + a + m
            else:
                x = x + a
                x = x + F.linear(F.relu(F.linear(norm(x), w["fc"])).square(), w["fc2"])
        # Only the last position leaves the prefill, so the head is a gemv over one row: the
        # slice happens BEFORE the unembedding, as in the generic body.
        x = norm(x[:, -1:, :])
        logits = F.linear(x, dw["lm_head"]).float()
        softcap = 15
        return softcap * torch.tanh(logits / softcap)

    def prefill_callable(self, width):
        """`_decode_prefill` behind `torch.compile(dynamic=False)`, one entry per prompt width.

        Returns None -- meaning "use the generic eager prefill" -- if the switch is off, if the
        width is below `_pf_min_width` (the 2- and 128-token prefills the width-1 validation
        does are not worth a retrace), if the compile raised, or if the compiled program fails
        the distributional check in `_validate_prefill`.
        """
        if self._pf_mode == "eager" or self._pf_building or width < self._pf_min_width:
            return None
        entry = self._pf_cache.get(width)
        if entry is None:
            self._pf_building = True
            try:
                if self._dw is None:
                    self.build_decode_weights()
                fn = torch.compile(self._decode_prefill, dynamic=False)
                ok, detail = self._validate_prefill(fn, width)
                entry = (fn if ok else None, detail)
            except Exception as exc:    # noqa: BLE001 -- recorded, then paid for in ms
                import traceback
                traceback.print_exc()
                entry = (None, f"{type(exc).__name__}: {exc}")
            finally:
                self._pf_building = False
            self._pf_cache[width] = entry
            print(f"[pf] width={width} compiled={entry[0] is not None} :: {entry[1]}",
                  flush=True)
        self._pf_status = 2 if entry[0] is not None else 3
        self._pf_reason = f"width={width} :: {entry[1]}"
        return entry[0]

    def prefill_graph_for(self, state, width):
        """The captured prefill graph for this state and width, or None.

        A request issues exactly one prefill, so the launch cost of its ~100 kernels is paid
        once rather than 512 times -- which is why this is measured rather than assumed to pay.
        Like the width-1 graph, it is bound to the state's cache addresses and therefore stored
        ON the state; `graph_enabled=False` (the cache-bytes probe) never captures, so no graph
        pool is ever charged as cache. The recorded computation is the SAME validated compiled
        callable the non-graphed path calls.
        """
        if self._pf_mode != "graph" or not state.get("graph_enabled", True):
            return None
        graphs = state.setdefault("pf_graphs", {})
        if width not in graphs:
            pg = _GraphedPrefill(self, state, width)
            graphs[width] = pg if pg.captured else None
        return graphs[width]

    @torch.no_grad()
    def _validate_prefill(self, fn, width, steps=4):
        return self._compare_prefill(
            lambda idx, state: fn(idx, state["kc"], state["vc"]), width, steps)

    @torch.no_grad()
    def _compare_prefill(self, run, width, steps=4):
        """Compare the compiled prefill against the eager one, distributionally, on the state it
        leaves behind as well as on the logits it returns.

        Two failure modes are distinguishable here and both matter: a prefill that computes the
        right last-position logits but writes the caches wrongly (or not at all) is only visible
        in the STEPS that follow, so a few generic width-1 steps are run on each of the two
        states and their next-token distributions compared too. Bit equality is the wrong test
        for the same reason as at width 1 (fused gemms and a reassociated rotary perturb bf16
        rounding), so the quantity is total variation, with the same 0.04 margin under the
        scored 0.05 gate.
        """
        cfg = self.config
        dev = self.transformer.wte.weight.device
        tv_tol = 0.04
        toks = self._pf_val_tokens
        if toks is None or toks.numel() < width + steps:
            toks = torch.randint(0, cfg.vocab_size, (width + steps,), device=dev)
        toks = toks.view(-1)[:width + steps].to(dev)
        # The scored probe allocates 2049 slots, so validating at that cache length warms the
        # width-1 body's trace for the shape the request will use instead of adding a new one.
        slots = max(2049, width + steps + 1)

        def tv_of(a, b):
            n = len(a)
            a = torch.cat(a, 0).float().view(n, -1)
            b = torch.cat(b, 0).float().view(n, -1)
            d = (torch.softmax(a, -1) - torch.softmax(b, -1)).abs().sum(-1)
            return float((0.5 * d).max())

        outs, states = [], []
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            for candidate in (False, True):
                # The candidate may want to capture a graph against the state it is handed, so
                # its state allows capture; the reference never captures anything.
                state = self.init_decode_state(batch=1, max_len=slots, graph=candidate)
                if candidate:
                    logits = run(toks[:width].view(1, width), state)
                else:
                    logits = self._decode_body(toks[:width].view(1, width), state,
                                               prefill=True)
                state["seq"].fill_(width)
                outs.append([logits[:, -1, :]])
                states.append(state)
            for s in range(steps):
                idx = toks[width + s].view(1, 1)
                for j, state in enumerate(states):
                    outs[j].append(self._decode_body(idx, state, prefill=False)[:, -1, :])
                    state["seq"].add_(1)
        tv = tv_of(outs[0], outs[1])
        kdiff = max(float((states[0]["kc"][i][:, :width] -
                           states[1]["kc"][i][:, :width]).abs().max())
                    for i in range(cfg.n_stages))
        kscale = max(float(states[0]["kc"][i][:, :width].abs().max())
                     for i in range(cfg.n_stages))
        written = min(float(states[1]["kc"][i][:, width - 1].abs().sum())
                      for i in range(cfg.n_stages))
        finite = bool(torch.cat(outs[1], 0).isfinite().all())
        ok = tv < tv_tol and written > 0.0 and finite and kdiff <= 0.05 * max(kscale, 1e-6)
        del states, outs
        gc.collect()
        return ok, (f"TV(eager,candidate over prefill+{steps} steps)={tv:.5f} "
                    f"max|dk|={kdiff:.4f} (|k|max {kscale:.3f}) "
                    f"written|sum|={written:.3e} finite={finite}")

    def decode_step(self, idx, state):
        """`logits, state = model.decode_step(idx, state)`, logits for the LAST position only.

        Width > 1 is the prefill and runs once per request; width 1 is the step, and it is
        the call the graph replays. `seq` is advanced on both paths, so the tensor is the
        single source of the position.
        """
        if idx.size(1) > 1:
            pg = self.prefill_graph_for(state, idx.size(1))
            if pg is not None:
                logits = pg.replay(idx)
            else:
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
        self._install_fast_step(graph, state)
        return graph.replay(idx), state

    def _install_fast_step(self, graph, state):
        """Publish a near-zero-Python replay path as `decode_step` itself, bound to `state`.

        A request is 512 strictly sequential calls, so every microsecond of host work per call
        is half a millisecond of request time -- and if the host falls behind, the GPU idles
        between replays. The generic path pays, per call, a `torch.compile` wrapper's
        `__getattr__` forward, a bound-method creation, a `.size(1)` dispatch, two dict
        lookups, two attribute chains, a nested method call and a tuple build. None of that
        depends on the step, so all of it is resolved once, here:

          * the two device ops become pre-bound callables (`static_idx.copy_`, `graph.replay`),
          * the return value becomes one preallocated tuple (both elements are fixed for the
            life of the state: the logits buffer is the graph's output and `state` is itself),
          * the closure is written into the instance dict of the model AND of every host the
            instrument looks the method up on, so the lookup is a plain dict hit.

        The guard is `state is` the state this was captured against plus the width test, and
        anything else falls back to the full path -- so a second state, a prefill and the
        `graph=False` cache probe all still work, and the state stays the owner of the graph.
        """
        slow = GPT.decode_step.__get__(self, GPT)
        out = (graph.static_logits, state)

        def fast_step(idx, st, _copy=graph.static_idx.copy_, _replay=graph.graph.replay,
                      _out=out, _state=state, _slow=slow):
            if st is _state and idx.size(1) == 1:
                _copy(idx)
                _replay()
                return _out
            return _slow(idx, st)

        self.__dict__["decode_step"] = fast_step
        for host in self._ds_hosts:
            host.__dict__["decode_step"] = fast_step
        self._fast_step = fast_step


class _GraphedDecodeStep:
    """A manually captured `torch.cuda.CUDAGraph` over one cached decode step.

    What is captured is the SAME `torch.compile(dynamic=False)` callable the eager path calls
    (`GPT.decode_width1_callable`), so the graph is a recording of the eager computation and
    `graph` only chooses which of the two issues it.

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
            self.graph, self.static_logits = None, None
        # Every capture is announced: a capture that fails inside the SCORED probe (which runs
        # after every print in this file, in prepare.py) is otherwise invisible and shows up
        # only as milliseconds. `_capture` runs the compiled callable first, so a compile that
        # was demoted for this state's shapes is visible here too.
        print(f"[cap] slots={state['kc'][0].size(1)} captured={self.captured} "
              f"variant={model._w1_variant[0]} w1_status={model._w1_status} "
              f"reason={self.reason!r}", flush=True)

    def _advance(self):
        logits = self.model._decode_body(self.static_idx, self.state, prefill=False)
        self.state["seq"].add_(1)
        return logits

    def _capture(self):
        seq0 = self.state["seq"].clone()
        # Compilation runs on the default stream, before the side stream and well before the
        # capture: inductor's first call compiles, autotunes and allocates, none of which
        # belongs inside a capture, and Dynamo's guards then hit on every later call.
        self._advance()
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


class _GraphedPrefill:
    """A manually captured CUDA graph over the compiled prefill program (node 1.6).

    The prompt width is fixed (the probe's 1536) and the program writes the caches in place, so
    the whole prompt pass is one replay: the prompt tokens are copied into a static input buffer
    and every kernel launch, every guard and every allocation is pre-recorded. Captured by hand
    for the same reason as the width-1 step -- `mode="reduce-overhead"` would copy the cache
    tensors into its own pool and the prefill's cache writes would land in the copy.

    The recorded callable is the one `_validate_prefill` already gated, and the graph itself is
    re-validated distributionally by `_compare_prefill` before the diagnostic pins this mode, so
    a capture that lost the cache writes cannot reach the scored probe.
    """

    WARMUP = 2

    def __init__(self, model, state, width):
        self.captured = False
        self.reason = ""
        self.graph = None
        self.static_logits = None
        dev = state["seq"].device
        self.static_idx = torch.zeros(1, width, dtype=torch.int64, device=dev)
        fn = model.prefill_callable(width)
        if fn is None:
            self.reason = "no compiled prefill program"
        else:
            try:
                seq0 = state["seq"].clone()
                fn(self.static_idx, state["kc"], state["vc"])   # compile on the default stream
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(self.WARMUP):
                        fn(self.static_idx, state["kc"], state["vc"])
                torch.cuda.current_stream().wait_stream(stream)
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):
                    self.static_logits = fn(self.static_idx, state["kc"], state["vc"])
                state["seq"].copy_(seq0)
                self.captured = True
            except Exception as exc:        # noqa: BLE001 -- recorded, then paid for in ms
                self.reason = f"{type(exc).__name__}: {exc}"
                self.graph, self.static_logits = None, None
        print(f"[cap-pf] width={width} slots={state['kc'][0].size(1)} "
              f"captured={self.captured} reason={self.reason!r}", flush=True)

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
WINDOW_PATTERN = "SSLL" # sliding window pattern: L=full, S=half context
MLP_RATIO = 5.0         # MLP hidden per BRANCH (k branches sum, so the stage's MLP is k*this)
VE_LAYERS = (0, 1, 2, 3)  # value-embedding blocks; branches of a stage must agree (see pack)
# Explicit model width. ASPECT_RATIO would give depth*64 rounded up to a HEAD_DIM multiple,
# which at low depth picks the width by accident; None keeps the derived value.
MODEL_DIM = 512
# Grouped-query attention: number of KV heads. None = one per query head (MHA). Fewer KV heads
# shrink c_k/c_v (bytes the step streams), the kv cache (bytes the step reads) AND the value
# embeddings (which are free at decode time, so shrinking them is pure capacity loss).
N_KV_HEAD = None
DECODE_NUM_SPLITS = 16  # split-K width of the cached-decode attention kernel (see sweep below)

# Optimization
TOTAL_BATCH_SIZE = 2**18 # tokens per optimizer step (node 5.1 recipe, from trunk)
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.75    # fraction of time budget for LR warmdown (node 5.1, from trunk)
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# Model size
# PARALLEL-BRANCH TOPOLOGY (node 3.5): the serial chain is STAGES long, the capacity is
# STAGES*BRANCHES blocks. 2x2 = capacity of depth 4 at the serial latency of depth 2. MLP_RATIO
# is dropped from 6.0 to 4.0 so that four branches hold exactly the parameter count, the decode
# weight bytes and the per-token FLOPs of the trunk's three fat serial layers: the ONLY thing
# this configuration changes against the trunk is the topology.
STAGES = 2              # serial stages (what one decode step pays serially)
# False = the MLP branches read norm(x + attention): a stage is two SEQUENTIAL sublayers,
# but the k branches inside each sublayer are still parallel, so the stage still issues
# ONE batched flash call and the branch projections are still single merged matvecs.
PARALLEL_MLP = False
BRANCHES = 2            # parallel branches per stage (one flash call at batch=BRANCHES)
DEPTH = STAGES * BRANCHES   # total attention+MLP blocks (capacity)
# 64 instead of 128: the fat MLP raises per-layer training activations, and peak_vram_bytes has
# zero slack against the reference. Halving the micro-batch keeps the training peak below it;
# the math is unchanged (2x grad accumulation at the same TOTAL_BATCH_SIZE) and the frozen
# 128x2048 validation forward is untouched.
DEVICE_BATCH_SIZE = 64  # per-device batch size (reduce if OOM)

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
    base_dim = MODEL_DIM if MODEL_DIM is not None else depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    kv_heads = N_KV_HEAD if N_KV_HEAD is not None else num_heads
    assert depth % BRANCHES == 0
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=kv_heads, n_embd=model_dim,
        n_stages=depth // BRANCHES, n_branch=BRANCHES, parallel_mlp=PARALLEL_MLP,
        window_pattern=WINDOW_PATTERN,
        mlp_ratio=MLP_RATIO,
        ve_layers=tuple(i for i in VE_LAYERS if i < depth) if VE_LAYERS else None,
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

# Budget diagnostics for the depth experiment: everything the ceilings are read against.
REF_PARAMS, REF_FLOPS = 50_332_176, 239_078_400
print(f"[mark] NODE-3.5-PARALLEL-BRANCHES parallel_mlp={PARALLEL_MLP}")
print(f"[arch] blocks={config.n_layer} stages={config.n_stages} branches={config.n_branch} "
      f"n_embd={config.n_embd} n_head={config.n_head} "
      f"head_dim={config.n_embd // config.n_head} mlp_hidden={int(round(MLP_RATIO * config.n_embd))} "
      f"ve_layers={sorted(int(k) for k in model.value_embeds)} "
      f"stage_windows={[w[0] for w in model.window_sizes]} device_batch={DEVICE_BATCH_SIZE}")
print(f"[arch] params={num_params:,} ({num_params - REF_PARAMS:+,} vs ceiling) "
      f"analytic_flops={num_flops_per_token:,} ({num_flops_per_token - REF_FLOPS:+,} vs ceiling)")
_kv_bytes = (2 * config.n_layer * 2049 * config.n_kv_head *
             (config.n_embd // config.n_head) * 2)
print(f"[arch] kv cache for a 2049-slot batch-1 request: {_kv_bytes / 1e6:.2f} MB")

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
# The instrument holds the compiled wrapper and looks `decode_step` up on it once per step,
# which is a `__getattr__` forward plus a bound-method creation 512 times per request. Telling
# the model about the wrapper lets the captured fast step be published on it directly.
getattr(model, "_orig_mod", model)._ds_hosts = [model]

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

# Peak memory of TRAINING only, read before validation runs (report_efficiency_metrics reads
# max_memory_allocated over training+validation and the ceiling has zero slack). Never reset.
print(f"[diag] steps={step} tokens={tokens_consumed:,} training_seconds={total_training_time:.1f}")
print(f"[diag] peak_vram after training: {torch.cuda.max_memory_allocated():,} bytes "
      f"(reference ceiling 47,198,976,512)")

# ---------------------------------------------------------------------------
# Decode diagnostics. (1) A joint sweep of the attention mode (how many launches per layer the
# flash call costs) and the split-K width, both re-measured at THIS geometry against the
# compiled body. (2) The kernel-name histogram of one captured width-1 step, which is how the
# remaining launches per step are found: at width 1 every launch costs ~2.6 us no matter how
# little work it does. (3) An occupancy probe that prices the attention call in isolation
# against a hypothetical depth-batched one. Batch 1 throughout, so nothing here can move
# peak_vram off the training peak.
# ---------------------------------------------------------------------------

DIAG_ATTN_MODES = ("kvcache", "sched", "append", "append_sched")
DIAG_SPLITS = (1, 2, 4, 8, 16, 32, 64)
DIAG_POSITIONS = (1, 1536, 2047)
DIAG_REPS = 9
DIAG_STEPS = 16
# Launch A measured which matvecs autotuning already serves well: after it, the attention
# out-projection (12.2 us/step) and the MLP down-projection (33.1 us/step, the largest single
# gemm in the step) were still cuBLAS gemv calls at ~200 and ~560 GB/s, while q/k/v, the MLP
# up-projection and the head had moved to triton templates. So the variants tried here are the
# incumbent plus the reduction form on those two holdouts, on all five sites, and in between.
W1_VARIANTS = (
    ("linear", (), True, (), "reduce"),
    ("reduce_fc2", ("fc2",), True, (), "reduce"),
    ("reduce_fc2_o", ("fc2", "o"), True, (), "reduce"),
    ("reduce_fc2_o_head", ("fc2", "o", "head"), True, (), "reduce"),
    ("reduce_all", ("qkvg", "o", "fc", "fc2", "head"), True, (), "reduce"),
)

# Phase 2 of the sweep: int8 site sets, composed on top of whichever bf16 variant phase 1 pins.
# Ordered smallest-to-largest so the ledger shows what each extra quantised site buys. The MLP
# pair comes first because fc/fc2 are the two largest matrices in the step; the head comes last
# and alone in the full set because it is the site whose bytes are already served at >1000 GB/s
# and whose quantisation perturbs the logits most directly, so it must earn its place.
W1_INT8_SETS = (
    ("fc2",),
    ("fc", "fc2"),
    ("fc", "fc2", "qkvg", "o"),
    ("fc", "fc2", "qkvg", "head"),
    ("fc", "fc2", "qkvg", "o", "head"),
)


def sweep_decode_attn(raw):
    """Time (attention mode, num_splits) on captured graphs and pin the fastest validated pair.

    The trunk step spends 63 of its 150 us in attention and only 42.5 of those in the kernel
    that does the arithmetic: 13.7 us is the split-K combine and 7.0 us is
    `prepare_varlen_num_blocks`. This sweep is the joint version of the two knobs that move
    those, because they interact: hoisting the metadata is worth nothing if the winning split
    width is 1 (no dynamic split, no combine), and a small split width is only affordable if the
    main kernel does not collapse without the extra CTAs.

    Two phases so the grid stays inside a sane number of inductor retraces: all modes at the
    incumbent split width, then the split width re-swept -- at THIS geometry, dim 512 / 4 heads
    / depth 3, not the 3x larger one the pinned 16 came from -- for the best mode and for the
    trunk mode. Every pair is validated distributionally (`_validate_width1`, which now also
    checks a 3-key context in a full-length cache) before it is timed; a pair that fails
    validation or capture is never timed and so can never be pinned.

    Returns ({(mode, splits, pos): ms}, {(mode, splits): status}).
    """
    timings, status = {}, {}
    raw.eval()
    orig = (raw.decode_attn_mode, raw.decode_num_splits)
    idx = torch.zeros(1, 1, dtype=torch.int64, device=device)

    def measure(pairs):
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            # +DIAG_STEPS of slack above 2048 so replays past position 2047 stay in bounds
            state = raw.init_decode_state(batch=1, max_len=2049 + DIAG_STEPS)
            for pair in pairs:
                if pair in status:
                    continue
                raw.decode_attn_mode, raw.decode_num_splits = pair
                t0 = time.time()
                try:
                    fn = raw.decode_width1_callable()
                except Exception as exc:                    # noqa: BLE001
                    status[pair] = f"callable FAILED({type(exc).__name__}: {exc})"
                    continue
                build_s = time.time() - t0
                if fn is None:
                    status[pair] = f"unvalidated ({raw._w1_reason})"
                    continue
                graph = _GraphedDecodeStep(raw, state)
                if not graph.captured:
                    status[pair] = f"capture FAILED({graph.reason})"
                    del graph
                    continue
                status[pair] = f"ok build={build_s:.0f}s"
                for pos in DIAG_POSITIONS:
                    samples = []
                    for _ in range(DIAG_REPS):
                        state["seq"].fill_(pos)
                        torch.cuda.synchronize()
                        t1 = time.perf_counter()
                        for _ in range(DIAG_STEPS):
                            graph.replay(idx)
                        torch.cuda.synchronize()
                        samples.append((time.perf_counter() - t1) * 1000.0 / DIAG_STEPS)
                    samples.sort()
                    timings[(pair[0], pair[1], pos)] = samples[len(samples) // 2]
                del graph
            del state
        gc.collect()
        torch.cuda.empty_cache()

    def long_ms(pair):
        keys = [(pair[0], pair[1], p) for p in (1536, 2047)]
        if not all(k in timings for k in keys):
            return float("inf")
        return sum(timings[k] for k in keys)

    phase1 = [(mode, DECODE_NUM_SPLITS) for mode in DIAG_ATTN_MODES]
    measure(phase1)
    best_mode = min(DIAG_ATTN_MODES, key=lambda m: long_ms((m, DECODE_NUM_SPLITS)))
    phase2 = [(best_mode, ns) for ns in DIAG_SPLITS]
    if best_mode != "kvcache":
        phase2 += [("kvcache", ns) for ns in DIAG_SPLITS]
    measure(phase2)
    raw.decode_attn_mode, raw.decode_num_splits = orig
    return timings, status


raw_model = getattr(model, "_orig_mod", model)
print("[mark] NODE-2.3-ONE-ATTENTION-LAUNCH")
try:
    t_sweep = time.time()
    diag_timings, diag_status = sweep_decode_attn(raw_model)
    print(f"[diag] attention-mode x split-K sweep took {time.time() - t_sweep:.0f}s")
    print("[diag] cached-decode step latency, batch 1, median of "
          f"{DIAG_REPS} x {DIAG_STEPS} replays (ms/step)")
    print("[diag] mode / num_splits | " +
          " | ".join(f"pos={p}" for p in DIAG_POSITIONS) + " | status")
    pairs = sorted(diag_status, key=lambda p: (p[0], p[1]))
    for pair in pairs:
        cells = " | ".join(
            f"{diag_timings[(pair[0], pair[1], p)]:.3f}"
            if (pair[0], pair[1], p) in diag_timings else "  -  " for p in DIAG_POSITIONS)
        print(f"[diag] {pair[0]:>13} / {pair[1]:<4} | {cells} | {diag_status[pair]}")
    # The scored request prefills 1536 and then decodes positions 1536..2047, so rank on the
    # long-context end; print the short-context ranking too so the trade-off is visible.
    def rank(positions):
        usable = [p for p in pairs
                  if all((p[0], p[1], q) in diag_timings for q in positions)]
        return sorted(usable, key=lambda p: sum(diag_timings[(p[0], p[1], q)] for q in positions))
    long_rank = rank((1536, 2047))
    print(f"[diag] ranking at long context (1536,2047): {long_rank[:6]}")
    print(f"[diag] ranking at short context (1):        {rank((1,))[:6]}")
    print(f"[diag] ranking over all positions:          {rank(DIAG_POSITIONS)[:6]}")
    if long_rank:
        raw_model.decode_attn_mode, raw_model.decode_num_splits = long_rank[0]
        print(f"[diag] pinning attn mode={raw_model.decode_attn_mode} "
              f"decode_num_splits={raw_model.decode_num_splits} for the timed probe")
    else:
        print("[diag] nothing validated+captured; leaving "
              f"mode={raw_model.decode_attn_mode} splits={raw_model.decode_num_splits}")
except Exception as exc:
    import traceback
    print(f"[diag] sweep failed: {type(exc).__name__}: {exc}")
    traceback.print_exc()
    raw_model.decode_attn_mode = DECODE_ATTN_MODE
    raw_model.decode_num_splits = DECODE_NUM_SPLITS


def sweep_w1_variants(raw, variants=W1_VARIANTS):
    """Time every width-1 body variant on a captured graph and pin the fastest.

    Same shape as the split-K sweep and for the same reason: the fastest matvec form for a
    batch-1 gemv is a property of the machine, not of the source, so it is measured in-run
    rather than asserted. A variant whose compile, agreement check or capture fails simply
    never wins, so the worst case is the trunk configuration.
    """
    timings, status = {}, {}
    raw.eval()
    idx = torch.zeros(1, 1, dtype=torch.int64, device=device)
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        state = raw.init_decode_state(batch=1, max_len=2049 + DIAG_STEPS)
        for variant in variants:
            raw._w1_variant = variant
            t0 = time.time()
            fn = raw.decode_width1_callable()
            build_s = time.time() - t0
            if fn is None:
                status[variant] = f"no compiled body ({raw._w1_reason})"
                continue
            graph = _GraphedDecodeStep(raw, state)
            if not graph.captured:
                status[variant] = f"capture FAILED({graph.reason})"
                del graph
                continue
            status[variant] = f"ok build={build_s:.0f}s {raw._w1_reason}"
            for pos in (1536, 2047):
                samples = []
                for _ in range(DIAG_REPS):
                    state["seq"].fill_(pos)
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    for _ in range(DIAG_STEPS):
                        graph.replay(idx)
                    torch.cuda.synchronize()
                    samples.append((time.perf_counter() - t1) * 1000.0 / DIAG_STEPS)
                samples.sort()
                timings[(variant, pos)] = samples[len(samples) // 2]
            del graph
        del state
    gc.collect()
    torch.cuda.empty_cache()
    return timings, status


def tune_int8_gemv(raw, reps=7, inner=32):
    """Pick (BLOCK_M, BLOCK_K, num_warps) per decode-weight shape, on THIS machine.

    Run before any compile or capture: the winner is a Python constant the traced body bakes
    in, and the JIT warm-up happens here on the default stream, so no triton compilation or
    autotuning can occur inside a CUDA graph. Each candidate is timed as `inner` back-to-back
    launches inside a graph, so the ~2.6 us launch cost of a 4 us kernel is out of the number.
    """
    dw = raw._dw if raw._dw is not None else raw.build_decode_weights()
    shapes = {}
    for w in dw["layers"]:
        for name in ("qkvg", "o", "fc", "fc2"):
            shapes[tuple(w[name + "_q"].shape)] = (w[name + "_q"], w[name + "_s"])
    shapes[tuple(dw["lm_head_q"].shape)] = (dw["lm_head_q"], dw["lm_head_s"])
    with torch.no_grad():
        for (m, k), (q, sc) in shapes.items():
            x = torch.zeros(1, 1, k, dtype=torch.bfloat16, device=q.device)
            y = torch.empty((m,), dtype=torch.bfloat16, device=q.device)
            xf = x.reshape(-1)
            best, results = None, []
            bks = sorted({min(b, triton.next_power_of_2(k)) for b in (256, 512, 1024)})
            for bm in (1, 2, 4, 8, 16, 32):
                for bk in bks:
                    for warps in (2, 4, 8):
                        # The accumulator is a BLOCK_M x BLOCK_K fp32 tile; past ~8k elements it
                        # spills and the config cannot win, so it is not worth a compile.
                        if bm * bk > 8192:
                            continue
                        grid = (triton.cdiv(m, bm),)
                        try:
                            for _ in range(3):
                                _int8_gemv_kernel[grid](q, sc, xf, y, m, k, BLOCK_M=bm,
                                                        BLOCK_K=bk, num_warps=warps)
                            torch.cuda.synchronize()
                            g = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(g):
                                for _ in range(inner):
                                    _int8_gemv_kernel[grid](q, sc, xf, y, m, k, BLOCK_M=bm,
                                                            BLOCK_K=bk, num_warps=warps)
                            samples = []
                            for _ in range(reps):
                                torch.cuda.synchronize()
                                t0 = time.perf_counter()
                                g.replay()
                                torch.cuda.synchronize()
                                samples.append((time.perf_counter() - t0) * 1e6 / inner)
                            samples.sort()
                            us = samples[len(samples) // 2]
                            del g
                        except Exception:               # noqa: BLE001 -- config simply loses
                            continue
                        results.append((us, (bm, bk, warps)))
                        if best is None or us < best[0]:
                            best = (us, (bm, bk, warps))
            if best is not None:
                _INT8_TILING[(m, k)] = best[1]
                results.sort()
                top = " ".join(f"{cfg}:{us:.2f}" for us, cfg in results[:4])
                print(f"[i8tune] ({m},{k}) -> BLOCK_M,BLOCK_K,warps={best[1]} at "
                      f"{best[0]:.2f} us ({len(results)} configs) | top: {top}")
    gc.collect()
    torch.cuda.empty_cache()


try:
    t0 = time.time()
    tune_int8_gemv(raw_model)
    print(f"[i8tune] tiling table for {len(_INT8_TILING)} shapes in {time.time() - t0:.0f}s")
except Exception as exc:                               # noqa: BLE001
    import traceback
    print(f"[i8tune] failed: {type(exc).__name__}: {exc}")
    traceback.print_exc()


def report_w1_sweep(variants, timings, status, tag):
    print(f"[w1] {tag}: ms/step, median of {DIAG_REPS} x {DIAG_STEPS} replays")
    for variant in variants:
        cells = " | ".join(f"{timings[(variant, p)]:.3f}" if (variant, p) in timings
                           else "  -  " for p in (1536, 2047))
        print(f"[w1] {variant[0]:>26} | {cells} | {status.get(variant, 'not reached')}")
    usable = [v for v in variants if (v, 1536) in timings and (v, 2047) in timings]
    usable.sort(key=lambda v: timings[(v, 1536)] + timings[(v, 2047)])
    return usable


print("[mark] NODE-6.1-INT8-DECODE-WEIGHTS-C (tuned triton int8 gemv tiling)")
try:
    var_timings, var_status = sweep_w1_variants(raw_model, W1_VARIANTS)
    usable = report_w1_sweep(W1_VARIANTS, var_timings, var_status, "phase 1 (bf16 pack)")
    best_bf16 = usable[0] if usable else W1_VARIANT_DEFAULT
    BEST_BF16_VARIANT = best_bf16
    print(f"[w1] phase-1 ranking: {[v[0] for v in usable]} -> best bf16 {best_bf16[0]}")
    # Phase 2: the same body with int8 decode weights at a growing set of sites, composed on
    # top of the bf16 variant phase 1 pinned, so the only difference measured here is the
    # precision of the weight stream. A set whose distributional check (TV against the generic
    # bf16 body on a real prefilled cache) fails never produces a callable and therefore never
    # wins: the demotion to bf16 is automatic and visible in the status column.
    int8_variants = tuple((f"int8_{eng}_{'_'.join(s)}", best_bf16[1], best_bf16[2], s, eng)
                          for eng in ("reduce", "triton") for s in W1_INT8_SETS)
    i8_timings, i8_status = sweep_w1_variants(raw_model, int8_variants)
    i8_usable = report_w1_sweep(int8_variants, i8_timings, i8_status, "phase 2 (int8 sites)")
    print(f"[w1] phase-2 ranking: {[v[0] for v in i8_usable]}")
    all_timings = {**var_timings, **i8_timings}
    allv = usable + i8_usable
    allv.sort(key=lambda v: all_timings[(v, 1536)] + all_timings[(v, 2047)])
    W1_RANKED = allv
    raw_model._w1_variant = allv[0] if allv else W1_VARIANT_DEFAULT
    print(f"[w1] overall ranking: {[(v[0], round(all_timings[(v, 1536)], 4)) for v in allv]}")
    print(f"[w1] pinning width-1 variant {raw_model._w1_variant} for the timed probe")
except Exception as exc:
    import traceback
    print(f"[w1] variant sweep failed: {type(exc).__name__}: {exc}")
    traceback.print_exc()
    raw_model._w1_variant = W1_VARIANT_DEFAULT
    BEST_BF16_VARIANT = W1_VARIANT_DEFAULT
    W1_RANKED = [W1_VARIANT_DEFAULT]


FLOOR_REPS = 64


def decode_floor_diagnostic(raw, wrapper, pos=1792, reps=FLOOR_REPS):
    """Split one decode step into host-side microseconds and GPU-side microseconds.

    Three quantities, all per step:
      * host: the time to ISSUE a call, measured with the device queue absorbing the work, so
        it is pure Python/dispatch cost. Measured for the call as the instrument makes it
        (`wrapper.decode_step(idx, state)`), for the bare `graph.replay(idx)` wrapper, and for
        the bare `CUDAGraph.replay` with no Python around it.
      * gpu: cuda-event elapsed time across a back-to-back replay burst, i.e. the device
        timeline including whatever gaps the host leaves.
      * wall: the synchronised per-call time the probe actually pays.
    host << wall means the floor is on the device and no amount of Python removal moves it;
    host ~ wall means the GPU is idling on the host between replays.
    """
    idx = torch.zeros(1, 1, dtype=torch.int64, device=device)
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        state = raw.init_decode_state(batch=1, max_len=2049 + reps + 8)
        wrapper.decode_step(idx, state)             # builds the graph, installs the fast step
        graph = state["graph"]
        print(f"[floor] capture={graph.captured} variant={raw._w1_variant} "
              f"num_splits={raw.decode_num_splits} mode={raw.decode_attn_mode} "
              f"fast_step_on_model={'decode_step' in raw.__dict__} "
              f"fast_step_on_wrapper={'decode_step' in wrapper.__dict__}")
        if not graph.captured:
            del state
            return

        def host_us(fn):
            """Issue cost only: the queue is deep enough that the host never waits."""
            state["seq"].fill_(pos)
            fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            elapsed = time.perf_counter() - t0
            torch.cuda.synchronize()
            return elapsed * 1e6 / reps

        def wall_us(fn):
            state["seq"].fill_(pos)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) * 1e6 / reps

        def gpu_us(fn):
            state["seq"].fill_(pos)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            for _ in range(reps):
                fn()
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) * 1000.0 / reps

        def median_of(fn, measure, n=5):
            vals = sorted(measure(fn) for _ in range(n))
            return vals[n // 2]

        bare = graph.graph.replay
        wrapped = lambda: graph.replay(idx)
        probe = lambda: wrapper.decode_step(idx, state)
        fast = raw.__dict__.get("decode_step")
        rows = [("probe decode_step (fast path)", probe),
                ("graph.replay(idx) wrapper", wrapped),
                ("bare CUDAGraph.replay", bare)]
        print("[floor] per step, us:            host_issue    gpu_events    wall_sync")
        for name, fn in rows:
            print(f"[floor] {name:<30} {median_of(fn, host_us):8.2f} "
                  f"{median_of(fn, gpu_us):13.2f} {median_of(fn, wall_us):12.2f}")
        # What the fast path removed: the same call with the installed closure taken back out,
        # so the generic dict-lookup/attribute-chain path is measured on the same graph.
        if fast is not None:
            raw.__dict__.pop("decode_step", None)
            wrapper.__dict__.pop("decode_step", None)
            print(f"[floor] {'probe decode_step (generic)':<30} "
                  f"{median_of(probe, host_us):8.2f} {median_of(probe, gpu_us):13.2f} "
                  f"{median_of(probe, wall_us):12.2f}")
            raw.__dict__["decode_step"] = fast
            wrapper.__dict__["decode_step"] = fast
        # The installed closure pins this state and its graph, and this state is not the one
        # the probe will use, so drop it: the probe's first width-1 call installs its own.
        raw.__dict__.pop("decode_step", None)
        wrapper.__dict__.pop("decode_step", None)
        raw._fast_step = None
        del graph, state
        state = None
    gc.collect()
    torch.cuda.empty_cache()


try:
    decode_floor_diagnostic(raw_model, model)
except Exception as exc:
    import traceback
    print(f"[floor] diagnostic failed: {type(exc).__name__}: {exc}")
    traceback.print_exc()


def decode_kernel_histogram(raw, pos=1792, reps=8):
    """Kernel-name histogram and per-step time for one captured width-1 step.

    Counts CUDA kernel events inside a cuda-graph replay: at width 1 a launch costs ~2.6 us
    regardless of the trivial work it does, so this histogram is the latency budget itemised,
    and every line of it is a candidate for fusion into its neighbour.
    """
    import collections
    from torch.profiler import profile, ProfilerActivity
    raw.eval()
    idx = torch.zeros(1, 1, dtype=torch.int64, device=device)
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        state = raw.init_decode_state(batch=1, max_len=2049 + reps + 4)
        graph = _GraphedDecodeStep(raw, state)
        print(f"[kern] compile status={raw._w1_status} (2=compiled, 3=disabled) :: "
              f"{raw._w1_reason}")
        print(f"[kern] capture={graph.captured} reason={graph.reason!r} "
              f"num_splits={raw.decode_num_splits} mode={raw.decode_attn_mode}")
        if graph.captured:
            state["seq"].fill_(pos)
            for _ in range(reps):
                graph.replay(idx)
            torch.cuda.synchronize()
            state["seq"].fill_(pos)
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                for _ in range(reps):
                    graph.replay(idx)
                torch.cuda.synchronize()
            cuda_dev = torch.autograd.DeviceType.CUDA
            hist = collections.Counter()
            times = collections.Counter()
            total = 0
            for e in prof.events():
                if e.device_type == cuda_dev:
                    hist[e.key] += 1
                    times[e.key] += getattr(e, "self_device_time_total", 0) or 0
                    total += 1
            print(f"[kern] kernels/step at pos={pos}: {total / reps:.1f} "
                  f"(over {reps} replays, {len(hist)} distinct)")
            for name, count in hist.most_common(60):
                print(f"[kern] {count / reps:7.2f}/step {times[name] / reps:9.2f} us/step  "
                      f"{name[:110]}")
        del graph, state
    gc.collect()
    torch.cuda.empty_cache()


try:
    decode_kernel_histogram(raw_model)
except Exception as exc:
    import traceback
    print(f"[kern] histogram failed: {type(exc).__name__}: {exc}")
    traceback.print_exc()


def attn_occupancy_diagnostic(raw, pos=1792, reps=15, inner=32):
    """Price ONE layer's attention call in isolation, and what depth-batching it would buy.

    Part (c) of this node: the three layers' attention calls cannot actually be merged -- layer
    i+1's query is a function of layer i's output, so the calls are serial in the residual
    stream, not merely adjacent -- but the cost of that serialisation is measurable. A single
    call at batch 1 with 4 query heads occupies a handful of CTAs; the same total arithmetic
    issued as ONE call with the three caches stacked along the batch axis is what a depth-
    parallel model would pay. Both are timed here, on graphs of `inner` back-to-back calls so
    no launch gap and no neighbouring kernel is in the number.

    Every configuration is also timed with each of the sweep's split widths, so the shape of
    the main-kernel-versus-combine trade-off is visible per launch rather than per step.
    """
    cfg = raw.config
    hd = cfg.n_embd // cfg.n_head
    dev = raw.transformer.wte.weight.device
    kw = {"dtype": torch.bfloat16, "device": dev}
    slots = 2049
    md_cache = {}

    def timed(call):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                call()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(inner):
                call()
        samples = []
        for _ in range(reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            g.replay()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1e6 / inner)
        samples.sort()
        del g
        return samples[len(samples) // 2]

    print(f"[occ] isolated flash calls, us per call, cache {slots} slots, seqlen {pos}, "
          f"median of {reps} graphs of {inner}")
    print("[occ]  shape                    splits   us/call   us for the whole step's attention")
    with torch.no_grad():
        for batch, label in ((1, "batch1 (one block)"),
                             (cfg.n_branch, f"batch{cfg.n_branch} (a stage's branches)"),
                             (cfg.n_layer, f"batch{cfg.n_layer} (all blocks)")):
            q = torch.zeros(batch, 1, cfg.n_head, hd, **kw)
            k = torch.zeros(batch, 1, cfg.n_kv_head, hd, **kw)
            v = torch.zeros(batch, 1, cfg.n_kv_head, hd, **kw)
            kc = torch.zeros(batch, slots, cfg.n_kv_head, hd, **kw)
            vc = torch.zeros(batch, slots, cfg.n_kv_head, hd, **kw)
            seq = torch.full((batch,), pos, dtype=torch.int32, device=dev)
            window = raw.window_sizes[-1]
            for splits in DIAG_SPLITS:
                for sched in (False, True):
                    key = (batch, splits)
                    if sched:
                        if key not in md_cache:
                            md_cache[key] = fa3.get_scheduler_metadata(
                                batch, 1, slots, cfg.n_head, cfg.n_kv_head, hd, seq,
                                qkv_dtype=torch.bfloat16, max_seqlen_k_new=1, causal=True,
                                window_size=window, num_splits=splits)
                        md = md_cache[key]
                    else:
                        md = None
                    call = lambda md=md, splits=splits, q=q, k=k, v=v, kc=kc, vc=vc, seq=seq: \
                        fa3.flash_attn_with_kvcache(q, kc, vc, k=k, v=v, cache_seqlens=seq,
                                                    causal=True, window_size=window,
                                                    num_splits=splits, scheduler_metadata=md)
                    try:
                        us = timed(call)
                    except Exception as exc:            # noqa: BLE001
                        print(f"[occ]  {label:<24} {splits:<6}  failed: "
                              f"{type(exc).__name__}: {exc}")
                        continue
                    total = us * (cfg.n_layer // batch)
                    tag = "sched" if sched else "     "
                    print(f"[occ]  {label:<18}{tag} {splits:<6} {us:9.2f} {total:12.2f}")
            del q, k, v, kc, vc, seq
    gc.collect()
    torch.cuda.empty_cache()


try:
    attn_occupancy_diagnostic(raw_model)
except Exception as exc:
    import traceback
    print(f"[occ] diagnostic failed: {type(exc).__name__}: {exc}")
    traceback.print_exc()


def decode_bytes_diagnostic(raw, pos=1792, reps=15, inner=32):
    """The bandwidth ledger of one decode step: bytes streamed per site and GB/s achieved.

    Node 1.3 established the width-1 step is GPU-bound and gap-free, so per-step latency is
    (bytes the step must read) / (bandwidth it achieves). This itemises the numerator. Two
    kinds of bytes are read per step:

      * weight bytes -- every decode matrix is read in full for a single token: the packed
        qkv(+gate) stack, the attention out-projection, both MLP matrices, and the lm_head.
        Equal to 2 * (params that are not the token embedding or a value embedding), because
        those two are a single-row gather per step and cost ~nothing.
      * kv bytes -- one key and one value row per in-window cached position, per layer, which
        is what grouped-query attention shrinks.

    Then two bandwidths: the aggregate (total bytes / measured ms per captured step) and, per
    site, the GB/s an isolated cuBLAS gemv of exactly that shape reaches. The second is the
    one that says whether a narrower matrix buys its bytes back or merely gets a worse kernel.
    """
    cfg = raw.config
    hd = cfg.n_embd // cfg.n_head
    dw = raw._dw if raw._dw is not None else raw.build_decode_weights()
    # Which sites the sweep pinned to int8: those stream one byte per weight plus a per-row
    # fp32 scale instead of two bytes, and the ledger must count the bytes the PINNED body
    # actually reads, not the bf16 pack's.
    i8 = raw._w1_variant[3]
    rows = []
    for i, w in enumerate(dw["layers"]):
        for name in ("qkvg", "o", "fc", "fc2"):
            rows.append((f"L{i}.{name}", name, w[name], w[name + "_q"], w[name + "_s"]))
    rows.append(("lm_head", "head", dw["lm_head"], dw["lm_head_q"], dw["lm_head_s"]))

    def site_bytes(site, t, q, s):
        if site in i8:
            return q.numel() * q.element_size() + s.numel() * s.element_size()
        return t.numel() * t.element_size()

    weight_bytes = sum(site_bytes(site, t, q, s) for _, site, t, q, s in rows)
    bf16_bytes = sum(t.numel() * t.element_size() for _, _, t, _, _ in rows)
    print(f"[bytes] int8 sites pinned by the sweep: {i8 if i8 else '(none)'} -> weight stream "
          f"{bf16_bytes / 1e6:.3f} MB (all bf16) -> {weight_bytes / 1e6:.3f} MB")
    kv_row_bytes = cfg.n_kv_head * hd * 2          # bf16 key row + bf16 value row is 2x this
    kv_bytes = sum(2 * kv_row_bytes * min(w[0], pos) for w in raw.attn_windows)
    emb_bytes = (cfg.n_embd + len(raw.value_embeds) * cfg.n_kv_head * hd) * 2
    total_bytes = weight_bytes + kv_bytes + emb_bytes

    idx = torch.zeros(1, 1, dtype=torch.int64, device=device)
    step_ms = float("nan")
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        state = raw.init_decode_state(batch=1, max_len=2049 + DIAG_STEPS)
        graph = _GraphedDecodeStep(raw, state)
        if graph.captured:
            samples = []
            for _ in range(reps):
                state["seq"].fill_(pos)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(DIAG_STEPS):
                    graph.replay(idx)
                torch.cuda.synchronize()
                samples.append((time.perf_counter() - t0) * 1000.0 / DIAG_STEPS)
            samples.sort()
            step_ms = samples[len(samples) // 2]
        del graph, state

    print(f"[bytes] arch depth={cfg.n_layer} n_embd={cfg.n_embd} n_head={cfg.n_head} "
          f"n_kv_head={cfg.n_kv_head} mlp_hidden={dw['layers'][0]['fc'].size(0)}")
    print(f"[bytes] per decode step at pos={pos}: weights {weight_bytes / 1e6:8.3f} MB in "
          f"{len(rows)} matvecs | kv {kv_bytes / 1e6:.3f} MB | gathers {emb_bytes / 1e3:.1f} kB "
          f"| TOTAL {total_bytes / 1e6:.3f} MB")
    print(f"[bytes] captured step {step_ms:.4f} ms -> aggregate "
          f"{total_bytes / (step_ms * 1e6):.1f} GB/s "
          f"(weights alone {weight_bytes / (step_ms * 1e6):.1f} GB/s)")

    # Per-site, in isolation: a cuda graph of `inner` back-to-back gemvs of exactly this shape,
    # so no launch gap and no neighbouring kernel is in the number. Three forms per site:
    # `F.linear` over bf16 (cuBLAS/autotuned triton, the trunk's incumbent), the inductor bf16
    # reduction, and the inductor INT8 reduction (half the weight bytes, dequantised in
    # registers). The int8 column is the whole question of this node: does halving the bytes of
    # a site halve its microseconds, or does the narrower load simply get a worse kernel?
    # Compiles are cached per shape, and each form is warmed on a side stream before capture so
    # no triton autotuning happens inside a graph.
    print("[bytes]   site              shape            MB/step  us_lin  us_bf16r "
          "us_int8  us_t8i8  GB/s_lin GB/s_bf16r GB/s_int8 GB/s_t8i8")
    cmp_cache = {}

    def timed(fn, args):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(5):
                fn(*args)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(inner):
                fn(*args)
        samples = []
        for _ in range(reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            g.replay()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1e6 / inner)
        samples.sort()
        del g
        return samples[len(samples) // 2]

    with torch.no_grad():
        for label, site, t, q, sc in rows:
            x = torch.zeros(1, 1, t.size(1), dtype=torch.bfloat16, device=t.device)
            mb = t.numel() * t.element_size() / 1e6
            mb8 = (q.numel() * q.element_size() + sc.numel() * sc.element_size()) / 1e6
            key = tuple(t.shape)
            if key not in cmp_cache:
                cmp_cache[key] = (torch.compile(mv_reduce, dynamic=False),
                                  torch.compile(mv_reduce_int8, dynamic=False))
            f_bf16r, f_int8r = cmp_cache[key]
            us = {}
            for name, fn, args in (("lin", F.linear, (x, t)),
                                   ("bf16r", f_bf16r, (x, t)),
                                   ("int8", f_int8r, (x, q, sc)),
                                   ("t8i8", mv_triton_int8, (x, q, sc))):
                try:
                    us[name] = timed(fn, args)
                except Exception as exc:            # noqa: BLE001
                    us[name] = float("nan")
                    print(f"[bytes]   {label} {name} failed: {type(exc).__name__}: {exc}")
            print(f"[bytes]   {label:<14} {str(key):>16} {mb:8.3f} {us['lin']:7.2f} "
                  f"{us['bf16r']:8.2f} {us['int8']:8.2f} {us['t8i8']:8.2f} "
                  f"{mb * 1e6 / (us['lin'] * 1e3):9.1f} {mb * 1e6 / (us['bf16r'] * 1e3):10.1f} "
                  f"{mb8 * 1e6 / (us['int8'] * 1e3):9.1f} "
                  f"{mb8 * 1e6 / (us['t8i8'] * 1e3):9.1f}")
    gc.collect()
    torch.cuda.empty_cache()


print("[mark] NODE-3.3-SHRINK-THE-BYTES")
try:
    decode_bytes_diagnostic(raw_model)
except Exception as exc:
    import traceback
    print(f"[bytes] diagnostic failed: {type(exc).__name__}: {exc}")
    traceback.print_exc()


# ---------------------------------------------------------------------------
# NODE 1.6: where the 15.2 ms between the prompted and the no-prompt request actually goes.
#
# request_ms_median (63.25) - nopref_request_ms_median (48.07) = 15.2 ms, and only two things
# differ between the two shapes: the prompted request pays ONE width-1536 prefill, and it decodes
# at cache length 1536..2047 instead of 1..512. This prices both terms on the trained model:
#   (a) the prefill alone, eager (fp32 params, per-layer unfused op chain) against the compiled
#       program over the packed bf16 weights, median of several synchronised calls;
#   (b) the captured width-1 step as a function of cache length, in BOTH cache geometries the
#       two scored requests allocate (2049 slots and 514 slots).
# Then the two sums are reconstructed and compared against the measured gap, so the split is an
# arithmetic identity rather than an assertion. The faster prefill program is pinned.
# ---------------------------------------------------------------------------

PF_WIDTH = 1536
PF_REPS = 9
PF_POSITIONS_LONG = (1, 256, 512, 1024, 1536, 1792, 2047)
PF_POSITIONS_SHORT = (1, 64, 128, 256, 384, 512)


def prefill_split_diagnostic(raw, tokenizer, width=PF_WIDTH, reps=PF_REPS):
    from prepare import make_dataloader as _mk
    loader = _mk(tokenizer, 1, MAX_SEQ_LEN, "val")
    x, _, _ = next(loader)
    toks = x[0, :MAX_SEQ_LEN].clone()
    del loader, x
    raw._pf_val_tokens = toks
    raw.eval()
    idx_pf = toks[:width].view(1, width)
    idx1 = torch.zeros(1, 1, dtype=torch.int64, device=device)

    # (a) the prefill itself, all three programs, on a state of the scored geometry, through
    # the same `decode_step` entry point the probe uses. Wall time is the synchronised per-call
    # cost the request pays; host time is the issue cost with the queue absorbing the work, so
    # host << wall means the prefill is device-bound and a CUDA-graph capture cannot help it.
    step = GPT.decode_step.__get__(raw, GPT)
    pf_ms, pf_host = {}, {}
    for label in ("eager", "compiled", "graph"):
        raw._pf_mode = label
        samples = []
        try:
            with torch.no_grad(), torch.amp.autocast(device_type="cuda",
                                                     dtype=torch.bfloat16):
                # graph=True only for the mode that wants a capture, so the other two are timed
                # on exactly the trunk path.
                state = raw.init_decode_state(batch=1, max_len=2049,
                                              graph=(label == "graph"))
                for r in range(reps + 3):          # 3 warmups: the probe also warms 3 passes,
                    state["seq"].zero_()           # so a first-call compile is amortised there
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    out = step(idx_pf, state)
                    torch.cuda.synchronize()
                    dt = (time.perf_counter() - t0) * 1000.0
                    del out
                    if r >= 3:
                        samples.append(dt)
                    elif r == 0:
                        print(f"[pf] {label} first call (compile/capture included) {dt:.1f} ms",
                              flush=True)
                # host issue cost: the same call with nothing waiting on the device
                state["seq"].zero_()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(reps):
                    state["seq"].zero_()
                    step(idx_pf, state)
                pf_host[label] = (time.perf_counter() - t0) * 1000.0 / reps
                torch.cuda.synchronize()
                del state
        except Exception as exc:                   # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"[pf] {label} prefill FAILED {type(exc).__name__}: {exc}")
        gc.collect()
        torch.cuda.empty_cache()
        if samples:
            samples.sort()
            pf_ms[label] = samples[len(samples) // 2]
            print(f"[pf] prefill({width}) {label:>8}: wall {pf_ms[label]:.3f} ms "
                  f"host_issue {pf_host.get(label, float('nan')):.3f} ms "
                  f"(median of {len(samples)}, min {samples[0]:.3f} max {samples[-1]:.3f}) "
                  f"status={raw._pf_status} {raw._pf_reason}", flush=True)
    # The graph mode records the already-validated compiled program, but the capture itself can
    # lose an in-place cache write, so it is re-checked distributionally on its own state before
    # it can be pinned -- same TV quantity, same 0.04 margin, same auto-demotion.
    if "graph" in pf_ms:
        raw._pf_mode = "graph"
        try:
            ok, detail = raw._compare_prefill(
                lambda idx, st: _GraphedPrefill(raw, st, width).replay(idx), width)
        except Exception as exc:                   # noqa: BLE001
            import traceback
            traceback.print_exc()
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        print(f"[pf] graph-mode validation ok={ok} :: {detail}", flush=True)
        if not ok:
            pf_ms.pop("graph")
        gc.collect()
        torch.cuda.empty_cache()
    # Pin the fastest program that validated; if nothing beat it, the eager branch is the trunk.
    best = min(pf_ms, key=lambda k: pf_ms[k]) if pf_ms else "eager"
    raw._pf_mode = best
    print(f"[pf] pinning prefill program: {best}", flush=True)

    # Kernel count and device time of one prefill, for the pinned program: at ~77 GFLOP the
    # prefill's arithmetic is ~1 ms of this device, so what is left is launches and traffic.
    try:
        import collections
        from torch.profiler import profile, ProfilerActivity
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            state = raw.init_decode_state(batch=1, max_len=2049, graph=(best == "graph"))
            for _ in range(3):
                state["seq"].zero_()
                step(idx_pf, state)
            torch.cuda.synchronize()
            state["seq"].zero_()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                step(idx_pf, state)
                torch.cuda.synchronize()
            cuda_dev = torch.autograd.DeviceType.CUDA
            hist, times, total = collections.Counter(), collections.Counter(), 0
            for e in prof.events():
                if e.device_type == cuda_dev:
                    hist[e.key] += 1
                    times[e.key] += getattr(e, "self_device_time_total", 0) or 0
                    total += 1
            print(f"[pf] kernels per prefill ({best}): {total} launches, "
                  f"{sum(times.values()):.1f} us of device time, {len(hist)} distinct")
            for name, count in hist.most_common(12):
                print(f"[pf]   {count:4d}x {times[name]:9.1f} us  {name[:100]}")
            del state
    except Exception as exc:                       # noqa: BLE001
        print(f"[pf] prefill profile failed: {type(exc).__name__}: {exc}")
    gc.collect()
    torch.cuda.empty_cache()

    # (b) the captured width-1 step versus cache length, in both scored cache geometries.
    curves = {}
    for tag, slots, positions in (("2049", 2049, PF_POSITIONS_LONG),
                                  ("514", 514, PF_POSITIONS_SHORT)):
        try:
            with torch.no_grad(), torch.amp.autocast(device_type="cuda",
                                                     dtype=torch.bfloat16):
                # +DIAG_STEPS of slack so a burst that starts at the last position stays in
                # bounds; the step reads `cache_seqlens` keys, so the extra slots are not read.
                state = raw.init_decode_state(batch=1, max_len=slots + DIAG_STEPS)
                graph = _GraphedDecodeStep(raw, state)
                if graph.captured:
                    for pos in positions:
                        samples = []
                        for _ in range(reps):
                            state["seq"].fill_(pos)
                            torch.cuda.synchronize()
                            t1 = time.perf_counter()
                            for _ in range(DIAG_STEPS):
                                graph.replay(idx1)
                            torch.cuda.synchronize()
                            samples.append((time.perf_counter() - t1) * 1000.0 / DIAG_STEPS)
                        samples.sort()
                        curves.setdefault(tag, {})[pos] = samples[len(samples) // 2]
                else:
                    print(f"[pf] step curve slots={slots}: capture FAILED {graph.reason!r}")
                del graph, state
        except Exception as exc:                   # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"[pf] step curve slots={slots} FAILED {type(exc).__name__}: {exc}")
        gc.collect()
        torch.cuda.empty_cache()
        if tag in curves:
            cells = " ".join(f"{p}:{curves[tag][p]:.4f}" for p in positions)
            print(f"[pf] captured width-1 step, {tag}-slot cache, ms by cache length: {cells}",
                  flush=True)

    def interp(curve, pos):
        keys = sorted(curve)
        if pos <= keys[0]:
            return curve[keys[0]]
        for a, b in zip(keys, keys[1:]):
            if pos <= b:
                return curve[a] + (curve[b] - curve[a]) * (pos - a) / (b - a)
        return curve[keys[-1]]

    def integral(curve, lo, hi):
        return sum(interp(curve, p) for p in range(lo, hi + 1))

    if "2049" in curves and "514" in curves:
        long_sum = integral(curves["2049"], 1536, 2047)     # 512 steps of the prompted request
        short_sum = integral(curves["514"], 0, 512)         # 513 steps of the no-prompt request
        pf = pf_ms.get(best, float("nan"))
        pf_eager = pf_ms.get("eager", float("nan"))
        print(f"[pf] ACCOUNTING of request({width}+512) - nopref_request(1+512):")
        print(f"[pf]   prefill (pinned program)                      {pf:8.3f} ms")
        print(f"[pf]   512 steps at ctx 1536..2047 (2049-slot cache) {long_sum:8.3f} ms")
        print(f"[pf]   513 steps at ctx 0..512     (514-slot cache)  {short_sum:8.3f} ms")
        print(f"[pf]   long-context step increment                   "
              f"{long_sum - short_sum * 512 / 513:8.3f} ms")
        print(f"[pf]   modelled gap = prefill + increment            "
              f"{pf + long_sum - short_sum * 512 / 513:8.3f} ms "
              f"(with the eager prefill: {pf_eager + long_sum - short_sum * 512 / 513:.3f} ms; "
              f"measured on trunk: 15.18 ms)")
        print(f"[pf]   modelled request  {pf + long_sum:8.3f} ms | modelled nopref_request "
              f"{short_sum:8.3f} ms", flush=True)
    return pf_ms, curves


print("[mark] NODE-1.6-PREFILL-PROGRAM")
try:
    t0 = time.time()
    prefill_split_diagnostic(raw_model, tokenizer)
    print(f"[pf] diagnostic took {time.time() - t0:.0f}s", flush=True)
except Exception as exc:
    import traceback
    print(f"[pf] diagnostic failed: {type(exc).__name__}: {exc}")
    traceback.print_exc()
    raw_model._pf_mode = "eager"


# The isolated per-step sweep above measures a REPLAYED graph on a state the sweep built. The
# score is a whole request through prepare.py's protocol: reset, one width-1536 prefill, then
# 512 width-1 steps on a state whose cache is exactly `window` long. Those are not the same
# measurement -- a variant whose graph captures at 2065 slots and does not at 2049, or whose
# prefill got slower, is fast per replayed step and slow per request. So the request itself is
# measured here, in-run, for the pinned variant AND for the best bf16 variant, with the frozen
# probe from prepare.py. Whatever this ledger says about int8 vs bf16 per request is the thing
# the score is made of, and the faster of the two is what gets pinned for the scored run.
def request_ledger(raw, wrapper, tokenizer, tv_budget=0.042):
    """Price the top candidates on WHOLE REQUESTS, with prepare.py's own probe, and pin.

    Two things the per-step sweep cannot see:
      * the scored number is a request, i.e. one width-1536 prefill plus 512 steps on a cache
        of exactly `window` slots -- a different traced shape from the sweep's state;
      * the scored FIDELITY gate is TV over 513 positions of that request, which is a maximum
        over 64x more positions than the 8-step in-run validation and therefore strictly larger.
        int8 spends fidelity, so the pinned variant must be checked against the gate on the
        quantity the gate is made of, not on a proxy. `tv_budget` keeps a margin under 0.05.
    Candidates: the fastest variant overall, the fastest int8 variant that does NOT quantise the
    lm_head (the site whose error lands directly on the scored distribution), and the fastest
    bf16 variant as the floor. The fastest one inside the budget is pinned.
    """
    from prepare import measure_decode_request
    pinned = raw._w1_variant
    cands = [("fastest " + pinned[0], pinned)]
    nohead = next((v for v in W1_RANKED if v[3] and "head" not in v[3]), None)
    if nohead is not None and nohead[0] != pinned[0]:
        cands.append(("int8 no-head " + nohead[0], nohead))
    if BEST_BF16_VARIANT[0] != pinned[0]:
        cands.append(("bf16 " + BEST_BF16_VARIANT[0], BEST_BF16_VARIANT))
    rows = []
    for label, variant in cands:
        raw._w1_variant = variant
        # A state (and its captured graph) is bound to the variant it was captured under, and
        # the probe allocates its own state per call, so switching the variant here is enough.
        raw.__dict__.pop("decode_step", None)
        wrapper.__dict__.pop("decode_step", None)
        raw._fast_step = None
        try:
            res = measure_decode_request(wrapper, tokenizer, warmup=2, passes=7)
            ms, tv = res["request_ms_median"], res["decode_tv_distance_max"]
            ok = tv < tv_budget
            rows.append((ms, tv, ok, variant))
            print(f"[req] {label:<34} request_ms_median={ms:8.3f} "
                  f"per_step_ms={(ms / 512):.4f} request_tv_max={tv:.5f} "
                  f"{'ok' if ok else 'OVER BUDGET'}", flush=True)
        except Exception as exc:                       # noqa: BLE001
            print(f"[req] {label:<34} FAILED {type(exc).__name__}: {exc}", flush=True)
        raw.__dict__.pop("decode_step", None)
        wrapper.__dict__.pop("decode_step", None)
        raw._fast_step = None
        gc.collect()
        torch.cuda.empty_cache()
    inside = sorted([r for r in rows if r[2]])
    raw._w1_variant = inside[0][3] if inside else (sorted(rows)[0][3] if rows else pinned)
    print(f"[req] pinning {raw._w1_variant[0]} for the scored probe "
          f"(int8 sites {raw._w1_variant[3]})", flush=True)


print("[mark] NODE-6.1-REQUEST-LEDGER")
try:
    request_ledger(raw_model, model, tokenizer)
except Exception as exc:
    import traceback
    print(f"[req] ledger failed: {type(exc).__name__}: {exc}")
    traceback.print_exc()


# The isolated prefill timing above prices one prefill call; the score is a whole request, and
# only a request can say whether the prefill program pays (its graph capture happens in the
# probe's warmup, its compiled kernels compete for no cache the steps need). So the pinned
# prefill mode is re-decided on the SCORED quantity, with the w1 variant already pinned, and
# the eager branch -- the trunk path -- is one of the candidates, so this can only pin a mode
# that measured faster than trunk on the request itself. Both scored shapes are measured: the
# no-prompt request never takes the prefill branch, so it is the control.
def prefill_request_ledger(raw, wrapper, tokenizer, tv_budget=0.042):
    from prepare import measure_decode_request
    modes = ["eager", "compiled"]
    if raw._pf_mode == "graph":
        modes.append("graph")
    rows = []
    for mode in modes:
        raw._pf_mode = mode
        raw.__dict__.pop("decode_step", None)
        wrapper.__dict__.pop("decode_step", None)
        raw._fast_step = None
        try:
            res = measure_decode_request(wrapper, tokenizer, warmup=2, passes=7)
            ms, tv = res["request_ms_median"], res["decode_tv_distance_max"]
            ok = tv < tv_budget
            rows.append((ms, tv, ok, mode))
            print(f"[pfreq] prefill={mode:<9} request_ms_median={ms:8.3f} "
                  f"request_tv_max={tv:.5f} {'ok' if ok else 'OVER BUDGET'}", flush=True)
        except Exception as exc:                       # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"[pfreq] prefill={mode} FAILED {type(exc).__name__}: {exc}", flush=True)
        raw.__dict__.pop("decode_step", None)
        wrapper.__dict__.pop("decode_step", None)
        raw._fast_step = None
        gc.collect()
        torch.cuda.empty_cache()
    inside = sorted([r for r in rows if r[2]])
    raw._pf_mode = inside[0][3] if inside else "eager"
    print(f"[pfreq] pinning prefill program {raw._pf_mode} for the scored probe", flush=True)


print("[mark] NODE-1.6-PREFILL-REQUEST-LEDGER")
try:
    prefill_request_ledger(raw_model, model, tokenizer)
except Exception as exc:
    import traceback
    print(f"[pfreq] ledger failed: {type(exc).__name__}: {exc}")
    traceback.print_exc()
    raw_model._pf_mode = "eager"


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
