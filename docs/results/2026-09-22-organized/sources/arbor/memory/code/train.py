"""
Autoresearch pretraining script. Single-GPU, single-file.
Cherry-picked and simplified from nanochat.
Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import functools
import gc
import math
import queue
import threading
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import torch._dynamo.utils   # counters: used to assert the fused loss tail really compiled

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


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


# ---------------------------------------------------------------------------
# PARAMETER-COUNT reduction. num_params_total is a CAP (<= 50,332,176), not a requirement,
# and at db=1 the reported peak is ONE microbatch's fwd+bwd sitting on top of the resident
# fp32 parameters (0.201GB) and their fp32 gradients (0.201GB) -- i.e. 8 bytes per parameter
# of the 0.789GB peak, plus 4 more bytes/param of bf16 moments outside the peak window.
# Two cuts on the cheapest-quality axes:
#   * SHARE_VALUE_EMBEDS: the 4 value-embedding tables (vocab x kv_dim = 4.19M each) become
#     ONE shared table, -12.58M params. Every layer's ve_gate stays per-layer, so each layer
#     still scales its own value residual. Embedding gathers are NOT counted by the FLOPs
#     meter, so this lowers only the analytic cross-check and never the measured number; the
#     shared table is gathered ONCE per forward and the result reused across the VE layers
#     (also removing 3 gather activations per microbatch).
#   * MLP_EXPANSION 4x -> 3x: -4.19M params. This DOES lower counted FLOPs/token, which is
#     legal (the cap is an upper bound, and was exactly saturated) and makes a microbatch
#     cheaper, buying back some of the tokens the smaller MLP costs in quality.
SHARE_VALUE_EMBEDS = True
MLP_EXPANSION = 3.0

# ---------------------------------------------------------------------------
# NODE 7.2 (A): dtype of the MUON MATRIX parameters and their gradients.
#
# MEASURED (node 6.2/7.1): the reported peak is an EAGER fwd/bwd of ONE sequence on the
# UNCOMPILED module (the frozen prepare.measure_flops_dispatch probe and the two setup
# verification passes have the same footprint) = resident params + resident grads + ~0.39GB
# of eager activations, and the T-independent optimizer step is a 0.537GB floor made of the
# same params+grads plus the moments. params and grads therefore appear in BOTH binding
# windows, and 20.97M of the 33.55M parameters are the transformer.h matrices.
# Holding those matrices (and their .grad buffers, which are slices of the stacked Muon
# buffers) in bf16 removes 3 x 42MB from the probe window: 42MB of params, 42MB of grads and
# 42MB of the autocast weight-cast copies that an fp32 weight needs on every linear.
# Muon already orthogonalises in bf16, so the update arithmetic is unchanged in kind; the
# only new rounding is the parameter STORE and the 52-way gradient accumulation.
# Node 2.2's warning is about EMBEDDING tables (bf16 wte .grad had 9.8e-1 relative error
# under many-way accumulation because a row's gradient is a sum over the few tokens that hit
# it), so wte / lm_head / the shared value-embedding table and the scalars stay fp32.
MATRIX_PARAM_DTYPE = torch.bfloat16

# ---------------------------------------------------------------------------
# NODE 5.5 (A): dtype of the EMBEDDING-LIKE tables (wte, lm_head, the shared value-embedding
# table: 3 x 4.19M elems = 50MB of fp32 params + 50MB of fp32 grads) with KAHAN-COMPENSATED
# gradient accumulation.
#
# Node 7.2's stage trace: probe window 0.416GB, training loop 0.436GB, and BOTH contain the
# resident params+grads. 0.050+0.050GB of that is these three fp32 tables. Node 2.2 measured
# that NAIVE bf16 accumulation of an embedding gradient over the ~52 microbatches of a step
# has 9.8e-1 relative error, so the tables are held in bf16 here only together with a
# compensated (Kahan) accumulator:
#   * p (bf16) and p.grad (bf16) as usual, plus ONE bf16 accumulator per table;
#   * after each microbatch the accumulator absorbs p.grad in fp32 and the ROUNDING RESIDUAL
#     is written BACK into p.grad, which is exactly the Kahan compensation term: autograd's
#     next in-place accumulation then starts from -(lost bits) instead of from zero. The
#     compensation therefore costs NO extra buffer (it reuses the grad slot that already
#     exists), so the net resident change is -50MB of fp32 params/grads +25MB of accumulator,
#     and the probe window (where the accumulators do not exist yet, they are installed after
#     the frozen probe) sees the full -50MB.
# AdamW then reads the fp32-accurate accumulator; its update math stays fp32 and only the
# STORE into the bf16 table is rounded (stochastically, see _sr_round_to_bf16, so a run of
# sub-ulp updates cannot be silently dropped).
EMBED_PARAM_DTYPE = torch.bfloat16
EMBED_KAHAN_ACCUM = True
ADAMW_SR_STORE = True

# ---------------------------------------------------------------------------
# NODE 5.5 (B): the optimizer step's TRANSIENT workspace. Muon's Newton-Schulz iteration
# allocates A = X.mT@X, A@A, X@B and the NorMuon fp32 square/mean on the WHOLE stacked shape
# group. The largest group here is the 32 512x512 attention matrices (8.4M elems), so those
# temporaries are ~0.1GB -- the measured gap between the 0.331GB steady resident set and the
# 0.436GB peak. The math is per-matrix (every reduction is over the last two dims with
# keepdim), so running the identical fused step on sub-slices of the stacked buffer is
# BIT-IDENTICAL arithmetic with temporaries a factor n/MUON_CHUNK smaller. 4 divides every
# group size here (32, 8, 8, 4), so torch.compile sees one shape per group and does not
# recompile on a remainder.
MUON_CHUNK = 4

# NODE 7.2 (B): spend the FLOPs headroom (trunk 213,912,576 of a 239,078,400 cap) on
# recomputing the attention block's COUNTED q/k/v projections in backward. The region is
# the block input -> norm -> c_q/c_k/c_v -> value-residual gate -> rotary -> qk-norm, i.e.
# everything up to (but NOT including) the flash-attention call: prepare.py tallies attention
# FLOPs in a python wrapper around flash_attn_func, so a recomputed attention kernel would
# double-count. q, k and v are the region's outputs and stay saved (flash-attn's own backward
# needs them); what is dropped is norm(x), the three raw projection outputs, the two rotary
# outputs and the gate -- ~3072 bf16 elements per token per layer.
# Cost: +3 * 2 * d^2 = +1.57M FLOPs/token/layer = +12.58M FLOPs/token over 8 layers.
RECOMPUTE_ATTN_QKV = True


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


# Number of flattened (B*T) rows whose logits exist at once inside the loss. The full
# logits tensor for a 128x2048 batch is B*T*V = 2.1e9 elements; materialising it in bf16
# plus the fp32 softcap copies is what the reference peak is made of. Bounding it to
# CHUNK_ROWS*V keeps that term at ~0.5GB instead of ~20GB. 16384 makes ONE training
# microbatch (DEVICE_BATCH_SIZE*2048 rows <= 16384) a single chunk: the retained fp32 bytes
# are identical either way (every chunk's softcap/log-softmax copies are held for backward),
# so the chunk loop only buys python/launch overhead -- which now matters, because a step is
# 40+ microbatches instead of 4.
LOSS_CHUNK_ROWS = 16384
# Number of SEQUENCES resident at once inside the frozen validation forward. evaluate_bpb
# calls the model at EVAL_BATCH_SIZE=128 x 2048 with reduction='none' under no_grad; once
# the training microbatch is small, that single frozen forward would become the new peak
# (block activations scale linearly with the resident batch). Splitting it into sub-batches
# of EVAL_MICRO_SEQS sequences and concatenating the per-token losses is numerically
# identical (per-row cross-entropy is independent across rows) and costs no extra
# arithmetic, so flops_per_token_measured is untouched. MEASURED: at micro=8 the eval forward
# peaks at 2.25GB, which is BELOW the db=3 training peak (2.58GB) but ABOVE the db=2 training
# peak (1.98GB) -- at db=2 the frozen validation forward, not training, set the reported peak.
# 4 keeps eval out of the peak at any microbatch this config might use.
# MEASURED at db=2 with the microbatch captured as a CUDA graph: training peaks at 1.505GB but
# the reported peak was 1.592GB, set by the frozen eval forward at micro=4 (alloc 0.62GB +
# ~0.97GB of transient logits/softcap bytes for 4 resident sequences). micro=2 halves that
# transient term and puts eval back below the training peak. It is numerically identical
# (per-row CE is row-independent), runs under no_grad, and is not part of the FLOPs probe.
# NODE 4.2: with the fused loss tail the eval forward at micro=2 peaked at 0.96GB against a
# 1.103GB training peak -- only 0.14GB of slack. At db=1 the training peak drops further, so
# eval would set it; micro=1 halves the eval transient again (MEASURED 0.58GB at 2048 rows)
# and also makes the eval rows shape identical to training's, reusing the same compiled tail.
EVAL_MICRO_SEQS = 1
# If the whole logits tensor would be small enough to keep, keep it: recomputation only
# pays for itself (and only costs extra matmuls) above this budget.
RETAIN_LOGITS_BYTES = 512 * 2**20
_LOSS_DEBUG_SEEN = set()

# ---------------------------------------------------------------------------
# Selective ELEMENTWISE-ONLY recomputation.
#
# The task's FLOPs meter "counts matrix, convolution and attention work; excludes
# elementwise and normalization work", and our flops_per_token_measured sits EXACTLY on the
# cap (239,078,400). So recomputing a matmul in backward would breach the cap, while
# recomputing a norm / relu**2 / fp32 cast / tanh / log-softmax is free on that axis.
#
# Two regions are recomputed, chosen from the MEASURED db=3 breakdown of the 2.582GB peak
# (params+grads+optimizer 0.53GB, block activations 1.15GB, fp32 loss-path residue 0.41GB):
#   (a) the MLP half of every block: only the block input is handed in; norm(x), relu(h)
#       and relu(h)**2 are recomputed in backward, while the two c_fc/c_proj matmul results
#       are MUST_SAVE, so no counted arithmetic is re-executed. Per layer that drops one
#       d-sized and two 4d-sized bf16 tensors (~56MB at db=3) of the ~144MB it retains.
#   (b) the loss tail: the bf16 lm_head output is kept (its matmul is counted) and only the
#       fp32 copy, the softcap tanh and the cross-entropy's log-softmax are recomputed, in
#       row sub-chunks so those fp32 tensors are bounded by LOSS_TAIL_ROWS*V at every moment
#       -- in forward AND in the backward recompute, which is what actually moves the peak.
# ATTENTION IS DELIBERATELY NOT WRAPPED: prepare.py's attention tally is a python wrapper
# around flash_attn_func, so re-executing that python call in backward would double-count
# attention FLOPs even though the kernel's output is served from the SAC cache.
#
# MEASURED (launch 1, both regions on, db=3): alloc at loss entry 1.68 -> 1.09GB, fp32 loss
# residue 0.41 -> 0.10GB, peak 2,582,018,048 -> 1,860,401,664 (-28%), and
# flops_per_token_measured EXACTLY 239,078,400 (delta +0) -- the cap is untouched, as the
# mechanism predicts. The cost is time: 304K -> 255K tok/s, so 155.0M instead of 183.9M
# tokens and val_bpb 1.0589 > the 1.05 gate. The two regions are NOT equally priced: the MLP
# region accounts for 0.59GB of the 0.72GB saved, while the loss tail's 2048-row sub-chunking
# adds 3x the eager python work of the whole loss path per microbatch (26-40 of them per step,
# each already only ~15ms) for the remaining ~0.13GB. So the loss tail is switched OFF and the
# MLP region -- the cheap 80% of the saving -- is kept.
SELECTIVE_RECOMPUTE = True
RECOMPUTE_LOSS_TAIL = False
LOSS_TAIL_ROWS = LOSS_CHUNK_ROWS   # one tail per loss chunk: the trunk's op sequence exactly
_COUNTED_MATMUL_OPS = {
    torch.ops.aten.mm.default,
    torch.ops.aten.addmm.default,
    torch.ops.aten.bmm.default,
    torch.ops.aten.baddbmm.default,
    torch.ops.aten._scaled_mm.default,
}
_RECOMPUTE_DEBUG = [True, True]


def _elementwise_only_policy(ctx, op, *args, **kwargs):
    """Save every counted (matmul) result; recompute everything else in backward."""
    Policy = torch.utils.checkpoint.CheckpointPolicy
    return Policy.MUST_SAVE if op in _COUNTED_MATMUL_OPS else Policy.MUST_RECOMPUTE


# The documented torch.compile-friendly spelling of the context factory.
_SAC_CONTEXT_FN = functools.partial(
    torch.utils.checkpoint.create_selective_checkpoint_contexts,
    _elementwise_only_policy)


# ---------------------------------------------------------------------------
# FUSED loss tail (ported from node 3.2): the same uncounted arithmetic, but as ONE
# compiled kernel per microbatch instead of an eager op chain.
#
# The trunk's tail is `logits.float() -> softcap*tanh -> F.cross_entropy`, executed in the
# eager @torch.compiler.disable'd loss function. That chain RETAINS two full fp32
# vocab-sized tensors for backward (the tanh output and the log-softmax output) and, in
# backward, walks nll -> log_softmax -> tanh -> cast as five more full-size fp32 kernels.
# Both terms land on the peak, which is reached DURING the loss backward.
#
# Here the whole tail is a single torch.autograd.Function per microbatch:
#   * forward saves ONLY the bf16 logits (its matmul is COUNTED so it may not be
#     recomputed) plus the per-row logsumexp (n floats), and returns the per-row loss;
#   * backward reconstructs softmax and the softcap derivative from those two in ONE
#     pointwise expression: grad = (softmax(c) - onehot) * (1 - tanh(l/s)**2) * gout,
#     with c = s*tanh(l/s). Written as a single pointwise formula on purpose, so inductor
#     emits ONE kernel that reads the bf16 logits and writes the bf16 grad, with no
#     full-size fp32 buffer materialised at all.
# Every op here is elementwise or a row reduction -- UNCOUNTED by the FLOPs meter -- and no
# matmul is re-executed, so flops_per_token_measured is untouched (the cap is exactly
# saturated). The bodies are torch.compile'd separately from the model: the loss lives in a
# compiler-disabled region, so this is the only way to get the tail into a fused graph
# without making the eager chunk loop traceable.
#
# CUDA-graph compatibility (this node): the compiled bodies are warmed up on the capture
# side stream before capture, so no compilation, no host sync and no allocation outside the
# graph's private pool happens during capture or replay.
FUSED_LOSS_TAIL = True
_FUSED = {"compile": True, "logged": False}


def _softcap_ce_fwd(logits, targets, softcap):
    """(per-row loss, per-row logsumexp) for softcapped logits. fp32 math, uncounted."""
    c = softcap * torch.tanh(logits.float() / softcap)
    lse = torch.logsumexp(c, -1)
    ct = c.gather(1, targets.clamp_min(0).unsqueeze(1)).squeeze(1)
    return (lse - ct) * (targets >= 0), lse


def _softcap_ce_bwd(logits, lse, targets, gout, softcap):
    """d(loss)/d(logits) from the bf16 logits and the saved logsumexp only.

    One pointwise expression: no reduction, no in-place aliasing, so inductor fuses it into
    a single kernel whose only large tensors are the bf16 input and the bf16 output.
    """
    t = torch.tanh(logits.float() / softcap)
    p = torch.exp(t * softcap - lse.unsqueeze(1))
    col = torch.arange(logits.size(1), device=logits.device)
    onehot = col.unsqueeze(0) == targets.unsqueeze(1)
    g = (p - onehot.to(p.dtype)) * (1.0 - t * t) * (gout * (targets >= 0)).unsqueeze(1)
    return g.to(logits.dtype)


_softcap_ce_fwd_compiled = torch.compile(_softcap_ce_fwd, dynamic=False)
_softcap_ce_bwd_compiled = torch.compile(_softcap_ce_bwd, dynamic=False)

# The eager fallback exists for exactly two callers: the frozen FLOPs probe (which runs
# inside FlopCounterMode, where dynamo may refuse to trace) and a compile failure. The
# pointwise bodies above are written for a fusing compiler and would allocate several
# full-size fp32 temporaries when run op-by-op, so the eager route walks the rows in small
# slices -- slower, but it can never set the peak.
# NODE 7.2 launch 1 MEASURED that the claim above was false at 1024 rows: with the bf16
# matrix params and the recomputed attention region, train.py's own eager fwd/bwd peaked at
# 0.393GB while the reported peak was 0.564GB, and the only thing between them is the frozen
# FLOPs probe -- which is exactly where this eager tail runs. At 1024 rows each of the
# op-by-op fp32 temporaries is 1024 x 8192 x 4 = 33.5MB and three or four are live at once
# inside the backward, ON TOP of the whole retained activation set, i.e. ~0.13GB of the
# reported peak. 128 rows makes every one of them 4.2MB. This is pure elementwise work in a
# code path that runs ONLY under a dispatch mode, so it changes no counted FLOP, no training
# kernel and no number the model is judged on except the peak.
_EAGER_TAIL_ROWS = 128


def _softcap_ce_fwd_eager(logits, targets, softcap):
    n = targets.numel()
    loss = torch.empty(n, dtype=torch.float32, device=logits.device)
    lse = torch.empty(n, dtype=torch.float32, device=logits.device)
    for s in range(0, n, _EAGER_TAIL_ROWS):
        e = min(s + _EAGER_TAIL_ROWS, n)
        loss[s:e], lse[s:e] = _softcap_ce_fwd(logits[s:e], targets[s:e], softcap)
    return loss, lse


def _softcap_ce_bwd_eager(logits, lse, targets, gout, softcap):
    n = targets.numel()
    g = torch.empty_like(logits)
    for s in range(0, n, _EAGER_TAIL_ROWS):
        e = min(s + _EAGER_TAIL_ROWS, n)
        g[s:e] = _softcap_ce_bwd(logits[s:e], lse[s:e], targets[s:e], gout[s:e], softcap)
    return g


def _fused_call(compiled, eager, args, tag):
    """Run the compiled body, with an eager fallback that cannot lose a launch.

    Skipped under an active torch dispatch mode: the frozen FLOPs probe runs inside
    FlopCounterMode, and dynamo may refuse to trace there. The eager body is the same
    arithmetic, so the measurement is unaffected either way.
    """
    if not _FUSED["compile"] or torch._C._len_torch_dispatch_stack() > 0:
        return eager(*args)
    try:
        return compiled(*args)
    except Exception as exc:
        _FUSED["compile"] = False
        print(f"\n[fused-loss] compile DISABLED at {tag}, raised "
              f"{type(exc).__name__}: {exc}", flush=True)
        return eager(*args)


class _SoftcapCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, softcap):
        if _FUSED["logged"]:
            loss, lse = _fused_call(_softcap_ce_fwd_compiled, _softcap_ce_fwd_eager,
                                    (logits, targets, softcap), "forward")
        else:
            graphs_before = torch._dynamo.utils.counters["stats"]["unique_graphs"]
            loss, lse = _fused_call(_softcap_ce_fwd_compiled, _softcap_ce_fwd_eager,
                                    (logits, targets, softcap), "forward")
            _FUSED["logged"] = True
            graphs_after = torch._dynamo.utils.counters["stats"]["unique_graphs"]
            print(f"\n[fused-loss] first forward rows={targets.numel()} V={logits.size(1)} "
                  f"unique_graphs {graphs_before}->{graphs_after} "
                  f"fused={graphs_after > graphs_before} "
                  f"dispatch_modes={torch._C._len_torch_dispatch_stack()} "
                  f"alloc={torch.cuda.memory_allocated()/1e9:.2f}GB "
                  f"peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)
        ctx.save_for_backward(logits, lse, targets)
        ctx.softcap = softcap
        return loss

    @staticmethod
    def backward(ctx, gout):
        logits, lse, targets = ctx.saved_tensors
        return _fused_call(_softcap_ce_bwd_compiled, _softcap_ce_bwd_eager,
                           (logits, lse, targets, gout.contiguous(), ctx.softcap),
                           "backward"), None, None


def _ce_tail(logits, t_chunk, softcap):
    """fp32 softcap -> per-token CE. All uncounted: no matmul here."""
    if FUSED_LOSS_TAIL:
        if torch.is_grad_enabled() and logits.requires_grad:
            return _SoftcapCE.apply(logits, t_chunk, softcap)
        return _fused_call(_softcap_ce_fwd_compiled, _softcap_ce_fwd_eager,
                           (logits, t_chunk, softcap), "forward-nograd")[0]
    logits = logits.float()
    logits = softcap * torch.tanh(logits / softcap)
    return F.cross_entropy(logits, t_chunk, ignore_index=-1, reduction='none')


def _chunk_ce(weight, x_chunk, t_chunk, softcap):
    """lm_head -> fp32 softcap -> per-token CE for one row-chunk (no reduction).

    The lm_head matmul runs ONCE per row (its bf16 result is retained, because that matmul is
    counted arithmetic) and only the fp32 tail is recomputed in backward. The rows are walked
    in LOSS_TAIL_ROWS sub-chunks so the three fp32 vocab-sized copies are bounded by
    LOSS_TAIL_ROWS*V both in forward and inside the backward recompute. The matmul is issued
    per sub-chunk rather than once over a sliced result on purpose: slicing a single retained
    logits tensor would make every sub-chunk's backward allocate a full-size bf16 grad via
    slice_backward, which costs more than the fp32 copies it saves.
    """
    n = t_chunk.numel()
    recompute = (SELECTIVE_RECOMPUTE and RECOMPUTE_LOSS_TAIL
                 and torch.is_grad_enabled() and x_chunk.requires_grad)
    parts = []
    for s in range(0, n, LOSS_TAIL_ROWS):
        e = min(s + LOSS_TAIL_ROWS, n)
        logits = F.linear(x_chunk[s:e], weight)
        if recompute:
            parts.append(torch.utils.checkpoint.checkpoint(
                _ce_tail, logits, t_chunk[s:e], softcap, use_reentrant=False))
        else:
            parts.append(_ce_tail(logits, t_chunk[s:e], softcap))
        del logits
    return parts[0] if len(parts) == 1 else torch.cat(parts)


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

    def _qkv(self, x, ve, cos, sin):
        """block input -> (q, k, v) ready for flash-attention. NO attention kernel here:
        this whole region is recomputed in backward when RECOMPUTE_ATTN_QKV is on, and
        re-executing flash_attn_func would double-count attention in the frozen FLOPs probe."""
        x = norm(x)
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        return q, k, v

    def forward(self, x, ve, cos_sin, window_size):
        cos, sin = cos_sin
        if RECOMPUTE_ATTN_QKV and torch.is_grad_enabled() and x.requires_grad:
            if _RECOMPUTE_DEBUG[1] and not torch.compiler.is_compiling():
                _RECOMPUTE_DEBUG[1] = False
                print(f"\n[recompute] attention q/k/v region checkpointed "
                      f"(counted projections recomputed in backward)", flush=True)
            q, k, v = torch.utils.checkpoint.checkpoint(
                self._qkv, x, ve, cos, sin, use_reentrant=False)
        else:
            q, k, v = self._qkv(x, ve, cos, sin)

        B, T = q.size(0), q.size(1)
        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = int(round(config.n_embd * MLP_EXPANSION))
        self.c_fc = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)

    def _inner(self, x):
        x = norm(x)
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

    def forward(self, x):
        # x is the BLOCK input (un-normed): the norm is inside the recomputed region.
        if SELECTIVE_RECOMPUTE and torch.is_grad_enabled() and x.requires_grad:
            if _RECOMPUTE_DEBUG[0] and not torch.compiler.is_compiling():
                _RECOMPUTE_DEBUG[0] = False
                print(f"\n[recompute] MLP selective checkpoint ACTIVE "
                      f"(save matmul results, recompute norm/relu/square)", flush=True)
            return torch.utils.checkpoint.checkpoint(
                self._inner, x, use_reentrant=False,
                context_fn=_SAC_CONTEXT_FN)
        return self._inner(x)


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        # x is the BLOCK input (un-normed): the attention norm lives inside attn._qkv, which
        # is the recomputed region, exactly as the MLP's norm lives inside MLP._inner.
        x = x + self.attn(x, ve, cos_sin, window_size)
        x = x + self.mlp(x)
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
        ve_layers = [i for i in range(config.n_layer) if has_ve(i, config.n_layer)]
        if SHARE_VALUE_EMBEDS:
            # One table under every VE key: nn.Module.parameters() (and hence the frozen
            # census, the optimizer, and the gradient/moment footprint) sees it ONCE.
            shared = nn.Embedding(config.vocab_size, kv_dim)
            self.value_embeds = nn.ModuleDict({str(i): shared for i in ve_layers})
        else:
            self.value_embeds = nn.ModuleDict({
                str(i): nn.Embedding(config.vocab_size, kv_dim) for i in ve_layers
            })
        self.ve_shared = SHARE_VALUE_EMBEDS
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
        # Embedding tables stay FP32 (cast to bf16 at use in _forward_impl, so activations
        # and matmuls are bit-for-bit the old ones). MEASURED: with 43 grad accumulations a
        # bf16 .grad buffer for wte has 9.8e-01 relative error against the fp32 sum, i.e.
        # the bf16 table turned the embedding gradient into noise once accumulation was
        # introduced. Keeping the table (and hence .grad and the Adam moments) in fp32 costs
        # +0.042GB of the ~4.7GB peak and was worth 0.0013 val_bpb.

    @torch.no_grad()
    def compress_matrix_params(self):
        """Hold the Muon matrix weights (transformer.h) in MATRIX_PARAM_DTYPE.

        Called AFTER init_weights, so every initialiser still runs in fp32 and only the
        stored copy is narrowed. Their .grad buffers follow the parameter dtype (autograd
        requires a matching dtype), which is the point: params and grads are both resident in
        the two windows that set peak_vram_bytes. Embedding-like tensors (wte, lm_head, the
        shared value-embedding table) and the per-layer scalars are deliberately left fp32.
        """
        n = 0
        for p in self.transformer.h.parameters():
            if p.dtype != MATRIX_PARAM_DTYPE:
                p.data = p.data.to(MATRIX_PARAM_DTYPE)
                n += p.numel()
        return n

    @torch.no_grad()
    def compress_embedding_params(self):
        """NODE 5.5: hold the embedding-like tables (wte, lm_head, the shared value-embedding
        table) in EMBED_PARAM_DTYPE. Called after init_weights, so every initialiser still
        runs in fp32. Their .grad buffers follow the parameter dtype; the many-way gradient
        accumulation that node 2.2 measured as unusable in naive bf16 is made sound by the
        Kahan accumulator installed on the optimizer (see EMBED_KAHAN_ACCUM).
        """
        n = 0
        seen = set()
        tables = [self.transformer.wte.weight, self.lm_head.weight]
        tables += [ve.weight for ve in self.value_embeds.values()]
        for p in tables:
            if id(p) in seen or p.dtype == EMBED_PARAM_DTYPE:
                continue
            seen.add(id(p))
            p.data = p.data.to(EMBED_PARAM_DTYPE)
            n += p.numel()
        return n

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
        value_embeds_numel = sum({id(ve.weight): ve.weight.numel()
                                  for ve in self.value_embeds.values()}.values())
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
        # The only caller that asks for per-token losses is the frozen evaluator, and it
        # does so under torch.no_grad(). Route that (and only that) through the sub-batched
        # path: the training loss must stay a single differentiable graph.
        if (targets is not None and reduction == 'none'
                and not torch.is_grad_enabled()
                and idx.size(0) > EVAL_MICRO_SEQS):
            return self._eval_forward_subbatched(idx, targets)
        return self._forward_impl(idx, targets, reduction)

    @torch.compiler.disable
    @torch.no_grad()
    def _eval_forward_subbatched(self, idx, targets):
        B = idx.size(0)
        # max_memory_allocated() is monotone over the process and is already pinned by the
        # training step, so it cannot say what the EVAL forward costs. Sample the live
        # allocated bytes per chunk instead: that is the number node 5.2 is trying to move.
        first = 'eval' not in _LOSS_DEBUG_SEEN
        alloc_in = torch.cuda.memory_allocated()
        hi = alloc_in
        out = torch.empty(targets.shape, dtype=torch.float32, device=targets.device)
        for s in range(0, B, EVAL_MICRO_SEQS):
            e = min(s + EVAL_MICRO_SEQS, B)
            part = self._forward_impl(idx[s:e], targets[s:e], 'none')
            out[s:e] = part
            del part
            if first:
                hi = max(hi, torch.cuda.memory_allocated())
        if first:
            _LOSS_DEBUG_SEEN.add('eval')
            print(f"\n[eval-subbatch] B={B} micro={EVAL_MICRO_SEQS} "
                  f"chunks={(B + EVAL_MICRO_SEQS - 1) // EVAL_MICRO_SEQS} "
                  f"alloc_at_entry={alloc_in/1e9:.3f}GB "
                  f"alloc_max_between_chunks={hi/1e9:.3f}GB "
                  f"alloc={torch.cuda.memory_allocated()/1e9:.3f}GB "
                  f"peak={torch.cuda.max_memory_allocated()/1e9:.3f}GB", flush=True)
        return out

    def _forward_impl(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(idx).bfloat16()
        x = norm(x)
        x0 = x
        ve_cache = None
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = None
            if str(i) in self.value_embeds:
                if self.ve_shared:
                    # Same table for every VE layer: gather once, reuse the activation
                    # (autograd accumulates the gradient from each use).
                    if ve_cache is None:
                        ve_cache = self.value_embeds[str(i)](idx).bfloat16()
                    ve = ve_cache
                else:
                    ve = self.value_embeds[str(i)](idx).bfloat16()
            x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)

        softcap = 15
        if targets is not None:
            return self._chunked_loss(x, targets, softcap, reduction)

        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)
        return logits

    @torch.compiler.disable
    def _chunked_loss(self, x, targets, softcap, reduction):
        """Streamed cross-entropy over row-chunks of the flattened hidden state.

        Mathematically identical to computing the whole logits tensor and calling
        F.cross_entropy on it, but only CHUNK_ROWS x V logits exist at any moment. When
        grad is required the chunk logits are recomputed in backward (activation
        checkpointing) so they are never retained between forward and backward.
        """
        alloc_in = torch.cuda.memory_allocated()
        peak_in = torch.cuda.max_memory_allocated()
        x_flat = x.reshape(-1, x.size(-1))
        t_flat = targets.reshape(-1)
        n = t_flat.numel()
        chunk = LOSS_CHUNK_ROWS
        num_chunks = (n + chunk - 1) // chunk
        # Recompute chunk logits in backward only when retaining them all would cost more
        # than RETAIN_LOGITS_BYTES; below that the logits are not what bounds the peak and
        # recomputation would only add matmul work.
        full_logits_bytes = n * self.lm_head.weight.size(0) * 2
        use_ckpt = (full_logits_bytes > RETAIN_LOGITS_BYTES
                    and torch.is_grad_enabled() and x_flat.requires_grad)
        w = self.lm_head.weight
        parts = []
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            if use_ckpt:
                part = torch.utils.checkpoint.checkpoint(
                    _chunk_ce, w, x_flat[s:e], t_flat[s:e], softcap, use_reentrant=False)
            else:
                part = _chunk_ce(w, x_flat[s:e], t_flat[s:e], softcap)
            parts.append(part)
        losses = parts[0] if len(parts) == 1 else torch.cat(parts)
        key = (n, use_ckpt, reduction)
        if key not in _LOSS_DEBUG_SEEN:
            _LOSS_DEBUG_SEEN.add(key)
            print(f"\n[chunked-loss] rows={n} chunk={chunk} num_chunks={num_chunks} "
                  f"use_ckpt={use_ckpt} grad_enabled={torch.is_grad_enabled()} "
                  f"reduction={reduction} losses={tuple(losses.shape)}/{losses.dtype} "
                  f"| alloc_at_entry={alloc_in/1e9:.2f}GB peak_at_entry={peak_in/1e9:.2f}GB "
                  f"alloc_now={torch.cuda.memory_allocated()/1e9:.2f}GB "
                  f"peak_now={torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)
        if reduction == 'none':
            return losses.view(targets.shape)
        if reduction == 'sum':
            return losses.sum()
        assert reduction == 'mean'
        return losses.sum() / (t_flat != -1).sum().clamp_min(1)

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

# ---------------------------------------------------------------------------
# NODE 5.5: stochastic rounding fp32 -> bf16, and Kahan-compensated bf16 accumulation.
# Both are written as single fused expressions and compiled, so the fp32 intermediate never
# becomes a full-size buffer: inductor emits ONE kernel that reads the bf16 inputs and writes
# the bf16 outputs (this is the "one arena" property -- the working set of the update is the
# tensors it mutates and nothing else).
# ---------------------------------------------------------------------------

def _sr_round_to_bf16(x):
    """Round fp32 -> bf16 with probability equal to the discarded fraction (unbiased).

    The dither is an xorshift32 hash of the value's own bits (shifts/xors only, no RNG state
    and therefore safe under CUDA-graph capture and deterministic across replays). Adding a
    16-bit dither to the fp32 bit pattern and truncating the low 16 bits rounds up exactly
    when low16 + dither carries, i.e. with probability low16/2**16; a value already exactly
    representable in bf16 is returned unchanged.
    """
    bits = x.view(torch.int32)
    h = bits ^ (bits << 13)
    h = h ^ (h >> 17)
    h = h ^ (h << 5)
    r = h & 0xFFFF
    return ((bits + r) & -65536).view(torch.float32).bfloat16()


def _kahan_merge_body(acc, grad):
    s = acc.float() + grad.float()
    new = s.bfloat16()
    resid = s - new.float()
    acc.copy_(new)
    grad.copy_(resid)


_kahan_merge_compiled = torch.compile(_kahan_merge_body, dynamic=False)
_KAHAN = {"compile": True}


def kahan_merge_(acc, grad):
    """acc += grad (bf16 value + bf16 compensation left in `grad`)."""
    if _KAHAN["compile"]:
        try:
            _kahan_merge_compiled(acc, grad)
            return
        except Exception as exc:
            _KAHAN["compile"] = False
            print(f"\n[kahan] compile DISABLED, raised {type(exc).__name__}: {exc}", flush=True)
    _kahan_merge_body(acc, grad)


@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused_sr(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    """AdamW for a narrow (bf16) parameter: all math in fp32, stochastically-rounded store."""
    exp_avg.lerp_(grad.to(exp_avg.dtype), (1 - beta1_t).to(exp_avg.dtype))
    exp_avg_sq.lerp_(grad.float().square().to(exp_avg_sq.dtype),
                     (1 - beta2_t).to(exp_avg_sq.dtype))
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq.float() / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    new_p = p.float() * (1 - lr_t * wd_t) - step_size * (exp_avg.float() / denom)
    p.copy_(_sr_round_to_bf16(new_p) if ADAMW_SR_STORE else new_p.bfloat16())


@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    # The two moments may be held in a narrower dtype than the parameter (OPT_MOMENT_DTYPE).
    # The moment UPDATES are the only thing that loses precision: both the denominator and
    # the parameter update itself are still formed in fp32 and applied to an fp32 parameter,
    # so a bf16 moment costs ~0.4% relative noise on one step's update and nothing else.
    # Parameters and gradients are NEVER narrowed (see init_weights' note on bf16 tables).
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad.to(exp_avg.dtype), (1 - beta1_t).to(exp_avg.dtype))
    exp_avg_sq.lerp_(grad.square().to(exp_avg_sq.dtype), (1 - beta2_t).to(exp_avg_sq.dtype))
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq.float() / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg.float() / denom, alpha=-step_size)

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    # Nesterov momentum. momentum_buffer may be bf16 while the gradients stay fp32; the
    # Nesterov combination itself is formed in fp32 and X is cast to bf16 on the very next
    # line anyway, so the buffer's dtype only bounds the precision of the running average.
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads.to(momentum_buffer.dtype),
                          (1 - momentum).to(momentum_buffer.dtype))
    g = stacked_grads.lerp_(momentum_buffer.to(stacked_grads.dtype), momentum)
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
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype),
                                 (1 - beta2_t.to(second_momentum_buffer.dtype)))
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


# Dtype of the OPTIMIZER MOMENTS only (Muon's Nesterov buffer, AdamW's exp_avg/exp_avg_sq).
# Parameters and gradients stay fp32 unconditionally: node 2.2 MEASURED that a bf16 .grad
# buffer for wte has 9.8e-01 relative error against the fp32 sum under 43-way accumulation.
# A moment is a per-step exponential average, not a many-way sum, so the same argument does
# not apply to it. At 50.3M params the three moment buffers are 25.2M(muon)+25.2M*2(adamw)
# = 75.5M elements, i.e. 302MB in fp32 and 151MB in bf16 of a ~760MB resident floor.
OPT_MOMENT_DTYPE = torch.bfloat16


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        self.stacked_views = False
        self.kahan_accums = {}      # id(param) -> bf16 Kahan accumulator (NODE 5.5)
        self._kahan_pairs = []      # [(param, accumulator)] in a fixed order
        self._sr_ok = ADAMW_SR_STORE
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

    @torch.no_grad()
    def install_stacked_views(self):
        """Make every Muon matrix param (and its .grad) a SLICE of one contiguous per-shape
        buffer, so the per-step `torch.stack(grads)` / `torch.stack(params)` + copy-back
        disappears.

        Those two stacks were a transient FULL duplication of the largest shape group's
        params and grads (8.4M elements each at depth 8 -> 2 x 33.6MB) allocated inside
        every optimizer step, and the optimizer step is what sets the reported peak. With
        the params/grads living IN the stacked layout there is nothing to duplicate and
        nothing to copy back: `muon_step_fused` mutates the parameters in place through the
        same storage. Numerically this is a no-op (the copy-back was exact).
        """
        total_elems = 0
        for group in self.param_groups:
            if group['kind'] != 'muon' or not group['params']:
                continue
            params = group['params']
            p0 = params[0]
            n, shape, dtype, dev = len(params), p0.shape, p0.dtype, p0.device
            pbuf = torch.empty(n, *shape, dtype=dtype, device=dev)
            for i, p in enumerate(params):
                pbuf[i].copy_(p.detach())
            for i, p in enumerate(params):
                p.data = pbuf[i]
            gbuf = torch.zeros(n, *shape, dtype=dtype, device=dev)
            for i, p in enumerate(params):
                p.grad = gbuf[i]
            group['_stacked_params'] = pbuf
            group['_stacked_grads'] = gbuf
            total_elems += pbuf.numel()
        self.stacked_views = True
        gc.collect()
        torch.cuda.empty_cache()
        return total_elems

    @torch.no_grad()
    def install_kahan_accums(self):
        """One bf16 accumulator per narrow (bf16) AdamW parameter -- the embedding tables.

        Installed AFTER the frozen FLOPs probe so the probe window never holds them. The
        compensation term is NOT a second buffer: it lives in p.grad between microbatches.
        """
        bytes_ = 0
        for group in self.param_groups:
            if group['kind'] != 'adamw':
                continue
            for p in group['params']:
                if p.dtype == torch.float32 or p.numel() < 1024:
                    continue
                acc = torch.zeros_like(p)
                self.kahan_accums[id(p)] = acc
                self._kahan_pairs.append((p, acc))
                bytes_ += acc.numel() * acc.element_size()
        return bytes_

    @torch.no_grad()
    def merge_kahan_grads(self):
        """Absorb this microbatch's bf16 gradient into the accumulator, leaving the rounding
        residual in p.grad as the Kahan compensation for the next microbatch."""
        for p, acc in self._kahan_pairs:
            if p.grad is not None:
                kahan_merge_(acc, p.grad)

    def kahan_grad_bytes(self):
        return sum(a.numel() * a.element_size() for _, a in self._kahan_pairs)

    def stacked_grads_intact(self):
        """True iff every Muon .grad is still the slice that was handed to autograd."""
        for group in self.param_groups:
            gbuf = group.get('_stacked_grads')
            if gbuf is None:
                continue
            base = gbuf.data_ptr()
            stride = gbuf.stride(0) * gbuf.element_size()
            for i, p in enumerate(group['params']):
                if p.grad is None or p.grad.data_ptr() != base + i * stride:
                    return False
        return True

    def drop_stacked_grad_views(self):
        for group in self.param_groups:
            group['_stacked_grads'] = None
        self.stacked_views = False

    def release_state(self):
        """Free every tensor the frozen validation forward does not need.

        Called after the training loop: gradients and optimizer moments are ~0.5GB of the
        resident footprint and are pure dead weight during the eval forward. The stacked
        PARAM buffers are kept -- the parameters are views into them.
        """
        freed = 0
        for _, acc in self._kahan_pairs:
            freed += acc.numel() * acc.element_size()
        self._kahan_pairs = []
        self.kahan_accums = {}
        for state in self.state.values():
            for k, v in list(state.items()):
                if torch.is_tensor(v):
                    freed += v.numel() * v.element_size()
                    del state[k]
        self.state.clear()
        for group in self.param_groups:
            gbuf = group.get('_stacked_grads')
            if gbuf is not None:
                freed += gbuf.numel() * gbuf.element_size()
            group['_stacked_grads'] = None
        self.stacked_views = False
        return freed

    def state_bytes(self):
        return sum(v.numel() * v.element_size()
                   for state in self.state.values()
                   for v in state.values() if torch.is_tensor(v))

    def _step_adamw(self, group):
        for p in group['params']:
            # NODE 5.5: for a bf16 embedding table the authoritative gradient of the step is
            # the Kahan accumulator, not p.grad (which holds the compensation residual).
            acc = self.kahan_accums.get(id(p))
            grad = acc if acc is not None else p.grad
            if grad is None:
                continue
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros(p.shape, dtype=OPT_MOMENT_DTYPE, device=p.device)
                state['exp_avg_sq'] = torch.zeros(p.shape, dtype=OPT_MOMENT_DTYPE, device=p.device)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            args = (p, grad, state['exp_avg'], state['exp_avg_sq'],
                    self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                    self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)
            if p.dtype == torch.float32:
                adamw_step_fused(*args)
            else:
                # Narrow parameter: fp32 update math, (stochastically) rounded store.
                if self._sr_ok:
                    try:
                        adamw_step_fused_sr(*args)
                    except Exception as exc:
                        self._sr_ok = False
                        print(f"\n[sr-store] DISABLED, raised {type(exc).__name__}: {exc}",
                              flush=True)
                if not self._sr_ok:
                    state['exp_avg'].lerp_(grad.to(OPT_MOMENT_DTYPE),
                                           (1 - group['betas'][0]))
                    state['exp_avg_sq'].lerp_(grad.float().square().to(OPT_MOMENT_DTYPE),
                                              (1 - group['betas'][1]))
                    b1 = 1 - group['betas'][0] ** state['step']
                    b2 = 1 - group['betas'][1] ** state['step']
                    denom = (state['exp_avg_sq'].float() / b2).sqrt() + group['eps']
                    p.copy_((p.float() * (1 - group['lr'] * group['weight_decay'])
                             - (group['lr'] / b1) * (state['exp_avg'].float() / denom)))
            if acc is not None:
                acc.zero_()

    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape,
                                                   dtype=OPT_MOMENT_DTYPE, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            # Kept in fp32 regardless of the parameter dtype: it is a running second moment
            # (one scalar per row/column, a few thousand elements) and rsqrt of a bf16
            # variance is the one place a narrow dtype would actually change the step size.
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=torch.float32, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        # Fast path: params and grads already live in the stacked layout, so there is
        # nothing to stack and nothing to copy back. Guarded every step on the first
        # param's grad address: if autograd ever handed out a fresh grad tensor (which is
        # what set_to_none=True would cause) the buffer would be stale and the update would
        # silently use zeros, so fall back to stacking for the rest of the run.
        gbuf = group.get('_stacked_grads')
        if gbuf is not None and (params[0].grad is not None
                                 and params[0].grad.data_ptr() == gbuf.data_ptr()):
            stacked_grads, stacked_params = gbuf, group['_stacked_params']
            copy_back = False
        else:
            if gbuf is not None:
                print(f"\n[stacked] grad views LOST for shape {tuple(shape)}, "
                      f"falling back to torch.stack", flush=True)
                group['_stacked_grads'] = None
            stacked_grads = torch.stack([q.grad for q in params])
            stacked_params = torch.stack(params)
            copy_back = True
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        self._muon_apply(state, stacked_grads, stacked_params, red_dim, group["ns_steps"])
        if copy_back:
            torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    def _muon_apply(self, state, stacked_grads, stacked_params, red_dim, ns_steps):
        """NODE 5.5: run the fused Muon step over sub-slices of the stacked group.

        Every operation in muon_step_fused is per-matrix (all reductions carry keepdim over
        the last two dims), so slicing along dim 0 is BIT-IDENTICAL arithmetic while the
        Newton-Schulz temporaries (X, A, A@A, X@B and the fp32 NorMuon square) shrink by
        n/MUON_CHUNK. This is what removes the ~0.1GB transient workspace of the step.
        """
        n = stacked_grads.size(0)
        chunk = MUON_CHUNK if MUON_CHUNK > 0 else n
        mbuf, sbuf = state["momentum_buffer"], state["second_momentum_buffer"]
        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            muon_step_fused(stacked_grads[s:e], stacked_params[s:e], mbuf[s:e], sbuf[s:e],
                            self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                            self._muon_beta2_t, ns_steps, red_dim)

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

# Optimization
# TOTAL_BATCH_SIZE is DERIVED from (DEVICE_BATCH_SIZE, GRAD_ACCUM_STEPS) instead of being a
# fixed power of two: the microbatch is the memory lever and it need not divide 2**19. See
# the DEVICE_BATCH_SIZE / GRAD_ACCUM_STEPS block below for the chosen point.
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# Model size
DEPTH = 8               # number of transformer layers
DEVICE_BATCH_SIZE = 1   # per-device batch size. Peak VRAM is dominated by the retained
                        # block activations of ONE microbatch (~0.43GB per sequence over the
                        # 8 layers, plus ~0.2GB/sequence of fp32 loss-path copies), so this is
                        # the primary memory lever and it scales near-linearly. It does NOT
                        # have to be a power of two: TOTAL_BATCH_SIZE is derived below, so any
                        # db is legal as long as tokens_per_step stays a constant across steps.
                        # MEASURED peak: db=8 5.598GB, 6 4.702, 4 3.227, 3 2.582, 2 2.253.
                        # db=2 breaks the linear trend (only -13% vs db=3, the base 0.53GB of
                        # params+optimizer and the 0.54GB single loss chunk start to dominate).
                        # db=2 WAS ineligible in the eager loop (223K tok/s, 134M tokens,
                        # val_bpb 1.0714) because the ~4-6ms of FIXED host-side cost per
                        # microbatch is paid 39 times per step instead of 26. With the
                        # microbatch captured as a CUDA graph that cost is ~0.1ms, so db=2 is
                        # revisited here.
                        # NODE 4.2 launch 2: with the fused loss tail composed in, db=2 ran at
                        # 400K tok/s (241.1M tokens) and val_bpb 1.027074 -- a 2.3e-2 gate
                        # margin -- at peak 1,102,654,464 set by the graph's private pool. That
                        # surplus is spent here on db=1: the pool holds one sequence instead of
                        # two, so the resident-activation term halves again.
GRAD_ACCUM_STEPS = 52   # tokens/step = 1*2048*52 = 106,496. MEASURED at db=1 with accum=78
                        # (159,744 tok/step, the db=2/db=3 constant): peak 945,827,840 and
                        # val_bpb 1.047861 -- eligible but only 2.1e-3 above the gate, because
                        # db=1 costs 26% of the tokens (178.9M vs 241.1M at db=2: the GPU spends
                        # 3.32us/token at db=1 vs 2.44us/token at db=2, the matmuls are simply
                        # less efficient). Peak VRAM at db=1 is NOT set by the microbatch any
                        # more (the graph pool peaks at 0.789GB) but by the optimizer step
                        # (0.946GB), so tokens/step is now free on the memory axis and is spent
                        # on quality instead: every previous sweep of this axis improved val_bpb
                        # monotonically as the step shrank at fixed FLOPs/token
                        # (db=6: 528,384 -> 319,488 gave +0.0195; db=3: 319,488 -> 196,608 ->
                        # 159,744 gave 1.049941 -> 1.048698 -> 1.046288), since TOTAL_BATCH_SIZE
                        # is far above the critical batch size. 106,496 costs only ~7ms of extra
                        # optimizer time per step out of ~360ms.
                        # History of this axis at db=3 (all at fixed FLOPs/token):
                        #   319,488 -> val_bpb 1.049941 (203.5M tokens, 339K tok/s)
                        #   196,608 -> val_bpb 1.048698 (187.2M tokens, 309K tok/s)
                        #   159,744 -> val_bpb 1.046288 (183.9M tokens, 304K tok/s)  <-- best
                        # Quality improves monotonically as the step shrinks even though FEWER
                        # tokens are processed (a shorter step pays ~2ms more python/sync per
                        # microbatch), so the currency is still buying more than the tokens it
                        # costs at 159,744. This is what turns db=3's 6e-5 margin into 3.7e-3.

# Per-step overhead removal, MEASURED and reported honestly: the threaded prefetcher below
# does remove the dataloader from the critical path (data_wait 317-494ms/step, 28-41% of the
# step, falls to 7-21ms ~1%) but buys NO throughput -- dt stayed ~1150ms, so the step is
# GPU-bound and the python packing was already overlapping with GPU work. It is therefore
# left DISABLED (depth 0), since a worker thread racing the loader's reusable pinned buffer
# is pure risk for zero gain. Set PREFETCH_DEPTH>0 to re-run the A/B. The chunked-loss side
# of the overhead fix DID pay: one chunk per microbatch raised throughput from ~408K tok/s
# (db=8 trunk, 4 chunks x 32 microbatches) to ~458K tok/s at db=6.
PREFETCH_DEPTH = 0
PREFETCH_AB_STEPS = 6

# ---------------------------------------------------------------------------
# CUDA-graph capture of ONE accumulation microbatch.
#
# A microbatch at db=3 is ~15ms of which ~3.8ms is host-side: dynamo guard evaluation, the
# eager @torch.compiler.disable'd loss tail, the autograd engine, and 1000+ individual kernel
# launches. That term is FIXED per microbatch, so it grows as a fraction as db shrinks (26
# microbatches/step at db=3, 39 at db=2). Capturing forward+backward of one microbatch -- with
# static x/y input buffers and gradients accumulating IN PLACE into pre-allocated .grad
# buffers -- and replaying it per microbatch replaces all of that with a single graph launch.
# The optimizer step, the LR schedule and the dataloader stay outside the graph; only the
# H2D-free copy of the next microbatch into the static buffers is added.
#
# Two things are checked at runtime before the graph is trusted, because both failure modes
# are silent:
#   (a) .grad addresses must not move during capture, and two replays must produce exactly 2x
#       the gradient of one replay -- if autograd chose OUT-OF-PLACE accumulation the captured
#       kernels would overwrite instead of accumulate, silently training on 1/accum of the
#       gradient. On failure we fall back to the eager loop.
#   (b) peak VRAM: the graph's private memory pool counts toward max_memory_allocated, and the
#       pool is not shared with the frozen validation forward, so it is freed before the
#       final evaluation and the peak is printed right after capture.
CUDA_GRAPH_MICROBATCH = True
GRAPH_WARMUP_ITERS = 3

# Muon's params/grads are rebound to slices of one contiguous buffer per shape group, which
# removes the per-step torch.stack duplication from the optimizer step (the step is what sets
# the reported peak at db=1). Set False to restore the stacking path.
MUON_STACKED_VIEWS = True

# Release gradients + optimizer moments + the graph pool before the frozen validation
# forward. The reported peak is the max over the WHOLE process, and node 4.2 measured the
# eval forward at 0.95GB, level with the 0.946GB optimizer-step peak, of which ~0.5GB is
# grads+moments that the eval does not read. Freeing them is what lets an optimizer-side
# reduction show up in the reported number at all.
RELEASE_BEFORE_EVAL = True
TOTAL_BATCH_SIZE = DEVICE_BATCH_SIZE * MAX_SEQ_LEN * GRAD_ACCUM_STEPS

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
_matrix_elems_bf16 = model.compress_matrix_params()
_embed_elems_bf16 = model.compress_embedding_params()
print(f"[matrix-dtype] transformer.h weights held in {MATRIX_PARAM_DTYPE}: "
      f"{_matrix_elems_bf16:,} elems, embedding tables in {EMBED_PARAM_DTYPE}: "
      f"{_embed_elems_bf16:,} elems, params resident "
      f"{sum(p.numel()*p.element_size() for p in model.parameters())/1e9:.3f}GB", flush=True)

param_counts = count_params(model)
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE == tokens_per_fwdbwd * GRAD_ACCUM_STEPS
grad_accum_steps = GRAD_ACCUM_STEPS
print(f"Tokens per step: {TOTAL_BATCH_SIZE:,} = {DEVICE_BATCH_SIZE} x {MAX_SEQ_LEN} x {grad_accum_steps}")

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
# Selective-recomputation smoke probe, EAGER (the frozen FLOPs probe below also runs the
# uncompiled module). If the SAC context/policy is unusable on this torch build, disable the
# feature here instead of losing the whole launch; the compiled path is guarded the same way
# at the first microbatch.
if SELECTIVE_RECOMPUTE:
    try:
        with autocast_ctx:
            _probe_loss = model._orig_mod(x[:1], y[:1])
        _probe_loss.backward()
        del _probe_loss
        print(f"[recompute] eager smoke probe OK "
              f"(attn_qkv_recompute={RECOMPUTE_ATTN_QKV}) "
              f"alloc={torch.cuda.memory_allocated()/1e9:.3f}GB "
              f"peak={torch.cuda.max_memory_allocated()/1e9:.3f}GB", flush=True)
    except Exception as exc:
        SELECTIVE_RECOMPUTE = False
        RECOMPUTE_ATTN_QKV = False
        print(f"[recompute] DISABLED, eager smoke probe raised "
              f"{type(exc).__name__}: {exc}", flush=True)
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
# Dispatch-level FLOPs, measured HERE while memory still holds only the model and the
# optimizer state. Counted on the uncompiled module by the frozen probe.
from prepare import measure_flops_dispatch
flops_per_token_measured = measure_flops_dispatch(model, x, y)
print(f"FLOPs per token (measured at dispatch): {flops_per_token_measured:,}")
FLOPS_CAP = 239_078_400
print(f"[flops-invariant] measured={flops_per_token_measured:,} cap={FLOPS_CAP:,} "
      f"delta={flops_per_token_measured - FLOPS_CAP:+,} "
      f"selective_recompute={SELECTIVE_RECOMPUTE} "
      f"attn_qkv_recompute={RECOMPUTE_ATTN_QKV} "
      f"eager_tail_rows={_EAGER_TAIL_ROWS} "
      f"alloc={torch.cuda.memory_allocated()/1e9:.3f}GB "
      f"peak_after_frozen_probe={torch.cuda.max_memory_allocated()/1e9:.3f}GB", flush=True)

# ---------------------------------------------------------------------------
# Optimizer-memory install. Done AFTER the frozen FLOPs probe (which ends with
# zero_grad(set_to_none=True)) and BEFORE the first compiled call / graph capture, so the
# parameter addresses the compiled kernels and the captured graph bake in are already the
# final, stacked ones.
# ---------------------------------------------------------------------------
if MUON_STACKED_VIEWS:
    _stacked_elems = optimizer.install_stacked_views()
    # Autograd only accumulates IN PLACE into an existing .grad; if it ever replaced the
    # tensor, the stacked buffer would go stale and Muon would update on zeros. One real
    # eager fwd/bwd proves it before anything depends on it.
    with autocast_ctx:
        _vl = model._orig_mod(x[:1], y[:1])
    _vl.backward()
    del _vl
    _views_ok = optimizer.stacked_grads_intact()
    _gsum = sum(g.abs().sum().item()
                for grp in optimizer.param_groups
                if grp.get('_stacked_grads') is not None
                for g in [grp['_stacked_grads']])
    print(f"[stacked] muon params in stacked buffers: {_stacked_elems:,} elems "
          f"grads_accumulate_in_place={_views_ok} stacked_grad_abs_sum={_gsum:.6e}",
          flush=True)
    if not (_views_ok and _gsum > 0):
        print("[stacked] DISABLED: autograd did not write into the stacked grad buffers",
              flush=True)
        optimizer.drop_stacked_grad_views()
    for _p in model.parameters():
        if _p.grad is not None:
            _p.grad.zero_()
    gc.collect()
    torch.cuda.empty_cache()
_kahan_bytes = 0
if EMBED_KAHAN_ACCUM:
    _kahan_bytes = optimizer.install_kahan_accums()
    # Prove the compensated accumulation really is more accurate than the naive bf16 sum that
    # node 2.2 measured at 9.8e-1 relative error: accumulate the SAME per-microbatch gradient
    # `n` times, once naively in bf16 and once through the Kahan merge, and compare both with
    # the fp32 reference. Uses the gradients the verification pass above just produced.
    _ref = []
    with torch.no_grad():
        for _p, _acc in optimizer._kahan_pairs[:1]:
            _g = _p.grad
            if _g is None:
                continue
            _one = (torch.randn(_p.shape, device=_p.device) * 1e-3).bfloat16()
            _naive = torch.zeros_like(_g)
            _acc.zero_()
            _g.zero_()
            for _i in range(grad_accum_steps):
                _naive.add_(_one)         # naive bf16 running sum (node 2.2's regime)
                _g.add_(_one)             # autograd adds on top of the Kahan compensation
                kahan_merge_(_acc, _g)    # ... which this leaves in _g again
            _exact = _one.float() * grad_accum_steps
            _rel_n = ((_naive.float() - _exact).norm() / _exact.norm()).item()
            _rel_k = ((_acc.float() - _exact).norm() / _exact.norm()).item()
            _ref.append((tuple(_p.shape), _rel_n, _rel_k))
            _acc.zero_()
            _g.zero_()
            del _one, _naive, _exact
    for _shape, _rel_n, _rel_k in _ref:
        print(f"[kahan] {_shape} {grad_accum_steps}-way accumulation rel-err "
              f"naive_bf16={_rel_n:.3e} kahan_bf16={_rel_k:.3e}", flush=True)
    print(f"[kahan] accumulators installed: {_kahan_bytes/1e9:.3f}GB "
          f"tables={len(optimizer._kahan_pairs)} sr_store={optimizer._sr_ok}", flush=True)
    gc.collect()
    torch.cuda.empty_cache()
print(f"[opt-mem] after install: alloc={torch.cuda.memory_allocated()/1e9:.3f}GB "
      f"peak={torch.cuda.max_memory_allocated()/1e9:.3f}GB "
      f"params={sum(p.numel()*p.element_size() for p in model.parameters())/1e9:.3f}GB "
      f"grads={sum(p.grad.numel()*p.grad.element_size() for p in model.parameters() if p.grad is not None)/1e9:.3f}GB "
      f"kahan={_kahan_bytes/1e9:.3f}GB "
      f"matrix_dtype={MATRIX_PARAM_DTYPE} embed_dtype={EMBED_PARAM_DTYPE} "
      f"muon_chunk={MUON_CHUNK} "
      f"moment_dtype={OPT_MOMENT_DTYPE} stacked_views={optimizer.stacked_views}", flush=True)

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

class BatchPrefetcher:
    """Runs the (pure-python, main-thread) dataloader on a worker thread.

    Correctness details, both of which matter once the worker runs AHEAD of the GPU:
      * the loader yields views into ONE reusable pinned CPU buffer + ONE reusable GPU
        buffer, so the worker hands over private clones (2*db*2048 int64 = a few hundred
        KB, negligible next to the microbatch activations);
      * the loader's H2D copy is non_blocking, so the pinned buffer may not be re-filled
        until that copy has actually executed. The worker therefore runs the loader and
        the clone on its OWN stream and blocks (on the worker thread only) on an event
        after the clone, which waits for the tiny copy chain and NOT for the main stream's
        fwd/bwd queue. The consumer records the clones on the main stream so the caching
        allocator cannot hand their memory back to the worker stream while they are in use.
    """

    def __init__(self, loader, depth):
        self.q = queue.Queue(maxsize=depth)
        self.loader = loader
        self.stream = torch.cuda.Stream()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        with torch.cuda.stream(self.stream):
            for x, y, epoch in self.loader:
                xc, yc = x.clone(), y.clone()
                self.stream.synchronize()
                self.q.put((xc, yc, epoch))

    def next(self):
        x, y, epoch = self.q.get()
        x.record_stream(torch.cuda.current_stream())
        y.record_stream(torch.cuda.current_stream())
        return x, y, epoch


def _capture_microbatch_graph(x0, y0):
    """Warm up on a side stream, then capture fwd+bwd of one microbatch."""
    sx, sy = x0.clone(), y0.clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(GRAPH_WARMUP_ITERS):
            with autocast_ctx:
                l = model(sx, sy)
            (l / grad_accum_steps).backward()
            del l
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    # Pre-allocate/keep .grad at fixed addresses: the captured accumulate kernels bake those
    # addresses in, so .grad must never be set to None again while the graph lives.
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"no grad from warmup for {missing}"
    for p in model.parameters():
        p.grad.zero_()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        with autocast_ctx:
            l = model(sx, sy)
        sloss = l.detach()
        (l / grad_accum_steps).backward()
    del l
    return g, sx, sy, sloss


graph = None
static_x = static_y = static_loss = None
copy_done = torch.cuda.Event()
if CUDA_GRAPH_MICROBATCH:
    _pre_ptrs = None
    try:
        _alloc_pre = torch.cuda.memory_allocated()
        graph, static_x, static_y, static_loss = _capture_microbatch_graph(x, y)
        _pre_ptrs = [p.grad.data_ptr() for p in model.parameters()]
        # (a) accumulation check: 2 replays must give exactly 2x the gradient of 1 replay
        for p in model.parameters():
            p.grad.zero_()
        graph.replay()
        _g1 = torch.stack([p.grad.float().abs().sum() for p in model.parameters()]).sum().item()
        graph.replay()
        _g2 = torch.stack([p.grad.float().abs().sum() for p in model.parameters()]).sum().item()
        _post_ptrs = [p.grad.data_ptr() for p in model.parameters()]
        _ratio = _g2 / _g1 if _g1 > 0 else float('nan')
        _ptrs_ok = _post_ptrs == _pre_ptrs
        _acc_ok = _ptrs_ok and abs(_ratio - 2.0) < 1e-3
        print(f"\n[graph] capture OK warmup={GRAPH_WARMUP_ITERS} "
              f"loss={static_loss.item():.4f} grad_sum_1replay={_g1:.6e} "
              f"2replays={_g2:.6e} ratio={_ratio:.6f} grad_ptrs_stable={_ptrs_ok} "
              f"accumulates={_acc_ok}", flush=True)
        print(f"[graph] mem after capture: alloc={torch.cuda.memory_allocated()/1e9:.3f}GB "
              f"(pre-capture {_alloc_pre/1e9:.3f}GB) "
              f"peak={torch.cuda.max_memory_allocated()/1e9:.3f}GB "
              f"reserved={torch.cuda.memory_reserved()/1e9:.3f}GB", flush=True)
        if not _acc_ok:
            raise RuntimeError(f"captured graph does not accumulate into .grad "
                               f"(ratio={_ratio}, ptrs_stable={_ptrs_ok})")
        for p in model.parameters():
            p.grad.zero_()
    except Exception as exc:
        print(f"\n[graph] DISABLED: {type(exc).__name__}: {exc}", flush=True)
        graph = None
        static_x = static_y = static_loss = None
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()

prefetcher = None
t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
tokens_consumed = 0
step = 0
TRACE_TOKEN_INTERVAL = 20_000_000
next_trace_tokens = TRACE_TOKEN_INTERVAL

while True:
    torch.cuda.synchronize()
    t0 = time.time()
    data_wait = 0.0
    timed_mb = step <= 11
    if timed_mb:
        ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ev0.record()
        t_mb0 = time.time()
    for micro_step in range(grad_accum_steps):
        if graph is not None:
            # The loader hands out views into ONE reusable pinned CPU buffer and ONE reusable
            # GPU buffer and issues a non_blocking H2D between them, so the host may not
            # refill the pinned buffer until the previous microbatch's copy has EXECUTED.
            # Eager mode was throttled by the >1000 kernel launches per microbatch filling
            # the CUDA queue; one graph replay is a single launch, so the host would otherwise
            # race a whole step ahead and the loader would overwrite the staging buffer under
            # in-flight copies (MEASURED: that races the data and costs ~1.4 train loss).
            # The event is recorded AFTER the static copies and waited on AFTER the replay is
            # enqueued, so exactly one replay is in flight: the GPU is never idle while the
            # host prepares the next microbatch, and the pinned buffer is never reused early.
            static_x.copy_(x, non_blocking=True)
            static_y.copy_(y, non_blocking=True)
            copy_done.record()
            graph.replay()
            copy_done.synchronize()
            train_loss = static_loss
            if step <= 11 and micro_step < 2:
                assert torch.equal(static_x[:, 1:], static_y[:, :-1]), \
                    "graph input buffers are inconsistent (torn H2D copy)"
        else:
            try:
                with autocast_ctx:
                    loss = model(x, y)
            except Exception as exc:
                if not (SELECTIVE_RECOMPUTE or RECOMPUTE_ATTN_QKV):
                    raise
                # Compiling the selective-checkpoint region failed: fall back to the plain
                # forward rather than losing the launch.
                SELECTIVE_RECOMPUTE = False
                RECOMPUTE_ATTN_QKV = False
                print(f"\n[recompute] DISABLED under compile, raised "
                      f"{type(exc).__name__}: {exc}", flush=True)
                torch._dynamo.reset()
                with autocast_ctx:
                    loss = model(x, y)
            if step == 2 and micro_step == 0:
                print(f"\n[mem] after fwd  : alloc={torch.cuda.memory_allocated()/1e9:.2f}GB "
                      f"peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)
            train_loss = loss.detach()
            loss = loss / grad_accum_steps
            loss.backward()
            if step == 2 and micro_step == 0:
                print(f"[mem] after bwd  : alloc={torch.cuda.memory_allocated()/1e9:.2f}GB "
                      f"peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)
        # NODE 5.5: the bf16 embedding gradient of THIS microbatch is absorbed into its Kahan
        # accumulator now, and the rounding residual is left in p.grad as the compensation
        # that autograd's next in-place accumulation starts from. Deliberately OUTSIDE the
        # captured graph: it needs no new buffer, is one fused kernel per table, and keeps the
        # graph's baked-in .grad addresses untouched.
        optimizer.merge_kahan_grads()
        t_data = time.time()
        x, y, epoch = next(train_loader) if prefetcher is None else prefetcher.next()
        data_wait += time.time() - t_data
        # Integrity check for the prefetched path: the loader builds inputs = row[:-1] and
        # targets = row[1:] from the same packed row, so x[:, 1:] must equal y[:, :-1]. A
        # torn / not-yet-executed H2D copy would break this. Checked on the untimed steps.
        if prefetcher is not None and step <= 11:
            assert torch.equal(x[:, 1:], y[:, :-1]), "prefetched batch is inconsistent"
    if timed_mb:
        # Host-overhead diagnostic: wall time of the accumulation loop (minus the dataloader)
        # versus the GPU time it enqueued. The gap is the per-microbatch host cost that the
        # captured graph is meant to remove.
        ev1.record()
        mb_wall = time.time() - t_mb0 - data_wait
        torch.cuda.synchronize()
        mb_gpu = ev0.elapsed_time(ev1) / 1000.0
        print(f"\n[mb] step {step:02d} graph={graph is not None} n={grad_accum_steps} "
              f"wall={mb_wall*1000:.1f}ms gpu={mb_gpu*1000:.1f}ms "
              f"per_mb_wall={mb_wall*1000/grad_accum_steps:.2f}ms "
              f"per_mb_gpu={mb_gpu*1000/grad_accum_steps:.2f}ms "
              f"host_gap={max(0.0, mb_wall-mb_gpu)*1000/grad_accum_steps:.2f}ms", flush=True)

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
    if step <= 11:
        torch.cuda.synchronize()
        t_opt = time.time()
    optimizer.step()
    # With a captured graph the .grad buffers must keep their addresses, so they are zeroed
    # in place instead of being released. The same holds for the stacked Muon grad views:
    # releasing them would make autograd hand out fresh tensors and the stacked buffer stale.
    model.zero_grad(set_to_none=graph is None and not optimizer.stacked_views)
    if step <= 11:
        torch.cuda.synchronize()
        opt_seconds = time.time() - t_opt
        # NODE 4.2 diagnostic: at db=1 the captured microbatch pool peaks at only 0.789GB but
        # the reported peak is 0.946GB, so the OPTIMIZER STEP (Muon's Newton-Schulz workspace +
        # the fused AdamW temporaries on top of params+grads+optimizer state) is what now
        # bounds the configuration. Print its own contribution so the next node can target it.
        if step in (2, 11):
            print(f"[opt-mem] step {step:02d} alloc={torch.cuda.memory_allocated()/1e9:.3f}GB "
                  f"peak={torch.cuda.max_memory_allocated()/1e9:.3f}GB "
                  f"reserved={torch.cuda.memory_reserved()/1e9:.3f}GB "
                  f"opt_ms={opt_seconds*1000:.1f} "
                  f"state={optimizer.state_bytes()/1e9:.3f}GB "
                  f"grads={sum(p.grad.numel()*p.grad.element_size() for p in model.parameters() if p.grad is not None)/1e9:.3f}GB "
                  f"params={sum(p.numel()*p.element_size() for p in model.parameters())/1e9:.3f}GB "
                  f"stacked={optimizer.stacked_views}", flush=True)
    else:
        opt_seconds = 0.0

    train_loss_f = train_loss.item()

    # Fast fail: abort if loss is exploding or NaN
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    # Per-step overhead A/B, all inside the untimed warmup steps: report the step rate with
    # the synchronous loader and then with the prefetching worker, plus the time the main
    # thread actually spent blocked in the dataloader.
    if step <= 11:
        print(f"\n[rate] step {step:02d} mode={'sync' if prefetcher is None else 'prefetch'} "
              f"dt={dt*1000:.0f}ms data_wait={data_wait*1000:.0f}ms "
              f"({100*data_wait/dt:.1f}%) opt={opt_seconds*1000:.0f}ms "
              f"tok/s={int(TOTAL_BATCH_SIZE/dt):,}", flush=True)
    if PREFETCH_DEPTH > 0 and prefetcher is None and step + 1 >= PREFETCH_AB_STEPS:
        # x, y are views into the loader's reusable buffer; the worker is about to overwrite
        # it, so take ownership of the batch that is already in flight.
        x, y = x.clone(), y.clone()
        prefetcher = BatchPrefetcher(train_loader, PREFETCH_DEPTH)
        print(f"\n[prefetch] worker started after step {step} (depth={PREFETCH_DEPTH})", flush=True)

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

    # Token-milestone loss trace: the per-step \r line is not comparable across configs
    # (different tokens/step), so emit the smoothed train loss at fixed TOKEN milestones.
    # This is how the (db, tokens/step) points are compared at equal tokens processed.
    if tokens_consumed >= next_trace_tokens:
        print(f"\n[trace] tokens={tokens_consumed:,} step={step} "
              f"loss={debiased_smooth_loss:.5f} t={total_training_time:.0f}s", flush=True)
        while next_trace_tokens <= tokens_consumed:
            next_trace_tokens += TRACE_TOKEN_INTERVAL

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

print(f"[mem] end of training: alloc={torch.cuda.memory_allocated()/1e9:.2f}GB "
      f"peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)

# Release the graph's private memory pool before the frozen validation forward: the pool holds
# one microbatch's activations and is NOT reused by the evaluator, so leaving it alive could
# make training-pool + eval the reported peak.
if graph is not None:
    train_loss = train_loss.clone()
    graph.reset()
    del graph, static_x, static_y, static_loss
    graph = None
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[graph] pool released: alloc={torch.cuda.memory_allocated()/1e9:.3f}GB "
          f"reserved={torch.cuda.memory_reserved()/1e9:.3f}GB "
          f"peak={torch.cuda.max_memory_allocated()/1e9:.3f}GB", flush=True)

# ---------------------------------------------------------------------------
# NODE 5.2: minimise residency at the frozen validation forward.
#
# peak_vram_bytes is the max over the WHOLE process. Node 4.2 measured the eval forward at
# 0.95GB, essentially tied with the 0.946GB optimizer-step peak -- and roughly half of that
# 0.95GB is gradients + optimizer moments that evaluate_bpb never reads. While the eval term
# sits level with the training term, ANY training-side reduction is invisible in the reported
# number. So, after the training loop and before the single report_efficiency_metrics call,
# release everything the evaluation does not need: gradients (set_to_none), every optimizer
# state tensor, the stacked Muon grad buffer, and the dataloader / prefetch buffers.
#
# The reporter is handed the SAME trained model object with the SAME arguments, and
# torch.cuda.reset_peak_memory_stats() is never called: nothing about how the number is
# measured changes, only how much memory is alive while it is measured.
# ---------------------------------------------------------------------------
if RELEASE_BEFORE_EVAL:
    _alloc_before_release = torch.cuda.memory_allocated()
    # train_loss / x / y are views into loader or graph-pool buffers.
    train_loss = train_loss.clone()
    del x, y
    prefetcher = None
    train_loader = None
    model.zero_grad(set_to_none=True)
    _freed_state = optimizer.release_state()
    for _g in optimizer.param_groups:
        _g['params'] = []
    optimizer = None
    gc.collect()
    torch.cuda.empty_cache()
    _alloc_after_release = torch.cuda.memory_allocated()
    _resident_params = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"[eval-prep] released {(_alloc_before_release - _alloc_after_release)/1e9:.3f}GB "
          f"(optimizer state+grads accounted {_freed_state/1e9:.3f}GB) "
          f"alloc={_alloc_after_release/1e9:.3f}GB (params alone {_resident_params/1e9:.3f}GB) "
          f"reserved={torch.cuda.memory_reserved()/1e9:.3f}GB "
          f"peak={torch.cuda.max_memory_allocated()/1e9:.3f}GB "
          f"grads_live={sum(1 for p in model.parameters() if p.grad is not None)}", flush=True)

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
