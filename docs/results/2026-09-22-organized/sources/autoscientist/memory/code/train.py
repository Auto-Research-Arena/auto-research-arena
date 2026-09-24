"""intercept_exp_avg_host_resident -- the first moment of `wte` and `lm_head` lives in PINNED HOST
memory and is staged to the device inside the optimizer step. The update is BITWISE identical to
champion v35; the only thing that changes is where 16,777,216 B sleeps between steps.

MECHANISM (INTERCEPT `queue.md` v61, [PROPOSAL] 77499a03, @autoscts__memory_analyst2). Champion v35's
`[optstate]` line is `total=17,158,712 exp_avg:bfloat16=16,777,244`, and 16,777,216 of that is two
tables: `wte` and `lm_head`, 8,388,608 each. They are the ONLY remaining resident optimizer state on
this board. There are exactly two routes to those bytes and this is the one that costs no quality:

  * DELETION (`beta1 = 0`, TAIL's `tail_adamw_beta1_zero_dense_riderfree`) -- changes the update, and
    my own two paid launches price it at +0.0022270 of `val_bpb` (launch 46's full six-table scope
    +0.0025795 token-corrected, MINUS launch 48's four-table VE-only +0.0003525 paired against gpu2's
    byte-neutral 0047). Champion v35 sits at 1.0457713376389435 and standing rule 6 refuses promotion
    at 1.048, so that route has 0.0000017 of margin -- 300x under `noise_floor.md` v7's
    `val_bpb_readable_floor` of 0.0005. It cannot be pre-registered as promotable.
  * RESIDENCE (this candidate) -- does not change the update at all. `adamw_step_fused_factored` runs
    unmodified on a device copy of the same bf16 bytes, so `val_bpb` is a function of dt ONLY.

THE DIFF, three hunks, all unconditional and none of them keyed on a shape, a batch size or an
experiment id:

  (a) `OPT_STATE_HOST_RESIDENT = True` beside `VALUE_EMBED_BETA1`.
  (b) `MuonAdamW._step_adamw` lazy allocation: for a 2-D AdamW parameter that keeps its first moment,
      allocate `state['exp_avg']` as `torch.zeros(p.shape, dtype=OPT_STATE_DTYPE, device="cpu",
      pin_memory=True)` instead of `torch.zeros_like(p, ...)`. The predicate is
      `OPT_STATE_HOST_RESIDENT and p.ndim == 2` INSIDE the existing `if not no_first_moment:`, so it
      selects exactly `wte` and `lm_head`: the four value-embedding tables already take the
      `no_first_moment` branch and allocate nothing, and `resid_lambdas` / `x0_lambdas` are 1-D and
      keep their dense device buffers byte for byte.
  (c) the `elif p.ndim == 2:` dispatch: if `exp_avg` is not on the device, stage it with
      `.to(p.device, non_blocking=True)`, call the champion's `adamw_step_fused_factored` on the
      staged copy UNCHANGED, then write it back with `copy_(..., non_blocking=False)`. The `is_cuda`
      test means a device-resident buffer still takes the champion's exact path, so hunk (c) is a
      no-op when hunk (a) is False.

WHY THE BYTES ARE CERTAIN: THIS IS A PAID PRE-IMAGE, NOT A MODEL. Launch 46 (mine) also ran with no
device `exp_avg` for `wte`/`lm_head`, so its `[mem]` ladder IS this candidate's ladder. Diffed against
launch 49 (= champion v35) line by line, every instant from `s0_after_opt` onward differs by EXACTLY
16,777,216 and every instant before it is byte-identical:

  [mem]                 launch 46 = PREDICTED HERE          launch 49 = v35          diff
  before_probe     alloc  94,964,736 peak 197,659,648        same                       0
  after_probe      alloc 112,004,096 peak 565,531,136        same                       0
  s0m0_after_bwd   alloc 206,380,032 peak 565,531,136        same                       0
  s0_after_accum   alloc 206,380,032 peak 572,174,848        same                       0
  s0_after_opt     alloc 206,763,520 peak 572,174,848   alloc 223,540,736    -16,777,216
  s1m0_after_fwd   alloc 388,006,912 peak 572,174,848   alloc 404,784,128    -16,777,216
  s1_after_accum   alloc 206,763,520 peak 572,558,336 <- ARM  peak 589,335,552 -16,777,216
  before_reporter  alloc 156,104,704 peak 572,558,336        peak 589,335,552 -16,777,216

  [optstate] total=381,496 exp_avg:bfloat16=28 exp_avg_sq:bfloat16=28
             second_momentum_buffer:float32=172,544 v_c:float32=12,288 v_r:float32=196,608

  headline peak_vram_bytes  572,558,336  = 589,335,552 - 16,777,216, EXACT, +/- 0 B

The `[optstate]` line above is a CORRECTION to the queue row, which pre-registered `total 381,468`
with `exp_avg` reading 0 or absent. The `[optstate]` reporter tallies `optimizer.state.values()` under
`torch.is_tensor(_v) and _v.is_cuda`, and this diff relocates only the two 2-D buffers --
`resid_lambdas` + `x0_lambdas` are `(7 + 7) x 2` bf16 = 28 B and stay on the device. So `exp_avg`
reads 28, not 0, and the total is 381,496, which is what launch 46 measured. The same 28 B has now
caught three rows on this axis, including my own launch-46 pre-registration.

THE STAGING TRANSIENT IS MASKED BY A FACTOR OF 43.6, so it cannot reach the headline. It is one table
at a time, 8,388,608 B, allocated inside the per-parameter loop and freed when the loop advances. At
that instant launch 46 read `s0_after_opt allocated = 206,763,520` against a then-peak of 572,174,848:
365,411,328 B of headroom. `report_memory` is also called BEFORE `optimizer.step()`, so no ladder line
can observe it at all.

THE WHOLE RISK IS dt, AND IT IS THE ONLY THING THIS LAUNCH BUYS THAT IS NOT ALREADY PAID. Traffic is
16,777,216 B in + 16,777,216 B out per step = 33,554,432 B. Champion v35 measured median dt 213.0 ms
and the falsifier is 225 ms, so the budget is 12 ms and the transfer must sustain 2.80 GB/s. Effective
staged bandwidth on this lane has never been measured in this run; that measurement is the point.
Because the arithmetic is bitwise, `val_bpb` moves ONLY through the token account, so dt and val_bpb
are one falsifier read two ways.

PRE-REGISTRATION (13 columns EXACT, one band):
  peak_vram_bytes                572,558,336   EXACT +/- 0 B  (paid pre-image, launch 46)
  s1_after_accum          alloc  206,763,520   peak 572,558,336
  s0_after_opt            alloc  206,763,520   peak 572,174,848
  s1m0_after_fwd          alloc  388,006,912
  after_probe             alloc  112,004,096   peak 565,531,136   UNCHANGED
  before_probe            alloc   94,964,736   peak 197,659,648   UNCHANGED
  before_reporter         alloc  156,104,704   peak 572,558,336
  [optstate] total            381,496   exp_avg:bfloat16 = 28
  flops_per_token_measured  229,641,216   EXACT (ceiling 239,078,400)
  num_params_total           47,186,446   EXACT (ceiling 50,332,176)
  training_data_tokens_available 631,241,817  EXACT (equality constraint)
  median_dt_ms                     <= 225   falsifier
  val_bpb                    band [1.0458, 1.0480], predicted ~1.0466 at 3 ms of staging

FALSIFIERS:
 (1) peak != 572,558,336 -> either a pinned HOST tensor is being charged by
     torch.cuda.max_memory_allocated, which would re-open every residence question on this task, or
     the staging transient is NOT masked. Both are large results. sigma on this metric is 0 and the
     band is 512 B (`noise_floor.md` v7), so any difference is real.
 (2) [optstate] total != 381,496 or exp_avg != 28 -> the relocation did not take. NOT 381,468.
 (3) flops != 229,641,216 or params != 47,186,446 or data_tokens != 631,241,817 -> build error,
     FAILED and not DISCARD. This candidate changes no counted arithmetic whatsoever.
 (4) median dt > 225 ms -> staged bandwidth is under 2.80 GB/s and the residence route closes on this
     lane; the deletion route is then the only route to these bytes and it is a coin flip against
     standing rule 6, so the 16,777,216 B is effectively unbuyable and INTERCEPT's board floor rises
     from 565,531,136 to 572,558,336.
 (5) val_bpb moves by more than the token account explains -> the copies are not bitwise. FAILED.
 (6) before_probe or after_probe move -> the diff reached the eager probe window, which it must not:
     optimizer state does not exist yet at either instant.

CARRY / DECLINED: nothing. No rider, no instrument, no new print -- the champion's three read-only
`report_memory` reads are inherited unchanged and are sufficient for every column above. I also owe
the board one FREE reading that needs no code: `task.json` says the headline includes VALIDATION, run
at a frozen 128 x 2048 AFTER `before_reporter`, and no ladder line observes it. Comparing
`before_reporter`'s peak to the final METRICS_JSON headline bounds that window at zero cost, and I
will report it whatever it says -- the tightest bound this run owns is launch 46's "below
572,558,336", which is exactly this candidate's arm.

CREDIT: the row, the residence argument and the 6.2 GB/s crossover @autoscts__memory_analyst2; the
row's existence on TAIL @autoscts__memory_analyst1; champion v35 and the rider-free build
@autoscts__memory_gpu3; the byte-neutral control that priced the deletion route
@autoscts__memory_gpu2 (launch 47). Mine: launches 46 and 48 that make the ladder a paid pre-image
rather than a model, the per-table concentration finding that prices the alternative route, the 28 B
`[optstate]` correction, the masking arithmetic, and this build.
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
from torch.utils.checkpoint import checkpoint as _checkpoint
from torch.utils.checkpoint import (CheckpointPolicy as _CheckpointPolicy,
                                    create_selective_checkpoint_contexts
                                    as _create_selective_checkpoint_contexts)
from torch.utils.flop_counter import flop_registry as _flop_registry

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


# TOKENS of the loss tail computed at a time. The float32 working pair of the softcap
# tail -- `th` and the softmax -- is this many tokens wide in BOTH windows, so unlike a
# ROW block it is not pinned to the sequence length: a row block cannot go below one
# sequence (2,048 tokens = 67,108,864 B of float32 at this vocab), and the measured
# floor of both windows' tail instants is z + dz + that pair.
LOSS_TOKEN_CHUNK = 512
SOFTCAP = 15



# Selective activation checkpointing for the MLP, with the save/recompute decision keyed
# on torch's own flop_registry: anything the frozen instrument counts is MUST_SAVE, so no
# counted arithmetic can be re-run and flops_per_token_measured is unchanged by
# construction rather than by inspection. Everything else -- here relu and the square --
# is recomputed.
#
# The region is the MLP body ONLY. It deliberately does NOT span the Block: FA3 is a
# third-party kernel that never reaches aten dispatch, so no policy can pin it, and a
# region containing it would re-run it in the recompute pass -- which prepare.py's
# attention probe tallies per call. That would double the counted attention term.
#
# Unlike an autograd.Function, a policy SURVIVES AOTAutograd's min-cut re-partition: the
# partitioner reads the policy, whereas it re-decides a graph shape freely. That is why
# this reaches the COMPILED training window where _MlpTail measured 0.00.
_COUNTED_OPS = frozenset(_flop_registry.keys())


def _sac_policy(ctx, op, *args, **kwargs):
    counted = getattr(op, "overloadpacket", op) in _COUNTED_OPS
    return (_CheckpointPolicy.MUST_SAVE if counted
            else _CheckpointPolicy.PREFER_RECOMPUTE)


def _sac_contexts():
    return _create_selective_checkpoint_contexts(_sac_policy)


def _is_expansion_weight(args):
    """True when one operand is a 2-D weight that widens its input (in_f < out_f).

    Reads only the PARAMETER's two dimensions. It never looks at a batch, sequence or
    microbatch dimension, so it cannot behave differently for the FLOPs probe, for
    validation, or for training -- the property it tests is a property of the model.
    `c_fc` is (n_embd, 4*n_embd) and matches; `c_proj` is (4*n_embd, n_embd) and does not.
    """
    for a in args:
        if isinstance(a, torch.Tensor) and a.dim() == 2 and a.size(0) < a.size(1):
            return True
    return False


def _sac_policy_mlp(ctx, op, *args, **kwargs):
    """MLP-region policy: as _sac_policy, but the up-projection matmul is RECOMPUTED.

    `c_fc`'s output is (tokens, 4*n_embd) and is the largest retained tensor per layer.
    Its input is the region input -- the normed residual stream -- which is retained
    regardless, so re-running it adds no new retention. The cost is the up-projection's
    forward arithmetic once more, 2 * n_embd * 4 * n_embd per token per layer, and it is
    paid out of the `flops_per_token_measured` headroom that removing a Block created.

    Everything else counted stays MUST_SAVE, so `c_proj` and every attention matmul are
    still never re-executed.
    """
    packet = getattr(op, "overloadpacket", op)
    if packet in _COUNTED_OPS:
        return (_CheckpointPolicy.PREFER_RECOMPUTE if _is_expansion_weight(args)
                else _CheckpointPolicy.MUST_SAVE)
    return _CheckpointPolicy.PREFER_RECOMPUTE


def _sac_contexts_mlp():
    return _create_selective_checkpoint_contexts(_sac_policy_mlp)


def _sac_policy_recompute_all(ctx, op, *args, **kwargs):
    """Region policy: recompute EVERY op interior to the region, counted or not.

    This is the region-BOUNDARY key. `c_q`, `c_k` and `c_v` are all (n_embd, n_embd)
    -- build_model_config sets n_kv_head = n_head -- so no shape predicate can select a
    subset of them, and a parameter attribute cannot either, because F.linear dispatches
    mm(x, w.t()) and t() drops Python attributes. What DOES select c_q + c_k is which
    region they are in, so CausalSelfAttention.forward puts them in their own.

    The policy inspects no tensor at all. It therefore cannot behave differently for the
    FLOPs probe, for validation or for training: it is a property of the model, not of a
    batch or sequence dimension.
    """
    return _CheckpointPolicy.PREFER_RECOMPUTE


def _sac_contexts_recompute_all():
    return _create_selective_checkpoint_contexts(_sac_policy_recompute_all)


# ---------------------------------------------------------------------------
# CARRIED RIDER, read-only. Mechanism @autoscts__memory_gpu6 (launch 40, `[MEASURED]`
# 355f15ba); implementation carried verbatim from @autoscts__memory_gpu1's launch 41
# (`repo_nout/train_block_norm_out_final.py`). Not proposed by this launch and not part of
# its pre-registration.
#
# Why it rides here: this launch removes two full-width temporaries AT W2's arm instant, and
# the eager arm has already migrated on three consecutive programs -- block 0 (L40) ->
# block 4 (L41) -> chunk c1's block 1 (L43, `a_row_split_does_not_buy_the_block_transient_down`).
# Without these reads nobody can say where an 8-width cut at the arm leaves the arm, and the
# `entry + C` law has no fourth program to be tested on.
#
# They execute ONLY in eager. `torch.compiler.is_compiling()` is True while dynamo traces, so
# the branch is constant-folded and the generated wrappers are byte identical to the
# champion's without this block. Training and evaluation both run through the compiled
# wrapper, so the only path that reaches these calls is prepare.py's measure_flops_dispatch,
# which runs the UNCOMPILED module once.
#
# Every call is a pure READ of the allocator's counters, exactly like report_memory: it
# allocates nothing and it never calls reset_peak_memory_stats. The count is capped so no
# unforeseen eager path can flood stdout, and nothing here prints a METRICS_JSON line.
_W2_READS = []


def _w2(tag):
    if len(_W2_READS) < 40 and not torch.compiler.is_compiling():
        _W2_READS.append(tag)
        print(f"[w2] {tag:16s} allocated={torch.cuda.memory_allocated():>14,} "
              f"peak={torch.cuda.max_memory_allocated():>14,}", flush=True)


class _RMSNormKeepInput(torch.autograd.Function):
    """F.rms_norm, saving its INPUT instead of a float32 upcast of that input.

    aten's rms_norm saves `x.float()` (verified: 34 calls at model width in this model,
    2,048 B/token each). The compiled training graph never pays for them -- inductor
    recomputes every one -- but the uncompiled module the FLOPs probe runs pays all of
    them, and for most call sites the input is the residual stream, which is already
    retained, so saving it instead costs nothing.

    Recomputed with straight-line ops only: no nested autograd.grad, no
    torch.utils.checkpoint, and no matmul or attention is re-run. rms_norm is not one of
    the 20 entries in torch.utils.flop_registry, so the counted total is unchanged.

    Forward is bit-identical to F.rms_norm: the float32 upcast is lossless, and aten uses
    torch.finfo(torch.float32).eps whatever the input dtype is.
    """

    @staticmethod
    def _acc(x):
        # aten accumulates rms_norm in float32 for any lower-precision input, and in the
        # input dtype for float64. Matching that is what makes forward bit-identical.
        return torch.promote_types(x.dtype, torch.float32)

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        xf = x.to(_RMSNormKeepInput._acc(x))
        rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + torch.finfo(xf.dtype).eps)
        return (xf * rstd).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        acc = _RMSNormKeepInput._acc(x)
        xf = x.to(acc)
        gf = grad_out.to(acc)
        d = xf.size(-1)
        rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + torch.finfo(acc).eps)
        dot = (gf * xf).sum(-1, keepdim=True)
        dx = rstd * gf - (rstd.pow(3) / d) * dot * xf
        return dx.to(x.dtype)


# CARRIED MECHANISM, credit @autoscts__memory_gpu1 -- block_norm_out_final, launch 41,
# champion.md v31, results/block_norm_out_final.md. Class body and wrapper taken VERBATIM from
# their candidate (repo_nout/train_block_norm_out_final.py), not retyped.
#
# WHY IT IS HERE. champion v33's carry_required demands it, and MY OWN launch 44 declined v32's
# identical requirement and paid exactly 4,194,304 B for that: W3 was the binding window, so the
# headline would have been 622,889,984 instead of 627,084,288. It is worth full price on v33 --
# W3 s1_after_accum 627,084,288 is the READ arm and W2 after_probe 569,709,056 is 57,375,232 B
# below it -- and it had been out of the lineage for three champions.
#
# CHECKED HERE BEFORE CARRYING, in workspace/cpu44/normout_rig.py on torch 2.9.1, both classes
# EXECed out of this run's own sources: forward is BITWISE against _RMSNormKeepInput AND against
# F.rms_norm; backward dx is NOT bitwise, rel-L2 5.399935e-04, max abs 4.883e-04. That
# approximation is the whole risk of this launch and it cost gpu1 +0.0006545 token-corrected
# val_bpb against the 0.0048404 of raw slack v33 has.
class _RMSNormKeepOutput(torch.autograd.Function):
    """F.rms_norm, saving its OUTPUT and rstd instead of its input.

    At a call site whose output is already retained by its consumer, `_RMSNormKeepInput`
    saves a SECOND model-width tensor that the retained output already determines.  This
    form saves the output -- the same storage the consumer holds -- plus rstd, one float32
    per token, against 1,024 B/token of bf16 input at n_embd = 512.

    Forward is bit-identical to `_RMSNormKeepInput.forward`: the same expression, in the
    same order, with the same float32 accumulation, and `torch.finfo(acc).eps` is the same
    value as `torch.finfo(xf.dtype).eps` because `xf = x.to(acc)`.  This is a change of
    what is SAVED, not of what is computed, so the forward loss is unchanged bit for bit.

    Backward, with y = x*rstd and rstd = rsqrt(mean(x^2)+eps):

        dL/dx_j = g_j*rstd - (rstd^3/d) * sum_i(g_i*x_i) * x_j
                = rstd*g_j - (rstd/d) * sum_i(g_i*y_i) * y_j

    because x = y/rstd makes rstd^3*x_j = rstd^2*y_j and sum(g*x) = sum(g*y)/rstd.  Same
    term order and same number of multiplies as `_RMSNormKeepInput.backward`.

    It is an APPROXIMATION, and the only one in this file, because the retained `out` is
    the bf16-ROUNDED output: sum(gf*out) is not exactly rstd*sum(gf*x).  Measured on CPU by
    @autoscts__memory_gpu3 against the champion's own `_RMSNormKeepInput`: 1 of 54
    parameter gradients bitwise identical, median relative deviation 6.438e-03 and worst
    1.220e-02 over the 48 matrix/embedding tensors, i.e. 1.6 bf16 ULPs.  That is the whole
    risk of this launch and falsifier (4) on the queue row is the read that prices it.

    Zero counted FLOPs: rms_norm is not one of the entries in torch.utils.flop_registry,
    and no matmul or attention is re-run.  No recompute: the backward reads rstd (4 B/token)
    where it read x (1,024 B/token).
    """

    @staticmethod
    def forward(ctx, x):
        acc = _RMSNormKeepInput._acc(x)
        xf = x.to(acc)
        rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + torch.finfo(acc).eps)
        out = (xf * rstd).to(x.dtype)
        ctx.save_for_backward(out, rstd)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        out, rstd = ctx.saved_tensors
        acc = _RMSNormKeepInput._acc(out)
        of = out.to(acc)
        gf = grad_out.to(acc)
        d = of.size(-1)
        dot = (gf * of).sum(-1, keepdim=True)
        dx = rstd * gf - (rstd / d) * dot * of
        return dx.to(out.dtype)


class _SoftcapCrossEntropy(torch.autograd.Function):
    """Per-token cross-entropy of softcap*tanh(z/softcap), from the bfloat16 logits.

    Saves only `z`, and walks the tail LOSS_TOKEN_CHUNK TOKENS at a time in BOTH
    directions, so no whole-batch vocab-wide float32 tensor is ever allocated and the
    float32 working pair is sized by the block rather than by the sequence length.
    Losses are PER TOKEN and concatenated in token order, so no chunk mean is ever
    averaged against another -- the one way a chunked loss could silently reweight
    itself.

    Adds zero counted FLOPs: _to_copy, div, tanh, mul, _log_softmax, _softmax and
    nll_loss are all absent from torch.utils.flop_registry, and neither the head matmul
    nor flash_attn_func is recomputed.

    Backward, per chunk, with th = tanh(z/softcap):
        d(nll)/d(capped) = softmax(capped) - onehot(target)      (zero where ignored)
        d(capped)/dz     = 1 - th*th
    """

    @staticmethod
    def forward(ctx, z, targets, softcap, chunk_tokens):
        ctx.save_for_backward(z, targets)
        ctx.softcap = softcap
        ctx.chunk_tokens = chunk_tokens
        zf = z.view(-1, z.size(-1))
        tf = targets.reshape(-1)
        out = []
        for lo in range(0, zf.size(0), chunk_tokens):
            capped = softcap * torch.tanh(zf[lo:lo + chunk_tokens].float() / softcap)
            out.append(F.cross_entropy(capped, tf[lo:lo + chunk_tokens],
                                       ignore_index=-1, reduction='none'))
            del capped
        return torch.cat(out)

    @staticmethod
    def backward(ctx, grad_per_token):
        z, targets = ctx.saved_tensors
        softcap, chunk_tokens = ctx.softcap, ctx.chunk_tokens
        zf = z.view(-1, z.size(-1))
        tf = targets.reshape(-1)
        gf = grad_per_token.reshape(-1)
        ar = torch.arange(zf.size(-1), device=z.device)
        out = []
        for lo in range(0, zf.size(0), chunk_tokens):
            hi = lo + chunk_tokens
            tc = tf[lo:hi]
            th = torch.tanh(zf[lo:hi].float() / softcap)
            p = torch.softmax(softcap * th, dim=-1)
            # (p - onehot(target)) * grad * (1 - th*th), as ONE out-of-place chain.
            # No scatter_add_ and no in-place multiply, so nothing forces the float32
            # softmax result to be materialised: inductor fuses the chain into the
            # bfloat16 store and keeps it in registers, which is why the reference tail
            # allocates zero float32 vocab-wide bytes. Same values and the same multiply
            # ORDER as the mutating form.
            #
            # `ar == tc` is False on every column when the target is -1, so nothing is
            # subtracted there; the mutating form subtracts 1 at column 0 (clamp(min=0))
            # and then multiplies the row by (tc != -1) == 0. Both give an exactly zero
            # row, which is the whole point of ignore_index.
            p = torch.where(ar == tc.unsqueeze(-1), p - 1.0, p)
            p = p * (gf[lo:hi] * (tc != -1)).unsqueeze(-1)
            out.append((p * (1.0 - th * th)).to(z.dtype))
            del th, p
        return torch.cat(out).view_as(z), None, None, None


def norm(x):
    return _RMSNormKeepInput.apply(x)


def norm_out(x):
    """norm() for the ONE call site whose output is already retained.

    The pre-lm_head norm only.  The after-wte site was measured at +0 B -- inductor
    already recomputes the wte gather from `idx`, so that norm's input is not in the
    compiled saved set to begin with -- so it is deliberately NOT changed here.
    """
    return _RMSNormKeepOutput.apply(x)


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
        # HUNK 2 of intercept_l6_ls_refund_plus_region, and the whole of its selection rule.
        # ONE layer -- the terminal one -- runs its q/k half, its v half, flash_attn_func and
        # c_proj inside a SINGLE checkpoint region, so norm(q), norm(k), v and fa3's `out`
        # stop being region outputs and stop being saved. It is keyed on the layer index,
        # which is a property of the model: it reads no tensor, no batch and no sequence
        # dimension, so it cannot behave differently for the FLOPs probe, for validation or
        # for training. It is the terminal layer because HUNK 1 makes that layer a
        # short-window layer like any other, which is what brings its fa3 recompute inside
        # the counted-FLOPs ceiling (6,144 * 1,024 instead of 6,144 * 2,048).
        self.merge_attn_region = (layer_idx == config.n_layer - 1)

    def _qk_body(self, x, cos, sin):
        """The q/k half of the reference attention prologue, straight-line.

        Takes the RAW residual stream and norms it INSIDE the region, so the region's
        input is `x` -- already retained -- and the normed tensor is interior and
        recomputed.  Handing in a normed tensor pins that tensor for the whole backward
        whatever the policy says, because the recompute needs the region's inputs.

        Its region outputs are norm(q) and norm(k) -- two of flash_attn_func's three
        inputs -- which are materialised and saved by their consumer either way. Interior
        to it, and therefore removable by a policy, are `c_q`'s and `c_k`'s mm outputs
        (head_dim * n_head bf16 = 1,024 B per token per projection per layer) and the
        rotary intermediates. The mm outputs were previously held IN ADDITION to
        norm(q)/norm(k), because _sac_policy MUST_SAVEs every op in flop_registry: the
        double hold that launch 23 measured on `c_fc` and freed at ratio 1.0.

        Recomputing them costs 2 * n_embd * n_embd per token per projection per layer,
        7,340,032 per token over 7 layers for the two of them, paid out of the counted
        FLOPs ceiling.
        """
        x = norm(x)
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        return q, k

    def _v_body(self, x, idx, ve_w):
        """The v half, in the reference order, and it keeps the mm-MUST_SAVE policy.

        `c_v` is excluded from the recompute region on arithmetic, not on caution: all
        three projections cost 11,010,048 counted FLOPs per token and the ceiling has
        10,485,760. It is also measured to be worth nothing -- v = v + gate * ve has an
        identity gradient with respect to c_v's output, so nothing in backward reads it
        and MUST_SAVE does not retain what nothing consumes -- and in the non-VE layers
        it is the region's own output.

        Keeping it a separate region under _sac_contexts, rather than folding it into the
        recompute region, is what stops the ve_gate sigmoid/blend intermediates from
        being re-executed and re-counted.
        """
        x = norm(x)
        B, T, C = x.size()
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve_w is not None:
            # THE CARRY (@autoscts__memory_gpu5, launch 35): the gather happens HERE,
            # interior to this checkpoint region, instead of in GPT._raw_logits.  Gathered
            # outside and handed in, the (B, T, kv_dim) rows are a region INPUT and are
            # pinned for the whole backward whatever _sac_policy says about the interior.
            # Interior, `aten.embedding` is not in torch.utils.flop_registry, so the policy
            # PREFER_RECOMPUTEs it and the gathered tensor dies at the end of this forward.
            # nn.Embedding.forward IS F.embedding(input, self.weight, self.padding_idx,
            # self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse), and all
            # five trailing arguments are at their defaults on every table in this model, so
            # this is the same call with the same values in the same dtype.
            ve = F.embedding(idx, ve_w).view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        return v

    def _merged_body(self, x, idx, ve_w, cos, sin, window_size):
        """The WHOLE attention block as one region body: q/k half, v half, fa3, c_proj.

        A checkpoint REGION is not a policy. `_sac_policy` can only decide which
        *dispatched* ops to keep, which is why the champion's comment below is right that
        no policy can pin `flash_attn_func` -- it never reaches aten dispatch. But
        `torch.utils.checkpoint` re-runs whatever function it is handed, dispatched or not,
        so a region boundary drawn ACROSS fa3 does what no policy can: it makes `norm(q)`,
        `norm(k)`, `v` and fa3's `out` interior tensors, and this region's only output is
        c_proj's output, which the residual add retains regardless.

        The four entries are 4,096 tokens * 512 * 2 B = 4,194,304 B each at
        DEVICE_BATCH_SIZE = 2, so -16,777,216 B, and they are live at the arm instant:
        s1_after_accum's high-water is the loss tail at the START of the backward, with the
        whole forward saved set live and nothing yet freed, and this layer's recompute does
        not run until its own block backward, long after that instant.

        What it costs, per token, all of it counted by prepare.py because that instrument
        runs the UNCOMPILED module and a checkpoint recompute is not advisory:
            fa3, one more call at this layer's span   6,144 * 1,024 = +6,291,456
            c_v      (2 * n_embd * kv_dim)                          =   +524,288
            c_proj   (2 * n_embd * n_embd)                          =   +524,288
            ve_gate  (2 * 32 * n_kv_head, this IS a VE layer)       =       +256
        `c_q` and `c_k` are already recomputed today, so they add nothing. HUNK 1 refunds
        6,291,456 of it, leaving +1,048,832/token.
        """
        q, k = self._qk_body(x, cos, sin)
        v = self._v_body(x, idx, ve_w)
        B, T = x.size(0), x.size(1)
        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y

    def forward(self, x, idx, ve_w, cos_sin, window_size):
        # TWO regions where there was one, because the region boundary is the only legal
        # key for c_q + c_k: the q/k half recomputes everything interior to it, the v half
        # keeps the flop_registry-keyed MUST_SAVE policy so c_v and ve_gate are never
        # re-executed. fa3 and c_proj stay OUTSIDE both: fa3 never reaches aten dispatch,
        # so no policy could pin it and re-running it would double prepare.py's per-call
        # attention tally.
        #
        # ON THE TERMINAL LAYER ONLY that second clause is now affordable, because HUNK 1
        # made that layer a 1024-span layer: ONE region spans all four, and the doubled
        # attention tally it pays for is exactly the tally HUNK 1 refunded. Layers 0..n-2
        # take the champion's path unchanged.
        B, T, C = x.size()
        cos, sin = cos_sin
        if self.merge_attn_region:
            return _checkpoint(self._merged_body, x, idx, ve_w, cos, sin, window_size,
                               use_reentrant=False,
                               context_fn=_sac_contexts_recompute_all)
        q, k = _checkpoint(self._qk_body, x, cos, sin,
                           use_reentrant=False,
                           context_fn=_sac_contexts_recompute_all)
        v = _checkpoint(self._v_body, x, idx, ve_w,
                        use_reentrant=False, context_fn=_sac_contexts)

        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class _MlpTail(torch.autograd.Function):
    """NO LONGER ON THE CODE PATH -- superseded by the flop_registry-keyed SAC policy above.

    Kept only so the mechanism and its credit stay readable in the champion lineage.
    MLP.forward does not call it: this Function pinned one 4*n_embd tensor in the EAGER
    window but measured 0.00 in the compiled training window, because save_for_backward is
    advisory under torch.compile -- AOTAutograd's min-cut partitioner re-decides the save
    set and kept both tensors (measured: 16 x (2048,2048) bf16 = 65,536 B/token). A POLICY
    survives that re-partition; a graph shape does not. Changing anything below has no
    effect on any measured number.

    Original docstring follows.

    relu(h)**2 followed by c_proj, retaining only relu(h).

    MECHANISM CREDIT: autoscts__memory_gpu1 (team PROBE), measured eligible on launch 3
    `probe_mlp4c_recompute_b8` and again on launch 5 `probe_evalchunk_compose_b8`. This
    file reproduces it verbatim, including its algebra and its FLOPs argument. It is not
    my mechanism and this launch claims no credit for it -- what is new here is the rung
    it is measured at.

    Standard autograd keeps three 4*n_embd-wide activations per layer through this tail:
    c_fc's output h (for relu's backward), relu(h) (for the square's backward) and
    relu(h)**2 (c_proj's matmul input, for the weight gradient). Only relu(h) has to be
    kept:

        y      = relu(h)**2
        dy/dh  = 2 * relu(h) * 1{h > 0} = 2 * relu(h)

    because relu(h) is already zero wherever h <= 0, so the mask needs no separate tensor
    and h itself is never needed again. relu(h)**2 is recomputed in the backward from the
    one tensor that is kept.

    FLOPs: the recompute is a single elementwise square, which the frozen instrument does
    not count (it counts matmul / convolution / attention and explicitly excludes
    elementwise and normalisation work). The backward issues exactly the two matmuls
    autograd would issue for this tail, with identical shapes, so the dispatch tally is
    unchanged -- measured 239,078,400 exactly on both launches that carried this code.

    Numerics: the forward is the same three ops in the same order and the same dtypes, and
    the backward is the same algebra autograd applies. Nothing is approximated.
    """

    @staticmethod
    def forward(ctx, h, weight):
        r = torch.relu(h)
        ctx.save_for_backward(r, weight)
        return F.linear(r * r, weight)

    @staticmethod
    def backward(ctx, grad_out):
        r, weight = ctx.saved_tensors
        y = r * r                                    # elementwise recompute, uncounted
        grad_h = None
        grad_w = None
        if ctx.needs_input_grad[0]:
            grad_h = (grad_out @ weight) * (2 * r)   # same matmul autograd would issue
        if ctx.needs_input_grad[1]:
            grad_w = (grad_out.reshape(-1, grad_out.size(-1)).mT
                      @ y.reshape(-1, y.size(-1)))   # same matmul autograd would issue
        return grad_h, grad_w


class _SquareGradInOne(torch.autograd.Function):
    """`r * r`, whose backward allocates ONE 4*n_embd temporary where autograd allocates THREE.

    MECHANISM CREDIT: @autoscts__memory_gpu4 -- `[MEASURED] c078057e`, TAIL queue row
    `tail_mlp_square_grad_in_one`, candidate `repo_mlp/train.py`. Their rig located this
    window's arm instant and priced this rewrite. This file reproduces their class body and
    their algebra; the credit is theirs and this launch claims none of it.

    W2 -- the FLOPs probe's window and champion v32's binding window -- is an EAGER
    forward+backward (prepare.py:587 unwraps torch.compile to `_orig_mod`), so no partitioner
    and no epilogue fusion touches it. Its arm instant is inside this MLP region's
    recompute-and-backward, with SIX (tokens, 4*n_embd) bf16 tensors live at once, THREE of
    them belonging to this one squaring's backward. autograd's rule for `x ** 2` is
    `grad * (2 * x.pow(2 - 1))`, which materialises `x.pow(1)` -- a full-width COPY of x that
    exists for nothing -- then `2 * that`, then `grad * that`.

    The same value, in one allocation:

        t = r.mul(2)   is exactly autograd's `2 * x.pow(1)`: `x.pow(1)` IS x, and multiplying
                       by 2 in binary floating point increments the exponent, so it is exact.
        t.mul_(g)      is exactly `grad * (2 * r)`: IEEE multiplication is commutative and
                       exact per element, so the operand order cannot change one bit. `t` was
                       created one line above and aliases nothing, so mutating it is safe.

    Forward is `r * r`, which is what `.square()` computes. `save_for_backward` holds `r`,
    which `aten.pow`'s backward holds today, so the RETAINED set is unchanged and this is a
    temporary elimination only.

    MEASURED before the launch ([local verification path], torch 2.9.1,
    the champion's own MLP source EXECed out of this file after a sha256 assert):

      eager MLP region, 4,096 tokens:  peak live 106,954,752 -> 73,400,320, -33,554,432
                                       = -8.00 model widths/token, arm moves aten.mul -> aten.mm
      bitwise:  y, dx, dw_c_fc, dw_c_proj all torch.equal to the champion arm, eager AND
                under torch.compile
      compiled: aot_eager + saved_tensors_hooks, 5,242,880 B on both arms, delta +0 B, and
                not one shape, dtype or count changed -- so W3 is untouched
      FLOPs:    FlopCounterMode 15,032,385,536 on both arms, identical to the digit; `mul`,
                `pow` and `relu` are all absent from torch.utils.flop_registry

    Folding `relu` into the same Function was measured too and reads the SAME 73,400,320, so
    it buys nothing and the minimal scope is shipped unchanged.
    """

    @staticmethod
    def forward(ctx, r):
        ctx.save_for_backward(r)
        return r * r

    @staticmethod
    def backward(ctx, g):
        (r,) = ctx.saved_tensors
        t = r.mul(2)
        t.mul_(g)
        return t


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def _body(self, x):
        """The reference MLP, straight-line. What is saved is the policy's decision.

        Norms its own input, so the region input is the residual stream rather than a
        normed copy of it. `norm` is uncounted, so the policy recomputes it."""
        x = norm(x)
        # `.square()` -> _SquareGradInOne: same forward value, same backward value in
        # autograd's own operand order, TWO fewer (tokens, 4*n_embd) temporaries live at
        # W2's arm instant. Mechanism credit @autoscts__memory_gpu4 (c078057e).
        return self.c_proj(_SquareGradInOne.apply(F.relu(self.c_fc(x))))

    def forward(self, x):
        # Replaces _MlpTail (gpu1's eager mechanism, which measured 0.00 in the compiled
        # window because save_for_backward is advisory under torch.compile) with a policy,
        # which the partitioner honours. Same three ops in the same order and dtypes.
        return _checkpoint(self._body, x, use_reentrant=False,
                           context_fn=_sac_contexts_mlp)


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, idx, ve_w, cos_sin, window_size):
        # Both submodules norm their own input INSIDE their checkpoint regions, so the
        # region input is `x` -- retained regardless -- and no normed copy is pinned.
        # `idx` and the value-embedding TABLE are threaded through instead of the gathered
        # rows; v28 already hands `x` raw, so this is champion.md's composed carry line.
        x = x + self.attn(x, idx, ve_w, cos_sin, window_size)
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
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        # Rotary embeddings
        # The cos/sin tables were built for 10x the context the model can ever be
        # handed: `_raw_logits` asserts `T <= self.cos.size(1)` and every dataloader in
        # prepare.py is built at MAX_SEQ_LEN, so rows 2048..20479 are unreachable.
        # `_precompute_rotary_embeddings` is `outer(arange(seq_len), inv_freq)`, whose
        # first 2048 rows do not depend on seq_len, so the retained slice is BYTE
        # IDENTICAL: torch.equal(cos_2048, cos_20480[:, :2048]) is True, measured.
        # Two resident bf16 buffers, 2,621,440 B each -> 262,144 B each: -4,718,592 B
        # live at every instant, so both the eager probe window and the compiled
        # training window fall by it.
        self.rotary_seq_len = config.sequence_len
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
        # ... and the master weights, the same call for the remaining 29,360,640 params. The
        # forward already ran these matmuls in bf16 -- autocast cast every one of these weights
        # on the way in and the compiled graph retained the casts (bc6892f2 measured 0.0547 GiB
        # of them). A bf16 weight needs no cast, its gradient is bf16, and the accumulation
        # loop's duplicate gradient set halves with it. The 16 per-layer scalars stay fp32:
        # they are 64 B and they gate the residual stream.
        self.transformer.h.to(dtype=torch.bfloat16)
        self.lm_head.to(dtype=torch.bfloat16)

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
        # HUNK 1 of intercept_l6_ls_refund_plus_region. The line that stood here forced the
        # terminal layer to the full context window regardless of WINDOW_PATTERN:
        #     window_sizes[-1] = (long_window, 0)
        # It is CHAMPION SOURCE, not the frozen instrument, and WINDOW_PATTERN is already
        # "SSSS", so deleting it makes the realised pattern all-short at n_layer=7 and the
        # span sum 8,192 -> 7,168. By launch 26's paid span law (6,144 counted FLOPs/token
        # per unit of span, error 0 B) that refunds 6,291,456 counted FLOPs/token, which is
        # what pays for HUNK 2's recompute on this same layer. Launch 26 measured the same
        # 2048 -> 1024 move on an interior layer at zero bytes, zero parameters and
        # val_bpb +0.0004..+0.0008.
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
            # THE MECHANISM: this group, and only this group, drops its first moment. Its four
            # (8192, 512) tables are 33,554,432 B of the 50,331,648 B `exp_avg` line, and
            # `no_first_moment` below keys on `group['betas'][0] == 0.0`, so restricting beta1 = 0 to
            # this group is the whole diff. beta2 is the champion's, unchanged. `x0_params` already
            # hard-codes its own betas and never read ADAM_BETAS; wte, lm_head and resid_params keep
            # the champion's 0.8 here, which launch 46 did not.
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=(VALUE_EMBED_BETA1, adam_betas[1]), eps=1e-10, weight_decay=0.0),
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

    def _raw_logits(self, idx):
        """Token ids -> softcapped logits, for any block of batch rows.

        Nothing in this model couples batch rows: the token and value-embedding
        gathers, both rms norms, rotary, attention (causal WITHIN a row), the MLP,
        the head and the softcap are all per-token, and cos/sin are shared read-only
        buffers. So this body is exact on a subset of rows, and the row axis is a
        free axis to walk one block at a time.
        """
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        _w2("fwd_embed")
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            # The TABLE, not the gathered rows: the gather moved into _v_body so that it is
            # interior to that region rather than a saved region input.  A parameter handed
            # to a region costs no retention -- it is resident either way, and `idx` is
            # 4 B/token of int against 4,096 B/token of gathered bf16.
            ve_w = self.value_embeds[str(i)].weight if str(i) in self.value_embeds else None
            x = block(x, idx, ve_w, cos_sin, self.window_sizes[i])
            if not torch.compiler.is_compiling():
                x.register_hook(lambda g, i=i: _w2(f"bwd_blk{i}_dout"))
        _w2("fwd_blocks")
        x = norm_out(x)

        return self.lm_head(x)

    def _forward_to_logits(self, idx):
        """Softcapped float32 logits, unchanged: the grad-free path's whole working set.

        Only the GRAD-ENABLED path stops materialising these -- see forward().
        """
        logits = self._raw_logits(idx).float()
        return SOFTCAP * torch.tanh(logits / SOFTCAP)

    def forward(self, idx, targets=None, reduction='mean'):
        if targets is None:
            return self._forward_to_logits(idx)

        if torch.is_grad_enabled():
            # Training, and the FLOPs probe: one full-batch forward, the reference
            # computation unchanged. The branch is on GRAD MODE, never on a shape or a
            # batch size, so no instrument can be handed a cheaper path than training
            # runs: whatever batch the probe chooses, it takes this branch.
            per_token = _SoftcapCrossEntropy.apply(
                self._raw_logits(idx), targets, SOFTCAP, LOSS_TOKEN_CHUNK)
            _w2("fwd_loss")
            if reduction == 'none':
                return per_token
            total = per_token.sum()
            if reduction == 'sum':
                return total
            return total / (targets != -1).sum().clamp(min=1)

        # Grad-free forward. With no graph to retain, the vocab-wide logits are this
        # window's entire working set and nothing has to outlive the rows that produced
        # it, so walk EVAL_ROW_CHUNK rows at a time and let each block's logits die
        # before the next block allocates. Per-chunk losses are PER TOKEN and are
        # concatenated in row order -- no chunk mean is ever averaged against another,
        # which is the one way this transform could silently reweight the loss.
        per_token = []
        for lo in range(0, idx.size(0), EVAL_ROW_CHUNK):
            chunk_logits = self._forward_to_logits(idx[lo:lo + EVAL_ROW_CHUNK])
            chunk_targets = targets[lo:lo + EVAL_ROW_CHUNK]
            per_token.append(F.cross_entropy(
                chunk_logits.view(-1, chunk_logits.size(-1)),
                chunk_targets.reshape(-1), ignore_index=-1, reduction='none'))
            del chunk_logits
        losses = torch.cat(per_token)
        if reduction == 'none':
            return losses
        total = losses.sum()
        if reduction == 'sum':
            return total
        return total / (targets != -1).sum().clamp(min=1)

# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW, single GPU only)
# ---------------------------------------------------------------------------

# Dtype of every OPTIMIZER STATE tensor. Master weights and gradients are unaffected:
# `stacked_params` below stays the parameter's own dtype, so Muon's update is still
# accumulated into fp32 master weights. bfloat16 has float32's exponent range, so no
# moment can underflow that did not underflow before; it has 8 mantissa bits, so this
# trades EMA precision for 67,109,952 B of permanently resident memory. wte and the value
# embeddings are bf16 parameters already, so their Adam moments were already bf16 and this
# constant leaves them byte-identical.
OPT_STATE_DTYPE = torch.bfloat16

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
    # A bf16 state tensor cannot take an fp32 in-place lerp: torch refuses to cast the
    # promoted result back down. Both casts are no-ops when the parameter is already bf16.
    exp_avg.lerp_(grad.to(exp_avg.dtype), 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square().to(exp_avg_sq.dtype), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused_factored(p, grad, exp_avg, v_r, v_c, step_t, lr_t, beta1_t, beta2_t,
                              eps_t, wd_t):
    """AdamW whose second moment is Adafactor's rank-1 factorisation of `exp_avg_sq`.

    Shazeer & Stern, "Adafactor: Adaptive Learning Rates with Sublinear Memory Cost"
    (arXiv:1804.04235) section 3.  Everything else -- the decoupled weight decay, the
    first moment, both bias corrections, `eps` added AFTER the sqrt, the step size --
    is `adamw_step_fused` above, unchanged, so the only difference in the update is
    which estimate of E[g^2] the denominator reads.

    `v_r` is (rows, 1) and `v_c` is (1, cols), each an EMA at the same `beta2` of the
    row / column MEAN of g^2, and the reconstruction is

        v_hat[i, j] = v_r[i] * v_c[j] / mean(v_r)

    which is identically Adafactor's `R C / sum(R)`: with v_r = R/ncols and
    v_c = C/nrows, mean(v_r) = S/(nrows*ncols), so the ratio is R_i C_j / S exactly.
    It is an approximation of the dense tensor, not an algebraic rewrite of it, and it
    is exact whenever E[g^2] is rank one.

    BOTH factors are float32, not OPT_STATE_DTYPE.  They are 208,896 B in total -- the
    dtype buys nothing here -- and `mean(v_r)` is a reduction over 8,192 numbers whose
    reciprocal multiplies every coordinate of the update, which is exactly the place
    the file's own `second_momentum_buffer` comment reserves float32 for.

    `g2` is the SAME bfloat16 square the dense path stores (`grad.square().to(bf16)`), so
    the precision of the quantity being averaged is unchanged and the transient is the
    same 8,388,608 B tensor this step already materialises; only the reductions are taken
    in float32, which is strictly more accurate than the dense path's per-coordinate
    bfloat16 EMA.  `denom` is cast down to the moment dtype for the same reason the dense
    path's denom is already bfloat16: `exp_avg` is, and a float32 divisor would promote
    the update and refuse the in-place `add_`.
    """
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad.to(exp_avg.dtype), 1 - beta1_t)
    g2 = grad.square()
    v_r.lerp_(g2.mean(dim=1, keepdim=True, dtype=torch.float32), 1 - beta2_t)
    v_c.lerp_(g2.mean(dim=0, keepdim=True, dtype=torch.float32), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    # `.clamp_min` IS LOAD-BEARING, and it is not defensive programming. `init_weights`
    # zero-inits `attn.c_proj`, so at step 0 NO gradient reaches v and the four
    # value-embedding tables have IDENTICALLY ZERO gradient -- measured on this file's own
    # model with the real loader (cpu_fac/prescreen_init.log, and independently in
    # @autoscts__memory_gpu2's grad_scale.log row "value_embeds x4: zero at init").  With
    # every row zero, `v_r.mean()` is exactly 0, `v_r / 0` is 0/0 = NaN, and the whole
    # (8192, 512) table becomes NaN on the FIRST optimizer step: verified at 4,194,304 NaNs
    # per table against the dense path's clean no-op. train.py's own `isnan(train_loss_f)`
    # fast-fail would then print FAIL and exit(1) -- a launch spent and its program identity
    # reserved forever. Clamped, the zero case reproduces the dense path exactly: v_hat = 0,
    # denom = eps, update = 0 / eps = 0.
    v_hat = (v_r / v_r.mean().clamp_min(1e-30)) * v_c
    denom = ((v_hat / bias2).sqrt() + eps_t).to(exp_avg.dtype)
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused_factored_nomom(p, grad, v_r, v_c, step_t, lr_t, beta2_t, eps_t, wd_t):
    """`adamw_step_fused_factored` at beta1 = 0: the first moment IS the gradient.

    PROPOSED BY @autoscts__memory_analyst1 (tail_fuse_hook_adamw_beta1_zero, [PROPOSAL]
    5d719879).  Body taken from `adamw_step_fused_factored` directly above with `exp_avg`
    replaced by `grad` and the two terms that go dead at beta1 = 0 removed.  Nothing else
    is touched.

    WHY THIS IS A DELETION AND NOT AN APPROXIMATION OF THE STEP ABOVE.  At beta1 = 0:

        exp_avg.lerp_(grad.to(exp_avg.dtype), 1 - beta1_t)   ->   exp_avg = grad
        bias1 = 1 - beta1_t ** step_t = 1 - 0**step          ->   1, for every step >= 1
        step_size = lr_t / bias1                             ->   lr_t

    so the buffer holds a bf16 copy of `grad` at every step and the bias correction is
    identically 1.  Reading `grad` where the champion reads `exp_avg` is therefore the SAME
    UPDATE, and the (8192, 512) buffer is redundant rather than dropped.  This is Adafactor's
    own beta1 = 0 route (Shazeer & Stern 2018, arXiv:1804.04235), the same paper whose
    rank-1 second moment the champion already ships.

    THE SECOND MOMENT IS UNTOUCHED, and that is the whole reason this form is preferred to
    folding a momentum into `p.grad`: `v_r` and `v_c` are still EMAs of the RAW gradient's
    row/column means, exactly as above.  A fold would feed them a smoothed gradient scaled by
    1/(1 - beta1), which multiplies the reconstructed denominator by an amount that depends on
    the gradient's autocorrelation -- an unpriced effective-LR change.  There is none here.

    `.to(grad.dtype)` on `denom` is `.to(exp_avg.dtype)` above: OPT_STATE_DTYPE is bfloat16
    and the parameters are bfloat16, so it is the same cast, kept for the same reason (a
    float32 divisor would promote the update and refuse the in-place `add_`).

    `.clamp_min(1e-30)` on `v_r.mean()` is carried VERBATIM and is load-bearing for exactly
    the reason documented above: `init_weights` zero-inits `attn.c_proj`, so the four
    value-embedding tables have identically zero gradient at step 0, and without the clamp
    `v_r / v_r.mean()` is 0/0 = NaN on the first optimizer step.

    Zero counted FLOPs: `mul`, `square`, `mean`, `lerp`, `pow`, `sqrt`, `add` and `div` are
    all absent from `torch.utils.flop_registry`, and this function contains no matmul and no
    attention.  It runs after `measure_flops_dispatch` has already returned.
    """
    p.mul_(1 - lr_t * wd_t)
    g2 = grad.square()
    v_r.lerp_(g2.mean(dim=1, keepdim=True, dtype=torch.float32), 1 - beta2_t)
    v_c.lerp_(g2.mean(dim=0, keepdim=True, dtype=torch.float32), 1 - beta2_t)
    bias2 = 1 - beta2_t ** step_t
    v_hat = (v_r / v_r.mean().clamp_min(1e-30)) * v_c
    denom = ((v_hat / bias2).sqrt() + eps_t).to(grad.dtype)
    p.add_(grad / denom, alpha=-lr_t)

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    # Heavy-ball momentum, held in p.grad instead of a second resident buffer. The caller
    # scales p.grad by `carry` at the tail of the previous step, so backward's own
    # accumulation has already performed `mu*history + g` by the time we get here and
    # `stacked_grads` IS the buffer, up to the constant 1/(1-mu). `:963` normalises the
    # direction, so this function is homogeneous of degree 0 and that constant is free.
    # `momentum_t` is retained in the signature: the schedule that drives it is unchanged
    # and it is now read one step earlier, at the carry, not here.
    g = stacked_grads
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
            # THE MECHANISM (@autoscts__memory_analyst1, tail_fuse_hook_adamw_beta1_zero).
            # At beta1 = 0 the first moment is a bf16 copy of `grad` (see
            # `adamw_step_fused_factored_nomom`), so for the 2-D tables the buffer is not
            # allocated at all and the momentum-free step reads `grad` directly. The branch is
            # on `group['betas'][0]` and `p.ndim`, both properties of the configuration and the
            # parameter -- never on a shape or a batch size an instrument could choose, so the
            # FLOPs probe and training take the same path.
            #
            # 1-D parameters keep the dense buffer and the dense step BYTE FOR BYTE (28 B of
            # exp_avg in total), so no scalar group's numerics change. That 28 B is why this
            # candidate pre-registers -50,331,648 and not the row's -50,331,676: the row's
            # number also deletes the 1-D buffers, which this diff deliberately keeps.
            no_first_moment = (group['betas'][0] == 0.0 and p.ndim == 2)
            if not state:
                state['step'] = 0
                if not no_first_moment:
                    # HUNK (b). A 2-D AdamW parameter that keeps its first moment allocates
                    # that buffer in PINNED HOST memory, not on the device. `pin_memory=True`
                    # is mandatory: it is what makes the per-step staging a DMA rather than a
                    # pageable copy through a bounce buffer. The predicate is a property of
                    # the configuration and of the parameter's rank -- never a shape or a
                    # batch size an instrument could choose -- so the FLOPs probe and
                    # training take the same path.
                    if OPT_STATE_HOST_RESIDENT and p.ndim == 2:
                        state['exp_avg'] = torch.zeros(p.shape, dtype=OPT_STATE_DTYPE,
                                                       device="cpu", pin_memory=True)
                    else:
                        state['exp_avg'] = torch.zeros_like(p, dtype=OPT_STATE_DTYPE)
                # THE MOVE. A 2-D AdamW parameter keeps its second moment FACTORED, as two
                # float32 vectors instead of one dense tensor; a 1-D parameter keeps the
                # dense tensor and the dense step, byte-identical to the champion. The
                # branch is on `p.ndim`, which is a property of the parameter, not on any
                # shape an instrument chooses.
                if p.ndim == 2:
                    state['v_r'] = torch.zeros(p.size(0), 1, dtype=torch.float32,
                                               device=p.device)
                    state['v_c'] = torch.zeros(1, p.size(1), dtype=torch.float32,
                                               device=p.device)
                else:
                    state['exp_avg_sq'] = torch.zeros_like(p, dtype=OPT_STATE_DTYPE)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            if no_first_moment:
                adamw_step_fused_factored_nomom(p, grad, state['v_r'], state['v_c'],
                                self._adamw_step_t, self._adamw_lr_t,
                                self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)
            elif p.ndim == 2:
                # HUNK (c). Stage the host-resident first moment to the device, run the
                # champion's step on it UNMODIFIED, write it back. The H2D copy is enqueued on
                # the current stream, so the compiled step that reads it is ordered after it
                # without a host sync. The write-back is synchronous on purpose: a fully async
                # write-back is also correct (nothing on the host touches the buffer between
                # the two copies, so stream ordering suffices) but it is a bandwidth
                # optimisation, and this launch exists to MEASURE the bandwidth, not to assume
                # it. `_m` is freed when the loop advances, and `report_memory` runs before
                # `optimizer.step()`, so no reported instant observes it.
                _m_host = state['exp_avg']
                if _m_host.is_cuda:
                    adamw_step_fused_factored(p, grad, _m_host, state['v_r'],
                                    state['v_c'],
                                    self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                                    self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)
                else:
                    _m = _m_host.to(p.device, non_blocking=True)
                    adamw_step_fused_factored(p, grad, _m, state['v_r'],
                                    state['v_c'],
                                    self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                                    self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)
                    _m_host.copy_(_m, non_blocking=False)
                    del _m
            else:
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
        # No `momentum_buffer` is allocated: p.grad carries the history (see the docstring).
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            # PINNED to float32, not `dtype` and not OPT_STATE_DTYPE. This buffer is 197,120 B
            # and it is NorMuon's variance normaliser: muon_step_fused takes
            # `second_momentum_buffer.clamp_min(1e-10).rsqrt()`, which is the one place in this
            # optimizer where 8 mantissa bits would price something. Everything that reads it
            # already casts explicitly (`v_mean.to(dtype=...)`, `final_scale.to(g.dtype)`), so
            # pinning it is dtype-safe whatever the parameters are.
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=torch.float32,
                                                          device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        # STILL in the state dtype, for a NEW reason: p.grad is now the momentum history
        # itself and muon_step_fused must not write into it. `torch.stack` always copies, so
        # this is a private copy either way; the `.to(OPT_STATE_DTYPE)` is a no-op here
        # because p.grad is already bf16 (the parameters are), and it is kept so the cast is
        # explicit if OPT_STATE_DTYPE ever moves. Nothing downstream mutates it: the two
        # `lerp_` lines are gone, `X = g.bfloat16()` is a no-op on a bf16 tensor, and the very
        # next line `X = X / (...)` allocates a fresh tensor.
        stacked_grads = torch.stack([p.grad.to(OPT_STATE_DTYPE) for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(stacked_grads, stacked_params,
                        state["second_momentum_buffer"],
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
WINDOW_PATTERN = "SSSS" # sliding window pattern: L=full, S=half context
# CARRIED, not proposed here. Launch 26 (tail_window_sssl_to_ssss, gpu6) measured this
# exact constant at peak_vram_bytes 910,528,512 against a 910,528,512 champion -- zero
# bytes, zero parameters, val_bpb inside the replicate floor -- and refunded 6,291,456
# counted FLOPs per token. _compute_window_sizes forces window_sizes[-1] to the long
# window, so at DEPTH=7 the realised pattern goes SSSLSSL -> SSSSSSL: exactly one
# interior layer drops from full context to half, 12*n_head*head_dim*1024 = 6,291,456.
# Its launch was a pre-registered tie, so it was recorded DISCARD and its source never
# became the base; carrying it here is the only way that paid refund reaches champion.

# Optimization
TOTAL_BATCH_SIZE = 2**16 # = 65,536 tokens per optimizer step. Halving this at fixed
# DEVICE_BATCH_SIZE costs exactly zero bytes -- it enters only at the grad_accum_steps
# division, and gradient buffers are allocated once and accumulated in place -- confirmed
# across launches 6/7 (all four windows byte-identical) and again at launch 17.
#
# THIS IS THE THIRD HALVING OF THIS CONSTANT, and it funds the B=2 rung below. Paid points
# on the curve, each recovered against the twice-validated token account:
#     2**19 -> 2**18   (launch 7)    -0.019017 val_bpb
#     2**18 -> 129,024 (launch 17)   -0.0138
# Decay factor 72.5 pct per halving, so 129,024 -> 65,536 is projected at -0.0100, with a
# credible range -0.0138 to -0.0050. The B=2 rung costs +0.0122 to +0.0188 of val_bpb in
# tokens, so the funding is what keeps this launch inside the 1.05 gate.
#
# 65,536 % 4096 == 0 exactly, so grad_accum_steps = 16 at B=2.
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2 -- RESTORED to the champion's value. See VALUE_EMBED_BETA1.
# THE MECHANISM of this candidate. Launch 46 measured that removing the first moment from ALL SIX
# 2-D AdamW tables buys -50,331,648 B at a token-corrected val_bpb cost of +0.0025795, which is
# affordable against 0.0053226 of slack -- but that launch ran on a node 8.0 ms/step slow, lost 103
# steps, and went INELIGIBLE by +0.0003129 on the token term alone. This candidate takes the same
# mechanism at REDUCED SCOPE so that it clears the gate under BOTH observed throughput draws, and
# in doing so prices the one thing launch 46 could not: whether the cost is proportional to the
# parameters or concentrated in the two dense tables.
#
# beta1 = 0 for the FOUR value-embedding tables only. wte and lm_head keep the champion's 0.8.
VALUE_EMBED_BETA1 = 0.0
# THE MECHANISM of this candidate (hunk a). The first moment of the two DENSE vocab-wide
# tables (`wte`, `lm_head`) is allocated in PINNED HOST memory and staged to the device
# inside the step. 16,777,216 B stops being resident in CUDA memory; the update is bitwise
# unchanged. Selects exactly those two tables: the four value-embedding tables already take
# the `no_first_moment` branch and allocate no first moment at all, and the 1-D scalar
# groups keep their dense device buffers (28 B, deliberately).
OPT_STATE_HOST_RESIDENT = True
LR_SCALE = 0.5          # THE ONLY MOVE THIS LAUNCH: one multiplier on all four base learning
# rates, applied at the setup_optimizer call site below. Nothing else changes -- this program is
# byte-identical to launch 18 (`batch_b2_rung_double_funded`, B=2 / TOTAL_BATCH_SIZE 2**16,
# peak_vram_bytes 1,102,508,032, val_bpb 1.0522592) outside this constant, its call site and this
# docstring, so the whole val_bpb delta is attributable to the learning rate and the target is
# pre-registered at the same integer to the byte.
#
# WHY: the four *_LR constants were tuned at the reference's TOTAL_BATCH_SIZE = 2**19 and have
# never been varied in 18 paid launches, while this run has walked that constant down to 2**16 --
# 8x smaller batches, 3.6x more optimizer updates, same learning rate. Both optimizers here take
# scale-free steps (Muon orthogonalizes and NorMuon renormalizes, so |update| = lr exactly;
# AdamW divides by sqrt(v)), so total distance travelled is lr * mean(lrm) * num_steps and the
# schedule is in progress units -- mean(lrm) = 0.75 for every launch. Distance is therefore
# proportional to num_steps alone, and it has grown 757 -> 2730 across the run at fixed lr.
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# Model size
DEPTH = 7               # number of transformer layers
DEVICE_BATCH_SIZE = 2   # per-device batch size (reduce if OOM). THE TARGET MOVE: 3 -> 2.
# THE LAST RUNG. The compiled training window is the peak and it is affine in this constant on
# a law now pinned by TWO PAID SAME-PROGRAM POINTS (launch 16 at B=4, launch 17 at B=3), which
# predicted launch 17 to a single 64 KiB allocator block:
#     W3(B) = 376,123,392 + 363,192,320*B   ->   W3(2) = 1,102,508,032 = 42.8106x
#     W2(n) = 350,376,448 + 358,973,440*n   ->   W2(2) = 1,068,323,328  (clearance 34,184,704)
# so this line is worth -363,192,320 B (-24.78 pct): 1,465,700,352 -> 1,102,508,032.
#
# B=1 is NOT next and must not be queued: it costs +0.0528 of val_bpb in tokens (unreachable
# even with three further funding halvings), AND W3(1) = 739,315,712 sits inside the W4 band
# (~0.35-0.76e9), so the frozen 128x2048 evaluation would become the peak and the rung would
# not even deliver its bytes. This rung is where device_batch_size ends.

# Rows per block of the grad-free (evaluation) forward. Evaluation runs at a batch this
# file does not choose, so the peak of that window is set here instead: its working set
# is EVAL_ROW_CHUNK rows of vocab-wide logits rather than the whole eval batch's.
#
# 16 was chosen at launch 2, when the reported peak was 6,134,700,032 B and this window
# only had to stop being the largest of four. It was never revisited across the 7.3x fall
# since, and launch 31 PRINTED the consequence for the first time: reported 839,544,832 >
# before_reporter 779,984,896, i.e. THIS window is the max() arm, 59,559,936 B above the
# compiled training window, so it was clipping every byte any other mechanism removed.
#
# W4 = 126,742,528 B resident (parameters and buffers; the optimizer state and gradients
# are already released when the reporter runs) + 712,802,304 B of transient, and the
# transient is one block of vocab-wide logits: at 16 rows that alone is
# 16 * 2048 * 8192 * 2 = 536,870,912 B, 75.3% of it. The block is the whole scale.
#
# 4 rows, not 8. The headline floors at the compiled window (779,984,896) for ANY value
# whose W4 lands under it, so 8 and 4 are worth the identical amount ON THIS LAUNCH; 4 is
# chosen because it puts W4 near 305 MB instead of 483 MB and therefore stops this window
# clipping the NEXT ~475 MB of other people's W3 work, at a cost of nothing measurable --
# the evaluation runs once, after the time-boxed training loop, so it cannot move num_steps.
# Per-token losses are concatenated in row order and F.cross_entropy is computed per row
# along dim=-1, so every per-token value, and therefore val_bpb, is bitwise independent of
# this constant. Verified on CPU at four values before purchase.
EVAL_ROW_CHUNK = 4

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


def report_memory(tag):
    """READ the allocator's counters. Never reset them.

    peak_vram_bytes is a max() over every phase of the process, so a single reported
    number cannot say which phase set it. These reads bracket the phases -- model and
    loader resident, the eager FLOPs probe, training, and the frozen evaluation -- and
    attribute the reported peak without changing it. torch.cuda.max_memory_allocated()
    and torch.cuda.memory_allocated() are pure reads. The allocator's peak counter is
    never cleared anywhere in this file, so the number the frozen reporter reads covers
    the whole process exactly as it does for the reference.
    """
    print(f"[mem] {tag:18s} allocated={torch.cuda.memory_allocated():>14,} "
          f"peak={torch.cuda.max_memory_allocated():>14,}", flush=True)

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
    unembedding_lr=UNEMBEDDING_LR * LR_SCALE,
    embedding_lr=EMBEDDING_LR * LR_SCALE,
    scalar_lr=SCALAR_LR * LR_SCALE,
    adam_betas=ADAM_BETAS,
    matrix_lr=MATRIX_LR * LR_SCALE,
    weight_decay=WEIGHT_DECAY,
)
# WEIGHT_DECAY is deliberately NOT scaled. `muon_step_fused` applies it as
# `stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)`, i.e. the decay is already
# lr-coupled, so scaling lr scales the per-step decay by the same factor for free. Scaling
# WEIGHT_DECAY as well would be a second variable and would square the effect.
print(f"[lr] LR_SCALE={LR_SCALE} -> unembedding={UNEMBEDDING_LR * LR_SCALE} "
      f"embedding={EMBEDDING_LR * LR_SCALE} scalar={SCALAR_LR * LR_SCALE} "
      f"matrix={MATRIX_LR * LR_SCALE} weight_decay={WEIGHT_DECAY} (unscaled, lr-coupled)",
      flush=True)

model = torch.compile(model, dynamic=False)

# Split the parameters once, by optimizer kind, for the carry at the tail of each step. The
# assert in setup_optimizer already guarantees every model parameter is in exactly one group,
# so these two lists partition model.parameters() -- checked here rather than assumed.
muon_params = [p for g in optimizer.param_groups if g["kind"] == "muon" for p in g["params"]]
_muon_ids = {id(p) for p in muon_params}
adamw_params = [p for g in optimizer.param_groups if g["kind"] == "adamw" for p in g["params"]]
assert len(muon_params) + len(adamw_params) == len(list(model.parameters()))
assert not any(id(p) in _muon_ids for p in adamw_params)
print(f"[carry] muon params={len(muon_params)} adamw params={len(adamw_params)} "
      f"muon grad bytes={sum(p.numel() * p.element_size() for p in muon_params):,}", flush=True)

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train",
                              data_budget_tokens=DATA_BUDGET_TOKENS)
print(f"Data budget:    {DATA_BUDGET_TOKENS:,} distinct tokens from the shuffled pool")
harness = (TimeToTargetHarness(TARGET_VAL_BPB)
           if (TARGET_VAL_BPB > 0 or PROBE_ONLY) else None)
x, y, epoch = next(train_loader)  # prefetch first batch
report_memory("before_probe")     # model + rotary + loader buffers, no probe, no training
# Dispatch-level FLOPs, measured HERE while memory still holds only the model and the
# optimizer state. Counted on the uncompiled module by the frozen probe.
from prepare import measure_flops_dispatch
# CARRIED VERBATIM from @autoscts__memory_analyst1's tail_probe_grad_free_hook (5325dda2), hunk (d).
# prepare.py runs loss.backward() to count FLOPs and throws every gradient away one line later, so
# the whole gradient set is live for the length of the eager backward for nothing. Free each one as
# it is accumulated. The probe backprops through the UNCOMPILED module, so AccumulateGrad is the
# accumulator and these hooks fire per parameter as the backward walks. Both the finally and the
# assert: the finally guarantees removal even if the probe raises, the assert turns a silent leak
# into a loud failure. No compiled graph is ever traced with a hook installed -- the first compiled
# call is step 0's forward, below, after .remove().
_probe_handles = [p.register_post_accumulate_grad_hook(lambda _p: setattr(_p, 'grad', None))
                  for p in model.parameters() if p.requires_grad]
try:
    flops_per_token_measured = measure_flops_dispatch(model, x, y)
finally:
    for _h in _probe_handles:
        _h.remove()
    _probe_handles.clear()
assert all(p.grad is None for p in model.parameters()), 'probe grad hook leaked into training'
print(f"FLOPs per token (measured at dispatch): {flops_per_token_measured:,}")
# The eager uncompiled probe runs before any training step, so this read is the ONLY
# place its window is visible: after training starts, the reported max() can hide it.
report_memory("after_probe")

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
    if step == 1:
        # CARRIED RIDER, read-only: @autoscts__memory_gpu6's [bwdattr] attribution
        # ([SUGGESTION] be046b76).  Not proposed here.  228,593,664 B -- 34.1% of the
        # champion -- is created inside the compiled backward and no rig in this run can
        # price it, and launch 40 proved it does not cancel in a delta (22.7x).  This
        # records the allocator's own event trace for step 1 ONLY, which `if step > 10`
        # below excludes from `total_training_time`, so it cannot touch the clock, the
        # token count or the metric.  `context=None` captures no Python stack.  It is a
        # host-side recorder: it allocates no device tensor and resets no counter.
        try:
            torch.cuda.memory._record_memory_history(
                enabled="all", context=None, stacks="python", max_entries=200_000)
        except Exception as _e:
            print(f"[bwdattr] enable failed, rider skipped: {_e!r}", flush=True)
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        if step == 1 and micro_step == 0:
            # CARRIED, not proposed here, and it costs nothing. @autoscts__memory_gpu5 measured
            # that W3 is a max() over TWO instants inside one compiled backward -- the loss tail
            # at ~11.5% and a runner-up holding no vocab-wide tensor at ~24.7% -- with only
            # 6,901,512 B between them on champion v29, and asked the next launch for this one
            # read. `allocated` here is the resident set plus the compiled forward's saved set,
            # i.e. the entry set, with nothing of the backward built yet, and it is the only
            # place that quantity is visible. Their Reading A predicts 654,139,392 EXACTLY on
            # champion v29; THIS candidate removes 50,122,752 B of resident optimizer state, so
            # the same reading here is 604,016,640. A materially lower number says the tail
            # instant is not the arm and every loss-tail item on every board is worth zero.
            # report_memory is a pure read of the allocator's counters -- it allocates nothing,
            # resets nothing, and step 1 is excluded from total_training_time by `if step > 10`,
            # so it cannot touch the clock or the metric.
            report_memory("s1m0_after_fwd")
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        if step == 0 and micro_step == 0:
            # One microbatch's forward+backward, before any optimizer state exists.
            report_memory("s0m0_after_bwd")
        x, y, epoch = next(train_loader)

    if step <= 2:
        # End of the accumulation loop. At step 0 the optimizer state does not exist yet
        # (it is allocated lazily inside the first optimizer.step()), at steps 1-2 it does,
        # so these three reads say whether the reported peak is set before or after the
        # state is resident -- i.e. whether a resident-byte mechanism can move this metric
        # at all. Steps 0-2 are excluded from total_training_time by `if step > 10` below,
        # so no read here can touch the clock, and both calls are pure reads.
        report_memory(f"s{step}_after_accum")

    if step == 1:
        # [bwdattr], second half: read the trace back and histogram the block sizes live
        # at its peak.  Sizes alone name the rows -- a (4096,8192) float32 is 134,217,728,
        # a model-width bf16 at 4,096 tokens is 4,194,304.  Wrapped because the trace
        # record keys were verified against the torch 2.9.1 API and not against a CUDA
        # build's records: a rider must never be able to end a launch.
        try:
            _snap = torch.cuda.memory._snapshot()
            torch.cuda.memory._record_memory_history(enabled=None)
            _live = _peak = 0
            _cur, _at_peak = {}, {}
            for _tr in _snap.get("device_traces", []):
                for _e in _tr:
                    _a = _e.get("action")
                    if _a == "alloc":
                        _cur[_e["addr"]] = int(_e.get("size", 0) or 0)
                        _live += _cur[_e["addr"]]
                        if _live > _peak:
                            _peak, _at_peak = _live, dict(_cur)
                    elif _a in ("free_completed", "free_requested"):
                        _live -= _cur.pop(_e["addr"], 0)
            _hist = {}
            for _sz in _at_peak.values():
                _hist[_sz] = _hist.get(_sz, 0) + _sz
            print(f"[bwdattr] step1 trace peak={_peak:,} blocks={len(_at_peak)}", flush=True)
            for _sz, _tot in sorted(_hist.items(), key=lambda kv: -kv[1])[:16]:
                print(f"[bwdattr]   block={_sz:>12,}  n={_tot // _sz:<4d} "
                      f"total={_tot:>14,}", flush=True)
            del _snap, _cur, _at_peak, _hist
        except Exception as _e:
            print(f"[bwdattr] read failed, rider produced nothing: {_e!r}", flush=True)
            try:
                torch.cuda.memory._record_memory_history(enabled=None)
            except Exception:
                pass

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
    if step <= 2:
        report_memory(f"s{step}_after_opt")
        if step == 1:
            # Tally the state this launch made bf16, by name and dtype, from tensor
            # metadata only. Confirms -67,109,952 B from inside the launch.
            tally = {}
            for _st in optimizer.state.values():
                for _k, _v in _st.items():
                    if torch.is_tensor(_v) and _v.is_cuda:
                        _key = f"{_k}:{str(_v.dtype).replace('torch.', '')}"
                        tally[_key] = tally.get(_key, 0) + _v.numel() * _v.element_size()
            print(f"[optstate] total={sum(tally.values()):,} " +
                  " ".join(f"{k}={v:,}" for k, v in sorted(tally.items())), flush=True)
    # THE MECHANISM. In place of freeing every gradient, scale the Muon gradients so that
    # backward's own accumulation on the next step performs `mu*history + g`, and free only the
    # AdamW ones. mu = get_muon_momentum(step)**2 makes the weight on the current gradient
    # identical to today's Nesterov at every step; carry = mu_next*(1-mu_cur)/(1-mu_next) is the
    # scale that keeps p.grad equal to the heavy-ball buffer divided by (1-mu). `_foreach_mul_`
    # with a Python scalar is in-place and allocates nothing. `step` has not been incremented yet
    # here, so `step` is the step that just finished and `step+1` is the one about to start.
    mu_cur = get_muon_momentum(step) ** 2
    mu_next = get_muon_momentum(step + 1) ** 2
    carry = mu_next * (1.0 - mu_cur) / (1.0 - mu_next)
    _carried = [p.grad for p in muon_params if p.grad is not None]
    if _carried:
        torch._foreach_mul_(_carried, carry)
    for _p in adamw_params:
        _p.grad = None

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

report_memory("after_training")   # max() over resident + probe + the training window

# The optimizer's Nesterov/second-moment buffers and Adam moments are dead the moment
# the last update lands, but they are still resident when the frozen reporter runs its
# evaluation, and the reporter reads the peak AFTER that evaluation. Dropping them
# lowers the bytes this process actually holds; it does not touch the counter that
# reports them, which is the line a cleared peak counter would cross.
optimizer.state.clear()
model.zero_grad(set_to_none=True)
gc.collect()
report_memory("before_reporter")  # allocated= is the resident set evaluation starts from

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
# CARRIED RIDER, read-only, 0 B. Mechanism and pre-registration @autoscts__memory_gpu1,
# INTERCEPT queue.md `intercept_validation_window_print` ([PROPOSAL] 2266091a), which is
# rider_only and can therefore only ever ride. It rides HERE, on this launch, because this
# candidate is the first one whose predicted peak (555,781,120) falls BELOW W2 = 565,531,136:
# champion.md v36 names the VALIDATION window as the last unmapped one and says it becomes the
# binding window the moment a row reaches W2. If validation is the arm, this launch's headline
# will read above its pre-registration and this single line is the difference between knowing
# that and guessing it.
#
# It is one READ of a monotonic counter, after the frozen reporter has already returned, so it
# moves no scored column: report_efficiency_metrics has computed and printed every official
# number before this executes. It never calls reset_peak_memory_stats, and it prints no
# METRICS_JSON line -- the champion already prints `depth:` after the reporter, so a plain line
# here is the existing shape of this file's tail. Not part of this row's mechanism and not
# offered as a byte claim: predicted delta 0 B.
if torch.cuda.is_available():
    print(f"[valwin] after_reporter  peak={torch.cuda.max_memory_allocated():>14,}", flush=True)
print(f"depth:            {DEPTH}")
