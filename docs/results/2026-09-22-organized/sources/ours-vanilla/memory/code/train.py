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


class _RmsNormOutputSaved(torch.autograd.Function):
    """rms_norm holding its OUTPUT for backward instead of a FLOAT32 COPY of its INPUT.

    THIS IS c0063's MECHANISM WITH c0063's PRICE REMOVED. c0063 measured the mechanism and it
    worked: peak_vram_bytes 846,045,696 -> 651,086,336, -23.05%, the largest key reduction in
    this arm's history and the first to get through the 822,992,896 evaluation floor. It was
    REJECTED, on the gate, at val_bpb 1.0525208 -- and the breach was fully attributed: -4.06%
    of throughput x 0.1095 sd per 1% of tokens = -0.445 sd against an incumbent margin of
    +0.035 sd. The arithmetic contributed nothing; flops_per_token_measured did not move by a
    single unit, and the verifier had already shown the gradients sitting BELOW the parent's own
    float32 rounding floor. So the mechanism is not in question. Only its cost is.

    WHERE THE 4.06% WENT, AND IT IS BANDWIDTH NOT LAUNCHES. c0063's backward was a chain of
    elementwise ops -- mul, addcmul, mul -- and each one WRITES a full-size tensor where ATen's
    fused rms_norm backward writes one. driver/fused_bwd_r64.py counts them directly, on the GPU,
    through a TorchDispatchMode that reports every op whose output has the activation's numel
    (`detach` is a view and writes nothing):

        parent  F.rms_norm  : 1 full-size write   [_fused_rms_norm_backward]
        c0063   closed form : 3 full-size writes  [mul, addcmul, mul]
        THIS    fused bwd   : 2 full-size writes  [mul, _fused_rms_norm_backward]

    At 20 norm sites x 8 grad-accumulation microbatches = 160 norm backwards per step, and ~25 MB
    of read+write traffic per full-size bf16 activation, c0063's two extra writes are ~12 GB of
    extra traffic per step -- about 8 ms on a 180 ms step, 4.4%, against 4.06% measured. Kernel
    LAUNCH overhead accounts for only ~1.3% of it and is the wrong diagnosis. This form carries
    ONE extra write instead of three, so the same model prices it at ~1.35%.

    HOW THE INPUT COMES BACK WITHOUT BEING SAVED. aten::_fused_rms_norm_backward needs the input,
    which is exactly the tensor this card refuses to hold. But y = x*r, so x = y*(1/r), and rstd
    is one float32 element per row -- 1/384th of a unit. Reconstruction is one kernel and it is
    exact to half an ulp at every dtype (measured: 0.509 fp64, 0.518 fp32, 0.510 bf16). So the
    trade is: drop a full-size FLOAT32 tensor from the FORWARD, add a full-size BF16 transient to
    the BACKWARD. Round 49 measured a backward buffer at 0.247x the key-leverage of a forward one
    and this one is also half the width in bytes, so the trade is priced in this arm's own
    currency at roughly 8:1 and c0063 confirmed the sign at -252,198,912 on the probe.

    IT IS MORE ACCURATE THAN THE FORM THAT WAS LAUNCHED, not less. Against ATen's own autograd,
    at the shape the model uses:

        c0063 closed form : 0.442 / 0.442 / 0.883 ulp   at fp64 / fp32 / bf16
        THIS  fused bwd   : 0.442 / 0.442 / 0.441 ulp

    -- because it IS ATen's kernel, run on an input recovered to half an ulp, rather than a
    re-derivation of it. The bf16 figure halves, and bf16 is the only dtype the launch runs in.
    The negative control that makes those numbers mean anything perturbs rstd by 64 ulp of the
    working dtype and IS caught at all three (64.5 / 63.6 / 64.4 ulp). The FIRST version of that
    control perturbed by a flat 1e-3 and was NOT caught at bf16 -- correctly, because one bf16
    ulp is 7.8e-3, so the error was below the representable resolution. The control was mis-sized,
    not blind, and a negative control that is not scaled to the resolution of the dtype it tests
    silently tests nothing in the only dtype that matters.

    WHAT THIS COSTS IN VERIFICATION, STATED PLAINLY. aten::_fused_rms_norm_backward has NO CPU
    KERNEL -- it is registered for CUDA and Meta only. This arm's verifier runs the real model on
    CPU, so the backward CANNOT be checked there, and driver/verify_c0064.py asserts that
    CUDA-only fact rather than skipping past it. The backward is instead verified on the GPU,
    against ATen's own autograd, at the exact kernel and the exact dtype the launch uses, with a
    working negative control. That is a stronger check of the backward than the CPU one was; it is
    a weaker check of the backward's interaction with the whole model, and the reason that is
    acceptable is that c0063 already passed the whole-model check with a LESS accurate backward
    and its gradients came in at 0.63x the parent's own float32 rounding floor.

    Nothing here is written in place: in-place operations inside the checkpointed, compiled region
    are a class this arm closed after paying for it twice at c0043 and c0061.
    """

    @staticmethod
    def forward(ctx, x):
        y, rstd = torch.ops.aten._fused_rms_norm(x, [x.shape[-1]], None, None)
        ctx.save_for_backward(y, rstd)
        return y

    @staticmethod
    def backward(ctx, g):
        y, rstd = ctx.saved_tensors
        # One full-size write to recover the input, then ATen's own fused backward. reciprocal()
        # is on the tiny per-row rstd, not on the activation, so it is free.
        x = y * rstd.reciprocal().to(y.dtype)
        dx, _ = torch.ops.aten._fused_rms_norm_backward(
            g, x, [y.shape[-1]], rstd, None, [True, False])
        return dx


def norm(x):
    # Under grad the output-saving Function; with no backward to feed there is nothing to save
    # for and the untouched call is both cheaper and, more importantly, the SAME CODE the frozen
    # evaluation has always run -- so val_bpb's arithmetic is unchanged code, not merely
    # equivalent arithmetic, and this card's only exposure to the gate is through training
    # dynamics. The branch is on grad mode, which Dynamo already specialises on, and NOT on
    # shape, on batch size, or on anything that could distinguish the FLOPs probe from a training
    # microbatch -- the two share the shape (8, 2048) and a discriminator would be fraud against
    # the measurement.
    if torch.is_grad_enabled():
        return _RmsNormOutputSaved.apply(x)
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
        q, k = norm(q), norm(k)

        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        y = y.contiguous().view(B, T, -1)
        y = F.linear(y, self.c_proj.weight)
        return y


def _relu_squared(z):
    """The MLP's activation, recomputed in backward rather than retained.

    MLP.forward wraps this in torch.utils.checkpoint (imported below; module globals are
    resolved at call time, so the ordering in this file is cosmetic). Both operations are
    elementwise, which prepare.py's measure_flops_dispatch does not count, so recomputing
    them is free against the FLOPs ceiling. Neither c_fc nor c_proj is inside this
    function: they are matmuls, the probe counts matmuls, and this axis has zero slack.
    """
    return F.relu(z).square()


class _ReluSquaredExact(torch.autograd.Function):
    """relu(z)**2 holding ONE hidden-width tensor at its peak instead of two.

    THIS DOCSTRING OVERTURNS THE ONE IT REPLACES, WHICH DECLINED EXACTLY THIS CHANGE.

    The previous version of this class said of the output-only identity: 'It measures at FIVE as
    well, not four. It buys no further reduction ... so it is declined on measurement rather than
    preferred on cleverness.' That measurement was taken with a census of hidden-width tensor
    OBJECTS, and round 49 retired that census as defective: it counted an in-place result and a
    detach alias as two separate buffers, which is precisely the aliasing this change creates, so
    it was blind to the only saving on offer. The replacement instrument counts co-live storage
    BYTES keyed on untyped_storage().data_ptr(). Under it the previous version measures FOUR
    co-live hidden-width activations across a five-layer stack and this one measures THREE.

    The instrument that licenses the change is the same one that must not price it: it failed
    against a paid launch pair at 59.1% relative error, so it says a reduction EXISTS and never
    how large it is. The size comes from a paid pair in this same region (one forward buffer of
    this Function, at this shape, moved the ranking key by 96,993,280 bytes) and from nowhere else.

    What changed. The previous forward allocated a second hidden-width buffer for the clamp and
    squared it in place, leaving c_fc's z and the clamp's r co-live. This one clamps IN PLACE ON z
    and squares that in place too, so the only hidden-width buffer in the forward is the one c_fc
    already allocated, which becomes s.

    Why mutating z is legal. z is F.linear(x_chunk, fc_weight)'s output. Linear's backward needs
    g_z^T x_chunk and g_z fc_weight; it saves its INPUTS, not its output, so no autograd node holds
    z's values. _mlp_chunk binds z to h and immediately rebinds h to this Function's result, which
    is the same object, so nothing outside the chunk can see it either. mark_dirty(z) bumps the
    version counter, which is the point: if either statement above is ever false, a saved-tensor
    read raises loudly instead of quietly returning mutated values. The chunk is wrapped in
    checkpoint(use_reentrant=False), so the forward runs twice and each pass mutates the z it
    allocated itself.

    Why the gradient needs no z. relu(z)**2 = s gives relu(z) = sqrt(s), and where z < 0 the
    output is 0 EXACTLY -- squaring cannot round zero -- so sqrt(0) = 0 is exactly the gradient
    wanted and no mask is needed. ds/dz = 2*sqrt(s) therefore depends on the output alone, and
    c_proj retains s for its own grad_weight regardless, so saving it costs nothing.

    It is BIT-IDENTICAL in bf16 and fp32, which is not what I expected when I wrote it. Squaring
    and square-rooting with correct rounding recovers any NORMAL float exactly, so
    sqrt(relu(z)**2) == relu(z) bit for bit, and IEEE multiplication commutes, so
    (sqrt(s)*g)*2 == (g*relu(z))*2 exactly. My first draft of this paragraph claimed the opposite
    on the grounds that 'a sqrt loses an ulp' -- an argument about the shape of an operation
    instead of its arithmetic, and this sqrt is the inverse of the square immediately before it.

    The exception is underflow, and it is real. The theorem needs the square to be normal. In fp16
    the gradient is NOT bit-identical, because fp16's smallest normal is 6.10e-05 so relu(z)**2
    underflows for activations below about 7.8e-03, which is common at unit scale; the discrepancy
    there is bounded at 2.4e-32 absolute. bf16's threshold is 1.1e-19 and an activation cannot
    reach it, and this substrate trains under bf16 autocast. The failing fp16 case is kept in the
    verifier's table rather than dropped from it.

    The no-grad inference path still calls the untouched _relu_squared below, so val_bpb's own
    arithmetic is unchanged code, and the sqrt appears only in a backward the inference path never
    runs.
    """

    @staticmethod
    def forward(ctx, z):
        z.clamp_min_(0)
        ctx.mark_dirty(z)
        s = z.square_()
        ctx.save_for_backward(s)
        return s

    @staticmethod
    def backward(ctx, g):
        (s,) = ctx.saved_tensors
        # g is grad_output: it belongs to autograd, it may be a view and it may be shared with
        # another edge, so it is never mutated. Only the buffer torch.sqrt allocates on the next
        # line is written in place, and it is not a graph input, not saved, and reachable from
        # nothing else. Round 49 measured that a buffer saved HERE is worth 3,503,104 bytes of the
        # ranking key against 96,993,280 for one saved in the forward, so the backward is written
        # for clarity and the forward is written for the key.
        r = torch.sqrt(s)
        r.mul_(g)
        r.mul_(2.0)
        return r


def _mlp_chunk(x_chunk, fc_weight, proj_weight):
    """One token-slice of the MLP: c_fc, relu-squared, c_proj. Recomputed in backward.

    Mathematically identical to the unchunked MLP, because the MLP is pointwise in the
    token axis: every token's output depends only on that token's input. Slicing that
    axis changes when arithmetic happens and how much is co-live, never what is
    computed. Round 37 confirmed this empirically as well as by construction -- its
    val_bpb moved by exactly the amount its token count explains at the measured
    elasticity of 0.090176 sd per percent, with nothing left over.

    _relu_squared is called directly rather than wrapped in its own checkpoint here.
    This slice is already checkpointed, so its interior is discarded and rebuilt only
    for its own backward, and round 8 measured that checkpoint BOUNDARY COUNT is what
    costs throughput -- 16 elementwise-sized boundaries cost 33 percent, while 5
    block-sized ones cost 9.88 percent for 47 percent more FLOPs. One boundary per
    slice instead of two halves that count.
    """
    h = F.linear(x_chunk, fc_weight)
    # _ReluSquaredExact, not _relu_squared: same values, one fewer hidden-width tensor live at
    # the peak. This is the ONE region the ranking key has been shown to see -- the probe's
    # demand carries a term of 14.006 tensors of (16,384/k) x W, measured across three shapes
    # and two values of k -- where the head chunk, which c0046 and c0047 both attacked, is
    # outside the probe's peak entirely.
    h = _ReluSquaredExact.apply(h)
    return F.linear(h, proj_weight)


class MLP(nn.Module):
    # 1728 = 4.5 x n_embd, and the first width in this run ABOVE the substrate's own
    # 4 x 384. Round 38 measured width at 0.00463777 sd of val_bpb per unit over
    # 1376-1536, and measured that it is REVERSIBLE: widening back returned what
    # shrinking cost, at the same rate, matching the downward rate at the same place
    # to within 4.5 percent. It also measured why width is now cheap in bytes --
    # PK-forward-done moved 262,144 bytes and PK-backward-done 1,179,648 across a
    # 160-unit increase, so the only strongly width-dependent term left in the key is
    # the probe's overhead, which scales as W/k. About 87 k bytes per unit against
    # 0.0046 sd is ~19 M bytes per sd, where raising k yields ~125 M per sd.
    #
    # These 192 units are here to PAY for the chunk-count increase below, and the
    # extrapolation is declared: the 0.0046 rate was measured entirely at or below
    # 1536, so its last 192 units are an extension of a straight line past the only
    # neighbourhood that has ever been measured.
    # ROUND 85g, parent c0079, ONE INTEGER: MLP hidden width 1536 (was 1728, -192).
    #
    # WHY WIDTH IS BACK, AND WHY IT IS NOT A REPEAT. For sixty rounds the ranking key was the
    # uncompiled FLOPs probe, and round 38 measured width as CHEAP IN BYTES against that producer:
    # PK-forward-done moved 262,144 bytes and PK-backward-done 1,179,648 across a 160-unit
    # increase, i.e. ~9 k bytes per unit, which is why width was priced out as a byte lever and
    # used as a QUALITY currency instead. c0079 changed the producer. The probe no longer binds
    # (its peak collapsed to 533,352,448, below the peak-diag stage) and the compiled TRAINING
    # LOOP binds at 623,297,536. Round 85g's snapshot attributed that loop's peak instant and the
    # four largest live tensors are 28,311,552 bytes each, from _mlp_chunk lines 302 and 309 --
    # exactly 8192 x 1728 x 2 bytes, one MLP hidden interior for one token chunk at bf16, four of
    # them co-live. So against THIS producer width costs 4 x 8192 x 2 = 65,536 bytes per unit,
    # about 7x what it cost against the probe. The axis was priced out on a producer that no longer
    # exists; this is the first time it has been priced against the loop.
    #
    # THE BYTE PREDICTION, WHICH IS THE POINT OF THE LADDER. If the peak instant holds exactly four
    # hidden interiors, the key falls by 65,536 x 192 = 12,582,912 bytes, to 610,714,624. Three rungs at
    # -192 / -384 / -576 test the SLOPE, not just the sign: 65,536/unit means the snapshot's
    # four-tensor account is right; ~16,384/unit means only one interior is co-live at the peak and
    # the other three are elsewhere in the step; ~0 means the loop's max is not this instant at all
    # and the attribution is wrong. Each of those outcomes is a different next round, so no rung is
    # a replicate of another.
    #
    # THE QUALITY PRICE, MEASURED AND NOT ASSUMED. Round 38 fitted 0.00463777 sd of val_bpb per
    # unit of width over 1376-1536, and measured REVERSIBILITY: widening returned what shrinking
    # cost at the same rate to within 4.5%. At 192 units that is 0.890 sd. Converted OUT of sd
    # units, because a sigma multiple is for 'this is not noise' and never for 'this is
    # affordable': at the n=13 residual sd of 0.00016283 it is 0.000145 in val_bpb, against the
    # incumbent's entire gate margin of 0.000578627 -- 25% of the margin available. That
    # comparison has no ruler in it: both numbers are val_bpb.
    #
    # WHAT RUNS THE OTHER WAY, AND IT IS NOT SMALL. Narrowing the MLP LOWERS FLOPs per token, so
    # the fixed 600 s window fits more steps, and val_bpb on this arm is ~99% a step-count channel:
    # the r84 line gives -1.317e-05 per step (n=13, leave-one-out slope range -1.349e-05 to
    # -1.289e-05). Round 38's 0.00463777 rate was measured AT LAUNCH and therefore already nets
    # whatever throughput it bought at those widths, so adding a separate step-count credit here
    # would double-count. I do not add one, and I state that the two effects have opposite signs
    # and that this is the reason the verdict is registered as genuinely two-sided rather than as
    # an expected pass.
    #
    # ADMISSIBILITY. flops_per_token_measured falls with width (the arm's fitted law is affine in
    # HIDDEN at 32,640 per unit), and the ceiling is an upper bound at 239,078,400 against a
    # current 196,217,088, so every rung moves further inside the ceiling. training_data_tokens
    # available is untouched. Width is NOT arithmetically inert -- it is a different model, and
    # this card does not claim otherwise; what it claims is that the price is measured on both
    # sides in the same units as the gate.
    #
    # EXTRAPOLATION DECLARED. The 0.00463777 rate was measured entirely at or below 1536, so the
    # first 192 units of this descent RE-ENTER the measured band and the rest of the ladder walks
    # below it. Width was also measured FLAT above 1856 at 0.00108 sd/unit, which is evidence the
    # curve has structure, so the rate at 1152 is an extension of a line, not a reading.
    # ROUND 85i, parent c0086 (THE LEADER, key 610,104,320, margin 0.000603192), ONE INTEGER:
    # MLP hidden width 1504 (was 1536, -32).
    #
    # WHY THIS EXACT WIDTH AND NOT A DEEPER ONE. The round 85g ladder measured the margin curve at
    # three widths of this family: 1728 -> +0.000578627, 1536 -> +0.000603192, 1344 -> -0.003392106.
    # Nearly flat, then falling hard. Interpolating BETWEEN the two bracketing measured rungs (no
    # extrapolation, and no line fitted through a visibly nonlinear curve) the half-margin floor is
    # crossed at 206.5 units of reduction from 1728, i.e. width 1521.5, and the zero-margin crossing
    # sits near width 1507. So the affordable band is width 1521 down to about 1507 and this rung at
    # 1504 is inside or just below it BY CONSTRUCTION. Deeper rungs were BUILT (c0092-c0098, down to
    # 768) and are NOT dispatched: the shelf rule's output was that they are unaffordable and, at the
    # shallow end, that they cannot even displace the leader. An unbuilt rung costs nothing; a
    # wrongly dispatched one costs a charge.
    #
    # THE BYTE PRICE, RE-FITTED ON THIS FAMILY'S OWN ROWS AND NOT ASSUMED. Adjacent-pair slopes are
    # 68,714.7 bytes per unit over 1728->1536 and 57,104.0 over 1536->1344, so the price is falling
    # with width and the four-co-live-interior account (65,536/unit) is inside the band it predicted
    # but is NOT a constant. This rung is priced at the LOWER, more recent slope: predicted key
    # 608,276,992 = 610,104,320 - 57,104 x 32. If the true slope keeps falling the row lands above
    # that and the shortfall is the measurement I want.
    #
    # WHAT MAKES IT WORTH A CHARGE AT ALL, STATED HONESTLY. It buys at most about 1.8 MB, which is
    # 0.30% of the key -- small. The reason to spend it is that it also PINS THE CROSSING of a
    # curve that currently rests on two bracketing points 192 units apart, and the crossing is what
    # decides whether the remaining charges belong on this axis at all.
    #
    # WHAT IS NOT CLAIMED. Not arithmetically inert; a different model, and no row of it joins the
    # inert-substitution series. The r84 quality line does not apply to it: c0086 sits +27.4 sd above
    # that line and PASSES, because the line describes rows whose val_bpb moves only through the step
    # count. That is why this rung is priced in the gate's own observed margin and not in sigma.
    # FLOPs stay affine and fall at the CORRECTED 30,720 per unit (32,640 was falsified by both
    # ladder rows), so this rung moves further inside the 239,078,400 ceiling, never nearer.
    HIDDEN = 1504
    # ROUND 64: 4 -> 2. This is the first time in this arm's history that k moves DOWN, and it
    # is not a retreat -- it is the first purchase this arm has been able to afford.
    #
    # WHAT CHANGED THAT MAKES IT AVAILABLE. For sixty rounds bytes were the scarce resource and k
    # only ever went up, because every doubling of k BUYS key. Round 44 closed the axis upward:
    # 'the chunking axis is priced out at EVERY admissible width', because k=8 costs 1.708 sd of
    # tokens and no admissible width supplies it. THAT ENUMERATION WAS OVER INCREASES IN k. It is
    # exhaustive over increases and says nothing about decreases, because a decrease was never
    # worth considering while the key had no slack. c0063 created 194,959,360 bytes of slack --
    # 23% of the objective, in hand, from a change that is arithmetically exact and FLOPs-neutral.
    # The scarce resource is now MARGIN, and k is the cheapest margin this arm has ever priced.
    #
    # THE PRICE, IN THIS ARM'S OWN MEASUREMENTS. Round 39's CORRECTION established that boundary
    # cost is roughly constant PER DOUBLING of k rather than per boundary: k=1->2 cost 7.81% of
    # throughput and k=2->4 cost 7.15%, netted of FLOPs. So k=4->2 should RETURN about 7.15%,
    # which at the paired elasticity of 0.099052 sd per 1% is +0.708 sd. The key side: round 38
    # fitted the probe's instrument overhead as ~0.2554 M bytes per unit of W/k, so halving k
    # roughly doubles it. On c0063 that overhead is 593,846,784 - 533,352,448 = 60,494,336, so the
    # doubling costs ~60 M rather than the ~110 M the same doubling would have cost on c0059 --
    # BECAUSE THE NORM CHANGE SHRANK THE INTERIOR THAT k MULTIPLIES. The two edits in this card
    # interact in the card's favour on the key, which is the reason they belong together and not
    # in two rounds. That makes this ~60 M bytes for ~0.708 sd = ~85 M bytes/sd, against ~190 M
    # for the width purchase that is the only alternative, and width is measured FLAT above 1856
    # at 0.00108 sd/unit so it cannot supply this much at any price the key can pay.
    #
    # k does NOT enter the FLOPs law. flops_per_token_measured = 143,132,928 + HIDDEN * 32,640 is
    # exact and independent of k across every k this arm has run, so this edit is admissibility-
    # neutral and the ceiling is untouched.
    #
    # It costs a larger co-live MLP hidden chunk in the training loop -- ten slice boundaries
    # instead of twenty, and each chunk twice the width -- and that cost is real and is included
    # in the ~60 M above. k=2 is not a new configuration for this lineage: c0038 ran it and held
    # the gate, so this is a return to a measured point, not an extrapolation past one.
    MLP_CHUNKS = 2

    def __init__(self, config):
        super().__init__()
        # Hidden width 1376 -> 1360: a SIXTEEN wide step. 16 is a multiple of 8, which
        # is all bf16 storage requires; every 'multiple of 64' floor earlier versions of
        # this file assumed was a tiling argument, never a correctness one.
        #
        # Sized by margin rather than by appetite. 0.11356 sd of gate margin remains and
        # the capacity frontier has priced four measurements at 75-85 M bytes per sd, so
        # what is left to buy is about 9 M bytes whatever it is spent on. Two things are
        # known and adverse: the width curve steepens below 1376 (the 64-wide step
        # 1408 -> 1344 cost 0.142 sd for its first 32 and 0.339 sd for its second), and
        # every width price on record was measured with fp32 weights while this model is
        # now entirely bf16, so those prices are a prior and not a measurement.
        #
        # Round 26 profiled the probe's own shape (8 x MAX_SEQ_LEN = 16,384 tokens) and
        # found the ranking key is set inside the BACKWARD, where what is live is every
        # block's retained inputs plus ONE block's recompute interior. Round 27 then cut
        # hidden width by 384 and measured that interior fall by 75,792,384 = SIX tensors
        # of 16,384 x 384 x 2 = 12,582,912 -- six hidden-width tensors are co-live in a
        # block interior, not the two I had predicted. They are the recomputed forward's
        # two hidden tensors, the inner activation checkpoint's recompute of one of them,
        # the gradients of both, and one more. Meanwhile per-block RETENTION moved by
        # under 1.7 MB, confirming these tensors exist only inside the recompute window.
        #
        # Round 28 then took a 64-wide step at this shape and measured BOTH sides of it,
        # which is what this round is priced against rather than against round 27:
        # the key fell 29,704,192 (1,545,149,440 -> 1,515,445,248) and val_bpb fell to
        # 1.0466216845712069, so the gate margin GREW to 0.0033783 = 0.742 sd. Against
        # the same-instrument parent c0026 that step cost 0.0017090 of val_bpb, which is
        # 0.375 sd, and the 384-wide step cost 2.005 sd, so the price is 0.334 to 0.375
        # sd per 64 and close to linear in the cut.
        #
        # Two things this comment used to assert are now measured false, and both
        # retractions are why the step above is 32 and not 64.
        #
        # RETRACTED 1: that 64 is the smallest honest step. That was a tiling argument,
        # not a correctness one -- bf16 tensor cores require a multiple of 8. So 32 is
        # legitimate, and it is the first step of this run sized with real room instead
        # of a sliver.
        #
        # RETRACTED 2: that the key delta per step decays as the model shrinks. Two
        # points said so (-29,704,192 then -27,639,808) and I promoted them to a rule and
        # used the rule to size a step. The third point is -33,701,888, LARGER than both.
        # Honest replacement: a 64-wide step is worth 28 M to 34 M of key, a spread with
        # no trend inside it, so a 32-wide step is sized at half of that spread and no
        # more precisely than that.
        #
        # And the reason the 64-step that cost 0.4811 sd cannot be compared cleanly to
        # the ones before it: TIME_BUDGET is 600 seconds, so total_tokens is an OUTPUT.
        # That launch saw 7,733,248 FEWER tokens than this parent (464,125,952 against
        # 471,859,200) because throughput fell rather than rose, and at 0.735 epochs
        # every one of those tokens was new data. So a val_bpb difference between two
        # launches is capacity change PLUS data-seen change, and every width price in the
        # paragraph above is contaminated by whichever way its throughput moved. This
        # file does not correct those prices by an invented scaling exponent; it just
        # stops treating their agreement as evidence, and takes a step small enough that
        # the contamination cannot decide the outcome.
        #
        # A width that is not a multiple of 64 is also a measurement this run has never
        # taken: if 1376 tiles worse than 1408, throughput falls, tokens fall, and the
        # confound shows its own sign directly. That is worth as much as the bytes.
        #
        # The interior arithmetic that sizes the byte count is unchanged and has now been
        # measured at three widths -- 1536, 1472 and 1152: 6 x 64 x 16,384 x 2 =
        # 12,582,912 of one block's interior per step, amplified onto the key by a factor
        # measured between 1.240 (the 64 step) and 1.5145 (the 384 step). This is a
        # SIZING decision from a measured rate; it is not a prediction about val_bpb, and
        # this file makes none.
        self.c_fc = nn.Linear(config.n_embd, self.HIDDEN, bias=False)
        self.c_proj = nn.Linear(self.HIDDEN, config.n_embd, bias=False)

    def forward(self, x):
        # Split the MLP over the TOKEN axis and checkpoint each slice separately.
        #
        # Round 37 measured what this does, at k=4 and width 1376: the key fell
        # 504,758,272 bytes against c0035, of which PK-forward-done moved by ZERO and
        # PK-backward-done by only 50,527,744 -- 90 percent of the win was the
        # instrument overhead collapsing from 543,680,512 to 89,449,984. So the term
        # this attacks is the probe's own uncompiled forward-backward holding EVERY
        # block's MLP interior at once, which is a model-side fact about how much this
        # model asks to have live, not a property of the instrument.
        #
        # The price is throughput, and round 37 priced it: 14.97 percent at k=4, worth
        # 1.34627 sd of val_bpb at the token elasticity that launch measured. That is
        # why k is 2 here and why the width goes back up in the same candidate.
        #
        # The inference path is left exactly as it was, so eval and the decode timings
        # see none of this.
        if not torch.is_grad_enabled():
            x = self.c_fc(x)
            x = checkpoint(_relu_squared, x, use_reentrant=False)
            x = self.c_proj(x)
            return x
        parts = [checkpoint(_mlp_chunk, x_chunk, self.c_fc.weight, self.c_proj.weight,
                            use_reentrant=False)
                 for x_chunk in torch.chunk(x, self.MLP_CHUNKS, dim=1)]
        return torch.cat(parts, dim=1)

    def _forward_unchunked(self, x):
        x = self.c_fc(x)
        # Recompute the activation in backward instead of retaining it. Round 27 measured
        # what this actually leaves co-live in the interior: six hidden-width tensors, of
        # which this recompute accounts for one. Removing this checkpoint was measured to
        # RAISE the key at round 21, so it stays. c_fc's output is still saved (checkpoint's input) and c_proj still
        # saves its own input, so exactly one of the three comes off.
        x = checkpoint(_relu_squared, x, use_reentrant=False)
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


from torch.utils.checkpoint import checkpoint


def _softcap_ce_chunk(logits_chunk, targets_chunk):
    """Softcap + cross-entropy for one slice of tokens. Recomputed in backward.

    GPT.forward wraps this in torch.utils.checkpoint so the fp32 softcap chain and the
    log-softmax are not retained across backward. Every operation here is elementwise or a
    normalisation, and prepare.py's measure_flops_dispatch states that it does not count
    those ("Elementwise and normalisation work is still uncounted"), so recomputing them is
    free against the FLOPs ceiling. lm_head is deliberately NOT inside this function: it is
    a matmul, the probe counts matmuls, and this axis has zero slack on that ceiling.
    """
    softcap = 15
    lg = logits_chunk.float()
    lg = softcap * torch.tanh(lg / softcap)
    return F.cross_entropy(lg, targets_chunk, ignore_index=-1, reduction='none')


# Autocast, explicitly OFF, for the grad-path head chunk only. Constructed once at module
# scope the way the training loop's own autocast_ctx is, entered from inside a checkpoint
# recompute, and never nested within itself.
#
# It is here so that _head_softcap_ce_chunk's dtype is a property of THIS CODE rather than of
# whatever the caller's autocast state happens to be. Two callers matter and they disagree:
# the training forward runs under bf16 autocast, while prepare.py's frozen flops probe and
# this file's own peak-diag stage both call loss.backward() OUTSIDE the autocast context --
# and the chunk is wrapped in checkpoint(use_reentrant=False), so its forward is RECOMPUTED
# during that backward with no autocast in scope. Under an ambient dtype the recomputation
# therefore materialises full-size fp32 exactly in the pass that sets the ranking key, which
# is the reason round 47's version of this idea measured a 268MB eager reduction and carried
# almost none of it into its key.
_HEAD_DTYPE_EXPLICIT = torch.amp.autocast(device_type="cuda", enabled=False)


def _head_softcap_ce_chunk(x_chunk, head_weight, targets_chunk):
    """lm_head, softcap and cross-entropy for one slice of tokens, in bf16 end to end.

    This is _softcap_ce_chunk with the unembedding matmul pulled INSIDE the checkpoint, so
    the whole-batch vocab-wide logits tensor never exists: only one chunk's worth does, and
    only during that chunk's own forward and its own recomputation. What is retained per
    chunk instead is a slice of x, which is n_embd wide rather than vocab_size wide and was
    already retained anyway.

    Recomputing a matmul IS counted by measure_flops_dispatch, which is why the parent kept
    lm_head outside this call. That constraint was real at the reference shape, where
    flops_per_token_measured sat exactly on the ceiling with no slack at all. It is not real
    at this candidate's scale: the second forward through lm_head costs
    2 * n_embd * vocab_size per token, which fits inside the headroom the depth reductions
    opened up. The launch measures that against the ceiling rather than assuming it.

    WHAT THIS REVISION CHANGES, and the byte it is worth. The parent wrote
    `F.linear(x_chunk, head_weight).float()`. Under the training forward's autocast the linear
    returns bf16 -- 4096 * 8192 * 2 = 67,108,864 bytes for one chunk -- and `.float()` then
    allocates a 4-byte copy of 134,217,728 beside it, which is the tensor tanh and
    cross_entropy go on to carry through the forward, the recomputation and the backward.
    Dropping the promotion removes that copy. c0047 measured exactly 134,217,728 off its
    after-head stage marker for this region at this shape, which is where the number comes
    from; it is not derived from a model of the allocator, because this run has established
    that its allocator models are good for existence and bad for size.

    Why bf16 and not fp16. Under bf16 autocast F.linear was already casting both operands to
    bf16 and running this same kernel, so casting them by hand to bf16 leaves the matmul and
    its backward arithmetically what the parent did, and leaves lm_head's weight gradient in
    the format it was already accumulated in. fp16 would put that weight gradient in a format
    whose smallest normal is 6.10e-05, and a weight gradient -- unlike a logit the softcap
    bounds to [-15, 15] -- has nothing making that safe. bf16's smallest normal is 1.1e-19.

    What genuinely changes, stated rather than buried: the softcap's tanh and the
    cross-entropy now run in bf16 where the parent ran them in fp32. That is a real reduction
    in training-gradient precision and it is this candidate's actual risk. It does NOT touch
    the reported metric: val_bpb comes from the no_grad path _softcap_ce_chunk, which stays
    fp32 end to end and is not edited.

    The `.float()` on the way out is not cosmetic. It keeps per_token's dtype identical to the
    parent's, so every branch downstream of this call is unchanged; the cast is rows-only,
    4096 floats, against the 134,217,728 it lets go.
    """
    softcap = 15
    with _HEAD_DTYPE_EXPLICIT:
        lg = F.linear(x_chunk.to(torch.bfloat16), head_weight.to(torch.bfloat16))
        lg = softcap * torch.tanh(lg / softcap)
        loss = F.cross_entropy(lg, targets_chunk, ignore_index=-1, reduction='none')
    return loss.float()


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        # Held in bf16 (cast below, with wte). Under the bf16 autocast that wraps
        # the forward, this matmul is performed in bf16 whatever the leaf dtype is:
        # autocast casts an fp32 weight down at every call site. So fp32 storage buys
        # no arithmetic precision here -- only a wider master copy for the optimizer
        # update and a wider gradient. wte is the same 8192 x 384 tensor in the same
        # AdamW path at a 35x larger lr and has always been bf16, so the asymmetry
        # between the embedding and the unembedding was not buying anything.
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
        # Rotary embeddings.
        #
        # Was config.sequence_len * 10 = 20480 rows. That is not slack: BENCH_SEQ_LEN is 16384
        # and LATENCY_PROMPT_LEN is 4096, and 16384 + 4096 = 20480 exactly, so the substrate
        # provisioned the tables for a cached-decode context plus its prefill. c0055 cut them to
        # sequence_len on the false premise that nothing exceeds it and crashed on the assert
        # below, inside prepare.py's decode-latency probe.
        #
        # The bound that applies here is smaller because the 16384 half is unreachable:
        # prepare._supports_cache requires init_decode_state and decode_step, this model has
        # neither, so the kv-cache and cached-decode probes return early. The call sites that do
        # reach a forward present 2048 (train/eval), 512 (active-params probe and the latency
        # bridge) and 4096 (measure_decode_latency_at at LATENCY_PROMPT_LEN). Decode appends
        # tokens but slices idx[:, -prompt_len:], so T stays at prompt_len.
        #
        # max(sequence_len, LATENCY_PROMPT_LEN) = 4096 rows. MEASURED by c0056, launch 57:
        # peak_vram_bytes 851,550,720 -> 846,045,696 and the eager peak-diag 563,417,088 ->
        # 558,518,272. Surviving rows are bit-identical -- the table is built from arange and row
        # i is i * inv_freq, independent of the length -- and c0056's loss matched its parent's
        # to six decimals for the first twenty steps in the real run.
        #
        # Derived from the frozen constant rather than written as 4096, so it follows prepare.py
        # instead of guessing. This forgoes a KV cache in this lineage, which on a peak-VRAM axis
        # is not a lever: a cache only adds resident bytes.
        LATENCY_PROMPT_LEN = 4096
        self.rotary_seq_len = max(config.sequence_len, LATENCY_PROMPT_LEN)
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
        # Cast embeddings AND the unembedding to bf16. Init above runs in fp32 and
        # the cast happens after it, so lm_head follows exactly the pattern wte and
        # the value_embeds tables already follow.
        self.transformer.wte.to(dtype=torch.bfloat16)
        self.lm_head.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)
        # The block matrices too, which are the last fp32 mass in the model.
        # Everything under transformer.h is an nn.Linear weight with bias=False
        # (c_q, c_k, c_v, c_proj, mlp.c_fc, mlp.c_proj, and ve_gate on alternating
        # layers); normalisation here is the functional F.rms_norm, so there are no
        # learnable norm weights caught by this. Muon runs its Newton-Schulz
        # iteration in bf16 already (X = g.bfloat16()), so this is the precision the
        # orthogonalisation is computed at whatever the leaf dtype is; and under
        # autocast each fp32 weight was being cast per call, twice per block because
        # the block forward is wrapped in checkpoint(). Narrowing the leaves removes
        # that cast traffic instead of adding to it.
        self.transformer.h.to(dtype=torch.bfloat16)

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
        # factored_v selects the ROW-FACTORED AdamW second moment (adamw_step_factored) instead
        # of the full-size one. It is declared HERE, per group, where the parameters are still
        # named, and it is never inferred from a group's position or from a tensor's shape.
        #
        # WHICH GROUPS, AND WHY NOT ALL OF THEM. The AdamW side holds five 8192x384 matrices --
        # lm_head, wte and three value_embeds. Factoring the four embedding-side ones returns
        # 4 x (6,291,456 - 16,384) = 25,100,288 bytes of persistently live optimizer state. The
        # reduction only needs to clear 19,377,664: that is the gap between the two producers of
        # the ranking key on c0064, the compiled training loop at 647,693,312 and the frozen FLOPs
        # probe at 628,315,648. peak_vram_bytes is the maximum over both, so anything beyond the
        # gap is TRUNCATED at the probe and would be quality paid for nothing. 25,100,288 clears
        # it with 5,722,624 of headroom for prediction error, and lm_head -- the parameter most
        # directly coupled to val_bpb -- keeps its second moment EXACT.
        #
        # The two scalar groups are not factored because a 1-D parameter has no row axis.
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0, factored_v=False),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0, factored_v=True),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0, factored_v=True),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0, factored_v=False),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0, factored_v=False),
        ]
        # THE CONTRACT, checked against the NAMED lists and not against a count. A count is the
        # wrong invariant: rotating this list so lm_head lands last still yields two flagged
        # groups, which is exactly the mis-read that would train the card's promise backwards.
        assert {id(p) for g in param_groups if g['factored_v'] for p in g['params']} \
            == {id(p) for p in embedding_params + value_embeds_params}, \
            'factored_v must cover exactly wte and the value_embeds, and never lm_head'
        assert all(p.ndim == 2 for g in param_groups if g['factored_v'] for p in g['params']), \
            'a factored group must hold only 2-D parameters: a 1-D parameter has no row axis'
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
        # The frozen evaluation binds the ranking key as of c0016: both [mem] prints, after
        # the FLOPs probe and after the training loop, read 1,545,149,440, while the reported
        # peak_vram_bytes is 1,627,522,560, and the only thing that runs in between is
        # report_efficiency_metrics. prepare.py's evaluate_bpb pushes EVAL_BATCH_SIZE = 128
        # sequences of MAX_SEQ_LEN through the stack at once -- 262,144 tokens, sixteen times
        # the grad-path microbatch -- and under no_grad nothing is retained but the transients
        # are huge: one layer's MLP hidden state alone is 262,144 x 1376 x 2 = 721,420,288.
        #
        # EVAL_BATCH_SIZE is frozen and is not touched. This runs the same stack over
        # EVAL_CHUNK_SEQS sequences at a time instead. Splitting on the SEQUENCE axis is
        # exact: attention and rotary act within a sequence and never across the batch, so no
        # token's context changes, and evaluate_bpb asks for reduction='none', so it receives
        # per-token losses and does its own masking and summing -- this code combines no two
        # tokens' losses and therefore introduces no reassociation at all.
        #
        # Free on both counts that matter: nothing is recomputed, so the FLOPs tallies cannot
        # move and the grad-enabled probe never enters this branch; and evaluation runs after
        # the training loop, outside the two CUDA synchronisation points that bracket dt,
        # so it consumes no TIME_BUDGET and costs no steps.
        if targets is not None and not torch.is_grad_enabled() and idx.size(0) > EVAL_CHUNK_SEQS:
            parts = [self._forward_chunk(idx[i:i + EVAL_CHUNK_SEQS],
                                         targets[i:i + EVAL_CHUNK_SEQS],
                                         reduction='none')
                     for i in range(0, idx.size(0), EVAL_CHUNK_SEQS)]
            per_token = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
            if reduction == 'none':
                return per_token
            if reduction == 'sum':
                return per_token.sum()
            return per_token.sum() / (targets.view(-1) != -1).sum().clamp(min=1)
        return self._forward_chunk(idx, targets, reduction)

    def _forward_chunk(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        _stage("entry")
        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        _stage("after-embed")
        _stage_bwd("after-embed", x)
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            if torch.is_grad_enabled():
                # Recompute this block's whole interior in backward instead of retaining it.
                # Called directly, every block keeps its qkv projections, its rotary output,
                # its attention output and its MLP hidden state alive from the moment they are
                # computed until that block's backward runs, so all n_layer sets are resident
                # at once. Through checkpoint only the block's inputs are kept and at most ONE
                # block's interior is live at any instant.
                #
                # The price is one extra forward through the blocks, which the frozen probe in
                # prepare.py counts: the block matmuls appear in the dispatch tally and the
                # attention work appears in the attention tally, which should rise by exactly
                # 4/3 since one extra forward is a third of a forward-plus-backward. It is paid
                # out of FLOPs ceiling that is otherwise going unspent.
                #
                # The residual mix and the value-embedding lookup stay OUTSIDE the checkpoint,
                # so ve is a retained tensor either way and this changes nothing the model
                # computes -- only when it is computed. The no-grad branch below keeps the
                # frozen evaluation path calling the block directly, since with no backward to
                # feed there is nothing to recompute for and the wrapper would only add cost.
                x = checkpoint(block, x, ve, cos_sin, self.window_sizes[i],
                               use_reentrant=False)
            else:
                x = block(x, ve, cos_sin, self.window_sizes[i])
            _stage(f"after-block-{i}")
            _stage_bwd(f"after-block-{i}", x)
        x = norm(x)
        _stage("after-final-norm")
        _stage_bwd("after-final-norm", x)

        if targets is not None and not torch.is_grad_enabled():
            # Inference path: no backward will follow, so lm_head can be chunked over tokens
            # with NO recomputation, and therefore at no FLOPs cost of any kind rather than
            # merely an uncounted one. Each token's cross-entropy depends only on its own
            # logits row, so slicing changes no token's loss. Under grad this chunking is
            # not available: it would put a recomputed matmul in backward, which the frozen
            # probe counts against a ceiling with zero slack.
            flat_x = x.view(-1, x.size(-1))
            flat_targets = targets.view(-1)
            parts = [_softcap_ce_chunk(self.lm_head(flat_x[i:i + HEAD_CHUNK_TOKENS]),
                                       flat_targets[i:i + HEAD_CHUNK_TOKENS])
                     for i in range(0, flat_x.size(0), HEAD_CHUNK_TOKENS)]
            per_token = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
            if reduction == 'none':
                return per_token
            if reduction == 'sum':
                return per_token.sum()
            return per_token.sum() / (flat_targets != -1).sum().clamp(min=1)

        _stage("before-head")
        softcap = 15

        if targets is None:
            return softcap * torch.tanh(self.lm_head(x).float() / softcap)

        # Chunk lm_head, the softcap and the cross-entropy together over the token axis,
        # recomputing all three in backward instead of keeping them. Because the unembedding
        # is now inside the checkpoint, the whole-batch logits tensor is never allocated in
        # either direction: it was tokens x vocab_size in bf16, live across the whole
        # backward pass through the blocks, and it carried a gradient buffer of the same size
        # on top of that. The price is one extra forward through lm_head, which the frozen
        # probe does count, and which is paid out of unused FLOPs ceiling.
        flat_x = x.view(-1, x.size(-1))
        flat_targets = targets.view(-1)
        parts = [checkpoint(_head_softcap_ce_chunk,
                            flat_x[i:i + HEAD_CHUNK_TOKENS],
                            self.lm_head.weight,
                            flat_targets[i:i + HEAD_CHUNK_TOKENS],
                            use_reentrant=False)
                 for i in range(0, flat_x.size(0), HEAD_CHUNK_TOKENS)]
        per_token = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
        _stage("after-head")
        if reduction == 'none':
            return per_token
        if reduction == 'sum':
            return per_token.sum()
        return per_token.sum() / (flat_targets != -1).sum().clamp(min=1)

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
def adamw_step_factored(p, grad, exp_avg, exp_avg_sq_row, step_t, lr_t, beta1_t, beta2_t, eps_t,
                        wd_t):
    """AdamW with a per-ROW second moment: identical to adamw_step_fused except that exp_avg_sq
    tracks the row MEAN of grad^2 and the denominator broadcasts along the row.

    WHY THIS EXISTS, AND WHY IT IS NOT A NEW OPTIMIZER. Round 65 censused what is persistently
    live on the device between optimizer steps, from the live optimizer rather than from the
    source, and found a SHAPE asymmetry inside this very file:

        state[muon]/second_momentum_buffer      50,112 bytes for  9,584,928 params  <- rank-1
        state[adamw]/exp_avg_sq             31,457,280 bytes for 15,728,640 params  <- full size

    _step_muon has always kept its second moment as (n, rows, 1) or (n, 1, cols) --
    `v_mean = g.float().square().mean(dim=red_dim, keepdim=True)` -- and this arm has trained 64
    launches with that. This function extends the factorisation THIS FILE ALREADY USES to the one
    group that lacked it. Nothing is imported and no update rule is replaced; one buffer changes
    shape.

    The same census closed the axis I expected to spend this round on: every optimizer buffer here
    is ALREADY bfloat16, params included, so the dtype axis does not exist. The standing note that
    "no candidate has moved the optimizer state's dtype" was true about the CANDIDATES and false
    about the STATE, and the check that settled it cost nothing while the card it prevented would
    have cost a charge for a mechanism the incumbent already had.

    THE ROW MEAN, NOT THE ROW SUM. The mean keeps the buffer on the same SCALE as the full-size
    exp_avg_sq it replaces, so eps=1e-10 and beta2 keep the meaning they were tuned with. A sum
    would rescale the denominator by sqrt(384) and silently change the effective learning rate --
    a quality change disguised as a memory change, which is the thing this arm most wants not to
    do.

    WHY A PER-ROW SECOND MOMENT IS THE RIGHT FACTORISATION FOR AN EMBEDDING, mechanically. Rows of
    wte and of each value_embeds are VOCABULARY ENTRIES; columns are model dimensions. A token's
    gradient is non-zero only in the steps where that token appears, so the dominant variance
    structure across an embedding table is per-token, and exact Adam's benefit on embeddings is
    precisely that a rare token keeps a large effective step. A per-ROW moment preserves that
    exactly. What it gives up is per-dimension adaptivity WITHIN one token's 384-vector, which no
    frequency argument supports and which the Muon side of this same file has done without for
    every launch this arm has recorded.

    grad.float() before the square, matching _step_muon's own `g.float().square()`: the square of
    a bf16 gradient can underflow bf16's smallest normal (1.18e-38) at gradient magnitudes around
    1e-19, and a row mean over 384 entries is exactly where a handful of underflowed terms would
    bias the denominator low and the step high. The reduction is done in float32 and only the
    result is cast back, so the buffer stays bf16 and the arithmetic does not.
    """
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq_row.lerp_(grad.float().square().mean(dim=-1, keepdim=True)
                         .to(exp_avg_sq_row.dtype), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq_row / bias2).sqrt() + eps_t
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
                # One column instead of 384. Same dtype and same device as the parameter, so the
                # buffer stays bf16 exactly as every other optimizer buffer in this file already
                # is; only its shape changes.
                state['exp_avg_sq'] = (
                    torch.zeros(p.shape[0], 1, dtype=p.dtype, device=p.device)
                    if group['factored_v'] else torch.zeros_like(p))
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            # Two compiled step functions, selected by a flag fixed at construction. They are
            # separate functions rather than one function with a branch, so neither graph carries
            # a conditional and neither can recompile because of the other.
            #
            # group['factored_v'] and not .get(): _step_adamw is only ever called on a group that
            # setup_optimizer built, and setup_optimizer sets the flag on every one of them. A
            # missing key is a bug in the group construction and should raise here rather than
            # default silently to the exact path, which would look like a clean run that quietly
            # measured the parent.
            step_fn = adamw_step_factored if group['factored_v'] else adamw_step_fused
            step_fn(p, grad, state['exp_avg'], state['exp_avg_sq'],
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
WINDOW_PATTERN = "LLLL" # sliding window pattern: L=full, S=half context
# Every layer now sees the full 2048-token context instead of three of five seeing 1024. This
# spends FLOPs and allocates nothing: the window is an integer pair passed to fa3.flash_attn_func,
# q/k/v/out are the same tensors at any window size, and flash attention never materialises the
# T x T matrix. The point is that this task prepays compute -- the flops clause is an upper bound
# and 56 percent of it was going unused at 104,990,400 of 239,078,400 -- so converting unused
# ceiling into context is the cheapest quality available. Expected to land near 138,000,000.

# Optimization
TOTAL_BATCH_SIZE = 2**17 # ~131K tokens per optimizer step
# Free on the ranking key: peak memory is set by the microbatch
# (DEVICE_BATCH_SIZE * MAX_SEQ_LEN = 16,384 tokens), and TOTAL_BATCH_SIZE only sets
# grad_accum_steps, i.e. how many times that identical microbatch runs before the optimizer
# steps. Round 5 measured 2**19 -> 2**18 as a bit-for-bit tie on the key with every
# intermediate memory print unchanged, and found this axis update-starved rather than
# batch-starved: it bought 3.1 sd of gate margin for 0.95% of throughput.
# Every learning rate, scaled by 1.25. One factor on all four groups, so this is one knob and
# not four, and the factor is exact in binary on all four constants (0.75, 0.005, 0.05, 0.625).
#
# WHY THIS AXIS AT ALL: in 57 rounds not one candidate has moved a learning rate, a schedule
# shape, a warmup, an init or the data order. Only WINDOW_PATTERN, TOTAL_BATCH_SIZE,
# WEIGHT_DECAY, DEVICE_BATCH_SIZE, DEPTH, HIDDEN, MLP_CHUNKS, HEAD_CHUNK_TOKENS and dtypes have
# ever moved. The gate is now the binding constraint -- the incumbent holds 0.058 sd of margin,
# which is 0.55% of throughput -- so the only way memory cards become affordable again is to buy
# quality that does not cost tokens. A learning rate allocates nothing, changes no shape, adds no
# kernel and alters no FLOP, so it is the cheapest possible way to try.
#
# WHY UP, from this run's own paid measurements and one free reading:
#  - the loss is STILL FALLING at the end of the run. Windowed over 50 steps on c0051, the mean
#    drops 0.038 nats per 150 steps at step 3000 of 3123, where progress is 0.96 and the schedule
#    has already decayed the LR to about 8% of peak. A model that is still descending as its LR
#    goes to zero is under-trained, not LR-limited.
#  - round 5 measured this model as UPDATE-STARVED and bought 3.1 sd by quadrupling the update
#    count. More total movement in parameter space was worth a great deal.
#  - round 12 found less weight decay worth +1.74 sd. Underfitting, not overfitting.
#  - round 45 measured LONGER momentum timescales at -0.694 sd: more gradient averaging HURT. A
#    noise-limited model would have liked it. This one wants to adapt fast.
# The one contrary datum is round 25, where halving the batch again cost 1.27 sd -- but that
# doubled the update count AND doubled the gradient noise per update AND halved the decay, and
# the record says it cannot separate them. Raising the LR at a FIXED batch adds movement without
# adding noise per update, so round 25 does not close this axis; it closes the batch axis, which
# is what it was about.
#
# 1.25 rather than something bigger because this is the first measurement on the axis and I would
# rather read the sign cleanly than overshoot it. Expected effect is a few tenths of a sd in
# either direction, against a dispersion of 0.0045534.
EMBEDDING_LR = 0.75     # was 0.6; token embeddings (Adam), x1.25
UNEMBEDDING_LR = 0.005  # was 0.004; lm_head (Adam), x1.25
MATRIX_LR = 0.05        # was 0.04; matrix parameters (Muon), x1.25
SCALAR_LR = 0.625       # was 0.5; per-layer scalars (Adam), x1.25
WEIGHT_DECAY = 0.05     # cautious weight decay for Muon
# Decay is applied per optimizer step, so the total shrinkage over a run scales with the step
# count. TOTAL_BATCH_SIZE has gone from the reference's 2**19 to 2**17, quadrupling the steps
# taken per unit of data, while this coefficient stayed at the reference's 0.2 -- so the model
# has been receiving roughly four times the integrated regularisation the substrate was
# calibrated for. Scaling by 1/4 restores approximately the reference's total shrinkage. Free on
# the ranking key: this is a float multiplied into an already-allocated parameter tensor.
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# Model size
DEPTH = 5               # number of transformer layers; at HEAD_DIM 128 the width stays 384
# A pure depth cut, unlike depth 8 -> 6. build_model_config computes base_dim = 5*64 = 320 and
# then rounds up to a multiple of HEAD_DIM = 128, giving model_dim 384 and 3 heads -- exactly the
# incumbent's width and head count, with one fewer layer. Parameters fall only 6.7 percent
# (26,345,772 -> about 24,576,010), where depth 8 -> 6 also cut the width and fell 47.7 percent.
# DEPTH is a model-scale knob, not a depth knob: build_model_config sets
# base_dim = depth * ASPECT_RATIO and rounds up to a multiple of HEAD_DIM, so depth 8 gives
# 512 wide with 4 heads and depth 6 gives 384 wide with 3 heads. Round 9 measured this shape
# at peak_vram_bytes 2,896,781,824 (-33.18%) with flops_per_token at half the ceiling, and it
# missed the val_bpb gate by 0.0000622. Note that 7*64=448 rounds up to 512, so depth 7 is the
# only depth below 8 that keeps the full width.
DEVICE_BATCH_SIZE = 8  # per-device batch size (reduce if OOM)
HEAD_CHUNK_TOKENS = 4096  # tokens per softcap+cross-entropy chunk; see GPT.forward
# Sequences per chunk of the no-grad forward; see GPT.forward. Only the frozen evaluation
# reaches this: it is the term that binds the ranking key as of c0016, and cutting it costs
# no FLOPs (nothing is recomputed) and no training time (evaluation runs outside the timed
# region). 16 gives an eightfold cut in the eval transient, which on the measured 5,411
# bytes per token lands about four times below the 1,545,149,440 grad-path floor -- chosen
# for that margin of safety rather than for tightness, since transient accounting has been
# wrong by a factor of two before, and chosen over 8 to keep torch.compile's inlining modest.
#
# ROUND 63 set this to 8 and ROUND 64 KEEPS IT THERE, and c0063 is why it is not optional. The
# post-training evaluation was the SECOND producer of the key and round 62 pinned its value
# exactly at 822,992,896, only 23,052,800 below the incumbent's 846,045,696 -- so it TRUNCATED
# every probe-side reduction to -2.72%. c0063 halved this constant and the recorded key came in at
# 651,086,336: the evaluation producer no longer binds at all, and without this edit the same
# launch would have recorded -2.72% instead of -23.05%.
#
# Costs nothing measurable: no FLOPs (nothing is recomputed), no training time (evaluation runs
# outside the timed region), and no reassociation of the loss (per-token values are concatenated
# in the same order for any chunk size and summed once). NOT claimed bit-identical -- a different
# batch size can select a different GEMM kernel -- but that is a ~1e-7 relative effect on logits
# against a val_bpb sd of 4.55e-3.
EVAL_CHUNK_SEQS = 8

# Stage tracing, c0020, retried in c0021. Rounds 16, 17 and 19 each predicted the ranking key from a model of
# what is live at the peak and missed by 73%, 16% and 41%. The common cause is that no stage
# has ever been measured: the run reports four CUMULATIVE high-water marks and nothing else.
# This reads the allocator's CURRENT allocated bytes at stage boundaries. It never reads or
# resets the peak counters -- resetting them would lower the reported peak_vram_bytes without
# lowering any memory, which falsifies the objective rather than improving it.
#
# The flag is False for every compiled call, so Dynamo specialises on False and prunes these
# calls out of the graph the training loop runs; it is set True only around the one eager
# diagnostic pass below, which is why the tracing cannot cost the training loop a single step.
_STAGE_TRACE = False


_STAGE_SEQ = [0]


def _stage(tag):
    if _STAGE_TRACE:
        _STAGE_SEQ[0] += 1
        print(f"[stage] {_STAGE_SEQ[0]:03d} {tag}"
              f" current={torch.cuda.memory_allocated()}"
              f" max={torch.cuda.max_memory_allocated()}")


def _stage_bwd(tag, t):
    """Backward-side companion to _stage(). Registers a tensor hook that fires when the gradient
    with respect to t is produced, i.e. INSIDE the backward pass, and reports the allocator's
    current and running-max bytes at that instant.

    WHY THIS EXISTS. c0058 bought the forward half of the map and the answer was categorical: the
    probe's forward moves the running max by EXACTLY ZERO. max reads 558,518,272 at all ten forward
    marks, unchanged from before the probe, while current climbs 69,749,248 -> 271,337,984. Then
    [mem] after flops probe reads 846,045,696. So the whole 287,527,424 the probe adds -- and the
    peak moment that IS peak_vram_bytes -- is in the probe's BACKWARD, which carries 574,707,712 of
    transient on top of what its forward retained. measure_flops_dispatch owns that backward and
    prepare.py is frozen, so a hook registered from here during the forward is the only way in.

    WHAT IT RESOLVES. The hooks fire in reverse order, so the sequence reads: bwd-after-final-norm,
    bwd-after-block-4 down to bwd-after-block-0, bwd-after-embed. If max is already 846,045,696 at
    the first of those, the peak is in the HEAD's backward. If it climbs across the block marks, the
    peak is in a block's recompute-plus-backward and the marks say which. If it only reaches
    846,045,696 after the last mark, the peak is in the embedding backward -- which would put it on
    the vocabulary tables' dense gradients, the largest never-tried lever. Those three answers point
    at different cards and there is currently no way to tell them apart.

    WHY IT CANNOT MOVE THE KEY. Three properties, each deliberate. (1) The closure captures the TAG
    STRING ONLY and never t; capturing the tensor would extend its lifetime to the end of the
    backward and could RAISE the peak, corrupting the very number being observed. (2) The hook
    returns None implicitly, so the gradient passes through unmodified -- it is not a gradient
    transform. (3) It reads memory_allocated() and max_memory_allocated() and never calls
    reset_peak_memory_stats(), which would not perturb the harness's measurement but delete it.
    Neither counter read is an aten dispatch, so FlopCounterMode counts nothing and
    flops_per_token_measured is unchanged.

    WHY THE GUARD. Registration is gated on _STAGE_TRACE, which is False for every compiled call, so
    Dynamo specialises on False and the training loop pays nothing -- no hook is ever registered
    across its 3,100+ steps. torch.is_grad_enabled() and t.requires_grad keep it out of the frozen
    evaluation path, where there is no backward for a hook to fire in.
    """
    if _STAGE_TRACE and torch.is_grad_enabled() and t.requires_grad:
        def _hook(_grad, tag=tag):
            if _STAGE_TRACE:
                _STAGE_SEQ[0] += 1
                print(f"[stage] {_STAGE_SEQ[0]:03d} bwd-{tag}"
                      f" current={torch.cuda.memory_allocated()}"
                      f" max={torch.cuda.max_memory_allocated()}")

        t.register_hook(_hook)

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

# Observational only: reads the allocator's own counters. Nothing is reset, so the
# frozen instrument in prepare.py still reports the true peak over the whole run.
print(f"[mem] after model+optimizer: current={torch.cuda.memory_allocated()} peak={torch.cuda.max_memory_allocated()}")

model = torch.compile(model, dynamic=False)

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train",
                              data_budget_tokens=DATA_BUDGET_TOKENS)
print(f"Data budget:    {DATA_BUDGET_TOKENS:,} distinct tokens from the shuffled pool")
_time_cap = float(os.environ.get("AUTORESEARCH_TIME_CAP", "0") or 0)
harness = (TimeToTargetHarness(TARGET_VAL_BPB,
                               **({"cap": _time_cap} if _time_cap > 0 else {}))
           if (TARGET_VAL_BPB > 0 or PROBE_ONLY) else None)
x, y, epoch = next(train_loader)  # prefetch first batch

# ---------------------------------------------------------------------------
# PEAK-RESOLVED profile at the probe's OWN shape (c0026, observational).
#
# Why this exists. The profile further down reads memory_allocated at each boundary, which
# measures RETENTION, and the objective is a PEAK. Round 24 removed 117,453,824 bytes of
# retention -- predicted per layer to within 7 percent and confirmed directly by those very
# boundary readings -- and the reported peak ROSE by 50,462,720. The trace this block produced
# then said why: retention at the end of the forward is 288,624,128, 18.7 percent of the key,
# and the peak instant stands about 682 MB ABOVE the retention line, so removing retention
# cannot move the key unless it also lowers that instant -- and round 24 moved allocations
# INTO it. The forward's own maximum is 717,410,304, which is 827,739,136 below the key, so no
# forward transient can set it either. The key is set inside the BACKWARD.
#
# Why it must run HERE, before the probe. max_memory_allocated is a running maximum over the
# process, and resetting the peak statistics is forbidden in this run: it would lower the
# reported peak_vram_bytes without lowering one byte of memory, which falsifies the objective.
# So the only honest way to observe this pass's own peak is to run while the process maximum is
# still just the model and the optimizer, about 125,652,480. Every max reading below is then a
# genuine running maximum of THIS pass. Nothing is reset anywhere.
#
# Why at EIGHT sequences. The two earlier profiles ran ONE sequence, an eighth of the probe's
# 8 x MAX_SEQ_LEN. At one sequence the head's transient is fixed at HEAD_CHUNK_TOKENS x vocab
# while every per-token term sits at an eighth of its true size, so the mix of fixed and
# per-token terms is wrong by a factor of eight, and extrapolating it is what produced the
# 41 percent, 11 percent and 73 percent prediction misses. x[:8] is the probe's exact slice, so
# these numbers need no extrapolation: the largest jump IS the term that sets the key.
#
# What it costs. Prints only, and no hooks: flag-guarded print sites were measured free
# (+0.12%) because Dynamo specialises on the False flag and prunes them, while flag-guarded
# hook registrations cost 1.75%. This runs before t_start_training, so it spends setup time
# rather than TIME_BUDGET and costs zero steps.
#
# What it was measured to cost, on the key: NOTHING. c0026 ran this block against this same
# parent and reported 1,545,149,440 bit for bit, with both FLOPs tallies bit-identical. The
# page-high shift predicted from round 22 did not materialise, and the reason is visible in
# the trace: this pass peaks at 971,051,008, comfortably BELOW the probe's own peak, and the
# probe's collect-and-empty absorbs the rest. Allocations before the probe perturb the key
# only when their own peak approaches it.
#
# What the trace is here to check THIS round is the width arithmetic: whether one block's
# recompute interior falls by exactly the two tensor-widths the MLP no longer holds.
#
# Both call sites that matter wrap the forward in bf16 autocast and leave the backward outside
# it; c0020 died because a diagnostic forgot that, so this mirrors them exactly. The whole pass
# is inside try/except because an observational insert must never be able to abort the
# measurement it is observing.
try:
    _pk = model._orig_mod if hasattr(model, "_orig_mod") else model
    _px, _py = x[:8].contiguous(), y[:8].contiguous()
    print(f"[stage] peak-diag-tokens={_px.numel()}"
          f" baseline_max={torch.cuda.max_memory_allocated()}")
    _STAGE_TRACE = True
    with autocast_ctx:
        _ploss = _pk(_px, _py)
    _stage("PK-forward-done")
    _ploss.backward()
    _stage("PK-backward-done")
    _STAGE_TRACE = False
    _pk.zero_grad(set_to_none=True)
    del _ploss, _px, _py
    print(f"[stage] peak-diag-end current={torch.cuda.memory_allocated()}"
          f" max={torch.cuda.max_memory_allocated()}")
except Exception as _pk_exc:
    _STAGE_TRACE = False
    (model._orig_mod if hasattr(model, "_orig_mod") else model).zero_grad(set_to_none=True)
    print(f"[stage] peak-diag-FAILED {type(_pk_exc).__name__}: {_pk_exc}")
# Observational only, as above. Read immediately before the frozen FLOPs probe so that
# the probe's own high-water mark can be separated from the training loop's.
print(f"[mem] before flops probe: current={torch.cuda.memory_allocated()} peak={torch.cuda.max_memory_allocated()}")
# Dispatch-level FLOPs, measured HERE while memory still holds only the model and the
# optimizer state. Counted on the uncompiled module by the frozen probe.
from prepare import measure_flops_dispatch
# Stage-trace the ONE pass that sets the ranked key (c0058, observational).
#
# WHY. peak_vram_bytes IS the flops-probe peak. The trace above proves it: [mem] before flops
# probe reads 558,518,272 and after reads 846,045,696, and nothing later -- not the post-probe
# diag, not the training loop, not the evaluation -- ever raises it again. Round 47 paid to
# establish that lowering the EAGER peak buys nothing: c0047 cut it 83,934,720 and the key went
# UP 573,440, because the bytes it removed were not live at the probe's peak instant. So a card
# that cannot name that instant is not verified, and after 57 launches nothing measures it. The
# _stage() instrument is already in this file and already runs around two other passes; it is
# simply switched off during the only pass that matters.
#
# WHAT IT BUYS. measure_flops_dispatch runs exactly one forward+backward over FLOPS_PROBE_ROWS=8
# sequences (16,384 tokens) under FlopCounterMode. The marks fire in the probe's FORWARD, so
# each one reports how far the running max has climbed at that point. The probe's backward is
# inside prepare.py and out of reach, but the gap between the last forward mark and the [mem]
# after flops probe line gives the backward's contribution exactly. That is the decomposition:
# how much of the 287,527,424 the instrument adds is standing by the end of the forward, and how
# much only appears in the backward.
#
# WHY IT CANNOT MOVE THE KEY. _stage() creates no tensor. It prints two integers read from the
# caching allocator's counters -- memory_allocated() and max_memory_allocated() -- and it never
# calls reset_peak_memory_stats(), which would destroy the harness's own measurement of the very
# number being ranked. Neither counter read is an aten dispatch, so FlopCounterMode sees nothing
# and flops_per_token_measured is unchanged too.
#
# WHY finally. If measure_flops_dispatch raised with the flag still True, every subsequent
# forward in the training loop would print two stage lines per block for 3,100+ steps -- megabytes
# of stdout and real throughput, on the axis where the whole gate margin is worth 1.8% of
# throughput. The finally clause makes that impossible. The flag is restored to its prior value
# rather than hard-set to False, so this cannot silently disable the diag pass below.
_STAGE_TRACE_PREV = _STAGE_TRACE
_STAGE_TRACE = True
try:
    flops_per_token_measured = measure_flops_dispatch(model, x, y)
finally:
    _STAGE_TRACE = _STAGE_TRACE_PREV
print(f"FLOPs per token (measured at dispatch): {flops_per_token_measured:,}")
print(f"[mem] after flops probe: current={torch.cuda.memory_allocated()} peak={torch.cuda.max_memory_allocated()}")

# ---------------------------------------------------------------------------
# Stage profile of the grad path (c0021, observational). Placed HERE, immediately after the
# frozen probe, for four reasons that together make it free:
#
#   1. It cannot touch the ranking key. The probe above has already set the high-water mark at
#      ~1.545e9. This pass runs ONE sequence, an eighth of the probe's 8 x MAX_SEQ_LEN, so its
#      own footprint is a few hundred megabytes against a mark already standing far above it.
#   2. It cannot touch flops_per_token_measured. measure_flops_dispatch has already returned.
#   3. It cannot touch throughput, which after round 19 is the constraint that actually binds:
#      this runs before t_start_training, so it spends setup time, not TIME_BUDGET, and costs
#      zero steps. That matters because the gate margin is worth only about 2.2% of throughput.
#   4. It measures the RIGHT graph. prepare.py takes getattr(model, "_orig_mod", model) and
#      wraps the pass in FlopCounterMode, so the forward-and-backward that sets the key runs
#      UNCOMPILED and eager -- its own comment says inductor would otherwise hide elementwise
#      work from dispatch. Reasoning about the compiled graph produced three consecutive
#      mis-predictions. This pass runs the uncompiled module for the same reason.
#
# c0020 ran this without autocast and died on a dtype mismatch, bf16 activations against fp32
# weights, having produced two of its nine stage lines. Both call sites that matter -- the
# training loop above and prepare.py's probe -- wrap the forward in bf16 autocast and leave the
# backward outside it, so this mirrors that exactly. And because an observational insert must
# never be able to abort the measurement it is observing, the whole thing is inside try/except:
# if it fails again the run prints why and trains anyway, instead of spending a launch to learn
# nothing.
#
# Deltas between consecutive [stage] lines give per-stage retention; divide by the token count
# to get bytes per token, which is the quantity every future card needs and none has had.
try:
    _diag = model._orig_mod if hasattr(model, "_orig_mod") else model
    _dx, _dy = x[:1].contiguous(), y[:1].contiguous()
    print(f"[stage] diag-tokens={_dx.numel()}")
    _STAGE_TRACE = True
    with autocast_ctx:
        _dloss = _diag(_dx, _dy)
    _stage("forward-done")
    _dloss.backward()
    _stage("backward-done")
    _STAGE_TRACE = False
    _diag.zero_grad(set_to_none=True)
    del _dloss, _dx, _dy
    print(f"[stage] diag-end current={torch.cuda.memory_allocated()} peak={torch.cuda.max_memory_allocated()}")
except Exception as _diag_exc:
    _STAGE_TRACE = False
    (model._orig_mod if hasattr(model, "_orig_mod") else model).zero_grad(set_to_none=True)
    print(f"[stage] diag-FAILED {type(_diag_exc).__name__}: {_diag_exc}")

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

# Observational only, as above: the training-loop high-water mark, read before the frozen
# instrument runs the certifying eval. The difference between this and the reported
# peak_vram_bytes says whether the frozen eval or the training loop binds the ranking key.
print(f"[mem] after training loop: current={torch.cuda.memory_allocated()} peak={torch.cuda.max_memory_allocated()}")

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
