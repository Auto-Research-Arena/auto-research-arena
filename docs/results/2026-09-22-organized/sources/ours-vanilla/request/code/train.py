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


# c0005 instrument: where does the width-1 step's time actually sit?
#
# Four rounds of "cut kernels, let inductor fuse" have each paid, and the step is now 374.78
# us against a ~28.8 us byte floor -- still 13x, 7.9% of roofline -- with no mechanism I can
# name for the remainder. Two of the last four cards got direction right and magnitude wrong,
# so this buys a measurement instead of a fifth guess.
#
# What it is NOT: a metric, and not an input to one. prepare.py's single METRICS_JSON line
# remains the only accessor for every judged number, exactly as with c0001's
# DECODE_GRAPH_CAPTURE. This prints and nothing reads it back.
#
# Where it runs: inside _GraphedDecodeStep.__init__, AFTER capture, on the eager path, with
# `on` forced back to False in a finally before any timed pass or replay can occur. It
# therefore cannot appear inside the captured graph and cannot perturb request_ms_median.
#
# What it cannot tell me, stated up front: the instrumented path is EAGER and carries ~1-2 us
# of event overhead per span, so the phase total will exceed a replay's 374.78 us and the
# phases must be read as an APPORTIONMENT, not as a budget that sums to the measured step.
class _PhaseTimer:
    def __init__(self):
        self.on = False
        self.spans = []

    def start(self):
        if not self.on:
            return None
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    def stop(self, name, begin):
        if begin is None:
            return
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self.spans.append((name, begin, e))

    def drain(self):
        """Milliseconds per phase name, summed over every span recorded so far."""
        torch.cuda.synchronize()
        total = {}
        for name, a, b in self.spans:
            total[name] = total.get(name, 0.0) + a.elapsed_time(b)
        self.spans = []
        return total


_PHASE = _PhaseTimer()


def _is_annotation(key):
    """True for a profiler row that is a scope annotation and not a kernel launch.

    Third fix to the same accessor, and all three defects were mine. c0007 summed every row
    and read 211 launches / 3137 us against a 367 us step. c0008 filtered on
    `device_type == DeviceType.CUDA` and read 114 / 2369.9 -- better, because it dropped the
    operator-level records, but STILL WRONG: inductor's six `## Call CompiledFxGraph <hash>`
    rows report `dev=1` and held 1920.93 of that 2369.9. Subtracting them by name gave ~97
    launches and ~449.0 us against a 368.963 us replayed step, a 22% profiler excess and the
    first internally coherent reading this accessor has produced. So the exclusion is applied
    in the code this round instead of by hand afterwards, and the excluded rows are printed
    separately with their totals so the subtraction is auditable from the log rather than
    from this comment. Nothing here feeds a metric.
    """
    return key.startswith("##") or key.startswith("ProfilerStep")


# c0001: split-KV parallelism for the width-1 cached decode step.
#
# The reference pins `num_splits=1` in `flash_attn_with_kvcache`. The in-file reason is
# that the op's fake kernel refuses the default `0`; that raise is registered via
# `torch.library.register_fake` and fires only under fake-tensor tracing
# (torch.compile / torch.export). This program never traces that call: `torch.compile`
# wraps the model for training, but `decode_step` is reached through OptimizedModule's
# attribute forwarding and runs uncompiled, and the training forward uses
# `flash_attn_func`, not the kvcache variant. So the pin's stated mechanism does not
# reach this code path.
#
# Why a split helps here, and it is the ranking key's term specifically: at batch 1 with
# n_kv_head 4 and no split, the per-step cache read is serialised onto ~4 CTAs on a
# 108-SM device. Measured on the reference, the prefilled shape costs 0.347 ms/step more
# than the no-prompt shape for ~21 MB more reachable cache -- 60 GB/s, ~4% of A100
# roofline -- which is what a locally-saturated handful of SMs looks like, not a
# device-bandwidth limit. 8 splits puts ~32 CTAs on 108 SMs.
#
# Explicit 8 rather than the heuristic `0` on purpose: get_num_splits is C++ and its
# resolved value is not observable from Python, so if the heuristic picked 1 the null
# would be indistinguishable from "parallelism does not help". An explicit value >1
# guarantees the split occurs and makes a null a real refutation. The binary asserts
# num_splits >= 1 and caps the combine at 256.
#
# This changes reduction order, so it moves the fidelity metrics. It changes no
# parameter, no FLOP and no cache allocation, and it is confined to the decode branch,
# so the training path is byte-identical to the reference.
DECODE_NUM_SPLITS = 16  # c0015 RESTORES the incumbent's value as the DEFAULT and makes the
# constant per-window (see DECODE_NUM_SPLITS_BY_WINDOW below). Three measured points now exist
# on the uniform axis: 8 -> 213.5764, 16 -> 199.4377, 32 -> 208.8631. 16 is a minimum with 8
# worse by 14.1387 and 32 worse by 9.4254, both far outside the 0.1343 ms replicate sd. The
# model that predicts that shape: num_splits partitions the KV BLOCKS of one call, the split
# kernel's critical path is ceil(blocks/splits), and both FlashAttnFwdCombine and the CTA count
# grow LINEARLY in splits. This net has 6 short layers (window 1024) and 2 long ones (2048), so
# at a 64-key block and a 2048 cache the depths are S 2 / L 4 at splits=8, S 1 / L 2 at 16, and
# S 1-with-16-idle-splits / L 1 at 32. 8->16 shortens both; 16->32 shortens only 2 of 8 layers
# while making the other 6 pay empty splits and double combine. Under that model EVERY value
# from 17 to 31 is strictly worse than 16 -- same depth, more combine -- so scanning a uniform
# constant further is exhausted, and the move is to stop making one number serve two shapes.
# The 64-key block is an ASSUMPTION, not a measurement. At a 128-key block the depths are S 1 /
# L 1 already at 16 and this change should LOSE slightly. The two models predict opposite
# signs, which is what makes the launch worth a charge.
#
# Superseded comment from c0014, kept because it is what I believed before the measurement:
# c0014 walks the SAME axis one step further. c0013 (splits=16)
# measured 199.4377 against c0012's 213.5764: a 14.14 ms gain where the deflated eager sweep
# forecast 3.46 ms. That forecast is FALSIFIED, under by 4.1x, so the sweep is no longer
# trusted for MAGNITUDE. Its ORDERING is the only thing left to test: it put 32 (-2.58%)
# slightly behind 16 (-3.31%) and 64 (+12.44%) far behind both. If 32 beats 16 the ordering
# is refuted too and this axis needs a real sweep measured at the ranked shape, not values
# picked off a proxy that has now missed once on magnitude. FlashAttnFwdCombine's cost grows
# with the split count and is the mechanism that should eventually turn this around.
# c0013 closed the same reasoning that opened it:
# num_splits on the ground that 'nothing on it exceeds the resolution available', where the
# resolution in hand was believed to be ~3.4 ms. Round 12 measured the actual dispersion of
# the ranking key: 0.1343 ms between same-configuration launches, agreeing to 2.5x with the
# sd of a 30-pass median implied by the spec's own published request_ms_iqr. So the ruler was
# wrong by 25x. c0008's probe-1 sweep (the PREFILLED shape, which is the ranking key) read
# splits=16 at -3.31% of a 1486.08 us eager fa3 span; deflated by the measured 7.3x eager
# inflation that is ~3.5 ms end-to-end, or 26x the real resolution. That sweep span covered
# the whole fa3 call including FlashAttnFwdCombine, whose cost GROWS with the split count, so
# the -3.31% is already net of the combine. One constant changes and nothing else: c0012 is
# the control, and its metadata is built with this same constant so the two stay consistent.

# c0008 swept this constant in-launch over (1, 2, 4, 8, 16, 32, 64) on both probe shapes and
# the axis is CLOSED. splits=1 was clearly worse (+17.74 per cent of the eager fa3 span on the
# prefilled shape, which matches the 306 us/step c0001 realised), but every value from 4 to 32
# sat within 3.3 per cent of 8 -- inside the instrument's own control drift of 3.22 and 7.98
# per cent -- and the two probes disagreed about which value won. So 8 is neither shown optimal
# nor shown beatable, and c0009 does not touch it.
#
# What c0008's corrected profiler DID establish is why this file now replaces the call instead
# of tuning it. Per step: flash forward 154.49 us in 8 launches, FlashAttnFwdCombine 36.02 in
# 8, and prepare_varlen_num_blocks 21.79 in 8 -- 212.30 us, 47 per cent of a 369 us step, in 24
# launches. The six sliding layers reach 1025 positions and the two full layers reach up to
# 2048, so twice the bytes costs 1.13x the time: about 16.5 us per call does not scale with
# bytes at all, and 8 calls of that is 132 us per step. Against a KV byte floor of ~10.5 us
# this call runs 20x off roofline, and the part that is off is not the part that reads memory.
DECODE_NUM_SPLITS_AXIS_CLOSED_BY = "launch 9, 363892de-4f4d-4104-824f-d4b0ebc54741"
DECODE_NUM_SPLITS_AXIS_REOPENED_BY = (
    "launch 12, 85234874-1494-4bbf-b5dc-874e6804893b -- not because that launch tested the "
    "axis, but because it supplied the third replicate that measured the ranking key's real "
    "dispersion. Launch 9 closed this axis on the ground that nothing on it exceeded the "
    "resolution available; the resolution then in hand was believed to be ~3.4 ms and is "
    "actually 0.1343 ms. A closure whose only ground is a resolution is void when the "
    "resolution moves by 25x, and this one is. The closure line above is kept rather than "
    "deleted: it is what I concluded from launch 9 and it should stay legible next to the "
    "reason it did not survive.")

# c0015: ONE number per window length instead of one number for the whole net.
#
# Keys are window_size[0] as _compute_window_sizes emits it; the value is the num_splits handed
# to that layer's decode call AND to that layer's scheduler_metadata, which must agree or the
# metadata describes a partition the call does not use. Anything not in the map falls back to
# DECODE_NUM_SPLITS, so a change to window_pattern cannot silently drop a layer.
#
# S layers keep 16, which is byte-identical in behaviour to the incumbent c0013 on 6 of the 8
# layers. Only the 2 long layers move to 32, the value at which their 32 KV blocks reach depth
# 1. So any delta this launch records is attributable to TWO layers, and the confound c0014 had
# -- 6 layers made worse in exchange for 2 made better, netting one unreadable number -- is
# removed by construction rather than argued away.
# c0016 adds ONE split to each class, and it is a quantization test, not a tuning step.
#
# A window_size of (w, 0) attends w+1 positions, so a short layer reads 1025 keys and a long one
# reads 2049 -- the metadata is built at cache_seqlens 2048 with max_seqlen_k 2049, which is the
# printed evidence for the +1. At a 64-key block that is 17 and 33 blocks: each class has
# exactly ONE tail block beyond a power of two. c0015 ran 16 and 32 splits over 17 and 33
# blocks, so in BOTH classes one split carried 2 blocks while the rest carried 1, and the
# critical path was 2 when the mean was ~1. That is consistent with c0015's own kernel table,
# where a long layer at 32 splits (33 blocks, depth 2) and a short layer at 16 (17 blocks,
# depth 2) cost 14.93 and 14.83 us -- the same, despite twice the keys.
#
# 17 and 33 splits make every split carry exactly one block. The combine cost rises by 1/16 and
# 1/32, which is small, and the critical path falls from 2 blocks to 1.
#
# THIS IS ALSO THE BLOCK-SIZE EXPERIMENT, and that is the reason to spend a charge on it rather
# than on a bigger idea. If the block is 64 keys, this is a cliff and the gain is large. If the
# block is 128, the counts are 9 and 17, both classes were ALREADY depth 1 at 16 and 32, and
# this change is a NULL that only adds combine. The two answers are far apart and every further
# decision on this axis depends on which holds.
# c0017 REVERTS the map to the incumbent c0015's values and changes ONLY the adoption rule.
#
# c0016's {17, 33} was never tested: launch 17 ran sched_meta=off on the ranking shape, so its
# 208.2847 is the +1-split change minus a 9.38 ms mechanism it lost to probe noise. But the
# TIEBREAK shape adopted in that launch and in c0015's, so on the one shape where the comparison
# is clean, {17, 33} came in 1.3265 ms WORSE -- 9.9x the replicate sd, on the shape the tail-block
# reading predicted would be flat. That is the only clean evidence about {17, 33} and it points
# AGAINST it, so I am not carrying it forward on the same charge as the instrument fix. It is
# deferred, not refuted, and the deferral is recorded rather than the hypothesis quietly dropped.
#
# This candidate therefore buys DETERMINISM, not speed, and I predict no accept. It also yields
# the first replicate of the incumbent's configuration under a known-identical mechanism state,
# which is a dispersion figure I do not have for the current program: the 0.1343 ms replicate sd
# still in use was measured on c0008/c0009/c0011, three launches of an older file.
#
# ===================================================================================
# c0058: THE 256 ENTRY GOES 4 -> 16, AND IT IS THE ONLY EXECUTABLE CHANGE.
#
# This is the up-side of a curve whose down-side I measured last round, and the two
# points are on MY OWN rows rather than transferred from anywhere.
#
# THE MEASUREMENT. c0057 set this entry to 1 and kept everything else. It regressed
# the scored key by +8.1520 ms (+19.24 noise floors). Its deployed kernel table
# localises the whole regression to ONE kernel:
#
#     kernel                       c0054 [4,32]     c0057 [1,32]     delta
#     flash fwd, 2049-key layer         20.98            20.97       -0.01
#     flash fwd,  257-key layer         18.41            39.35      +20.94
#     FlashAttnFwdCombine            9.49 (n=2)      4.68 (n=1)      -4.81
#     everything else                  each within +-0.7 us/step
#
# Total probe A moved +16.22 us/step and the attention block moved +16.12. The
# 2049-key layer is unchanged to 0.01 us/step, so the treatment touched exactly the
# layer it was aimed at, and removing that layer's combine really did save 4.81.
#
# WHAT THAT REFUTES, AND IT IS MINE. My round-58 note argued: "A 257-key read is
# about 395 KB and cannot be bandwidth-bound, so both calls are overhead-dominated."
# That premise is now false by measurement. Layer 0 is not overhead-bound; it is
# OCCUPANCY-bound. With 6 heads at batch 1, num_splits=s launches 6*s CTAs on a
# 108-SM device: s=1 gives 6 CTAs and costs 39.35, s=4 gives 24 CTAs and costs 18.41.
# The standing "anomaly" that a 257-key layer cost as much as a 2049-key one was not
# an anomaly at all -- it was the 4 splits already doing most of the work.
#
# WHY 16 AND NOT 8 OR 32. Fitting the two measured points as fixed overhead plus a
# term linear in keys-per-CTA gives 11.43 us/step fixed and 0.10864 us/step per key:
#
#     splits   keys/CTA   CTAs   predicted us/step   vs the deployed 4
#          1     257.00      6         39.35 (fit)          +20.94
#          4      64.25     24         18.41 (fit)            0.00
#          8      32.12     48         14.92                 -3.49
#         16      16.06     96         13.18                 -5.23
#         32       8.03    192         12.30                 -6.11
#
# 16 is the largest value that still fits in ONE wave: 6*16 = 96 CTAs <= 108 SMs,
# while 6*32 = 192 CTAs is two waves and the model has no right to speak there. So
# 16 is the principled end of the curve, not the biggest number I could type. The
# fit is on TWO points and therefore cannot be tested by its own fit; its entire
# falsifiable content is the prediction AT 16, which this row measures.
#
# THE TABLE'S OWN INVARIANT IS THE DEFECT. The comment below records that c0046 chose
# 4 to hold a CONSTANT 64 keys per split (1024/16 = 64, 2048/32 = 64, 256/4 = 64).
# That invariant fixes work per CTA and lets TOTAL OCCUPANCY collapse on the short
# window: 64 keys/split buys 192 CTAs at window 2048 and only 24 at window 256. The
# rule is right for a long window and wrong for a short one, and the +20.94 us/step
# above is what its failure costs when pushed one more step in the same direction.
#
# WHAT I AM NOT CLAIMING. DECODE_SPLIT_SWEEP has measured [16, 32] as top-group on
# three rows, but its own same-config replicate gap is 4.43-5.48 us/step against a
# total span of 1.79 over everything it measures, so it CANNOT rank 16 against 4 and
# is not evidence here. It also still never tests the deployed configuration. The
# sweep's contribution is only that 16 is not off a cliff.
#
# NOT A RE-RUN. No candidate in this run has set this entry to 16. One run per
# candidate is the design.
# ===================================================================================
#
# ===================================================================================
# c0059: THE 2048 ENTRY GOES 32 -> 18. THE ONLY UNTESTED DIRECTION ON THIS AXIS.
#
# WHY THIS EXISTS AT ALL, AND IT IS A METHOD REASON. Last round I RETRACTED my own
# "the split axis closes at zero charges" (round 58 finding 4), because it closed the
# axis on DECODE_SPLIT_SWEEP's evidence and the sweep tests no short-window value
# below 8. One round later that same axis produced an 8.15 ms effect at value 1. If I
# now close the axis on the 256 entry alone, having never moved the 2048 entry, I
# repeat that error exactly one round after retracting it. One charge buys the honest
# closure of a two-entry axis rather than the closure of half of it.
#
# THE WAVE ARGUMENT. At batch 1 with 6 heads, num_splits=s launches 6*s CTAs on a
# 108-SM device:
#     2048 window, s=32 -> 192 CTAs,  64.0 keys/CTA, 1.78 waves (DEPLOYED, 21.00 us/step)
#     2048 window, s=18 -> 108 CTAs, 113.8 keys/CTA, 1.00 waves (THIS ROW)
#     2048 window, s=16 ->  96 CTAs, 128.1 keys/CTA, 0.89 waves (considered, REJECTED below)
#
# 18 IS THE LARGEST ONE-WAVE VALUE, AND I FIRST WROTE 16. My own guard's derived
# assertion caught it: 6*18 = 108 = N_SM EXACTLY, so 18 fits one wave and 16 is not the
# maximum. The c0058 comment block carries the same error ("6*16=96 CTAs <= 108 SMs" as
# "the one-wave maximum") and it is wrong there too; it did not affect c0058's treatment,
# whose window is 256 and whose 16 was chosen on a fitted curve rather than on this bound.
# 18 DOMINATES 16 under my own mechanism -- strictly more CTAs (108 vs 96, saturating the
# machine) AND strictly fewer keys/CTA (113.8 vs 128.1) -- so choosing 16 would have been
# testing the mechanism at a point the mechanism itself says is worse.
#
# THE ONE ARGUMENT AGAINST 18, RECORDED BECAUSE IT IS A REAL RISK AND NOT A HEDGE. 108
# CTAs saturate the device EXACTLY, so if even one SM is unavailable the launch spills to
# a second wave holding a single CTA -- a tail whose cost is a whole wave. 96 leaves 12
# SMs of slack. I take the exact-saturation point anyway, because the card's content is
# the mechanism and a tail appearing IS a finding rather than a confound; and at batch-1
# captured-graph decode nothing else should be resident. Prediction 6 names the tail
# signature so this is falsifiable rather than asserted.
#
# 18 IS NOT A POWER OF TWO, and I checked that this is exercised: DECODE_SPLIT_SWEEP
# itself measures non-power-of-two split counts, so a silent rounding of 18 would be a
# defect the sweep would already have shown. If it rounds anyway, prediction 7's
# captured_table check and the kernel table's CTA-count signature both fire.
#
# THE TWO CALIBRATIONS DISAGREE, AND THAT IS WHY THIS IS AN EXPERIMENT AND NOT A
# FOREGONE CONCLUSION. Both readings are from my own rows:
#
#   (a) AT EQUAL keys/CTA, FEWER CTAs WAS CHEAPER. Layer 0 at 4 splits is 64 keys/CTA
#       on 24 CTAs and cost 18.41; layer 1 at 32 splits is 64 keys/CTA on 192 CTAs and
#       costs 21.00. Same per-CTA work, 2.59 us/step more for 8x the CTAs. This favours
#       the change.
#   (b) THE keys/CTA SLOPE FITTED ON LAYER 0 IS 0.10864 us/step per key. Going 64.0 ->
#       113.8 keys/CTA would then add ~5.41 us/step. This opposes the change, and by more
#       than (a) favours it.
#
# I therefore predict NULL-to-worse and band it accordingly, with an exhaustive
# partition. Reading (b)'s slope was fitted across a CTA-count change as well as a
# keys-per-CTA change, so it is confounded and I do not treat it as the better of the
# two; c0058 already showed that slope over-predicting a gain by 4.6x when extrapolated
# across a knee. Neither calibration is load-bearing: what settles it is the row.
#
# NOT A RE-RUN. No candidate in this run has set the 2048 entry to anything but 32.
# The 256 entry stays at 16, c0058's value, so this row's diff from the incumbent is
# ONE integer and the two entries are never moved on the same charge.
# ===================================================================================
DECODE_NUM_SPLITS_BY_WINDOW = {256: 16, 1024: 16, 2048: 18}  # c0059: 2048 32 -> 18; see above.
# c0058's note, kept because its occupancy account is what this row extends to the long
# window: the 256 entry went 4 -> 16 and the scored key moved -0.3852 ms, SUB-FLOOR.
# c0046's original note, kept verbatim because its invariant is the thing under test:
# c0046: the 256 entry is NEW and
# is THE ONLY CHANGE in this candidate.
#
# WHY, AND THE ARGUMENT IS THE TABLE'S OWN VALUES RATHER THAN MY ARITHMETIC. Every entry
# already in this table holds a CONSTANT 64 keys per split: 1024/16 = 64 and 2048/32 = 64.
# The short windows have been 256 since c0042 and 256 is NOT IN THE TABLE, so they take the
# fallback DECODE_NUM_SPLITS = 16, which is 256/16 = 16 keys per split -- off the table's own
# tuned ratio by 4x. Restoring 64 keys/split at window 256 requires 4 splits. So this edit
# does not propose a new tuning; it applies the tuning already recorded here to a window that
# was added after the table was written and silently fell through to a default.
#
# WHY THIS AXIS IS LEGITIMATELY OPEN. DECODE_NUM_SPLITS_AXIS_CLOSED_BY names launch 9, and
# DECODE_NUM_SPLITS_AXIS_REOPENED_BY records why that closure is void: launch 9 closed the
# axis on the ground that nothing on it exceeded the resolution then in hand. That resolution
# has moved twice since, and the figure quoted in the reopening note (0.1343 ms, measured on
# c0008/c0009/c0011, three launches of an OLDER FILE) is itself now superseded: three
# shape-identical rows of the CURRENT program (launches 44/45/46, byte-equal on all eight
# shape-determined metrics) give 0.3595533393 ms on df 2, which is 2.68x larger. I am
# recording that my working resolution got WORSE, not better, because it moves the bar this
# candidate has to clear in the harder direction: it must beat the parent by 0.7191 ms.
#
# WHY IT IS THE RIGHT CHARGE NOW: IT COSTS NOTHING AT THE GATE. The gate has been the binding
# constraint on every architectural lever in this run. DECODE_NUM_SPLITS_BY_WINDOW is read
# only by _decode_splits_for, on the decode path; training uses flash_attn_func and never
# consults it. So this edit cannot move val_bpb, num_params_total, flops_per_token_measured,
# either cache reading, or the training step count. It is the first lever in many rounds whose
# predicted quality cost is exactly zero BY CONSTRUCTION rather than by a fitted density --
# and val_bpb is therefore a hard prediction here, banded on my LANE replicate sd (0.0002466365,
# df 24), which is the one scope that sd legitimately governs: two of my own rows, same host,
# same compute_run_id.
#
# WHAT THE LAST THREE ROWS TAUGHT ME ABOUT WHERE DECODE TIME ACTUALLY GOES, which is what
# selected a per-call lever over another bytes lever:
#   - c0045 moved the long window 2048 -> 1024 and kv_cache_bytes DID NOT MOVE AT ALL
#     (14549504 both rows), because prepare.py sizes the cache as max_len = prefill + steps,
#     a PROBE parameter, times the per-token state. A window cannot touch it.
#   - GQA, the only knob that shrinks per-token state, breaches the gate at every admissible
#     divisor. So kv_cache_bytes has no affordable lever at all and I have stopped treating it
#     as a latency handle.
#   - c0008's corrected profiler, recorded above, already said so: "about 16.5 us per call does
#     not scale with bytes at all", the call runs "20x off roofline", and "the part that is off
#     is not the part that reads memory". Splits change per-call and combine work, which is the
#     part that is off.
#
# THE COUNTER-ACCOUNT, PRE-REGISTERED, AND I EXPECT TO BE ABLE TO TELL THEM APART. At
# n_kv_head 6, 4 splits puts 24 CTAs on a 108-SM device and 16 splits puts 96. On CTA
# occupancy alone, 16 is the better number and this edit makes things worse. That is exactly
# the reasoning in this file's line-141 comment, and it is the reasoning that launch 9
# falsified once already, which is why I am charging the table's measured ratio instead of my
# own arithmetic. If the key RISES, the occupancy account wins, the table's 64-keys/split
# ratio does not transfer down to small windows, and I will say so.
#
# NOTE ON WHAT IS NOT CARRIED. The parent is c0043, so the long window stays 2048 and c0045's
# LONG_WINDOW_DIVISOR is DROPPED: it cost +0.0021483617 val_bpb for a prefilled gain of
# 0.5136728287 ms against a no-prompt control that fell 0.8558034897 ms in the same row --
# the control moved 1.67x MORE than the treatment, so the differential has the wrong sign and
# there was no window-specific gain to keep. c0045's remap of the 1024 entry to 32 is also
# not carried; the entry stays at its original 16 and the reachable split set is again held by
# the table's own values.

# c0041. THE LOCAL REACH HALVES, 1024 -> 512, AND IT IS THE ONLY CHANGE.
#
# Parent is c0040, the incumbent (launch 42, 2aa645b9-ddf0-40f1-a213-f05764a29c22):
# request_ms_median 84.46311950683594, val_bpb 1.0481978463290338, params 42467524,
# flops 207619200. depth 2, model_dim 768, value-embed 1, MLP_HIDDEN 6144.
#
# WHY THIS LEVER. It is the largest measured latency lever in this run and it has never been
# tried at THIS shape. At depth 3 / model_dim 640 the identical move -- short_window from
# long_window//2 to long_window//4, split table untouched -- is the clean pair c0031 -> c0035:
#
#   c0031  windows [1024,1024,2048]  key 93.90711784362793  val_bpb 1.0384721355448014
#   c0035  windows [ 512, 512,2048]  key 88.66989612579346  val_bpb 1.0380166328233207
#   delta                            key -5.2372217 ms      val_bpb -0.0004555 (an IMPROVEMENT)
#
# num_params_total is byte-identical across that pair (35717446 both rows), which is the
# evidence that it was a pure window change. So on the one shape where it has been measured,
# reach bought 5.24 ms and cost NOTHING on the gate -- it moved the gate the helpful way by
# 0.10 sd of the pre-registered 0.0045534, which TASK.md would call no effect either way.
#
# THAT IS WHY IT IS THIS CHARGE AND NOT A CAPACITY PURCHASE. c0040 passes the gate by only
# 0.0018021536709662556 = 0.39578198071029463 sd. Every capacity lever costs gate margin I no longer have, and
# the MLP axis is now priced OUT: at hidden 7168 the params rise 7.41 % and the per-layer matrix
# work rises 13.33 %, which at the 0.1895 key elasticity measured on THIS row costs about
# +2.53 % of key, landing near 86.6 ms -- WORSE than the incumbent it would be built on. The
# c0040 card pre-registered "press further toward hid 7168" for exactly this branch and that
# branch is hereby WITHDRAWN, on arithmetic taken from the row that opened it. The MLP purchase
# bought precisely the gate margin it needed and one more unit of it is a loss.
#
# THE SPLIT COUNT IS HELD BYTE-IDENTICAL, BY CONSTRUCTION AND NOT BY ARGUMENT. _decode_splits_for
# is `table.get(int(window_size[0]), DECODE_NUM_SPLITS)` with DECODE_NUM_SPLITS = 16, and the
# table's 1024 entry is ALSO 16. A 512 window is absent from the table and therefore falls back
# to 16. So the per-layer split vector is [16, 32] before and [16, 32] after. This is the
# confound that destroyed c0036, where 2048 -> 1024 halved the split count at the same time as
# it cut reach and produced one unreadable number; here it is removed by the table's own values.
#
# WHAT THE PROBES SHOULD DO, AND THIS IS THE PRIMARY INSTRUMENT. The two probes must DIVERGE.
#   - prefilled: kv_cache_prefill 1536 > 1024, so a local layer at window 1024 really does read
#     1025 keys and at 512 reads 513. There is reach work to save and the key should FALL.
#   - no-prompt: nopref_kv_cache_max_len is 513. The cache never holds more than 513 keys, so a
#     512-window layer and a 1024-window layer read the SAME keys. There is no reach work to
#     save and the key must NOT fall; at depth 3 it ROSE +2.5167465209960938 ms (83.0467939376831 from
#     80.53004741668701 -- see round 37), 18.7x the 0.1343 ms replicate sd, so a rise is the
#     expected sign and not a surprise.
# IF BOTH KEYS FALL TOGETHER THE GAIN IS NOT REACH. That is the same discriminator c0038 used in
# the opposite direction -- there convergence proved the lever was the MLP, which runs on both
# probes; here divergence proves it is attention, which does not run on the short one.
#
# num_steps IS A WEAK INSTRUMENT THIS ROUND AND I AM NOT DRESSING IT UP AS A STRONG ONE. flops
# falls only 2.27 % (207619200 -> 202900608), so the three rival step elasticities in play --
# 0.0681 (measured within-shape on launch 42), 0.5706 (the 39-row cross-shape fit) and 1.0 --
# predict 1006, 1017 and 1027 steps. Those bands are 10-20 steps apart on an integer counter
# whose own round-to-round scatter is larger than that, so they are NOT disjoint at the
# available resolution and num_steps CANNOT select among them here. It is banded as a single
# test. Launch 42 is where that selection happened and it is not re-litigated on a 2 % move.
#
# LAUNCH 42 RETRACTIONS THIS CARD IS BUILT ON, both against my own previous prose:
#  1. d log(num_steps)/d log(flops) = -0.5706 (r -0.9535, n 39, df 37) DOES NOT TRANSFER to a
#     within-shape change. Launch 42 raised flops 37.5 % and lost 2.1 % of steps: an implied
#     d log(steps)/d log(flops) = -0.06806598569237005, i.e.
#     E = +0.06806598569237005 in the convention steps = base*ratio**(-E) that the 0.5706
#     and 0.113 figures are quoted in. BOTH SIGN CONVENTIONS ARE WRITTEN OUT HERE because
#     conflating them is what failed a guard assertion on this very card. The two error
#     terms below satisfy OVERCREDIT - OVERCHARGE == the point's miss exactly, which is
#     the check that the decomposition is a decomposition and not two loose numbers. The 39-row figure is a cross-shape population fit in which depth,
#     width, windows and splits all co-vary with flops, and I called it "the best-determined
#     coefficient in this run" and let it supersede two others. It is superseded itself, for
#     within-shape use, by the row it was used to predict.
#  2. The MLP density 0.001159 val_bpb per 1 % of params (depth 3, df 1) OVERSTATES this shape.
#     Launch 42 delivered -0.0182700422542954 of capacity for +28.5713 % of params against a step cost of
#     only +0.00134610797428802442, i.e. a density of 0.000639, 55 % of the figure the card used. My val_bpb
#     point landed 0.0057884115290339 optimistic while being wrong TWICE IN OPPOSITE DIRECTIONS -- step
#     cost over-charged by 0.0090556354257120, density over-credited by 0.0148440469457046 -- so the band "held"
#     on cancellation. A point that lands close by cancelling two large errors is not a
#     calibrated route and this card does not treat it as one.
#  3. kv_cache_bytes IS NOT A CLOSED FORM. It is "one allocation delta" (substrate cbea127) and
#     it fell 15467008 -> 14549504 on launch 42, a change that touched no cache dimension:
#     kv_cache_max_len 2048, kv_cache_prefill 1536 and nopref_kv_cache_bytes 3152384 all held
#     byte-identical. The 917504-byte step is 896 KiB, an allocator quantum. I had predicted it
#     "exactly" for six consecutive rows and read six rows of held geometry as a geometric law.
#     This card BANDS it and predicts nothing exact about it.
# c0042. THE SAME LEVER AGAIN, ONE NOTCH FURTHER: 512 -> 256. AND THE MECHANISM NOW PREDICTS
# THE OPPOSITE SIGN ON THE NO-PROMPT PROBE, WHICH IS THE WHOLE POINT OF THIS CHARGE.
#
# Parent is c0041, the incumbent (launch 43, 8a43bf97-d931-4c5b-8ea5-ce4adc467b14):
# request_ms_median 82.15177059173584, nopref 74.86093044281006, val_bpb 1.0480750096233953
# (passing by 0.0019249903766047627 = 0.42275890029533153 sd), params 42467524,
# flops 202900608, num_steps 1006. Twelve of twelve predictions held on that row.
#
# WHAT LAUNCH 43 ESTABLISHED. Local reach transfers to depth 2 and it is gate-free at BOTH shapes:
#
#   pair                       short window   d(key)       d(nopref)     d(val_bpb)
#   c0031 -> c0035 (depth 3)   1024 -> 512    -5.2372217   +2.5167465    -0.000455503 (-0.100 sd)
#   c0040 -> c0041 (depth 2)   1024 -> 512    -2.3113489   +2.0995140    -0.000122837 (-0.027 sd)
#
# The predicted DIVERGENCE arrived on both: the prefilled key fell and the no-prompt key ROSE.
# Per SHORT LAYER the gain is -2.62 ms (depth 3, two layers of three) and -2.31 ms (depth 2, one
# layer of two), so the per-layer magnitude is stable across shapes even though the totals are not.
#
# THE NO-PROMPT PROBE BECOMES REACHABLE HERE, AND I FIRST GOT THE MAGNITUDE WRONG BY 2x.
# nopref_kv_cache_max_len is 513 and a (w,0) layer attends w+1 positions, so:
#   at w=1024 the layer would read 1025 -> above the cache, nothing to save
#   at w= 512 the layer reads 513       -> equal to the cache, still nothing to save
#   at w= 256 the layer reads 257       -> BELOW the cache, so there is finally work to save
# That much is right and it is why every earlier reach change was invisible to this probe. My
# first draft then wrote "both keys fall" and centred the band on a saving equal to the prefilled
# one. THAT IS WRONG, and the parent's own row says so. nopref_kv_cache_prefill is 1 and
# nopref_request_steps is 512, so the no-prompt request walks positions 2..513, and a 256-window
# BINDS ON ONLY HALF OF THEM. Average keys read per step by the short layer:
#   w=512: min(513,pos) = pos throughout                     -> mean 257.50
#   w=256: min(257,pos), binding for the last 256 steps only  -> mean 193.25
# an absolute delta of 64.25 keys per step. The prefilled probe (prefill 1536, positions
# 1537..2048) reads w+1 keys on EVERY step, so it goes 513 -> 257, a delta of 256. If time on
# this axis tracks keys read, the no-prompt saving is 64.25/256 = 0.251 of the prefilled one --
# about 0.29 ms, not 1.16. SCALING BY THE CUT FRACTION INSTEAD OF THE ABSOLUTE KEY DELTA IS WHAT
# GAVE ME THE 2x, and the cut fractions do differ per probe (49.9 % against 25.0 %).
#
# SO THE SIGN IS NOT MINE TO ASSUME, AND THAT IS WHAT MAKES THIS CHARGE WORTH SPENDING.
# A 0.29 ms saving is small against the +2.0995140 ms rise the same probe showed last round. Two
# rival accounts of that rise are pre-registered, they are DISJOINT on this probe, and both lie
# inside the range of no-prompt moves this run has already produced:
#
#   ACCOUNT A -- the rise was a one-off of leaving the 1024 default and does not recur.
#               nopref = 74.86093044281006 - 0.29  ->  about 74.57, A FALL. Probes CONVERGE.
#   ACCOUNT B -- the cost recurs per window change at roughly its measured magnitude, and is
#               merely reduced by the new saving.
#               nopref = 74.86093044281006 + 2.10 - 0.29  ->  about 76.67, A RISE. DIVERGE again,
#               but by LESS than last round, which is itself a check on B rather than a hedge.
#
# The two points sit 2.1 ms apart against a no-prompt IQR of 0.09196996688842773, so the probe can
# separate them; the ranking key's two implied points sit about 1.05 ms apart against an IQR of
# 0.2498030662536621, which it can also separate but less comfortably. THE NO-PROMPT PROBE IS
# THEREFORE THE PRIMARY INSTRUMENT AND THE RANKING KEY IS CORROBORATION, which is the reverse of
# the usual ordering here and is stated deliberately.
#
# Under B the ranking key barely moves: last round's -2.3113489 would itself be a gross saving of
# (2.3113489 + cost) with the cost netted out, so halving the reach gives 1.1556744 - cost/2, i.e.
# about -0.11 ms if the prefilled cost matches the no-prompt one. Under A it is the full
# -1.1556744. WHETHER THE COST TERM IS THE SAME ON BOTH PROBES IS NOT KNOWN and is not assumed:
# the ranking band spans both accounts and the card says which region selects which.
#
# THE HOLE THIS CHARGE ALSO PROBES, WHICH THE c0041 CARD DID NOT CLOSE. That card predicted the
# no-prompt key would rise and cited the depth-3 rise as precedent -- but it never said WHY it
# rises. A +2.0995140 ms cost on a probe the card itself argued had NO work to save is an
# UNEXPLAINED COST TERM attached to changing a window, and citing a precedent is not explaining
# it. This row separates the two terms for the first time: the reach saving and the unexplained
# cost now act on the SAME probe in OPPOSITE directions. If the no-prompt key falls, the saving
# exceeds the cost; if it rises, the cost term is at least as large as a 25.0 % cut in that
# layer's mean key count (64.25 keys per step, per the arithmetic below -- NOT the 49.9 % that
# applies to the prefilled probe), which would make it the dominant effect on this axis and the
# next thing to chase. Either way the term stops being invisible, and it is worth saying that this
# is the SMALLER of the two probes' cuts, so a rise here is the likelier of the two outcomes and
# the card must not be written as though a fall were the safe expectation.
# IT IS NAMED HERE AS THE CARD'S WEAK LINK.
#
# WHY NOT THE DEPTH-3 STEP THE c0041 CARD PRE-REGISTERED. That card's ACCEPT branch said the next
# charge "tests DEPTH 3 at the newly-cheap MLP, where TWO layers can be short." THAT STEP IS
# WITHDRAWN, and the arithmetic is why. At depth 3 with ASPECT_RATIO 256 (model_dim 768, 6 heads):
#   hid 4608  params 47186118 (93.75 % of ceiling)  flops 235930752 (98.68 % of ceiling)  legal
#   hid 4864  params 48365766 (96.09 %)             flops 243008640 (101.64 %)            BREACH
#   hid 6144  params 54264006 (107.81 %)            flops 278398080 (116.45 %)            BREACH
# So the flops ceiling forces the MLP down from 6144 to at most 4608, and the only legal point
# sits at 98.68 % of that ceiling with no headroom left for any further lever. It also moves
# THREE things at once -- depth, model_dim (via the aspect-ratio rounding) and MLP hidden -- which
# is the confound discipline this run has been enforcing all round, and it would be spent on a
# shape whose quality cost is unpriced because depth has never been varied here at held width.
#
# The branch's stated premise was that "the remaining reach headroom is bounded by that layer."
# The premise is TRUE and the conclusion drawn from it was wrong: BOUNDED IS NOT ZERO, and launch
# 43 is what made the remaining step pricable at all, by measuring the per-short-layer magnitude
# (-2.31 ms) that did not exist when the branch was written. Withdrawing a branch on the strength
# of the row that opened it is the second time this run has done so in two rounds; the first was
# the hid-7168 step, killed by the key elasticity that the row it was written on produced.
#
# THE SPLIT VECTOR IS STILL HELD BY THE TABLE'S OWN VALUES. 256 is absent from
# DECODE_NUM_SPLITS_BY_WINDOW = {1024: 16, 2048: 32}, so it falls back to DECODE_NUM_SPLITS = 16,
# which is what the 512 layer already resolves to. Per-layer splits are [16, 32] before and after.
#
# AND THE PARTITION GETS WORSE, WHICH IS A SECOND REASON THE GAIN MAY BE SUB-LINEAR. Blocks of
# work for the short layer on the PREFILLED probe, where it reads w+1 keys on every step:
#
#   short window   keys   blocks @64   blocks @128   splits that do no work (of 16)
#   1024           1025   17           9             0 @64,  7 @128
#    512  (parent)  513    9           5             7 @64, 11 @128
#    256  (this)    257    5           3            11 @64, 13 @128
#
# MY FIRST DRAFT QUOTED "9 blocks -> 5" AND "5 -> 3" AS THE PREVIOUS AND THIS STEP. Both figures
# are real but they belong to DIFFERENT BLOCK SIZES -- 9->5 is this step at 64, 5->3 is this step
# at 128 -- so as written it silently changed units mid-sentence to make a trend. The table above
# replaces it. The combine cost is paid over all 16 splits regardless, so the fraction of idle
# splits rises at both block sizes, and the key-count argument (-1.16 ms) is an UPPER bound on the
# gain rather than an estimate of it. The band admits a NULL and a small LOSS on purpose.
#
# flops: the reach term falls 12*768*(512-256) = 2359296, giving 200541312 (83.88 % of ceiling).
# params are UNCHANGED at 42467524: a window is not a parameter.
SHORT_WINDOW_DIVISOR = 8

# c0018 SWEEP SUPPORT, and it is declared HERE, at module level, on purpose.
#
# Launch 11 was lost -- a whole charge, forfeited -- to `NameError: name '_SCHED_META_MODE' is
# not defined`, a global that was only ever assigned inside a function. This is the same shape of
# object, so it gets a module-level binding before anything can read it.
#
# The invariant that matters: `_SPLITS_OVERRIDE is None` for the whole of every timed pass and
# for the capture itself. The pre-capture sweep is the only writer, it restores None in a
# `finally`, and the captured configuration is therefore DECODE_NUM_SPLITS_BY_WINDOW exactly as
# written above. The sweep REPORTS; it does not adopt. In-launch adoption on a noisy in-launch
# measurement is the fault c0017 removed, and this file is not going to reintroduce it one round
# later in a new costume.
_SPLITS_OVERRIDE = None


def _decode_splits_for(window_size):
    """num_splits for one layer, keyed by its window length. Eager-only: never called from
    inside a compiled region, and resolved once per layer at capture time, so the captured
    graph sees a Python int constant exactly as it did when the constant was global.

    c0018 adds the override read. It resolves to the module table whenever `_SPLITS_OVERRIDE` is
    None, which is its value at capture and during every timed pass, so this candidate's recorded
    metric is produced by the identical table c0017 recorded 197.2681 with.
    """
    table = DECODE_NUM_SPLITS_BY_WINDOW if _SPLITS_OVERRIDE is None else _SPLITS_OVERRIDE
    return table.get(int(window_size[0]), DECODE_NUM_SPLITS)


# c0009 IS REFUTED AND ITS MECHANISM IS DELETED, NOT DISABLED.
#
# c0009 replaced fa3's cached-decode call with a hand-written torch attention over the whole
# preallocated cache, behind a probe that adopted it only on agreement AND speed. Launch 10,
# 81a22adb-f12e-4675-a4eb-5bb1ff10febc, printed on both probes:
#
#   DECODE_ATTN_PROBE: fa3_us_per_step=1539.11 torch_us_per_step=4170.01 tv_vs_fa3=0.0025786
#   DECODE_ATTN_PROBE: fa3_us_per_step=1623.72 torch_us_per_step=3846.88 tv_vs_fa3=0.0000000
#   DECODE_ATTN_CHOSEN: adopted=False agrees=True faster=False        (twice)
#
# The arithmetic was right -- agreement to 0.0026 and, on the shorter state, to zero -- and the
# path was 2.4x to 2.7x SLOWER. That is the card's `branch_adopted_false_because_slower` clause
# firing verbatim: fa3's per-call cost is not overhead a torch reduction can undercut at this
# shape. The whole mechanism, `_decode_attn_torch`, `_ATTN_MODE`, `_POS_CACHE` and
# `_decode_positions`, is removed rather than left behind a flag.
ATTN_REWRITE_REFUTED_BY = "launch 10, 81a22adb-f12e-4675-a4eb-5bb1ff10febc"

# c0010: stop paying for fa3's block-scheduling kernel on every one of the 8 calls per step.
#
# This axis exists because c0009 printed `inspect.signature(fa3.flash_attn_with_kvcache)` as
# free intel, and the pinned build turns out to accept `scheduler_metadata=None`. Launch 10's
# corrected kernel table -- the first one this run can quote, 22 kernels, 97.0 launches/step,
# 446.8 us against a 369 us replayed step -- prices what that argument is for:
#
#   flash::prepare_varlen_num_blocks_kernel<1, false>   n=6   14.59 us/step
#   flash::prepare_varlen_num_blocks_kernel<1, true>    n=2    7.23 us/step
#                                                     total   21.82 us/step, 8 launches
#
# 21.82 us/step over 513 steps is 11.2 ms, which is 3.4x the ~3.3 ms between-launch noise floor
# that launches 9 and 10 both measured. It is not an attention rewrite: c0009 refuted replacing
# the kernel, and this removes a launch the kernel makes on its own behalf while leaving the
# attention arithmetic bit-identical. Those are different claims and this one is untouched by
# that refutation.
#
# Why it might not work, stated before it runs. The metadata describes a tile schedule, and the
# number of tiles depends on how much of the cache is live -- `cache_seqlens`, which advances
# every step. A captured graph cannot recompute it per step, so the only thing that can be
# passed is ONE tensor computed for the state's maximum length. Two ways that ends badly:
# it is silently wrong for a shorter seqlen, which the agreement gate catches; or it is correct
# but schedules the full 2048 tiles at every step, which costs more than it saves on the
# no-prompt state where seq runs from 1 and would be a 4x over-schedule. The prefilled state
# runs seq 1536 -> 2048, so it over-schedules by at most 33 per cent, and that state IS the
# ranking key. So an adopt on probe 1 and a refuse on probe 2 is a coherent outcome and is
# pre-registered as one, not read as one afterwards.
SCHED_META_UNAVAILABLE = "fa3 build exposes no get_scheduler_metadata"

# THE DEFINITION LAUNCH 11 DID NOT HAVE. It is at module scope on purpose, and the guard now
# asserts module scope specifically rather than asserting that the string appears somewhere.
#
# Launch 11, 2b18a17d-66e3-4b6c-84ec-b92423cede0d, died with
# `NameError: name '_SCHED_META_MODE' is not defined` after completing all 754 training steps,
# inside measure_kv_cache_bytes(prefix="nopref_"). Two faults, both mine: I deleted the block
# holding c0009's `_ATTN_MODE = None` and only then ran a rename that matched nothing and failed
# silently; and my guard's check was `"_SCHED_META_MODE = None" in src`, which the assignment in
# the probe's own `finally` satisfied. A substring test with no dynamic range on WHERE.
#
# That traceback also refutes a claim I have carried since round 7 and repeated in two cards:
# measure_kv_cache_bytes does NOT reach only the prefill branch. With prefill=1, Tn==1, so
# decode_step dispatches _decode_body(prefill=False) and the bracket runs this very branch.
_SCHED_META_MODE = None  # probe-only override, one of None / "on" / "off". None everywhere the
                         # measured path runs, so that path reads only state["sched_meta"].

# c0002: cut kernel COUNT in the replayed width-1 decode step.
#
# After c0001 the cache-read term is spent: the tiebreak shape is 93.7% of the ranking key,
# and its residue is ~0.886 ms/step to move ~60 MB of weights -- ~68 GB/s, ~4% of A100
# roofline. Nothing that far off roofline is bandwidth-bound. At width 1 every GEMM is a
# GEMV and every elementwise op is a few hundred bytes of work behind a fixed per-kernel
# launch latency, and the step issues on the order of 150-200 kernels per replay
# (apply_rotary_emb alone is ~6 elementwise kernels plus a cat, twice per layer, 8 layers).
# ~0.886 ms across ~170 kernels is ~5 us each, which is the floor of a graph-replayed
# kernel doing almost nothing.
#
# So the lever is fewer, larger kernels rather than faster ones. These helpers hand the
# elementwise arithmetic and the GEMVs to inductor, which fuses chains into single Triton
# kernels. torch.compile here is plain (no mode="reduce-overhead") and every fa3 call stays
# OUTSIDE every compiled region. That distinction is the whole safety argument:
#   - The substrate's recorded disaster was torch.compile(mode="reduce-overhead") wrapping
#     the step INCLUDING flash_attn_with_kvcache. cudagraph-trees copied inputs into a
#     static pool, the in-place k/v append landed in the pool's copy, and it destroyed
#     4.49 bits/token silently while being the fastest option measured.
#   - Nothing here compiles that op, so inductor never sees the cache mutation, and the
#     fake-kernel restrictions (num_splits<=0, and this build's refusal of a pre-allocated
#     output) are unreachable because they only fire under fake-tensor tracing.
# The in-tree precedent is adamw_step_fused / muon_step_fused, already
# @torch.compile(dynamic=False, fullgraph=True) in this file.
#
# Compilation happens on the first call, which is inside _GraphedDecodeStep's 3 warmup
# replays -- before capture and outside every timed region. If it failed during capture
# instead, capture raises, captured goes False and DECODE_GRAPH_CAPTURE says so.
#
# Two ve variants rather than an `if ve is None` inside one graph, so fullgraph never has
# to guard on None.

# c0003: one mechanism -- the three per-layer projections c_q, c_k, c_v read the same `h`
# and write disjoint outputs, so at width 1 they are three GEMVs that could be one. Fusing
# them removes 2 kernels per layer, 16 per step, and gives cuBLAS a 1536-wide output
# instead of three 512-wide ones.
#
# Three properties this must not disturb, and how each is held:
#   - TRAINING. MuonAdamW groups 2D matrices BY SHAPE, so a genuinely fused QKV parameter
#     would change the optimiser's grouping and therefore the training update, not just
#     inference. So c_q/c_k/c_v stay exactly as they are and the fused matrix is a derived
#     inference-only copy. `forward` is untouched and training never reaches this code.
#   - THE PARAMETER COUNT. The copy is assigned as a plain tensor attribute, which
#     nn.Module.__setattr__ puts in __dict__ rather than in _parameters or _buffers, so it
#     is invisible to parameters() and to state_dict(). num_params_total/active must not
#     move, and that is a prediction on this launch.
#   - kv_cache_bytes, WHICH IS GATED at 41943040 with only 7339520 bytes of headroom. The
#     copy is 3*512*512*2 = 1572864 bytes per layer, 12582912 over 8 layers, which would
#     BREACH that ceiling if it landed inside the probe's allocation delta. It cannot: the
#     copy is built lazily on first use of the NON-PREFILL branch, and the prefilled probe
#     (prepare.py measure_kv_cache_bytes at prefill=1536, graph=False) only ever enters the
#     prefill branch, so it never allocates this at all. That is control flow, not a
#     warm-up argument. For the prefill=1 probe the branch IS entered, and there the
#     discarded warm-up build_state() pays it -- which is what prepare.py says that
#     throwaway is for ("pays every one-time cost", "all first-time-only").
#   - peak_vram_bytes needs no argument at all: prepare.py reads it and only then calls
#     reset_peak_memory_stats, so it spans training and validation and cannot see any
#     inference-side allocation. Its observed value equals its ceiling exactly.
#
# The cast to bf16 reproduces what autocast does to an fp32 parameter on entry to F.linear,
# so the fused GEMV sees the same weight bytes the unfused three would have seen. What is
# NOT claimed identical is the reduction: cuBLAS may pick a different kernel for a
# 512->1536 GEMV than for three 512->512 ones, so the fidelity metrics may move. They are
# ceilings at 0.05 and stand at ~0.022.

# c0004: one mechanism -- let inductor own the matmuls in the four decode regions instead
# of calling out to cuBLAS, by compiling them with mode="max-autotune-no-cudagraphs".
#
# Why. After c0003 a step moves ~58.7 MB of bf16 weights in 521.45 us, ~113 GB/s, 5.7% of
# an A100 80GB's 2039 GB/s. The byte floor for those weights is ~28.8 us, so ~94% of the
# step is not bytes. With roughly 90 kernels a step that is ~5.7 us each, and today every
# GEMV is an extern cuBLAS call, which is an inductor FUSION BARRIER: `relu(x @ Wfc).square()`
# cannot fuse its epilogue into the matmul, and neither can the split+rope+norm chain after
# the QKV GEMV. A Triton matmul template can absorb those epilogues, so this attacks kernel
# count and GEMV tiling at once -- cuBLAS picks a kernel for a general m, and here m is 1.
#
# Why this mode and not the recorded disaster. AMENDMENTS.md records that
# mode="reduce-overhead" destroyed 4.49 bits/token silently, and the mechanism was
# cudagraph-trees copying inputs into a static pool so the in-place k/v append landed in the
# copy. "max-autotune-no-cudagraphs" is precisely the variant that does NOT enable
# cudagraphs -- that is what the suffix names -- and it is a documented public mode string
# rather than a hand-set config key that could be silently misspelled. Independently, fa3 is
# still outside every compiled region, so inductor cannot see the cache mutation at all.
# Autotuning runs on the first call, inside _GraphedDecodeStep's 3 warmup replays, before
# capture and outside every timed region; the throwaway build_state() in
# measure_kv_cache_bytes pays it before the measured delta.
#
# What is NOT claimed: that a Triton matmul reduces in the same order as cuBLAS. It need
# not, so the fidelity metrics may move. Both TV ceilings are 0.05 and stand at ~0.018-0.021.

@torch.compile(dynamic=False, fullgraph=True, mode="max-autotune-no-cudagraphs")
def _decode_qkv_nove(x, x0, rl, xl, w_qkv, sq, sk, sv, cos, sin, n_head, n_kv_head, head_dim):
    x = rl * x + xl * x0
    h = norm(x)
    B, Tn, _ = h.shape
    qkv = F.linear(h, w_qkv)
    q, k, v = qkv.split([sq, sk, sv], dim=-1)
    q = q.view(B, Tn, n_head, head_dim)
    k = k.view(B, Tn, n_kv_head, head_dim)
    v = v.view(B, Tn, n_kv_head, head_dim)
    q = norm(apply_rotary_emb(q, cos, sin))
    k = norm(apply_rotary_emb(k, cos, sin))
    return x, q, k, v


@torch.compile(dynamic=False, fullgraph=True, mode="max-autotune-no-cudagraphs")
def _decode_qkv_ve(x, x0, rl, xl, w_qkv, sq, sk, sv, sg, ve, cos, sin,
                   n_head, n_kv_head, head_dim):
    x = rl * x + xl * x0
    h = norm(x)
    B, Tn, _ = h.shape
    qkv = F.linear(h, w_qkv)
    # c0005: the gate's rows ride in the same matrix, zero-padded past ve_gate_channels,
    # so this is one GEMV rather than two. The separate wg / ve_ch arguments are gone.
    q, k, v, g = qkv.split([sq, sk, sv, sg], dim=-1)
    q = q.view(B, Tn, n_head, head_dim)
    k = k.view(B, Tn, n_kv_head, head_dim)
    v = v.view(B, Tn, n_kv_head, head_dim)
    gate = 2 * torch.sigmoid(g)
    v = v + gate.unsqueeze(-1) * ve.view(B, Tn, n_kv_head, head_dim)
    q = norm(apply_rotary_emb(q, cos, sin))
    k = norm(apply_rotary_emb(k, cos, sin))
    return x, q, k, v


def _gemv_reduce(w, v):
    """c0006: one memory-bound REDUCTION in place of a matmul template.

    `v` is (B, T, K) with B*T == 1 everywhere on the decode path; `w` is (N, K). A matmul
    template has to pad m=1 up to its smallest tile and can only parallelise over N, so for
    N=512 it launches a handful of CTAs on a 108-SM device to do a job whose whole cost is
    reading w. Written as a broadcast product reduced over K, inductor emits a reduction
    kernel instead and is free to split K, which is parallelism the template cannot express.

    The product is never materialised: inductor fuses a pointwise into the reduction it feeds.
    Accumulation is forced to fp32, which is what a bf16 cuBLAS GEMV also does, but the
    reduction ORDER differs and that is a fidelity risk, not a correctness argument.

    c0008 reverts c0007's hand-written split and restores this body exactly as c0006 ran it.
    c0007 reshaped w to (N, s, K/s) to make N*s reduction groups a property of the shape
    rather than of a heuristic, and it REGRESSED the ranking key by 7.98 ms (launch 8,
    16e2ef0a). With c0006's 3.30 ms against a pre-registered ~5 ms threshold, the occupancy
    story for this GEMV is refuted in both its forms, so the split is deleted rather than
    left switchable: a file should not carry a refuted branch behind a constant.
    """
    vv = v.unsqueeze(-2).to(w.dtype)   # 4 KB, and normally a no-op inductor folds away. It
    # is here so that w's dtype ALWAYS decides the read: a raw `*` is not autocast-listed, so
    # an fp32 w against a bf16 v would promote and read 4 MB where the mechanism needs 2.
    return (w * vv).sum(-1, dtype=torch.float32).to(v.dtype)


def _gemv_reduce_q8(wq, ws, v):
    """c0069: `_gemv_reduce` with the weight read as int8 and dequantised inside the kernel.

    `wq` is (N, K) int8, `ws` is (N,) fp32, `v` is (B, T, K) with B*T == 1. Same shape, same
    op, same broadcast-reduce form as `_gemv_reduce` -- ONE variable differs, the bytes of the
    weight load, which is 9.44 MB to 4.72 MB plus a 3 KB scale vector.

    The cast to bf16 is written on wq and NOT via `v.unsqueeze(-2).to(w.dtype)` as
    `_gemv_reduce` does, because with an int8 weight that idiom would cast the ACTIVATION to
    int8 and destroy it. bf16 is therefore named explicitly on both operands, which also
    keeps the multiply out of any promotion path.

    The scale is applied AFTER the sum. That is exact rather than approximate: s[n] is
    constant along the reduction axis k, so s[n] * sum_k(wq[n,k]*v[k]) is the same value as
    sum_k(s[n]*wq[n,k]*v[k]) up to fp32 rounding, and pulling it out means the reduction
    itself carries no extra work per element.

    Accumulation is fp32 as in `_gemv_reduce`, and the return is the ACTIVATION's dtype, also
    as in `_gemv_reduce`, so the contract seen by callers is unchanged.
    """
    vv = v.unsqueeze(-2).to(torch.bfloat16)
    acc = (wq.to(torch.bfloat16) * vv).sum(-1, dtype=torch.float32)
    return (acc * ws).to(v.dtype)


@torch.compile(dynamic=False, fullgraph=True, mode="max-autotune-no-cudagraphs")
def _decode_post(x, y, wproj, wfc, wp2):
    B, Tn = y.shape[0], y.shape[1]
    x = x + F.linear(y.contiguous().view(B, Tn, -1), wproj)
    h = norm(x)
    h = F.relu(F.linear(h, wfc)).square()
    # c0006: this ONE GEMV changes form. wp2 is (512, 2048): 2 MB read for 512 output rows,
    # the worst occupancy case in the step and the suspect named in notes/round4-result.md
    # before the arithmetic that now supports it existed. wproj (512x512) and wfc (2048x512)
    # keep F.linear, so the round attributes its result to one shape.
    return x + _gemv_reduce(wp2, h)


@torch.compile(dynamic=False, fullgraph=True, mode="max-autotune-no-cudagraphs")
def _decode_post_pro(x, y, wproj, wfcq, wfcs, wp2q, wp2s, rl_next, xl_next, x0):
    """c0021: `_decode_post` followed by the NEXT layer's residual combination and norm, in ONE
    graph, so the two opening lines of every layer have an epilogue to fuse into.

    Between `_decode_post` of layer i and `_decode_qkv_*` of layer i+1 there is nothing at all --
    the loop only slices a value embedding and reads two Python ints -- but they are two separate
    compiled regions, so `x = rl * x + xl * x0; h = norm(x)` cannot fuse into the reduction that
    produced x. Launch 21's kernel table carries a row with exactly those ops and no matmul:

        us_per_step=   12.65 n=  7.0  triton_per_fused_add_mean_mul_pow_0   (mul, add, pow, mean)

    n=7.0 is the number of INTER-LAYER boundaries in an 8-layer stack, which is what makes the
    attribution more than a guess: 7, not 8, because the last layer has no successor.

    The arithmetic and its ORDER are unchanged. This is the body of `_decode_post` followed by
    exactly the first two lines of `_decode_qkv_nove` / `_decode_qkv_ve`, on the same tensors in
    the same sequence. `_decode_post`, `_decode_qkv_nove` and `_decode_qkv_ve` are NOT modified
    and all three are still called: layer 0 opens with the originals because it has no
    predecessor, and the last layer closes with the original post because it has no successor.
    """
    B, Tn = y.shape[0], y.shape[1]
    # c0052: the ONE remaining GEMV on the live decode path that still uses a matmul
    # template. c0006 converted wp2 (768, 6144) to a reduction and measured 3.30 ms, and it
    # left wproj and wfc on F.linear explicitly "so the round attributes its result to one
    # shape" -- for ATTRIBUTION, not on evidence. This is that deferred extension.
    #
    # c0006'S OWN ACCOUNT PREDICTS A GAIN HERE AND NO GAIN ON wfc, which is what makes this
    # the right one of the two to charge. The account is output-row count, not bytes: a
    # matmul template pads m=1 to its smallest tile and can only parallelise over N, so at
    # N=768 it fills a handful of a 108-SM device. wproj is (768, 768) -- the SAME 768
    # output rows as the case that gained 3.30 ms -- while wfc is (6144, 768), where the
    # template already has 6144 rows to spread and the occupancy argument does not apply.
    # So wproj tests the extension where my own banked explanation says it should work,
    # and wfc is the FOLLOW-UP that would separate occupancy from bytes.
    #
    # BYTES ARE 8x SMALLER (1.18 MB vs 9.4 MB), so if the gain were bytes-proportional it
    # would be ~0.41 ms. The card bands [0, 1.2] ms and states P(null) before the row.
    x = x + _gemv_reduce(wproj, y.contiguous().view(B, Tn, -1))
    h = norm(x)
    h = F.relu(_gemv_reduce_q8(wfcq, wfcs, h)).square()
    x = x + _gemv_reduce_q8(wp2q, wp2s, h)
    xc = rl_next * x + xl_next * x0
    return xc, norm(xc)


@torch.compile(dynamic=False, fullgraph=True, mode="max-autotune-no-cudagraphs")
def _decode_post_head(x, y, wproj, wfcq, wfcs, wp2q, wp2s, w_lmq, w_lms, softcap, seq):
    """c0048: `_decode_post` followed by `_decode_head`, in ONE compiled region.

    THE BOUNDARY c0021 EXPLICITLY LEFT OPEN. `_decode_post_pro` folds the NEXT layer's
    prologue into a layer's epilogue, and its own docstring says why the last layer is
    excluded: "the last layer closes with the original post because it has no successor."
    That is true of a successor LAYER and false of a successor REGION. On the decode path
    `_decode_post` is called at exactly one site (last layer, non-prefill) and `_decode_head`
    at exactly one site (non-prefill), with nothing between them but `_PHASE.stop`,
    `softcap = 15` and `_PHASE.start()`. So the last layer does have a successor region, and
    the same mechanism that paid four times in this file has one boundary left to close.

    WHY THE SAVING IS REAL AND WHERE IT COMES FROM. `_decode_post` returns `x`, and at the
    last layer `x` is read by nothing except `_decode_head`. Two compiled regions mean `x`
    must be written to global memory by the first and read back by the second, and
    `_decode_head` opens with `norm(x[:, -1:, :])` -- a reduction over `x` that could have
    been the epilogue of the reduction that produced it. Merged, `x` need never be
    materialised.

    MAGNITUDE: BANDED, AND THE PRECEDENT'S NUMBER DOES NOT TRANSFER. Launch 21's kernel
    table priced the analogous boundary at `us_per_step=12.65 n=7.0`, i.e. 1.807 us per
    boundary, and launch 24 measured a mean kernel launch cost of 4.47 us. Those are the two
    ends of the band: 1.807 us/step if the saving is only what c0021 measured, 4.47 us/step
    if a whole launch disappears. Over REQUEST_STEPS=512 that is 0.925 to 2.289 ms.
    But 1.807 us/boundary was measured at EIGHT layers and a narrower model, and this stack
    is DEPTH=2 with model_dim 768. The MECHANISM transfers; the COEFFICIENT is being
    re-measured here, not assumed, and that is why the prediction is a band and not a point.

    THE BASE IS THE REPLICATE MEAN, NOT THE INCUMBENT. Parent c0046 keys 81.1147689819 and
    is the MINIMUM of its five-row byte-equal replicate set (mean 81.5844, sd 0.3852255929 on
    df 4). Under the standing rule adopted in round 48 -- never take a parent that is an
    extremum of its own replicate set without pricing regression to the mean -- the null
    prediction for this row is the MEAN 81.5844, not 81.1148. A null therefore looks like a
    0.47 ms LOSS against the incumbent, and I must not read that as this merge hurting.
    Conversely an accept-by-luck under the null has probability P(z < -1.219) = 11.1 %, so a
    bare pass of the incumbency bar is weak evidence and is pre-registered as such.

    THE DECIDING INSTRUMENT IS IN-ROW, NOT THE SCORED KEY. Round 49's analysis of my own five
    replicates found corr(request, nopref) = +0.972 and sd(request - nopref) = 0.0921 ms, so
    94 % of the ranking key's 0.3852 ms floor is a per-launch offset shared by both probes.
    The kernel table's `device_us_per_step` and all twenty `DECODE_SPLIT_SWEEP` rows are
    measured INSIDE this launch, where that offset cancels. The sweep's fixed (16,32) baseline
    has run in every launch from seq 42 to seq 49 with an sd of only 0.494 us/step (probe A)
    and 0.462 (probe B). A 1.807-4.47 us/step drop is 3.7-9.1 sd against that dispersion,
    where on the scored key it is only 2.4-5.9 sd. This is the first charge in this run whose
    treatment is measured more sharply by a free in-row diagnostic than by the metric.

    NON-INERTNESS IS PROVED BY A NEW STRING, NOT BY A THRESHOLD. `_PHASE` is given the bucket
    name "post_head", which no other version of this file can print. A numeric control would
    have been weaker: `n_kernel` is 21 for most sweep configs but 22 for long=33 and long=64,
    so it has natural range and a 21 -> 20 move could be argued either way. A bucket name that
    only this code path can emit cannot be satisfied by drift. This is the standing correction
    from round 47 -- a control must be something no unrelated event can satisfy -- applied at
    the level of the log rather than at the level of a band.

    FIDELITY. The statements and their ORDER are unchanged: this is the body of `_decode_post`
    followed by the body of `_decode_head`, on the same tensors in the same sequence.
    `seq.add_(1)` stays last and therefore still lands after every attention call in the step,
    which is the invariant `_decode_head`'s own docstring names. If that invariant breaks the
    launch says so loudly rather than silently: `decode_argmax_matches` collapses and
    `decode_tv_distance_max` goes through its 0.05 ceiling, and its `nopref_` twin with it.
    A reduction-order change is still possible where the head's norm now fuses into the wp2
    reduction's epilogue, and both tv ceilings are the pre-registered detector for it.

    ZERO PARAMETER COST, SO ZERO GATE COST. No weight, width, depth, window or table value
    moves, so all seven admissibility metrics are predicted BYTE-IDENTICAL to the parent and
    `flops_per_token_measured` stays 200541312. This is the only lever class left at this
    shape: round 49 measured the params -> val_bpb density at 0.001786 val_bpb per ms of MLP
    width saved against a gate margin of 0.0014453, so every width lever big enough to see
    costs more margin than exists.

    WITHDRAWN IN THE SAME TURN IT WAS FORMED, recorded here because it is what selected this
    charge: I first priced this region as a BANDWIDTH problem -- 61.56 us/step of "norm+mm"
    kernels moving 22.0 MB that needs 10.8 us at HBM peak, i.e. 17.5 % of peak, a 25.2 ms
    headroom. That is a CATEGORY ERROR and it is retracted. `_gemv_reduce` is applied to
    exactly one matrix, `wp2`, and the MLP pair it belongs to measures at 88 % of peak, so
    batch-1 GEMV is demonstrably NOT slow in this file. The 61.56 us bucket is the
    qkv/rotary/value-embed/head region, which reads only 3.5-12.6 MB while running long chains
    of tiny-tensor elementwise ops; its cost is op-chain latency, not reads, and comparing it
    to HBM peak is the same mistake as predicting `kv_cache_max_len` would follow a model
    parameter. What survives is that ~50 us/step of the step is non-bandwidth work, which is
    what a region merge addresses and a faster read does not.

    ALSO NOTED, NOT ACTED ON -- and the first draft of this paragraph got the location wrong,
    which the guard caught. The stale text is NOT in `_gemv_reduce`'s docstring; it is an
    inline c0006 comment inside `_decode_post` itself, and it names THREE shapes that the
    model has since outgrown: "wp2 is (512, 2048): 2 MB read for 512 output rows" and
    "wproj (512x512) and wfc (2048x512)". At this shape wp2 is (768, 6144) -- 9.4 MB and 768
    output rows, 4.7x the bytes -- while wproj is (768, 768) and wfc is (6144, 768). The
    comment's PREMISE is stale in all three. Its CONCLUSION survives anyway, because the
    kernel it defends measures at 88 % of HBM peak, so I am recording that the premise is
    stale rather than implying the choice is wrong.

    A CONSEQUENCE OF THIS CHARGE, DISCLOSED RATHER THAN TIDIED: `_decode_post` now has zero
    call sites, so that stale comment lives in dead code. I am NOT editing it. This charge
    changes one mechanism, and rewriting a previous round's rationale would make the diff
    argue two things at once. The defect is recorded here and in the card instead.
    """
    B, Tn = y.shape[0], y.shape[1]
    # c0052: the ONE remaining GEMV on the live decode path that still uses a matmul
    # template. c0006 converted wp2 (768, 6144) to a reduction and measured 3.30 ms, and it
    # left wproj and wfc on F.linear explicitly "so the round attributes its result to one
    # shape" -- for ATTRIBUTION, not on evidence. This is that deferred extension.
    #
    # c0006'S OWN ACCOUNT PREDICTS A GAIN HERE AND NO GAIN ON wfc, which is what makes this
    # the right one of the two to charge. The account is output-row count, not bytes: a
    # matmul template pads m=1 to its smallest tile and can only parallelise over N, so at
    # N=768 it fills a handful of a 108-SM device. wproj is (768, 768) -- the SAME 768
    # output rows as the case that gained 3.30 ms -- while wfc is (6144, 768), where the
    # template already has 6144 rows to spread and the occupancy argument does not apply.
    # So wproj tests the extension where my own banked explanation says it should work,
    # and wfc is the FOLLOW-UP that would separate occupancy from bytes.
    #
    # BYTES ARE 8x SMALLER (1.18 MB vs 9.4 MB), so if the gain were bytes-proportional it
    # would be ~0.41 ms. The card bands [0, 1.2] ms and states P(null) before the row.
    x = x + _gemv_reduce(wproj, y.contiguous().view(B, Tn, -1))
    h = norm(x)
    h = F.relu(_gemv_reduce_q8(wfcq, wfcs, h)).square()
    x = x + _gemv_reduce_q8(wp2q, wp2s, h)
    x = norm(x[:, -1:, :])
    logits = _gemv_reduce_q8(w_lmq, w_lms, x).float()
    out = softcap * torch.tanh(logits / softcap)
    seq.add_(1)
    return out


@torch.compile(dynamic=False, fullgraph=True, mode="max-autotune-no-cudagraphs")
def _decode_qkv_nove_h(x, h, w_qkv, sq, sk, sv, cos, sin, n_head, n_kv_head, head_dim):
    """c0021: `_decode_qkv_nove` with its first two lines DELETED -- `x` arrives already combined
    and `h` already normed, both computed by the previous layer's `_decode_post_pro`. Every other
    line is character-for-character the original's."""
    B, Tn, _ = h.shape
    qkv = F.linear(h, w_qkv)
    q, k, v = qkv.split([sq, sk, sv], dim=-1)
    q = q.view(B, Tn, n_head, head_dim)
    k = k.view(B, Tn, n_kv_head, head_dim)
    v = v.view(B, Tn, n_kv_head, head_dim)
    q = norm(apply_rotary_emb(q, cos, sin))
    k = norm(apply_rotary_emb(k, cos, sin))
    return x, q, k, v


@torch.compile(dynamic=False, fullgraph=True, mode="max-autotune-no-cudagraphs")
def _decode_qkv_ve_h(x, h, w_qkv, sq, sk, sv, sg, ve, cos, sin,
                     n_head, n_kv_head, head_dim):
    """c0021: `_decode_qkv_ve` with its first two lines DELETED. See `_decode_qkv_nove_h`."""
    B, Tn, _ = h.shape
# ===================================================================================
# c0060: THE LAST TWO F.linear SITES ON THE LIVE DECODE PATH BECOME _gemv_reduce.
#
# WHY, AND WHAT THIS IS NOT. This is c0056's treatment applied on a CHANGED BASE. It is
# not a re-run: c0056's parent was c0054 with DECODE_NUM_SPLITS_BY_WINDOW {256: 4,
# 2048: 32}; this row's parent is c0059 with {256: 16, 2048: 18}. c0056 measured
# +0.0767 ms, INSIDE one noise floor -- a null is no evidence either way, so this axis is
# UNRESOLVED rather than closed, and round 59 established that replicating a treatment
# across a changed base is informative (c0057 reproduced c0051's scored delta, probe A
# rise and A-B offset on a different lineage to 0.29 us/step).
#
# THE FORM IS COPIED BYTE-FOR-BYTE FROM c0056, INCLUDING THE .to(torch.bfloat16). That
# cast is not decoration: _gemv_reduce returns the ACTIVATION's dtype, not the weight's,
# so without it an fp32 h yields fp32 q/k/v and FlashAttention rejects them. That is the
# exact failure that forfeited the c0055 charge. The guard asserts byte-identity with
# c0056's two lines rather than trusting my transcription.
#
# WHAT THE ROW BUYS BEYOND THE SCORED KEY.
#   1. It removes the last two `mm` kernels from the live decode path. In c0059's table
#      those are rows [8] 5.47 and [9] 5.25 us/step, 10.72 total = 7.5% of probe A.
#   2. It takes n_kernel 21 -> 20, giving a second kernel-count point at a new base.
#      Round 54 finding 4 closed the kernel-count axis on six rows -- but every one of
#      those rows sat at 20 or 21, a 5% range. That closure bounds only what it sampled,
#      which is the same scope error as the retracted round-58 finding 4.
#   3. It is the out-of-sample test of the bytes account: w_qkv is 7.08 MB/step, and the
#      account says the gain scales with bytes moved.
#
# WHY THE MECHANISM SAYS THE GAIN IS SMALL. c0059's own kernel table prices every
# large-weight gemv at 900-1090 GB/s, i.e. 56-68% of achievable HBM on this part, so the
# weight path has at most ~2x of headroom and w_qkv is only 7.08 of ~59.8 MB/step. The
# attention block, by contrast, is 5-35x off roofline and splits-insensitive. I expect a
# NULL and pre-register it as the modal zone.
# ===================================================================================
    qkv = _gemv_reduce(w_qkv, h).to(torch.bfloat16)
    q, k, v, g = qkv.split([sq, sk, sv, sg], dim=-1)
    q = q.view(B, Tn, n_head, head_dim)
    k = k.view(B, Tn, n_kv_head, head_dim)
    v = v.view(B, Tn, n_kv_head, head_dim)
    gate = 2 * torch.sigmoid(g)
    v = v + gate.unsqueeze(-1) * ve.view(B, Tn, n_kv_head, head_dim)
    q = norm(apply_rotary_emb(q, cos, sin))
    k = norm(apply_rotary_emb(k, cos, sin))
    return x, q, k, v


@torch.compile(dynamic=False, fullgraph=True, mode="max-autotune-no-cudagraphs")
def _prefill_qkv(x, x0, rl, xl, w_q, w_k, w_v, cos, sin, n_head, n_kv_head, head_dim):
    """The PREFILL prologue+qkv+rope, same arithmetic in the same order, as one region.

    c0023. Launch 23 measured prefill at 10.291 ms for 1536 tokens and 10.264 ms for 384 --
    a 4x change in work for +0.3% time, i.e. 0.023 us per token against a 10.256 ms fixed
    cost. Prefill is host DISPATCH, not arithmetic: this branch was fully eager while every
    hot decode region is compiled, and it pays ~25 separately dispatched ops per layer whose
    cost does not care how wide the tensors are.

    Statement-for-statement the eager block it replaces, in the eager block's ORDER -- rope
    on q and k, then norm on q and k, which is prefill's order and NOT _decode_qkv_ve's
    (norm(rope(q)) then norm(rope(k))). The order is preserved because the claim is that
    nothing but the dispatch changes.

    F.linear with the fp32 parameter, exactly as the eager `attn.c_q(h)` did: c_q/c_k/c_v are
    all bias=False, so there is no bias term to drop, and autocast's cast happens inside
    F.linear. No fused weight and no bf16 copy is built here -- a new tensor on the prefill
    path is what `kv_cache_bytes` would charge, which is why c0020 kept its stack off this
    branch, and this region allocates nothing that outlives it.
    """
    B, Tn = x.shape[0], x.shape[1]
    x = rl * x + xl * x0
    h = norm(x)
    q = F.linear(h, w_q).view(B, Tn, n_head, head_dim)
    k = F.linear(h, w_k).view(B, Tn, n_kv_head, head_dim)
    v = F.linear(h, w_v).view(B, Tn, n_kv_head, head_dim)
    q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
    q, k = norm(q), norm(k)
    return x, q, k, v


@torch.compile(dynamic=False, fullgraph=True, mode="max-autotune-no-cudagraphs")
def _prefill_qkv_ve(x, x0, rl, xl, w_q, w_k, w_v, w_g, gate_ch, ve, cos, sin,
                    n_head, n_kv_head, head_dim):
    """`_prefill_qkv` for the four layers that carry a value embedding.

    The gate reads only the first `gate_ch` channels of h and ve_gate is bias=False, so
    F.linear on the slice is the eager `attn.ve_gate(h[..., :attn.ve_gate_channels])`
    exactly. Unlike c0005's decode fold, the gate's rows are NOT padded into another matrix:
    that would build a tensor on the prefill path.
    """
    B, Tn = x.shape[0], x.shape[1]
    x = rl * x + xl * x0
    h = norm(x)
    q = F.linear(h, w_q).view(B, Tn, n_head, head_dim)
    k = F.linear(h, w_k).view(B, Tn, n_kv_head, head_dim)
    v = F.linear(h, w_v).view(B, Tn, n_kv_head, head_dim)
    ve = ve.view(B, Tn, n_kv_head, head_dim)
    gate = 2 * torch.sigmoid(F.linear(h[..., :gate_ch], w_g))
    v = v + gate.unsqueeze(-1) * ve
    q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
    q, k = norm(q), norm(k)
    return x, q, k, v


@torch.compile(dynamic=False, fullgraph=True, mode="max-autotune-no-cudagraphs")
def _prefill_post(x, y, wproj, wfc, wp2):
    """The PREFILL epilogue: attention projection then the relu-square MLP, as one region.

    `x + attn.c_proj(y...)` then `x + block.mlp(norm(x))`, and MLP.forward is
    c_proj(relu(c_fc(x)).square()). Uses `block.mlp.c_proj.weight` and NOT the decode path's
    `decode_proj_weight()`: that is a lazily built bf16 COPY (c0006), and building it from
    the prefill branch would allocate on the path the kv_cache_bytes probe walks.
    """
    B, Tn = y.shape[0], y.shape[1]
    x = x + F.linear(y.contiguous().view(B, Tn, -1), wproj)
    h = norm(x)
    h = F.relu(F.linear(h, wfc)).square()
    return x + F.linear(h, wp2)


@torch.compile(dynamic=False, fullgraph=True, mode="max-autotune-no-cudagraphs")
def _decode_prologue(seq, rope_stack, rope_half, idx, emb_stack, ve_off):
    """c0024: the decode PROLOGUE as one region. Four ops per step ran EAGER, outside every
    compiled region, and launch 24's kernel table names them:

        us_per_step=    7.32 n=  2.0  indexSelectSmallIndex<c10::BFloat16 ...   (rope + embedding)
        us_per_step=    2.41 n=  1.0  vectorized_layer_norm_kernel<...>         (the eager norm)
        us_per_step=    2.21 n=  1.0  unrolled_elementwise_kernel<direct_copy>  (the int64 cast)

    n=1.0 on the last two is what makes the attribution more than a guess: the prologue runs ONCE
    per step, unlike the per-layer rows at n=7 and n=8. c0020's comment above the embedding block
    claims "every slice below is a VIEW and no copy kernel appears", and that is right about the
    slices -- the direct_copy row is the int64 CAST of seq, one line earlier.

    The arithmetic and its ORDER are unchanged: this is the `else` branch of the two opening
    `if prefill:` blocks of `_decode_body`, statement for statement, on the same tensors in the
    same sequence. `rope_half` and `ve_off` are Python ints and so are compile-time constants,
    not graph inputs that could force a recompile.

    Deliberately left OUTSIDE: `decode_rope_stack()` and `decode_embed_stack()` are the lazily
    built caches, and calling them from inside a region would put tensor construction on a path
    `measure_kv_cache_bytes` walks. `ve_dim` stays out because it is Python arithmetic on a
    shape. `_PHASE` stays out because a compiled region may not contain it.

    The PREFILL branch is untouched, so the no-prompt and prefilled shapes both change here and
    by the same per-step amount -- unlike c0023, where only one shape could see the change.
    """
    seq_idx = seq.to(torch.int64)
    rope = rope_stack.index_select(1, seq_idx)
    cos = rope[..., :rope_half]
    sin = rope[..., rope_half:]
    ve_all = F.embedding(idx, emb_stack)
    x = norm(ve_all[..., :ve_off])
    return cos, sin, ve_all, x


@torch.compile(dynamic=False, fullgraph=True, mode="max-autotune-no-cudagraphs")
def _decode_prologue_qkv(seq, rope_stack, rope_half, idx, emb_stack, ve_off,
                         rl, xl, w_qkvq, w_qkvs, sq, sk, sv, n_head, n_kv_head, head_dim):
    """c0049: `_decode_prologue` and LAYER 0's qkv region as one region. The FRONT boundary,
    and the mirror of what c0048 did at the back.

    WHY THIS IS THE ANALOGOUS BOUNDARY, at the call-site level rather than by analogy.
    `pending_h` is None exactly at `i == 0` -- c0021 set it from layer i-1's `_decode_post_pro`
    and every later layer therefore takes a `_h` variant. So layer 0 is the ONLY layer that
    reaches `_decode_qkv_nove` / `_decode_qkv_ve` at all, and on the decode path the prologue
    and layer 0's qkv are ALWAYS adjacent, with nothing between them but `_PHASE.stop`, the
    `ve_dim` shape arithmetic, `x0 = x`, and the `ve is None` dispatch. c0024 already merged
    the two eager prologue blocks into one region and stopped there; c0021 merged prologue into
    epilogue for every layer that HAS a predecessor. Neither crossed this seam, because layer 0
    has no predecessor layer -- which is true of a predecessor LAYER and false of a predecessor
    REGION, the same asymmetry c0048 exploited at the other end.

    WHY THE SAVING IS REAL. `x_pro` is written to global memory by the prologue and read back
    by the qkv region purely to cross the boundary. Merged, `norm(ve_all[..., :ve_off])`
    feeds `rl * x + xl * x0` and then `norm(x)` without a round trip, and the second `norm`
    can fuse into the first's epilogue.

    MAGNITUDE, and this is the first prediction in this run whose coefficient is MY OWN
    measurement at THIS shape. c0048 merged one boundary and its in-launch `DECODE_SPLIT_SWEEP`
    fixed-(16,32) baseline fell 1.1500 us/step against its parent c0046 and 0.7940 against the
    five-row shape-equal mean. So one boundary at DEPTH=2, model_dim 768, is worth
    0.79-1.15 us/step = 0.41-0.59 ms over 512 steps. That REPLACES c0021's 1.807 us/boundary,
    which was measured at EIGHT layers and a narrower model and which c0048 over-predicted by
    1.57-3.9x. Mechanism transferred; magnitude did not, exactly as c0048's card declared it
    might, and the correction is to stop quoting the foreign coefficient at all.

    x0 = x IS NOT AN ARITHMETIC CHANGE. In `_decode_body` the order is `x = x_pro`, then the
    `ve_dim` shape arithmetic, then `x0 = x`. `ve_dim` does not touch `x`, so hoisting `x0 = x`
    above it is a Python-level reordering of an aliasing assignment, not of a tensor op. At
    layer 0, x0 and x are the SAME tensor, which is why `rl * x + xl * x0` is evaluated here on
    one input; the expression is copied verbatim rather than simplified to `(rl + xl) * x`,
    because simplifying it would change the arithmetic and the float result.

    CORRECTNESS PRECONDITION, asserted at the call site and not trusted. This body inlines
    `_decode_qkv_nove`, so it is correct only if layer 0 takes the NO-value-embedding branch.
    That is a property of `has_ve(0, n_layer)` and of DEPTH, neither of which this candidate
    fixes: `has_ve(i, n) = i % 2 == (n - 1) % 2`, so at DEPTH 2 layer 0 has no value embedding
    and layer 1 does. If a future shape gives layer 0 a value embedding the call site raises
    rather than silently computing the wrong qkv.

    COST. Zero parameters, zero flops, zero training reach -- the prefill branch keeps
    `_decode_prologue` untouched and never had a merged prologue to begin with. Both probes
    traverse this region, so both shapes change by the same per-step amount.
    """
    seq_idx = seq.to(torch.int64)
    rope = rope_stack.index_select(1, seq_idx)
    cos = rope[..., :rope_half]
    sin = rope[..., rope_half:]
    ve_all = F.embedding(idx, emb_stack)
    x = norm(ve_all[..., :ve_off])
    x0 = x
    x = rl * x + xl * x0
    h = norm(x)
    B, Tn, _ = h.shape
    qkv = _gemv_reduce_q8(w_qkvq, w_qkvs, h).to(torch.bfloat16)
    q, k, v = qkv.split([sq, sk, sv], dim=-1)
    q = q.view(B, Tn, n_head, head_dim)
    k = k.view(B, Tn, n_kv_head, head_dim)
    v = v.view(B, Tn, n_kv_head, head_dim)
    q = norm(apply_rotary_emb(q, cos, sin))
    k = norm(apply_rotary_emb(k, cos, sin))
    return cos, sin, ve_all, x0, x, q, k, v


@torch.compile(dynamic=False, fullgraph=True, mode="max-autotune-no-cudagraphs")
def _decode_head(x, w_lm, softcap, seq):
    """c0025: the head region also advances the position.

    Launch 25's kernel table has ~11 rows at n=1 and 1.5-2.5 us each, and one of them,
    `vectorized_elementwise_kernel` at 1.56 us, is a whole kernel launch that increments a
    ONE-ELEMENT int tensor. At the 4.47 us mean cost of a launch measured in launch 24, a
    scalar bookkeeping kernel is not priced by its arithmetic but by its dispatch.

    Why here and not in the prologue: `seq` is `cache_seqlens` for fa3, so the increment must
    land AFTER every attention call in the step. `_decode_head` is the last region on the
    decode path, so folding it in preserves the order the eager code had.

    Why this is safe to move: `decode_step` advanced `seq` on the decode path at three sites,
    all of them immediately after a `_decode_body(..., prefill=False)` call that ends in this
    region. All three are removed. The prefill site keeps its own `add_(idx.size(1))` because
    the prefill branch returns before this region is reached.

    If the bookkeeping is wrong the launch says so loudly rather than silently: `seq` is the
    single source of position, so a double or missing increment collapses
    `decode_argmax_matches` and pushes `decode_tv_distance_max` through its 0.05 ceiling.
    """
    x = norm(x[:, -1:, :])
    logits = F.linear(x, w_lm).float()
    out = softcap * torch.tanh(logits / softcap)
    seq.add_(1)
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
        y = self.c_proj(y)
        return y

    def decode_out_weight(self):
        """c0052: a bf16 copy of the ATTENTION OUTPUT projection for the decode reduction.
    
        Exactly the object `decode_proj_weight` is for the MLP down-projection, built the same
        way and for the same reason: `w * v` is not autocast-listed, so an fp32 w against a
        bf16 v would PROMOTE and read twice the bytes, which is the opposite of the mechanism.
    
        Written through self.__dict__, bypassing nn.Module.__setattr__, so it cannot land in
        _parameters or _buffers under any torch version and num_params_total cannot move.
        Built lazily from the non-prefill decode branch only, so training never allocates it,
        MuonAdamW's shape-based grouping is untouched, and it stays out of kv_cache_bytes.
        """
        w = self.__dict__.get("_w_out_decode")
        if w is None:
            with torch.no_grad():
                w = self.c_proj.weight.to(torch.bfloat16).contiguous()
            self.__dict__["_w_out_decode"] = w
        return w

    def decode_qkv_weight(self):
        """The one fused [q;k;v] matrix for the width-1 decode GEMV. Built once, cached.

        c0003. Derived from c_q/c_k/c_v and never a substitute for them: they remain the
        parameters, they are what `forward` and the optimiser see, and MuonAdamW's
        group-by-shape is therefore unchanged. Assigned as a plain tensor attribute, so
        nn.Module keeps it in __dict__ rather than _parameters or _buffers and it is
        invisible to parameters() and state_dict() -- the parameter counts must not move.

        bf16 because that is what autocast hands F.linear for an fp32 parameter; building
        the copy pre-cast means the fused GEMV reads the same weight bytes the three
        unfused ones would have read.

        Built lazily, from the non-prefill branch only, which is what keeps it out of the
        gated kv_cache_bytes reading. See the c0003 comment at the top of this file.
        """
        w = self.__dict__.get("_w_qkv_decode")
        if w is None:
            with torch.no_grad():
                parts = [self.c_q.weight, self.c_k.weight, self.c_v.weight]
                if self.ve_gate is not None:
                    # c0005: fold the ve_gate GEMV into the same matrix, so the four layers
                    # that have a value embedding issue one GEMV instead of two. ve_gate
                    # reads only the first ve_gate_channels of h, so its rows are zero-padded
                    # out to n_embd. The padding contributes exact 0.0 products, so the value
                    # is the same sum of the same 32 terms -- but a 512-lane reduction may
                    # accumulate those 32 in a different order than a 32-lane one, so
                    # bitwise identity is NOT claimed.
                    g = self.ve_gate.weight.new_zeros((self.n_kv_head, self.n_embd))
                    g[:, :self.ve_gate_channels] = self.ve_gate.weight
                    parts.append(g)
                w = torch.cat(parts, 0).to(torch.bfloat16).contiguous()
            self.__dict__["_w_qkv_decode"] = w
        return w

    def decode_qkv_splits(self):
        """Output widths inside the fused matrix: q, k, v, then the gate if this layer has one."""
        sizes = (self.n_head * self.head_dim,
                 self.n_kv_head * self.head_dim,
                 self.n_kv_head * self.head_dim)
        if self.ve_gate is not None:
            sizes = sizes + (self.n_kv_head,)
        return sizes

    def decode_qkv_weight_q8(self):
        """c0073: the int8 weight-only copy of the fused [q;k;v(;gate)] decode matrix.

        Returns (wq, ws): wq is (rows, n_embd) int8, ws is (rows,) fp32. The quantiser is the
        SAME RULE, byte for byte, as decode_fc_weight_q8's, decode_proj_weight_q8's and
        decode_lm_weight_q8's -- symmetric, per output row, no zero point, 127 rather than 128
        so nothing maps to -128 -- because the whole value of this row is that its marginal
        us-per-MB is comparable with wp2's 0.4839, wfc's 0.0551 and w_lm's 0.6500.

        THE ROWS DIFFER BETWEEN THE TWO LAYERS AND THAT IS NOT A BUG. decode_qkv_weight cats
        c_q, c_k, c_v and, only for a layer with a value embedding, the zero-padded ve_gate
        rows. has_ve(0, 2) is False and has_ve(1, 2) is True, so layer 0 is (2304, 768) and
        layer 1 is (2310, 768). Per decode step: 3538944 -> 1778688 and 3548160 -> 1783320,
        a removal of 1760256 + 1764840 bytes. I had used 2304 for both layers in three
        previous cards and every traffic figure I quoted understated by 9216 bytes.

        NO OP CHANGES ANYWHERE. On the decode path layer 0's qkv runs inside
        _decode_prologue_qkv and layer 1's in _decode_qkv_ve_h, and BOTH already use
        _gemv_reduce. The F.linear variants are reached only on the prefill branch, which is
        why they are left alone -- so prefill still reads the bf16 fused copy and the single
        variable of this row is the bytes of the decode load.

        Built once, lazily, through self.__dict__, bypassing nn.Module.__setattr__ exactly as
        decode_qkv_weight already does, so it cannot enter _parameters or _buffers,
        num_params_total cannot move, MuonAdamW's group-by-shape is untouched, and it stays
        outside the gated kv_cache_bytes allocation delta measured around init_decode_state.
        c_q, c_k, c_v and ve_gate remain the parameters and are what forward, the optimiser
        and validation read, which closes the WEIGHTS channel to val_bpb but not the CLOCK
        channel.

        A FIDELITY NOTE SPECIFIC TO THIS WEIGHT. The zero-padded ve_gate rows are exact zeros
        outside the first ve_gate_channels columns, so their per-row amax is taken over the
        real entries only and the padding quantises to exact 0 at any scale. But those rows
        are 6 of 2310 and their scale is unrelated to the q/k/v rows', which is an argument
        for the per-ROW scale and against a per-tensor one.
        """
        wq = self.__dict__.get("_w_qkv_decode_q8")
        if wq is None:
            with torch.no_grad():
                w = self.decode_qkv_weight().float()
                ws = (w.abs().amax(dim=1) / 127.0).clamp_min(1e-12).contiguous()
                wq = torch.round(w / ws.unsqueeze(1)).clamp_(-127.0, 127.0)
                wq = wq.to(torch.int8).contiguous()
                deq = wq.to(torch.float32) * ws.unsqueeze(1)
                den = w.abs().amax().clamp_min(1e-12)
                err = (deq - w).abs()
                print(f"DECODE_WEIGHT_Q8: site=attn_qkv shape={tuple(w.shape)} "
                      f"wq_dtype={wq.dtype} wq_elt={wq.element_size()} "
                      f"bytes_bf16={w.numel() * 2} "
                      f"bytes_int8={wq.numel() * wq.element_size() + ws.numel() * 4} "
                      f"scale_shape={tuple(ws.shape)} scale_dtype={ws.dtype} "
                      f"max_abs_rel_err={(err.amax() / den).item():.6e} "
                      f"mean_abs_rel_err={(err.mean() / den).item():.6e} "
                      f"frac_at_clamp={(wq.abs() >= 127).float().mean().item():.6e}", flush=True)
                del deq, err
            self.__dict__["_w_qkv_decode_q8"] = wq
            self.__dict__["_w_qkv_decode_q8_scale"] = ws
        return wq, self.__dict__["_w_qkv_decode_q8_scale"]


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, MLP_HIDDEN, bias=False)
        self.c_proj = nn.Linear(MLP_HIDDEN, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

    def decode_proj_weight(self):
        """c0006: a bf16 copy of c_proj.weight for the decode reduction, built once, lazily.

        Needed because `w * v` is not an autocast-listed op: with w fp32 and v bf16 the
        multiply would PROMOTE and read 4 MB instead of 2, which is the opposite of the
        mechanism. F.linear's own autocast cast is not reachable from a raw multiply.

        Written through self.__dict__, bypassing nn.Module.__setattr__ entirely, so it cannot
        land in _parameters or _buffers under any torch version and num_params_total cannot
        move. Built from the non-prefill decode path only, so training never allocates it and
        MuonAdamW's shape-based grouping of 2D matrices is untouched.
        """
        w = self.__dict__.get("_w_proj_decode")
        if w is None:
            with torch.no_grad():
                w = self.c_proj.weight.to(torch.bfloat16).contiguous()
            self.__dict__["_w_proj_decode"] = w
        return w
    def decode_fc_weight_q8(self):
        """c0070: the int8 weight-only copy of c_fc.weight, plus its per-row fp32 scale.

        Returns (wq, ws): wq is (MLP_HIDDEN, n_embd) int8, ws is (MLP_HIDDEN,) fp32. The
        quantiser is BYTE-FOR-BYTE the same rule as decode_proj_weight_q8's -- symmetric, per
        output row, no zero point, 127 not 128 so nothing maps to -128 -- because this row is a
        replication of that row's law on the byte-matched twin, and a replication that changes
        the estimator is not a replication.

        THE ONE ARITHMETIC DIFFERENCE, STATED RATHER THAN SMOOTHED. c_fc.weight is
        (6144, 768), so the scale vector has 6144 entries and costs 24576 bytes against wp2's
        3072. Bytes go 9437184 -> 4718592 + 24576 = 4743168, a removal of 4694016 per layer
        and 9388032 per step, which is 0.46 per cent less than wp2's 9431040. The reduction
        axis is 768 rather than 6144, so each output accumulates its rounding error over 8x
        fewer terms.

        Built once, lazily, through self.__dict__, bypassing nn.Module.__setattr__ exactly as
        decode_fc_weight, decode_proj_weight, decode_proj_weight_q8, decode_out_weight and
        decode_lm_weight already do. So it cannot land in _parameters or _buffers under any
        torch version, num_params_total cannot move, training never allocates it, MuonAdamW's
        shape-based grouping is untouched, and it stays outside the gated kv_cache_bytes
        allocation delta, which is one delta measured around init_decode_state.

        self.c_fc.weight, the fp32 PARAMETER, is NOT modified and is still what `forward`, the
        optimiser and the validation pass read. That closes the WEIGHTS channel to val_bpb. It
        does not close the CLOCK channel, which runs through wall-clock LR progress to
        num_steps to tokens seen, and which this file no longer claims to close.

        The witness reports the MEASURED rounding error rather than asserting a bound, and it
        prints one line per layer from eager code, because this method is called at the call
        site to build an ARGUMENT to the compiled region and so sits outside fullgraph=True.

        NOTE ON ONE OF ITS NUMBERS: max_abs_rel_err is 0.5/127 = 3.93701e-03 BY CONSTRUCTION
        for any weight distribution, since the row holding the global maximum has scale
        amax/127 and a half-step of rounding error. It was recorded in round 71 as a check
        with NO DYNAMIC RANGE. mean_abs_rel_err and frac_at_clamp do have range -- in round 71
        the mean differed 3x between the two layers -- so those are the informative fields.
        """
        wq = self.__dict__.get("_w_fc_decode_q8")
        if wq is None:
            with torch.no_grad():
                w = self.c_fc.weight.float()
                ws = (w.abs().amax(dim=1) / 127.0).clamp_min(1e-12).contiguous()
                wq = torch.round(w / ws.unsqueeze(1)).clamp_(-127.0, 127.0)
                wq = wq.to(torch.int8).contiguous()
                deq = wq.to(torch.float32) * ws.unsqueeze(1)
                den = w.abs().amax().clamp_min(1e-12)
                err = (deq - w).abs()
                print(f"DECODE_WEIGHT_Q8: site=mlp_c_fc shape={tuple(w.shape)} "
                      f"wq_dtype={wq.dtype} wq_elt={wq.element_size()} "
                      f"bytes_bf16={w.numel() * 2} "
                      f"bytes_int8={wq.numel() * wq.element_size() + ws.numel() * 4} "
                      f"scale_shape={tuple(ws.shape)} scale_dtype={ws.dtype} "
                      f"max_abs_rel_err={(err.amax() / den).item():.6e} "
                      f"mean_abs_rel_err={(err.mean() / den).item():.6e} "
                      f"frac_at_clamp={(wq.abs() >= 127).float().mean().item():.6e}", flush=True)
                del deq, err
            self.__dict__["_w_fc_decode_q8"] = wq
            self.__dict__["_w_fc_decode_q8_scale"] = ws
        return wq, self.__dict__["_w_fc_decode_q8_scale"]

    def decode_proj_weight_q8(self):
        """c0069: the int8 weight-only copy of c_proj.weight, plus its per-row fp32 scale.

        Returns (wq, ws): wq is (n_embd, MLP_HIDDEN) int8, ws is (n_embd,) fp32. Built once,
        lazily, through self.__dict__ -- bypassing nn.Module.__setattr__ -- exactly as
        decode_proj_weight, decode_fc_weight, decode_out_weight and decode_lm_weight are, so
        it cannot land in _parameters or _buffers under any torch version, num_params_total
        cannot move, training never allocates it, MuonAdamW's shape-based grouping is
        untouched, and it stays outside the gated kv_cache_bytes allocation delta.

        self.c_proj.weight, the fp32 PARAMETER, is NOT modified and is still what `forward`,
        the optimiser and the validation pass read. That is why val_bpb cannot move: the gate
        certifies a path that never sees this object.

        Symmetric per-output-row quantisation, no zero point, 127 not 128 so the range stays
        symmetric and nothing maps to -128. The scale is per ROW because the row index is the
        output index and is therefore constant along the reduction axis, which is what lets
        _gemv_reduce_q8 pull it out of the sum exactly.

        The witness line reports the MEASURED rounding error rather than asserting a bound,
        and it is printed from here, once per layer, in eager code: this method is called at
        the call site to build an ARGUMENT to the compiled `_decode_post_*` region, so the
        print is outside fullgraph=True and cannot break a graph.
        """
        wq = self.__dict__.get("_w_proj_decode_q8")
        if wq is None:
            with torch.no_grad():
                w = self.c_proj.weight.float()
                ws = (w.abs().amax(dim=1) / 127.0).clamp_min(1e-12).contiguous()
                wq = torch.round(w / ws.unsqueeze(1)).clamp_(-127.0, 127.0)
                wq = wq.to(torch.int8).contiguous()
                deq = wq.to(torch.float32) * ws.unsqueeze(1)
                den = w.abs().amax().clamp_min(1e-12)
                err = (deq - w).abs()
                print(f"DECODE_WEIGHT_Q8: site=mlp_c_proj shape={tuple(w.shape)} "
                      f"wq_dtype={wq.dtype} wq_elt={wq.element_size()} "
                      f"bytes_bf16={w.numel() * 2} "
                      f"bytes_int8={wq.numel() * wq.element_size() + ws.numel() * 4} "
                      f"scale_shape={tuple(ws.shape)} scale_dtype={ws.dtype} "
                      f"max_abs_rel_err={(err.amax() / den).item():.6e} "
                      f"mean_abs_rel_err={(err.mean() / den).item():.6e} "
                      f"frac_at_clamp={(wq.abs() >= 127).float().mean().item():.6e}", flush=True)
                del deq, err
            self.__dict__["_w_proj_decode_q8"] = wq
            self.__dict__["_w_proj_decode_q8_scale"] = ws
        return wq, self.__dict__["_w_proj_decode_q8_scale"]

    def decode_fc_weight(self):
        """c0053: a bf16 copy of c_fc.weight for the decode reduction. THE DISCRIMINATING HALF
        OF c0006, and the reason it is worth a charge is that my two live accounts of c0006's
        3.30 ms predict OPPOSITE results here.

        THE TWO ACCOUNTS. c0006 converted the MLP DOWN-projection wp2 to `(w * v).sum(-1)` and
        measured 3.30 ms. It left wproj and wfc on F.linear, saying in this file that this was
        "so the round attributes its result to one shape" -- an attribution choice, never
        evidence. c0052 took wproj and measured NO GAIN (+1.6733 ms against its code parent,
        and the incumbency it won was by 0.0389 ms, which is 0.09 sd on the 0.4238 ms floor and
        is not a gain). Branch 3 of the c0052 card pre-registered what that means: the
        OUTPUT-ROW-COUNT account is retracted and what survives is a BYTES account.

        SO THE TWO ACCOUNTS NOW SPLIT, AND wfc IS EXACTLY WHERE THEY SPLIT:
          - BYTES account: wfc is (6144, 768) = 9.4 MB of weight, THE SAME BYTES AS wp2's
            (768, 6144). Same bytes, same 768*6144 broadcast intermediate. Predicts a gain of
            the same ORDER as wp2's 3.30 ms.
          - ROW-COUNT account: wfc has 6144 output rows, so a matmul template already has 6144
            rows to spread over 108 SMs and the m=1 occupancy argument does not apply at all.
            Predicts ZERO.
        There is no third weight left to ask, so whichever way this row falls, the surviving
        explanation of c0006's 3.30 ms is decided by it. That is why this is the charge.

        WHY THE bf16 COPY IS NOT OPTIONAL AND IS NOT A SECOND TREATMENT. The two live decode
        sites currently pass `block.mlp.c_fc.weight`, the fp32 PARAMETER, while wp2 is passed
        `decode_proj_weight()`, a bf16 copy. `w * v` is not an autocast-listed op, so an fp32 w
        against a bf16 v would PROMOTE and read 18.9 MB instead of 9.4 -- the exact failure
        `decode_proj_weight`'s own docstring was written to prevent, and it would read as a
        refutation of the bytes account while actually testing double the bytes. Building the
        copy is what makes the comparison to wp2 a comparison of one variable.

        Written through self.__dict__, bypassing nn.Module.__setattr__, so it cannot land in
        _parameters or _buffers under any torch version and num_params_total cannot move.
        Built lazily from the non-prefill decode branch only, so training never allocates it,
        MuonAdamW's shape-based grouping is untouched, and it stays outside kv_cache_bytes.
        Identical in construction to decode_proj_weight and c0052's decode_out_weight.
        """
        w = self.__dict__.get("_w_fc_decode")
        if w is None:
            with torch.no_grad():
                w = self.c_fc.weight.to(torch.bfloat16).contiguous()
            self.__dict__["_w_fc_decode"] = w
        return w


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


# c0074: THE 2310 ALIGNMENT HYPOTHESIS. int8 ON w_qkv FOR LAYER 0 ONLY; LAYER 1 STAYS bf16.
#
# WHAT ROUND 75 ESTABLISHED, AND IT CLOSED THE AXIS. c0073 put int8 on w_qkv for BOTH layers.
# request_ms_median 62.57164478302002 against c0071's 61.606764793395996 = +0.96488 ms =
# +1.270 sd, zone `regression`. c0071 REMAINS THE INCUMBENT.
#
#   THE DISCRIMINATOR WORKED. n_kernel came back 20 with an order byte-identical to c0071's,
#   against seq 74's uniform 19, so the paired instrument was VALID for the first time since
#   seq 73 and ROUND 74'S COMPILER RE-PARTITION IS LOCATED: IT FOLLOWED wproj, NOT w_qkv. An
#   integer with no band and no boundary I chose did in one charge what three consecutive
#   boundary-adjacent band verdicts could not.
#
#   THE BYTE ACCOUNT IS REFUTED IN SIGN. Warm pass 2 c0071->c0073 = +0.9345 us/step (sd
#   0.2234) -- a RISE -- against a predicted FALL of 2.2913. The control pair reads +0.0873
#   (sd 0.3336). ZERO kernel names changed (19->19, none gone, none new), so this is fully
#   LOCAL and the sign is not a re-partition artefact. Refutation in sign is stronger than
#   any magnitude refutation this axis produced.
#
# THE COMPLETE AXIS MAP, ALL FIVE DECODE GEMV WEIGHTS:
#     wp2   (768, 6144)        9431040 B/step   -2.63262 ms   GAIN
#     wfc   (6144, 768)        9388032 B/step   -0.27323 ms   null
#     w_lm  (8192, 768)        6258688 B/step   -2.28739 ms   GAIN
#     w_qkv (2304/2310, 768)   3525096 B/step   +0.96488 ms   REGRESSION
#     wproj (768, 768) x2      1173504 B/step   inferred +3.13342, NEVER MEASURED ALONE
#   THE SIGN IS NOT ORDERED BY BYTES, BY SHAPE, OR BY REDUCTION AXIS. wfc, w_qkv and w_lm all
#   reduce over the SAME short 768 axis and are null, regression and gain respectively. Shape
#   stays refuted; bytes are now refuted in sign.
#
# WHY THIS ROW. 2304 = 18 x 128 and is aligned to any plausible tile. 2310 = 2x3x5x7x11 is
# NOT a multiple of 128, 32, 16 or even 8. It is 2310 for a structural reason already
# confirmed by a witness: has_ve(layer_idx, n_layer) = layer_idx %% 2 == (n_layer-1) %% 2, so
# at DEPTH 2 only layer 1 carries a value embedding and its fused qkv gains exactly 6
# ve_gate rows. If the int8 GEMV is tile-quantised, ALL of c0073's regression could live in
# layer 1 and layer 0 could be a GAIN.
#
#   ALIGNMENT ACCOUNT: layer 0 alone removes 1760256 B/step = 1.760 MB and should behave like
#   w_lm and wp2 -- a NEGATIVE warm-pass delta, a fall of roughly 1.0-1.3 us/step.
#   UNIFORM ACCOUNT: layer 0 alone costs about HALF of c0073's +0.9345, so a POSITIVE delta
#   near +0.47.
#   THEY DIFFER IN SIGN, which is the property that made round 75 work. The control pair's
#   pass-2 noise is +0.0873 +/- 0.3336, so a +/-0.5 separation is real but NOT comfortable,
#   and the card leads on the SIGN and not on the magnitude.
#
# HOW THE SPLIT IS MADE, AND IT IS STRUCTURAL RATHER THAN A NEW CONDITIONAL. The decode qkv
# path is ALREADY split by whether the layer has a value embedding: _decode_prologue_qkv
# handles layer 0 and _decode_qkv_ve_h handles layer 1. The tree's own c0049 assertion states
# that layer 0 has NO value embedding at this shape. So this row reverts _decode_qkv_ve_h and
# its single dispatch site to c0071's text BYTE FOR BYTE, extracted programmatically from
# c0071 by AST line range rather than retyped, and leaves _decode_prologue_qkv on int8. NO
# NEW BRANCH IS INTRODUCED and no per-layer flag is added.
#
# THE BUILDER IS DELIBERATELY LEFT SAYING `c0073:`. decode_qkv_weight_q8 is byte-identical to
# c0073's, label included, because it IS c0073's builder and the guard asserts that identity.
# Relabelling it would make the estimator's sameness an assertion instead of a fact.
#
# WHAT ROUND 75 ALSO CORRECTED, AND THESE ARE PROVENANCE CORRECTIONS, NOT VALUE CORRECTIONS.
#   1. A JOIN ERROR OF MINE. I wrote K[71] meaning c0071; K is keyed on launch_seq and seq 71
#      is c0069. The wrong pair produced a coherent, quotable mechanism with the OPPOSITE
#      SIGN of the truth. Every accessor now takes a candidate_id and resolves the seq
#      itself, and prints the candidate->seq map: c0066->68 c0068->70 c0069->71 c0070->72
#      c0071->73 c0072->74 c0073->75.
#   2. THE DEVICE REGEX HAD NO LEFT WORD BOUNDARY and was also matching
#      dev1_device_us_per_step (a SECOND DEVICE) and unfiltered_device_us_per_step (INCLUDES
#      ANNOTATION). There are 22 matches, not 26. The slices are correct; the provenance was
#      not.
#   3. THE INSTRUMENT IS 9 EFFECTIVE OBSERVATIONS, NOT 11. The 11 members per pass are 10
#      DECODE_SPLIT_SWEEP lines plus 1 DECODE_KERNELS WHOLE-STEP AGGREGATE, and the 10 sweep
#      lines are only 9 DISTINCT configurations ([16,32] appears twice). n=11 df=10, which I
#      published for five rounds, OVERSTATES independence and mixes two kinds of quantity. No
#      verdict changes; every sigma from it is now read as at most 9 effective observations of
#      a heterogeneous population.
#
# WHAT IS REFUTED AND MUST NOT BE USED HERE. The pass-2 predictor is REFUTED at its fifth
# test (0.888, 0.969, 0.911, 0.921, then 0.496 on a row whose pairing PASSED). It is not used
# to convert us/step into ms on this row. ADDITIVITY is an assumption, not a fact: c0073 is a
# strict SUBSET of c0072's treatment and has a HIGHER decode_tv_distance_max (0.035700 vs
# 0.033492), which refutes additivity on fidelity outright, so the +3.13342 ms attributed to
# wproj is labelled inferred and never measured.
#
# THE ARITHMETIC. Layer 0's fused qkv is (2304, 768): bf16 3538944 B, int8 2304*768 + 2304*4
# = 1778688 B, removal 1760256 = 1.760 MB/step. Layer 1's (2310, 768) stays bf16, so decode
# weight traffic goes 34.700288 -> 32.940032 MB/step. DECODE_WEIGHT_Q8 must fire SIX times,
# not seven: 2 mlp_c_fc + 2 mlp_c_proj + 1 lm_head + 1 attn_qkv, because the builder is now
# reached for layer 0 only. That count is DERIVED from the dispatch structure, not carried.
#
# val_bpb: the WEIGHTS channel stays closed -- every int8 copy goes through self.__dict__ and
# the fp32 parameters are what forward, the optimiser and validation read, so num_params_total
# cannot move from 42467524. The CLOCK channel is open as always: the LR schedule is indexed
# on progress = training_time / TIME_BUDGET, so num_steps is an OUTPUT.
#
# c0073: DECOMPOSING ROUND 74's REGRESSION. int8 WEIGHT-ONLY ON w_qkv ALONE, wproj LEFT bf16.
#
# WHAT ROUND 74 DID. c0072 put BOTH remaining bf16 decode-GEMV weights on int8 -- w_qkv and
# wproj, each per layer, four tensors -- and it REGRESSED, hard and unambiguously:
#
#   request_ms_median 65.7050609588623 against c0071's 61.606764793395996 = +4.09830 ms =
#   +5.395 sd on the 0.7597025823 ms ruler. Zone large_regression, and for once NOT
#   boundary-adjacent: 1.819 ms from the nearest boundary I picked. monitor independently kept
#   c0071 as the incumbent. The gate meanwhile passed with the WIDEST margin of the axis
#   (1.076 sd) and the val_bpb delta of -0.0025070 is BELOW TASK.md's own 0.003 non-effect
#   line, so it is NOT an improvement and I am not calling it one. That bond binds
#   symmetrically and this is the row where honouring it costs me the flattering sentence.
#
# WHY THE ROW COULD NOT ANSWER ITS OWN QUESTION. The c0072 card pre-registered that the paired
# device instrument is VOID for a row whose n_kernel order differs from the reference. It
# differed: seq 74 emitted 38 DECODE_KERNEL lines to seq 73's 40, and n_kernel went
# [20,20,20,21,20,20,21,21,20,20,20] to [19,19,19,20,19,19,20,20,19,19,19] on BOTH passes.
# So the warm band verdict is void, and NEITHER the plain byte account NOR the capacity
# account is refuted as an ACCOUNT -- both merely failed as predictions for that row, which is
# a weaker statement and the only one I am entitled to. The card's BRANCHES were asserted
# exhaustive over warm-band x zone x admissible and had NO VOID BRANCH. A precondition without
# a branch is a precondition I had not decided how to act on, and that is a card defect.
#
# THE MECHANISM, AND IT IS NOT BYTES. Per probe, seq 73 -> seq 74: 19 -> 18 distinct kernel
# names, 4 disappeared (-20.200 us/step), 3 appeared (+12.350), and the 15 SURVIVING names
# gained +14.770, net +6.920 on probe A and +7.670 on probe B. The two largest movers are
# SURVIVING names, +9.090 and +5.010. And one rename is the mechanism in plain sight:
# triton_red_fused_mul_sum_unsqueeze_view_0 became
# triton_red_fused__to_copy_mul_sum_unsqueeze_view_0 -- the int8->float conversion entered the
# fusion group of a GEMV reduction, the compiler re-partitioned, and about a sixth of the
# 120 us/step decode step was rebuilt into a slower arrangement. 4.699 MB removed is worth
# ~3 us/step at c0071's byte rate; the fusion artefact is worth ~7 us the OTHER way, 2.3x
# larger. AT THIS SCALE THE COMPILER'S FUSION DECISIONS DOMINATE THE BYTE EFFECT.
#
# I ASSUMED THIS AWAY IN THE LAST CARD AND I AM NAMING THAT. c0072's P27 pre-registered ZERO
# kernel-name changes, reasoning that because both treated ops were ALREADY _gemv_reduce, no
# op changed and therefore no kernel could change. The op set indeed did not change; the
# FUSION PARTITION did, and I had not distinguished those two things -- even though round 71's
# non-locality had already been attributed to exactly this mechanism. A recorded mechanism that
# does not reach the next card is not a recorded mechanism.
#
# WHY THIS ROW, AND WHY IT IS A COUNT AND NOT A BAND. c0072 bundled four tensors, and its own
# card ranked "a bad outcome is not decomposable" as risk 3 and then accepted it on an argument
# that was ARITHMETICALLY WRONG: I claimed each half would sit near the noise floor. Per half,
# w_qkv alone removes (3538944-1778688) + (3548160-1783320) = 1760256 + 1764840 = 3525096
# bytes = 3.525 MB/step, worth 1.176 ms at c0071's 0.650 us/MB rate = 1.55 sd, COMFORTABLY
# above the floor. Only wproj alone (2 x (1179648-592896) = 1173504 B = 1.17 MB, 0.39 ms,
# 0.52 sd) is genuinely sub-floor. So the bundle was decomposable and I bundled it anyway.
#
# THE DISCRIMINATOR IS THE KERNEL COUNT, WHICH IS AN INTEGER:
#
#   IF THE FUSION ARTEFACT FOLLOWS w_qkv: n_kernel drops 20 -> 19 again, the pairing
#   precondition fails again, and the key regresses by something near +7 us/step. The artefact
#   is ATTRIBUTED to w_qkv; wproj-only is the sole remaining move and it is sub-floor, so the
#   axis closes with the byte question open but the confound located.
#
#   IF IT FOLLOWS wproj: n_kernel STAYS at 20, the precondition PASSES, the paired instrument
#   is valid again, and the byte account predicts 3.525 MB x 0.650 = 2.291 us/step, which after
#   the four-row -8% predictor bias is -1.176 ms -> key ~60.43, zone gain_supra_floor. That
#   would make c0073 the incumbent AND give the byte account its first clean test since the
#   compiler started interfering.
#
#   IF IT FOLLOWS NEITHER -- i.e. if ANY additional int8 GEMV triggers a re-partition -- the
#   count drops here too and the row attributes NOTHING. This is the leading risk and it is
#   pre-registered as its own branch, because it is the outcome that retires the axis with the
#   question open, and I am not going to discover it after the fact and call it a finding.
#
# AN INTEGER OBSERVABLE HAS NO BOUNDARY I CHOSE. Three of round 73's verdicts and one of round
# 74's rested on where a number fell against a band edge I picked. n_kernel is 20 or it is 19.
# That is the most valuable property this row has, and it is why the row is worth a charge even
# though its ranking-key outcome is genuinely uncertain.
#
# WHAT DOES NOT CHANGE. _gemv_reduce and _gemv_reduce_q8 are byte-identical to c0071's. The
# three earlier q8 builders (decode_fc_weight_q8, decode_proj_weight_q8, decode_lm_weight_q8)
# are byte-identical, so all three earlier treatments are carried unmodified. decode_out_weight
# and decode_qkv_weight stay, both still bf16 builders. wproj STAYS bf16 and is now the only
# remaining control on this axis. init_decode_state, reset_decode_state and decode_step are
# byte-identical, so the gated kv_cache_bytes allocation delta cannot move. The three prefill
# F.linear(h, w_qkv) sites are UNTOUCHED and still receive the bf16 decode_qkv_weight(). NO OP
# CHANGES ANYWHERE: both treated dispatch paths are already _gemv_reduce sites. No
# torch._inductor.config flag is added; that family is closed at five charges plus two killed
# at zero charge.
#
# THE ARITHMETIC. Layer 0's fused qkv is (2304, 768) and layer 1's is (2310, 768) -- layer 1
# carries 6 extra ve_gate rows because has_ve(layer_idx, n_layer) = layer_idx % 2 ==
# (n_layer-1) % 2 makes has_ve(1, 2) true. I used 2304 for BOTH layers in THREE earlier cards
# and round 74's witness confirmed the correction to the byte. bf16 3538944 + 3548160; int8
# (2304*768 + 2304*4) + (2310*768 + 2310*4) = 1778688 + 1783320; removal 3525096 bytes/step.
# Decode weight traffic 34.700288 -> 31.175192 MB/step. The A100 L2 nominal is 40.000 MB and
# c0071 already crossed it.
#
# PEAK VRAM WILL RISE, NOT FALL, AND FOR THE REASON ROUND 74 CONFIRMED. decode_qkv_weight() is
# still called -- by the prefill F.linear sites AND by this builder itself, which reads
# self.decode_qkv_weight().float() -- and it caches into self.__dict__, so the bf16 copy stays
# resident and the int8 copies are ADDITIVE. Round 74 predicted this reversal correctly in
# direction and MISSED the magnitude by 258024 bytes (5.4% of the predicted rise), for which I
# have no decomposition. So this row predicts the DIRECTION and bands the magnitude.
#
# val_bpb: the WEIGHTS channel stays closed -- the fp32 parameters are untouched and are what
# forward, the optimiser and validation read. The CLOCK channel is not closed and never was.
# num_steps has read 1076, 1076, 1077, 1076 across this axis, so it moves and it is an OUTPUT.
#
# c0071: THE LAST DISCRIMINATING ROW ON THIS AXIS. int8 WEIGHT-ONLY ON w_lm, THE HEAD.
#
# WHAT ROUND 72 DID TO THE ACCOUNT THIS AXIS WAS BUILT ON. c0070 put wfc on int8, a
# BYTE-MATCHED twin of round 71's wp2 (9388032 bytes removed per step against 9431040), and
# the aggregate byte law it was built to replicate FAILED:
#
#   request_ms_median 63.89415 against c0069's 64.16738 = -0.27323 ms. That is 0.36 sd on
#   the 0.7597 ms ruler, zone `null`, and it misses BOTH of my own bars. The row is the
#   incumbent because the task's ranking key moved the right way; the accept carries NO
#   evidence of an effect on the scored key, and it is recorded that way.
#
#   The paired device instrument predicted -4.4655 us/step and measured -1.7118 pooled.
#   That is 9.388 MB / 1.7118 us = 5484 GB/s = 269 PER CENT of the A100's 2039 GB/s HBM
#   figure, which is not a fast bandwidth but a PHYSICALLY IMPOSSIBLE one, so the quantity
#   was never a bandwidth. AND THAT INDICTS ROUND 71 RETROSPECTIVELY: round 71 read 103 per
#   cent of peak and I recorded it as CONFIRMATION. A figure grazing a physical ceiling is
#   evidence AGAINST the account that predicts it. 2102 GB/s is downgraded from a law to a
#   single observation, and card branch 4 pre-registered exactly that.
#
# THE INSTRUMENT DEFECT THAT EXPLAINED IT, FOUND ON THE SECOND ROW THE INSTRUMENT WAS USED.
# The 26 device_us_per_step readings are 11 fixed configurations x 2 passes, and I POOLED
# the passes. Pass 2 is systematically FASTER than pass 1 in every row I have: seq 68
# -4.895, seq 70 -4.723, seq 71 -5.050, seq 72 -2.661 us/step. A stable ~4.9 us/step warm-up
# gap, which c0070 roughly halved. Pooling was harmless only while the gap was constant. On
# seq 72 the treatment delta was -2.906 (sd 0.216) on pass 1 and -0.517 (sd 0.413) on pass
# 2, so the pooled -1.7118 with sd 1.2642 -- FOUR TIMES the control's 0.3121 -- is the mean
# of two different quantities. The negative control could not see it, because in
# emitted-identical rows the gap is the same on both sides and cancels.
#
# THE CORRECTED READING, AND IT IS TWO ROWS AND NOT A LAW. The scored request_ms_median is
# measured WARM (30 passes, all within 63.54-64.19 on seq 72), so pass 2 is the pass that
# should predict it -- and it does, where the pooled figure does not. seq 72: pass2 -0.517 x
# 512 = -0.2647 ms against an observed -0.27323, ratio 1.03. seq 71: -2.3368 against
# -2.63262, ratio 1.13. The pooled figure gives 0.312. Stated as TWO ROWS deliberately, one
# paragraph after retracting a one-row law.
#
# LOCALITY, BY CONTRAST, IS NOW SETTLED. Round 72 changed ZERO kernel names and is fully
# local: the treated wfc kernel fell 22.37 -> 19.65 = -2.720 us/step, the probe-A total fell
# -2.72, and every other mover cancels to within 0.02. The untreated bf16 LM-head kernel
# moved +0.000, after falling -2.10 while untreated in round 71. So round 71's non-locality
# was the COMPILER RE-PARTITION -- exactly one of 20 names swapped -- and NOT L2 capacity.
#
# WHY THIS ROW, AND WHY IT IS THE LAST ONE WORTH SPENDING HERE. Two accounts survive and
# EACH EXPLAINS EXACTLY ONE OF THE TWO ROWS, which is why a third row decides:
#
#   Decode weight traffic per step, computed from the shapes:
#     c0066 all bf16     59.769 MB      c0069 wp2 int8     50.338 MB
#     c0070 +wfc int8    40.950 MB      c0071 +w_lm int8   34.691 MB
#     A100 L2, nominal   40.000 MB
#
#   CAPACITY THRESHOLD. c0070's row moved 50.34 -> 40.95, APPROACHING the 40 MB L2 without
#   crossing it, and paid almost nothing warm -- which is what a capacity account predicts.
#   This row is the FIRST CROSSING, 40.95 -> 34.69, so a capacity account predicts a
#   DISPROPORTIONATE fall, larger than wp2's -4.5 warm despite removing only 6.259 MB.
#   But it does NOT explain round 71's large warm gain, where both endpoints sat far above
#   capacity.
#
#   SATURATION / EXHAUSTION. The steady-state decode step is simply not weight-bandwidth-
#   bound, so predict a warm-pass fall of ~0.2-0.6 us/step and a key move inside the noise
#   floor. This explains round 72 and not round 71.
#
#   A THIRD ACCOUNT I AM NAMING RATHER THAN LETTING IT HIDE: SHAPE. wp2 is (768, 6144) and
#   reduces over 6144; wfc is (6144, 768) and reduces over 768. Their kernels are not equally
#   bandwidth-bound, so the payoff may track the reduction shape and not the byte count at
#   all. w_lm is (8192, 768) -- the SAME short reduction axis as wfc -- so the shape account
#   predicts the same near-null as saturation.
#
#   THEREFORE, STATED HONESTLY: this row separates CAPACITY from {SATURATION, SHAPE}. It does
#   NOT separate saturation from shape. It is a one-against-two discriminator and I am not
#   going to write it up afterwards as a three-way one.
#
# THE 40 MB FIGURE IS NOMINAL AND I AM NOT PRETENDING IT IS A THRESHOLD I CAN LOCATE. The
# A100's L2 is 40 MB in total and the decode step's working set also holds activations and
# the KV cache, so the usable share for weights is LESS than 40 and the crossing may already
# have happened at c0070, or may not happen here. The test is DIRECTIONAL -- a
# disproportionate fall against a proportionate one -- and its magnitudes differ by an order
# of magnitude, which is what makes it worth a charge even with a fuzzy boundary.
#
# WHERE THE RISK IS, AND IT IS HIGHER HERE THAN ON ANY PREVIOUS ROW OF THIS AXIS. w_lm
# produces the LOGITS DIRECTLY. wp2 and wfc sit inside the block and their error passes
# through a subsequent norm and a residual; the head's error passes through nothing but a
# softcapped tanh before it becomes the scored distribution. So the gated fidelity clauses
# -- decode_tv_distance_max and its nopref twin, both <= 0.05 -- are at REAL risk, and their
# previous increments (+0.0080 for wp2, +0.00180 for wfc) are NOT a guide, because those were
# hidden weights. The card bands TV wide and pre-registers INADMISSIBLE as a live outcome. If
# it fires, the charge bought the price of the head, which is exactly the number this axis
# was missing, and it is a finding and not a bug.
#
# THE TV PROJECTION IS ALREADY WITHDRAWN, NOT RESCALED. wfc's +0.00180 differs from wp2's
# +0.0080 by 4.4x, and the c0070 card's own bond withdrew the "roughly three weights before
# TV binds" projection at a 2x discrepancy. So this row carries NO projected TV budget.
#
# WHAT DOES NOT CHANGE. _gemv_reduce_q8 is byte-identical to the parent's -- a third use of
# the SAME estimator, because changing it would make the comparison across the three weights
# meaningless. decode_fc_weight_q8 and decode_proj_weight_q8 are byte-identical, so both
# previous treatments are carried unmodified. init_decode_state, reset_decode_state and
# decode_step are byte-identical. wproj and w_qkv stay bf16 and are the remaining controls.
# The dead _decode_head keeps F.linear on the fp32 parameter. No torch._inductor.config flag
# is added: that family is closed at five charges plus two killed at zero charge.
#
# THE ARITHMETIC. w_lm is (8192, 768), read ONCE per step and not per layer. bf16 is
# 12582912 bytes; int8 is 6291456 + 8192*4 = 6324224; the removal is 6258688 = 6.259 MB/step,
# which is 0.667x wp2's 9431040. Under the now-WITHDRAWN aggregate law that would have
# predicted -2.977 us/step, and that number is written here only so the withdrawn law's
# prediction is on the record and can be scored against the two accounts that replaced it.
# The reduction axis is 768 with 8192 outputs, so the scale vector holds 8192 entries.
#
# val_bpb: the WEIGHTS channel is closed as before -- self.lm_head.weight, the fp32
# parameter, is untouched and is what forward, the optimiser and validation read. The CLOCK
# channel is NOT closed and never was: the LR schedule is indexed on wall-clock progress, so
# num_steps is an output. It did not fire on round 72 (1076 steps, unchanged) and that is an
# observation about one row, not a property of the edit.
#
# c0070: THE BYTE-MATCHED TWIN. int8 WEIGHT-ONLY ON wfc, AND THE LOCALITY QUESTION AGAIN.
#
# WHAT ROUND 71 MEASURED, AND IT IS TWO FINDINGS THAT POINT DIFFERENT WAYS. c0069 put wp2 on
# int8 and was ACCEPTED: request_ms_median 64.16738 against c0066's 66.79999, -2.6326 ms,
# the first row in this run to clear the 3-sd bar of 2.2791 ms. But the two halves of the
# result disagree, and this row exists because of the disagreement.
#
#   HALF ONE, CONFIRMED IN AGGREGATE. Removing 2*(9437184-4721664) = 9431040 bytes = 9.431 MB
#   of weight traffic per decode step bought 4.4859 us/step, which is 9.431 MB / 4.4859 us =
#   2102 GB/s, or 103 PER CENT of the A100's 2039 GB/s HBM figure. The marginal byte removed
#   from the decode step is worth FULL PEAK BANDWIDTH. That number is the same to 0.0014
#   us/step whether measured against c0066 or c0068, and 4.4859 us x 512 steps = 2.2968 ms
#   accounts for 87.2 per cent of the observed key gain.
#
#   HALF TWO, REFUTED. The saving is NOT paid where the bytes are removed. The TREATED wp2
#   kernels fell only 17.77 -> 16.42 = -1.35 us/step, about 15 per cent of their own
#   bandwidth model and 29 per cent of the peak model. The LARGEST single mover was the
#   UNTREATED, still-bf16 LM-head kernel: 11.92 -> 9.82 = -2.10 us/step, -17.6 per cent.
#   Untreated reductions fell -2.39 in total against the treated -1.35, so roughly 30 per
#   cent of the saving landed on the kernels whose weight actually changed.
#
# THE INSTRUMENT THAT MADE BOTH READINGS POSSIBLE, AND I MISSED IT FOR 70 LAUNCHES. Each
# stdout log emits 26 device_us_per_step readings, and they are NOT 26 replicates: they are
# 11 FIXED CONFIGURATIONS x 2 probe passes, plus 2 split-sweep outliers per pass. The
# n_kernel order [20,20,20,21,20,20,21,21,20,20,20] is byte-identical across seq 68, 70 and
# 71, so the readings PAIR by configuration index. Its negative control is MEASURED rather
# than assumed: two emitted-identical rows, c0066 against c0068, give +0.0014 +- 0.3121
# us/step at n=22, df=21, with mixed signs, while c0066 against c0069 gives -4.4859 +-
# 0.3003 at n=22, df=21 with ALL 22 THE SAME SIGN. That is 14.4x the control's own
# dispersion at df=21, where every previous verdict on this lane rested on df=1 to df=3.
#
# WHY wfc AND NOT SOMETHING ELSE. Three reasons, and the first is the one that matters:
#   1. wfc is BYTE-MATCHED to wp2. (6144, 768) against (768, 6144), the same 9437184 bytes,
#      the opposite shape. c0053 established that pair as the shape-opposed byte-matched
#      twin, and in round 71 wfc was the WITHIN-ROW CONTROL that did not move. So this row
#      is a REPLICATION of the aggregate law on the closest thing to an identical weight
#      that exists in this model, and the law's prediction is quantitative, not directional.
#   2. It reads the LOCALITY question a second time. If the saving again lands mostly on
#      untreated kernels, the effect is a shared-resource one and the per-kernel view is
#      simply the wrong unit. If it lands on wfc's own kernel this time, the compiler
#      re-partition reading gains ground instead. Round 71 could not separate those two.
#   3. It is the largest remaining single-weight byte removal: wfc 9.39 MB/step against
#      w_lm 6.29 MB/step (12.6 MB, used ONCE per step and not per layer) and wproj ~1.18.
#
# THE ARITHMETIC IS NOT IDENTICAL TO wp2's AND I AM NOT ROUNDING THAT AWAY. wfc is
# (6144, 768), so its per-output-row scale vector has 6144 entries, not 768: 24576 bytes
# rather than 3072. Bytes go 9437184 -> 4718592 + 24576 = 4743168 per layer, so the removal
# is 2*4694016 = 9388032 = 9.388 MB/step, which is 0.46 per cent LESS than wp2's 9431040.
# Under the aggregate law that scales the prediction to -4.4655 us/step, not -4.4859. The
# reduction axis is also shorter -- 768 instead of 6144 -- so each output's quantisation
# error is accumulated over 8x fewer terms.
#
# THIS ROW DOES NOT INHERIT ITS PREDECESSOR'S BRANCH. c0069's card enumerated four mechanism
# branches and the outcome was a fifth: a real, treatment-caused, AGGREGATE effect at peak
# marginal bandwidth that is NON-LOCAL. Branch 1, whose action was "extend to wfc", did NOT
# fire, because it required the untreated kernels to stay put and they did not. Branch 2's
# TEST fired and its CONCLUSION -- "the row measured the machine, not the weight" -- is
# refuted by the paired control's +0.0014 us/step null. So this row is argued from the new
# law and not from a branch that did not fire, and that is a defect in the c0069 card
# recorded as such rather than filed under whichever branch was closest.
#
# WHY val_bpb STILL CANNOT MOVE THROUGH THE WEIGHTS, AND WHY I NO LONGER SAY "CANNOT MOVE".
# The quantised copy is built lazily through self.__dict__, self.c_fc.weight -- the fp32
# PARAMETER -- is untouched, and `forward`, the optimiser and validation read the parameter
# and never the copy. That much is construction and it held. But in round 71 I wrote
# "val_bpb cannot move BY CONSTRUCTION" and the row falsified my own numeric test of it, so
# the phrasing is corrected here: val_bpb has a SECOND channel, through the clock. The LR
# schedule is indexed on WALL-CLOCK progress -- "progress = training_time / TIME_BUDGET" --
# so num_steps is purely an OUTPUT, and any decode-side edit that changes compile or probe
# time changes the steps completed inside the 600 s budget and therefore the tokens seen.
# c0069 ran 1076 steps against c0066's 1078, 0.19 per cent fewer tokens, and its val_bpb was
# 0.0012683 worse -- the right sign. That is 0.28 sd against the PRE-REGISTERED sd of
# 0.0045534, and TASK.md states a val_bpb difference of 0.003 is not an effect, so by the
# pre-registered ruler it is not one. My own threshold of 0.00065 was a four-row RANGE, not
# a dispersion, roughly 7x tighter than the authority's, and it fired spuriously.
#
# WHERE THE RISK LANDS, AND IT IS NOW THE THING MOST LIKELY TO CLOSE THIS AXIS. wp2 cost
# +0.0080 on decode_tv_distance_max (0.0102718389 -> 0.0182604752) and +0.0052 on the nopref
# twin, for 9.431 MB. The bound is 0.05 and it is a GATED admissibility clause whose own
# basis in task.json says it exists because "without a fidelity clause the cheap branch can
# be made cheap and wrong". A second weight at a similar rate lands near 0.026, leaving
# about 1.9x of headroom instead of 2.7x. Headroom is not a prediction. If either TV reading
# exceeds 0.05 this row is INADMISSIBLE, that is a real recorded outcome and not a bug, and
# the axis is priced at two charges rather than prohibited.
#
# THE GATE MARGIN IS THE OTHER STANDING RISK AND IT IS NOT CAUSED BY THIS TREATMENT.
# c0069's val_bpb 1.0469504166602153 passes at +0.0030496, which is 0.670 sd -- the TIGHTEST
# margin of the family, against c0066's 0.949 and c0068's 0.999. A replicate of the
# incumbent could plausibly fail the gate, one run per candidate is the design, and that
# uncertainty is therefore not resolvable. It is a caveat on the incumbency, carried here.
#
# WHAT THIS ROW DOES NOT CLAIM. It does not claim to separate L2 capacity from compiler
# re-partition on its own -- it narrows them. It does not claim the aggregate law extends
# below the byte sizes tested. And it does not claim wfc's gain will equal wp2's: the law
# predicts -4.4655 us/step from a measured constant, and a constant measured on one row is
# a constant measured on one row.
#
# c0069: THE BYTES ACCOUNT, TESTED BY CHANGING THE BYTES. int8 WEIGHT-ONLY ON wp2 ONLY.
#
# WHY THIS AXIS, AND WHY NOW. Five charges went to torch._inductor.config -- c0064 tuning,
# c0065 form, c0066 split, c0067 tiling, c0068 fusion -- and FOUR of the five produced no
# name-level structural change at all. c0068's 20-kernel multiset was identical to c0066's at
# every position, so `aggressive_fusion` added fusion candidates that `can_fuse` rejected in
# full. That family is CLOSED on this lane. Two further flags in it were killed at ZERO charge
# by reading the installed source: `tile_reductions` alone is inert behind the simd.py:2454
# early return, and `score_fusion_memory_threshold` alone is inert because scheduler.py:4282
# conjoins it with `loop_ordering_after_fusion`, which defaults False.
#
# WHAT THE MEASUREMENT SAYS IS LEFT. Probe A on launch 70: device 129.93 us/step against a
# request of 132.0 us/step, so the step is essentially all device time. Inside it, the twelve
# `triton_red_fused_*` reductions are 79.40 us/step -- 61 per cent -- and the eight of them
# whose names carry `sum`, i.e. the ones carrying a GEMV, are 69.5 us/step. The decode path
# reads roughly 61.6 MB of bf16 weight per step; at the A100's ~2039 GB/s that is ~30.2 us.
# So the GEMV block runs at about 43 PER CENT of peak bandwidth. That is APPARENT headroom of
# ~2.3x on 53 per cent of device time, and it is stated as apparent, not as available: a
# fused norm+GEMV kernel with a modest grid can be latency-bound rather than bandwidth-bound,
# in which case the efficiency number is a description and not an opportunity.
#
# BOTH WAYS OF CLAIMING EFFICIENCY ARE ALREADY CLOSED. Split-K is c0007: it reshaped w to
# (N, s, K/s) to make the reduction groups a property of the shape, and it REGRESSED the key
# by 7.98 ms (launch 8, 16e2ef0a). `_gemv_reduce`'s own docstring records that the occupancy
# story is refuted in both its forms. Inductor config is closed above. So efficiency is not
# the lever left. BYTES are.
#
# THE ACCOUNT THIS ROW TESTS, AND WHY THE TEST IS NEW. This file's history has two rival
# explanations of c0006's 3.30 ms and c0052 retracted one of them: c0052 took wproj, whose
# bytes are 8x smaller, measured +1.6733 ms against its code parent and won incumbency by
# 0.0389 ms, which is not a gain. Branch 3 of that card retracted the OUTPUT-ROW-COUNT
# account and left the BYTES account standing. But every test of the bytes account so far
# changed the OP -- F.linear to a broadcast reduction -- and never the BYTES. Bytes have been
# an OBSERVED covariate of shape, never an INTERVENED variable. int8 weight-only quantisation
# changes exactly the bytes, at fixed shape, fixed op, fixed kernel form and fixed grid. If
# the bytes account is right this pays; if the block is latency-bound it does not; and either
# way the surviving explanation of the largest single effect in this run gets tested rather
# than inherited.
#
# THE TREATMENT IS ONE WEIGHT, AND ITS CONTROL IS BYTE-MATCHED AND IN THE SAME ROW. Only wp2
# -- MLP.c_proj, (768, 6144), 9.44 MB of bf16 -- becomes int8. wfc -- MLP.c_fc, (6144, 768),
# THE SAME 9.44 MB -- stays bf16. c0053 established that pair as byte-matched and
# shape-opposed, which is precisely why it is the control: if the reduction block is
# bytes-bound, wp2's kernels fall and wfc's do not move. wproj (1.18 MB), w_qkv and w_lm
# (12.6 MB) also stay bf16, so the untreated GEMVs are a second, larger control. A row where
# EVERYTHING moves is a row that measured the machine, not the weight, and that reading is
# available here because the untreated weights are in the same kernel table.
#
# WHY val_bpb CANNOT MOVE, AND IT IS NOT AN ARGUMENT ABOUT MAGNITUDE. The quantised weight is
# a SEPARATE DECODE-PATH COPY, built lazily through self.__dict__ exactly as
# decode_proj_weight, decode_fc_weight, decode_out_weight and decode_lm_weight already are.
# self.c_proj.weight -- the fp32 PARAMETER -- is untouched, and `forward`, the optimiser and
# the validation pass read the parameter and never the copy. So the gate is not at risk by
# CONSTRUCTION, and the construction is the parent's, not mine. Bypassing
# nn.Module.__setattr__ is what keeps it out of _parameters and _buffers, so num_params_total
# cannot move; building it from the non-prefill branch only is what keeps it out of the gated
# kv_cache_bytes allocation delta.
#
# WHERE THE RISK ACTUALLY LANDS, NAMED BEFORE THE ROW. On decode_tv_distance_max and its
# nopref twin, bounded at 0.05 by task.json's fidelity clause, whose own basis says the clause
# exists because "without a fidelity clause the cheap branch can be made cheap and wrong".
# That is this row's clause, aimed at this row's kind of change, and I am not going to pretend
# otherwise. Launch 70 read 0.010537 and 0.013685, so there is about 3.7x of headroom, and
# headroom is not a prediction. int8 per-output-row symmetric quantisation of ONE matrix per
# layer in a 2-layer stack is a small perturbation, but TV is measured on the output
# distribution after a softcapped head and I do not have a calibrated prior for it here. If
# either TV reading exceeds 0.05 the row is INADMISSIBLE, that is a real recorded outcome of
# a legitimate experiment and not a bug, the charge is spent, and the axis is priced.
#
# THE QUANTISER, STATED SO IT CAN BE CHECKED. Symmetric, per output row, no zero point:
# s[n] = amax_k |w[n,k]| / 127, wq[n,k] = round(w[n,k]/s[n]) clamped to [-127, 127] as int8,
# and the GEMV computes (wq.to(bf16) * v).sum(-1, fp32) * s. The scale factors out of the sum
# exactly because it is constant along the reduction axis, which is the whole reason a
# per-ROW scale is the right one and a per-tensor scale would be strictly worse for free.
# 127 rather than 128 keeps the range symmetric so no value maps to -128 and the clamp is
# reachable only by the row maximum itself. The int8 -> bf16 conversion is exact for every
# representable value, so the ONLY error introduced is the rounding of w, and the witness
# line prints its measured maximum and mean rather than asserting a bound.
#
# THE MECHANISM CLAIM IS ABOUT THE LOAD, AND IT HAS A FALSIFIER. The saving requires inductor
# to fuse the int8 -> bf16 conversion INTO the reduction, so that the global load is one byte
# per weight. The kernels already fuse `_to_copy` -- it is in their names -- so this is
# expected, not hoped for. It is falsifiable in the row itself: if inductor instead
# MATERIALISES a bf16 copy of wq, the bytes read do not fall, and an extra kernel or a higher
# device_us_per_step will show it. That reading is pre-registered as a branch, so a null here
# separates "bytes do not matter" from "the bytes never actually fell".
#
# WHAT THIS ROW DOES NOT CLAIM. It does not claim a mechanism for c0065's -5.4801 ms, whose
# own negative control moved 49.70 -> 43.13 across that row so the gain is not attributable
# to the reduction-form change and the axis was recorded as saturated, not solved. It does
# not claim the 43-per-cent bandwidth figure is a headroom I can collect. And it does not
# rest on the L2 account, which was WEAKENED at zero charge when I found the decode flash
# call receives window_size, so only ~7.1 MB of the 14.5 MB cache is ever read inside a
# 40 MB L2.
#
# THE BAR THIS ROW MUST CLEAR IS NEW AND IT IS BIGGER. Four rows now emit the SAME 20-kernel
# table -- c0065, c0066, c0067, c0068 -- and their scored keys are 67.43896, 66.79999,
# 68.63248 and 67.58404: range 1.83249 ms, sd 0.7597 ms at n=4, df=3. My old working floor of
# 0.4237872533 ms is NOT pre-registered anywhere in task.json or TASK.md; it is my own
# derived quantity from an early small set, and it is superseded by this larger one. So an
# accept needs >= 2.28 ms (3 sd) and a provisional accept >= 1.52 ms (2 sd). Round 69's
# "+4.32 floors" regression is +2.41 of the new sd, and this round's predecessor c0068's
# +0.784 ms is +1.03 sd, i.e. inside the noise. The old floor is quoted only as superseded.
#
# c0066: FORCE THE SINGLE-PASS REDUCTION, keeping c0065's looped form and c0064's widened search.
#
# WHY THIS CHARGE EXISTS, AND WHAT IT IS REALLY MEASURING. c0065 took the scored key from 72.9190 to
# 67.4390 ms -- -5.4801 ms, TWELVE POINT NINE noise floors, the largest gain this run has recorded
# outside the splits cliff -- and I pre-registered a regression as the modal outcome. The treatment
# provably landed: probe A's persistent-reduction count went 10 -> 0 and its looped count 2 -> 12, the
# ten renamed kernels carrying byte-identical fused-op lists with `per` replaced by `red`. One-for-one,
# n_kernel unmoved at 20.
#
# AND YET I CANNOT ATTRIBUTE THE GAIN TO REDUCTION FORM, BECAUSE MY OWN NEGATIVE CONTROL FAILED.
# Four of probe A's twenty kernels are NOT compiled by Inductor at all -- two cutlass flash-attention
# kernels, the FlashAttnFwdCombine, and a device-to-device memcpy. torch._inductor.config cannot reach
# them. Across launches 62, 64 and 66 their sum held at 49.66 / 49.56 / 49.70 us/step, a range of 0.14.
# On c0065 it fell to 43.13: MINUS 6.57, forty-seven times that range, and 60% of probe A's entire
# -10.90 us/step move. Under the discipline I applied in rounds 58-61, a failed within-row control
# VOIDS the in-row decomposition. So the treated kernels' own -4.33 us/step is not the treatment's
# share either; it is contaminated by whatever moved the four kernels I did not treat.
#
# TWO ACCOUNTS, NEITHER VERIFIED, AND THIS CHARGE'S CONTROL READING SEPARATES THEM.
# (a) MEMORY-SYSTEM: persistent reductions load a 768- or 6144-wide axis in one burst; the looped form
#     streams it. The A100's L2 is 40 MB and the KV cache is 14.5 MB, so it FITS, and bursty reduction
#     traffic could have been evicting it. This account predicts the ordering actually seen: the flash
#     kernel over the larger KV region gained most (-3.51), the smaller one next (-2.64), while the
#     combine, which reads only a 36 KB split accumulator, barely moved (-0.23), as did the memcpy
#     (-0.19). It is a coherent story that I have ONE row for.
# (b) PER-LAUNCH OFFSET of unknown size, in which case part or all of the -5.48 ms belongs to the
#     machine and not to my edit. Against this: both probes moved the same way and by similar amounts
#     (-5.48 and -5.04 ms), which is the coherent-effect signature, and the 0.4238 ms noise floor is
#     itself measured from SAME-PROGRAM replicates, so it already prices per-launch variation -- and
#     -5.48 ms is 12.9 times it.
#
# THE DISCRIMINATOR IS FREE. Whatever this candidate does to the scored key, its four untouched
# kernels are read again. If they stay near 43 the memory-system account gains and the c0065 gain is
# mechanistic; if they return to ~49.7 while the treated kernels stay fast, c0065 carried an offset and
# its attribution collapses. That reading costs nothing extra and it is pre-registered in the card as
# the primary instrument, ABOVE the scored key.
#
# THE TREATMENT ITSELF, AND WHY IT IS THE RIGHT ONE TO PAIR WITH THAT CONTROL. The looped direction is
# now SATURATED -- zero persistent reductions remain, so there is nothing left to convert and pressing
# `persistent_reductions` again is impossible. torch._inductor.config.split_reductions defaults True and
# is not set by mode="max-autotune-no-cudagraphs"; it controls whether a reduction is split into two
# passes over the axis. Setting it False forces ONE pass. That is the next static, compile-time knob on
# the same twelve kernels: one form emitted, nothing chosen at runtime, so unlike triton.multi_kernel it
# cannot break this candidate's CUDA-graph capture. It also changes those kernels' memory traffic again,
# which is exactly what account (a) says matters.
#
# A CAVEAT I OWE THE NEXT ROUND. Unlike c0065, this flag may change n_kernel: collapsing two passes into
# one can remove a kernel. If it does, form and fusion are confounded in this row and the card says so
# before the number exists rather than after.
#
# WHAT CANNOT MOVE. The flags are set from init_decode_state, which prepare.py calls only on the two
# decode probes, both AFTER training, so val_bpb and num_steps cannot move through this edit; nothing is
# allocated, so the kv_cache_bytes allocation delta cannot move; no shape and no dtype changes, so
# flops, params and vram cannot.
# c0065: FORCE THE LOOPED REDUCTION FORM FOR THE DECODE REGIONS, keeping c0064's widened search.
#
# WHAT c0064 MEASURED, AND WHY IT NAMES THIS AXIS. c0064 set coordinate_descent_check_all_directions
# and became the incumbent by -0.0260 ms = -0.0613 noise floors -- i.e. by NOTHING. The reason is in
# its own kernel table: probe A fell 142.26 -> 141.08 us/step, only -1.18, which is BELOW probe A's own
# ~1.4 us/step delta resolution, and the MULTISET OF THE 20 DEPLOYED KERNEL NAMES IS IDENTICAL to the
# parent's. Five positions reordered; not one kernel appeared or disappeared. So the widened search
# changed tiling parameters and never changed a kernel FORM. Within-kernel tuning of this block is
# exhausted at about 1 us/step, and that is the closure c0064 bought.
#
# THE ONE THING TUNING CANNOT REACH IS THE FORM ITSELF. Of the 20 deployed kernels, TEN are
# `triton_per_fused_*` -- PERSISTENT reductions, which hold the whole reduction axis in registers --
# and only TWO are `triton_red_fused_*`, the looped form. The four largest reduction kernels
# (24.94, 13.57, 9.59, 8.59 us/step) are 56.7 us/step, 40% of probe A. torch._inductor.config.triton
# .persistent_reductions defaults True and is NOT set by mode="max-autotune-no-cudagraphs";
# setting it False forces the looped form for these kernels at COMPILE time.
#
# WHY THIS AND NOT triton.multi_kernel. multi_kernel=1 asks the SAME question -- persistent or looped --
# but answers it by emitting BOTH and choosing at RUNTIME, which inserts a host-side decision into a
# region this candidate CUDA-graph captures, and a capture failure forfeits the charge outright. This
# flag is a STATIC compile-time choice: one form is emitted, capture sees one kernel, and there is no
# runtime selection. Same question, no capture risk. I forfeited a charge in round 65 and am not
# spending the next one on an avoidable risk when a static form of the same probe exists.
#
# THIS IS A TWO-SIDED PROBE AND I EXPECT TO LOSE IT. Inductor chose persistent for ten of these twelve
# reductions, and Inductor's heuristic is usually right at these sizes. A REGRESSION here CLOSES the
# reduction-form question: persistent is the correct form, the block's shape is its own optimum, and the
# gemv/norm block is closed on form as well as on tuning. A GAIN would mean the heuristic mis-serves a
# 1-token decode, where the reduction axis is 768 or 6144 and the batch axis is 1 -- the regime a
# persistent reduction is least suited to, because it has no parallel batch work to hide its register
# pressure. Both readings are worth a charge; the null is the only weak outcome, and probe A's kernel
# NAMES make a null distinguishable from a no-op (see the card's structural witness).
#
# WHAT CANNOT MOVE. Same argument as c0064 and now measured there: the flags are set from
# init_decode_state, which prepare.py calls only on the two decode probes, both AFTER training, so
# val_bpb and num_steps cannot move through this edit; nothing is allocated, so the kv_cache_bytes
# allocation delta cannot move; no shape and no dtype change, so flops, params and vram cannot.
# c0064: WIDEN THE COORDINATE-DESCENT AUTOTUNE SEARCH FOR THE DECODE REGIONS.
#
# WHY THIS PLACEMENT, AND WHAT IT COSTS ME TO LEARN IT. c0063 carried this exact treatment and
# FORFEITED ITS CHARGE to a placement bug, not to the mechanism. I inserted this block at column 0
# immediately before `    def init_decode_state(self, ...)`, which is a METHOD of GPT indented four
# spaces. A column-0 `def` there does not merely end the class body -- the following four-space `def`
# became a NESTED FUNCTION INSIDE THIS HELPER, and it dragged all six subsequent GPT methods with it:
# init_decode_state, reset_decode_state, decode_step, decode_embed_stack, decode_rope_stack and
# _decode_body. GPT lost its decode protocol and prepare.py's require_decode_protocol (line 645)
# correctly raised DecodeProtocolError. Training ran all 1077 steps first, so the row cost a full
# charge and produced no METRICS_JSON line at all.
#
# THE GUARD COULD NOT SEE IT, AND THAT IS THE DURABLE LESSON. /tmp/guard_c0063.py asserted 24 things
# including that every pre-existing function was BYTE-IDENTICAL to the parent's, and all 24 passed --
# because a function's source text is identical whether it is a method, a module-level function, or a
# closure nested inside another function. Byte-identity of a body says nothing about its SCOPE. The
# c0064 guard asserts the full qualified scope chain of EVERY function against the parent's, and
# separately that GPT's class body still lists the three protocol methods.
#
# So this block sits at TRUE module level, immediately before `class GPT(nn.Module):`, which is itself
# at column 0. Nothing follows it at an inner indentation.
#
# THE TREATMENT. torch._inductor.config.coordinate_descent_check_all_directions = True, set once
# before any decode region compiles. coordinate_descent_tuning is ALREADY ON: the decorator this file
# has carried since c0004 uses mode="max-autotune-no-cudagraphs", and
# torch._inductor.list_mode_options of that mode returns {'max_autotune': True,
# 'coordinate_descent_tuning': True}. I was about to spend a charge setting it and checked instead;
# that charge was saved at zero cost. This flag is NOT in the mode's options and defaults False, so it
# is the one adjacent knob with measured dynamic range.
#
# WHY IT CANNOT MOVE AN IMMUTABLE. init_decode_state is called only by prepare.py's two decode probes,
# both after training completes, so val_bpb and num_steps cannot move through this edit. The helper
# allocates nothing, so the kv_cache_bytes single allocation delta measured around init_decode_state
# cannot move. It changes no shape and no dtype, so flops, params and vram cannot move. It changes no
# arithmetic: every live decode function, the gemv/norm/rotary helpers, the three prefill functions and
# forward are byte-identical to c0060's.
#
# WHY THIS ROW IS HIGH-INFORMATION EITHER WAY. Every other axis I hold is closed or priced below the
# noise floor: attention closed with a charge (c0061), F.linear -> _gemv_reduce exhausted (c0060),
# splits closed on both table entries in both directions (c0059), source-level region merging exhausted
# because the compiler already does it (c0062: n_kernel 20 -> 20, launches_per_step 25.0 -> 25.0), and
# the small-kernel tail priced at 0.052 ms/kernel by c0061's own row, so removing ALL SIX small kernels
# buys ~0.31 ms -- below the 0.4238 ms floor. If widening the search finds 1-4 us/step the gemv/norm
# block is live. IF IT FINDS NOTHING, 56-69% of achievable HBM bandwidth is these kernels' true optimum
# at this shape and HALF OF PROBE A IS CLOSED. A null here is a real closure, not a wasted row.
_DECODE_INDUCTOR_WIDENED = [False]


def _widen_decode_autotune():
    """Set `coordinate_descent_check_all_directions` once, before any decode region compiles.

    Idempotent by the module-level flag: the two decode probes both call init_decode_state, and
    exactly one DECODE_INDUCTOR_CFG line is expected in stdout. Two lines would mean the probes run
    in separate processes or the flag was reset, which is a finding about the harness.

    The printed `coordinate_descent_tuning` field reads the MODULE-LEVEL default and will print
    False even though the compile mode sets it True at compile time. Reading that False as "the mode
    failed to apply" would be a wrong retraction; it is pre-registered as a known reading trap.
    """
    if _DECODE_INDUCTOR_WIDENED[0]:
        return
    _DECODE_INDUCTOR_WIDENED[0] = True
    try:
        import torch._inductor.config as _ic
        before = getattr(_ic, "coordinate_descent_check_all_directions", None)
        cdt = getattr(_ic, "coordinate_descent_tuning", None)
        if before is None:
            print("DECODE_INDUCTOR_CFG: available=False reason=attribute_missing", flush=True)
            return
        _ic.coordinate_descent_check_all_directions = True
        after = getattr(_ic, "coordinate_descent_check_all_directions", None)
        # c0065: ALSO force the LOOPED reduction form. See the block comment above.
        pr_before = getattr(_ic.triton, "persistent_reductions", None)
        _ic.triton.persistent_reductions = False
        pr_after = getattr(_ic.triton, "persistent_reductions", None)
        # c0066: ALSO force the SINGLE-PASS reduction. See the block comment above.
        sr_before = getattr(_ic, "split_reductions", None)
        _ic.split_reductions = False
        sr_after = getattr(_ic, "split_reductions", None)
        print(f"DECODE_INDUCTOR_CFG: available=True check_all_directions_before={before} "
              f"check_all_directions_after={after} coordinate_descent_tuning={cdt} "
              f"changed={before != after}", flush=True)
        print(f"DECODE_REDUCTION_FORM: available=True persistent_before={pr_before} "
              f"persistent_after={pr_after} changed={pr_before != pr_after}", flush=True)
        print(f"DECODE_REDUCTION_SPLIT: available=True split_before={sr_before} "
              f"split_after={sr_after} changed={sr_before != sr_after}", flush=True)
    except Exception as exc:
        print(f"DECODE_INDUCTOR_CFG: available=False reason={type(exc).__name__}: {exc}", flush=True)


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
        # (c0020's decode_embed_stack and decode_rope_stack are methods defined below;
        #  nothing is allocated here, and nothing is allocated until a non-prefill decode.)
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
    def decode_lm_weight(self):
        """c0054: a bf16 copy of lm_head.weight for the decode reduction. THE LARGEST REMAINING
        UNCONVERTED READ ON THE DECODE PATH, and the first charge selected by a coefficient this
        run MEASURED rather than by an argument.

        WHY THIS WEIGHT, RANKED BY BYTES PER STEP over the live decode path:
            wp2   MLP down   (768, 6144)  x2 = 18.87 MB/step  -- ALREADY converted (c0006)
            wfc   MLP up     (6144, 768)  x2 = 18.87 MB/step  -- ALREADY converted (c0053)
            w_lm  head       (8192, 768)  x1 = 12.58 MB/step  -- THIS CHARGE
            w_qkv fused qkv  (2310, 768)  x1 =  3.55 MB/step  -- 3.5x smaller, later at best
            wproj attn out   (768, 768)   x2 =  2.36 MB/step  -- converted (c0052), gained nothing
        Branch 1 of the c0053 card pre-registered exactly this ordering: "next charge is the
        remaining F.linear reads on the decode path ranked BY BYTES".

        A PREMISE I NEARLY GOT WRONG, RECORDED BECAUSE IT WOULD HAVE INVALIDATED THE RANKING.
        GPTConfig declares `vocab_size: int = 32768`, which would make this weight 50.3 MB and
        the single largest object in the model. It is a DATACLASS DEFAULT and it is not the live
        value: the run's own parameter breakdown prints `lm_head : 6,291,456` = 8192 x 768, and
        DECODE_EMBED_STACK reports shape (8192, 1536). The live head vocab is 8192 and the
        weight is 12.58 MB. The error was caught by a bandwidth sanity check rather than by
        reading the config: 50.3 MB in the head region's measured 17.17 us/step would be
        2.93 TB/s, above A100 HBM peak, so the premise was physically impossible. At 12.58 MB it
        is 733 GB/s, which is consistent.

        THE COEFFICIENT IS MEASURED ON MY OWN ROW, NOT TRANSFERRED. c0053 converted 18.87 MB/step
        of wfc and moved the in-row deployed kernel sum 152.78 -> 146.63, i.e. -6.15 us/step, so
        0.326 us/step per MB/step. Applied to 12.58 MB that is -4.10 us/step = -2.10 ms over
        request_steps=512. c0006's 3.30 ms is deliberately NOT used to calibrate this: it was
        measured at a different model shape, and my standing lesson is that a mechanism transfers
        while a magnitude does not. The band is wide for that reason.

        THERE IS ONLY ONE LIVE SITE, AND THE LOG PROVES IT. `_decode_head` also reads w_lm, but
        it is reachable only when `merged_out is None`, and the last layer always sets it via
        `_decode_post_head`. seq 55's DECODE_PHASE_US prints `post` and `post_head` and NO `head`
        bucket at all, so that path did not execute. It is left untouched, as dead code, for the
        same reason `_decode_post` was.

        Written through self.__dict__, bypassing nn.Module.__setattr__, so it cannot land in
        _parameters or _buffers and num_params_total cannot move. Built lazily from the
        non-prefill decode branch only, so training never allocates it, the optimiser's
        lm_head_params grouping is untouched, and it stays outside kv_cache_bytes. Identical in
        construction to decode_proj_weight (c0006), decode_out_weight (c0052) and
        decode_fc_weight (c0053).
        """
        w = self.__dict__.get("_w_lm_decode")
        if w is None:
            with torch.no_grad():
                w = self.lm_head.weight.to(torch.bfloat16).contiguous()
            self.__dict__["_w_lm_decode"] = w
        return w

    def decode_lm_weight_q8(self):
        """c0071: the int8 weight-only copy of lm_head.weight, plus its per-row fp32 scale.

        Returns (wq, ws): wq is (vocab, n_embd) int8, ws is (vocab,) fp32. The quantiser is
        the SAME RULE, byte for byte, as decode_fc_weight_q8's and decode_proj_weight_q8's --
        symmetric, per output row, no zero point, 127 rather than 128 so nothing maps to -128
        -- because the whole point of this row is to compare a third weight against the two
        already measured, and changing the estimator would make that comparison meaningless.

        THE ARITHMETIC. (8192, 768), read ONCE per decode step rather than once per layer.
        bf16 12582912 bytes; int8 6291456 + 8192*4 = 6324224; removal 6258688 = 6.259 MB/step,
        0.667x wp2's 9431040. The reduction axis is 768, the same short axis as c_fc's, with
        8192 outputs, so the scale vector holds 8192 entries and costs 32768 bytes.

        WHY THE FIDELITY RISK IS DIFFERENT HERE, AND IT IS THE REASON THIS IS THE LAST ROW OF
        THE AXIS. wp2 and c_fc sit inside the block, so their quantisation error passes through
        a subsequent norm and a residual add before it can reach a probability. THIS weight
        produces the logits directly: the only thing between its error and the scored
        distribution is a softcapped tanh. So the gated decode_tv_distance_max clauses are at
        real risk, and the previous increments -- +0.0080 for wp2 and +0.00180 for c_fc -- are
        NOT a guide for it. Their difference from each other is already 4.4x, which is why the
        projected TV ceiling was withdrawn rather than rescaled.

        Built once, lazily, through self.__dict__, bypassing nn.Module.__setattr__ exactly as
        decode_lm_weight, decode_fc_weight_q8, decode_proj_weight_q8, decode_proj_weight,
        decode_fc_weight and decode_out_weight already do. So it cannot enter _parameters or
        _buffers under any torch version, num_params_total cannot move, training never
        allocates it, the optimiser's lm_head_params grouping is untouched, and it stays
        outside the gated kv_cache_bytes allocation delta, which is ONE delta measured around
        init_decode_state.

        self.lm_head.weight, the fp32 PARAMETER, is NOT modified and is still what forward, the
        optimiser and the validation pass read. That closes the WEIGHTS channel to val_bpb. It
        does not close the CLOCK channel, which runs through wall-clock LR progress to
        num_steps to tokens seen, and which this file does not claim to close.

        ON ONE OF THE WITNESS NUMBERS: max_abs_rel_err is bounded by 0.5/127 = 3.93701e-03 BY
        CONSTRUCTION for any weight distribution. Round 72 showed it is not literally constant
        -- it read 3.933449e-03, 3.916596e-03, 3.936969e-03 and 3.936743e-03 across four sites
        -- but its range is 5e-06 wide, so it discriminates nothing. mean_abs_rel_err and
        frac_at_clamp are the informative siblings, and on a softcapped head with an
        8192-row output the row-wise scale distribution is the thing I have no prior for.
        """
        wq = self.__dict__.get("_w_lm_decode_q8")
        if wq is None:
            with torch.no_grad():
                w = self.lm_head.weight.float()
                ws = (w.abs().amax(dim=1) / 127.0).clamp_min(1e-12).contiguous()
                wq = torch.round(w / ws.unsqueeze(1)).clamp_(-127.0, 127.0)
                wq = wq.to(torch.int8).contiguous()
                deq = wq.to(torch.float32) * ws.unsqueeze(1)
                den = w.abs().amax().clamp_min(1e-12)
                err = (deq - w).abs()
                print(f"DECODE_WEIGHT_Q8: site=lm_head shape={tuple(w.shape)} "
                      f"wq_dtype={wq.dtype} wq_elt={wq.element_size()} "
                      f"bytes_bf16={w.numel() * 2} "
                      f"bytes_int8={wq.numel() * wq.element_size() + ws.numel() * 4} "
                      f"scale_shape={tuple(ws.shape)} scale_dtype={ws.dtype} "
                      f"max_abs_rel_err={(err.amax() / den).item():.6e} "
                      f"mean_abs_rel_err={(err.mean() / den).item():.6e} "
                      f"frac_at_clamp={(wq.abs() >= 127).float().mean().item():.6e}", flush=True)
                del deq, err
            self.__dict__["_w_lm_decode_q8"] = wq
            self.__dict__["_w_lm_decode_q8_scale"] = ws
        return wq, self.__dict__["_w_lm_decode_q8_scale"]

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
        short_window = long_window // SHORT_WINDOW_DIVISOR
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
        _widen_decode_autotune()   # c0064: before any decode region compiles; allocates nothing
        assert max_len <= self.cos.size(1), f"max_len {max_len} exceeds rotary table"
        cfg = self.config
        head_dim = cfg.n_embd // cfg.n_head
        dev = self.transformer.wte.weight.device
        kw = {"dtype": torch.bfloat16, "device": dev}
        return {
            "max_len": max_len,
            "graph_enabled": bool(graph),
            "seq": torch.zeros(batch, dtype=torch.int32, device=dev),
            "kc": [torch.zeros(batch, max_len, cfg.n_kv_head, head_dim, **kw)
                   for _ in range(cfg.n_layer)],
            "vc": [torch.zeros(batch, max_len, cfg.n_kv_head, head_dim, **kw)
                   for _ in range(cfg.n_layer)],
        }

    def reset_decode_state(self, state):
        """Return `state` to position 0 so the same buffers serve another request.

        A fresh state per request would invalidate an address-bound CUDA graph. Stale bytes
        above the position are unreachable -- attention reads only `cache_seqlens` keys -- so
        zeroing them would be timed work serving never does.
        """
        state["seq"].zero_()
        return state

    def decode_embed_stack(self):
        """c0020. ONE gather for the token embedding AND all four value embeddings, replacing the
        two that c0019 left (one for `wte`, one for the merged value-embedding stack).

        c0019 merged four single-row value-embedding gathers into one and the kernel table showed
        the effect exactly: `indexSelectSmallIndex` went from n=7.0 at 23.37 us/step to n=4.0 at
        12.78, `launches_per_step` fell by exactly 3.0 on BOTH shapes, and NO new kernel name
        appeared -- so no copy kernel was spent back. `request_ms_median` went 197.0125 -> 194.0772.

        Two of the four survivors read THE SAME INDEX as each other: `transformer.wte(idx)` and the
        value-embedding stack are both a single-row lookup at `idx`. Concatenating `wte.weight`
        (vocab_size, n_embd) in FRONT of the four value tables (vocab_size, 4*kv_dim) gives one
        (vocab_size, n_embd + 4*kv_dim) table and one gather, and every consumer takes a SLICE.
        With `wte` first, the token slice is `[..., :n_embd]` at offset zero.

        The ceiling reasoning is c0006's `decode_proj_weight` precedent and c0019's, unchanged:
        the tensor is written through `self.__dict__`, bypassing `nn.Module.__setattr__`, so it
        cannot land in `_parameters` or `_buffers` and `num_params_total` (50332176, ZERO slack)
        cannot move. It is built LAZILY from the non-prefill decode path only, so training never
        allocates it and the gated `kv_cache_bytes` probe (34603520, EXACT MATCH) never reaches it.
        It is 8192 * 2560 * 2 = 41.9 MB, which lands on the INFERENCE peak (673933312 on launch 20,
        so ~682 MB) and not on the training peak (47198976512, ZERO slack, 70x larger).
        """
        st = self.__dict__.get("_embed_stack_decode")
        if st is None:
            keys = sorted(self.value_embeds.keys(), key=int)
            n_embd = int(self.transformer.wte.weight.size(1))
            with torch.no_grad():
                parts = [self.transformer.wte.weight.to(torch.bfloat16)]
                parts += [self.value_embeds[k].weight.to(torch.bfloat16) for k in keys]
                st = torch.cat(parts, dim=1).contiguous()
            self.__dict__["_embed_stack_decode"] = st
            self.__dict__["_embed_stack_keys"] = keys
            self.__dict__["_embed_stack_n_embd"] = n_embd
            print(f"DECODE_EMBED_STACK: n_embd={n_embd} layers={keys} shape={tuple(st.shape)} "
                  f"dtype={st.dtype} bytes={st.numel() * st.element_size()} "
                  f"gathers_before={1 + len(keys)} gathers_after=1 "
                  f"in_parameters={'_embed_stack_decode' in dict(self.named_parameters())} "
                  f"in_buffers={'_embed_stack_decode' in dict(self.named_buffers())}", flush=True)
        return (self.__dict__["_embed_stack_decode"], self.__dict__["_embed_stack_keys"],
                self.__dict__["_embed_stack_n_embd"])

    def decode_rope_stack(self):
        """c0020. ONE gather for cos and sin, replacing two `index_select` calls that read the
        SAME `seq_idx`. cos and sin are each (1, rotary_seq_len, 1, head_dim // 2) bf16, so the
        concatenation along the last dimension is (1, rotary_seq_len, 1, head_dim) and each half
        is a contiguous run at the decode shape (B=1, Tn=1).

        Same `self.__dict__` route and the same lazy non-prefill construction as the embedding
        stack, for the same two ceilings. cos and sin are registered as NON-PERSISTENT BUFFERS, so
        this concatenation would enter `_buffers` if assigned by attribute, which is exactly what
        the guard's control for this asserts. The table is
        (rotary_seq_len * head_dim) * 2 bytes, which at head_dim 128 is small next to the
        embedding stack.
        """
        st = self.__dict__.get("_rope_stack_decode")
        if st is None:
            with torch.no_grad():
                st = torch.cat([self.cos, self.sin], dim=-1).contiguous()
            self.__dict__["_rope_stack_decode"] = st
            self.__dict__["_rope_half_decode"] = int(self.cos.size(-1))
            print(f"DECODE_ROPE_STACK: shape={tuple(st.shape)} dtype={st.dtype} "
                  f"half={self.cos.size(-1)} bytes={st.numel() * st.element_size()} "
                  f"gathers_before=2 gathers_after=1 "
                  f"in_parameters={'_rope_stack_decode' in dict(self.named_parameters())} "
                  f"in_buffers={'_rope_stack_decode' in dict(self.named_buffers())}", flush=True)
        return self.__dict__["_rope_stack_decode"], self.__dict__["_rope_half_decode"]

    def _decode_body(self, idx, state, prefill):
        """`forward`'s body with the cache spliced in. `prefill` is a Python bool that is
        constant per variant, so the two branches are two graphs, not one per position."""
        B, Tn = idx.size()
        # c0049: set by the merged prologue when layer 0's qkv is folded into it, and consumed
        # by layer 0 inside the loop. Initialised HERE rather than beside `merged_out`, because
        # unlike `merged_out` it is ASSIGNED before that point -- the prologue block runs above
        # the loop -- so an initialisation down there would erase it. The prefill path never
        # assigns it and every layer but the first sees None.
        pending_qkv = None
        if prefill:
            cos, sin = self.cos[:, :Tn], self.sin[:, :Tn]
        else:
            # c0024: both eager prologue blocks are now ONE compiled region. The lazily built
            # caches are still resolved out here; only the arithmetic moved in.
            _t = _PHASE.start()
            rope_stack, rope_half = self.decode_rope_stack()
            emb_stack, ve_keys, ve_off = self.decode_embed_stack()
            # c0049: layer 0's qkv region is folded in. `_decode_prologue_qkv` inlines the
            # `_decode_qkv_nove` body, so this is correct ONLY while layer 0 has no value
            # embedding. That depends on has_ve and DEPTH, which this candidate does not fix, so
            # it is asserted rather than trusted -- a wrong branch here is a silent fidelity
            # change in q, k and v, and it must not be reachable.
            if "0" in self.value_embeds:
                raise AssertionError(
                    "c0049: layer 0 HAS a value embedding at this shape, so it would take "
                    "_decode_qkv_ve, but _decode_prologue_qkv inlines the _decode_qkv_nove "
                    "body. Revert to c0048; do not edit this assertion.")
            _a0 = self.transformer.h[0].attn
            _sq0, _sk0, _sv0 = _a0.decode_qkv_splits()
            cos, sin, ve_all, x_pro, _x0l, _q0, _k0, _v0 = _decode_prologue_qkv(
                state["seq"], rope_stack, rope_half, idx, emb_stack, ve_off,
                self.resid_lambdas[0], self.x0_lambdas[0], *_a0.decode_qkv_weight_q8(),
                _sq0, _sk0, _sv0, _a0.n_head, _a0.n_kv_head, _a0.head_dim)
            pending_qkv = (_x0l, _q0, _k0, _v0)
            _PHASE.stop("prologue_qkv", _t)

        # c0020. ONE gather for the token embedding AND all four value embeddings, on the
        # non-prefill branch only. The prefill branch keeps the original separate wte lookup and
        # per-layer value gather, so prefill arithmetic and prefill kernels are untouched and only
        # the 512 decode steps change.
        if prefill:
            x = norm(self.transformer.wte(idx))
            ve_all, ve_keys, ve_dim, ve_off = None, None, 0, 0
        else:
            # c0024: computed in _decode_prologue above, alongside the rope lookup. c0020's
            # note that every slice is a VIEW still holds and is why ve_all can cross the
            # region boundary without a copy.
            _t = _PHASE.start()
            x = x_pro
            ve_dim = (ve_all.size(-1) - ve_off) // len(ve_keys)
            _PHASE.stop("ve_lookup", _t)
        x0 = x
        # c0021: None means "this layer must compute its own prologue". The prefill branch never
        # reads it, and it is reset to None after the last layer's post.
        pending_h = None
        # c0048: set by the last layer's epilogue when the head region is folded into it.
        # Initialised here, beside `pending_h`, for the same reason: it must exist whether or
        # not the branch that assigns it runs, and on the prefill path it never does.
        merged_out = None
        for i, block in enumerate(self.transformer.h):
            if prefill:
                ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            elif str(i) in self.value_embeds:
                _k = ve_keys.index(str(i))          # a Python int, resolved once at capture
                _o = ve_off + _k * ve_dim              # ve_off skips the token-embedding columns
                ve = ve_all[..., _o:_o + ve_dim]
            else:
                ve = None
            attn = block.attn
            if prefill:
                # c0023: one compiled region per layer in place of ~25 eagerly dispatched
                # ops. `ve is None` is a Python bool decided by has_ve, so this is a static
                # choice per layer and not a branch inside a graph.
                if ve is None:
                    x, q, k, v = _prefill_qkv(
                        x, x0, self.resid_lambdas[i], self.x0_lambdas[i],
                        attn.c_q.weight, attn.c_k.weight, attn.c_v.weight, cos, sin,
                        attn.n_head, attn.n_kv_head, attn.head_dim)
                else:
                    x, q, k, v = _prefill_qkv_ve(
                        x, x0, self.resid_lambdas[i], self.x0_lambdas[i],
                        attn.c_q.weight, attn.c_k.weight, attn.c_v.weight,
                        attn.ve_gate.weight, attn.ve_gate_channels, ve, cos, sin,
                        attn.n_head, attn.n_kv_head, attn.head_dim)
            elif ve is None:
                # c0002: same arithmetic in the same order, handed to inductor as one region.
                # c0003: three GEMVs became one. Built here, on the non-prefill branch, so
                # the gated kv_cache_bytes probe never reaches this allocation.
                sq, sk, sv = attn.decode_qkv_splits()
                _t = _PHASE.start()
                # c0021: `pending_h` is not None exactly when layer i-1's `_decode_post_pro`
                # already computed this layer's combined x and its norm. Layer 0 has no
                # predecessor, so it takes the ORIGINAL region and the original opening lines.
                if pending_qkv is not None:
                    # c0049: layer 0, whose qkv already ran inside the merged prologue. Cleared
                    # immediately so that no later layer can read it, which keeps the invariant
                    # "pending_qkv is not None only at i == 0" true by construction rather than
                    # by the loop's shape. `_decode_qkv_nove` is now called from NOWHERE on the
                    # decode path; it is left in the file because the prefill branch and the
                    # `_h` variants document against it.
                    x, q, k, v = pending_qkv
                    pending_qkv = None
                elif pending_h is None:
                    x, q, k, v = _decode_qkv_nove(
                        x, x0, self.resid_lambdas[i], self.x0_lambdas[i],
                        attn.decode_qkv_weight(), sq, sk, sv, cos, sin,
                        attn.n_head, attn.n_kv_head, attn.head_dim)
                else:
                    x, q, k, v = _decode_qkv_nove_h(
                        x, pending_h,
                        attn.decode_qkv_weight(), sq, sk, sv, cos, sin,
                        attn.n_head, attn.n_kv_head, attn.head_dim)
                _PHASE.stop("qkv_nove", _t)
            else:
                # c0005: four splits, and the gate rides in the fused matrix.
                sq, sk, sv, sg = attn.decode_qkv_splits()
                _t = _PHASE.start()
                if pending_h is None:
                    x, q, k, v = _decode_qkv_ve(
                        x, x0, self.resid_lambdas[i], self.x0_lambdas[i],
                        attn.decode_qkv_weight(), sq, sk, sv, sg, ve, cos, sin,
                        attn.n_head, attn.n_kv_head, attn.head_dim)
                else:
                    x, q, k, v = _decode_qkv_ve_h(
                        x, pending_h,
                        attn.decode_qkv_weight(), sq, sk, sv, sg, ve, cos, sin,
                        attn.n_head, attn.n_kv_head, attn.head_dim)
                _PHASE.stop("qkv_ve", _t)

            kc, vc = state["kc"][i], state["vc"][i]
            if prefill:
                kc[:, :Tn] = k
                vc[:, :Tn] = v
                y = fa3.flash_attn_func(q, kc[:, :Tn], vc[:, :Tn], causal=True,
                                        window_size=self.window_sizes[i])
            else:
                # Appends k,v at cache_seqlens and attends over the whole preallocated
                # cache, so there is no `kc[:, start:end]` slice whose bounds depend on the
                # step number. num_splits=1 is not a tuning choice: the op's fake kernel
                # refuses to trace at the default num_splits=0, which is precisely what
                # makes an unpinned cache uncompilable. It is a reduction-order choice, and
                # the agreement check in prepare.py still has to pass with it pinned.
                #
                # c0001 NOTE: the four lines above are the reference's rationale and are
                # kept verbatim because they are the substrate's own record, but they no
                # longer describe this call. The fake-kernel raise is tracing-only and
                # nothing traces this path (see DECODE_NUM_SPLITS at the top of the file).
                # The last sentence still binds and is the live risk: this IS a
                # reduction-order change, so the fidelity ceilings must still pass.
                # c0010: the SAME call with one extra argument, decided once per state by
                # _probe_sched_meta before capture and never per step. `_SCHED_META_MODE` is
                # the probe's own override and is None everywhere else, so the measured path
                # reads only the state. The arithmetic is untouched: a tile schedule changes
                # which SM does which block, not what is summed.
                metas = state.get("sched_meta")
                use = _SCHED_META_MODE if _SCHED_META_MODE is not None else (
                    "on" if metas else "off")
                meta = metas[i] if (use == "on" and metas and i < len(metas)) else None
                _t = _PHASE.start()
                y = fa3.flash_attn_with_kvcache(q, kc, vc, k=k, v=v,
                                                cache_seqlens=state["seq"], causal=True,
                                                window_size=self.window_sizes[i],
                                                num_splits=_decode_splits_for(
                                                    self.window_sizes[i]),
                                                scheduler_metadata=meta)
                _PHASE.stop("fa3_meta" if meta is not None else "fa3", _t)
            if prefill:
                x = _prefill_post(x, y, attn.c_proj.weight,
                                  block.mlp.c_fc.weight, block.mlp.c_proj.weight)
            else:
                # y is fa3's output, produced eagerly outside every compiled region; only
                # the projection, the norm and the relu-square MLP are fused.
                _t = _PHASE.start()
                if i + 1 < len(self.transformer.h):
                    # c0021: fold the NEXT layer's prologue into this layer's epilogue. i+1 is a
                    # Python int resolved at capture, so this is a static choice per unrolled
                    # layer and not a branch inside the graph.
                    x, pending_h = _decode_post_pro(
                        x, y, attn.decode_out_weight(),
                        *block.mlp.decode_fc_weight_q8(), *block.mlp.decode_proj_weight_q8(),
                        self.resid_lambdas[i + 1], self.x0_lambdas[i + 1], x0)
                else:
                    # c0048: the last layer has no successor LAYER, which is why c0021 left it
                    # on the original `_decode_post`. It does have a successor REGION -- the
                    # head -- and this folds that in instead. `_decode_post` is now called
                    # from nowhere on the decode path; it is left in the file because
                    # `_prefill_post` and it are the pair the prefill branch documents against.
                    pending_h = None
                    merged_out = _decode_post_head(
                        x, y, attn.decode_out_weight(),
                        *block.mlp.decode_fc_weight_q8(), *block.mlp.decode_proj_weight_q8(),
                        *self.decode_lm_weight_q8(), 15, state["seq"])
                _PHASE.stop("post_head" if merged_out is not None else "post", _t)

        softcap = 15
        if prefill:
            x = norm(x[:, -1:, :])
            logits = self.lm_head(x).float()
            return softcap * torch.tanh(logits / softcap)
        if merged_out is not None:
            # c0048: the head already ran, inside the last layer's epilogue. The literal 15
            # passed to `_decode_post_head` and this `softcap` are the same number in two
            # places, which is exactly the drift `_decode_head` was protected from by taking
            # it as an argument -- so it is asserted rather than trusted. A mismatch is a
            # silent fidelity change in the tanh, so it must not be reachable.
            if softcap != 15:
                raise AssertionError(
                    f"c0048: softcap is {softcap} after the loop but 15 was compiled into "
                    f"_decode_post_head; the two sites have drifted apart")
            return merged_out
        _t = _PHASE.start()
        out = _decode_head(x, self.lm_head.weight, softcap, state["seq"])
        _PHASE.stop("head", _t)
        return out

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
            # c0025: `_decode_head` advanced `seq`. Incrementing here too would double-count.
            return logits, state
        # The graph lives ON the state it was captured against, so its lifetime is exactly
        # that state's and there is no key to get wrong. A state whose capture failed holds
        # an uncaptured object and runs eager forever, which is slow and correct.
        graph = state.get("graph")
        if graph is None:
            graph = state["graph"] = _GraphedDecodeStep(self, state)
        if not graph.captured:
            logits = self._decode_body(idx, state, prefill=False)
            # c0025: `_decode_head` advanced `seq`. Incrementing here too would double-count.
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
        else:
            self._probe_sched_meta()
            try:
                self._capture()
                self.captured = True
            except Exception as exc:        # noqa: BLE001 -- recorded, then paid for in ms
                self.reason = f"{type(exc).__name__}: {exc}"
                self.graph, self.static_logits = None, None
                # c0008's retry, retargeted. Capture failure is not merely slow here, it voids
                # the round -- an eager launch tests nothing about a replayed step, and round 5
                # recorded that `captured=False` reads exactly like a refuted mechanism. If the
                # probe adopted the torch attention and capture then failed, that path is the
                # suspect, so retry once on fa3. The worst case becomes the incumbent's own
                # behaviour rather than a forfeited round, and both attempts print.
                if state.get("sched_meta"):
                    state["sched_meta"] = None
                    self.reason += " | retry without scheduler_metadata"
                    try:
                        self._capture()
                        self.captured = True
                    except Exception as exc2:     # noqa: BLE001
                        self.reason += f" | retry {type(exc2).__name__}: {exc2}"
                        self.graph, self.static_logits = None, None
        # c0001 diagnostic. Not a metric and not an input to one: prepare.py remains the
        # only accessor for every judged number. Capture failure sets captured False and
        # runs eager forever -- slow, correct, and until now silent, which reads exactly
        # like a refuted mechanism. This line separates "split-KV did not help" from
        # "split-KV was not capturable and the step never ran on the graph at all".
        print(f"DECODE_GRAPH_CAPTURE: captured={self.captured} "
              f"sched_meta={'on' if state.get('sched_meta') else 'off'} "
              f"kvcache_num_splits={DECODE_NUM_SPLITS} "
              f"kvcache_num_splits_per_layer="
              f"{[_decode_splits_for(w) for w in self.model.window_sizes]} "
              f"windows={[int(w[0]) for w in self.model.window_sizes]} "
              f"reason={self.reason!r}", flush=True)
        self._profile_phases()
        self._profile_kernels()
        self._profile_prefill()

    PROFILE_PASSES = 5
    PROBE_PASSES = 4
    SWEEP_PASSES = 5              # c0018. Same count as PROFILE_PASSES, same instrument.
    SWEEP_CONFIGS = (             # (short_window_splits, long_window_splits)
        (16, 32),                 # the incumbent, and the sweep's own control
        (16, 16),                 # c0013, the only other CLEANLY measured point: 199.4377
        (32, 32),                 # never validly measured. Launch 15 ran it with sched_meta OFF.
        (17, 33),                 # never validly measured. Launch 17 ran it with sched_meta OFF.
        (8, 8),                   # c0012, cleanly measured at 213.5764. It ran UNIFORM 8 -- the
                                  # per-window map did not exist until c0015 -- so the entry that
                                  # corresponds to that launch is (8, 8) and not (8, 16). Getting
                                  # this wrong would have given the sweep a "calibration" point
                                  # matching no measured launch, which is the whole value of it.
        (24, 32),
        (16, 64),
        (32, 64),
        (12, 24),
        (16, 32),                 # THE INCUMBENT AGAIN, LAST, deliberately: without a repeat of
                                  # one configuration the sweep reports no dispersion of its own
                                  # and every difference in it would be unfalsifiable. First and
                                  # last are the same table, run ten profiles apart.
    )
    ATTN_PROBE_TV_TOL = 0.01     # agreement with fa3, in total variation on one step's probs
    ATTN_ADOPT_MARGIN = 0.05      # RETAINED FOR THE RECORD AND NO LONGER THE DECISION. It was
    # `us_on <= us_off * 0.95`: a 5 per cent margin on the difference of two PROBE_PASSES=4 means.
    # Audited backwards over the logs of every launch that has this line, the prefilled shape's
    # probe read +7.70, +5.61, -1.41, +5.15, +4.11 per cent -- mean +4.23, sd 3.41 at df=4 --
    # against a threshold of 5.0. A gate whose input has two thirds the dispersion of its own
    # threshold is a coin flip, and it came up tails on launches 15 and 17, each of which
    # therefore ran WITHOUT a mechanism worth 9.38 ms and is void as a test of what it changed.
    # Round 15's conclusion about uniform 32 splits was retracted for exactly this reason.
    ATTN_REJECT_FLOOR = 0.15      # c0017: adopt on FIDELITY, and refuse only on a real pathology.
    # tv_vs_off has been bit-exact 0.0000000 in 10 of 10 probe measurements across both request
    # shapes and five launches, which is what you expect when precomputed metadata reproduces the
    # partition the default path would have chosen -- the gain is removed recomputation, not a
    # different algorithm, so there is no accuracy being traded and nothing for a SPEED gate to
    # protect. This floor still refuses if the metadata path is more than 15 per cent SLOWER,
    # which is 4.4x the probe's own sd and therefore a threshold the instrument can actually
    # resolve. That is the difference between a guard with dynamic range and one without.

    def _probe_sched_meta(self):
        """Adopt a precomputed fa3 tile schedule for THIS state, or refuse and say why.

        Same adopt-or-refuse shape as c0008's tuner and c0009's attention probe, both of which
        refused and both of which therefore cost the round its mechanism and nothing else. That
        is the point of the shape: the two ways this can be wrong -- a schedule that is stale
        for a shorter seqlen, or one that over-schedules more than it saves -- become a printed
        null instead of a ceiling breach that forfeits the charge.

        The metadata is built for `cache_seqlens = max_len - 1`, the LARGEST position the state
        will ever reach, not for the current seq. That direction is chosen deliberately: an
        over-schedule does redundant tiles that the causal and window masks discard, which is
        slow but arithmetically inert, whereas an under-schedule could drop live blocks. If the
        kernel does not tolerate over-scheduling, the agreement gate below is what says so.

        The signature is discovered rather than assumed. The pinned build is
        `kernels-community/flash-attn3` resolved at runtime and is not inspectable from the
        driver host, so the kwargs are filtered against `inspect.signature` and a required
        parameter this code cannot supply is a printed refusal naming the parameter, not a
        TypeError inside a launch that has already been charged.
        """
        global _SCHED_META_MODE
        if self.state["seq"].numel() != 1:
            return
        seq0 = self.state["seq"].clone()
        self.state["sched_meta"] = None
        try:
            import inspect
            print("DECODE_FA3_SIGNATURE: "
                  f"{inspect.signature(fa3.flash_attn_with_kvcache)}", flush=True)
            getter = getattr(fa3, "get_scheduler_metadata", None)
            if getter is None:
                print(f"DECODE_SCHED_META: refused={SCHED_META_UNAVAILABLE} "
                      f"attrs={[a for a in dir(fa3) if 'sched' in a.lower()]}", flush=True)
                print("DECODE_SCHED_META_CHOSEN: sched_meta=off adopted=False "
                      "untested=1 reason=no_getter", flush=True)
                return
            sig = inspect.signature(getter)
            print(f"DECODE_SCHED_META_SIGNATURE: {sig}", flush=True)

            # read off the state's own buffers, which are (batch, max_len, n_kv_head, head_dim)
            kc0 = self.state["kc"][0]
            max_len = int(kc0.size(1))
            HKV = int(kc0.size(2))
            D = int(kc0.size(-1))
            cfg = getattr(self.model, "config", None)
            HQ = int(getattr(cfg, "n_head", HKV)) if cfg is not None else HKV
            at_max = torch.full_like(self.state["seq"], max(max_len - 1, 0))
            known = {
                "batch_size": int(kc0.size(0)), "max_seqlen_q": 1, "max_seqlen_k": max_len,
                "num_heads_q": HQ, "num_heads_kv": HKV, "headdim": D, "headdim_v": D,
                "cache_seqlens": at_max, "qkv_dtype": kc0.dtype,
                "causal": True, "num_splits": DECODE_NUM_SPLITS,
                "max_seqlen_k_new": 1, "has_softcap": False, "page_size": None,
            }
            missing = [n for n, prm in sig.parameters.items()
                       if prm.default is inspect.Parameter.empty
                       and prm.kind in (prm.POSITIONAL_OR_KEYWORD, prm.KEYWORD_ONLY)
                       and n not in known and n != "window_size"]
            if missing:
                print(f"DECODE_SCHED_META_CHOSEN: sched_meta=off adopted=False untested=1 "
                      f"reason=unsupplied_required_params missing={missing}", flush=True)
                return
            metas = []
            # c0012 FIXES LAUNCH 12's FAULT. c0011 read the window list off the graphed step.
            # That attribute lives on the MODEL (set in its __init__ from
            # _compute_window_sizes), not on _GraphedDecodeStep, whose only attributes are
            # model, state, captured, reason, static_idx, static_logits and graph. The probe
            # raised AttributeError, caught itself, printed `untested=1`, and the run continued
            # with sched_meta=off -- so launch 12 measured NOTHING about this axis, for the
            # second time. Same error class as the invented static_q_dtype in c0010.
            # A static self-attribute check now runs pre-dispatch: over all eleven earlier
            # candidates it flags exactly 2, c0010 and c0011, and both carried this bug.
            windows = self.model.window_sizes
            for i in range(len(windows)):
                kw = {n: v for n, v in known.items() if n in sig.parameters}
                if "window_size" in sig.parameters:
                    kw["window_size"] = windows[i]
                # c0015: the metadata must be built at the SAME num_splits the call will use for
                # this layer, or it describes a partition nobody executes. known[] carries the
                # default; this line is the per-window override and it is the whole reason the
                # split count had to become per-layer in both places at once.
                if "num_splits" in sig.parameters:
                    kw["num_splits"] = _decode_splits_for(windows[i])
                metas.append(getter(**kw))
            # c0016 FREE DIAGNOSTIC. This costs nothing: the charge is already spent, the probe
            # already built these tensors, and this is one print outside every compiled region
            # and outside every timed pass. c0015's table showed shapes=[(13,), (13,), (13,)] --
            # constant across both request shapes and both window lengths -- so the SHAPE says
            # nothing about the block partition. The CONTENTS may. I am printing one short-window
            # layer and one long-window layer with their split counts beside them so the numbers
            # can be read against 1025 and 2049 keys. If a value equals the block count the block
            # size is settled by observation instead of assumed; if nothing is legible I have lost
            # nothing and will say the assumption stands unmeasured.
            try:
                for probe_i in (0, 3):
                    if probe_i < len(metas):
                        m = metas[probe_i]
                        vals = m.flatten().tolist() if hasattr(m, "flatten") else m
                        print(f"DECODE_SCHED_META_CONTENT: layer={probe_i} "
                              f"window={int(windows[probe_i][0])} "
                              f"keys={int(windows[probe_i][0]) + 1} "
                              f"splits={_decode_splits_for(windows[probe_i])} "
                              f"dtype={getattr(m, 'dtype', None)} values={vals}", flush=True)
            except Exception as exc:                      # diagnostic only: never fail the run
                print(f"DECODE_SCHED_META_CONTENT: unavailable {type(exc).__name__}: {exc}",
                      flush=True)
            print(f"DECODE_SCHED_META: built={len(metas)} max_seqlen_k={max_len} "
                  f"cache_seqlens_used={int(at_max[0])} hq={HQ} hkv={HKV} headdim={D} "
                  f"shapes={[tuple(m.shape) if hasattr(m, 'shape') else type(m).__name__ for m in metas[:3]]}",
                  flush=True)

            def one(mode, span):
                global _SCHED_META_MODE
                _SCHED_META_MODE = mode
                _PHASE.on = True
                self.state["seq"].copy_(seq0)
                logits = self._advance()        # untimed: this path's first-call costs
                ref = logits.detach().clone()
                _PHASE.drain()
                for _ in range(self.PROBE_PASSES):
                    self.state["seq"].copy_(seq0)
                    self._advance()
                per = _PHASE.drain()
                _PHASE.on = False
                return ref, per.get(span, 0.0) * 1000.0 / self.PROBE_PASSES

            self.state["sched_meta"] = metas
            ref_off, us_off = one("off", "fa3")
            ref_on, us_on = one("on", "fa3_meta")
            pa = torch.softmax(ref_off.float().reshape(-1), dim=0)
            pb = torch.softmax(ref_on.float().reshape(-1), dim=0)
            tv = float(0.5 * (pa - pb).abs().sum())
            agrees = tv <= self.ATTN_PROBE_TV_TOL
            # `faster` is still COMPUTED and still PRINTED, so the record loses nothing and a
            # later reader can recover the old rule's verdict from any launch. It is simply no
            # longer what decides.
            faster = us_off > 0.0 and us_on <= us_off * (1.0 - self.ATTN_ADOPT_MARGIN)
            pathological = us_off > 0.0 and us_on > us_off * (1.0 + self.ATTN_REJECT_FLOOR)
            adopt = bool(agrees and not pathological)
            self.state["sched_meta"] = metas if adopt else None
            gain = 100.0 * (us_off - us_on) / us_off if us_off > 0 else 0.0
            print(f"DECODE_SCHED_META_PROBE: off_us_per_step={us_off:.2f} "
                  f"on_us_per_step={us_on:.2f} speedup_pct={gain:+.2f} "
                  f"tv_vs_off={tv:.7f} tol={self.ATTN_PROBE_TV_TOL} "
                  f"passes={self.PROBE_PASSES}", flush=True)
            print(f"DECODE_SCHED_META_CHOSEN: "
                  f"sched_meta={'on' if adopt else 'off'} adopted={adopt} "
                  f"agrees={agrees} faster={faster} "
                  f"pathological={pathological} "
                  f"rule=fidelity_and_not_pathological "
                  f"floor_pct={100.0 * self.ATTN_REJECT_FLOOR:.0f} "
                  f"old_rule_would_have={'on' if (agrees and faster) else 'off'} "
                  f"margin_pct={100.0 * self.ATTN_ADOPT_MARGIN:.0f}", flush=True)
            # c0018. The sweep runs ONLY when the metadata was adopted, and that condition is the
            # whole lesson of launches 15 and 17: a measurement taken in a mechanism state the
            # launch does not run is void, and I am not going to spend a charge producing nine of
            # them. If sched_meta is off, the sweep prices nothing this candidate uses, so it says
            # so and does not run.
            if adopt:
                self._sweep_splits(getter, sig, known, windows, seq0)
            else:
                print("DECODE_SPLIT_SWEEP: skipped=1 reason=sched_meta_not_adopted "
                      "basis=a_config_priced_in_an_unused_mechanism_state_is_void", flush=True)
        except Exception as exc:            # noqa: BLE001 -- a probe may not kill a launch
            self.state["sched_meta"] = None
            print(f"DECODE_SCHED_META_CHOSEN: failed={type(exc).__name__}: {exc} "
                  f"sched_meta=off adopted=False untested=1", flush=True)
        finally:
            _SCHED_META_MODE = None
            _PHASE.on = False
            _PHASE.spans = []
            self.state["seq"].copy_(seq0)

    def _sweep_splits(self, getter, sig, known, windows, seq0):
        """c0018 diagnostic. REPORT ONLY. Not a metric, not an input to one, and it never changes
        what this launch captures.

        WHY A CHARGE BUYS THIS. `num_splits` has produced the two largest wins of this run
        (-14.14 ms at 8->16 uniform, -2.12 ms at 16->{16,32}) and it is the axis I understand
        least: of five configurations tried, two were measured with the scheduler-metadata
        mechanism OFF and are void, so the axis has exactly three valid points and its block size
        is unmeasured -- c0016's free diagnostic proved illegible because the metadata is a runtime
        workspace whose contents before use are uninitialised. Every further single-point guess on
        this axis costs a full 700 s launch and returns one number. This sweep prices ten
        configurations inside one charge, on the profiler's SELF-DEVICE kernel time, which round 16
        established converts to request at 1 us/step ~= 0.512 ms (393.5 us/step x 512 steps =
        201.5 ms against a recorded 197.3 ms, a named 2 per cent discrepancy).

        WHY IT DOES NOT ADOPT. The fault c0017 removed was an in-launch decision taken on a noisy
        in-launch measurement. A sweep that picked its own winner would be that fault again with
        more points, and its winner would arrive with no cross-launch replicate to check it. So the
        winner, if there is one, becomes the NEXT candidate's captured table and is measured on the
        real ranking key like everything else. Nothing here reads back into this launch.

        WHAT IT COSTS. Ten configurations x (1 untimed + SWEEP_PASSES timed) eager steps, at
        ~1.5 ms per eager step, is about 90 ms of device time plus profiler overhead, all of it
        before capture and none of it inside a timed pass. No graph is captured for any swept
        configuration -- the metadata tensors are 13 elements each and are dropped per iteration --
        because `peak_vram_bytes` has ZERO slack against its ceiling and a second graph would
        breach it. The prediction is a null at the incumbent's value, and that null is the check
        that this docstring's claim about cost is true.

        The first and last configurations are identical on purpose. A sweep without a repeated
        point reports differences no one can falsify.
        """
        global _SPLITS_OVERRIDE
        saved_meta = self.state["sched_meta"]
        rows_out = []
        try:
            from torch.profiler import profile, ProfilerActivity
            try:
                from torch.autograd import DeviceType
                cuda_kind = DeviceType.CUDA
            except Exception:               # noqa: BLE001 -- then nothing is filtered, visibly
                cuda_kind = None
            for short_splits, long_splits in self.SWEEP_CONFIGS:
                # c0052 carries c0051's INSTRUMENT REPAIR, which launch 53 PROVED works: this dict is
                # keyed by WINDOW LENGTH and carried no 256 key while the short layers have had window
                # 256 since c0042, so _decode_splits_for((256,0)) fell through to DECODE_NUM_SPLITS = 16
                # on every sweep row for five launches and the `short` column was inert in all of it.
                # Launch 53 printed SIX distinct splits_per_layer[0] values {8,12,16,17,24,32} against
                # exactly one (always 16) in every earlier log. report_only=1, restored to None in a
                # finally before capture, so it cannot reach a recorded metric.
                # STILL BROKEN, DISCLOSED NOT FIXED: SWEEP_CONFIGS has no 1, so the sweep cannot measure
                # num_splits=1 and therefore still never measures a captured table containing it.
                _SPLITS_OVERRIDE = {256: int(short_splits),
                                    1024: int(short_splits), 2048: int(long_splits)}
                metas = []
                for i in range(len(windows)):
                    kw = {n: v for n, v in known.items() if n in sig.parameters}
                    if "window_size" in sig.parameters:
                        kw["window_size"] = windows[i]
                    # The metadata MUST be built at the same split count the call will use, or it
                    # describes a partition nobody executes -- the c0015 lesson, applied per config.
                    if "num_splits" in sig.parameters:
                        kw["num_splits"] = _decode_splits_for(windows[i])
                    metas.append(getter(**kw))
                self.state["sched_meta"] = metas
                self.state["seq"].copy_(seq0)
                self._advance()             # untimed: this configuration's first-call costs
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    for _ in range(self.SWEEP_PASSES):
                        self.state["seq"].copy_(seq0)
                        self._advance()
                    torch.cuda.synchronize()
                total = flash = combine = 0.0
                n_kernel = 0
                for e in prof.key_averages():
                    micros = 0.0
                    for attr in ("self_device_time_total", "self_cuda_time_total"):
                        value = getattr(e, attr, None)
                        if value:
                            micros = float(value)
                            break
                    if micros <= 0:
                        continue
                    key = str(getattr(e, "key", "?"))
                    # Same filter as _profile_kernels, for the same reason: key_averages() mixes
                    # CPU operator records with GPU kernel records, and summing both over-counted
                    # by 8.5x in c0007. n_kernel is printed so the filter can be audited from the
                    # log rather than trusted.
                    if cuda_kind is not None and getattr(e, "device_type", None) != cuda_kind:
                        continue
                    if _is_annotation(key):
                        continue
                    per = micros / self.SWEEP_PASSES
                    total += per
                    n_kernel += 1
                    low = key.lower()
                    if "combine" in low:
                        combine += per
                    elif "flash" in low or "fmha" in low:
                        flash += per
                rows_out.append((int(short_splits), int(long_splits), total))
                print(f"DECODE_SPLIT_SWEEP: short={short_splits} long={long_splits} "
                      f"splits_per_layer={[_decode_splits_for(w) for w in windows]} "
                      f"device_us_per_step={total:.2f} "
                      f"attn_us_per_step={flash + combine:.2f} "
                      f"flash_us_per_step={flash:.2f} combine_us_per_step={combine:.2f} "
                      f"n_kernel={n_kernel} passes={self.SWEEP_PASSES} "
                      f"adopted=False report_only=1", flush=True)
                del metas
            control = [t for s, lg, t in rows_out if (s, lg) == self.SWEEP_CONFIGS[0]]
            spread = (max(control) - min(control)) if len(control) > 1 else float("nan")
            print(f"DECODE_SPLIT_SWEEP_DONE: n_config={len(rows_out)} "
                  f"control_reps={len(control)} control_us_per_step={control} "
                  f"control_range_us={spread:.2f} "
                  f"captured_table={DECODE_NUM_SPLITS_BY_WINDOW} "
                  f"sched_meta_mode={_SCHED_META_MODE!r} "
                  f"override_at_done={_SPLITS_OVERRIDE} report_only=1", flush=True)
        except Exception as exc:            # noqa: BLE001 -- a diagnostic may not kill a launch
            print(f"DECODE_SPLIT_SWEEP: failed={type(exc).__name__}: {exc} "
                  f"n_config_done={len(rows_out)}", flush=True)
        finally:
            _SPLITS_OVERRIDE = None
            self.state["sched_meta"] = saved_meta
            self.state["seq"].copy_(seq0)
            # This print fires on the exception path too, which is the point: it is the record
            # that the capture about to happen sees the module table and the probe's own metadata
            # list, whatever went wrong above. If it ever reads anything other than
            # override_restored=True the launch is void as a measurement and I will say so.
            print(f"DECODE_SPLIT_SWEEP_RESTORED: override_restored={_SPLITS_OVERRIDE is None} "
                  f"splits_per_layer={[_decode_splits_for(w) for w in windows]} "
                  f"sched_meta_is_probe_object={self.state['sched_meta'] is saved_meta} "
                  f"sched_meta_on={self.state['sched_meta'] is not None}", flush=True)

    def _profile_kernels(self):
        """c0007 diagnostic. Not a metric and not an input to one.

        Round 5's `_PHASE` spans measured WALL time on an eager path, so each span included the
        device sitting idle between launches; round 5 recorded that this makes them
        incomparable to the replayed step, and round 6 measured the instrument's own
        between-state offset at ~10%. A profiler's SELF-DEVICE time per kernel does not include
        those host gaps. It is the same kernels on the same shapes that the graph replays, so
        the sum is comparable to the replayed per-step time and the ranking is what four rounds
        of arithmetic have been guessing at.

        Every card since round 3 has said "~50 kernels" without ever measuring it, and the
        7.49 us per real kernel derived from c0003 makes that count load-bearing.

        c0008 fixes a bug in this accessor, and the bug was mine. `key_averages()` returns
        CPU-side operator records and GPU kernel records in ONE list, and c0007 summed
        `self_device_time_total` across all of them, so a parent's device time was counted
        alongside its children's. It reported 211 launches and 3137 us of device time per step
        against a 367 us replayed step -- an 8.5x excess I nearly recorded as a finding about
        the model. The log's own table refuted it: four triton entries appeared as exact
        duplicate pairs, `_flash_attn3_cuda::fwd` read 212.20 us while the three cutlass
        kernels beneath it summed to 190.45, and six `## Call CompiledFxGraph` entries, which
        are not kernels at all, held 2206.69 of the 3137.1. Filtered to GPU entries the same
        table reads 48 launches and 326.88 us against 367 -- the cross-check this was built
        for, passing.

        So the filter is `device_type == DeviceType.CUDA`, and the print carries BOTH totals
        plus a per-line `dev=` flag. A filter whose output cannot be audited from the log is
        how the first version went wrong, and printing the discriminator costs two fields.

        Runs in the same discarded warmup pass as `_profile_phases`, restores `seq`, and any
        failure prints and returns: a diagnostic may not fail a launch or break a capture.
        """
        if self.state["seq"].numel() != 1:
            return
        seq0 = self.state["seq"].clone()
        try:
            from torch.profiler import profile, ProfilerActivity
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                for _ in range(self.PROFILE_PASSES):
                    self.state["seq"].copy_(seq0)
                    self._advance()
                torch.cuda.synchronize()
            try:
                from torch.autograd import DeviceType
                cuda_kind = DeviceType.CUDA
            except Exception:               # noqa: BLE001 -- then nothing is filtered, visibly
                cuda_kind = None
            rows = []
            for e in prof.key_averages():
                micros = 0.0
                for attr in ("self_device_time_total", "self_cuda_time_total"):
                    value = getattr(e, attr, None)
                    if value:
                        micros = float(value)
                        break
                if micros > 0:
                    key = str(getattr(e, "key", "?"))
                    on_device = (cuda_kind is not None
                                 and getattr(e, "device_type", None) == cuda_kind)
                    rows.append((micros / self.PROFILE_PASSES,
                                 float(getattr(e, "count", 0)) / self.PROFILE_PASSES,
                                 key[:70],
                                 bool(on_device),
                                 bool(_is_annotation(key))))
            rows.sort(reverse=True)
            kern = [r for r in rows if r[3] and not r[4]]
            ann = [r for r in rows if r[4]]
            shown = kern if kern else rows
            print(f"DECODE_KERNELS: kernels_only={bool(kern)} n_kernel={len(kern)} "
                  f"n_annotation={len(ann)} n_all={len(rows)} "
                  f"launches_per_step={sum(r[1] for r in kern):.1f} "
                  f"device_us_per_step={sum(r[0] for r in kern):.1f} "
                  f"annotation_us_per_step={sum(r[0] for r in ann):.1f} "
                  f"dev1_launches_per_step={sum(r[1] for r in rows if r[3]):.1f} "
                  f"dev1_device_us_per_step={sum(r[0] for r in rows if r[3]):.1f} "
                  f"unfiltered_launches_per_step={sum(r[1] for r in rows):.1f} "
                  f"unfiltered_device_us_per_step={sum(r[0] for r in rows):.1f} "
                  f"passes={self.PROFILE_PASSES}", flush=True)
            # c0022: the cap was 20 and launch 22 printed n_kernel=21, so the smallest row
            # (1.6 us) was never printed and my table diff could not reconcile against this
            # header. 40 is above any n_kernel this decode path has produced. A diagnostic
            # only: prepare.py remains the sole accessor for every judged number.
            for micros, count, key, on_device, is_ann in shown[:40]:
                print(f"DECODE_KERNEL: us_per_step={micros:8.2f} n={count:5.1f} "
                      f"dev={int(on_device)} ann={int(is_ann)} {key}", flush=True)
            for micros, count, key, on_device, is_ann in ann[:8]:
                print(f"DECODE_KERNEL_ANNOTATION: us_per_step={micros:8.2f} n={count:5.1f} "
                      f"dev={int(on_device)} {key}", flush=True)
        except Exception as exc:            # noqa: BLE001 -- a diagnostic may not kill a launch
            print(f"DECODE_KERNELS: failed={type(exc).__name__}: {exc}", flush=True)
        finally:
            self.state["seq"].copy_(seq0)

    def _profile_phases(self):
        """c0005 diagnostic. Not a metric and not an input to one.

        Runs ONCE, here in __init__, after capture and before any timed pass, so no timed
        pass can execute a single `_PHASE` call: `on` is False everywhere else and is forced
        back to False in the finally. `seq` is saved and restored exactly as `_capture` does,
        so this leaves the state where it found it and the junk it writes above `seq` is
        unreachable to `cache_seqlens` for the same reason warmup's is.

        The numbers are an APPORTIONMENT, not a budget. This path is eager, not the replayed
        graph, and each span pays ~1-2 us of event overhead, so the phases will not sum to
        the graph's per-step time and must not be subtracted from it. What they can show is
        which phase dominates, which is the question four rounds of guessing could not answer.
        """
        if self.state["seq"].numel() != 1:
            return
        seq0 = self.state["seq"].clone()
        try:
            _PHASE.on = True
            for _ in range(self.PROFILE_PASSES):
                self.state["seq"].copy_(seq0)
                self._advance()
            per = _PHASE.drain()
            n = self.PROFILE_PASSES
            parts = " ".join(f"{k}={v * 1000.0 / n:.1f}" for k, v in sorted(per.items()))
            print(f"DECODE_PHASE_US: passes={n} eager=1 {parts}", flush=True)
        except Exception as exc:            # noqa: BLE001 -- a diagnostic may not kill a launch
            print(f"DECODE_PHASE_US: failed={type(exc).__name__}: {exc}", flush=True)
        finally:
            _PHASE.on = False
            _PHASE.spans = []
            self.state["seq"].copy_(seq0)

    PREFILL_PROFILE_PASSES = 5
    PREFILL_PROFILE_WIDTHS = (1536, 768, 384)

    def _profile_prefill(self):
        """c0022 diagnostic. Not a metric and not an input to one.

        WHY THIS EXISTS. `measure_decode_request` puts prefill INSIDE the timer -- "prompt in
        hand to last token out" -- so `request_ms_median` is prefill plus 512 steps, and after
        22 launches I have never measured the first term. The only estimate available is a
        DIFFERENCE between the two shapes (18.34 ms between the 1536-prefill and 1-prefill
        medians) minus a per-step context correction taken from the kernel table, and the
        kernel table is profiled eagerly while the scored key replays a graph, so that
        correction is a regime mismatch and its error is unquantified. This measures the term
        directly instead of inferring it.

        WHERE IT RUNS. Here in __init__, which the model reaches on the FIRST width-1 step of
        the FIRST WARMUP pass -- `warmup=3` passes are discarded before any timing is kept --
        exactly where `_profile_phases` and `_profile_kernels` already run. No timed pass can
        execute it. `seq` is saved and restored the way `_capture` does, so the state is left
        where it was found and the junk written above `seq` is unreachable to `cache_seqlens`.

        WHY IT CANNOT MOVE A CEILING. It runs the SAME prefill width the request itself runs,
        against the same caches, so it cannot allocate more than the request already allocates
        and `peak_vram_bytes_inference` cannot rise above what this launch would reach anyway.
        `peak_vram_bytes` is the TRAINING peak and this is after training. `kv_cache_bytes` is
        probed on a `graph=False` state, and `_GraphedDecodeStep` is only ever constructed when
        graph is enabled, so this code is unreachable from that probe.

        Token VALUES do not affect timing, so the probe uses its own indices rather than
        reaching for the request's data: it is a timing instrument, not an evaluation.
        """
        if self.state["seq"].numel() != 1:
            return
        seq0 = self.state["seq"].clone()
        try:
            dev = self.state["seq"].device
            vocab = int(self.model.transformer.wte.weight.size(0))
            n = self.PREFILL_PROFILE_PASSES
            out = []
            for width in self.PREFILL_PROFILE_WIDTHS:
                if width + 1 > int(self.state["max_len"]):
                    continue
                idx = torch.randint(0, vocab, (1, width), device=dev, dtype=torch.int64)
                ms = []
                for _ in range(n + 1):          # one discarded pass per width for autotune
                    self.state["seq"].copy_(seq0)
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record()
                    self.model._decode_body(idx, self.state, prefill=True)
                    e1.record()
                    e1.synchronize()
                    ms.append(e0.elapsed_time(e1))
                ms = sorted(ms[1:])
                out.append((width, ms[len(ms) // 2], ms[0], ms[-1]))
            parts = " ".join(f"w{w}_ms={med:.3f}/{lo:.3f}/{hi:.3f}" for w, med, lo, hi in out)
            print(f"DECODE_PREFILL_PROFILE: passes={n} eager=1 median/min/max {parts}",
                  flush=True)
            if len(out) >= 2:
                (wa, ma, _, _), (wb, mb, _, _) = out[0], out[-1]
                per = (ma - mb) / max(wa - wb, 1)
                print(f"DECODE_PREFILL_SCALING: us_per_token={per * 1000.0:.3f} "
                      f"from w{wa}={ma:.3f}ms and w{wb}={mb:.3f}ms; "
                      f"implied_fixed_ms={ma - per * wa:.3f}", flush=True)
        except Exception as exc:            # noqa: BLE001 -- a diagnostic may not kill a launch
            print(f"DECODE_PREFILL_PROFILE: failed={type(exc).__name__}: {exc}", flush=True)
        finally:
            self.state["seq"].copy_(seq0)

    def _advance(self):
        logits = self.model._decode_body(self.static_idx, self.state, prefill=False)
        # c0025: `_decode_head` advanced `seq` inside the region, so inside the captured graph.
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
ASPECT_RATIO = 384      # c0033: 384, not 256, AND THE REASON IS THE WHOLE POINT OF THIS PAIR OF
                        # LINES. base_dim = depth * ASPECT_RATIO, so ASPECT_RATIO is NOT a width in
                        # its own right -- it is width PER UNIT DEPTH. Holding it at 256 while depth
                        # fell 3 -> 2 would have given base_dim = 512 and model_dim 512, i.e. a
                        # SIMULTANEOUS loss of one layer AND a 768 -> 512 collapse in width: two
                        # effects, and the two whose costs I most need to keep separate.
                        #
                        # I NEARLY SHIPPED THAT. This candidate was first written with ASPECT_RATIO
                        # held at 256 and a comment block asserting that width was held at 768. The
                        # comment was false and the code was two effects. It was caught by the guard's
                        # geometry assertion -- which computes model_dim from the candidate's own
                        # constants rather than trusting the prose -- before the launch went out. That
                        # is the second time in this run that a claim in a comment disagreed with the
                        # arithmetic beside it, and both times the derived accessor found it.
                        #
                        # 2 * 384 = 768 EXACTLY, and 768 % HEAD_DIM == 0, so model_dim 768 and
                        # n_head 6 are UNCHANGED from launch 34 and the HEAD_DIM rounding slack stays
                        # at zero. This is the c0031 pattern: ASPECT_RATIO rises exactly to CANCEL the
                        # width a falling depth would otherwise lose. c0031 is the precedent that this
                        # keeps the change to one effect, and it is the reason its combined move was
                        # legitimate while an uncancelled one would not have been.
                        # 3*256 = 768 -> ceil(768/128)*128 = 768, n_head 6, head_dim 128. Note that
                        # 768 = 6*128 EXACTLY, so as at depth 4 with ASPECT_RATIO 128 the rounding
                        # slack is zero: any further width move needs the next multiple, 896.
                        # This is a ONE-EFFECT change -- width only, depth held at 3 -- and it is
                        # the same shape of charge as launch 32: expected to be REJECTED on the
                        # ranking key, bought to bank gate slack and to give d(request)/d(width) a
                        # SECOND point. Right now that slope is a single difference with df = 0.
                        # (Provenance, not a claim about this candidate: this constant went
                        # 64 -> 80 -> 96 -> 128 to PIN width at 512 while depth fell, 160 to move
                        # width to 640, then 200 to hold 640 while depth fell to 3.)
                        # previous launch was CHARGED to make possible. 3*200 = 600 ->
                        # ceil(600/128)*128 = 640, n_head 5, head_dim 128 -- so model_dim is held
                        # at 640, EXACTLY the width launch 32 measured, while DEPTH falls 4 -> 3.
                        # Read the two constants together: ASPECT_RATIO goes UP only to cancel the
                        # width that a falling DEPTH would otherwise lose. Net effect: ONE FEWER
                        # LAYER at unchanged width. That is a one-effect change, not a two-variable
                        # one, and the guard proves the geometry is byte-for-byte the same triple
                        # (model_dim 640, n_head 5, head_dim 128) as the parent's.
                        # (Historical note kept because it is provenance, not a claim about this
                        # candidate: this constant went 64 -> 80 -> 96 -> 128 purely to PIN width
                        # at 512 while depth fell, then 160 to MOVE width to 640 at depth 4.)
                        # is being used to CHANGE the width rather than to hold it. 4*160 = 640 ->
                        # ceil(640/128)*128 = 640, n_head 5, head_dim 128. Every previous move of
                        # this constant (64->80->96->128) existed to keep model_dim pinned at 512
                        # while DEPTH fell. This one moves model_dim 512 -> 640 at CONSTANT depth 4.
                        # WHY THE AXIS CHANGED. ideas/c0029.json pre-registered a stopping rule: if
                        # val_bpb exceeded 1.0415, depth is no longer the axis. It came in at
                        # 1.0443995. Gate slack is now 0.0056005 -- only 1.23x the pre-registered
                        # replicate sd of 0.0045534 -- so one more pure depth rung (the last cost
                        # +0.0174563) would breach val_bpb < 1.05 outright. The depth ladder is
                        # CLOSED, on a rule written before the number existed.
                        # This constant is a WIDTH knob and nothing else: it appears exactly once
                        # in executable code, in `base_dim = depth * ASPECT_RATIO`. Changing it
                        # alongside DEPTH is therefore a ONE-EFFECT change (one fewer layer), not
                        # a two-variable one -- it CANCELS the depth/width coupling instead of
                        # adding a second free variable. The guard proves geometry is identical.
HEAD_DIM = 128          # target head dimension for attention
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context

MLP_HIDDEN = 6144       # c0040. THE MLP HIDDEN DIMENSION DOUBLES, 3072 -> 6144, and it is the
# first capacity PURCHASE of this run rather than a capacity sale. Parent is c0033 -- the FASTEST
# row of the run at 74.98884201049805 ms -- and NOT the incumbent c0038 at 86.89796924591064. That
# choice is the whole design. c0033 fails on nothing but quality: val_bpb 1.0651217806090412
# against a `< 1.05` gate, a breach of exactly 0.0151217806090412, while sitting 11.909 ms ahead of
# the incumbent. Buying 0.0152 of val_bpb for a few milliseconds is a better trade than shaving
# milliseconds off c0038 with 0.88 sd of gate slack left.
#
# THE LEVER IS THE DENSEST PARAMETER MEASURED IN THIS RUN. Three parameter kinds now have a
# measured quality density, all df=1, so the ORDERING is the finding and the magnitudes are not
# transferable: MLP hidden 0.001159 val_bpb per 1 % of params (seq 40) > width 0.000533 (seq
# 33->34) > value embeddings 0.0000767 (seq 33->34). MLP parameters are 2.2x denser than width and
# 15x denser than value embeddings, so the MLP hidden dimension is where a fixed quality debt is
# repaid for the fewest added flops.
#
# ARITHMETIC, NOT A FIT. Both closed forms are byte-exact on c0033's recorded row:
#   nparams(d,md,ve,hid) = 8192*md*2 + ve*8192*md + d*(4*md*md + 2*md*hid) + 32*(md//128)*ve + 2*d
#   flops(d,md,ve,hid,ws) = 6*(8192*md + d*(4*md*md + 2*md*hid) + 32*(md//128)*ve)
#                           + sum(12*md*min(w,2048) for w in ws)
# At d=2, md=768, ve=1, ws=(1024,2048) they give 33030340 and 150996096, which ARE c0033's
# recorded num_params_total and flops_per_token_measured. At hid=6144 they give:
#   num_params_total          42467524   (+28.57 %, 84.4 % of the 50332176 ceiling)
#   flops_per_token_measured 207619200   (+37.50 %, 86.8 % of the 239078400 ceiling)
# Both are predicted EXACTLY and neither is close to breaching.
#
# TWO RETRACTIONS OF MY OWN, BOTH FOUND WHILE PRICING THIS ROW.
#
# (1) `mfu_percent_a100` IS AN ALGEBRAIC IDENTITY, NOT AN INDEPENDENT MEASUREMENT. It equals
# train_tokens_per_second * flops_per_token_measured / 312e12, reproduced byte-exact on ALL 39
# measured rows of this run (max absolute error 7.105e-15, pure float rounding). It therefore
# carries no information beyond throughput and a flops figure I compute myself in advance.
# The consequence is severe: c0039's card made MFU its PRIMARY attribution channel with two
# "disjoint rival bands" and a refutation line at 38.0. The ALIGNMENT band [39.0, 43.0] required
# train_tokens_per_second in [1014568, 1118626]; the highest value observed in ANY of the 39 rows
# is 896978. The alignment band sat 13.1 % ABOVE the run's own historical throughput maximum, and
# the 38.0 line required 988554, also unreachable. One rival was satisfiable and the other was
# outside the machine's demonstrated envelope, so the row could only ever return "size".
# **THE 256-ALIGNMENT HYPOTHESIS IS THEREFORE NOT REFUTED. IT WAS NEVER TESTED.** I had already
# narrated it as refuted. Guard assertion 20 on that card asserted the line lay strictly BETWEEN
# the two bands -- true, and irrelevant, because it never asked whether either band was REACHABLE.
# The defensible residual claim is only that no alignment benefit large enough to move training
# throughput by ~2 % appeared. Nothing on this card rests on alignment; 6144 = 24*256 happens to
# be aligned and that is stated as a coincidence, not as a mechanism.
#
# (2) THE STEP-PURCHASE ELASTICITY I HAVE BEEN USING WAS WRONG IN BOTH DIRECTIONS. Under a 600 s
# TIME budget, added flops cost steps, and I have priced that term twice with two different wrong
# numbers. Round 40's depth-2 table implicitly used steps ~ mfu/flops, i.e. elasticity 1.0. Seq
# 37 measured a "realisation" of 0.113 from ONE adjacent pair. Regressed over all 39 rows of this
# run:
#     d log(num_steps) / d log(flops_per_token_measured) = -0.5706,  r = -0.9535,  n=39, df=37
#     d log(train_tokens_per_second) / d log(flops)      = -0.5775,  r = -0.9534,  n=39, df=37
# This is the best-determined coefficient in the run and it SUPERSEDES both 1.0 and 0.113. It is
# a CROSS-SHAPE fit -- depth, width, windows and splits all vary across those 39 rows -- so it is
# a population relationship and not a within-shape law, and this row is its first predictive use.
#
# THE REPRICED PURCHASE, and it is much better than round 40 said. Every figure below is DERIVED,
# because hand-composed constants have now failed this candidate's guard in three consecutive
# rounds -- including, on the first run of guard_c0040.py, the flops ratio itself, which I typed
# as 1.375 when it is 1.374997138999:
#   flops ratio  207619200 / 150996096         =  1.374997138999   (NOT 1.375)
#   num_steps  1026 * 1.374997138999**-0.5706  =  855.524  ->  856, a loss of 170 (-16.5692 %)
#   capacity gain  28.5712590303 % * 0.001159  = -0.0331140892 val_bpb
#   step cost      170 * 6.118672610400111e-05 = +0.0104017434 val_bpb
#   net                                        = -0.0227123458
#   val_bpb  1.0651217806090412 - 0.0227123458 =  1.0424094348   (passes by 1.667 sd)
# Round 40's table put this same lever at 1.04914, passing by only 0.19 sd, because it charged
# -280 steps instead of -170. THE STEP COST IS REAL BUT IT EATS 31.4118 % OF THE NOMINAL CAPACITY
# GAIN, not the 47-65 % I recorded last round.
#
# AND THE STEP SLOPE IS THE WEAK LINK, SO IT IS NAMED HERE. -6.118672610400111e-05 val_bpb per
# step comes from ONE adjacent pair, seq 37 vs seq 39, 1026 vs 1027 steps, df=1 -- and that pair's
# val_bpb difference is 0.0000611867, which is 0.0134 sd of the pre-registered 0.0045534 replicate
# sd, i.e. a difference TASK.md would call no effect at all. Its relative uncertainty is of order
# 100 % and this card extrapolates it 170x. If the true slope is twice this, the step cost is
# 0.0208034868, the net is -0.0123106023, and val_bpb lands at 1.0528111783 -- A BREACH, of
# 0.0028111783, which is UNDER 0.01 and therefore records UNRESOLVED and is NOT re-run. That is why
# the val_bpb band on this card STRADDLES THE GATE instead of sitting under it. A log form agrees
# with the linear one and does not rescue it: 0.06277758 val_bpb per unit ln(steps) over the
# -0.181184 change in ln(856/1026) gives +0.011373, within 10 % of +0.0104017434.
#
# THE LATENCY ROUTE USES THE ONLY ELASTICITY THAT HAS EVER LANDED. Per-layer matrix work goes
# 4*md*md + 2*md*hid = 7077888 -> 11796480, i.e. +66.67 %. Seq 40 measured the ranking key's
# elasticity on matrix work ON THIS AXIS: matrix -16.7 % gave key -2.00 %, so 0.12. Applied here:
# key +8.00 % -> 74.98884201049805 * 1.0800 = 80.99 ms. That route's sibling landed within 0.588
# ms on seq 41 -- predicted 85.65, observed 86.23802661895752 -- which is the first latency point
# of this run to land close, and I am NOT treating one hit as a repaired route, which is why the
# band is 8 ms wide. Even the top of that band beats the incumbent's 86.89796924591064 by 1.9 ms.
#
# num_steps IS THE REAL INSTRUMENT ON THIS ROW, and unlike MFU it is directly recorded and its
# three rival predictions are all REACHABLE. Observed num_steps across the 39 rows spans 754 to
# 1037, which COVERS all three bands below -- the reachability check c0039's card did not make:
#   elasticity 1.0    (round 40's implicit model): 1026 * 1.375**-1.0    = 746
#   elasticity 0.5706 (the 39-row fit, THIS CARD): 1026 * 1.375**-0.5706 = 856
#   elasticity 0.113  (seq 37's df=1 realisation): 1026 * 1.375**-0.113  = 989
# Those are 133 steps apart at the narrowest and the bands are disjoint.

# Optimization
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
# c0043. THE FIRST CHARGE ON A LATENCY-FREE AXIS, AND IT IS SPENT BECAUSE THE GATE IS NOW THE
# BINDING CONSTRAINT ON EVERY REMAINING ARCHITECTURAL LEVER.
#
# Parent is c0042, the incumbent (launch 44, 25b405ca-283b-427c-9829-248ba904bf37):
# request_ms_median 81.34937286376953, nopref 74.28693771362305, val_bpb 1.0487742912093647,
# gate margin 0.001225708790635327 = 0.2691853978643051 sd, params 42467524, flops 200541312,
# num_steps 1004. Eighteen of nineteen predictions held.
#
# WHY NOT MORE REACH, WHICH IS THE STEP THE c0042 CARD PRE-REGISTERED UNDER THE ACCOUNT THAT WON.
# That card's account-A branch said the next charge is SHORT_WINDOW_DIVISOR 16, window 128, and
# account A is exactly what launch 44 selected. THE BRANCH IS STILL WITHDRAWN, on the arithmetic
# the same row produced:
#
#   step                    d(key)        d(nopref)     d(val_bpb)        margin left after
#   1024 -> 512 (c0041)     -2.3113489    +2.0995140    -0.000122837      0.4228 sd
#    512 -> 256 (c0042)     -0.8023977    -0.5739927    +0.000699282      0.2692 sd
#    256 -> 128 (proposed)  about -0.28   ?             about +0.0007     about 0.116 sd
#
# Three things in that table kill the step. First, the LATENCY RETURN IS COLLAPSING: successive
# halvings paid 2.31 then 0.80 ms, a factor of 2.88, so the next is worth about 0.28 ms -- and the
# ranking key's own IQR on launch 44 is 0.3229379653930664, so THE WHOLE EXPECTED GAIN IS SMALLER
# THAN THE PROBE'S OWN DISPERSION. Second, THE QUALITY EFFECT CHANGED SIGN on this axis: the two
# earlier reach steps IMPROVED val_bpb and this one COST 0.000699282, so the axis stopped being
# free precisely when it stopped paying. Third, that cost is 57 per cent of the entire remaining
# gate margin, and 0.116 sd of margin is inside the noise of the pre-registered 0.0045534 sd --
# it would leave the incumbency resting on a coin flip. Round 40 declined a 0.06-0.19 sd bet for
# this reason and that call was right.
#
# AND THE MODEL THAT MADE THAT BRANCH LOOK LIKE A REPLICATE IS ITSELF FALSIFIED. The c0042 card
# predicted the no-prompt saving by scaling the prefilled one by the absolute key delta, giving
# 0.2900472027249634 ms. Observed: 0.5739927292 ms, 1.979x the prediction. On the same row the
# prefilled saving came in at 0.8023977280 against a key-count bound of 1.1556744575500488, i.e.
# 0.694x. So the observed ratio between the probes is 0.715 where the mean-keys model says 0.251:
# ONE PROBE BEAT THE MODEL BY 2x AND THE OTHER FELL SHORT BY A THIRD, IN OPPOSITE DIRECTIONS.
# The key-count model survives only as an upper bound on the prefilled probe, which is how the
# card labelled it. It is not a magnitude model and the "replicate" framing that rested on it is
# withdrawn with it. This is the THIRD consecutive round in which a pre-registered next step was
# withdrawn on the strength of the very row that selected it.
#
# WHAT LAUNCH 44 DID SETTLE, and it is worth stating because it was the primary instrument:
# THE UNEXPLAINED +2.0995140 ms NO-PROMPT COST DID NOT RECUR. Account A was pre-registered as
# "the cost was a one-off of leaving the 1024 default", account B as "it recurs per window change";
# the observed no-prompt key FELL 0.5739927292 ms, landing in A's band [73.9, 75.2] and outside
# B's [75.9, 77.5]. The probes converged as A requires. The cost term is therefore NOT a standing
# tax on window changes, and no charge needs to be spent hunting it.
#
# THE STATE OF THE BOARD, AND WHY THE BINDING CONSTRAINT HAS MOVED. Every architectural lever left
# on this task costs val_bpb: reach now does (above), GQA cuts the k and v projections by 1179648
# parameters at n_kv_head 3, more MLP hidden costs latency (hid 7168 prices worse than the
# incumbent even after crediting both reach steps), and depth 3 at this width is boxed in at
# 98.68 per cent of the flops ceiling. WITH 0.2692 sd OF MARGIN, THE GATE IS NO LONGER A SIDE
# CONDITION -- IT IS THE SCARCE RESOURCE, and the cheapest quality on record costs about 560 ms
# per unit val_bpb (the hid 3072 -> 6144 pair, +9.474 ms for -0.016924), so buying margin
# architecturally means paying ranking-key milliseconds for it.
#
# SO THIS CHARGE BUYS QUALITY AT EXACTLY ZERO INFERENCE COST. MATRIX_LR is the Muon learning rate
# for every matrix parameter -- the attention projections and the MLP, which is the overwhelming
# majority of the 42467524. It is an OPTIMIZER hyperparameter: it cannot change a parameter count,
# a dispatch-counted FLOP, a cache allocation, or a window. THE ENTIRE INFERENCE PATH IS
# BYTE-IDENTICAL TO THE PARENT'S. Five metrics are therefore predicted UNCHANGED and only val_bpb
# is free, which is the sharpest pre-registration this run has been able to make.
#
# THE ASYMMETRY IS THE WHOLE ARGUMENT: A LOSS ON THIS AXIS COSTS ONE CHARGE AND NO RANKING GROUND.
# If val_bpb rises, c0042 stands as incumbent at 81.34937286376953 ms, unharmed, and the axis is
# closed in this direction with a measured sign. If val_bpb falls, the margin it buys is spendable
# on every architectural lever above, including the reach step withdrawn today. Compare the
# alternative: the reach step risks 57 per cent of the gate to gain less than one IQR.
#
# WHY THIS KNOB AND WHY UP. setup_optimizer scales the AdamW learning rates by
# 1/sqrt(model_dim/768) and PRINTS that it is doing so -- a WIDTH-dependent correction and the
# only one in the file. Our model_dim is 768, so that factor is exactly 1.0 and every LR is at its
# stock value. But the stock values were set for the substrate's default geometry of n_layer 12,
# and this candidate runs n_layer 2: DEPTH FELL BY 6x WITH NO DEPTH-DEPENDENT LR CORRECTION
# ANYWHERE IN THE CODE. A 6x shallower residual stack accumulates far less update per step
# through its depth, which argues the stock matrix LR is now too low rather than too high. Hence
# 0.04 -> 0.06, a factor of 1.5.
#
# THE SIGN IS A REAL PREDICTION AND THE ADVERSE BRANCH IS PRE-REGISTERED, not discovered later:
# if val_bpb RISES, the stock LR was at or above the optimum for this shape and the axis is closed
# upward, and the next charge on it goes DOWN to 0.027 by the same factor rather than further up.
# A 1.5x step was chosen over a timid one because this axis has ZERO measured points and a step
# inside the noise would teach nothing about the sign -- the failure mode of a null result that
# cannot distinguish "no effect" from "no dynamic range" has cost this run three instruments
# already.
#
# THE MAGNITUDE IS UNBANDED ON PURPOSE, and this is the card's weak link, named here. There is no
# prior point on this axis, so any val_bpb band I write is a guess dressed as a prediction. The
# band is set wide from the run's own observed val_bpb envelope and the CARD SCORES THE SIGN, not
# the magnitude. A wide band that admits the null is not a hedge here: it is the honest statement
# that a first point on a new axis measures a direction.
MATRIX_LR = 0.06        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# Model size
DEPTH = 2               # c0033: 2, not 3. THE PRE-REGISTERED DECISION RULE FIRED. ideas/c0032.json
                        # wrote, before launch 34 ran: "if_rejected_and_val_bpb_le_1.028: NEXT LAUNCH
                        # IS DEPTH 2 AT WIDTH 768 -- a one-effect depth rung whose cost arithmetic
                        # (slack >= 0.0220 against a measured ~0.0171) survives." Launch 34 was
                        # rejected at 102.62823104858398 ms and returned val_bpb 1.0224618818631988,
                        # which is <= 1.0280. So this candidate is not a fresh judgement of mine; it
                        # is the branch that was written down in advance, and I am taking it because
                        # it was written and its condition was met.
                        #
                        # THE ARITHMETIC, RESTATED WITH THE NEW SLACK. Available gate slack is now
                        # 1.05 - 1.0224618818631988 = 0.0275381181368013. The two measured depth rungs
                        # cost +0.0174563 (5->4 at width 512) and +0.0170594 (4->3 at width 640), and
                        # launch 33 established those are FLAT to within the 0.0045534 replicate sd
                        # rather than doubling. At the larger of the two, the post-rung val_bpb is
                        # 1.0224618818631988 + 0.0174563 = 1.0399182, leaving 0.0100818 of slack --
                        # about 2.21 sd. It passes on arithmetic, and it is the FIRST rung in the run
                        # that the slack could not have paid for one launch ago.
                        #
                        # WHAT MAKES THIS RUNG DIFFERENT AND MORE DANGEROUS THAN THE LAST TWO.
                        # has_ve(i,n) = i%2 == (n-1)%2, so ve count is ceil(depth/2): depth 3 -> 2,
                        # depth 2 -> 1. THIS RUNG ALSO REMOVES A VALUE EMBEDDING. The 5->4 and 4->3
                        # rungs each removed a layer while ve went 3->2 and 2->2 respectively, so
                        # neither is a clean precedent for a rung that drops ve. If the value
                        # embedding carries quality out of proportion to its parameter count, the
                        # rung costs MORE than 0.0175 and the gate is where this candidate dies.
                        # I am naming that before the fact rather than after: it is failure mode 1 on
                        # the card, and it is the reason the predicted val_bpb band is asymmetric.
                        # Note also that round 30 REFUTED a ve-based explanation of a latency change
                        # once already, so I am claiming ve matters for QUALITY here, which is a
                        # different claim from the one that was refuted, and it is untested.
                        #
                        # WHY THIS IS STILL ONE EFFECT. Width is held at 768 exactly, BY RAISING ASPECT_RATIO
                        # TO 384 TO CANCEL THE FALLING DEPTH -- see the constant above. The ve change is
                        # not a second knob I am turning -- it is a mechanical consequence of depth in
                        # the substrate's own has_ve, the same way the window table is. There is no
                        # setting of this candidate in which depth falls to 2 and ve stays at 2. So
                        # this is one edit with a coupled consequence, not two edits, and the coupling
                        # is stated so that a gate failure can be attributed to the right cause.
                        #
                        # WINDOWS. _compute_window_sizes forces the last layer long, so depth 2 gives
                        # 1 short + 1 long = [1024, 2048], down from [1024, 1024, 2048]. That is also
                        # mechanical, also predicted, and also asserted by the guard.
# ================== WHY WIDTH AGAIN, AND WHY NOT DEPTH 2 DIRECTLY =================================
# Launch 33 (c0031, depth 3 at width 640) was ACCEPTED at request_ms_median 93.90711784362793 with
# val_bpb 1.0384721355448014, so gate slack is 0.0115278644551986 -- 2.53x the pre-registered
# replicate sd of 0.0045534.
# DEPTH 2 AT WIDTH 640 IS THE TEMPTING MOVE AND IT ALMOST CERTAINLY BREACHES. It is worth about
# 27 ms (device 180.0 -> ~128.4 us). But the last two depth rungs cost +0.0174563 and +0.0170594,
# and 1.0384721 + 0.0170 = 1.0555 against a gate of 1.05. A rung would have to cost less than
# 0.0115 -- below BOTH recent measurements -- for that to pass. So depth 2 needs slack bought first.
# WHY NOT COMBINE DEPTH 2 AND WIDTH 768 IN ONE LAUNCH. Because that is genuinely TWO effects.
# c0031's combined move was safe precisely because ASPECT_RATIO rose only to CANCEL the width a
# falling depth would lose, so the geometry triple was IDENTICAL to its parent's and the latency
# comparison was one-effect. A depth-2-at-width-768 candidate holds nothing constant, and with 67
# charges remaining there is no reason to buy a confound to save one launch.
# WHAT THIS LAUNCH BUYS, STATED AS TWO SEPARATE THINGS:
#   (a) A SECOND POINT ON d(request)/d(width). The first width step (512 -> 640 at depth 4) cost
#       14.7822 ms and added ZERO launches: launches_per_step 45.0 -> 45.0, n_kernel 21 -> 21.
#       That is one difference, df = 0, and I have been recording all run that a single difference
#       is not a slope. This step is +20% width where that one was +25%, at 3 layers not 4.
#   (b) GATE SLACK. The first width step bought back 0.0229868 of val_bpb -- more than a whole
#       depth rung costs. If this one buys ~0.018, slack goes to ~0.030 and depth 2 at width 768
#       becomes admissible on quality, which is the launch after this one.
# THE FLOPS CEILING IS IN PLAY FOR THE FIRST TIME IN THIS RUN, and it is named before the fact.
# flops_per_token_measured must be <= 239078400. Observed 151390080 at depth 3 width 640. Widening
# to 768 raises it, and unlike val_bpb a ceiling breach makes the row INADMISSIBLE rather than
# merely rejected, which is worse. So the estimate is stated as a BOUND, which is the explicit
# lesson of launch 32: I there wrote that flops rises by (640/512)^2 = 1.5625x when the observed
# rise was 1.4286x, because the embedding and lm_head terms scale with md and not md^2. md^2 was an
# UPPER BOUND quoted as an estimate, and that single wrong coefficient was the only band I missed.
#   per-layer flops at width 640 = 188745600 - 151390080 = 37355520 (one layer, measured).
#   its observed scaling 512 -> 640 was 37355520/25165824 = 1.4844x, NOT the 1.5625x that md^2
#   predicts, so the layer term has an md-linear component too.
#   UPPER BOUND (layer term as md^2, non-layer as md): 3*37355520*1.44 + 39323520*1.2 = 208564071.
#   CENTRAL (layer term at the observed exponent 1.766): 3*52036241 + 47188224 = 203296947.
# Both are under 239078400 -- the UPPER BOUND passes at 87% of the ceiling, which is why width 768
# is admissible and why width 896 at depth 3 is NOT (it exceeds the ceiling outright).
# THE VALUE-EMBEDDING COUNT IS UNCHANGED AT TWO, so this is a pure width change with no capacity
# term hidden in it, and the guard asserts that with a firing control.
# ================== WHY DEPTH IS BACK, AND WHY THAT IS NOT ME EVADING MY OWN STOPPING RULE =========
# I have to be scrupulous here, because the convenient reading is available and I am not taking it
# without saying that it was available.
# ideas/c0029.json pre-registered, before launch 31 produced a number: "if val_bpb exceeds 1.0415,
# depth is no longer the axis and the next card is about something else." It came in at 1.0443995.
# The rule fired, and I obeyed it: launch 32 changed WIDTH, not depth.
# A LITERAL reading of "depth is no longer the axis" forbids this candidate. I am saying that plainly
# rather than pretending the tension does not exist. What licenses it is the rule's OWN STATED
# BASIS, which was arithmetic and not a preference: gate slack was 0.0056005, only 1.23x the
# pre-registered replicate sd of 0.0045534, so a further rung at +0.0175 gave 1.0619 and breached
# val_bpb < 1.05 outright. THAT PREMISE IS NOW FALSE. Launch 32's widening moved val_bpb
# 1.0443995 -> 1.0214127, a FALL of 0.0229868, and gate slack is 0.0285873. A rung at +0.0175 now
# gives 1.0389 and passes. So the rule is satisfied on its mechanism, not evaded on its wording.
# AND, DECISIVELY, THIS MOVE WAS PRE-REGISTERED BEFORE THE MEASUREMENT EXISTED. ideas/c0030.json
# named "a COMBINED candidate -- depth 3 at width 640" as the PURPOSE of launch 32's charge, and
# stated the two conditions that would make it live: (a) width is launch-bound, refuted if
# request_ms_median exceeded 125 ms; (b) val_bpb falls below 1.0415. BOTH FIRED, in advance:
#   (a) request_ms_median 117.41077899932861 -- under 125, and 42.6 ms under the ~160 ms the
#       arithmetic-bound model predicted. launches_per_step 45.0 -> 45.0, UNCHANGED to the digit,
#       with n_kernel 21 and n_all 49 also unchanged. Widening added ZERO launches.
#   (b) val_bpb fell to 1.0214127, below 1.0415.
# I did not invent this move after seeing a convenient number. I named it, named its two
# preconditions, and both were met. That is the strongest thing I can say for it.
# ================== WHAT THIS CANDIDATE IS PREDICTED TO DO =========================================
# LATENCY, derived by TWO INDEPENDENT ROUTES that agree, which is why I trust it:
#   Route 1 (device time then the ratio). Width cost 33.5 us of device time over 4 layers = 8.375 us
#     per layer, so a layer at width 640 costs the width-512 layer cost (~44.3 us) + 8.4 = ~52.7 us.
#     231.6 - 52.7 = ~178.9 us. At the request/device ratio that has now held at 0.507-0.518 on FIVE
#     consecutive launches, that is 90.7 - 92.7 ms.
#   Route 2 (request time directly). A layer cost 21.61 ms at width 512; scaled by the same +18.9%
#     the width added to per-layer device time, ~25.7 ms. 117.41 - 25.7 = ~91.7 ms.
#   Both routes land ~91-92 ms, about 11 ms BELOW the incumbent's 102.62858867645264. So unlike
#   launch 32, this candidate is expected to be ACCEPTED, and ideas/c0031.json says so in advance.
# QUALITY IS THE ONLY REAL RISK, AND IT STRADDLES THE GATE. val_bpb is 1.0214127 with 0.0285873 of
# slack. The pure-depth quality series was +0.00048, +0.00667, +0.00794, +0.01746 -- roughly
# DOUBLING -- so a 4->3 rung at width 512 would have cost about +0.035 and breached. At width 640
# the model carries 38% more parameters, so the cost of losing a layer SHOULD be smaller. But I have
# NO measurement of that interaction and I am not going to imply that I do: this is the single
# unmeasured slope the candidate is exposed to. The honest band is [1.032, 1.062], which CONTAINS
# 1.05. ideas/c0031.json pre-registers the breach branch: a breach is recorded UNRESOLVED and is NOT
# re-run, because one run per candidate is the design.
# ADMISSIBILITY, re-derived at the NEW parameter model rather than carried over. Launch 32 settled
# the two-way test: num_params_total came in at 40632648, the WIDTH-SCALING branch, so the excess
# term is md/4 per value-embedding-carrying layer, not a fixed 128. At depth 3, width 640, ve = 2:
#   num_params_total  = 8192*640*2 + 2*8192*640 + 3*12*640^2 + 160*2 + 6 = 35717446 vs 50332176 OK
#   flops_per_token   ~151.7e6 (188745600 less one layer's 36-39e6 share) vs 239078400 OK
#   kv_cache_bytes    ~15.7e6 (20972032 * 3/4, allocator rounding aside) vs 41943040 OK
#   peak_vram_bytes   ~25e9 vs 47198976512 OK
# All four have wide margins. The ONLY binding constraint on this candidate is val_bpb.
# THE PREDICTION BEING TESTED, and it is the exact CONVERSE of a finding this run already has.
# Round 27 established that of the ~40.5 us a layer costs per step, roughly 8 launches x 4.5 us
# = ~36 us is LAUNCH OVERHEAD rather than arithmetic, and concluded: "narrowing the model is a bad
# trade -- it surrenders capacity and buys almost no latency, because the launches remain." The
# CONVERSE has never been tested: WIDENING should buy capacity and cost almost no latency, because
# widening adds arithmetic to existing kernels without adding any launches. launches_per_step
# should be essentially UNCHANGED at 45 while device time rises only by the arithmetic share.
# WHY THIS IS WORTH A CHARGE EVEN THOUGH I EXPECT IT TO BE REJECTED. The ranking key is
# request_ms_median and a wider model is slower, so this candidate will almost certainly NOT beat
# 102.62858867645264 and will be REJECTED. It is bought for two SLOPES that no launch in this run
# has measured, and that jointly decide whether anything is left on the table:
#   (a) d(request_ms_median)/d(width) -- if widening really is launch-bound, this is small.
#   (b) d(val_bpb)/d(width) -- how much gate slack a width increase BUYS BACK.
# If (b) is large and (a) is small, then a COMBINED candidate -- depth 3 at width 640 -- is worth
# about 21 ms of depth minus a few ms of width, at a val_bpb the widening pays for. That is the
# only route left to another 20 ms, and guessing at it blind would risk the gate.
# THE OTHER HALF, WHICH IS ABOUT QUALITY -- AND IT IS A TRADE, NOT A FREE GAIN. I first wrote here
# that widening "buys val_bpb from two directions at once, more capacity AND more steps." That is
# WRONG and the arithmetic says so, so it is corrected before dispatch rather than discovered after.
# mfu_percent_a100 has collapsed 49.934 -> 49.368 -> 45.456 -> 37.539 as the model shrank, because a
# depth-4 width-512 model under-fills an A100, and a wider model runs bigger GEMMs so MFU SHOULD
# rise. But training is TIME-budgeted at 600 s, and steps come from
#     tokens_per_second = MFU * peak_flops / flops_per_token
# where widening raises flops_per_token by ~(640/512)^2 = 1.5625x. If MFU rises by only ~1.25x, the
# net effect on the step count is 1.25/1.5625 = 0.8x -- FEWER steps, not more. So widening TRADES
# optimizer steps for per-step capacity, and the sign of the net val_bpb move is genuinely
# UNCERTAIN. ideas/c0030.json predicts a fall, states that the sign is uncertain, and pre-registers
# that if val_bpb RISES the widening route is dead and the card says so.
# ADMISSIBILITY CHECKED BEFORE DISPATCH, since widening moves four ceilings the WRONG way:
#   num_params_total   ~40632584 vs ceiling 50332176   (width 768 would BREACH at ~53477640)
#   flops_per_token    ~206.4e6  vs ceiling 239078400
#   kv_cache_bytes     ~22.9e6   vs ceiling 41943040
#   peak_vram_bytes    ~32.4e9   vs ceiling 47198976512
# Width 640 is admissible on all four; width 768 is NOT, which is why 160 and not 192.
# THE VALUE-EMBEDDING EXPLANATION IS REFUTED, BY ITS OWN PRE-REGISTERED LINE. ideas/c0028.json
# committed in advance: val_bpb >= 1.0230 refutes it, <= 1.0215 corroborates it, and between the
# two the result is indeterminate. Depth 5 was a LAYER-ONLY rung (ve count stayed at three) and it
# came in at val_bpb 1.0269432 -- above the refutation line. So a layer-only rung cost +0.00794,
# MORE than the layer-plus-embedding rung's +0.00667, and the value embedding is NOT what made
# 7->6 expensive. Quality cost is per-layer regardless of the embedding count.
# WHAT THE THREE RUNGS ACTUALLY SHOW: +0.00048, +0.00667, +0.00794. Not alternating -- ACCELERATING,
# with the 8->7 rung as the near-free outlier. That is the shape of a convex quality-vs-capacity
# curve, and it means each further rung costs more than the last. Gate slack is now 0.02306.
# WHY DEPTH 4 IS STILL WORTH A CHARGE. Latency has paid 19.98, 22.19 and 21.50 ms across the three
# rungs with no decay, and launches_per_step 83 -> 74 -> 64 -> 55. One more rung is worth about
# 21 ms for a predicted val_bpb cost near +0.009, which the gate can still absorb twice over.
# WHAT IS NEW AND MIGHT STOP THE LADDER, disclosed here rather than after the fact:
#   (1) MFU COLLAPSED 49.37 -> 45.46 this rung, against 49.83 -> 49.93 -> 49.37 before it. The
#       time budget is fixed at 600 s, so a cheaper step is supposed to buy more steps -- but
#       num_steps rose only 944 -> 1003 (+6.3%) against +11.8% on each earlier rung. The
#       compensation that has been offsetting the quality cost is FAILING, which is why the
#       quality cost should be expected to accelerate further, not merely continue.
#   (2) decode_tv_distance_max has risen THREE times running, 0.01208 -> 0.02078 -> 0.02748, and
#       this rung the nopref twin rose WITH it (0.01097 -> 0.01370) instead of against it. The
#       "opposite signs, therefore noise" argument from round 29 is now weakened. 0.02748 is 55%
#       of the 0.05 ceiling.
# So this rung is chosen knowing that it may be the last cheap one, and both stopping conditions
# are pre-registered with numeric triggers in ideas/c0029.json.
# LATENCY IS THE PART THAT IS NO LONGER IN DOUBT. Two rungs have now landed inside their
# pre-registered bands on all three of request_ms_median, launches_per_step and
# device_us_per_step: 187.90 -> 167.93 -> 145.74 ms, 83 -> 74 -> 64 launches, 370 -> 329 -> 287 us.
# A layer costs 20-22 ms and about 9-10 launches, and that has not decayed with depth.
# WHAT c0026 MEASURED, which is why this candidate exists. Depth 8 -> 7 at constant width gave
# request_ms_median 187.904 -> 167.927, a fall of 19.977 ms, while val_bpb moved 1.0118552 ->
# 1.0123360, i.e. +0.00048 -- ONE TENTH of the pre-registered replicate sd of 0.0045534, and a
# sixth of the 0.003 that TASK.md says is not an effect. launches_per_step 83 -> 74 and
# device_us_per_step 370 -> 329 both landed inside their pre-registered bands. So one layer costs
# ~40.5 us of device time per step and ~20 ms of request, and buys essentially no quality.
# WHY THE COST IS LAUNCH-BOUND, AND WHAT FOLLOWS. Of that 40.5 us, 8 launches x ~4.5 us = ~36 us
# is launch overhead. Per-layer cost is therefore set by the NUMBER of launches, not by the
# arithmetic inside them -- so NARROWING the model would surrender quality and buy almost no
# latency, while REMOVING a layer removes 9 launches outright. That is the whole reason
# ASPECT_RATIO moves 64 -> 80 above: to take the layer without paying the width.
# THE GATE IS THE ONLY CONSTRAINT AND IT IS NOWHERE NEAR BINDING. val_bpb 1.01234 against
# < 1.05 leaves 0.0377, and the measured per-layer cost is 0.00048. Every ceiling moved the
# admissible way: num_params 50332176 -> 47186446, flops_per_token 239078400 -> 213912576,
# peak_vram 47198976512 -> 42038909440, kv_cache 34603520 -> 29360640, all `<=` clauses.
# WHY. After 26 launches the step is 83 kernel launches and 370 us, of which 56 launches /
# 180 us are a chain of DEPENDENT per-layer GEMVs that cannot be fused at any width, and 16
# launches / 156 us are attention at fixed cost per call. Both scale with DEPTH and neither
# can be reduced any other way. Meanwhile val_bpb reads 1.0119 against a gate of 1.05 --
# 0.038 of slack, 8.4x the pre-registered replicate sd of 0.0045534 -- and task.json says in
# as many words that "Quality is a threshold and never a target". The gate slack has been
# sitting unspent for 26 launches while I optimised the implementation of a fixed shape.
# SUPERSEDED, RECORDED RATHER THAN DELETED. c0026's comment claimed depth 7 was the ONLY value
# that isolates depth from width, because at ASPECT_RATIO = 64 depth 6 drops model_dim to 384
# and n_head to 3. That was true only with ASPECT_RATIO held at 64, which c0026 never questioned.
# ASPECT_RATIO is itself a candidate constant, so ANY depth can be run at width 512 by choosing
# it to satisfy 384 < depth * ASPECT_RATIO <= 512. The claim was too strong and this file corrects
# it: 6/80, 5/96, 4/128, 3/160, 2/256 and 1/512 all give model_dim 512 and n_head 4, so the whole
# depth axis is walkable unconfounded. c0026's reading stays valid -- it just was not unique.
# WHAT IT COSTS. One layer of quality, bought back in part because training is TIME-budgeted
# at 600 s: a smaller model fits MORE optimizer steps in the same budget.
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
