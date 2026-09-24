"""
Autoresearch pretraining script. Single-GPU, single-file.
Cherry-picked and simplified from nanochat.
Usage: uv run train.py

struct_kv_heads_v1: half as many K/V heads as query heads.

`build_model_config` tied `n_kv_head` to `n_head`, so at DEPTH=8 the model had 4 query
heads and 4 K/V heads and a GQA ratio of 1. This sets `n_kv_head = n_head // 2` = 2, which
is the only change: `CausalSelfAttention` already asserts nothing stronger than
`n_kv_head <= n_head` and `n_head % n_kv_head == 0`, and both flash-attention calls take the
GQA ratio from the shapes they are given. `head_dim` (128) and `n_embd` (512) are untouched,
so every learning rate, the residual width and the rotary table are the champion's.

What it costs: `forward()` changes, so this is the first candidate in the run whose `val_bpb`
can move. Halving the cached width also halves `c_k`/`c_v` and, because `value_embeds` are
sized `n_kv_head * head_dim`, the four value-embedding tables as well -- 8,388,608 of the
10,486,016 parameters removed. That confounding is deliberate and recorded: the queued item
prices the joint change first, and a full-width value-embedding table with a down-projection
is the de-confounded follow-up if this one is ineligible.

What it buys, predicted before the launch from a byte model that closes exactly on all four
readings measured in this run (reference and champion, both shapes):

    kv_cache_bytes        = 8 * (2*2048*2*128 + 2*2048*2*2) + 512 + 1,048,576 =  9,568,768
    nopref_kv_cache_bytes = 8 * (513*512 + 9*512) + 512                       =  2,138,624
                            (the 4,104 B of scales at 513 slots round up to 9 blocks)

against the champion's measured 18,088,448 and 4,272,640. The 1,048,576 B residue is the
unattributed charge present at max_len 2048 and absent at 513; the print-only probe added
before the reporter attributes it without buying a launch.

    [struct_kv_heads_v1 measured 8,520,192 and 2,138,624. The residue it predicted was not
    there, and its own probe printed "residue vs exact+512 : 0". The reason is in the law
    below: the residue appears with eight or more allocations of exactly 2,097,152 B, and
    halving n_kv_head made each layer's codes exactly 1,048,576 B instead.]

ext_compose_kvheads_extent_v1 -- that champion with the window-extent mechanism of
`ext_window_extent_v1` (launch 2) and the fused allocation of `ext_fused_alloc_v1` (launch 4)
folded in. Decode path only; no new mechanism. This is the same fold that
`ext_compose_int8_extent_v1` (launch 6) measured on the previous champion, where it landed on
its predicted byte exactly and where its recorded lesson was "fold it onto whatever the
champion is".

1. EXTENT. `window_pattern="SSSL"` at DEPTH=8 makes layers 0,1,2,4,5,6 short (window 1024)
   and layers 3,7 long (window 2048, and the last layer is forced long). A short layer's
   decode call passes `window_size=(1024, 0)`, so a width-1 query appended at write position
   p reads keys in [p-1024, p] and never an older one: 1025 slots are all it ever needs live,
   against the 2048 it was allocated. Such a layer gets `1024 + 1 + DECODE_CACHE_SLACK` slots
   and a write position of its own, and every `DECODE_CACHE_SLACK` steps its live window is
   copied back to slot 0 (`_rebase_decode_state`) on the host, between steps, outside the
   captured graph. Long layers are untouched: full extent, absolute position.

2. FUSION. One allocation per distinct extent instead of one per layer, with layer i handed
   `base[slot]`; slicing the outermost dimension of a contiguous tensor gives a contiguous
   tensor, so every kernel call sees the same buffer, strides and in-place append, and the
   cache tensor bytes are unchanged by construction.

Predicted `kv_cache_bytes`, from the charge law measured on device in launch 4 and now exact
on 9 of the 10 readings this run has taken (`knowledge/kv_cache_charge_measured.md`;
`charge(size) = 2 MiB` if `1 MiB < size < 2 MiB`, else `512*ceil(size/512)`; a residue of
1,048,576 only with eight or more allocations of exactly 2,097,152):

    codes, six short layers  [6,2,1,1057,2,128] int8    3,247,104 -> 3,247,104
    codes, two long layers   [2,2,1,2048,2,128] int8    2,097,152 -> 2,097,152
    scales, short            [6,2,1,1057,2,1]   bf16       50,736 ->    51,200
    scales, long             [2,2,1,2048,2,1]   bf16       32,768 ->    32,768
    seq + the rolling write position                            8 ->     1,024
    kv_cache_bytes                                                  5,429,248

-3,090,944 B (-36.28%) on the champion's 8,520,192, and -84.31% on the reference. Predicted
`nopref_kv_cache_bytes` 2,135,040, -3,584 on the champion (at max_len 513 no short layer's
window+1+slack fits, so every extent collapses to 513 and the no-prompt path is the
champion's; the -3,584 is eight scale tensors' 512-B rounding slack folded into one).

Note what changed about which mechanism is load-bearing, because it inverts launch 6's
finding. At `n_kv_head=4` a short layer's codes at 1057 slots were 1,082,368 B -- inside the
1-2 MiB band, charged a full 2,097,152 -- so the extent change bought almost nothing unless
the allocations were fused, and fusion was 98.7% of that launch's gain. At `n_kv_head=2` the
same tensor is 541,184 B, below 1 MiB and charged exactly, so the extent change pays on its
own: unfused this program is predicted at 5,430,272, just 1,024 B worse. The band moves when
the element size does, and so does which of the two mechanisms is doing the work. Both are
kept because both are measured and the fused form is the better of the two by 1,024 B.

Predicted `request_ms_median`: 690-712 against the 750 ceiling. The champion reads 697.836.
The rebase adds work and the smaller cache removes more: launch 6 measured this trade
directly at `n_kv_head=4`, where the identical fold took 739.615 to 721.691 -- -17.924 ms for
-5,946 slot-layers, because `kv = kvq * kvs` dequantises the whole allocated cache every step.
Halving `n_kv_head` has already halved that traffic, so the same -36.3% of slot-layers should
return roughly half as many ms, around -9, against a rebase cost that is itself halved again.
The ceiling is not the binding constraint on this lineage: launch 6 left 28.3 ms of it.

Watching, and the reason this could be refused: `decode_tv_distance_max` and
`nopref_decode_tv_distance_max`, ceiling 0.05 each. The champion reads 0.027049 and 0.030562.
The fold moved the prompted figure +0.004935 at `n_kv_head=4` (0.021370 -> 0.026305), which
would put this near 0.032; the no-prompt figure is measured on a path this program does not
change, and launch 6 measured that this metric scatters about 0.023 under int8 on
semantically identical no-prompt paths, so it is the one number here that cannot be predicted
from a single prior reading.

Nothing outside the decode protocol changes: `forward`, the model, the optimizer, the data
budget and the training loop are the champion's, so `val_bpb`, `flops_per_token_measured`,
`num_params_total`, `peak_vram_bytes` and `training_data_tokens_available` are not on this
axis, and `prepare.py`, `pyproject.toml`, `uv.lock` and all three frozen regions are
byte-identical to `champion/`.

struct_shortwin512_v1 -- champion v7 (`ext_unfuse_short_v1`, 5,430,272) with the trained
short-attention window halved, and nothing else. `SHORT_WINDOW = 512` is now a top-level
constant and `_compute_window_sizes` reads it where it used to compute `long_window // 2`.
Two lines, unconditional, no `EXPERIMENT_ID` gate.

`self.window_sizes` feeds both `forward()` (line 436) and both decode calls, so training and
serving read the same window: this is NOT the `ext_decwin_long1024_v1` family, whose launch 12
shortened the decode window alone and measured `decode_tv_distance_max` 0.366628 because
`forward()` still read the long keys. That launch is this candidate's control, not its
precedent.

Since champion v6 the decode extent is `left + 1 + DECODE_CACHE_SLACK` per layer, so the
window WIDTH converts directly into allocated slots: the six short layers go from 1057 to 545
slots and the two long layers are untouched. All six rolling extents stay equal, so the
uniform-extent guard in `init_decode_state` passes; `win_hist` becomes 512 and the rebase
period is `extent - hist` = 33 steps, exactly the champion's, because the period is
`DECODE_CACHE_SLACK + 1`.

Predicted, under the charge law that has reproduced launches 7, 8, 10 and 11 to the byte
(`charge(n) = 512*ceil(n/512)` at or below 1 MiB, which every allocation here is):

    codes, six short layers  [2,1,545,2,128] int8  x6    6 x 279,040 = 1,674,240
    codes, two long layers   [2,1,2048,2,128] int8 x2    2 x 1,048,576 = 2,097,152
    scales, six short        [2,1,545,2,1] bf16   x6     6 x 4,608   =    27,648
    scales, two long         [2,1,2048,2,1] bf16  x2     2 x 16,384  =    32,768
    seq + the rolling write position                                 =     1,024
    kv_cache_bytes                                                      3,832,832

    -1,597,440 B, -29.42% on champion v7, -88.92% on the reference.

    nopref_kv_cache_bytes = 8 x (262,656 + 4,608) + 512 = 2,138,624, UNCHANGED: at max_len
    513 the short need of 545 is not < 513, so every extent collapses to 513 and the
    no-prompt path is byte-for-byte the champion's. The tiebreak does not move.

What it spends: quality-gate headroom only. `forward()` changes, so `val_bpb` moves, and this
is the first candidate in the run to move the trained window. Champion v7 reads 1.046802 with
0.003198 of margin. The registered prediction is 1.049057, from the run's only measured
bytes-per-`val_bpb` rate (the kv_dim rung step-corrected to a common update count: 2,711,552 B
for 0.0038273, i.e. 708.5 MB per unit), which transfers a rate measured on cached WIDTH to
cached REACH and is an ASSUMPTION, labelled as one. It leaves 0.000943 of margin -- 8.0
sigma_cond (0.000117 residual sd) or 2.3 sigma_uncond (0.000416). This run has already seen a
two-point val_bpb extrapolation miss by 0.0020 on the `WARMDOWN_RATIO` axis, so the byte
figures above are an identity and the val_bpb figure is a fit.

Every other ceiling moves the safe way or not at all: `flops_per_token_measured` falls (the
attention term is proportional to the effective context), `peak_vram_bytes` falls, the rebase
copies half as many keys so `request_ms_median` falls from the champion's clearance of 82.5 ms,
`num_params_total` and `training_data_tokens_available` are untouched, and both TV figures
should stay near champion v7's because training and decode read the same window.
`prepare.py`, `pyproject.toml`, `uv.lock`, the whole decode protocol and all three frozen
regions are byte-identical to `champion/`.

struct_shortwin512_mqa_v1 -- champion v9 (`struct_shortwin512_v1`, 3,832,832) with
`KV_HEAD_RATIO` 2 -> 4, i.e. `n_kv_head` 2 -> 1, MQA. ONE constant. Both halves of this
composition are already measured on this run's own board and neither is speculative:

- the 545-slot extent and the 512-key trained window are champion v9's, measured at launch 17
  (`kv_cache_bytes` 3,832,832 and `val_bpb` 1.0361385, both exactly as recorded);
- kv_dim 256 -> 128 is launch 7's `struct_kv_heads_mqa_v1`, which measured `kv_cache_bytes`
  4,260,352 on its own host and `val_bpb` 1.0504027 at 829 updates, against launch 8's
  1.0469403 at 801. Step-corrected at -1.3022e-05 per update that rung cost **+0.0038273**, and
  it was refused by 0.00040 against the < 1.05 gate.

The reason this is now claimable is entirely launch 17: halving the trained window bought
0.0098 of `val_bpb`, so the margin is 0.0139 where launch 7 had 0.0036. The refusal that
blocked kv_dim 128 for four launches was a property of a program that no longer exists.

Predicted `kv_cache_bytes`, an IDENTITY under the charge law that has now closed on six
readings (`charge(n) = 512*ceil(n/512)` at or below 1 MiB, which every allocation here is):

    codes,  six short layers  [2,1,545,1,128] int8   6 x 139,520 -> 6 x 139,776 =   838,656
    codes,  two long layers   [2,1,2048,1,128] int8  2 x 524,288 -> exact       = 1,048,576
    scales, six short         [2,1,545,1,1] bf16     6 x   2,180 -> 6 x   2,560 =    15,360
    scales, two long          [2,1,2048,1,1] bf16    2 x   8,192 -> exact       =    16,384
    seq + the rolling write position                                            =     1,024
    kv_cache_bytes                                                                1,920,000

    -1,912,832 B, -49.91% on champion v9, and -94.45% on the reference.

    nopref_kv_cache_bytes = 8 x (131,584 + 2,560) + 512 = 1,073,664 -- and that is not a
    derivation, it is the figure launches 7 and 11 BOTH measured at kv_dim 128, because at
    max_len 513 every extent collapses to 513 and the no-prompt path does not see the window.
    The tiebreak improves by 1,064,960.

Predicted `val_bpb` 1.0400 (band 1.0380-1.0420), which is a PREDICTION and additive by
assumption: champion v9's measured 1.0361385 plus launch 7's step-corrected +0.0038273 for the
same rung. It leaves ~0.0100 of margin, 24x the unconditional launch sd of 0.0004159. The
assumption being spent is additivity of two quality deltas measured on different hosts; note
that this run has already seen non-additivity in the other direction (a rate transferred across
axes missed by 0.0129 at launch 17), so the band is wide on purpose. Even the whole of launch
7's cost landing twice over would clear the gate.

Also predicted: `num_params_total` 34,603,152 (launch 7's exact figure -- MQA removes 5,243,008
parameters, half of `c_k`/`c_v` and the four value-embedding tables, and the window does not
enter the count); `flops_per_token_measured` 201,327,360 (champion v9's 207,619,584 minus the
6,292,224 that the same rung removed between launches 8 and 7 at fixed window);
`request_ms_median` below champion v9's 627.621, because `kv = kvq * kvs` dequantises the whole
allocated cache every step and this halves it again.

The number to watch is `decode_tv_distance_max`, ceiling 0.05. Champion v9 reads 0.0358163, its
highest value on this lineage, and the same MQA rung moved the prompted figure +0.0053 at fixed
extent between launches 8 and 11 (0.0241 -> 0.0294). That predicts ~0.041, with 0.009 of ceiling
left, and it is the one figure here that could refuse this candidate. `nopref` is safer: 0.0229
on v9 and 0.0250 at launch 11's kv_dim 128.

Nothing else moves. `WINDOW_PATTERN`, `SHORT_WINDOW`, `DEPTH`, `HEAD_DIM`, `DECODE_CACHE_SLACK`,
the optimizer, every learning rate, `WARMDOWN_RATIO`, the data budget, the allocation shape
(one per layer) and all three frozen regions are champion v9's, and `prepare.py`,
`pyproject.toml` and `uv.lock` are byte-identical to `champion/`.

prec_slack4_recompose_v1 -- `DECODE_CACHE_SLACK` 32 -> 4 on champion v10. ONE constant, and the
whole of the diff. analyst2's item (post `e4f94314`), claimed by gpu4.

**This is not a new mechanism. It is the re-application of a measured KEEP that a lane race
dropped.** `prec_slack4_v1` (launch 16) measured this exact edit at the 1024 window and was the
champion at 5,344,256. Champions v9 and v10 were both built on v7, which predates it, so
`champion/train.py` came back to `DECODE_CACHE_SLACK = 32`. Nothing refuted slack 4; the byte
saving was simply left on the floor when the lineage moved through a different lane. Neither
`SHORT_WINDOW 512` nor `KV_HEAD_RATIO 4` interacts with it: `init_decode_state` computes
`need = left + 1 + slack` from whatever the window is, and `_rebase_decode_state` fires on
`win_pos + headroom > win_extent`, so the schedule follows the constant.

Predicted, an IDENTITY, from this file's own `init_decode_state` under the charge law
(`charge(n) = 512*ceil(n/512)`, every allocation at or below 1 MiB), verified on CPU by first
reproducing champion v10's measured 1,920,000 and 1,073,664 exactly
(`checks/slack4_recompose_check.py`):

    codes,  six short layers [2,1,517,1,128] i8   132,352 ->   132,608  (x6)    795,648
    codes,  two long layers  [2,1,2048,1,128] i8  524,288 ->   524,288  (x2)  1,048,576
    scales, six short layers [2,1,517,1,1] bf16     2,068 ->     2,560  (x6)     15,360
    scales, two long layers  [2,1,2048,1,1] bf16    8,192 ->     8,192  (x2)     16,384
    seq + the rolling write position                    8 ->     1,024            1,024
    kv_cache_bytes                                                            1,876,992

**-43,008 B (-2.24%)** on champion v10's measured 1,920,000, and **-94.58%** on the reference.
Per short layer the codes go 139,776 -> 132,608 (-7,168) and the rounded scale tensor does not
move; at `n_kv_head 2` the same edit is worth twice that, which is why launch 16 read -86,016.
`nopref_kv_cache_bytes` is **UNCHANGED at 1,073,664**: at `max_len` 513 neither 545 nor 517 fits
under `need < max_len`, so every extent collapses to 513 either way and the no-prompt program is
byte-for-byte champion v10's. A candidate that cannot move the tiebreak should say so, because
the tiebreak is the only thing separating two equal targets.

`val_bpb`, `flops_per_token_measured`, `num_params_total` and `peak_vram_bytes` are champion
v10's by construction: `DECODE_CACHE_SLACK` is read only inside `init_decode_state` and
`_rebase_decode_state`, and `forward()`, the model, the optimizer, the data budget and the
training loop are untouched. This is the rare candidate whose exposure to the quality gate is
zero rather than small -- which matters, because champion v10 clears the gate by only 0.0008666.
What it does inherit is that reading's own re-draw: the training program is quality-identical, so
`val_bpb` is re-measured, and ~92% of this run's 0.0004159 unconditional spread is node
throughput against the fixed `TIME_BUDGET`. 0.00087 is 2.1 sigma of that. Stated, not hidden.

**The number at risk is `decode_tv_distance_max`, and the margin is thin.** Champion v10 reads
0.0427447 against the 0.05 ceiling: **0.0072553 left**, the least this lineage has ever had, and
it has risen at every step (0.0214 -> 0.0358 -> 0.0427). No cached value changes here and no
slot's CONTENTS change, so element fidelity has no mechanism to move -- but the slot OFFSETS
change, and launch 16 is this run's only clean reading of what that costs: on a value-preserving
slack change it re-drew the scored figure **+0.004481** and the no-prompt figure +0.006913. So
the one measurement of this exact mechanism predicts ~0.0472 with ~0.0028 to spare, and the
available margin is 1.6x the observed re-draw. That is the bet, priced from the run's own
measurement rather than assumed, and it is a bet either way: if a value-preserving layout change
re-draws past 0.05 from 0.0427, then the FIDELITY ceiling and not the byte model is what now
bounds this whole family, which re-prices `struct_shortwin256_v1` (another window halving on the
same 0.0073, where launch 17's halving moved the figure +0.0144) before a launch is spent on it.

`request_ms_median` predicted 618-622 against the 750 ceiling: the rebase goes from ~16 to ~102
firings over a 512-step request at launch 16's measured 0.116 ms each, so about +10 ms on
champion v10's 610.426. Launch 18 measured 602.042 with an IQR of 0.180, so 147.96 ms of that
ceiling is free and this is not a real risk.

Free prints, none of which can perturb an official number (they run after
`report_efficiency_metrics` and emit no metrics line of their own): the `[charge]` rows are from
the inherited 1057 -- two windows out of date -- onto this candidate's real 517, the 545 it
supersedes, the slack-3 shape at 516, and the fused short-scale group that
`prec_smallalloc_tidy_v1` proposes, so all four are priced on ONE launch's allocator.

================================================================================
struct_uniwin_v1 -- champion v11 with the SECOND window literal cut: LONG_WINDOW 512
================================================================================

Two lines, unconditional, no `EXPERIMENT_ID` gate, wired exactly the way `SHORT_WINDOW` already
is: a module constant `LONG_WINDOW = 512`, and `_compute_window_sizes` reads it where it read
`config.sequence_len`. `config.sequence_len` keeps 2048 everywhere else -- `max_len`, the rotary
table, the training sequence length are all untouched. All eight windows become 512, so layers 3
and 7 stop being globally attending and `window_sizes[-1] = (long_window, 0)` at line 484 becomes
a no-op instead of a forced long layer.

**Why the windows must be EQUAL and not merely both smaller.** `init_decode_state` gives a layer
`left + 1 + DECODE_CACHE_SLACK` slots when that is below `max_len`, and then (lines 628-634) if
the ROLLING layers do not all share one extent it throws the whole thing away and allocates
`max_len` everywhere, because two rolling extents would need two rebasing counters. `LONG_WINDOW`
1024 against `SHORT_WINDOW` 512 therefore measures 8,520,192 -- a +354% regression on champion
v11, not a cut. Uniform, or nothing.

`kv_cache_bytes` 1,082,368, an IDENTITY. All eight extents are 512+1+4 = 517, uniform, guard
satisfied:

    codes     8 * charge(2*517*128 = 132,352) = 8 * 132,608 = 1,060,864
    scales    8 * charge(4*517     =   2,068) = 8 *   2,560 =    20,480
    counters  seq + seq_win                                 =     1,024
                                                              ---------
                                                              1,082,368   -794,624, -42.33%

That is -96.87% on the reference's 34,603,520. The closed form is
`[local verification path]`, which reproduces **six launches and
twelve readings with zero error** before pricing this one -- launches 7, 10, 16, 17, 19 and 20, on
`kv_cache_bytes` and `nopref_kv_cache_bytes` both. It priced champion v11 at 1,876,992 before
launch 20 landed, which is also the correction to the 1,893,376 circulating in the workshop (the
two long layers' scale tensors are 8,192 B each, not 16,384).

`nopref_kv_cache_bytes` is UNCHANGED at 1,073,664: at `max_len` 513 no layer's 517 fits, so every
extent collapses to 513 and the no-prompt program is byte-for-byte champion v11's. **This
candidate does not buy the tiebreak** -- said plainly rather than left for a reader to assume.
`flops_per_token_measured` 182,452,992: the window sum falls 7,168 -> 4,096 and `estimate_flops`
charges `12*h*q` per key with `h*q = 512`, which is the identity launch 17 confirmed to the FLOP.
`num_params_total` unchanged at 34,603,152 -- a window is not a parameter.

**The two fitted figures, and which one actually decides this launch.**

`val_bpb`: predicted 1.0434, band 1.0400-1.0520, against a gate of 1.05 and champion v11's
1.0467764. The basis is same-axis and same-host: launch 17 cut six layers' reach 1024 -> 512 and
val_bpb FELL 0.0098-0.0103 while `flops_per_token` fell 8.33%, i.e. this substrate was
over-attending, not capacity-starved. This candidate is four layer-rungs (layers 3 and 7, each
2048 -> 512) where that was six. **The risk here is the SIGN, not the sd**: 0.0034 of predicted
margin is 6.2 sigma at this team's 0.0005474, but no launch in this run has measured a model with
NO globally attending layer, and line 484 exists to guarantee one. If reach on the last two layers
is load-bearing where it was not on the other six, this is refused, and that refusal is the
finding -- it prices `WINDOW_PATTERN "SSSS"` (the half dose, 1,479,680) without spending a launch
on it.

`decode_tv_distance_max`: predicted 0.031, band 0.021-0.045, ceiling 0.05. This is no longer the
binding figure and the reason is launch 20, which measured it FALLING 0.042745 -> 0.021543 on a
decode-only constant that touches no trained weight and no cached width. So the metric is not a
budget that accumulates down a lineage, and the ~0.0024-per-layer-rung rate I had fitted from
launches 10 and 17 is withdrawn as confounded. The band above is the champion's reading plus the
largest single move this metric has made in the whole lineage; even its top clears the ceiling.

Not folded in: `DECODE_CACHE_SLACK` 4 is inherited from champion v11 rather than stacked, and
nothing else is added. The `[residue]` and `[charge]` probes ride along unchanged; `[residue]`
will print the eight 517-slot extents, which is the on-device confirmation of the identity above.

================================================================================
struct_uniwin256_v1 -- champion v12 with BOTH window literals at 256
================================================================================

`SHORT_WINDOW = 256` and `LONG_WINDOW = 256`. Two constants, unconditional, no `EXPERIMENT_ID` gate.
**They have to move together.** `init_decode_state` (lines 628-634) throws away every rolling extent
and allocates `max_len` on all eight layers unless the rolling extents are identical, so
`SHORT_WINDOW` 256 against `LONG_WINDOW` 512 measures **4,260,352** -- a 3.9x regression that looks
like a composition of launches 21 and 22. gpu6 derived that collapse in workshop `[SUGGESTION]`
227f7428 and I verified it independently against the same lines. Uniform, or nothing.

`kv_cache_bytes` 549,888, an IDENTITY: all eight windows 256, so all eight extents are 256+1+4 = 261.

    codes     8 * charge(2*261*128 = 66,816) = 8 * 67,072 = 536,576   97.6%
    scales    8 * charge(4*261     =  1,044) = 8 *  1,536 =  12,288    2.2%
    counters  seq + seq_win                                =   1,024    0.2%
                                                             -------
                                                             549,888   -532,480, -49.20%

That is **-98.41% on the reference's 34,603,520**. The closed form is
`[local verification path]`, which reproduces **seven launches and
fourteen readings with zero error** before pricing this one (7, 10, 16, 17, 19, 20, 21 and 22, on
`kv_cache_bytes` and `nopref_kv_cache_bytes` both).

**`nopref_kv_cache_bytes` 549,888 -- the tiebreak moves for the first time in this run.** At `max_len`
513 the extent 261 *fits* (261 < 513), so the no-prompt program rolls too instead of collapsing to
513-slot buffers. Every window candidate measured so far left the declared tiebreak at exactly
1,073,664 or 2,138,624; this one takes it from 1,073,664 to 549,888.

`flops_per_token_measured` 169,870,080: the window sum falls 4,096 -> 2,048, and `estimate_flops`
charges `12*h*q` per key with `h*q = 512`. `num_params_total` unchanged at 34,603,152.

**The gate, and why this is the first comfortable gate position in several launches.** Two same-axis
measurements taken fifteen minutes apart, both at exactly 900 updates:

    launch 20 (v11)  6x512 + 2x2048   870 updates   val_bpb 1.0467764
    launch 21 (v12)  8x512            900 updates   val_bpb 1.0478327   two global layers 2048->512: +0.00145
    launch 22        6x256 + 2x2048   900 updates   val_bpb 1.0354081   six local layers 512->256: -0.0110

Reach keeps PAYING on the local layers -- 0.0103 at rung one, 0.0110 at rung two -- and cost only
+0.00145 on the two global ones. This candidate is launch 22 plus those two global layers taken to
256. Predicted `val_bpb` **1.0350**, band 1.0330-1.0420; even the top of that band clears the gate by
0.008, which is 3.4x the largest same-program redraw this run has measured (0.0023570, launches 19 and
20, whose training programs are identical). Labelled a *fit*, but a fit built on two same-axis,
same-host, same-update-count readings rather than a transferred rate.

`decode_tv_distance_max` predicted 0.026, band 0.020-0.035, ceiling 0.05. Banded rather than modelled:
my earlier layer-rung rate stays withdrawn after launch 20 refuted it, and the three most recent
readings are 0.021543, 0.024289 and 0.021285 -- the last of which is this same 256-key reach on six
layers.

Nothing else moves.

================================================================================
struct_uniwin256_batch18_v1 -- launch 23's refused byte cut, re-hosted on more updates
================================================================================

`TOTAL_BATCH_SIZE` 2**19 -> 2**18. ONE constant, unconditional, and the whole of the diff against
launch 23's frozen source. Built by gpu6 on `struct_uniwin256_v1` (launch 23, gpu5), which measured
**549,888 B, -49.2% on champion v12 and -98.41% on the reference** -- the largest byte cut left on
this board -- and was refused on the quality gate alone at `val_bpb` 1.0572295, missing by 0.0072295.
Every other constraint it measured is clear: `decode_tv_distance_max` 0.0322447,
`nopref_decode_tv_distance_max` 0.0308352, `request_ms_median` 538.590 of 750,
`flops_per_token_measured` 169,870,080 of 239,078,400, `num_params_total` 34,603,152,
`peak_vram_bytes` 41,384,621,056 of 47,198,976,512.

This is the same move launch 19 made: take a byte cut that only the gate refused, and re-host it on
something that pays the gate back. It is not a new mechanism and it does not touch the decode
protocol, the model, the window literals or `forward()`.

**The byte figures are not predictions here, they are carried over as MEASURED.** `TOTAL_BATCH_SIZE`
is read in exactly eight places below the docstring -- the assert, `grad_accum_steps`, `tok_per_sec`,
`mfu`, `tokens_consumed`, the harness tick and the reporter -- and not one of them is inside
`init_decode_state`, `_compute_window_sizes` or `forward()`. `check8.py` derives the geometry off this
source's own bound methods and confirms all eight extents stay 261 with headroom 5:

    kv_cache_bytes            549,888   IDENTITY, and measured on launch 23
    nopref_kv_cache_bytes     549,888   IDENTITY, and measured on launch 23 (at 261 < 513 the
                                        no-prompt request rolls too, so the two shapes coincide)
    flops_per_token_measured 169,870,080  unchanged: per-TOKEN counted work does not depend on
                                        how many tokens make an optimizer step
    num_params_total       34,603,152   unchanged
    peak_vram_bytes    41,384,621,056   `DEVICE_BATCH_SIZE` stays 128, so the microbatch and the
                                        frozen 128x2048 validation forward are untouched; only
                                        `grad_accum_steps` moves, 2 -> 1

`assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0` holds exactly at 2**18 = 262,144 = 128*2048, so
`grad_accum_steps` becomes 1. 2**17 would fail that assert and is not available.

WHAT IS ON TRIAL, and it is a FIT: whether roughly twice as many optimizer updates over roughly the
same number of tokens buys back the 0.0072295 the gate refused. This run has never measured it.
`WARMDOWN_RATIO` is the only byte-free quality knob anyone has run here and it moved 0.0010-0.0015 in
both directions (launches 11 and 13), which is 5x too small; `TOTAL_BATCH_SIZE`, `MATRIX_LR` and MLP
width have never been run. analyst1's `ext_mqa_batch18_v1` row holds this mechanism and analyst3's
framing on it is the reason the null result is worth buying: **this is the only experiment available
that separates TOKENS from UPDATES**, because every step-count correlation recorded in this run so far
came with proportionally more tokens.

Predicted, and labelled: `dt` should fall from launch 23's 645 ms to roughly 330-390 ms, because the
per-step compute halves while the fixed per-step cost (the optimizer, Muon's iterations, the host
work) does not, giving about **1,550-1,850 updates against launch 23's 942** at roughly unchanged
total tokens. `val_bpb` central 1.049, band 1.042-1.062 -- **the band straddles the gate and I am
claiming it as a straddle, not as a margin.** The model is measurably undertrained on this substrate:
launch 22's loss was still falling 0.0887 over its last fifth with `lrm` at 0.00. Against that, the
LRs in this file were tuned at 2**19 and halving the batch at a fixed LR is not a neutral change, so
the sign is genuinely open.

I am also correcting a claim I made in launch 22's own record, because launch 23 refuted it before
this candidate was built. I wrote that counted FLOPs had stopped buying steps, from 192M -> 182M
moving `dt` by 0.1% and updates by 0. Launch 23 ran at 169,870,080 and got **942** updates at 645 ms.
So `dt` is not flat below 192M, it is lumpy in it: 201M/698ms, 192M/675ms, 182M/676ms, 170M/645ms. The
correct statement is the weaker one -- counted FLOPs are a poor predictor of `dt` on this substrate --
and my "price the next rung at zero extra updates" advice was wrong. Recorded rather than quietly
dropped.

The second thing that can refuse this candidate is fidelity, and I register no point prediction for
it, per launch 20. Launch 23 read `decode_tv_distance_max` 0.0322447, the highest this lineage has
held, leaving 0.0178 of ceiling. The decode layout here is byte-identical to launch 23's -- same
extents, same slots, same rebase cadence -- but the WEIGHTS differ, because this program trains
differently, and this metric is a max over 513 positions of a quantisation difference, so it re-draws
with the values. 0.0178 of room against a metric this run has watched move 0.021 is the real risk and
it is not the one I am buying deliberately.

================================================================================
struct_share4_uniwin_v1 -- champion v13 with FOUR cache owners instead of eight
================================================================================
`KV_GROUP = 2`: layers are grouped in adjacent pairs and the FIRST layer of each pair owns that
pair's K/V. The owner computes `c_k`/`c_v`, mixes in its value embedding, applies rotary and the
norm, and writes the pair's one int8 cache. The reader computes only its own queries and attends
over its owner's K/V with its OWN window. The query side, the residual width, `head_dim`,
`n_kv_head`, `KV_HEAD_RATIO`, both window literals, `DEPTH`, `DECODE_CACHE_SLACK`,
`TOTAL_BATCH_SIZE`, the optimizer, every learning rate, the data budget and all three frozen
regions are champion v13's, and `prepare.py`, `pyproject.toml` and `uv.lock` are byte-identical
to `champion/`.

**This mechanism is not new and is not being re-derived.** It is launch 9's
`struct_cla_pairs_v1`, whose owner map, VE relocation and decode contract are transplanted in
semantics onto the current champion. Launch 9 measured `kv_cache_bytes` 4,260,352 and
`nopref_kv_cache_bytes` 1,069,568 -- both exactly what the four-owner charge model predicts --
and `val_bpb` 1.0549410 against its host's 1.0475767, so the mechanism's measured quality cost on
this run's own board is **+0.0073643 raw** (the team's step-corrected figure for the same pair is
+0.0081390). What is new here is the HOST, and that is the whole of the experiment.

**Why the host is what makes it claimable.** Launch 9 was refused with 0.0032 of gate margin on a
program with kv_dim 256, one cache per layer and 2048-key long layers. Champion v13 clears the
gate by **0.0231506**, because launch 24 measured `TOTAL_BATCH_SIZE` 2**19 -> 2**18 buying 0.0304
of `val_bpb` at unchanged tokens. Predicted `val_bpb` 1.0350, band 1.0310-1.0430, leaving ~0.0150
of margin = 6.4x the largest same-training-program spread this run has measured (0.0023570,
launches 19 and 20). That is a FIT, and the assumption it spends is additivity of a quality delta
across hosts, which this run has caught being wrong three times (window vs MQA 3.5x, a lever's
price depending on layers it does not touch 15x, and a non-monotone fidelity metric). The band is
wide on purpose and the sign is not in doubt -- sharing removes capacity -- only the magnitude.

**Predicted `kv_cache_bytes` 275,456, an IDENTITY, and this one is arithmetic on measured
charges rather than a law.** Champion v13's own `[charge]` probe (launch 24 stdout) measured the
extent-261 code tensor charged 67,072 B and its scale tensor charged 1,536 B, i.e. 68,608 B per
cache, and 8 x 68,608 + 1,024 = 549,888 is exactly what launch 24 scored. This candidate holds
four caches of the same shape, so 4 x 68,608 + 1,024 = **275,456**: -274,432 B, **-49.91% on
champion v13** and **-99.20% on the reference**. `nopref_kv_cache_bytes` is the same 275,456 and
the tiebreak moves with the target, because at `max_len` 513 the need of 261 still fits, so the
no-prompt shape rolls at the same extents (this is champion v13's own geometry, measured).
`workspace/checks/share4_check.py` reproduces **eleven launches and twenty-two byte readings with
zero error** -- including launch 9's four-owner pair -- before pricing this one.
`num_params_total` 34,078,864: four layers lose `c_k` and `c_v`, 4 x 2 x 512 x 128 = 524,288, and
the number and width of the value-embedding tables are unchanged (four tables of kv_dim 128 move
from {1,3,5,7} to {0,2,4,6}). `flops_per_token_measured` 166,724,352 = 169,870,080 - 6 x 524,288
on the 6x-matmul-parameter identity this run has confirmed twice; the window sum does not move.

**Proved on CPU before the launch, in `workspace/checks/share4_cpu_check.py`.** It execs this
file's model-definition region against a CPU stand-in for FA3 and compares `forward` to
`prefill + decode_step` at every position, on RANDOMISED weights -- necessary because
`init_weights` zeroes every `c_proj`, so with the shipped initialisation attention cannot reach
the logits and any decode-vs-forward check reads 0 for a completely wrong cache. At `KV_GROUP = 1`
this file reproduces the champion path exactly (eight distinct allocations, VE on {1,3,5,7}, max
TV 0.005563 -- the substrate's int8 floor at these weights). At `KV_GROUP = 2` it reads four
distinct allocations for eight layer entries and max TV 0.006362, 1.14x the champion's on the
same seed. Two deliberate mutants confirm the check has power: dropping the reader's `+1`
`cache_seqlens`, and rebasing per layer over the alias list, each read TV ~0.30, 47x the
candidate and 6x the ceiling.

**The two places the transplant is NOT a copy, both forced by champion v13's rolling cache,
which launch 9's host did not have.** (1) `init_decode_state` allocates per OWNER at that group's
extent and asserts a group's layers agree on it; the per-layer `kvq`/`kvs` lists are aliases, so
nothing else in the decode path needs to know about ownership. (2) `_rebase_decode_state`
iterates the OWNER buffers, because copying a live window back to slot 0 is not idempotent -- a
second copy reads the slots the first one overwrote -- so rebasing once per reader would corrupt
every shared cache. That is mutant B above.

**What can refuse this, in the order the run's evidence ranks them.** First
`nopref_decode_tv_distance_max`: champion v13 reads 0.0393705 of 0.05, the highest figure this
run has taken and the tightest constraint on the board, and this candidate genuinely changes WHAT
IS CACHED rather than only where it sits. I register no point prediction, per launch 20. The one
measurement of this mechanism moved both TV figures DOWN (launch 9 read 0.0233865 no-prompt and
0.0234888 prompted against its host's 0.0305619 and 0.0270493), and the two nearest same-geometry
readings on this lineage span 0.0308 to 0.0394, so the honest statement is that ~0.0106 of ceiling
against a ~0.009 same-family spread is a coin flip that this candidate shares with every other
item on the board. Second `val_bpb`, above. Nothing else is at risk: `flops_per_token_measured`,
`num_params_total` and `peak_vram_bytes` all fall, `request_ms_median` should fall because
`kv = kvq * kvs` dequantises half as many caches per step, and
`training_data_tokens_available` is untouched.

**What it teaches if it fails.** A gate refusal prices cross-layer sharing at maximum available
margin and closes team structure's central hypothesis -- that the COUNT of cached tensors is the
lever -- with a fifth refutation at the only host where it could ever have been afforded. A TV
refusal would be the first in this run and would establish `nopref_decode_tv_distance_max` as the
binding ceiling, repricing every remaining item across all three teams. Either way the four-owner
byte identity is confirmed or refuted on a program nobody has run.

================================================================================
prec_fuseall_sub1mib_v1 -- re-fuse every cache allocation, now that all of them are below 1 MiB
================================================================================

`init_decode_state` only. Nothing else in this file changes: `forward()`, `KV_GROUP`, the window
literals, the extents, the rebase cadence, the quantisation and every hyperparameter are champion
v14's. Parent champion: champion.md **version 14**, `struct_share4_uniwin_v1`, 275,456 (Step 2c).
This is analyst2's queued row `prec_fuseall_sub1mib_v1` (post 0256da1d), which priced the same
mechanism at 543,744 on champion v13. I derived that same 543,744 independently, to the byte, before
reading the row; what changed is only the host. Built on v13, REBASED onto v14 when gpu5's KEEP
published while it was being priced, and adapted in one place the v13 row did not need: allocate per
distinct OWNER extent rather than per layer, keeping v14's group-extent assert. All figures below are
v14's; v13's -6,144 B at eight owners is recorded in the result file rather than dropped.

The four owner code tensors become slices of one `[4, 2, 1, 261, 1, 128]` int8 allocation, the four
scale tensors slices of one `[4, 2, 1, 261, 1, 1]` bfloat16 allocation, and the two int32 write
positions slices of one `[2, 1]` int32 allocation: **six allocations become three.** This is
`ext_unfuse_short_v1` (launch 10, a KEEP) run backwards, and it is a KEEP in both directions because
the shapes moved underneath it. Fusing is a property of where an allocation sits relative to 1 MiB,
not of the mechanism: at `n_kv_head=2` the fused group was 3,247,104 B, above 1 MiB with a
non-integral MiB remainder, and the launch-10 probe measured it charged 4,194,304, so unfusing paid
946,176 B. Since then `KV_HEAD_RATIO` 4 (launch 17), the windows 1024 -> 512 -> 256 (launches 21, 23)
and four-owner sharing (launch 26) have cut the per-owner tensor 126x, and the whole fused group is
now 267,264 B -- below 1 MiB, where both charge regimes this run has observed agree.

    kv_cache_bytes           272,384   IDENTITY, -3,072 B (-1.115%) on champion v14
    nopref_kv_cache_bytes    272,384   IDENTITY, -3,072 B (the two shapes coincide at 261 < 513)
    num_params_total      34,078,864   unchanged: no parameter is touched
    flops_per_token_measured 166,724,352  unchanged: `estimate_flops` reads neither allocation
                                        shapes nor `init_decode_state`
    peak_vram_bytes   ~40,448,223,232   unchanged to within its own noise; the cache is 7 ppm of it

The byte arithmetic, on charges MEASURED on this hardware for these exact shapes by launch 24's and
launch 25's probes (stdout 0024 lines 84-95), not on a rule applied to a new shape:

    v14 pays   4 x 67,072 codes + 4 x 1,536 scales + 2 x 512 counters = 275,456   <- scored
    this pays  267,264 codes  +  4,608 scales  +  512 counter         = 272,384
    the whole difference is pure 512-rounding waste: 4 x 256 on the codes (1,024), 4 x 492 on the
    scales (1,968) and one 512 B block holding 4 B, less 432 B the fused scales still round up.

267,264 is an exact multiple of 512 and 4,176 rounds to 4,608; launch 25's probe measured a fused
scale tensor of the same kind at 6,264 -> 6,656, the same rule. v14's residue over exact tensor bytes
plus one counter block is 275,456 - 271,440 - 512 = 3,504, which is the 3,072 B removed here plus the
432 B kept; this candidate's `[residue]` line should read **432**.

WHAT IS ON TRIAL, and it is the reason to spend a launch rather than fold this in later: **nothing
the model computes changes at all.** A slice of the outermost dimension of a contiguous tensor is
contiguous, so every kernel receives the shapes, strides and in-place appends it received before; the
extents, the slot each token is written to, the rebase schedule and the cached values are
bit-identical, and the training path never calls `init_decode_state`. A CPU check asserts exactly
that: prefill plus 25 decode steps on randomised weights, champion against candidate, **bit-identical
logits at every scored position**, with two mutants to show the check can fail.

So this launch is the run's first **same-program replicate** of `val_bpb`,
`decode_tv_distance_max` and `nopref_decode_tv_distance_max` at byte-identical cache geometry AND
byte-identical training arithmetic -- the band gpu1's SUGGESTION `b66b4096` says is unmeasured and
cannot be bought because `task.json` pins `launch.seed: 42`. It can be bought: not as a second seed,
but as a semantically null change that is independently worth 3,072 B. `val_bpb` has one such pair
(launches 19/20, 0.0023570 apart); the two TV metrics have none, and v14 holds 0.0163138 of `nopref`
headroom while launches 23 and 24 differ by 0.008535 on it at byte-identical geometry. Whatever this
launch reads IS that band, measured. Registered as a FIT with no point prediction, per launch 20.

Not folded in: `DECODE_CACHE_SLACK` 4 -> 3 (queued as `struct_slack3_v1`) would take the fused codes
to 266,240, a further 1,024 B, and it moves slot offsets, so it is a different lever and a different
launch. The last 512 B -- one allocation for all three groups via a byte-offset view -- is priced and
declined: dtype punning across an int8 buffer for 0.19% of the target is not worth the failure mode.
================================================================================
struct_share2_uniwin_v1 -- champion v14 with TWO cache owners instead of four
================================================================================
`KV_GROUP` 2 -> 4 on champion v15. ONE module constant, unconditional, no `EXPERIMENT_ID` gate. Layers are
grouped in adjacent FOURS and the first layer of each group owns that group's K/V: owners
{0, 4} compute `c_k`/`c_v`, mix in the value embedding, apply rotary and the norm and write
the group's one int8 cache; readers {1,2,3} and {5,6,7} compute only their own queries and
attend over their owner's K/V with their own window. `HEAD_DIM`, `KV_HEAD_RATIO`, `n_embd`,
the residual width, both window literals, `DEPTH`, `DECODE_CACHE_SLACK`, `TOTAL_BATCH_SIZE`,
gpu4's launch-28 fused allocation, the optimizer, every learning rate, the data budget and all
three frozen regions are champion v15's, and `prepare.py`, `pyproject.toml` and `uv.lock` are
byte-identical to `champion/`. An AST body diff against `champion/train.py` shows the only
executable difference is `KV_GROUP`; the docstring is the rest of the diff.

**The mechanism is champion v14's own, moved one notch.** Nothing is re-derived: the owner
map, the VE relocation, the per-owner allocation, the alias list and the per-owner rebase are
already general in `KV_GROUP`, so this is the second rung of the axis whose first rung was
measured twice (launch 9 `struct_cla_pairs_v1` and launch 26 `struct_share4_uniwin_v1`).

**PREDICTED `kv_cache_bytes` 136,704, an IDENTITY on champion v15's FUSED geometry.**
Launch 28 fused the caches into one allocation per distinct owner extent, so the charge is now
r512(n_owners x 2 x 261 x 128) + r512(n_owners x 4 x 261) + 512 for the one counter pair. At
four owners that is 267,264 + 4,608 + 512 = 272,384, exactly what launch 28 scored. This
candidate has TWO owners: r512(133,632) = 133,632 + r512(2,088) = 2,560 + 512 = **136,704**.
That is -135,680 B, **-49.81% on champion v15** and **-99.605% on the reference**.
`nopref_kv_cache_bytes` is the same 136,704 and the tiebreak moves with the target, because at
`max_len` 513 the need of 261 still fits so the no-prompt shape rolls at the same extents.
`workspace/checks/share2_bytes_check.py` carries BOTH charge models: the unfused one is gpu5's
`share4_check.py`, imported rather than re-written so its twelve-launch / twenty-four-reading
validation runs before anything is priced, and the fused one is transcribed from champion v15's
own `init_decode_state` and reproduces launch 28 at both request shapes. On the unfused v14 the
same program would have priced at 138,240, so launch 28's KEEP is worth 1,536 B here and is
present in this program rather than dropped -- Step 2c.

**REBASED, and the rebase is the reason this docstring was rewritten once.** This candidate was
built and CPU-checked against champion v14 (275,456); champion v15 published while it was being
checked, so it was rebuilt from v15's `train.py` and every figure above was recomputed rather
than reused. Both prior KEEPs on this lineage -- launch 26's four-owner sharing and launch 28's
fuse -- are in the file.

**PREDICTED `num_params_total` 31,719,504 and `flops_per_token_measured` 165,151,104, both
identities, from `workspace/checks/share2_params_check.py`**, which instantiates this file's
own model definition at the launch config and reproduces champion v14's MEASURED 34,078,864
before quoting the candidate. The -2,359,360 parameters are 2 x 2 x 512 x 128 = 262,144 of
`c_k`/`c_v`, 2 x 32 = 64 of `ve_gate`, and 2,097,152 of value-embedding table. FLOPs move by
6 x (262,144 + 64) = 1,573,248 on the 6x-matmul-parameter identity, which launch 26 confirmed
to the FLOP (-3,145,728 = 6 x 524,288) and launch 15 confirmed on the `ve_gate` term alone
(+768 = 6 x 4 x 32). The window sum is unchanged at 2,048, so the attention span term does
not move: this is a WEIGHT change and not a REACH change, deliberately.

**The forced VE consequence, named and not waived.** `has_ve` returns `is_kv_owner` under
sharing, so two cache owners means TWO value-embedding tables of kv_dim 128 instead of four.
That is not a choice: the cached V is what every reader in the group attends over, so it must
already be the VE-mixed V, and a reader computes no V at all -- the assert in
`CausalSelfAttention.__init__` states exactly this. The one measurement this run has of VE
table COUNT at fixed geometry is launch 15 `struct_mqa_ve_all_layers_v1`, which took 4 tables
to 8 at identical cache bytes, identical allocations and identical decode kernels and
recovered NOTHING (+0.0005819, ~1.5 sd on the same-`forward()` sd). That bounds the UPSIDE of
VE coverage at about zero; it is weak evidence about removing coverage below four, because a
capacity that is already sufficient can be worthless to add and still costly to remove. So
the VE half is banded at 0 to +0.005 rather than priced at zero.

**`val_bpb` is a FIT and this candidate STRADDLES the gate. Claimed as such.** Champion v14
clears by 0.0160225. Rung 1 of this axis measured +0.0073643 (launch 9) and +0.0071281
(launch 26), 3.2% apart across a host change of kv_dim, both window literals and
`TOTAL_BATCH_SIZE` -- which is the measured basis for transferring a WEIGHT delta across
hosts at all (`knowledge/additivity_component_vs_reach.md`). What is NOT measured anywhere is
the CONVEXITY of rung 2, and this run's three models of it disagree by an order of magnitude:

    linear in removed K/V sets (dead_ends' model): 4->2 removes 2 sets vs rung 1's 4  +0.0037
    equal cost per halving of distinct caches                                          +0.0071
    the kv_dim axis's measured convexity (rung 1 +0.00053, rung 2 +0.00283, 5.3x)      +0.0378

Champion v15's `val_bpb` is the number this is added to. Central 1.0455 (the geometric
middle, +0.0095 sharing + 0.0020 VE, on v14/v15's 1.0340-class value), band 1.0378-1.0760
against a gate of 1.05. Two of the three models clear and one refuses by a wide margin. This
is a coin flip on a fit, said out loud, and it is the whole of what the launch buys beyond the
byte identity.

**NO point prediction for either TV metric, and here is the measured reason.** The CPU check
below reads max TV 0.005563 at `KV_GROUP` 1, 0.006362 at 2 and 0.009160 at 4 -- monotone, and
a naive 1.44x on champion v14's live `nopref_decode_tv_distance_max` of 0.0336862 would land
0.0485 against a 0.05 ceiling. That ratio is not a predictor and this run has the measurement
that says so: from launch 24 to launch 26 the CPU ratio was 1.14x UP while the live figure
went 0.0393705 -> 0.0336862, i.e. 0.86x DOWN. So the honest statement is a band, 0.024-0.049,
with 0.0163138 of ceiling left and no number registered. It is the second of the two ways
this candidate can be refused, and unlike a slot-offset change it has a genuine fidelity
mechanism: one int8 cache now serves four windows instead of two.

**Proved on CPU before the launch, and again after the v15 rebase.** `share4_cpu_check.py`
(gpu5's, run unmodified against this file) execs the model-definition region with a CPU stand-in for FA3 and compares
`forward` to `prefill + decode_step` at every position on RANDOMISED weights -- necessary
because `init_weights` zeroes every `c_proj` and `ve_gate`, so the shipped initialisation
makes any decode-vs-forward check read 0 even for a completely wrong cache. This file reads
two distinct allocations for eight layer entries, owners {0,4}, `value_embeds` on {0,4}, and
max TV 0.009160 over 26 positions with the shared-cache rebase firing about five times.
Forced to `KV_GROUP` 1 the same file reproduces the champion-v13 path exactly (eight
allocations, VE on {1,3,5,7}, 0.005563). Two deliberate mutants confirm the check has power at
group size FOUR, where the alias chain is twice as long as a pair's: dropping the reader's +1
`cache_seqlens` reads 0.331166 and rebasing per LAYER over the alias list reads 0.361721, 36x
and 40x the candidate and 6.6x and 7.2x the ceiling.

**Failure-range check, stated rather than glossed.** `dead_ends.md` carries
`struct_cla_share2_v1` at (kv_layer_sharing, replace, 2 owners of 8) as WITHDRAWN UNRUN. No
launch was spent, and its closing argument was explicitly a statement about a host: it
required 4 owners to be eligible at val_bpb <= ~1.0485 and read 1.0549410 on a program with
0.0032 of margin. Champion v14 is that same 4-owner mechanism, eligible, with 0.0160225. This
run has twice re-hosted a gate-refused cut onto a program with margin and KEPT (launch 19 on
launch 7's MQA, launch 26 on launch 9's sharing), and this is the third instance of that move.
The interpolation in that dead-end entry is superseded too: it was arithmetic on launch 9's
byte figures, which are 31x this candidate's.

**What it teaches if it fails.** It measures the convexity coefficient of cross-layer sharing,
which is the only mechanism left on this board that can halve the target again, and which
three models currently disagree about by 10x. With that coefficient the 3-owner map
(~205,312 B, -24.6%) is priceable rather than guessed, and so is the 1-owner floor
(69,120 B, -74.6%) -- both of which are otherwise unqueueable. A TV refusal would be the
first in this run and would establish `nopref_decode_tv_distance_max` as the binding ceiling,
repricing every remaining item across all three teams. Either way the two-owner byte identity
is confirmed or refuted on a program nobody has run.

parent champion: champion.md version 15, `prec_fuseall_sub1mib_v1`, 272,384 (launch 28, gpu4),
rebased onto from version 14, `struct_share4_uniwin_v1`, 275,456 (launch 26, gpu5).
================================================================================
struct_slack3_share2_v1 -- champion v16 at the slack floor
================================================================================
`GPT.DECODE_CACHE_SLACK` 4 -> 3 on champion v16 (`struct_share2_uniwin_v1`, 136,704). ONE class
constant, unconditional, no `EXPERIMENT_ID` gate. Rolling extents 261 -> 260 on both shared caches.

**This is NOT launch 27's program and it must not be confused with it.** Launch 27 measured nothing
and its bytes can never be purchased: `engine/runtime.py::evaluate_batch` refuses any candidate
whose `program_identity` sha256 is already under `engine/source-identities/`, that file is written
`write_once` BEFORE compute, and `program_identity` digests the declared task files' BYTES -- so a
new request id and a new frozen directory do not help, and a cosmetic byte change would be evading
the rule rather than satisfying it. This candidate is a different program for a REASON: its base is
champion v16, two cache owners and launch 28's fused allocation, where launch 27's base was champion
v14 with four unfused caches. Different mechanism host, different bytes, different figure (-512 here
against -2,048 there).

**PREDICTED `kv_cache_bytes` 136,192 and `nopref_kv_cache_bytes` 136,192, IDENTITIES.** On v16's
fused layout the charge is r512(n_owners x 2 x extent x 128) + r512(n_owners x 4 x extent) + 512.
At two owners and extent 260: r512(133,120) = 133,120 (an exact multiple of 512, where 261 was not)
+ r512(2,080) = 2,560 + 512 = **136,192**, i.e. **-512 B (-0.375%)** and **-99.6064% on the
reference**. The tiebreak moves with the target because 260 < 513, so the no-prompt shape rolls at
the same extent. `num_params_total` 31,719,504 and `flops_per_token_measured` 165,151,104 are
UNCHANGED: an extent is neither a parameter nor counted work.

**`val_bpb` exposure is ZERO by mechanism, and that is the point of the launch.** An AST walk of
this file finds `DECODE_CACHE_SLACK` read in exactly two places: `init_decode_state` and the
print-only `_charge_probe`, both of which run after the reporter has computed `val_bpb`. The
training program is bit-identical to champion v16's. So whatever this reads IS a same-training-
program re-draw, and it is the measurement the whole board now needs: **champion v16 is eligible by
0.0005397, and the only measured same-training-program `val_bpb` spread in this run is 0.0023570
(launches 19/20), 4.4x that margin.** Nobody knows whether v16's eligibility is robust or a lucky
draw, and every remaining item on every team's queue is priced against the assumption that it is
robust. Registered as a FIT with NO point prediction, band 1.047-1.052 -- which straddles the gate
on purpose, because that is the honest width of one re-draw.

**What it teaches either way, which is why it is worth a launch for 0.375% of bytes.** If it comes
back eligible, v16's margin survives a re-draw and the board can keep building on it. If it comes
back ineligible at byte-identical training, then v16 sits ON the eligibility boundary, the run has
learned that its champion is a boundary case rather than a comfortable one, and every team should
re-host on v15 (272,384, margin 0.0162773) rather than spend launches discovering this one at a
time. It is also the run's FOURTH same-training-program `val_bpb` reading, the first on a
two-owner program, and the measurement every team's sigma is built on -- and it closes the slack
axis at its floor.

**Legality, checked and tight.** `WARMUP_REPLAYS = 3` and `decode_step` passes
`headroom = WARMUP_REPLAYS + 1 = 4`, so slack 3 gives `extent - win_hist = 260 - 256 = 4` exactly:
it fits with ZERO spare. **Slack 2 does not fit and must never be proposed** -- at slack 2 the
headroom-4 call would fire after a long prefill and copy never-written slots into the live window.
The CPU check below runs the shared cache at exactly this zero-spare geometry.

**Cost, named.** The rebase period falls 5 -> 4 steps, ~128 firings per 512-step request instead of
~102. At launch 16's measured 0.116 ms per firing that is ~+3 ms on v16's measured 407.016 against
a 750 ms ceiling, where 343 ms is unspent. No point prediction for either TV metric: no cached
value changes, but the slot offsets and the rebase cadence do, and launch 16 is the run's only
clean reading of that (+0.004481 prompted, +0.006913 no-prompt) against margins of 0.0266732 and
0.0157622.

parent champion: champion.md version 16, `struct_share2_uniwin_v1`, 136,704 (launch 29, gpu6).

================================================================================
ext_slack1_share2_v1 -- the last affordable byte on the board, and it is affordable because
the slack floor of 3 was never a floor
================================================================================

TWO edits on champion v18 (`struct_slack3_share2_v1`, 136,192, launch 31, gpu6), unconditional,
no `EXPERIMENT_ID` gate: `GPT.DECODE_CACHE_SLACK` 3 -> 1 and `decode_step`'s pre-capture
`headroom=_GraphedDecodeStep.WARMUP_REPLAYS + 1` -> `headroom=1`. Plus gpu4's print-only
`[capture]` probe. Nothing else: `KV_GROUP`, both window literals, `HEAD_DIM`, `KV_HEAD_RATIO`,
`DEPTH`, `TOTAL_BATCH_SIZE`, `DEVICE_BATCH_SIZE`, the optimizer, every learning rate, the data
budget and all three frozen regions are champion v18's.

**This is a ZERO-`val_bpb`-EXPOSURE change, by mechanism and by AST walk.** `DECODE_CACHE_SLACK`
is read only in `init_decode_state` and the print-only probes; `headroom` only in `decode_step`.
Neither runs during training, and `report_efficiency_metrics` has taken `val_bpb` before either is
touched. So this program's training is **bit-identical to champion v18's** and its `val_bpb` is a
same-training-program RE-DRAW, not a new draw. This run has two such pairs: launches 26/28 differ
by 0.0002548 and launches 29/31 -- v18's own pair -- by **0.0001443**. Champion v18 is eligible by
0.0003954, i.e. 2.7x that spread. That is the whole of this launch's risk on the gate, and it is
stated rather than hidden: it is a re-draw bet at ~2.7 sigma, not a quality bet.

**Why this byte was not already bought.** gpu6's `[SUGGESTION]` tonight records
`decode_cache_slack` as "closed at its floor (slack 2 does not fit)", and that is the reading the
whole run has had. It is wrong for a reason gpu4 diagnosed in launch 30: the pre-capture
over-request FIRES below slack 3 and copies never-written slots into the live window -- silent
corruption, not a capture failure -- so 3 looked like a capacity floor. With `headroom=1` the real
requirement is ONE free slot. gpu4 measured the pair live at four owners (launch 30, 269,312) and
MY launch 32 measured it live on THIS two-owner geometry:

    [capture] slack 1  extent 258  win_hist 256  spare 2
    [capture] captured : True
    request_ms_median 411.895 vs champion v17's 407.016   (+4.9 ms, ~154 extra firings)

Launch 32 was refused on `val_bpb` because it ALSO carried `TOTAL_BATCH_SIZE` 2**17, which that
launch measured as a +0.0026087 cost. The slack half was never what was refused, and it is the
half taken here.

PREDICTIONS, labelled (Step 3d):

    kv_cache_bytes           135,168   IDENTITY, and MEASURED on launch 32 at this exact decode
                                       geometry: r512(2*2*258*128 = 132,096) + r512(2,064) = 2,560
                                       + 512. -1,024 (-0.752%) on v18, -99.6094% on the reference.
    nopref_kv_cache_bytes    135,168   IDENTITY, same reason, and also measured on launch 32.
    num_params_total      31,719,504   IDENTITY, unchanged; the diff touches no module.
    flops_per_token_measured 165,151,104 IDENTITY, unchanged; slack is not counted work.
    val_bpb            RE-DRAW of 1.0496046, spread 0.0001443 on this lineage's own pair.
                                       No new central: the training program is bit-identical.
    request_ms_median        ~413      FIT, from launch 32's measured +4.9 ms at these offsets,
                                       against 408.0 for v18. Ceiling 750.
    peak_vram_bytes    39,894,040,576  unchanged: the training microbatch is v18's.
    both TV maxima     NO PREDICTION. The slot offsets move by 2, and the maxima re-draw
                                       0.0089-0.0132 on this run's pairs (my launch 32 moved them
                                       in OPPOSITE directions). v18 holds 0.0257 and 0.0278 of
                                       ceiling, ~2x the re-draw, which is the widest TV margin any
                                       champion has had since launch 21.

WHAT IT TEACHES WHEN IT FAILS. If the gate refuses it, the refusal IS the measurement the whole
board's arithmetic depends on: a third same-training-program `val_bpb` re-draw, and the first one
to land on the wrong side of a 0.0004 margin. Every queue row in this run is priced on the
assumption that a champion eligible by 0.0004 is robust; nobody has tested it, and gpu6's
`[SUGGESTION]` asks the question explicitly. If instead the TV max refuses it, that is the first
TV refusal of the run and it establishes the 0.013 re-draw as a hard planning band rather than an
observation.

parent champion: champion.md version 18, `struct_slack3_share2_v1`, 136,192 (launch 31, gpu6).
Step 2c: launch 31's `DECODE_CACHE_SLACK` 4 -> 3 is SUPERSEDED on its own axis by 3 -> 1 here, not
dropped -- the constant it moved is the constant this moves further, and every other prior KEEP in
v18's `SOURCE` log is present unchanged.

================================================================================
prec_bytefloor_v1 -- the byte-only floor, bought standalone
================================================================================
gpu3's `prec_bytefloor_rider_v1` source (post `ba9cd418`,
`knowledge/byte_only_floor_closed_at_134144.md`), bought as a launch of its own. The executable diff
against champion v19 is AST-identical to gpu3's -- 6 lines removed, 12 added, confined to
`GPT.DECODE_CACHE_SLACK` and the allocation loop -- plus three repriced rows in the print-only
`[charge]` probe. Two changes:

  1. `DECODE_CACHE_SLACK` 1 -> 0. Extent 258 -> 257, spare 2 -> 1.
  2. The two int32 write positions become the tail of the first bfloat16 scale pool, so the
     counter's own 512 B block disappears.

**PREDICTED `kv_cache_bytes` 134,144 and `nopref_kv_cache_bytes` 134,144, both IDENTITIES**, -1,024
against v19's measured 135,168 (-0.758%; -99.6094% -> -99.6123% on the reference). Re-derived at
this candidate's own geometry rather than copied from gpu3's post, by a harness that reproduces
v19's MEASURED 135,168 at BOTH request shapes first (`checks/bytefloor_check.py`, PART A):

    codes    int8      (2,2,1,257,1,128)   requested 131,584   charged 131,584    0 rounding
    pool     bfloat16  (1032,)             requested   2,064   charged   2,560  496 rounding
                                                     TOTAL             134,144   in 2 allocations

Against v19's three of 132,096 + 2,560 + 512. `num_params_total` 31,719,504 and
`flops_per_token_measured` 165,151,104 are unchanged IDENTITIES: the diff touches no module and an
extent is not counted work. **134,144 is the floor of this axis**, not a rung on it: the remaining
496 B of rounding cannot be recovered without dropping real cache data, and 131,584 is already an
exact multiple of 512 so there is no third block to fuse.

**Why standalone, when gpu3 filed it `standalone: NOT RECOMMENDED`.** gpu3 is right about the odds
and I am not re-weighing them: `val_bpb` here is a pure RE-DRAW of the v18/v19 training program,
whose three readings are 1.0496046 / 1.0494249 / 1.0504014, mean 1.049810, sd 0.00052, so the margin
from the program mean is 0.00019 = 0.37 sigma and roughly a third of the draws are refused. The new
argument is about what blocks the rider, not about its odds. **The rider is currently un-rideable:**
`struct_mlp3_v1` declined to carry it, on the record and correctly, because slack 0 / spare 1 has
never been exercised on device and attaching an unexercised legality risk to a launch whose whole
value is a clean quality reading would risk the reading. Every future host has that same objection.
A standalone byte-only candidate is the one place in this run where exercising it costs no reading,
because it has no reading to lose -- so this launch is where that objection gets retired, for the
`ext_share8_uniwin_v1` rung and for everything after it.

**Legality, and the guard is in this file rather than in an argument.**
`_rebase_decode_state(state, headroom=1)` returns early only while `win_pos + 1 <= win_extent`, i.e.
only while the slot about to be written exists. With headroom 1 the guard's predicate IS that bound,
for any spare, so spare 1 is not a special case of it; the rebase then fires at
`win_pos == win_extent` exactly, where its source slots `[win_extent - hist, win_extent - 1]` have
all been written. That is why launch 30's corruption cannot recur here: it needed headroom 4, which
fires early and copies slots nothing wrote.

Checked on CPU (`checks/bytefloor_check.py`, torch 2.9.1 as uv.lock pins), scaled to window 8 so
extent 9 / spare 1 is the same quantity, on the shipped decode path with the CUDA-graph capture
emulated on the host (the pre-capture rebase, `WARMUP_REPLAYS + 1` real appends each preceded by
`_restore`, then the restores) -- the part `graph=False` never runs and the part spare 1 was doubted
on. All four capture appends land on the SAME slot 8 of 0..8, and the pre-capture rebase does not
fire. The bfloat16 scale buffers are POISONED WITH NaN at init, so a read of any never-written slot
is a NaN in the logits rather than a statistical argument: the candidate reads no NaN and max TV
0.007248, the int8 floor, identical to champion v19's through the same harness. Two mutants show the
check has power: restoring `headroom = WARMUP_REPLAYS + 1` at this slack puts NaN in the logits, and
spare 0 raises `IndexError` from `index_copy_`.

**What is exactly null and what is only nearly null, decomposed rather than asserted.** The counter
fusion is EXACTLY value-null: the champion file forced to slack 0 and this candidate give max
|logit diff| 0.000e+00 at every scored position on randomised weights. The slack step is null in
real arithmetic but not in float: v19 hands the decode kernel `win_hist + 2` keys on alternate steps
and this candidate always hands it `win_hist + 1` (measured: keys {9, 10} vs {9}), so one fewer
masked `-inf` term enters the softmax and the logits move 1.669e-06 -- five orders below the
0.007248 int8 floor, and both TV metrics re-draw ~0.010 on this substrate anyway. So both TV maxima
are RE-DRAWS with no mechanism, not predictions; live margins are 0.0238835 and 0.0178413.

`request_ms_median` is the one FIT: the copy fires every step instead of every second step, ~512
firings per request instead of ~256, priced at +8.2 ms from launch 32's measured 0.032 ms/firing, so
~419 ms against v19's 411.256 and a 750 ms ceiling. `peak_vram_bytes` falls by 1,024.

parent champion: champion.md version 19, `ext_slack1_share2_v1`, 135,168 (launch 34, gpu2).
Step 2c: this supersedes launch 34's `DECODE_CACHE_SLACK` 1 on its own axis by moving the same
constant further; every other prior KEEP in v19's `SOURCE` log is present unchanged.

================================================================================
struct_perlayer_reach_v1 -- the uniform-rolling-extent guard, LIFTED: reach is spent where
this run measured it cheap, and not where it measured it dear
================================================================================

FOUR literals and ONE mechanism on champion v20 (`prec_bytefloor_v1`, 134,144, launch 38, gpu4),
all unconditional, no `EXPERIMENT_ID` gate:

    KV_GROUP        4 -> 2         two cache owners -> four, {0,1} {2,3} {4,5} {6,7}
    WINDOW_PATTERN  "SSSL" -> "SSSSSSLL"
    SHORT_WINDOW    256 -> 63
    LONG_WINDOW     256 -> 255
    the guard in `init_decode_state` that collapsed every extent to `max_len` whenever two
    rolling extents differed: LIFTED, by carrying one write position and one rebase schedule
    per DISTINCT ROLLING EXTENT instead of one in total.

`prepare.py`, `pyproject.toml`, `uv.lock`, all three frozen regions, `DECODE_CACHE_SLACK`,
`HEAD_DIM`, `KV_HEAD_RATIO`, `DEPTH`, `TOTAL_BATCH_SIZE`, `DEVICE_BATCH_SIZE`, the optimizer,
every learning rate and the data budget are champion v20's.

**The mechanism, and why it was bookkeeping rather than a limit.** The champion's guard read:

    if hist is not None and len({e for e, r in zip(extents, rolling) if r}) != 1:
        extents = [max_len] * cfg.n_layer

and its own comment gave the reason -- "two different short windows would need two rebasing
counters and two schedules" -- naming no fidelity and no kernel constraint. So this candidate
generalises the bookkeeping instead of accepting the collapse: `win_extents` is the ascending
list of distinct rolling extents, `win_hists[k]` its group's `left` (recovered from the loop's own
identity `extent = left + 1 + slack`), `win_slots[i]` the group layer `i` reads, and `seq_wins`,
`win_poss` and `_rebase_decode_state`'s predicate all become per-group. `ctr` grows from two int32
rows to `1 + len(win_extents)`, still slices of the first scale pool's tail, so the row count now
follows the number of schedules rather than being fixed at one. **With a single distinct rolling
extent every one of those lists has length one and the arithmetic is the champion's, unchanged** --
CPU-checked below as an equality, not asserted.

Everything the lift touches -- `init_decode_state`, `_rebase_decode_state`, `_step_device_counters`,
`_step_host_position`, `_decode_body`'s non-prefill branch, `decode_step`, `_GraphedDecodeStep` --
runs only in the decode path, after `report_efficiency_metrics` has taken `val_bpb`. **The lift
itself spends ZERO gate.** What spends gate is the three window literals, and by this run's own
measurements it spends it on the layers where reach was cheap.

**Why the reach goes where it goes.** Local shortening has GAINED `val_bpb` at every rung this run
has run (launch 22 and its convexity pair: -0.0098, -0.0103, -0.0114); the only long-layer
shortening run COST (+0.00145). Launch 40 then measured the whole uniform 256 -> 127 halving at
**+0.0203608** once gpu2's measured sharing revert (`KV_GROUP` 4 -> 2, -0.0157376) is subtracted --
a single-host number, not a decomposition. This candidate pays a reach price on six locally
attending layers and **nothing** on layers 6-7, which keep 255 keys against the champion's 256, and
banks the -0.0157376 revert as margin. Bytes are `256 * sum over OWNERS of extent`, so four owners
at [64, 64, 64, 256] is 448 units against the champion's 2 x 257 = 514.

**The one bet, in one parameter.** Let `p` be the first-rung (256 -> 127) price on a locally
attending layer, second rung at the two ladders' measured 2.2x escalation, so two rungs cost 3.2p:
this program costs `19.2p - 0.0157376`, i.e. it is eligible iff `p` is under ~31% of launch 40's
eight-layer average of +0.002545. It does NOT need local reach to still be gaining below the
coverage line; it needs locals to be cheaper than average, which is the weaker of the two claims
and the only one three measured rungs support. `val_bpb` band **1.0379 - 1.0679, no central**, which
crosses the gate -- said out loud rather than buried. A refusal isolates `p` for the first time and
every remaining reach row on every team is priced on it.

Identities (CPU-verified by `checks/perlayer_reach_check.py`, torch 2.9.1 as uv.lock pins, which
reproduces champion v20's MEASURED 134,144 at both request shapes and its MEASURED 31,719,504
params and 165,151,104 flops/token, and launch 40's MEASURED 34,078,864 and 160,383,744, before
reading this candidate):

    kv_cache_bytes         116,736   65,536 + 49,152 + 1,024 + 1,024, every allocation < 1 MiB
    nopref_kv_cache_bytes  116,736   at max_len 513 both needs (64, 256) still roll: same program
    num_params_total    34,078,864   the four-owner count, MEASURED at launches 26/27/29/40
    flops_per_token    159,597,312   154,141,440 + 6,144 * (6*63 + 2*255)

No point prediction for either TV maximum: the slot layout, two rebase cadences and every weight
move, and this run's fifth agreeing reading says that metric is draw-dominated. `request_ms_median`
is a FIT: two rebase copies per step where the champion had one, but of 64 and 256 slots against
its 257, so at or below v20's measured 418.7 ms against a 750 ms ceiling. `peak_vram_bytes` is a
FIT at or below launch 40's measured 40,448,223,232 against 47,198,976,512.

**The lift is a CORRECTNESS change, not a literal swap, so it can be wrong in a way the ranking
metric never shows.** `checks/perlayer_reach_check.py` therefore also shows: (C1) forced to uniform
windows this file reproduces the champion's 134,144, its single rolling counter, its (2, batch)
`ctr` and -- at windows 8 -- its decode-vs-forward max TV to every printed digit (0.008391 both),
so the generalisation reduces; (C2) the SHIPPED champion given 63/255 still collapses to
`extents = [2048] * 8`, so the guard is real and is what this row buys; (C3) `"SSSL"` at
`KV_GROUP` 2 -- long layers at depths 3 and 7, straddling a group boundary -- is still REFUSED by
the group-agreement assert; (D) at NON-UNIFORM extents [9, 17] with the scale buffers POISONED WITH
NaN and the CUDA-graph capture emulated on the host, max TV 0.005263, the int8 floor, and no NaN.
Two mutants give that check power: forcing every rolling layer onto the FIRST group's counter --
precisely the coupling this generalisation removes -- reads 0.351175, and gpu4's launch-30 headroom
mutant puts NaN in the logits.

Not folded in: the two-owner variant at extents [64, 256] (83,456 B, -37.79%) is the SUCCESSOR, not
a sibling. It drops the sharing revert, so it needs `p <= 0` -- local reach still strictly gaining
below the coverage line -- and launch 40's 2.17x escalation is exactly what puts that in doubt. It
is claimable only if this row lands eligible and publishes `p <= 0`.

Depth placement is the second unmeasured freedom and is disclosed rather than defended: at
`KV_GROUP` 2 the long pair must be {6, 7}, while this run's best `val_bpb` (launch 22, 1.0354081)
had its long layers at depths 3 and 7. Nobody has measured the placement.

parent champion: champion.md version 20, `prec_bytefloor_v1`, 134,144 (launch 38, gpu4).
Step 2c: `champion/train.py` copied byte-identical (md5 192435947fd0f96844d9361bd6860384), then
four literals and the guard lift applied. Diffed against `champion/SOURCE`: every prior KEEP is
present -- MQA, the fused sub-1-MiB allocation, `KV_GROUP` sharing (moved on its own axis here,
4 -> 2, and that revert is the change, not a dropped KEEP), `DECODE_CACHE_SLACK` 0 and the
counters-in-the-scale-pool floor -- and the window literals are the axis this row moves.
================================================================================
struct_locals31_reach_v1 -- the third local rung, bought against a measured coefficient
instead of a sign
================================================================================

ONE literal on champion v21 (`struct_perlayer_reach_v1`, 116,736, launch 41): `SHORT_WINDOW` 63 -> 31.
Nothing else moves -- `KV_GROUP` 2, `WINDOW_PATTERN "SSSSSSLL"`, `LONG_WINDOW` 255,
`DECODE_CACHE_SLACK` 0, the guard lift, `HEAD_DIM`, `KV_HEAD_RATIO`, `DEPTH`, `TOTAL_BATCH_SIZE`,
`DEVICE_BATCH_SIZE`, the optimizer, every learning rate, the data budget and all three frozen regions
are champion v21's. Owner extents [32, 32, 32, 256] = 352 units against v21's 448.

**This is the first row in this run priced against a MEASURED per-layer-class coefficient rather than
a sign.** Launch 41 published `p = +0.00020684`, the first-rung `val_bpb` price of a locally attending
layer, and the same launch measured that reach is ~12x dearer on the two globally attending layers
(`L = +0.009559` per layer per rung, from launch 40's eight-layer average of +0.002545). So the long
pair is not touched: two long rungs would cost +0.019118 against 0.0118732 of margin and refuse on
their own. Only the six local layers move, and they move one more rung.

At the two ladders' measured 2.2x escalation a third rung on a local layer costs `4.84p`, so six of
them cost `29.04p = +0.006010` and land `val_bpb` at **1.044137, eligible by ~0.005863** -- half the
margin launch 41 opened, spent on 21.5% of the remaining bytes.

Identities (CPU-verified by `checks/locals31_check.py`, which first reproduces champion v21's MEASURED
116,736 at both request shapes and its MEASURED 34,078,864 params and 159,597,312 flops/token):

    kv_cache_bytes         91,648   24,576 + 512 + 65,536 + 1,024, every allocation < 1 MiB
    nopref_kv_cache_bytes  91,648   at max_len 513 both needs (32, 256) still roll: same program
    num_params_total   34,078,864   UNCHANGED -- KV_GROUP does not move, so no VE table is added
    flops_per_token   158,417,664   154,141,440 + 6,144 * (6*31 + 2*255)

`val_bpb` is the FIT and the whole bet: **band 1.0394 (escalation 1.0) - 1.0493 (escalation 3.0),
central 1.0441 at the measured 2.2x.** The band is inside the gate at every escalation this run has
measured, which is what makes this different from launch 41: there the band crossed the gate, here it
does not, and the launch fails only if the escalation is steeper than anything two ladders have shown.
What a refusal would teach is that the escalation compounds faster than 2.2x below the coverage line,
which is the one thing the 2.17x reading could not settle.

FIRST-NAMED RISK, and it is not the gate: **`decode_tv_distance_max`.** v21 read 0.0394996, leaving
0.0105004. That reading is a DRAW, not a property this candidate inherits -- v21's
`decode_tv_distance_mean` was 0.00800, inside the 0.00772-0.00856 range of all 15 MQA-era launches, so
fidelity is unchanged -- but across those 15 launches tv_max has mean 0.02950, sd 0.00742 and a worst
reading of 0.04969, i.e. launch 37 came within 0.00031 of refusal on a program with no fidelity change
at all. 0 of 15 breached 0.05. So this launch carries roughly a 1-in-30-to-1-in-100 chance of a
refusal nothing in the candidate controls. **No point prediction is registered for either TV maximum**
and none should be.

`request_ms_median` is a FIT and should FALL: the same 8 buffer copies per step on the same two
schedules, but copying 3x31 + 255 = 348 slots against v21's 444. Launch 41 measured the rebase as
launch-bound rather than bandwidth-bound (+0.060 ms/step for 4 extra copies while copying fewer
slots), so the honest prediction is **449.5 ms +- a few**, not a proportional fall -- and 300 ms of the
750 ms ceiling is unspent either way. `peak_vram_bytes` is a FIT at or below v21's 40,448,223,232.

parent champion: champion.md version 21/22, `struct_perlayer_reach_v1`, 116,736 (launch 41, gpu5).
Step 2c: `champion/train.py` copied byte-identical, then ONE literal moved on the axis launch 41 owns.
Every earlier KEEP in v21's `SOURCE` log is present unchanged; `SHORT_WINDOW` is superseded on its own
axis, not dropped.

================================================================================
ext_longreach_frac_v1 -- the first purchase on the LONG-layer reach axis, at a
fraction of a rung instead of a rung
================================================================================

ONE literal on champion v23 (`struct_locals31_reach_v1`, 91,648, launch 43): `LONG_WINDOW`
255 -> 225. Nothing else moves -- `SHORT_WINDOW` 31, `KV_GROUP` 2, `WINDOW_PATTERN "SSSSSSLL"`,
`DECODE_CACHE_SLACK` 0, the guard lift, `HEAD_DIM`, `KV_HEAD_RATIO`, `ASPECT_RATIO`, `DEPTH`,
`TOTAL_BATCH_SIZE`, `DEVICE_BATCH_SIZE`, the optimizer, every learning rate, `WEIGHT_DECAY`, the
data budget and all three frozen regions are champion v23's. Owner extents
[32, 32, 32, 256] -> [32, 32, 32, 226], 352 -> 322 units.

WHY THIS AXIS, AND WHY A FRACTION. Launch 43's own result file states where the metric now lives:
of 91,648 B, **65,536 (71.5%) is the single long-pair owner buffer at extent 256**, and the three
local buffers are 24,576. The local ladder has been walked three rungs and the long pair has never
been touched, because the only price anyone had for it was a WHOLE rung: `L = +0.009559` per
globally attending layer per halving, so 255 -> 127 costs `2L = +0.019118` against 0.0073442 of
margin and refuses on its own arithmetic. Launch 43 concluded "the long pair remains untouchable
and ... Do not file it", and that is correct **of a rung**. It is not correct of the axis: the
charge is `256 * sum(extents)` and `extents` is an integer, so this axis is purchasable in units of
**2 slots = 512 B**, not in halvings. Nobody has priced a partial dose of it.

I derive the fraction from the run's own coefficient rather than picking one. Taking the response
as locally log-linear in the extent, `cost(256 -> E) = 2L * log2(256 / E)`:

    E = 226  ->  2L * log2(1.132743) = 0.019118 * 0.179891 = +0.0034397

which spends **47% of the margin** and leaves 0.0039045. E = 226 is chosen because `256 * E` must
be a multiple of 512 for the codes allocation not to waste a block, i.e. E even, and because a
bolder point does not survive the error this coefficient actually carries (below).

IDENTITIES (CPU-verified by `checks/longreach_frac_check.py`, which first reproduces champion v23's
MEASURED 91,648 at both request shapes and its MEASURED 34,078,864 params and 158,417,664
flops/token, and launch 41's MEASURED 116,736 and 159,597,312, before pricing anything of mine):

    kv_cache_bytes         83,968   24,576 + 512 + 57,856 + 1,024, every allocation < 1 MiB
    nopref_kv_cache_bytes  83,968   at max_len 513 both 32 and 226 still roll: same allocation list
    num_params_total   34,078,864   UNCHANGED -- a window literal reaches no module and no table
    flops_per_token    158,049,024  154,141,440 + 6,144 * (6*31 + 2*225)

`val_bpb` is the FIT and the whole bet, and the honest statement is about the coefficient rather
than about the point. **`L` is DERIVED, not measured**: it comes from launch 40's uniform
eight-layer rung (+0.0206045, i.e. +0.002575/layer) minus launch 41's `p = +0.00020684` on the six
local layers, divided by two. So it inherits the error of both, and no launch in this run has ever
put a number on a globally attending layer's reach directly. Priced as a band on `L` itself:

    L over-charged 24% (the direction launch 43 measured on the LOCAL ladder)  +0.0026  -> 1.0453
    L exactly as derived                                                       +0.0034  -> 1.0461
    L under-charged 50%                                                        +0.0052  -> 1.0479
    L under-charged 100%                                                       +0.0069  -> 1.0496

**Every one of those clears the gate**, the last by 0.0004. A refusal requires `L` to be more than
2.1x its derived value, which is a bigger miss than any coefficient-based fit in this run has
produced (launch 43's was -0.0015, in the safe direction). That is the point of buying a fraction
rather than a rung: it converts an axis that refuses by arithmetic into one whose whole band is
inside the gate.

WHAT IT TEACHES WHEN IT LOSES, which is the part that makes it worth a launch either way. **It is
the first direct measurement of `L`.** Whatever `val_bpb` comes back, dividing the delta by
`2 * log2(256/226)` yields `L` measured on the two globally attending layers alone, at update
counts that cannot drift more than ~0.05% (flops/token falls 0.233%). That number decides whether
71.5% of the remaining metric is ever fundable, and it currently rests on a two-equation split
nobody has checked. If it comes back at or below the derived value, the next row can take a much
larger bite of the same buffer; if it comes back above, the long pair is closed **on a measurement**
instead of on an inference, and the board stops re-deriving it.

`decode_tv_distance_max`: **no point prediction, and none should be registered.** Champion v23's
own record and `knowledge/decode_tv_max_is_a_weight_redraw_not_a_margin.md` establish that this
maximum is a weight DRAW -- v22 read 0.0394996 and v23 read 0.0248737 on a change that shortened
attention, i.e. the wrong direction for a fidelity reading -- while `decode_tv_distance_mean` stays
in the 0.00772-0.00856 band of every MQA-era launch. This candidate shortens two windows and adds
no mechanism, so fidelity should not move; the launch still carries the run's standing
~1-in-30-to-1-in-100 chance of a TV refusal it does not control.

`request_ms_median` is a FIT and should NOT fall proportionally: the same 8 buffer copies per step
on the same two schedules, copying 3x31 + 225 = 318 slots against v23's 348. Launch 43's controlled
confirmation put the rebase at **per buffer-copy CALL, not per slot** (a 21.6% cut in slots moved
`request_ms_median` by -0.020 ms against an IQR of 1.74), so the prediction is **~449.5 ms**, and
300 ms of the 750 ms ceiling is unspent either way. `peak_vram_bytes` is a FIT at or below v23's
40,448,223,232.

parent champion: champion.md version 23, `struct_locals31_reach_v1`, 91,648 (launch 43, gpu5).
Step 2c: `champion/train.py` copied byte-identical (md5 f6a9dbbd82106b44ae5eb3d81c67716f), then ONE
literal moved. `prepare.py`, `pyproject.toml`, `uv.lock` and all three frozen regions are champion
v23's, byte-identical. Diffed against `champion/SOURCE`: every earlier KEEP is present;
`LONG_WINDOW` is superseded on its own axis (launches 21/22/23/41 set the window literals, 41 split
them), not dropped. No recompose item is owed. Recorded because I found it and it matters to the
audit trail: this KEEP was written to `results/` at 03:38 and published at 03:40, and for those two
minutes `champion.md` still read 116,736 -- I parented on launch 43's frozen source rather than on
the then-current `champion.md`, and the publication landed before I submitted, so the two agree.

================================================================================
prec_autotune_long201_v1 -- the long-reach axis walked one more partial dose,
carrying the run's only unspent margin source, which a byte TIE could never bank
================================================================================

TWO executable changes on champion v24 (`ext_longreach_frac_v1`, 83,968, launch 44, gpu1),
copied byte-identical (md5 98aff44c5bce1c4fc1e778b29b7591a5) and nothing else moved --
`SHORT_WINDOW` 31, `KV_GROUP` 2, `WINDOW_PATTERN "SSSSSSLL"`, `DECODE_CACHE_SLACK` 0, the
guard lift, the pooled counters, `HEAD_DIM`, `KV_HEAD_RATIO`, `ASPECT_RATIO`, `DEPTH`,
`TOTAL_BATCH_SIZE`, `DEVICE_BATCH_SIZE`, the optimizer, every learning rate, `WEIGHT_DECAY`,
the data budget and all three frozen regions are v24's; `prepare.py`, `pyproject.toml` and
`uv.lock` are byte-identical to `champion/`:

  1. `LONG_WINDOW` 225 -> 201. Owner extents [32, 32, 32, 226] -> [32, 32, 32, 202],
     322 -> 298 units. 0.1620 of a rung on the axis launch 44 opened.
  2. `torch.compile(model, dynamic=False)`
     -> `torch.compile(model, dynamic=False, mode="max-autotune-no-cudagraphs")`.

WHY THE SECOND CHANGE IS HERE AND NOT IN A ROW OF ITS OWN. Launch 42 measured that mode at
`val_bpb` 1.0479096 against launch 38's 1.0498930 -- a GAIN of 0.0019834, of which 0.0015423
is exactly +3.582% optimizer updates at launch 24's measured exchange rate -- and was recorded
DISCARD BY TIE, because `kv_cache_bytes` came back byte-identical at 134,144. A tie can never
be champion, so that line cannot enter `champion/train.py` on its own merits at all: it has to
ride a byte row or it is lost. Three byte rows (43, 44, 45) have gone by and v24 line 2402
still reads `torch.compile(model, dynamic=False)`. This run has 55 launches and 0.0042716 of
gate margin left, and both byte-free quality levers that were TESTED came back negative
(`MLP_HIDDEN_MULT` 3 at 1.0541274, `MATRIX_LR` 0.027 at 1.0503525). Margin, not launches, is
the scarce resource, and this is the only measured positive on the board.

THE BYTES, AN IDENTITY. `kv_cache_bytes = 256*sum(owner extents)` plus one 512-rounded
bfloat16 scale pool per DISTINCT owner extent, the `n_ctr = 3` int32 rows riding in the first
pool's tail. From this file's own `init_decode_state`, at batch 1 and both request shapes
(`need` = 202 < 513, so the no-prompt shape is the same allocation):

    codes   3 x 32-slot owners   24,576   unchanged      pool 0     512   unchanged
    codes   1 x 202-slot owner   51,712   was 57,856     pool 1   1,024   unchanged (129<=E<=256)
                                 -----------------------------------------------------------
                                 77,824   vs 83,968  =  -6,144 B (-7.32%), -99.7751% on reference

202 is EVEN, so no code block is wasted; `LONG_WINDOW` 200 (extent 201) would throw 256 B away
and measure the same 77,824, which is why the literal is odd. `nopref_kv_cache_bytes` 77,824.
`num_params_total` 34,078,864, unmoved -- a window touches no parameter.
`flops_per_token_measured` = 154,141,440 + 12*512*(6*31 + 2*201) = 157,754,112, against a
239,078,400 ceiling; that formula reproduces launches 41, 43, 44 and 45 exactly.

THE GATE, PRICED AS A BAND ON `L` RATHER THAN A POINT ON `val_bpb`. gpu1 measured
`L = 0.0085434891` per globally attending layer per halving at identical update counts
(1990 = 1990), so cost = 2*L*log2(226/202) = 0.3239404*L:

    L measured, rider contributes its measured 0.0019834            1.0465124  margin 0.0034876
    L measured, rider contributes only its 0.00154 update part      1.0469558  margin 0.0030442
    L measured, RIDER CONTRIBUTES NOTHING                           1.0484958  margin 0.0015042
    L = the old DERIVED 0.009559 (over-charges 10.6%) AND no rider   1.0488247  margin 0.0011753

Refusal needs `L >= 1.543x` its measured value with the rider contributing nothing, or 2.10x
with it working; the superseded DERIVED value is 1.119x. That is why 202 and not 196 -- at
extent 196 the zero-rider margin is 0.0003, so the row's KEEP would depend on a fit about a
DIFFERENT metric.

WHY 202 AND NOT 204, WHICH IS WHAT THIS FILE SAID HALF AN HOUR AGO (Step 3f). This row was
built at `LONG_WINDOW` 203, extent 204, predicting 78,336 B. gpu2's `ext_dualreach_equalmc_v1`,
claimed at 04:48:42Z -- 2m48s before this row's claim -- predicts THE SAME 78,336 B from
[26,26,26,222] (300 units) by splitting the cut across both reach axes. Two different programs,
so the engine charges both, and an eligible TIE is not strictly better, so whichever landed
second would be a DISCARD however exact it was. Their claim is older, so THIS row moved its own
rung: extent 204 -> 202, 300 -> 298 units, 78,336 -> 77,824 B, one allocator block lower, for
0.0002632 more predicted gate cost. The pair now prices HOW a ~24-unit cut is SPLIT at nearly
constant bytes -- 18 local + 4 long against 0 local + 24 long -- which is worth more than either
launch alone and is what neither could measure by itself.

THE THINNEST CEILING THIS ROW CARRIES IS `peak_vram_bytes`, AT 3.69%, AND IT IS A FIT. Launch
42 measured +5,008,872,960 B for the mode (37 autotuned kernels x 19 candidates, each
benchmarked with its own output and workspace buffers at M = 262,144). v22, v23 and v24 all
read `peak_vram_bytes` 40,448,223,232 EXACTLY across three different cache geometries, so the
base is geometry-insensitive and the additive fit sits on firm ground; the autotuned GEMM
shapes are v24's unchanged. 40,448,223,232 + 5,008,872,960 = 45,457,096,192 of 47,198,976,512,
leaving 1,741,880,320. Nothing memory-raising is composed with it, and shortening attention
moves activation memory the safe way. Compile time is pre-registered as outside the 600 s
training clock and launch 42 spent 798.4 s of the 1,800 s timeout.

NO PREDICTION IS REGISTERED FOR EITHER TV MAXIMUM, and that is deliberate. gpu6's launch-45
amendment measures the breach rate at 1 in 17 (~6%) and a TV refusal is unappealable, since
`task.json` pins `launch.seed: 42`. This row does not move any LOCAL extent -- `SHORT_WINDOW`
stays 31, and local extent 32 has two clean readings (0.0248737, 0.0205099) while the only
breach in the run sits at local extent 20 -- so it is nowhere near the few-keys regime that
mechanism note describes. It is priced as a ~6% tax I cannot control.

parent champion: champion.md version 24, `ext_longreach_frac_v1`, 83,968 (launch 44, gpu1).
Step 2c: `champion/train.py` copied byte-identical, then one literal moved on launch 44's own
axis and one keyword added to one call. Every earlier KEEP in v24's `SOURCE` log is present
unchanged; `LONG_WINDOW` is SUPERSEDED ON ITS OWN AXIS (launches 21/22/23/41/44 set it), not
dropped, and `SHORT_WINDOW` 31 is preserved. No recompose item is owed. CPU pre-flight:
`checks/longreach201_autotune_check.py`, which reproduces four measured launches at three
metrics each before it is allowed to predict anything, and carries mutants.

prec_k7_dualreach_recompose_v1: SEVEN BITS PER CACHED K ELEMENT, on champion v27's geometry.

One mechanism, in the decode cache only. A cached K vector is `head_dim` = 128 codes, and at 7
bits that is 896 bits = 112 BYTES EXACTLY -- no remainder, no padding slot, and the only
sub-byte width with a clean layout at this head_dim. Per cached position and head the pair
costs K 112 + V 128 = 240 B instead of 256. V's quantiser is arithmetically UNCHANGED (checked
bit-identical on CPU: same codes, same scale) and the scale tensors keep their shape, dtype,
count and granularity, so both scale pools and the `n_ctr = 3` counter rows riding in the first
pool's tail are byte-identical and `prec_bytefloor_v1`'s fusion survives untouched.

`forward()` IS NOT TOUCHED. `val_bpb`, `num_params_total` and `flops_per_token_measured`
therefore cannot move by mechanism, and that is the whole reason this row exists. Champion v25's
record closed the reach axes -- equal marginal cost per byte, within one sub-noise step of
exhausted -- and v27's own `spend_the_next_margin_here` prices the next long-reach rung at
+0.0015768 of the 0.0026826 it holds. This row buys 4,608 B and spends NONE of it.

PREDICTED, and registered as IDENTITIES: `kv_cache_bytes` 73,216 and `nopref_kv_cache_bytes`
73,216 (-4,608, -5.921% on v27; -99.7884% on the measured reference 34,603,520),
`num_params_total` 34,078,864, `flops_per_token_measured` 157,754,112 -- the last two are v27's
own measured values, unchanged because `forward()` is. The byte figure is not arithmetic typed
into a file: `checks/k7_pack_and_bytes_check.py` builds a REAL `init_decode_state` at both
request shapes and applies the run's charge law (512*ceil(nbytes/512) per live allocation) to
the allocations the state actually holds. It reproduces v27's measured 77,824 exactly at both
shapes before it is allowed to predict anything, and reads 73,216 for this candidate at both:

    pool                     v27 (256 B/slot)      int7-K (240 B/slot)
    codes, extent 32 (x3)    24,576 -> 24,576      23,040 -> 23,040   (45 blocks, EXACT)
    codes, extent 202 (x1)   51,712 -> 51,712      48,480 -> 48,640   (95 blocks, +160)
    scales, ext 32 + ctrs        396 ->    512        396 ->    512    UNCHANGED
    scales, ext 202              808 ->  1,024        808 ->  1,024    UNCHANGED
                             ---------------       ---------------
                             77,824 = MEASURED     **73,216**

One allocation per distinct owner extent is preserved and all four are far below 1 MiB, where
this run's charge law is exact. Note the parity rule of v25/v27 does NOT carry across the width
change: at 240 B/slot the local pool is exact at extent 32 and the long pool wastes 160 B at
202, so future rungs must be re-rounded at 240 and not at 256.

WHAT IT COSTS, priced on v27's OWN draws and not on a lower one. The cache's element error rises
by a MEASURED x1.334 (`checks/k7_pack_and_bytes_check.py`: K int7 rms error 0.012698 against
int8's 0.006185 = x2.053, combined with V's unchanged 0.010921), reproduced at x1.358 on a
heavy-tailed input. NO PREDICTION IS REGISTERED FOR EITHER TV MAXIMUM. The band, stated four
ways because the exponent is a two-point fit ACROSS geometries and this run has measured the
metric's same-geometry spread at 0.026 and its same-program re-draw at 0.0025: linear in the
floor-subtracted part 0.0356, linear in the whole reading 0.0388, the run's e=1.92 fit 0.0441,
its e=2.4 fit 0.0497, against 0.05. Three of four are inside with room and the fourth is at the
line, so this is bought as a ~6% per-launch tax (gpu6, launch 45) plus an exponent the run
cannot pin -- and pinning it from above at a multiple below x1.5, at FIXED weights and FIXED
geometry with `forward()` untouched, is what the launch produces if it is refused. Every other
(element error, TV) point in this run confounds the quantiser with a weight re-draw: launch 14's
was at a cache 126x larger, and the four window-length points moved the geometry. This one
cannot, because the training path is arithmetically identical to v27's.

A zero-byte rider was measured and REFUTED rather than assumed: `analyst2` recorded MSE-optimal
clipping as a free error reducer riding on this row. Swept on CPU over rms-normed K, every clip
factor below the absmax is strictly worse -- 0.95 costs x2.82 of int8 error against absmax's
x2.05, and 0.90 costs x4.73 -- because an rms-normed vector has no outlier to clip and the
largest components dominate the round trip. This is the champion quantiser's own docstring
finding (power-of-two scales cost 9.5% instead of 0.7%) at a second width. Clipping is closed as
a rider on this axis, and the multiple stays x1.334.

Other exposures, all stated: `request_ms_median` gains one compiled pointwise unpack region per
cache owner per step, predicted +20 to +150 ms on v27's 448.231 against 301.769 of margin (a
FIT; launch 14 measured +88.973 ms for an EAGER 6-bit unpack at ~10 kernels/layer on a cache
126x larger). `peak_vram_bytes` gains one transient int8 intermediate of at most 26 KB against
6,750,753,280 unspent. One more compiled region costs step-0 wall clock, which the 600 s
training clock provably does not see (`known_constraints`: "training work only, not compilation
or evaluation") and which has ~900 s of the 1,800 s timeout unspent at launch 47's 881.8 s.
`val_bpb` is a PURE RE-DRAW of v27's 1.0473174 with 0.0026826 of margin -- 2.09x what v25 held,
which is the second reason this parent was chosen over v25.

parent champion: champion.md version 27, `prec_autotune_long201_v1`, 77,824 (launch 47, gpu4).
REBASED, Convention 2: this candidate was built on champion.md version 25
(`ext_dualreach_equalmc_v1`, 78,336) where the same mechanism predicts 74,240, and rebased onto
v27 when gpu4 published at 05:40:37Z, before submission. Step 2c: `champion/train.py` copied
byte-identical (md5 23d76d57922154d566fbd0c56ceeff5d), then `_quantise_kv_eager`,
`init_decode_state`, `_rebase_decode_state` and `_decode_body` changed and `_pack7` / `unpack7`
added. `prepare.py`, `pyproject.toml`, `uv.lock` and all three frozen regions are v27's,
byte-identical. NO window, batch, optimizer, learning-rate or data constant moved -- the CPU
protocol harness refuses to run if they differ from `champion/`.

LINEAGE DEBT, INHERITED AND NOT DISCHARGED BY THIS ROW. v27 does not contain launch 46's
`SHORT_WINDOW` 25 (gpu4 recorded that itself under `dropped_change`), and this row does not put
it back, because putting it back is a reach purchase that spends the margin this row is
advertising it does not spend. The recompose item stays OPEN. Its value RISES if this row lands:
on v27 it is extents [26,26,26,202] at 256 B/slot = 73,216 B, and on this candidate's geometry
it is 3*26*240 = 18,720 -> 18,944 plus 202*240 = 48,640 plus 1,536 = **69,120 B**.

STEP 3f, A TIE THAT IS NOT WITH AN IN-FLIGHT ROW AND IS RECORDED ANYWAY: that same recompose row
predicts exactly 73,216 B on v27, which is this candidate's own predicted value. Nobody holds a
claim on it -- all three queues read `claims: {}` at 05:45Z -- but an eligible tie is not
strictly better, so if it is claimed while this is in flight one of the two is a DISCARD by
construction. Posted on the board rather than resolved by moving this row, because this row has
no adjacent value to move to: int6 K and int7/int7 are both refused by this run's own
predictions, so x1.334 at 73,216 is the whole of this axis.

ext_locals29_on240_v1: SHORT_WINDOW 31 -> 28, and the EVEN-EXTENT RULE IS DEAD AT 240 B/SLOT.

ONE literal on champion v28 (`prec_k7_dualreach_recompose_v1`, 73,216, launch 48), copied
byte-identical (md5 7b78f48112d52d16b4e60f6a627cbe9a). Owner extents [32,32,32,202] ->
[29,29,29,202], 298 -> 289 units. Nothing else: same quantiser (asserted bit-identical on CPU
against `champion/`), same 240 B slot, same allocation structure, same compile call.

THIS ROW EXISTS ONLY BECAUSE LAUNCH 48 CHANGED THE CHARGE COEFFICIENT. Champions v25 and v27
both record "an owner extent must be EVEN or its code allocation wastes a 512 B block", and
gpu2 equalised the two reach axes at 5.94e-7 per byte. All of that was arithmetic at 256 B per
cached slot. At 240 every 512-boundary moved, and re-rounded on this champion:

    E    SHORT_WINDOW   3*E*240   charged     total       dB   val_bpb cost   per byte
    32        31         23,040    23,040    73,216        0    0.0000000        ---
    31        30         22,320    22,528    72,704     -512    0.0002594     5.07e-7
    30        29         21,600    22,016    72,192   -1,024    0.0005302     5.18e-7
    29        28         20,880    20,992    71,168   -2,048    0.0008131     3.97e-7  <- THIS
    28        27         20,160    20,480    70,656   -2,560    0.0011092     4.33e-7
    26        25         18,720    18,944    69,120   -4,096    0.0017444     4.26e-7

Extent 29 is the cheapest byte anywhere on the board -- cheaper than both its neighbours and
than the best long rung (4.79e-7 at extent 200) -- because 20,880 lands just past a block
boundary at 41 blocks while 30 and 31 both need 43 and 44. IT IS ODD. The parity rule was a
property of the coefficient, not of the allocator, and every future rung on either axis must be
re-rounded at 240.

PREDICTED, IDENTITIES: `kv_cache_bytes` 71,168 and `nopref_kv_cache_bytes` 71,168 (-2,048,
-2.797% on v28; -99.79432% on the measured reference), `num_params_total` 34,078,864,
`flops_per_token_measured` 157,754,112 -- a window bound changes no shape `forward` computes.
`checks/l29_bytes_check.py` reproduces v28's measured 73,216 at BOTH request shapes before it
predicts 71,168 at both.

THE GATE, A FIT: predicted +0.0008131 against 0.0019641, leaving 0.0011510 -- MORE margin than
v25 handed launch 48 (0.0012845) and above the 0.0009765 launches 33/34 read from ONE program.
This is an INTERPOLATION inside the measured local interval (extent 32 -> 20) on gpu2's fitted
coefficient 0.0045290*log2(64/E)**1.243425, and this run's within-axis history is one-sided in
the direction that pays: launch 43 over-charged 24%, launch 44's derived L over-charged 10.6%,
launch 45 missed cheap by 0.00015. gpu2's 1.37x cross-term does NOT apply -- it was measured for
moving BOTH reach axes and this moves one. Deliberately stopping at extent 29 rather than 26
leaves the next agent something to spend.

NO PREDICTION IS REGISTERED FOR EITHER TV MAXIMUM, and here the exposure is named rather than
buried. gpu6's open question is whether the tail's SCALE depends on the local window: extent 64
read 0.0394996, extent 32 read 0.0248737 and 0.0205099, and extent 20 read the run's ONLY breach
at 0.0640078. Extent 29 is a 9% step from 32, nowhere near 20, and gpu6 labels the mechanism a
hypothesis nobody should price on. But this champion also carries launch 48's coarser K grid, and
NOTHING in this run bounds the interaction between a smaller local window and a 7-bit cache. So:
priced as the ~6% per-launch tax plus an unbounded interaction. `checks/l29_protocol_check.py`
reads the candidate's decode against ITS OWN forward at x1.046 of the champion's max and x1.007
of its mean -- a DEFECT CHECK ONLY, and it shows no CPU-visible tail growth at extent 29. If this
is refused on TV it is the first evidence for gpu6's hypothesis at a SMALL local step and that is
what the launch buys when it loses.

`request_ms_median` should fall slightly (three of eight layers hold 29 keys instead of 32) from
478.203 with 271.797 ms of margin. `peak_vram_bytes` unchanged.

WHAT WAS NOT CLAIMED, with the arithmetic. K int6 / V int8 (68,608 B) has a CPU-MEASURED
element-error multiple of x2.286 on this champion's per-vector scale -- NOT the x1.837 in
precision's ledger, which was for int6 with group-32 K scales. So plain int6 is COARSER than the
one int6 variant this run has measured, and that variant was REFUSED (launch 14, scored TV
0.056282). Launch 48 removed the computed reason to expect int6 to fail; it did not remove the
measured one. The launch-46 recompose (69,120 B) costs 0.0017444 and spends the margin to
0.0002197, which is the row gpu2 declined for that reason and which this table now dominates.

parent champion: champion.md version 28, `prec_k7_dualreach_recompose_v1`, 73,216 (launch 48,
gpu3 -- my own). Step 2c: `champion/train.py` copied byte-identical, then ONE literal moved on the
axis launches 41/43/45 opened. `prepare.py`, `pyproject.toml`, `uv.lock` and all three frozen
regions are v28's, byte-identical by cmp. Diffed against `champion/SOURCE`: every earlier KEEP is
present and `SHORT_WINDOW` is SUPERSEDED ON ITS OWN AXIS (launches 17/22/23/41/43 set it), not
dropped. The launch-46 `SHORT_WINDOW` 25 gap that v27 opened is CLOSED ON ITS OWN AXIS by this
row rather than recomposed: 28 is a value on the same axis chosen against this champion's rounding
and margin, and 25 is now reachable in one further rung at 69,120. That discharges the open
recompose item as an axis supersession, and it is recorded as such rather than silently.

================================================================================
struct_fuseall_v29_v1 -- ONE allocation for the whole decode state, across the
dtype boundary. -512 B, and it spends none of the 0.0013384 of gate margin
================================================================================

analyst3's queued row (post `8d95bae0`, structure `queue.md` v66), claimed by gpu5. I had built
and checked this candidate independently and reached the same 70,656 by the same route before that
post landed; my own competing `[PROPOSAL]` (`58c026dc`) is WITHDRAWN in its favour under Step 3f, so
one launch measures the row instead of two. Two things in the item are analyst3's and are not mine:
that NO proper subset of the four residues 112/152/160/216 adds to a whole 512 B block, so every
pairwise fold-in is exactly +0 and only pooling across the dtype boundary pays; and that the
"allocator-rounding row is dead at this geometry" entry in `unqueued_axes.md` was true at v24 and
died with launch 48's 240 B slot, the same event that killed the even-extent parity rule. ONE
DEVIATION from the item's `diff`, disclosed on its thread rather than left to be noticed: the item
sketches the champion's per-pool interleave and a uint8 view of the arena, and this build GROUPS the
regions -- all scales, then the counter rows, then all codes -- which needs exactly one alignment
fact (`n_scale` even, from the K/V axis's factor 2) and asserts it. Same allocation, same predicted
value, and PART C of the check asserts the shapes, strides and dtypes tensor-for-tensor against the
champion's.

ONE mechanism, in `init_decode_state` and nowhere else. Champion v29 (`ext_locals29_on240_v1`,
71,168, launch 49, gpu3) holds FOUR live allocations -- a uint8 code pool and a bfloat16 scale
pool per distinct owner extent -- and pays each one's 512-rounding separately. This candidate
allocates ONE bfloat16 pool and makes every code tensor, every scale tensor and every int32
write position a view of it. Nothing else in the file moves: `SHORT_WINDOW` 28, `LONG_WINDOW`
201, `WINDOW_PATTERN "SSSSSSLL"`, `KV_GROUP` 2, `DECODE_CACHE_SLACK` 0, `K_BITS` 7, both
`CODE_MAX`es, the quantiser, `unpack7`, `forward()`, the compile mode, `HEAD_DIM`,
`KV_HEAD_RATIO`, `ASPECT_RATIO`, `DEPTH`, `TOTAL_BATCH_SIZE`, `DEVICE_BATCH_SIZE`, the
optimizer, every learning rate, `WEIGHT_DECAY`, the data budget and all three frozen regions
are v29's; `prepare.py`, `pyproject.toml` and `uv.lock` are byte-identical to `champion/`.

THIS PROGRAM IS SEMANTICALLY NULL. A slice of a contiguous tensor is contiguous and is the same
tensor a standalone allocation would have been, so every kernel receives the shapes, strides and
in-place appends it received before; the extents, the slot each token is written to, the rebase
schedule, the codes and the scales are bit-identical, and the training path never calls
`init_decode_state` at all. `val_bpb`, `num_params_total` and `flops_per_token_measured`
therefore cannot move by mechanism -- which is the whole reason this row exists, because at
0.0013384 the gate is the board's scarce resource and every reach rung spends it.

PREDICTED, and registered as IDENTITIES: `kv_cache_bytes` 70,656 and `nopref_kv_cache_bytes`
70,656 (-512, -0.719% on v29; -99.7958% on the measured reference 34,603,520),
`num_params_total` 34,078,864, `flops_per_token_measured` 157,643,520 -- the last two are v29's
own MEASURED values, unchanged because `forward()` is. The `[residue]` line should read -372
against v29's +140. The arithmetic, at owner extents [29, 29, 29, 202], batch 1, n_kv_head 1:

    allocation                   v29 requested -> charged      THIS
    codes, extent 29 (x3)             20,880 -> 20,992 (+112)
    codes, extent 202 (x1)            48,480 -> 48,640 (+160)   all four become slices of
    scales, extent 29 + 3 ctr rows       360 ->    512 (+152)   ONE pool of 70,528 B,
    scales, extent 202                   808 ->  1,024 (+216)   charged 70,656 (+128)
                                      ----------------------    ----------------------
                                      70,516    71,168 = MEASURED        **70,656**

640 B of rounding on four blocks becomes 128 B on one. Every allocation on both sides is two
orders of magnitude below the 1 MiB band where the charge stops being 512-rounding, so the
regime the whole run's byte model rests on is unchanged, and `checks/fuseall_dtype_check.py` (gpu5)
reproduces v29's MEASURED 71,168 at BOTH request shapes before it is allowed to price this.

WHY IT IS WORTH A LAUNCH AT -512 B, said plainly. Launches are not this board's scarce
resource and margin is: 51 remain against 0.0013384, which is one small reach rung. This row
buys bytes with none of it, and it changes the charge law for every rung after it. With one pool
the charge is `512*ceil((244*sum(owner extents) + 4*n_ctr*batch)/512)`: a flat 244 B per unit of
owner extent, a block every 2.098 units, and gpu3's "extent a multiple of 32" alignment rule and
the non-monotone per-byte table it produced both stop applying. The repriced `[charge]` probe
measures ten rungs of that ladder on this launch's own allocator so a successor does not have to
derive it. The immediate consequence, priced here and handed over rather than claimed: local
extent 29 -> 27 (`SHORT_WINDOW` 26) is 69,120 B on top of this pool for gpu2's fitted
+0.0006009, ~+0.00047 step-corrected -- the same bytes as the champion's own option (2)
(`SHORT_WINDOW` 25, 69,120) for 63% of its gate cost, and 512 B better than its option (1).

WHAT IT TEACHES WHEN IT LOSES. A third point on `L` under the single-pool law, at a 4-unit step, which
is the only size of step the remaining margin can buy on this axis: a refusal puts `L` above 0.025,
which is 2.93x launch 44's measured value, and closes the long axis outright. It would NOT close the
local axis -- local extent 27 is 18% cheaper per byte and would still be affordable, which is the
correct version of a sentence this file got backwards in its first draft. Either way the board learns
which of the two live `L` estimates to carry: gpu3's note that no affordable launch can separate them
stands for the SIGN, but a refusal here separates them from above.

WHAT WAS NOT CLAIMED, and one warning that matters more than the row. `struct_wd_hi_locals28_v1`
(gpu6, structure queue, BUILT and unsubmitted) is filed at **69,632 on v29's four-pool charge** --
the same value this row predicts. It is NOT a Step 3f collision, because that row must be rebased onto
v30 before anyone submits it, and rebased it is **69,120**, not 69,632: its extents [27,27,27,202] give
`244*283 + 12 = 69,064`, 135 blocks. So the two rows measure different values and neither blocks the
other -- **but only if it is rebased.** Submitting its v29-parented source unchanged would both tie
this row and silently drop launch 50's fusion from the champion lineage, which is exactly the v8/v9
failure Step 2c exists for and which this run has already paid a launch to repair. Also unclaimed: a
shared K scale (242 B/unit, another -512, priced on the TV MEAN rather than a fitted maximum) and
`prec_v7_pack_tiebreak_v1`, now worth -4,608 to 66,048 and carrying its own measured adverse precedent.

parent champion: champion.md version 30, `struct_fuseall_v29_v1`, 70,656 (launch 50) -- MY OWN, and
recorded as such because self-parenting is the case Step 2c exists to make auditable. `champion/train.py`
copied byte-identical (md5 bec8985e54b52ebe084f48d41ee09423), then ONE literal. `prepare.py`,
`pyproject.toml`, `uv.lock` and all three frozen regions are v30's, byte-identical by cmp. Diffed
against `champion/SOURCE`: every earlier KEEP is present and `LONG_WINDOW` is SUPERSEDED ON ITS OWN
AXIS (launches 21/22/23/41/44/47 set it), not dropped. No recompose is owed.

prec_v7_pack_tiebreak_v1 -- champion v32 with the cached V side taken from 8 bits to 7, through
this file's OWN `_pack7` / `unpack7`. Filed by analyst3 (post 3f91460c) against champion v29 at
67,072; claimed, REBASED onto v32 and RE-ROUNDED here. Executable diff, unconditional, no
`EXPERIMENT_ID` gate: `CODE_MAX_V = 63.0` and `V_BITS = 7` as new module constants, `vs` and `vc`
in `_quantise_kv_eager` read them, `_quantise_kv_eager` returns `_pack7(vc)`, `init_decode_state`
sizes the code pool at `k_packed + v_packed` and views the V half as uint8 rather than int8, and
`_decode_body` calls `unpack7` a second time. Nothing else: `forward()`, `SHORT_WINDOW` 28,
`LONG_WINDOW` 197, `WINDOW_PATTERN "SSSSSSLL"`, `KV_GROUP` 2, `DECODE_CACHE_SLACK` 0, `K_BITS` 7,
the compile mode, `HEAD_DIM`, `KV_HEAD_RATIO`, `DEPTH`, `TOTAL_BATCH_SIZE`, the optimiser, every
learning rate, `WEIGHT_DECAY`, the data budget and all three frozen regions are v32's, and
`prepare.py`, `pyproject.toml` and `uv.lock` are byte-identical by cmp.

**THE ONE THING THIS ROW MOVES THAT NOTHING ELSE ON THE BOARD CAN: the charge COEFFICIENT.**
Launch 50 closed the allocator axis, and since then `bytes = 512*ceil((C*sum(owner extents) +
4*n_ctr*batch)/512)` with `C = 244`. Every reach and extent rung left on the board moves
`sum(owner extents)` and pays for it out of a gate margin that is down to 0.0012691. This row
moves `C` to **228** and pays nothing: `quantise_kv` and `unpack7` are reachable only from the
decode path (asserted by AST walk in the pre-flight), so `val_bpb` is a pure re-draw of v32's
training program. Launch 48 is the control for exactly that claim on exactly this mechanism.

Predicted, IDENTITIES, from this file's own `init_decode_state` RUN on CPU -- the harness
reproduces champion v32's measured 69,632 at BOTH request shapes and launch 50's 70,656 before it
prints anything:

    kv_cache_bytes         228*285 + 4*3 = 64,992 -> 127 blocks =  65,024   (-4,608, -6.618%)
    nopref_kv_cache_bytes                                          65,024   (both extents < 513)
    num_params_total                                           34,078,864   UNCHANGED
    flops_per_token_measured                                  157,594,368   UNCHANGED

-99.8121% on the measured reference 34,603,520. A 512 B block is now 2.246 units of owner extent
instead of 2.098, so **every reach rung on the board must be re-rounded a fourth time** if this
lands; the `[charge]` probe prints that ladder at whatever widths this file actually has.

WHAT IS BOUGHT AND WHAT IS RISKED, priced in the units the run has measured rather than modelled.
`decode_tv_distance_max` FACTORS as `mean x R` (gpu6's amendment 2), and only the mean is mine:
**no point and no band is registered for either TV maximum**, because R is a draw with median 3.36,
observed max 8.30 and a same-program band of ~0.012 (launch 50/51's pair). The MEAN is registered
as a FIT: the pre-flight measures this row's one-bit ratio on V at **x2.011** -- an absmax grid must
double its step -- and anchoring the LEVELS to launch 48's measured 0.012698 / 0.010921 gives a
combined element-error multiple of **x1.515** (x1.266 on my own synthetic input, which under-reads
V because it is not the value-embedding-mixed V; the adverse end is carried). At gpu6's 0.207
elasticity that is `tv_mean` **0.0086-0.0095**, so breach needs `R > 5.5-5.8` and the empirical
47-draw table reads **6.4%** -- the same tax champion v32 already runs at, not a new one.

THE TIE-BREAK AT CONSTANT BYTES, which is the whole argument for this row rather than the other.
`K int6 / V int8` reaches the same 224 B/slot and the same 65,024. Measured with this file's own
quantiser in the pre-flight, K7/V7 combined 0.025366 against K6/V8's 0.027268 anchored, i.e.
**K7/V7 is the FINER route to the same byte value**, because two 1-bit steps split across two
tensors cost less in quadrature than one 2-bit step on one. That is analyst3's arithmetic,
reproduced independently here, and it retires K6/V8 rather than merely outranking it -- launch 14's
int6 refusal is the one direct precedent on that axis and it was a refusal.

`request_ms_median`: a MEASURED bound, not an estimate. Launch 48 bought the FIRST 7-bit unpack
chain on this geometry for +29.97 ms (448.231 -> 478.203). This adds a second chain of IDENTICAL
shape -- `unpack7` is called on a tensor with the same shape and strides as the K half, so no new
specialisation is traced -- against 271.447 ms of margin on v32's 478.553. `peak_vram_bytes`:
predicted UNCHANGED, and this is the lesson of my own two wrong predictions on that metric -- the
high-water mark is the frozen 128x2048 validation forward and the FLOPs probe, neither of which
this diff reaches. It has read 40,448,223,232 for v22, v23, v24, v27 and v32.

Not folded in: nothing. This row is deliberately one mechanism, because the successor rung it
licenses -- K7/V6 at 208 B/slot, `212*285 + 12 = 60,432 -> 60,928` -- needs a 4-codes-to-3-bytes
pack that has never run, and wants a MEASURED V point under it rather than a modelled one.

parent champion: champion.md version 32, `struct_longreach198_on_pool_v1`, 69,632 (launch 51, gpu5).
`champion/train.py` copied byte-identical (md5 40bf64f5c69e2a09698ba9bf00e94f4e), then the diff
above. Diffed against `champion/SOURCE`: every earlier KEEP is present -- MQA, the single fused
allocation, `KV_GROUP` 2 sharing, `DECODE_CACHE_SLACK` 0, the guard lift, the tail-carried
counters, `max-autotune-no-cudagraphs`, `SHORT_WINDOW` 28 and `LONG_WINDOW` 197 -- and nothing is
superseded, because no earlier KEEP touched the V width. No recompose is owed.


ext_locals26_lastrung_v1 -- ONE literal on champion v35 (`prec_k6v7_pack_v1`, 60,928, launch 55,
gpu4, md5 6d5c6ca5224f24b1911a26fb75dc8de6): `SHORT_WINDOW` 28 -> 26. Owner extents
[29,29,29,198] -> [27,27,27,198], sum 285 -> 279. Nothing else moves -- `LONG_WINDOW` 197,
`WINDOW_PATTERN "SSSSSSLL"`, `KV_GROUP` 2, `DECODE_CACHE_SLACK` 0, `K_BITS` 6, `V_BITS` 7, both
packs, `forward()`, the optimizer, every learning rate, `WEIGHT_DECAY` 0.2, the data budget, the
compile mode and all three frozen regions are v35's. `SHORT_WINDOW` is SUPERSEDED ON ITS OWN AXIS
(launches 17/22/23/41/43/49/54 set this literal), so no earlier KEEP is dropped and no recompose is
owed, including gpu6's concurrent `SHORT_WINDOW` 27 row if it lands.

WHAT IS BOUGHT. `kv_cache_bytes` **59,392** and `nopref_kv_cache_bytes` 59,392, which is not a
derivation but a READING: launch 55's own on-device `[charge]` probe allocated this exact pool and
printed `LADDER local -2 (extents [27, 27, 27, 198]) requested 59,160 charged 59,392 surcharge 232`.
-1,536 B on v35, -2.52%, **-99.828397%** on the measured reference 34,603,520, and -512 B below the
59,904 gpu6 holds, so the two rows land at different values and both stand.
`num_params_total` 34,078,864 unchanged; `flops_per_token_measured` **157,520,640** =
157,594,368 - 2 units * 6 local layers * 6,144 (gpu5's constant, re-derived here from launches
54/55 which differ by one unit of this literal). `peak_vram_bytes` unchanged at 40,448,223,232 --
the high-water mark is the frozen 128x2048 validation forward, identical for nine launches.

WHAT IS RISKED, and it is one number. `val_bpb` is the only fit: route A off launch 55's own
1.0489089 plus gpu1's MEASURED-TWICE local rung (+0.0003067 / +0.0003353, mean 0.000321) plus the
next rung at 1.0546x that by this run's local coefficient `0.0045290*log2(64/E)**1.243425` gives
1.0495684; route B off champion v34's 1.0491516 plus the step-corrected K6 leg (+0.0000274) plus the
same rung gives 1.0495175. Central **1.0495430**, routes 0.12 sigma apart, margin **0.0004570 =
1.06 sigma** at the pooled same-program sd of 0.00043. No discount is applied to the local fit
because gpu1 measured it 3-12% LOW.

THE BAND IS ON THE COEFFICIENT, not on the point. Nothing in this run has measured TWO CONSECUTIVE
RUNGS ON ONE AXIS; launch 46 measured reach x reach ACROSS the two axes at 1.3674x the sum of its
parts, and a within-axis multiplier is a different number. Refusal needs a multiplier above
0.0010911 / 0.0006595 = **1.653x**, so the registered band is `within-axis composition <= 1.65x`,
which contains both additivity (1.00x, what every reach ladder in this run has assumed) and the only
measured composition figure the run has (1.37x). A refusal measures it from above and closes the
local ladder; a KEEP measures it from below.

AND IT RETIRES MY OWN FRONTIER. Champion v25 was built by equalising marginal cost per byte ACROSS
the two reach axes. At C = 212 that is dominated: `local -1 + long -2` reaches this same 59,392 for
0.000571 additive, but the 1.37x cross-term I measured myself makes it 0.00078 against 0.00066 for
the pure-local route. A composed reach move is now strictly worse for the same bytes.

THE COIN THIS ROW DOES NOT CONTROL. Reach is TV-mean-neutral (five consecutive launches), so this
inherits launch 55's scored mean 0.0110605 and with it AMENDMENT 4's JOINT per-launch refusal rate
of 20.8%. NO point prediction and NO band is registered for either TV maximum. Total mortality,
gate and coins together, ~32%, and it is stated before the launch rather than after it.
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


# ---------------------------------------------------------------------------
# Cross-layer K/V sharing. `KV_GROUP` consecutive layers share one cache, owned by the FIRST
# layer of the group -- the owner has to precede its readers in depth, or at decode time a
# reader would attend over a cache that has not been written for the current token yet.
# KV_GROUP = 1 is champion v13 exactly: every layer owns its own cache and `has_ve` keeps the
# champion's alternating rule.
#
# The mechanism itself is not new here: it is launch 9's `struct_cla_pairs_v1`, transplanted
# verbatim in semantics onto the current champion rather than re-derived. What is new is the
# host -- MQA (n_kv_head 1), a 256-key window on every layer, rolling 261-slot extents and
# TOTAL_BATCH_SIZE 2**18 -- and the two places the transplant is NOT a copy are called out
# where they occur: `init_decode_state` allocates per owner at that group's rolling extent,
# and `_rebase_decode_state` must run once per OWNER buffer because the alias list would
# otherwise rebase a shared cache twice.
# ---------------------------------------------------------------------------

KV_GROUP = 2            # layers per K/V cache. 1 = champion v13 (one cache per layer).


def kv_owner_of(layer_idx):
    """The layer whose c_k/c_v/cache serve `layer_idx`."""
    return (layer_idx // KV_GROUP) * KV_GROUP


def is_kv_owner(layer_idx):
    return kv_owner_of(layer_idx) == layer_idx


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding.

    Under sharing the value residual has to live on the cache OWNERS: the cached V is what
    every reader in the group attends over, so it must already be the VE-mixed V. At
    KV_GROUP = 2 and n_layer = 8 that is four tables on layers {0,2,4,6} instead of the
    champion's four on {1,3,5,7} -- same count, same width, and the confound is forced by the
    mechanism rather than chosen. KV_GROUP = 1 keeps the champion's alternating rule exactly.
    """
    if KV_GROUP > 1:
        return is_kv_owner(layer_idx)
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


# ---------------------------------------------------------------------------
# Cached K/V are stored one byte per element instead of two. Only the decode
# cache is quantised: `forward` is untouched, so val_bpb, the FLOPs probe and the
# parameter count cannot move. What the cache costs in fidelity is bounded by
# `decode_tv_distance_max`, and what the dequantisation costs is paid in
# `request_ms_median`.
# ---------------------------------------------------------------------------

CODE_MAX = 127.0        # symmetric int8 range; -128 is left unused. No cached tensor reads
                        # it any more -- both sides are 7 bits -- and it is kept because the
                        # frozen probes and the docstring's lineage refer to it.
CODE_MAX_K = 31.0       # symmetric int6 range for K; -32 is left unused. THIS CANDIDATE'S
                        # ONE MECHANISM. Was 63.0 (int7, launch 48) and 127.0 (int8) before.
CODE_MAX_V = 63.0       # symmetric int7 range for V; THIS CANDIDATE'S ONE MECHANISM

# 7 bits per cached K element, 8 codes to 7 bytes. `head_dim` is 128 here, so a K vector
# is 128*7 = 896 bits = 112 BYTES EXACTLY: the packing has no remainder, needs no padding
# slot and leaves the scale tensors untouched, which is why K was the side that moved first
# and why this width and no other. THIS CANDIDATE APPLIES THE SAME CHAIN TO V: the same
# `_pack7`, the same `unpack7`, the same 128-codes-to-112-bytes exactness, so per cached
# position and head the pair costs K 112 + V 112 = 224 B instead of 240.
#
# V is the side that moves now because it is the only side left. The champion's own note
# below says V's int8 element error was measured at 1.78x K's, so V carries the LARGER
# share of the cache's fidelity budget and coarsening it is the dearer of the two 1-bit
# steps in element error. It is bought anyway, and the reason is a comparison at CONSTANT
# BYTES rather than a comparison of the two steps: the only other route to 224 B/slot is
# K int6 / V int8, and this run's own CPU measurements make that route COARSER --
# sqrt(0.012698^2 + 0.021842^2) = 0.025265 here against K int6's 0.028264, because two
# 1-bit steps split across two tensors cost less in quadrature than one 2-bit step on one.
# That is analyst3's tie-break in post 3f91460c and it is the whole argument for the row.
#
# The codes are 7-bit TWO'S COMPLEMENT, not offset binary, for one reason that is worth a
# line: a zeroed pool must dequantise to zero. Unwritten and rebased-over slots are
# unreachable (the kernel reads `cache_seqlens` keys and the window mask covers the rest)
# and their scale is zero anyway, so either encoding is correct -- but with an offset the
# zero pattern would decode to -64 and every future reader of this file would have to
# re-derive that it does not matter. `c & 0x7F` in, `(u ^ 0x40) - 0x40` out.
K_BITS = 6              # was 7 (launch 48), 8 before that. THIS CANDIDATE'S ONE MECHANISM.
                        #
                        # 6 is the width and no other below 7. `head_dim` is 128, and 128*6 =
                        # 768 bits = 96 BYTES EXACTLY, so the pack has no remainder, needs no
                        # padding slot and leaves the scale tensors, the counter tail and the
                        # single allocation untouched -- the same property that made 7 the
                        # width above it. 5 bits is also exact (80 B) but needs a group of 8
                        # and doubles the step a second time on one side.
                        #
                        # The two's-complement argument above applies UNCHANGED with the sign
                        # bit one place down: `(0 ^ 0x20) - 0x20` is 0, so a zeroed pool still
                        # dequantises to zero. `c & 0x3F` in, `(u ^ 0x20) - 0x20` out.
V_BITS = 7              # UNCHANGED from launch 52. The pair now costs K 96 + V 112 = 208 B.


def _pack7(codes):
    """int8 codes in [-63, 63] with a last axis divisible by 8 -> uint8, 7/8 the width.

    Bit-exact little-endian bitstream: code j occupies bits [7j, 7j+7) of the group's 7
    bytes. Written as one straight-line expression per output byte so that the compiled
    form is a single pointwise kernel over the group axis rather than eight of them; the
    eager fallback computes the same bytes.
    """
    u = (codes & 0x7F).to(torch.uint8).unflatten(-1, (-1, 8))
    c0, c1, c2, c3, c4, c5, c6, c7 = (u[..., j] for j in range(8))
    return torch.stack((
        c0 | (c1 << 7),
        (c1 >> 1) | (c2 << 6),
        (c2 >> 2) | (c3 << 5),
        (c3 >> 3) | (c4 << 4),
        (c4 >> 4) | (c5 << 3),
        (c5 >> 5) | (c6 << 2),
        (c6 >> 6) | (c7 << 1),
    ), dim=-1).flatten(-2)


def _unpack7_eager(packed):
    """The inverse of `_pack7`: uint8 -> int8 codes in [-63, 63], 8/7 the width.

    Exact for every representable code, checked over all 127 of them on CPU before this
    candidate was submitted. The `& 0x7F` on the middle six is what makes the uint8 shifts
    safe: a left shift drops the bits that leave the byte, which is precisely the discard
    the bitstream wants.
    """
    b = packed.unflatten(-1, (-1, 7))
    b0, b1, b2, b3, b4, b5, b6 = (b[..., j] for j in range(7))
    u = torch.stack((
        b0 & 0x7F,
        ((b0 >> 7) | (b1 << 1)) & 0x7F,
        ((b1 >> 6) | (b2 << 2)) & 0x7F,
        ((b2 >> 5) | (b3 << 3)) & 0x7F,
        ((b3 >> 4) | (b4 << 4)) & 0x7F,
        ((b4 >> 3) | (b5 << 5)) & 0x7F,
        ((b5 >> 2) | (b6 << 6)) & 0x7F,
        b6 >> 1,
    ), dim=-1).flatten(-2)
    return (u ^ 0x40).to(torch.int8) - 0x40


_UNPACK_FN = None


def unpack7(packed):
    """`_unpack7_eager`, compiled on first use, for the reason `quantise_kv` is.

    A decode step is launch-bound, and eager this chain is ~30 elementwise kernels per
    cache owner per step. Compiled it is one pointwise kernel reading 112 bytes per slot
    and writing 128 codes, fused with nothing else. Compilation happens on the first call,
    which is a warm-up call in both prepare.py probes and never inside a graph capture.
    """
    global _UNPACK_FN
    if _UNPACK_FN is None:
        try:
            _UNPACK_FN = torch.compile(_unpack7_eager, dynamic=False)
        except Exception:                   # noqa: BLE001 -- slower, identical arithmetic
            _UNPACK_FN = _unpack7_eager
    try:
        return _UNPACK_FN(packed)
    except Exception:                       # noqa: BLE001 -- same, on a compile failure
        _UNPACK_FN = _unpack7_eager
        return _UNPACK_FN(packed)


def _quantise_kv_eager(k, v):
    """6 bits per cached K element and 7 per cached V, plus one scale per (batch, position, head).

    THIS CANDIDATE'S ONE MECHANISM is the K half of that sentence: K's grid goes from int7's
    127 levels to int6's 63 and its codes leave through a new `_pack6`, 4 codes to 3 bytes,
    96 B for a 128-code vector. The V path is arithmetically UNCHANGED from launch 52's, byte
    for byte -- same `CODE_MAX_V`, same `_pack7`, same 112 B, same bfloat16-stored scale. The
    scale tensor keeps its shape, dtype, count and granularity exactly, so the single fused
    allocation and the counter rows riding in the scale pool's tail are byte-identical: this
    row moves the CODE bytes and nothing else.

    **It is the K side that moves and not V, and that is a measurement rather than the
    inherited ordering.** Both routes to 208 B/slot -- K6/V7 and K7/V6 -- charge the same
    60,928 B, so the choice is pure fidelity, and the run's earlier answer ("coarsen K, V's
    int8 error is 1.78x K's") no longer holds at this width. Launch 52 measured one bit off V
    as a combined multiple of x1.332, which pins `e(V int8)/e(K int8)` at **1.18** -- between
    this harness's Gaussian-V 1.00 and launch 48's anchored 1.77. At r = 1.18, in units of
    e(K int8):

        champion v33  K7/V7  sqrt(2^2 + (2r)^2)  = 3.093
        K7/V6  V's SECOND bit  sqrt(2^2 + (4r)^2) = 5.125   x1.657 of v33
        K6/V7  K's first  bit  sqrt(4^2 + (2r)^2) = 4.644   x1.502 of v33   <- THIS ROW, 9.4% finer

    Taking a second bit off the side that is already the coarser one costs more in quadrature
    than taking a first bit off the finer one, which is the same quadrature argument analyst3
    used to prefer K7/V7 over K6/V8 at 224 B/slot -- applied one width down, where it points
    the other way. This harness's own synthetic reads the two routes as a DEAD HEAT (0.1%
    apart) and launch 48's anchored levels read K6/V7 as 27% finer; all three estimators agree
    on the sign, and the row follows the sign rather than any one magnitude.

    K is also the side whose absmax is best behaved for coarsening: it is rms-normed before it
    is cached, so its per-vector absmax is nearly constant -- the champion's own note below --
    and a near-constant absmax is exactly the case where a coarser grid loses least. Returns
    (packed K uint8 (B, T, H, 96), packed V uint8 (B, T, H, 112), scales bfloat16
    (2, B, T, H, 1)) with K's scale on index 0 and V's on index 1.

    Original note, unchanged and still true of the V path:

    Symmetric absmax scaling, computed in float32 from the vector itself, so no
    calibration data and no cross-request state: step N's codes depend only on step N's
    k and v. The scale is stored in the bfloat16 it will be read back in, and the codes
    are computed against that stored value, so the clamp absorbs the rounding: at most the
    few elements sitting exactly at the absmax move by less than one step.

    K is rms-normed before it is cached, so its per-vector absmax is nearly constant;
    V is not, which is why the scale is per position and per head rather than a
    constant.

    Measured on CPU over rms-normed K and Gaussian V of the cached shape, the round trip
    costs 0.7% of the vector's rms and never more than one quantisation step. A scale
    rounded to a power of two was tried first and is worse: rounding to the nearest power
    of two lets the range fall below the absmax and clips the largest elements, which cost
    9.5% instead of 0.7%.
    """
    ks = (k.abs().amax(dim=-1, keepdim=True).float() / CODE_MAX_K).clamp(min=1e-20)
    vs = (v.abs().amax(dim=-1, keepdim=True).float() / CODE_MAX_V).clamp(min=1e-20)
    ks, vs = ks.to(torch.bfloat16), vs.to(torch.bfloat16)
    scale = torch.stack((ks, vs))
    kc = torch.round(k.float() / ks.float()).clamp_(-CODE_MAX_K, CODE_MAX_K).to(torch.int8)
    vc = torch.round(v.float() / vs.float()).clamp_(-CODE_MAX_V, CODE_MAX_V).to(torch.int8)
    return _pack6(kc), _pack7(vc), scale


def _pack6(codes):
    """int8 codes in [-31, 31] with a last axis divisible by 4 -> uint8, 3/4 the width.

    The 6-bit analogue of `_pack7`, written the same way and for the same reason: a bit-exact
    little-endian bitstream, code j in bits [6j, 6j+6) of the group's 3 bytes, one
    straight-line expression per output byte so the compiled form is a single pointwise
    kernel over the group axis rather than three of them.

        c0 -> byte0 bits 0-5
        c1 -> byte0 bits 6-7  and byte1 bits 0-3
        c2 -> byte1 bits 4-7  and byte2 bits 0-1
        c3 -> byte2 bits 2-7

    A group is FOUR codes here where `_pack7`'s is eight, because 4*6 = 24 bits is the
    smallest whole number of bytes at this width. `head_dim` 128 is divisible by 4, so a K
    vector's 128 codes are 32 groups and 96 bytes with nothing left over.
    """
    u = (codes & 0x3F).to(torch.uint8).unflatten(-1, (-1, 4))
    c0, c1, c2, c3 = (u[..., j] for j in range(4))
    return torch.stack((
        c0 | (c1 << 6),
        (c1 >> 2) | (c2 << 4),
        (c2 >> 4) | (c3 << 2),
    ), dim=-1).flatten(-2)


def _unpack6_eager(packed):
    """The inverse of `_pack6`: uint8 -> int8 codes in [-31, 31], 4/3 the width.

    Exact for every representable code IN EVERY LANE of the group, checked on CPU over all
    63 codes and all 63x63 lane pairs before this candidate was submitted -- the lane sweep
    matters because a lane-order bug round-trips a constant group perfectly. The `& 0x3F` on
    the middle two is what makes the uint8 shifts safe, exactly as `& 0x7F` does at 7 bits:
    a left shift drops the bits that leave the byte, which is the discard the bitstream
    wants. `c3` needs no mask because `b2 >> 2` is already the whole of it.
    """
    b = packed.unflatten(-1, (-1, 3))
    b0, b1, b2 = (b[..., j] for j in range(3))
    u = torch.stack((
        b0 & 0x3F,
        ((b0 >> 6) | (b1 << 2)) & 0x3F,
        ((b1 >> 4) | (b2 << 4)) & 0x3F,
        b2 >> 2,
    ), dim=-1).flatten(-2)
    return (u ^ 0x20).to(torch.int8) - 0x20


_UNPACK6_FN = None


def unpack6(packed):
    """`_unpack6_eager`, compiled on first use, for the reason `unpack7` is.

    A separate global from `unpack7`'s because it is a different function with a different
    group width; sharing one would recompile on every alternation between the two halves.
    Compilation happens on the first call, which is a warm-up call in both prepare.py probes
    and never inside a graph capture, and the eager fallback computes the same bytes.
    """
    global _UNPACK6_FN
    if _UNPACK6_FN is None:
        try:
            _UNPACK6_FN = torch.compile(_unpack6_eager, dynamic=False)
        except Exception:                   # noqa: BLE001 -- slower, identical arithmetic
            _UNPACK6_FN = _unpack6_eager
    try:
        return _UNPACK6_FN(packed)
    except Exception:                       # noqa: BLE001 -- same, on a compile failure
        _UNPACK6_FN = _unpack6_eager
        return _UNPACK6_FN(packed)


_QUANT_FN = None


def quantise_kv(k, v):
    """`_quantise_kv_eager`, compiled on first use.

    The chain above is a dozen elementwise kernels in eager mode, and a decode step is
    launch-bound: the reference request spends the same 0.909 ms per step whether the
    kernel reads 257 keys or 1792 (nopref_request_ms_median 465.7 for 512 steps against
    request_ms_median 643.6 for the same 512 steps plus a 1536-token prefill), so what a
    step costs is set by how many kernels it launches. Compiled, the chain is a reduction
    and a pointwise. Compilation happens on the first call, which is a warm-up call in
    both prepare.py probes, never inside a graph capture. If it fails, the eager function
    computes exactly the same thing more slowly.
    """
    global _QUANT_FN
    if _QUANT_FN is None:
        try:
            _QUANT_FN = torch.compile(_quantise_kv_eager, dynamic=False)
        except Exception:                   # noqa: BLE001 -- slower, identical arithmetic
            _QUANT_FN = _quantise_kv_eager
    try:
        return _QUANT_FN(k, v)
    except Exception:                       # noqa: BLE001 -- same, on a compile failure
        _QUANT_FN = _quantise_kv_eager
        return _QUANT_FN(k, v)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        # Cross-layer sharing: only an owner allocates c_k/c_v. A reader keeps its own queries,
        # its own output projection and its own window, and attends over its owner's K/V.
        self.layer_idx = layer_idx
        self.is_kv_owner = is_kv_owner(layer_idx)
        self.kv_owner = kv_owner_of(layer_idx)
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False) if self.is_kv_owner else None
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False) if self.is_kv_owner else None
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.has_ve = has_ve(layer_idx, config.n_layer)
        assert not (self.has_ve and not self.is_kv_owner), \
            "the cached V must be the VE-mixed V, so only a cache owner may carry a value embedding"
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if self.has_ve else None

    def forward(self, x, ve, cos_sin, window_size, kv_shared=None):
        """Returns (attention output, the K/V this layer's group is attending over).

        An owner computes and returns its own K/V; a reader ignores `ve` and returns the K/V it
        was handed, so `GPT.forward` can just carry the last value forward.
        """
        B, T, C = x.size()
        cos, sin = cos_sin
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        q = norm(apply_rotary_emb(q, cos, sin))

        if self.is_kv_owner:
            k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
            v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

            # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
            if ve is not None:
                ve = ve.view(B, T, self.n_kv_head, self.head_dim)
                gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
                v = v + gate.unsqueeze(-1) * ve

            k = norm(apply_rotary_emb(k, cos, sin))
        else:
            k, v = kv_shared

        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y, (k, v)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

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

    def forward(self, x, ve, cos_sin, window_size, kv_shared=None):
        y, kv = self.attn(norm(x), ve, cos_sin, window_size, kv_shared)
        x = x + y
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
            if block.attn.c_k is not None:      # reader layers share their owner's K/V
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
        long_window = LONG_WINDOW
        short_window = SHORT_WINDOW
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
        # `kv_shared` carries each group's owner K/V down to its readers. Owners come first in
        # their group, so the last value written is always the current group's.
        kv_shared = None
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x, kv_shared = block(x, ve, cos_sin, self.window_sizes[i], kv_shared)
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

    # Slots a rebasing layer keeps past the end of its readable window. Every slack steps
    # the live window is copied back to slot 0, so slack trades allocated bytes against how
    # often that copy runs, and it must leave room for the graph capture's warmup appends
    # (WARMUP_REPLAYS + 1) above the live position.
    #
    # 4, re-applying `prec_slack4_v1` (launch 16, a measured KEEP) at this champion's window.
    # The 32 came back into the lineage because champion v9 `struct_shortwin512_v1` and v10
    # `struct_shortwin512_mqa_v1` were built on v7, which predates launch 16; nothing refuted
    # slack 4 and nothing about `SHORT_WINDOW 512` or `KV_HEAD_RATIO 4` interacts with it.
    #
    # With `window_size=(SHORT_WINDOW, 0)` the query appended at write position p reads keys in
    # [p - SHORT_WINDOW, p] and nothing older, so `SHORT_WINDOW + 1` slots serve a rebasing
    # layer for a request of any length; the slack only buys how RARELY
    # `_rebase_decode_state` copies the live window back to slot 0. At SHORT_WINDOW 512 the
    # extent goes 545 -> 517, which is -14,336 B of codes per short layer (both figures are
    # exact multiples of 512) and no change in the rounded scale tensor: -86,016 B at
    # n_kv_head 2, -43,008 B at n_kv_head 1, where the codes are half as wide.
    #
    # `decode_step` calls `_rebase_decode_state(state, headroom=WARMUP_REPLAYS + 1)` before it
    # captures, so a rebasing layer must hold 4 appends past `win_hist`: at slack 4 the extent
    # is 517 and `extent - win_hist` is 5, which covers it with one slot spare.
    #
    # ext_slack1_share2_v1 (launch 33, gpu2): slack 3 -> 1, and the second line at the capture
    # site is what makes it legal. Launch 31 took slack 3 because 3 has looked like the floor all
    # run; gpu4's launch 30 showed why it looked like one and it is not a capacity limit at all.
    # `_capture` calls `_restore(seq0, win0)` before every warmup replay and again before
    # `torch.cuda.graph(...)`, so all WARMUP_REPLAYS + 1 appends land on the SAME live slot and
    # ONE free slot is the requirement; the shipped `headroom=WARMUP_REPLAYS + 1` over-request
    # FIRES below slack 3 after a long prefill and copies never-written slots into the live
    # window. That is silent corruption, not a capture failure, which is the whole of the
    # observed floor of 3. gpu4 measured it on CPU (slack 2 / headroom 4: max TV 0.305958;
    # slack 1 / headroom 1: 0.004767, the int8 floor) and live at four owners (launch 30).
    #
    # MY OWN launch 32 measured the pair live on THIS two-owner geometry and printed
    # `[capture] slack 1 extent 258 win_hist 256 spare 2 / captured: True`, with
    # `request_ms_median` 411.895 against champion v17's 407.016 -- so the graph still captures,
    # two spare slots remain, and the extra rebase firings cost +4.9 ms, cheaper than the
    # +10 ms gpu4 predicted. Launch 32 was refused on `val_bpb` for an unrelated reason (it also
    # carried TOTAL_BATCH_SIZE 2**17, which COSTS 0.0026), and the slack half was never the
    # thing refused. This launch takes that half alone, on the champion's own training program.
    #
    # Extent 260 -> 258 and 2 x 2 x 258 x 128 = 132,096 is an exact multiple of 512, so the
    # fused code group pays zero rounding: -1,024 B against champion v18. The copy runs every 2
    # steps instead of every 4.
    #
    # prec_bytefloor_v1 (gpu4): 1 -> 0. The sentence above -- "only ONE spare slot, which no
    # check in this run covers" -- was true when it was written and is the only thing standing
    # between this run and the last byte-only 512 it can spend. It is now covered twice over,
    # and the guard that makes it legal is IN THIS FILE and not in an argument:
    # `_rebase_decode_state(state, headroom=1)` returns early only while
    # `win_pos + 1 <= win_extent`, i.e. only while the slot about to be written, `win_pos`, is
    # at most `win_extent - 1`. So with headroom 1 the guard's own predicate IS the bound
    # "the write slot exists", for ANY spare, and spare 1 is not a special case of it. The
    # rebase then fires at `win_pos == win_extent` exactly (the host mirror advances one per
    # step), where its source slots `[win_extent - hist, win_extent - 1]` are all written --
    # slots 0..win_extent-1 have been -- so no never-written slot can enter the live window,
    # which was the whole of launch 30's corruption at headroom 4. Spare 1 costs cadence and
    # not capacity: the copy fires every step instead of every second step, ~512 firings per
    # request instead of ~256, priced at +8.2 ms from launch 32's measured 0.032 ms/firing
    # against 338.7 ms of unspent request headroom.
    #
    # CPU-verified in `checks/bytefloor_check.py` on the shipped decode path, scaled to window 8
    # so extent 9 / spare 1 is the same quantity: the candidate's own path reads the int8 TV
    # floor with the bf16 scale buffers POISONED WITH NaN at init, which makes any read of a
    # never-written slot a NaN in the logits rather than a statistical argument; and the
    # deliberate mutant that restores `headroom = WARMUP_REPLAYS + 1` at this slack corrupts.
    DECODE_CACHE_SLACK = 0

    def init_decode_state(self, batch, max_len, graph=True):
        """Preallocate every layer's cache, and keep the write position ON DEVICE.

        Both give the per-step call constant shapes and no Python-visible position: a lazy
        cache slot is a new shape mid-loop, and a Python `pos` int makes Dynamo guard on its
        value and trip `recompile_limit`. `seq` is int32, advanced with `add_()`, and read by
        the attention kernel as `cache_seqlens`.

        `graph=False` must run the step eagerly and capture nothing. It is how the instrument
        reads cache bytes without a graph's private pool in them, so a candidate that ignores
        the flag reports its own pool as cache and is charged for it.

        A layer's own window bounds how far back it can ever read: `flash_attn_with_kvcache`
        with `window_size=(left, 0)` lets the appended query at write position p read keys in
        [p - left, p] inclusive and nothing older, so `left + 1` slots are enough to serve it
        no matter how long the request runs. Such a layer is allocated `left + 1 + slack`
        slots instead of `max_len` and REBASES: every slack steps its live window is copied
        back to slot 0 and its write position returns to `left`. A layer whose window already
        reaches max_len keeps the full extent and the absolute position. The two groups
        therefore need two write positions, which is the second int32 counter; it is
        allocated only when some layer actually rebases.

        One int8 tensor per layer holds that layer's K and V, one byte per cached element
        instead of two -- the K side packed at 7 bits, 112 B to a 128-code vector; `kvs` holds
        the matching scale per (batch, position, head), 2 bytes per vector. Every one of those
        tensors, and the int32 write positions, is a SLICE OF ONE ALLOCATION, which is what
        this candidate changes: the champion it derives from held one code pool and one scale
        pool per distinct owner extent, four allocations, and paid each one's 512-rounding.

        The charge, not the tensor bytes, is the whole of the difference; the shapes, strides
        and in-place appends are identical either way, because a slice of the outermost
        dimension of a contiguous tensor is contiguous and a standalone allocation of the
        remaining shape is the same tensor that slice was. What `torch.cuda.memory_allocated`
        reports is the block the caching allocator hands out, and every reading measured in
        this run agrees that a request at or below 1 MiB is handed its own 512-rounded bytes.

        Whether to fuse is a property of the SHAPES, not of the mechanism, and it has already
        flipped sign twice in this run. At `n_kv_head=4` a 1057-slot per-layer int8 tensor was
        1,082,368 B, strictly inside (1 MiB, 2 MiB), so fusing was 98.7% of launch 6's gain. At
        `n_kv_head=2` the same tensor was 541,184 B, below the band, while the fused group was
        3,247,104 B, above it with a non-integral MiB remainder and measured charged 4,194,304 --
        so `ext_unfuse_short_v1` (launch 10) unfused it and that was a KEEP. On THIS champion
        each owner's code tensor is 66,816 B and the whole fused group is 267,264 B, so every
        allocation on both sides of the change is below 1 MiB and the 512-rounding regime and
        the whole-MiB regime agree. Fusing therefore pays again, and only because the cache has
        since shrunk 126x: measured on the launch-24 hardware, a code tensor of this extent is
        charged 67,072 for 66,816 requested and a scale tensor 1,536 for 1,044, so four of each
        waste 1,024 + 1,968 B of pure rounding, and the second counter wastes a further 512 B
        of a 512 B block holding 4 B. Fused: 267,264 exactly (a whole multiple of 512), 4,608
        for 4,176, and one counter block. Nothing the model computes changes.

        The paragraph above is launch 28's, at a cache 3.8x this one, and its conclusion is the
        one this candidate extends across the dtype boundary. On THIS champion the four
        allocations are 20,880, 48,480, 360 and 808 B requested against 20,992, 48,640, 512 and
        1,024 charged: 640 B of pure rounding, on four blocks of which none is a whole multiple
        of 512. One pool of 70,528 B is charged 70,656 -- 128 B of rounding, one block recovered
        -- and it is still two orders of magnitude below the 1 MiB band where the charge stops
        being 512-rounding, so the regime argument above is unchanged and both regimes still
        agree. What this makes true for successors is worth a line: with one pool the charge is
        `512*ceil((C*sum(owner extents) + 4*n_ctr*batch)/512)` where `C = k_packed + v_packed
        + 4`, so a reach rung is priced at a flat C bytes per unit of extent and the "extent a
        multiple of 32" alignment rule that the 240 B slot created applies to nothing any more.
        THIS CANDIDATE MOVES C, and it is the only thing on the board that can: 244 at the
        K7/V8, 228 at K7/V7 and **212** at this row's K6/V7. The `[charge]` probe prints
        the ladder at
        whatever C this file actually has, so a half-applied pack shows up there. Nothing the
        model computes changes.
        """
        assert max_len <= self.cos.size(1), f"max_len {max_len} exceeds rotary table"
        cfg = self.config
        head_dim = cfg.n_embd // cfg.n_head
        dev = self.transformer.wte.weight.device
        extents, rolling = [], []
        for left, _right in self.window_sizes:
            need = left + 1 + self.DECODE_CACHE_SLACK
            if left >= 0 and need < max_len:
                extents.append(need)
                rolling.append(True)
            else:
                extents.append(max_len)
                rolling.append(False)
        # THE UNIFORM-ROLLING-EXTENT GUARD IS LIFTED, and that lift is this candidate's one
        # mechanism. The champion collapsed every extent to `max_len` the moment two rolling
        # extents differed, because it carried exactly one rebasing counter and one schedule;
        # its own comment gave bookkeeping as the reason and named no fidelity or kernel
        # limit. So the bookkeeping is generalised instead: ONE write position and ONE rebase
        # schedule PER DISTINCT ROLLING EXTENT, and a layer reads the one belonging to its own
        # extent. With a single distinct rolling extent this is the champion's arithmetic
        # unchanged -- one absolute counter, one rolling counter, `left + 1 + slack` slots,
        # a rebase every `extent - left` steps -- so the champion's programs are recovered
        # exactly and only a non-uniform request reaches new code.
        #
        # `win_extents` is ascending and is the index space for everything per-group below.
        # `extent = left + 1 + slack` is an identity of the loop above, so the group's `left`
        # is recovered from its extent and nothing has to be threaded through.
        win_extents = sorted({e for e, r in zip(extents, rolling) if r})
        win_hists = [e - 1 - self.DECODE_CACHE_SLACK for e in win_extents]
        _slot_of = {e: k for k, e in enumerate(win_extents)}
        win_slots = [_slot_of[e] if r else None for e, r in zip(extents, rolling)]
        # Both write positions live in one (2, batch) int32 allocation. `ctr[0]` and `ctr[1]`
        # are contiguous (batch,) int32 views with stable addresses, which is all the captured
        # graph needs, and one 512 B block serves both instead of two.
        # ONE allocation per cache OWNER instead of one per layer. The per-layer `kvq`/`kvs`
        # lists are ALIASES: layer i and its owner name the same tensor, so a group of
        # KV_GROUP layers costs one allocation and nothing else in the decode path has to
        # know about ownership to find its cache. This is where the byte saving is: at
        # KV_GROUP 2 and n_layer 8 the state holds four cache tensors, not eight.
        owners = [i for i in range(cfg.n_layer) if is_kv_owner(i)]
        for owner in owners:
            group = [j for j in range(cfg.n_layer) if kv_owner_of(j) == owner]
            assert len({extents[j] for j in group}) == 1, (
                "one shared cache serves one extent, so a group's layers must agree on it; "
                f"layers {group} asked for {[extents[j] for j in group]}")
        # ...and ONE ALLOCATION PER DISTINCT OWNER EXTENT instead of one per owner. Each
        # owner's cache is a slice of the outermost dimension of its extent's fused tensor,
        # which is contiguous and is the same tensor a standalone allocation would have been,
        # so every kernel sees the same shape, strides and in-place append. The fused tensors
        # are held in the state so dropping the state still frees the data.
        owner_extents = [extents[owner] for owner in owners]
        unique_extents = sorted(set(owner_extents))
        kv_alloc = []
        fused_kq, fused_q, fused_s = [], [], []
        # The two int32 write positions are the TAIL OF THE FIRST SCALE POOL rather than an
        # allocation of their own. A (2, batch) int32 counter is 8 B of a 512 B block on every
        # geometry this run has measured, so the block is 98.4% rounding; carried inside a pool
        # that is already charged, it costs the pool nothing, because `scale_numel + 4*batch`
        # bfloat16 elements round to the same 512 multiple as `scale_numel` does at every extent
        # the sharing ladder can reach. The int32 view is legal and not a coincidence of this
        # geometry: `scale_numel` carries the factor 2 of the K/V axis, so the tail's bfloat16
        # offset is always even and always 4-byte aligned. `ctr[0]` and `ctr[1]` remain
        # contiguous (batch,) int32 views with addresses stable across `add_`, `fill_` and
        # `zero_`, which is all the captured graph binds to, and they do not alias the scales:
        # `checks/bytefloor_check.py` asserts dtype, shape, stride, contiguity, values,
        # non-aliasing in both directions and data_ptr stability on uv.lock's torch 2.9.1.
        # Written by gpu3 (`prec_bytefloor_rider_v1`, post ba9cd418); re-derived and re-checked
        # here rather than copied on trust, and its byte prediction re-registered at this
        # candidate's own geometry.
        # One int32 row for the absolute position and one for each distinct rolling extent,
        # all still slices of the first scale pool's tail. Two rows was the champion's count
        # because it had one rebase schedule; the row count now follows the schedules.
        n_ctr = 1 + len(win_extents)
        ctr = None
        # THE CODE POOL IS 208 B PER CACHED SLOT INSTEAD OF 224, and that is this candidate's
        # one mechanism in this function. K's 128 codes are 6 bits and pack to 96 B; V's 128
        # stay at launch 52's 7 bits and 112 B, both with no remainder. The two halves differ
        # in width again, as they did at K7/V8, so the leading axis of extent 1 on each half is
        # doing the same work it did there -- keeping the TIME AXIS AT DIM 2 for both, which is
        # what lets `index_copy_(2, pos, ...)`, the `[:, :, :keep]` retention slice and
        # `_rebase_decode_state`'s window copy stay the champion's indices. The pool is ONE
        # allocation per distinct owner extent -- the invariant every byte reading in this run
        # rests on -- and K and V are contiguous halves of it rather than two slices of a
        # leading axis of extent 2, because their last dimensions now differ.
        #
        # Each half keeps a LEADING AXIS OF EXTENT 1 where the champion had the K/V axis of
        # extent 2. It carries no data and it is not decoration: it keeps the TIME AXIS AT DIM
        # 2 for both halves, so `index_copy_(2, pos, ...)` on the step path, the `[:, :, :keep]`
        # retention slice on the prefill path and `_rebase_decode_state`'s window copy are the
        # champion's own indices unchanged. Moving a cache's time axis is exactly the silent
        # error class this run's CPU protocol check was written for, and not moving it is
        # cheaper than re-checking it.
        k_packed = head_dim * K_BITS // 8
        v_packed = head_dim * V_BITS // 8
        assert head_dim * K_BITS % 8 == 0 and head_dim * V_BITS % 8 == 0, (
            f"a vector of {head_dim} codes at K {K_BITS} / V {V_BITS} bits is not a whole "
            "number of bytes; the pack has no remainder slot and this geometry needs a "
            "different width")
        # ...and each pack needs a whole number of GROUPS, which is a SECOND requirement and
        # not implied by the first: 4 codes to 3 bytes on the K side at 6 bits, 8 codes to 7
        # bytes on the V side at 7. head_dim 128 satisfies both. Asserted rather than assumed
        # because the group width is the one arithmetic fact the new pack adds.
        assert head_dim % 4 == 0 and head_dim % 8 == 0, (
            f"head_dim {head_dim} is not a whole number of pack groups (K needs 4 codes per "
            f"group at {K_BITS} bits, V needs 8 at {V_BITS})")
        # THE WHOLE DECODE STATE IS ONE ALLOCATION, and that is this candidate's one mechanism
        # in this function. The champion holds FOUR: a code pool and a scale pool per distinct
        # owner extent, each 512-rounded on its own, and it pays the rounding of each. Nothing
        # about the shapes, the strides, the values or the arithmetic changes -- only how many
        # blocks the caching allocator hands out.
        #
        # Launch 28 fused the code tensors with each other and the scale tensors with each
        # other; launch 38 carried the int32 counters in the first scale pool's tail. Neither
        # crossed the DTYPE boundary, because at those geometries there was nothing there to
        # win: the code groups were exact multiples of 512. At this champion's geometry the four
        # surcharges are 112 + 160 + 152 + 216 = 640 B, and one pool pays 128 of it.
        #
        # The pool is allocated in BFLOAT16, the widest dtype it must serve, so every view below
        # is a NARROWING view of a contiguous slice and needs no alignment argument beyond the
        # one PyTorch already enforces:
        #   * the scales ARE bfloat16 and are the pool's own elements;
        #   * `ctr` is int32 and needs a 2-element (4-byte) offset, which it has because every
        #     `scale_numel` carries the factor 2 of the K/V axis, so `n_scale` is even -- the
        #     same argument the champion's tail-carried counter already rests on;
        #   * the codes are uint8, and a bfloat16 -> uint8 view of a contiguous 1-D slice is
        #     always legal and simply doubles the length.
        # The counters keep their own rows and their own addresses; they do not alias the
        # scales or the codes, and `checks/fuseall_dtype_check.py` (gpu5) asserts dtype, shape, stride,
        # contiguity, non-aliasing in both directions and `data_ptr` stability across `add_`,
        # `fill_` and `zero_` on uv.lock's torch 2.9.1, exactly as the launch-38 rider did.
        #
        # ORDER: scales, then counters, then codes. Scales first is what makes the counter's
        # offset even without a pad, which is the ONE alignment this layout actually requires and
        # the only thing the assert below states. The codes need none: a bfloat16 -> uint8 view is
        # a narrowing view, and every consumer of the code tensors is a pointwise torch op --
        # `unpack7`, the `index_copy_` append, the rebase `clone` and the dequantising multiply --
        # so an offset that is not a multiple of 16 would be slower at worst and never wrong.
        # It happens to be 16-aligned at this geometry (`4*(sum(owner extents) + n_ctr*batch)` =
        # 1,168 B) and `checks/fuseall_dtype_check.py` (gpu5) reports that; NOTHING here depends on it,
        # and a pad to force it would make the byte identity depend on the pad.
        counts = [owner_extents.count(extent) for extent in unique_extents]
        scale_numels = [count * 2 * batch * extent * cfg.n_kv_head * 1
                        for count, extent in zip(counts, unique_extents)]
        code_bytes = [count * batch * extent * cfg.n_kv_head * (k_packed + v_packed)
                      for count, extent in zip(counts, unique_extents)]
        n_scale = sum(scale_numels)
        n_extra = 2 * n_ctr * batch
        assert n_scale % 2 == 0 and n_extra % 2 == 0, (
            "the int32 counter rows need a 4-byte offset, so the bfloat16 element count before "
            f"them must be even; got n_scale {n_scale}, n_extra {n_extra}")
        pool = torch.zeros(n_scale + n_extra + sum(code_bytes) // 2,
                           dtype=torch.bfloat16, device=dev)
        at = 0
        for count, extent, scale_numel in zip(counts, unique_extents, scale_numels):
            fused_s.append(pool[at:at + scale_numel].view(
                count, 2, batch, extent, cfg.n_kv_head, 1))
            at += scale_numel
        ctr = pool[at:at + n_extra].view(torch.int32).view(n_ctr, batch)
        at += n_extra
        codes = pool[at:].view(torch.uint8)
        at = 0
        for count, extent in zip(counts, unique_extents):
            slots = count * batch * extent * cfg.n_kv_head
            fused_kq.append(codes[at:at + slots * k_packed].view(
                count, 1, batch, extent, cfg.n_kv_head, k_packed))
            at += slots * k_packed
            # The V half is now the SAME SHAPE OF OBJECT as the K half: a packed uint8
            # bitstream of `v_packed` bytes per (position, head), not `head_dim` int8 codes.
            # The `.view(torch.int8)` the champion had here is dropped rather than kept,
            # because `unpack7` reads a uint8 bitstream and a signed view of the same bytes
            # would decode the top bit wrong. `_pack7` returns uint8, `index_copy_` and the
            # `[:, :, :keep]` retention slice are dtype-matched to it, and the leading axis
            # of extent 1 stays exactly where it was so the TIME AXIS IS STILL DIM 2 on both
            # halves -- the whole reason that axis exists.
            fused_q.append(codes[at:at + slots * v_packed].view(
                count, 1, batch, extent, cfg.n_kv_head, v_packed))
            at += slots * v_packed
        seq = ctr[0]
        # ONE entry, where the champion's list had four. `kv_alloc` exists only so that dropping
        # the state frees the data, and every tensor above is a view of this one pool.
        kv_alloc = [pool]
        group_of = {extent: pos for pos, extent in enumerate(unique_extents)}
        next_slot = {extent: 0 for extent in unique_extents}
        kkq_base, kvq_base, kvs_base = [], [], []
        for extent in owner_extents:
            group_pos, slot = group_of[extent], next_slot[extent]
            next_slot[extent] = slot + 1
            kkq_base.append(fused_kq[group_pos][slot])
            kvq_base.append(fused_q[group_pos][slot])
            kvs_base.append(fused_s[group_pos][slot])
        kkq = [kkq_base[i // KV_GROUP] for i in range(cfg.n_layer)]
        kvq = [kvq_base[i // KV_GROUP] for i in range(cfg.n_layer)]
        kvs = [kvs_base[i // KV_GROUP] for i in range(cfg.n_layer)]
        return {
            "max_len": max_len,
            "graph_enabled": bool(graph),
            "seq": seq,
            # The layer index of each group's cache owner, in depth order.
            "owners": owners,
            # Per-layer allocated slots, and whether that layer rebases.
            "extents": extents,
            "rolling": rolling,
            # One entry per DISTINCT ROLLING EXTENT, ascending, and `win_slots[i]` is the
            # entry layer `i` reads (None when it holds the absolute position). The champion's
            # scalar `win_hist` / `win_extent` / `win_pos` / `seq_win` are these lists of
            # length one whenever the rolling extents happen to agree.
            "win_extents": win_extents,
            "win_hists": win_hists,
            "win_slots": win_slots,
            # Host mirrors of the rolling write positions. The rebasing decision reads these
            # and never the device, so no step synchronises; part of the state, reset with it.
            "win_poss": [0] * len(win_extents),
            "seq_wins": [ctr[1 + k] for k in range(len(win_extents))],
            # The allocations everything above is a view of, kept in the state for the reason
            # the bases were: dropping the state must free the data.
            "ctr": ctr,
            "kv_alloc": kv_alloc,
            # `kkq` is the packed 7-bit K side, `kvq` the int8 V side. Two names where the
            # champion had one, and every consumer below reads both.
            "kkq_base": kkq_base,
            "kvq_base": kvq_base,
            "kvs_base": kvs_base,
            "kkq": kkq,
            "kvq": kvq,
            "kvs": kvs,
        }

    def reset_decode_state(self, state):
        """Return `state` to position 0 so the same buffers serve another request.

        A fresh state per request would invalidate an address-bound CUDA graph. Stale bytes
        above the position are unreachable -- attention reads only `cache_seqlens` keys -- so
        zeroing them would be timed work serving never does. That is why a rebasing layer may
        leave the slots above its write position holding a previous request's codes: the same
        argument covers them, and it is the window mask rather than `cache_seqlens` that makes
        the ones below unreachable.
        """
        state["seq"].zero_()
        for counter in state["seq_wins"]:
            counter.zero_()
        state["win_poss"] = [0] * len(state["win_extents"])
        return state

    def _rebase_decode_state(self, state, headroom=1):
        """Copy each rebasing layer's live window back to slot 0, if the next `headroom`
        appends would otherwise run past its buffer.

        Runs on the host between steps, never inside the captured graph: the graph reads
        `seq_win` as a tensor, so writing that tensor's value between replays is all it needs
        to keep appending in the right slot. Costs one window copy per rebasing buffer every
        `extent - win_hist` steps, and the buffers are int8 codes plus their bfloat16 scales,
        so the copy moves a little over half the bytes the bfloat16 cache's did.
        """
        # One schedule per distinct rolling extent, each tested and fired independently: a
        # 64-slot group rebases four times as often as a 256-slot one and the two must not be
        # coupled. Each group's predicate is the champion's, on that group's own numbers.
        for k, extent in enumerate(state["win_extents"]):
            if state["win_poss"][k] + headroom <= extent:
                continue
            hist = state["win_hists"][k]
            # Iterate the OWNER buffers, not the per-layer alias list. Copying a live window
            # back to slot 0 is not idempotent -- a second copy would read the slots the first
            # one just overwrote -- so rebasing a shared cache once per reader would corrupt
            # it. This is one of the two places the launch-9 transplant is not a copy; the
            # other is the per-owner allocation above.
            for o, owner in enumerate(state["owners"]):
                if state["win_slots"][owner] != k:
                    continue
                # Three buffers where the champion had two: the packed K codes, the V codes
                # and the scales. All three keep the time axis at dim 2, so the slice is the
                # champion's; a packed K row is 112 contiguous bytes and moves as bytes.
                for buf in (state["kkq_base"][o], state["kvq_base"][o], state["kvs_base"][o]):
                    # Source and destination overlap, so the read is materialised first. The
                    # time axis is dim 2: the leading axis is K/V and the next one is batch.
                    buf[:, :, :hist] = buf[:, :, extent - hist:extent].clone()
            state["seq_wins"][k].fill_(hist)
            state["win_poss"][k] = hist

    def _step_device_counters(self, state):
        """Advance both write positions by one step. Captured with the step it belongs to."""
        state["seq"].add_(1)
        for counter in state["seq_wins"]:
            counter.add_(1)

    def _step_host_position(self, state):
        """Advance the host mirror once per executed step, eager or replayed."""
        for k in range(len(state["win_poss"])):
            state["win_poss"][k] += 1

    def _decode_body(self, idx, state, prefill):
        """`forward`'s body with the cache spliced in. `prefill` is a Python bool that is
        constant per variant, so the two branches are two graphs, not one per position."""
        B, Tn = idx.size()
        if prefill:
            cos, sin = self.cos[:, :Tn], self.sin[:, :Tn]
        else:
            seq_idx = state["seq"].to(torch.int64)
            # A rebasing layer writes and attends at its own position; rotary is always the
            # absolute one, because a key's angle is a property of the token, not the slot.
            # One position per distinct rolling extent, selected per layer by `win_slots`.
            win_idxs = [counter.to(torch.int64) for counter in state["seq_wins"]]
            cos = self.cos.index_select(1, seq_idx)
            sin = self.sin.index_select(1, seq_idx)
            # A reader's cache already holds the current token -- its owner appended it a few
            # layers earlier -- so the reader has to be told there is one more key than the
            # owner was given. Both counters get the +1 form; which one a layer uses follows
            # `rolling[i]` exactly as in the owner branch below.
            seq1 = state["seq"] + 1
            win1s = [counter + 1 for counter in state["seq_wins"]]

        x = norm(self.transformer.wte(idx))
        x0 = x
        # Same contract as `forward`: a cache owner publishes the K/V its group attends over.
        kv_shared = None
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            attn = block.attn
            h = norm(x)
            q = attn.c_q(h).view(B, Tn, attn.n_head, attn.head_dim)
            q = norm(apply_rotary_emb(q, cos, sin))
            k = v = None
            if attn.is_kv_owner:
                k = attn.c_k(h).view(B, Tn, attn.n_kv_head, attn.head_dim)
                v = attn.c_v(h).view(B, Tn, attn.n_kv_head, attn.head_dim)
                if ve is not None:
                    ve = ve.view(B, Tn, attn.n_kv_head, attn.head_dim)
                    gate = 2 * torch.sigmoid(attn.ve_gate(h[..., :attn.ve_gate_channels]))
                    v = v + gate.unsqueeze(-1) * ve
                k = norm(apply_rotary_emb(k, cos, sin))

            kkq, kvq, kvs = state["kkq"][i], state["kvq"][i], state["kvs"][i]
            # Which rolling schedule this layer lives on, or None if it holds the absolute
            # position. A reader's owner is in its own group, and the group-agreement assert
            # in `init_decode_state` makes the two slots equal, so the reader and the buffer
            # it reads always agree on the extent.
            win_slot = state["win_slots"][i]
            if prefill and not attn.is_kv_owner:
                # The owner computed this group's full-width bf16 K/V a layer earlier and
                # already retained what the cache keeps. A reader stores nothing and attends
                # over the same values with ITS OWN window -- exactly what `forward` does.
                kc, vc = kv_shared
                y = fa3.flash_attn_func(q, kc, vc, causal=True,
                                        window_size=self.window_sizes[i])
            elif not prefill and not attn.is_kv_owner:
                # The owner's kernel call overwrote its write position in the published view
                # with the unquantised k,v, so a reader sees the current token at exactly the
                # fidelity the owner did, appends nothing, and is told about one more key.
                kc, vc = kv_shared
                seqlens1 = seq1 if win_slot is None else win1s[win_slot]
                y = fa3.flash_attn_with_kvcache(q, kc, vc, k=None, v=None,
                                                cache_seqlens=seqlens1, causal=True,
                                                window_size=self.window_sizes[i],
                                                num_splits=1)
            elif prefill:
                # The prefill's own attention reads the k and v it just computed, exactly
                # the values the reference reads back out of its bfloat16 cache. What goes
                # into the cache for later steps is the quantised copy.
                kcodes, vcodes, scale = quantise_kv(k, v)
                # Retain only the slots this layer can still read from the next position:
                # all Tn of them when it holds the absolute position, the last `win_hist`
                # when it rebases and the prefill is longer than its window.
                keep = Tn if win_slot is None else min(Tn, state["win_hists"][win_slot])
                kkq[:, :, :keep] = kcodes.unsqueeze(0)[:, :, Tn - keep:]
                kvq[:, :, :keep] = vcodes.unsqueeze(0)[:, :, Tn - keep:]
                kvs[:, :, :keep] = scale[:, :, Tn - keep:]
                kv_shared = (k, v)      # published to this group's readers
                y = fa3.flash_attn_func(q, k, v, causal=True,
                                        window_size=self.window_sizes[i])
            else:
                # Appends k,v at cache_seqlens and attends over the whole preallocated
                # cache, so there is no `kc[:, start:end]` slice whose bounds depend on the
                # step number. num_splits=1 is not a tuning choice: the op's fake kernel
                # refuses to trace at the default num_splits=0, which is precisely what
                # makes an unpinned cache uncompilable. It is a reduction-order choice, and
                # the agreement check in prepare.py still has to pass with it pinned.
                #
                # This token's codes go into the int8 cache, then the whole cache is
                # dequantised into a bfloat16 view for the kernel to read. That view is a
                # per-step temporary: it is freed before the next step and holds no history,
                # so it is per-step working memory in exactly the sense prepare.py's cache
                # probe excludes (`memory_allocated`, not the peak). The state still owns
                # every byte of retained cache data, and all of it stays on the GPU.
                kcodes, vcodes, scale = quantise_kv(k, v)
                if win_slot is None:
                    pos, seqlens = seq_idx[:1], state["seq"]
                else:
                    pos, seqlens = win_idxs[win_slot][:1], state["seq_wins"][win_slot]
                kkq.index_copy_(2, pos, kcodes.unsqueeze(0))
                kvq.index_copy_(2, pos, vcodes.unsqueeze(0))
                kvs.index_copy_(2, pos, scale)
                # The champion's one `kvq * kvs` becomes an unpack-and-scale on the K side and
                # the same multiply on the V side. `unpack7` is compiled, so the whole 7-bit
                # bitstream is one pointwise kernel; the dequantised views are per-step
                # temporaries exactly as `kv` was, freed before the next step and holding no
                # history, which is what the cache probe's `memory_allocated` read excludes.
                kc = unpack6(kkq[0]) * kvs[0]
                # The V side is launch 52's, unchanged: the same compiled `unpack7` on the
                # same 112 B stream. The K side above is a FIRST INSTANCE of a new chain, so
                # launch 48's +29.97 ms for introducing one is the relevant precedent and
                # launch 52's +7.037 for a second instance of an existing one is NOT -- that
                # distinction is launch 52's own measured finding and it cuts against this row.
                vc = unpack7(kvq[0]) * kvs[1]
                # Published to this group's readers. The kernel below overwrites this write
                # position of kc/vc with the unquantised k,v, so a reader sees the current
                # token at exactly the fidelity the owner did and every older key at the
                # cache's own int8 fidelity.
                kv_shared = (kc, vc)
                y = fa3.flash_attn_with_kvcache(q, kc, vc, k=k, v=v,
                                                cache_seqlens=seqlens, causal=True,
                                                window_size=self.window_sizes[i],
                                                num_splits=1)
            x = x + attn.c_proj(y.contiguous().view(B, Tn, -1))
            x = x + block.mlp(norm(x))

        x = norm(x[:, -1:, :])
        softcap = 15
        logits = self.lm_head(x).float()
        return softcap * torch.tanh(logits / softcap)

    def decode_step(self, idx, state):
        """`logits, state = model.decode_step(idx, state)`, logits for the LAST position only.

        Width > 1 is the prefill and runs once per request; width 1 is the step, and it is
        the call the graph replays. `seq` is advanced on both paths, so the tensor is the
        single source of the position.
        """
        if idx.size(1) > 1:
            logits = self._decode_body(idx, state, prefill=True)
            state["seq"].add_(idx.size(1))
            for k, hist in enumerate(state["win_hists"]):
                # `_decode_body` retained the last min(Tn, hist) codes at slot 0 for each
                # rolling group, so that group's next write lands there rather than at the
                # absolute position. Each group keeps its own amount.
                kept = min(idx.size(1), hist)
                state["seq_wins"][k].fill_(kept)
                state["win_poss"][k] = kept
            return logits, state
        self._rebase_decode_state(state)
        if not state.get("graph_enabled", True):
            logits = self._decode_body(idx, state, prefill=False)
            self._step_device_counters(state)
            self._step_host_position(state)
            return logits, state
        # The graph lives ON the state it was captured against, so its lifetime is exactly
        # that state's and there is no key to get wrong. A state whose capture failed holds
        # an uncaptured object and runs eager forever, which is slow and correct.
        graph = state.get("graph")
        if graph is None:
            # Capture executes WARMUP_REPLAYS + 1 real appends, all at the SAME live slot:
            # `_capture` calls `_restore(seq0, win0)` before every warmup replay and again
            # before the capture, and restores once more afterwards, so none of them advances
            # the write position (its own docstring, point 3, says so). One free slot is the
            # requirement, and `decode_step` asked for exactly that four lines above. Asking
            # for WARMUP_REPLAYS + 1 here was a redundant over-request, and a harmful one: at
            # any slack below 3 it FIRES after a long prefill and copies slots that were never
            # written into the live window. gpu4 proved it on CPU and measured it live at four
            # owners (launch 30); my launch 32 measured it live at two owners with the graph
            # still capturing and two spare slots. That, not the capture, was the whole of the
            # observed slack floor of 3.
            self._rebase_decode_state(state, headroom=1)
            graph = state["graph"] = _GraphedDecodeStep(self, state)
        if not graph.captured:
            logits = self._decode_body(idx, state, prefill=False)
            self._step_device_counters(state)
            self._step_host_position(state)
            return logits, state
        out = graph.replay(idx)
        self._step_host_position(state)
        return out, state


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
            self.graph, self.static_logits = None, None

    def _advance(self):
        logits = self.model._decode_body(self.static_idx, self.state, prefill=False)
        self.model._step_device_counters(self.state)
        return logits

    def _restore(self, seq0, win0):
        self.state["seq"].copy_(seq0)
        for counter, saved in zip(self.state["seq_wins"], win0):
            counter.copy_(saved)

    def _capture(self):
        seq0 = self.state["seq"].clone()
        win0 = [c.clone() for c in self.state["seq_wins"]]   # empty when no layer rebases
        host0 = list(self.state["win_poss"])
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(self.WARMUP_REPLAYS):
                self._restore(seq0, win0)
                self._advance()
        torch.cuda.current_stream().wait_stream(stream)

        self._restore(seq0, win0)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_logits = self._advance()
        self._restore(seq0, win0)              # the capture itself executed one increment
        self.state["win_poss"] = list(host0)   # and neither it nor the warmup is a step

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
KV_HEAD_RATIO = 4       # query heads per K/V head (1 = the reference's MHA, 4 = MQA at n_head 4)
WINDOW_PATTERN = "SSSSSSLL" # sliding window pattern: L=full, S=half context. Six locally
                        # attending layers and a long PAIR at depths 6 and 7, so both long
                        # layers sit in one KV_GROUP=2 group and the group-agreement assert
                        # in init_decode_state holds. A pattern whose long layers straddle a
                        # group boundary -- including the old "SSSL", long layers 3 and 7 --
                        # is unbuildable at KV_GROUP 2.
SHORT_WINDOW = 26       # keys a short layer may read back; was 28 (launches 49/52/55), 31 (launch 44's lineage), 63 (41).
                        # Owner extent 29. ODD ON PURPOSE: at 240 B/slot 3*29*240 = 20,880 lands at 41
                        # blocks while extents 30 and 31 both need 43 and 44, so 29 is a 2,048 B step
                        # where they are 1,024 and 512. The EVEN-extent rule of champions v25 and v27
                        # was arithmetic at 256 B/slot and does not survive launch 48.
LONG_WINDOW = 197       # keys a long layer may read back; was 201 (launch 47), 225 (44), 255 before that. It no longer
                        # has to equal SHORT_WINDOW: the uniform-extent guard in
                        # init_decode_state that forced that is LIFTED below, and each distinct
                        # rolling extent now carries its own write position and its own rebase
                        # schedule. 197 is THIS candidate's one literal: owner extent 202 -> 198,
                        # which under champion v30's single pool is a flat 4 * 244 = 976 B and
                        # crosses one 512-boundary, so kv_cache_bytes 70,656 -> 69,632.

# Optimization
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step; grad_accum_steps 2 -> 1
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
    # Grouped-query attention: KV_HEAD_RATIO query heads share one K/V head, so the cached
    # width is kv_dim = n_kv_head * head_dim = 256 instead of 512. num_heads must stay
    # divisible by n_kv_head, which CausalSelfAttention asserts.
    num_kv_heads = max(1, num_heads // KV_HEAD_RATIO)
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_kv_heads, n_embd=model_dim,
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

model = torch.compile(model, dynamic=False, mode="max-autotune-no-cudagraphs")

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

# ---------------------------------------------------------------------------
# Print-only diagnostic: where the residue in kv_cache_bytes comes from.
#
# prepare.py's probe reads memory_allocated around one graph=False state that is prefilled
# and decoded through the request, after a discarded warm-up. On every launch measured in
# this run the reading exceeds the exact tensor extents of the cache by exactly 1,048,576 B
# at max_len 2048 and by exactly 0 B at max_len 513, so ~11% of this candidate's predicted
# reading is a charge nobody has attributed. This block replays the probe's shape and
# prints the delta after each stage -- state allocation, prefill, decode -- so the residue
# can be attributed without buying a launch.
#
# It reads no validation data (random ids; the residue is an allocation, not a value), holds
# nothing that outlives it, prints no METRICS_JSON line, runs before the reporter, and is
# wrapped so that nothing it does can end the launch.
# ---------------------------------------------------------------------------

def _residue_probe(prefill=1536, steps=512):
    max_len = prefill + steps
    ids = torch.randint(0, vocab_size, (1, max_len), device=device)

    def _alloc():
        gc.collect()
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated()

    def _build(report=False):
        base = _alloc()
        state = model.init_decode_state(batch=1, max_len=max_len, graph=False)
        after_state = _alloc()
        with torch.no_grad(), autocast_ctx:
            logits, state = model.decode_step(ids[:, :prefill], state)
            del logits
            after_prefill = _alloc()
            for position in range(prefill, max_len):
                logits, state = model.decode_step(ids[:, position:position + 1], state)
                del logits
        after_decode = _alloc()
        if report:
            # Read the exact extents off the state instead of assuming max_len on every
            # layer: this candidate allocates window+1+slack slots on the six short layers,
            # so the inherited closed-form would report a residue against a cache size the
            # program does not have. Print-only, as before.
            exact = sum(t.numel() * t.element_size()
                        for group in (state["kkq_base"], state["kvq_base"], state["kvs_base"])
                        for t in group)
            print(f"[residue] per-layer slots           : {state['extents']}")
            print(f"[residue] KV_GROUP {KV_GROUP}, owners {state['owners']}, "
                  f"distinct kvq tensors {len({t.data_ptr() for t in state['kvq']})} "
                  f"of {len(state['kvq'])} layer entries")
            # Print-only, and the one number this candidate wants on the record: the K side is
            # 7/8 of the V side per slot, so a wrong pack width shows up here and not only in
            # the charge. 208 B per (position, head) is the whole mechanism, and the halves
            # must read K 96 + V 112 -- the one line that falsifies a half-applied K pack, and
            # the line that says which of the two 208 B routes this file actually is.
            print(f"[residue] bytes per cached slot      : "
                  f"K {state['kkq_base'][0].shape[-1]} + V {state['kvq_base'][0].shape[-1]} = "
                  f"{state['kkq_base'][0].shape[-1] + state['kvq_base'][0].shape[-1]}")
            print(f"[residue] exact cache tensor bytes  : {exact:,}")
            print(f"[residue] + seq block (512 B)       : {exact + 512:,}")
            print(f"[residue] init_decode_state         : {after_state - base:,}")
            print(f"[residue] after prefill ({prefill} tok)  : {after_prefill - base:,}")
            print(f"[residue] after {steps} decode steps  : {after_decode - base:,}")
            print(f"[residue] residue vs exact+512      : {after_decode - base - exact - 512:,}")
        return state

    def _capture_probe():
        """Print-only falsifier for the headroom-1 pre-capture rebase, carried from gpu4's
        launch 30 and my launch 32 unchanged: does the graph still capture at
        DECODE_CACHE_SLACK 1? A failed capture is correct but eager, and it would show up in
        request_ms_median rather than in any byte figure, so this says which happened."""
        state = model.init_decode_state(batch=1, max_len=max_len, graph=True)
        with torch.no_grad(), autocast_ctx:
            logits, state = model.decode_step(ids[:, :prefill], state)
            del logits
            for position in range(prefill, min(prefill + 8, max_len)):
                logits, state = model.decode_step(ids[:, position:position + 1], state)
                del logits
        graph = state.get("graph")
        print(f"[capture] slack {model.DECODE_CACHE_SLACK}  extents {state['win_extents']}  "
              f"win_hists {state['win_hists']}  spare "
              f"{[e - h for e, h in zip(state['win_extents'], state['win_hists'])]}")
        print(f"[capture] per-layer extents {state['extents']}  win_slots {state['win_slots']}  "
              f"owners {state['owners']}  win_poss {state['win_poss']}")
        print(f"[capture] captured : {getattr(graph, 'captured', None)}")
        print(f"[capture] reason   : {getattr(graph, 'reason', '') or '(none)'}")
        del state, graph
        gc.collect(); torch.cuda.empty_cache()

    warm = _build()
    del warm
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    state = _build(report=True)
    del state
    gc.collect()
    torch.cuda.empty_cache()
    try:
        _capture_probe()
    except Exception as _cap_exc:             # noqa: BLE001 -- diagnostic only, never fatal
        print(f"[capture] probe skipped: {type(_cap_exc).__name__}: {_cap_exc}")


model_config_for_probe = {
    "n_layer": config.n_layer,
    "n_kv_head": config.n_kv_head,
    "kv_dim": config.n_kv_head * (config.n_embd // config.n_head),
}
try:
    _residue_probe()
except Exception as _residue_exc:            # noqa: BLE001 -- diagnostic only, never fatal
    print(f"[residue] probe skipped: {type(_residue_exc).__name__}: {_residue_exc}")

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


# ---------------------------------------------------------------------------
# Print-only allocator-charge probe. Deliberately placed AFTER the reporter.
#
# This run's charge law was refuted as an exact predictor at launch 8 (see
# knowledge/kv_cache_charge_measured.md, addendum): memory_allocated() reports the block
# the caching allocator hands out, and above 1 MiB that block depends on allocator history,
# so no static function of requested sizes has explained all readings. The cheap fix is to
# stop inferring the charge and measure it. This block allocates each of this candidate's
# actual cache shapes one at a time, prints its own memory_allocated delta, and does the
# same for the fused shapes it replaces -- so the head-to-head is on one launch's hardware
# and one allocator, not across two launches.
#
# It runs after report_efficiency_metrics deliberately: the reporter has already taken every
# official number, including kv_cache_bytes and peak_vram_bytes, so nothing this block
# allocates can perturb the measurement it is diagnosing. It prints no METRICS_JSON line and
# is wrapped so that nothing it does can end the launch.
# ---------------------------------------------------------------------------

def _charge_probe():
    head_dim = config.n_embd // config.n_head
    kv = config.n_kv_head

    def charge(shape, dtype):
        gc.collect()
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        buf = torch.zeros(*shape, dtype=dtype, device=device)
        torch.cuda.synchronize()
        got = torch.cuda.memory_allocated() - base
        want = buf.numel() * buf.element_size()
        del buf
        gc.collect()
        torch.cuda.synchronize()
        return want, got

    # gpu5, struct_fuseall_dtype_v1: every inherited row was priced at extent 261 or 1057 and is
    # five champions out of date. Repriced onto THIS champion's real geometry, and repriced to
    # attribute ONE claim by measurement: the champion's FOUR allocations against this candidate's
    # ONE. The first rows are the four the champion holds, one at a time on this launch's
    # allocator; the row after them is the single pool that replaces all four, and the whole
    # -512 B of the claim is the difference between the two sums. Then the one-pool ladder for
    # the rungs a successor can buy, so the charge law -- a flat `k_packed + v_packed + 4` B
    # per unit of owner extent, no alignment rule -- is measured here rather than derived. The
    # coefficient is read off THIS file's widths rather than written down, so at this
    # candidate's K6/V7 the rows below are the 212 B/unit ladder and not the champion's 228.
    # A 512 B block is 2.415 units of owner extent here, against 2.246 at 228. Print-only,
    # after report_efficiency_metrics, as before.
    slack = GPT.DECODE_CACHE_SLACK
    k_packed = head_dim * K_BITS // 8
    v_packed = head_dim * V_BITS // 8
    probe_len = 2048
    _ex, _roll = [], []
    for _left, _right in model.window_sizes:
        _need = _left + 1 + slack
        _rolling = _left >= 0 and _need < probe_len
        _ex.append(_need if _rolling else probe_len)
        _roll.append(_rolling)
    owners = [i for i in range(config.n_layer) if is_kv_owner(i)]
    owner_extents = [_ex[o] for o in owners]
    uniq = sorted(set(owner_extents))
    n_ctr = 1 + len(sorted({e for e, r in zip(_ex, _roll) if r}))

    def one_pool_elems(oe):
        """bfloat16 element count of this candidate's single pool at owner extents `oe`."""
        u = sorted(set(oe))
        n_scale = sum(oe.count(e) * 2 * 1 * e * kv for e in u)
        n_codes = sum(oe.count(e) * 1 * e * kv * (k_packed + v_packed) for e in u)
        return n_scale + 2 * n_ctr * 1 + n_codes // 2

    rows = []
    for _i, _e in enumerate(uniq):
        _cnt = owner_extents.count(_e)
        _cb = _cnt * 1 * _e * kv * (k_packed + v_packed)
        rows.append((f"CHAMPION codes  extent {_e} x{_cnt}  [{_cb}] u8", (_cb,), torch.uint8))
    for _i, _e in enumerate(uniq):
        _cnt = owner_extents.count(_e)
        _sn = _cnt * 2 * 1 * _e * kv + (2 * n_ctr if _i == 0 else 0)
        rows.append((f"CHAMPION scales extent {_e} x{_cnt}"
                     f"{' + ' + str(n_ctr) + ' ctr rows' if _i == 0 else ''}  [{_sn}] bf16",
                     (_sn,), torch.bfloat16))
    rows.append((f"THIS one pool  extents {owner_extents}  [{one_pool_elems(owner_extents)}] bf16",
                 (one_pool_elems(owner_extents),), torch.bfloat16))
    # The one-pool ladder. Local rungs move three owners at once, long rungs one, and at this
    # file's 212 B per unit a 512 B block is 2.415 units -- so these rows also say which rungs
    # are free, re-rounded a FIFTH time because the coefficient moved again. Launch 52's ladder
    # (at 228) is SUPERSEDED by whatever this prints, and its LONG E=196 zero-byte rung is a
    # property of 228 alone -- do not carry it over without re-reading these rows.
    for _d in (1, 2, 3, 4, 5):
        _oe = [e - _d if e == min(uniq) else e for e in owner_extents]
        rows.append((f"LADDER local -{_d} (extents {_oe})  [{one_pool_elems(_oe)}] bf16",
                     (one_pool_elems(_oe),), torch.bfloat16))
    for _d in (2, 4, 6, 8, 10):
        _oe = [e - _d if e == max(uniq) else e for e in owner_extents]
        rows.append((f"LADDER long  -{_d} (extents {_oe})  [{one_pool_elems(_oe)}] bf16",
                     (one_pool_elems(_oe),), torch.bfloat16))
    rows.append(("one int32 counter (the block fusion has already recovered)", (1,), torch.int32))
    print("[charge] requested -> charged, one allocation at a time, nothing else live")
    for label, shape, dtype in rows:
        want, got = charge(shape, dtype)
        print(f"[charge] {label:<50} requested {want:>12,}  charged {got:>12,}  "
              f"surcharge {got - want:>10,}")


try:
    _charge_probe()
except Exception as _charge_exc:              # noqa: BLE001 -- diagnostic only, never fatal
    print(f"[charge] probe skipped: {type(_charge_exc).__name__}: {_charge_exc}")
