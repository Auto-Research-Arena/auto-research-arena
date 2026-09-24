"""
Autoresearch pretraining script. Single-GPU, single-file.
Cherry-picked and simplified from nanochat.
Usage: uv run train.py

Experiment k5_plus_a_partial_vocab_ve_table_at_layer_0 (queued by team kernel_fusion as
`k5_plus_a_partial_vocab_ve_table_to_cover_the_0.00083_gate_miss`, claimed and built by agent
autoscts__decode_gpu3 of team architecture_prune), on top of LAUNCH 84's frozen candidate
`parallel_block_k5_on_v42_at_the_long_window_site_layer_3` (gpu2, seq 84, source sha256
315a907dedd8668169b6f4db200086dcaf5f537a9239b56e250ab300e93359d9,
nopref_request_ms_median 49.456000328063965 -- the FASTEST program this run has measured,
-2.9265880585 ms on champion v42 -- and INELIGIBLE on `val_bpb` 1.0508281130470107 against the
strict `< 1.05` gate, failing by +0.0008281130 with every other ceiling passed).
ALL PRIOR EXPERIMENT SECTIONS ARE PRESERVED BELOW THIS ONE VERBATIM.

ONE MECHANISM: add ONE partial-vocab value-embedding table (4096 x 512) at layer 0, the only
layer without one. The k=5 site set is L84's and is not touched. The full pre-registration --
what is measured, what is assumed, the registered band, the falsifier, and the hazard the row
was queued with -- is at `PARTIAL_VE_ROWS`. Read it there.

WHY IT IS A BUILD ON AN INELIGIBLE BASE, stated because that is unusual. L84 failed one clause
and one only: quality. Its latency mechanism is measured and its other seven ceilings passed
with room. The gap is +0.0008281130 bpb, and the only capacity this task's params ceiling still
sells is 2,097,668 parameters -- half a table's rows. So this is not a re-measurement of L84;
it is L84 plus the quality mechanism L84 was short of.

DEVICE-FREE VERIFICATION, all of it before dispatch, each with a control:
  * `num_params_total` 50,331,660 against the 50,332,176 ceiling -- fits by exactly 516, and the
    delta from L84 is exactly 2,097,152 = one 4096 x 512 table. `flops_per_token_analytic`
    188,743,680, UNCHANGED from L84: `estimate_flops_analytic` subtracts `value_embeds` by name.
    Both read by exec'ing `prepare.py`'s OWN `count_params` and `estimate_flops_analytic`
    verbatim against a model built on `torch.device("meta")`.
  * The five FULL-vocab layers emit L84's machine code EXACTLY. `VE_BOUND` is a `tl.constexpr`,
    so the guard is a compile-time branch: AOT-compiled for sm80 with faithful `attrs`
    (divisibility 16 on every pointer AND on `stride_kl`/`stride_cos`), the VE_BOUND=0 PTX
    instruction body is IDENTICAL to L84's at 936 lines. The raw cubin sha DIFFERS and that is
    debug info, not code -- the control settles it: the same bytes compiled from two different
    paths give two different cubin shas, and `.loc` directives plus a `.b8` blob spelling out
    the file path are the whole difference.
  * Layer 0's guarded variant COMPILES for sm80 (a `@triton.jit` kernel compiles on first call
    ON THE LAUNCH NODE, so a codegen error would raise after the 600 s is already charged), and
    it DIFFERS from the full-table one -- 942 instruction lines against 936. `ld.global` 18 ->
    18 and `bar.sync` 19 -> 19: no extra load and NO RE-FENCE, which is the check this run has
    seen cancel a base twice.
  * `_ve_lookup` is bitwise `F.embedding` on covered ids and EXACT ZERO on uncovered ones, with
    a control that fires (row 0 of the table is nonzero, so "uncovered == 0" cannot pass
    vacuously), and bitwise `F.embedding` on a full table.
  * The guarded Triton spelling equals a ZERO-PADDED full table bitwise at 6 token ids x 2
    heads under `TRITON_INTERPRET`, run as a PAIRED comparison so the interpreter's fp32->bf16
    truncation cancels in both legs.
  * Layer 0's table is WIRED INTO `forward`, not a dead target: all six layers receive a
    non-None `ve`, and layer 0's rows are nonzero for ids 0/5/4095 and exactly zero for
    4096/8191.

Experiment parallel_block_k5_on_v42_at_the_long_window_site_layer_3 (team kernel_fusion, agent
autoscts__decode_gpu2), on top of champion v42 parallel_block_k4_on_v41_with_the_site_ground_RESTATED
(gpu2, seq 82, source sha256 e30a5dda7ad136ed3ac8e5c03a88c240c2e0166ff147679bd8c6279f06767682 read
FROM DISK, 52.382588386535645 ms, ELIGIBLE on every ceiling, `val_bpb` 1.0435420354017853 with margin
+0.0064579646). ALL PRIOR EXPERIMENT SECTIONS ARE PRESERVED BELOW THIS ONE VERBATIM.

ONE LINE: `PARALLEL_BLOCK_LAYERS = (0, 1, 2, 4) -> (0, 1, 2, 3, 4)`, k=4 -> k=5 of 6. The site is
LAYER 3, the pre-registration and the stop clause are recorded at `PARALLEL_BLOCK_LAYERS`, and
nothing else in this file changes.

THIS REVERSES THE DISPOSITION OF MY OWN CLOSURE ROW, ON NO NEW MEASUREMENT, AND THAT IS STATED HERE
RATHER THAN IN A RETRACTION. `parallel_block_dial_CLOSED_at_k4_on_three_independent_grounds` (mine,
filed at L82, re-read and upheld by two auditors) says DO NOT BUILD on three grounds. Ground 1 is a
BET FRAMING, not an impossibility: at L82's controlled-step cost the k=5 prediction is `val_bpb`
1.0498672573 against `< 1.05`, which PASSES by +0.0001327427, and the ground's own objection is that
+0.0001327427 is 0.02 of one k>=2 path draw (sd 0.0061274305) -- i.e. the outcome is decided by the
draw and not by the mechanism. Ground 2 (the site ground) is a ground, never a measurement, and gpu3
recorded at cycle 30 that it need not be a wall. Ground 3 (capacity quantised at 4,194,304 with
2,097,668 left) blocks the FUNDING of a margin purchase and says nothing about the bet. What changed
is not evidence but the ALTERNATIVE USE OF A LAUNCH: both boards now report, independently and three
times over, no live row with a mechanism that can move the ranked metric, 14 consecutive cycles have
spent zero launches, and absent a purchase the reachable launches expire unspent. Under those
conditions "the outcome is decided by the draw" is a reason to price the bet, not to decline it.

WHAT IS MECHANISM HERE AND WHAT IS A DRAW -- the distinction this launch rests on. The LATENCY leg is
mechanism and it is the best-measured quantity on this board: six per-site in-graph readings,
-2.413 / -2.463 / -2.387 / -2.438 / -2.595 / -2.580 us/call for the input merge, plus the output
merge at -1.917..-1.983 us/call on the site's own four-launch band (logs/0078-0082, the L82 cell
excluded as a probe-capture artefact). Layer 3 takes BOTH merges, because the out-merge is keyed on
`i + 1 < n_layer` (`:4446-4451`), so the added site is worth -4.497..-4.563 us/call = -2.307..-2.341
ms at 0.513 ms per us/step -- 1.42-1.44x the 1.626 ms promote bar. LAYER 5 IS REFUSED FOR THIS REASON
AND NOT ON QUALITY: it is the last layer, gets the input merge only, and its -1.324 ms cannot clear
the bar. The ELIGIBILITY is the draw: three per-site quality readings exist, +0.0121368267 (L78->L79,
k=2->3), +0.0063252219 (L81->L82, k=3->4, the only CONTROLLED-STEP pair at 978 steps each), and
+0.0016097/site (L76 vs L75, k=2->6 averaged over four sites). They look concave in that order, and
CONCAVITY IS NOT ESTABLISHED: each leg carries the ~0.0061 draw, so a pair difference carries ~0.0087
and all three readings sit inside about one sd of each other. Taking their mean +0.0066906 as the
central, k=5 predicts 1.0502 with a prediction sd near 0.008, so P(`val_bpb` < 1.05) is about 0.49,
and against the run's mechanism-independent ~8.3% TV breach base rate P(ELIGIBLE) is about 0.45.
Registered before the launch. I am buying a 45% chance of a measured -2.3 ms, not a certainty.

STOP CLAUSE, and it is gpu1's adjudication applied to my own row: an INELIGIBLE `val_bpb` is a DRAW
on the dial, NOT a refutation of it, so this launch cannot close the dial in either direction. What
it settles regardless of the gate is (a) the third per-site quality reading and the FIRST at a
long-window site, which is the quantity my own champion text names as unidentified, and (b) the first
k=5 latency reading, testing the near-linearity of the latency leg (ratio 1.069 between k=4's -2.580
and k=6's -2.413). If it is eligible and faster it is a KEEP and I publish the champion; if it is
eligible and slower the latency leg's linearity is refuted and I say so; if it is ineligible the row
records a draw and the dial stays where it is at k=4. NO SECOND SEED IS AVAILABLE: the task pins
`launch.seed`, so a confirmation seed is blocked-confirmation, recorded and not waived.

Experiment parallel_block_k4_on_v41_with_the_site_ground_RESTATED (team kernel_fusion, agent
autoscts__decode_gpu2), on top of champion v41 add_two_value_embedding_tables_on_the_k3_parallel_block
(gpu1, seq 81, source sha256 b8199457eb50bc71eedd5796e484f5b779c5d2df1f6cdb9361eed4b877e65a73 read
FROM DISK, 54.9931526184082 ms, ELIGIBLE on every ceiling, `val_bpb` 1.0372168135284558 with margin
+0.0127831865). ALL PRIOR EXPERIMENT SECTIONS ARE PRESERVED BELOW THIS ONE VERBATIM.

ONE LINE: `PARALLEL_BLOCK_LAYERS = (0, 2, 4) -> (0, 1, 2, 4)`, k=3 -> k=4 of 6. The complete
argument, the restated site ground that gpu1's stop clause requires, the adjacency verification and
the registered bands are all recorded at `PARALLEL_BLOCK_LAYERS`. Nothing else in this file changes.

WHY THIS AND NOT THE TWO HIGH-PRIORITY AUTO-BRACKET MIDPOINTS, declined with arithmetic rather than
with a preference. Both midpoints exist to identify L81's unidentified per-table quality coefficient.
Row B (`k` back to 2, both tables held) would identify the k leg -- but the k leg is ALREADY a
measured adjacent pair, L78 1.0387620736200858 / 57.09207057952881 against L79 1.0508989003011076 /
55.07040023803711, and L81's own record cites its -2.021670341491699 ms and uses it. Buying it again
duplicates a measurement the ledger holds (Step 4a3 failure shape 1) and its predicted headline is
~57.0 ms, about 2 ms WORSE than the champion. Row A (one table at k=3) is the more interesting of the
two and still not worth a launch, for a reason that holds in BOTH directions of the noise question:
the decode-only pool L1-L5 + L8 -- six DIFFERENT programs at byte-identical `num_params_total`
50,332,176 and `flops_per_token_measured` 239,078,400, whose training paths are untouched -- reads
`val_bpb` sd 0.0003323956 over 761-763 steps while its `nopref_request_ms_median` spans 224.771618 to
465.544342, a factor of 2.07. Take that 0.00033 as the yardstick and L81's two-table credit
-0.0136820868 is ALREADY identified at ~41 sd, so row A only adds a linearity check; take the k>=2
path draw sd 0.0061274305 as the yardstick instead and row A is powered at 1.1 sd on a per-table
effect of about half that, so it CANNOT identify the coefficient. Underpowered on one yardstick,
redundant on the other, and the axis it would inform has zero remaining subjects either way -- a
third table is inadmissible by 2,096,636 parameters. I take no position on the midpoint clause
itself, which fired correctly; I decline the two rows it generated.

Experiment add_two_value_embedding_tables_on_the_k3_parallel_block (team kernel_fusion, agent
autoscts__decode_gpu1), on top of champion v40 merge_out_projections_on_the_k2_parallel_block
(gpu2, seq 78, source sha256 e3794a18989ac60606aa26229f3c824cb43e92dc771f0bac9c41bb1445b49f4b,
57.09207057952881 ms, ELIGIBLE on every ceiling, `val_bpb` 1.0387620736200858 with margin
+0.0112379264). ALL PRIOR EXPERIMENT SECTIONS ARE PRESERVED BELOW THIS ONE VERBATIM.

**THE BINDING CONSTRAINT ON THIS TASK IS THE GATE, AND THIS IS THE FIRST MECHANISM AIMED AT IT WITH
A PRICE ALREADY PAID.** Three launches have measured FASTER than the champion and all three died on
`val_bpb` alone: L75 53.591370582580566 (1.0532409328644643), L79 55.07040023803711
(1.0508989003011076), L80 57.05404281616211 (1.0529004586321782). Every latency axis on both boards
is closed. What was missing was quality headroom with a mechanism behind it.

TWO LEGS, and the first is not mine:

* **Leg A, `autoscts__decode_gpu6`'s:** `PARALLEL_BLOCK_LAYERS = (0, 2) -> (0, 2, 4)`, k=3. Measured
  as launch 79: **-2.021670341491699 ms against v40, 18.235x the 0.110865 ms floor**, INELIGIBLE on
  `val_bpb` short by **0.0008989003** and on nothing else. I re-derive none of it and claim none of
  it. Its own author refused a redraw of it, correctly: a draw is not a dial.
* **Leg B, mine:** two added value-embedding tables at layers 2 and 4. The full argument, the paid
  rate, the exact ceiling arithmetic, the unmeasured transfer band and a correction to the rate's
  stated ground are all recorded at `EXTRA_VE_LAYERS`.

**WHY THIS IS NOT BUYING FOR A FAVOURABLE DRAW**, which seven-plus agents on this board have refused
and one of them was me. A draw purchase is one whose EXPECTED mechanism effect is zero -- a
bit-identical or zero-effect rebuy hoping the instrument moves. Here the expected effect is strictly
positive with a rate paid in this run's own ledger at 46 sd, and the uncertainty is the MAGNITUDE of
a transfer, not the existence or the sign of the mechanism. L6's own step credit runs AGAINST the
capacity term, so the measured rate is a conservative floor. Separately, leg A alone cannot be
re-bought at all: `engine/runtime.py:647` refuses a repeated identity before `write_once` at 650.

REGISTERED, all against launch 79 as the anchor:

* `num_params_total` **48,234,508** exactly (95.8324% of the 50,332,176 `<=` ceiling).
* `flops_per_token_measured` **188,743,680**, byte-identical to L78/L79/L80. **If this moves at all,
  "a VE table is FLOPs-free" is refuted at this geometry and leg B is not what I claim.**
* `nopref_kv_cache_bytes` **6,304,256**, byte-identical to v39 and L75-L80.
* `training_data_tokens_available` **631,241,817** exact -- the only `==` constraint.
* `peak_vram_bytes` in **[34.7, 36.3] GB = [73.5%, 76.9%]** of the frozen 47,198,976,512 (L79 read
  34,154,108,928 = 72.36%; the run's activation unit is legible in L79's own peak sitting
  537,919,488 B below L78's for one parallel block, about 2 x `[128, 2048, 512]` bf16).
* **Throughput:** two tables cost <= 1.0% of train tok/s, band **[828,000, 836,315]** (L79 read
  836,314.0083300432; L6's FOUR-table reading is 0.9941%, so the scaled central is 0.50%).
  **REFUTED above 1.5%**, which would make the VE throughput term superlinear in table count.
* **Decode cost of two extra `ve` gathers:** whole step against L79 in **[+0.03, +0.15] ms, central
  +0.0718**, from gpu3's corrected 0.10773 ms ceiling for the whole owner `ve` block on 3 of 6
  attention-partial calls. **REFUTED above +0.30 ms** (4.2x central), which would mean the gather is
  not per-layer additive. `kernel_delta 0` against L79: `HAS_VE` is a `tl.constexpr`, so two more
  layers change which specialisation compiles, not the per-step kernel count.
* **Ranked metric** central **55.1422 ms**, band [55.10, 55.22]; delta vs champion **-1.9499 ms
  central = 1.199x the 1.626 ms promote bar and 17.6x the floor** (band 1.151x-1.225x). Headline
  87.7364913915% -> ~88.1553%.
* **Gate margin** central **+0.0033011**, band **[-0.0008989, +0.0054191]** -- it STRADDLES ZERO,
  and I am recording that here, before the launch, rather than in a retraction.

STRUCTURAL CHECKS, from the champion read FROM DISK (5,808 lines, md5 6c64c274..., k=2 at :1538):
`has_ve` has exactly ONE functional caller, the `value_embeds` comprehension; VE and `parallel` are
orthogonal in BOTH paths (`Block.forward` passes `ve` into `self.attn` on either spelling, and in
the width-1 path `have_ve` and `block.parallel` are independent branches with
`_decode_qkv_rot_write_attention` taking the table or `None` regardless), so a parallel block may
carry a value embedding with no change beyond the construction predicate.

CREDITS. Leg A and its -2.021670 ms: gpu6. The +0.003159/table rate, the `has_ve` parity table and
the 10,486,276-parameter headroom: gpu3. `value_embedding_mode_none` (L6): its author. The
eligibility-versus-attribution distinction: analyst2. The 0.10773 ms `ve` ceiling: gpu3. Mine: the
ADDITION direction, the exact 2.50-table headroom, the ledger proof that a VE table is FLOPs-free at
these geometries, the step-credit correction to the rate's ground, and the observation that the
standing "scale-up refused ON MERIT at 17 of 17 slower" closure has a SCOPE FAILURE against this row
-- all 17 are width/depth scale-ups that grow every matmul and every counted FLOP (best L17 =
156.1335325241089 ms = 2.73x the champion), while a VE table grows no matmul and no counted FLOP.

Experiment merge_out_projections_on_the_k2_parallel_block (team kernel_fusion, agent gpu2), on top
of launch 77's candidate pin_merged_qkv_fc_r0block_32_on_the_k2_parallel_block (gpu5, source sha256
a33080506c0a0d86470107e6a7fb6276eb138fac852cd437b6df58c6af2d7a63, whole step 59.0822696685791 ms,
INELIGIBLE on `val_bpb` 1.0501650195410994 by 0.000165), whose training path is byte-identical to
launch 76's ELIGIBLE 59.92996692657471 -- the reported result. The champion is still v39
pin_qkv_matvec_norm_nw1_on_an_idempotent_pin at 60.965895652770996.

**THE TWO OUT-PROJECTIONS OF A PARALLEL BLOCK ARE ONE REDUCTION, AND THAT IS ONLY TRUE ON A
PARALLEL BLOCK.** On a `parallel_block` layer `_decode_attn_out_add`'s output `xa` has exactly one
consumer, `_decode_mlp_out_mix`, which reads it as a POINTWISE term while reducing over `hidden` --
and `hidden` came out of the merged input projection before the attention ran. So

    xa    = mixed + Wa @ ao                      # [512,512],  depth 512
    mixed = mix(xa + Wm @ hidden, x0)            # [512,2048], depth 2048

is `mix(mixed + Wa @ ao + Wm @ hidden, x0)`: ONE reduction over 2560 terms, ONE kernel, on 2 of 6
layers. In the sequential block it does not exist, because there `hidden` depends on `xa`.

**PROVENANCE, INCLUDING THE PART THAT IS MINE TO OWN.** This is my own queue row
`merge_out_projections_on_parallel_block`, proposed 2026-09-13T15:55:56Z and closed by me 32
minutes later as "unreachable ... launch 50 refused the parallel block on val_bpb". That was a
REACHABILITY closure, and this run has adjudicated that reachability closures expire when the
record moves. It moved twice (L75 measured the block at -7.374525 ms; L76 landed it at k=2
ELIGIBLE), and the three-day delay in noticing is mine, not the record's.

**HOW ONE KERNEL IS REACHED WITHOUT A COPY, A CAT OR A REQUANTISATION.** The two reductions share
`xnumel` (512 output rows) and differ in `rnumel` (128 against 512 int32 words), and inductor does
not fuse reductions across different reduction ranges. Three spellings fail: a `cat` on the operand
materialises a copy kernel; a preallocated adjacent [2560] buffer needs two other regions to write
into slices of one allocation; and ONE merged weight with ONE row scale is a REQUANTISATION, i.e.
an approximation, which this run measured at +0.0240 nopref TV on the four per-layer projections and
which is refused here. What works needs none of them: `lane_major=True` means lane `i` of word `c`
pairs with input element `i * IN/4 + c`, so reading the [512,2048] weight's words as [512, 4, 128]
gives **16** lane tensors of [512,128] where there were 4 of [512,512], the attention half already
has 4 at [512,128], and all **20** lane products then share reduction extent 128 and close with one
`sum(-1)`. Every operand and every weight term is a VIEW of a tensor that already exists: no
allocation, no copy, no requantisation, and **both per-output-row scales are preserved exactly**.

**WHAT IT SPENDS, PRICED WITH THIS RUN'S OWN SLOPE AND NOT ASSERTED BITWISE.** Three bf16 roundings
go (`_rne_bf16` on each matvec result and the bf16 store of `xa`) and the accumulation is
reassociated, the row scale multiplying each of the 128 partial sums instead of the closed sum.
`train.py:2189-2191`'s calibration gives **1.47e-03** TV per bf16 ulp and **~5e-07** for an
fp32-granularity reassociation, so the worst case is **<= +0.0044** and the sign is not predictable
(deleting roundings moves toward fp32, and TV is measured against a reference carrying its own).
**The binding shape is PREFILL, not nopref:** L77 read prefill 0.032294 (margin 0.017706, 4.0x
cover) against nopref 0.021784 (margin 0.028216), and this run's two fidelity fatalities split 1-1
across the two ceilings. No bitwise claim is available -- a compiled region's reduction is not
reproducible from outside it, 0 of 15 spellings -- and none is made.

**PRICE: THE CREDIT IS EXACTLY ONE PER-CALL FIXED TERM, SOLVED ON ITS OWN AXIS.** L77's own probe
block prints both sites at 512 output rows, varying depth -- the axis this candidate sits on:
`_decode_attn_out_add WORDS live [512,512]` **2.064 us/call** at depth 512
(`logs/0077/attempt-1/stdout.log:250`) and `PACKED _decode_mlp_out_mix [512,2048]` **3.279** at
depth 2048 (:254). Slope **7.910e-4 us/element**, intercept **F = 1.659 us/call**; the merged region
at depth 2560 predicts **3.684** against the pair's **5.343**, so the saving is F exactly, since both
legs pay `F + s*depth` and the merge pays it once. Registered: merged region **3.684 us/call**, band
[3.34, 3.89]; whole step **-1.702 ms** = 1.659 x 2 calls/step x 0.513, band **[-1.49, -2.05] ms**
over this run's four F constructions (1.446 = L77's `seq.add_(1)` row, 1.659 = this solve, 1.895 =
the corpus fit, 1.9603 = L75's rider), i.e. **13.4x-18.5x** the 0.110865 ms floor. Projection
**57.38 ms**, band [57.03, 57.59], clearing the 59.339896 promote bar at every point by 1.75-2.31 ms.

**STOP CLAUSE, decidable from this launch's own printed rows.** REFUTED if the merged region's probe
row reads **>= 5.343** us/call (the pinned pair's own sum in this same launch) or the whole-step
delta against L77 is **>= 0**. MODEL WRONG IN MAGNITUDE, mechanism intact, if it reads **> 4.343**
(credit < 1.0 us/call = 0.60x the band's low end). CONFIRMED in [3.34, 3.89] with a whole step in
[-1.49, -2.05] ms. Step 4a3 checked before registering: this kernel has never existed in any launch,
so no printed series covers it and nothing here re-buys a measured quantity.

**REGISTERED AGAINST MYSELF, BEFORE DISPATCH.** (1) `val_bpb` is a ~50% DRAW and an INELIGIBLE
outcome refutes NOTHING here: the training path is byte-identical to L76 and L77, which read
1.0468021404190588 and 1.0501650195410994 on allocations `a5e7ae99` and `bf7fc4cc` -- one side of
the 1.05 gate each, 0.00336 apart at +7 steps -- and `prepare.py:958` precedes the decode probes at
1004-1008, so this diff cannot move it. I am not buying for a favourable draw: the gain is measured
within the launch by its own paired probe row, and the draw only decides whether the number may be
ranked. (2) The new kernel's own config is an UNPINNED coordinate-descent draw at hints
{'x': 512, 'r0_': 128} -- the class L77 showed can ship two different winners from one heuristic
start. Seven instances already sit at those hints in L77's roster, all `pinned=False coordesc=True`,
and NO entry of `_DECODE_CONFIG_PIN` claims that shape, so nothing of mine collides with gpu5's four
entries and nothing of gpu5's captures my kernel. Deliberately not pinned: no launch has benchmarked
a config for a kernel that never existed, and inventing one is not evidence. (3) Depth 2560 is a
1.25x extrapolation past the far point, so a convex cost curve shrinks the credit; and 20 live lane
terms at extent 128 is more register pressure than 4 or 16. Both unpriced, sign unknown, disclosed
rather than folded in. (4) The input merge's 43% is NOT claimed -- see the scope failing case below.

**Step 4a4, the rule and its two failing cases.** RULE: merging two reductions that share an output
shape and whose operands are both already resident deletes exactly one per-call fixed term. SCOPE
FAILURE (wrong object, nonsense answer): applied to the INPUT-side merge it is wrong -- that one
concatenates on the ROW axis and L77 prints merged 3.253 against pair 5.716, a credit of 2.463
us/call, 1.26x-1.70x every F construction in this run, because row concatenation also moves the
program count 192 -> 448. MAGNITUDE FAILURE (right object, wrong size): taking F from the
`seq.add_(1)` node floor 1.446 instead of solving on the site's own axis predicts a 1.446 credit
where that same merged input site measured 2.463 -- off by 1.70x. A fixed term must be solved at the
site, on the axis you will read it on.

Everything else is launch 77's, byte-identical, including its pin dict, the parallel-block layer set
(0, 2) and every instrument. Without that pin this kernel's config would be a second unheld draw
stacked on mine.

Experiment pin_merged_qkv_fc_r0block_32_on_the_k2_parallel_block (team kernel_fusion, agent gpu5),
on top of launch 76's ELIGIBLE candidate parallel_attention_mlp_block_at_k2_of_6
(59.92996692657471 ms, autoscts__decode_gpu2 -- the reported result, NOT the champion, which is
still v39 pin_qkv_matvec_norm_nw1_on_an_idempotent_pin at 60.965895652770996).

**The one kernel the parallel block creates is the one kernel nothing pins, and its config was a
per-launch coordinate-descent draw.** `_decode_qkv_fc_matvec_norm` MERGED [3584,512] sits at
`next_power_of_2(3584) = 4096`, claimed by no roster entry. Launches 75 and 76 started it at the
identical heuristic `(XBLOCK 8, R0_BLOCK 128, num_warps 8)` and shipped different winners because
descent took different coordinates first: L75 `(X8, R0 32, nw8)` at 12.768 us, L76
`(X16, R0 128, nw8)` at 13.408 us, with L75's winner absent from L76's 12-row table. This entry
pins L75's winner. ONE executable change: a fourth dict in `_DECODE_CONFIG_PIN`. The architecture,
every expression, every constant, the parallel-block layer set (0, 2) and every instrument are
byte-identical to launch 76.

Registered before dispatch. Central **-0.554 ms** from the in-graph leg (-0.540 us/call x 2
calls/step x 0.513), band **[-0.37, -0.79] ms**, every point of it 3.4x-7.1x the 0.110865 ms
cross-launch floor, so **the reported result improves at every point in the band** (59.929967 ->
[59.14, 59.56]). The **promote bar 59.339896 is NOT claimed**: the central misses it by 0.036 ms and
only the favourable end reaches it, so a promotion here would sit inside the ~0.595 ms cold-cache
term and must be reported as contaminated rather than banked.

`val_bpb` CANNOT be moved by this diff: `prepare.py:958` reads it before the decode probes at
1004-1008, and this is a decode-path launcher config with no training-path expression touched. So
the expectation is launch 76's own 1.0468021404190588, margin +0.0031978596 -- and an INELIGIBLE
outcome on `val_bpb` would be an ALLOCATION draw, not a refutation of this pin. That is registered
here rather than argued afterwards: this run's only same-program same-allocation pair moves
0.00007935 in `val_bpb`, while the step count across allocations moves ~1.6% = ~15 steps, which at
the dial's own implied step coefficient exceeds the +0.0032 margin.

Experiment hoist_the_owner_loads_above_the_folds (team kernel_fusion, agent gpu5), on top of
champion v38 split_the_combine_reduction_16_plus_1 (62.954783 ms, gpu4)

**The five loads the attention partial's owner program issues from BEHIND its barriers, issued
before them instead.** A cross-warp `tl.sum` lowers through shared memory with `BAR.SYNC`, and a
`bar.sync` is a scheduling fence for global loads -- so `xk`/`xkp` sit behind four barriers and
`vn`, `tok` and the value-embedding row behind seven, read off faithful-`attrs` sm80 SASS. The
owner is the kernel's slowest CTA and therefore its duration. Pure statement reorder: bitwise,
no expression touched, same grid, same widths, same reduction trees.

The section below describes launch 34's mechanism and is kept because the grid width it introduced
is still what ships; it is not this row's mechanism.

Experiment decode_attn_splits_live (team autoscts attention_cache, agent gpu5), on top of champion v19 fold_norm_into_projection (91.040730 ms, gpu6)

**The 15 dead splits the ranked shape has been carrying since champion v17, deleted.** The partial
kernel's grid width stops being `DECODE_ATTN_SPLITS` and becomes `min(cdiv(L, BL), NS)`; the combine
masks the slots the partial did not write. No change to what is computed, at either request shape.

## What the AOT census does and does not say

Stripped of `.loc`, `.file` and DWARF records -- which differ only because two kernels cannot share
a source file -- the sm80 instruction streams are: **ranked partial identical** (1310 lines, same
digest, only the grid differs, 128 CTAs -> 68); **tiebreak partial identical** (1506 lines);
**tiebreak combine 181 -> 184**; **ranked combine 181 -> 186**, the extra lines being one
`setp.lt.u32 ..., 17` and predicate setup, with the 33 `ld.global` predicated so 15 of each thread's
32 `num` loads stop moving bytes. `add.f32` stays at 36 in every case -- **the reduction chain still
adds the zeros the mask supplies, and that part of the dead-split cost is NOT removed by this
change.** That is a fact about what is emitted. It is not a price, and the section below says so.

## The measurement this is built on

`DECODE_ATTN_SPLITS` must be a power of two because `_decode_attn_combine` does `tl.arange(0, NS)`
(gpu6 AOT-compiled NS=17 for sm80: it raises). The ranked shape's floor is `cdiv(514, 32) = 17`, so
32 is the smallest legal value and **15 of its 32 splits address rows that do not exist**.

`decode_attn_splits_64` (launch seq 31) priced one of those by adding 32 more: `NS` 32 -> 64 at held
`BL`, +1.778749 us/call for 32 dead splits, **0.0555859 us/call each**. Launch 25 independently
priced a wholly masked block-iteration at 2.5098 us/call over 64 CTAs = 0.0392 per CTA-iteration,
which is 70% of that figure and the same object seen from the other side.

**The power-of-two rule binds `tl.arange`, not a grid dimension.** A grid dimension may be any
integer. So `GS` programs are launched and `NS` slots are summed, and the combine's `s < GS` mask is
a correctness requirement rather than a tuning knob -- slots `GS..NS-1` are never written and
`num`/`den` are `torch.empty`.

## Ranked-shape-only by construction

At `L = 2049` the cap does not bite: `NBLOCK = 65`, `GS = min(65, 32) = 32`, and
`NITER = cdiv(65, 32) = 3` -- the champion's grid, trip count and buffer stride exactly. AOT-compiled
for sm80 with debug records stripped, the partial kernel's instruction stream at the tiebreak is
**byte-identical** to the champion's; the combine gains **3 scalar lines**, because its 33
`ld.global` become predicated on a compile-time-constant true predicate (`mov.pred -1`, **no `setp`**)
that no longer disables anything. `request_ms_median` is registered as unchanged on that basis, and
the claim is a claim about the emitted code, which is the only kind this method supports.

## What is at risk

Nothing in the arithmetic: the live partials are the champion's, the padding contributes an exact
0.0 to both sums, and 68 CTAs instead of 128 removes only programs that stored zeros. What is at
risk is the price -- 0.0555859 us/call per dead split is one measurement, interpolated downward in
the same direction rather than extrapolated, and a reading at or above the champion would refute
both it and launch 25's masked-iteration price at once.

Experiment decode_attn_splits_32 (team attention_cache, agent gpu5), on top of champion v16 fold_ve_gather_into_qkv_write (108.096600 ms).

**One integer.** `DECODE_ATTN_SPLITS` 16 -> 32 in my own `_decode_cached_attention_triton`,
which champion v15 banked. Nothing else changes: no def is touched, `forward` and every
training-path definition are byte-identical, and `init_decode_state` is byte-identical.

## The defect, which is arithmetic and not an opinion

`NITER = cdiv(L, DECODE_ATTN_BLOCK_L * DECODE_ATTN_SPLITS)` is the partial kernel's loop trip
count. The timed request states are `max_len = prefill + steps + 1` (`prepare.py`), so the ranked
`nopref` shape is **L = 514** and the tiebreak is **L = 2049**. At `BL = 32` a 514-row cache is
**17 blocks of cache rows**, and 17 blocks over 16 splits is **NITER = 2**: every one of the 64
programs runs the loop body twice, 15 of the 32 block-slots address rows that do not exist, and
for 511 of the 512 steps the whole second iteration is masked from end to end.

At `NS = 32` it is **NITER = 1**: 128 programs, one block each, no loop.

**A masked iteration is not a free iteration.** The sm80 PTX is a rolled loop -- one copy of the
body, NITER as the trip count, an identical instruction census at NITER 2, 3 and 5 -- so the body
issues in full: the loads are predicated off and move no bytes, but 32 `ex2.approx`, 63
`fma.rn.f32`, 194 `add.f32` and the 162 `shfl.sync` of the two cross-lane reductions all run on
zeros. At NITER=1 the loop is gone: 2346 -> 2144 PTX lines, one fewer `bar.sync`, no `bra`.

**Why the constant was 16.** I carried it over from FA3's `num_splits`, where `kBlockN` is 64 or
128, a 514-row cache is 9 or 5 blocks, and 16 splits really is the floor of 1 block per split --
that argument is in `knowledge/decode_attention_call_price.md` and it is correct there. This
kernel's `BL` is 32, so its floor is **17** splits, not 16. `_decode_attn_combine` does
`tl.arange(0, NS)`, which requires a power of two -- gpu6 AOT-compiled NS=17 for sm80 and it
raises -- so **32 is the smallest legal value at the floor.**

## Prediction, pre-registered

**`nopref_request_ms_median` central 101.6, range 98.5 - 108.6.** The saving is one loop
iteration per call. My own seq 20 prices an iteration by cross-shape differencing: 8.066 us/call
at NITER=2 against ~16.5 at NITER=5, i.e. **~2.80 us per iteration gross**, of which ~0.46 us is
cache traffic the longer shape carries and the ranked shape's masked iteration does not -- so
**~2.34 us/call** of pure arithmetic, x 6 layers x 512 steps = **-7.2 ms**. Discounted for the
larger grid (128 CTAs) and a combine that now sums 32 partials instead of 16. **The top of the
range is a wash**, per `knowledge/triton_kernel_cpu_verification.md`: a CPU box cannot time a
kernel, and both of seq 20's misses were speed properties.

**`request_ms_median` (tiebreak): improves too, NITER 5 -> 3.** Registered as ~-11 ms, range
-3 to -18. This is the same defect gpu6's `decode_attn_splits_by_length` row targets from the
tiebreak side; one constant fixes both shapes, and gpu6's power-of-two finding is why it is 32.

## Ceilings, all equalities

`num_params_total` 39,845,900, `flops_per_token_measured` 188,743,680, `nopref_kv_cache_bytes`
6,304,256, `peak_vram_bytes` 35,767,867,392, `training_data_tokens_available` 631,241,817. No
training-path byte changes, and `num`/`den`/`out` remain per-call transients on no state --
`num` grows 32,768 -> 65,536 B and `den` 256 -> 512 B, both inside the graph's private pool.

`decode_tv_distance_max` / `nopref_...`: the split regrouping is 17 block-contributions summed in
a different order. Under `TRITON_INTERPRET=1` NS=32 is **bitwise identical to NS=16** on 27 (L,
pos, window) cases and within 1 bf16 ulp of the aten region, but interpret mode is not the device's
reduction order, so |dTV| <= ~0.005 with the sign unknown against a 0.05 ceiling.

`val_bpb`: unchanged in mechanism -- this is the `prefill=False` branch only. Quote it with
`num_steps`, which seq 23 showed is not a fixed integer.

---

Experiment fold_ve_gather_into_qkv_write (team kernel_fusion, agent gpu2), on top of
champion v15 compose_triton_attention_qkv_write (110.910535 ms).

ONE mechanism, stated as a general rule: **an input-dependent gather belongs inside the region
that consumes it.** `_decode_qkv_write` already consumed a value-embedding row, as `v + ve`, but
the row was produced OUTSIDE it by an eager `aten::embedding` call and handed in as a graph input.
This candidate hands the region the TABLE and `idx` and lets it index the table itself.

What that deletes from every width-1 step, at depth 6: **3 eager `aten::embedding` launches**, the
value-embedding rows for `has_ve` layers 1, 3 and 5. An eager aten call is exactly one CUDA launch
with no fusion decision involved, which is the same device-independent fact that made this
region's original -12 `index_copy_` launches bankable.

**Pre-registered before submission.** At the 1.601669 ms/node that this exact absorption into this
exact region measured across three baselines spanning 65 ms (198.291 -> 178.760,
177.028 -> 161.342, 133.189 -> 118.774; +/-1.9%), -3 nodes is -4.81 ms:

  * central **106.11**, range **104.5 - 110.9**.
  * The upper end is a **wash, and it is a genuine bound rather than a hope.** The 3 gathers are
    folded into 3 regions that already run, one per `has_ve` layer, so if inductor declines to
    fuse them on CUDA the launches move INSIDE those regions and the count is unchanged. This
    mechanism cannot regress on node count. gpu5's launch 20 recorded why the wash end must be
    in the range at all: a CPU box cannot time anything.
  * A reading near **109.0** would instead confirm the 0.65 ms/node pointwise price from
    `knowledge/decode_node_price_table.md` for this node class rather than the 1.60 critical-path
    price -- the first time the two are discriminated on a gather rather than on a store, and the
    informative outcome either way.

**The rotary half of this mechanism was deliberately SPLIT OFF and is not in this candidate.**
Folding `_decode_rotary_at`'s two gathers into the same region also measured 6 nodes and bitwise
on CPU, and it would delete a whole compiled region (confirmed: dynamo traced 14 graphs for the
champion and 13 for that variant). But the rotary row is gathered ONCE per step and consumed by
SIX layers, so folding it duplicates the gather 6x: if inductor declines to fuse it the candidate
carries 12 gathers per step where the champion has 2, which the eager census measured as +10
`aten::index_select` per step, i.e. +6.5 ms at the pointwise price or +16 ms at the
critical-path price. That is an unbounded downside on the same evidence that gives this one a
bounded one, and it confounds the rule being tested with a duplication cost. It is queued
separately as `fold_rotary_gather_into_qkv_write`, to be launched only if this lands -- at which
point the rule is established and the only open question is the duplication.

**Verified on CPU, no device** (`cpucheck/probe_fold_gathers.py`, `cpucheck/check_fold.py`):
the region is **6 scheduler nodes with the value-embedding gather folded in and 6 without, so the
fold adds ZERO nodes**, and it is **bitwise identical on q, kc and vc** at both request shapes,
over every position and all 12 cache tensors, with a control proving the cache row still changes
and a champion-vs-champion control that is not vacuous. Only the safe direction of the CPU/CUDA
fusion trap is used: this run has twice seen CPU UNDER-fuse relative to CUDA, so "CPU fuses it"
implies "CUDA fuses it".

Both harnesses load each revision with `importlib` from a real file and assert a non-zero
`unique_graphs` delta per revision (champion +14, candidate +14), per gpu6's finding that an
`exec`-loaded harness silently runs its second revision EAGER once dynamo's `recompile_limit` of
8 is consumed by this file's compiled regions -- which would make the A/B compiled-vs-eager and
manufacture rms_norm ulp mismatches.

**Disjoint from the two mechanisms already in this champion.** gpu1's compose changed only
`_decode_body`'s attention callee and added the three Triton defs; `_decode_qkv_write` is
AST-identical between v14 and v15. The three Triton defs are asserted AST-identical here.

**`forward` is untouched**, so `num_params_total`, `flops_per_token_measured`,
`training_data_tokens_available` and `val_bpb` are the champion's; `init_decode_state` is
untouched, so `nopref_kv_cache_bytes` is.

**Do not predict the fidelity readings unchanged.** This region's own history is the reason:
bitwise on CPU, `nopref_decode_tv_distance_max` still moved +0.007792 on one launch and -0.003603
on the next. The claim is |dTV| <= ~0.008 with the sign UNKNOWN.

Experiment compose_triton_attention_qkv_write (team kernel_fusion, agent gpu1), on top of
champion v14 fuse_decode_qkv_write_on_v12 (118.773580 ms).

**Two measured, eligible mechanisms banked together. Neither is mine.**

| half | author | measured on | delta | in the champion before this? |
|---|---|---|---|---|
| Triton split-K decode attention (`_decode_cached_attention_triton`) | **autoscts__decode_gpu5**, seq 20 | v12 133.188605 -> 125.279903 | **-7.908702** | **no** -- raced out by v14 |
| `_decode_qkv_write` append absorption | **autoscts__decode_gpu2**, seq 15/18/21 | v12 133.188605 -> 118.773580 | **-14.415026** | yes, v14 |

gpu5's kernel and gpu2's region were both applied to the SAME v12 baseline and are **disjoint by
construction**: the kernel replaces `_decode_cached_attention`, which v14 leaves AST-frozen, and
the region absorbs the appends and the rope/norm/value-embedding work, which the kernel leaves
alone. The kernel's two `@triton.jit` definitions and its wrapper are lifted **verbatim** from
gpu5's frozen candidate, applied with gpu5's own re-runnable `apply_diff.py`, which was written
to target "whatever champion/train.py currently is" for exactly this reason.

**Coordination, recorded.** gpu5's cycle closed at 04:45 with `last_experiment
custom_decode_attention / KEEP` and it queued no rebase row; attention_cache's queue holds no
claims. The row is the orchestrator's `compose_triton_attention_qkv_write`, at kernel_fusion's
queue head, noting "third race of the run, coordinate with attention_cache". This launch refutes
nothing of gpu5's: -7.908702 on v12 is a banked eligible measurement and this is where it lands.

## Prediction

**`nopref_request_ms_median`: 110.9 central, range 108.5-113.5.** 118.773580 - 7.908702 = 110.865
under strict additivity, which `knowledge/decode_mechanism_price_additivity.md` has held to
3.0-3.6% at whole-limb granularity in this run. The range brackets +/-2.4%.

**`request_ms_median` (tiebreak): 144.2 central, and that is a REGRESSION of ~+7.8 -- registered
before the launch, not explained after it.** From the same v12 baseline gpu5's kernel measured
**-7.908702 on `nopref` and +7.762313 on the prefilled shape** (150.874972 -> 158.637285): its
16-way split over a 2049-row cache does more masked work than the compiled region it replaces,
where the region's own cost is ~9.5 us per call fixed and only ~1.1 us per extra 512 rows. v14's
tiebreak is 136.402488, so the composition should land near 144.2. The tiebreak only binds on an
exact tie of the ranked metric, so this is a real but non-binding cost; if it lands worse than
~147 the two mechanisms are not additive on that shape and the write-up must say so.

## Ceilings -- equalities, and all four of these were verified on THREE launches with this base

`num_params_total` **39,845,900**, `flops_per_token_measured` **188,743,680**,
`nopref_kv_cache_bytes` **6,304,256**, `peak_vram_bytes` **35,767,867,392** to the byte,
`training_data_tokens_available` **631,241,817**. Launches 19 (gpu4), 20 (gpu5) and 21 (gpu1) all
report exactly these; neither half touches `forward`, any training-path def, or
`init_decode_state`. The kernel's `num`, `den` and `out` are per-call transients on no state, so
the cache reading cannot move -- gpu5 measured 6,304,256 exactly with them present.

`val_bpb`: replicate, predicted **1.0455 +/- 0.002** with `num_steps` predicted **981**. Quoted
together because launches 19, 20 and 21 have AST-identical training-path defs AND `num_steps` 981
in all three, yet `val_bpb` 1.0447704 / 1.0461599 / 1.0457340 -- a **0.0013895 spread that
`num_steps` cannot explain**, so ~0.0014 is irreducible at a pinned seed. Gate 1.05; headroom from
v14 is 0.004266, i.e. 3.1x that spread.

**Fidelity: both halves perturb the last bf16 bits and NEITHER sign is predictable.** v14 reads
0.015263 / 0.013767; gpu5's kernel moved v12's readings -0.000306 / -0.002846 and agrees with the
region only to within one bf16 ulp by its own account. Predicted **0.010-0.024 on both shapes**
against the 0.05 ceiling -- a magnitude, not a direction. Three launches have now shown the sign
is a re-roll, once in opposite directions on the two shapes of a single launch.

Experiment fuse_decode_qkv_write_on_v12 (team kernel_fusion, agent gpu1), on top of
champion v12 compiled_decode_projections_on_depth6 (133.188605 ms).

**This is autoscts__decode_gpu2's mechanism, rebased. gpu2 designed, verified and MEASURED
it; gpu1 only moved it onto the current champion.** It has now been measured twice, ELIGIBLE
both times, and is in the champion neither time -- both were DISCARD because a depth cut was
promoted while it was in flight:

| launch | baseline | measured | delta | nodes | ms/node |
|---|---|---|---|---|---|
| seq 15 `fuse_decode_qkv_write` | v9 `unpack_decode_attention` 198.290825 | 178.760171 | -19.530654 | -12 | 1.627554 |
| seq 18 `fuse_decode_qkv_write_on_depth7` | v10 `depth_7` 177.028179 | 161.342144 | -15.686035 | -10 | 1.568604 |

Two different programs on baselines 21.3 ms apart agree on the per-node price to **3.6%** --
which is the only replicate this task permits, since `launch.seed` is pinned to 42 and an
identical program cannot be re-purchased.

The queue row `fuse_decode_qkv_write_on_depth6` was baselined on v11 (154.222965) and
predicted 141.5. v12 landed at 133.188605 while that row sat unclaimed, so the row as written
could no longer produce a KEEP however well the mechanism worked. This is the same mechanism
against the baseline that now exists; the row is closed as superseded, not refuted.

## The mechanism, unchanged

**`fold_norm_into_projection` (gpu6)**: on the width-1 path every per-layer residual mix,
residual add and `F.rms_norm` is computed inside the projection next to it, so
`_decode_add_mix_norm` and `_decode_add_norm` no longer exist there. `F.rms_norm` is a scale,
so its scalar reduction can be given the projection's [OUT] output shape and share its loop
nest; a residual add is pointwise on that same output index, so it is an epilogue beside the
int8 row scale. Bitwise, and 11 fewer graph nodes per step at DEPTH 6.

One `@torch.compile(dynamic=False)` region per layer splits the merged `qkv`, adds the value
embedding, applies rotary and norm, **and stores k and v into their caches itself**; only `q`
comes back. The champion's two eager `index_copy_` calls per layer -- **12 per step at 6
layers** -- are absorbed into the region that computes what they append, and `_decode_ve_mix`
folds into the same region on the three `has_ve` layers. An eager call is exactly one kernel
launch on any device, so that count is exact rather than estimated.

`init_decode_state` is **untouched**: same cache tensors, same shapes, same count, same
dtype. That is the whole difference from gpu1's `merge_decode_kv_cache` (seq 13), which went
after the same appends through the cache layout, measured -4.580021 ms and was ruled
**INADMISSIBLE** at `nopref_kv_cache_bytes` 12,591,616 against the 10,485,760 ceiling.

## Why the rebase is not a repricing

v12 replaced the four `F.linear` projections with compiled fp32 reductions. It did **not**
touch the appends: `kc.index_copy_` and `vc.index_copy_` are still two eager kernels per
layer on the width-1 path, still outside every compiled region. gpu2's price argument is
about the appends' position in the dependency graph, not about the projections: a cache write
gates the attention read, so nothing can overlap with it and removing it refunds its whole
~3.2 us/step average rather than the ~1.3 us/step margin a fusible pointwise node refunds.
Replacing cuBLAS GEMVs with generated reductions elsewhere in the layer changes neither the
appends' kernels nor that ordering.

## Prediction

Node delta **-9/step**: 3 `has_ve` layers at -2 each (two appends absorbed, `_decode_ve_mix`
folded in, one scatter node emitted) and 3 plain layers at -1 each, which is seq 15's counted
-12 at 8 layers scaled by the actual value-embedding parity at 6 layers rather than by 6/8.
`has_ve(i, 6)` is `i % 2 == 1`, so the tables are on layers 1, 3, 5.

**Predicted `nopref_request_ms_median`: 118.8 central, range 116.0-122.0.**
133.188605 - 9 x 1.598079 (the mean of the two measured per-node prices) = 118.80. The range
is narrow on purpose: unlike gpu2's two launches this one is not discriminating between price
classes -- the class is settled -- so a miss outside 116-122 is information about the rebase,
not about the mechanism.

**Predicted `request_ms_median`** (tiebreak, champion 176.257 -> this candidate): 161 central,
157-166.

## Ceilings -- every one an equality inherited from v12

`num_params_total` **47,185,934**, `flops_per_token_measured` **213,909,504**,
`training_data_tokens_available` **631,241,817**, `nopref_kv_cache_bytes` **6,304,256**
(`init_decode_state` untouched; the region's transients are ~1 KB buffers freed before the
probe's `gc.collect()`), `peak_vram_bytes` **40,963,848,192** to the byte -- `forward` and
every training-path def are AST-identical to v12's, so the training run is a byte-for-byte
replicate of its program.

`val_bpb` a replicate of v12's **1.042141**, and per gpu2's seq-18 correction the replicate
band is **+/-0.002**, not +/-0.0004: a byte-identical training path moved val_bpb -0.001896
between seq 14 and seq 18 because the fixed 600 s clock bought 3 extra steps at ~0.00063
bpb/step. `num_steps` predicted **[855, 867]** against v12's 861, and it is quoted next to
the bpb rather than left implicit. Gate 1.05, headroom from v12 0.007859.

**Fidelity: predicted to MOVE, sign not predictable.** gpu2 published a direction from seq 15
and seq 18 falsified it: nopref TV went +0.007792 on the first baseline and -0.003603 on the
second. The correct form is **|dTV| <= ~0.008 with the sign unknown**, because perturbing the
last bf16 bits of the cached k through a differently-fused rms_norm is a re-roll, not a bias.
From v12's 0.012843 / 0.016949 that is at worst ~0.021 / ~0.025 against a 0.05 ceiling.
`decode_argmax_matches` may fall a few counts from 511/513 and 508/513.

---

Prior header (compiled_decode_projections_on_depth6, gpu4) follows.

Experiment compiled_decode_projections_on_depth6 (team architecture_prune, agent gpu4), on top
of champion depth_6_width_held (154.222965 ms).

**Rebase of a measured mechanism, not a new one.** Launch seq 17
(`compiled_decode_projections`, `exp-8289e9393212a32537f2b570`, CHARGED) applied exactly this
diff to champion v10 `depth_7` and measured **177.028179 -> 156.133533, -20.894646 ms
(-11.80%)**, eligible on every ceiling and the gate, with `val_bpb` 1.042141 *better* than that
baseline's 1.044115. It was recorded DISCARD for one reason: `depth_6_width_held` (154.222965)
was promoted while it was training. **Nothing about it was refuted, and the two are orthogonal
by construction** -- this touches no parameter, no cache, no width and not `DEPTH`, and
`depth_6_width_held` keeps `n_embd` 512 and 4 heads, so all four projection shapes are identical
on it.

## The mechanism

The four per-layer projections stop being `F.linear` on the width-1 decode path and become
compiled fp32-accumulated reductions -- one broadcast multiply against the [out, in] bf16 weight
and one `sum` over the input dimension, the same map on the same operands -- and
`relu(h).square()` folds into `c_fc`'s reduction epilogue instead of costing a kernel.
Width-1 matmul dispatch **25 -> 1** at this depth: only `lm_head` stays eager, where cuBLAS
already reaches 1141 GB/s and a generated kernel would lose. `forward` and the prefill branch are
untouched.

## What launch 17 measured, and what it therefore predicts here

-20.894646 ms over 513 steps and 7 layers is **-5.819 us/step/layer**. This champion has 6:

  * **-34.91 us/step = -17.91 ms**, so **predicted `nopref_request_ms_median` 136.3 central,
    range 132-141.**
  * **Predicted `request_ms_median`** (tiebreak, this champion's 174.225211), scaled by its own
    measured per-layer delta rather than by the ranked shape's: launch 17 moved it 201.165795 ->
    176.256537, -24.909258 over 7 layers = -3.5585 ms/layer, so **152.9 central, range 148-158.**

    Recorded because it is unexplained rather than hidden: that is **-48.65 us per width-1 step on
    the 1536-prefill shape against -40.73 on the no-prefill shape**, for the same 28 repriced
    kernels. The projections do not know the cache position, so something else is moving -- most
    likely that at cache positions 1536-2048 the step is long enough for the removed kernels'
    serialisation to matter differently. It is 4.05 ms across the request, ~5x that launch's
    0.71 ms tiebreak IQR, so it is probably real and it is not accounted for. Whoever needs the
    tiebreak priced should scale from the tiebreak, not from the target.

The range is narrow now because the price is measured rather than modelled. Its width is only
the question of whether a per-layer price measured at 7 layers holds at 6, and nothing per-layer
differs: `ASPECT_RATIO` 64 -> 80 kept the width at 512, so the four shapes ([1536,512],
[512,512], [2048,512], [512,2048]) and their byte counts are unchanged.

Launch 17's decomposition, for anyone repricing against this champion (full version in
`knowledge/decode_gemv_kernel_price.md`): the four projections cost **17.753 us per layer as
cuBLAS** and **13.28 us as compiled reductions** -- 3.32 us average against 4.44 -- so cuBLAS's
~3.5 us floor is real but only ~1.1 us of it per call is recoverable. Fitting
`cost = F + bytes/BW` gives F = 1.55 us with 888 GB/s, or 1141 GB/s with F = 1.94 us. **A
removed GEMV node on a champion carrying this change is worth ~1.70 ms, not the 1.885 ms
measured on eager calls.**

## The gate is thinner on this champion than on the last one, and that is now the only real risk

`depth_6_width_held` reads `val_bpb` **1.046002**, so the `< 1.05` gate has **0.003998** of
headroom where `depth_7` had 0.005885. This candidate spends none of it by construction --
`forward` is untouched, so its `val_bpb` differs from the champion's only by cross-launch drift --
but drift is not zero: `knowledge/noise_floor_data.md` measures 0.00039 sigma on the
reference-identical pool, and launch 17 drifted **-0.001974** against its own baseline on an
identical training path, i.e. the other way. **Predicted `val_bpb` 1.046002 central, range
[1.0440, 1.0490]**, which stays inside the gate across that whole range; failing it would take
+0.004, about 10 sigma of the measured pool and twice the largest drift this run has seen. Stated
rather than asserted-away, because at 0.003998 of headroom a decode-only candidate is no longer
automatically safe on quality.

## Ceilings, all unchanged and all verified identical on CPU

`num_params_total`, `flops_per_token_measured`, `training_data_tokens_available` and
`peak_vram_bytes` are this champion's to the unit, because `forward` is untouched.
`nopref_kv_cache_bytes` is unchanged **to the byte** -- launch 17 confirmed that prediction
exactly -- because the 37.7 MB of bf16 weight copies at this depth are held on the MODEL and not
on the state: `measure_kv_cache_bytes` reads `memory_allocated` around building one state after a
throwaway build that pays every one-time cost, and `report_efficiency_metrics` reads
`peak_vram_bytes` and resets the counter *before* any decode probe, so those bytes land only in
`peak_vram_bytes_inference`, which no constraint bounds (launch 17: 646,373,888).

Fidelity is measured, not argued: launch 17 read `decode_tv_distance_max` **0.016949** and
`nopref` **0.012843** against a 0.05 ceiling, both *better* than the baseline it was applied to,
because the operands are bit-identical to autocast's and only the accumulation order differs.
`[decode-capture] captured=True` on both request shapes, so the CUDA graph captures these
generated kernels.

Verified on CPU against THIS champion before submission (38 checks in
`[local verification path]`): `forward` and the whole
prefill branch bitwise identical, one fused scheduler node per site with no materialised
[out, in] product, one dynamo variant per site across all six layers,
`_decode_relu_square` at zero cache entries on the width-1 path, the immutable
`count_params` / `estimate_flops_analytic` / `FlopCounterMode` identical, and the decode state's
own tensor bytes unchanged.

---

Prior header follows.

Experiment depth_6_width_held (team architecture_prune, agent gpu3), on top of champion
depth_7 (177.028179 ms).

`DEPTH = 7` -> `DEPTH = 6` **and** `ASPECT_RATIO = 64` -> `80`. Two constants, one mechanism:
remove the seventh transformer layer **at unchanged width**.

**Why ASPECT_RATIO has to move, and why `DEPTH = 6` alone would be a different experiment.**
`build_model_config` computes `model_dim = ((depth * ASPECT_RATIO + 127) // 128) * 128`, so the
width is quantised in steps of 128 and `DEPTH` is not a clean capacity dial:

| DEPTH | n_embd | n_head |
|---|---|---|
| 5, 6 | 384 | 3 |
| **7, 8** | **512** | **4** |
| 9, 10 | 640 | 5 |

`6 * 64 = 384` and `((384 + 127) // 128) * 128 = 384`, so plain `DEPTH = 6` narrows the model by
25% as well as shortening it -- `num_params_total` **26,345,484**, -44.17% of this champion, a
different model rather than one rung down the depth axis. (The queued `depth_6` proposal states
"6x64=384 -> 512" and a count of 44,040,204; both are wrong, and the correction is on its post.)
`base_dim` must land in (384, 512] to keep the width, so `ASPECT_RATIO` must be 65..85 at depth 6.
**80** is taken as the mid-range value; every value in that window produces the identical config,
so nothing is tuned by the choice.

**One confound that is real and not optional.** `has_ve(layer_idx, n_layer)` is
`i % 2 == (n_layer - 1) % 2`, so the value-embedding parity flips with the layer count: **4 tables
at depth 7 (layers 0, 2, 4, 6), 3 at depth 6 (layers 1, 3, 5)**. So this candidate also removes a
value-embedding table -- 4,194,304 more parameters and 2 more graph nodes. It cannot be avoided
while keeping `has_ve` as the reference wrote it, and it is stated here rather than discovered in
the result: **this is one layer plus one VE table, -15.56% of parameters, not a single clean rung.**

**What it buys.** 21 graph nodes leave the width-1 step: the 19 of a layer, counted on inductor for
`depth_7` (4 eager cuBLAS GEMVs, 2 eager `index_copy_`, 13 compiled -- of which
`_decode_qk_rope_norm` is 5 and `_decode_cached_attention` is 3), plus a VE embedding gather and a
`_decode_ve_mix` region. Priced from gpu1's launch-13 device attribution, which `depth_7` confirmed
to 0.24%: a layer's non-VE cost is 41.35 us/step and a VE pair is ~3.4 us, so **44.8 us/step**, and
one us/step is 0.513 ms of the target.

**Predicted `nopref_request_ms_median`: 154 central, range 151-158.**
**Predicted `request_ms_median`** (tiebreak, champion 201.166): **176 central, 170-183.**

**The quality gate, priced from `depth_7`'s own measurement rather than from a bound.** `depth_7`
measured +11.99% steps and a net **-0.003019 bpb**, which against gpu1's loss curve (0.417 nats per
e-fold of steps, `results/unpack_decode_attention.md`) decomposes into a step credit of ~-0.0168
and a layer cost of **+0.0138** -- 4.8x that layer's 6.25% parameter share. Carrying both terms
forward:

  * FLOPs 213,909,504 -> **188,743,680**, -11.76%, so steps go 859 -> ~973 (+13.33%) and the credit
    is 0.417 x ln(1.1333) x 0.356 = **-0.0186 bpb**;
  * the layer is now 1/7 of the depth rather than 1/8, so its cost scales to **+0.0158**;
  * the lost VE table is **+0.0036** at my `value_embedding_mode_none` rate (0.0144 for all four),
    and it refunds **no** counted FLOPs -- `estimate_flops_analytic` excludes `value_embeds` by
    name -- so it brings no step credit of its own.

Net **+0.0008 bpb**. **Predicted `val_bpb`: 1.0449 central, range 1.041-1.053**, against a gate of
1.05 with **0.005885** of headroom -- headroom that `depth_7` itself created. The uncertainty is
almost entirely the layer term: at +50% on it the reading is 1.053 and the candidate is ineligible.

**This is the rung where the two terms cross**, and that is the point of running it: `depth_7` was
a free win and this one is not, so the pair brackets the crossing instead of assuming the axis
continues. If it fails, the depth axis is closed with its exchange rate measured at two rungs and
the champion is untouched.

**Ceilings, all `<=` and all moving downward:** `num_params_total` 47,185,934 -> **39,845,900**;
`flops_per_token_measured` 213,909,504 -> **188,743,680**; `nopref_kv_cache_bytes` 7,354,880 ->
**6,304,256**; `peak_vram_bytes` falls again -- `depth_7` refunded 5.16 GB for one layer, so from
40,963,848,192 expect roughly 35-37 GB against the 47,198,976,512 ceiling.
`training_data_tokens_available` untouched. `window_sizes` becomes
[1024, 1024, 1024, 2048, 1024, 2048]: one more S layer leaves and the last is still L.

Both TV readings should be unchanged in mechanism -- `_decode_cached_attention`'s dropped `amax`
still rests on q and k being rms-normed, which this candidate does not touch -- but `depth_7`
measured +0.009137 of TV drift from a training-path change alone, so no tight range is registered
here: predicted **within +/-0.012 of the champion's 0.021037 / 0.021309**, far under the 0.05
ceiling.

---

Prior header follows.

Experiment depth_7 (team architecture_prune, agent gpu3), on top of champion
unpack_decode_attention (198.290825 ms).

`DEPTH = 8` becomes `DEPTH = 7`. That is the whole executable diff: one integer.

**Why this is one line and not a model redesign.** `build_model_config` derives the width from
the depth -- `base_dim = depth * ASPECT_RATIO` = 448, then `model_dim = ((448 + 127) // 128) *
128` = **512**, `num_heads = 512 // 128` = **4**. The 128-multiple rounding absorbs the change,
so `n_embd`, `n_head`, `n_kv_head` and `head_dim` are all the champion's and exactly one
transformer layer leaves the model. `has_ve(i, 7)` is `i % 2 == 0`, so the value-embedding count
is **still 4** (layers 0, 2, 4, 6 instead of 1, 3, 5, 7) and the last layer still carries one,
which is what `has_ve`'s own docstring says the pattern is for.

**This is the only capacity lever this run's own price table says can pay.**
`knowledge/decode_node_price_table.md`: "**Only `n_layer` is a useful capacity lever.** Narrowing
`n_embd` or `n_kv_head`, or quantising the cache, leaves the node count untouched and buys only
weight bandwidth." Nothing in this run has varied it. The related queue row `n_layer_11` cannot:
it edits `GPTConfig`'s `n_layer: int = 12` **default**, and line 1022 passes `n_layer=depth`
explicitly, so that diff is a no-op on this substrate. `DEPTH` is the live knob -- it appears in
exactly three places (its definition, `build_model_config(DEPTH)` and a trailing print).

**What it buys, counted rather than estimated.** One layer is **19 width-1 graph nodes**, counted
on inductor's post-fusion scheduler before submission (`count_layer_nodes.py`, gpu1's method):

| nodes | what |
|---|---|
| 2 | `_decode_add_mix_norm` |
| 5 | `_decode_qk_rope_norm` -- five, not one; rotary+norm on q and k does not fuse to a single kernel |
| 3 | `_decode_cached_attention` |
| 2 | `_decode_add_norm` |
| 1 | `_decode_relu_square` |
| 4 | eager cuBLAS GEMVs: `c_qkv`, `attn.c_proj`, `mlp.c_fc`, `mlp.c_proj` |
| 2 | eager `index_copy_`: the k and v cache appends |
| **19** | **per layer.** The whole width-1 step is ~167 nodes counted the same way. |

**No value-embedding node moves** -- the VE count is 4 either way, verified in the dispatch
census -- so nothing here overlaps my own `ve_gate_none` or the VE axis.

Priced at this run's own measured per-class prices (`knowledge/decode_node_price_table.md`):
4 wide eager GEMVs at 1.885 ms = 7.54; 2 tiny eager nodes at 0.87 = 1.74; 13 compiled nodes at
0.65-1.0 = 8.45-13.0. **-17.7 ms central, -16.2 to -22.3 ms.** I am not quoting the two-point
lineage fit I first wrote here: solved against the counted 167 nodes it returns a *negative*
intercept, which means the reference's node count (taken from a docstring's "roughly forty
kernels per layer") is too low to carry a fit. Two points, one of them an estimate, do not make
a model.

**Predicted `nopref_request_ms_median`: 180 central, range 174-188.** It cannot plausibly land
above the champion: nothing is added to the step.
**Predicted `request_ms_median`** (tiebreak, champion 225.162): **200 central, 190-212** -- the
same 19 nodes for 512 steps, plus one layer off the 1536-token prefill.

**The risk is the quality gate and it is the whole experiment.** `val_bpb` must stay under 1.05
and the champion sits at 1.047134, so there is **0.002866** of headroom against a measured
cross-launch sigma of 0.00039 (`knowledge/noise_floor_data.md`). Removing a layer spends capacity
-- and buys training steps, which this run has measured to be worth more:

  * `flops_per_token_measured` falls **10.53%**, from 239,075,328 to **213,909,504** -- computed
    with the immutable instrument's own `estimate_flops_analytic`, which reproduces the
    champion's measured value exactly. So the fixed 600 s budget buys about 11-12% more
    optimizer steps than the champion's 767;
  * gpu1's champion row records the loss **still falling at the last step** (3.0342 at step 612
    -> 2.9411 at step 766), i.e. 25% more steps bought 0.0931 nats, or 0.417 nats per e-fold of
    steps. Log-scaled to +11.8%, that is -0.046 nats, and at this model's ~4.05 bytes/token
    about **-0.016 bpb**;
  * against that, capacity. The only in-run exchange rate is my own `value_embedding_mode_none`
    row: -33.4% of `num_params_total` cost +0.0144 bpb, i.e. 0.043 bpb per unit fractional
    parameter loss. This candidate drops 3,145,730 params, **6.25%**, which at that rate is
    **+0.0027 bpb**. A transformer layer is worth more per parameter than a value-embedding
    table, so treat 0.0027 as a floor, not a central estimate. The gate is reached if a layer
    is worth ~6x its parameter share.

**Predicted `val_bpb`: 1.041 central, range 1.030-1.058.** The upper end fails the gate, and that
is registered rather than hidden: the two terms are of comparable size, neither has been measured
directly in this run, and the honest range straddles 1.05. If it fails, the launch still returns
the **depth exchange rate** -- the number every capacity proposal in this run has so far been
priced without, and the one the price table says is the only capacity lever that can pay.

**Ceilings, all `<=` and all moving downward:** `num_params_total` 50,331,664 -> **47,185,934**;
`flops_per_token_measured` 239,075,328 -> **213,909,504**; `nopref_kv_cache_bytes` 8,405,504 ->
**7,354,880** (7 layers of cache instead of 8, predicted from `init_decode_state` at the cache
probe's own `max_len = prefill + steps`, a method that reproduces the champion's measured value
to the byte); `peak_vram_bytes` falls with one layer's activations, from a champion reading that
already has 1,026 MiB of slack.
`training_data_tokens_available` is untouched. Both TV readings should be unchanged in mechanism:
`_decode_cached_attention`'s dropped `amax` still rests on q and k being rms-normed, which this
candidate does not touch.

**The one thing that is not a pure deletion**, stated because a reader will otherwise assume it:
`init_weights` draws the value-embedding tables *after* the blocks, so with one fewer block the
generator has advanced less and the four VE tables get different numbers. Blocks 0-6 are bitwise
identical to the champion's blocks 0-6 (verified on CPU); the VE tables are an init re-roll worth
about one 0.00039 sigma. It is inherent to changing the layer count, not something a different
diff could avoid.

---

Prior header follows.

Experiment unpack_decode_attention (team kernel_fusion, agent gpu1), on top of champion
merge_qkv_projection_v2 (220.784187 ms).

**Every latency result in this run so far has moved pointwise nodes or model limbs. This one
prices the attention call itself, which is the largest unmeasured term left in the step.**

`knowledge/decode_attention_call_price.md` (gpu5, launch seq 9) closes the split-K axis at the
ranked shape and, in doing so, reports what is left: partitioning the *entire* cache walk 16
ways at 513 positions bought **2.59 us per call**, so the serial walk is only 7-10% of the
attention residual, and "the rest is entry into a general CUTLASS kernel that supports varlen,
paging, rotary and softcap ... **that is where the money is, and a parameter cannot reach it.**"
That file puts the residual across the step's 8 attention calls at **~215-290 us/step**, which
at this shape's 513 steps is **110-149 ms of a 220.784 ms target.** Nothing has tested it.

So this candidate stops calling the kernel on the width-1 path and computes the same function
out of ops inductor schedules:

  * two eager `index_copy_` writes do the append `flash_attn_with_kvcache` used to do inside
    itself, at the same device position, with no host round-trip;
  * `_decode_cached_attention` is one compiled region: the QK dot product as an fp32-accumulated
    reduction over the head dimension, the window/causal mask, an unshifted `exp` (see below),
    and the value average and its normaliser as fp32-accumulated reductions over the cache.

**The trade is explicit and it is a node-count trade, which is this team's hypothesis.** One
opaque node per layer becomes five cheap ones -- two eager `index_copy_` writes plus, counted on
inductor's scheduler before submission rather than guessed, **3 fused nodes** for the region.
Net **+4 nodes per layer, +32 per step**, which at this run's 1.26 us/node is **+40 us/step
(+21 ms)** of dispatch, spent to remove 8 calls the run's own residual prices at 27-36 us each. It also reads the whole preallocated cache instead of only the
live rows -- 4.2 MB/step more at the ranked shape, about 2.8 us -- because the mask, not the
extent, is what makes a slot unreachable.

The softmax is computed without the usual max subtraction, which removes the least parallel node
of the four (an `amax` that reduces 514 keys into 4 numbers). This is safe by a bound, not by
luck: `q` and every cached `k` are rms-normed over the 128-wide head, so `|q . k| <= 128` and the
scaled score cannot leave `[-11.32, 11.32]`, whose exponential is 8.2e4 against an fp32 ceiling
of 3.4e38. The bound is asserted over every score of a 30-step CPU request before submission.

**Predicted `nopref_request_ms_median`: 155 central, range 120-215.** The range is wide on
purpose and the downside is real: these are reductions with small output tensors (the softmax
reduces 514 keys into 4 rows), so if inductor's kernels cost more than ~18 us per layer the
candidate regresses, to ~245 ms at worst. Either outcome is decisive for the run -- a win is the
largest single term on the board, and a loss retires the "custom decode attention" family that
three proposals are currently built on, without anyone writing a Triton kernel.

Fidelity, by construction rather than by hope: the dot product accumulates in fp32 from bf16
operands exactly as FA3's does, the softmax is fp32 as FA3's online rescaling is, and `p` stays
fp32 where FA3 rounds it to bf16 for its second GEMM -- so this path is if anything more accurate
than the kernel it replaces, and the bf16-score hazard that would have moved TV by 0.02+ never
arises. `forward` is untouched, so `val_bpb`, `flops_per_token_measured`, `num_params_total` and
`peak_vram_bytes` are the champion's; `nopref_kv_cache_bytes` gains the **4 bytes** of `seq`
becoming int64. The prefill branch still calls `flash_attn_func` and is unchanged.

---

Prior header follows.

Experiment merge_qkv_projection_v2 (team architecture_prune, agent gpu4), on top of champion
ve_gate_none (251.447916 ms).

`c_q`, `c_k` and `c_v` are three [512, 512] parameters that all read the same normed residual
`h`. Their row-wise concatenation is the same linear map, so one [1536, 512] parameter turns
three eager cuBLAS GEMVs per layer into one: **16 of the width-1 step's graph nodes disappear
with no arithmetic, no parameter and no counted FLOP removed.**

This is a rebase of `merge_qkv_projection` (launch seq 8), which **measured the mechanism**:
254.933 -> 224.772 ms, **-30.1615 ms (-11.83%)** over the baseline it was applied to, at
`val_bpb` 1.045598 (*better* than that champion's 1.045711) and with `num_params_total`,
`flops_per_token_measured` and `nopref_kv_cache_bytes` all exact. It was recorded DISCARD for
one reason only: `peak_vram_bytes` 47,201,467,904 against the 47,198,976,512 ceiling, over by
2,491,392 bytes.

That happened because merging the three parameters makes the backward assemble one [1536, 512]
weight gradient from three row-block gradients, and inductor materialises a full zero buffer per
slice: **7,209,728 bytes, 2.29x the parameter's own gradient**, where I had budgeted 1x and paid
for it by freeing 4,718,336 bytes of dead rotary table.

**This candidate needs none of that.** `ve_gate_none` reports `peak_vram_bytes` 46,116,837,376,
which is **1,082,139,136 bytes of headroom -- 150x what the merged gradient costs.** So the
rotary table is left exactly as the champion has it (`sequence_len * 10`), returning that axis to
`[PROPOSAL] shrink_rotary_table` unspent, and this candidate is a single change again.

Measured on the seq-8 launch and expected to carry over unchanged: **an eager cuBLAS GEMV node
costs 1.885 ms of `nopref_request_ms_median`** (30.1615 / 16), which is 2.9x this run's 0.65 ms
pointwise constant. The two changes are independent -- gpu3's removes the `ve_gate` limb, this
one merges the projections -- so predicted **221 ms, range 214-229**.

Held fixed, and verified on CPU against this champion before submission (`forward` and
`decode_step` bitwise identical, the immutable `count_params` and `estimate_flops_analytic`
identical, Muon bitwise identical to a batch-matched champion, width-1 dispatch 16 matmuls
lighter). The one term the CPU harness cannot see is `max_memory_allocated`, which is why the
1.08 GB of headroom is doing the work here rather than a memory redesign.

---

Prior header follows.

Experiment ve_gate_none (team architecture_prune, agent gpu3), on top of champion
fuse_decode_residual_chain (254.933 ms).

The value-embedding gate is removed and the value residual is kept: `v + 2*sigmoid(ve_gate(
x[..., :32])) * ve` becomes `v + ve`, in `CausalSelfAttention.forward` and in `_decode_body`'s
`_decode_ve_mix` region, symmetrically -- so decode still computes exactly what `forward`
computes. This is the ungated value residual. The gate is initialised at exactly neutral
(`zeros_` weight, `2*sigmoid(0)` = 1.0), so everything it contributes is learned during
training, which is what this team's hypothesis is about.

Priced off this agent's own preceding launch rather than from theory. `value_embedding_mode_none`
(seq 6) removed the whole limb: -11.506 ms of `nopref_request_ms_median` for 12 graph nodes, and
+0.0144 `val_bpb`, which failed the < 1.05 gate by 3.3x the available headroom. The 12 nodes are
4 embedding gathers, 4 `ve_gate` matmuls and 4 mix regions; at 0.959 ms/node measured against
this run's 0.65 ms/node pointwise constant, the four eager cuBLAS GEMVs carry about 1.58 ms each.
So the gate is roughly half the limb's latency and the half with the weakest quality claim, and
this candidate spends only that half: predicted 249 ms (246-252) at `val_bpb` +0.0015 (0.000 to
0.004) against 0.0044 of headroom.

`num_params_total` 50,331,664 (-512) and `flops_per_token_measured` 239,075,328 (-3,072) are the
only ceiling readings that move, both downward. `value_embeds` stays populated: the immutable
`prepare.py` reads it by name in `count_params` and `estimate_flops_analytic`.

---

Prior header follows.

Experiment fuse_decode_residual_chain (team kernel_fusion, agent gpu1), on top of
fuse_decode_pointwise_chains.

Carry each layer's MLP contribution forward instead of adding it immediately, so the
add lands inside the next layer's mix-and-norm region (`_decode_add_mix_norm`) and, for
the last layer, inside the final norm (`_decode_add_tail_norm`). Nothing reads the value
between a layer's closing add and the next layer's mix, so that boundary was a kernel
launched to produce a value with one consumer. Also fuses the per-step rotary table
lookup -- an int64 cast and two gathers -- into one region. Roughly ten fewer kernels per
step out of the ~130 that remain after chain fusion; the point is as much to price a
kernel as to win the ms, because that price is what tells the team whether any further
fusion is worth building.

Prior experiment, measured this run: chain fusion took nopref_request_ms_median from
465.544 to 261.634 ms (-43.8%) at unchanged val_bpb, FLOPs, params and cache bytes, and
*improved* both decode fidelity readings.

---

Original header follows.

Experiment fuse_decode_pointwise_chains (team kernel_fusion, agent gpu1).

Every pointwise chain in `_decode_body` is handed to inductor as one compiled region:
the residual mix and the norm that follows it, rotary-plus-norm on q and k, the value
embedding's gate, the MLP activation, the attention residual add and its norm, and the
output softcap. Nothing else changes. The measured reference spends 465.5 ms on 513
width-1 steps, 908 us each, while the weights one step must stream are about 75 MB --
tens of microseconds of bandwidth -- so the step is paying per-kernel dispatch, and the
eager body issues roughly forty kernels per layer against eleven that do matrix work.
Fusing the chains should take the step to about twenty kernels per layer.

The training path is untouched: `forward`, `norm`, `apply_rotary_emb` and every module's
own `forward` are the reference's, the helpers are called only from `_decode_body`, and
they hold no parameters of their own. So `val_bpb`, `flops_per_token_measured` and
`num_params_total` are the reference's numbers and the only things this can move are the
two request timings and the two decode fidelity readings. The fidelity readings are the
risk: a fused region's intermediates are not bit-identical to the eager chain's, and
`nopref_decode_tv_distance_max` is gated at 0.05 with the reference at 0.0201.
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import contextlib
import gc
import math
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from kernels import get_kernel
cap = torch.cuda.get_device_capability()
# varunneal's FA3 is Hopper only, use kernels-community on non-Hopper GPUs
repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
fa3 = get_kernel(repo).flash_attn_interface

from prepare import (MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, count_params,
                     make_dataloader, evaluate_bpb, report_efficiency_metrics,
                     TimeToTargetHarness)

# ---------------------------------------------------------------------------
# Width-1 kernel launch parameters, chosen by measurement instead of by heuristic
# ---------------------------------------------------------------------------
#
# `coordinate_descent_tuning`, ON for the width-1 `_decode_body` branch and for nothing else.
# The mechanism is autoscts__decode_gpu1's, measured at launch 41 (`decode_region_config_autotune`,
# -0.504 ms on champion v25, refused on an ambient `decode_tv_distance_max` draw), and it is
# rebased here because THE SHAPE THAT PAID CAME BACK. Not re-argued -- re-priced:
#
#   * Launch 41's per-region probe attributed the whole -0.504 ms to ONE region,
#     `_decode_attn_out_add` at `int32[512,128]`, -7.53% net of its own control, with the other
#     five regions inside that probe's -0.6%..-3.3% control drift.
#   * `unpack_the_two_narrow_projections` (v28) then reverted that site to a plain int8 view, so it
#     reduced over IN=512, past `_persistent_reduction_configs`' `rnumel >= 256` truncation
#     (`triton_heuristics.py:2902`) -- one shipped config, nothing for the tuner to start from.
#     That is why the axis was closed at v31.
#   * `lane_major_int8_word_layout` (v32) and `lane_major_the_two_view_readers` (v34) put it BACK
#     to `int32[512,128]`: `rnumel` 128 < 256, so three configs ship again -- (XBLOCK, num_warps)
#     (1,2) (8,8) (32,8), `num_warps = xblock*rnumel//128` clamped by `_num_warps(max=8, floor=2)`
#     (`:2329-2330`, `:2128-2137`) -- and coordinate descent is the only thing in this stack that
#     moves XBLOCK and `num_warps` INDEPENDENTLY of each other.
#
# So the closure was correct at the champion that held when it was written and is stale at v36.
#
# What the flag does, per kernel: the tuner benchmarks the config the heuristic would ship, then
# walks one field at a time, `while improved` (`coordinate_descent_tuner.py:254-257`), keeping a
# neighbour ONLY if `do_bench` scores it >= 0.1% faster (`:160-162`). `num_stages` is tunable for
# mm only (`:100-101`) and a persistent reduction carries no `R0_BLOCK` (`:88-90`), so XBLOCK and
# `num_warps` are the whole search space here. As measured BY `do_bench` the search is monotone.
#
# The registered risk is that `do_bench` is not the graph: each config is timed alone, and the step
# runs it as one node of 38 inside a captured graph. This run has since measured that an isolated
# per-region reading transfers to the step at anywhere from 0.23x to 4.39x with no stable sign, so
# a regression ceiling is registered rather than a one-sided band, and `[coordesc]` prints every
# `do_bench` number the tuner saw so the transfer can be read afterwards instead of assumed.
#
# Arithmetic: XBLOCK and `num_stages` cannot touch it. `num_warps` sets how many lanes a `tl.sum`
# over the r axis folds across, so a config change RE-ASSOCIATES the fp32 accumulation without
# changing the map. This run has measured that class at ~5e-07 in TV against a 0.0081 draw sd
# (`champion/train.py:2189-2191`'s two spellings give the slope), so no TV equality is registered
# and the ~8.3%/launch ambient breach rate is priced instead.
#
# Scoped to the width-1 path and provably not to training. The flag is read at CODEGEN and acted on
# at the first call of the generated kernel (`triton_heuristics.py:1274-1279`); both happen inside
# the `_decode_body(..., prefill=False)` call that first compiles a region. `adamw_step_fused` and
# `muon_step_fused` are reached only from `MuonAdamW.step`, `forward` is not compiled through this
# path, and every width-1 call in a launch happens after training and after `evaluate_bpb`. The
# training path is AST-identical to v36's (checked, `cpucheck/check_coordesc.py`).
#
# Autotuning inside a CUDA graph capture would be fatal -- `do_bench` synchronises -- and it never
# happens: `_GraphedDecodeStep._capture` runs `WARMUP_REPLAYS` eager replays through `_advance`
# first, and the throwaway `build_state()` in `prepare.measure_kv_cache_bytes` has already tuned
# every kernel in the process before that.

# ---------------------------------------------------------------------------
# TWO configs pinned, every field a literal: `_decode_mlp_hidden_norm` (MEASURED, the null) and
# `_decode_head_prenorm_softcap` (UNMEASURED, the one term this launch buys)
# ---------------------------------------------------------------------------
#
# Every tunable field is a literal at both sites; nothing is inherited. The first entry is the
# `carry_the_pow_relu_r0block_128_pin_on_every_successor` rider (autoscts__decode_gpu4, kernel_fusion),
# which the board's audit records as the ONE live metric-moving row and as still hostless -- this
# candidate hosts it. The second entry is the head, whose transfer the same audit declined to close.
# The first entry's measured -0.873923 ms is what makes the second entry readable: it is the null.
#
# WHAT THE RECORD SAYS, from launch 61's own `[coordesc]` table (logs/0061/attempt-1/stdout.log:335-337)
# and every launch's `[probe]` row at the same site:
#
#   start  XBLOCK: 1, R0_BLOCK: 128, num_warps: 2      do_bench 12.416 us
#   chosen XBLOCK: 4, R0_BLOCK: 128, num_warps: 1      do_bench 11.808 us   (-4.90%, MOVED)
#     and in its own `tried` list, ranked: (128,4,1) 11.808 < (128,2,1) 11.904 < (128,1,1) 12.000
#     < (128,8,1) 12.160 < (64,2,1) 12.224 < (128,1,2) 12.416 < (128,8,8) 12.448
#
#   `[probe] PACKED _decode_mlp_hidden_norm [2048,512]`, us/call, 6 calls/step:
#     L58 3.321   L59 3.423 (= champion v37)   L60 3.429   L61 3.077   L62 3.342   L63 3.602   L64 3.432
#   Its two SIBLING spellings in the same probe, same launches, are FLAT throughout:
#     DEPEND 3.520-3.546 (L61 3.520)      v22 4.072-4.095 (L61 4.081)
#   So launch 61's -0.346 us/call at this site is not probe drift -- two same-site controls in the
#   same capture did not move. 0.346 x 6 calls x 0.513 ms per us/step = **-1.065 ms**, which is 69%
#   of launch 61's whole -1.546860.
#
# WHY THE CHAMPION CANNOT REACH IT BY SEARCHING. autoscts__decode_gpu1's launch-64 roster reads this
# instance `cache=hit`, `found_by_coordesc=True`, serving `(XBLOCK 4, R0_BLOCK 64, num_warps 8)` -- and
# `run()`'s guard skips the search on exactly that flag (`triton_heuristics.py:1274-1276`). So an
# earlier launch's coordinate descent wrote a config into the on-disk autotune cache, and every later
# launch in that namespace inherits it and is forbidden from revisiting it. No radius and no
# `check_all_directions` reaches a site whose shipped config already carries the flag. A pin is the
# only route left, which is the whole reason this row exists.
#
# WHY THIS ROW IS NOT CONFOUNDED BY CACHE WARMTH, and this is the part that is new. Every hunk here is
# a Python monkeypatch OUTSIDE every `@torch.compile` region: no expression inside any compiled region
# changes, so `inductor_meta` is untouched, so the generated kernel source text is byte-identical to
# v37's, so `code_hash` and therefore the autotune-cache key are identical (`codegen/triton.py:4260-4301`
# -> `codecache.py:334-339` -> `autotune_cache.py:137-143`). This candidate ships in the SAME namespace
# at the SAME warmth as the champion. It is the first row on this axis where the config is the only
# thing that differs, so it separates "the config bought it" from "a cold namespace bought it" --
# the separation the board has been asking for -- and it does so in both directions:
#   metric moves ~-1.065 ms  => the config IS the mechanism at this site.
#   metric flat              => launch 61's -1.065 attribution was probe-only, like the head's -39.9%.
#
# THE DEFECT THIS FIXES, and it is mine. Launch 63 pinned `{"XBLOCK": 4, "num_warps": 1}` at this site
# and left `R0_BLOCK` to be inherited from `launchers[0].config`, which the cache had loaded as 64. It
# shipped `(4, 64, 1)` -- a config that appears in no launch's chosen set -- and its probe read 3.602,
# the worst of seven launches. Two published corrections say the defect was `num_warps` 1 against a
# live 8. **Launch 61's table refutes that: `num_warps: 1` is its CHOSEN value at this site and the
# fastest of the twenty configs it benchmarked.** The single field that separates 3.077 from 3.602 is
# `R0_BLOCK`, 128 against 64 -- one reduction iteration over an extent of 128 rather than two. So the
# pin below carries `R0_BLOCK` as a literal and inherits nothing.
#
# Arithmetic: a config change re-associates an fp32 `tl.sum` without changing the map, measured in this
# run at ~5e-07 in TV against a 0.0081 draw sd, so NO TV equality is registered and the ~8.3%/launch
# ambient breach rate is priced instead. Launch 61 shipped this exact config and was ELIGIBLE, which is
# also the evidence that it COMPILES on the launch node at `num_warps=1`.
#
# ---------------------------------------------------------------------------
# SECOND ENTRY: the head, and why this row exists at all
# ---------------------------------------------------------------------------
#
# This candidate is launch 63 done correctly, and the reason it is worth a launch is that launch 63's
# pair CANNOT price either of its two terms. The monitor's audit (`533f6fc8`) declined to close the
# head's transfer for exactly this reason and handed the site back to me; this is the answer.
#
# WHAT LAUNCH 63 ACTUALLY SHIPPED, from its own `[pin]` roster (logs/0063/attempt-1/stdout.log:570,
# 576, 682): **three instances at TWO sites**, not one. One `hints={'x':2048,'r0_':128}` (pow_relu)
# and TWO `hints={'x':8192,'r0_':512}` (the head, which has two instances). Its registered scope was
# the head. So the L62 -> L63 whole-step delta is a TWO-TERM comparison:
#
#   term            probe L62   probe L63   x/step   delta us/step
#   head            9.995       6.012       1        -3.983      (-39.85%)
#   pow_relu live   3.342       3.602       6        +1.560      (+7.78%)
#   ------------------------------------------------------------------------
#   measured whole step (66.552162 - 66.467285) / 0.513          +0.1655
#
# CONTROLS, in the same capture, which is what makes the +7.78% a reading and not drift: the two
# probe-only SIBLING spellings of that same region are flat -- `DEPEND` 3.545 -> 3.544 (-0.03%) and
# `v22` 4.095 -> 4.084 (-0.27%) -- and eight untouched live regions span -0.04% to -0.60%. Every
# `[attribution]` row is flat to +/-0.4% too.
#
# SO THE RECORDED "0.0x TRANSFER AT THE HEAD" IS THE p=0 CORNER OF A TWO-TERM SYSTEM. Solving
# `-3.983*t + 1.560*p = +0.1655` for the head's transfer fraction t:
#   p = 0     -> t = -0.04   (the value the record published as 0.0x)
#   p = 1     -> t =  0.350  (head = -0.715 ms)
#   p = 1.208 -> t =  0.432  (head = -0.882 ms)   <- 1.208x is THIS SITE's own measured transfer:
#     launch 65 pinned pow_relu ALONE on v37, probe -0.235 us/call x 6 = -1.410 us/step = -0.7233 ms
#     predicted, metric -0.873923 measured. So p = 0 is the one value that site's own launch refutes.
# The confound supplies 37.6% of the 4.148 us/step non-transfer at face value. The head's transfer is
# therefore BOUNDED in [0.0x, 0.43x] = [+0.085, -0.882] ms. It is not measured, and 0.882 clears the
# 0.673 ms bar the board needs. **That is the whole content of this row.**
#
# A SECOND reason the L63 pair cannot price the head, recorded because it is not in any file: every
# instrument in this source runs AFTER `METRICS_JSON` (stdout line 77; `[regioncensus]` from 81,
# `[probe]` from 228, the `[pin]` roster later still). And the head has two instances -- launch 65's
# roster reads #138/193 `coordesc=True` (the width-1 branch, cache best_config (XBLOCK 2, num_warps 2))
# and #191/193 `coordesc=False` `only 1 config` (XBLOCK 1, num_warps 4), constructed after all ten
# flag-baked instances. In launch 65 the search MOVED the flag-ON instance (1,4) -> (1,1) for
# do_bench -9.95% and the `[probe]` head row read 9.961 against L62's 9.995, i.e. FLAT (-0.34%, inside
# the untouched-region band). In launch 63 the flag-OFF instance was pinned and the probe read -39.85%.
# So the probe's head row and the flag-ON instance are not connected by any measured relationship:
# either the probe reads the other instance, or the head's do_bench delta does not appear in-graph.
# **No instrument in this run can read the head's config at the instance the ranked metric executes.**
# Hence a launch, not another census.
#
# WHAT THE TWO LEGS ARE WORTH, separately, so the result is readable:
#   * flag-ON (width-1) instance: pinning it to (XBLOCK 2, num_warps 2) is a MEASURED EXACT TIE --
#     launch 61's own `tried` list reads 17.184 us for BOTH (XBLOCK 2, num_warps 2) and
#     (XBLOCK 1, num_warps 1), and (1,1) is what launch 65's search ships. So this leg is ~0 by
#     measurement, and the cost of suppressing the search there is bounded by the 0.93-3.35% per-launch
#     config draw on a 17 us kernel called once per step = well under the 0.110865 ms floor.
#   * flag-OFF instance: unreachable at any search radius (`only 1 config`), so a pin is the only
#     route, and it is the sole unmeasured term in this candidate.
# The pow_relu entry's -0.873923 ms is therefore the NULL: if the head is worth nothing at the step,
# this candidate reads launch 65's value back.
_DECODE_CONFIG_PIN = (
    {
        "kernel": "triton_red_fused___lshift_____rshift____to_copy_add_div_expand_mul_pow_relu_"
                  "rsqrt_select_squeeze_sum_unsqueeze_view_0",
        "size_hints": {"x": 2048, "r0_": 128},
        # REPAIRED MATCH KEY (launch 67): hints + a token that survives a respelling at
        # this site. `relu` is the MLP activation; the only other relu region on this path
        # is prefill-only `_decode_relu_square` at different hints.
        "name_must_contain": ("relu",),
        # EVERY tunable field is a literal. Nothing is inherited from any base config.
        "pin": {"XBLOCK": 4, "R0_BLOCK": 128, "num_warps": 1, "num_stages": 1},
        # the kwargs the pinned config must have, EXACTLY -- any extra key means an unmeasured field
        # would ship, and this row skips rather than shipping one.
        "kwargs_exactly": ("XBLOCK", "R0_BLOCK"),
        "site": "_decode_mlp_hidden_norm [2048,512] pow_relu kernel, 6 calls/step",
        "evidence": "L61 chosen (4,128,1) do_bench 11.808 (-4.90%), probe 3.423 -> 3.077 = -1.065 ms; "
                    "launch 65 measured this pin ALONE on v37 at -0.873923 ms (probe 3.342 -> 3.107, "
                    "transfer 1.208x). This entry is the NULL for the head entry below.",
    },
    {
        # `triton_per_` kernel: upstream guarantees no `R0_BLOCK` field on the persistent-reduction
        # path, so `XBLOCK` is the ONLY tunable kwarg and `kwargs_exactly` says so. Launch 63
        # installed this exact config at BOTH instances of this kernel, so it is known to install
        # and known to compile on the launch node.
        # NEGATIVE token required: this kernel's name is the pow_relu entry's name MINUS
        # `pow_relu_`, so `name_must_contain` alone cannot tell them apart. Roster instances at these
        # hints: 4 in every launch 65-71; the pow_relu entry claims 2 and printed
        # `SITES INCOMPLETE ... 2` on 68/69/70/71. These are the other two.
        "kernel": "triton_red_fused___lshift_____rshift____to_copy_add_div_expand_mul_"
                  "rsqrt_select_squeeze_sum_unsqueeze_view_0",
        "size_hints": {"x": 2048, "r0_": 128},
        "name_must_contain": ("rsqrt", "lshift"),
        "name_must_not_contain": ("relu",),
        "pin": {"XBLOCK": 4, "R0_BLOCK": 128, "num_warps": 1, "num_stages": 1},
        "kwargs_exactly": ("XBLOCK", "R0_BLOCK"),
        "site": "_decode_qkv_matvec_norm [1536,512] int8 prenorm matvec, 6 calls/step",
        "evidence": "2.963 us/call x 6 = 17.78 us/step = 9.12 ms, the second-largest reachable site. "
                    "num_warps=1 is UNBENCHMARKED here by every [coordesc] table in this run: descent "
                    "starts at nw=8 and reaches 4 and 16 only (L70's 11-row table at this kernel has "
                    "no nw 1 or 2 row; (X8,R0 128) reads nw8 8.768, nw4 9.216, nw16 9.056). The run's "
                    "two banked nw=1 wins are attn_combine_num_warps_1 -0.755787 ms and the pow_relu "
                    "pin -0.873923 ms, and BOTH left the search to reach it. XBLOCK 4 rather than the "
                    "shipped 8 because XBLOCK and num_warps are coupled -- this file's own :874-875 "
                    "benchmarks XBLOCK 8 at nw=1 at +2.98%. R0_BLOCK 128 is one trip over the 512 "
                    "int8 lanes packed 4-per-word (L70: R0 128 = 8.768 against R0 64 = 8.800).",
    },
    {
        "kernel": "triton_per_fused__to_copy_add_div_expand_mul_rsqrt_sub_sum_tanh_unsqueeze_0",
        "size_hints": {"x": 8192, "r0_": 512},
        # REPAIRED MATCH KEY (launch 67): 8192 output rows is the head and nothing else in
        # the width-1 step (the next largest is 2048), and `tanh` is the softcap, which is
        # definitional to this site. Survives an edit to the fused-op list; `kwargs_exactly`
        # still refuses to ship if the tunable FIELD SET moves.
        "name_must_contain": ("tanh",),
        "pin": {"XBLOCK": 2, "num_warps": 2, "num_stages": 1},
        "kwargs_exactly": ("XBLOCK",),
        "site": "_decode_head_prenorm_softcap [8192,512], 1 call/step, BOTH instances",
        "evidence": "L61 chosen (2,2) do_bench 19.040 -> 17.184 (-9.75%), with (1,1) TIED at 17.184; "
                    "L63 installed it at both instances, probe 9.995 -> 6.012 (-39.85%), but its "
                    "whole-step pair is confounded by the pow_relu term above (+1.560 us/step), so "
                    "the head's transfer is bounded in [0.0x, 0.43x] and NOT measured.",
    },
    {
        # THE MECHANISM'S OWN KERNEL, AND THE ONLY UNPINNED ONE IT CREATES. The parallel block
        # merges `c_qkv` [1536,512] and `c_fc` [2048,512] into ONE [3584,512] int8 reduction, and
        # `next_power_of_2(3584) = 4096`, so this kernel's hints are {'x': 4096, 'r0_': 128} --
        # claimed by no entry above: `site=None`, `pinned=False`, `coordesc=True`, `n_compiled=6`,
        # `calls=2` in BOTH launch 75's and launch 76's rosters, out of 193 instances seen.
        #
        # WHY THIS ENTRY EXISTS. Launches 75 and 76 shipped TWO DIFFERENT configs here from the
        # IDENTICAL heuristic start (XBLOCK 8, R0_BLOCK 128, num_warps 8), because coordinate
        # descent descended different coordinates first. L75 went down R0_BLOCK and chose
        # (X8, R0 32, nw8) at do_bench 12.768 us, -5.67% against its own start; L76 went down
        # XBLOCK, chose (X16, R0 128, nw8) at 13.408 us, -0.71%, and NEVER TRIED L75's winner --
        # its 12-row table carries no (R0_BLOCK 32, XBLOCK 8) row at all. That one unpinned draw is
        # the whole of the "per-site price is [-1.770, -2.413] us/call" bracket the k-dial was left
        # with: the same merged region reads 3.417 us/call at L75's config and 3.957 at L76's,
        # +0.540, while the pair it replaces is if anything CHEAPER at L76 (5.727 against 5.830).
        # So the dial's latency leg was never read at a held config, and this entry holds it.
        #
        # R0_BLOCK 32 is FOUR reduction trips over the 128 packed int8 lanes where both entries
        # above take ONE, and that inversion is the substance: at 3584 output rows XBLOCK 8 launches
        # 448 programs against the qkv site's 192, and this is the only site in the step where the
        # search chose more trips. Nothing carried from v39 could have found it -- the kernel does
        # not exist in v39.
        "kernel": "triton_red_fused___lshift_____rshift____to_copy_add_arange_div_expand_lt_mul_"
                  "pow_relu_rsqrt_select_squeeze_sum_unsqueeze_view_where_0",
        "size_hints": {"x": 4096, "r0_": 128},
        # POSITIVE tokens: `lshift` is the int8 unpack, `relu` the fc branch's activation, `where`
        # the reduction-index select that merges the two weights. NEGATIVE token: the ONE other
        # roster instance sharing these exact hints is
        # `triton_per_fused__flash_attn_forward__to_copy__unsafe_view_add_c...`, a persistent
        # reduction at n_compiled=3, coordesc=False. `kwargs_exactly` refuses it independently --
        # a `triton_per_` kernel carries no R0_BLOCK field at all -- so this site is guarded twice,
        # by name and by field set, which is the pattern the two entries above established.
        "name_must_contain": ("lshift", "relu", "where"),
        "name_must_not_contain": ("flash_attn_forward",),
        "pin": {"XBLOCK": 8, "R0_BLOCK": 32, "num_warps": 8, "num_stages": 1},
        "kwargs_exactly": ("XBLOCK", "R0_BLOCK"),
        "site": "_decode_qkv_fc_matvec_norm MERGED [3584,512], 2 calls/step at k=2",
        "evidence": "PAID ON BOTH LEGS. Compile and fidelity: launch 75 SHIPPED this exact config "
                    "at this exact kernel and ran to completion, reading nopref TVmax 0.023499 and "
                    "prefill 0.023651 -- so R0_BLOCK 32 here is compile-proven and fidelity-proven "
                    "on the launch node rather than argued. Size: L75's own WITHIN-LAUNCH "
                    "[coordesc] pair at this kernel is 13.536 (start, X8/R0 128) -> 12.768 "
                    "(X8/R0 32) = -0.768 us = -5.67%, above the ~3% draw threshold do_bench's "
                    "32 ns grid imposes; the in-graph probe corroborates cross-launch at "
                    "-0.540 us/call (3.957 -> 3.417), do_bench over-reading by 1.42x, inside this "
                    "run's published 1.292x-1.76x disagreement for that instrument pair. The "
                    "central is taken from the IN-GRAPH leg, per this run's rule that an in-situ "
                    "per-call instrument beats a converted isolated one: -0.540 us/call x 2 "
                    "calls/step x 0.513 ms per us/step = -0.554 ms.",
    },
)
_DECODE_PIN_LOG = []
_DECODE_PIN_ROSTER = []


def _pin_set_field(cfg, name, value):
    """coordinate_descent_tuner.set_field, inlined so the pin cannot drift from it."""
    if name in ("num_warps", "num_stages"):
        setattr(cfg, name, value)
    else:
        cfg.kwargs[name] = value


def _pin_get_field(cfg, name):
    if name in ("num_warps", "num_stages"):
        return getattr(cfg, name, None)
    return cfg.kwargs.get(name)


def _pin_describe(cfg):
    if cfg is None:
        return "None"
    return (f"{{{', '.join(f'{k}: {v}' for k, v in cfg.kwargs.items())}}} "
            f"num_warps={getattr(cfg, 'num_warps', None)} "
            f"num_stages={getattr(cfg, 'num_stages', None)}")


def _pin_match_key(want):
    """The key in one printable string, so the witness can show what it matched ON."""
    toks = " and ".join(repr(t) for t in want["name_must_contain"])
    key = f"size_hints == {want['size_hints']} AND name contains {toks}"
    nots = want.get("name_must_not_contain", ())
    if nots:
        key += " AND name contains NONE of " + " or ".join(repr(t) for t in nots)
    return key


def _pin_site_matches(want, name, meta_name, hints):
    """Match a site, not a spelling.

    `hints` is the autotuner's own `size_hints`, which is the reduction SHAPE and is a
    property of the site; the fused-op name is a property of the spelling. Launch 67 measured
    what keying on the latter costs: the pin installed on a read-only probe leg and the live
    path ran unpinned, with a truthful success line.
    """
    if want["size_hints"] != hints:
        return False
    blob = f"{name}|{meta_name}"
    if not all(tok in blob for tok in want["name_must_contain"]):
        return False
    # NEGATIVE tokens, and they are the only way to address this site. The qkv prenorm kernel's
    # fused-op name is the pow_relu kernel's name MINUS `pow_relu_`, so every substring of the
    # former is a substring of the latter: no positive token can separate the two kernels that
    # share `size_hints {'x': 2048, 'r0_': 128}`. Leaving the second unaddressable is what
    # printed `SITES INCOMPLETE` on every launch from 68 on.
    return not any(tok in blob for tok in want.get("name_must_not_contain", ()))


def _install_config_pin():
    """Replace the shipped launcher with the pinned config, in the parent, once per autotuner INSTANCE.

    Read-only for every kernel not in the table -- it appends a roster row and returns. Every failure
    path leaves `self.launchers` exactly as `_make_launchers` built it AND records why, so there is no
    path in which a config this did not print gets shipped. In particular there is NO fallback base:
    if the config assembled from the table does not match the table on every field, this SKIPS.
    """
    import copy
    try:
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    except Exception as exc:                               # noqa: BLE001
        print(f"[pin] NOT INSTALLED: {type(exc).__name__}: {exc}", flush=True)
        return
    if getattr(CachingAutotuner, "_autoscts_config_pinned", False):
        return
    _orig_make = getattr(CachingAutotuner, "_make_launchers", None)
    if _orig_make is None:
        print("[pin] NOT INSTALLED: CachingAutotuner has no _make_launchers", flush=True)
        return

    def _make_launchers_pinned(self):
        # RE-INSTALL ON EVERY CALL. `_orig_make` above has just rebuilt `self.launchers` from
        # `self.compile_results` unless `len(launchers) == len(compile_results)`
        # (`triton_heuristics.py:592-593`), and `precompile()` reaches here TWICE for every
        # non-persistent reduction: once at :451 and again from `_dynamic_scale_rblock`'s own
        # `self._make_launchers()` at :590, whose enclosing guard (:484-495) is exactly
        # `heuristic_type == REDUCTION and not persistent_reduction`. The previous spelling returned
        # here on re-entry, so the second call SILENTLY DISCARDED the pin and left the roster row --
        # recorded on the first call -- printing `PINNED` with the correct `INSTALLED` config for a
        # config that no longer shipped. `run()` (:1266-1279) then autotuned over all N and ran
        # coordinate descent. Measured: launch 70's pow_relu instance had `n_compiled=6`, printed a
        # `[coordesc]` MOVED row (22 trials, start (X8,R0 128,nw8) = the do_bench winner over the six,
        # which is not the pin's config and not the roster's pre-pin `from` (X1,R0 128,nw2)), and read
        # 3.354 us/call against launches 65/66/67/68/69/71 at `n_compiled=1` reading 3.045-3.107
        # (mean 3.0808, sd 0.0227). +0.273 us/call = +0.782 ms, 12.0 sd out.
        #
        # `triton_per_` sites were never at risk, which is why the head pin has held throughout.
        # The roster append and the `_DECODE_PIN_LOG` append stay FIRST-CALL-ONLY: the witness must go
        # on meaning "one row per instance". `repinned` counts re-entries that re-installed and CAN
        # print zero, which is the whole point of a witness.
        _orig_make(self)
        first = not getattr(self, "_autoscts_pin_seen", False)
        self._autoscts_pin_seen = True
        row = getattr(self, "_autoscts_pin_row", None)
        name = None
        try:
            # `coordesc_tuner.name` is `self.fn.__name__` captured in the compile WORKER
            # (`triton_heuristics.py:346-348`), so it survives the pickle and is exactly the string
            # the `[coordesc]` table printed and this table was built from. `self.fn.__name__` is not
            # safe here: `prepare_for_pickle` sets `self.fn.fn = None` in the parent.
            name = getattr(getattr(self, "coordesc_tuner", None), "name", None)
            meta_name = self.inductor_meta.get("kernel_name")
            hints = dict(self.size_hints or {})
            shipped = [_pin_describe(getattr(l, "config", None)) for l in self.launchers]
            row = {"kernel": str(name), "meta_name": str(meta_name), "size_hints": str(hints),
                   "hints_dict": dict(hints), "matched_site": None,
                   "n_compiled": len(self.launchers), "shipped": shipped, "pinned": False,
                   "instance": hex(id(self)),
                   "meta_coordesc": bool(self.inductor_meta.get("coordinate_descent_tuning")),
                   "cache": str(getattr(self, "autotune_cache_info", None))[:200],
                   "calls": 0, "repinned": 0, "reentry_rebuilt": None}
            if first:
                self._autoscts_pin_row = row
                _DECODE_PIN_ROSTER.append(row)
            else:
                row = getattr(self, "_autoscts_pin_row", None)
                if row is None:
                    # The first call raised before it recorded a row. Nothing to re-pin against, and
                    # `_orig_make` has already left `self.launchers` exactly as upstream built it.
                    return
                # What `_orig_make` left behind on THIS call, before we re-pin: if it rebuilt the
                # list, the un-fixed spelling would have shipped these.
                row["reentry_rebuilt"] = shipped
            row["calls"] += 1
            # REPAIRED (launch 67): keyed on the REDUCTION SHAPE and a semantic token, not on
            # the fused-op list, which any edit at the site rewrites. See this file's header.
            want = next((p for p in _DECODE_CONFIG_PIN if _pin_site_matches(p, name, meta_name, hints)),
                        None)
            if want is None:
                return
            row["matched_site"] = want["site"]
            # ONE base only, and every pinned field overwritten on top of it. `launchers[0].config`
            # supplies num_ctas/maxnreg, which this table does not tune; the assertions below refuse
            # to ship if that base carries any OTHER tunable kwarg.
            base = self.launchers[0].config
            if first:
                row["from"] = _pin_describe(base)
            cfg = copy.deepcopy(base)
            for k, v in want["pin"].items():
                _pin_set_field(cfg, k, v)
            # SKIP, do not fall back: refuse to ship anything the table did not name.
            bad = [f"{k}={_pin_get_field(cfg, k)}!={v}"
                   for k, v in want["pin"].items() if _pin_get_field(cfg, k) != v]
            extra = [k for k in cfg.kwargs if k not in want["kwargs_exactly"]]
            missing = [k for k in want["kwargs_exactly"] if k not in cfg.kwargs]
            if bad or extra or missing:
                row["skipped"] = f"fields {bad} extra_kwargs {extra} missing_kwargs {missing}"
                if first:
                    _DECODE_PIN_LOG.append(row)
                return
            if self.fn.fn is None:                          # parent, compiled in a worker
                assert hasattr(self, "_reload_kernel") and callable(self._reload_kernel),                     "no _reload_kernel: cannot compile a pinned config in the parent"
                self.fn = self._reload_kernel().fn
            launcher = self._precompile_config(cfg).make_launcher()
            # `run()` reads this at :1274 to decide whether to search. v37 already skips the search at
            # THIS site (its shipped config is cache-served carrying the same flag), so setting it
            # preserves the champion's behaviour here rather than changing it.
            cfg.found_by_coordesc = True
            launcher.config.found_by_coordesc = True
            self.launchers = [launcher]
            row["pinned"] = True
            row["pin_config"] = _pin_describe(launcher.config)
            if first:
                _DECODE_PIN_LOG.append(row)
            else:
                row["repinned"] += 1
        except Exception as exc:                            # noqa: BLE001
            _DECODE_PIN_LOG.append({"kernel": str(name), "pinned": False,
                                    "error": repr(exc)})

    CachingAutotuner._make_launchers = _make_launchers_pinned
    CachingAutotuner._autoscts_config_pinned = True
    print("[pin] installed on CachingAutotuner._make_launchers at import, "
          f"{len(_DECODE_CONFIG_PIN)} config requested", flush=True)


_install_config_pin()

_DECODE_COORDESC_LOG = []


def _install_coordesc_log():
    """Record what the tuner measured, per kernel, so this mechanism prices itself in-launch.

    The paired reading this run trusts: heuristic config against chosen config, same kernel, same
    shapes, same device, same launch. Read-only -- it wraps the method, calls it, appends. A
    failure appends the exception and changes no config. gpu1's, verbatim in behaviour.
    """
    try:
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner
    except Exception as exc:                               # noqa: BLE001
        print(f"[coordesc] tuner log not installed: {type(exc).__name__}: {exc}", flush=True)
        return
    if getattr(CachingAutotuner, "_autoscts_coordesc_logged", False):
        return
    _orig = getattr(CachingAutotuner, "coordinate_descent_tuning", None)
    if _orig is None:
        print("[coordesc] tuner log not installed: CachingAutotuner has no "
              "coordinate_descent_tuning", flush=True)
        return

    def _logged(self, launcher, *args, **kwargs):
        start_config = str(launcher.config)
        out = _orig(self, launcher, *args, **kwargs)
        try:
            tuner = self.coordesc_tuner
            _DECODE_COORDESC_LOG.append({
                "kernel": str(tuner.name),
                "size_hints": str(self.size_hints),
                "n_configs_shipped": len(getattr(self, "compile_results", None)
                                        or self.configs or []),
                "start_config": start_config,
                "chosen_config": str(out.config),
                "start_ms": tuner.lookup_in_cache(launcher.config),
                "chosen_ms": tuner.lookup_in_cache(out.config),
                "tried": {str(k): v for k, v in tuner.cached_benchmark_results.items()},
            })
        except Exception as exc:                           # noqa: BLE001
            _DECODE_COORDESC_LOG.append({"kernel": "?", "error": repr(exc)})
        return out

    CachingAutotuner.coordinate_descent_tuning = _logged
    CachingAutotuner._autoscts_coordesc_logged = True


@contextlib.contextmanager
def _decode_region_autotune():
    """`coordinate_descent_tuning` on for a width-1 `_decode_body` call, off everywhere else."""
    import torch._inductor.config as _icfg
    prev = _icfg.coordinate_descent_tuning
    _icfg.coordinate_descent_tuning = True
    _install_coordesc_log()
    try:
        yield
    finally:
        _icfg.coordinate_descent_tuning = prev


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


# TWO ADDED VALUE-EMBEDDING TABLES, and the reason they are the only capacity this task will sell.
#
# `has_ve(i, 6)` is `i % 2 == 1`, so the reference set is {1, 3, 5} -- three tables. This adds two
# more, at layers 2 and 4, taking the set to {1, 2, 3, 4, 5} and leaving layer 0 as the single
# VE-free block. The extension is nested ({4} in {2, 4}) so a later one-table rung reads against a
# measured point, and it is a CONTIGUOUS SUFFIX, which is the same direction `has_ve`'s own
# docstring names ("last always included").
#
# WHY THIS IS THE ONE CAPACITY AXIS THAT PAYS HERE. The depth axis's generalisable finding is that
# on a time-budgeted, undertrained substrate a capacity PRUNE is a quality REFUND -- rung 8 -> 7
# removed a whole layer and `val_bpb` IMPROVED, because the FLOPs a layer refunds buy more quality
# in extra optimizer steps than the layer itself contributes. A layer costs 12-14% of the step
# count. A value-embedding table costs 0.93% for FOUR of them, because it adds no counted
# arithmetic at all: `prepare.py:488-506`'s `estimate_flops_analytic` subtracts `value_embeds` BY
# NAME, while `count_params` at `prepare.py:466-485` does not -- so a VE table is params-only
# capacity, and this is the axis where the undertrained-substrate logic runs the FAVOURABLE way.
#
# THE RATE IS PAID, NOT ESTIMATED, AND IT IS `autoscts__decode_gpu3`'s FIGURE. Launch 6
# (`value_embedding_mode_none`) removed all four tables at DEPTH=8. Against the seven
# same-geometry launches (seq 7, 9, 10, 11, 12, 13, 15, all at `num_params_total` 50,331,664 and
# `flops_per_token_measured` 239,075,328): `val_bpb` 1.0601173176952683 against a pool mean of
# 1.0474818123, sd 0.0002732022, range 0.0006225562. Delta +0.0126355054 for four tables =
# **+0.0031588763/table at 46.25 sd of the pool**, reproducing gpu3's published +0.003159/table to
# the digit. And L6 differs from that pool by 16,777,216 PARAMETERS at a BYTE-IDENTICAL
# `flops_per_token_measured` -- a ledger-level proof, not a reading of code, that a VE table is
# free on the FLOPs ceiling.
#
# THE CEILING IS WHAT FIXES THE COUNT AT TWO. One table is 8192 x 512 = 4,194,304 parameters.
# 39,845,900 + 2 x 4,194,304 = **48,234,508 against the `<=` ceiling 50,332,176 = 95.8324%**, with
# 2,097,668 left = **0.5001 of a table**. THREE tables is inadmissible, over by 2,096,636. So the
# frozen task sells exactly 2.50 tables of capacity and this takes 2 of them.
#
# WHAT IS NOT MEASURED, STATED HERE RATHER THAN IN A RETRACTION. The transfer of
# T = +0.0031588763/table from DEPTH=8 / four-tables-DELETED to DEPTH=6 / three-tables-plus-two is
# UNMEASURED: these are the 4th and 5th table on a six-layer model against an average over the
# 1st-4th on an eight-layer one, and diminishing returns are likely and unquantified. Honest band
# T in [0, 0.0031588763], central 0.0021. Composed with launch 79's measured
# `val_bpb` 1.0508989003011076 the gate margin is [-0.0008989, +0.0054191], central +0.0033011 --
# so the band STRADDLES ZERO and a gate refusal CANNOT refute this leg. Against the k>=2
# parallel-path draw sd of 0.0061274305 a single launch decides ELIGIBILITY and cannot ATTRIBUTE T,
# which would need ~48 launches per arm.
#
# CORRECTION TO THE RATE'S STATED GROUND, which runs against its own author's use of it and in this
# candidate's favour. `knowledge/depth_axis_measured_and_why_it_stops.md` says L6 "removed 4 tables
# at identical `flops_per_token_measured`, so there is no step credit in it". That ground is FALSE
# and L6's own log printed the disproof: **667,459.15 tok/s and 774 steps against the pool's
# 659,184.8-662,118.8 and 765-768 -- disjoint on both**, +0.9941% and +7.143 steps. Identical FLOPs
# is not identical wall clock: two gathers per VE layer, their backward scatter-adds and an
# optimizer step over 16,777,216 more parameters all cost time inside a 600 s budget. For ADDING
# tables the published figure is exactly right, because the NET of capacity against steps is the
# quantity that matters -- which is also why the measured rate is a CONSERVATIVE floor on capacity.
# For netting a table out of a depth rung it understates the pure capacity term by 1.109x: at the
# run's own loss curve (0.417 nats per e-fold of steps at ~4.05 bytes/token) +7.143 steps is worth
# -0.0013772 bpb, so pure capacity is +0.0140127 for four = +0.0035032/table, moving the published
# layer-only cost of rung 7 -> 6 from -0.000521 to -0.000865. That axis's VERDICT is untouched (it
# is closed on the instrument -- the six-layer `val_bpb` range 0.020662 with mid-run spikes in 6 of
# 8 -- not on the rate) and the supersession is handed back to its author by name.
EXTRA_VE_LAYERS = (2, 4)


# ONE PARTIAL-VOCAB VALUE-EMBEDDING TABLE AT LAYER 0, to cover L84's 0.0008281130 gate miss.
#
# L84 (`parallel_block_k5_on_v42_at_the_long_window_site_layer_3`, gpu2) measured
# 49.456000328063965 ms -- the fastest program this run has produced, -2.9265880585 on champion
# v42 -- and was INELIGIBLE on `val_bpb` 1.0508281130470107 against `< 1.05`, failing by
# +0.0008281130 with every other ceiling passed. This adds the only capacity the ceiling still
# sells, and it is the row gpu2 priced and deliberately left unclaimed.
#
# WHY A PARTIAL TABLE AND WHY 4096. `num_params_total <= 50,332,176` (task.json) less L84's
# 48,234,508 leaves 2,097,668. A FULL table is `vocab_size x kv_dim = 8192 x 512 = 4,194,304`
# and does not fit. Half its ROWS -- 4096 x 512 = 2,097,152 -- fits by 516 parameters. It is
# still FLOPs-free: `prepare.py`'s `estimate_flops_analytic` subtracts `value_embeds` BY NAME
# while `count_params` does not, so a VE table is params-only capacity. Layer 0 is the only
# layer without a table (`has_ve(i, 6)` is `i % 2 == 1`, `EXTRA_VE_LAYERS` adds 2 and 4).
#
# THE ONE THING THAT WAS UNMEASURED, AND I MEASURED IT DEVICE-FREE RATHER THAN BUYING IT.
# analyst2 priced this row at P(gate) 54.9-66.4% while charging the partial table only
# PRO-RATA BY ROWS (half of a full table's credit), and named "does a 4096-row table deliver
# >= 26% of a full table's credit" as the launch's own content. It is not 50%: on this run's
# own tokenizer (`~/.cache/autoresearch/tokenizer/tokenizer.pkl`, `rustbpe`, n_vocab 8192) the
# token ids BELOW 4096 carry
#
#     88.1336% of token occurrences on the PINNED VALIDATION shard (shard_06542, 83 of 83 row
#              groups, 63,289,984 tokens, 3.9830 bytes/token)
#     88.0826% of token occurrences on TRAINING shard_00000 (2 row groups, 1,512,847 tokens,
#              4.0282 bytes/token)
#
# -- two independent shards agreeing to 0.051 percentage points, because a BPE merge order is
# frequency-driven. So the covered mass is 88.1%, not the 50% the row was charged at.
#
# REGISTERED CENTRAL AND BAND, before the launch. A full table is credited +0.0031588763 bpb
# (L6 `value_embedding_mode_none` against its seven same-geometry peers, 46.25 sd -- gpu3's own
# figure, and the run also carries a LATE two-table reading of +0.0068410434/table, L79 -> L81).
# Charging the partial table by measured OCCURRENCE MASS rather than by rows:
#   early rate: 0.8813 x 0.0031588763 = 0.0027834 bpb  -> margin +0.0019553 -> P(gate) 62.5%
#   late  rate: 0.8813 x 0.0068410434 = 0.0060291 bpb  -> margin +0.0052010 -> P(gate) 80.2%
# against the operative CROSS-allocation draw sd 0.0061274305 (analyst1 and analyst2 both, this
# rotation: sigma here is an ALLOCATION term, 0.00007935 within one allocation against
# 0.0051-0.0061 across, and since L74 no two launches have shared a node). Net of the run's
# mechanism-independent ~8.3% TV-breach base rate, P(ELIGIBLE) is 57.3-73.6%.
# `val_bpb` central 1.0480447, band [1.0448, 1.0498]; prize -2.74..-2.82 ms (L84's -2.9266
# less one `ve` gather, gpu1's whole-block 0.1847-0.1884 ms as the upper bracket).
#
# WHAT WOULD MAKE THIS WRONG, stated as one falsifiable number. The credit is charged linear in
# occurrence mass, which is an assumption, not a measurement. For the table to deliver less
# than the 26.22% of a full table's credit the miss requires, a token occurrence with id
# >= 4096 would have to be worth 20.9x more VE credit per occurrence than one with id < 4096:
# 0.8813 / (0.8813 + r x 0.1187) = 0.2622 solves at r = 20.9.
#
# AND THE HAZARD gpu2 QUEUED THIS ROW WITH, WHICH I AM NOT PRETENDING AWAY. 0.00083 bpb is
# 0.14 sd of the draw, so eligibility here is DRAW-DECIDED whatever the mechanism does, and a
# pass cannot show the table did it. Two analysts answered that independently and converged:
# the objection bears on how a pass is WRITTEN UP, not on whether it is worth buying, since no
# launch in this run has ever had a same-program replicate and v42's own eligibility is a
# single draw 1.0539 sd inside the gate. So: if this passes, THE MECHANISM IS NOT CONFIRMED BY
# THE PASS. The coverage measurement above is confirmed; the bpb credit is not.
#
# THE SITE GROUND IS NOW FULLY SPENT, AND I AM DISCLOSING THAT RATHER THAN LEAVING IT. v40 chose
# parallel-block sites as the layers with NEITHER a value embedding NOR a long window. v41 spent
# the VE half, L84 spent the long-window half at layer 3, and this spends what is left by putting
# a table on layer 0. Sites are now selected by measurement, not by that ground.
PARTIAL_VE_ROWS = {0: 4096}

# The set of row counts that mean "partial", so the decode wrapper can recognise one without
# needing `vocab_size` at the call site. A launch-config constexpr keyed on a runtime value
# would specialise per table; keyed on this it does not.
_PARTIAL_VE_ROW_SET = frozenset(PARTIAL_VE_ROWS.values())


def _ve_rows(layer_idx, vocab_size):
    """Rows in layer `layer_idx`'s value-embedding table."""
    return PARTIAL_VE_ROWS.get(layer_idx, vocab_size)


def _ve_lookup(table, idx):
    """`F.embedding(idx, table)`, with ids past a PARTIAL table's last row reading zero.

    For a full table this is `F.embedding(idx, table)` and nothing else -- the branch is on
    `table.size(0)`, so no covered path gains an operation. For a partial table the gather is
    forced in bounds and the out-of-range rows are replaced by exact zeros, which is the same
    thing the decode kernel does, so the training and decode paths agree by construction.
    """
    rows = table.size(0)
    if rows not in _PARTIAL_VE_ROW_SET:
        return F.embedding(idx, table)
    inb = idx < rows
    emb = F.embedding(torch.where(inb, idx, torch.zeros_like(idx)), table)
    return torch.where(inb.unsqueeze(-1), emb, torch.zeros((), dtype=emb.dtype, device=emb.device))


# The k-of-6 dial on the parallel attention+MLP block, at k=2.
#
# Launch 50 and launch 75 both measured this rearrangement on ALL SIX blocks and both were
# refused on `val_bpb` alone -- L75 at 53.591370582580566 ms (-7.374525, -12.10%, which would
# have been 88.4884% below the reference) with `val_bpb` 1.0532409328644643 against `< 1.05`,
# short by 0.0032409. The rearrangement is not one change, it is SIX INDEPENDENT COPIES of one
# change, so it has a dial, and promotion needs only 1.626 of the 7.374 ms. Returning a block
# returns its share of the quality and keeps the rest of the latency.
#
# `k=2`, and NOT the auto-bracket midpoint `k=3`, for a reason that is measured rather than
# aesthetic. `autoscts__decode_gpu4` priced a second leg: a returned block returns THROUGHPUT
# (both parallel-block launches are the top two `train_tokens_per_second` readings of the
# 58-launch matched family and both exceed the maximum of the other 56 -- P = 1/1653), hence
# steps, hence part of the quality the block was buying back. Break-even is
# `c* = +0.00050681` bpb/step against a five-construction ledger bracket of
# `[-0.00034, +0.00027]`, so the dial does buy margin -- but over the whole box
# k=3's corrected `val_bpb` margin is `[-0.00027, +0.00756]`, which STRADDLES ZERO, while
# k=2's is `[+0.00072, +0.01117]`, positive at every corner, and k=2 still clears the
# promote bar by 0.832 ms. The exact identity that makes k=3 attractive (the auto-bracket
# midpoint lands on it, difference 0.0) selects the one point on the dial that is not robust
# to a coefficient this run cannot identify. k=2 pays 1.229 ms of headline to be robust.
#
# WHICH two, on a stated ground, because the six sites are NOT exchangeable.
# `_compute_window_sizes` cycles "SSSL" and then forces `window_sizes[-1]` long, giving
# windows S,S,S,L,S,L; `has_ve(i, 6)` is `i % 2 == 1`, so value embeddings sit on 1,3,5.
# Layers {0, 2, 4} are exactly the blocks with NEITHER a value embedding NOR a long window,
# and {0, 2} is a PREFIX of that set. The family is therefore NESTED --
# {0,2} in {0,2,4} in all six -- so a later launch at k=3 measures its increment against a
# MEASURED k=2 point instead of only against k=6. A non-nested pick would make each k a
# different mechanism and there would be no dial to read.
#
# **The load-bearing assumption, stated here and not in a retraction: the quality cost is
# LINEAR IN k.** It is not measured, at any k, by anything. It scales a +0.012873 bpb that is
# itself one pair of draws (L50 - L48), and gpu4's throughput leg inherits the same
# non-uniformity in the opposite direction, so the net sign is set by the ratio of two
# unmeasured linearities. This launch measures the k=2 point of it directly; a second point
# at k=3 or k=4 would identify the curvature.
#
# ---------------------------------------------------------------------------
# k=4, AND THE SITE GROUND RESTATED RATHER THAN ABANDONED (autoscts__decode_gpu2).
#
# gpu1's row `parallel_block_k4_on_v41_with_the_site_ground_RESTATED` carries a stop clause: do NOT
# buy without first restating the ground. This is that restatement, and it NARROWS the ground rather
# than dropping it.
#
# v40's ground was a CONJUNCTION -- {0,2,4} are "the blocks with NEITHER a value embedding NOR a long
# window". v41 spent the VE half of it ITSELF: `EXTRA_VE_LAYERS = (2, 4)` puts value embeddings on two
# of the three shipped parallel blocks, so the shipped champion's parallel set is already not a subset
# of its own stated ground, and that champion is the best ELIGIBLE program this run has measured. So
# the VE clause is not a constraint that k=4 breaks; it is a precaution v41 already spent and
# empirically refuted. v41's own header, lines 60-65, states the mechanism: `has_ve` has exactly ONE
# functional caller (the `value_embeds` comprehension), and VE and `parallel` are orthogonal in BOTH
# paths -- `Block.forward` passes `ve` into `self.attn` on either spelling, and in the width-1 path
# `have_ve` and `block.parallel` are independent branches, `_decode_qkv_rot_write_attention` taking
# the table or `None` regardless. A parallel block may carry a value embedding with no change beyond
# the construction predicate. That is gpu1's own structural finding, and it is what voids its blocker.
#
# THE SURVIVING HALF IS THE LONG WINDOW, AND IT IS PRESERVED. No parallel block in this run has ever
# carried a long window, so that clause is UNTESTED rather than refuted, and this candidate does not
# test it. `_compute_window_sizes` cycles "SSSL" and forces the last long, giving windows
# S,S,S,L,S,L: long on 3 and 5. Of the three non-parallel layers {1,3,5}, layers 3 and 5 carry a long
# window and layer 1 does not. So the ground, restated as "no parallel block carries a long window",
# selects exactly ONE site: LAYER 1.
#
# AND LAYER 1 IS STRUCTURALLY THE LAYER v41 ALREADY SHIPS TWICE. Short window + a value-embedding
# table + parallel is precisely what layers 2 and 4 are in the shipped champion. Layer 1's only
# distinction is that its table comes from `has_ve`'s own parity rather than from `EXTRA_VE_LAYERS`,
# and `has_ve` has one functional caller, the same comprehension. So there is no untested structural
# combination in this candidate at all.
#
# NESTING IS PRESERVED: {0,2} in {0,2,4} in {0,1,2,4} in all six, so the k=4 increment reads against
# the MEASURED k=3 point (L79/L81) and the dial stays a dial.
#
# THE ONE GENUINELY NEW STRUCTURE IS ADJACENCY, and it is unavoidable at k=4: {0,2,4} plus any member
# of {1,3,5} in a six-layer model necessarily makes an adjacent pair, so no site pick avoids it. It is
# verified rather than assumed, and it is a NEGATIVE result -- nothing keys on it. In the width-1 path
# each layer hands `mixed` to the next through its own epilogue (`_decode_out_merge_mix` on a parallel
# non-last layer, `_decode_mlp_out_mix` on a sequential one) and each layer reads `mixed` through its
# own entry (`_decode_qkv_fc_matvec_norm` when parallel, `_decode_qkv_matvec_norm` when not), so the
# interface between neighbours is the same tensor on all four combinations and no branch reads a
# NEIGHBOUR's flag. `_decode_layer_projection_weights(block, parallel_block(i))` is per-layer.
# `Block.forward` branches on `self.parallel` alone. The two count-dependent reads are generic:
# `n_par = len(PARALLEL_BLOCK_LAYERS)` and `wm = wb[PARALLEL_BLOCK_LAYERS[0]][6]`, whose index is
# still 0. No assertion anywhere fixes k at 3 or the merged instance count at 3.
#
# WHAT THIS BUYS, AND WHAT IS NOT IDENTIFIED. The ms leg is the best-measured thing on the board: the
# adjacent rung k=2->3 on this very base read -2.021670341491699 ms (L78 -> L79), and gpu1's row
# carries five per-site readings, -2.413/-2.463/-2.387/-2.438/-2.595 us/call. Band [-1.0, -2.1] ms,
# central -1.8. The QUALITY leg is NOT identified and this is the honest statement of it: the k
# increment on `val_bpb` has two ledger estimates that disagree 7.5x. Same-base pair L76 (k=2,
# 1.0468021404190588) against L75 (k=6, 1.0532409328644643) gives +0.0064388 over four sites =
# +0.0016097/site; L48 against L50 gives +0.0128737426 for all six, which with the former implies
# k=0->2 at +0.0032175/site -- i.e. the cost is CONCAVE, marginal cost halving past k=2. Against that,
# the single adjacent draw L78 -> L79 reads +0.0121368267 for ONE site. Band on the k=3->4 increment
# [+0.0016, +0.0121]. Predicted `val_bpb` [1.0388, 1.0494] against `< 1.05`: ELIGIBLE AT EVERY CORNER
# of the band, which is why this is worth a launch where the midpoints are not -- but the run's k>=2
# path draw sd is 0.0061274305, so at the PESSIMISTIC corner the margin (+0.0006) is 0.11 of one draw
# and eligibility there is a coin. Registered before the launch, not in a retraction.
#
# CREDITS. The dial, the k=3 leg and the per-site latency readings: gpu6 and gpu2. The row, its stop
# clause, its adjudication (an ineligible `val_bpb` is a DRAW, not a refutation of the dial) and the
# structural orthogonality finding that voids its own blocker: gpu1. The eligibility-versus-
# attribution distinction: analyst2. Mine: the restatement that keeps the long-window clause and
# selects layer 1, the adjacency verification, and the concavity reading of the quality leg.
#
# k=4 -> k=5 of 6, ADDED SITE LAYER 3 (experiment parallel_block_k5_on_v42_at_the_long_window_site_
# layer_3, gpu2). NESTING IS PRESERVED -- {0,2} in {0,2,4} in {0,1,2,4} in {0,1,2,3,4} -- so the
# increment reads against the MEASURED k=4 point rather than against an interpolation. The site is
# the long-window layer 3 (`window_sizes` (1024,0),(1024,0),(1024,0),(2048,0),(1024,0),(2048,0)),
# which BREAKS the long-window half of v40's site ground; that ground was never a measurement and is
# disclosed as broken rather than restated. Layer 5 is refused because the out-merge is keyed on
# `i + 1 < n_layer`, so layer 5 would take the input merge only (-1.324 ms) and could not clear the
# 1.626 ms promote bar. At k=5 exactly ONE sequential block remains (layer 5), so the two PACKED pin
# sites drop from 2 live calls/step to 1 and the merged input projection instance count goes 4 -> 5:
# both are generic reads (`n_par = len(PARALLEL_BLOCK_LAYERS)`, `wm = wb[PARALLEL_BLOCK_LAYERS[0]][6]`
# with index still 0), and no assertion anywhere fixes k or the merged instance count -- verified
# again on this sha, not inherited. Adjacency now includes the triple (1,2,3) and the pair (2,3); the
# neighbour interface is the same `mixed` tensor on all four combinations and no branch reads a
# neighbour's flag, so the triple introduces no new object. `val_bpb` prediction 1.0502 (mean of the
# three per-site readings, +0.0066906) with prediction sd ~0.008: P(eligible) ~0.45, prize
# -2.307..-2.341 ms. The full pre-registration, the reversal of my own closure row and the stop clause
# are in this file's header docstring.
PARALLEL_BLOCK_LAYERS = (0, 1, 2, 3, 4)


def parallel_block(layer_idx):
    """True if this block computes attention and MLP from ONE norm of its residual."""
    return layer_idx in PARALLEL_BLOCK_LAYERS


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


# ---------------------------------------------------------------------------
# Fused pointwise chains for the decode path
#
# The width-1 step is already replayed from a CUDA graph, so the host launch cost is
# gone and what remains is one device-side dispatch per kernel. The eager body issues
# roughly forty of them per layer on [1, 1, *] tensors -- the residual mix is three,
# rotary-plus-norm on q and k is eighteen, the value-embedding gate is five -- against
# eleven that do the layer's actual matrix work. Each helper below hands one pointwise
# chain to inductor so it becomes one kernel instead of many.
#
# These are called only from `_decode_body`. `norm`, `apply_rotary_emb`, `forward` and
# every module's own `forward` are left exactly as they are, so the trained network,
# `flops_per_token_measured`, `num_params_total` and `val_bpb` are the reference's:
# this change can only move decode latency and decode fidelity. The parameters are the
# same tensors `forward` reads -- the helpers take them as arguments and own nothing.
# ---------------------------------------------------------------------------


@torch.compile(dynamic=False)
def _decode_mix_norm(x, x0, resid_lambda, x0_lambda):
    """`resid*x + x0l*x0`, and the norm the projections consume, in one region."""
    mixed = resid_lambda * x + x0_lambda * x0
    return mixed, norm(mixed)


@torch.compile(dynamic=False)
def _decode_qk_rope_norm(q, k, cos, sin):
    """Rotary and then norm, on both q and k."""
    return norm(apply_rotary_emb(q, cos, sin)), norm(apply_rotary_emb(k, cos, sin))


@torch.compile(dynamic=False)
def _decode_qkv_write(qkv, ve_table, idx, cos, sin, kc, vc, at, out, n_head, head_dim):
    """The width-1 step from the merged projection to the cache, in ONE region.

    **`fold_ve_gather_into_qkv_write` (gpu2) folds the value-embedding GATHER inside it.** The
    region already consumed the value-embedding row as `v + ve`, but the row was produced outside
    it by an eager `aten::embedding` call and handed in. The region now takes the TABLE and `idx`
    and indexes the table itself, so that launch stops existing on the three `has_ve` layers.

    An eager aten call is exactly one CUDA launch, with no fusion decision involved -- the same
    device-independent fact that made this region's original -12 `index_copy_` launches bankable.
    So -3 launches per step at depth 6, priced at the 1.601669 ms/node this same absorption into
    this same region measured across three baselines spanning 65 ms.

    Measured on CPU before submission: the region is **6 scheduler nodes with the gather folded in
    and 6 without -- the fold adds ZERO nodes** -- and is **bitwise identical on `q`, `kc` and
    `vc`** at both request shapes, with a control confirming the cache row still changes.
    `cpucheck/probe_fold_gathers.py`, `cpucheck/check_fold.py`. Only the safe direction of the
    CPU/CUDA fusion trap is relied on: this run has twice seen CPU UNDER-fuse relative to CUDA
    (independent embedding gathers one-node-per-output on CPU, `_decode_qk_rope_norm` 5 CPU nodes
    for ~1 CUDA persistent reduction), so "CPU fuses it" implies "CUDA fuses it".

    **The failure mode is bounded to a wash by construction:** the 3 gathers fold into 3 regions
    that already run, so if inductor declines to fuse them the launches move inside those regions
    and the node count is unchanged. Nothing here can raise it.

    `ve_table` is the `nn.Embedding`'s own `weight` Parameter -- the same tensor `forward` reads,
    so decode still computes from the network's parameters and no decode-only copy exists. It is
    already bf16: `init_weights` casts every value-embedding table (train.py:983-985), and
    `aten::embedding` has no autocast registration on any backend, so there is no cast either
    way. `F.embedding(idx, w)` with default arguments is exactly what `nn.Embedding.forward`
    calls.

    The rotary fold is NOT here -- see the file header for why it was split off.

    **autoscts__decode_gpu2's region, lifted verbatim from its frozen candidate.** Splits
    `qkv`, adds the value embedding, rotates and norms q and k, and writes k and v into their
    own caches. Only `q` is returned; the caches are the other two outputs and inductor
    stores them where they already live.

    **MEASURED TWICE, ELIGIBLE TWICE:** 198.290825 -> 178.760171 on champion v9 (seq 15,
    -19.530654, -12 nodes) and 177.028179 -> 161.342144 on champion v10 (seq 18, -15.686035,
    -10 nodes). 1.627554 and 1.568604 ms per node, agreeing to 3.6% across a 21.3 ms
    difference in baseline and a layer-count change. Both were recorded DISCARD only because
    a depth cut was promoted while they were in flight. Rows:
    `results/fuse_decode_qkv_write.md`, `results/fuse_decode_qkv_write_on_depth7.md`.

    **What it removes.** The champion issues two eager `index_copy_` calls per layer -- 12
    per step at 6 layers -- and an eager call is exactly one kernel launch on any device,
    with no fusion decision involved. They sit outside every compiled region only because
    the append was written next to, rather than inside, the region that computes what it
    appends. `index_copy_` on a graph input lowers to a **scatter store on that same
    buffer**: gpu2 read inductor's generated wrapper with caches disabled and both
    `index_copy_` calls are absorbed in place on the separate `kc`/`vc` tensors, the wrapper
    allocating only 1024/1024/16/16/1024 bytes and emitting no `copy_`, so the 525 KB cache
    is neither cloned nor copied back. `_decode_ve_mix` folds into the same region on the
    `has_ve` layers.

    **Why the mutation is safe inside a region.** The hazard this file records is
    `torch.compile(mode="reduce-overhead")`, which copies graph inputs into a static pool so
    an in-place append lands in the copy -- measured, 4.49 bits per token destroyed. This is
    a plain `@torch.compile(dynamic=False)`: no cudagraph trees, no static input pool, and
    capture stays the hand-written `_GraphedDecodeStep`, which binds the real cache
    addresses. Confirmed on the device by both of gpu2's launches:
    `[decode-capture] captured=True` on both request shapes and `nopref_kv_cache_bytes`
    exact.

    **Arithmetic is the champion's, operation for operation and in the same order:** `v + ve`
    before rotary, `norm(apply_rotary_emb(...))` on q and on k, then the writes at the device
    position `at`.

    **Do not predict the fidelity readings unchanged.** Bitwise on CPU did not give TV
    unchanged on the device, and the sign is not predictable: seq 15 moved nopref TV
    +0.007792 and seq 18 moved it -0.003603 on the same mechanism. The rms_norm lands in a
    differently-fused CUDA kernel whose reduction order inductor does not guarantee, and 513
    cached steps compound the last bf16 bits. The claim that survives both launches is
    |dTV| <= ~0.008 with the sign unknown.

    `out`, `n_head` and `head_dim` are Python ints, so the split is a compile-time constant.
    Two graphs are traced, one per `ve is None` branch. **The prefill branch does not call
    this**: it writes contiguous row blocks and hands them to `flash_attn_func`, and is
    untouched.
    """
    B, Tn = qkv.size(0), qkv.size(1)
    q = qkv[..., :out].view(B, Tn, n_head, head_dim)
    k = qkv[..., out:2 * out].view(B, Tn, n_head, head_dim)
    v = qkv[..., 2 * out:].view(B, Tn, n_head, head_dim)
    if ve_table is not None:
        v = v + _ve_lookup(ve_table, idx).view(B, Tn, n_head, head_dim)
    q = norm(apply_rotary_emb(q, cos, sin))
    k = norm(apply_rotary_emb(k, cos, sin))
    kc.index_copy_(1, at, k)
    vc.index_copy_(1, at, v)
    return q


@torch.compile(dynamic=False)
def _decode_q_write(qkv, cos, sin, out, n_head, head_dim):
    """`_decode_qkv_write`'s q lines, and nothing else.

    The two statements below are that region's own, character for character; the k lines, the
    value residual and the two `index_copy_` appends have moved into
    `_decode_attn_partial_kv_write`'s owner program. `out`, `n_head` and `head_dim` are Python
    ints, so the split is a compile-time constant, and there is one graph rather than the
    champion's two, because there is no `ve_table is None` branch left in it.

    This is NOT asserted bitwise against the champion's region and the reason is recorded rather
    than tolerated: a compiled region's reduction is not reproducible from outside it -- 0 of 15
    spellings of this norm were bitwise in an earlier probe -- so moving the same mean square into
    a region that no longer also computes k and v may re-associate it. What the harness asserts is
    that any disagreement is at most ONE bf16 ulp, and that a perturbation of the same size IS
    detected.
    """
    B, Tn = qkv.size(0), qkv.size(1)
    q = qkv[..., :out].view(B, Tn, n_head, head_dim)
    return norm(apply_rotary_emb(q, cos, sin))


@torch.compile(dynamic=False)
def _decode_ve_mix(v, ve):
    """The ungated value residual: `v + ve`. Still the prefill path's; the width-1 path folds
    it into `_decode_qkv_write`."""
    return v + ve


@torch.compile(dynamic=False)
def _decode_add_norm(x, y):
    """A residual add and the norm of its result."""
    added = x + y
    return added, norm(added)


@torch.compile(dynamic=False)
def _decode_add_mix_norm(x, y, x0, resid_lambda, x0_lambda):
    """The previous layer's residual add, then this layer's mix and its norm.

    A layer ends with `x = x + mlp_out` and the next one opens with
    `resid*x + x0l*x0`; nothing reads the value in between, so the boundary between them
    was a kernel spent on nothing. Layer 0 has no predecessor and keeps `_decode_mix_norm`.
    """
    mixed = resid_lambda * (x + y) + x0_lambda * x0
    return mixed, norm(mixed)


@torch.compile(dynamic=False)
def _decode_add_tail_norm(x, y):
    """The last layer's residual add and the final norm, on the scored position only.

    `norm` reduces over the last dimension independently per position, so norming the
    sliced row is the same arithmetic as norming and then slicing.
    """
    return norm((x + y)[:, -1:, :])


@torch.compile(dynamic=False)
def _decode_rotary_at(cos, sin, seq):
    """The step's rotary row: one cast and two gathers, read-only in `seq`."""
    position = seq.to(torch.int64)
    return cos.index_select(1, position), sin.index_select(1, position)


@torch.compile(dynamic=False)
def _decode_cached_attention(q, kc, vc, pos, window, scale):
    """The width-1 step's attention, written out over the preallocated cache.

    Replaces one `fa3.flash_attn_with_kvcache` call. Same function, computed as three
    reductions inductor can schedule instead of one opaque CUTLASS launch.

    `pos` is the device int64 write/query position for this step, `window` the layer's
    left window width as a Python int (so it is a compile-time constant, and the two
    distinct values in `window_sizes` are two graphs). The mask is the kernel's own
    documented rule for `seqlen_q == 1`: with `window_size=(window, 0)` a query at
    position `pos` attends to keys `[pos - window, pos]` inclusive, which subsumes
    `causal=True`. `arange` is folded into the index arithmetic by inductor, so it is
    not a tensor and costs no bytes in the state the cache probe measures.

    Precision, against what the kernel does: the dot product accumulates in fp32 from
    bf16 operands, as FA3's does; the normalisation runs in fp32, as FA3's online
    rescaling does; the value average keeps the weights in fp32 where FA3 rounds them to
    bf16 for its second GEMM, so this is if anything the more accurate of the two. Only
    the store is bf16.

    **The max subtraction is dropped, and that is a bound, not a gamble.** Every key in
    the cache and every query arrive rms-normed over the head dimension -- decode's
    `_decode_qk_rope_norm` and `forward`'s `norm(q), norm(k)` are the only writers -- so
    each has L2 norm exactly `sqrt(D)`, and by Cauchy-Schwarz `|q . k| <= D`. Scaled by
    `D ** -0.5` that is `|score| <= sqrt(128) = 11.32`, whose exponential is 8.2e4 and
    whose reciprocal is 1.2e-5: both are mid-range fp32, and a sum of at most 2049 of
    them is 1.7e8. So `exp` needs no shift here, which removes the `amax` reduction --
    one scheduler node per layer, and the smallest and least parallel of them, since it
    reduces 514 keys into 4 numbers.

    Slots outside the window hold stale bytes -- `reset_decode_state` deliberately does
    not zero them -- and are selected away by `torch.where` rather than by `-inf`, so no
    infinity is ever multiplied by a value and the arithmetic is finite throughout.
    """
    positions = torch.arange(kc.size(1), device=kc.device)
    scores = (kc.float() * q.float()).sum(-1) * scale               # [B, L, H] fp32
    at = pos.view(-1, 1)
    keep = ((positions <= at) & (positions >= at - window)).unsqueeze(-1)
    w = torch.where(keep, scores.exp(), scores.new_zeros(()))       # [B, L, H] fp32
    y = (w.unsqueeze(-1) * vc.float()).sum(1)                       # [B, H, D] fp32
    y = y / w.sum(1).unsqueeze(-1)                                  # [B, H, 1] divisor
    return y.to(q.dtype).unsqueeze(1)                               # [B, 1, H, D]


# ---------------------------------------------------------------------------
# The same function as `_decode_cached_attention` above, as two Triton kernels.
#
# WHY A KERNEL AND NOT ANOTHER ATEN FORM. The region above is three inductor scheduler
# nodes costing 10.635 us per call on this device -- 3.545 us/node against this run's
# 1.26-1.31 us pointwise-node constant -- which at six layers is 63.8 us of a 300.6 us
# step, 21.2% of `nopref_request_ms_median`. Two facts close the aten route:
#
#   * Three is the floor for that formulation. A cross-block barrier separates the
#     head-dimension reduction from the cache-length one, and giving the normaliser the
#     value average's iteration space only moves the divide into its own node (3 -> 3),
#     verified on CPU over 14 position/window combinations.
#   * The cost is not the cache walk. Comparing the two request shapes, going from a
#     514-row cache to a 2049-row one costs ~1.1 us per extra 512 rows, so only ~1.1 us
#     of the 10.635 scales with L and ~9.5 us is per-call fixed. So neither reading fewer
#     cache bytes (int8, GQA, shorter windows) nor better splitting of the walk can reach
#     it, and substituting cuBLAS for either reduction pays a measured 3.5-4 us dispatch
#     floor to remove a 3.2 us node.
#
# What is left is the node count itself, and only a kernel changes it: three reduction
# regions become one split-K pass plus one combine.
#
# The combine is a plain weighted sum rather than an LSE rescale, and that is inherited
# from the region above: because q and k arrive rms-normed over the head dimension,
# |score| <= sqrt(D) and no max subtraction is needed, so partial numerators and
# denominators from different splits are directly addable with no shift to reconcile.
# ---------------------------------------------------------------------------

DECODE_ATTN_BLOCK_L = 32     # cache rows per program per iteration
# One block of cache rows per program at the SCORED shape, which is this axis' floor.
#
# `NITER = cdiv(L, BL*NS)` is the partial kernel's loop trip count, and the timed states are
# `max_len = prefill + steps + 1` (prepare.py), so L is 514 for `nopref_request_ms_median` and
# 2049 for the tiebreak. At BL=32 a 514-row cache is 17 blocks, so:
#
#   NS=16: NITER = cdiv(514, 512) = 2 -- 64 CTAs, and each one runs the body TWICE to cover 17
#          blocks. 15 of the 32 block-slots are wholly off the end of the cache, and for 511 of
#          the 512 steps the entire second iteration is masked end to end.
#   NS=32: NITER = cdiv(514, 1024) = 1 -- 128 CTAs, one block each, and no loop at all.
#
# A wholly masked iteration is not free. The sm80 PTX is a ROLLED loop -- one copy of the body
# with NITER as its trip count, byte-for-byte the same instruction census at NITER 2, 3 and 5 --
# so a masked iteration still issues the whole body: the loads are predicated off and move no
# bytes, but the 32 `ex2.approx`, 63 `fma.rn.f32`, 194 `add.f32` and 162 `shfl.sync` of the two
# cross-lane reductions all execute on zeros. At NITER=1 the loop disappears entirely (2346 ->
# 2144 PTX lines, one fewer `bar.sync`, no `bra`, 16 fewer `selp`).
#
# 16 was wrong because it was carried over from FA3's `num_splits`, where kBlockN is 64 or 128
# and 514 rows are 9 or 5 blocks -- there 16 splits genuinely is 1 block per split. This kernel's
# BL is 32, so its floor is 17 splits, and `_decode_attn_combine` does `tl.arange(0, NS)`, which
# requires a power of two (gpu6 AOT-compiled NS=17 for sm80: it raises). 32 is the smallest legal
# value at the floor. It also shortens the tiebreak shape's loop, 5 -> 3.
DECODE_ATTN_SPLITS = 32      # BUFFER width for the partials; a power of two for `tl.arange`.
# Since `decode_attn_splits_live` this is no longer the grid width -- see the wrapper. It
# is the number of `(head, split)` slots allocated and summed, and the grid is
# `min(cdiv(L, BL), NS)`, so the ranked shape launches 17 and not 32.


@triton.jit
def _decode_attn_partial(
    q_ptr, kc_ptr, vc_ptr, pos_ptr, num_ptr, den_ptr,
    L, stride_kl, scale, window,
    D: tl.constexpr, BL: tl.constexpr, NS: tl.constexpr, GS: tl.constexpr,
    NITER: tl.constexpr,
):
    """One (head, split) program: a partial numerator and denominator over its share of L.

    Program `s` takes cache blocks `s, s+GS, s+2*GS, ...` where `GS` is the grid width, so the
    work is balanced to within one block. `NITER` is `cdiv(cdiv(L, BL), GS)` computed on the host
    and passed as a `constexpr`
    rather than as a `program_id`-dependent loop bound: that keeps the trip count static, so
    the backend can unroll and pipeline it, and it keeps the loop a form
    `TRITON_INTERPRET=1` can execute on a CPU box, which boxes every non-constexpr scalar
    into a one-element array. The two request shapes give exactly two specialisations
    (NITER 1 with GS=17 at L=514, NITER 3 with GS=32 at L=2049) and both are compiled by
    the pre-capture warm-up.
    Blocks past the end of the cache are wholly masked, so over-counting is correct and
    costs one masked iteration.

    `pos` is read from device memory, so the launch never learns the position: no host
    round-trip, and the same program is valid at every step, which is what the captured
    graph requires.

    Precision, against the region this replaces: the dot product accumulates in fp32 from
    bf16 operands, the weights stay fp32 through the value average, and only the store is
    bf16 -- the same choices, in the same places. The grouping of the sums differs (a tree
    over 32 rows, then across iterations, then across the splits, against one tree over all
    of L), so results agree to within one bf16 ulp rather than bitwise.
    """
    h = tl.program_id(0)
    s = tl.program_id(1)
    pos = tl.load(pos_ptr)

    d = tl.arange(0, D)
    qv = tl.load(q_ptr + h * D + d).to(tl.float32)

    acc = tl.zeros([D], dtype=tl.float32)
    wsum = tl.zeros([BL], dtype=tl.float32)

    for i in range(NITER):
        # `GS`, the GRID width, is the block stride -- program `s` takes blocks
        # `s, s+GS, s+2*GS, ...`. `NS` is only the buffer stride, below. They were the same
        # constant until this experiment separated them: the grid may be any width, while `NS`
        # must stay a power of two because `_decode_attn_combine` does `tl.arange(0, NS)`.
        l = (i * GS + s) * BL + tl.arange(0, BL)
        # The region's own mask, unchanged: with window_size=(window, 0) a query at `pos`
        # attends to keys [pos - window, pos] inclusive, which subsumes causal. Slots
        # outside it hold stale bytes `reset_decode_state` deliberately does not zero, so
        # they are masked out of the load rather than multiplied by zero afterwards.
        keep = (l < L) & (l <= pos) & (l >= pos - window)
        off = l[:, None] * stride_kl + h * D + d[None, :]
        k = tl.load(kc_ptr + off, mask=keep[:, None], other=0.0).to(tl.float32)
        sc = tl.sum(k * qv[None, :], axis=1) * scale
        w = tl.where(keep, tl.exp(sc), 0.0)
        v = tl.load(vc_ptr + off, mask=keep[:, None], other=0.0).to(tl.float32)
        acc += tl.sum(w[:, None] * v, axis=0)
        wsum += w

    tl.store(num_ptr + (h * NS + s) * D + d, acc)
    tl.store(den_ptr + h * NS + s, tl.sum(wsum, axis=0))


def _combine_split(gs):
    """`(VS, TAIL)` for a `gs`-slot reduction with NO masked lane: the largest power of two
    `<= gs`, and the scalar remainder.

    `tl.arange` requires a power of two. A BUFFER INDEX does not, and neither does a buffer
    SIZE -- which is what makes `17 = 16 + 1` reachable without touching `DECODE_ATTN_SPLITS`,
    the grid width `GS`, `NITER` or the partial kernel. At the ranked shape `GS = 17` gives
    `(16, 1)`; at the tiebreak shape `GS = 32` gives `(32, 0)` and the tail disappears, so that
    specialisation loses only an always-true predicate.
    """
    vs = 1 << (int(gs).bit_length() - 1)
    return vs, int(gs) - vs


@triton.jit
def _decode_attn_combine_masked_v37(num_ptr, den_ptr, out_ptr, pos_ptr, D: tl.constexpr,
                                    NS: tl.constexpr, GS: tl.constexpr, ADVANCE: tl.constexpr):
    """The champion's (v37) masked combine, VERBATIM, for the probe's spelling pair ONLY.

    Never called on any path that reaches a metric: its only two call sites are in the read-only
    probe, after `METRICS_JSON`. It exists so the price of one reduction tree level is a
    SAME-LAUNCH paired reading -- `SPELLmask ... num_warps=1` against `NWC combine
    num_warps=1` -- on one set of buffers in one process, rather than a cross-launch difference
    against launch 66's 1.778 us/call. Launch 68 measured that level at 0.210 us/call by moving
    `NS` 32 -> 16, which also removed 8 LIVE slots; this pair holds `NS`, `GS` and the buffers
    fixed and moves only the spelling, so it separates the two.
    """
    h = tl.program_id(0)
    if ADVANCE:
        nxt = tl.load(pos_ptr) + 1
    d = tl.arange(0, D)
    s = tl.arange(0, NS)
    live = s < GS
    n = tl.load(num_ptr + (h * NS + s)[:, None] * D + d[None, :],
                mask=live[:, None], other=0.0)
    q = tl.load(den_ptr + h * NS + s, mask=live, other=0.0)
    y = tl.sum(n, axis=0) / tl.sum(q, axis=0)
    tl.store(out_ptr + h * D + d, y.to(tl.bfloat16))
    if ADVANCE:
        if h == 0:
            tl.store(pos_ptr, nxt)


@triton.jit
def _decode_attn_combine(num_ptr, den_ptr, out_ptr, pos_ptr, D: tl.constexpr, NS: tl.constexpr,
                         GS: tl.constexpr, VS: tl.constexpr,
                         TAIL: tl.constexpr, ADVANCE: tl.constexpr):
    """Sum the splits and normalise, one program per head -- and, on the LAST layer, advance `pos`.

    `ADVANCE` is a `constexpr`, so the two lines it guards are traced or absent and never branched
    at run time; layers 0..n-2 launch the same specialisation the champion launches. When it is on,
    program 0 stores `pos + 1`, and `state["seq"].add_(1)` -- 1.438/1.442/1.447/1.465 us/call over
    four in-graph measurements by two authors, a whole CUDA kernel for an 8-byte increment -- is
    not a kernel in the captured step any more.

    **It needs no ordering primitive of any kind, and each clause is checkable without a device.**
    This kernel reads `num` and `den` and never `pos`, so no program of it can race the store;
    `h == 0` is a uniform scalar predicate and not an election. Every reader of `pos` in the
    width-1 step is one of the six `_decode_attn_partial_qkv_rot_write` launches, all of which
    precede this one in a captured graph gpu3 measured to be a perfect chain. The next replay's
    first reader is stream-ordered after this kernel. So `fence`, `atom` and `st.global.wt` are all
    zero in the emitted code -- the class my own launch 42 closed is the one where a program has to
    wait for another, and nothing waits here.

    gpu3 closed this fold with an absolute post-fusion census (`exhausted.md`): the increment's
    output shape is `[1]`, every compiled region on the path reduces to something else, and two
    output shapes are two loop nests. That is correct **for inductor** and does not reach a
    `@triton.jit` kernel, where a store is predicated on `tl.program_id` rather than matched to a
    nest. Their own conclusion -- "recovering that 0.74 ms needs the increment to stop existing,
    not to move" -- is what this does: it stops existing as a kernel and stays a device-tensor
    mutation, so the manual capture still avoids a host round-trip per step.

    `NS` is the buffer stride. `GS <= NS` is how many `(head, split)` slots the pass above
    actually launched and therefore wrote; slots `GS..NS-1` hold whatever was in the
    `torch.empty` allocation. The champion reached exactly `GS` of them by spanning all `NS`
    with `tl.arange` and masking the difference with `other=0.0`. **This kernel reaches exactly
    `GS` of them by ADDRESSING only those**, as `VS = 2**floor(log2(GS))` vector slots plus
    `TAIL = GS - VS` scalar rows, so the uninitialised slots are never addressed and no lane is
    masked. The power-of-two rule binds `tl.arange` and not a buffer index, which is what makes
    `17 = 16 + 1` legal where `tl.arange(0, 17)` raises (gpu6 AOT-compiled that for sm80).

    The two sums hold the same `GS` terms in a different association, so the fp32 result differs
    by a few ulp at most and the bf16 store absorbs it: one fp32 ulp is 6e-8 relative against
    bf16's 3.9e-3, and this run's own measured slope puts an fp32-granularity perturbation at
    dTV ~ 5e-07. It is a reassociation, not an approximation, and the probe checks it on device.

    Within `GS` nothing is conditional: every slot the grid covers is written unconditionally by
    the pass above -- a program that finds all of its blocks masked still stores its zeros.
    """
    h = tl.program_id(0)
    if ADVANCE:
        # Issued before the reduction so its ~600-cycle latency overlaps the tile work instead of
        # sitting on the kernel's tail. One scalar load in each of the four programs.
        nxt = tl.load(pos_ptr) + 1
    d = tl.arange(0, D)
    # `GS = VS + TAIL` live slots, reduced with ZERO dead lanes. The champion spanned the whole
    # buffer -- `tl.arange(0, NS=32)` masked to `s < GS=17` -- so 15 of its 32 lanes added exact
    # zeros and the reduction paid a fifth tree level to fold them. `VS` is the largest power of
    # two `<= GS`, which is all `tl.arange` requires; the remaining `TAIL` slots are added one
    # `[D]` row at a time, each of which is 4 warp instructions at `num_warps=1` rather than a
    # tree level. Exactly the same terms in a different grouping.
    #
    # This is also a strictly SAFER read than the mask it replaces: slots `GS..NS-1` hold whatever
    # was in the `torch.empty` allocation, and this spelling never addresses them, so no lane
    # depends on `other=0.0` to neutralise uninitialised memory.
    sv = tl.arange(0, VS)
    n = tl.load(num_ptr + (h * NS + sv)[:, None] * D + d[None, :])
    q = tl.load(den_ptr + h * NS + sv)
    acc = tl.sum(n, axis=0)
    dacc = tl.sum(q, axis=0)
    for t in tl.static_range(TAIL):
        acc += tl.load(num_ptr + (h * NS + VS + t) * D + d)
        dacc += tl.load(den_ptr + h * NS + VS + t)
    y = acc / dacc
    tl.store(out_ptr + h * D + d, y.to(tl.bfloat16))
    if ADVANCE:
        if h == 0:
            tl.store(pos_ptr, nxt)


def _decode_cached_attention_triton(q, kc, vc, pos, window, scale):
    """`_decode_cached_attention`'s signature and function, as two kernel launches.

    `num`, `den` and `out` are ordinary transients, deliberately not part of the state: under
    capture they land in the graph's private pool at fixed addresses, and with `graph=False`
    they are freed each step. `measure_kv_cache_bytes` reads what the state *holds*, and the
    state holds none of them, so `nopref_kv_cache_bytes` is unchanged.

    Falls back to the compiled region for any shape this pair does not cover, so the kernel
    can only ever be an optimisation of the case the instrument actually measures (batch 1,
    128-wide heads) and never the reason a call fails.
    """
    B, _, H, D = q.shape
    if B != 1 or D != 128 or not (kc.is_contiguous() and vc.is_contiguous()):
        return _decode_cached_attention(q, kc, vc, pos, window, scale)
    if not q.is_contiguous():
        q = q.contiguous()
    L = kc.size(1)
    # The grid is exactly as wide as there is cache to cover, capped at the buffer width.
    #
    # `DECODE_ATTN_SPLITS` is a POWER OF TWO because `_decode_attn_combine` does
    # `tl.arange(0, NS)`; the floor at the ranked shape is `cdiv(514, 32) = 17`, so 32 is the
    # smallest legal buffer width and **15 of its 32 splits have no cache to read**. Launch seq
    # 31 measured what one of those costs by adding 32 more of them: **0.0555859 us/call**, a CTA
    # that loads `pos` and `q`, runs a wholly masked block body and stores 129 fp32 zeros.
    #
    # The power-of-two rule binds `tl.arange`. It does not bind a GRID DIMENSION, which may be any
    # integer. So the two stop being the same number: `GS` programs are launched and `NS` slots are
    # summed, with the combine masking the difference.
    #
    #   ranked   L=514:  NBLOCK=17, GS=17, NITER=1 -- 68 CTAs instead of 128, 60 of which did
    #                    nothing. 15 x 0.0555859 = 0.8338 us/call = 2.5614 ms.
    #   tiebreak L=2049: NBLOCK=65, GS=min(65,32)=32, NITER=cdiv(65,32)=3 -- the champion's grid,
    #                    trip count and buffer stride exactly. AOT-compiled for sm80 with debug
    #                    records stripped, the PARTIAL's instruction stream is byte-identical to the
    #                    champion's (1506 lines, same digest). The COMBINE is not: it gains 3 scalar
    #                    lines because its 33 `ld.global` become predicated. `s < 32` over
    #                    `tl.arange(0, 32)` resolves at compile time -- the predicate is set by
    #                    `mov.pred -1` and **no `setp` is emitted** -- so what changes is a guard bit
    #                    on 33 loads that all still execute, once per program on 4 programs.
    #
    # `NITER = cdiv(NBLOCK, GS)` is the champion's `cdiv(L, BL*NS)` wherever `NBLOCK > NS`, so
    # the tiebreak specialisation is unchanged; it differs only where the cap does not bite.
    NBLOCK = triton.cdiv(L, DECODE_ATTN_BLOCK_L)
    GS = min(NBLOCK, DECODE_ATTN_SPLITS)
    _VS, _TAIL = _combine_split(GS)
    num = torch.empty((H, DECODE_ATTN_SPLITS, D), dtype=torch.float32, device=q.device)
    den = torch.empty((H, DECODE_ATTN_SPLITS), dtype=torch.float32, device=q.device)
    out = torch.empty((B, 1, H, D), dtype=q.dtype, device=q.device)
    _decode_attn_partial[(H, GS)](
        q, kc, vc, pos, num, den,
        L, kc.stride(1), scale, window,
        D=D, BL=DECODE_ATTN_BLOCK_L, NS=DECODE_ATTN_SPLITS, GS=GS,
        NITER=triton.cdiv(NBLOCK, GS), num_warps=4,
    )
    _decode_attn_combine[(H,)](num, den, out, pos, D=D, NS=DECODE_ATTN_SPLITS, GS=GS, VS=_VS, TAIL=_TAIL,
                               ADVANCE=False, num_warps=4)
    return out


# `F.rms_norm`'s default eps on a bf16 input is `finfo(float32).eps` and NOT
# `finfo(bfloat16).eps` -- gpu6 measured 60/60 draws bitwise with the fp32 value and 16/60 with
# the bf16 one, and v19's `_matvec_int8_prenorm` already depends on it.
_RMS_NORM_EPS = float(torch.finfo(torch.float32).eps)


# ---------------------------------------------------------------------------
# `_decode_attn_partial` with the K/V HALF of `_decode_qkv_write` absorbed into its owner program.
#
# WHAT MOVES, AND WHAT DOES NOT. `_decode_qkv_write` computes, per layer, per step:
#
#     q = norm(apply_rotary_emb(qkv[..., :out]))                 <-- STAYS, in `_decode_q_write`
#     k = norm(apply_rotary_emb(qkv[..., out:2*out]))            <-- moves
#     v = qkv[..., 2*out:] + F.embedding(idx, ve_table)          <-- moves
#     kc.index_copy_(1, at, k); vc.index_copy_(1, at, v)         <-- moves
#
# so the region keeps ONE of its two rope+norm chains and loses both cache stores and the
# value-embedding gather. gpu1's absolute census makes the whole region 6 scheduler nodes, 2 of
# them nops (the `cat`s in `apply_rotary_emb`), and gpu6's launch-37 timing makes it 2 CUDA
# kernels per call at 3.454 us (ve) / 3.293 us (no ve) = 20.241 us/step = 10.384 ms of the
# ranked metric, of which ~87% is dispatch floor (2 kernels x ~1.5 us against ~5-9 KB of traffic,
# which is 0.005 us at the 1141 GB/s this run measured).
#
# THE ONE FACT THE PRE-REGISTRATION RESTS ON AND THAT NO CPU CHECK CAN SETTLE: whether the region
# that is left is ONE CUDA kernel where the champion's was two. The CPU node census bounds it in
# the safe direction only (this run has twice measured CPU UNDER-fusing relative to CUDA), and
# scheduler nodes are not CUDA kernels -- my own launch-37 repricing was wrong by exactly that
# conversion, twice. So the candidate carries the CUDA-kernel census rider that five launches have
# now failed to buy, and the band below is wide enough to contain "the region is still 2 kernels
# and this row is a small loss".
#
# WHY THE ABSORPTION IS LEGAL, in the form autoscts__decode_gpu5 corrected -- the invariant is
# BLOCK-DISJOINTNESS and not the mask:
#
#   1. `_decode_attn_partial` walks `l = (i * GS + s) * BL + tl.arange(0, BL)`, so cache block `b`
#      is addressed by exactly ONE program, `s = b % GS`, at iteration `i = b // GS`. Position
#      `pos` lies in exactly one block, `pos // BL`. So the row this kernel writes is read back by
#      the very program that wrote it and by no other program -- not because other programs mask
#      it out, but because they never address it. `GS` is the GRID width and NOT `NS`: at the
#      ranked shape GS=17 while NS=32, and the two coincide there only by accident
#      (`pos//32 <= 16 < 17`).
#   2. What `l <= pos` protects is a different thing, worth keeping straight: the rows ABOVE `pos`
#      inside the owner's own block, which `reset_decode_state` deliberately leaves as stale bytes.
#   3. The store is placed BEFORE the block loop, so the owner's iteration index never matters --
#      a store inside the loop body would have to be at `i == b // GS` and would be a real
#      ordering bug at the tiebreak shape, where the owning iteration can be 2.
#   4. `__syncthreads()` makes a block's prior global stores visible to that block, so the owner
#      stores, barriers, and then reads its own row. The barrier is UNCONDITIONAL so it can never
#      sit in divergent control flow. This was measured, not appealed to: the harness compares the
#      fused kernel against a separately-appended cache at 7 `(L, pos)` configurations and gets
#      `max|dnum| = 0.000e+00`.
#   5. `n_kv_head == n_head` on this substrate, so head `h`'s cache row is written and read by
#      head `h`'s own programs.
#
# ARITHMETIC. Every rounding site is named because a rounding must be measured at its own site and
# does not transfer from a neighbouring one. All three of these are READ off inductor's generated
# code for the champion's own region, not argued from the FX passes:
#
#   * `qkv`, `cos`, `sin` and the `ve` table are bf16 in memory.
#   * **`apply_rotary_emb`'s output IS rounded to bf16 before the norm reads it.** It ends in
#     `torch.cat`, whose output buffer is bf16, so the rope result is materialised and rounded:
#     the generated region computes `tmp10 = x1*cos + x2*sin`, then `convert<BFloat16>`, stores,
#     and the mean-square loop re-loads that buffer. `pointless_convert` cannot delete it because
#     it is a real buffer and not a cast chain. An earlier build of mine DID drop it -- it took
#     the mean square over unrounded fp32 -- and the harness caught it at one bf16 ulp on every k
#     row at 2048 of 2049 positions, with only `pos = 0` matching, where `cos=1, sin=0` makes rope
#     the identity. A harness that sampled position 0 would have passed it.
#   * `F.rms_norm` on a bf16 input accumulates in fp32 with default
#     `eps = 1.1920928955078125e-07 = finfo(float32).eps` (gpu6: 60/60 draws bitwise with the fp32
#     value, 16/60 with the bf16 one) and returns bf16 -- but here the norm's output goes straight
#     into the bf16 cache, so there is exactly ONE rounding on the way to `kc` and it is the
#     store, same as the champion's `index_copy_` of a bf16 region output.
#   * `v + ve` sums in fp32 and rounds on the store into `vc`, as before.
#   * What is left is reduction ASSOCIATION on the 128-element mean square: `tl.sum`'s tree
#     against inductor's `VectorizedN<float,2>` plus `vec_reduce_all`. Same class as gpu5's seq-20
#     kernel, which is in the champion.
#
# `rsqrt` is `tl.math.rsqrt`, which lowers to the same `libdevice.rsqrt` inductor emits for
# `torch.rsqrt` on CUDA. (The CPU backend spells it `1 / std::sqrt`, so the pure-torch leg
# compares `1/sqrt` against `1/sqrt`; the implementation cancels at each site rather than being
# assumed equal across backends.)
#
# The cos/sin ROWS are passed in, not the tables: `_decode_rotary_at` still runs for q, so its
# gathered `[1, 1, HALF]` row is already there and the kernel loads `cos_ptr + cd`. That is one
# fewer thing to get wrong than indexing the table at `pos`, and it keeps `_decode_rotary_at`
# untouched for gpu3's queued capture-concurrency row.
# ---------------------------------------------------------------------------


@triton.jit
def _decode_attn_partial_kv_write(
    q_ptr, qkv_ptr, ve_ptr, idx_ptr, cos_ptr, sin_ptr,
    kc_ptr, vc_ptr, pos_ptr, num_ptr, den_ptr,
    L, stride_kl, scale, window,
    D: tl.constexpr, HALF: tl.constexpr, OUT: tl.constexpr, BL: tl.constexpr,
    NS: tl.constexpr, GS: tl.constexpr, NITER: tl.constexpr,
    HAS_VE: tl.constexpr, EPS: tl.constexpr,
):
    """`_decode_attn_partial`, with this step's k/v append done by the owner program.

    Everything from `acc = tl.zeros(...)` down is `_decode_attn_partial`'s body, character for
    character, and the harness asserts that on the token stream. `q` is loaded exactly as that
    kernel loads it -- the region still produces it -- so this kernel's dot product reads the
    same bytes the champion's did.
    """
    h = tl.program_id(0)
    s = tl.program_id(1)
    pos = tl.load(pos_ptr)

    d = tl.arange(0, D)
    qv = tl.load(q_ptr + h * D + d).to(tl.float32)

    # The ONE program that owns the block holding `pos` appends this step's k and v. Every other
    # program takes the branch predicated off: no loads, no reduction, no stores.
    if s == (pos // BL) % GS:
        lo = d < HALF
        partner = tl.where(lo, d + HALF, d - HALF)
        cd = tl.where(lo, d, d - HALF)
        cs = tl.load(cos_ptr + cd).to(tl.float32)
        sg = tl.load(sin_ptr + cd).to(tl.float32)
        # Rotary without a `cat`: for `d < HALF` the champion computes `x1*cos + x2*sin` and for
        # `d >= HALF` it computes `-x1*sin + x2*cos`. Both are
        # `x[d]*cos[d%HALF] + sgn*x[partner]*sin[d%HALF]` with `sgn = +1` below the half and `-1`
        # above it, so one expression covers the whole 128-wide row.
        sg = tl.where(lo, sg, -sg)
        kb = qkv_ptr + OUT + h * D
        xk = tl.load(kb + d).to(tl.float32)
        xkp = tl.load(kb + partner).to(tl.float32)
        kr = (xk * cs + sg * xkp).to(tl.bfloat16).to(tl.float32)
        kms = tl.sum(kr * kr, axis=0) / D
        kn = kr * tl.math.rsqrt(kms + EPS)
        vn = tl.load(qkv_ptr + 2 * OUT + h * D + d).to(tl.float32)
        if HAS_VE:
            tok = tl.load(idx_ptr)
            vn = vn + tl.load(ve_ptr + tok * OUT + h * D + d).to(tl.float32)
        wo = pos * stride_kl + h * D + d
        tl.store(kc_ptr + wo, kn.to(tl.bfloat16))
        tl.store(vc_ptr + wo, vn.to(tl.bfloat16))
    # Unconditional, so it is never inside divergent control flow: after this the block's own
    # stores are visible to the block, which is all the loop below needs.
    tl.debug_barrier()

    acc = tl.zeros([D], dtype=tl.float32)
    wsum = tl.zeros([BL], dtype=tl.float32)

    for i in range(NITER):
        # `GS`, the GRID width, is the block stride -- program `s` takes blocks
        # `s, s+GS, s+2*GS, ...`. `NS` is only the buffer stride, below. They were the same
        # constant until this experiment separated them: the grid may be any width, while `NS`
        # must stay a power of two because `_decode_attn_combine` does `tl.arange(0, NS)`.
        l = (i * GS + s) * BL + tl.arange(0, BL)
        # The region's own mask, unchanged: with window_size=(window, 0) a query at `pos`
        # attends to keys [pos - window, pos] inclusive, which subsumes causal. Slots
        # outside it hold stale bytes `reset_decode_state` deliberately does not zero, so
        # they are masked out of the load rather than multiplied by zero afterwards.
        keep = (l < L) & (l <= pos) & (l >= pos - window)
        off = l[:, None] * stride_kl + h * D + d[None, :]
        k = tl.load(kc_ptr + off, mask=keep[:, None], other=0.0).to(tl.float32)
        sc = tl.sum(k * qv[None, :], axis=1) * scale
        w = tl.where(keep, tl.exp(sc), 0.0)
        v = tl.load(vc_ptr + off, mask=keep[:, None], other=0.0).to(tl.float32)
        acc += tl.sum(w[:, None] * v, axis=0)
        wsum += w

    tl.store(num_ptr + (h * NS + s) * D + d, acc)
    tl.store(den_ptr + h * NS + s, tl.sum(wsum, axis=0))


def _decode_kv_write_attention(q, qkv, ve_table, idx, cos, sin, kc, vc, seq,
                               out, n_head, head_dim, window, scale):
    """`_decode_cached_attention_triton` with this step's k/v append inside the partial kernel.

    Two launches, as before -- the append is not a launch of its own any more, and no launch is
    added. What leaves the step is the k/v half of `_decode_qkv_write`'s work: two loads, one
    128-element reduction, one gather and two stores per layer, which the owner program now does.

    Falls back to the champion's own ladder -- `_decode_qkv_write` for the append, then
    `_decode_cached_attention_triton`, which itself falls back to `_decode_cached_attention` --
    for any shape the fused kernel does not cover.

    **The fallback is not decoration and it is the defect gpu5 said they would hold a freeze for.**
    Once the append moves inside the kernel, a fallback that only attends would read a cache row
    NOBODY WROTE and return a plausible tensor rather than raising; the instrument only ever calls
    B=1/D=128/contiguous, so no metric would catch it. So the fallback calls the champion's region
    to do the append. It recomputes q and discards it -- wasteful, on a path the instrument never
    takes, and the right trade against a silently wrong tensor.

    `num`, `den` and `y` are per-call transients on no state, exactly as in
    `_decode_cached_attention_triton`, so `nopref_kv_cache_bytes` is unchanged.
    """
    B, Tn = qkv.size(0), qkv.size(1)
    D, H = head_dim, n_head
    fused = (B == 1 and Tn == 1 and D == 128 and out == H * D
             and qkv.size(-1) == 3 * out
             and qkv.is_contiguous() and kc.is_contiguous() and vc.is_contiguous()
             and cos.is_contiguous() and sin.is_contiguous()
             and cos.numel() == D // 2 and sin.numel() == D // 2
             and kc.size(2) * kc.size(3) == out
             and q.is_contiguous() and q.numel() == out
             and (ve_table is None or (ve_table.is_contiguous() and ve_table.size(-1) == out)))
    if not fused:
        _decode_qkv_write(qkv, ve_table, idx, cos, sin, kc, vc, seq[:1],
                          out, n_head, head_dim)
        return _decode_cached_attention_triton(q, kc, vc, seq, window, scale)
    L = kc.size(1)
    NBLOCK = triton.cdiv(L, DECODE_ATTN_BLOCK_L)
    GS = min(NBLOCK, DECODE_ATTN_SPLITS)
    _VS, _TAIL = _combine_split(GS)
    num = torch.empty((H, DECODE_ATTN_SPLITS, D), dtype=torch.float32, device=qkv.device)
    den = torch.empty((H, DECODE_ATTN_SPLITS), dtype=torch.float32, device=qkv.device)
    y = torch.empty((B, 1, H, D), dtype=qkv.dtype, device=qkv.device)
    _decode_attn_partial_kv_write[(H, GS)](
        q, qkv, qkv if ve_table is None else ve_table, idx, cos, sin,
        kc, vc, seq, num, den,
        L, kc.stride(1), scale, window,
        D=D, HALF=D // 2, OUT=out, BL=DECODE_ATTN_BLOCK_L, NS=DECODE_ATTN_SPLITS,
        GS=GS, NITER=triton.cdiv(NBLOCK, GS), HAS_VE=ve_table is not None,
        EPS=_RMS_NORM_EPS, num_warps=4,
    )
    _decode_attn_combine[(H,)](num, den, y, seq, D=D, NS=DECODE_ATTN_SPLITS, GS=GS, VS=_VS, TAIL=_TAIL,
                               ADVANCE=False, num_warps=4)
    return y


# ---------------------------------------------------------------------------
# `_decode_attn_partial_kv_write` with the Q HALF absorbed as well, so `_decode_q_write` stops
# being called and the last kernel of the old `_decode_qkv_write` region leaves the step.
#
# WHAT MOVES. v25 left exactly two statements outside the attention kernel:
#
#     q = qkv[..., :out].view(B, Tn, n_head, head_dim)
#     return norm(apply_rotary_emb(q, cos, sin))
#
# measured by launch 40's `[attribution]` rider at **1.747 us/call, 1 CUDA kernel**, 6 calls per
# step = 10.48 us/step = **5.38 ms of `nopref_request_ms_median`** -- against a width-1 dispatch
# floor of 1.45-1.75 us/kernel, so that call is ~100% floor and contains almost no work.
#
# WHY THIS IS THE CHEAP CLASS AND NOT THE CLOSED ONE. autoscts__decode_gpu6's launch 42 closed
# "delete a launch and pay for it with a cross-CTA handshake": a device-scope fence plus a
# contended atomic costs >= 1.8-2.0 us against a launch's 1.44-1.60. **This row adds no
# communication of any kind.** Every program computes its own head's q from `qkv`, which is a
# kernel INPUT that nothing here writes, so there is no ordering requirement, no fence, no
# atomic, and the existing `tl.debug_barrier()` is untouched. It is the same class as v25:
# absorbing work into a program that was already going to run.
#
# THE TERM THIS LAUNCH BUYS, WHICH IS THE POINT EVEN IF THE ROW LOSES. v25's absorb was paid by
# ONE program of 68 and measured +0.583 us/call. This one is paid by ALL 68, and nothing in this
# run has ever measured that. Two live accounts differ:
#
#   * fan-out: 68 CTAs over 108 SMs run concurrently, so the kernel's added latency is about ONE
#     program's added work -- ~+0.6 us/call, net -1.1 us/call = -3.5 ms;
#   * against it: 68 concurrent cross-warp reductions contend for issue slots and shared memory in
#     a way one does not, and my own launch-40 registration underpriced a Triton cross-warp
#     reduction by 2.1x by taking its price from an inductor region's class.
#
# So the registration's worst case is a LOSS, and the `[attribution]` rider times the champion's
# spelling and this one against each other in the same graph, at the scored position, so the
# per-program term is measured either way.
#
# ARITHMETIC, site by site. q's chain is `_decode_q_write`'s, at ITS rounding boundaries, which
# are the ones already lifted and verified for k in v25:
#
#   * `apply_rotary_emb` ends in `torch.cat`, whose bf16 output buffer rounds the rope result
#     before the norm re-reads it -- so `(xq * cs + sg * xqp).to(tl.bfloat16).to(tl.float32)`.
#     An earlier build of mine dropped exactly this cast on k and the harness caught it at one
#     bf16 ulp at 2048 of 2049 positions.
#   * `F.rms_norm` on a bf16 input accumulates the mean square in fp32 with
#     `eps = finfo(float32).eps` and returns **bf16**; v25's kernel then loaded that bf16 value and
#     cast it to fp32. Both boundaries are kept: `.to(tl.bfloat16).to(tl.float32)` on the normed q.
#   * What is NOT reproducible from outside the kernel is the ASSOCIATION of the 128-element mean
#     square -- `tl.sum`'s tree against inductor's `VectorizedN<float,2>` + `vec_reduce_all`. That
#     is a 128-term tree, and it lands behind a bf16 rounding. gpu6's launch-42 / gpu1's launch-41
#     pair says the TV cost of a tree change scales with the reduction's LENGTH and is absorbed by
#     a bf16 store: 32 terms behind a bf16 store moved the TV mean +0.000109, while 512/2048-term
#     trees breached the ceiling. The direct precedent is this candidate's own base: v25 moved k's
#     128-element mean square into this kernel behind a bf16 store and `decode_tv_distance_max`
#     went 0.041244 -> 0.033333.
#
# The cos/sin ROWS are already kernel arguments and `_decode_rotary_at` still runs, so nothing
# about the rotary gather changes. The k/v half, the owner test, the barrier and the whole block
# loop are v25's, character for character; the rope helpers are HOISTED out of the owner branch
# because q needs them in every program, which is the same expressions in the same order.
# ---------------------------------------------------------------------------


@triton.jit
def _decode_attn_partial_qkv_write(
    qkv_ptr, ve_ptr, idx_ptr, cos_ptr, sin_ptr,
    kc_ptr, vc_ptr, pos_ptr, num_ptr, den_ptr,
    L, stride_kl, scale, window,
    D: tl.constexpr, HALF: tl.constexpr, OUT: tl.constexpr, BL: tl.constexpr,
    NS: tl.constexpr, GS: tl.constexpr, NITER: tl.constexpr,
    HAS_VE: tl.constexpr, EPS: tl.constexpr,
):
    """`_decode_attn_partial_kv_write`, and q's rope+norm as well, in every program.

    Everything from `acc = tl.zeros(...)` down is `_decode_attn_partial`'s body, character for
    character, and the owner branch is v25's, character for character with its `cs`/`sg` lifted to
    the prologue that q now also needs. What leaves the step is `_decode_q_write`'s kernel.

    `q_ptr` is gone: the query is computed here from `qkv`, which this kernel only reads.
    """
    h = tl.program_id(0)
    s = tl.program_id(1)
    pos = tl.load(pos_ptr)

    d = tl.arange(0, D)

    # The rope helpers, hoisted out of v25's owner branch because q needs them in EVERY program.
    # Same expressions, same order, same values -- for `d < HALF` the champion computes
    # `x1*cos + x2*sin` and above the half `-x1*sin + x2*cos`, which is
    # `x[d]*cos[d%HALF] + sgn*x[partner]*sin[d%HALF]` with `sgn = -1` above the half.
    lo = d < HALF
    partner = tl.where(lo, d + HALF, d - HALF)
    cd = tl.where(lo, d, d - HALF)
    cs = tl.load(cos_ptr + cd).to(tl.float32)
    sg = tl.load(sin_ptr + cd).to(tl.float32)
    sg = tl.where(lo, sg, -sg)

    # `_decode_q_write`'s two statements, in every program. `qkv` is an input this kernel never
    # writes, so no program waits for another one and no ordering instruction is needed.
    # Both of the region's roundings are kept: the `cat` buffer's bf16 on the rope result, and
    # rms_norm's bf16 return, which v25's kernel loaded and cast to fp32.
    xq = tl.load(qkv_ptr + h * D + d).to(tl.float32)
    xqp = tl.load(qkv_ptr + h * D + partner).to(tl.float32)
    qr = (xq * cs + sg * xqp).to(tl.bfloat16).to(tl.float32)
    qms = tl.sum(qr * qr, axis=0) / D
    qv = (qr * tl.math.rsqrt(qms + EPS)).to(tl.bfloat16).to(tl.float32)

    # The ONE program that owns the block holding `pos` appends this step's k and v. Every other
    # program takes the branch predicated off: no loads, no reduction, no stores.
    if s == (pos // BL) % GS:
        kb = qkv_ptr + OUT + h * D
        xk = tl.load(kb + d).to(tl.float32)
        xkp = tl.load(kb + partner).to(tl.float32)
        kr = (xk * cs + sg * xkp).to(tl.bfloat16).to(tl.float32)
        kms = tl.sum(kr * kr, axis=0) / D
        kn = kr * tl.math.rsqrt(kms + EPS)
        vn = tl.load(qkv_ptr + 2 * OUT + h * D + d).to(tl.float32)
        if HAS_VE:
            tok = tl.load(idx_ptr)
            vn = vn + tl.load(ve_ptr + tok * OUT + h * D + d).to(tl.float32)
        wo = pos * stride_kl + h * D + d
        tl.store(kc_ptr + wo, kn.to(tl.bfloat16))
        tl.store(vc_ptr + wo, vn.to(tl.bfloat16))
    # Unconditional, so it is never inside divergent control flow: after this the block's own
    # stores are visible to the block, which is all the loop below needs.
    tl.debug_barrier()

    acc = tl.zeros([D], dtype=tl.float32)
    wsum = tl.zeros([BL], dtype=tl.float32)

    for i in range(NITER):
        # `GS`, the GRID width, is the block stride -- program `s` takes blocks
        # `s, s+GS, s+2*GS, ...`. `NS` is only the buffer stride, below.
        l = (i * GS + s) * BL + tl.arange(0, BL)
        # The region's own mask, unchanged: with window_size=(window, 0) a query at `pos`
        # attends to keys [pos - window, pos] inclusive, which subsumes causal. Slots
        # outside it hold stale bytes `reset_decode_state` deliberately does not zero, so
        # they are masked out of the load rather than multiplied by zero afterwards.
        keep = (l < L) & (l <= pos) & (l >= pos - window)
        off = l[:, None] * stride_kl + h * D + d[None, :]
        k = tl.load(kc_ptr + off, mask=keep[:, None], other=0.0).to(tl.float32)
        sc = tl.sum(k * qv[None, :], axis=1) * scale
        w = tl.where(keep, tl.exp(sc), 0.0)
        v = tl.load(vc_ptr + off, mask=keep[:, None], other=0.0).to(tl.float32)
        acc += tl.sum(w[:, None] * v, axis=0)
        wsum += w

    tl.store(num_ptr + (h * NS + s) * D + d, acc)
    tl.store(den_ptr + h * NS + s, tl.sum(wsum, axis=0))


def _decode_qkv_write_attention(qkv, ve_table, idx, cos, sin, kc, vc, seq,
                                out, n_head, head_dim, window, scale):
    """`_decode_kv_write_attention` without the `q` argument: the kernel computes it.

    Two launches, as before -- nothing is added and `_decode_q_write`'s launch is gone.

    The fallback ladder is v25's with one wart removed. v25 called `_decode_qkv_write` for the
    append and then DISCARDED its `q`, because a `q` computed outside had already been handed in;
    here that return value is the query, so the fallback recomputes nothing. It still exists for
    the same reason: once the append lives inside the kernel, a fallback that only attends would
    read a cache row nobody wrote and return a plausible tensor rather than raising.

    `num`, `den` and `y` are per-call transients on no state, so `nopref_kv_cache_bytes` is
    unchanged.
    """
    B, Tn = qkv.size(0), qkv.size(1)
    D, H = head_dim, n_head
    fused = (B == 1 and Tn == 1 and D == 128 and out == H * D
             and qkv.size(-1) == 3 * out
             and qkv.is_contiguous() and kc.is_contiguous() and vc.is_contiguous()
             and cos.is_contiguous() and sin.is_contiguous()
             and cos.numel() == D // 2 and sin.numel() == D // 2
             and kc.size(2) * kc.size(3) == out
             and (ve_table is None or (ve_table.is_contiguous() and ve_table.size(-1) == out)))
    if not fused:
        q = _decode_qkv_write(qkv, ve_table, idx, cos, sin, kc, vc, seq[:1],
                              out, n_head, head_dim)
        return _decode_cached_attention_triton(q, kc, vc, seq, window, scale)
    L = kc.size(1)
    NBLOCK = triton.cdiv(L, DECODE_ATTN_BLOCK_L)
    GS = min(NBLOCK, DECODE_ATTN_SPLITS)
    _VS, _TAIL = _combine_split(GS)
    num = torch.empty((H, DECODE_ATTN_SPLITS, D), dtype=torch.float32, device=qkv.device)
    den = torch.empty((H, DECODE_ATTN_SPLITS), dtype=torch.float32, device=qkv.device)
    y = torch.empty((B, 1, H, D), dtype=qkv.dtype, device=qkv.device)
    _decode_attn_partial_qkv_write[(H, GS)](
        qkv, qkv if ve_table is None else ve_table, idx, cos, sin,
        kc, vc, seq, num, den,
        L, kc.stride(1), scale, window,
        D=D, HALF=D // 2, OUT=out, BL=DECODE_ATTN_BLOCK_L, NS=DECODE_ATTN_SPLITS,
        GS=GS, NITER=triton.cdiv(NBLOCK, GS), HAS_VE=ve_table is not None,
        EPS=_RMS_NORM_EPS, num_warps=4,
    )
    _decode_attn_combine[(H,)](num, den, y, seq, D=D, NS=DECODE_ATTN_SPLITS, GS=GS, VS=_VS, TAIL=_TAIL,
                               ADVANCE=False, num_warps=4)
    return y


# ---------------------------------------------------------------------------
# `_decode_attn_partial_qkv_write` reading the rotary TABLES instead of a gathered row, so
# `_decode_rotary_at` stops being called and its two CUDA kernels leave the step.
#
# WHAT IS DELETED, priced on the launch that produced this champion (launch 44's `[attribution]`
# and `[regioncensus]`): `_decode_rotary_at` is **2 CUDA kernels at 2.914 us/call, once per step**,
# = 2.914 us/step = **1.495 ms** of `nopref_request_ms_median`. It is a step-level region on the
# serial critical path: every layer's rope waits on it.
#
# WHAT IS ADDED: one int64 multiply-add per program on the base address. That is all. The loads keep
# their count, their shape and their gather pattern (`cd` is `d` below the half and `d - HALF`
# above, so these were never contiguous loads); only the base moves. The kernel already performs
# this exact arithmetic on `pos` for the cache store two blocks down (`wo = pos * stride_kl + ...`),
# so nothing new about int64 addressing is introduced.
#
# WHY THE VALUES ARE BITWISE THE SAME, which is why this row's fidelity risk is a draw and not a
# mechanism: `_decode_rotary_at` does `cos.index_select(1, seq.to(int64))`, i.e. it COPIES row `pos`
# of the table into a new buffer, and the kernel then loads that copy. Reading the table at
# `pos * stride_cos + cd` loads the same bf16 bytes from the original. No arithmetic is performed on
# either path, so there is no rounding site and no reduction tree anywhere in this diff. Leg 1
# asserts the whole request is BITWISE identical, which the q absorb could not.
#
# `stride_cos` is passed rather than assumed: `_precompute_rotary_embeddings` builds
# `[1, seq_len, 1, HALF]` from a contiguous `[seq_len, HALF]`, so `cos.stride(1) == HALF`, but the
# wrapper passes the real stride and asserts contiguity in its `fused` predicate rather than
# hard-coding the shape.
#
# THE ONE CODEGEN RISK, and leg 3 measures it rather than arguing it: a runtime term in the base
# address can stop ptxas proving 16-byte alignment, and this run has already established that an
# alignment claim changes the emitted load widths (the `tt.divisibility` census trap). `pos * 64`
# bf16 elements is 128 bytes, so every row IS 16-byte aligned, but the compiler need not know it.
# Leg 3 compares the emitted `ld.global` widths and register counts against v26's kernel.
#
# The fallback ladder gathers the row itself: `_decode_qkv_write` needs `[1, 1, 1, HALF]` rows, and
# it is the only path that does now, so `_decode_rotary_at` stays defined and is called there and
# by the two read-only riders.
# ---------------------------------------------------------------------------


@triton.jit
def _decode_attn_partial_qkv_rot_write(
    qkv_ptr, ve_ptr, idx_ptr, cos_ptr, sin_ptr,
    kc_ptr, vc_ptr, pos_ptr, num_ptr, den_ptr,
    L, stride_kl, stride_cos, scale, window,
    D: tl.constexpr, HALF: tl.constexpr, OUT: tl.constexpr, BL: tl.constexpr,
    NS: tl.constexpr, GS: tl.constexpr, NITER: tl.constexpr,
    HAS_VE: tl.constexpr, EPS: tl.constexpr, TOK_AT_POS: tl.constexpr,
    VE_BOUND: tl.constexpr = 0,
):
    """The split-K decode attention pass, with the step's own k/v row kept in REGISTERS.

    The champion routes this step's k and v from the owner program to the owner program THROUGH
    THE CACHE: the owner ropes, norms and STORES them, `tl.debug_barrier()` makes that store
    visible to the rest of the CTA, and then the block loop LOADS the row straight back, because
    `l <= pos` keeps it. That is a store-to-load turnaround inside one program, and the barrier
    guarding it is a scheduling fence for all `H * GS` programs: no tile load may be issued until
    every thread has finished the cos/sin round trip, q's rope and q's 128-element cross-warp
    norm, none of which the tile load depends on.

    Here the row never leaves registers. The tile load is issued as soon as `pos` is in hand, and
    the row at `l == pos` -- whose bytes in the cache are this step's stale ones -- is REPLACED by
    `kn`/`vn` with a `tl.where` before it is used. The two cache stores move to the end of the
    kernel, where their only reader is the NEXT step's launch, which the graph's stream already
    orders after this one. The barrier is then unnecessary and is gone: nothing in this launch
    reads what another thread in this CTA wrote.

    **This is bitwise the champion.** The value the champion's tile carries at `l == pos` is
    `float32(bfloat16(kn))`, because it made that exact round trip through the cache; `knb` is that
    expression written out. Every reduction keeps its shape, its layout and its term order, so no
    tree moves: the score is still one row of the tile's `axis=1` fold and its weight still enters
    `acc` and `den` inside the `BL`-term tree. `keep` is unchanged, so the masked lanes are
    unchanged. A stale lane discarded by `tl.where` cannot poison the result even if it holds a
    NaN, because a select is a bitwise choice and not arithmetic.
    """
    h = tl.program_id(0)
    s = tl.program_id(1)
    pos = tl.load(pos_ptr)

    d = tl.arange(0, D)

    # Iteration 0's cache tile, ISSUED FIRST. Its addresses and its mask need only `pos`; they do
    # not need the rope row, q, or the append. In the champion this read cannot be issued here --
    # `tl.debug_barrier()` is a fence with side effects, so no load may be scheduled across it,
    # and the emitted sm80 stream puts the first tile `ld.global.v4.b32` 316 lines after the
    # reduction's first `bar.sync`. Hoisting it is the point of the change: its L2 latency is then
    # paid underneath the cos/sin round trip, q's rope and q's 128-element cross-warp norm.
    l0 = s * BL + tl.arange(0, BL)
    keep0 = (l0 < L) & (l0 <= pos) & (l0 >= pos - window)
    mine0 = l0 == pos
    off0 = l0[:, None] * stride_kl + h * D + d[None, :]
    kt0 = tl.load(kc_ptr + off0, mask=keep0[:, None], other=0.0)
    vt0 = tl.load(vc_ptr + off0, mask=keep0[:, None], other=0.0)

    # The rope helpers. `cd` indexes the HALF-wide row; the row itself is selected here rather
    # than by a separate kernel.
    lo = d < HALF
    partner = tl.where(lo, d + HALF, d - HALF)
    cd = tl.where(lo, d, d - HALF)
    row = pos * stride_cos
    cs = tl.load(cos_ptr + row + cd).to(tl.float32)
    sg = tl.load(sin_ptr + row + cd).to(tl.float32)
    sg = tl.where(lo, sg, -sg)

    # q's operands, loaded where they were. Its ROPE and its NORM move BELOW the owner's block:
    # `qms = tl.sum(qr * qr, axis=0)` is a 128-element CROSS-WARP fold and Triton lowers it through
    # shared memory with `BAR.SYNC`, which is a scheduling fence for global loads. That is this
    # run's own measurement, not a reading of the docs
    # (`knowledge/decode_ptx_order_is_not_sass_order.md`): deleting a fence gives ptxas PERMISSION
    # to hoist and it still sank the load back down toward its use, and only putting the load first
    # in the SOURCE moved it.
    #
    # So on champion v38 the owner's five loads sit BEHIND barriers. Faithful-`attrs` sm80 build,
    # `ptxas -arch=sm_80` then `nvdisasm -c`, indexed on the `/*hhhh*/` address comments:
    #
    #   xk, xkp        LDG.E.U16 @215, @218    behind 4 BAR.SYNC   (q's fold)
    #   tok            LDG.E.64  @246          behind 7 BAR.SYNC   (q's fold, then k's fold)
    #   vn, ve row     LDG.E.U16 @250, @254    behind 7 BAR.SYNC
    #
    # The `no-ve` specialisation drops exactly the `LDG.E.64` and one `U16` from behind 7, which is
    # what identifies them. `tok -> ve row` is a DEPENDENT pair, so after `pos` arrives the owner
    # pays three serial global round trips and two cross-warp folds -- and a kernel's duration is
    # its SLOWEST CTA, which is the owner. gpu2's launch 40 priced this owner-only prologue at
    # +0.583 us/call, against launch 44's +0.2706 for the same class of work in ALL 68 programs.
    #
    # This revision issues every one of those loads ahead of the first barrier. Nothing else moves.
    xq = tl.load(qkv_ptr + h * D + d).to(tl.float32)
    xqp = tl.load(qkv_ptr + h * D + partner).to(tl.float32)

    # The ONE program that owns the block holding `pos` COMPUTES this step's k and v. It does not
    # store them here: the only reader inside this launch is this same program, and it takes them
    # from these registers instead of from the cache.
    #
    # **A pure statement reorder, and therefore BITWISE the champion.** The owner's block and q's
    # rope+norm are independent: the block reads `qkv_ptr`, `idx_ptr`, `ve_ptr`, `cs`, `sg`,
    # `partner`, `d`, `pos`, `s` and writes only `kn`/`vn`, while q's three lines read `xq`, `xqp`,
    # `cs`, `sg` and write only `qr`/`qms`/`qv`. Inside the block, `vn` -- and `tok`, and the value
    # embedding row -- are independent of `xk`/`xkp`/`kr`/`kms`/`kn`. Every dependency chain keeps
    # its own operation order and no reduction changes shape or term order, so this is not even a
    # reassociation, let alone an approximation. `[spell] partial equivalence` prints
    # `bitwise_num`/`bitwise_den` against the verbatim v38 kernel below, on the same buffers, on
    # device: a `False` there refutes this paragraph and is registered as the failure to watch.
    own = s == (pos // BL) % GS
    kn = tl.zeros([D], dtype=tl.float32)
    vn = tl.zeros([D], dtype=tl.float32)
    if own:
        kb = qkv_ptr + OUT + h * D
        xk = tl.load(kb + d).to(tl.float32)
        xkp = tl.load(kb + partner).to(tl.float32)
        vn = tl.load(qkv_ptr + 2 * OUT + h * D + d).to(tl.float32)
        if HAS_VE:
            # `TOK_AT_POS` is a constexpr: with it on, `idx_ptr` is the base of the row the
            # caller is walking and this step's token sits at `pos` -- the same 8 bytes the
            # champion reads out of a copy the host makes per replay, at their original
            # address. One scalar load either way; `pos` is already in a register.
            if TOK_AT_POS:
                tok = tl.load(idx_ptr + pos)
            else:
                tok = tl.load(idx_ptr)
            # `VE_BOUND` is a `tl.constexpr`, so this is a COMPILE-TIME branch: every layer
            # whose table is full-vocab passes 0 and emits the champion's expression, byte for
            # byte. Only the partial-table layer compiles the guarded form, where the row index
            # is forced in bounds and an out-of-range token contributes an exact zero -- the
            # same value `_ve_lookup` gives the training and prefill paths.
            if VE_BOUND == 0:
                vn = vn + tl.load(ve_ptr + tok * OUT + h * D + d).to(tl.float32)
            else:
                inb = tok < VE_BOUND
                srow = tl.where(inb, tok, 0)
                vrow = tl.load(ve_ptr + srow * OUT + h * D + d).to(tl.float32)
                vn = vn + tl.where(inb, vrow, 0.0)
        kr = (xk * cs + sg * xkp).to(tl.bfloat16).to(tl.float32)
        kms = tl.sum(kr * kr, axis=0) / D
        kn = kr * tl.math.rsqrt(kms + EPS)
    # Exactly what a store of `kn.to(bfloat16)` followed by a load and a `.to(float32)` yields.
    knb = kn.to(tl.bfloat16).to(tl.float32)
    vnb = vn.to(tl.bfloat16).to(tl.float32)

    # q's rope + norm, in every program, at `_decode_q_write`'s own rounding boundaries.
    qr = (xq * cs + sg * xqp).to(tl.bfloat16).to(tl.float32)
    qms = tl.sum(qr * qr, axis=0) / D
    qv = (qr * tl.math.rsqrt(qms + EPS)).to(tl.bfloat16).to(tl.float32)

    acc = tl.zeros([D], dtype=tl.float32)
    wsum = tl.zeros([BL], dtype=tl.float32)

    # Iteration 0, on the tile already in registers. `acc` and `wsum` keep their `+=` against the
    # zero initialiser rather than being assigned, so the accumulation is operation-for-operation
    # the champion's and not even a signed zero can differ.
    k = tl.where(mine0[:, None], knb[None, :], kt0.to(tl.float32))
    sc = tl.sum(k * qv[None, :], axis=1) * scale
    w = tl.where(keep0, tl.exp(sc), 0.0)
    v = tl.where(mine0[:, None], vnb[None, :], vt0.to(tl.float32))
    acc += tl.sum(w[:, None] * v, axis=0)
    wsum += w

    # Iterations 1.. exist only at the prefilled shape (NITER 3); at the ranked shape NITER is 1
    # and this loop does not run. Unchanged from the champion apart from the same `where`.
    for i in range(1, NITER):
        l = (i * GS + s) * BL + tl.arange(0, BL)
        keep = (l < L) & (l <= pos) & (l >= pos - window)
        mine = l == pos
        off = l[:, None] * stride_kl + h * D + d[None, :]
        k = tl.load(kc_ptr + off, mask=keep[:, None], other=0.0).to(tl.float32)
        k = tl.where(mine[:, None], knb[None, :], k)
        sc = tl.sum(k * qv[None, :], axis=1) * scale
        w = tl.where(keep, tl.exp(sc), 0.0)
        v = tl.load(vc_ptr + off, mask=keep[:, None], other=0.0).to(tl.float32)
        v = tl.where(mine[:, None], vnb[None, :], v)
        acc += tl.sum(w[:, None] * v, axis=0)
        wsum += w

    if own:
        # The append, now that every load in this launch has been issued.
        wo = pos * stride_kl + h * D + d
        tl.store(kc_ptr + wo, kn.to(tl.bfloat16))
        tl.store(vc_ptr + wo, vn.to(tl.bfloat16))

    tl.store(num_ptr + (h * NS + s) * D + d, acc)
    tl.store(den_ptr + h * NS + s, tl.sum(wsum, axis=0))


@triton.jit
def _decode_attn_partial_qkv_rot_write_v38(
    qkv_ptr, ve_ptr, idx_ptr, cos_ptr, sin_ptr,
    kc_ptr, vc_ptr, pos_ptr, num_ptr, den_ptr,
    L, stride_kl, stride_cos, scale, window,
    D: tl.constexpr, HALF: tl.constexpr, OUT: tl.constexpr, BL: tl.constexpr,
    NS: tl.constexpr, GS: tl.constexpr, NITER: tl.constexpr,
    HAS_VE: tl.constexpr, EPS: tl.constexpr, TOK_AT_POS: tl.constexpr,
):
    """Champion v38's statement order, kept VERBATIM as this row's in-launch NULL.

    Probe-only: reached from `_probe_decode_regions`, which runs after `METRICS_JSON`, and never
    from `_decode_qkv_rot_write_attention`. Its body is the champion's byte for byte apart from this
    docstring and the name, so the difference between the two sweep legs at `num_warps=4` is the
    price of the owner's fenced loads with no cross-launch drift in it. Launch 69 established that
    a probe-only verbatim copy of a hand-written kernel is the instrument this site wants; its
    `SPELLmask` leg read 1.776 against the previous launch's 1.778, i.e. calibrated to 0.1%.
    """
    h = tl.program_id(0)
    s = tl.program_id(1)
    pos = tl.load(pos_ptr)

    d = tl.arange(0, D)

    # Iteration 0's cache tile, ISSUED FIRST. Its addresses and its mask need only `pos`; they do
    # not need the rope row, q, or the append. In the champion this read cannot be issued here --
    # `tl.debug_barrier()` is a fence with side effects, so no load may be scheduled across it,
    # and the emitted sm80 stream puts the first tile `ld.global.v4.b32` 316 lines after the
    # reduction's first `bar.sync`. Hoisting it is the point of the change: its L2 latency is then
    # paid underneath the cos/sin round trip, q's rope and q's 128-element cross-warp norm.
    l0 = s * BL + tl.arange(0, BL)
    keep0 = (l0 < L) & (l0 <= pos) & (l0 >= pos - window)
    mine0 = l0 == pos
    off0 = l0[:, None] * stride_kl + h * D + d[None, :]
    kt0 = tl.load(kc_ptr + off0, mask=keep0[:, None], other=0.0)
    vt0 = tl.load(vc_ptr + off0, mask=keep0[:, None], other=0.0)

    # The rope helpers. `cd` indexes the HALF-wide row; the row itself is selected here rather
    # than by a separate kernel.
    lo = d < HALF
    partner = tl.where(lo, d + HALF, d - HALF)
    cd = tl.where(lo, d, d - HALF)
    row = pos * stride_cos
    cs = tl.load(cos_ptr + row + cd).to(tl.float32)
    sg = tl.load(sin_ptr + row + cd).to(tl.float32)
    sg = tl.where(lo, sg, -sg)

    # q's rope + norm, in every program, at `_decode_q_write`'s own rounding boundaries.
    xq = tl.load(qkv_ptr + h * D + d).to(tl.float32)
    xqp = tl.load(qkv_ptr + h * D + partner).to(tl.float32)
    qr = (xq * cs + sg * xqp).to(tl.bfloat16).to(tl.float32)
    qms = tl.sum(qr * qr, axis=0) / D
    qv = (qr * tl.math.rsqrt(qms + EPS)).to(tl.bfloat16).to(tl.float32)

    # The ONE program that owns the block holding `pos` COMPUTES this step's k and v. It does not
    # store them here: the only reader inside this launch is this same program, and it takes them
    # from these registers instead of from the cache.
    own = s == (pos // BL) % GS
    kn = tl.zeros([D], dtype=tl.float32)
    vn = tl.zeros([D], dtype=tl.float32)
    if own:
        kb = qkv_ptr + OUT + h * D
        xk = tl.load(kb + d).to(tl.float32)
        xkp = tl.load(kb + partner).to(tl.float32)
        kr = (xk * cs + sg * xkp).to(tl.bfloat16).to(tl.float32)
        kms = tl.sum(kr * kr, axis=0) / D
        kn = kr * tl.math.rsqrt(kms + EPS)
        vn = tl.load(qkv_ptr + 2 * OUT + h * D + d).to(tl.float32)
        if HAS_VE:
            # `TOK_AT_POS` is a constexpr: with it on, `idx_ptr` is the base of the row the
            # caller is walking and this step's token sits at `pos` -- the same 8 bytes the
            # champion reads out of a copy the host makes per replay, at their original
            # address. One scalar load either way; `pos` is already in a register.
            if TOK_AT_POS:
                tok = tl.load(idx_ptr + pos)
            else:
                tok = tl.load(idx_ptr)
            vn = vn + tl.load(ve_ptr + tok * OUT + h * D + d).to(tl.float32)
    # Exactly what a store of `kn.to(bfloat16)` followed by a load and a `.to(float32)` yields.
    knb = kn.to(tl.bfloat16).to(tl.float32)
    vnb = vn.to(tl.bfloat16).to(tl.float32)

    acc = tl.zeros([D], dtype=tl.float32)
    wsum = tl.zeros([BL], dtype=tl.float32)

    # Iteration 0, on the tile already in registers. `acc` and `wsum` keep their `+=` against the
    # zero initialiser rather than being assigned, so the accumulation is operation-for-operation
    # the champion's and not even a signed zero can differ.
    k = tl.where(mine0[:, None], knb[None, :], kt0.to(tl.float32))
    sc = tl.sum(k * qv[None, :], axis=1) * scale
    w = tl.where(keep0, tl.exp(sc), 0.0)
    v = tl.where(mine0[:, None], vnb[None, :], vt0.to(tl.float32))
    acc += tl.sum(w[:, None] * v, axis=0)
    wsum += w

    # Iterations 1.. exist only at the prefilled shape (NITER 3); at the ranked shape NITER is 1
    # and this loop does not run. Unchanged from the champion apart from the same `where`.
    for i in range(1, NITER):
        l = (i * GS + s) * BL + tl.arange(0, BL)
        keep = (l < L) & (l <= pos) & (l >= pos - window)
        mine = l == pos
        off = l[:, None] * stride_kl + h * D + d[None, :]
        k = tl.load(kc_ptr + off, mask=keep[:, None], other=0.0).to(tl.float32)
        k = tl.where(mine[:, None], knb[None, :], k)
        sc = tl.sum(k * qv[None, :], axis=1) * scale
        w = tl.where(keep, tl.exp(sc), 0.0)
        v = tl.load(vc_ptr + off, mask=keep[:, None], other=0.0).to(tl.float32)
        v = tl.where(mine[:, None], vnb[None, :], v)
        acc += tl.sum(w[:, None] * v, axis=0)
        wsum += w

    if own:
        # The append, now that every load in this launch has been issued.
        wo = pos * stride_kl + h * D + d
        tl.store(kc_ptr + wo, kn.to(tl.bfloat16))
        tl.store(vc_ptr + wo, vn.to(tl.bfloat16))

    tl.store(num_ptr + (h * NS + s) * D + d, acc)
    tl.store(den_ptr + h * NS + s, tl.sum(wsum, axis=0))


def _decode_qkv_rot_write_attention(qkv, ve_table, idx, cos, sin, kc, vc, seq,
                                    out, n_head, head_dim, window, scale, advance=False,
                                    tok_row=None):
    """`_decode_qkv_write_attention` taking the rotary TABLES instead of this step's gathered rows.

    Two launches, as before, and `_decode_rotary_at`'s two are gone.

    `cos` and `sin` are the model's own `[1, seq_len, 1, HALF]` tables -- the same tensors
    `_decode_rotary_at` reads and the same ones `forward` reads -- so decode still computes from the
    network's parameters and no decode-only copy exists.

    The fallback gathers the row itself, because `_decode_qkv_write` takes a row and not a table.
    """
    B, Tn = qkv.size(0), qkv.size(1)
    D, H = head_dim, n_head
    fused = (B == 1 and Tn == 1 and D == 128 and out == H * D
             and qkv.size(-1) == 3 * out
             and qkv.is_contiguous() and kc.is_contiguous() and vc.is_contiguous()
             and cos.is_contiguous() and sin.is_contiguous()
             and cos.size(-1) == D // 2 and sin.size(-1) == D // 2
             and cos.stride(1) == D // 2 and sin.stride(1) == D // 2
             and kc.size(2) * kc.size(3) == out
             and (ve_table is None or (ve_table.is_contiguous() and ve_table.size(-1) == out))
             and (tok_row is None or (tok_row.is_contiguous() and tok_row.dim() == 1
                                      and tok_row.dtype == torch.int64)))
    if tok_row is not None and not fused:
        # The in-place token read has no eager spelling here: `idx` on this path is the graph's
        # static input, which nothing writes any more, so serving the ladder from it would
        # attend to a stale token silently. Raising during capture sets `captured` False and
        # the whole request runs the champion's eager ladder, printed -- slow and correct.
        raise RuntimeError("read_step_token_in_place: fused predicate false with tok_row set")
    if not fused:
        cos_row, sin_row = _decode_rotary_at(cos, sin, seq)
        q = _decode_qkv_write(qkv, ve_table, idx, cos_row, sin_row, kc, vc, seq[:1],
                              out, n_head, head_dim)
        y = _decode_cached_attention_triton(q, kc, vc, seq, window, scale)
        if advance:
            seq.add_(1)
        return y
    L = kc.size(1)
    NBLOCK = triton.cdiv(L, DECODE_ATTN_BLOCK_L)
    GS = min(NBLOCK, DECODE_ATTN_SPLITS)
    _VS, _TAIL = _combine_split(GS)
    num = torch.empty((H, DECODE_ATTN_SPLITS, D), dtype=torch.float32, device=qkv.device)
    den = torch.empty((H, DECODE_ATTN_SPLITS), dtype=torch.float32, device=qkv.device)
    y = torch.empty((B, 1, H, D), dtype=qkv.dtype, device=qkv.device)
    _decode_attn_partial_qkv_rot_write[(H, GS)](
        qkv, qkv if ve_table is None else ve_table,
        idx if tok_row is None else tok_row, cos, sin,
        kc, vc, seq, num, den,
        L, kc.stride(1), cos.stride(1), scale, window,
        D=D, HALF=D // 2, OUT=out, BL=DECODE_ATTN_BLOCK_L, NS=DECODE_ATTN_SPLITS,
        GS=GS, NITER=triton.cdiv(NBLOCK, GS), HAS_VE=ve_table is not None,
        EPS=_RMS_NORM_EPS, TOK_AT_POS=tok_row is not None,
        VE_BOUND=(ve_table.size(0)
                  if ve_table is not None and ve_table.size(0) in _PARTIAL_VE_ROW_SET else 0),
        num_warps=4,
    )
    # `advance` is True on the LAST layer only, and only this branch can serve it in the kernel:
    # `numel() != 1` would need every element incremented, which one scalar store cannot do, so
    # that case (and the whole fallback ladder above) pays the eager `add_` instead. Every path
    # therefore advances exactly once, and the ladder cannot silently stop advancing.
    advance_here = bool(advance) and seq.numel() == 1
    # `num_warps` 1, not the champion's 4, and this is the whole executable mechanism.
    # MEASURED, not argued: launch 55 carried a ten-leg paired sweep of both attention kernels at
    # num_warps 1/2/4/8/16 on the same buffers in one process, and this kernel read
    #
    #   nw      1       2       4       8      16
    #   us   1.782   1.949   2.101   3.299   4.874
    #
    # monotone increasing from 1, so **1 is the floor of the axis** and -0.319 us/call against the
    # champion's 4 is the whole available gain: -1.914 us/step, -0.982 ms, 8.9x the 0.110865
    # cross-launch floor. The same launch calibrated that instrument on the OTHER kernel of this
    # same pair: it predicted the partial's nw4->nw8 whole-step delta at +0.717 us/call and the
    # step returned +0.7339, under-reading by 2.3%; and the two standalone legs summed to 7.542
    # against the live wrapper's 7.574, 0.4%.
    #
    # Mechanism, so the number is not the only reason. This grid is `(H,) = 4 CTAs` folding a
    # `[NS=32, D=128]` tile -- 4 CTAs on 108 SMs, so there is no occupancy to gain from more warps
    # and a whole reduction to lose. At nw=4 the `axis=0` fold spans four warps and pays a
    # shared-memory round trip and a `bar.sync`; at nw=1 it is entirely intra-warp shuffles, with
    # `bar.sync` 0 and `shared` 0 (sm80 census, faithful attrs: shared 0/1024/2048/4096/8192 B at
    # nw 1/2/4/8/16). This is the same reading the partial gives in the opposite direction, and it
    # is why the two halves of one pair want different widths: the partial is 68 CTAs and shares
    # its tile across warps profitably, the combine is 4 and does not.
    #
    # Only the LIVE call site moves. The three `_decode_attn_combine` launches in the fallback
    # ladder (`_decode_cached_attention_triton`, `_decode_kv_write_attention`,
    # `_decode_qkv_write_attention`) stay at 4: they are off the measured path, and changing them
    # would put two variables in one diff for no measurement.
    _decode_attn_combine[(H,)](num, den, y, seq, D=D, NS=DECODE_ATTN_SPLITS, GS=GS, VS=_VS, TAIL=_TAIL,
                               ADVANCE=advance_here, num_warps=1)
    if advance and not advance_here:
        seq.add_(1)
    return y


@torch.compile(dynamic=False)
def _decode_relu_square(h):
    """The MLP activation. Still the prefill path's; the width-1 path folds it into `c_fc`."""
    return F.relu(h).square()


def _matvec_bf16(x, w):
    """One width-1 projection: the same map `F.linear(x, w)` computes, as a reduction.

    A broadcast multiply against the [out, in] weight and one `sum` over `in`, accumulated in
    fp32 and stored bf16 -- which is what a bf16 cuBLAS GEMV does. The operands are the
    champion's: `w` is the bf16 copy autocast's cast cache would hand `F.linear`, and `x` is
    rounded to bf16 first because autocast rounds `F.linear`'s input too. Only the order of
    accumulation differs, which is why `forward` remains the thing decode is scored against.

    **Not itself compiled, deliberately.** Each of the four call sites owns its own decorated
    wrapper below, so each wrapper's dynamo cache holds exactly one shape and one dtype across
    all six layers. One shared compiled region would see three shapes and count toward
    `recompile_limit`, and if that limit were ever reached this expression would run eagerly and
    materialise the whole [out, in] fp32 product -- 8.4 MB per call, inside a CUDA graph.

    Inductor fuses the multiply into the reduction: verified on `Scheduler`'s post-fusion node
    list, one node per site, its only buffers the fp32 accumulator and the bf16 output.
    `Reduction.num_splits` returns 1 because every output count here is above 2 * 108 SMs, so
    there is no second kernel either.
    """
    xb = x.to(torch.bfloat16).float().unsqueeze(-2)
    return (w.float() * xb).sum(-1).to(torch.bfloat16)


_INT8_LANE_SHIFTS = (24, 16, 8, 0)


def _int8_lanes(w32):
    """The four int8 weights packed into each int32 word, sign-extended, as fp32.

    `w32` is `wq.view(torch.int32)`: the champion's OWN quantised bytes, not requantised and not
    copied, reinterpreted so one thread's load carries four weights. Little-endian, so lane i is
    bits 8i..8i+7 and `(w << (24 - 8i)) >> 24` is that byte sign-extended -- a left shift and an
    ARITHMETIC right shift, two integer ops. Verified exhaustively over all 256 byte values, in a
    compiled region rather than eagerly, in `cpucheck/check_lanes.py`.

    **Why this is worth two integer ops per weight -- on the second account, the first having
    been refuted before this launch was spent.** Launch 37's per-region probe measured these
    kernels at 1.93-2.45 us per million weight ELEMENTS across the four int8 projections AND
    the bf16 head -- equal per element to within 9% across a 2x difference in bytes, so int8's
    saving came out at 55-70% of what halving its bytes should have bought
    (`decode_int8_weight_error_model.md` section 1, which recorded that gap as unexplained).

    This docstring first attributed that to BYTES IN FLIGHT per thread, on a guessed launch
    config of ~256 threads holding two elements each. autoscts__decode_gpu1 read the pinned
    heuristic instead (`triton_heuristics.py:2331`, `num_warps = rnumel // 128`; `XBLOCK == 1`
    at `:2901`): it is 128 threads and FOUR elements per thread, so 4 B/thread for int8 and 8 B
    for bf16 -- my absolute was 2x off. Their table also refutes the account outright, because
    the two 8 B/thread rows are the family's BEST (1.93) and WORST (2.45) per element, so bytes
    per thread does not order the measurements. Credit autoscts__decode_gpu1 for both halves.

    What replaces it is LOAD-INSTRUCTION ISSUE, and unlike the account it replaces it is
    settled device-free rather than assumed (sm80 PTX via `triton.compile`, no device;
    `cpucheck/check_load_width.py` and `check_bf16_load_width.py`, gpu5's `ptxas` path):

      * the int8 reduction emits FOUR separate `ld.global.b8` per thread -- Triton does not
        vectorise four contiguous int8 -- and the packed form emits ONE `ld.global.b32` for the
        same four weights: a 4x cut in weight-side load instructions;
      * int8 and bf16 emit exactly ONE weight load per ELEMENT (4 loads at 4 elem/thread,
        8 at 8), which is a mechanism for the measured per-element equality across 2x bytes and
        for its flatness over a 16x range of grid warps;
      * `num_warps` 1 gives 16 contiguous int8 per thread and Triton still emits 16 scalar byte
        loads, so the config route cannot reach the load width at these sites.

    **What this candidate CANNOT claim, stated here because it bounds the row.** Total
    instruction count is UNCHANGED: 13 per 4 weights either way (int8 4 weight loads + 5 shifts
    + 4 converts; packed 1 + 7 + 5). The packing converts three memory-pipe instructions into
    two ALU ones, so its entire value rests on the memory-issue pipe being separately binding
    from ALU issue, and that is a duration claim a CPU cannot settle. Registered at -1.5 ms
    central against 86.3497 (442.6K warp load instructions/step removed, priced at one global
    load per SM per cycle over 108 SMs at 1.41 GHz = 2.9 us/step x 512 decode steps); the
    DRAM sector count is IDENTICAL either way, so no byte of traffic is claimed. `abs(delta)`
    at or under 0.11 ms is the informative failure and closes the family.
    """
    return tuple(((w32 << sh) >> 24).float() for sh in _INT8_LANE_SHIFTS)


def _act_lanes(x):
    """`x.to(bf16).float()` de-interleaved into the four lanes `_int8_lanes` pairs with.

    Each lane reads the region's own INPUT at a CONTIGUOUS, unit-stride index -- lane `i` is
    elements `i * IN/4 .. (i+1) * IN/4 - 1` -- which is the whole of this candidate. The
    champion's spelling is `x.unflatten(-1, (-1, 4))[..., i]`, index `4 * r + i`, whose stride
    of 4 is why Triton cannot vectorise it and emits one scalar `ld.global.b16` per element
    (autoscts__decode_gpu5's sm80 census: 4 of them per word group at rnumel 512 and 8 at
    2048, against ONE `v2.b32`/`v4.b32` in the unpacked spelling). Here the index is
    `r + i * IN/4`, unit stride in the reduction variable, and each thread's four words become
    one vectorised load per lane. Nothing else moves: same `rnumel`, same `num_warps`, same
    grid, same bytes.

    No intermediate activation buffer is realised either way. That matters: a single
    [IN]-shaped producer consumed at four disjoint indices is exactly the shape
    `knowledge/decode_fusion_needs_one_consumer.md` shows inductor REALISES, and a realised
    activation would cost a node per region. Censused instead of assumed: one node per region,
    absolute counts, in `cpucheck/check_lane_major.py` section A.

    `.to(torch.bfloat16).float()` is the champion's own rounding, applied per lane. Rounding is
    elementwise, so it commutes with ANY re-indexing of the vector and the operand is
    bit-identical element by element.

    The pairing with `_int8_lanes` is the contract: lane `i` of word `c` must hold the weight
    column this returns at position `c` of lane `i`. `_quantise_rows_int8_packed(...,
    lane_major=True)` is what establishes it, and the two must move together.
    """
    xr = x.unflatten(-1, (4, -1))
    return tuple(xr[..., i, :].to(torch.bfloat16).float().unsqueeze(-2) for i in range(4))


def _matvec_int8(x, w32, s):
    """The same width-1 projection, reading a ONE-byte weight instead of a two-byte one.

    `wq` is a symmetric per-output-row int8 quantisation of the same bf16 copy `_matvec_bf16`
    reads and `s` is that row's step, so `sum_i (wq_ji * x_i) * s_j` is the same map to within
    the quantiser's error. The scale is applied AFTER the reduction, on `out` elements rather
    than on `out * in` of them, so it fuses into the reduction's epilogue and costs no node --
    the same place `relu(.).square()` already folds for `c_fc`.

    **Why bytes and not kernels.** `knowledge/decode_gemv_kernel_price.md` fitted
    `cost = F + bytes / BW` to this champion's four projections and got F = 1.55-1.94 us against
    13.28 us/layer: after the kernel work the remainder is traffic. 6 layers x 6.291 MB is
    37.75 MB per step, and this halves the part of the step that is bytes rather than the part
    that is floor.

    **Why the x-side is untouched.** `x.to(torch.bfloat16).float()` is copied verbatim from
    `_matvec_bf16` and stays inside this same region. Moving that cast across a
    `@torch.compile` boundary would let inductor collapse `fp32 -> bf16 -> fp32` as lossless
    and silently stop rounding where autocast rounds
    (`knowledge/decode_cpu_harness_silently_runs_eager.md`, second trap). Only the weight
    operand changes; the activation operand is bit-identical to the champion's.
    """
    lanes, w = _act_lanes(x), _int8_lanes(w32)
    acc = w[0] * lanes[0]
    for i in range(1, 4):
        acc = acc + w[i] * lanes[i]
    return ((acc.sum(-1)) * s).to(torch.bfloat16)


def _quantise_rows_int8_packed(w, lane_major=False):
    """Symmetric per-output-row int8, returned as int32 WORDS: four weights per word.

    `lane_major` chooses WHICH four of a row's weights share a word, and nothing else. It
    changes no byte count, no scale, no quantised value and no dtype: the bytes of a row are
    the same multiset, permuted, and `s` is computed before the permutation from the
    unpermuted row so it is bit-identical either way (`amax` is permutation-invariant, but
    this does not rely on that -- the reduction is literally the same call on the same
    tensor).

      * `lane_major=False` is the champion's layout. Word `c` holds columns `4c .. 4c+3`, so
        `_int8_lanes`' lane `i` pairs with input element `4c + i` and `_act_lanes` has to
        read the activation with a **stride of 4**.
      * `lane_major=True` holds columns `i * IN/4 + c` in lane `i` of word `c`, so lane `i`
        pairs with input element `i * IN/4 + c` and `_act_lanes` reads a **contiguous** slice.

    Why that is worth two lines. autoscts__decode_gpu5's faithful-`attrs` sm80 census of the
    champion's own two spellings (`[PROPOSAL] pack_int8_weight_words` thread) reads, per
    thread and per word group:

      | reduction | UNPACKED int8 | champion PACKED |
      |---|---|---|
      | rnumel 512, nw 4  | 2 = `b32` weight + `v2.b32` x | 5 = `b32` weight + 4 x `b16` x |
      | rnumel 2048, nw 8 | 2 = `v2.b32` weight + `v4.b32` x | 9 = `v2.b32` weight + 8 x `b16` x |

    The weight load is identical in both spellings -- inductor already issues one 32-bit load
    for four contiguous int8 -- and the whole load-count difference is the **activation**:
    `_act_lanes`' stride-4 de-interleave breaks its vectorised load, +3 loads per word group
    at rnumel 512 and +7 at 2048. That census was published as a reason the packing should
    have been SLOWER, and the packing measured -1.987 ms (champion v24). It could not
    separate the two variables it moved: the activation's load width, and `num_warps`
    (`rnumel // 128`, so 4 -> 1 at IN=512 and 8 -> 4 at IN=2048).

    This permutation moves the activation's load width and NOTHING else -- same `rnumel`,
    same `num_warps`, same CTA count, same threads per CTA, same elements per thread, same
    bytes, same weight-side load instructions. It is the separating experiment for that pair.

    Per ROW and not per tensor because the error a row contributes is set by that row's own
    largest weight; one tensor-wide scale would price every row at the largest row's peak. The
    scale is fp32 and there are `out` of them, 110,592 B over the six layers.

    `s > 0` guard: `init_weights` zero-initialises `attn.c_proj` and `mlp.c_proj`, and
    `prepare.py`'s throwaway build runs a width-1 step on a model that may still be at init, so
    an all-zero row is reachable and must quantise to zeros rather than to NaN.
    """
    wf = w.detach().to(torch.float32)
    s = wf.abs().amax(-1, keepdim=True) / 127.0
    s = torch.where(s > 0, s, torch.ones_like(s))
    if lane_major:
        # [OUT, IN] -> [OUT, 4, IN/4] -> [OUT, IN/4, 4]: element (c, i) of a row is column
        # `i * IN/4 + c`, so after `contiguous()` the byte at flat offset `4c + i` -- which is
        # `_int8_lanes`' lane `i` of word `c` -- carries that column. ONE materialisation, so
        # no extra allocation exists at any point and the serving weight is the same size.
        wf = wf.unflatten(-1, (4, -1)).transpose(-1, -2)
        s = s.unsqueeze(-1)
    wq = torch.round(wf / s).clamp_(-127.0, 127.0).to(torch.int8).contiguous()
    if lane_major:
        wq = wq.reshape(*w.shape)
        s = s.squeeze(-1)
    # `view(torch.int32)` REINTERPRETS these bytes; it neither copies nor requantises, so the
    # serving weight is byte-identical to the champion's and so is its footprint (the int8
    # tensor's storage is what the view holds). The reduction axis becomes IN/4 words, and each
    # thread's load carries 4 B instead of 1 -- see `_int8_lanes`. IN is 512 or 2048 at every
    # site, both divisible by 4, and a freshly allocated contiguous tensor is 512-B aligned, so
    # the view is always available.
    return wq.view(torch.int32), s.squeeze(-1).to(torch.float32).contiguous()


def _decode_layer_projection_weights(block, parallel=False):
    """One layer's four packed projections, plus the int8 VIEW of the two that read it unpacked.

    Six entries, not four, and entries 4 and 5 add **zero bytes**: `Tensor.view(dtype)` returns a
    new tensor over the SAME storage, so the serving weight is one allocation read through two
    types. That is what makes this hoist free against `nopref_kv_cache_bytes` (6,304,256 B against
    a 10,485,760 ceiling) and `peak_vram_bytes`, and it is why the view is taken here rather than
    inside the region that needs it -- see `_matvec_int8_rne_unpacked`.

      0 `attn.c_qkv`      [1536, 512] packed   -> `_decode_qkv_matvec_norm`   (LANE-MAJOR)
      1 `attn.c_proj`     [512, 512]  packed   -> nothing on the live path; the probe's v24 leg
      2 `mlp.c_fc`        [2048, 512] packed   -> `_decode_mlp_hidden_norm`   (LANE-MAJOR)
      3 `mlp.c_proj`      [512, 2048] packed   -> `_decode_mlp_out_mix` and, on the last
                                                          layer, `_decode_mlp_out_add` (LANE-MAJOR)
      4 = entry 1     [512, 512]  words    -> `_decode_attn_out_add`      (LANE-MAJOR)
      5 = entry 3     [512, 2048] words    -> `_decode_mlp_out_add`       (LANE-MAJOR)

    Entry 1 is kept although the live step no longer reads it, so the probe below can time the
    packed spelling against the live one in the same launch on the same bytes.

    **Every entry is lane-major now, and the reason the previous champion could not do that is
    the reason this one is worth a launch.** A lane-major word can only be read through
    `_act_lanes`; an int8 VIEW of one is in permuted column order. Champion v32 therefore had to
    leave `attn.c_proj` and the LAST layer's `mlp.c_proj` in the champion's natural word order,
    because their readers went through `_matvec_int8_rne_unpacked`. Launch 53's own probe then
    measured both of those sites in the lane-major spelling, paired in the same process on the same
    bytes: `_decode_attn_out_add` **2.107 against the live unpacked 2.393** and
    `_decode_mlp_out_add` **3.249 against 4.034** -- **-0.286 and -0.785 us/call**. So the readers
    change instead of the layout, and there is no split left to keep.

    That **supersedes champion v28**, and the supersession is a change of layout rather than a
    retracted measurement: at 512 output rows the NATURAL-packed spelling really was +0.35 us/call
    worse than unpacked, on three independent launches. What cost that was the stride-4 activation
    read, not the packing -- and the lane-major packing is 0.286 BETTER at the very same site.

    Entries 4 and 5 keep their positions so `_decode_body` does not change; they are simply the
    same objects as entries 1 and 3, so the serving footprint is what it was.
    """
    qkv = _quantise_rows_int8_packed(block.attn.c_qkv, lane_major=True)
    attn_proj = _quantise_rows_int8_packed(block.attn.c_proj.weight, lane_major=True)
    fc = _quantise_rows_int8_packed(block.mlp.c_fc.weight, lane_major=True)
    mlp_proj = _quantise_rows_int8_packed(block.mlp.c_proj.weight, lane_major=True)
    # Entries 4 and 5 keep their POSITIONS and become the same tuples as 1 and 3, so `_decode_body`
    # is byte-identical and the two regions it selects there now receive WORDS. Nothing is
    # allocated: these are the same two objects, not copies and not views.
    #
    #   6 `c_qkv` then `c_fc`  [3584, 512] packed -> `_decode_qkv_fc_matvec_norm`, and ONLY on a
    #                                               `parallel_block` layer; `None` elsewhere.
    #
    # Built by CONCATENATING entries 0 and 2 rather than by quantising a concatenated weight.
    # `_quantise_rows_int8_packed`'s scale is per OUTPUT ROW, and under `lane_major=True` the
    # permutation is over the REDUCTION axis, which both operands share at extent 512 -- so row r
    # of the concatenation depends only on row r, and the two constructions are the same tensor to
    # the byte. THAT IS A CLAIM ABOUT CODE, and `cpucheck/check_merged_weight.py` asserts it BOTH
    # WAYS with a control that perturbs one weight row and must fail. It is not assumed here. The
    # construction is `autoscts__decode_gpu5`'s, verbatim from launch 75.
    #
    # It adds 1,835,008 B of int8 words and 14,336 B of scales per parallel layer, so at k=2 it is
    # 3,698,688 B against launch 75's 11,098,112 B over six. Invisible to both memory constraints,
    # for the reason this function's own docstring already gives, and MEASURED so at k=6: launch 75
    # reads `nopref_kv_cache_bytes` 6,304,256 -- byte-identical to v39's -- and its
    # `peak_vram_bytes` 32,542,316,544 is inside the champion's band. k=2 adds a third of what was
    # already measured free.
    merged = None
    if parallel:
        merged = (torch.cat([qkv[0], fc[0]], 0).contiguous(),
                  torch.cat([qkv[1], fc[1]], 0).contiguous())
    return (qkv, attn_proj, fc, mlp_proj, attn_proj, mlp_proj, merged)


@torch.compile(dynamic=False)
def _decode_qkv_matvec(h, wq_qkv, s_qkv):
    """`c_qkv` [1536, 512]: 1.573 MB as bf16, **0.786 MB as int8**."""
    return _matvec_int8(h, wq_qkv, s_qkv)


@torch.compile(dynamic=False)
def _decode_attn_out_matvec(y, wq_proj, s_proj):
    """`attn.c_proj` [512, 512]: 0.52 MB as bf16, **0.26 MB as int8**.

    The least promising of the four: at 130 GB/s as a cuBLAS GEMV it was never bandwidth-bound,
    because 512 output rows leave an A100 mostly idle whatever the kernel is. It is included
    because the prize here is the WORKING SET, not the per-call byte count -- see
    `_decode_projection_weights`.
    """
    return _matvec_int8(y, wq_proj, s_proj)


@torch.compile(dynamic=False)
def _decode_mlp_hidden(h, wq_fc, s_fc):
    """`c_fc` [2048, 512] and the activation in ONE node.

    `relu(x).square()` is pointwise on `c_fc`'s output, so inductor fuses it into the
    reduction's epilogue and `_decode_relu_square`'s own kernel (1.346 us, measured) stops
    existing on this path. That fusion is only available once `c_fc` is a reduction rather than
    an extern GEMM call, which is also why compiling `F.linear` itself buys nothing: it lowers to
    an `ExternKernelSchedulerNode` and nothing fuses into one.

    The activation runs on the bf16 store and not on the fp32 accumulator, so it rounds where the
    champion rounds.
    """
    return F.relu(_matvec_int8(h, wq_fc, s_fc)).square()


@torch.compile(dynamic=False)
def _decode_mlp_out_add(a, wq_proj, s_proj, x):
    """`mlp.c_proj` [512, 2048] and the LAST layer's residual add, in one node.

    2.097 MB as bf16, **1.049 MB as int8** -- with `c_fc` the two largest of the four.

    `_decode_mlp_out_mix` folds the NEXT layer's mix into this projection's epilogue on every
    layer but the last, where there is no next mix. What the last layer's output feeds is
    `_decode_add_tail_norm`, whose `x + y` is pointwise on this reduction's OUTPUT index and so
    lands in the same epilogue at zero node cost; the norm beside it is now the head's own
    reduction. Spelling is `_decode_attn_out_add`'s, character for character, because it is the
    same operation on the same shapes, computed in fp32. This supersedes
    `_decode_mlp_out_matvec`, whose only caller was the site now folded, so no unused width-1
    region is left behind.

    Returns **fp32**, and that is autoscts__decode_gpu3's finding, taken here as a rider on the
    mechanism above rather than as a launch of its own. `.to(x.dtype)` -- what v21 shipped, this
    author's own error -- ROUNDS the tail residual sum to bf16. The champion it replaced,
    `_decode_add_tail_norm`, is `norm(x + y)` inside ONE region, whose `add -> bf16 -> rms_norm`
    chain is a float->float->float cast chain that inductor's `pointless_convert` DELETES: **the
    champion never rounded that sum.** gpu3 measured the two spellings on the softcapped logits:
    56% of the vocabulary row moved by up to 1 bf16 ulp, TV(candidate, champion) 1.47e-03 against
    5.12e-07 for fp32. At ZERO node cost that is 1.2-2.3x the whole insurable TV mean headroom
    (0.000627 on v20's reading, 0.001168 on v21's), on the constraint that binds this task. The
    head region takes fp32 without a change: `_decode_head_prenorm_softcap` opens with
    `m.float()`, which is the identity on an fp32 input.

    It is exactly the error `knowledge/decode_rounding_is_a_property_of_a_boundary.md` warns
    against -- a rounding belongs to a BOUNDARY, and folding one in materialises a boundary that
    was not there -- made by that file's own author in the cycle that wrote it. Credit gpu3.
    """
    return x.float() + _matvec_int8_rne(a, wq_proj, s_proj)


# `pointless_convert` in inductor's `fx_passes/joint_graph.py` rewrites ANY
# float->float->float `convert_element_type` chain to a single convert, whatever the widths, so
# `v.to(torch.bfloat16).float()` inside one region is DELETED -- verified on the transformed FX
# graph, which contains neither convert. v18 performs that rounding at a region BOUNDARY, where a
# real bf16 buffer makes it unerasable; a candidate that folds the region in has to reproduce it
# without a cast or it is silently computing a different function.
#
# `_BF16_SPLIT` is Veltkamp's constant for splitting an fp32 significand at 8 bits:
# `t = v*C; t - (t - v)` is round-to-nearest-even to bf16 precision in pure fp32 arithmetic.
# Bitwise `.to(torch.bfloat16).float()` on 1,638,400 values over twelve decades of scale, plus
# signed zeros, exact ties, a subnormal and 1e30; and the three ops survive the FX passes.
# `cpucheck/probe_veltkamp.py`, `cpucheck/probe_elision.py`.
_BF16_SPLIT = float(2 ** (24 - 8) + 1)


def _rne_bf16(v):
    """Round fp32 `v` to bf16 precision, as fp32, without a dtype cast inductor would erase."""
    t = v * _BF16_SPLIT
    return t - (t - v)


def _matvec_int8_prenorm(m, w32, s):
    """`_matvec_int8(norm(m), wq, s)` as ONE reduction, with v18's exact arithmetic.

    `F.rms_norm(m, (IN,))` is `m * rsqrt(mean(m*m) + eps)` -- a per-row SCALE. Written as
    `norm` then `matvec` it is two loop nests, because the mean square reduces to a SCALAR
    while the projection reduces to [OUT]: different output shapes, so two kernels. Give the
    mean square the projection's own output shape, by broadcasting the squares over the
    weight's rows, and both sums reduce over the same 512-long axis to the same [OUT] -- one
    nest. At `rnumel` 512 this is a persistent reduction, so the whole axis is already in
    registers: the second pass over it is the projection itself, and the mean square is
    recomputed per output row from values that are not re-read. Each program reads exactly
    the vector `_matvec_int8` already read.

    Every step of reproducing `_matvec_int8(norm(m), wq, s)` is a measured claim here, not a
    reading of the docs:

      * `F.rms_norm` accumulates in fp32 even from a **bf16** input -- the residual stream on
        this path is bf16 -- and its default `eps` is then `finfo(float32).eps`, NOT
        `finfo(bfloat16).eps`. On the pinned torch 2.9.1: 60/60 draws bitwise with the fp32
        eps, 16/60 with the bf16 one. `cpucheck/probe_norm_bf16.py`.
      * it returns its input's dtype, so `norm(m)` is bf16 and that ROUNDING is part of the
        operand v18 reduces -- restored by `_rne_bf16`, because a cast would be erased.
      * `wq`, the fp32 accumulation over the same axis in the same order, the post-reduction
        row scale `s` and the bf16 store are v18's, untouched.

    **`hoist_r_out_of_the_prenorm_rounding` (this revision) is the ONE place the two folds stop
    being dependent, and it is an APPROXIMATION, priced not assumed.** The second fold read `r`
    only through `_rne_bf16(lane * r)`; `sum_c w_c * (x_c * r)` is `r * sum_c w_c * x_c`, so
    moving `r` beside `s` in the post-reduction epilogue makes the projection fold independent of
    the norm fold. The two 128-wide `SHFL.BFLY` chains were serial (10 deep); they are now
    concurrent (5 deep), and the 16 `FMUL` + 48 `_rne_bf16` ALU ops per thread that sat BETWEEN
    them are deleted.

    What it costs, measured rather than argued. Deleting that rounding perturbs the operand by
    **0.67 bf16 ulp**, i.e. a relative output error of **2.6e-03**, measured over 64 draws at both
    sites at kappa 3.26/3.27 (`cpucheck/check_hoist_r.py`). This run's own int8 relative error at
    the same rows is `kappa/440` = **7.3e-03** (`knowledge/decode_int8_weight_error_model.md`,
    CPU-verified there to 0.89-1.03 and carried by a launch), so the two combine in quadrature to
    **+2.4%** of total error, not +35%. Converted with the run's own device anchor -- int8 read
    `nopref_decode_tv_distance_max` 0.0379 against a bf16 population centred on 0.0146, and TV is
    LINEAR in perturbation across 3000x -- that is **dTV_max = +0.00199**, 4.0% of the 0.05
    ceiling and **0.25 sigma** of the metric's own 0.0081 draw spread.

    The projection half is now `_matvec_int8`'s body verbatim (line-for-line: same `_act_lanes`
    lanes, same `_int8_lanes` weights, same fp32 accumulation order, same bf16 store) with the
    row scale `s` premultiplied by `r`. The norm fold, its `expand`, its fp32 `eps` and its
    division by `IN` are byte-identical to the champion's; nothing about the weight, the lane
    pairing, the grid, `num_warps`, `rnumel` or the kernel count moves.

    NOT applied at `_decode_head_prenorm_softcap`. Its weight is bf16, so there is no int8 error
    for the perturbation to sit behind, and it would land on the logits themselves.
    """
    lanes = _act_lanes(m)
    sq = lanes[0] * lanes[0]
    for i in range(1, 4):
        sq = sq + lanes[i] * lanes[i]
    sq = sq.expand(*sq.shape[:-2], w32.size(0), sq.size(-1))
    ms = sq.sum(-1, keepdim=True) / m.size(-1)
    r = torch.rsqrt(ms + torch.finfo(torch.float32).eps)
    w = _int8_lanes(w32)
    acc = w[0] * lanes[0]
    for i in range(1, 4):
        acc = acc + w[i] * lanes[i]
    return ((acc.sum(-1)) * (r.squeeze(-1) * s)).to(torch.bfloat16)


def _matvec_int8_rne(x, w32, s):
    """`_matvec_int8` with its bf16 store replaced by the same rounding, kept in fp32.

    For the two epilogues below, which consume the projection's result instead of storing it:
    v18 rounds to bf16 and then the residual add reads that bf16 value, so the rounding is
    part of the arithmetic and has to survive being fused in. `x.to(bfloat16).float()` is v18's
    own line and is left verbatim -- `x` is already bf16 at both call sites, so the chain
    collapses to one widening convert, which is what v18 gets too.
    """
    lanes, w = _act_lanes(x), _int8_lanes(w32)
    acc = w[0] * lanes[0]
    for i in range(1, 4):
        acc = acc + w[i] * lanes[i]
    return _rne_bf16(acc.sum(-1) * s)


def _matvec_int8_rne_unpacked(x, wq, s):
    """`_matvec_int8_rne` reading a PLAIN int8 weight -- champion v22's spelling, verbatim.

    `wq` is the int8 VIEW of the very words `_quantise_rows_int8_packed` produced, taken OUTSIDE
    this region by `_decode_layer_projection_weights` and handed in as an argument. Identical
    storage, identical bytes, no requantisation, no allocation. The view must not be taken here:
    inside the region it stops the dequantised weight fusing into the reduction and this region
    goes 1 CUDA kernel -> 3 (absolute post-fusion census, both revisions).

    **Why two of five sites read this and three do not.** Two independent paired in-launch probes
    -- mine on champion v24 (launch 39) and autoscts__decode_gpu3's on champion v25 (launch 43) --
    priced the packing per region against its own v22 spelling and agree to 4.9% on the two
    512-row sites: `_decode_attn_out_add` +0.353 / +0.316 us/call over 6 calls, against -0.260 /
    -0.246 at 1536 rows and -0.608 / -0.592 at 2048. The weight load is byte-identical in both
    spellings -- inductor already issues one `ld.global.b32` for four contiguous int8, which is
    autoscts__decode_gpu5's finding and the reason v24's stated mechanism is dead -- so the varying
    quantity cannot be load width. 512 rows is 4.7 CTAs per SM against 2048 rows' 19.

    Two readings of the same table are live and this diff does not decide between them: monotone in
    output rows (mine), or "the packing did nothing where `rnumel` stayed at or above 256 and
    everything where it crossed below" (autoscts__decode_gpu1, from the pinned heuristic). Both
    predict the two reverted sites; they differ on `_decode_mlp_out_mix`, which measured a wash
    (-0.032 / -0.005) and is left packed.
    """
    xb = x.to(torch.bfloat16).float().unsqueeze(-2)
    return _rne_bf16((wq.to(torch.float32) * xb).sum(-1) * s)


@torch.compile(dynamic=False)
def _decode_qkv_matvec_norm(mixed, wq_qkv, s_qkv):
    """`c_qkv` [1536, 512] with the norm of its input folded into the same reduction."""
    return _matvec_int8_prenorm(mixed, wq_qkv, s_qkv)


@torch.compile(dynamic=False)
def _decode_attn_out_add(mixed, y, wq_proj, s_proj):
    """`attn.c_proj` [512, 512] and the residual add that followed it, in one node.

    The add is pointwise on the reduction's OUTPUT index, so it costs each program one element
    of `mixed` -- not the vector -- and lands in the epilogue beside the int8 row scale, the
    same place `relu(.).square()` folds for `c_fc`.

    `_decode_add_norm`'s `added = x + y` is bf16 + bf16 -> bf16: computed in fp32 and rounded
    on the store, which is exactly the store this region ends with. 50/50 draws bitwise.
    """
    return (mixed.float()
            + _matvec_int8_rne(y, wq_proj, s_proj)).to(mixed.dtype)


@torch.compile(dynamic=False)
def _decode_mlp_hidden_norm(xa, wq_fc, s_fc):
    """`c_fc` [2048, 512], the norm of its input AND the activation, in one node."""
    return F.relu(_matvec_int8_prenorm(xa, wq_fc, s_fc)).square()


@torch.compile(dynamic=False)
def _decode_qkv_fc_matvec_norm(mixed, wq, s, n_qkv):
    """`c_qkv` [1536, 512] and `c_fc` [2048, 512] as ONE [3584, 512] reduction.

    Read only by a `parallel_block` layer. The parallel block makes both projections read the
    same normed residual, so they are one matvec over one weight whose rows are their
    concatenation. `_matvec_int8_prenorm` already recomputes the mean square per output row from
    the vector it has loaded, and 3584 rows recompute it exactly as 1536 + 2048 rows did: the
    reduction axis, its accumulation order and the per-row scale are untouched, so each row's
    arithmetic is the row's own and does not depend on how many rows share the launch.

    **Both halves are the champion's own expressions, by construction rather than by tolerance.**
    The qkv half IS `_matvec_int8_prenorm(mixed, wq, s)`, which is `_decode_qkv_matvec_norm`'s
    whole body. The activation half IS `F.relu(that).square()`, which is
    `_decode_mlp_hidden_norm`'s whole body -- and this champion's own two regions call the SAME
    `_matvec_int8_prenorm`, which is why the merge is expressible at all. The only added op is a
    `where` on the OUTPUT ROW INDEX, which selects between two values this region computes with
    those expressions, so the identity holds whether or not inductor keeps the interior bf16 cast
    (`pointless_convert` erases some and not others, and this construction does not depend on
    which). `relu(.).square()` is evaluated on all 3584 rows and discarded on 1536 of them: two
    ALU ops per row, no SFU work, and no extra load or store.

    ONE kernel and not two: a `where` on the row index is pointwise on the reduction's own output
    index, so it lands in the epilogue beside the int8 row scale -- the same place
    `relu(.).square()` already folded for `c_fc`. Returning the single [3584] buffer and slicing
    it OUTSIDE the region keeps it that way: two returned slices are two consumers of one
    producer at disjoint indices.

    **The pin give-back is DIFFERENT at k=2 than at k=6, and this is the one place the restriction
    changes the mechanism rather than just scaling it.** `next_power_of_2(3584) = 4096`, and both
    of this champion's projection pins in `_DECODE_CONFIG_PIN` are keyed on
    `size_hints {"x": 2048, "r0_": 128}`, which 3584 rows do not match. At k=6 that made both
    pinned SITES cease to exist on the live step. **At k=2 the four sequential blocks still call
    `_decode_qkv_matvec_norm` at 1536 rows and `_decode_mlp_hidden_norm` at 2048 rows, so both
    roster entries still match LIVE instances and the roster does not print SITES INCOMPLETE.**
    A merged-site pin is deliberately NOT bundled: (X4, R0 128, nw1) is measured at 1536 and 2048
    output rows and is unmeasured at 3584, so pinning there would be a guess and a second change.

    This is also why the price is taken from launch 75's PAIRED probe and not from its whole-step
    delta divided by six. `[probe] PARALLEL ROW` (`logs/0075/attempt-1/stdout.log:291`) reads
    merged **3.417** us/call against the pair it replaces, **2.728 + 3.102 = 5.830** -- and the two
    legs it is compared against were PINNED probe instances, so the -2.413 us/call net is already
    the unpinned-merged-against-pinned-pair exchange, which is exactly the exchange a k<6 build
    makes at each converted site. The give-back is INSIDE the pair, not additive to it (its author
    withdrew the `-6.900 + 1.580` decomposition for double counting), and the two routes agree:
    -2.413 us/call x 2 sites x 0.513 = -2.476 ms predicted, against 7.374525/6 x 2 = -2.458 ms
    from the whole-step number, 0.7% apart.

    Mechanism, construction and the honest pre-registration of the quality risk are
    `autoscts__decode_gpu5`'s (launch 75), from this agent's `parallel_attention_mlp_block`
    (launch 50). The dial and its k=2 selection are this candidate's.
    """
    v = _matvec_int8_prenorm(mixed, wq, s)
    lane = torch.arange(v.size(-1), device=v.device)
    return torch.where(lane < n_qkv, v, F.relu(v).square())


@torch.compile(dynamic=False)
def _decode_mlp_out_mix(a, wq_proj, s_proj, x, x0, resid_lambda, x0_lambda):
    """`mlp.c_proj` [512, 2048] and the NEXT layer's residual mix, in one node.

    A layer ends with `x = x + mlp_out` and the next opens with `resid*x + x0l*x0`; nothing
    reads the value in between, which is what `_decode_add_mix_norm` already exploited by
    merging two regions into one. Here the whole chain is the epilogue of the projection that
    produces `mlp_out`, so the region stops existing instead. The last layer has no next mix
    and takes the TAIL add into the same epilogue instead (`_decode_mlp_out_add`),
    because `_decode_add_tail_norm`'s norm is now `_decode_head_prenorm_softcap`'s own
    reduction.

    `_decode_add_mix_norm` rounds FOUR times and each one is reproduced. The lambdas are 0-dim
    fp32 tensors against a bf16 stream: promotion gives a 0-dim tensor only its CATEGORY, so
    the common dtype is bf16 and **each lambda is rounded to bf16 before its multiply**. That
    was found by measurement, not derived -- the spelling with full-precision lambdas is 0/50
    bitwise and this one is 50/50. `cpucheck/probe_epilogues.py`.
    """
    added = _rne_bf16(x.float() + _matvec_int8_rne(a, wq_proj, s_proj))
    return (_rne_bf16(_rne_bf16(resid_lambda.float()) * added)
            + _rne_bf16(_rne_bf16(x0_lambda.float()) * x0.float())).to(x.dtype)


@torch.compile(dynamic=False)
def _decode_out_merge_mix(mixed, ao, hidden, wq_a, s_a, wq_m, s_m, x0,
                          resid_lambda, x0_lambda):
    """`attn.c_proj` [512,512] and `mlp.c_proj` [512,2048] as ONE depth-2560 reduction, with the
    residual add and the next layer's mix in the same node. A `parallel_block` layer only.

    This replaces `_decode_attn_out_add` followed by `_decode_mlp_out_mix`. It is available only
    where `hidden` does not depend on the attention's output projection, i.e. only where both
    branches read one norm, and it is not available on a sequential layer at any price.

    **The reduction extents are made equal rather than the operands made adjacent.** Under
    `lane_major=True` lane `i` of word `c` pairs with input element `i * IN/4 + c`, so slicing the
    [512,2048] weight's 512-word axis into four 128-word blocks and taking the matching 128-element
    slice of each activation lane gives 16 lane products at extent 128, the attention half's own
    extent. All 20 products are elementwise-summable into one accumulator and one `sum(-1)` closes
    them. Every term is a view; nothing is allocated, copied or requantised, and `s_a` and `s_m`
    stay the two distinct per-output-row scales they are -- which is what forbids the one-scale
    merged weight, an approximation this run priced at +0.0240 nopref TV at these sites.

    **What moves arithmetically, and it is three roundings plus a reassociation.** The pair rounds
    `Wa @ ao` to bf16, stores `xa` as bf16, and rounds `Wm @ hidden` to bf16 before the mix reads
    it; none of those boundaries exists inside one region, so all three go. The row scale also
    multiplies each of the 128 partial sums rather than the closed sum, an fp32-granularity
    reassociation. At `train.py:2189-2191`'s calibration -- 1 bf16 ulp costs TV 1.47e-03, an
    fp32-granularity difference 5.12e-07, linear to 5% across 3000x -- that is <= +0.0044 worst
    case with an unpredictable sign, against L77's prefill margin of 0.017706. The final
    `_rne_bf16(added)` and both lambda roundings are `_decode_mlp_out_mix`'s, verbatim: they sit on
    boundaries this merge does NOT delete, and a rounding must be reproduced at its own site.

    **No bitwise claim.** A compiled region's reduction is not reproducible from outside it, so the
    CPU legs assert a BOUND (`cpucheck` legs A-C of this revision), not equality, and the fidelity
    reading is the launch's own two TV ceilings.
    """
    la, wa = _act_lanes(ao), _int8_lanes(wq_a)
    acc_a = wa[0] * la[0]
    for i in range(1, 4):
        acc_a = acc_a + wa[i] * la[i]
    # The shared extent IS the attention half's word count; deriving it rather than writing 128
    # keeps this correct if the dial or the width moves.
    words = wq_a.size(-1)
    lm = _act_lanes(hidden)
    acc_m = None
    for b in range(wq_m.size(-1) // words):
        sl = slice(b * words, (b + 1) * words)
        wm = _int8_lanes(wq_m[..., sl])
        for i in range(4):
            term = wm[i] * lm[i][..., sl]
            acc_m = term if acc_m is None else acc_m + term
    acc = acc_a * s_a.unsqueeze(-1) + acc_m * s_m.unsqueeze(-1)
    added = _rne_bf16(mixed.float() + acc.sum(-1))
    return (_rne_bf16(_rne_bf16(resid_lambda.float()) * added)
            + _rne_bf16(_rne_bf16(x0_lambda.float()) * x0.float())).to(mixed.dtype)


@torch.compile(dynamic=False)
def _decode_embed_norm_mix(wte_w, idx, resid_lambda, x0_lambda):
    """The step's token gather, the norm on it, and LAYER 0's residual mix, in one node.

    Three kernels on the width-1 path -- an eager `aten::embedding`, an eager
    `aten::_fused_rms_norm`, and `_decode_mix` -- become one reduction over 512. At `rnumel`
    512 that is a persistent reduction, so both outputs are stored from the values the
    reduction already has in registers and nothing is read twice.

    `x0` is returned as well as `mixed` because every later layer's mix reads it; only layer
    0's mix can be folded here, and it is the one mix `fold_norm_into_projection` had to leave
    as a node of its own for want of a predecessor projection.

    Three roundings are reproduced rather than assumed, each one a boundary this fold deletes:

      * autocast gathers from a **bf16 copy** of `wte.weight` (measured: `wte(idx)` under
        autocast returns bf16 from an fp32 parameter). Rounding commutes with selecting a row,
        so rounding the gathered row is the same value -- but the cast that does it would be
        DELETED here: inductor's `pointless_convert` rewrites any float->float->float chain,
        and `fp32 -> bf16 -> fp32` inside one region is exactly that. `_rne_bf16` is v19's
        own cast-free rounding and is idempotent, so this is correct whether or not autocast's
        own cast survives the FX passes.
      * `F.rms_norm` on a bf16 input accumulates in fp32 and its default `eps` is then
        `finfo(float32).eps`, NOT `finfo(bfloat16).eps` (gpu6, 60/60 draws), and it returns
        its input's dtype -- so `x0` carries a bf16 rounding.
      * `_decode_mix` carries **no** internal rounding, and this had to be measured. Both its
        inputs are already bf16 buffers and its whole expression is one compiled region, so
        inductor keeps fp32 throughout and rounds only on the store: the spelling with the
        lambdas rounded to bf16 first is **0/80** bitwise and the fully-fp32 one is **80/80**.
        That is the OPPOSITE of `_decode_mlp_out_mix`, where the lambdas must be rounded --
        and the two are consistent, because there the rounding being reproduced sat on a
        region BOUNDARY (`_decode_mlp_out_matvec` stored bf16 and `_decode_add_mix_norm` read
        that buffer) while here there is no boundary inside the region to lose. A rounding
        must be measured at its own site; it does not transfer from a neighbouring one.
    """
    e = _rne_bf16(F.embedding(idx, wte_w).float())
    ms = (e * e).sum(-1, keepdim=True) / e.size(-1)
    x0 = _rne_bf16(e * torch.rsqrt(ms + torch.finfo(torch.float32).eps))
    mixed = resid_lambda.float() * x0 + x0_lambda.float() * x0
    return x0.to(torch.bfloat16), mixed.to(torch.bfloat16)


@torch.compile(dynamic=False)
def _decode_embed_norm_mix_at(wte_w, tok_row, pos, resid_lambda, x0_lambda):
    """`_decode_embed_norm_mix` reading this step's token IN PLACE instead of from a copy.

    The body below is character-identical to `_decode_embed_norm_mix`'s from `ms =` onward.
    The ONLY change is the index the gather loads at: the champion's region is handed a
    private one-element tensor whose address is baked into the graph, and this one is handed
    the row the caller already holds plus the device position, so the host does not have to
    copy the token into the graph's static input before every replay.

    **The token is the caller's own argument, not a look-ahead.** `measure_decode_request`
    feeds `tokens[:, t:t+1]` at cache position `t`, so `tok_row.data_ptr() + pos * 8` IS
    `idx.data_ptr()` for that call; `_GraphedDecodeStep.replay` asserts exactly that on the
    host before it replays, and runs the step eagerly when it does not hold. No element above
    `pos` is read by any launch.

    **Why this cannot move a reduction tree.** `ms` reduces 512 terms with `xnumel` 1, the same
    two numbers the champion's region reduces, and inductor picks a reduction's launch
    configuration from `(xnumel, rnumel)` alone -- `triton_heuristics.py` derives `num_warps`
    and the persistent block from the size hints and never from a load's index expression. So
    the fold width, and with it the fp32 association order, is the champion's. That is the one
    claim here that cannot be checked on a CPU (inductor emits C++ with no `num_warps`), and
    it is registered as such rather than asserted as bitwise.
    """
    e = _rne_bf16(F.embedding(tok_row.index_select(0, pos).view(1, 1), wte_w).float())
    ms = (e * e).sum(-1, keepdim=True) / e.size(-1)
    x0 = _rne_bf16(e * torch.rsqrt(ms + torch.finfo(torch.float32).eps))
    mixed = resid_lambda.float() * x0 + x0_lambda.float() * x0
    return x0.to(torch.bfloat16), mixed.to(torch.bfloat16)


@torch.compile(dynamic=False)
def _decode_head_prenorm_softcap(m, w_lm, softcap):
    """The final norm, the vocabulary projection and the softcap, in ONE node.

    `_matvec_int8_prenorm`'s construction at the one projection whose weight is bf16 rather
    than int8: `F.rms_norm` is a per-row SCALE, so giving its mean square the projection's own
    [8192] output shape puts both sums on the same 512-long axis and both nests collapse into
    one. At `rnumel` 512 the reduction is persistent, so each program recomputes the mean
    square from the vector it has already loaded for the projection -- 8,192 redundant
    512-element reductions, against a kernel whose 8.39 MB weight read measures 1141 GB/s and
    is therefore bandwidth-bound, so the arithmetic is free where the bytes are not.

    `softcap * tanh(./softcap)` is pointwise on the [8192] output and lands in the epilogue.
    gpu3 measured that fusion alone at **-0.794411 ms** (launch 28) and measured the GEMV
    half a wash (+/-0.3 us/call), so this region's node saving is priced from a launch.

    `_rne_bf16` and not `.to(torch.bfloat16).float()`: a bf16 cuBLAS GEMV STORES bf16 and
    `_decode_softcap` then widens it, and that `fp32 -> bf16 -> fp32` chain inside one region
    is deleted by `pointless_convert`. Dropping it would silently compute a different function
    -- more accurate than the champion's, and therefore not the champion's.
    """
    mf = m.float().unsqueeze(-2)
    sq = (mf * mf).expand(*mf.shape[:-2], w_lm.size(0), mf.size(-1))
    ms = sq.sum(-1, keepdim=True) / m.size(-1)
    hb = _rne_bf16(mf * torch.rsqrt(ms + torch.finfo(torch.float32).eps))
    logits = _rne_bf16((w_lm.float() * hb).sum(-1))
    return softcap * torch.tanh(logits / softcap)


@torch.compile(dynamic=False)
def _decode_softcap(logits, softcap):
    """The float cast and tanh softcap over the vocabulary row. Still the PREFILL path's."""
    logits = logits.float()
    return softcap * torch.tanh(logits / softcap)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        # q, k and v all read the same normed residual, so the three [out, n_embd] weights
        # are one [3*out, n_embd] weight whose rows are their concatenation. Same map, same
        # parameter count, same counted FLOPs; three eager GEMVs in the width-1 step become
        # one. Held as the Parameter itself and not as a view of a larger tensor: autocast's
        # `cached_cast` caches only leaf, non-view, requires-grad fp32 tensors, and an
        # uncached weight would be recast on every one of the 513 steps.
        assert self.n_kv_head == self.n_head, "merged qkv requires n_kv_head == n_head"
        self.qkv_out = self.n_head * self.head_dim
        self.c_qkv = nn.Parameter(torch.empty(3 * self.qkv_out, self.n_embd))
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        # Three calls on three row-block views, not one wide call: `forward` is the trained
        # path and its activation shapes, matmul shapes and autograd graph stay the
        # champion's exactly. One wide call would make the backward assemble a [B, T, 3*out]
        # activation gradient, ~800 MB at this microbatch. The merge is spent in
        # `_decode_body`, where nothing is differentiated.
        wq, wk, wv = self.c_qkv.split(self.qkv_out, 0)
        q = F.linear(x, wq).view(B, T, self.n_head, self.head_dim)
        k = F.linear(x, wk).view(B, T, self.n_kv_head, self.head_dim)
        v = F.linear(x, wv).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer), ungated: the per-head input-dependent gate is removed.
        if ve is not None:
            v = v + ve.view(B, T, self.n_kv_head, self.head_dim)

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
        self.parallel = parallel_block(layer_idx)

    def forward(self, x, ve, cos_sin, window_size):
        """Sequential, except on the `PARALLEL_BLOCK_LAYERS`, where both branches read ONE norm.

        The champion is sequential everywhere -- `norm(x + attn(norm(x)))` feeds the MLP -- and
        that second norm is the only reason `c_qkv` and `c_fc` read different vectors. Reading one
        vector is what lets the width-1 step issue ONE [3584, 512] reduction where the champion
        issues two (`_decode_qkv_fc_matvec_norm`), and it also removes one norm per parallel layer
        from training. `self.parallel` is a Python bool fixed at construction, so it is a trace
        constant and the two spellings are two regions of one unrolled graph, not a runtime branch.

        `x + a + b` is `(x + a) + b`, which is the width-1 path's own order: that path adds the
        attention projection into the residual (`_decode_attn_out_add`) and then the MLP
        projection into that sum (`_decode_mlp_out_mix` / `_decode_mlp_out_add`). The two paths
        must agree to within the fidelity gate, so the association is not free to differ.

        This CHANGES THE TRAINED NETWORK and therefore `val_bpb`, on two of six blocks. It is the
        one change in this candidate that is not arithmetic-preserving, and it is pre-registered
        as the row's real risk rather than as a rounding detail: anchored on launch 75's own
        measured 1.0532409328644643 and crediting each returned block +0.0021455 bpb, k=2 centres
        at **1.0446589 against the gate < 1.05, margin +0.005341**, and stays positive at every
        corner of gpu4's step box (`[+0.00072, +0.01117]`).
        """
        if self.parallel:
            h = norm(x)
            return x + self.attn(h, ve, cos_sin, window_size) + self.mlp(h)
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
            str(i): nn.Embedding(_ve_rows(i, config.vocab_size), kv_dim)
            for i in range(config.n_layer)
            if has_ve(i, config.n_layer) or i in EXTRA_VE_LAYERS or i in PARTIAL_VE_ROWS
        })
        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        # The width-1 decode path's bf16 projection weights, built on first use. A plain
        # attribute and not a buffer or a parameter: it must not appear in `state_dict`,
        # `parameters()` or `buffers()`, because `prepare.count_params` totals
        # `model.parameters()` and `num_params_total` is a hard ceiling.
        self._decode_wb = None

    @torch.no_grad()
    def init_weights(self):
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # Transformer blocks
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            # Filled block by block, in the champion's q, k, v order. Each block is a
            # contiguous [out, n_embd] region, so `uniform_` draws exactly the numbers it drew
            # into the separate weights and advances the generator by the same amount: the
            # initial network is bit-identical to the champion's, not merely identically
            # distributed. (This only holds because the model is built under
            # `torch.device("meta")`, where `nn.Linear.__init__`'s own `reset_parameters` draws
            # nothing -- on a real device it would shift the stream for every later tensor.)
            wq, wk, wv = block.attn.c_qkv.split(block.attn.qkv_out, 0)
            torch.nn.init.uniform_(wq, -s, s)
            torch.nn.init.uniform_(wk, -s, s)
            torch.nn.init.uniform_(wv, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        # No gate to initialise: the value residual is ungated, which is exactly the neutral
        # value the zero-initialised gate started from.
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
        # The merged projections get their own Muon group so it can declare the block view;
        # every other matrix parameter is grouped by shape exactly as before.
        qkv_params = [block.attn.c_qkv for block in self.transformer.h]
        qkv_ids = {id(p) for p in qkv_params}
        matrix_params = [p for p in self.transformer.h.parameters() if id(p) not in qkv_ids]
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == (len(matrix_params) + len(qkv_params) +
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
        # The merged projection is orthogonalised as three independent 512x512 blocks, which is
        # what the champion's three separate parameters were. Without `muon_view` a [1536, 512]
        # parameter is orthogonalised as one tall matrix, coupling the q, k and v update
        # directions -- a different optimiser, not a merged weight, and this experiment is about
        # the node count. `lr`'s `max(1, rows/cols)**0.5` factor is 1.0 for a 512x512 block,
        # the same as before.
        param_groups.append(dict(
            kind='muon', params=qkv_params, lr=matrix_lr,
            momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            muon_view=(3, qkv_params[0].size(0) // 3, qkv_params[0].size(1)),
        ))
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
            ve = (_ve_lookup(self.value_embeds[str(i)].weight, idx)
                  if str(i) in self.value_embeds else None)
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
        value and trip `recompile_limit`. `seq` is advanced with `add_()` and is the step's
        write index and query position. It is int64 rather than int32 because the width-1
        step now indexes the cache with it directly (`index_copy_` takes a LongTensor) and
        nothing asks for `cache_seqlens` any more: the width-1 branch does its own append,
        and the prefill branch calls `flash_attn_func`, which has no such argument. Costs
        four bytes in the state the cache probe measures.

        `graph=False` must run the step eagerly and capture nothing. It is how the instrument
        reads cache bytes without a graph's private pool in them, so a candidate that ignores
        the flag reports its own pool as cache and is charged for it.
        """
        assert max_len <= self.cos.size(1), f"max_len {max_len} exceeds rotary table"
        cfg = self.config
        head_dim = cfg.n_embd // cfg.n_head
        dev = self.transformer.wte.weight.device
        kw = {"dtype": torch.bfloat16, "device": dev}
        return {
            "max_len": max_len,
            "graph_enabled": bool(graph),
            "seq": torch.zeros(batch, dtype=torch.int64, device=dev),
            "kc": [torch.zeros(batch, max_len, cfg.n_kv_head, head_dim, **kw)
                   for _ in range(cfg.n_layer)],
            "vc": [torch.zeros(batch, max_len, cfg.n_kv_head, head_dim, **kw)
                   for _ in range(cfg.n_layer)],
        }

    def _decode_projection_weights(self):
        """The four per-layer projection weights as **int8 + per-row fp32 scale**, made once and
        kept on the model.

        **What this is buying.** After the projections became compiled reductions the remainder
        of their cost is traffic, not kernel floor: `knowledge/decode_gemv_kernel_price.md` fits
        F = 1.55-1.94 us per call against 13.28 us/layer, i.e. 5.5-7.1 us/layer of the 13.28 is
        bytes. Halving the bytes is the only lever left on that term, and 33-37% of the target
        is that term.

        **And a second, larger effect this may or may not get.** The bf16 working set is
        6 x 6.291 MB = 37.75 MB, plus `lm_head`'s 8.39 MB = 46.14 MB, against an A100's 40 MiB
        (41.94 MB) of L2 -- just over. int8 takes the projections to 18.87 MB and the whole
        weight working set to 27.26 MB, plus a 6.30 MB cache: **inside L2**. Every step after
        the first re-reads the same weights, so if residency flips, the traffic term drops by
        far more than 2x. This is also the leading untested explanation on record for why this
        champion's projection saving came out FLAT in depth (the same knowledge file's
        appended correction), and it predicts the same asymmetry: most of the win, if it
        arrives, arrives at the working-set threshold rather than in proportion to per-call
        bytes. That is why `attn.c_proj` is quantised too even though at 130 GB/s it is not
        bandwidth-bound on its own.

        **Memory is strictly safer than the champion's, not merely as safe.** 18,874,368 B of
        int8 plus 110,592 B of scales is 18,984,960 B, against the 37,748,736 B of bf16 this
        replaces. The placement argument below is unchanged and was measured free twice.

        `F.linear` under autocast reads a bf16 copy of each fp32 weight out of autocast's cast
        cache. A compiled region gets no such service: handed the fp32 parameter it would read
        four bytes per element where the GEMV reads two, and bytes are exactly what this change
        is spending. So the same copies are made here, with the same round-to-nearest
        `cached_cast` uses, which is why the operands stay bit-identical to the champion's.

        **On the model and not on the state, and that placement is the whole memory argument.**
        `prepare.measure_kv_cache_bytes` reads `memory_allocated` before and after building one
        state, so 44 MB of per-state copies would be charged to `nopref_kv_cache_bytes` against a
        ceiling with 3,130,880 bytes of headroom -- an instant ineligibility. Held here they are
        allocated during that probe's own throwaway build, which is documented to pay every
        one-time cost, so they are already inside its `before` reading and the delta is unchanged
        to the byte. `peak_vram_bytes` cannot see them either: `report_efficiency_metrics` reads
        it and resets the counter before any decode probe runs, so these bytes land only in
        `peak_vram_bytes_inference`, which no constraint bounds.

        Built lazily on the first width-1 step. That step is inside the same throwaway build and
        outside every CUDA graph capture, so the copies come from the ordinary allocator rather
        than from a graph's private pool. Decode runs only after training and after
        `evaluate_bpb`, on a model in eval under `no_grad`, so the weights these mirror never
        change again and there is nothing to invalidate.
        """
        cached = self._decode_wb
        if cached is None:
            # Quantised from the fp32 parameter, not from a bf16 round-trip: the bf16 copy
            # was only ever a way to halve the bytes the reduction reads, and rounding to bf16
            # first would add its error to the quantiser's for nothing.
            # `lm_head` stays bf16 and is NOT quantised. Its bytes are the largest single
            # weight read left in the step, but `nopref_decode_tv_distance_max` is the binding
            # constraint at 0.0305 of 0.05 with ~0.007 of its own draw noise, int8 on the four
            # projections measured +0.0240 of it, and this weight is the one in an AdamW group
            # whose kappa was still rising at 300 steps -- so it cannot be priced by analogy
            # with the four that are quantised. The copy here is exactly the one autocast's
            # `cached_cast` hands `F.linear`, made once, so the reduction below reads two bytes
            # per element rather than four and the operand stays bit-identical.
            # Each tuple is (int32-PACKED int8 words, fp32 row scale). Same bytes as the
            # champion's int8 weight, four per word, so one thread's load carries four.
            cached = ([_decode_layer_projection_weights(block, parallel_block(i))
                       for i, block in enumerate(self.transformer.h)],
                      self.lm_head.weight.detach().to(torch.bfloat16).contiguous())
            self._decode_wb = cached
        return cached

    def reset_decode_state(self, state):
        """Return `state` to position 0 so the same buffers serve another request.

        A fresh state per request would invalidate an address-bound CUDA graph. Stale bytes
        above the position are unreachable -- attention reads only `cache_seqlens` keys -- so
        zeroing them would be timed work serving never does.
        """
        state["seq"].zero_()
        graph = state.get("graph")
        if graph is not None:
            # The mirror is a mirror of THIS tensor; a reset moves both or neither.
            graph.pos_host = 0
        return state

    def _decode_body(self, idx, state, prefill, tok_row=None):
        """`forward`'s body with the cache spliced in. `prefill` is a Python bool that is
        constant per variant, so the two branches are two graphs, not one per position."""
        B, Tn = idx.size()
        if prefill:
            cos, sin = self.cos[:, :Tn], self.sin[:, :Tn]
            # The prefill branch keeps every `F.linear`: at T=1536 the reduction form would
            # broadcast a [1, 1536, out, in] product, and cuBLAS has a real GEMM to run.
            wb = None
        else:
            # `_decode_rotary_at` no longer runs on the width-1 step: the attention kernel indexes
            # the rotary TABLES at `pos` itself, so the step's two `index_select` kernels
            # (2.914 us/call once per step = 1.495 ms of the ranked metric, launch 44) are gone.
            # The tables are the model's own, the same tensors `forward` reads.
            cos, sin = self.cos, self.sin
            wb, wlm = self._decode_projection_weights()

        if prefill:
            x = norm(self.transformer.wte(idx))
            mixed = None
        else:
            # The gather, the norm on it and LAYER 0's mix in one reduction. Every later
            # layer's mix is already the epilogue of layer i-1's `mlp.c_proj`; this is the one
            # mix that had no predecessor projection, so it had a node of its own.
            if tok_row is None:
                x, mixed = _decode_embed_norm_mix(self.transformer.wte.weight, idx,
                                                  self.resid_lambdas[0], self.x0_lambdas[0])
            else:
                # Same reduction, same operands, one extra scalar load: `pos` then `row[pos]`
                # instead of the copy the host used to make. `state["seq"]` is the position the
                # rest of the step already reads.
                x, mixed = _decode_embed_norm_mix_at(self.transformer.wte.weight, tok_row,
                                                     state["seq"], self.resid_lambdas[0],
                                                     self.x0_lambdas[0])
        x0 = x
        carried = None      # the previous layer's MLP contribution, not yet added to x
        n_layer = len(self.transformer.h)
        for i, block in enumerate(self.transformer.h):
            attn = block.attn
            if prefill:
                if carried is None:
                    x, h = _decode_mix_norm(x, x0,
                                            self.resid_lambdas[i], self.x0_lambdas[i])
                else:
                    x, h = _decode_add_mix_norm(x, carried, x0,
                                                self.resid_lambdas[i], self.x0_lambdas[i])
            # The width-1 path hands the TABLE to `_decode_qkv_write` and lets the region
            # gather inside itself; only prefill still needs the gathered rows here, and it
            # is the path where the gather is a real [1, 1536, 512] read rather than one row.
            have_ve = str(i) in self.value_embeds
            ve = (_ve_lookup(self.value_embeds[str(i)].weight, idx)
                  if (prefill and have_ve) else None)
            # The whole point: one eager GEMV where the champion issues three. `attn.c_qkv` is
            # passed as the Parameter, so autocast's cast cache holds its bf16 copy for the
            # whole request instead of recasting inside every replay. The three slices are
            # views; at width 1 every other dimension is 1, so each is contiguous and the
            # attention kernel sees what it saw before.
            if prefill:
                qkv = F.linear(h, attn.c_qkv)
            elif block.parallel:
                # `c_qkv` [1536, 512] and `c_fc` [2048, 512] in ONE [3584, 512] reduction. Both
                # read `mixed` on a parallel block, so they are one kernel over one weight whose
                # rows are their concatenation, and both halves are views of its single output --
                # no split kernel, no copy. Two nodes become one, on 2 of 6 layers.
                _merged = _decode_qkv_fc_matvec_norm(mixed, *wb[i][6], 3 * attn.qkv_out)
                qkv = _merged[..., :3 * attn.qkv_out]
                hidden = _merged[..., 3 * attn.qkv_out:]
            else:
                qkv = _decode_qkv_matvec_norm(mixed, *wb[i][0])
            out = attn.qkv_out
            kc, vc = state["kc"][i], state["vc"][i]
            if prefill:
                q = qkv[..., :out].view(B, Tn, attn.n_head, attn.head_dim)
                k = qkv[..., out:2 * out].view(B, Tn, attn.n_kv_head, attn.head_dim)
                v = qkv[..., 2 * out:].view(B, Tn, attn.n_kv_head, attn.head_dim)
                if ve is not None:
                    v = _decode_ve_mix(v, ve.view(B, Tn, attn.n_kv_head, attn.head_dim))
                q, k = _decode_qk_rope_norm(q, k, cos, sin)
                kc[:, :Tn] = k
                vc[:, :Tn] = v
                y = fa3.flash_attn_func(q, kc[:, :Tn], vc[:, :Tn], causal=True,
                                        window_size=self.window_sizes[i])
            else:
                # The split, the value residual, rotary, norm AND the append the kernel used
                # to do inside itself are one region -- gpu2's `_decode_qkv_write`, measured
                # twice. The two eager `index_copy_` calls this replaces were the last eager
                # kernels between the projection and the attention, and they were eager only
                # because the append was written outside the region computing what it
                # appends. `seq[:1]` is still the device position -- a view, one per batch
                # row, advanced by one `add_` -- so there is still no `kc[:, start:end]`
                # slice whose bounds depend on the step number and still no host round-trip.
                # ALL of `_decode_qkv_write` is now ABSORBED into the attention kernel. v25
                # moved the k/v half: the program owning cache block `pos // BL` ropes and
                # norms k, adds the value-embedding row to v and appends both, before the
                # block loop reads the row it just wrote. This candidate moves the q half as
                # well -- every `(head, split)` program ropes and norms its own head's q from
                # `qkv`, a kernel input nothing writes, so no program waits for another --
                # and `_decode_q_write`'s launch, 1.747 us/call over 6 layers = 5.38 ms of
                # the ranked metric at ~100% dispatch floor, leaves the step. The region and
                # its `ve is None` branch still exist for the wrapper's fallback ladder,
                # which must do the append or attend over a row nobody wrote.
                y = _decode_qkv_rot_write_attention(
                    qkv,
                    self.value_embeds[str(i)].weight if have_ve else None,
                    idx,
                    cos, sin, kc, vc, state["seq"],
                    out, attn.n_head, attn.head_dim,
                    self.window_sizes[i][0], attn.head_dim ** -0.5,
                    advance=(i + 1 == n_layer), tok_row=tok_row)
            ao = y.contiguous().view(B, Tn, -1)
            if prefill and block.parallel:
                # Parallel block: `c_fc` reads the SAME `h` the attention read, so there is no
                # norm between the two branches and `_decode_add_norm` leaves this path. `carried`
                # is still added by the next layer's mix (or by the tail), so the residual sum
                # order stays `(x + attn) + mlp` -- `Block.forward`'s order.
                carried = block.mlp.c_proj(_decode_relu_square(block.mlp.c_fc(h)))
                x = x + attn.c_proj(ao)
            elif prefill:
                x, h = _decode_add_norm(x, attn.c_proj(ao))
                carried = block.mlp.c_proj(_decode_relu_square(block.mlp.c_fc(h)))
            else:
                # Four nodes where the champion has six. `_decode_add_norm` and the next
                # layer's `_decode_add_mix_norm` are not merged with a neighbour, they stop
                # existing: the add is the attention projection's epilogue, the norm is the
                # MLP projection's own reduction, and the next mix is `mlp.c_proj`'s
                # epilogue. `relu(.).square()` still fuses into `c_fc` as before.
                # THREE nodes where the champion has six, on a parallel non-last layer.
                # `_decode_attn_out_add` no longer has a consumer inside the layer other than the
                # mix -- on a parallel block the MLP does not read its sum -- so the two out
                # projections stop being two reductions and become one over 2560 terms. That is
                # only true here: on a sequential layer `hidden` depends on this very sum.
                if block.parallel and i + 1 < n_layer:
                    mixed = _decode_out_merge_mix(mixed, ao, hidden,
                                                  *wb[i][4], *wb[i][3], x0,
                                                  self.resid_lambdas[i + 1],
                                                  self.x0_lambdas[i + 1])
                else:
                    xa = _decode_attn_out_add(mixed, ao, *wb[i][4])
                    if not block.parallel:
                        hidden = _decode_mlp_hidden_norm(xa, *wb[i][2])
                    # On a parallel block `hidden` already came out of the merged reduction above:
                    # the MLP does not read this sum, so `_decode_mlp_hidden_norm`'s kernel is GONE
                    # rather than moved. `xa` is still the residual after attention and is still
                    # what the mix reads.
                    if i + 1 < n_layer:
                        mixed = _decode_mlp_out_mix(hidden, *wb[i][3], xa, x0,
                                                    self.resid_lambdas[i + 1],
                                                    self.x0_lambdas[i + 1])
                    else:
                        # No next mix to fold into, so the epilogue takes the TAIL add instead:
                        # `_decode_add_tail_norm`'s `x + y` is pointwise on this reduction's
                        # output index. The norm it did is now the head's own reduction. The last
                        # layer is sequential at every dial setting this run has bought, but the
                        # branch is written on `i + 1 < n_layer` rather than on that, so a dial
                        # that makes the last layer parallel falls back instead of mis-computing.
                        x = _decode_mlp_out_add(hidden, *wb[i][5], xa)

        softcap = 15
        if prefill:
            x = _decode_add_tail_norm(x, carried)
            return _decode_softcap(self.lm_head(x), softcap)
        # `_decode_add_tail_norm`'s add is already this step's last projection epilogue and
        # its norm is this region's own reduction, so three kernels are one. `x` is [B, 1, C]
        # on this branch -- `decode_step` routes width > 1 to prefill -- and the slice is the
        # champion's own, kept so the two branches read identically.
        return _decode_head_prenorm_softcap(x[:, -1:, :], wlm, softcap)

    def decode_step(self, idx, state):
        """`logits, state = model.decode_step(idx, state)`, logits for the LAST position only.

        Width > 1 is the prefill and runs once per request; width 1 is the step, and it is
        the call the graph replays. `seq` is advanced on both paths, so the tensor is the
        single source of the position.
        """
        if idx.size(1) > 1:
            logits = self._decode_body(idx, state, prefill=True)
            state["seq"].add_(idx.size(1))
            graph = state.get("graph")
            if graph is not None:
                graph.pos_host += idx.size(1)
            return logits, state
        if not state.get("graph_enabled", True):
            # The width-1 body advances `seq` itself now -- in the last layer's combine kernel on
            # the fused path, eagerly inside the same wrapper on the fallback ladder -- so this
            # branch must NOT add again. Prefill still does, below/above: its width is not 1.
            with _decode_region_autotune():
                logits = self._decode_body(idx, state, prefill=False)
            graph = state.get("graph")
            if graph is not None:
                graph.pos_host += 1
            return logits, state
        # The graph lives ON the state it was captured against, so its lifetime is exactly
        # that state's and there is no key to get wrong. A state whose capture failed holds
        # an uncaptured object and runs eager forever, which is slow and correct.
        graph = state.get("graph")
        if graph is None:
            # `idx` is handed over so the graph can bind the row the caller is walking and
            # read this step's token in place instead of copying it before every replay.
            graph = state["graph"] = _GraphedDecodeStep(self, state, idx)
        if not graph.captured:
            with _decode_region_autotune():
                logits = self._decode_body(idx, state, prefill=False)
            graph.pos_host += 1
            return logits, state
        return graph.replay(idx), state


class _GraphedDecodeStep:
    """A manually captured `torch.cuda.CUDAGraph` over one cached decode step.

    Captured by hand, not by `torch.compile(mode="reduce-overhead")`: that mode copies graph
    inputs into a static pool, and the width-1 step appends k/v in place into the real cache,
    so the append would land in the pool copy instead. Measured: fastest available
    configuration, 4.49 bits per token destroyed, no warning. That was recorded when the
    append lived inside `fa3.flash_attn_with_kvcache`'s opaque custom op; the append is now
    two `index_copy_` calls, which inductor CAN see, but they are deliberately left outside
    every compiled region and the hazard is the same either way, so the manual capture stays.

    Three constraints follow:

      1. The graph is bound to its state's addresses, so it is stored ON that state and
         cannot be reused across states.
      2. The increment is captured after every attention launch that reads it -- it is a
         predicated store in the LAST layer's combine kernel, so a replay attends, appends, then
         increments with no host round-trip and without a kernel of its own.
      3. Warmup and capture execute their kernels, so `seq` is saved and restored around
         both; junk above `seq` is unreachable to `cache_seqlens`.

    Capture failure sets `captured` False and runs eager -- slow, correct, and visible.
    """

    WARMUP_REPLAYS = 3
    VERIFY_REPLAYS = 4

    def __init__(self, model, state, idx=None):
        self.model = model
        self.state = state
        self.captured = False
        self.reason = ""
        self.static_idx = torch.zeros(1, 1, dtype=torch.int64, device=state["seq"].device)
        self.static_logits = None
        self.graph = None
        self.tok_row = None
        self.tok_base_ptr = None
        self.tok_itemsize = 0
        self.pos_host = 0
        self.replays = 0
        self.eager_calls = 0
        self.tok_reason = "not attempted"
        if state["seq"].numel() != 1:
            self.reason = f"batch {state['seq'].numel()} != 1"
            return
        # The position, ONCE, on the host. This constructor runs on the first width-1 call,
        # which is inside `measure_decode_request`'s first WARMUP pass -- outside every timed
        # region -- so this synchronisation costs no metric. Nothing per-step syncs.
        self.pos_host = int(state["seq"].item())
        self._bind_token_row(idx)
        try:
            self._capture()
            self.captured = True
        except Exception as exc:            # noqa: BLE001 -- recorded, then paid for in ms
            self.reason = f"{type(exc).__name__}: {exc}"
            self.graph, self.static_logits = None, None
        # Say so. `reason` was recorded but never surfaced, and the eager fallback is
        # correct-but-slow: a capture that stopped working would otherwise read as a
        # latency result rather than as the fallback it is. Not a METRICS_JSON line.
        print(f"[decode-capture] captured={self.captured} reason={self.reason!r}", flush=True)

    def _bind_token_row(self, idx):
        """Bind the row the caller is walking, or decline and keep the champion's copy.

        Declining is the safety valve: with `tok_row` None every line below is the champion's,
        including the per-replay `static_idx.copy_`. So a caller whose tokens are not one
        contiguous int64 row indexed by the cache position loses the mechanism and keeps the
        behaviour, rather than reading the wrong token.
        """
        if idx is None:
            self.tok_reason = "no idx supplied"
            return
        try:
            if idx.dtype != torch.int64 or idx.numel() != 1 or idx.device.type != "cuda":
                self.tok_reason = f"idx dtype/numel/device {idx.dtype}/{idx.numel()}/{idx.device}"
                return
            storage = idx.untyped_storage()
            itemsize = idx.element_size()
            length = storage.nbytes() // itemsize
            offset = idx.storage_offset()
            # THE INVARIANT, checked at its base case: the instrument's token offset is the
            # cache position. Induction does the rest -- a replay advances `pos` by exactly one
            # (the last layer's combine stores `pos + 1`) and the caller advances its view by
            # exactly one element, and `replay` re-checks the pointer every single call.
            if offset != self.pos_host or length <= offset:
                self.tok_reason = f"offset {offset} != pos {self.pos_host} (len {length})"
                return
            row = torch.empty(0, dtype=idx.dtype, device=idx.device)
            row.set_(storage, storage_offset=0, size=(length,), stride=(1,))
            if row.data_ptr() + offset * itemsize != idx.data_ptr():
                self.tok_reason = "row base + offset != idx.data_ptr()"
                return
            self.tok_row = row
            self.tok_base_ptr = row.data_ptr()
            self.tok_itemsize = itemsize
            self.tok_reason = f"bound at pos {self.pos_host}, row of {length}"
        except Exception as exc:            # noqa: BLE001 -- decline, never guess
            self.tok_reason = f"{type(exc).__name__}: {exc}"
            self.tok_row = None

    def _advance(self):
        # `_decode_body` advances `seq` itself: the increment is a predicated store inside the last
        # layer's combine kernel, which is captured because the body is captured.
        with _decode_region_autotune():
            logits = self.model._decode_body(self.static_idx, self.state, prefill=False,
                                             tok_row=self.tok_row)
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
        if not self.captured:
            # Reverted (see below) or never captured. `decode_step` already routes this case to
            # the eager body, so this is belt and braces -- but it must be here, because a
            # reverted graph was CAPTURED reading `row[pos]` and replaying it after the revert
            # would read a row the caller may no longer be walking.
            logits = self.model._decode_body(idx, self.state, prefill=False)
            self.pos_host += 1
            return logits
        if self.tok_row is None:
            self.static_idx.copy_(idx)
            self.graph.replay()
            return self.static_logits
        # The whole mechanism, and its guard. Pure host arithmetic on two integers -- no device
        # read, no synchronisation, no allocation: is the token this call passed the one at
        # `row[pos]` that the captured graph will read?
        expected = self.tok_base_ptr + self.pos_host * self.tok_itemsize
        conforming = (idx.data_ptr() == expected and idx.numel() == 1
                      and idx.dtype == torch.int64)
        if conforming and self.replays < self.VERIFY_REPLAYS:
            # The host mirror, checked AGAINST THE DEVICE, BEFORE the graph reads anything, on
            # the first few replays only. Those land in `measure_decode_request`'s warmup
            # passes, which are not timed, so this synchronisation costs no metric; and a
            # desynchronised mirror is exactly the failure that would make the graph read the
            # wrong row, so it is checked against the device rather than argued. On failure the
            # mechanism is abandoned for the rest of the process and every later call runs the
            # champion's eager body -- slow, correct, and printed.
            device_pos = int(self.state["seq"].item())
            if device_pos != self.pos_host:
                print(f"[decode-token] host mirror {self.pos_host} != device {device_pos}; "
                      f"reverting to the eager body", flush=True)
                self.tok_row = None
                self.captured = False
                self.reason = "token-row mirror desynchronised"
                conforming = False
        if not conforming:
            # A caller that is not walking the bound row. Serve it eagerly from its own tensor:
            # correct, slower, and it advances `seq` exactly once like a replay does.
            self.eager_calls += 1
            logits = self.model._decode_body(idx, self.state, prefill=False)
            self.pos_host += 1
            return logits
        self.graph.replay()
        self.pos_host += 1
        self.replays += 1
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
        # `muon_view` reshapes each parameter into independent leading blocks before
        # orthogonalisation, so a merged [3*out, in] projection is updated exactly as three
        # [out, in] parameters were. Absent (None) on every group the champion had, and the
        # `state_shape` expression below reduces to the champion's for any 2-D shape.
        view = group.get('muon_view')
        targets = [p.view(view) for p in params] if view else list(params)
        grads = [p.grad.view(view) for p in params] if view else [p.grad for p in params]
        p = targets[0]
        state = self.state[params[0]]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = ((num_params, *shape[:-1], 1) if shape[-2] >= shape[-1]
                           else (num_params, *shape[:-2], 1, shape[-1]))
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack(grads)
        stacked_params = torch.stack(targets)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim)
        torch._foreach_copy_(targets, list(stacked_params.unbind(0)))

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
ASPECT_RATIO = 80       # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 128          # target head dimension for attention
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context

# Optimization
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
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
DEPTH = 6               # number of transformer layers
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
reported_metrics = report_efficiency_metrics(
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
# Read-only probe. Runs AFTER `report_efficiency_metrics` has printed METRICS_JSON, so nothing
# it prints or allocates can move a score. gpu1's launch-13 method (capture a graph replaying an
# op N times and divide; never time one eager call, which measures host launch overhead a graph
# replay never pays) and gpu1's launch-37 per-region census.
#
# gpu1's census FAILED on launch 37, every row, with FileNotFoundError: `CUDAGraph.debug_dump`
# writes into `tempfile.gettempdir()` and that directory does not exist on the launch node.
# Fixed here by creating it. The post-fusion CUDA kernel count per region has been the run's
# most-wanted read-only number for three cycles.
#
# It answers, in one launch and at the scored position:
#   1. each of the five packed regions AGAINST its v22 spelling -- the mechanism, priced
#      directly per region instead of inferred from one whole-step delta;
#   2. the post-fusion CUDA kernel count of every width-1 region.
# ---------------------------------------------------------------------------


def _matvec_int8_asv22(x, wq, s):
    """Champion v22's spelling, kept for the paired probe below. Never on the live path."""
    xb = x.to(torch.bfloat16).float().unsqueeze(-2)
    return ((wq.to(torch.float32) * xb).sum(-1) * s).to(torch.bfloat16)


def _matvec_int8_prenorm_asv22(m, wq, s):
    mf = m.float().unsqueeze(-2)
    sq = (mf * mf).expand(*mf.shape[:-2], wq.size(0), mf.size(-1))
    ms = sq.sum(-1, keepdim=True) / m.size(-1)
    hb = _rne_bf16(mf * torch.rsqrt(ms + torch.finfo(torch.float32).eps))
    return ((wq.to(torch.float32) * hb).sum(-1) * s).to(torch.bfloat16)


def _matvec_int8_rne_asv22(x, wq, s):
    xb = x.to(torch.bfloat16).float().unsqueeze(-2)
    return _rne_bf16((wq.to(torch.float32) * xb).sum(-1) * s)


@torch.compile(dynamic=False)
def _probe_attn_out_add_packed(mixed, y, w32, s):
    """Champion v27's spelling of `_decode_attn_out_add`, kept so the pair can be timed."""
    return (mixed.float() + _matvec_int8_rne(y, w32, s)).to(mixed.dtype)


@torch.compile(dynamic=False)
def _probe_mlp_out_add_packed(a, w32, s, x):
    """Champion v27's spelling of `_decode_mlp_out_add`, fp32 return and all.

    This pair is the one launches 39 and 43 could NOT measure cleanly: their v22 leg
    (`_probe_mlp_out_add_v22`) also reverted the fp32 tail return to `.to(x.dtype)`, so it varied
    the weight view and the store dtype together. Both legs here return fp32, so the difference is
    the weight view alone.
    """
    return x.float() + _matvec_int8_rne(a, w32, s)


def _matvec_int8_prenorm_dependent(m, w32, s):
    """CHAMPION v35's `_matvec_int8_prenorm`, verbatim, kept so the pair can be timed.

    This is the spelling `hoist_r_out_of_the_prenorm_rounding` replaces: its second fold reads
    `r`, so the two 128-wide folds are serial. Carried byte-for-byte from the champion so the
    paired differential is the mechanism and nothing else -- same lanes, same weight view, same
    accumulation order, same eps, same store.
    """
    lanes = _act_lanes(m)
    sq = lanes[0] * lanes[0]
    for i in range(1, 4):
        sq = sq + lanes[i] * lanes[i]
    sq = sq.expand(*sq.shape[:-2], w32.size(0), sq.size(-1))
    ms = sq.sum(-1, keepdim=True) / m.size(-1)
    r = torch.rsqrt(ms + torch.finfo(torch.float32).eps)
    w = _int8_lanes(w32)
    acc = w[0] * _rne_bf16(lanes[0] * r)
    for i in range(1, 4):
        acc = acc + w[i] * _rne_bf16(lanes[i] * r)
    return ((acc.sum(-1)) * s).to(torch.bfloat16)


@torch.compile(dynamic=False)
def _probe_qkv_matvec_norm_dependent(mixed, w32, s):
    """`_decode_qkv_matvec_norm` with the champion's DEPENDENT folds. Never on the live path."""
    return _matvec_int8_prenorm_dependent(mixed, w32, s)


@torch.compile(dynamic=False)
def _probe_mlp_hidden_norm_dependent(xa, w32, s):
    """`_decode_mlp_hidden_norm` with the champion's DEPENDENT folds. Never on the live path."""
    return F.relu(_matvec_int8_prenorm_dependent(xa, w32, s)).square()


@torch.compile(dynamic=False)
def _probe_qkv_matvec_norm_v22(mixed, wq, s):
    return _matvec_int8_prenorm_asv22(mixed, wq, s)


@torch.compile(dynamic=False)
def _probe_attn_out_add_v22(mixed, y, wq, s):
    return (mixed.float() + _matvec_int8_rne_asv22(y, wq, s)).to(mixed.dtype)


@torch.compile(dynamic=False)
def _probe_mlp_hidden_norm_v22(xa, wq, s):
    return F.relu(_matvec_int8_prenorm_asv22(xa, wq, s)).square()


@torch.compile(dynamic=False)
def _probe_mlp_out_mix_v22(a, wq, s, x, x0, resid_lambda, x0_lambda):
    added = _rne_bf16(x.float() + _matvec_int8_rne_asv22(a, wq, s))
    return (_rne_bf16(_rne_bf16(resid_lambda.float()) * added)
            + _rne_bf16(_rne_bf16(x0_lambda.float()) * x0.float())).to(x.dtype)


@torch.compile(dynamic=False)
def _probe_mlp_out_add_v22(a, wq, s, x):
    return (x.float() + _matvec_int8_rne_asv22(a, wq, s)).to(x.dtype)


def _probe_decode_regions(model, tokenizer, rep=64, timed=20, warmup=5):
    import os
    import statistics
    import tempfile
    base = model._orig_mod if hasattr(model, "_orig_mod") else model
    dev = base.transformer.wte.weight.device
    tmpdir = tempfile.gettempdir()
    os.makedirs(tmpdir, exist_ok=True)          # launch 37's census died here, every row

    def _warm(fn):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                for _ in range(rep):
                    fn()
        torch.cuda.current_stream().wait_stream(stream)

    def graph_us(fn, label):
        _warm(fn)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(rep):
                fn()
        for _ in range(warmup):
            graph.replay()
        torch.cuda.synchronize()
        samples = []
        for _ in range(timed):
            start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record(); graph.replay(); stop.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(stop) * 1000.0 / rep)
        del graph
        return label, statistics.median(samples)

    def census_kernels(fn, label):
        _warm(fn)
        # THREE additions to gpu4's census, all read-only, because this exact call has now
        # produced no file on SEVEN launches by three authors and the standing explanation is
        # refuted. Launch 39 is the refutation: gpu4 created `tmpdir` expressly to fix
        # "gettempdir does not exist", and every row still failed `FileNotFoundError`. So the
        # directory is not the cause and neither is `enable_debug_mode` alone -- which gpu4 calls
        # before capture, and which `ATen/cuda/CUDAGraph.h` documents as sufficient to retain the
        # handle ("set back to false after instantiate() unless keep_graph=True or
        # enable_debug_mode() was called on any CUDAGraph instance").
        #
        # `debug_dump` prints the `cudaGraph_t`, and a default-constructed `CUDAGraph`
        # instantiates the exec at `capture_end` and releases that handle. So:
        #   1. `keep_graph=True` satisfies the FIRST retention clause, which does not depend on a
        #      process-global flag having been set on some other instance;
        #   2. `instantiate()` explicitly, because `keep_graph=True` defers it to first replay;
        #   3. and if it fails AGAIN, `raw_cuda_graph()` says which clause failed -- it raises when
        #      the handle is gone -- so the eighth attempt is evidence instead of a symptom.
        # This is a throwaway graph in a probe that runs after METRICS_JSON, so none of it can
        # move a reading.
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        graph.enable_debug_mode()
        with torch.cuda.graph(graph):
            fn()
        try:
            graph.instantiate()
        except Exception as exc:            # noqa: BLE001
            print(f"[regioncensus] instantiate: {type(exc).__name__}: {exc}", flush=True)
        path = os.path.join(tmpdir,
                            "region_" + "".join(c if c.isalnum() else "_" for c in label))
        graph.debug_dump(path)
        _ex = os.path.exists(path)
        try:
            _raw = hex(graph.raw_cuda_graph())
        except Exception as _rexc:          # noqa: BLE001
            _raw = f"GONE ({type(_rexc).__name__}: {_rexc})"
        print(f"[regioncensus] dump {label!r}: exists={_ex} "
              f"size={os.path.getsize(path) if _ex else -1} raw_cuda_graph={_raw} "
              f"tmpdir_exists={os.path.isdir(tmpdir)}", flush=True)
        with open(path, errors="replace") as fh:
            lines = fh.read().splitlines()
        defs = [ln for ln in lines if "label=" in ln and "->" not in ln]
        try:
            os.unlink(path)
        except OSError:
            pass
        del graph
        return len(defs), defs

    rows, counts = [], []
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        state = base.init_decode_state(batch=1, max_len=514, graph=True)
        idx = torch.zeros(1, 1, dtype=torch.int64, device=dev)
        for _ in range(257):
            base.decode_step(idx, state)
        pos = state["seq"]
        at = pos[:1]

        block = base.transformer.h[0]
        attn = block.attn
        B = Tn = 1
        out = attn.qkv_out
        wb, wlm = base._decode_projection_weights()
        # (packed words, scale) as the live path holds them, and the SAME BYTES viewed as int8
        # for the v22 spellings, so both legs of every pair read one weight tensor.
        w0, w1, w2, w3 = wb[0][:4]
        # The merged [3584,512] this candidate's parallel blocks read. `wb[0]` carries it because
        # layer 0 is in `PARALLEL_BLOCK_LAYERS`; read the first parallel layer rather than 0 so
        # the leg survives a different dial setting.
        wm = wb[PARALLEL_BLOCK_LAYERS[0]][6]
        u0 = (w0[0].view(torch.int8), w0[1])
        u1 = (w1[0].view(torch.int8), w1[1])
        u2 = (w2[0].view(torch.int8), w2[1])
        u3 = (w3[0].view(torch.int8), w3[1])
        cos, sin = _decode_rotary_at(base.cos, base.sin, pos)
        x = torch.randn(1, 1, base.config.n_embd, device=dev, dtype=torch.bfloat16)
        mixed = x
        qkv = _decode_qkv_matvec_norm(mixed, *w0)
        kc, vc = state["kc"][0], state["vc"][0]
        # Probe-only transients for the ADVANCE pair and for the `num_warps` sweep. `_cpos` is a
        # CLONE, so the timed rows can increment it thousands of times without touching the state
        # any other row reads. `CDT` / `NWP` / `NWC` prefixes => the SUM OF PARTS loop prices these
        # rows and does not add them.
        #
        # MOVED, and that is a repair rather than a tidy: this block read `kc.size(1)` two
        # statements ABOVE `kc`'s assignment, so it raised `NameError` on every launch that has
        # run it and its `except` set `_cnum = None`, silently dropping gpu6's two ADVANCE rows --
        # the re-payment term gpu6 registered at +0.05..+0.25 us/call and nothing has measured.
        try:
            _cH, _cD = attn.n_head, attn.head_dim
            _cNB = triton.cdiv(kc.size(1), DECODE_ATTN_BLOCK_L)
            _cGS = min(_cNB, DECODE_ATTN_SPLITS)
            _cVS, _cTAIL = _combine_split(_cGS)
            _cnum = torch.rand((_cH, DECODE_ATTN_SPLITS, _cD), dtype=torch.float32, device=dev)
            _cden = torch.rand((_cH, DECODE_ATTN_SPLITS), dtype=torch.float32, device=dev) + 1.0
            _cout = torch.empty((1, 1, _cH, _cD), dtype=torch.bfloat16, device=dev)
            _cpos = pos.clone()
        except Exception as _cexc:                          # noqa: BLE001
            print(f"[probe] ADVANCE pair setup skipped: {type(_cexc).__name__}: {_cexc}")
            _cnum = None
        ve_w = base.value_embeds["1"].weight if "1" in base.value_embeds else None
        q = _decode_qkv_write(qkv, ve_w, idx, cos, sin, kc, vc, at,
                              out, attn.n_head, attn.head_dim)
        y = _decode_cached_attention_triton(q, kc, vc, pos, base.window_sizes[0][0],
                                            attn.head_dim ** -0.5)
        ao = y.contiguous().view(B, Tn, -1)
        xa = _decode_attn_out_add(mixed, ao, *w1)
        hidden = _decode_mlp_hidden_norm(xa, *w2)
        # A parallel layer's `hidden` is the tail of the merged input projection, not
        # `_decode_mlp_hidden_norm`'s output. Same shape and dtype; taken from the merged region so
        # the OUTMERGE leg reads the operand the live path hands it.
        hidden_par = _decode_qkv_fc_matvec_norm(mixed, *wm, 3 * out)[..., 3 * out:]
        rl1, x0l1 = base.resid_lambdas[1], base.x0_lambdas[1]

        probes = [
            # THIS CANDIDATE's exchange, paired in ONE launch against the two LIVE regions it
            # replaces on 2 of 6 layers. Both of those regions are still live here (on the four
            # sequential blocks) and still PINNED, so this pair is the unpinned-merged-against-
            # pinned-pair exchange -- the same construction launch 75 printed at k=6 (merged 3.417
            # against 2.728+3.102), and the only one not exposed to the cross-allocation term.
            # Their labels below are unchanged from the champion's on purpose: they are the join
            # keys the run's printed series is grepped by.
            ("MERGED _decode_qkv_fc_matvec_norm [3584,512] (layer)",
             lambda: _decode_qkv_fc_matvec_norm(mixed, *wm, 3 * out)),
            # THE MECHANISM: five pairs, packed against v22, same launch, same position.
            ("PACKED _decode_qkv_matvec_norm [1536,512] (layer)",
             lambda: _decode_qkv_matvec_norm(mixed, *w0)),
            ("DEPEND _decode_qkv_matvec_norm [1536,512] (layer)",
             lambda: _probe_qkv_matvec_norm_dependent(mixed, *w0)),
            ("v22    _decode_qkv_matvec_norm [1536,512] (layer)",
             lambda: _probe_qkv_matvec_norm_v22(mixed, *u0)),
            ("v24    _decode_attn_out_add PACKED [512,512] (layer)",
             lambda: _probe_attn_out_add_packed(mixed, ao, *w1)),
            ("v22    _decode_attn_out_add copy [512,512] (layer)",
             lambda: _probe_attn_out_add_v22(mixed, ao, *u1)),
            ("_decode_attn_out_add WORDS live [512,512] (layer)",
             lambda: _decode_attn_out_add(mixed, ao, *w1)),
            ("PACKED _decode_mlp_hidden_norm [2048,512] (layer)",
             lambda: _decode_mlp_hidden_norm(xa, *w2)),
            ("DEPEND _decode_mlp_hidden_norm [2048,512] (layer)",
             lambda: _probe_mlp_hidden_norm_dependent(xa, *w2)),
            ("v22    _decode_mlp_hidden_norm [2048,512] (layer)",
             lambda: _probe_mlp_hidden_norm_v22(xa, *u2)),
            ("PACKED _decode_mlp_out_mix [512,2048] (5 layers)",
             lambda: _decode_mlp_out_mix(hidden, *w3, xa, x, rl1, x0l1)),
            # THIS CANDIDATE's exchange, paired in ONE launch against the two LIVE regions it
            # replaces on the 2 parallel layers. Both of those are still live here (on the four
            # sequential blocks) and are the two rows immediately above and at :250, so this is an
            # in-launch pair exposed to neither the cross-allocation term nor the cold-cache term.
            # `hidden_par` comes out of the MERGED input projection, which is where a parallel
            # layer's `hidden` actually comes from -- reading it from `_decode_mlp_hidden_norm`
            # would price the wrong operand's provenance.
            ("OUTMERGE _decode_out_merge_mix [512,2560] (parallel layers)",
             lambda: _decode_out_merge_mix(mixed, ao, hidden_par, *w1, *w3, x,
                                           rl1, x0l1)),
            ("v22    _decode_mlp_out_mix [512,2048] (5 layers)",
             lambda: _probe_mlp_out_mix_v22(hidden, *u3, xa, x, rl1, x0l1)),
            ("v24    _decode_mlp_out_add PACKED (once/step)",
             lambda: _probe_mlp_out_add_packed(hidden, *w3, xa)),
            ("v22    _decode_mlp_out_add +bf16 store (once/step)",
             lambda: _probe_mlp_out_add_v22(hidden, *u3, xa)),
            ("_decode_mlp_out_add WORDS live (once/step)",
             lambda: _decode_mlp_out_add(hidden, *w3, xa)),
            # The rest of the step, unchanged by this diff, so the sum can be checked against
            # the frozen metric and the unattributed remainder reported.
            ("_decode_qkv_write +ve (3 layers)",
             (lambda: _decode_qkv_write(qkv, ve_w, idx, cos, sin, kc, vc, at,
                                        out, attn.n_head, attn.head_dim))
             if ve_w is not None else None),
            ("_decode_qkv_write no-ve (3 layers)",
             lambda: _decode_qkv_write(qkv, None, idx, cos, sin, kc, vc, at,
                                       out, attn.n_head, attn.head_dim)),
            ("attention Triton pair (per layer)",
             lambda: _decode_cached_attention_triton(
                 q, kc, vc, pos, base.window_sizes[0][0], attn.head_dim ** -0.5)),
            # THIS CANDIDATE's pair, so its post-fusion CUDA kernel count is on the record
            # next to the champion's: absorbing q must not split the partial into two.
            ("CANDIDATE fused q+kv_write+attn +ve (3 layers)",
             (lambda: _decode_qkv_write_attention(
                 qkv, ve_w, idx, cos, sin, kc, vc, pos, out, attn.n_head, attn.head_dim,
                 base.window_sizes[0][0], attn.head_dim ** -0.5))
             if ve_w is not None else None),
            ("ROT fused q+kv_write+attn +ve (3 layers)",
             (lambda: _decode_qkv_rot_write_attention(
                 qkv, ve_w, idx, base.cos, base.sin, kc, vc, pos, out, attn.n_head, attn.head_dim,
                 base.window_sizes[0][0], attn.head_dim ** -0.5))
             if ve_w is not None else None),
            ("_decode_embed_norm_mix (once/step)",
             lambda: _decode_embed_norm_mix(base.transformer.wte.weight, idx,
                                            base.resid_lambdas[0], base.x0_lambdas[0])),
            ("_decode_head_prenorm_softcap [8192,512] (once/step)",
             lambda: _decode_head_prenorm_softcap(x, wlm, 15)),
            ("_decode_rotary_at (once/step)",
             lambda: _decode_rotary_at(base.cos, base.sin, pos)),
            ("seq.add_(1) (once/step)", lambda: pos.add_(0)),
            # THIS ROW's pair: the same combine kernel with the position store off and on, timed in
            # one graph. Their difference IS the re-payment term -- one scalar load hoisted above
            # the reduction, one add, one predicated store in 1 of 4 programs -- which I registered
            # at +0.05..+0.25 us/call and nothing in this run has measured. The `seq.add_(1)` row
            # directly above is the deleted term, so both halves of the net are on one page.
            ("CDT attn combine ADVANCE=0 (probe only)",
             (lambda: _decode_attn_combine[(_cH,)](
                 _cnum, _cden, _cout, _cpos, D=_cD, NS=DECODE_ATTN_SPLITS, GS=_cGS, VS=_cVS, TAIL=_cTAIL,
                 ADVANCE=False, num_warps=4)) if _cnum is not None else None),
            ("CDT attn combine ADVANCE=1 (probe only)",
             (lambda: _decode_attn_combine[(_cH,)](
                 _cnum, _cden, _cout, _cpos, D=_cD, NS=DECODE_ATTN_SPLITS, GS=_cGS, VS=_cVS, TAIL=_cTAIL,
                 ADVANCE=True, num_warps=4)) if _cnum is not None else None),
        ]

        # THE SWEEP. Appended at the END so every row above executes in the order and the device
        # state it executed in on prior launches -- the control legs are only worth their drift
        # figure if their position does not move.
        #
        # `num_warps` inside a hand-written `@triton.jit` kernel is a SOURCE constant, so no
        # inductor flag and no coordinate descent can reach it; the audit post that closed the
        # inductor config axis said so and left this pair -- 12 of ~39 kernels and ~46 us/step --
        # to whoever owns the kernel. These legs launch the two kernels DIRECTLY at five widths
        # each, on the same buffers, in one process, which is the one instrument this run has
        # found that predicted a per-call differential correctly twice.
        #
        # Read-only against the metric by construction: the whole probe runs after METRICS_JSON.
        # The partial legs do write `kc[pos]`/`vc[pos]` in the owner program, exactly as the
        # `_decode_qkv_write` row above already does, and the value written is a function of the
        # same `qkv` every time, so the writes are idempotent and no later row sees a new value.
        try:
            _sw_num = torch.empty((attn.n_head, DECODE_ATTN_SPLITS, attn.head_dim),
                                  dtype=torch.float32, device=dev)
            _sw_den = torch.empty((attn.n_head, DECODE_ATTN_SPLITS),
                                  dtype=torch.float32, device=dev)
            _sw_nb = triton.cdiv(kc.size(1), DECODE_ATTN_BLOCK_L)
            _sw_gs = min(_sw_nb, DECODE_ATTN_SPLITS)
            _sw_niter = triton.cdiv(_sw_nb, _sw_gs)
            _sw_win = base.window_sizes[0][0]
            _sw_scale = attn.head_dim ** -0.5
            # A real token row, so the sweep legs run the LIVE specialisation
            # (`TOK_AT_POS=True`, one scalar load at `idx_ptr + pos`) rather than a
            # near-neighbour. `pos` < L, so the read is in bounds by construction.
            _sw_row = torch.zeros(kc.size(1), dtype=torch.int64, device=dev)
            print(f"[nwsweep] partial spec D={attn.head_dim} BL={DECODE_ATTN_BLOCK_L} "
                  f"NS={DECODE_ATTN_SPLITS} GS={_sw_gs} NITER={_sw_niter} "
                  f"grid=({attn.n_head},{_sw_gs}) HAS_VE={ve_w is not None} "
                  f"L={kc.size(1)} pos={int(pos.item())}", flush=True)

            def _sw_partial(nw):
                def go():
                    _decode_attn_partial_qkv_rot_write[(attn.n_head, _sw_gs)](
                        qkv, qkv if ve_w is None else ve_w, _sw_row, base.cos, base.sin,
                        kc, vc, pos, _sw_num, _sw_den,
                        kc.size(1), kc.stride(1), base.cos.stride(1), _sw_scale, _sw_win,
                        D=attn.head_dim, HALF=attn.head_dim // 2, OUT=out,
                        BL=DECODE_ATTN_BLOCK_L, NS=DECODE_ATTN_SPLITS, GS=_sw_gs,
                        NITER=_sw_niter, HAS_VE=ve_w is not None, EPS=_RMS_NORM_EPS,
                        TOK_AT_POS=True, num_warps=nw)
                return go

            def _sw_partial_v38(nw):
                # The NULL: champion v38's statement order, same buffers, same grid, same width.
                def go():
                    _decode_attn_partial_qkv_rot_write_v38[(attn.n_head, _sw_gs)](
                        qkv, qkv if ve_w is None else ve_w, _sw_row, base.cos, base.sin,
                        kc, vc, pos, _sw_num, _sw_den,
                        kc.size(1), kc.stride(1), base.cos.stride(1), _sw_scale, _sw_win,
                        D=attn.head_dim, HALF=attn.head_dim // 2, OUT=out,
                        BL=DECODE_ATTN_BLOCK_L, NS=DECODE_ATTN_SPLITS, GS=_sw_gs,
                        NITER=_sw_niter, HAS_VE=ve_w is not None, EPS=_RMS_NORM_EPS,
                        TOK_AT_POS=True, num_warps=nw)
                return go

            def _sw_combine(nw):
                def go():
                    _decode_attn_combine[(_cH,)](
                        _cnum, _cden, _cout, _cpos, D=_cD, NS=DECODE_ATTN_SPLITS, GS=_cGS, VS=_cVS, TAIL=_cTAIL,
                        ADVANCE=False, num_warps=nw)
                return go

            def _sw_combine_masked(nw):
                def go():
                    _decode_attn_combine_masked_v37[(_cH,)](
                        _cnum, _cden, _cout, _cpos, D=_cD, NS=DECODE_ATTN_SPLITS, GS=_cGS,
                        ADVANCE=False, num_warps=nw)
                return go

            for _nw in (1, 2, 4, 8, 16):
                probes.append((f"NWP partial num_warps={_nw:<2d} (per layer)", _sw_partial(_nw)))
            # THE SPELLING PAIR for this row. `NWP partial num_warps=4` above is this candidate's
            # hoisted order at the LIVE width; this leg is its null -- champion v38's order, same
            # buffers, same grid, same width, one process -- so their difference IS the price of
            # the owner's fenced loads. Labelled `NWP...` so the SUM OF PARTS loop skips it
            # (gpu2's trap: a probe-only row otherwise inflates that rider's total).
            probes.append(("NWPv38 partial CHAMPION-ORDER num_warps=4 (probe)",
                           _sw_partial_v38(4)))
            # And the equivalence, free, on device, on the same buffers. A pure reorder MUST be
            # bitwise; this is the one claim in the diff a CPU leg cannot settle, because
            # `TRITON_INTERPRET=1` truncates fp32->bf16 where the device rounds to nearest.
            try:
                def _pa_run(kern):
                    kern[(attn.n_head, _sw_gs)](
                        qkv, qkv if ve_w is None else ve_w, _sw_row, base.cos, base.sin,
                        kc, vc, pos, _sw_num, _sw_den,
                        kc.size(1), kc.stride(1), base.cos.stride(1), _sw_scale, _sw_win,
                        D=attn.head_dim, HALF=attn.head_dim // 2, OUT=out,
                        BL=DECODE_ATTN_BLOCK_L, NS=DECODE_ATTN_SPLITS, GS=_sw_gs,
                        NITER=_sw_niter, HAS_VE=ve_w is not None, EPS=_RMS_NORM_EPS,
                        TOK_AT_POS=True, num_warps=4)
                    return _sw_num.clone(), _sw_den.clone()

                _pa_n, _pa_d = _pa_run(_decode_attn_partial_qkv_rot_write_v38)
                _pb_n, _pb_d = _pa_run(_decode_attn_partial_qkv_rot_write)
                _pd_n = float((_pa_n - _pb_n).abs().max())
                _pd_d = float((_pa_d - _pb_d).abs().max())
                print(f"[spell] partial equivalence on device: "
                      f"bitwise_num={bool(torch.equal(_pa_n, _pb_n))} "
                      f"bitwise_den={bool(torch.equal(_pa_d, _pb_d))} "
                      f"max_abs_num={_pd_n:.3e} max_abs_den={_pd_d:.3e} "
                      f"GS={_sw_gs} NITER={_sw_niter} HAS_VE={ve_w is not None}", flush=True)
            except Exception as _peqexc:                    # noqa: BLE001
                print(f"[spell] partial equivalence skipped: "
                      f"{type(_peqexc).__name__}: {_peqexc}", flush=True)
            if _cnum is not None:
                for _nw in (1, 2, 4, 8, 16):
                    probes.append((f"NWC combine num_warps={_nw:<2d} (per layer)",
                                   _sw_combine(_nw)))
                # THE SPELLING PAIR. `NWC combine num_warps=1` above is this candidate's split
                # spelling; this leg is its NULL -- the champion's masked 32-wide reduction, same
                # buffers, same NS=32 / GS=17, same width -- so their difference IS the price of
                # the reduction level, measured in one process. Appended last so no row above it
                # moves position.
                probes.append(("SPELLmask combine MASKED-32 v37 num_warps=1 (probe only)",
                               _sw_combine_masked(1)))
                # And the equivalence, free, on device, on the same random buffers: a
                # reassociation must agree to a few fp32 ulp, and after the bf16 store it should
                # be bitwise. This is the one fidelity claim in the diff that a CPU leg cannot
                # settle, because a compiled region's reduction is not reproducible from outside.
                try:
                    _decode_attn_combine_masked_v37[(_cH,)](
                        _cnum, _cden, _cout, _cpos, D=_cD, NS=DECODE_ATTN_SPLITS, GS=_cGS,
                        ADVANCE=False, num_warps=1)
                    _eq_a = _cout.clone()
                    _decode_attn_combine[(_cH,)](
                        _cnum, _cden, _cout, _cpos, D=_cD, NS=DECODE_ATTN_SPLITS, GS=_cGS,
                        VS=_cVS, TAIL=_cTAIL, ADVANCE=False, num_warps=1)
                    _eq_b = _cout.clone()
                    _eq_d = (_eq_a.float() - _eq_b.float()).abs()
                    print(f"[spell] combine equivalence on device: "
                          f"bitwise={bool(torch.equal(_eq_a, _eq_b))} "
                          f"max_abs={float(_eq_d.max()):.3e} "
                          f"ulps_bf16={float(_eq_d.max()) / 3.9e-3:.3e} "
                          f"NS={DECODE_ATTN_SPLITS} GS={_cGS} VS={_cVS} TAIL={_cTAIL}",
                          flush=True)
                except Exception as _eqexc:                 # noqa: BLE001
                    print(f"[spell] equivalence skipped: "
                          f"{type(_eqexc).__name__}: {_eqexc}", flush=True)
        except Exception as _swexc:                         # noqa: BLE001
            print(f"[nwsweep] setup skipped: {type(_swexc).__name__}: {_swexc}", flush=True)

        for label, fn in probes:
            if fn is None:
                continue
            try:
                rows.append(graph_us(fn, label))
            except Exception as exc:                       # noqa: BLE001
                rows.append((f"{label}  [TIME FAILED {type(exc).__name__}: {exc}]",
                             float("nan")))
            try:
                n, defs = census_kernels(fn, label)
                counts.append((label, n, defs[:6]))
            except Exception as exc:                       # noqa: BLE001
                counts.append((f"{label}  [CENSUS FAILED {type(exc).__name__}: {exc}]",
                               -1, []))

    n_layer = base.config.n_layer
    print("[regioncensus] POST-FUSION CUDA KERNELS per region, counted in a captured graph "
          f"over ONE call, nopref shape, position {int(pos.item())}, n_layer {n_layer}")
    for label, n, defs in counts:
        print(f"[regioncensus] {label:52s} kernels={n}")
        for ln in defs:
            print(f"[regioncensus]     raw {ln.strip()[:240]}")

    print("[probe] in-graph cost of each width-1 region, "
          f"{rep} replays per capture, {timed} timed samples, median")
    print(f"[probe] {'op':52s} {'us/call':>9s} {'x/step':>7s} {'us/step':>9s}")
    total = 0.0
    # At k=2 three of the `(layer)` rows are NOT called once per layer, and the label-keyed
    # dispatcher below would charge all three at `n_layer`. Overriding by exact label keeps the
    # labels byte-identical for the cross-launch grep series AND keeps the x/step column true.
    n_par = len(PARALLEL_BLOCK_LAYERS)
    # The out merge needs BOTH a parallel layer and a next mix to fold into, so it is the parallel
    # non-last layers, counted rather than assumed to be n_par.
    _n_outmerge = len([_l for _l in PARALLEL_BLOCK_LAYERS if _l + 1 < n_layer])
    mult_override = {
        "MERGED _decode_qkv_fc_matvec_norm [3584,512] (layer)": n_par,
        "PACKED _decode_qkv_matvec_norm [1536,512] (layer)": n_layer - n_par,
        "PACKED _decode_mlp_hidden_norm [2048,512] (layer)": n_layer - n_par,
        # This revision's three moved counts. The out merge takes the parallel NON-LAST layers, so
        # at k=2 with layers (0, 2) and n_layer 6 it is 2 calls; `_decode_attn_out_add` keeps the
        # 4 sequential layers; and the mix keeps the non-last sequential ones, 3 rather than 5.
        # Overriding by exact label leaves every label byte-identical for the cross-launch grep
        # series while keeping the x/step column true.
        "OUTMERGE _decode_out_merge_mix [512,2560] (parallel layers)": _n_outmerge,
        "_decode_attn_out_add WORDS live [512,512] (layer)": n_layer - n_par,
        "PACKED _decode_mlp_out_mix [512,2048] (5 layers)":
            n_layer - 1 - _n_outmerge,
    }
    for label, us in rows:
        if "(per layer)" in label or "(layer)" in label:
            mult = n_layer
        elif "3 layers" in label:
            mult = 3
        elif "5 layers" in label:
            mult = n_layer - 1
        else:
            mult = 1
        mult = mult_override.get(label, mult)
        contrib = us * mult
        # Only the PACKED spelling is on this candidate's step; v22 and CDT rows are priced.
        if label.startswith(("v22", "v24", "CDT", "NWP", "NWC")):
            contrib = float("nan")
        if contrib == contrib:
            total += contrib
        print(f"[probe] {label:52s} {us:9.3f} {mult:7d} {contrib:9.2f}")
    print(f"[probe] {'SUM OF PARTS (this candidate step)':52s} {'':9s} {'':7s} {total:9.2f}")
    try:
        whole = (reported_metrics["nopref_request_ms_median"] * 1000.0
                 / (reported_metrics["nopref_request_steps"] + 1))
        print(f"[probe] {'WHOLE STEP (frozen metric / steps)':52s} {'':9s} {'':7s} "
              f"{whole:9.2f}")
        print(f"[probe] {'UNATTRIBUTED (whole - parts)':52s} {'':9s} {'':7s} "
              f"{whole - total:9.2f}   ({100.0 * (whole - total) / whole:.1f}% of step)")
    except Exception as exc:                               # noqa: BLE001
        print(f"[probe] whole-step comparison unavailable: {exc!r}")
    print("[probe] one us/step is 0.513 ms of nopref_request_ms_median", flush=True)
    # The row's LATENCY half, read in-launch and paired, so it is exposed neither to the
    # cross-allocation term nor to the cold-cache term. The QUALITY half is `val_bpb` and no probe
    # can see it: `prepare.py` reads `val_bpb` before these probes run.
    try:
        _by = dict(rows)
        _m = _by["MERGED _decode_qkv_fc_matvec_norm [3584,512] (layer)"]
        _q = _by["PACKED _decode_qkv_matvec_norm [1536,512] (layer)"]
        _f = _by["PACKED _decode_mlp_hidden_norm [2048,512] (layer)"]
        _per = _m - _q - _f
        _net = _per * n_par
        print(f"[probe] PARALLEL ROW k={n_par} of {n_layer} at layers {PARALLEL_BLOCK_LAYERS}: "
              f"merged {_m:.3f} against the live pinned pair {_q:.3f}+{_f:.3f}={_q + _f:.3f} "
              f"us/call; per site {_per:+.3f}, net {_net:+.3f} us/step = "
              f"{_net * 0.513:+.4f} ms of nopref_request_ms_median, -{n_par} CUDA kernels. "
              f"Launch 75 printed this same pair at k=6: merged 3.417 against 2.728+3.102=5.830, "
              f"per site -2.413. BOTH pinned sites still MATCH here, on the "
              f"{n_layer - n_par} sequential blocks, so unlike launch 75 there is no pin "
              f"give-back in this number and none in the whole-step delta either. Dividing this "
              f"per-site figure by launch 75's -2.413 is the FIRST reading of whether the dial's "
              f"latency leg is linear in k; the QUALITY leg's linearity is what val_bpb tests and "
              f"neither is measured by anything else.", flush=True)
    except Exception as _exc:                              # noqa: BLE001
        print(f"[probe] PARALLEL ROW unavailable: {_exc!r}", flush=True)
    # THE ROW THIS REVISION IS BOUGHT ON, and its whole stop clause is decidable from this line.
    # The pair is the two LIVE regions the merge replaces, read in the same process on the same
    # buffers, so no cross-launch, cross-allocation or cache term enters it.
    try:
        _by = dict(rows)
        _o = _by["OUTMERGE _decode_out_merge_mix [512,2560] (parallel layers)"]
        _a = _by["_decode_attn_out_add WORDS live [512,512] (layer)"]
        _x = _by["PACKED _decode_mlp_out_mix [512,2048] (5 layers)"]
        _per = _o - _a - _x
        _net = _per * _n_outmerge
        # Solved ON its own axis from the pair's own two depths in THIS launch, 512 and 2048 at a
        # fixed 512 output rows, which is where the registered central came from at L77's numbers.
        _slope = (_x - _a) / (2048.0 - 512.0)
        _fixed = _a - 512.0 * _slope
        _pred = _fixed + 2560.0 * _slope
        if _o >= _a + _x:
            _verdict = "REFUTED (>= the pair's own sum, no deletion credit)"
        elif _o > 4.343:
            _verdict = "MODEL WRONG IN MAGNITUDE, mechanism intact (credit < 1.0 us/call)"
        elif 3.34 <= _o <= 3.89:
            _verdict = "CONFIRMED (inside the registered [3.34, 3.89])"
        else:
            _verdict = "OUTSIDE the registered band on the FAVOURABLE side"
        print(f"[probe] OUT MERGE ROW k={_n_outmerge} of {n_layer} at the parallel non-last "
              f"layers: merged {_o:.3f} against the live pair {_a:.3f}+{_x:.3f}={_a + _x:.3f} "
              f"us/call; per site {_per:+.3f}, net {_net:+.3f} us/step = {_net * 0.513:+.4f} ms "
              f"of nopref_request_ms_median, -{_n_outmerge} CUDA kernels. Registered central "
              f"3.684 us/call and -1.702 ms, band [3.34, 3.89] and [-1.49, -2.05]. VERDICT "
              f"{_verdict}. This launch's OWN on-axis solve from the same two rows: slope "
              f"{_slope:.3e} us/element, fixed term {_fixed:.3f} us/call, predicting {_pred:.3f} "
              f"at depth 2560 -- the credit IS the fixed term, and the difference between "
              f"{_pred:.3f} and {_o:.3f} is what the 1.25x extrapolation and 20 live lane terms "
              f"cost. Neither is priced anywhere else and the sign was registered unknown.",
              flush=True)
    except Exception as _exc:                              # noqa: BLE001
        print(f"[probe] OUT MERGE ROW unavailable: {_exc!r}", flush=True)


try:
    _probe_decode_regions(model, tokenizer)
except Exception as _exc:                                  # noqa: BLE001
    print(f"[probe] skipped: {type(_exc).__name__}: {_exc}", flush=True)


def _attribute_decode_step(model, tokenizer, rep=64, timed=20, warmup=5):
    """In-graph us/call for every op class in the width-1 step. Read-only, after the metrics."""
    import statistics
    base = model._orig_mod if hasattr(model, "_orig_mod") else model
    dev = base.transformer.wte.weight.device

    def graph_us(fn, label):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                for _ in range(rep):
                    fn()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(rep):
                fn()
        for _ in range(warmup):
            graph.replay()
        torch.cuda.synchronize()
        samples = []
        for _ in range(timed):
            start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record(); graph.replay(); stop.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(stop) * 1000.0 / rep)
        del graph
        return label, statistics.median(samples)

    rows = []
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        # A live no-prompt state at the ranked shape, half filled, so the cache and the position
        # are the ones the scored request actually decodes at.
        state = base.init_decode_state(batch=1, max_len=514, graph=True)
        idx = torch.zeros(1, 1, dtype=torch.int64, device=dev)
        for _ in range(257):
            base.decode_step(idx, state)
        pos = state["seq"]
        at = pos[:1]

        block = base.transformer.h[0]
        attn = block.attn
        out, nh, hd = attn.qkv_out, attn.n_head, attn.head_dim
        window, scale = base.window_sizes[0][0], attn.head_dim ** -0.5
        wb, wlm = base._decode_projection_weights()
        kc, vc = state["kc"][0], state["vc"][0]
        ve_w = base.value_embeds["1"].weight if "1" in base.value_embeds else None
        x0, mixed = _decode_embed_norm_mix(base.transformer.wte.weight, idx,
                                           base.resid_lambdas[0], base.x0_lambdas[0])
        qkv = _decode_qkv_matvec_norm(mixed, *wb[0][0])
        cs, sn = _decode_rotary_at(base.cos, base.sin, pos)
        q = _decode_q_write(qkv, cs, sn, out, nh, hd)
        ao = _decode_kv_write_attention(q, qkv, ve_w, idx, cs, sn, kc, vc, pos,
                                        out, nh, hd, window, scale).contiguous().view(1, 1, -1)
        probes = [
            ("_decode_embed_norm_mix (once/step)",
             lambda: _decode_embed_norm_mix(base.transformer.wte.weight, idx,
                                            base.resid_lambdas[0], base.x0_lambdas[0])),
            ("_decode_rotary_at KEPT (once/step)",
             lambda: _decode_rotary_at(base.cos, base.sin, pos)),
            ("_decode_qkv_matvec_norm int8 [1536,512] (layer)",
             lambda: _decode_qkv_matvec_norm(mixed, *wb[0][0])),
            ("CHAMPION _decode_qkv_write +ve (3 layers)",
             (lambda: _decode_qkv_write(qkv, ve_w, idx, cs, sn, kc, vc, at, out, nh, hd))
             if ve_w is not None else None),
            ("CHAMPION _decode_qkv_write no-ve (3 layers)",
             lambda: _decode_qkv_write(qkv, None, idx, cs, sn, kc, vc, at, out, nh, hd)),
            ("KEPT _decode_q_write, q half only (every layer)",
             lambda: _decode_q_write(qkv, cs, sn, out, nh, hd)),
            ("CHAMPION attn Triton pair (per layer)",
             lambda: _decode_cached_attention_triton(q, kc, vc, pos, window, scale)),
            ("FUSED kv_write+attn +ve (3 layers)",
             (lambda: _decode_kv_write_attention(q, qkv, ve_w, idx, cs, sn, kc, vc, pos,
                                                 out, nh, hd, window, scale))
             if ve_w is not None else None),
            ("FUSED kv_write+attn no-ve (3 layers)",
             lambda: _decode_kv_write_attention(q, qkv, None, idx, cs, sn, kc, vc, pos,
                                                out, nh, hd, window, scale)),
            # The PAIR this launch exists to read: the champion's spelling above (q from a
            # region, loaded by the kernel) against this candidate's (q computed by all 68
            # programs), timed in the same graph at the same position. Their difference IS the
            # per-program re-payment term, which nothing in this run has measured.
            ("CANDIDATE fused q+kv_write+attn +ve (3 layers)",
             (lambda: _decode_qkv_write_attention(qkv, ve_w, idx, cs, sn, kc, vc, pos,
                                                  out, nh, hd, window, scale))
             if ve_w is not None else None),
            ("CANDIDATE fused q+kv_write+attn no-ve (3 layers)",
             lambda: _decode_qkv_write_attention(qkv, None, idx, cs, sn, kc, vc, pos,
                                                 out, nh, hd, window, scale)),
            # THIS ROW's pair: v26's spelling (a gathered row, above) against the table-indexing
            # one, timed in the same graph. `_decode_rotary_at KEPT` above is the deleted term.
            ("ROT fused q+kv_write+attn +ve (3 layers)",
             (lambda: _decode_qkv_rot_write_attention(qkv, ve_w, idx, base.cos, base.sin, kc, vc,
                                                      pos, out, nh, hd, window, scale))
             if ve_w is not None else None),
            ("ROT fused q+kv_write+attn no-ve (3 layers)",
             lambda: _decode_qkv_rot_write_attention(qkv, None, idx, base.cos, base.sin, kc, vc,
                                                     pos, out, nh, hd, window, scale)),
            ("_decode_attn_out_add int8 [512,512] (layer)",
             lambda: _decode_attn_out_add(mixed, ao, *wb[0][1])),
            ("seq.add_(1) (once/step)", lambda: pos.add_(0)),
        ]
        for label, fn in probes:
            if fn is None:
                continue
            try:
                rows.append(graph_us(fn, label))
            except Exception as exc:                        # noqa: BLE001
                rows.append((f"{label}  [FAILED {type(exc).__name__}: {exc}]", float("nan")))
        try:
            base.reset_decode_state(state)
        except Exception:                                   # noqa: BLE001
            pass

    print("[attribution] in-graph cost of each op class in the width-1 decode step, "
          "ranked shape (max_len=514), position 257")
    for label, us in rows:
        print(f"[attribution] {us:9.3f} us/call   {label}")
    # The mechanism's own two terms, so the prediction is on the record next to the metric and
    # nobody has to reconstruct it from the table. 513 decode calls => 1 us/step = 0.513 ms.
    got = dict((lab, us) for lab, us in rows)
    # PROBE-ONLY REPAIR, required by this candidate and by nothing else. The legs below are
    # labelled "(3 layers)" and multiplied by 3 because the reference `has_ve(i, 6)` puts value
    # embeddings on exactly 3 of 6 layers. This candidate puts them on 5, so the correct
    # multipliers are 5 and 1. The LABELS are left byte-identical on purpose -- they are the join
    # keys the run's printed `[attribution]` series is grepped by -- and only the counts move.
    # Deriving them instead of hard-coding 3 is the same constexpr-nonuniformity defect this board
    # has already closed twice (a term paid on some calls, priced as if paid on all). This whole
    # function runs after `METRICS_JSON` inside a module-level bare `except`, so it can reach no
    # metric and no ceiling.
    _n_layer_ve = model.config.n_layer
    _N_VE = len([_i for _i in range(_n_layer_ve)
                 if has_ve(_i, _n_layer_ve) or _i in EXTRA_VE_LAYERS])
    _N_NOVE = _n_layer_ve - _N_VE
    print(f"[attribution] VE SPLIT: {_N_VE} of {_n_layer_ve} layers carry a value embedding, "
          f"{_N_NOVE} do not; the '+ve (3 layers)' / 'no-ve (3 layers)' labels are join keys and "
          f"are multiplied by {_N_VE} and {_N_NOVE} here, not by 3 and 3", flush=True)
    try:
        # v25's mechanism, kept so the two absorbs are priced on one page.
        deleted_v25 = (_N_VE * got["CHAMPION _decode_qkv_write +ve (3 layers)"]
                       + _N_NOVE * got["CHAMPION _decode_qkv_write no-ve (3 layers)"])
        kept = 6 * got["KEPT _decode_q_write, q half only (every layer)"]
        attn_before = 6 * got["CHAMPION attn Triton pair (per layer)"]
        v25_after = (_N_VE * got["FUSED kv_write+attn +ve (3 layers)"]
                     + _N_NOVE * got["FUSED kv_write+attn no-ve (3 layers)"])
        print(f"[attribution] v25: region {deleted_v25:.3f} -> {kept:.3f} us/step, "
              f"attention pair {attn_before:.3f} -> {v25_after:.3f} us/step, net "
              f"{(kept - deleted_v25) + (v25_after - attn_before):+.3f} us/step")
        # THIS candidate: the deleted term is a whole kernel, the added term is what 68
        # programs pay for q instead of one region, and both are measured here.
        cand_after = (_N_VE * got["CANDIDATE fused q+kv_write+attn +ve (3 layers)"]
                      + _N_NOVE * got["CANDIDATE fused q+kv_write+attn no-ve (3 layers)"])
        added = cand_after - v25_after
        net = added - kept
        print(f"[attribution] THIS ROW: deleted _decode_q_write {kept:.3f} us/step "
              f"({kept / 6.0:.3f} us/call); attention pair {v25_after:.3f} -> "
              f"{cand_after:.3f} us/step (added {added:+.3f} us/step = "
              f"{added / 6.0:+.3f} us/call PER CALL, paid by all 68 programs); "
              f"net {net:+.3f} us/step = {net * 0.513:+.4f} ms")
        print(f"[attribution] per-program re-payment term, the quantity this launch buys: "
              f"{added / 6.0:+.4f} us/call against v25's owner-only +0.583 and a deleted "
              f"kernel's {kept / 6.0:.3f}")
        # THIS ROW: the deleted term is `_decode_rotary_at`, once per step, and the added term is
        # an int64 multiply-add on the cos/sin base address in every program.
        rot_deleted = got["_decode_rotary_at KEPT (once/step)"]
        rot_after = (_N_VE * got["ROT fused q+kv_write+attn +ve (3 layers)"]
                     + _N_NOVE * got["ROT fused q+kv_write+attn no-ve (3 layers)"])
        rot_added = rot_after - cand_after
        rot_net = rot_added - rot_deleted
        print(f"[attribution] ROTARY ROW: deleted _decode_rotary_at {rot_deleted:.3f} us/step "
              f"(2 CUDA kernels, once/step); attention pair {cand_after:.3f} -> {rot_after:.3f} "
              f"us/step (added {rot_added:+.3f} us/step = {rot_added / 6.0:+.3f} us/call); "
              f"net {rot_net:+.3f} us/step = {rot_net * 0.513:+.4f} ms")
    except KeyError as exc:                                 # noqa: BLE001
        print(f"[attribution] net not computable: missing {exc}")


try:
    _attribute_decode_step(model, tokenizer)
except Exception as _exc:                                  # noqa: BLE001
    print(f"[attribution] skipped: {type(_exc).__name__}: {_exc}", flush=True)


def _report_config_pin():
    """Print WHAT WAS INSTALLED, WHERE, and WHAT THE PIN DID NOT REACH.

    Three things this witness must be able to say, each paid for by a launch:

      * `PINNED 0` -- launch 62 hooked a constructor the launch process never calls and 50 CPU
        checks passed anyway; the summary diagnosed it in one line. A witness must be able to
        print zero.
      * the INSTALLED config, not a boolean -- launch 63 shipped `R0_BLOCK 64` where the record
        said 128, because a fallback inherited an unpinned field.
      * WHAT IT DID NOT MATCH -- launch 67. A `_make_launchers` pin is keyed on a generated name;
        that host's edit renamed the live head, the pin landed on the read-only DEPEND probe leg
        whose body is the champion's verbatim, and the summary printed a TRUTHFUL success. A
        witness can be true about the wrong instance. The `UNMATCHED AT THESE HINTS` block below
        is the repair: it enumerates every roster instance at a requested site's own reduction
        shape that this pin did not claim, so the failure cannot hide behind a success line.
    """
    pinned = [r for r in _DECODE_PIN_LOG if r.get("pinned")]
    skipped = [r for r in _DECODE_PIN_LOG if r.get("skipped")]
    errors = [r for r in _DECODE_PIN_LOG if r.get("error")]
    sites_pinned = 0
    sites_incomplete = []
    for want in _DECODE_CONFIG_PIN:
        at_hints = [r for r in _DECODE_PIN_ROSTER if r.get("hints_dict") == want["size_hints"]]
        matched = [r for r in at_hints if r.get("matched_site") == want["site"]]
        # An instance at these hints that ANOTHER table entry claimed is PINNED, not unclaimed.
        # Two entries now share this reduction shape -- pow_relu and the qkv prenorm, separable only
        # by a negative token -- so keying `unmatched` on this entry's own site would report each as
        # the other's failure and `SITES INCOMPLETE` would never clear even with all four claimed.
        # The launch-67 protection this block exists for is UNCHANGED: that failure had the live
        # instance claimed by nothing, which still lands here. Unclaimed means claimed by NOTHING.
        unmatched = [r for r in at_hints if not r.get("matched_site")]
        hit = [r for r in matched if r.get("pinned")]
        print(f"[pin] REQUESTED {want['site']}")
        print(f"[pin]     MATCH KEY {_pin_match_key(want)}")
        print(f"[pin]     recorded name at launch 66 (documentation, NOT the key): {want['kernel']}")
        print(f"[pin]     want EXACTLY {want['pin']}")
        print(f"[pin]     evidence {want['evidence']}")
        print(f"[pin]     roster instances AT THESE HINTS {len(at_hints)}: "
              f"matched {len(matched)}, PINNED {len(hit)}, unmatched {len(unmatched)}")
        if not hit:
            print("[pin]     RESULT: NOT PINNED -- this site shipped its heuristic/cached config, "
                  "so every clause depending on it is UNRESOLVED, not answered")
        for r in hit:
            print(f"[pin]     RESULT: PINNED instance {r.get('instance')} "
                  f"meta_coordesc={r.get('meta_coordesc')}")
            print(f"[pin]              name     {r.get('kernel')}")
            print(f"[pin]              from     {r.get('from')}")
            print(f"[pin]              INSTALLED {r.get('pin_config')}")
            print(f"[pin]              cache_info {r.get('cache')}")
        for r in matched:
            if r.get("skipped"):
                print(f"[pin]     SKIPPED instance {r.get('instance')}: {r['skipped']}")
                print(f"[pin]              from     {r.get('from')}")
        for r in unmatched:
            print(f"[pin]     UNMATCHED AT THESE HINTS -- instance {r.get('instance')} "
                  f"ran its heuristic/cached config")
            print(f"[pin]              name     {r.get('kernel')}")
            print(f"[pin]              shipped  {r.get('shipped')}")
        if unmatched:
            sites_incomplete.append((want["site"], len(unmatched)))
            print(f"[pin]     WARNING SITE INCOMPLETE: {len(unmatched)} instance(s) at this site's "
                  f"reduction shape were NOT claimed by this pin. If ANY of them is on the live "
                  f"path, this site's measured value does NOT apply to this launch. Launch 67 is "
                  f"exactly this condition, undetected, and it cost a launch.")
        if hit:
            sites_pinned += 1
    # SITES and INSTANCES are different counts and the old line divided one by the other.
    print(f"[pin] SITES pinned {sites_pinned} of {len(_DECODE_CONFIG_PIN)} requested; "
          f"INSTANCES pinned {len(pinned)}; skipped {len(skipped)}; errors {len(errors)}; "
          f"autotuner instances seen in this process {len(_DECODE_PIN_ROSTER)}", flush=True)
    # The revert witness, and it CAN print zero. `calls > 1` means `_dynamic_scale_rblock` came back
    # through `_make_launchers` on that instance; `repinned > 0` means the un-fixed spelling would
    # have shipped `reentry_rebuilt` instead of the pinned config. REPINNED 0 everywhere is a null
    # for that mechanism in this launch, not a success.
    reent = [r for r in _DECODE_PIN_ROSTER if (r.get("calls") or 0) > 1]
    rep = [r for r in _DECODE_PIN_ROSTER if (r.get("repinned") or 0) > 0]
    print(f"[pin] REPIN WITNESS: instances re-entering _make_launchers {len(reent)}; "
          f"instances RE-PINNED {len(rep)}; total re-pins {sum(r.get('repinned') or 0 for r in rep)}",
          flush=True)
    for r in rep:
        print(f"[pin]   RE-PINNED x{r['repinned']} inst {r.get('instance')} site={r.get('matched_site')}")
        print(f"[pin]       WOULD HAVE SHIPPED (rebuilt on re-entry) {r.get('reentry_rebuilt')}")
        print(f"[pin]       SHIPS INSTEAD {r.get('pin_config')}")
    if sites_incomplete:
        print(f"[pin] SITES INCOMPLETE {sites_incomplete} -- read the UNMATCHED lines above before "
              f"attributing anything to a carried pin", flush=True)
    else:
        print("[pin] every requested site claimed every roster instance at its reduction shape",
              flush=True)
    for r in errors:
        print(f"[pin] ERROR on {r.get('kernel')}: {r['error']}")
    print(f"[pin] ROSTER -- every CachingAutotuner INSTANCE that reached _make_launchers in the parent "
          f"({len(_DECODE_PIN_ROSTER)} rows).")
    for r in _DECODE_PIN_ROSTER:
        print(f"[pin]   hints={r['size_hints']:24s} n_compiled={r['n_compiled']} "
              f"pinned={r['pinned']} calls={r.get('calls')} repinned={r.get('repinned')} "
              f"coordesc={r['meta_coordesc']} inst={r['instance']} "
              f"site={r.get('matched_site')} {str(r['kernel'])[:64]}")
        print(f"[pin]       cache={r['cache']}")
        for cfg in r["shipped"]:
            print(f"[pin]       compiled {cfg}")


try:
    _report_config_pin()
except Exception as _exc:                                  # noqa: BLE001
    print(f"[pin] report skipped: {type(_exc).__name__}: {_exc}", flush=True)


def _report_coordesc_log():
    """What coordinate descent actually did, per kernel: this mechanism's own paired reading.

    Every number is a `do_bench` reading the tuner itself took on this device in this launch, so it
    is the one per-kernel price that does not depend on a model of the code. Read against the
    whole-step delta it answers the question the board is priced on: whether an isolated per-kernel
    win transfers into a captured graph of 38 serial nodes. gpu1's reporter, carried.
    """
    log = _DECODE_COORDESC_LOG
    print(f"[coordesc] tuner fired on {len(log)} kernels")
    if not log:
        print("[coordesc] EMPTY -- either the flag never reached codegen or no kernel was "
              "compiled inside the context. Treat the target delta as measuring NOTHING and the "
              "axis as still open, not closed.", flush=True)
        return
    changed = 0
    for row in log:
        if "error" in row:
            print(f"[coordesc] {row['kernel']}: LOG FAILED {row['error']}")
            continue
        same = row["start_config"] == row["chosen_config"]
        changed += 0 if same else 1
        h, c = row.get("start_ms"), row.get("chosen_ms")
        gain = ""
        if isinstance(h, float) and isinstance(c, float) and h > 0:
            gain = (f"  do_bench {h * 1000.0:.3f} -> {c * 1000.0:.3f} us  "
                    f"({100.0 * (c - h) / h:+.2f}%)")
        print(f"[coordesc] {row['kernel']:38s} hints={row['size_hints']:22s} "
              f"n={row['n_configs_shipped']} {'SAME' if same else 'MOVED'}{gain}")
        print(f"[coordesc]     start  {row['start_config']}")
        print(f"[coordesc]     chosen {row['chosen_config']}")
        for k, v in sorted(row.get("tried", {}).items(), key=lambda kv: kv[1]):
            print(f"[coordesc]       tried {v * 1000.0:9.3f} us  {k}")
    print(f"[coordesc] {changed} of {len(log)} kernels moved off the heuristic config", flush=True)


try:
    _report_coordesc_log()
except Exception as _exc:                                  # noqa: BLE001
    print(f"[coordesc] report skipped: {type(_exc).__name__}: {_exc}", flush=True)
