"""
Autoresearch pretraining script. Single-GPU, single-file.
Cherry-picked and simplified from nanochat.
Usage: uv run train.py

Experiment red_mlp784_ungated_pair_on_v44 (team redundancy, run by autoscts__params_gpu3).
Champion v44's frozen bytes (launch 98, red_mlp784_ungated_l0_on_v43, mine, MEASURED at num_params_total
13,926,444 / val_bpb 1.0496969014170003 / bare margin 0.0003030985830 = 1.8607x the same-program spread
0.0001628969454063) with ONE literal:

    MLP_NARROW5_LAYERS  (0,)  ->  (0, 2)
        layer 2 JOINS layer 0 at the fifth rung: 896 - 16 - 32 - 16 - 32 - 16 = 784
        widths [784, 880, 784, 880, 800, 880], floor 784, layer 4 the only ungated layer above it
        -16,384  ->  num_params_total = 13,910,060
        784 = 16 x 49 and 784 x 2 = 1,568 bytes = 32 x 49, the same 32-byte free alignment class as
        800/816/832/848/864/880/896. hidden % 16 == 0 holds at 49 x 16; 784 % 8 == 0.

NO NEW CODE. Launch 98 built the mlp_narrow5 branch and this row only extends its tuple, exactly as launch
95 extended launch 92's mlp_narrow4 tuple. Rung 2 of a three-rung single-granule ladder, and it is priced
on the margin launch 98 MEASURED rather than on any extrapolation.

THIS IS A LEVELLING STEP, WHICH IS THE POINT. Layer 2 lands ON the floor 784 that launch 98 established; it
does not lower it. That is the run's cheapest and best-sampled position class:

    +0.0000180   L83 / L84   layer 4 864 -> 848 levelling      +0.11 spreads
    +0.0000600   L89 -> L90  layer 4 848 -> 832 levelling      +0.37 spreads
    -0.0001208   L96 -> L97  layer 4 816 -> 800 levelling      -0.74 spreads
    +0.0000060   L97 -> L98  layer 0 800 -> 784 DEEPENING      +0.04 spreads   (adjacent bound, mine)

    dearest levelling reading         +0.0000600  ->  cover 5.05x on 0.0003030985830
    dearest single granule, ANY class +0.0002955  ->  cover 1.026x   (L81->L84 interior, a class this
                                                      step is not in, carried so the dear end is not hidden)

Both intervals EXCLUDE 1 and there is no lever anywhere in this candidate to be the upside, which is the
strongest form of knowledge/a-lever-may-only-be-upside-never-collateral.md.

WHAT LAUNCH 98 SETTLED AND WHAT IT DID NOT. It settled that a single ungated granule that ENDS RAGGED is
free: +0.0000060 = +0.0368 spreads, refuting autoscts__params_gpu6's fixed raggedness penalty
P = 0.000213..0.000278 by 51x-73x and their proportional reading by 30x-56x. Six single-granule readings on
this axis now exist and every one is inside the instrument, while all four multi-granule readings are
outside it. It did NOT settle where the rung's cost lives. val_bpb is a function of the source and not of
the path, so [784,784,784] reached in three steps must cost exactly what it costs reached in one, and the
uniform-triple k-ladder puts the whole 800 -> 784 rung at +0.000231..+0.000344. Launch 98 paid +0.0000060 of
that. Either the remaining two granules carry +0.000225..+0.000338 between them, or the k-ladder over-prices
this rung. THIS LAUNCH IS THE MEASUREMENT THAT DECIDES IT, and it is why the ladder is walked one granule at
a time instead of taken as -32,768.

WHY NOT -32,768 IN ONE LAUNCH. Layers 2 and 4 together reach 13,893,676 and complete the uniform rung, which
would end UNIFORM and is the position class gpu6 argued is cheap. It is still refused: the remainder of the
rung is +0.000225..+0.000338 on its own k-ladder, cover 0.90x-1.35x, CONTAINS 1. And the specific warning is
already measured on this board -- L92 -> L95 is a LEVELLING PAIR and it cost +0.0004084 = +2.51 spreads,
outside the instrument, while every individual levelling reading is inside it. Levelling granules are not
reliably free in pairs. If this launch measures cheap on the measured margin, layer 4 is one more literal and
one more launch, which is exactly how this rung should be bought.

NO LEVER, AGAIN. The resid multiplier stays at the banked scalar_lr * 0.005 inherited from launches 96/97.
My next rung 0.0025 remains unmeasured, its sign at that rung unestablished, and my own launch 91 is this
run's proof that the next rung of a monotone axis can reverse; on a 1.8607-spread margin it would be
collateral and it would confound this granule's rate.

WHAT IS NOT CHANGED. MLP_RATIO 1.75, MLP_NARROW_LAYERS (0,1,2,3,4,5) at 16, MLP_NARROW2_LAYERS (0,2,4) at
32, MLP_NARROW3_LAYERS (0,2,4) at 16, MLP_NARROW4_LAYERS (0,2,4) at 32, MLP_NARROW5_ELEMENTS 16, so layer 4
stays at 800 and every gated layer stays at 880 -- gated mass is still 3.865197e-08/param measured (gpu4's
launch 87), cover 0.49x, and this row buys none of it. SCALAR_LR 1.0, resid_params lr scalar_lr * 0.005,
x0_params lr scalar_lr with betas (0.96, 0.95) which launch 94 measured LOAD-BEARING, ADAM_BETAS (0.8, 0.95),
spans [256,256,256,2048,256,2048], rotary base 100000, LOGIT_RANK 256, ve_gate_channels 8 with ONE shared
gate function, c_q h[1:-1], c_v h[3]<-h[1], c_k h[5]<-h[3], c_proj h[4]<-h[2], the value_embeds removal,
EMBEDDING_LR 0.3, UNEMBEDDING_LR 0.004, MATRIX_LR 0.04, WEIGHT_DECAY 0.2, TOTAL_BATCH_SIZE 2**18,
DEVICE_BATCH_SIZE 128, WARMDOWN_RATIO 0.7, WARMUP_RATIO 0.0, FINAL_LR_FRAC 0.0. No frozen region is touched
and prepare.py, pyproject.toml and uv.lock are byte-identical to champion/.

PEAK_VRAM IS DECLARED NON-BINDING RATHER THAN BANDED TIGHT. My launch-98 band was REFUTED: I built it from
the extremes of the run's three prior single-granule refunds and this granule refunded 17,890,304, more than
any of them, so the reading fell 556,032 below my low end. A band whose ends are the extremes of prior
observations is closed on exactly the side a new extreme arrives. Expected here: roughly 17.3-18.5 MB below
launch 98's 27,937,130,496, i.e. about 27.918-27.920 GB, against a ceiling of 47,198,976,512 -- two thirds of
the ceiling unused, so this metric constrains nothing and I am not pretending to a tight prediction on it.

CREDIT. The mlp_narrow rung mechanism (seventh use), the matched uniform-triple ladder whose k-values price
this rung, launch 92 which made launch 98 predictable, and the correction that caught my r_uniform
substitution before launch 98 was spent are all autoscts__params_gpu6's. The MLP_NARROW5 spelling, the
bare-margin sizing discipline and the transport measurement are autoscts__params_gpu4's (96). The at-floor
correction and a-sub-noise-cell-is-not-a-regime.md are autoscts__params_gpu1's (95), and it is gpu1's launch
95 that supplies the levelling-pair warning this row obeys. The inherited resid literal is mine (93). No
number in this docstring comes from any run other than this one.

Experiment red_mlp784_ungated_l0_on_v43 (team redundancy, run by autoscts__params_gpu3).
Champion v43's frozen bytes (launch 97, thr_mlp800_ungated_triple_resid_half_on_l95,
autoscts__params_gpu6, MEASURED at num_params_total 13,942,828 / val_bpb 1.049690903616262 / bare margin
0.000309096383738 = 1.8975x the same-program spread 0.0001628969454063) with ONE mechanism and no lever:

    NEW MLP_NARROW5_LAYERS = (0,) at MLP_NARROW5_ELEMENTS = 16
        layer 0 only: 896 - 16 - 32 - 16 - 32 - 16 = 784; layers 2 and 4 stay 800; gated {1,3,5} stay 880
        widths [784, 880, 800, 880, 800, 880], floor 784;  -16,384  ->  num_params_total 13,926,444
        784 = 16 x 49 and 784 x 2 = 1,568 bytes = 32 x 49, so 784 stays in the SAME 32-byte free
        alignment class as 800/816/832/848/864/880/896 (launch 52 -0.041% at 32-byte, launch 54 -0.992%
        at 16-byte, launch 51 -4.258% at 8-byte). The hidden % 16 == 0 assert holds at 49 x 16.

THE SIZE IS THE ARGUMENT AND IT IS THE SMALLEST GRANULE ON THE AXIS. Champion v43 holds 1.8975 spreads.

FIRST, A CORRECTION TO MY OWN PROPOSAL, WHICH autoscts__params_gpu6 CAUGHT. My proposal e07b855d priced
this step at 16,384 x r_uniform, cover [1.41x, 4.02x], taking gpu6's own axis_claims.md v61 table. In v62
they corrected that table against themselves: this step ENDS RAGGED (floor 784 with layers 2 and 4 sitting
16 above it) and r_uniform is the rate for a step that ends UNIFORM, so the substitution is the one their
own launch-97 finding forbids. THE [1.41x, 4.02x] IN MY PROPOSAL IS WITHDRAWN. Their re-pricing offers two
models fitted to the L96/L97 differencing cell, and under both the cover CONTAINS 1:

    READING 1  ragged rate = 1.937-2.156 x r_uniform     cost 0.000179 .. 0.000338   cover 1.72x .. 0.91x
    READING 2  fixed raggedness penalty P = 0.000213 ..  cost 0.000290 .. 0.000435   cover 1.07x .. 0.71x
               0.000278 on top of mass x r_uniform

I ACCEPT THE CORRECTION AND REFUSE READING 2, on this run's own measurement of THIS EXACT STEP TYPE.
Launch 92 is one ungated layer taken a rung below the other two, ending ragged with two layers 16 above
the new floor -- the identical shape of this candidate, one rung up, on the same bytes as launch 90 with
zero transport between them:

    L90  [832,880,832,880,832,880]  UNIFORM floor 832   val_bpb 1.0491167355999189
    L92  [816,880,832,880,832,880]  RAGGED  floor 816   val_bpb 1.0490519972641803   -0.397 spreads
    L95  [816,880,816,880,816,880]  UNIFORM floor 816   val_bpb 1.0494603493060290

    measured r for the matched ragged-ending granule   -3.951314e-09/param, a GAIN
    READING 2 predicts for that launch                 +1.88 .. +2.67 spreads
    error                                              2.27 .. 3.07 spreads, WRONG SIGN

READING 2 mis-predicts by 2.3-3.1 spreads the one launch that measured its own step type, and on that
lineage the RAGGED configuration has the LOWEST val_bpb of the three -- it beats the uniform floor above
it by 0.40 spreads and the uniform floor below it by 2.51. A fixed penalty of 1.31-1.70 spreads cannot be
carried by a configuration that reads better than both of its uniform neighbours. The reason it does not
hold is the one gpu6 wrote themselves ninety seconds earlier in their launch-97 close: the L96/L97
differencing cell is -0.7416 spreads, INSIDE the one-spread instrument, and is "a BOUND, not a rate to
multiply". P is 1.8-2.3x the size of the cell it is fitted from. This is
knowledge/a-sub-noise-cell-is-not-a-regime.md, which is autoscts__params_gpu1's file and which gpu6
themselves cited to refuse my at-floor rate.

RE-PRICED ON DIRECT MEASUREMENTS OF SINGLE GRANULES, WHICH IS ALL I WILL CLAIM. The run has five, and I
take the DEAREST as the bound rather than the matched one, so nothing here rests on launch 92 alone:

    L81 -> L84   interior granule 880 -> 864, a DIFFERENT position class   +0.0002955   cover 1.046x
    L89 -> L90   levelling granule                                        +0.0000600   cover 5.148x
    L83 /  L84   levelling granule                                        +0.0000180   cover 17.17x
    L90 -> L92   MATCHED ragged-ending granule                            -0.0000647   GAIN
    L96 -> L97   levelling granule                                        -0.0001208   GAIN

    union over all five:  cover [1.046x, INF)  --  EXCLUDES 1, and the dear end is 1.046x

THAT IS A THIN 1.046x AND MATERIALLY WEAKER THAN THE 1.41x I PROPOSED. It is stated as the reason to spend
the launch anyway rather than hidden: the dear end is the ONE reading taken from a layer far above the
floor, which is the position class this step is not in, and the matched reading is a gain. What I am not
doing is the thing gpu6 just did and recorded -- turning one sub-instrument differencing cell into a
structural constant and closing an axis with it.

AND THIS ROW IS THE ONLY REMAINING ROW THAT CAN DECIDE BETWEEN THE TWO READINGS. Band A or B kills
READING 2 and re-opens 13,910,060 on a measured margin; band C or D confirms gpu6's closure and the run's
floor is 13,942,828. Every branch buys the board a resolved structural question and one of them also banks
16,384. That asymmetry is the argument, and it is the same one that made launch 93 worth buying.

THE TWO DEEPER ROWS ARE STILL REFUSED, and gpu6's correction makes them worse, not better: -49,152 uniform
is cover [0.66x, 1.11x] under BOTH readings, and -32,768 ends ragged too. gpu6's structural point survives
their arithmetic and is the durable half of v62: hidden % 16 == 0 is asserted, so the smallest step that
ends UNIFORM is the full triple at 49,152, and shrinking a cut on this axis is exactly what makes it
ragged. Any gated granule is 3.865197e-08/param measured (gpu4's launch 87), cover 0.49x, refused by an
order of magnitude.

WHY NOT ADD MY OWN NEXT LEVER RUNG. My launch 93 measured resid_params lr scalar_lr * 0.01 -> * 0.005 at
-0.0004288267785 on champion v38's bytes; that literal is BANKED in this base already (launches 96 and 97).
The next rung 0.0025 is unmeasured and its SIGN at that rung is not established: my axis is monotone over
{0.005, 0.01, 0.02} on ONE base, the log2 vertex 0.00169 lies OUTSIDE that sampled range and
knowledge/exhausted.md sec 3 makes it inadmissible, and my own launch 91 is the run's proof that the
next rung of a monotone axis can reverse. An unmeasured lever cannot be upside on a 1.8975-spread margin:
if it turns it eats margin this cut was sized without it, which is the definition of collateral, and it
would also confound the granule rate this launch exists to measure. ONE mechanism, one measurement.

WHAT THIS LAUNCH MEASURES THAT NO EXISTING LAUNCH DOES. Every single-granule cell on this axis is inside
the same-program instrument (+0.0000180 at 864->848 levelling, +0.0000600 at 848->832 levelling,
-0.0000647 at 832->816 deepening, -0.0001208 at 816->800 levelling), while every uniform-triple rung is
outside it (+0.000668, +0.000595, +0.000344, and +0.000231 at f = 0). val_bpb is a function of the source
and not of the path, so those two families MUST reconcile: the four free single granules and the four
costly triples imply the cost inside a rung is NOT spread evenly across its three granules. This row is
the FIRST granule of the 800 -> 784 rung and it is the deepening one, so it reads the front of that
distribution directly, on the frontier's own bytes, at zero transport.

RATE BANDS ARE REGISTERED OPEN BELOW ZERO. autoscts__params_gpu6 recorded as a method error that their
four bands for launch 92, and my own four in 2235f717 which they adopted, were all floored at zero and
the outcome fell below every one of them. A saturating axis must be allowed to answer "this is free".

WHAT IS NOT CHANGED. MLP_RATIO 1.75, MLP_NARROW_LAYERS (0,1,2,3,4,5) at 16, MLP_NARROW2_LAYERS (0,2,4) at
32, MLP_NARROW3_LAYERS (0,2,4) at 16, MLP_NARROW4_LAYERS (0,2,4) at 32, so layers 2 and 4 stay at 800 and
every gated layer stays at 880 -- gated mass is the dearest parameter on the board at 3.865197e-08/param
(gpu4's launch 87), cover 0.30x here, and this row buys none of it. SCALAR_LR 1.0, resid_params lr
scalar_lr * 0.005, x0_params lr scalar_lr with betas (0.96, 0.95) which launch 94 measured LOAD-BEARING,
ADAM_BETAS (0.8, 0.95), spans [256,256,256,2048,256,2048], rotary base 100000, LOGIT_RANK 256,
ve_gate_channels 8 with ONE shared gate function, c_q h[1:-1], c_v h[3]<-h[1], c_k h[5]<-h[3],
c_proj h[4]<-h[2], the value_embeds removal, EMBEDDING_LR 0.3, UNEMBEDDING_LR 0.004, MATRIX_LR 0.04,
WEIGHT_DECAY 0.2, TOTAL_BATCH_SIZE 2**18, DEVICE_BATCH_SIZE 128, WARMDOWN_RATIO 0.7, WARMUP_RATIO 0.0,
FINAL_LR_FRAC 0.0. No frozen region is touched -- the "Parameter counts:" print block, the single
measure_flops_dispatch call with its returned value, and the single report_efficiency_metrics call with
its arguments are byte-identical to champion/train.py, harness=None -- and prepare.py, pyproject.toml and
uv.lock are byte-identical to champion/.

CREDIT. The base integer 13,942,828, the matched-mechanism uniform ladder, the ragged/uniform differencing
cell, the r_uniform band and the three-size cover table this row is chosen from are all
autoscts__params_gpu6's (launches 78, 80, 83, 90, 92, 97), as is the mlp_narrow rung mechanism this is the
sixth use of, the rule a-lever-may-only-be-upside-never-collateral.md, and the instruction that a rate band
must be open below zero. The MLP_NARROW5 spelling and the bare-margin sizing discipline are
autoscts__params_gpu4's (launch 96), together with the transport measurement that makes every f-carrying
cover table on this board too generous. The at-floor correction and
a-sub-noise-cell-is-not-a-regime.md are autoscts__params_gpu1's (launch 95). The banked resid literal is
mine (launch 93) and the banking pattern is autoscts__params_gpu2's (launch 89). No number in this
docstring comes from any run other than this one.

Experiment thr_mlp800_ungated_triple_resid_half_on_l95 (team throughput, run by autoscts__params_gpu6).
Champion v41's frozen bytes (launch 95, cap_mlp816_ungated_pair_on_l92, autoscts__params_gpu1, MEASURED at
num_params_total 13,991,980 / val_bpb 1.049460349306029 / bare margin 0.0005396506939710 = 3.313x the
same-program spread 0.0001628969454063) with TWO literals:

    MLP_NARROW4_ELEMENTS  16  ->  32
        mlp_narrow4_layers is ALREADY (0, 2, 4), so all three UNGATED layers step together:
        896 - 16 - 32 - 16 - 32 = 800 each;  gated {1,3,5} untouched at 880
        widths [800, 880, 800, 880, 800, 880];  -49,152  ->  num_params_total = 13,942,828
        800 = 16 x 50 and 800 x 2 = 1,600 bytes = 32 x 50, so 800 stays in the SAME 32-byte free
        alignment class as 816/832/848/864/880/896 (launch 52 -0.041% at 32-byte, launch 54 -0.992%
        at 16-byte, launch 51 -4.258% at 8-byte). The hidden % 16 == 0 assert holds at 50 x 16.

    setup_optimizer  resid_params lr  scalar_lr * 0.01  ->  scalar_lr * 0.005
        autoscts__params_gpu3's launch-93 lever. resid_params is [self.resid_lambdas], six scalars,
        read at exactly one site: x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0.
        ZERO parameters, zero counted FLOPs, zero allocation. x0_params is NOT touched: its lr stays
        scalar_lr and its betas stay (0.96, 0.95), which launch 94 measured as LOAD-BEARING.

NO NEW CODE. The mlp_narrow4 branch was built by launch 92 and this row only changes its element count;
the optimizer literal is one multiplier in an existing param_group dict.

THE RATE, FILTERED TO THE MATCHED MECHANISM. Every prior quotation of "the ungated width rate" pools four
different granule counts and three different base shapes. Filter to the IDENTICAL mechanism -- one
16-element granule on each of the three ungated layers, uniform floor before and after, 49,152 parameters
-- and exactly three readings survive, ALL of them above the same-program instrument:

    rung  floor          launches      d(val_bpb)      spreads   rate/param
    k2    864 -> 848     L80 -> L83    +0.0006681563    +4.102   1.359367e-08
    k3    848 -> 832     L86 -> L90    +0.0005954223    +3.655   1.211390e-08
    k4    832 -> 816     L90 -> L95    +0.0003436137    +2.109   6.990839e-09
    k5    816 -> 800     THIS ROW           unmeasured

MONOTONE DECREASING, ratios 0.891 then 0.577. Because the smallest cell is 2.109 spreads, none of these
is a sub-noise bound divided by mass -- the error autoscts__params_gpu5 retracted on this axis and the one
autoscts__params_gpu1 recorded against my own at-floor reading (3.664667e-09, which over-predicted by
3.40x and which this candidate does NOT use). k3 and k4 each span two launches, but the NET of each pair
is exactly the uniform one-granule triple step and nothing else.

THIS ROW IS DECLARED FINANCED ON THE LEVER, IN ACKNOWLEDGED BREACH OF MY OWN RULE. Bare cover for -49,152:

    6.990839e-09  cost +0.000343614  cover 1.571x   <- k4, nearest base, this exact mechanism
    1.211390e-08  cost +0.000595422  cover 0.906x
    1.359367e-08  cost +0.000668156  cover 0.808x   <- dearest matched

The interval [0.81x, 1.57x] CONTAINS 1, and knowledge/a-lever-may-only-be-upside-never-collateral.md --
which is mine -- says a contains-1 interval is undecidable and the fix is a smaller cut, not a better
argument. I am not pretending this row satisfies it. The lever must deliver f of its measured
-0.000428826779 (launch 93 vs launch 89, same integer 14,057,516, same bytes):

    f >= 0      at 6.990839e-09      f >= 0.170  at 1.246192e-08 (gpu1's realised at-floor pair)
    f >= 0.130  at 1.211390e-08      f >= 0.300  at 1.359367e-08 (dearest matched)
    f >= 0.808  at 1.803539e-08 -- the SINGLE-granule non-uniform-base outlier (L81 -> L84), which is
                a different mechanism class and is recorded here so the dear end is not hidden

WHY 30% IS NOT THE ASK THAT FAILED THREE TIMES. Launch 69, launch 73 and my own withdrawn
thr_longwin_quarter_gate_mlp864_on_v30 all financed a cut on long_window, and that lever did not weaken --
it REVERSED SIGN across a width step, +0.00029343 -> -0.00017831, because a span cut and a width cut are
one resource. Those rows needed f ~ 1 and got f < 0. resid_lambdas shares no tensor, no shape, no counted
FLOP and no allocation with MLP c_fc/c_proj hidden width, so the two limbs have no resource to contend
over; and gpu3's axis is monotone over THREE rungs on ONE base ({0.005, 0.01, 0.02}) with a stable
train-to-gate ratio (+3.43 up, +3.47 down) and better in every decile mean at 0.005.

THE ROW I AM DELIBERATELY NOT TAKING. -32,768 (two of three ungated to 800) -> 13,959,212 has bare cover
[1.008x, 1.32x] on its own matched readings, EXCLUDES 1, and therefore satisfies the rule this row
breaches. autoscts__params_gpu4 claimed it at 14:4xZ with the same lever banked as upside. My row is the
deep one that needs the lever; theirs is the shallow one that does not. Whichever clears sets the number.

WHAT IS NOT CHANGED. MLP_RATIO 1.75, MLP_NARROW_LAYERS (0,1,2,3,4,5) at 16, MLP_NARROW2_LAYERS (0,2,4) at
32, MLP_NARROW3_LAYERS (0,2,4) at 16, MLP_NARROW4_LAYERS (0,2,4), SCALAR_LR 1.0, x0_params betas
(0.96, 0.95), ADAM_BETAS (0.8, 0.95), spans [256,256,256,2048,256,2048], rotary base 100000, LOGIT_RANK
256, ve_gate_channels 8 with ONE shared gate function, c_q h[1:-1], c_v h[3]<-h[1], c_k h[5]<-h[3],
c_proj h[4]<-h[2], TOTAL_BATCH_SIZE 2**18, WARMDOWN_RATIO 0.7. No frozen region is touched and
prepare.py, pyproject.toml and uv.lock are byte-identical to champion/.

CREDIT. The lever and its three-rung bracket are autoscts__params_gpu3's (launches 89/91/93). The banking
pattern is autoscts__params_gpu2's (89). The base integer and the at-floor correction that stopped me
over-predicting are autoscts__params_gpu1's (95). The hand-off arithmetic and the retraction that made the
dear end honest are autoscts__params_gpu5's. mlp_narrow4 and the k2/k4 endpoints are mine (78, 80, 83, 90,
92). No number in this docstring comes from any run other than this one.

Experiment cap_mlp816_ungated_pair_on_l92 (team capacity, run by autoscts__params_gpu1).
Launch 92's frozen bytes (thr_mlp816_ungated_l0_on_v39, autoscts__params_gpu6, MEASURED at
num_params_total 14,024,748 / val_bpb 1.0490519972641803 / bare margin 0.0009480027358197 = 5.820x the
same-program spread 0.0001628969454063) with ONE literal:

    MLP_NARROW4_LAYERS  (0,)  ->  (0, 2, 4)
        layers 2 and 4 JOIN layer 0 at the fourth rung: 896 - 16 - 32 - 16 - 16 = 816 each
        widths [816, 880, 816, 880, 816, 880];  -32,768  ->  num_params_total = 13,991,980
        the run's first count below 14,000,000, and -72.20% of the 50,332,176 measured reference
        816 = 16 x 51 and 816 x 2 = 1,632 bytes = 32 x 51, so 816 stays in the SAME 32-byte free
        alignment class as 832/848/864/880/896 (launch 52 -0.041% at 32-byte, launch 54 -0.992% at
        16-byte, launch 51 -4.258% at 8-byte). The hidden % 16 == 0 assert holds at 51 x 16.

NO NEW CODE: launch 92 built the mlp_narrow4 branch and this row only extends its tuple. Gated {1,3,5}
untouched at 880.

BOTH GRANULES LAND ON THE FLOOR, AND THAT IS THE WHOLE DESIGN OF THIS ROW. After launch 92 the widths
are [816, 880, 832, 880, 832, 880], so the network minimum is ALREADY 816. Layers 2 and 4 stepping
832 -> 816 therefore each land EXACTLY ON the existing minimum and neither lowers it. That is the
LEVELLING regime of autoscts__params_gpu6's three-price table, the cheapest of the three, and this row
buys two granules of exactly the kind that regime was measured on -- one granule at a time, not a
multi-granule leap with the discount smeared across it. The misuse gpu6 named first in
knowledge/a-width-granule-has-three-prices-by-where-it-lands-on-the-minimum.md, and then caught
themselves committing, is applying the at-min rate to granules that do not land at the min. This row
cannot commit it: there is no interior granule in it.

I HELD THIS ROW FOR TWENTY MINUTES TO BUY THAT PROPERTY, AND THE HOLD IS THE POINT.
I had built the same target integer as -49,152 in one literal on champion v39 (MLP_NARROW3_ELEMENTS
16 -> 32, all three ungated 832 -> 816 at once, 47/47 static checks, never frozen and never submitted).
That row was a DEEPENING step priced by extrapolation on a two-member class, cover [1.32x, 1.48x], and
autoscts__params_gpu6 priced the same row at [0.66x, 1.10x] and refused it. I did not submit it, because
launch 92 was already in flight measuring ONE granule of the very rate that row needed THREE granules
of -- and because the lane serialises, so waiting cost about two minutes of lane time rather than a
launch. autoscts__params_gpu2 published that sequencing argument at 12:49Z when they declined a second
slot; this is the same argument and it paid twice over. Same target integer, one third less mass
(-32,768 not -49,152), bought in the cheap regime instead of the dear one, and priced on a measurement
taken ten minutes before submission instead of on a trend.

WHAT LAUNCH 92 ACTUALLY MEASURED, AND IT IS A REFUND, NOT A COST.

    v39 (L90)  14,041,132   val_bpb 1.049116735599919    widths [832,880,832,880,832,880]
    L92        14,024,748   val_bpb 1.0490519972641803   widths [816,880,832,880,832,880]
    -16,384 parameters for  -0.0000647383357386 of val_bpb  =  -0.397 spreads
    rate -3.951314e-09/param -- NEGATIVE. The k=4 deepening granule PAID the run 0.4 spreads of margin.

So gpu6's own pre-registered branch is the one that fired: the deepening ladder did not merely
decelerate past k=3, it went through zero. Their bound-based sizing needed k4 < 5.391018e-08 and it
came in negative. Honest reading: at -0.397 spreads the granule is FREE TO THE INSTRUMENT rather than
profitable -- one must not book a 0.4-spread gain as income, and I do not price any of it as income
below. Margin rose 0.0008832644001 -> 0.0009480027358 only because the measurement moved, not because
this candidate earns anything.

SIZING: COVER EXCLUDES 1 UNDER EVERY UNGATED RATE THIS RUN HAS EVER MEASURED.
Base val_bpb 1.0490519972641803, my bytes are its bytes plus one tuple element, so ZERO TRANSPORT:
predicted val_bpb = 1.0490519972641803 + cost(32,768). Margin 0.0009480027358197.

    measured levelling rate (L89->L90)  3.664667e-09  +0.0001201  1.0491720811  cover 7.89x  +5.08 sp
    levelling bound + 1 spread          1.360723e-08  +0.0004459  1.0494978750  cover 2.13x  +3.08 sp
    k3 PARTIAL deepening rate           1.633851e-08  +0.0005354  1.0495873776  cover 1.77x  +2.53 sp
    dearest UNGATED ever (L81->L84)     1.803539e-08  +0.0005910  1.0496429809  cover 1.60x  +2.19 sp
    BREAK-EVEN                          2.893075e-08              1.0500000000  cover 1.00x

COVER [1.60x, 7.89x] AND THE LEFTOVER CLEARS THE INSTRUMENT AT EVERY END, including the dearest ungated
granule ever measured on this substrate and a levelling rate inflated by a full same-program spread. The
break-even rate 2.893075e-08 is 1.60x the dearest ungated reading in 92 launches and is only exceeded by
the GATED interior rate (autoscts__params_gpu4's 3.865197e-08, launch 87) -- i.e. for this row to miss the
gate, levelling granules on UNGATED layers would have to cost more than interior granules on GATED ones,
inverting both axes of the three-price table at once. I predict that does not happen.

The second line of that table is the honest dear end and I want it stated plainly: launch 90's levelling
reading is only 0.37 spreads wide (+0.0000600419 for 16,384), so it bounds a granule rather than pricing
one. I therefore also carry the reading inflated by a whole spread, and the row still clears at 2.13x.

REGISTERED REFUTING OUTCOMES, IN THE RATE'S UNITS. realised rate = (val_bpb - 1.0490519972641803)/32,768.

    rate <= 3.664667e-09    val_bpb <= 1.0491721   LEVELLING CONFIRMED at a second base and at 2x mass.
                            The regime table is a rule and not a coincidence, and the remaining ungated
                            ladder should be priced by where a granule lands, not by k.
    3.665e-09 .. 1.361e-08  1.0491721 .. 1.0494979  levelling holds in KIND but is dearer per granule
                            when two are taken together; the discount is real and not fully additive.
    1.361e-08 .. 2.893e-08  1.0494979 .. 1.0500000  levelling is NOT distinguishable from a deepening
                            step at this mass: launch 90's 0.37-spread reading was mostly instrument and
                            my two-granule extension over-read it. Eligible, but the regime table should
                            then be demoted to "one granule, one base" and gpu6's caution was right.
    rate >= 2.893075e-08    val_bpb >= 1.05  INELIGIBLE. Ungated levelling mass costs more than gated
                            interior mass, which inverts the whole three-price table. I would report that
                            as my error in those words: I would have bought 32,768 of the cheapest-looking
                            mass on the board at the dearest realised price in the run.

DETERMINISTIC PREDICTIONS, each derived from measured pairs on this lineage.
    num_params_total              13,991,980   EXACT. -32,768 = 2 layers x 16 elements x 2 x 512.
    flops_per_token_measured     113,836,608   EXACT. flops_per_token_measured is exactly linear in the
                                 sum of MLP widths at 6,144 per element per layer, with zero residual on
                                 SEVEN consecutive pairs (L81->L84, L83->L87, L86->L89, L89->L90,
                                 L86->L90, L80->L83, L90->L92). 32 elements -> -196,608 from launch 92's
                                 114,033,216.
    step0_loss                      9.010914   EXACT, and step1_loss 8.906942 EXACT. MLP init reads
                                 n_embd, not hidden, and both c_proj are zero-initialised, so no width
                                 change can move either. If either moves, the diff touched something it
                                 should not have and this row is VOID as a width measurement.
    peak_vram_bytes    [28,004,000,000, 28,022,000,000]  BAND, not exact. The allocator charge is not
                                 linear in width -- measured per-element deltas run 948k-1,084k across
                                 pairs because the six layers carry different window spans. Launch 92
                                 read 28,038,055,936; two 16-element ungated granules should remove
                                 roughly 16-34 MB.
    num_steps                     [2620, 2665]  BAND. Launch 92 ran 2,631 (689,700,864 total tokens) and
                                 v39 ran 2,639; fewer counted FLOPs buys steps, but this lane's
                                 cross-launch throughput varies more than this row's effect, so a band.

WHAT IS NOT CHANGED. MLP_RATIO 1.75, MLP_NARROW_LAYERS (0,1,2,3,4,5) at 16, MLP_NARROW2_LAYERS (0,2,4)
at 32, MLP_NARROW3_LAYERS (0,2,4) at 16, MLP_NARROW4_ELEMENTS 16, SCALAR_LR 1.0 (launch 89's banked
lever), the c_q share h[1:-1], the c_v pair {1,3}, the c_k pair {3,5}, the attn c_proj pair {2,4},
ve_gate_channels 8 with ONE shared gate function, has_ve {1,3,5}, WINDOW_PATTERN "SSSL", LOGIT_RANK 256,
rotary base 100000, EMBEDDING_LR 0.3, UNEMBEDDING_LR 0.004, MATRIX_LR 0.04, WEIGHT_DECAY 0.2,
ADAM_BETAS (0.8, 0.95), WARMUP_RATIO 0.0, WARMDOWN_RATIO 0.7, FINAL_LR_FRAC 0.0, TOTAL_BATCH_SIZE 2**18,
DEVICE_BATCH_SIZE 128, DEPTH 6, ASPECT_RATIO 85, HEAD_DIM 128, DATA_BUDGET_TOKENS and seed 42 are all
exactly as launch 92 measured them. This is a ONE-LITERAL candidate.

WHAT I DELIBERATELY DID NOT BUILD. The gated {1,3,5} limb at any size -- 147,456 parameters, the largest
remaining inventory, and none of it buyable on gpu4's launch-87 rate of 3.865197e-08/param. A gated layer
taken 880 -> 832 is THREE granules of which only the last lands on the floor, so it prices at
[0.63x, 2.29x] and the interval contains 1; and it cannot be made smaller, because a gated layer's
distance to the floor is exactly 48 elements. autoscts__params_gpu6 published that row, withdrew it
themselves at 13:27:57Z, and autoscts__params_gpu5 claimed and then withdrew it at 13:40Z on the same
arithmetic. Three agents reached the same refusal independently; it is recorded here so a fourth does not
have to.

CREDIT. The mlp_narrow4 mechanism, the base measurement this row is priced on, the levelling reading, the
three-price regime table and the correction that closes the gated limb are all autoscts__params_gpu6
(78, 80, 83, 90, 92). The k-ladder's third rung and the mlp_narrow3 spelling are autoscts__params_gpu3
(84) and autoscts__params_gpu6. The gated interior rate is autoscts__params_gpu4 (87). The base integer
14,057,516 and the SCALAR_LR banking are autoscts__params_gpu2 (89), whose sequencing argument this row
is an application of. The cover-plus-leftover test is mine (59, 67, 70, 73). No number in this docstring
comes from any run other than this one.

Experiment thr_mlp816_ungated_l0_on_v39 (team throughput, run by autoscts__params_gpu6).
Champion v39's frozen bytes (launch 90, thr_mlp832_ungated_triple_on_l89, mine, MEASURED at
num_params_total 14,041,132 / val_bpb 1.049116735599919 / bare margin 0.0008832644000811 = 5.422x the
same-program spread 0.0001628969454063) with a FOURTH narrowing set on ONE ungated layer:

    MLP_NARROW4_LAYERS = (0,) at MLP_NARROW4_ELEMENTS = 16
        layer 0: 896 - 16 - 32 - 16 - 16 = 816; every other layer unchanged
        widths [816, 880, 832, 880, 832, 880];  -16,384  ->  num_params_total = 14,024,748
        816 = 16 x 51 and 816 x 2 = 1,632 bytes = 32 x 51, so 816 stays in the SAME 32-byte free
        alignment class as 832/848/864/880/896 (launch 52 -0.041% at 32-byte, launch 54 -0.992% at
        16-byte, launch 51 -4.258% at 8-byte), and 816 % 8 == 0.

THIS IS A DEEPENING STEP AND IS DELIBERATELY *NOT* PRICED AT MY OWN LAUNCH-90 RATE. Launch 90
measured a LEVELLING step -- a layer brought down TO the existing minimum -- at 3.6646670e-09/param,
0.54x the first rung. This step LOWERS the minimum (832 -> 816 on one layer while two stay at 832), so
it belongs to the deepening family and must be priced there. Carrying launch 90's number onto it is the
single misuse my own knowledge file
(knowledge/a-width-granule-has-three-prices-by-where-it-lands-on-the-minimum.md) names first, and I am
not committing it in the candidate that follows it.

FINANCED ON A BOUND, NOT ON AN EXTRAPOLATION, AND THAT DISTINCTION IS THE POINT. The deepening ladder
has THREE measured points on this lineage and its increments DECELERATE:

    k=1  base-min 880  {0,2} 880 -> 864   -32,768   6.813304e-09/param   1.00x r1   launch 78
    k=2  base-min 864  {0,2,4} 864 -> 848 -49,152   1.359367e-08/param   2.00x r1   launches 80->83
    k=3  base-min 848  {0,2} 848 -> 832   -32,768   1.633851e-08/param   2.40x r1   launches 86->89
    increments  +6.78e-09 then +2.75e-09, ratio 0.406

k=4 is unmeasured, so I do not assert a value for it; I price the row across the whole plausible range
and require it to clear at the PESSIMISTIC end:

    k3 rate held flat   1.633851e-08  cost +0.0002677  cover 3.30x  leftover 3.78 spreads
    gpu3 power law k^0.8422 = 2.19e-08  cost +0.0003588  cover 2.46x  leftover 3.22 spreads
    LINEAR in k         2.725322e-08  cost +0.0004465  cover 1.98x  leftover 2.68 spreads
    2x the k3 rate      3.267702e-08  cost +0.0005354  cover 1.65x  leftover 2.14 spreads

COVER EXCLUDES 1 AND THE LEFTOVER CLEARS THE INSTRUMENT AT EVERY END, including a linear-in-k reading
that every measured increment on this ladder contradicts. That is autoscts__params_gpu1's test passed on
a bound rather than on a point estimate. BREAK-EVEN is a k=4 rate of 5.391018e-08 = 3.30x k3 = 7.91x r1;
for the gate to be missed the ladder would have to REVERSE from decelerating to explosive between k3 and
k4. I register that as the refuting outcome and I predict it does not happen.

WHY THIS ROW AND NOT THE GATED LIMB -- A CORRECTION TO MY OWN PUBLISHED SIZING, MADE BEFORE ANYONE ACTS
ON IT. In champion v39's own record and in knowledge/axis_claims.md I wrote that ONE gated layer taken
880 -> 832 (-49,152, target 13,991,980) is "financed with cover 2.29x-4.90x". THAT NUMBER IS TOO CHEAP
AND THE ROW SHOULD BE REFUSED. I priced all 49,152 parameters at the gated at-min rate, but 880 -> 832
is THREE 16-element granules and only the LAST of them lands on the minimum: 880 -> 864 and 864 -> 848
are both INTERIOR steps, and autoscts__params_gpu4's launch 87 measured a gated interior granule at
3.865197e-08/param (+0.000633273829336 for -16,384). Priced marginally the row costs
2 x 0.000633274 + 16,384 x 7.85e-09 = +0.0013952, which is INELIGIBLE at 1.050512.

    published (all 49,152 at the gated at-min rate)   +0.0003860  cover 2.29x  ELIGIBLE
    marginal (2 interior granules + 1 at-min)         +0.0013952  cover 0.63x  INELIGIBLE
    cover interval [0.63x, 2.29x] CONTAINS 1  ->  REFUSE

The single gated granule 880 -> 864 on its own is no better: at gpu4's measured cost it is cover 1.39x
with 1.53 spreads left, but marked up 1.5x for base thinness it is cover 0.93x and INELIGIBLE, so that
interval contains 1 too. And all three ungated layers to 816 at once (MLP_NARROW3_ELEMENTS 16 -> 32,
-49,152, the same 13,991,980 integer by another route) is cover [0.66x, 1.10x] -- also containing 1.
So -16,384 on ONE ungated layer is the ONLY size on this board that clears at both ends, which is the
same conclusion autoscts__params_gpu2 reached one champion earlier by the same test.

WHAT autoscts__params_gpu2 DECLINED IS NOT THIS ROW. Their 12:49:42Z [SUGGESTION] declined 14,024,748
as "priced on an UNMEASURED r4" -- correctly, because from champion v38 that integer was -32,768, TWO
granules. From champion v39 it is -16,384, ONE granule, and the pessimistic end now clears at 1.98x
where theirs did not clear at all. My own launch 90 is what moved it; the row changed, not the test.

REGISTERED PREDICTIONS. Deterministic and EXACT or the row is wrong: num_params_total 14,024,748;
census wte 4,194,304 / value_embeds 0 / lm_head 131,072 / transformer_matrices 9,699,360 / scalars 12;
mlp 5,242,880; flops_per_token_measured 114,033,216 (= v39's 114,131,520 - 6 x 16,384) with
flops_per_token_analytic 90,439,872 and the 23,593,344 gap invariant for an EIGHTH launch;
training_data_tokens_available 631,241,817; step0_loss 9.010914; widths [816,880,832,880,832,880] --
if layer 0 reads 832 the new set did not bind. Banded: peak_vram_bytes 28,038,000,000 +/- 8%;
num_steps [2615, 2665], NO directional claim.

FOUR BRANCHES, STATED IN THE RATE'S UNITS as autoscts__params_gpu3 does it (2235f717), because a KEEP
can hide a large rate error and the rate is what the next row needs:
    k4 <= 1.90e-08                  val_bpb <= 1.0494280   ladder still decelerating; a FIFTH rung is
                                                           sizable and the gated limb stays unaffordable
    1.90e-08 .. 2.725322e-08        1.0494280 .. 1.0495633  power law or linear-in-k; ladder is bounded
    2.725322e-08 .. 5.391018e-08    1.0495633 .. 1.05       ACCELERATING -- still a KEEP, but the width
                                                           axis closes and no further rung is buyable
    k4 >= 5.391018e-08              val_bpb >= 1.05         INELIGIBLE; the ladder reversed to explosive
                                                           between k3 and k4, refuting all three measured
                                                           increments. Registered as the refuting outcome.

CREDIT. The deepening ladder's three measured points are autoscts__params_gpu6 (launch 78), gpu6 (80->83)
and autoscts__params_gpu5 + autoscts__params_gpu2 (86->89). The k-index, the power-law fit I price
against and the rate-units falsifier structure are autoscts__params_gpu3's (launch 84, 2235f717). The
gated interior rate that refuses the alternative limb is autoscts__params_gpu4's (launch 87). The
per-layer width mechanism, the ungated/gated split, the 32-byte free alignment class and the
cover-plus-leftover test are autoscts__params_gpu1's (launches 59/67/70/73). The "size so the bare
margin covers the pessimistic end" discipline is autoscts__params_gpu2's (launch 89). Mine are the
levelling/deepening distinction that keeps launch 90's rate OFF this row, and the correction to my own
gated sizing above.

Experiment thr_mlp832_ungated_triple_on_l89 (team throughput, run by autoscts__params_gpu6).
Launch 89's frozen bytes (cap_scalar_lr_bank_mlp832_pair_on_v36, autoscts__params_gpu2, MEASURED at
num_params_total 14,057,516 / val_bpb 1.0490566936955112 / bare margin 0.000943306304 = 5.791x the
same-program spread 0.0001628969454063) with ONE literal:

    MLP_NARROW3_LAYERS (0, 2) -> (0, 2, 4)
        LAYER 4 JOINS the third rung: 848 -> 832, so all three UNGATED layers sit at 832
        widths [832, 880, 832, 880, 832, 880];  -16,384  ->  num_params_total = 14,041,132

WHY THE BASE IS LAUNCH 89 AND NOT champion.md's v37. At build time champion.md reads v37 =
red_mlp_gated_l1_rung2_on_v36 (launch 87, autoscts__params_gpu4) at 14,073,900 / val_bpb
1.049815684921359 / margin 0.000184315079 = 1.13 spreads, and NOTHING is buyable on it: the smallest
width granule is 16,384 and the cheapest measured rate in the run puts that at 1.6x its whole margin.
Launch 89 measured 30 minutes ago and is strictly better on BOTH axes -- 16,384 fewer parameters AND
5.1x the margin -- because it banks autoscts__params_gpu5's SCALAR_LR 1.0 lever. Its result is
canonical (arena_result.json, status ok, eligible) and its source is frozen in the benchmark's own
candidates/ store, so it is this run's measured evidence, not a queued number. My target 14,041,132 is
strictly below v37 (-32,768) AND strictly below launch 89 (-16,384), so it promotes against whichever
of the two champion.md reads when I record. That is deliberate: building BELOW the frontier rather than
AT it is what makes the row independent of who publishes first.

THE RATE IS MEASURED ON THIS EXACT STEP, THIS LAYER CLASS AND THIS PRE-STEP WIDTH, AT ZERO TRANSPORT.
Launch 86 (thr_scalar_lr_hi_on_v36, autoscts__params_gpu5) and launch 89 differ by exactly this
mechanism and nothing else -- launch 89 IS launch 86's bytes plus MLP_NARROW3_LAYERS (0, 2) -- so the
pair prices the third rung with NO transport at all:

    launch 86   14,090,284   val_bpb 1.0485213133031897
    launch 89   14,057,516   val_bpb 1.0490566936955112
    delta       -32,768      +0.0005353803923215938  over TWO ungated layers at span 256,
                                                     pre-step width 848  ->  1.633851e-08/param
                                                     per 16,384-layer: +0.0002676901961608

My step is the THIRD layer taking that same step from that same pre-step width 848. Cost +0.0002676902
against the bare 0.000943306304: COVER 3.52x, LEFTOVER 0.0006756161 = 4.15 spreads. The row stays
ELIGIBLE up to 3.52x the measured rate, and it is financed on the BARE margin with no lever borrowed
(the lever is already banked in the base, so there is no lever to be collateral here at all).

AND IT SHOULD IF ANYTHING BE CHEAPER THAN THE MEASURED RATE, WHICH I AM NOT PRICING IN. Launch 89's
step took the network's minimum width DOWN (848 -> 832); mine brings the last layer down TO an existing
minimum, so base-min is 832 before and after. On autoscts__params_gpu3's k-indexes-the-PRE-step-width
reading both steps are k=3, and on knowledge/a-cut-price-is-convex-in-base-thinness.md a step that does
not deepen the minimum is the cheaper of the two. I price at 1.0x and report the markup the outcome
implies; I do NOT claim the discount in advance.

THIS RE-PRICES A ROW autoscts__params_gpu5 CORRECTLY REFUSED, AND THE REFUSAL WAS NOT A MISTAKE. In
knowledge/axis_claims.md v38, "Rows I am explicitly NOT claiming", gpu5 priced this same target 14,041,132
(reached as the one-literal MLP_NARROW2_ELEMENTS 32 -> 48, -49,152 in one step from v36) and refused it:
with no rung-3 measurement in existence they applied the measured rung2/rung1 convexity of 1.995 to a
rung-2 reading, got 2.712e-08..3.598e-08, and found cover [0.84x, 2.21x] -- an interval CONTAINING 1,
with the dear end ineligible at 1.050290. That was the right call on the evidence then. Launch 89 has
since MEASURED the rung-3 rate at 1.633851e-08, which is 1.66x BELOW the cheap end of that extrapolated
interval, and the measured rung3/rung2 convexity is 1.2019 rather than the assumed 1.995. Two things
changed, not one: the rate is measured instead of extrapolated, and the margin is 0.000943306 instead of
v36's bare 0.000817589. Same integer, different evidence.

WHAT I REFUSED, AND ON THE RUN'S OWN LEFTOVER TEST. Going to -32,768 needs a second granule, and both
candidates fail autoscts__params_gpu1's requirement that the leftover clear the instrument:
  - + the GATED limb (a {1,3,5} layer 880 -> 864), rate MEASURED by autoscts__params_gpu4's launch 87 at
    3.865197e-08/param (v36 -> v37, +0.000633273829336 for -16,384): total cost +0.0009010, leftover
    0.0000423 = 0.26 spreads, cover 1.047x. Refused -- below the instrument.
  - + a FOURTH rung (832 -> 816) on one layer: needs a new MLP_NARROW4 set AND its rate is unmeasured.
    The run's two convexity ratios are 1.995 (r2/r1) and 1.2019 (r3/r2), so r4 sits in
    [1.2, 2.0] x r3 and at the dear end the leftover is 0.86 spreads. Refused -- the interval's dear end
    is below the instrument, and this is exactly the extrapolate-from-same-direction-rungs error
    autoscts__params_gpu5's launch 88 recorded as REFUTED IN SIGN.
Declining 16,384 more parameters buys a row whose eligibility does not depend on any unmeasured number.

REGISTERED PREDICTIONS. Deterministic and EXACT or the row is wrong: num_params_total 14,041,132;
census wte 4,194,304 / value_embeds 0 / lm_head 131,072 / transformer_matrices 9,715,744 / scalars 12;
mlp 5,259,264; flops_per_token_measured 114,131,520 (= launch 89's 114,229,824 - 6 x 16,384, and the
identity 6 x P_matmul + 12 x 512 x 5,120 + 23,593,344 reproduces launches 83 and 89 exactly);
training_data_tokens_available 631,241,817; step0_loss 9.010914. Banded: peak_vram_bytes
28,055,000,000 +/- 8% (launch 89 read 28,072,560,640 and the per-element-layer coefficient has spanned
1,017,856..1,100,480 across four readings, so this band is centred on the mean and widened per
knowledge/peak-vram-bands-must-not-come-from-one-launch.md, which is my own file after missing three
times); num_steps [2610, 2650] with NO directional claim (launch 88 established +/-3 steps as node
noise at bit-identical counted work).

FOUR BRANCHES, and the third is still a KEEP. val_bpb <= 1.0493244 means the levelling step cost at or
below the measured rung-3 rate -- KEEP, and a fourth rung becomes sizable on a measured number.
1.0493244 .. 1.0495921 is a markup up to 2.0x -- KEEP, and the width axis is priced to its floor.
1.0495921 .. 1.05 is a markup of 2.0x-3.52x -- STILL A KEEP on the cut alone, and the width axis closes
because no further granule fits. val_bpb >= 1.05 is INELIGIBLE and requires the levelling rate to exceed
5.757e-08/param = 3.52x a rate measured on this same step two layers at a time, which would also refute
k-indexes-the-pre-step-width in the expensive direction; registered as the refuting outcome.

CREDIT. The base, the sizing-as-a-refusal discipline and the banked lever are autoscts__params_gpu2's
launch 89; the SCALAR_LR axis is autoscts__params_analyst3's (4582d135) and its measurement
autoscts__params_gpu5's launch 86; the MLP_NARROW3 spelling and k-indexes-the-pre-step-width are
autoscts__params_gpu3's launch 84; the per-layer width mechanism, the ungated/gated split, the 32-byte
free alignment class and the cover-plus-leftover test are autoscts__params_gpu1's launches 59/67/70/73;
the gated rung-2 rate that prices what I refused is autoscts__params_gpu4's launch 87; the target
integer 14,041,132 and the refusal I am re-pricing are autoscts__params_gpu5's. Mine is the levelling
reading and the re-pricing.

Experiment cap_scalar_lr_bank_mlp832_pair_on_v36 (team capacity, run by autoscts__params_gpu2).
Champion v36's frozen bytes (launch 83, thr_mlp848_ungated_triple_on_v34, autoscts__params_gpu6,
14,090,284 at val_bpb 1.049182411092023, bare margin 0.000817588907977 = 5.019x the same-program
spread 0.0001628969454063) with TWO literals, one of which costs nothing:

    SCALAR_LR 0.5 -> 1.0
        ZERO parameters. autoscts__params_gpu5's launch 86 set exactly this literal on exactly these
        bytes and measured val_bpb 1.0485213133031897: -0.0006610977888333 = 4.058 spreads, with
        peak_vram_bytes, flops_per_token_measured and step0_loss all bit-identical.
    MLP_NARROW3_LAYERS (0, 2) at MLP_NARROW3_ELEMENTS 16
        the two UNGATED layers {0,2} 848 -> 832; layer 4 stays 848, gated {1,3,5} stay 880
        widths [832, 880, 832, 880, 848, 880];  -32,768  ->  num_params_total = 14,057,516

THE SIZING IS THE POINT OF THIS ROW, AND IT IS A REFUSAL. A zero-parameter lever ties its base's
num_params_total, and a tie cannot promote under a strict comparison, so launch 86's 4.058 spreads can
only reach the ledger inside a candidate that also cuts. knowledge/a-lever-may-only-be-upside-never-
collateral.md (gpu6's) says which side of such a candidate carries the risk: size the cut so the BARE
margin covers it with the lever priced at exactly zero, and the lever becomes upside instead of a
shared point of failure.

    32,768 at 1.803539e-08  (launch 84, measured at base-min 848)  = +0.0005910  cover 1.38x, 1.39 sp
    32,768 at ~2.18e-08     (pessimistic post-step k=4 reading)    = +0.0007144  cover 1.14x, 0.63 sp

against the BARE 0.000817588907977, lever at zero. Cover interval [1.14x, 1.38x] EXCLUDES 1. With the
lever at the value launch 86 measured on these bytes the margin is 0.0014786866968103 and cover is
2.50x with 5.45 spreads left over.

WHAT I AM REFUSING, EXPLICITLY. knowledge/zero-parameter-income-pays-and-parameter-priced-income-does-
not.md (gpu5's, and the file that makes this row possible) recommends "size at -49,152 and stop" at
cover 1.67x WITH the lever. On the bare margin -49,152 is 0.0008865 against 0.000817588907977 = cover
0.92x, ineligible by 0.42 spreads if the lever delivers nothing -- and that is the exact shape this
champion's own financing_correction records v36 itself falling into, registered as bare-financed at
1.49x-6.74x and measured by its own launch at 0.92x-1.11x, an interval containing 1. Declining 16,384
more parameters buys a row that cannot fail on the lever, which is worth more with 13 launches left.

THE RATE IS NOT TRANSPORTED, and k INDEXES THE PRE-STEP WIDTH. autoscts__params_gpu3's launch 84 took a
16-element step on an UNGATED layer at span 256 on a base whose thinnest layer was 848 -- this
champion's own thinness, layer class and span -- and measured 1.803539e-08/param. And launch 83 stepped
its base's OWN thinnest layers 864 -> 848 at base-min 864 and measured 1.359367e-08 = 2.00x r1, the k=2
rate rather than the k=3 rate its post-step width would imply; this step likewise takes the min down, so
it prices at k=3. The k=4 reading is carried above as the pessimistic end regardless.

FOUR BRANCHES. val_bpb < 1.0491123 means the lever transported whole AND the width rate is at or below
launch 84's -- KEEP, and the next row can be sized on a re-measured margin. 1.0491123 .. 1.0497734 is
partial transport -- KEEP either way, and the launch returns the first SCALAR_LR transport factor across
a width step. 1.0497734 .. 1.05 means the lever delivered about nothing -- STILL A KEEP, which is what
sizing on the bare margin bought. val_bpb >= 1.05 is INELIGIBLE and requires the width rate to exceed
2.4951e-08 = 1.383x launch 84's with the lever at zero, or 4.512e-08 = 2.50x with the lever whole;
registered as the refuting outcome.

CREDIT. The SCALAR_LR axis is autoscts__params_analyst3's (proposal 4582d135); its measurement, the
zero-transport argument and the composition table are autoscts__params_gpu5's launch 86. The width rate
at base-min 848, the rate ~= 6.8e-09 x k formula and the MLP_NARROW3 spelling are
autoscts__params_gpu3's launch 84. The per-layer width mechanism, the ungated/gated split, the 32-byte
alignment class and the cover-plus-leftover test are autoscts__params_gpu1's launches 59/67/70/73. The
lever-as-upside rule and the convexity reading are autoscts__params_gpu6's. Mine is the sizing refusal.

Experiment thr_mlp848_ungated_triple_on_v34 (team throughput, run by autoscts__params_gpu6).
Champion v34's frozen bytes (launch 80, thr_rope100k_mlp864_l4_on_v32, mine, 14,139,436 at val_bpb
1.0485142548118378, margin 0.0014857451882 = 9.12x the same-program spread) with ONE literal:

    MLP_NARROW2_ELEMENTS 16 -> 32
        a SECOND rung on the same three UNGATED layers {0,2,4}: hidden 880 - 32 = 848
        widths [848, 880, 848, 880, 848, 880];  -49,152  ->  num_params_total = 14,090,284

THE POINT OF THIS ROW IS THE SECOND-RUNG RATE, AND IT IS ALREADY VISIBLE IN TWO MEASURED LAUNCHES.
Launches 80 (mine) and 81 (autoscts__params_gpu3's red_rope100k_mlp848_pair_on_v32) share base v32 AND
the same rotary base=100000 lever, and differ only in HOW they spend MLP width: I took a FIRST rung on
one ungated layer (-16,384, layer 4 880->864), they took a SECOND rung on the ungated pair (-32,768,
{0,2} 864->848). Two equations, and launch 78 supplies the first-rung rate independently:

    L80:  16,384 * r1 - lever*f = -0.0010562508   with r1 = 6.813e-09 (launch 78, my own)
          -> lever*f = 0.0011678750, i.e. f = 0.8988 -- the rotary transport factor
    L81:  32,768 * r2 - lever*f = -0.0007015606
          -> r2 = 1.4231e-08/param = 2.09x r1

THE MLP WIDTH AXIS IS CONVEX. No single launch in this run could show that, my own v34 pricing did not
know it, and it is the reason this candidate is priced on r2 rather than on the r1 I measured myself.
The crude marginal (L81-L80)/16,384 = 2.1649e-08 is carried as the pessimistic end because it conflates
the rung with the distribution; the decomposed 1.4231e-08 is the better estimate.

    49,152 at r2 1.4231e-08        = +0.0006995   cover 2.12x, leftover 4.83 spreads
    49,152 at crude 2.1649e-08     = +0.0010641   cover 1.40x, leftover 2.59 spreads

against v34's margin 0.0014857451882. FINANCED UNDER BOTH READINGS, and under the dearest the leftover
still clears the instrument by 2.59 spreads. That leftover test is autoscts__params_gpu1's, from the
row where they rejected -49,152 at v32's margin for leaving only 0.33 spreads -- CORRECTLY, at that
margin. Launch 80 tripled the margin to 9.12 spreads. This is the same cut re-priced on the new
frontier, which is exactly the Step 3a re-pricing their own row asks the next claimer to do.

FOUR BRANCHES. val_bpb <= 1.0492137 means r2 <= 1.4231e-08 and the axis is cheaper than the
decomposition says -- KEEP with >= 4.83 spreads left and a THIRD rung becomes buyable. 1.0492137 ..
1.0495783 brackets r2 between the decomposed and crude readings -- KEEP, and the width axis is
priced for the rest of the run. 1.0495783 .. 1.05 means r2 > 2.1649e-08, dearer than ANY reading in
this run -- STILL A KEEP, and the width axis closes here. val_bpb >= 1.05 is INELIGIBLE and means
r2 > 3.022e-08, 4.4x the first rung, i.e. convexity far sharper than two rungs can extrapolate;
registered as the refuting outcome.

THE TARGET INTEGER 14,090,284 IS THE ONE RELEASED AFTER LAUNCH 79 and this row takes it with
byte-distinct source and a fresh exp_id, as that release requires. Launch 79 (gpu3's
red_rope_bank_100k_mlp864_on_v31, uniform 864 on all six from v31, -98,304) was CHARGED and abandoned
after a preemption with no result row and no metrics, and its bytes are permanently reserved. This
candidate reaches the same integer from v34 by one literal rather than from v31 by two, so it is
byte-distinct by construction. Its predicted flops_per_token_analytic 90,833,088 and measured
114,426,432 are EXACTLY what launch 79 printed before it died -- an independent confirmation of this
census from a launch that shared the target and nothing else.

CREDIT. The second-rung rate this row is priced on comes out of autoscts__params_gpu3's launch 81 as
much as my launch 80; neither launch alone separates the rungs. The leftover-clears-the-instrument
test is autoscts__params_gpu1's. The per-layer width mechanism, the ungated/gated split and the
alignment-class argument are autoscts__params_gpu1's launches 59/67/70/73. The rotary lever inside the
base is autoscts__params_gpu3's launch 77, banked by my launch 80 and NOT re-sold here.

Experiment thr_rope100k_mlp864_l4_on_v32 (team throughput, run by autoscts__params_gpu6).
Champion v32's frozen bytes (launch 78, thr_mlp864_pair_on_v31, mine, 14,155,820 at val_bpb
1.049570505647074, margin 0.0004294943529 = 2.64x the same-program spread) with TWO literals:

    MLP_NARROW2_LAYERS (0, 2) -> (0, 2, 4)
        the same SECOND narrowing step, extended to the THIRD ungated layer: layer 4 hidden 880 -> 864
        widths [864, 880, 864, 880, 864, 880];  -16,384  ->  num_params_total = 14,139,436
    _precompute_rotary_embeddings(..., base=10000 -> 100000)
        launch 77's MEASURED zero-parameter lever, carried onto the frontier. Both call sites
        (__init__ and init_weights) take the default, so this one literal moves both.

THE CUT IS FINANCED WITHOUT THE LEVER, AND THAT IS THE WHOLE DESIGN. This is the shape that launches 69
and 73 -- and my own withdrawn thr_longwin_quarter_gate_mlp864_on_v30 -- got wrong by letting a lever's
income pay for a cut. Here it does not. Priced on the rate MY OWN launch 78 measured, on THIS mechanism,
on THIS layer class ({0,2} ungated, 880 -> 864), on the immediately preceding base:

    16,384 at launch 78's net 6.813e-09/param  =  +0.0001116  against 0.0004294943529 of margin
                                                  cover 3.85x, and 1.95x the spread LEFT OVER

so the candidate is eligible if the rotary lever delivers EXACTLY NOTHING. Launch 78 is the only launch
in this run that has priced a width step on this lineage at span 2048, and layer 4 is ungated exactly as
{0,2} are (has_ve(i,6) is i%2==1), so no rate is transported across a span, a base or a layer class. That
is the one thing the three attempts before this one could not say.

WHY THE LEVER RIDES ALONG RATHER THAN WAITING FOR ITS OWN LAUNCH. Launch 77 (red_rope_base_100k_on_l70,
gpu3) measured base 10000 -> 100000 at val_bpb -0.0012993172 for ZERO parameters on v30's bytes -- 3.0x
this champion's entire margin, the largest zero-parameter gain in the run. It cannot bank itself: at zero
parameters it ties its base's integer and a tie can never promote under a strict comparison. So the lever
reaches the ledger only inside a candidate that also cuts, and three such candidates have now failed to
report for reasons that are not about their mechanisms (launch 79 unresolved; gpu2's and gpu5's requests
refused by a stopped allocation). The measurement is the run's most-wanted number by revealed preference
of three independent teams, and this row gets it without paying for it.

FOUR BRANCHES, THREE OF WHICH ARE KEEPS, AND THE RESIDUAL IS THE LEVER'S TRANSPORT. Subtracting the cut's
own measured cost from the realised val_bpb leaves the lever's transport factor f onto v32:

    val_bpb <= 1.0487401   f >= 0.725  the lever HOLDS on the frontier (0.725 is this run's own measured
                                       transport floor). KEEP, margin >= 7.73x the spread, and the board
                                       reopens for ~190,000 parameters of further width
    1.0487401 .. 1.0495705  0 < f < 0.725   partial transport; the lever decays on the thinner base the
                                       way the long_window lever did between v29 and v30. KEEP
    1.0495705 .. 1.05      f <= 0        the lever delivers nothing or costs. STILL A KEEP, on the cut
                                       alone, and the rotary axis stops being a financing instrument
    val_bpb >= 1.05        f <= -0.245   INELIGIBLE. The lever has REVERSED SIGN by 9.9 spreads across a
                                       0.23% width step. Nothing in this run has reversed by more than
                                       2.9 spreads (my long_window lever, v29 -> v30), so this branch is
                                       registered as the refuting outcome, not as a risk that was priced

CREDIT. The rotary axis, both its measured rungs and the banking idea are autoscts__params_gpu3
(launches 74, 77, 79). The target integer 14,139,436 and the {0,2,4} layer set are
autoscts__params_gpu1's, released unclaimed in knowledge/axis_claims.md ("UNCLAIMED -- I have spent my
two launches. First claimer adds the row here"); autoscts__params_gpu2 then claimed it, had its request
refused by the stopped allocation, and handed it back explicitly ("I am handing it back unclaimed ...
whoever takes it must write byte-distinct bytes"). These bytes are built on v32 rather than v31 and
narrow one layer rather than three, so they are byte-distinct from that reserved program by construction.
I declined this integer one cycle ago on a cover interval whose dear end was my own reading of launch 73;
launch 78 refuted that reading, and this candidate is the correction.

Experiment thr_mlp864_pair_on_v31 (team throughput, run by autoscts__params_gpu6).
Champion v31's frozen bytes (launch 75, thr_ve_gate_share_on_v30, mine, 14,188,588 at val_bpb
1.0493472473121355, margin 0.0006527526878646 = 4.01x the same-program spread) with ONE mechanism:

    MLP_NARROW2_LAYERS (0, 2) at MLP_NARROW2_ELEMENTS 16
        a SECOND narrowing step on the two UNGATED layers {0,2}: hidden 880 -> 864
        widths [864, 880, 864, 880, 880, 880];  -32,768  ->  num_params_total = 14,155,820
        NO span change, no share change, nothing else.

THIS LAUNCH EXISTS TO SETTLE A RATE THAT TWO AGENTS NOW READ DIFFERENTLY BY 1.62x, AND IT PROMOTES
WHILE DOING IT. Launch 73 (cap_win_long_half_mlp864_even_on_l70, gpu1) came in INELIGIBLE and its
residual admits two incompatible explanations, both published from this run:

  (A) autoscts__params_gpu1, post 4c250d18 / knowledge/a-span-cut-and-a-width-cut-are-one-resource.md:
      launches 72 and 73 differ ONLY in MLP width (65,536 parameters), at identical spans, identical
      num_steps 2650 and identical total_tokens to the byte, so width prices at 1.567775e-08/param at
      long_window 1024 against 9.70027e-09 blended at 2048 -- 1.6162x, with NO income term anywhere.
      Their reading: a span cut and a width cut are ONE resource, and the premium applies only when a
      span limb is in the same candidate. Width on a 2048-span base is still launch 67's 7.65679e-09.

  (B) mine, knowledge/a-span-rate-does-not-survive-a-width-step.md: the same three launches are equally
      consistent with the WIDTH rate being dearer on the narrower base whatever the span does, which
      would make launch 67's rate stale rather than span-conditional.

The three measured launches cannot separate these, because none of them is a width step on this base at
span 2048. THIS ONE IS. It changes width and nothing else, on a base whose spans
[256,256,256,2048,256,2048] are exactly where launches 67 and 70 measured width.

    measured rate ~= 7.7e-09   ->  (A) is right, the premium is span-conditional, and MY knowledge file
                                    has the wrong mechanism (its practical rule survives either way)
    measured rate ~= 1.57e-08  ->  (B) is right, launch 67's rate is stale, and gpu1's advertised
                                    "THE BUY" of 49,152 at 14,139,436 would have been INELIGIBLE
                                    (0.000771 against 0.00065275 of margin)

WHY 32,768 AND NOT 16,384, AND NOT gpu1's 49,152. Sized so the candidate is financed under EVERY live
reading of the rate AND so the answer is resolvable against the instrument:

    mass     (A) L67 net   L67 gross   L70 gated   (B) L72/L73    cover interval
    16,384    0.000097     0.000125     0.000170     0.000257     [2.54x, 6.73x]  -- but the DELTA is
                                                                   0.60x the spread under (A): the rate
                                                                   would be unpublishable by this run's
                                                                   own rule, so it cannot settle anything
    32,768    0.000194     0.000251     0.000340     0.000514     [1.27x, 3.36x]  -- financed under all
                                                                   four, and (A) vs (B) are separated by
                                                                   TWO spreads.  ** THIS ROW **
    49,152    0.000291     0.000376     0.000510     0.000771     [0.85x, 2.24x]  CONTAINS 1 -- refused

gpu1 advertised 49,152 as "THE BUY" and asked for a claimer. I am not taking it: its cover interval
contains 1 precisely because it is priced on the disputed rate, so it would be decided by the number the
launch exists to find -- the error launch 69 paid for and the one that took my own paired candidate down
two hours ago. 32,768 is their own stated principle ("two bites with a measurement between them dominate
one bite") applied one granule finer, because here it is the RATE and not just the base that is unknown.

WHY {0,2}. Both are UNGATED: has_ve(i,6) is i%2==1, and launch 70 measured {1,3,5} at 1.038160e-08
against launch 67's 7.65679e-09 on {0,2,4} -- 1.356x dearer. There is now a second reason to spare
{1,3,5} that gpu1 identified and I am recording because it is theirs: launch 70's gated rate was measured
with THREE separate gate maps, and champion v31 has ONE shared gate function serving all three, so a
{1,3,5} limb would carry an unmeasured width-against-shared-gate interaction on top of a known premium.
{0,2} touches neither. 864 = 16 x 54 and 864 x 2 = 1,728 bytes = 32 x 54, so both widths stay 32-BYTE
aligned, the free class (launch 51 -4.258% at 8-byte, launch 54 -0.992% at 16-byte, launch 52 -0.041%).

CREDIT. The per-layer width mechanism, the ungated/gated split and reading (A) are autoscts__params_gpu1
(launches 59, 67, 70, 73 and post 4c250d18); the target integer 14,155,820 is one I claimed and stood
down this cycle under a different mechanism, re-claimed here with the span limb removed.

Experiment thr_ve_gate_share_on_v30 (team throughput, run by autoscts__params_gpu6).
Champion v30's frozen bytes (launch 70, cap_mlp_gated880_on_l68, gpu1, 14,188,652 at val_bpb
1.0499233471485316, margin 0.0000766528514684 = 0.471x the same-program spread, the narrowest of the
run) with TWO statements and no other change:

    self.transformer.h[3].attn.ve_gate = self.transformer.h[1].attn.ve_gate
    self.transformer.h[5].attn.ve_gate = self.transformer.h[1].attn.ve_gate
        one ve_gate FUNCTION serves all three has_ve layers, 3 distinct gate maps -> 1
        -64 parameters -> num_params_total = 14,188,588   (-71.812% vs the 50,332,176 reference)

Mechanism transplanted verbatim from launch 71 (red_ve_gate_share_135_on_v29, autoscts__params_gpu4,
team redundancy). The integer 14,188,588 was priced and explicitly RELEASED unclaimed by
autoscts__params_gpu1 in knowledge/axis_claims.md: "launch 71 measured that share FREE, so this is
promotable with no lever at all".

WHY THIS AND NOTHING ELSE: LAUNCH 73 REFUTED EVERY OTHER PURCHASE ON THIS BOARD, INCLUDING MINE.
I had built, verified and claimed thr_longwin_quarter_gate_mlp864_on_v30 (14,155,820: long_window // 4
plus this same gate share plus a 32,768 MLP cut on the ungated pair), financed on launch 72's measured
+0.00029343 for long_window 2048 -> 1024. Launch 73 (cap_win_long_half_mlp864_even_on_l70, gpu1) landed
while I was verifying and measured that exact lever ON THESE BYTES:

    launch 73  = v30 + long_window // 2 + MLP{0} 880 -> 864     14,172,268   val_bpb 1.0502271061503041
                                                                             INELIGIBLE by +0.0002271

Decomposed against v30 (2,626 steps) at the run's standing token-income rate 0.00041940 per 1%:

    delta val_bpb                    +0.00030376  (a COST)
    steps 2,626 -> 2,650 = +0.9139%  income +0.00038331
    total quality term               +0.00068706
    less MLP{0} 16,384 at L67's 7.65679e-09         -0.00012545
    = long-span quality term at 1024 ON v30        **+0.00056162**

against the SAME rung's **+0.00017062** on v29 (launch 72). Both limbs of the lever degraded on the
thinner base: the income elasticity 0.06497 -> 0.05366 (0.826x) and the span-quality term by **3.29x**,
so the lever's NET went **+0.00029343 -> -0.00017831, a sign reversal**. My candidate carried that same
lever one rung DEEPER plus twice the cut, so it is refused under the span reading (predicted 1.0502306,
ineligible by 0.00023) AND under the alternative reading that the per-layer MLP rate on this lineage is
really 4.1935e-08/param (5.48x launch 67's, which would put 32,768 at cover 0.06x). Two decompositions,
one conclusion: I withdrew it unsubmitted rather than refine it.

WHAT SURVIVES IS THE ONE LIMB NEITHER READING TOUCHES. This candidate changes no span and no width, so
neither the reversed span term nor the disputed width rate can reach it. Launch 71 measured this exact
mechanism, one champion back, at val_bpb **-0.0001219240431327** -- a GAIN at 0.749x the same-program
spread, i.e. a draw -- with num_steps +2, flops_per_token_measured bit-identical (the gate is charged
per USE and is still used by three layers), and step-0 AND step-1 loss bit-identical. Its axis is closed
in both directions: launch 50 took gate WIDTH 32 -> 8 for a gain, cap_ve_gate_off_on_v21 deleted the
gate for +0.0023857, and launch 71 took gate LAYER COUNT 3 -> 1 for a draw, so only the gate's PRESENCE
is load-bearing and every gate here stays fed at ve_gate_channels 8.

    budget    0.0000766528514684   (v30's own margin)
    cut cost  0.000000             (64 parameters at a mechanism measured FREE; the -0.000122 is
                                    registered as UPSIDE, not spent)

This is the only promotable purchase left on the board that does not have to borrow a number the same
launch is trying to find. It is 64 parameters, and it is also the launch that makes the NEXT one
affordable: at the measured -0.000122 the champion's margin goes 0.471x -> **1.22x** the instrument
spread, which is the first time the 16,384 MLP granule clears (cover 1.58x). I take the safest
affordable rung and leave the rest of the margin for whoever claims next.

REGISTERED SECOND MEASUREMENT. Launch 71 is n=1 on a base one width-step wider. This is the same
mechanism on the narrowed lineage, which is exactly the transport that just failed for the span lever --
so the result is a real test either way: if the share is free here too, "free" is a property of the
mechanism; if it is not, then knowledge/val-bpb-deltas-are-amplified-on-the-narrowed-lineage.md governs
64-parameter auxiliary paths as well as spans, and that is worth knowing before anyone prices another
transported rate.

CREDIT. The mechanism and its measurement are autoscts__params_gpu4 (launch 71). The integer was priced
and released by autoscts__params_gpu1, whose launch 73 is also what refuted my own row. The lever
decomposition above rests on autoscts__params_gpu5's launches 61, 66 and 72.

Experiment cap_mlp_gated880_on_l68 (team capacity, run by autoscts__params_gpu1).
Launch 68's own frozen bytes (gpu4's red_attn_share_cproj_24_on_v27, the ledger frontier at
14,237,804) with ONE literal moved:

    MLP_NARROW_LAYERS  (0, 2, 4)  ->  (0, 1, 2, 3, 4, 5)
        MLP hidden on layers {1,3,5}: 896 -> 880. {0,2,4} are already 880, so all six become 880.
        -49,152 parameters -> num_params_total = 14,188,652   (-71.810% vs the reference)

Nothing else moves, so this and launch 68 are a two-point bracket differing only in whether the three
has_ve layers are narrowed one step.

WHY: THIS RUN HAS STEERED FOUR LAUNCHES ON AN ASSUMPTION IT NEVER MEASURED. Launch 67 (mine) measured
per-layer MLP width on the UNGATED set {0,2,4} at 7.6568e-09 per parameter across 880 -> 832, in a clean
two-point bracket, and refuted the per-layer discount I had published: width prices flat per parameter
however it is distributed, and there is no concentration penalty. What launch 67 could not see is
whether the three has_ve layers cost the same. Launches 59, 66, 67 and launch 68's own base have ALL
narrowed {0,2,4} and spared {1,3,5}, and the stated reason has always been that {1,3,5} own a ve_gate
and read the value residual, i.e. that they are the expensive layers -- see the MLP_NARROW_LAYERS
comment this candidate rewrites, which asserts exactly that. **It has never been measured.**

THE GATE IS THE DISCRIMINATOR, WHICH IS THE CHEAPEST WAY TO ASK. Launch 68 holds
0.000506921572861474 of margin. Against 49,152 parameters that is a break-even rate of
1.03133e-08/param = 1.347x launch 67's measured ungated rate:

    gated == ungated (launch 67, 7.6568e-9)   +0.00037635   1.0498694   ELIGIBLE, cover 1.347x
    gated == launch 52 uniform (7.0047e-9)    +0.00034430   1.0498374   ELIGIBLE, cover 1.472x
    gated 1.35x dearer (1.0313e-8)            +0.00050692   1.0500000   exactly break-even
    gated 1.5x dearer  (1.1485e-8)            +0.00056451   1.0500576   INELIGIBLE by 0.0000576

DECISION RULE, pre-registered:
    realised < 1.031e-8    the gated layers are NOT meaningfully dearer, the four-launch assumption was
                           unfounded, and MLP width is ONE uniform resource of 6 x 512 x 2 per element
                           whose whole remaining pool prices at ~7.7e-9.
    >= 1.031e-8            the gated layers ARE dearer by at least 1.35x: the assumption is vindicated
                           and finally quantified, and every future width cut should target {0,2,4},
                           whose remaining depth to 768 is 3 x 112 x 1024 = 344,064 parameters.

WHY NOT THE ALTERNATIVES, ALL PRICED FIRST. A zero-parameter margin lever is worth more than a cut on a
board this margin-bound, but the two untested ones are WARMUP_RATIO and FINAL_LR_FRAC and this run
already owns the control: launches 40/43 measured a MORE aggressive head gaining 0.023611 of train loss
by 11.9% and a LESS aggressive one losing 0.044436 and repaying only 91%. Both knobs remove
learning-rate mass from the head, i.e. both bet the head has slack, and that pair says it does not --
predicted to lose, and I will not spend a launch confirming a control we have. A second attention share
is the cheapest mass in the run now that gpu4's launch 68 prices c_proj at 5.5896e-09/param, below MLP
width, but its granule is 262,144 = 0.001465 against 0.000507: cover 0.35x. The 128-tile crossing
{0,2,4} 880 -> 768 is the rung I most want, because
knowledge/mlp-hidden-refund-is-quantised-to-128.md measures a crossing converting at 0.70 of its counted
refund against 0.10-0.23 for a non-crossing rung -- but it is 344,064 parameters = 0.002635, cover
0.19x, and its counted refund is only 1.79% of this model's FLOPs so the income cannot finance it. Both
are queued, not claimed.

PRE-REGISTERED, every line a falsifier:
    num_params_total               14,188,652  EXACT. mlp 6 x 1024 x 880 = 5,406,720; attn unique
                                   17 x 262,144 = 4,456,448 (24 slots less 4 re-used c_q, 1 re-used
                                   c_v, 1 re-used c_k, 1 re-used c_proj); ve_gates 3 x 4 x 8 = 96;
                                   wte 4,194,304; lm_head 512 x 256 = 131,072; scalars 12.
    flops_per_token_measured       115,016,256  EXACT = launch 68's 115,311,168 - 6 x 49,152.
    flops_per_token_analytic       91,423,296   EXACT = 6 x 9,994,336 + 5,120 x 6,144
    analytic_measured_gap          23,592,960   EXACT and unchanged from launch 68: 6 x 7 x 262,144 for
                                   the seven re-used module uses plus 12,582,912 for the wte column
                                   slice. Sharing a module never moves use-weighted mass.
    training_data_tokens_available 631,241,817  EXACT
    step0_loss                     9.010914     EXACT (lm_head zero-init, ln 8192)
    num_steps                      2,616..2,634, centre 2,623 (launch 68 read 2,621). No 128-tile
                                   crosses -- 880 and 896 both need seven -- so the 0.2557% counted
                                   refund returns at the measured non-crossing conversion of 0.10-0.23,
                                   i.e. +0.7 to +1.5 steps, and the four MLP Muon shape groups collapse
                                   back to two, worth about the 0.04% launch 59 measured.
    total_tokens                   num_steps x 262,144
    peak_vram_bytes                28.16e9..28.24e9 (launch 68 read 28,259,537,920, less TWO
                                   hidden-width activations on three layers at the frozen 128 x 2048 =
                                   3 x 2 x 128 x 2048 x 16 x 2 = 50,331,648 -- the multiplier launch
                                   67's band miss corrected from one to two)
    val_bpb                        1.0498694 centre; band 1.04983..1.05006 across the four hypotheses
    second falsifier               num_steps <= 2,608 breaks eligibility on its own at the centre rate

WHAT IS NOT CHANGED. MLP_RATIO 1.75, MLP_NARROW_ELEMENTS 16, the attn.c_proj share h[4] <- h[2], the
c_q share h[1:-1], c_v h[3] <- h[1], c_k h[5] <- h[3], short_window = long_window // 8 (spans
[256,256,256,2048,256,2048]), LOGIT_RANK 256, ve_gate_channels 8, has_ve {1,3,5} -- the gates
themselves are untouched and every one is still fed; only the MLP width of those layers moves.
WARMDOWN_RATIO 0.7, WARMUP_RATIO 0.0, FINAL_LR_FRAC 0.0, every LR, ADAM_BETAS, ns_steps 5,
TOTAL_BATCH_SIZE 2**18, DEVICE_BATCH_SIZE 128, DEPTH 6, ASPECT_RATIO 85, HEAD_DIM 128, n_kv_head 4,
seed 42. init_weights is width-agnostic (s = 3**0.5 * n_embd**-0.5). setup_optimizer's partition assert
counts tensors, not shapes, and holds. Muon groups are keyed on shape, so the MLP's four groups become
two -- (880,512) and (512,880) -- with no group left holding tensors that no longer exist, and the
per-shape LR factor max(1, shape[-2]/shape[-1])**0.5 becomes 1.31122 for the single c_fc group, which
is exactly what launch 52 ran at uniform 880. NorMuon's red_dim branch (shape[-2] >= shape[-1]) is
unchanged. The hidden % 16 == 0 assert holds (880 = 16 x 55) and 880 x 2 = 1,760 bytes = 32 x 55 is in
the free alignment class launch 52 measured at -0.041%. The three frozen regions -- the
"Parameter counts:" print block, the single measure_flops_dispatch call with its unmodified returned
integer, and the single report_efficiency_metrics call -- are byte-identical, and prepare.py,
pyproject.toml and uv.lock are the champion's bytes.

--- launch 68's own record follows, unedited ---

Experiment red_attn_share_cproj_24_on_v27 (team redundancy, run by autoscts__params_gpu4).
Champion v27's own frozen bytes (launch 66, autoscts__params_gpu6). ONE statement, ONE cut, and for
the first time in this run a 262,144 granule that the live margin covers:

    self.transformer.h[4].attn.c_proj = self.transformer.h[2].attn.c_proj
        one attn.c_proj serves layers 2 and 4, 6 distinct attention output projections -> 5,
        -262,144 -> 14,237,804

WHY NOW, AND NOT TWO LAUNCHES AGO. Every ledger walk since cycle 6 said the affordable set excluded
every 262,144 granule and that the run's one open question was an INCOME question, not a cut question.
Launch 66 answered it: the margin went 0.0004234763181 -> 0.0020041800894774 (4.73x) and this granule
is inside it for the first time. The row is autoscts__params_analyst2's (post c30e9317), written
against champion v20 where it would land at 14,811,244 -- 311,296 ABOVE the live champion and
unpromotable. Only the base moves; the mechanism and the pair are theirs unchanged.

THE PRICE, ON THE THREE RATES THAT SHARE THIS MECHANISM'S CLASS. A FIRST merge of an ATTENTION role,
both measured on this lineage inside the last nine launches, plus the same PAIR in a different role:

    launch 57   attn.c_v   pair {1,3}    262,144    0.004208 / 1M   ->  0.001103    cover 1.82x
    launch 64   attn.c_k   pair {3,5}    262,144    0.007030 / 1M   ->  0.001843    cover 1.09x
    launch 38   mlp.c_proj pair {2,4}    491,520    0.007444 / 1M   ->  0.001952    cover 1.03x

x1.035 for convexity (1.059x per 6.5% of base thinness; v27 is 3.8% thinner than launch 57's base)
gives cover 1.75x .. 1.05x .. 0.99x. THREE OF THREE MEASURED RATES ARE INSIDE THE MARGIN and only the
same-pair-different-role rate is marginal, which is exactly the comparison this candidate is built to
make.

WHAT IS ACTUALLY UNKNOWN. attn.c_proj is the third and last unpriced attention role. autoscts__params_analyst3
objected on c30e9317 that layers 2 and 4 already share c_q (h[1:-1]), so this stacks two merges on one
pair. That objection is ANSWERED BY THE CONTROL rather than waived: launch 38 shared mlp.c_proj on this
exact pair on a base that already carried the same h[1:-1] c_q share (v16 is downstream of launch 30),
so the pair, the layer indices and the c_q stacking are all held FIXED and the ONLY moving part is
mlp.c_proj -> attn.c_proj. If the measured rate comes in below 0.007444/1M the role term is real and
ordered c_v < c_proj; if it comes in above, the double merge on {2,4} is the reason and the follow-up
is a c_proj pair containing layer 5, which does not share c_q.

PRE-REGISTERED, every line a falsifier:
    num_params_total            14,237,804   EXACT = 14,499,948 - 262,144. Attention unique tensors
                                             18 -> 17 x 262,144 = 4,456,448; mlp 5,455,872; ve_gates
                                             96; wte 4,194,304; lm_head 131,072; scalars 12.
    census                      wte 4,194,304 / value_embeds 0 / lm_head 131,072 /
                                transformer_matrices 9,912,416 / scalars 12, NO UNACCOUNTED row
    flops_per_token_measured   115,311,168   EXACT, UNCHANGED. The dispatch counter charges by USE and
                                             both matmuls survive; launches 23, 57 and 64 measured
                                             this property directly on the three other shares.
    flops_per_token_analytic    91,718,208   EXACT. estimate_flops charges 6*(nparams - excluded), so
                                             it DROPS by exactly 6 * 262,144 = 1,572,864.
    analytic-measured gap       23,592,960   = 22,020,096 + 1,572,864. The gap moved by exactly
                                             1,572,864 at each prior share: 18,874,368 (v21) ->
                                             20,447,232 (c_v) -> 22,020,096 (c_k).
    step-0 loss                   9.010914   BIT-IDENTICAL. lm_head is zeros_ so step-0 logits are
                                             exactly 0 and the loss is ln(8192); attn.c_proj is zeros_
                                             so an aliased module draws NO RNG and every other tensor
                                             is bit-identical. STEP 1 IS NOT REGISTERED AS IDENTICAL:
                                             the shared module receives the SUM of two gradients.
    training_data_tokens_available  631,241,817   EXACT
    num_steps                   band [2618, 2632], central 2624. Launch 50 -> 57 moved 2420 -> 2421
                                             (+1) for this same mechanism; the (512,512) Muon group
                                             goes from 18 members to 17, no new shape, no group added
                                             or removed, max(1.0, 512/512)**0.5 = 1.0 and red_dim = -1
                                             at both, so ZERO optimizer-behaviour change.
    total_tokens                band [686,292,992, 689,963,008], registered as the band INDUCED by the
                                             num_steps band and not at its centre.
    peak_vram_bytes             DOWN, band -1.0 to -5.0 MB. DIRECTION ONLY: the two measured
                                             precedents disagree in magnitude (launch 50 -> 57 gave
                                             -1,705,984; the c_k share gave -3,540,992), and no
                                             magnitude is priced from a retention calculation here.
    val_bpb                     central 1.0495208, band [1.049096, 1.050096]; cost band +0.0011 ..
                                             +0.0021. ELIGIBLE IFF cost < 0.0020041800894774.

WHAT IS NOT TOUCHED. WINDOW_PATTERN "SSSL" with short_window = long_window // 8 (spans
[256,256,256,2048,256,2048], span sum 5120), mlp_narrow_layers (0,2,4) with hidden 880/896,
MLP_RATIO 1.75, LOGIT_RANK 256, ve_gate_channels 8, WARMDOWN_RATIO 0.7, WARMUP_RATIO 0.0,
FINAL_LR_FRAC 0.0, EMBEDDING_LR 0.3, UNEMBEDDING_LR 0.004, MATRIX_LR 0.04, SCALAR_LR 0.5,
WEIGHT_DECAY 0.2, ADAM_BETAS (0.8, 0.95), ns_steps 5, muon beta2 0.95, TOTAL_BATCH_SIZE 2**18,
DEVICE_BATCH_SIZE 128, DEPTH 6, ASPECT_RATIO 85, HEAD_DIM 128 and DATA_BUDGET_TOKENS are exactly as
launch 66 measured them, and the three existing shares (c_q over h[1:-1], c_v on {1,3}, c_k on {3,5})
are untouched. setup_optimizer is untouched. prepare.py, pyproject.toml and uv.lock are bit-identical
to champion/, and the three frozen regions -- the "Parameter counts:" print block, the single
measure_flops_dispatch call with its unmodified returned integer, and the single
report_efficiency_metrics call -- are preserved verbatim.

Experiment thr_win_eighth_split880_on_v25 (team throughput, run by autoscts__params_gpu6).
Champion v25's own frozen bytes (launch 64, mine). ONE cut priced on money I already hold, and ONE
unmeasured rung of the run's largest lever:

    _compute_window_sizes:  short_window = long_window // 4  ->  long_window // 8
        spans [512,512,512,2048,512,2048] -> [256,256,256,2048,256,2048], span sum 6144 -> 5120
        ZERO parameters. UNMEASURED: launch 61 established span 512, nothing has gone below it.
    MLP hidden 896 -> 880 on layers {0,2,4} only, 896 kept on {1,3,5}
        -49,152 -> 14,499,948. Mechanism transplanted from launch 59 (gpu1), measured 0.002719/1M.

WHY THE CUT IS SIZED AT 49,152 AND NOT AT WHAT THE LEVER MIGHT PAY. Champion v25 holds
0.0004234763181 of margin. At launch 59's measured split rate this cut costs 0.0001336 (cover 3.17x);
at the UNIFORM rate launch 52 measured for the same width step (7.005e-9/param) it costs 0.0003443
(cover 1.23x). **The interval does not contain 1 on either measured rate, so the cut is affordable on
money already in hand and the lever below is upside, not a dependency.** The two-step version of the
same mechanism ({0,2,4} to 864, -98,304) has cover 1.58x..0.61x, which does contain 1, and I am not
taking it. Sizing a cut at what an UNMEASURED lever might pay is how launch 62 lost its launch.

WHY THE LEVER RUNG, AND WHY IT IS WORTH A LAUNCH EVEN THOUGH IT CANNOT KEEP ALONE. Launch 61 (gpu5)
measured `short_window = long_window // 4` at -0.0021326 of val_bpb for ZERO parameters -- 15.7x the
margin the champion held at the time -- and all of it was accounted for by +5.0847% of tokens at the
run's independently measured income rate of 0.00041940 per 1%, which means the span-quality term AT 512
IS INDISTINGUISHABLE FROM ZERO. Two launches have now banked it (63 and my 64). Nothing in this run has
gone below 512, and the whole board's affordability depends on whether that lever has another rung:
after this candidate the smallest structural granule left is 262,144 (attention c_proj) at a measured
7.03-7.44e-9 per parameter = 0.00184-0.00195, which NO live margin covers. So "does the window lever
continue below 512" decides whether the remaining launches can buy another 262,144 granule at all.

THE INCOME ESTIMATE, ON THE ELASTICITY GPU5's OWN LAUNCH CORRECTED. Counted FLOPs charge attention as
12*n_head*head_dim*min(window, T), linear in the window, and gpu5 showed that mis-prices a window by
2.5x; the right basis is the causal-average work mean_i min(i+1, w):

    w      per-layer mean context      4 short + 2 long (T=2048)     vs previous
    1024   768.25                      5,122.00
     512   448.13                      3,841.50                      -25.0%   -> +5.0847% steps (L61)
     256   240.06                      3,009.25                      -21.7%   -> +4.4% at L61's ratio

Launch 61's realised ratio was 0.203 steps% per work%, and gpu5 recorded it as a LOWER bound because
FA3 runs cheaper per counted FLOP than the matmuls do. But attention's share of the step also shrinks
as the window shrinks -- it was ~20.8% of 250 ms at w=1024, so ~16.5% of 237 ms now -- and -21.7% of
that is -8.5 ms, i.e. +3.7% of steps. **I register +3.7% as the centre and +4.4% as the optimistic end,
and the honest statement is that this is the third point on a curve whose first inversion was wrong by
3.84x in the good direction.**

WHAT IS ACTUALLY UNKNOWN AND WHY IT IS THE WHOLE EXPERIMENT: truncation. At span 256 the four S layers
see 1/8 of the sequence; only layers 3 and 5 keep the full 2048. Launches 20/28 bounded the value of
span ABOVE 1024 at ~0 and launch 61 extended that to 512, but 256 is one octave further and this is
where a truncation term should first appear if it exists anywhere.

PRE-REGISTERED, every line a falsifier:
    num_params_total            14,499,948   EXACT. Widths [880,896,880,896,880,896]: mlp
                                             3*2*512*880 + 3*2*512*896 = 5,455,872; attn unique 18 x
                                             262,144 = 4,718,592 (24 tensors less 4 re-used c_q, 1
                                             re-used c_v, 1 re-used c_k); ve_gates 96; wte 4,194,304;
                                             lm_head 131,072; scalars 12.
    census                      wte 4,194,304 / value_embeds 0 / lm_head 131,072 /
                                transformer_matrices 10,174,560 / scalars 12, NO UNACCOUNTED row
    flops_per_token_measured   115,311,168   EXACT = 121,897,536 - 6*49,152 - 6,144*1,024. The width
                                             limb refunds use-weighted parameters, the window limb
                                             refunds 6,144 per unit of span sum (launch 61 measured
                                             that coefficient to the unit). 48.2% of the ceiling.
    flops_per_token_analytic    93,291,072   EXACT = 6*(14,499,948 - 4,194,304 - 12) + 6,144*5,120
    analytic/measured gap       22,020,096   UNCHANGED from v25: no aliasing is added or removed
    step-0 loss                   9.010914   EXACT. MLP init reads n_embd, not hidden, and c_proj plus
                                             lm_head are zeros-init.
    training_data_tokens_available  631,241,817   EXACT (equality constraint)
    num_steps                   2,590-2,680, centred 2,638 = v25's 2,544 x 1.037
    median_step_ms              ~228 (v25's 237 less the 8.5 ms the attention refund implies)
    total_tokens                678,952,960-702,545,920, i.e. 1.076-1.113 epochs. Repeats are allowed
                                and launch 61 measured the income rate holding past one epoch.
    peak_vram_bytes             < 28,312,034,304, direction only. A narrower hidden width removes
                                activations and parameters; launch 61 measured a window change as
                                BIT-IDENTICAL here, so the whole move is the width limb's.
    val_bpb                     1.0479 (income lands at +3.7%, cut at the measured split rate) ..
                                1.0499205 (income exactly ZERO, cut at the dear uniform rate).
                                ELIGIBLE across that whole interval -- which is the point of sizing
                                the cut on held margin -- so the gate is decided by TRUNCATION alone.

WHAT A MISS MEANS, stated before the measurement. val_bpb >= 1.05 with num_steps in band means the
truncation term at span 256 exceeds 0.0018 and the window axis is BRACKETED at 512: an interior
optimum, closed, and nobody should propose //16. num_steps below 2,590 with val_bpb still under 1.05
means FA3 stops refunding wall clock below 512 -- the axis is closed for a different reason, and the
candidate still promotes on the width limb. Both outcomes close an axis that currently reads as the
run's only source of new margin, and they are distinguishable by num_steps alone.

WHAT IS NOT CHANGED. MLP_RATIO itself stays 1.75, so {1,3,5} keep hidden 896. The c_q share stays
h[1:-1], the c_v share stays h[3]<-h[1], the c_k share stays h[5]<-h[3]. has_ve untouched,
ve_gate_channels 8, WINDOW_PATTERN "SSSL", LOGIT_RANK 256 (launch 62 measured 192 at 0.337e-6 per
parameter, 80x the dear share band -- that axis is closed), WARMDOWN_RATIO 0.7, WARMUP_RATIO 0.0,
FINAL_LR_FRAC 0.0, EMBEDDING_LR 0.3, UNEMBEDDING_LR 0.004, MATRIX_LR 0.04, SCALAR_LR 0.5,
WEIGHT_DECAY 0.2, ADAM_BETAS (0.8, 0.95), ns_steps 5, TOTAL_BATCH_SIZE 2**18, DEVICE_BATCH_SIZE 128,
DEPTH 6, ASPECT_RATIO 85, HEAD_DIM 128 and DATA_BUDGET_TOKENS are exactly as launch 64 measured them.
setup_optimizer is untouched; the Muon groups are keyed on shape, so the two narrowed (880,512) and
(512,880) matrices join their own group and no group is added or removed. prepare.py, pyproject.toml
and uv.lock are bit-identical to champion/, and the three frozen regions -- the "Parameter counts:"
print block, the single measure_flops_dispatch call with its unmodified returned integer, and the
single report_efficiency_metrics call -- are preserved verbatim.

Experiment thr_share_ck_35_win_quarter_on_v22 (team throughput, run by autoscts__params_gpu6).
Champion v22's own frozen bytes. TWO statements, one CUT and one MEASURED LEVER, and the lever is
here because the cut cannot be financed without it:

    self.transformer.h[5].attn.c_k = self.transformer.h[3].attn.c_k
        one attn.c_k serves layers 3 and 5, 6 distinct key maps -> 5, -262,144 -> 14,549,100
    _compute_window_sizes:  short_window = long_window // 2  ->  long_window // 4
        spans [1024,1024,1024,2048,1024,2048] -> [512,512,512,2048,512,2048], span sum 8192 -> 6144
        ZERO parameters. Measured at -0.0021325632 of val_bpb by launch 61, credit gpu5.

WHY TWO LIMBS AND NOT ONE. This row (`thr_share_ck_35_on_v22`, proposal 67a171f8) was queued by me
last rotation UNCLAIMED with its precondition written into the row: "needs 0.0009673 of margin ON TOP
of v22's 0.0001356489746 ... claim only after a lever lands". v22 is my own champion and I recorded
its margin as 0.833x the run's same-program val_bpb spread -- the affordable-cut set at v22 is EMPTY
by measurement, so a bare 262,144 cut on these bytes is not a coin flip, it is arithmetic against it.
Launch 61 landed the lever 45 minutes ago at 2.2x the required size, at zero parameters, and gpu5's
own result post says every future candidate should carry it. So the composite is not ambition: the
one-limb version of this candidate is unbuyable and the lever limb is the precondition being met.

WHAT THE PAIR {3,5} IS FOR, AND WHY IT IS THE ONLY BUYABLE c_k PAIR HERE. shared_c_q = h[0].attn.c_q
is assigned to h[1:-1], so layers 0-4 hold ONE c_q and layer 5 owns its own. An attention score is
score_ij = x_i (W_q W_k^T) x_j^T, so any c_k pair drawn from {0,1,2,3,4} shares BOTH factors and
gives two layers a bit-identical QK map. That is a function-class restriction, whose own price band
this run measured at 0.0238-0.0241 per 1M (launch 27), and it is 5.7x the entire budget here. A pair
containing layer 5 keeps two distinct bilinear forms over one shared key basis. Of the five such
pairs I take {3,5} because both are has_ve layers, which holds the value-residual property FIXED
against launch 57's c_v merge on {1,3} -- the comparison this launch exists to make.

THE QUESTION IT BUYS, which launch 57 could not isolate. Launch 57 (mine, champion v22) measured a
FIRST cross-layer merge on a fresh attention role at 4.2075015e-9 per parameter and I reported in its
result that its stated reason was NOT established: the c_v-with-ve_gate-compensation argument cannot
be seen, because that first c_v merge and launch 23's c_q five-way average (4.297e-9) are 2.1% apart,
inside reproducibility. c_k has NO per-layer compensating path. So if c_k on {3,5} lands near 4.2e-9
the role term is ~0, "merge count is the whole law" survives its second test, and attention c_proj is
priced for free at 262,144 for whoever holds margin next. If it lands distinctly dearer with the QK
maps kept distinct, a role term exists and my own knowledge file's claim is refuted by me.

PRE-REGISTERED, EVERY LINE A FALSIFIER, and the two limbs are separable in the instruments:
    num_params_total            14,549,100      EXACT (-262,144, -1.7699% vs v22; -71.094% vs the
                                                50,332,176 reference)
    census                      wte 4,194,304 / value_embeds 0 / lm_head 131,072 /
                                transformer_matrices 10,223,712 / scalars 12, NO UNACCOUNTED row
    flops_per_token_measured    121,897,536     EXACT = 134,480,448 - 6,144*2,048. The share adds
                                                nothing: counted FLOPs are use-weighted, so an
                                                aliased map is still charged per use (launches 23,
                                                57). The window subtracts 6,144 per unit of span sum
                                                (launch 61 measured that coefficient exactly).
                                                51.0% of the 239,078,400 ceiling.
    flops_per_token_analytic    99,877,440      EXACT = 114,033,216 - 6*262,144 - 12,582,912
    analytic/measured gap       22,020,096      = v22's 20,447,232 + 6*262,144 for the one re-used
                                                c_k, the identity launches 23 and 57 both recorded
    step-0 loss                 9.010914        EXACT. attn.c_proj and lm_head are zeros-init, so
                                                the aliased c_k cannot move it, and init_weights
                                                draws the same number of uniforms either way.
    training_data_tokens_available  631,241,817 EXACT (equality constraint)
    num_steps                   2,536-2,552, centred 2,544 = v22's 2,421 x launch 61's realised
                                +5.0847%. A share is step-neutral (23: 2,371->2,370; 36: 2,379->
                                2,381; 57: 2,420->2,421), so the whole move is the window.
    median_step_ms              ~236 (v22's 249 x launch 61's -5.20%)
    total_tokens                664,797,184-669,515,776, i.e. 1.053-1.061 epochs. Repeated use is
                                explicitly allowed by the task and launch 61 measured the income
                                rate holding past one epoch.
    peak_vram_bytes             < 28,315,575,296, direction only. Launch 61 measured the window as
                                BIT-IDENTICAL on this instrument, and a removed (512,512) parameter
                                takes its Muon state with it.

THE PRICE AS AN INTERVAL THAT CONTAINS 1, not as a forecast. Budget = v22's 0.0001356489746 plus the
lever's 0.0021325632 = 0.0022682122. Cost of 262,144:
    at launch 57's first-merge-fresh-role rate 4.2075e-9    0.0011030   cover 2.06x
    at analyst3's marginal-single-map band 7.1e-9-9.5e-9    0.00186-0.00249   cover 1.22x-0.91x
Predicted val_bpb 1.0488348 (cheap end) to 1.0502218 (dear end). The dear end MISSES, and it is the
band a role term would put this in. I am not reporting a point estimate.

WHAT A MISS MEANS, stated before the measurement. If num_steps lands in band and val_bpb >= 1.05,
the lever transported and c_k on a distinct-QK pair is dearer than 8.65e-9 per parameter, which
establishes the role term, closes attention c_proj at this granule on any live fork, and says
launch 57's cheap rate was about c_v specifically. If num_steps is BELOW band the lever did not
transport and the c_k price is not readable from this launch -- that is the one outcome that makes
this composite ambiguous, and it is why num_steps is pre-registered as a hard falsifier rather than
a watch item.

WHAT IS NOT CHANGED. The c_q share stays h[1:-1]. The c_v share stays h[3]<-h[1]. has_ve untouched,
all three ve_gates still fed at ve_gate_channels 8. MLP_RATIO 1.75 (hidden 896, uniform -- the width
axis belongs to gpu5's thr_win_mlp864_on_v22 and gpu1's split rung this rotation, and I am not
touching it). LOGIT_RANK 256 (redundancy's red_logit_rank_192_on_v22 owns that literal),
WINDOW_PATTERN "SSSL", WARMDOWN_RATIO 0.7, WARMUP_RATIO 0.0, FINAL_LR_FRAC 0.0, EMBEDDING_LR 0.3,
UNEMBEDDING_LR 0.004, MATRIX_LR 0.04, SCALAR_LR 0.5, WEIGHT_DECAY 0.2, ADAM_BETAS (0.8, 0.95),
ns_steps 5, TOTAL_BATCH_SIZE 2**18, DEVICE_BATCH_SIZE 128, DEPTH 6, ASPECT_RATIO 85, HEAD_DIM 128
and DATA_BUDGET_TOKENS are exactly as launch 57 measured them. setup_optimizer is untouched: its
partition assert reads self.transformer.h.parameters(), which de-duplicates, so the aliased c_k is
counted once on both sides -- the same reason the existing c_q and c_v shares pass it. Muon groups
are keyed on shape and c_k is 512x512 like the rest, so one member leaves a group and no group is
added, removed or re-weighted. The share is of the c_k MODULE, never of its Parameter: the model is
built on meta and then to_empty()ed, and nn.Module._apply rebuilds _parameters[key] per module, so
an aliased raw Parameter in two distinct Linears would be un-aliased there. prepare.py,
pyproject.toml and uv.lock are bit-identical to champion/, and the three frozen regions -- the
"Parameter counts:" print block, the single measure_flops_dispatch call with its unmodified returned
integer, and the single report_efficiency_metrics call -- are preserved verbatim.

Experiment thr_share_cv_13_on_v20 (team throughput, run by autoscts__params_gpu6).
ONE statement added to champion v20's source. No constant touched, no lever:

    self.transformer.h[3].attn.c_v = self.transformer.h[1].attn.c_v
        one attn.c_v serves layers 1 and 3, 6 distinct value maps -> 5, -262,144 -> 14,811,244

WHY THIS ROW AND WHY NOW. This is the FIRST price this run has ever put on an attention tensor role
other than c_q. knowledge/unqueued_axes.md has carried c_k, c_v and c_proj cross-layer sharing --
262,144 parameters each -- with the note "cold and valuable, unbuyable until a lever lands", and
[SUGGESTION] a16ff981 handed the pair over with the arithmetic done and asked for a free lane. Two
independent posts this rotation ([SUGGESTION] a16ff981 and [PROPOSAL] d0ede7ad) reached the same
conclusion from opposite directions: the affordable band contains exactly ONE mechanism,
mlp_hidden_width, because it is the only axis whose granule (6,144 per unit of width) is smaller than
the margin. Every other granule is 262,144 or larger. So the board is one-dimensional until either a
lever lands or a 262,144 granule is shown to fit. This launch tests the second.

WHY THE BASE IS v20 AND NOT THE LIVE CHAMPION v21. v21 (launch 52, hidden 880) spent 0.0006886 of
v20's margin to buy 98,304 parameters at 7.0047e-9 each and holds 0.0005500265058662. v20's fork is
unspent and holds 0.001238620249468969, which covers 262,144 parameters at any per-merge rate at or
below 4.7255e-9. The target 14,811,244 is 163,840 BELOW the live champion and 114,688 below gpu1's
in-flight cap_mlp872_on_v21, so this candidate cannot be raced out by either of them.

THE PRICE, ON MARGINAL RATES AND NOT AVERAGES. a16ff981 priced this row at 0.0036087 per 1M, which is
launch 30's FIVE-WAY AVERAGE, for cover 1.31x. That is the wrong unit -- what is being bought here is
ONE marginal merge. The three measured module merges in this run are:

    launch 23  c_q, all six merged, on v9      -1,310,720 (5 merges)  +0.005632525  0.001127/merge
    launch 36  c_q, the SIXTH merge, on v16      -262,144 (1 merge)   +0.0025034    9.5497e-9/param
    launch 38  mlp c_proj pair, on v16           -491,520             +0.0036590919 7.4444e-9/param

Cover on v20's 0.001238620249468969: 1.099x at launch 23's per-merge average, 0.670x at launch 36's
own matched-pair figure of 0.0018499, 0.495x at launch 36's realised last-merge rate. The honest
interval is 1.10x..0.50x and it contains 1. I am reporting that rather than the handed-over 1.31x.

THE TWO REASONS TO EXPECT THE CHEAP END, both measured rather than argued. First, merge cost is convex
in merge COUNT on the one role this run has measured twice: launch 23's average over five merges is
4.297e-9 per parameter and launch 36's SIXTH merge alone is 9.5497e-9, a ratio of 2.22, and launch 36
recorded its own convexity ratio as 1.3533. This is the FIRST merge on a fresh role, so it belongs at
the cheap end of that curve, not the dear end. Second, layers 1 and 3 are both has_ve layers --
has_ve(i,6) is i%2==1, so {1,3,5} -- and each therefore keeps its OWN ve_gate and computes
v = c_v_shared(x) + gate_i * ve. c_v is the only one of the three unpriced roles with a per-layer
compensating path already in the model, and after launch 50 that path is MEASURED, not assumed: 96
parameters of gate were worth 0.0008219 of val_bpb. c_k has no such path, and sharing it beside the
already five-way-shared c_q would make the whole QK bilinear form layer-invariant, which is a
function-class restriction in its own price band. Attention c_proj is the role-analogue of the MLP
c_proj that measured the DEAREST share rate in the run.

WHY A SHARE OUTRANKS A WIDTH RUNG ON A MARGIN-BOUND BOARD, which is my own launch 51's argument turned
around. A share's entire price is mechanism. A width rung's price is mechanism PLUS a step-count term,
and launch 51 (thr_mlp892_on_v19, mine) measured that term at +4.42% of median step and 103 updates the
moment a matmul dimension leaves the 8-grid. c_v is 512 -> 512: no dimension moves, nothing leaves the
8-grid, and counted FLOPs do not move either because they are use-weighted on this substrate and an
aliased map is still charged per use. Launch 23 confirmed both halves of that directly --
flops_per_token_measured unchanged, num_steps 2,370 against a base 2,371.

PRE-REGISTERED, AND EVERY ONE OF THESE IS A FALSIFIER. num_params_total 14,811,244; census wte
4,194,304 / value_embeds 0 / lm_head 131,072 / transformer_matrices 10,485,856 / scalars 12 with no
UNACCOUNTED row; flops_per_token_measured 134,480,448 UNCHANGED; flops_per_token_analytic 114,033,216
= 6*(14,811,244 - 4,194,316) + 12*4*128*8192; the analytic/measured gap 20,447,232, which is v20's
18,874,368 plus exactly 6*262,144 for the one re-used c_v -- the same identity launch 23 recorded for
its five re-used c_q; step-0 loss 9.010914 exact, because attn.c_proj and lm_head both init to zeros;
training_data_tokens_available 631,241,817; num_steps 2,414-2,426 CENTRED ON NO CHANGE from 2,420;
median_step_ms 249; total_tokens 632,946,688-636,092,416; peak_vram_bytes below 28,317,281,280,
direction only, launch 23 having measured -12,068,864 for five merges. If flops_per_token_measured
moves at all, the use-weighted charging model is wrong. If num_steps leaves the band, share-is-step-
neutral is wrong, and that property is the whole reason this outranks a width rung.

WHAT A MISS MEANS, stated before the measurement. val_bpb >= 1.05 puts the FIRST merge on a fresh
attention role above 4.7255e-9 per parameter, which -- since no live fork holds more margin than v20 --
CLOSES all three unpriced attention roles at the 262,144 granule for the rest of this run, and the
three rows carrying them in three queues should be retired rather than re-priced. That is worth a
launch on its own: those rows have been held open across several rotations on an average rate that the
marginal measurement above says is 2.65x too cheap.

WHAT IS NOT CHANGED. The c_q share stays h[1:-1]. ve_gate_channels stays 8, has_ve is untouched and
every surviving ve_gate is still fed, so layers 1, 3 and 5 each keep their own gate. MLP_RATIO stays
1.75 (hidden 896) -- this is v20's fork, not v21's. LOGIT_RANK 256, WARMDOWN_RATIO 0.7, WARMUP_RATIO
0.0, EMBEDDING_LR 0.3, UNEMBEDDING_LR 0.004, MATRIX_LR 0.04, SCALAR_LR 0.5, WEIGHT_DECAY 0.2,
ADAM_BETAS, ns_steps 5, TOTAL_BATCH_SIZE 2**18, DEVICE_BATCH_SIZE 128, DEPTH 6, ASPECT_RATIO 85,
HEAD_DIM 128, WINDOW_PATTERN "SSSL" and DATA_BUDGET_TOKENS are exactly as launch 50 measured them.
setup_optimizer is untouched and its partition assert still holds: matrix_params comes from
self.transformer.h.parameters(), which de-duplicates, so the aliased c_v is counted once on both sides
of the assert -- the same reason the existing c_q share passes it. The Muon groups are keyed on shape,
and c_v is 512x512 like the rest, so one member leaves a group and no group is added, removed or
re-weighted. prepare.py, pyproject.toml and uv.lock are bit-identical to champion/, and the three
frozen regions -- the "Parameter counts:" print block, the single measure_flops_dispatch call with its
unmodified returned integer, and the single report_efficiency_metrics call -- are preserved verbatim.

Experiment red_ve_gate_ch8_on_v19 (team redundancy, run by autoscts__params_gpu3).
ONE literal on champion v19's source, no lever, and it is a CUT:

    CausalSelfAttention.ve_gate_channels  32 -> 8      three ve_gates (4,32) -> (4,8)
                                                       -288 parameters -> 15,073,388

WHY THIS ROW EXISTS. Champion v19 holds 0.000416728792299 of gate margin, and every structural rung
in every team's inventory is 262,144 or larger except three: LOGIT_RANK 192 (-32,768, throughput's,
re-priced expensive because it is a RANK cut), HEAD_DIM 256 (-192) and this one. autoscts__params_analyst1
(team capacity) re-opened this row this rotation after four ledger passes carried it as `closed` --
"all four gates together are 512 parameters, cannot move the headline" -- a closure written against a
50,332,176 champion and false at 15,073,676. Capacity recorded it and did not queue it: "a -192 row
belongs to whoever has a free slot first". This is that slot, at a LARGER rung than the -192 they named,
for a reason given below.

WHAT THE GATE IS. On layers {1,3,5} (has_ve(i,6) is i%2==1) the value residual is
    gate = 2 * sigmoid(ve_gate(x[..., :ve_gate_channels]));  v = c_v(x) + gate.unsqueeze(-1) * ve
with ve = tok, the raw unnormalised wte lookup, and ve_gate.weight ZERO-INITIALISED (init_weights).
So at step 0, 2*sigmoid(0) = 1.0 exactly and this candidate computes a function BIT-IDENTICAL to
champion v19's. The entire measured difference is what a 32-channel gate learns that an 8-channel gate
cannot. That makes the early train-loss curve a free falsifier: it must track launch 37's step-for-step
at the start, and step-0 loss must be exactly ln(8192) = 9.010914.

WHY 8 AND NOT 16, AND NOT 4 OR 0. The rung is chosen by what leaves the gate's own optimizer
untouched, read off the live source rather than assumed:
  * MuonAdamW._step_muon (L1070) scales the group LR by max(1.0, shape[-2]/shape[-1])**0.5.
    ve_gate is (n_kv_head, channels) = (4, C). C >= 4 keeps that factor at exactly 1.0;
    C = 2 makes it 1.4142 and C = 1 makes it 2.0. So every rung below 4 silently RAISES the gate's
    learning rate and is not a pure width probe.
  * muon_step_fused's NorMuon reduction dimension is red_dim = -1 if shape[-2] >= shape[-1] else -2
    (L1065). At (4,32) and (4,16) and (4,8) that is -2; at (4,4) it FLIPS to -1 and the second-moment
    state changes from (n,1,C) to (n,4,1). So C = 4 changes the gate's variance normalisation too.
  * Deleting the gate entirely (-384) removes a whole Muon shape group from param_groups.
    C = 8 is therefore the LARGEST rung that changes no optimizer behaviour of any kind: same group
    count, same per-shape LR factor, same reduction axis, same polar_express branch (g.size(-2) >
    g.size(-1) is False at both (4,32) and (4,8)).
  * And it is the rung whose MISS is decisive, which is gpu4's point 2 on [SUGGESTION] 48667f29:
    if 8 channels clear the gate, C = 16 is dominated (-192 < -288 on a minimised target) and the axis
    has 96 parameters left, i.e. it is spent in ONE launch instead of a ladder. If 8 misses, 16 is the
    follow-up and the axis is bracketed.

PRE-REGISTERED PREDICTION -- a falsifiable BOUND, not a point estimate, because no launch in this run
has priced this tensor role. Team redundancy's strategy.md adopted that rule after launch 38, where six
independent point estimates agreed with each other and were all LOW because they were six copies of one
transported error.

    CLAIM: |delta val_bpb| < 0.0001628969454063, the run's only measured same-program spread
           (launches 32/33). FALSIFIED if |delta| >= that.

    Mass-rate sanity, stated as a spread and not as votes: at the cheapest rate this run has ever
    measured (0.0036087/1M, the c_q five-way share at launch 30) 288 parameters cost 0.00000104, and at
    the DEAREST (0.0241/1M, launch 27's function-class restriction) 0.00000694. That is 0.006x-0.043x
    of the noise spread and 0.0025x-0.017x of the margin. Mass says free by a factor of 60-400.

    So this launch does not really ask "can I afford 288 parameters". It asks THE open question after
    launch 38, which found that a tensor's ROLE dominates its per-parameter rate by 3.87x on one base:
    can a MECHANISM cost more than its mass implies, at the smallest mass the run has ever tested?
    A miss here means a 288-parameter gate costs >= 1.45 per 1M, which is 60x the dearest rate ever
    measured. That would be the run's most important pricing finding and would explain why the board
    is stuck; it would not be a small negative result.

DECISION RULE, pre-registered.
  |delta| < 0.0001629            KEEP at 15,073,388. Gate width is not priceable at this rung; mass
                                 rates are safe at small masses; C = 16 is dominated and the axis is
                                 spent. Follow-up: C = 4 (-336) is worth ONE launch only if a lever
                                 ever lands, and its red_dim flip must be declared when it is.
  0.0001629 <= delta < 0.0004167 KEEP at 15,073,388 AND record the multiplier: the gate is priced
                                 above its mass by >=23x. Bank the target, spend no more on the axis.
  delta >= 0.0004167             INELIGIBLE, DISCARD. The 32-channel gate is load-bearing. Follow-up
                                 rung C = 16, and the finding above is the result.
  delta < 0 (a gain)             KEEP, and the extra 24 channels were actively harmful -- then C = 4
                                 and gate removal both become live rows rather than curiosities.

DETERMINISTIC INSTRUMENTS, pre-registered exactly. Any miss means the edit did something I did not
intend, and I would rather read that off the witness than off val_bpb:
    num_params_total            15,073,388   (v19's 15,073,676 - 288)
    census wte                   4,194,304   unchanged
    census lm_head                 131,072   unchanged
    census transformer_matrices 10,748,000   (10,748,288 - 288)
    census scalars                      12   unchanged
    census value_embeds                  0   unchanged, still an empty nn.ModuleDict
    flops_per_token_analytic   115,606,080   (115,607,808 - 6*288); the gate is inside transformer.h
                                             so it is NOT in estimate_flops_analytic's exclude list
    flops_per_token_measured   134,480,448   (134,482,176 - 6*288)
    analytic/measured gap       18,874,368   UNCHANGED -- the gap is the wte column slice charged by
                                             use in the logit path, which this edit does not touch
    step0_loss                    9.010914   exactly ln(8192); lm_head is zeros_ and the gate is
                                             neutral at init, so step 0 is champion-identical
    num_steps                    2419 +- 4   1,728 of 134,482,176 FLOPs/token is 0.0013%, so there is
                                             no throughput story here; launches 37 and 44 both read
                                             2419 on this base and were bit-identical on it
    total_tokens               634,126,336   at 2419 x 262,144
    peak_vram_bytes         <= 28,318,068,736 + allocator noise. Directionally LOWER: the
                                             x[..., :C] slice is materialised contiguous for F.linear
                                             and retained for the weight gradient, 128x2048xC bf16 =
                                             16.78 MB/layer at C=32 and 4.19 MB at C=8, so up to
                                             ~37.7 MB less retention across the three VE layers.
                                             Ceiling is 47,198,976,512; no risk either way.

WHAT IS NOT CHANGED, recorded because each would otherwise cost a charged launch. n_kv_head stays 4,
so the gate's OUTPUT width and ve's (B,T,4,128) view are untouched. Every ve_gate stays fed and stays
per-layer: has_ve is unchanged, so layers {1,3,5} still own a gate and MuonAdamW._step_muon still
receives a grad for each (it stacks p.grad with no None guard). value_embeds stays an empty
nn.ModuleDict, which the frozen prepare.py indexes in count_params and estimate_flops_analytic.
setup_optimizer is byte-identical: the gate is still exactly one shape in exactly one Muon group, so
no group is added, removed or re-weighted, and the parameter-partition assert holds unchanged. No LR,
schedule, width, depth, window, batch or data constant moves. The three frozen regions -- the
"Parameter counts:" print block, the single measure_flops_dispatch call with its unmodified returned
integer, and the single report_efficiency_metrics call -- are verbatim from champion v19, and
prepare.py, pyproject.toml and uv.lock are byte-identical to champion/.

--- inherited record from champion v19 and its lineage, unchanged below this line ---

Experiment thr_mlp896_on_v15 (team throughput, self-proposed and run by autoscts__params_gpu5).
ONE constant on champion v15's source, and no lever:

    MLP_RATIO 1.875 -> 1.75     MLP hidden width 960 -> 896, -393,216 parameters

WHY, in one line: v15 has 0.0028007832340883 of margin because launch 34's lever over-delivered, and
this rung costs less than that on every estimate the run can make of it.

THIS IS FINANCED BY MARGIN, NOT BY A LEVER, and that is the point. v15 already banks both of the
run's measured zero-parameter levers -- WARMDOWN_RATIO 0.7 and EMBEDDING_LR 0.3 -- so there is nothing
left to pair with. If a cut can still be bought out of margin alone, the run is not margin-bound at
this width; if it cannot, then it is, and the next move has to be a new lever rather than a new rung.

THE ARITHMETIC, pre-registered. Four independent estimates of the SAME rung, and all four clear:

  base       champion v15 thr_mlp960_embed03_on_v13, 15,466,892 at val_bpb 1.0471992167659117,
             gate margin 0.0028007832340883   (champion.md v15/v16, launch 34)

  estimate                              cost        cover   predicted val_bpb   margin left
  direct v10 -> v11 pair, low W        0.0013040    2.15x      1.0485032        +0.0014968
  direct v10 -> v11 pair, high W       0.0017797    1.57x      1.0489789        +0.0010211
  flat per-parameter 5.556e-9          0.0021847    1.28x      1.0493839        +0.0006161
  backward from the 896 -> 864 rung    0.0026007    1.08x      1.0497999        +0.0002001

  The FIRST TWO are the relevant ones: results/cap_mlp896_warmdown7.md measured exactly this rung
  (hidden 960 -> 896) on exactly this substrate, moving v10 -> v11 by net +0.0000483 while ALSO adding
  WARMDOWN_RATIO 0.7 in the same step; subtracting that lever's own measured range
  (0.0012557 realised, 0.0017314 at origin) brackets the rung at 0.0013040-0.0017797. The third and
  fourth are extrapolations kept only as a worst case, and even the worst case is eligible by
  0.0002001. Central expectation: val_bpb ~1.0487, margin ~0.0013.

  Two tailwinds NOT counted above, both left out on purpose so the prediction stays falsifiable:
    * hidden 896 is 7*128, 128-ALIGNED. The 960 rung v15 holds is not, and launch 34 measured its
      wall-clock conversion at 0.10 against the 0.23 that file had recorded. An aligned rung converts
      0.70, so this should return ~+2.4% steps where the last one returned +0.17%.
    * the two estimates that bracket the rung both come from a base at WARMDOWN 0.5 or in transition;
      v15 is at 0.7 throughout.

PRE-REGISTERED INTEGERS, from workspace/census2.py (14/14 on the original oracle, 7/7 on the
v9-lineage extension, and 8/8 counting launch 34, which it predicted exactly):

    num_params_total            15,073,676   = -393,216 vs v15 (-2.542%)
                                             = -70.052% vs the measured reference 50,332,176
                                             -- the first candidate in this run below -70%
                                 = wte 4,194,304 + lm_head basis 131,072 + scalars 12
                                 + transformer unique 10,748,288
    flops_per_token_measured   134,482,176   ceiling 239,078,400, uses 56.2%
                                 -- identical to launch 25's measured value, which is the check:
                                    c_q sharing is free in a tally that charges a weight once per
                                    matmul that reads it, so hidden 896 reads the same total either
                                    way
    flops_per_token_analytic   115,607,808   (log only)
    training_data_tokens_available  631,241,817   must be == , untouched
    step-0 loss                  9.010914   = ln(8192); the MLP init bound
                                             s = 3**0.5 * n_embd**-0.5 does not read hidden
    num_steps                    ~2,430-2,445 (v15 ran 2,379). A 128-aligned -1.72% counted refund at
                                 the 0.70 conversion that file measured for aligned widths.
    peak_vram_bytes              below v15's 28,722,560,000, ceiling 47,198,976,512

WHAT A MISS WOULD MEAN, pre-registered:
  * a miss of ANY size refutes all four estimates at once, since the loosest already leaves
    +0.0002001. The reading would be that the mlp_hidden_width rung price RISES steeply once the MLP
    is already narrowed -- consistent with results/bracket_cap_mlp_ratio1.md, where 8,388,608
    parameters of hidden width were free and the next 3,670,016 cost +0.0076330 -- and that the axis
    is CLOSED at hidden 960 on this lineage rather than merely dearer.
  * it would also answer this launch's own question in the negative: margin alone cannot buy a rung at
    this width, and every further cut needs a new zero-parameter lever. The unmeasured candidates are
    WEIGHT_DECAY, the Muon momentum ramp's hardcoded 300 steps, WARMUP_RATIO, FINAL_LR_FRAC and
    UNEMBEDDING_LR, none of which has a launch.
  * a hit sets the axis's third consecutive rung on this lineage and leaves 864 and 832 to price.

FALSIFIERS: num_params_total must be exactly 15,073,676 and flops_per_token_measured exactly
134,482,176, with training_data_tokens_available unchanged. If either integer differs the census is
wrong and no val_bpb reading from this launch may be transported.

--- inherited docstring of launch 34 (champion v15) follows, unchanged ---

Experiment thr_mlp960_embed03_on_v13 (team throughput, self-proposed and run by
autoscts__params_gpu5). TWO constants on champion v13's source, each with a price this run has
already paid for:

    MLP_RATIO    2   -> 1.875   MLP hidden width 1024 -> 960, -393,216 parameters
    EMBEDDING_LR 0.6 -> 0.3     zero parameters, the financing

plus the three-line mlp_ratio float/assert/int mechanism that launch 22 introduced and that champion
v13 does not carry, because v13 forked from v9.

WHY THIS PAIR, in one line: champion v13 dropped the MLP narrowing when it forked from v9, and
launch 31 has just measured a margin lever big enough to buy the first rung of it back.

THE ARITHMETIC, pre-registered. Every number below is a measurement of THIS run, cited to the
result file that recorded it.

  base            champion v13 red_attn_share_q5_warmdown_hi, 15,860,108 at val_bpb
                  1.0482225755426247, gate margin 0.0017774244573753   (champion.md v13)
  the cut         MLP hidden 1024 -> 960. Measured at launch 22 on champion v9, the SAME rung on
                  the SAME 6x512 substrate: -393,216 parameters for +0.0021847 of val_bpb
                  (results/cap_mlp_hidden960.md, measured_rate_val_bpb_per_1M 0.005556).
  the financing   EMBEDDING_LR 0.6 -> 0.3. Measured at launch 31 on champion v11 as
                  -0.0019284260561595 (results/red_embed_lr_lo_03.md). Its knowledge file
                  (knowledge/embedding-lr-is-the-runs-newest-margin-lever.md) instructs a ~70%
                  discount when folding it into a cut on another base, citing WARMDOWN_RATIO's
                  own 73% transport, and asks the builder to say which they used.
                  I AM NOT BUILDING ON v11's SOURCE, so I use the discount: 0.0013.
  budget          0.0017774244573753 + 0.0013 = 0.0030774
  cost            0.0021847
  cover           1.41x discounted, 1.70x if the lever transports in full

  predicted val_bpb   1.0482226 + 0.0021847 - 0.0013     = 1.0491073   (margin 0.0008927)
                      1.0482226 + 0.0021847 - 0.0019284  = 1.0484789   (margin 0.0015211)
  ELIGIBLE across the whole transport band. Ineligible requires the lever to deliver less than
  0.0004073, i.e. under 21% of what it measured -- outside the 0.7x-1.25x transport budget this
  team's strategy.md records for a bit-identical-instrument lever.

PRE-REGISTERED INTEGERS, from workspace/census2.py, which is exact on 14/14 of the original
oracle's cases and 7/7 of the v9-lineage extension (launches 15, 20, 22, 23, 25, 29, 30):

    num_params_total            15,466,892   = -393,216 vs v13 (-2.479%), -69.270% vs the
                                               measured reference 50,332,176
                                 = wte 4,194,304 + lm_head basis 131,072 + scalars 12
                                 + transformer unique 11,141,504
                                 (read-by-use 12,190,080 less the four de-duplicated c_q maps
                                  1,048,576; the census transformer_matrices row is 11,141,504)
    flops_per_token_measured   136,841,472   ceiling 239,078,400, uses 57.2%
                                 -- IDENTICAL to launch 22's measured value, because sharing c_q
                                    is free in a tally that charges a weight once per matmul that
                                    reads it, and the MLP refund is the same 6*393,216 = 2,359,296
    flops_per_token_analytic   117,967,104   (log only)
    training_data_tokens_available  631,241,817   must be == , untouched
    step-0 loss                  9.010914   = ln(8192); the MLP init bound
                                             s = 3**0.5 * n_embd**-0.5 does not read hidden, and
                                             an LR cannot move step 0
    num_steps                    ~2,380-2,390 (v13 ran 2,375). Hidden 960 is NOT 128-aligned, so
                                 knowledge/mlp-hidden-refund-is-quantised-to-128.md predicts only
                                 0.23 of the counted refund in wall clock: +0.42% steps, not
                                 +1.7%. EMBEDDING_LR is step-neutral (launch 31 moved num_steps by
                                 exactly 0).
    peak_vram_bytes              below v13's 29,127,444,480, ceiling 47,198,976,512. Narrowing the
                                 MLP removes activation and Muon state; an LR removes nothing.

WHAT A MISS WOULD MEAN, pre-registered, because it is worth a launch either way:
  * miss by less than 0.0009 -> the lever transported at under 70% and the discount in its own
    knowledge file is too generous. Records as the second transport measurement of EMBEDDING_LR
    and re-prices every candidate now being written against it.
  * miss by more than 0.0009 -> the 960 rung is dearer on a c_q-shared base than it was on v9,
    i.e. the SAME base-dependence I measured on the n_kv_head axis (results/thr_gqa_kv1.md, where
    one rung cost -0.0013831 on one base and +0.0096530 on another). That would say sharing and
    narrowing compete for the same capacity and must not be added as independent prices.
  * a hit ALSO answers an open question nobody has bought: whether the run's two levers compose.
    WARMDOWN_RATIO 0.7 is a pure generalisation gain (train loss flat, gate better) and
    EMBEDDING_LR 0.3 is a pure optimisation gain (train loss better at every probe, gate better
    by the same amount converted). v13 carries the first; this adds the second. Their knowledge
    file says "nobody has measured them together".

FALSIFIERS, checked before the result is written: both deterministic instruments must return the
pre-registered integers exactly, and training_data_tokens_available must be unchanged. If
num_params_total is not 15,466,892 the census is wrong and no val_bpb reading from this launch may
be transported anywhere.

--- inherited docstring of launch 30 (champion v13) follows, unchanged ---

Experiment red_attn_share_q5_warmdown_hi (team redundancy, proposed and run by
autoscts__params_gpu4). This is launch 26's candidate with the shared group made ONE LAYER
SMALLER, and that is the only difference: `for block in self.transformer.h[1:-1]` instead of
`[1:]`, so the last layer keeps its own c_q and five layers share one.

WHY, in one line: launch 26 missed the gate by 0.0000724316870533 and the finest granularity this
axis has is one layer, worth 262,144 parameters.

The two numbers this candidate is priced from are both mine, both measured, and both measured on
this base and this tensor family -- which is the whole reason to run it rather than guess again:

  launch 23  six-way c_q merge, WARMDOWN 0.5   15,597,964   val_bpb 1.05132809022773
             => 0.005632525 for 1,310,720 params = 0.004297 per 1M
  launch 26  the same source, WARMDOWN 0.7     15,597,964   val_bpb 1.0500724316870533
             => the lever delivers -0.0012556585406767 here (72.5% of the 0.0017314 gpu5
                measured on the kv1 base at launch 21)

  this candidate: five-way merge = 4 distinct maps lost instead of 5
      sharing cost   1.048576 M x 0.004297 per 1M  = +0.004506
      warmdown 0.7                                 = -0.0012557
      predicted val_bpb  1.045695565 + 0.004506 - 0.001256 = 1.048946
      predicted margin   0.001054  -- 14.5x launch 26's miss

THE ONE ASSUMPTION, NAMED. That the merge cost is PROPORTIONAL to the number of distinct maps
lost rather than concave in it (i.e. that the first merge does not do most of the damage). That
proportionality is exactly what knowledge/sharing-price-is-set-by-distinct-maps-lost.md claims, so
this rung is also the test of the rule that closed this team's own limb. If the cost is concave,
this lands near launch 26's 1.0500724 and misses again -- and that answer is worth having, because
it would mean the whole one-factor ladder is priced by its FIRST merge and every rung of it is
equally unaffordable.

WHICH LAYER KEEPS ITS OWN c_q, AND WHY THE LAST. Its output is what the classifier reads through
norm(x); it is one of the two long-context layers (WINDOW_PATTERN "SSSL" at DEPTH 6 gives spans
[1024,1024,1024,2048,1024,2048], and window_sizes[-1] is forced long); and excluding it leaves the
shared map serving spans [1024,1024,1024,2048,1024], which is less heterogeneous than before. The
choice is recorded rather than optimised: no other layer was tried, because trying one is a launch.

WHAT I AM DELIBERATELY NOT DOING. Not WARMDOWN 0.9 on launch 26's source, even though the
bracket's decaying slope (-0.0037651 per 0.2 from 0.3->0.5, then -0.0017314 per 0.2 from 0.5->0.7)
extrapolates to about -0.0018 total and 1.88x cover. gpu6's launch 24 measured the price of that
exact extrapolation: the batch lever returned -0.0167157 at its first rung and +0.0014287 at its
second, reversing sign one rung past where it had been measured. Buy the rung that was measured.

EXACT ARITHMETIC. Only the count of unique (512,512) tensors changes, 19 -> 20:

  unique (512,512)         1 shared c_q + layer 5's own c_q + 6 c_k + 6 c_v + 6 c_proj = 20
                           20 x 262,144 = 5,242,880   (launch 26: 19 x 262,144 = 4,980,736)
  transformer_matrices     11,272,576 -> 11,534,720
                           = attn 5,242,880 + mlp 6,291,456 + 3 ve_gates 384
  num_params_total         15,597,964 -> 15,860,108   (-6.20% vs champion v9's 16,908,684;
                           -1.63% vs the live v11 champion's 16,122,252; -68.49% vs the reference)
  census check             wte 4,194,304 + lm_head 131,072 + 11,534,720 + scalars 12 = 15,860,108
  flops_per_token_measured 139,200,768, UNCHANGED for the third launch running: the dispatch
                           counter charges by USE and all six layers still issue their own
                           512x512 query matmul. 58.2% of the 239,078,400 ceiling.
  flops_per_token_analytic 120,326,400 (log only), so the measured-minus-analytic gap is
                           18,874,368 = 6 x (the wte slice 2,097,152 + the FOUR re-used c_q
                           1,048,576). Launch 23/26 read 20,447,232 with five re-used; the
                           champion reads 12,582,912 with none. That integer is the witness for
                           exactly how many layers share.
  step-0 loss              exactly ln(8192) = 9.010913. lm_head is still zero-initialised and
                           get_lr_multiplier(0) is 1.0 at WARMDOWN 0.7 since progress 0 < 0.3.
  num_steps                ~2372, unchanged: one more Muon matrix in a group whose Newton-Schulz
                           cost launch 23 already measured as unobservable (2370 vs 2371 steps at
                           24 vs 19 matrices).
  peak_vram_bytes          ~29,127,000,000, i.e. launch 26's 29,125,869,568 plus one more
                           (512,512) fp32 momentum slot and its stack temporaries (~+1.0 MB each).
                           61.7% of the 47,198,976,512 ceiling.

--- inherited docstring of launch 26 follows, unchanged ---

Experiment red_attn_share_q_warmdown_hi (team redundancy, proposed and run by
autoscts__params_gpu4). This is launch 23's candidate, financed, and it is the whole file plus
ONE constant: WARMDOWN_RATIO 0.5 -> 0.7.

Launch 23 measured this exact source at num_params_total 15,597,964 -- exactly as predicted,
along with both FLOPs numbers and the step-0 loss -- and val_bpb 1.05132809022773, which misses
the gate by 0.00132809023 and nothing else. Every ceiling had room (FLOPs 58.2% of its ceiling,
peak_vram 61.7% of its). So the mechanism is not in question here; only the financing is, and
the deficit is 0.00132809023 of val_bpb.

CHANGE, and there is only one. WARMDOWN_RATIO 0.5 -> 0.7. gpu5 measured this exact constant on
this run's launch 16 source: launch 21 (thr_warmdown_hi_on_kv1) took val_bpb
1.0538166844620749 -> 1.0520852912719298 at a byte-identical parameter count (16,253,070) and
byte-identical flops_per_token_measured (141,558,528), i.e. -0.0017314 for free. That base also
carries the opposite rung, so the axis is bracketed with three points in one direction:

  WARMDOWN 0.3  (launch 18)   +0.0037651   worse
  WARMDOWN 0.5  (launch 16)    0           the run's value everywhere until launch 21
  WARMDOWN 0.7  (launch 21)   -0.0017314   better

-0.0017314 against a 0.00132809023 deficit is 1.304x cover, so 76.7% of the measured effect has
to survive transport onto this source. STATED PLAINLY: THAT IS A COIN FLIP AND IT IS THE ONLY
UNKNOWN IN THIS LAUNCH. There is no additivity assumption anywhere in it -- the base val_bpb is
a measured number from launch 23's own bytes, not a predicted sum of two deltas -- which is what
makes the transport the entire experiment.

WHY 0.7 AND NOT 0.9, WHICH WOULD HAVE HAD BETTER COVER ON PAPER. The bracket's slope is decaying
(-0.0037651 per 0.2 from 0.3->0.5, then -0.0017314 per 0.2 from 0.5->0.7, a ratio of 0.46), so a
geometric read gives 0.9 about -0.0025 from 0.5, i.e. 1.88x cover. I am not taking it, because
this run measured the cost of exactly that extrapolation 20 minutes ago: the batch lever returned
-0.0167157 at its first rung (2**19 -> 2**18, launch 10) and gpu6's launch 24
(thr_batch_quarter_mlp768) measured its SECOND rung at **+0.0014287** -- the same monotone lever
REVERSED SIGN one rung past where it had been measured, after doubling completed updates from
2,489 to 4,809. Extrapolating a decaying lever one rung beyond its bracket is precisely the move
that just failed, so this candidate buys the rung that was measured and nothing further.

WHAT A MISS WOULD MEAN, pre-registered. If val_bpb >= 1.05 the result is about TRANSPORT, not
about warmdown: it would mean the run's last measured margin lever does not carry across bases
either, and with the batch axis now closed at its second rung there would be no measured
zero-parameter margin source left on this lineage. That is a statement about every financed
candidate in all three teams' queues and should be recorded as such rather than charged to the
c_q share. If it lands, the champion goes to 15,597,964 (-5.56% against the live v10 champion's
16,515,468, -69.01% against the measured reference) and every other team can finance with the
same constant.

EXACT ARITHMETIC. A learning-rate schedule touches no tensor and issues no matmul, so all four
deterministic integers are launch 23's, unchanged, and a different one means the edit did not
land where I think it did:

  num_params_total         15,597,964   (census: wte 4,194,304 + lm_head 131,072
                                        + transformer_matrices 11,272,576 + scalars 12)
  flops_per_token_measured 139,200,768  (58.2% of the 239,078,400 ceiling)
  flops_per_token_analytic 118,753,536  (log only; the 20,447,232 gap to the measured number is
                                        6 x (the wte slice 2,097,152 + the five re-used c_q
                                        1,310,720), which is the witness that the share is real)
  step-0 loss              exactly ln(8192) = 9.010913. get_lr_multiplier(0) is 1.0 at both
                           WARMDOWN values since progress starts at 0 < 1 - 0.7, and lm_head is
                           still zero-initialised, so step 0 is untouched by this change.
  num_steps                ~2370, unchanged: the schedule changes the LR at a step, not its cost.
  peak_vram_bytes          ~29,125,869,568, unchanged: identical shapes and identical batch.

The schedule itself: 30% of the clock at peak LR instead of 50%, then a linear decay to
FINAL_LR_FRAC 0.0 over the remaining 70%. Launch 23's own curve is the reason to expect this to
help at all -- it was still descending 0.0104 over its last 5% against the champion's 0.0032, so
it is further from converged than its own base was, and the anneal is where that gap closes.

--- inherited docstring of launch 23 follows, unchanged ---

Experiment red_attn_share_q_alldepth (team redundancy, proposed and run by
autoscts__params_gpu4). Built on champion.md v9 (cap_depth6_on_champion: num_params_total
16,908,684, val_bpb 1.045695565351953, gate margin 0.004304434648047,
flops_per_token_measured 139,200,768, 2371 steps at 262,144 tokens/update).

ONE conceptual change, three lines: every layer projects its queries through ONE shared
c_q. The other three attention matrices, and both MLP matrices, stay per-layer.

WHY THIS AND NOT A WHOLE MODULE. An attention score is a bilinear form,
score_ij = x_i (W_q W_k^T) x_j^T, so the form is layer-specific whenever EITHER factor is.
Sharing c_q across depth therefore leaves all six score forms distinct through the six
per-layer c_k, and leaves the entire output path (c_v, the value residual, c_proj)
untouched and per-layer. That is the weakest sharing per parameter removed that this
architecture admits, and it is strictly weaker than the three whole-module k=2 rows sitting
in team redundancy's queue, where a shared layer keeps no attention individuality at all.

WHY THIS EXPERIMENT NOW. The champion's block matrices are 12,583,296 = 74.4% of the metric
and nothing in this run has shared one. Every mechanism measured so far touched a vocabulary
table (launches 3, 5, 11, 13, 19), attention K/V rank (16, 17), MLP hidden width (7, 20) or
whole blocks (2, 4, 9, 15). The run is now margin-bound rather than mass-bound: v9 holds
0.004304434648047 and the one measured zero-parameter lever on this lineage,
TOTAL_BATCH_SIZE 2**19 -> 2**18, is already spent inside it. So what remains reachable is
decided by the cheapest cost-per-parameter available on the block mass, and this run has
measured that rate for nothing in the blocks. The two rank-preserving rates it HAS measured
are 0.00031 val_bpb per 1M params (launch 5, four VE tables -> wte) and 0.00146 per 1M
(launch 3, the lm_head <-> wte tie); the rank-REDUCING ones are 0.00210 (launch 11),
0.00520 (launches 16/17) and 0.00619 (launch 19). Layer sharing is rank-preserving --
each shared map keeps its full 512x512 shape and simply serves more layers -- so it should
price with the first pair, and this candidate is sized so that it KEEPs if it does.

NO HYPERPARAMETER IS RE-PICKED, and that is a property of this optimizer rather than a
choice. The shared c_q receives the SUM of six layers' gradients, but muon_step_fused
normalises that away: it divides the stacked gradient by its own norm
(X / (X.norm(dim=(-2,-1)) * 1.02 + 1e-6)), runs the polar-express iteration on the result,
and computes the NorMuon rescale from the ORTHOGONALISED g, so the update magnitude does
not depend on the gradient's scale. A c_q serving six layers moves exactly as far per step
as it did serving one. setup_optimizer is byte-identical to the champion's; the (512,512)
Muon group simply holds 19 tensors instead of 24, at the same per-shape LR multiplier
max(1.0, 512/512)**0.5 = 1.

EXACT ARITHMETIC. num_params_total and flops_per_token_measured are task-declared
deterministic and count_params is an exact sum over unique tensors, so these are
predictions: a different integer means the edit did not land.

  unique (512,512) tensors  24 (6 x c_q,c_k,c_v,c_proj) -> 19 (1 + 6 + 6 + 6)
  parameters removed        5 x 262,144                         -1,310,720
  transformer_matrices     12,583,296 -> 11,272,576
      = attn 19 x 262,144 = 4,980,736  +  mlp 6 x 1,048,576 = 6,291,456  +  3 ve_gates 384
  num_params_total         16,908,684 -> 15,597,964   (-7.75% vs champion, -69.01% vs the
                                                       measured reference's 50,332,176)
  census, which must reconcile against the frozen count_params():
      wte 4,194,304 + lm_head 131,072 + transformer_matrices 11,272,576 + scalars 12
      = 15,597,964

  flops_per_token_measured = 139,200,768, UNCHANGED. The dispatch counter charges by USE,
    not by name: all six layers still issue their own 512x512 query matmul, so the
    use-weighted sum stays 14,811,520 (blocks 12,582,912 + ve_gates 384 + map 131,072 +
    the wte column slice the classifier multiplies by, 2,097,152) and the span term stays
    12*4*128*8192 = 50,331,648. 6*14,811,520 + 50,331,648 = 139,200,768. This candidate
    therefore earns NO FLOPs refund and buys NO extra updates -- which is deliberate: it
    keeps the measurement a price for sharing and not a throughput result. Ceiling
    239,078,400, so 58.2% of it.

  flops_per_token_analytic (log only) = 6*(15,597,964 - 4,194,304 - 12) + 50,331,648
    = 118,753,536, and the measured-minus-analytic gap becomes exactly 20,447,232
    = 6*(the wte slice 2,097,152 + the five re-used c_q's 1,310,720). The champion's own gap
    is 12,582,912 = 6*2,097,152. So the analytic number is an independent witness that the
    sharing landed: 118,753,536 if it did, 126,617,856 if it silently did not.

  peak_vram_bytes <= the champion's 29,137,938,432 (61.7% of the 47,198,976,512 ceiling).
    Every deterministic component moves down: Muon's momentum_buffer for that shape group is
    (19,512,512) fp32 instead of (24,512,512) (-5,242,880), its second_momentum_buffer
    (19,512,1) instead of (24,512,1) (-10,240), and the per-step torch.stack of params and
    of grads each allocate 5,242,880 less. Not a binding axis either way.

  Predicted step-0 loss is again exactly ln(8192) = 9.010913: lm_head is still
    zero-initialised, so step-0 logits are exactly 0. A different step-0 loss means the
    init changed, which this experiment does not touch.

PRE-REGISTERED VERDICT. At the two rank-preserving rates this run has measured, 1,310,720
parameters cost +0.00041 (VE rate) to +0.00191 (tie rate), i.e. val_bpb 1.04611 to 1.04761
against the 1.05 gate, leaving 0.0039 to 0.0024 of margin. The candidate is INELIGIBLE only
if depth-sharing of one attention factor costs more than 0.00328 per 1M -- 2.25x the dearest
rank-preserving rate measured here, and 0.63x the GQA rate. So a miss is itself the answer
to team redundancy's central open question: it would mean layer sharing prices with
rank REDUCTION rather than with sharing, and it would close the three queued whole-module
k=2 rows (all -3,145,728) by arithmetic, since they are 2.4x this mass. A KEEP gives the
first block-matrix sharing rate in the run and the next rungs are already enumerated:
c_v (the other product's factor, another -1,310,720), then c_k or c_proj, then the MLP pair
at -2,621,440 each.

Honest caveat, stated because it is the reason this rung is small rather than the -2,621,440
double. The tie's +0.00146 per 1M was one table serving two consumers of the SAME vocabulary
geometry; this asks one map to serve six different residual-stream states. Treat 0.00146 as
a floor, not a central estimate. That is exactly why the size was chosen to survive 2.25x
the floor instead of 1.1x it.

RIDERS, named rather than compensated:
  - The six layers do not share a context length: with WINDOW_PATTERN "SSSL" at DEPTH 6 the
    spans are [1024,1024,1024,2048,1024,2048]. The shared c_q therefore serves both. It is
    named and not fixed because q is projected BEFORE any window is applied, is rms-normed
    per head immediately after (norm(q)), and carries no span-dependent shape; the window
    enters only in the FA3 call. The queued whole-module rows do not have this excuse, since
    a shared c_k/c_v pair is what a window actually reads over.
  - ve_gate is untouched and stays per-layer. It lives on the attention MODULE, and this
    change shares only the c_q submodule, so all three gates on layers {1,3,5} still exist
    and still receive gradient -- which MuonAdamW._step_muon requires, since it stacks
    p.grad with no None guard.
  - The share is of the c_q MODULE, not of its weight Parameter, and that is load-bearing.
    The model is built on meta and then model.to_empty(device=...); nn.Module._apply
    rebuilds self._parameters[key] per visited module, so aliasing one Parameter into six
    distinct Linear modules would be UNDONE by to_empty and would silently restore
    16,908,684. Assigning the same submodule keeps one _parameters dict, which every
    reference sees. init_weights then draws uniform_ into that one tensor six times; the
    last draw stands, which is harmless and deterministic.
  - No parameter group is added, removed or re-weighted, and setup_optimizer's partition
    assert still holds exactly: both sides of it de-duplicate, so a shared tensor is counted
    once on each side.
  - A small unpriced throughput gain is expected and NOT claimed: the polar-express
    iteration for the (512,512) group runs over 19 matrices instead of 24. On a 254 ms step
    that is worth single-digit steps out of 2371 and it is not part of the prediction above.

--- inherited docstring of the base champion follows, unchanged ---

Experiment cap_depth6_on_champion (team capacity, queued by autoscts__params_analyst1, run by
autoscts__params_gpu1). Built on the red_unembed_via_wte_r256_bhalf champion (champion.md v8:
num_params_total 19,005,966, val_bpb 1.039742373256662, gate margin 0.010257626743338077,
flops_per_token_measured 158,075,904, 2076 steps at 262,144 tokens/update).

TWO constants, ONE conceptual change: remove a transformer block at HELD width.
  ASPECT_RATIO 64 -> 85   (6*85 = 510, which rounds up to model_dim 512 at HEAD_DIM 128)
  DEPTH        7  -> 6
ASPECT_RATIO exists only to keep model_dim at 512. Without it, DEPTH 6 at ASPECT_RATIO 64 gives
base_dim 384 and model_dim 384, which is a different and already-refuted candidate: launch 9
(cap_depth6_overshoot) measured exactly that at val_bpb 1.0626667248581214 and team capacity's
dead_ends.md closes it, attributing the loss to model_dim 384 rather than to depth 6. Holding
width keeps n_head 4, head_dim 128, kv_dim 512 and dmodel_lr_scale 1.224745 all at the
champion's values, and keeps every Muon per-shape multiplier unchanged, so the only thing that
changes is the number of blocks.

The queue item's predicted integers were priced against champion.md v6 (cap_mlp_ratio2,
23,069,198) and are superseded: the champion moved twice while the item sat in the queue
(-> 21,233,934 v7 -> 19,005,966 v8). The diff itself still applies verbatim, because v8
inherited ASPECT_RATIO 64 and DEPTH 7 unchanged. Rebased predictions, exact:

  blocks           7 x 2,097,152 -> 6 x 2,097,152        -2,097,152
  ve_gate          has_ve(i,7) = {0,2,4,6} (4 gates) -> has_ve(i,6) = {1,3,5} (3)   -128
  scalars          2*7 = 14 -> 2*6 = 12                        -2
  num_params_total 19,005,966 -> 16,908,684   (-11.03% vs champion, -66.41% vs the reference)

  Census check, which must reconcile against the frozen count_params():
    wte 4,194,304 + lm_head(512x256 map) 131,072 + transformer_matrices 12,583,296
    + scalars 12 = 16,908,684.

  flops_per_token_measured = 6*(matmul-charged params) + 12*n_head*head_dim*sum(span)
    matmul-charged = blocks 12,582,912 + ve_gates 384 + map 131,072 + the wte column slice the
                     classifier multiplies by, 8192*256 = 2,097,152   ->  14,811,520
    span at DEPTH 6 = [1024,1024,1024,2048,1024,2048] = 8192  (pattern[i%4] of "SSSL" with
                      window_sizes[-1] forced long; at DEPTH 7 it was 9216)
    = 6*14,811,520 + 6144*8192 = 88,869,120 + 50,331,648 = 139,200,768   (ceiling 239,078,400)

  Note the wte slice term: prepare.py's analytic formula excludes wte BY NAME while the dispatch
  counter charges BY USE, and this champion routes the classifier through wte[:, :256]. Omitting
  that 2,097,152 would mispredict the measured FLOPs by 12,582,912. Reconstructing the champion's
  own 158,075,904 the same way reproduces it exactly, which is what makes this prediction a check
  on the edit rather than a guess.

WHY THIS RUNG, AND THE HONEST CASE AGAINST IT. Depth at held width is the one structural rung this
run has measured a val_bpb *gain* on: launch 4 (cap_depth6_ar85, 1.0389276776694476) against
launch 2 (cap_depth7, 1.0423158898319644) is +0.0033882 of margin for -3,145,730 parameters. Two
things make this base different and neither is favourable:
  1. The champion already runs 2076 updates where launch 2 ran 843. The refund here is 18,875,136
     (-11.9%), and knowledge/flops-refund-return-is-sublinear.md measures the return on extra
     updates as decaying; 2076 is far down that curve, so the gain should not be assumed to repeat.
  2. The champion is much thinner than launch 2's base (MLP_RATIO 2 not 4, unembedding factored to
     rank 256), so the marginal block carries more of what is left.
Working in favour, and measured this cycle in results/bracket_cap_mlp_ratio1.md: a refund earned by
deleting whole blocks OVER-realises into wall clock (cap_depth7 converted 1.05 of nominal), whereas
a refund earned by making matmuls smaller under-realises (0.74-0.77). This candidate deletes a
block and leaves every matmul shape untouched, so it is the favourable kind. Pre-registered: the
champion holds 0.010257626743338077 of margin and I am willing to spend it on this rung, because a
loss here is the second reading of the depth axis on a thin base and a win is 16,908,684.

RIDERS, named rather than compensated:
  - has_ve parity flips from even to odd layers, so layer 0 stops reading the value residual and
    the last layer keeps it. Launch 4 already exercised this exact flip successfully.
  - The VE width assert kv_dim == n_embd still holds: n_head = 512/128 = 4, n_kv_head = 4,
    head_dim = 128, so kv_dim = 512 = n_embd.
  - Long-context layers move from {3,6} to {3,5}; the span sum falls 9216 -> 8192.
  - value_embeds stays an empty nn.ModuleDict and every surviving ve_gate is still fed, so the
    frozen count_params()/estimate_flops_analytic() indexing and MuonAdamW's no-None-guard grad
    stacking both remain satisfied.

--- inherited docstring of the base champion follows, unchanged ---

Experiment red_unembed_via_wte_r256_bhalf (team redundancy, proposed and run by
autoscts__params_gpu4). This is launch 11's candidate, financed. It composes TWO deltas that
were each measured against the SAME base, the cap_mlp_ratio2 champion (num_params_total
23,069,198, val_bpb 1.0440886169425738, flops_per_token_measured 169,872,384, 1008 steps),
which is what makes it the run's cleanest test of additivity rather than another guess:

  launch 11  the unembedding cut below      -4,063,232 params, val_bpb +0.0085292, 1045 steps
  launch 10  TOTAL_BATCH_SIZE 2**19->2**18          0 params, val_bpb -0.0167157, 1998 steps

Launch 11 was inadmissible on its own by 0.0026179 and nothing else: its target landed exactly
as predicted and every ceiling had room. Launch 10 moved val_bpb by 2.0x that whole debt at a
byte-identical parameter count, and could not KEEP because it ties the champion by
construction. Composed, the prediction is num_params_total 19,005,966 at val_bpb ~1.0359022,
i.e. 0.0140978 of slack -- and launch 8's additivity prediction on this substrate failed by
0.0032169, so that slack absorbs an additivity error of that size 4.4 times over.

WHY THIS PAIRING AND NOT A LARGER ONE. The batch lever buys optimizer updates, so it is worth
what updates are worth, which falls as a candidate already has more of them. Launch 10
measured it on a program running 1008 updates. This candidate runs 1045 -- the same regime, one
mechanism apart. The run's other unfinanced candidate (launch 9, 13,369,644 at val_bpb
1.0626667) already runs 1576 updates before any lever is applied, so it sits further down the
same diminishing curve and is the weaker transport case, not the stronger one. That is an
argument about where a measured lever applies, and it is why this candidate is the one worth a
launch first.

CHANGE 1 of 2: the unembedding is no longer a dedicated 8192 x 512 table. The classifier reads
the vocabulary geometry out of wte -- the only remaining vocabulary table -- through a learned
n_embd x logit_rank map, and lm_head now holds that map instead of a vocabulary:

    logits = F.linear(self.lm_head(x), wte.weight[:, :logit_rank]) * logit_rank**-0.5
    lm_head : nn.Linear(512, 8192) -> nn.Linear(512, 256)

This is the "factor" limb of H-redundancy, which knowledge/unqueued_axes.md records as
re-opened and unqueued. It is strictly more expressive than the measured tie: red_tie_lm_head
(launch 3) is this candidate with the map pinned to the identity and the rank pinned to
n_embd, and it cost +0.0061305 val_bpb. Measured at launch 11, learning the map did NOT come
in cheaper (+0.0085292), so rank and expressiveness are not where the price is -- which is
recorded in results/red_unembed_via_wte_r256.md and is why this candidate changes the financing
rather than the mechanism.

CHANGE 2 of 2: TOTAL_BATCH_SIZE 2**19 -> 2**18. At DEVICE_BATCH_SIZE 128 that makes
grad_accum_steps 1 instead of 2, so an optimizer update costs one microbatch instead of two and
the fixed 600 s clock returns roughly twice as many updates at half the tokens each. Launch 10
measured 1998 updates against 1008 at 523,763,712 total tokens against 528,482,304, i.e. the
same token count spent through twice as many updates. The CAUSE of its val_bpb gain is not
established by this run -- team throughput has thr_accum_sync_control queued to separate "more
updates helped" from "grad_accum 2 was itself lossy" -- and this candidate does not need the
cause settled, only the effect, which is measured.

Exact arithmetic (num_params_total and flops_per_token_measured are task-declared
deterministic and count_params is an exact sum, so these are predictions, not estimates -- a
different integer means the edit did not land). Neither is a function of the batch size, so
both are the integers launch 11 already confirmed:
  lm_head      4,194,304 -> 131,072              (8192x512 -> 256x512)
  census       wte 4,194,304 + lm_head 131,072 + transformer_matrices 14,680,576 + scalars 14
  num_params_total    23,069,198 -> 19,005,966   (-4,063,232, -17.61%; -62.24% vs reference)
  flops_per_token_measured = 6*(every weight a matmul reads) + 12*n_head*head_dim*sum(span)
    = 6*(14,680,576 + 131,072 + 8192*256) + 56,623,104 = 158,075,904   (ceiling 239,078,400)
    i.e. a refund of 11,796,480 (-6.94%): the 8192-side matmul narrows from 512 to 256 and
    the map's own matmul costs 786,432 back. The FLOPs probe reads 8 sequences whatever
    TOTAL_BATCH_SIZE is, so this integer is unchanged by change 2.
  flops_per_token_analytic (log only) = 145,492,992, which UNDER-reports the measured number
    by exactly 12,582,912 -- prepare.py's analytic formula excludes wte BY NAME while the
    dispatch counter charges it BY USE, and wte is now inside an F.linear. The gap being
    exactly 6*8192*256 is a free check that the logit path is the one described here.
  peak_vram_bytes ~ 33,186,000,000 (launch 11's 33,269,873,152 plus launch 10's -83,889,152),
    against a 47,198,976,512 ceiling. Both deltas are small and both are downward-safe.
  Predicted step-0 loss is again exactly ln(8192) = 9.010913, since the map is zero-initialised
    and the batch size does not touch initialisation.

WHAT A MISS WOULD MEAN, pre-registered. If val_bpb >= 1.05 with 0.0141 of predicted slack, the
failure is additivity itself, not this mechanism: it would mean the batch lever's -0.0167157
does not transport to a program whose own quality cost is already paid at the same step count.
That is a result about every paired candidate in all three teams' queues, and it should be
recorded as such rather than charged to the unembedding cut. If it lands, the remaining
question on this axis stays the one launch 11 opened: the 150x speed mismatch between
EMBEDDING_LR 0.7348 on the shared table and UNEMBEDDING_LR 0.0049 on the map.

Two decisions inherited unchanged from launch 11, recorded rather than left implicit:
  - THE LOGIT SCALE IS DERIVED, NOT TUNED. wte is initialised at std 1.0 because the input
    path rms-norms it, so its scale is free there and not free in a logit path (this is the
    hazard results/red_tie_lm_head.md hit, and it fixed it with a learned temperature). The
    factor logit_rank**-0.5 = 1/16 exactly restores the champion's logit scale per unit of
    map: sigma_logit was sqrt(n_embd)*sigma_lm_head and is now
    sqrt(logit_rank)*sqrt(n_embd)*sigma_map*std(wte). It therefore also matches the
    champion's per-step logit displacement under the same UNEMBEDDING_LR, so no LR was
    re-picked and setup_optimizer is byte-identical to the champion's.
  - THE MAP IS ZERO-INITIALISED, this file's own convention for an output projection
    (attn.c_proj, mlp.c_proj). Step-0 logits are exactly 0, so step-0 loss is exactly
    ln(8192) = 9.010913 and the softcap starts unsaturated by construction. dead_ends.md
    records that re-picking the init std or the group LR re-tests a decision
    red_tie_lm_head already spent, so this experiment removes both instead.

PRE-REGISTERED, in case the gate is missed. wte's first 256 columns now serve two consumers at
two different speeds: the input/value-residual path at EMBEDDING_LR 0.6*1.2247 = 0.7348 and
the classifier, whose own map moves at UNEMBEDDING_LR 0.004*1.2247 = 0.0049. If val_bpb >=
1.05, the next rung is parameter-count-neutral: split those columns into their own tensor and
give it the unembedding group's LR. That separates "the vocabulary geometry cannot be shared"
from "the shared table moves too fast to be a classifier basis". Do NOT read a miss as closing
the factor limb; the fallback rung (an independent rank-256 lm_head factorisation, -1,966,080
with the same refund) is untouched by either answer.

Inherited from the champion and unchanged, recorded because each would otherwise cost a
charged launch:
  - self.value_embeds must survive as an empty nn.ModuleDict. The frozen prepare.py indexes
    model.value_embeds in count_params() and estimate_flops_analytic(), and init_weights()
    and estimate_flops() iterate its .values().
  - Every ve_gate must stay fed. ve_gate exists iff has_ve(layer_idx, n_layer) and lives in
    transformer.h, so it is a Muon parameter, and MuonAdamW._step_muon stacks p.grad with no
    None guard. Sourcing ve for exactly the has_ve layers keeps all of them getting gradient.
  - The empty value_embeds param group is dropped rather than passed empty; the
    parameter-partition assert is left as written, and still holds -- lm_head is still exactly
    one tensor in exactly one group, so no group was added, removed or re-weighted.
  - wte's absolute magnitude is load-bearing: the value residual consumes wte(idx)
    unnormalised. This candidate does not touch that path: the slice is taken in the logit
    path only, and the value residual still reads the full-width unnormalised lookup.
  - The in-script model.estimate_flops() print is left as written. It is the MFU log's input,
    not the scored metric, and it now under-reports by 12,582,912 for the reason above.
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
    mlp_ratio: float = 4.0
    logit_rank: int = 256
    mlp_narrow_layers: tuple = ()
    mlp_narrow2_layers: tuple = ()
    mlp_narrow3_layers: tuple = ()
    mlp_narrow4_layers: tuple = ()
    mlp_narrow5_layers: tuple = ()


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
        # 32 -> 8 (red_ve_gate_ch8_on_v19). The gate is a linear map from the first
        # ve_gate_channels channels of the normalised residual to one multiplier per kv head, and it
        # is zero-initialised, so at step 0 it is exactly 1.0 whatever this width is. 8 is the widest
        # rung that leaves the gate's optimizer untouched: (4, C) keeps _step_muon's
        # max(1.0, shape[-2]/shape[-1])**0.5 factor at 1.0 for C >= 4 and keeps muon_step_fused's
        # red_dim at -2 for C > 4, so C = 8 changes no LR, no reduction axis and no group count.
        # The forward slice below reads this same attribute, so the constant is the only site.
        self.ve_gate_channels = 8
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


class MLP(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        # mlp_ratio is a ratio, not an integer count of copies of n_embd: the hidden width is
        # quantised to 6*2*n_embd = 6,144 parameters per unit of width on this six-layer model,
        # so a ratio that does not land on a whole width fails loudly instead of silently
        # rounding. Mechanism introduced by launch 22 (cap_mlp_hidden960, team capacity) and
        # ported here unchanged from that candidate's source.
        hidden = config.mlp_ratio * config.n_embd
        assert float(hidden).is_integer(), \
            f"mlp_ratio {config.mlp_ratio} x n_embd {config.n_embd} = {hidden} is not integral"
        hidden = int(hidden)
        # Per-layer width, transplanted verbatim in mechanism from launch 59
        # (cap_mlp_split864_on_v21, autoscts__params_gpu1, team capacity), which measured this
        # mechanism at 0.002719 per 1M -- the cheapest parameter mass anyone has priced in this
        # run and 2.6x cheaper than the same mass taken as a UNIFORM width (launch 52, 0.007005
        # per 1M). A uniform step is quantised to 98,304 parameters on this six-layer model,
        # which is 1.6x this champion's whole margin; one 16-element step on three layers is
        # 49,152 and every width stays in the free (multiple-of-16-elements) alignment class.
        if layer_idx in config.mlp_narrow_layers:
            hidden -= MLP_NARROW_ELEMENTS
        # A SECOND narrowing step, so two widths can coexist without a second ratio. Mechanism from
        # autoscts__params_gpu1's launch 73 (cap_win_long_half_mlp864_even_on_l70), which spelled it
        # exactly this way; the set is {0,2}, both UNGATED, because launch 70 measured the has_ve
        # layers at 1.038160e-08/param against launch 67's 7.65679e-09 on the ungated ones (1.356x),
        # and because launch 70's gated rate was measured with THREE separate gate maps while this
        # base has ONE shared gate function -- so a {1,3,5} limb would add an unmeasured
        # width-against-shared-gate interaction to a known premium. This candidate's whole purpose is
        # to price THIS mechanism on THIS base with no span limb in the way, which is the one
        # measurement that separates the two published readings of launch 73's residual.
        if layer_idx in config.mlp_narrow2_layers:
            hidden -= MLP_NARROW2_ELEMENTS
        # A THIRD narrowing step, spelled exactly as autoscts__params_gpu3 spelled it for launch 84
        # (red_mlp_l4_rung2_deepbase_on_v35) so that the two rows are read on one mechanism. It stacks
        # on the two above rather than replacing them, which is what lets ONE subset of an already
        # narrowed set take a further rung: {0,2} here are in mlp_narrow_layers AND in
        # mlp_narrow2_layers, so 896 - 16 - 32 - 16 = 832, while layer 4 stays at 848 and the gated
        # layers {1,3,5} stay at 880. The set is the UNGATED pair, so this adds no
        # width-against-shared-gate interaction and the single shared ve_gate function is untouched.
        if layer_idx in config.mlp_narrow3_layers:
            hidden -= MLP_NARROW3_ELEMENTS
        # A FOURTH narrowing step, an exact mirror of the third above and stacking on it the same
        # way. It exists so that ONE ungated layer can go a rung below the other two: layer 0 is in
        # mlp_narrow_layers, mlp_narrow2_layers and mlp_narrow3_layers, so
        # 896 - 16 - 32 - 16 - 16 = 816, while layers 2 and 4 stay at 832 and the gated {1,3,5}
        # stay at 880. This LOWERS the network minimum from 832 to 816, so it is a DEEPENING step
        # and is priced on the k-ladder (k=4), NOT at launch 90's levelling rate 3.6646670e-09 --
        # see knowledge/a-width-granule-has-three-prices-by-where-it-lands-on-the-minimum.md, whose
        # first named misuse is exactly that substitution.
        if layer_idx in config.mlp_narrow4_layers:
            hidden -= MLP_NARROW4_ELEMENTS
        # A FIFTH narrowing step, spelled exactly as autoscts__params_gpu4 spelled it for launch 96
        # (red_mlp800_ungated_pair_resid_half_on_l95) so the two rows read on ONE mechanism, and stacking
        # on the four above the same way. It exists so that ONE ungated layer can go a rung below the
        # other two: layer 0 is in mlp_narrow_layers, mlp_narrow2_layers, mlp_narrow3_layers AND
        # mlp_narrow4_layers, so 896 - 16 - 32 - 16 - 32 - 16 = 784, while layers 2 and 4 stay at 800 and
        # the gated {1,3,5} stay at 880. Widths become [784,880,800,880,800,880], floor 784.
        # THIS IS THE SAME MECHANISM, THE SAME LAYER AND THE ADJACENT RUNG OF LAUNCH 92, which is the
        # closest-matched precedent this run contains: launch 92 (thr_mlp816_ungated_l0_on_v39,
        # autoscts__params_gpu6) introduced mlp_narrow4_layers = (0,) to take layer 0 from 832 to 816 --
        # one 16-element granule, one ungated layer, lowering the floor -- and measured -0.0000647383,
        # a GAIN of 0.40 spreads, against launch 90 on the same bytes.
        # 784 = 16 x 49 and 784 x 2 = 1,568 bytes = 32 x 49, so 784 stays in the SAME 32-byte free
        # alignment class as 800/816/832/848/864/880/896 (launch 52 -0.041% at 32-byte, launch 54
        # -0.992% at 16-byte, launch 51 -4.258% at 8-byte), so no alignment penalty is bought. The
        # hidden % 16 == 0 assert below holds at 49 x 16, and 784 % 8 == 0 keeps every matmul dimension
        # a multiple of 8.
        if layer_idx in config.mlp_narrow5_layers:
            hidden -= MLP_NARROW5_ELEMENTS
        assert hidden % 16 == 0, \
            f"MLP hidden {hidden} at layer {layer_idx} is not a multiple of 16 elements; " \
            f"launch 54 measured a 0.992% wall-clock penalty for exactly that"
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
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config, layer_idx)

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
        # One query projection for the whole stack. An attention score is a bilinear form,
        # score_ij = x_i (W_q W_k^T) x_j^T, so it stays layer-specific through the six
        # per-layer c_k even when W_q does not; c_v, the value residual and c_proj are
        # untouched. Share the MODULE, never the Parameter: the model is built on meta and
        # then to_empty()ed, and nn.Module._apply rebuilds _parameters[key] on each module
        # it visits, so an aliased Parameter in six distinct Linears would be un-aliased
        # there while one shared submodule keeps a single _parameters dict.
        # h[1:-1], not h[1:]: the LAST layer keeps its own c_q. Launch 23 measured the six-way
        # merge at 0.004297 val_bpb per 1M and it was 262,144 parameters too large to fit the
        # margin; one layer is the finest granularity this axis has.
        shared_c_q = self.transformer.h[0].attn.c_q
        for block in self.transformer.h[1:-1]:
            block.attn.c_q = shared_c_q
        # One VALUE projection serves layers 1 and 3. Same aliasing discipline as the c_q share
        # above -- share the MODULE, never the Parameter, because the model is built on meta and
        # then to_empty()ed and nn.Module._apply would un-alias a raw Parameter.
        # Layers 1 and 3 are chosen because has_ve(i, 6) is i % 2 == 1, so both are value-residual
        # layers and each keeps its OWN ve_gate: layer i computes v = c_v_shared(x) + gate_i * ve.
        # c_v is the only one of the three never-priced attention roles that has a per-layer
        # compensating path already in the model, and launch 50 measured that path as worth
        # 0.000821891457169821 of val_bpb for 96 parameters. c_k has no such path and merging it
        # beside the five-way-shared c_q would make the whole QK bilinear form layer-invariant.
        # 512 -> 512, so no dimension leaves the 8-grid and launch 51's alignment tax cannot apply;
        # counted FLOPs are use-weighted, so an aliased map is still charged per use and
        # flops_per_token_measured is unchanged, as launch 23 measured directly.
        self.transformer.h[3].attn.c_v = self.transformer.h[1].attn.c_v
        # One KEY projection serves layers 3 and 5. The sentence above says merging c_k "beside the
        # five-way-shared c_q" makes the QK form layer-invariant, and that is exactly why the pair is
        # {3,5} and not any pair inside {0,1,2,3,4}: shared_c_q is h[0].attn.c_q assigned to h[1:-1],
        # so layers 0-4 hold ONE c_q and layer 5 owns its own. score_ij = x_i (W_q W_k^T) x_j^T, so a
        # c_k pair drawn from {0..4} would share BOTH factors and give two layers a bit-identical QK
        # map -- a function-class restriction in its own price band, not a share. Layer 5 keeps a
        # distinct W_q, so layer 3's form is shared_c_q^T c_k_shared and layer 5's is c_q_5^T
        # c_k_shared: two distinct bilinear forms over one key basis. Both 3 and 5 are has_ve layers
        # (has_ve(i,6) is i%2==1), which holds the value-residual property FIXED against launch 57's
        # c_v merge on {1,3} and makes this the controlled second point on "first merge, fresh role".
        self.transformer.h[5].attn.c_k = self.transformer.h[3].attn.c_k
        # One attention OUTPUT projection serves layers 2 and 4. Same aliasing discipline as the three
        # shares above -- share the MODULE, never the Parameter, because the model is built on meta and
        # then to_empty()ed and nn.Module._apply would un-alias a raw Parameter.
        # WHY attn.c_proj, AND WHY THE PAIR IS {2,4}. attn.c_proj is the third and last unpriced
        # attention role: c_v is measured at first merge (launch 57, 0.004208 per 1M), c_k at first
        # merge (launch 64, 0.007030 per 1M), c_q at five-way (launch 30) and at the sixth-map fold
        # (launch 34). {2,4} is chosen so that the ONLY thing that moves against the run's one measured
        # pair-share is the role: launch 38 shared mlp.c_proj on THIS EXACT PAIR, on a base that
        # already carried this same h[1:-1] c_q share, and measured 0.0074444 per 1M. So the pair, the
        # layer indices and the c_q stacking are held FIXED and mlp.c_proj -> attn.c_proj is the whole
        # difference. Both layers are non-ve (has_ve(i,6) is i%2==1) and both were narrowed to
        # mlp_hidden 880 by launch 66, which holds the value-residual property and the MLP width fixed
        # across the pair too. attn.c_proj is zeros_-initialised in init_weights, so an aliased module
        # draws NO RNG and the init of every tensor after it is bit-identical -- the only pure share of
        # the three attention roles, and the reason step-0 loss is a free falsifier here. 512 -> 512,
        # so no dimension leaves the 8-grid and launch 51's alignment tax cannot apply; counted FLOPs
        # are charged per USE, so flops_per_token_measured is unchanged while the analytic estimate,
        # which counts parameters, drops by exactly 6 * 262,144.
        self.transformer.h[4].attn.c_proj = self.transformer.h[2].attn.c_proj
        # One ve_gate FUNCTION serves all three has_ve layers, 3 distinct gate maps -> 1, -64.
        # Mechanism transplanted verbatim from launch 71 (red_ve_gate_share_135_on_v29,
        # autoscts__params_gpu4, team redundancy), which measured it on launch 68's bytes at
        # val_bpb -0.0001219240431327 -- a GAIN, 0.749x the same-program spread, i.e. a draw --
        # with num_steps +2, flops_per_token_measured bit-identical (the gate is charged per USE and
        # is still used by three layers), flops_per_token_analytic -384 = 6 x 64, and step-0 AND
        # step-1 loss bit-identical. That last one was their registered risky claim and it held: at
        # step 0 every attn.c_proj and mlp.c_proj is zeros_, so the gradient reaching ve_gate carries
        # a c_proj^T = 0 factor and is exactly zero, and the gate cannot move at step 1 whether it is
        # shared or not.
        # WHY THIS IS THE WHOLE CANDIDATE. Champion v30 holds 0.0000766528514684 = 0.471x the
        # instrument's own spread, which buys 10,012 parameters at the cheapest measured width rate
        # against a smallest MLP granule of 16,384: nothing in the priced inventory fits. This is the
        # only mass in that inventory that costs NOTHING at any margin, so it is the only cut that
        # does not have to borrow the number a launch is trying to find. Launch 73 is what made that
        # distinction expensive rather than academic -- it measured the run's last zero-parameter
        # lever ON these bytes at a NET -0.00017831, a sign reversal from the +0.00029343 launch 72
        # measured one width-step wider, and it took my own paired candidate down with it.
        # Their axis is closed in both directions (launch 50 gate width 32 -> 8 a gain, launch 71
        # gate layer count 3 -> 1 a draw, cap_ve_gate_off_on_v21 deleting it +0.0023857): only the
        # gate's PRESENCE is load-bearing, and every gate here stays fed at ve_gate_channels 8.
        # Share the MODULE, never the Parameter -- the model is built on meta and then to_empty()ed,
        # and nn.Module._apply rebuilds _parameters[key] on every module it visits, so an aliased raw
        # Parameter in three distinct Linears would be un-aliased there while one shared submodule
        # keeps a single _parameters dict. init_weights zeros_ every non-None ve_gate, so aliasing
        # draws NO RNG and the init of every tensor after it is bit-identical.
        self.transformer.h[3].attn.ve_gate = self.transformer.h[1].attn.ve_gate
        self.transformer.h[5].attn.ve_gate = self.transformer.h[1].attn.ve_gate
        # Unembedding: no dedicated vocab x n_embd table. The classifier reads the
        # vocabulary geometry out of wte -- the table the model already has -- through a
        # learned n_embd x logit_rank map, so lm_head holds the map and not the vocabulary.
        assert 0 < config.logit_rank <= config.n_embd, \
            f"logit_rank {config.logit_rank} must be in (0, n_embd={config.n_embd}]"
        self.lm_head = nn.Linear(config.n_embd, config.logit_rank, bias=False)
        self.logit_rank = config.logit_rank
        # wte is initialised at std 1.0 because the input path rms-norms it, so its scale is
        # free there and NOT free in a logit path. This factor restores the per-unit-of-map
        # logit scale the dedicated lm_head had: sigma_logit was sqrt(n_embd)*sigma_lm_head
        # and is now sqrt(logit_rank)*sqrt(n_embd)*sigma_map*std(wte). Exactly 1/16 at 256.
        self.logit_scale = config.logit_rank ** -0.5
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Value embeddings: no dedicated tables. The value residual is read out of wte, the
        # table the model already has, so the four 8192 x 512 VE tables are gone.
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        # The shared lookup is consumed at the VE path's width with no projection, which is
        # only sound while that width is the embedding width. It is, whenever this config
        # family sets n_kv_head == n_head; assert rather than reshape a wrong tensor.
        assert kv_dim == config.n_embd, f"VE width {kv_dim} != embedding width {config.n_embd}"
        # Keep the attribute: the frozen count_params() and estimate_flops_analytic() both
        # index model.value_embeds, and init_weights/estimate_flops iterate .values().
        self.value_embeds = nn.ModuleDict()
        self.ve_layers = {i for i in range(config.n_layer) if has_ve(i, config.n_layer)}
        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        # The map is an output projection, so it takes this file's own convention for one
        # (attn.c_proj, mlp.c_proj): zeros. Step-0 logits are then exactly 0 and step-0 loss
        # is exactly ln(vocab_size) = 9.010913, which removes the init scale from this
        # experiment instead of re-picking it -- dead_ends.md records that a re-pick of the
        # init std or the group LR re-tests a decision red_tie_lm_head already spent.
        torch.nn.init.zeros_(self.lm_head.weight)
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

    # base 100000, not 10000: launch 77 (red_rope_base_100k_on_l70, autoscts__params_gpu3) measured this
    # one literal at val_bpb -0.0012993172 for ZERO parameters, ZERO counted FLOPs and ZERO allocation on
    # champion v30's bytes -- 3.0x this champion's whole margin and the largest zero-parameter gain in the
    # run. Launch 74 measured the DOWN rung (base 1024) at +0.0049632, so the axis is bracketed and this
    # is its measured favourable side. Both call sites (__init__ line ~1614 and init_weights line ~1650)
    # take the default, so this single literal moves the buffer and its re-initialisation together; the
    # tables are recomputed identically in both places, exactly as launch 77 spelled it. Nothing else in
    # the file reads `base`. This lever is carried as UPSIDE and is NOT financing the MLP cut above --
    # that cut is covered 3.85x by margin alone.
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
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
        # // 4, not // 2: launch 61 (thr_win_quarter_on_v21, gpu5) measured this ONE literal at
        # -0.0021325632 of val_bpb for ZERO parameters -- +123 steps (+5.0847% of total_tokens) at
        # 0.00041940 per 1%, which is the run's standing token-income rate to three digits, so the
        # span-quality term at 512 is indistinguishable from zero. It is the lever this row's
        # precondition asked for, and it is banked here, not re-priced.
        short_window = long_window // 8
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
        # Empty by construction here. Kept in the partition arithmetic below, which still
        # holds exactly: self.parameters() lost precisely the four tensors that left.
        assert not value_embeds_params, "value_embeds must be empty in this candidate"
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
            # thr_mlp800_ungated_triple_resid_half_on_l95: 0.01 -> 0.005, autoscts__params_gpu3's
            # launch-93 lever, MEASURED on champion v38's bytes at the declared tie 14,057,516 --
            # val_bpb 1.0486278669169802 against launch 89's 1.0490566936955112 on the SAME integer
            # and the SAME bytes = -0.000428826779 = 2.633 same-program spreads for ZERO parameters.
            # Their axis is monotone over three rungs on one base (0.005 / 0.01 / 0.02 ->
            # 1.0486278669169802 / 1.0490566936955112 / 1.0496931155182745), so the low side is not
            # bracketed; this candidate banks the MEASURED rung 0.005 and deliberately not the
            # unmeasured 0.0025, which is theirs and which an extrapolation cannot price.
            # x0_params below is untouched: launch 94 measured its beta1 0.96 as LOAD-BEARING
            # (+2.653 spreads at 0.8), and this literal moves the GENTLE group's rate alone.
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.005, betas=adam_betas, eps=1e-10, weight_decay=0.0),
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

        tok = self.transformer.wte(idx)
        x = norm(tok)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            # One lookup, reused by every VE layer: four retained (B, T, n_embd) bf16
            # activations collapse to one. Per-layer specialisation stays in ve_gate.
            ve = tok if i in self.ve_layers else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        x = norm(x)

        softcap = 15
        # Unembedding through wte: map x into the rank-logit_rank vocabulary basis, then read
        # the logits off wte's first logit_rank channels. A slice issues no matmul and holds
        # no tensor of its own; the channel choice is arbitrary because wte is learned from
        # scratch under one LR from an isotropic init, so the first logit_rank columns are
        # exchangeable with any others at initialisation.
        logits = F.linear(self.lm_head(x), self.transformer.wte.weight[:, :self.logit_rank])
        logits = logits.float() * self.logit_scale
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
ASPECT_RATIO = 85       # model_dim = depth * ASPECT_RATIO (6*85 = 510 -> 512, width held)
HEAD_DIM = 128          # target head dimension for attention
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
MLP_RATIO = 1.75        # MLP hidden width as a multiple of model_dim: 1.75*512 = 896 (shipped 4;
                        # champion v15 carries 1.875 = hidden 960, which launch 34 put there). The
                        # NEXT rung down, and unlike 960 it is 7*128 = 128-ALIGNED, so
                        # knowledge/mlp-hidden-refund-is-quantised-to-128.md predicts it returns 0.70
                        # of its counted-FLOPs refund in wall clock instead of the 0.10-0.23 an
                        # unaligned width returns. This rung's price is measured on this substrate by
                        # the v10 -> v11 pair (results/cap_mlp896_warmdown7.md): net +0.0000483 with
                        # WARMDOWN_RATIO 0.7 added in the same step, so 0.0013040-0.0017797 once that
                        # lever's own 0.0012557-0.0017314 is subtracted. v15 already holds
                        # WARMDOWN 0.7 and EMBEDDING_LR 0.3, so this rung is financed by MARGIN
                        # (0.0028007832340883) and not by a new lever.
MLP_NARROW_LAYERS = (0, 1, 2, 3, 4, 5)  # cap_mlp_gated880_on_l68: the three has_ve layers {1,3,5}
                        # JOIN the narrowed set, so all six sit at 880. Launch 67 measured the ungated
                        # set at 7.6568e-09 per parameter and refuted the per-layer discount the
                        # comment below records; what it could not see is whether the gated layers
                        # cost the same. Four launches (59, 66, 67 and launch 68's own base) have
                        # spared {1,3,5} on the strength of an assumption that has never been
                        # measured. Break-even here is 1.03133e-08/param = 1.347x the ungated rate,
                        # so the gate itself answers it. Original comment kept below unedited:
                        # PER-LAYER width, mechanism transplanted from launch 59
                        # (cap_mlp_split864_on_v21, autoscts__params_gpu1, team capacity), which
                        # measured it at 0.002719 per 1M -- the cheapest parameter mass priced in
                        # this run, 2.6x under the same mass taken as a uniform width (launch 52,
                        # 0.007005 per 1M), and 0.82x the same-program val_bpb spread. The layer set
                        # is theirs and their reason is measured: has_ve(i,6) is i%2==1, so {1,3,5}
                        # own a ve_gate and {0,2,4} do not; launch 50 priced that gate at 0.0008219
                        # of val_bpb for 96 parameters, so the ungated layers are where spare MLP
                        # capacity should sit. Their launch 59 found narrowing exactly these three
                        # free to the instrument's resolution AND 0.000766 BETTER on final train
                        # loss. Three layers x one step x 2 matrices x 512 = -49,152.
MLP_NARROW2_LAYERS = (0, 2, 4)  # thr_rope100k_mlp864_l4_on_v32: layer 4 JOINS the second narrowing
                        # step, so all three UNGATED layers sit at 864 and widths become
                        # [864,880,864,880,864,880]. -16,384 -> 14,139,436. The rate is not transported:
                        # launch 78 measured THIS mechanism on THIS layer class one base back at
                        # 6.813e-09/param net (0.00022326 for 32,768 on {0,2}), so 16,384 costs
                        # +0.0001116 against 0.0004294943529 of margin -- cover 3.85x, 1.95x the
                        # same-program spread left over, and the candidate is eligible even if the
                        # rotary literal below delivers nothing at all. 864 = 16 x 54 and 864 x 2 =
                        # 1,728 = 32 x 54, so layer 4 stays in the 32-byte free alignment class.
                        # Layer 4 is UNGATED (has_ve(i,6) is i%2==1 -> {1,3,5}), so this adds no
                        # width-against-shared-gate interaction: the gated layers keep 880 and the one
                        # shared ve_gate function is untouched. Original comment kept below unedited:
                        # thr_mlp864_pair_on_v31: a SECOND narrowing step on the two UNGATED
                        # layers {0,2}, hidden 880 -> 864, -32,768 -> 14,155,820. Sized at 32,768 and not
                        # 16,384 or 49,152 for two reasons that point the same way. FINANCED UNDER EVERY
                        # LIVE READING of the width rate: 0.000194 (L67 net), 0.000251 (L67 gross),
                        # 0.000340 (L70 gated), 0.000514 (the L72/L73 blend at span 1024) against
                        # 0.0006527526878646 of margin -- cover [1.27x, 3.36x], an interval that does not
                        # contain 1. RESOLVABLE: those readings differ by TWO same-program spreads at this
                        # mass and by only ONE at 16,384, where the cheapest reading's delta would be 0.60x
                        # the spread and the rate therefore unpublishable. 49,152 is refused: cover
                        # [0.85x, 2.24x] contains 1, so it would be decided by the number the launch
                        # exists to find.
MLP_NARROW2_ELEMENTS = 32 # thr_mlp848_ungated_triple_on_v34: a SECOND rung of the same free alignment
                        # class on the same three UNGATED layers {0,2,4}: 880 - 32 = 848, so widths
                        # become [848,880,848,880,848,880] and -49,152 -> num_params_total 14,090,284.
                        # 848 = 16 x 53 and 848 x 2 = 1,696 bytes = 32 x 53, so 848 stays in the SAME
                        # 32-byte free alignment class as 864, 880 and 896 (launch 52 measured -0.041%
                        # at 32-byte against launch 51's -4.258% at 8-byte and launch 54's -0.992% at
                        # 16-byte), so this rung buys no alignment penalty.
                        #
                        # PRICED ON THE SECOND-RUNG RATE, WHICH IS NOT THE FIRST-RUNG RATE. Launches 80
                        # and 81 share base v32 AND the same rotary lever and differ only in how they
                        # spend width, so together they separate the two rungs. Solving them with launch
                        # 78's own first-rung r1 = 6.813e-09 gives lever*f = 0.0011678750 (f = 0.8988,
                        # from launch 80) and then r2 = 1.4231e-08/param (from launch 81) -- the SECOND
                        # rung is 2.09x the first. The width axis is CONVEX, which no single launch in
                        # this run could show and which my own v34 pricing did not know.
                        #
                        #     49,152 at r2 1.4231e-08  = +0.0006995  cover 2.12x, leftover 4.83 spreads
                        #     49,152 at the crude marginal 2.1649e-08 = +0.0010641  cover 1.40x, 2.59
                        #
                        # against v34's margin 0.0014857452 (9.12 spreads). FINANCED UNDER BOTH READINGS
                        # and the leftover clears the instrument by >= 2.59 spreads under the dearest --
                        # which is the test gpu1 used to reject -49,152 at v32's margin (0.33 spreads
                        # left over). That rejection was correct THEN; launch 80 tripled the margin and
                        # this is the same cut re-priced on the new frontier, not a dispute with it.
                        #
                        # Original comment kept unedited: one step of the FREE alignment class, on top
                        # of the first: 880 - 16 = 864 = 16 x 54, and 864 x 2 = 1,728 bytes = 32 x 54,
                        # so 32-BYTE aligned. The graded penalty (launch 51 -4.258% at 8-byte, launch 54
                        # -0.992% at 16-byte, launch 52 -0.041% at 32-byte) puts 864 in the same free
                        # class as 880 and 896.
MLP_NARROW3_LAYERS = (0, 2, 4)  # thr_mlp832_ungated_triple_on_l89: LAYER 4 JOINS the third rung, so
                        # all three UNGATED layers {0,2,4} sit at 832 and the gated {1,3,5} stay at
                        # 880: widths [832,880,832,880,832,880], -16,384 -> 14,041,132. This is a
                        # LEVELLING step, not a deepening one: base-min is 832 before and after, so
                        # it is the cheaper side of a-cut-price-is-convex-in-base-thinness.md, which
                        # I price at 1.0x and do not claim. Rate MEASURED at zero transport by the
                        # launch 86 / launch 89 pair (those bytes differ by exactly this mechanism):
                        # +0.0005353803923215938 for -32,768 over two ungated layers at pre-step
                        # width 848 = 1.633851e-08/param, so 16,384 costs +0.0002676902 against the
                        # bare margin 0.000943306304 -- cover 3.52x, leftover 4.15 spreads.
                        # Original comment kept unedited from here down:
                        # cap_scalar_lr_bank_mlp832_pair_on_v36: a THIRD rung on the two UNGATED
                        # layers {0,2}, 848 -> 832, widths [832,880,832,880,848,880] and -32,768 ->
                        # num_params_total 14,057,516. Layer 4 is deliberately EXCLUDED: taking all three
                        # ungated layers would be -49,152, which costs 0.0008865 at the rate below against
                        # champion v36's bare margin 0.000817588907977 -- cover 0.92x, ineligible by 0.42
                        # spreads if the SCALAR_LR lever below delivers nothing. That is the shape
                        # knowledge/a-lever-may-only-be-upside-never-collateral.md forbids and the shape
                        # this champion's own financing_correction records v36 itself falling into.
                        #
                        # PRICED ON A MEASUREMENT AT THIS BASE'S OWN THINNESS, NOT A TRANSPORTED RATE.
                        # autoscts__params_gpu3's launch 84 took a 16-element step on an UNGATED layer at
                        # span 256 on a base whose thinnest layer was 848 -- champion v36's thinness, its
                        # layer class and its span -- and measured 1.803539e-08/param (0.0002954917916993
                        # for 16,384). So 32,768 costs +0.0005910 against 0.000817588907977: cover 1.38x
                        # with 1.39 same-program spreads left over, WITH THE LEVER AT EXACTLY ZERO.
                        #
                        # k INDEXES THE PRE-STEP WIDTH, which is why this is a k=3 step and not a k=4 one.
                        # Launch 83 stepped its base's OWN thinnest layers 864 -> 848 at base-min 864 and
                        # measured 1.359367e-08 = 2.00x r1, the k=2 rate, not the k=3 rate its post-step
                        # width would imply. This step likewise takes the min down (848 -> 832). The
                        # pessimistic k=4 reading ~2.18e-08 is carried anyway: 32,768 costs +0.0007144,
                        # cover 1.14x. The cover interval [1.14x, 1.38x] EXCLUDES 1 on the bare margin.
                        #
                        # 832 = 16 x 52 and 832 x 2 = 1,664 bytes = 32 x 52, so 832 stays in the SAME
                        # 32-byte free alignment class as 848, 864, 880 and 896 (launch 52 -0.041% at
                        # 32-byte, launch 54 -0.992% at 16-byte, launch 51 -4.258% at 8-byte).
MLP_NARROW3_ELEMENTS = 16 # one further step of the free alignment class, on top of the first two
MLP_NARROW4_LAYERS = (0, 2, 4)  # thr_mlp816_ungated_l0_on_v39: ONE ungated layer goes a rung below the
                        # other two, 832 -> 816, widths [816,880,832,880,832,880], -16,384 ->
                        # 14,024,748. Layer 0 is UNGATED and at span 256, the cheapest layer class
                        # this run has priced. This step LOWERS the minimum, so it is priced on the
                        # DEEPENING ladder at k=4 and NOT at launch 90's levelling rate: cover is
                        # 3.30x at the k3 rate held flat, 2.46x at gpu3's power law and 1.98x at a
                        # LINEAR-in-k reading that every measured increment contradicts, with the
                        # leftover clearing the instrument at every end. Break-even needs k4 to be
                        # 5.391018e-08 = 3.30x k3 = 7.91x r1, i.e. the ladder reversing from
                        # decelerating to explosive in one rung.
                        # Only ONE layer: all three at 816 is -49,152 at cover [0.66x, 1.10x],
                        # which contains 1 and is refused on gpu1's test.
MLP_NARROW4_ELEMENTS = 32 # thr_mlp800_ungated_triple_resid_half_on_l95: a FIFTH rung of the same
                        # free alignment class, taken as an element count rather than a new set
                        # because mlp_narrow4_layers is already (0, 2, 4). All three UNGATED layers
                        # step 816 -> 800 together: 896 - 16 - 32 - 16 - 32 = 800 = 16 x 50, and
                        # 800 x 2 = 1,600 bytes = 32 x 50, so 800 is 32-BYTE aligned and 800 % 8 == 0
                        # -- the same free class as 816/832/848/864/880/896, so launch 54's 0.992%
                        # wall-clock penalty (16-byte) and launch 51's 4.258% (8-byte) cannot apply.
                        # -49,152 -> num_params_total 13,942,828. Priced on the MATCHED-mechanism
                        # ladder (one granule on each of the three ungated layers, uniform floor
                        # before and after): 1.359367e-08 (k2, L80->L83), 1.211390e-08 (k3,
                        # L86->L90), 6.990839e-09 (k4, L90->L95) -- monotone DECREASING, every cell
                        # above the instrument. Bare cover [0.81x, 1.57x] CONTAINS 1, so this row is
                        # declared financed on gpu3's launch-93 resid lever below and needs it to
                        # deliver f >= 0.300 at the dearest matched rate. NOT priced at my own
                        # at-floor 3.664667e-09, which gpu1's launch 95 measured over-predicting by
                        # 3.40x, and NOT at the single-granule non-uniform-base 1.803539e-08.
MLP_NARROW5_LAYERS = (0, 2)  # red_mlp784_ungated_pair_on_v44: layer 2 JOINS layer 0 at the fifth rung,
                        # 800 -> 784, widths [784,880,784,880,800,880], floor 784 with layer 4 the only
                        # ungated layer above it, -16,384 -> num_params_total 13,910,060.
                        # THIS IS A LEVELLING STEP, the run's cheapest and best-measured position class:
                        # layer 2 lands ON the floor 784 that launch 98 established, it does not lower it.
                        # Four levelling readings exist and all four are inside the same-program
                        # instrument 0.0001628969454063: +0.0000180 (L83/L84), +0.0000600 (L89->L90),
                        # -0.0001208 (L96->L97), and launch 98's own +0.0000060 deepening cell as the
                        # adjacent bound. Dearest levelling reading +0.0000600 -> cover 5.05x on the
                        # MEASURED margin 0.0003030985830 = 1.8607 spreads; dearest single-granule
                        # reading of ANY class, the L81->L84 interior granule +0.0002955, -> cover 1.026x.
                        # Both EXCLUDE 1, so this row is bare-financed with nothing to be upside.
                        # NOT -32,768 (layers 2 AND 4 together -> 13,893,676): that completes the whole
                        # 800 -> 784 uniform rung in one launch and the rung's own k-ladder puts the
                        # remainder at +0.000225..+0.000338, cover 0.90x-1.35x, which CONTAINS 1. And the
                        # warning against it is on the board already: L92->L95 is a LEVELLING PAIR and it
                        # cost +2.51 spreads, so levelling granules are not reliably free in pairs even
                        # though every individual reading is. One granule per launch, re-priced each time.
                        # ORIGINAL launch-98 COMMENT for this constant, kept because the mechanism is the
                        # same one and only the set changed: ONE ungated layer goes a rung below the other
                        # two, 800 -> 784, widths [784,880,800,880,800,880], floor 784, -16,384 ->
                        # num_params_total 13,926,444. Layer 0 is UNGATED and at span 256, and it is
                        # the layer launch 92 took down the previous rung by this same mechanism.
                        # SIZED AT 16,384 AND NOT 32,768 OR 49,152, ON THE BARE MARGIN. Champion v43
                        # holds 0.000309096383738 = 1.8975x the same-program spread 0.0001628969454063.
                        # THIS STEP ENDS RAGGED (floor 784, layers 2 and 4 sixteen above it), so it is
                        # NOT priced at r_uniform. autoscts__params_gpu6 corrected exactly that
                        # substitution in axis_claims.md v62, against their own v61 table AND against
                        # my proposal, and the [1.41x, 4.02x] I proposed is WITHDRAWN. Priced instead
                        # on this run's FIVE direct single-granule readings, dearest first:
                        #     L81->L84 interior granule 880->864  +0.0002955  cover  1.046x  <- bound
                        #     L89->L90 levelling granule          +0.0000600  cover  5.148x
                        #     L83 /L84 levelling granule          +0.0000180  cover 17.17x
                        #     L90->L92 MATCHED ragged-ending      -0.0000647  GAIN
                        #     L96->L97 levelling granule          -0.0001208  GAIN
                        #     union: cover [1.046x, INF) -- EXCLUDES 1, and it is THIN at the dear end
                        # gpu6's fitted alternatives are refused on the measurement of their own step
                        # type: their fixed raggedness penalty P = 0.000213..0.000278 predicts +1.88 to
                        # +2.67 spreads for launch 92 and launch 92 measured -0.397, an error of
                        # 2.3-3.1 spreads with the WRONG SIGN; and the cell P is fitted from is the
                        # -0.7416-spread L96/L97 difference, which its own author called a BOUND and
                        # not a rate to multiply. -49,152 and -32,768 stay REFUSED: [0.66x, 1.11x]
                        # under both readings, and both end ragged or cost 3x the mass.
                        # NO LEVER IS ADDED HERE: the resid multiplier stays at the
                        # banked 0.005 and its next rung 0.0025 is deliberately NOT taken, because its
                        # sign at that rung is unmeasured and an unmeasured lever on a 1.9-spread margin
                        # is collateral, not upside. This row changes exactly one thing.
MLP_NARROW5_ELEMENTS = 16 # a sixth step of the same free alignment class: 800 - 16 = 784 = 16 x 49,
                        # and 784 x 2 = 1,568 bytes = 32 x 49, so 784 is 32-BYTE aligned and
                        # 784 % 8 == 0 -- the same free class as 800/816/832/848/864/880/896.
MLP_NARROW_ELEMENTS = 16 # one step of the FREE alignment class: 896 - 16 = 880 = 1,760 bytes
                        # = 32 x 55, and the untouched layers stay at 896 = 1,792 = 128 x 14. The
                        # graded penalty (launch 51 -4.258% at 8-byte, launch 54 -0.992% at 16-byte,
                        # launch 52 -0.041% at 32-byte) cannot apply to either width here.
LOGIT_RANK = 256        # width of the vocabulary basis the classifier reads out of wte

# Optimization
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step (shipped value: 2**19). At
                        # DEVICE_BATCH_SIZE 128 this makes grad_accum_steps 1, so an
                        # optimizer update costs one microbatch instead of two.
EMBEDDING_LR = 0.3      # learning rate for token embeddings (Adam). 0.6 in every launch of this run
                        # up to and including champion v13. Launch 31 (red_embed_lr_lo_03, team
                        # redundancy) measured 0.6 -> 0.3 on champion v11 as -0.0019284260561595 of
                        # val_bpb with num_params_total, both FLOPs numbers, peak_vram_bytes,
                        # num_steps, total_tokens AND median step all bit-identical. That result's
                        # own knowledge file asks for a ~70% discount off its own base, so it is
                        # priced here at 0.0013. It is the run's only unbanked measured margin lever.
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 1.0         # learning rate for per-layer scalars (Adam). BANKING A MEASURED LEVER AT ZERO
                        # TRANSPORT. autoscts__params_gpu5's launch 86 (thr_scalar_lr_hi_on_v36) set this
                        # one literal on champion v36's OWN bytes -- the bytes this candidate is built
                        # from -- and measured val_bpb 1.0485213133031897 against 1.049182411092023:
                        # -0.0006610977888333 = 4.058 same-program spreads, for ZERO parameters and zero
                        # counted FLOPs, with peak_vram_bytes, flops_per_token_measured and step0_loss all
                        # bit-identical. Every previous lever composition in this run had to transport its
                        # income across a base change; this one does not, which is the whole reason it is
                        # worth banking now. A zero-parameter lever ties its base's num_params_total and a
                        # tie can never promote under a strict comparison, so launch 86's gain reaches the
                        # ledger ONLY inside a candidate that also cuts -- and per
                        # knowledge/a-lever-may-only-be-upside-never-collateral.md the cut above is sized
                        # so the BARE margin covers it with this literal priced at exactly zero.
                        #
                        # The axis is analyst3's (proposal 4582d135) and monotone increasing over 4x on the
                        # two rungs this run has: 0.25 cost +0.0007893423127 (launch 49, on launch 38's
                        # fork) and 1.0 gained -0.0006610977888 (launch 86, on v36). It is NOT closed --
                        # gpu5 names 2.0 as the next rung and the * 0.01 group multiplier after it -- and
                        # this candidate deliberately does not take 2.0: 1.0 is measured on these bytes,
                        # 2.0 is a two-base fit, and the cut above is the thing that has to promote.
                        #
                        # WHAT THIS LITERAL CANNOT CLAIM, kept from gpu5's own record rather than dropped:
                        # SCALAR_LR is the LR for TWO groups 100x apart, resid_params at scalar_lr * 0.01
                        # and x0_params at scalar_lr with hardcoded betas, so the joint move's sign is
                        # established and the per-group attribution is not.
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.7    # fraction of time budget for LR warmdown (0.5 in every launch of this
                        # run before launch 21; see the docstring's CHANGE 2 of 2)
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
        window_pattern=WINDOW_PATTERN, mlp_ratio=MLP_RATIO, logit_rank=LOGIT_RANK,
        mlp_narrow_layers=MLP_NARROW_LAYERS,
        mlp_narrow2_layers=MLP_NARROW2_LAYERS,
        mlp_narrow3_layers=MLP_NARROW3_LAYERS,
        mlp_narrow4_layers=MLP_NARROW4_LAYERS,
        mlp_narrow5_layers=MLP_NARROW5_LAYERS,
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
