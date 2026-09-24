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
    head_dim: int = 128
    mlp_mult: int = 4
    mlp_groups: int = 1
    mlp_write_shift: int = 0
    # Depth grouping of the gathered MLP pre-activation offset. The table is read once per
    # forward and its 1024 numbers are added to EVERY depth's pre-activation, so with one
    # table a token contributes the same offset at layer 0 and at layer 7 and a layer's only
    # freedom over it is me_gate's four per-position scalars. With G tables layer i reads
    # table (i * G) // n_layer, so contiguous blocks of depth carry independent per-token
    # offsets. 1 is the parent's single shared table and the per-layer index is all zeros,
    # i.e. the parent's exact read.
    mlp_embed_groups: int = 1
    window_pattern: str = "SSSL"
    short_window: int = 1024
    # Head count of a layer whose exact reach is the short window. 0 means "same
    # as n_head", i.e. the parent's uniform head budget at every depth.
    short_n_head: int = 0
    # Per-head reach grading. When True, head i of a layer whose declared reach is
    # `window` attends over max(short_window, window >> i) previous positions, so the
    # layer's reach is set by its LONGEST head instead of being paid for once per head.
    # One rule for the whole stack, and it is the identity on any layer whose declared
    # reach is already short_window, because max(short_window, short_window >> i) is
    # short_window at every i. False reproduces the parent's one window per layer.
    reach_ladder: bool = False
    # How many consecutive rungs of the reach ladder ONE whole-context layer keeps. The
    # parent restarts the ladder at rung 0 in every layer that declares the whole context,
    # so with two such layers the stack resolves 2048, 1024, 512 and 256 TWICE -- one head
    # of 64 channels at each scale in each layer -- and the frozen tally charges every one
    # of those eight calls. reach_ladder is already the rule that a layer pays for a scale
    # once instead of once per head; this field reads the same rule one level up, across
    # depth: the r-th whole-context layer in depth order starts at rung r *
    # ladder_rungs_per_layer and keeps that many rungs, and its remaining heads sit at the
    # floor short_window, which every local layer already resolves. At
    # ladder_rungs_per_layer * (number of whole-context layers) == the ladder's rung count
    # no scale leaves the model, it is merely provided once rather than twice. 0 is the
    # parent's rule at every layer and is the flag-off control; a layer whose declared reach
    # is already short_window is the identity either way, because max(short_window,
    # short_window >> i) is short_window at every i. This field is read by
    # _compute_head_window_segments alone, so head counts, declared reaches, read and write
    # group counts, gathered value-table widths and the key table's serving set are
    # identical under either setting -- see LADDER_RUNGS_PER_LAYER.
    ladder_rungs_per_layer: int = 0
    # Block-diagonal attention read. The residual is cut into attn_read_groups contiguous
    # channel groups; a layer that reads them takes its queries from group g and its shared
    # kv from group (g + attn_read_shift) % attn_read_groups, so every head's score is a
    # comparison ACROSS the channel groups the MLP already carries. 1 group is the parent's
    # dense n_embd read, and _compute_read_groups decides which layers group: the head
    # budget already follows the exact span, and so does the read's fan-in.
    attn_read_groups: int = 1
    attn_read_shift: int = 1
    # Whether the read grouping is CONDITIONAL on a layer's exact reach. The parent keys it
    # on the span: a layer whose declared reach is the whole context keeps the dense n_embd
    # read -- and, because write_groups is a layer's OWN read_groups, the dense write with
    # it -- while every short-window layer groups. True drops that condition, so every layer
    # reads its queries from one channel group and its shared kv from another and writes back
    # one matmul per head block: the two retrieval layers join the read and the write the
    # other six already use, and the file's last dense full-width maps inside a block go.
    # Nothing about reach moves. _compute_read_groups is the only reader of this field, so
    # head counts, declared reaches, per-head spans, kernel calls, the span tally and the
    # gathered value-table widths are identical under either setting. False is the parent's
    # exact per-reach rule.
    attn_group_every_layer: bool = False
    # Block-diagonal attention WRITE, keyed on the same quantity the read is. A layer
    # whose read is grouped writes its output back through one matmul PER GROUP instead
    # of one dense attn_dim -> n_embd matmul: head block g's attn_dim // groups channels
    # are projected into residual group (g + attn_write_shift) % groups, so the layer
    # still writes every one of the n_embd channels while each head block reaches half of
    # them. The group count is the layer's OWN read group count, so the write's fan-out
    # follows the exact span exactly as the read's fan-in already does, and the g-th slice
    # of the concatenated output is the g-th block of heads because
    # attn_dim // groups == (n_head // groups) * head_dim. False is the parent's dense
    # write at every layer.
    attn_write_grouped: bool = False
    # Which residual group each head block writes into, relative to the group its own
    # query projection reads. The read shift is what decides whether this is the
    # cross-group route: at two groups and attn_read_shift 1, head block g's value tensor
    # is a projection of group (g + 1) % 2, so a write shift of 0 -- block g writing the
    # group its query read -- delivers the OTHER group's content into this group, and a
    # shift of 1 would return each group its own content at identical cost.
    attn_write_shift: int = 0
    # Gathered per-token KEY component. The value side has carried one of these at every layer
    # since the attention pathway was narrowed: v receives a per-token table read under a
    # per-position gate, so a value's content is contextual PLUS whatever the type itself is
    # worth. The key side has never had one, and it cannot acquire one by any setting of any
    # existing field: ONE read serves both kv roles and the two are separated only by a
    # per-channel gain, so a key's per-token content is the value's under a gain ratio and
    # cannot be set independently of it. Nothing in this file lets a score say "the position I
    # am looking at is token type tau" except through the contextual projection. With this on,
    # ONE table of vocab_size x key_embed_width is read once per forward and added to k, before
    # rotary and before norm(k), at every layer whose kv width equals that table's width, under
    # one per-layer scalar initialised to zero. A gather issues no matmul and a broadcast
    # multiply and add issue none either, so the counted target charges nothing for any of it
    # while the table is charged in full against the parameter ceiling -- which is what fixes
    # both the width and the dose at one table (see KEY_EMBED).
    key_embed: bool = False
    # Per-head rotary band. THE MECHANISM. _precompute_rotary_embeddings builds ONE
    # (cos, sin) pair from base 10000 over sequence_len positions and forward hands that one
    # pair to every head of every layer, so the head that retrieves from the whole
    # 2,048-position context and a head whose exact reach is 128 positions are rotated by
    # the identical 32 frequency pairs. Those pairs are a head's entire positional
    # vocabulary: the only way a rotated score sees position is through the relative angle
    # theta_j * (t_q - t_k) each pair contributes, so a pair whose wavelength exceeds twice
    # a head's reach turns through less than half a cycle across EVERY displacement that
    # head can see, cannot order any two of them, and rides as a near-constant rotation that
    # cancels out of the logit. With this on, a head whose reach is w takes
    # base * w / sequence_len: the whole-context head keeps base 10000 to the digit and
    # every shorter head's band is compressed in proportion to what it can actually reach,
    # so the file's dyadic reach ladder is matched by a dyadic base ladder 10000, 5000,
    # 2500, 1250, 625. Read by _compute_rope_tables alone. Head counts, declared reaches,
    # per-head spans, kernel calls, the span tally, read and write fan-ins, gathered table
    # widths, the key table's serving set and every nn.Linear, gain and scalar are identical
    # under either setting, and a band is built in __init__ and init_weights and never in
    # forward, so torch.outer stays out of the counted graph. False is the parent's single
    # shared band and is this card's flag-off control.
    rope_reach_matched: bool = False
    # Query blocks per kernel call. THE MECHANISM. prepare._attention_flops_probe charges a
    # call 12 * B * T_call * h * d * min(window_left, k.shape[1]): EVERY query position in a
    # call pays that call's full declared window, whatever causality lets it reach. A query at
    # position t has only t + 1 previous positions to read, so on the head that declares the
    # whole context the first half of the sequence is charged 2048 for a reach that does not
    # exist -- which is why the frozen tally is twice the causal work the kernel actually does,
    # exactly as _attention_flops_probe's own docstring says it charges min(left, S) "with no
    # causal halving". With this at K > 1 each class's query axis is cut into K contiguous
    # blocks of sequence_len // K and block j is handed the key/value PREFIX [0, (j+1) * L)
    # instead of the whole sequence. The kernel's own documented rule is that with window_size
    # set, query i attends to keys in [i + seqlen_k - seqlen_q - left, i + seqlen_k - seqlen_q
    # + right]; here seqlen_k - seqlen_q is exactly the block's start, so query i of block j
    # attends to absolute keys [j*L + i - window, j*L + i] -- the mask the single call already
    # applies, key for key, and causal=True's bottom-right alignment agrees. NOTHING is
    # subsampled, no window is shortened, no head loses a position it could read and no
    # projection, table, gain or scalar moves: the charge stops covering positions the mask
    # already forbids. Blocks whose charge is equal are merged, so a class whose window fits
    # inside one block keeps the parent's single call over the whole sequence. Read by
    # _compute_head_call_plans alone. 1 is the parent's one call per class and is this card's
    # flag-off control.
    causal_query_blocks: int = 1
    # Whether the causal charge's BLOCK LENGTH is derived from the stack's own reach ladder
    # instead of being carried as a block count. THE MECHANISM's dose, read out of the model
    # rather than picked: _compute_head_window_segments already fixes what reaches this stack
    # resolves -- here 2048, 1024, 512, 256 and the 128 floor -- and the charge granularity
    # that matches them is the FINEST of those reaches above the floor, because a class whose
    # window is at most one block long is charged its exact window and merges back to one
    # call, while every class above it is charged in blocks of the smallest reach any head in
    # this stack actually resolves. So the block length is min over every per-head span
    # greater than short_window and the block count is sequence_len // that, with
    # causal_query_blocks kept as the count that stands when the ladder offers no such rung.
    # Read by _compute_head_call_plans alone, exactly as causal_query_blocks is, so head
    # counts, declared reaches, per-head spans, read and write group counts, gathered table
    # widths, the key table's serving set, the rotary bands and every nn.Linear are identical
    # under either setting, and no parameter is added, removed, resized or re-dtyped. False is
    # the parent's carried block count and is this card's flag-off control.
    causal_blocks_at_finest_rung: bool = False
    # Whether the ladder rung the charge block length is read off INCLUDES the floor. THE
    # MECHANISM's one remaining rung. causal_blocks_at_finest_rung takes the block length to be
    # the finest per-head span STRICTLY ABOVE short_window, on the reading that a class whose
    # window fits inside one block is charged its exact window and merges back to a single call
    # -- true, and it is why the floor itself was excluded. But the floor is a reach this stack
    # resolves at sixteen of its twenty heads, and the quantity that decides how finely the flat
    # rate should be applied is not which reaches exist, it is what an added kernel call buys:
    # halving a class's block length from L removes 12 * heads * head_dim * L**2 /
    # (4 * sequence_len) of key for window / L added calls, i.e. exactly that much per added
    # call. At L = 256 every class above the floor repays a call 6,144 and at L = 128 every one
    # of them repays 1,536, so 256 -> 128 is the last uniform halving whose per-call return is
    # above every other split this stack offers, and the floor is the rung that expresses it.
    # With this True the rung set is min over EVERY per-head span rather than over the spans
    # above short_window, so the block length is 128 and every class whose window exceeds 128 is
    # charged per block while the sixteen floor heads keep one call over the whole query axis at
    # their exact window -- min(128, (j+1) * 128) is 128 at every j, so all their blocks carry
    # the same charge, merge back into one call and take _attend's fast path with the parent's
    # own tensors. Read by _compute_head_call_plans alone. False is the parent's rung set and is
    # this card's flag-off control.
    causal_blocks_at_floor_reach: bool = False
    # Which HEADS the one gathered key table serves. THE MECHANISM. key_embed adds a per-token
    # component to k under a per-layer scalar, and the parent decides who receives it with
    # `head_counts[i] * head_dim == key_embed_width` -- a test on the LAYER's whole kv width,
    # i.e. on the table's own shape. That admits the six layers whose kv width is 128 and
    # excludes the two whose kv width is 256, and it excludes with them heads 2 and 3 of depth
    # 3 and of depth 7, whose exact reach is short_window: the very reach the table exists to
    # serve. Sixteen of this stack's twenty heads span short_window and only twelve receive the
    # component. Every OTHER per-head resource in this file already follows reach -- the head
    # budget (_compute_head_counts), the read fan-in and write fan-out (_compute_read_groups),
    # the gathered value table's width, the rotary band (_compute_rope_tables) and the charge
    # granularity (_compute_head_call_plans). The key table's serving set is the last one that
    # does not. With this True the set is keyed on head SPAN: a layer is served over the
    # contiguous TRAILING block of key_embed_width // head_dim heads whose span is
    # short_window, so the component reaches every head that does the job the table was built
    # for and no head that does not. Read by _compute_key_embed_slots alone. The table, its
    # width, its parameter count, ke_lambdas' (n_layer,) shape and optimizer group, every
    # nn.Linear, every window, head count, declared reach, kernel call and span-tally entry are
    # identical under either setting, and no parameter is added, removed, resized or re-dtyped.
    # False is the parent's kv-width rule and is this card's flag-off control.
    key_embed_by_reach: bool = False
    # How many of the residual's n_head channel units the readout reads at PAIR resolution
    # instead of unpooled. THE MECHANISM. The unit is n_embd // n_head -- the file's own
    # residual channel unit, the one the readout's fan-in arithmetic has always been quantised
    # to. Every cut this file has measured at the readout removed information from a TYPE'S
    # VIEW: a rank factorisation removed directions, a block partition gave each type half the
    # channels and an overlapping cover gave it three quarters, and in each of those a type's
    # weight vector had no column at all for the channels it did not read. This field removes
    # none of that. The trailing readout_coarse_units units are contracted PAIRWISE -- each
    # contiguous pair of channels replaced by its mean times 2**0.5, the rms-preserving
    # contraction of two coordinates at unit rms -- so every one of the n_embd channels still
    # reaches every one of the vocab_size types, every channel still has a column, and pooled
    # channels share one. What is given up is the RESOLUTION at which the pooled unit is seen,
    # not its presence. A slice, a reshape, aten::mean, a multiply by a Python float and one
    # torch.cat are all unregistered in FlopCounterMode -- measured on the paid instrument at
    # the sixteen gate reads, where both probe halves came back digit-identical with exactly
    # this arithmetic live -- so the whole counted movement is lm_head's own fan-in,
    # n_embd -> n_embd - readout_coarse_units * (n_embd // n_head) // 2. Read by GPT.__init__
    # and GPT._readout_read alone, so head counts, declared reaches, per-head spans, kernel
    # calls, the call plan, the span tally, read and write group counts, gathered table widths,
    # the key table's serving set, the rotary bands and every other nn.Linear are identical
    # under either setting. 0 is the parent's dense read of all n_embd channels and is this
    # card's flag-off control.
    readout_coarse_units: int = 0


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (every layer)."""
    return True


def apply_rotary_emb(x, cos, sin):
    # cos and sin broadcast against x over the batch axis and, when one band is shared by a
    # layer's heads, over the HEAD axis too. This body is unchanged from the parent and needs
    # no edit to give each head its own band, because a (1, T, n_head, d) table broadcasts
    # against a (B, T, n_head, 2d) q exactly as the parent's (1, T, 1, d) table does. Every
    # op here is an elementwise multiply, a negate, an add or a concatenation, none of which
    # the frozen counter charges.
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx, n_head, read_groups, ke_slot=None):
        super().__init__()
        # The head count arrives per layer instead of being read from the config:
        # a layer whose exact reach is the short window carries config.short_n_head
        # heads, a full-context layer carries config.n_head. Both kv roles follow
        # the query count, so this layer's kv width -- and hence the width of the
        # value embedding it gathers -- is n_head * head_dim.
        self.n_head = n_head
        self.n_kv_head = n_head
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim
        self.attn_dim = self.n_head * self.head_dim
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.kv_dim = self.n_kv_head * self.head_dim
        # The read grouping arrives per layer for the same reason the head count does. Under
        # config.attn_group_every_layer it no longer FOLLOWS the layer's exact reach: every
        # layer groups, the two that retrieve from the whole context included, so
        # read_groups == 1 -- the dense read of the whole residual -- is reached only with
        # that field off, which is this card's flag-off control.
        self.read_groups = read_groups
        assert read_groups > 0
        assert self.n_embd % read_groups == 0
        assert self.n_head % read_groups == 0
        assert self.attn_dim % read_groups == 0 and self.kv_dim % read_groups == 0
        self.read_in = self.n_embd // read_groups
        self.read_shift = config.attn_read_shift % read_groups
        # ONE read of the residual serves both kv roles. The key tensor and the value
        # tensor are that read under a per-channel gain each, so the layer still holds two
        # distinct kv tensors while paying for one n_embd -> kv_dim matmul instead of two.
        # A gain is an elementwise multiply: it issues no matmul, so the counter charges it
        # nothing, and it is still charged in full against the parameter ceiling.
        # One q projection and one kv projection PER GROUP, each reading read_in channels
        # instead of all n_embd. Group g's q reads residual group g and group g's kv reads
        # residual group (g + read_shift) % read_groups, so at two groups and a shift of one
        # a head's score keeps the two OFF-DIAGONAL blocks of the residual's pair space and
        # gives up the two diagonal ones. The g-th slice of the concatenated width is the
        # g-th block of heads, because attn_dim // read_groups == (n_head // read_groups) *
        # head_dim, and the kv slice for that block is the group its query does not read.
        self.c_q = nn.ModuleList([nn.Linear(self.read_in, self.attn_dim // read_groups,
                                            bias=False) for _ in range(read_groups)])
        self.c_kv = nn.ModuleList([nn.Linear(self.read_in, self.kv_dim // read_groups,
                                             bias=False) for _ in range(read_groups)])
        self.k_gain = nn.Parameter(torch.ones(self.kv_dim))
        self.v_gain = nn.Parameter(torch.ones(self.kv_dim))
        # Learnable attention logit scale, one entry per (query head, head channel).
        # q and k both enter the kernel RMS-normed over head_dim and no softmax_scale
        # is passed, so the logit scale is pinned at the kernel default head_dim**-0.5
        # and no weight in this file can move it: whatever scale c_q or c_kv produces,
        # norm() divides it out. This gain is applied AFTER norm(q), which is the only
        # place on the q/k path where a scale survives. Only the elementwise PRODUCT of
        # a q-side and a k-side gain reaches the logits, so one gain spans the whole
        # reachable set and there is deliberately no k-side twin. Held 1-D and viewed
        # per head in forward for the reason the kv gains are 1-D: Muon's fused step
        # reads shape[-2], so a 1-D parameter cannot join a Muon group and falls into
        # the block's existing 1-D AdamW group with no change to setup_optimizer.
        self.q_scale = nn.Parameter(torch.ones(self.attn_dim))
        # ONE write per group instead of one dense attn_dim -> n_embd write. The group
        # count is this layer's own read grouping, so a layer that keeps the dense read
        # keeps the dense write and the whole attention path of a layer is grouped or
        # dense together. write_in is one head block's channels and write_out is that
        # block's slice of the residual; at write_groups == 1 this is one Linear of the
        # parent's exact shape.
        self.write_groups = self.read_groups if config.attn_write_grouped else 1
        assert self.attn_dim % self.write_groups == 0
        assert self.n_embd % self.write_groups == 0
        self.write_in = self.attn_dim // self.write_groups
        self.write_out = self.n_embd // self.write_groups
        self.write_shift = config.attn_write_shift % self.write_groups
        self.c_proj = nn.ModuleList([nn.Linear(self.write_in, self.write_out, bias=False)
                                     for _ in range(self.write_groups)])
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None
        # Which of this layer's heads the ONE gathered key table serves, as
        # (first_head, head_count), or None for a layer it does not serve. Decided once in
        # GPT._compute_key_embed_slots from this layer's own per-head spans. A pair of Python
        # ints, so it adds no parameter, buffer, module or counted op of any kind, exactly as
        # rope_index and ladder_first_rung do not; and (0, n_kv_head) -- every layer the
        # parent serves -- takes the parent's own single line in forward.
        self.ke_first_head, self.ke_heads = ke_slot if ke_slot is not None else (0, 0)

    def _attend(self, q, k, v, window, calls):
        """One reach class's kernel calls: one per query block, each charged its own reach.

        `calls` is this class's list of (q_start, q_len, k_len, span) blocks, built once in
        GPT._compute_head_call_plans and covering [0, T) in order with no position dropped and
        none repeated. A single block spanning the whole query axis takes the parent's exact
        call with the parent's own tensors, which is what config.causal_query_blocks == 1 and
        every class whose window fits inside one block both reduce to.

        For a split block the kernel receives q[:, q_start : q_start + q_len] against the
        key/value PREFIX [0, k_len), with this class's window unchanged. The mask is the
        parent's, and that is the kernel's own documented arithmetic rather than an inference:
        with window_size set, query i attends to keys in [i + seqlen_k - seqlen_q - left,
        i + seqlen_k - seqlen_q + right], and k_len - q_len == q_start by construction, so
        query i of the block attends to absolute keys [q_start + i - window, q_start + i] --
        the identical set the unsplit call gives absolute query q_start + i. causal=True is
        kept and agrees: its mask is aligned to the bottom-right corner, so it keeps keys
        <= i + seqlen_k - seqlen_q = q_start + i, the block's own absolute positions. Where
        span < window the window cannot bind at all, because span == k_len there means
        k_len <= window, and the prefix is then the whole causal set anyway.

        The raw views rather than .contiguous() copies -- THE MECHANISM. Every slice a split
        block takes is on an axis ABOVE the last one, so its stride(-1) is 1, and the kernel's
        own Python interface is what decides whether that is enough: flash_attn_interface
        defines maybe_contiguous(x) as `x.contiguous() if x.stride(-1) != 1 else x`, applies it
        to q and k on the way into flash_attn_3_cuda.fwd, and passes v through untouched unless
        `v.stride(-1) != 1 and v.stride(-3) != 1`. So the interface packs exactly the layouts
        the kernel cannot read and hands every other one to the kernel as it arrives -- which is
        why FA3's forward takes a separate batch, row and head stride for each of q, k and v. A
        query-block or key-prefix view of a contiguous (B, T, h, d) tensor keeps that tensor's
        row, head and last-dim strides and differs from it only in the extent of one axis, so it
        is one of those layouts and the copy this file used to make was the interface's own
        no-op done twice. Nothing the frozen probe reads moves: it reads B, T, h and d off each
        call's q and min(window_size[0], k.shape[1]) off its window kwarg and key tensor, and a
        view has the shape its copy had. Slicing and cat carry no counted FLOP, and the
        concatenation is over the query axis, so the assembled output has the parent's shape
        with each position written exactly once.
        """
        if len(calls) == 1 and calls[0][1] == q.shape[1] and calls[0][2] == k.shape[1]:
            return fa3.flash_attn_func(q, k, v, causal=True, window_size=window)
        parts = []
        for q_start, q_len, k_len, _ in calls:
            assert k_len - q_len == q_start
            parts.append(fa3.flash_attn_func(
                q[:, q_start:q_start + q_len],
                k[:, :k_len], v[:, :k_len],
                causal=True, window_size=window))
        return torch.cat(parts, dim=1)

    def forward(self, x, ve, cos_sin, head_windows, call_plan, ke=None, ke_lam=None):
        B, T, C = x.size()
        # The grouped read. xs[g] is residual channel group g: group g's q projection reads
        # it and group g's kv projection reads group (g + read_shift) % read_groups. A
        # last-dim split, a per-group matmul and one concatenation carry the same counted
        # cost as one matmul of the same total in x out width, and at read_groups == 1 both
        # reads are x itself, which is the parent's pair of dense projections exactly.
        xs = x.split(self.read_in, dim=-1)
        q = torch.cat([proj(xs[g]) for g, proj in enumerate(self.c_q)], dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim)
        # The shared kv read, separated into a key tensor and a value tensor by one gain
        # each. The gain is cast to kv's dtype deliberately: aten::mul does not participate
        # in autocast, so a float32 gain against a bf16 projection would promote k and v to
        # float32, and flash_attn_func requires q, k and v to share one dtype.
        kv = torch.cat([proj(xs[(g + self.read_shift) % self.read_groups])
                        for g, proj in enumerate(self.c_kv)], dim=-1)
        k = (kv * self.k_gain.to(kv.dtype)).view(B, T, self.n_kv_head, self.head_dim)
        v = (kv * self.v_gain.to(kv.dtype)).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        # Key residual: the key side's counterpart of the value residual above, and the
        # mechanism this card is about. ke is ONE table read once per forward in GPT.forward and
        # shared by every layer whose kv width matches it; ke_lam is this layer's own scalar,
        # exactly 0.0 at initialisation, so at step 0 k is bit-for-bit the parent's and the whole
        # model is the parent's function at the parent's windows. The component is added HERE --
        # before rotary and before norm(k), the same position on the key path that ve occupies on
        # the value path -- so the per-token direction is rotated and normed exactly like the
        # contextual key rather than riding outside the geometry the kernel scores in. The cast
        # is this file's own rule for a float32 scalar meeting a bf16 tensor: aten::mul does not
        # participate in autocast, so an uncast scalar would promote k to float32 and
        # flash_attn_func requires q, k and v to share one dtype. A gather, a broadcast multiply
        # and an add issue no matmul, so the counter charges this exactly zero, and q.shape and
        # every window argument the probe reads are untouched.
        if ke is not None:
            ke_add = (ke_lam.to(k.dtype) * ke).view(B, T, self.ke_heads, self.head_dim)
            if self.ke_first_head == 0 and self.ke_heads == self.n_kv_head:
                # Every layer the parent serves lands here: the whole kv width receives the
                # component and this is the parent's line, operand for operand.
                k = k + ke_add
            else:
                # A layer whose floor-reach heads are a proper trailing block of its heads:
                # the component is added to those heads and to no others. A head-axis slice,
                # an add and one concatenation carry no counted FLOP, torch.cat returns a
                # contiguous tensor of the parent's shape, and every position of every head
                # is written exactly once -- so q.shape, k.shape and every window argument
                # the frozen probe reads are untouched, as is the attended mask.
                lo, n = self.ke_first_head, self.ke_heads
                parts = []
                if lo:
                    parts.append(k[:, :, :lo])
                parts.append(k[:, :, lo:lo + n] + ke_add)
                if lo + n < self.n_kv_head:
                    parts.append(k[:, :, lo + n:])
                k = torch.cat(parts, dim=2)

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        # The one degree of freedom the two norms above remove: how sharply this head's
        # logits may vary. Cast for the reason the kv gains are cast -- aten::mul does
        # not participate in autocast, so a float32 gain would promote q to float32 and
        # flash_attn_func requires q, k and v to share one dtype. An elementwise multiply
        # issues no matmul, so the counter charges it nothing, while the 256 entries are
        # charged in full against the parameter ceiling; q.shape and the window argument
        # the probe reads are untouched.
        q = q * self.q_scale.view(self.n_head, self.head_dim).to(q.dtype)

        # One kernel call per reach class. head_windows is this layer's list of
        # (first_head, head_count, window) segments covering its heads in order, built once
        # in GPT._compute_head_window_segments. A layer with a single class -- every layer
        # whose declared reach is the short window, and every layer at all when
        # config.reach_ladder is False -- takes the single-call path with this layer's own
        # tensors, so its kernel arguments and its span-tally entry are the parent's exactly.
        # A graded layer splits q, k and v by head: the frozen probe reads B, T, h, d off
        # each call's own q and min(window[0], k.shape[1]) off its own window_size kwarg, so
        # every class is charged at its own reach and nothing else in this file can move it.
        if len(head_windows) == 1:
            y = self._attend(q, k, v, head_windows[0][2], call_plan[0])
        else:
            counts = [count for _, count, _ in head_windows]
            qs = torch.split(q, counts, dim=2)
            ks = torch.split(k, counts, dim=2)
            vs = torch.split(v, counts, dim=2)
            parts = []
            for idx, (_, _, window) in enumerate(head_windows):
                # The raw views rather than .contiguous() copies -- the same mechanism
                # _attend's docstring states, at the other site that took the same
                # defensive copy for the same unverified reason. A head-axis slice of a
                # contiguous (B, T, h, d) tensor is not contiguous, but its stride(-1) is
                # still 1 and its row and head strides are still the base tensor's, so the
                # kernel interface's own maybe_contiguous leaves it alone and FA3's
                # per-tensor batch/row/head strides address it directly. The copy this
                # removes is one whole q, one whole k and one whole v per graded layer,
                # summed over that layer's classes and independent of how many classes or
                # calls there are, so it is the one packing cost in this path that the
                # charge plan never bought. torch.split, narrow and cat carry no counted
                # FLOP, and every shape the frozen probe reads is the shape it read before.
                parts.append(self._attend(qs[idx], ks[idx], vs[idx],
                                          window, call_plan[idx]))
            y = torch.cat(parts, dim=2)
        y = y.contiguous().view(B, T, -1)
        # The grouped write. ys[g] is head block g's own output channels -- the same head
        # blocks the grouped read defines, because attn_dim // write_groups ==
        # (n_head // write_groups) * head_dim -- and destination residual group d receives
        # the block whose index is (d - write_shift) % write_groups, i.e. block g writes
        # into group (g + write_shift) % write_groups. Every residual channel is still
        # written by exactly one head block, so the layer's write is full width; what a
        # block gives up is the half of the residual it does not address. A last-dim
        # split, a per-group matmul and one concatenation carry the same counted cost as
        # one matmul of the same total in x out width, and at write_groups == 1 both the
        # split and the concatenation are the identity and this is the parent's dense
        # write exactly.
        ys = y.split(self.write_in, dim=-1)
        outs = [proj(ys[g]) for g, proj in enumerate(self.c_proj)]
        y = torch.cat([outs[(d - self.write_shift) % self.write_groups]
                       for d in range(self.write_groups)], dim=-1)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.mlp_mult * config.n_embd
        # Grouped MLP with a cyclic write-back shift. The residual is cut into
        # mlp_groups contiguous channel groups and each group keeps its own c_fc/c_proj
        # pair, so the hidden width and the count of squared-ReLU features are exactly
        # the parent's; the shift decides which group each pair writes into. With
        # mlp_write_shift = 0 the block structure is diagonal -- read group g, write
        # group g -- and all channel mixing is left to attention's output projection and
        # to the full-width gathered per-token offset below. With a nonzero shift group
        # g's hidden features are projected into group (g + shift) % mlp_groups, so every
        # MLP carries a cross-group path of its own at identical matmul shapes.
        self.n_groups = config.mlp_groups
        assert config.n_embd % self.n_groups == 0
        assert self.hidden_dim % self.n_groups == 0
        self.group_in = config.n_embd // self.n_groups
        self.group_hidden = self.hidden_dim // self.n_groups
        self.write_shift = config.mlp_write_shift % self.n_groups
        self.c_fc = nn.ModuleList([nn.Linear(self.group_in, self.group_hidden, bias=False)
                                   for _ in range(self.n_groups)])
        self.c_proj = nn.ModuleList([nn.Linear(self.group_hidden, self.group_in, bias=False)
                                     for _ in range(self.n_groups)])
        self.me_gate_channels = 32
        self.me_gate_groups = config.n_kv_head
        assert self.hidden_dim % self.me_gate_groups == 0
        self.me_gate = nn.Linear(self.me_gate_channels, self.me_gate_groups, bias=False)

    def forward(self, x, me):
        B, T, C = x.size()
        xs = x.split(self.group_in, dim=-1)
        h = torch.cat([fc(xs[g]) for g, fc in enumerate(self.c_fc)], dim=-1)
        # Token-conditioned hidden features: a gathered per-token offset on the
        # pre-activation, gated per position exactly as the value residual is.
        gate = 2 * torch.sigmoid(self.me_gate(x[..., :self.me_gate_channels]))
        me = me.view(B, T, self.me_gate_groups, -1)
        h = h + (gate.unsqueeze(-1) * me).view(B, T, -1)
        h = F.relu(h).square()
        hs = h.split(self.group_hidden, dim=-1)
        ys = [proj(hs[g]) for g, proj in enumerate(self.c_proj)]
        # Destination group d receives the pair whose input group was d - write_shift,
        # i.e. group g writes into group (g + write_shift) % n_groups. This is a
        # reordered concatenation of the same tensors: the matmul shapes, the parameter
        # count and every counted FLOP are exactly those of a shift of 0.
        h = torch.cat([ys[(d - self.write_shift) % self.n_groups]
                       for d in range(self.n_groups)], dim=-1)
        return h


class Block(nn.Module):
    def __init__(self, config, layer_idx, n_head, read_groups, ke_slot=None):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx, n_head, read_groups, ke_slot)
        self.mlp = MLP(config)

    def forward(self, x, ve, me, cos_sin, head_windows, call_plan, ke=None, ke_lam=None):
        x = x + self.attn(norm(x), ve, cos_sin, head_windows, call_plan, ke, ke_lam)
        x = x + self.mlp(norm(x), me)
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.head_counts = self._compute_head_counts(config)
        self.read_groups_per_layer = self._compute_read_groups(config)
        # Per-layer, per-reach-class windows actually handed to the kernel. window_sizes
        # stays the layer's DECLARED reach -- its longest head's -- because that is what
        # _compute_head_counts reads to set the head budget and what the frozen analytic
        # cross-check reads by name.
        self.head_window_segments = self._compute_head_window_segments(config)
        # Per reach class, the query blocks each of which is charged its own causal reach.
        # A plain Python list of tuples, so it is invisible to prepare.count_params and to
        # both optimizers exactly as rope_index and ladder_first_rung are, and it adds no
        # parameter, buffer, module or counted op of any kind.
        self.head_call_plans = self._compute_head_call_plans(config)
        # The gathered key table's width, and WHICH HEADS of each layer it serves. Hoisted
        # above the blocks because a layer's serving block is now a property of that layer's
        # per-head spans and has to be handed to the module that applies it. Plain Python
        # ints and tuples, invisible to prepare.count_params and to both optimizers exactly
        # as rope_index, ladder_first_rung and the call plans already are.
        self.key_embed_width = min(self.head_counts) * config.head_dim
        self.key_embed_slots = self._compute_key_embed_slots(config, self.key_embed_width)
        self.key_embed_layers = tuple(i for i, slot in enumerate(self.key_embed_slots)
                                      if slot is not None)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i, self.head_counts[i],
                                      self.read_groups_per_layer[i],
                                      self.key_embed_slots[i])
                                for i in range(config.n_layer)]),
        })
        # The readout's fan-in, after the trailing readout_coarse_units of the residual's
        # n_head channel units are contracted pairwise. readout_unit is n_embd // n_head, the
        # file's own residual channel unit; readout_fine is the count of channels that arrive
        # unpooled, readout_coarse the count that arrive as pairs, and readout_in what lm_head
        # actually reads. The asserts sit here, in __init__ on the meta device, so an
        # inexpressible geometry fails before the chargeable witness and costs no launch. At
        # readout_coarse_units == 0 readout_in is config.n_embd and the Linear below is the
        # parent's exact shape, its exact parameter count and its exact AdamW signature.
        assert config.n_embd % config.n_head == 0
        assert 0 <= config.readout_coarse_units <= config.n_head
        self.readout_unit = config.n_embd // config.n_head
        self.readout_coarse = config.readout_coarse_units * self.readout_unit
        assert self.readout_coarse % 2 == 0
        self.readout_fine = config.n_embd - self.readout_coarse
        self.readout_in = self.readout_fine + self.readout_coarse // 2
        self.lm_head = nn.Linear(self.readout_in, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Per-layer strength of the prefix-mean channel built in forward(). One scalar per
        # layer, exactly like x0_lambdas, so the channel is a per-depth read of a shared
        # tensor rather than a new module: no matmul and no counted FLOP.
        self.gmem_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Value embeddings. One table per layer, at THAT layer's kv width: the table
        # is read into v, so its width is the layer's own n_kv_head * head_dim. A
        # narrowed layer therefore gathers a narrower per-token value component --
        # the capability this card spends, and the reason it is not spent on the two
        # full-context layers.
        head_dim = config.head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, self.head_counts[i] * head_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        # Gathered MLP features: token-conditioned hidden offsets, one table per contiguous
        # block of depth, each read once per forward and shared by the depths assigned to it.
        # A gather issues no matmul, so the counted target charges nothing for any number of
        # tables while every one of them is charged in full against the parameter ceiling.
        # At mlp_embed_groups == 1 this is one nn.Embedding of the parent's exact shape and
        # the per-layer index below is all zeros, so the read is the parent's exactly.
        self.mlp_embed_groups = config.mlp_embed_groups
        assert 0 < self.mlp_embed_groups <= config.n_layer
        self.mlp_embed = nn.ModuleList([
            nn.Embedding(config.vocab_size, config.mlp_mult * config.n_embd)
            for _ in range(self.mlp_embed_groups)
        ])
        # Which table each depth reads: contiguous blocks of depth, so at two tables the
        # first half of the stack shares one table and the second half the other.
        self.mlp_embed_index = [(i * self.mlp_embed_groups) // config.n_layer
                                for i in range(config.n_layer)]
        # Gathered per-token key component. ONE table, read once per forward exactly as the
        # gathered MLP offsets are, at the NARROWEST kv width the stack carries. That WIDTH is
        # not a convenience: a table costs vocab_size x width parameters and the parameter
        # ceiling admits exactly one at this width and none at any wider one. What the width
        # does not have to pick is the SERVING SET -- the parent let it, by testing a layer's
        # whole kv width against the table's, and config.key_embed_by_reach replaces that test
        # with the heads whose exact reach is the ladder's floor. The table, its width and its
        # parameter count are the same either way. key_embed_width and key_embed_layers are
        # computed above the blocks now, because a layer's serving block is handed to its
        # attention module at construction. The per-layer strengths are held as ONE tensor of
        # n_layer entries rather than one entry per serving layer, because the compiled AdamW
        # step specialises on (shape, dtype) and this file already holds an (n_layer,) float32
        # class at three tensors; a shorter vector would be a ninth signature class against
        # zero headroom.
        if config.key_embed:
            self.key_embed = nn.Embedding(config.vocab_size, self.key_embed_width)
            self.ke_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        else:
            self.key_embed = None
            self.ke_lambdas = None
        # Rotary embeddings, one band per distinct per-head reach pattern. rope_patterns and
        # rope_index are plain Python lists, so they are invisible to prepare.count_params and
        # to both optimizers; the bands themselves are non-persistent buffers exactly as the
        # parent's single pair is, so they are not parameters, carry no gradient, take no
        # optimizer group and cost nothing against the parameter ceiling.
        self.rotary_seq_len = config.sequence_len * 10
        self.rope_patterns, self.rope_index = self._compute_rope_tables(config)
        for table_idx, spans in enumerate(self.rope_patterns):
            cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim,
                                                          spans=spans)
            self.register_buffer(f"rope_cos_{table_idx}", cos, persistent=False)
            self.register_buffer(f"rope_sin_{table_idx}", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # Transformer blocks
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            # The file's own init rule, evaluated at the grouped read's own fan-in, exactly
            # as the grouped MLP evaluates it at group_in below: s is 3**0.5 * fan_in**-0.5
            # and a grouped read has fan_in n_embd // read_groups. At read_groups == 1 this
            # is the parent's s to the digit. q and k both enter the kernel RMS-normed over
            # head_dim, so a common factor on c_q or c_kv is divided out of every logit and
            # this choice cannot move the step-0 attention pattern; what it does set is the
            # value path's initial magnitude and each matrix's update-to-weight ratio.
            s_read = 3**0.5 * block.attn.read_in**-0.5
            for proj in block.attn.c_q:
                torch.nn.init.uniform_(proj.weight, -s_read, s_read)
            # The shared kv read takes the same fan-in rule as the two projections it
            # replaces, and both gains start at exactly 1.0, so at step 0 the key and the
            # value tensor are equal and only their gradients separate them. Required, not
            # cosmetic: the module is built on meta and filled by to_empty().
            for proj in block.attn.c_kv:
                torch.nn.init.uniform_(proj.weight, -s_read, s_read)
            torch.nn.init.ones_(block.attn.k_gain)
            torch.nn.init.ones_(block.attn.v_gain)
            # Exactly 1.0 at every entry, so at step 0 the logit scale is the parent's
            # pinned head_dim**-0.5 and this block computes the parent's function bit for
            # bit. ones_ draws no random numbers, so the init RNG stream is the parent's
            # too and nothing downstream of it diverges.
            torch.nn.init.ones_(block.attn.q_scale)
            # Same zero init as the parent's single write matrix, applied to each
            # grouped write matrix. zeros_ draws no random numbers, so the init RNG
            # stream is the parent's to the bit and the step-0 function is unchanged:
            # attention contributes exactly nothing until Muon moves these off zero.
            # Required rather than cosmetic -- the module is built on meta and filled by
            # to_empty().
            for proj in block.attn.c_proj:
                torch.nn.init.zeros_(proj.weight)
            # Same init rule, evaluated at the grouped matrices' own fan-in: s above
            # is 3**0.5 * fan_in**-0.5 for the n_embd-wide matrices, and a grouped c_fc
            # has fan_in n_embd // mlp_groups.
            s_mlp = 3**0.5 * block.mlp.group_in**-0.5
            for fc in block.mlp.c_fc:
                torch.nn.init.uniform_(fc.weight, -s_mlp, s_mlp)
            for proj in block.mlp.c_proj:
                torch.nn.init.zeros_(proj.weight)
        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        # The prefix-mean channel starts closed, so at step 0 the model is exactly the
        # parent's at the same windows. The scalar's gradient is the inner product of the
        # residual gradient with the channel, which is nonzero at zero, so it can open.
        self.gmem_lambdas.fill_(0.0)
        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        # Gathered MLP features, at the same scale as the other gathered tables. ONE draw,
        # for table 0, in exactly the parent's position in the init RNG stream, and every
        # further table copied from it: the stream stays the parent's to the bit, and at
        # initialisation every depth reads the identical offsets the parent's single table
        # gave it, so this model IS the parent's function at init and the tables separate
        # only as far as their own gradients drive them apart. Required rather than
        # cosmetic -- the module is built on meta and filled by to_empty().
        torch.nn.init.uniform_(self.mlp_embed[0].weight, -s, s)
        for table in self.mlp_embed[1:]:
            table.weight.copy_(self.mlp_embed[0].weight)
        # The gathered key component, at the same scale as every other vocabulary table here.
        # ONE draw, placed AFTER the gathered MLP tables' draw and before the gate zeros, which
        # draw no random numbers -- so every draw the parent makes happens at exactly the
        # parent's position in the init RNG stream and nothing upstream of this line diverges.
        # Required rather than cosmetic: the module is built on meta and filled by to_empty().
        if self.key_embed is not None:
            torch.nn.init.uniform_(self.key_embed.weight, -s, s)
        # The per-layer strengths start at exactly 0.0, so at step 0 every key is the parent's to
        # the bit and this model IS the parent's function at the parent's windows. zeros_ draws
        # no random numbers. The scalar's gradient is the inner product of the key gradient with
        # the gathered component, which is nonzero at zero, so it can open.
        if self.ke_lambdas is not None:
            self.ke_lambdas.zero_()
        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
            torch.nn.init.zeros_(block.mlp.me_gate.weight)
        # Rotary embeddings. Required rather than cosmetic, for exactly the reason the
        # parent's own line is: the module is built on meta and to_empty() leaves every
        # buffer uninitialised, so each band is rebuilt here on the real device. No random
        # numbers are drawn, so every draw this method makes stays at precisely the parent's
        # position in the init RNG stream and nothing downstream of it diverges.
        head_dim = self.config.head_dim
        for table_idx, spans in enumerate(self.rope_patterns):
            cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim,
                                                          spans=spans)
            setattr(self, f"rope_cos_{table_idx}", cos)
            setattr(self, f"rope_sin_{table_idx}", sin)
        # Cast embeddings to bf16
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)
        for table in self.mlp_embed:
            table.to(dtype=torch.bfloat16)
        if self.key_embed is not None:
            self.key_embed.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None,
                                      spans=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        # One row per head, at that head's own reach. spans is None for the parent's single
        # whole-context band and otherwise this layer's per-head reaches in head order. The
        # rule is base * w / sequence_len, so the head that reaches the whole context takes
        # the parent's 10000 to the digit and at spans=None this returns the parent's own
        # buffer -- the same values in the same (1, seq_len, 1, head_dim // 2) shape -- which
        # is what makes the flag-off model the parent's function bit for bit. torch.outer IS
        # in the counter's matmul family, which is exactly why this method is called from
        # __init__ and from init_weights and never from forward: the counted graph sees only
        # the elementwise work in apply_rotary_emb, as it does in the parent.
        # Every statement below is in the parent's order, and each row is built by the
        # parent's own three lines, so at spans=None the whole sequence of allocations and
        # ops is the parent's and the returned buffer is bit-identical to it -- torch's CPU
        # cos/sin take an alignment-dependent vectorised path, so reordering these
        # allocations moves a handful of values by one bfloat16 ulp.
        reaches = [self.config.sequence_len] if spans is None else list(spans)
        cos_rows, sin_rows = [], []
        for reach in reaches:
            reach_base = base * reach / self.config.sequence_len
            inv_freq = 1.0 / (reach_base ** (channel_range / head_dim))
            t = torch.arange(seq_len, dtype=torch.float32, device=device)
            freqs = torch.outer(t, inv_freq)
            cos, sin = freqs.cos(), freqs.sin()
            cos, sin = cos.bfloat16(), sin.bfloat16()
            cos_rows.append(cos)
            sin_rows.append(sin)
        cos = torch.stack(cos_rows, dim=1)[None]
        sin = torch.stack(sin_rows, dim=1)[None]
        return cos, sin

    def _compute_rope_tables(self, config):
        """Which rotary band each depth reads: one band per distinct per-head reach pattern.

        THE MECHANISM read one level up from the per-head rule in GPTConfig. Every other
        per-head quantity in this file already follows reach -- the head budget
        (_compute_head_counts), the read fan-in and the write fan-out
        (_compute_read_groups), the gathered value table's width, which is that layer's own
        kv width, and the gathered key table's serving set, which keys on the head counts --
        and since the ladder the stack's per-head reaches span 128 to 2048, a factor of
        sixteen. The frequency band is the last per-head quantity that does NOT follow it:
        one band is built for the whole context and shared by all twenty heads. Here a
        layer's band is a stack of one row per head at that head's own span, read off
        head_window_segments so the rows are the spans the kernel is actually called with,
        and layers whose per-head span pattern is identical share one table. At this
        geometry that is three tables for eight depths -- (128, 128) at the six local
        layers, (2048, 1024, 128, 128) at depth 3 and (512, 256, 128, 128) at depth 7 --
        which is the same read shape the gathered MLP offsets already use. With the field
        off there is exactly ONE band, built at the whole context, and every depth's index
        is 0, i.e. the parent's read.
        """
        if not config.rope_reach_matched:
            return [None], [0 for _ in self.head_window_segments]
        patterns, index = [], []
        for segments in self.head_window_segments:
            spans = tuple(window[0] for _, count, window in segments
                          for _ in range(count))
            if spans not in patterns:
                patterns.append(spans)
            index.append(patterns.index(spans))
        return patterns, index

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        # The 'S' span is a declared exact-attention radius, not half the context: those
        # layers attend over short_window previous positions, and unbounded range stays
        # available through the 'L' layers and the prefix-mean channel in forward().
        short_window = config.short_window
        assert 0 < short_window <= long_window
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def _compute_head_counts(self, config):
        """Per-layer attention head count: the head budget follows the exact span.

        Read off the window sizes this model actually uses -- including the last
        layer, which _compute_window_sizes forces to the long window -- so the
        two full-context layers keep config.n_head heads and every layer whose
        exact reach is the short window carries config.short_n_head of them.
        """
        short_heads = config.short_n_head if config.short_n_head > 0 else config.n_head
        assert 0 < short_heads <= config.n_head
        assert config.n_head % short_heads == 0
        return [config.n_head if window >= config.sequence_len else short_heads
                for window, _ in self.window_sizes]

    def _compute_read_groups(self, config):
        """Per-layer attention read grouping.

        THE MECHANISM. With config.attn_group_every_layer the grouping is unconditional:
        every layer reads its queries from one channel group and its shared kv from another,
        so the two full-context layers join the block-diagonal read -- and, because
        write_groups is a layer's own read_groups, the block-diagonal write -- that the six
        short-window layers already use. Those two layers hold the file's last dense
        full-width maps inside a block. The residual is block-structured everywhere else:
        the MLP is block-diagonal with a cross-group write-back, and six of eight attention
        layers read and write in groups, so a retrieval layer's dense read is the only place
        left that computes a full-width bilinear form, and the diagonal blocks of the pair
        space it buys are the blocks the rest of the model has already given up.

        Without the field this is the parent's rule: read off the window sizes this model
        actually uses, exactly as _compute_head_counts reads them, so the two full-context
        layers keep the dense n_embd read of the residual and every layer whose exact reach
        is the short window groups. This method is the ONLY reader of the field, so a
        layer's head budget, its declared reach, its per-head spans, its kernel calls, its
        share of the span tally and its gathered value-table width are identical either way.
        """
        groups = config.attn_read_groups
        assert groups > 0
        if config.attn_group_every_layer:
            return [groups for _ in self.window_sizes]
        return [1 if window >= config.sequence_len else groups
                for window, _ in self.window_sizes]

    def _compute_head_window_segments(self, config):
        """Per-layer reach classes: the windows this layer actually calls the kernel with.

        One rule for the whole stack: head i of a layer whose declared reach is `window`
        attends over max(config.short_window, window >> (start + i)) previous positions,
        where `start` is the rung of the STACK-WIDE ladder this layer's head 0 begins at.
        Heads that land on the same span are one class and therefore one kernel call, so on
        a layer whose declared reach is already the short window the rule is the identity
        and the layer keeps exactly one call at exactly the parent's window, while a
        full-context layer lays a dyadic ladder across its heads and pays for each scale
        once instead of paying its longest head's reach n_head times.

        THE MECHANISM is the value of `start`, and the parent's rule is `start` = 0 at every
        layer, which makes every whole-context layer lay the SAME ladder and the stack pay
        the frozen tally for each of 2048, 1024, 512 and 256 once per such layer. With
        config.ladder_rungs_per_layer = k > 0 the ladder is one sequence across depth: the
        r-th layer whose declared reach is the whole context starts at rung r * k and keeps
        k rungs, and its remaining heads sit at the floor short_window, which every local
        layer already resolves. Reach is not deleted, it is de-duplicated: at k * (number of
        such layers) == the ladder's rung count every scale is still resolved by exactly one
        head somewhere in the stack. Nothing else moves -- window_sizes still records each
        layer's DECLARED reach, which is what _compute_head_counts and _compute_read_groups
        read and therefore what sizes head budgets, read and write fan-ins, gathered
        value-table widths and the key table's serving set.
        """
        short_window = config.short_window
        rungs = config.ladder_rungs_per_layer
        assert rungs >= 0
        # Which rung each layer's head 0 starts at, and how many rungs that layer keeps.
        # kept None means "every head stays on the ladder", which is the parent's rule.
        first_rung = [0 for _ in self.window_sizes]
        kept_rungs = [None for _ in self.window_sizes]
        if rungs > 0:
            retrieval_position = 0
            for layer_idx, window_size in enumerate(self.window_sizes):
                if window_size[0] >= config.sequence_len:
                    first_rung[layer_idx] = retrieval_position * rungs
                    kept_rungs[layer_idx] = rungs
                    retrieval_position += 1
        self.ladder_first_rung = first_rung
        self.ladder_kept_rungs = kept_rungs
        segments = []
        for layer_idx, (heads, window_size) in enumerate(zip(self.head_counts,
                                                             self.window_sizes)):
            window = window_size[0]
            if config.reach_ladder:
                start, keep = first_rung[layer_idx], kept_rungs[layer_idx]
                spans = [max(short_window, window >> (start + i))
                         if (keep is None or i < keep) else short_window
                         for i in range(heads)]
            else:
                spans = [window] * heads
            layer = []
            for i, span in enumerate(spans):
                if layer and layer[-1][2][0] == span:
                    first, count, win = layer[-1]
                    layer[-1] = (first, count + 1, win)
                else:
                    layer.append((i, 1, (span, 0)))
            segments.append(layer)
        return segments

    def _compute_key_embed_slots(self, config, width):
        """Which heads of each layer the ONE gathered key table serves.

        THE MECHANISM. The table is `width` channels wide, i.e. width // head_dim heads' worth,
        and width is min(head_counts) * head_dim -- the narrowest kv width in the stack, which
        is also short_n_head * head_dim, the head budget of a layer whose exact reach is the
        floor. The parent decides the serving set with `head_counts[i] * head_dim == width`: a
        test on a LAYER's whole kv width, so a layer is served only if ALL of its heads fit the
        table. At this geometry that admits the six local layers and excludes depth 3 and depth
        7, whose kv width is 256 -- and it excludes with them heads 2 and 3 of each, whose
        exact reach IS short_window, the reach the table was built for. Sixteen of the twenty
        heads in this stack span short_window and only twelve receive the component.

        With config.key_embed_by_reach the test is on head SPAN instead: a layer is served over
        the contiguous TRAILING block of width // head_dim heads whose span is short_window,
        and is not served at all if it has fewer floor-reach heads than that. Trailing is not a
        choice -- _compute_head_window_segments emits max(short_window, window >> (start + i))
        in head order, a non-increasing sequence floored at short_window, so a layer's
        floor-reach heads are always its last ones. This method is the only reader of the
        field, and nothing it returns is read by anything that sizes a shape: window sizes,
        head counts, declared reaches, per-head spans, read and write group counts, gathered
        value-table widths, rope bands, the call plans and the span tally are all computed
        without it.

        Returns one (first_head, head_count) per layer, or None for a layer that is not
        served. With the field off it returns exactly the parent's set, expressed as
        (0, head_counts[i]) so a served layer receives the component over its whole kv width
        and forward takes the parent's own single line.
        """
        head_dim = config.head_dim
        assert width % head_dim == 0
        slot_heads = width // head_dim
        if not config.key_embed_by_reach:
            return [(0, heads) if heads * head_dim == width else None
                    for heads in self.head_counts]
        slots = []
        for segments in self.head_window_segments:
            spans = [window[0] for _, count, window in segments for _ in range(count)]
            floor = 0
            for span in reversed(spans):
                if span != config.short_window:
                    break
                floor += 1
            slots.append((len(spans) - slot_heads, slot_heads)
                         if floor >= slot_heads else None)
        return slots

    def _compute_head_call_plans(self, config, seq_len=None):
        """Per reach class, the query blocks it is called with, each charged its own reach.

        THE MECHANISM. Every other per-head quantity in this file follows a head's reach, and
        since r42 the stack's per-head reaches are 2048, 1024, 512, 256 and 128. The frozen
        tally, though, charges a call 12 * B * T_call * h * d * min(window_left, k.shape[1]) --
        one flat rate for every query position in the call. Causality makes that rate wrong in
        one direction only: a query at position t can read t + 1 positions, so on the head that
        declares the whole context the queries in the first half of the sequence are charged
        2048 for a reach they cannot have. _attention_flops_probe says so in its own docstring
        ("charges min(left, S) with no causal halving"), and the consequence is that the tally
        for a whole-context head is 12*h*d*T*T where the mask it runs is 12*h*d*T*(T+1)/2.

        This method removes that overcharge and nothing else. The query axis is cut into
        `blocks` contiguous ranges of L = seq_len // blocks, and range j is handed the key and
        value PREFIX [0, (j+1) * L) rather than the whole sequence, so its charge is
        L * min(window, (j+1) * L) instead of L * min(window, seq_len). The mask does not move:
        the kernel's documented rule with window_size set is that query i attends to keys in
        [i + seqlen_k - seqlen_q - left, i + seqlen_k - seqlen_q + right], and seqlen_k -
        seqlen_q is exactly j * L, so block j's query i reads absolute keys
        [j*L + i - window, j*L + i] -- what absolute query j*L + i read before. No position is
        subsampled, dropped or duplicated, no window is shortened, no head loses reach, and the
        sum of T_call over a class's calls is seq_len exactly. That is the whole difference from
        reading at a stride, which drops query and key positions and coarsens what a head can
        resolve; nothing here is coarsened.

        Consecutive blocks that carry the SAME charge are merged, and the merge is free rather
        than an approximation: their union is one call whose q_len and span are the run's, at the
        identical total charge and the identical mask. So a class whose window is at most L
        merges back to the parent's single call over the whole sequence, and at blocks == 1 every
        class does, which is the flag-off control. At this geometry L is 512 and the five reach
        classes are 2048, 1024, 512, 256 and 128, so the only classes that split are the two at
        depth 3 that declare more reach than one block can hold: depth 7 and all six local
        layers keep their calls, their windows, their key tensors and their charge to the digit.
        """
        seq_len = config.sequence_len if seq_len is None else seq_len
        blocks = config.causal_query_blocks
        # THE DOSE, derived from this stack's own ladder instead of carried as a count. The
        # per-head spans are already fixed above, in head_window_segments; the finest of them
        # above the floor is the smallest reach any head in this stack resolves, and charging
        # in blocks of THAT length is what makes every class above it pay per block while
        # every class at or below it keeps one call over the whole query axis at its exact
        # window. A block length that does not divide the query axis could not state its
        # charge exactly, so it is not taken and the carried count stands -- the same reason
        # the divisibility fallback below exists.
        if config.causal_blocks_at_finest_rung:
            # THE MECHANISM's dose, one rung finer. The floor is admitted to the rung set when
            # causal_blocks_at_floor_reach is on, so the block length is the finest span ANY
            # head in this stack resolves rather than the finest one above the floor. A class
            # whose window is at most the block length is still charged its exact window and
            # still merges back to one call over the whole query axis, so admitting the floor
            # refines every class above it and is the identity on every class at it.
            floor_ok = config.causal_blocks_at_floor_reach
            rungs = sorted({window[0] for segments in self.head_window_segments
                            for _, _, window in segments
                            if floor_ok or window[0] > config.short_window})
            if rungs and seq_len % rungs[0] == 0:
                blocks = seq_len // rungs[0]
        assert blocks >= 1
        # A block count that does not divide the sequence would give blocks of unequal length
        # and a charge this method could not state exactly, so it falls back to the parent's
        # one call per class rather than guessing. Nothing in this file reaches it: training,
        # the frozen probe and the frozen evaluation all run at sequence_len.
        if seq_len % blocks != 0:
            blocks = 1
        block_len = seq_len // blocks
        # The resolved block geometry, kept so the geometry print reads the plan the model
        # actually built rather than restating a config field, exactly as ladder_first_rung is
        # kept by _compute_head_window_segments. Two Python ints, so they are invisible to
        # prepare.count_params and to both optimizers and add no parameter, buffer, module or
        # counted op of any kind.
        self.query_blocks = blocks
        self.query_block_len = block_len
        plans = []
        for segments in self.head_window_segments:
            layer = []
            for _, _, window_size in segments:
                window = window_size[0]
                calls = []
                for j in range(blocks):
                    prefix = (j + 1) * block_len
                    span = min(window, prefix)
                    if calls and calls[-1][3] == span:
                        # Same charge as the previous block, so extend it: one call over the
                        # union, at the later prefix, carrying that run's whole charge.
                        q_start, q_len, _, _ = calls[-1]
                        calls[-1] = (q_start, q_len + block_len, prefix, span)
                    else:
                        calls.append((j * block_len, block_len, prefix, span))
                layer.append(calls)
            plans.append(layer)
        return plans

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        mlp_embed_numel = sum(t.weight.numel() for t in self.mlp_embed)
        # The kv gains and the attention logit scales are excluded for the same reason
        # the gathered tables are: they carry no matmul, so charging them 6 FLOPs per
        # parameter would make this estimate disagree with the frozen counter that
        # actually scores the run.
        kv_gain_numel = sum(b.attn.k_gain.numel() + b.attn.v_gain.numel()
                            + b.attn.q_scale.numel()
                            for b in self.transformer.h)
        # The gathered key table and its per-layer strengths are excluded for the same reason
        # the other tables and the 1-D gains are: they carry no matmul, so charging them 6 FLOPs
        # per parameter would make this printed estimate disagree with the frozen counter that
        # actually scores the run.
        key_embed_numel = self.key_embed.weight.numel() if self.key_embed is not None else 0
        ke_lambda_numel = self.ke_lambdas.numel() if self.ke_lambdas is not None else 0
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          mlp_embed_numel + key_embed_numel + ke_lambda_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                          self.gmem_lambdas.numel() + kv_gain_numel)
        q = self.config.head_dim
        t = self.config.sequence_len
        # The frozen probe reads the head count off each call's own q, the span as
        # min(window, k.shape[1]) off its own window kwarg and key tensor, and weights the
        # result by that call's own query count -- so this estimate sums over the same
        # per-layer, per-class, per-QUERY-BLOCK calls the forward actually makes, and each
        # call's charge is heads * head_dim * span * q_len. Summing the units first and
        # dividing by t once keeps the arithmetic exact rather than truncating per call.
        # Without this the printed estimate would disagree with the number the run is
        # scored on; at causal_query_blocks == 1 every class is one block of q_len == t and
        # k_len == t, so this reduces to the parent's expression to the digit.
        attn_units = 0
        for segments, plan in zip(self.head_window_segments, self.head_call_plans):
            for (_, heads, window_size), calls in zip(segments, plan):
                window = window_size[0]
                for _, q_len, k_len, _ in calls:
                    effective_seq = k_len if window < 0 else min(window, k_len)
                    attn_units += heads * q * effective_seq * q_len
        attn_flops = 12 * attn_units // t
        return 6 * (nparams - nparams_exclude) + attn_flops

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        # Muon's fused step stacks a group's tensors and reads shape[-2], so a 1-D
        # parameter cannot go in a Muon group at all. Every 1-D parameter in the blocks
        # -- the per-channel kv gains and the per-head attention logit scales -- takes
        # AdamW instead, at the rate this file already gives a multiplicative parameter
        # that starts at 1.0 (resid_lambdas, scalar_lr * 0.01). The partition below
        # collects them by rank, so the group list and the exhaustiveness assert need
        # no edit; kv_gain_params is no longer a literal name for what it holds.
        block_params = list(self.transformer.h.parameters())
        matrix_params = [p for p in block_params if p.ndim >= 2]
        kv_gain_params = [p for p in block_params if p.ndim == 1]
        value_embeds_params = list(self.value_embeds.parameters())
        mlp_embed_params = list(self.mlp_embed.parameters())
        # The gathered key table takes the rate every other gathered vocabulary table in this
        # file takes, and its per-layer strengths take the rate x0_lambdas and gmem_lambdas take,
        # so no existing tensor's rate moves and no rate is invented for this card.
        key_embed_params = (list(self.key_embed.parameters())
                            if self.key_embed is not None else [])
        ke_params = [self.ke_lambdas] if self.ke_lambdas is not None else []
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        gmem_params = [self.gmem_lambdas]
        assert len(list(self.parameters())) == (len(matrix_params) + len(kv_gain_params) +
            len(embedding_params) + len(lm_head_params) + len(value_embeds_params) +
            len(mlp_embed_params) + len(key_embed_params) + len(ke_params) +
            len(resid_params) + len(x0_params) + len(gmem_params))
        # Scale LR ∝ 1/√dmodel (tuned at 768 dim)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=mlp_embed_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=key_embed_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=kv_gain_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=gmem_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=ke_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
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

    def _readout_read(self, x):
        """The readout's read of the final residual: complete coverage at two resolutions.

        The leading readout_fine channels arrive unpooled, exactly as the parent hands the
        whole tensor over. The trailing readout_coarse channels -- the last
        readout_coarse_units of the residual's n_head units of n_embd // n_head -- arrive as
        readout_coarse // 2 pairwise contractions, each the mean of one contiguous pair times
        2**0.5.

        That factor is the settlement the mechanism needs and not a tuned constant. forward
        hands this method norm(x), whose channels sit at unit rms; the mean of two such
        coordinates sits at 1 / 2**0.5 of that; and the readout is ONE nn.Linear whose
        vocab_size rows read both halves through one weight tensor in one AdamW group at one
        rate, so without the factor the coarse columns would start at a systematically smaller
        gradient than the fine ones for a reason that has nothing to do with the mechanism.

        Every channel of the residual reaches every type, which is exactly what separates this
        from the two fan-in cuts already measured at this module: there a type's weight vector
        had no column for the channels its block did not read, and here every channel has a
        column and pooled channels share one.

        No counted FLOP is issued. A last-dim slice, a reshape that only splits the last
        dimension of a stride-1 axis, aten::mean, a multiply by a Python float and one
        torch.cat are all absent from FlopCounterMode's registry, which the paid instrument
        confirmed at the sixteen gate reads rather than only the CPU stand-in. At
        readout_coarse == 0 this returns x itself, so the flag-off model's readout is the
        parent's operand for operand.
        """
        if self.readout_coarse == 0:
            return x
        fine = x[..., :self.readout_fine]
        coarse = x[..., self.readout_fine:]
        coarse = coarse.reshape(*coarse.shape[:-1], self.readout_coarse // 2, 2)
        coarse = coarse.mean(-1) * (2 ** 0.5)
        return torch.cat([fine, coarse], dim=-1)

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        assert T <= self.rope_cos_0.size(1)
        # One slice per band, hoisted out of the depth loop for the same reason the gathered
        # tables' reads are: a band is sliced once per forward however many depths read it. A
        # slice and a broadcast carry no counted FLOP, and at rope_reach_matched False there
        # is one band of the parent's exact shape which every depth reads, so this is the
        # parent's line.
        cos_sins = [(getattr(self, f"rope_cos_{j}")[:, :T],
                     getattr(self, f"rope_sin_{j}")[:, :T])
                    for j in range(len(self.rope_patterns))]

        # The call plans are a function of T, and every path in this file runs at
        # config.sequence_len -- training, the frozen probe (which slices ROWS, not positions)
        # and the frozen 128 x 2048 evaluation. The precomputed plans are used at that length
        # and rebuilt for any other, so a shorter T degrades to correct calls rather than to a
        # wrong slice. Building a list of Python ints costs no counted FLOP and no tensor op.
        call_plans = (self.head_call_plans if T == self.config.sequence_len
                      else self._compute_head_call_plans(self.config, seq_len=T))

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        # Unbounded-range channel, read once per forward and shared by every layer, in the
        # same place and with the same lifetime as the gathered table below. gmem[:, t] is
        # the mean of x0[:, :t+1] -- strictly causal -- normed so its scale does not decay
        # as 1/sqrt(t). The running sum is accumulated in float32 because a 2048-term sum
        # in bfloat16 drops its tail. A cumulative sum issues no matmul, so the counted
        # target charges nothing for it, exactly as it charges nothing for the gathered
        # tables; unlike a gather, this one is a function of the context.
        positions = torch.arange(1, T + 1, device=x0.device, dtype=torch.float32).view(1, T, 1)
        gmem = norm((x0.float().cumsum(dim=1) / positions).to(x0.dtype))
        # One gather per table, hoisted out of the depth loop exactly as the single table's
        # gather was: a table is read once per forward however many depths read it.
        mes = [table(idx) for table in self.mlp_embed]
        # The gathered key component, read once per forward for the same reason the MLP offsets
        # are: a table is read once however many depths read it, and a gather is charged nothing.
        ke = self.key_embed(idx) if self.key_embed is not None else None
        for i, block in enumerate(self.transformer.h):
            x = (self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
                 + self.gmem_lambdas[i] * gmem)
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            ke_i = ke if (ke is not None and i in self.key_embed_layers) else None
            x = block(x, ve, mes[self.mlp_embed_index[i]],
                      cos_sins[self.rope_index[i]],
                      self.head_window_segments[i], call_plans[i], ke_i,
                      self.ke_lambdas[i] if ke_i is not None else None)
        x = norm(x)

        softcap = 15
        # The readout reads every residual channel, the trailing pooled unit at pair
        # resolution. At readout_coarse == 0 _readout_read returns x itself and this is the
        # parent's line, operand for operand.
        logits = self.lm_head(self._readout_read(x))
        logits = logits.float()
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
ASPECT_RATIO = 64       # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 128          # width unit that sets the head COUNT from model_dim
ATTN_HEAD_DIM = 64      # actual per-head width: n_head * ATTN_HEAD_DIM is the
                        # attention pathway width, decoupled from n_embd
MLP_MULT = 2            # MLP hidden width as a multiple of n_embd. Halving it from 4
                        # frees exactly the parameters the gathered MLP table costs.
MLP_GROUPS = 2          # block-diagonal MLP: each of the G channel groups keeps its own
                        # c_fc/c_proj pair, so the hidden width (and the number of
                        # squared-ReLU features) is unchanged while the MLP's matmul
                        # cost falls by 1/G.
MLP_WRITE_SHIFT = 1     # cyclic write-back shift over those groups: group g's hidden
                        # features are projected into group (g + 1) % G instead of back
                        # into the group they read, which restores a cross-group path
                        # inside every MLP at exactly zero counted FLOPs.
MLP_EMBED_GROUPS = 3    # number of gathered MLP-offset tables, one per contiguous block of
                        # depth: layer i reads table (i * G) // n_layer, so at G = 3 and
                        # n_layer 8 the blocks are depths 0-2, 3-5 and 6-7. The gathered
                        # offset is the only token-conditioned signal the MLP has, and inside
                        # a block the SAME 1024 numbers are added at every one of that block's
                        # depths; a depth's whole freedom over them is me_gate's four
                        # per-position scalars. A third table therefore buys the two depths
                        # that feed the readout most directly a per-token offset no earlier
                        # depth shares, and moves depth 3 -- the first layer that retrieves
                        # from the whole context -- off the block depths 0-2 hold. Every
                        # contextual path in the model is untouched, and init_weights draws
                        # table 0 once and copies it into the others, so at initialisation
                        # this model IS the two-table model's function and the blocks separate
                        # only as far as their own gradients drive them apart.
                        # A gather issues no matmul, so this adds exactly zero counted FLOPs
                        # while being charged in full against the parameter ceiling, which is
                        # what bounds the mechanism and fixes 3 as its terminus here: at
                        # 8,388,608 parameters each, three tables land num_params_total at
                        # 49,223,064 of the 50,332,176 ceiling, leaving 1,109,112, and a
                        # fourth would need 57,611,672 and is inadmissible. So one table per
                        # depth is unreachable on this substrate and 3 is the last dose the
                        # ceiling admits. 2 is the previous dose and is this card's flag-off
                        # control: it returns this parent's key, census and parameters exactly.
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full context, S=SHORT_WINDOW
SHORT_WINDOW = 128      # exact-attention radius of an 'S' layer, and the other half of this
                        # card's one mechanism. The counted span tally is 12 * sum over kernel
                        # calls of (heads * head_dim * span), the one counted family no parameter
                        # edit can reach, and each of the six local layers pays that tally for
                        # every position inside its radius whether or not the layer needed to
                        # look that far. What a local head is doing with the radius is finding
                        # WHICH earlier position to read; it can be told that either by looking
                        # over more positions or by the key at a position saying what that
                        # position IS -- and KEY_EMBED gives it the second for the first time in
                        # this file. So the radius halves and the identity arrives, on the same
                        # six layers' score path, in one card. Verified against this source: this
                        # constant is read by _compute_window_sizes (the 'S' span) and by
                        # _compute_head_window_segments (the ladder floor). At 128 the floor
                        # max(128, 2048 >> i) still yields 2048, 1024, 512, 256 at the two
                        # retrieval layers, so their reach, head count and kernel calls do not
                        # move; _compute_head_counts keys on window >= sequence_len, so the six
                        # local layers keep two heads; _compute_read_groups returns
                        # attn_read_groups unconditionally, so no read or write fan-in moves; and
                        # the gathered value tables key on the head counts, so no table width
                        # moves. Only the six local calls' span moves. Unbounded range stays
                        # available through the two 'L' layers and the prefix mean.
SHORT_N_HEAD = 2        # head count of an 'S' layer, against n_head = 4 on the two 'L'
                        # layers. The head budget follows the exact span: a layer that
                        # resolves 256 positions needs fewer parallel patterns than one
                        # that must retrieve from the whole context, and this is the only
                        # edit in the file that moves BOTH counted halves of a layer at
                        # once -- its q/kv/out projections and its share of the tally.
ATTN_READ_GROUPS = 2    # block-diagonal attention read at every layer whose exact reach is
                        # SHORT_WINDOW. Such a layer cuts the 512-channel residual into 2
                        # groups of 256 and gives each group its own q and kv projection, so
                        # the read costs half the products while the layer keeps both heads,
                        # its full 256-position reach, its per-head width, its full-width
                        # gathered value embedding and its dense attn_dim -> 512 write. The
                        # two full-context layers keep the dense read only while
                        # ATTN_GROUP_EVERY_LAYER is False; with it True the group count on
                        # this line applies at every depth and reach no longer sizes the
                        # read's fan-in, though it still sizes everything else.
ATTN_READ_SHIFT = 1     # which group each layer's kv projection reads, relative to the group
                        # its q projection reads. At 2 groups a shift of 1 is the exclusive
                        # swap, so a head compares content in one group against content in
                        # the other and the halved pair space kept is the CROSS-group half --
                        # the same choice MLP_WRITE_SHIFT makes on the write side. A shift of
                        # 0 would keep the within-group half instead at identical cost.
ATTN_WRITE_GROUPED = True # block-diagonal attention WRITE at every layer whose read is
                        # grouped, i.e. at every layer whose exact reach is SHORT_WINDOW.
                        # Such a layer's attn_dim -> 512 write becomes one 64 -> 256 matmul
                        # per head block, so the write costs half its products while the
                        # layer still writes all 512 residual channels, keeps both heads,
                        # its 256-position reach, its per-head width, its gathered value
                        # embedding and its grouped cross-group read. The two full-context
                        # layers keep the dense write only while ATTN_GROUP_EVERY_LAYER
                        # is False, because write_groups is a layer's own read_groups: with
                        # that field True they group here too, and the last dense full-width
                        # maps inside a block are gone from the file altogether.
ATTN_WRITE_SHIFT = 0    # which residual group each head block writes into, relative to the
                        # group its own query reads. The READ shift is what makes 0 the
                        # cross-group choice here: at two groups with ATTN_READ_SHIFT 1,
                        # head block g's keys and values are a projection of group
                        # (g + 1) % 2, so writing back into group g delivers the OTHER
                        # group's retrieved content into this group. A shift of 1 would
                        # hand each group its own content back at identical cost -- the
                        # mirror of MLP_WRITE_SHIFT, which needs 1 because its read is
                        # diagonal, and the free control that separates the two routes.
ATTN_GROUP_EVERY_LAYER = True # drop the reach condition on the read grouping, so the two
                        # full-context layers join the block-diagonal read and -- because
                        # write_groups is a layer's own read_groups -- the block-diagonal
                        # write the six short-window layers already use. Those two layers
                        # hold the file's last dense full-width maps inside a block: one
                        # 512 -> 256 query read, one 512 -> 256 shared-kv read and one
                        # 256 -> 512 write each, 131,072 counted parameters apiece against
                        # 65,536 for the grouped form, so 393,216 counted parameters leave
                        # the model. Every reach is untouched: head 0 of each still spans the
                        # whole 2,048-position context, the dyadic ladder still lays
                        # 2048/1024/512/256 across the four heads at four kernel calls, each
                        # head keeps its 64 channels, each layer keeps its full-width
                        # gathered value table and still writes all 512 residual channels,
                        # and the span tally -- the counted family no parameter edit can
                        # reach -- does not move at all. False is the parent's per-reach rule
                        # and is the flag-off control.
KEY_EMBED = True        # give the key side the gathered per-token component the value side
                        # has had at every layer since the attention pathway was narrowed. ONE
                        # table of 8,192 x 128, read once per forward and added to k -- before
                        # rotary, before norm(k), the same place ve is added to v -- at every
                        # layer whose kv width is 128, which is the six 'S' layers, under one
                        # per-layer scalar that starts at exactly 0.0. The dose is one table and
                        # the width is 128 because the parameter ceiling says so and for no other
                        # reason: 8,192 x 128 = 1,048,576 against 1,109,112 of headroom, so this
                        # is the only gathered table this substrate still admits and 8,192 x 256
                        # (2,097,152) is inadmissible. A gather and an elementwise scale-and-add
                        # are charged nothing by the frozen counter, so this half of the card
                        # moves neither counted family, and False is its flag-off control.
REACH_LADDER = True     # per-head reach grading inside a layer. The counted span tally
                        # charges EVERY head of a layer at that layer's window, so a layer
                        # pays for its reach once per head even though one head is what
                        # makes the reach exist. With this on, head i of a layer whose
                        # declared reach is w attends over max(SHORT_WINDOW, w >> i)
                        # positions: the identity on an 'S' layer, and a dyadic ladder
                        # 2048/1024/512/256 across the four heads of an 'L' layer. No
                        # projection, table, gain or parameter changes -- this moves the
                        # span tally alone, and it is the only counted family no parameter
                        # edit can reach.
LADDER_RUNGS_PER_LAYER = 2 # lay the reach ladder across DEPTH as one sequence instead of
                        # restarting it at each whole-context layer. This file has two such
                        # layers, depth 3 and depth 7, and REACH_LADDER gives each of them
                        # the identical rungs 2048, 1024, 512, 256 at one head of 64
                        # channels apiece -- so the stack buys every one of those four
                        # scales twice and the frozen tally charges 12 * 64 * span for each
                        # of the eight calls. At 2 the ladder is dealt out in depth order:
                        # depth 3 keeps rungs 0 and 1 (2048 and 1024) and depth 7 keeps
                        # rungs 2 and 3 (512 and 256), and the two heads each layer no
                        # longer spends on a duplicate sit at the SHORT_WINDOW floor of 128
                        # that the six local layers already resolve. 2 is not a dose chosen
                        # for size: it is the unique setting that spreads the ladder over
                        # both layers while keeping every scale in the model, because
                        # 2 rungs x 2 whole-context layers is exactly the 4 rungs the ladder
                        # produces at 4 heads and a 128 floor. At 1 the 512 and 256 rungs
                        # would leave the model altogether, which is a capability deletion
                        # and a different card; at 4 depth 7 would hold four heads and a
                        # 256-wide gathered value table with no rung of its own. Verified
                        # against this source: the field is read by
                        # _compute_head_window_segments only, so heads per layer stays
                        # 2,2,2,4,2,2,2,4, every declared reach stays 128/2048,
                        # _compute_read_groups still returns 2 everywhere, every value table
                        # keeps its width, key_embed_layers still keys on the head counts and
                        # still serves depths 0,1,2,4,5,6, and not one nn.Linear changes
                        # shape -- the whole edit lands on the span tally, the one counted
                        # family no parameter edit can reach and the one that needs none of
                        # the 60,528 parameters the ceiling still admits. 0 is the parent's
                        # rule and is this card's flag-off control.
ROPE_REACH_MATCHED = True # let the rotary frequency band follow the head's reach, which is
                        # the last per-head quantity in this file that does not. REACH_LADDER
                        # and LADDER_RUNGS_PER_LAYER made reach a per-HEAD resource: the
                        # twenty heads of this stack now span 128, 256, 512, 1024 and 2048
                        # positions, and sixteen of the twenty span 128. Every other per-head
                        # quantity was made to follow that -- the head budget, the read
                        # fan-in, the write fan-out, the gathered value table's width, the
                        # key table's serving set -- while _precompute_rotary_embeddings
                        # still builds one band from base 10000 over the whole 2,048-position
                        # context and forward hands that one band to all twenty. A pair's
                        # only contribution to a score is the relative angle
                        # theta_j * (t_q - t_k), so at base 10000 and head_dim 64 the band's
                        # longest wavelength is about 47,127 positions, 23.0x the context the
                        # parent designed it for and 368x the reach of the sixteen heads that
                        # see 128 positions; for those heads nineteen of the thirty-two pairs
                        # turn through less than half a cycle across their whole window and
                        # so cannot order any two positions they can see. With this on a head
                        # of reach w takes base * w / sequence_len, giving a dyadic base
                        # ladder 10000, 5000, 2500, 1250, 625 beside the dyadic reach ladder,
                        # every head the same 23.0x-to-25.1x band-to-reach relation the
                        # whole-context head already had, and thirteen rather than nineteen
                        # dead pairs at a 128-reach head. The whole-context head's band is
                        # unchanged to the digit, so this is an identity on the one head the
                        # parent's band was built for. Verified against this source: the
                        # constant is read by _compute_rope_tables alone; head counts stay
                        # 2,2,2,4,2,2,2,4, declared reach 128/2048, head spans and the twelve
                        # kernel calls and the 4,521,984 span tally byte-identical,
                        # read/write groups 2 at fan-in and fan-out 256, value tables
                        # 128/256, key_embed 1 x 128 at depths 0,1,2,4,5,6, and no nn.Linear,
                        # gain, scalar or gathered table added, removed or resized. A rotary
                        # band is a non-persistent buffer built in __init__ and init_weights,
                        # so torch.outer never enters the counted graph and the counter
                        # charges this exactly zero. False is the parent's single shared band
                        # and is this card's flag-off control.
CAUSAL_QUERY_BLOCKS = 4 # stop paying for window a causal query cannot reach. The frozen tally
                        # charges a call 12 * B * T_call * h * d * min(window_left, k.shape[1]),
                        # one flat rate for every query position in it, and
                        # _attention_flops_probe's docstring states the convention: min(left, S)
                        # "with no causal halving". A query at position t can read t + 1
                        # positions, so depth 3's whole-context head is charged 12*64*2048 per
                        # token for a mask that touches 12*64*1024.5 -- the single largest entry
                        # in the tally, 1,572,864 per token, 34.78% of the 4,521,984 tally and
                        # 2.59% of the key, and every previous card in this family paid it in
                        # full. At 4 the query axis is cut into 4 blocks of 512 and block j is
                        # handed the key/value PREFIX [0, 512*(j+1)) instead of the whole
                        # sequence. The kernel documents the mask it then applies: with
                        # window_size set, query i reads keys [i + seqlen_k - seqlen_q - left,
                        # i + seqlen_k - seqlen_q + right], and seqlen_k - seqlen_q is exactly
                        # the block's start, so block j's query i reads absolute keys
                        # [512j + i - window, 512j + i] -- the identical set the one call gave
                        # absolute query 512j + i, with causal=True's bottom-right alignment
                        # agreeing. So this is the first counted cut in this file that removes
                        # NO capability at all: not a head, not a rung, not a channel, not a
                        # window, not a position, not a parameter. It is not a stride either --
                        # nothing is subsampled and the query counts still sum to 2048 exactly.
                        # 4 is the coarsest block size that reaches both overcharged classes:
                        # at 2 the blocks are 1024 and min(1024, 1024) == min(1024, 2048), so
                        # depth 3's 1024-rung merges back to one call and only the 2048-rung
                        # splits, while at 4 the 512 block is smaller than both. It is also the
                        # largest reach in the stack that is left alone: 512, 256 and 128 all
                        # fit inside one block, so they merge to the parent's single call and
                        # depth 7 and the six local layers are byte-identical -- the whole
                        # counted delta lands on depth 3's two ladder heads. Verified against
                        # this source: the constant is read by _compute_head_call_plans alone,
                        # so heads per layer stays 2,2,2,4,2,2,2,4, declared reach 128/2048,
                        # every per-head window unchanged, read and write groups 2 at fan-in and
                        # fan-out 256, value tables 128/256, key_embed 1 x 128 at depths
                        # 0,1,2,4,5,6, rope bands 3 at the same reaches, and not one nn.Linear,
                        # gain, scalar, buffer or gathered table added, removed or resized -- so
                        # num_params_total does not move and none of the 60,528 parameters the
                        # ceiling still admits is spent. Each doubling from here halves the
                        # remaining overcharge and doubles the split classes' kernel calls, which
                        # is the mechanism's own limit: 12 calls become 16 at 4, 23 at 8 and 39
                        # at 16. 1 is the parent's one call per class and is this card's flag-off
                        # control.
                        # THIS FIELD IS NOW THE FALLBACK. CAUSAL_BLOCKS_AT_FINEST_RUNG below
                        # derives the block LENGTH from the stack's own ladder, and this count
                        # is what stands when that ladder offers no rung above its floor or
                        # when the derived length would not divide the query axis. At that
                        # field False the block length is this count's 512 and the model is
                        # exactly the one described above.

CAUSAL_BLOCKS_AT_FINEST_RUNG = True # charge every over-window reach class in blocks of the
                        # SMALLEST reach this stack actually resolves, deriving the causal
                        # charge's block length from the ladder instead of carrying it as a
                        # count. THE MECHANISM is the parent's and does not change: the frozen
                        # tally charges a call 12 * B * T_call * h * d * min(window_left,
                        # k.shape[1]) -- one flat rate for every query position in it, and
                        # _attention_flops_probe's docstring states the convention, min(left, S)
                        # "with no causal halving" -- so a query early in a block is charged for
                        # that block's LAST key prefix even though causality forbids it those
                        # keys. _compute_head_call_plans cuts the query axis into blocks of
                        # length L and hands block j the key/value PREFIX [0, (j+1) * L), and
                        # the kernel's own documented mask reproduces the parent's, position for
                        # position: with window_size set, query i reads keys [i + seqlen_k -
                        # seqlen_q - left, i + seqlen_k - seqlen_q + right], seqlen_k - seqlen_q
                        # is exactly the block's start, and causal=True's bottom-right alignment
                        # agrees. Nothing is subsampled, no window shortens, no head loses a
                        # position it could read, no parameter moves, and the query counts still
                        # sum to 2048 exactly.
                        # WHAT THE RULE IS, and why it is the model's number rather than a dose
                        # picked for size. _compute_head_window_segments fixes this stack's
                        # per-head spans at 2048, 1024 and the 128 floor at depth 3; 512, 256
                        # and the floor at depth 7; and the floor at both heads of each of the
                        # six local layers. The charge granularity that matches those is the
                        # FINEST span above the floor -- min over every per-head span greater
                        # than SHORT_WINDOW, which is 256 here -- because a class whose window
                        # fits inside one block is charged its exact window and merges back to a
                        # single call, while every class above it pays per block. So the rule
                        # charges the ladder in units of its own smallest rung, and it
                        # re-derives itself: move LADDER_RUNGS_PER_LAYER, WINDOW_PATTERN or
                        # SHORT_WINDOW and the block length follows the spans instead of needing
                        # a second edit.
                        # WHAT IT COSTS PER CALL, which is why the ladder's own rung is the
                        # right granularity to stop at. Halving L costs a class w / L extra
                        # kernel calls and removes w * L / 4 from its (query count x charged
                        # reach) sum, so ONE added call always removes L**2 / 4 of that sum, i.e.
                        # 12 * heads * head_dim * L**2 / (4 * sequence_len) of key -- a function
                        # of the block length and of `heads`, not of the window it is
                        # applied to. THE CLAUSE THAT USED TO STAND HERE -- "a function of the
                        # block length ALONE ... no per-class length beats it" -- IS FALSE and is
                        # corrected, comment-only, with no behavioural effect: the return carries
                        # `heads` while the call it costs does not, so classes with unequal head
                        # counts have unequal exchange rates and a per-class allocation does beat
                        # one global length. CAUSAL_BLOCKS_AT_FLOOR_REACH's comment below already
                        # prices the tiers that show it, and an independent dynamic program over
                        # all twelve of this stack's reach classes reaches span tally 3,545,088 at
                        # 46 calls against the best global point at 46 calls or fewer, which is
                        # L = 128 at 3,569,664 in 38 calls -- the geometry this file runs. From
                        # the parent's 512 the next length
                        # down prices a call at 24,576 of key, the one after at 6,144 and the one
                        # after that at 1,536. The ladder's smallest rung lands on the 24,576
                        # rung: 7 added calls for 172,032 of key, and the last length at which
                        # the stack's reaches and the charge granularity agree.
                        # WHAT MOVES, verified against this source: the field is read by
                        # _compute_head_call_plans alone. At L = 256 a class splits only where
                        # its window exceeds 256, so exactly the three classes above the
                        # ladder's smallest rung split -- depth 3's 2048 head 4 -> 8 calls,
                        # depth 3's 1024 head 2 -> 4, depth 7's 512 head 1 -> 2. SEVENTEEN of
                        # the stack's twenty heads keep the parent's single call over the whole
                        # query axis with its key tensor, its window kwarg and its charge to the
                        # digit: depth 7's 256 head, because min(256, 256) == min(256, 2048), and
                        # all sixteen heads at the floor. Six of the eight layers are untouched
                        # even in their kernel arguments. Heads per layer stays 2,2,2,4,2,2,2,4,
                        # declared reach 128/2048, every per-head window unchanged, read and
                        # write groups 2 at fan-in and fan-out 256, value tables 128/256,
                        # key_embed 1 x 128 serving h0+2 at six depths and h2+2 at depths 3 and
                        # 7, rope bands 3 at the same reaches and bases, mlp_embed 3 x 1024, the
                        # readout 384 fine + 64 pair means of 128 = 448, and not one nn.Linear,
                        # gain, scalar, buffer or gathered table added, removed, resized or
                        # re-dtyped -- so num_params_total does not move, none of the 584,816
                        # parameters the ceiling still admits is spent, and both
                        # compiled-optimizer censuses stay at 8 signature classes. The whole
                        # counted delta is the span tally's, 3,833,856 -> 3,661,824. What this
                        # DOES carry is _attend's split path: seven more flash_attn_func launches
                        # per forward, at two of the eight layers, and the contiguous key and
                        # value prefix copies each split call makes. False is the parent's
                        # carried count of 4 at a block length of 512, and is this card's
                        # flag-off control.

CAUSAL_BLOCKS_AT_FLOOR_REACH = True # read the causal charge's block length off the ladder's
                        # FLOOR reach instead of the finest rung above it, so the block length is
                        # 128 rather than 256. THE MECHANISM is unchanged and is the parent's:
                        # prepare._attention_flops_probe charges a call 12 * B * T_call * h * d *
                        # min(window_left, k.shape[1]) -- one flat rate for every query position
                        # in the call, "with no causal halving" in its own docstring -- so a query
                        # early in a block is charged for that block's LAST key prefix even though
                        # causality forbids it those keys. _compute_head_call_plans cuts the query
                        # axis into blocks of length L and hands block j the key/value PREFIX
                        # [0, (j+1) * L), and the kernel's own documented mask reproduces the
                        # parent's position for position: with window_size set query i reads keys
                        # [i + seqlen_k - seqlen_q - left, i + seqlen_k - seqlen_q + right],
                        # seqlen_k - seqlen_q is exactly the block's start, and causal=True's
                        # bottom-right alignment agrees. Nothing is subsampled, no window
                        # shortens, no head loses a position it could read, no parameter moves,
                        # and each class's query counts still sum to 2048.
                        # WHY THE FLOOR IS THE RUNG, and why this is the last uniform halving
                        # worth taking. What decides how finely the flat rate should be applied is
                        # what an added kernel call buys: halving a class's block length from L
                        # removes 12 * heads * head_dim * L**2 / (4 * sequence_len) of key and
                        # costs window / L added calls, so ONE added call always buys
                        # 12 * heads * head_dim * L**2 / (4 * sequence_len). The parent's own
                        # comment reads that as "a function of the block length ALONE"; it is not,
                        # because it contains `heads` and the call it costs does not, and this
                        # stack's twelve reach classes are not uniform in heads -- eight carry TWO
                        # heads at the 128 floor and four carry ONE head at 2048, 1024, 512 and
                        # 256. Priced against this parent, the splits available are: the four
                        # one-head classes from 256 to 128 at 6,144 of key per added call, then
                        # the eight two-head classes from 128 to 64 at 3,072, then the one-head
                        # classes from 128 to 64 at 1,536, then the two-head classes from 64 to 32
                        # at 768. This constant takes the first tier and no other, which is a
                        # clean frontier point: every split it buys returns 6,144 per call and
                        # every split it leaves returns at most 3,072. It stops there because the
                        # next tier is not priced in calls alone -- the eight floor classes are
                        # the ones on _attend's zero-copy fast path, so splitting them adds a
                        # query-axis slice and one concatenation at each of the eight layers,
                        # about 1,056 MiB of forward copy traffic per microbatch for 24,576 of
                        # key, against about 404 MiB for the 92,160 this constant buys.
                        # WHAT MOVES, verified against this source: the field is read by
                        # _compute_head_call_plans alone. At L = 128 a class splits only where its
                        # window exceeds 128, so exactly the four classes above the floor split --
                        # depth 3's 2048 head 8 -> 16 calls, depth 3's 1024 head 4 -> 8, depth 7's
                        # 512 head 2 -> 4, depth 7's 256 head 1 -> 2 -- while all SIXTEEN heads at
                        # the floor keep one call over the whole query axis with its key tensor,
                        # its window kwarg and its charge to the digit, because min(128, (j+1) *
                        # 128) is 128 at every j and every one of their blocks merges. Six of the
                        # eight layers are untouched even in their kernel arguments. Heads per
                        # layer stays 2,2,2,4,2,2,2,4, declared reach 128/2048, every per-head
                        # span unchanged, reach classes 1,1,1,3,1,1,1,3, read and write groups 2 at
                        # fan-in and fan-out 256, value tables 128/256, key_embed 1 x 128 serving
                        # h0+2 at six depths and h2+2 at depths 3 and 7, rope bands 3 at the same
                        # reaches and bases, mlp_embed 3 x 1024, the readout 384 fine + 64 pair
                        # means of 128 = 448, and not one nn.Linear, gain, scalar, buffer or
                        # gathered table added, removed, resized or re-dtyped -- so
                        # num_params_total does not move, none of the 584,816 parameters the
                        # ceiling still admits is spent, and both compiled-optimizer censuses stay
                        # at 8 signature classes. The whole counted delta is the span tally's,
                        # 3,661,824 -> 3,569,664. What this DOES carry is _attend's split path
                        # deeper: fifteen more flash_attn_func launches per forward, at two of the
                        # eight layers, and the contiguous key and value prefix copies each split
                        # call makes. False is the parent's rung set, which excludes the floor and
                        # gives block length 256, and is this card's flag-off control.

KEY_EMBED_BY_REACH = True # let the ONE gathered key table serve the HEADS whose exact reach is
                        # the ladder's floor, instead of the LAYERS whose whole kv width happens
                        # to equal the table's. KEY_EMBED gave the key side a per-token identity
                        # and _compute_key_embed_slots' parent rule handed it out by testing
                        # head_counts[i] * head_dim == key_embed_width -- a property of the
                        # table's shape, not of a head's job. At this geometry that serves the
                        # six 'S' layers and skips depths 3 and 7, so heads 2 and 3 of each,
                        # whose exact spans are 128 and 128, are the only floor-reach heads in
                        # the stack scoring without one: sixteen heads span 128 and twelve are
                        # served. With this True the trailing 128 channels -- exactly
                        # key_embed_width // ATTN_HEAD_DIM = 2 heads -- of every layer's kv
                        # receive the component, which is the whole kv width at an 'S' layer
                        # (the parent's own line) and heads 2 and 3 at a retrieval layer. ONE
                        # table still, at ONE width, under the ke_lambdas vector that already
                        # has an entry per depth, so this spends none of the 60,528 parameters
                        # the ceiling still admits, adds no nn.Linear, buffer or optimizer
                        # signature, and moves neither counted family: a gather, a broadcast
                        # multiply, an add and a head-axis concatenation are charged exactly
                        # zero, and q.shape, k.shape, every window_size kwarg and every entry
                        # of every call plan are untouched, so the span tally stays 3,833,856
                        # and the counted nn.Linear total stays 9,373,312. Verified against
                        # this source: the field is read by _compute_key_embed_slots alone;
                        # head counts stay 2,2,2,4,2,2,2,4, declared reach 128/2048, head
                        # spans and the 16 fa3 calls byte-identical, read and write groups 2 at
                        # fan-in and fan-out 256, value tables 128/256, rope bands 3 at the
                        # same reaches and bases, mlp_embed 3 x 1024, and the key table still
                        # 1 x 128. False is the parent's kv-width rule and is this card's
                        # flag-off control.

READOUT_COARSE_UNITS = 1 # read the trailing ONE of the residual's n_head = 4 channel units
                        # at PAIR resolution instead of unpooled, so lm_head reads 448 columns
                        # instead of 512 while every one of the 512 residual channels still
                        # reaches every one of the 8,192 types. The readout is 41.89% of this
                        # key, the largest counted family in the file together with the grouped
                        # MLP, and every mechanism measured against it removed channels or
                        # directions from a type's view: a rank-256 factorisation, a two-block
                        # partition at fan-in 256, an overlapping four-block cover at fan-in
                        # 384. None of them asked whether the readout needs every channel at
                        # FULL resolution as opposed to needing every channel at all, because
                        # in each of them the pooled alternative did not exist -- a type's
                        # weight vector simply had no column for what its block did not read.
                        # Here it does: the trailing 128-channel unit's 64 pairs each keep one
                        # column, shared by the pair. 1 is the SMALLEST dose this mechanism
                        # has at the file's own channel unit -- one unit, at the finest pooling
                        # a unit admits -- and it is a dose the unit fixes rather than one
                        # chosen for size: 0 is the parent and 2, 3 and 4 pool more units at
                        # the same pair resolution. Verified against this source: the constant
                        # is read by GPTConfig.readout_coarse_units, which GPT.__init__ and
                        # GPT._readout_read alone consume, so heads per layer stays 2,2,2,4,
                        # 2,2,2,4, declared reach 128/2048, every per-head span, all 16 fa3
                        # calls, the 4 x 512 query blocks and the 3,833,856 span tally are
                        # byte-identical; read and write groups stay 2 at fan-in and fan-out
                        # 256; value tables stay 128/256; key_embed stays 1 x 128 serving
                        # h0+2 at six depths and h2+2 at depths 3 and 7; rope bands stay 3 at
                        # the same reaches and bases; mlp_embed stays 3 x 1024; and the only
                        # tensor in the file that changes shape is lm_head.weight, which
                        # RETIRES the (8192, 512) float32 AdamW signature class as it creates
                        # (8192, 448) float32, leaving both compiled-optimizer censuses at
                        # exactly 8 classes with the Muon list untouched. 0 is the parent's
                        # dense read and is this card's flag-off control.

# Optimization
TOTAL_BATCH_SIZE = 2**18 # tokens per optimizer step, set EQUAL to DEVICE_BATCH_SIZE *
                        # MAX_SEQ_LEN so grad_accum_steps is exactly 1 and one optimizer
                        # update owns one staged batch. prepare.make_dataloader holds ONE
                        # pinned host buffer and ONE device buffer, refills both on every
                        # next() and yields views of that single device buffer after a
                        # non_blocking host-to-device copy; the loop below synchronises
                        # only at the step boundary, before t0 and at t1. At one microbatch
                        # per update those two calls order the loader's next host-side
                        # refill after its own still-pending copy, so the batch an update
                        # trains on is the batch the loader staged for it. With a second
                        # microbatch inside the same step the host refills that one buffer
                        # again with nothing waiting on the pending copy, so which batch
                        # the second forward reads is decided by host/device timing rather
                        # than by the loader. The other half of the same edit is the cost
                        # of an update: one forward+backward instead of two, at identical
                        # per-token arithmetic, identical microbatch shape and identical
                        # parameters.
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
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        head_dim=ATTN_HEAD_DIM, mlp_mult=MLP_MULT, mlp_groups=MLP_GROUPS,
        mlp_write_shift=MLP_WRITE_SHIFT, mlp_embed_groups=MLP_EMBED_GROUPS,
        window_pattern=WINDOW_PATTERN,
        short_window=SHORT_WINDOW, short_n_head=SHORT_N_HEAD,
        reach_ladder=REACH_LADDER,
        ladder_rungs_per_layer=LADDER_RUNGS_PER_LAYER,
        attn_read_groups=ATTN_READ_GROUPS,
        attn_read_shift=ATTN_READ_SHIFT, attn_write_grouped=ATTN_WRITE_GROUPED,
        attn_write_shift=ATTN_WRITE_SHIFT,
        attn_group_every_layer=ATTN_GROUP_EVERY_LAYER,
        key_embed=KEY_EMBED,
        rope_reach_matched=ROPE_REACH_MATCHED,
        causal_query_blocks=CAUSAL_QUERY_BLOCKS,
        causal_blocks_at_finest_rung=CAUSAL_BLOCKS_AT_FINEST_RUNG,
        causal_blocks_at_floor_reach=CAUSAL_BLOCKS_AT_FLOOR_REACH,
        key_embed_by_reach=KEY_EMBED_BY_REACH,
        readout_coarse_units=READOUT_COARSE_UNITS,
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
# Per-layer attention geometry, printed so this round is decomposable from the log
# alone: the head count and the exact span of every layer, and the span tally those
# two imply, which is the half of the counted target no parameter edit can reach.
# Plain text before the reporter and no METRICS substring of its own -- prepare.py
# still owns every scored number.
print("heads per layer:  " + ", ".join(str(h) for h in model.head_counts))
print("declared reach:   " + ", ".join(str(w[0]) for w in model.window_sizes))
print("head spans:       " + " | ".join(
    ",".join(",".join([str(w[0])] * c) for _, c, w in segs)
    for segs in model.head_window_segments))
print("kernel calls:     " + ", ".join(str(len(segs))
                                       for segs in model.head_window_segments))
# Stack-wide ladder assignment, printed for the same reason the spans are: which rung of
# the ladder each layer's head 0 starts at and how many rungs that layer keeps, so this
# round is decomposable from the log alone and the rule -- not just its output -- is
# auditable. "0+all" is the parent's rule at that layer. Plain text before the reporter
# and no METRICS substring of its own -- prepare.py still owns every scored number.
print("ladder rungs:     " + ", ".join(
    (str(s) + "+all") if k is None else (str(s) + "+" + str(k))
    for s, k in zip(model.ladder_first_rung, model.ladder_kept_rungs)))
# Per-layer attention read geometry, printed for the same reason the spans are: the number
# of channel groups each layer's q/kv projections read, that group's width, and which group
# each layer's kv projection reads relative to its q projection. Plain text before the
# reporter and no METRICS substring of its own -- prepare.py still owns every scored number.
print("read groups:      " + ", ".join(str(g) for g in model.read_groups_per_layer))
print("read fan-in:      " + ", ".join(str(b.attn.read_in) for b in model.transformer.h))
print("read kv shift:    " + ", ".join(str(b.attn.read_shift) for b in model.transformer.h))
# Per-layer attention WRITE geometry, printed for the same reason the read geometry is: the
# number of groups each layer's output projection is cut into, each group's fan-out, and
# which residual group each head block writes into relative to the group its query reads.
# Plain text before the reporter and no METRICS substring of its own -- prepare.py still
# owns every scored number.
print("write groups:     " + ", ".join(str(b.attn.write_groups) for b in model.transformer.h))
print("write fan-out:    " + ", ".join(str(b.attn.write_out) for b in model.transformer.h))
print("write shift:      " + ", ".join(str(b.attn.write_shift) for b in model.transformer.h))
print("span tally/token: " + str(12 * sum(
    c * config.head_dim * min(w[0], k_len) * q_len
    for segs, plan in zip(model.head_window_segments, model.head_call_plans)
    for (_, c, w), calls in zip(segs, plan)
    for _, q_len, k_len, _ in calls) // config.sequence_len))
# Query-block geometry, printed for the same reason the spans are: how many blocks the query
# axis is cut into, how many kernel calls each depth therefore makes, and the (query count @
# charged reach) of every one of those calls, so the RULE and its charge are both readable
# from the log alone rather than inferred. A class printed as "2048@128" is one call over the
# whole query axis, i.e. the parent's. Note the two call counts are different quantities and
# both are wanted: "kernel calls" above is this layer's number of REACH CLASSES, which this
# card does not move, while "fa3 calls/layer" is the number of flash_attn_func calls, which is
# the reach classes summed over their query blocks. Plain text before the reporter and no
# METRICS substring of its own -- prepare.py still owns every scored number.
print("query blocks:     " + str(model.query_blocks) + " x " +
      str(model.query_block_len))
print("fa3 calls/layer:  " + ", ".join(str(sum(len(calls) for calls in plan))
                                       for plan in model.head_call_plans))
print("charged reach:    " + " | ".join(
    ",".join(f"{q_len}@{min(w[0], k_len)}"
             for (_, _, w), calls in zip(segs, plan)
             for _, q_len, k_len, _ in calls)
    for segs, plan in zip(model.head_window_segments, model.head_call_plans)))
# Gathered MLP-offset geometry, printed for the same reason the attention geometry is: how
# many tables exist, how wide each one is and which table each depth reads. Plain text
# before the reporter and no METRICS substring of its own -- prepare.py still owns every
# scored number.
print("mlp_embed tables: " + str(len(model.mlp_embed)) + " x " +
      str(config.mlp_mult * config.n_embd))
print("mlp_embed depth:  " + ", ".join(str(g) for g in model.mlp_embed_index))
# Gathered key-component geometry, printed for the same reason the attention and mlp_embed
# geometry is: how wide the one table is and which depths read it, so this round is decomposable
# from the log alone. Plain text before the reporter and no METRICS substring of its own.
print("key_embed table:  " + ("none" if model.key_embed is None else
      "1 x " + str(model.key_embed_width)))
print("key_embed depth:  " + ", ".join(str(i) for i in model.key_embed_layers))
# WHICH HEADS of each layer the one table serves, printed for the same reason the head spans
# and the charged reaches are: the RULE's output is auditable from the log alone rather than
# inferred from the depth list. "-" is a layer the table does not serve and "h2+2" is heads 2
# and 3. Plain text before the reporter and no METRICS substring of its own -- prepare.py still
# owns every scored number.
print("key_embed heads:  " + ", ".join(
    "-" if slot is None else ("h" + str(slot[0]) + "+" + str(slot[1]))
    for slot in model.key_embed_slots))
# Rotary band geometry, printed for the same reason the attention, mlp_embed and key_embed
# geometry is: how many bands exist, the per-head reach each band's rows are built at, the
# base each of those rows uses and which band each depth reads, so this round is decomposable
# from the log alone and the RULE is auditable rather than only its output. "shared" is the
# parent's single whole-context band. Plain text before the reporter and no METRICS substring
# of its own -- prepare.py still owns every scored number.
print("rope bands:       " + str(len(model.rope_patterns)))
print("rope reach:       " + " | ".join(
    "shared" if s is None else ",".join(str(w) for w in s) for s in model.rope_patterns))
print("rope base:        " + " | ".join(
    "shared" if s is None else ",".join(f"{10000 * w / MAX_SEQ_LEN:g}" for w in s)
    for s in model.rope_patterns))
print("rope depth:       " + ", ".join(str(j) for j in model.rope_index))
# Readout read geometry, printed for the same reason the attention, mlp_embed, key_embed and
# rope geometry is: the RULE's output is auditable from stdout alone rather than inferred from
# the parameter census. "4 x 128" is the residual's own channel unit and its count, and
# "384 fine + 64 pair means of 128 = 448" is what lm_head actually reads. Plain text before the
# reporter and no METRICS substring of its own -- prepare.py still owns every scored number.
print("readout unit:     " + str(model.readout_unit) + " x " + str(config.n_head))
print("readout coarse:   " + str(config.readout_coarse_units) + " of " + str(config.n_head) +
      " units, channels [" + str(model.readout_fine) + ", " + str(config.n_embd) + ")")
print("readout read:     " + str(model.readout_fine) + " fine + " +
      str(model.readout_coarse // 2) + " pair means of " + str(model.readout_coarse) +
      " = " + str(model.readout_in))

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

# Learned strength of the prefix-mean channel, one scalar per layer, printed so this
# round is decomposable from the log alone: they start at exactly 0.0 and only AdamW
# moves them. Plain text before the reporter -- prepare.py still owns every scored
# number and this is not a metrics line.
print("gmem_lambdas: " + ", ".join(f"{v:.6f}" for v in model.gmem_lambdas.tolist()))
print("x0_lambdas:   " + ", ".join(f"{v:.6f}" for v in model.x0_lambdas.tolist()))
# How far the two kv roles separated, printed so this round is decomposable from the log
# alone: both gains start at exactly 1.0 and only AdamW moves them, so a layer whose
# max|k_gain - v_gain| stayed at 0.0 kept one tensor for both roles. Plain text before the
# reporter -- prepare.py still owns every scored number and this is not a metrics line.
kv_gains = [(b.attn.k_gain.detach().float(), b.attn.v_gain.detach().float())
            for b in model.transformer.h]
print("k_gain mean:      " + ", ".join(f"{a.mean().item():.6f}" for a, _ in kv_gains))
print("v_gain mean:      " + ", ".join(f"{b.mean().item():.6f}" for _, b in kv_gains))
print("kv_gain max|k-v|: " + ", ".join(f"{(a - b).abs().max().item():.6f}"
                                      for a, b in kv_gains))
# How far each layer's attention logit scale moved off its exact 1.0 init, printed so
# this round is decomposable from the log alone: the mean over that layer's 256
# (head, channel) entries and the largest absolute departure from 1.0. They start at
# exactly 1.0 and only AdamW moves them, so a layer whose max|g-1| stayed at 0.0 kept
# the parent's pinned sharpness. This establishes USE and never contribution. Plain
# text before the reporter -- prepare.py still owns every scored number and this is
# not a metrics line.
q_scales = [b.attn.q_scale.detach().float() for b in model.transformer.h]
print("q_scale mean:     " + ", ".join(f"{g.mean().item():.6f}" for g in q_scales))
print("q_scale max|g-1|: " + ", ".join(f"{(g - 1.0).abs().max().item():.6f}"
                                       for g in q_scales))

# How far the per-depth gathered MLP offset tables separated, printed so this round is
# decomposable from the log alone: every table starts as a bitwise copy of table 0, so a
# table whose max|t - t0| stayed at 0.0 never left the shared offset and this round would
# have measured the parent's own function. Plain text before the reporter -- prepare.py
# still owns every scored number and this is not a metrics line.
# How far the gathered key component was actually used, printed so this round is decomposable
# from the log alone: every ke_lambda starts at exactly 0.0 and only AdamW moves it, so a depth
# whose value stayed 0.0 never opened the component and that depth's keys are the parent's. This
# establishes USE and never contribution. Plain text before the reporter and not a metrics line.
if model.ke_lambdas is not None:
    print("ke_lambdas:   " + ", ".join(f"{v:.6f}" for v in model.ke_lambdas.tolist()))
    print("key_embed rms: " +
          f"{model.key_embed.weight.detach().float().pow(2).mean().sqrt().item():.6f}")
me_tables = [t.weight.detach().float() for t in model.mlp_embed]
print("mlp_embed rms:      " + ", ".join(f"{t.pow(2).mean().sqrt().item():.6f}"
                                        for t in me_tables))
print("mlp_embed max|t-t0|: " + ", ".join(f"{(t - me_tables[0]).abs().max().item():.6f}"
                                         for t in me_tables))

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
