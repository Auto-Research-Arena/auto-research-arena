"""
Autoresearch pretraining script. Single-GPU, single-file.
Cherry-picked and simplified from nanochat.
Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import inspect
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

# Pre-witness guard, free and before any GPU work: the decode step hands `num_splits` to
# flash_attn_with_kvcache, so a build that renamed or dropped that parameter must fail the
# launch here rather than mid-request. A signature that takes **kwargs accepts the keyword,
# so that counts as readable-and-accepting; an unreadable signature is not evidence either
# way and is left to the pre-witness self-test below.
try:
    _kvcache_sig = inspect.signature(fa3.flash_attn_with_kvcache)
except (TypeError, ValueError):
    _kvcache_sig = None
if _kvcache_sig is not None:
    _kvcache_params = _kvcache_sig.parameters
    assert ("num_splits" in _kvcache_params
            or any(p.kind is inspect.Parameter.VAR_KEYWORD
                   for p in _kvcache_params.values())), (
        f"flash_attn_with_kvcache does not accept num_splits: {_kvcache_sig}")

# 0 asks the kernel to choose its split count for the reduction over the cached key range
# from the shapes and the device; 1 was pinned below only because the op's fake kernel
# refuses to trace at 0, a tracing requirement the decode path does not have (it runs eager
# or replays a manually captured CUDA graph, neither of which needs a fake kernel). The
# reduction order therefore changes, and prepare.py's decode-agreement check must still pass.
DECODE_NUM_SPLITS = 0

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


# Blocks that keep their residual MLP and lose their attention sublayer entirely. Attention is
# not required in every block: the model still mixes across positions in the blocks that keep
# it. The tuple is left exactly as the parent has it -- (2, 4) -- while DEPTH falls to 4, where
# only block 2 exists to be selected, so the attending set is 0, 1 and 3: three of four blocks,
# the same three the parent attended in. The literal is NOT tidied to (2,): it is the parent's,
# and what it means at four blocks is what this comment records.
# At depth 4 has_ve selects the ODD blocks 1 and 3, the mirror image of its selection at depth 5,
# and BOTH of those blocks attend. So both value-embedding tables are gathered, both of those
# blocks carry a ve_gate -- now nn.Linear(32, 8), because the gate's output width is n_kv_head and
# HEAD_DIM 64 makes that 8 -- where only block 0 did, block 0 now attends without a value-embedding
# stream, and for the first time on this lineage no table is stranded. That flip is a consequence
# of the depth; it is recorded as part of the bundle, not compensated for, and it changes which of
# the two region-one code objects serves which block -- see their comment above.
# _compute_window_sizes gives 1024, 1024, 1024, 2048 at four blocks, where block 3 is both the
# pattern's own L and the forced last block, so the attending spans are 1024, 1024 and 2048:
# unchanged from the parent.
# This tuple is the single source of truth: Block.__init__, GPT.forward,
# init_decode_state and _decode_body all read it through has_attn, so the training forward and
# the decode body can never disagree about which blocks attend.
ATTENTION_FREE_BLOCKS = (2, 4)

# The mirror image of the tuple above: blocks that keep their attention sublayer and lose their
# residual MLP entirely. Every block in the parent owns an MLP unconditionally -- Block.__init__
# constructs one with no condition, where the attention sublayer is already conditional -- and the
# MLP is the body's largest per-block weight class: 512 * 2048 + 2048 * 512 = 2,097,152 parameters
# against 4 * 512 * 512 = 1,048,576 for an attention sublayer. Block 1 is the block that loses its
# MLP: it keeps its four projections, its rotary and qk norms, its value-embedding stream -- at
# depth 4 has_ve selects the odd blocks, so block 1 has one -- its cache slot, its flash call and
# its output projection, and it is `x + attn(norm(x))` on the mixed residual and nothing else.
# This tuple is the single source of truth for which blocks own an MLP, read only through has_mlp,
# by Block.__init__, Block.forward, GPT.init_weights, build_decode_weights, drop_decode_weights and
# _decode_body -- exactly as ATTENTION_FREE_BLOCKS is read only through has_attn -- so the training
# forward and the decode body can never disagree. No block index is written at any other site.
MLP_FREE_BLOCKS = (1,)


def has_attn(layer_idx):
    """Returns True if the block holds an attention sublayer."""
    return layer_idx not in ATTENTION_FREE_BLOCKS


def has_mlp(layer_idx):
    """Returns True if the block holds a residual MLP."""
    return layer_idx not in MLP_FREE_BLOCKS


# A block that lost both sublayers would be `a * x + b * x0` and nothing else -- a no-op layer with
# two scalars -- so the two tuples must not overlap. Checked at import, before any GPU work.
assert not (set(ATTENTION_FREE_BLOCKS) & set(MLP_FREE_BLOCKS)), (
    f"a block cannot lose both sublayers: {sorted(set(ATTENTION_FREE_BLOCKS) & set(MLP_FREE_BLOCKS))}")


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


# ---------------------------------------------------------------------------
# Decode-only compiled regions, partitioned at the attention call and autotuned.
#
# The training forward is compiled at the bottom of this file, so its arithmetic is lowered
# into fused kernels. `decode_step` is reached through the OptimizedModule's attribute
# passthrough, so whatever the decode body leaves outside a region runs as individual eager
# aten ops. The parent's seven regions held only pointwise and normalisation work, so every
# one of the step's matmul call sites and its per-block residual add dispatched eagerly into
# cuBLAS between them -- and an autotune mode over regions that contain no matmul has nothing
# to template.
#
# Each region below spans a block's GEMVs together with the pointwise and normalisation work
# that feeds and consumes them, and is lowered at `mode="max-autotune-no-cudagraphs"`, so
# inductor picks the lowering for the width-1 matmuls -- Triton templates it benchmarks
# itself, with their pointwise consumers as candidate epilogues -- instead of them being
# extern calls between pointwise islands. The `-no-cudagraphs` variant is required rather
# than cosmetic: inductor's own cudagraph wrapping must not be introduced underneath the
# manual capture in `_GraphedDecodeStep`, which owns the graph for this path.
#
# The partition boundary is the attention call. The flash kvcache custom op stays OUTSIDE
# every region, which is what keeps `DECODE_NUM_SPLITS = 0` available at all: the op's fake
# kernel refuses to trace at 0, so a region containing that call could not be compiled. A
# block that attends runs region one, then the eager flash call, then region two.
#
# `norm` and `apply_rotary_emb` themselves stay undecorated, and every module's `forward`
# keeps calling them, so training and val_bpb are unchanged. No region owns a parameter, a
# buffer or a weight copy: each takes the model's own tensors as arguments.
# ---------------------------------------------------------------------------

def _attn_in(x, x0, a, b, w_q, w_k, w_v, w_gate, idx, w_ve, cos_table, sin_table, pos,
             n_head, n_kv_head, head_dim):
    """Everything an attending block does before its attention call: THE ROTARY POSITION LOOKUP,
    the residual mix, the pre-attention norm, the three input projections, THE VALUE-EMBEDDING
    GATHER, the value-embedding gate and the rotary rotation with the qk norms. The mixed residual
    is returned because it is needed again after attention.

    The rotary rows arrive as the two whole tables `cos_table` and `sin_table` plus a position
    tensor `pos`, and the two `index_select` calls are performed HERE, in the region that
    multiplies by their result. In the parent they were a compiled region of their own
    (`_decode_pos_cos_sin`), so each step wrote two rows to HBM and handed them across a region
    boundary; performed here each is an indirect load the rotary multiply two statements below can
    fold into its own kernel. The price is named rather than hidden: every attending block does the
    lookup where one region did it once per step. `pos` is `state["seq"]` on the step path and
    `self.pos_ids[:Tn]` on the prefill path, which are the same rows the parent's `seq` index and
    `self.cos[:, :Tn]` slice named, so the values the rotary sees are unchanged on both paths. The
    int64 cast is the same cast `_decode_pos_cos_sin` did, because index_select requires an int64
    index and both position tensors are int32.

    The table arrives as `w_ve`, an argument like every other weight, and the row index as `idx`,
    the FLATTENED view of the tensor `_decode_body` received. `F.embedding(idx, w_ve)` is the lookup
    `nn.Embedding.__call__` performs, on the same tensor, so the decode path still reads nothing
    trained or optimized independently of the evaluated model. Performed here the gather is an
    indirect load the gate-and-add three lines below can fold into its own kernel, instead of a
    launch of its own whose (1, 1, kv_dim) result crosses the region boundary through HBM.

    Undecorated, and inlined into whichever of the two regions below calls it, exactly as
    `norm` and `apply_rotary_emb` are inlined into every region. It exists so the arithmetic
    is written once while the two regions stay two code objects -- see their comment.
    """
    B, Tn = x.shape[0], x.shape[1]
    pos_i = pos.to(torch.int64)
    cos = cos_table.index_select(1, pos_i)
    sin = sin_table.index_select(1, pos_i)
    xm = a * x + b * x0
    h = norm(xm)
    q = F.linear(h, w_q).view(B, Tn, n_head, head_dim)
    k = F.linear(h, w_k).view(B, Tn, n_kv_head, head_dim)
    v = F.linear(h, w_v).view(B, Tn, n_kv_head, head_dim)
    if w_ve is not None:
        ve = F.embedding(idx, w_ve)
        # Value residual (ResFormer): the gate reads the first w_gate.shape[-1] channels of h.
        gate = 2 * torch.sigmoid(F.linear(h[..., :w_gate.shape[-1]], w_gate))
        v = v + gate.unsqueeze(-1) * ve.view(B, Tn, n_kv_head, head_dim)
    # .contiguous() because both tensors are handed to the flash-attention op.
    q = norm(apply_rotary_emb(q, cos, sin)).contiguous()
    k = norm(apply_rotary_emb(k, cos, sin)).contiguous()
    return xm, q, k, v


# Region one is TWO functions, one per value-embedding case, because dynamo's cache is per code
# object and this region needs more specialisations than one code object may hold. Written as
# one function it takes three: `x is x0` on block 0, where `_decode_body` passes the same tensor
# twice, plus value-embedding-present and value-embedding-absent. The decode path is driven at
# three widths -- 1536 and 1 by the request probes, 2 by the pre-witness self-test -- so one
# function would need 9 and refuse the 9th under fullgraph=True. At depth 4 has_ve selects the
# ODD blocks, so `_decode_attn_in` serves the attending block 0 ALONE -- which is where
# `_decode_body` passes x and x0 as the same tensor -- and takes one specialisation per width,
# while `_decode_attn_in_ve` serves the attending blocks 1 and 3, neither of which aliases x0, and
# also takes one. At three driven widths that is 3 and 3 artifacts against a recompile limit of 8.
#
# What the two rotary arguments and the row index cost the guarded signature, since dynamo guards
# STRIDES as well as shapes. `cos_table` and `sin_table` are the model's own fixed
# (1, rotary_seq_len, 1, head_dim // 2) buffers, one layout at every extent. `pos` is `state["seq"]`,
# shape (1,) stride (1,), or `self.pos_ids[:Tn]`, also stride (1,), at every cache extent. And `idx`
# arrives FLATTENED: `_decode_body` computes `idx.reshape(-1)` once, so the region sees a
# one-dimensional stride-(1,) tensor rather than a `tokens[:, t:t + 1]` slice whose extent-1 leading
# stride IS the cache extent -- which is what made the parent's `_decode_attn_in_ve` hold one
# artifact per width-1 extent, measured at 8 of 8. Flattened it is the same elements in the same
# row-major order, so `F.embedding` returns the same rows and the `.view(B, Tn, n_kv_head, head_dim)`
# on the add below still holds. The counts are therefore one per driven width: 3 and 3 against a
# recompile limit of 8, and no cache tensor, cache-extent-derived stride or window value is in
# either signature, so the five width-1 cache extents prepare.py drives (2048 and 513 from
# measure_kv_cache_bytes, 2049 and 514 from measure_decode_request, 8 from the pre-witness
# self-test) reuse one width-1 artifact instead of multiplying it.

@torch.compile(dynamic=False, fullgraph=True, mode="max-autotune-no-cudagraphs")
def _decode_attn_in(x, x0, a, b, w_q, w_k, w_v, cos_table, sin_table, pos,
                    n_head, n_kv_head, head_dim):
    return _attn_in(x, x0, a, b, w_q, w_k, w_v, None, None, None, cos_table, sin_table, pos,
                    n_head, n_kv_head, head_dim)


@torch.compile(dynamic=False, fullgraph=True, mode="max-autotune-no-cudagraphs")
def _decode_attn_in_ve(x, x0, a, b, w_q, w_k, w_v, w_gate, idx, w_ve, cos_table, sin_table, pos,
                       n_head, n_kv_head, head_dim):
    return _attn_in(x, x0, a, b, w_q, w_k, w_v, w_gate, idx, w_ve, cos_table, sin_table, pos,
                    n_head, n_kv_head, head_dim)


def _attn_out(x, y, w_proj, w_fc, w_fc2):
    """Everything an attending block does after its attention call: the output projection and
    its residual add, the pre-MLP norm, and the MLP with its own residual add.

    Undecorated, for the same reason `_attn_in` is: the wrapper below is the compiled region,
    and this body is also what the eager fallback calls, so one piece of arithmetic serves both
    and no code object is ever traced at two weight dtypes.
    """
    y = y.contiguous().view(x.shape[0], x.shape[1], -1)
    xm = x + F.linear(y, w_proj)
    h = norm(xm)
    return xm + F.linear(F.relu(F.linear(h, w_fc)).square(), w_fc2)


@torch.compile(dynamic=False, fullgraph=True, mode="max-autotune-no-cudagraphs")
def _decode_attn_out(x, y, w_proj, w_fc, w_fc2):
    return _attn_out(x, y, w_proj, w_fc, w_fc2)


def _attn_out_only(x, y, w_proj):
    """Region two for an MLP-FREE attending block: the attention output projection and its residual
    add, and nothing after it. The same view and the same residual add `_attn_out` uses; what is
    absent is the pre-MLP norm and the MLP.

    Undecorated, for the same reason `_attn_out` is: the wrapper below is the compiled region, and
    this body is also what the eager fallback calls, so one piece of arithmetic serves both and no
    code object is ever traced at two weight dtypes.
    """
    y = y.contiguous().view(x.shape[0], x.shape[1], -1)
    return x + F.linear(y, w_proj)


# A sibling region at the same lowering as `_decode_attn_out`, so its matmul is templated exactly as
# the other regions' are. It serves the MLP-free attending block alone. Its signature is (x, y,
# w_proj): an activation, the attention output of the same step, and a weight copy. No cache tensor
# and no window value is in it, and none of its shapes depends on the cache extent, so it needs one
# specialisation per driven width -- 1536 from the prefill probe, 1 from the ranked step, 2 from the
# pre-witness self-test -- for 3 artifacts against a recompile limit of 8, and that width-1 artifact
# serves all five width-1 cache extents prepare.py drives (2048 and 513 from measure_kv_cache_bytes,
# 2049 and 514 from measure_decode_request, 8 from the self-test).
@torch.compile(dynamic=False, fullgraph=True, mode="max-autotune-no-cudagraphs")
def _decode_attn_out_only(x, y, w_proj):
    return _attn_out_only(x, y, w_proj)


def _mlp_only(x, x0, a, b, w_fc, w_fc2):
    """An attention-free block in one region: the residual mix, the norm and the MLP, which is
    exactly `x + mlp(norm(x))` on the mixed residual."""
    xm = a * x + b * x0
    h = norm(xm)
    return xm + F.linear(F.relu(F.linear(h, w_fc)).square(), w_fc2)


@torch.compile(dynamic=False, fullgraph=True, mode="max-autotune-no-cudagraphs")
def _decode_mlp_only(x, x0, a, b, w_fc, w_fc2):
    return _mlp_only(x, x0, a, b, w_fc, w_fc2)


def _tail(x, w_head, softcap):
    """The tail norm at the last position, the unembedding and the softcap."""
    h = norm(x[:, -1:, :])
    l = F.linear(h, w_head).float()
    return softcap * torch.tanh(l / softcap)


@torch.compile(dynamic=False, fullgraph=True, mode="max-autotune-no-cudagraphs")
def _decode_tail(x, w_head, softcap):
    return _tail(x, w_head, softcap)


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
        # Decode-only derived weights: bf16 value copies of this sublayer's own trained
        # parameters, built by `GPT.build_decode_weights()` after training and read only by
        # `_decode_body`. Plain tensor attributes on purpose -- not nn.Parameter and not
        # registered buffers -- so `count_params`, `state_dict` and `setup_optimizer` see
        # nothing new.
        self._dw_q = None
        self._dw_k = None
        self._dw_v = None
        self._dw_proj = None
        self._dw_gate = None

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
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        # Decode-only derived weights; see CausalSelfAttention.__init__.
        self._dw_fc = None
        self._dw_proj = None

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx) if has_attn(layer_idx) else None
        self.mlp = MLP(config) if has_mlp(layer_idx) else None

    def forward(self, x, ve, cos_sin, window_size):
        if self.attn is not None:
            x = x + self.attn(norm(x), ve, cos_sin, window_size)
        if self.mlp is not None:
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
        # Decode-only derived weight for the unembedding; see CausalSelfAttention.__init__. It
        # lives on the lm_head submodule, not on the GPT object, like every other copy.
        self.lm_head._dw = None
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
        # The prefill position index, as a non-persistent buffer beside the two tables it indexes.
        # Region one now performs the rotary lookup itself, so it needs a POSITION TENSOR on both
        # paths: on the step path that is `state["seq"]`, and on the prefill path it is
        # `self.pos_ids[:Tn]`, which names the same consecutive rows `self.cos[:, :Tn]` named. It is
        # a buffer and not an nn.Parameter, so count_params, estimate_flops_analytic and
        # GPT.estimate_flops do not see it and setup_optimizer's census assertion, which counts
        # parameters, still balances; non-persistent, like cos and sin, so state_dict is unchanged.
        pos_ids = torch.arange(self.rotary_seq_len, dtype=torch.int32)
        self.register_buffer("pos_ids", pos_ids, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # Transformer blocks
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            if block.attn is not None:
                torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
                torch.nn.init.zeros_(block.attn.c_proj.weight)
            if block.mlp is not None:
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
            if block.attn is not None and block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Rebuilt here for the same reason cos and sin are: the model is constructed under
        # torch.device("meta") and `to_empty` leaves every buffer uninitialised, so the arange from
        # __init__ holds garbage by the time this runs. Left uninitialised it would be out-of-range
        # or wrong indices into the rotary tables on the prefill path.
        self.pos_ids = torch.arange(self.rotary_seq_len, dtype=torch.int32,
                                    device=self.transformer.wte.weight.device)
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
            ve = (self.value_embeds[str(i)](idx)
                  if (str(i) in self.value_embeds and block.attn is not None) else None)
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
        cfg = self.config
        head_dim = cfg.n_embd // cfg.n_head
        dev = self.transformer.wte.weight.device
        kw = {"dtype": torch.bfloat16, "device": dev}
        return {
            "max_len": max_len,
            "graph_enabled": bool(graph),
            "seq": torch.zeros(batch, dtype=torch.int32, device=dev),
            # One entry per block so `state["kc"][i]` still indexes by block, but only the
            # blocks that attend own a tensor; an attention-free block's slot is None.
            "kc": [torch.zeros(batch, max_len, cfg.n_kv_head, head_dim, **kw)
                   if has_attn(i) else None for i in range(cfg.n_layer)],
            "vc": [torch.zeros(batch, max_len, cfg.n_kv_head, head_dim, **kw)
                   if has_attn(i) else None for i in range(cfg.n_layer)],
        }

    def reset_decode_state(self, state):
        """Return `state` to position 0 so the same buffers serve another request.

        A fresh state per request would invalidate an address-bound CUDA graph. Stale bytes
        above the position are unreachable -- attention reads only `cache_seqlens` keys -- so
        zeroing them would be timed work serving never does.
        """
        state["seq"].zero_()
        return state

    @torch.no_grad()
    def build_decode_weights(self):
        """Derive a bf16 copy of every weight the width-1 step reads, for the decode path only.

        Each copy is a VALUE COPY of this model's own trained parameter, taken after the last
        optimizer step, so `decode_step` uses nothing trained or optimized independently of the
        model that is evaluated. Nothing in the training path reads them: `GPT.forward`,
        `CausalSelfAttention.forward` and `MLP.forward` read the parameters, and these
        attributes are None until this method runs.

        The point is bandwidth. The compiled regions receive the weight tensors as graph inputs,
        so with fp32 parameters the conversion to bf16 sits inside each region and therefore
        inside the CUDA graph `_GraphedDecodeStep` captures, re-executing on every one of the
        513 replays a request makes. A bf16 copy halves the bytes each GEMV streams and takes
        that conversion out of the replayed graph.

        Where a copy is absent the decode body calls the region's UNDECORATED implementation
        with the fp32 parameters, eagerly. That fallback is what the CPU pre-flight tool
        exercises -- it re-randomises parameters after import, so the decode path must read the
        same tensors `forward` reads -- and it is uncompiled on purpose, so no single code
        object is ever traced at two weight dtypes.
        """
        for block in self.transformer.h:
            attn = block.attn
            if attn is not None:
                attn._dw_q = attn.c_q.weight.detach().to(torch.bfloat16).contiguous()
                attn._dw_k = attn.c_k.weight.detach().to(torch.bfloat16).contiguous()
                attn._dw_v = attn.c_v.weight.detach().to(torch.bfloat16).contiguous()
                attn._dw_proj = attn.c_proj.weight.detach().to(torch.bfloat16).contiguous()
                if attn.ve_gate is not None:
                    attn._dw_gate = (attn.ve_gate.weight.detach()
                                     .to(torch.bfloat16).contiguous())
            mlp = block.mlp
            if mlp is not None:
                mlp._dw_fc = mlp.c_fc.weight.detach().to(torch.bfloat16).contiguous()
                mlp._dw_proj = mlp.c_proj.weight.detach().to(torch.bfloat16).contiguous()
        self.lm_head._dw = self.lm_head.weight.detach().to(torch.bfloat16).contiguous()

    @torch.no_grad()
    def drop_decode_weights(self):
        """Forget every derived copy, so no stale copy can ever be read."""
        for block in self.transformer.h:
            attn = block.attn
            if attn is not None:
                attn._dw_q = None
                attn._dw_k = None
                attn._dw_v = None
                attn._dw_proj = None
                attn._dw_gate = None
            if block.mlp is not None:
                block.mlp._dw_fc = None
                block.mlp._dw_proj = None
        self.lm_head._dw = None

    def _decode_body(self, idx, state, prefill):
        """`forward`'s body with the cache spliced in. `prefill` is a Python bool that is
        constant per variant, so the two branches are two graphs, not one per position."""
        B, Tn = idx.size()
        # The POSITION only; the two index_selects that turn it into rotary rows happen inside
        # region one, on the model's own tables. At prefill these are the consecutive positions
        # 0..Tn-1, exactly the rows `self.cos[:, :Tn]` named in the parent.
        pos = self.pos_ids[:Tn] if prefill else state["seq"]
        # Flattened once here, so the region's guarded signature holds a stride-(1,) 1-D tensor
        # instead of a slice whose extent-1 leading stride is the cache extent. Same elements in the
        # same row-major order, so F.embedding returns the same rows.
        idx_flat = idx.reshape(-1)

        x = norm(self.transformer.wte(idx))
        x0 = x
        for i, block in enumerate(self.transformer.h):
            attn = block.attn
            mlp = block.mlp
            if attn is not None:
                # Region one: the residual mix through to the tensors the flash call consumes.
                # The mixed residual comes back out because region two adds to it. The weights
                # are the bf16 copies; when they are absent the undecorated implementation runs
                # eagerly on the fp32 parameters instead.
                # The module lookup alone, which launches nothing: the gather itself happens
                # INSIDE region one, on this table's own weight tensor.
                ve_weight = (self.value_embeds[str(i)].weight
                             if str(i) in self.value_embeds else None)
                if attn._dw_q is None:
                    x, q, k, v = _attn_in(
                        x, x0, self.resid_lambdas[i], self.x0_lambdas[i],
                        attn.c_q.weight, attn.c_k.weight, attn.c_v.weight,
                        None if ve_weight is None else attn.ve_gate.weight,
                        idx_flat, ve_weight, self.cos, self.sin, pos,
                        attn.n_head, attn.n_kv_head, attn.head_dim)
                elif ve_weight is not None:
                    x, q, k, v = _decode_attn_in_ve(
                        x, x0, self.resid_lambdas[i], self.x0_lambdas[i],
                        attn._dw_q, attn._dw_k, attn._dw_v,
                        attn._dw_gate, idx_flat, ve_weight, self.cos, self.sin, pos,
                        attn.n_head, attn.n_kv_head, attn.head_dim)
                else:
                    x, q, k, v = _decode_attn_in(
                        x, x0, self.resid_lambdas[i], self.x0_lambdas[i],
                        attn._dw_q, attn._dw_k, attn._dw_v, self.cos, self.sin, pos,
                        attn.n_head, attn.n_kv_head, attn.head_dim)

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
                    y = fa3.flash_attn_with_kvcache(q, kc, vc, k=k, v=v,
                                                    cache_seqlens=state["seq"], causal=True,
                                                    window_size=self.window_sizes[i],
                                                    num_splits=DECODE_NUM_SPLITS)
                # Region two: the attention output projection and the whole MLP -- or, on an
                # MLP-free block, the output projection and its residual add alone, which is its
                # own region because the two have different arithmetic and different signatures.
                if mlp is None:
                    if attn._dw_proj is None:
                        x = _attn_out_only(x, y, attn.c_proj.weight)
                    else:
                        x = _decode_attn_out_only(x, y, attn._dw_proj)
                elif attn._dw_proj is None:
                    x = _attn_out(x, y, attn.c_proj.weight,
                                  mlp.c_fc.weight, mlp.c_proj.weight)
                else:
                    x = _decode_attn_out(x, y, attn._dw_proj,
                                         mlp._dw_fc, mlp._dw_proj)
            else:
                # Region three: an attention-free block is one region end to end.
                if mlp._dw_fc is None:
                    x = _mlp_only(x, x0, self.resid_lambdas[i], self.x0_lambdas[i],
                                  mlp.c_fc.weight, mlp.c_proj.weight)
                else:
                    x = _decode_mlp_only(x, x0, self.resid_lambdas[i], self.x0_lambdas[i],
                                         mlp._dw_fc, mlp._dw_proj)

        softcap = 15
        if self.lm_head._dw is None:
            return _tail(x, self.lm_head.weight, softcap)
        return _decode_tail(x, self.lm_head._dw, softcap)

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
            return
        try:
            self._capture()
            self.captured = True
        except Exception as exc:            # noqa: BLE001 -- recorded, then paid for in ms
            self.reason = f"{type(exc).__name__}: {exc}"
            self.graph, self.static_logits = None, None
        print(f"[decode] graph captured={self.captured} reason={self.reason!r}")

    def _advance(self):
        logits = self.model._decode_body(self.static_idx, self.state, prefill=False)
        self.state["seq"].add_(1)
        return logits

    def _capture(self):
        seq0 = self.state["seq"].clone()
        # One eager warm-up on the DEFAULT stream first, so every decode helper is compiled
        # outside the capture stream: a dynamo/inductor compile inside the side-stream
        # warm-up or inside capture would abort the graph. The save/restore of seq0 below
        # makes this extra call invisible to what is computed.
        self.state["seq"].copy_(seq0)
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
# ASPECT_RATIO and DEPTH are COUPLED here and must be read together, and the rounding target is
# HEAD_DIM, which moves in this file. build_model_config rounds depth * ASPECT_RATIO up to a
# multiple of HEAD_DIM, so at HEAD_DIM 64 this pair holds model_dim at
# ((4 * 128 + 63) // 64) * 64 = ((512 + 63) // 64) * 64 = 8 * 64 = 512 and sets n_head to
# 512 // 64 = 8: the parent's width, with the residual cut into EIGHT heads instead of four. At 128
# the product 4 * 128 lands on 512 with NO rounding at all, which is why ASPECT_RATIO keeps this
# value -- the same 128 that landed depth 4 on 512 at HEAD_DIM 128 lands it on 512 at HEAD_DIM 64 as
# well -- and 85 would have landed it on 384 and silently narrowed the model, which is a different
# experiment.
ASPECT_RATIO = 128      # model_dim = depth * ASPECT_RATIO, rounded up to HEAD_DIM: 4*128 -> 512
HEAD_DIM = 64           # target head dimension for attention: the 512-channel residual is cut
                        # into 512 // 64 = 8 heads of 64 channels instead of 4 of 128. Every
                        # projection keeps its shape because n_kv_head * head_dim stays 512, the
                        # cached width per attending block stays 512, and the counted attention
                        # term 12 * n_head * head_dim * span is invariant to the split
                        # (12 * 8 * 64 = 12 * 4 * 128). What moves is the shape the width-1 flash
                        # kvcache call is handed, the number of rotary frequency pairs (32 rather
                        # than 64), and the extent the two qk rms_norms reduce over.
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context

# Optimization
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step: ONE optimizer update per microbatch.
                        # tokens_per_fwdbwd is DEVICE_BATCH_SIZE 128 * MAX_SEQ_LEN 2048 =
                        # 262,144, so grad_accum_steps = 2**18 // 262,144 = 1 and the
                        # accumulation loop runs once -- every forward-backward is an update.
                        # The microbatch is unchanged, so no shape, no parameter, no counted
                        # operation and nothing on the decode path moves; the same token stream
                        # inside the same TIME_BUDGET is cut into twice as many updates.
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
DEPTH = 4               # number of transformer layers; coupled to ASPECT_RATIO 128 above, which
                        # is what keeps model_dim at 512 at this depth. n_head is now 8, because
                        # HEAD_DIM is 64: model_dim is unchanged and the head count is 512 // 64
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

# Pre-witness self-test: run the decode path once here, BEFORE the GPU-work witness, so a
# compile failure, a shape error or a non-contiguous tensor kills the launch while it is
# still uncharged. graph=False, so no CUDA graph is captured and no private graph pool is
# created before training. Prints nothing and touches no counter.
with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
    _selftest_state = model.init_decode_state(batch=1, max_len=8, graph=False)
    _selftest_wide = torch.zeros(1, 2, dtype=torch.int64, device=device)
    _selftest_step = torch.zeros(1, 1, dtype=torch.int64, device=device)
    # The bf16 weight copies are what serving will run, so they are what the self-test must
    # exercise: a wrong shape, a dtype the kernel refuses or a compile failure dies here,
    # uncharged. Dropped in the same block, so nothing derived from pre-training weights
    # survives into training and no copy is read before it is built.
    model.build_decode_weights()
    model.decode_step(_selftest_wide, _selftest_state)
    model.decode_step(_selftest_step, _selftest_state)
    model.decode_step(_selftest_step, _selftest_state)
    model.drop_decode_weights()
    del _selftest_state, _selftest_wide, _selftest_step

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

# Derive the bf16 decode weights from the weights this launch actually trained, before the
# reporter is entered: every probe it runs then reads the copies, and the tensors are already
# allocated when measure_kv_cache_bytes takes its "before" reading.
model.build_decode_weights()

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
# Resubmission of g06c4-six-blocks-at-held-width, generation 6.
#
# The batch this candidate was first submitted in never reached a GPU: the compute allocation
# was reclaimed by the scheduler before any child was leased, so no launch sequence was taken,
# nothing was charged and the launch budget is unchanged. The engine had already reserved this
# program's exact file identity, and a reservation is released only for a candidate that has a
# NOT_CHARGED ledger row -- this one has no row at all -- so the original bytes can never be
# purchased.
#
# This file is that candidate's `train.py` with this comment appended and nothing else changed.
# Comment-stripped, it is byte-identical to the original, whose token stream hashes to
# 4edfcbb0d0aecf413636228ab00fc357afbc45ae2346e46cf87b9c1ffb1d11e7. The card, the mechanism, the frozen parent and the
# preregistered predictions are the ones recorded for g06c4-six-blocks-at-held-width; the
# generation is resumed, not replaced.
# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------
# Candidate g07c1-depth-five-at-held-width, generation 7, card g07c1, child of
# g06c4-six-blocks-at-held-width-r2.
#
# The resubmission block above belongs to the PARENT and is kept byte-unchanged because it is
# that program's own record. This file is not that resubmission: it is a child of it, carrying
# card g07c1's edit, so the token-stream hash quoted above is the parent's and not this file's.
# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------
# Candidate g08c2-one-optimizer-update-per-microbatch, generation 8, card g08c2, child of
# g07c1-depth-five-at-held-width.
#
# One constant: TOTAL_BATCH_SIZE 2**19 -> 2**18, so grad_accum_steps falls from 2 to 1 and every
# forward-backward is its own optimizer update. Nothing else in the file is touched. The
# reporter's arguments keep describing the work that actually happened: one step is one completed
# update, tokens_per_step is TOTAL_BATCH_SIZE, total_tokens is the summed consumption, and the
# training clock still skips the first ten updates. The assert above grad_accum_steps still
# holds, because 2**18 % (128 * 2048) == 0.
#
# The diff cannot reach the decode path, the model or the data budget, so the ranked target
# should land at the parent's value up to the launch component; what the card asks for is gate
# margin on the member holding the run's best target, which the depth channel wants next.
#
# The resubmission block above belongs to the PARENT's parent and the block after it to the
# parent; both are kept byte-unchanged because they are those programs' own records, so neither
# token-stream hash quoted above describes this file.
# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------
# Candidate g09c5-depth-four-at-held-width, generation 9, card g09c5, child of
# g08c2-one-optimizer-update-per-microbatch.
#
# Two coupled constants: ASPECT_RATIO 102 -> 128 and DEPTH 5 -> 4, which holds n_embd at 512 and
# n_head at 4 -- ((4 * 128 + 127) // 128) * 128 = 512 and 512 // 128 = 4, with no rounding at all
# -- while the body loses block 4, an attention-free block that owned an MLP and a value-embedding
# table nothing gathered. HEAD_DIM, WINDOW_PATTERN, ATTENTION_FREE_BLOCKS, has_ve, TOTAL_BATCH_SIZE
# 2**18 and DEVICE_BATCH_SIZE 128, every learning rate and schedule, all six compiled regions and
# their lowerings, DECODE_NUM_SPLITS, build_decode_weights, the capture and its print, the
# pre-witness self-test and both frozen call sites are the parent's.
#
# What follows from the file rather than from any new literal, and is recorded in the two comments
# above: ATTENTION_FREE_BLOCKS stays (2, 4) and now selects block 2 alone, so the attending set is
# still 0, 1 and 3 and the attending spans are still 1024, 1024 and 2048; and has_ve flips from the
# even blocks to the ODD ones, so the two value-embedding tables sit on blocks 1 and 3, both of
# which attend, both of them carry a ve_gate, block 0 attends without a value-embedding stream, and
# no table is stranded. That flip also swaps which region-one code object serves which block:
# `_decode_attn_in` serves block 0 alone and `_decode_attn_in_ve` serves blocks 1 and 3. No region
# is added, removed or re-lowered.
#
# The resubmission block above belongs to this lineage's grandparent and the two blocks after it to
# g07c1 and to the parent; all three are kept byte-unchanged because they are those programs' own
# records, so no token-stream hash quoted above describes this file.
# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------
# Candidate g11c2-head-granularity-on-the-leader, generation 11, card g11c2, child of
# g10c4-mlp-free-attending-block-at-depth-four.
#
# One constant: HEAD_DIM 128 -> 64. build_model_config computes model_dim = ceil(4 * 128 / 64) * 64
# = 512 and num_heads = 512 // 64 = 8, so n_embd stays 512 while n_head and n_kv_head become 8 and
# head_dim becomes 64. n_kv_head * head_dim stays 512, so c_q, c_k, c_v and c_proj all keep their
# (512, 512) shapes and the cached width per attending block is unchanged. HEAD_DIM is read only by
# build_model_config; every other consumer -- CausalSelfAttention.__init__, GPT.__init__,
# _precompute_rotary_embeddings, _compute_window_sizes and init_decode_state -- derives head_dim
# from n_embd // n_head, so no other line moves.
#
# The only derivable movement is the two ve_gates widening with n_kv_head: each is nn.Linear(32, 8)
# instead of nn.Linear(32, 4), so 32 * 4 -> 32 * 8 on two blocks, +256 parameters and 6 * 256 =
# +1,536 counted FLOPs per token. The rotary table carries 32 frequency pairs instead of 64, because
# _precompute_rotary_embeddings builds head_dim // 2 of them, and cos/sin keep shape
# (1, rotary_seq_len, 1, 32). The attending set and the window list are untouched, so the spans the
# flash calls are given stay 1024, 1024 and 2048, and the cached tensor bytes are byte-identical
# because n_kv_head * head_dim is held.
#
# The blocks above belong to this lineage's ancestors and are kept byte-unchanged because they are
# those programs' own records, so no token-stream hash quoted in them describes this file.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Resubmission of g11c2-head-granularity-on-the-leader, generation 11.
#
# The batch this candidate was first submitted in never reached a GPU: the compute allocation
# was reclaimed by the scheduler before any child was leased, so no launch sequence was taken,
# nothing was charged and the launch budget is unchanged. The engine had already reserved this
# program's exact file identity, and a reservation is released only for a candidate that has a
# NOT_CHARGED ledger row -- this one has no row at all -- so the original bytes can never be
# purchased.
#
# This file is that candidate's `train.py` with this comment appended and nothing else changed.
# Comment-stripped, it is byte-identical to the original, whose token stream hashes to
# fc98c81d31d57e085eaa62fdabf6253b46129d7637b33cd4d6d5acddc5277e05. The card, the mechanism, the frozen parent and the
# preregistered predictions are the ones recorded for g11c2-head-granularity-on-the-leader; the
# generation is resumed, not replaced.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Candidate g12c3-value-embedding-gather-inside-region-one, generation 12, card g12c3, child of
# g11c2-head-granularity-on-the-leader-r2.
#
# The value-embedding lookup moves from an eager call in `_decode_body` into the compiled region
# that consumes it. In the parent, `self.value_embeds[str(i)](idx)` ran eager, so its (1, 1, 512)
# result was written to HBM by a kernel of its own and read back as a graph input to
# `_decode_attn_in_ve`, whose gate-and-add three statements in is its only consumer. Here
# `_decode_body` performs the MODULE LOOKUP only -- `self.value_embeds[str(i)].weight`, which
# launches nothing -- and hands the table and the row index into the region, where
# `F.embedding(idx, w_ve)` is an indirect load the consumer can fold into its own kernel. has_ve
# selects blocks 1 and 3 at DEPTH 4 and both attend, so two such launches leave the replayed
# width-1 graph.
#
# F.embedding(idx, self.value_embeds[str(i)].weight) is the same lookup, on the same tensor, that
# nn.Embedding.__call__ performs, so decode_step still reads nothing trained or optimized
# independently of the evaluated model, and no parameter, buffer or weight copy becomes owned by a
# region -- the table arrives as an argument like every other weight. `idx` is the tensor
# `_decode_body` itself received and only that one name is in scope, so the region cannot gather a
# different row from the one the step is for.
#
# GPT.forward's own gather (`self.value_embeds[str(i)](idx)` in the block loop) is byte-unchanged
# and stays eager, which is what holds val_bpb and the decode-agreement reference fixed. No
# parameter, no constant and no module changes, so num_params_total is the parent's 26,214,920 and
# flops_per_token_measured is the parent's 106,957,824: the probe runs FlopCounterMode over the
# uncompiled TRAINING module, which this diff does not touch, and a gather issues no matmul
# wherever it is performed.
#
# `_attn_in` takes `idx` and `w_ve` where it took `ve`, and its `if ve is not None` branch becomes
# `if w_ve is not None`. `_decode_attn_in_ve` forwards the two new arguments; `_decode_attn_in`
# keeps its own signature and passes None for w_gate, idx and w_ve, so the block with no stream
# gathers nothing. Both region-one code objects, their decorators and
# mode="max-autotune-no-cudagraphs" are unchanged, and so is their split: `_decode_attn_in` serves
# attending block 0, where x and x0 are the same tensor, at one specialisation per driven width,
# and `_decode_attn_in_ve` serves attending blocks 1 and 3, neither aliased, also at one. The two
# new arguments add no specialisation -- idx's width already tracks the width the region is driven
# at, and the table's shape is fixed at (8192, 512) -- so the count stays 3 and 3 against a
# recompile limit of 8, and no cache tensor and no window value is in either signature, so the
# width-1 artifact serves all five width-1 cache extents prepare.py drives.
#
# CausalSelfAttention.forward, MLP, Block, init_weights, build_decode_weights,
# drop_decode_weights, every constant, all seven regions' lowerings, DECODE_NUM_SPLITS, the
# capture and its print, the pre-witness self-test and both frozen call sites are byte-unchanged.
#
# The blocks above belong to this lineage's ancestors and are kept byte-unchanged because they are
# those programs' own records, so no token-stream hash quoted in them describes this file.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Candidate g13c1-rotary-position-lookup-inside-region-one, generation 13, card g13c1, child of
# g12c3-value-embedding-gather-inside-region-one.
#
# The rotary position lookup moves from a compiled region of its OWN into the region that multiplies
# by its result. In the parent, `_decode_pos_cos_sin(self.cos, self.sin, state["seq"])` cast seq to
# int64 and took two index_select rows out of the (1, 20480, 1, 32) cos and sin tables on every
# width-1 step, wrote both to HBM, and handed them across a region boundary into whichever
# region-one calls followed. Here `_decode_body` computes the POSITION only -- `state["seq"]` on the
# step path, `self.pos_ids[:Tn]` on the prefill path -- and region one receives the two whole tables
# and that position, so `cos_table.index_select(1, pos_i)` is an indirect load the rotary multiply
# two statements below can fold into its own kernel and one compiled region leaves the step. The
# cost is the mechanism's own and is not hidden: three attending blocks each do the lookup where one
# region did it once, so two extra index_select pairs per step are the price of not materialising
# the rows. `_decode_pos_cos_sin` and its decorator are deleted; nothing else calls them.
#
# The values are the parent's on both paths. On the step path the index is `state["seq"]`, the same
# tensor the deleted region indexed with and still the single source of the position, so
# reset_decode_state semantics are untouched and the number of previous calls does not change what
# is computed. On the prefill path `self.pos_ids[:Tn]` is arange(Tn), which names exactly the rows
# `self.cos[:, :Tn]` named. `pos_ids` is a non-persistent buffer beside cos and sin, registered in
# GPT.__init__ and REBUILT in init_weights beside the cos/sin rebuild, because the model is
# constructed under torch.device("meta") and to_empty leaves every buffer uninitialised. It is a
# buffer and not an nn.Parameter, so count_params does not see it and setup_optimizer's census
# assertion, which counts parameters, still balances; it is a constant index, not request state, so
# everything a request needs is still owned by the object init_decode_state returns.
#
# num_params_total is exactly the parent's 26,214,920 = 4,194,304 (wte) + 4,194,304 (lm_head) +
# 8,388,608 (two value-embedding tables) + 9,437,184 (transformer.h) + 512 (two ve_gates) + 8
# (scalars). flops_per_token_measured is exactly the parent's 106,957,824 = 6 * (9,437,184 + 512 +
# 4,194,304) + 12 * 8 * 64 * (1024 + 1024 + 2048): an index_select dispatches no matmul, and the
# probe runs FlopCounterMode over the uncompiled TRAINING module, which this diff does not enter --
# GPT.forward still reads self.cos[:, :T] and self.sin[:, :T] and is byte-identical, as are
# CausalSelfAttention.forward, MLP.forward and Block.forward. nopref_kv_cache_tensor_bytes is
# 3 * 2 * 513 * 512 * 2 + 4 = 3,151,876, untouched. DATA_BUDGET_TOKENS is untouched.
#
# The artifact arithmetic, which is why the row index is flattened. dynamo guards strides as well as
# shapes, and `idx` on this path is always a slice of a row whose length is the probe's own max_len,
# so the leading stride of its extent-1 batch dimension IS the cache extent: that is what put the
# parent's `_decode_attn_in_ve` at the measured 8 of config.recompile_limit 8, one artifact per
# driven (width, stride) pair. `_decode_body` now computes `idx.reshape(-1)` once and passes that,
# so the region sees a 1-D stride-(1,) tensor at every extent -- the same elements in the same
# row-major order, so F.embedding returns the same rows and the existing
# `.view(B, Tn, n_kv_head, head_dim)` on the add still holds. The two new rotary arguments add no
# specialisation either: the tables are fixed buffers and `pos` is stride (1,) on both paths. So
# each region-one code object holds one artifact per driven width -- 1 from the ranked step, 1,536
# from the prefill probe, 2 from the pre-witness self-test -- for 3 each, `_decode_pos_cos_sin` is
# gone, and every other region keeps 3 or fewer: no cache tensor, no cache-extent-derived stride and
# no window value is in any guarded signature.
#
# init_weights, build_decode_weights, drop_decode_weights, setup_optimizer, every constant,
# init_decode_state, reset_decode_state, _GraphedDecodeStep with its capture print, DECODE_NUM_SPLITS,
# the six remaining regions' decorators and lowerings, the pre-witness self-test and both frozen call
# sites are byte-unchanged.
#
# The blocks above belong to this lineage's ancestors and are kept byte-unchanged because they are
# those programs' own records, so no token-stream hash quoted in them describes this file.
# ---------------------------------------------------------------------------
