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
    window_pattern: str = "SSSL"
    kv_group_size: int = 8  # query heads per KV head (GQA); 1 restores per-head KV

    def __post_init__(self):
        # n_kv_head is DERIVED, not passed: the grouping is an invariant of the config, so
        # every construction site -- build_model_config below, and any checker that builds
        # the same dataclass -- gets the same relation without threading an argument.
        assert self.n_head % self.kv_group_size == 0
        self.n_kv_head = self.n_head // self.kv_group_size


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


KV_STREAM_COUNT = 2  # K/V streams among layers 1.., see `kv_producers`


def kv_producers(n_layer):
    """The layers that own a K/V stream. Every other layer attends one of theirs.

    A stream is produced at a value-embedding layer, so that the value embedding mixed into a
    stream is the one belonging to the layer that computes it. It does not follow that every
    value-embedding layer needs a stream of its own. `KV_STREAM_COUNT` says how many streams
    the network has, and they are the LOWEST value-embedding layers, because the lowest ones
    are the streams the decode state recomputes from the token-id ring at zero bytes (see
    `_decode_streams`) while every stream above them is stored and is the whole of what the
    cache pays for. At TWO the network's streams ARE the two the decode state rebuilds, so
    NOTHING is stored: the token-id book is the whole of the retained K/V data, and the window
    of the top group stops charging cache bytes at 80 B per extent unit and starts charging the
    book's 1 + 5/8 B per ring slot instead (see `_compute_window_sizes` and `_id_ring_len`).

    The price is stated here rather than hidden: the layers above the highest producer attend
    its stream, so they compute no v and one group of layers sees one shared set of keys and
    values. Their OWN value embeddings are not orphaned -- a layer that attends another
    layer's stream mixes its own value embedding into its attention output instead of into
    the stream it does not own (see `CausalSelfAttention.forward`), which costs no cache
    bytes because it is a function of the query position's own token alone.
    """
    ve_layers = [i for i in range(1, n_layer) if has_ve(i, n_layer)]
    return ve_layers[:KV_STREAM_COUNT]


def kv_source(layer_idx, n_layer):
    """The layer whose K/V stream layer `layer_idx` attends.

    A layer attends the nearest producer at or below it, so K and V are computed once per
    GROUP. With `has_ve` alternating, `n_layer` even and `KV_STREAM_COUNT` TWO below the
    number of value-embedding layers, the groups are (1,2) and (3,4,5,6,7): the top group runs
    from the highest rebuilt stream to the last layer, so every layer above 2 attends a stream
    the decode state recomputes from the token ids. Layer 0 is
    always its own source: it holds no K/V storage at all, only a ring of token ids (see
    `init_decode_state`).
    """
    if layer_idx == 0:
        return 0
    return max((p for p in kv_producers(n_layer) if p <= layer_idx), default=layer_idx)


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
        # A layer that shares another layer's K/V stream computes no K and no V, so it owns
        # neither projection: the GROUP's keys and values are the producer's, parameters and
        # all. Queries and the output projection stay per layer, so the layers of one group
        # still attend the shared stream differently. A consumer keeps its own `ve_gate`,
        # because its value embedding now enters its attention OUTPUT (see `forward`).
        self.shares_kv = kv_source(layer_idx, config.n_layer) != layer_idx
        self.c_k = None if self.shares_kv else nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = None if self.shares_kv else nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv=None):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = cos_sin
        if kv is None:
            k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
            v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

            # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
            if ve is not None:
                ve = ve.view(B, T, self.n_kv_head, self.head_dim)
                gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
                v = v + gate.unsqueeze(-1) * ve

            q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
            q, k = norm(q), norm(k)
        else:
            # The producer's keys and values, already rotated and normed at their own layer.
            k, v = kv
            q = norm(apply_rotary_emb(q, cos, sin))

        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        if self.shares_kv and ve is not None:
            # A layer that attends another layer's stream computes no v, so the ResFormer
            # mix above is not available to it: its value embedding would reach nothing.
            # Add it to this layer's attention OUTPUT instead, through the same `ve_gate`
            # and the same `c_proj` the mixed-in term passes through. With `y` the sum
            # `sum_j a_j v_j` over the attended keys, a producer's term is
            # `sum_j a_j gate_j ve_j`, the attention-weighted mean of the window's value
            # embeddings; this is the same term evaluated at the QUERY position,
            # `gate_p ve_p`. It depends on the query's own token and on nothing else, so
            # in decode it needs no history and the cache stores nothing for it.
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            y = y + gate.unsqueeze(-1) * ve
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

    def forward(self, x, ve, cos_sin, window_size, kv=None):
        y, kv = self.attn(norm(x), ve, cos_sin, window_size, kv)
        x = x + y
        x = x + self.mlp(norm(x))
        return x, kv


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.kv_sources = [kv_source(i, config.n_layer) for i in range(config.n_layer)]
        # A shared stream must come from a layer that owns a K/V ring (never layer 0, which
        # stores token ids) and that attends the same window, since the consumer reads the
        # producer's ring as its own window extent.
        assert all(src >= 1 and self.window_sizes[src] == self.window_sizes[i]
                   for i, src in enumerate(self.kv_sources) if src != i)
        # Slot in the stacked ring storages, one per DISTINCT stream among layers 1..,
        # in layer order; a consumer indexes its producer's slot.
        producers = [i for i in range(1, config.n_layer) if self.kv_sources[i] == i]
        self.kv_stream_index = [None] + [producers.index(self.kv_sources[i])
                                         for i in range(1, config.n_layer)]
        self.n_kv_streams = len(producers)
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
            if not block.attn.shares_kv:
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
        assert all(c in "SLNBT" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        # "N" is the window of the LOW GROUP -- the lowest rebuilt stream, layer 1, and the one
        # layer that reads it. It is a LITERAL and not `long_window // 8`, because what fixes it
        # is not a fraction of the context but the book's own block boundary: the id ring holds
        # `book_window + narrow_window + top_window + 1` slots (see `_id_ring_len`), and TWO
        # 512 B blocks need that sum <= 625. With the top group held at 384 and layer 0 sold down
        # to 1, this level is what lands the sum on the boundary exactly: 1 + 239 + 384 + 1 = 625.
        # A unit here is read by TWO layers against ONE at layer 0, so it is the second-cheapest
        # extent in the network, and it is the only place the last 17 slots can come from without
        # taking them back off the top group.
        # AND THE SPLIT BETWEEN THIS LEVEL AND THE TOP GROUP CANNOT PAY FOR THE NEXT BLOCK DOWN.
        # Hold `book_window` at 1 and this level plus the top group at 623, so `L` stays 625 and
        # the book stays 1,024 B. The network's layer-extent units are then
        # `1 + 2*narrow + 5*(623 - narrow) = 3116 - 3*narrow`, so against today's 2,399 a
        # reallocation GAINS `717 - 3*narrow` units. The ONE block below needs `L <= 312`, i.e.
        # 313 span units sold -- 1 at layer 0, `narrow` here at two layers each, the remaining
        # `312 - narrow` off the top group at five each -- which COSTS
        # `1 + 2*narrow + 5*(312 - narrow) = 1561 - 3*narrow` layer-extent units. Both move by 3
        # per unit of `narrow`, so cost minus gain is `1561 - 717 = 844` units at EVERY spelling
        # with `narrow <= 312`: every unit of reach moved up to the top group is a unit the 512 B
        # block has to buy back. Only the layer counts enter that, so it holds at any per-group
        # price; and past `narrow = 312` the difference is `3*narrow - 92`, strictly worse. The
        # 512 B block is therefore 844 units away from any spelling of this split, and moving the
        # split is a quality trade only, never a byte one.
        narrow_window = 239
        # "T" is the window of the TOP GROUP -- the highest rebuilt stream and every layer
        # that READS it. It charges NO cache byte. Nothing is stored (see `_decode_streams`), so
        # the only storage any window can size is the token-id `book`, and this level reaches it
        # through `_id_ring_len`'s hop sum as ONE ring slot per extent unit, which the book holds
        # at 1 + 5/8 B. FIVE layers read this one window, so an extent unit per layer costs
        # 1.625 / 5 = 0.325 B here, against the 80 B per extent unit -- 26.67 B per layer over
        # the three layers that read it -- that the SAME context cost while this group's stream
        # was STORED as merged five-bit codes. That ratio is what this level is for, and it is
        # why this level is the one the book cut below does NOT touch: the 384 is held, and the
        # 317 ring slots the book gives up are taken off layer 0 and the low group instead.
        # It is a window and not a tolerance: every key at distance <= 384 is still attended, by
        # `forward` and by `decode_step` alike, out of the one position-ordered reconstruction
        # `_rebuild_kv_from_ids` produces for this stream.
        top_window = 384
        # "B" is layer 0's window, and layer 0 stores no K or V: its left window is
        # `_id_ring_len`'s `left0`, so it is what sizes the token-id ring and therefore the
        # whole `book` storage. The ring holds `left0 + 239 + 384 + 1` slots here (one
        # `_history_len` per rebuilt stream plus one), at one LOW byte plus a FIVE-bit HIGH
        # field each, and `init_decode_state` allocates
        # `1 + (5*ceil(L/8) + L + 3)//4` int32 words, which the allocator rounds up to a
        # multiple of 512 B. Swept over that expression, the shelves are exact: TWO blocks
        # need `5*ceil(L/8) + L <= 1020`, which `L = 625` meets at 5*79 + 625 = 1020 -> 256
        # words -> 1,024 B, while `L = 626` gives 1021 -> 257 words -> 1,028 B, which rounds
        # to 1,536. THREE blocks need <= 1532, met at `L = 942`; ONE block needs <= 508, met
        # at `L = 312`. BOTH ENDS of the two-block shelf are confirmed on device through
        # `prepare.measure_kv_cache_bytes`'s own procedure: `L = 626` reads 1,536 and `L = 624`
        # reads 1,024, so 625 is the top and there is no slack above it to take. The shelf is a
        # BAND and not a point: every `L` in 313 .. 625 reads 1,024 at the headline shape, and
        # every `L >= 313` reads 1,024 at the no-prefill shape as well, because there `L` is
        # `min(513, ...)` and 513 gives 4 + 325 + 513 = 844 -> 1,024. At `L <= 312` BOTH shapes
        # read 512. So the whole composed span this network may keep, at two blocks, is
        # `L - 1 = 624`, and this file spends it 1 + 239 + 384: the top group's 384 is held
        # because that is where the record has measured reach to be worth most, the low
        # group's 239 pays the last 17 slots at two layers each, and layer 0 -- the one layer
        # whose keys and queries are a function of the token id alone, and the only layer
        # with a single reader -- pays the other 300.
        # There is NO slack left: the `seq` word plus the two planes is 4 + 395 + 625 =
        # 1,024 B of payload in 1,024 B of storage, so a one-slot arithmetic slip costs a
        # whole block back. And 625 is NOT a multiple of 8, so the high plane's last
        # five-byte group carries SEVEN padding slots -- which `hi_groups` sizes, both
        # prefill branches fill with -16.0, and `_read_ids` truncates away.
        # It is a window and not a tolerance: every key at distance <= 1 is still attended,
        # by `forward` and by `decode_step`, from this one table. It is kept at 1 rather than
        # at 0 so that every attention call in the file stays on the ordinary local-window
        # path, at a cost of exactly one extent unit.
        book_window = 1
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0),
                          "B": (book_window, 0),
                          "N": (narrow_window, 0),
                          "T": (top_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
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
        kvs = [None] * self.config.n_layer
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            src = self.kv_sources[i]
            x, kvs[i] = block(x, ve, cos_sin, self.window_sizes[i],
                              None if src == i else kvs[src])
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

    def _cache_len(self, layer_idx, max_len):
        """Slots layer `layer_idx` can ever read: its window extent, capped at `max_len`.

        The layer's attention (`forward` -> `Block.forward` -> `CausalSelfAttention.forward`)
        passes `window_size=(left, 0)`, so a query can only see `left + 1` keys. Anything
        older is masked out of every kernel call, so storing it buys nothing.
        """
        left = self.window_sizes[layer_idx][0]
        if left is None or left < 0:
            return max_len
        return min(max_len, left + 1)

    def _history_len(self, layer_idx, max_len):
        """Slots a K/V ring must RETAIN: the layer's `left` most recent PREVIOUS positions.

        A query with `window_size=(left, 0)` sees `left + 1` keys -- its own and the `left`
        before it -- but this step's own k and v are computed at the step and handed to the
        kernel directly, concatenated onto the ring, so the ring retains only the
        predecessors a LATER step reads. A request spanning `max_len` positions has at most
        `max_len - 1` predecessors, which caps it.
        """
        left = self.window_sizes[layer_idx][0]
        if left is None or left < 0:
            return max(max_len - 1, 0)
        return min(max_len - 1, left)

    def _id_ring_len(self, max_len):
        """Slots the token-id ring must hold, capped at `max_len`. DECODE ONLY.

        The ids stand in for layer 0's k and v and for EVERY rebuilt K/V stream, all of which
        `_rebuild_kv_from_ids` recomputes from them. The reach is a sum over HOPS, one hop per
        rebuilt stream, and each hop's reach is the window extent that hop reads through:

        * layer 0 holds no value embedding, so its k and v at `s` are a function of the id at
          `s` and of `s`; its window is `(left0, 0)`, so its block OUTPUT at `s` is a function
          of the ids at `s - left0 .. s`.
        * the LOWEST rebuilt stream's k and v at `s` are formed from that output, so they need
          the ids at `s - left0 .. s`.
        * each FURTHER rebuilt stream sits one hop above the last: the layers between them read
          the lower stream under their own `(left, 0)` window, so the residual entering the
          higher stream at `q` is a function of the lower stream's keys at `q - left .. q`,
          hence of ids `_history_len` further back again.
        * and a query at `p` reads the HIGHEST rebuilt stream's keys at `p - left .. p`, which
          is one more `_history_len` back.

        Summing those hops, the oldest id any step of the request can need is at
        `p - left0 - sum(_history_len(r) for r in rebuilt)`, so the ring holds
        `left0 + sum(...) + 1` positions, inclusive. With one rebuilt stream that is exactly
        the `left0 + hist + 1` the single-hop version held.

        The cap is exact rather than a truncation. A request is at most `max_len` positions
        long, so a ring capped at `max_len` never wraps and holds every position the request
        ever had; and when the sum is the smaller of the two the ring holds every position any
        query can reach. Either way no key is rebuilt from an id the ring no longer has.
        """
        left0 = self.window_sizes[0][0]
        if left0 is None or left0 < 0:
            return max_len
        rebuilt, _, _ = self._decode_streams()
        reach = sum(self._history_len(r, max_len) for r in rebuilt)
        return min(max_len, left0 + reach + 1)

    def _decode_streams(self):
        """The streams recomputed from the id ring, and the store slot of every other. DECODE ONLY.

        `self.kv_sources`, `self.kv_stream_index` and `self.n_kv_streams` are what `forward`
        reads and are untouched. The LOWEST TWO shared K/V streams -- layers 1's and 3's at this
        config, the ones a chain of blocks separates from the token ids -- are not stored at all:
        the step recomputes their k and v from the id ring exactly as layer 0's have been
        recomputed since the ids replaced that layer's ring, through ONE MORE HOP per stream.
        At `KV_STREAM_COUNT` 2 those two ARE every producer, so `stored` is EMPTY, the code and
        scale storages take extent 0 on their stream axis and hold no bytes, and `slot_of` is
        `None` for every layer. The store and its codec are kept whole, and are what a tree with
        a producer above the rebuilt ones uses.
        """
        producers = [i for i in range(1, self.config.n_layer) if self.kv_sources[i] == i]
        rebuilt = producers[:2]
        stored = [p for p in producers if p not in rebuilt]
        slot_of = [None] + [None if self.kv_sources[i] in rebuilt
                            else stored.index(self.kv_sources[i])
                            for i in range(1, self.config.n_layer)]
        return rebuilt, stored, slot_of

    def _decode_phase(self, state):
        """Per layer, whether its ring is FULL (host-side, constant over a graph's life).

        "Full" is read one step earlier for the ID ring than for a K/V ring, and the asymmetry
        is a property of the write order, not a tolerance. Layer 0 writes this step's own id
        into the id ring BEFORE `_rebuild_kv_from_ids` reads it, so at `pos == L-1` all `L`
        slots hold a position of this request and the position-ordered map
        `base = seq - (L-1) + arange(L)` has `min(base) = pos - (L-1) == 0`: it degenerates to
        `arange(L)`, the filling branch's own ordering. A K/V ring is written AFTER its own
        read, so at `pos == L-1` its last slot is still unwritten and its element must stay at
        `pos >= length`; widening it would flip the label one step before the branch and replay
        a graph captured for the other branch. Element 0 must equal `_decode_body`'s
        `ring_wrapped` for every `pos`, and elements 1.. the stored-ring branch condition.
        """
        lens = state["cache_lens"]
        return ((state["pos"] >= lens[0] - 1,)
                + tuple(state["pos"] >= length for length in lens[1:]))

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
        # Each layer's cache is addressed as a ring: absolute position p lives in slot
        # p % L_i, and the ring is attended with no positional mask -- RoPE is applied
        # before the key is stored, so key order carries no positional meaning, and softmax
        # over keys is permutation-invariant.
        # A K/V ring is sized at the layer's window extent MINUS ONE, because it retains
        # only the predecessors a future step reads: this step's own k and v are already in
        # hand and are concatenated onto the ring for the one kernel call that reads them,
        # so the ring is not the place the current key is parked on its way in.
        # The id ring is not a K/V ring and is not sized like one: it holds token ids, its
        # newest entry is this step's own id, and `_rebuild_kv_from_ids` reconstructs from it
        # layer 0's k and v and EVERY rebuilt stream's, once the ring has wrapped and while it
        # is still filling. Its width is `_id_ring_len(max_len)` -- layer 0's window extent plus
        # ONE `_history_len` PER REBUILT STREAM plus one -- because each rebuilt stream sits one
        # hop higher than the last and its query's oldest key is a function of ids one more
        # window further back again.
        lens = ([self._id_ring_len(max_len)]
                + [self._history_len(i, max_len) for i in range(1, cfg.n_layer)])
        # ONE storage holds the codes for BOTH sides of the store, on a leading axis of extent
        # 2, because the two sides share ONE code width -- and at this config the STREAM axis
        # after it has extent ZERO, because every producer is rebuilt from the token ids (see
        # `_decode_streams`). So `codes` and `scales` have no elements and cost NO bytes, and
        # `torch.zeros` of an empty shape asks the allocator for nothing. They are still created,
        # at the shape a stored producer would need, so the branches below are the same code on a
        # tree that has one. The retained K/V data of THIS state is therefore the token-id `book`
        # alone: it is created here, reachable only from this dict and on this device, so
        # dropping the state frees it and it stays on the GPU for the whole request.
        # Layer 0 and EVERY shared K/V stream own no K/V storage at all: their k and
        # v are a pure function of the token ids in one window per hop and of the tokens'
        # absolute positions (see `_rebuild_kv_from_ids`), so the state keeps ONE ring of
        # token IDS for all of them -- 1 + 5/8 B per slot, against n_kv_head*head_dim*2*2 per
        # slot per stream -- and the step recomputes their resident k/v into scratch. The id ring is
        # the retained history held in place of that KV data, it is created here, it is reachable
        # only from this dict, and it is on the same device, so dropping the state frees it and
        # it stays on the GPU for the whole request.
        # The K/V rings are stored as narrow codes at ONE WIDTH, in one uint8 storage.
        # A KEY head vector is 64 FIVE-bit codes packed eight fields to five bytes, so a key
        # slot costs head_dim*5/8 = 40 B AND NOTHING ELSE. A VALUE head vector is the SAME 64
        # five-bit codes plus its own scale, so a value slot costs head_dim*5/8 + 2 = 42 B.
        # Against 2*head_dim = 128 B each.
        # A VALUE keeps its scale, which is that vector's own absmax over 15 -- a property of
        # the slot's own contents and nothing else, no running statistic, nothing carried
        # across a reset, nothing that depends on how many steps have run. It has to be
        # stored: a value is `v + gate*ve` and is never normed, so its per-vector rms is not a
        # constant (0.77 to 1.51 over one request, measured on random weights) and there is no
        # identity to recover it from.
        # A KEY needs NO stored scale, because its scale is not independent information. The
        # key is rms-normed at its own layer before it is stored, so the reconstruction
        # `code * scale` has rms 1 by construction and therefore `scale == 1/rms(code)`: the
        # promotion in `_decode_body` re-derives it from the codes themselves and reconstructs
        # `code / rms(code)`, which is the same vector renormalised. The whole residue is the
        # stored key's own rms deviation from 1 -- 0.973402 to 1.024506 per vector on the
        # FIVE-bit grid the key side carries, measured through the expressions above over
        # 4 x 4,096 random unit-rms head vectors at seeds 0, 1, 2 and 35, the bf16 reading --
        # against a quantisation error, on those same vectors, of mean |delta| 0.042458 on that
        # five-bit key grid. The VALUE side is now on the SAME five-bit grid: the scale stored
        # beside it is that vector's own absmax over 15, and its reconstruction error is at most
        # half a stored scale per channel -- amax/30 up to the scale's own bf16 rounding -- by
        # construction.
        # BOTH storages carry the stored-stream count on a single axis, the codes under their
        # own leading k/v axis of extent 2, because the allocator rounds each request up to a
        # 512 B block and because one axis lets the step promote, quantise, pack and write the
        # WHOLE store with one op each instead of one per side and one per stream.
        rebuilt, stored, _ = self._decode_streams()
        # The clause is a statement about the ONE stacked code storage, and only the STORED
        # streams occupy it: they share its ring axis, so they must share one history length.
        # A REBUILT stream's `_history_len` sizes no storage at all -- it enters
        # `_id_ring_len`'s hop sum and is a window extent that `forward` and
        # `_rebuild_kv_from_ids` read -- so it is not required to equal `ring`. The stored
        # stream's CONSUMERS need no clause here either: `GPT.__init__` already requires a
        # consumer's window to equal its producer's, and a consumer reads the producer's
        # slot. The `else` branch keeps the parent's expression for a store-free tree, whose
        # stream axis has extent 0 and costs zero bytes either way.
        ring = lens[stored[0]] if stored else lens[1]
        assert all(lens[p] == ring for p in stored), (
            f"the stored streams {stored} share one stacked ring storage, so they must share "
            f"one history length; got {[lens[p] for p in stored]}")
        # One ring per shared STREAM, not per layer: a pair's keys and values are computed
        # once, by the producer, so they are stored once and both layers read the same slot.
        # And TWO rings fewer than there are streams, because BOTH shared streams are rebuilt:
        # the leading axis is the number of streams STILL STORED, which is ZERO here, and a
        # layer indexes it through `_decode_streams`'s `slot_of`, which is `None` for every layer.
        producers = [i for i in range(1, cfg.n_layer) if self.kv_sources[i] == i]
        assert (rebuilt == producers[:len(rebuilt)] and rebuilt[0] == 1
                and self.kv_sources[0] == 0 and not has_ve(0, cfg.n_layer)), (
            f"rebuilt streams {rebuilt} of producers {producers}, layer 0 source "
            f"{self.kv_sources[0]} has_ve={has_ve(0, cfg.n_layer)}: the rebuild reaches the "
            "token ids through layer 0 ALONE, so the rebuilt streams must be the CONTIGUOUS "
            "LOWEST producers starting at layer 1, and layer 0 must be its own source and hold "
            "no value embedding, so that its k and v are a function of the token id alone")
        assert all(self.kv_sources[i] == rebuilt[n]
                   for n in range(len(rebuilt) - 1)
                   for i in range(rebuilt[n], rebuilt[n + 1])), (
            f"every layer between one rebuilt stream and the next must READ the lower one, or "
            f"the chain misses a hop: rebuilt {rebuilt}, kv_sources {self.kv_sources}")
        # A hop is read from `cache_lens` at the REBUILT stream's own index, by `_id_ring_len`
        # and by `_rebuild_kv_from_ids` alike, so a hop need not equal the stored `ring`:
        # here the hops are 239 and 384 wide and no stream is stored at all.
        assert head_dim % 8 == 0, (
            f"the codes are packed eight FIELDS to five bytes on BOTH sides of the store, which "
            f"share ONE five-bit width, so head_dim must be a multiple of 8; got {head_dim}")
        codes = torch.zeros(2, len(stored), batch, ring, cfg.n_kv_head, head_dim * 5 // 8,
                            dtype=torch.uint8, device=dev)
        scales = torch.zeros(len(stored), batch, ring, cfg.n_kv_head, 1, **kw)
        # The state's integer bookkeeping is ONE storage: both planes of the token-id ring and
        # the write position share a single int32 buffer, so the allocator's 512 B rounding is
        # paid once instead of three times.
        # A SLOT IS 13 BITS, SPLIT 8 + 5, and the two halves live in two planes rather than in
        # one packed array, because a slot's LOW byte then owns a whole byte and its write is an
        # `index_copy_` of one element exactly as the int16 ring's was. The HIGH plane is PACKED
        # EIGHT FIELDS TO FIVE BYTES by the `_pack5` / `_unpack5` pair, which the WHOLE K/V
        # store now shares: TWO callers, TWO alphabets, ONE 40-bit-word construction -- eight
        # consecutive five-bit fields are one word, field `n` at bits `5n .. 5n+4` and byte `m`
        # of the five at bits `8m .. 8m+7`, so a word is at most 2**40 - 1, is never negative,
        # and every shift in `_read_ids` and `_write_id` is exact. BOTH sides of the store are
        # five bits and are packed by that one pair, so the store has ONE width and ONE proven
        # bound again (15.051852, re-derived in `_quantise_kv`).
        # A slot of the ID ring costs 1 + 5/8 B instead of 1 + 2/3,
        # which with the pad below removed is what takes the book from six 512 B blocks to five
        # at the headline instrument shape. The packing is LOSSLESS -- the assert below pins the
        # 13-bit precondition in the source rather than assuming it -- so the ids the step reads
        # back are bit-for-bit the ids the int16 ring held.
        # `_pack5`'s alphabet HERE is the full signed five-bit range -16 .. 15, i.e. field values
        # 0 .. 31: an id's high five bits are `idx >> 8` for `idx < 2**13`, so field 0 occurs and
        # is offset to code -16 on the way in and back to 0 on the way out. The codec is a
        # bit-field operation, so it is exact over that whole range by construction.
        # `seq` does NOT narrow, and is NOT packed. It is the `cache_seqlens` argument of
        # `flash_attn_with_kvcache`, and that kernel reads int32: so the buffer is allocated in
        # int32 WORDS, `seq` is the LEADING word per row -- still int32, still contiguous, still
        # `numel == batch` -- and the two id planes are uint8 views of the bytes AFTER it.
        # `seq` FIRST is what replaces the parent's alignment pad. At offset 0 `seq`'s pointer is
        # the storage's own base pointer, so it is at least as aligned as a freshly allocated
        # tensor's -- strictly better than the 128-byte offset the parent rounded up to, and
        # bought rather than paid for, since the 123 B of pad that rounding cost is now gone.
        # The position and the two planes occupy DISJOINT byte ranges of the one storage, so a
        # write to any of them cannot touch the others, and all three are reachable only from
        # the returned dict: dropping the state drops the views and frees the storage, on the
        # same device.
        assert cfg.vocab_size <= 8192, (
            f"a slot of the id ring is one LOW byte plus a FIVE-bit HIGH field, so every token "
            f"id must be under 2**13; got vocab_size {cfg.vocab_size}")
        hi_groups = (lens[0] + 7) // 8
        hi_bytes = batch * hi_groups * 5
        lo_bytes = batch * lens[0]
        book = torch.zeros(batch + (hi_bytes + lo_bytes + 3) // 4,
                           dtype=torch.int32, device=dev)
        seq = book[:batch]
        plane = book.view(torch.uint8)
        ids_hi = plane[batch * 4:batch * 4 + hi_bytes].view(batch, hi_groups * 5)
        ids_lo = plane[batch * 4 + hi_bytes:batch * 4 + hi_bytes + lo_bytes].view(
            batch, lens[0])
        return {
            "max_len": max_len,
            "graph_enabled": bool(graph),
            "cache_lens": lens,
            "pos": 0,
            "seq": seq,
            "codes": codes,
            "scales": scales,
            "ids_lo": ids_lo,
            "ids_hi": ids_hi,
        }

    @staticmethod
    def _pack5(codes):
        """Five-bit signed codes in [-16, 15] -> bytes, eight codes to five bytes.

        The code is offset to [0, 31] so the layout is unsigned, and eight consecutive FIELDS
        share five bytes as ONE 40-BIT WORD: field `c` of the group occupies bits
        `5c .. 5c+4`, and byte `n` of the five is bits `8n .. 8n+7`. TWO callers, with TWO
        alphabets: the token-id ring passes eight consecutive SLOTS' high five bits, `idx >> 8`
        offset by -16, which uses the FULL signed five-bit alphabet -16 .. 15, i.e. field values
        0 .. 31; and the K/V store passes codes -15 .. 15 on BOTH of its sides, one step inside
        the field, under the bound 15.051852 < 15.5. This is a bit-field operation, so it is
        exact over the whole alphabet by construction.

        The word is FORMED and CUT by BROADCAST shifts rather than by a per-byte chain of
        masked shifts and ors: one shift against the eight field offsets `0, 5 .. 35` places
        every code, a three-step OR tree folds the eight shifted codes into the word, and one
        shift against the five byte offsets `0, 8 .. 32` cuts it. Both offset vectors are
        strided views of ONE `arange(40)`, so the whole spelling costs nine ops at any code
        width, where the per-byte chain costs three masked shift-or triples at six bits and
        would cost seven at five, a field crossing a byte boundary for six of every eight
        channels instead of two of every four.
        The fold is an OR and not a SUM. The fields are disjoint, so the two are the same
        number; `sum.dim_IntList` is one of the few ops with an `AutocastCUDA` registration
        (`pow` and `rsqrt` are the others this file already avoids), and while its fp32 cast
        policy does not touch an integer tensor, an OR tree needs no such argument and adds no
        op kind this file does not already dispatch.
        `to(torch.uint8)` truncates each shifted word modulo 256, which is exactly the byte
        wanted -- the same truncation `_write_id` takes for the id ring's low byte -- so no
        mask is needed.
        """
        u = codes.add(16.0).to(torch.int64).unflatten(-1, (-1, 8))
        offset = torch.arange(40, device=codes.device)
        w = u << offset[::5]
        w = w[..., :4] | w[..., 4:]
        w = w[..., :2] | w[..., 2:]
        word = w[..., :1] | w[..., 1:]
        return (word >> offset[::8]).to(torch.uint8).flatten(-2)

    @staticmethod
    def _unpack5(packed):
        """`_pack5`'s inverse, as bf16 values in [-16, 15]. Exact: integers that small are
        bf16-exact, so the round trip loses nothing and the promotion below is one multiply.

        The five bytes are reassembled into their 40-bit word by one broadcast left shift and
        the same three-step OR tree, and the eight fields are cut out by one broadcast right
        shift and one mask. Every intermediate is a non-negative int64 below 2**40, so every
        shift is exact; the top field would need no mask, but one broadcast mask covers all
        eight.
        """
        g = packed.unflatten(-1, (-1, 5)).to(torch.int64)
        offset = torch.arange(40, device=packed.device)
        w = g << offset[::8]
        w = w[..., :4] | w[..., 4:]
        w = w[..., :2] | w[..., 2:]
        word = w[..., :1] | w[..., 1:]
        u = (word >> offset[::5]) & 31
        return u.flatten(-2).to(torch.bfloat16) - 16.0

    @staticmethod
    def _quantise_kv(kv):
        """k AND v as packed FIVE-bit codes in one tensor, plus the bf16 scale that
        reconstructs the VALUES.

        The CALLER stacks k and v on a leading axis of extent 2, and may stack the STREAMS
        on any number of axes after it, so ONE call quantises the whole store -- and ONE call
        serves both sides because they share ONE grid, `absmax/15` over a head vector's own
        channels: there is ONE scale expression, ONE `round` and ONE packer for both sides
        rather than one each. The scale is rounded to bf16
        BEFORE the codes are computed, so a code is the nearest integer multiple of the scale
        that is actually stored and reconstruction error is at most half a stored scale with no
        drift from the scale's own rounding. Only the VALUE scale is returned to be stored: the
        key's is recovered from the codes at reconstruction time (see `init_decode_state` and
        `_decode_body`), because a stored key has unit rms by construction, while a value is
        `v + gate*ve`, never normed, and has no such identity. `clamp_min` only keeps an
        all-zero vector (an unfilled slot) from dividing by zero.
        `1.0/15.0` is not a bf16 value, so there are TWO roundings, not one, and the bound is
        not a relative power of two: swept over every positive finite
        bf16 value, `max(amax/scale)` is 15.051852 across the 18,094 values `clamp_min(1e-4)`
        admits, and it governs BOTH sides because both are on that one grid. It sits below 15.5,
        so no rounded code can reach the limit of five signed bits on either side.
        Below the clamp the bound fails outright,
        which is what the clamp is for. The division is done in float32 because a bf16
        quotient near the top of the range carries half a code of error of its own.
        """
        amax = kv.abs().amax(-1, keepdim=True).clamp_min(1e-4)
        scale = amax.mul(1.0 / 15.0)
        codes = kv.float().div(scale).round()
        return GPT._pack5(codes), scale[1]

    @staticmethod
    def _read_ids(state, L):
        """The whole id ring as int64 token ids in SLOT order. One unpack per step.

        The high plane holds EIGHT five-bit fields per five bytes, so the id ring's own `_unpack5`
        reads it: one broadcast shift reassembles each 40-bit word, one broadcast shift and one
        mask cut the eight fields out, and the result is already in slot order and is truncated
        to `L`. `_unpack5` returns `field - 16` as bf16, and 0 .. 31 is bf16-exact, so the `+ 16`
        restores the field exactly rather than approximately. Nothing is written back, so the
        cache-byte instrument sees no retained scratch.
        """
        fields = GPT._unpack5(state["ids_hi"]).to(torch.int64) + 16
        return state["ids_lo"] | (fields[:, :L] << 8)

    @staticmethod
    def _write_id(state, slots, idx):
        """Store ONE position's token id at ring slot `slots` (a one-element index).

        The low byte lands in its own plane, where a slot owns a whole byte, so that write is
        the parent's `index_copy_` with a narrower dtype. The high five bits share a FIVE-BYTE
        GROUP with seven other slots, so that write is the same read-modify-write of one field
        the int16 word took, one 40-bit group wide instead of one word: the group's five bytes
        are gathered and folded into their word by `_pack5`'s own broadcast shift and OR tree,
        `old + ((new - old_field) << shift)` replaces this slot's field and keeps the other
        seven exactly, forming no mask constant beyond the `& 31` that reads the old field, and
        the word is cut back into five bytes by the same broadcast shift. `to(torch.uint8)`
        truncates each shifted word modulo 256, which is exactly the byte wanted, and truncates
        the id modulo 256 for the low plane.
        """
        state["ids_lo"].index_copy_(1, slots, idx.to(torch.uint8))
        groups = state["ids_hi"].unflatten(-1, (-1, 5))
        group = slots // 8
        shift = (slots - group * 8) * 5
        g = groups.index_select(1, group).to(torch.int64)
        offset = torch.arange(40, device=g.device)
        w = g << offset[::8]
        w = w[..., :4] | w[..., 4:]
        w = w[..., :2] | w[..., 2:]
        word = w[..., :1] | w[..., 1:]
        field = (idx >> 8).unsqueeze(-1)
        word = word + ((field - ((word >> shift) & 31)) << shift)
        groups.index_copy_(1, group, (word >> offset[::8]).to(torch.uint8))

    def _rebuild_kv_from_ids(self, state, L, B, wrapped):
        """Layer 0's resident k/v AND every rebuilt stream's, recomputed from the id ring.

        Layer 0's block input is `norm(resid_lambdas[0]*x0 + x0_lambdas[0]*x0)` with
        `x0 = norm(wte(idx))` and the layer has no value embedding, so its k and v depend on
        nothing but the token id; RoPE then depends on nothing but the token's absolute
        position. Each rebuilt stream sits one more HOP above: the lowest one's k and v are
        formed from layer 0's block OUTPUT, which that layer's `(left0, 0)` window makes a
        function of the ids at `p - left0 .. p`; the layers between two rebuilt streams read the
        lower stream under their own `(left, 0)` window, so the residual entering the higher
        stream at `q` is a function of the lower stream's keys at `q - left .. q` and of `x0` at
        `q`; and the higher stream's k and v then come from that residual through its own `c_k`,
        `c_v`, `ve_gate` and `value_embeds`. So every rebuilt stream is a function of the ids in
        one window PER HOP and of the positions, and the id ring is the retained history held in
        place of all of that KV data.

        ROW WIDTHS are the chain's correctness argument, and each hop is one `_history_len`
        wider than the hop above it. A query at `p` reads the HIGHEST rebuilt stream's keys at
        `p - hist .. p`, so that stream is rebuilt over `hist + 1` rows; each of those rows is a
        function of the stream BELOW it over its own `hist` further back, so the lower stream is
        rebuilt over `hist_lower + hist + 1` rows; and layer 0's own k and v are rebuilt over
        all `L` ring rows, because the lowest stream's rows each read a `left0 + 1` key window
        (`rows_lowest + left0 == L` by `_id_ring_len`). At this config that is `L` rows of the
        cheap per-row work, `1 + 384 + 239 = 624` rows for layer 0's block and the lowest
        stream, and `1 + 384 = 385` rows for the highest. Every row count is capped at `L`: when
        the hop sum exceeds `max_len - 1` the running count asks for rows at positions older
        than the request's own first one, which do not exist, and the whole `L`-row
        reconstruction is then what every stage reads.

        Two orders, both chosen by a host-side flag that is constant over a graph's life:

        * wrapped -- the ring holds the L most recent positions, so the POSITION-ordered array
          is `slots = (p - (L-1) + arange(L)) % L` at positions `p - (L-1) .. p`. That is the
          same map the slot-ordered rebuild used, read the other way round: slot
          `j = (p - (L-1) + m) % L` holds position `p - ((p - j) mod L) = p - (L-1) + m`.
          Bottom-right alignment then places the LAST `rows` rows of every stage at exactly
          their own positions, so ONE `window_size=self.window_sizes[i]` call per hop gives each
          of those rows its own left-window with no dynamic slice: 624 queries over L keys under
          `(1, 0)` gives each query its own 2-key window, and 385 queries over 624 keys
          under `(239, 0)` gives each its own 240-key window.
        * filling -- the request has written slot `j` from position `j`, in order, so the ring
          itself is the ordered array and EVERY row of EVERY stage is computed, because the rows
          a query reads sit at a position-dependent offset that must not be sliced away.

        The ordered map is valid from `pos == L-1`, which is where the flag that selects it
        flips: `min(base) = pos - (L-1)`, so no row reads a negative position exactly when
        `pos >= L-1`, and at `pos == L-1` the map IS `arange(L)`, the filling branch's own
        ordering. Every slot it reads has been written, because layer 0 writes this step's id
        before this call (the prefill writes slots `0 .. Tn-1` when `Tn <= L`, and if `Tn > L`
        the ring has already wrapped and `pos > L`). So `clamp_min(0)` is not load-bearing on
        the ordered path; it only keeps the cos/sin gather in range on the filling path, for
        slots this request has not filled yet, which `cache_seqlens` masks out of the kernel.
        Nothing is written back into the state, so the instrument sees no retained scratch.
        """
        rebuilt, _, _ = self._decode_streams()
        attn = self.transformer.h[0].attn
        ids = self._read_ids(state, L)
        if wrapped:
            # One hop per rebuilt stream: the highest is `hist + 1` rows and each lower one is
            # that plus its own `hist`, so the widths are the reversed running sum.
            rows_of = {}
            acc = 1
            for r in reversed(rebuilt):
                acc += state["cache_lens"][r]
                # Capped at `L`, which is `_id_ring_len` and which `max_len` caps in turn: with
                # a hop sum above `max_len - 1` the running count asks for rows before position
                # 0 of the request, so the whole ring is the row set and every query still gets
                # its own window by bottom-right alignment. Uncapped, the `view(B, rows, ...)`
                # below raises at that geometry.
                rows_of[r] = min(acc, L)
            base = (state["seq"][:1].to(torch.int64) - (L - 1)
                    + torch.arange(L, device=ids.device, dtype=torch.int64))
            ids_ord = ids.index_select(1, base % L)
            pos = base.clamp_min(0)
        else:
            rows_of = {r: L for r in rebuilt}
            ids_ord = ids
            pos = torch.arange(L, device=ids.device, dtype=torch.int64)
        z = norm(self.transformer.wte(ids_ord))
        xc = self.resid_lambdas[0] * z + self.x0_lambdas[0] * z
        h = norm(xc)
        k = attn.c_k(h).view(B, L, attn.n_kv_head, attn.head_dim)
        v = attn.c_v(h).view(B, L, attn.n_kv_head, attn.head_dim)
        cos = self.cos.index_select(1, pos)
        sin = self.sin.index_select(1, pos)
        k = norm(apply_rotary_emb(k, cos, sin))
        # Hop 1: layer 0's own block, over the rows the LOWEST rebuilt stream needs and no
        # others. Its keys and values are the whole L-row reconstruction above, attended under
        # layer 0's own window, so each of these rows gets exactly its own `left0 + 1` keys.
        rows = rows_of[rebuilt[0]]
        cos_r, sin_r = cos[:, -rows:], sin[:, -rows:]
        q = attn.c_q(h[:, -rows:]).view(B, rows, attn.n_head, attn.head_dim)
        q = norm(apply_rotary_emb(q, cos_r, sin_r))
        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=self.window_sizes[0])
        x = xc[:, -rows:] + attn.c_proj(y.contiguous().view(B, rows, -1))
        x = x + self.transformer.h[0].mlp(norm(x))
        # Each rebuilt stream's own k and v, from the residual at that depth -- its own c_k,
        # c_v, ve_gate and value embedding table, the same parameters `forward` uses, and no
        # others -- and then the blocks that READ that stream, which carry the residual up to
        # the next one over its own narrower row set.
        out = []
        for n, r in enumerate(rebuilt):
            rows = rows_of[r]
            cos_r, sin_r = cos[:, -rows:], sin[:, -rows:]
            attn_r = self.transformer.h[r].attn
            x = self.resid_lambdas[r] * x[:, -rows:] + self.x0_lambdas[r] * z[:, -rows:]
            h_r = norm(x)
            k_r = attn_r.c_k(h_r).view(B, rows, attn_r.n_kv_head, attn_r.head_dim)
            v_r = attn_r.c_v(h_r).view(B, rows, attn_r.n_kv_head, attn_r.head_dim)
            if str(r) in self.value_embeds:
                ve = self.value_embeds[str(r)](ids_ord[:, -rows:]).view(
                    B, rows, attn_r.n_kv_head, attn_r.head_dim)
                gate = 2 * torch.sigmoid(attn_r.ve_gate(h_r[..., :attn_r.ve_gate_channels]))
                v_r = v_r + gate.unsqueeze(-1) * ve
            k_r = norm(apply_rotary_emb(k_r, cos_r, sin_r))
            out.append((k_r, v_r))
            if n + 1 == len(rebuilt):
                break
            # The hop up to the next rebuilt stream: every layer from `r` to the next one reads
            # THIS stream's reconstruction, each under its OWN window, over the next stream's
            # narrower row set. `window_size` is named rather than left at `(-1, -1)`, because
            # this reconstruction is WIDER than those layers' window and `(-1, -1)` would let
            # them attend keys older than it.
            nxt = rebuilt[n + 1]
            rows_n = rows_of[nxt]
            cos_n, sin_n = cos[:, -rows_n:], sin[:, -rows_n:]
            x, h_n = x[:, -rows_n:], h_r[:, -rows_n:]
            for i in range(r, nxt):
                attn_i = self.transformer.h[i].attn
                block_i = self.transformer.h[i]
                q_i = attn_i.c_q(h_n).view(B, rows_n, attn_i.n_head, attn_i.head_dim)
                q_i = norm(apply_rotary_emb(q_i, cos_n, sin_n))
                y_i = fa3.flash_attn_func(q_i, k_r, v_r, causal=True,
                                          window_size=self.window_sizes[i])
                x = x + attn_i.c_proj(y_i.contiguous().view(B, rows_n, -1))
                x = x + block_i.mlp(norm(x))
                if i + 1 < nxt:
                    x = (self.resid_lambdas[i + 1] * x
                         + self.x0_lambdas[i + 1] * z[:, -rows_n:])
                    h_n = norm(x)
        return k, v, out

    def reset_decode_state(self, state):
        """Return `state` to position 0 so the same buffers serve another request.

        A fresh state per request would invalidate an address-bound CUDA graph. Stale bytes
        above the position are unreachable -- attention reads only `cache_seqlens` keys -- so
        zeroing them would be timed work serving never does.
        """
        state["seq"].zero_()
        state["pos"] = 0
        return state

    def _decode_body(self, idx, state, prefill):
        """`forward`'s body with the cache spliced in. `prefill` is a Python bool that is
        constant per variant, so the two branches are two graphs, not one per position."""
        B, Tn = idx.size()
        if prefill:
            cos, sin = self.cos[:, :Tn], self.sin[:, :Tn]
        else:
            seq_idx = state["seq"].to(torch.int64)
            cos = self.cos.index_select(1, seq_idx)
            sin = self.sin.index_select(1, seq_idx)

        # Every retained ring holds only PREVIOUS positions, and no layer writes its ring
        # before it reads it, so the whole narrow ring set is promoted to bf16 here rather
        # than once per layer: ONE promotion for the WHOLE store, because the sides share one
        # width again -- one `_unpack5` over the single code storage, whose leading axis is
        # already k/v -- and the two sides restacked into the
        # (stream, k/v, ...) layout every downstream reader already expects. FA3 never
        # sees a narrow dtype. The reconstruction is a local of this call and is freed when it
        # returns, so the cache-byte instrument, which reads current allocation after the
        # request, never sees it.
        # The VALUES are unpacked and then rescaled by their stored scale, which is the WHOLE
        # of the difference between the two sides again. The KEYS are
        # RENORMALISED instead,
        # which is the same number: a stored key was rms-normed at its own layer, so
        # `code * scale` has rms 1 and `scale == 1/rms(code)`, and `code * rsqrt(mean(code**2))`
        # reconstructs it without a stored key scale at all. The reduction is over the head
        # vector's own 64 channels -- the same grouping the absmax and the norm used -- so it
        # is a property of the slot's own contents and of nothing else, exactly as the stored
        # scale was: it does not depend on the step number, on the phase, or on any other slot.
        # `code / sqrt(mean(code*code) + eps)`, and the spelling is load-bearing rather than a
        # style: under `torch.autocast` on CUDA both `pow` and `rsqrt` are fp32-cast ops, so
        # `code * rsqrt(mean(code**2) + eps)` returns a float32 reconstruction and
        # `flash_attn_*` refuses a float32 key against a bf16 query. `mul`, `mean`, `add`,
        # `sqrt` and `div` all keep the store's own dtype, so this form is the same arithmetic
        # at the same op count with the dtype contract intact.
        # `1e-6` inside the reduction is not a tolerance: it is what keeps an all-zero code
        # vector (an unfilled slot, or a key that was exactly zero) reconstructing as 0 rather
        # than as 0 times an infinite scale. An unfilled slot's zero bytes decode to FIELD 0 on
        # both sides -- code -16, which the key side renormalises to -1 in every channel and
        # which the value side's stored zero scale takes to exactly 0 -- so the invariant that
        # no unfilled slot is ever READ is what makes both
        # harmless, and it is the parent's invariant, unchanged: a ring is full or masked by
        # `cache_seqlens`.
        # Which streams are rebuilt rather than stored, and which storage slot each stored
        # stream occupies. Host-side and derived from `self.kv_sources`, so `forward`'s own view
        # of the pairing is untouched and nothing here depends on the step number. It is read
        # HERE, above the promotion, because when NOTHING is stored there is nothing to promote:
        # the code and scale storages have extent 0 on their stream axis, and unpacking and
        # rescaling them would run the whole codec over empty tensors for no values at all. The
        # flag is host-side and constant over a graph's life, exactly as `prefill` is.
        rebuilt, stored, slot_of = self._decode_streams()
        promote = (not prefill) and bool(stored)
        udeq = self._unpack5(state["codes"]) if promote else None
        deq = torch.stack(
            [udeq[0].div(udeq[0].mul(udeq[0]).mean(-1, keepdim=True).add(1e-6).sqrt()),
             udeq[1] * state["scales"]], 1) if promote else None
        rebuilt_kv = {}
        pend = [None] * len(stored)
        # Whether the ID RING IS FULL, which is what decides the reconstruction's order and
        # therefore which branch EVERY rebuilt-stream layer takes. It is `_decode_phase`'s first
        # element, host-side and constant over a graph's life, and the two must agree for every
        # `pos` or a replay runs the branch it was not captured for.
        # It is read at `L - 1`, not at `L`: this step's own id is written into the ring above
        # before the rebuild reads it, so from `pos == L-1` every slot holds a position of this
        # request and `min(base) = pos - (L-1) >= 0` (see `_rebuild_kv_from_ids`). One step
        # earlier both of those fail, which is why the offset is exactly one.
        ring_wrapped = (not prefill) and state["pos"] >= state["cache_lens"][0] - 1

        x = norm(self.transformer.wte(idx))
        x0 = x
        kvs = [None] * self.config.n_layer
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            attn = block.attn
            h = norm(x)
            L = state["cache_lens"][i]
            shares_kv = self.kv_sources[i] != i
            # A wrapped id ring already holds THIS step's own id, so the rebuild below supplies
            # this step's k and v as well as the history's, for layer 0 and for every rebuilt
            # stream alike: c_k, c_v, k's RoPE and norm(k) are then not computed at all. Host-
            # side flag, constant over a graph's life for the same reason `_decode_phase` is.
            rebuild_only = (i == 0 or i in rebuilt) and ring_wrapped
            q = attn.c_q(h).view(B, Tn, attn.n_head, attn.head_dim)
            if shares_kv:
                # This step's keys and values for this stream were computed by the producer
                # earlier in this same loop, and its wrapped-ring concatenation with them: so
                # c_k, c_v, the value-embedding mix, k's RoPE, norm(k), the quantisation and
                # the ring write are not repeated, and this layer only forms its own query.
                k, v, ring = kvs[self.kv_sources[i]]
                q = norm(apply_rotary_emb(q, cos, sin))
            elif rebuild_only:
                q = norm(apply_rotary_emb(q, cos, sin))
            else:
                k = attn.c_k(h).view(B, Tn, attn.n_kv_head, attn.head_dim)
                v = attn.c_v(h).view(B, Tn, attn.n_kv_head, attn.head_dim)
                if ve is not None:
                    ve = ve.view(B, Tn, attn.n_kv_head, attn.head_dim)
                    gate = 2 * torch.sigmoid(attn.ve_gate(h[..., :attn.ve_gate_channels]))
                    v = v + gate.unsqueeze(-1) * ve
                q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
                q, k = norm(q), norm(k)
                kvs[i] = (k, v, None)
            # A consumer's own value residual, at the query position: the same expression
            # `CausalSelfAttention.forward` adds to a consumer's attention output, on the
            # same parameters, from this step's own token alone. It is added to `y` in each
            # of the branches below, after the attention call and before `c_proj`, so the
            # decode path and the forward path compute the same sum. Nothing about the
            # history enters it, so no ring stores anything for it.
            ve_resid = None
            if shares_kv and ve is not None:
                gate = 2 * torch.sigmoid(attn.ve_gate(h[..., :attn.ve_gate_channels]))
                ve_resid = gate.unsqueeze(-1) * ve.view(B, Tn, attn.n_kv_head,
                                                        attn.head_dim)

            if i == 0:
                # Layer 0 stores ids where every other layer stores k and v. Everything else
                # about the three branches -- which kernel, which window, which mask -- is the
                # parent's, and is justified identically.
                lo, hi = state["ids_lo"], state["ids_hi"]
                if prefill:
                    y = fa3.flash_attn_func(q, k, v, causal=True,
                                            window_size=self.window_sizes[i])
                    # A prefill writes a CONTIGUOUS RUN of slots, so the high plane is packed
                    # wholesale rather than one field at a time: a per-slot read-modify-write
                    # would need the run's own earlier writes to be visible to its later ones,
                    # which `index_copy_` does not promise when two indices share a word.
                    if Tn <= L:
                        lo[:, :Tn] = idx.to(torch.uint8)
                        groups = (Tn + 7) // 8
                        h = (idx >> 8).float().sub_(16.0)
                        pad = groups * 8 - Tn
                        if pad:
                            h = torch.cat([h, h.new_full((B, pad), -16.0)], 1)
                        hi[:, :groups * 5] = self._pack5(h)
                    else:
                        slots = torch.arange(Tn - L, Tn, device=idx.device) % L
                        tail = idx[:, Tn - L:].contiguous()
                        lo.index_copy_(1, slots, tail.to(torch.uint8))
                        ordered = torch.full((B, L), -16.0, device=idx.device)
                        ordered.index_copy_(1, slots, (tail >> 8).float().sub_(16.0))
                        groups = (L + 7) // 8
                        pad = groups * 8 - L
                        if pad:
                            ordered = torch.cat(
                                [ordered, ordered.new_full((B, pad), -16.0)], 1)
                        hi[:, :groups * 5] = self._pack5(ordered)
                else:
                    slot = state["seq"][:1].to(torch.int64) % L
                    self._write_id(state, slot, idx)
                    kr, vr, rebuilt_out = self._rebuild_kv_from_ids(state, L, B, ring_wrapped)
                    rebuilt_kv = dict(zip(rebuilt, rebuilt_out))
                    if ring_wrapped:
                        # The position-ordered reconstruction is L = left0 + hist + 1 rows
                        # ending at this step's own position, which is WIDER than this layer's
                        # window, so the window has to be named: bottom-right alignment puts
                        # this step's query on the last row, and `window_size` then selects
                        # exactly the `left0 + 1` keys at `p - left0 .. p` -- the same key set
                        # a ring of the window's own extent handed the kernel whole.
                        y = fa3.flash_attn_func(q, kr, vr, causal=True,
                                                window_size=self.window_sizes[i])
                    else:
                        y = fa3.flash_attn_with_kvcache(q, kr, vr, k=k, v=v,
                                                        cache_seqlens=state["seq"],
                                                        causal=True,
                                                        window_size=self.window_sizes[i],
                                                        num_splits=1)
                x = x + attn.c_proj(y.contiguous().view(B, Tn, -1))
                x = x + block.mlp(norm(x))
                continue
            s = slot_of[i]
            owns_kv = not shares_kv
            if s is None:
                # A rebuilt stream: no codes, no scales, no `_quantise_kv`, no `index_copy_`
                # and no ring write. `_rebuild_kv_from_ids` has already produced this stream's
                # k and v for every position its window can reach, so the step attends that
                # reconstruction and stores nothing; the consumer reads the same reconstruction
                # through `kv_sources` exactly as it reads a producer's ring. The three branches
                # and the key sets they attend are the stored streams', unchanged.
                if prefill:
                    y = fa3.flash_attn_func(q, k, v, causal=True,
                                            window_size=self.window_sizes[i])
                elif ring_wrapped:
                    # The reconstruction holds this stream's rows in POSITION order ending at
                    # this step's own key, and it may be WIDER than this layer's window: the
                    # lowest rebuilt stream is rebuilt over its own history plus the stream
                    # above it, because the higher stream's rows each read it further back. So
                    # the window is NAMED rather than left at `(-1, -1)`: with one query row and
                    # bottom-right alignment it selects exactly the `hist + 1` keys at
                    # `p - hist .. p` -- the key set a stored stream's `cat(ring, current)`
                    # forms -- for every rebuilt stream, whatever its own row count. This step's
                    # own k and v were never computed.
                    if owns_kv:
                        ring = rebuilt_kv[i]
                        kvs[i] = (None, None, ring)
                    y = fa3.flash_attn_func(q, ring[0], ring[1], causal=True,
                                            window_size=self.window_sizes[i])
                else:
                    # Slot order while the ring fills, so the rows this query needs sit at a
                    # position-dependent offset and the reconstruction must NOT be sliced: the
                    # whole array goes to the kernel under `cache_seqlens`, exactly as a stored
                    # stream hands its whole preallocated ring. The in-place append lands on row
                    # `seq` of this step's scratch, which the reconstruction already holds the
                    # same value for, so the write changes nothing and this step's own key is
                    # visible to producer and consumer alike. A consumer is handed its
                    # PRODUCER's reconstruction, which is what `kv_sources` names.
                    rk, rv = rebuilt_kv[self.kv_sources[i]]
                    y = fa3.flash_attn_with_kvcache(q, rk, rv, k=k, v=v,
                                                    cache_seqlens=state["seq"], causal=True,
                                                    window_size=self.window_sizes[i],
                                                    num_splits=1)
                if ve_resid is not None:
                    y = y + ve_resid
                x = x + attn.c_proj(y.contiguous().view(B, Tn, -1))
                x = x + block.mlp(norm(x))
                continue
            if prefill:
                # Attend the freshly computed k, v: the same tensors and the same values the
                # preallocated-cache version attended via kc[:, :Tn], and required because
                # the ring may hold fewer slots than Tn. Then store the tail the future can
                # still read: the last min(Tn, L) positions, each at slot p % L.
                y = fa3.flash_attn_func(q, k, v, causal=True,
                                        window_size=self.window_sizes[i])
                if owns_kv:
                    pend[s] = torch.stack([k, v])
            elif state["pos"] >= L:
                # The ring is full, so it holds exactly the L predecessors this query may
                # read; its own k and v are already in hand, so the kernel is handed ring +
                # current and attends all L + 1 keys whole with no positional mask.
                # Only after that call is this step's k/v written into slot p % L, which
                # held position p - L -- exactly the position that has just left the window,
                # since the next query, at p + 1, reads p - L + 1 .. p + 1. So the evicted
                # key is one no later step in the request may read, the ring afterwards
                # holds p - L + 1 .. p, and no slot ever holds a key before the step that
                # produced it. The slot comes from the device position, so a captured graph
                # advances it without a host round-trip.
                if owns_kv:
                    kv = torch.stack([k, v])
                    pend[s] = kv
                    ring = torch.cat([deq[s], kv], 2)
                    kvs[i] = (k, v, ring)
                y = fa3.flash_attn_func(q, ring[0], ring[1], causal=True,
                                        window_size=(-1, -1))
            else:
                # Appends k,v at cache_seqlens and attends over the whole preallocated
                # cache, so there is no `kc[:, start:end]` slice whose bounds depend on the
                # step number. num_splits=1 is not a tuning choice: the op's fake kernel
                # refuses to trace at the default num_splits=0, which is precisely what
                # makes an unpinned cache uncompilable. It is a reduction-order choice, and
                # the agreement check in prepare.py still has to pass with it pinned.
                # The reconstruction is this step's scratch, so the kernel's in-place
                # append lands there and not in the ring; the ring's own copy of this step's
                # k and v is the quantised write below, at the same slot the parent wrote
                # (cache_seqlens == pos < L, so slot p % L is p). Junk above the position is
                # still unreachable to cache_seqlens.
                if owns_kv:
                    pend[s] = torch.stack([k, v])
                # A consumer hands the kernel the same reconstruction, the same k and v and
                # the same `cache_seqlens` its producer did, so the in-place append writes
                # the identical values to the identical slot of this step's scratch: the
                # duplicate write changes nothing, and it is what makes this step's own key
                # visible to the consumer's query exactly as it is to the producer's.
                y = fa3.flash_attn_with_kvcache(q, deq[s][0], deq[s][1], k=k, v=v,
                                                cache_seqlens=state["seq"], causal=True,
                                                window_size=self.window_sizes[i],
                                                num_splits=1)
            if ve_resid is not None:
                y = y + ve_resid
            x = x + attn.c_proj(y.contiguous().view(B, Tn, -1))
            x = x + block.mlp(norm(x))

        if pend and pend[0] is not None:
            # ONE store write per STORAGE per step, not one per stored stream. The streams are
            # stacked on the store's OWN stream axis -- `pend` is indexed by `slot_of`, so a
            # stream's position in the stack is the slot it is written to and cannot depend on
            # loop order -- quantised and packed by one `_quantise_kv` call, and written with one
            # `index_copy_` per storage instead of one per stream: TWO storages, the merged code
            # plane and the value scales, because the sides share one width again, and the RING
            # AXIS is index 3 of the codes, whose leading axis is k/v, and index 2 of the scales.
            # Deferring is legal because the step reads the store exactly ONCE, in `deq` above,
            # before any layer runs: nothing between there and here reads `codes`
            # or `scales`, every stored stream shares one ring width and therefore one slot, and
            # the values written are the same values computed from the same k and v. So this
            # changes the ORDER of the writes and nothing else -- same slots, same bytes.
            L = state["cache_lens"][stored[0]]
            codes_new, scale_new = self._quantise_kv(torch.stack(pend, 1))
            if prefill and Tn <= L:
                state["codes"][:, :, :, :Tn] = codes_new
                state["scales"][:, :, :Tn] = scale_new
            elif prefill:
                slots = torch.arange(Tn - L, Tn, device=idx.device) % L
                state["codes"].index_copy_(
                    3, slots, codes_new[:, :, :, Tn - L:].contiguous())
                state["scales"].index_copy_(
                    2, slots, scale_new[:, :, Tn - L:].contiguous())
            else:
                slot = state["seq"][:1].to(torch.int64) % L
                state["codes"].index_copy_(3, slot, codes_new)
                state["scales"].index_copy_(2, slot, scale_new)
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
            state["pos"] += idx.size(1)
            return logits, state
        if not state.get("graph_enabled", True):
            logits = self._decode_body(idx, state, prefill=False)
            state["seq"].add_(1)
            state["pos"] += 1
            return logits, state
        # The graph lives ON the state it was captured against, so its lifetime is exactly
        # that state's and there is no key to get wrong. A state whose capture failed holds
        # an uncaptured object and runs eager forever, which is slow and correct.
        graph = state.get("graph")
        # A graph is captured in one ring phase and replays the branch it captured, so it is
        # rebuilt if a layer's ring has wrapped since. The phase is host-side and changes at
        # most once per layer per request.
        if graph is None or graph.phase != self._decode_phase(state):
            graph = state["graph"] = _GraphedDecodeStep(self, state)
        if not graph.captured:
            logits = self._decode_body(idx, state, prefill=False)
            state["seq"].add_(1)
            state["pos"] += 1
            return logits, state
        logits = graph.replay(idx)
        state["pos"] += 1
        return logits, state


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
    _reported = False

    def __init__(self, model, state):
        self.model = model
        self.state = state
        # The ring phase this graph is captured in. decode_step rebuilds the graph rather
        # than replay one captured in another phase.
        self.phase = model._decode_phase(state)
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
        if not _GraphedDecodeStep._reported:
            _GraphedDecodeStep._reported = True
            print(f"[decode-graph] captured={self.captured} reason={self.reason!r} phase={self.phase}", flush=True)

    def _advance(self):
        logits = self.model._decode_body(self.static_idx, self.state, prefill=False)
        self.state["seq"].add_(1)
        return logits

    def _capture(self):
        seq0 = self.state["seq"].clone()
        # A wrapped ring is read in full, so "junk above `seq` is unreachable" no longer
        # holds: a warmup write lands on a live slot. Save the cache with `seq` and restore
        # both, so capture leaves no trace in the request. Addresses are unchanged, so the
        # captured graph still points at these tensors.
        # The layer-0 id ring is live data a warmup write corrupts exactly as a wrapped k/v
        # ring is. The merged code plane and the value scales are one storage each and the id
        # ring is TWO views of a third, so FOUR tensors cover every ring -- and the two id views
        # are disjoint byte ranges of one storage, so cloning and restoring each covers it
        # exactly once. On a store-free tree the first two hold no elements and no bytes at all
        # (`codes` and `scales` have extent 0 on their stream axis), so the real work of the list
        # is the two id views, and the THIRD tensor of their storage -- `seq`, the leading int32
        # word -- is saved and restored separately as `seq0` above. The list is kept whole so that
        # a tree with a stored producer needs no edit here. Addresses are unchanged, so the
        # captured graph still points at them.
        live = [self.state["codes"], self.state["scales"],
                self.state["ids_lo"], self.state["ids_hi"]]
        saved_live = [t.clone() for t in live]
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
        for tensor, saved in zip(live, saved_live):
            tensor.copy_(saved)

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
HEAD_DIM = 64           # target head dimension for attention
WINDOW_PATTERN = "BNNTTTTT"  # per-layer window: L=full, S=half, B=book-sized (see _compute_window_sizes), N=the low group (the lowest rebuilt stream and its reader), T=the top group (the highest rebuilt stream and its readers)

# Optimization
TOTAL_BATCH_SIZE = 2**18 # 262,144 tokens per optimizer step = DEVICE_BATCH_SIZE * MAX_SEQ_LEN, so every microbatch is one update and the accumulation loop runs once (grad_accum_steps = 1)
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
