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
        """Preallocate every layer's cache to ITS WINDOW EXTENT, and keep the write position
        ON DEVICE.

        Both give the per-step call constant shapes and no Python-visible position: a lazy
        cache slot is a new shape mid-loop, and a Python `pos` int makes Dynamo guard on its
        value and trip `recompile_limit`. `seq` is int32, advanced with `add_()`, and read by
        the attention kernel as `cache_seqlens`.

        A layer with left window `w` can attend to exactly `w + 1` positions at any position,
        so that is the whole cache it needs: `min(max_len, w + 1)`. Where that is shorter than
        `max_len` the buffer is written as a ring and every key in it is attendable, so the
        step's attention reads the window's extent instead of the whole prefilled context with
        the out-of-window part masked. The extent is derived from `self.window_sizes` and never
        written as a constant: the window pattern is what decides it.

        A ring layer also needs its next write slot and its live key count on device, for the
        same reason `seq` is: a Python value makes Dynamo guard on it, and reading one back on
        the host inside the step would break the capture. One pair per DISTINCT ring length,
        because the slot is a function of the position and the length and of nothing else.

        `graph=False` must run the step eagerly and capture nothing. It is how the instrument
        reads cache bytes without a graph's private pool in them, so a candidate that ignores
        the flag reports its own pool as cache and is charged for it.

        `key_splits` is how many chunks the width-1 step's attention call cuts the cached key
        range into, at BOTH of that step's call sites -- the ring layers' mask-free read of the
        whole live buffer, and the non-ring layers' read of a range that grows with the step.
        It is DERIVED, here, from the device and from this request's own shape -- one split per
        streaming multiprocessor that the call's query tiles leave idle -- and it is a plain
        Python int so that the op receives a tracing constant rather than a traced value. The
        count is fixed for the life of the state, which is what lets a captured replay reuse
        the kernel's split buffers. Computing it here rather than in the step also keeps the
        device-property read out of every step and out of every timed region. It is a property
        of the device and of the call's query tiles, which are the same on every layer, so the
        two call sites take the same count.
        """
        assert max_len <= self.cos.size(1), f"max_len {max_len} exceeds rotary table"
        cfg = self.config
        head_dim = cfg.n_embd // cfg.n_head
        dev = self.transformer.wte.weight.device
        kw = {"dtype": torch.bfloat16, "device": dev}
        # One partial reduction per SM, as far as the call's own query tiles allow: the width-1
        # step issues batch * n_head query tiles and nothing else, so this is the count that
        # makes the partials match the device. `max(1, ...)` is the floor the op requires -- it
        # refuses to trace at num_splits <= 0 -- and int() keeps it a Python value.
        sm_count = int(torch.cuda.get_device_properties(dev).multi_processor_count)
        splits = max(1, sm_count // (batch * cfg.n_head))
        # Free diagnostic, host side, at state construction: outside every timed region, and a
        # Python int holds no device bytes for the cache probe to charge. The SM count is on the
        # same line so the derivation is auditable from stdout rather than from the quotient.
        print(f"decode key_splits={splits} (sm_count={sm_count}, batch={batch}, "
              f"n_head={cfg.n_head})")
        lengths = [min(max_len, window[0] + 1) for window in self.window_sizes]
        # A layer whose extent already covers the whole request is NOT a ring: the shorter
        # allocation would be the same allocation and the ring's index writes would buy
        # nothing. Those layers keep the reference's path exactly.
        is_ring = [length < max_len for length in lengths]
        ring_lengths = sorted({length for length, ring in zip(lengths, is_ring) if ring})
        # int64 write slot because `index_copy_` takes a long index; int32 count because the
        # attention kernel reads it as `cache_seqlens`. The count is the number of keys live
        # AFTER the next append, i.e. min(seq + 1, length), so the step can append, read with
        # it, and advance -- see `_decode_body`.
        return {
            "max_len": max_len,
            "graph_enabled": bool(graph),
            "key_splits": splits,
            "seq": torch.zeros(batch, dtype=torch.int32, device=dev),
            "cache_lengths": lengths,
            "is_ring": is_ring,
            "ring_write": {length: torch.zeros(1, dtype=torch.int64, device=dev)
                           for length in ring_lengths},
            "ring_count": {length: torch.ones(batch, dtype=torch.int32, device=dev)
                           for length in ring_lengths},
            "kc": [torch.zeros(batch, length, cfg.n_kv_head, head_dim, **kw)
                   for length in lengths],
            "vc": [torch.zeros(batch, length, cfg.n_kv_head, head_dim, **kw)
                   for length in lengths],
        }

    def reset_decode_state(self, state):
        """Return `state` to position 0 so the same buffers serve another request.

        A fresh state per request would invalidate an address-bound CUDA graph. Stale bytes
        above the position are unreachable -- attention reads only `cache_seqlens` keys, and a
        ring's slots at or above its count are unreachable the same way -- so zeroing them
        would be timed work serving never does.
        """
        state["seq"].zero_()
        for index in state.get("ring_write", {}).values():
            index.zero_()
        for count in state.get("ring_count", {}).values():
            count.fill_(1)         # min(seq + 1, length) at seq = 0
        return state

    def _serving_weights(self):
        """bfloat16 copies of every weight the decode path multiplies, with each layer's
        input-side projections folded into ONE packed matrix.

        The packing is this lineage's, unchanged: `c_q`, `c_k` and `c_v` are three bias-free
        `n_embd -> *_head * head_dim` linears applied to the SAME `h`, so `[Wq; Wk; Wv]`
        stacked on the output dimension computes all three in one call and the three results
        are slices of one output. `ve_gate` is a fourth linear on the same vector, applied to
        `h[..., :ve_gate_channels]`; because it is linear, the same weight zero-extended to
        `n_embd` columns computes exactly `ve_gate(h[..., :ve_gate_channels])`, so it folds
        into the same matrix. The step still issues ONE projection per layer instead of three
        or four, over the same weight bytes to within the extension's zeros.

        What is new is the DTYPE of what the step multiplies. Every decode call runs inside
        prepare.py's autocast bfloat16 context, so each matmul receives a bfloat16 operand
        either way. The only question is whether those bytes already exist or are produced
        from the float32 master on the call. Holding them here answers it by construction:
        `tensor.to(torch.bfloat16)` is the conversion autocast itself performs -- round to
        nearest, ties to even -- so every matmul receives exactly the operands it received
        before, and the per-call conversion of the same bytes goes away. This matters most
        inside the traced and captured step, where autocast's cast cache is dead and the trace
        re-derives one conversion per matmul weight on every call.

        Which weights: per layer the packed input projection, the attention output projection
        and both MLP matrices, plus one copy of the unembedding. `wte` and the value
        embeddings are absent deliberately -- they are already bfloat16, and they are gathers,
        not matmuls. So are `resid_lambdas` and `x0_lambdas`, which autocast does not convert.

        `ve_gate` is folded by placing its rows in a FRESH zero tensor of full width and
        copying its weight into the leading columns -- never by reshaping or padding
        `ve_gate.weight` itself, which `GPT.forward` still multiplies and from which `val_bpb`
        is read. The extension columns are exactly zero and the layer is bias free, so the
        wider product is the same sum with exact zeros added.

        Built ONCE per model, lazily, on the first decode call -- never at construction and
        never in `init_decode_state`:

          - from the FINAL weights, so nothing here is stale with respect to the last
            optimizer step, and `forward` keeps using the modules themselves. The first decode
            call happens inside `report_efficiency_metrics`, after the last optimizer step;
            nothing the training loop reaches calls this;
          - held in `self.__dict__` as plain tensors, not parameters, buffers or submodule
            attributes, so `parameters()`, the census and `state_dict` cannot see them and
            `num_params_total` cannot move;
          - once per MODEL and not once per state -- the same place this file already keeps
            the compiled step -- so the copies are not part of any state's allocation, a
            state still owns every buffer a request needs, and a second request reuses them.

        No `requires_grad_(True)`, unlike the float32 packed matrices this replaces. That flag
        was never about gradients: it was what kept autocast's weight-cast cache alive for an
        fp32 leaf. A bfloat16 tensor is not cast at all, so there is nothing to cache and the
        flag would buy nothing. The float32 packed list is gone with it, so no duplicate copy
        of the packed matrices is held.
        """
        cache = self.__dict__.get("_serving_weight_copies")
        if cache is not None:
            return cache
        layers = []
        with torch.no_grad():
            for block in self.transformer.h:
                attn, mlp = block.attn, block.mlp
                # The q block is read back with the SAME head count as k and v, which is the
                # one layout that keeps a strided read a view rather than a copy.
                assert attn.n_head == attn.n_kv_head, (
                    "the packed projection reads q, k and v as one three-way split, which "
                    f"needs n_head == n_kv_head, got {attn.n_head} != {attn.n_kv_head}")
                parts = [attn.c_q.weight.detach(), attn.c_k.weight.detach(),
                         attn.c_v.weight.detach()]
                if attn.ve_gate is not None:
                    # A zero tensor with the gate's columns copied in -- NOT a reshape or an
                    # in-place pad of `ve_gate.weight`, which `forward` still uses and which
                    # the gated quality metric is read from.
                    gate_w = attn.ve_gate.weight.detach()
                    gate = torch.zeros(gate_w.size(0), attn.n_embd,
                                       dtype=gate_w.dtype, device=gate_w.device)
                    gate[:, :attn.ve_gate_channels] = gate_w
                    parts.append(gate)
                layers.append({
                    "packed": torch.cat(parts, dim=0).detach().to(torch.bfloat16),
                    "attn_c_proj": attn.c_proj.weight.detach().to(torch.bfloat16),
                    "c_fc": mlp.c_fc.weight.detach().to(torch.bfloat16),
                    "mlp_c_proj": mlp.c_proj.weight.detach().to(torch.bfloat16),
                })
            cache = (layers, self.lm_head.weight.detach().to(torch.bfloat16))
        self.__dict__["_serving_weight_copies"] = cache
        return cache

    def _decode_body(self, idx, state, prefill):
        """`forward`'s body with the cache spliced in. `prefill` is a Python bool that is
        constant per variant, so the two branches are two graphs, not one per position.

        The width-1 arm now lives in `_decode_body_step`, unchanged, so that one region can be
        handed to `torch.compile`. `prefill` still selects between the two, and they are still
        two variants.

        THE LAST BLOCK'S TAIL IS COMPUTED ON THE FINAL ROW ONLY. Nothing in this protocol reads
        the last block's output at any other position: the loop ends, and the only statement
        after it that touches `x` is `x = norm(x[:, -1:, :])` feeding one head product. Every
        EARLIER layer's full-width output is live, because the next layer's cache write needs its
        keys and values; the last layer's is not, because no layer follows it. So for that layer
        this arm keeps the residual mix, the norm, the packed projection, the value-embedding
        gate, the full-width `v`, the full-width rotary and norm on `k` and the cache writes --
        all of which the request needs -- and computes the rotary and norm on ONE query row, a
        one-query attention call over the same keys, a one-row output projection and a one-row
        MLP.

        The trim fires only under a precondition checked here rather than assumed: the layer must
        not be a ring (so the keys live in `kc[:, :Tn]` exactly as the parent wrote them), and
        its left window must cover the whole prefill (`window_left < 0 or Tn <= window_left + 1`),
        so that at the final query position no mask can exclude a key and the attendable set is
        all `Tn` of them. That is why the one-query call is made with `causal=False` and
        `window_size=(-1, -1)`: the mask is dropped because it excludes nothing under the
        precondition, which avoids depending on any query-alignment convention. Where the
        precondition does not hold, the parent's full-width path runs and the diagnostic says so.
        `_decode_body_full_tail_prefill` preserves that full-width tail as its own method for the
        in-launch comparator below; nothing on the scored path calls it."""
        if not prefill:
            return self._decode_body_step(idx, state)
        B, Tn = idx.size()
        cos, sin = self.cos[:, :Tn], self.sin[:, :Tn]

        # Fetched once, not per layer: the loop reads tensors rather than module attributes.
        sw_layers, sw_lm_head = self._serving_weights()

        x = norm(self.transformer.wte(idx))
        x0 = x
        last = len(self.transformer.h) - 1
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            attn = block.attn
            w = sw_layers[i]
            # The last block's tail, and ONLY the last block's: `trim` is a Python bool read
            # from the state's own per-layer ring flag and from this layer's window, both fixed
            # for the life of the state, so it selects a branch and is never a traced value.
            ring = state["is_ring"][i]
            window_left = self.window_sizes[i][0]
            trim = (i == last and not ring
                    and (window_left < 0 or Tn <= window_left + 1))
            if i == last:
                self._report_prefill_tail(i, state, Tn, ring, window_left, trim)
            h = norm(x)
            # ONE projection call for q, k, v and the gate. The three head blocks are read
            # back as one three-way split of the last dimension, which is a VIEW at either
            # width: the split dims are contiguous within the block, and the token stride
            # simply stays the packed row length. No `.reshape` and no copy -- at the
            # prefill width a repack would move more bytes than the removed calls cost.
            out = F.linear(h, w["packed"])
            kv_width = attn.n_kv_head * attn.head_dim
            qkv = out[..., :3 * kv_width].view(B, Tn, 3, attn.n_kv_head, attn.head_dim)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            if ve is not None:
                ve = ve.view(B, Tn, attn.n_kv_head, attn.head_dim)
                # The gate's rows are the trailing block of the same output: the zero
                # extension makes this exactly `ve_gate(h[..., :ve_gate_channels])`.
                gate = 2 * torch.sigmoid(out[..., 3 * kv_width:3 * kv_width + attn.n_kv_head])
                v = v + gate.unsqueeze(-1) * ve
            # k keeps its full width: every prefilled key is written to this layer's cache and
            # is read by the step. Only the QUERY narrows, to the one row whose output is read,
            # and it takes that position's own rotary table rows -- `apply_rotary_emb` is
            # elementwise in the position and `norm` reduces within a row, so this is the row
            # the full-width pair would have produced.
            if trim:
                q_rot, cos_q, sin_q = q[:, -1:], cos[:, -1:], sin[:, -1:]
            else:
                q_rot, cos_q, sin_q = q, cos, sin
            q, k = apply_rotary_emb(q_rot, cos_q, sin_q), apply_rotary_emb(k, cos, sin)
            q, k = norm(q), norm(k)

            kc, vc = state["kc"][i], state["vc"][i]
            length = state["cache_lengths"][i]
            if ring:
                # Attend over the freshly computed local k and v. That is the same
                # computation as the reference's kc[:, :Tn]/vc[:, :Tn], which is written
                # from these very tensors immediately before it is read.
                y = fa3.flash_attn_func(q, k, v, causal=True,
                                        window_size=self.window_sizes[i])
                # Then write the prefilled keys into the ring modulo its length, which
                # leaves it holding the last `length` positions in ring order. Tn is
                # constant per variant, so the slice count is too.
                position = 0
                while position < Tn:
                    slot = position % length
                    n = min(length - slot, Tn - position)
                    kc[:, slot:slot + n] = k[:, position:position + n]
                    vc[:, slot:slot + n] = v[:, position:position + n]
                    position += n
            else:
                kc[:, :Tn] = k
                vc[:, :Tn] = v
                if trim:
                    # One query at the final position over the same Tn keys. `causal=False`
                    # and `window_size=(-1, -1)` because under this branch's precondition --
                    # not a ring, and Tn <= window_left + 1 -- neither the causal mask nor
                    # the window can exclude any of those keys at that position, so dropping
                    # both keeps the attended set and depends on no alignment convention.
                    y = fa3.flash_attn_func(q, kc[:, :Tn], vc[:, :Tn], causal=False,
                                            window_size=(-1, -1))
                else:
                    y = fa3.flash_attn_func(q, kc[:, :Tn], vc[:, :Tn], causal=True,
                                            window_size=self.window_sizes[i])
            if trim:
                # The residual is sliced to the same one row before the add, so the output
                # projection and both MLP statements below run on one row instead of Tn.
                x = x[:, -1:, :] + F.linear(y.contiguous().view(B, 1, -1), w["attn_c_proj"])
            else:
                x = x + F.linear(y.contiguous().view(B, Tn, -1), w["attn_c_proj"])
            # `MLP.forward`'s own two statements, inlined against the copies: the same ops in
            # the same order, reading bytes that already exist.
            m = F.linear(norm(x), w["c_fc"])
            x = x + F.linear(F.relu(m).square(), w["mlp_c_proj"])

        # The ring positions, after every layer has read them: one slot and one count per
        # distinct length, so this cannot run inside the loop without a second ring layer
        # reading an already-advanced count.
        for length, index in state["ring_write"].items():
            index.fill_(Tn % length)
        for length, count in state["ring_count"].items():
            count.fill_(min(Tn + 1, length))

        # `x` is already one row when the tail was trimmed, and this slice is then a no-op;
        # with the parent's path it is the slice the parent took.
        x = norm(x[:, -1:, :])
        softcap = 15
        logits = F.linear(x, sw_lm_head).float()
        return softcap * torch.tanh(logits / softcap)

    def _report_prefill_tail(self, i, state, Tn, ring, window_left, trim):
        """One free line per distinct (max_len, Tn), the first time the last layer is reached.

        Host side only: it states the PREMISE the branch above was taken on -- the layer, the
        prefill width, that layer's `window_sizes` entry, whether the state made it a ring, how
        many keys the call attends and how wide the query is -- and then which path ran, so a
        fallback is visible rather than silent. It allocates no tensor; the table it dedupes on
        is a Python set held on the model, so the cache probe's before/after bracket has nothing
        to charge. The two shapes that reach it are (2048, 1536), which the cache probe builds
        untimed, and (2049, 1536), whose first occurrence is the request probe's FIRST WARMUP
        pass -- a pass whose timing is discarded -- so no line is printed during any of the 30
        passes whose median is reported. Not a METRICS_JSON line.
        """
        seen = self.__dict__.setdefault("_prefill_tail_seen", set())
        key = (state["max_len"], Tn)
        if key in seen:
            return
        seen.add(key)
        keys_attended = Tn if window_left < 0 else min(Tn, window_left + 1)
        print(f"[prefill-tail] layer={i} max_len={state['max_len']} Tn={Tn} "
              f"window_sizes_entry={tuple(self.window_sizes[i])} is_ring={bool(ring)} "
              f"keys_attended={keys_attended} query_width={1 if trim else Tn} "
              f"path={'trimmed_last_row' if trim else 'parent_full_width_tail'} "
              f"premise=not a ring and (window_left < 0 or Tn <= window_left + 1)")

    def _decode_body_full_tail_prefill(self, idx, state):
        """The PARENT's prefill arm, verbatim, kept for the in-launch comparator below.

        It exists so that one launch can time the full-width last-block tail against the
        trimmed one on the same device, with the same weights and the same state. Nothing on
        the scored path calls it: `decode_step` routes width > 1 to `_decode_body`, and this
        method's only caller is the post-report `[prefill-compare]` block. It adds one method
        and no tensor.
        """
        B, Tn = idx.size()
        cos, sin = self.cos[:, :Tn], self.sin[:, :Tn]

        sw_layers, sw_lm_head = self._serving_weights()

        x = norm(self.transformer.wte(idx))
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            attn = block.attn
            w = sw_layers[i]
            h = norm(x)
            out = F.linear(h, w["packed"])
            kv_width = attn.n_kv_head * attn.head_dim
            qkv = out[..., :3 * kv_width].view(B, Tn, 3, attn.n_kv_head, attn.head_dim)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            if ve is not None:
                ve = ve.view(B, Tn, attn.n_kv_head, attn.head_dim)
                gate = 2 * torch.sigmoid(out[..., 3 * kv_width:3 * kv_width + attn.n_kv_head])
                v = v + gate.unsqueeze(-1) * ve
            q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
            q, k = norm(q), norm(k)

            kc, vc = state["kc"][i], state["vc"][i]
            length = state["cache_lengths"][i]
            ring = state["is_ring"][i]
            if ring:
                y = fa3.flash_attn_func(q, k, v, causal=True,
                                        window_size=self.window_sizes[i])
                position = 0
                while position < Tn:
                    slot = position % length
                    n = min(length - slot, Tn - position)
                    kc[:, slot:slot + n] = k[:, position:position + n]
                    vc[:, slot:slot + n] = v[:, position:position + n]
                    position += n
            else:
                kc[:, :Tn] = k
                vc[:, :Tn] = v
                y = fa3.flash_attn_func(q, kc[:, :Tn], vc[:, :Tn], causal=True,
                                        window_size=self.window_sizes[i])
            x = x + F.linear(y.contiguous().view(B, Tn, -1), w["attn_c_proj"])
            m = F.linear(norm(x), w["c_fc"])
            x = x + F.linear(F.relu(m).square(), w["mlp_c_proj"])

        for length, index in state["ring_write"].items():
            index.fill_(Tn % length)
        for length, count in state["ring_count"].items():
            count.fill_(min(Tn + 1, length))

        x = norm(x[:, -1:, :])
        softcap = 15
        logits = F.linear(x, sw_lm_head).float()
        return softcap * torch.tanh(logits / softcap)

    def _decode_body_step(self, idx, state):
        """One cached width-1 step: the `prefill=False` arm of `_decode_body`, unchanged.

        Factored out so the whole step is a single region for `torch.compile`. Eager, this is
        several hundred aten kernels per step, almost all of them elementwise, and each one
        re-reads the activation the previous one wrote. Compiled, the same arithmetic lowers
        into generated kernels that keep the row in registers between operations; the matmuls
        stay extern and the kvcache call stays an opaque op, so the parts that are already one
        kernel each are untouched.

        This lineage's step carries two things the reference's does not, and both are inside
        the region: a ring layer's `index_copy_` append, and the per-length slot and count
        advances at the end. They are in-place device ops on tensors the region receives as
        inputs, so they must survive tracing as mutations of those very tensors -- a
        functionalised copy would leave later steps attending stale slots, which is what the
        decode-agreement probe bounds.
        """
        B, Tn = idx.size()
        seq_idx = state["seq"].to(torch.int64)
        cos = self.cos.index_select(1, seq_idx)
        sin = self.sin.index_select(1, seq_idx)

        # Fetched once, not per layer: the loop reads tensors rather than module attributes.
        sw_layers, sw_lm_head = self._serving_weights()

        x = norm(self.transformer.wte(idx))
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            attn = block.attn
            w = sw_layers[i]
            h = norm(x)
            # ONE projection call for q, k, v and the gate. The three head blocks are read
            # back as one three-way split of the last dimension, which is a VIEW at either
            # width: the split dims are contiguous within the block, and the token stride
            # simply stays the packed row length. No `.reshape` and no copy -- at the
            # prefill width a repack would move more bytes than the removed calls cost.
            out = F.linear(h, w["packed"])
            kv_width = attn.n_kv_head * attn.head_dim
            qkv = out[..., :3 * kv_width].view(B, Tn, 3, attn.n_kv_head, attn.head_dim)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
            if ve is not None:
                ve = ve.view(B, Tn, attn.n_kv_head, attn.head_dim)
                # The gate's rows are the trailing block of the same output: the zero
                # extension makes this exactly `ve_gate(h[..., :ve_gate_channels])`.
                gate = 2 * torch.sigmoid(out[..., 3 * kv_width:3 * kv_width + attn.n_kv_head])
                v = v + gate.unsqueeze(-1) * ve
            q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
            q, k = norm(q), norm(k)

            kc, vc = state["kc"][i], state["vc"][i]
            length = state["cache_lengths"][i]
            ring = state["is_ring"][i]
            if ring:
                # The kernel appends at cache_seqlens; a ring appends at seq mod length,
                # which is a different index, so the append is done here and k and v are
                # withheld from the call. `.to` is a no-op at equal dtype, which it is --
                # the reference's own call requires k to match the cache.
                index = state["ring_write"][length]
                count = state["ring_count"][length]
                kc.index_copy_(1, index, k.to(kc.dtype))
                vc.index_copy_(1, index, v.to(vc.dtype))
                # Every key in the buffer is a legitimate attendee, so there is no window to
                # mask and no context length for the key loop to grow with: for a width-1
                # query the attendable set is the last min(t + 1, length) positions, which is
                # exactly the ring's live content, wrapped or not. This is correct ONLY at
                # width 1, which is why it lives in the width-1 arm and not behind a flag.
                # num_splits as below: `cache_seqlens` bounds the SET of keys read and the
                # split count only partitions that set, so splitting changes the order of the
                # reduction over the ring's live content and nothing about its membership.
                y = fa3.flash_attn_with_kvcache(q, kc, vc,
                                                cache_seqlens=count, causal=False,
                                                window_size=(-1, -1),
                                                num_splits=state["key_splits"])
            else:
                # Appends k,v at cache_seqlens and attends over the whole preallocated
                # cache, so there is no `kc[:, start:end]` slice whose bounds depend on the
                # step number. What the op's fake kernel requires is only that the split
                # count be POSITIVE -- it refuses to trace at the default num_splits=0, the
                # heuristic, and that is what makes an unpinned cache uncompilable -- so any
                # pinned count above zero is traceable, and above one it splits the key
                # range. The count comes from the state, derived once at init from the device
                # and this request's shape, so it is a tracing constant per state exactly as
                # the cache length already is. It is a reduction-order choice, and the
                # agreement check in prepare.py still has to pass with the range split.
                y = fa3.flash_attn_with_kvcache(q, kc, vc, k=k, v=v,
                                                cache_seqlens=state["seq"], causal=True,
                                                window_size=self.window_sizes[i],
                                                num_splits=state["key_splits"])
            x = x + F.linear(y.contiguous().view(B, Tn, -1), w["attn_c_proj"])
            # `MLP.forward`'s own two statements, inlined against the copies: the same ops in
            # the same order, reading bytes that already exist.
            m = F.linear(norm(x), w["c_fc"])
            x = x + F.linear(F.relu(m).square(), w["mlp_c_proj"])

        # The ring positions, after every layer has read them: one slot and one count per
        # distinct length, so this cannot run inside the loop without a second ring layer
        # reading an already-advanced count.
        # Device ops with constant scalars, placed after the reads, so a capture replays
        # append, attend, advance with no host round trip -- the discipline `seq.add_(1)`
        # already follows.
        for length, index in state["ring_write"].items():
            index.add_(1).remainder_(length)
        for length, count in state["ring_count"].items():
            count.add_(1).clamp_max_(length)

        x = norm(x[:, -1:, :])
        softcap = 15
        logits = F.linear(x, sw_lm_head).float()
        return softcap * torch.tanh(logits / softcap)

    def _decode_step_callable(self):
        """The compiled width-1 step, built once per model and held on the model.

        Held on the model and not on a state, so it is shared by every request and is not part
        of what `init_decode_state` hands out: a state still owns everything a request needs
        and dropping it still frees it.

        `dynamic=False` keeps the cache lengths static, so each set of lengths the probes build
        is its own specialisation rather than a dynamic guard; the ring branch is taken per
        layer on a Python bool read from the state, so the two request shapes specialise on
        their per-layer branches as well as on their lengths. `fullgraph=False` on purpose: an
        op Dynamo cannot trace -- the ring's `index_copy_`, the in-place slot and count
        advances, or the opaque kvcache call -- must become a graph break, not an exception
        that would end the launch. No `mode="reduce-overhead"` and no inductor cudagraphs --
        that copies graph inputs into a static pool while `flash_attn_with_kvcache` appends k/v
        in place inside an opaque op, so the append would land in the copy; see
        `_GraphedDecodeStep`. If compiling raises, the uncompiled method is used and the step
        runs as it did before.

        `coordinate_descent_tuning` is passed for THIS compilation only. It makes inductor
        benchmark perturbations of each GENERATED kernel's tile size and warp count -- the
        norms, the rotary chains, the residual mixes, the gathers and the head's softcap --
        instead of taking the configuration its heuristic picks. It removes no operation and
        adds none: the source-level census of this region is unchanged, and only the
        configuration each generated kernel is compiled with moves.

        `max_autotune_pointwise` is added HERE, and it is this candidate's only change to what
        is compiled. It moves which candidate configurations EXIST for a generated kernel, not
        how the walk above moves once it has a starting point. On the pinned torch:

          * `runtime/triton_heuristics.py::_reduction_configs` returns its heuristic list plus a
            SINGLE hint config -- `contiguous_config`, `outer_config` or `tiny_config` -- and
            returns early, whenever the reduction's hint is INNER, OUTER or OUTER_TINY; it
            reaches the wide benchmarked tail (contiguous, outer, tiny, 64x64, 8x512 and more)
            only when `max_autotune` OR `max_autotune_pointwise` is set.
          * `pointwise()` checks the same pair at one and at two size hints. At one hint it is
            already wide here, because `disable_pointwise_autotuning` is `not
            autotune_pointwise` and `triton.autotune_pointwise` defaults True; at two hints the
            single forced config applies when the tile hint is SQUARE, and that is the case this
            option opens.
          * `CachingAutotuner.coordinate_descent_tuning`'s own docstring: the only difference
            max-autotune makes to the descent is which config it starts from. So wherever the
            list widens, the parent's search also starts from a benchmarked winner instead of
            from a heuristic pick.

        What it does NOT reach, recorded here because it bounds the mechanism rather than the
        edit: `_persistent_reduction_configs` consults neither key, so a reduction small enough
        to be made persistent -- `choices.py` uses a reduction-numel threshold of 1024 at an
        INNER hint and 64 otherwise -- keeps exactly the config list it had. The scored step is
        batch 1 and width 1, so a persistent reduction's non-reduction extent is one and that
        list is one config either way; `tunable_fields` drops R0_BLOCK for it and
        `value_too_large` caps XBLOCK at the size hint, which is 1, leaving `num_warps` as the
        one field the descent can move there, with or without this option.

        Both are passed as `options=` and not as `mode=`: torch raises if both are given, and
        every mode string that turns these on also turns on `max_autotune` or inductor
        cudagraphs -- `mode="max-autotune"` maps to
        `{'max_autotune': True, 'triton.cudagraphs': True, 'coordinate_descent_tuning': True}`
        on the pinned torch -- and cudagraphs is exactly what the paragraph above forbids.
        `max_autotune` and `max_autotune_gemm` are deliberately NOT set: the matmuls are extern
        calls and how they are lowered is a different question from this one, bought separately.
        The global `torch._inductor.config` is not touched, so the training compile of `model`
        and prepare.py's own evaluation graphs are compiled exactly as this parent compiled them.

        THIS CANDIDATE'S EDIT: `coordinate_descent_check_all_directions` and
        `coordinate_descent_search_radius = 2` join the same dict, and nothing else moves. On the
        pinned torch both keys are inert unless they are named here, and both are read only from
        `inductor_meta`:

          * `codegen/triton.py::TritonKernel.inductor_meta_common` copies
            `coordinate_descent_search_radius` and `coordinate_descent_check_all_directions` into
            a generated kernel's `inductor_meta` ONLY inside `if config.coordinate_descent_tuning:`
            -- so the parent, which set that option but neither of these, generated kernels whose
            `inductor_meta` carried the defaults: radius 1 and all-directions False.
          * `runtime/coordinate_descent_tuner.py::CoordescTuner.autotune`'s main loop calls
            `self.get_neighbour_values(name, cur_val)` at the signature default, i.e. radius 1 --
            one doubling and one halving of ONE field at a time. The radius is read in exactly one
            place, `check_all_tuning_directions`, as
            `self.inductor_meta.get("coordinate_descent_search_radius", 1)`, and the descent
            reaches that method only inside
            `if not improved and self.inductor_meta.get("coordinate_descent_check_all_directions")`.
            So without the flag the radius cannot be read at all, and the two keys are one
            mechanism on this torch, not two.

        What the pair changes is therefore the SHAPE OF THE SEARCH and nothing else: with the flag
        set, once the one-field-at-a-time walk stops improving, `check_all_tuning_directions` builds
        `itertools.product` over each tunable field's neighbour list and tries the whole
        neighbourhood at once; with radius 2 each of those lists is two doublings and two halvings
        plus the current value. `get_neighbour_values` still stops at `value_too_large` and at zero,
        and `compare_config` still accepts a candidate only when it BENCHMARKED faster on this
        device past `has_improvement`'s 0.1% threshold. No dispatched operation, shape, stride,
        dtype or value can move; only the tile size and warp count a generated kernel is compiled
        with can differ.

        The dict is built ONCE as a local and the SAME object is handed to `torch.compile`, to the
        guard's print and to the first-call print, so stdout cannot disagree with what was
        compiled.

        Under `fullgraph=False` a backend failure raises at CALL time, not here, so the
        `try/except` around `torch.compile` cannot catch it and the first call happens inside
        `report_efficiency_metrics`. `dynamic=False` makes each cache length its own backend
        compile, so each one raises on its own first call. The callable is therefore wrapped in
        a guard that is keyed on the cache length and stays installed, which chooses which
        callable runs and nothing else.
        """
        fn = self.__dict__.get("_compiled_decode_step")
        if fn is None:
            # ONE dict, built here, handed to `torch.compile` and to every print that names it.
            options = {"coordinate_descent_tuning": True,
                       "max_autotune_pointwise": True,
                       "coordinate_descent_check_all_directions": True,
                       "coordinate_descent_search_radius": 2}
            # Free diagnostic, host side, at construction only: what was asked for, so that
            # "the options were in effect and the matmul lowering was not touched" is auditable
            # from stdout rather than asserted. RENDERED FROM THE DICT ABOVE and not retyped as a
            # literal, so a future edit to the dict cannot leave this line describing the previous
            # compilation. Not a METRICS_JSON line.
            print("decode step compile options="
                  + ",".join(f"{key}={options[key]}" for key in sorted(options))
                  + " fullgraph=False dynamic=False cudagraphs=False max_autotune=False "
                    "max_autotune_gemm=False")
            try:
                fn = torch.compile(self._decode_body_step, dynamic=False, fullgraph=False,
                                   options=options)
            except Exception as exc:        # noqa: BLE001 -- recorded, then run uncompiled
                print(f"decode step compile failed, running eager: {type(exc).__name__}: {exc}")
                fn = self._decode_body_step
            else:
                fn = self._guard_decode_step_call(fn, options)
            self.__dict__["_compiled_decode_step"] = fn
        return fn

    def _guard_decode_step_call(self, compiled, options=None):
        """Wrap `compiled` so NO cache length's first invocation can end the launch.

        Compilation of the region is deferred to call time by `fullgraph=False`, and
        `torch._dynamo.config.suppress_errors` is False on the pinned torch, so a codegen or
        benchmarking failure raises out of the call that triggers it. Those calls are made from
        inside `report_efficiency_metrics` -- the first `measure_kv_cache_bytes`'s throwaway
        `build_state`, then each further cache length the probes build -- which is after the
        GPU-work witness has printed and before any metric is printed, so an unguarded failure
        would spend the launch and print no METRICS_JSON line at all.

        The wrapper STAYS INSTALLED and is keyed on `state["max_len"]`, a plain Python int the
        state already owns. `dynamic=False` makes each cache length its own specialisation, so
        each one compiles on its own first call and each one can raise on its own; a guard that
        retired after its own first call would cover one of the four lengths the reporter
        builds. Per key: 'ok' means call the compiled callable, 'failed' means call the
        uncompiled method, absent means try the compiled one once inside `try/except` and record
        which. The timing and the print happen only on that first call for a given `max_len`.

        A permanent wrapper costs nothing that is measured. The scored request probes build
        graph-enabled states, so their timed steps are `_GraphedDecodeStep.replay()` calls, and
        `replay` copies into `static_idx` and calls `graph.replay()` -- it never reaches
        `_decode_step_callable`. This wrapper's callers are `measure_kv_cache_bytes`'s
        `graph=False` states, the three capture warm-ups and the capture itself, none of which is
        timed. It allocates no tensor, never synchronizes, and a dict of Python ints holds no
        device bytes for the cache probe's before/after bracket to charge. Both branches call the
        same `_decode_body_step`, so the eager and captured paths still compute the same function
        of the same data and `graph` still chooses only which one runs. On a failure
        `state["seq"]` is advanced by `decode_step` after this returns, so the re-run writes the
        same values into the same cache slot the failed attempt would have.
        """
        # A FRESH table PER INSTALLATION, published on the model for readability -- never
        # fetched with `setdefault` from a table that outlives an installation. The wrapper is
        # installed once per `_compiled_decode_step`, and the inherited `[body-weights]` probe
        # POPS that key twice per case, so each case installs a new wrapper over a new
        # compilation. A table carried over from the previous installation would already read
        # 'ok' for that max_len, so the new compilation's first call would be made with no
        # try/except, would not be timed and would not be printed -- which is what left that
        # probe's readings unrecorded. A fresh dict makes each installation report its own
        # first call at each cache length, and nothing else about that probe is touched.
        seen = {}
        self.__dict__["_decode_step_guard_state"] = seen
        # Read from the dict that was actually passed to `torch.compile`, never retyped: the
        # first-call line below is this launch's only per-specialisation compile price, and the
        # searched neighbourhood is what that price is being paid for. `options=None` is the
        # uncompiled/legacy path and reports the two keys as absent.
        opts = options or {}
        radius = opts.get("coordinate_descent_search_radius", "absent")
        all_dirs = opts.get("coordinate_descent_check_all_directions", "absent")

        def guarded(idx, state):
            max_len = state["max_len"]
            status = seen.get(max_len)
            if status == "ok":
                return compiled(idx, state)
            if status == "failed":
                return self._decode_body_step(idx, state)
            t0 = time.time()
            try:
                out = compiled(idx, state)
            except Exception as exc:        # noqa: BLE001 -- recorded, then run uncompiled
                seen[max_len] = "failed"
                print(f"decode step compile failed at max_len={max_len}, running uncompiled: "
                      f"{type(exc).__name__}: {exc}")
                return self._decode_body_step(idx, state)
            seen[max_len] = "ok"
            # Host side, outside every timed region: this specialisation's benchmarking and
            # tuning wall cost against the launch's own clock, and the number that says whether
            # the mechanism fits inside the launch timeout. Not a METRICS_JSON line.
            print(f"decode step first compiled call max_len={max_len} "
                  f"seconds={time.time() - t0:.3f} "
                  f"coordinate_descent_search_radius={radius} "
                  f"coordinate_descent_check_all_directions={all_dirs}")
            return out
        return guarded

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
            logits = self._decode_step_callable()(idx, state)
            state["seq"].add_(1)
            return logits, state
        # The graph lives ON the state it was captured against, so its lifetime is exactly
        # that state's and there is no key to get wrong. A state whose capture failed holds
        # an uncaptured object and runs eager forever, which is slow and correct.
        graph = state.get("graph")
        if graph is None:
            graph = state["graph"] = _GraphedDecodeStep(self, state)
        if not graph.captured:
            logits = self._decode_step_callable()(idx, state)
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
      3. Warmup and capture execute their kernels, so EVERY position tensor -- `seq` and each
         ring's write slot and live count -- is saved and restored around both; junk above
         `seq`, or above a ring's count, is unreachable to `cache_seqlens`. Missing one would
         leak the capture's own executed increment into the first request, and its slot would
         be off by the number of warmups.

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
            # Printed on this path too, so the line exists for EVERY graph-enabled state
            # rather than only for the ones that reach a capture attempt.
            print(f"decode step graph captured={self.captured} reason={self.reason!r}")
            return
        try:
            self._capture()
            self.captured = True
        except Exception as exc:            # noqa: BLE001 -- recorded, then paid for in ms
            self.reason = f"{type(exc).__name__}: {exc}"
            self.graph, self.static_logits = None, None
        # One line per state, printed while the graph is being constructed -- which happens
        # inside the first warmup pass and therefore outside every timed interval. It records
        # whether the scored steps run as a graph replay over the compiled region or as a
        # per-step compiled call, which nothing in the metrics reports, and whether a hand
        # capture succeeds over a ring's in-place append. Not a metrics line of any kind.
        print(f"decode step graph captured={self.captured} reason={self.reason!r}")

    def _advance(self):
        # The same compiled callable the eager route runs, so warmup, capture and the
        # graph=False path are one computation and `graph` still chooses only which one runs.
        # The three warmup replays below are what leave it fully compiled before capture.
        logits = self.model._decode_step_callable()(self.static_idx, self.state)
        self.state["seq"].add_(1)
        return logits

    def _positions(self):
        """Every tensor a replay advances: the position, and each ring's slot and count."""
        state = self.state
        return ([state["seq"]]
                + list(state.get("ring_write", {}).values())
                + list(state.get("ring_count", {}).values()))

    def _capture(self):
        saved = [tensor.clone() for tensor in self._positions()]

        def restore():
            for tensor, value in zip(self._positions(), saved):
                tensor.copy_(value)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(self.WARMUP_REPLAYS):
                restore()
                self._advance()
        torch.cuda.current_stream().wait_stream(stream)

        restore()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_logits = self._advance()
        restore()                              # the capture itself executed one increment

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


# ---------------------------------------------------------------------------
# FREE INSTRUMENT: did this launch SEARCH, or did it read a stored answer?
# ---------------------------------------------------------------------------
# This candidate's whole edit is the shape of inductor's coordinate-descent search. A coordinate
# descent that never ran is a null by construction, and until now nothing in this run could tell
# the two apart: a searched result is written to a local `.best_config` with
# `found_by_coordesc=True` (`runtime/autotune_cache.py::AutotuneCache.save`), and on a later
# compile of the same generated source `_load_cached_autotuning` returns that config with
# `found_by_coordesc` set, after which `CachingAutotuner.run` skips
# `coordinate_descent_tuning` entirely -- `runtime/triton_heuristics.py` guards that call with
# `if not getattr(self.launchers[0].config, "found_by_coordesc", False) and
# self.inductor_meta.get("coordinate_descent_tuning", False)`. So a launch can carry the option,
# print its compile seconds, and still have searched nothing.
#
# WHAT IT READS, and only reads. It walks `torch._inductor.codecache.PyCodeCache.modules`, then
# `vars(module).values()`, and for every object whose class name contains `CachingAutotuner` -- the
# object `async_compile.triton` returns per generated Triton kernel -- it reads three ALREADY
# EXISTING attributes and nothing else:
#
#   * `autotune_cache_info`, set in `CachingAutotuner.__init__` and again in `precompile` from
#     `check_autotune_cache`. On the pinned torch that function writes exactly one of
#     'hit', 'miss', 'only 1 config' or 'force_disabled' into `autotune_cache_state`, and writes
#     `coordesc_tuning=True` on a miss when `coordinate_descent_tuning` is in `inductor_meta`.
#   * `inductor_meta.get('kernel_name')`, with a fallback, so the count of named kernels is
#     readable without depending on the key being present.
#   * `launchers`, and only if it is ALREADY a populated list, `launcher.config.found_by_coordesc`
#     -- a plain attribute on the launcher's config, not a property.
#
# No property and no method is called, so nothing here can trigger a compile, an autotune or a
# kernel launch. One try/except per module, and an unreadable cache prints -1 rather than raising.
# It allocates no device memory, so the cache probe -- which closed long before this point -- has
# nothing to charge.
#
# HOW TO READ IT: the count is process-wide. It includes kernels from the training compile, the
# optimizer's two, prepare.py's evaluation graphs and every decode specialisation, so only a
# DELTA between two tags says anything about the work between them. `states=` distinguishes a
# launch that searched (`miss` with `coordesc_tuning=True`) from one that read a stored answer
# (`hit`, and `found_by_coordesc` already true on the launcher), and it fires either way -- which
# is what no witness this run has built could do.
def _autotune_cache_witness(tag):
    """Counts, by autotune-cache state, of the generated Triton kernels this process can see."""
    try:
        from torch._inductor.codecache import PyCodeCache
        _ac_modules = list(PyCodeCache.modules)
    except Exception as exc:    # noqa: BLE001 -- a private cache; -1 records that it is absent
        print(f"[autotune-cache] tag={tag} modules=-1 kernels=-1 "
              f"unreadable={type(exc).__name__}")
        return
    _ac_states = {}
    _ac_kernels = 0
    _ac_named = 0
    _ac_coordesc_meta = 0
    _ac_launchers_populated = 0
    _ac_found_by_coordesc = 0
    for _ac_module in _ac_modules:
        try:
            for _ac_obj in list(vars(_ac_module).values()):
                if "CachingAutotuner" not in type(_ac_obj).__name__:
                    continue
                _ac_kernels += 1
                _ac_info = getattr(_ac_obj, "autotune_cache_info", None)
                _ac_state = "absent"
                if isinstance(_ac_info, dict):
                    _ac_state = str(_ac_info.get("autotune_cache_state", "absent"))
                    if _ac_info.get("coordesc_tuning") is True:
                        _ac_coordesc_meta += 1
                _ac_states[_ac_state] = _ac_states.get(_ac_state, 0) + 1
                _ac_meta = getattr(_ac_obj, "inductor_meta", None)
                _ac_name = (_ac_meta.get("kernel_name", "unnamed")
                            if isinstance(_ac_meta, dict) else "unnamed")
                if _ac_name != "unnamed":
                    _ac_named += 1
                _ac_launchers = getattr(_ac_obj, "launchers", None)
                if isinstance(_ac_launchers, list) and _ac_launchers:
                    _ac_launchers_populated += 1
                    if any(getattr(getattr(_ac_launcher, "config", None),
                                   "found_by_coordesc", False)
                           for _ac_launcher in _ac_launchers):
                        _ac_found_by_coordesc += 1
        except Exception:       # noqa: BLE001 -- one module may not cost the walk
            continue
    print(f"[autotune-cache] tag={tag} modules={len(_ac_modules)} kernels={_ac_kernels} "
          f"states={{{', '.join(f'{k}={_ac_states[k]}' for k in sorted(_ac_states))}}} "
          f"coordesc_tuning={_ac_coordesc_meta} named_kernels={_ac_named} "
          f"launchers_populated={_ac_launchers_populated} "
          f"found_by_coordesc={_ac_found_by_coordesc} "
          f"elapsed_s={time.time() - t_start:.0f} "
          f"basis=attribute reads only over this process's inductor code cache, no call, "
          f"process-wide count so only a delta between tags is attributable")


try:
    # Once here, host side, from the post-report block and never from inside the guard: at this
    # point the reporter has returned, so the guard has already made its first compiled call at
    # every cache length the reporter's probes built, and this is the reading for those four
    # specialisations before any later block compiles anything of its own.
    _autotune_cache_witness("post-report-guard")
except Exception as exc:        # noqa: BLE001 -- a diagnostic may not fail a launch
    print(f"[autotune-cache] instrument failed: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# FREE INSTRUMENT: this launch's own price for the step it scored
# ---------------------------------------------------------------------------
# Runs ONLY after `report_efficiency_metrics` has returned, so every official number -- val_bpb,
# both cache-byte readings, both request shapes, peak_vram_bytes and peak_vram_bytes_inference --
# has already been read and printed, and the cache probe's before/after bracket closed long
# before this point. It prints no METRICS_JSON line, so the engine still scores the reporter's.
# It never reads the validation split: its tokens are a constant id tensor, the same constant id
# the inherited [body-weights] block below uses, so the two blocks' per-call numbers are directly
# comparable within this one launch.
#
# WHY IT EXISTS: this lineage has never priced its own decode body inside the launch that scored
# it. Every case here REPLAYS the compilation the reporter's own probes already built at that
# cache length -- there is no weight swap, no `_compiled_decode_step` pop and no
# `torch._dynamo.reset()` -- so it cannot become a measurement of a different program and it
# cannot force a recompile. It carries no threshold and decides nothing.
#
# `torch.no_grad()` is mandatory and is why this form and not another: generation 8 measured four
# for four that the same block raises BackendCompilerFailed without it. It is placed HERE, ahead
# of this file's inherited serving-weight diagnostics, because those pop `_compiled_decode_step`
# and swap the weight set, and a calibration that must force no recompile has to run while the
# reporter's own compilations are still installed. Their source is unchanged; they simply start
# at a later wall clock, which their own printed 1500 s guard already reports.
#
# WHAT IT CANNOT SEE, stated rather than left to a reader: one pass per case, not a 30-pass
# median, so it sees neither cross-launch drift nor within-launch scatter -- the reporter's own
# request_ms_iqr and nopref_request_ms_iqr are the spread figures for this launch. It also cannot
# say which generated kernels either compile option changed; nothing in this run can observe a
# compiled region's kernel count.
#
# The `guard=` field is the AUTHORITATIVE per-specialisation compile evidence and `compiled=` is
# not: with a guard that stays installed, a failure at one cache length no longer replaces
# `_compiled_decode_step`, so the `!=` test reads True even for a length that fell back. `guard=`
# reads that length's own recorded status. `elapsed_s=` is the launch's wall clock, printed here
# because a benchmarked candidate list at four specialisations is a wall-clock risk and this is
# the only place the remaining headroom is readable.
#
# One try/except around the whole thing: a diagnostic may not fail a launch.
_STEP_CALIBRATION_CASES = [("real", 514, 1), ("real", 2049, 1536)]

try:
    _cal_inner = getattr(model, "_orig_mod", model)
    for variant, max_len, prefill_width in _STEP_CALIBRATION_CASES:
        state = None
        try:
            with torch.no_grad(), torch.amp.autocast(device_type="cuda",
                                                     dtype=torch.bfloat16):
                state = _cal_inner.init_decode_state(batch=1, max_len=max_len, graph=True)
                # The capture is CONSTRUCTED BEFORE THE TIMER OPENS, so the first timed call is
                # a replay and carries neither the capture nor a compilation.
                state["graph"] = _GraphedDecodeStep(_cal_inner, state)
                captured = bool(state["graph"].captured)
                # `!=` and NOT `is`: attribute access builds a fresh bound method object every
                # time, so an identity test would read True even for the uncompiled fallback.
                compiled = (_cal_inner._decode_step_callable() != _cal_inner._decode_body_step)
                tokens = torch.full((1, prefill_width), 1, dtype=torch.int64, device=device)
                one = torch.full((1, 1), 1, dtype=torch.int64, device=device)
                logits, state = _cal_inner.decode_step(tokens, state)
                del logits
                torch.cuda.synchronize()
                t0 = time.time()
                logits, state = _cal_inner.decode_step(one, state)
                torch.cuda.synchronize()
                first_ms = (time.time() - t0) * 1000.0
                del logits
                # 512 calls as ONE batch: one synchronize at each end, never one per call.
                torch.cuda.synchronize()
                t1 = time.time()
                for _ in range(512):
                    logits, state = _cal_inner.decode_step(one, state)
                torch.cuda.synchronize()
                batch_ms = (time.time() - t1) * 1000.0
                del logits, tokens, one
            guard = _cal_inner.__dict__.get("_decode_step_guard_state", {}).get(max_len,
                                                                               "absent")
            print(f"[step-calibration] variant={variant} max_len={max_len} "
                  f"prefill_width={prefill_width} captured={captured} compiled={compiled} "
                  f"guard={guard} first_call_ms={first_ms:.4f} batch_calls=512 "
                  f"batch_ms={batch_ms:.4f} batch_ms_per_call={batch_ms / 512:.4f} "
                  f"elapsed_s={time.time() - t_start:.0f} "
                  f"basis=one pass, one launch, post-report, capture constructed before the "
                  f"timer, constant tokens, no recompile forced")
        finally:
            if state is not None:
                state.pop("graph", None)
            del state
            gc.collect()
            torch.cuda.empty_cache()
except Exception as exc:        # noqa: BLE001 -- a diagnostic may not fail a launch
    print(f"[step-calibration] instrument failed: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# FREE INSTRUMENT: this launch's own price for the one prefill call, full tail against trimmed
# ---------------------------------------------------------------------------
# This is the card's mechanism priced inside the launch that scored it, and it is the point of
# the card rather than a decoration: the run's working resolution on the scored median is wide
# enough that a scored delta on its own attributes nothing, so the two implementations are timed
# here against each other on the same device, the same weights and the same state.
#
# WHAT IT BRACKETS, exactly. One `_decode_body(tokens, state, prefill=True)` call per reading --
# the prefill call and nothing else: no width-1 step, no capture, no reset inside the bracket, no
# head beyond the one the prefill itself computes. `reset_decode_state` is called BEFORE the
# timer opens, `torch.cuda.synchronize()` immediately before and immediately after. Batch 1,
# prefill width 1536 from position 0, max_len 2049 -- the ranked request probe's own shape -- with
# `graph=False`, so nothing is captured. Five readings per implementation, the parent's full-width
# tail first, then the card's trimmed tail. The tokens are a constant id tensor; the validation
# split is never read.
#
# Placed HERE: after the inherited [step-calibration] block and before the serving-weight
# diagnostic, because that diagnostic and the [body-weights] probe below pop
# `_compiled_decode_step` and swap the weight set, and this block must run while the reporter's
# own compilations and the model's own weights are still installed. It forces no compilation of
# its own -- the prefill arm is eager on this lineage, and `_decode_body_full_tail_prefill` is
# the parent's prefill arm verbatim -- and it pops nothing.
#
# `torch.no_grad()` is opened BEFORE `init_decode_state` and the autocast context is inside it,
# which is the form generation 8 settled: without it a generated flash-attn3 backward can fire
# and destroy the diagnostic.
#
# WHAT IT CANNOT SEE, stated rather than left to a reader: each reading is a SINGLE prefill call,
# not a 30-pass median, so it sees neither cross-launch drift nor the scored probe's within-launch
# scatter; it is NOT comparable with `request_ms_median` or `nopref_request_ms_median`, which
# time a whole request of one prefill plus 512 steps; it prices ONE prefill call and nothing else,
# so it says nothing about the 512 timed steps; the first reading of each label may carry that
# shape's one-time kernel workspace, which is why all five readings and the min are printed
# rather than a mean; and it cannot separate the removed MLP and projection rows from the
# narrowed attention call -- it prices the whole trim as one thing.
#
# One try/except around the whole thing: a diagnostic may not fail a launch.
_PREFILL_COMPARE_MAX_LEN = 2049
_PREFILL_COMPARE_WIDTH = 1536
_PREFILL_COMPARE_CALLS = 5

try:
    _pc_inner = getattr(model, "_orig_mod", model)
    _pc_state = None
    try:
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            _pc_state = _pc_inner.init_decode_state(batch=1, max_len=_PREFILL_COMPARE_MAX_LEN,
                                                    graph=False)
            _pc_tokens = torch.full((1, _PREFILL_COMPARE_WIDTH), 1, dtype=torch.int64,
                                    device=device)
            _pc_cases = (
                ("parent_full_tail", lambda t, s: _pc_inner._decode_body_full_tail_prefill(t, s)),
                ("card_trimmed_tail", lambda t, s: _pc_inner._decode_body(t, s, True)),
            )
            for _pc_label, _pc_call in _pc_cases:
                _pc_times = []
                for _ in range(_PREFILL_COMPARE_CALLS):
                    _pc_inner.reset_decode_state(_pc_state)
                    torch.cuda.synchronize()
                    _pc_t0 = time.time()
                    _pc_logits = _pc_call(_pc_tokens, _pc_state)
                    torch.cuda.synchronize()
                    _pc_times.append((time.time() - _pc_t0) * 1000.0)
                    del _pc_logits
                print(f"[prefill-compare] impl={_pc_label} "
                      f"max_len={_PREFILL_COMPARE_MAX_LEN} "
                      f"prefill_width={_PREFILL_COMPARE_WIDTH} batch=1 graph=False "
                      f"calls={_PREFILL_COMPARE_CALLS} "
                      f"ms=[{', '.join(f'{v:.4f}' for v in _pc_times)}] "
                      f"min_ms={min(_pc_times):.4f} "
                      f"spread_ms={max(_pc_times) - min(_pc_times):.4f} "
                      f"elapsed_s={time.time() - t_start:.0f} "
                      f"basis=one _decode_body(prefill=True) call per reading, "
                      f"synchronised at each end, reset before the timer, constant tokens, "
                      f"post-report, no recompile forced; single-call timings, not a 30-pass "
                      f"median; NOT comparable with the reporter's request medians, which time "
                      f"one prefill plus 512 steps; prices one prefill call and nothing else")
            del _pc_tokens
    finally:
        if _pc_state is not None:
            _pc_state.pop("graph", None)
        del _pc_state
        gc.collect()
        torch.cuda.empty_cache()
except Exception as exc:        # noqa: BLE001 -- a diagnostic may not fail a launch
    print(f"[prefill-compare] instrument failed: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# FREE INSTRUMENT: the two-neighbourhood comparator
# ---------------------------------------------------------------------------
# This candidate's edit is two keys in one options dict, and the run's working resolution on the
# scored median is wide enough that a scored delta on its own attributes nothing. So this block
# prices BOTH searches inside THIS launch, on the same host, the same weights and the same state:
# at each request shape it compiles `_decode_body_step` twice, once under the parent's option dict
# and once under that dict plus `coordinate_descent_check_all_directions` and
# `coordinate_descent_search_radius=2`, captures each, and times replays of each.
#
# It runs after `report_efficiency_metrics` has returned and after the two inherited instruments
# above, so every official number -- val_bpb, both cache readings, all four request medians,
# peak_vram_bytes and peak_vram_bytes_inference -- has already been read and printed, and the cache
# probe's before/after bracket closed long before this point. It prints no METRICS_JSON line and
# never reads the validation split: its tokens are a constant id tensor. It is placed AFTER
# `[step-calibration]` and after `[prefill-compare]` rather than between them, on purpose: those
# two are the parent's own instruments, they must run while the reporter's compilations and the
# model's own weights are still installed, and leaving them where the parent had them keeps their
# readings comparable with the parent's launch instead of shifting them behind four fresh compiles.
#
# `torch.no_grad()` is opened before `init_decode_state` and the autocast context is inside it,
# which is the form generation 8 settled: forcing this region to retrace with grad enabled is what
# raises BackendCompilerFailed while a backward is generated for the FA3 custom op.
#
# SYMMETRY, and the one respect in which it is not what the card asked for. The card asked for
# `force_disable_caches: True` in BOTH labels' option dicts, so that neither label could be served
# from a cache and both would pay real codegen and a real coordinate-descent search. On the pinned
# torch that key CANNOT go in an options dict: `torch._inductor.config.force_disable_caches` is a
# `Config(alias="torch.compiler.config.force_disable_caches")`, it is therefore absent from
# `config.get_config_copy()`, and `_TorchCompileInductorWrapper.apply_options` raises
# `RuntimeError: Unexpected optimization option force_disable_caches` for every key not in that
# copy -- checked on CPU against the pinned 2.9.1 before this file was written. Written as the card
# asked, all four cases would raise at `torch.compile` and the block would print nothing. So it is
# set the only way the pinned torch allows, as a scoped patch of the global inductor config around
# each case, which is what the readers actually consult: `compile_fx` computes `use_cache` from
# `not config.force_disable_caches` and wraps the compile in `with_fresh_cache_if_config`, and
# `codegen/triton.py::inductor_meta_common` copies `force_disable_caches` into every generated
# kernel's `inductor_meta`, where `check_autotune_cache` reads it first and writes
# `autotune_cache_state='force_disabled'`. The patch is entered and exited per case, is post-report,
# and touches no scored compilation.
#
# WHAT THAT COSTS, stated rather than hidden. Because the key is not in the options dict, it does
# not enter `_TorchCompileInductorWrapper.config`, so it does not make the parent label a distinct
# backend. Dynamo's BACKEND_MATCH guard compares those dicts, and the parent label's dict is
# byte-for-byte the scored dict, which `[step-calibration]` above has already compiled at these two
# lengths -- so the parent label can be served from that in-process cache entry and may pay no
# codegen at all, while the card label's dict differs and must compile. `entries=` and `new_entry=`
# are printed per label so that this is readable rather than assumed, and `[autotune-cache]` is
# printed per label so that whether each label's kernels searched or read a stored answer is
# readable too. A parent label served from that entry is still a valid timing OF THE PARENT
# LOWERING; what it is not is a fresh search.
#
# Two more things it cannot do. Neither label is the scored artefact -- the pair prices the two
# lowerings against each other and does not replay the compilation that produced the official
# numbers. And it cannot say WHICH generated kernels differ: the kernel count above is
# process-wide.
#
# Each case is timed THREE times, not once, so the pair is read against the instrument's own
# within-launch scatter instead of an assumed precision. Every repeat is an independent pass at the
# same shape: `reset_decode_state`, the prefill, then 512 replays -- which is also what keeps the
# write position inside the cache, since 512 steps from a 1536-token prefill reach exactly max_len
# 2049 and a second uninterrupted batch would run past the end of the allocation. The parent label
# is measured FIRST at each shape, so any warm-up or thermal drift across the pair works against
# the card rather than for it.
#
# A failure is contained per case, so a raise at one shape or one label still leaves the other
# three readings. The installed decode callable is saved before the block and restored in a
# `finally`, so the model leaves this block exactly as the reporter left it.
_NB_PARENT_OPTIONS = {"coordinate_descent_tuning": True,
                      "max_autotune_pointwise": True}
_NB_CARD_OPTIONS = {"coordinate_descent_tuning": True,
                    "max_autotune_pointwise": True,
                    "coordinate_descent_check_all_directions": True,
                    "coordinate_descent_search_radius": 2}

try:
    nb_inner = getattr(model, "_orig_mod", model)
    nb_had_saved = "_compiled_decode_step" in nb_inner.__dict__
    nb_saved = nb_inner.__dict__.get("_compiled_decode_step")
    from torch._inductor import config as nb_inductor_config
    try:
        from torch._dynamo.eval_frame import _debug_get_cache_entry_list as _nb_entry_list
    except Exception:       # noqa: BLE001 -- a private helper; -1 records that it is absent
        _nb_entry_list = None

    def nb_entry_count():
        """How many Dynamo cache entries the step's code object holds. Host side, no device work."""
        if _nb_entry_list is None:
            return -1
        try:
            return len(_nb_entry_list(nb_inner._decode_body_step))
        except Exception:   # noqa: BLE001 -- a diagnostic may not fail a launch
            return -1

    try:
        for nb_max_len, nb_prefill_width in ((514, 1), (2049, 1536)):
            # The parent's dict FIRST at each shape, then the same dict plus this card's two keys.
            for nb_label, nb_opts in (("parent", _NB_PARENT_OPTIONS),
                                      ("card", _NB_CARD_OPTIONS)):
                nb_state = None
                try:
                    with torch.no_grad(), \
                         nb_inductor_config.patch(force_disable_caches=True), \
                         torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        nb_before = nb_entry_count()
                        nb_fresh = torch.compile(nb_inner._decode_body_step, dynamic=False,
                                                 fullgraph=False, options=nb_opts)
                        # The backend runs on the FIRST call, not at `torch.compile`, so this is
                        # where the lowering and its search are paid for. Timed on a graph=False
                        # state, which captures nothing, and printed immediately so the figure and
                        # the entry delta survive even if the capture stage below raises. Not
                        # wrapped in `_guard_decode_step_call`: a guard would silently fall back to
                        # the uncompiled method, and the replays would then time eager execution
                        # under this label instead of this lowering.
                        nb_warm = nb_inner.init_decode_state(batch=1, max_len=nb_max_len,
                                                             graph=False)
                        nb_idx = torch.zeros(1, 1, dtype=torch.int64,
                                             device=nb_warm["seq"].device)
                        torch.cuda.synchronize()
                        nb_t0 = time.time()
                        nb_out = nb_fresh(nb_idx, nb_warm)
                        torch.cuda.synchronize()
                        nb_compile_seconds = time.time() - nb_t0
                        nb_after = nb_entry_count()
                        del nb_out, nb_idx, nb_warm
                        gc.collect()
                        torch.cuda.empty_cache()
                        print(f"[neighbourhood-ab] label={nb_label} max_len={nb_max_len} "
                              f"radius={nb_opts.get('coordinate_descent_search_radius', 'absent')} "
                              f"all_directions="
                              f"{nb_opts.get('coordinate_descent_check_all_directions', 'absent')} "
                              f"force_disable_caches={nb_inductor_config.force_disable_caches} "
                              f"first_call_seconds={nb_compile_seconds:.3f} "
                              f"entries={nb_before}->{nb_after} "
                              f"new_entry={nb_after > nb_before} "
                              f"elapsed_s={time.time() - t_start:.0f}")
                        _autotune_cache_witness(f"neighbourhood-ab-{nb_label}-{nb_max_len}")
                        # Install this lowering as the model's step, so the capture below and its
                        # replays run exactly this compilation and nothing else.
                        nb_inner.__dict__["_compiled_decode_step"] = nb_fresh
                        nb_state = nb_inner.init_decode_state(batch=1, max_len=nb_max_len,
                                                              graph=True)
                        # CONSTRUCT THE CAPTURE BEFORE OPENING ANY TIMER, so every timed call is a
                        # replay and not a capture plus a compilation.
                        nb_state["graph"] = _GraphedDecodeStep(nb_inner, nb_state)
                        nb_captured = bool(nb_state["graph"].captured)
                        nb_tokens_prefill = torch.zeros(1, nb_prefill_width, dtype=torch.int64,
                                                        device=nb_state["seq"].device)
                        nb_tokens_step = torch.zeros(1, 1, dtype=torch.int64,
                                                     device=nb_state["seq"].device)
                        nb_reps = []
                        for _ in range(3):
                            # One independent pass per repeat, in the order serving uses: the
                            # position returns to 0 and the prefill is re-run, so all three batches
                            # decode the same cached extent and none of them writes past `max_len`.
                            nb_inner.reset_decode_state(nb_state)
                            nb_logits, nb_state = nb_inner.decode_step(nb_tokens_prefill, nb_state)
                            del nb_logits
                            # 512 calls as ONE batch: one synchronize at each end, never one per
                            # call, and at the same cached extent `[step-calibration]` reads.
                            torch.cuda.synchronize()
                            nb_t1 = time.time()
                            for _ in range(512):
                                nb_logits, nb_state = nb_inner.decode_step(nb_tokens_step,
                                                                          nb_state)
                            torch.cuda.synchronize()
                            nb_reps.append((time.time() - nb_t1) * 1000.0)
                            del nb_logits
                        del nb_tokens_prefill, nb_tokens_step
                    nb_per_call = [nb_ms / 512.0 for nb_ms in nb_reps]
                    print(f"[neighbourhood-ab] label={nb_label} max_len={nb_max_len} "
                          f"prefill_width={nb_prefill_width} captured={nb_captured} "
                          f"compile_seconds={nb_compile_seconds:.3f} batch_calls=512 "
                          f"passes={len(nb_per_call)} "
                          f"ms_per_call=" + ",".join(f"{v:.4f}" for v in nb_per_call) + " "
                          f"min={min(nb_per_call):.4f} max={max(nb_per_call):.4f} "
                          f"spread={max(nb_per_call) - min(nb_per_call):.4f} "
                          f"basis=three independent passes of 512 replays of one freshly compiled "
                          f"region in one launch, each reset and re-prefilled, capture built "
                          f"before the timer, constant tokens, post-report, caches force-disabled "
                          f"for the compile; NOT the scored artefact")
                except Exception as exc:    # noqa: BLE001 -- one case may not cost the others
                    print(f"[neighbourhood-ab] case failed label={nb_label} "
                          f"max_len={nb_max_len}: {type(exc).__name__}: {exc}")
                finally:
                    if nb_state is not None:
                        nb_state.pop("graph", None)
                    del nb_state
                    gc.collect()
                    torch.cuda.empty_cache()
    finally:
        # The model leaves this block with the callable the reporter left installed.
        if nb_had_saved:
            nb_inner.__dict__["_compiled_decode_step"] = nb_saved
        else:
            nb_inner.__dict__.pop("_compiled_decode_step", None)
except Exception as exc:        # noqa: BLE001 -- a diagnostic may not fail a launch
    print(f"[neighbourhood-ab] instrument failed: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Free diagnostic, after every number is read. `report_efficiency_metrics` has already
# returned, having printed the final METRICS_JSON line above and with it both peak readings,
# both cache readings and all four request medians. Nothing allocated, computed or spent
# below can reach a reported key, and nothing below prints a METRICS_JSON line.
#
# It exists because this candidate's claim is about operand conversions -- that the step no
# longer converts a float32 master per matmul -- and that count is not derivable from the
# source. The same two lines were published by an earlier float32-copy candidate, so the
# numbers are comparable; this one records them on a lineage whose masters are gone from the
# decode path. The throwaway state is built with graph=False, so nothing is captured, and it
# is deleted here.
# ---------------------------------------------------------------------------

def _count_to_copy(fn, idx, state):
    """`aten._to_copy` calls, and their output bytes, issued by one call of `fn`."""
    from torch.utils._python_dispatch import TorchDispatchMode

    class _Counter(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.calls = 0
            self.bytes = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            if func is torch.ops.aten._to_copy.default and isinstance(out, torch.Tensor):
                self.calls += 1
                self.bytes += out.numel() * out.element_size()
            return out

    counter = _Counter()
    with torch.no_grad(), autocast_ctx, counter:
        fn(idx, state)
    torch.cuda.synchronize()
    return counter.calls, counter.bytes


def _report_serving_weight_copies(wrapped):
    """What the decode path's weight copies are, and what one eager step still converts."""
    inner = getattr(wrapped, "_orig_mod", wrapped)
    sw_layers, sw_lm_head = inner._serving_weights()
    flat = [sw_lm_head] + [t for w in sw_layers for t in w.values() if t is not None]
    print("serving weight copies: n={} dtypes={} all_leaf={} any_requires_grad={} bytes={}"
          .format(len(flat), sorted({str(t.dtype) for t in flat}),
                  all(t.is_leaf for t in flat), any(t.requires_grad for t in flat),
                  sum(t.numel() * t.element_size() for t in flat)))
    # The deployed width-1 shape: batch 1, the ranked probe's cache length, and a position
    # inside the prompt the ranked shape decodes from.
    idx = torch.zeros(1, 1, dtype=torch.int64, device=device)
    state = inner.init_decode_state(batch=1, max_len=2048, graph=False)
    state["seq"].fill_(1536)
    eager_calls, eager_bytes = _count_to_copy(inner._decode_body_step, idx, state)
    print(f"decode step _to_copy: eager calls={eager_calls} bytes={eager_bytes}")
    del state


try:
    _report_serving_weight_copies(model)
except Exception as exc:                    # noqa: BLE001 -- a diagnostic may not fail a run
    print(f"serving weight copy diagnostic failed: {type(exc).__name__}: {exc}")

# ---------------------------------------------------------------------------
# FREE INSTRUMENT: how much of the width-1 step is its weight traffic
# ---------------------------------------------------------------------------
# Runs AFTER report_efficiency_metrics has returned, so every official number -- val_bpb,
# both cache-byte readings, both request shapes, peak_vram_bytes and
# peak_vram_bytes_inference -- has already been read and printed. The cache probe's
# before/after bracket closed long before this. It prints no METRICS_JSON line, never reads
# the validation split (its tokens are a constant id tensor), never calls the reporter, and
# builds and drops its own states. The whole block is wrapped so it cannot fail the launch.
#
# WHAT IT SEPARATES: nothing so far has timed the width-1 body against a SMALLER SET OF
# DISTINCT WEIGHT BYTES at identical kernels, shapes, dtypes and call counts. The `shared`
# variant points every non-value-embedding layer at layer 0's weight dict object and every
# value-embedding layer at layer 1's, so the step issues exactly the same calls on exactly the
# same shapes and reads fewer distinct bytes.
#
# WHAT IT CANNOT SEE, stated rather than left to a reader: it is one pass, not a 30-pass
# median, so it cannot see cross-launch drift; the shared set's distinct bytes are small
# enough to sit inside this device's L2 while the real set's are not, so the delta bounds what
# making the weight set resident would be worth and does not isolate memory traffic from
# kernel time; and the shared set changes weight VALUES, so its logits are meaningless and are
# discarded.
#
# On this candidate each variant forces a fresh tuning pass over the recompiled region, so if
# the launch's own wall clock is already close to the launch timeout the two `shared` cases
# are skipped and the skip is printed. That fallback is the card's, and 1500 s against the
# 1800 s timeout is the threshold this file uses for "close".
_BODY_WEIGHTS_CASES = [("real", 2049, 1536), ("shared", 2049, 1536),
                       ("real", 514, 1), ("shared", 514, 1)]
_BODY_WEIGHTS_WALL_LIMIT = 1500.0

def _body_weights_probe():
    inner = getattr(model, "_orig_mod", model)
    real_layers, real_lm_head = inner._serving_weights()
    # Same objects, so the same tensors: one weight dict serves every non-value-embedding
    # layer and one serves every value-embedding layer. The unembedding is unchanged.
    shared_layers = [real_layers[1] if block.attn.ve_gate is not None else real_layers[0]
                     for block in inner.transformer.h]
    sets = {"real": (real_layers, real_lm_head), "shared": (shared_layers, real_lm_head)}

    def byte_counts(layers, lm_head):
        flat = [t for lw in layers for t in (lw["packed"], lw["attn_c_proj"],
                                            lw["c_fc"], lw["mlp_c_proj"])] + [lm_head]
        total = sum(t.numel() * t.element_size() for t in flat)
        distinct = {}
        for t in flat:
            distinct[t.data_ptr()] = t.numel() * t.element_size()
        return sum(distinct.values()), total

    for variant, max_len, prefill_width in _BODY_WEIGHTS_CASES:
        if variant == "shared" and time.time() - t_start > _BODY_WEIGHTS_WALL_LIMIT:
            print(f"[body-weights] variant={variant} max_len={max_len} "
                  f"prefill_width={prefill_width} skipped=wall_clock_guard "
                  f"elapsed_s={time.time() - t_start:.0f} limit_s={_BODY_WEIGHTS_WALL_LIMIT:.0f}")
            continue
        layers, lm_head = sets[variant]
        distinct_bytes, total_bytes = byte_counts(layers, lm_head)
        # Swap the weight set and drop the compiled callable, so the region is recompiled
        # against the tensors this case actually uses rather than replayed against the other
        # case's. Both are the model's own bfloat16 copies; nothing is trained here.
        inner.__dict__["_serving_weight_copies"] = (layers, lm_head)
        inner.__dict__.pop("_compiled_decode_step", None)
        state = inner.init_decode_state(batch=1, max_len=max_len, graph=True)
        # The capture is CONSTRUCTED BEFORE THE TIMER OPENS, so the first timed call is a
        # replay and carries neither the capture nor the compilation.
        state["graph"] = _GraphedDecodeStep(inner, state)
        captured = bool(state["graph"].captured)
        tokens = torch.full((1, prefill_width), 1, dtype=torch.int64, device=device)
        one = torch.full((1, 1), 1, dtype=torch.int64, device=device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, state = inner.decode_step(tokens, state)
            del logits
            torch.cuda.synchronize()
            t0 = time.time()
            logits, state = inner.decode_step(one, state)
            torch.cuda.synchronize()
            first_ms = (time.time() - t0) * 1000.0
            del logits
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(512):
                logits, state = inner.decode_step(one, state)
            torch.cuda.synchronize()
            batch_ms = (time.time() - t0) * 1000.0
            del logits
        print(f"[body-weights] variant={variant} max_len={max_len} "
              f"prefill_width={prefill_width} captured={captured} "
              f"distinct_weight_bytes={distinct_bytes} total_weight_bytes={total_bytes} "
              f"first_call_ms={first_ms:.3f} batch_calls=512 batch_ms={batch_ms:.3f} "
              f"batch_ms_per_call={batch_ms / 512:.4f} "
              f"basis=one pass, one launch, post-report, capture constructed before the "
              f"timer, constant tokens")
        state.pop("graph", None)
        del state, tokens, one
        gc.collect()
        torch.cuda.empty_cache()

    # Leave the model holding its own weights again. Nothing reads them after this point.
    inner.__dict__["_serving_weight_copies"] = (real_layers, real_lm_head)
    inner.__dict__.pop("_compiled_decode_step", None)

try:
    _body_weights_probe()
except Exception as _exc:        # noqa: BLE001 -- a diagnostic may never fail the launch
    print(f"[body-weights] probe failed: {type(_exc).__name__}: {_exc}")
