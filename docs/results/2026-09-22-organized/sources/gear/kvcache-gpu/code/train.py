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


def reuses_kv(layer_idx, n_layer):
    """Whether this layer reads the previous layer's keys and values instead of computing its own.

    A layer that reuses holds no cache of its own, so the decode state shrinks by exactly the number of
    reusing layers. The layers that must keep their own are the ones carrying a value embedding --
    `has_ve` puts those on the odd indices here -- plus layer 0, which has no predecessor. That leaves
    the even layers above 0 reusing, i.e. five cached streams instead of eight.
    """
    return layer_idx > 0 and not has_ve(layer_idx, n_layer)


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


# ---------------------------------------------------------------------------
# The decode cache's storage format. Compiled so that each of these is ONE kernel: the cast,
# the scale and the rounding fuse, and the per-step kernel count stays a small constant.
# Both are decode-only, so no FLOPs formula is owed for them.
# ---------------------------------------------------------------------------

# Offset that puts every scale this model produces inside a single unsigned byte. A code is an
# eighth of an octave, so 255 codes span 31 octaves; the scales here are a maximum activation over
# 127, which lands around 2**-10, well inside that.
_SCALE_BIAS = 200.0
@torch.compile(dynamic=False, fullgraph=True)
def _quantise_kv(x):
    """int8 payload plus a ONE-BYTE scale per (batch, layer, position, head).

    The scale is rounded to its stored precision before it divides, so the payload is chosen
    against the same number the dequantiser will multiply by and a coarse scale costs range, not
    accuracy - with one exception: a stored scale BELOW the true maximum over 127 clamps the
    largest component at 127 and loses the peak, which is accuracy. So the scale is stored as an
    eighth-of-an-octave exponent code, rounded UP. Two consequences. It never clips, with no
    headroom factor to tune. And a code step is 2**(1/8), 9.05%, so the payload peaks at worst at
    127/2**(1/8) = 116 instead of 127: a step of 1/116 of the vector's maximum instead of 1/127.
    That halves the scale tensors, 10 of the 660 bytes this cache spends per key.

    An exponent code rather than float8 because the kernels are compiled for this GPU and Triton
    on it supports only e5m2 and e4b15 - e4m3 fails to compile - and e5m2's two mantissa bits are
    a 25% step, three times coarser than this. Code 0 is reserved for an all-zero vector.
    """
    amax = x.abs().amax(dim=-1, keepdim=True).float()
    code = torch.ceil(torch.log2(torch.clamp(amax / 127.0, min=1e-30)) * 8.0) + _SCALE_BIAS
    code = torch.clamp(code, 1.0, 255.0)
    scale = torch.where(amax > 0, torch.exp2((code - _SCALE_BIAS) / 8.0), torch.zeros_like(amax))
    inv = torch.where(scale > 0, 1.0 / scale, torch.zeros_like(scale))
    payload = torch.clamp(torch.round(x.float() * inv), -127.0, 127.0).to(torch.int8)
    return payload, torch.where(amax > 0, code, torch.zeros_like(code)).to(torch.uint8)


@torch.compile(dynamic=False, fullgraph=True)
def _dequantise_ring(payload, scale):
    """One extent group's int8 ring -> the bfloat16 cache the kernel reads, one slot longer.

    The extra slot is where `flash_attn_with_kvcache` appends this step's key and value: the
    ring holds only the PREVIOUS keys, so nothing valid is overwritten and the kernel's own
    fused append still does the current position.
    """
    code = scale.float()
    factor = torch.where(code > 0, torch.exp2((code - _SCALE_BIAS) / 8.0),
                         torch.zeros_like(code)).to(torch.bfloat16)
    return F.pad(payload.to(torch.bfloat16) * factor, (0, 0, 0, 0, 0, 1))


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.reuses_kv = reuses_kv(layer_idx, config.n_layer)
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        if not self.reuses_kv:
            self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
            self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def keys_values(self, x, ve, cos_sin):
        """This layer's own keys and values, rotated and normalised as the kernel wants them."""
        B, T, C = x.size()
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        cos, sin = cos_sin
        return norm(apply_rotary_emb(k, cos, sin)), v

    def queries(self, x, cos_sin):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = cos_sin
        return norm(apply_rotary_emb(q, cos, sin))

    def forward(self, x, ve, cos_sin, window_size, shared_kv=None):
        B, T, C = x.size()
        k, v = shared_kv if shared_kv is not None else self.keys_values(x, ve, cos_sin)
        q = self.queries(x, cos_sin)
        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        y = y.contiguous().view(B, T, -1)
        return self.c_proj(y), (k, v)


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

    def forward(self, x, ve, cos_sin, window_size, shared_kv=None):
        y, kv = self.attn(norm(x), ve, cos_sin, window_size, shared_kv)
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
            if not block.attn.reuses_kv:
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
        short_window = min(SHORT_WINDOW, long_window)
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
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
        kv = None
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x, kv = block(x, ve, cos_sin, self.window_sizes[i],
                          kv if block.attn.reuses_kv else None)
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
        # Each layer's cache is sized to its OWN attention window, not to max_len. A layer whose
        # left window is W is only ever asked for keys [p-W, p], so W+1 slots written cyclically
        # hold exactly the set the kernel reads, at every position and for any prefill length --
        # the window bound becomes the buffer's geometry instead of a mask over dead slots.
        # `window_pattern="SSSL"` gives six of eight layers a 1024-key window here, so this drops
        # 37.5% of the slots without approximating anything.
        #
        # Layers are grouped by extent, so the slot arithmetic and the valid-key count are one
        # small op per group rather than per layer, and each group is one allocation whose
        # `[:, j]` view is the contiguous [batch, extent, heads, dim] tensor the kernel takes.
        extents, groups = self._cache_groups(max_len)
        # The whole decode state is ONE allocation, carved into views. The instrument charges the
        # allocator's block rounding on every allocation the state holds, so separate tensors pay it
        # separately: the int32 write position is four bytes and is charged 512, and each scale tensor
        # is charged up to 500 bytes of slack it never uses. Packing pays that rounding once. The
        # bytes the caches actually use, the arithmetic, the kernel calls and the numerics are
        # untouched -- every view is contiguous and the same shape the kernel took before.
        # Each payload next to its own scales. Ordering is free now that the scales are single bytes:
        # nothing in the pool needs more than 4-byte alignment and `seq` holds the front, so every
        # carve-out is contiguous whatever the order. This order puts the four tensors a group's ring
        # writes in one step - keys, their scales, values, their scales - adjacent in the allocation.
        specs = []          # (key, dtype, shape)
        for ext, layers in groups:
            specs.append(("kq", torch.int8, (batch, len(layers), ext, cfg.n_kv_head, head_dim)))
            specs.append(("ks", torch.uint8, (batch, len(layers), ext, cfg.n_kv_head, 1)))
            specs.append(("vq", torch.int8, (batch, len(layers), ext, cfg.n_kv_head, head_dim)))
            specs.append(("vs", torch.uint8, (batch, len(layers), ext, cfg.n_kv_head, 1)))
        itemsize = {torch.bfloat16: 2, torch.uint8: 1, torch.int8: 1}

        def nbytes(dtype, shape):
            count = 1
            for size in shape:
                count *= size
            return count * itemsize[dtype]

        seq_bytes = 4 * batch                      # int32, and first so its offset is 4-byte aligned
        total = seq_bytes + sum(nbytes(dt, shape) for _, dt, shape in specs)
        pool = torch.zeros(total, dtype=torch.uint8, device=dev)
        views = {"kq": [], "vq": [], "ks": [], "vs": []}
        offset = seq_bytes
        for key, dtype, shape in specs:
            span = nbytes(dtype, shape)
            views[key].append(pool[offset:offset + span].view(dtype).view(*shape))
            offset += span
        assert offset == total
        return {
            "max_len": max_len,
            "graph_enabled": bool(graph),
            "pool": pool,
            "seq": pool[:seq_bytes].view(torch.int32),
            # Extents and group membership are structural constants of the model's window
            # pattern, not positions: they do not change as the request advances.
            "groups": groups,
            "slot_of": {i: (g, j) for g, (_, layers) in enumerate(groups)
                        for j, i in enumerate(layers)},
            "kq": views["kq"],
            "vq": views["vq"],
            "ks": views["ks"],
            "vs": views["vs"],
        }

    def _cache_groups(self, max_len):
        """Per-layer cache extents from the window pattern, and the layers grouped by extent.

        W slots, not W+1: this step's own key and value are appended into the dequantised
        scratch by the attention kernel, so the ring only ever has to hold the W keys before
        the current position. At position p it holds [p-W, p-1] and the kernel adds p, which is
        exactly the window [p-W, p].
        """
        extents = [min(max_len - 1, max_len if w[0] < 0 else w[0]) for w in self.window_sizes]
        # Only the layers that compute their own keys and values hold a cache; a reusing layer reads its
        # predecessor's, so the state has as many streams as there are caching layers.
        owners = [i for i in range(len(extents)) if not self.transformer.h[i].attn.reuses_kv]
        order = sorted({extents[i] for i in owners}, reverse=True)
        return extents, [(ext, [i for i in owners if extents[i] == ext]) for ext in order]

    def reset_decode_state(self, state):
        """Return `state` to position 0 so the same buffers serve another request.

        A fresh state per request would invalidate an address-bound CUDA graph. Stale bytes
        above the position are unreachable -- attention reads only `cache_seqlens` keys -- so
        zeroing them would be timed work serving never does.
        """
        state["seq"].zero_()
        return state

    def _decode_body(self, idx, state, prefill):
        """`forward`'s body with the cache spliced in. `prefill` is a Python bool that is
        constant per variant, so the two branches are two graphs, not one per position."""
        B, Tn = idx.size()
        if prefill:
            cos, sin = self.cos[:, :Tn], self.sin[:, :Tn]
            slots, counts, kc, vc = None, None, None, None
        else:
            seq_idx = state["seq"].to(torch.int64)
            cos = self.cos.index_select(1, seq_idx)
            sin = self.sin.index_select(1, seq_idx)
            # One slot index and one valid-key count per extent group, read from `seq` on the
            # device: the write position stays a tensor, so a replayed graph advances with it.
            slots = [torch.remainder(seq_idx[:1], ext) for ext, _ in state["groups"]]
            # keys already cached, i.e. where the kernel appends this step's key
            counts = [torch.clamp(state["seq"], max=ext) for ext, _ in state["groups"]]
            kc = [_dequantise_ring(payload, scale)
                  for payload, scale in zip(state["kq"], state["ks"])]
            vc = [_dequantise_ring(payload, scale)
                  for payload, scale in zip(state["vq"], state["vs"])]

        x = norm(self.transformer.wte(idx))
        x0 = x
        fresh_k = [[] for _ in state["groups"]]
        fresh_v = [[] for _ in state["groups"]]
        shared_k = shared_v = None
        owner_slot = None
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            attn = block.attn
            h = norm(x)
            q = attn.queries(h, (cos, sin))
            if attn.reuses_kv:
                # No keys or values of its own: it attends what the layer above it already put in the
                # scratch, including this step's own pair, so it appends nothing and counts one more.
                g, j = owner_slot
                y = (fa3.flash_attn_func(q, shared_k, shared_v, causal=True,
                                         window_size=self.window_sizes[i]) if prefill else
                     fa3.flash_attn_with_kvcache(q, kc[g][:, j], vc[g][:, j],
                                                 cache_seqlens=counts[g] + 1, causal=True,
                                                 window_size=(-1, -1), num_splits=1))
                x = x + attn.c_proj(y.contiguous().view(B, Tn, -1))
                x = x + block.mlp(norm(x))
                continue
            k, v = attn.keys_values(h, ve, (cos, sin))
            shared_k, shared_v = k, v

            g, j = state["slot_of"][i]
            owner_slot = (g, j)
            ext = state["groups"][g][0]
            if prefill:
                # Attend the keys and values just computed, with this layer's real window; then
                # keep only the tail the buffer can hold, at the slots a wrap would put them in.
                # Writing every prefill position would write some slots twice and leave which
                # write survives undefined; the tail is exactly the set later steps can read.
                y = fa3.flash_attn_func(q, k, v, causal=True,
                                        window_size=self.window_sizes[i])
            else:
                # Attend ALL of this ring's valid slots plus the appended current position, and
                # pass no window: after a wrap a slot index is not a position, so a distance mask
                # would measure the wrong distances -- while the ring itself already holds
                # exactly the keys inside this layer's window. Causal alignment is bottom-right,
                # so a single query at cache_seqlens sees every valid slot and nothing else.
                # num_splits=1 is not a tuning choice: the op's fake kernel refuses to trace at
                # the default num_splits=0, which is precisely what makes an unpinned cache
                # uncompilable. It is a reduction-order choice, and the agreement check in
                # prepare.py still has to pass with it pinned.
                y = fa3.flash_attn_with_kvcache(q, kc[g][:, j], vc[g][:, j], k=k, v=v,
                                                cache_seqlens=counts[g], causal=True,
                                                window_size=(-1, -1),
                                                num_splits=1)
            fresh_k[g].append(k)
            fresh_v[g].append(v)
            x = x + attn.c_proj(y.contiguous().view(B, Tn, -1))
            x = x + block.mlp(norm(x))

        # Store what this call computed, quantised, for the steps that follow. Stacked over each
        # extent group's layers first, so this is a constant handful of kernels rather than one
        # set per layer, and the slots are read from `seq` on the device.
        for g, (ext, layers) in enumerate(state["groups"]):
            if prefill:
                # Only the tail the ring can hold: writing every prefill position would write
                # some slots twice and leave which write survives undefined. The next call is a
                # step at position Tn, and its window needs [Tn-W, Tn-1], so the tail to keep is
                # the last `ext` positions of the prefill, position Tn-1 included.
                start = max(0, Tn - ext)
                pos = torch.remainder(torch.arange(start, Tn, device=idx.device), ext)
                new_k = torch.stack([t[:, start:] for t in fresh_k[g]], dim=1)
                new_v = torch.stack([t[:, start:] for t in fresh_v[g]], dim=1)
            else:
                pos = slots[g]
                new_k = torch.stack(fresh_k[g], dim=1)
                new_v = torch.stack(fresh_v[g], dim=1)
            kq, ks = _quantise_kv(new_k)
            vq, vs = _quantise_kv(new_v)
            state["kq"][g].index_copy_(2, pos, kq)
            state["ks"][g].index_copy_(2, pos, ks)
            state["vq"][g].index_copy_(2, pos, vq)
            state["vs"][g].index_copy_(2, pos, vs)

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

    def _advance(self):
        logits = self.model._decode_body(self.static_idx, self.state, prefill=False)
        self.state["seq"].add_(1)
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
ASPECT_RATIO = 64       # model_dim = depth * ASPECT_RATIO
# Half the head dimension: eight query heads of 64 channels at the same model width, still one
# key/value head, so a cached position holds 64 channels instead of 128 and the state halves again.
# `ve_gate` stays Linear(32, 1), so nothing widens - exp0023's 3,072-FLOP ceiling breach came from a gate
# sized by head count - and c_k, c_v and the value-embedding tables halve with the cached channel count.
# Measured price on a sibling base: +0.0112 val_bpb (exp0034).
HEAD_DIM = 64           # target head dimension for attention
# Grouped-query attention: query heads per key/value head. Every cache tensor is proportional to
# n_kv_head, so this divides the ranking target directly; the value-embedding tables are indexed
# by kv_dim = n_kv_head * head_dim and shrink with it, which is where the quality is paid.
KV_GROUP = 8
# Every layer keeps the short window: no full-context layer at all. With the cache held at each
# layer's window extent the two "L" layers hold 2,047 slots each against a short layer's 1,024, so
# they carry 40% of the state on 25% of the layers; and a stack of eight 1024-key layers still
# composes a receptive field past the 2,048-token request. Measured free on another base: exp0017 read
# -0.0012 val_bpb and -42.6 ms for exactly this change.
WINDOW_PATTERN = "S"    # sliding window pattern: L=full, S=short context
# The short window as a fraction of the sequence length. Each layer's cache is exactly W slots, so
# this halves the state. 512 keys is four FA3 key tiles, and at one key/value head the rung measured
# free: exp0031 read -0.0015 val_bpb and -71.9 ms for it, because the attention FLOPs it frees turn
# into 7% more updates in the fixed training clock.
# elite/0's window idea at the next fraction: 256 keys is elite/0's own value and would reproduce its
# program, so this takes 240 - 1.09 halvings from 512, about +0.012 at the measured 0.011 per halving,
# against this base's 0.0119 of margin. It is a coin flip by construction, and its target, 159,232 B, is
# 6% below the run's best.
# 216 keys. Priced on the axis's own measured slope rather than its average: below 238 keys a halving costs
# 0.0161 (exp0065), not the 0.011 it cost above, so 240 -> 216 is 0.152 of a halving and about +0.0024 against
# this base's 0.0043 of margin. The larger step - 192 keys, 127,488 B - needs 0.0050 and missed by 0.0009 when
# it was tried at this learning rate.
# 193 keys. With the state in one allocation the charge is ceil512(660W + 4), so it steps every 0.78 keys and
# 193 needs 127,384 bytes and is charged 127,488, wasting 104. That is -3.86% against the recorded 132,608.
# The price: these constants measured 204 keys at 1.0484035 (exp0072), and 11 keys further at the axis's
# 0.000116 per key is +0.0013, so about 1.0497 - inside the gate, by less than a launch's scatter.
# 191 keys, charged ceil512(660 x 191 + 4) = 126,464, which is -0.8% against 127,488. Two keys is what the
# margin holds: this program read 1.0496671 at 193 keys and 1.0512235 at 184, so the local cost is 0.00017
# per key - not the 0.000116 the wider part of the axis charges - and 0.0003 of margin buys two of them.
# 188 keys, charged ceil512(650 x 188 + 4) = 122,368, -3.2% against 126,464. With one-byte scales the cache
# spends 650 bytes per key and 650 > 512, so every width is now its own rung and a draw can be spent on each.
# Predicted about 1.0505 against a gate of 1.05: two draws of the 191-key geometry read 1.0497797 and
# 1.0502929, so its mean is 1.0500, and three keys cost 0.0005 at the axis's 0.00017 per key.
# 193 keys, elite/1's own window, with the scales in one byte: charged ceil512(650 x 193 + 4) = 125,952,
# -0.4% against 126,464. The smallest step available and the likeliest to be admitted - this width read
# 1.0496671, the only draw on this lineage with 0.0003 of margin - because the gate crossing is between 191
# and 192 keys and six widths now bracket it.
# 192 keys, which is the width between the two parents' - elite/2's 191 and elite/0's 193 - and the one the
# pair has not measured: charged ceil512(650 x 192 + 4) = 124,928, -0.8% against 125,952. It is also the
# width the gate crossing sits on. Two draws of 193 read 1.04965 and two of 191 read 1.0500, and one key at
# the axis's 0.00017 puts this at about 1.0498 with 0.0002 of margin.
# 191 keys, the width of the recorded best, drawn a third time. Its two draws read 1.0499119 and 1.0502929,
# so the rung straddles the gate and the recorded result is one clearing draw of it. This launch cannot lower
# the record - the rung's charge is the record's 124,416 B - and is the only thing left that can say how
# reproducible the record is.
SHORT_WINDOW = 191

# Optimization
# One microbatch per update, and half the update size again. Two reasons. (1) prepare.py's loader
# yields views of ONE pinned host buffer and refills it on the host straight after an asynchronous
# H2D of it, so an accumulation loop trains partly on overwritten host memory - nondeterministically,
# which is what a 0.0025 val_bpb move at a byte-identical training program measured. With a single
# microbatch the `train_loss.item()` already in the loop synchronises between the H2D and the next
# host write. (2) The first halving of the update size, 2**19 -> 2**18 at device batch 128, was worth
# -0.0499 val_bpb; this is the next rung, 2**17 at device batch 64, which doubles the number of
# updates again and halves the activation memory.
TOTAL_BATCH_SIZE = 2**17 # ~131K tokens per optimizer step, one microbatch, no accumulation
# Learning rate for the token embeddings and the value embeddings (Adam). Same argument that made the matrix
# rate worth re-reading, applied to the other constant the search has moved out from under: this group covers
# `wte` (8192 x 512, unchanged) and the value-embedding tables, which are vocab_size x n_kv_head x head_dim and
# are now 8192 x 64 - an eighth of the width 0.6 was tuned at. Narrower tables get fewer gradient terms per
# row, so a higher rate is the direction; 0.9 is a 50% step rather than a doubling.
EMBEDDING_LR = 0.9      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
# Learning rate for the matrix parameters (Muon). The substrate's 0.04 was tuned at TOTAL_BATCH_SIZE 2**19
# with ~745 updates; this program runs 2**17 with ~3,700, a four-times smaller batch and five times the
# updates, and no launch in this run has re-read the optimiser at that shape. Square-root batch scaling
# points at half, so 0.03 is a quarter off - one rung rather than a jump, because the sign of an optimiser
# knob on this substrate has not been measured and the whole quality budget left is 0.00097.
MATRIX_LR = 0.02        # learning rate for matrix parameters (Muon)
# elite/2's idea is that the substrate's constant stopped being tuned once the search moved the program from
# TOTAL_BATCH_SIZE 2**19 and ~745 updates to 2**17 and ~3,700; elite/1 already carries its first rung (0.03,
# worth -0.0033 twice), so this takes the second, which read -0.0009 more on the 240-key geometry. It costs
# nothing on the target and buys margin the window axis can spend.
# Learning rate for the per-layer scalars (Adam): `x0_lambdas`, which start at 0.1 and control how much of
# the embedding each block sees, and `resid_lambdas`, which start at 1.0. elite/2's idea is that the
# substrate's optimiser constants stopped being tuned when the search moved the program; its own realisation
# was the matrix rate, and this is the same idea at the constant that took the largest change in *step count* -
# 3,700 updates against the ~745 it was set at, so these two scalars now travel five times as far. 0.3 is a
# 40% cut rather than a fifth, because the x0 shortcut is load-bearing on this substrate.
# Back to the substrate's 0.5, which both readings of this constant settle on: 0.3 cost +0.0013 (exp0070) and
# 0.7 cost +0.0022 (exp0077), so 0.5 is bracketed. The margin that buys is what the window below spends.
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# Model size
DEPTH = 8               # number of transformer layers
DEVICE_BATCH_SIZE = 64   # per-device batch size (reduce if OOM)

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
        n_layer=depth, n_head=num_heads, n_kv_head=max(1, num_heads // KV_GROUP),
        n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )

config = build_model_config(DEPTH)
print(f"Model config: {asdict(config)}")

with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()

# Every parameter of rank >= 2 in bfloat16. init_weights() already does this for wte and the
# value embeddings; this extends it to the transformer matrices and the head, so autocast stops
# making a bfloat16 copy of each weight per matmul, and the Adam moments follow through
# zeros_like(p). Muon orthogonalises in bfloat16 already, so the update precision is unchanged.
# The two rank-1 scalar vectors stay float32: bfloat16's epsilon at 1.0 is 0.0078, and
# x0_lambdas starts at 0.1 with lr 0.5, so its updates would round away.
with torch.no_grad():
    for _p in model.parameters():
        if _p.dim() >= 2 and _p.dtype == torch.float32:
            _p.data = _p.data.to(torch.bfloat16)

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
