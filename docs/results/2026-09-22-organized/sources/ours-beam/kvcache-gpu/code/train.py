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
    def __init__(self, config, layer_idx, shares_kv=False):
        super().__init__()
        self.shares_kv = bool(shares_kv)
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        if self.shares_kv:
            # A follower reads its leader's keys and values, so its own projections would be
            # parameters and counted FLOPs that no forward pass executes.
            self.c_k = None
            self.c_v = None
            self.ve_gate = None
        else:
            self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
            self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
            self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, shared_kv=None):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = cos_sin
        if shared_kv is None:
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
            # The leader's keys and values, already rotary-applied and normed by the layer
            # that produced them. This layer contributes only its own query.
            shared_k_in, shared_v_in = shared_kv
            k, v = shared_k_in, shared_v_in
            q = norm(apply_rotary_emb(q, cos, sin))

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


class TokenShiftMixer(nn.Module):
    """Mix each position with the one before it, so the retained state is ONE position.

    A layer that mixes across positions this way needs no key/value cache at all: what a
    request must carry forward is the previous position's input, which does not grow with
    the context.

    Two bias-free Linears over the current and the shifted input rather than one Linear over
    their concatenation: identical parameters and identical counted FLOPs, without
    materialising a 2 * n_embd-wide activation for the backward pass to retain.
    """

    def __init__(self, config):
        super().__init__()
        self.mix_self = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.mix_prev = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x, shift_prev):
        if shift_prev is None:
            # The training path: position 0 has no predecessor, as in the decode path's
            # first position after a reset.
            shift_input = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        else:
            shift_input = torch.cat([shift_prev, x[:, :-1]], dim=1)
        return self.mix_self(x) + self.mix_prev(shift_input)


class Block(nn.Module):
    def __init__(self, config, layer_idx, uses_attention=True, shares_kv=False):
        super().__init__()
        self.uses_attention = bool(uses_attention)
        self.attn = (CausalSelfAttention(config, layer_idx, shares_kv=shares_kv)
                     if self.uses_attention else None)
        self.mixer = None if self.uses_attention else TokenShiftMixer(config)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, shared_kv=None, inject=None):
        if self.uses_attention:
            attn_out, produced_kv = self.attn(norm(x), ve, cos_sin, window_size, shared_kv)
            x = x + attn_out
        else:
            # A mixer layer ignores ve, cos_sin and window_size: value embeddings sit on the
            # odd layers, which keep attention. It produces no keys or values, and says so
            # explicitly rather than by omission.
            mixer_out_local = self.mixer(norm(x), None)
            if inject is not None:
                # The per-token injection, added to the mixer's output. Defined on the
                # cache-free layers only, so the attention branch ignores `inject`.
                mixer_out_local = mixer_out_local + inject
            x = x + mixer_out_local
            produced_kv = None
        x = x + self.mlp(norm(x))
        return x, produced_kv


def quant_ring_pack(tensor_in, group_size):
    """Symmetric int8 quantisation of the last dimension, one bf16 absmax scale per group.

    Module level so the SAME arithmetic serves both ring write paths and both ring read
    paths and cannot drift between them. No zero point: keys are rms-normed and values carry
    a gated value-embedding residual, so both are centred and a zero point would cost a
    second stored tensor for no accuracy. The absmax is clamped because a zeroed ring slot
    has absmax 0, which would divide by zero on the first append of a request.
    """
    reshaped = tensor_in.reshape(*tensor_in.shape[:-1],
                                tensor_in.shape[-1] // group_size, group_size)
    absmax_local = reshaped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
    scale_local = absmax_local / 127.0
    return ((reshaped / scale_local).round().clamp(-127, 127).to(torch.int8).flatten(-2),
            scale_local.squeeze(-1).to(torch.bfloat16))


def quant_ring_unpack(payload_in, scale_in, group_size):
    """The inverse of quant_ring_pack: int8 payload and its scales back to bf16.

    Both ring read paths -- the leader's own step read and the follower's read of the same
    slot -- call THIS function, so they dequantise identically by construction and cannot
    disagree about the same keys.
    """
    return ((payload_in.reshape(*payload_in.shape[:-1],
                                payload_in.shape[-1] // group_size, group_size)
             .to(torch.bfloat16) * scale_in.unsqueeze(-1))
            .flatten(-2).contiguous())


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        # The even layers mix positions with a fixed-size state instead of attention, so they
        # hold no cache. has_ve puts the value embeddings on the odd layers, which keep
        # attention, and layers 3 and 7 -- the long-window ones -- are odd as well.
        self.mixer_layers = [i for i in range(config.n_layer) if i % 2 == 0]
        self.attn_layers = [i for i in range(config.n_layer) if i % 2 == 1]
        # PAIRS: the upper attention layer of each pair reads the lower one's keys and values,
        # so half the attention layers hold no cache. A follower is always the HIGHER index of
        # its pair, so the leader's write precedes the follower's read inside one layer loop.
        # A mixer sits between each pair, which is why the carry must survive a layer that
        # produces nothing. Leaders are 1 and 5 because both carry a value embedding, so the
        # shared basis is the one that receives per-token value information.
        self.kv_leader = {3: 1, 7: 5}
        self.kv_leader_set = set(self.kv_leader.values())
        # One cache per CACHED layer: a follower holds none and reads its leader's.
        self.cached_layers = [j for j in self.attn_layers if j not in self.kv_leader]
        self.num_kv_caches = len(self.cached_layers)
        self.kv_cache_slot = [self.cached_layers.index(self.kv_leader.get(j, j))
                              if j in self.attn_layers else -1
                              for j in range(config.n_layer)]
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i, uses_attention=(i % 2 == 1),
                                      shares_kv=(i in self.kv_leader))
                                for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Value embeddings
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        # A follower has no c_v to add a value residual into and no ve_gate to weight it, so
        # its table would be an unreachable parameter counted against num_params_total and
        # optimised without a gradient. has_ve selects 1, 3, 5, 7 here; 3 and 7 are the
        # followers, so only layers 1 and 5 keep a table.
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer)
            if has_ve(i, config.n_layer) and i not in self.kv_leader
        })
        # A direct token-to-depth pathway at ALL FOUR of the CACHE-FREE layers. A mixer holds no
        # key or value cache and its retained state is one position of the residual stream
        # whatever is added to its output, so this adds no retained state; a lookup is a
        # gather, which issues no matmul and so is not counted arithmetic. The keys cannot
        # collide with str(layer_index), which is what the value-residual path looks up.
        # They live in THIS ModuleDict deliberately: count_params' census groups
        # model.value_embeds, and setup_optimizer asserts len(list(self.parameters()))
        # against its groups, so a table registered anywhere else would fail that assert
        # before any GPU work -- and this is the group every token-indexed table in this
        # model is optimised in, at embedding_lr rather than under Muon. Layers 0, 2, 4 and 6
        # are the mixer layers (i % 2 == 0), so none of them holds a cache and the pathway
        # extends to the two that did not have a table rather than repeating on the two that
        # did.
        for inject_layer_local in (0, 2, 4, 6):
            self.value_embeds[f"inject{inject_layer_local}"] = nn.Embedding(config.vocab_size,
                                                                           config.n_embd)
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
            if block.attn is not None:
                torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
                if not block.attn.shares_kv:
                    torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
                    torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
                torch.nn.init.zeros_(block.attn.c_proj.weight)
            if block.mixer is not None:
                torch.nn.init.uniform_(block.mixer.mix_self.weight, -s, s)
                torch.nn.init.uniform_(block.mixer.mix_prev.weight, -s, s)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        # Value embeddings. The injection tables start at ZERO, so this model's starting
        # function is exactly the parent's and the tables have to earn their contribution;
        # uniform(-s, s) would inject noise of embedding scale into the residual stream.
        for ve_key, ve in self.value_embeds.items():
            if ve_key.startswith("inject"):
                torch.nn.init.zeros_(ve.weight)
            else:
                torch.nn.init.uniform_(ve.weight, -s, s)
        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
        for block in self.transformer.h:
            if block.attn is not None and block.attn.ve_gate is not None:
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
        # No layer spans the whole context: every layer, whatever the pattern says, spans a
        # QUARTER of it. window_size=(512, 0) admits keys [i - 512, i] inclusive = 513
        # positions, which is also the ring width init_decode_state derives as
        # min(window + 1, max_len), so no cache is allocated at max_len. The last-layer
        # full-context override is dropped for the same reason: a 2048 window makes
        # min(window + 1, max_len) equal max_len and the ring has nothing to bound.
        short_window = long_window // 4
        char_to_window = {"L": (short_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # The LOWER leader-follower pair retains 128 positions instead of 192, 64 fewer.
        # (127, 0) admits keys [i - 127, i] inclusive = 128 positions, and
        # init_decode_state derives min(127 + 1, max_len) = 128 at both measured max_lens.
        # 128 rather than an arbitrary number because every section of that slot stays
        # block-exact: 128 * 1 * 64 = 8192 payload bytes is 16 whole 512-byte blocks and
        # 128 * 1 * 4 * 2 = 1024 scale bytes is 2, so the reading is the extent and not new
        # padding. 128 < 513 keeps ring_is_on True on this slot, so the int8 ring branch stays
        # the only reachable one and both RuntimeError branches stay unreached.
        # Layer 3 is layer 1's FOLLOWER, so its window moves WITH the leader's: a wider
        # follower window would let training and prefill admit positions the step cannot
        # supply from the shared ring. The override names layers by index and is
        # applied after the pattern loop, so the pattern string is not the mechanism. Layers 5
        # and 7 take the SAME 64-position dose below, so the total cut is 128 positions split
        # across the two pairs rather than concentrated on one.
        pair_extent_layers = (1, 3)
        for pair_extent_layer in pair_extent_layers:
            window_sizes[pair_extent_layer] = (127, 0)
        # The UPPER leader-follower pair retains 320 positions instead of 384, 64 fewer.
        # (319, 0) admits keys [i - 319, i] inclusive = 320 positions, and
        # init_decode_state derives min(319 + 1, max_len) = 320 at both measured max_lens.
        # 320 < 513 keeps ring_is_on True, so the int8 ring branch remains the only reachable
        # one and the branch that appends k and v INSIDE the fa3 kernel -- which cannot write
        # an int8 payload or its scales -- stays unreached. 320 rather than an arbitrary
        # number because every section stays block-exact: 320 * 1 * 64 = 20480 payload bytes
        # is 40 whole 512-byte blocks and 320 * 1 * 4 * 2 = 2560 scale bytes is 5, so the
        # reading is the extent and not new padding. Layers 5 and 7 move together, because 7
        # is 5's follower and its prefill call passes its own window over the leader's fresh
        # keys; a follower whose window is wider than its leader's ring would ask for keys the
        # step cannot supply. The lower pair is cut to 128 above.
        for upper_pair_extent_layer_local in (5, 7):
            window_sizes[upper_pair_extent_layer_local] = (319, 0)
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
        # The leader's freshly computed keys and values, carried to the follower above it.
        carry_shared_kv = None
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            # The injected token vector for a cache-free layer that has a table, None
            # otherwise. The mixer lives inside Block.forward, so it is passed in rather
            # than reached for from the Block.
            inject_vec = (self.value_embeds[f"inject{i}"](idx)
                          if f"inject{i}" in self.value_embeds else None)
            x, produced_shared_kv = block(x, ve, cos_sin, self.window_sizes[i],
                                          carry_shared_kv if i in self.kv_leader else None,
                                          inject=inject_vec)
            # The carry must SURVIVE every non-leader layer. A mixer sits between each pair
            # here, so clearing it at layer 2 would hand layer 3 a None.
            if i in self.kv_leader_set:
                carry_shared_kv = produced_shared_kv
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
        assert head_dim % QUANT_GROUP == 0, (
            f"head_dim {head_dim} must be a whole number of {QUANT_GROUP}-channel scale groups")
        mixer_slot_count = len(self.mixer_layers)
        # Retain, per layer, only as many positions as that layer's own window can read.
        # window_size=(left, 0) admits keys [i - left, i] INCLUSIVE, so left + 1 slots.
        # Derived from the CACHED layers, not every attention layer: a follower holds no cache
        # and reads its leader's, so allocating one per attention layer would build four
        # caches and write two.
        cache_ring_widths = [min(self.window_sizes[j][0] + 1, max_len)
                             for j in self.cached_layers]
        st = {
            "max_len": max_len,
            "graph_enabled": bool(graph),
            "seq": torch.zeros(batch, dtype=torch.int32, device=dev),
            "ring_widths": cache_ring_widths,
            # One cache per ATTENTION layer, not one per layer, and its extent is that
            # layer's own window rather than max_len. The payload is int8 and the bf16
            # absmax scales that interpret it are held beside it: a compressed
            # representation and its scales are both retained state, both owned by this
            # object and both resident on the GPU, so both are charged here.
            "kc": [torch.zeros(batch, cache_ring_widths[s], cfg.n_kv_head, head_dim,
                               dtype=torch.int8, device=dev)
                   for s in range(self.num_kv_caches)],
            "vc": [torch.zeros(batch, cache_ring_widths[s], cfg.n_kv_head, head_dim,
                               dtype=torch.int8, device=dev)
                   for s in range(self.num_kv_caches)],
            "kc_scale": [torch.zeros(batch, cache_ring_widths[s], cfg.n_kv_head,
                                     head_dim // QUANT_GROUP, **kw)
                         for s in range(self.num_kv_caches)],
            "vc_scale": [torch.zeros(batch, cache_ring_widths[s], cfg.n_kv_head,
                                     head_dim // QUANT_GROUP, **kw)
                         for s in range(self.num_kv_caches)],
            # The retained history a mixer layer uses in place of a cache: one position,
            # whatever max_len is, owned by this object and resident on the GPU.
            "shift": [torch.zeros(batch, 1, cfg.n_embd, **kw)
                      for _ in range(mixer_slot_count)],
        }
        if any(w < max_len for w in cache_ring_widths):
            # How many ring slots are valid: min(positions written since the last reset,
            # ring width). Once a slot index stops equalling an absolute position, `seq`
            # can no longer serve as cache_seqlens.
            st["ring_count"] = torch.zeros(batch, dtype=torch.int32, device=dev)
        return st

    def reset_decode_state(self, state):
        """Return `state` to position 0 so the same buffers serve another request.

        A fresh state per request would invalidate an address-bound CUDA graph. Stale bytes
        above the position are unreachable -- attention reads only `cache_seqlens` keys -- so
        zeroing them would be timed work serving never does.
        """
        state["seq"].zero_()
        # Unlike the key/value caches, the shift state is READ on the first position of a
        # request, so a fresh request would otherwise see the previous one's last token.
        for shift_state in state["shift"]:
            shift_state.zero_()
        if "ring_count" in state:
            # The ring's valid-slot count is the other thing a new request must not inherit.
            state["ring_count"].zero_()
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

        x = norm(self.transformer.wte(idx))
        x0 = x
        # The leader's freshly computed keys and values, carried to the follower above it.
        # Without the carry a follower has nothing causal to read on the prefill.
        carry_shared_k = carry_shared_v = None
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            attn = block.attn
            h = norm(x)
            if attn is None:
                mixer_index = self.mixer_layers.index(i)
                shift_state = state["shift"][mixer_index]
                if prefill:
                    shift_input = torch.cat([shift_state, h[:, :-1]], dim=1)
                else:
                    shift_input = shift_state
                mixer_out = block.mixer.mix_self(h) + block.mixer.mix_prev(shift_input)
                # The same injection, in the same place, as in forward -- decode_tv is
                # measured against forward, so it has to land on both paths. It is a pure
                # function of the current token: nothing is retained, reset_decode_state
                # needs no change, and the eager and captured paths run this same code, the
                # gather reading the static idx buffer the graph already copies into. In
                # prefill idx is (B, Tn) so the gather is (B, Tn, n_embd) and in a step it is
                # (B, 1); mixer_out has the matching shape on both.
                inject_key_local = f"inject{i}"
                if inject_key_local in self.value_embeds:
                    mixer_out = mixer_out + self.value_embeds[inject_key_local](idx)
                # Read the retained position, THEN overwrite it. A reversed order would read
                # the current token as the previous one; the order is recorded in eager and
                # inside a captured graph alike.
                shift_state.copy_(h[:, -1:])
                x = x + mixer_out
                x = x + block.mlp(norm(x))
                continue
            cache_slot = self.kv_cache_slot[i]
            kc, vc = state["kc"][cache_slot], state["vc"][cache_slot]
            kc_scale_local, vc_scale_local = (state["kc_scale"][cache_slot],
                                              state["vc_scale"][cache_slot])
            ring_extent = state["ring_widths"][cache_slot]
            ring_is_on = ring_extent < state["max_len"]
            q = attn.c_q(h).view(B, Tn, attn.n_head, attn.head_dim)
            if attn.shares_kv:
                # A FOLLOWER: its own query against the leader's key/value basis. cache_slot
                # is the leader's slot, so the tensors read here are the ones the leader
                # wrote earlier in this same layer loop.
                q = norm(apply_rotary_emb(q, cos, sin))
                if prefill:
                    # Bit-for-bit the leader's own prefill call: same causal mask, same
                    # window, over the leader's freshly computed keys and values.
                    y = fa3.flash_attn_func(q, carry_shared_k, carry_shared_v, causal=True,
                                            window_size=self.window_sizes[i])
                elif ring_is_on:
                    # The leader has already written this position's ring slot and advanced
                    # ring_count this step, so read the valid slots and write nothing. The
                    # SAME unpack the leader uses, so the two layers cannot disagree about
                    # the same stored keys.
                    y = fa3.flash_attn_with_kvcache(
                        q,
                        quant_ring_unpack(kc, kc_scale_local, QUANT_GROUP),
                        quant_ring_unpack(vc, vc_scale_local, QUANT_GROUP),
                        k=None, v=None,
                        cache_seqlens=state["ring_count"],
                        causal=False, window_size=(-1, -1),
                        num_splits=1)
                else:
                    # No ring at this max_len: the leader appended at `seq` and attended over
                    # seq + 1 keys, so read that same set without appending a second time.
                    y = fa3.flash_attn_with_kvcache(q, kc, vc, k=None, v=None,
                                                    cache_seqlens=state["seq"] + 1,
                                                    causal=True,
                                                    window_size=self.window_sizes[i],
                                                    num_splits=1)
                x = x + attn.c_proj(y.contiguous().view(B, Tn, -1))
                x = x + block.mlp(norm(x))
                continue
            k = attn.c_k(h).view(B, Tn, attn.n_kv_head, attn.head_dim)
            v = attn.c_v(h).view(B, Tn, attn.n_kv_head, attn.head_dim)
            if ve is not None:
                ve = ve.view(B, Tn, attn.n_kv_head, attn.head_dim)
                gate = 2 * torch.sigmoid(attn.ve_gate(h[..., :attn.ve_gate_channels]))
                v = v + gate.unsqueeze(-1) * ve
            q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
            q, k = norm(q), norm(k)

            if prefill and ring_is_on:
                # No `kc[:, :Tn]` slice exists when the ring is narrower than the prefill
                # width, so attend over the fresh tensors -- the same values the full-length
                # branch reads back out of the slice it has just written -- and then retain
                # only the tail the steps can read.
                y = fa3.flash_attn_func(q, k, v, causal=True,
                                        window_size=self.window_sizes[i])
                ring_write_start = max(0, Tn - ring_extent)
                ring_write_slots = (torch.arange(ring_write_start, Tn, device=k.device)
                                    % ring_extent)
                # Payload and scale are written at the SAME slots, in adjacent lines: a
                # scale that does not correspond to the payload beside it would still read
                # as a number and the error would show only as decode TV.
                k_packed_local, k_scale_new_local = quant_ring_pack(k[:, ring_write_start:],
                                                                   QUANT_GROUP)
                v_packed_local, v_scale_new_local = quant_ring_pack(v[:, ring_write_start:],
                                                                   QUANT_GROUP)
                kc.index_copy_(1, ring_write_slots, k_packed_local)
                kc_scale_local.index_copy_(1, ring_write_slots, k_scale_new_local)
                vc.index_copy_(1, ring_write_slots, v_packed_local)
                vc_scale_local.index_copy_(1, ring_write_slots, v_scale_new_local)
                state["ring_count"].copy_(torch.clamp(state["seq"] + Tn, max=ring_extent))
            elif not prefill and ring_is_on:
                # Append by hand at this position's ring slot, THEN read exactly the valid
                # slots: the current token is inside its own window. Every index is a
                # device tensor, so no Python-visible position is introduced.
                ring_step_slot = (state["seq"] % ring_extent).to(torch.int64)
                k_packed_local, k_scale_new_local = quant_ring_pack(k, QUANT_GROUP)
                v_packed_local, v_scale_new_local = quant_ring_pack(v, QUANT_GROUP)
                kc.index_copy_(1, ring_step_slot, k_packed_local)
                kc_scale_local.index_copy_(1, ring_step_slot, k_scale_new_local)
                vc.index_copy_(1, ring_step_slot, v_packed_local)
                vc_scale_local.index_copy_(1, ring_step_slot, v_scale_new_local)
                state["ring_count"].copy_(torch.clamp(state["seq"] + 1, max=ring_extent))
                # Unmasked over the ring: attention over a key set is permutation
                # invariant once RoPE and the norm are baked into the stored keys, and
                # every retained slot is inside this layer's window. causal=True would
                # mask by slot order, which the ring deliberately breaks.
                y = fa3.flash_attn_with_kvcache(
                    q,
                    quant_ring_unpack(kc, kc_scale_local, QUANT_GROUP),
                    quant_ring_unpack(vc, vc_scale_local, QUANT_GROUP),
                    k=None, v=None,
                    cache_seqlens=state["ring_count"],
                    causal=False, window_size=(-1, -1),
                    num_splits=1)
            elif prefill:
                # Unreachable: every extent is below every max_len the instrument passes
                # (2048, 2049, 513, 514), so ring_is_on is True on every cached layer. This
                # branch wrote bf16 k and v straight into the cache, which an int8 payload
                # with separate scales cannot hold, so it raises rather than silently
                # corrupting the store.
                raise RuntimeError(
                    "full-length prefill write into an int8 ring cache: this branch cannot "
                    "write a quantised payload and its scales")
            else:
                # Unreachable for the same reason, and the one that would be worse: the
                # kernel appends k and v INSIDE the call, in bf16, into an int8 cache.
                raise RuntimeError(
                    "kernel-side k/v append into an int8 ring cache: this branch cannot "
                    "write a quantised payload and its scales")
            # A BARE if, with no else. A mixer between each pair produces nothing, and the
            # mixer branch above has already `continue`d, so the carry survives it; an else
            # clause here would clear it and hand the follower a None.
            if i in self.kv_leader_set:
                carry_shared_k, carry_shared_v = k, v
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
HEAD_DIM = 64           # target head dimension for attention
KV_GROUP_SIZE = 8       # query heads sharing one key/value head (grouped-query attention)
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
QUANT_GROUP = 16        # cached channels per bf16 scale

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
        n_layer=depth, n_head=num_heads, n_kv_head=max(1, num_heads // KV_GROUP_SIZE),
        n_embd=model_dim,
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
