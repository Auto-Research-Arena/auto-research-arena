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
    mlp_hidden: int = 2048


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


# ---------------------------------------------------------------------------
# Compiled elementwise glue for the fused width-1 decode step. Every F.linear stays
# OUTSIDE these: under autocast dynamo would trace the fp32->bf16 weight cast into the
# graph and inductor would re-cast on every call instead of hitting autocast's cache.
# The residual stream dtypes mirror the eager decode body exactly: x0 is bf16, the
# per-layer mixed residual is fp32 (fp32 lambda * bf16 promotes), and every tensor that
# feeds a matmul or the attention kernel is cast back to bf16.
# ---------------------------------------------------------------------------

@torch.compile(dynamic=False, fullgraph=True)
def _g_head0(e, rl, x0l):
    x0 = norm(e)
    m = rl * x0 + x0l * x0
    return x0, m, norm(m).to(torch.bfloat16)


@torch.compile(dynamic=False, fullgraph=True)
def _g_mix(m_prev, out, x0, rl, x0l):
    u = m_prev + out
    m = rl * u + x0l * x0
    return m, norm(m).to(torch.bfloat16)


@torch.compile(dynamic=False, fullgraph=True)
def _g_rope(qk, rope, hd):
    # sign-folded rotary + QK-norm in one kernel. With r = roll(x, hd/2, -1) = cat([x2, x1]),
    # x * cat([cos, cos]) + r * cat([sin, -sin]) == cat([x1*cos + x2*sin, x2*cos - x1*sin]),
    # which is apply_rotary_emb. rms_norm over the last dim of (B,T,nh+nkv,hd) equals
    # norming q and k separately.
    r = torch.roll(qk, hd // 2, -1)
    return norm(qk * rope[..., :hd] + r * rope[..., hd:]).to(torch.bfloat16)


@torch.compile(dynamic=False, fullgraph=True)
def _g_vgate(v, z, ve):
    return torch.addcmul(v, torch.sigmoid(z).unsqueeze(-1), ve, value=2.0).to(torch.bfloat16)


@torch.compile(dynamic=False, fullgraph=True)
def _g_cat(y, fc):
    return torch.cat([y.flatten(2), F.relu(fc).square()], -1).to(torch.bfloat16)


@torch.compile(dynamic=False, fullgraph=True)
def _g_relu2(t):
    return F.relu(t).square().to(torch.bfloat16)


@torch.compile(dynamic=False, fullgraph=True)
def _g_tail(m, out):
    return norm(m + out).to(torch.bfloat16)


@torch.compile(dynamic=False, fullgraph=True)
def _g_cap(logits):
    l = logits.float()
    return 15.0 * torch.tanh(l / 15.0)


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included),
    plus any layer listed in EXTRA_VE_LAYERS."""
    return layer_idx % 2 == (n_layer - 1) % 2 or layer_idx in EXTRA_VE_LAYERS


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
        self.c_fc = nn.Linear(config.n_embd, config.mlp_hidden, bias=False)
        self.c_proj = nn.Linear(config.mlp_hidden, config.n_embd, bias=False)

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
        # GPT-J/PaLM parallel block: the FFN reads the same norm(x) attention reads, so at
        # width-1 decode c_fc rides the block's input projection pack and c_proj rides the
        # attention output pack. A plain bool is invisible to count_params and to
        # nn.Module.__setattr__'s registries.
        self.parallel_ffn = bool(PARALLEL_FFN_BLOCKS[layer_idx])

    def forward(self, x, ve, cos_sin, window_size):
        h = norm(x)
        a = self.attn(h, ve, cos_sin, window_size)
        if self.parallel_ffn:
            return x + a + self.mlp(h)
        x = x + a
        return x + self.mlp(norm(x))


class GPT(nn.Module):
    # Class-level defaults so _decode_body works before the decode packs exist. Both are
    # plain Python attributes: no parameter, no buffer, nothing count_params can see.
    _fast_decode = False
    _dw = None

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

    def _decode_fast(self, idx, state):
        """The fused width-1 step. Same arithmetic as `_decode_body(prefill=False)`, fewer
        kernels: one packed embedding lookup for wte and every value embedding, one input
        projection per block holding q|k|v|ve_gate|c_fc, and one output projection holding
        [c_proj | mlp.c_proj] whose column-concatenation sums attn_out + mlp_out for free.
        All weights are decode-only bf16 copies of the trained weights, built after training.
        """
        cfg = self.config
        B, Tn = idx.size()
        d, nh, nkv = cfg.n_embd, cfg.n_head, cfg.n_kv_head
        hd = d // nh
        P = self._dw
        seq = state["seq"]
        rope = P["rope"].index_select(1, seq.to(torch.int64) if P["seq_i64"] else seq)
        e = F.embedding(idx, P["emb"])                     # wte AND every VE in one lookup
        qk_end = (nh + nkv) * hd
        v_end = (nh + 2 * nkv) * hd
        x0, m, h = _g_head0(e[..., :d], self.resid_lambdas[0], self.x0_lambdas[0])
        out = None
        for i, blk in enumerate(self.transformer.h):
            if i > 0:
                m, h = _g_mix(m, out, x0, self.resid_lambdas[i], self.x0_lambdas[i])
            p = F.linear(h, P["win"][i])                   # q|k|v|gate|c_fc in ONE extern
            # For (1, 1, N) a trailing-dim slice keeps stride 1 and reports contiguous (the
            # size-1 dims are ignored), so these views and head slices cost zero kernels.
            qk = _g_rope(p[..., :qk_end].view(B, Tn, nh + nkv, hd), rope, hd)
            v = p[..., qk_end:v_end].view(B, Tn, nkv, hd)
            o = v_end
            slot = P["slot"].get(i)
            if slot is not None:
                ve = e[..., d + slot * nkv * hd:d + (slot + 1) * nkv * hd].view(B, Tn, nkv, hd)
                v = _g_vgate(v, p[..., o:o + nkv], ve)
                o += nkv
            y = fa3.flash_attn_with_kvcache(
                qk[:, :, :nh], state["kc"][i], state["vc"][i],
                k=qk[:, :, nh:], v=v, cache_seqlens=state["seq"], causal=True,
                window_size=self.window_sizes[i], num_splits=DECODE_NUM_SPLITS)
            if blk.parallel_ffn:
                out = F.linear(_g_cat(y, p[..., o:]), P["wout"][i])   # == attn_out + mlp_out
            else:
                a = F.linear(y.flatten(2), P["wout"][i])
                h2 = _g_tail(m, a)
                out = a + F.linear(_g_relu2(F.linear(h2, P["fc"][i])), P["mp"][i])
        logits = F.linear(_g_tail(m, out), P["lmh"])
        return _g_cap(logits)

    def _decode_body(self, idx, state, prefill):
        """`forward`'s body with the cache spliced in. `prefill` is a Python bool that is
        constant per variant, so the two branches are two graphs, not one per position."""
        if not prefill and self._fast_decode:
            return self._decode_fast(idx, state)
        B, Tn = idx.size()
        if prefill:
            cos, sin = self.cos[:, :Tn], self.sin[:, :Tn]
        else:
            seq_idx = state["seq"].to(torch.int64)
            cos = self.cos.index_select(1, seq_idx)
            sin = self.sin.index_select(1, seq_idx)

        x = norm(self.transformer.wte(idx))
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            attn = block.attn
            h = norm(x)
            q = attn.c_q(h).view(B, Tn, attn.n_head, attn.head_dim)
            k = attn.c_k(h).view(B, Tn, attn.n_kv_head, attn.head_dim)
            v = attn.c_v(h).view(B, Tn, attn.n_kv_head, attn.head_dim)
            if ve is not None:
                ve = ve.view(B, Tn, attn.n_kv_head, attn.head_dim)
                gate = 2 * torch.sigmoid(attn.ve_gate(h[..., :attn.ve_gate_channels]))
                v = v + gate.unsqueeze(-1) * ve
            q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
            q, k = norm(q), norm(k)

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
            a = attn.c_proj(y.contiguous().view(B, Tn, -1))
            if block.parallel_ffn:
                x = x + a + block.mlp(h)
            else:
                x = x + a
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
ASPECT_RATIO = 64       # model_dim = depth * ASPECT_RATIO (unused: MODEL_DIM is pinned)
MODEL_DIM = 512         # PIN d. depth * ASPECT_RATIO would silently give 256 at depth 4
HEAD_DIM = 128          # target head dimension for attention
WINDOW_PATTERN = "SSLL" # sliding window pattern: L=full, S=half context
MLP_HIDDEN = 3072       # uniform FFN width. 6144 saturates the FLOPs ceiling; 4608
                        # spends 0.842x of it, buying ~1.16x optimizer updates in the same
                        # 600s and cutting 12.6 MB of the 67.1 MB a decode step streams
EXTRA_VE_LAYERS = (2,)  # gate insurance: a third value-embedding table, +1 decode kernel
PARALLEL_FFN_BLOCKS = (True, True, True, True)  # per-block: FFN reads the attention's norm
DECODE_NUM_SPLITS = 1   # flash-attention kvcache split count (pinned; see _decode_body)
FAST_DECODE_TV_LIMIT = 0.015  # acceptance threshold for the fused decode path. Measured
                              # TV(fast, eager) = 0.0113 on the fully trained model and
                              # 0.0052 at 60 steps, with TV(eager, forward) = 0.000000 on
                              # the same positions: pure bf16 reordering noise, since the
                              # output pack fuses two roundings of (attn_out + mlp_out) into
                              # one. TV is a metric, so accepting at 0.015 bounds the graded
                              # max at the eager path's own 0.0238 + 0.015 = 0.039 < 0.05.

# Optimization
TOTAL_BATCH_SIZE = 2**17 # ~131K tokens per optimizer step
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.033       # learning rate for matrix parameters (Muon). Measured on this
                        # rung: 0.029 -> val_bpb 1.050917, 0.031 -> 1.050088, so the
                        # displacement hedge points the wrong way here and the local
                        # gradient is -0.0004 bpb per +0.001 of LR
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# Model size
DEPTH = 4               # number of transformer layers
DEVICE_BATCH_SIZE = 64  # per-device batch size (reduce if OOM)
assert len(PARALLEL_FFN_BLOCKS) >= DEPTH, "one parallel-FFN flag per block"

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
    model_dim = MODEL_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN, mlp_hidden=MLP_HIDDEN,
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

def get_muon_momentum(tokens_seen):
    # Clocked in tokens, not steps, so a change of TOTAL_BATCH_SIZE keeps the same wall-clock
    # warmup. 157,286,400 = 300 * 2**19 reproduces the reference's 300-step warmup exactly.
    frac = min(tokens_seen / 157_286_400, 1)
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
    muon_momentum = get_muon_momentum(tokens_consumed)
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
# Decode packs and the fidelity self-check. Everything below is post-training: it builds
# bf16 decode-only copies of the trained weights (no new parameters, no trained-apart
# weights), verifies the fused width-1 step against the eager decode body on the training
# batch already in hand, and only then enables it. Never reads the validation split, never
# resets the peak-memory counter, never prints a METRICS_JSON line.
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_decode_packs(m):
    cfg = m.config
    d, nkv = cfg.n_embd, cfg.n_kv_head
    dev = m.transformer.wte.weight.device
    ve_keys = sorted(m.value_embeds.keys(), key=int)
    P = {"slot": {int(k): j for j, k in enumerate(ve_keys)},
         "win": [], "wout": [], "fc": [], "mp": []}
    # Build every pack from .float() copies and cast once at the end.
    P["emb"] = torch.cat([m.transformer.wte.weight.float()] +
                         [m.value_embeds[k].weight.float() for k in ve_keys], 1).bfloat16()
    P["rope"] = torch.cat([m.cos.float(), m.cos.float(),
                           m.sin.float(), -m.sin.float()], -1).bfloat16().contiguous()
    P["lmh"] = m.lm_head.weight.float().bfloat16()
    for b in m.transformer.h:
        a = b.attn
        rows = [a.c_q.weight.float(), a.c_k.weight.float(), a.c_v.weight.float()]
        if a.ve_gate is not None:
            g = torch.zeros(a.n_kv_head, d, dtype=torch.float32, device=dev)
            g[:, :a.ve_gate_channels] = a.ve_gate.weight.float()   # exact zeros elsewhere
            rows.append(g)
        if b.parallel_ffn:
            rows.append(b.mlp.c_fc.weight.float())
            P["wout"].append(torch.cat([a.c_proj.weight.float(),
                                        b.mlp.c_proj.weight.float()], 1).bfloat16())
            P["fc"].append(None)
            P["mp"].append(None)
        else:
            P["wout"].append(a.c_proj.weight.float().bfloat16())
            P["fc"].append(b.mlp.c_fc.weight.float().bfloat16())
            P["mp"].append(b.mlp.c_proj.weight.float().bfloat16())
        P["win"].append(torch.cat(rows, 0).bfloat16())
    # One kernel less per step if index_select accepts the int32 cache position directly.
    # Decided once here, so both the eager and the captured path do the same thing.
    try:
        P["rope"].index_select(1, torch.zeros(1, dtype=torch.int32, device=dev))
        P["seq_i64"] = False
    except Exception:                          # noqa: BLE001 -- capability probe
        P["seq_i64"] = True
    m._dw = P                                  # a plain dict: invisible to count_params


_m = getattr(model, "_orig_mod", model)        # flags live on the eager module
try:
    build_decode_packs(_m)
    toks = x[0:1, :34].clone()                 # the training batch in hand, not the val split
    outs = {}
    with torch.no_grad(), autocast_ctx:
        ref = _m(toks[:, :33]).float()          # forward row j predicts token j+1
        for flag in (False, True):
            _m._fast_decode = flag
            st = _m.init_decode_state(1, 64, graph=False)
            _m.decode_step(toks[:, :25], st)    # prefill, tokens 0..24
            lg = [_m.decode_step(toks[:, t:t + 1], st)[0].float() for t in range(25, 33)]
            outs[flag] = torch.cat([t.view(1, -1) for t in lg], 0)
            del st
    def _tv(a, b):
        return (0.5 * (a.softmax(-1) - b.softmax(-1)).abs().sum(-1)).max().item()
    ref_tail = ref[0, 25:33, :]                 # forward's rows for the same 8 predictions
    tv_fs = _tv(outs[True], outs[False])
    print(f"decode selfcheck: TV(fast,slow)={tv_fs:.6f} "
          f"TV(slow,fwd)={_tv(outs[False], ref_tail):.6f} "
          f"TV(fast,fwd)={_tv(outs[True], ref_tail):.6f} "
          f"TV(slow,fwd_off1)={_tv(outs[False], ref[0, 24:32, :]):.6f} "
          f"maxabs(fast-slow)={(outs[True] - outs[False]).abs().max().item():.6f}")
    _m._fast_decode = bool(tv_fs < FAST_DECODE_TV_LIMIT)
    # Exercise the fast path eagerly BEFORE any capture, so the compiled glue is warm: a
    # first dynamo trace inside torch.cuda.graph() fails the capture silently. Under
    # autocast, exactly like every path prepare.py measures.
    with torch.no_grad(), autocast_ctx:
        st = _m.init_decode_state(1, 8, graph=True)
        _m.decode_step(toks[:, :1], st)
        _m.decode_step(toks[:, 1:2], st)
        _g = st.get("graph")
        print(f"graph captured={getattr(_g, 'captured', None)} "
              f"reason={getattr(_g, 'reason', '')!r}")
        del st, _g
except Exception as exc:                        # noqa: BLE001 -- recorded, then paid for in ms
    import traceback
    _m._fast_decode = False
    print(f"decode selfcheck FAILED, falling back to eager path: {type(exc).__name__}: {exc}")
    traceback.print_exc()
print(f"fast_decode = {_m._fast_decode}")

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
