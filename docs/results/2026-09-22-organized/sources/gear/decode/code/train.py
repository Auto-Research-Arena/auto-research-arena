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

import math

import triton
import triton.language as tl


@triton.jit
def _decode_attn_walk(k_ptr, v_ptr, kc_ptr, vc_ptr, seq_ptr, q_ptr,
                      m_ptr, l_ptr, acc_ptr,
                      HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr, SPLITS: tl.constexpr,
                      N_KV_HEADS: tl.constexpr, WINDOW: tl.constexpr, SCALE: tl.constexpr):
    """One (head, split): append at the position if split 0, then walk this split's chunk."""
    h = tl.program_id(0)
    s = tl.program_id(1)
    pos = tl.load(seq_ptr).to(tl.int32)
    d = tl.arange(0, HEAD_DIM)
    row = h * HEAD_DIM + d

    if s == 0:
        tl.store(kc_ptr + pos * (N_KV_HEADS * HEAD_DIM) + row,
                 tl.load(k_ptr + row))
        tl.store(vc_ptr + pos * (N_KV_HEADS * HEAD_DIM) + row,
                 tl.load(v_ptr + row))

    q = tl.load(q_ptr + row).to(tl.float32) * SCALE
    lo = tl.maximum(0, pos - WINDOW)
    per = (pos - lo + SPLITS - 1) // SPLITS
    start = lo + s * per
    end = tl.minimum(lo + (s + 1) * per, pos)

    m = float("-inf")
    l = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    for base in range(start, end, BLOCK_N):
        offs = base + tl.arange(0, BLOCK_N)
        live = offs < end
        rows = offs[:, None] * (N_KV_HEADS * HEAD_DIM) + h * HEAD_DIM + d[None, :]
        keys = tl.load(kc_ptr + rows, mask=live[:, None], other=0.0).to(tl.float32)
        scores = tl.sum(keys * q[None, :], axis=1)
        scores = tl.where(live, scores, float("-inf"))
        m_new = tl.maximum(m, tl.max(scores))
        alpha = tl.exp(m - m_new)
        probs = tl.exp(scores - m_new)
        vals = tl.load(vc_ptr + rows, mask=live[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(probs[:, None] * vals, axis=0)
        l = l * alpha + tl.sum(probs)
        m = m_new

    tl.store(m_ptr + h * SPLITS + s, m)
    tl.store(l_ptr + h * SPLITS + s, l)
    tl.store(acc_ptr + (h * SPLITS + s) * HEAD_DIM + d, acc)


@triton.jit
def _decode_attn_combine(q_ptr, kc_ptr, vc_ptr, seq_ptr, m_ptr, l_ptr, acc_ptr, out_ptr,
                         HEAD_DIM: tl.constexpr, SPLITS: tl.constexpr,
                         N_KV_HEADS: tl.constexpr, SCALE: tl.constexpr):
    """One head: fold the splits together and add the current position.

    The current position is read from the cache, not from registers: the walk kernel stored it
    and kernel ordering on the stream makes that store visible here.
    """
    h = tl.program_id(0)
    pos = tl.load(seq_ptr).to(tl.int32)
    d = tl.arange(0, HEAD_DIM)
    row = h * HEAD_DIM + d

    q = tl.load(q_ptr + row).to(tl.float32) * SCALE
    slot = pos * (N_KV_HEADS * HEAD_DIM) + row
    k_new = tl.load(kc_ptr + slot).to(tl.float32)
    v_new = tl.load(vc_ptr + slot).to(tl.float32)
    m_cur = tl.sum(q * k_new)

    s = tl.arange(0, SPLITS)
    ms = tl.load(m_ptr + h * SPLITS + s)
    ls = tl.load(l_ptr + h * SPLITS + s)
    accs = tl.load(acc_ptr + (h * SPLITS + s)[:, None] * HEAD_DIM + d[None, :])

    top = tl.maximum(m_cur, tl.max(ms))
    w_cur = tl.exp(m_cur - top)
    w = tl.exp(ms - top)
    total = w_cur + tl.sum(w * ls)
    out = v_new * w_cur + tl.sum(accs * w[:, None], axis=0)
    tl.store(out_ptr + row, (out / total).to(tl.bfloat16))



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
        short_window = long_window // 4
# 256 rather than the substrate's 1024, the last step this argument supports. At depth 3 the pattern is S, S, L,
# so this is two of three layers, and the ranked target still cannot move: at 513 positions a query reads keys
# p-256..p and the cache never holds more than 514, so a 256-wide window binds on nothing the no-prompt request
# measures. 512 has been measured three times -- ranked unchanged, tiebreak -3.4 to -4.4 ms, val_bpb -0.0005 to
# -0.0025, counted FLOPs 188.7M -> 100.7M -- and this asks whether the trade continues or whether 256 is finally
# too little context for the short layers to be useful in training.
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
        self._build_projection_group()
        return {
            "max_len": max_len,
            "graph_enabled": bool(graph),
            "seq": torch.zeros(batch, dtype=torch.int32, device=dev),
            "kc": [torch.zeros(batch, max_len, cfg.n_kv_head, head_dim, **kw)
                   for _ in range(cfg.n_layer)],
            "vc": [torch.zeros(batch, max_len, cfg.n_kv_head, head_dim, **kw)
                   for _ in range(cfg.n_layer)],
        }

    def _build_projection_group(self):
        """Every layer's q, k, v and value-gate weights in ONE buffer, layer i a slice of it.

        A layer serving one token issues four matmuls that each produce a few hundred numbers,
        and a launch costs more than that arithmetic does. Stacked along the output dimension
        they are a single GEMV per layer, computing the same products in one launch. The gate
        reads only the first `ve_gate_channels` columns of its input, so its rows are padded
        with zeros to the full width to join the stack; the padded multiply-adds are
        decode-only work the FLOPs probe never sees.

        One allocation for the whole model rather than one per layer: every layer's group has
        the same shape once the non-gated layers are padded to the gated width, so the stack is
        a single (n_layer, out, in) tensor and `group[i]` is a contiguous view. Rows a layer
        does not own are zero, and they are never read -- the slices taken from the output are
        by offset, so a zero row produces a zero the layer ignores.

        Held in bf16, the dtype autocast would have cast the fp32 weights to for the matmul.
        Kept on the model and rebuilt with each request state: it is weights, not request
        state, so it must not land in the cache reading, and rebuilding means it cannot fall
        behind the parameters it is derived from.
        """
        cfg = self.config
        blocks = list(self.transformer.h)
        gate_rows = max((b.attn.ve_gate.weight.size(0) for b in blocks
                         if b.attn.ve_gate is not None), default=0)
        width = 3 * cfg.n_embd + gate_rows
        stack = torch.zeros(len(blocks), width, cfg.n_embd,
                            dtype=torch.bfloat16, device=self.lm_head.weight.device)
        for i, block in enumerate(blocks):
            attn = block.attn
            stack[i, :cfg.n_embd] = attn.c_q.weight
            stack[i, cfg.n_embd:2 * cfg.n_embd] = attn.c_k.weight
            stack[i, 2 * cfg.n_embd:3 * cfg.n_embd] = attn.c_v.weight
            if attn.ve_gate is not None:
                rows = attn.ve_gate.weight.size(0)
                stack[i, 3 * cfg.n_embd:3 * cfg.n_embd + rows,
                      :attn.ve_gate_channels] = attn.ve_gate.weight
        self.__dict__["_projection_stack"] = stack

    def reset_decode_state(self, state):
        """Return `state` to position 0 so the same buffers serve another request.

        A fresh state per request would invalidate an address-bound CUDA graph. Stale bytes
        above the position are unreachable -- attention reads only `cache_seqlens` keys -- so
        zeroing them would be timed work serving never does.
        """
        state["seq"].zero_()
        return state

    def _step_callable(self):
        """`_decode_compute` for the width-1 step, compiled once and reused.

        Eager, a step is ~300 tiny kernels: the residual mix, the rotary rotation, the two
        q/k norms, the value gate and the MLP nonlinearity are all elementwise work on one
        token, and each one is its own launch. Inductor fuses those clusters into far fewer
        kernels for exactly the same arithmetic. `dynamic=False` keeps every shape static, so
        a replayable graph is still what comes out; `fullgraph=False` lets the attention
        custom op break the graph instead of refusing to compile.

        `DECODE_COMPILE_OPTIONS` carries what `mode="max-autotune-no-cudagraphs"` sets, spelled
        out, plus the fusion settings under test. Options rather than a mode because the two
        cannot both be passed, and because every flag then applies to this compilation only --
        the training compile must not move, since `peak_vram_bytes` sits exactly on its
        ceiling. Inductor's own cudagraph pass stays off either way: the capture in
        `_GraphedDecodeStep` owns that job, and that pass copies graph inputs into a private
        pool where the attention op's in-place k/v append would be invisible.
        """
        fn = self.__dict__.get("_decode_step_compiled")
        if fn is None:
            fn = torch.compile(self._decode_compute, dynamic=False,
                               options=DECODE_COMPILE_OPTIONS)
            self.__dict__["_decode_step_compiled"] = fn
        return fn

    def _decode_body(self, idx, state, prefill):
        """`forward`'s body with the cache spliced in. `prefill` is a Python bool that is
        constant per variant, so the two branches are two graphs, not one per position.

        The state's tensors are unpacked here so the compiled step is handed plain tensors
        and lists of tensors, never the dict that also carries the graph object.
        """
        args = (idx, state["seq"], state["kc"], state["vc"])
        if prefill:
            return self._decode_compute(*args, True)
        return self._step_callable()(*args, False)

    def _decode_compute(self, idx, seq, kc_all, vc_all, prefill):
        B, Tn = idx.size()
        if prefill:
            cos, sin = self.cos[:, :Tn], self.sin[:, :Tn]
        else:
            seq_idx = seq.to(torch.int64)
            cos = self.cos.index_select(1, seq_idx)
            sin = self.sin.index_select(1, seq_idx)

        x = norm(self.transformer.wte(idx))
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            attn = block.attn
            h = norm(x)
            # One GEMV for the whole projection group. The pieces are slices of the last
            # dimension of a contiguous row, so the views cost nothing and no copy is made.
            proj = F.linear(h, self._projection_stack[i])
            nq = attn.n_head * attn.head_dim
            nkv = attn.n_kv_head * attn.head_dim
            q = proj[..., :nq].view(B, Tn, attn.n_head, attn.head_dim)
            k = proj[..., nq:nq + nkv].view(B, Tn, attn.n_kv_head, attn.head_dim)
            v = proj[..., nq + nkv:nq + 2 * nkv].view(B, Tn, attn.n_kv_head, attn.head_dim)
            if ve is not None:
                ve = ve.view(B, Tn, attn.n_kv_head, attn.head_dim)
                gate_rows = attn.ve_gate.weight.size(0)
                base = nq + 2 * nkv
                gate = 2 * torch.sigmoid(proj[..., base:base + gate_rows])
                v = v + gate.unsqueeze(-1) * ve
            q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
            q, k = norm(q), norm(k)

            kc, vc = kc_all[i], vc_all[i]
            if prefill:
                kc[:, :Tn] = k
                vc[:, :Tn] = v
                y = fa3.flash_attn_func(q, kc[:, :Tn], vc[:, :Tn], causal=True,
                                        window_size=self.window_sizes[i])
            else:
                # The same split key reduction the library does at num_splits > 1, in two
                # kernels of our own: one walks a chunk of the window per (head, split) and
                # writes a partial max, sum and accumulator, the second folds them and adds
                # the current position. Shapes are constant -- the span is a runtime loop
                # bound inside the kernel, never a slice extent -- and the position is read
                # from the device tensor, so nothing here is Python-visible.
                heads, hd = attn.n_head, attn.head_dim
                scale = 1.0 / math.sqrt(hd)
                part_m = torch.empty(heads, DECODE_ATTN_SPLITS, dtype=torch.float32,
                                     device=q.device)
                part_l = torch.empty(heads, DECODE_ATTN_SPLITS, dtype=torch.float32,
                                     device=q.device)
                part_acc = torch.empty(heads, DECODE_ATTN_SPLITS, hd, dtype=torch.float32,
                                       device=q.device)
                y = torch.empty(B, Tn, heads, hd, dtype=torch.bfloat16, device=q.device)
                _decode_attn_walk[(heads, DECODE_ATTN_SPLITS)](
                    k, v, kc, vc, seq, q, part_m, part_l, part_acc,
                    HEAD_DIM=hd, BLOCK_N=DECODE_ATTN_BLOCK, SPLITS=DECODE_ATTN_SPLITS,
                    N_KV_HEADS=attn.n_kv_head, WINDOW=self.window_sizes[i][0], SCALE=scale)
                _decode_attn_combine[(heads,)](
                    q, kc, vc, seq, part_m, part_l, part_acc, y,
                    HEAD_DIM=hd, SPLITS=DECODE_ATTN_SPLITS,
                    N_KV_HEADS=attn.n_kv_head, SCALE=scale)
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
            self._print_costs()
        except Exception as exc:            # noqa: BLE001 -- recorded, then paid for in ms
            self.reason = f"{type(exc).__name__}: {exc}"
            self.graph, self.static_logits = None, None

    PRINTED_COSTS = set()

    def _print_costs(self):
        """What a step costs and what it is made of, once per cache length.

        Attribution only. Nothing here changes what the instrument reads: the replays are
        reset to a fixed position on every call and `seq` is restored before this returns, and
        it runs in the request probe's first warmup pass, before any timed pass.

        Two positions, because the only part of a step whose work grows with the position is
        attention reading the cache -- pricing a step at the capture position alone prices the
        cheapest step in the request.

        The census is the exact thing the earlier eager per-call table could not give. An eager
        call costs about 20 us of dispatch whatever it computes, which says nothing about the
        same kernel inside a replay; the captured graph's own node list says how many kernels a
        step actually issues, so `step us / kernels` is a real per-kernel budget instead of an
        inference from deltas.
        """
        max_len = self.state["max_len"]
        if max_len in self.PRINTED_COSTS:
            return
        self.PRINTED_COSTS.add(max_len)
        seq0 = self.state["seq"].clone()
        print(f"\n=== decode step cost, max_len {max_len} ===")
        try:
            mid = max(0, min(max_len - 60, max_len // 2))
            for label, position in (("empty cache", 0), (f"cache at {mid}", mid)):
                self._timed_replay(position)
                start, end = torch.cuda.Event(True), torch.cuda.Event(True)
                torch.cuda.synchronize()
                start.record()
                for _ in range(50):
                    self._timed_replay(position)
                end.record()
                torch.cuda.synchronize()
                us = start.elapsed_time(end) * 1000.0 / 50.0
                print(f"  replay, {label:16s} {us:8.2f} us/step  "
                      f"{us * 513 / 1000.0:8.2f} ms per 513-step request")
        except Exception as exc:                  # noqa: BLE001 -- attribution, not behaviour
            print(f"  replay timing unavailable: {type(exc).__name__}: {exc}")
        finally:
            self.state["seq"].copy_(seq0)
            torch.cuda.synchronize()
        self._print_kernel_census()
        try:
            self._print_kernel_costs()
        except Exception as exc:          # noqa: BLE001 -- attribution, not behaviour
            print(f"  single-op costs unavailable: {type(exc).__name__}: {exc}")
        self.state["seq"].copy_(seq0)
        torch.cuda.synchronize()

    def _print_kernel_census(self):
        """Count the kernels in the captured graph, by name, from the graph's own dump."""
        import collections
        import os
        import re
        import tempfile
        try:
            path = os.path.join(tempfile.gettempdir(), f"decode_graph_{id(self)}.dot")
            self.graph.debug_dump(path)
            source = path if os.path.isfile(path) else os.path.join(path, os.listdir(path)[0])
            labels = re.findall(r'label="([^"]*)"', open(source, errors="replace").read())
            names = [label.split("\n")[0].split("(")[0].strip()
                     for label in labels if label.strip()]
            counts = collections.Counter(n for n in names if n)
            print(f"  captured graph: {len(names)} labelled nodes, "
                  f"{len(counts)} distinct")
            for name, count in counts.most_common(30):
                print(f"    {count:4d}  {name[:90]}")
        except Exception as exc:                  # noqa: BLE001 -- attribution, not behaviour
            print(f"  kernel census unavailable: {type(exc).__name__}: {exc}")

    def _graph_time(self, build, iters=50):
        """Capture one call into its own graph and time replays: kernel cost, no dispatch.

        Eager timing cannot answer this -- the attention wrapper alone measured 167 us of
        Python in an earlier launch -- and the step graph only gives the total. Capturing a
        single op and replaying it isolates what that op costs where it actually runs.
        """
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                build()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            build()
        graph.replay()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) * 1000.0 / iters

    def _print_kernel_costs(self):
        """Print the graph-replay cost of the pieces a step is made of."""
        cfg = self.model.config
        dev = self.state["seq"].device
        kw = {"dtype": torch.bfloat16, "device": dev}
        d, hd = cfg.n_embd, cfg.n_embd // cfg.n_head
        row = torch.zeros(1, 1, d, **kw)
        wide = torch.zeros(1, 1, 4 * d, **kw)
        q = torch.zeros(1, 1, cfg.n_head, hd, **kw)
        max_len = self.state["max_len"]
        mid = max(0, min(max_len - 60, max_len // 2))
        cases = []
        for position in (0, mid):
            part_m = torch.empty(cfg.n_head, DECODE_ATTN_SPLITS, dtype=torch.float32,
                                 device=dev)
            part_l = torch.empty(cfg.n_head, DECODE_ATTN_SPLITS, dtype=torch.float32,
                                 device=dev)
            part_acc = torch.empty(cfg.n_head, DECODE_ATTN_SPLITS, hd, dtype=torch.float32,
                                   device=dev)
            out = torch.zeros(1, 1, cfg.n_head, hd, **kw)
            def attn(position=position, part_m=part_m, part_l=part_l, part_acc=part_acc,
                     out=out):
                # the two kernels the step actually runs
                self.state["seq"].fill_(position)
                scale = 1.0 / math.sqrt(hd)
                _decode_attn_walk[(cfg.n_head, DECODE_ATTN_SPLITS)](
                    q, q, self.state["kc"][0], self.state["vc"][0], self.state["seq"], q,
                    part_m, part_l, part_acc, HEAD_DIM=hd, BLOCK_N=DECODE_ATTN_BLOCK,
                    SPLITS=DECODE_ATTN_SPLITS, N_KV_HEADS=cfg.n_kv_head,
                    WINDOW=self.model.window_sizes[0][0], SCALE=scale)
                _decode_attn_combine[(cfg.n_head,)](
                    q, self.state["kc"][0], self.state["vc"][0], self.state["seq"],
                    part_m, part_l, part_acc, out, HEAD_DIM=hd,
                    SPLITS=DECODE_ATTN_SPLITS, N_KV_HEADS=cfg.n_kv_head, SCALE=scale)
                return out
            cases.append((f"attention x1, cache at {position}", attn))
            def attn_lib(position=position):
                # the library call it replaces, at the split value it was measured with
                self.state["seq"].fill_(position)
                return fa3.flash_attn_with_kvcache(
                    q, self.state["kc"][0], self.state["vc"][0],
                    cache_seqlens=self.state["seq"], causal=True,
                    window_size=self.model.window_sizes[0],
                    num_splits=8)
            cases.append((f"library attention x1, cache at {position}", attn_lib))
        for name, act, shape in (
                (f"qkv GEMV ({d}x{d})", row, (d, d)),
                (f"mlp c_fc GEMV ({4 * d}x{d})", row, (4 * d, d)),
                (f"mlp c_proj GEMV ({d}x{4 * d})", wide, (d, 4 * d)),
                (f"lm_head GEMV ({cfg.vocab_size}x{d})", row, (cfg.vocab_size, d))):
            weight = torch.zeros(*shape, **kw)
            cases.append((name, lambda a=act, w=weight: F.linear(a, w)))
        cases.append((f"rms_norm (1x{d})", lambda: norm(row)))
        print("  --- single ops, captured and replayed (us per call) ---")
        for name, build in cases:
            try:
                us = self._graph_time(build)
                print(f"  {name:34s} {us:8.2f} us  {us * 513 / 1000.0:8.2f} ms x513")
            except Exception as exc:          # noqa: BLE001 -- attribution, not behaviour
                print(f"  {name:34s} unavailable: {type(exc).__name__}: {exc}")

    def _timed_replay(self, position):
        """One replay from a fixed cache position. A replay executes the captured `seq.add_(1)`,
        so the position is reset on every call; leaving it advanced runs the request that
        follows off the end of its own cache."""
        self.state["seq"].fill_(position)
        self.graph.replay()

    def _advance(self):
        logits = self.model._decode_body(self.static_idx, self.state, prefill=False)
        self.state["seq"].add_(1)
        return logits

    def _capture(self):
        seq0 = self.state["seq"].clone()
        # Compile before the side stream, not on it: tracing and autotuning run their own
        # kernels and allocate, and that work has no business happening inside the stream the
        # capture is about to record. `seq` is saved and restored around it like the warmups.
        self._advance()
        self.state["seq"].copy_(seq0)
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(self.WARMUP_REPLAYS):
                self.state["seq"].copy_(seq0)
                self._advance()
        torch.cuda.current_stream().wait_stream(stream)

        self.state["seq"].copy_(seq0)
        # keep_graph so the cudaGraph_t survives instantiation and the census can read it.
        try:
            self.graph = torch.cuda.CUDAGraph(keep_graph=True)
        except TypeError:
            self.graph = torch.cuda.CUDAGraph()
        try:
            self.graph.enable_debug_mode()
        except Exception:                      # noqa: BLE001 -- attribution, not behaviour
            pass
        try:
            self.graph.enable_debug_mode()     # so the census below can read the node list
        except Exception:                      # noqa: BLE001 -- attribution, not behaviour
            pass
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

# Serving: how the width-1 decode step is compiled. A step's flat cost is its kernel count
# times about 3.4 us, measured three ways, and at 99 kernels roughly half of them are
# elementwise or reduction work on a single token. These settings are aimed at that half.
DECODE_COMPILE_OPTIONS = {
    # what mode="max-autotune-no-cudagraphs" sets
    "max_autotune": True,
    "coordinate_descent_tuning": True,
    # fuse harder: aggressive_fusion drops the heuristics that keep pointwise groups apart,
    # and a larger fusion budget lets a longer chain land in one kernel
    "aggressive_fusion": True,
    "max_fusion_size": 256,
    # do not split a reduction across two kernels. The split heuristic exists for reductions
    # with too little parallelism to fill the device, which is every reduction here -- but a
    # 512-element row does not need filling, and paying a second launch for it is the worst
    # possible trade on this axis.
    "split_reductions": False,
    # elite/0's contribution: prefer a matmul choice that can absorb its epilogue over the
    # fastest matmul. Measured as worth nothing on its own base (-0.167 ms) while removing about
    # nine launches, which is one of the four results that showed kernel count does not price
    # this axis. Carried because this crossover's base is where it lives.
    "epilogue_fusion_first": True,
}

# Serving: the decode attention's own split. SPLITS chunks of the window are walked in
# parallel, BLOCK_N keys at a time, and folded by a second kernel, at Triton's own choice of warps
# and stages -- two warps costs +6.239 ms at this tiling and eight costs +5.125 ms at the other, so
# four, the default, is the peak of that dial. Sixteen chunks of 32 keys
# puts 64 programs on the device and gives each a full block at 513 positions; eight of 64 wasted
# half a block at 257. Measured as -6.178 ms on another base, where it also gained -23.296 ms on
# the tiebreak, whose walk is four times longer.
DECODE_ATTN_SPLITS = 16
DECODE_ATTN_BLOCK = 32

# Model architecture
ASPECT_RATIO = 160      # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 128          # target head dimension for attention
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context

# Optimization
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.02        # learning rate for matrix parameters (Muon)
# The reported configuration, seventh reading. Six so far: ranked mean 56.68 ms and median 56.49, tiebreak mean
# 65.24 and median 65.11, val_bpb mean 1.03866 over a spread of 0.0004, against a gate of 1.05 and a reference of
# 462.877 and 639.059 ms.
# The next point down the ladder, at the halved batch this base already uses:
#   0.08 -> 1.046950, 0.04 -> 1.041779 (and three more within 0.0002), 0.02 -> 1.039517.
# Monotone so far, so 0.01 says whether the optimum has been passed. The same ladder at the substrate's
# batch runs the other way -- 0.04 -> 1.045405, 0.02 -> 1.048065 -- so what matters is the rate relative
# to the batch, not either alone, which is the ordinary shape for a learning rate and is what the
# substrate half-implements: setup_optimizer rescales every Adam group by 1/sqrt(model_dim/768) and
# leaves the Muon groups at whatever MATRIX_LR says.
# Doubled. This is the one learning rate in the file that is not rescaled for the model being trained:
# setup_optimizer multiplies every Adam group by 1/sqrt(model_dim/768) and prints that it does, but the
# Muon groups take MATRIX_LR as given, and they cover every matrix parameter in the transformer. The
# model is 512 wide rather than 768 and makes 3198 updates rather than 751, so the value it was chosen
# at describes neither. Free on the ranked axis.
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial
# The anneal reaches zero, which four readings show is right. That makes this the reported configuration, whose
# fifteenth reading this is -- and the second-to-last launch of the pre-registered hundred.
# Fourteen readings so far: ranked mean 56.65 ms, tiebreak mean 65.2, val_bpb mean 1.03867 with sd 0.00014, against
# a reference of 462.877 and 639.059 ms and a gate of 1.05.
# The last constant in this file never varied in 87 launches. At 0.0 the schedule anneals the learning rate to
# exactly zero at the end of the clock; at 0.1 it stops at a tenth of the initial value.
#
# The expectation is that this is worse, and the reason is the one thing the run has learned about this schedule:
# the anneal to zero is what sharpens the final loss, and a run that stops annealing early leaves the model at a
# noisier point. It is worth one launch anyway because it is the only knob left unexamined, and because the
# argument that made the batch worth -0.048 bpb at depth 8 -- constants chosen for 751 updates, used at 3198 --
# applies to the schedule's endpoint as much as to its shape.
# Free on the ranked axis.

# Model size
# The model resized for the clock: three layers at 512 wide, and half the batch to match. These are one
# change, because they act through the same quantity -- the number of optimizer updates the fixed 600 s
# buys. Depth 8 at 2**19 tokens an update makes 751 of them; this makes 3198.
#
# Both halves are measured. Depth: quality bottoms out near six layers and depth 3 is the optimum of the
# axis from both sides -- depth 4 is 71.6 ms at val_bpb 1.031, and two layers fails the gate at every
# width that would be faster than three (1.0869 at 512 wide and 40.3 ms, 1.0530 at 768 and 55.0 ms).
# Batch: halving it is worth -0.048 bpb at depth 8 where updates are scarce and -0.0036 at depth 3 where
# they are not, for 0.2-0.3 ms, and both readings replicate to better than 0.0002.
#
# Together they should give the fastest program in this run that is also comfortably eligible: about
# 56.5 ms with the gate clear by 0.008 rather than 0.005.
DEPTH = 3               # number of transformer layers
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
