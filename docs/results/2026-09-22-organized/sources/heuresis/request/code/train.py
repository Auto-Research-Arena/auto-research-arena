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
    vocab_size: int = 8192
    n_head: int = 12
    n_kv_head: int = 4
    n_embd: int = 768
    head_dim: int = 64
    mlp_dim: int = 5184
    block_pattern: str = "AMA"   # one char per block: A = attention+MLP, M = MLP only
    windows: tuple = (2048, 2048)  # one left-window per 'A' block
    n_layer: int = 0              # derived from block_pattern

    def __post_init__(self):
        self.block_pattern = self.block_pattern.upper()
        assert all(c in "AM" for c in self.block_pattern)
        self.n_layer = len(self.block_pattern)
        assert self.n_head * self.head_dim == self.n_embd
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        assert len(self.windows) == self.block_pattern.count("A")


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim
        self.qk_heads = config.n_head + config.n_kv_head
        # Fused q|k|v projection: one GEMM, one dispatch node.
        self.c_qkv = nn.Linear(self.n_embd, (self.n_head + 2 * self.n_kv_head) * self.head_dim,
                               bias=False)
        self.c_proj = nn.Linear(self.n_head * self.head_dim, self.n_embd, bias=False)

    def project(self, h, ve, rope, perm):
        """Shared by `forward` and `_decode_body` so the two paths cannot drift."""
        hd = self.head_dim
        qkv = self.c_qkv(h)
        split = self.qk_heads * hd
        qk = qkv[..., :split].unflatten(-1, (self.qk_heads, hd))
        v = qkv[..., split:].unflatten(-1, (self.n_kv_head, hd))
        # Value residual (ResFormer), ungated: one add.
        v = v + ve
        # Half-swap rope on q and k jointly. Algebraically identical to
        # cat([x1*cos + x2*sin, -x1*sin + x2*cos]) with the table carrying [cos|cos] and
        # [sin|-sin] and `perm` swapping the halves.
        cos2, sin2 = rope[..., :hd], rope[..., hd:]
        qk = torch.addcmul(qk * cos2, qk.index_select(-1, perm), sin2)
        qk = norm(qk)
        q = qk[:, :, :self.n_head]
        k = qk[:, :, self.n_head:]
        return q, k, v


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.mlp_dim, bias=False)
        self.c_proj = nn.Linear(config.mlp_dim, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, has_attn):
        super().__init__()
        self.attn = CausalSelfAttention(config) if has_attn else None
        self.mlp = MLP(config)


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_attn = config.block_pattern.count("A")
        self.kv_dim = config.n_kv_head * config.head_dim
        # One entry per ATTENTION layer, so the analytic span term stays right.
        self.window_sizes = [(w, 0) for w in config.windows]
        self.transformer = nn.ModuleDict({
            # Fused token table: the residual-stream columns plus one value-embedding block
            # per attention layer. A gather dispatches no matmul, so the extra columns are
            # FLOPs-free capacity, and one gather is one node.
            "wte": nn.Embedding(config.vocab_size, config.n_embd + self.n_attn * self.kv_dim),
            "h": nn.ModuleList([Block(config, c == "A") for c in config.block_pattern]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Kept (empty) because prepare.count_params and estimate_flops_analytic index it.
        self.value_embeds = nn.ModuleDict({})
        # Rotary embeddings, packed as [cos|cos | sin|-sin] in one table.
        self.rotary_seq_len = config.sequence_len * 10
        rope, rope_perm = self._precompute_rope(self.rotary_seq_len, config.head_dim)
        self.register_buffer("rope", rope, persistent=False)
        self.register_buffer("rope_perm", rope_perm, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        # Embedding: residual columns ~ N(0,1), value-embedding columns ~ U(-s, s), which is
        # exactly how the separate tables were initialised before they were fused.
        w = self.transformer.wte.weight
        torch.nn.init.normal_(w[:, :n_embd], mean=0.0, std=1.0)
        if w.size(1) > n_embd:
            torch.nn.init.uniform_(w[:, n_embd:], -s, s)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # Transformer blocks
        for block in self.transformer.h:
            if block.attn is not None:
                torch.nn.init.uniform_(block.attn.c_qkv.weight, -s, s)
                torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        # Buffers registered on the meta device hold garbage after to_empty(): recompute.
        rope, rope_perm = self._precompute_rope(self.rotary_seq_len, self.config.head_dim)
        self.rope, self.rope_perm = rope, rope_perm
        # Cast embeddings to bf16
        self.transformer.wte.to(dtype=torch.bfloat16)

    def _precompute_rope(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos2 = torch.cat([cos, cos], dim=-1).bfloat16()
        sin2 = torch.cat([sin, -sin], dim=-1).bfloat16()
        table = torch.cat([cos2, sin2], dim=-1)[None, :, None, :]
        half = head_dim // 2
        perm = torch.cat([torch.arange(half, head_dim, device=device),
                          torch.arange(0, half, device=device)]).to(torch.int32)
        return table, perm

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
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(resid_params) + len(x0_params))
        # Scale LR ∝ 1/√dmodel (tuned at 768 dim)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        # The fused qkv matrix takes the LR its three unfused parts took: scale 1.0.
        qkv_shapes = {block.attn.c_qkv.weight.shape
                      for block in self.transformer.h if block.attn is not None}
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
                lr_scale=(1.0 if shape in qkv_shapes else None),
            ))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def _embed(self, idx):
        e = self.transformer.wte(idx)
        return norm(e[..., :self.config.n_embd]), e

    def _value_embed(self, e, j):
        lo = self.config.n_embd + j * self.kv_dim
        return e[..., lo:lo + self.kv_dim].unflatten(-1, (self.config.n_kv_head,
                                                          self.config.head_dim))

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        assert T <= self.rope.size(1)
        rope = self.rope[:, :T]

        x, e = self._embed(idx)
        x0 = x
        rl = self.resid_lambdas.to(torch.bfloat16)
        xl = self.x0_lambdas.to(torch.bfloat16)
        j = 0
        for i, block in enumerate(self.transformer.h):
            x = torch.addcmul(x * rl[i], x0, xl[i])
            attn = block.attn
            if attn is not None:
                q, k, v = attn.project(norm(x), self._value_embed(e, j), rope, self.rope_perm)
                y = fa3.flash_attn_func(q, k, v, causal=True, window_size=self.window_sizes[j])
                x = x + attn.c_proj(y.contiguous().view(B, T, -1))
                j += 1
            x = x + block.mlp(norm(x))
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
        """Preallocate every ATTENTION layer's cache, and keep the write position ON DEVICE.

        Both give the per-step call constant shapes and no Python-visible position: a lazy
        cache slot is a new shape mid-loop, and a Python `pos` int makes Dynamo guard on its
        value and trip `recompile_limit`. `seq` is int32, advanced with `add_()`, and read by
        the attention kernel as `cache_seqlens`.

        `graph=False` must run the step eagerly and capture nothing. It is how the instrument
        reads cache bytes without a graph's private pool in them, so a candidate that ignores
        the flag reports its own pool as cache and is charged for it.
        """
        assert max_len <= self.rope.size(1), f"max_len {max_len} exceeds rotary table"
        cfg = self.config
        dev = self.transformer.wte.weight.device
        kw = {"dtype": torch.bfloat16, "device": dev}
        return {
            "max_len": max_len,
            "graph_enabled": bool(graph),
            "seq": torch.zeros(batch, dtype=torch.int32, device=dev),
            "kc": [torch.zeros(batch, max_len, cfg.n_kv_head, cfg.head_dim, **kw)
                   for _ in range(self.n_attn)],
            "vc": [torch.zeros(batch, max_len, cfg.n_kv_head, cfg.head_dim, **kw)
                   for _ in range(self.n_attn)],
        }

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
            rope = self.rope[:, :Tn]
        else:
            rope = self.rope.index_select(1, state["seq"])

        x, e = self._embed(idx)
        x0 = x
        rl = self.resid_lambdas.to(torch.bfloat16)
        xl = self.x0_lambdas.to(torch.bfloat16)
        j = 0
        for i, block in enumerate(self.transformer.h):
            x = torch.addcmul(x * rl[i], x0, xl[i])
            attn = block.attn
            if attn is not None:
                q, k, v = attn.project(norm(x), self._value_embed(e, j), rope, self.rope_perm)
                kc, vc = state["kc"][j], state["vc"][j]
                if prefill:
                    kc[:, :Tn] = k
                    vc[:, :Tn] = v
                    y = fa3.flash_attn_func(q, kc[:, :Tn], vc[:, :Tn], causal=True,
                                            window_size=self.window_sizes[j])
                else:
                    # Appends k,v at cache_seqlens and attends over the whole preallocated
                    # cache, so there is no `kc[:, start:end]` slice whose bounds depend on
                    # the step number. num_splits is a reduction-order choice, and the
                    # agreement check in prepare.py still has to pass with it pinned.
                    y = fa3.flash_attn_with_kvcache(q, kc, vc, k=k, v=v,
                                                    cache_seqlens=state["seq"], causal=True,
                                                    window_size=self.window_sizes[j],
                                                    num_splits=DECODE_NUM_SPLITS)
                x = x + attn.c_proj(y.contiguous().view(B, Tn, -1))
                j += 1
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
            print(f"[decode] graph capture skipped: {self.reason}", flush=True)
            return
        try:
            self._capture()
            self.captured = True
        except Exception as exc:            # noqa: BLE001 -- recorded, then paid for in ms
            self.reason = f"{type(exc).__name__}: {exc}"
            print(f"[decode] graph capture FAILED: {self.reason}", flush=True)
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
        lr_scale = group.get("lr_scale") or min(max(1.0, shape[-2] / shape[-1])**0.5,
                                                MUON_LR_SCALE_CAP)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * lr_scale)
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

# Model architecture: attention sublayers are decoupled from MLP sublayers. 'A' is a block
# with attention + MLP, 'M' is MLP only. An attention sublayer is ~11 ATen nodes in the
# graph-replayed width-1 decode step and an MLP sublayer is ~6, so dropping one attention
# while keeping the MLP is the finest-grained latency cut available.
CONFIG_NAME = "N2"
CONFIGS = {
    # name:      model_dim, head_dim, n_kv_head, pattern, windows,                   mlp_dim
    "N2":  dict(model_dim=768, head_dim=64,  n_kv_head=4, pattern="AMA",
                windows=(2048, 2048), mlp_dim=5184),
    "N2S": dict(model_dim=768, head_dim=64,  n_kv_head=4, pattern="AA",
                windows=(2048, 2048), mlp_dim=7808),
    "N4":  dict(model_dim=768, head_dim=64,  n_kv_head=4, pattern="AMAM",
                windows=(2048, 2048), mlp_dim=3904),
    "R3":  dict(model_dim=512, head_dim=128, n_kv_head=2, pattern="AAA",
                windows=(1024, 2048, 2048), mlp_dim=9088),
}
DECODE_NUM_SPLITS = 32  # num_splits for the width-1 flash_attn_with_kvcache call

# Optimization
TOTAL_BATCH_SIZE = 2**17 # ~131K tokens per optimizer step
EMBEDDING_LR = 0.48     # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.0032 # learning rate for lm_head (Adam)
MATRIX_LR = 0.029       # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.85, 0.95) # Adam beta1, beta2
WARMUP_STEPS = 20       # step-based LR warmup
WARMDOWN_RATIO = 0.375  # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial
MUON_LR_SCALE_CAP = 2.0 # cap on Muon's per-shape sqrt(fan-out/fan-in) LR scale

# Model size
DEVICE_BATCH_SIZE = 64  # per-device batch size (reduce if OOM)

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

def build_model_config(name):
    spec = CONFIGS[name]
    model_dim, head_dim = spec["model_dim"], spec["head_dim"]
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_head=model_dim // head_dim, n_kv_head=spec["n_kv_head"],
        n_embd=model_dim, head_dim=head_dim, mlp_dim=spec["mlp_dim"],
        block_pattern=spec["pattern"], windows=spec["windows"],
    )

config = build_model_config(CONFIG_NAME)
print(f"Model config [{CONFIG_NAME}]: {asdict(config)}")

with torch.device("meta"):
    model = GPT(config)
model.to_empty(device=device)
model.init_weights()

# Pre-flight cap check: both identities are exact for this model, so a configuration that
# would fail the harness's caps fails here instead, before any GPU work is charged.
_params = sum(p.numel() for p in model.parameters())
_flops = model.estimate_flops()
print(f"[preflight] params={_params:,} analytic_flops={_flops:,}")
assert _params <= 50_332_176, f"param cap: {_params:,}"
assert _flops <= 239_078_400, f"flops cap: {_flops:,}"

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

def get_lr_multiplier(progress, step):
    if step < WARMUP_STEPS:
        return (step + 1) / WARMUP_STEPS
    if progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    frac = (progress - (1.0 - WARMDOWN_RATIO)) / WARMDOWN_RATIO     # 0 -> 1
    return FINAL_LR_FRAC + (1.0 - FINAL_LR_FRAC) * (1.0 - frac ** 0.5)

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
    lrm = get_lr_multiplier(progress, step)
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
print(f"config:           {CONFIG_NAME}")
