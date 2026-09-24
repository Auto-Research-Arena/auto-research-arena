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


RMS_EPS = torch.finfo(torch.float32).eps  # what F.rms_norm defaults to, for any input dtype


class RMSNorm(torch.autograd.Function):
    """rms_norm that keeps no activation of its own.

    F.rms_norm on a bf16 input saves a float32 upcast of that input for its backward:
    4 bytes per element, on top of the 2 bytes the value already costs. This form saves
    the normalised OUTPUT plus the per-row scale instead, and the output is already
    retained by whatever consumes it (every call site here feeds a Linear or the
    attention kernel), so the only new bytes are one float32 per row.

    y = x * r with r = rsqrt(mean(x^2) + eps), so dL/dx = r * (g - y * mean(g * y)),
    which needs y and r and never x. Arithmetic is elementwise, so nothing here is
    counted arithmetic.
    """

    @staticmethod
    def forward(ctx, x):
        xf = x.float()
        r = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + RMS_EPS)
        y = (xf * r).to(x.dtype)
        ctx.save_for_backward(y, r)
        return y

    @staticmethod
    def backward(ctx, g):
        y, r = ctx.saved_tensors
        # yf and gf are our own float32 copies, so the result is built in them instead of in three
        # more tensors of the same width. Backward transients are about 357 MB of this model's peak.
        yf, gf = y.float(), g.float()
        m = (gf * yf).mean(-1, keepdim=True)
        gf.sub_(yf.mul_(m)).mul_(r)
        return gf.to(g.dtype)


def norm(x):
    return RMSNorm.apply(x)


SOFTCAP = 15
# Rows of the flattened batch per chunk. The cost of the idea is chunk_rows x 8192 x 4 bytes per
# live float32 tensor, so the width trades bytes against kernel count -- measured at 4096 -> 2048
# as −301,869,056 bytes for +0.48% throughput, and at 2048 -> 1024 as only −98,691,584 for
# −1.96%. With the in-place form below there are two live float32 tensors instead of five, so the
# width can go back up: one chunk per microbatch at DEVICE_BATCH_SIZE 4 costs the same bytes as
# the old five-tensor form at 4096 and launches a fifth of the kernels.
LOSS_CHUNK_TOKENS = 1024
FORWARD_CHUNK_ROWS = 2      # sequences per trunk group. Groups are independent, so their saved
                            # tensors coexist but their TRANSIENTS do not: autograd runs one
                            # group's backward to completion before the next, so halving the
                            # group halves every intermediate that is alive at the peak.


# The head's two wide float32 tensors, replaced by fused kernels over the bf16 logits.
#
# `F.linear(x, weight)` under autocast already returns bf16, and the forward's float32 logits are
# exactly that upcast, so anything derived from the bf16 tensor is derived from the same numbers. What
# changes is what is resident while it happens: 1,024 x 8,192 in float32 is 33,554,432 bytes and the
# backward held two of them at once. Reading the bf16 logits and writing the bf16 gradient holds
# 16,777,216 each instead, and this is the phase that sets the mark -- the head's backward runs first,
# when every block's retention is still live.

@torch.compile(dynamic=False, fullgraph=True)
def _head_lse(zb, softcap, targets):
    """logsumexp and the picked logit, without materialising the softcapped logits in float32."""
    z = zb.float().div_(softcap).tanh_().mul_(softcap)
    return torch.logsumexp(z, -1), z.gather(1, targets.clamp_min(0).unsqueeze(1)).squeeze(1)


@torch.compile(dynamic=False, fullgraph=True)
def _head_grad(zb, lse, targets, g, softcap):
    """(softmax(z) - onehot) * g * dz/du, in one pass over the bf16 logits.

    The onehot subtraction is a comparison against the column index rather than a scatter, so the
    whole expression stays pointwise and lowers without a wide float32 buffer of its own.
    """
    z = zb.float().div_(softcap).tanh_().mul_(softcap)
    p = (z - lse.unsqueeze(1)).exp()
    hit = torch.arange(z.size(1), device=z.device).unsqueeze(0) == targets.unsqueeze(1)
    p = torch.where(hit, p - 1.0, p)
    p = p * (g.unsqueeze(1) * (targets >= 0).unsqueeze(1))
    return (p * (1.0 - (z / softcap).square())).to(zb.dtype)


class SoftcapCrossEntropyHead(torch.autograd.Function):
    """Unembedding plus softcapped cross entropy, retaining neither the logits nor anything wide.

    The logits are vocab-wide -- 8,192 columns -- and were the last large tensor the compiled graph
    kept: 134 MB at this microbatch. Their only reader is this backward, which needs them to rebuild
    softmax(z) and dz/du, so recomputing them from x and the weight frees all of it. The cost is one
    extra unembedding matmul per chunk: 2 * n_embd * vocab = 8.39 MFLOPs per token, against the
    65.01 MFLOPs per token still unspent under the FLOPs ceiling.

    What is saved instead is x -- the final norm's output, which RMSNorm already retains -- the weight,
    the targets, and one float32 logsumexp per row.

    z = softcap * tanh(u / softcap) with u the logits, so dnll/dz = softmax(z) - onehot and
    dz/du = 1 - (z / softcap)**2, and dL/dx = dL/du @ w with dL/dw = dL/du^T @ x.
    """

    @staticmethod
    def forward(ctx, x, weight, targets, softcap):
        lse, picked = _head_lse(F.linear(x, weight), softcap, targets)
        nll = torch.where(targets >= 0, lse - picked, 0.0)
        ctx.save_for_backward(x, weight, targets, lse)
        ctx.softcap = softcap
        return nll

    @staticmethod
    def backward(ctx, g):
        x, weight, targets, lse = ctx.saved_tensors
        du = _head_grad(F.linear(x, weight), lse, targets, g, ctx.softcap)
        return du @ weight, du.mT @ x, None, None


def softcap_cross_entropy(weight, x, targets, reduction):
    """Run the head a chunk of rows at a time, so nothing vocab-wide spans the whole batch.

    prepare.py's evaluation calls the model at a frozen 128x2048 whatever the training microbatch is,
    and one unembedding there would be 262,144 x 8,192. Chunking keeps every wide tensor to
    LOSS_CHUNK_TOKENS rows, and splitting a matmul does not change counted arithmetic: FlopCounterMode
    sums what dispatches.
    """
    rows = x.size(0)
    pieces = [SoftcapCrossEntropyHead.apply(x[i:i + LOSS_CHUNK_TOKENS], weight,
                                            targets[i:i + LOSS_CHUNK_TOKENS], SOFTCAP)
              for i in range(0, rows, LOSS_CHUNK_TOKENS)]
    nll = pieces[0] if len(pieces) == 1 else torch.cat(pieces)
    if reduction == 'none':
        return nll
    return nll.sum() / (targets >= 0).sum().clamp_min(1)


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


def rotary_grad(g, cos, sin):
    """The rotary is a rotation in each pair, so its backward is the inverse rotation."""
    d = g.shape[3] // 2
    g1, g2 = g[..., :d], g[..., d:]
    return torch.cat([g1 * cos - g2 * sin, g1 * sin + g2 * cos], 3)


def rms_apply(x):
    """RMSNorm's forward, returning the row scale so a backward can rebuild the output."""
    xf = x.float()
    r = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + RMS_EPS)
    return (xf * r).to(x.dtype), r


def rms_grad(g, y, r):
    """r * (g - y * mean(g * y)), built in our own float32 copies."""
    yf, gf = y.float(), g.float()
    m = (gf * yf).mean(-1, keepdim=True)
    return gf.sub_(yf.mul_(m)).mul_(r).to(g.dtype)


class MixResid(torch.autograd.Function):
    """The residual mix alone, retaining only its output.

    `z = a*x + b*x0` needs x to form the scalar a's gradient, sum(dz * x), and z is the one tensor
    every consumer in the block can be rebuilt from: x = (z - b*x0) / a recovers the mix's input,
    and the norm that follows is z * r. x0 is one tensor shared by every layer and already
    retained. Elementwise throughout, so nothing here is counted arithmetic.
    """

    @staticmethod
    def forward(ctx, x, x0, a, b):
        z = a * x + b * x0
        ctx.save_for_backward(z, x0, a, b)
        return z

    @staticmethod
    def backward(ctx, gz):
        z, x0, a, b = ctx.saved_tensors
        af, bf = a.float(), b.float()
        x0f = x0.float()
        xf = z.float().sub_(x0f * bf).div_(af)   # x = (z - b*x0) / a, built in the recomputed copy
        gzf = gz.float()
        return ((gzf * af).to(gz.dtype), (gzf * bf).to(x0.dtype),
                (gzf * xf).sum().to(a.dtype), (gzf * x0f).sum().to(b.dtype))


class AttnCore(torch.autograd.Function):
    """Everything from the residual mix's output to the attention output, retaining only out and lse.

    flash-attention's own Function saves q, k, v and its output for its backward: four 512-wide bf16
    tensors per block, 33,554,432 bytes at the probe's 4-row shape and 160 MiB over five blocks. They
    are saved inside the kernel's wrapper, so no rewrite in this file can reach them -- but three of
    the four are reachable another way, because everything that produces them is here. q, k and v are
    the norm of z, three projections, the rotary and two more norms, all of which rebuild from z, and
    `flash_attn_interface` exports `_flash_attn_backward`, which takes q, k, v, out and lse as
    arguments rather than reading them from a ctx.

    So the forward runs the attention under `no_grad` and keeps only out and lse; the backward rebuilds
    q, k and v and calls the kernel's backward directly. 25,165,824 bytes per block, and the only
    counted arithmetic added is the three projections -- 1,572,864 FLOPs per token per block. Nothing
    changes about how the attention itself is measured: the call still goes through `flash_attn_func`,
    which is what prepare.py's probe patches and tallies, once per block exactly as before.
    """

    @staticmethod
    def forward(ctx, z, w_q, w_k, w_v, w_gate, ve, ve_weight, idx, cos, sin,
                n_head, n_kv_head, head_dim, gate_channels, window):
        B, T, C = z.size()
        zf = z.float()
        r = torch.rsqrt(zf.pow(2).mean(-1, keepdim=True) + RMS_EPS)
        y = (zf * r).to(z.dtype)
        q, _ = rms_apply(apply_rotary_emb(F.linear(y, w_q).view(B, T, n_head, head_dim), cos, sin))
        k, _ = rms_apply(apply_rotary_emb(F.linear(y, w_k).view(B, T, n_kv_head, head_dim), cos, sin))
        v = F.linear(y, w_v).view(B, T, n_kv_head, head_dim)
        gate = None
        if w_gate is not None:
            gate = 2 * torch.sigmoid(F.linear(y[..., :gate_channels], w_gate))
            v = v + gate.unsqueeze(-1) * ve.view(B, T, n_kv_head, head_dim)
        scale = head_dim ** -0.5          # what flash_attn_func computes when softmax_scale is None
        with torch.no_grad():
            out, lse = fa3.flash_attn_func(q, k, v, softmax_scale=scale, causal=True,
                                           window_size=window, return_attn_probs=True)
        ctx.save_for_backward(z, r, w_q, w_k, w_v, w_gate, ve_weight, idx, gate, cos, sin, out, lse)
        ctx.meta = (n_head, n_kv_head, head_dim, gate_channels, window, scale)
        ctx.ve_shape = None if ve is None else ve.shape
        return out

    @staticmethod
    def backward(ctx, gout):
        (z, r, w_q, w_k, w_v, w_gate, ve_weight, idx, gate, cos, sin, out, lse) = ctx.saved_tensors
        n_head, n_kv_head, head_dim, gate_channels, window, scale = ctx.meta
        B, T, C = z.size()
        y = (z.float() * r).to(z.dtype)                       # the norm output, rebuilt exactly
        q, rq = rms_apply(apply_rotary_emb(
            F.linear(y, w_q).view(B, T, n_head, head_dim), cos, sin))
        k, rk = rms_apply(apply_rotary_emb(
            F.linear(y, w_k).view(B, T, n_kv_head, head_dim), cos, sin))
        v = F.linear(y, w_v).view(B, T, n_kv_head, head_dim)
        if w_gate is not None:
            v = v + gate.unsqueeze(-1) * F.embedding(idx, ve_weight).view(B, T, n_kv_head, head_dim)
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        fa3._flash_attn_backward(gout, q, k, v, out, lse,
                                 None, None,      # cu_seqlens_q, cu_seqlens_k
                                 None, None,      # sequed_q, sequed_k
                                 None, None,      # max_seqlen_q, max_seqlen_k
                                 dq, dk, dv,
                                 scale, True, window[0], window[1], 0.0, False, 0)
        gq = rotary_grad(rms_grad(dq, q, rq), cos, sin)
        gk = rotary_grad(rms_grad(dk, k, rk), cos, sin)
        del q, k, rq, rk, dq, dk
        gq2 = gq.reshape(B, T, n_head * head_dim)
        gk2 = gk.reshape(B, T, n_kv_head * head_dim)
        gv2 = dv.reshape(B, T, n_kv_head * head_dim)
        dve = dw_gate = None
        dy = gq2 @ w_q
        dy = dy.add_(gk2 @ w_k).add_(gv2 @ w_v)
        yflat = y.reshape(-1, C)
        dw_q = gq2.reshape(-1, n_head * head_dim).mT @ yflat
        dw_k = gk2.reshape(-1, n_kv_head * head_dim).mT @ yflat
        dw_v = gv2.reshape(-1, n_kv_head * head_dim).mT @ yflat
        if w_gate is not None:
            veh = F.embedding(idx, ve_weight).view(B, T, n_kv_head, head_dim)
            dgate = (dv * veh).sum(-1)
            del veh
            dve = (gate.unsqueeze(-1) * dv).reshape(ctx.ve_shape)
            dpre = dgate * gate * (1 - gate / 2)
            dw_gate = dpre.reshape(-1, n_kv_head).mT @ yflat[:, :gate_channels]
            dy[..., :gate_channels] += dpre @ w_gate
        dz = rms_grad(dy, y, r)
        return (dz, dw_q, dw_k, dw_v, dw_gate, dve, None, None, None, None,
                None, None, None, None, None)


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

    def forward(self, z, ve, ve_weight, idx, cos_sin, window_size):
        # The norm, the three projections and the value-embedding gate are one Function so that the
        # norm's output need not be retained; c_proj belongs to MlpIn, which rebuilds the residual.
        B, T, C = z.size()
        cos, sin = cos_sin
        y = AttnCore.apply(z, self.c_q.weight, self.c_k.weight, self.c_v.weight,
                           None if self.ve_gate is None else self.ve_gate.weight,
                           ve, ve_weight, idx, cos, sin,
                           self.n_head, self.n_kv_head, self.head_dim, self.ve_gate_channels,
                           window_size)
        return y.contiguous().view(B, T, -1)


class MlpIn(torch.autograd.Function):
    """Attention's output projection, the residual add, the norm after it and the whole MLP.

    Four things retained one tensor each in the reference, and this keeps none of them. The residual
    `x2 = z + c_proj(att)` and the norm output after it are rebuilt in backward from z, which
    MixResid retains anyway, from att, which is flash-attention's own output and already saved by
    its backward, and from the row scale r. `relu(c_fc(y))**2` is 4*n_embd wide and is rebuilt as
    before. So the block's second narrow tensor per layer -- 8,388,608 bytes at the probe's 4-row
    shape, 41,943,040 across five blocks -- disappears.

    The rebuilt values are bit-identical to the forward's: one extra c_proj matmul,
    2 * n_embd * n_embd = 0.52 MFLOPs per token per block and 2.62 MFLOPs over five, plus the c_fc
    matmul MLPRecompute already paid, against 56.62 MFLOPs per token of slack under the ceiling.
    """

    @staticmethod
    def forward(ctx, z, att, w_proj, w_fc, w_mlp):
        x2 = z + F.linear(att, w_proj)
        x2f = x2.float()
        r = torch.rsqrt(x2f.pow(2).mean(-1, keepdim=True) + RMS_EPS)
        y = (x2f * r).to(x2.dtype)
        h = F.relu(F.linear(y, w_fc)).square()
        o = F.linear(h, w_mlp)
        ctx.save_for_backward(z, att, w_proj, w_fc, w_mlp, r)
        return x2, o

    @staticmethod
    def backward(ctx, gx2, go):
        z, att, w_proj, w_fc, w_mlp, r = ctx.saved_tensors
        wide, narrow = w_fc.size(0), w_fc.size(1)
        # rebuild the residual and the norm output, then drop the residual before the wide tensors
        x2 = z + F.linear(att, w_proj)
        y = (x2.float() * r).to(x2.dtype)
        del x2
        rl = F.relu(F.linear(y, w_fc))         # relu(u): one wide tensor
        h = rl.square()                        # two wide tensors live
        g = go @ w_mlp                         # three; g = dL/dh
        g_w_mlp = go.reshape(-1, narrow).mT @ h.reshape(-1, wide)
        del h                                  # back to two
        g.mul_(rl).mul_(2.0)                   # g = dL/du, built in place
        del rl
        g_w_fc = g.reshape(-1, wide).mT @ y.reshape(-1, narrow)
        dy = g @ w_fc
        del g
        yf = y.float()
        dyf = dy.float()
        m = (dyf * yf).mean(-1, keepdim=True)
        dx2 = dyf.sub_(yf.mul_(m)).mul_(r).to(z.dtype).add_(gx2)
        g_w_proj = dx2.reshape(-1, narrow).mT @ att.reshape(-1, att.size(-1))
        return dx2, dx2 @ w_proj, g_w_proj, g_w_fc, g_w_mlp


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, x0, a, b, ve, ve_weight, idx, cos_sin, window_size):
        z = MixResid.apply(x, x0, a, b)
        att = self.attn(z, ve, ve_weight, idx, cos_sin, window_size)
        x2, o = MlpIn.apply(z, att, self.attn.c_proj.weight,
                            self.mlp.c_fc.weight, self.mlp.c_proj.weight)
        return x2 + o


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
        # Only positions below sequence_len are ever indexed -- forward asserts T <= cos.size(1)
        # and T is at most sequence_len -- so a table ten times that long is nine tenths unused
        # resident bytes. cos and sin are 2 x 2048 x 64 bf16 here instead of 2 x 20480 x 64.
        self.rotary_seq_len = config.sequence_len
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
        # Every matrix parameter to bf16 as well, keeping the two rank-1 scalar vectors in float32.
        # These weights are only ever multiplied in bf16 -- autocast casts them for every matmul --
        # and Muon's own step casts the gradient to bf16 before orthogonalising it, so the arithmetic
        # they take part in does not change. What changes is what is resident: the parameters
        # themselves, Muon's momentum buffer, which mirrors them, and the stacked copies its step
        # builds. Their gradients become bf16 too, which is already true of wte and the value
        # embeddings in the reference.
        self.transformer.h.to(dtype=torch.bfloat16)
        self.lm_head.to(dtype=torch.bfloat16)

    @torch.no_grad()
    def flatten_matrix_params(self):
        """Give Muon's shape groups the stacked layout they are stepped in, permanently.

        `_step_muon` stacks a group's parameters and their gradients on every optimizer step, so
        the step holds two fresh full-group copies -- 10,485,760 bytes each for the three large
        groups here -- on top of the parameters and gradients they duplicate, and then copies the
        result back. Allocating each group as one contiguous buffer up front and binding its
        parameters as views into it makes both stacks the buffers themselves: the same tensors,
        in the same order, so the arithmetic is unchanged, but nothing is allocated or copied.
        The gradient buffer is pre-bound to `.grad`, which autograd accumulates into in place.
        """
        entries = [(mod, name, p) for mod in self.transformer.h.modules()
                   for name, p in list(mod._parameters.items()) if p is not None]
        assert [p for _, _, p in entries] == list(self.transformer.h.parameters())
        self.muon_flat = {}
        for shape in sorted({p.shape for _, _, p in entries}):
            selected = [(mod, name, p) for mod, name, p in entries if p.shape == shape]
            param_buffer = torch.stack([p.detach() for _, _, p in selected]).contiguous()
            grad_buffer = torch.zeros_like(param_buffer)
            for i, (mod, name, _) in enumerate(selected):
                flat = nn.Parameter(param_buffer[i])
                flat.flat_grad = True
                mod._parameters[name] = flat
            self.muon_flat[shape] = (param_buffer, grad_buffer)
        self.bind_matrix_grads()

    @torch.no_grad()
    def bind_matrix_grads(self):
        """Point each matrix parameter's .grad at its row of the group's gradient buffer.

        Separate from flatten_matrix_params because zero_grad(set_to_none=True) drops these
        bindings, and prepare.measure_flops_dispatch calls exactly that after the FLOPs probe.
        Autograd accumulates into a .grad that is already defined, so binding zeroed buffers
        before the first backward gives the same gradients as letting it allocate them.
        """
        for shape, (_, grad_buffer) in self.muon_flat.items():
            grad_buffer.zero_()
            i = 0
            for p in self.transformer.h.parameters():
                if p.shape == shape:
                    p.grad = grad_buffer[i]
                    i += 1
            assert i == grad_buffer.size(0)

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
            flat = getattr(self, "muon_flat", {}).get(shape)
            if flat is not None:
                # the buffer's rows are this group's parameters, in this order
                assert all(p.data_ptr() == flat[0][i].data_ptr() for i, p in enumerate(group_params))
                assert all(p.grad is not None and p.grad.data_ptr() == flat[1][i].data_ptr()
                           for i, p in enumerate(group_params))
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr, flat=flat,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def trunk(self, idx):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            has = str(i) in self.value_embeds
            ve = self.value_embeds[str(i)](idx) if has else None
            ve_weight = self.value_embeds[str(i)].weight if has else None
            x = block(x, x0, self.resid_lambdas[i], self.x0_lambdas[i],
                      ve, ve_weight, idx, cos_sin, self.window_sizes[i])
        return norm(x)

    def forward(self, idx, targets=None, reduction='mean'):
        if targets is None:
            logits = self.lm_head(self.trunk(idx))
            logits = logits.float()
            return SOFTCAP * torch.tanh(logits / SOFTCAP)

        # Sequences are independent -- attention is causal inside a row and the rotary tables are
        # per position -- so running the trunk on a group of rows at a time and concatenating the
        # per-row losses is exactly equivalent. prepare.py's evaluation calls this with 128 rows
        # whatever the training microbatch is, and the MLP's intermediates there are
        # 262,144 x 2048 x 2 = 1,073,741,824 bytes each, which is the peak. A group of 8 rows makes
        # them 67 MB, and under no_grad each group is freed before the next.
        pieces = []
        for i in range(0, idx.size(0), FORWARD_CHUNK_ROWS):
            x = self.trunk(idx[i:i + FORWARD_CHUNK_ROWS])
            pieces.append(softcap_cross_entropy(
                self.lm_head.weight, x.view(-1, x.size(-1)),
                targets[i:i + FORWARD_CHUNK_ROWS].reshape(-1), 'none'))
        nll = pieces[0] if len(pieces) == 1 else torch.cat(pieces)
        if reduction == 'none':
            return nll
        return nll.sum() / (targets >= 0).sum().clamp_min(1)

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
def adamw_fp8_step_fused(p, grad, exp_avg, v_q, v_s, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    """AdamW with the second moment stored in fp8, converted inside this graph.

    exp0068 measured this compression's quality at zero: about 6% per element costs nothing, because the
    second moment enters the update only as 1/sqrt(v) and the sensitivity is half the storage error. What
    it cost was 3.41% of throughput with the conversions eager, and exp0074 confirmed that at 0.0024 of
    val_bpb it does not fit the headroom this line has. Inside the graph the dequantise fuses into the
    lerp and the requantise into the write after it.

    The scale is per row, because fp8 buys range rather than precision and the range here spans token
    frequency: a rare token's row carries a second moment orders of magnitude below a common one, and
    under one tensor-wide scale those rows fall through e5m2's floor at 2**-16 and quantise to zero,
    which is a denominator of eps and an unbounded step.
    """
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    v = (v_q.float() * v_s).lerp_(grad.float().square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (v / bias2).sqrt().to(p.dtype) + eps_t
    p.add_(exp_avg / denom, alpha=-(lr_t / bias1))
    scale = v.amax(dim=1, keepdim=True).div_(57344.0).clamp_min(1e-30)          # e5m2's largest finite
    v_q.copy_((v / scale).to(torch.float8_e5m2))
    v_s.copy_(scale)


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
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim, decay):
    # Nesterov momentum, in bf16. exp0080 quantised this buffer as well and cost 0.0057 of val_bpb
    # where its parts measured 0.0010 -- so this row measures the second moment's compression on its own,
    # and gives back the 1.2% of throughput the frontier pays for its eager momentum conversions.
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    # Polar express orthogonalization. Every step here is a group-sized tensor and this function is
    # where the peak lives: the marks read +48 MB at the first optimizer step and +73 MB at the
    # second, against +42 MB for the whole compiled backward. So the same arithmetic is written to
    # allocate less -- the normalisation in place on our own stack copy, and each iteration's
    # a*X + X@B as one baddbmm instead of three tensors.
    X = g.bfloat16()
    X = X.div_(X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = torch.baddbmm(X, X, B, beta=a)
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = torch.baddbmm(X, B, X, beta=a)
    g = X
    # NorMuon variance reduction
    beta2 = beta2_t.to(g.dtype)
    # square in place on the float32 copy: one full-width float32 tensor instead of two
    v_mean = g.float().square_().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious weight decay + parameter update. The cautious mask costs a full-width product and a
    # full-width bool, and both are multiplied by wd -- so with no decay they are pure waste.
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    if decay:
        mask = (g * stacked_params) >= 0
        stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)
    else:
        stacked_params.sub_(g.mul_(lr))


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
                if p.ndim == 2:
                    state['v_q'] = torch.zeros(p.shape, dtype=torch.float8_e5m2, device=p.device)
                    state['v_s'] = torch.ones(p.size(0), 1, dtype=torch.float32, device=p.device)
                else:
                    state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            if p.ndim == 2:
                adamw_fp8_step_fused(p, grad, state['exp_avg'], state['v_q'], state['v_s'],
                                     self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                                     self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)
            else:
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
        flat = group.get('flat')
        if flat is not None and not all(p.grad is not None and p.grad.data_ptr() == flat[1][i].data_ptr()
                                       for i, p in enumerate(params)):
            print("[muon] gradient views were replaced; stacking this group instead")
            flat = None
        if flat is None:
            stacked_grads = torch.stack([p.grad for p in params])
            stacked_params = torch.stack(params)
        else:
            stacked_params, stacked_grads = flat
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim,
                        group["weight_decay"] != 0.0)
        if flat is None:
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
ASPECT_RATIO = 96       # model_dim = depth * ASPECT_RATIO, rounded up to a multiple of HEAD_DIM
HEAD_DIM = 128          # target head dimension for attention
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context

# Optimization
TOTAL_BATCH_SIZE = 2**17 # ~131K tokens per optimizer step
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.0      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
# 0.4, below the reference's 0.5, because exp0064 measured 0.65 as +0.0016426 against 0.5 at one step
# and a tenth of a percent of throughput apart. That is above this task's resolution floor, unlike the
# learning-rate magnitude, so the axis has a real slope and it points at a longer constant leg rather
# than a lower average rate. One step of 0.1 on the paying side.
WARMDOWN_RATIO = 0.4    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# Model size
DEPTH = 5               # number of transformer layers
# 2, down from 4. Retention is per microbatch: the graph for one microbatch is freed before the next
# begins, so this halves every retained activation -- about 102,558,000 bytes at the frontier -- while
# leaving transients alone, because the trunk group stays at FORWARD_CHUNK_ROWS = 2 and every kernel
# therefore keeps the shape it has now. exp0062 is the control for that distinction: it shrank the group
# instead and lost 36.3% of throughput to 2,048-token matmuls.
#
# What doubles is the number of accumulation steps, 16 to 32, and with it the per-microbatch overhead
# outside the kernels. The FLOPs probe follows this knob too -- prepare.py measures min(8, microbatch)
# sequences -- which halves the probe's own peak, and flops per token is invariant to its row count.
DEVICE_BATCH_SIZE = 2  # per-device batch size (reduce if OOM)

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

def mem_mark(label):
    """Attribute the high-water mark to a phase. max_memory_allocated is monotone and this
    only reads it -- reset_peak_memory_stats() would falsify the reported peak."""
    print(f"[mem] {label:24s} peak={torch.cuda.max_memory_allocated():,} "
          f"live={torch.cuda.memory_allocated():,}")


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
model.flatten_matrix_params()

param_counts = count_params(model)
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")
mem_mark("model allocated")

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

mem_mark("optimizer state")
# Inductor's matmul autotuning, off by default. The loop runs at 24.8% of the A100 denominator and
# every byte idea left on the board is priced in throughput -- 2.65% of it would bank the block fold's
# 41,943,040 bytes -- so the cheapest place to look for throughput is the matmul configs themselves,
# which have never been tuned on this task.
#
# Two things are deliberately NOT done. The `max-autotune` mode string also switches on
# `coordinate_descent_tuning`, which killed exp0030: it searched configs for evaluate_bpb's no_grad
# graphs after the 600 s training clock was spent and the launch died at 1801.3 s, and it ran training
# 11.5% slower besides. And cudagraphs stays off, because the backward here is a chain of custom
# Functions. So only max_autotune is set, and it is switched off again the moment the training loop
# ends, before the reporter's evaluation compiles its own graphs.
import torch._inductor.config as inductor_config
inductor_config.max_autotune = True
inductor_config.coordinate_descent_tuning = False
# Inductor's static memory planner, off by default. exp0046 measured it as byte-identical -- every phase
# mark the same to the byte -- but that was at DEVICE_BATCH_SIZE 4, before the block fold, AttnCore, the
# fused head and the microbatch halving, on a peak of 703,039,488 whose binding phase has since moved. A
# price is only valid at the shape it was measured on, and exp0075 just showed the same thing from the
# other side: LOSS_CHUNK_TOKENS 512 was -24,318,976 bytes then and is 0 now.
#
# It is also the only idea left that cannot cost throughput: it changes how a compiled graph's
# intermediates are placed, not what is computed. About 156,000,000 bytes of the mark are backward
# transients that no intervention has attributed -- seven have tried -- and this is the one lever that
# addresses them without restructuring anything.
inductor_config.memory_planning = True
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
# The probe ends with target.zero_grad(set_to_none=True), which drops the gradient views.
model.bind_matrix_grads()

mem_mark("after flops probe")
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
        if step == 1 and micro_step == 0:
            # Two prints, no reset, so neither can move the reported peak. exp0062 read them: the
            # first microbatch's pass is 41,943,040 below the window's mark, which is the AdamW
            # gradient set that does not exist yet while it runs.
            mem_mark("step 1 after forward")
        loss = loss / grad_accum_steps
        loss.backward()
        if step == 1 and micro_step == 0:
            mem_mark("step 1 after backward")
            pb = sum(p.numel() * p.element_size() for p in model.parameters())
            gb = sum(p.grad.numel() * p.grad.element_size()
                     for p in model.parameters() if p.grad is not None)
            ob = {}
            for st in optimizer.state.values():
                for k, v in st.items():
                    if torch.is_tensor(v):
                        ob[k] = ob.get(k, 0) + v.numel() * v.element_size()
            named = pb + gb + sum(ob.values())
            print(f"[resident] parameters={pb:,} gradients={gb:,} "
                  + " ".join(f"{k}={v:,}" for k, v in sorted(ob.items()))
                  + f" named_total={named:,} allocated={torch.cuda.memory_allocated():,}"
                  + f" unaccounted={torch.cuda.memory_allocated() - named:,}")
            pb = sum(p.numel() * p.element_size() for p in model.parameters())
            gb = sum(p.grad.numel() * p.grad.element_size()
                     for p in model.parameters() if p.grad is not None)
            ob = {}
            for st in optimizer.state.values():
                for k, v in st.items():
                    if torch.is_tensor(v):
                        ob[k] = ob.get(k, 0) + v.numel() * v.element_size()
            named = pb + gb + sum(ob.values())
            print(f"[resident] parameters={pb:,} gradients={gb:,} "
                  + " ".join(f"{k}={v:,}" for k, v in sorted(ob.items()))
                  + f" named_total={named:,} allocated={torch.cuda.memory_allocated():,}"
                  + f" unaccounted={torch.cuda.memory_allocated() - named:,}")
        x, y, epoch = next(train_loader)

    if step < 3:
        # Which phase owns the high-water mark. Reads the monotone counter and prints; it cannot
        # move the reported peak. exp0048 read it in step 1's forward+backward, never the optimizer.
        mem_mark(f"step {step} fwd+bwd")

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
    if step < 3:
        mem_mark(f"step {step} optimizer")
    # Same as zero_grad(set_to_none=True) for every parameter Muon does not step; the matrix
    # gradients are zeroed in their buffers instead, since dropping them would drop the views
    # autograd accumulates into. They are live through the optimizer step either way.
    for p in model.parameters():
        if p.grad is not None and not getattr(p, "flat_grad", False):
            p.grad = None
    for _, grad_buffer in model.muon_flat.values():
        grad_buffer.zero_()

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
    if step <= 2:
        mem_mark(f"end of step {step}")
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
# The clock is spent; evaluate_bpb compiles its own no_grad graphs from here and must not be searched.
inductor_config.max_autotune = False
mem_mark("end of training loop")

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
# after the reporter: shows whether the frozen 128x2048 evaluation forward is the peak.
# a plain print, not a METRICS_JSON line -- the reporter remains the only source of those.
mem_mark("after reporter+eval")
