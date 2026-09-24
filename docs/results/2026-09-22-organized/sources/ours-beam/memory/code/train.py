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


RMS_NORM_EPS = torch.finfo(torch.float32).eps   # F.rms_norm's default epsilon: it is the
                                                # epsilon of the accumulation dtype, which
                                                # is float32 for bfloat16 input.


class NormProj(torch.autograd.Function):
    """rms_norm folded into the projections that consume it.

    At a block site today, F.rms_norm saves a float32 upcast of its input plus a per-row
    reciprocal scale, and the projections save the normalised tensor to form their weight
    gradients. Here the normalised tensor is never handed to autograd: the Function saves
    the bfloat16 residual x and the per-row scale s, and rebuilds the normalised tensor in
    backward, which is elementwise plus one reduction and repeats no matmul. So the saved
    set at each site loses the float32 upcast, and trades the normalised tensor for the
    residual of the same shape and dtype.

    With ms the mean square over the last dimension, s = (ms + eps)**-0.5 and n = x * s,
        dL/dx = s * (g - n * (sum_j g_j n_j) / D),   D = x.size(-1)
    where g is the gradient summed over every consumer of n. A weight narrower than x
    consumes the leading channels of n, as `x[..., :ve_gate_channels]` does today.

    The residual x is not retained either. The backward reads only n and s, so instead of x
    the saved set holds n quantised to int8 with a per-row float32 amplitude mx:
        q = round(n * LEVELS / mx),   mx = max_j |n_j|,   n = q * mx / LEVELS
    and the per-element absolute error of the rebuilt n is at most mx / (2 * LEVELS). The
    forward's own outputs are computed from the exact n, so the forward arithmetic is
    unchanged; only the copy handed to backward is quantised.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, *weights):
        s = (x.float().pow(2).mean(dim=-1, keepdim=True) + RMS_NORM_EPS).rsqrt()
        n = (x.float() * s).to(x.dtype)
        outs = []
        for w in weights:
            k = w.size(1)
            outs.append(F.linear(n if k == n.size(-1) else n[..., :k], w))
        # The quantised store. Every tensor this backward consumes is n and s; x appears in
        # no returned gradient except through n, so it was retained only as a means of
        # rebuilding n. Saving a quantised n instead means x is not named in the saved set
        # at all. The outputs above are computed from the exact n, so the forward arithmetic
        # is untouched; only the saved copy is quantised. n has per-row root-mean-square
        # exactly 1 by construction, so a per-row amplitude over the channels puts the whole
        # row inside a known range, and n is signed so the int8 range is used two-sided.
        mx = n.abs().amax(dim=-1, keepdim=True).float().clamp_min(torch.finfo(torch.float32).tiny)
        q = (n.float() * (NORM_STORE_LEVELS / mx)).round().to(torch.int8)
        del n
        ctx.x_dtype = x.dtype
        ctx.save_for_backward(q, mx, s, *weights)
        return tuple(outs)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, *grad_outs):
        q, mx, s = ctx.saved_tensors[0:3]
        weights = ctx.saved_tensors[3:]
        D = q.size(-1)
        n = q.to(torch.float32).mul_(mx / NORM_STORE_LEVELS).to(ctx.x_dtype)
        grad_weights = []
        g = None                      # gradient wrt n, accumulated in float32
        for i, (grad_y, w) in enumerate(zip(grad_outs, weights)):
            k = w.size(1)
            n_in = n if k == D else n[..., :k]
            if ctx.needs_input_grad[1 + i]:
                g2 = grad_y.reshape(-1, grad_y.size(-1))
                n2 = n_in.reshape(-1, k)
                grad_weights.append(torch.matmul(g2.t(), n2.to(g2.dtype)).to(w.dtype))
                del g2, n2
            else:
                grad_weights.append(None)
            gn = torch.matmul(grad_y, w.to(grad_y.dtype)).float()
            if k == D:
                g = gn if g is None else g.add_(gn)
            else:
                if g is None:
                    g = gn.new_zeros(*gn.shape[:-1], D)
                g[..., :k] += gn
            del gn, n_in
        grad_x = None
        if ctx.needs_input_grad[0]:
            nf = n.float()
            dot = (g * nf).sum(dim=-1, keepdim=True)
            grad_x = (s * (g - nf * (dot / D))).to(ctx.x_dtype)
            del nf, dot
        del n, g
        return (grad_x, *grad_weights)


class RotaryNormQK(torch.autograd.Function):
    """apply_rotary_emb followed by norm, on q or on k, in one Function.

    What the rotary output costs today is F.rms_norm's own backward, which saves a float32
    upcast of its input plus a per-row reciprocal scale. This Function saves the normalised
    output instead -- the same tensor the attention kernel holds anyway -- plus a per-row
    scale of the same shape as the one rms_norm saved, and it hands neither the rotary
    output nor an upcast of it to autograd.

    With y the normalised output, s the reciprocal root mean square and D the channel
    count, the rms gradient mentions only the output:
        dL/dr = s * (g - y * (sum_j g_j y_j) / D)
    and the rotary map is a rotation on channel pairs, so its transpose is the same
    arithmetic with the sign of sin flipped:
        grad_x1 = cos * g1 - sin * g2,    grad_x2 = sin * g1 + cos * g2
    Neither term needs a saved activation of the rotary input.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, t, cos, sin):
        assert t.ndim == 4
        d = t.shape[3] // 2
        x1, x2 = t[..., :d], t[..., d:]
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        r = torch.cat([y1, y2], 3)
        del x1, x2, y1, y2
        s = (r.float().pow(2).mean(dim=-1, keepdim=True) + RMS_NORM_EPS).rsqrt()
        y = (r.float() * s).to(r.dtype)
        del r
        ctx.save_for_backward(y, s, cos, sin)
        return y

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_y):
        if not ctx.needs_input_grad[0]:
            return None, None, None
        y, s, cos, sin = ctx.saved_tensors
        D = y.size(-1)
        gy = grad_y.float()
        yf = y.float()
        dot = (gy * yf).sum(dim=-1, keepdim=True)
        grad_r = (s * (gy - yf * (dot / D))).to(grad_y.dtype)
        del gy, yf, dot
        d = D // 2
        g1, g2 = grad_r[..., :d], grad_r[..., d:]
        grad_t = torch.cat([g1 * cos - g2 * sin, g1 * sin + g2 * cos], 3)
        del grad_r, g1, g2
        return grad_t, None, None


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
        # x is the block's residual tensor: the normalisation this site used to be handed
        # is now owned by NormProj, which never retains its normalised output.
        B, T, C = x.size()
        if ve is not None:
            q_, k_, v_, gate_ = NormProj.apply(
                x, self.c_q.weight, self.c_k.weight, self.c_v.weight, self.ve_gate.weight)
        else:
            q_, k_, v_ = NormProj.apply(
                x, self.c_q.weight, self.c_k.weight, self.c_v.weight)
        q = q_.view(B, T, self.n_head, self.head_dim)
        k = k_.view(B, T, self.n_kv_head, self.head_dim)
        v = v_.view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(gate_)
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        # apply_rotary_emb followed by norm at both sites, owned by RotaryNormQK, which
        # retains neither the rotary output nor an upcast of it. apply_rotary_emb itself
        # stays defined above and is the arithmetic this Function's forward performs.
        q = RotaryNormQK.apply(q, cos, sin)
        k = RotaryNormQK.apply(k, cos, sin)

        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class SquaredReluProj(torch.autograd.Function):
    """relu(h)**2 followed by the MLP's second projection, in one Function.

    Autograd would retain two [B, T, 4*n_embd] tensors here: r = relu(h), which both the
    relu and the squaring read in backward, and a = r * r, which is c_proj's input and so
    is needed for c_proj's weight gradient. Only r is retained here; a is rebuilt in
    backward from r, which is elementwise and repeats no matmul.

    d(relu(h)**2)/dh = 2 * relu(h), which is already zero wherever relu is zero, so no
    sign information about h is needed.

    r is the widest tensor this block hands to autograd, so it is stored as int8 with a
    per-row float32 scale instead of at h's own width: q = round(r * LEVELS / m) with
    m = max_j r_j over the row, and r is rebuilt in backward as q * m / LEVELS. r is
    non-negative, so the signed range is used one-sided and the per-element absolute error
    is at most m / (2 * LEVELS). The forward output is still computed from the exact r, so
    the forward arithmetic is untouched; only the saved copy is quantised. relu's exact
    zeros round to exact zeros, so the sparsity pattern is preserved bit-exactly, and a
    row that is entirely non-positive has m = 0, which clamp_min keeps out of the divisor.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, h, weight):
        r = F.relu(h)
        a = r * r
        y = F.linear(a, weight)
        del a
        m = r.amax(dim=-1, keepdim=True).float().clamp_min(torch.finfo(torch.float32).tiny)
        q = (r.float() * (RELU_STORE_LEVELS / m)).round().to(torch.int8)
        del r
        ctx.h_dtype = h.dtype
        ctx.save_for_backward(q, m, weight)
        return y

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_y):
        q, m, weight = ctx.saved_tensors
        r = q.to(torch.float32).mul_(m / RELU_STORE_LEVELS).to(ctx.h_dtype)
        grad_h = grad_weight = None
        a = r * r
        if ctx.needs_input_grad[1]:
            # The same contraction F.linear's backward performs: grad_y^T @ a over the
            # flattened leading extent. Under autocast every cast below is a no-op.
            g2 = grad_y.reshape(-1, grad_y.size(-1))
            a2 = a.reshape(-1, a.size(-1))
            grad_weight = torch.matmul(g2.t(), a2.to(g2.dtype)).to(weight.dtype)
            del g2, a2
        del a
        if ctx.needs_input_grad[0]:
            grad_a = torch.matmul(grad_y, weight.to(grad_y.dtype))
            grad_h = grad_a.mul_(r.to(grad_a.dtype)).mul_(2.0)
            del grad_a
            if grad_h.dtype != ctx.h_dtype:
                grad_h = grad_h.to(ctx.h_dtype)
        del r
        return grad_h, grad_weight


class NormFcSquaredReluProj(torch.autograd.Function):
    """The MLP site's normalisation, c_fc, relu(h)**2 and c_proj, in one Function.

    On this parent the MLP site is two Functions in series: NormProj hands the normalised
    input to c_fc and retains only the block residual and the per-row scale, and
    SquaredReluProj retains r = F.relu(h) as int8 with a per-row scale. That int8 tensor is
    the widest thing the block still hands to autograd, and it is what a c_fc recompute
    removes -- but the recompute needs the normalised input, which NormProj deliberately
    does not retain, so on a NormProj host the recompute is only reachable as one Function
    that owns the normalisation too. Nothing of 4 * n_embd width survives this forward: h,
    r and a are all freed inside it and rebuilt in backward.

    The saved set is exactly what NormProj saves at this site, so the int8 relu store and its
    per-row scale are removed with nothing added. What NormProj saves is now the quantised
    normalised tensor, and so is this: the block residual x is not retained, because the only
    thing this backward needs x for is to rebuild n. Instead the saved set holds n quantised
    to int8 with a per-row float32 amplitude mx,
        q = round(n * LEVELS / mx),   mx = max_j |n_j|,   n = q * mx / LEVELS
    with the rebuilt n in error by at most mx / (2 * LEVELS) per element. The forward's own
    outputs are computed from the exact n, so the forward arithmetic is untouched; only the
    copy handed to backward is quantised, and here that copy is also what the c_fc recompute
    reads, so the recompute's input carries the quantisation error too.

    One c_fc matmul per call is therefore executed in backward that the parent did not
    execute, and measure_flops_dispatch counts it. It is admissible only because the
    attention windows clamped in the same edit refund exactly as much counted arithmetic
    per token.

    The gradients are the ones c_fc's, the squaring's, the relu's and c_proj's own backwards
    would form, composed with the normalisation's, which mentions only n and s:
        d(relu(h)**2)/dh = 2 * relu(h),
        dL/dx = s * (gn - n * (sum_j gn_j n_j) / D),   D = x.size(-1)
    where gn is the gradient arriving at the normalised tensor.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, fc_weight, proj_weight):
        s = (x.float().pow(2).mean(dim=-1, keepdim=True) + RMS_NORM_EPS).rsqrt()
        n = (x.float() * s).to(x.dtype)
        h = F.linear(n, fc_weight)
        # The quantised store, the same one NormProj applies, on this site's own n. The
        # backward reads only n and s; x appeared in no returned gradient except through n,
        # so it was retained only as a means of rebuilding n. n has per-row root-mean-square
        # exactly 1 by construction, so a per-row amplitude over the channels puts the whole
        # row inside a known range, and n is signed so the int8 range is used two-sided.
        mx = n.abs().amax(dim=-1, keepdim=True).float().clamp_min(torch.finfo(torch.float32).tiny)
        q = (n.float() * (NORM_STORE_LEVELS / mx)).round().to(torch.int8)
        del n
        r = F.relu(h)
        del h
        a = r * r
        del r
        y = F.linear(a, proj_weight)
        del a
        ctx.x_dtype = x.dtype
        ctx.save_for_backward(q, mx, s, fc_weight, proj_weight)
        return y

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_y):
        global _MLP_TRACE_ONCE
        trace = _MLP_TRACE_ONCE and not torch.compiler.is_compiling()
        if trace:
            mem_mark("mlp fused backward entry")
        q, mx, s, fc_weight, proj_weight = ctx.saved_tensors
        D = q.size(-1)
        n = q.to(torch.float32).mul_(mx / NORM_STORE_LEVELS).to(ctx.x_dtype)
        grad_x = grad_fc_weight = grad_proj_weight = None
        h = F.linear(n, fc_weight.to(n.dtype))
        r = F.relu(h)
        del h
        a = r * r
        if trace:
            mem_mark("mlp fused backward after r rebuilt")
        if ctx.needs_input_grad[2]:
            # The same contraction F.linear's backward performs, over the flattened
            # leading extent. Under autocast every cast below is a no-op.
            g2 = grad_y.reshape(-1, grad_y.size(-1))
            a2 = a.reshape(-1, a.size(-1))
            grad_proj_weight = torch.matmul(g2.t(), a2.to(g2.dtype)).to(proj_weight.dtype)
            del g2, a2
        del a
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            grad_a = torch.matmul(grad_y, proj_weight.to(grad_y.dtype))
            grad_h = grad_a.mul_(r.to(grad_a.dtype)).mul_(2.0)
            del grad_a, r
            if ctx.needs_input_grad[1]:
                gh2 = grad_h.reshape(-1, grad_h.size(-1))
                n2 = n.reshape(-1, n.size(-1))
                grad_fc_weight = torch.matmul(gh2.t(), n2.to(gh2.dtype)).to(fc_weight.dtype)
                del gh2, n2
            if ctx.needs_input_grad[0]:
                gn = torch.matmul(grad_h, fc_weight.to(grad_h.dtype)).float()
                nf = n.float()
                dot = (gn * nf).sum(dim=-1, keepdim=True)
                grad_x = (s * (gn - nf * (dot / D))).to(ctx.x_dtype)
                del gn, nf, dot
            del grad_h
        else:
            del r
        del n
        if trace:
            mem_mark("mlp fused backward after grad_h freed")
            _MLP_TRACE_ONCE = False
        return grad_x, grad_fc_weight, grad_proj_weight


# Instrumentation only (generation 11): a one-shot flag for the allocator reads inside
# NormFcSquaredReluProj.backward above. Its first term is this flag and its second is
# `not torch.compiler.is_compiling()`, which folds to False while torch.compile traces, so
# the measured graph is unchanged and no graph break is introduced. The one backward they
# fire in is prepare.measure_flops_dispatch's, which runs the UNCOMPILED module and is the
# first call of the launch; the flag is cleared there. Reading the allocator's counters
# allocates nothing, dispatches nothing and resets nothing.
_MLP_TRACE_ONCE = True


def recomputing_block_indices(window_sizes, sequence_len):
    """Blocks whose attention window is shorter than the full context.

    A block recomputes c_fc in backward if and only if its own window is clamped, which is
    the rule the clamped windows' FLOPs refund pays for. The flag is resolved once, at
    construction, so nothing in forward looks a block up by name or index.
    """
    return [i for i, (left, _right) in enumerate(window_sizes) if left < sequence_len]


class MLP(nn.Module):
    def __init__(self, config, recompute_fc=False):
        super().__init__()
        self.recompute_fc = recompute_fc
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        # x is the block's residual tensor; this site's normalisation is owned by whichever
        # Function consumes it -- NormFcSquaredReluProj where this block recomputes c_fc,
        # NormProj otherwise.
        if self.recompute_fc:
            return NormFcSquaredReluProj.apply(x, self.c_fc.weight, self.c_proj.weight)
        (h,) = NormProj.apply(x, self.c_fc.weight)
        return SquaredReluProj.apply(h, self.c_proj.weight)


class Block(nn.Module):
    def __init__(self, config, layer_idx, recompute_fc=False):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config, recompute_fc)

    def forward(self, x, ve, cos_sin, window_size):
        # Both normalisations are owned by NormProj at their consuming site.
        x = x + self.attn(x, ve, cos_sin, window_size)
        x = x + self.mlp(x)
        return x


class FusedChunkedHeadLoss(torch.autograd.Function):
    """Unembedding + float32 softcap + cross-entropy, streamed in row blocks.

    The tail of GPT.forward is the only place a tensor of vocabulary width exists. This
    Function walks the rows in blocks of `chunk_rows`, so at most one [chunk_rows, vocab]
    tensor is live at a time, and -- when a gradient is required -- forms the head's input
    and weight gradients inside the same block instead of leaving the softcapped logits
    for autograd to read in backward. Per-row softcapped cross-entropy does not couple
    rows, so the blocking is exact.

    The gradient with respect to the pre-softcap activations u is analytic:
        z = softcap * tanh(u / softcap)
        dz/du = 1 - tanh(u / softcap)**2 = 1 - (z / softcap)**2
        dL/dz = softmax(z) - onehot(target)      (zero on ignored rows)
    scaled by 1/(number of counted target rows) for reduction == 'mean'.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x_flat, weight, targets_flat, softcap, chunk_rows, reduction):
        assert reduction in ("mean", "none")
        assert x_flat.dim() == 2 and targets_flat.dim() == 1
        N = x_flat.size(0)
        want_gx = ctx.needs_input_grad[0]
        want_gw = ctx.needs_input_grad[1]
        need_grad = want_gx or want_gw

        valid = targets_flat != -1
        idx = targets_flat.clamp_min(0).unsqueeze(1)
        # One cast of the weight, matching the single cached cast autocast performs for
        # the reference F.linear.
        w = weight if weight.dtype == x_flat.dtype else weight.to(x_flat.dtype)

        if reduction == "mean":
            loss_out = x_flat.new_zeros((), dtype=torch.float32)
            denom = valid.sum().to(torch.float32)
        else:
            loss_out = x_flat.new_empty(N, dtype=torch.float32)
            denom = None

        # Gradient buffers, pre-allocated once; nothing of vocabulary width is saved.
        # Only the reduction == 'mean' path can pre-accumulate the weight gradient here,
        # because a per-row incoming gradient cannot be applied to an already summed
        # weight gradient afterwards; reduction == 'none' recomputes in backward.
        prebuilt = need_grad and reduction == "mean"
        grad_x = torch.empty_like(x_flat) if (prebuilt and want_gx) else None
        grad_w = weight.new_zeros(weight.shape) if (prebuilt and want_gw) else None

        for lo in range(0, N, chunk_rows):
            hi = min(lo + chunk_rows, N)
            xb = x_flat[lo:hi]
            ib = idx[lo:hi]
            vb = valid[lo:hi]
            z = torch.matmul(xb, w.t()).float()
            z.div_(softcap).tanh_().mul_(softcap)
            # log-softmax cross-entropy without materialising the log-softmax
            lse = torch.logsumexp(z, dim=-1)
            lb = torch.where(vb, lse - z.gather(1, ib).squeeze(1),
                             torch.zeros_like(lse))
            if reduction == "mean":
                loss_out = loss_out + lb.sum()
            else:
                loss_out[lo:hi] = lb
            del lb
            if prebuilt:
                p = (z - lse.unsqueeze(1)).exp_()        # softmax(z)
                # z becomes the tanh chain factor 1 - (z / softcap)**2, in place.
                z.div_(softcap).square_().neg_().add_(1.0)
                s = (vb.to(torch.float32) / denom).unsqueeze(1)
                du32 = p.mul_(s).scatter_add_(1, ib, -s).mul_(z)
                del z, p, s                              # frees one block before the cast
                du = du32.to(x_flat.dtype)
                del du32
                if grad_x is not None:
                    grad_x[lo:hi] = torch.matmul(du, w)
                if grad_w is not None:
                    grad_w.add_(torch.matmul(du.t(), xb).to(grad_w.dtype))
                del du
            else:
                del z
            del lse, xb, ib, vb

        if reduction == "mean":
            loss_out = loss_out / denom

        ctx.reduction = reduction
        ctx.softcap = softcap
        ctx.chunk_rows = chunk_rows
        ctx.want_gx = want_gx
        ctx.want_gw = want_gw
        if prebuilt:
            ctx.mode = "prebuilt"
            ctx.save_for_backward(grad_x, grad_w)
        elif need_grad:
            ctx.mode = "recompute"
            ctx.save_for_backward(x_flat, weight, targets_flat)
        else:
            ctx.mode = "nograd"
        return loss_out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        if ctx.mode == "nograd":
            return None, None, None, None, None, None

        if ctx.mode == "prebuilt":
            grad_x, grad_w = ctx.saved_tensors
            gx = None
            gw = None
            if ctx.want_gx:
                gx = grad_x * grad_output.to(grad_x.dtype)
            if ctx.want_gw:
                gw = grad_w * grad_output.to(grad_w.dtype)
            return gx, gw, None, None, None, None

        # reduction == 'none' with a gradient required: rebuild the blocks, scaling each
        # row by its own incoming gradient. Never taken by this program, which backwards
        # only through the scalar reduction == 'mean' loss.
        x_flat, weight, targets_flat = ctx.saved_tensors
        softcap, chunk_rows = ctx.softcap, ctx.chunk_rows
        N = x_flat.size(0)
        valid = targets_flat != -1
        idx = targets_flat.clamp_min(0).unsqueeze(1)
        w = weight if weight.dtype == x_flat.dtype else weight.to(x_flat.dtype)
        go = grad_output.reshape(-1)
        gx = torch.empty_like(x_flat) if ctx.want_gx else None
        gw = weight.new_zeros(weight.shape) if ctx.want_gw else None
        for lo in range(0, N, chunk_rows):
            hi = min(lo + chunk_rows, N)
            xb = x_flat[lo:hi]
            ib = idx[lo:hi]
            z = torch.matmul(xb, w.t()).float()
            z.div_(softcap).tanh_().mul_(softcap)
            p = (z - torch.logsumexp(z, dim=-1, keepdim=True)).exp_()
            z.div_(softcap).square_().neg_().add_(1.0)
            s = (valid[lo:hi].to(torch.float32) * go[lo:hi].to(torch.float32)).unsqueeze(1)
            du32 = p.mul_(s).scatter_add_(1, ib, -s).mul_(z)
            del z, p, s
            du = du32.to(x_flat.dtype)
            del du32
            if gx is not None:
                gx[lo:hi] = torch.matmul(du, w)
            if gw is not None:
                gw.add_(torch.matmul(du.t(), xb).to(gw.dtype))
            del du, xb, ib
        return gx, gw, None, None, None, None


# Instrumentation only (generation 5): the reads inside the blocked evaluation forward
# print on the first evaluation call. Reading allocates nothing and resets nothing.
_EVAL_BLOCK_TRACED = False


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.recomputing_blocks = recomputing_block_indices(self.window_sizes,
                                                            config.sequence_len)
        recompute = set(self.recomputing_blocks)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i, i in recompute)
                                for i in range(config.n_layer)]),
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
        # The blocks' matrices in bf16 as well, as wte and the value embeddings already
        # are. Every matmul that reads them already runs under bfloat16 autocast, so the
        # float32 storage served only the update's own accumulation. This runs AFTER every
        # uniform_ and zeros_ above, so no initial value is drawn differently -- each is
        # drawn in float32 and then rounded once. lm_head, resid_lambdas and x0_lambdas are
        # deliberately left in float32.
        for block in self.transformer.h:
            block.to(dtype=torch.bfloat16)

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
        assert all(c in "SLH" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0),
                          "H": (long_window // 4, 0)}
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

    def _forward_rows(self, idx, targets=None, reduction='mean'):
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
        if targets is not None:
            # Same head, same float32 softcap, same per-row cross-entropy, streamed in
            # row blocks so no tensor of vocabulary width spans the whole batch.
            # ctx.needs_input_grad inside a Function reads the inputs' requires_grad and
            # ignores grad mode, so under torch.no_grad -- the frozen validation forward --
            # the weight must be detached here for the Function to know that no gradient
            # is wanted and allocate no gradient buffers.
            head_weight = (self.lm_head.weight if torch.is_grad_enabled()
                           else self.lm_head.weight.detach())
            return FusedChunkedHeadLoss.apply(
                x.view(-1, x.size(-1)), head_weight, targets.view(-1),
                softcap, HEAD_CHUNK_ROWS, reduction)

        logits = self.lm_head(x)
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)
        return logits

    @torch._dynamo.disable
    def _blocked_eval_loss(self, idx, targets):
        """Per-token losses at the frozen evaluation shape, computed in row blocks.

        Only a call with targets, reduction == 'none' and more than EVAL_ROWS_PER_BLOCK
        rows reaches this method, which is prepare.evaluate_bpb's call and nothing else:
        the training loop and prepare.measure_flops_dispatch both use the default
        reduction. A row's per-token losses depend on that row alone -- the embedding,
        every normalisation, the causal attention window and the streamed head are all
        within-row -- so each row is computed from the same inputs by the same operations
        as in the unblocked forward, and no tensor in the forward spans more than
        EVAL_ROWS_PER_BLOCK rows.
        """
        global _EVAL_BLOCK_TRACED
        trace = not _EVAL_BLOCK_TRACED
        _EVAL_BLOCK_TRACED = True
        B, T = idx.size()
        out = torch.empty(B * T, dtype=torch.float32, device=idx.device)
        num_blocks = (B + EVAL_ROWS_PER_BLOCK - 1) // EVAL_ROWS_PER_BLOCK
        if trace:
            mem_mark("eval blocked forward entry")
        done = 0
        for lo in range(0, B, EVAL_ROWS_PER_BLOCK):
            hi = min(lo + EVAL_ROWS_PER_BLOCK, B)
            # Leading-dimension slices of contiguous tensors: views, copying nothing. The
            # final block is short whenever EVAL_ROWS_PER_BLOCK does not divide B.
            out[lo * T:hi * T] = self._forward_rows(idx[lo:hi], targets[lo:hi], 'none')
            done += 1
            if trace:
                if done == 1:
                    mem_mark("eval block 1")
                elif done == 2:
                    mem_mark("eval block 2")
                if done == num_blocks:
                    mem_mark("eval last block")
        return out

    def forward(self, idx, targets=None, reduction='mean'):
        if (targets is not None and reduction == 'none'
                and idx.size(0) > EVAL_ROWS_PER_BLOCK):
            return self._blocked_eval_loss(idx, targets)
        return self._forward_rows(idx, targets, reduction)

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
WINDOW_PATTERN = "HHSLHHSL"  # sliding window pattern: L=full, S=half, H=quarter context.
                            # Consumed as pattern[layer_idx % len(pattern)] and its length
                            # equals DEPTH, so each block reads its own character. The
                            # per-layer left windows are [512, 512, 1024, 2048, 512, 512,
                            # 1024, 2048], summing to 8192 against the parent's 10240.
RELU_STORE_LEVELS = 127 # positive levels used to store SquaredReluProj's retained relu
                        # output as int8 with a per-row scale; 127 is int8's positive range
NORM_STORE_LEVELS = 127 # levels used to store the normalised tensor as int8 with a per-row
                        # amplitude, at every site that retains it -- NormProj's ten and
                        # NormFcSquaredReluProj's six. 127 is int8's positive range and the
                        # normalised tensor is signed, so the range is used two-sided.

# Optimization
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step
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
DEVICE_BATCH_SIZE = 4   # per-device batch size (reduce if OOM)
HEAD_CHUNK_ROWS = 1024  # row extent of ONE FusedChunkedHeadLoss block, narrowed from the
                        # parent's 4096. Per-row softcapped cross-entropy does not couple
                        # rows, so the blocking is exact at any extent; only the size of
                        # the vocabulary-width tensors that live inside one block changes.
EVAL_ROWS_PER_BLOCK = 8 # rows per block in the FROZEN 128 x 2048 validation forward ONLY.
                        # It cannot apply to training or to the FLOPs probe: the blocked
                        # path is guarded on reduction == 'none', and both of those call
                        # the model with the default reduction.

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

# Instrumentation only (generation 2): read the allocator's peak counter at fixed points
# and print it. Reading allocates nothing, dispatches nothing and changes no arithmetic,
# and reset_peak_memory_stats is never called.
MEM_TRACE_STEP = 12     # one designated step past compilation
# The forward/backward pair of reads is taken on these steps only, and outside the
# compiled region, so it cannot change what inductor fuses inside the model.
MEM_TRACE_STEPS = (0, 1, 2, MEM_TRACE_STEP)


def mem_mark(label):
    print(f"[mem] {label}: allocated={torch.cuda.memory_allocated()} "
          f"peak={torch.cuda.max_memory_allocated()}", flush=True)


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
# One free print: the byte size of one rotary-output tensor at this shape, which is what
# this candidate stops handing to autograd at each of the two sites in every block.
_rot_bytes = DEVICE_BATCH_SIZE * MAX_SEQ_LEN * config.n_head * (config.n_embd // config.n_head) * 2
print(f"Per-site rotary-output bytes: {_rot_bytes} "
      f"(sites={2 * config.n_layer} total={_rot_bytes * 2 * config.n_layer})")
# Two free prints: the number of head blocks one training call now walks, and the byte size
# of one vocabulary-width block at each dtype the head builds. Computing integers from
# constants allocates nothing, dispatches nothing, and is outside the model and the probe.
_head_rows = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
print(f"Head blocks per training call: {_head_rows // HEAD_CHUNK_ROWS} "
      f"({_head_rows} rows / {HEAD_CHUNK_ROWS} rows per block)")
print(f"Per-block vocabulary-width bytes: float32={HEAD_CHUNK_ROWS * vocab_size * 4} "
      f"bfloat16={HEAD_CHUNK_ROWS * vocab_size * 2}")
# One more free print: the bytes SquaredReluProj hands to autograd per block, before and
# after the quantised store. Integers from constants: nothing is allocated or dispatched.
_relu_elems = DEVICE_BATCH_SIZE * MAX_SEQ_LEN * 4 * config.n_embd
_relu_rows = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
print(f"Per-block relu store bytes: bfloat16={_relu_elems * 2} "
      f"int8={_relu_elems} scale_float32={_relu_rows * 4}")
# One free print: the per-site bytes this fold trades -- the normalised block input it
# stops handing to autograd against the per-row scale it saves instead.
_norm_bytes = DEVICE_BATCH_SIZE * MAX_SEQ_LEN * config.n_embd * 2
_scale_bytes = DEVICE_BATCH_SIZE * MAX_SEQ_LEN * 4
print(f"Per-site normalised-input bytes: {_norm_bytes} "
      f"scale_float32={_scale_bytes} (sites={2 * config.n_layer})")
# Three free prints: the two numbers the FLOPs budget of this candidate is built from, and
# the set of blocks that recompute. Reading attributes allocates nothing and dispatches
# nothing, and none of it is inside the model or the probe.
_windows = [w[0] for w in model.window_sizes]
_span_sum = sum(_windows)
_recompute_cost = len(model.recomputing_blocks) * 2 * config.n_embd * 4 * config.n_embd
print(f"Per-layer left windows: {_windows}")
print(f"Recomputing blocks: {model.recomputing_blocks}")
print(f"Attention span sum: {_span_sum} | c_fc recompute FLOPs/token: {_recompute_cost}")
# One free print: the resident parameter bytes by dtype, which is what this candidate
# changes. numel() and element_size() read metadata -- nothing is allocated or dispatched.
_bytes_by_dtype = {}
for _p in {id(p): p for p in model.parameters()}.values():
    _k = str(_p.dtype)
    _bytes_by_dtype[_k] = _bytes_by_dtype.get(_k, 0) + _p.numel() * _p.element_size()
print(f"Parameter bytes by dtype: {_bytes_by_dtype}")
# One free print: the per-site bytes this store trades -- the bfloat16 residual the ctx stops
# naming, against the int8 copy of the normalised tensor and its per-row amplitude, with the
# per-row scale s unchanged. Integers from constants: nothing is allocated.
_store_rows = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
print(f"Per-site saved bytes before: residual_bfloat16={_store_rows * config.n_embd * 2} "
      f"after: q_int8={_store_rows * config.n_embd} amplitude_float32={_store_rows * 4} "
      f"scale_float32={_store_rows * 4} (sites={2 * config.n_layer})")
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")
mem_mark("after model allocation")

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
    # The denominator is doubled because TOTAL_BATCH_SIZE is halved: the plateau at 0.95 is
    # still reached after 600 * 262144 = 157,286,400 consumed tokens, the same count as
    # 300 * 524288 at the parent's batch. The endpoints 0.85 and 0.95 are untouched.
    frac = min(step / 600, 1)
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
        if step in MEM_TRACE_STEPS:
            mem_mark(f"step {step} mb {micro_step + 1} after forward")
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        if step in MEM_TRACE_STEPS:
            mem_mark(f"step {step} mb {micro_step + 1} after backward")
        if step == MEM_TRACE_STEP:
            mem_mark(f"step {step} microbatch {micro_step + 1}/{grad_accum_steps}")
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
    if step == MEM_TRACE_STEP:
        mem_mark(f"step {step} after optimizer step")

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
mem_mark("before reporter")
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
