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

# ---------------------------------------------------------------------------
# The width-1 decode step, written as Triton kernels
# ---------------------------------------------------------------------------
# A decode step is not short of arithmetic or of bandwidth; it is short of work per launch.
# Eager, one step at full context issues about 330 kernels, and a kernel that does nothing
# still costs 1.4 us of device time inside a replayed graph, so the launches alone are most of
# the step. cuBLAS bears this out: a width-1 512x512 projection takes 4.0 us to read half a
# megabyte, which is 130 GB/s on a device that streams fifteen times that.
#
# So: one kernel per matrix, each carrying the pointwise work on either side of it. A
# normalisation of the input vector is free in a matrix-vector product -- every program needs
# the whole vector anyway -- and a residual add, a squared relu or a tanh softcap on the output
# is free for the same reason on the other side. That is the whole idea. It takes the step from
# about 330 launches to 74.
#
# Everything is matched to the reference arithmetic rather than merely to its algebra:
# `F.rms_norm` on bf16 accumulates in fp32 with no epsilon and rounds its result back to bf16,
# autocast makes every linear output bf16, and `relu(x).square()` runs in fp32 because autocast
# promotes `pow`. Each kernel rounds where the reference rounds.

import triton
import triton.language as tl
from triton.language.extra.cuda.libdevice import tanh as _tl_tanh

# (rows per program, k-block, warps) per matrix, measured on these shapes
MV_CONFIG = {
    # Re-swept against the current step, after both occupancy fixes and after the weights became
    # bf16: 216.7 us a step against 225.7 for the configuration chosen at exp0021. Every shape's
    # optimum moved to one warp and the widest k-block, and three of them to one output row per
    # program -- the step now has far fewer other kernels competing for the device, so the matvecs
    # want more programs each rather than fewer and fatter ones.
    "qkv": (1, 512, 1),      # 3 x 512 x 512 in one launch
    "proj": (1, 512, 1),     # 512 x 512
    "fc": (2, 512, 1),       # 2048 x 512
    "mlp": (1, 2048, 1),     # 512 x 2048
    "head": (2, 512, 1),     # 8192 x 512
}

PRO_NONE, PRO_NORM = 0, 1
EPI_NONE, EPI_ADD, EPI_RELU2, EPI_SOFTCAP, EPI_ADD_SCALE = 0, 1, 2, 3, 4
SOFTCAP = 15.0


@triton.jit
def _mv(x_ptr, w_ptr, o_ptr, res_ptr, x0_ptr, sa_ptr, sb_ptr, seq_ptr,
        K: tl.constexpr, ROWS: tl.constexpr,
        KB: tl.constexpr, PRO: tl.constexpr, EPI: tl.constexpr,
        CAP: tl.constexpr, INC: tl.constexpr):
    """`out = W @ f(x)` with a pointwise epilogue; one program per ROWS output rows.

    The input vector is read by every program, so a reduction over it -- which is what an rms
    norm is -- costs nothing beyond L2 traffic, while as a separate kernel it costs a launch.
    """
    offs = tl.arange(0, KB)
    scale = 1.0
    if PRO == 1:
        acc_sq = tl.zeros([KB], dtype=tl.float32)
        for k0 in range(0, K, KB):
            xv = tl.load(x_ptr + k0 + offs).to(tl.float32)
            acc_sq += xv * xv
        scale = 1.0 / tl.sqrt(tl.sum(acc_sq) / K)

    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    acc = tl.zeros([ROWS], dtype=tl.float32)
    for k0 in range(0, K, KB):
        xv = tl.load(x_ptr + k0 + offs).to(tl.float32)
        if PRO == 1:
            xv = (xv * scale).to(tl.bfloat16).to(tl.float32)   # rms_norm hands bf16 onward
        w = tl.load(w_ptr + rows[:, None] * K + (k0 + offs)[None, :]).to(tl.float32)
        acc += tl.sum(w * xv[None, :], axis=1)

    y = acc.to(tl.bfloat16)                       # autocast: every linear output is bf16
    if EPI == 1:
        tl.store(o_ptr + rows, y + tl.load(res_ptr + rows))
    elif EPI == 4:
        # residual add, then the NEXT layer's `resid*x + x0l*x0`. Both are pointwise on rows this
        # program already owns, so neither needs a launch of its own.
        xn = (y + tl.load(res_ptr + rows)).to(tl.float32)
        a = tl.load(sa_ptr).to(tl.bfloat16).to(tl.float32)
        b = tl.load(sb_ptr).to(tl.bfloat16).to(tl.float32)
        x0 = tl.load(x0_ptr + rows).to(tl.float32)
        tl.store(o_ptr + rows, ((a * xn).to(tl.bfloat16).to(tl.float32)
                                + (b * x0).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16))
    elif EPI == 2:
        r = tl.maximum(y, 0.0).to(tl.float32)     # relu in bf16, square in fp32
        tl.store(o_ptr + rows, (r * r).to(tl.bfloat16))
    elif EPI == 3:
        f = y.to(tl.float32)                      # .float() after the bf16 linear
        tl.store(o_ptr + rows, CAP * _tl_tanh(f / CAP))
    else:
        tl.store(o_ptr + rows, y)

    # The position counter, advanced by the last matvec of the step instead of by a kernel of its
    # own. Nothing after the unembedding reads `seq`, and every kernel that does has finished by
    # the time this one starts, so one program of the last launch can carry the increment. The
    # graph replays it in place, as it did the elementwise add it replaces.
    if INC:
        if tl.program_id(0) == 0:
            tl.store(seq_ptr, tl.load(seq_ptr) + 1)


def mv(x, w, out, key, pro=PRO_NONE, epi=EPI_NONE, res=None, x0=None, sa=None, sb=None,
       seq=None, inc=False):
    n_out, K = w.shape
    rows, kb, warps = MV_CONFIG[key]
    _mv[(n_out // rows,)](x, w, out, res if res is not None else x,
                          x0 if x0 is not None else x, sa if sa is not None else x,
                          sb if sb is not None else x, seq if seq is not None else x,
                          K=K, ROWS=rows, KB=kb, PRO=pro, EPI=epi, CAP=SOFTCAP,
                          INC=inc, num_warps=warps)
    return out


# ---------------------------------------------------------------------------
# The width-1 decode attention, as two Triton kernels
# ---------------------------------------------------------------------------
# fa3's cached call costs about 60 us per short-window layer at full context, against roughly
# 2 us of bytes: at batch 1 with 4 heads and one query row there is nothing to spread over the
# SMs, so a handful of CTAs walk the whole window. The long axis is the key range.

# Chosen over eight layers' caches chained, across chunk sizes from 16 to 256 and one to eight
# warps: 57.9 us for eight layers at 16 keys and one warp against 62.6 at 64 keys and two. The
# optimum at one warp says the kernel is short of programs, not of threads inside one.
# Channels per combine program, swept on this step: 16 gives 105.8 us, 32 gives 106.7, 64 gives
# 110.0 and 128 gives 130.4. Smaller blocks mean more programs -- 32 per layer at 16 channels -- and
# the redundant recomputation of the new position's score in each of them is cheap against the
# partials they read.
DECODE_COMBINE_BLOCK = 8
DECODE_CHUNK = 16
# Chunks per head, the same number at every window. A 2048-window layer's partial pass moves twice
# the cache bytes of a 1024-window layer for 19% more time (5.70 us against 4.80, profiled), so the
# short layers are not bandwidth-bound -- 65 one-warp programs over four heads is 2080 threads on a
# device with 108 multiprocessors. Sizing the chunk from the window instead of fixing it gives every
# layer the wide layer's program count.
DECODE_CHUNKS = 128
DECODE_WARPS = 1


@triton.jit
def _decode_attn_partial(q_ptr, kc_ptr, vc_ptr, seq_ptr, m_ptr, l_ptr, acc_ptr,
                         k_ptr, s_ptr, stride_t, stride_h, window, ring, scale,
                         D: tl.constexpr, CHUNK: tl.constexpr, SPLITS: tl.constexpr):
    """One head, one key chunk: a partial online softmax over cached keys."""
    h = tl.program_id(0)
    s = tl.program_id(1)
    seq = tl.load(seq_ptr).to(tl.int32)
    lo = tl.maximum(seq - window, 0)
    start = lo + s * CHUNK
    end = tl.minimum(start + CHUNK, seq)          # position `seq` is the combine pass's job

    offs_d = tl.arange(0, D)
    q = tl.load(q_ptr + h * D + offs_d).to(tl.float32) * scale
    offs_n = start + tl.arange(0, CHUNK)
    mask = offs_n < end
    # The cache holds only as many positions as the window can reach, so an absolute position maps
    # to a slot modulo that length -- one conditional subtraction, since no position this kernel can
    # see reaches twice the length; init_decode_state asserts that.
    slot_n = tl.where(offs_n >= ring, offs_n - ring, offs_n)
    base = slot_n[:, None] * stride_t + h * stride_h + offs_d[None, :]
    k = tl.load(kc_ptr + base, mask=mask[:, None], other=0.0).to(tl.float32)
    scores = tl.where(mask, tl.sum(k * q[None, :], axis=1), float("-inf"))
    m_i = tl.max(scores, axis=0)
    # A chunk can be entirely past the end -- the chunk count is fixed by the window, and early
    # in a request there are fewer keys than that. Every lane is then -inf, so subtracting m_i
    # would be inf - inf. Clamping only the subtrahend leaves p at 0 and m_i at -inf, which is
    # what the combine pass reads as an empty chunk.
    p = tl.exp(scores - tl.maximum(m_i, -1e30))
    v = tl.load(vc_ptr + base, mask=mask[:, None], other=0.0).to(tl.float32)
    tl.store(m_ptr + h * SPLITS + s, m_i)
    tl.store(l_ptr + h * SPLITS + s, tl.sum(p, axis=0))
    tl.store(acc_ptr + (h * SPLITS + s) * D + offs_d, tl.sum(p[:, None] * v, axis=0))
    # The new position's score, once per head rather than once per combine program. The combine pass
    # splits the channels, and each of its programs was recomputing this whole 128-element dot product
    # from q and the new k -- sixteen times a head at an 8-channel block, for one float.
    if s == 0:
        kn = tl.load(k_ptr + h * D + offs_d).to(tl.float32)
        tl.store(s_ptr + h, tl.sum(q * kn, axis=0))


@triton.jit
def _decode_attn_combine(q_ptr, k_ptr, v_ptr, kc_ptr, vc_ptr, seq_ptr,
                         m_ptr, l_ptr, acc_ptr, out_ptr, s_ptr, stride_t, stride_h, ring, scale,
                         D: tl.constexpr, CB: tl.constexpr, SPLITS: tl.constexpr,
                         SPAD: tl.constexpr):
    """Fold the chunks together with the new position, and append the new k and v.

    One program per (head, channel block) rather than per head. Per head this pass reduces one float
    and 128 floats for every chunk -- 129 of them at a 16-key chunk, so a quarter of a megabyte --
    and with one program a head that is four CTAs on a device with 108. The channels are independent
    for everything except the new position's score, which each program recomputes from the whole of q
    and k: 128 multiplies against the 16 KB of partials it reads.
    """
    h = tl.program_id(0)
    cb = tl.program_id(1)
    offs_c = cb * CB + tl.arange(0, CB)
    offs_d = tl.arange(0, D)
    seq = tl.load(seq_ptr).to(tl.int32)

    k_new = tl.load(k_ptr + h * D + offs_c)
    v_new = tl.load(v_ptr + h * D + offs_c)
    slot = tl.where(seq >= ring, seq - ring, seq) * stride_t + h * stride_h + offs_c
    tl.store(kc_ptr + slot, k_new)          # disjoint channels, so no program overlaps another
    tl.store(vc_ptr + slot, v_new)

    s_new = tl.load(s_ptr + h)

    offs_s = tl.arange(0, SPAD)
    live = offs_s < SPLITS
    m = tl.load(m_ptr + h * SPLITS + offs_s, mask=live, other=float("-inf"))
    l = tl.load(l_ptr + h * SPLITS + offs_s, mask=live, other=0.0)
    m_max = tl.maximum(tl.max(m, axis=0), s_new)
    w = tl.exp(m - m_max)                        # an empty chunk carries m = -inf, so w = 0
    p_new = tl.exp(s_new - m_max)
    l_tot = tl.sum(l * w, axis=0) + p_new
    acc = tl.load(acc_ptr + (h * SPLITS + offs_s)[:, None] * D + offs_c[None, :],
                  mask=live[:, None], other=0.0)
    out = tl.sum(acc * w[:, None], axis=0) + p_new * v_new.to(tl.float32)
    tl.store(out_ptr + h * D + offs_c, (out / l_tot).to(out_ptr.dtype.element_ty))


def decode_attention(q, k, v, kc, vc, seq, window, sm, sl, sacc, out, snew):
    """`fa3.flash_attn_with_kvcache`'s width-1 case: attend, then append.

    `window` is a Python constant per layer, so the number of chunks is known on the host and
    every program gets exactly one full key block. The partials belong to the request's state.
    """
    H, D = q.shape[2], q.shape[3]
    ring = kc.shape[1]
    chunk = max(8, window // DECODE_CHUNKS)
    # The partial pass covers positions [seq - window, seq), which is at most `window` keys, so
    # ceil(window / chunk) chunks reach all of them -- the new position is the combine pass's job and
    # needs no chunk of its own. The extra chunk this used to add was always empty, and it cost more
    # than one idle program: the combine pass reduces over a power-of-two padding of the chunk count,
    # so 129 chunks made it 256 lanes wide where 128 makes it 128.
    splits = -(-window // chunk)
    _decode_attn_partial[(H, splits)](
        q, kc, vc, seq, sm, sl, sacc, k, snew,
        kc.stride(1), kc.stride(2), window, ring, D ** -0.5,
        D=D, CHUNK=chunk, SPLITS=splits, num_warps=DECODE_WARPS)
    _decode_attn_combine[(H, D // DECODE_COMBINE_BLOCK)](
        q, k, v, kc, vc, seq, sm, sl, sacc, out, snew, kc.stride(1), kc.stride(2), ring, D ** -0.5,
        D=D, CB=DECODE_COMBINE_BLOCK, SPLITS=splits, SPAD=triton.next_power_of_2(splits),
        num_warps=1)
    return out


@triton.jit
def _mv3(x_ptr, wq_ptr, wk_ptr, wv_ptr, oq_ptr, ok_ptr, ov_ptr,
         idx_ptr, wte_ptr, x0_ptr, resid_ptr, x0l_ptr,
         K: tl.constexpr, ROWS: tl.constexpr, KB: tl.constexpr, NBLK: tl.constexpr,
         EMB: tl.constexpr):
    """q, k and v in one launch.

    The three matrices share an input, so they have no reason to be three launches -- but they
    are three separate parameters at three addresses, so a single flat matvec would need a
    stacked copy of them. A branch on the block index instead picks which address this program
    reads, which costs nothing: the index is uniform inside a program, so each program compiles
    to one of the three paths rather than executing all of them.
    """
    pid = tl.program_id(0)
    which = pid // NBLK
    rows = (pid % NBLK) * ROWS + tl.arange(0, ROWS)
    offs = tl.arange(0, KB)
    # The rms norm rides here rather than in a launch of its own: this program has to read the
    # whole input vector anyway, and with the three projections sharing one launch there is one
    # redundant reduction per layer instead of three.
    # Layer 0 only: the embedding row, normed and mixed, which used to be a launch of its own whose
    # whole job was a gather and a reduction over the same 512 values this program has to read
    # anyway. Every program recomputes it -- one extra reduction against one fewer launch -- and one
    # program writes x and x0 out for the residual adds and the later layers' mixing to read.
    if EMB:
        token = tl.load(idx_ptr).to(tl.int64)
        e = tl.load(wte_ptr + token * K + offs).to(tl.float32)
        e0 = (e / tl.sqrt(tl.sum(e * e) / K)).to(tl.bfloat16)
        ef = e0.to(tl.float32)
        a0 = tl.load(resid_ptr).to(tl.bfloat16).to(tl.float32)
        b0 = tl.load(x0l_ptr).to(tl.bfloat16).to(tl.float32)
        xmix = ((a0 * ef).to(tl.bfloat16).to(tl.float32)
                + (b0 * ef).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
        if pid == 0:
            tl.store(x0_ptr + offs, e0)
            tl.store(x_ptr + offs, xmix)

    acc_sq = tl.zeros([KB], dtype=tl.float32)
    for k0 in range(0, K, KB):
        if EMB:
            xv = xmix.to(tl.float32)
        else:
            xv = tl.load(x_ptr + k0 + offs).to(tl.float32)
        acc_sq += xv * xv
    scale = 1.0 / tl.sqrt(tl.sum(acc_sq) / K)
    acc = tl.zeros([ROWS], dtype=tl.float32)
    if which == 0:
        for k0 in range(0, K, KB):
            if EMB:
                xb = xmix.to(tl.float32)
            else:
                xb = tl.load(x_ptr + k0 + offs).to(tl.float32)
            xv = (xb * scale).to(tl.bfloat16).to(tl.float32)
            w = tl.load(wq_ptr + rows[:, None] * K + (k0 + offs)[None, :]).to(tl.float32)
            acc += tl.sum(w * xv[None, :], axis=1)
        tl.store(oq_ptr + rows, acc.to(tl.bfloat16))
    elif which == 1:
        for k0 in range(0, K, KB):
            if EMB:
                xb = xmix.to(tl.float32)
            else:
                xb = tl.load(x_ptr + k0 + offs).to(tl.float32)
            xv = (xb * scale).to(tl.bfloat16).to(tl.float32)
            w = tl.load(wk_ptr + rows[:, None] * K + (k0 + offs)[None, :]).to(tl.float32)
            acc += tl.sum(w * xv[None, :], axis=1)
        tl.store(ok_ptr + rows, acc.to(tl.bfloat16))
    else:
        for k0 in range(0, K, KB):
            if EMB:
                xb = xmix.to(tl.float32)
            else:
                xb = tl.load(x_ptr + k0 + offs).to(tl.float32)
            xv = (xb * scale).to(tl.bfloat16).to(tl.float32)
            w = tl.load(wv_ptr + rows[:, None] * K + (k0 + offs)[None, :]).to(tl.float32)
            acc += tl.sum(w * xv[None, :], axis=1)
        tl.store(ov_ptr + rows, acc.to(tl.bfloat16))


def mv3(x, wq, wk, wv, oq, ok, ov, emb=None):
    """`emb` is (idx, wte, x0, resid_lambda, x0_lambda) for layer 0, which builds its own input."""
    n_out, K = wq.shape
    rows, kb, warps = MV_CONFIG["qkv"]
    # The embedding path keeps the whole vector in registers, so it needs the k-block to span it.
    assert emb is None or kb == K, "the qkv k-block must cover K to fold the embedding in"
    idx, wte, x0, resid, x0l = emb if emb is not None else (x, x, x, x, x)
    _mv3[(3 * (n_out // rows),)](x, wq, wk, wv, oq, ok, ov,
                                 idx, wte, x0, resid, x0l,
                                 K=K, ROWS=rows, KB=kb, NBLK=n_out // rows,
                                 EMB=emb is not None, num_warps=warps)


@triton.jit
def _scale_norm(x_ptr, x0_ptr, resid_ptr, x0l_ptr, layer, h_ptr, C: tl.constexpr):
    """`x = resid[i]*x + x0l[i]*x0`, then `h = norm(x)`. Both scalars are fp32 parameters and
    the arithmetic runs in bf16, so they are rounded before use, as the reference's own
    type promotion does."""
    offs = tl.arange(0, C)
    a = tl.load(resid_ptr + layer).to(tl.bfloat16).to(tl.float32)
    b = tl.load(x0l_ptr + layer).to(tl.bfloat16).to(tl.float32)
    x = tl.load(x_ptr + offs).to(tl.float32)
    x0 = tl.load(x0_ptr + offs).to(tl.float32)
    xs = (a * x).to(tl.bfloat16).to(tl.float32) + (b * x0).to(tl.bfloat16).to(tl.float32)
    xs = xs.to(tl.bfloat16)
    tl.store(x_ptr + offs, xs)
    xf = xs.to(tl.float32)
    tl.store(h_ptr + offs, (xf / tl.sqrt(tl.sum(xf * xf) / C)).to(tl.bfloat16))


@triton.jit
def _rope_norm(ptr, offs, lo, cos, sin, HALF: tl.constexpr, D: tl.constexpr):
    """Rotary, then rms-norm, in place over one head's D channels."""
    a = tl.load(ptr + offs).to(tl.float32)
    partner = tl.load(ptr + tl.where(lo, offs + HALF, offs - HALF)).to(tl.float32)
    # Rounded term by term, as the reference's own bf16 arithmetic rounds it: two bf16
    # products and then a bf16 sum. Doing the whole thing in fp32 and rounding once is more
    # accurate and therefore further from `forward`, which is what the fidelity ceiling
    # measures against.
    first = tl.where(lo, a * cos, -partner * sin).to(tl.bfloat16).to(tl.float32)
    second = tl.where(lo, partner * sin, a * cos).to(tl.bfloat16).to(tl.float32)
    r = (first + second).to(tl.bfloat16).to(tl.float32)
    r = r / tl.sqrt(tl.sum(r * r) / D)
    tl.store(ptr + offs, r.to(tl.bfloat16))


@triton.jit
def _qkv_post(q_ptr, k_ptr, v_ptr, cos_ptr, sin_ptr, seq_ptr, h_ptr, idx_ptr,
              ve_ptr, gate_ptr, D: tl.constexpr, HALF: tl.constexpr, GC: tl.constexpr,
              KVDIM: tl.constexpr, C: tl.constexpr, HAS_VE: tl.constexpr):
    """One program per (head, tensor): rotary and rms-norm on q and on k, the gated value embedding
    on v.

    Twelve programs rather than four. q and k each need their head's whole 128 channels for the norm,
    so they cannot be split by channel; what they can be split by is which tensor they are. At four
    programs this kernel cost 3.8 us to move a few kilobytes, which is the shape of problem the
    combine pass had.

    In place: each program touches only its own head's own tensor, and rotary pairs channel c with
    c + HALF inside that head, so both halves are read before either is written.
    """
    h = tl.program_id(0)
    which = tl.program_id(1)
    offs = tl.arange(0, D)
    base = h * D
    pos = tl.load(seq_ptr).to(tl.int64)
    lo = offs < HALF
    ci = tl.where(lo, offs, offs - HALF)
    cos = tl.load(cos_ptr + pos * HALF + ci).to(tl.float32)
    sin = tl.load(sin_ptr + pos * HALF + ci).to(tl.float32)

    if which == 0:
        _rope_norm(q_ptr + base, offs, lo, cos, sin, HALF, D)
    elif which == 1:
        _rope_norm(k_ptr + base, offs, lo, cos, sin, HALF, D)
    elif HAS_VE:
        token = tl.load(idx_ptr).to(tl.int64)
        # the gate reads the first GC channels of norm(x); one reduction over a vector already in L2
        # is cheaper than the launch a separate norm would cost
        xall = tl.load(h_ptr + tl.arange(0, C)).to(tl.float32)
        xn = xall / tl.sqrt(tl.sum(xall * xall) / C)
        hv = tl.where(tl.arange(0, C) < GC, xn, 0.0).to(tl.bfloat16).to(tl.float32)
        gw = tl.where(tl.arange(0, C) < GC,
                      tl.load(gate_ptr + h * GC + tl.arange(0, C) % GC), 0.0
                      ).to(tl.bfloat16).to(tl.float32)
        g = tl.sum(gw * hv).to(tl.bfloat16).to(tl.float32)
        gate = (2.0 / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
        ve = tl.load(ve_ptr + token * KVDIM + base + offs).to(tl.float32)
        v = tl.load(v_ptr + base + offs).to(tl.float32)
        tl.store(v_ptr + base + offs,
                 (v + (gate * ve).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16))


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


def _ring_write(kc, vc, k, v, Tn):
    """Keep the last `kc.shape[1]` prefill positions at their cyclic slots: two contiguous copies
    at most, because a wrapped range of consecutive positions breaks in one place."""
    ring = kc.shape[1]
    keep = min(Tn, ring)
    first = Tn - keep
    start = first % ring
    cut = (-first) % ring
    if cut == 0 or cut >= keep:
        kc[:, start:start + keep] = k[:, first:]
        vc[:, start:start + keep] = v[:, first:]
    else:
        kc[:, start:] = k[:, first:first + cut]
        vc[:, start:] = v[:, first:first + cut]
        kc[:, :keep - cut] = k[:, first + cut:]
        vc[:, :keep - cut] = v[:, first + cut:]


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
        assert all(max_len <= 2 * min(max_len, w[0] + 1) for w in self.window_sizes), \
            "a cyclic cache shorter than half the context would need more than one wrap"
        C, heads = cfg.n_embd, cfg.n_kv_head
        nchunk = max(-(-w[0] // max(8, w[0] // DECODE_CHUNKS)) + 1
                     for w in self.window_sizes)
        return {
            "max_len": max_len,
            "graph_enabled": bool(graph),
            "seq": torch.zeros(batch, dtype=torch.int32, device=dev),
            # Window-extent: a layer whose window is W can never read further back than W
            # positions, so it needs W + 1 slots rather than the whole context. Six of the eight
            # layers have a half-width window, which takes the cache from 33.9 MB to about 23.
            "kc": [torch.zeros(batch, min(max_len, w[0] + 1), heads, head_dim, **kw)
                   for w in self.window_sizes],
            "vc": [torch.zeros(batch, min(max_len, w[0] + 1), heads, head_dim, **kw)
                   for w in self.window_sizes],
            # The fused step's working vectors, owned here so dropping the state frees them.
            # One residual stream, its layer-0 copy, one normalised copy, q/k/v, the MLP's
            # hidden vector and the logit row: about 30 KB in total.
            "b_x": torch.zeros(C, **kw),
            "b_x0": torch.zeros(C, **kw),
            "b_q": torch.zeros(batch, 1, heads, head_dim, **kw),
            "b_k": torch.zeros(batch, 1, heads, head_dim, **kw),
            "b_v": torch.zeros(batch, 1, heads, head_dim, **kw),
            "b_fc": torch.zeros(4 * C, **kw),
            # the chunked attention's partials and its output row
            "snew": torch.zeros(heads, dtype=torch.float32, device=dev),
            "sm": torch.zeros(heads, nchunk, dtype=torch.float32, device=dev),
            "sl": torch.zeros(heads, nchunk, dtype=torch.float32, device=dev),
            "sacc": torch.zeros(heads, nchunk, head_dim, dtype=torch.float32, device=dev),
            "sout": torch.zeros(batch, 1, heads, head_dim, **kw),
            "b_logits": torch.zeros(batch, 1, cfg.vocab_size, dtype=torch.float32,
                                    device=dev),
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
                y = fa3.flash_attn_func(q, k, v, causal=True,
                                        window_size=self.window_sizes[i])
                _ring_write(kc, vc, k, v, Tn)
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
                                                num_splits=1)
            x = x + attn.c_proj(y.contiguous().view(B, Tn, -1))
            x = x + block.mlp(norm(x))

        x = norm(x[:, -1:, :])
        softcap = 15
        logits = self.lm_head(x).float()
        return softcap * torch.tanh(logits / softcap)

    def _fused_step(self, idx, state):
        """The width-1 branch as 58 kernel launches instead of about 330.

        Same arithmetic as `_decode_body(prefill=False)`. One launch per matrix, and no launch at
        all for work that is pointwise: the input normalisation rides in the fused projection's
        prologue, and the residual add, the next layer's residual mixing, a squared relu or a tanh
        softcap ride in an epilogue. The prefill branch is untouched.
        """
        cfg = self.config
        C, H = cfg.n_embd, cfg.n_head
        D = C // H
        last = cfg.n_layer - 1
        seq = state["seq"]
        x, x0 = state["b_x"], state["b_x0"]
        q, k, v = state["b_q"], state["b_k"], state["b_v"]
        for i, block in enumerate(self.transformer.h):
            attn = block.attn
            mv3(x, attn.c_q.weight, attn.c_k.weight, attn.c_v.weight, q, k, v,
                emb=(idx, self.transformer.wte.weight, x0,
                     self.resid_lambdas, self.x0_lambdas) if i == 0 else None)
            has_ve = str(i) in self.value_embeds
            ve_weight = self.value_embeds[str(i)].weight if has_ve else x
            gate_weight = attn.ve_gate.weight if has_ve else x
            _qkv_post[(H, 3)](q, k, v, self.cos, self.sin, seq, x, idx,
                            ve_weight, gate_weight, D=D, HALF=D // 2,
                            GC=attn.ve_gate_channels, KVDIM=cfg.n_kv_head * D, C=C,
                            HAS_VE=has_ve, num_warps=1)
            y = decode_attention(q, k, v, state["kc"][i], state["vc"][i], seq,
                                 self.window_sizes[i][0], state["sm"], state["sl"],
                                 state["sacc"], state["sout"], state["snew"])
            mv(y, attn.c_proj.weight, x, "proj", epi=EPI_ADD, res=x)
            mv(x, block.mlp.c_fc.weight, state["b_fc"], "fc", pro=PRO_NORM, epi=EPI_RELU2)
            if i == last:
                mv(state["b_fc"], block.mlp.c_proj.weight, x, "mlp", epi=EPI_ADD, res=x)
            else:
                mv(state["b_fc"], block.mlp.c_proj.weight, x, "mlp", epi=EPI_ADD_SCALE,
                   res=x, x0=x0, sa=self.resid_lambdas[i + 1], sb=self.x0_lambdas[i + 1])
        mv(x, self.lm_head.weight, state["b_logits"], "head", pro=PRO_NORM, epi=EPI_SOFTCAP,
           seq=seq, inc=True)
        return state["b_logits"]

    def _prefill_graphed(self, idx, state):
        """The prefill call, replayed from a graph like the step.

        Profiled: the 1536-token prefill holds 1.34 ms of device work inside a 4.60 ms window, so
        about three quarters of it is the host enqueueing roughly ninety kernels through a compiled
        function's guards while the device waits. One replay removes that. There are two widths in a
        request measurement -- 1536 for the ordinary shape and 1 for the no-prefill probe -- so the
        graphs are keyed by width; capture failure falls back to the compiled call.
        """
        if not state.get("graph_enabled", True):
            return self._prefill(idx, state)
        graphs = state.setdefault("pgraph", {})
        T = idx.size(1)
        graph = graphs.get(T)
        if graph is None:
            graph = graphs[T] = _GraphedPrefill(self, state, T)
        if not graph.captured:
            return self._prefill(idx, state)
        return graph.replay(idx)

    def _prefill(self, idx, state):
        """The multi-token branch, compiled. One call per request against 512 steps, so the win is
        small, but it is the only part of a request still running eagerly. The cache tensors are
        unpacked out of the state: Dynamo guards a dict's Python-valued entries by value, and
        `graph_enabled` differs between the cache probe and the request probe."""
        body = getattr(self, "_compiled_prefill", None)
        if body is None:
            body = self._compiled_prefill = torch.compile(GPT._prefill_body, dynamic=False)
        return body(self, idx, state["kc"], state["vc"])

    def _prefill_body(self, idx, kc_all, vc_all):
        """`_decode_body`'s prefill branch, with the caches passed in."""
        B, Tn = idx.size()
        cos, sin = self.cos[:, :Tn], self.sin[:, :Tn]
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
            kc, vc = kc_all[i], vc_all[i]
            # Attend over k and v directly -- the reference writes them into the cache and reads them
            # back, which is the same values -- then keep the last `ring` positions, the only ones a
            # later step can reach, at their slots.
            y = fa3.flash_attn_func(q, k, v, causal=True,
                                    window_size=self.window_sizes[i])
            _ring_write(kc, vc, k, v, Tn)
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
            logits = self._prefill_graphed(idx, state)
            state["seq"].add_(idx.size(1))
            return logits, state
        if not state.get("graph_enabled", True):
            return self._fused_step(idx, state), state
        # The graph lives ON the state it was captured against, so its lifetime is exactly
        # that state's and there is no key to get wrong. A state whose capture failed holds
        # an uncaptured object and runs eager forever, which is slow and correct.
        graph = state.get("graph")
        if graph is None:
            graph = state["graph"] = _GraphedDecodeStep(self, state)
        if not graph.captured:
            return self._fused_step(idx, state), state
        return graph.replay(idx), state


class _GraphedPrefill:
    """A captured graph over one prefill width.

    The prefill reads no position counter -- it takes its rotary tables from `[:, :Tn]` -- so a
    capture is valid at any point in a request, and `seq` is advanced by the caller as before.
    Warmup and capture write junk into the cache slots this width will write, and every one of them
    is overwritten by the replay that follows; slots above the prefilled positions stay junk and are
    unreachable, exactly as they are for the step's graph.
    """

    WARMUP = 2

    def __init__(self, model, state, width):
        self.model, self.state = model, state
        self.captured, self.reason = False, ""
        self.static_idx = torch.zeros(1, width, dtype=torch.int64,
                                      device=state["seq"].device)
        self.static_logits, self.graph = None, None
        try:
            self._capture()
            self.captured = True
        except Exception as exc:             # noqa: BLE001 -- recorded, then paid for in ms
            self.reason = f"{type(exc).__name__}: {exc}"
            self.graph, self.static_logits = None, None

    def _capture(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(self.WARMUP):
                self.model._prefill(self.static_idx, self.state)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_logits = self.model._prefill(self.static_idx, self.state)

    def replay(self, idx):
        self.static_idx.copy_(idx)
        self.graph.replay()
        # The caller holds these logits while the next call overwrites the buffer, and a prefill
        # happens once a request, so the copy costs one launch and removes the aliasing question.
        return self.static_logits.clone()


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
        return self.model._fused_step(self.static_idx, self.state)

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

# Serving precision, applied once, after training and before anything is measured.
#
# Every forward runs under autocast, which casts each fp32 weight to bf16 before its matmul, so
# casting the parameter itself changes no arithmetic: checked directly, the maximum absolute logit
# difference over 8 x 2048 x 8192 logits is exactly 0.0. What it changes is that these kernels read
# half the bytes -- they load the weight and convert in registers, so with fp32 parameters a step
# reads about 50 MB of weights, and a step sweeps the key/value cache through L2 between those
# reads, which evicts them.
#
# Nothing prepare.py measures moves: val_bpb is computed on this model below and is bit-identical,
# the parameter count counts elements, and peak memory is already the training high-water mark,
# which a cast afterwards cannot raise.
#
# Every floating parameter, including the rank-1 per-layer scalars. Those multiply bf16 tensors as
# 0-dim operands and PyTorch computes such a product in the dimensioned operand's dtype, so they were
# already being rounded to bf16 on every use; and the kernels that load them round them explicitly.
with torch.no_grad():
    for p in model.parameters():
        if p.dtype == torch.float32:
            p.data = p.data.bfloat16()

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
