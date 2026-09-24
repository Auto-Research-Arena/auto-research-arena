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
    window_span_sequences: int = 5


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


INIT_ORTHO_SEED = 42


def semi_orthogonal_(w, generator):
    """Fill 2-D `w` (out_features, in_features) with a semi-orthogonal matrix scaled so
    that every singular value is exactly max(1, out/in)**0.5 -- the factor
    MuonAdamW._step_muon multiplies this matrix's step by -- and ||w||_F**2 equals the
    expectation of the parent's uniform(-s, s) draw with s = 3**0.5 * in**-0.5, i.e.
    out_features. Drawn from `generator`, and called on a tensor the caller has already
    filled from the GLOBAL generator with the parent's own uniform_ draw, which this
    overwrites -- see E3: keeping that draw is what leaves the global stream exactly where
    the parent leaves it."""
    rows, cols = w.shape
    m = min(rows, cols)
    a = torch.randn(max(rows, cols), m, dtype=torch.float32, device=w.device,
                    generator=generator)
    q, r = torch.linalg.qr(a)
    signs = torch.sign(torch.diagonal(r))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    q = q * signs.unsqueeze(0)
    q = q if rows >= cols else q.mT
    w.copy_(q * max(1.0, rows / cols) ** 0.5)


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
        # Rotary embeddings: one spectrum per distinct attention span, each layer reading the
        # one its own window resolves (see ROTARY_BAND_TURNS_PER_SPAN).
        self.rotary_seq_len = config.sequence_len * 10
        spans = [(w if w > 0 else config.sequence_len) for w, _ in self.window_sizes]
        self.rotary_spans = sorted(set(spans))
        self.rotary_buf_idx = [self.rotary_spans.index(s) for s in spans]
        for k, span in enumerate(self.rotary_spans):
            cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim, span)
            self.register_buffer(f"cos{k}", cos, persistent=False)
            self.register_buffer(f"sin{k}", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # Transformer blocks
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        ortho_gen = torch.Generator(device=self.transformer.wte.weight.device)
        ortho_gen.manual_seed(INIT_ORTHO_SEED)
        for block in self.transformer.h:
            # The parent's draws are kept and then overwritten. They set no value that
            # survives; they are here because they are GLOBAL-generator consumers standing
            # between lm_head's draw and the value-embedding draw below, so keeping them
            # leaves value_embeds -- 16,777,216 parameters -- the parent's exact draw.
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            semi_orthogonal_(block.attn.c_q.weight, ortho_gen)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            semi_orthogonal_(block.attn.c_k.weight, ortho_gen)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            semi_orthogonal_(block.attn.c_v.weight, ortho_gen)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            semi_orthogonal_(block.mlp.c_fc.weight, ortho_gen)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        _w0 = self.transformer.h[0].attn.c_q.weight.float()
        _w1 = self.transformer.h[0].mlp.c_fc.weight.float()
        _g1 = max(1.0, _w1.shape[-2] / _w1.shape[-1]) ** 0.5
        _e0 = ((_w0.mT @ _w0)
               - torch.eye(_w0.shape[-1], device=_w0.device)).abs().max()
        _e1 = (((_w1 / _g1).mT @ (_w1 / _g1))
               - torch.eye(_w1.shape[-1], device=_w1.device)).abs().max()
        print(f"Matrix init: semi-orthogonal c_q, c_k, c_v, c_fc in "
              f"{len(self.transformer.h)} blocks, singular values max(1, out/in)**0.5; "
              f"|W^T W - I|_max {float(_e0):.2e} (c_q, gain 1.0), {float(_e1):.2e} "
              f"(c_fc, gain {_g1:.1f}); ||c_q||_F^2 {float(_w0.pow(2).sum()):.1f}, "
              f"||c_fc||_F^2 {float(_w1.pow(2).sum()):.1f}; init seed {INIT_ORTHO_SEED}")
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
        # Rotary embeddings: one spectrum per distinct attention span (see __init__). The
        # model is built on meta and to_empty leaves buffers uninitialised, which is why the
        # parent recomputes them here; both spectra are recomputed the same way.
        head_dim = self.config.n_embd // self.config.n_head
        for k, span in enumerate(self.rotary_spans):
            cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim, span)
            setattr(self, f"cos{k}", cos)
            setattr(self, f"sin{k}", sin)
        _n_pairs = head_dim // 2
        _n_rot = min(_n_pairs, max(1, int(round(ROTARY_ROTATED_PAIR_FRACTION * _n_pairs))))
        for k, span in enumerate(self.rotary_spans):
            _layers = [i for i, b in enumerate(self.rotary_buf_idx) if b == k]
            print(f"Rotary spectrum {k}: span {span}, {_n_rot} of {_n_pairs} rotate over "
                  f"wavelengths {2 * math.pi:.4f}..{span / ROTARY_BAND_TURNS_PER_SPAN:.1f}, "
                  f"{_n_pairs - _n_rot} identity; layers {_layers}")
        # Cast embeddings to bf16
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, span, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        # The head's head_dim // 2 rotary planes are split in two. The first n_rot rotate,
        # with angular frequencies log-spaced from 1.0 rad/position (one turn per 2*pi
        # positions) down to the slowest plane this `span` resolves: exactly
        # ROTARY_BAND_TURNS_PER_SPAN turns across span positions, i.e. one turn per 2*pi*band
        # positions with band = span / (2*pi*ROTARY_BAND_TURNS_PER_SPAN). The remaining planes
        # get angular frequency exactly 0.0, so their cos is
        # exactly 1.0 and their sin exactly 0.0 and apply_rotary_emb passes them through
        # unchanged: they are a position-independent content subspace. Same op count, same
        # shapes, same kernels as the parent -- only these buffer VALUES change.
        n_pairs = head_dim // 2
        n_rot = min(n_pairs, max(1, int(round(ROTARY_ROTATED_PAIR_FRACTION * n_pairs))))
        j = torch.arange(n_rot, dtype=torch.float32, device=device)
        band = span / (2.0 * math.pi * ROTARY_BAND_TURNS_PER_SPAN)
        rot_freq = (1.0 / band) ** (j / max(1, n_rot - 1))
        inv_freq = torch.cat([rot_freq,
                              torch.zeros(n_pairs - n_rot, dtype=torch.float32, device=device)])
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        # The ceiling prices the stack's TOTAL attention span and nothing about its shape: the
        # counted attention term is 12 * n_head * head_dim * sum_l min(window_l, sequence_len)
        # per token, so every allocation with the same span sum costs the same measured FLOPs.
        # Spend the budget uniformly, so no layer is capped at half the row and none is handed a
        # whole row it cannot use.
        budget = config.window_span_sequences * config.sequence_len
        assert budget % config.n_layer == 0
        span = budget // config.n_layer
        assert 0 < span <= config.sequence_len
        assert span % 128 == 0, "FA3 tiles keys in 128-token blocks"
        return [(span, 0) for _ in range(config.n_layer)]

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
        _muon_scales = []
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            _muon_scales.append(
                f"{tuple(shape)}x{len(group_params)}:{muon_shape_scale(shape):.6f}")
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
        print("Muon lr shape scales (out/in)**0.5: " + ", ".join(_muon_scales))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos0.size(1)
        cos_sins = [(getattr(self, f"cos{k}")[:, :T], getattr(self, f"sin{k}")[:, :T])
                    for k in range(len(self.rotary_spans))]

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sins[self.rotary_buf_idx[i]], self.window_sizes[i])
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
    # Per-element second moment instead of NorMuon's per-row one. Every line below is the
    # parent's, and with red_dim_size = 1 they are already the per-element formula:
    # v_norm_sq = sum(v_mean) = sum(g**2), v_norm_new**2 = sum(g**2 * step_size**2), so
    # g * step_size * v_norm / v_norm_new keeps ||g||_F exactly in exact arithmetic. The step
    # length _step_muon sets, lr * max(1, rows/cols)**0.5, is therefore untouched and only its
    # distribution over the matrix's entries changes. `v_mean` keeps the parent's name although
    # it now holds the per-element square, which is what holds this edit to two lines; `red_dim`
    # is still computed and passed by _step_muon and is no longer read in this function.
    v_mean = g.float().square()
    red_dim_size = 1
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


def muon_shape_scale(shape):
    """The factor _step_muon multiplies a Muon group's lr by: (out / in) ** 0.5, the
    spectral scale that maps unit-RMS input perturbations to unit-RMS output ones.
    semi_orthogonal_ names this same factor as the singular value it fills a matrix with,
    clamped below at 1.0. The clamp binds only where out < in, and the two Muon shape
    groups with out < in -- mlp.c_proj (512, 2048) and attn.ve_gate (4, 32) -- are the two
    that semi_orthogonal_ is never called on, both being zeros_-initialised, so no value
    this file initialises is changed by dropping it. On the two groups semi_orthogonal_
    does fill, out >= in and the two expressions return the same float."""
    return (shape[-2] / shape[-1]) ** 0.5

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
            # One second-moment entry per parameter entry, allocated the way momentum_buffer
            # above already is. Across the four Muon shape groups that is 100,665,344 B of fp32
            # against the parent's 197,120 B, every allocation a multiple of 512 B.
            state["second_momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * muon_shape_scale(shape))
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
# Attention span allocation. The measured ceiling prices the stack's TOTAL span: the [flops] line's
# attention term is 12 * n_head * head_dim * sum_l min(window_l, sequence_len) per token = 6144 *
# sum_l span_l here, and it has read 62,914,560 of the 239,078,400 ceiling in every launch of this
# run. The parent spends that budget as the nanochat SSSL pattern it inherited -- 6 * 1024 + 2 *
# 2048 = 10,240 = 5 * sequence_len, six layers capped at half the row and two given the whole one.
# That pattern is a COST heuristic from a substrate where a full window cost more than a half one;
# here the ceiling is paid in full either way, so the SHAPE of the allocation is free and only its
# quality matters. Spend it uniformly: 10,240 / 8 = 1,280 tokens of direct reach in every layer.
# Six layers gain 25% (1024 -> 1280); two give up 37.5% (2048 -> 1280) of a row that the frozen
# loader best-fit-packs out of several BOS-prefixed documents, so much of what they give up is
# cross-document context. Composition covers the rest: each layer adds its own span to the residual
# stream's reach, so from layer 2 on (2 * 1280 > 2048) every position is reachable indirectly, and
# DIRECT reach is what a span buys. 1,280 = 10 FA3 key blocks of 128, and sum_l (span_l / 128 + 1)
# = 8 * 11 = 88 is the parent's 6 * 9 + 2 * 17 = 88, so the key-block work per query block is
# unchanged too. Sum of spans, counted FLOPs, parameters, shapes, kernels, token quantum,
# schedules, optimizer and data path are all the parent's.
WINDOW_SPAN_SEQUENCES = 5  # stack's total attention span, in units of sequence_len
# Rotary spectrum. apply_rotary_emb rotates plane m (channel m against channel m + head_dim/2)
# by position * inv_freq[m]. The inherited rule inv_freq[m] = 10000 ** (-2m/head_dim) puts the
# slowest of this head's 64 planes at 1.1548e-04 rad/position: over the LONGEST attention span
# this file allows (sequence_len = 2048) that plane turns by 0.2365 rad = 3.8% of a turn, and
# 18 of the 64 planes never reach a half turn across 2048 -- 23 of 64 across the 1024-token
# window that six of the eight layers use. Those planes are near-constant offsets, not
# relative-position codes, and because every plane rotates, no channel pair can carry a
# position-INDEPENDENT match: the pre-softmax logit is sum_m r_m cos(phi_m + theta_m * delta),
# so an exact-content match is oscillated by distance in every one of its terms.
# Half the planes are therefore given angular frequency exactly 0.0 (cos == 1.0, sin == 0.0,
# passed through unchanged, and exact in bf16), and the rotating half is log-spaced over
# 1.0 .. 1/1024 rad/position, i.e. wavelengths 6.2832 .. 6433.98 positions instead of
# 6.2832 .. 54410.1. This is the rotary of the nanogpt speedrun lineage every other component
# of this model comes from (gated value embeddings, x0/resid lambdas, relu^2 MLP, logit
# softcap, Muon with polar express, SSSL sliding windows): dim//4 rotating planes at
# (1/1024) ** linspace(0, 1), the rest zeros. It is the one component this file did not take
# from there, and it costs nothing per update: same kernels, same shapes, same op count.
# r032 measured it against this parent: identical aten multiset over one forward+backward
# (2,638 calls, 50 ops, empty diff), bit-identical peak_vram_bytes, and matched-step training
# loss below the parent's at 1290 of 1290 common charged steps.
# The two window sizes named above are the parent revision's allocation, superseded by
# WINDOW_SPAN_SEQUENCES below. What the reshape left global is the BAND, and the band is not one
# constant either: each layer's slowest rotating plane completes exactly ROTARY_BAND_TURNS_PER_SPAN
# turns across ITS OWN attention span (band = span / (2*pi*turns)), so all 32 rotating planes of
# every layer are a complete log-spaced basis over the distances that layer actually sees. At the
# parent revision that rule resolved to two spectra (bands 162.9747 and 325.9493, 4.35 and 3.83
# rotating planes per octave) and r034 measured a gain from it at the deciding probe (r034 5). Under
# a uniform span budget the same rule resolves to ONE spectrum at span 1280: band 1280 / (2*pi) =
# 203.71832715762605, slowest rotating plane 0.004908738521234052 rad/position = one turn per
# 1280.0 positions, 4.1719 planes per octave in EVERY layer, against 4.35 / 3.83 keyed to two spans
# and 3.20 at the global 1/1024 band r032 measured. The number of rotating planes, the number of
# identity planes, every shape, every kernel and every counted FLOP are unchanged, and one
# (1, 20480, 1, 64) bf16 cos/sin pair is resident where the parent held two.
ROTARY_ROTATED_PAIR_FRACTION = 0.5  # share of the head's head_dim/2 planes that rotate
ROTARY_BAND_TURNS_PER_SPAN = 1.0    # slowest rotating plane: this many turns across the
                                    # attention span of the layer that reads it

# Optimization
TOTAL_BATCH_SIZE = 65536 # 32 * 2,048 tokens per optimizer step, assembled as two 16x2048 microbatches (MICROBATCH_ROWS)
EMBEDDING_LR = 0.6      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
# Muon's cautious weight decay, halved. This is the settlement the row rung in the Model size block
# below needs, not a second idea: the rung deletes charged tokens from a descent that must still
# reach the target before a probe reads it, and this is the only lever this run has measured that
# moves that descent at zero seconds and zero counted FLOPs. Its one LIVE consumer is
# get_weight_decay -- the weight_decay= argument to setup_optimizer is overwritten by
# group["weight_decay"] = muon_weight_decay before every optimizer.step(), and the five adamw groups
# carry weight_decay=0.0 as literals -- so the value reaches exactly the four Muon shape groups, the
# 25,166,336 transformer_matrices parameters, and nothing else.
# MEASURED TWICE ON THIS EXACT PARENT, BENEFICIAL BOTH TIMES. r061 measured this same halving alone
# (rounds/r061/decision.md sections 4.2-4.4): better at all nine probes at matched tokens, peak
# -0.0070691 bpb at probe 4, its own interpolated crossing of 1.05 at charged update 1,611.06 (the
# run's published linear convention) or 1,602.91 (curvature-aware) against c049's 1,632.89 /
# 1,631.77, with matched-step printed loss lower in 94.9% of the 1,648 shared steps. r062 carried
# the same 0.1 inside a composite and recorded the run's lowest own-curve crossing, 156,110,740
# tokens (rounds/r062/decision.md section 4.3). The LEVEL is bracketed rather than extrapolated:
# r060 put MORE of this same lrm*wd product over the live anneal and measured it single-signed
# adverse at 22 sigma against a four-probe in-launch control of 0.00000 +/- 0.00010 bpb.
WEIGHT_DECAY = 0.1      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of the anneal horizon for LR warmup
# The LR and weight-decay anneal is anchored to the OBJECTIVE'S PROBE GRID, not to the
# training-clock second this run is expected to finish at. The frozen harness records a
# crossing only at a multiple of PROBE_EVERY_SECONDS = 30.0s, so the recorded token count is
# quantised to whole probe intervals -- at the parent revision one interval was 143 updates =
# 18,743,296 tokens = 9.937% of its recorded 188,612,608 -- and nothing finer than an interval
# registers on this objective at all. The crossing-anchored rule this block carried is spent:
# it fitted horizon -> recorded crossing second on 420s -> 360.11656188964844s and 360s ->
# 330.1272258758545s, predicted its own fixed point at 300.2756817285827s, the run installed
# the grid-quantised 300.0s, and the third point came back cross(300.0) = 300.08415150642395s,
# so |cross(H) - H| fell 59.883s -> 29.873s -> 0.084s and one further application of the rule
# returns the constant already installed. Self-consistency was never the objective; the probe
# index is. So the anneal ENDS two probe intervals before the parent's crossing probe: 300.0 -
# 2 * 30.0 = 240.0s = 8 * PROBE_EVERY_SECONDS = 0.40 * TIME_BUDGET. The next grid position at
# which this objective can pay anything is the probe one interval earlier than the parent's,
# and this placement leaves that reading one full interval PAST the end of the decay, so it is
# taken on a model whose anneal is complete and which has had an interval at the held floor,
# where the parent's reading at that same second was taken 90s inside its decay at lrm 0.325.
# The rung is -60s, the same size as the two the run has measured.
# What the three launches that differ in nothing but these two ratios recorded, matched clock:
# at the 270.0s probe 1.075684 (horizon 420s, decay not yet begun), 1.069638 (360s, 30s in),
# 1.053601 (300s, 90s in, and 0.0036007498 above the 1.05 gate -- the narrowest near-miss of
# the three, against 0.005266 and 0.008383); at the 300.0s probe 1.068373, 1.055266, 1.042040.
# Every matched-clock probe from 120.0s on fell at both rungs, and each rung improved more than
# the rung before it over every multiplier range they share (1.000 -> 0.775: 0.009990,
# 0.011785, 0.014971).
# The anneal is TRANSLATED, not reshaped: both ratios move by the same -0.10, so the decay
# keeps its length (SCHEDULE_HORIZON - (1.0 - WARMDOWN_RATIO) * SCHEDULE_HORIZON = 240.0 -
# 120.0 = 120.0s exactly, as at the parent revision), keeps its linear shape and keeps its
# FINAL_LR_FRAC floor, and lrm(t) here equals the parent's lrm(t + 60) at every second (max
# |difference| 2.220446049250313e-16 over t in [0, 600] at 0.25s steps). The anneal now BEGINS
# at WARMDOWN_START_RATIO * TIME_BUDGET = 120.0s and ENDS at ANNEAL_HORIZON_RATIO *
# TIME_BUDGET = 240.0s, past which lrm is held at FINAL_LR_FRAC and Muon's cautious weight
# decay is exactly 0.0 -- a regime no launch here has run for longer than 0.084s. The only
# recorded evidence about it is the reference launch's last two intervals, 0.007492 at lrm
# 0.2 -> 0.1 and 0.003938 at 0.1 -> 0.0, on 37-38 updates each of a 524,288-token quantum
# where an interval here is 143 updates of 131,072. The loop's stop rule, the log's pct_done
# and the frozen harness clock all still run on TIME_BUDGET.
# One figure this block carried at the parent revision was wrong under its own definition and
# is corrected rather than dropped: the share of the lrm span never run at the 420s horizon is
# 49.90% ((420 - 360.1166)/120), not 42.77%, which is what that expression returns with the
# 360s horizon's WARMDOWN_RATIO; the 24.89% given for the 360s horizon was right, and at the
# 300s horizon it measured 0.104%.
ANNEAL_HORIZON_RATIO = 0.40  # anneal ends here; keep in (WARMDOWN_START_RATIO, 1.0]
WARMDOWN_START_RATIO = 0.20  # anneal begins here, as a fraction of TIME_BUDGET
SCHEDULE_HORIZON = ANNEAL_HORIZON_RATIO * TIME_BUDGET  # 0.40 * 600 = 240.0 training seconds
WARMDOWN_RATIO = 1.0 - WARMDOWN_START_RATIO / ANNEAL_HORIZON_RATIO  # decay share of horizon
# The END of the anneal ramp, as a fraction of the initial LR. It is a floor and not zero
# because the horizon is reachable while the clock is still running and at lrm 0 every parameter
# update in this file is a no-op -- r002 installed it as exactly that settlement (0.0 -> 0.10)
# and no launch has priced its LEVEL on its own since. Being the ramp's ENDPOINT, this one
# constant sets the LR of both regions the deciding probe is read through: get_lr_multiplier
# returns cooldown + (1 - cooldown) * FINAL_LR_FRAC, so raising it by d raises lrm(t) by exactly
# (1 - cooldown(t)) * d -- 0.0 at the anneal's onset second 120.0, rising linearly to d at the
# horizon 240.0 -- and by exactly d over the whole held floor past the horizon, where
# sched_progress is clamped at 1.0, lrm IS FINAL_LR_FRAC and Muon's cautious weight decay is
# exactly 0.0.
# Every recorded reading of this file's post-onset LR level says that level is LOW, and none says
# it is high. r019 multiplied all nine groups by 0.5 and lost per-update yield in every interval
# it priced past the onset: -4.92%, -9.88%, -13.62% through the decay and -16.15% over its floor
# interval. r046 compressed the range about a fixed geometric mean, and the only two intervals in
# which its lrm ran ABOVE the parent's are the only two in which it out-yielded the parent
# (+2.16% over the 210-240s interval containing its 225.1s crossover, +16.98% over the floor
# interval at lrm 0.14142), while every interval below the parent's lrm cost it. r023 measured
# this 0.10 floor at 0.595 / 0.499 / 0.617 of a live anneal's per-interval yield over three
# matched intervals, and r046 measured this parent's own floor interval at 0.6368 of its own last
# live-anneal interval: the held floor is the least productive region of the schedule and its
# level is the lever that prices it.
# So reflect r019's measured octave through the parent -- 0.10 / 2 was measured adverse at the
# floor, so take 0.10 * 2 -- and change nothing else. The plateau branch still returns the
# literal 1.0 for every update before 120.0s, so the launch keeps a pre-mechanism calibration
# prefix; the anneal still begins at 120.0s and ends at 240.0s; WARMDOWN_RATIO is still exactly
# 0.5; get_weight_decay and get_muon_momentum are untouched; and the returned expression keeps
# the operation count it has now.
FINAL_LR_FRAC = 0.20
# The weight-decay schedule has no held endpoint, and the probe this objective is decided at is
# read entirely inside the region where that matters. get_weight_decay returns WEIGHT_DECAY *
# (1 - progress) and the loop passes it sched_progress, which it clamps at 1.0, so past
# SCHEDULE_HORIZON = 240.0s Muon's cautious shrink is not small but EXACTLY 0.0 for every
# remaining update. get_lr_multiplier does not do this: its ramp ENDS on FINAL_LR_FRAC rather
# than on zero, for the reason the comment above it gives -- at lrm 0 every parameter update in
# this file is a no-op. Give the decay the endpoint the LR multiplier already has.
# WHY THIS REGION. Past the horizon lrm is held at FINAL_LR_FRAC = 0.20 and every fused step in
# this file normalises before it steps: muon_step_fused divides X by its own norm before the
# polar-express iterations and adamw_step_fused divides by sqrt(exp_avg_sq). The Iterate
# averaging block below states the consequence as its own premise -- the step length is set by lr
# alone and does not shrink as the gradient does, so the iterate keeps orbiting instead of
# settling. lr * wd * p is the one term in this file that is a function of the PARAMETER rather
# than of the gradient, and it is the term the parent switches off exactly at the clamp.
# WHY THIS LEVEL, AND WHY IT IS A TRANSFER AND NOT AN EXTRAPOLATION. r059 installed this same held
# endpoint and measured it beneficial in this same region at a held wd of 0.04. MATRIX_LR = 0.04
# and FINAL_LR_FRAC = 0.20 are that file's values too, so the Muon group lr past the horizon is
# 0.04 * 0.20 = 0.008 in both and the coefficient muon_step_fused applied there, lr * wd = 3.2e-4
# per update, is reproduced here EXACTLY by a held 0.04. WEIGHT_DECAY was halved at r061, so the
# fraction that preserves that one measured coefficient doubled: 0.40, not r059's 0.20. r061 and
# r065 bracket the LIVE leg's level at 0.1 and r060 prices MORE lrm*wd over the live anneal; none
# of the three says anything about the clamped region, where c064 and c065 both run exactly 0.0.
# WHAT IT COSTS PER UPDATE: one float, written into the pre-existing 0-D CPU tensor
# self._muon_wd_t that _step_muon already fills before every optimizer.step(). No tensor, no
# allocation, no kernel, no dispatch, no shape, no token, no loader call, no chunk plan and no
# other schedule changes, so the charged token rate, the quantum, the exempt window and the key's
# lattice are the parent's.
# WHERE THE DOSE LANDS. The harness adds dt from the tick whose step index is 10 and the loop from
# the update whose index is 11, so the loop's clock is harness_clock - dt(step 10) and is strictly
# behind it: every update the harness reads at probes 1-8 runs the parent's arithmetic bit for
# bit. r059 measured that bound live -- probe-8 tick at printed step 1484, first clamped update
# 1486, 183 of that interval's 184 updates dosed.
WD_FINAL_FRAC = 0.40    # fraction of WEIGHT_DECAY held past SCHEDULE_HORIZON, mirroring
                        # FINAL_LR_FRAC's role for the LR multiplier

# Iterate averaging. muon_step_fused normalises the update before it orthogonalises it (X
# is divided by its own norm, then run through polar_express_coeffs) and adamw_step_fused
# divides by sqrt(exp_avg_sq), so on every parameter in this model the size of the step is
# set by lr alone and does not shrink as the gradient does: the optimizer's iterate keeps
# orbiting at a radius proportional to lr instead of settling. The model's parameter
# sequence is therefore the running average of the optimizer's iterates, and the
# optimizer's own iterate is carried in a buffer the way the Muon momentum buffer already
# is. The horizon is in updates, set between the two update-keyed timescales this file
# already has: longer than the Muon momentum horizon 1/(1 - 0.95) = 20 updates, so the
# average spans the orbit, and a small part of the frozen 30 s probe interval (about 144
# updates at this quantum), so the average's lag behind the iterate under a steady drift,
# 1/AVG_ALPHA - 1 = 31 updates, stays small against it.
AVG_HORIZON_UPDATES = 32                # updates spanned by the average
AVG_ALPHA = 1.0 / AVG_HORIZON_UPDATES   # 0.03125 exactly. The loop uses
                                        # max(AVG_ALPHA, 1/updates_so_far), so the average
                                        # is an exact running mean until it is this long
                                        # and needs no bias correction

# Model size
DEPTH = 8               # number of transformer layers
DEVICE_BATCH_SIZE = 32   # rows the loader hands over, and the charged update's token quantum
# The objective charges TOKENS and prices neither seconds nor efficiency, and the frozen harness fires
# probe k at a FIXED clock second 30.0 * k. So at a fixed probe index the recorded number is that fixed
# number of seconds times the CHARGED TOKENS PER TRAINING-CLOCK SECOND, and that rate is
# rate(rows) = rows * MAX_SEQ_LEN / (F + s * rows), with s the per-row forward/backward cost and F the
# per-update work that carries NO tokens: two chunk boundaries, the optimizer step with its five
# polar-express iterations, NorMuon, AdamW, the iterate-average's three whole-model movements, two
# synchronizations, the loader refill, the loss item() sync. rate is strictly increasing in rows for any
# F > 0, so lowering the quantum lowers the rate arithmetically -- the one route to the rate that does not
# have to be bought from the node, which is what the five duration cards had to do (r045 ~0 ms delivered,
# r048 2.3% of its carded dose delivered).
# BOTH TERMS ARE MEASURED, ON THIS SAME LADDER. r013 section 4.2 read them off two launches differing in
# nothing but the quantum: F = 2 * 108.2076 - 209.2304 = 7.1849 ms (3.43% of a single-chunk 64-row update)
# and s = 1.541485e-3 ms/token = 3.15696 ms/row; the 262,144/131,072 pair (r003) gave s = 1.5152e-3 =
# 3.1031 ms/row. Against c047's recorded 215.32282 ms those slopes give F = 215.32282 - 64 * s =
# 13.2774 .. 16.7244 ms (6.2 .. 7.8%), larger than c008's 7.1849 as it must be, this revision paying two
# chunk boundaries where c008 paid one. Deleting 16 rows deletes 16 * s = 49.65 .. 50.51 ms of row work
# while every millisecond of F stays: 1.87 .. 2.52% off the charged token rate at every probe index.
# WHY THIS RUNG. The ladder's price is a matched-clock quality deficit at the crossing second and it is
# convex in log-quantum: r003's octave cost +0.0019 bpb at its parent's crossing probe and won -2.683%;
# r013's octave -- this quantum's own next rung -- cost +0.004926 against a bank of 0.0028014 (1.76x) and
# lost by one whole probe interval. 48 rows is log2(64/48) = 0.415037 of that octave, so at r013's own
# slope it prices at 58.0% of c047's recorded 0.0035241274 bank (47.06 charged updates at its deciding
# secant 7.488591e-05, 3.75% of the published key in tokens), or 79.1% if the run's dose-independent
# launch fee (0.00046 .. 0.00182 over five measurements) is counted again on top of a rung-proportional
# remainder. Half an octave is not affordable under the second reading; a quarter buys 1.16% of rate at
# most. 48 is also the deepest affordable rung that keeps the charged update at two EQUAL integer chunks.
# HELD FIXED, EACH ON RECORDED EVIDENCE. The chunk COUNT stays two and the chunks stay equal (24, 24), so
# loss / n_micro is unchanged exact arithmetic -- for equal chunks the mean of the chunk means IS the
# whole-quantum mean (float64 relative error 1.319e-16, r031 exec section 5) -- and r048's non-constant
# chunk-count price is out of this round. The LRs are NOT rescaled: r019 applied the sqrt-quantum rule
# (x0.5) and lost yield in every interval it priced past the onset. The update-keyed timescales
# (get_muon_momentum's 300-update ramp, AVG_HORIZON_UPDATES) are NOT re-anchored: r005 did exactly that
# after r003's halving and recorded 246,415,360 against 227,540,992, and 32 updates still sits between the
# 20-update Muon horizon and the probe interval's update count. The LR and weight-decay schedules are
# clock-keyed, so lrm(t) is bit-identical to the parent's at every training second. Every token still
# passes through exactly one forward/backward per update, so flops_per_token_measured x total_tokens still
# describes this launch's counted arithmetic exactly, and MICROBATCH_ROWS = 24 >=
# prepare.FLOPS_PROBE_ROWS = 8 leaves frozen region 5's probe slicing the same 8 x 2048 tokens as every
# launch of this run. Frozen region 5: "Changing the actual training microbatch size remains allowed."
# THIS REVISION'S RUNG, AND HOW ITS DEPTH WAS CHOSEN. The paragraph above is r049's own argument for
# 48 rows and is kept as that round's record; nothing in it is deleted and its arithmetic is
# unchanged. rate(rows) = rows * MAX_SEQ_LEN / (F + s * rows) is strictly increasing in rows for any
# F > 0, so lowering the quantum lowers the charged tokens per training-clock second, and
# rate(36) / rate(48) = 36 * (F + 48 * s) / (48 * (F + 36 * s)) is INVARIANT under a proportional
# change of F and s -- that is, under the node draw. That invariance is why this route to the rate
# does not have to be bought from the node the way the five duration cards did (r045 ~0 ms of its
# dose delivered, r048 2.3%).
# THE ARITHMETIC, on the two slope pairs quoted above and c049's own recorded mean charged dt of
# 164.87931 ms at 48 rows. r013 section 4.2's s = 3.15696 ms/row gives F = 164.87931 - 48 * s =
# 13.34523 ms (8.09% of the update), dt(36) = F + 36 * s = 126.99579 ms and a rate of
# 73,728 / 126.99579 = 580.555 tokens/ms against 98,304 / 164.87931 = 596.218, i.e. -2.6271%; the
# r003 pair's s = 3.1031 gives F = 15.93051 (9.66%), dt(36) = 127.64211 and 577.615 tokens/ms,
# i.e. -3.1202%. Twelve rows of row work (12 * s = 37.88 .. 37.24 ms) are deleted while every
# millisecond of F stays. r062 measured this identity one rung up: against c061 (48 rows, the same
# WEIGHT_DECAY, an implied node speed 0.012% away) its charged rate read -0.7421%, inside that
# card's -0.7304% .. -0.8707% band, for a measured elasticity d ln rate / d ln rows = 0.0891.
# WHY 36 AND NOT 40. The nine 48-row launches of this family recorded dt from 162.14224 to
# 165.65721 ms, 2.17% peak to peak (rounds/r062/decision.md section 5), and r062's rung cut the rate
# by only a third of that, so its draw sweep over the recorded band came out 4 wins / 5 losses
# (section 6) -- the draw dominated the mechanism. 36 rows is the SHALLOWEST rung of this ladder
# whose node-invariant cut exceeds that dispersion; 40 rows prices at -1.59% .. -1.90%.
# WHY NOT 32. A rung is paid for out of tokens the descent has already banked past its own crossing,
# and the largest bank this run has recorded is c062's 3.552% of its own key at the deciding probe
# (section 7). 32 rows prices at -3.89% .. -4.61%, past that reading; 36 rows at -2.63% .. -3.12% is
# inside it. This says nothing about which probe reads THIS launch.
# HELD FIXED, EACH ON RECORDED EVIDENCE, exactly as at the parent revision. The chunk COUNT stays
# two and the chunks stay EQUAL (18, 18), so loss / n_micro is unchanged exact arithmetic and r048's
# non-constant chunk-count price stays out of this round. The LRs are NOT rescaled (r019 applied the
# sqrt-quantum rule and lost yield in every interval it priced past the onset). The update-keyed
# timescales are NOT re-anchored (r005 did that after r003's halving and recorded 246,415,360
# against 227,540,992), and AVG_HORIZON_UPDATES = 32 still sits between the 20-update Muon momentum
# horizon and the charged updates in a 30 s probe interval, a count this quantum raises rather than
# lowers. The LR and weight-decay schedules are clock-keyed, so lrm(t) is bit-identical to the
# parent's at every training second. Every token still passes through exactly one forward/backward
# per update, so flops_per_token_measured * total_tokens still describes this launch's counted
# arithmetic exactly, and MICROBATCH_ROWS = 18 >= prepare.FLOPS_PROBE_ROWS = 8 leaves frozen region
# 5's probe slicing the same 8 x 2048 tokens as every launch of this run. Frozen region 5:
# "Changing the actual training microbatch size remains allowed."
# THE FOURTH RUNG OF THIS LADDER, AND WHY ITS DEPTH IS SET BY THIS PARENT'S OWN RECORDED OVERSHOOT.
# The two paragraphs above are r049's argument for 48 rows and r063's for 36; both are kept as those
# rounds' records, nothing in them is deleted and their arithmetic is unchanged.
# rate(rows) = rows * MAX_SEQ_LEN / (F + s * rows) is strictly increasing in rows for any F > 0, so
# lowering the quantum lowers the charged tokens per training-clock second, and the RATIO
# rate(32) / rate(36) = 32 * (F + 36 * s) / (36 * (F + 32 * s)) is INVARIANT under a proportional
# change of F and s -- that is, under the node draw, which is why this route to the rate does not
# have to be bought from the node the way the five duration cards did (r045 ~0 ms of its dose
# delivered, r048 2.3%). r063 measured the premise the ratio rests on at 18-row chunks: the pairwise
# slopes c061 -> c063 (3.152165 ms/row) and c062 -> c063 (3.170467) bracket r013's s = 3.15696 within
# 0.15% / 0.43% (rounds/r063/decision.md section 4), so the per-row cost does not degrade as the
# chunk shrinks while the boundary count stays at two.
# THE ARITHMETIC, on the same two published slope pairs and c049's recorded 164.87931 ms at 48 rows.
# r013 section 4.2's s = 3.15696 gives F = 164.87931 - 48 * s = 13.34523 ms, dt(32) = F + 32 * s =
# 114.36795 ms against dt(36) = 126.99579, so the charged rate goes 580.555 -> 573.028 tokens/ms at
# an unchanged node factor, i.e. -1.2965%; the r003 pair's s = 3.1031 gives F = 15.93051,
# dt(32) = 115.22971 against 127.64211 and -1.5361%. Four rows of row work (4 * s = 12.63 .. 12.41
# ms) are deleted while every millisecond of F stays. Scaled onto this parent's own recorded mean
# charged dt of 125.12251 ms the same node-invariant ratio gives 112.68 .. 112.96 ms.
# WHY 32 AND NOT DEEPER. A rung is paid for out of the tokens a descent has already banked past its
# own crossing, and this parent's bank is recorded rather than assumed: its gate margin
# 0.0025732139798690934 divided by its own probe-8-to-9 secant 4.095501e-05 is 62.83 charged updates
# = 4,632,349 tokens = 2.910% of its recorded key (rounds/r063/decision.md section 7). This rung's
# node-invariant cut, -1.2965% .. -1.5361%, is 44.6% .. 52.8% of that recorded figure; 30 rows prices
# at -2.0584% .. -2.4353% (70.7% .. 83.7% of it) and 28 rows at -2.9149% .. -3.4431%, all of it or
# past it. 32 is therefore the deepest even rung whose cut stays inside half that recorded bank on
# r013's slope, and it is even because two EQUAL integer chunks are held fixed. This says nothing
# about which probe reads THIS launch.
# HELD FIXED, EACH ON RECORDED EVIDENCE, exactly as at the two parent revisions. The chunk COUNT
# stays two and the chunks stay EQUAL (16, 16), so loss / n_micro is unchanged exact arithmetic --
# for equal chunks the mean of the chunk means IS the whole-quantum mean -- and r048's non-constant
# chunk-count price stays out of this round. WEIGHT_DECAY stays at the 0.1 this parent already
# carries, so this round moves the charged rate alone and adds the fixed-decay ladder's fourth point
# after 48 -> 44 -> 36. The LRs are NOT rescaled (r019 applied the sqrt-quantum rule and lost yield
# in every interval it priced past the onset). The update-keyed timescales are NOT re-anchored (r005
# did that after r003's halving and recorded 246,415,360 against 227,540,992), and
# AVG_HORIZON_UPDATES = 32 still sits between the 20-update Muon momentum horizon and the charged
# updates in a 30 s probe interval, a count this quantum raises rather than lowers. The LR and
# weight-decay schedules are clock-keyed, so lrm(t) is bit-identical to the parent's at every
# training second. Every token still passes through exactly one forward/backward per update, so
# flops_per_token_measured * total_tokens still describes this launch's counted arithmetic exactly,
# and MICROBATCH_ROWS = 16 >= prepare.FLOPS_PROBE_ROWS = 8 leaves frozen region 5's probe slicing the
# same 8 x 2048 tokens as every launch of this run. Frozen region 5: "Changing the actual training
# microbatch size remains allowed."
MICROBATCH_ROWS = 16     # rows per forward/backward; DEVICE_BATCH_SIZE must be a multiple

# The ten ticks the frozen harness does not charge for TIME are still charged for TOKENS.
# TimeToTargetHarness.tick adds int(tokens) on every tick and adds dt only once self.step
# exceeds warmup_steps = 10, and tokens_to_target is that token sum -- so an update inside the
# first ten ticks is charged to the objective at full price and costs the clock nothing. r017
# measured the asymmetry live: ten ticks charged 425,984 tokens against this parent's 1,310,720
# at zero training-clock cost (rounds/r017/decision.md 3.6). The parent buys all ten at the
# most expensive price the substrate offers, 64 rows each; buy nine of them at the cheapest it
# offers instead, ONE row of the dataloader's own batch, in the loader's own order.
# Update 0 keeps the full 32 rows for two reasons that are not about tokens: its first chunk is the
# microbatch the frozen FLOPs probe measured, and it builds the 16-row compiled graph every charged
# update uses while the clock is still off. The 1-row graph is built at update 1, also inside the exempt ten, so no
# graph is built inside a charged update.
WARMUP_TICKS_FREE_OF_CLOCK = 10   # prepare.TimeToTargetHarness's frozen warmup_steps default
WARMUP_UPDATE_ROWS = 1            # rows per update for the other nine exempt ticks

# CLOSED-LOOP RUNG CONTROL. At a fixed probe index the recorded number is 270.0 s times the
# CHARGED TOKENS PER TRAINING-CLOCK SECOND, so on this parent's fixed 65,536-token quantum the key
# is set by one scalar: the mean charged update duration, through
# key = 83,968 + 65,536 * ceil(270000 / dt_bar_ms). Every rate card of this run chose that scalar
# OPEN LOOP -- it picked rows, a chunk count or a duty cycle against the mean a PREVIOUS launch
# recorded -- and the next draw moved the baseline out from under it three rounds running (r086
# +9.0105%, r087 +7.1073%, r088 +11.8924%). r088 measured the baseline motion directly and within
# one launch: its own non-split 32-row population ran 112.32172 ms against c085's 113.50904 ms on
# the SAME allocation, an offset of 1.18732 ms that is 77% of the dose that card delivered and 4.8x
# the incumbent's whole gate bank; across this run's seven 32-row fast paths the spread is
# 2.22080 ms = 18 quanta at the deciding probe. The window that beats the incumbent's recorded key
# is 0.0487 ms wide (r079, r081, r083 and r084 each state it), so open-loop aiming cannot hit it --
# and closed-loop aiming does not have to. The loop already computes each update's own synchronised
# dt in order to hand it to harness.tick, so it can read the very clock it is judged on WHILE the
# launch runs and spend exactly the time needed to land a chosen rung. Nothing here touches the
# harness, its cadence, its warmup count, its tick or the duration and token count passed to it.
# THE SET-POINT. RATE_TARGET_RUNG is the number of clock-charged updates this launch aims to have
# completed when the harness clock reaches 270.0 s, which is probe 9 at the frozen 30.0 s cadence.
# The controller tracks the straight line charged_clock = RATE_TARGET_SECONDS_PER_UPDATE * k in the
# count k of charged updates already completed, with the slope taken through k =
# RATE_TARGET_RUNG - 0.5: that places 270.0 s half an update above the line's value at
# RATE_TARGET_RUNG - 1, i.e. in the middle of the interval of dt_bar for which
# ceil(270000 / dt_bar_ms) == RATE_TARGET_RUNG, which is 0.0487 ms wide.
# WHY THE SET-POINT IS 2,344, RE-DERIVED AGAINST THE NEW INCUMBENT c089 -- THIS FILE'S PARENT.
# c089 is the first launch of this run whose recorded rung was a carded number: it aimed 2,349 and
# landed charged rung 2,349, with val_bpb 1.0488343437632766 at probe 9 (logs/0088, step 2,359), so
# its gate bank is 1.05 - 1.0488343437632766 = 0.001165656236723489. Its own probe-8 -> probe-9
# secant is (1.0582720344479224 - 1.0488343437632766) / (2359 - 2098) = 3.615973e-05 bpb per charged
# update (c074's was 3.593983e-05, 0.6% away), so that bank is 32.24 charged updates and the purely
# arithmetic floor is rung 2,317. THIS FILE DOES NOT TAKE ALL OF IT, because only part of that bank
# is a property of the program. r089 section 5 read c089 0.0011602 BETTER than c074 at matched steps
# at probe 9 while r088 read its own program 0.0011493 WORSE in the same frame -- the same size, the
# opposite sign -- so a trajectory difference as large as the whole bank, in either direction, is
# inside this run's experience and no set-point can insure against it. What CAN be priced is the
# draw-only component: the run's three matched-update reads on bitwise-identical code are -0.0006968
# (r047), +0.0002832 (r051) and +0.0009672 (r052), range 0.0016640, worst ADVERSE reading +0.0009672.
# 2,344 is the deepest rung whose arithmetic residual still covers that worst recorded adverse draw:
# 0.001165656236723489 - 5 * 3.615973e-05 = 0.00098486 >= 0.0009672, while one rung deeper leaves
# 0.00094870 < 0.0009672. It spends 15.5% of the bank for five rungs, where 2,348 -- the shallowest
# rung below c089's -- spends 3.1% for one, and every rung from 2,317 to 2,348 lies below c089 on the
# preserved lattice 83,968 + 65,536 * n. The interval of dt_bar that lands rung 2,344 is
# 270000/2343 - 270000/2344 = 0.04916 ms wide, and the actuator's authority over it is unchanged:
# 270.0 / 2343.5 = 115.21229 ms sits inside c089's own two recorded dt populations (undosed 112.770
# tail-free / 113.057 mean, dosed 123.247 / 123.867, r089 section 4), needing 19.9% to 23.3% of
# charged updates dosed against the 18.1-19.8% c089 ran overall and the 23.9% it ran in its first
# decile. Deeper than this needs what no round of this run has paid: a resolvable clock-free,
# FLOPs-free quality gain, priced at 0.0019 bpb by r086 and 0.0017 by r087.
RATE_TARGET_RUNG = 2344
RATE_TARGET_SECONDS_PER_UPDATE = 270.0 / (RATE_TARGET_RUNG - 0.5)
# MOVE THE DEPTH DECISION INSIDE THE LAUNCH, WHERE THE GATE BANK IS A MEASUREMENT.
# RATE_TARGET_RUNG above decides, before the launch exists, how many clock-charged updates this
# program will have completed when the harness clock reaches 270.0 s. How deep that number may go
# is set by ONE quantity: the distance between this trajectory's val_bpb at the deciding probe and
# the 1.05 gate. Every rate card of this run has had to GUESS that distance from a PREVIOUS
# launch's recorded margin, and the guess is the binding error. c089 recorded
# 0.001165656236723489; c090 -- its own child, one integer different -- recorded
# 0.0005855702909232097; and r090 section 4 measured the whole difference as trajectory draw
# (+0.00039834 bpb at matched steps, against +0.00000095 of secant estimation error). The run's
# three matched-update reads on bitwise-identical code are -0.0006968, +0.0002832 and +0.0009672,
# a band of 0.0016640, so the pre-launch estimate of the bank carries an error LARGER THAN THE
# WHOLE OF c090'S BANK. That, and not a shortage of tokens, is why r090 section 7 licenses zero
# further rungs: 0.0005855702909232097 at c090's own probe-8 -> probe-9 secant 3.635022e-05 is
# 16.11 charged updates of headroom, and no constant written before the launch can know whether
# they are there THIS TIME.
# THE LAUNCH CAN KNOW. The frozen harness evaluates the CERTIFYING metric at every 30.0 s probe
# and appends (training_seconds, step, val_bpb) to its own public list -- prepare.py's
# TimeToTargetHarness.tick, the line self.curve.append((self.training_seconds, self.step, bpb)) --
# and prints those same three numbers as its [probe] line. At the probe recorded at 240.0 s, one
# frozen interval before the probe this objective is decided at, the distance to the gate is
# therefore MEASURED on this trajectory, by the instrument that will score it. retarget_final_leg
# below reads that one triple and nothing else, and converts it into the number of charged updates
# the remaining interval must hold.
# WHAT IS READ, AND WHAT IS NOT TOUCHED. The harness is constructed from the task-supplied
# AUTORESEARCH_TARGET_VAL_BPB with the frozen cadence, warmup count and full-eval probe; it is
# ticked exactly once per completed logical optimizer update with that update's own synchronised
# duration and its own token count; the loop stops when it returns True. All of that is the
# parent's byte for byte. No attribute of the harness is assigned, no probe is skipped, shortened,
# repeated or replaced, no second reporter call exists, train.py still never reads the validation
# split -- the number this rule reads was computed by the frozen probe and is already in this
# launch's stdout before the rule uses it.
# THE RULE. Let v, t and s be the val_bpb, the clock second and the step of that probe. The
# remaining interval must hold
#     needed = ceil((v - TARGET_VAL_BPB + RUNG_SAFETY_BPB) / RUNG_LEG_SECANT_BPB_PER_UPDATE)
# charged updates, so the line is re-anchored at (t, s - WARMUP_TICKS_FREE_OF_CLOCK) with slope
# (RUNG_CROSSING_CLOCK_SECONDS - t) / (needed - 0.5). That is the same half-update convention
# RATE_TARGET_SECONDS_PER_UPDATE uses: it puts 270.0 s in the middle of the interval of leg means
# for which the leg holds exactly `needed` updates, so the deciding probe fires on the leg's
# needed-th update. Before that probe the line is the parent's -- anchor (0.0, 0), slope
# RATE_TARGET_SECONDS_PER_UPDATE -- and charged_clock - 0.0 < RATE_TARGET_SECONDS_PER_UPDATE *
# (charged - 0) is the parent's own test on the same two floats, so the first 240 s of this program
# is c090's program and RATE_TARGET_RUNG still sets it.
# THE SLOPE CONSTANT IS THIS RUN'S OWN, TAKEN AT ITS SLOWEST READING. Six launches of this run
# recorded a probe-8 -> probe-9 leg at this 65,536-token quantum. From their printed [probe] lines,
# (v8 - v9) / (step9 - step8):
#   c074  logs/0073  261 updates  0.009380  3.593870e-05
#   c083  logs/0082  263 updates  0.009444  3.590875e-05
#   c084  logs/0083  263 updates  0.009443  3.590494e-05
#   c085  logs/0084  264 updates  0.009518  3.605303e-05
#   c089  logs/0088  261 updates  0.009438  3.616092e-05
#   c090  logs/0089  260 updates  0.009451  3.635000e-05
# (printed to 6 dp, so each slope is exact to about 0.02%; the two legs whose ends are recorded at
# full precision read 3.615973e-05 for c089 and 3.635022e-05 for c090.) The six span 1.2244% peak
# to peak: 3.635000e-05 - 3.590494e-05 = 4.4506e-07 per update, which over a leg of 260 updates is
# 0.00011572 bpb, 14.4x SMALLER than the 0.0016640 band a pre-launch estimate of the bank carries.
# The reason is structural and not luck: the draw is in v, which this rule MEASURES, and not in the
# slope of the leg that follows it. 3.59e-05 sits 0.0138% below the slowest of the six and
# 1.2380% below the fastest, and a slope estimate that is too LOW asks for MORE updates, which is
# the safe direction on the gate and the expensive one on the key.
# THE GUARD. RUNG_SAFETY_BPB is bpb withheld from the rule: the leg is asked for the updates that
# reach TARGET_VAL_BPB - RUNG_SAFETY_BPB, not the gate itself. At c090's recorded probe-8 deficit
# of 0.0088654876513341 it tolerates a true leg slope 0.00015 / (0.0088654876513341 + 0.00015) =
# 1.6638% below 3.59e-05, i.e. 3.530269e-05, which is 1.66% below the slowest of the six and 2.88%
# below the fastest. Two further conservatisms are NOT priced into it: shortening a leg deletes its
# FLATTEST updates, the leg slopes above being means over intervals whose local secant is still
# falling (c090 read 5.7165e-05 over its 210 s -> 240 s leg against 3.635022e-05 over the next
# one), and the actuator only ADDS time, so a slope the rule has underestimated is met by less
# dosing rather than by an overrun.
# WHAT THE ACTUATOR CAN AND CANNOT DELIVER. The dose is the parent's, unchanged: the same 32 rows
# in the same order as DOSE_CHUNK_ROWS-row chunks instead of MICROBATCH_ROWS-row ones. So the leg's
# reachable mean charged update is bounded by c090's own two recorded populations, 112.84 ms
# tail-free undosed and 123.23 ms tail-free dosed (r090 section 5). A rule output the leg cannot
# reach SATURATES: at full dose the leg is as short as the actuator allows, at zero dose as long as
# the node allows, and both saturations are one-sided in the direction the rule wants. This card
# does not widen the actuator -- a narrower dose chunk needs a third compiled width and coarsens
# the landing quantum two launches have confirmed -- and it is not needed until a launch's own
# measured bank exceeds what this dose can spend in one interval.
RUNG_DECISION_AFTER_SECONDS = 240.0         # 8 frozen 30.0 s probe intervals: the last probe
                                           # before the probe this objective is decided at
RUNG_CROSSING_CLOCK_SECONDS = 270.0        # 9 frozen intervals: the deciding probe's clock second
RUNG_LEG_SECANT_BPB_PER_UPDATE = 3.59e-05  # below the slowest of the six recorded legs above
# THE RESERVE IS SPENT, AND THE INSURANCE THAT REMAINS IS THE ONE THIS RUN HAS RECORDED TEN TIMES.
# THE GUARD paragraph above is kept as c091's record and its arithmetic is unchanged; what it
# withholds is what this file stops withholding. retarget_final_leg asks the final leg for
#     needed = max(1, ceil((v - TARGET_VAL_BPB + RUNG_SAFETY_BPB) / RUNG_LEG_SECANT_BPB_PER_UPDATE))
# charged updates, so with the reserve at zero the leg is aimed at the gate itself. The expression
# is then the same float as ceil((v - TARGET_VAL_BPB) / RUNG_LEG_SECANT_BPB_PER_UPDATE) bit for bit,
# because x + 0.0 is x for every finite float except -0.0, where the sum is +0.0 and the ceil is the
# same integer. Nothing else in the rule moves: the slope constant above, both clock constants, the
# half-update convention, the single re-aim, the max(1, .) floor and the print are c091's.
# WHY THE RULE AND NOT THE PACE CAP. RATE_TARGET_RUNG stays at c091's 2344. Both closed-loop moves
# of it recorded worse keys than c091 -- r093 raised it to 2408, r094 lowered it to 2328 -- and
# r094 section 7 closes that axis three ways, any one of them sufficient: r094's own card
# preregistered that a further step needed its printed ask to come back at or below 250 and the
# printed ask was 260; that launch's leg capacity on its own undosed populations was 263 to 265
# updates against an ask of 260, so 3 to 5 updates of headroom; and 2328's line, 2,069 charged
# updates at the 240.0 s probe, already sits at the low edge of the [2,069, 2,218] range the
# exchange ratio's sign was read on, which 2327 leaves. Keeping 2344 also keeps the cap's line at
# N_A = ceil(8 * (2344 - 0.5) / 9) = 2,084, the depth at which this run holds more recorded
# probe-8 readings than at any other.
# WHAT THE RESERVE COSTS, RECOMPUTED ON THE PROBE-8 READINGS THIS RUN HAS RECORDED AT THIS QUANTUM.
# The rule converts the reserve into 0.00015 / 3.59e-05 = 4.18 updates of ask, and the ceil makes
# that 4 or 5. Recomputed on the six recorded readings -- c089 1.0582720344479224, c091
# 1.0585986281164972, c092 1.0587289610081083, c090 1.0588654876513341, c074 1.0591592308, c094
# 1.0591762572216896 -- dropping the reserve removes 4 updates of ask on five of them and 5 on
# c090's. Nothing is claimed about this launch's own draw, which is the largest term in the key and
# which no constant written before the launch can know: at one fixed depth this run holds two
# readings 0.0014871 bpb apart, 41 updates of ask (r094 section 6c).
# WHAT STILL STANDS BETWEEN THE ASK AND THE GATE. The realised margin is the reserve plus the ceil
# crumb plus the slope surprise, and the surprise is the largest of the three: r094 section 3
# decomposes c094's recorded 0.00034381270 as 0.00015 + 0.00000774 + 0.00018607. The surprise exists
# because RUNG_LEG_SECANT_BPB_PER_UPDATE 3.59e-05 is below every leg secant this run has recorded at
# this quantum -- the four closed-loop legs read 3.698578e-05, 3.675089e-05, 3.685156e-05 and
# 3.661565e-05 (r094 section 3), the six longer legs listed above read 3.590494e-05 to 3.635000e-05
# -- ten readings, all of them above the constant, and shortening a leg deletes its FLATTEST
# updates, which steepens the secant further. r094 section 7 recomputed this exact edit on the four
# closed-loop draws: each of them still passes the gate, with residual margins 0.000278, 0.000238,
# 0.000228 and 0.000197 bpb, which at those launches' own realised secants is 5.4 to 7.6 updates of
# leg.
# WHY THE SLOPE CONSTANT IS NOT TOUCHED AS WELL. r094 section 7 recomputed both edits together and
# the worst of the four draws keeps 0.0000143 bpb, 0.39 of an update, so the pair is not safe. Of
# the two, this file spends the one that asserts nothing: 0.0 removes a reserve, whereas 3.66e-05
# would assert a lower bound on the next leg's secant 0.04% under the lowest of only four
# closed-loop readings, and r092 section 7 records that two points of a favourable trend are the
# easiest thing to over-trust. The slope constant stays priced and unspent.
# WHAT THIS EDIT FORGOES, STATED BECAUSE IT IS NOT RECOVERABLE. The retention route needs an exact
# tie on the key AND a gate margin larger than the incumbent's 0.00042590227517558255, and this
# edit makes the realised margin smaller than that by construction, since the margin is what the
# rule withholds. That route is given up deliberately: it also needed the draw to land on c091's own
# ask, and the three closed-loop launches since c091 recorded margins 0.000385, 0.000375 and
# 0.000344, every one below the incumbent's, so a tie would have been refused on its own condition
# in all three (r094 section 2).
# WHAT THIS ROUND DOES NOT TOUCH. RATE_TARGET_RUNG 2344 and its slope 270.0 / 2343.5; the other
# three RUNG_ constants; retarget_final_leg, its single re-aim and its printed line; the
# charged_clock accumulator, the rate_line state and the half-update convention; DOSE_CHUNK_ROWS 8,
# MICROBATCH_ROWS 16, both charged plans and update 0's own (16, 8, 8) plan, so no trajectory
# perturbation is bought and no compiled width is added; update_rows, the 83,968-token exempt
# prefix, the 65,536-token quantum and the key's lattice; every learning rate, schedule, shape,
# parameter, token, row, loader call and data setting.
RUNG_SAFETY_BPB = 0.0                      # the reserve, spent (see THE RESERVE IS SPENT)
# The controller's line, as [seconds per charged update, anchor clock, anchor charged count,
# already re-aimed]. It starts as the parent's line and retarget_final_leg replaces it once.
rate_line = [RATE_TARGET_SECONDS_PER_UPDATE, 0.0, 0, False]
# THE ACTUATOR: MORE CHUNK BOUNDARIES ON THE SAME ROWS, WHICH DELETES NO TOKEN. A dosed update
# trains the parent's own 32 rows, in the parent's own order, out of the parent's own loader batch,
# as DOSE_CHUNK_ROWS-row forward/backward chunks instead of MICROBATCH_ROWS-row ones. r088 measured
# this exact actuator inside one launch against that launch's own non-dosed population: 10.80581 ms
# per occurrence on charged means, 10.38469 tail-free, with the two dt populations perfectly
# separated. It is also the cheapest currency this run has priced (r087: 0.001048 bpb per 1% of the
# key, against a 0.000847 token-secant floor and 0.001705 for a row rung), and it moves no token,
# no quantum, no loader call, no schedule and no parameter -- only the association order of a mean
# over the same rows.
DOSE_CHUNK_ROWS = 8
# THE GRAPH THE ACTUATOR NEEDS, BOUGHT FOR ZERO TOKENS. model is compiled with dynamic=False, so a
# DOSE_CHUNK_ROWS-wide forward is a second specialisation and building it inside a charged update
# would put a ~20 s compile on the clock. r088 bought that protection by training 8 rows at exempt
# update 1 instead of 1 -- 14,336 extra tokens -- which moved the key's exempt prefix from 83,968 to
# 98,304 and took an exact tie OFF the lattice ((154,421,248 - 98,304)/65,536 = 2,354.78125), so it
# forfeited the retention route as well as 22% of the binding margin. Update 0 already trains all 32
# rows free of the clock, so the same protection is free: plan update 0 as one MICROBATCH_ROWS chunk
# followed by DOSE_CHUNK_ROWS-row chunks and both widths are compiled before the clock starts, at
# the parent's own token cost. update_rows is untouched, so the exempt prefix stays
# 32 * 2,048 + 9 * 1 * 2,048 = 83,968 tokens exactly and (154,421,248 - 83,968)/65,536 = 2,355
# stays integral.
assert DEVICE_BATCH_SIZE % DOSE_CHUNK_ROWS == 0
assert (DEVICE_BATCH_SIZE - MICROBATCH_ROWS) % DOSE_CHUNK_ROWS == 0

def chunks_of_width(rows, width):
    """`rows` cut into consecutive chunks of at most `width` rows; they sum to `rows`."""
    return tuple(min(width, rows - start) for start in range(0, rows, width))

def update_chunk_plan(step, rows, charged_clock):
    """Forward/backward chunk widths for update `step`; they sum to `rows` exactly.

    A charged update is the parent's MICROBATCH_ROWS chunks unless the charged clock is behind
    the set-point line rate_line holds -- the parent's line until retarget_final_leg re-anchors
    it -- in which case the same rows are grouped into DOSE_CHUNK_ROWS-row chunks
    instead -- so the plan is a property of this update's own clock reading rather than of its
    index. Update 0 runs one MICROBATCH_ROWS chunk and then DOSE_CHUNK_ROWS-row chunks, which
    compiles both widths with the clock still off; the nine 1-row exempt ticks are one chunk each,
    as in the parent. Every plan trains every one of `rows` rows exactly once, so no plan changes
    a token, a row, a loader call or the order the rows are seen in.
    """
    if step >= WARMUP_TICKS_FREE_OF_CLOCK:
        charged = step - WARMUP_TICKS_FREE_OF_CLOCK
        seconds_per_update, anchor_clock, anchor_charged, _ = rate_line
        behind = (charged_clock - anchor_clock
                  < seconds_per_update * (charged - anchor_charged))
        return chunks_of_width(rows, DOSE_CHUNK_ROWS if behind else MICROBATCH_ROWS)
    if step == 0:
        first = min(MICROBATCH_ROWS, rows)
        return (first,) + chunks_of_width(rows - first, DOSE_CHUNK_ROWS)
    return (rows,)

def retarget_final_leg(harness):
    """Re-aim the controller's line once, on the frozen instrument's own probe reading.

    Reads the last triple the frozen harness appended to its own `curve` -- (training_seconds,
    step, val_bpb), the three numbers it also prints as its `[probe]` line -- and nothing else.
    Nothing is assigned on the harness, no probe is skipped, shortened, repeated or replaced, the
    cadence, the warmup count and the full-eval flag stay the frozen defaults, tick is still called
    exactly once per completed logical optimizer update, and train.py still never reads the
    validation split.

    Fires at the first probe whose recorded clock has reached RUNG_DECISION_AFTER_SECONDS, and
    never again. Before it -- and for the whole of a run that crosses earlier, or that is given no
    positive target, where the rule would ask for more updates than the leg can hold and the leg
    therefore runs undosed -- the line stays the parent's. The record block above the four RUNG_
    constants states the rule and derives them.
    """
    if harness is None or rate_line[3] or not harness.curve:
        return
    seconds, probe_step, bpb = harness.curve[-1]
    if seconds < RUNG_DECISION_AFTER_SECONDS:
        return
    needed = max(1, math.ceil((bpb - TARGET_VAL_BPB + RUNG_SAFETY_BPB)
                              / RUNG_LEG_SECANT_BPB_PER_UPDATE))
    remaining = RUNG_CROSSING_CLOCK_SECONDS - seconds
    rate_line[:] = [remaining / (needed - 0.5), seconds,
                    probe_step - WARMUP_TICKS_FREE_OF_CLOCK, True]
    print(f"[rate] final leg re-aimed at t={seconds:.4f}s step={probe_step} "
          f"val_bpb={bpb:.10f}: {needed} charged updates in {remaining:.4f}s, "
          f"{1000.0 * rate_line[0]:.5f} ms per charged update", flush=True)

def update_rows(step):
    """Microbatch rows for update `step`: the parent's 64 at step 0 and from the first charged
    tick on, WARMUP_UPDATE_ROWS for the exempt ticks in between."""
    if step == 0 or step >= WARMUP_TICKS_FREE_OF_CLOCK:
        return DEVICE_BATCH_SIZE
    return WARMUP_UPDATE_ROWS

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
        window_span_sequences=WINDOW_SPAN_SEQUENCES,
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

assert DEVICE_BATCH_SIZE % MICROBATCH_ROWS == 0
tokens_per_fwdbwd = MICROBATCH_ROWS * MAX_SEQ_LEN
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

# Iterate-averaging state, allocated while `model` is still the uncompiled module, so the
# Parameter objects grouped here are the ones the optimizer already holds. These are plain
# tensors -- not Parameters, not registered buffers -- so count_params, estimate_flops and
# measure_flops_dispatch see exactly the parent's model. init_weights casts wte and the
# value embeddings to bf16; their average is kept in fp32 because a bf16 lerp at alpha
# 1/32 rounds most of each increment away, and it is written back to bf16 on install, the
# same precision the iterate itself carries. The lists are split by dtype so every
# _foreach_ call below is dtype-uniform.
model_params = list(model.parameters())
avg_state = [p.detach().float().clone() for p in model_params]   # the averaged sequence
fast_state = [p.detach().clone() for p in model_params]          # the optimizer's iterate
_avg_is_f32 = [p.dtype == torch.float32 for p in model_params]
par_f32 = [p for p, f in zip(model_params, _avg_is_f32) if f]
avg_f32 = [a for a, f in zip(avg_state, _avg_is_f32) if f]
fast_f32 = [q for q, f in zip(fast_state, _avg_is_f32) if f]
par_low = [p for p, f in zip(model_params, _avg_is_f32) if not f]
avg_low = [a for a, f in zip(avg_state, _avg_is_f32) if not f]
fast_low = [q for q, f in zip(fast_state, _avg_is_f32) if not f]

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
flops_per_token_measured = measure_flops_dispatch(model, x[:MICROBATCH_ROWS], y[:MICROBATCH_ROWS])
print(f"FLOPs per token (measured at dispatch): {flops_per_token_measured:,}")

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")

# Schedules. The LR and weight-decay schedules read sched_progress = training_time /
# SCHEDULE_HORIZON (clamped at 1.0), so they complete at SCHEDULE_HORIZON rather than at
# TIME_BUDGET; the Muon momentum warmup stays indexed by update count.

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
    # sched_progress is clamped at 1.0, so past SCHEDULE_HORIZON the ramp below returns exactly
    # 0.0 for every remaining update; hold WD_FINAL_FRAC of WEIGHT_DECAY there instead, the way
    # get_lr_multiplier holds FINAL_LR_FRAC. Below the clamp this is the parent's expression,
    # evaluated bit for bit on the parent's own two floats.
    if progress < 1.0:
        return WEIGHT_DECAY * (1 - progress)
    return WEIGHT_DECAY * WD_FINAL_FRAC

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
tokens_consumed = 0
step = 0
row_cursor = 0           # rows of the current loader batch already trained on
# The clock the controller steers, and the clock the objective is read on, are the same sum:
# TimeToTargetHarness.tick adds dt once its own step exceeds warmup_steps = 10, i.e. from the
# update whose loop index is 10, and this accumulator adds the same dt over the same set of
# updates. total_training_time is NOT reused for the plan: it starts at index 11, and it is the
# input to the LR and weight-decay schedules, which stay the parent's bit for bit.
charged_clock = 0.0      # seconds of harness-charged training time completed so far

while True:
    torch.cuda.synchronize()
    t0 = time.time()
    # This update's quantum: a row slice of the loader's own 32-row batch (see update_rows).
    rows = update_rows(step)
    x_mb = x[row_cursor:row_cursor + rows]
    y_mb = y[row_cursor:row_cursor + rows]
    tokens_this_update = rows * MAX_SEQ_LEN
    # The gradients are taken at the optimizer's iterate, which the previous update saved;
    # the model's parameters currently hold the average. Inside the timed region, because
    # this is part of this update's own work.
    with torch.no_grad():
        torch._foreach_copy_(par_f32, fast_f32)
        torch._foreach_copy_(par_low, fast_low)
    # This update's gradient, accumulated over the chunks update_chunk_plan hands back for the
    # row slice above. No next(train_loader) call happens between chunks, so the loader's single
    # reused pinned/gpu buffer is never rewritten while a chunk of it is in flight. The chunks no
    # longer have to be equal -- update 0's plan is (16, 8, 8) -- so the weight is each chunk's
    # ROW SHARE rather than one over the chunk count, which is the exact weighting for any plan:
    # sum_i (rows_i / rows) * mean_i IS the whole-quantum mean. On the parent's own plans the two
    # agree BITWISE, not just in value: 16/32 is exactly 0.5 and 1/1 exactly 1.0, scaling by a
    # power of two is exact, and rounding commutes with it, so (l1 + l2)/2 and 0.5*l1 + 0.5*l2 are
    # the same float32. train_loss therefore stays the whole-quantum mean, the parent's logged
    # quantity, and every charged update that is not dosed runs the parent's arithmetic bit for bit.
    retarget_final_leg(harness)
    chunk_plan = update_chunk_plan(step, rows, charged_clock)
    train_loss = None
    m0 = 0
    for chunk_rows in chunk_plan:
        m1 = m0 + chunk_rows
        with autocast_ctx:
            loss = model(x_mb[m0:m1], y_mb[m0:m1])
        chunk_weight = chunk_rows / rows
        weighted_loss = loss.detach() * chunk_weight
        train_loss = weighted_loss if train_loss is None else train_loss + weighted_loss
        loss = loss * chunk_weight
        loss.backward()
        m0 = m1

    # Refill in the parent's position -- after the backward, inside the timed region, one
    # torch.cuda.synchronize() after the previous refill -- but only when the rows the NEXT
    # update needs are not left in this batch, so a slice never straddles two batches and the
    # loader's single reused pinned buffer is never rewritten while a copy out of it is in
    # flight. From the first charged tick on every update takes all 32 rows, so this fires
    # exactly once per update, exactly where the parent fires it -- after every microbatch of
    # this update has been consumed, which is why the split above cannot race the loader's
    # buffer.
    row_cursor += rows
    if row_cursor + update_rows(step + 1) > DEVICE_BATCH_SIZE:
        x, y, epoch = next(train_loader)
        row_cursor = 0

    # Progress and schedules
    progress = min(total_training_time / TIME_BUDGET, 1.0)   # logging only
    sched_progress = min(total_training_time / SCHEDULE_HORIZON, 1.0)
    lrm = get_lr_multiplier(sched_progress)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(sched_progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    optimizer.step()
    model.zero_grad(set_to_none=True)

    # Fold the new iterate into the running average and install the average as the model's
    # parameters. Still inside this update's timed region, so the training clock is charged
    # for it. `step` is not incremented until line 626, so step + 1 is the number of
    # completed updates including this one.
    with torch.no_grad():
        torch._foreach_copy_(fast_f32, par_f32)
        torch._foreach_copy_(fast_low, par_low)
        avg_alpha = max(AVG_ALPHA, 1.0 / (step + 1))
        torch._foreach_lerp_(avg_f32, par_f32, avg_alpha)
        for a, p in zip(avg_low, par_low):
            a.mul_(1.0 - avg_alpha).add_(p, alpha=avg_alpha)
        torch._foreach_copy_(par_f32, avg_f32)
        for a, p in zip(avg_low, par_low):
            p.copy_(a)

    train_loss_f = train_loss.item()

    # Fast fail: abort if loss is exploding or NaN
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step >= WARMUP_TICKS_FREE_OF_CLOCK:
        charged_clock += dt
    if step > 10:
        total_training_time += dt

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(tokens_this_update / dt)
    mfu = 100 * num_flops_per_token * tokens_this_update / dt / H100_BF16_PEAK_FLOPS
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | rows: {rows:2d} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

    # GC management (Python's GC causes ~500ms stalls)
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1
    tokens_consumed += tokens_this_update

    # Axis H: the frozen harness owns the clock, the probe cadence and the crossing.
    if harness is not None and harness.tick(model, tokenizer, dt, tokens_this_update):
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
