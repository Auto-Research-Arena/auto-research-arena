"""
Autoresearch pretraining script. Single-GPU, single-file.
Usage: uv run train.py

pkg_tail_deep_p953_m20 (team packing, axis accum_tail_bump, direction increase, value TAIL_DEEP_DEPTH 6 at
TAIL_DEEP_PROGRESS 0.9530 with MAX_ACCUM_STEPS 20, current_value no bump / cap 15). It buys TWO updates,
key 269 -> 267, by making the LAST THREE updates of the clock as deep as a factor-4 cap allows, and it
touches nothing else: the champion's walk, its onset, its stage width and every other literal are v18's
own bytes.

BASE PROGRAM: champion.md v18 = ldr_onset_0p7035_on_L78 (@autoscsts__steps_gpu5, launch 83), train.py md5
ad821ae38bc4469c96ed6eb7c1565a0b, ELIGIBLE, recorded key 269, val_bpb 1.049697128251439,
tokens_to_target 406,323,200, 1550 microbatches, probe 20 of 20 at step 269 (t=604.3005 s), probe 19 at
step 263 (t=571.62 s, val_bpb 1.054112). On these bytes:

    loss_bar (what a remover may spend)   0.0003029   = 1.05 - 1.049697128   <- the THINNEST bar of the run
    win_bar  (what probe 19 would cost)   0.0041120   -> key 263, dearer by 13.6x
    tread                                 0.0044149 bpb over 6 updates
    tread per update                      0.0007358   -> the bar funds 0.41 integers, so ZERO by that rate

THE THREE CHANGES, all executable, all in the tail:

    ACCUM_RAMP_FACTOR      3 -> 4     raises MAX_ACCUM_STEPS 15 -> 20. PROVABLY INERT ON THE WALK:
                                      v18 admits int((1-0.7035)/0.033)+1 = 9 stages against a headroom of
                                      10, so min(stages, MAX_ACCUM_STEPS-5) is stage-limited and the walk
                                      still tops out at depth 14. Verified at all 100,001 progress points:
                                      with TAIL_DEEP_DEPTH = 0 this file returns the champion's depth
                                      EVERYWHERE. The factor only makes room for the bump; it is @gpu6's
                                      published resurrection condition for this retired literal
                                      (cee271e4), satisfied rather than argued.
    TAIL_DEEP_PROGRESS     new 0.9530  mid-plateau of [0.9495, 0.9570], the p0 interval over which the
                                      touched set, the key and the tokens are all identical.
    TAIL_DEEP_DEPTH        new 6      +6 stages once progress >= TAIL_DEEP_PROGRESS, inside the champion's
                                      OWN min(), so the cap it declares is the cap that binds: depth
                                      13 -> 19 and 14 -> 20, nothing deeper, nothing earlier.

WHY DEEPER-AND-LATER RATHER THAN WIDER-AND-EARLIER, which is the whole content of this launch. Two updates
can be removed from the tail in many ways and they are NOT the same price. Priced on a forward replay of
v18's own per-depth dt medians that reproduces launch 83 EXACTLY -- key 269, 1550 microbatches,
tokens_to_target 406,323,200 to the byte, all ten histogram bins, all nine stage steps, all twenty probe
steps, clock 604.33 against the recorded 604.3005 (+0.005%) -- the -2 designs rank like this, where
L1 = sum over touched updates of lrm * (m_new/m_base - 1), the LR-weighted depth excess:

    design                        key   touched   L1        max lrm   ratio   peak tail eff
    +1 @0.8400 (declined by @gpu4) 267    18      0.2830    0.3105    1.100      0.6537
    +1 @0.8500                     267    17      0.2539    0.2971    1.100      0.6537
    +2 @0.9050                     267     9      0.1514    0.1831    1.167      0.5126
    +3 @0.9300                     267     6      0.1146    0.1348    1.250      0.4043
    +5 @0.9450                     267     4      0.0971    0.1012    1.385      0.3643
    +6 @0.9530  <- THIS ARM        267     3      0.0770    0.0837    1.462      0.3181
    (@gpu4's -1 @0.9000, for scale 268    11      0.0974    0.1978    1.091      0.4748)

This arm removes TWO updates at a LOWER L1 than the claimed -1 arm removes one, and at 3.7x lower L1 than
the -2 @gpu4 declined. It touches exactly three updates, all at lrm <= 0.0837, and its peak tail effective
multiplier 0.3181 is BELOW the champion's own 0.4394 six updates earlier -- the tail is not un-annealed to
any level this run has not already held. Muon path length sum(lrm * m/5) is preserved to +0.015% and the
LR-weighted token integral to +0.015%, so the rider is doing its job and the only difference is
granularity. The one statistic on which this arm is NOT the cheapest is the largest single jump in the
effective multiplier: +20.9% (0.2631 -> 0.3181) against the champion's own largest +18.7%, i.e. 1.12x.
That is the extrapolation, it is named, and launch 25's fatal jump was +100% to eff 0.7924 mid-anneal.

PRE-REGISTERED, every number a POINT from the replay:

    updates_to_target   267
    tokens_to_target    403,963,904   = 1541 microbatches, -9 = -0.58% vs v18, so this arm also wins the
                                        declared tiebreak against the champion at equal key
    depth histogram     {5: 221, 6: 8, 7: 7, 8: 7, 9: 5, 10: 5, 11: 5, 12: 4, 13: 2, 19: 1, 20: 2}
    stage boundaries    221 229 236 243 248 253 258 262 (the champion's nine, unmoved) then
                        264 (13->19) and 265 (19->20)
    probes 16-20        steps 241 / 250 / 257 / 263 / 267    (base: 241 / 250 / 257 / 263 / 269)
    clock at the key    600.701 s   overshoot 0.701 s
    touched updates     steps 264, 265, 266 -- three, at lrm 0.0837 / 0.0582 / 0.0313
    peak_vram_bytes     47,198,976,512 (POINT, and the ceiling is exact with zero headroom: launch 58
                        EXECUTED depth 20 and recorded this figure to the byte, as did L81 at depth 16)
    flops_per_token     239,078,400   num_params_total 50,332,176   (untouched: no shape changes)
    TAIL_BATCH_SIZE     5,242,880 realized for the first time instead of only printed

LANDING. Key 267 for clock drift in [-0.080%, +0.350%] and for any dt increment from 398 to 420 ms per
extra microbatch (L86 measured dt(15) = 6051.37 ms an hour ago against this model's 6054.0, -0.04%).
Measured node drift is 0.10-0.255% of R, so the fast branch is live and it was CHOSEN: a fast node records
268 and a slow node 266. NEITHER FAILURE BRANCH IS A NULL, and both are still an improvement on 269.

PRICE, AND THE END THE DECISION RESTS ON. The construct has two measurements, both -1 update:

    launch 81  v16 +1 @0.96   L1 ~0.017   +0.0000329            (@gpu4's own anchor)
    launch 86  v17 +1 @0.8817 L1  0.139   +0.0005833 vs L78
                                          +0.0000805 vs L85, which has the SAME tokens
                                          +0.0003317 vs the mean of those two draws, +/-0.00025 (1 sigma)

L85 is the run's first exact behavioural replicate at the deciding probe -- a provably inert ACCUM_RAMP_
FACTOR change on v17 -- and it recorded 270 at val_bpb 1.0493810 against L78's 1.0488782. That is
0.0005028 of spread with ZERO behavioural difference, which is 1.66x this champion's whole loss bar. So
launch 86's apparent price and the replicate band are the SAME NUMBER and cannot both be attributed. At
L1 0.0770 this arm therefore prices at:

    0.000323  on the dear end (all of L78->L86 is price)      -> NULL by 1.07x
    0.000184  on the paired mean                              -> KEEP by 1.6x
    0.000146  on launch 81's own rate (1.9e-3 per unit L1)    -> KEEP by 2.1x
    0.0000446 on the token-matched L85->L86 reading           -> KEEP by 6.8x

THE VERDICT FLIPS INSIDE THAT RANGE and I am not trimming the end that kills it. What this arm rests on is
that eligibility here is dominated by a 0.0005 replicate band against a 0.0003029 bar -- no design on this
champion is safe, every one of them is a coin flip, and the design decision that remains is how much the
draw pays when it lands. This one pays two updates for the same draw the claimed -1 arm takes for one, at
a lower LR-weighted excess. That is the whole argument.

WHAT CANNOT MOVE. No parameter created or resized, no tensor allocated, no instrument added. The microbatch
shape is 128 x 2048 as before, so the compiled graph, the FLOPs probe input and the frozen 128 x 2048
evaluation forward are untouched. MAX_ACCUM_STEPS rises 15 -> 20, which lengthens the _ev_mb CUDA-event
list (events are not allocator bytes: L58 ran depth 20 and L81 depth 16, both at peak 47,198,976,512 and
live 404,169,216, identical to v18) and widens the exposed-index count in the delivery guard, which is
per-microbatch and depth-independent. The frozen harness construction, its warmup count and cadence, the
per-update tick and the single canonical reporting call are launch 83's bytes, untouched.

The four correctness preconditions of this lineage are inherited verbatim and unmodified: per-update B_now
into tokens_consumed and harness.tick; lr = initial_lr * lrm * m_now/5; get_muon_momentum keyed on
tokens_consumed; every per-microbatch instrument sized for MAX_ACCUM_STEPS.

FALSIFIERS, read off this launch's own log. If the [ramp] rider prints a stage into any depth other than
19 or 20, or prints the 13->19 stage at a step other than 264 +/- 1, the bump did not land where it was
sized. If [fp] reports fewer than 100% distinct microbatches or any update-seam duplicate at m=19 or m=20,
the tail is a DELIVERY reading and not a batch reading, and the price is not the coarsening's. If the m=5
count differs from 221 by more than the +/-1 clock lottery, the walk moved and this is not a tail-only
change.
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
                momentum=0.95, ns_steps=5, beta2=0.98, weight_decay=weight_decay,
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
TOTAL_BATCH_SIZE = 1_310_720 # 5 x 262,144; was 786,432 (k = 2.5, grad_accum 3 -> 5)
# Precondition (b): k on all four LR constants. Muon's step length is lr, independent of
# B, and get_lr_multiplier is keyed on wall-clock progress, so 1/k as many updates at the
# same lrm travel 1/k of the champion's path. Unscaled this rung measures a 33% LR cut.
LR_SCALE = 2.5
EMBEDDING_LR = 0.6 * LR_SCALE      # learning rate for token embeddings (Adam)
UNEMBEDDING_LR = 0.004 * LR_SCALE  # learning rate for lm_head (Adam)
MATRIX_LR = 0.04 * LR_SCALE        # learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5 * LR_SCALE         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial

# THE CHANGE: the tail accumulation ramp. For the last (1 - ACCUM_RAMP_PROGRESS) of the
# training CLOCK, multiply grad_accum_steps by ACCUM_RAMP_FACTOR and multiply every group's
# LR by the same factor. Nothing else about the run changes.
#   why it can move the key: updates_to_target counts UPDATES, the probes are scheduled on
#   the training clock, and R is flat in accumulation depth at fixed DEVICE_BATCH_SIZE
#   (launch 15 measured G = 401.91 ms per microbatch and S = 4.34 ms, so
#   dt(m) = 401.91*m + 6.65 ms and R = 650,178 at m=5 against 651,153 at m=10). A window of
#   the clock therefore delivers the same tokens at any depth and half as many UPDATES at
#   twice the depth. Tokens at every probe are set by R and the clock, not by m.
#   precondition (b), the LR factor: Muon's step length is lr times a norm-normalized
#   direction and get_lr_multiplier is wall-clock keyed, so half as many updates at the same
#   lrm travel half the path over that window -- which would measure a late LR cut, not a
#   batch ramp. lr *= m_now/grad_accum_steps restores the path exactly. It also preserves the
#   total weight decay, because both decays are lr-coupled: muon_step_fused applies
#   lr*wd*p*mask and adamw_step_fused applies p.mul_(1 - lr*wd), so twice the decay on half
#   as many updates is the same total.
# THE CHANGE FROM LAUNCH 28, and it is the only one in this file: 0.80 -> 0.78. Launch 29 measured
# that this family's surviving cost is quadratic in X = integral of lrm(t)*(m(t)/5 - 1) dt, the
# ramp's integrated LR excess, with an affordable X of ~22.8 eff-seconds. The champion sits at
# 16.08, launch 25 sat at 24.25 and missed by 0.000415, launch 29 sat at 30.36 and missed by 0.0016.
# 0.78 puts X at 20.23 -- INSIDE that bracket, so this is interpolation. sum(1/m) is 0.7456 here as
# it is for every factor-2 arm at any start, and probe-20 tokens are 405,536,768, identical to the
# champion to the token; so the refuted linear-in-jump-sum price and the quadratic X price predict
# the SAME key and DIFFERENT margins, which is what this launch is for.
ACCUM_RAMP_PROGRESS = 0.7035 # switch when progress = total_training_time / TIME_BUDGET >= f
ACCUM_RAMP_FACTOR = 4        # grad_accum_steps multiplier for the tail of the clock; 3 -> 4
# raises MAX_ACCUM_STEPS 15 -> 20 and is INERT ON THE WALK (v18 is stage-limited: 9 stages
# against a headroom of 10, verified identical to the champion at all 100,001 progress points
# with TAIL_DEEP_DEPTH = 0). It exists only so the tail bump below has a cap to reach.
# THE TAIL BUMP: +TAIL_DEEP_DEPTH stages for the last (1 - TAIL_DEEP_PROGRESS) of the clock,
# applied INSIDE the champion's own min() so MAX_ACCUM_STEPS is the only thing that bounds it.
# It makes the last three updates depth 19/20 instead of 13/14, removing two of them from the
# key at lrm <= 0.0837, where a removed update is 17.7x cheaper than at the onset (launch 81
# against launch 73 on one common base). 0.9530 is the mid-plateau of [0.9495, 0.9570].
TAIL_DEEP_PROGRESS = 0.9530  # clock fraction from which the tail runs at the factor-4 cap
TAIL_DEEP_DEPTH = 6          # extra stages granted from TAIL_DEEP_PROGRESS (cap-limited to 20)
# THE ONE CHANGE FROM LAUNCH 25: the depth walks to its final value one microbatch at a time,
# ACCUM_RAMP_STAGE_PROGRESS of the clock per stage, instead of stepping there in a single
# update. Launch 25 measured why this matters. Its LR rider is correct about path length and
# it is what makes the ramp measure a batch change rather than a late LR cut, but as a step
# function it multiplied lrm by 2.0 in one update: at the switch lrm was 0.3962, so the
# effective multiplier jumped to 0.7924, the value get_lr_multiplier last returned at
# progress ~0.604. That un-anneals ~118 s of a monotone warmdown instantly, and the probe
# straddling it recorded val_bpb RISING 0.000343 over 30 s that delivered 20.1M tokens, where
# the champion gained 0.009134 -- 0.009476 bpb of forgone gain inside one probe. It then
# recovered at 1.504x / 1.207x / 1.329x the champion's per-probe gain, clawing back 0.006208
# by the end of the clock, and missed the 1.05 gate by 0.000415 bpb. So the tail's TOKENS are
# not the problem -- at k=5 the tail is ahead of the champion on every 30 s window once it
# settles -- the DISCONTINUITY is. Walking 5 -> 6 -> 7 -> 8 -> 9 -> 10 makes the largest
# single jump in the effective multiplier +20% instead of +100% (peak excess +28% against
# +100%), reaches the same final depth by progress 0.88, and leaves three full probes at full
# depth. Everything else in this file is launch 25's bytes.
ACCUM_RAMP_STAGE_PROGRESS = 0.033  # clock fraction per +1 microbatch of accumulation depth

# Model size
DEPTH = 8               # number of transformer layers
DEVICE_BATCH_SIZE = 128  # per-device batch size (reduce if OOM)

# Data budget: the number of DISTINCT training tokens this run may see, taken from the
# head of prepare.py's fixed global shuffle of the whole pool. This single integer is the
# entire data-efficiency surface -- there is no shard list to choose. 631,241,817 is the
# whole pool. Reducing it exposes fewer tokens of the same representative sample, and
# repetition follows automatically as consumption / budget.
DATA_BUDGET_TOKENS = 631_241_817

# Packing/staging: how many encoded documents best-fit packing may choose from. THE CHANGE:
# 4000 -> 1000, back to prepare.py's make_dataloader default, which is also the value the
# frozen evaluate_bpb builds ITS loader with. 4000 was carried for delivery, and the
# inherited one-deep event wait now owns delivery at any buffer size and any depth.
PACK_BUFFER_SIZE = 1000

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
# The deepest accumulation this run can reach, i.e. the size every per-microbatch instrument
# must be allocated for. DEVICE_BATCH_SIZE is unchanged, so the microbatch shape -- and with it
# the compiled graph, the FLOPs probe input and the peak allocation -- is identical at any m.
MAX_ACCUM_STEPS = grad_accum_steps * ACCUM_RAMP_FACTOR
TAIL_BATCH_SIZE = MAX_ACCUM_STEPS * tokens_per_fwdbwd

optimizer = model.setup_optimizer(
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    scalar_lr=SCALAR_LR,
    adam_betas=ADAM_BETAS,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY,
)

# Read-only rung/LR witness, so this launch's own record states the two numbers the
# paired control (cal_lr_level_x1p5, the same LR_SCALE at the champion's B) needs.
print(f"Rung:            TOTAL_BATCH_SIZE {TOTAL_BATCH_SIZE:,} = {grad_accum_steps} x "
      f"{tokens_per_fwdbwd:,} (DEVICE_BATCH_SIZE {DEVICE_BATCH_SIZE} x MAX_SEQ_LEN "
      f"{MAX_SEQ_LEN}); k = {TOTAL_BATCH_SIZE / 2**19:g} vs the reference's 2**19")
print(f"Tail ramp:       from progress {ACCUM_RAMP_PROGRESS} the depth WALKS "
      f"{grad_accum_steps} -> {MAX_ACCUM_STEPS} one microbatch per "
      f"{ACCUM_RAMP_STAGE_PROGRESS} of the clock, reaching {MAX_ACCUM_STEPS} at progress "
      f"{ACCUM_RAMP_PROGRESS + (MAX_ACCUM_STEPS - grad_accum_steps - 1) * ACCUM_RAMP_STAGE_PROGRESS:g}"
      f"; B {TOTAL_BATCH_SIZE:,} -> {TAIL_BATCH_SIZE:,} (k {TOTAL_BATCH_SIZE / 2**19:g} -> "
      f"{TAIL_BATCH_SIZE / 2**19:g}), lr always x m_now/{grad_accum_steps}")
print("Depth schedule:  " + "  ".join(
    f"p>={ACCUM_RAMP_PROGRESS + i * ACCUM_RAMP_STAGE_PROGRESS:.2f}:m={grad_accum_steps + i + 1}"
    f"(eff={2 * (1 - (ACCUM_RAMP_PROGRESS + i * ACCUM_RAMP_STAGE_PROGRESS)) * (grad_accum_steps + i + 1) / grad_accum_steps:.4f})"
    for i in range(MAX_ACCUM_STEPS - grad_accum_steps)))
print(f"LR level:        x{LR_SCALE} (embedding {EMBEDDING_LR:g}, unembedding "
      f"{UNEMBEDDING_LR:g}, matrix {MATRIX_LR:g}, scalar {SCALAR_LR:g})")
for _g in optimizer.param_groups:
    print(f"[INSTR] lr group kind={_g['kind']:9s} initial_lr={_g['initial_lr']:.6g} "
          f"n_params={sum(p.numel() for p in _g['params']):,}")

model = torch.compile(model, dynamic=False)

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train",
                              buffer_size=PACK_BUFFER_SIZE,
                              data_budget_tokens=DATA_BUDGET_TOKENS)
print(f"Packing buffer:  {PACK_BUFFER_SIZE:,} documents (make_dataloader default 1000)")

# Peak attribution rider. Read-only: max_memory_allocated() never resets a counter, and
# reset_peak_memory_stats() is exactly what the task forbids. These four reads locate the
# high-water mark that peak_vram_bytes reports, which is the headroom every VRAM-touching
# axis on the board is currently guessing at.
def _vram(tag):
    print(f"[INSTR] vram {tag:26s} peak={torch.cuda.max_memory_allocated():,} "
          f"live={torch.cuda.memory_allocated():,}")

_vram("model+optim+loader")
print(f"Data budget:    {DATA_BUDGET_TOKENS:,} distinct tokens from the shuffled pool")
harness = (TimeToTargetHarness(TARGET_VAL_BPB)
           if (TARGET_VAL_BPB > 0 or PROBE_ONLY) else None)
x, y, epoch = next(train_loader)  # prefetch first batch
# Dispatch-level FLOPs, measured HERE while memory still holds only the model and the
# optimizer state. Counted on the uncompiled module by the frozen probe.
from prepare import measure_flops_dispatch
flops_per_token_measured = measure_flops_dispatch(model, x, y)
print(f"FLOPs per token (measured at dispatch): {flops_per_token_measured:,}")

_vram("after flops probe")

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")

# Host/device attribution rider. CUDA events are not caching-allocator objects, so these
# cost nothing against the zero-headroom peak_vram ceiling. Allocated once and reused.
#   P = host seconds inside one next(train_loader)      (perf_counter, host only)
#   G = device ms for one microbatch's forward+backward (events, read after the sync)
#   S = device ms for optimizer.step() + zero_grad()    (events, read after the sync)
# Every event read happens AFTER the update's own torch.cuda.synchronize(), so no
# instrument adds a synchronization the reference did not already have -- which matters,
# because a sync placed inside the accumulation loop is itself the defect's masking fix.
# Allocated for the DEEPEST phase, so the ramp reuses the same events and adds no allocation.
_ev_mb = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
          for _ in range(MAX_ACCUM_STEPS)]
_ev_opt = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
_instr_P, _instr_G, _instr_S, _instr_dt = [], [], [], []
_instr_wait = []   # host stall on C_{i-1}: the realized margin G - h, per update
_instr_m = []      # the accumulation depth each update actually ran at (variable under the ramp)

# Precondition (c): the delivery guard, as a one-deep event wait on the loader's staging
# copy. prepare.py:382-388 allocates row_buffer / pinned cpu_buffer / gpu_buffer ONCE and
# every yield returns the same views; each next() writes the pinned buffer (W_i) and then
# enqueues gpu_buffer.copy_(cpu_buffer, non_blocking=True) (C_i) on the default stream,
# behind the backward just issued. Waiting for C_{i-1} -- and only C_{i-1} -- before
# overwriting the pinned buffer makes the host beat i*G rather than (i+1)*G, so it costs
# nothing iff h <= G. Measured free on this substrate: +118.13 ms margin per microbatch
# (gpu1 launch 3), and that condition is per-microbatch, so it holds at every m. Zero
# device bytes: a CUDA event is not a caching-allocator object, which is what makes this
# admissible against a peak_vram ceiling the reference sits on exactly.
copy_done = torch.cuda.Event()
copy_done_armed = False

# Delivery fingerprints, in page-locked HOST memory (peak_vram_bytes does not count it).
# x[row, 1:9] is a contiguous view -- no reduction, no temporary, no device allocation.
# Never .item() inside the loop: that host sync is itself the masking fix for the defect
# being measured. Read once after the loop behind one synchronize().
FP_CAP = 4096
fp_host = torch.empty((FP_CAP, 24), dtype=torch.long, pin_memory=True)
fp_idx = 0

# Schedules (all based on progress = training_time / TIME_BUDGET)

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

# Precondition (a): the file's only step-keyed schedule, re-keyed to tokens so the ramp
# covers the same fraction of a 600 s run at any TOTAL_BATCH_SIZE. Exactly the reference's
# min(step/300, 1) at TOTAL_BATCH_SIZE = 2**19; min(step/200, 1) at this rung.
MOMENTUM_RAMP_TOKENS = 300 * 2**19   # the reference's 300-update ramp, expressed in tokens

def get_muon_momentum(tokens_so_far):
    # Keyed on the tokens ACTUALLY consumed rather than step * TOTAL_BATCH_SIZE. At constant
    # depth the two are the same integer at every call (tokens_consumed is incremented by
    # TOTAL_BATCH_SIZE once per update, after this call), so this is bit-identical to the
    # champion; under the ramp the product would undercount the tail. Provably inert here
    # either way: frac saturates at 1 after 120 updates (157,286,400 / 1,310,720) and the
    # switch cannot fire before ~update 250.
    frac = min(tokens_so_far / MOMENTUM_RAMP_TOKENS, 1)
    return (1 - frac) * 0.85 + frac * 0.92

# THE CHANGE: the shape of the weight-decay schedule, at matched realized total. The exponent goes
# 1 -> 2 and WD_SHAPE_MATCH rescales the level so sum(lrm * m/5 * shape) over the champion's own
# realized 275 updates is unchanged at 148.122523 per unit WEIGHT_DECAY. WEIGHT_DECAY itself stays
# 0.2. gpu2's launch 19 took the same total the other way (exponent 0, flat) and measured +0.001086
# bpb at its deciding probe; with the champion at exponent 1 these three points are equally spaced in
# the exponent, so the parabola through them has no fitting freedom.
WD_SHAPE_EXPONENT = 2
WD_SHAPE_MATCH = 1.37464969   # = 148.122523 / 107.752923, computed on launch 53's realized schedule


def get_weight_decay(progress):
    return WEIGHT_DECAY * WD_SHAPE_MATCH * (1 - progress) ** WD_SHAPE_EXPONENT

def get_accum_depth(progress):
    """Accumulation depth for this update: grad_accum_steps, walking up by one microbatch every
    ACCUM_RAMP_STAGE_PROGRESS of the clock once progress reaches ACCUM_RAMP_PROGRESS, and held
    at MAX_ACCUM_STEPS thereafter. Monotone non-decreasing and a pure function of progress, so
    the depth this update ran at is reconstructible from the log without any state."""
    if progress < ACCUM_RAMP_PROGRESS:
        return grad_accum_steps
    stages = int((progress - ACCUM_RAMP_PROGRESS) / ACCUM_RAMP_STAGE_PROGRESS) + 1
    stages += TAIL_DEEP_DEPTH if progress >= TAIL_DEEP_PROGRESS else 0
    return grad_accum_steps + min(stages, MAX_ACCUM_STEPS - grad_accum_steps)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
tokens_consumed = 0
step = 0
_switch_step = None   # the update index at which the ramp fired, for the record
# THE CHANGE's own witness: the realized decay coefficient this run actually applied, summed over
# the updates it actually performed. muon_step_fused uses lr*wd with lr = initial_lr*lrm*lr_factor,
# so this sum IS the quantity WD_SHAPE_MATCH was chosen to hold. Read-only, off both clocks.
_wd_realized_sum = 0.0

while True:
    torch.cuda.synchronize()
    t0 = time.time()
    _P = 0.0
    _W = 0.0
    # Progress, read BEFORE the accumulation loop because the loop's depth depends on it.
    # total_training_time is not modified anywhere inside this update, so this line is
    # value-identical to the champion's placement after the loop; the schedules below still
    # consume the same number they consumed there.
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    m_now = get_accum_depth(progress)
    B_now = m_now * tokens_per_fwdbwd
    for micro_step in range(m_now):
        _ev_mb[micro_step][0].record()
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / m_now
        loss.backward()
        _ev_mb[micro_step][1].record()
        # Fingerprint what THIS microbatch actually trained on. Enqueued after bwd_i and
        # before C_{i+1}, so it reads exactly the bytes the forward read.
        if fp_idx < FP_CAP:
            fp_host[fp_idx, 0:8].copy_(x[0, 1:9], non_blocking=True)
            fp_host[fp_idx, 8:16].copy_(x[64, 1:9], non_blocking=True)
            fp_host[fp_idx, 16:24].copy_(x[127, 1:9], non_blocking=True)
            fp_idx += 1
        _t_wait = time.perf_counter()
        if copy_done_armed:
            copy_done.synchronize()  # C_{i-1} has landed; W_i may now overwrite cpu_buffer
        _t_next = time.perf_counter()
        x, y, epoch = next(train_loader)
        copy_done.record()           # covers C_i, enqueued inside next()
        copy_done_armed = True
        _P += time.perf_counter() - _t_next
        _W += _t_next - _t_wait

    # Schedules (progress was read above, before the accumulation loop)
    lrm = get_lr_multiplier(progress)
    lr_factor = m_now / grad_accum_steps   # precondition (b): 1.0 in phase 1, 2.0 in the tail
    muon_momentum = get_muon_momentum(tokens_consumed)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm * lr_factor
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    _ev_opt[0].record()
    optimizer.step()
    model.zero_grad(set_to_none=True)
    _ev_opt[1].record()

    train_loss_f = train_loss.item()

    # Fast fail: abort if loss is exploding or NaN
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    # Instrument reads. Placed after the update's own synchronize() and after dt is
    # closed, so the reads add nothing to the duration the harness is ticked with, and
    # the events they read are already complete -- no instrument sync is introduced.
    # _P is the host total across this update's own m_now calls to next(), and _instr_m records
    # that depth so every median below can be taken per phase.
    _instr_P.append(_P)
    _instr_G.append([_ev_mb[i][0].elapsed_time(_ev_mb[i][1])
                     for i in range(m_now)])
    _instr_S.append(_ev_opt[0].elapsed_time(_ev_opt[1]))
    _instr_dt.append(dt)
    _instr_wait.append(_W)
    _instr_m.append(m_now)
    _wd_realized_sum += lrm * lr_factor * muon_weight_decay

    # The ramp's own witness: one line per depth STAGE, off the clock and read-only. The
    # effective multiplier is the number launch 25 says decides this experiment, so it is
    # printed at every stage alongside the champion's own lrm at the same progress.
    _depth_changed = len(_instr_m) > 1 and _instr_m[-1] != _instr_m[-2]
    if _depth_changed:
        if _switch_step is None:
            _switch_step = step
        print(f"\n[ramp] stage depth {_instr_m[-2]} -> {m_now} at step {step:05d} "
              f"(clocked {step - 10}) progress={progress:.4f} lrm={lrm:.4f} "
              f"lr_factor={lr_factor:g} effective_multiplier={lrm * lr_factor:.4f} "
              f"(champion's is lrm itself = {lrm:.4f}; launch 25 jumped to 0.7924 from 0.3962 "
              f"in one update) B={B_now:,} dt={dt * 1000:.1f}ms", flush=True)

    if step > 10:
        total_training_time += dt

    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(B_now / dt)
    mfu = 100 * num_flops_per_token * B_now / dt / H100_BF16_PEAK_FLOPS
    remaining = max(0, TIME_BUDGET - total_training_time)

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | m: {m_now} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

    if step in (1, 11) or step % 100 == 0 or (_switch_step is not None
                                             and step - _switch_step in (0, 1, 2)):
        _g = _instr_G[-1]
        _acct = _P * 1000 + sum(_g) + _instr_S[-1]
        print(f"\n[INSTR] step {step:05d} m={m_now} B={B_now:,} "
              f"host_next_total={_P*1000:.1f}ms "
              f"(={_P*1000/m_now:.1f}ms x {m_now}) "
              f"G={'/'.join(f'{v:.1f}' for v in _g)}ms S={_instr_S[-1]:.1f}ms "
              f"dt={dt*1000:.1f}ms accounted={_acct:.1f}ms "
              f"host_frac={_P/dt:.3f} "
              f"unguarded_race_safe_strict={_P*1000 > (m_now - 1) * _g[0]} "
              f"guard_wait={_W*1000:.1f}ms "
              f"peak={torch.cuda.max_memory_allocated():,}")

    # GC management (Python's GC causes ~500ms stalls)
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1
    tokens_consumed += B_now

    # Axis H: the frozen harness owns the clock, the probe cadence and the crossing. tick
    # takes THIS update's own token count -- prepare.py:769 says total_tokens is summed as
    # consumed "not num_steps * tokens_per_step: that product is only right when every update
    # used the same number of tokens" -- so passing the constant here would make the declared
    # tiebreak tokens_to_target wrong by the whole tail.
    if harness is not None and harness.tick(model, tokenizer, dt, B_now):
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
# Rider summary. Everything below is printed BEFORE the canonical reporter, reads only
# instrument state, and prints no official metrics line of its own.
# ---------------------------------------------------------------------------
import statistics as _st

_clocked = [i for i in range(len(_instr_dt)) if i > 10]
if _clocked:
    _P_tot = [_instr_P[i] * 1000 for i in _clocked]                     # host ms per update
    _P_one = [_instr_P[i] * 1000 / _instr_m[i] for i in _clocked]       # host ms per next()
    _S_all = [_instr_S[i] for i in _clocked]
    _dt_all = [_instr_dt[i] * 1000 for i in _clocked]
    _Pm, _Sm = _st.median(_P_one), _st.median(_S_all)
    # Under the ramp there is no fixed set of microbatch indices to take a per-index median
    # over, so G is pooled over every microbatch of every clocked update, and the per-phase
    # table below carries the two dt values this launch is actually judged on.
    _G_flat = [g for i in _clocked for g in _instr_G[i]]
    _G1 = _st.median([_instr_G[i][0] for i in _clocked])
    _Gm_one = _st.median(_G_flat)
    _safe = sum(1 for i in _clocked if _instr_P[i] * 1000 > _instr_G[i][0])
    print("[INSTR] ---- host/device attribution over "
          f"{len(_clocked)} clocked updates (buffer_size={PACK_BUFFER_SIZE}) ----")
    print(f"[INSTR] P per next()      median={_Pm:.2f}ms  mean={_st.mean(_P_one):.2f}ms  "
          f"min={min(_P_one):.2f}ms  max={max(_P_one):.2f}ms  "
          f"sd={(_st.stdev(_P_one) if len(_P_one) > 1 else 0.0):.2f}ms")
    print(f"[INSTR] P per update      median={_st.median(_P_tot):.2f}ms "
          f"(= m x P, m in {sorted(set(_instr_m[i] for i in _clocked))})")
    print(f"[INSTR] G per microbatch  median={_Gm_one:.2f}ms over {len(_G_flat)} "
          f"microbatches (index 0 only: {_G1:.2f}ms)")
    print(f"[INSTR] S optimizer.step  median={_Sm:.2f}ms")
    print(f"[INSTR] dt               median={_st.median(_dt_all):.2f}ms  "
          f"host_frac={_st.median(_P_tot) / _st.median(_dt_all):.4f}")
    # THE deliverable of this launch: the realized dt of each phase against the pre-registered
    # dt(m) = 401.91*m + 6.65 ms, and the tokens each phase actually delivered.
    for _mv in sorted(set(_instr_m[i] for i in _clocked)):
        _rm = [i for i in _clocked if _instr_m[i] == _mv]
        _dt_m = _st.median([_instr_dt[i] * 1000 for i in _rm])
        _G_m = _st.median([g for i in _rm for g in _instr_G[i]])
        _S_m = _st.median([_instr_S[i] for i in _rm])
        print(f"[INSTR] phase m={_mv:<3d} clocked_updates={len(_rm):<4d} "
              f"dt median={_dt_m:.2f}ms (L25-fitted {400.366 * _mv + 11.35:.2f}ms, "
              f"error {_dt_m - (400.366 * _mv + 11.35):+.2f}ms)  G/microbatch={_G_m:.2f}ms  "
              f"S={_S_m:.2f}ms  accounted={_st.median([_instr_P[i] * 1000 for i in _rm]) + _mv * _G_m + _S_m:.2f}ms  "
              f"clocked_tokens={len(_rm) * _mv * tokens_per_fwdbwd:,}  "
              f"R_phase={len(_rm) * _mv * tokens_per_fwdbwd / (len(_rm) * _dt_m / 1000):,.0f} tok/s")
    # Staging boundary. C_i executes at about i*G, W_{i+1} lands at about (i+1)*P, so the
    # first clobbered microbatch index is i0 = ceil(P/(G-P)); at grad_accum_steps = 2 the
    # only exposed index is i = 1 and it is safe exactly when 2P > G.
    if _Pm >= _G1:
        _i0 = "none (P >= G: the host never runs ahead)"
    else:
        _i0 = str(math.ceil(_Pm / (_G1 - _Pm)))
    print(f"[INSTR] staging boundary  P_update={_st.median(_P_tot):.2f}ms vs "
          f"G_microbatch={_G1:.2f}ms -> first clobbered index i0={_i0}; "
          f"exposed indices per update = "
          f"{max(0, max(_instr_m[i] for i in _clocked) - 1)} at the deepest phase; "
          f"updates with P_update > G = {_safe}/{len(_clocked)}")
    # Precondition (c)'s price, measured. The guard is free iff h <= G; the host stall on
    # C_{i-1} IS the realized margin, so a stall near zero means the host is the limiter
    # and the guard cost R. Compare against the +118.13 ms/microbatch gpu1 measured at
    # buffer_size=1000, m=2 -- at buffer_size=4000 h is larger, so this is the reading
    # that says whether the guard stayed free at this rung.
    _Wm = _st.median([_instr_wait[i] * 1000 / _instr_m[i] for i in _clocked])
    print(f"[INSTR] guard wait-on-C(i-1) median={_Wm:.2f}ms per microbatch "
          f"(gpu1 launch 3: 118.13ms at buffer_size=1000, m=2); "
          f"guard_free={_Wm > 0.5}")
    _vram("end of training")

# Delivery fingerprints. All of this is off the harness clock (total_training_time stopped
# accumulating when the loop exited) and strictly before the reporter, so it cannot touch a
# scored number. The reporter below remains the only official METRICS_JSON line.
torch.cuda.synchronize()   # the fingerprint D2H copies are async; read them only now
_n_fp = min(fp_idx, FP_CAP)
if _n_fp:
    _rows = [tuple(r) for r in fp_host[:_n_fp].tolist()]
    _distinct = len(set(_rows))
    _adj = sum(1 for k in range(1, _n_fp) if _rows[k] == _rows[k - 1])
    # Grouped by the depth each update ACTUALLY ran at: a fixed stride would misalign every
    # update after the ramp fires and would report seam duplicates that are not there.
    _full = _clean = _seam = 0
    _clean_by_m, _full_by_m = {}, {}
    _off, _prev_last = 0, None
    for _mu in _instr_m:
        if _off + _mu > _n_fp:
            break
        _grp = _rows[_off:_off + _mu]
        _full += 1
        _full_by_m[_mu] = _full_by_m.get(_mu, 0) + 1
        if len(set(_grp)) == _mu:
            _clean += 1
            _clean_by_m[_mu] = _clean_by_m.get(_mu, 0) + 1
        if _prev_last is not None and _grp[0] == _prev_last:
            _seam += 1
        _prev_last = _grp[-1]
        _off += _mu
    print(f"[fp] microbatches={_n_fp} distinct_fingerprints={_distinct} "
          f"({100.0 * _distinct / _n_fp:.1f}%)  adjacent_duplicates={_adj}")
    print(f"[fp] updates with all m positions distinct: {_clean}/{_full}   "
          f"update-seam duplicates: {_seam}/{max(1, _full - 1)}   "
          f"by depth: {{m: clean/total}} = "
          f"{ {k: f'{_clean_by_m.get(k, 0)}/{v}' for k, v in sorted(_full_by_m.items())} }")
    print(f"[fp] PRE-REGISTERED: 100% distinct and 0 seam duplicates AT BOTH DEPTHS => every "
          f"update was delivered every microbatch it was charged for, so this launch's curve "
          f"prices the TAIL token-efficiency penalty at k {TOTAL_BATCH_SIZE / 2**19:g} -> "
          f"{TAIL_BATCH_SIZE / 2**19:g}. Anything less at m={MAX_ACCUM_STEPS} and the tail "
          f"reading is a delivery reading, not a batch reading -- at buffer_size "
          f"{PACK_BUFFER_SIZE} the unguarded i0 is 1, so all {MAX_ACCUM_STEPS - 1} exposed "
          f"indices depend on the event-wait guard.")
    print(f"[fp] first 6 fingerprints (row0 tokens 1..3): {[r[:3] for r in _rows[:6]]}")
print(f"[rung] TOTAL_BATCH_SIZE={TOTAL_BATCH_SIZE:,} DEVICE_BATCH_SIZE={DEVICE_BATCH_SIZE} "
      f"grad_accum_steps={grad_accum_steps} LR_SCALE={LR_SCALE} "
      f"momentum_ramp_updates={MOMENTUM_RAMP_TOKENS / TOTAL_BATCH_SIZE:.1f}")

# The ramp's accounting, in full, because tokens_per_step is no longer a single constant of
# this program. Every number below is a count this launch actually performed.
_upd_by_m = {}
for _mu in _instr_m:
    _upd_by_m[_mu] = _upd_by_m.get(_mu, 0) + 1
_tok_by_m = {k: v * k * tokens_per_fwdbwd for k, v in _upd_by_m.items()}
_mean_tokens_per_update = tokens_consumed / step if step else 0.0
print(f"[ramp] ACCUM_RAMP_PROGRESS={ACCUM_RAMP_PROGRESS} ACCUM_RAMP_FACTOR={ACCUM_RAMP_FACTOR} "
      f"switch_at_step={_switch_step} (clocked "
      f"{None if _switch_step is None else _switch_step - 10})")
print(f"[ramp] updates by depth {_upd_by_m}  tokens by depth {_tok_by_m}  "
      f"sum={sum(_tok_by_m.values()):,} == tokens_consumed {tokens_consumed:,}: "
      f"{sum(_tok_by_m.values()) == tokens_consumed}")
print(f"[ramp] tokens_per_step passed to the reporter = tokens_consumed / num_steps = "
      f"{tokens_consumed:,} / {step} = {_mean_tokens_per_update:.2f}. It is the MEAN tokens "
      f"per completed update, which is what 'tokens per step' means for a run whose updates "
      f"are not all the same size; the phase constants are {TOTAL_BATCH_SIZE:,} and "
      f"{TAIL_BATCH_SIZE:,} above, and total_tokens is summed as consumed either way. "
      f"prepare.py uses this argument only for train_tokens_per_second, and the mean is what "
      f"makes that number the throughput this launch actually achieved.")

# THE CHANGE's witness. champion_sum is WEIGHT_DECAY * 148.122523 = 29.624505, launch 53's own
# realized total. matched=True requires the ratio inside 1.000 +/- 0.005; anything outside means the
# level moved as well as the shape and the reading is a level reading, not a shape reading. The sum
# runs over REALIZED updates, so a run that crosses at a different step will differ by that step's
# own contribution -- at this exponent the last updates carry wd ~ 1e-5, so the ratio is insensitive
# to the crossing step, which is the defect gpu2 reported in launch 19's pre-registration.
_wd_champion_sum = 29.624505
print(f"[wd] shape exponent={WD_SHAPE_EXPONENT} match_constant={WD_SHAPE_MATCH} "
      f"WEIGHT_DECAY={WEIGHT_DECAY} sum(lrm*lr_factor*wd)={_wd_realized_sum:.6f} "
      f"champion_sum={_wd_champion_sum:.6f} ratio={_wd_realized_sum / _wd_champion_sum:.6f} "
      f"matched={abs(_wd_realized_sum / _wd_champion_sum - 1.0) <= 0.005} "
      f"updates={len(_instr_m)}")

# Every score comes from the FROZEN prepare.py. train.py may change the model; it may
# not compute the number the model is judged on.
report_efficiency_metrics(
    model, tokenizer,
    num_steps=step,
    tokens_per_step=_mean_tokens_per_update,
    total_tokens=tokens_consumed,
    training_seconds=total_training_time,
    total_seconds=t_end - t_start,
    final_epoch=epoch,
    flops_measured=flops_per_token_measured,
    harness=harness,
)
print(f"depth:            {DEPTH}")
