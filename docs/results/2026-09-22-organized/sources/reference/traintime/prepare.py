"""
One-time data preparation for autoresearch experiments.
Downloads data shards and trains a BPE tokenizer.

Usage:
    python prepare.py                  # full prep (download + tokenizer)
    python prepare.py --num-shards 8   # download only 8 shards (for testing)

Data and tokenizer are stored in ~/.cache/autoresearch/.
"""

import gc
import json
import os
import random
import sys
import time
import math
import argparse
import pickle
from multiprocessing import Pool

import requests
import pyarrow.parquet as pq
import rustbpe
import tiktoken
import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 2048       # context length
TIME_BUDGET = 600        # training time budget in seconds (10 minutes)
EVAL_TOKENS = 40 * 524288  # number of tokens for val eval
EVAL_BATCH_SIZE = 128    # FROZEN eval batch. It must not come from train.py: a batch
                         # the candidate chooses moves the number of evaluated tokens with
                         # a tuning knob. Frozen here so val_bpb means one thing on every
                         # candidate.

# Deterministic global document order. The training stream is a fixed shuffle of the
# WHOLE pool, and a data budget takes a prefix of it -- so reducing data means "fewer
# tokens", never "a different set of shards". Choosing WHICH shards to expose -- one shard
# alone, validation-proximal shards, token-matched prefixes -- is not a degree of freedom
# this offers.
DATA_SHUFFLE_SEED = 20260819
DATA_BUDGET_TOKENS_DEFAULT = None   # None = the whole pool

#: The training pool the loader was actually given, recorded by make_dataloader when it
#: resolves it. The reporter reads this rather than a constant: the metric is what the run
#: really exposed, so the `training_data_tokens_available == 631241817` ceiling can still
#: fail. A hard-coded constant would make that pin unfailable and therefore meaningless.
#: It is not an argument of the reporter either, because a number train.py hands over is a
#: number the ceiling would rest on train.py's word for.
#:
#: The sentinel is not None, because None is a legitimate value for that argument and not
#: only an "unset" marker: DATA_BUDGET_TOKENS_DEFAULT is None and the loader reads it as
#: "the whole pool". Conflating the two would make a train loader built without a budget
#: look like no train loader at all, and the reporter would then crash on a false diagnosis
#: instead of reporting a value that fails the pin -- which is the outcome that pin is for.
_UNSET = object()
_RESOLVED_TRAIN_BUDGET = _UNSET

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")
BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542 # the last datashard is shard_06542.parquet
VAL_SHARD = MAX_SHARD  # pinned validation shard (shard_06542)
VAL_FILENAME = f"shard_{VAL_SHARD:05d}.parquet"
VOCAB_SIZE = 8192

# BPE split pattern (GPT-4 style, with \p{N}{1,2} instead of {1,3})
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_single_shard(index):
    """Download one parquet shard with retries. Returns True on success."""
    filename = f"shard_{index:05d}.parquet"
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        return True

    url = f"{BASE_URL}/{filename}"
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            temp_path = filepath + ".tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, filepath)
            print(f"  Downloaded {filename}")
            return True
        except (requests.RequestException, IOError) as e:
            print(f"  Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
    return False


def download_data(num_shards, download_workers=8):
    """Download training shards + pinned validation shard."""
    os.makedirs(DATA_DIR, exist_ok=True)
    num_train = min(num_shards, MAX_SHARD)
    ids = list(range(num_train))
    if VAL_SHARD not in ids:
        ids.append(VAL_SHARD)

    # Count what's already downloaded
    existing = sum(1 for i in ids if os.path.exists(os.path.join(DATA_DIR, f"shard_{i:05d}.parquet")))
    if existing == len(ids):
        print(f"Data: all {len(ids)} shards already downloaded at {DATA_DIR}")
        return

    needed = len(ids) - existing
    print(f"Data: downloading {needed} shards ({existing} already exist)...")

    workers = max(1, min(download_workers, needed))
    with Pool(processes=workers) as pool:
        results = pool.map(download_single_shard, ids)

    ok = sum(1 for r in results if r)
    print(f"Data: {ok}/{len(ids)} shards ready at {DATA_DIR}")

# ---------------------------------------------------------------------------
# Tokenizer training
# ---------------------------------------------------------------------------

def list_parquet_files():
    """Return sorted list of parquet file paths in the data directory."""
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet") and not f.endswith(".tmp"))
    return [os.path.join(DATA_DIR, f) for f in files]


def text_iterator(max_chars=1_000_000_000, doc_cap=10_000):
    """Yield documents from training split (all shards except pinned val shard)."""
    parquet_paths = [p for p in list_parquet_files() if not p.endswith(VAL_FILENAME)]
    nchars = 0
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            for text in rg.column("text").to_pylist():
                doc = text[:doc_cap] if len(text) > doc_cap else text
                nchars += len(doc)
                yield doc
                if nchars >= max_chars:
                    return


def train_tokenizer():
    """Train BPE tokenizer using rustbpe, save as tiktoken pickle."""
    tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")

    if os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_path):
        print(f"Tokenizer: already trained at {TOKENIZER_DIR}")
        return

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    parquet_files = list_parquet_files()
    if len(parquet_files) < 2:
        print("Tokenizer: need at least 2 data shards (1 train + 1 val). Download more data first.")
        sys.exit(1)

    # --- Train with rustbpe ---
    print("Tokenizer: training BPE tokenizer...")
    t0 = time.time()

    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = VOCAB_SIZE - len(SPECIAL_TOKENS)
    tokenizer.train_from_iterator(text_iterator(), vocab_size_no_special, pattern=SPLIT_PATTERN)

    # Build tiktoken encoding from trained merges
    pattern = tokenizer.get_pattern()
    mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
    tokens_offset = len(mergeable_ranks)
    special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="rustbpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    # Save tokenizer
    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)

    t1 = time.time()
    print(f"Tokenizer: trained in {t1 - t0:.1f}s, saved to {tokenizer_pkl}")

    # --- Build token_bytes lookup for BPB evaluation ---
    print("Tokenizer: building token_bytes lookup...")
    special_set = set(SPECIAL_TOKENS)
    token_bytes_list = []
    for token_id in range(enc.n_vocab):
        token_str = enc.decode([token_id])
        if token_str in special_set:
            token_bytes_list.append(0)
        else:
            token_bytes_list.append(len(token_str.encode("utf-8")))
    token_bytes_tensor = torch.tensor(token_bytes_list, dtype=torch.int32)
    torch.save(token_bytes_tensor, token_bytes_path)
    print(f"Tokenizer: saved token_bytes to {token_bytes_path}")

    # Sanity check
    test = "Hello world! Numbers: 123. Unicode: 你好"
    encoded = enc.encode_ordinary(test)
    decoded = enc.decode(encoded)
    assert decoded == test, f"Tokenizer roundtrip failed: {test!r} -> {decoded!r}"
    print(f"Tokenizer: sanity check passed (vocab_size={enc.n_vocab})")

# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

class Tokenizer:
    """Minimal tokenizer wrapper. Training is handled above."""

    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def decode(self, ids):
        return self.enc.decode(ids)


def get_token_bytes(device="cpu"):
    path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    with open(path, "rb") as f:
        return torch.load(f, map_location=device)


def _row_group_order(split):
    """Deterministic shuffled list of (filepath, row_group_index) over the split.

    Training draws from every shard in a fixed pseudo-random order, so any prefix of
    the stream is a representative sample of the whole pool rather than the contents of
    whichever shards happen to sort first. Validation is never shuffled: it is one
    pinned shard read in file order, so the eval set is byte-stable.
    """
    parquet_paths = list_parquet_files()
    assert len(parquet_paths) > 0, "No parquet files found. Run prepare.py first."
    val_path = os.path.join(DATA_DIR, VAL_FILENAME)
    if split == "val":
        pf = pq.ParquetFile(val_path)
        return [(val_path, i) for i in range(pf.num_row_groups)]

    parquet_paths = [p for p in parquet_paths if p != val_path]
    assert len(parquet_paths) > 0, "No training shards found."
    groups = []
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        groups.extend((filepath, i) for i in range(pf.num_row_groups))
    random.Random(DATA_SHUFFLE_SEED).shuffle(groups)
    return groups


def _document_batches(split, tokenizer_batch_size=128):
    """Infinite iterator over document batches in the deterministic shuffled order.

    Yields (documents, epoch). One epoch is one pass over the order; rows inside a row
    group are shuffled too, with a seed derived from the group's position, so the mix is
    document-level and still reproducible.
    """
    order = _row_group_order(split)
    shuffle_rows = split == "train"
    epoch = 1
    while True:
        for position, (filepath, rg_idx) in enumerate(order):
            rg = pq.ParquetFile(filepath).read_row_group(rg_idx)
            batch = rg.column('text').to_pylist()
            if shuffle_rows:
                random.Random(DATA_SHUFFLE_SEED + 1 + position).shuffle(batch)
            for i in range(0, len(batch), tokenizer_batch_size):
                yield batch[i:i+tokenizer_batch_size], epoch
        epoch += 1


def make_dataloader(tokenizer, B, T, split, buffer_size=1000,
                    data_budget_tokens=DATA_BUDGET_TOKENS_DEFAULT):
    """
    BOS-aligned dataloader with best-fit packing.
    Every row starts with BOS. Documents packed using best-fit to minimize cropping.
    When no document fits remaining space, crops shortest doc to fill exactly.
    100% utilization (no padding).
    """
    assert split in ["train", "val"]
    row_capacity = T + 1
    batches = _document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    # The data budget, in ENCODED tokens, is the whole data-efficiency knob. The loader
    # exposes exactly this many distinct tokens from the head of the shuffled order and
    # then starts the same prefix again, so `training_data_tokens_available` equals the
    # budget by construction and repetition is simply consumption / budget.
    budget = data_budget_tokens
    spent = [0]
    if split == "train":
        global _RESOLVED_TRAIN_BUDGET
        _RESOLVED_TRAIN_BUDGET = budget   # including None, which means the whole pool

    def refill_buffer():
        nonlocal epoch, batches
        if budget is None:
            doc_batch, epoch = next(batches)
            doc_buffer.extend(tokenizer.encode(doc_batch, prepend=bos_token))
            return
        while len(doc_buffer) == 0 or spent[0] < budget:
            if spent[0] >= budget:
                break
            doc_batch, stream_epoch = next(batches)
            for tokens in tokenizer.encode(doc_batch, prepend=bos_token):
                room = budget - spent[0]
                if room <= 0:
                    break
                if len(tokens) > room:
                    tokens = tokens[:room]        # land exactly on the budget
                doc_buffer.append(tokens)
                spent[0] += len(tokens)
            if spent[0] >= budget:
                # budget consumed: restart the SAME prefix, which is what an epoch is here
                batches = _document_batches(split)
                spent[0] = 0
                epoch += 1
            if len(doc_buffer) >= buffer_size:
                return

    # Pre-allocate buffers: [inputs (B*T) | targets (B*T)]
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device="cuda")
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    # No doc fits — crop shortest to fill remaining
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, epoch

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size):
    """
    Bits per byte (BPB): vocab size-independent evaluation metric.
    Sums per-token cross-entropy (in nats), sums target byte lengths,
    then converts nats/byte to bits/byte. Special tokens (byte length 0)
    are excluded from both sums.
    Uses fixed MAX_SEQ_LEN so results are comparable across configs.
    """
    token_bytes = get_token_bytes(device="cuda")
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction='none').view(-1)
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)



# ---------------------------------------------------------------------------
# Metrics (FROZEN). Every number a candidate is scored on is computed here, never in
# train.py. A number printed by the editable file is a number the candidate can choose.
# ---------------------------------------------------------------------------

METRICS_SCHEMA_VERSION = 1

BENCH_DECODE_STEPS = 128
QUICK_EVAL_TOKENS = 4 * 524288  # mid-run probe eval for the time-to-target harness
PROBE_EVERY_SECONDS = 30.0
A100_BF16_PEAK_FLOPS = 312e12
H100_BF16_PEAK_FLOPS = 989.5e12


def count_params(model):
    """Total parameters over UNIQUE tensors, plus a breakdown that must reconcile.

    Summing named groups is a census, not a total: anything registered elsewhere would be
    invisible while still being trained.
    """
    unique = {id(q): q.numel() for q in model.parameters()}
    total = sum(unique.values())
    def group(module):
        return sum({id(q): q.numel() for q in module.parameters()}.values())
    census = {
        "wte": group(model.transformer.wte),
        "value_embeds": group(model.value_embeds),
        "lm_head": group(model.lm_head) if hasattr(model, "lm_head") else 0,
        "transformer_matrices": group(model.transformer.h),
        "scalars": model.resid_lambdas.numel() + model.x0_lambdas.numel(),
    }
    if sum(census.values()) != total:
        census["UNACCOUNTED"] = total - sum(census.values())
    return {"total": total, **census}


def estimate_flops_analytic(model):
    """Analytic formula. CROSS-CHECK ONLY, never the objective.

    It subtracts wte, the value embeddings and the per-layer scalars by NAME, so any
    arithmetic routed through those tensors is unpriced -- which is how a count can reach 0
    while the model still spends 262,144 FLOPs per token on a tied unembedding.
    """
    nparams = sum(q.numel() for q in model.parameters())
    ve = sum(v.weight.numel() for v in model.value_embeds.values())
    exclude = (model.transformer.wte.weight.numel() + ve
               + model.resid_lambdas.numel() + model.x0_lambdas.numel())
    h = model.config.n_head
    q = model.config.n_embd // model.config.n_head
    t = model.config.sequence_len
    attn = 0
    for window_size in model.window_sizes:
        w = window_size[0]
        attn += 12 * h * q * (t if w < 0 else min(w, t))
    return int(6 * (nparams - exclude) + attn)


FLOPS_PROBE_ROWS = 8   # FLOPs per token is batch-invariant at fixed T, so a small
                       # batch measures the same quantity without a second full-size
                       # activation set. Probed BEFORE the loop, when memory is empty.


def _attention_flops_probe():
    """Context manager tallying attention FLOPs from the shapes FA3 is actually called with.

    FlopCounterMode only knows aten ops with a registered formula, and flash-attention-3
    arrives as a third-party kernel that never reaches aten dispatch -- so attention counts
    ZERO there. Measured on the unmodified baseline: dispatch alone gives 176,163,840,
    which is exactly the analytic formula's 6N weights term with its 62,914,560 attention
    term missing. Left uncorrected, an attention-heavy candidate would be free on this axis.

    This wrapper reads q, k and the window actually passed to the kernel, so it cannot be
    fooled by a declared window pattern: emptying the window LIST sends the analytic span
    term to zero while attention still runs.
    """
    import contextlib

    @contextlib.contextmanager
    def probe(tally):
        # Patch every ALREADY-IMPORTED flash-attention interface, not a fresh get_kernel
        # handle: get_kernel returns a new module object per call, so patching that one
        # leaves the caller's reference untouched and the tally reads zero.
        targets = [m for name, m in list(sys.modules.items())
                   if m is not None and "flash_attn" in name
                   and hasattr(m, "flash_attn_func")]
        if not targets:
            yield          # no FA3 in this process: SDPA paths are counted by the registry
            return
        originals = [(m, m.flash_attn_func) for m in targets]

        def _make_wrapped(original_fn):
            def wrapped(q, k, v, *args, **kwargs):
                # No `except` here: a call this cannot read is a measurement that failed,
                # and a counter that reports zero for work it did not understand is worse
                # than a launch that stops.
                B, T, h, d = q.shape
                S = k.shape[1]
                window = kwargs.get("window_size", (-1, -1))
                if not isinstance(window, (tuple, list)):
                    window = (-1, -1)
                left = window[0]
                span = S if left is None or left < 0 else min(left, S)
                # same convention as the analytic term: 12 * h * d * span per token
                tally[0] += 12 * B * T * h * d * span
                return original_fn(q, k, v, *args, **kwargs)
            return wrapped

        for module, original_fn in originals:
            module.flash_attn_func = _make_wrapped(original_fn)
        try:
            yield
        finally:
            for module, original_fn in originals:
                module.flash_attn_func = original_fn

    return probe


def measure_flops_dispatch(model, x, y, tokens=None):
    """FLOPs per token over one forward+backward, from what actually executed.

    Two instruments, both observational: FlopCounterMode for the matmul family,
    convolution and SDPA as they dispatch, plus an attention tally read off the shapes
    handed to the flash-attention kernel, which dispatch cannot see. Renaming or
    re-purposing a tensor changes nothing: a gather contributes zero because no matmul is
    issued, and the same table inside F.linear contributes its full cost because one is.
    Elementwise and normalisation work is still uncounted, which is why the universal
    target > 0 rule exists.
    """
    from torch.utils.flop_counter import FlopCounterMode
    # Count on the UNCOMPILED module: inductor fuses elementwise work into generated
    # kernels that never reach aten dispatch, so a compiled wrapper can silently
    # undercount. Matmuls stay extern either way; this removes the dependence.
    target = getattr(model, "_orig_mod", model)
    x = x[:FLOPS_PROBE_ROWS].contiguous()
    y = y[:FLOPS_PROBE_ROWS].contiguous()
    tokens = x.numel()
    gc.collect()
    torch.cuda.empty_cache()
    tally = [0]
    attention_probe = _attention_flops_probe()
    counter = FlopCounterMode(display=False)
    with attention_probe(tally):
        with counter:
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = target(x, y)
            loss.backward()
    target.zero_grad(set_to_none=True)
    dispatch = counter.get_total_flops()
    print(f"[flops] dispatch={dispatch:,} attention={tally[0]:,} tokens={tokens:,}")
    return int((dispatch + tally[0]) / tokens)


@torch.no_grad()


def _supports_cache(model):
    return callable(getattr(model, "init_decode_state", None)) and \
           callable(getattr(model, "decode_step", None))


@torch.no_grad()
def _retained_at(model, tokenizer, prefill):
    """Allocated bytes held while a decode state built from `prefill` tokens is alive."""
    loader = make_dataloader(tokenizer, 1, max(MAX_SEQ_LEN, prefill + 8), "val")
    x, _, _ = next(loader)
    prompt = x[:, :prefill].contiguous()
    del loader, x
    torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    state = model.init_decode_state(batch=1, max_len=prefill + BENCH_DECODE_STEPS)
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, state = model.decode_step(prompt, state)
    del logits
    gc.collect(); torch.cuda.synchronize()
    held = torch.cuda.memory_allocated() - base
    del state, prompt
    gc.collect(); torch.cuda.empty_cache()
    return held


@torch.no_grad()
def evaluate_bpb_quick(model, tokenizer):
    """The certifying eval's procedure on a tenth of the budget, for the harness."""
    token_bytes = get_token_bytes(device="cuda")
    loader = make_dataloader(tokenizer, EVAL_BATCH_SIZE, MAX_SEQ_LEN, "val")
    steps = max(1, QUICK_EVAL_TOKENS // (EVAL_BATCH_SIZE * MAX_SEQ_LEN))
    nats, nbytes_total = 0.0, 0
    for _ in range(steps):
        x, y, _ = next(loader)
        loss_flat = model(x, y, reduction="none").view(-1)
        nbytes = token_bytes[y.view(-1)]
        mask = nbytes > 0
        nats += (loss_flat * mask).sum().item()
        nbytes_total += int(nbytes.sum().item())
    return nats / (math.log(2) * nbytes_total)


class TimeToTargetHarness:
    """Owns the clock, the probe cadence, the crossing test and the cap for axis H.

    A loop that trains until TIME_BUDGET pins training_seconds at ~600 in every replica, so
    time cannot be an objective that way. Here quality is the budget. target=0.0 is
    probe-only mode, which is how the calibration run produces the curve that pins tau.
    """

    def __init__(self, target, probe_every=PROBE_EVERY_SECONDS,
                 warmup_steps=10, full_eval=True):
        # full_eval=True probes with the CERTIFYING eval, not the cheap one. The crossing
        # test then uses the same metric as the gate: a quick-eval crossing at 1.05 can
        # certify at 1.052 on the full budget, which would make every candidate on this
        # axis inadmissible for a reason that has nothing to do with its speed. The probe
        # costs ~12 s each and its time is excluded from the clock, so it inflates
        # wall-clock cost and not the measured seconds-to-target.
        self.full_eval = full_eval
        self.target, self.probe_every = target, probe_every
        self.warmup_steps = warmup_steps
        self.training_seconds, self.step, self.tokens = 0.0, 0, 0
        self.next_probe_at = probe_every
        self.curve, self.reached = [], False
        self.seconds_to_target, self.updates_to_target = None, None
        self.tokens_to_target = None

    def tick(self, model, tokenizer, dt, tokens):
        self.step += 1
        self.tokens += int(tokens)
        if self.step > self.warmup_steps:
            self.training_seconds += dt
        if self.training_seconds < self.next_probe_at:
            return False
        self.next_probe_at += self.probe_every
        was_training = model.training
        model.eval()
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            bpb = (evaluate_bpb(model, tokenizer, EVAL_BATCH_SIZE) if self.full_eval
                   else evaluate_bpb_quick(model, tokenizer))
        if was_training:
            model.train()
        self.curve.append((self.training_seconds, self.step, bpb))
        label = "val_bpb" if self.full_eval else "val_bpb_quick"
        print(f"[probe] t={self.training_seconds:7.1f}s step={self.step:6d} "
              f"{label}={bpb:.6f}", flush=True)
        if self.target > 0.0 and bpb < self.target and not self.reached:
            self.reached = True
            self.seconds_to_target = self.training_seconds
            self.updates_to_target = self.step
            self.tokens_to_target = self.tokens
            return True
        return False

    def result(self):
        return {"train_seconds_to_target": self.seconds_to_target,
                "updates_to_target": self.updates_to_target,
                "tokens_to_target": self.tokens_to_target,
                "reached_target": self.reached, "target_val_bpb": self.target,
                "probe_used_full_eval": self.full_eval,
                "quality_time_curve": [{"training_seconds": a, "step": b,
                                        "val_bpb": c} for a, b, c in self.curve]}


def record_harness_result(metrics, *, train_seconds_to_target, updates_to_target,
                          tokens_to_target, reached_target, target_val_bpb,
                          probe_used_full_eval, quality_time_curve):
    """Merge only the keys the harness owns. This signature IS the whitelist.

    `metrics.update(harness.result())` accepted any object with a result(), and update()
    overwrites, so a substitute could replace a measured number. Binding by keyword makes
    an extra key a TypeError and a missing one a TypeError, so a fabricated result stops
    the launch instead of scoring.
    """
    metrics["train_seconds_to_target"] = train_seconds_to_target
    metrics["updates_to_target"] = updates_to_target
    metrics["tokens_to_target"] = tokens_to_target
    metrics["reached_target"] = reached_target
    metrics["target_val_bpb"] = target_val_bpb
    metrics["probe_used_full_eval"] = probe_used_full_eval
    metrics["quality_time_curve"] = quality_time_curve


def report_efficiency_metrics(model, tokenizer, *, num_steps, tokens_per_step,
                              total_tokens, training_seconds, total_seconds,
                              final_epoch, flops_measured=None, harness=None):
    """The ONE metric printer. METRICS_JSON is the accessor's only source.

    val_bpb is always measured here, on the model handed over. The data budget is this
    module's own constant. Neither is an argument: a number a candidate can pass is a
    number the gate and a hard ceiling would rest on train.py's word for.
    """
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        val_bpb_value = evaluate_bpb(model, tokenizer, EVAL_BATCH_SIZE)

    if _RESOLVED_TRAIN_BUDGET is _UNSET:
        raise RuntimeError(
            "no training dataloader was built through prepare.make_dataloader, so the training "
            "pool this run exposed is unknown. training_data_tokens_available is a pinned "
            "ceiling on every task; reporting a guess for it would make that pin unfailable"
        )

    params = count_params(model)
    steady = max(0, int(num_steps) - 10)
    tps = tokens_per_step * steady / training_seconds if training_seconds > 0 else 0.0

    metrics = {
        "metrics_schema_version": METRICS_SCHEMA_VERSION,
        "val_bpb": float(val_bpb_value),
        "eval_batch_size": EVAL_BATCH_SIZE,
        "num_params_total": params["total"],
        "num_params_breakdown": {k: v for k, v in params.items() if k != "total"},
        "flops_per_token_analytic": estimate_flops_analytic(model),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_vram_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "num_steps": int(num_steps),
        "tokens_per_step": int(tokens_per_step),
        # Summed by train.py as it consumed them, not num_steps * tokens_per_step:
        # that product is only right when every update used the same number of tokens.
        "total_tokens": int(total_tokens),
        "training_data_tokens_available": _RESOLVED_TRAIN_BUDGET,
        # Two different quantities, deliberately named apart. epochs_consumed is exact:
        # tokens consumed divided by the budget. The loader's own counter advances when the
        # budget's worth of tokens has been BUFFERED, and best-fit packing crops documents
        # that do not fit a row and discards the remainder, so the counter runs slightly
        # ahead of true passes. Read epochs_consumed for repetition; the counter only tells
        # you the loader restarted.
        "loader_passes_buffered_basis": int(final_epoch),
        "train_tokens_per_second": tps,
        "training_seconds": float(training_seconds),
        "total_seconds": float(total_seconds),
    }
    if _RESOLVED_TRAIN_BUDGET:
        metrics["epochs_consumed"] = metrics["total_tokens"] / _RESOLVED_TRAIN_BUDGET
    if flops_measured is not None:
        metrics["flops_per_token_measured"] = int(flops_measured)
    achieved = metrics.get("flops_per_token_measured") or metrics["flops_per_token_analytic"]
    metrics["mfu_percent_a100"] = 100.0 * achieved * tps / A100_BF16_PEAK_FLOPS
    metrics["mfu_percent_h100_denominator"] = 100.0 * achieved * tps / H100_BF16_PEAK_FLOPS
    if harness is not None:
        record_harness_result(metrics, **harness.result())

    print("=== METRICS ===")
    for key, value in metrics.items():
        if not isinstance(value, (dict, list)):
            print(f"{key}: {value!r}")
    print("METRICS_JSON: " + json.dumps(metrics, sort_keys=True, default=str))
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data and tokenizer for autoresearch")
    parser.add_argument("--num-shards", type=int, default=10, help="Number of training shards to download (-1 = all). Val shard is always pinned.")
    parser.add_argument("--download-workers", type=int, default=8, help="Number of parallel download workers")
    args = parser.parse_args()

    num_shards = MAX_SHARD if args.num_shards == -1 else args.num_shards

    print(f"Cache directory: {CACHE_DIR}")
    print()

    # Step 1: Download data
    download_data(num_shards, download_workers=args.download_workers)
    print()

    # Step 2: Train tokenizer
    train_tokenizer()
    print()
    print("Done! Ready to train.")
