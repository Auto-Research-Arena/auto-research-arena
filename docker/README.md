# AutoArena release execution image

This image is the common **GPU execution substrate**, not a preinstalled search
method. It preserves the CUDA environment beneath the image used for the benchmark
while removing unrelated robotics, video, editor, serving, and interactive-shell
stacks.

The image intentionally contains only:

- Ubuntu 22.04 on `linux/amd64`;
- CUDA 12.8.1 development libraries, cuDNN 9.8, and NCCL 2.25.1 from the
  digest-pinned NVIDIA base;
- Python 3.10 from Ubuntu, `uv` 0.11.6, Git, CA certificates, and the minimal
  native build toolchain needed by candidate code and isolated method setup;
- an unprivileged `autoarena` user and writable node-local cache locations.

It does **not** contain a dataset, model, checkpoint, research-model credential,
benchmark substrate checkout, or any method-specific environment. The two active
substrates have identical `pyproject.toml` and `uv.lock` files. Candidate execution
must create or reuse an environment from that immutable lock with:

```bash
uv sync --frozen --python 3.10
```

During measured launches, set `UV_FROZEN=1`; set `UV_OFFLINE=1` once the locked
packages and kernel artifacts have been hydrated. A method requiring another
runtime, such as the TPE controller's Python 3.12 environment, owns a separate
environment under runtime storage. It is not added globally to this image.

## Build

Build from the repository root so this Dockerfile remains independent of the
current working directory:

```bash
docker build \
  --file docker/Dockerfile \
  --build-arg AUTOARENA_UID="$(id -u)" \
  --build-arg AUTOARENA_GID="$(id -g)" \
  --tag autoarena-executor:release .
```

Both external stages are locked to their Linux/amd64 manifest digests in
[`image-lock.json`](image-lock.json). Updating a tag without updating and reviewing its
platform manifest is not a release rebuild. The first successful release build should
record the resulting image digest alongside this lock; this host has no Docker-compatible
builder, so that derived digest is not claimed here.

## Runtime contract

Run with the NVIDIA Container Runtime on an A100-80GB node. The harness assigns a
lane by narrowing `CUDA_VISIBLE_DEVICES`; the image only exposes the devices the
container runtime grants it.

Mount two distinct storage classes:

1. `/workspace` is the source/run-record location. It must survive for as long as
   its ledgers and reports are required.
2. `/workspace/.local` is writable runtime storage for dependency caches, the
   dataset, and per-launch temporary state. The image maps
   `$HOME/.cache/autoresearch` to `/workspace/data/autoresearch` and sets
   `UV_CACHE_DIR=/workspace/.local/cache/uv`.

For example:

```bash
docker run --rm --gpus all \
  --volume "$PWD:/workspace" \
  --volume "$PWD":/workspace \
  autoarena-executor:release
```

The host directory mounted at `/workspace/.local` must be writable by the configured
image UID/GID. Inject AWS and research-model credentials at runtime when a method
requires them; never add them to an image layer or build argument. Initial dataset
preparation needs access to the pinned Hugging Face objects unless the cache is
already populated.

## Why the image remains a CUDA development image

The immutable substrate currently resolves CUDA-enabled PyTorch and related
libraries from its lock file, but `train.py` is the benchmark's mutable research
surface. A candidate may compile a CUDA or C++ extension. Removing the CUDA
toolchain would silently narrow that surface and change which methods can
participate.

The smaller `nvidia/cuda:12.8.1-devel-ubuntu22.04` base might eventually replace
the cuDNN development base because the locked PyTorch environment also supplies
cuDNN. That is an execution-environment change, not cleanup: it requires a
reference equivalence run before the release lock may change.

## What was excluded

The inspected source image was about 23.18 GB compressed and also contained
Cosmos Predict, TensorFlow/Wan, vLLM/Qwen, OpenPI/RLDS, MuJoCo, image/video and GL
libraries, VS Code, custom Zsh tooling, GPU dashboards, object-store utilities,
RDMA/InfiniBand development packages, and privilege-escalation tooling. None is
imported by the benchmark substrate or required by the compute contract, so none
belongs in the standard release image.
