# syntax=docker/dockerfile:1

# uv is copied from its release image instead of being installed by an
# unverified network script. This release selects and locks linux/amd64.
FROM --platform=linux/amd64 ghcr.io/astral-sh/uv:0.11.6@sha256:43cb71695fcad1516c2fbe0f56e500184c42d8bce838d9f64593b8aff2c16298 AS uvbin

# This is the exact linux/amd64 CUDA base beneath the execution image used to
# establish the benchmark. Keep the platform explicit: the benchmark hardware
# is an NVIDIA A100 and the task is not architecture-portable by declaration.
FROM --platform=linux/amd64 nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04@sha256:61f6c08f2b59036cb935e56d1e31a6b64e3ae2c7ddb86d33fa0b044c7917b719

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        pkg-config \
        python3 \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uvbin /uv /uvx /usr/local/bin/

ARG AUTOARENA_UID=1000
ARG AUTOARENA_GID=1000
RUN groupadd --gid "${AUTOARENA_GID}" autoarena \
    && useradd \
        --uid "${AUTOARENA_UID}" \
        --gid "${AUTOARENA_GID}" \
        --create-home \
        --shell /bin/bash \
        autoarena \
    && mkdir -p \
        /home/autoarena/.cache \
        /workspace/data/autoresearch \
        /workspace/.local/cache/uv \
        /workspace \
    && ln -s /workspace/data/autoresearch /home/autoarena/.cache/autoresearch \
    && chown -R autoarena:autoarena /home/autoarena /workspace

ENV HOME=/home/autoarena \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_VISIBLE_DEVICES=all \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_CACHE_DIR=/workspace/.local/cache/uv

WORKDIR /workspace
USER autoarena

CMD ["/bin/bash"]
