# CUDA 12.8 "base" image (driver shim + cuda-compat only, ~250 MB). All CUDA user-space
# libraries (cuBLAS, cuDNN 9, cuFFT, cuRAND, runtime) come from the nvidia-*-cu12 pip wheels
# and are registered with ldconfig below so that CTranslate2 and ONNX Runtime find them.
ARG BASE_IMAGE=nvidia/cuda:12.8.1-base-ubuntu24.04
# hadolint ignore=DL3006
FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.source="https://github.com/koekaverna/meetscribe-server"
LABEL org.opencontainers.image.licenses="MIT"

# hadolint ignore=DL3008
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ubuntu:24.04 ships user `ubuntu` (uid 1000); the shared hf-hub-cache volume is owned by uid 1000.
RUN useradd --create-home --shell /bin/bash --uid 1000 ubuntu || true
USER ubuntu
ENV HOME=/home/ubuntu \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_CACHE_DIR=/home/ubuntu/.cache/uv \
    UV_PYTHON_INSTALL_DIR=/home/ubuntu/.local/share/uv/python
WORKDIR $HOME/server

COPY --chown=ubuntu --from=ghcr.io/astral-sh/uv:0.9 /uv /bin/uv

# Dependencies first (cached layer), project afterwards.
RUN --mount=type=cache,target=/home/ubuntu/.cache/uv,uid=1000,gid=1000 \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev
COPY --chown=ubuntu . .
RUN --mount=type=cache,target=/home/ubuntu/.cache/uv,uid=1000,gid=1000 \
    uv sync --frozen --no-dev

# Pre-create the model cache mount point so the volume is writable by uid 1000.
RUN mkdir -p $HOME/.cache/huggingface/hub

# Register pip-installed CUDA libraries with the dynamic linker.
USER root
RUN find /home/ubuntu/server/.venv -maxdepth 7 -path "*/nvidia/*/lib" -type d \
    > /etc/ld.so.conf.d/venv-nvidia.conf && ldconfig
USER ubuntu

ENV PATH="$HOME/server/.venv/bin:$PATH" \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1
CMD ["uvicorn", "--factory", "meetscribe_server.main:create_app", "--host", "0.0.0.0", "--port", "8000"]
