# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.12.10
ARG UV_VERSION=0.9.26
ARG DEBIAN_MIRROR=https://mirrors.cloud.tencent.com

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

ARG DEBIAN_MIRROR
ARG PYTORCH_VARIANT=cpu
ARG PYTORCH_INDEX_BASE=https://download.pytorch.org/whl

ENV DEBIAN_FRONTEND=noninteractive \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_HTTP_TIMEOUT=300 \
    UV_HTTP_RETRIES=10 \
    UV_CONCURRENT_DOWNLOADS=4

RUN sed -i -E \
        -e "s|https?://deb\\.debian\\.org/debian-security|${DEBIAN_MIRROR}/debian-security|g" \
        -e "s|https?://deb\\.debian\\.org/debian|${DEBIAN_MIRROR}/debian|g" \
        /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install --yes --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# The frozen Linux environment is CPU-only by default. GPU builds deliberately
# replace only the PyTorch runtime packages from an official CUDA wheel index;
# all other application dependencies remain locked by uv.lock.
RUN --mount=type=cache,target=/root/.cache/uv \
    case "${PYTORCH_VARIANT}" in \
        cpu) \
            ;; \
        cu126|cu128|cu130) \
            uv pip install \
                --python /app/.venv/bin/python \
                --default-index "${PYTORCH_INDEX_BASE%/}/${PYTORCH_VARIANT}" \
                --reinstall-package torch \
                --reinstall-package torchvision \
                --reinstall-package torchaudio \
                "torch==2.10.0" \
                "torchvision==0.25.0" \
                "torchaudio==2.10.0" \
            ;; \
        *) \
            echo "Unsupported PYTORCH_VARIANT=${PYTORCH_VARIANT}" >&2; \
            exit 64 \
            ;; \
    esac

FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ARG DEBIAN_MIRROR
ARG APP_UID=10001
ARG APP_GID=10001

LABEL org.opencontainers.image.title="Knowledge Agent" \
    org.opencontainers.image.description="Tenant-aware RAG and Agent API" \
    org.opencontainers.image.vendor="Knowledge Agent" \
    org.opencontainers.image.version="0.1.0"

ENV DEBIAN_FRONTEND=noninteractive \
    HOME=/home/app \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    IMPORT_WORK_ROOT=/var/lib/knowledge-agent/imports \
    HF_HOME=/var/cache/knowledge-agent/huggingface \
    MODELSCOPE_CACHE=/var/cache/knowledge-agent/modelscope

RUN sed -i -E \
        -e "s|https?://deb\\.debian\\.org/debian-security|${DEBIAN_MIRROR}/debian-security|g" \
        -e "s|https?://deb\\.debian\\.org/debian|${DEBIAN_MIRROR}/debian|g" \
        /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libmagic1 \
        tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid "${APP_GID}" app \
    && useradd \
        --uid "${APP_UID}" \
        --gid "${APP_GID}" \
        --create-home \
        --home-dir /home/app \
        --shell /usr/sbin/nologin \
        app \
    && install -d -o "${APP_UID}" -g "${APP_GID}" \
        /app \
        /app/logs \
        /models/bge-m3 \
        /var/cache/knowledge-agent/huggingface \
        /var/cache/knowledge-agent/modelscope \
        /var/lib/knowledge-agent/imports

WORKDIR /app

COPY --from=builder --chown=${APP_UID}:${APP_GID} /app/.venv /app/.venv
COPY --chown=${APP_UID}:${APP_GID} app ./app
COPY --chown=${APP_UID}:${APP_GID} prompts ./prompts
COPY --chown=${APP_UID}:${APP_GID} scripts ./scripts
COPY deploy/docker/entrypoint.sh /usr/local/bin/container-entrypoint

RUN chmod 0555 /usr/local/bin/container-entrypoint

USER ${APP_UID}:${APP_GID}

EXPOSE 8000

STOPSIGNAL SIGTERM

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=3).read()"]

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/container-entrypoint"]

CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--no-server-header", "--timeout-keep-alive", "5", "--timeout-graceful-shutdown", "45"]
