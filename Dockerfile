# Claim triage API: multi-stage build for a slim runtime image.
#
#   builder  installs only the `serve` dependency group from uv.lock
#            (FastAPI, ONNX Runtime, tokenizers, Prometheus client, MLflow skinny)
#   runtime  copies that environment and the serving code. No pytest, optuna,
#            torch, transformers, pandas or scikit-learn.
#
# Models are not baked in: the API loads @champion from the MLflow registry
# (docker-compose.yml mounts it) or from a bundle folder (make bundle).

FROM python:3.11-slim AS builder
# git: uv fetches the shared data-ml-toolkit SDK from GitHub (builder stage only)
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /build
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-default-groups --group serve --no-install-project

FROM python:3.11-slim AS runtime
RUN useradd --create-home --uid 10001 app
COPY --from=builder /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    INFERENCE_LOG_PATH=/data/inference/inference_log.db
WORKDIR /app
COPY src/__init__.py src/
COPY src/common src/common
COPY src/features/__init__.py src/features/row.py src/features/
COPY src/serving src/serving
COPY configs configs
RUN mkdir -p /data/inference && chown -R app /data
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
