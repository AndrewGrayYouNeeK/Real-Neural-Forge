FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/AndrewGrayYouNeeK/Real-Neural-Forge"
LABEL org.opencontainers.image.url="https://github.com/AndrewGrayYouNeeK/Real-Neural-Forge"
LABEL org.opencontainers.image.title="Real Neural Forge"
LABEL org.opencontainers.image.description="Transformer pipeline for time-series prediction (PyTorch + FastAPI)"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp1 is the OpenMP runtime the CPU torch wheel links against.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 forge \
    && useradd --uid 1000 --gid forge --create-home --no-log-init forge

WORKDIR /app

# CPU wheels by default, so linux/amd64 does not pull NVIDIA CUDA libraries.
# GPU image: docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 .
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

# workers_ci is an editable lockfile entry, so the directory has to be in the
# image before pip runs. The hook writes wrangler.jsonc only when
# WRANGLER_CI_OVERRIDE_NAME is set; a normal image build leaves the tree unchanged.
COPY requirements.txt .
COPY workers_ci ./workers_ci
RUN torch_req="$(awk '/^torch([^A-Za-z0-9_]|$)/ { print; exit }' requirements.txt)" \
    && test -n "${torch_req}" \
    && awk '!/^torch([^A-Za-z0-9_]|$)/' requirements.txt > /tmp/requirements-notorch.txt \
    && pip install --no-cache-dir --index-url "${TORCH_INDEX_URL}" "${torch_req}" \
    && pip install --no-cache-dir -r /tmp/requirements-notorch.txt \
    && rm /tmp/requirements-notorch.txt \
    && python -c "import os, torch; idx=os.environ['TORCH_INDEX_URL'].rstrip('/'); cuda=torch.version.cuda; print(torch.__version__, cuda); raise SystemExit(1 if idx.endswith('/cpu') and cuda else 0)" \
    && test ! -e /app/wrangler.jsonc

COPY --chown=forge:forge . .

RUN mkdir -p /app/checkpoints /app/data \
    && chown forge:forge /app/checkpoints /app/data

USER forge

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
