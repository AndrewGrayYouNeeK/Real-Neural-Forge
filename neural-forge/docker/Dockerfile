# ── Neural Forge — Training image ──────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (cached unless pyproject.toml changes)
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir -e ".[dev]"

# Copy source
COPY src/ ./src/
COPY configs/ ./configs/
COPY run_training.sh ./

RUN chmod +x run_training.sh

# ── API / Serve image ───────────────────────────────────────────────────────
FROM base AS api

COPY api/ ./api/

EXPOSE 8000

CMD ["uvicorn", "neural_forge.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

# ── Training image ───────────────────────────────────────────────────────────
FROM base AS train

ENV PYTHONPATH=/app/src

ENTRYPOINT ["bash", "run_training.sh"]
CMD ["--config", "configs/default.yaml"]
