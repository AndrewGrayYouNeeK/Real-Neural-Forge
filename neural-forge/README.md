# Neural Forge

A modular deep-learning training and inference framework built on PyTorch.  Neural Forge lets you
define models, configure training pipelines, and serve predictions through a REST API — all from
a single cohesive project.

---

## Project layout

```
neural-forge/
├── src/
│   ├── models/        # Network architectures (MLP, CNN, Transformer, …)
│   ├── data/          # Dataset classes and data-loading utilities
│   ├── training/      # Trainer, optimiser helpers, LR schedulers
│   ├── inference/     # Predictor and batch-inference pipeline
│   ├── evaluation/    # Metrics (accuracy, F1, AUROC, …) and reports
│   └── utils/         # Logging, seeding, checkpoint I/O, config helpers
├── api/               # FastAPI application and route handlers
├── configs/           # YAML experiment configuration files
├── docker/            # Dockerfile and docker-compose
├── tests/             # Unit and integration tests
├── pyproject.toml
├── README.md
└── run_training.sh
```

---

## Quick start

### 1 — Install

```bash
# Create and activate a virtual environment first
python -m venv .venv && source .venv/bin/activate

# Editable install with dev extras
pip install -e ".[dev]"
```

### 2 — Train a model

```bash
# Using the shell wrapper
bash run_training.sh --config configs/default.yaml

# Or call the module directly
python -m neural_forge.training.trainer --config configs/default.yaml
```

### 3 — Serve predictions

```bash
uvicorn neural_forge.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Then POST to `http://localhost:8000/predict` with a JSON body.

### 4 — Run tests

```bash
pytest
```

---

## Configuration

Experiment settings live in `configs/`.  The default configuration is
`configs/default.yaml`.  Override any value from the command line using
[OmegaConf](https://omegaconf.readthedocs.io/) dot-notation syntax:

```bash
python -m neural_forge.training.trainer \
    --config configs/default.yaml \
    training.epochs=50 \
    training.lr=3e-4
```

---

## Docker

```bash
cd docker
docker compose up --build
```

This starts the training service and the API server as separate containers.

---

## License

MIT
