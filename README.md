# Real-Neural-Forge

[![CI](https://github.com/AndrewGrayYouNeeK/Real-Neural-Forge/actions/workflows/ci.yml/badge.svg)](https://github.com/AndrewGrayYouNeeK/Real-Neural-Forge/actions/workflows/ci.yml)
[![Publish image](https://github.com/AndrewGrayYouNeeK/Real-Neural-Forge/actions/workflows/publish.yml/badge.svg)](https://github.com/AndrewGrayYouNeeK/Real-Neural-Forge/actions/workflows/publish.yml)

A production-ready transformer pipeline for time-series prediction, built with PyTorch, CUDA, and FastAPI. Source of truth is this GitHub repository — no Base44 SDK, no Vercel, no vendor-hosted backend.

**GitHub:** <https://github.com/AndrewGrayYouNeeK/Real-Neural-Forge>  
**Image:** `ghcr.io/andrewgrayyouneek/real-neural-forge`

## Features

- **Transformer encoder architecture** – positional encoding + stacked encoder layers
- **GPU-accelerated training & inference** – CUDA support via PyTorch; automatically falls back to CPU when no GPU is available
- **FastAPI REST API** – prediction, training, experiment tracking, and model metadata
- **Native web dashboard** – browser UI served directly from FastAPI at `/`
- **SQLite experiment store** – replaces Base44 entity persistence for runs and metrics
- **Modular Python package** – models, data, training, evaluation, inference, and storage layers
- **Docker & GitHub Container Registry** – `docker compose up --build` locally, or pull `ghcr.io/andrewgrayyouneek/real-neural-forge`
- **Configurable via YAML** – all hyper-parameters in `config/config.yaml`

## Project Structure

```
.
├── config/
│   └── config.yaml
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── src/
│   ├── api.py
│   ├── model.py
│   ├── train.py
│   ├── models/
│   ├── data/
│   ├── training/
│   ├── evaluation/
│   ├── inference/
│   ├── storage/
│   └── utils/
├── tests/
├── .github/
│   ├── dependabot.yml
│   └── workflows/
│       ├── ci.yml
│       └── publish.yml
├── Dockerfile
├── docker-compose.yml
├── docker-compose.ghcr.yml
└── requirements.txt
```

## Quick Start

### Clone from GitHub

```bash
git clone https://github.com/AndrewGrayYouNeeK/Real-Neural-Forge.git
cd Real-Neural-Forge
```

### GitHub Container Registry (recommended)

After CI succeeds on `main`, the [Publish image](https://github.com/AndrewGrayYouNeeK/Real-Neural-Forge/actions/workflows/publish.yml) workflow pushes `:latest`:

```bash
docker pull ghcr.io/andrewgrayyouneek/real-neural-forge:latest
docker compose -f docker-compose.ghcr.yml up
```

The API and dashboard will be available at <http://localhost:8000>.

The first package published to GHCR is private by default. If `docker pull` returns `denied`, open the package on GitHub → **Package settings** → **Change visibility** → **Public**.

### Docker (build locally)

```bash
mkdir -p checkpoints data
docker compose up --build
```

The image installs CPU PyTorch from the official CPU wheel index, so the `linux/amd64` build stays free of the large NVIDIA CUDA wheels. For a GPU image, point `TORCH_INDEX_URL` at a CUDA index:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 docker compose build
```

The same override is the `TORCH_INDEX_URL` repository variable used by the publish workflow. Leave it unset to publish the CPU image.

The process runs as `forge` (uid 1000). The image creates `checkpoints/` and `data/` writable by that user. Create those directories on the host before `docker compose up` so the bind mounts are writable too; if they are owned by another user, `chown 1000:1000 checkpoints data`.

The image declares a `HEALTHCHECK` against `/health`, so `docker ps` shows the container as `healthy` once the API is up. `.dockerignore` keeps the virtualenv, caches, git history, local checkpoints, and `data/` (including a local experiments database) out of the build context. The pattern is `/data`, so the `src/data` package is still copied.

The API and dashboard will be available at <http://localhost:8000>.

### Local Development

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
# Optional: CPU-only PyTorch (much smaller download; skip if you have CUDA)
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt -r requirements-dev.txt

# Train a model
python -m src.train --config config/config.yaml

# Start the API server
uvicorn src.api:app --reload
```

Open <http://localhost:8000> for the dashboard or <http://localhost:8000/docs> for API docs.

## Configuration

Edit `config/config.yaml` to change model hyper-parameters, data source, training settings, or inference device:

```yaml
model:
  name: time_series_transformer
  input_dim: 1
  output_dim: 1
  d_model: 64
  nhead: 4
  num_encoder_layers: 3

data:
  source: synthetic
  n_samples: 1024
  seq_len: 64

training:
  experiment_name: default
  batch_size: 32
  learning_rate: 0.001
  epochs: 50
  device: cpu
  checkpoint_dir: checkpoints

inference:
  device: cpu
  checkpoint_path: checkpoints/best_model.pt
```

`data.val_split` (default `0.0` in code, `0.2` in the shipped config) holds out a fraction of samples. `data.split_mode` picks how: `random` (seeded by `data.split_seed`; the default for synthetic data) or `temporal` (the most recent windows; the default for CSV data, whose overlapping windows would otherwise leak into validation); when set, the best checkpoint is selected on validation loss and the final mse/rmse/mae are reported on the validation set.

`training.patience` stops training once the selection loss (validation loss when there is a split, otherwise train loss) has not improved for that many epochs, and `training.max_grad_norm` clips the gradient norm each step; both default to `0` (disabled).

For reproducible runs, set `data.seed` (seeds the synthetic dataset) and `training.seed` (seeds model initialisation, shuffling and dropout); both are `0` in the shipped config and unset (random) by default in code.

To train from CSV instead of synthetic data:

```yaml
data:
  source: csv
  path: data/timeseries.csv
  feature_columns: ["value"]
  target_column: value
  seq_len: 64
```

CSV data is standardised (per-feature z-score) by default; set `data.normalize: false` to disable it, or `true` to also scale synthetic data. The mean/std are fitted on the training split only, saved in the checkpoint, and applied automatically by `/predict`, so requests and responses stay in the CSV's original units. Final mse/rmse/mae are reported in original units; per-epoch `train_loss`/`val_loss` are in scaled units.

## API Reference

### `GET /health`

Returns `{"status": "ok"}` when the service is running.

### `GET /model/info`

Returns architecture, parameter count, device, and checkpoint status.

### `POST /predict`

**Request body**

```json
{
  "sequence": [[0.1], [0.2], [0.3], [0.4], [0.5]]
}
```

**Response**

```json
{
  "prediction": [0.612]
}
```

### `POST /train`

Starts a background training job. Optional JSON body:

```json
{ "config_path": "config/config.yaml" }
```

`config_path` is resolved from the repository root and must be an existing `.yaml` or `.yml` file inside `config/`. Paths outside that directory (absolute paths, `..` traversal, symlinks that leave `config/`) are rejected before a job starts. The file must be a mapping with `model` and `training` sections.

### `GET /training/status`

Returns the current background training state.

### `GET /experiments`

Lists recent training runs stored in the native SQLite experiment database.

### YouNeeK Time

YouNeeK Time is **100 units / 100 minutes / 100 seconds** (not App Store 10/10/10). Noon is `50:00:00`. The year is 354 days with 10-day weeks, epoch `2026-01-01`. Lunar phase uses the mean synodic month (`00:00:00` new, `50:00:00` full).

- `GET /youneek/now` — current YouNeeK clocks
- `GET /youneek/convert?at=<ISO-8601>` — convert a Gregorian timestamp
- `GET /youneek/forecast/next-minute` — next YouNeeK minute boundary
- `GET /youneek/calendar` — year, 10-day week, year clock
- `GET /youneek/lunar` — lunar YouNeeK clock

## Deployment

GitHub is the home for this project:

| Surface | Where |
| --- | --- |
| Source, issues, pull requests | [github.com/AndrewGrayYouNeeK/Real-Neural-Forge](https://github.com/AndrewGrayYouNeeK/Real-Neural-Forge) |
| CI (test, lint, typecheck, Docker build) | [GitHub Actions CI](https://github.com/AndrewGrayYouNeeK/Real-Neural-Forge/actions/workflows/ci.yml) |
| Published image | `ghcr.io/andrewgrayyouneek/real-neural-forge` |
| Run the published image | `docker compose -f docker-compose.ghcr.yml up` |

Tagged releases (`vX.Y.Z`) publish matching image tags after CI passes on that tag. Run the container on any host with Docker; no Vercel project is required.

`vercel.json` remains only to disable leftover Vercel GitHub auto-deployments. If Vercel **or Cloudflare Workers** status checks still appear on pull requests, disconnect those apps:

1. Open GitHub → **Settings** → **Integrations** → **Applications**
2. For **Vercel**, click **Configure**, then remove **Real-Neural-Forge** from the repository list
3. For **Cloudflare Workers**, click **Configure**, then remove **Real-Neural-Forge** (this repo is a FastAPI/Docker app, not a Worker)
4. In the [Vercel dashboard](https://vercel.com/dashboard), delete any linked `real-neural-forge` projects
5. On the GitHub repo, clear **About → Website** if it still points at `*.vercel.app`

## Development

### Running Tests

With the virtualenv from *Local Development* active:

```bash
pytest            # full suite (~1 min on CPU; no GPU needed)
pytest --cov=src  # with coverage
```

### Code Quality

```bash
ruff check src/ tests/
mypy src/ --ignore-missing-imports
```

### CI/CD

- **CI** (`.github/workflows/ci.yml`) runs tests, linting, type checks, coverage, and a Docker image smoke test on every pull request and on pushes to `main` and `develop`.
- **Publish** (`.github/workflows/publish.yml`) pushes `:latest` to GitHub Container Registry only after that CI workflow succeeds for a push to `main`. Version tags (`vX.Y.Z`) and manual runs publish after the same CI workflow completes successfully inside the publish run.
- **Dependabot** (`.github/dependabot.yml`) opens weekly PRs for GitHub Actions and pip updates.
