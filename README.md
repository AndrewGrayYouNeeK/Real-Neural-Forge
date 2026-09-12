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

After the [Publish image](https://github.com/AndrewGrayYouNeeK/Real-Neural-Forge/actions/workflows/publish.yml) workflow has run on `main`:

```bash
docker pull ghcr.io/andrewgrayyouneek/real-neural-forge:latest
docker compose -f docker-compose.ghcr.yml up
```

The API and dashboard will be available at <http://localhost:8000>.

The first package published to GHCR is private by default. If `docker pull` returns `denied`, open the package on GitHub → **Package settings** → **Change visibility** → **Public**.

### Docker (build locally)

```bash
docker compose up --build
```

The API and dashboard will be available at <http://localhost:8000>.

### Local Development

```bash
pip install -r requirements.txt

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

To train from CSV instead of synthetic data:

```yaml
data:
  source: csv
  path: data/timeseries.csv
  feature_columns: ["value"]
  target_column: value
  seq_len: 64
```

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

Starts a background training job using the configured YAML file.

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

Tagged releases (`vX.Y.Z`) also publish matching image tags. Run the container on any host with Docker; no Vercel project is required.

`vercel.json` remains only to disable leftover Vercel GitHub auto-deployments. If Vercel status checks still appear on pull requests, remove the integration:

1. Open GitHub → **Settings** → **Integrations** → **Applications** → **Vercel**
2. Click **Configure**, then remove **Real-Neural-Forge** from the repository list
3. In the [Vercel dashboard](https://vercel.com/dashboard), delete any linked `real-neural-forge` projects
4. On the GitHub repo, clear **About → Website** if it still points at `*.vercel.app`

## Development

### Running Tests

```bash
pip install -r requirements-dev.txt
pytest
```

### Code Quality

```bash
ruff check src/ tests/
mypy src/
```

### CI/CD

- **CI** (`.github/workflows/ci.yml`) runs tests, linting, type checks, coverage, and a Docker image smoke test on every pull request.
- **Publish** (`.github/workflows/publish.yml`) pushes the image to GitHub Container Registry on every push to `main` and on version tags.
- **Dependabot** (`.github/dependabot.yml`) opens weekly PRs for GitHub Actions and pip updates.
