# Real-Neural-Forge

A production-ready transformer pipeline for time-series prediction, built with PyTorch, CUDA, and FastAPI. This is a fully native stack — no Base44 SDK, no vendor-hosted backend.

## Features

- **Transformer encoder architecture** – positional encoding + stacked encoder layers
- **GPU-accelerated training & inference** – CUDA support via PyTorch; automatically falls back to CPU when no GPU is available
- **FastAPI REST API** – prediction, training, experiment tracking, and model metadata
- **Native web dashboard** – browser UI served directly from FastAPI at `/`
- **SQLite experiment store** – replaces Base44 entity persistence for runs and metrics
- **Modular Python package** – models, data, training, evaluation, inference, and storage layers
- **Docker & docker-compose support** – single `docker compose up --build` to get started
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
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Quick Start

### Docker (recommended)

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

The project includes a GitHub Actions workflow that runs tests, linting, type checks, coverage, and Docker image validation on every pull request.
