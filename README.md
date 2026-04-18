# Real-Neural-Forge

A production-ready transformer pipeline for time-series prediction, built with PyTorch, CUDA, and FastAPI.

## Features

- **Transformer encoder architecture** – positional encoding + stacked encoder layers
- **GPU-accelerated training & inference** – CUDA support via PyTorch; automatically falls back to CPU when no GPU is available
- **FastAPI REST endpoint** – `POST /predict` for inference, `GET /health` for liveness
- **Docker & docker-compose support** – single `docker compose up --build` to get started
- **Configurable via YAML** – all hyper-parameters in `config/config.yaml`

## Project Structure

```
.
├── config/
│   └── config.yaml        # YAML configuration (model, training, inference)
├── src/
│   ├── model.py           # TimeSeriesTransformer + PositionalEncoding
│   ├── train.py           # Training loop & checkpoint saving
│   └── api.py             # FastAPI application
├── tests/
│   ├── test_model.py
│   ├── test_api.py
│   └── test_train.py
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Quick Start

### Docker (recommended)

```bash
docker compose up --build
```

The API will be available at <http://localhost:8000>.

### Local Development

```bash
pip install -r requirements.txt

# (Optional) train a model first
python -m src.train --config config/config.yaml

# Start the API server
uvicorn src.api:app --reload
```

## Configuration

Edit `config/config.yaml` to change model hyper-parameters, training settings, or
the inference device:

```yaml
model:
  input_dim: 1        # number of input features per time step
  output_dim: 1       # number of output values per prediction
  d_model: 64         # transformer embedding dimension
  nhead: 4            # number of attention heads
  num_encoder_layers: 3
  dim_feedforward: 256
  dropout: 0.1
  max_seq_len: 512

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 50
  device: cpu         # change to "cuda" to use GPU
  checkpoint_dir: checkpoints

inference:
  device: cpu
  checkpoint_path: checkpoints/best_model.pt
```

## API Reference

### `GET /health`

Returns `{"status": "ok"}` when the service is running.

### `POST /predict`

**Request body**

```json
{
  "sequence": [[0.1], [0.2], [0.3], [0.4], [0.5]]
}
```

`sequence` is a 2-D list of shape `[seq_len, input_dim]`.

**Response**

```json
{
  "prediction": [0.612]
}
```

Interactive docs are available at <http://localhost:8000/docs>.

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run tests with coverage
pytest --cov=src --cov-report=html
```

### Code Quality

This project uses automated code quality tools:

```bash
# Install pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install

# Run linting
ruff check src/ tests/

# Auto-fix linting issues
ruff check --fix src/ tests/

# Run type checking
mypy src/
```

### CI/CD

The project includes a GitHub Actions workflow that automatically:
- Runs tests on Python 3.10, 3.11, and 3.12
- Performs linting with Ruff
- Type checks with mypy
- Generates test coverage reports
- Builds and tests the Docker image

All pull requests are automatically checked against these quality standards.