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
- **Forecast dashboard** – recent history, the next forecast (with a p10–p90 band when the API provides one), past predictions against actuals, and experiment metrics, served at `/`
- **SQLite experiment store** – replaces Base44 entity persistence for runs and metrics
- **Modular Python package** – models, data, training, evaluation, inference, and storage layers
- **Docker & GitHub Container Registry** – `docker compose up --build` locally, or pull `ghcr.io/andrewgrayyouneek/real-neural-forge`
- **Configurable via YAML** – all hyper-parameters in `config/config.yaml`

## Project Structure

```
.
├── config/
│   ├── config.yaml
│   ├── forecast.yaml
│   ├── lottery.yaml
│   └── weather.yaml
├── scripts/
│   ├── fetch_powerball.py
│   ├── fetch_weather.py
│   └── run_lottery_experiment.py
├── frontend/
│   ├── index.html
│   ├── app.js
│   ├── chart.js
│   └── styles.css
├── src/
│   ├── api.py
│   ├── compare.py
│   ├── dashboard_data.py
│   ├── model.py
│   ├── train.py
│   ├── models/
│   ├── data/
│   ├── training/
│   ├── evaluation/
│   ├── inference/
│   ├── retrain/
│   ├── serving/
│   ├── storage/
│   └── utils/
├── tests/
├── deploy/
│   └── cloudflare/          # Worker that fronts the container (not the dashboard Workers)
├── .github/
│   ├── dependabot.yml
│   └── workflows/
│       ├── ci.yml
│       ├── publish.yml
│       └── retrain.yml
├── Dockerfile
├── docker-compose.yml
├── docker-compose.ghcr.yml
├── fly.toml
├── render.yaml
├── requirements.in
├── requirements.txt
├── requirements-dev.in
├── requirements-dev.txt
├── workers_ci/
└── vercel.json
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

Leave `NEURAL_FORGE_API_TOKEN` unset to keep `POST /train` open. Set it (for example in the compose `environment` block) when that endpoint should require a bearer token.

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

Requires Python 3.10+ (CI tests 3.10–3.13).

`requirements.txt` and `requirements-dev.txt` are pinned locks. They install the CPU torch wheel from the PyTorch CPU index (no NVIDIA packages). A GPU machine can still run that build; it just will not use CUDA.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

# Train a model
python -m src.train --config config/config.yaml

# Start the API server
uvicorn src.api:app --reload
```

Open <http://localhost:8000> for the forecast dashboard or <http://localhost:8000/docs> for API docs.

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

api:
  training_enabled: true

storage:
  db_path: data/experiments.db

forecast:
  horizon: 1
  strategy: direct
```

Relative paths in the configuration — `training.checkpoint_dir`, `inference.checkpoint_path`, `storage.db_path`, and a CSV `data.path` — are resolved from the repository root. The default config path (`config/config.yaml`) is resolved the same way, so the server and `python -m src.train` work from any working directory. Absolute paths are used as given.

`storage.db_path` (default `data/experiments.db`) is the SQLite file used by both the API and the trainer.

`api.training_enabled` defaults to `true` when the `api` section is omitted. Set it to `false` to reject `POST /train` with `403`.

`POST /train` stays open when `NEURAL_FORGE_API_TOKEN` is unset, which is the localhost default. When that environment variable is a non-empty string, `POST /train` requires `Authorization: Bearer <token>` and otherwise returns `401`. The dashboard has an optional token field; it sends the header when a token is entered and shows a clear message on `401` or `403`. The token is not written into the YAML config, because training runs persist that file in the experiment store.

`data.val_split` (default `0.0` in code, `0.2` in the shipped config) holds out a fraction of samples. `data.split_mode` picks how: `random` (seeded by `data.split_seed`; the default for synthetic data) or `temporal` (the most recent windows; the default for CSV data, whose overlapping windows would otherwise leak into validation); when set, the best checkpoint is selected on validation loss and the final mse/rmse/mae are reported on the validation set.

`training.patience` stops training once the selection loss (validation loss when there is a split, otherwise train loss) has not improved for that many epochs, and `training.max_grad_norm` clips the gradient norm each step; both default to `0` (disabled).

For reproducible runs, set `data.seed` (seeds the synthetic dataset) and `training.seed` (seeds model initialisation, shuffling and dropout); both are `0` in the shipped config and unset (random) by default in code.

`dataset.id` names the series for scheduled retraining. It defaults to the file stem. The shipped `config/config.yaml` uses `default`. `dataset.enabled: false` or `retrain.enabled: false` skips that file. Any other `.yaml` or `.yml` under `config/` that has `model` and `training` sections is picked up with no code change.

`retrain.metric` is `mae`, `rmse`, or `mse` (default `mae`, lower is better). `retrain.baselines` defaults to `persistence` and `mean`. A positive integer `data.season_length` also adds a seasonal naive baseline. `retrain.val_fraction` (default `0.2`) is the latest-window holdout when `data.val_split` is `0`.

Fresh data is optional. Set `data.fetch` to a list of strings, a string split with `shlex` (no shell), or a mapping with `command` / `script` / `argv` and `timeout_seconds`. `data.fetch_script` and a top-level `fetch` key work the same way. If none of those is set, a conventional script is used when the file exists: `scripts/fetch_<id>.py`, `scripts/fetch_<file stem>.py`, `scripts/<id>_fetch.py`, `scripts/fetch_<id>.sh`, or the same names under `data/`. The stem of `data.path` is included, so `config/lottery.yaml` (`data/powerball.csv`) runs `scripts/fetch_powerball.py` and `config/weather.yaml` runs `scripts/fetch_weather.py`. `.py` scripts run with the current interpreter, with the repo root on `PYTHONPATH`.

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

Set `data.report_baselines: true` on a one-step CSV run (`forecast.horizon` 1 or omitted) with a temporal validation split to also score two reference forecasts on the same validation days, in original units:

- **Persistence** — tomorrow equals today (the previous observation).
- **Seasonal naive** — the same calendar day one year earlier. 29 February uses 28 February when the previous year is not a leap year. This requires `data.date_column`.

The logged model metrics are from the best checkpoint (lowest validation loss), which is the file written to `training.checkpoint_dir`. A horizon longer than 1 is rejected for this flag, because those reference rows are the next observation only. `python -m src.compare` scores multi-step horizons.

### Forecasting

`forecast.horizon` is how many future steps to predict (default `1`). `forecast.strategy` is `direct` (one multi-output head) or `recursive` (a one-step model rolled forward). With `horizon: 1` and `strategy: direct` and no `quantiles`, the network, checkpoints, and `POST /predict` response are unchanged.

`forecast.quantiles`, for example `[0.1, 0.5, 0.9]`, trains a non-crossing quantile head with pinball loss. The outer levels are the prediction interval. After training, conformalized quantile regression fits an offset on the earlier half of the validation rows (in time order) and `coverage` in the experiment metrics is the fraction of the later half that falls inside the adjusted interval. `raw_coverage` is the uncalibrated quantile interval on the full validation set. Neither number is fit on the training rows. They are omitted when there is no validation split, and `coverage` is omitted when the validation split is too small to hold out a report half.

Without quantiles, a validation split still fits a residual conformal interval around the point forecast (`interval_method: residual`) and records that same holdout coverage. `forecast.nominal_coverage` (default `0.8`, or the gap between the outer quantiles) is the coverage the offset targets.

`forecast.season_length` is the period, in steps, used by the in-window seasonal-naive baseline. For the bundled sine generator the period is `(seq_len + horizon - 1) / 2` samples (32 with the default window and a one-step horizon, 35 for `config/forecast.yaml`). On a daily CSV, leave it unset: the comparison then uses the same calendar day one year earlier when `data.date_column` is set and the horizon is 1.

`forecast.target_feature_index` tells recursive forecasts which input feature receives the prediction when `input_dim` is not `output_dim`. Other features are held at their last observed value. CSV configs infer the index from `feature_columns` and `target_column` when the target is one of the features. Set it explicitly to `null` to disable that inference.

`config/forecast.yaml` is a 7-step p10/p50/p90 example. Train it with:

```bash
python -m src.train --config config/forecast.yaml
```

### Model comparison

```bash
python -m src.compare --config config/config.yaml --epochs 5 --season-length 32
python -m src.compare --config config/forecast.yaml --epochs 5
python -m src.compare --config config/weather.yaml
```

The command fits persistence, seasonal naive (in-window when a season length fits the input window, otherwise the calendar baseline on a dated one-step CSV), ordinary least squares on target lags, a small LSTM, and the model named in the config. Every model is scored on the same temporal holdout — the last `data.val_split` fraction of windows, or the last 20% when that split is 0 — with mse, rmse, and mae in original units. `coverage` and `width` are the conformal interval on the later half of that holdout; the earlier half only chooses the offset. `--epochs` limits neural training for the comparison and does not change `training.epochs` for `python -m src.train`. The comparison scores the weights after that many epochs. It does not reload the best checkpoint and it does not write `checkpoints/best_model.pt`.

The bundled sine windows are drawn independently (a random phase per window). Their temporal split is the last generated windows, not one continuous series. The baselines still read only inside each window.

Results are written to the experiment store as one `comparison` run. Metrics are flat floats such as `persistence.mse` and `time_series_transformer.coverage`. `evaluation.lstm_hidden_size` (default 32), `evaluation.lstm_layers` (default 1), and `evaluation.epochs` override the LSTM baseline and the comparison's epoch budget.

## Daily weather series

`config/weather.yaml` trains on a real daily temperature record instead of the synthetic sine wave.

**Series.** Daily maximum temperature in degrees Celsius at NOAA GHCNd station `USW00094728` (New York Central Park), 1981-01-01 through 2025-12-31. The station record for this span is continuous (16,436 days).

**Source.** NOAA National Centers for Environmental Information, Daily Summaries access service: <https://www.ncei.noaa.gov/access/services/data/v1>. The underlying archive is the Global Historical Climatology Network–Daily (Menne et al., 2012, *Journal of Atmospheric and Oceanic Technology*).

**Licence.** Public domain. The series is a work of the United States government (17 U.S.C. § 105), so it can be copied and redistributed. The CSV is not committed; `data/` is gitignored.

Fetch it from the repository root, then train:

```bash
python -m scripts.fetch_weather
python -m src.train --config config/weather.yaml
```

`python -m src.data.weather` is the same download. The script writes `data/weather_central_park.csv` with columns `date`, `temperature_c`, `doy_sin`, and `doy_cos`. The calendar columns are `sin` and `cos` of `2π * day_of_year / 365.25`. They depend only on the date, so they are known before that day's temperature.

The weather config uses the CSV source, a temporal 20% validation holdout, z-score normalization fit on the training split only, early stopping (`patience: 5`), gradient clipping, and `training.seed: 0`. The held-out labels are the 3,281 days from 2017-01-07 through 2025-12-31. Checkpoints go to `checkpoints/weather/` (gitignored), not the API default `checkpoints/best_model.pt`. The model sees 28 days of temperature plus the calendar features and predicts the next day's maximum.

### Validation results

One CPU run of `config/weather.yaml` (seed 0, 67,265 parameters). Training stopped at epoch 17 because validation loss had not improved for 5 epochs. The best checkpoint is epoch 12. Wall time was 52.9 seconds.

Metrics are degrees Celsius on those 3,281 validation days, scored from that checkpoint:

| Method | RMSE (°C) | MAE (°C) |
| --- | ---: | ---: |
| Transformer | 3.7959 | 2.9376 |
| Persistence (tomorrow = today) | 4.3009 | 3.2516 |
| Seasonal naive (same calendar day last year) | 6.4337 | 5.0134 |

The transformer RMSE and MAE are lower than both references, so the model is learning the next-day maximum rather than fitting noise. Repeating yesterday is the tighter of the two references. This table is that first run; the hyperparameters in `config/weather.yaml` were not changed afterward.

## API Reference

### `GET /health`

Returns `{"status": "ok"}` when the service is running.

### `GET /model/info`

Returns architecture, parameter count, device, and checkpoint status.

On startup, a non-empty `NEURAL_FORGE_CHECKPOINT_URL` is downloaded before the model loads. `NEURAL_FORGE_CONFIG` selects the YAML file (default `config/config.yaml`). `checkpoint_loaded` is true only when this process actually loaded weights from a checkpoint. It stays false if the file was missing at load time, even if a checkpoint appears later (for example after training from the CLI while the server is running). `checkpoint_stale` is true when a checkpoint file on disk is newer than the weights in memory, or when a file is present that was never loaded. Restart the server, or finish a training run through `POST /train` (which reloads the model), to pick up that file. The OpenAPI `info.version` is the `[project].version` from `pyproject.toml`.

The payload also includes `horizon`, `forecast_strategy`, and `quantiles` for the loaded forecast settings.

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

Each step must contain `model.input_dim` finite numbers. A sequence longer than `model.max_seq_len` (512 in the shipped config), or any `NaN` or infinity, is rejected with `422`.

Optional fields:

```json
{
  "sequence": [[0.1], [0.2], [0.3], [0.4], [0.5]],
  "horizon": 7,
  "include_intervals": true
}
```

When `horizon` is omitted or `1` and `include_intervals` is false, the response is still `{"prediction": [0.612]}`. `prediction` is always the first-step point forecast (the quantile nearest 0.5 when a quantile head is loaded) and stays a flat list of floats.

When `horizon` is greater than 1, the response adds the full path. Intervals are included for a multi-step request, or for a one-step request when `include_intervals` is true, and only when the checkpoint can bound every requested step:

```json
{
  "prediction": [0.61],
  "forecast": [[0.61], [0.58]],
  "horizon": 2,
  "strategy": "direct",
  "lower": [[0.4], [0.22]],
  "upper": [[0.8], [0.91]],
  "p10": [0.4, 0.22],
  "p50": [0.61, 0.58],
  "p90": [0.8, 0.91],
  "quantiles": [0.1, 0.5, 0.9],
  "nominal_coverage": 0.8,
  "validation_coverage": 0.78,
  "interval_method": "cqr"
}
```

`validation_coverage` is present only when the requested horizon is the one measured at training time. `interval_method` is `cqr` (quantile bounds plus a conformal offset), `residual` (conformal absolute error around the point forecast), or `quantile` (raw quantile bounds, no offset yet).

`p50` is the point forecast for the first target channel, one value per step. `p10` and `p90` are that same channel's served interval endpoints: the outer quantiles when a quantile head is loaded, otherwise the conformal bounds. They are omitted on a one-step `POST /predict` that does not ask for intervals, so that response stays `{"prediction": [0.612]}`. The forecast dashboard reads these three lists.

A horizon longer than a direct head is rolled forward recursively when the prediction can be written back into the window (`input_dim == output_dim`, or `forecast.target_feature_index`). Otherwise the API returns `422`.

### `POST /forecast`

Same request body as `/predict`. `horizon` defaults to the configured forecast horizon instead of 1. The response always includes `horizon`, `strategy`, `prediction`, `forecast`, and `p50`, plus interval fields when the checkpoint has them.

### `GET /comparison`

Returns the latest comparison already stored by `python -m src.compare` for the loaded experiment database. It does not train a model. `models` is a list of rows with `name` and the stored metrics (`mae`, `rmse`, `mse`, and coverage when present). The list is empty when no comparison run has been stored.

### `POST /train`

Starts a background training job. Optional JSON body:

```json
{ "config_path": "config/config.yaml" }
```

`config_path` is resolved from the repository root and must be an existing `.yaml` or `.yml` file inside `config/`. Paths outside that directory (absolute paths, `..` traversal, symlinks that leave `config/`) are rejected before a job starts. The file must be a mapping with `model` and `training` sections.

When `NEURAL_FORGE_API_TOKEN` is set, send it as `Authorization: Bearer <token>`. A missing or wrong token returns `401` and does not start a job. When `api.training_enabled` is `false`, the endpoint returns `403` even if the token is valid. With the variable unset and training enabled, the endpoint behaves as before.

### `GET /training/status`

Returns the current background training state.

### `GET /experiments`

Lists recent training runs stored in the native SQLite experiment database.

### `GET /series/recent`

Read-only recent history for the **loaded** config, plus a one-step backtest of the loaded model.

`points` (default 96, 8–400) is how many of the latest target values to return. Synthetic data is one seeded series from `data.seed` (a fixed display seed when that is unset), not the full training set. A config with `data.path` is read as a table — `source: csv`, or another source name such as a weather or price file — using `feature_columns`, `target_column`, and `time_column` or `date_column` when set. A timestamp-like column (`timestamp`, `time`, `date`, `datetime`, or a name ending in `_date`) is used when neither is set.

The payload includes `history` (actuals and one-step `predicted` values), `forecast` (the next value; `p10` / `p50` / `p90` are null until a model provides them), `predict_window` (the feature window to POST to `/predict`), and `metrics` for the model against persistence and mean baselines on that window. `checkpoint_loaded` mirrors `/model/info`. A missing dataset file or a feature width that does not match the model returns `422`.

### `GET /experiments/{experiment_id}/predictions`

Read-only validation predictions for one run. `limit` (default 200, 1–2000) keeps the latest points.

When the run's `metrics` already contain `validation_predictions`, `predictions`, or parallel `actual` / `predicted` arrays (optionally with `p10`, `p50`, `p90`), those points are returned with `"source": "stored"`. A `comparison` or `model_comparison` object on the metrics is passed through. Otherwise the endpoint replays the saved config on the checkpoint file and returns `"source": "replay"`. The split matches training: temporal for `source: csv`, random otherwise, unless `split_mode` is set. If nothing is stored and the checkpoint cannot be replayed, the response is still `200` with `"available": false` and an empty `points` list. An unknown id is `404`.

The dashboard charts this payload, and it also plots multi-step forecasts, p10–p90 intervals, and model-comparison results when `/predict`, `/model/info`, or a forecast/comparison route includes them. Metric keys shaped like `baseline_persistence_mae` (written when `data.report_baselines` is set) are charted against the run's own MAE. Those fields are optional; the page still renders the one-step forecast and experiment metrics without them.

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
| Cloudflare Workers | `real-neural-forge` and `youneek-xyz` |

Tagged releases (`vX.Y.Z`) publish matching image tags after CI passes on that tag. Run the container on any host with Docker.

Cloudflare Workers builds this repository for `real-neural-forge` and `youneek-xyz`. Each build sets `WRANGLER_CI_OVERRIDE_NAME`. Installing the locked dependencies runs the editable `workers_ci` hook, which writes `wrangler.jsonc` for that Worker and points assets at `frontend/`. Installs that do not set the variable, including the Docker image build, leave the tree unchanged. Do not commit a `wrangler.jsonc` at the repository root, and do not point those two Workers Builds at `deploy/cloudflare/`. That directory is a different Worker, `real-neural-forge-api`, described below.

`vercel.json` stays so a leftover Vercel GitHub app cannot auto-deploy this repository. The running app is the container image above.

## Automatic retraining

`.github/workflows/retrain.yml` runs every Monday at 06:00 UTC and can be started from the Actions tab. For each enabled config it fetches fresh data when a fetch command or conventional script exists, retrains on CPU, and scores the new checkpoint on the latest temporal holdout. The checkpoint is promoted only when its metric is strictly better than the current champion and strictly better than every baseline. The first champion still has to beat the baselines. A run that loses, including one that errors, still writes `forecast-<id>.json` with that decision so a briefing can read it. `config/weather.yaml` and `config/lottery.yaml` are included with the shipped config. The Powerball series is a negative control and should lose to the mean baseline, so it stays unpromoted while its forecast file is still published.

Nothing large is committed. The rolling GitHub Release `champions` (a prerelease, and not the repository's latest software release) stores:

| Asset | What it is |
| --- | --- |
| `champion-<dataset_id>.pt` | Promoted weights |
| `forecast-<dataset_id>.json` | Latest forecast summary for that dataset |
| `forecasts-index.json` | One entry per dataset, for a daily briefing |
| `metrics-history.json` | Append-only promotion history (newest 400 rows per dataset) |

The release is anchored by a git tag named `champions`. That tag is not a software version. The first run creates it; later runs replace the assets in place, so the release has to stay mutable. The workflow token has `contents: write` for that. A ruleset that blocks tag creation has to allow this tag.

Public URLs, after the first successful run:

```text
https://github.com/AndrewGrayYouNeeK/Real-Neural-Forge/releases/download/champions/forecast-default.json
https://github.com/AndrewGrayYouNeeK/Real-Neural-Forge/releases/download/champions/forecasts-index.json
https://github.com/AndrewGrayYouNeeK/Real-Neural-Forge/releases/download/champions/champion-default.pt
https://github.com/AndrewGrayYouNeeK/Real-Neural-Forge/releases/download/champions/metrics-history.json
```

Replace `default` with the dataset id. The same JSON is uploaded as the `latest-forecasts` workflow artifact (kept 90 days). A private repository needs a token with contents read to download the release.

The scheduled job uses GitHub-hosted `ubuntu-latest`, stops at 60 minutes, and caps each dataset at 8 epochs and 10 minutes. On a private repository the free plan includes 2,000 minutes per month; a weekly run of this size stays inside that allowance. Raise the `max_epochs` input on a manual run when a dataset needs a longer fit. Manual `datasets` accepts `all` or a comma-separated list of ids or config paths.

The same command runs locally:

```bash
python -m src.retrain --all --max-epochs 8 \
  --release-dir release-assets --pull-release --publish
```

`--publish` needs the GitHub CLI authenticated with permission to write releases. Omit `--pull-release` and `--publish` to keep champions under `champions/` and forecasts under `forecasts/`. `--dry-run` prints the datasets and fetch commands as JSON and does not train.

Forecast JSON uses `schema_version` 1. `forecast.values` is the next model output (`forecast.horizon` is its length, so a later multi-step head shows up without a schema change). `forecast.as_of` is `beyond_last_observation` for CSV data and `last_window` for synthetic data. `promoted`, `decision`, `serving`, `candidate`, `champion`, and `baselines` describe the gate. `decision` is one of `promoted`, `kept_champion`, `rejected_baseline`, `champion_unevaluable`, or `error`.

## Live hosting

Workers cannot run PyTorch. The API container is a different Cloudflare product from the dashboard Workers. This repository does not deploy it. The steps below are what the owner runs, and they create paid resources.

### Cloudflare Containers

Plan: **Workers Paid** ($5 USD per month). Containers are not available on the free Workers plan. The included allowance and overages are on the [Containers pricing](https://developers.cloudflare.com/containers/platform/pricing/) page.

Instance: `standard-1` (1/2 vCPU, 4 GiB RAM, 8 GB disk) with `max_instances` 1. The CPU PyTorch image does not fit `lite` (256 MiB) or reliably fit `basic` (1 GiB). The container sleeps after 10 minutes with no requests, which stops compute charges until the next request. Memory is billed from the provisioned size while the instance is awake.

The Worker name is `real-neural-forge-api`. Leave the existing Workers Builds for `real-neural-forge` and `youneek-xyz` on the repository root so they keep publishing `frontend/`.

GHCR is not a registry Containers pulls from directly. Push the image CI already published into Cloudflare's registry, then deploy the Worker from `deploy/cloudflare/`:

```bash
docker pull ghcr.io/andrewgrayyouneek/real-neural-forge:latest
docker tag ghcr.io/andrewgrayyouneek/real-neural-forge:latest real-neural-forge:latest
cd deploy/cloudflare
npm install
npx wrangler login
npx wrangler containers push real-neural-forge:latest
```

`wrangler containers push` prints `registry.cloudflare.com/<ACCOUNT_ID>/real-neural-forge:latest`. Replace `REPLACE_ACCOUNT_ID` in `deploy/cloudflare/wrangler.jsonc` with that account id. To serve a dataset other than `default`, change `vars.NEURAL_FORGE_CONFIG` and `vars.NEURAL_FORGE_CHECKPOINT_URL` in that file. The URL pattern is the champion asset above.

Secrets are not in git. From `deploy/cloudflare/`:

```bash
npx wrangler secret put NEURAL_FORGE_API_TOKEN
npx wrangler secret put NEURAL_FORGE_CHECKPOINT_TOKEN
```

| Secret or variable | Required | Purpose |
| --- | --- | --- |
| `NEURAL_FORGE_API_TOKEN` | Yes on a public URL | Bearer token for `POST /train`. Use a long random string. If it is unset, that endpoint is open. |
| `NEURAL_FORGE_CHECKPOINT_TOKEN` | Only when the champions release is private | GitHub fine-grained PAT with contents read. Sent only to `github.com`, `api.github.com`, and `uploads.github.com`. Omit it for a public release. |
| `NEURAL_FORGE_CHECKPOINT_URL` | Before the first real deploy | Set in `wrangler.jsonc` `vars`, not as a secret. Startup fails when this is set and the download fails, so run **Retrain champions** once before the first deploy (or clear the URL to boot an untrained model). |
| `NEURAL_FORGE_CONFIG` | No | Config path inside the image. Default `config/config.yaml`. |
| `NEURAL_FORGE_CHECKPOINT_PATH` | No | Where the file is saved. Default is `inference.checkpoint_path`. |
| `NEURAL_FORGE_CHECKPOINT_MAX_BYTES` | No | Download cap. Default 256 MiB. |

Then:

```bash
npx wrangler deploy
```

The container needs outbound internet so it can download the champion. A restart picks up a newly promoted file. Changing a secret does not affect an already-running instance until it sleeps and starts again.

### Fly.io (one command after the token is filled in)

`fly.toml` runs the published GHCR image, listens on port 8000, checks `/health`, and stops the machine when it is idle. The VM is a shared CPU with 2 GB RAM, which is a paid Fly machine (the free allowance, if any remains on the account, may not cover it). App names are global; edit `app` in `fly.toml` when `real-neural-forge` is taken.

```bash
fly auth login
fly launch --copy-config --ha=false --yes --no-deploy \
  && fly secrets set \
       NEURAL_FORGE_API_TOKEN="replace-with-a-long-random-string" \
       NEURAL_FORGE_CHECKPOINT_URL="https://github.com/AndrewGrayYouNeeK/Real-Neural-Forge/releases/download/champions/champion-default.pt" \
       NEURAL_FORGE_CHECKPOINT_TOKEN="github-token-or-omit-this-line-when-the-release-is-public" \
  && fly deploy --ha=false
```

Omit `NEURAL_FORGE_CHECKPOINT_TOKEN` when the release is public. This block is the whole launch; it is not run by CI.

### Render

`render.yaml` is a blueprint for a paid Standard web service (2 GB RAM) using the same image and the same environment variables. `NEURAL_FORGE_API_TOKEN` and `NEURAL_FORGE_CHECKPOINT_TOKEN` are `sync: false`, so Render asks for them in the dashboard instead of reading them from git. Applying the blueprint is a manual step in the Render dashboard (**New Blueprint Instance**) or `render blueprint apply` after `render login`. CI does not apply it.

## Development

### Running Tests

With the virtualenv from *Local Development* active:

```bash
pytest            # full suite (~1 min on CPU; no GPU needed)
pytest --cov=src  # with coverage
```

### Code Quality

```bash
ruff check src/ tests/ workers_ci/
ruff format --check src/ tests/ workers_ci/
mypy src/ workers_ci/ --ignore-missing-imports
```

### CI/CD

- **CI** (`.github/workflows/ci.yml`) runs tests, linting, formatting, type checks, coverage, and a Docker image smoke test on Python 3.10–3.13 for every pull request and for pushes to `main` and `develop`.
- **Publish** (`.github/workflows/publish.yml`) pushes `:latest` to GitHub Container Registry only after that CI workflow succeeds for a push to `main`. Version tags (`vX.Y.Z`) and manual runs publish after the same CI workflow completes successfully inside the publish run.
- **Retrain** (`.github/workflows/retrain.yml`) fetches, retrains, and promotes champion checkpoints weekly. It also runs from the Actions tab. Forecast JSON and checkpoints are release assets, not commits.
- **Dependabot** (`.github/dependabot.yml`) opens weekly grouped PRs for GitHub Actions and the pip locks. Pip updates resolve with the Python in `.python-version` (3.10) so they stay installable on the whole CI matrix.
- **actionlint** lints `.github/workflows/` in CI.

## Lottery sanity check

This demo trains the time-series transformer on historical Powerball numbers and scores it against chance. A Powerball draw is a fresh sample from a fixed range, so an honest forecast names the next number about `1/26` of the time. The run is a check that the evaluation says so.

### Data and licence

Winning numbers come from New York State's open dataset **Lottery Powerball Winning Numbers: Beginning 2010** (`d6yy-54nr`), published by the New York State Gaming Commission:

- Page: <https://data.ny.gov/Government-Finance/Lottery-Powerball-Winning-Numbers-Beginning-2010/d6yy-54nr>
- API: `https://data.ny.gov/resource/d6yy-54nr.json`

Use of the file follows the [OPEN-NY Terms of Use](https://data.ny.gov/api/views/77gx-ii52/files/ef0c1840-ad54-4240-92fd-6397c49fde46?filename=OPEN-NY_20Terms_20of_20Use.pdf) (last modified March 8, 2013). Those terms grant a non-exclusive, revocable licence to use content published on data.ny.gov. They do not require attribution, share-alike, or pre-approval. The dataset view does not name a separate SPDX licence.

`python scripts/fetch_powerball.py` downloads the resource and writes `data/powerball.csv`. `data/` is gitignored, so the script is what makes the file reproducible. The CSV keeps draws from **2015-10-07** onward, the first draw of the current matrix (five white balls from 1–69, Powerball from 1–26). This pull skipped **592** earlier draws that used a different range and kept **1,422** draws, **2015-10-07** through **2026-10-19**.

The training target is the Powerball, the last of the six `winning_numbers` fields. White balls are written too, sorted ascending, so the file is complete. A sorted white-ball position is an order statistic, so the chance rate in this demo is the Powerball's **1/26**.

On this file the sample mean is **13.484** (a discrete uniform on 1–26 has mean 13.5) and the population standard deviation is **7.661** (uniform 7.5). The lag-1 correlation is **-0.067**, which accounts for **0.4%** of the variance. The Powerball equals the previous draw on **47/1,421** steps (**3.31%**), beside the **3.85%** chance rate.

### Training

`config/lottery.yaml` uses the shipped transformer (d_model 64, 4 heads, 3 encoder layers, feed-forward 256) on windows of 32 past Powerballs. The holdout is temporal, the CSV default: `split_mode: temporal` and `val_split: 0.2`. That is **1,112** training windows and **278** held-out draws, the most recent ones. Inputs and targets are z-scored with statistics from the training windows only. Reported MAE, RMSE, and MSE are in original Powerball units, after the inverse transform. Seeds 0, 1, and 2 fix initialisation and batch order.

Checkpoints are written to `checkpoints/lottery/seed-<n>/` and are gitignored. They do not replace `checkpoints/best_model.pt`. The experiment database is `data/lottery_experiments.db`, separate from the API store. The trainer keeps the checkpoint with the lowest scaled validation MSE, and the table below uses that same holdout, so the model's squared error is slightly optimistic.

```bash
python scripts/fetch_powerball.py
python scripts/run_lottery_experiment.py --seeds 0 1 2
```

The second command rewrites `results/lottery_sanity.json`. The figures in this section are that file for the 1,422-draw extract ending 2026-10-19. A later fetch appends new draws, and a rerun will move the counts inside sampling noise.

### Results

Chance on 278 draws is **10.7/278 (3.85%)**. The null standard error is **0.0115** (about 3.2 hits). Roughly ±2 standard errors is a band of about 4 to 17 hits. The three seeds share one holdout, so the z-scores repeat a single comparison.

| Predictor | MAE | RMSE | Exact match | z vs chance |
| --- | ---: | ---: | ---: | ---: |
| Transformer seed 0 | 6.789 | 7.865 | 13/278 (4.68%) | 0.72 |
| Transformer seed 1 | 6.835 | 7.869 | 9/278 (3.24%) | -0.53 |
| Transformer seed 2 | 6.825 | 7.867 | 13/278 (4.68%) | 0.72 |
| Historical mean | 6.870 | 7.897 | 17/278 (6.12%) | 1.97 |
| Persistence (last draw) | 9.471 | 11.622 | 6/278 (2.16%) | -1.46 |
| Uniform expected | 8.888 | 10.889 | 10.7/278 (3.85%) | 0.00 |
| Uniform sample (seed 0) | 8.295 | 10.348 | 10/278 (3.60%) | -0.22 |

Exact match rounds the forecast to the nearest integer, with halves rounded to even, then clips it to 1–26. The historical mean is the mean of every draw before the first held-out target (**13.559**). That value rounds to **14**, and a forecast that always says 14 hits **17** holdout draws. Persistence repeats the previous Powerball. "Uniform expected" is the average error of a fresh uniform draw from 1–26 on these targets. "Uniform sample" is one seeded draw of those guesses.

MSE skill against the historical mean is **0.83%**, **0.71%**, and **0.76%** on seeds 0, 1, and 2. Scaled validation loss at the chosen checkpoint is about **1.07**, the floor for a z-scored target whose conditional mean is flat. MAE and RMSE do beat persistence and a uniform guess: those baselines sit farther from the average, and squared error rewards a forecast that stays on the average. Early stopping ran **13**, **14**, and **7** epochs out of 20.

### Takeaway

The next Powerball is unpredictable. Rounded forecasts from three seeds hit the true number **9**, **13**, and **13** times out of 278, against **10.7** hits expected from a uniform 1–26 guess, and every rate sits inside two null standard errors of chance. The rounded historical mean, a forecast that always says 14, records more hits than the network. A lag-1 correlation of -0.067 leaves the transformer copying that mean. Lottery draws are independent for any purpose that needs the next number, and this evaluation reports that result.

## Monthly Powerball scorecard

The same draw file feeds a public log. On the 1st of each month a GitHub Actions workflow (`.github/workflows/powerball-monthly.yml`, 16:00 UTC, or a manual run) writes one ticket per strategy for every Monday, Wednesday, and Saturday in that month, then scores any logged draw whose winning numbers are now in the file. The job opens a pull request with `GITHUB_TOKEN`. It does not push to `main`, and it does not buy a ticket.

Logs live in [`predictions/powerball/`](predictions/powerball/). [`scorecard.json`](predictions/powerball/scorecard.json) is rebuilt from those logs. A month file is not rewritten after it exists, so a later result cannot change the ticket. The dashboard reads `GET /powerball/scorecard` and shows an empty card when the file is absent.

Each draw date carries seven tickets. Five white balls are unique and sorted, from 1–69, plus a Powerball from 1–26.

- **model** — the transformer, trained on the six numbers only.
- **model_weather** — the same model with draw-day weather on the dates already inside the input window.
- **most_hit**, **hot_pairs**, **neighbors**, **overdue** — counts, co-occurrence, one step off the last draw (the Powerball stays as drawn), and the longest gaps.
- **random** — one uniform ticket. The seed is `crc32("powerball-control-{month}-{YYYY-MM-DD}")`, so the control is public and stable.

Model and pattern tickets are the one forecast from data through the day before the month, copied onto every draw that month. Only the random ticket changes with the date. Weather on the draw being forecast is not an input. A gap of more than two days stays missing; those rows are dropped, not filled with zeros.

### Weather

Powerball draws are held in Tallahassee. The weather model uses NOAA NCEI Daily Summaries for GHCNd station **USW00093805** (Tallahassee airport): daily maximum temperature, minimum temperature, and precipitation, metric units. Central Park **USW00094728** maximum temperature is added only when every training draw has an observation within two days. Both series are U.S. government works (17 U.S.C. § 105). On the 1,422-draw file ending **2026-10-19**, Tallahassee had no gap over two days, and Central Park covered every draw, so the weather model saw ten inputs: the six numbers, three Tallahassee fields, and Central Park.

### Does weather change the forecast?

Weather leaves the Powerball forecast where the plain model already had it. `results/lottery_weather_holdout.json` trains both models on the same 1,422 weather-complete draws. The temporal holdout is the last **281** windows (1,125 fit, seed 0, at most 12 epochs). The checkpoint is the lowest scaled validation loss on that holdout, so the squared error is slightly optimistic, and the protocol is the same for both models.

| Forecast | Powerball MAE | Powerball hits | z vs 1/26 | Mean white matches |
| --- | ---: | ---: | ---: | ---: |
| Plain model | 6.836 | 18/281 (6.41%) | 2.23 | 0.356 |
| Weather model | 6.832 | 18/281 (6.41%) | 2.23 | 0.367 |
| Always 14 | 6.865 | 18/281 (6.41%) | 2.23 | — |
| Chance | — | 10.8/281 (3.85%) | 0.00 | 0.362 |

Weather minus plain is **−0.0048** MAE. The hit counts are equal. Mean white-ball matches sit on **25/69 = 0.362**. The holdout contains **18** Powerballs equal to 14, and the training mean **13.540** rounds to 14, so 18 hits is what that constant forecast scores. Neither network hit a full ticket. Draws do not depend on the weather in Tallahassee, and this comparison does not find that they do.

### First logged months

These files were written on **2026-10-21**. October is a backfill: its tickets use draws through **2026-09-30** only, even though October results were already public. November uses draws through **2026-10-19** (the latest row in the file; the 21 October draw was not published yet) and is the coming month. The October draw on the 21st and the four after it are still unscored. November has nothing to score.

The same model and pattern tickets are repeated for all 13 dates in the month. October:

| Strategy | Whites | Powerball |
| --- | --- | ---: |
| model | 12 23 35 48 59 | 13 |
| model_weather | 12 23 35 48 60 | 14 |
| most_hit | 21 23 28 61 64 | 14 |
| hot_pairs | 3 16 21 32 37 | 14 |
| neighbors | 5 7 24 34 45 | 13 |
| overdue | 1 22 34 43 51 | 19 |

November:

| Strategy | Whites | Powerball |
| --- | --- | ---: |
| model | 12 24 37 47 58 | 13 |
| model_weather | 12 23 36 47 59 | 13 |
| most_hit | 21 28 32 61 64 | 14 |
| hot_pairs | 3 16 21 32 37 | 14 |
| neighbors | 9 32 33 43 49 | 23 |
| overdue | 27 43 60 63 66 | 21 |

Eight October draws had results. Chance on eight tickets is **2.90** white-ball matches and **0.31** Powerball hits. The plain model scored 5 white matches and 0 Powerball hits. The weather model scored 4 and 1. The one weather hit is 3 October, Powerball 14, the $4 Powerball-only tier, and the most-hit and hot-pair tickets hit that same 14. The random control scored 5 white matches and 0 Powerball hits. Eight draws cannot separate any of these from chance. The scorecard says the same thing in [`predictions/powerball/scorecard.json`](predictions/powerball/scorecard.json).

```bash
python -m scripts.run_powerball_month --month 2026-10 --holdout-output results/lottery_weather_holdout.json
python -m scripts.run_powerball_month --month 2026-11 --no-fetch
```

The second command leaves an existing month file alone and rebuilds the scorecard. `config/lottery_tickets.yaml` is the model used for both tickets. Checkpoints from these runs are not written.
