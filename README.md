# Real-Neural-Forge

[![CI](https://github.com/AndrewGrayYouNeeK/Real-Neural-Forge/actions/workflows/ci.yml/badge.svg)](https://github.com/AndrewGrayYouNeeK/Real-Neural-Forge/actions/workflows/ci.yml)
[![Publish image](https://github.com/AndrewGrayYouNeeK/Real-Neural-Forge/actions/workflows/publish.yml/badge.svg)](https://github.com/AndrewGrayYouNeeK/Real-Neural-Forge/actions/workflows/publish.yml)

A production-ready transformer pipeline for time-series prediction, built with PyTorch, CUDA, and FastAPI. Source of truth is this GitHub repository — no Vercel, no vendor-hosted backend.

**GitHub:** <https://github.com/AndrewGrayYouNeeK/Real-Neural-Forge>  
**Image:** `ghcr.io/andrewgrayyouneek/real-neural-forge`

## Features

- **Transformer encoder architecture** – positional encoding + stacked encoder layers
- **GPU-accelerated training & inference** – CUDA support via PyTorch; automatically falls back to CPU when no GPU is available
- **FastAPI REST API** – prediction, training, experiment tracking, and model metadata
- **Forecast dashboard** – recent history, the next forecast (with a p10–p90 band when the API provides one), past predictions against actuals, and experiment metrics, served at `/`
- **SQLite experiment store** – local persistence for training runs and metrics
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

After CI succeeds on `main`, the [Publish image](https://github.com/AndrewGrayYouNeeK/Real-Neural-Forge/actions/workflows/publish.yml) workflow pushes `:latest`, `sha-<sha>`, and `main-<sha>` (the first 7 characters of that commit):

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

`GET /space/near-earth` and `GET /space/solar-system` read event files under `data/space/`. Those files are not in the build context. The image bakes the current snapshot from `space_snapshot/` into `/app/data/space/`, and the same files stay at `/app/space_snapshot/` so a `./data` bind mount can be filled on startup when those files are missing. The API then refreshes them in the background and once a day. A failed fetch leaves the previous files in place, which are the baked snapshot until a refresh succeeds. Set `NEURAL_FORGE_SPACE_REFRESH=0` to serve the snapshot only.

The Powerball scorecard (`predictions/powerball/scorecard.json`), the forecast JSON under `results/` (including the near-Earth and solar-system outlooks), and the default dashboard series are already in the image. The default series is synthetic (`config/config.yaml`), so it does not need a CSV from `data/`. `GET /comparison` stays empty until a comparison experiment is stored in the runtime database. That database is created on startup and is not a published dataset.

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

`data.val_start` and `data.test_start` (`YYYY-MM-DD`) hold out label dates instead of a fraction, so a different `seq_len` still scores the same days. With only `test_start`, every earlier label is training and validation is empty. `data.missing_indicators: true` adds a 0/1 column for each feature that is missing in the fitting rows and fills the holes with that block's median. `data.calendar_features` (`doy`, `dow`, `month`) and `data.future_calendar` add the same encodings from the row date or the label date.

The transformer accepts `patch_len` (default 1; patches keep the latest observations), `use_revin` (default false), and `residual` (default false; adds the last target observation to the head). `training.weight_decay` above 0 selects AdamW. `training.checkpoint_rule: last` writes the final epoch instead of the best selection loss.

`data.blend_with` (`climatology`, `anomaly_ar`, `ridge`, or `gradient_boosting`) averages the network with that classical forecast. `data.blend_weight` is the network's share (default 0.5) and is not fit on the scored days. Those classical models train only on rows before the first scored label. On a validation run the blend is part of the logged model metrics; a refit with an empty validation split leaves the blend for `python -m src.evaluation.holdout`.

`python -m src.training.search --family neural` (12 trials, at most 20 epochs, patience 5) or `--family boosting` (20 trials) minimizes the validation RMSE of the config you pass. Point the config's `val_start` at an inner slice when the decision split has to stay untouched. `python -m src.evaluation.holdout --split test` scores a saved checkpoint and does not run during training.

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

`python -m src.data.weather` is the same download. The script writes `data/weather_central_park.csv` with `date`, `temperature_c` (TMAX), `tmin_c`, `prcp_mm`, `snow_mm`, `snwd_mm`, `awnd_ms` (metres per second), `doy_sin`, and `doy_cos`. A trace in the water-equivalent fields is stored as 0. Snow depth and wind are missing on part of the record and stay missing in the file; the trainer adds a missingness indicator for each and fills holes with the median of the fitting rows. The calendar columns are `sin` and `cos` of `2π * day_of_year / 365.25`.

`config/weather.yaml` predicts the next day's maximum from 28 days of those columns. Validation labels are 2009-10-31 through 2017-01-06 (2,625 days). The 3,281 days from 2017-01-07 through 2025-12-31 are the test split. Training does not score them. Checkpoints go to `checkpoints/weather/` (gitignored), separate from the API default `checkpoints/best_model.pt`.

### How a change was allowed to stay

Every candidate was scored on the 2009–2016 validation days in degrees Celsius. A change stayed only when a circular block bootstrap (blocks of 14 days, 1,000 resamples, seed 0) put the 5th percentile of its RMSE drop above zero. The point estimate was not enough. The 2017–2025 days were not used for any of these choices.

The search that proposed hyperparameters used an earlier slice, labels 2004-01-31 through 2009-10-30, so the decision days stayed out of the objective. Neural search: 12 trials, at most 20 epochs, patience 5, TPE sampler seed 0, pruning a trial that was worse than the running median after 6 scored epochs. The architecture was frozen to the residual network below (RevIN and patching had already lost). Boosting search: 20 trials of `HistGradientBoostingRegressor` on the same slice. Both searches ran on CPU. Re-run them with `python -m src.training.search --config <inner.yaml> --family neural` or `--family boosting`.

### Decision-validation results

| Step | RMSE | MAE | RMSE drop | Bootstrap 5% | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Persistence (tomorrow = today) | 4.0750 | 3.0968 |  |  | reference |
| Seasonal naive (same calendar day last year) | 6.5683 | 5.1566 |  |  | reference |
| Day-of-year climatology | 4.6240 | 3.6402 |  |  | reference |
| Anomaly autoregression | 3.6272 | 2.8240 |  |  | reference |
| Ridge regression on the anomaly | 3.4786 | 2.7278 |  |  | reference |
| Calendar-only transformer (temperature, day-of-year) | 3.6881 | 2.8649 |  |  | starting point |
| Multivariate transformer | 3.4564 | 2.7137 | +0.2317 | +0.1753 | kept |
| Gradient boosting, same covariates | 3.3831 | 2.6564 | +0.0733 | +0.0352 | kept |
| Residual multivariate transformer | 3.3277 | 2.5950 | +0.0553 | +0.0131 | kept |
| RevIN on that residual network | 3.3669 | 2.6314 | −0.0391 | −0.0839 | dropped |
| Residual network plus the label date | 3.3397 | 2.6251 | −0.0120 | −0.0359 | dropped |
| Patched context (84 days, patch length 7) with RevIN | 3.4027 | 2.6785 | −0.0750 | −0.1227 | dropped |
| Five-seed ensemble of the residual network | 3.3170 | 2.5979 | +0.0107 | −0.0054 | dropped |
| Equal blend of the residual network and gradient boosting | 3.3007 | 2.5819 | +0.0271 | +0.0043 | kept |
| Boosting hyperparameters from the 20-trial search | 3.4132 | 2.6733 | −0.0301 | −0.0497 | dropped |
| Equal blend using that tuned boosting model | 3.3180 |  | −0.0174 | −0.0274 | dropped |
| Network from the 12-trial search, refit on this split | 3.3405 | 2.6175 | −0.0128 | −0.0367 | dropped |
| Equal blend using that tuned network | 3.3187 |  | −0.0180 | −0.0303 | dropped |

The RMSE drop is against the previous kept forecast (the multivariate transformer against the calendar-only transformer, boosting against that multivariate network, the residual network against boosting, and the blend against the residual network). The two tuned blends are compared with the equal blend of 3.3007. Positive means a lower RMSE.

The shipped recipe is that equal blend. The network is a residual transformer (64-wide, 4 heads, 2 layers, dropout 0.1, context 28, seed 0, 67,713 parameters) on maximum temperature, minimum temperature, precipitation, snowfall, snow depth, wind speed, and the two day-of-year features, with missingness indicators for snow depth and wind. The gradient-boosting partner uses the library's small default grid, chosen on the last 20% of the fitting labels and then refit on all of them. The blend weight is 0.5, chosen before this split was scored. RevIN, patching, a longer context, the label-date features, the five-seed ensemble, and both hyperparameter searches stay in the code with the defaults that lost left off, so another dataset can turn them on.

The inner-slice winners that failed to transfer were: boosting `max_depth=2`, `learning_rate=0.192`, `max_iter=300`, `min_samples_leaf=40`, `l2_regularization=0.0213` (inner RMSE 3.4553); and a network with `d_model=128`, `nhead=2`, one layer, `dim_feedforward=256`, `dropout=0.111`, `seq_len=14`, `learning_rate=2.83e-4`, `weight_decay=1.12e-4` (inner RMSE 3.4061).

A GPU was not used. The wider network from the search scored 3.3405 against the residual network's 3.3277 on this split, so extra capacity did not show up as accuracy. A T4-class GPU would shorten the same search (these runs are about a minute of CPU per full training, and the 12-trial search took about 15 minutes) at a spot price of roughly $0.10–$0.50 per hour.

### Held-out test

One refit, then one score. `config/weather_refit.yaml` trains the residual network for 17 epochs on every label before 2017-01-07. Seventeen is the epoch with the best decision-validation loss. `checkpoint_rule: last` and `patience: 0` keep that epoch, with no early stop on the test days. The same 0.5 blend is applied by `python -m src.evaluation.holdout --config config/weather_refit.yaml --split test`. The refit took 49.9 seconds on CPU.

| Method | RMSE (°C) | MAE (°C) |
| --- | ---: | ---: |
| Equal blend (the shipped forecast) | 3.4599 | 2.7083 |
| Gradient boosting | 3.4998 | 2.7420 |
| Ridge regression on the anomaly | 3.5888 | 2.8169 |
| Anomaly autoregression | 3.7804 | 2.9313 |
| Persistence | 4.3009 | 3.2516 |
| Day-of-year climatology | 4.6016 | 3.5916 |
| Seasonal naive | 6.4337 | 5.0134 |

The blend was chosen on 2009–2016. The other rows are references on the same 3,281 days, not candidates. Persistence and the seasonal naive match the previously published figures for this window (4.3009 and 6.4337). An earlier transformer RMSE of 3.7959 on these days was early-stopped on the test window itself, so it is not a comparable score.

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

Returns the current background training state. Each change is also written to `data/training_status.json` (the same `data/` directory as the experiment database; gitignored, and mounted by Compose). After a restart, a completed or failed run is restored from that file. A file left at `running` means the process stopped mid-job, so startup reports `failed` instead of a live run.

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

### `GET /space/near-earth`

Upcoming objects that pass close to Earth, plus the saved monthly forecasts. `days` defaults to **730** and must be an integer from 1 to 730. The window is inclusive: UTC today through today plus `days`. Shortening `days` filters the approach list and the JPL month bars. It does not change the model path.

The list is JPL CAD approaches to Earth and the Moon with `dist <= 0.05` AU, and any IAU nI interstellar approach out to 1 AU. CAD times are TDB calendar dates, not UTC. `forecasts` holds the 24-month recursive outlook for each trained series, with `p10`, `p50`, and `p90`. A missing cache or score file still returns `200` with `"available": false` and empty lists. `731` returns `422`.

The dashboard panel "What's coming close to Earth" reads this route. The day menu is 90, 180, 365, or 730.

### `GET /space/solar-system`

Upcoming events for the rest of the solar system, plus the saved monthly forecasts. `days` defaults to **730** and must be an integer from 1 to 730. The window is inclusive: UTC today through today plus `days`.

The payload lists JPL CAD approaches to Mercury, Venus, Mars, Jupiter, Saturn, Uranus, Neptune, and Pluto, comet perihelia stepped from SBDB, and Horizons elongation extrema (oppositions, conjunctions, and greatest elongations). Mercury, Venus, and Mars are limited to 0.05 AU. Jupiter is limited to 1 AU. The other four bodies are queried at 10 AU and are not trained. `forecasts` holds the 24-month outlook for each trained series. Distance series include a Horizons value on each outlook month and `physics_outlook_mae` against that ephemeris. A missing cache or score file still returns `200` with `"available": false`. `showers.available` is false. `731` returns `422`.

The dashboard panel "The rest of the solar system" reads this route. The day menu is 90, 180, 365, or 730.

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

Workers cannot run PyTorch. The API container is a different Cloudflare product from the dashboard Workers. CI does not deploy it. The owner runs the redeploy steps below, and they use paid resources.

The live API is <https://real-neural-forge-api.youneekartifacts.workers.dev>.

### Cloudflare Containers

Plan: **Workers Paid** ($5 USD per month). Containers are not available on the free Workers plan. The included allowance and overages are on the [Containers pricing](https://developers.cloudflare.com/containers/platform/pricing/) page.

Instance: `standard-1` (1/2 vCPU, 4 GiB RAM, 8 GB disk) with `max_instances` 1. The CPU PyTorch image does not fit `lite` (256 MiB) or reliably fit `basic` (1 GiB). The container sleeps after 10 minutes with no requests, which stops compute charges until the next request. Memory is billed from the provisioned size while the instance is awake. A `POST /train` that starts a job, or returns 409 because one is already running, arms a schedule inside the container Durable Object. Every three minutes that schedule renews the activity timeout and reads `GET /training/status`. It keeps scheduling while the status is `running`, so the instance stays awake for the whole run, then the 10-minute idle timeout applies again.

The Worker name is `real-neural-forge-api`. Leave the existing Workers Builds for `real-neural-forge` and `youneek-xyz` on the repository root so they keep publishing `frontend/`.

GHCR is not a registry Containers pulls from directly. The Worker config pins one image tag. It does not follow `:latest`. The pin in `deploy/cloudflare/wrangler.jsonc` is `registry.cloudflare.com/d74b0de637f4d9d9eb3efb9e13142784/real-neural-forge:main-6ce3b6f`. Account id `d74b0de637f4d9d9eb3efb9e13142784` is the Cloudflare account that owns the Worker. The `migrations` entry `v1` created the SQLite Durable Object class `NeuralForgeContainer`. Leave that entry in the file. Add a new tag only when that class's storage changes. Do not add a Durable Object `exports` entry beside it. Current Wrangler rejects `migrations` and `exports` together. The class is still exported from `deploy/cloudflare/src/index.ts`, and the container binds it by `class_name`.

To serve a dataset other than `default`, change `vars.NEURAL_FORGE_CONFIG` and `vars.NEURAL_FORGE_CHECKPOINT_URL` in that file. The URL pattern is the champion asset above.

### Redeploy

Wait until [Publish image](https://github.com/AndrewGrayYouNeeK/Real-Neural-Forge/actions/workflows/publish.yml) succeeds for the commit you want. On a push to `main` it pushes three GHCR tags for that commit: `:latest`, `sha-<sha>`, and `main-<sha>`. `<sha>` is the first 7 characters of the commit. Tagged releases and manual runs push the same commit tags.

Copy that image into Cloudflare's registry and point the Worker at the new tag. From a machine that can pull GHCR and is logged into the same Cloudflare account:

```bash
SHA=6ce3b6f
docker pull ghcr.io/andrewgrayyouneek/real-neural-forge:main-$SHA
docker tag ghcr.io/andrewgrayyouneek/real-neural-forge:main-$SHA real-neural-forge:main-$SHA
cd deploy/cloudflare
npm install
npx wrangler login
npx wrangler containers push real-neural-forge:main-$SHA
```

`wrangler containers push` prints `registry.cloudflare.com/d74b0de637f4d9d9eb3efb9e13142784/real-neural-forge:main-<sha>`. Set `containers[0].image` in `deploy/cloudflare/wrangler.jsonc` to that string and commit the pin. Do not set the tag to `latest`. Then, still in `deploy/cloudflare/`:

```bash
npx wrangler deploy
```

A restart of the container picks up a newly promoted champion. Changing a secret does not affect an already-running instance until it sleeps and starts again.

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

The container needs outbound internet so it can download the champion.

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
- **Publish** (`.github/workflows/publish.yml`) pushes `:latest`, `sha-<sha>`, and `main-<sha>` to GitHub Container Registry only after that CI workflow succeeds for a push to `main`. `<sha>` is the first 7 characters of the commit. Version tags (`vX.Y.Z`) and manual runs publish those commit tags after the same CI workflow completes successfully inside the publish run. The Cloudflare Worker pins `main-<sha>` and does not follow `:latest`.
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

## Space

Outer-space series for the same time-series transformer. Every forecast is one step ahead: the next day, the next month, or the next year. The label day is not an input. Metrics below are in the original units, from the checkpoint with the lowest scaled validation loss. Seed is 0. Device is CPU. This run took **251.1 seconds** in total.

Checkpoints go to `checkpoints/space/<id>/` and the experiment rows go to `data/space/experiments.db`. Both are gitignored, and neither replaces `checkpoints/best_model.pt`. The CSVs under `data/space/` are gitignored too. The score file that matches this section is `results/space_forecast.json`. The four SVG charts are `results/space/charts/` (each series drawn at 480 points).

Fetch, then train, from the repository root:

```bash
python -m scripts.fetch_space
python -m scripts.train_space
```

`python -m src.space.fetch` and `python -m src.space.train` are the same commands. `--refresh` re-downloads the raw files. `--only f107,kp` trains a subset. A later fetch moves the end dates forward and will change the holdout.

The solar, launch, exoplanet, and catalog close-approach configs set `retrain.enabled: false`. The weekly champion job trains every enabled file under `config/`. Those series stay out of that job: one fetch script builds all of them, and the CSVs are not in git. The near-Earth group and the solar-system group below are the exceptions: those configs are enabled and refresh through `python -m scripts.fetch_near_earth` or `python -m scripts.fetch_solar_system`.

Configs live in `config/space/`. Each one uses a temporal 20% holdout, z-score normalization fit on the training split only, early stopping, gradient clipping at 1.0, and `training.seed: 0`. Daily models see 27 days (one solar rotation) and may train for 20 epochs with patience 5. Monthly models see 24 months and may train for 40 epochs with patience 6. The univariate network has 67,137 parameters (d_model 64, 4 heads, 2 layers). The exoplanet model is smaller (2,273 parameters, 5-year windows, 80 epochs, patience 10) because the series has 34 points.

`data.report_baselines` scores persistence (tomorrow equals today) and, when the CSV has a `date` column, the calendar seasonal naive (the same month-day one year earlier, or the same month one year earlier). Feb 29 uses Feb 28. Two further lags are scored for the solar series: **27 days**, and **4018 days** (`round(11 × 365.25)`, one Schwabe cycle). Monthly flare counts use **132 months**. Launches, close approaches, and NEO discoveries use a drift baseline instead of those solar lags: yesterday plus the mean step fit only on rows before the first validation label. The yearly exoplanet file has no `date` column, because a one-year lag on 1 January dates is the same forecast as persistence.

### Sources and licences

The SILSO sunspot number is a column in the CelesTrak file and is not read. Another batch trains that series.

| Series | What is counted | Source | Licence |
| --- | --- | --- | --- |
| Daily Kp and Ap | Max of the eight 3-hour Kp values, and the daily Ap | GFZ Kp (Matzka et al. 2021, doi:10.5880/Kp.0001), taken from the CelesTrak redistribution [`SW-All.csv`](https://celestrak.org/SpaceData/SW-All.csv) because the GFZ host did not answer here. Format notes: [SpaceWx-format.php](https://celestrak.org/SpaceData/SpaceWx-format.php). The direct file is [`Kp_ap_Ap_SN_F107_since_1932.txt`](https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_Ap_SN_F107_since_1932.txt). | GFZ Helmholtz Centre for Geosciences, CC BY 4.0. Cite GFZ and CelesTrak. |
| Daily F10.7 | Observed 10.7 cm flux, sfu | Same CelesTrak file, column `F10.7_OBS`. Measurements: Natural Resources Canada, Dominion Radio Astrophysical Observatory (Penticton). | U.S. government redistribution via NOAA SWPC is public domain. Cite Penticton and CelesTrak. |
| Solar-wind speed and density, Dst | Daily mean speed (km/s), density (n/cm³), and Dst (nT) | NASA OMNI2 daily file [`omni_01_av.dat`](https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni_01_av.dat). Layout: [`omni2.text`](https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2.text). | Public domain, U.S. government work. Cite King and Papitashvili, NASA/GSFC. Dst in this file is the daily mean of the Kyoto WDC index, not the daily minimum. |
| Flare counts | Monthly B, C, M, and X class counts | NOAA NCEI GOES XRS reports, 1975 through [`goes-xrs-report_2017-ytd.txt`](https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/solar-flares/x-rays/goes/xrs/). The directory has no later year file. | Public domain, U.S. government work. |
| Close approaches | Monthly approaches within 0.05 AU | JPL SBDB Close-Approach Data API, [`cad.api`](https://ssd-api.jpl.nasa.gov/cad.api) | Public domain, U.S. government work. |
| NEO discoveries | Monthly counts by `first_obs` | JPL SBDB Query API, [`sbdb_query.api`](https://ssd-api.jpl.nasa.gov/sbdb_query.api), classes APO, AMO, ATE, IEO | Public domain, U.S. government work. |
| Orbital launch attempts | Monthly attempts | Jonathan McDowell, GCAT [`launch.tsv`](https://planet4589.org/space/gcat/tsv/launch/launch.tsv). Column notes: [lcols.html](https://planet4589.org/space/gcat/web/launch/lcols.html). | Creative Commons CC-BY. Cite "data from GCAT (J. McDowell, planet4589.org/space/gcat)". |
| Exoplanet discoveries | Planets per discovery year | NASA Exoplanet Archive TAP, table `pscomppars`, [`TAP/sync`](https://exoplanetarchive.ipac.caltech.edu/TAP/sync) | Free to use. Cite the NASA Exoplanet Archive. |

### What each file contains

The fetch on **2026-10-21** wrote these spans. The open day, month, or year is dropped, and a trailing event-month that does not reach the last day of the month is dropped too.

- **Kp.** 25,222 days, 1957-10-01 through 2026-10-20, no gaps filled. Kp in the file is in tenths and is divided by 10. The target is the day's maximum 3-hour value. Daily Ap is stored as `ap` and is not a training target. 3,777 days (15.0%) have Kp ≥ 5.0. A day with Kp 5− (4.7) is quiet under that cutoff. NOAA's G1 scale starts at Kp 5−, so these storm counts are a stricter subset of G1 days.
- **F10.7.** The same 25,222 days. Rows flagged `OBS` or `INT` with flux above 0 are kept (60 days are the file's own `INT` interpolations). `PRD` and `PRM` predictions are dropped. Seven days in the full record exceed 400 sfu. The largest is 938.6 on 2011-03-07, a flare-contaminated Penticton observation, and it sits before the validation window.
- **Solar wind and Dst.** Speed and density need at least 4 plasma hours and are not fill values. The contiguous suffix is 11,663 days, 1994-11-07 through 2026-10-12, with 87 days linearly filled where the hole was at most 5 days. Dst is kept on its own calendar: 23,300 days, 1963-01-01 through 2026-10-16, with no filled days.
- **Flares.** 501 months, 1975-09 through 2017-05. Totals: 22,776 B, 47,800 C, 6,335 M, 469 X. Empty months inside the span are zeros. The trained targets are the C, M, and X counts.
- **Close approaches.** 1,521 months, 1900-01 through 2026-09, 33,304 approaches within 0.05 AU. The count includes approaches, before discovery, of objects that are in the catalog now, so the rise is mostly catalog growth.
- **NEO discoveries.** 1,596 months, 1893-10 through 2026-09, 42,564 distinct designations. `first_obs` is the start of the orbit arc. The monthly count mostly tracks survey effort.
- **Launches.** 828 months, 1957-10 through 2026-09, 7,414 attempts. `LaunchCode` prefix `O` (orbital) or `D` (deep space) is kept, including failures. Missile, suborbital, atmospheric, test, and near-orbital rows are excluded. A trailing `?` on a vague GCAT time is ignored and the numeric day is kept.
- **Exoplanets.** 34 years, 1992 through 2025, 6,090 planets. Missing years are zeros. 2026 is dropped as an open year. Discovery year 2014 has 872 planets and 2016 has 1,504, which are catalog releases (Kepler) rather than a smooth discovery rate.

### Validation results

Cells are **MAE / RMSE**. The epoch column is the best epoch and how many epochs ran before early stopping.

Daily solar indices. The 27-day, 11-year, and 1-year columns are seasonal-naive lags.

| Series | Validation | n | Train | Epoch | Model | Persistence | 27-day | 11-year | 1-year |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| F10.7 (sfu) | 2013-01-03 to 2026-10-20 | 5039 | 40.3s | 4/9 | 4.4375 / 8.7429 | 4.5310 / 9.9623 | 14.8019 / 22.9756 | 25.5354 / 42.6028 | 26.2382 / 38.0930 |
| Kp maximum | 2013-01-03 to 2026-10-20 | 5039 | 44.3s | 5/10 | 0.8609 / 1.1298 | 0.9939 / 1.3398 | 1.2221 / 1.6140 | 1.5278 / 1.9504 | 1.4491 / 1.8406 |
| Dst daily mean (nT) | 2014-01-19 to 2026-10-16 | 4654 | 35.0s | 3/8 | 7.4785 / 12.4289 | 8.4175 / 14.1839 | 12.6887 / 20.5044 | 16.0408 / 23.0616 | 15.6186 / 22.5513 |
| Solar-wind speed (km/s) | 2020-05-30 to 2026-10-12 | 2327 | 42.6s | 15/20 | 38.7088 / 53.7716 | 45.8569 / 62.8073 | 73.8582 / 99.5875 | 96.8580 / 126.7139 | 95.7589 / 123.7986 |
| Solar-wind density (n/cm³) | 2020-05-30 to 2026-10-12 | 2327 | 41.2s | 16/20 | 2.3546 / 3.3495 | 2.8200 / 4.2597 | 3.3446 / 4.7822 | 3.9494 / 5.4243 | 4.0230 / 5.5678 |

On these five series the transformer has lower MAE and lower RMSE than persistence, the 27-day rotation, the 11-year lag, and the calendar year. The F10.7 margin over persistence is 0.09 sfu of MAE. Yesterday's flux already carries the rotation and the cycle, which is why those longer lags lose by a wide margin on a one-day horizon. Pearson correlation of the F10.7 forecast with the holdout is 0.980. For Kp it is 0.550.

Monthly and yearly counts. The last column is the 132-month solar cycle for flares, and the drift baseline for the other rows. The exoplanet row has no 12-month score.

| Series | Validation | n | Train | Epoch | Model | Persistence | 12-month | Cycle or drift |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| C-class flares | 2009-07 to 2017-05 | 95 | 2.1s | 14/20 | 31.9908 / 42.8771 | 37.2526 / 49.8964 | 57.8105 / 75.3519 | 79.8737 / 97.9434 |
| M-class flares | 2009-07 to 2017-05 | 95 | 1.2s | 5/11 | 5.8292 / 9.1097 | 6.9895 / 10.8773 | 8.0105 / 12.7076 | 11.8842 / 17.0377 |
| X-class flares | 2009-07 to 2017-05 | 95 | 1.1s | 5/11 | 0.7531 / 1.0898 | 0.6947 / 1.4510 | 0.7789 / 1.4582 | 1.2316 / 2.1692 |
| Launch attempts | 2013-06 to 2026-09 | 160 | 1.7s | 4/10 | 4.8192 / 6.6532 | 4.0188 / 5.1448 | 3.5000 / 4.4441 | 4.0188 / 5.1446 |
| Close approaches | 2001-11 to 2026-09 | 299 | 2.9s | 3/9 | 51.2750 / 74.9283 | 18.5217 / 28.0782 | 13.7458 / 20.3365 | 18.5219 / 28.0781 |
| NEO discoveries | 2000-08 to 2026-09 | 314 | 4.9s | 9/15 | 110.2570 / 144.1762 | 39.0096 / 58.0285 | 33.2866 / 46.0913 | 39.0098 / 58.0284 |
| Exoplanet discoveries | 2021 to 2025 | 5 | 0.2s | 13/23 | 76.2774 / 109.2830 | 129.6000 / 175.3910 | same as persistence | 134.5714 / 175.4733 |

C-class and M-class monthly counts are the two flare series where the transformer beats persistence and both seasonal lags. The holdout is the late GOES XRS catalog (cycle 24), and the catalog itself ends in May 2017. X-class months are sparse (median 0 in the full record). Persistence has the lower MAE (0.6947 against 0.7531). The transformer has the lower RMSE (1.0898 against 1.4510).

Launch attempts, close approaches, and NEO discoveries are the series where a naive forecast wins. For launches the same month a year earlier has MAE 3.50, persistence 4.02, and the transformer 4.82. The validation window is the years when the monthly rate climbed, and last year's count tracks that climb. Drift matches persistence because the mean historical step is about zero next to the month-to-month scatter. Close approaches and discoveries rise because the catalog and the surveys grow. The 12-month lag has the lowest error on both. The discovery forecast is anti-correlated with the holdout (Pearson −0.37): the network does not follow the survey-effort trend.

The exoplanet comparison is five years, 2021–2025, after the Kepler release spikes of 2014 (872) and 2016 (1,504). On those five points the transformer MAE is 76.3 and persistence is 129.6. That is a short trend check on a catalog-release process. It is the whole yearly record the archive returned, and it is too short to treat as a settled forecast skill.

### Storm days (Kp ≥ 5)

A storm day is a day with maximum Kp ≥ 5.0. The alarm fires when the forecast itself is ≥ 5.0. On the Kp holdout (2013-01-03 to 2026-10-20) there are **432 storm days out of 5,039** (8.6%).

| Forecast | Alarms | Hits | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Transformer | 61 | 36 | 0.5902 | 0.0833 | 0.1460 |
| Persistence | 432 | 150 | 0.3472 | 0.3472 | 0.3472 |
| 27-day lag | 429 | 88 | 0.2051 | 0.2037 | 0.2044 |
| 11-year lag | 479 | 41 | 0.0856 | 0.0949 | 0.0900 |

Persistence is the better storm detector. The transformer raises 61 alarms and catches 36 of the 432 storm days. When it does alarm, precision is 0.59, and the missed storms dominate the F1. The chart in `results/space/charts/kp.svg` shows the forecast sitting under the Kp 5 line through most of the window. Regression skill (MAE 0.86 against persistence 0.99) comes from tracking the quiet-day level, which is where most days are.

### Does solar wind help the Kp forecast?

`config/space/kp_solar_wind.yaml` predicts the same daily Kp maximum from the previous 27 days of Kp, speed, and density. `config/space/kp_omni_window.yaml` is the paired run: Kp history only, on the same OMNI dates, the same 27-day window, and the same holdout (2020-05-30 to 2026-10-12, 2,327 days, 236 storm days). Speed and density from the label day are not inputs.

| Forecast | Train | MAE | RMSE | Alarms | Hits | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Kp history only | 21.2s, epoch 5/10 | 0.8860 | 1.1798 | 11 | 6 | 0.5455 | 0.0254 | 0.0486 |
| Kp, speed, and density | 12.4s, epoch 1/6 | 0.8753 | 1.1548 | 40 | 26 | 0.6500 | 0.1102 | 0.1884 |
| Persistence |  | 1.0339 | 1.3987 | 236 | 85 | 0.3602 | 0.3602 | 0.3602 |

Adding speed and density lowers MAE by 0.011 Kp and raises storm-day recall from 6/236 to 26/236. Both models beat persistence on MAE and RMSE. Persistence still catches more storm days (85 hits, F1 0.36 against 0.19). The wind features help a little on this split. They do not turn the regression model into a reliable storm alarm. The Kp-only model trained on the longer 1957–2026 record (the first storm table) is a different window, so its MAE of 0.8609 is not the comparison to use here.

### What's coming close to Earth

This group is separate from the catalog close-approach row above. That older series is one column, approaches within 0.05 AU, and it stays out of the weekly job. The files here train Earth and Moon counts, a known-at-the-time control, a per-1,000-known-NEO rate, closest-miss distance, large-object counts (`H < 22`, about 140 m), fireball counts and impact energy, and CelesTrak decay counts. The score file is `results/near_earth_forecast.json`. This run took **43.3 seconds** on CPU, seed 0, on 2026-10-22.

```bash
python -m scripts.fetch_near_earth
python -m src.space.train --results results/near_earth_forecast.json --package \
  --only earth_within_1ld,earth_within_0_05au,earth_closest_au,earth_large_h22,earth_known_0_05au,earth_0_05au_per_1000,moon_within_1ld,moon_within_0_05au,moon_closest_au,moon_large_h22,fireball_count,fireball_energy,reentries
```

`GET /space/near-earth?days=` defaults to **730** (today through two years, inclusive). The approach list follows `days`. Each model is still a one-step network, scored on a temporal 20% holdout against persistence, the same month a year earlier, and the mean of every observation before the first validation label. The 24-month path is that one-step model rolled forward. `p10` and `p90` are residual quantiles (nominal 0.1 and 0.9) fit on the earlier half of the validation origins that still contain that lead. Coverage in the table is the later half of that same validation split. Early stopping already used the validation loss, so the coverage is not a second untouched test. Counts, energy, and the rate are clipped at 0. Closest distance is clipped to `[0, 0.05]` AU because a month with no approach inside the search is stored as 0.05 AU with a censor flag. The flag is not a model input.

Other planets, comet perihelia, discovery counts by orbit class, and Horizons distances are the next section.

#### Sources

No API key. Raw files stay under `data/space/` and are gitignored. The fetch on 2026-10-22 used `date-max=2028-10-21`.

| Series | What is counted | Source | Licence |
| --- | --- | --- | --- |
| Earth and Moon approaches | Monthly counts within 1 LD (384,400 km) and within 0.05 AU, closest miss, `H < 22` | JPL SBDB Close-Approach Data API, [`cad.api`](https://ssd-api.jpl.nasa.gov/cad.api), `body=Earth` and `body=Moon`, `fullname=true` | Public domain, U.S. government work. CAD times are TDB. |
| Known-at-the-time control | Same approaches, kept only when SBDB `first_obs` is on or before the approach | JPL SBDB Query API, classes APO, AMO, ATE, IEO, plus `sb-kind=c` | Public domain, U.S. government work. |
| Fireballs | Monthly count and summed impact energy (kt TNT). Radiated energy is stored in units of 10^10 J and is not a training target | [`fireball.api`](https://ssd-api.jpl.nasa.gov/fireball.api) | Public domain, U.S. government work. Sensor-network detections, not a complete flux. |
| Sentry | Current impact-risk snapshot, not a trained series | [`sentry.api`](https://ssd-api.jpl.nasa.gov/sentry.api) | Public domain, U.S. government work. |
| Interstellar flag | IAU nI objects resolved from SBDB (`1I`, `2I`, `3I`, then stop). Hyperbolic comets and asteroids without an I designation are not flagged | [`sbdb.api?sstr=`](https://ssd-api.jpl.nasa.gov/sbdb.api) | Public domain, U.S. government work. |
| Reentries | Monthly count of SATCAT `DECAY_DATE` | CelesTrak [`satcat.csv`](https://celestrak.org/pub/satcat.csv) | Cite CelesTrak (T.S. Kelso). Decays are recorded after they happen. Space-Track TIP is not used. |

The fetch wrote 1,521 months, 1900-01 through 2026-09 (the open month and every future CAD row stay out of the training CSV): **33,307** Earth approaches and **6,313** Moon approaches within 0.05 AU. Every Earth designation in that CAD pull matched an SBDB `pdes`. Since 2020, 61% of Earth approaches had `first_obs` on or before the flyby and 39% were added after the object was found. That second group is the survey backfill. Fireballs: 1,073 events, 1988-04 through 2026-09, 1,207 kt summed impact energy. Reentries: 35,652 decay months from 1957-12 through 2026-09. Sentry listed 2,223 objects; the highest cumulative Palermo was 29075 (1950 DA), −0.98. Interstellar objects on the fetch day were 1I/'Oumuamua, 2I/Borisov, and 3I/ATLAS. None of their designation aliases had a CAD approach within 1 AU inside the downloaded window.

Future CAD months count objects known today. They are not the same statistic as the backfilled history, and they omit anything surveys have not found yet. On 2026-10-22 the next 730 days contained **316** Earth and **32** Moon approaches within 0.05 AU (44 in the rest of 2026, 165 in 2027, 139 in 2028). SATCAT had **no** decay date on or after that day. The prior 120 days had 444 decay rows. The panel says so instead of inventing a reentry prediction.

#### Holdout scores

Cells are **MAE / RMSE**. The epoch column is the best epoch and how many epochs ran. The 12-month column is the calendar seasonal naive (it matches a 12-step lag on these month-start series). Discovery regression is ordinary least squares of the count on `log1p(known NEOs before the month)`, fit only before the first validation label. It is blank when the series is already a rate or is not a close-approach count. The mean for the rate series is not a useful baseline: 1901-04 has 21 approaches and one known NEO, which is 21,000 per 1,000, and that single month pulls the pre-holdout mean to about 1,698.

| Series | Validation | n | Epoch | Model | Persistence | 12-month | Mean | Discovery | Band coverage |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Earth within 1 LD | 2001-11 to 2026-09 | 299 | 1/7 | 4.9835 / 7.7407 | 2.7659 / 4.2493 | 2.4716 / 3.7251 | 5.0591 / 7.8181 | 5.0023 / 7.7557 | 0.27 |
| Earth within 0.05 AU | 2001-11 to 2026-09 | 299 | 3/9 | 51.2539 / 74.9242 | 18.5217 / 28.0773 | 13.7391 / 20.3286 | 58.0272 / 80.4750 | 53.9359 / 77.0095 | 0.11 |
| Earth closest miss (AU) | 2001-11 to 2026-09 | 299 | 9/15 | 0.0036 / 0.0039 | 0.0018 / 0.0034 | 0.0017 / 0.0034 | 0.0095 / 0.0097 | 0.0045 / 0.0047 | 1.00 |
| Earth H < 22 | 2001-11 to 2026-09 | 299 | 4/10 | 1.2741 / 1.7253 | 1.7258 / 2.3262 | 1.8562 / 2.3963 | 1.3468 / 1.7994 | 1.3152 / 1.7658 | 0.74 |
| Earth known at the time | 2001-11 to 2026-09 | 299 | 21/27 | 33.7492 / 48.5861 | 11.6823 / 17.8647 | 8.8729 / 12.8411 | 37.3034 / 51.2294 | 36.6263 / 50.6835 | 0.10 |
| Earth per 1,000 known NEOs | 2001-11 to 2026-09 | 299 | 2/8 | 14.9794 / 15.0590 | 1.3120 / 1.6950 | 1.1330 / 1.5243 | 1697.9478 / 1697.9487 |  | 0.91 |
| Moon within 1 LD | 2001-11 to 2026-09 | 299 | 4/10 | 3.5414 / 5.4937 | 2.2943 / 3.4076 | 2.0134 / 3.0028 | 3.5622 / 5.5187 | 3.5210 / 5.4662 | 0.24 |
| Moon within 0.05 AU | 2001-11 to 2026-09 | 299 | 3/9 | 16.4485 / 24.3747 | 6.6388 / 10.1288 | 5.4615 / 8.2731 | 17.1219 / 24.9666 | 16.5822 / 24.4958 | 0.16 |
| Moon closest miss (AU) | 2001-11 to 2026-09 | 299 | 12/18 | 0.0128 / 0.0136 | 0.0045 / 0.0127 | 0.0043 / 0.0123 | 0.0248 / 0.0248 | 0.0130 / 0.0137 | 1.00 |
| Moon H < 22 | 2001-11 to 2026-09 | 299 | 3/9 | 0.1560 / 0.2718 | 0.1538 / 0.3922 | 0.1438 / 0.3792 | 0.1339 / 0.2722 | 0.1333 / 0.2722 | 0.79 |
| Fireballs per month | 2019-07 to 2026-09 | 87 | 6/12 | 1.2611 / 1.6354 | 1.8621 / 2.2944 | 1.8621 / 2.3586 | 1.4225 / 1.8211 |  | 0.83 |
| Fireball energy (kt) | 2019-07 to 2026-09 | 87 | 4/10 | 1.3176 / 2.0868 | 1.6230 / 2.7438 | 2.5374 / 6.1600 | 2.3853 / 2.6045 |  | 0.65 |
| SATCAT reentries | 2013-06 to 2026-09 | 160 | 5/11 | 35.0027 / 58.0903 | 16.1187 / 26.0527 | 39.4188 / 64.4295 | 58.4066 / 90.0380 |  | 0.17 |

The raw Earth and Moon counts lose to the 12-month lag, the same pattern as the older close-approach model. Discovery regression beats the transformer on those counts and still loses to the seasonal naive, so removing the catalog-size trend does not make the network the best count forecast. The known-at-the-time series loses the same way (seasonal MAE 8.87, model 33.75). The rate series loses to persistence and the 12-month lag; its published 24-month p50 sits on the zero clip while p90 is about 6.5 per 1,000.

Earth approaches with `H < 22` are the close-approach series where the transformer has the lowest MAE of the three naive baselines (1.27 against a mean of 1.35, persistence 1.73, and the 12-month lag 1.86). Discovery regression is 1.32. The margin is small next to a series that is often a handful of objects a month. Moon `H < 22` does not repeat that result: the mean (0.134) and the discovery regression (0.133) beat the model (0.156).

Fireball counts and impact energy are the other two wins against persistence, the 12-month lag, and the mean. They are observed detections. The outlook is a statistical extrapolation, and a few large events dominate the energy. Reentry counts lose to persistence (16.1 against 35.0). Last month's decay count tracks the catalog better than the network does.

Band coverage near 0.10 on the growing count series means those p10–p90 ranges did not cover the later validation half. Closest-miss coverage near 1.00 means those bands are wide relative to the errors. Fireball-count coverage (0.83) is the one near the nominal 0.80. The chart draws the band anyway and overlays the JPL month counts inside the selected window, which are known objects only.

The weekly job includes these 13 configs (`retrain.enabled: true`, fetch timeout 600 seconds, shared cached fetch). Promotion still requires a fresh run to beat persistence, the seasonal naive, and the mean. A count model that loses that comparison does not replace a champion. The dashboard reads the committed score file above.

Closest Earth approaches in the 730-day CAD list on the fetch day, distance in lunar distances:

| Object | Date (TDB) | Distance | H |
| --- | --- | ---: | ---: |
| (2024 QP2) | 2028-10-15 17:32 | 0.57 LD | 21.5 |
| (2026 UA1) | 2026-10-23 17:35 | 0.58 LD | 27.8 |
| 153814 (2001 WN5) | 2028-06-26 05:23 | 0.65 LD | 18.3 |
| (2025 UK9) | 2026-10-31 02:35 | 0.82 LD | 29.8 |
| (2021 EO2) | 2028-05-19 05:29 | 0.97 LD | 27.9 |
| 137108 (1999 AN10) | 2027-08-07 07:11 | 1.01 LD | 18.1 |

Closest Moon approaches in that same list: (2025 UK9) on 2026-10-30 at 0.32 LD, (2026 UA1) on 2026-10-23 at 0.35 LD, and (2024 QP2) on 2028-10-16 at 0.54 LD.

### The rest of the solar system

This group does not retrain the solar-activity or solar-wind configs. It adds planetary close approaches, discovery counts, comet perihelia, and Earth-planet distances. The score file is `results/solar_system_forecast.json`. This run took **130.5 seconds** on CPU, seed 0, on 2026-10-22.

```bash
python -m scripts.fetch_solar_system
python -m src.space.train --results results/solar_system_forecast.json --package \
  --only mercury_approaches,venus_approaches,mars_approaches,jupiter_approaches,discoveries_neo,discoveries_main_belt,discoveries_trojan,discoveries_centaur,discoveries_tno,discoveries_comet,comet_perihelia,mercury_distance,venus_distance,mars_distance,jupiter_distance,saturn_distance,uranus_distance,neptune_distance,pluto_distance
```

`GET /space/solar-system?days=` defaults to **730**. The lists follow `days`. Each model is still one-step, scored on a temporal 20% holdout against persistence, the same month a year earlier, and the mean of every observation before the first validation label. The 24-month path rolls that one-step model forward. `p10` and `p90` are residual quantiles fit on the earlier half of the validation origins. Coverage is the later half of that same split, not a second untouched test. Counts are clipped at 0. Distances are clipped at 0.

#### Sources

No API key. Raw files stay under `data/space/` and are gitignored. The fetch on 2026-10-22 used CAD `date-max=2028-10-21` and Horizons through the same day. CAD times are TDB. Horizons observer times are UT.

| Series | What is counted | Source | Licence |
| --- | --- | --- | --- |
| Mercury, Venus, Mars approaches | Monthly counts within 0.05 AU | JPL CAD, body codes `Merc`, `Venus`, `Mars` | Public domain, U.S. government work |
| Jupiter approaches | Monthly counts within 1 AU. 0.05 AU has 8 Jupiter encounters since 1900, which is too thin to train | JPL CAD, `body=Juptr` | Public domain, U.S. government work |
| Saturn, Uranus, Neptune, Pluto | Listed from a 10 AU query. Not trained | JPL CAD, `Satrn`, `Urnus`, `Neptn`, `Pluto` | Public domain, U.S. government work. Wider cutoffs do not add rows |
| Discoveries | Monthly counts for NEO (APO, AMO, ATE, IEO), main belt (IMB, MBA, OMB), Jupiter trojans (TJN), centaurs (CEN), TNOs (TNO), and comets (`sb-kind=c`) | JPL SBDB Query API | Public domain, U.S. government work. `first_obs` values like `YYYY-??-??` are dropped |
| Comet perihelia | Passages stepped from the osculating perihelion `tp` and period `per` when the period is 100 days to 200 years. The training series keeps a passage only when `first_obs` is on or before it | Same comet query, fields `pdes,full_name,first_obs,tp,per` | Public domain, U.S. government work. One osculating perihelion is not a published history |
| Earth-planet distance | Monthly geocentric range. Sun-centered range is stored and not trained | JPL Horizons, quantities 20 and 23 (`S-O-T` elongation, not quantity 24) | Public domain, U.S. government work |
| Oppositions and conjunctions | Interior daily elongation extrema over the lookahead window | Horizons quantity 23, step 1 day | Public domain, U.S. government work |
| Moons | Discovery year and planet, shown and not trained | [JPL planetary satellite discoveries](https://ssd.jpl.nasa.gov/sats/discovery.html) | Public domain, U.S. government work. A year cell such as `1975, 2000` keeps the first year |
| Meteor showers | Not available | IMO calendar pages returned a restoration notice, and the linked PDFs were HTML | No ZHR table is invented |

The fetch wrote approach months 1900-01 through 2026-09 and discovery months 1701-10 through 2026-09. Dated discoveries, after dropping incomplete dates: **42,566** NEOs (1 dropped), **1,470,106** main-belt (4,791 dropped), **16,366** Jupiter trojans (45 dropped), **1,057** centaurs (5 dropped), **7,109** TNOs (193 dropped), **3,967** comets (114 dropped). Known-at-the-time perihelia in the training series: **5,186**. Horizons training months: 1,521, 1900-01 through 2026-09. The open month and later ephemeris months stay out of that CSV. The full monthly file runs through 2028-10 so the outlook can be compared with Horizons. Moons parsed: Mars 2, Jupiter 115, Saturn 293, Uranus 29, Neptune 16, Pluto 5 (460, matching the page headers).

On 2026-10-22 the next 730 days contained **77** Mercury, **137** Venus, **81** Mars, and **31** Jupiter approaches inside those distance limits, **266** comet perihelia, and **53** elongation extrema. Saturn has **12** historical CAD rows and **0** in the window. Uranus has **1** historical row and **0** in the window. Neptune and Pluto returned no rows at 10 AU.

#### Holdout scores

Cells are **MAE / RMSE**. The 12-month column is the calendar seasonal naive. Discovery regression is ordinary least squares of the count on `log1p(objects of that class known before the month)` for discoveries, or on known NEOs for the approach counts, fit only before the first validation label. It is blank for distances. The Physics column is the mean absolute error of the published 24-month p50 against Horizons on those outlook months. It is not a holdout score.

| Series | Validation | n | Epoch | Model | Persistence | 12-month | Mean | Discovery | Band | Physics |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mercury within 0.05 AU | 2001-11 to 2026-09 | 299 | 1/7 | 1.6171 / 1.9825 | 2.2575 / 2.8806 | 2.0669 / 2.6413 | 1.6781 / 2.0982 | 1.6036 | 0.78 | |
| Venus within 0.05 AU | 2001-11 to 2026-09 | 299 | 4/10 | 1.9027 / 2.4056 | 2.5686 / 3.2064 | 2.6154 / 3.3830 | 2.2900 / 3.0148 | 1.8947 | 0.65 | |
| Mars within 0.05 AU | 2001-11 to 2026-09 | 299 | 3/9 | 1.5339 / 2.1405 | 1.9565 / 2.6242 | 2.3512 / 3.1266 | 1.9321 / 2.6703 | 1.6527 | 0.75 | |
| Jupiter within 1 AU | 2001-11 to 2026-09 | 299 | 1/7 | 0.9381 / 1.3099 | 1.2709 / 1.7253 | 1.2542 / 1.7165 | 0.9583 / 1.3482 | 0.9369 | 0.79 | |
| NEO discoveries | 1962-03 to 2026-09 | 775 | 7/13 | 53.9712 / 101.5576 | 16.9781 / 37.0474 | 14.6684 / 29.5014 | 54.7674 / 102.2108 | 54.3640 | 0.03 | |
| Main-belt discoveries | 1962-03 to 2026-09 | 775 | 10/16 | 1806.9941 / 3327.6521 | 990.1561 / 2108.6204 | 942.4542 / 2156.5391 | 1876.8949 / 3395.9138 | 1860.9949 | 0.16 | |
| Jupiter-trojan discoveries | 1962-03 to 2026-09 | 775 | 7/13 | 20.0177 / 61.5473 | 21.6852 / 59.4767 | 22.7974 / 62.5724 | 20.7686 / 62.4905 | 20.5520 | 0.27 | |
| Centaur discoveries | 1962-03 to 2026-09 | 775 | 2/8 | 1.3596 / 3.2209 | 1.1045 / 2.8502 | 1.2735 / 3.2359 | 1.3602 / 3.2213 | 1.3589 | 0.03 | |
| TNO discoveries | 1962-03 to 2026-09 | 775 | 18/24 | 9.0722 / 33.4662 | 8.9639 / 32.3393 | 10.4194 / 39.2675 | 9.1551 / 33.5364 | 9.1128 | 0.01 | |
| Comet discoveries | 1962-03 to 2026-09 | 775 | 4/10 | 4.1912 / 7.4383 | 2.4271 / 4.1474 | 2.3923 / 4.4423 | 4.4546 / 7.7630 | 4.3563 | 0.12 | |
| Comet perihelia known at the time | 1962-03 to 2026-09 | 775 | 3/9 | 5.7030 / 9.0336 | 2.7019 / 5.0333 | 2.8129 / 5.1240 | 5.9974 / 9.3662 | 5.8487 | 0.04 | |
| Mercury distance (AU) | 2001-11 to 2026-09 | 299 | 25/31 | 0.0204 / 0.0264 | 0.3482 / 0.3923 | 0.2209 / 0.2479 | 0.2432 / 0.2738 |  | 0.80 | 0.0179 |
| Venus distance (AU) | 2001-11 to 2026-09 | 299 | 7/13 | 0.0180 / 0.0249 | 0.1483 / 0.1654 | 0.7824 / 0.8838 | 0.4253 / 0.4811 |  | 0.81 | 0.0642 |
| Mars distance (AU) | 2001-11 to 2026-09 | 299 | 13/19 | 0.0203 / 0.0252 | 0.1573 / 0.1772 | 1.1882 / 1.3485 | 0.6019 / 0.6843 |  | 0.82 | 0.0461 |
| Jupiter distance (AU) | 2001-11 to 2026-09 | 299 | 14/20 | 0.0275 / 0.0327 | 0.3023 / 0.3355 | 0.3363 / 0.3802 | 0.6462 / 0.7283 |  | 0.80 | 0.0988 |
| Saturn distance (AU) | 2001-11 to 2026-09 | 299 | 8/14 | 0.0291 / 0.0387 | 0.3190 / 0.3542 | 0.1413 / 0.1653 | 0.6785 / 0.7921 |  | 0.19 | 0.1274 |
| Uranus distance (AU) | 2001-11 to 2026-09 | 299 | 11/17 | 0.0179 / 0.0243 | 0.3265 / 0.3631 | 0.0498 / 0.0600 | 0.8019 / 0.9873 |  | 0.02 | 0.0995 |
| Neptune distance (AU) | 2001-11 to 2026-09 | 299 | 10/16 | 0.0293 / 0.0331 | 0.3285 / 0.3654 | 0.0251 / 0.0289 | 0.6481 / 0.7371 |  | 0.84 | 0.0341 |
| Pluto distance (AU) | 2001-11 to 2026-09 | 299 | 5/11 | 0.1616 / 0.1713 | 0.3245 / 0.3616 | 0.2051 / 0.2101 | 4.0809 / 4.4064 |  | 0.91 | 0.4217 |

Discovery counts are survey effort. NEO, main-belt, centaur, TNO, and comet discoveries, and known-at-the-time perihelia, all lose to persistence or the 12-month lag. Main-belt MAE is about 1,807 against a 12-month lag of 942. Discovery regression beats the transformer on those counts and still loses to the seasonal naive, except for trojans, where the regression (20.55) is between the model (20.02) and the mean (20.77). Trojan discoveries are the only discovery series where the transformer has a lower MAE than persistence, the 12-month lag, and the mean. The margin over the mean is under one discovery a month. Band coverage on the discovery series is 0.01 to 0.27, so those p10–p90 ranges did not cover the later validation half.

Mars approaches within 0.05 AU are the approach series where the transformer also beats the discovery regression (1.53 against 1.65, persistence 1.96, and the 12-month lag 2.35). Mercury (1.62) and Venus (1.90) beat the three naive baselines and lose to the discovery regression by a few hundredths (1.60 and 1.89). Jupiter within 1 AU does the same: model 0.938, mean 0.958, discovery regression 0.937. That Jupiter edge over the mean is smaller than the discovery control.

Earth-planet distance is an orbital signal. The 12-month lag is a poor description of Mars (synodic period about 26 months; seasonal MAE 1.19 AU against model 0.020). The transformer beats persistence, the 12-month lag, and the mean for Mercury, Venus, Mars, Jupiter, Saturn, Uranus, and Pluto. Neptune does not: the 12-month lag MAE is 0.0251 and the model is 0.0293. A slow outer planet is close to where it was a year ago. The Physics column is the honest check against the ephemeris on the 24 outlook months. Mercury's rolled-forward p50 is 0.018 AU from Horizons. Pluto's is 0.42 AU. The network is not a substitute for Horizons, and a holdout win over a 12-month lag does not mean the outlook reproduces the orbit. Uranus point MAE is 0.018 with band coverage 0.02: the point forecast tracked the holdout and the quantile band did not.

The weekly job includes these 19 configs (`retrain.enabled: true`, fetch timeout 600 seconds, shared cached fetch). A cold fetch on 2026-10-22 paged the main belt at 100,000 rows and finished well inside that cap; later configs reuse the cache. Promotion still requires a fresh run to beat persistence, the seasonal naive, and the mean. Discovery-count models that lose that comparison do not replace a champion. The dashboard reads the committed score file above.

Closest approaches in the 730-day window on the fetch day:

| Body | Object | Date (TDB) | Distance |
| --- | --- | --- | ---: |
| Mercury | (2001 CP36) | 2027-12-23 18:10 | 0.0027 AU (1.06 LD) |
| Mercury | (2018 VB1) | 2028-05-23 01:21 | 0.0029 AU (1.12 LD) |
| Venus | (2013 UB3) | 2028-05-20 11:56 | 0.0030 AU (1.18 LD) |
| Venus | (2021 CU1) | 2027-11-12 04:26 | 0.0051 AU (1.98 LD) |
| Mars | (2015 MY53) | 2027-05-18 23:06 | 0.0038 AU (1.48 LD) |
| Mars | (2011 PU1) | 2028-09-24 18:49 | 0.0080 AU (3.13 LD) |
| Jupiter | 323P/SOHO | 2026-12-08 19:59 | 0.283 AU |
| Jupiter | (2002 GJ8) | 2027-04-24 06:05 | 0.302 AU |

Saturn, Uranus, Neptune, and Pluto have no CAD approach inside the window. Upcoming elongation extrema from the daily Horizons file include a Venus conjunction on 2026-10-25, a Mercury conjunction on 2026-11-05, Uranus at opposition on 2026-11-26 (elongation 179.8°), Jupiter at opposition on 2027-02-11, Mars at opposition on 2027-02-20, and Saturn at opposition on 2027-10-18. The next comet perihelia in the catalog stepping are 73P/Schwassmann-Wachmann 3-BM on 2026-10-22 and 324P/La Sagra on 2026-10-24. The newest satellites in the JPL year table are S/2025 U1, S/2024 J1, and the S/2023 S designations at Saturn.

## Real-world series

Separate from the synthetic sine demo, `config/realworld/` trains the same one-step transformer on public series and on one synthetic website example. Each config uses a temporal holdout (the last 20% of windows), a fixed seed, z-score normalization fit on the training windows only, and early stopping on validation loss. Scores below are in the original units of the CSV. The checkpoint is the one with the best validation loss, and that loss is computed on the same holdout, so the model errors are slightly optimistic relative to the baselines, which are not tuned.

Downloaded files go to `data/realworld/` (gitignored, including the EIA bulk cache). Checkpoints go to `checkpoints/realworld/` (also gitignored). The small results files are `reports/realworld/forecast_results.json` and `reports/realworld/conflict_summary.json`.

```bash
python -m scripts.realworld.fetch_all
python -m scripts.realworld.analyze_conflict
python -m scripts.realworld.train_eval
```

`fetch_all` stops at the first failed download. The website step always refreshes the committed synthetic example; a Cloudflare download runs only when `CLOUDFLARE_API_TOKEN` is set together with `CLOUDFLARE_ACCOUNT_ID` (optional `CLOUDFLARE_SITE_TAG`) or `CLOUDFLARE_ZONE_ID`, and that live file is not what the default config trains on.

One CPU run of `train_eval` on this machine took **383 seconds (6.4 minutes)**. Electricity accounts for 239 of those seconds.

The weekly champion job includes Berlin ozone, Berlin PM2.5, and the daily M4.5+ earthquake counts. On the holdout below, each of those has a lower MAE than persistence, the seasonal naive, and the mean, and each fetch fits the job's 10-minute dataset cap. A checkpoint from that job is promoted only when the fresh run still beats those three baselines. The other configs set `retrain.enabled: false`. Pollen, CPI inflation, gasoline, unemployment, Arsenal goal difference, and monthly conflict events lose to at least one baseline. US48 demand beats the baselines, and the EIA bulk extract does not fit the time cap. Sunspots beat the baselines, and the series is CC BY-NC, so it stays out of the public champion release. Monthly fatalities and onset gaps are aggregate conflict series and are not published as an operational forecast. The website file is synthetic.

### Sources and licences

| Series | What is trained | Source and licence |
| --- | --- | --- |
| US48 hourly demand | Last 17,520 hours (2024-10-21 09:00Z through 2026-10-21 08:00Z) of EIA-930 series `EBA.US48-ALL.D.H`, megawatthours. The full bulk extract (68,409 hours from 2019-01-01) is written beside it and is not the training file. | Keyless `EBA.zip` from the U.S. Energy Information Administration. U.S. government work, public domain. |
| Berlin daily PM2.5, ozone, pollen | Daily means, 2022-01-01 through 2026-10-20 (1,754 days; pollen on 1,678 days). Pollen is the sum of alder, birch, grass, mugwort, olive, and ragweed. | Open-Meteo Air Quality API, CC BY 4.0, with CAMS / Copernicus attribution. Berlin is used because that pollen product covers Europe. |
| Earthquakes | Global daily count of magnitude 4.5+ events, 1990-01-01 through 2026-10-20 (13,442 days, 231,094 events). | USGS FDSN event service. Public domain. A relative energy column is stored and is not the training target. |
| Sunspots | Monthly mean total sunspot number, version 2.0, 1749-01 through 2026-09 (3,333 months). | WDC-SILSO, Royal Observatory of Belgium. **CC BY-NC 4.0** (attribution, non-commercial). |
| Unemployment | U.S. unemployment rate, monthly, 1948-01 through 2026-09. | FRED `UNRATE` (BLS), public domain, via keyless fredgraph CSV. |
| CPI inflation | Year-over-year percent change of CPI-U, all items, not seasonally adjusted (`CPIAUCNS`), 1914-01 through 2026-09. | FRED / BLS, public domain. |
| Gasoline | Weekly U.S. regular gasoline, all formulations, dollars per gallon (`GASREGW`), 1990-08-20 through 2026-10-19. | EIA via FRED, public domain. |
| Arsenal | Match goal difference (goals for minus goals against), Premier League, 1,269 matches from 1993-08-14 through 2026-10-18. The 1992–93 file was missing at the source. | Historical CSV from [football-data.co.uk](https://www.football-data.co.uk/). The site offers the files as a free download and does not attach an SPDX licence. Scores only; odds are unused. |
| Website visits | Synthetic daily counts, 2023-01-01 through 2025-12-31, committed at `examples/realworld/website_visits.csv`. | Generated in this repository. It is not a real site. |
| Conflict events and fatalities | Global monthly UCDP Georeferenced Event Dataset totals, 1989-01 through 2025-12 (444 months), all violence types. | UCDP GED 26.1, CC BY 4.0. Cite Sundberg and Melander, *Journal of Peace Research* (2013). |
| Days between new conflict onsets | Built from the UCDP/PRIO Armed Conflict Dataset. See the next section. | UCDP/PRIO ACD 26.1, CC BY 4.0. Cite Gleditsch, Wallensteen, Eriksson, Sollenberg, and Strand, *Journal of Peace Research* 39(5) (2002). |

### Holdout error

MAE and RMSE are on the temporal holdout. Seasonal naive uses the period in the config (168 hours for demand, 365 days for air quality and earthquakes, 132 months for sunspots, 52 weeks for gasoline, 38 matches for Arsenal, 12 months for unemployment, CPI, and the monthly conflict counts, 7 days for the website example, 4 gaps for onset spacing). Air-quality models see 90 days of context, which is shorter than that 365-day seasonal baseline.

| Series | Holdout | Model MAE (RMSE) | Persistence | Seasonal naive | Mean | Train |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Berlin ozone, µg/m³ | 332 | 7.34 (9.46) | 8.18 (10.39) | 13.06 (16.93) | 16.22 (19.94) | 11 s |
| Berlin PM2.5, µg/m³ | 332 | 2.50 (3.44) | 2.69 (3.71) | 4.29 (6.25) | 4.46 (5.73) | 8 s |
| Berlin pollen, grains/m³ | 317 | 16.10 (34.50) | **12.70 (37.83)** | 17.22 (45.58) | 27.95 (57.66) | 13 s |
| M4.5+ earthquakes, events/day | 2,676 | **5.84 (12.62)** | 7.55 (14.86) | 9.91 (20.60) | 6.80 (15.17) | 53 s |
| US48 demand, MWh | 3,470 | **4,695 (6,201)** | 16,964 (19,483) | 27,713 (36,096) | 90,716 (115,851) | 239 s |
| CPI inflation, percent | 265 | 0.353 (0.492) | **0.337 (0.470)** | 1.649 (2.241) | 1.606 (2.023) | 1 s |
| Regular gasoline, USD/gallon | 366 | 0.101 (0.153) | **0.049 (0.074)** | 0.560 (0.718) | 1.186 (1.342) | 5 s |
| Unemployment, percent | 184 | 0.220 (0.826) | **0.209 (0.825)** | 1.095 (1.985) | 1.665 (1.974) | 1 s |
| Arsenal goal difference, goals | 251 | 1.399 (1.846) | 1.924 (2.505) | 1.924 (2.478) | **1.413 (1.848)** | 1 s |
| Sunspots, monthly number | 640 | **16.70 (22.47)** | 18.04 (25.20) | 38.97 (53.61) | 58.13 (69.39) | 42 s |
| Conflict events, events/month | 84 | 216.9 (291.2) | **206.7 (275.2)** | 376.1 (484.1) | 1,026 (1,178) | 1 s |
| Conflict fatalities, deaths/month | 84 | **8,323 (22,751)** | 11,903 (32,240) | 11,833 (31,061) | 8,991 (23,620) | 1 s |
| Days between new onsets | 57 | **99.6 (170.2)** | 141.6 (239.8) | 127.2 (219.0) | 109.0 (183.8) | 1 s |
| Example website visits, visits/day | 213 | **33.3 (42.8)** | 147.8 (228.9) | 40.1 (49.9) | 182.3 (196.8) | 4 s |

Bold marks the lowest MAE in that row. The model beats the best of persistence, seasonal naive, and the training mean on ozone, PM2.5, earthquakes, electricity, sunspots, fatalities, onset gaps, and the synthetic visits. It loses on pollen, CPI, gasoline, unemployment, Arsenal (a tie with the mean, skill about 0.01), and monthly event counts.

A few of those outcomes are the series, not a tuning accident. Hourly demand has lag-1 correlation 0.98, and the previous hour is a much harder baseline than the same hour last week; on this two-year slice the model still cuts that error by about 72%. Gasoline (lag-1 0.999), unemployment (0.97), and CPI inflation (0.99) are smoother still, and yesterday's value wins. Arsenal goal difference has lag-1 correlation 0.014: the match margin sits on top of the historical average, and a 38-match seasonal naive (one modern league season; earlier seasons had 42) does not help. Monthly conflict fatalities have lag-1 correlation 0.02, so the historical mean is the serious baseline and the model's edge over it is about 7% of MAE. Monthly event counts are persistent (lag-1 0.96), and last month's count wins. The website file is synthetic, with a built-in weekly cycle; the model beats a 7-day seasonal naive there by about 17% of MAE.

The same onset-gap holdout (57 gaps, first target dated 2007-10-07) was also scored with an exponential trend in the gap length and a Poisson model of annual onset counts. Both were fit only on years before that holdout. A constant-rate exponential forecast is the training mean, so it is not a separate row. The time-varying exponential has MAE 120.4 days (RMSE 180.1). The Poisson trend, whose log mean declines by about 0.0095 per year over the 61 training years, has MAE 125.6 (RMSE 180.3). The transformer MAE of 99.6 days beats the mean (109.0) and both of those count models. The RMSE is still about 170 days, so the gain is a modest shift in a noisy series. Treat the whole conflict block as an aggregate description of the published record.

### Time between wars

This section describes intervals in the UCDP tables. It is an aggregate historical summary of those tables. It does not say when or where a particular war will start.

Definitions, using the dates UCDP published (most ACD start dates are day-precision; coarser precision codes are left as published):

- A new onset is the first `start_date` of each `conflict_id` on or after 1946-01-01. The gap is the number of days until the next such onset. Seven conflicts that start before 1946 are outside this clock. Same-day onsets produce a gap of 0 (there are 10).
- A conflict peace spell is the number of days from an episode's `ep_end_date` to the next episode's `start_date2` in the same `conflict_id`. If violence has not resumed by 2025-12-31, the spell is right-censored on that date. Overlapping episodes are not peace, and a non-positive gap is dropped.
- A country peace spell merges active intervals for each Gleditsch–Ward code in `gwno_loc`. A conflict that lists several countries contributes to each code.
- An episode duration runs from `start_date2` to `ep_end_date`. Episodes still open at the end of 2025 are censored. A same-day episode has duration 0.

The UCDP Conflict Termination dataset v4-2024 (codebook dated 24 June 2025; file version id 4.2024002) is a separate check. It follows ACD 25.1 and its years run through 2024, so its spells are not pooled with ACD 26.1. Cite Kreutz, *Journal of Peace Research* 47(2) (2010), for that dataset. Outcome codes follow the codebook: 1 peace agreement, 2 ceasefire, 3 victory for side A (government), 4 victory for side B (non-state), 5 low activity, 6 actor ceases to exist.

`python -m scripts.realworld.analyze_conflict` writes `data/realworld/ucdp_onset_gaps.csv`, `reports/realworld/conflict_summary.json`, and the charts under the run's artifact directory (histogram of gaps against an exponential density, gaps over time, Kaplan–Meier curves, episode durations, mean gap by decade, and holdout MAE).

**Onset gaps.** 295 gaps from 1946. Mean 97.5 days, median 56, standard deviation 126, minimum 0, maximum 978. The coefficient of variation squared is 1.66. Lag-1 correlation is 0.07, so a long gap does not reliably follow a long gap. On the 285 strictly positive gaps, a pure exponential (mean 101 days) is a poor description: Kolmogorov–Smirnov distance 0.086, bootstrap p = 0.005. A Weibull fit has shape 0.80 (mean 97 days, median 55). The likelihood-ratio test against an exponential rejects shape 1 (p ≈ 3×10⁻⁷). Shape below 1 is a decreasing hazard: short gaps are more common than an exponential with the same mean, and a long tail remains. That is overdispersion with weak serial correlation, which is a different pattern from a clustered run of long waits.

An ordinary least-squares slope is +0.79 days per year (+7.9 days per decade), permutation p = 0.011. The Mann–Kendall test on the same series is weaker (z = 1.43, p = 0.15). Decade means move around rather than climbing in a straight line: 1940s 46 days (n = 30), 1950s 125 (29), 1960s 75 (47), 1970s 98 (39), 1980s 107 (34), 1990s 75 (47), 2000s 215 (17), 2010s 80 (45), 2020s 288 (n = 7, and that decade is incomplete). The linear slope and the rank test do not tell the same story, and the 2000s and 2020s cells are small.

Regional gaps use episodes whose ACD region field is a single code. A few onsets carry more than one region code; those are omitted here.

| Region | Gaps | Mean days | Median | cv² |
| --- | ---: | ---: | ---: | ---: |
| Africa | 107 | 261 | 165 | 1.68 |
| Asia | 78 | 339 | 142 | 1.55 |
| Europe | 34 | 741 | 103 | 3.73 |
| Middle East | 35 | 820 | 456 | 1.63 |
| Americas | 25 | 806 | 535 | 0.89 |

Europe's mean is a long-tail mean: the median is 103 days and cv² is 3.73.

**Peace spells.** Means computed only on spells where violence resumed are biased short, because the censored spells (violence had not resumed by the censor date) are dropped. The Kaplan–Meier median keeps those spells in the risk set. The Weibull mean also uses the censored spells, and with shape below 1 it is a heavy-tail mean, not a typical length. The Kaplan–Meier median is the better single summary of a typical spell.

| Spell | n | Observed | Censored | KM median | Still ongoing at the end | Weibull shape | Weibull median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Conflict / incompatibility (ACD 26.1) | 563 | 325 | 238 | 4,916 days | 0.36 | 0.60 | 7,508 days |
| Country (merged `gwno_loc`) | 417 | 331 | 86 | 1,867 days | 0.15 | 0.65 | 2,812 days |
| Termination, conflict (ACD 25.1 window) | 505 | 277 | 228 | 6,127 days | 0.38 | 0.60 | 8,500 days |
| Termination, dyad | 914 | 299 | 615 | not reached | 0.64 | 0.49 | 30,200 days |

About 36% of conflict-level spells are still open at the end of 2025, so many incompatibilities in this table do not resume inside the observation window. The country-level median is shorter (about 5.1 years) because a country can enter a new conflict that is a different `conflict_id`. The dyad curve never crosses one half (survival at the end of 2024 is 0.64), so a median time until the same dyad resumes is not identified. Conflict-level termination outcomes, as a robustness check on the older file: low activity 209, government victory 106, ceasefire 67, peace agreement 60, non-state victory 43, actor ceases to exist 20. Dyad outcomes: low activity 395, government victory 149, peace agreement 109, ceasefire 101, actor ceases to exist 88, non-state victory 73. Low activity is the modal code in both files.

**Episode durations.** 628 episodes, 563 ended, 65 still open at the end of 2025. Kaplan–Meier median 385 days. Episodes that have ended: mean 1,126 days, median 305, minimum 0, maximum 17,075, cv² 3.66. Weibull shape 0.42 (median 325 days, mean 2,251 days); an exponential is rejected. Among episodes that have already ended, duration declines by about 18.5 days per year (about 185 days per decade), permutation p = 0.0005. That slope ignores episodes that are still open.

## Market forecasts (stocks, currencies, crypto, commodities)

Not financial advice. This is an educational evaluation of a small transformer against naive baselines on past prices. It is not a forecast you should trade on, and it is not a claim about future returns.

The model predicts the next **log return**, not the price level. Features are that return and a 10-observation rolling standard deviation. Returns use consecutive published observations only (business days for ECB rates and equity sessions, exchange sessions for futures, calendar days for crypto) and only when both prices are positive. Inputs and the target are z-scored with statistics fit on the training windows. The holdout is the most recent 20% of windows (`split_mode: temporal`). Early stopping keeps the best validation checkpoint (`patience: 3`, at most `8` epochs, seed `0`). The network is a small transformer (`d_model=16`, `1` encoder layer, sequence length `16`).

Two baselines are scored in those same return units:

- **Persistence (random walk).** Predict a return of zero. On the price, that is "tomorrow's price equals today's price."
- **Historical mean.** Predict the training-window average return, held constant.

Directional accuracy is the fraction of holdout observations with a nonzero return whose sign the model matches. A coin flip is 50%. The fraction of up moves in the holdout is the hit rate of always predicting an increase, so a print near that rate is not skill. RMSE below is the root mean squared **log return** on the holdout. A ratio above 1 means the model was worse than that baseline.

### Sources and terms

Prices are not committed. `scripts/fetch_markets.py` downloads them into `data/markets/` (gitignored by `/data/`). Checkpoints written by training are gitignored as well.

- **ECB euro foreign-exchange reference rates**, file `eurofxref-hist.zip`. The file quotes foreign currency units per 1 euro. This project converts each rate to units of that currency per 1 USD using the ECB dollar rate, and stores the euro as EUR per 1 USD. Reuse is allowed with attribution to the European Central Bank. The file is newest-first; the loader sorts it into time order before the split.
- **Frankfurter** (`api.frankfurter.app`) mirrors those ECB rates. The fetch records a same-day cross-check of the USD crosses. It is not a second training source.
- **FRED** `fredgraph.csv` (no API key) is the preferred source for the S&P 500 (`SP500`), WTI (`DCOILWTICO`), Brent (`DCOILBRENTEU`), Henry Hub natural gas (`DHHNGSP`), and the H.10 New Taiwan dollar rate (`DEXTWUS`), which the ECB file does not carry. S&P 500 values are S&P Dow Jones Indices data via FRED. H.10 rates are Board of Governors data. Oil and gas are EIA data via FRED. If those downloads time out, the script says so and uses the Yahoo fallback below where one exists. IMF monthly aluminum (`PALUMUSDM`) is not used: a daily aluminum futures history is available.
- **Stooq** daily CSVs were requested first for equities. The host returned an HTML browser check rather than a file, so equities were not taken from Stooq.
- **Yahoo Finance chart** (no API key) supplies AAPL, MSFT, and NVDA, plus commodity futures (gold `GC=F`, silver `SI=F`, copper `HG=F`, aluminum `ALI=F`, platinum `PL=F`, palladium `PA=F`, and the oil and gas contracts when FRED is unavailable) and the S&P 500 (`^GSPC`) when FRED is unavailable. Adjusted close is used when the payload includes it. Front-month futures include roll gaps. Yahoo's terms apply. The series are not redistributed in this repository.
- **CoinGecko** `coins/markets` (no API key) supplies the market-cap order. If it is unavailable, CoinPaprika tickers are the fallback. Prices are not taken from either ranking source.
- **Binance** public spot klines from `data-api.binance.vision` (no API key), daily, quote asset USDT. `api.binance.com` returned HTTP 451 from this environment, so it is not used. A `1000…USDT` contract is accepted when that is the listed market; its log return matches the token while the scale factor is constant. **Coinbase** exchange candles are the fallback for a ranked coin with no Binance USDT market.
- Stablecoins are skipped. Wrapped and staked duplicates (WBTC, stETH, and the same pattern) are skipped when that still leaves 50 coins with at least two years of daily candles.

Run notes from the fetch that produced these tables:

- FRED graph CSV responded.
- Stooq returned an HTML browser check rather than a CSV, so AAPL, MSFT, and NVDA come from Yahoo Finance.
- Frankfurter USD rates on 2026-10-20 differed from the ECB cross by at most 0.0047% across 29 currencies.
- ECB file 1999-01-04 to 2026-10-20: 41 currencies converted to per-USD, 41 long enough to train.
- CoinGecko market-cap ranking: 250 tickers on page 1.
- Binance lists 516 trading USDT symbols.
- Crypto sample: 50 coins trained (cap 50); 53 higher-ranked or ineligible names skipped.

### Fetch and train

From the repository root, with the project virtualenv:

```bash
python scripts/fetch_markets.py
python scripts/train_markets.py
```

Training reads `config/stocks.yaml`, `config/fx.yaml`, `config/crypto.yaml`, and `config/commodities.yaml`. Those files hold the shared model, the temporal split, the seed, and early stopping. The trainer writes one ordinary CSV config per series under `data/markets/runs/` and evaluates the best checkpoint, not the weights from the final epoch. One currency or one coin is one model. Total CPU training time for this run was 61 seconds.

### Results

Holdout metrics are log returns. Persistence RMSE is the RMSE of predicting zero. Mean RMSE predicts the training-set mean return. Directional accuracy is compared with 50% in the text under each table.

| Group | Series | Beat persistence | Beat mean return | Median RMSE / persistence | Median RMSE / mean | Median directional accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stocks and the S&P 500 | 4 | 2/4 | 2/4 | 0.999 | 1.000 | 52.3% |
| Currencies (per 1 USD) | 41 | 19/41 | 20/41 | 1.000 | 1.000 | 51.0% |
| Cryptocurrencies | 50 | 12/50 | 18/50 | 1.002 | 1.001 | 50.8% |
| Commodities | 9 | 5/9 | 5/9 | 1.000 | 1.000 | 50.8% |

The largest improvement on persistence is Turkish lira (RMSE 0.972 times the zero forecast, directional accuracy 76.8%). The training-set mean sign hits 77.0% of that same holdout, and the model RMSE is 0.995 times the historical-mean forecast. A gap that shrinks once the average drift is the baseline is the drift itself. The same reading applies when directional accuracy sits next to the fraction of up days.

#### Stocks and the S&P 500

Equity and index results use the same return baseline. A single recent holdout is not evidence of a tradable edge. 2 of 4 series had a lower RMSE than persistence, but the median ratio was 0.999 and median directional accuracy was 52.3%. The sign of the training-set mean return matched the holdout more often (median 54.1%) than the model did.

| Series | Source | Coverage | Holdout | N | Model RMSE | Persistence RMSE | Mean RMSE | Dir. acc. |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Apple | Yahoo AAPL | 2010-01-04 → 2026-10-21 | 2023-06-16 → 2026-10-21 | 840 | 0.01644 | 0.01643 | 0.01642 | 52.3% |
| Microsoft | Yahoo MSFT | 2010-01-04 → 2026-10-21 | 2023-06-16 → 2026-10-21 | 840 | 0.01616 | 0.01615 | 0.01615 | 53.6% |
| NVIDIA | Yahoo NVDA | 2010-01-04 → 2026-10-21 | 2023-06-16 → 2026-10-21 | 840 | 0.02881 | 0.02888 | 0.02882 | 50.9% |
| S&P 500 | FRED SP500 | 2016-10-21 → 2026-10-20 | 2024-10-25 → 2026-10-20 | 497 | 0.01010 | 0.01013 | 0.01012 | 52.3% |

#### Currencies (per 1 USD)

FX reference rates are close to a random walk. The typical series did not beat a no-change forecast (median RMSE 1.000 times persistence, 19 of 41 series lower). Median directional accuracy was 51.0%.

| Series | Source | Coverage | Holdout | N | Model RMSE | Persistence RMSE | Mean RMSE | Dir. acc. |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Australian dollar | ECB eurofxref-hist | 1999-01-04 → 2026-10-20 | 2021-04-12 → 2026-10-20 | 1418 | 0.00652 | 0.00654 | 0.00654 | 51.6% |
| Bulgarian lev | ECB eurofxref-hist | 2000-07-19 → 2025-12-31 | 2020-12-09 → 2025-12-31 | 1297 | 0.00477 | 0.00477 | 0.00477 | 50.4% |
| Brazilian real | ECB eurofxref-hist | 2008-01-02 → 2026-10-20 | 2023-01-20 → 2026-10-20 | 957 | 0.00741 | 0.00742 | 0.00742 | 51.1% |
| Canadian dollar | ECB eurofxref-hist | 1999-01-04 → 2026-10-20 | 2021-04-12 → 2026-10-20 | 1418 | 0.00398 | 0.00400 | 0.00400 | 51.3% |
| Swiss franc | ECB eurofxref-hist | 1999-01-04 → 2026-10-20 | 2021-04-12 → 2026-10-20 | 1418 | 0.00497 | 0.00496 | 0.00496 | 46.4% |
| Chinese yuan | ECB eurofxref-hist | 2005-04-01 → 2026-10-20 | 2022-07-06 → 2026-10-20 | 1098 | 0.00258 | 0.00258 | 0.00258 | 51.8% |
| Cypriot pound | ECB eurofxref-hist | 1999-01-04 → 2007-12-31 | 2006-03-20 → 2007-12-31 | 455 | 0.00442 | 0.00442 | 0.00442 | 55.2% |
| Czech koruna | ECB eurofxref-hist | 1999-01-04 → 2026-10-20 | 2021-04-12 → 2026-10-20 | 1418 | 0.00581 | 0.00580 | 0.00580 | 51.3% |
| Danish krone | ECB eurofxref-hist | 1999-01-04 → 2026-10-20 | 2021-04-12 → 2026-10-20 | 1418 | 0.00467 | 0.00467 | 0.00467 | 50.2% |
| Estonian kroon | ECB eurofxref-hist | 1999-01-04 → 2010-12-31 | 2008-08-19 → 2010-12-31 | 609 | 0.00850 | 0.00851 | 0.00851 | 52.1% |
| Euro | ECB eurofxref-hist | 1999-01-04 → 2026-10-20 | 2021-04-12 → 2026-10-20 | 1418 | 0.00468 | 0.00467 | 0.00467 | 51.0% |
| British pound | ECB eurofxref-hist | 1999-01-04 → 2026-10-20 | 2021-04-12 → 2026-10-20 | 1418 | 0.00519 | 0.00518 | 0.00518 | 49.8% |
| Hong Kong dollar | ECB eurofxref-hist | 1999-01-04 → 2026-10-20 | 2021-04-12 → 2026-10-20 | 1418 | 0.00050 | 0.00050 | 0.00050 | 47.1% |
| Croatian kuna | ECB eurofxref-hist | 2005-04-01 → 2022-12-30 | 2019-06-27 → 2022-12-30 | 904 | 0.00483 | 0.00482 | 0.00482 | 49.1% |
| Hungarian forint | ECB eurofxref-hist | 1999-01-04 → 2026-10-20 | 2021-04-12 → 2026-10-20 | 1418 | 0.00831 | 0.00831 | 0.00831 | 49.8% |
| Indonesian rupiah | ECB eurofxref-hist | 2005-04-01 → 2026-10-20 | 2022-07-06 → 2026-10-20 | 1098 | 0.00389 | 0.00391 | 0.00391 | 56.4% |
| Israeli shekel | ECB eurofxref-hist | 2011-01-03 → 2026-10-20 | 2023-08-29 → 2026-10-20 | 803 | 0.00616 | 0.00615 | 0.00615 | 49.9% |
| Indian rupee | ECB eurofxref-hist | 2009-01-02 → 2026-10-20 | 2023-04-03 → 2026-10-20 | 906 | 0.00237 | 0.00238 | 0.00237 | 54.1% |
| Icelandic krona | ECB eurofxref-hist | 1999-01-04 → 2026-10-20 | 2023-01-31 → 2026-10-20 | 950 | 0.00493 | 0.00492 | 0.00493 | 50.2% |
| Japanese yen | ECB eurofxref-hist | 1999-01-04 → 2026-10-20 | 2021-04-12 → 2026-10-20 | 1418 | 0.00630 | 0.00631 | 0.00631 | 56.4% |
| South Korean won | ECB eurofxref-hist | 1999-01-04 → 2026-10-20 | 2021-04-12 → 2026-10-20 | 1418 | 0.00584 | 0.00584 | 0.00584 | 51.3% |
| Lithuanian litas | ECB eurofxref-hist | 1999-01-04 → 2014-12-31 | 2011-10-25 → 2014-12-31 | 814 | 0.00488 | 0.00488 | 0.00488 | 49.3% |
| Latvian lats | ECB eurofxref-hist | 1999-01-04 → 2013-12-31 | 2011-01-10 → 2013-12-31 | 763 | 0.00575 | 0.00576 | 0.00576 | 53.1% |
| Maltese lira | ECB eurofxref-hist | 1999-01-04 → 2007-12-31 | 2006-03-20 → 2007-12-31 | 455 | 0.00439 | 0.00440 | 0.00440 | 56.5% |
| Mexican peso | ECB eurofxref-hist | 2008-01-02 → 2026-10-20 | 2023-01-20 → 2026-10-20 | 957 | 0.00707 | 0.00706 | 0.00707 | 48.7% |
| Malaysian ringgit | ECB eurofxref-hist | 2005-04-01 → 2026-10-20 | 2022-07-06 → 2026-10-20 | 1098 | 0.00369 | 0.00370 | 0.00370 | 49.5% |
| Norwegian krone | ECB eurofxref-hist | 1999-01-04 → 2026-10-20 | 2021-04-12 → 2026-10-20 | 1418 | 0.00749 | 0.00750 | 0.00750 | 51.8% |
| New Zealand dollar | ECB eurofxref-hist | 1999-01-04 → 2026-10-20 | 2021-04-12 → 2026-10-20 | 1418 | 0.00657 | 0.00658 | 0.00658 | 52.8% |
| Philippine peso | ECB eurofxref-hist | 2005-04-01 → 2026-10-20 | 2022-07-06 → 2026-10-20 | 1098 | 0.00370 | 0.00370 | 0.00370 | 50.8% |
| Polish zloty | ECB eurofxref-hist | 1999-01-04 → 2026-10-20 | 2021-04-12 → 2026-10-20 | 1418 | 0.00660 | 0.00660 | 0.00660 | 49.8% |
| Romanian leu (old) | ECB eurofxref-hist | 1999-01-04 → 2005-06-30 | 2004-03-25 → 2005-06-30 | 327 | 0.00857 | 0.00856 | 0.00862 | 48.9% |
| Romanian leu | ECB eurofxref-hist | 2005-07-01 → 2026-10-20 | 2022-07-25 → 2026-10-20 | 1085 | 0.00504 | 0.00503 | 0.00503 | 49.3% |
| Russian ruble | ECB eurofxref-hist | 2005-04-01 → 2022-03-01 | 2018-10-22 → 2022-03-01 | 861 | 0.01165 | 0.01158 | 0.01157 | 48.4% |
| Swedish krona | ECB eurofxref-hist | 1999-01-04 → 2026-10-20 | 2021-04-12 → 2026-10-20 | 1418 | 0.00699 | 0.00699 | 0.00699 | 52.7% |
| Singapore dollar | ECB eurofxref-hist | 1999-01-04 → 2026-10-20 | 2021-04-12 → 2026-10-20 | 1418 | 0.00287 | 0.00287 | 0.00287 | 50.4% |
| Slovenian tolar | ECB eurofxref-hist | 1999-01-04 → 2006-12-29 | 2005-06-06 → 2006-12-29 | 404 | 0.00517 | 0.00516 | 0.00517 | 53.1% |
| Slovak koruna | ECB eurofxref-hist | 1999-01-04 → 2008-12-31 | 2007-01-09 → 2008-12-31 | 506 | 0.00800 | 0.00801 | 0.00801 | 52.4% |
| Thai baht | ECB eurofxref-hist | 2005-04-01 → 2026-10-20 | 2022-07-06 → 2026-10-20 | 1098 | 0.00534 | 0.00532 | 0.00532 | 48.5% |
| Turkish lira (old) | ECB eurofxref-hist | 1999-01-04 → 2004-12-31 | 2003-10-30 → 2004-12-31 | 302 | 0.00783 | 0.00776 | 0.00792 | 57.3% |
| Turkish lira | ECB eurofxref-hist | 2005-01-03 → 2026-10-20 | 2022-06-17 → 2026-10-20 | 1111 | 0.00401 | 0.00412 | 0.00403 | 76.8% |
| South African rand | ECB eurofxref-hist | 1999-01-04 → 2026-10-20 | 2021-04-12 → 2026-10-20 | 1418 | 0.00819 | 0.00819 | 0.00819 | 48.3% |

Coverage for every currencies (per 1 usd) series the fetch considered:

| Series | Source | Start | End | Observations | Trained | Note |
| --- | --- | --- | --- | ---: | --- | --- |
| Australian dollar | ECB eurofxref-hist | 1999-01-04 | 2026-10-20 | 7118 | yes | AUD per 1 USD |
| Bulgarian lev | ECB eurofxref-hist | 2000-07-19 | 2025-12-31 | 6515 | yes | BGN per 1 USD |
| Brazilian real | ECB eurofxref-hist | 2008-01-02 | 2026-10-20 | 4814 | yes | BRL per 1 USD |
| Canadian dollar | ECB eurofxref-hist | 1999-01-04 | 2026-10-20 | 7118 | yes | CAD per 1 USD |
| Swiss franc | ECB eurofxref-hist | 1999-01-04 | 2026-10-20 | 7118 | yes | CHF per 1 USD |
| Chinese yuan | ECB eurofxref-hist | 2005-04-01 | 2026-10-20 | 5519 | yes | CNY per 1 USD |
| Cypriot pound | ECB eurofxref-hist | 1999-01-04 | 2007-12-31 | 2304 | yes | CYP per 1 USD |
| Czech koruna | ECB eurofxref-hist | 1999-01-04 | 2026-10-20 | 7118 | yes | CZK per 1 USD |
| Danish krone | ECB eurofxref-hist | 1999-01-04 | 2026-10-20 | 7118 | yes | DKK per 1 USD |
| Estonian kroon | ECB eurofxref-hist | 1999-01-04 | 2010-12-31 | 3074 | yes | EEK per 1 USD |
| Euro | ECB eurofxref-hist | 1999-01-04 | 2026-10-20 | 7118 | yes | EUR per 1 USD |
| British pound | ECB eurofxref-hist | 1999-01-04 | 2026-10-20 | 7118 | yes | GBP per 1 USD |
| Hong Kong dollar | ECB eurofxref-hist | 1999-01-04 | 2026-10-20 | 7118 | yes | HKD per 1 USD |
| Croatian kuna | ECB eurofxref-hist | 2005-04-01 | 2022-12-30 | 4548 | yes | HRK per 1 USD |
| Hungarian forint | ECB eurofxref-hist | 1999-01-04 | 2026-10-20 | 7118 | yes | HUF per 1 USD |
| Indonesian rupiah | ECB eurofxref-hist | 2005-04-01 | 2026-10-20 | 5519 | yes | IDR per 1 USD |
| Israeli shekel | ECB eurofxref-hist | 2011-01-03 | 2026-10-20 | 4044 | yes | ILS per 1 USD |
| Indian rupee | ECB eurofxref-hist | 2009-01-02 | 2026-10-20 | 4558 | yes | INR per 1 USD |
| Icelandic krona | ECB eurofxref-hist | 1999-01-04 | 2026-10-20 | 4777 | yes | ISK per 1 USD |
| Japanese yen | ECB eurofxref-hist | 1999-01-04 | 2026-10-20 | 7118 | yes | JPY per 1 USD |
| South Korean won | ECB eurofxref-hist | 1999-01-04 | 2026-10-20 | 7118 | yes | KRW per 1 USD |
| Lithuanian litas | ECB eurofxref-hist | 1999-01-04 | 2014-12-31 | 4097 | yes | LTL per 1 USD |
| Latvian lats | ECB eurofxref-hist | 1999-01-04 | 2013-12-31 | 3842 | yes | LVL per 1 USD |
| Maltese lira | ECB eurofxref-hist | 1999-01-04 | 2007-12-31 | 2304 | yes | MTL per 1 USD |
| Mexican peso | ECB eurofxref-hist | 2008-01-02 | 2026-10-20 | 4814 | yes | MXN per 1 USD |
| Malaysian ringgit | ECB eurofxref-hist | 2005-04-01 | 2026-10-20 | 5519 | yes | MYR per 1 USD |
| Norwegian krone | ECB eurofxref-hist | 1999-01-04 | 2026-10-20 | 7118 | yes | NOK per 1 USD |
| New Zealand dollar | ECB eurofxref-hist | 1999-01-04 | 2026-10-20 | 7118 | yes | NZD per 1 USD |
| Philippine peso | ECB eurofxref-hist | 2005-04-01 | 2026-10-20 | 5519 | yes | PHP per 1 USD |
| Polish zloty | ECB eurofxref-hist | 1999-01-04 | 2026-10-20 | 7118 | yes | PLN per 1 USD |
| Romanian leu (old) | ECB eurofxref-hist | 1999-01-04 | 2005-06-30 | 1664 | yes | ROL per 1 USD |
| Romanian leu | ECB eurofxref-hist | 2005-07-01 | 2026-10-20 | 5454 | yes | RON per 1 USD |
| Russian ruble | ECB eurofxref-hist | 2005-04-01 | 2022-03-01 | 4333 | yes | RUB per 1 USD |
| Swedish krona | ECB eurofxref-hist | 1999-01-04 | 2026-10-20 | 7118 | yes | SEK per 1 USD |
| Singapore dollar | ECB eurofxref-hist | 1999-01-04 | 2026-10-20 | 7118 | yes | SGD per 1 USD |
| Slovenian tolar | ECB eurofxref-hist | 1999-01-04 | 2006-12-29 | 2049 | yes | SIT per 1 USD |
| Slovak koruna | ECB eurofxref-hist | 1999-01-04 | 2008-12-31 | 2560 | yes | SKK per 1 USD |
| Thai baht | ECB eurofxref-hist | 2005-04-01 | 2026-10-20 | 5519 | yes | THB per 1 USD |
| Turkish lira (old) | ECB eurofxref-hist | 1999-01-04 | 2004-12-31 | 1537 | yes | TRL per 1 USD |
| Turkish lira | ECB eurofxref-hist | 2005-01-03 | 2026-10-20 | 5581 | yes | TRY per 1 USD |
| New Taiwan dollar | unavailable |  |  | 0 | no | HTTP 404 for https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXTWUS |
| South African rand | ECB eurofxref-hist | 1999-01-04 | 2026-10-20 | 7118 | yes | ZAR per 1 USD |

#### Cryptocurrencies

Daily crypto closes are volatile, and a useful forecast still has to beat predicting that tomorrow's return is zero. The typical series did not beat a no-change forecast (median RMSE 1.002 times persistence, 12 of 50 series lower). Median directional accuracy was 50.8%.

| Series | Source | Coverage | Holdout | N | Model RMSE | Persistence RMSE | Mean RMSE | Dir. acc. |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Bitcoin | Binance BTCUSDT | 2017-08-17 → 2026-10-21 | 2024-12-26 → 2026-10-21 | 665 | 0.02251 | 0.02252 | 0.02256 | 51.1% |
| Ethereum | Binance ETHUSDT | 2017-08-17 → 2026-10-21 | 2024-12-26 → 2026-10-21 | 665 | 0.03579 | 0.03579 | 0.03581 | 53.8% |
| BNB | Binance BNBUSDT | 2017-11-06 → 2026-10-21 | 2025-01-11 → 2026-10-21 | 649 | 0.02558 | 0.02548 | 0.02557 | 48.7% |
| XRP | Binance XRPUSDT | 2018-05-04 → 2026-10-21 | 2025-02-16 → 2026-10-21 | 613 | 0.03811 | 0.03814 | 0.03816 | 55.7% |
| Solana | Binance SOLUSDT | 2020-08-11 → 2026-10-21 | 2025-08-01 → 2026-10-21 | 447 | 0.03594 | 0.03583 | 0.03597 | 49.8% |
| TRON | Binance TRXUSDT | 2018-06-11 → 2026-10-21 | 2025-02-24 → 2026-10-21 | 605 | 0.01516 | 0.01525 | 0.01525 | 54.5% |
| Zcash | Binance ZECUSDT | 2019-03-21 → 2026-10-21 | 2025-04-21 → 2026-10-21 | 549 | 0.07237 | 0.07213 | 0.07217 | 47.6% |
| Dogecoin | Binance DOGEUSDT | 2019-07-05 → 2026-10-21 | 2025-05-12 → 2026-10-21 | 528 | 0.03939 | 0.03931 | 0.03947 | 50.7% |
| Chainlink | Binance LINKUSDT | 2019-01-16 → 2026-10-21 | 2025-04-08 → 2026-10-21 | 562 | 0.03953 | 0.03951 | 0.03953 | 51.0% |
| Cardano | Binance ADAUSDT | 2018-04-17 → 2026-10-21 | 2025-02-13 → 2026-10-21 | 616 | 0.04787 | 0.04776 | 0.04778 | 52.4% |
| NEAR Protocol | Binance NEARUSDT | 2020-10-14 → 2026-10-21 | 2025-08-14 → 2026-10-21 | 434 | 0.05464 | 0.05444 | 0.05442 | 51.2% |
| Stellar | Binance XLMUSDT | 2018-05-31 → 2026-10-21 | 2025-02-21 → 2026-10-21 | 608 | 0.04140 | 0.04135 | 0.04136 | 55.1% |
| Bitcoin Cash | Binance BCHUSDT | 2019-11-28 → 2026-10-21 | 2025-06-11 → 2026-10-21 | 498 | 0.03851 | 0.03855 | 0.03857 | 53.4% |
| Litecoin | Binance LTCUSDT | 2017-12-13 → 2026-10-21 | 2025-01-19 → 2026-10-21 | 641 | 0.03517 | 0.03514 | 0.03513 | 50.3% |
| Uniswap | Binance UNIUSDT | 2020-09-17 → 2026-10-21 | 2025-08-08 → 2026-10-21 | 440 | 0.05096 | 0.05064 | 0.05065 | 48.6% |
| Avalanche | Binance AVAXUSDT | 2020-09-22 → 2026-10-21 | 2025-08-09 → 2026-10-21 | 439 | 0.04192 | 0.04178 | 0.04184 | 52.8% |
| Sui | Binance SUIUSDT | 2023-05-03 → 2026-10-21 | 2026-02-16 → 2026-10-21 | 248 | 0.04078 | 0.04035 | 0.04035 | 48.8% |
| Hedera | Binance HBARUSDT | 2019-09-29 → 2026-10-21 | 2025-05-30 → 2026-10-21 | 510 | 0.04062 | 0.04048 | 0.04052 | 53.7% |
| Bittensor | Binance TAOUSDT | 2024-04-11 → 2026-10-21 | 2026-04-26 → 2026-10-21 | 179 | 0.04319 | 0.04309 | 0.04311 | 51.1% |
| Shiba Inu | Binance SHIBUSDT | 2021-05-10 → 2026-10-21 | 2025-09-24 → 2026-10-21 | 393 | 0.03613 | 0.03605 | 0.03607 | 48.8% |
| Cronos | Coinbase CRO-USD | 2021-11-03 → 2026-10-21 | 2025-10-30 → 2026-10-21 | 357 | 0.03039 | 0.03042 | 0.03035 | 60.1% |
| Quant | Binance QNTUSDT | 2021-07-29 → 2026-10-21 | 2025-10-10 → 2026-10-21 | 377 | 0.05777 | 0.05742 | 0.05743 | 53.1% |
| Ethena | Binance ENAUSDT | 2024-04-02 → 2026-10-21 | 2026-04-24 → 2026-10-21 | 181 | 0.05989 | 0.05889 | 0.05912 | 44.4% |
| Aave | Binance AAVEUSDT | 2020-10-15 → 2026-10-21 | 2025-08-14 → 2026-10-21 | 434 | 0.04515 | 0.04509 | 0.04514 | 50.6% |
| Worldcoin | Binance WLDUSDT | 2023-07-24 → 2026-10-21 | 2026-03-04 → 2026-10-21 | 232 | 0.06695 | 0.06677 | 0.06682 | 47.0% |
| Polkadot | Binance DOTUSDT | 2020-08-18 → 2026-10-21 | 2025-08-02 → 2026-10-21 | 446 | 0.04447 | 0.04450 | 0.04448 | 57.2% |
| Pepe | Binance PEPEUSDT | 2023-05-05 → 2026-10-21 | 2026-02-16 → 2026-10-21 | 248 | 0.04275 | 0.04230 | 0.04232 | 50.0% |
| PAX Gold | Binance PAXGUSDT | 2020-08-28 → 2026-10-21 | 2025-08-04 → 2026-10-21 | 444 | 0.01467 | 0.01465 | 0.01465 | 49.3% |
| Internet Computer | Binance ICPUSDT | 2021-05-11 → 2026-10-21 | 2025-09-25 → 2026-10-21 | 392 | 0.05587 | 0.05597 | 0.05597 | 58.2% |
| Arbitrum | Binance ARBUSDT | 2023-03-23 → 2026-10-21 | 2026-02-08 → 2026-10-21 | 256 | 0.05105 | 0.05100 | 0.05115 | 49.0% |
| Ethereum Classic | Binance ETCUSDT | 2018-06-12 → 2026-10-21 | 2025-02-24 → 2026-10-21 | 605 | 0.03800 | 0.03804 | 0.03804 | 50.6% |
| JUST | Binance JSTUSDT | 2020-08-11 → 2026-10-21 | 2025-08-01 → 2026-10-21 | 447 | 0.03002 | 0.03025 | 0.03025 | 58.5% |
| Jupiter | Binance JUPUSDT | 2024-01-31 → 2026-10-21 | 2026-04-12 → 2026-10-21 | 193 | 0.05160 | 0.05150 | 0.05162 | 46.6% |
| Algorand | Binance ALGOUSDT | 2019-06-22 → 2026-10-21 | 2025-05-10 → 2026-10-21 | 530 | 0.04199 | 0.04185 | 0.04184 | 49.0% |
| POL (ex-MATIC) | Binance POLUSDT | 2024-09-13 → 2026-10-21 | 2026-05-27 → 2026-10-21 | 148 | 0.03681 | 0.03639 | 0.03649 | 47.3% |
| Render | Binance RENDERUSDT | 2024-07-26 → 2026-10-21 | 2026-05-17 → 2026-10-21 | 158 | 0.03803 | 0.03777 | 0.03781 | 46.1% |
| Cosmos Hub | Binance ATOMUSDT | 2019-04-29 → 2026-10-21 | 2025-04-29 → 2026-10-21 | 541 | 0.03783 | 0.03783 | 0.03783 | 50.9% |
| Injective | Binance INJUSDT | 2020-10-21 → 2026-10-21 | 2025-08-15 → 2026-10-21 | 433 | 0.05259 | 0.05241 | 0.05247 | 50.6% |
| Filecoin | Binance FILUSDT | 2020-10-15 → 2026-10-21 | 2025-08-14 → 2026-10-21 | 434 | 0.05746 | 0.05750 | 0.05746 | 51.0% |
| NEXO | Binance NEXOUSDT | 2022-04-29 → 2026-10-21 | 2025-12-04 → 2026-10-21 | 322 | 0.02370 | 0.02359 | 0.02358 | 49.0% |
| LayerZero | Binance ZROUSDT | 2024-06-20 → 2026-10-21 | 2026-05-10 → 2026-10-21 | 165 | 0.05327 | 0.05313 | 0.05321 | 52.4% |
| Aptos | Binance APTUSDT | 2022-10-19 → 2026-10-21 | 2026-01-08 → 2026-10-21 | 287 | 0.04461 | 0.04471 | 0.04467 | 55.0% |
| PancakeSwap | Binance CAKEUSDT | 2021-02-19 → 2026-10-21 | 2025-09-08 → 2026-10-21 | 409 | 0.04285 | 0.04286 | 0.04287 | 56.0% |
| Stacks | Binance STXUSDT | 2019-10-25 → 2026-10-21 | 2025-06-04 → 2026-10-21 | 505 | 0.04700 | 0.04672 | 0.04675 | 50.7% |
| Ether.fi | Binance ETHFIUSDT | 2024-03-18 → 2026-10-21 | 2026-04-21 → 2026-10-21 | 184 | 0.04981 | 0.04919 | 0.04943 | 51.4% |
| Dash | Binance DASHUSDT | 2019-03-28 → 2026-10-21 | 2025-04-23 → 2026-10-21 | 547 | 0.06567 | 0.06538 | 0.06540 | 50.5% |
| VeChain | Binance VETUSDT | 2018-07-25 → 2026-10-21 | 2025-03-04 → 2026-10-21 | 597 | 0.04093 | 0.04092 | 0.04094 | 53.1% |
| Pyth Network | Binance PYTHUSDT | 2024-02-02 → 2026-10-21 | 2026-04-12 → 2026-10-21 | 193 | 0.04733 | 0.04720 | 0.04756 | 49.2% |
| Curve DAO | Binance CRVUSDT | 2020-08-15 → 2026-10-21 | 2025-08-02 → 2026-10-21 | 446 | 0.04860 | 0.04860 | 0.04859 | 54.2% |
| Celestia | Binance TIAUSDT | 2023-10-31 → 2026-10-21 | 2026-03-24 → 2026-10-21 | 212 | 0.05402 | 0.05402 | 0.05432 | 50.5% |

Coverage for every cryptocurrencies series the fetch considered:

| Series | Source | Start | End | Observations | Trained | Note |
| --- | --- | --- | --- | ---: | --- | --- |
| Bitcoin | Binance BTCUSDT | 2017-08-17 | 2026-10-21 | 3353 | yes | USDT close |
| Ethereum | Binance ETHUSDT | 2017-08-17 | 2026-10-21 | 3353 | yes | USDT close |
| Tether | not downloaded |  |  | 0 | no | stablecoin |
| BNB | Binance BNBUSDT | 2017-11-06 | 2026-10-21 | 3272 | yes | USDT close |
| XRP | Binance XRPUSDT | 2018-05-04 | 2026-10-21 | 3093 | yes | USDT close |
| USDC | not downloaded |  |  | 0 | no | stablecoin |
| Solana | Binance SOLUSDT | 2020-08-11 | 2026-10-21 | 2263 | yes | USDT close |
| TRON | Binance TRXUSDT | 2018-06-11 | 2026-10-21 | 3055 | yes | USDT close |
| Figure Heloc | not downloaded |  |  | 0 | no | no Binance USDT market and no Coinbase FIGR_HELOC-USD candles |
| Zcash | Binance ZECUSDT | 2019-03-21 | 2026-10-21 | 2772 | yes | USDT close |
| Hyperliquid | Binance HYPEUSDT | 2026-09-24 | 2026-10-21 | 28 | no | coverage is 27 days, shorter than 700 |
| Dogecoin | Binance DOGEUSDT | 2019-07-05 | 2026-10-21 | 2666 | yes | USDT close |
| Monero | not downloaded |  |  | 0 | no | no Binance USDT market and no Coinbase XMR-USD candles |
| USDS | not downloaded |  |  | 0 | no | stablecoin |
| WhiteBIT Coin | not downloaded |  |  | 0 | no | no Binance USDT market and no Coinbase WBT-USD candles |
| Chainlink | Binance LINKUSDT | 2019-01-16 | 2026-10-21 | 2836 | yes | USDT close |
| Cardano | Binance ADAUSDT | 2018-04-17 | 2026-10-21 | 3110 | yes | USDT close |
| LEO Token | not downloaded |  |  | 0 | no | no Binance USDT market and no Coinbase LEO-USD candles |
| NEAR Protocol | Binance NEARUSDT | 2020-10-14 | 2026-10-21 | 2199 | yes | USDT close |
| Rain | not downloaded |  |  | 0 | no | no Binance USDT market and no Coinbase RAIN-USD candles |
| Stellar | Binance XLMUSDT | 2018-05-31 | 2026-10-21 | 3066 | yes | USDT close |
| Bitcoin Cash | Binance BCHUSDT | 2019-11-28 | 2026-10-21 | 2520 | yes | USDT close |
| Litecoin | Binance LTCUSDT | 2017-12-13 | 2026-10-21 | 3235 | yes | USDT close |
| Ethena USDe | not downloaded |  |  | 0 | no | stablecoin |
| Uniswap | Binance UNIUSDT | 2020-09-17 | 2026-10-21 | 2226 | yes | USDT close |
| Dai | not downloaded |  |  | 0 | no | stablecoin |
| Avalanche | Binance AVAXUSDT | 2020-09-22 | 2026-10-21 | 2221 | yes | USDT close |
| Canton | not downloaded |  |  | 0 | no | no Binance USDT market and no Coinbase CC-USD candles |
| Sui | Binance SUIUSDT | 2023-05-03 | 2026-10-21 | 1268 | yes | USDT close |
| USD1 | not downloaded |  |  | 0 | no | stablecoin |
| Gram (prev. Toncoin) | Binance GRAMUSDT | 2026-07-02 | 2026-10-21 | 112 | no | coverage is 111 days, shorter than 700 |
| Hedera | Binance HBARUSDT | 2019-09-29 | 2026-10-21 | 2580 | yes | USDT close |
| Tether Gold | not downloaded |  |  | 0 | no | stablecoin |
| Global Dollar | not downloaded |  |  | 0 | no | stablecoin |
| Bittensor | Binance TAOUSDT | 2024-04-11 | 2026-10-21 | 924 | yes | USDT close |
| Shiba Inu | Binance SHIBUSDT | 2021-05-10 | 2026-10-21 | 1991 | yes | USDT close |
| Cronos | Coinbase CRO-USD | 2021-11-03 | 2026-10-21 | 1814 | yes | USD close |
| PayPal USD | not downloaded |  |  | 0 | no | stablecoin |
| Quant | Binance QNTUSDT | 2021-07-29 | 2026-10-21 | 1911 | yes | USDT close |
| Ethena | Binance ENAUSDT | 2024-04-02 | 2026-10-21 | 933 | yes | USDT close |
| Bitway | not downloaded |  |  | 0 | no | no Binance USDT market and no Coinbase BTW-USD candles |
| OKB | not downloaded |  |  | 0 | no | no Binance USDT market and no Coinbase OKB-USD candles |
| Aave | Binance AAVEUSDT | 2020-10-15 | 2026-10-21 | 2198 | yes | USDT close |
| Circle USYC | not downloaded |  |  | 0 | no | no Binance USDT market and no Coinbase USYC-USD candles |
| Ripple USD | not downloaded |  |  | 0 | no | stablecoin |
| Ondo US Dollar Yield | not downloaded |  |  | 0 | no | stablecoin |
| Ondo | Binance ONDOUSDT | 2025-04-11 | 2026-10-21 | 559 | no | coverage is 558 days, shorter than 700 |
| Worldcoin | Binance WLDUSDT | 2023-07-24 | 2026-10-21 | 1186 | yes | USDT close |
| BlackRock USD Institutional Digital Liquidity Fund | not downloaded |  |  | 0 | no | no Binance USDT market and no Coinbase BUIDL-USD candles |
| Pump.fun | Binance PUMPUSDT | 2025-09-11 | 2026-10-21 | 406 | no | coverage is 405 days, shorter than 700 |
| Polkadot | Binance DOTUSDT | 2020-08-18 | 2026-10-21 | 2256 | yes | USDT close |
| Aster | Binance ASTERUSDT | 2025-10-06 | 2026-10-21 | 381 | no | coverage is 380 days, shorter than 700 |
| Pepe | Binance PEPEUSDT | 2023-05-05 | 2026-10-21 | 1266 | yes | USDT close |
| Falcon USD | not downloaded |  |  | 0 | no | no Binance USDT market and no Coinbase USDF-USD candles |
| PAX Gold | Binance PAXGUSDT | 2020-08-28 | 2026-10-21 | 2246 | yes | USDT close |
| Mantle | not downloaded |  |  | 0 | no | no Binance USDT market and no Coinbase MNT-USD candles |
| World Liberty Financial | Binance WLFIUSDT | 2025-09-01 | 2026-10-21 | 416 | no | coverage is 415 days, shorter than 700 |
| Sky | Binance SKYUSDT | 2025-09-17 | 2026-10-21 | 400 | no | coverage is 399 days, shorter than 700 |
| USDD | not downloaded |  |  | 0 | no | stablecoin |
| MemeCore | not downloaded |  |  | 0 | no | no Binance USDT market and no Coinbase M-USD candles |
| Spiko Amundi Overnight Swap Fund (EUR) | not downloaded |  |  | 0 | no | no Binance USDT market and no Coinbase EURSAFO-USD candles |
| Morpho | Binance MORPHOUSDT | 2025-10-03 | 2026-10-21 | 384 | no | coverage is 383 days, shorter than 700 |
| Internet Computer | Binance ICPUSDT | 2021-05-11 | 2026-10-21 | 1990 | yes | USDT close |
| United Stables | Binance UUSDT | 2026-01-13 | 2026-10-21 | 282 | no | coverage is 281 days, shorter than 700 |
| HTX DAO | not downloaded |  |  | 0 | no | no Binance USDT market and no Coinbase HTX-USD candles |
| Bitget Token | not downloaded |  |  | 0 | no | no Binance USDT market and no Coinbase BGB-USD candles |
| Arbitrum | Binance ARBUSDT | 2023-03-23 | 2026-10-21 | 1309 | yes | USDT close |
| BFUSD | not downloaded |  |  | 0 | no | stablecoin |
| USDGO | not downloaded |  |  | 0 | no | no Binance USDT market and no Coinbase USDGO-USD candles |
| Ethereum Classic | Binance ETCUSDT | 2018-06-12 | 2026-10-21 | 3054 | yes | USDT close |
| Venice Token | Coinbase VVV-USD | 2025-01-28 | 2026-10-21 | 632 | no | coverage is 631 days, shorter than 700 |
| JUST | Binance JSTUSDT | 2020-08-11 | 2026-10-21 | 2263 | yes | USDT close |
| Jupiter | Binance JUPUSDT | 2024-01-31 | 2026-10-21 | 995 | yes | USDT close |
| Gate | not downloaded |  |  | 0 | no | no Binance USDT market and no Coinbase GT-USD candles |
| Kaspa | not downloaded |  |  | 0 | no | no Binance USDT market and no Coinbase KAS-USD candles |
| KuCoin | not downloaded |  |  | 0 | no | no Binance USDT market and no Coinbase KCS-USD candles |
| Algorand | Binance ALGOUSDT | 2019-06-22 | 2026-10-21 | 2679 | yes | USDT close |
| POL (ex-MATIC) | Binance POLUSDT | 2024-09-13 | 2026-10-21 | 769 | yes | USDT close |
| Render | Binance RENDERUSDT | 2024-07-26 | 2026-10-21 | 818 | yes | USDT close |
| Cosmos Hub | Binance ATOMUSDT | 2019-04-29 | 2026-10-21 | 2733 | yes | USDT close |
| Pi Network | not downloaded |  |  | 0 | no | no Binance USDT market and no Coinbase PI-USD candles |
| Injective | Binance INJUSDT | 2020-10-21 | 2026-10-21 | 2192 | yes | USDT close |
| Filecoin | Binance FILUSDT | 2020-10-15 | 2026-10-21 | 2198 | yes | USDT close |
| Pearl | not downloaded |  |  | 0 | no | no Binance USDT market and no Coinbase PEARL-USD candles |
| Lighter | not downloaded |  |  | 0 | no | no Binance USDT market and no Coinbase LIT-USD candles |
| Aerodrome Finance | Binance AEROUSDT | 2026-07-17 | 2026-10-21 | 97 | no | coverage is 96 days, shorter than 700 |
| Open USD | not downloaded |  |  | 0 | no | no Binance USDT market and no Coinbase OUSD-USD candles |
| NEXO | Binance NEXOUSDT | 2022-04-29 | 2026-10-21 | 1637 | yes | USDT close |
| LayerZero | Binance ZROUSDT | 2024-06-20 | 2026-10-21 | 854 | yes | USDT close |
| Invesco Short Duration US Government Securities Fund | not downloaded |  |  | 0 | no | no Binance USDT market and no Coinbase USTB-USD candles |
| Aptos | Binance APTUSDT | 2022-10-19 | 2026-10-21 | 1464 | yes | USDT close |
| PancakeSwap | Binance CAKEUSDT | 2021-02-19 | 2026-10-21 | 2071 | yes | USDT close |
| GHO | not downloaded |  |  | 0 | no | no Binance USDT market and no Coinbase GHO-USD candles |
| Stacks | Binance STXUSDT | 2019-10-25 | 2026-10-21 | 2554 | yes | USDT close |
| Ether.fi | Binance ETHFIUSDT | 2024-03-18 | 2026-10-21 | 948 | yes | USDT close |
| XDC Network | not downloaded |  |  | 0 | no | no Binance USDT market and no Coinbase XDC-USD candles |
| Midnight | Binance NIGHTUSDT | 2026-03-11 | 2026-10-21 | 225 | no | coverage is 224 days, shorter than 700 |
| Dash | Binance DASHUSDT | 2019-03-28 | 2026-10-21 | 2765 | yes | USDT close |
| Akedo | not downloaded |  |  | 0 | no | no Binance USDT market and no Coinbase AKE-USD candles |
| VeChain | Binance VETUSDT | 2018-07-25 | 2026-10-21 | 3011 | yes | USDT close |
| Pyth Network | Binance PYTHUSDT | 2024-02-02 | 2026-10-21 | 993 | yes | USDT close |
| Curve DAO | Binance CRVUSDT | 2020-08-15 | 2026-10-21 | 2259 | yes | USDT close |
| Celestia | Binance TIAUSDT | 2023-10-31 | 2026-10-21 | 1087 | yes | USDT close |

#### Commodities

Commodity results use the same return baseline. Front-month futures include contract rolls, so they are not a pure spot series. 5 of 9 series had a lower RMSE than persistence, but the median ratio was 1.000 and median directional accuracy was 50.8%. The sign of the training-set mean return matched the holdout more often (median 52.2%) than the model did.

| Series | Source | Coverage | Holdout | N | Model RMSE | Persistence RMSE | Mean RMSE | Dir. acc. |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Aluminum futures (Yahoo ALI=F) | Yahoo ALI=F | 2014-05-06 → 2026-10-21 | 2024-05-13 → 2026-10-21 | 614 | 0.01522 | 0.01523 | 0.01523 | 49.8% |
| Brent crude (spot) | FRED DCOILBRENTEU | 1987-05-20 → 2026-10-14 | 2019-08-14 → 2026-10-14 | 1816 | 0.03613 | 0.03613 | 0.03613 | 52.9% |
| Copper futures (Yahoo HG=F) | Yahoo HG=F | 2010-01-04 → 2026-10-21 | 2023-06-22 → 2026-10-21 | 840 | 0.01831 | 0.01828 | 0.01827 | 51.6% |
| Gold futures (Yahoo GC=F) | Yahoo GC=F | 2010-01-04 → 2026-10-21 | 2023-06-22 → 2026-10-21 | 840 | 0.01313 | 0.01316 | 0.01315 | 50.3% |
| Henry Hub natural gas (spot) | FRED DHHNGSP | 1997-01-07 → 2026-10-14 | 2022-03-28 → 2026-10-14 | 1136 | 0.11118 | 0.11160 | 0.11160 | 50.5% |
| Palladium futures (Yahoo PA=F) | Yahoo PA=F | 2010-01-04 → 2026-10-21 | 2023-06-26 → 2026-10-21 | 836 | 0.02874 | 0.02864 | 0.02865 | 46.4% |
| Platinum futures (Yahoo PL=F) | Yahoo PL=F | 2010-01-04 → 2026-10-21 | 2023-06-21 → 2026-10-21 | 839 | 0.02455 | 0.02448 | 0.02448 | 53.3% |
| Silver futures (Yahoo SI=F) | Yahoo SI=F | 2010-01-04 → 2026-10-21 | 2023-06-22 → 2026-10-21 | 839 | 0.02831 | 0.02836 | 0.02836 | 51.8% |
| WTI crude (Cushing spot) | FRED DCOILWTICO | 1986-01-02 → 2026-10-14 | 2019-02-28 → 2026-10-14 | 1897 | 0.03164 | 0.03164 | 0.03164 | 50.8% |

### Limits

A lower RMSE than persistence on one holdout, especially by a fraction of a percent, is not a trading edge. The model does not see order flow, news, or funding costs, and the commodity futures are front-month contracts. Discontinued ECB currencies (for example the old Romanian leu, or the ruble after the ECB stopped publishing it) are scored on the last 20% of their own sample, which is not 2026. Crypto coins with less than two years of daily candles are listed and not trained. Pegged currencies (the Danish krone, the Hong Kong dollar, and others) mostly import the dollar's move against the peg partner, so their errors are not independent of the euro or the dollar.

Retrain with the commands above. Live rankings and the latest ECB file move, so a later run will not reproduce these digits exactly; the seed, the split, and the checkpoint selection are fixed.
## Domain forecasts

Horse racing, AI progress, and the quantum series below use the existing `time_series_transformer`. Each run is a temporal split (the last `test_fraction` of windows or races is untouched by fitting, early stopping, and baseline coefficients; `val_fraction` is the slice immediately before that block). Inputs are standardised from the training split only. `training.seed` is 0. Each of these configs sets `retrain.enabled: false`, so the weekly champion job does not train them. Checkpoints go under `checkpoints/` and are not committed. Small derived tables live in `datasets/`; metrics live in `results/`.

```bash
python -m scripts.plot_domain_forecasts
```

That writes the comparison charts. matplotlib is used only by the plot script and is not part of the API lock.

### Horse racing

Public Hong Kong Jockey Club results, with Singapore rows dropped. The fetch script downloads `performances.csv.gz` from [eprochasson/horserace_data](https://github.com/eprochasson/horserace_data) into `data/raw/horse_racing/`, which is gitignored. That repository has no licence file (the GitHub licence API returned 404 on 2026-10-21). Its README says the tables were scraped from public HKJC and Singapore Turf Club pages and are provided without a warranty. This project does not redistribute the raw table.

```bash
python -m scripts.fetch_horse_racing
python -m scripts.train_horse_racing --config config/horse_racing.yaml
```

The kept table is 273,977 Hong Kong runners in 24,005 races from 1979-09-22 to 2018-06-27 (median field 12). Each runner is a sequence of up to six earlier runs plus one current-race token. The current token has days since the last run, weight, draw, and the jockey and trainer strike rates from races already run. It omits the current result and the current odds. Settlement still uses the published dividends, so the market comparison is on form. The win pick is the runner with the lowest predicted finish percentile. Top-3 hit rate is how often that single pick finished in the first three. Probabilities are a softmax of the negated scores; the temperature (0.2) is chosen on the validation races. Flat-stake profit is the published decimal dividend minus one when the pick wins, and minus one otherwise. Those dividends are stake-inclusive, and the test-set mean overround is 0.230, close to the 0.212 implied by HKJC's 17.5% win-pool deduction. The take is not applied a second time.

The test block is 4,606 races from 2012-09-26 to 2018-06-27. Training took 49.8 seconds on CPU and stopped at epoch 3.

| | Win pick | Top-3 hit | Log loss | Brier | Flat-stake ROI |
| --- | ---: | ---: | ---: | ---: | ---: |
| Model | 22.5% | 52.8% | 2.257 | 0.873 | −25.0% |
| Favourite (lowest odds) | 30.4% | 60.7% | 2.020 | 0.820 | −15.1% |
| Uniform pick | 8.2% | 24.6% | 2.511 | 0.918 | |

Log loss and Brier for the favourite column are the market probabilities after normalising the implied odds. The model is ahead of a uniform pick and behind the market on every column. A one-unit stake on the model lost 1,149.7 units; the same stake on the favourite lost 695.3. `beats_market` is false.

Not betting advice. This is a historical comparison of a forecasting model with the closing market. A simulated return is not a reason to bet, and public form usually does not beat the odds once the track take is included.

### AI progress

Three public series, each scored against one-step persistence, a log-linear (or already-log10 linear) trend fit on the training window, and the training-window mean.

**Frontier training compute.** [Epoch AI, Data on AI Models](https://epoch.ai/data/ai-models) (`notable_ai_models.csv`, CC BY 4.0). The monthly series is the cumulative maximum of frontier-model training compute from 2010-09 through 2026-09 (193 months, 72 frontier models with compute on or after 2010). Earlier systems stay in `datasets/ai_progress/frontier_models.csv` and are left out of the forecast series.

```bash
python -m scripts.fetch_epoch_ai_models
python -m scripts.train_domain_series --config config/ai_frontier_compute.yaml
```

Test months are 2023-10 through 2026-09 (36 months). Training took 1.4 seconds and stopped at epoch 12. Errors are on log10 FLOP.

| | MAE | RMSE |
| --- | ---: | ---: |
| Model | 0.896 | 0.975 |
| Persistence | 0.046 | 0.167 |
| Log-linear | 0.495 | 0.557 |
| Training mean | 5.305 | 5.331 |

The record is a step function, so persistence is the best one-step forecast. On the training window the log10 series rises by 0.700 per year (R² 0.944), a doubling time of 0.43 years (about 5.2 months). A fit that also includes the test months, reported only as a description of the post-2010 frontier, gives 0.45 years (R² 0.973, 72 models). Across all 114 frontier models with compute, including pre-2010 systems, the descriptive doubling time is 0.96 years (R² 0.891).

**arXiv submissions.** Monthly counts for `cs.AI`, `cs.LG`, and `cs.CL` from the public arXiv API, January 2012 through September 2026 (177 months). The script waits at least three seconds between requests and stores totals only, not paper text. Provenance is `datasets/ai_progress/arxiv_provenance.json`.

```bash
python -m scripts.fetch_arxiv_monthly_counts
python -m scripts.train_domain_series --config config/ai_arxiv_counts.yaml
```

The same fetch also writes the `quant-ph` series used below. Test months are 2024-01 through 2026-09 (33 months). Training took 1.5 seconds and stopped at epoch 19. Errors are in submissions per month. Doubling times are the training-window log-linear fit.

| Category | Model MAE / RMSE | Persistence | Log-linear | Training mean | Doubling |
| --- | ---: | ---: | ---: | ---: | ---: |
| cs.AI | 2132 / 2403 | 670 / 816 | 890 / 1101 | 3393 / 3590 | 2.20 years |
| cs.LG | 1304 / 1602 | 864 / 1083 | 8161 / 8796 | 2900 / 3047 | 1.73 years |
| cs.CL | 1228 / 1336 | 556 / 660 | 1459 / 1703 | 1702 / 1781 | 1.80 years |

Persistence is the best one-step forecast in every category. The cs.LG training slope (R² 0.931) doubles in 1.73 years and then overshoots the test period, which is why its test error is the worst of the four. The model beats the training-mean baseline in all three categories.

**GitHub repositories.** Unauthenticated search for `topic:machine-learning created:` each month from 2015-01 through 2026-09 (141 months, 10 requests per minute, no token). Repositories that never set the topic are missing, so the level is an undercount. No month was flagged `incomplete_results`. Provenance is `datasets/ai_progress/github_provenance.json`.

```bash
python -m scripts.fetch_github_ai_repos
python -m scripts.train_domain_series --config config/ai_github_repos.yaml
```

Test months are 2024-08 through 2026-09 (26 months). Training took 1.2 seconds and stopped at epoch 17.

| | MAE | RMSE |
| --- | ---: | ---: |
| Model | 2085 | 3212 |
| Persistence | 564 | 974 |
| Log-linear | 2258 | 2382 |
| Training mean | 3501 | 4632 |

Persistence is again the best one-step forecast. The training-window slope is 0.301 nats per year (R² 0.682), a doubling time of 2.30 years. Counts jump in 2026 (above 11,000 in July), and both the model and the log-linear trend miss that jump.

### Quantum

**Computing progress.** Annual records from [Awesome Quantum Computing Experiments](https://github.com/francois-marie/awesome-quantum-computing-experiments) (MIT). `qubit_count.csv` supplies the best reported physical qubit count. `entangled_state_error_exp.csv` mixes two-qubit gate error and entangled-state error; the series is the running minimum of that column, and fidelity is one minus the error. This dataset has no quantum-volume column. Some 2024–2026 neutral-atom rows are marked "Not rearranged" (6,100 and 11,000). A second series drops those rows before the cumulative maximum (2024 record 2,088, 2025 record 2,400, carried through 2026).

```bash
python -m scripts.fetch_quantum_hardware
python -m scripts.train_domain_series --config config/quantum_qubits.yaml
python -m scripts.train_domain_series --config config/quantum_qubits_rearranged.yaml
python -m scripts.train_domain_series --config config/quantum_gate_error.yaml
```

Both qubit series and the error series are 29 annual points from 1998 through 2026. The test block is 2022–2026 (5 years). Training took 0.38s, 0.39s, and 0.43s, stopping at epochs 57, 57, and 67. Errors are on the log10 record.

| Series | Model MAE / RMSE | Persistence | Log-linear | Training mean | Train-window time |
| --- | ---: | ---: | ---: | ---: | ---: |
| All qubit records | 1.532 / 1.672 | 0.327 / 0.583 | 1.353 / 1.472 | 2.417 / 2.509 | doubling 4.10 years |
| Excluding unrearranged | 1.226 / 1.293 | 0.194 / 0.366 | 1.047 / 1.097 | 2.111 / 2.151 | doubling 4.10 years |
| Best reported error | 0.385 / 0.508 | 0.155 / 0.266 | 0.211 / 0.220 | 1.618 / 1.656 | halving 2.58 years |

The training-window qubit slope is 0.073 log10 per year (R² 0.765). It is the same for both qubit series because the unrearranged rows sit in the test years. Persistence is the best one-step forecast on all three series. The error slope is −0.116 log10 per year (R² 0.803). The best error in the file is about 8.4×10⁻⁵ in 2025, a fidelity of about 0.9999. The 2024–2026 atom-array jumps sit well above the 1998–2021 trend, which is why the log-linear qubit forecast and the transformer both miss the test block.

**Research activity.** Monthly `quant-ph` counts from the same arXiv fetch, 2012-01 through 2026-09.

```bash
python -m scripts.train_domain_series --config config/quantum_arxiv.yaml
```

Test months are 2024-01 through 2026-09. Training took 1.3 seconds and stopped at epoch 13.

| | MAE | RMSE |
| --- | ---: | ---: |
| Model | 419 | 509 |
| Persistence | 139 | 184 |
| Log-linear | 317 | 403 |
| Training mean | 725 | 782 |

Persistence is the best one-step forecast. The training-window slope is 0.068 nats per year (R² 0.841), a doubling time of 10.3 years, much slower than the CS categories above.

**Quantum randomness.** A sanity check in the same spirit as a lottery draw. The keyless ANU QRNG endpoint did not respond, and the current API requires a key, so it is not used. The series is the concatenated symbol codes from [ProCreations/quantum-randomness](https://huggingface.co/datasets/ProCreations/quantum-randomness) (`quantum_random_1000_final.csv`, MIT): 1,024 short slices attributed to the ANU QRNG, 6,144 symbols over an alphabet of 64. Timestamps in the file are microseconds apart inside one dump. The trainer maps the symbol index onto a synthetic day axis so the shared year code has a defined scale; those dates are not a sampling clock, and a doubling time on them is not a physical result.

```bash
python -m scripts.fetch_quantum_random
python -m scripts.train_domain_series --config config/quantum_random.yaml
```

The test block is 1,226 symbols. Training took 2.5 seconds and stopped at epoch 2.

| | MAE | RMSE |
| --- | ---: | ---: |
| Model | 18.60 | 21.65 |
| Persistence | 25.47 | 31.11 |
| Log-linear | 18.52 | 21.79 |
| Training mean | 18.56 | 21.63 |

The model does not beat the training mean (`beats_constant_mean` is false; RMSE 21.650 against 21.635). The log-linear training R² is about 5.5×10⁻⁵. Persistence is worse, which is what uncorrelated noise should do. The transformer cannot predict the next quantum-random symbol.

**Cosmic-ray neutrons.** The space series above has no neutron-monitor record, so the Oulu station from NMDB is included here. Daily `corr_for_efficiency` values, reduced to months with at least 20 days, from 1964-04 through 2026-10 (751 months). October 2026 is a partial month: the fetch date was 2026-10-21 and the month still cleared the 20-day rule. The query is keyless (`wget=1` on the NMDB nest interface).

We acknowledge the NMDB database (www.nmdb.eu), founded under the European Union's FP7 programme (contract no. 213007), and the PIs of the neutron monitor at Oulu (Sodankylä Geophysical Observatory of the University of Oulu). The series is free for non-commercial research, education, and public services. The station readme is <https://cosmicrays.oulu.fi/readme.html>.

```bash
python -m scripts.fetch_oulu_neutrons
python -m scripts.train_domain_series --config config/quantum_oulu_neutrons.yaml
```

The input window is 36 months, shorter than an ~11-year solar cycle. The test block is 2014-12 through 2026-10 (143 months). Training took 4.3 seconds and stopped at epoch 19.

| | MAE | RMSE |
| --- | ---: | ---: |
| Model | 0.743 | 1.006 |
| Persistence | 0.734 | 1.022 |
| Log-linear | 6.228 | 6.874 |
| Training mean | 6.703 | 7.629 |

The model and persistence are close (the model is slightly better on RMSE and slightly worse on MAE). Both follow the solar cycle. The log-linear fit on the training window has R² 0.021 and a slope of 0.0007 nats per year, so its doubling time of 979 years is not a physical timescale. A straight exponential is the wrong baseline for this series.
