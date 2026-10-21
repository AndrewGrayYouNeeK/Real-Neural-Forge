# Real-Neural-Forge

A transformer pipeline for time-series prediction (PyTorch + FastAPI). See `README.md`
for the full product overview, config reference, and API documentation.

## Cursor Cloud specific instructions

### Service overview
There is a single service: a FastAPI app (`src/api:app`) that serves both the JSON REST
API and the static web dashboard (mounted from `frontend/`) at `http://localhost:8000`.
Experiments are persisted to a local SQLite DB (`data/experiments.db`); model checkpoints
are written to `checkpoints/`.

### Python environment
- Dependencies are installed into a virtualenv at `.venv` (the startup update script keeps
  it in sync). Use `.venv/bin/python`, `.venv/bin/pytest`, etc., or activate with
  `source .venv/bin/activate`. There is no `uv`/`poetry` — plain `pip` + `venv`.
- `torch` is intentionally the CPU-only build (installed from the PyTorch CPU wheel index).
  The app/config default to `device: cpu`; there is no GPU in this environment and the code
  already falls back to CPU automatically.

### Lint / type-check / test / run
Standard commands are documented in `README.md`; run them via the venv:
- Tests: `.venv/bin/pytest` (236 tests, all passing).
- Lint: `.venv/bin/ruff check src/ tests/ workers_ci/`.
- Type check: `.venv/bin/mypy src/ workers_ci/ --ignore-missing-imports`.
- Train (creates `checkpoints/best_model.pt`, ~1 min on CPU): `.venv/bin/python -m src.train --config config/config.yaml`.
- Retrain locally: `.venv/bin/python -m src.retrain --all --max-epochs 8`. Champions land in `champions/`, forecast JSON in `forecasts/`. The weekly workflow publishes the same files to the `champions` GitHub Release.
- Run dev server: `.venv/bin/uvicorn src.api:app --reload --host 0.0.0.0 --port 8000`.

### Non-obvious caveats
- Runtime and dev dependencies are locked in `requirements.txt` and `requirements-dev.txt`
  (generated from `requirements.in` / `requirements-dev.in`). The lock is the newest set that
  still installs on Python 3.10–3.13, and it pins the CPU torch wheel. The lock
  also editable-installs `workers_ci` (`-e ./workers_ci`). That hook writes
  `wrangler.jsonc` only when `WRANGLER_CI_OVERRIDE_NAME` is set; Docker and other
  normal installs leave the tree unchanged. CI runs mypy with
  `--python-version 3.12`.
- The API loads a model on startup. `NEURAL_FORGE_CHECKPOINT_URL` downloads a champion
  first when it is set. If `checkpoints/best_model.pt` is missing and that variable is
  unset, the server uses an **untrained** model (predictions are effectively random) and
  `/model/info` reports `checkpoint_loaded: false`.
- `deploy/cloudflare/` is the Containers Worker `real-neural-forge-api`. Do not add a
  root `wrangler.jsonc`; `workers_ci` writes that file only when
  `WRANGLER_CI_OVERRIDE_NAME` is set, for the `real-neural-forge` and `youneek-xyz`
  dashboard builds.
- The dashboard is served by the same FastAPI process at `/`; static assets are mounted at
  `/assets`. No separate frontend build/server is needed.
