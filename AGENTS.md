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
- Tests: `.venv/bin/pytest` (54 tests, all passing).
- Lint: `.venv/bin/ruff check src/ tests/`.
- Type check: `.venv/bin/mypy src/ --ignore-missing-imports`.
- Train (creates `checkpoints/best_model.pt`, ~1 min on CPU): `.venv/bin/python -m src.train --config config/config.yaml`.
- Run dev server: `.venv/bin/uvicorn src.api:app --reload --host 0.0.0.0 --port 8000`.

### Non-obvious caveats
- Because dependency versions are unpinned (`>=`), the latest `ruff`/`mypy`/`numpy` get
  installed. With these current versions there are pre-existing findings unrelated to
  environment setup: `ruff` reports unused imports / unsorted imports in `tests/test_api.py`
  and `tests/test_train.py`, and `mypy` errors on numpy's stubs (`type` statement requires
  the target set higher than the configured `python_version = 3.10`). These are code/version
  concerns, not environment breakage — the tools themselves run correctly.
- The API loads a model on startup: if `checkpoints/best_model.pt` is missing it serves an
  **untrained** model (predictions are effectively random) and `/model/info` reports
  `checkpoint_loaded: false`. Run training once to get a meaningful checkpoint.
- The dashboard is served by the same FastAPI process at `/`; static assets are mounted at
  `/assets`. No separate frontend build/server is needed.
