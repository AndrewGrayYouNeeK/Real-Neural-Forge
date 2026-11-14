# Real-Neural-Forge roadmap

Shipped and planned product phases. Constraints that stay fixed across phases:
GPT-OSS 120B primary, Unhinged mode unchanged, Talk / GQ voice, hidden thinking
text, ~25k neurons/day global cost caps, no lottery, no container size increase,
and never deploy the frontend Worker `real-neural-forge`.

## Phase 1 — Agentic core (shipped)

- Tool-using `/agent/chat` with forecast, live data, evaluation, workspace list
- Grok-like dashboard shell and live tool registry
- Weather °F conversion in tool payloads; Cache API key versioning

## Phase 2 — Interactive & stateful (in progress)

1. **Per-user memory** — D1 `user_memory` (migration `0006_*`); chat tools
   `remember_fact` / `recall_memory` / `forget_memory`; view/clear in the Memory
   tool panel; owner, signed-in, and visitor scopes.
2. **Persistent context** — D1 `chat_threads` / `chat_messages` (migration
   `0007_*`); sidebar Chats list loads saved threads; reopening restores
   messages and continues `/agent/chat` with `thread_id` context.
3. **Saved scenarios** — D1 `saved_scenarios` (migration `0008_*`); save/list/run
   via agent tools and the Memory panel; re-run opens chat with the stored
   question/series.
4. **Manage your own series** — workspace UI + `PATCH /workspace/datasets/{id}`
   rename; delete and retrain actions on uploaded CSV series (Train on my CSV).

Flip matching “coming soon” tool-registry entries to live as each piece ships
(Memory first; saved scenarios when that PR lands).

## Phase 3 — Explainability & trust

- **PCA column analysis (in progress in this track)** — eigenvectors of the
  covariance matrix for multi-column CSV uploads; explained variance and
  co-moving groups; optional `data.pca_reduce` / `data.pca_components` before
  training; API + `analyze_pca` agent tool; landing registry stays **In progress**.
  Feeds the later **Why this forecast** narratives.
- **K-Means regimes / anomalies (in progress in this track)** — cluster windows
  or multi-column rows into `k≤8` regimes (auto-pick via silhouette); plain
  labels (e.g. calm/volatile, low/high); flag points far from centroids. API +
  `analyze_regimes` agent tool; landing registry stays **In progress**. Anomaly
  flags feed later **Alerts**.
- Why-this-forecast narratives grounded in model/features (still coming soon)
- Shareable accuracy / forecast reports
- Threshold alerts on series (fed by K-Means anomaly flags)

## Phase 4 — Synthetic & counterfactual

- Synthetic training series with known patterns
- What-if shocks on a forecast path
- **Diffusion forecaster experiment (selectable, not default)** — small 1D
  conditional denoising diffusion `f_θ(x_t, t, c)` → many futures → p10/p50/p90
  bands; train offline/CPU-small with few inference steps; compare via
  backtests/metrics against the transformer before any default swap
  (`config/diffusion_experiment.yaml`, `model.name: diffusion_forecaster`)

## Phase 5 — World model / simulator

- Interactive multi-step simulator for rolling scenarios forward
- **Diffusion forecaster experiment** (shared with Phase 4 synthetic path)
- **JEPA-style (joint-embedding predictive) self-supervised pretraining
  experiment for time series** — learn latent representations by predicting
  embeddings of future (or masked) windows rather than raw values; evaluate
  transfer into the existing supervised forecast fine-tune path before any
  production swap
