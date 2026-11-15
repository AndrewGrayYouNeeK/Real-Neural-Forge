# Real-Neural-Forge roadmap

Shipped and planned product phases. Constraints that stay fixed across phases:
GPT-OSS 120B primary, Unhinged mode unchanged, Talk / GQ voice, hidden thinking
text, ~25k neurons/day global cost caps, no lottery, no container size increase,
and never deploy the frontend Worker `real-neural-forge`.

## Phase 1 — Agentic core (shipped)

- Tool-using `/agent/chat` with forecast, live data, evaluation, workspace list
- Grok-like dashboard shell and live tool registry
- Weather °F conversion in tool payloads; Cache API key versioning

## Phase 2 — Interactive & stateful (shipped)

1. **Per-user memory** — D1 `user_memory`; chat tools + Memory panel.
2. **Persistent context** — D1 chat threads; sidebar restore with `thread_id`.
3. **Saved scenarios** — D1 `saved_scenarios`; save/list/run tools + panel.
4. **Manage your own series** — rename / delete / retrain uploaded CSVs.

## Foundations that feed Phase 3+

- **PCA column analysis (shipped)** — eigenvectors / co-moving groups;
  `analyze_pca` + optional `data.pca_reduce`. Feeds **Why this forecast**.
- **K-Means regimes / anomalies (shipped)** — `analyze_regimes`; anomaly flags
  feed later **Alerts**.
- **Diffusion forecaster experiment (shipped; selectable, not default)** —
  `config/diffusion_experiment.yaml`.
- **JEPA SSL experiment (shipped; selectable, not default)** —
  `config/jepa_experiment.yaml`.

## Phase 3 — Generative Layer (in progress)

1. **Why-this-forecast** — short plain-English explanation under each prediction
   from recent trend / seasonality, plus PCA drivers and K-Means regimes when
   available (`POST /analysis/why`, `why` on starter / predict-latest payloads,
   `explain_prediction` agent tool). Landing tool **Live**.
2. **What-if** — shock the last input on a starter or saved-scenario series
   (`POST /analysis/what-if`, workspace `…/what-if`, agent `what_if`); compare
   baseline vs shocked path. Landing tool **Live**.
3. **Reports** — one-click shareable summary (forecast, accuracy, regimes).
4. **Alerts** — threshold or unusual-point alerts from K-Means flags (in-app).
5. **Synthetic data** — generate realistic sample series (diffusion can help).

## Phase 4 — Multimodal

- Charts, images, and text in one workflow

## Phase 5 — World model / simulator

- Interactive multi-step simulator for rolling scenarios forward
