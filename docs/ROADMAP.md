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

## Phase 3 — Generative Layer (shipped)

1. **Why-this-forecast** — plain-English explanations under predictions.
2. **What-if** — shock last input; compare baseline vs shocked path.
3. **Reports** — shareable forecast / accuracy / regimes snapshots.
4. **Alerts** — in-app threshold and K-Means anomaly alerts (no email/SMS).
5. **Synthetic data** — `POST /analysis/synthetic` (sine/trend/seasonal/noise/walk
   + optional tiny diffusion); download CSV / Train-on-my-CSV; agent
   `generate_synthetic_series`. Landing **Live**.

## Phase 4 — Multimodal

- Charts, images, and text in one workflow

## Phase 5 — World model / simulator

- Interactive multi-step simulator for rolling scenarios forward
