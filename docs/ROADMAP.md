# Nous roadmap

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

## Phase 4 — Multimodal (shipped)

Charts, images, and text in one workflow.

1. **Read images in chat** — attach (🖼), paste, or drop a PNG/JPEG/WebP/GIF;
   `POST /agent/vision` describes it with Workers AI
   `@cf/google/gemma-4-26b-a4b-it`, and GPT-OSS 120B writes the reply.
2. **Chart to series** — chart or table numbers come back as a series
   (API °C for temperatures, °F on screen) with a quick damped-trend forecast,
   Download CSV, and Train on this series (32+ points).
3. **Caps** — 5 images/day per visitor, 120/day for all visitors, 60/day for
   the owner; every image also counts toward the 25k neuron/day safety cap.
   Landing **Live**.

## Phase 5 — World model / simulator (shipped)

Interactive multi-step simulator for rolling scenarios forward.

1. **Learned latent world model** — `POST /analysis/simulate` fits a small
   model to one series on the CPU container: PCA encoder over the recent window
   (latent state), ridge dynamics, and K-Means regimes that shape the rollout
   noise. Series with 1.5+ years of daily dates (weather) also get an annual
   seasonal baseline. Rollouts are damped to stay stable and clipped to a
   plausible range; counts and fractions stay non-negative.
2. **Scenarios** — up to 3 scenarios × 4 events: one-off shock, sustained
   drift per step, volatility multiplier, or forcing a value. Baseline and
   scenarios share the same random draws, so deltas come from the events.
3. **Honest backtest** — every run reports a rolling-origin holdout (up to 6
   origins) against persistence, plus 80% range coverage.
4. **Where** — Simulator panel (Play / step scrubber), agent tool
   `simulate_world` (weather in °F), and **Simulate scenarios** on chart-to-series
   replies. No new Workers AI calls, no D1 migration, no container change.
   Landing **Live**.
