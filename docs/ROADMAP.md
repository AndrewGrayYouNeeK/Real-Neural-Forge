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

1. **Per-user memory** — D1 `user_memory`; chat tools remember / recall / forget;
   Memory tool panel (owned by a separate Phase 2 PR).
2. **Persistent context** — chat history per user in D1; Grok sidebar Chats list
   reopens a thread into `/agent/chat`.
3. **Saved scenarios** — named forecast/question (series, horizon, settings);
   re-run from the sidebar.
4. **Manage your own series** — list / rename / delete uploaded CSV series and
   retrain (on top of Train on my CSV).

Flip matching “coming soon” tool-registry entries to live as each piece ships
(Memory first; saved scenarios when that PR lands).

## Phase 3 — Explainability & trust

- **PCA column analysis (shipped in this track)** — eigenvectors of the
  covariance matrix for multi-column CSV uploads; explained variance and
  co-moving groups; optional `data.pca_reduce` / `data.pca_components` before
  training; API + `analyze_pca` agent tool; live “Column analysis (PCA)” tool.
  Feeds the later **Why this forecast** narratives.
- Why-this-forecast narratives grounded in model/features (still coming soon)
- Shareable accuracy / forecast reports
- Threshold alerts on series

## Phase 4 — Synthetic & counterfactual

- Synthetic training series with known patterns
- What-if shocks on a forecast path
- **Diffusion forecaster experiment** (also tracked under Phase 5) — small 1D
  denoising diffusion conditioned on recent history → p10/p50/p90 bands; train
  offline/CPU-small; selectable vs transformer via backtests; not default unless
  it wins

## Phase 5 — World model / simulator

- Interactive multi-step simulator for rolling scenarios forward
- **Diffusion forecaster experiment** (shared with Phase 4 synthetic path)
- **JEPA-style (joint-embedding predictive) self-supervised pretraining
  experiment for time series** — learn latent representations by predicting
  embeddings of future (or masked) windows rather than raw values; evaluate
  transfer into the existing supervised forecast fine-tune path before any
  production swap

