# Workers AI model comparison for Nous (2026-11-27)

Same 24 questions, with Nous's production `SYSTEM_PROMPT` (from
`deploy/cloudflare/src/live/ai.ts`), run on 13 Cloudflare Workers AI text
models. Nothing here changes the production model.

Page with the leaderboard, picks and side-by-side answers:
https://cursor.com/canvas/store-70c6888f-701f-45da-9684-46650e9bcd30/62a88da2-4347-4bdf-b847-1d81b456796a

## Files

| File | What it is |
| --- | --- |
| `compare.py` | Runner. Warm-up call, 24 prompts, 3 streamed calls for first-word time, plus thinking-on reruns of 4 hard prompts. |
| `analyze.py` | Turns `results.json` into `scores.json` (script checks for math, code, creator wording, plus hand-reviewed judgments listed at the top of the file). |
| `extract_prompt.mjs` | Pulls the production system prompt into `system_prompt.txt`. |
| `worker/` | Throwaway Worker with an `AI` binding. Run only with `wrangler dev --remote`; never deployed. |
| `results.json` | Raw answers, latency, tokens and neurons for every call. |
| `scores.json` | Final scores per model. |

## Re-run

```bash
cd research/model-compare/worker && npx wrangler dev --remote --port 8799 &
cd .. && python3 compare.py            # all models, or: python3 compare.py llama
python3 analyze.py
```

## Method notes

- Temperature 0.85 (production `NORMAL_TEMPERATURE`), 500 max tokens (production uses 160).
- Hidden thinking is off or low for every model; GPT-OSS runs exactly as production does (Responses API, low effort).
- Cost is the `usage.neurons` figure Cloudflare returns on each call; $0.011 per 1,000 neurons.
- Retired on 2026-11-20 and not benchmarked: deepseek-v4-pro-0813, deepseek-v4-flash-0731, kimi-k2.6,
  glm-5.3-flash, nemotron-3-120b-a12b, gemma-7b-it-lora (Workers AI error 5028).
- `deepseek-v4.1-flash-1017`, `inkling-2-small`, `llama-3.3-70b-instruct-fp8-fast` and
  `llama-3.1-8b-instruct-fp8-fast` were added because they are the live replacements or the current fallback.
- Scores for funny, tone and balance are judgment calls after reading every answer.
