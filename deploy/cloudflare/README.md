# Cloudflare Containers front door

This directory is a Worker named `real-neural-forge-api`. It is not the
dashboard Worker `real-neural-forge` or `youneek-xyz`.

The live API is https://real-neural-forge-api.youneekartifacts.workers.dev.
`wrangler.jsonc` pins `main-<sha>` in the Cloudflare registry. It does not
follow `:latest`. Redeploy steps, the Workers Paid plan, and the required
secrets are in the repository README under **Live hosting**. Run Wrangler
from this directory after `npm install`. Do not move `wrangler.jsonc` to
the repository root, and do not delete the `v1` Durable Object migration.
Do not add a Durable Object `exports` entry next to that migration.

`NeuralForgeContainer` sleeps after 10 minutes without activity. After a
`POST /train` is accepted (or returns 409 because training is already
running), the class schedules `watchTraining` every three minutes. That
callback renews the activity timeout and polls `GET /training/status` until
the run is no longer `running`. Do not replace `Container.alarm()`; the base
class uses it to apply `sleepAfter`.
