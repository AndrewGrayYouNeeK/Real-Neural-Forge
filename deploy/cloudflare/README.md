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

Visitor accounts, sessions, and workspace metadata are in the D1 database
`neural-forge-workspaces` (`WORKSPACES_DB`). Dataset and checkpoint bytes are
in the R2 bucket of the same name (`WORKSPACES`). Create them once with the
commands in the repository README under **Live hosting**. The container
reaches those bindings only through `/_bindings/sql` and `/_bindings/object`,
which require `NEURAL_FORGE_STORAGE_SECRET`.

`routes` attach `forge.youneek.xyz` and `forge.youneek.dev`. `workers_dev`
stays true. Session cookies are host-only; this Worker does not set a cookie
`Domain`.

`NeuralForgeContainer` sleeps after 10 minutes without activity. After a
`POST /train` is accepted (or returns 409 because training is already
running), the class schedules `watchTraining` every three minutes. That
callback renews the activity timeout and polls `GET /training/status` until
the run is no longer `running`. Do not replace `Container.alarm()`; the base
class uses it to apply `sleepAfter`.
