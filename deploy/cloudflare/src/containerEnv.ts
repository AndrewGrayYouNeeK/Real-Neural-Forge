// The container calls http://bindings.neuralforge/_bindings/... . That host
// never leaves the Worker: outboundByHost runs in index.ts with the D1 and R2
// bindings, so signup does not depend on a public workers.dev round trip.

// Secrets set on the Worker are only visible to Worker code. The FastAPI
// process inside the container reads them from its own environment, so they
// must be listed here and passed through startOptions.envVars. The voice id
// secret predates the NEURAL_FORGE_ prefix (FORGE_VOICE_ID) and the container
// reads NEURAL_FORGE_VOICE_ID, so it is aliased here rather than renamed —
// renaming would require re-entering the secret with `wrangler secret put`.

export const BINDINGS_HOST = "bindings.neuralforge";

const CONTAINER_ENV_KEYS = [
  "NEURAL_FORGE_CONFIG",
  "NEURAL_FORGE_CHECKPOINT_URL",
  "NEURAL_FORGE_CHECKPOINT_TOKEN",
  "NEURAL_FORGE_API_TOKEN",
  "NEURAL_FORGE_STORAGE",
  "NEURAL_FORGE_BINDINGS_URL",
  "NEURAL_FORGE_STORAGE_SECRET",
  "NEURAL_FORGE_OWNER_EMAIL",
  "NEURAL_FORGE_MAX_DATASET_BYTES",
  "NEURAL_FORGE_MAX_DATASETS",
  "NEURAL_FORGE_MAX_CONFIG_BYTES",
  "NEURAL_FORGE_MAX_EPOCHS",
  "NEURAL_FORGE_MAX_RUN_SECONDS",
  "NEURAL_FORGE_MAX_SAMPLES",
  "NEURAL_FORGE_MAX_SEQ_LEN",
  "NEURAL_FORGE_GLOBAL_QUEUE_LIMIT",
  "NEURAL_FORGE_MAX_PREDICTIONS",
  "NEURAL_FORGE_MAX_PREDICTION_POINTS",
  "NEURAL_FORGE_MAX_CHECKPOINT_BYTES",
  "NEURAL_FORGE_MAX_ACTIVE_JOBS",
  "NEURAL_FORGE_MAX_BATCH_SIZE",
  "NEURAL_FORGE_QUEUE_RUNNER",
  "ELEVENLABS_API_KEY",
  "NEURAL_FORGE_VOICE_ID",
  "STRIPE_SECRET_KEY",
  "STRIPE_WEBHOOK_SECRET",
] as const;

export function containerEnv<T extends object>(env: T): Record<string, string> {
  const source = env as unknown as Record<string, string | undefined>;
  const vars = Object.fromEntries(
    CONTAINER_ENV_KEYS.flatMap((key) => {
      const value = source[key];
      return typeof value === "string" && value.length > 0 ? [[key, value]] : [];
    }),
  );
  if (!vars.NEURAL_FORGE_BINDINGS_URL) {
    vars.NEURAL_FORGE_BINDINGS_URL = `http://${BINDINGS_HOST}`;
  }
  if (!vars.NEURAL_FORGE_VOICE_ID) {
    const alias = source.FORGE_VOICE_ID;
    if (typeof alias === "string" && alias.length > 0) {
      vars.NEURAL_FORGE_VOICE_ID = alias;
    }
  }
  return vars;
}
