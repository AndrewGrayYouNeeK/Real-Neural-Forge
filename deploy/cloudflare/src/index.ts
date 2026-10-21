import { Container, getContainer } from "@cloudflare/containers";

// Hand-written to match wrangler.jsonc. `npx wrangler types` can replace it.
interface Env {
  NEURAL_FORGE: Parameters<typeof getContainer>[0];
  NEURAL_FORGE_CONFIG?: string;
  NEURAL_FORGE_CHECKPOINT_URL?: string;
  NEURAL_FORGE_CHECKPOINT_TOKEN?: string;
  NEURAL_FORGE_API_TOKEN?: string;
}

function containerEnv(env: Env): Record<string, string> {
  const source: Record<string, string | undefined> = {
    NEURAL_FORGE_CONFIG: env.NEURAL_FORGE_CONFIG,
    NEURAL_FORGE_CHECKPOINT_URL: env.NEURAL_FORGE_CHECKPOINT_URL,
    NEURAL_FORGE_CHECKPOINT_TOKEN: env.NEURAL_FORGE_CHECKPOINT_TOKEN,
    NEURAL_FORGE_API_TOKEN: env.NEURAL_FORGE_API_TOKEN,
  };
  return Object.fromEntries(
    Object.entries(source).filter((entry): entry is [string, string] => {
      return typeof entry[1] === "string" && entry[1].length > 0;
    }),
  );
}

export class NeuralForgeContainer extends Container {
  defaultPort = 8000;
  sleepAfter = "10m";
  enableInternet = true;
  // The image serves FastAPI. Readiness is GET /health, not the class default.
  pingEndpoint = "localhost/health";
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const container = getContainer(env.NEURAL_FORGE, "primary");
    await container.startAndWaitForPorts({
      ports: 8000,
      startOptions: { envVars: containerEnv(env) },
    });
    return container.fetch(request);
  },
};
