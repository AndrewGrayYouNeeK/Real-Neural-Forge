import { Container, getContainer, type State } from "@cloudflare/containers";

// Hand-written to match wrangler.jsonc. `npx wrangler types` can replace it.
interface Env {
  NEURAL_FORGE: Parameters<typeof getContainer>[0];
  NEURAL_FORGE_CONFIG?: string;
  NEURAL_FORGE_CHECKPOINT_URL?: string;
  NEURAL_FORGE_CHECKPOINT_TOKEN?: string;
  NEURAL_FORGE_API_TOKEN?: string;
}

// Idle sleep stays at 10 minutes. Poll well inside that window so a training
// run with no client traffic still counts as activity.
const TRAINING_POLL_SECONDS = 3 * 60;
const TRAINING_WATCH_CALLBACK = "watchTraining";
const TRAINING_WATCH_FAILURE_LIMIT = 4;

type TrainingWatchPayload = {
  failures?: number;
};

type ContainerStatus = State["status"];

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

export class NeuralForgeContainer extends Container<Env> {
  defaultPort = 8000;
  sleepAfter = "10m";
  enableInternet = true;
  // The image serves FastAPI. Readiness is GET /health, not the class default.
  pingEndpoint = "localhost/health";

  // Container owns alarm() for sleepAfter. A scheduled poll is the supported
  // way to renew activity while POST /train's background job is still running.
  override async fetch(request: Request): Promise<Response> {
    const response = await super.fetch(request);
    if (trainingStartAccepted(request, response.status)) {
      try {
        await this.armTrainingWatch();
      } catch (error) {
        console.error(
          JSON.stringify({
            event: "training.watch_arm_failed",
            error: error instanceof Error ? error.message : String(error),
          }),
        );
      }
    }
    return response;
  }

  async watchTraining(payload?: TrainingWatchPayload): Promise<void> {
    this.renewActivityTimeout();
    const failures = payload?.failures ?? 0;
    const containerStatus = await this.containerStatus();
    if (containerStatus === "unknown") {
      await this.continueTrainingWatch(failures + 1, "container state unavailable");
      return;
    }
    if (!containerAcceptsRequests(containerStatus)) {
      console.log(
        JSON.stringify({
          event: "training.watch_stopped",
          reason: "container_down",
          containerStatus,
        }),
      );
      return;
    }

    try {
      const response = await this.containerFetch(
        "http://localhost/training/status",
        { method: "GET" },
      );
      if (!response.ok) {
        await response.body?.cancel();
        await this.continueTrainingWatch(failures + 1, `HTTP ${response.status}`);
        return;
      }
      const body = (await response.json()) as { status?: unknown };
      if (body.status === "running") {
        this.renewActivityTimeout();
        await this.continueTrainingWatch(0);
        return;
      }
      console.log(
        JSON.stringify({
          event: "training.watch_stopped",
          status: typeof body.status === "string" ? body.status : "unknown",
        }),
      );
    } catch (error) {
      await this.continueTrainingWatch(failures + 1, error);
    }
  }

  private async armTrainingWatch(): Promise<void> {
    const pending = await this.listSchedules(TRAINING_WATCH_CALLBACK);
    if (pending.length > 0) {
      this.renewActivityTimeout();
      return;
    }
    await this.continueTrainingWatch(0);
  }

  private async continueTrainingWatch(
    failures: number,
    error?: unknown,
  ): Promise<void> {
    if (error !== undefined) {
      console.log(
        JSON.stringify({
          event: "training.watch_error",
          failures,
          error: error instanceof Error ? error.message : String(error),
        }),
      );
    }
    if (failures >= TRAINING_WATCH_FAILURE_LIMIT) {
      console.log(
        JSON.stringify({
          event: "training.watch_stopped",
          reason: "status_unavailable",
        }),
      );
      return;
    }
    this.renewActivityTimeout();
    await this.schedule<TrainingWatchPayload>(
      TRAINING_POLL_SECONDS,
      TRAINING_WATCH_CALLBACK,
      { failures },
    );
  }

  private async containerStatus(): Promise<ContainerStatus | "unknown"> {
    try {
      const state = await this.getState();
      return state.status;
    } catch (error) {
      console.log(
        JSON.stringify({
          event: "training.watch_error",
          error: error instanceof Error ? error.message : String(error),
        }),
      );
      return "unknown";
    }
  }
}

function trainingStartAccepted(request: Request, status: number): boolean {
  if (request.method !== "POST") {
    return false;
  }
  const pathname = new URL(request.url).pathname.replace(/\/+$/, "");
  if ((pathname.length > 0 ? pathname : "/") !== "/train") {
    return false;
  }
  return status === 200 || status === 409;
}

function containerAcceptsRequests(status: ContainerStatus): boolean {
  switch (status) {
    case "healthy":
    case "running":
      return true;
    case "stopping":
    case "stopped":
    case "stopped_with_code":
      return false;
    default: {
      const unexpected: never = status;
      throw new Error(`Unhandled container status: ${String(unexpected)}`);
    }
  }
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
