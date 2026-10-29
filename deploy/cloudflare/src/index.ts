import { Container, ContainerProxy, getContainer, type State } from "@cloudflare/containers";

export { ContainerProxy };

// The container calls http://bindings.neuralforge/_bindings/... . That host
// never leaves the Worker: outboundByHost runs here with the D1 and R2
// bindings, so signup does not depend on a public workers.dev round trip.

// Hand-written to match wrangler.jsonc. `npx wrangler types` can replace it.
interface D1Prepared {
  bind(...values: unknown[]): D1Prepared;
  all(): Promise<{ results?: Record<string, unknown>[]; meta?: { changes?: number } }>;
}

interface D1Database {
  prepare(sql: string): D1Prepared;
  batch(
    statements: D1Prepared[],
  ): Promise<Array<{ results?: Record<string, unknown>[]; meta?: { changes?: number } }>>;
}

interface R2ObjectBody {
  body: ReadableStream;
  size: number;
}

interface R2Bucket {
  put(key: string, value: ArrayBuffer | ArrayBufferView): Promise<unknown>;
  get(key: string): Promise<R2ObjectBody | null>;
  delete(key: string): Promise<void>;
}

interface Env {
  NEURAL_FORGE: Parameters<typeof getContainer>[0];
  WORKSPACES_DB: D1Database;
  WORKSPACES: R2Bucket;
  NEURAL_FORGE_CONFIG?: string;
  NEURAL_FORGE_CHECKPOINT_URL?: string;
  NEURAL_FORGE_CHECKPOINT_TOKEN?: string;
  NEURAL_FORGE_API_TOKEN?: string;
  NEURAL_FORGE_STORAGE?: string;
  NEURAL_FORGE_BINDINGS_URL?: string;
  NEURAL_FORGE_STORAGE_SECRET?: string;
  NEURAL_FORGE_OWNER_EMAIL?: string;
  NEURAL_FORGE_MAX_DATASET_BYTES?: string;
  NEURAL_FORGE_MAX_DATASETS?: string;
  NEURAL_FORGE_MAX_CONFIG_BYTES?: string;
  NEURAL_FORGE_MAX_EPOCHS?: string;
  NEURAL_FORGE_MAX_RUN_SECONDS?: string;
  NEURAL_FORGE_MAX_SAMPLES?: string;
  NEURAL_FORGE_MAX_SEQ_LEN?: string;
  NEURAL_FORGE_GLOBAL_QUEUE_LIMIT?: string;
  NEURAL_FORGE_MAX_PREDICTIONS?: string;
  NEURAL_FORGE_MAX_PREDICTION_POINTS?: string;
  NEURAL_FORGE_MAX_CHECKPOINT_BYTES?: string;
  NEURAL_FORGE_MAX_ACTIVE_JOBS?: string;
  NEURAL_FORGE_MAX_BATCH_SIZE?: string;
  NEURAL_FORGE_QUEUE_RUNNER?: string;
}

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
] as const;

// Binding uploads stop here even if a limit env var is set higher.
const BINDINGS_HOST = "bindings.neuralforge";
const BINDING_OBJECT_BYTES = 64 * 1024 * 1024;
const OBJECT_KEY = /^((bases\/starters\/[a-z0-9-]+\/(dataset\.csv|model\.pt))|(users\/[A-Za-z0-9_]+\/(datasets|models)\/[A-Za-z0-9_.-]+))$/;

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
  return vars;
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
      const headers = new Headers();
      if (this.env.NEURAL_FORGE_API_TOKEN) {
        headers.set("X-Neural-Forge-Watch", this.env.NEURAL_FORGE_API_TOKEN);
      }
      const response = await this.containerFetch(
        "http://localhost/training/status",
        { method: "GET", headers },
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

async function digest(value: string): Promise<Uint8Array> {
  const hashed = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value));
  return new Uint8Array(hashed);
}

async function authorized(request: Request, secret: string | undefined): Promise<boolean> {
  if (!secret) {
    return false;
  }
  const header = request.headers.get("Authorization") ?? "";
  const match = /^Bearer\s+(\S+)$/i.exec(header);
  if (!match) {
    return false;
  }
  const [left, right] = await Promise.all([digest(match[1]), digest(secret)]);
  let mismatch = 0;
  for (let index = 0; index < left.length; index += 1) {
    mismatch |= left[index] ^ right[index];
  }
  return mismatch === 0;
}

function objectKey(request: Request): string | null {
  const key = request.headers.get("X-Object-Key") ?? "";
  if (!OBJECT_KEY.test(key) || key.includes("..") || key.includes("\\")) {
    return null;
  }
  return key;
}

async function handleBindings(request: Request, env: Env): Promise<Response> {
  if (!(await authorized(request, env.NEURAL_FORGE_STORAGE_SECRET))) {
    return Response.json({ error: "Unauthorized." }, { status: 401 });
  }
  const url = new URL(request.url);
  if (url.pathname === "/_bindings/sql") {
    if (request.method !== "POST") {
      return Response.json({ error: "Method not allowed." }, { status: 405 });
    }
    return handleSql(request, env);
  }
  if (url.pathname === "/_bindings/object") {
    return handleObject(request, env);
  }
  return Response.json({ error: "Not found." }, { status: 404 });
}

async function handleSql(request: Request, env: Env): Promise<Response> {
  let payload: { statements?: Array<{ sql?: unknown; params?: unknown }> };
  try {
    payload = (await request.json()) as { statements?: Array<{ sql?: unknown; params?: unknown }> };
  } catch {
    return Response.json({ error: "Expected JSON." }, { status: 400 });
  }
  const statements = payload.statements;
  if (!Array.isArray(statements) || statements.length === 0 || statements.length > 40) {
    return Response.json({ error: "Expected statements." }, { status: 400 });
  }
  try {
    const prepared = statements.map((statement) => {
      if (typeof statement.sql !== "string" || statement.sql.trim().length === 0) {
        throw new Error("Missing SQL.");
      }
      const query = env.WORKSPACES_DB.prepare(statement.sql);
      const params = Array.isArray(statement.params) ? statement.params : [];
      return params.length > 0 ? query.bind(...params) : query;
    });
    const results = await env.WORKSPACES_DB.batch(prepared);
    return Response.json({
      results: results.map((item) => ({
        rows: item.results ?? [],
        changes: item.meta?.changes ?? 0,
      })),
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return Response.json({ error: message.slice(0, 500) }, { status: 400 });
  }
}

async function handleObject(request: Request, env: Env): Promise<Response> {
  const key = objectKey(request);
  if (key === null) {
    return Response.json({ error: "Invalid object key." }, { status: 400 });
  }
  if (request.method === "GET") {
    const object = await env.WORKSPACES.get(key);
    if (object === null) {
      return new Response("Not found.", { status: 404 });
    }
    return new Response(object.body, {
      headers: { "Content-Type": "application/octet-stream" },
    });
  }
  if (request.method === "PUT") {
    const body = await request.arrayBuffer();
    if (body.byteLength > BINDING_OBJECT_BYTES) {
      return Response.json({ error: "Object is too large." }, { status: 413 });
    }
    await env.WORKSPACES.put(key, body);
    return new Response(null, { status: 204 });
  }
  if (request.method === "DELETE") {
    if (key.startsWith("bases/")) {
      return Response.json({ error: "Shared starter objects stay in place." }, { status: 403 });
    }
    await env.WORKSPACES.delete(key);
    return new Response(null, { status: 204 });
  }
  return Response.json({ error: "Method not allowed." }, { status: 405 });
}

// Intercept the container's binding calls before they can reach the public
// internet. ContainerProxy must be exported for this to run.
NeuralForgeContainer.outboundByHost = {
  [BINDINGS_HOST]: (request: Request, env: Env) => handleBindings(request, env),
};

function forwardedRequest(request: Request): Request {
  const url = new URL(request.url);
  const headers = new Headers(request.headers);
  headers.set("X-Forwarded-Proto", url.protocol.replace(/:$/, ""));
  headers.set("X-Forwarded-Host", url.host);
  // The training watch header is added only by the container Durable Object.
  headers.delete("X-Neural-Forge-Watch");
  return new Request(request, { headers });
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname === "/_bindings/sql" || url.pathname === "/_bindings/object") {
      return handleBindings(request, env);
    }
    const container = getContainer(env.NEURAL_FORGE, "primary");
    await container.startAndWaitForPorts({
      ports: 8000,
      startOptions: { envVars: containerEnv(env) },
    });
    return container.fetch(forwardedRequest(request));
  },
};
