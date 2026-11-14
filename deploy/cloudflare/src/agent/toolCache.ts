/**
 * Cache API helpers for agent container tools (starter forecast + eval).
 * Starter forecasts are deterministic for a given model file; serve instantly
 * and refresh in the background via waitUntil when a refresh hook is provided.
 */

import type { D1Like } from "../live/ai";

export const FORECAST_CACHE_TTL_SECONDS = 60 * 60;
export const EVAL_CACHE_TTL_SECONDS = 6 * 60 * 60;
/**
 * Bump whenever agent tool cache payload shape/units change (e.g. weather °C → °F).
 * Invalidates both forecast and eval Cache API entries in one place.
 */
export const AGENT_TOOL_CACHE_VERSION = 2;
const CACHE_ORIGIN = "https://agent-tool-cache.forge.invalid/";

export type WaitUntil = (promise: Promise<unknown>) => void;

export function forecastCacheKey(series: string, horizon: number, points = 120): string {
  return `forecast:v${AGENT_TOOL_CACHE_VERSION}:${series}:h${horizon}:p${points}`;
}

export function evalCacheKey(modelId: string, horizon: number): string {
  return `eval:v${AGENT_TOOL_CACHE_VERSION}:${modelId}:h${horizon}`;
}

function cacheRequest(key: string): Request {
  return new Request(`${CACHE_ORIGIN}${encodeURIComponent(key)}`);
}

export async function cacheGetJson<T>(key: string): Promise<T | null> {
  const cache = globalThis.caches?.default;
  if (!cache) return null;
  try {
    const hit = await cache.match(cacheRequest(key));
    if (!hit) return null;
    return (await hit.json()) as T;
  } catch {
    return null;
  }
}

export async function cachePutJson(key: string, data: unknown, ttlSeconds: number): Promise<void> {
  const cache = globalThis.caches?.default;
  if (!cache) return;
  try {
    await cache.put(
      cacheRequest(key),
      new Response(JSON.stringify(data), {
        headers: {
          "Content-Type": "application/json",
          "Cache-Control": `public, max-age=${ttlSeconds}`,
        },
      }),
    );
  } catch {
    // Best-effort; a miss only means an extra container call.
  }
}

/**
 * Return a cached value immediately. When `refresh` is set and `waitUntil`
 * is available, schedule a background refresh that updates the cache.
 */
export async function cacheGetOrLoad<T>(options: {
  key: string;
  ttlSeconds: number;
  load: () => Promise<T>;
  waitUntil?: WaitUntil;
  /** When true, still schedule a background refresh on hit. */
  refreshOnHit?: boolean;
}): Promise<{ value: T; cacheHit: boolean }> {
  const cached = await cacheGetJson<T>(options.key);
  if (cached != null) {
    if (options.refreshOnHit && options.waitUntil) {
      options.waitUntil(
        (async () => {
          try {
            const fresh = await options.load();
            await cachePutJson(options.key, fresh, options.ttlSeconds);
          } catch {
            // Keep serving the previous cache entry.
          }
        })(),
      );
    }
    return { value: cached, cacheHit: true };
  }
  const value = await options.load();
  await cachePutJson(options.key, value, options.ttlSeconds);
  return { value, cacheHit: false };
}

/** Read a stored backtest report from D1 (migration 0005_backtest). */
export async function readStoredBacktest(
  db: D1Like | undefined,
  modelId: string,
): Promise<Record<string, unknown> | null> {
  if (!db) return null;
  try {
    const row = await db
      .prepare(`SELECT report_json FROM backtest_reports WHERE model_id = ? LIMIT 1`)
      .bind(modelId)
      .first<{ report_json?: string }>();
    if (!row || typeof row.report_json !== "string") return null;
    const parsed = JSON.parse(row.report_json) as unknown;
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : null;
  } catch {
    return null;
  }
}
