import { afterEach, describe, expect, it, vi } from "vitest";

import {
  AGENT_TOOL_CACHE_VERSION,
  EVAL_CACHE_TTL_SECONDS,
  FORECAST_CACHE_TTL_SECONDS,
  cacheGetJson,
  cacheGetOrLoad,
  cachePutJson,
  evalCacheKey,
  forecastCacheKey,
  readStoredBacktest,
} from "../src/agent/toolCache";

class MemoryCache {
  store = new Map<string, Response>();

  async match(request: Request): Promise<Response | undefined> {
    const hit = this.store.get(request.url);
    return hit ? hit.clone() : undefined;
  }

  async put(request: Request, response: Response): Promise<void> {
    this.store.set(request.url, response.clone());
  }
}

describe("agent tool cache", () => {
  const memory = new MemoryCache();

  afterEach(() => {
    memory.store.clear();
    vi.unstubAllGlobals();
  });

  it("keys forecast and eval by series/model and horizon", () => {
    // v2 invalidates pre-#105 °C / unconverted-MAE payloads still in Cache API.
    expect(AGENT_TOOL_CACHE_VERSION).toBe(2);
    expect(forecastCacheKey("weather", 3)).toBe(
      `forecast:v${AGENT_TOOL_CACHE_VERSION}:weather:h3:p120`,
    );
    expect(evalCacheKey("weather", 1)).toBe(`eval:v${AGENT_TOOL_CACHE_VERSION}:weather:h1`);
    expect(forecastCacheKey("weather", 3)).toBe("forecast:v2:weather:h3:p120");
    expect(evalCacheKey("weather", 1)).toBe("eval:v2:weather:h1");
    expect(FORECAST_CACHE_TTL_SECONDS).toBeGreaterThanOrEqual(3_600);
    expect(EVAL_CACHE_TTL_SECONDS).toBeGreaterThanOrEqual(3_600);
  });

  it("serves cached forecast JSON without calling the loader again", async () => {
    vi.stubGlobal("caches", { default: memory });
    let loads = 0;
    const first = await cacheGetOrLoad({
      key: forecastCacheKey("weather", 3),
      ttlSeconds: 60,
      load: async () => {
        loads += 1;
        return { series: "weather", forecast: [{ value: 15.15 }] };
      },
    });
    expect(first.cacheHit).toBe(false);
    expect(loads).toBe(1);

    const second = await cacheGetOrLoad({
      key: forecastCacheKey("weather", 3),
      ttlSeconds: 60,
      load: async () => {
        loads += 1;
        return { series: "weather", forecast: [{ value: 99 }] };
      },
    });
    expect(second.cacheHit).toBe(true);
    expect(loads).toBe(1);
    expect((second.value as { forecast: Array<{ value: number }> }).forecast[0].value).toBe(15.15);
  });

  it("schedules waitUntil refresh on cache hit when requested", async () => {
    vi.stubGlobal("caches", { default: memory });
    await cachePutJson(forecastCacheKey("weather", 1), { series: "weather", v: 1 }, 60);
    const background: Promise<unknown>[] = [];
    let loads = 0;
    const hit = await cacheGetOrLoad({
      key: forecastCacheKey("weather", 1),
      ttlSeconds: 60,
      refreshOnHit: true,
      waitUntil: (promise) => {
        background.push(promise);
      },
      load: async () => {
        loads += 1;
        return { series: "weather", v: 2 };
      },
    });
    expect(hit.cacheHit).toBe(true);
    expect(hit.value).toEqual({ series: "weather", v: 1 });
    expect(background).toHaveLength(1);
    await background[0];
    expect(loads).toBe(1);
    expect(await cacheGetJson(forecastCacheKey("weather", 1))).toEqual({ series: "weather", v: 2 });
  });

  it("reads stored backtest rows from D1", async () => {
    const db = {
      prepare: () => ({
        bind: () => ({
          first: async () => ({
            report_json: JSON.stringify({ comparison: "2% better", metrics: { mae: 1.2 } }),
          }),
          all: async () => ({ results: [] }),
          run: async () => ({}),
        }),
        all: async () => ({ results: [] }),
        run: async () => ({}),
      }),
    };
    const report = await readStoredBacktest(db, "weather");
    expect(report?.comparison).toBe("2% better");
    expect(await readStoredBacktest(undefined, "weather")).toBeNull();
  });
});
