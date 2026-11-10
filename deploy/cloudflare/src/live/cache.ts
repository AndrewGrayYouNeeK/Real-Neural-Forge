/** Short-lived cache for outbound live-data fetches. */

export const WEATHER_TTL_SECONDS = 30 * 60;
export const PRICE_TTL_SECONDS = 60;
export const LOTTERY_TTL_SECONDS = 6 * 60 * 60;
export const GEOCODE_TTL_SECONDS = 24 * 60 * 60;

const CACHE_ORIGIN = "https://live-cache.forge.invalid/";

export const FETCH_UA = "NeuralForge/1.0 (https://forge.youneek.dev)";

export async function cachedJson<T>(
  key: string,
  ttlSeconds: number,
  loader: () => Promise<T>,
): Promise<T> {
  const cache = globalThis.caches?.default;
  const request = new Request(`${CACHE_ORIGIN}${encodeURIComponent(key)}`);
  if (cache) {
    const hit = await cache.match(request);
    if (hit) {
      return (await hit.json()) as T;
    }
  }
  const data = await loader();
  if (cache) {
    const response = new Response(JSON.stringify(data), {
      headers: {
        "Content-Type": "application/json",
        "Cache-Control": `public, max-age=${ttlSeconds}`,
      },
    });
    try {
      await cache.put(request, response);
    } catch {
      // Cache writes are best-effort; a miss only means an extra fetch.
    }
  }
  return data;
}

export async function fetchText(
  url: string,
  init: RequestInit = {},
): Promise<{ ok: boolean; status: number; text: string }> {
  const headers = new Headers(init.headers);
  if (!headers.has("User-Agent")) headers.set("User-Agent", FETCH_UA);
  const response = await fetch(url, { ...init, headers });
  const text = await response.text();
  return { ok: response.ok, status: response.status, text };
}

export async function fetchJson<T>(url: string, init: RequestInit = {}): Promise<T> {
  const result = await fetchText(url, init);
  if (!result.ok) {
    throw new Error(`Request failed (${result.status})`);
  }
  return JSON.parse(result.text) as T;
}
