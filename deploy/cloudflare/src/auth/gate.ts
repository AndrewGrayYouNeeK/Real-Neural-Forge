/**
 * Front door for real-neural-forge-api:
 *  - nous.youneek.* and forge.youneek.* → 301 to henad.youneek.* (same path + query)
 *    for page and asset loads; API calls from installed apps are served in place.
 *    A signed-in browser visit gets a one-time hand-off instead (see handoff.ts),
 *    because the session cookie is host-only and would not follow the redirect.
 *  - everything except the public pages, docs, health and the auth routes
 *    needs a signed-in session.
 */

export const LEGACY_HOSTS: Record<string, string> = {
  "nous.youneek.dev": "henad.youneek.dev",
  "nous.youneek.xyz": "henad.youneek.xyz",
  "forge.youneek.dev": "henad.youneek.dev",
  "forge.youneek.xyz": "henad.youneek.xyz",
};

// Machine callers configured outside this repo keep working on the old host.
const NO_REDIRECT_PATHS = new Set(["/voice/stripe/webhook"]);

/** The henad.* host an old host moves to, or null when the host is not an old one. */
export function legacyTarget(url: URL): string | null {
  return LEGACY_HOSTS[url.hostname.toLowerCase()] ?? null;
}

export function movedUrl(url: URL, target: string): URL {
  const next = new URL(url.toString());
  next.hostname = target;
  next.protocol = "https:";
  next.port = "";
  return next;
}

/**
 * True for old-host requests that are served in place instead of redirected:
 * the Stripe webhook, internal bindings, and API calls from installed apps
 * (anything that is not a plain GET/HEAD, or that carries a Bearer token).
 * Death Terminal's `--nous` mode calls https://nous.youneek.dev with a Bearer
 * session and does not follow redirects; sessions are not tied to a host, so
 * those calls keep working on the old host.
 */
export function servedOnLegacyHost(url: URL, request?: Request): boolean {
  if (NO_REDIRECT_PATHS.has(url.pathname) || url.pathname.startsWith("/_bindings/")) return true;
  if (!request) return false;
  if (request.method !== "GET" && request.method !== "HEAD") return true;
  return /^Bearer\s+\S/i.test(request.headers.get("Authorization") || "");
}

export function legacyRedirect(request: Request): Response | null {
  const url = new URL(request.url);
  const target = legacyTarget(url);
  if (!target || servedOnLegacyHost(url, request)) return null;
  return new Response(null, {
    status: 301,
    // no-store: a browser that cached this redirect would skip the signed-in
    // hand-off on a later visit.
    headers: { Location: movedUrl(url, target).toString(), "Cache-Control": "no-store" },
  });
}

const PUBLIC_EXACT = new Set([
  "/",
  "/app",
  "/index.html",
  "/favicon.ico",
  "/manifest.webmanifest",
  "/robots.txt",
  "/health",
  "/docs",
  "/redoc",
  "/openapi.json",
  "/auth/signup",
  "/auth/login",
  "/auth/logout",
  "/auth/me",
  "/auth/handoff",
  "/voice/stripe/webhook",
]);

const PUBLIC_PREFIXES = ["/assets/", "/docs/", "/openapi/", "/youneek/"];

export function isPublicPath(method: string, pathname: string): boolean {
  if (PUBLIC_EXACT.has(pathname)) return true;
  if (PUBLIC_PREFIXES.some((prefix) => pathname.startsWith(prefix))) return true;
  // Shared forecast reports are read-only links people send each other.
  if (method === "GET" && /^\/agent\/reports\/[A-Za-z0-9_-]+$/.test(pathname)) return true;
  return false;
}

export function authRequired(): Response {
  return Response.json(
    { detail: "Sign in to continue.", auth_required: true },
    { status: 401, headers: { "Cache-Control": "no-store" } },
  );
}
