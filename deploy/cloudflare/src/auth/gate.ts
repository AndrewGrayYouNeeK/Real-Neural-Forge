/**
 * Front door for real-neural-forge-api:
 *  - nous.youneek.* and forge.youneek.* → 301 to henad.youneek.* (same path + query).
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

/** True for old-host paths that are served in place instead of redirected. */
export function servedOnLegacyHost(url: URL): boolean {
  return NO_REDIRECT_PATHS.has(url.pathname) || url.pathname.startsWith("/_bindings/");
}

export function legacyRedirect(request: Request): Response | null {
  const url = new URL(request.url);
  const target = legacyTarget(url);
  if (!target || servedOnLegacyHost(url)) return null;
  // 301 for GET/HEAD; 308 keeps the method and body for API calls.
  const status = request.method === "GET" || request.method === "HEAD" ? 301 : 308;
  return new Response(null, {
    status,
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
