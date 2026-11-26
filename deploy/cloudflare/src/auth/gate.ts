/**
 * Front door for real-neural-forge-api:
 *  - forge.youneek.* → 301 to nous.youneek.* (same path + query)
 *  - everything except the public pages, docs, health and the auth routes
 *    needs a signed-in session.
 */

export const LEGACY_HOSTS: Record<string, string> = {
  "forge.youneek.dev": "nous.youneek.dev",
  "forge.youneek.xyz": "nous.youneek.xyz",
};

// Machine callers configured outside this repo keep working on the old host.
const NO_REDIRECT_PATHS = new Set(["/voice/stripe/webhook"]);

export function legacyRedirect(request: Request): Response | null {
  const url = new URL(request.url);
  const target = LEGACY_HOSTS[url.hostname.toLowerCase()];
  if (!target) return null;
  if (NO_REDIRECT_PATHS.has(url.pathname) || url.pathname.startsWith("/_bindings/")) return null;
  const next = new URL(url.toString());
  next.hostname = target;
  next.protocol = "https:";
  next.port = "";
  return new Response(null, {
    status: 301,
    headers: { Location: next.toString(), "Cache-Control": "public, max-age=3600" },
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
