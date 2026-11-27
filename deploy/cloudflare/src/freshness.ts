// Make browsers re-check the page and its files every visit.
//
// The container sends the site's HTML, CSS, JS and images with only an ETag and
// Last-Modified. With no Cache-Control, Safari and Chrome may reuse their saved
// copy for hours without asking, so a phone can keep showing an older Nous after a
// deploy. `no-cache` still lets them keep the file, but they must check first, and
// an unchanged file costs a tiny 304 reply.

const FRESH_TYPES = /^(text\/html|text\/css|(text|application)\/javascript|application\/manifest\+json|application\/json\+manifest|image\/|font\/)/i;

export function needsRevalidation(method: string, pathname: string, response: Response): boolean {
  if (method !== "GET" && method !== "HEAD") return false;
  if (pathname.startsWith("/agent/") || pathname.startsWith("/live/") || pathname.startsWith("/auth/")) return false;
  if (response.status !== 200 && response.status !== 304) return false;
  if (response.headers.get("Cache-Control")) return false;
  const type = response.headers.get("Content-Type") || "";
  return FRESH_TYPES.test(type) || /\.webmanifest$/.test(pathname);
}

export function withFreshness(request: Request, response: Response): Response {
  const { pathname } = new URL(request.url);
  if (!needsRevalidation(request.method, pathname, response)) return response;
  const headers = new Headers(response.headers);
  headers.set("Cache-Control", "no-cache");
  return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
}
