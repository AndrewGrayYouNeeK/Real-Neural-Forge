import { describe, expect, it } from "vitest";
import { needsRevalidation, withFreshness } from "../src/freshness";

function page(type: string, headers: Record<string, string> = {}, status = 200): Response {
  return new Response("x", { status, headers: { "Content-Type": type, ...headers } });
}

describe("freshness", () => {
  it("asks browsers to re-check html, css, js and images", () => {
    for (const type of ["text/html; charset=utf-8", "text/css", "text/javascript", "application/javascript", "image/webp", "image/png"]) {
      expect(needsRevalidation("GET", "/assets/x", page(type))).toBe(true);
    }
    expect(needsRevalidation("GET", "/manifest.webmanifest", page("application/octet-stream"))).toBe(true);
  });

  it("leaves API replies, errors, writes and explicit cache rules alone", () => {
    expect(needsRevalidation("GET", "/agent/chats", page("application/json"))).toBe(false);
    expect(needsRevalidation("GET", "/health", page("application/json"))).toBe(false);
    expect(needsRevalidation("POST", "/", page("text/html"))).toBe(false);
    expect(needsRevalidation("GET", "/", page("text/html", {}, 500))).toBe(false);
    expect(needsRevalidation("GET", "/assets/pyworker.js", page("text/javascript", { "Cache-Control": "public, max-age=3600" }))).toBe(false);
    expect(needsRevalidation("GET", "/agent/images/1", page("image/png"))).toBe(false);
  });

  it("adds no-cache and keeps the rest of the reply", async () => {
    const out = withFreshness(new Request("https://nous.youneek.dev/"), page("text/html", { ETag: '"abc"' }));
    expect(out.headers.get("Cache-Control")).toBe("no-cache");
    expect(out.headers.get("ETag")).toBe('"abc"');
    expect(await out.text()).toBe("x");
  });
});
