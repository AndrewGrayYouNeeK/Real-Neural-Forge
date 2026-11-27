import { afterEach, describe, expect, it } from "vitest";

import {
  KNOWLEDGE_LIMITS,
  clearKnowledgeMemoryCache,
  collectSources,
  currentEventsTitle,
  extractReadable,
  parseCurrentEvents,
  queryTerms,
  runReadPage,
  runWebSearch,
  runWikiSummary,
  safePublicUrl,
  takeLookup,
  wantsRecent,
  type KnowledgeContext,
} from "../src/agent/knowledge";
import { AGENT_TOOL_NAMES } from "../src/agent/protocol";
import { TOOL_DEFINITIONS } from "../src/agent/tools";

afterEach(() => clearKnowledgeMemoryCache());

const CE_WIKITEXT = [
  "{{Current events|year=2026|month=11|day=25}}",
  "'''Disasters and accidents'''",
  "*[[2026 Atlantic hurricane season]]",
  "**[[Hurricane Zelda]] makes landfall near [[Tampa, Florida]] as a Category 3 storm. [https://apnews.com/article/zelda (AP)] [https://www.reuters.com/zelda (Reuters)]",
  "'''Sports'''",
  "*The [[Detroit Lions]] beat the [[Chicago Bears]] 31–17 on Thanksgiving. [https://www.espn.com/nfl/lions (ESPN)]",
].join("\n");

function fakeFetch(routes: Array<[RegExp, unknown, number?]>): { fetcher: typeof fetch; calls: string[] } {
  const calls: string[] = [];
  const fetcher = (async (input: RequestInfo | URL) => {
    const url = String(input);
    calls.push(url);
    for (const [pattern, body, status] of routes) {
      if (pattern.test(url)) {
        const text = typeof body === "string" ? body : JSON.stringify(body);
        return new Response(text, {
          status: status ?? 200,
          headers: { "Content-Type": typeof body === "string" ? "text/html; charset=utf-8" : "application/json" },
        });
      }
    }
    return new Response("{}", { status: 404 });
  }) as typeof fetch;
  return { fetcher, calls };
}

function ctx(overrides: Partial<KnowledgeContext> = {}): KnowledgeContext {
  return { question: "", now: new Date("2026-11-26T20:00:00Z"), memoryScope: "ip:abc", ...overrides };
}

describe("knowledge tools are registered", () => {
  it("exposes web_search, wiki_summary and read_page to the model", () => {
    for (const name of ["web_search", "wiki_summary", "read_page"]) {
      expect(AGENT_TOOL_NAMES).toContain(name);
      expect(TOOL_DEFINITIONS.some((tool) => tool.name === name)).toBe(true);
    }
  });
});

describe("recency + query helpers", () => {
  it("detects current-events asks", () => {
    const now = new Date("2026-11-26T00:00:00Z");
    expect(wantsRecent("what's the latest news on the hurricane", now)).toBe(true);
    expect(wantsRecent("who won the Lions game yesterday", now)).toBe(true);
    expect(wantsRecent("what happened in 2026 with SpaceX", now)).toBe(true);
    expect(wantsRecent("why is the sky blue", now)).toBe(false);
    expect(wantsRecent("who was Ada Lovelace", now)).toBe(false);
  });

  it("drops filler words from queries", () => {
    expect(queryTerms("What is the latest on Hurricane Zelda?")).toEqual(["hurricane", "zelda"]);
  });

  it("builds Current events page titles", () => {
    expect(currentEventsTitle(new Date("2026-11-05T12:00:00Z"))).toBe("Portal:Current events/2026 November 5");
  });
});

describe("Current events parsing", () => {
  it("turns wikitext into news items with outlet links", () => {
    const items = parseCurrentEvents(CE_WIKITEXT, "2026-11-25");
    expect(items).toHaveLength(2);
    expect(items[0].text).toContain("Hurricane Zelda makes landfall near Tampa, Florida");
    expect(items[0].text).toContain("2026 Atlantic hurricane season");
    expect(items[0].links[0]).toEqual({ label: "AP", url: "https://apnews.com/article/zelda" });
    expect(items[1].links[0].url).toBe("https://www.espn.com/nfl/lions");
  });
});

describe("web_search", () => {
  it("combines recent news and Wikipedia results with sources", async () => {
    const { fetcher } = fakeFetch([
      [/action=parse.*2026%20November%2025/, { parse: { wikitext: CE_WIKITEXT } }],
      [/action=parse/, { parse: { wikitext: "" } }],
      [
        /list=search/,
        { query: { search: [{ title: "Hurricane Zelda", snippet: "<span>Hurricane Zelda</span> was a 2026 storm" }] } },
      ],
    ]);
    const data = await runWebSearch(ctx({ fetcher, question: "latest on hurricane zelda" }), { query: "hurricane zelda" });
    expect(data.ok).toBe(true);
    expect(data.recent).toBe(true);
    const results = data.results as Array<{ url: string; site: string; snippet: string }>;
    expect(results[0].url).toBe("https://apnews.com/article/zelda");
    expect(results.some((item) => item.url === "https://en.wikipedia.org/wiki/Hurricane_Zelda")).toBe(true);
    expect(results.find((item) => item.site === "Wikipedia")?.snippet).toBe("Hurricane Zelda was a 2026 storm");
    expect(collectSources([{ ok: true, name: "web_search", data }]).length).toBe(results.length);
  });

  it("reads the top news story so answers have real details", async () => {
    const article = `<html><head><title>Zelda hits Tampa</title></head><body><article><p>${"Hurricane Zelda came ashore near Tampa with 120 mph winds, officials said on Wednesday. ".repeat(6)}</p></article></body></html>`;
    const { fetcher } = fakeFetch([
      [/apnews\.com/, article],
      [/action=parse.*2026%20November%2025/, { parse: { wikitext: CE_WIKITEXT } }],
      [/action=parse/, { parse: { wikitext: "" } }],
      [/list=search/, { query: { search: [] } }],
    ]);
    const data = await runWebSearch(ctx({ fetcher }), { query: "hurricane zelda", recent: true });
    const story = data.top_story as { url: string; text: string };
    expect(story.url).toBe("https://apnews.com/article/zelda");
    expect(story.text).toContain("120 mph winds");
  });

  it("caches repeat searches", async () => {
    const { fetcher, calls } = fakeFetch([[/list=search/, { query: { search: [{ title: "Ada Lovelace", snippet: "math" }] } }]]);
    await runWebSearch(ctx({ fetcher }), { query: "Ada Lovelace" });
    const before = calls.length;
    const again = await runWebSearch(ctx({ fetcher }), { query: "Ada Lovelace" });
    expect(calls.length).toBe(before);
    expect(again.cached).toBe(true);
  });

  it("stops after the daily per-visitor allowance", async () => {
    const { fetcher } = fakeFetch([[/list=search/, { query: { search: [] } }]]);
    for (let i = 0; i < KNOWLEDGE_LIMITS.visitor; i++) {
      expect(await takeLookup(ctx({ fetcher }))).toBe(true);
    }
    const data = await runWebSearch(ctx({ fetcher }), { query: "anything" });
    expect(data.limited).toBe(true);
    // Another visitor still has their own allowance.
    expect(await takeLookup(ctx({ memoryScope: "ip:other" }))).toBe(true);
  });
});

describe("wiki_summary", () => {
  it("returns the REST summary with its URL", async () => {
    const { fetcher } = fakeFetch([
      [
        /rest_v1\/page\/summary\/Black_hole/,
        {
          type: "standard",
          title: "Black hole",
          description: "Region of spacetime",
          extract: "A black hole is a region of spacetime where gravity is so strong that nothing can escape.",
          content_urls: { desktop: { page: "https://en.wikipedia.org/wiki/Black_hole" } },
        },
      ],
    ]);
    const data = await runWikiSummary(ctx({ fetcher }), { topic: "Black hole" });
    expect(data.ok).toBe(true);
    expect(data.url).toBe("https://en.wikipedia.org/wiki/Black_hole");
    expect(String(data.extract)).toContain("nothing can escape");
  });
});

describe("read_page", () => {
  it("refuses private, local and non-http URLs", () => {
    for (const bad of [
      "http://localhost/admin",
      "http://127.0.0.1/",
      "http://169.254.169.254/latest/meta-data",
      "file:///etc/passwd",
      "https://nous.youneek.dev/agent/chat",
      "http://[::1]/",
      "https://user:pw@example.org/",
      "https://intranet:8443/",
    ]) {
      expect(safePublicUrl(bad)).toBeNull();
    }
    expect(safePublicUrl("https://www.nasa.gov/news")?.hostname).toBe("www.nasa.gov");
  });

  it("extracts readable text and drops scripts and nav", () => {
    const html = `<html><head><title>Moon facts | NASA</title><script>alert(1)</script></head>
      <body><nav><a href="/">Home</a> Subscribe to our newsletter</nav>
      <article><h1>The Moon</h1><p>The Moon is Earth's only natural satellite and it is about 384,400 km away from us on average.</p>
      <p>It formed about 4.5 billion years ago, most likely from debris after a Mars-sized body hit the early Earth.</p></article>
      <footer>All rights reserved</footer></body></html>`;
    const page = extractReadable(html);
    expect(page.title).toBe("Moon facts | NASA");
    expect(page.text).toContain("Earth's only natural satellite");
    expect(page.text).not.toContain("alert");
    expect(page.text).not.toContain("newsletter");
  });

  it("reads a public page through the fetcher", async () => {
    const html = `<html><head><title>Story</title></head><body><main><p>${"Real readable sentence about the topic at hand. ".repeat(10)}</p></main></body></html>`;
    const { fetcher } = fakeFetch([[/example-news\.org/, html]]);
    const data = await runReadPage(ctx({ fetcher }), { url: "https://example-news.org/story" });
    expect(data.ok).toBe(true);
    expect(String(data.text)).toContain("Real readable sentence");
    expect(data._sources).toEqual([{ title: "Story", url: "https://example-news.org/story" }]);
  });
});
