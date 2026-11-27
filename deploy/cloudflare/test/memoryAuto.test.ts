import { describe, expect, it } from "vitest";

import {
  MAX_MEMORY_PROMPT_CHARS,
  autoRemember,
  compactMemoryPrompt,
  extractMemoryFacts,
  listMemory,
  memoryEnabled,
  setMemoryEnabled,
} from "../src/agent/memory";
import { SqliteD1 } from "./helpers/sqliteD1";

describe("automatic memory", () => {
  it("picks up names, places, units, style and ongoing work", () => {
    expect(extractMemoryFacts("hey my name is andrew, I live in Atlanta and I'm working on a space game")).toEqual([
      { key: "name", value: "Andrew" },
      { key: "location", value: "Atlanta" },
      { key: "working_on", value: "a space game" },
    ]);
    expect(extractMemoryFacts("Please use metric. I prefer short answers")).toEqual([
      { key: "units", value: "metric" },
      { key: "answer_style", value: "short answers" },
    ]);
    expect(extractMemoryFacts("I'm learning how black holes work")).toEqual([{ key: "learning", value: "how black holes work" }]);
  });

  it("ignores things that are not facts about the user", () => {
    expect(extractMemoryFacts("call me later please")).toEqual([]);
    expect(extractMemoryFacts("what is the weather in Paris?")).toEqual([]);
    expect(extractMemoryFacts("is metric better than imperial")).toEqual([]);
  });

  it("keeps each account's memory separate and respects the off switch", async () => {
    const db = new SqliteD1();
    await autoRemember(db, "user:usr_a", "My name is Ana");
    await autoRemember(db, "user:usr_b", "My name is Bo");
    expect((await listMemory(db, "user:usr_a")).map((f) => f.value)).toEqual(["Ana"]);
    expect((await listMemory(db, "user:usr_b")).map((f) => f.value)).toEqual(["Bo"]);

    expect(await memoryEnabled(db, "user:usr_a")).toBe(true);
    await setMemoryEnabled(db, "user:usr_a", false);
    expect(await memoryEnabled(db, "user:usr_a")).toBe(false);
    expect(await memoryEnabled(db, "user:usr_b")).toBe(true);
    // Anonymous visitors (shared networks) never get memory.
    expect(await memoryEnabled(db, "ip:abc")).toBe(false);
  });

  it("keeps the chat context small", () => {
    const facts = Array.from({ length: 32 }, (_, i) => ({ key: `k${i}`, value: "x".repeat(200), updated_at: "" }));
    const block = compactMemoryPrompt(facts, ["Black holes", "Stocks"]);
    expect(block.length).toBeLessThanOrEqual(MAX_MEMORY_PROMPT_CHARS);
    expect(compactMemoryPrompt([], ["Moon phases"])).toContain("Recent chat topics: Moon phases");
  });
});
