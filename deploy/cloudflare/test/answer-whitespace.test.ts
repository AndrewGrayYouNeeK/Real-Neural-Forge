import { describe, expect, it } from "vitest";
import { sanitizeAgentAnswer, tidyAnswerWhitespace } from "../src/agent/protocol";

describe("answer whitespace", () => {
  it("keeps paragraphs, lists, tables and code blocks", () => {
    const answer = [
      "Short answer: yes.",
      "",
      "Here's why:",
      "- First point",
      "  - Nested point",
      "1. Step one",
      "",
      "| a | b |",
      "|---|---|",
      "| 1 | 2 |",
      "",
      "```python",
      "def f(x):",
      "    if x:",
      "        return  x",
      "```",
    ].join("\n");
    expect(sanitizeAgentAnswer(answer)).toBe(answer);
  });

  it("normalizes CRLF, trailing spaces, extra blank lines and space runs", () => {
    expect(sanitizeAgentAnswer("One  two   three  \r\n\r\n\r\n\r\nFour")).toBe("One two three\n\nFour");
  });

  it("strips control and invisible characters but keeps emoji joiners", () => {
    const dirty = "Hel\u0000lo\u200B wor\u202Eld\uFEFF\u0007\u00AD!\n\u2066Next\u2069 line 👨\u200D👩\u200D👧";
    expect(tidyAnswerWhitespace(dirty)).toBe("Hello world!\nNext line 👨\u200D👩\u200D👧");
  });

  it("still strips leaked planning and tool JSON", () => {
    const out = sanitizeAgentAnswer(
      'We need data, so call weather_history for Louisville.\n{"tool":"weather_history","args":{}}\nIt was 61°F.\n\nDry all week.',
    );
    expect(out).toBe("It was 61°F.\n\nDry all week.");
  });
});
