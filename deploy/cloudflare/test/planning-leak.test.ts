import { describe, expect, it } from "vitest";
import { finalAnswerText, sanitizeAgentAnswer } from "../src/agent/protocol";
import { isPlanningSentence, stripLeadingPlanning } from "../src/live/ai";

describe("hidden planning never reaches the user", () => {
  it("drops the Louisville leak and keeps the answer", () => {
    const out = sanitizeAgentAnswer(
      "We need data, so call weather_history for Louisville. Louisville averaged a high of 61°F last week.",
    );
    expect(out).toBe("Louisville averaged a high of 61°F last week.");
    expect(out).not.toMatch(/weather_history|we need data/i);
  });

  it("returns no answer when the reply is only planning", () => {
    expect(sanitizeAgentAnswer("We need data, so call weather_history for Louisville.")).toBe("");
    expect(finalAnswerText({ response: "We need data, so call weather_history for Louisville." })).toBeNull();
  });

  it("strips several self-talk sentences in a row", () => {
    const out = stripLeadingPlanning(
      "The user asks about Louisville weather. Let me use the forecast tool. I'll answer briefly.\nTomorrow looks sunny, high 58°F.",
    );
    expect(out).toBe("Tomorrow looks sunny, high 58°F.");
  });

  it("leaves normal answers alone", () => {
    for (const answer of [
      "We need water to live, so drink regularly.",
      "I can help with that: the sky is blue because of Rayleigh scattering.",
      "Let me be clear: the Moon landing happened in 1969.",
      "The user interface is simple.",
      "Louisville is in Kentucky.",
    ]) {
      expect(sanitizeAgentAnswer(answer)).toBe(answer);
    }
    expect(isPlanningSentence("So we should call fetch_live_data for bitcoin.")).toBe(true);
  });
});
