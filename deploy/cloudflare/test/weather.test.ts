import { describe, expect, it } from "vitest";

import {
  conditionFromCode,
  currentReplyText,
  formatTempF,
  weatherReplyText,
} from "../src/live/weather";
import { CHICAGO } from "../src/live/geo";

describe("weather replies", () => {
  it("describes current conditions without tomorrow language", () => {
    const text = currentReplyText(
      { ...CHICAGO, source: "question", name: "New York", region: "New York" },
      {
        tempF: 54.2,
        condition: "partly cloudy skies",
        humidity: 61,
        windMph: 8.4,
        asOf: "2026-11-13T15:00",
        timezone: "America/New_York",
      },
    );
    expect(text).toContain("Right now");
    expect(text).toContain("New York");
    expect(text).toContain("54°F");
    expect(text).toContain("partly cloudy skies");
    expect(text).toContain("61% humidity");
    expect(text).toContain("8 mph");
    expect(text.toLowerCase()).not.toContain("tomorrow");
    expect(text).toMatch(/current conditions/i);
  });

  it("describes high, low, rain, and conditions in Fahrenheit", () => {
    const text = weatherReplyText(
      { ...CHICAGO, source: "question", name: "Dallas", region: "Texas" },
      { kind: "tomorrow", dates: ["2026-11-11"], label: "tomorrow" },
      [
        {
          date: "2026-11-11",
          highF: 71.4,
          lowF: 52.6,
          rainChance: 20.2,
          condition: "partly cloudy skies",
        },
      ],
    );
    expect(text).toContain("Dallas");
    expect(text).toContain("71°F");
    expect(text).toContain("53°F");
    expect(text).toContain("20% chance of rain");
    expect(text).toContain("partly cloudy skies");
    expect(text).not.toMatch(/synthetic/i);
  });

  it("says when the fallback city was used", () => {
    const text = weatherReplyText(
      CHICAGO,
      { kind: "tomorrow", dates: ["2026-11-11"], label: "tomorrow" },
      [{ date: "2026-11-11", highF: 48, lowF: 33, rainChance: 10, condition: "overcast skies" }],
    );
    expect(text).toMatch(/Chicago as a default/);
  });

  it("maps WMO codes and formats temperatures", () => {
    expect(conditionFromCode(0)).toBe("clear skies");
    expect(conditionFromCode(95)).toBe("thunderstorms");
    expect(formatTempF(68.2)).toBe("68°F");
  });
});
