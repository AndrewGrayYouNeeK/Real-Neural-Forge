import { describe, expect, it } from "vitest";

import { conditionFromCode, formatTempF, weatherReplyText } from "../src/live/weather";
import { CHICAGO } from "../src/live/geo";

describe("weather replies", () => {
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
