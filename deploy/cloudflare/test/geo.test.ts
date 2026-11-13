import { describe, expect, it } from "vitest";

import {
  CHICAGO,
  addIsoDays,
  extractPlaceName,
  parseDateWindow,
  visitorFromCf,
  ymdInZone,
} from "../src/live/geo";

describe("extractPlaceName", () => {
  it("reads a city after in/for/at", () => {
    expect(extractPlaceName("Weather in Dallas this weekend?")).toBe("Dallas");
    expect(extractPlaceName("What's the weather in New York tomorrow?")).toBe("New York");
    expect(extractPlaceName("forecast for Chicago, IL today")).toBe("Chicago, IL");
  });

  it("returns null when no place is given", () => {
    expect(extractPlaceName("What will the weather be tomorrow?")).toBeNull();
    expect(extractPlaceName("Is it going to rain?")).toBeNull();
  });

  it("never geocodes right/now/bitcoin leftovers", () => {
    expect(extractPlaceName("what's bitcoin at right now")).toBeNull();
    expect(extractPlaceName("bitcoin at right now")).toBeNull();
    expect(extractPlaceName("what's the price at right now")).toBeNull();
    expect(extractPlaceName("weather at right now")).toBeNull();
    expect(extractPlaceName("what's the weather in New York right now")).toBe("New York");
  });
});

describe("parseDateWindow", () => {
  const noonUtc = new Date("2026-11-10T18:00:00Z"); // 12:00 in Chicago on Tue Nov 10

  it("defaults to tomorrow", () => {
    const window = parseDateWindow("What will the weather be?", "America/Chicago", noonUtc);
    expect(window.kind).toBe("tomorrow");
    expect(window.dates).toEqual(["2026-11-11"]);
  });

  it("handles tomorrow and this weekend", () => {
    expect(parseDateWindow("tomorrow", "America/Chicago", noonUtc).dates).toEqual(["2026-11-11"]);
    const weekend = parseDateWindow("Weather in Dallas this weekend?", "America/Chicago", noonUtc);
    expect(weekend.kind).toBe("weekend");
    expect(weekend.dates).toEqual(["2026-11-14", "2026-11-15"]);
  });

  it("uses current conditions for now / right now / currently", () => {
    const rightNow = parseDateWindow(
      "what's the weather in New York right now",
      "America/New_York",
      noonUtc,
    );
    expect(rightNow.kind).toBe("now");
    expect(rightNow.label).toBe("right now");
    expect(rightNow.dates).toEqual(["2026-11-10"]);
    expect(parseDateWindow("currently in Dallas", "America/Chicago", noonUtc).kind).toBe("now");
    expect(parseDateWindow("weather now", "America/Chicago", noonUtc).kind).toBe("now");
    expect(
      parseDateWindow("tell me now what tomorrow looks like", "America/Chicago", noonUtc).kind,
    ).toBe("tomorrow");
  });

  it("keeps New York out of the place name when asking right now", () => {
    expect(extractPlaceName("what's the weather in New York right now")).toBe("New York");
  });
});

describe("visitorFromCf", () => {
  it("parses Cloudflare geo fields", () => {
    const visitor = visitorFromCf({
      city: "Dallas",
      latitude: "32.78",
      longitude: "-96.8",
      timezone: "America/Chicago",
      region: "Texas",
      country: "US",
    });
    expect(visitor.city).toBe("Dallas");
    expect(visitor.latitude).toBeCloseTo(32.78);
    expect(visitor.longitude).toBeCloseTo(-96.8);
  });

  it("keeps Chicago as the documented fallback coordinates", () => {
    expect(CHICAGO.timezone).toBe("America/Chicago");
    expect(addIsoDays(ymdInZone("UTC", new Date("2026-11-10T00:00:00Z")), 1)).toBe("2026-11-11");
  });
});
