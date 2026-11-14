import { describe, expect, it } from "vitest";

import {
  detectMarketKind,
  isWeatherAsk,
  resolveFetchLiveKind,
} from "../src/live/kind";

describe("live data kind routing", () => {
  it("detects crypto and stock before weather", () => {
    expect(detectMarketKind("what's bitcoin at right now")).toBe("crypto");
    expect(detectMarketKind("btc price")).toBe("crypto");
    expect(detectMarketKind("ethereum trading at")).toBe("crypto");
    expect(detectMarketKind("what's AAPL at right now")).toBe("stock");
    expect(detectMarketKind("$TSLA price")).toBe("stock");
    expect(detectMarketKind("what's the weather in New York right now")).toBeNull();
  });

  it("only treats weather/temperature/rain as weather asks", () => {
    expect(isWeatherAsk("what's the weather in New York right now")).toBe(true);
    expect(isWeatherAsk("temperature in Dallas")).toBe(true);
    expect(isWeatherAsk("will it rain tomorrow")).toBe(true);
    expect(isWeatherAsk("Forecast tomorrow's high and low in New York")).toBe(true);
    expect(isWeatherAsk("forecast the high for Chicago tomorrow")).toBe(true);
    expect(isWeatherAsk("what's bitcoin at right now")).toBe(false);
    expect(isWeatherAsk("AAPL price")).toBe(false);
  });

  it("overrides mistaken weather kind for market questions", () => {
    expect(
      resolveFetchLiveKind("weather", "Rightangle", "what's bitcoin at right now"),
    ).toBe("crypto");
    expect(resolveFetchLiveKind("weather", "New York", "weather in New York right now")).toBe(
      "weather",
    );
    expect(resolveFetchLiveKind("moon", "ignored", "bitcoin")).toBe("moon");
  });
});
