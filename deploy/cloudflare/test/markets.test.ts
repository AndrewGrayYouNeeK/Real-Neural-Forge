import { describe, expect, it } from "vitest";

import { parseCrypto, parseStooqCsv, parseTicker, quoteReplyText } from "../src/live/markets";

describe("quote parsing", () => {
  it("reads tickers and coin names from questions", () => {
    expect(parseTicker("How is AAPL doing?")).toBe("AAPL");
    expect(parseTicker("How are the markets doing?")).toBe("SPY");
    expect(parseTicker("What is Apple stock at?")).toBe("AAPL");
    expect(parseCrypto("What's Bitcoin at?")?.symbol).toBe("BTC");
    expect(parseCrypto("How is ETH doing?")?.id).toBe("ethereum");
  });

  it("parses a Stooq CSV row", () => {
    const csv = "Symbol,Date,Time,Open,High,Low,Close,Volume\nAAPL.US,2026-11-09,22:00:00,1,2,3,268.5,100\n";
    const quote = parseStooqCsv(csv);
    expect(quote?.symbol).toBe("AAPL");
    expect(quote?.price).toBe(268.5);
    expect(quote?.asOf).toBe("2026-11-09");
  });

  it("phrases a live quote with a not-advice caveat", () => {
    const text = quoteReplyText({
      kind: "crypto",
      symbol: "BTC",
      name: "Bitcoin",
      price: 104321.4,
      changePct: -1.25,
      asOf: null,
      currency: "USD",
    });
    expect(text).toContain("Bitcoin");
    expect(text).toContain("$104,321.40");
    expect(text).toMatch(/not financial advice/i);
  });
});
