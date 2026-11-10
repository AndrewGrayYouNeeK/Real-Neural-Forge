import { describe, expect, it } from "vitest";

import {
  frequencies,
  lotteryReplyText,
  parseMegaMillionsCsv,
  parsePowerballCsv,
  wantsFrequencies,
} from "../src/live/lottery";

const POWERBALL = `draw_date,winning_numbers,multiplier
2026-11-08T00:00:00.000,04 18 25 40 48 12,3
2026-11-05T00:00:00.000,01 09 12 33 55 12,2
`;

const MEGA = `draw_date,winning_numbers,mega_ball,multiplier
2026-11-07T00:00:00.000,05 11 22 30 44,9,4
2026-11-04T00:00:00.000,05 16 18 41 52,3,3
`;

describe("lottery CSV and frequencies", () => {
  it("parses NY Powerball and Mega Millions CSV rows", () => {
    const powerball = parsePowerballCsv(POWERBALL);
    const mega = parseMegaMillionsCsv(MEGA);
    expect(powerball[0]).toMatchObject({ date: "2026-11-08", bonus: 12, bonusName: "Powerball" });
    expect(powerball[0].whites).toEqual([4, 18, 25, 40, 48]);
    expect(mega[0]).toMatchObject({ date: "2026-11-07", bonus: 9, bonusName: "Mega Ball" });
    expect(mega[0].whites).toEqual([5, 11, 22, 30, 44]);
  });

  it("reports the latest draws and keeps the random-draw caveat", () => {
    const text = lotteryReplyText(parsePowerballCsv(POWERBALL), parseMegaMillionsCsv(MEGA), false);
    expect(text).toContain("Powerball");
    expect(text).toContain("2026-11-08");
    expect(text).toContain("12");
    expect(text).toContain("Mega Millions");
    expect(text).toMatch(/random/i);
    expect(text).not.toMatch(/synthetic/i);
  });

  it("adds most- and least-frequent numbers when asked", () => {
    expect(wantsFrequencies("What are the most frequent Powerball numbers?")).toBe(true);
    const whites = frequencies([1, 1, 2, 3, 3, 3]);
    expect(whites.most[0]).toEqual({ value: 3, count: 3 });
    expect(whites.least[0]).toEqual({ value: 2, count: 1 });
    const text = lotteryReplyText(parsePowerballCsv(POWERBALL), parseMegaMillionsCsv(MEGA), true);
    expect(text).toMatch(/frequent/i);
  });
});
