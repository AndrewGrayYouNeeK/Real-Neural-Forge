import { LOTTERY_TTL_SECONDS, cachedJson, fetchText } from "./cache";

export type LotteryDraw = {
  game: "Powerball" | "Mega Millions";
  date: string;
  whites: number[];
  bonus: number;
  bonusName: string;
};

const POWERBALL_CSV = "https://data.ny.gov/resource/d6yy-54nr.csv?$order=draw_date DESC&$limit=500";
const MEGA_CSV = "https://data.ny.gov/resource/5xaw-6ayf.csv?$order=draw_date DESC&$limit=500";

export function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = "";
  let inQuotes = false;
  const source = text.replace(/^\uFEFF/, "");
  for (let index = 0; index < source.length; index += 1) {
    const char = source[index];
    if (inQuotes) {
      if (char === '"') {
        if (source[index + 1] === '"') {
          cell += '"';
          index += 1;
        } else {
          inQuotes = false;
        }
      } else {
        cell += char;
      }
      continue;
    }
    if (char === '"') {
      inQuotes = true;
      continue;
    }
    if (char === ",") {
      row.push(cell);
      cell = "";
      continue;
    }
    if (char === "\n") {
      row.push(cell);
      if (row.some((value) => value.trim())) rows.push(row.map((value) => value.trim()));
      row = [];
      cell = "";
      continue;
    }
    if (char !== "\r") cell += char;
  }
  if (cell || row.length) {
    row.push(cell);
    if (row.some((value) => value.trim())) rows.push(row.map((value) => value.trim()));
  }
  return rows;
}

function columnIndex(header: string[], aliases: string[]): number {
  const normalized = header.map((name) => name.toLowerCase().replace(/[^a-z0-9]+/g, "_"));
  for (const alias of aliases) {
    const index = normalized.indexOf(alias);
    if (index >= 0) return index;
  }
  return -1;
}

function parseWhites(value: string, expect: number): number[] {
  const numbers = value
    .split(/[^\d]+/)
    .map((part) => Number(part))
    .filter((number) => Number.isFinite(number) && number > 0);
  return numbers.slice(0, expect);
}

function shortDate(value: string): string {
  const match = value.match(/(\d{4}-\d{2}-\d{2})/);
  return match ? match[1] : value.slice(0, 10);
}

export function parsePowerballCsv(text: string): LotteryDraw[] {
  const rows = parseCsv(text);
  if (rows.length < 2) return [];
  const header = rows[0];
  const dateIndex = columnIndex(header, ["draw_date", "drawdate", "date"]);
  const numbersIndex = columnIndex(header, ["winning_numbers", "winningnumbers", "numbers"]);
  if (dateIndex < 0 || numbersIndex < 0) return [];
  const draws: LotteryDraw[] = [];
  for (const row of rows.slice(1)) {
    const numbers = parseWhites(row[numbersIndex] || "", 6);
    if (numbers.length < 6) continue;
    draws.push({
      game: "Powerball",
      date: shortDate(row[dateIndex] || ""),
      whites: numbers.slice(0, 5).sort((a, b) => a - b),
      bonus: numbers[5],
      bonusName: "Powerball",
    });
  }
  return draws;
}

export function parseMegaMillionsCsv(text: string): LotteryDraw[] {
  const rows = parseCsv(text);
  if (rows.length < 2) return [];
  const header = rows[0];
  const dateIndex = columnIndex(header, ["draw_date", "drawdate", "date"]);
  const numbersIndex = columnIndex(header, ["winning_numbers", "winningnumbers", "numbers"]);
  const megaIndex = columnIndex(header, ["mega_ball", "megaball", "mega"]);
  if (dateIndex < 0 || numbersIndex < 0) return [];
  const draws: LotteryDraw[] = [];
  for (const row of rows.slice(1)) {
    const whites = parseWhites(row[numbersIndex] || "", 5);
    const bonusParts = megaIndex >= 0 ? parseWhites(row[megaIndex] || "", 1) : parseWhites(row[numbersIndex] || "", 6).slice(5);
    if (whites.length < 5 || !bonusParts.length) continue;
    draws.push({
      game: "Mega Millions",
      date: shortDate(row[dateIndex] || ""),
      whites: whites.slice(0, 5).sort((a, b) => a - b),
      bonus: bonusParts[0],
      bonusName: "Mega Ball",
    });
  }
  return draws;
}

export type Frequency = { value: number; count: number };

export function frequencies(values: number[]): { most: Frequency[]; least: Frequency[] } {
  const counts = new Map<number, number>();
  for (const value of values) counts.set(value, (counts.get(value) || 0) + 1);
  const ranked = [...counts.entries()]
    .map(([value, count]) => ({ value, count }))
    .sort((a, b) => b.count - a.count || a.value - b.value);
  if (!ranked.length) return { most: [], least: [] };
  const topCount = ranked[0].count;
  const lowCount = ranked[ranked.length - 1].count;
  return {
    most: ranked.filter((row) => row.count === topCount),
    least: ranked.filter((row) => row.count === lowCount),
  };
}

export function wantsFrequencies(question: string): boolean {
  return /\b(most|least|frequent|common|hottest|coldest|hot numbers|cold numbers|how often)\b/i.test(
    question,
  );
}

function listNumbers(values: number[]): string {
  return values.map(String).join(", ");
}

function describeFrequencies(label: string, rows: Frequency[]): string {
  if (!rows.length) return "";
  const numbers = listNumbers(rows.map((row) => row.value));
  const count = rows[0].count;
  if (rows.length === 1) return `${label} ${numbers} (${count} times)`;
  return `${label} ${numbers} (each ${count} times)`;
}

function latestLine(draw: LotteryDraw | undefined): string {
  if (!draw) return "";
  return `The latest ${draw.game} draw on ${draw.date} was ${listNumbers(draw.whites)}, and the ${draw.bonusName} was ${draw.bonus}.`;
}

export function lotteryReplyText(
  powerball: LotteryDraw[],
  mega: LotteryDraw[],
  includeFrequencies: boolean,
): string {
  const lines = [latestLine(powerball[0]), latestLine(mega[0])].filter(Boolean);
  if (!lines.length) {
    return "I could not load the latest Powerball or Mega Millions draws right now. Draws are random either way — no model can pick the next winning numbers.";
  }
  if (includeFrequencies) {
    const sample = [...powerball, ...mega];
    const whites = frequencies(sample.flatMap((draw) => draw.whites));
    const powerballBonus = frequencies(powerball.map((draw) => draw.bonus));
    const megaBonus = frequencies(mega.map((draw) => draw.bonus));
    const bits = [
      describeFrequencies("most frequent white balls", whites.most.slice(0, 5)),
      describeFrequencies("least frequent white balls", whites.least.slice(0, 5)),
      describeFrequencies("most frequent Powerball", powerballBonus.most.slice(0, 3)),
      describeFrequencies("most frequent Mega Ball", megaBonus.most.slice(0, 3)),
    ].filter(Boolean);
    if (bits.length) {
      lines.push(`Across the last ${sample.length} published draws: ${bits.join("; ")}.`);
    }
  }
  lines.push("Lottery draws are random, so no model can tell you the next winning numbers. This is just a look at published results from New York's open data.");
  return lines.join(" ");
}

async function loadCsv(url: string, key: string): Promise<string> {
  const result = await cachedJson(`lottery:${key}`, LOTTERY_TTL_SECONDS, async () => {
    const response = await fetchText(url, {
      headers: { Accept: "text/csv,text/plain" },
    });
    if (!response.ok) throw new Error(`lottery fetch ${response.status}`);
    return { text: response.text };
  });
  return result.text;
}

export async function answerLottery(question: string): Promise<{ text: string }> {
  try {
    const [powerballText, megaText] = await Promise.all([
      loadCsv(POWERBALL_CSV, "powerball"),
      loadCsv(MEGA_CSV, "mega"),
    ]);
    const powerball = parsePowerballCsv(powerballText);
    const mega = parseMegaMillionsCsv(megaText);
    return { text: lotteryReplyText(powerball, mega, wantsFrequencies(question)) };
  } catch {
    return {
      text: "I could not load the latest lottery draws right now. Draws are random either way — no model can pick the next winning numbers.",
    };
  }
}
