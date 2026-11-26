/**
 * Multi-step planning for the agent loop.
 *
 * The model plans silently (hidden reasoning), but the loop also keeps a
 * deterministic checklist of what the user asked for so it can tell when a
 * reply stopped early ("I'll chart that next…") and re-loop instead of
 * returning half the job. Charts and CSV files are built here from tool data,
 * so the model never has to draw or format them itself.
 */

import type { ParsedToolCall } from "./protocol";
import type { ToolResult } from "./tools";

export type Deliverable =
  | "history"
  | "forecast"
  | "simulate"
  | "compare"
  | "backtest"
  | "train"
  | "chart"
  | "csv"
  | "explain";

export type RequestPlan = {
  /** Human-readable clauses, in the order the user asked. */
  steps: string[];
  deliverables: Deliverable[];
  multiStep: boolean;
  /** The ask names a scenario (cold snap, heat wave, crash…) that must actually be simulated. */
  namedScenario?: boolean;
};

const DETECTORS: Array<[Deliverable, RegExp]> = [
  [
    "history",
    /\b(?:last|past|previous)\s+(?:\d{1,3}\s+|few\s+|couple\s+(?:of\s+)?)?(?:days?|weeks?|months?)\b|\bhistor(?:y|ical)\b|\bso far this (?:week|month)\b/i,
  ],
  ["forecast", /\b(?:forecast|predict(?:ion)?|projection|project|next\s+\d{1,2}\s+(?:days?|steps?|values?|weeks?))\b/i],
  ["simulate", /\b(?:simulat\w*|what[\s-]if|scenarios?|cold\s*snap|heat\s*wave|shock|stress[\s-]test)\b/i],
  ["compare", /\b(?:compar\w*|vs\.?|versus|against (?:the )?(?:normal|baseline))\b/i],
  ["backtest", /\b(?:back-?test\w*|how accurate|accuracy|evaluate|evaluation|mae|rmse)\b/i],
  ["train", /\b(?:train\w*|retrain\w*|fit (?:a |the )?model)\b/i],
  ["chart", /\b(?:chart|graph|plot|visuali[sz]e|draw)\b/i],
  ["csv", /\b(?:csv|spreadsheet|excel|export|download(?:able)?\s+(?:the\s+)?(?:data|file|numbers))\b/i],
  ["explain", /\b(?:explain\w*|why|walk me through|break (?:it|that) down|what does (?:it|that) mean|tell me what it means)\b/i],
];

const STEP_SPLIT = /\s*(?:,\s*(?:and\s+)?(?:then\s+)?|;\s*|\band then\b|\bthen\b|\bafter that\b|\bplus\b|\balso\b)\s*/i;

/** Build a deterministic checklist from the user's request. */
export function planRequest(question: string): RequestPlan {
  const text = question.replace(/\s+/g, " ").trim();
  const deliverables: Deliverable[] = [];
  for (const [name, pattern] of DETECTORS) {
    if (pattern.test(text)) deliverables.push(name);
  }
  // A pasted number list is not a step list; drop it before splitting.
  const withoutNumbers = text.replace(/(?:-?\d+(?:\.\d+)?\s*,\s*){3,}-?\d+(?:\.\d+)?/g, "<series>");
  const steps = withoutNumbers
    .split(STEP_SPLIT)
    .map((part) => part.trim().replace(/^and\s+/i, ""))
    .filter((part) => part.length >= 3 && /[a-z]/i.test(part))
    .slice(0, 8);
  const multiStep = deliverables.length >= 2 || steps.length >= 2;
  const namedScenario = deliverables.includes("simulate") && scenarioFrom(text) != null;
  return { steps: steps.length ? steps : [text], deliverables, multiStep, namedScenario };
}

function okResults(results: ToolResult[], ...names: string[]): ToolResult[] {
  return results.filter((item) => item.ok && names.includes(item.name) && dataOk(item.data));
}

function dataOk(data: unknown): boolean {
  // Some tools resolve with { ok: false, text } for soft failures.
  if (data && typeof data === "object" && (data as { ok?: unknown }).ok === false) return false;
  return true;
}

function record(data: unknown): Record<string, unknown> {
  return data && typeof data === "object" && !Array.isArray(data) ? (data as Record<string, unknown>) : {};
}

function simulateHasScenarios(result: ToolResult): boolean {
  const scenarios = record(result.data).scenarios;
  return Array.isArray(scenarios) && scenarios.length > 0;
}

function simulateHasBacktest(result: ToolResult): boolean {
  const backtest = record(record(result.data).backtest);
  // "Series too short for a holdout backtest" counts as attempted: the model
  // sees that note and reports it instead of re-running the same fit.
  return Number(backtest.holdout) > 0 || typeof backtest.text === "string";
}

/** True when one requested deliverable is covered by this turn's tool results. */
export function deliverableDone(
  deliverable: Deliverable,
  results: ToolResult[],
  answer: string | null = null,
  wanted: Deliverable[] = [],
  namedScenario = false,
): boolean {
  switch (deliverable) {
    case "history":
      return okResults(results, "weather_history").length > 0 || okResults(results, "list_user_series").length > 0;
    case "forecast":
      return okResults(results, "forecast", "simulate_world", "what_if", "explain_prediction", "compare_horizons").length > 0;
    case "simulate":
      if (namedScenario) return okResults(results, "simulate_world").some(simulateHasScenarios);
      return (
        okResults(results, "simulate_world").length > 0 ||
        okResults(results, "what_if").some((item) => typeof record(item.data).summary === "string")
      );
    case "compare": {
      if (okResults(results, "compare_horizons").length) return true;
      if (okResults(results, "simulate_world").some(simulateHasScenarios)) return true;
      // "Simulate a cold snap": a generic one-step what_if is not that scenario.
      if (namedScenario) return false;
      // what_if only counts when it really produced a shocked-vs-baseline summary.
      if (okResults(results, "what_if").some((item) => typeof record(item.data).summary === "string")) return true;
      // Scenario asks ("simulate a cold snap and compare") need an actual scenario run.
      if (wanted.includes("simulate")) return false;
      // "compare X and Y" style asks: two successful lookups is a comparison.
      return results.filter((item) => item.ok).length >= 2;
    }
    case "backtest":
      if (okResults(results, "get_evaluation_metrics", "compare_horizons").length) return true;
      if (okResults(results, "simulate_world").some(simulateHasBacktest)) return true;
      return okResults(results, "forecast").some((item) => record(item.data).recent_accuracy != null);
    case "train":
      return (
        okResults(results, "retrain_user_series").length > 0 ||
        okResults(results, "simulate_world").some((item) => record(item.data).series === "custom")
      );
    case "chart":
      return buildChart(results) != null;
    case "csv":
      return buildChart(results) != null;
    case "explain":
      return Boolean(answer && answer.replace(/\s+/g, " ").trim().length >= 60);
    default:
      return true;
  }
}

/** Requested deliverables still missing (explain is judged against the answer). */
export function missingDeliverables(
  plan: RequestPlan,
  results: ToolResult[],
  answer: string | null = null,
): Deliverable[] {
  return plan.deliverables.filter(
    (item) => !deliverableDone(item, results, answer, plan.deliverables, Boolean(plan.namedScenario)),
  );
}

const LABELS: Record<Deliverable, string> = {
  history: "fetch the past data (weather_history for past weather)",
  forecast: "run the forecast (after weather_history, use simulate_world with values_from='weather_history')",
  simulate: "run the simulation with simulate_world",
  compare: "run the comparison: call simulate_world with a named scenario (e.g. a Cold snap shock of -15) so it is compared against the baseline",
  backtest: "run the backtest",
  train: "fit/train a model on the series (simulate_world with the user's values)",
  chart: "produce chartable data with weather_history, forecast, or simulate_world (what_if and create_report do not make chart data); the chart is drawn automatically",
  csv: "produce exportable data with weather_history, forecast, or simulate_world (what_if and create_report do not make CSV data); the CSV is attached automatically",
  explain: "explain the result in plain words",
};

export function deliverableLabel(item: Deliverable): string {
  return LABELS[item];
}

const SHORT_LABELS: Record<Deliverable, string> = {
  history: "the past data",
  forecast: "the forecast",
  simulate: "the simulation",
  compare: "the comparison",
  backtest: "the backtest",
  train: "the model fit",
  chart: "the chart",
  csv: "the CSV",
  explain: "the explanation",
};

/** Plain list for honest "couldn't finish" notes. */
export function shortLabels(items: Deliverable[]): string {
  const labels = items.map((item) => SHORT_LABELS[item]);
  if (labels.length <= 1) return labels.join("");
  return `${labels.slice(0, -1).join(", ")} and ${labels[labels.length - 1]}`;
}

/** Checklist block appended to tool notes for multi-step asks. */
export function checklistNote(plan: RequestPlan, results: ToolResult[]): string {
  const missing = missingDeliverables(plan, results).filter((item) => item !== "explain");
  const done = plan.deliverables.filter((item) => item !== "explain" && !missing.includes(item));
  const lines = [
    `Checklist for this request (${plan.steps.length} step${plan.steps.length === 1 ? "" : "s"}): ${plan.steps.map((step, i) => `${i + 1}) ${step}`).join(" ")}`,
  ];
  if (done.length) lines.push(`Done: ${done.map((item) => SHORT_LABELS[item]).join(", ")}.`);
  if (missing.length) {
    lines.push(
      `Still to do: ${missing.map(deliverableLabel).join("; ")}. Reply with ONLY the next tool JSON now (an array is fine for independent steps). Do not answer in prose yet and do not say you "will" do it.`,
    );
  } else {
    lines.push(
      "Every data step is done. Now write the final answer covering every step the user asked for" +
        (plan.deliverables.includes("explain") ? ", including a short plain-words explanation" : "") +
        ". Keep it tight: lead with the key numbers, summarize the series instead of listing every value, use plain words, no model jargon. " +
        (plan.deliverables.includes("chart") || plan.deliverables.includes("csv")
          ? "The requested chart/CSV is attached automatically from the tool data, so just mention it; never say you cannot chart or export."
          : "Do not mention charts or files; none are attached."),
    );
  }
  return lines.join("\n");
}

function horizonFrom(question: string, fallback: number): number {
  const match = question.match(/\bnext\s+(\d{1,2})\b/i) || question.match(/\b(\d{1,2})[\s-]+(?:days?|steps?)\s+(?:ahead|out)\b/i);
  const value = match ? Number(match[1]) : fallback;
  return Number.isFinite(value) ? Math.min(60, Math.max(1, Math.floor(value))) : fallback;
}

function daysFrom(question: string): number {
  const match = question.match(/\b(?:last|past|previous)\s+(\d{1,3})\s+days?\b/i);
  if (match) return Math.min(92, Math.max(7, Number(match[1])));
  const weeks = question.match(/\b(?:last|past|previous)\s+(\d{1,2})\s+weeks?\b/i);
  if (weeks) return Math.min(92, Math.max(7, Number(weeks[1]) * 7));
  if (/\b(?:last|past|previous)\s+month\b/i.test(question)) return 30;
  return 30;
}

function starterFrom(question: string): string {
  const q = question.toLowerCase();
  if (/\b(bitcoin|btc|crypto|eth)\b/.test(q)) return "crypto";
  if (/\b(stocks?|aapl|s&p|market)\b/.test(q)) return "stocks";
  if (/\b(moon|lunar)\b/.test(q)) return "lunar";
  if (/\b(asteroid|near-?earth)\b/.test(q)) return "near-earth";
  return "weather";
}

/** Numbers the user pasted inline (oldest first), when there are enough to model. */
export function pastedSeries(question: string): number[] {
  const matches = question.match(/-?\d+(?:\.\d+)?/g) || [];
  const values = matches.map(Number).filter((value) => Number.isFinite(value));
  return values.length >= 12 ? values : [];
}

/** Default scenario for a named ask (cold snap, heat wave, crash, spike, volatility). */
export function scenarioFrom(question: string): Record<string, unknown> | null {
  const q = question.toLowerCase();
  if (/cold\s*snap|freeze|cold front/.test(q)) {
    return { name: "Cold snap", events: [{ kind: "shock", step: 1, value: -15 }] };
  }
  if (/heat\s*wave|heatwave/.test(q)) {
    return { name: "Heat wave", events: [{ kind: "shock", step: 1, value: 15 }] };
  }
  if (/crash|drop|plunge/.test(q)) {
    return { name: "Drop", events: [{ kind: "shock", step: 1, value: -10, mode: "pct" }] };
  }
  if (/spike|jump|surge|rally/.test(q)) {
    return { name: "Jump", events: [{ kind: "shock", step: 1, value: 10, mode: "pct" }] };
  }
  if (/volatil|chaos|random/.test(q)) {
    return { name: "More randomness", events: [{ kind: "volatility", step: 1, value: 2 }] };
  }
  return null;
}

/**
 * Deterministic tool call for the next missing data step, used when the
 * model keeps answering in prose without doing it. Returns null when there is
 * no safe default (the loop then reports the gap honestly).
 */
export function routeMissingStep(
  missing: Deliverable[],
  question: string,
  results: ToolResult[],
  wanted: Deliverable[] = missing,
): ParsedToolCall | null {
  const q = question.toLowerCase();
  const weatherish = /\b(weather|temp\w*|highs?|lows?|cold|hot|rain|snow|°f|degrees)\b/.test(q);
  const pasted = pastedSeries(question);
  const haveHistory = okResults(results, "weather_history").length > 0;
  if (missing.includes("history") && weatherish) {
    return { id: "plan_history", name: "weather_history", args: { query: question, days: daysFrom(question) } };
  }
  const needsArtifactData = missing.includes("chart") || missing.includes("csv");
  const needsSim =
    (needsArtifactData && (wanted.includes("simulate") || wanted.includes("compare") || wanted.includes("train"))) ||
    missing.includes("simulate") ||
    missing.includes("compare") ||
    missing.includes("train") ||
    (missing.includes("forecast") && (pasted.length >= 24 || haveHistory)) ||
    (missing.includes("backtest") && pasted.length >= 24);
  if (needsSim) {
    const scenario = scenarioFrom(question);
    const args: Record<string, unknown> = { horizon: horizonFrom(question, 14) };
    if (pasted.length >= 24) {
      args.values = pasted;
    } else if (haveHistory) {
      args.values_from = "weather_history";
    } else {
      args.series = starterFrom(question);
    }
    if (scenario) args.scenarios = [scenario];
    return { id: "plan_simulate", name: "simulate_world", args };
  }
  if (missing.includes("forecast") || (needsArtifactData && wanted.includes("forecast") && !weatherish)) {
    return { id: "plan_forecast", name: "forecast", args: { series: starterFrom(question), horizon: Math.min(24, horizonFrom(question, 7)) } };
  }
  if (missing.includes("backtest")) {
    return { id: "plan_backtest", name: "get_evaluation_metrics", args: { model_id: starterFrom(question), horizon: 1 } };
  }
  if ((missing.includes("chart") || missing.includes("csv")) && weatherish && !haveHistory) {
    return { id: "plan_history", name: "weather_history", args: { query: question, days: daysFrom(question) } };
  }
  return null;
}

// ---------------------------------------------------------------------------
// Artifacts: chart spec (frontend charts.renderLineChart) and CSV text.

export type ChartSpec = {
  ariaLabel: string;
  labels: string[];
  series: Array<{ name: string; color: string; values: Array<number | null>; markers?: boolean }>;
  bands: Array<{ name: string; color: string; low: Array<number | null>; high: Array<number | null> }>;
  empty: string;
  unit?: string;
};

const COLORS = ["#7cffd6", "#ff9b7c", "#c3a6ff", "#ffd36b"];

function numbers(list: unknown): Array<number | null> {
  if (!Array.isArray(list)) return [];
  return list.map((value) => (typeof value === "number" && Number.isFinite(value) ? round(value) : null));
}

function round(value: number): number {
  return Math.round(value * 100) / 100;
}

function full(result: ToolResult): Record<string, unknown> {
  return record(record(result.data)._full);
}

/** Build one chart from this turn's tool data, newest relevant result wins. */
export function buildChart(results: ToolResult[]): ChartSpec | null {
  const history = [...okResults(results, "weather_history")].reverse()[0];
  const sim = [...okResults(results, "simulate_world")].reverse()[0];
  const forecast = [...okResults(results, "forecast")].reverse()[0];
  const whatIf = [...okResults(results, "what_if")]
    .reverse()
    .find((item) => numbers(record(record(item.data).shocked).p50).length > 0);
  const whatIfShocked = whatIf ? numbers(record(record(whatIf.data).shocked).p50) : [];
  const whatIfBase = whatIf ? numbers(record(record(whatIf.data).baseline).p50) : [];
  const whatIfName = whatIf ? whatIfLabel(record(whatIf.data)) : "";

  const labels: string[] = [];
  const actual: Array<number | null> = [];
  let unit = "";
  let title = "";

  const histFull = history ? full(history) : {};
  const histDates = Array.isArray(histFull.dates) ? (histFull.dates as unknown[]).map(String) : [];
  const histHighs = numbers(histFull.highs);
  if (history && histHighs.length) {
    unit = "°F";
    title = `Daily highs in ${String(record(history.data).place || "your area")}`;
  }

  if (sim) {
    const simFull = full(sim);
    const simHistory = Array.isArray(simFull.history) ? (simFull.history as Array<Record<string, unknown>>) : [];
    const baseline = record(simFull.baseline);
    const p50 = numbers(baseline.p50);
    if (!p50.length) return historyOnlyChart(histDates, histHighs, title, unit);
    unit = String(record(sim.data).unit || unit || "");
    const usedHistory = histHighs.length && record(sim.data).series === "custom" && record(sim.data).from_history === true;
    if (usedHistory) {
      histDates.forEach((date, i) => {
        labels.push(date);
        actual.push(histHighs[i] ?? null);
      });
    } else {
      simHistory.slice(-30).forEach((row, i) => {
        labels.push(typeof row.date === "string" ? row.date : `-${simHistory.slice(-30).length - i}`);
        actual.push(typeof row.value === "number" ? round(row.value) : null);
      });
    }
    const pad = labels.length;
    const future = (values: Array<number | null>) => [...new Array(pad).fill(null), ...values];
    for (let step = 1; step <= p50.length; step++) labels.push(`+${step}`);
    // Join the forecast to the last actual point so the line is continuous.
    const joinFuture = (values: Array<number | null>) => {
      const out = future(values);
      if (pad > 0 && actual[pad - 1] != null) out[pad - 1] = actual[pad - 1];
      return out;
    };
    const series: ChartSpec["series"] = [
      { name: "Actual", color: "#e8eaed", values: [...actual, ...new Array(p50.length).fill(null)] },
      { name: "Baseline (median)", color: COLORS[0], markers: true, values: joinFuture(p50) },
    ];
    const scenarios = Array.isArray(simFull.scenarios) ? (simFull.scenarios as Array<Record<string, unknown>>) : [];
    scenarios.slice(0, 3).forEach((scenario, i) => {
      series.push({
        name: String(scenario.name || `Scenario ${i + 1}`),
        color: COLORS[(i + 1) % COLORS.length],
        markers: true,
        values: joinFuture(numbers(scenario.p50)),
      });
    });
    if (!scenarios.length && whatIfShocked.length) {
      // The comparison came from what_if: show its shocked path on the same axis.
      const aligned = p50.map((_, i) => whatIfShocked[i] ?? null);
      series.push({ name: whatIfName, color: COLORS[1], markers: true, values: joinFuture(aligned) });
    }
    return {
      ariaLabel: title ? `${title} with simulated forecast` : "Simulated forecast",
      labels,
      series,
      bands: [{ name: "Baseline 80% range", color: COLORS[0], low: future(numbers(baseline.p10)), high: future(numbers(baseline.p90)) }],
      empty: "Not enough data to draw this chart.",
      unit,
    };
  }

  if (forecast) {
    const data = record(forecast.data);
    const rows = Array.isArray(data.forecast) ? (data.forecast as Array<Record<string, unknown>>) : [];
    const values = rows.map((row) => (typeof row.value === "number" ? round(row.value) : null));
    if (values.some((v) => v != null)) {
      const pad = histHighs.length;
      if (pad) histDates.forEach((date) => labels.push(date));
      rows.forEach((row, i) => labels.push(`+${typeof row.step === "number" ? row.step : i + 1}`));
      const series: ChartSpec["series"] = [];
      if (pad) series.push({ name: "Actual", color: "#e8eaed", values: [...histHighs, ...new Array(values.length).fill(null)] });
      series.push({ name: "Forecast", color: COLORS[0], markers: true, values: [...new Array(pad).fill(null), ...values] });
      return {
        ariaLabel: "Model forecast",
        labels,
        series,
        bands: [],
        empty: "Not enough data to draw this chart.",
        unit: typeof data.unit === "string" ? data.unit : unit,
      };
    }
  }

  if (whatIfShocked.length) {
    const data = record(whatIf!.data);
    const steps = Math.max(whatIfShocked.length, whatIfBase.length);
    return {
      ariaLabel: "Baseline versus what-if forecast",
      labels: Array.from({ length: steps }, (_, i) => `+${i + 1}`),
      series: [
        { name: "Baseline", color: COLORS[0], markers: true, values: Array.from({ length: steps }, (_, i) => whatIfBase[i] ?? null) },
        { name: whatIfName, color: COLORS[1], markers: true, values: Array.from({ length: steps }, (_, i) => whatIfShocked[i] ?? null) },
      ],
      bands: [],
      empty: "Not enough data to draw this chart.",
      unit: typeof data.unit === "string" ? data.unit : unit,
    };
  }

  return historyOnlyChart(histDates, histHighs, title, unit);
}

function whatIfLabel(data: Record<string, unknown>): string {
  const shock = record(data.shock);
  const value = typeof shock.value === "number" ? shock.value : null;
  if (value == null) return "What-if";
  const sign = value > 0 ? "+" : "";
  return shock.mode === "pct" ? `What-if (${sign}${value}%)` : `What-if (${sign}${value})`;
}

function historyOnlyChart(dates: string[], highs: Array<number | null>, title: string, unit: string): ChartSpec | null {
  if (!highs.length) return null;
  return {
    ariaLabel: title || "Recent history",
    labels: dates.length === highs.length ? dates : highs.map((_, i) => String(i + 1)),
    series: [{ name: "Daily high", color: "#e8eaed", markers: true, values: highs }],
    bands: [],
    empty: "Not enough data to draw this chart.",
    unit,
  };
}

function csvCell(value: unknown): string {
  if (value == null) return "";
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

/** CSV with one row per chart label and one column per line / band edge. */
export function chartToCsv(chart: ChartSpec): string {
  const unitSuffix = chart.unit ? ` (${chart.unit})` : "";
  const header = ["label", ...chart.series.map((s) => `${s.name}${unitSuffix}`)];
  for (const band of chart.bands) header.push(`${band.name} low${unitSuffix}`, `${band.name} high${unitSuffix}`);
  const rows = [header.map(csvCell).join(",")];
  chart.labels.forEach((label, i) => {
    const cells: unknown[] = [label, ...chart.series.map((s) => s.values[i] ?? null)];
    for (const band of chart.bands) cells.push(band.low[i] ?? null, band.high[i] ?? null);
    rows.push(cells.map(csvCell).join(","));
  });
  return `${rows.join("\n")}\n`;
}

export type TurnArtifacts = {
  chart?: ChartSpec;
  csv?: { filename: string; text: string };
};

/** Attach the chart / CSV the user asked for (only when asked). */
export function buildArtifacts(plan: RequestPlan, results: ToolResult[]): TurnArtifacts {
  const wantsChart = plan.deliverables.includes("chart");
  const wantsCsv = plan.deliverables.includes("csv");
  if (!wantsChart && !wantsCsv) return {};
  const chart = buildChart(results);
  if (!chart) return {};
  const out: TurnArtifacts = {};
  if (wantsChart) out.chart = chart;
  if (wantsCsv) out.csv = { filename: "neural-forge-data.csv", text: chartToCsv(chart) };
  return out;
}

const EVENT_KINDS = new Set(["shock", "drift", "volatility", "set"]);

/** True when at least one scenario carries a usable event (kind + numeric value). */
export function hasUsableScenario(raw: unknown): boolean {
  const list = Array.isArray(raw) ? raw : raw && typeof raw === "object" ? [raw] : [];
  return list.some((scenario) => {
    const events = record(scenario).events;
    return (
      Array.isArray(events) &&
      events.some((event) => {
        const e = record(event);
        return typeof e.kind === "string" && EVENT_KINDS.has(e.kind) && typeof e.value === "number" && Number.isFinite(e.value) && e.value !== 0;
      })
    );
  });
}
