/**
 * Quick baseline forecast for a short series read out of an image.
 *
 * Damped Holt (level + damped trend) with a small grid search over alpha and
 * beta on one-step errors. The p10/p90 band comes from the one-step residual
 * spread widened by sqrt(step). This is a fast first look, not the trained
 * transformer; "Train on this series" runs the real model.
 */

export type QuickStep = { step: number; p10: number; p50: number; p90: number };

export type QuickForecast = {
  method: "damped_holt";
  horizon: number;
  steps: QuickStep[];
  residual_mae: number;
  naive_mae: number;
};

export const MIN_FORECAST_POINTS = 4;
export const MAX_FORECAST_HORIZON = 14;
const PHI = 0.9;
const Z80 = 1.2816;

function round(value: number, digits = 4): number {
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

function fit(values: number[], alpha: number, beta: number) {
  let level = values[0];
  let trend = values.length > 1 ? values[1] - values[0] : 0;
  const errors: number[] = [];
  for (let index = 1; index < values.length; index += 1) {
    const predicted = level + PHI * trend;
    const actual = values[index];
    errors.push(actual - predicted);
    const nextLevel = alpha * actual + (1 - alpha) * predicted;
    trend = beta * (nextLevel - level) + (1 - beta) * PHI * trend;
    level = nextLevel;
  }
  const sse = errors.reduce((sum, error) => sum + error * error, 0);
  return { level, trend, errors, sse };
}

export function defaultHorizon(count: number): number {
  return Math.max(3, Math.min(7, Math.round(count / 3)));
}

export function quickForecast(raw: number[], horizon?: number): QuickForecast | null {
  const values = raw.filter((value) => Number.isFinite(value));
  if (values.length < MIN_FORECAST_POINTS) return null;
  const steps = Math.max(
    1,
    Math.min(MAX_FORECAST_HORIZON, Math.floor(horizon ?? defaultHorizon(values.length))),
  );
  let best = fit(values, 0.5, 0.1);
  for (const alpha of [0.2, 0.35, 0.5, 0.65, 0.8, 0.95]) {
    for (const beta of [0.05, 0.1, 0.2, 0.35]) {
      const candidate = fit(values, alpha, beta);
      if (candidate.sse < best.sse) best = candidate;
    }
  }
  const absErrors = best.errors.map((error) => Math.abs(error));
  const residualMae = absErrors.reduce((sum, value) => sum + value, 0) / Math.max(1, absErrors.length);
  const naive: number[] = [];
  for (let index = 1; index < values.length; index += 1) {
    naive.push(Math.abs(values[index] - values[index - 1]));
  }
  const naiveMae = naive.reduce((sum, value) => sum + value, 0) / Math.max(1, naive.length);
  const variance =
    best.errors.reduce((sum, error) => sum + error * error, 0) / Math.max(1, best.errors.length);
  const sigma = Math.sqrt(variance);
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const span = Math.max(hi - lo, Math.abs(hi) * 0.05, 1e-6);
  const out: QuickStep[] = [];
  let damp = 0;
  for (let step = 1; step <= steps; step += 1) {
    damp += PHI ** step;
    // Keep the median within the observed range plus half its span; image
    // reads are short and a runaway trend would mislead more than help.
    const p50 = Math.min(hi + span / 2, Math.max(lo - span / 2, best.level + damp * best.trend));
    const width = Z80 * sigma * Math.sqrt(step);
    out.push({ step, p10: round(p50 - width), p50: round(p50), p90: round(p50 + width) });
  }
  return {
    method: "damped_holt",
    horizon: steps,
    steps: out,
    residual_mae: round(residualMae),
    naive_mae: round(naiveMae),
  };
}
