const charts = window.NeuralForgeCharts;
const TOKEN_KEY = "neuralForgeApiToken";
const COLORS = ["#7cffd6", "#ffd166", "#9db4ff", "#ff8f8f", "#d6c2ff"];
const FORECAST_PATHS = [
  "/forecast",
  "/forecast/latest",
  "/model/forecast",
  "/predict/forecast",
  "/inference/forecast",
];
const COMPARISON_PATHS = [
  "/comparison",
  "/models/compare",
  "/experiments/compare",
  "/model/comparison",
];

const state = {
  health: null,
  modelInfo: null,
  training: null,
  experiments: [],
  series: null,
  steps: [],
  selectedId: null,
  selectedStamp: null,
  comparisonRows: null,
  detailToken: 0,
};

let holdTrainingMessage = false;
let lastTrainingStatus = null;
let openapiPromise = null;

function errorMessage(response, data, text) {
  const detail = data && typeof data === "object" ? data.detail : null;
  if (typeof detail === "string" && detail) return detail;
  if (Array.isArray(detail)) {
    const parts = detail.map((item) => {
      if (item && typeof item === "object" && typeof item.msg === "string") return item.msg;
      return String(item);
    });
    const joined = parts.filter(Boolean).join("; ");
    if (joined) return joined;
  } else if (detail != null) {
    return JSON.stringify(detail);
  }
  const body = text.trim();
  return body || response.statusText || `Request failed (${response.status})`;
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, { credentials: "same-origin", ...options });
  const text = await response.text();
  let data = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      const error = new Error(
        response.ok
          ? `Expected JSON but received a non-JSON response (${response.status})`
          : text.trim() || response.statusText || `Request failed (${response.status})`
      );
      error.status = response.status;
      throw error;
    }
  }
  if (!response.ok) {
    const error = new Error(errorMessage(response, data, text));
    error.status = response.status;
    throw error;
  }
  return data;
}

function asNumber(value) {
  if (typeof value === "boolean" || value == null || value === "") return null;
  const number = typeof value === "number" ? value : Number(value);
  return Number.isFinite(number) ? number : null;
}

function formatMetric(value) {
  const number = asNumber(value);
  if (number == null) return "—";
  const abs = Math.abs(number);
  if (abs !== 0 && (abs < 0.001 || abs >= 10000)) return number.toExponential(2);
  return number.toLocaleString(undefined, { maximumSignificantDigits: 4 });
}

function formatSkill(value) {
  if (value == null || !Number.isFinite(value)) return "—";
  const rounded = Math.round(value * 100);
  return `${rounded > 0 ? "+" : ""}${rounded}%`;
}

function reportError(error) {
  document.getElementById("notice").textContent = error && error.message ? error.message : "Request failed.";
}

function clearNotice() {
  document.getElementById("notice").textContent = "";
}

function fillStatGrid(grid, cards) {
  grid.replaceChildren(
    ...cards.map(([label, value]) => {
      const card = document.createElement("div");
      card.className = "stat-card";
      const span = document.createElement("span");
      span.textContent = label;
      const strong = document.createElement("strong");
      strong.textContent = value == null ? "—" : String(value);
      card.append(span, strong);
      return card;
    })
  );
}

function checkpointLabel(modelInfo) {
  const loaded = Boolean(modelInfo.checkpoint_loaded);
  const stale = Boolean(modelInfo.checkpoint_stale);
  if (loaded && stale) return "newer file on disk";
  if (stale) return "file not loaded";
  return loaded ? "loaded" : "missing";
}

function numericArray(value) {
  if (!Array.isArray(value)) return null;
  const numbers = value.map(asNumber);
  return numbers.some((number) => number != null) ? numbers : null;
}

function zipSteps(values, p10, p50, p90) {
  const length = Math.max(
    values ? values.length : 0,
    p10 ? p10.length : 0,
    p50 ? p50.length : 0,
    p90 ? p90.length : 0
  );
  const steps = [];
  for (let index = 0; index < length; index += 1) {
    const mid =
      p50 && p50[index] != null ? p50[index] : values && values[index] != null ? values[index] : null;
    const low = p10 ? p10[index] : null;
    const high = p90 ? p90[index] : null;
    if (mid == null && low == null && high == null) continue;
    steps.push({
      step: index + 1,
      value: mid,
      p10: low == null ? null : low,
      p50: p50 ? p50[index] : null,
      p90: high == null ? null : high,
    });
  }
  return steps;
}

function stepFromObject(row, index) {
  if (!row || typeof row !== "object" || Array.isArray(row)) return null;
  const p50 = asNumber(row.p50 ?? row.median);
  const value = asNumber(row.value ?? row.prediction ?? row.predicted ?? p50);
  const p10 = asNumber(row.p10 ?? row.lower);
  const p90 = asNumber(row.p90 ?? row.upper);
  if (value == null && p10 == null && p90 == null) return null;
  return {
    step: asNumber(row.step) ?? index + 1,
    value: value ?? p50,
    p10,
    p50,
    p90,
  };
}

function stepsFromObjects(payload) {
  const candidates = [payload.forecast, payload.horizon, payload.intervals];
  for (const candidate of candidates) {
    if (!Array.isArray(candidate) || !candidate.length || typeof candidate[0] !== "object") {
      continue;
    }
    const steps = candidate.map(stepFromObject).filter(Boolean);
    if (steps.length) return steps;
  }
  return [];
}

function stepsFromArrays(payload) {
  const intervals = payload.intervals && !Array.isArray(payload.intervals) ? payload.intervals : {};
  const p10 = numericArray(payload.p10) || numericArray(intervals.p10);
  const p50 = numericArray(payload.p50) || numericArray(intervals.p50);
  const p90 = numericArray(payload.p90) || numericArray(intervals.p90);
  const forecast = numericArray(payload.forecast);
  const prediction = numericArray(payload.prediction);
  if (forecast) return zipSteps(forecast, p10, p50, p90);
  const explicitHorizon =
    payload.multi_step === true || (typeof payload.horizon === "number" && payload.horizon > 1);
  if (prediction && (p10 || p90 || explicitHorizon)) return zipSteps(prediction, p10, p50, p90);
  if (p50 && (p10 || p90)) return zipSteps(p50, p10, p50, p90);
  if (prediction && prediction[0] != null) {
    return [
      {
        step: 1,
        value: prediction[0],
        p10: p10 ? p10[0] : null,
        p50: p50 ? p50[0] : null,
        p90: p90 ? p90[0] : null,
      },
    ];
  }
  return [];
}

function normalizeForecast(payload) {
  if (!payload || typeof payload !== "object") return [];
  const objects = stepsFromObjects(payload);
  if (objects.length) return objects;
  return stepsFromArrays(payload);
}

function hasInterval(steps) {
  return steps.some((step) => step.p10 != null && step.p90 != null);
}

function preferRicher(current, extra) {
  if (!extra || !extra.length) return current || [];
  if (!current || !current.length) return extra;
  if (extra.length > current.length) return extra;
  if (hasInterval(extra) && !hasInterval(current)) return extra;
  if (hasInterval(extra) && extra.length === current.length) return extra;
  return current;
}

function rowsFromComparison(payload) {
  const list = Array.isArray(payload)
    ? payload
    : payload && (payload.models || payload.results || payload.runs || payload.comparison);
  if (!Array.isArray(list)) return [];
  const metric = ["mae", "rmse", "mse"].find((key) =>
    list.some((row) => row && asNumber(row[key] ?? (row.metrics && row.metrics[key])) != null)
  );
  if (!metric) return [];
  return list.flatMap((row) => {
    if (!row || typeof row !== "object") return [];
    const value = asNumber(row[metric] ?? (row.metrics && row.metrics[metric]));
    const name = row.name ?? row.model ?? row.model_name;
    if (value == null || name == null) return [];
    return [{ name: String(name), value }];
  });
}

function baselineLabel(name) {
  const known = {
    persistence: "Persistence",
    mean: "Mean",
    model: "Model",
    seasonal_naive: "Seasonal naive",
  };
  if (Object.prototype.hasOwnProperty.call(known, name)) return known[name];
  return String(name).replaceAll("_", " ");
}

function rowsFromExperiment(experiment) {
  const metrics = experiment && experiment.metrics;
  if (!metrics || typeof metrics !== "object" || Array.isArray(metrics)) return [];
  const comparison = rowsFromComparison(metrics.comparison || metrics.model_comparison);
  if (comparison.length > 1) return comparison;
  const rows = [];
  const modelMae = asNumber(metrics.mae);
  if (modelMae != null) {
    rows.push({ name: experiment.model_name ? String(experiment.model_name) : "Model", value: modelMae });
  }
  const baselines = metrics.baselines;
  if (baselines && typeof baselines === "object" && !Array.isArray(baselines)) {
    Object.entries(baselines).forEach(([name, block]) => {
      if (!block || typeof block !== "object") return;
      const value = asNumber(block.mae ?? block.rmse ?? block.mse);
      if (value != null) rows.push({ name: baselineLabel(name), value });
    });
  }
  if (rows.length > 1) return rows;
  return rowsFromFlatBaselines(metrics, experiment.model_name);
}

function rowsFromFlatBaselines(metrics, modelName) {
  const rows = [];
  const modelMae = asNumber(metrics.mae);
  if (modelMae != null) {
    rows.push({ name: modelName ? String(modelName) : "Model", value: modelMae });
  }
  Object.keys(metrics)
    .filter((key) => /^baseline_.+_mae$/.test(key))
    .sort()
    .forEach((key) => {
      const value = asNumber(metrics[key]);
      if (value == null) return;
      rows.push({
        name: baselineLabel(key.slice("baseline_".length, -"_mae".length)),
        value,
      });
    });
  return rows.length > 1 ? rows : [];
}

function seriesRows(metrics) {
  if (!metrics || typeof metrics !== "object") return [];
  const rows = [];
  if (metrics.model && asNumber(metrics.model.mae) != null) {
    rows.push({ name: "Model", value: metrics.model.mae });
  }
  const baselines = metrics.baselines || {};
  Object.entries(baselines).forEach(([name, block]) => {
    const value = block && asNumber(block.mae);
    if (value != null) rows.push({ name: baselineLabel(name), value });
  });
  return rows;
}

function accuracyView() {
  if (state.comparisonRows && state.comparisonRows.length > 1) {
    return {
      rows: state.comparisonRows,
      source: "Reported by the model comparison endpoint.",
    };
  }
  const selected = state.experiments.find((experiment) => experiment.id === state.selectedId);
  const stored = rowsFromExperiment(selected);
  if (stored.length > 1) {
    return { rows: stored, source: "From the selected run in the experiment store." };
  }
  const recent = seriesRows(state.series && state.series.metrics);
  if (recent.length) {
    return {
      rows: recent,
      source:
        "From one-step predictions on the recent window. A run that stores baseline or comparison metrics will replace these.",
    };
  }
  return { rows: [], source: "No accuracy metrics yet." };
}

function coloredRows(rows) {
  return rows.map((row, index) => ({
    name: row.name,
    value: row.value,
    color: COLORS[index % COLORS.length],
  }));
}

function renderStatus() {
  const modelInfo = state.modelInfo;
  const health = state.health;
  const training = state.training;
  if (!modelInfo || !health || !training) return;
  const cards = [
    ["API", health.status],
    ["Model", modelInfo.architecture],
    ["Parameters", Number(modelInfo.parameters).toLocaleString()],
    ["Device", modelInfo.device],
    ["Checkpoint", checkpointLabel(modelInfo)],
    ["Training", training.status],
  ];
  if (state.series && state.series.dataset) cards.push(["Dataset", state.series.dataset]);
  fillStatGrid(document.getElementById("status-grid"), cards);
  const line = document.getElementById("status-line");
  line.textContent = modelInfo.checkpoint_loaded
    ? "Checkpoint loaded"
    : "No checkpoint loaded";
}

function renderAccuracy() {
  const view = accuracyView();
  document.getElementById("accuracy-caption").textContent = view.source;
  const model = view.rows.find((row) => /model|transformer/i.test(row.name));
  const persistence = view.rows.find((row) => /persistence/i.test(row.name));
  const mean = view.rows.find((row) => /^mean$/i.test(row.name));
  const skill =
    model && persistence && persistence.value
      ? 1 - model.value / persistence.value
      : null;
  const cards = view.rows.slice(0, 3).map((row) => [row.name, formatMetric(row.value)]);
  cards.push(["Skill vs persistence", formatSkill(skill)]);
  if (mean && !cards.some(([label]) => label === mean.name)) {
    cards.splice(cards.length - 1, 0, [mean.name, formatMetric(mean.value)]);
  }
  fillStatGrid(document.getElementById("accuracy-grid"), cards.length ? cards : [["Accuracy", "—"]]);
  charts.renderBarChart(document.getElementById("accuracy-chart"), {
    ariaLabel: "Accuracy compared with baselines. Lower error is better.",
    rows: coloredRows(view.rows),
    valueLabel: "Error",
    empty: "No baseline comparison is available yet.",
  });
}

function renderForecast() {
  const series = state.series;
  const steps = state.steps || [];
  const kicker = document.getElementById("forecast-kicker");
  const summary = document.getElementById("forecast-summary");
  const note = document.getElementById("forecast-note");
  if (!series) {
    kicker.textContent = "";
    summary.textContent = "";
    note.textContent = "";
    charts.renderLineChart(document.getElementById("forecast-chart"), {
      ariaLabel: "Forecast",
      empty: "Recent history is not available.",
    });
    return;
  }
  kicker.textContent = `${series.dataset || series.source || "Series"} · ${series.target || "target"}`;
  const history = Array.isArray(series.history) ? series.history : [];
  const labels = [];
  const actual = [];
  const backtest = [];
  history.forEach((point) => {
    labels.push(point.time || String(point.index));
    actual.push(asNumber(point.actual));
    backtest.push(asNumber(point.predicted));
  });
  const forecast = new Array(history.length).fill(null);
  const low = new Array(history.length).fill(null);
  const high = new Array(history.length).fill(null);
  const band = hasInterval(steps);
  const lastActual = actual.length ? actual[actual.length - 1] : null;
  if (lastActual != null && steps.length) {
    forecast[history.length - 1] = lastActual;
    if (band) {
      low[history.length - 1] = lastActual;
      high[history.length - 1] = lastActual;
    }
  }
  steps.forEach((step, index) => {
    labels.push(`+${step.step || index + 1}`);
    actual.push(null);
    backtest.push(null);
    forecast.push(step.p50 != null ? step.p50 : step.value);
    low.push(step.p10);
    high.push(step.p90);
  });
  const chartSeries = [
    { name: "Actual", color: "#e8f4ff", values: actual },
  ];
  if (backtest.some((value) => value != null)) {
    chartSeries.push({
      name: "Backtest",
      color: "#ffd166",
      values: backtest,
      dashed: true,
    });
  }
  if (forecast.some((value) => value != null)) {
    chartSeries.push({
      name: "Forecast",
      color: "#7cffd6",
      values: forecast,
      markers: true,
    });
  }
  const bands = band
    ? [{ name: "p10–p90", color: "#7cffd6", low, high }]
    : [];
  charts.renderLineChart(document.getElementById("forecast-chart"), {
    ariaLabel: `Forecast for ${series.dataset || "the loaded series"}`,
    labels,
    series: chartSeries,
    bands,
    empty: "Not enough history to draw a forecast.",
  });
  charts.renderLineChart(document.getElementById("backtest-chart"), {
    ariaLabel: "Recent predictions compared with actual values",
    labels: history.map((point) => point.time || String(point.index)),
    series: [
      { name: "Actual", color: "#e8f4ff", values: history.map((point) => asNumber(point.actual)) },
      {
        name: "Predicted",
        color: "#7cffd6",
        values: history.map((point) => asNumber(point.predicted)),
        dashed: true,
      },
    ],
    empty: "No overlapping predictions and actuals on this window.",
  });

  const next = steps[0];
  const nextValue = next ? (next.p50 != null ? next.p50 : next.value) : null;
  const parts = [];
  if (nextValue != null) {
    parts.push(
      steps.length > 1
        ? `${steps.length}-step forecast starts at ${formatMetric(nextValue)}.`
        : `Next value ${formatMetric(nextValue)}.`
    );
  }
  if (band) parts.push("Shaded band is the p10–p90 prediction interval.");
  else if (steps.length) parts.push("This response has no p10–p90 interval.");
  else parts.push("No forecast was returned.");
  summary.textContent = parts.join(" ");
  const windowNote = windowSummary(series.metrics);
  const seriesNote = typeof series.note === "string" ? series.note : "";
  note.textContent = [seriesNote, windowNote].filter(Boolean).join(" ");
  const caption = document.getElementById("backtest-caption");
  caption.textContent = windowNote
    ? `One-step predictions on the recent window. ${windowNote}`
    : "One-step predictions on the recent window, plotted against what happened.";
}

function windowSummary(metrics) {
  if (!metrics || !metrics.model || asNumber(metrics.model.mae) == null) return "";
  const persistence = metrics.baselines && metrics.baselines.persistence;
  const persistenceText =
    persistence && asNumber(persistence.mae) != null
      ? ` Persistence MAE ${formatMetric(persistence.mae)}.`
      : "";
  return `Recent window MAE ${formatMetric(metrics.model.mae)}.${persistenceText}`;
}

function flattenMetrics(value, prefix, depth, out) {
  if (!value || typeof value !== "object" || Array.isArray(value) || depth > 2) return;
  Object.entries(value).forEach(([key, child]) => {
    if (key === "validation_predictions" || key === "predictions" || key === "points") return;
    const name = prefix ? `${prefix} ${key}` : key;
    if (typeof child === "number" && Number.isFinite(child)) out[name] = child;
    else flattenMetrics(child, name, depth + 1, out);
  });
}

function metricColumns(experiments) {
  const skip = new Set(["train_loss", "val_loss", "epoch"]);
  const keys = new Set();
  experiments.forEach((experiment) => {
    const flat = {};
    flattenMetrics(experiment.metrics, "", 0, flat);
    Object.keys(flat).forEach((key) => {
      if (!skip.has(key)) keys.add(key);
    });
  });
  const preferred = ["mae", "rmse", "mse"];
  const baselineMae = [...keys].filter((key) => /^baseline_.+_mae$/.test(key)).sort();
  const rest = [...keys]
    .filter((key) => !preferred.includes(key) && !baselineMae.includes(key))
    .sort((a, b) => a.localeCompare(b));
  const ordered = ["mae", ...baselineMae, "rmse", "mse", ...rest].filter(
    (key, index, all) => keys.has(key) && all.indexOf(key) === index
  );
  return ordered.slice(0, 8);
}

function renderExperiments() {
  const experiments = state.experiments;
  const columns = metricColumns(experiments);
  const head = document.getElementById("experiments-head");
  const body = document.getElementById("experiments-body");
  const header = document.createElement("tr");
  ["Run", "Model", "Status", "When", "Best loss", ...columns].forEach((label) => {
    const cell = document.createElement("th");
    cell.scope = "col";
    cell.textContent = label;
    header.append(cell);
  });
  head.replaceChildren(header);

  if (!experiments.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 5 + columns.length;
    cell.textContent = "No experiments yet. Start a training run.";
    row.append(cell);
    body.replaceChildren(row);
  } else {
    body.replaceChildren(
      ...experiments.map((experiment) => {
        const row = document.createElement("tr");
        if (experiment.id === state.selectedId) row.classList.add("is-selected");
        const nameCell = document.createElement("th");
        nameCell.scope = "row";
        const button = document.createElement("button");
        button.type = "button";
        button.className = "row-button";
        button.textContent = experiment.name || `Run ${experiment.id}`;
        button.setAttribute("aria-pressed", experiment.id === state.selectedId ? "true" : "false");
        button.setAttribute("aria-controls", "experiment-detail");
        button.addEventListener("click", () => selectExperiment(experiment.id));
        nameCell.append(button);

        const model = document.createElement("td");
        model.textContent = experiment.model_name || "—";
        const statusCell = document.createElement("td");
        const badge = document.createElement("span");
        const status = String(experiment.status || "");
        badge.className = "badge";
        if (status === "running" || status === "completed" || status === "failed") {
          badge.classList.add(status);
        }
        badge.textContent = status || "—";
        statusCell.append(badge);
        const when = document.createElement("td");
        when.textContent = experiment.created_at ? String(experiment.created_at).slice(0, 19) : "—";
        if (experiment.created_at) when.title = String(experiment.created_at);
        const loss = document.createElement("td");
        loss.textContent = formatMetric(experiment.best_loss);
        row.append(nameCell, model, statusCell, when, loss);

        const flat = {};
        flattenMetrics(experiment.metrics, "", 0, flat);
        columns.forEach((key) => {
          const cell = document.createElement("td");
          cell.textContent = formatMetric(flat[key]);
          row.append(cell);
        });
        return row;
      })
    );
  }

  const key = comparisonKey(experiments);
  const rows = experiments
    .slice(0, 8)
    .flatMap((experiment) => {
      const value = experimentMetric(experiment, key);
      if (value == null) return [];
      return [{ name: `${experiment.name || "run"} #${experiment.id}`, value, color: "#9db4ff" }];
    });
  document.getElementById("experiments-caption").textContent = key
    ? `Select a run to compare metrics and open its validation predictions. Bars show ${key}.`
    : "Select a run to compare metrics and open its validation predictions.";
  charts.renderBarChart(document.getElementById("experiments-chart"), {
    ariaLabel: key ? `Experiment comparison by ${key}` : "Experiment comparison",
    rows,
    valueLabel: key || "Metric",
    empty: "Train a run to compare metrics.",
  });
}

function comparisonKey(experiments) {
  const flats = experiments.map((experiment) => {
    const flat = {};
    flattenMetrics(experiment.metrics, "", 0, flat);
    return flat;
  });
  let best = null;
  let bestCount = 0;
  ["mae", "rmse", "mse"].forEach((key) => {
    const count = flats.filter((flat) => asNumber(flat[key]) != null).length;
    if (count > bestCount) {
      best = key;
      bestCount = count;
    }
  });
  if (best) return best;
  if (experiments.some((experiment) => asNumber(experiment.best_loss) != null)) return "best loss";
  return null;
}

function experimentMetric(experiment, key) {
  if (!key) return null;
  if (key === "best loss") return asNumber(experiment.best_loss);
  const flat = {};
  flattenMetrics(experiment.metrics, "", 0, flat);
  return asNumber(flat[key]);
}

function renderYouneek(now) {
  fillStatGrid(document.getElementById("youneek-grid"), [
    ["Clock", now.time.display],
    ["Year clock", now.calendar.year_clock.display],
    [
      "Year / week / day",
      `${now.calendar.year_index} / ${now.calendar.week} / ${now.calendar.weekday}`,
    ],
    ["Lunar", now.lunar.clock.display],
    ["Next minute", now.forecast.next_minute.youneek_display],
    ["Scale", "100/100/100"],
  ]);
  document.getElementById("youneek-output").textContent = JSON.stringify(now, null, 2);
}

function renderLoss(history) {
  const epochs = [];
  const train = new Map();
  const val = new Map();
  (history || []).forEach((row) => {
    const epoch = asNumber(row.epoch);
    if (epoch == null) return;
    if (!epochs.includes(epoch)) epochs.push(epoch);
    if (row.name === "train_loss") train.set(epoch, asNumber(row.value));
    if (row.name === "val_loss") val.set(epoch, asNumber(row.value));
  });
  epochs.sort((a, b) => a - b);
  charts.renderLineChart(document.getElementById("loss-chart"), {
    ariaLabel: "Training and validation loss by epoch",
    labels: epochs.map(String),
    series: [
      { name: "Train loss", color: "#9db4ff", values: epochs.map((epoch) => train.get(epoch) ?? null) },
      { name: "Val loss", color: "#ffd166", values: epochs.map((epoch) => val.get(epoch) ?? null) },
    ],
    empty: "No per-epoch loss was stored for this run.",
  });
}

function renderValidation(payload) {
  const note = document.getElementById("validation-note");
  note.textContent = payload && payload.note ? String(payload.note) : "";
  const points = payload && Array.isArray(payload.points) ? payload.points : [];
  if (!payload || !payload.available || !points.length) {
    charts.renderLineChart(document.getElementById("validation-chart"), {
      ariaLabel: "Validation predictions",
      empty: "No validation predictions for this run.",
    });
    return;
  }
  const low = points.map((point) => asNumber(point.p10));
  const high = points.map((point) => asNumber(point.p90));
  const band = low.some((value, index) => value != null && high[index] != null);
  charts.renderLineChart(document.getElementById("validation-chart"), {
    ariaLabel: "Validation predictions compared with actuals",
    labels: points.map((point) => String(point.index)),
    series: [
      { name: "Actual", color: "#e8f4ff", values: points.map((point) => asNumber(point.actual)) },
      {
        name: "Predicted",
        color: "#7cffd6",
        values: points.map((point) => asNumber(point.predicted ?? point.p50)),
        markers: true,
      },
    ],
    bands: band ? [{ name: "p10–p90", color: "#7cffd6", low, high }] : [],
    empty: "No validation predictions for this run.",
  });
}

async function loadExperimentDetail(id) {
  const token = state.detailToken + 1;
  state.detailToken = token;
  try {
    const [detail, predictions] = await Promise.all([
      fetchJson(`/experiments/${id}`),
      fetchJson(`/experiments/${id}/predictions?limit=200`),
    ]);
    if (token !== state.detailToken || id !== state.selectedId) return;
    const heading = document.getElementById("experiment-detail-heading");
    heading.textContent = detail.name ? String(detail.name) : "Run detail";
    renderLoss(detail.metric_history);
    renderValidation(predictions);
    renderAccuracy();
  } catch (error) {
    if (token !== state.detailToken) return;
    document.getElementById("validation-note").textContent = error.message;
    reportError(error);
  }
}

function selectExperiment(id) {
  state.selectedId = id;
  const selected = state.experiments.find((experiment) => experiment.id === id);
  state.selectedStamp = selected ? selected.updated_at : null;
  renderExperiments();
  renderAccuracy();
  loadExperimentDetail(id);
}

function openapiPaths() {
  if (!openapiPromise) {
    openapiPromise = fetchJson("/openapi.json")
      .then((spec) => (spec && spec.paths) || {})
      .catch(() => ({}));
  }
  return openapiPromise;
}

async function fetchOptionalForecast(paths, predictWindow) {
  for (const path of FORECAST_PATHS) {
    const item = paths[path];
    if (!item) continue;
    try {
      if (item.post && predictWindow && predictWindow.length) {
        return await fetchJson(path, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ sequence: predictWindow, horizon: 7 }),
        });
      }
      if (item.get) return await fetchJson(path);
    } catch {
      // This optional route did not accept the request. Keep the base forecast.
    }
  }
  return null;
}

async function fetchComparison(paths) {
  for (const path of COMPARISON_PATHS) {
    if (!paths[path] || !paths[path].get) continue;
    try {
      return rowsFromComparison(await fetchJson(path));
    } catch {
      // Try the next comparison route the API happens to expose.
    }
  }
  return null;
}

async function refreshSeries() {
  const series = await fetchJson("/series/recent?points=120");
  state.series = series;
  let steps = normalizeForecast({ forecast: series.forecast });
  if (Array.isArray(series.predict_window) && series.predict_window.length) {
    try {
      const predicted = await fetchJson("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sequence: series.predict_window }),
      });
      steps = preferRicher(steps, normalizeForecast(predicted));
    } catch {
      // The one-step forecast from /series/recent still draws.
    }
  }
  const paths = await openapiPaths();
  const extra = await fetchOptionalForecast(paths, series.predict_window);
  if (extra) steps = preferRicher(steps, normalizeForecast(extra));
  if (state.modelInfo && state.modelInfo.forecast) {
    steps = preferRicher(steps, normalizeForecast(state.modelInfo.forecast));
  }
  state.steps = steps;
  renderStatus();
  renderForecast();
  renderAccuracy();
}

function showAccount(account) {
  state.account = account;
  const signedIn = document.getElementById("account-signed-in");
  const signedOut = document.getElementById("account-signed-out");
  const label = document.getElementById("account-label");
  if (!account) {
    signedIn.hidden = true;
    signedOut.hidden = false;
    label.textContent = "";
    return;
  }
  signedIn.hidden = false;
  signedOut.hidden = account.role === "member";
  const auth = account.auth === "local" ? "local owner" : account.role;
  label.textContent = `${account.email} (${auth})`;
}

function appendTextItem(list, text) {
  const item = document.createElement("li");
  item.textContent = text;
  list.append(item);
}

function renderWorkspace(payload) {
  const starters = (payload && payload.starters) || [];
  const host = document.getElementById("starter-list");
  host.replaceChildren();
  const note = document.getElementById("getting-started-note");
  if (!starters.length) {
    note.textContent = state.account && state.account.role === "owner"
      ? "Starter samples are added to new visitor accounts. This owner workspace keeps its existing runs."
      : "Sign in to load the sample datasets and starter models.";
    return;
  }
  note.textContent = "Each sample is yours. Predict uses the shared starter weights. Train on this sample saves a private model and leaves the original in place.";
  for (const starter of starters) {
    const card = document.createElement("article");
    card.className = "starter-card";
    const title = document.createElement("h3");
    title.textContent = starter.name;
    const summary = document.createElement("p");
    summary.className = "hint";
    summary.textContent = starter.summary;
    const output = document.createElement("pre");
    output.className = "output";
    const predict = document.createElement("button");
    predict.type = "button";
    predict.className = "btn secondary";
    predict.textContent = "Predict";
    predict.addEventListener("click", () => {
      runStarter(starter.id, "predict", output);
    });
    const train = document.createElement("button");
    train.type = "button";
    train.className = "btn primary";
    train.textContent = "Train on this sample";
    train.addEventListener("click", () => {
      runStarter(starter.id, "train", output);
    });
    card.append(title, summary, predict, train, output);
    host.append(card);
  }
}

async function runStarter(starterId, action, output) {
  try {
    const result = await fetchJson(`/workspace/starters/${starterId}/${action}`, { method: "POST" });
    output.textContent = JSON.stringify(result, null, 2);
    clearNotice();
    if (action === "train") refreshWorkspace().catch(reportError);
  } catch (error) {
    output.textContent = error.message;
  }
}

function renderWorkspaceLists(datasets, models, predictions) {
  const datasetList = document.getElementById("dataset-list");
  const modelList = document.getElementById("model-list");
  const predictionList = document.getElementById("prediction-list");
  datasetList.replaceChildren();
  modelList.replaceChildren();
  predictionList.replaceChildren();
  for (const dataset of datasets) {
    const shared = dataset.shared ? "shared starter" : "private copy";
    appendTextItem(datasetList, `${dataset.name} · ${shared} · ${dataset.bytes} bytes`);
  }
  if (!datasets.length) appendTextItem(datasetList, "No datasets yet.");
  for (const model of models) {
    const shared = model.shared ? "starter weights" : "your checkpoint";
    appendTextItem(modelList, `${model.name} · ${shared}`);
  }
  if (!models.length) appendTextItem(modelList, "No models yet.");
  for (const prediction of predictions) {
    const value = prediction.result && prediction.result.prediction
      ? JSON.stringify(prediction.result.prediction)
      : "";
    appendTextItem(predictionList, `${prediction.starter_id || prediction.model_id} ${value}`);
  }
  if (!predictions.length) appendTextItem(predictionList, "No saved predictions yet.");
}

async function refreshWorkspace() {
  const [starters, datasets, models, predictions] = await Promise.all([
    fetchJson("/workspace/starters"),
    fetchJson("/workspace/datasets"),
    fetchJson("/workspace/models"),
    fetchJson("/workspace/predictions"),
  ]);
  renderWorkspace(starters);
  renderWorkspaceLists(
    datasets.datasets || [],
    models.models || [],
    predictions.predictions || []
  );
}

async function refreshAccountAndWorkspace() {
  try {
    const me = await fetchJson("/auth/me");
    showAccount(me);
    const [training, experiments] = await Promise.all([
      fetchJson("/training/status"),
      fetchJson("/experiments?limit=50"),
    ]);
    state.training = training;
    state.experiments = experiments.experiments || [];
    try {
      await refreshWorkspace();
    } catch (workspaceError) {
      if (workspaceError.status === 401) throw workspaceError;
      renderWorkspace({ starters: [] });
      renderWorkspaceLists([], [], []);
    }
  } catch (error) {
    if (error.status === 401) {
      showAccount(null);
      state.training = { status: "idle", result: null };
      state.experiments = [];
      renderWorkspace({ starters: [] });
      renderWorkspaceLists([], [], []);
      return;
    }
    throw error;
  }
}

async function refreshStatus() {
  const [health, modelInfo, youneek] = await Promise.all([
    fetchJson("/health"),
    fetchJson("/model/info"),
    fetchJson("/youneek/now"),
  ]);
  state.health = health;
  state.modelInfo = modelInfo;
  await refreshAccountAndWorkspace();
  const paths = await openapiPaths();
  state.comparisonRows = await fetchComparison(paths);
  renderStatus();
  renderYouneek(youneek);
  if (!state.experiments.some((experiment) => experiment.id === state.selectedId)) {
    state.selectedId = null;
    state.selectedStamp = null;
  }
  if (state.selectedId == null && state.experiments.length) {
    const completed = state.experiments.find((experiment) => experiment.status === "completed");
    state.selectedId = (completed || state.experiments[0]).id;
  }
  renderExperiments();
  if (state.series && modelInfo.forecast) {
    state.steps = preferRicher(state.steps, normalizeForecast(modelInfo.forecast));
    renderForecast();
  }
  renderAccuracy();
  const selected = state.experiments.find((experiment) => experiment.id === state.selectedId);
  if (selected && selected.updated_at !== state.selectedStamp) {
    state.selectedStamp = selected.updated_at;
    loadExperimentDetail(selected.id);
  }
  const training = state.training || { status: "idle", result: null };
  if (!holdTrainingMessage) {
    document.getElementById("training-output").textContent = JSON.stringify(training, null, 2);
  }
  if (lastTrainingStatus === "running" && training.status === "completed") {
    refreshSeries().catch(reportError);
  }
  lastTrainingStatus = training.status;
}

function renderAdhoc(result) {
  const steps = normalizeForecast(result);
  const host = document.getElementById("adhoc-chart");
  if (steps.length > 1 || hasInterval(steps)) {
    const labels = steps.map((step, index) => `+${step.step || index + 1}`);
    const low = steps.map((step) => step.p10);
    const high = steps.map((step) => step.p90);
    charts.renderLineChart(host, {
      ariaLabel: "Ad-hoc forecast",
      labels,
      series: [
        {
          name: "Forecast",
          color: "#7cffd6",
          markers: true,
          values: steps.map((step) => (step.p50 != null ? step.p50 : step.value)),
        },
      ],
      bands: hasInterval(steps) ? [{ name: "p10–p90", color: "#7cffd6", low, high }] : [],
    });
    return;
  }
  host.replaceChildren();
}

function scorecardCell(text) {
  const cell = document.createElement("td");
  cell.textContent = text;
  return cell;
}

function formatChance(value) {
  const number = asNumber(value);
  return number == null ? "—" : number.toFixed(2);
}

function formatP(value) {
  const number = asNumber(value);
  if (number == null) return "—";
  if (number < 0.001) return number.toExponential(2);
  return number.toFixed(3);
}

function formatEffect(test) {
  const number = asNumber(test.effect);
  if (number == null) return "—";
  const name = test.effect_name === "cramers_v" ? "V" : "η²";
  return `${name} ${number.toFixed(4)}`;
}

async function refreshLunar() {
  const summary = document.getElementById("lunar-summary");
  const today = document.getElementById("lunar-today");
  const body = document.getElementById("lunar-body");
  const models = document.getElementById("lunar-models");
  try {
    const moon = await fetchJson("/lunar");
    const eclipse = moon.eclipse ? moon.eclipse.kind : "none";
    fillStatGrid(today, [
      ["Phase", moon.named_phase],
      ["Illumination", `${moon.illumination_percent}%`],
      ["Days since new", moon.days_since_new],
      ["Chinese date", moon.chinese_date],
      ["Full moon", moon.is_full_moon ? "yes" : "no"],
      ["New moon", moon.is_new_moon ? "yes" : "no"],
      ["Supermoon", moon.is_supermoon ? "yes" : "no"],
      ["Near perigee", moon.near_perigee ? "yes" : "no"],
      ["Eclipse", eclipse],
      ["Distance km", moon.distance_km],
    ]);
  } catch (error) {
    fillStatGrid(today, [["Moon", "unavailable"]]);
    summary.textContent = error.message;
  }
  try {
    const report = await fetchJson("/lottery/lunar-analysis");
    summary.textContent = report.summary || "No lunar lottery analysis has been published.";
    body.replaceChildren();
    const games = report.games || [];
    for (const game of games) {
      for (const test of game.tests || []) {
        const row = document.createElement("tr");
        for (const value of [
          game.game,
          test.label,
          formatEffect(test),
          formatP(test.p_permutation),
          formatP(test.p_bh),
          test.survives_bh ? "yes" : "no",
        ]) {
          row.append(scorecardCell(value));
        }
        body.append(row);
      }
    }
    const notes = [];
    if (report.answer_in_brief) notes.push(report.answer_in_brief);
    if (report.caveat) notes.push(report.caveat);
    models.textContent = notes.join(" ");
  } catch {
    summary.textContent = "Lunar analysis is unavailable.";
    body.replaceChildren();
    models.textContent = "";
  }
}

async function refreshScorecard() {
  const summary = document.getElementById("powerball-summary");
  const body = document.getElementById("powerball-body");
  try {
    const card = await fetchJson("/powerball/scorecard");
    summary.textContent = card.summary || "No Powerball scorecard has been published.";
    body.replaceChildren();
    if (!card.available) return;
    const cumulative = card.cumulative || {};
    for (const name of Object.keys(cumulative)) {
      const item = cumulative[name] || {};
      const row = document.createElement("tr");
      for (const value of [
        name,
        String(item.scored ?? 0),
        String(item.white_matches ?? 0),
        formatChance(item.expected_white_matches),
        String(item.powerball_hits ?? 0),
        formatChance(item.expected_powerball_hits),
        String(item.prize_tickets ?? 0),
      ]) {
        row.append(scorecardCell(value));
      }
      body.append(row);
    }
  } catch {
    summary.textContent = "Powerball scorecard is unavailable.";
    body.replaceChildren();
  }
}

const NEAR_COLUMNS = [
  ["date", "Date"],
  ["name", "Name"],
  ["body", "Body"],
  ["dist_ld", "Distance (LD)"],
  ["h", "H"],
  ["kind", "Kind"],
];

let nearEarthPayload = null;
let nearSort = { key: "date", direction: 1 };

function nearDays() {
  const value = Number(document.getElementById("near-earth-days").value);
  return Number.isFinite(value) ? value : 730;
}

function formatLd(value) {
  const number = asNumber(value);
  return number == null ? "—" : number.toFixed(2);
}

function formatMaybe(value, digits) {
  const number = asNumber(value);
  return number == null ? "—" : number.toFixed(digits);
}

function nearPassesFilter(row) {
  const body = document.getElementById("near-earth-body").value;
  return body === "all" || row.body === body;
}

function compareNear(left, right) {
  const key = nearSort.key;
  const direction = nearSort.direction;
  const a = left[key];
  const b = right[key];
  const aNumber = asNumber(a);
  const bNumber = asNumber(b);
  if (aNumber != null || bNumber != null) {
    if (aNumber == null) return 1;
    if (bNumber == null) return -1;
    return (aNumber - bNumber) * direction;
  }
  return String(a == null ? "" : a).localeCompare(String(b == null ? "" : b)) * direction;
}

function renderNearTable() {
  const head = document.getElementById("near-earth-head");
  const body = document.getElementById("near-earth-body-rows");
  head.replaceChildren();
  body.replaceChildren();
  for (const [key, label] of NEAR_COLUMNS) {
    const cell = document.createElement("th");
    cell.setAttribute("scope", "col");
    if (nearSort.key === key) {
      cell.setAttribute("aria-sort", nearSort.direction > 0 ? "ascending" : "descending");
    }
    const button = document.createElement("button");
    button.type = "button";
    button.className = "header-sort";
    button.textContent = label;
    button.addEventListener("click", () => {
      nearSort = {
        key,
        direction: nearSort.key === key ? -nearSort.direction : 1,
      };
      renderNearTable();
    });
    cell.append(button);
    head.append(cell);
  }
  const rows = ((nearEarthPayload && nearEarthPayload.approaches) || [])
    .filter(nearPassesFilter)
    .slice()
    .sort(compareNear);
  for (const row of rows) {
    const tr = document.createElement("tr");
    for (const [key] of NEAR_COLUMNS) {
      const cell = document.createElement("td");
      if (key === "dist_ld") cell.textContent = formatLd(row.dist_ld);
      else if (key === "h") cell.textContent = formatMaybe(row.h, 1);
      else cell.textContent = row[key] == null ? "—" : String(row[key]);
      tr.append(cell);
    }
    body.append(tr);
  }
}

function renderNearChart() {
  const host = document.getElementById("near-earth-chart");
  const select = document.getElementById("near-earth-series");
  const forecasts = (nearEarthPayload && nearEarthPayload.forecasts) || [];
  const selected = select.value;
  const forecast = forecasts.find((item) => item.id === selected) || forecasts[0];
  if (!forecast || !forecast.outlook || !Array.isArray(forecast.outlook.months)) {
    charts.renderLineChart(host, {
      ariaLabel: "Near-Earth monthly forecast",
      empty: "No monthly forecast has been saved yet.",
      labels: [],
      series: [],
    });
    return;
  }
  const months = forecast.outlook.months;
  const labels = months.map((month) => String(month.date).slice(0, 7));
  const jplKey = forecast.jpl_monthly_key;
  const jplByMonth = new Map(
    ((nearEarthPayload && nearEarthPayload.monthly) || []).map((month) => [
      month.month,
      month,
    ])
  );
  const jpl = months.map((month) => {
    const bucket = jplByMonth.get(month.date);
    if (!bucket || !jplKey) return null;
    return asNumber(bucket[jplKey]);
  });
  const series = [
    {
      name: "Model p50",
      color: "#7cffd6",
      markers: true,
      values: months.map((month) => asNumber(month.p50)),
    },
  ];
  if (jpl.some((value) => value != null)) {
    series.push({
      name: "JPL in window",
      color: "#ffd166",
      markers: true,
      values: jpl,
    });
  }
  charts.renderLineChart(host, {
    ariaLabel: `${forecast.title || "Forecast"} over 24 months`,
    labels,
    series,
    bands: [
      {
        name: "p10–p90",
        color: "#7cffd6",
        low: months.map((month) => asNumber(month.p10)),
        high: months.map((month) => asNumber(month.p90)),
      },
    ],
  });
}

function renderNearContext() {
  const note = document.getElementById("near-earth-context");
  const payload = nearEarthPayload || {};
  const sentry = payload.sentry || {};
  const interstellar = payload.interstellar || {};
  const known = (interstellar.known || []).map((item) => item.id || item.name).filter(Boolean);
  const sentryCount = sentry.count == null ? "0" : String(sentry.count);
  const top = (sentry.objects || [])[0];
  const topText = top ? `${top.name || top.designation} (Palermo ${top.palermo})` : "none listed";
  const upcoming = (interstellar.upcoming || []).length;
  note.textContent =
    `Known interstellar objects: ${known.length ? known.join(", ") : "none resolved"}. ` +
    `${upcoming} interstellar approach${upcoming === 1 ? "" : "es"} in this window. ` +
    `Sentry currently lists ${sentryCount} impact-risk objects. Highest Palermo: ${topText}. ` +
    (payload.fireball_note || "");
}

function renderNearReentries() {
  const body = document.getElementById("near-earth-reentries");
  const note = document.getElementById("near-earth-reentry-note");
  const payload = nearEarthPayload || {};
  note.textContent = payload.reentries_note || "";
  body.replaceChildren();
  const rows = payload.reentries || [];
  if (!rows.length) {
    const tr = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 3;
    cell.textContent = "No decay date in this window falls in the future.";
    tr.append(cell);
    body.append(tr);
    return;
  }
  for (const row of rows) {
    const tr = document.createElement("tr");
    for (const value of [row.name, row.norad_id, row.date]) {
      const cell = document.createElement("td");
      cell.textContent = value == null ? "—" : String(value);
      tr.append(cell);
    }
    body.append(tr);
  }
}

function renderNearEarth() {
  const payload = nearEarthPayload || {};
  const summary = document.getElementById("near-earth-summary");
  const note = document.getElementById("near-earth-note");
  const approaches = (payload.approaches || []).filter(nearPassesFilter);
  const cap = payload.truncated
    ? " The list keeps the closest approaches and drops the rest."
    : "";
  summary.textContent = payload.available
    ? `${approaches.length} approaches shown for ${payload.window_start} through ${payload.window_end}.${cap}`
    : "No cached close-approach file yet. Run python -m scripts.fetch_near_earth.";
  note.textContent = payload.survey_growth_note || "";
  const closest = approaches
    .slice()
    .sort((left, right) => Number(left.dist_ld) - Number(right.dist_ld))[0];
  fillStatGrid(document.getElementById("near-earth-stats"), [
    ["Window", payload.window_end ? `${payload.days} days` : "—"],
    ["Shown", String(approaches.length)],
    [
      "Closest",
      closest ? `${closest.name} · ${formatLd(closest.dist_ld)} LD` : "—",
    ],
    [
      "Forecasts",
      String((payload.forecasts || []).length),
    ],
  ]);
  const select = document.getElementById("near-earth-series");
  const current = select.value;
  select.replaceChildren();
  for (const forecast of payload.forecasts || []) {
    const option = document.createElement("option");
    option.value = String(forecast.id);
    option.textContent = String(forecast.title || forecast.id);
    select.append(option);
  }
  if (current && [...select.options].some((option) => option.value === current)) {
    select.value = current;
  }
  renderNearChart();
  renderNearTable();
  renderNearReentries();
  renderNearContext();
}

async function refreshNearEarth() {
  const summary = document.getElementById("near-earth-summary");
  try {
    nearEarthPayload = await fetchJson(`/space/near-earth?days=${nearDays()}`);
    renderNearEarth();
  } catch (error) {
    summary.textContent = "Close-approach feed is unavailable.";
    document.getElementById("near-earth-note").textContent = error.message;
  }
}

document.getElementById("near-earth-days").addEventListener("change", () => {
  refreshNearEarth().catch(reportError);
});
document.getElementById("near-earth-body").addEventListener("change", renderNearEarth);
document.getElementById("near-earth-series").addEventListener("change", renderNearChart);

const SOLAR_COLUMNS = [
  ["date", "Date"],
  ["name", "Name"],
  ["body", "Body"],
  ["dist_au", "Distance (AU)"],
  ["dist_ld", "Distance (LD)"],
];

let solarPayload = null;
let solarSort = { key: "date", direction: 1 };

function solarDays() {
  const value = Number(document.getElementById("solar-system-days").value);
  return Number.isFinite(value) ? value : 730;
}

function solarBody() {
  return document.getElementById("solar-system-body").value;
}

function formatAu(value) {
  const number = asNumber(value);
  return number == null ? "—" : number.toFixed(4);
}

function elongationSide(value) {
  if (value === "L") return "leading";
  if (value === "T") return "trailing";
  return value == null || value === "" ? "—" : String(value);
}

function solarPassesBody(row) {
  const body = solarBody();
  return body === "all" || row.body === body;
}

function compareSolar(left, right) {
  const key = solarSort.key;
  const direction = solarSort.direction;
  const aNumber = asNumber(left[key]);
  const bNumber = asNumber(right[key]);
  if (aNumber != null || bNumber != null) {
    if (aNumber == null) return 1;
    if (bNumber == null) return -1;
    return (aNumber - bNumber) * direction;
  }
  return String(left[key] == null ? "" : left[key]).localeCompare(
    String(right[key] == null ? "" : right[key])
  ) * direction;
}

function renderSolarTable() {
  const head = document.getElementById("solar-approaches-head");
  const body = document.getElementById("solar-approaches-rows");
  head.replaceChildren();
  body.replaceChildren();
  for (const [key, label] of SOLAR_COLUMNS) {
    const cell = document.createElement("th");
    cell.setAttribute("scope", "col");
    if (solarSort.key === key) {
      cell.setAttribute("aria-sort", solarSort.direction > 0 ? "ascending" : "descending");
    }
    const button = document.createElement("button");
    button.type = "button";
    button.className = "header-sort";
    button.textContent = label;
    button.addEventListener("click", () => {
      solarSort = {
        key,
        direction: solarSort.key === key ? -solarSort.direction : 1,
      };
      renderSolarTable();
    });
    cell.append(button);
    head.append(cell);
  }
  const rows = ((solarPayload && solarPayload.approaches) || [])
    .filter(solarPassesBody)
    .slice()
    .sort(compareSolar);
  if (!rows.length) {
    const tr = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = SOLAR_COLUMNS.length;
    const selected = (solarPayload && solarPayload.bodies || []).find(
      (item) => item.name === solarBody()
    );
    cell.textContent = selected
      ? `No approaches within ${selected.dist_max_au} AU in this window.`
      : "No close approaches in this window.";
    tr.append(cell);
    body.append(tr);
    return;
  }
  for (const row of rows) {
    const tr = document.createElement("tr");
    for (const [key] of SOLAR_COLUMNS) {
      const cell = document.createElement("td");
      if (key === "dist_au") cell.textContent = formatAu(row.dist_au);
      else if (key === "dist_ld") cell.textContent = formatLd(row.dist_ld);
      else if (key === "date") cell.textContent = String(row.date || "—").replace("T", " ");
      else cell.textContent = row[key] == null ? "—" : String(row[key]);
      tr.append(cell);
    }
    body.append(tr);
  }
}

function appendSolarRows(target, columns, rows, emptyText) {
  target.replaceChildren();
  if (!rows.length) {
    const tr = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = columns;
    cell.textContent = emptyText;
    tr.append(cell);
    target.append(tr);
    return;
  }
  for (const values of rows) {
    const tr = document.createElement("tr");
    for (const value of values) {
      const cell = document.createElement("td");
      cell.textContent = value == null || value === "" ? "—" : String(value);
      tr.append(cell);
    }
    target.append(tr);
  }
}

function renderSolarLists() {
  const payload = solarPayload || {};
  const perihelia = payload.perihelia || [];
  appendSolarRows(
    document.getElementById("solar-perihelia"),
    4,
    perihelia.map((row) => [
      row.date,
      row.name,
      row.designation,
      formatMaybe(row.period_days, 0),
    ]),
    "No comet perihelion in this window."
  );
  const alignments = (payload.alignments || []).filter(solarPassesBody);
  appendSolarRows(
    document.getElementById("solar-alignments"),
    5,
    alignments.map((row) => [
      row.date,
      row.body,
      row.kind,
      formatMaybe(row.elongation_deg, 1),
      elongationSide(row.side),
    ]),
    "No opposition, conjunction, or greatest elongation in this window."
  );
  const moons = payload.moons || {};
  const counts = (moons.by_planet || [])
    .map((row) => `${row.planet} ${row.count}`)
    .join(", ");
  const moonNote = document.getElementById("solar-moons");
  moonNote.textContent = [moons.note, counts].filter(Boolean).join(" ");
  appendSolarRows(
    document.getElementById("solar-moon-rows"),
    4,
    (moons.recent || []).map((row) => [row.year, row.planet, row.name, row.designation]),
    moons.available
      ? "No satellite in this table was discovered in the last ten years."
      : "Moon discovery table is not cached yet."
  );
  const showers = payload.showers || {};
  document.getElementById("solar-showers").textContent =
    showers.note || "Meteor shower calendar is unavailable.";
}

function renderSolarChart() {
  const host = document.getElementById("solar-system-chart");
  const select = document.getElementById("solar-system-series");
  const forecasts = (solarPayload && solarPayload.forecasts) || [];
  const forecast = forecasts.find((item) => item.id === select.value) || forecasts[0];
  if (!forecast || !forecast.outlook || !Array.isArray(forecast.outlook.months)) {
    charts.renderLineChart(host, {
      ariaLabel: "Solar-system monthly forecast",
      empty: "No monthly forecast has been saved yet.",
      labels: [],
      series: [],
    });
    return;
  }
  const months = forecast.outlook.months;
  const labels = months.map((month) => String(month.date).slice(0, 7));
  const jplKey = forecast.jpl_monthly_key;
  const jplByMonth = new Map(
    ((solarPayload && solarPayload.monthly) || []).map((month) => [month.month, month])
  );
  const jpl = months.map((month) => {
    const bucket = jplByMonth.get(month.date);
    if (!bucket || !jplKey) return null;
    return asNumber(bucket[jplKey]);
  });
  const series = [
    {
      name: "Model p50",
      color: "#7cffd6",
      markers: true,
      values: months.map((month) => asNumber(month.p50)),
    },
  ];
  if (jpl.some((value) => value != null)) {
    series.push({
      name: "Catalog in window",
      color: "#ffd166",
      markers: true,
      values: jpl,
    });
  }
  if (months.some((month) => asNumber(month.physics) != null)) {
    series.push({
      name: "JPL Horizons",
      color: "#7eb6ff",
      markers: true,
      values: months.map((month) => asNumber(month.physics)),
    });
  }
  charts.renderLineChart(host, {
    ariaLabel: `${forecast.title || "Forecast"} over 24 months`,
    labels,
    series,
    bands: [
      {
        name: "p10–p90",
        color: "#7cffd6",
        low: months.map((month) => asNumber(month.p10)),
        high: months.map((month) => asNumber(month.p90)),
      },
    ],
  });
}

function renderSolarSystem() {
  const payload = solarPayload || {};
  const summary = document.getElementById("solar-system-summary");
  const note = document.getElementById("solar-system-note");
  const approaches = (payload.approaches || []).filter(solarPassesBody);
  const perihelia = payload.perihelia || [];
  const alignments = (payload.alignments || []).filter(solarPassesBody);
  summary.textContent = payload.available
    ? `${approaches.length} approaches, ${perihelia.length} perihelia, and ${alignments.length} alignments from ${payload.window_start} through ${payload.window_end}.`
    : "No cached solar-system file yet. Run python -m scripts.fetch_solar_system.";
  note.textContent = payload.survey_growth_note || payload.time_scale || "";
  const closest = approaches
    .slice()
    .sort((left, right) => Number(left.dist_au) - Number(right.dist_au))[0];
  fillStatGrid(document.getElementById("solar-system-stats"), [
    ["Window", payload.window_end ? `${payload.days} days` : "—"],
    ["Approaches", String(approaches.length)],
    ["Perihelia", String(perihelia.length)],
    [
      "Closest",
      closest ? `${closest.name} · ${formatAu(closest.dist_au)} AU` : "—",
    ],
  ]);
  const select = document.getElementById("solar-system-series");
  const current = select.value;
  select.replaceChildren();
  for (const forecast of payload.forecasts || []) {
    const option = document.createElement("option");
    option.value = String(forecast.id);
    const physics = forecast.physics_outlook_mae;
    const suffix = physics == null ? "" : ` (Horizons MAE ${Number(physics).toFixed(3)})`;
    option.textContent = `${forecast.title || forecast.id}${suffix}`;
    select.append(option);
  }
  if (current && [...select.options].some((option) => option.value === current)) {
    select.value = current;
  }
  renderSolarChart();
  renderSolarTable();
  renderSolarLists();
}

async function refreshSolar() {
  const summary = document.getElementById("solar-system-summary");
  try {
    solarPayload = await fetchJson(`/space/solar-system?days=${solarDays()}`);
    renderSolarSystem();
  } catch (error) {
    summary.textContent = "Solar-system feed is unavailable.";
    document.getElementById("solar-system-note").textContent = error.message;
  }
}

document.getElementById("solar-system-days").addEventListener("change", () => {
  refreshSolar().catch(reportError);
});
document.getElementById("solar-system-body").addEventListener("change", renderSolarSystem);
document.getElementById("solar-system-series").addEventListener("change", renderSolarChart);

document.getElementById("refresh-btn").addEventListener("click", () => {
  holdTrainingMessage = false;
  clearNotice();
  refreshLunar();
  refreshScorecard();
  refreshNearEarth().catch(reportError);
  refreshSolar().catch(reportError);
  Promise.all([refreshStatus(), refreshSeries()]).catch(reportError);
});

document.getElementById("predict-btn").addEventListener("click", async () => {
  const output = document.getElementById("prediction-output");
  try {
    const sequence = JSON.parse(document.getElementById("sequence-input").value);
    const result = await fetchJson("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sequence }),
    });
    output.textContent = JSON.stringify(result, null, 2);
    renderAdhoc(result);
    clearNotice();
  } catch (error) {
    output.textContent = error.message;
    document.getElementById("adhoc-chart").replaceChildren();
  }
});

const tokenInput = document.getElementById("api-token");
tokenInput.value = localStorage.getItem(TOKEN_KEY) || "";
tokenInput.addEventListener("change", () => {
  const value = tokenInput.value.trim();
  if (value) localStorage.setItem(TOKEN_KEY, value);
  else localStorage.removeItem(TOKEN_KEY);
});

function trainingHeaders() {
  const headers = { "Content-Type": "application/json" };
  const token = tokenInput.value.trim();
  if (token) {
    localStorage.setItem(TOKEN_KEY, token);
    headers.Authorization = `Bearer ${token}`;
  } else {
    localStorage.removeItem(TOKEN_KEY);
  }
  return headers;
}

document.getElementById("train-btn").addEventListener("click", async () => {
  const output = document.getElementById("training-output");
  try {
    const response = await fetch("/train", {
      method: "POST",
      credentials: "same-origin",
      headers: trainingHeaders(),
      body: JSON.stringify({ config_path: "config/config.yaml" }),
    });
    let data = {};
    try {
      data = await response.json();
    } catch {
      data = {};
    }
    if (response.status === 401) {
      holdTrainingMessage = true;
      output.textContent =
        "Unauthorized (401). Enter the API token configured on the server and try again.";
      return;
    }
    if (response.status === 403) {
      holdTrainingMessage = true;
      output.textContent =
        typeof data.detail === "string" ? data.detail : "Training is disabled on this server.";
      return;
    }
    if (!response.ok) {
      holdTrainingMessage = true;
      const detail = typeof data.detail === "string" ? data.detail : response.statusText;
      output.textContent = detail;
      return;
    }
    holdTrainingMessage = false;
    output.textContent = JSON.stringify(data, null, 2);
    setTimeout(() => {
      refreshStatus().catch(() => {});
    }, 1500);
  } catch (error) {
    output.textContent = error.message;
  }
});

document.getElementById("sign-in-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  try {
    await fetchJson("/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        email: form.email.value,
        password: form.password.value,
      }),
    });
    form.reset();
    clearNotice();
    await refreshStatus();
  } catch (error) {
    reportError(error);
  }
});

document.getElementById("sign-up-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  try {
    await fetchJson("/auth/signup", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        email: form.email.value,
        password: form.password.value,
      }),
    });
    form.reset();
    clearNotice();
    await refreshStatus();
  } catch (error) {
    reportError(error);
  }
});

document.getElementById("sign-out-btn").addEventListener("click", async () => {
  try {
    await fetchJson("/auth/logout", { method: "POST" });
    await refreshStatus();
  } catch (error) {
    reportError(error);
  }
});

document.getElementById("dataset-upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = document.getElementById("dataset-file");
  const file = input.files && input.files[0];
  if (!file) {
    reportError(new Error("Choose a CSV file first."));
    return;
  }
  try {
    const response = await fetch(
      `/workspace/datasets?name=${encodeURIComponent(file.name || "dataset.csv")}`,
      {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "text/csv" },
        body: file,
      }
    );
    const text = await response.text();
    const data = text ? JSON.parse(text) : {};
    if (!response.ok) {
      const error = new Error(errorMessage(response, data, text));
      error.status = response.status;
      throw error;
    }
    input.value = "";
    clearNotice();
    await refreshWorkspace();
  } catch (error) {
    reportError(error);
  }
});

document.getElementById("owner-sign-in-btn").addEventListener("click", async () => {
  try {
    const response = await fetch("/auth/owner", {
      method: "POST",
      credentials: "same-origin",
      headers: trainingHeaders(),
    });
    const text = await response.text();
    const data = text ? JSON.parse(text) : {};
    if (!response.ok) {
      throw new Error(errorMessage(response, data, text));
    }
    clearNotice();
    await refreshStatus();
  } catch (error) {
    reportError(error);
  }
});

refreshLunar();
refreshScorecard();
refreshNearEarth().catch(reportError);
refreshSolar().catch(reportError);
Promise.all([refreshStatus(), refreshSeries()]).catch((error) => {
  reportError(error);
  const grid = document.getElementById("status-grid");
  const message = document.createElement("p");
  message.textContent = error.message;
  grid.replaceChildren(message);
});

setInterval(() => {
  refreshStatus().catch(() => {});
}, 5000);
