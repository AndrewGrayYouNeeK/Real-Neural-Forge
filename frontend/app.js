/* Nous chat-first app. All API text is written with textContent only. */
const charts = window.NousCharts;
const MODE_KEY = "neuralForgeChatMode";
const TALK_PREF_KEY = "neuralForgeTalkPref";
const MAX_VOICE_CHARS = 2000;
const VOICE_PICK_KEY = "nousVoicePick";
const DEFAULT_VOICES = [
  { key: "gq", name: "GQ" },
  { key: "female", name: "Ivy" },
];

const state = {
  account: null,
  voiceStatus: null,
  checkoutSession: null,
  training: null,
  experiments: [],
  series: null,
  trainingWatch: null,
  csvTrainPending: false,
  csvTrainJobId: null,
  voiceResting: false,
  toldVoiceResting: false,
  lastPca: null,
  lastRegimes: null,
  lastDiffusion: null,
  lastJepa: null,
};

const VOICE_RESTING_MESSAGE = "Voice is resting for today, text still works.";

const SPEAK_LABEL = "Listen";
const SPEAK_BUSY_LABEL = "Loading GQ…";

/* ---------------------------------------------------------------- helpers */

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

// Sign-in lives in the HttpOnly session cookie only. Nothing secret is kept in
// localStorage, and no shared owner token exists any more.
function authHeaders(extra = {}) {
  return { ...extra };
}

async function fetchJson(url, options = {}) {
  const headers = authHeaders(options.headers);
  const response = await fetch(url, {
    credentials: "same-origin",
    ...options,
    headers,
  });
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
    error.data = data;
    if (response.status === 401 && data && data.auth_required && typeof onSessionLost === "function") {
      onSessionLost();
    }
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
  const digits = abs !== 0 && abs < 0.001 ? 6 : 4;
  return number.toLocaleString(undefined, {
    maximumSignificantDigits: digits,
    notation: "standard",
  });
}

function reportError(error) {
  document.getElementById("notice").textContent =
    error && error.message ? error.message : "Request failed.";
}

function clearNotice() {
  document.getElementById("notice").textContent = "";
}

function numericArray(value) {
  if (!Array.isArray(value)) return null;
  const numbers = value.map(asNumber);
  return numbers.some((number) => number != null) ? numbers : null;
}

function stepsFromArrays(payload) {
  const forecast = numericArray(payload.forecast);
  const p10 = numericArray(payload.p10);
  const p50 = numericArray(payload.p50);
  const p90 = numericArray(payload.p90);
  if (forecast) return forecast.map((value, index) => ({
    step: index + 1,
    value,
    p10: p10 ? p10[index] : null,
    p50: p50 ? p50[index] : null,
    p90: p90 ? p90[index] : null,
  }));
  return [];
}

function stepsFromObjects(payload) {
  const list = Array.isArray(payload.forecast) ? payload.forecast : [];
  const steps = list
    .map((row, index) => {
      if (!row || typeof row !== "object" || Array.isArray(row)) return null;
      const value = asNumber(row.value ?? row.p50 ?? row.prediction);
      if (value == null) return null;
      return {
        step: asNumber(row.step) ?? index + 1,
        value,
        p10: asNumber(row.p10),
        p50: asNumber(row.p50),
        p90: asNumber(row.p90),
      };
    })
    .filter(Boolean);
  return steps.length ? steps : stepsFromArrays(payload);
}

function normalizeForecast(payload) {
  if (!payload || typeof payload !== "object") return [];
  return stepsFromObjects(payload);
}

function hasInterval(steps) {
  return steps.some((step) => step.p10 != null && step.p90 != null);
}

/* ------------------------------------------------------------------ chat */

function chatLog() {
  return document.getElementById("chat-log");
}

function scrollToBottom() {
  const log = chatLog();
  log.scrollTop = log.scrollHeight;
}

function makeButton(label, action) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "btn secondary btn-small";
  button.textContent = label;
  button.addEventListener("click", action);
  return button;
}

const MARKDOWN_LINK = /\[([^\]\n]{1,120})\]\((https?:\/\/[^\s)]+)\)/g;

/** Render text with [label](https://…) source links as safe anchors (no innerHTML). */
function appendLinkedText(host, text) {
  const value = String(text || "");
  let last = 0;
  for (const match of value.matchAll(MARKDOWN_LINK)) {
    if (match.index > last) host.append(value.slice(last, match.index));
    const anchor = document.createElement("a");
    anchor.href = match[2];
    anchor.textContent = match[1];
    anchor.target = "_blank";
    anchor.rel = "noopener noreferrer nofollow";
    anchor.className = "source-link";
    host.append(anchor);
    last = match.index + match[0].length;
  }
  if (last < value.length) host.append(value.slice(last));
}

function addMessage(role, text) {
  const message = document.createElement("div");
  message.className = `message ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if (role === "assistant") appendLinkedText(bubble, text);
  else bubble.textContent = text;
  message.append(bubble);

  if (role === "assistant") {
    message.dataset.speech = String(text || "").replace(MARKDOWN_LINK, "$1");
    const actions = document.createElement("div");
    actions.className = "message-actions";
    const speak = document.createElement("button");
    speak.type = "button";
    speak.className = "speak-btn";
    speak.textContent = `🔊 ${SPEAK_LABEL}`;
    speak.setAttribute("aria-label", "Listen to this reply in GQ voice");
    speak.addEventListener("click", () => speakMessage(message, speak));
    actions.append(speak);
    message.append(actions);
  }

  chatLog().append(message);
  scrollToBottom();
  return message;
}

function addReply(text, chartSpec) {
  const message = addMessage("assistant", text);
  if (chartSpec) {
    const host = document.createElement("div");
    host.className = "chart-host";
    message.querySelector(".bubble").append(host);
    charts.renderLineChart(host, chartSpec);
    scrollToBottom();
  }
  return message;
}

function showHint(message, text, buttons) {
  const actions = message.querySelector(".message-actions");
  if (!actions) return;
  const hint = document.createElement("p");
  hint.className = "inline-hint";
  hint.textContent = text;
  actions.append(hint);
  for (const [label, action] of buttons || []) {
    actions.append(makeButton(label, action));
  }
  scrollToBottom();
}

let currentAudio = null;
let speakToken = 0;
let activeSpeakButton = null;

function resetSpeakButton(button) {
  if (!button) return;
  button.disabled = false;
  button.dataset.state = "idle";
  button.textContent = `🔊 ${SPEAK_LABEL}`;
}

function stopCurrentAudio() {
  const audio = currentAudio;
  currentAudio = null;
  if (!audio) return;
  audio.pause();
  if (audio.src) URL.revokeObjectURL(audio.src);
}

function showSpeakStop(button, token) {
  if (activeSpeakButton && activeSpeakButton !== button) {
    resetSpeakButton(activeSpeakButton);
  }
  activeSpeakButton = button;
  button.disabled = false;
  button.dataset.state = "playing";
  button.dataset.token = String(token);
  button.textContent = "Stop";
}

function rememberVoiceBudget(result) {
  if (result && result.voice_resting) state.voiceResting = true;
}

async function voicePlaybackAllowed() {
  // Subscribers pay for their own voice, so the free-tier rest day never blocks them.
  if (state.voiceStatus && state.voiceStatus.subscribed) return true;
  if (state.voiceResting) return false;
  try {
    rememberVoiceBudget(await fetchJson("/live/budget"));
  } catch {
    // A missed status check does not block a listen the chat has not marked resting.
  }
  return !state.voiceResting;
}

async function speakMessage(message, speakButton) {
  if (
    speakButton.dataset.state === "playing" &&
    Number(speakButton.dataset.token) === speakToken
  ) {
    speakToken += 1;
    stopCurrentAudio();
    resetSpeakButton(speakButton);
    activeSpeakButton = null;
    return;
  }
  const text = (message.dataset.speech || "").trim();
  if (!text) {
    showHint(message, "There is nothing in this reply to read aloud.");
    return;
  }
  if (!(await voicePlaybackAllowed())) {
    showHint(message, VOICE_RESTING_MESSAGE);
    return;
  }
  const token = ++speakToken;
  stopCurrentAudio();
  showSpeakStop(speakButton, token);
  let playing = false;
  try {
    const response = await fetch("/voice/speak", {
      method: "POST",
      credentials: "same-origin",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({ text: text.slice(0, MAX_VOICE_CHARS), voice: selectedVoice() }),
    });
    if (token !== speakToken) return;
    if (response.status === 401) {
      showHint(
        message,
        "GQ voice needs a free account. Reading the chat in text stays free.",
        [["Sign in", openSignInModal]]
      );
      return;
    }
    if (response.status === 402) {
      let detail = "You are out of free GQ voice. Nous Plus includes voice. Text stays free.";
      try {
        const data = await response.json();
        if (typeof data.detail === "string" && data.detail) detail = data.detail;
      } catch {
        // Keep the plain message when the body is not JSON.
      }
      showHint(message, detail, [["Subscribe", startSubscribe]]);
      return;
    }
    if (!response.ok) {
      let detail = `GQ voice is unavailable (${response.status}). Text stays free.`;
      try {
        const data = await response.json();
        if (typeof data.detail === "string" && data.detail) detail = data.detail;
      } catch {
        // Keep the generic message when the body is not JSON.
      }
      showHint(message, detail);
      return;
    }
    const audio = new Audio(URL.createObjectURL(await response.blob()));
    if (token !== speakToken) {
      URL.revokeObjectURL(audio.src);
      return;
    }
    currentAudio = audio;
    playing = true;
    audio.addEventListener("ended", () => {
      if (currentAudio !== audio) return;
      URL.revokeObjectURL(audio.src);
      currentAudio = null;
      if (activeSpeakButton === speakButton) {
        resetSpeakButton(speakButton);
        activeSpeakButton = null;
      }
    });
    audio.play().catch(() => {
      if (token !== speakToken) return;
      showHint(message, "Your browser blocked audio playback.");
    });
    refreshVoiceStatus().catch(() => {});
  } catch (error) {
    if (token !== speakToken) return;
    showHint(message, error.message || "GQ voice failed. Text stays free.");
  } finally {
    if (!playing && token === speakToken) {
      resetSpeakButton(speakButton);
      if (activeSpeakButton === speakButton) activeSpeakButton = null;
    }
  }
}

/* --------------------------------------------------------- topic replies */

function chartValue(starterId, value) {
  const number = asNumber(value);
  if (number == null) return null;
  return starterId === "weather" ? celsiusToFahrenheit(number) : number;
}

function forecastChart(series, steps, starterId) {
  const history = Array.isArray(series.history) ? series.history : [];
  const labels = history.map((point) => point.time || String(point.index));
  const actual = history.map((point) => chartValue(starterId, point.actual));
  const forecast = new Array(history.length).fill(null);
  const low = new Array(history.length).fill(null);
  const high = new Array(history.length).fill(null);
  const lastActual = actual.length ? actual[actual.length - 1] : null;
  if (lastActual != null) forecast[history.length - 1] = lastActual;
  steps.forEach((step, index) => {
    labels.push(`+${step.step || index + 1}`);
    actual.push(null);
    forecast.push(chartValue(starterId, step.p50 != null ? step.p50 : step.value));
    low.push(chartValue(starterId, step.p10));
    high.push(chartValue(starterId, step.p90));
  });
  const lines = [{ name: "Actual", color: "#e8eaed", values: actual }];
  if (forecast.some((value) => value != null)) {
    lines.push({
      name: "Forecast",
      color: "#a8b4c8",
      markers: true,
      values: forecast,
    });
  }
  return {
    ariaLabel: "Recent history with the model's forecast",
    labels,
    series: lines,
    bands: hasInterval(steps)
      ? [{ name: "p10–p90", color: "#a8b4c8", low, high }]
      : [],
    empty: "Not enough history to draw a forecast.",
  };
}

function stepText(steps, starterId) {
  if (!steps.length) return null;
  const format = (raw) => formatStarterValue(starterId || "", raw) || formatMetric(raw);
  const parts = steps.slice(0, 5).map((step, index) => {
    const value = format(step.p50 != null ? step.p50 : step.value);
    const band =
      step.p10 != null && step.p90 != null
        ? ` (likely between ${format(step.p10)} and ${format(step.p90)})`
        : "";
    const label = steps.length > 1 ? `Step ${step.step || index + 1}` : "Next value";
    return `${label}: ${value}${band}.`;
  });
  if (steps.length > 5) {
    parts.push(`…and ${steps.length - 5} more steps after that.`);
  }
  return parts.join(" ");
}

const FORECAST_STARTERS = {
  weather: "weather",
  markets: "stocks",
  crypto: "crypto",
};

function celsiusToFahrenheit(celsius) {
  return celsius * (9 / 5) + 32;
}

function formatStarterValue(kind, value) {
  const number = asNumber(value);
  if (number == null) return null;
  // Weather samples are stored in °C; user-facing replies always label °F.
  if (kind === "weather") return `${formatMetric(celsiusToFahrenheit(number))}°F`;
  if (kind === "lunar") return formatMetric(number);
  if (kind === "near-earth") return formatMetric(number);
  return formatMetric(number);
}

function displayUnitForStarter(starterId, rawUnit) {
  if (starterId === "weather") return "°F";
  return typeof rawUnit === "string" && rawUnit ? rawUnit : "";
}

function formatWeatherMaeFahrenheit(maeCelsius) {
  const mae = asNumber(maeCelsius);
  if (mae == null) return null;
  // Temperature MAE scales by 9/5 (no offset).
  return formatMetric(mae * (9 / 5));
}

function forecastPlainLines(kind, series, steps) {
  const name = series.name || series.dataset || "this sample series";
  const next = steps[0];
  const value = next
    ? formatStarterValue(kind, next.p50 != null ? next.p50 : next.value)
    : null;
  const history = Array.isArray(series.history) ? series.history : [];
  const last = history.length ? history[history.length - 1] : null;
  const lastValue =
    last && last.actual != null ? formatStarterValue(kind, last.actual) : null;
  const lastTime = last && last.time ? String(last.time) : null;
  const lines = [];
  if (kind === "weather") {
    lines.push(
      "I do not have a live forecast for your location. I can show the sample model for daily high temperatures at Central Park."
    );
    if (value) {
      lines.push(`On the next day in that sample, it expects a high of about ${value}.`);
    } else {
      lines.push("The sample is too short right now for a next-day read.");
    }
  } else if (kind === "markets" || kind === "crypto") {
    lines.push(
      `This is not live market advice. It is the sample model for ${name}.`
    );
    if (value) {
      lines.push(`It expects the next day's log return to be about ${value}.`);
    } else {
      lines.push("The sample is too short right now for a next-step read.");
    }
  } else {
    lines.push(`Here is what the sample model expects next on ${name}.`);
    const nextText = stepText(steps, kind);
    if (nextText) lines.push(nextText);
    else lines.push("The sample is too short right now for a forecast.");
  }
  if (lastValue && lastTime) {
    lines.push(`The last point in the sample is ${lastValue} on ${lastTime}.`);
  }
  if (series.checkpoint_loaded === false) {
    lines.push(
      "Heads up: no trained checkpoint is loaded at the moment, so treat this as a rough sketch."
    );
  }
  if (typeof series.summary === "string" && series.summary) lines.push(series.summary);
  const why = whyExplanationText(series);
  if (why) lines.push(why);
  return lines;
}

function whyExplanationText(payload) {
  if (!payload || typeof payload !== "object") return "";
  const why = payload.why;
  if (why && typeof why === "object" && typeof why.explanation === "string") {
    const text = why.explanation.trim();
    return text ? `Why: ${text}` : "";
  }
  if (typeof payload.explanation === "string" && payload.explanation.trim()) {
    return `Why: ${payload.explanation.trim()}`;
  }
  return "";
}

function liveUrl(path, question) {
  const join = path.includes("?") ? "&" : "?";
  return `${path}${join}q=${encodeURIComponent(question || "")}`;
}

async function optionalWeatherSampleLine() {
  try {
    const series = await fetchJson("/starters/weather/forecast?points=120");
    const steps = normalizeForecast(series);
    const next = steps[0];
    const celsius = asNumber(next && next.p50 != null ? next.p50 : next && next.value);
    if (celsius == null) return "";
    const fahrenheit = celsius * (9 / 5) + 32;
    return (
      `For comparison, the sample Central Park model expects a next-day high of about ${formatMetric(fahrenheit)}°F on that historical series. That is not the live forecast.`
    );
  } catch {
    return "";
  }
}

function starterIdFromQuestion(question) {
  const q = String(question || "").toLowerCase();
  if (/\b(bitcoin|btc|crypto)\b/.test(q)) return "crypto";
  if (/\b(aapl|stocks?)\b/.test(q)) return "stocks";
  if (/\b(near-?earth|approaches)\b/.test(q)) return "near-earth";
  if (/\b(moon|lunar|illumination)\b/.test(q)) return "lunar";
  return "weather";
}

function sampleModelLines(starterId, series, steps) {
  const name = series.name || series.dataset || "this sample series";
  const unitLabel = displayUnitForStarter(starterId, series.unit);
  const unit = unitLabel ? ` (${unitLabel})` : "";
  const next = steps[0];
  const value = next
    ? formatStarterValue(starterId, next.p50 != null ? next.p50 : next.value)
    : null;
  const history = Array.isArray(series.history) ? series.history : [];
  const last = history.length ? history[history.length - 1] : null;
  const lastValue =
    last && last.actual != null
      ? formatStarterValue(starterId, last.actual)
      : null;
  const lastTime = last && last.time ? String(last.time) : null;
  const lines = [
    `This is the shipped “${name}” sample model, not a live forecast. No account is required to try it.`,
  ];
  if (value) {
    // formatStarterValue already includes °F for weather — do not append raw °C.
    const suffix = starterId === "weather" ? "" : unit;
    lines.push(`It expects the next step to be about ${value}${suffix}.`);
  } else {
    lines.push("The sample is too short right now for a next-step read.");
  }
  if (lastValue && lastTime) {
    lines.push(`The last point in the sample is ${lastValue} on ${lastTime}.`);
  }
  if (typeof series.summary === "string" && series.summary) lines.push(series.summary);
  const why = whyExplanationText(series);
  if (why) lines.push(why);
  return lines;
}

function approachHighlightLine(highlights) {
  if (!Array.isArray(highlights) || !highlights.length) return "";
  const bits = highlights.slice(0, 5).map((row) => {
    const name = row.name || row.designation || "An object";
    const when = row.date ? String(row.date).slice(0, 10) : "an upcoming date";
    const dist = asNumber(row.dist_ld);
    const distance = dist == null ? "" : ` (${dist.toFixed(2)} lunar distances)`;
    return `${name} on ${when}${distance}`;
  });
  return `Closest catalogued approaches in this saved snapshot: ${bits.join("; ")}.`;
}

function uncertaintyText(uncertainty, starterId) {
  if (!uncertainty || typeof uncertainty !== "object") return "";
  const lows = Array.isArray(uncertainty.p10) ? uncertainty.p10 : [];
  const highs = Array.isArray(uncertainty.p90) ? uncertainty.p90 : [];
  const low = asNumber(lows[0]);
  const high = asNumber(highs[0]);
  const lines = [];
  if (low != null && high != null && low !== high) {
    if (starterId === "weather") {
      lines.push(
        `Likely range: ${formatMetric(celsiusToFahrenheit(low))}°F to ${formatMetric(celsiusToFahrenheit(high))}°F.`
      );
    } else {
      lines.push(`Likely range: ${formatMetric(low)} to ${formatMetric(high)}.`);
    }
  }
  const confidence = asNumber(uncertainty.confidence);
  // Omit uncalibrated / zero confidence — it reads as noise, not skill.
  if (confidence != null && uncertainty.calibrated === true && confidence > 0) {
    const percent = Math.round(confidence * 100);
    lines.push(`Confidence ${percent}% (calibrated).`);
  }
  return lines.join(" ");
}

function sampleReplyParts(starterId, series, steps) {
  const lines = sampleModelLines(starterId, series, steps);
  const extra = approachHighlightLine(series.highlights);
  if (extra) lines.push(extra);
  if (steps.length > 1) {
    const path = stepText(steps, starterId);
    if (path) lines.push(path);
  }
  const band = uncertaintyText(series.uncertainty, starterId);
  if (band) lines.push(band);
  const why = whyExplanationText(series);
  if (why) lines.push(why);
  return { text: lines.join("\n\n"), chart: forecastChart(series, steps, starterId) };
}

async function baselineComparison(modelId) {
  try {
    const report = await fetchJson(`/evaluation/${encodeURIComponent(modelId)}/backtest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    if (!report || typeof report !== "object") return "";
    const comparison =
      typeof report.comparison === "string" ? report.comparison.trim() : "";
    const mae = report.metrics && asNumber(report.metrics.mae);
    if (modelId === "weather") {
      const maeF = formatWeatherMaeFahrenheit(mae);
      const bits = [];
      if (comparison) bits.push(comparison);
      if (maeF != null) bits.push(`MAE about ${maeF}°F on the holdout.`);
      return bits.join(" ");
    }
    return comparison;
  } catch {
    return "";
  }
}

async function loadDefaultDemos() {
  // Warm starter forecasts in the background only — never auto-post sample
  // replies into the empty Grok-style chat.
  let ids = ["weather", "near-earth"];
  try {
    const catalog = await fetchJson("/starters");
    const marked = (catalog.starters || [])
      .filter((row) => row && row.demo)
      .map((row) => String(row.id));
    if (marked.length) ids = marked;
  } catch {
    // The two shipped demos still warm if the catalog request fails.
  }
  await Promise.all(
    ids.map(async (id) => {
      try {
        await fetchJson(`/starters/${id}/forecast?points=90`);
      } catch {
        // Sample chips still open the same model if warmup fails.
      }
    }),
  );
}

async function sampleModelReply(question) {
  const starterId = starterIdFromQuestion(question);
  try {
    const series = await fetchJson(`/starters/${starterId}/forecast?points=120`);
    const steps = normalizeForecast(series);
    const reply = sampleReplyParts(starterId, series, steps);
    const comparison = await baselineComparison(starterId);
    if (comparison) reply.text = `${reply.text}\n\n${comparison}`;
    return reply;
  } catch (error) {
    return {
      text:
        "I could not load that sample model right now. " +
        (error.message || "Please try again in a moment."),
    };
  }
}

async function forecastReply(kind, question) {
  const path =
    kind === "weather"
      ? "/live/weather"
      : kind === "crypto"
        ? "/live/quote?kind=crypto"
        : "/live/quote?kind=stock";
  try {
    const result = await fetchJson(liveUrl(path, question));
    const lines = [];
    if (result && typeof result.text === "string" && result.text.trim()) {
      lines.push(result.text.trim());
    } else {
      lines.push("I could not load that live answer right now.");
    }
    if (kind === "weather") {
      const extra = await optionalWeatherSampleLine();
      if (extra) lines.push(extra);
    }
    return { text: lines.join("\n\n") };
  } catch (error) {
    return {
      text:
        "I could not load that live answer right now. " +
        (error.message || "Please try again in a moment."),
    };
  }
}

async function moonReply() {
  try {
    const moon = await fetchJson("/lunar");
    const eclipse = moon.eclipse && moon.eclipse.kind ? moon.eclipse.kind : "none";
    const lines = [
      `Tonight's moon is a ${moon.named_phase} — about ${moon.illumination_percent}% lit.`,
      `It sits about ${formatMetric(moon.distance_km)} km away${
        moon.is_supermoon ? " (a supermoon window)" : ""
      }.`,
      `Chinese lunar date: ${moon.chinese_date}. Eclipse: ${eclipse}.`,
    ];
    return { text: lines.join("\n") };
  } catch (error) {
    return { text: `The lunar feed is unavailable right now. ${error.message}` };
  }
}

function closestApproachText(approaches, distanceKey, format) {
  const closest = approaches
    .filter((row) => asNumber(row[distanceKey]) != null)
    .sort((left, right) => Number(left[distanceKey]) - Number(right[distanceKey]))[0];
  if (!closest) return null;
  return `${closest.name || "An object"} passes about ${format(
    Number(closest[distanceKey])
  )} ${distanceKey === "dist_ld" ? "lunar distances" : "astronomical units"} away on ${
    closest.date || "an upcoming date"
  }.`;
}

async function spaceReply() {
  try {
    const payload = await fetchJson("/space/near-earth?days=90");
    if (!payload.available) {
      return {
        text:
          "The close-approach feed has not been cached yet on this server. " +
          "It fills in after a scheduled fetch runs.",
      };
    }
    const approaches = payload.approaches || [];
    const lines = [
      `Over the next ${payload.days || 90} days I see ${approaches.length} close approaches to Earth and the Moon.`,
    ];
    const closest = closestApproachText(approaches, "dist_ld", (value) =>
      value.toFixed(2)
    );
    if (closest) lines.push(`The closest: ${closest}`);
    const sentry = payload.sentry || {};
    if (sentry.count != null) {
      lines.push(
        `NASA's Sentry list currently tracks ${sentry.count} objects with any impact risk. ${
          sentry.count ? "None of these are near-term worries — the probabilities are tiny." : ""
        }`.trim()
      );
    }
    if (typeof payload.fireball_note === "string" && payload.fireball_note) {
      lines.push(payload.fireball_note);
    }
    return { text: lines.join("\n\n") };
  } catch (error) {
    return { text: `The near-Earth feed is unavailable right now. ${error.message}` };
  }
}

async function solarReply() {
  try {
    const payload = await fetchJson("/space/solar-system?days=365");
    if (!payload.available) {
      return {
        text:
          "The solar-system feed has not been cached yet on this server. " +
          "It fills in after a scheduled fetch runs.",
      };
    }
    const approaches = payload.approaches || [];
    const perihelia = payload.perihelia || [];
    const lines = [
      `In the next ${payload.days || 365} days: ${approaches.length} close approaches between planets, and ${perihelia.length} comet perihelia.`,
    ];
    const closest = closestApproachText(approaches, "dist_au", (value) =>
      value.toFixed(3)
    );
    if (closest) lines.push(`The closest: ${closest}`);
    const firstComet = perihelia[0];
    if (firstComet) {
      lines.push(`Next comet perihelion: ${firstComet.name || firstComet.designation} on ${firstComet.date}.`);
    }
    return { text: lines.join("\n\n") };
  } catch (error) {
    return { text: `The solar-system feed is unavailable right now. ${error.message}` };
  }
}

function trainingOutcomeText(status, message) {
  if (status === "running") {
    return "Training started in the background. It takes about a minute on this server — ask me again in a moment and I will check on it.";
  }
  if (status === "queued") {
    return "Your training run is queued. It will train in your own workspace, leaving the shared champion alone. Check Experiments in the Tools sidebar for progress.";
  }
  return message || "Training did not start.";
}

function isTrainStartRequest(question) {
  return /\b(start training|retrain|train(ing)? now|fit a model|train (a |the |my )?model)\b/i.test(
    question || ""
  );
}

function isCsvTrainRequest(question) {
  return /\b(train on my csv|upload (a |my )?csv|train on (a |this |my )?csv)\b/i.test(
    question || ""
  );
}

function formatBytesCap(bytes) {
  const number = asNumber(bytes);
  if (number == null) return "a small size limit";
  if (number >= 1_000_000) {
    const mb = Math.round((number / 1_000_000) * 10) / 10;
    return `${mb} MB`;
  }
  const kb = Math.max(1, Math.round(number / 1000));
  return `${kb} KB`;
}

function capLine(limits) {
  if (!limits) {
    return "Training runs are short and files are size-capped so this shared server stays on a small plan.";
  }
  return (
    `Runs are capped at ${limits.max_epochs} epochs and ${limits.max_run_seconds} seconds, ` +
    `and a CSV must stay under ${formatBytesCap(limits.max_dataset_bytes)}, ` +
    "so this shared server stays on a small plan."
  );
}

async function loadTrainLimits() {
  try {
    const data = await fetchJson("/workspace/limits");
    return data && data.limits ? data.limits : null;
  } catch {
    return null;
  }
}

function openSignUpModal() {
  openSignInModal();
  if (state.account) return;
  document.getElementById("sign-in-form").hidden = true;
  document.getElementById("sign-up-form").hidden = false;
  const field = document.getElementById("sign-up-email");
  if (field) field.focus();
}

function promptCsvPicker() {
  const input = document.getElementById("csv-train-file");
  if (!input) return;
  input.value = "";
  input.click();
}

function attachReplyActions(message, buttons) {
  const actions = message.querySelector(".message-actions");
  if (!actions) return;
  for (const [label, action] of buttons || []) {
    actions.append(makeButton(label, action));
  }
  scrollToBottom();
}

async function resumeCsvTrainIfPending() {
  if (!state.csvTrainPending || !state.account) return;
  state.csvTrainPending = false;
  closeSignInModal();
  const limits = await loadTrainLimits();
  const message = addReply(
    "You're in. Choose a CSV with a header row and at least 32 numeric values in the column you want to forecast. " +
      capLine(limits)
  );
  attachReplyActions(message, [["Choose a CSV", promptCsvPicker]]);
  promptCsvPicker();
}

async function csvTrainIntro() {
  const limits = await loadTrainLimits();
  const caps = capLine(limits);
  if (!state.account) {
    state.csvTrainPending = true;
    return {
      text:
        "I can train a small model on a CSV in your private workspace, then forecast what comes next. " +
        "Create a free account (or sign in), choose a file with a header row, pick the numeric column to learn, and I’ll queue a capped run. " +
        caps +
        " Nothing is started until you upload a file.",
      actions: [
        ["Create a free account", openSignUpModal],
        ["Sign in", openSignInModal],
      ],
    };
  }
  setTimeout(promptCsvPicker, 0);
  return {
    text:
      "Choose a CSV with a header row. I’ll list the numeric columns, you pick one, and I’ll train a small model in your private workspace. " +
      caps,
    actions: [["Choose a CSV", promptCsvPicker]],
  };
}

async function uploadCsvFile(file) {
  const response = await fetch(
    `/workspace/datasets?name=${encodeURIComponent(file.name || "dataset.csv")}`,
    {
      method: "POST",
      credentials: "same-origin",
      headers: authHeaders({ "Content-Type": "text/csv" }),
      body: file,
    }
  );
  const text = await response.text();
  let data = {};
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = {};
    }
  }
  if (!response.ok) {
    const error = new Error(errorMessage(response, data, text));
    error.status = response.status;
    throw error;
  }
  return data;
}

async function startDatasetTrain(datasetId, column) {
  const pending = addMessage("assistant", `Queueing a training run on “${column}”…`);
  try {
    const result = await fetchJson(`/workspace/datasets/${datasetId}/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ target_column: column }),
    });
    pending.remove();
    state.csvTrainJobId = result.id;
    addReply(
      `Training is queued in your private workspace on the “${result.target_column}” column. ` +
        `${capLine(result.limits)} I’ll post the first forecast here when the run finishes. ` +
        "Progress also shows under Experiments in the Tools sidebar."
    );
    watchTraining();
    refreshExperiments().catch(() => {});
    refreshWorkspace().catch(() => {});
  } catch (error) {
    pending.remove();
    addReply(`Training did not start. ${error.message}`);
  }
}

async function handleCsvTrainFile(file) {
  if (!state.account) {
    state.csvTrainPending = true;
    const message = addReply(
      "Training needs a free account first. Create one, then I’ll ask for the CSV again."
    );
    attachReplyActions(message, [
      ["Create a free account", openSignUpModal],
      ["Sign in", openSignInModal],
    ]);
    return;
  }
  addMessage("user", `Upload ${file.name || "dataset.csv"}`);
  const pending = addMessage("assistant", "Uploading your CSV…");
  try {
    const data = await uploadCsvFile(file);
    pending.remove();
    await refreshWorkspace().catch(() => {});
    const columns = (data.columns || []).filter((column) => column.usable);
    if (!columns.length) {
      addReply(
        "I need a CSV with a header row and at least 32 numeric values in one column. Empty files and non-numeric columns can’t be trained."
      );
      return;
    }
    if (data.pca) {
      state.lastPca = data.pca;
      renderPcaReport(data.pca);
    }
    if (columns.length === 1) {
      addReply(
        `I found a numeric column named “${columns[0].name}” (${columns[0].numeric_rows} rows). I’ll train a small capped model on it now.`
      );
      await startDatasetTrain(data.id, columns[0].name);
      return;
    }
    const suggested = data.suggested_target;
    const names = columns.map((column) => column.name).join(", ");
    const pcaLine =
      data.pca && data.pca.summary
        ? ` Column check (PCA): ${data.pca.summary}`
        : "";
    const message = addReply(
      `I found these numeric columns: ${names}.${pcaLine} Which should I train on?`
    );
    attachReplyActions(
      message,
      columns.map((column) => [
        column.name === suggested ? `${column.name} (suggested)` : column.name,
        () => startDatasetTrain(data.id, column.name),
      ])
    );
  } catch (error) {
    pending.remove();
    if (error.status === 401) {
      state.account = null;
      state.csvTrainPending = true;
      refreshAuthModal();
      const message = addReply(
        "Training needs a free account first. Create one, then I’ll ask for the CSV again."
      );
      attachReplyActions(message, [
        ["Create a free account", openSignUpModal],
        ["Sign in", openSignInModal],
      ]);
      return;
    }
    addReply(`I could not use that file. ${error.message}`);
  }
}

async function announceCsvTrainResult(status) {
  if (status.status === "failed") {
    const detail = status.result && status.result.error;
    addReply(`Training finished with an error. ${detail || "The run failed."}`);
    return;
  }
  const modelId = status.result && status.result.model_id;
  if (!modelId) {
    addReply("Training finished. Check Experiments in the Tools sidebar for the run.");
    refreshWorkspace().catch(() => {});
    return;
  }
  try {
    const forecast = await fetchJson(`/workspace/models/${modelId}/predict-latest`, {
      method: "POST",
    });
    const value = Array.isArray(forecast.prediction)
      ? forecast.prediction[0]
      : forecast.prediction;
    const column = forecast.target_column || "value";
    const lines = [
      `Training finished. The next value I forecast for “${column}” is ${formatMetric(value)}. ` +
        "The run and this prediction are in your private workspace under Tools.",
    ];
    const band = uncertaintyText(forecast.uncertainty);
    if (band) lines.push(band);
    const why = whyExplanationText(forecast);
    if (why) lines.push(why);
    try {
      const metrics = await fetchJson(`/workspace/models/${modelId}/metrics`);
      if (metrics && typeof metrics.comparison === "string" && metrics.comparison) {
        lines.push(metrics.comparison);
      }
    } catch {
      // The point forecast still stands when the backtest is unavailable.
    }
    addReply(lines.join("\n\n"));
    refreshWorkspace().catch(() => {});
  } catch (error) {
    addReply(
      `Training finished, but I could not build the first forecast. ${error.message} ` +
        "The run is under Experiments in the Tools sidebar."
    );
  }
}

async function trainReply(question) {
  if (isCsvTrainRequest(question)) {
    return csvTrainIntro();
  }
  if (!isTrainStartRequest(question)) {
    return {
      text:
        "I can train a small model on a CSV in your private workspace, and you can compare the runs under Experiments in the Tools sidebar. Use the Train on my CSV chip to upload a file and pick a column, or say “start training” when you want a run to begin. Until then, nothing is started.",
    };
  }
  let current = null;
  try {
    current = await fetchJson("/training/status");
  } catch {
    // Fall through and just try to start a run.
  }
  if (current && current.status === "running") {
    return {
      text: "A training run is already in progress. Give it about a minute, then ask me again for the status.",
    };
  }
  try {
    const result = await fetchJson("/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config_path: "config/config.yaml" }),
    });
    state.training = result;
    watchTraining();
    return { text: trainingOutcomeText(result.status, result.message) };
  } catch (error) {
    if (error.status === 401) {
      return {
        text:
          "Training needs an account. Create a free one and you can train on your own data in a private workspace — the reply lands in My workspace under Tools.",
        actions: [["Sign in", openSignInModal]],
      };
    }
    return { text: `Training did not start. ${error.message}` };
  }
}

async function voiceReply() {
  try {
    const status = await fetchJson("/voice/status");
    state.voiceStatus = status;
    renderVoicePanel();
    if (!status.signed_in) {
      return {
        text:
          "Every reply here has a small speaker button. Press it and my GQ voice reads the answer aloud. Reading text is always free; GQ voice comes with Nous Plus, and a free account starts with a small allowance.",
        actions: [["Sign in", openSignInModal]],
      };
    }
    if (status.owner_unlimited) {
      return {
        text: "You are signed in as the owner, so GQ voice is unlimited for you. Press any speaker button to listen.",
      };
    }
    const balance = status.balance || {};
    if (status.subscribed) {
      return {
        text:
          `Nous Plus includes GQ voice. You have ${Number(
            balance.total || 0
          ).toLocaleString()} characters left this month. Press any speaker button to listen.`,
      };
    }
    return {
      text:
        `You have ${Number(balance.free || 0).toLocaleString()} free GQ characters left. ` +
        "Nous Plus includes GQ voice every month, with no packs to buy. Press any speaker button to listen, or subscribe to keep going.",
      actions: status.checkout_available ? [["Subscribe", startSubscribe]] : [],
    };
  } catch (error) {
    return { text: `I could not check GQ voice right now. ${error.message}` };
  }
}

function helpReply() {
  return {
    text:
      "Here is what I can actually do, in plain words:\n" +
      "• Weather — a live forecast (high, low, rain, and conditions in °F) for a place you name, or your approximate location\n" +
      "• Markets and crypto — a live price for a ticker or coin, with the caveat that it is not advice\n" +
      "• Moon phase, near-Earth approaches, and the solar system\n" +
      "• Sample models — try the Central Park, AAPL, Bitcoin, near-Earth, or moon illumination starters. No account needed. These are historical samples, not live forecasts\n" +
      "• Training a small model on your CSV — use the Train on my CSV chip for the one-click flow\n" +
      "• GQ voice — the speaker button on each reply, once you have an account. Voice comes with Nous Plus; text stays free\n" +
      "• General questions — a short AI answer, within a daily free allowance\n\n" +
      "If the AI allowance is used up, I will say so and still help with the live topics and sample models above.",
  };
}

function greetReply() {
  return {
    text:
      "Hello! I am Nous. Ask me for a live weather forecast, a market or crypto price, the moon, space, or a general question. You can also try a sample model with no account, or train a model on your data. What would you like to know?",
  };
}

const INTENTS = [
  {
    name: "greet",
    test: /^\s*(hi|hello|hey|yo|good (morning|afternoon|evening))\b/i,
    handler: greetReply,
  },
  {
    name: "help",
    test: /\b(what can you do|what do you do|how do (i|you) use|help me|what are you)\b/i,
    handler: helpReply,
  },
  {
    name: "demo",
    test: /\btry\b[\s\S]{0,80}\bsample\b/i,
    handler: sampleModelReply,
  },
  { name: "moon", test: /\b(moon|lunar|eclipse)\b/i, handler: moonReply },
  {
    name: "space",
    test: /\b(asteroid|near.?earth|close to earth|outer space|in space|space (news|weather|junk|station)|satellites?|comet)\b/i,
    handler: spaceReply,
  },
  {
    name: "solar",
    test: /\b(solar system|planet|planets|jupiter|saturn|mars|venus|mercury)\b/i,
    handler: solarReply,
  },
  {
    name: "crypto",
    test: /\b(crypto|bitcoin|btc|ethereum|eth)\b/i,
    handler: (question) => forecastReply("crypto", question),
  },
  {
    name: "markets",
    test: /\b(stock|stocks|market|markets|s&p|nasdaq|aapl|share price)\b/i,
    handler: (question) => forecastReply("markets", question),
  },
  {
    name: "weather",
    test: /\b(weather|temperature|rainfall|humidity|climate|how hot|how cold|rain|snow|sunny)\b/i,
    handler: (question) => forecastReply("weather", question),
  },
  {
    name: "voice",
    test: /\b(gq( voice)?|voice credit|nous plus|read (it |this |that )?aloud|speaker button|text to speech)\b/i,
    handler: voiceReply,
  },
  {
    name: "train",
    test: /\b(train(ing)?|retrain|fit a model)\b/i,
    handler: trainReply,
  },
];

function matchIntent(question) {
  for (const intent of INTENTS) {
    if (intent.test.test(question)) return intent;
  }
  return null;
}

// Unhinged is never on by default. It lives in memory for this page only and
// starts off on every load, every new chat, and every opened chat. Nothing
// saved (storage, URL, history) turns it on; only the user flipping the switch.
let unhingedOn = false;

function chatMode() {
  return unhingedOn ? "unhinged" : "normal";
}

const NORMAL_PLACEHOLDER = "What do you want to understand?";
const UNHINGED_PLACEHOLDER =
  "Unhinged is on. Ask anything. I'll roast the forecast and keep every number exact.";
const ASK_UNAVAILABLE =
  "I could not reach the general-answer model right now. I can still help with live weather, market prices, the moon, near-Earth space, and training.";

function setChatMode(mode) {
  unhingedOn = mode === "unhinged";
  const on = unhingedOn;
  const button = document.getElementById("unhinged-toggle");
  if (button) {
    button.setAttribute("aria-checked", on ? "true" : "false");
    button.classList.toggle("is-on", on);
    const label = button.querySelector(".switch-label");
    if (label) label.textContent = on ? "Unhinged on" : "Unhinged off";
    button.title = on ? "Unhinged is on. Tap to turn it off." : "Unhinged is off. Tap to turn it on.";
  }
  const input = document.getElementById("chat-input");
  if (input) input.placeholder = on ? UNHINGED_PLACEHOLDER : NORMAL_PLACEHOLDER;
}

function chatHistoryPayload() {
  const log = chatLog();
  if (!log) return [];
  const messages = [...log.querySelectorAll(".message")].slice(-8);
  const history = [];
  for (const message of messages) {
    const role = message.classList.contains("user")
      ? "user"
      : message.classList.contains("assistant")
        ? "assistant"
        : null;
    if (!role) continue;
    const bubble = message.querySelector(".bubble");
    const text = bubble ? String(message.dataset.speech || bubble.textContent).trim() : "";
    if (!text || text === "One moment — checking…") continue;
    history.push({ role, content: text.slice(0, 500) });
  }
  return history;
}

function normalizeAskResult(result, facts) {
  rememberVoiceBudget(result);
  const text = result && typeof result.text === "string" ? result.text.trim() : "";
  const toolsUsed = Array.isArray(result && result.tools_used)
    ? result.tools_used.filter((name) => typeof name === "string" && name.trim())
    : [];
  if (result && result.capped) return { text, capped: true, toolsUsed };
  if (facts && !(result && result.model === true)) return { text, failed: true, toolsUsed };
  if (text) {
    return {
      text,
      voiceResting: Boolean(result && result.voice_resting),
      voiceMessage: typeof result.voice_message === "string" ? result.voice_message : "",
      toolsUsed,
      ...agentReplyExtras(result),
    };
  }
  return null;
}

async function askLive(question, options) {
  const facts = options && typeof options.facts === "string" ? options.facts : "";
  const payload = { question, mode: chatMode() };
  if (facts) payload.facts = facts;
  const history = chatHistoryPayload();
  if (history.length) payload.history = history;
  if (currentThreadId) payload.thread_id = currentThreadId;
  try {
    const agentResult = await fetchJson("/agent/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (agentResult && typeof agentResult.thread_id === "string" && agentResult.thread_id.trim()) {
      currentThreadId = agentResult.thread_id.trim();
      void refreshRemoteChatHistory();
    }
    const normalized = normalizeAskResult(agentResult, facts);
    if (normalized) return normalized;
  } catch {
    // Fall back to the unchanged /live/ask path when the agent route fails.
  }
  try {
    const livePayload = { question, mode: chatMode() };
    if (facts) livePayload.facts = facts;
    const result = await fetchJson("/live/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(livePayload),
    });
    const normalized = normalizeAskResult(result, facts);
    if (normalized) return normalized;
  } catch (error) {
    return { text: (error && error.message) || ASK_UNAVAILABLE, failed: true };
  }
  return { text: ASK_UNAVAILABLE, failed: true };
}

// Multi-step asks ("get X, forecast it, chart it, then explain") go straight to
// the agent so every step runs; the quick local intents only do one thing.
function isMultiStepAsk(question) {
  const text = String(question || "");
  if (/\b(then|and then|after that)\b/i.test(text)) return true;
  const steps = [
    /\b(forecast|predict)/i,
    /\b(chart|graph|plot)\b/i,
    /\b(csv|spreadsheet|export)\b/i,
    /\b(explain|why)\b/i,
    /\b(simulat\w*|what[\s-]if|scenario|cold snap|heat wave)\b/i,
    /\b(back-?test|accuracy)\b/i,
    /\b(compar\w*|versus|vs\.?)\b/i,
    /\b(last|past)\s+\d+\s+(days?|weeks?)\b/i,
    /\btrain\b/i,
  ];
  return steps.filter((pattern) => pattern.test(text)).length >= 2;
}

function agentReplyExtras(result) {
  const extras = {};
  if (result && result.chart && Array.isArray(result.chart.labels) && Array.isArray(result.chart.series)) {
    extras.chart = result.chart;
  }
  if (result && Array.isArray(result.sources) && result.sources.length) extras.sources = result.sources;
  const csv = result && result.csv;
  if (csv && typeof csv.text === "string" && csv.text) {
    const filename = typeof csv.filename === "string" && csv.filename ? csv.filename : "neural-forge-data.csv";
    extras.actions = [
      ["⬇ Download CSV", () => downloadText(csv.text, filename, "text/csv;charset=utf-8")],
    ];
  }
  return extras;
}

async function answer(question) {
  if (isMultiStepAsk(question)) return askLive(question);
  const intent = matchIntent(question);
  if (!intent) return askLive(question);
  if (chatMode() !== "unhinged") return intent.handler(question);
  if (intent.name === "greet" || intent.name === "help") return askLive(question);
  const factual = await intent.handler(question);
  const facts = factual && typeof factual.text === "string" ? factual.text.trim() : "";
  if (!facts) return factual;
  const rewritten = await askLive(question, { facts });
  if (!rewritten || rewritten.capped || rewritten.failed || !rewritten.text) return factual;
  return { ...factual, text: rewritten.text, toolsUsed: rewritten.toolsUsed };
}

const KNOWLEDGE_TOOLS = new Set(["web_search", "wiki_summary", "read_page"]);

/** Tiny "Searched N sources" line; tap to see the links. */
function showSources(message, sources) {
  if (!message || !Array.isArray(sources) || !sources.length) return;
  const bubble = message.querySelector(".bubble");
  if (!bubble) return;
  const details = document.createElement("details");
  details.className = "sources-line";
  const summary = document.createElement("summary");
  summary.textContent = `Searched ${sources.length} source${sources.length === 1 ? "" : "s"}`;
  details.append(summary);
  for (const item of sources.slice(0, 8)) {
    if (!item || typeof item.url !== "string" || !/^https?:\/\//.test(item.url)) continue;
    const anchor = document.createElement("a");
    anchor.href = item.url;
    anchor.textContent = String(item.title || item.url).slice(0, 120);
    anchor.target = "_blank";
    anchor.rel = "noopener noreferrer nofollow";
    details.append(anchor);
  }
  bubble.append(details);
}

function showToolsUsed(message, toolsUsed) {
  if (!message || !Array.isArray(toolsUsed) || !toolsUsed.length) return;
  const actions = message.querySelector(".message-actions");
  if (!actions) return;
  const unique = [...new Set(toolsUsed.map((name) => String(name).trim()).filter(Boolean))].filter(
    (name) => !KNOWLEDGE_TOOLS.has(name)
  );
  for (const name of unique.slice(0, 4)) {
    const chip = document.createElement("span");
    chip.className = "tool-used-chip";
    chip.textContent = `used: ${name}`;
    actions.append(chip);
  }
}

async function handleQuestion(question) {
  addMessage("user", question);
  const pending = addMessage("assistant", "One moment — checking…");
  try {
    const reply = await answer(question);
    pending.remove();
    const replyMessage = addReply(reply.text, reply.chart);
    showToolsUsed(replyMessage, reply.toolsUsed);
    showSources(replyMessage, reply.sources);
    if (reply.voiceResting && !state.toldVoiceResting) {
      state.toldVoiceResting = true;
      showHint(replyMessage, reply.voiceMessage || VOICE_RESTING_MESSAGE);
    }
    if (reply.actions) {
      const actions = replyMessage.querySelector(".message-actions");
      for (const [label, action] of reply.actions) {
        actions.append(makeButton(label, action));
      }
      scrollToBottom();
    }
    clearNotice();
    return typeof reply.text === "string" ? reply.text : "";
  } catch (error) {
    pending.remove();
    addReply(`Something went wrong on my side. ${error.message || error}`);
    reportError(error);
    return "";
  }
}

/* ------------------------------------------------- images & charts (Phase 4) */

const IMAGE_TYPES = ["image/png", "image/jpeg", "image/webp", "image/gif"];
const IMAGE_MAX_SIDE = 1280;
const IMAGE_MAX_BYTES = 1400000;
const IMAGE_TRAIN_MIN_POINTS = 32;
const pendingImage = { dataUrl: "", name: "" };

function firstImageFile(files) {
  if (!files) return null;
  for (const file of files) {
    if (file && typeof file.type === "string" && file.type.startsWith("image/")) return file;
  }
  return null;
}

function promptImagePicker() {
  const input = document.getElementById("image-attach-file");
  if (!input) return;
  input.value = "";
  input.click();
}

function dataUrlBytes(dataUrl) {
  const comma = dataUrl.indexOf(",");
  const base64 = comma >= 0 ? dataUrl.slice(comma + 1) : dataUrl;
  return Math.floor((base64.length * 3) / 4);
}

function readAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(new Error("Could not read that image."));
    reader.readAsDataURL(file);
  });
}

function loadImageElement(src) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("That file does not look like an image."));
    image.src = src;
  });
}

/** Downsize big photos in the browser so uploads stay small and cheap. */
async function prepareImage(file) {
  if (!IMAGE_TYPES.includes(file.type)) {
    throw new Error("Use a PNG, JPEG, WebP, or GIF image.");
  }
  const original = await readAsDataUrl(file);
  const image = await loadImageElement(original);
  const longest = Math.max(image.naturalWidth, image.naturalHeight);
  if (longest <= IMAGE_MAX_SIDE && dataUrlBytes(original) <= IMAGE_MAX_BYTES && file.type !== "image/gif") {
    return original;
  }
  for (const [side, quality] of [
    [IMAGE_MAX_SIDE, 0.88],
    [1024, 0.8],
    [768, 0.72],
  ]) {
    const scale = Math.min(1, side / longest);
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
    canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
    const context = canvas.getContext("2d");
    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    const resized = canvas.toDataURL("image/jpeg", quality);
    if (dataUrlBytes(resized) <= IMAGE_MAX_BYTES) return resized;
  }
  throw new Error("That image is too big even after shrinking. Try a smaller screenshot.");
}

async function setPendingImage(file) {
  const box = document.getElementById("image-attachment");
  const thumb = document.getElementById("image-attachment-thumb");
  const name = document.getElementById("image-attachment-name");
  try {
    const dataUrl = await prepareImage(file);
    pendingImage.dataUrl = dataUrl;
    pendingImage.name = file.name || "image";
    if (thumb) thumb.src = dataUrl;
    if (name) name.textContent = pendingImage.name;
    if (box) box.hidden = false;
    const input = document.getElementById("chat-input");
    if (input) {
      input.placeholder = "Ask about this image, or just press Ask";
      input.focus();
    }
    clearNotice();
  } catch (error) {
    clearPendingImage();
    reportError(error);
  }
}

function clearPendingImage() {
  pendingImage.dataUrl = "";
  pendingImage.name = "";
  const box = document.getElementById("image-attachment");
  const thumb = document.getElementById("image-attachment-thumb");
  if (thumb) thumb.removeAttribute("src");
  if (box) box.hidden = true;
  const input = document.getElementById("chat-input");
  if (input) input.placeholder = "Ask about your data, weather, markets… or drop in a chart";
}

function visionDisplayValue(series, value) {
  const number = asNumber(value);
  if (number == null) return null;
  const shown = series && series.unit === "°C" ? celsiusToFahrenheit(number) : number;
  return Math.round(shown * 100) / 100;
}

function visionChart(data) {
  const series = data && data.series;
  const display = data && data.display_series;
  if (!series || !display || !Array.isArray(display.values) || display.values.length < 2) return null;
  const values = display.values.map((value) => asNumber(value));
  const labels = Array.isArray(series.labels) ? series.labels.map((label) => String(label)) : [];
  while (labels.length < values.length) labels.push(String(labels.length + 1));
  const steps = data.forecast && Array.isArray(data.forecast.steps) ? data.forecast.steps : [];
  const pad = new Array(values.length - 1).fill(null);
  const last = values[values.length - 1];
  const median = [...pad, last];
  const low = [...pad, last];
  const high = [...pad, last];
  steps.forEach((step, index) => {
    labels.push(`+${index + 1}`);
    median.push(visionDisplayValue(series, step.p50));
    low.push(visionDisplayValue(series, step.p10));
    high.push(visionDisplayValue(series, step.p90));
  });
  const unit = display.unit ? ` (${display.unit})` : "";
  const chartSeries = [{ name: `${series.name || "From image"}${unit}`, color: "#e8eaed", values }];
  if (steps.length) {
    chartSeries.push({ name: "Quick forecast", color: "#7cffd6", markers: true, values: median });
  }
  return {
    ariaLabel: "Numbers read from the image with a quick forecast",
    labels,
    series: chartSeries,
    bands: steps.length ? [{ name: "Likely range", color: "#7cffd6", low, high }] : [],
    empty: "No numbers to chart.",
  };
}

function visionCsv(data) {
  const series = data.series;
  const display = data.display_series;
  const unit = String(display.unit || "")
    .replace("°", "")
    .replace(/[^a-z0-9]+/gi, "")
    .toLowerCase();
  const column =
    String(series.name || "value")
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "_")
      .replace(/^_+|_+$/g, "") || "value";
  const header = `label,${column}${unit ? `_${unit}` : ""}`;
  const labels = Array.isArray(series.labels) ? series.labels : [];
  const rows = display.values.map((value, index) => {
    const label = String(labels[index] ?? index + 1).replace(/[",\n]/g, " ");
    return `${label},${value}`;
  });
  return `${header}\n${rows.join("\n")}\n`;
}

function downloadText(text, filename, type) {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

function addImageUserMessage(question, dataUrl) {
  const message = addMessage("user", question || "Read this image");
  const image = document.createElement("img");
  image.className = "bubble-image";
  image.alt = "Image you attached";
  image.src = dataUrl;
  message.querySelector(".bubble").prepend(image);
  scrollToBottom();
  return message;
}

async function handleImageQuestion(question, dataUrl) {
  addImageUserMessage(question, dataUrl);
  const pending = addMessage("assistant", "Reading your image…");
  try {
    const payload = { image: dataUrl, question, mode: chatMode() };
    if (currentThreadId) payload.thread_id = currentThreadId;
    const data = await fetchJson("/agent/vision", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    pending.remove();
    if (data && typeof data.thread_id === "string" && data.thread_id.trim()) {
      currentThreadId = data.thread_id.trim();
      void refreshRemoteChatHistory();
    }
    const text = data && typeof data.text === "string" ? data.text.trim() : "";
    const replyMessage = addReply(text || "I couldn't read that image.", visionChart(data));
    showToolsUsed(replyMessage, data && data.tools_used);
    if (data && data.series && data.display_series) {
      const buttons = [
        [
          "Download CSV",
          () => downloadText(visionCsv(data), "image-series.csv", "text/csv;charset=utf-8"),
        ],
      ];
      if (data.display_series.values.length >= IMAGE_TRAIN_MIN_POINTS) {
        buttons.push([
          "Train on this series",
          () =>
            void handleCsvTrainFile(
              new File([visionCsv(data)], "image-series.csv", { type: "text/csv" })
            ),
        ]);
      }
      if (data.display_series.values.length >= 24) {
        buttons.push([
          "Simulate scenarios",
          () => openSimulatorWithValues(data.display_series.values, data.display_series.unit),
        ]);
      }
      attachReplyActions(replyMessage, buttons);
    }
    const allowance = data && data.allowance;
    if (allowance && Number(allowance.remaining) <= 2) {
      showHint(
        replyMessage,
        `${allowance.remaining} image read${Number(allowance.remaining) === 1 ? "" : "s"} left today.`
      );
    }
    clearNotice();
    return text;
  } catch (error) {
    pending.remove();
    addReply(`I couldn't read that image. ${error.message || error}`);
    return "";
  }
}

async function loadMultimodalAllowance() {
  const note = document.getElementById("multimodal-allowance");
  if (!note) return;
  try {
    const data = await fetchJson("/agent/vision/limits");
    const allowance = data && data.allowance;
    if (!allowance) return;
    note.textContent = `${allowance.remaining} of ${allowance.limit} image reads left today. Resets at midnight UTC.`;
  } catch {
    note.textContent = "";
  }
}

/* --------------------------------------------------------------- talk mode */

const TALK_UNSUPPORTED =
  "This browser can't use hands-free speech recognition, so Talk mode stays off. You can still type, and the speaker button reads replies in GQ voice.";
const TALK_NO_CREDIT =
  "You are out of GQ voice for now, so Talk mode will keep answering in text. Nous Plus includes voice; reading stays free.";
const TALK_MIC_BLOCKED = "The microphone is blocked. Allow it in the browser, or keep typing.";
const TALK_SIGN_IN =
  "GQ voice needs a free account, so Talk mode will keep answering in text. Tap Account to sign in; reading stays free.";
const TALK_PLAYBACK_BLOCKED =
  "This browser blocked GQ voice after the Talk tap. Tap Talk again to unlock sound, or use the speaker button on a reply. Text answers keep going.";
const TALK_VOICE_FAILED =
  "GQ voice could not play that reply, so Talk mode will keep answering in text. Reading stays free.";
const TALK_AFTER_SPEECH_MS = 550;
const TALK_RESTART_MS = 300;
const TALK_PLAYBACK_PAD_MS = 2500;
const TALK_PLAYBACK_MAX_MS = 120000;
// Tiny silent WAV used only to unlock autoplay during the Talk tap gesture.
const TALK_SILENT_WAV =
  "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=";

const talk = {
  active: false,
  state: "idle",
  recognition: null,
  audio: null,
  player: null,
  objectUrl: null,
  turn: 0,
  voiceOff: false,
  toldVoice: false,
  toldPlayback: false,
  restartTimer: 0,
  opening: false,
  armed: false,
  autoBound: false,
  overlayBound: false,
};

function speechRecognitionCtor() {
  return window.SpeechRecognition || window.webkitSpeechRecognition || null;
}

function speechPlainText(value) {
  // Money, dates, and units are normalized on the server before GQ voice.
  return String(value || "").replace(/\s+/g, " ").trim();
}

function setTalkState(state) {
  talk.state = state;
  const button = document.getElementById("talk-toggle");
  if (!button) return;
  const labels = {
    idle: "Talk",
    armed: "Talk on",
    listening: "Listening",
    thinking: "Thinking",
    speaking: "Speaking",
  };
  button.textContent = labels[state] || "Talk";
  button.dataset.state = state;
  const on = Boolean(talk.active || talk.armed);
  button.setAttribute("aria-pressed", on ? "true" : "false");
  button.setAttribute(
    "aria-label",
    talk.active
      ? `${button.textContent}. Tap to stop or interrupt.`
      : talk.armed
        ? "Talk mode is on. Tap to turn off."
        : "Start Talk mode"
  );
  document.body.classList.toggle("talk-armed", Boolean(talk.armed && !talk.active));
  const status = document.getElementById("talk-status");
  if (status) {
    if (talk.armed && !talk.active) {
      status.hidden = false;
      status.textContent = "Talk is on. Tap anywhere to start listening.";
    } else {
      status.hidden = true;
      status.textContent = "";
    }
  }
}

function isAppWrapper() {
  try {
    if (window.Capacitor || window.cordova) return true;
    if (window.navigator && window.navigator.standalone === true) return true;
    if (window.matchMedia && window.matchMedia("(display-mode: standalone)").matches) {
      return true;
    }
  } catch {
    // matchMedia can throw in odd test or older environments.
  }
  return false;
}

function isPhoneSurface() {
  if (isAppWrapper()) return true;
  try {
    if (window.matchMedia && window.matchMedia("(pointer: coarse)").matches) return true;
  } catch {
    // Ignore unsupportive matchMedia implementations.
  }
  try {
    if (window.matchMedia && window.matchMedia("(max-width: 900px)").matches) return true;
  } catch {
    // Ignore unsupportive matchMedia implementations.
  }
  const ua = String(
    (window.navigator && (window.navigator.userAgent || window.navigator.vendor)) || ""
  );
  return /iPhone|iPod|iPad|Android.+Mobile|webOS|BlackBerry|IEMobile|Opera Mini/i.test(ua);
}

function voiceChoices() {
  const list = state.voiceStatus && state.voiceStatus.voices;
  return Array.isArray(list) && list.length ? list : DEFAULT_VOICES;
}

// GQ unless the person picked the other voice. Falls back to GQ for any value
// that is not on the list.
function selectedVoice() {
  try {
    const raw = localStorage.getItem(VOICE_PICK_KEY);
    if (raw && voiceChoices().some((choice) => choice.key === raw)) return raw;
  } catch {
    // Storage can be blocked; GQ stays the default.
  }
  return "gq";
}

function renderVoicePickers() {
  const current = selectedVoice();
  document.querySelectorAll("[data-voice-picker]").forEach((box) => {
    box.replaceChildren();
    voiceChoices().forEach((choice) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "voice-picker-option";
      button.setAttribute("role", "radio");
      button.setAttribute("aria-checked", choice.key === current ? "true" : "false");
      button.textContent = choice.name;
      button.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        try {
          localStorage.setItem(VOICE_PICK_KEY, choice.key);
        } catch {
          // The pick still applies until the page reloads.
        }
        renderVoicePickers();
      });
      box.append(button);
    });
  });
}

function talkPreference() {
  try {
    const raw = localStorage.getItem(TALK_PREF_KEY);
    if (raw === "on" || raw === "off") return raw;
  } catch {
    // Storage can be blocked; fall through to the default.
  }
  return null;
}

function setTalkPreference(value) {
  try {
    localStorage.setItem(TALK_PREF_KEY, value === "on" ? "on" : "off");
  } catch {
    // The toggle still works for this page when storage is blocked.
  }
}

function shouldAutoArmTalk() {
  // Desktop stays manual. Phones default on unless the user turned Talk off.
  return isPhoneSurface() && talkPreference() !== "off";
}

function talkGestureRequired() {
  try {
    if (navigator.userActivation) {
      return !(navigator.userActivation.isActive || navigator.userActivation.hasBeenActive);
    }
  } catch {
    // Assume a gesture is required when the API is missing or throws.
  }
  return true;
}

function isTalkOffControl(target) {
  if (!target || typeof target.closest !== "function") return false;
  return Boolean(
    target.closest(
      "#talk-toggle, #talk-start-dismiss, .modal-backdrop, #sidebar-toggle, #sidebar, .sidebar-backdrop"
    )
  );
}

function talkStartOverlay() {
  return document.getElementById("talk-start-overlay");
}

function showTalkStartOverlay() {
  const overlay = talkStartOverlay();
  if (!overlay) return;
  overlay.hidden = false;
}

function hideTalkStartOverlay() {
  const overlay = talkStartOverlay();
  if (!overlay) return;
  overlay.hidden = true;
}

function startTalkFromGesture() {
  if (talk.active || talk.opening) return;
  talk.armed = false;
  hideTalkStartOverlay();
  unbindTalkAutoStart();
  setTalkPreference("on");
  unlockTalkAudio();
  beginTalk().catch(() => {
    if (talk.active) stopTalk();
    addReply(TALK_MIC_BLOCKED);
  });
}

function onTalkAutoStartPointer(event) {
  if (!talk.armed || talk.active || talk.opening) return;
  if (isTalkOffControl(event && event.target)) return;
  startTalkFromGesture();
}

function onTalkAutoStartFocus() {
  if (!talk.armed || talk.active || talk.opening) return;
  startTalkFromGesture();
}

function bindTalkOverlayControls() {
  if (talk.overlayBound) return;
  talk.overlayBound = true;
  renderVoicePickers();
  const startButton = document.getElementById("talk-start-button");
  if (startButton) {
    startButton.addEventListener("click", (event) => {
      event.preventDefault();
      startTalkFromGesture();
    });
  }
  const dismiss = document.getElementById("talk-start-dismiss");
  if (dismiss) {
    dismiss.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      disarmTalk({ persist: true });
    });
  }
}

function bindTalkAutoStart() {
  bindTalkOverlayControls();
  if (talk.autoBound) return;
  talk.autoBound = true;
  document.addEventListener("pointerdown", onTalkAutoStartPointer, true);
  const input = document.getElementById("chat-input");
  if (input) input.addEventListener("focus", onTalkAutoStartFocus);
}

function unbindTalkAutoStart() {
  if (!talk.autoBound) return;
  talk.autoBound = false;
  document.removeEventListener("pointerdown", onTalkAutoStartPointer, true);
  const input = document.getElementById("chat-input");
  if (input) input.removeEventListener("focus", onTalkAutoStartFocus);
}

function disarmTalk(options = {}) {
  talk.armed = false;
  hideTalkStartOverlay();
  unbindTalkAutoStart();
  if (talk.active || talk.opening) {
    stopTalk();
  } else {
    setTalkState("idle");
  }
  if (options.persist) setTalkPreference("off");
}

function armTalkForPhone() {
  if (!shouldAutoArmTalk()) return;
  if (!speechRecognitionCtor()) return;
  talk.armed = true;
  setTalkState("armed");
  bindTalkAutoStart();
  if (talkGestureRequired()) {
    showTalkStartOverlay();
  } else {
    startTalkFromGesture();
  }
}

function ensureTalkPlayer() {
  if (!talk.player) talk.player = new Audio();
  return talk.player;
}

function unlockTalkAudio() {
  const audio = ensureTalkPlayer();
  // Do not clear src in the play() callback — a later GQ blob may already
  // be assigned to this same unlocked element by the time the silent clip ends.
  try {
    audio.muted = true;
    audio.src = TALK_SILENT_WAV;
    const playPromise = audio.play();
    if (playPromise && typeof playPromise.then === "function") {
      playPromise
        .then(() => {
          try {
            audio.pause();
            audio.currentTime = 0;
          } catch {
            // Ignore seek failures on still-loading silent buffers.
          }
          audio.muted = false;
        })
        .catch(() => {
          audio.muted = false;
        });
    } else {
      audio.muted = false;
    }
  } catch {
    audio.muted = false;
  }
}

function clearTalkObjectUrl() {
  if (!talk.objectUrl) return;
  URL.revokeObjectURL(talk.objectUrl);
  talk.objectUrl = null;
}

function stopTalkAudio() {
  const audio = talk.audio;
  talk.audio = null;
  if (!audio) {
    clearTalkObjectUrl();
    return;
  }
  if (typeof audio._nfStop === "function") audio._nfStop();
  try {
    audio.pause();
  } catch {
    // Already stopped.
  }
  clearTalkObjectUrl();
  try {
    audio.removeAttribute("src");
    audio.load();
  } catch {
    // Keep the unlocked element even if src clear fails.
  }
}

function abortTalkRecognition() {
  const recognition = talk.recognition;
  talk.recognition = null;
  if (!recognition) return;
  recognition.onend = null;
  recognition.onresult = null;
  recognition.onerror = null;
  try {
    recognition.abort();
  } catch {
    // Already stopped.
  }
}

function scheduleListen(delay) {
  window.clearTimeout(talk.restartTimer);
  talk.restartTimer = window.setTimeout(() => {
    if (talk.active && talk.state === "listening") startListening();
  }, delay);
}

function stopTalk() {
  window.clearTimeout(talk.restartTimer);
  talk.active = false;
  talk.armed = false;
  talk.turn += 1;
  abortTalkRecognition();
  stopTalkAudio();
  hideTalkStartOverlay();
  unbindTalkAutoStart();
  setTalkState("idle");
}

function startListening(options = {}) {
  if (!talk.active) return;
  window.clearTimeout(talk.restartTimer);
  const Recognition = speechRecognitionCtor();
  if (!Recognition) {
    stopTalk();
    addReply(TALK_UNSUPPORTED);
    return;
  }
  stopTalkAudio();
  abortTalkRecognition();
  const recognition = new Recognition();
  talk.recognition = recognition;
  recognition.lang = "en-US";
  recognition.interimResults = false;
  recognition.continuous = false;
  recognition.onresult = (event) => {
    const first = event.results && event.results[0] && event.results[0][0];
    const transcript = first && first.transcript ? String(first.transcript) : "";
    setTalkState("thinking");
    onTalkUtterance(transcript).catch(() => {
      if (talk.active) startListening();
    });
  };
  recognition.onerror = (event) => {
    if (!talk.active) return;
    const code = event && event.error;
    if (code === "not-allowed" || code === "service-not-allowed") {
      stopTalk();
      addReply(TALK_MIC_BLOCKED);
      return;
    }
    if (code === "aborted") return;
    if (talk.state === "listening") scheduleListen(TALK_RESTART_MS);
  };
  recognition.onend = () => {
    if (talk.active && talk.state === "listening") scheduleListen(TALK_RESTART_MS);
  };
  setTalkState("listening");
  const kick = () => {
    if (!talk.active || talk.recognition !== recognition || talk.state !== "listening") return;
    try {
      recognition.start();
    } catch {
      scheduleListen(TALK_RESTART_MS);
    }
  };
  // First tap must start inside the user-activation window on iOS.
  if (options.immediate) {
    kick();
    return;
  }
  window.setTimeout(kick, TALK_RESTART_MS);
}

function tellTalkVoiceIssue(message) {
  if (talk.toldVoice || talk.toldPlayback) return;
  talk.toldPlayback = true;
  addReply(message);
}

function waitForTalkPlayback(audio) {
  return new Promise((resolve) => {
    let settled = false;
    let timer = 0;
    const finish = (result) => {
      if (settled) return;
      settled = true;
      window.clearTimeout(timer);
      audio.removeEventListener("ended", onEnded);
      audio.removeEventListener("error", onError);
      audio.removeEventListener("loadedmetadata", onMeta);
      resolve(result);
    };
    const armTimeout = (ms) => {
      window.clearTimeout(timer);
      timer = window.setTimeout(() => finish("timeout"), ms);
    };
    const onEnded = () => finish("ended");
    const onError = () => finish("error");
    const onMeta = () => {
      if (settled) return;
      const durationMs =
        Number.isFinite(audio.duration) && audio.duration > 0
          ? audio.duration * 1000 + TALK_PLAYBACK_PAD_MS
          : TALK_PLAYBACK_MAX_MS;
      armTimeout(Math.min(Math.max(durationMs, TALK_PLAYBACK_PAD_MS), TALK_PLAYBACK_MAX_MS));
    };
    audio._nfStop = () => finish("stopped");
    audio.addEventListener("ended", onEnded);
    audio.addEventListener("error", onError);
    audio.addEventListener("loadedmetadata", onMeta);
    armTimeout(TALK_PLAYBACK_MAX_MS);
    audio.play().catch((err) => {
      const name = err && err.name;
      finish(name === "NotAllowedError" ? "blocked" : "error");
    });
  });
}

async function speakTalkTurn(text) {
  const mine = talk.turn;
  const spoken = speechPlainText(text).slice(0, MAX_VOICE_CHARS);
  if (!talk.active || mine !== talk.turn) return;
  if (!spoken || talk.voiceOff) {
    startListening();
    return;
  }
  if (!(await voicePlaybackAllowed())) {
    if (!talk.active || mine !== talk.turn) return;
    if (!state.toldVoiceResting) {
      state.toldVoiceResting = true;
      addReply(VOICE_RESTING_MESSAGE);
    }
    startListening();
    return;
  }
  setTalkState("speaking");
  let response;
  try {
    response = await fetch("/voice/speak", {
      method: "POST",
      credentials: "same-origin",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({ text: spoken, voice: selectedVoice() }),
    });
  } catch {
    if (!talk.active || mine !== talk.turn) return;
    tellTalkVoiceIssue(TALK_VOICE_FAILED);
    startListening();
    return;
  }
  if (!talk.active || mine !== talk.turn) return;
  if (response.status === 401 || response.status === 402) {
    talk.voiceOff = true;
    if (!talk.toldVoice) {
      talk.toldVoice = true;
      const reply = addReply(response.status === 401 ? TALK_SIGN_IN : TALK_NO_CREDIT);
      const actions = reply.querySelector(".message-actions");
      if (actions && response.status === 402) {
        actions.append(makeButton("Subscribe", startSubscribe));
      }
    }
    startListening();
    return;
  }
  if (!response.ok) {
    tellTalkVoiceIssue(TALK_VOICE_FAILED);
    startListening();
    return;
  }
  const blob = await response.blob();
  if (!talk.active || mine !== talk.turn) return;
  const objectUrl = URL.createObjectURL(blob);
  const audio = ensureTalkPlayer();
  clearTalkObjectUrl();
  talk.objectUrl = objectUrl;
  talk.audio = audio;
  audio.src = objectUrl;
  const played = await waitForTalkPlayback(audio);
  if (talk.audio === audio) talk.audio = null;
  // Revoke only after playback finished or was interrupted — never while waiting.
  if (talk.objectUrl === objectUrl) clearTalkObjectUrl();
  try {
    audio.removeAttribute("src");
    audio.load();
  } catch {
    // Keep the unlocked element for the next turn.
  }
  if (!(talk.active && talk.state === "speaking" && mine === talk.turn)) return;
  if (played === "blocked") {
    tellTalkVoiceIssue(TALK_PLAYBACK_BLOCKED);
  } else if (played === "error" || played === "timeout") {
    tellTalkVoiceIssue(TALK_VOICE_FAILED);
  }
  if (played === "ended") {
    await new Promise((resolve) => window.setTimeout(resolve, TALK_AFTER_SPEECH_MS));
  }
  if (talk.active && mine === talk.turn) startListening();
}

async function onTalkUtterance(transcript) {
  const question = String(transcript || "").trim();
  if (!talk.active) return;
  if (!question) {
    startListening();
    return;
  }
  setTalkState("thinking");
  // handleQuestion returns the final answer text only (never the placeholder).
  const answerText = await handleQuestion(question);
  if (!talk.active) return;
  await speakTalkTurn(answerText);
}

async function beginTalk() {
  if (talk.opening || talk.active) return;
  if (!speechRecognitionCtor()) {
    addReply(TALK_UNSUPPORTED);
    return;
  }
  // Unlock autoplay and start recognition inside the Talk tap gesture.
  unlockTalkAudio();
  talk.opening = true;
  talk.active = true;
  talk.voiceOff = false;
  talk.toldVoice = false;
  talk.toldPlayback = false;
  startListening({ immediate: true });
  try {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      let stream = null;
      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      } catch {
        // Recognition may already be running; only stop when the mic is blocked.
        if (talk.active) {
          stopTalk();
          addReply(TALK_MIC_BLOCKED);
        }
        return;
      } finally {
        if (stream) {
          for (const track of stream.getTracks()) track.stop();
        }
      }
    }
  } finally {
    talk.opening = false;
  }
}

function onTalkClick() {
  if (talk.armed && !talk.active) {
    // Easy off switch while Talk is armed and waiting for a start gesture.
    disarmTalk({ persist: true });
    return;
  }
  if (!talk.active) {
    // Prime audio synchronously on the click path before any await.
    hideTalkStartOverlay();
    unbindTalkAutoStart();
    talk.armed = false;
    if (isPhoneSurface()) setTalkPreference("on");
    unlockTalkAudio();
    beginTalk().catch(() => {
      if (talk.active) stopTalk();
      addReply(TALK_MIC_BLOCKED);
    });
    return;
  }
  if (talk.state === "speaking") {
    talk.turn += 1;
    stopTalkAudio();
    startListening({ immediate: true });
    return;
  }
  stopTalk();
  if (isPhoneSurface()) setTalkPreference("off");
}

/* ------------------------------------------------------------------ voice */

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

function planPriceText(plan) {
  return `$${Number(plan.price_usd).toFixed(Number(plan.price_usd) % 1 ? 2 : 0)}/month`;
}

function dateLabel(stamp) {
  const date = new Date(stamp);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

function renderVoicePanel() {
  renderVoicePickers();
  const status = state.voiceStatus;
  const grid = document.getElementById("voice-status-grid");
  const planBox = document.getElementById("voice-plan");
  const usage = document.getElementById("voice-usage-list");
  const message = document.getElementById("voice-message");
  planBox.replaceChildren();
  usage.replaceChildren();
  if (!status) {
    fillStatGrid(grid, [["GQ voice", "checking"]]);
    return;
  }
  const plan = status.plan || { name: "Nous Plus", price_usd: 9, monthly_chars: 40000 };
  const sub = status.subscription || null;
  const cards = [["Text on the site", "always free"]];
  if (!status.signed_in) {
    cards.push(["GQ voice", status.voice ? "sign in to listen" : "unavailable"]);
  } else if (status.owner_unlimited) {
    cards.push(["GQ voice", status.voice ? "unlimited (owner)" : "unavailable"]);
  } else {
    const balance = status.balance || { free: 0, purchased: 0, plan: 0, spent: 0 };
    if (status.subscribed && sub) {
      cards.push([plan.name, sub.cancel_at_period_end ? "ends soon" : "active"]);
      cards.push([
        "Voice left this month",
        `${Number(balance.plan).toLocaleString()} of ${Number(sub.allowance_total).toLocaleString()}`,
      ]);
      cards.push([
        sub.cancel_at_period_end ? "Ends on" : "Renews on",
        dateLabel(sub.period_end) || "—",
      ]);
    }
    cards.push(["Free GQ left", Number(balance.free).toLocaleString()]);
    if (Number(balance.purchased) > 0) {
      cards.push(["Bonus GQ left", Number(balance.purchased).toLocaleString()]);
    }
  }
  fillStatGrid(grid, cards);
  if (!status.owner_unlimited) {
    const card = document.createElement("article");
    card.className = "starter-card";
    const title = document.createElement("h3");
    title.textContent = `${plan.name} · ${planPriceText(plan)}`;
    const detail = document.createElement("p");
    detail.className = "hint";
    const action = document.createElement("button");
    action.type = "button";
    action.className = "btn primary btn-small";
    if (status.subscribed) {
      detail.textContent =
        "GQ voice is included. Talk mode and every speaker button just work. " +
        "Cancel any time.";
      action.textContent = "Manage subscription";
      action.disabled = !(sub && sub.can_manage);
      action.addEventListener("click", manageSubscription);
    } else {
      detail.textContent =
        `Includes GQ voice: ${Number(plan.monthly_chars).toLocaleString()} characters ` +
        "every month. No packs, no add-ons.";
      if (!status.signed_in) {
        action.textContent = "Sign up to subscribe";
        action.addEventListener("click", openSignInModal);
      } else if (!status.checkout_available) {
        action.textContent = "Subscriptions are not live yet";
        action.disabled = true;
      } else {
        action.textContent = `Subscribe · ${planPriceText(plan)}`;
        action.addEventListener("click", startSubscribe);
      }
    }
    card.append(title, detail, action);
    if (status.checkout_test_mode && !status.subscribed) {
      const note = document.createElement("p");
      note.className = "hint";
      note.textContent = "Checkout is in test mode and will not charge a real card.";
      card.append(note);
    }
    planBox.append(card);
  }
  for (const row of status.usage || []) {
    const item = document.createElement("li");
    const cached = row.cached ? " (replayed, no charge)" : "";
    item.textContent =
      `${String(row.created_at).slice(0, 19)} · ${row.kind} · ` +
      `${Number(row.chars).toLocaleString()} chars${cached}`;
    usage.append(item);
  }
  if (!usage.children.length) {
    const item = document.createElement("li");
    item.textContent = "No GQ usage yet.";
    usage.append(item);
  }
  message.textContent = status.message || "";
}

// Stripe sends the buyer back to /?subscribed=<session id>#voice. Keep the id
// in memory, clean the address bar, and confirm it once the app is open.
(function readCheckoutReturn() {
  const params = new URLSearchParams(window.location.search);
  const session = params.get("subscribed");
  if (!session) return;
  state.checkoutSession = session;
  params.delete("subscribed");
  const rest = params.toString();
  window.history.replaceState(
    {},
    "",
    window.location.pathname + (rest ? `?${rest}` : "") + window.location.hash
  );
})();

async function confirmCheckout(sessionId) {
  const message = document.getElementById("voice-message");
  for (let attempt = 0; attempt < 4; attempt += 1) {
    try {
      const status = await fetchJson("/voice/sync", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      });
      state.voiceStatus = status;
      if (status.subscribed) {
        state.checkoutSession = null;
        renderVoicePanel();
        showToolView("voice");
        message.textContent = "You are subscribed. GQ voice is on.";
        return true;
      }
    } catch (error) {
      if (error.status === 401) return false;
    }
    await new Promise((resolve) => setTimeout(resolve, 2500));
  }
  state.checkoutSession = null;
  return false;
}

async function refreshVoiceStatus() {
  if (state.checkoutSession && state.account) {
    const sessionId = state.checkoutSession;
    state.checkoutSession = null;
    if (await confirmCheckout(sessionId)) return;
  }
  try {
    state.voiceStatus = await fetchJson("/voice/status");
  } catch (error) {
    document.getElementById("voice-message").textContent = error.message;
    return;
  }
  renderVoicePanel();
}

async function startSubscribe() {
  const message = document.getElementById("voice-message");
  if (!state.account) {
    openSignInModal();
    return;
  }
  try {
    const result = await fetchJson("/voice/checkout", { method: "POST" });
    if (result && result.available && result.url) {
      window.location.assign(result.url);
      return;
    }
    message.textContent =
      (result && result.detail) || "Subscriptions are not live yet. Text stays free.";
  } catch (error) {
    message.textContent = error.message;
    if (error.status === 409) refreshVoiceStatus().catch(() => {});
  }
  showToolView("voice");
}

async function manageSubscription() {
  const message = document.getElementById("voice-message");
  try {
    const result = await fetchJson("/voice/portal", { method: "POST" });
    if (result && result.url) {
      window.location.assign(result.url);
      return;
    }
    message.textContent = "Subscription management is not available right now.";
  } catch (error) {
    message.textContent = error.message;
  }
}

function openVoicePanel() {
  showToolView("voice");
}

/* ------------------------------------------------------------------- auth */

function openSignInModal() {
  const modal = document.getElementById("sign-in-modal");
  modal.hidden = false;
  refreshAuthModal();
  const field = document.getElementById("sign-in-email");
  if (field && !state.account) field.focus();
}

function closeSignInModal() {
  document.getElementById("sign-in-modal").hidden = true;
}

function refreshAuthModal() {
  const signedIn = document.getElementById("account-signed-in");
  const signedOut = document.getElementById("account-signed-out");
  const label = document.getElementById("account-label");
  const button = document.getElementById("account-btn");
  const security = document.getElementById("account-security");
  if (state.account) {
    signedIn.hidden = false;
    signedOut.hidden = true;
    const auth = state.account.auth === "local" ? "local owner" : state.account.role;
    label.textContent = `${state.account.email} (${auth})`;
    button.textContent = state.account.email;
    // Display only. The server decides what the account may do.
    if (security) {
      security.hidden = state.account.auth === "local";
      if (!security.hidden) refreshSecuritySummary().catch(() => {});
    }
  } else {
    signedIn.hidden = true;
    signedOut.hidden = false;
    if (security) security.hidden = true;
    button.textContent = "Log in";
  }
}

async function refreshAccount() {
  try {
    state.account = await fetchJson("/auth/me");
  } catch (error) {
    if (error.status !== 401) reportError(error);
    state.account = null;
  }
  refreshAuthModal();
  return state.account;
}

async function refreshAccountDependents() {
  await Promise.all([
    refreshVoiceStatus().catch(() => {}),
    refreshWorkspace().catch(() => {}),
    refreshExperiments().catch(() => {}),
    refreshStatusLine().catch(() => {}),
  ]);
}

async function refreshStatusLine() {
  try {
    const [health, modelInfo] = await Promise.all([
      fetchJson("/health"),
      fetchJson("/model/info"),
    ]);
    document.getElementById("status-line").textContent = modelInfo.checkpoint_loaded
      ? `Checkpoint loaded · ${modelInfo.architecture}`
      : "No checkpoint loaded";
  } catch {
    document.getElementById("status-line").textContent = "Model status unknown";
  }
}

/* -------------------------------------------------------------- workspace */

function appendTextItem(list, text) {
  const item = document.createElement("li");
  item.textContent = text;
  list.append(item);
}

async function renameWorkspaceDataset(dataset) {
  const next = window.prompt("Rename series", dataset.name || "");
  if (next == null) return;
  const name = String(next).trim();
  if (!name || name === dataset.name) return;
  await fetchJson(`/workspace/datasets/${encodeURIComponent(dataset.id)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  await refreshWorkspace();
}

async function deleteWorkspaceDataset(dataset) {
  const ok = window.confirm(`Delete series “${dataset.name}”? This cannot be undone.`);
  if (!ok) return;
  await fetchJson(`/workspace/datasets/${encodeURIComponent(dataset.id)}`, {
    method: "DELETE",
  });
  await refreshWorkspace();
}

async function retrainWorkspaceDataset(dataset) {
  const pending = addMessage("assistant", `Queueing a retrain on “${dataset.name}”…`);
  try {
    const result = await fetchJson(
      `/workspace/datasets/${encodeURIComponent(dataset.id)}/train`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      }
    );
    pending.querySelector(".bubble").textContent =
      `Queued retrain for “${dataset.name}”` +
      (result && result.job_id ? ` (job ${result.job_id}).` : ".");
    watchTraining();
  } catch (error) {
    pending.querySelector(".bubble").textContent =
      error.message || "Could not start retrain.";
  }
}

function renderWorkspaceLists(datasets, models, predictions) {
  const datasetList = document.getElementById("dataset-list");
  const modelList = document.getElementById("model-list");
  const predictionList = document.getElementById("prediction-list");
  datasetList.replaceChildren();
  modelList.replaceChildren();
  predictionList.replaceChildren();
  if (!datasets.length) appendTextItem(datasetList, "No datasets yet.");
  for (const dataset of datasets) {
    const row = document.createElement("li");
    const shared = dataset.shared ? "shared starter" : "private copy";
    const label = document.createElement("span");
    label.textContent = `${dataset.name} · ${shared}`;
    const actions = document.createElement("div");
    actions.className = "message-actions";
    const rename = document.createElement("button");
    rename.type = "button";
    rename.className = "btn secondary btn-small";
    rename.textContent = "Rename";
    rename.addEventListener("click", () => {
      void renameWorkspaceDataset(dataset).catch((error) => reportError(error));
    });
    const retrain = document.createElement("button");
    retrain.type = "button";
    retrain.className = "btn secondary btn-small";
    retrain.textContent = "Retrain";
    retrain.addEventListener("click", () => {
      showToolView("chat");
      void retrainWorkspaceDataset(dataset).catch((error) => reportError(error));
    });
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "btn secondary btn-small";
    remove.textContent = "Delete";
    remove.addEventListener("click", () => {
      void deleteWorkspaceDataset(dataset).catch((error) => reportError(error));
    });
    actions.append(rename, retrain, remove);
    row.append(label, actions);
    datasetList.append(row);
  }
  if (!models.length) appendTextItem(modelList, "No models yet.");
  for (const model of models) {
    const shared = model.shared ? "starter weights" : "your checkpoint";
    appendTextItem(modelList, `${model.name} · ${shared}`);
  }
  if (!predictions.length) appendTextItem(predictionList, "No saved predictions yet.");
  for (const prediction of predictions) {
    const value =
      prediction.result && prediction.result.prediction
        ? JSON.stringify(prediction.result.prediction)
        : "";
    appendTextItem(
      predictionList,
      `${prediction.starter_id || prediction.model_id} ${value}`
    );
  }
}

async function refreshWorkspace() {
  const note = document.getElementById("workspace-note");
  const signedInBox = document.getElementById("workspace-signed-in");
  if (!state.account) {
    signedInBox.hidden = true;
    note.hidden = false;
    note.textContent = "Sign in to see your sample datasets, models, and predictions.";
    return;
  }
  let payloads;
  try {
    payloads = await Promise.all([
      fetchJson("/workspace/starters"),
      fetchJson("/workspace/datasets"),
      fetchJson("/workspace/models"),
      fetchJson("/workspace/predictions"),
    ]);
  } catch (error) {
    if (error.status === 401) {
      state.account = null;
      refreshAuthModal();
      return refreshWorkspace();
    }
    throw error;
  }
  const [starters, datasets, models, predictions] = payloads;
  note.hidden = true;
  signedInBox.hidden = false;
  const host = document.getElementById("starter-list");
  host.replaceChildren();
  for (const starter of starters.starters || []) {
    const card = document.createElement("article");
    card.className = "starter-card";
    const title = document.createElement("h3");
    title.textContent = starter.name;
    const summary = document.createElement("p");
    summary.className = "hint";
    summary.textContent = starter.summary;
    const output = document.createElement("pre");
    output.className = "output";
    output.hidden = true;
    const predict = document.createElement("button");
    predict.type = "button";
    predict.className = "btn secondary btn-small";
    predict.textContent = "Predict";
    const train = document.createElement("button");
    train.type = "button";
    train.className = "btn primary btn-small";
    train.textContent = "Train on this sample";
    predict.addEventListener("click", async () => {
      output.hidden = false;
      output.textContent = "Predicting…";
      try {
        const result = await fetchJson(`/workspace/starters/${starter.id}/predict`, {
          method: "POST",
        });
        output.textContent = JSON.stringify(result, null, 2);
      } catch (error) {
        output.textContent = error.message;
      }
    });
    train.addEventListener("click", async () => {
      output.hidden = false;
      output.textContent = "Queueing…";
      try {
        const result = await fetchJson(`/workspace/starters/${starter.id}/train`, {
          method: "POST",
        });
        output.textContent = JSON.stringify(result, null, 2);
        watchTraining();
      } catch (error) {
        output.textContent = error.message;
      }
    });
    const buttons = document.createElement("div");
    buttons.className = "message-actions";
    buttons.append(predict, train);
    card.append(title, summary, buttons, output);
    host.append(card);
  }
  renderWorkspaceLists(
    datasets.datasets || [],
    models.models || [],
    predictions.predictions || []
  );
}

/* ------------------------------------------------------- training + runs */

async function refreshExperiments() {
  const body = document.getElementById("experiments-body");
  const note = document.getElementById("experiments-note");
  if (!state.account) {
    body.replaceChildren();
    note.textContent = "Sign in to see your training runs here.";
    return;
  }
  try {
    const payload = await fetchJson("/experiments?limit=25");
    state.experiments = payload.experiments || [];
  } catch (error) {
    if (error.status === 401) {
      state.account = null;
      refreshAuthModal();
      return;
    }
    throw error;
  }
  body.replaceChildren();
  if (!state.experiments.length) {
    note.textContent = "No training runs yet. Start one under Train a model.";
    return;
  }
  note.textContent = "Your most recent training runs.";
  for (const experiment of state.experiments) {
    const row = document.createElement("tr");
    const name = document.createElement("td");
    name.textContent = experiment.name || `Run ${experiment.id}`;
    const status = document.createElement("td");
    const badge = document.createElement("span");
    const statusText = String(experiment.status || "—");
    badge.className = "badge";
    if (["running", "completed", "failed"].includes(statusText)) {
      badge.classList.add(statusText);
    }
    badge.textContent = statusText;
    status.append(badge);
    const loss = document.createElement("td");
    loss.textContent = formatMetric(experiment.best_loss);
    row.append(name, status, loss);
    body.append(row);
  }
}

async function startTraining(output) {
  output.hidden = false;
  output.textContent = "Starting…";
  try {
    const response = await fetchJson("/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config_path: "config/config.yaml" }),
    });
    output.textContent = JSON.stringify(response, null, 2);
    watchTraining();
  } catch (error) {
    if (error.status === 401) {
      output.textContent =
        "Training needs an account. Use the Sign in button — owner accounts train the shared champion, free accounts train in their own workspace.";
      return;
    }
    output.textContent = error.message;
  }
}

function watchTraining() {
  if (state.trainingWatch) return;
  state.trainingWatch = setInterval(async () => {
    if (!state.account) {
      clearInterval(state.trainingWatch);
      state.trainingWatch = null;
      return;
    }
    try {
      const status = await fetchJson("/training/status");
      state.training = status;
      const busy = status.status === "running" || status.status === "queued";
      if (!busy) {
        clearInterval(state.trainingWatch);
        state.trainingWatch = null;
        refreshExperiments().catch(() => {});
        refreshStatusLine().catch(() => {});
        if (state.csvTrainJobId) {
          state.csvTrainJobId = null;
          announceCsvTrainResult(status).catch(() => {});
        }
      }
    } catch {
      // Keep watching; transient errors are fine.
    }
  }, 5000);
}

/* ----------------------------------------------------------- tool registry */

const HISTORY_KEY = "neuralForgeChatHistory";
let currentThreadId = null;
let remoteChatThreads = null;
const toolsApi =
  (typeof window !== "undefined" && window.NousTools) || {
    TOOL_REGISTRY: [],
    ROADMAP: [],
    byId() {
      return null;
    },
    statusLabel(status) {
      if (status === "live") return "Live";
      if (status === "in-progress") return "In progress";
      if (status === "coming-soon") return "Coming soon";
      return status || "";
    },
  };

function syncHasChatClass() {
  const log = chatLog();
  const hasUser = Boolean(log && log.querySelector(".message.user"));
  document.body.classList.toggle("has-chat", hasUser);
}

function closeMobileSidebar() {
  document.body.classList.remove("sidebar-open");
  const toggle = document.getElementById("sidebar-toggle");
  if (toggle) toggle.setAttribute("aria-expanded", "false");
  const backdrop = document.getElementById("sidebar-backdrop");
  if (backdrop) backdrop.hidden = true;
}

function renderRegimesReport(report) {
  const empty = document.getElementById("regimes-empty");
  const panel = document.getElementById("regimes-report");
  const summary = document.getElementById("regimes-summary");
  const list = document.getElementById("regimes-list");
  const anomalies = document.getElementById("regimes-anomalies");
  if (!empty || !panel || !summary || !list || !anomalies) return;
  if (!report || !Array.isArray(report.regimes)) {
    empty.hidden = false;
    panel.hidden = true;
    return;
  }
  empty.hidden = true;
  panel.hidden = false;
  summary.textContent = report.summary || "Regime report ready.";
  list.replaceChildren();
  report.regimes.forEach((regime) => {
    const item = document.createElement("li");
    const share = ((Number(regime.share) || 0) * 100).toFixed(0);
    item.textContent = `${regime.label || "regime"} — ${regime.count || 0} samples (${share}%)`;
    list.appendChild(item);
  });
  anomalies.replaceChildren();
  const rows = Array.isArray(report.anomalies) ? report.anomalies : [];
  if (!rows.length) {
    const item = document.createElement("li");
    item.textContent = "No points flagged as far from every centroid.";
    anomalies.appendChild(item);
    return;
  }
  rows.slice(0, 12).forEach((row) => {
    const item = document.createElement("li");
    item.textContent = `Index ${row.index}: distance ${Number(row.distance).toFixed(2)} (cluster ${row.cluster})`;
    anomalies.appendChild(item);
  });
  if (rows.length > 12) {
    const more = document.createElement("li");
    more.textContent = `…and ${rows.length - 12} more.`;
    anomalies.appendChild(more);
  }
}

function renderPcaReport(report) {
  const empty = document.getElementById("pca-empty");
  const panel = document.getElementById("pca-report");
  const summary = document.getElementById("pca-summary");
  const body = document.getElementById("pca-variance-body");
  const groups = document.getElementById("pca-groups");
  if (!empty || !panel || !summary || !body || !groups) return;
  if (!report || !Array.isArray(report.explained_variance_ratio)) {
    empty.hidden = false;
    panel.hidden = true;
    return;
  }
  empty.hidden = true;
  panel.hidden = false;
  summary.textContent = report.summary || "PCA report ready.";
  body.replaceChildren();
  const ratios = report.explained_variance_ratio;
  const cumulative = Array.isArray(report.cumulative_variance_ratio)
    ? report.cumulative_variance_ratio
    : ratios.map((_, index) =>
        ratios.slice(0, index + 1).reduce((sum, value) => sum + value, 0)
      );
  ratios.forEach((ratio, index) => {
    const row = document.createElement("tr");
    const cells = [
      `PC${index + 1}`,
      `${(Number(ratio) * 100).toFixed(1)}%`,
      `${(Number(cumulative[index]) * 100).toFixed(1)}%`,
    ];
    cells.forEach((text) => {
      const cell = document.createElement("td");
      cell.textContent = text;
      row.appendChild(cell);
    });
    body.appendChild(row);
  });
  groups.replaceChildren();
  const groupRows = Array.isArray(report.groups) ? report.groups : [];
  if (!groupRows.length) {
    const item = document.createElement("li");
    item.textContent = "No strong co-moving groups at the default loading threshold.";
    groups.appendChild(item);
    return;
  }
  groupRows.forEach((group) => {
    const item = document.createElement("li");
    const share = ((Number(group.explained_variance_ratio) || 0) * 100).toFixed(1);
    if (group.direction === "opposite") {
      item.textContent =
        `PC${group.component} (${share}%): ` +
        `${(group.features_positive || []).join(", ")} ↔ ` +
        `${(group.features_negative || []).join(", ")}`;
    } else {
      item.textContent =
        `PC${group.component} (${share}%): ${(group.features || []).join(", ")} move together`;
    }
    groups.appendChild(item);
  });
}

function diffusionChart(payload) {
  const starterId = payload.starter_id || "weather";
  const history = Array.isArray(payload.history) ? payload.history : [];
  const steps = Array.isArray(payload.forecast) ? payload.forecast : [];
  const labels = history.map((point) => String(point.index));
  const actual = history.map((point) => chartValue(starterId, point.actual));
  const median = new Array(history.length).fill(null);
  const low = new Array(history.length).fill(null);
  const high = new Array(history.length).fill(null);
  const lastActual = actual.length ? actual[actual.length - 1] : null;
  if (lastActual != null) median[history.length - 1] = lastActual;
  steps.forEach((step, index) => {
    labels.push(`+${step.step || index + 1}`);
    actual.push(null);
    median.push(chartValue(starterId, step.p50));
    low.push(chartValue(starterId, step.p10));
    high.push(chartValue(starterId, step.p90));
  });
  const series = [
    { name: "Recent", color: "#e8eaed", values: actual },
    { name: "Median (p50)", color: "#7cffd6", markers: true, values: median },
  ];
  const paths = Array.isArray(payload.sample_paths) ? payload.sample_paths : [];
  paths.slice(0, 3).forEach((path, pathIndex) => {
    const values = new Array(history.length).fill(null);
    if (lastActual != null) values[history.length - 1] = lastActual;
    path.forEach((value) => values.push(chartValue(starterId, value)));
    series.push({
      name: pathIndex === 0 ? "Sample paths" : `Path ${pathIndex + 1}`,
      color: "#5a6578",
      dashed: true,
      values,
    });
  });
  return {
    ariaLabel: "Diffusion futures with p10 to p90 band",
    labels,
    series,
    bands: steps.length
      ? [{ name: "p10–p90", color: "#7cffd6", low, high }]
      : [],
    empty: "Run the demo to draw a range of futures.",
  };
}

function renderDiffusionReport(payload) {
  const summary = document.getElementById("diffusion-summary");
  const host = document.getElementById("diffusion-chart");
  const status = document.getElementById("diffusion-status");
  if (!summary || !host) return;
  if (!payload || !Array.isArray(payload.forecast)) {
    summary.hidden = true;
    host.hidden = true;
    return;
  }
  if (status) status.hidden = true;
  summary.hidden = false;
  summary.textContent = payload.summary || "Diffusion futures ready.";
  host.hidden = false;
  charts.renderLineChart(host, diffusionChart(payload));
}

async function runDiffusionDemo() {
  const status = document.getElementById("diffusion-status");
  const button = document.getElementById("diffusion-run-btn");
  if (status) {
    status.hidden = false;
    status.textContent = "Drawing futures on the weather sample…";
  }
  if (button) button.disabled = true;
  try {
    const payload = await fetchJson("/starters/weather/diffusion?horizon=7&n_paths=24");
    state.lastDiffusion = payload;
    renderDiffusionReport(payload);
  } catch (error) {
    if (status) {
      status.hidden = false;
      status.textContent = error.message || "Could not run the diffusion demo.";
    }
  } finally {
    if (button) button.disabled = false;
  }
}

// ---- Phase 5: world model simulator ----------------------------------------

const SIM_COLORS = ["#ffb86b", "#8ab4ff", "#ff7aa8"];
const simState = { payload: null, step: 0, timer: null };

function simFormat(value, unit) {
  const number = asNumber(value);
  if (number == null) return "—";
  const text = Math.abs(number) >= 1 ? number.toFixed(1) : String(Number(number.toPrecision(3)));
  if (!unit) return text;
  return unit === "°F" || unit === "°C" ? `${text}${unit}` : `${text} ${unit}`;
}

function simChart(payload, upTo) {
  // Keep the last ~5 weeks so the rolled-forward steps stay readable.
  const history = Array.isArray(payload.history) ? payload.history.slice(-36) : [];
  const base = payload.baseline || {};
  const horizon = Array.isArray(base.p50) ? base.p50.length : 0;
  const shown = Math.max(1, Math.min(upTo || horizon, horizon));
  const labels = history.map((point) => point.date || String(point.index));
  const actual = history.map((point) => asNumber(point.value));
  const pad = () => new Array(history.length).fill(null);
  const lastActual = actual.length ? actual[actual.length - 1] : null;
  const future = (list) => {
    const values = pad();
    if (lastActual != null) values[history.length - 1] = lastActual;
    for (let i = 0; i < horizon; i += 1) {
      values.push(i < shown && Array.isArray(list) ? asNumber(list[i]) : null);
    }
    return values;
  };
  for (let i = 1; i <= horizon; i += 1) {
    labels.push(`+${i}`);
    actual.push(null);
  }
  const low = pad();
  const high = pad();
  for (let i = 0; i < horizon; i += 1) {
    low.push(i < shown ? asNumber(base.p10 && base.p10[i]) : null);
    high.push(i < shown ? asNumber(base.p90 && base.p90[i]) : null);
  }
  const series = [
    { name: "Recent", color: "#e8eaed", values: actual },
    { name: "Baseline (median)", color: "#7cffd6", markers: true, values: future(base.p50) },
  ];
  (payload.scenarios || []).forEach((scenario, index) => {
    series.push({
      name: scenario.name || `Scenario ${index + 1}`,
      color: SIM_COLORS[index % SIM_COLORS.length],
      dashed: true,
      values: future(scenario.p50),
    });
  });
  return {
    ariaLabel: "World model simulation: baseline band and scenario medians",
    labels,
    series,
    bands: horizon ? [{ name: "Baseline 80% range", color: "#7cffd6", low, high }] : [],
    empty: "Run a simulation to roll the world forward.",
  };
}

function renderSimStep(step) {
  const payload = simState.payload;
  if (!payload) return;
  const base = payload.baseline || {};
  const horizon = Array.isArray(base.p50) ? base.p50.length : 0;
  const k = Math.max(1, Math.min(step, horizon));
  simState.step = k;
  const unit = payload.unit || "";
  const host = document.getElementById("sim-chart");
  const label = document.getElementById("sim-step-label");
  const slider = document.getElementById("sim-step");
  const readout = document.getElementById("sim-readout");
  if (host) charts.renderLineChart(host, simChart(payload, k));
  if (label) label.textContent = `${k} of ${horizon}`;
  if (slider) slider.value = String(k);
  if (!readout) return;
  readout.replaceChildren();
  const i = k - 1;
  const regime = Array.isArray(base.regime) ? base.regime[i] : null;
  const baseItem = document.createElement("li");
  baseItem.textContent =
    `Baseline: ${simFormat(base.p50 && base.p50[i], unit)} ` +
    `(80% range ${simFormat(base.p10 && base.p10[i], unit)} to ${simFormat(base.p90 && base.p90[i], unit)})` +
    (regime ? ` · regime: ${regime}` : "");
  readout.appendChild(baseItem);
  (payload.scenarios || []).forEach((scenario) => {
    const item = document.createElement("li");
    const delta = asNumber(scenario.delta_p50 && scenario.delta_p50[i]);
    const sign = delta != null && delta > 0 ? "+" : "";
    item.textContent =
      `${scenario.name}: ${simFormat(scenario.p50 && scenario.p50[i], unit)} ` +
      `(${simFormat(scenario.p10 && scenario.p10[i], unit)} to ${simFormat(scenario.p90 && scenario.p90[i], unit)}), ` +
      `${delta == null ? "" : `${sign}${simFormat(delta, unit)} vs baseline`}`;
    readout.appendChild(item);
  });
}

function stopSimPlayback() {
  if (simState.timer) {
    clearInterval(simState.timer);
    simState.timer = null;
  }
  const play = document.getElementById("sim-play");
  if (play) play.textContent = "▶ Play";
}

function toggleSimPlayback() {
  const payload = simState.payload;
  if (!payload) return;
  if (simState.timer) {
    stopSimPlayback();
    return;
  }
  const horizon = (payload.baseline && payload.baseline.p50 || []).length;
  let step = simState.step >= horizon ? 1 : simState.step + 1;
  const play = document.getElementById("sim-play");
  if (play) play.textContent = "❚❚ Pause";
  renderSimStep(step);
  simState.timer = setInterval(() => {
    step += 1;
    if (step > horizon) {
      stopSimPlayback();
      return;
    }
    renderSimStep(step);
  }, 450);
}

function renderSimulation(payload) {
  const result = document.getElementById("sim-result");
  const summary = document.getElementById("sim-summary");
  const model = document.getElementById("sim-model");
  const slider = document.getElementById("sim-step");
  if (!result || !payload || !payload.baseline) return;
  stopSimPlayback();
  simState.payload = payload;
  result.hidden = false;
  if (summary) summary.textContent = payload.summary || "Simulation ready.";
  const horizon = (payload.baseline.p50 || []).length;
  if (slider) {
    slider.max = String(Math.max(1, horizon));
  }
  if (model) {
    const info = payload.model || {};
    const regimes = Array.isArray(info.regimes)
      ? info.regimes.map((r) => `${r.label} ${Math.round(Number(r.share) * 100)}%`).join(", ")
      : "";
    const bt = payload.backtest || {};
    const trust = bt.holdout
      ? bt.beats_persistence
        ? "It beat the persistence baseline on held-out data."
        : "It did not beat persistence on held-out data here, so treat runs as scenarios, not forecasts."
      : "";
    model.textContent =
      `${info.description || ""} ${regimes ? `Regimes: ${regimes}.` : ""} ${trust}`.trim();
  }
  renderSimStep(horizon);
}

function parseSimValues(text) {
  return String(text || "")
    .split(/[\s,;]+/)
    .map((part) => Number(part.replace(/[^0-9eE+.-]/g, "")))
    .filter((value) => Number.isFinite(value));
}

function simScenarioFromForm(form, suffix) {
  const kind = form[`kind${suffix}`] ? form[`kind${suffix}`].value : "";
  if (!kind) return null;
  const value = Number(form[`value${suffix}`].value);
  const step = Number(form[`step${suffix}`].value) || 1;
  const name = (form[`name${suffix}`].value || `Scenario ${suffix}`).trim().slice(0, 40);
  return {
    name,
    events: [{ kind, step, value: Number.isFinite(value) ? value : 0, mode: "abs" }],
  };
}

function syncSimForm() {
  const form = document.getElementById("sim-form");
  if (!form) return;
  const series = form.series.value;
  const custom = document.getElementById("sim-custom");
  if (custom) custom.hidden = series !== "custom";
  const hint = document.getElementById("sim-unit-hint");
  if (hint) {
    hint.textContent =
      series === "weather"
        ? "Weather values are °F. Volatility × multiplies the day-to-day noise."
        : "Values are in the series’ own units. Volatility × multiplies the noise.";
  }
}

async function submitSimForm(event) {
  if (event) event.preventDefault();
  const form = document.getElementById("sim-form");
  const status = document.getElementById("sim-status");
  const button = document.getElementById("sim-run");
  if (!form) return;
  const series = form.series.value || "weather";
  const horizon = Math.min(60, Math.max(1, Math.floor(Number(form.horizon.value) || 14)));
  const scenarios = ["A", "B"]
    .map((suffix) => simScenarioFromForm(form, suffix))
    .filter(Boolean)
    .map((scenario) => ({
      ...scenario,
      events: scenario.events.map((ev) => ({ ...ev, step: Math.min(horizon, Math.max(1, ev.step)) })),
    }));
  const body = { horizon, n_paths: 128, scenarios };
  if (series === "custom") {
    const values = parseSimValues(form.values.value);
    if (values.length < 24) {
      if (status) {
        status.hidden = false;
        status.textContent = `Paste at least 24 numbers (you have ${values.length}).`;
      }
      return;
    }
    body.values = values.slice(-2000);
    const unit = (form.unit.value || "").trim();
    if (unit) body.unit = unit.slice(0, 32);
  } else {
    body.starter_id = series;
    if (series === "weather") body.display_unit = "°F";
  }
  if (status) {
    status.hidden = false;
    status.textContent = "Rolling 128 futures forward…";
  }
  if (button) button.disabled = true;
  try {
    const payload = await fetchJson("/analysis/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (status) status.hidden = true;
    renderSimulation(payload);
  } catch (error) {
    if (status) {
      status.hidden = false;
      status.textContent = error.message || "Could not run the simulation.";
    }
  } finally {
    if (button) button.disabled = false;
  }
}

function openSimulatorWithValues(values, unit) {
  const form = document.getElementById("sim-form");
  if (!form) return;
  form.series.value = "custom";
  form.values.value = values.join(", ");
  form.unit.value = unit || "";
  syncSimForm();
  showToolView("simulator", { id: "simulator", name: "Simulator" });
  void submitSimForm();
}

function isWeatherPayload(payload) {
  return Boolean(payload) && (payload.starter_id === "weather" || payload.unit === "°C");
}

/** JEPA MAE is a °C delta for weather: scale to °F (no +32 offset). */
function jepaMaeText(payload, mae) {
  const number = asNumber(mae);
  if (number == null) return "—";
  if (isWeatherPayload(payload)) return `${formatWeatherMaeFahrenheit(number)}°F`;
  return number.toFixed(4);
}

function jepaSummaryText(payload) {
  if (!isWeatherPayload(payload) || !Array.isArray(payload.models)) {
    return payload.summary || "JEPA compare ready.";
  }
  const parts = payload.models
    .map((row) => {
      const mae = jepaMaeText(payload, row.mae);
      return mae === "—" ? null : `${row.label || row.name || "model"} MAE ${mae}`;
    })
    .filter(Boolean);
  const name = payload.name ? `the “${payload.name}” sample` : "the weather sample";
  return parts.length
    ? `On ${name} holdout: ${parts.join(", ")}.`
    : payload.summary || "JEPA compare ready.";
}

function renderJepaReport(payload) {
  const summary = document.getElementById("jepa-summary");
  const report = document.getElementById("jepa-report");
  const body = document.getElementById("jepa-compare-body");
  const status = document.getElementById("jepa-status");
  if (!summary || !report || !body) return;
  if (!payload || !Array.isArray(payload.models)) {
    summary.hidden = true;
    report.hidden = true;
    return;
  }
  if (status) status.hidden = true;
  summary.hidden = false;
  summary.textContent = jepaSummaryText(payload);
  report.hidden = false;
  body.replaceChildren();
  payload.models.forEach((row) => {
    const tr = document.createElement("tr");
    const name = document.createElement("td");
    name.textContent = row.label || row.name || "model";
    const maeCell = document.createElement("td");
    maeCell.textContent = jepaMaeText(payload, row.mae);
    tr.append(name, maeCell);
    body.appendChild(tr);
  });
}

async function runJepaDemo() {
  const status = document.getElementById("jepa-status");
  const button = document.getElementById("jepa-run-btn");
  if (status) {
    status.hidden = false;
    status.textContent = "Pretraining JEPA and comparing on the weather sample…";
  }
  if (button) button.disabled = true;
  try {
    const payload = await fetchJson("/starters/weather/jepa-compare");
    state.lastJepa = payload;
    renderJepaReport(payload);
  } catch (error) {
    if (status) {
      status.hidden = false;
      status.textContent = error.message || "Could not run the JEPA compare.";
    }
  } finally {
    if (button) button.disabled = false;
  }
}

function showToolView(viewId, toolMeta) {
  const stage = document.getElementById("main-stage");
  if (!stage) return;
  const next = viewId || "chat";
  stage.dataset.view = next;
  document.querySelectorAll("[data-tool-view]").forEach((panel) => {
    panel.hidden = panel.getAttribute("data-tool-view") !== next;
  });
  if (next === "coming-soon" && toolMeta) {
    const title = document.getElementById("coming-soon-title");
    const body = document.getElementById("coming-soon-body");
    if (title) title.textContent = toolMeta.name || "Coming soon";
    if (body) {
      body.textContent =
        (toolMeta.description || "This tool is on the roadmap.") +
        " Chat and the live tools stay available meanwhile.";
    }
  }
  if (next === "pca") {
    renderPcaReport(state.lastPca);
  }
  if (next === "regimes") {
    renderRegimesReport(state.lastRegimes);
  }
  if (next === "diffusion") {
    renderDiffusionReport(state.lastDiffusion);
  }
  if (next === "jepa") {
    renderJepaReport(state.lastJepa);
  }
  if (next !== "simulator") stopSimPlayback();
  document.querySelectorAll(".tool-nav-btn").forEach((button) => {
    const current = button.dataset.toolId === (toolMeta && toolMeta.id);
    button.setAttribute("aria-current", current || (next === "chat" && button.dataset.toolId === "chat") ? "page" : "false");
  });
  if (next === "chat") {
    document.querySelectorAll("[data-tool-view]").forEach((panel) => {
      panel.hidden = true;
    });
  }
  closeMobileSidebar();
  syncHasChatClass();
}

async function loadScenarioPanel() {
  const list = document.getElementById("scenario-list");
  const empty = document.getElementById("scenario-empty");
  const note = document.getElementById("scenario-note");
  if (!list) return;
  list.replaceChildren();
  try {
    const data = await fetchJson("/agent/scenarios");
    const scenarios = Array.isArray(data.scenarios) ? data.scenarios : [];
    if (!scenarios.length) {
      if (empty) empty.hidden = false;
      return;
    }
    if (empty) empty.hidden = true;
    for (const scenario of scenarios) {
      const row = document.createElement("li");
      const label = document.createElement("span");
      const seriesBit = scenario.series
        ? ` · ${scenario.series}${scenario.horizon ? ` h${scenario.horizon}` : ""}`
        : "";
      label.textContent = `${scenario.name}${seriesBit}`;
      const run = document.createElement("button");
      run.type = "button";
      run.className = "btn secondary btn-small";
      run.textContent = "Run";
      run.addEventListener("click", () => {
        void runSavedScenario(scenario.id);
      });
      const whatIf = document.createElement("button");
      whatIf.type = "button";
      whatIf.className = "btn secondary btn-small";
      whatIf.textContent = "What-if −10%";
      whatIf.disabled = !scenario.series;
      whatIf.addEventListener("click", () => {
        void runScenarioWhatIf(scenario).catch((error) => {
          if (note) note.textContent = error.message || "What-if failed.";
        });
      });
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "btn secondary btn-small";
      remove.textContent = "Delete";
      remove.addEventListener("click", async () => {
        try {
          await fetchJson(`/agent/scenarios/${encodeURIComponent(scenario.id)}`, {
            method: "DELETE",
          });
          await loadScenarioPanel();
        } catch (error) {
          if (note) note.textContent = error.message || "Could not delete scenario.";
        }
      });
      row.append(label, run, whatIf, remove);
      list.append(row);
    }
  } catch (error) {
    if (empty) empty.hidden = true;
    if (note) note.textContent = error.message || "Could not load scenarios.";
  }
}

async function runSavedScenario(scenarioId) {
  const note = document.getElementById("scenario-note");
  try {
    const data = await fetchJson(`/agent/scenarios/${encodeURIComponent(scenarioId)}`);
    const question =
      (data && typeof data.rerun_question === "string" && data.rerun_question.trim()) ||
      (data && data.scenario && typeof data.scenario.question === "string" && data.scenario.question.trim()) ||
      "";
    if (!question) {
      if (note) note.textContent = "Scenario has no question to run.";
      return;
    }
    showToolView("chat");
    rememberChatTurn(question);
    await handleQuestion(question);
    syncHasChatClass();
  } catch (error) {
    if (note) note.textContent = error.message || "Could not run scenario.";
  }
}

function weatherWhatIfDisplay(result) {
  if (!result || typeof result !== "object") return result;
  const c2f = (c) =>
    typeof c === "number" && Number.isFinite(c)
      ? Math.round((c * (9 / 5) + 32) * 100) / 100
      : c;
  const d2f = (c) =>
    typeof c === "number" && Number.isFinite(c)
      ? Math.round(c * (9 / 5) * 100) / 100
      : c;
  const mapPath = (block) => {
    if (!block || typeof block !== "object") return block;
    return {
      ...block,
      p50: Array.isArray(block.p50) ? block.p50.map(c2f) : block.p50,
      forecast: Array.isArray(block.forecast) ? block.forecast.map(c2f) : block.forecast,
    };
  };
  return {
    ...result,
    unit: "°F",
    storage_unit: "°C",
    shock: result.shock
      ? { ...result.shock, before: c2f(result.shock.before), after: c2f(result.shock.after) }
      : result.shock,
    baseline: mapPath(result.baseline),
    shocked: mapPath(result.shocked),
    delta: result.delta
      ? {
          ...result.delta,
          mean: d2f(result.delta.mean),
          p50: Array.isArray(result.delta.p50) ? result.delta.p50.map(d2f) : result.delta.p50,
          text: formatWhatIfSummary(result, "weather"),
        }
      : result.delta,
    summary: formatWhatIfSummary(result, "weather"),
  };
}

function formatWhatIfSummary(result, series) {
  const raw =
    (result && typeof result.summary === "string" && result.summary) ||
    `What-if on ${series || "series"} finished.`;
  if (!result || String(series || result.starter_id || "").toLowerCase() !== "weather") {
    return raw;
  }
  // API stores weather in °C; chat forecast replies use °F — match that here.
  const meanC = result.delta && typeof result.delta.mean === "number" ? result.delta.mean : null;
  if (meanC == null || !Number.isFinite(meanC)) {
    return `${raw} (weather values shown in chat as °F)`;
  }
  const meanF = meanC * (9 / 5);
  const abs = Math.abs(meanF);
  const approx = abs < 1e-9 ? "0" : String(Math.round(abs * 1000) / 1000);
  if (/leaves the .+ forecast almost unchanged/i.test(raw)) {
    return raw.replace(/\.$/, "") + " (°F).";
  }
  if (/lifts the /i.test(raw)) {
    return raw.replace(
      /lifts the (.+?) by about [-+0-9.eE]+(?:\s*\S+)? on average\.?/i,
      `lifts the $1 by about ${approx}°F on average.`
    );
  }
  if (/lowers the /i.test(raw)) {
    return raw.replace(
      /lowers the (.+?) by about [-+0-9.eE]+(?:\s*\S+)? on average\.?/i,
      `lowers the $1 by about ${approx}°F on average.`
    );
  }
  return `${raw} (Δ ≈ ${approx}°F)`;
}

async function runScenarioWhatIf(scenario) {
  const series = scenario && scenario.series ? String(scenario.series) : "";
  if (!series) throw new Error("Save a series on the scenario first.");
  const starters = new Set(["weather", "stocks", "crypto", "lunar", "near-earth"]);
  const body = {
    mode: "pct",
    value: -10,
    horizon: scenario.horizon || 1,
  };
  let result;
  if (starters.has(series.toLowerCase())) {
    result = await fetchJson("/analysis/what-if", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ starter_id: series.toLowerCase(), ...body }),
    });
  } else {
    result = await fetchJson(`/workspace/models/${encodeURIComponent(series)}/what-if`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  }
  const text = formatWhatIfSummary(result, series);
  showToolView("chat");
  addReply(`What-if on “${scenario.name || series}” (−10% last input):\n\n${text}`);
}

async function submitWhatIfForm(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const output = document.getElementById("what-if-output");
  const series = form.series.value;
  const value = Number(form.value.value);
  try {
    const result = await fetchJson("/analysis/what-if", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        starter_id: series,
        mode: "pct",
        value: Number.isFinite(value) ? value : -10,
      }),
    });
    if (output) {
      output.hidden = false;
      output.textContent = JSON.stringify(
        series === "weather" ? weatherWhatIfDisplay(result) : result,
        null,
        2
      );
    }
    const summary = formatWhatIfSummary(result, series);
    showToolView("chat");
    addReply(summary);
  } catch (error) {
    if (output) {
      output.hidden = false;
      output.textContent = error.message || "What-if failed.";
    }
  }
}

async function loadMemoryPanel() {
  const list = document.getElementById("memory-list");
  const empty = document.getElementById("memory-empty");
  const note = document.getElementById("memory-scope-note");
  if (!list) return;
  list.replaceChildren();
  try {
    const data = await fetchJson("/agent/memory");
    const facts = Array.isArray(data.facts) ? data.facts : [];
    if (note) {
      const scope = data.scope === "owner" ? "owner account" : data.scope === "user" ? "signed-in account" : "this visitor";
      note.textContent = `Showing memory for your ${scope}.`;
    }
    if (!facts.length) {
      if (empty) empty.hidden = false;
    } else {
      if (empty) empty.hidden = true;
      for (const fact of facts) {
        const row = document.createElement("li");
        const label = document.createElement("span");
        label.textContent = `${fact.key}: ${fact.value}`;
        const forget = document.createElement("button");
        forget.type = "button";
        forget.className = "btn secondary btn-small";
        forget.textContent = "Forget";
        forget.addEventListener("click", async () => {
          try {
            await fetchJson(`/agent/memory/${encodeURIComponent(fact.key)}`, { method: "DELETE" });
            await loadMemoryPanel();
          } catch (error) {
            if (note) note.textContent = error.message || "Could not forget that fact.";
          }
        });
        row.append(label, forget);
        list.append(row);
      }
    }
  } catch (error) {
    if (empty) empty.hidden = true;
    if (note) note.textContent = error.message || "Could not load memory.";
  }
  await loadScenarioPanel();
}

async function clearAllMemory() {
  const note = document.getElementById("memory-scope-note");
  try {
    await fetchJson("/agent/memory", { method: "DELETE" });
    await loadMemoryPanel();
    if (note) note.textContent = "Memory cleared.";
  } catch (error) {
    if (note) note.textContent = error.message || "Could not clear memory.";
  }
}

function activateTool(tool) {
  if (!tool) return;
  if (tool.href) {
    window.location.assign(tool.href);
    return;
  }
  if (tool.id === "talk") {
    showToolView("voice", tool);
    const talkBtn = document.getElementById("talk-toggle");
    if (talkBtn && !talk.active) talkBtn.focus();
    return;
  }
  const hasPanel =
    tool.view && document.querySelector(`[data-tool-view="${tool.view}"]`) !== null;
  if (tool.status === "coming-soon" || (tool.status === "in-progress" && !hasPanel)) {
    showToolView("coming-soon", tool);
    return;
  }
  showToolView(tool.view || "chat", tool);
  if (tool.id === "multimodal") {
    void loadMultimodalAllowance();
  }
  if (tool.id === "memory") {
    void loadMemoryPanel();
  }
  if (tool.id === "reports") {
    void loadReportsPanel();
  }
  if (tool.id === "alerts") {
    void loadAlertsPanel();
  }
}

let lastSyntheticCsv = "";
let lastSyntheticName = "synthetic.csv";

async function submitSyntheticForm(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const note = document.getElementById("synthetic-note");
  const preview = document.getElementById("synthetic-preview");
  const download = document.getElementById("synthetic-download");
  const train = document.getElementById("synthetic-train");
  const pattern = form.pattern.value || "sine";
  const length = Number(form.length.value) || 256;
  const noise = Number(form.noise.value);
  try {
    const data = await fetchJson("/analysis/synthetic", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        pattern,
        length,
        noise: Number.isFinite(noise) ? noise : 0.05,
        seed: 0,
      }),
    });
    lastSyntheticCsv = typeof data.csv === "string" ? data.csv : "";
    lastSyntheticName = `synthetic-${pattern}.csv`;
    if (note) note.textContent = data.summary || "Series generated.";
    if (preview) {
      const rows = Array.isArray(data.series) ? data.series.slice(0, 8) : [];
      preview.hidden = false;
      preview.textContent = JSON.stringify({ pattern: data.pattern, preview: rows }, null, 2);
    }
    if (download) download.hidden = !lastSyntheticCsv;
    if (train) train.hidden = !lastSyntheticCsv;
  } catch (error) {
    if (note) note.textContent = error.message || "Could not generate series.";
  }
}

function downloadSyntheticCsv() {
  if (!lastSyntheticCsv) return;
  const blob = new Blob([lastSyntheticCsv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = lastSyntheticName;
  anchor.click();
  URL.revokeObjectURL(url);
}

async function trainFromSyntheticCsv() {
  if (!lastSyntheticCsv) return;
  const note = document.getElementById("synthetic-note");
  const file = new File([lastSyntheticCsv], lastSyntheticName, { type: "text/csv" });
  try {
    showToolView("chat");
    await handleCsvTrainFile(file);
  } catch (error) {
    if (note) note.textContent = error.message || "Could not start train from synthetic CSV.";
  }
}

async function loadAlertsPanel() {
  const ruleList = document.getElementById("alert-rule-list");
  const eventList = document.getElementById("alert-event-list");
  const note = document.getElementById("alert-note");
  if (!ruleList || !eventList) return;
  ruleList.replaceChildren();
  eventList.replaceChildren();
  try {
    const data = await fetchJson("/agent/alerts");
    const rules = Array.isArray(data.rules) ? data.rules : [];
    const events = Array.isArray(data.events) ? data.events : [];
    if (!rules.length) {
      appendTextItem(ruleList, "No alert rules yet.");
    }
    for (const rule of rules) {
      const row = document.createElement("li");
      const label = document.createElement("span");
      label.textContent = `${rule.name} · ${rule.kind} · ${rule.series}`;
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "btn secondary btn-small";
      remove.textContent = "Delete";
      remove.addEventListener("click", async () => {
        try {
          await fetchJson(`/agent/alerts/${encodeURIComponent(rule.id)}`, {
            method: "DELETE",
          });
          await loadAlertsPanel();
        } catch (error) {
          if (note) note.textContent = error.message || "Could not delete alert.";
        }
      });
      row.append(label, remove);
      ruleList.append(row);
    }
    if (!events.length) {
      appendTextItem(eventList, "No fired alerts yet.");
    }
    for (const event of events) {
      const message =
        event.payload && typeof event.payload.message === "string"
          ? event.payload.message
          : JSON.stringify(event.payload || {});
      appendTextItem(
        eventList,
        `${event.acknowledged ? "✓ " : ""}${message}`
      );
    }
  } catch (error) {
    if (note) note.textContent = error.message || "Could not load alerts.";
  }
}

async function submitAlertForm(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const note = document.getElementById("alert-note");
  const kind = form.kind.value === "anomaly" ? "anomaly" : "threshold";
  const body = {
    name: form.name.value.trim(),
    series: form.series.value,
    kind,
    config:
      kind === "threshold"
        ? { op: form.op.value, value: Number(form.value.value) }
        : {},
  };
  try {
    await fetchJson("/agent/alerts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    form.name.value = "";
    if (note) note.textContent = "Alert saved.";
    await loadAlertsPanel();
  } catch (error) {
    if (note) note.textContent = error.message || "Could not save alert.";
  }
}

async function checkAlertsNow() {
  const note = document.getElementById("alert-note");
  try {
    const data = await fetchJson("/agent/alerts/check", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    const fired = Array.isArray(data.fired) ? data.fired : [];
    if (note) {
      note.textContent = fired.length
        ? `Fired ${fired.length} alert(s).`
        : "No alerts fired.";
    }
    if (fired.length) {
      const lines = fired.map((row) => {
        const message =
          row.result && typeof row.result.message === "string"
            ? row.result.message
            : "Alert fired.";
        const name = row.rule && row.rule.name ? row.rule.name : "alert";
        return `${name}: ${message}`;
      });
      addReply(lines.join("\n\n"));
    }
    await loadAlertsPanel();
  } catch (error) {
    if (note) note.textContent = error.message || "Check failed.";
  }
}

async function createShareableReport(series) {
  const note = document.getElementById("report-note");
  const data = await fetchJson("/agent/reports", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ series }),
  });
  const path =
    (data && data.report && data.report.share_path) ||
    (data && data.report && data.report.id
      ? `/agent/reports/${data.report.id}`
      : "");
  if (!path) throw new Error("Report was created without a share path.");
  const url = new URL(path, window.location.origin).href;
  try {
    await navigator.clipboard.writeText(url);
    if (note) note.textContent = `Copied share link: ${url}`;
  } catch {
    if (note) note.textContent = `Share link: ${url}`;
  }
  addReply(`Shareable report for ${series}: ${url}`);
  return data;
}

async function loadReportsPanel() {
  const list = document.getElementById("report-list");
  const note = document.getElementById("report-note");
  if (!list) return;
  list.replaceChildren();
  try {
    const data = await fetchJson("/agent/reports");
    const reports = Array.isArray(data.reports) ? data.reports : [];
    if (!reports.length) {
      if (note && !note.textContent) note.textContent = "No reports yet — create one above.";
      return;
    }
    for (const report of reports) {
      const row = document.createElement("li");
      const label = document.createElement("span");
      label.textContent = `${report.title || report.series} · ${report.id}`;
      const open = document.createElement("button");
      open.type = "button";
      open.className = "btn secondary btn-small";
      open.textContent = "Open";
      open.addEventListener("click", () => {
        void openShareableReport(report.id);
      });
      row.append(label, open);
      list.append(row);
    }
  } catch (error) {
    if (note) note.textContent = error.message || "Could not load reports.";
  }
}

async function openShareableReport(reportId) {
  const title = document.getElementById("report-view-title");
  const body = document.getElementById("report-view-body");
  const panel = document.getElementById("report-view");
  const data = await fetchJson(`/agent/reports/${encodeURIComponent(reportId)}`);
  if (title) title.textContent = data.title || data.series || reportId;
  if (body) {
    const snap = data.snapshot || data;
    const lines = [];
    if (snap.forecast && snap.forecast.why && snap.forecast.why.explanation) {
      lines.push(`Why: ${snap.forecast.why.explanation}`);
    }
    if (snap.accuracy && typeof snap.accuracy.comparison === "string") {
      lines.push(snap.accuracy.comparison);
    }
    if (snap.regimes && typeof snap.regimes.summary === "string") {
      lines.push(snap.regimes.summary);
    }
    body.textContent = lines.length ? lines.join("\n\n") : JSON.stringify(snap, null, 2);
  }
  if (panel) panel.hidden = false;
}

async function submitReportForm(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const series = form.series.value || "weather";
  try {
    await createShareableReport(series);
    await loadReportsPanel();
  } catch (error) {
    const note = document.getElementById("report-note");
    if (note) note.textContent = error.message || "Could not create report.";
  }
}

async function saveScenarioFromForm(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const note = document.getElementById("scenario-note");
  const name = form.name.value.trim();
  const question = form.question.value.trim();
  const series = form.series.value.trim();
  try {
    const body = { name, question };
    if (series) body.series = series;
    await fetchJson("/agent/scenarios", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    form.reset();
    if (note) note.textContent = "Scenario saved.";
    await loadScenarioPanel();
  } catch (error) {
    if (note) note.textContent = error.message || "Could not save scenario.";
  }
}

function toolStatusBadgeText(status) {
  if (typeof toolsApi.statusLabel === "function") {
    return toolsApi.statusLabel(status);
  }
  if (status === "live") return "Live";
  if (status === "in-progress") return "In progress";
  if (status === "coming-soon") return "Coming soon";
  return status || "";
}

const PINNED_TOOL_IDS = ["chat", "forecast", "simulator", "train", "live", "talk"];

function toolNavButton(tool) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "tool-nav-btn";
  button.dataset.toolId = tool.id;
  const icon = document.createElement("span");
  icon.className = "tool-nav-icon";
  icon.setAttribute("aria-hidden", "true");
  icon.textContent = tool.icon || "·";
  const meta = document.createElement("span");
  meta.className = "tool-nav-meta";
  const name = document.createElement("span");
  name.className = "tool-nav-name";
  name.textContent = tool.name;
  meta.append(name);
  if (tool.status === "coming-soon" || tool.status === "in-progress") {
    const status = document.createElement("span");
    status.className = "tool-nav-status";
    status.textContent = toolStatusBadgeText(tool.status);
    meta.append(status);
  }
  button.append(icon, meta);
  button.addEventListener("click", () => activateTool(tool));
  return button;
}

function renderToolNav() {
  const nav = document.getElementById("tool-nav");
  if (!nav || !toolsApi.TOOL_REGISTRY) return;
  nav.replaceChildren();
  const label = document.createElement("p");
  label.className = "sidebar-section-label";
  label.textContent = "Tools";
  nav.append(label);
  const pinned = [];
  const others = [];
  for (const tool of toolsApi.TOOL_REGISTRY) {
    (PINNED_TOOL_IDS.includes(tool.id) ? pinned : others).push(tool);
  }
  pinned.sort((a, b) => PINNED_TOOL_IDS.indexOf(a.id) - PINNED_TOOL_IDS.indexOf(b.id));
  pinned.forEach((tool) => nav.append(toolNavButton(tool)));
  if (others.length) {
    const more = document.createElement("details");
    more.className = "tool-nav-more";
    const summary = document.createElement("summary");
    summary.textContent = `All tools (${others.length})`;
    more.append(summary);
    others.forEach((tool) => more.append(toolNavButton(tool)));
    nav.append(more);
  }
}

function renderToolsMenu() {
  const menu = document.getElementById("tools-menu");
  if (!menu || !toolsApi.TOOL_REGISTRY) return;
  menu.replaceChildren();
  for (const tool of toolsApi.TOOL_REGISTRY) {
    const item = document.createElement("button");
    item.type = "button";
    item.className = "tools-menu-item";
    item.setAttribute("role", "menuitem");
    const title = document.createElement("strong");
    title.textContent = `${tool.icon || ""} ${tool.name}`.trim();
    const desc = document.createElement("span");
    desc.className = "muted";
    const pending =
      tool.status === "coming-soon" || tool.status === "in-progress";
    desc.textContent = pending
      ? `${toolStatusBadgeText(tool.status)} — ${tool.description || ""}`
      : tool.description || "";
    item.append(title, desc);
    item.addEventListener("click", () => {
      menu.hidden = true;
      const toggle = document.getElementById("tools-menu-btn");
      if (toggle) toggle.setAttribute("aria-expanded", "false");
      activateTool(tool);
    });
    menu.append(item);
  }
}

function renderRoadmap() {
  const root = document.getElementById("tools-roadmap-list");
  const phases = toolsApi.ROADMAP;
  if (!root || !Array.isArray(phases)) return;
  root.replaceChildren();
  for (const phase of phases) {
    const card = document.createElement("article");
    card.className = "roadmap-phase";
    card.dataset.status = phase.status || "coming-soon";

    const head = document.createElement("header");
    head.className = "roadmap-phase-head";
    const title = document.createElement("h3");
    title.className = "roadmap-phase-title";
    title.textContent = phase.title || "";
    const badge = document.createElement("span");
    badge.className = "roadmap-status";
    badge.dataset.status = phase.status || "coming-soon";
    badge.textContent = toolStatusBadgeText(phase.status);
    head.append(title, badge);

    const list = document.createElement("ul");
    list.className = "roadmap-item-list";
    for (const item of phase.items || []) {
      const li = document.createElement("li");
      li.className = "roadmap-item";
      const name = document.createElement("strong");
      name.textContent = item.name || "";
      const desc = document.createElement("span");
      desc.textContent = item.description || "";
      li.append(name, desc);
      if (item.toolId) {
        const tool = toolsApi.byId(item.toolId);
        if (tool) {
          li.classList.add("roadmap-item-link");
          li.tabIndex = 0;
          li.setAttribute("role", "link");
          li.addEventListener("click", () => activateTool(tool));
          li.addEventListener("keydown", (event) => {
            if (event.key === "Enter" || event.key === " ") {
              event.preventDefault();
              activateTool(tool);
            }
          });
        }
      }
      list.append(li);
    }

    card.append(head, list);
    root.append(card);
  }
}

function historyBucket(iso) {
  const when = new Date(iso);
  if (Number.isNaN(when.getTime())) return "Previous 7 days";
  const now = new Date();
  const startToday = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const startYesterday = new Date(startToday);
  startYesterday.setDate(startYesterday.getDate() - 1);
  const weekAgo = new Date(startToday);
  weekAgo.setDate(weekAgo.getDate() - 7);
  if (when >= startToday) return "Today";
  if (when >= startYesterday) return "Yesterday";
  if (when >= weekAgo) return "Previous 7 days";
  return "Older";
}

function loadChatHistory() {
  try {
    const raw = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
    return Array.isArray(raw) ? raw : [];
  } catch {
    return [];
  }
}

function saveChatHistory(entries) {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(entries.slice(0, 40)));
  } catch {
    // History is best-effort when storage is blocked.
  }
}

function rememberChatTurn(question) {
  const title = String(question || "").replace(/\s+/g, " ").trim().slice(0, 80);
  if (!title) return;
  const entries = loadChatHistory().filter((row) => row && row.title !== title);
  entries.unshift({
    title,
    at: new Date().toISOString(),
    id: currentThreadId || null,
  });
  saveChatHistory(entries);
  renderChatHistory();
}

async function refreshRemoteChatHistory() {
  try {
    const data = await fetchJson("/agent/chats");
    const threads = Array.isArray(data.threads) ? data.threads : [];
    remoteChatThreads = threads.map((row) => ({
      id: row.id,
      title: row.title,
      at: row.updated_at || row.created_at || new Date().toISOString(),
    }));
    saveChatHistory(remoteChatThreads);
    const search = document.getElementById("history-search");
    renderChatHistory(search ? search.value : "");
  } catch {
    remoteChatThreads = null;
    const search = document.getElementById("history-search");
    renderChatHistory(search ? search.value : "");
  }
}

async function openSavedChat(entry) {
  showToolView("chat");
  if (!entry || !entry.id) {
    if (entry && entry.title) await handleQuestion(entry.title);
    return;
  }
  try {
    const data = await fetchJson(`/agent/chats/${encodeURIComponent(entry.id)}`);
    const messages = Array.isArray(data.messages) ? data.messages : [];
    const log = chatLog();
    if (log) log.replaceChildren();
    currentThreadId = entry.id;
    setChatMode("normal");
    for (const message of messages) {
      if (!message || (message.role !== "user" && message.role !== "assistant")) continue;
      const text = String(message.content || "").trim();
      if (!text) continue;
      if (message.role === "user") addMessage("user", text);
      else addReply(text);
    }
    syncHasChatClass();
    closeMobileSidebar();
  } catch {
    if (entry.title) await handleQuestion(entry.title);
  }
}

function renderChatHistory(filterText) {
  const host = document.getElementById("history-groups");
  if (!host) return;
  const needle = String(filterText || "").trim().toLowerCase();
  let entries = remoteChatThreads || loadChatHistory();
  if (needle) {
    entries = entries.filter((row) => String(row.title || "").toLowerCase().includes(needle));
  }
  host.replaceChildren();
  if (!entries.length) {
    const empty = document.createElement("p");
    empty.className = "hint history-empty";
    empty.textContent = needle ? "No matching chats." : "No saved chats yet.";
    host.append(empty);
    return;
  }
  const order = ["Today", "Yesterday", "Previous 7 days", "Older"];
  const groups = new Map(order.map((name) => [name, []]));
  for (const entry of entries) {
    const bucket = historyBucket(entry.at);
    if (!groups.has(bucket)) groups.set(bucket, []);
    groups.get(bucket).push(entry);
  }
  for (const name of order) {
    const rows = groups.get(name) || [];
    if (!rows.length) continue;
    const label = document.createElement("p");
    label.className = "history-group-label";
    label.textContent = name;
    host.append(label);
    for (const row of rows) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "history-item";
      if (row.id && row.id === currentThreadId) button.setAttribute("aria-current", "true");
      button.textContent = row.title;
      button.addEventListener("click", () => {
        void openSavedChat(row);
      });
      host.append(button);
    }
  }
}

function startNewChat() {
  setChatMode("normal");
  const log = chatLog();
  if (log) log.replaceChildren();
  currentThreadId = null;
  showToolView("chat");
  syncHasChatClass();
  const search = document.getElementById("history-search");
  renderChatHistory(search ? search.value : "");
  const input = document.getElementById("chat-input");
  if (input) input.focus();
}

function resizeComposer() {
  const input = document.getElementById("chat-input");
  if (!input || input.tagName !== "TEXTAREA") return;
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
}

/* ------------------------------------------------------------------ wire-up */

// Older builds saved the mode; drop it so it can never switch Unhinged on.
try {
  localStorage.removeItem(MODE_KEY);
} catch {
  // Storage blocked: nothing saved to clear.
}
setChatMode("normal");
renderToolNav();
renderToolsMenu();
renderRoadmap();
renderChatHistory();
void refreshRemoteChatHistory();
showToolView("chat");

const memoryRefresh = document.getElementById("memory-refresh");
if (memoryRefresh) memoryRefresh.addEventListener("click", () => void loadMemoryPanel());
const memoryClear = document.getElementById("memory-clear");
if (memoryClear) memoryClear.addEventListener("click", () => void clearAllMemory());
const scenarioForm = document.getElementById("scenario-save-form");
if (scenarioForm) scenarioForm.addEventListener("submit", (event) => void saveScenarioFromForm(event));
const whatIfForm = document.getElementById("what-if-form");
if (whatIfForm) whatIfForm.addEventListener("submit", (event) => void submitWhatIfForm(event));
const reportForm = document.getElementById("report-create-form");
if (reportForm) reportForm.addEventListener("submit", (event) => void submitReportForm(event));
const alertForm = document.getElementById("alert-create-form");
if (alertForm) alertForm.addEventListener("submit", (event) => void submitAlertForm(event));
const alertCheck = document.getElementById("alert-check");
if (alertCheck) alertCheck.addEventListener("click", () => void checkAlertsNow());
const alertRefresh = document.getElementById("alert-refresh");
if (alertRefresh) alertRefresh.addEventListener("click", () => void loadAlertsPanel());
const syntheticForm = document.getElementById("synthetic-form");
if (syntheticForm) {
  syntheticForm.addEventListener("submit", (event) => void submitSyntheticForm(event));
}
const syntheticDownload = document.getElementById("synthetic-download");
if (syntheticDownload) {
  syntheticDownload.addEventListener("click", () => downloadSyntheticCsv());
}
const syntheticTrain = document.getElementById("synthetic-train");
if (syntheticTrain) {
  syntheticTrain.addEventListener("click", () => void trainFromSyntheticCsv());
}

document.getElementById("unhinged-toggle").addEventListener("click", () => {
  setChatMode(chatMode() === "unhinged" ? "normal" : "unhinged");
});
document.getElementById("talk-toggle").addEventListener("click", onTalkClick);

document.getElementById("chat-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const input = document.getElementById("chat-input");
  const question = input.value.trim();
  if (pendingImage.dataUrl) {
    const image = pendingImage.dataUrl;
    input.value = "";
    resizeComposer();
    clearPendingImage();
    rememberChatTurn(`[Image] ${question || "Read this image"}`);
    handleImageQuestion(question, image).then(() => syncHasChatClass());
    return;
  }
  if (!question) return;
  input.value = "";
  resizeComposer();
  rememberChatTurn(question);
  handleQuestion(question).then(() => syncHasChatClass());
});

const attachImageBtn = document.getElementById("attach-image-btn");
if (attachImageBtn) attachImageBtn.addEventListener("click", () => promptImagePicker());
const multimodalAttach = document.getElementById("multimodal-attach");
if (multimodalAttach) {
  multimodalAttach.addEventListener("click", () => {
    showToolView("chat");
    promptImagePicker();
  });
}
const imageAttachRemove = document.getElementById("image-attachment-remove");
if (imageAttachRemove) imageAttachRemove.addEventListener("click", () => clearPendingImage());
const imageAttachFile = document.getElementById("image-attach-file");
if (imageAttachFile) {
  imageAttachFile.addEventListener("change", (event) => {
    const input = event.currentTarget;
    const file = input.files && input.files[0];
    input.value = "";
    if (file) void setPendingImage(file);
  });
}
const chatFormEl = document.getElementById("chat-form");
if (chatFormEl) {
  chatFormEl.addEventListener("dragover", (event) => {
    if (event.dataTransfer && [...event.dataTransfer.items].some((item) => item.kind === "file")) {
      event.preventDefault();
    }
  });
  chatFormEl.addEventListener("drop", (event) => {
    const file = firstImageFile(event.dataTransfer && event.dataTransfer.files);
    if (!file) return;
    event.preventDefault();
    void setPendingImage(file);
  });
}

const chatInput = document.getElementById("chat-input");
if (chatInput) {
  chatInput.addEventListener("paste", (event) => {
    const file = firstImageFile(event.clipboardData && event.clipboardData.files);
    if (!file) return;
    event.preventDefault();
    void setPendingImage(file);
  });
  chatInput.addEventListener("input", resizeComposer);
  chatInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      document.getElementById("chat-form").requestSubmit();
    }
  });
}

function runChipAction(action) {
  const click = (id) => {
    const target = document.getElementById(id);
    if (target) target.click();
  };
  if (action === "simulator") showToolView("simulator");
  else if (action === "image") click("attach-image-btn");
  else if (action === "talk") click("talk-toggle");
  else if (action === "unhinged") click("unhinged-toggle");
}

document.querySelectorAll("#suggestion-chips .chip, #demo-chips .chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    if (chip.dataset.action) {
      runChipAction(chip.dataset.action);
      return;
    }
    rememberChatTurn(chip.dataset.question);
    handleQuestion(chip.dataset.question).then(() => syncHasChatClass());
  });
});

document.querySelectorAll(".tool-view .chip[data-question]").forEach((chip) => {
  chip.addEventListener("click", () => {
    showToolView("chat");
    rememberChatTurn(chip.dataset.question);
    handleQuestion(chip.dataset.question).then(() => syncHasChatClass());
  });
});

document.querySelectorAll(".back-to-chat").forEach((button) => {
  button.addEventListener("click", () => showToolView("chat"));
});

const simForm = document.getElementById("sim-form");
if (simForm) {
  simForm.addEventListener("submit", (event) => void submitSimForm(event));
  simForm.series.addEventListener("change", () => syncSimForm());
  syncSimForm();
}
const simPlay = document.getElementById("sim-play");
if (simPlay) simPlay.addEventListener("click", () => toggleSimPlayback());
const simStep = document.getElementById("sim-step");
if (simStep) {
  simStep.addEventListener("input", () => {
    stopSimPlayback();
    renderSimStep(Number(simStep.value) || 1);
  });
}
const diffusionRunBtn = document.getElementById("diffusion-run-btn");
if (diffusionRunBtn) {
  diffusionRunBtn.addEventListener("click", () => {
    void runDiffusionDemo();
  });
}

const jepaRunBtn = document.getElementById("jepa-run-btn");
if (jepaRunBtn) {
  jepaRunBtn.addEventListener("click", () => {
    void runJepaDemo();
  });
}

const newChatBtn = document.getElementById("new-chat-btn");
if (newChatBtn) newChatBtn.addEventListener("click", startNewChat);

const historySearch = document.getElementById("history-search");
if (historySearch) {
  historySearch.addEventListener("input", () => renderChatHistory(historySearch.value));
}

const toolsMenuBtn = document.getElementById("tools-menu-btn");
const toolsMenu = document.getElementById("tools-menu");
if (toolsMenuBtn && toolsMenu) {
  toolsMenuBtn.addEventListener("click", () => {
    const open = toolsMenu.hidden;
    toolsMenu.hidden = !open;
    toolsMenuBtn.setAttribute("aria-expanded", open ? "true" : "false");
  });
}

const attachCsvBtn = document.getElementById("attach-csv-btn");
if (attachCsvBtn) {
  attachCsvBtn.addEventListener("click", () => {
    promptCsvPicker();
  });
}

const trainCsvPanelBtn = document.getElementById("train-csv-panel-btn");
if (trainCsvPanelBtn) {
  trainCsvPanelBtn.addEventListener("click", () => {
    showToolView("chat");
    handleQuestion("Train on my CSV").then(() => syncHasChatClass());
  });
}

const forecastActions = document.getElementById("forecast-panel-actions");
if (forecastActions) {
  for (const [label, question] of [
    ["Central Park temperature", "Try the Central Park temperature sample"],
    ["AAPL sample", "Try the AAPL sample"],
    ["Bitcoin sample", "Try the Bitcoin sample"],
  ]) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.textContent = label;
    chip.addEventListener("click", () => {
      showToolView("chat");
      rememberChatTurn(question);
      handleQuestion(question).then(() => syncHasChatClass());
    });
    forecastActions.append(chip);
  }
  const share = document.createElement("button");
  share.type = "button";
  share.className = "chip";
  share.textContent = "Share weather report";
  share.addEventListener("click", () => {
    showToolView("reports");
    void createShareableReport("weather")
      .then(() => loadReportsPanel())
      .catch((error) => reportError(error));
  });
  forecastActions.append(share);
}

document.getElementById("account-btn").addEventListener("click", openSignInModal);
const heroSignup = document.getElementById("hero-signup-btn");
if (heroSignup) {
  heroSignup.addEventListener("click", () => openGate("signup"));
}
document.getElementById("modal-close").addEventListener("click", closeSignInModal);
document.getElementById("sign-in-modal").addEventListener("click", (event) => {
  if (event.target === event.currentTarget) closeSignInModal();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeSignInModal();
  }
});

document.getElementById("sign-out-btn").addEventListener("click", async () => {
  try {
    await fetchJson("/auth/logout", { method: "POST" });
  } catch (error) {
    reportError(error);
  }
  window.location.assign("/");
});

document.getElementById("train-btn").addEventListener("click", (event) => {
  startTraining(document.getElementById("training-output"));
  event.currentTarget.blur();
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
    await uploadCsvFile(file);
    input.value = "";
    clearNotice();
    await refreshWorkspace();
  } catch (error) {
    reportError(error);
  }
});

document.getElementById("csv-train-file").addEventListener("change", (event) => {
  const input = event.currentTarget;
  const file = input.files && input.files[0];
  input.value = "";
  if (file) handleCsvTrainFile(file);
});

const sidebarToggle = document.getElementById("sidebar-toggle");
const sidebarBackdrop = document.getElementById("sidebar-backdrop");
const isMobile = () => window.matchMedia("(max-width: 900px)").matches;

function setSidebarControls() {
  const expanded = isMobile()
    ? document.body.classList.contains("sidebar-open")
    : !document.body.classList.contains("sidebar-collapsed");
  sidebarToggle.setAttribute("aria-expanded", expanded ? "true" : "false");
  if (sidebarBackdrop) {
    sidebarBackdrop.hidden = !(isMobile() && document.body.classList.contains("sidebar-open"));
  }
}

sidebarToggle.addEventListener("click", () => {
  if (isMobile()) {
    const open = document.body.classList.toggle("sidebar-open");
    sidebarToggle.setAttribute("aria-expanded", open ? "true" : "false");
    if (sidebarBackdrop) sidebarBackdrop.hidden = !open;
  } else {
    const collapsed = document.body.classList.toggle("sidebar-collapsed");
    sidebarToggle.setAttribute("aria-expanded", collapsed ? "false" : "true");
  }
});

document.getElementById("sidebar-close").addEventListener("click", () => {
  if (isMobile()) {
    document.body.classList.remove("sidebar-open");
    sidebarToggle.setAttribute("aria-expanded", "false");
    if (sidebarBackdrop) sidebarBackdrop.hidden = true;
  } else {
    document.body.classList.add("sidebar-collapsed");
    sidebarToggle.setAttribute("aria-expanded", "false");
  }
});

if (sidebarBackdrop) {
  sidebarBackdrop.addEventListener("click", closeMobileSidebar);
}

window.addEventListener("resize", setSidebarControls);
setSidebarControls();
syncHasChatClass();


/* ------------------------------------------------------------- auth gate */

const HAS_ACCOUNT_KEY = "nousHasAccount";
const gate = { mode: "signup", step: "email", started: false, afterRecovery: null, setupToken: null };

// One-time owner setup link: /app?setup=<token>. Kept in memory only and
// stripped from the address bar right away.
(function readSetupLink() {
  const params = new URLSearchParams(window.location.search);
  const token = params.get("setup");
  if (!token) return;
  gate.setupToken = token;
  params.delete("setup");
  const rest = params.toString();
  window.history.replaceState({}, "", window.location.pathname + (rest ? `?${rest}` : ""));
})();

function setAuthView(view) {
  const body = document.body;
  body.classList.remove("auth-pending", "auth-in", "auth-out", "gate-open");
  if (view === "in") body.classList.add("auth-in");
  else if (view === "gate") body.classList.add("auth-out", "gate-open");
  else body.classList.add("auth-out");
}

function gateError(message) {
  const box = document.getElementById("gate-error");
  if (!box) return;
  box.textContent = message || "";
  box.hidden = !message;
}

function showGatePanel(id) {
  ["sign-in-form", "gate-mfa", "gate-setup", "gate-recovery"].forEach((panelId) => {
    const panel = document.getElementById(panelId);
    if (panel) panel.hidden = panelId !== id;
  });
  const signOut = document.getElementById("gate-sign-out");
  if (signOut) signOut.hidden = id === "sign-in-form";
  const title = document.getElementById("gate-title");
  if (title) {
    title.textContent =
      id === "gate-mfa"
        ? "Two-step sign-in"
        : id === "gate-setup"
          ? "Secure your account"
          : id === "gate-recovery"
            ? "Recovery codes"
            : "Sign up or log in";
  }
}

function syncGateMode() {
  const signup = gate.mode === "signup";
  const label = document.getElementById("gate-mode-label");
  const toggle = document.getElementById("show-sign-up");
  const password = document.getElementById("sign-in-password");
  const hint = document.getElementById("gate-password-hint");
  if (label) label.textContent = signup ? "Have an account?" : "New here?";
  if (toggle) toggle.textContent = signup ? "Log in" : "Create account";
  if (password) password.setAttribute("autocomplete", signup ? "new-password" : "current-password");
  if (hint) hint.hidden = !signup || gate.step !== "password";
  const button = document.getElementById("gate-continue");
  if (button) button.textContent = gate.step === "email" ? "Continue" : signup ? "Create account" : "Log in";
}

function openGate(mode) {
  if (mode === "login" || mode === "signup") gate.mode = mode;
  else gate.mode = localStorage.getItem(HAS_ACCOUNT_KEY) ? "login" : "signup";
  gate.step = "email";
  const row = document.getElementById("gate-password-row");
  if (row) row.hidden = true;
  gateError("");
  showGatePanel("sign-in-form");
  if (gate.setupToken) {
    gate.mode = "signup";
    const title = document.getElementById("gate-title");
    if (title) title.textContent = "Set up the owner account";
  }
  syncGateMode();
  setAuthView("gate");
  if (window.location.pathname !== "/app") window.history.pushState({}, "", "/app");
  const email = document.getElementById("sign-in-email");
  if (email) email.focus();
}

function onSessionLost() {
  if (!document.body.classList.contains("auth-in")) return;
  state.account = null;
  openGate("login");
}

async function bootAuth() {
  const account = await refreshAccount();
  if (account) {
    afterSignIn(account);
  } else if (window.location.pathname === "/app") {
    openGate();
  } else {
    setAuthView("out");
  }
}

function afterSignIn(account) {
  state.account = account;
  localStorage.setItem(HAS_ACCOUNT_KEY, "1");
  refreshAuthModal();
  if (account.role === "owner" && account.auth !== "local" && account.mfa !== "ok") {
    setAuthView("gate");
    gateError("");
    if (account.mfa === "setup") {
      showGatePanel("gate-setup");
      syncPasskeyButtons();
    } else {
      showGatePanel("gate-mfa");
      syncPasskeyButtons();
    }
    return;
  }
  enterApp();
}

function enterApp() {
  setAuthView("in");
  if (window.location.pathname === "/app") window.history.replaceState({}, "", "/");
  if (gate.started) return;
  gate.started = true;
  loadDefaultDemos().catch(() => {});
  refreshAccountDependents().catch(() => {});
  refreshVoiceStatus().catch(() => {});
  armTalkForPhone();
}

function passkeysAvailable() {
  return typeof window.PublicKeyCredential === "function" && !!navigator.credentials;
}

function syncPasskeyButtons() {
  const ok = passkeysAvailable();
  ["gate-passkey-btn", "gate-add-passkey", "security-add-passkey"].forEach((id) => {
    const button = document.getElementById(id);
    if (button) button.hidden = !ok;
  });
}

function b64urlToBuffer(text) {
  const clean = String(text).replace(/-/g, "+").replace(/_/g, "/");
  const padded = clean + "=".repeat((4 - (clean.length % 4)) % 4);
  const binary = atob(padded);
  const out = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) out[index] = binary.charCodeAt(index);
  return out.buffer;
}

function bufferToB64url(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function postJson(url, body) {
  return fetchJson(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
}

function showRecoveryCodes(codes, then) {
  if (!codes || !codes.length) {
    then();
    return;
  }
  const list = document.getElementById("gate-recovery-list");
  list.replaceChildren();
  codes.forEach((code) => {
    const item = document.createElement("li");
    item.textContent = code;
    list.append(item);
  });
  gate.afterRecovery = then;
  setAuthView("gate");
  showGatePanel("gate-recovery");
}

async function finishSecondFactor(result) {
  const account = await refreshAccount();
  const done = () => {
    if (account) afterSignIn(account);
    else openGate("login");
  };
  showRecoveryCodes(result && result.recovery_codes, done);
}

async function registerPasskey() {
  gateError("");
  const options = await postJson("/auth/webauthn/register/options");
  const publicKey = options.publicKey;
  publicKey.challenge = b64urlToBuffer(publicKey.challenge);
  publicKey.user.id = b64urlToBuffer(publicKey.user.id);
  publicKey.excludeCredentials = (publicKey.excludeCredentials || []).map((item) => ({
    ...item,
    id: b64urlToBuffer(item.id),
  }));
  const credential = await navigator.credentials.create({ publicKey });
  if (!credential) throw new Error("Passkey setup was cancelled.");
  const result = await postJson("/auth/webauthn/register/verify", {
    challenge_id: options.challenge_id,
    id: credential.id,
    clientDataJSON: bufferToB64url(credential.response.clientDataJSON),
    attestationObject: bufferToB64url(credential.response.attestationObject),
  });
  await finishSecondFactor(result);
}

async function usePasskey() {
  gateError("");
  const options = await postJson("/auth/webauthn/login/options");
  const publicKey = options.publicKey;
  publicKey.challenge = b64urlToBuffer(publicKey.challenge);
  publicKey.allowCredentials = (publicKey.allowCredentials || []).map((item) => ({
    ...item,
    id: b64urlToBuffer(item.id),
  }));
  const assertion = await navigator.credentials.get({ publicKey });
  if (!assertion) throw new Error("Passkey sign-in was cancelled.");
  const result = await postJson("/auth/webauthn/login/verify", {
    challenge_id: options.challenge_id,
    id: assertion.id,
    clientDataJSON: bufferToB64url(assertion.response.clientDataJSON),
    authenticatorData: bufferToB64url(assertion.response.authenticatorData),
    signature: bufferToB64url(assertion.response.signature),
  });
  await finishSecondFactor(result);
}

async function startTotpSetup() {
  gateError("");
  const result = await postJson("/auth/mfa/totp/setup");
  setAuthView("gate");
  showGatePanel("gate-setup");
  document.getElementById("gate-totp").hidden = false;
  document.getElementById("gate-totp-secret").textContent = result.secret.replace(/(.{4})/g, "$1 ").trim();
  document.getElementById("gate-totp-link").href = result.otpauth_url;
  document.getElementById("gate-totp-code").focus();
}

async function refreshSecuritySummary() {
  const summary = document.getElementById("security-summary");
  if (!summary) return;
  try {
    const status = await fetchJson("/auth/mfa/status");
    const parts = [];
    parts.push(status.passkeys ? `${status.passkeys} passkey${status.passkeys === 1 ? "" : "s"}` : "no passkey");
    parts.push(status.totp ? "authenticator app on" : "no authenticator app");
    parts.push(`${status.recovery_codes_left} recovery codes left`);
    summary.textContent = parts.join(" · ");
  } catch {
    summary.textContent = "";
  }
  syncPasskeyButtons();
}

function gateAction(fn) {
  return (event) => {
    if (event) event.preventDefault();
    Promise.resolve()
      .then(fn)
      .catch((error) => {
        const message =
          error && error.name === "NotAllowedError"
            ? "Passkey request was cancelled or timed out."
            : (error && error.message) || "Something went wrong.";
        if (document.body.classList.contains("gate-open")) gateError(message);
        else reportError(error);
      });
  };
}

document.getElementById("sign-in-form").addEventListener(
  "submit",
  gateAction(async () => {
    const form = document.getElementById("sign-in-form");
    const email = form.email.value.trim();
    if (!email || !/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email)) {
      gateError("Enter your email.");
      return;
    }
    gateError("");
    if (gate.step === "email") {
      gate.step = "password";
      document.getElementById("gate-password-row").hidden = false;
      syncGateMode();
      form.password.focus();
      return;
    }
    const password = form.password.value;
    if (!password) {
      gateError("Enter your password.");
      return;
    }
    const payload = { email, password };
    let account;
    if (gate.mode === "signup") {
      if (gate.setupToken) payload.setup_token = gate.setupToken;
      try {
        account = await postJson("/auth/signup", payload);
      } catch (error) {
        // Someone who already has an account often taps "Create account".
        // Try their password as a log-in before showing an error.
        if (!error || error.status !== 409 || !/already exists/i.test(error.message || "")) throw error;
        gate.mode = "login";
        syncGateMode();
        try {
          account = await postJson("/auth/login", { email, password });
        } catch (loginError) {
          if (loginError && loginError.status === 401 && /invalid email or password/i.test(loginError.message || "")) {
            throw new Error("That email already has an account. Enter its password and tap Log in.");
          }
          throw loginError;
        }
      }
    } else {
      account = await postJson("/auth/login", payload);
    }
    gate.setupToken = null;
    form.password.value = "";
    afterSignIn(account);
  })
);

document.getElementById("show-sign-up").addEventListener("click", () => {
  gate.mode = gate.mode === "signup" ? "login" : "signup";
  gateError("");
  syncGateMode();
});

document.getElementById("gate-code-form").addEventListener(
  "submit",
  gateAction(async () => {
    const field = document.getElementById("gate-code");
    const result = await postJson("/auth/mfa/verify", { code: field.value.trim() });
    field.value = "";
    await finishSecondFactor(result);
  })
);

document.getElementById("gate-totp-form").addEventListener(
  "submit",
  gateAction(async () => {
    const field = document.getElementById("gate-totp-code");
    const result = await postJson("/auth/mfa/totp/confirm", { code: field.value.trim() });
    field.value = "";
    document.getElementById("gate-totp").hidden = true;
    await finishSecondFactor(result);
  })
);

document.getElementById("gate-passkey-btn").addEventListener("click", gateAction(usePasskey));
document.getElementById("gate-add-passkey").addEventListener("click", gateAction(registerPasskey));
document.getElementById("gate-add-totp").addEventListener("click", gateAction(startTotpSetup));
document.getElementById("gate-recovery-done").addEventListener("click", () => {
  document.getElementById("gate-recovery-list").replaceChildren();
  const next = gate.afterRecovery;
  gate.afterRecovery = null;
  if (next) next();
});
document.getElementById("gate-sign-out").addEventListener("click", async () => {
  try {
    await fetchJson("/auth/logout", { method: "POST" });
  } catch {
    // Leaving anyway.
  }
  window.location.assign("/");
});
document.getElementById("security-add-passkey").addEventListener(
  "click",
  gateAction(async () => {
    closeSignInModal();
    await registerPasskey();
  })
);
document.getElementById("security-add-totp").addEventListener(
  "click",
  gateAction(async () => {
    closeSignInModal();
    await startTotpSetup();
  })
);
document.getElementById("security-new-codes").addEventListener(
  "click",
  gateAction(async () => {
    const result = await postJson("/auth/mfa/recovery/regenerate");
    closeSignInModal();
    showRecoveryCodes(result.recovery_codes, () => enterApp());
  })
);
document.getElementById("account-open-gate").addEventListener("click", () => {
  closeSignInModal();
  openGate("login");
});
document.querySelectorAll("[data-open-gate]").forEach((link) => {
  link.addEventListener("click", (event) => {
    event.preventDefault();
    openGate(link.dataset.openGate);
  });
});
window.addEventListener("popstate", () => {
  if (document.body.classList.contains("auth-in")) return;
  if (window.location.pathname === "/app") openGate();
  else setAuthView("out");
});

/* ------------------------------------------------------------------- boot */

document.querySelectorAll("[data-view-open]").forEach((button) => {
  button.addEventListener("click", () => showToolView(button.dataset.viewOpen));
});

bootAuth();
