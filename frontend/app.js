/* Neural Forge chat-first dashboard. All API text is written with textContent only. */
const charts = window.NeuralForgeCharts;
const TOKEN_KEY = "neuralForgeApiToken";
const MAX_VOICE_CHARS = 2000;

const state = {
  account: null,
  voiceStatus: null,
  training: null,
  experiments: [],
  series: null,
  trainingWatch: null,
};

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

function authHeaders(extra = {}) {
  const token = localStorage.getItem(TOKEN_KEY);
  return token ? { Authorization: `Bearer ${token}`, ...extra } : { ...extra };
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

function addMessage(role, text) {
  const message = document.createElement("div");
  message.className = `message ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  message.append(bubble);

  if (role === "assistant") {
    message.dataset.speech = text;
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

function setSpeakBusy(speakButton, busy) {
  speakButton.disabled = busy;
  speakButton.textContent = busy ? SPEAK_BUSY_LABEL : `🔊 ${SPEAK_LABEL}`;
}

async function speakMessage(message, speakButton) {
  const text = (message.dataset.speech || "").trim();
  if (!text) {
    showHint(message, "There is nothing in this reply to read aloud.");
    return;
  }
  setSpeakBusy(speakButton, true);
  try {
    const response = await fetch("/voice/speak", {
      method: "POST",
      credentials: "same-origin",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({ text: text.slice(0, MAX_VOICE_CHARS) }),
    });
    if (response.status === 401) {
      showHint(
        message,
        "GQ voice needs a free account. Reading the chat in text stays free.",
        [["Sign in", openSignInModal]]
      );
      return;
    }
    if (response.status === 402) {
      showHint(
        message,
        "You are out of GQ voice credit. Buy a credit pack in Voice & credits — text stays free.",
        [["Buy credits", openVoicePanel]]
      );
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
    audio.addEventListener("ended", () => URL.revokeObjectURL(audio.src));
    audio.play().catch(() => showHint(message, "Your browser blocked audio playback."));
    refreshVoiceStatus().catch(() => {});
  } catch (error) {
    showHint(message, error.message || "GQ voice failed. Text stays free.");
  } finally {
    setSpeakBusy(speakButton, false);
  }
}

/* --------------------------------------------------------- topic replies */

function forecastChart(series, steps) {
  const history = Array.isArray(series.history) ? series.history : [];
  const labels = history.map((point) => point.time || String(point.index));
  const actual = history.map((point) => asNumber(point.actual));
  const forecast = new Array(history.length).fill(null);
  const low = new Array(history.length).fill(null);
  const high = new Array(history.length).fill(null);
  const lastActual = actual.length ? actual[actual.length - 1] : null;
  if (lastActual != null) forecast[history.length - 1] = lastActual;
  steps.forEach((step, index) => {
    labels.push(`+${step.step || index + 1}`);
    actual.push(null);
    forecast.push(step.p50 != null ? step.p50 : step.value);
    low.push(step.p10);
    high.push(step.p90);
  });
  const lines = [{ name: "Actual", color: "#e8f4ff", values: actual }];
  if (forecast.some((value) => value != null)) {
    lines.push({
      name: "Forecast",
      color: "#7cffd6",
      markers: true,
      values: forecast,
    });
  }
  return {
    ariaLabel: "Recent history with the model's forecast",
    labels,
    series: lines,
    bands: hasInterval(steps)
      ? [{ name: "p10–p90", color: "#7cffd6", low, high }]
      : [],
    empty: "Not enough history to draw a forecast.",
  };
}

function stepText(steps) {
  if (!steps.length) return null;
  const parts = steps.slice(0, 5).map((step, index) => {
    const value = formatMetric(step.p50 != null ? step.p50 : step.value);
    const band =
      step.p10 != null && step.p90 != null
        ? ` (likely between ${formatMetric(step.p10)} and ${formatMetric(step.p90)})`
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

function formatStarterValue(kind, value) {
  const number = asNumber(value);
  if (number == null) return null;
  if (kind === "weather") return `${formatMetric(number)}°C`;
  return formatMetric(number);
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
    const nextText = stepText(steps);
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
  return lines;
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

async function lotteryReply(question) {
  try {
    const result = await fetchJson(liveUrl("/live/lottery", question));
    if (result && typeof result.text === "string" && result.text.trim()) {
      return { text: result.text.trim() };
    }
  } catch (error) {
    return {
      text:
        "I could not load the latest lottery draws right now. Draws are random either way — no model can pick the next winning numbers. " +
        (error.message || ""),
    };
  }
  return {
    text: "I could not load the latest lottery draws right now. Draws are random either way — no model can pick the next winning numbers.",
  };
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
    try {
      const study = await fetchJson("/lottery/lunar-analysis");
      if (study && typeof study.answer_in_brief === "string" && study.answer_in_brief) {
        lines.push(study.answer_in_brief);
      }
    } catch {
      // The lunar lottery study is optional garnish here.
    }
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

async function trainReply(question) {
  if (!isTrainStartRequest(question)) {
    return {
      text:
        "I can train a model on your data in a private workspace, or the owner can train the shared champion. Say “start training” or “train a model” when you want a run to begin. Until then, nothing is started.",
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
          "Every reply here has a small speaker button. Press it and my GQ voice reads the answer aloud. Reading text is always free; listening spends GQ voice credit, and a free account starts with a small allowance.",
        actions: [["Sign in", openSignInModal]],
      };
    }
    if (status.owner_unlimited) {
      return {
        text: "You are signed in as the owner, so GQ voice is unlimited for you. Press any speaker button to listen.",
      };
    }
    const balance = status.balance || {};
    return {
      text:
        `You have ${Number(balance.free || 0).toLocaleString()} free and ${Number(
          balance.purchased || 0
        ).toLocaleString()} purchased GQ characters left. ` +
        "Press any speaker button to listen; packs live under Voice & credits in the Tools sidebar.",
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
      "• Lottery — the latest Powerball and Mega Millions draws from published results; I will not pretend I can pick winners\n" +
      "• Moon phase, near-Earth approaches, and the solar system\n" +
      "• Training a model on your data, if you ask me to start a run\n" +
      "• GQ voice — the speaker button on each reply, once you have an account\n" +
      "• General questions — a short AI answer, within a daily free allowance\n\n" +
      "If the AI allowance is used up, I will say so and still help with the live topics above.",
  };
}

function greetReply() {
  return {
    text:
      "Hello! I am Neural Forge. Ask me for a live weather forecast, a market or crypto price, the latest lottery draws, the moon, space, or a general question. You can also tell me to train a model on your data. What would you like to know?",
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
    name: "lottery",
    test: /\b(lottery|lotto|powerball|mega millions|winning numbers|lucky numbers)\b/i,
    handler: lotteryReply,
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
    test: /\b(gq( voice)?|voice credit|read (it |this |that )?aloud|speaker button|text to speech)\b/i,
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

async function askLive(question) {
  try {
    const result = await fetchJson("/live/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (result && typeof result.text === "string" && result.text.trim()) {
      return { text: result.text.trim() };
    }
  } catch (error) {
    return {
      text:
        (error && error.message) ||
        "I could not reach the general-answer model right now. I can still help with live weather, market prices, lottery draws, the moon, near-Earth space, and training.",
    };
  }
  return {
    text: "I could not reach the general-answer model right now. I can still help with live weather, market prices, lottery draws, the moon, near-Earth space, and training.",
  };
}

async function answer(question) {
  const intent = matchIntent(question);
  if (intent) return intent.handler(question);
  return askLive(question);
}

async function handleQuestion(question) {
  addMessage("user", question);
  const pending = addMessage("assistant", "One moment — checking…");
  try {
    const reply = await answer(question);
    pending.remove();
    const replyMessage = addReply(reply.text, reply.chart);
    if (reply.actions) {
      const actions = replyMessage.querySelector(".message-actions");
      for (const [label, action] of reply.actions) {
        actions.append(makeButton(label, action));
      }
      scrollToBottom();
    }
    clearNotice();
  } catch (error) {
    pending.remove();
    addReply(`Something went wrong on my side. ${error.message || error}`);
    reportError(error);
  }
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

function renderVoicePanel() {
  const status = state.voiceStatus;
  const grid = document.getElementById("voice-status-grid");
  const packs = document.getElementById("voice-packs");
  const usage = document.getElementById("voice-usage-list");
  const message = document.getElementById("voice-message");
  packs.replaceChildren();
  usage.replaceChildren();
  if (!status) {
    fillStatGrid(grid, [["GQ voice", "checking"]]);
    return;
  }
  const cards = [["Text on the site", "always free"]];
  if (!status.signed_in) {
    cards.push(["GQ voice", status.voice ? "sign in to listen" : "unavailable"]);
  } else if (status.owner_unlimited) {
    cards.push(["GQ voice", status.voice ? "unlimited (owner)" : "unavailable"]);
  } else {
    const balance = status.balance || { free: 0, purchased: 0, spent: 0 };
    cards.push(["Free GQ left", Number(balance.free).toLocaleString()]);
    cards.push(["Purchased GQ left", Number(balance.purchased).toLocaleString()]);
    cards.push(["GQ spent", Number(balance.spent).toLocaleString()]);
  }
  fillStatGrid(grid, cards);
  for (const pack of status.packs || []) {
    const card = document.createElement("article");
    card.className = "starter-card";
    const title = document.createElement("h3");
    title.textContent = pack.name;
    const detail = document.createElement("p");
    detail.className = "hint";
    detail.textContent = `${Number(pack.chars).toLocaleString()} characters — $${Number(
      pack.price_usd
    ).toFixed(2)} one-time`;
    const buy = document.createElement("button");
    buy.type = "button";
    buy.className = "btn primary btn-small";
    buy.textContent = status.signed_in ? "Buy credit pack" : "Sign in to buy";
    buy.addEventListener("click", () => buyVoicePack(pack.id, status.signed_in));
    card.append(title, detail, buy);
    packs.append(card);
  }
  if (!(status.packs || []).length) {
    const item = document.createElement("p");
    item.className = "hint";
    item.textContent = "No credit packs are configured yet.";
    packs.append(item);
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

async function refreshVoiceStatus() {
  try {
    state.voiceStatus = await fetchJson("/voice/status");
  } catch (error) {
    document.getElementById("voice-message").textContent = error.message;
    return;
  }
  renderVoicePanel();
}

async function buyVoicePack(packId, signedIn) {
  const message = document.getElementById("voice-message");
  if (!signedIn) {
    message.textContent =
      "Create a free account first — then buy GQ credit here. Text stays free either way.";
    openSignInModal();
    return;
  }
  try {
    const result = await fetchJson("/voice/checkout", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pack_id: packId }),
    });
    if (result && result.available && result.url) {
      window.location.assign(result.url);
      return;
    }
    message.textContent =
      (result && result.detail) ||
      "Buying GQ voice credit is coming soon. Text stays free.";
  } catch (error) {
    message.textContent = error.message;
  }
}

function openVoicePanel() {
  document.getElementById("voice-panel").open = true;
  const sidebar = document.getElementById("sidebar");
  sidebar.scrollTop = 0;
  document.getElementById("voice-panel").scrollIntoView({ block: "start" });
  if (window.matchMedia("(max-width: 900px)").matches) {
    document.body.classList.add("sidebar-open");
    document.getElementById("sidebar-toggle").setAttribute("aria-expanded", "true");
  }
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
  if (state.account) {
    signedIn.hidden = false;
    signedOut.hidden = true;
    const auth = state.account.auth === "local" ? "local owner" : state.account.role;
    label.textContent = `${state.account.email} (${auth})`;
    button.textContent = state.account.email;
  } else {
    signedIn.hidden = true;
    signedOut.hidden = false;
    button.textContent = "Sign in";
  }
}

function selectOwnerHash() {
  const params = new URLSearchParams(window.location.search);
  const ownerUrl =
    params.get("owner") != null ||
    window.location.hash.replace(/^#\/?/, "").toLowerCase() === "owner";
  if (ownerUrl) openOwnerModal();
}

function openOwnerModal() {
  const modal = document.getElementById("owner-modal");
  modal.hidden = false;
  document.getElementById("owner-token").focus();
}

function closeOwnerModal() {
  document.getElementById("owner-modal").hidden = true;
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

function renderWorkspaceLists(datasets, models, predictions) {
  const datasetList = document.getElementById("dataset-list");
  const modelList = document.getElementById("model-list");
  const predictionList = document.getElementById("prediction-list");
  datasetList.replaceChildren();
  modelList.replaceChildren();
  predictionList.replaceChildren();
  if (!datasets.length) appendTextItem(datasetList, "No datasets yet.");
  for (const dataset of datasets) {
    const shared = dataset.shared ? "shared starter" : "private copy";
    appendTextItem(datasetList, `${dataset.name} · ${shared}`);
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
      if (status.status !== "running") {
        clearInterval(state.trainingWatch);
        state.trainingWatch = null;
        refreshExperiments().catch(() => {});
        refreshStatusLine().catch(() => {});
      }
    } catch {
      // Keep watching; transient errors are fine.
    }
  }, 8000);
}

/* ------------------------------------------------------------------ wire-up */

document.getElementById("chat-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const input = document.getElementById("chat-input");
  const question = input.value.trim();
  if (!question) return;
  input.value = "";
  handleQuestion(question);
});

document.querySelectorAll("#suggestion-chips .chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    handleQuestion(chip.dataset.question);
  });
});

document.getElementById("account-btn").addEventListener("click", openSignInModal);
document.getElementById("modal-close").addEventListener("click", closeSignInModal);
document.getElementById("sign-in-modal").addEventListener("click", (event) => {
  if (event.target === event.currentTarget) closeSignInModal();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeSignInModal();
    closeOwnerModal();
  }
});

document.getElementById("sign-in-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  const notice = form.querySelector(".form-error");
  if (notice) notice.remove();
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
    await refreshAccount();
    await refreshAccountDependents();
  } catch (error) {
    reportError(error);
  }
});

document.getElementById("show-sign-up").addEventListener("click", () => {
  document.getElementById("sign-in-form").hidden = true;
  document.getElementById("sign-up-form").hidden = false;
  document.getElementById("sign-up-email").focus();
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
    await refreshAccount();
    await refreshAccountDependents();
  } catch (error) {
    reportError(error);
  }
});

document.getElementById("owner-modal-close").addEventListener("click", closeOwnerModal);
document.getElementById("owner-modal").addEventListener("click", (event) => {
  if (event.target === event.currentTarget) closeOwnerModal();
});

document.getElementById("owner-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  const token = form.token.value.trim();
  if (!token) return;
  localStorage.setItem(TOKEN_KEY, token);
  try {
    await fetchJson("/auth/owner", { method: "POST" });
    form.reset();
    clearNotice();
    closeOwnerModal();
    await refreshAccount();
    await refreshAccountDependents();
  } catch (error) {
    localStorage.removeItem(TOKEN_KEY);
    reportError(new Error(error.message || "Invalid owner token."));
  }
});

document.getElementById("sign-out-btn").addEventListener("click", async () => {
  try {
    await fetchJson("/auth/logout", { method: "POST" });
  } catch (error) {
    reportError(error);
  }
  localStorage.removeItem(TOKEN_KEY);
  await refreshAccount();
  await refreshAccountDependents();
  closeSignInModal();
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

const sidebarToggle = document.getElementById("sidebar-toggle");
const isMobile = () => window.matchMedia("(max-width: 900px)").matches;

function setSidebarControls() {
  const expanded = isMobile()
    ? document.body.classList.contains("sidebar-open")
    : !document.body.classList.contains("sidebar-collapsed");
  sidebarToggle.setAttribute("aria-expanded", expanded ? "true" : "false");
}

sidebarToggle.addEventListener("click", () => {
  if (isMobile()) {
    const open = document.body.classList.toggle("sidebar-open");
    sidebarToggle.setAttribute("aria-expanded", open ? "true" : "false");
  } else {
    const collapsed = document.body.classList.toggle("sidebar-collapsed");
    sidebarToggle.setAttribute("aria-expanded", collapsed ? "false" : "true");
  }
});

document.getElementById("sidebar-close").addEventListener("click", () => {
  if (isMobile()) {
    document.body.classList.remove("sidebar-open");
    sidebarToggle.setAttribute("aria-expanded", "false");
  } else {
    document.body.classList.add("sidebar-collapsed");
    sidebarToggle.setAttribute("aria-expanded", "false");
  }
});

window.addEventListener("resize", setSidebarControls);
setSidebarControls();

/* ------------------------------------------------------------------- boot */

addMessage(
  "assistant",
  "Hi! I am Neural Forge. Ask me in plain words for a live weather forecast, a market or crypto price, the latest lottery draws, the moon, space, training, or a general question. Every reply comes with a speaker button that reads it aloud in GQ voice."
);

refreshAccount()
  .then(() => refreshAccountDependents())
  .catch(() => {});
refreshVoiceStatus().catch(() => {});
refreshStatusLine().catch(() => {});
selectOwnerHash();
