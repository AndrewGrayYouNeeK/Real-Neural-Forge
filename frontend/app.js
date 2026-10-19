async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || response.statusText);
  }
  return data;
}

const TOKEN_KEY = "neuralForgeApiToken";
let holdTrainingMessage = false;

function checkpointLabel(modelInfo) {
  const loaded = Boolean(modelInfo.checkpoint_loaded);
  const stale = Boolean(modelInfo.checkpoint_stale);
  if (loaded && stale) return "newer file on disk";
  if (stale) return "file not loaded";
  return loaded ? "loaded" : "missing";
}

function renderStatus(modelInfo, health, training) {
  const grid = document.getElementById("status-grid");
  const cards = [
    ["API Health", health.status],
    ["Architecture", modelInfo.architecture],
    ["Parameters", modelInfo.parameters.toLocaleString()],
    ["Device", modelInfo.device],
    ["Checkpoint", checkpointLabel(modelInfo)],
    ["Training", training.status],
  ];

  grid.innerHTML = cards
    .map(
      ([label, value]) => `
        <div class="stat-card">
          <span>${label}</span>
          <strong>${value}</strong>
        </div>
      `
    )
    .join("");
}

function renderExperiments(experiments) {
  const list = document.getElementById("experiments-list");
  if (!experiments.length) {
    list.innerHTML = "<p class='hint'>No experiments yet. Start a training run.</p>";
    return;
  }

  list.innerHTML = experiments
    .map(
      (exp) => `
        <div class="experiment-card">
          <div>
            <strong>${exp.name}</strong>
            <div class="hint">${exp.model_name} · ${exp.created_at}</div>
          </div>
          <div>
            <span class="badge ${exp.status}">${exp.status}</span>
            <div class="hint">loss: ${exp.best_loss ?? "n/a"}</div>
          </div>
        </div>
      `
    )
    .join("");
}

function renderYouneek(now) {
  const grid = document.getElementById("youneek-grid");
  const cards = [
    ["Clock", now.time.display],
    ["Year clock", now.calendar.year_clock.display],
    ["Year / week / day", `${now.calendar.year_index} / ${now.calendar.week} / ${now.calendar.weekday}`],
    ["Lunar", now.lunar.clock.display],
    ["Next minute", now.forecast.next_minute.youneek_display],
    ["Scale", "100/100/100"],
  ];
  grid.innerHTML = cards
    .map(
      ([label, value]) => `
        <div class="stat-card">
          <span>${label}</span>
          <strong>${value}</strong>
        </div>
      `
    )
    .join("");
  document.getElementById("youneek-output").textContent = JSON.stringify(now, null, 2);
}

async function refreshDashboard() {
  const [health, modelInfo, training, experiments, youneek] = await Promise.all([
    fetchJson("/health"),
    fetchJson("/model/info"),
    fetchJson("/training/status"),
    fetchJson("/experiments"),
    fetchJson("/youneek/now"),
  ]);

  renderStatus(modelInfo, health, training);
  renderYouneek(youneek);
  renderExperiments(experiments.experiments);
  if (!holdTrainingMessage) {
    document.getElementById("training-output").textContent = JSON.stringify(
      training,
      null,
      2
    );
  }
}

document.getElementById("refresh-btn").addEventListener("click", () => {
  holdTrainingMessage = false;
  refreshDashboard().catch((error) => alert(error.message));
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
  } catch (error) {
    output.textContent = error.message;
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
    setTimeout(() => refreshDashboard().catch(() => {}), 1500);
  } catch (error) {
    output.textContent = error.message;
  }
});

refreshDashboard().catch((error) => {
  document.getElementById("status-grid").innerHTML = `<p>${error.message}</p>`;
});

setInterval(() => {
  refreshDashboard().catch(() => {});
}, 5000);
