/**
 * Nous: self-supervised world model (JEPA) panel.
 * Runs /analysis/simulate with encoder "auto" so the container backtests the
 * PCA and JEPA encoders on the chosen series and rolls the winner forward.
 * Loaded after app.js and reuses its helpers (fetchJson, charts, simChart,
 * simFormat, parseSimValues, asNumber).
 */
(function () {
  const form = document.getElementById("wenc-form");
  if (!form) return;

  function el(id) {
    return document.getElementById(id);
  }

  function syncForm() {
    const custom = el("wenc-custom");
    if (custom) custom.hidden = form.series.value !== "custom";
  }

  function pct(value) {
    const number = asNumber(value);
    return number == null ? "—" : `${Math.round(number * 100)}%`;
  }

  function row(label, mae, coverage, unit, winner) {
    const tr = document.createElement("tr");
    const name = document.createElement("th");
    name.scope = "row";
    name.textContent = winner ? `${label} ✓` : label;
    const maeCell = document.createElement("td");
    maeCell.textContent = simFormat(mae, unit);
    const covCell = document.createElement("td");
    covCell.textContent = coverage == null ? "—" : pct(coverage);
    tr.append(name, maeCell, covCell);
    return tr;
  }

  function render(payload) {
    const result = el("wenc-result");
    if (!result || !payload) return;
    result.hidden = false;
    const cmp = payload.encoder_comparison || {};
    const unit = payload.unit || null;
    const chosen = cmp.chosen || (payload.model && payload.model.encoder) || "pca";
    const verdict = el("wenc-verdict");
    if (verdict) {
      const gain = asNumber(cmp.improvement_pct);
      let text;
      if (!cmp.jepa) {
        text = cmp.note || "JEPA could not be scored on this series, so Nous used PCA.";
      } else if (chosen === "jepa") {
        text = `JEPA wins on ${payload.name || "this series"}: ${Math.abs(gain || 0).toFixed(1)}% lower holdout error than PCA, so the simulation below uses the self-supervised encoder.`;
      } else if (gain != null && gain > 0) {
        text = `JEPA was only ${gain.toFixed(1)}% better than PCA here (under the 2% bar), so Nous kept the simpler PCA encoder.`;
      } else {
        text = `PCA wins on ${payload.name || "this series"}: JEPA was ${Math.abs(gain || 0).toFixed(1)}% worse, so Nous kept PCA. Learned features don't help every series.`;
      }
      verdict.textContent = text;
    }
    const table = el("wenc-table");
    if (table) {
      table.replaceChildren(
        row("Persistence (tomorrow = today)", cmp.mae_persistence, null, unit, false),
        row("PCA encoder", cmp.pca && cmp.pca.mae, cmp.pca && cmp.pca.coverage_80, unit, chosen === "pca"),
        row(
          "JEPA encoder",
          cmp.jepa ? cmp.jepa.mae : null,
          cmp.jepa ? cmp.jepa.coverage_80 : null,
          unit,
          chosen === "jepa",
        ),
      );
    }
    const host = el("wenc-chart");
    if (host && typeof charts !== "undefined" && typeof simChart === "function") {
      const horizon = (payload.baseline && payload.baseline.p50 ? payload.baseline.p50 : []).length;
      charts.renderLineChart(host, simChart(payload, horizon));
    }
    const model = el("wenc-model");
    if (model) {
      const info = payload.model || {};
      const jepa = info.jepa;
      const ssl = jepa
        ? ` JEPA training: ${jepa.train_steps} steps, prediction loss ${jepa.ssl_loss_start} → ${jepa.ssl_loss_end}, ${jepa.train_seconds}s on CPU.`
        : "";
      model.textContent = `${info.description || ""}${ssl}`.trim();
    }
    const note = el("wenc-note");
    if (note) {
      const holdouts = cmp.origins ? `${cmp.origins} holdouts of ${cmp.holdout} steps. ` : "";
      note.textContent = `${holdouts}${cmp.selection_note || ""}`.trim();
    }
  }

  async function submit(event) {
    if (event) event.preventDefault();
    const status = el("wenc-status");
    const button = el("wenc-run");
    const series = form.series.value;
    const horizon = Math.min(60, Math.max(1, Number(form.horizon.value) || 14));
    const body = { horizon, n_paths: 128, encoder: "auto", scenarios: [] };
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
      status.textContent = "Training the JEPA encoder and backtesting both encoders…";
    }
    if (button) button.disabled = true;
    try {
      const payload = await fetchJson("/analysis/simulate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (status) status.hidden = true;
      render(payload);
    } catch (error) {
      if (status) {
        status.hidden = false;
        status.textContent = (error && error.message) || "Could not run the encoder check.";
      }
    } finally {
      if (button) button.disabled = false;
    }
  }

  form.addEventListener("submit", (event) => void submit(event));
  form.series.addEventListener("change", syncForm);
  syncForm();
  window.NousWorldEncoder = { render, submit };
})();
