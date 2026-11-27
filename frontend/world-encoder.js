/**
 * Henad: self-supervised world model (JEPA) panel.
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

  function row(label, side, unit, winner, persistenceMae) {
    const tr = document.createElement("tr");
    const name = document.createElement("th");
    name.scope = "row";
    name.textContent = winner ? `${label} ✓` : label;
    const valCell = document.createElement("td");
    const maeCell = document.createElement("td");
    const covCell = document.createElement("td");
    if (persistenceMae != null) {
      valCell.textContent = "—";
      maeCell.textContent = simFormat(persistenceMae, unit);
      covCell.textContent = "—";
    } else if (!side) {
      valCell.textContent = "—";
      maeCell.textContent = "—";
      covCell.textContent = "—";
    } else {
      valCell.textContent = side.validation_mae == null ? "—" : simFormat(side.validation_mae, unit);
      maeCell.textContent = simFormat(side.mae, unit);
      const raw = side.coverage_80_raw;
      const cal = side.coverage_80;
      covCell.textContent =
        raw == null || cal == null || raw === cal ? pct(cal) : `${pct(raw)} → ${pct(cal)}`;
    }
    tr.append(name, valCell, maeCell, covCell);
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
      const vgain = asNumber(cmp.validation_improvement_pct);
      const name = payload.name || "this series";
      let text;
      if (!cmp.jepa) {
        text = cmp.note || "JEPA could not be scored on this series, so Henad used PCA.";
      } else if (vgain == null) {
        text = `${cmp.note || "Too little history for a separate validation split."} Henad used PCA.`;
      } else if (chosen === "jepa") {
        text = `JEPA wins on ${name}: ${Math.abs(vgain).toFixed(1)}% lower error than PCA on the validation windows, and ${gain != null && gain > 0 ? `${gain.toFixed(1)}% lower` : "no better"} on the untouched holdouts. The simulation below uses the self-supervised encoder.`;
      } else if (vgain > 0) {
        text = `JEPA was only ${vgain.toFixed(1)}% better than PCA on validation (it needs 5%), so Henad kept the simpler PCA encoder.`;
      } else {
        text = `PCA wins on ${name}: JEPA was ${Math.abs(vgain).toFixed(1)}% worse on validation, so Henad kept PCA. Learned features don't help every series.`;
      }
      verdict.textContent = text;
    }
    const table = el("wenc-table");
    if (table) {
      table.replaceChildren(
        row("Persistence (tomorrow = today)", null, unit, false, cmp.mae_persistence),
        row("PCA encoder", cmp.pca, unit, chosen === "pca"),
        row("JEPA encoder (3-seed ensemble)", cmp.jepa, unit, chosen === "jepa"),
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
      const holdouts = cmp.origins
        ? `${cmp.origins} newest holdouts of ${cmp.holdout} steps are graded; ${cmp.validation_origins || 0} older validation windows pick the encoder. `
        : "";
      const cal = payload.model && payload.model.calibration;
      const widen =
        cal && cal.applied ? `Ranges were widened x${Number(cal.scale).toFixed(2)} (conformal calibration). ` : "";
      note.textContent = `${holdouts}${widen}${cmp.selection_note || ""}`.trim();
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
  window.HenadWorldEncoder = { render, submit };
})();
