/* Offline SVG charts. API text is written with textContent only. */
(function () {
  const SVG_NS = "http://www.w3.org/2000/svg";

  function svg(name, attrs) {
    const node = document.createElementNS(SVG_NS, name);
    Object.entries(attrs || {}).forEach(([key, value]) => {
      if (value != null) node.setAttribute(key, String(value));
    });
    return node;
  }

  function safeColor(color, fallback) {
    if (typeof color === "string" && /^#[0-9a-fA-F]{6}$/.test(color)) return color;
    return fallback;
  }

  function formatNumber(value) {
    if (value == null || !Number.isFinite(value)) return "—";
    const abs = Math.abs(value);
    if (abs !== 0 && (abs < 0.001 || abs >= 100000)) return value.toExponential(2);
    return String(Math.round(value * 1000) / 1000);
  }

  function clip(value, max) {
    const text = value == null ? "" : String(value);
    return text.length > max ? `${text.slice(0, max - 1)}…` : text;
  }

  function hasFinite(values) {
    return values.some((value) => value != null && Number.isFinite(value));
  }

  function message(container, text) {
    container.replaceChildren();
    const paragraph = document.createElement("p");
    paragraph.className = "hint";
    paragraph.textContent = text;
    container.append(paragraph);
  }

  function legendItem(item) {
    const row = document.createElement("li");
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    if (item.dashed) swatch.classList.add("is-dashed");
    if (item.band) swatch.classList.add("is-band");
    const color = safeColor(item.color, "#7cffd6");
    if (item.band) swatch.style.background = color;
    else swatch.style.borderTopColor = color;
    const label = document.createElement("span");
    label.textContent = item.name == null ? "" : String(item.name);
    row.append(swatch, label);
    return row;
  }

  function dataTable(caption, columns, rows) {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = "View data";
    const wrap = document.createElement("div");
    wrap.className = "table-wrap";
    const table = document.createElement("table");
    const cap = document.createElement("caption");
    cap.className = "visually-hidden";
    cap.textContent = caption;
    const head = document.createElement("thead");
    const headRow = document.createElement("tr");
    columns.forEach((column) => {
      const cell = document.createElement("th");
      cell.scope = "col";
      cell.textContent = column;
      headRow.append(cell);
    });
    head.append(headRow);
    const body = document.createElement("tbody");
    rows.forEach((row) => {
      const tr = document.createElement("tr");
      row.forEach((value) => {
        const cell = document.createElement("td");
        cell.textContent = value == null ? "" : String(value);
        tr.append(cell);
      });
      body.append(tr);
    });
    table.append(cap, head, body);
    wrap.append(table);
    details.append(summary, wrap);
    return details;
  }

  function extent(groups) {
    let min = Infinity;
    let max = -Infinity;
    groups.forEach((values) => {
      values.forEach((value) => {
        if (value == null || !Number.isFinite(value)) return;
        min = Math.min(min, value);
        max = Math.max(max, value);
      });
    });
    if (!Number.isFinite(min) || !Number.isFinite(max)) return null;
    if (min === max) {
      min -= 1;
      max += 1;
    }
    const pad = (max - min) * 0.08;
    return [min - pad, max + pad];
  }

  function runs(values) {
    const segments = [];
    let current = [];
    values.forEach((value, index) => {
      if (value == null || !Number.isFinite(value)) {
        if (current.length) segments.push(current);
        current = [];
        return;
      }
      current.push([index, value]);
    });
    if (current.length) segments.push(current);
    return segments;
  }

  function renderLineChart(container, spec) {
    const series = (spec.series || []).filter((item) => hasFinite(item.values || []));
    const bands = (spec.bands || []).filter(
      (band) => hasFinite(band.low || []) && hasFinite(band.high || [])
    );
    if (!series.length && !bands.length) {
      message(container, spec.empty || "Not enough data to chart.");
      return;
    }

    const labels = spec.labels || [];
    const count = Math.max(
      labels.length,
      ...series.map((item) => item.values.length),
      ...bands.map((band) => Math.max(band.low.length, band.high.length))
    );
    const [minY, maxY] = extent([
      ...series.map((item) => item.values),
      ...bands.flatMap((band) => [band.low, band.high]),
    ]);
    const width = 640;
    const height = 280;
    const pad = { left: 56, right: 16, top: 16, bottom: 36 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;
    const xAt = (index) =>
      count <= 1 ? pad.left + plotW / 2 : pad.left + (index / (count - 1)) * plotW;
    const yAt = (value) => pad.top + ((maxY - value) / (maxY - minY)) * plotH;

    const figure = document.createElement("div");
    figure.className = "chart-block";
    const legend = document.createElement("ul");
    legend.className = "legend";
    series.forEach((item) => legend.append(legendItem(item)));
    bands.forEach((band) => legend.append(legendItem({ ...band, band: true })));

    const host = document.createElement("div");
    host.className = "chart-frame";
    const label = spec.ariaLabel || "Chart";
    host.setAttribute("role", "img");
    host.setAttribute("aria-label", label);
    const plot = svg("svg", {
      viewBox: `0 0 ${width} ${height}`,
      class: "chart-svg",
      "aria-hidden": "true",
    });

    for (let tick = 0; tick < 4; tick += 1) {
      const value = minY + ((maxY - minY) * tick) / 3;
      const y = yAt(value);
      plot.append(
        svg("line", {
          x1: pad.left,
          x2: width - pad.right,
          y1: y,
          y2: y,
          class: "chart-grid",
        })
      );
      const text = svg("text", { x: pad.left - 8, y: y + 4, "text-anchor": "end" });
      text.textContent = formatNumber(value);
      plot.append(text);
    }
    if (minY < 0 && maxY > 0) {
      plot.append(
        svg("line", {
          x1: pad.left,
          x2: width - pad.right,
          y1: yAt(0),
          y2: yAt(0),
          class: "chart-zero",
        })
      );
    }

    bands.forEach((band) => {
      const color = safeColor(band.color, "#7cffd6");
      let segment = [];
      const flush = () => {
        if (segment.length < 2) {
          segment = [];
          return;
        }
        const forward = segment
          .map((index) => `${xAt(index)},${yAt(band.high[index])}`)
          .join(" ");
        const back = segment
          .slice()
          .reverse()
          .map((index) => `${xAt(index)},${yAt(band.low[index])}`)
          .join(" ");
        plot.append(
          svg("polygon", {
            points: `${forward} ${back}`,
            fill: color,
            opacity: "0.22",
            stroke: "none",
          })
        );
        segment = [];
      };
      for (let index = 0; index < count; index += 1) {
        const low = band.low[index];
        const high = band.high[index];
        if (low == null || high == null || !Number.isFinite(low) || !Number.isFinite(high)) {
          flush();
        } else {
          segment.push(index);
        }
      }
      flush();
    });

    series.forEach((item) => {
      const color = safeColor(item.color, "#7cffd6");
      runs(item.values).forEach((segment) => {
        if (segment.length === 1 || item.markers) {
          segment.forEach(([index, value]) => {
            plot.append(
              svg("circle", {
                cx: xAt(index),
                cy: yAt(value),
                r: segment.length === 1 ? 4 : 2.5,
                fill: color,
              })
            );
          });
        }
        if (segment.length < 2) return;
        const points = segment.map(([index, value]) => `${xAt(index)},${yAt(value)}`).join(" ");
        const line = svg("polyline", {
          points,
          fill: "none",
          stroke: color,
          "stroke-width": "2",
          "stroke-linejoin": "round",
          "stroke-linecap": "round",
        });
        if (item.dashed) line.setAttribute("stroke-dasharray", "6 4");
        plot.append(line);
      });
    });

    const tickCount = Math.min(6, count);
    const seen = new Set();
    for (let tick = 0; tick < tickCount; tick += 1) {
      const index = count <= 1 ? 0 : Math.round((tick * (count - 1)) / (tickCount - 1));
      if (seen.has(index)) continue;
      seen.add(index);
      const text = svg("text", {
        x: xAt(index),
        y: height - 12,
        "text-anchor": "middle",
      });
      text.textContent = clip(labels[index] == null ? String(index) : labels[index], 14);
      plot.append(text);
    }

    host.append(plot);
    const columns = ["x"].concat(
      series.map((item) => String(item.name)),
      bands.flatMap((band) => [`${band.name} low`, `${band.name} high`])
    );
    const tableRows = [];
    for (let index = 0; index < count; index += 1) {
      const row = [labels[index] == null ? String(index) : String(labels[index])];
      series.forEach((item) => row.push(formatNumber(item.values[index])));
      bands.forEach((band) => {
        row.push(formatNumber(band.low[index]), formatNumber(band.high[index]));
      });
      tableRows.push(row);
    }
    figure.append(legend, host, dataTable(label, columns, tableRows));
    container.replaceChildren(figure);
  }

  function renderBarChart(container, spec) {
    const rows = (spec.rows || []).filter((row) => Number.isFinite(row.value));
    if (!rows.length) {
      message(container, spec.empty || "Not enough data to chart.");
      return;
    }
    const width = 640;
    const rowH = 36;
    const height = Math.max(88, rows.length * rowH + 20);
    const labelW = 156;
    const valueW = 72;
    const padRight = 12;
    const min = Math.min(0, ...rows.map((row) => row.value));
    const max = Math.max(0, ...rows.map((row) => row.value));
    const span = max - min || 1;
    const plotW = width - labelW - valueW - padRight;
    const xAt = (value) => labelW + ((value - min) / span) * plotW;
    const plot = svg("svg", {
      viewBox: `0 0 ${width} ${height}`,
      class: "chart-svg",
      "aria-hidden": "true",
    });
    const zero = xAt(0);
    plot.append(
      svg("line", { x1: zero, x2: zero, y1: 8, y2: height - 8, class: "chart-zero" })
    );
    rows.forEach((row, index) => {
      const y = 16 + index * rowH;
      const color = safeColor(row.color, "#7cffd6");
      const start = xAt(Math.min(0, row.value));
      const end = xAt(Math.max(0, row.value));
      plot.append(
        svg("rect", {
          x: start,
          y: y,
          width: Math.max(2, end - start),
          height: 16,
          rx: 4,
          fill: color,
        })
      );
      const name = svg("text", { x: 8, y: y + 13 });
      name.textContent = clip(row.name, 22);
      const full = svg("title");
      full.textContent = row.name == null ? "" : String(row.name);
      name.append(full);
      plot.append(name);
      const value = svg("text", { x: end + 8, y: y + 13 });
      value.textContent = formatNumber(row.value);
      plot.append(value);
    });

    const host = document.createElement("div");
    host.className = "chart-frame";
    const label = spec.ariaLabel || "Bar chart";
    host.setAttribute("role", "img");
    host.setAttribute("aria-label", label);
    host.append(plot);
    const figure = document.createElement("div");
    figure.className = "chart-block";
    figure.append(
      host,
      dataTable(
        label,
        [spec.nameLabel || "Name", spec.valueLabel || "Value"],
        rows.map((row) => [String(row.name), formatNumber(row.value)])
      )
    );
    container.replaceChildren(figure);
  }

  window.NeuralForgeCharts = { renderLineChart, renderBarChart };
})();
