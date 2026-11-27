/* Henad code runner (page side): shows code blocks in replies and runs Python
 * ones in a throwaway Pyodide worker (frontend/pyworker.js). Nothing runs on
 * the server; each run gets 15 seconds after Python has started.
 */
(function () {
  "use strict";

  const RUN_SECONDS = 15;
  const START_SECONDS = 60;
  const FENCE = /```([A-Za-z0-9_+.-]*)[ \t]*\n?([\s\S]*?)```/g;
  const PYTHON = new Set(["python", "py", "python3"]);

  function splitFences(text) {
    const parts = [];
    let last = 0;
    for (const match of text.matchAll(FENCE)) {
      if (match.index > last) parts.push({ kind: "text", text: text.slice(last, match.index) });
      parts.push({ kind: "code", lang: (match[1] || "").toLowerCase(), code: match[2].replace(/\n$/, "") });
      last = match.index + match[0].length;
    }
    if (last < text.length) parts.push({ kind: "text", text: text.slice(last) });
    return parts;
  }

  function button(label, onClick) {
    const el = document.createElement("button");
    el.type = "button";
    el.className = "code-btn";
    el.textContent = label;
    el.addEventListener("click", onClick);
    return el;
  }

  function runPython(code, out, runBtn) {
    runBtn.disabled = true;
    out.hidden = false;
    out.dataset.state = "running";
    out.textContent = "Starting Python…";
    let worker;
    try {
      worker = new Worker("/assets/pyworker.js");
    } catch {
      out.dataset.state = "error";
      out.textContent = "This browser can't run Python here.";
      runBtn.disabled = false;
      return;
    }
    let timer = null;
    let finished = false;
    const finish = (state, text) => {
      if (finished) return;
      finished = true;
      clearTimeout(timer);
      worker.terminate();
      out.dataset.state = state;
      out.textContent = text;
      runBtn.disabled = false;
      runBtn.textContent = "▶ Run again";
    };
    const arm = (seconds, message) => {
      clearTimeout(timer);
      timer = setTimeout(() => finish("error", message), seconds * 1000);
    };
    arm(START_SECONDS, "Python took too long to start. Check your connection and try again.");
    worker.onmessage = (event) => {
      const data = event.data || {};
      if (data.type === "status") {
        if (data.text === "Running…") arm(RUN_SECONDS, `Stopped after ${RUN_SECONDS} seconds. Long loops or waiting for input can cause this.`);
        if (!out.dataset.hasOutput) out.textContent = data.text;
      } else if (data.type === "output") {
        out.dataset.hasOutput = "1";
        out.textContent = data.text;
      } else if (data.type === "done") {
        const text = (data.text || "").replace(/\n$/, "");
        finish("done", text ? `${text}${data.truncated ? "\n… (output cut off)" : ""}` : "Done. (No output; add print() to see results.)");
      } else if (data.type === "error") {
        const text = (data.text || "").replace(/\n$/, "");
        finish("error", `${text ? `${text}\n` : ""}${data.error || "Error"}`);
      }
    };
    worker.onerror = () => finish("error", "Python couldn't start in this browser.");
    worker.postMessage({ code });
  }

  function codeBlock(lang, code) {
    const wrap = document.createElement("div");
    wrap.className = "code-block";
    const bar = document.createElement("div");
    bar.className = "code-bar";
    const label = document.createElement("span");
    label.className = "code-lang";
    label.textContent = lang || "code";
    bar.append(label);
    const pre = document.createElement("pre");
    const codeEl = document.createElement("code");
    codeEl.textContent = code;
    pre.append(codeEl);
    const copy = button("Copy", async () => {
      try {
        await navigator.clipboard.writeText(code);
        copy.textContent = "Copied";
        setTimeout(() => (copy.textContent = "Copy"), 1500);
      } catch {
        copy.textContent = "Copy failed";
      }
    });
    bar.append(copy);
    wrap.append(bar, pre);
    if (PYTHON.has(lang) && typeof Worker === "function") {
      const out = document.createElement("pre");
      out.className = "code-output";
      out.hidden = true;
      const run = button("▶ Run", () => runPython(code, out, run));
      run.classList.add("code-run");
      bar.append(run);
      wrap.append(out);
    }
    return wrap;
  }

  function decorate(message, raw, bubble) {
    if (!raw.includes("```")) return;
    const keep = [...bubble.children].filter((child) => child.matches(".chart-host, .gen-image"));
    const parts = splitFences(raw.replace(/\[\[image:img_[a-f0-9]{20}\]\]/g, ""));
    if (!parts.some((part) => part.kind === "code")) return;
    bubble.replaceChildren();
    for (const part of parts) {
      if (part.kind === "text") {
        const text = part.text.replace(/^\n+|\n+$/g, "");
        if (text) {
          const span = document.createElement("div");
          span.className = "bubble-text";
          span.textContent = text;
          bubble.append(span);
        }
      } else {
        bubble.append(codeBlock(part.lang, part.code));
      }
    }
    for (const child of keep) bubble.append(child);
    // Read prose aloud, not code.
    message.dataset.speech = parts
      .filter((part) => part.kind === "text")
      .map((part) => part.text.trim())
      .filter(Boolean)
      .join(" ");
  }

  if (window.henadAbilities) window.henadAbilities.addDecorator(decorate);
  window.henadCodeRunner = { splitFences };
})();
