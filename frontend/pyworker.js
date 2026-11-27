/* Henad code runner: runs one Python snippet in Pyodide, in this browser only.
 *
 * A fresh worker is started for every run and thrown away afterwards, so runs
 * never share state. Python packages the snippet imports are fetched from the
 * Pyodide CDN first; then every network door in this worker is shut before any
 * user code runs (fetch, XHR, WebSocket, importScripts, dynamic import), and
 * the response for this file carries a Content-Security-Policy that only lets
 * it talk to the Pyodide CDN. The worker has no DOM, no cookies and no
 * localStorage, so the code can't see the page, the session, or any secret.
 * The page kills the worker when the time limit is up.
 */
"use strict";

const PYODIDE_BASE = "https://cdn.jsdelivr.net/pyodide/v0.29.5/full/";
const OUTPUT_LIMIT = 20000;

let output = "";
let truncated = false;

function emit(text) {
  if (truncated) return;
  output += text;
  if (output.length > OUTPUT_LIMIT) {
    output = output.slice(0, OUTPUT_LIMIT);
    truncated = true;
  }
  self.postMessage({ type: "output", text: output, truncated });
}

function lockDown() {
  const names = [
    "fetch",
    "XMLHttpRequest",
    "WebSocket",
    "WebTransport",
    "EventSource",
    "importScripts",
    "Worker",
    "SharedWorker",
    "BroadcastChannel",
    "indexedDB",
    "caches",
    "navigator",
    "Request",
    "RTCPeerConnection",
  ];
  for (let scope = self; scope; scope = Object.getPrototypeOf(scope)) {
    for (const name of names) {
      try {
        delete scope[name];
      } catch {
        // keep going
      }
    }
  }
  for (const name of names) {
    try {
      Object.defineProperty(self, name, { value: undefined, writable: false, configurable: false });
    } catch {
      // keep going
    }
  }
}

self.onmessage = async (event) => {
  const code = String((event.data && event.data.code) || "");
  try {
    self.postMessage({ type: "status", text: "Starting Python…" });
    importScripts(`${PYODIDE_BASE}pyodide.js`);
    const pyodide = await self.loadPyodide({
      indexURL: PYODIDE_BASE,
      stdout: (line) => emit(`${line}\n`),
      stderr: (line) => emit(`${line}\n`),
    });
    try {
      await pyodide.loadPackagesFromImports(code, { messageCallback: () => {}, errorCallback: () => {} });
    } catch {
      // A package that isn't available fails at import time with a clear error.
    }
    pyodide.setStdin({
      stdin: () => {
        throw new Error("input() isn't available here. Put the values in the code instead.");
      },
    });
    lockDown();
    // Hide the JS bridge from user code so it can't reach back into the worker.
    pyodide.runPython(
      "import sys\nfor _m in ('js', 'pyodide.http', 'pyodide_js'):\n    sys.modules[_m] = None\ndel _m",
    );
    self.postMessage({ type: "status", text: "Running…" });
    const started = Date.now();
    let result = await pyodide.runPythonAsync(code);
    const ms = Date.now() - started;
    if (result !== undefined && result !== null && typeof result !== "function") {
      let shown = "";
      try {
        shown = String(result.toString ? result.toString() : result);
      } catch {
        shown = "";
      }
      if (result && typeof result.destroy === "function") result.destroy();
      if (shown && shown !== "None") emit(`${shown}\n`);
    }
    result = null;
    self.postMessage({ type: "done", text: output, truncated, ms });
  } catch (error) {
    const message = String((error && error.message) || error || "Error");
    // Trim Pyodide's internal frames so people see their own traceback.
    const lines = message.split("\n");
    const start = lines.findIndex((line) => line.includes('File "<exec>"'));
    const clean = start > 0 ? ["Traceback (most recent call last):", ...lines.slice(start)].join("\n") : message;
    self.postMessage({ type: "error", text: output, error: clean.trim(), truncated });
  }
};
