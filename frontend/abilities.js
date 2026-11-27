/* Henad chat abilities: reading attached files and showing images Henad made.
 * Everything triggers from normal chat; the only control is the attach button.
 */
(function () {
  "use strict";

  const FILE_MAX_BYTES = 5 * 1024 * 1024;
  const IMAGE_MARKER = /\[\[image:(img_[a-f0-9]{20})\]\]/g;
  const REMINDER_MARKER = /\[\[reminder:(rem_[a-f0-9]{20})\]\]/g;
  const POLL_MS = 60000;
  const pending = { files: [] };

  /* ------------------------------------------------------------- replies */

  function stripMarkers(text) {
    return String(text || "")
      .replace(IMAGE_MARKER, "")
      .replace(REMINDER_MARKER, "")
      .replace(/[ \t]+\n/g, "\n")
      .trim();
  }

  function addGeneratedImage(bubble, id) {
    const figure = document.createElement("figure");
    figure.className = "gen-image";
    const img = document.createElement("img");
    img.src = `/agent/images/${id}`;
    img.alt = "Image made by Henad";
    img.loading = "lazy";
    img.decoding = "async";
    img.addEventListener("load", () => {
      const log = document.getElementById("chat-log");
      if (log) log.scrollTop = log.scrollHeight;
    });
    img.addEventListener("error", () => {
      figure.replaceChildren(document.createTextNode("This image has expired or can't be loaded."));
    });
    const save = document.createElement("a");
    save.href = `/agent/images/${id}`;
    save.download = `henad-${id}.jpg`;
    save.className = "gen-image-save";
    save.textContent = "Save image";
    figure.append(img, save);
    bubble.append(figure);
  }

  /** Called by addReply for every assistant message (live and from history). */
  function decorateReply(message, text) {
    if (!message) return;
    const raw = String(text || "");
    const bubble = message.querySelector(".bubble");
    if (!bubble) return;
    const ids = [...raw.matchAll(IMAGE_MARKER)].map((match) => match[1]);
    const reminder = /\[\[reminder:rem_[a-f0-9]{20}\]\]/.test(raw);
    if (ids.length || reminder) {
      const clean = stripMarkers(raw);
      const host = bubble.querySelector(".chart-host");
      bubble.textContent = clean;
      if (host) bubble.append(host);
      message.dataset.speech = clean;
      for (const id of ids) addGeneratedImage(bubble, id);
      if (reminder) offerNotifications(message);
    }
    for (const hook of decorators) {
      try {
        hook(message, raw, bubble);
      } catch {
        // A decorator never breaks the reply.
      }
    }
  }

  const decorators = [];

  /* --------------------------------------------------------------- files */

  function chip() {
    let el = document.getElementById("file-attachment");
    if (el) return el;
    const form = document.getElementById("chat-form");
    if (!form) return null;
    el = document.createElement("div");
    el.id = "file-attachment";
    el.className = "file-attachment";
    el.hidden = true;
    const label = document.createElement("span");
    label.className = "file-attachment-name";
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "icon-btn icon-btn-small";
    remove.setAttribute("aria-label", "Remove attached file");
    remove.title = "Remove file";
    remove.textContent = "×";
    remove.addEventListener("click", () => clearFiles());
    el.append(label, remove);
    const row = form.querySelector(".composer-row");
    form.insertBefore(el, row || form.firstChild);
    return el;
  }

  function showChip(text, busy) {
    const el = chip();
    if (!el) return;
    el.hidden = false;
    el.dataset.busy = busy ? "true" : "false";
    el.querySelector(".file-attachment-name").textContent = text;
  }

  function clearFiles() {
    pending.files = [];
    const el = document.getElementById("file-attachment");
    if (el) el.hidden = true;
  }

  function notice(text) {
    const el = document.getElementById("notice");
    if (el) el.textContent = text;
  }

  function sizeLabel(bytes) {
    if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  }

  async function attach(file) {
    if (!file) return;
    if (file.size > FILE_MAX_BYTES) {
      notice("That file is too big. Keep it under 5 MB.");
      return;
    }
    showChip(`Reading ${file.name}…`, true);
    try {
      const response = await fetch("/agent/files", {
        method: "POST",
        credentials: "same-origin",
        headers: {
          "Content-Type": file.type || "application/octet-stream",
          "X-File-Name": encodeURIComponent(file.name),
        },
        body: file,
      });
      let data = {};
      try {
        data = await response.json();
      } catch {
        data = {};
      }
      if (!response.ok || !data.file) {
        clearFiles();
        notice(data.error || data.detail || "I couldn't read that file.");
        return;
      }
      pending.files = [data.file];
      showChip(`📄 ${data.file.name} · ${sizeLabel(data.file.bytes)}`, false);
      notice("");
      const input = document.getElementById("chat-input");
      if (input) input.focus();
    } catch {
      clearFiles();
      notice("I couldn't upload that file. Check your connection and try again.");
    }
  }

  function hasPendingFile() {
    return pending.files.length > 0;
  }

  function timeZone() {
    try {
      return Intl.DateTimeFormat().resolvedOptions().timeZone || "";
    } catch {
      return "";
    }
  }

  /** Extra fields for the next /agent/chat call; clears the attachment. */
  function takeChatExtras() {
    const extras = {};
    const tz = timeZone();
    if (tz) extras.tz = tz;
    if (!pending.files.length) return extras;
    extras.file_ids = pending.files.map((file) => file.id);
    clearFiles();
    return extras;
  }

  /* ----------------------------------------------------------- reminders */

  function offerNotifications(message) {
    if (!("Notification" in window) || Notification.permission !== "default") return;
    const actions = message.querySelector(".message-actions");
    if (!actions || actions.querySelector(".notify-btn")) return;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "btn secondary btn-small notify-btn";
    btn.textContent = "🔔 Also notify me on this device";
    btn.addEventListener("click", async () => {
      try {
        await Notification.requestPermission();
      } catch {
        // ignore
      }
      btn.remove();
    });
    actions.append(btn);
  }

  let polling = false;
  let pollStopped = false;

  async function pollReminders() {
    if (polling || pollStopped || document.visibilityState === "hidden" && !("Notification" in window)) return;
    polling = true;
    try {
      const response = await fetch("/agent/reminders/due", { credentials: "same-origin", headers: { Accept: "application/json" } });
      if (response.status === 401 || response.status === 404) return;
      if (!response.ok) return;
      const data = await response.json();
      const deliveries = Array.isArray(data.deliveries) ? data.deliveries : [];
      if (!deliveries.length) return;
      for (const item of deliveries) {
        const text = String(item.text || "");
        if (typeof window.addReply === "function") window.addReply(text);
        if ("Notification" in window && Notification.permission === "granted" && document.visibilityState === "hidden") {
          try {
            new Notification("Henad", { body: text, icon: "/assets/brand/icon-192.png", tag: item.id });
          } catch {
            // Some browsers only allow notifications from a service worker.
          }
        }
      }
      if (typeof window.syncHasChatClass === "function") window.syncHasChatClass();
      await fetch("/agent/reminders/ack", {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ids: deliveries.map((item) => item.id) }),
      });
    } catch {
      // Try again next minute.
    } finally {
      polling = false;
    }
  }

  function startReminderPolling() {
    setTimeout(pollReminders, 4000);
    setInterval(pollReminders, POLL_MS);
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState === "visible") void pollReminders();
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", startReminderPolling);
  } else {
    startReminderPolling();
  }

  function isImageFile(file) {
    return Boolean(file && typeof file.type === "string" && file.type.startsWith("image/"));
  }

  window.henadAbilities = {
    attach,
    hasPendingFile,
    takeChatExtras,
    isImageFile,
    decorateReply,
    addDecorator(hook) {
      if (typeof hook === "function") decorators.push(hook);
    },
  };
})();
