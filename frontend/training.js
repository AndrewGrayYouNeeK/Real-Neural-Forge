/* Henad training-example collector (owner only).
   Adds Save / Fix it / That was bad to each Henad reply, an Add example
   composer and a Training examples list with tags and JSONL export.
   Nothing here shows unless /owner/training/status says this session is the
   verified owner, and every route checks again on the server. */
(function () {
  "use strict";

  const TAGS = ["funny", "creator", "talk", "controversial", "serious", "facts", "cussing", "short"];
  const SOURCE_LABEL = { save: "saved", fix: "fixed", manual: "typed", bad: "bad" };
  let isOwner = false;
  let probe = null;

  async function call(url, options) {
    const response = await fetch(url, {
      credentials: "same-origin",
      ...(options || {}),
      headers: { "Content-Type": "application/json", ...((options && options.headers) || {}) },
    });
    let data = {};
    try {
      data = await response.json();
    } catch {
      data = {};
    }
    return { status: response.status, ok: response.ok, data: data || {} };
  }

  function el(tag, attrs, children) {
    const node = document.createElement(tag);
    Object.entries(attrs || {}).forEach(([key, value]) => {
      if (value === undefined || value === null || value === false) return;
      if (key === "text") node.textContent = value;
      else if (key === "value") node.value = value;
      else if (key === "on") Object.entries(value).forEach(([name, fn]) => node.addEventListener(name, fn));
      else node.setAttribute(key, value === true ? "" : value);
    });
    (children || []).forEach((child) => child && node.append(child));
    return node;
  }

  function toast(text, kind) {
    const node = el("div", { class: `nt-toast${kind === "err" ? " nt-err" : ""}`, role: "status", text });
    document.body.append(node);
    setTimeout(() => node.remove(), 2600);
  }

  /* -------------------------------------------------------- owner check */

  function checkOwner(force) {
    if (probe && !force) return probe;
    probe = call("/owner/training/status")
      .then((result) => result.ok && result.data.owner === true)
      .catch(() => false)
      .then((owner) => {
        setOwner(owner);
        return owner;
      });
    return probe;
  }

  function setOwner(owner) {
    isOwner = Boolean(owner);
    if (!isOwner) {
      document.querySelectorAll(".nt-actions, .nt-menu-item").forEach((node) => node.remove());
      return;
    }
    addMenuItems();
    document.querySelectorAll("#chat-log .message.assistant").forEach((message) => decorate(message));
  }

  /* ------------------------------------------------------ reply buttons */

  /** Only finished replies (from addReply) carry data-reply-text; pending
      "One moment…" bubbles never get buttons. */
  function replyText(message) {
    return (message.dataset.replyText || "").trim();
  }

  function promptFor(message) {
    let node = message.previousElementSibling;
    while (node) {
      if (node.classList && node.classList.contains("message")) {
        if (node.classList.contains("user")) {
          const bubble = node.querySelector(".bubble");
          return bubble ? (bubble.textContent || "").trim() : "";
        }
        if (node.classList.contains("assistant")) return "";
      }
      node = node.previousElementSibling;
    }
    return "";
  }

  function decorateReply(message, text) {
    if (!message) return;
    if (typeof text === "string" && !message.dataset.replyText) message.dataset.replyText = text;
    if (isOwner) decorate(message);
    else checkOwner(false);
  }

  function decorate(message) {
    const actions = message.querySelector(".message-actions");
    if (!actions || actions.querySelector(".nt-actions")) return;
    const answer = replyText(message);
    if (!answer) return;
    const row = el("div", { class: "nt-actions", role: "group", "aria-label": "Training example (owner only)" });
    const status = el("span", { class: "nt-status", "aria-live": "polite" });
    const save = el("button", { class: "nt-btn", type: "button", title: "Save this answer as a good example", text: "👍 Save" });
    const fix = el("button", { class: "nt-btn", type: "button", title: "Rewrite how Henad should have answered", text: "✏️ Fix it" });
    const bad = el("button", { class: "nt-btn", type: "button", title: "Save as a rejected example", text: "👎 That was bad" });
    save.addEventListener("click", async () => {
      const question = promptFor(message);
      if (!question) {
        openEditor({ title: "Save as example", turns: [["", answer]], source: "save" });
        return;
      }
      save.disabled = true;
      const result = await call("/owner/training/examples", {
        method: "POST",
        body: JSON.stringify({ source: "save", messages: turnsToMessages([[question, answer]]) }),
      });
      save.disabled = false;
      if (result.ok) {
        status.textContent = `Saved ✓ (${result.data.counts ? result.data.counts.good : ""} good)`;
        save.disabled = true;
      } else toast(result.data.detail || "Couldn't save.", "err");
    });
    fix.addEventListener("click", () => {
      openEditor({ title: "Fix it", hint: "Rewrite the answer the way Henad should have said it.", turns: [[promptFor(message), answer]], source: "fix", focusAnswer: true, onSaved: () => (status.textContent = "Fixed ✓") });
    });
    bad.addEventListener("click", async () => {
      bad.disabled = true;
      const question = promptFor(message) || "(no question)";
      const result = await call("/owner/training/examples", {
        method: "POST",
        body: JSON.stringify({ source: "bad", label: "rejected", messages: turnsToMessages([[question, answer]]) }),
      });
      if (!result.ok) {
        bad.disabled = false;
        toast(result.data.detail || "Couldn't save.", "err");
        return;
      }
      status.textContent = "Marked bad ✓";
      const badId = result.data.example && result.data.example.id;
      if (!row.querySelector(".nt-fix-offer")) {
        const offer = el("button", { class: "nt-btn nt-fix-offer", type: "button", text: "Fix it now?" });
        offer.addEventListener("click", () =>
          openEditor({
            title: "Fix the bad answer",
            hint: "Write the answer Henad should have given. It's saved as a good example; the bad one stays rejected.",
            turns: [[question === "(no question)" ? "" : question, answer]],
            source: "fix",
            pairId: badId,
            focusAnswer: true,
            onSaved: () => {
              status.textContent = "Bad + fix saved ✓";
              offer.remove();
            },
          }),
        );
        row.append(offer);
      }
    });
    row.append(save, fix, bad, status);
    actions.append(row);
  }

  function turnsToMessages(turns) {
    const messages = [];
    turns.forEach(([question, answer]) => {
      messages.push({ role: "user", content: question });
      messages.push({ role: "assistant", content: answer });
    });
    return messages;
  }

  function messagesToTurns(messages) {
    const turns = [];
    for (let index = 0; index < messages.length; index += 2) {
      turns.push([messages[index] ? messages[index].content : "", messages[index + 1] ? messages[index + 1].content : ""]);
    }
    return turns.length ? turns : [["", ""]];
  }

  /* ------------------------------------------------------------- sheets */

  function sheet(title, wide) {
    const previous = document.activeElement;
    const close = () => {
      backdrop.remove();
      document.removeEventListener("keydown", onKey);
      if (previous && previous.focus) previous.focus();
    };
    const onKey = (event) => event.key === "Escape" && close();
    const body = el("div", { class: "nt-body" });
    const card = el("div", { class: `nt-card${wide ? " nt-wide" : ""}`, role: "dialog", "aria-modal": "true", "aria-label": title }, [
      el("div", { class: "nt-head" }, [
        el("h2", { text: title }),
        el("button", { class: "modal-close", type: "button", "aria-label": "Close", text: "✕", on: { click: close } }),
      ]),
      body,
    ]);
    const backdrop = el("div", { class: "nt-backdrop", on: { click: (event) => event.target === backdrop && close() } }, [card]);
    document.addEventListener("keydown", onKey);
    document.body.append(backdrop);
    return { body, close };
  }

  function tagPicker(selected) {
    const chosen = new Set(selected || []);
    const wrap = el("div", { class: "nt-tags", role: "group", "aria-label": "Tags" });
    const all = [...TAGS, ...[...chosen].filter((tag) => !TAGS.includes(tag))];
    const draw = () => {
      wrap.replaceChildren();
      all.forEach((tag) => {
        const on = chosen.has(tag);
        wrap.append(
          el("button", {
            type: "button",
            class: `nt-tag${on ? " on" : ""}`,
            "aria-pressed": on ? "true" : "false",
            text: tag,
            on: {
              click: () => {
                if (chosen.has(tag)) chosen.delete(tag);
                else chosen.add(tag);
                draw();
              },
            },
          }),
        );
      });
      const custom = el("input", { class: "nt-tag-input", type: "text", placeholder: "+ tag", "aria-label": "Add a tag", maxlength: "24", autocapitalize: "none" });
      custom.addEventListener("keydown", (event) => {
        if (event.key !== "Enter" && event.key !== ",") return;
        event.preventDefault();
        const tag = custom.value.trim().toLowerCase().replace(/[^a-z0-9-]/g, "");
        if (tag) {
          chosen.add(tag);
          if (!all.includes(tag)) all.push(tag);
          draw();
          wrap.querySelector(".nt-tag-input").focus();
        }
      });
      custom.addEventListener("blur", () => {
        const tag = custom.value.trim().toLowerCase().replace(/[^a-z0-9-]/g, "");
        if (tag && !chosen.has(tag)) {
          chosen.add(tag);
          if (!all.includes(tag)) all.push(tag);
          draw();
        }
      });
      wrap.append(custom);
    };
    draw();
    return { node: wrap, value: () => [...chosen] };
  }

  /**
   * One editor for Fix it, Add example and Edit.
   * opts: {title, hint, turns:[[q,a]], source, tags, pairId, editId, focusAnswer, onSaved}
   */
  function openEditor(opts) {
    const { body, close } = sheet(opts.title || "Add example");
    if (opts.hint) body.append(el("p", { class: "nt-hint", text: opts.hint }));
    if (opts.showList) {
      body.append(el("button", { type: "button", class: "linklike nt-see-all", text: "See all examples, tags and export →", on: { click: () => { close(); openList(); } } }));
    }
    const turnsBox = el("div", { class: "nt-turns" });
    const turnNodes = [];
    const addTurn = (question, answer) => {
      const number = turnNodes.length + 1;
      const q = el("textarea", { class: "nt-input", rows: "2", placeholder: "What someone asks…", "aria-label": `Question ${number}`, value: question || "", enterkeyhint: "next" });
      const a = el("textarea", { class: "nt-input nt-answer", rows: "5", placeholder: "The perfect Henad answer…", "aria-label": `Answer ${number}`, value: answer || "" });
      const remove = number > 1 ? el("button", { type: "button", class: "linklike nt-remove", text: "Remove turn" }) : null;
      const block = el("div", { class: "nt-turn" }, [
        number > 1 ? el("p", { class: "nt-turn-label", text: `Turn ${number}` }) : null,
        el("label", { class: "nt-label", text: "Question" }),
        q,
        el("label", { class: "nt-label", text: "Answer" }),
        a,
        remove,
      ]);
      const entry = { q, a, block };
      if (remove) {
        remove.addEventListener("click", () => {
          turnNodes.splice(turnNodes.indexOf(entry), 1);
          block.remove();
        });
      }
      turnNodes.push(entry);
      turnsBox.append(block);
      return entry;
    };
    (opts.turns && opts.turns.length ? opts.turns : [["", ""]]).forEach(([q, a]) => addTurn(q, a));
    const more = el("button", { type: "button", class: "btn secondary btn-small", text: "+ Add another turn", on: { click: () => addTurn("", "").q.focus() } });
    const tags = tagPicker(opts.tags);
    const note = el("p", { class: "nt-msg", hidden: true });
    const save = el("button", { type: "submit", class: "btn primary nt-save", text: opts.editId ? "Save changes" : "Save example" });
    const again = opts.editId || opts.source !== "manual" ? null : el("label", { class: "nt-again" }, [el("input", { type: "checkbox", checked: true }), document.createTextNode(" Keep open for another")]);
    const form = el("form", { class: "nt-form" }, [turnsBox, more, el("label", { class: "nt-label", text: "Tags" }), tags.node, note, el("div", { class: "nt-save-row" }, [again, save])]);
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const turns = turnNodes.map(({ q, a }) => [q.value.trim(), a.value.trim()]);
      if (turns.some(([q, a]) => !q || !a)) {
        note.hidden = false;
        note.className = "nt-msg nt-err";
        note.textContent = "Fill in every question and answer.";
        return;
      }
      save.disabled = true;
      const payload = { messages: turnsToMessages(turns), tags: tags.value() };
      const result = opts.editId
        ? await call(`/owner/training/examples/${opts.editId}`, { method: "PUT", body: JSON.stringify(payload) })
        : await call("/owner/training/examples", { method: "POST", body: JSON.stringify({ ...payload, source: opts.source || "manual", pair_id: opts.pairId || undefined }) });
      save.disabled = false;
      if (!result.ok) {
        note.hidden = false;
        note.className = "nt-msg nt-err";
        note.textContent = result.data.detail || "Couldn't save. Try again.";
        return;
      }
      if (opts.onSaved) opts.onSaved(result.data.example);
      const counts = result.data.counts;
      toast(opts.editId ? "Saved ✓" : `Saved ✓${counts ? ` — ${counts.good} good examples` : ""}`);
      const keep = again && again.querySelector("input").checked;
      if (keep) {
        turnNodes.splice(1).forEach(({ block }) => block.remove());
        turnNodes[0].q.value = "";
        turnNodes[0].a.value = "";
        turnNodes[0].q.focus();
      } else close();
    });
    body.append(form);
    const target = opts.focusAnswer ? turnNodes[0].a : turnNodes[0].q.value ? turnNodes[0].a : turnNodes[0].q;
    setTimeout(() => target.focus(), 30);
  }

  function openAdd() {
    openEditor({ title: "Add example", hint: "Type a question and the perfect answer. It saves right away.", turns: [["", ""]], source: "manual", showList: true });
  }

  /* --------------------------------------------------------- list page */

  async function openList() {
    const { body } = sheet("Training examples", true);
    let filter = "all";
    const counts = el("p", { class: "nt-counts", text: "Loading…" });
    const filters = el("div", { class: "nt-filters", role: "group", "aria-label": "Filter" });
    const list = el("div", { class: "nt-list" });
    const search = el("input", { class: "nt-input nt-search", type: "search", placeholder: "Search examples", "aria-label": "Search examples" });
    const exportRow = el("div", { class: "nt-row" }, [
      el("button", { type: "button", class: "btn primary btn-small", text: "+ Add example", on: { click: () => openEditor({ title: "Add example", turns: [["", ""]], source: "manual", onSaved: () => load() }) } }),
      el("a", { class: "btn secondary btn-small", href: "/owner/training/export?label=good", download: "examples.jsonl", text: "Export JSONL" }),
      el("a", { class: "btn secondary btn-small", href: "/owner/training/export?label=rejected", download: "rejected.jsonl", text: "Export rejected" }),
    ]);
    body.append(counts, exportRow, filters, search, list);
    let rows = [];

    const drawFilters = (totals) => {
      filters.replaceChildren();
      [
        ["all", `All ${totals.total}`],
        ["good", `Good ${totals.good}`],
        ["rejected", `Rejected ${totals.rejected}`],
      ].forEach(([value, label]) => {
        filters.append(
          el("button", {
            type: "button",
            class: `nt-tag${filter === value ? " on" : ""}`,
            "aria-pressed": filter === value ? "true" : "false",
            text: label,
            on: { click: () => { filter = value; load(); } },
          }),
        );
      });
    };

    const drawList = () => {
      const needle = search.value.trim().toLowerCase();
      list.replaceChildren();
      const shown = rows.filter((row) => !needle || JSON.stringify(row.messages).toLowerCase().includes(needle) || row.tags.join(" ").includes(needle));
      if (!shown.length) {
        list.append(el("p", { class: "nt-hint", text: rows.length ? "No matches." : "No examples yet. Tap 👍 Save, ✏️ Fix it or 👎 on any reply, or + Add example." }));
        return;
      }
      shown.forEach((row) => {
        const first = row.messages[0] ? row.messages[0].content : "";
        const reply = row.messages[1] ? row.messages[1].content : "";
        const extra = row.messages.length > 2 ? ` (+${row.messages.length / 2 - 1} more turn${row.messages.length > 4 ? "s" : ""})` : "";
        const item = el("article", { class: `nt-item${row.label === "rejected" ? " nt-rejected" : ""}` }, [
          el("div", { class: "nt-item-head" }, [
            el("span", { class: `nt-badge nt-${row.label}`, text: row.label === "rejected" ? "rejected" : "good" }),
            el("span", { class: "nt-source", text: SOURCE_LABEL[row.source] || row.source }),
            row.pair_id ? el("span", { class: "nt-source", text: "↔ paired" }) : null,
            el("span", { class: "nt-date", text: new Date(row.created_at).toLocaleDateString() }),
          ]),
          el("p", { class: "nt-q", text: first }),
          el("p", { class: "nt-a", text: reply + extra }),
          row.tags.length ? el("div", { class: "nt-item-tags" }, row.tags.map((tag) => el("span", { class: "nt-chip", text: tag }))) : null,
          el("div", { class: "nt-row" }, [
            el("button", {
              type: "button",
              class: "btn secondary btn-small",
              text: "Edit",
              on: { click: () => openEditor({ title: "Edit example", turns: messagesToTurns(row.messages), tags: row.tags, editId: row.id, onSaved: () => load() }) },
            }),
            row.label === "rejected" && !row.pair_id
              ? el("button", {
                  type: "button",
                  class: "btn secondary btn-small",
                  text: "Fix it",
                  on: { click: () => openEditor({ title: "Fix the bad answer", turns: messagesToTurns(row.messages), source: "fix", pairId: row.id, focusAnswer: true, onSaved: () => load() }) },
                })
              : null,
            el("button", {
              type: "button",
              class: "btn secondary btn-small nt-delete",
              text: "Delete",
              on: {
                click: async (event) => {
                  const button = event.currentTarget;
                  if (button.dataset.armed !== "1") {
                    button.dataset.armed = "1";
                    button.textContent = "Tap again to delete";
                    setTimeout(() => { button.dataset.armed = ""; button.textContent = "Delete"; }, 3000);
                    return;
                  }
                  const result = await call(`/owner/training/examples/${row.id}`, { method: "DELETE" });
                  if (result.ok) load();
                  else toast(result.data.detail || "Couldn't delete.", "err");
                },
              },
            }),
          ]),
        ]);
        list.append(item);
      });
    };

    const load = async () => {
      const result = await call(`/owner/training/examples?label=${filter}`);
      if (!result.ok) {
        counts.textContent = result.data.detail || "Couldn't load examples.";
        return;
      }
      const totals = result.data.counts || { good: 0, rejected: 0, total: 0 };
      counts.textContent = `${totals.good} good examples for training · ${totals.rejected} rejected (kept separate)`;
      rows = Array.isArray(result.data.examples) ? result.data.examples : [];
      drawFilters(totals);
      drawList();
    };
    search.addEventListener("input", drawList);
    await load();
  }

  /* -------------------------------------------------------- entry points */

  /** Owner-only 🧠 button next to the attach buttons: opens Add example. */
  function addMenuItems() {
    const left = document.querySelector("#chat-form .composer-left");
    if (!left || left.querySelector(".nt-menu-item")) return;
    const button = el("button", {
      type: "button",
      class: "icon-btn nt-menu-item",
      "aria-label": "Add training example (owner only)",
      title: "Add training example",
      text: "🧠",
    });
    button.addEventListener("click", openAdd);
    left.append(button);
  }

  function boot() {
    checkOwner(true);
    window.addEventListener("henad:account", () => checkOwner(true));
    const log = document.getElementById("chat-log");
    if (log) {
      // Replies restored from history or added by other code paths.
      new MutationObserver(() => {
        if (!isOwner) return;
        log.querySelectorAll(".message.assistant").forEach((message) => decorate(message));
      }).observe(log, { childList: true });
    }
  }

  window.henadTraining = { decorateReply, openAdd, openList, checkOwner };

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
