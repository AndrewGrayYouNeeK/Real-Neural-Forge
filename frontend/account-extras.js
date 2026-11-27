/* Nous account extras: forgot password, reset/confirm links, memory settings,
   and the owner-only usage dashboard. Kept apart from app.js on purpose.
   The server decides everything; this file only shows what it is allowed to. */
(function () {
  "use strict";

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
      else if (key === "on") Object.entries(value).forEach(([name, fn]) => node.addEventListener(name, fn));
      else node.setAttribute(key, value === true ? "" : value);
    });
    (children || []).forEach((child) => child && node.append(child));
    return node;
  }

  /* ------------------------------------------------------------ overlay */

  function overlay(title, wide) {
    const close = () => backdrop.remove();
    const body = el("div", { class: "nx-body" });
    const card = el("div", { class: `nx-card${wide ? " nx-wide" : ""}`, role: "dialog", "aria-modal": "true", "aria-label": title }, [
      el("div", { class: "nx-head" }, [
        el("h2", { text: title }),
        el("button", { class: "modal-close", type: "button", "aria-label": "Close", text: "✕", on: { click: close } }),
      ]),
      body,
    ]);
    const backdrop = el("div", { class: "nx-backdrop", on: { click: (event) => event.target === backdrop && close() } }, [card]);
    document.body.append(backdrop);
    return { body, close };
  }

  function message(box, text, kind) {
    box.textContent = text || "";
    box.className = `nx-msg${kind ? ` nx-${kind}` : ""}`;
    box.hidden = !text;
  }

  /* ---------------------------------------------------- forgot password */

  function openForgot() {
    const { body } = overlay("Reset password");
    const start = document.getElementById("sign-in-email");
    const email = el("input", { type: "email", placeholder: "Email", autocomplete: "username", value: (start && start.value) || "" });
    const note = el("p", { class: "nx-msg", hidden: true });
    const send = el("button", { class: "btn primary gate-btn", type: "submit", text: "Send reset link" });
    const form = el("form", { class: "nx-form" }, [
      el("p", { class: "gate-hint", text: "Enter your email and we'll send you a link to choose a new password." }),
      email,
      send,
      note,
    ]);
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      send.disabled = true;
      const result = await call("/auth/password/forgot", { method: "POST", body: JSON.stringify({ email: email.value }) });
      send.disabled = false;
      message(note, result.data.detail || "Something went wrong. Try again.", result.ok ? "ok" : "err");
      if (result.ok) send.hidden = true;
    });
    body.append(form);
    email.focus();
  }

  function addForgotLink() {
    const row = document.getElementById("gate-password-row");
    if (!row || document.getElementById("nx-forgot")) return;
    const link = el("button", { id: "nx-forgot", class: "linklike nx-forgot", type: "button", text: "Forgot password?", on: { click: openForgot } });
    row.append(link);
    const sync = () => {
      const label = document.getElementById("gate-mode-label");
      // Only on the log-in step, not while creating an account.
      link.hidden = row.hidden || Boolean(label && /have an account/i.test(label.textContent || ""));
    };
    new MutationObserver(sync).observe(row, { attributes: true, attributeFilter: ["hidden"] });
    const label = document.getElementById("gate-mode-label");
    if (label) new MutationObserver(sync).observe(label, { childList: true, characterData: true, subtree: true });
    sync();
  }

  /* ------------------------------------------------- links from emails */

  async function openReset(token) {
    const { body, close } = overlay("Choose a new password");
    const note = el("p", { class: "nx-msg", hidden: true });
    const check = await call("/auth/password/reset/check", { method: "POST", body: JSON.stringify({ token }) });
    if (!check.ok) {
      message(note, check.data.detail || "This link is used or expired. Ask for a new one.", "err");
      body.append(note, el("button", { class: "btn secondary gate-btn", type: "button", text: "Send a new link", on: { click: () => { close(); openForgot(); } } }));
      return;
    }
    const password = el("input", { type: "password", placeholder: "New password (10+ characters)", autocomplete: "new-password", minlength: "10" });
    const code = el("input", { type: "text", placeholder: "6-digit code or recovery code", autocomplete: "one-time-code", hidden: !check.data.needs_code });
    const save = el("button", { class: "btn primary gate-btn", type: "submit", text: "Save new password" });
    const form = el("form", { class: "nx-form" }, [password, code, save, note]);
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      save.disabled = true;
      const result = await call("/auth/password/reset", {
        method: "POST",
        body: JSON.stringify({ token, password: password.value, code: code.hidden ? undefined : code.value }),
      });
      save.disabled = false;
      if (result.data.needs_code) code.hidden = false;
      message(note, result.data.detail || (result.ok ? "Done." : "Something went wrong."), result.ok ? "ok" : "err");
      if (result.ok) {
        form.replaceChildren(note, el("a", { class: "btn primary gate-btn", href: "/app", text: "Log in" }));
      }
    });
    body.append(form);
    password.focus();
  }

  async function confirmEmail(token) {
    const { body } = overlay("Confirm email");
    const note = el("p", { class: "nx-msg", text: "Checking…" });
    body.append(note);
    const result = await call("/auth/email/verify", { method: "POST", body: JSON.stringify({ token }) });
    message(note, result.data.detail || (result.ok ? "Email confirmed." : "That link didn't work."), result.ok ? "ok" : "err");
  }

  function readLinkFromHash() {
    const match = window.location.hash.match(/^#(reset|verify)=([A-Za-z0-9_-]{20,100})$/);
    if (!match) return;
    // Drop the token from the address bar and history right away.
    window.history.replaceState(null, "", window.location.pathname + window.location.search);
    if (match[1] === "reset") openReset(match[2]);
    else confirmEmail(match[2]);
  }

  /* ----------------------------------------------------------- settings */

  function section(title, children) {
    return el("section", { class: "nx-section" }, [el("h3", { text: title }), ...children]);
  }

  async function emailSection() {
    const status = await call("/auth/email/status");
    // Confirming email is optional, and hidden until email sending works.
    if (!status.ok || !status.data.email_on) return null;
    if (status.data.verified) return section("Email", [el("p", { class: "hint", text: "Your email is confirmed ✓" })]);
    const note = el("p", { class: "nx-msg", hidden: true });
    const send = el("button", {
      class: "btn secondary btn-small",
      type: "button",
      text: "Send confirm link",
      on: {
        click: async () => {
          send.disabled = true;
          const result = await call("/auth/email/verify/send", { method: "POST", body: "{}" });
          send.disabled = false;
          message(note, result.data.detail, result.ok ? "ok" : "err");
        },
      },
    });
    return section("Email", [el("p", { class: "hint", text: "Your email isn't confirmed yet." }), send, note]);
  }

  const LABELS = {
    name: "Name",
    location: "Lives in",
    units: "Units",
    answer_style: "Likes",
    working_on: "Working on",
    learning: "Learning",
  };

  function factLabel(key) {
    return LABELS[key] || key.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());
  }

  async function memorySection() {
    const box = el("section", { class: "nx-section" });
    const render = async () => {
      const result = await call("/agent/memory");
      if (!result.ok || !result.data.signed_in) {
        box.hidden = true;
        return;
      }
      const facts = Array.isArray(result.data.facts) ? result.data.facts : [];
      const on = result.data.enabled !== false;
      const toggle = el("input", { type: "checkbox", role: "switch", checked: on });
      toggle.addEventListener("change", async () => {
        toggle.disabled = true;
        await call("/agent/memory/settings", { method: "PUT", body: JSON.stringify({ enabled: toggle.checked }) });
        await render();
      });
      const list = el("ul", { class: "nx-facts" });
      facts.forEach((fact) => {
        const forget = el("button", {
          class: "nx-x",
          type: "button",
          "aria-label": `Forget ${factLabel(fact.key)}`,
          text: "✕",
          on: {
            click: async () => {
              await call(`/agent/memory/${encodeURIComponent(fact.key)}`, { method: "DELETE" });
              await render();
            },
          },
        });
        list.append(el("li", {}, [el("span", {}, [el("b", { text: `${factLabel(fact.key)}: ` }), document.createTextNode(fact.value)]), forget]));
      });
      const clear = el("button", {
        class: "btn secondary btn-small",
        type: "button",
        text: "Forget everything",
        on: {
          click: async () => {
            if (!window.confirm("Forget everything Nous remembers about you?")) return;
            await call("/agent/memory", { method: "DELETE" });
            await render();
          },
        },
      });
      box.replaceChildren(
        el("h3", { text: "Memory" }),
        el("label", { class: "nx-switch" }, [toggle, el("span", { text: "Remember things about me" })]),
        el("p", {
          class: "hint",
          text: on
            ? "Nous remembers things you tell it, like your name or what you're working on, so later chats go faster. Only you can see this."
            : "Memory is off. Nous won't remember anything new or use what's saved.",
        }),
        facts.length ? list : el("p", { class: "hint", text: "Nothing saved yet." }),
        facts.length ? clear : null,
      );
      box.hidden = false;
    };
    await render();
    return box;
  }

  async function ownerSection() {
    const probe = await call("/owner/usage?days=30");
    if (!probe.ok) return null;
    return section("Owner", [
      el("div", { class: "nx-row" }, [
        el("button", { class: "btn secondary btn-small", type: "button", text: "Usage dashboard", on: { click: () => openDashboard(probe.data) } }),
        window.nousTraining
          ? el("button", { class: "btn secondary btn-small", type: "button", text: "Training examples", on: { click: () => window.nousTraining.openList() } })
          : null,
        window.nousTraining
          ? el("button", { class: "btn secondary btn-small", type: "button", text: "Add example", on: { click: () => window.nousTraining.openAdd() } })
          : null,
      ]),
    ]);
  }

  let rendering = false;
  async function renderSettings() {
    const modal = document.querySelector("#sign-in-modal .modal");
    const signedIn = document.getElementById("account-signed-in");
    if (!modal || rendering) return;
    let host = document.getElementById("nx-settings");
    if (!signedIn || signedIn.hidden) {
      if (host) host.remove();
      return;
    }
    rendering = true;
    try {
      if (!host) {
        host = el("div", { id: "nx-settings", class: "modal-body" });
        modal.append(host);
      }
      const parts = await Promise.all([emailSection(), memorySection(), ownerSection()]);
      host.replaceChildren(...parts.filter(Boolean));
    } catch {
      // Settings extras are optional; the rest of the modal keeps working.
    } finally {
      rendering = false;
    }
  }

  /* ---------------------------------------------------------- dashboard */

  function money(value) {
    return `$${Number(value || 0).toFixed(2)}`;
  }

  function shortDay(day) {
    const [, month, date] = day.split("-");
    return `${Number(month)}/${Number(date)}`;
  }

  function chart(title, days, field, format, totalText) {
    const values = days.map((day) => Number(day[field] || 0));
    const max = Math.max(1, ...values);
    const total = values.reduce((a, b) => a + b, 0);
    const bars = el("div", { class: "nx-bars", role: "img", "aria-label": `${title} per day` });
    days.forEach((day, index) => {
      const value = values[index];
      bars.append(
        el("div", { class: "nx-bar", title: `${shortDay(day.day)}: ${format ? format(value) : value}` }, [
          el("span", { style: `height:${Math.max(value ? 4 : 0, Math.round((value / max) * 100))}%` }),
        ]),
      );
    });
    return el("div", { class: "nx-chart" }, [
      el("div", { class: "nx-chart-head" }, [el("span", { text: title }), el("b", { text: totalText || (format ? format(total) : String(total)) })]),
      bars,
      el("div", { class: "nx-axis" }, [el("span", { text: shortDay(days[0].day) }), el("span", { text: "today" })]),
    ]);
  }

  function tile(label, value) {
    return el("div", { class: "nx-tile" }, [el("b", { text: String(value) }), el("span", { text: label })]);
  }

  function openDashboard(initial) {
    const { body } = overlay("Usage", true);
    const draw = (data) => {
      const days = data.days || [];
      const subs = data.subscriptions || {};
      const totals = data.totals || {};
      const email = data.email || {};
      const emailNote = el("p", { class: "nx-msg", hidden: true });
      const emailText =
        email.state === "on" ? "Email is on." : email.state === "down" ? "Email stopped working. Check the setup, then send a test." : "Email is off.";
      const emailButtons = [];
      if (email.configured) {
        emailButtons.push(
          el("button", {
            class: "btn secondary btn-small",
            type: "button",
            text: email.state === "on" ? "Send test email" : "Send test & turn on",
            on: {
              click: async (event) => {
                event.target.disabled = true;
                const result = await call("/owner/email", { method: "POST", body: JSON.stringify({ action: "test" }) });
                message(emailNote, result.data.detail || "", result.ok ? "ok" : "err");
                const fresh = await call("/owner/usage?days=30");
                if (fresh.ok) setTimeout(() => draw(fresh.data), 1500);
              },
            },
          }),
        );
        if (email.state === "on") {
          emailButtons.push(
            el("button", {
              class: "btn secondary btn-small",
              type: "button",
              text: "Turn off",
              on: {
                click: async () => {
                  await call("/owner/email", { method: "POST", body: JSON.stringify({ action: "off" }) });
                  const fresh = await call("/owner/usage?days=30");
                  if (fresh.ok) draw(fresh.data);
                },
              },
            }),
          );
        }
      }
      body.replaceChildren(
        el("p", { class: "hint", text: "Last 30 days. Counts only — no one's personal info." }),
        el("div", { class: "nx-tiles" }, [
          tile("accounts", totals.accounts || 0),
          tile("active this week", totals.active_7d || 0),
          tile("Plus subscribers", subs.active || 0),
          tile("new subscribers", subs.new_in_range || 0),
          tile("canceled", (subs.canceled_in_range || 0) + (subs.canceling ? ` (+${subs.canceling} ending)` : "")),
          tile("AI + voice cost (rough)", money((totals.ai_cost_usd || 0) + (totals.voice_cost_usd || 0))),
        ]),
        days.length ? chart("New sign-ups", days, "signups") : null,
        days.length ? chart("Active people", days, "active_users", null, `${totals.active_30d || 0} different people`) : null,
        days.length ? chart("Messages", days, "messages") : null,
        days.length ? chart("New chats", days, "chats") : null,
        days.length ? chart("Talk replies", days, "voice_replies") : null,
        days.length ? chart("AI cost (rough)", days, "ai_cost_usd", money) : null,
        el("section", { class: "nx-section" }, [el("h3", { text: "Email" }), el("p", { class: "hint", text: emailText }), el("div", { class: "nx-row" }, emailButtons), emailNote]),
        el("ul", { class: "nx-notes" }, (data.notes || []).map((note) => el("li", { text: note }))),
      );
    };
    draw(initial);
  }

  /* --------------------------------------------------------------- boot */

  function boot() {
    addForgotLink();
    readLinkFromHash();
    window.addEventListener("hashchange", readLinkFromHash);
    const modal = document.getElementById("sign-in-modal");
    if (modal) {
      new MutationObserver(() => {
        if (!modal.hidden) renderSettings();
      }).observe(modal, { attributes: true, attributeFilter: ["hidden"] });
    }
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
