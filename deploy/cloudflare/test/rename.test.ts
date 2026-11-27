import { describe, expect, it } from "vitest";

import { legacyRedirect } from "../src/auth/gate";
import { handleHandoff, legacyHandoffPage } from "../src/auth/handoff";
import { sha256Hex } from "../src/auth/util";
import { rpIdFor } from "../src/auth/webauthn";
import { sayHenad } from "../src/agent/protocol";
import { cleanTranscript, normalizeName } from "../src/agent/transcribe";
import { SqliteD1 } from "./helpers/sqliteD1";

const TOKEN = "old-session-token-0123456789abcdefghijklmnop";

async function seed(db: SqliteD1, mfa = true) {
  db.exec("INSERT INTO users (id, email, password_hash, role, created_at) VALUES ('u1', 'u1@example.com', NULL, 'owner', ?)", new Date().toISOString());
  const hash = await sha256Hex(TOKEN);
  db.exec(
    "INSERT INTO sessions (token_hash, user_id, expires_at, created_at) VALUES (?, 'u1', ?, ?)",
    hash,
    new Date(Date.now() + 86_400_000).toISOString(),
    new Date().toISOString(),
  );
  if (mfa) db.exec("INSERT INTO session_mfa (token_hash, user_id, method, verified_at) VALUES (?, 'u1', 'totp', ?)", hash, new Date().toISOString());
}

function pageRequest(url: string, cookie?: string) {
  return new Request(url, { headers: { Accept: "text/html,application/xhtml+xml", ...(cookie ? { Cookie: cookie } : {}) } });
}

describe("old hosts move to henad.*", () => {
  it("301s nous.* and forge.* to the matching henad.* host with path and query", () => {
    for (const [from, to] of [
      ["https://nous.youneek.dev/app?x=1&y=2", "https://henad.youneek.dev/app?x=1&y=2"],
      ["https://nous.youneek.xyz/docs", "https://henad.youneek.xyz/docs"],
      ["https://forge.youneek.dev/?q=hi", "https://henad.youneek.dev/?q=hi"],
      ["https://forge.youneek.xyz/assets/app.js", "https://henad.youneek.xyz/assets/app.js"],
    ]) {
      const res = legacyRedirect(new Request(from))!;
      expect(res.status).toBe(301);
      expect(res.headers.get("Location")).toBe(to);
      expect(res.headers.get("Cache-Control")).toBe("no-store");
    }
  });

  it("serves API calls from installed apps and the Stripe webhook in place, and leaves henad.* alone", () => {
    expect(legacyRedirect(new Request("https://nous.youneek.dev/agent/chat", { method: "POST", body: "{}" }))).toBeNull();
    expect(legacyRedirect(new Request("https://nous.youneek.dev/auth/login", { method: "POST", body: "{}" }))).toBeNull();
    expect(legacyRedirect(new Request("https://nous.youneek.dev/auth/me", { headers: { Authorization: "Bearer abc" } }))).toBeNull();
    expect(legacyRedirect(new Request("https://nous.youneek.xyz/voice/stripe/webhook", { method: "POST", body: "{}" }))).toBeNull();
    expect(legacyRedirect(new Request("https://henad.youneek.dev/app"))).toBeNull();
  });

  it("uses the parent domain as the passkey RP ID on henad.*", () => {
    expect(rpIdFor(new URL("https://henad.youneek.dev/"))).toBe("youneek.dev");
    expect(rpIdFor(new URL("https://henad.youneek.xyz/"))).toBe("youneek.xyz");
    expect(rpIdFor(new URL("https://forge.youneek.xyz/"))).toBeNull();
  });
});

describe("signed-in hand-off from an old host", () => {
  it("only offers the hand-off page to a signed-in browser visit", async () => {
    const db = new SqliteD1();
    await seed(db);
    expect(await legacyHandoffPage(pageRequest("https://nous.youneek.dev/app"), { WORKSPACES_DB: db })).toBeNull();
    expect(await legacyHandoffPage(pageRequest("https://nous.youneek.dev/app", "nf_session=wrong"), { WORKSPACES_DB: db })).toBeNull();
    expect(await legacyHandoffPage(new Request("https://nous.youneek.dev/health", { headers: { Cookie: `nf_session=${TOKEN}` } }), { WORKSPACES_DB: db })).toBeNull();
    expect(await legacyHandoffPage(pageRequest("https://henad.youneek.dev/app", `nf_session=${TOKEN}`), { WORKSPACES_DB: db })).toBeNull();
  });

  it("carries the session, the second factor and local settings to henad.*, once", async () => {
    const db = new SqliteD1();
    await seed(db);
    const pageRes = await legacyHandoffPage(pageRequest("https://nous.youneek.dev/app?tab=1", `nf_session=${TOKEN}`), { WORKSPACES_DB: db });
    expect(pageRes!.status).toBe(200);
    expect(pageRes!.headers.get("Cache-Control")).toBe("no-store");
    expect(pageRes!.headers.get("Content-Security-Policy")).toContain("form-action https://henad.youneek.dev");
    const html = await pageRes!.text();
    expect(html).toContain('action="https://henad.youneek.dev/auth/handoff"');
    const code = /name="code" value="([^"]+)"/.exec(html)![1];
    expect(html).toContain('name="next" value="/app?tab=1"');

    const post = (origin: string, codeValue = code) =>
      new Request("https://henad.youneek.dev/auth/handoff", {
        method: "POST",
        headers: { Origin: origin, "Content-Type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({
          code: codeValue,
          next: "/app?tab=1",
          state: JSON.stringify({ nousVoicePick: "female", neuralForgeTalkPref: "on", evil: "x" }),
        }).toString(),
      });

    // Another site can't post a code (login CSRF).
    const forged = await handleHandoff(post("https://evil.example"), { WORKSPACES_DB: db });
    expect(forged!.status).toBe(303);
    expect(forged!.headers.get("Set-Cookie")).toBeNull();

    const ok = await handleHandoff(post("https://nous.youneek.dev"), { WORKSPACES_DB: db });
    expect(ok!.status).toBe(200);
    const cookie = ok!.headers.get("Set-Cookie")!;
    expect(cookie).toMatch(/^nf_session=[A-Za-z0-9_-]{40,};/);
    expect(cookie).toContain("HttpOnly");
    expect(cookie).toContain("Secure");
    expect(cookie).toContain("SameSite=Strict");
    expect(cookie).not.toContain("Domain=");
    const body = await ok!.text();
    expect(body).toContain('"henadVoicePick":"female"');
    expect(body).toContain('"neuralForgeTalkPref":"on"');
    expect(body).not.toContain("evil");
    expect(body).toContain('"next":"/app?tab=1"');

    const newToken = /^nf_session=([^;]+);/.exec(cookie)![1];
    const rows = db.rows(
      "SELECT sessions.user_id AS user_id, session_mfa.method AS method FROM sessions LEFT JOIN session_mfa ON session_mfa.token_hash = sessions.token_hash WHERE sessions.token_hash = ?",
      await sha256Hex(newToken),
    );
    expect(rows).toEqual([{ user_id: "u1", method: "totp" }]);

    // The code works once.
    const again = await handleHandoff(post("https://nous.youneek.dev"), { WORKSPACES_DB: db });
    expect(again!.status).toBe(303);
  });

  it("rejects an expired code and unsafe next paths", async () => {
    const db = new SqliteD1();
    await seed(db, false);
    const earlier = new Date(Date.now() - 10 * 60_000);
    const pageRes = await legacyHandoffPage(pageRequest("https://forge.youneek.xyz/", `nf_session=${TOKEN}`), { WORKSPACES_DB: db }, earlier);
    const code = /name="code" value="([^"]+)"/.exec(await pageRes!.text())![1];
    const res = await handleHandoff(
      new Request("https://henad.youneek.xyz/auth/handoff", {
        method: "POST",
        headers: { Origin: "https://forge.youneek.xyz", "Content-Type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({ code, next: "//evil.example/" }).toString(),
      }),
      { WORKSPACES_DB: db },
    );
    expect(res!.status).toBe(303);

    const fresh = await legacyHandoffPage(pageRequest("https://forge.youneek.xyz/", `nf_session=${TOKEN}`), { WORKSPACES_DB: db });
    const code2 = /name="code" value="([^"]+)"/.exec(await fresh!.text())![1];
    const ok = await handleHandoff(
      new Request("https://henad.youneek.xyz/auth/handoff", {
        method: "POST",
        headers: { Origin: "https://forge.youneek.xyz", "Content-Type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({ code: code2, next: "//evil.example/" }).toString(),
      }),
      { WORKSPACES_DB: db },
    );
    expect(ok!.status).toBe(200);
    expect(await ok!.text()).toContain('"next":"/"');
  });
});

describe("old names in chats and speech-to-text", () => {
  it("rewrites Nous and Neural Forge to Henad but leaves Nous Research alone", () => {
    expect(sayHenad("I'm Nous, formerly Real Neural Forge.")).toBe("I'm Henad, formerly Henad.");
    expect(sayHenad("NOUS can help")).toBe("Henad can help");
    expect(sayHenad("Nous Research trained Hermes.")).toBe("Nous Research trained Hermes.");
    expect(sayHenad("nous sommes ici")).toBe("nous sommes ici");
    expect(sayHenad("synchronous code")).toBe("synchronous code");
  });

  it("maps misheard names to Henad in Talk mode", () => {
    expect(normalizeName("Hen ad, what time is it?")).toBe("Henad, what time is it?");
    expect(normalizeName("hey henna what's the weather")).toBe("hey Henad what's the weather");
    expect(normalizeName("Hennad tell me a joke")).toBe("Henad tell me a joke");
    expect(normalizeName("hen-add is cool")).toBe("Henad is cool");
    expect(normalizeName("I dyed my hair with henna")).toBe("I dyed my hair with henna");
    expect(cleanTranscript("  Hen add,   who made you? ")).toBe("Henad, who made you?");
  });
});
