import { describe, expect, it } from "vitest";
import { authRequired, isPublicPath, legacyRedirect } from "../src/auth/gate";
import { capKey, mfaFresh } from "../src/auth/session";
import { base32Decode, base32Encode, totpAt, verifyTotp } from "../src/auth/totp";
import { b64urlEncode, sha256 } from "../src/auth/util";
import { derToRaw, rpIdFor, verifyAssertion, verifyRegistration } from "../src/auth/webauthn";

// --- tiny CBOR encoder for building WebAuthn fixtures -----------------------
function head(major: number, length: number): number[] {
  if (length < 24) return [(major << 5) | length];
  if (length < 256) return [(major << 5) | 24, length];
  return [(major << 5) | 25, length >> 8, length & 255];
}
type Enc = number | string | Uint8Array | Map<Enc, Enc>;
function cbor(value: Enc): number[] {
  if (typeof value === "number") return value >= 0 ? head(0, value) : head(1, -1 - value);
  if (typeof value === "string") {
    const bytes = new TextEncoder().encode(value);
    return [...head(3, bytes.length), ...bytes];
  }
  if (value instanceof Uint8Array) return [...head(2, value.length), ...value];
  const out = head(5, value.size);
  for (const [key, item] of value) out.push(...cbor(key), ...cbor(item));
  return out;
}

function b64urlToBytes(text: string): Uint8Array {
  const padded = text.replace(/-/g, "+").replace(/_/g, "/") + "=".repeat((4 - (text.length % 4)) % 4);
  return Uint8Array.from(atob(padded), (char) => char.charCodeAt(0));
}

function rawToDer(raw: Uint8Array): Uint8Array {
  const int = (bytes: Uint8Array) => {
    let value = [...bytes];
    while (value.length > 1 && value[0] === 0) value = value.slice(1);
    if (value[0] & 0x80) value = [0, ...value];
    return [0x02, value.length, ...value];
  };
  const body = [...int(raw.slice(0, 32)), ...int(raw.slice(32))];
  return new Uint8Array([0x30, body.length, ...body]);
}

async function authData(rpId: string, flags: number, counter: number, attested?: Uint8Array): Promise<Uint8Array> {
  const parts = [...(await sha256(new TextEncoder().encode(rpId))), flags, 0, 0, 0, counter];
  return new Uint8Array(attested ? [...parts, ...attested] : parts);
}

describe("TOTP", () => {
  it("matches the RFC 6238 SHA-1 vector", async () => {
    const secret = new TextEncoder().encode("12345678901234567890");
    // RFC 6238: T=59 → 94287082 (8 digits); 6-digit tail is 287082.
    expect(await totpAt(secret, 1)).toBe("287082");
    expect(base32Decode(base32Encode(secret))).toEqual(secret);
  });

  it("accepts the current code once and rejects replays", async () => {
    const secret = base32Encode(new TextEncoder().encode("12345678901234567890"));
    const now = new Date(59_000);
    const step = await verifyTotp(secret, "287082", 0, now);
    expect(step).toBe(1);
    expect(await verifyTotp(secret, "287082", step ?? 0, now)).toBeNull();
    expect(await verifyTotp(secret, "000000", 0, now)).toBeNull();
    expect(await verifyTotp(secret, "abc", 0, now)).toBeNull();
  });
});

describe("passkeys", () => {
  it("only offers passkeys on the henad hostnames", () => {
    expect(rpIdFor(new URL("https://henad.youneek.dev/app"))).toBe("youneek.dev");
    expect(rpIdFor(new URL("https://henad.youneek.xyz/"))).toBe("youneek.xyz");
    expect(rpIdFor(new URL("https://nous.youneek.dev/"))).toBeNull();
    expect(rpIdFor(new URL("https://forge.youneek.dev/"))).toBeNull();
    expect(rpIdFor(new URL("https://evil.example/"))).toBeNull();
  });

  it("registers an ES256 passkey and verifies a signed login", async () => {
    const rpId = "henad.youneek.dev";
    const origin = `https://${rpId}`;
    const pair = (await crypto.subtle.generateKey({ name: "ECDSA", namedCurve: "P-256" }, true, ["sign", "verify"])) as CryptoKeyPair;
    const jwk = (await crypto.subtle.exportKey("jwk", pair.publicKey)) as JsonWebKey;
    const cose = new Map<Enc, Enc>([
      [1, 2],
      [3, -7],
      [-1, 1],
      [-2, b64urlToBytes(jwk.x as string)],
      [-3, b64urlToBytes(jwk.y as string)],
    ]);
    const credId = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const attested = new Uint8Array([...new Uint8Array(16), 0, credId.length, ...credId, ...cbor(cose)]);
    const regAuth = await authData(rpId, 0x45, 0, attested);
    const attestationObject = new Uint8Array(cbor(new Map<Enc, Enc>([["fmt", "none"], ["attStmt", new Map()], ["authData", regAuth]])));
    const regChallenge = b64urlEncode(new Uint8Array(32).fill(7));
    const regClient = new TextEncoder().encode(JSON.stringify({ type: "webauthn.create", challenge: regChallenge, origin }));
    const credential = await verifyRegistration({
      clientDataJSON: b64urlEncode(regClient),
      attestationObject: b64urlEncode(attestationObject),
      challenge: regChallenge,
      origin,
      rpId,
    });
    expect(credential.id).toBe(b64urlEncode(credId));
    expect(credential.alg).toBe(-7);

    // Wrong origin is refused.
    await expect(
      verifyRegistration({
        clientDataJSON: b64urlEncode(regClient),
        attestationObject: b64urlEncode(attestationObject),
        challenge: regChallenge,
        origin: "https://forge.youneek.dev",
        rpId,
      }),
    ).rejects.toThrow();

    const loginChallenge = b64urlEncode(new Uint8Array(32).fill(9));
    const loginAuth = await authData(rpId, 0x05, 1);
    const loginClient = new TextEncoder().encode(JSON.stringify({ type: "webauthn.get", challenge: loginChallenge, origin }));
    const signed = new Uint8Array([...loginAuth, ...(await sha256(loginClient))]);
    const raw = new Uint8Array(await crypto.subtle.sign({ name: "ECDSA", hash: "SHA-256" }, pair.privateKey, signed));
    expect(derToRaw(rawToDer(raw))).toEqual(raw);
    const stored = { id: credential.id, publicKey: credential.publicKey, alg: credential.alg, signCount: 0 };
    const ok = await verifyAssertion({
      credential: stored,
      clientDataJSON: b64urlEncode(loginClient),
      authenticatorData: b64urlEncode(loginAuth),
      signature: b64urlEncode(rawToDer(raw)),
      challenge: loginChallenge,
      origin,
      rpId,
    });
    expect(ok.signCount).toBe(1);

    // Replayed counter and wrong challenge are refused.
    await expect(
      verifyAssertion({ ...{ credential: { ...stored, signCount: 1 } }, clientDataJSON: b64urlEncode(loginClient), authenticatorData: b64urlEncode(loginAuth), signature: b64urlEncode(rawToDer(raw)), challenge: loginChallenge, origin, rpId }),
    ).rejects.toThrow();
    await expect(
      verifyAssertion({ credential: stored, clientDataJSON: b64urlEncode(loginClient), authenticatorData: b64urlEncode(loginAuth), signature: b64urlEncode(rawToDer(raw)), challenge: "other", origin, rpId }),
    ).rejects.toThrow();
  });
});

describe("front door", () => {
  it("301s forge hosts to henad with path and query kept", () => {
    const moved = legacyRedirect(new Request("https://forge.youneek.dev/app?x=1"));
    expect(moved?.status).toBe(301);
    expect(moved?.headers.get("Location")).toBe("https://henad.youneek.dev/app?x=1");
    const xyz = legacyRedirect(new Request("https://forge.youneek.xyz/docs"));
    expect(xyz?.headers.get("Location")).toBe("https://henad.youneek.xyz/docs");
    expect(legacyRedirect(new Request("https://henad.youneek.dev/"))).toBeNull();
    expect(legacyRedirect(new Request("https://forge.youneek.dev/voice/stripe/webhook", { method: "POST" }))).toBeNull();
  });

  it("keeps landing, docs, health and auth public and gates tools", async () => {
    for (const path of ["/", "/app", "/health", "/docs", "/assets/app.js", "/auth/login", "/auth/me"]) {
      expect(isPublicPath("GET", path)).toBe(true);
    }
    expect(isPublicPath("GET", "/agent/reports/abc123")).toBe(true);
    for (const path of ["/agent/chat", "/live/ask", "/train", "/experiments", "/predict", "/agent/reports", "/auth/owner"]) {
      expect(isPublicPath("POST", path)).toBe(false);
    }
    const denied = authRequired();
    expect(denied.status).toBe(401);
    expect(await denied.json()).toMatchObject({ auth_required: true });
  });

  it("counts caps per account once signed in", () => {
    expect(capKey({ userId: "usr_1" }, "1.2.3.4")).toBe("acct:usr_1");
    expect(capKey(null, "1.2.3.4")).toBe("1.2.3.4");
  });

  it("expires the owner second factor after 12 hours", () => {
    const now = new Date("2026-11-26T20:00:00Z");
    expect(mfaFresh("2026-11-26T19:00:00+00:00", now)).toBe(true);
    expect(mfaFresh("2026-11-26T07:00:00+00:00", now)).toBe(false);
    expect(mfaFresh(null, now)).toBe(false);
  });
});
