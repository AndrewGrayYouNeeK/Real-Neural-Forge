/**
 * WebAuthn passkeys (ES256 / RS256) verified with WebCrypto. Attestation is
 * requested as "none": we trust the key the browser created on this origin,
 * which is the normal passkey setup. User verification is required.
 */

import { decodeCbor, type CborValue } from "./cbor";
import { b64urlDecode, b64urlEncode, bytesEqual, sha256 } from "./util";

export const PASSKEY_HOSTS = ["nous.youneek.dev", "nous.youneek.xyz"];

/** RP ID for this request, or null when passkeys are not offered on the host. */
export function rpIdFor(url: URL): string | null {
  const host = url.hostname.toLowerCase();
  if (PASSKEY_HOSTS.includes(host)) return host;
  if (host === "localhost" || host === "127.0.0.1") return host;
  return null;
}

export function expectedOrigin(url: URL): string {
  return url.origin;
}

export type StoredCredential = {
  id: string;
  publicKey: JsonWebKey;
  alg: number;
  signCount: number;
};

type AuthData = {
  rpIdHash: Uint8Array;
  flags: number;
  signCount: number;
  credentialId?: Uint8Array;
  publicKey?: Map<CborValue, CborValue>;
};

export function parseAuthData(bytes: Uint8Array): AuthData {
  if (bytes.length < 37) throw new Error("authenticator data too short");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out: AuthData = {
    rpIdHash: bytes.slice(0, 32),
    flags: bytes[32],
    signCount: view.getUint32(33),
  };
  if (out.flags & 0x40) {
    let offset = 37 + 16;
    if (bytes.length < offset + 2) throw new Error("attested data too short");
    const idLength = view.getUint16(offset);
    offset += 2;
    if (bytes.length < offset + idLength) throw new Error("credential id too short");
    out.credentialId = bytes.slice(offset, offset + idLength);
    offset += idLength;
    const decoded = decodeCbor(bytes.subarray(offset));
    if (!(decoded.value instanceof Map)) throw new Error("bad COSE key");
    out.publicKey = decoded.value;
  }
  return out;
}

function coseBytes(map: Map<CborValue, CborValue>, key: number): Uint8Array {
  const value = map.get(key);
  if (!(value instanceof Uint8Array)) throw new Error(`COSE key missing ${key}`);
  return value;
}

export function coseToJwk(cose: Map<CborValue, CborValue>): { jwk: JsonWebKey; alg: number } {
  const kty = cose.get(1);
  const alg = cose.get(3);
  if (kty === 2 && alg === -7) {
    if (cose.get(-1) !== 1) throw new Error("only P-256 passkeys are supported");
    return {
      jwk: { kty: "EC", crv: "P-256", x: b64urlEncode(coseBytes(cose, -2)), y: b64urlEncode(coseBytes(cose, -3)), ext: true },
      alg: -7,
    };
  }
  if (kty === 3 && alg === -257) {
    return {
      jwk: { kty: "RSA", n: b64urlEncode(coseBytes(cose, -1)), e: b64urlEncode(coseBytes(cose, -2)), alg: "RS256", ext: true },
      alg: -257,
    };
  }
  throw new Error("unsupported passkey algorithm");
}

async function importKey(jwk: JsonWebKey, alg: number): Promise<CryptoKey> {
  if (alg === -7) {
    return crypto.subtle.importKey("jwk", jwk, { name: "ECDSA", namedCurve: "P-256" }, false, ["verify"]);
  }
  return crypto.subtle.importKey("jwk", jwk, { name: "RSASSA-PKCS1-v1_5", hash: "SHA-256" }, false, ["verify"]);
}

/** DER ECDSA signature → raw r||s (64 bytes) for WebCrypto. */
export function derToRaw(der: Uint8Array): Uint8Array {
  if (der[0] !== 0x30) throw new Error("bad DER signature");
  let offset = 2;
  if (der[1] & 0x80) offset = 2 + (der[1] & 0x7f);
  const readInt = (): Uint8Array => {
    if (der[offset] !== 0x02) throw new Error("bad DER integer");
    const length = der[offset + 1];
    let value = der.slice(offset + 2, offset + 2 + length);
    offset += 2 + length;
    while (value.length > 32 && value[0] === 0) value = value.slice(1);
    if (value.length > 32) throw new Error("bad DER integer length");
    const padded = new Uint8Array(32);
    padded.set(value, 32 - value.length);
    return padded;
  };
  const r = readInt();
  const s = readInt();
  const out = new Uint8Array(64);
  out.set(r, 0);
  out.set(s, 32);
  return out;
}

type ClientData = { type?: string; challenge?: string; origin?: string; crossOrigin?: boolean };

function checkClientData(raw: Uint8Array, type: string, challenge: string, origin: string): void {
  const data = JSON.parse(new TextDecoder().decode(raw)) as ClientData;
  if (data.type !== type) throw new Error("wrong ceremony type");
  if (data.challenge !== challenge) throw new Error("challenge mismatch");
  if (data.origin !== origin) throw new Error("origin mismatch");
  if (data.crossOrigin) throw new Error("cross-origin passkey use is not allowed");
}

async function checkAuthData(auth: AuthData, rpId: string): Promise<void> {
  const expected = await sha256(new TextEncoder().encode(rpId));
  if (!bytesEqual(auth.rpIdHash, expected)) throw new Error("RP ID mismatch");
  if (!(auth.flags & 0x01)) throw new Error("user presence missing");
  if (!(auth.flags & 0x04)) throw new Error("user verification missing");
}

export async function verifyRegistration(input: {
  clientDataJSON: string;
  attestationObject: string;
  challenge: string;
  origin: string;
  rpId: string;
}): Promise<{ id: string; publicKey: JsonWebKey; alg: number; signCount: number }> {
  const clientData = b64urlDecode(input.clientDataJSON);
  checkClientData(clientData, "webauthn.create", input.challenge, input.origin);
  const attestation = decodeCbor(b64urlDecode(input.attestationObject)).value;
  if (!(attestation instanceof Map)) throw new Error("bad attestation object");
  const authBytes = attestation.get("authData");
  if (!(authBytes instanceof Uint8Array)) throw new Error("attestation has no authData");
  const auth = parseAuthData(authBytes);
  await checkAuthData(auth, input.rpId);
  if (!auth.credentialId || !auth.publicKey) throw new Error("no credential in attestation");
  const { jwk, alg } = coseToJwk(auth.publicKey);
  await importKey(jwk, alg);
  return { id: b64urlEncode(auth.credentialId), publicKey: jwk, alg, signCount: auth.signCount };
}

export async function verifyAssertion(input: {
  credential: StoredCredential;
  clientDataJSON: string;
  authenticatorData: string;
  signature: string;
  challenge: string;
  origin: string;
  rpId: string;
}): Promise<{ signCount: number }> {
  const clientData = b64urlDecode(input.clientDataJSON);
  checkClientData(clientData, "webauthn.get", input.challenge, input.origin);
  const authBytes = b64urlDecode(input.authenticatorData);
  const auth = parseAuthData(authBytes);
  await checkAuthData(auth, input.rpId);
  const clientHash = await sha256(clientData);
  const signed = new Uint8Array(authBytes.length + clientHash.length);
  signed.set(authBytes, 0);
  signed.set(clientHash, authBytes.length);
  const key = await importKey(input.credential.publicKey, input.credential.alg);
  let signature = b64urlDecode(input.signature);
  let ok: boolean;
  if (input.credential.alg === -7) {
    signature = derToRaw(signature);
    ok = await crypto.subtle.verify({ name: "ECDSA", hash: "SHA-256" }, key, signature, signed);
  } else {
    ok = await crypto.subtle.verify({ name: "RSASSA-PKCS1-v1_5" }, key, signature, signed);
  }
  if (!ok) throw new Error("bad passkey signature");
  if (auth.signCount !== 0 || input.credential.signCount !== 0) {
    if (auth.signCount <= input.credential.signCount) throw new Error("passkey counter went backwards");
  }
  return { signCount: auth.signCount };
}
