/** RFC 6238 TOTP (SHA-1, 6 digits, 30 s) with WebCrypto. */

const ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
export const TOTP_PERIOD = 30;

export function base32Encode(bytes: Uint8Array): string {
  let bits = 0;
  let value = 0;
  let out = "";
  for (const byte of bytes) {
    value = (value << 8) | byte;
    bits += 8;
    while (bits >= 5) {
      out += ALPHABET[(value >>> (bits - 5)) & 31];
      bits -= 5;
    }
  }
  if (bits > 0) out += ALPHABET[(value << (5 - bits)) & 31];
  return out;
}

export function base32Decode(text: string): Uint8Array {
  const clean = text.toUpperCase().replace(/[\s=-]+/g, "");
  let bits = 0;
  let value = 0;
  const out: number[] = [];
  for (const char of clean) {
    const index = ALPHABET.indexOf(char);
    if (index < 0) throw new Error("bad base32");
    value = (value << 5) | index;
    bits += 5;
    if (bits >= 8) {
      out.push((value >>> (bits - 8)) & 255);
      bits -= 8;
    }
  }
  return new Uint8Array(out);
}

export async function totpAt(secret: Uint8Array, step: number): Promise<string> {
  const counter = new Uint8Array(8);
  let rest = step;
  for (let index = 7; index >= 0; index -= 1) {
    counter[index] = rest & 255;
    rest = Math.floor(rest / 256);
  }
  const key = await crypto.subtle.importKey("raw", secret, { name: "HMAC", hash: "SHA-1" }, false, ["sign"]);
  const mac = new Uint8Array(await crypto.subtle.sign("HMAC", key, counter));
  const offset = mac[mac.length - 1] & 15;
  const code =
    (((mac[offset] & 127) << 24) | (mac[offset + 1] << 16) | (mac[offset + 2] << 8) | mac[offset + 3]) %
    1_000_000;
  return code.toString().padStart(6, "0");
}

/**
 * Returns the matching time step (±1 window), or null. Steps at or below
 * `lastStep` are rejected so a code cannot be replayed.
 */
export async function verifyTotp(
  secretBase32: string,
  code: string,
  lastStep: number,
  now: Date = new Date(),
): Promise<number | null> {
  const clean = code.replace(/\s+/g, "");
  if (!/^\d{6}$/.test(clean)) return null;
  let secret: Uint8Array;
  try {
    secret = base32Decode(secretBase32);
  } catch {
    return null;
  }
  const current = Math.floor(now.getTime() / 1000 / TOTP_PERIOD);
  for (const step of [current - 1, current, current + 1]) {
    if (step <= lastStep) continue;
    if ((await totpAt(secret, step)) === clean) return step;
  }
  return null;
}

export function otpauthUrl(secretBase32: string, account: string, issuer = "Henad"): string {
  const label = encodeURIComponent(`${issuer}:${account}`);
  return `otpauth://totp/${label}?secret=${secretBase32}&issuer=${encodeURIComponent(issuer)}&algorithm=SHA1&digits=6&period=${TOTP_PERIOD}`;
}
