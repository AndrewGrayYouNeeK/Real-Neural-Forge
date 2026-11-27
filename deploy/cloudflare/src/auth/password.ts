/**
 * Password hashing compatible with the container (src/accounts/passwords.py):
 * scrypt$N$r$p$<salt hex>$<digest hex>. Used only for password resets, which
 * run in the Worker. Plaintext is never stored or logged.
 */

import { scrypt } from "@noble/hashes/scrypt";
import { bytesToHex, randomBytes } from "./util";

const N = 2 ** 14;
const R = 8;
const P = 1;
const DKLEN = 32;
export const MIN_PASSWORD_CHARS = 10;
export const MAX_PASSWORD_CHARS = 200;

/** Same rules as validate_password in the container. */
export function validatePassword(password: unknown, email: string): string | null {
  if (typeof password !== "string") return "Password must be a string.";
  if (password.length < MIN_PASSWORD_CHARS) return `Password must be at least ${MIN_PASSWORD_CHARS} characters.`;
  if (password.length > MAX_PASSWORD_CHARS) return `Password must be at most ${MAX_PASSWORD_CHARS} characters.`;
  if (password.toLowerCase() === email.toLowerCase()) return "Password must be different from the email address.";
  return null;
}

export function hashPassword(password: string, salt: Uint8Array = randomBytes(16)): string {
  const digest = scrypt(new TextEncoder().encode(password), salt, { N, r: R, p: P, dkLen: DKLEN });
  return `scrypt$${N}$${R}$${P}$${bytesToHex(salt)}$${bytesToHex(digest)}`;
}
