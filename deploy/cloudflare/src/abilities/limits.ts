/**
 * Daily per-user counters for the chat abilities (files, images, reminders).
 *
 * They reuse the ai_usage table (migration 0003) with scopes such as
 * `files:user:<id>` and `files:public`, so no new table is needed and the
 * counters reset at 00:00 UTC like every other cap. Model spend for these
 * abilities still goes through tryReserve, so the 25,000 neurons/day global
 * safety cap (about $5/month) stays the hard ceiling.
 */

import { utcDay, type D1Like } from "../live/ai";

export type AbilityKind = "files" | "images" | "reminders";

export type AbilityCaps = { member: number; owner: number; global: number };

export const ABILITY_CAPS: Record<AbilityKind, AbilityCaps> = {
  files: { member: 10, owner: 50, global: 300 },
  images: { member: 5, owner: 30, global: 100 },
  reminders: { member: 10, owner: 50, global: 1000 },
};

function positiveInt(value: unknown, fallback: number): number {
  const parsed = Number(String(value ?? "").trim());
  if (!Number.isFinite(parsed) || parsed < 1) return fallback;
  return Math.floor(parsed);
}

/** Env overrides: NOUS_<KIND>_MEMBER_DAILY / _OWNER_DAILY / _GLOBAL_DAILY. */
export function abilityCaps(kind: AbilityKind, env: Record<string, unknown> = {}): AbilityCaps {
  const base = ABILITY_CAPS[kind];
  const prefix = `NOUS_${kind.toUpperCase()}`;
  return {
    member: positiveInt(env[`${prefix}_MEMBER_DAILY`], base.member),
    owner: positiveInt(env[`${prefix}_OWNER_DAILY`], base.owner),
    global: positiveInt(env[`${prefix}_GLOBAL_DAILY`], base.global),
  };
}

async function used(db: D1Like, day: string, scope: string): Promise<number> {
  const result = await db
    .prepare("SELECT requests FROM ai_usage WHERE day = ? AND scope = ?")
    .bind(day, scope)
    .all();
  return Number(result.results?.[0]?.requests || 0);
}

async function bump(db: D1Like, day: string, scope: string, max: number): Promise<boolean> {
  await db
    .prepare(
      "INSERT INTO ai_usage (day, scope, requests, neurons) VALUES (?, ?, 0, 0) ON CONFLICT(day, scope) DO NOTHING",
    )
    .bind(day, scope)
    .all();
  const updated = await db
    .prepare(
      "UPDATE ai_usage SET requests = requests + 1 WHERE day = ? AND scope = ? AND requests + 1 <= ?",
    )
    .bind(day, scope, max)
    .all();
  return (updated.meta?.changes || 0) > 0;
}

export type AbilityClaim =
  | { ok: true; used: number; limit: number }
  | { ok: false; reason: "user" | "global"; limit: number };

/** Count one use for this person today, or refuse when a cap is reached. */
export async function claimAbility(options: {
  db: D1Like;
  kind: AbilityKind;
  scope: string;
  owner: boolean;
  caps: AbilityCaps;
  now?: Date;
}): Promise<AbilityClaim> {
  const day = utcDay(options.now);
  const personal = `${options.kind}:${options.scope}`;
  const shared = `${options.kind}:public`;
  const limit = options.owner ? options.caps.owner : options.caps.member;
  if ((await used(options.db, day, personal)) + 1 > limit) {
    return { ok: false, reason: "user", limit };
  }
  if (!options.owner && (await used(options.db, day, shared)) + 1 > options.caps.global) {
    return { ok: false, reason: "global", limit };
  }
  if (!(await bump(options.db, day, personal, limit))) return { ok: false, reason: "user", limit };
  if (!options.owner && !(await bump(options.db, day, shared, options.caps.global))) {
    return { ok: false, reason: "global", limit };
  }
  return { ok: true, used: (await used(options.db, day, personal)), limit };
}

export async function abilityAllowance(options: {
  db: D1Like;
  kind: AbilityKind;
  scope: string;
  owner: boolean;
  caps: AbilityCaps;
  now?: Date;
}): Promise<{ used: number; limit: number; remaining: number }> {
  const day = utcDay(options.now);
  const limit = options.owner ? options.caps.owner : options.caps.member;
  const count = await used(options.db, day, `${options.kind}:${options.scope}`);
  return { used: count, limit, remaining: Math.max(0, limit - count) };
}

/** Short stable folder name for a scope in R2 (no raw ids in object keys). */
export async function scopeFolder(scope: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(`nous-scope:${scope}`));
  return [...new Uint8Array(digest)]
    .slice(0, 12)
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

export function newAbilityId(prefix: string): string {
  return `${prefix}_${crypto.randomUUID().replace(/-/g, "").slice(0, 20)}`;
}

export type R2Like = {
  put(key: string, value: ArrayBuffer | ArrayBufferView, options?: Record<string, unknown>): Promise<unknown>;
  get(key: string): Promise<{ body: ReadableStream; size: number; arrayBuffer?: () => Promise<ArrayBuffer>; text?: () => Promise<string> } | null>;
  delete(key: string): Promise<void>;
};
