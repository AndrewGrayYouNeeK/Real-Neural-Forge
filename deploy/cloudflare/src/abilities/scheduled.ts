/**
 * Worker cron (every minute, see wrangler.jsonc "triggers"):
 *  - fire due reminders and check-ins
 *  - once an hour, delete attached files and reminder rows past 30 days
 */

import type { D1Like } from "../live/ai";
import { appendMessage } from "../agent/chats";
import { cleanupOldFiles } from "./files";
import type { R2Like } from "./limits";
import { cleanupReminders, fireDueReminders } from "./reminders";

export async function runScheduledAbilities(env: Record<string, unknown>, now = Date.now()): Promise<void> {
  const db = env.WORKSPACES_DB as D1Like | undefined;
  if (!db) return;
  try {
    const fired = await fireDueReminders(db, env, async (threadId, text) => {
      await appendMessage(db, threadId, "assistant", text);
    }, now);
    if (fired) console.log(JSON.stringify({ event: "reminders.fired", count: fired }));
  } catch (error) {
    console.warn(`reminders cron failed: ${String((error as Error)?.message || error).slice(0, 200)}`);
  }
  if (new Date(now).getUTCMinutes() === 7) {
    try {
      const bucket = env.WORKSPACES as R2Like | undefined;
      if (bucket) await cleanupOldFiles(db, bucket, new Date(now));
      await cleanupReminders(db, now);
    } catch (error) {
      console.warn(`cleanup cron failed: ${String((error as Error)?.message || error).slice(0, 200)}`);
    }
  }
}
