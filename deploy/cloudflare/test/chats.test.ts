import { describe, expect, it } from "vitest";

import type { D1Like } from "../src/live/ai";
import {
  appendMessage,
  createThread,
  deleteThread,
  ensureThreadForTurn,
  getThreadOwned,
  historyFromMessages,
  listMessages,
  listThreads,
} from "../src/agent/chats";

type ThreadRow = {
  id: string;
  scope: string;
  title: string;
  created_at: string;
  updated_at: string;
};
type MessageRow = {
  id: string;
  thread_id: string;
  role: string;
  content: string;
  created_at: string;
};

class ChatTableD1 implements D1Like {
  threads = new Map<string, ThreadRow>();
  messages = new Map<string, MessageRow>();

  prepare(sql: string) {
    const run = (values: unknown[]) => this.exec(sql, values);
    return {
      bind: (...values: unknown[]) => ({
        all: async () => run(values),
        first: async () => (await run(values)).results?.[0] ?? null,
        run: async () => run(values),
      }),
      all: async () => run([]),
      run: async () => run([]),
    };
  }

  private exec(sql: string, values: unknown[]) {
    const text = sql.replace(/\s+/g, " ").trim();
    if (text.startsWith("SELECT id, title, created_at, updated_at FROM chat_threads WHERE scope = ?")) {
      const [scope, limit] = values as [string, number];
      const rows = [...this.threads.values()]
        .filter((row) => row.scope === scope)
        .sort((a, b) => b.updated_at.localeCompare(a.updated_at))
        .slice(0, Number(limit) || 40)
        .map(({ id, title, created_at, updated_at }) => ({ id, title, created_at, updated_at }));
      return { results: rows, meta: { changes: 0 } };
    }
    if (text.startsWith("SELECT id, title, created_at, updated_at FROM chat_threads WHERE id = ? AND scope = ?")) {
      const [id, scope] = values as string[];
      const row = this.threads.get(id);
      if (!row || row.scope !== scope) return { results: [], meta: { changes: 0 } };
      return {
        results: [{ id: row.id, title: row.title, created_at: row.created_at, updated_at: row.updated_at }],
        meta: { changes: 0 },
      };
    }
    if (text.startsWith("SELECT id, role, content, created_at FROM chat_messages")) {
      const [threadId] = values as string[];
      const rows = [...this.messages.values()]
        .filter((row) => row.thread_id === threadId)
        .sort((a, b) => a.created_at.localeCompare(b.created_at))
        .map(({ id, role, content, created_at }) => ({ id, role, content, created_at }));
      return { results: rows, meta: { changes: 0 } };
    }
    if (text.startsWith("INSERT INTO chat_threads")) {
      const [id, scope, title, created, updated] = values as string[];
      this.threads.set(id, { id, scope, title, created_at: created, updated_at: updated });
      return { results: [], meta: { changes: 1 } };
    }
    if (text.startsWith("UPDATE chat_threads SET updated_at")) {
      const [updated, id] = values as string[];
      const row = this.threads.get(id);
      if (row) row.updated_at = updated;
      return { results: [], meta: { changes: row ? 1 : 0 } };
    }
    if (text.startsWith("INSERT INTO chat_messages")) {
      const [id, threadId, role, content, created] = values as string[];
      this.messages.set(id, { id, thread_id: threadId, role, content, created_at: created });
      return { results: [], meta: { changes: 1 } };
    }
    if (text.startsWith("DELETE FROM chat_messages WHERE thread_id = ?")) {
      const [threadId] = values as string[];
      let changes = 0;
      for (const [id, row] of this.messages) {
        if (row.thread_id === threadId) {
          this.messages.delete(id);
          changes += 1;
        }
      }
      return { results: [], meta: { changes } };
    }
    if (text.startsWith("DELETE FROM chat_threads WHERE id = ? AND scope = ?")) {
      const [id, scope] = values as string[];
      const row = this.threads.get(id);
      if (!row || row.scope !== scope) return { results: [], meta: { changes: 0 } };
      this.threads.delete(id);
      return { results: [], meta: { changes: 1 } };
    }
    throw new Error(`unsupported sql: ${text}`);
  }
}

describe("persistent chat threads", () => {
  it("creates, lists, restores, and deletes threads for a scope", async () => {
    const db = new ChatTableD1();
    const thread = await createThread(db, "owner", "What is the weather?");
    expect(thread.title).toBe("What is the weather?");
    await appendMessage(db, thread.id, "user", "What is the weather?");
    await appendMessage(db, thread.id, "assistant", "High about 72°F.");
    const listed = await listThreads(db, "owner");
    expect(listed).toHaveLength(1);
    expect(listed[0].id).toBe(thread.id);
    const messages = await listMessages(db, thread.id);
    expect(messages.map((row) => row.role)).toEqual(["user", "assistant"]);
    const history = historyFromMessages(messages);
    expect(history).toEqual([{ role: "user", content: "What is the weather?" }]);
    expect(await getThreadOwned(db, "ip:other", thread.id)).toBeNull();
    expect(await deleteThread(db, "owner", thread.id)).toBe(true);
    expect(await listThreads(db, "owner")).toEqual([]);
  });

  it("reuses an owned thread id across turns", async () => {
    const db = new ChatTableD1();
    const first = await ensureThreadForTurn(db, "user:a", null, "Moon phase");
    const again = await ensureThreadForTurn(db, "user:a", first.id, "And tomorrow?");
    expect(again.id).toBe(first.id);
    const stranger = await ensureThreadForTurn(db, "user:b", first.id, "Hello");
    expect(stranger.id).not.toBe(first.id);
  });
});

import { normalizeChatContent as keepLines } from "../src/agent/chats";

describe("saved chat text", () => {
  it("keeps line breaks and indentation so code blocks survive a reopened chat", () => {
    const text = "Here:\r\n```python\nfor i in range(3):\n    print(i)   \n```\n\n\n\nDone.";
    expect(keepLines(text)).toBe("Here:\n```python\nfor i in range(3):\n    print(i)\n```\n\nDone.");
    expect(keepLines("   ")).toBeNull();
  });
});
