/** A real SQLite database behind the D1 interface, with every migration applied. */

import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";

import type { D1Like } from "../../src/live/ai";

type Row = Record<string, unknown>;

export class SqliteD1 implements D1Like {
  readonly sqlite: DatabaseSync;

  constructor() {
    this.sqlite = new DatabaseSync(":memory:");
    const dir = join(__dirname, "..", "..", "migrations");
    for (const file of readdirSync(dir).filter((name) => name.endsWith(".sql")).sort()) {
      this.sqlite.exec(readFileSync(join(dir, file), "utf8"));
    }
  }

  prepare(sql: string) {
    const exec = (values: unknown[]) => {
      const statement = this.sqlite.prepare(sql);
      const params = values.map((value) => (typeof value === "boolean" ? Number(value) : value)) as never[];
      if (/^\s*(SELECT|WITH)\b/i.test(sql)) {
        return { results: statement.all(...params) as Row[], meta: { changes: 0 } };
      }
      const info = statement.run(...params);
      return { results: [] as Row[], meta: { changes: Number(info.changes) } };
    };
    const bound = (values: unknown[]) => ({
      all: async () => exec(values),
      first: async <T = Row>() => ((exec(values).results[0] as T | undefined) ?? null),
      run: async () => exec(values),
    });
    return { bind: (...values: unknown[]) => bound(values), all: async () => exec([]), run: async () => exec([]) };
  }

  rows(sql: string, ...values: unknown[]): Row[] {
    return this.sqlite.prepare(sql).all(...(values as never[])) as Row[];
  }

  exec(sql: string, ...values: unknown[]): void {
    this.sqlite.prepare(sql).run(...(values as never[]));
  }
}
