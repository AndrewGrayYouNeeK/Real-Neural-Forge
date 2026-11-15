import { describe, expect, it } from "vitest";

import {
  createAlertRule,
  insertAlertEvent,
  listAlertEvents,
  listAlertRules,
  thresholdFired,
} from "../src/agent/alerts";

type Row = Record<string, unknown>;

function memoryDb() {
  const rules: Row[] = [];
  const events: Row[] = [];
  return {
    prepare(sql: string) {
      const text = sql.replace(/\s+/g, " ").trim();
      return {
        bind(...args: unknown[]) {
          return {
            async run() {
              if (text.startsWith("INSERT INTO alert_rules")) {
                rules.push({
                  id: args[0],
                  scope: args[1],
                  name: args[2],
                  series: args[3],
                  kind: args[4],
                  config_json: args[5],
                  enabled: 1,
                  created_at: args[6],
                  updated_at: args[7],
                });
              } else if (text.startsWith("INSERT INTO alert_events")) {
                events.push({
                  id: args[0],
                  scope: args[1],
                  rule_id: args[2],
                  fired_at: args[3],
                  payload_json: args[4],
                  acknowledged: 0,
                });
              } else if (text.startsWith("DELETE FROM alert_rules")) {
                // soft cap noop for tests
              } else if (text.startsWith("DELETE FROM alert_events")) {
                // soft cap / cascade noop
              }
              return { success: true, meta: { changes: 1 } };
            },
            async all() {
              if (text.includes("FROM alert_rules")) {
                return {
                  results: rules
                    .filter((row) => row.scope === args[0])
                    .slice(0, Number(args[1] ?? 40)),
                };
              }
              if (text.includes("FROM alert_events")) {
                return {
                  results: events
                    .filter((row) => row.scope === args[0])
                    .slice(0, Number(args[1] ?? 100)),
                };
              }
              return { results: [] };
            },
            async first() {
              return null;
            },
          };
        },
      };
    },
  };
}

describe("alerts", () => {
  it("stores rules and events", async () => {
    const db = memoryDb();
    const created = await createAlertRule(db as never, "ip:test", {
      name: "Hot",
      series: "weather",
      kind: "threshold",
      config: { op: ">", value: 30 },
    });
    expect(created.ok).toBe(true);
    if (!created.ok) return;
    await insertAlertEvent(db as never, "ip:test", created.rule.id, {
      message: "crossed",
    });
    const rules = await listAlertRules(db as never, "ip:test");
    const events = await listAlertEvents(db as never, "ip:test");
    expect(rules).toHaveLength(1);
    expect(events).toHaveLength(1);
  });

  it("evaluates threshold helper", () => {
    expect(thresholdFired(35, { op: ">", value: 30 }).fired).toBe(true);
    expect(thresholdFired(10, { op: ">", value: 30 }).fired).toBe(false);
  });
});
