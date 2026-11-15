import { describe, expect, it } from "vitest";

import { createReport, getReport, listReports } from "../src/agent/reports";

type Row = Record<string, unknown>;

function memoryDb() {
  const reports: Row[] = [];
  return {
    prepare(sql: string) {
      const text = sql.replace(/\s+/g, " ").trim();
      return {
        bind(...args: unknown[]) {
          return {
            async run() {
              if (text.startsWith("INSERT INTO public_reports")) {
                reports.push({
                  id: args[0],
                  scope: args[1],
                  series: args[2],
                  title: args[3],
                  snapshot_json: args[4],
                  created_at: args[5],
                });
                return { success: true };
              }
              if (text.startsWith("DELETE FROM public_reports")) {
                const scope = args[0];
                const limit = Number(args[2]);
                const kept = reports
                  .filter((row) => row.scope === scope)
                  .sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)))
                  .slice(0, limit)
                  .map((row) => row.id);
                for (let i = reports.length - 1; i >= 0; i -= 1) {
                  if (reports[i].scope === scope && !kept.includes(reports[i].id)) {
                    reports.splice(i, 1);
                  }
                }
                return { success: true };
              }
              return { success: true };
            },
            async first() {
              if (text.includes("WHERE id = ?")) {
                return reports.find((row) => row.id === args[0]) || null;
              }
              return null;
            },
            async all() {
              if (text.includes("WHERE scope = ?")) {
                const scope = args[0];
                const limit = Number(args[1] ?? 40);
                const rows = reports
                  .filter((row) => row.scope === scope)
                  .sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)))
                  .slice(0, limit);
                return { results: rows };
              }
              return { results: [] };
            },
          };
        },
      };
    },
  };
}

describe("public reports", () => {
  it("creates and fetches a shareable snapshot by id", async () => {
    const db = memoryDb();
    const created = await createReport(db as never, "ip:test", {
      series: "weather",
      title: "Weather report",
      snapshot: { v: 1, series: "weather", forecast: { name: "Central Park" } },
    });
    expect(created.ok).toBe(true);
    if (!created.ok) return;
    const loaded = await getReport(db as never, created.report.id);
    expect(loaded?.title).toBe("Weather report");
    expect(loaded?.snapshot.series).toBe("weather");
    const listed = await listReports(db as never, "ip:test");
    expect(listed).toHaveLength(1);
  });
});
