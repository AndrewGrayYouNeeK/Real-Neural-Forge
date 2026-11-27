import { describe, expect, it } from "vitest";

import { STT_MODEL, STT_MAX_BYTES, cleanTranscript, transcribeAudio, toBase64 } from "../src/agent/transcribe";
import { resolveAiCaps } from "../src/live/ai";
import { SqliteD1 } from "./helpers/sqliteD1";

function wav(seconds: number): Uint8Array {
  return new Uint8Array(44 + Math.floor(seconds * 32000));
}

describe("Talk mode transcription", () => {
  it("sends the audio to Whisper and returns clean text", async () => {
    const db = new SqliteD1();
    const calls: Array<{ model: string; inputs: Record<string, unknown> }> = [];
    const ai = {
      async run(model: string, inputs: Record<string, unknown>) {
        calls.push({ model, inputs });
        return { text: "  What's your name?  " };
      },
    };
    const result = await transcribeAudio({ ai, db, ip: "1.2.3.4" }, wav(3));
    expect(result).toEqual({ status: 200, body: { text: "What's your name?" } });
    expect(calls[0].model).toBe(STT_MODEL);
    expect(typeof calls[0].inputs.audio).toBe("string");
    const spent = db.rows("SELECT neurons FROM ai_usage WHERE scope = 'global'")[0];
    expect(Number(spent.neurons)).toBeGreaterThan(0);
    expect(Number(spent.neurons)).toBeLessThan(10);
  });

  it("ignores tiny clips and rejects huge ones without calling the model", async () => {
    const db = new SqliteD1();
    const ai = { run: async () => { throw new Error("should not run"); } };
    expect((await transcribeAudio({ ai, db, ip: null }, new Uint8Array(100))).body).toEqual({ text: "" });
    expect((await transcribeAudio({ ai, db, ip: null }, new Uint8Array(STT_MAX_BYTES + 1))).status).toBe(413);
  });

  it("stops at the daily neuron cap with a friendly text-fallback note", async () => {
    const db = new SqliteD1();
    const caps = { ...resolveAiCaps(), safety: 2 };
    const ai = { run: async () => ({ text: "hello" }) };
    const result = await transcribeAudio({ ai, db, ip: null, caps }, wav(3));
    expect(result.status).toBe(429);
    expect(String(result.body.error)).toContain("keep going in text");
  });

  it("reports a model failure as a retryable error", async () => {
    const db = new SqliteD1();
    const ai = { run: async () => { throw new Error("boom"); } };
    expect((await transcribeAudio({ ai, db, ip: null }, wav(2))).status).toBe(502);
  });

  it("drops Whisper's made-up text for silence", () => {
    expect(cleanTranscript("Thank you.")).toBe("");
    expect(cleanTranscript("[BLANK_AUDIO]")).toBe("");
    expect(cleanTranscript("tell me a joke")).toBe("tell me a joke");
    expect(toBase64(new Uint8Array([104, 105]))).toBe("aGk=");
  });
});
