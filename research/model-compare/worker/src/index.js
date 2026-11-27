// Throwaway Worker used only with `wrangler dev --remote` for the Nous model comparison.
// POST /run {model, inputs, stream?} -> {ok, ms, ttfb_ms, result | error}
export default {
  async fetch(req, env) {
    if (req.method !== "POST") return new Response("ok");
    const { model, inputs, stream } = await req.json();
    const t0 = Date.now();
    try {
      if (stream) {
        const s = await env.AI.run(model, { ...inputs, stream: true });
        const reader = s.getReader();
        const dec = new TextDecoder();
        let raw = "";
        let ttfb = null; // first chunk of any kind
        let firstText = null; // first visible (non-reasoning) text
        let buf = "";
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          const now = Date.now() - t0;
          if (ttfb === null) ttfb = now;
          const chunk = dec.decode(value, { stream: true });
          raw += chunk;
          buf += chunk;
          let nl;
          while ((nl = buf.indexOf("\n")) >= 0) {
            const line = buf.slice(0, nl).trim();
            buf = buf.slice(nl + 1);
            if (firstText === null && line.startsWith("data: {")) {
              try {
                const j = JSON.parse(line.slice(6));
                const c = j.choices && j.choices[0] && j.choices[0].delta && j.choices[0].delta.content;
                if (typeof c === "string" && c.trim()) firstText = now;
              } catch {}
            }
          }
        }
        return Response.json({ ok: true, ms: Date.now() - t0, ttfb_ms: ttfb, first_text_ms: firstText, raw_tail: raw.slice(-600) });
      }
      const result = await env.AI.run(model, inputs);
      return Response.json({ ok: true, ms: Date.now() - t0, result });
    } catch (e) {
      return Response.json({ ok: false, ms: Date.now() - t0, error: String(e && e.message || e), status: e && e.status });
    }
  },
};
