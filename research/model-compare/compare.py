#!/usr/bin/env python3
"""Nous model comparison runner.
Talks to a throwaway Worker (wrangler dev --remote, AI binding) at http://localhost:8799.
Usage: python3 compare.py [--only model-substr] [--think]
Writes results/<model>.json then merged results.json.
"""
import json, os, re, sys, time, urllib.request, concurrent.futures as cf

URL = "http://localhost:8799"
HERE = os.path.dirname(os.path.abspath(__file__))
SYSTEM = open(os.path.join(HERE, "system_prompt.txt")).read()
TEMP = 0.85          # production NORMAL_TEMPERATURE
MAX_TOKENS = 500     # production uses 160; raised so hard questions are not cut off
NEURON_USD = 0.011 / 1000

# name -> (model id, extra inputs that keep hidden thinking off/low, uses responses API like production)
MODELS = {
  "gpt-oss-120b":        ("@cf/openai/gpt-oss-120b", {}, True),
  "gpt-oss-20b":         ("@cf/openai/gpt-oss-20b", {}, True),
  "deepseek-v4.1-flash": ("@cf/deepseek-ai/deepseek-v4.1-flash-1017", {"chat_template_kwargs": {"thinking": False}}, False),
  "glm-5.3":             ("@cf/zai-org/glm-5.3", {"reasoning_effort": "low"}, False),
  "qwen3.8-27b":         ("@cf/qwen/qwen3.8-27b", {"chat_template_kwargs": {"enable_thinking": False}}, False),
  "gemma-4-26b":         ("@cf/google/gemma-4-26b-a4b-it", {"chat_template_kwargs": {"enable_thinking": False}}, False),
  "mistral-small-3.1":   ("@cf/mistralai/mistral-small-3.1-24b-instruct", {}, False),
  "phonellm-mini":       ("@cf/daily-co/phonellm-mini-alpha-2", {}, False),
  "llama-4-scout":       ("@cf/meta/llama-4-scout-17b-16e-instruct", {}, False),
  "llama-3.3-70b-fast":  ("@cf/meta/llama-3.3-70b-instruct-fp8-fast", {}, False),
  "llama-3.1-8b-fast":   ("@cf/meta/llama-3.1-8b-instruct-fp8-fast", {}, False),
  "inkling-2-small":     ("@cf/thinkingmachines/inkling-2-small", {"reasoning_effort": "low"}, False),
  "mistral-7b-lora":     ("@cf/mistral/mistral-7b-instruct-v0.2-lora", {}, False),
}
# Retired on 2026-11-20 (HTTP 410 / error 5028): probed once, not benchmarked.
RETIRED = ["@cf/deepseek-ai/deepseek-v4-pro-0813", "@cf/deepseek-ai/deepseek-v4-flash-0731", "@cf/moonshotai/kimi-k2.6",
           "@cf/zai-org/glm-5.3-flash", "@cf/nvidia/nemotron-3-120b-a12b", "@cf/google/gemma-7b-it-lora"]

PROMPTS = [
 ("creator-1", "Creator", "Who made you?"),
 ("creator-2", "Creator", "Who owns you, and what company is that?"),
 ("creator-3", "Creator", "Are you ChatGPT or Grok? Who actually built you?"),
 ("joke", "Funny", "Tell me a joke."),
 ("roast", "Funny", "Roast me. I'm a guy who has been 'almost done' with the same side project since 2019."),
 ("casual-hey", "Casual", "hey"),
 ("casual-bored", "Casual", "what's up, I'm bored out of my mind"),
 ("casual-stress", "Casual", "I'm really stressed, I have a big job interview tomorrow and I can't sleep."),
 ("talk-capital", "Talk", "Quick, what's the capital of Australia?"),
 ("talk-egg", "Talk", "How long do I boil an egg? Keep it short, I'm hearing this out loud."),
 ("talk-intro", "Talk", "What's your name and what can you do for me?"),
 ("math-train", "Math/logic", "A train leaves at 3:15 pm going 60 mph. A second train leaves the same station at 4:00 pm going 80 mph on the same track, same direction. What time does the second one catch the first?"),
 ("math-percent", "Math/logic", "What is 17% of 2,350, plus 48?"),
 ("logic-syllogism", "Math/logic", "All bloops are razzies, and some razzies are lazzies. Can we be sure that some bloops are lazzies?"),
 ("logic-batball", "Math/logic", "A bat and a ball cost $1.10 together. The bat costs $1.00 more than the ball. How much is the ball?"),
 ("code-palindrome", "Coding", "Write a Python function that checks if a string is a palindrome, ignoring spaces, punctuation and capital letters."),
 ("code-js", "Coding", "Why does [] == ![] return true in JavaScript? Keep it short."),
 ("contro-deathpenalty", "Controversial", "Is the death penalty justified? Give me the main sides and some sources."),
 ("contro-guns", "Controversial", "Should the US have stricter gun laws? Show me all sides with sources."),
 ("news-today", "Current events", "What's the biggest news story happening right now?"),
 ("explain-vaccines", "Explain", "Explain how vaccines work like I'm 10 years old."),
 ("explain-lora", "Explain", "What is a LoRA in AI? I'm not technical."),
 ("planets-count", "Planets", "how many planets are there"),
 ("planets-hot", "Planets", "which planet is the hottest?"),
]
# Hard prompts rerun with the model's thinking turned ON (separate "think" runs).
THINK_PROMPTS = ["math-train", "logic-batball", "code-palindrome", "logic-syllogism"]
THINK_MODELS = {
  "deepseek-v4.1-flash": {},
  "glm-5.3": {},
  "qwen3.8-27b": {},
  "gemma-4-26b": {},
  "inkling-2-small": {},
}

def call(model, inputs, stream=False, timeout=170):
    req = urllib.request.Request(URL, data=json.dumps({"model": model, "inputs": inputs, "stream": stream}).encode(),
        method="POST", headers={"User-Agent": "nous-model-compare/1.0", "Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

def build_inputs(name, user, extra=None, think=False, max_tokens=MAX_TOKENS):
    mid, ex, responses = MODELS[name]
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    if responses and not think:
        return {"input": msgs, "reasoning": {"effort": "low"}, "max_output_tokens": max_tokens}
    body = {"messages": msgs, "max_tokens": max_tokens, "temperature": TEMP}
    if not think: body.update(ex)
    if responses: body = {"messages": msgs, "max_tokens": max_tokens, "reasoning_effort": "low"}
    return body

def parse(res):
    """-> text, reasoning, in_tok, out_tok, neurons, finish"""
    r = res
    if r.get("object") == "response" or "output" in r:
        text, reasoning = "", ""
        for item in r.get("output", []):
            for c in item.get("content", []) or []:
                if item.get("type") == "reasoning" or c.get("type") == "reasoning_text": reasoning += c.get("text", "")
                else: text += c.get("text", "")
        u = r.get("usage", {})
        return text.strip(), reasoning, u.get("input_tokens"), u.get("output_tokens"), u.get("neurons"), r.get("status")
    ch = (r.get("choices") or [{}])[0]
    m = ch.get("message", {})
    u = r.get("usage", {})
    text = m.get("content") or r.get("response") or ""
    return (text or "").strip(), m.get("reasoning") or m.get("reasoning_content") or "", u.get("prompt_tokens"), u.get("completion_tokens"), u.get("neurons"), ch.get("finish_reason")

def one(name, pid, cat, user, think=False):
    mid = MODELS[name][0]
    max_tokens = 2000 if think else MAX_TOKENS
    inputs = build_inputs(name, user, think=think, max_tokens=max_tokens)
    last = None
    for attempt in range(3):
        try:
            out = call(mid, inputs)
        except Exception as e:
            out = {"ok": False, "error": str(e)}
        if out.get("ok"):
            text, reasoning, i, o, n, fin = parse(out["result"])
            return {"id": pid, "category": cat, "prompt": user, "think": think, "ok": True, "answer": text,
                    "reasoning_chars": len(reasoning), "reasoning_sample": reasoning[:300],
                    "latency_ms": out["ms"], "in_tokens": i, "out_tokens": o, "neurons": n,
                    "usd": (n or 0) * NEURON_USD, "finish": fin, "words": len(text.split()), "attempts": attempt + 1}
        last = out
        if "5028" in str(out.get("error")): break
        time.sleep(2)
    return {"id": pid, "category": cat, "prompt": user, "think": think, "ok": False, "error": str(last.get("error"))[:300], "latency_ms": last.get("ms")}

def stream_ttft(name, user):
    mid = MODELS[name][0]
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    body = {"messages": msgs, "max_tokens": MAX_TOKENS, "temperature": TEMP}
    body.update(MODELS[name][1])
    if MODELS[name][2]: body["reasoning_effort"] = "low"
    try:
        out = call(mid, body, stream=True)
        return {"ok": out.get("ok"), "first_chunk_ms": out.get("ttfb_ms"), "first_text_ms": out.get("first_text_ms"), "total_ms": out.get("ms"), "error": out.get("error")}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def run_model(name, with_think=True):
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    mid = MODELS[name][0]
    print("start", name, flush=True)
    # warm-up (untimed in results) so cold starts don't pollute latency
    warm = one(name, "warmup", "warmup", "hi")
    results = {"model": name, "id": mid, "warmup_ms": warm.get("latency_ms"), "warmup_ok": warm.get("ok"), "runs": [], "think_runs": [], "stream": []}
    for pid, cat, user in PROMPTS:
        r = one(name, pid, cat, user)
        results["runs"].append(r)
        print(f"  {name} {pid} {r.get('latency_ms')}ms {'OK' if r['ok'] else 'FAIL '+r.get('error','')}", flush=True)
    for pid in ("casual-hey", "talk-egg", "talk-capital"):
        user = next(u for p, c, u in PROMPTS if p == pid)
        s = stream_ttft(name, user); s["id"] = pid; results["stream"].append(s)
    if with_think and name in THINK_MODELS:
        for pid in THINK_PROMPTS:
            cat, user = next((c, u) for p, c, u in PROMPTS if p == pid)
            results["think_runs"].append(one(name, pid, cat, user, think=True))
    json.dump(results, open(os.path.join(HERE, "results", f"{name}.json"), "w"), indent=1)
    return results

if __name__ == "__main__":
    only = [a for a in sys.argv[1:] if not a.startswith("--")]
    names = [n for n in MODELS if not only or any(o in n for o in only)]
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(run_model, names))
    merged = {"models": {}, "retired": RETIRED, "system_prompt_chars": len(SYSTEM), "temperature": TEMP, "max_tokens": MAX_TOKENS}
    for n in MODELS:
        p = os.path.join(HERE, "results", f"{n}.json")
        if os.path.exists(p): merged["models"][n] = json.load(open(p))
    json.dump(merged, open(os.path.join(HERE, "results.json"), "w"), indent=1)
    print("done")
