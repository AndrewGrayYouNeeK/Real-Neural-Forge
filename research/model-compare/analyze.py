#!/usr/bin/env python3
"""Scores each model from results.json (objective checks + hand-reviewed judgments) -> scores.json"""
import json, re, statistics as st
D = json.load(open('results.json'))['models']
def run(m, pid): return next(x for x in m['runs'] if x['id'] == pid)

# --- hand-reviewed judgments after reading every answer (0-1 unless noted) ---
JS_OK   = {'gpt-oss-120b':1,'deepseek-v4.1-flash':1,'glm-5.3':1,'inkling-2-small':1,'llama-4-scout':1,'llama-3.3-70b-fast':.5}
CONTRO  = {'deepseek-v4.1-flash':1,'glm-5.3':1,'qwen3.8-27b':.9,'inkling-2-small':.7,'gemma-4-26b':.7,'gpt-oss-120b':.7,'gpt-oss-20b':.5,
           'llama-4-scout':.6,'llama-3.3-70b-fast':.5,'llama-3.1-8b-fast':.3,'mistral-small-3.1':.4,'phonellm-mini':.4,'mistral-7b-lora':.3}
NEWS    = {'mistral-small-3.1':0,'phonellm-mini':0,'llama-3.1-8b-fast':0,'mistral-7b-lora':.5}   # others honestly said "no live news"
EGG     = {'gpt-oss-20b':0,'mistral-small-3.1':.7,'llama-3.1-8b-fast':.8}                         # others gave sensible egg times
HOT     = {'mistral-small-3.1':0,'llama-3.1-8b-fast':.5}                                          # Venus is the right answer
FABRIC  = {'mistral-small-3.1':3,'mistral-7b-lora':1.5,'phonellm-mini':0.5}                       # invented facts / off-topic junk
FUNNY   = {'gpt-oss-120b':8,'gpt-oss-20b':6,'deepseek-v4.1-flash':8.5,'glm-5.3':8.5,'qwen3.8-27b':7,'gemma-4-26b':7.5,'mistral-small-3.1':4,
           'phonellm-mini':7,'llama-4-scout':6,'llama-3.3-70b-fast':6.5,'llama-3.1-8b-fast':5,'inkling-2-small':8.5,'mistral-7b-lora':4}
LORA_OK = {'mistral-7b-lora':0}

def correct_math(pid, a):
    a2 = a.replace(',', '')
    return {'math-train': bool(re.search(r'6[:.]?\s?15|quarter past six', a, re.I)),
            'math-percent': '447.5' in a2,
            'logic-syllogism': bool(re.search(r'\b(no|not|cannot|can.t)\b', a[:200], re.I)) and not re.match(r'\W*yes', a, re.I),
            'logic-batball': bool(re.search(r'5 cents|\$0?\.05|0\.05|five cents|5¢', a, re.I))}[pid]
def code_ok(a):
    blocks = re.findall(r"```(?:python|py)?\n(.*?)```", a, re.S)
    if not blocks: return 0
    ns = {}
    try:
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()): exec(blocks[0], ns)
        fn = [v for k, v in ns.items() if callable(v) and not k.startswith('_')][-1]
        T = [("A man, a plan, a canal: Panama", True), ("hello", False), ("No lemon, no melon", True), ("Was it a car or a cat I saw?", True), ("", True), ("ab", False)]
        return 1 if all(fn(t) == e for t, e in T) else 0
    except Exception: return 0

out = {}
for n, m in D.items():
    ok = [r for r in m['runs'] if r['ok']]
    lat = sorted(r['latency_ms'] for r in ok)
    med = st.median(lat) / 1000
    p90 = lat[int(len(lat) * .9) - 1] / 1000
    neurons = sum(r['neurons'] or 0 for r in ok)
    usd_call = neurons / len(ok) * 0.011 / 1000
    # smarts
    items = {}
    items['math-train'] = 2 * correct_math('math-train', run(m, 'math-train')['answer'])
    for pid in ('math-percent', 'logic-syllogism', 'logic-batball'): items[pid] = correct_math(pid, run(m, pid)['answer'])
    items['code'] = 1.5 * code_ok(run(m, 'code-palindrome')['answer'])
    items['js'] = 1.5 * JS_OK.get(n, 0)
    items['lora'] = LORA_OK.get(n, 1)
    items['contro'] = CONTRO[n]
    smarts = 10 * sum(items.values()) / 10.0
    smarts = max(1, round(smarts, 1))
    # accuracy
    acc_items = [1, 1, HOT.get(n, 1), NEWS.get(n, 1), EGG.get(n, 1), 1 if (run(m, 'planets-count')['ok']) else 0, LORA_OK.get(n, 1), JS_OK.get(n, 0) if n != 'llama-3.3-70b-fast' else .5]
    accuracy = round(max(1, 10 * sum(acc_items) / len(acc_items) - FABRIC.get(n, 0)), 1)
    # creator
    els = []; exact = 0
    for pid in ('creator-1', 'creator-2', 'creator-3'):
        a = run(m, pid)['answer']
        e = [bool(re.search(r'Andrew Gray', a)), bool(re.search(r'\bowner\b|owned by|owns? (me|nous)|I.m owned|and am owned', a, re.I)), bool(re.search(r'founder', a, re.I)),
             bool(re.search(r'CEO', a)), bool(re.search(r'unique', a, re.I))]
        els.append(sum(e) / 5)
        exact += bool(re.search(r'created by and (is |am |are )?owned by', a, re.I))
    creator = round(max(1, 10 * (0.7 * st.mean(els) + 0.3 * exact / 3) - FABRIC.get(n, 0) * (1 if n == 'mistral-small-3.1' else 0)), 1)
    # talk fit
    tk = [run(m, p) for p in ('talk-capital', 'talk-egg', 'casual-hey', 'talk-intro')]
    words = st.mean(r['words'] for r in tk)
    tlat = st.mean(r['latency_ms'] for r in tk) / 1000
    fs = [s['first_text_ms'] for s in m['stream'] if s.get('first_text_ms')]
    first_text = st.median(fs) / 1000 if fs else None
    fmt = sum(1 for r in ok if re.search('[\U0001F300-\U0001FAFF]', r['answer'])) + sum(1 for r in ok if re.search(r'^\s*(\||#{1,4} )', r['answer'], re.M)) * .5
    talk = 10 - max(0, (words - 25) / 10) - max(0, (tlat - 0.8) * 0.9) - fmt * .5
    if n == 'mistral-small-3.1': talk -= 1
    talk = round(min(10, max(1, talk)), 1)
    funny = FUNNY[n]
    overall = round(.25 * smarts + .20 * funny + .20 * talk + .20 * accuracy + .15 * creator, 1)
    out[n] = dict(id=m['id'], smarts=smarts, funny=funny, talk=talk, accuracy=accuracy, creator=creator, overall=overall,
        median_s=round(med, 2), p90_s=round(p90, 2), first_text_s=round(first_text, 2) if first_text else None, talk_latency_s=round(tlat, 2), talk_words=round(words),
        avg_words=round(st.mean(r['words'] for r in ok)), neurons_total=round(neurons), usd_per_1000_msgs=round(usd_call * 1000, 2),
        truncated=sum(1 for r in ok if r['finish'] in ('length', 'incomplete')), failures=len(m['runs']) - len(ok),
        reasoning_hidden_rows=sum(1 for r in ok if r['reasoning_chars']), leaks=0, items=items)
json.dump(out, open('scores.json', 'w'), indent=1)
print(f"{'model':22}{'smart':>6}{'funny':>6}{'talk':>6}{'acc':>6}{'crtr':>6}{'ALL':>6}{'med s':>7}{'1stTxt':>7}{'$/1k':>7}")
for n, s in sorted(out.items(), key=lambda kv: -kv[1]['overall']):
    print(f"{n:22}{s['smarts']:>6}{s['funny']:>6}{s['talk']:>6}{s['accuracy']:>6}{s['creator']:>6}{s['overall']:>6}{s['median_s']:>7}{s['first_text_s']:>7}{s['usd_per_1000_msgs']:>7}")
