#!/usr/bin/env python3
"""Check training/nous_chat.jsonl before you train on it.

Run from the repo root:  python3 training/tools/validate_dataset.py

It checks that every line is valid JSON, roles are in the right order, the system prompt matches
training/system_prompt.txt, the canonical creator answer is word-for-word right, nothing looks like
a secret, and no hidden-reasoning tags slipped in. Exit code 0 means all good.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "nous_chat.jsonl"
SYSTEM = (HERE / "system_prompt.txt").read_text(encoding="utf-8").strip()
CREATOR = 'created by and is owned by Andrew Gray, founder, owner and CEO of YouNeeK (pronounced "unique")'

SECRET_PATTERNS = [
    r"sk-[A-Za-z0-9]{16,}", r"ghp_[A-Za-z0-9]{20,}", r"github_pat_[A-Za-z0-9_]{20,}",
    r"AKIA[0-9A-Z]{16}", r"xox[baprs]-[A-Za-z0-9-]{10,}", r"Bearer\s+[A-Za-z0-9._-]{20,}",
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----", r"\b[a-f0-9]{40,}\b",
]
BAD_TAGS = ["<think>", "</think>", "<analysis>", "chain of thought:", "analysis:"]

errors, creator_hits, multi = [], 0, 0
lines = DATA.read_text(encoding="utf-8").splitlines()
firsts = set()
for n, line in enumerate(lines, 1):
    try:
        msgs = json.loads(line)["messages"]
    except Exception as exc:  # noqa: BLE001
        errors.append(f"line {n}: bad JSON ({exc})")
        continue
    if msgs[0]["role"] != "system" or msgs[0]["content"].strip() != SYSTEM:
        errors.append(f"line {n}: system message must match training/system_prompt.txt")
    roles = [m["role"] for m in msgs[1:]]
    if not roles or roles[0] != "user" or roles[-1] != "assistant" or any(
        r != ("user" if i % 2 == 0 else "assistant") for i, r in enumerate(roles)
    ):
        errors.append(f"line {n}: roles must alternate user/assistant, starting with user and ending with assistant")
    if len(roles) > 2:
        multi += 1
    for m in msgs:
        text = m["content"]
        if not text.strip():
            errors.append(f"line {n}: empty message")
        for pat in SECRET_PATTERNS:
            if re.search(pat, text):
                errors.append(f"line {n}: looks like a secret ({pat})")
        for tag in BAD_TAGS:
            if tag in text.lower():
                errors.append(f"line {n}: hidden-reasoning marker {tag!r}")
        if m["role"] == "assistant" and "Andrew Gray" in text and CREATOR in text:
            creator_hits += 1
        elif m["role"] == "assistant" and "owned by Andrew Gray" in text:
            errors.append(f"line {n}: creator answer is not word-for-word canonical")
    firsts.add(msgs[1]["content"])

if creator_hits < 5:
    errors.append(f"only {creator_hits} examples use the canonical creator answer")

print(f"{len(lines)} examples ({multi} multi-turn), {creator_hits} use the canonical creator answer")
if errors:
    print("\n".join(errors))
    sys.exit(1)
print("OK")
