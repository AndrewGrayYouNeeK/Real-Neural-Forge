# Nous LoRA starter kit

Teach a small, fast AI model to talk like Nous (funny, short, honest, spoken-friendly), then run it on Cloudflare Workers AI next to GPT-OSS 120B.

**The plan:** keep GPT-OSS 120B for hard questions. Add one small model with a Nous LoRA on it for everyday chat and Talk mode.

> **Name swap:** if Nous gets renamed (for example Tholun), nothing here breaks. Set `APP_NAME` in the notebook's first cell and it swaps the name in every example before training. See [Renaming](#renaming-the-app).

## What's in this folder

| File | What it is |
| --- | --- |
| `nous_chat.jsonl` | 256 training chats in the Nous voice, one per line |
| `system_prompt.txt` | The short persona prompt every example starts with (use the same one at chat time) |
| `nous_lora_colab.ipynb` | The Colab notebook that trains and saves the LoRA |
| `tools/validate_dataset.py` | Checks the examples file before you train |

## Which model, and why

Checked against Cloudflare's docs and model catalog on Nov 27, 2026.

**Workers AI accepts BYO LoRA adapters only on these dedicated base models:**

| Base model on Workers AI | Context window | Notes |
| --- | --- | --- |
| `@cf/mistral/mistral-7b-instruct-v0.2-lora` | 15,000 tokens | No deprecation date listed. **This is the pick.** |
| `@cf/google/gemma-7b-it-lora` | 3,500 tokens | Beta. The catalog lists a planned deprecation date of Nov 20, 2026. |
| `@cf/google/gemma-2b-it-lora` | 8,192 tokens | Beta. Same planned deprecation date. |
| `@cf/meta-llama/llama-2-7b-chat-hf-lora` | 8,192 tokens | Beta. Same planned deprecation date. |

Cloudflare's docs say the base model it runs for LoRA must not be quantized, and that `model_type` in `adapter_config.json` must be `mistral`, `gemma` or `llama`.

**One honest caveat:** the free Colab T4 can't fit a full-precision 7B model for training, so the notebook trains on a 4-bit copy of Mistral (this is called QLoRA). The adapter it saves is the same shape and gets applied to Cloudflare's full-precision Mistral. That's a standard, widely used setup, but it hasn't been tested end to end on Workers AI yet, so do step 4 (test) before routing any real traffic. If the answers come out off, train on a bigger GPU with `load_in_4bit = False` in the notebook.

**GPT-OSS 120B does not support LoRA.** It isn't on the list above, and its catalog entry has no LoRA flag. (Its request schema shows a generic `lora` field, but Cloudflare's docs only promise LoRA on the models in the table.) Don't plan on tuning 120B on Workers AI.

**Limits (from Cloudflare's docs):**
- Rank `r` must be 8 or lower (the docs say larger ranks up to 32 can work, but 8 is the safe value, and the notebook uses 8).
- Adapter file must be under 300 MB. Ours is roughly 15 MB.
- Files must be named exactly `adapter_model.safetensors` and `adapter_config.json`.
- Up to 100 adapters per account.
- LoRA is in open beta and free during the beta.

**Recommendation: Mistral-7B-Instruct-v0.2.** It's the only LoRA-capable model with no deprecation date, it has the biggest context window of the four (15,000 tokens, so Talk mode can keep a long conversation), it's strong for its size at chatty, instruction-following replies, and it's Apache-2.0 licensed. Gemma 2B would be a bit faster, but it's weaker at humor and facts, and it's on Cloudflare's deprecation list.

> Heads up on speed: a 7B model is much faster than 120B, but "fast" depends on Workers AI load. Test it with Talk mode before you switch everyone over.

## Beginner steps

### 1. Train in Colab (free, about 20 minutes)

1. Open the notebook: **[Open in Colab](https://colab.research.google.com/github/AndrewGrayYouNeeK/Nous/blob/main/training/nous_lora_colab.ipynb)**
2. Sign in with a Google account.
3. Top menu: **Runtime → Change runtime type → T4 GPU → Save**.
4. Top menu: **Runtime → Run all**. Wait for it. When it asks permission to download a file at the end, allow it.
5. Read the test chats near the bottom. Does it sound like Nous? If so, you're done. A file called `nous-lora.zip` downloads to your computer.

If it doesn't sound right, see [Making it better](#making-it-better).

Note: Colab's free GPU isn't guaranteed, and Unsloth changes often. If the install cell fails, open Unsloth's latest Mistral notebook and copy its install cell over ours.

### 2. Unzip and check the files

```bash
unzip nous-lora.zip -d nous-lora
ls nous-lora
# adapter_config.json  adapter_model.safetensors
grep -E '"r"|model_type' nous-lora/adapter_config.json
# "r": 8   and   "model_type": "mistral"
```

### 3. Upload the adapter to Cloudflare with wrangler

You'll need to be logged in to Cloudflare once (`npx wrangler login`). The account is `d74b0de637f4d9d9eb3efb9e13142784`.

```bash
export CLOUDFLARE_ACCOUNT_ID=d74b0de637f4d9d9eb3efb9e13142784
npx wrangler ai finetune create @cf/mistral/mistral-7b-instruct-v0.2-lora nous-chat-v1 ./nous-lora
npx wrangler ai finetune list
```

`nous-chat-v1` is the name you'll use in code. You can't edit an adapter after upload, so for a new version upload a new one (`nous-chat-v2`) and switch the name.

The same upload also works through Cloudflare's REST API if you prefer curl (token needs `Workers AI: Edit`). Wrangler is simpler.

### 4. Test it before wiring it in

Use the same prompt shape the model was trained on (Mistral's `[INST]` format, with the system prompt at the top of the first user turn). Replace `$TOKEN` with an API token that has Workers AI access (never paste the token in chat or commit it).

```bash
SYSTEM="$(cat training/system_prompt.txt)"
curl -s "https://api.cloudflare.com/client/v4/accounts/d74b0de637f4d9d9eb3efb9e13142784/ai/run/@cf/mistral/mistral-7b-instruct-v0.2-lora" \
  -H "Authorization: Bearer $TOKEN" \
  -d "$(jq -n --arg p "<s>[INST] $SYSTEM

Who made you? [/INST]" '{prompt:$p, raw:true, lora:"nous-chat-v1", max_tokens:160, temperature:0.8}')"
```

In a Worker it looks like this:

```ts
const result = await env.AI.run("@cf/mistral/mistral-7b-instruct-v0.2-lora", {
  prompt: `<s>[INST] ${system}\n\n${question} [/INST]`, // same format as training
  raw: true,                                           // we supply the chat template ourselves
  lora: "nous-chat-v1",                                // the finetune name (or id) from wrangler
  max_tokens: 160,
  temperature: 0.8,
});
```

Cloudflare's docs show `raw: true` together with a `messages` array. We use `prompt` so the text matches training exactly. If the reply is garbled or starts with extra tokens, try removing the leading `<s>` (Workers AI may add it itself) and compare.

## Making it better

- **Too robotic or bland?** Add more funny examples to `nous_chat.jsonl` and train again. 200 to 1,000 good examples beats thousands of sloppy ones.
- **Repeats examples word for word?** It's overtrained. Set `EPOCHS = 2` in the notebook.
- **Barely changed?** Add more examples, or try `EPOCHS = 4`.
- Add examples the same way as the others, as one JSON object per line:
  ```json
  {"messages":[{"role":"system","content":"<paste training/system_prompt.txt>"},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
  ```
- Always run `python3 training/tools/validate_dataset.py` after editing. It checks the format, the system prompt, the canonical creator answer, and scans for anything that looks like a secret.

Rules for examples: short and spoken-friendly, funny, honest when unsure, controversial topics from every side with named sources, no hidden reasoning, no secrets, no live weather or prices made up. Serious topics get kindness first.

The creator answer must stay word for word: *created by and is owned by Andrew Gray, founder, owner and CEO of YouNeeK (pronounced "unique")*.

## Renaming the app

1. In the notebook's settings cell, set `APP_NAME = "Tholun"` (or the new name). The notebook swaps `Nous` for it in every example and in the system prompt before training.
2. Fix by hand in `nous_chat.jsonl` and `system_prompt.txt` any lines that depend on how the old name sounds. Search for `noose`. The "What's your name?" examples spell out the pronunciation.
3. Train again and upload as a new finetune name (`tholun-chat-v1`), then change the name in the routing code. Keep the system prompt identical in training and in the Worker.

## What this doesn't cover (on purpose)

- **Unhinged mode** isn't in the training data. It stays on GPT-OSS 120B, off by default, as it is today.
- **Live data answers** (weather, markets, space, forecasts) still come from the app's own tools. The data includes a few examples of rewriting a factual reply in the Nous voice without changing any numbers, but the app should still check that every number survived (it already does for 120B).
- **The chat code is not changed.** Nothing in `deploy/cloudflare/` is touched by this kit.

## Code change needed later (not done yet)

When the adapter is uploaded and tested, the routing change lives in `deploy/cloudflare/src/live/ai.ts` plus its tests:

1. **Add constants** next to `PRIMARY_MODEL` and `FALLBACK_MODEL`: `LORA_MODEL = "@cf/mistral/mistral-7b-instruct-v0.2-lora"`, and read the finetune name from an env var (for example `NOUS_LORA_NAME`, default empty). If the name is empty, routing stays exactly as it is today, so the change ships safely off.
2. **Add a LoRA system prompt:** import the exact text of `training/system_prompt.txt`. The LoRA model must see the same persona text it was trained on, not the long `SYSTEM_PROMPT`. Keep the `HARD_LIMITS` rules too.
3. **Add a branch in `chatInputs`** for `LORA_MODEL`: return `{ prompt: "<s>[INST] " + system + "\n\n" + content + " [/INST]", raw: true, lora: name, max_tokens, temperature }` instead of `messages`. Multi-turn history needs the `[INST] ... [/INST] answer</s>` format, built like `to_mistral_text` in the notebook.
4. **Add a small router**, for example `pickChatModel({ mode, talk, question, hasTools })`:
   - Send to the LoRA model: Talk mode, short everyday questions, small talk, jokes, quick explainers.
   - Keep on GPT-OSS 120B: Unhinged mode, tool and agent-loop calls, code, math and anything that needs live data, long or research-style questions, and controversial topics where sources matter.
5. **Fallback in both directions:** if the LoRA call errors or returns empty text, retry on 120B. Strip stray `[INST]`, `[/INST]`, `<s>` and `</s>` from the output before showing or speaking it.
6. **Quota accounting:** `usageNeurons` and the neuron constants are written for 120B. LoRA inference is free during Cloudflare's open beta, so count it at zero for now and recheck pricing when beta ends. Update `deploy/cloudflare/test/aiQuota.test.ts`, and add tests for the router and for the fallback.
7. **Voice:** Talk mode's speech substitutions (Nous spoken as "Noose", YouNeeK as "unique") are applied after the model answers, so they don't change.
8. **Roll out behind the env var,** compare 120B and LoRA answers side by side on the same questions, then turn it on. Don't redeploy an older image to roll back, just clear the env var.

Worker-only changes need no container image or pin, but do follow the normal deploy flow for the Worker.
