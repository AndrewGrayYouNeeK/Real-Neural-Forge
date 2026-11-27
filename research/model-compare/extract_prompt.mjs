// Extracts the production SYSTEM_PROMPT from the repo's ai.ts (bundled with esbuild) into system_prompt.txt
const { build } = await import(process.env.ESBUILD_PATH || "esbuild");
import fs from "node:fs";
const repo = process.argv[2] || "repo";
const out = await build({ entryPoints: [`${repo}/deploy/cloudflare/src/live/ai.ts`], bundle: true, write: false, format: "esm", platform: "neutral", logLevel: "error" });
const code = out.outputFiles[0].text;
const mod = await import("data:text/javascript;base64," + Buffer.from(code).toString("base64"));
fs.writeFileSync("system_prompt.txt", mod.SYSTEM_PROMPT);
console.log("chars", mod.SYSTEM_PROMPT.length, "temp", mod.NORMAL_TEMPERATURE, "maxTokens", mod.MAX_OUTPUT_TOKENS);
