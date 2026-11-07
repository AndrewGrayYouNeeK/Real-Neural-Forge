import { describe, expect, it } from "vitest";

import { containerEnv } from "../src/containerEnv";

function env(overrides: Partial<Record<string, string>> = {}): Record<string, string | undefined> {
  return {
    NEURAL_FORGE_CONFIG: "config/config.yaml",
    NEURAL_FORGE_CHECKPOINT_URL: "https://example.com/champion.pt",
    NEURAL_FORGE_STORAGE: "cloudflare",
    NEURAL_FORGE_OWNER_EMAIL: "andrew@youneek.xyz",
    ...overrides,
  };
}

describe("containerEnv", () => {
  it("forwards the voice and payments secrets to the container", () => {
    const vars = containerEnv(
      env({
        ELEVENLABS_API_KEY: "sk-elevenlabs",
        STRIPE_SECRET_KEY: "sk_test_stripe",
        STRIPE_WEBHOOK_SECRET: "whsec_test",
      }),
    );
    expect(vars.ELEVENLABS_API_KEY).toBe("sk-elevenlabs");
    expect(vars.STRIPE_SECRET_KEY).toBe("sk_test_stripe");
    expect(vars.STRIPE_WEBHOOK_SECRET).toBe("whsec_test");
  });

  it("aliases the FORGE_VOICE_ID secret to the name the container reads", () => {
    const vars = containerEnv(env({ FORGE_VOICE_ID: "QMnwdN2oOwNXPcl9oXIl" }));
    expect(vars.NEURAL_FORGE_VOICE_ID).toBe("QMnwdN2oOwNXPcl9oXIl");
    expect(vars.FORGE_VOICE_ID).toBeUndefined();
  });

  it("prefers NEURAL_FORGE_VOICE_ID over the legacy alias", () => {
    const vars = containerEnv(
      env({ FORGE_VOICE_ID: "legacy", NEURAL_FORGE_VOICE_ID: "current" }),
    );
    expect(vars.NEURAL_FORGE_VOICE_ID).toBe("current");
  });

  it("drops unset and empty values instead of passing empty strings", () => {
    const vars = containerEnv(
      env({ ELEVENLABS_API_KEY: undefined, STRIPE_SECRET_KEY: "" }),
    );
    expect(vars.ELEVENLABS_API_KEY).toBeUndefined();
    expect(vars.STRIPE_SECRET_KEY).toBeUndefined();
    expect(vars.NEURAL_FORGE_CHECKPOINT_URL).toBe("https://example.com/champion.pt");
  });

  it("keeps the default bindings URL", () => {
    const vars = containerEnv(env());
    expect(vars.NEURAL_FORGE_BINDINGS_URL).toBe("http://bindings.neuralforge");
  });
});
