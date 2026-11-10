import { describe, expect, it } from "vitest";

import { isAllowedObjectKey } from "../src/objectKey";

describe("isAllowedObjectKey", () => {
  it("allows starter bases and user objects", () => {
    expect(isAllowedObjectKey("bases/starters/weather/dataset.csv")).toBe(true);
    expect(isAllowedObjectKey("bases/starters/weather/model.pt")).toBe(true);
    expect(isAllowedObjectKey("users/owner/datasets/dst_abc.csv")).toBe(true);
    expect(isAllowedObjectKey("users/usr_123/models/model.pt")).toBe(true);
  });

  it("allows GQ voice cache objects", () => {
    const digest = "a".repeat(64);
    expect(isAllowedObjectKey(`voice/cache/${digest}.mp3`)).toBe(true);
  });

  it("rejects unsafe or unknown keys", () => {
    expect(isAllowedObjectKey("voice/cache/not-a-digest.mp3")).toBe(false);
    expect(isAllowedObjectKey("../voice/cache/" + "a".repeat(64) + ".mp3")).toBe(false);
    expect(isAllowedObjectKey("voice/cache/" + "a".repeat(64) + ".mp3/../x")).toBe(false);
    expect(isAllowedObjectKey("tmp/secret.mp3")).toBe(false);
    expect(isAllowedObjectKey("")).toBe(false);
  });
});
