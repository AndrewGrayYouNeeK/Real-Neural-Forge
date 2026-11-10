// Keys the container may read or write through the Worker R2 proxy.
// Keep this in lockstep with src/workspace/keys.py valid_object_key().

const OBJECT_KEY =
  /^((bases\/starters\/[a-z0-9-]+\/(dataset\.csv|model\.pt))|(users\/[A-Za-z0-9_]+\/(datasets|models)\/[A-Za-z0-9_.-]+)|(voice\/cache\/[a-f0-9]{64}\.mp3))$/;

export function isAllowedObjectKey(key: string): boolean {
  return OBJECT_KEY.test(key) && !key.includes("..") && !key.includes("\\");
}
