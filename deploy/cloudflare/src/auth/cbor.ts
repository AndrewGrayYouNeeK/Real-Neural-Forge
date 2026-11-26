/** Minimal CBOR decoder: enough for WebAuthn attestation objects and COSE keys. */

export type CborValue =
  | number
  | bigint
  | string
  | boolean
  | null
  | undefined
  | Uint8Array
  | CborValue[]
  | Map<CborValue, CborValue>;

export function decodeCbor(bytes: Uint8Array): { value: CborValue; length: number } {
  let offset = 0;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

  function need(count: number): void {
    if (offset + count > bytes.length) throw new Error("CBOR truncated");
  }

  function readLength(info: number): number {
    if (info < 24) return info;
    if (info === 24) {
      need(1);
      return bytes[offset++];
    }
    if (info === 25) {
      need(2);
      const value = view.getUint16(offset);
      offset += 2;
      return value;
    }
    if (info === 26) {
      need(4);
      const value = view.getUint32(offset);
      offset += 4;
      return value;
    }
    if (info === 27) {
      need(8);
      const value = Number(view.getBigUint64(offset));
      offset += 8;
      return value;
    }
    throw new Error("CBOR indefinite lengths are not supported");
  }

  function item(depth: number): CborValue {
    if (depth > 16) throw new Error("CBOR too deep");
    need(1);
    const initial = bytes[offset++];
    const major = initial >> 5;
    const info = initial & 31;
    switch (major) {
      case 0:
        return readLength(info);
      case 1:
        return -1 - readLength(info);
      case 2: {
        const length = readLength(info);
        need(length);
        const out = bytes.slice(offset, offset + length);
        offset += length;
        return out;
      }
      case 3: {
        const length = readLength(info);
        need(length);
        const out = new TextDecoder().decode(bytes.subarray(offset, offset + length));
        offset += length;
        return out;
      }
      case 4: {
        const length = readLength(info);
        const out: CborValue[] = [];
        for (let index = 0; index < length; index += 1) out.push(item(depth + 1));
        return out;
      }
      case 5: {
        const length = readLength(info);
        const out = new Map<CborValue, CborValue>();
        for (let index = 0; index < length; index += 1) {
          const key = item(depth + 1);
          out.set(key, item(depth + 1));
        }
        return out;
      }
      case 6:
        readLength(info);
        return item(depth + 1);
      case 7:
        if (info === 20) return false;
        if (info === 21) return true;
        if (info === 22) return null;
        if (info === 23) return undefined;
        throw new Error("CBOR floats are not supported");
      default:
        throw new Error("CBOR bad major type");
    }
  }

  const value = item(0);
  return { value, length: offset };
}
