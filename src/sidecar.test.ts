// Sidecar file-format library tests. Ports the behavioral cases from the spike's
// 30-case resilience suite (the pure perf-timing cases — 15k-record cold-scan
// latency etc. — are environment benchmarks, not unit assertions, and are left to
// the on-device probe) and adds the production-specific cases: same-device
// tombstone scoping, partial-arrival self-heal, conflict-copy ignore, version
// gate, and the 4 MB shard-roll boundary.

import { describe, it, expect } from 'vitest';
import type { DataAdapter } from 'obsidian';
import { quantizeInt8 } from './quant';
import { packSignBits } from './binary';
import {
    appendTombstone,
    bm25PathFor,
    bulkAppend,
    clearDevice,
    coalesceSmallShards,
    compactDevice,
    decodeRecord,
    deviceShardBytes,
    encodeRecord,
    isOffsetInRange,
    isValidRecord,
    jsonlPathFor,
    listDeviceJsonls,
    listDeviceShards,
    listSidecarDeviceIds,
    metaPathFor,
    readRecordAt,
    scanJsonl,
    shardPathFor,
    shouldReconcileSidecar,
    sidecarDirSignature,
    staleSidecarFormat,
    sweepOrphanTmpFiles,
    withDirLock,
    MAX_VECTORS_PER_SHARD,
    Q_BYTES,
    SIGN_BYTES,
    SIDECAR_FORMAT,
    VEC_BYTES,
    type TierBytes,
} from './sidecar';
import { readDeviceMeta, writeDeviceMeta, metaAccepts, type SidecarMeta } from './sidecar-meta';

// ---- in-memory DataAdapter fake ----

class FakeAdapter {
    files = new Map<string, string>();
    bins = new Map<string, ArrayBuffer>();

    async exists(p: string): Promise<boolean> {
        if (this.files.has(p) || this.bins.has(p)) return true;
        // A directory "exists" if any stored file lives under it (mirrors Obsidian).
        const prefix = p.endsWith('/') ? p : p + '/';
        for (const k of [...this.files.keys(), ...this.bins.keys()]) if (k.startsWith(prefix)) return true;
        return false;
    }
    async mkdir(_p: string): Promise<void> {
        /* directories are implicit in this fake — no-op */
    }
    async read(p: string): Promise<string> {
        const v = this.files.get(p);
        if (v === undefined) throw new Error(`ENOENT ${p}`);
        return v;
    }
    async write(p: string, data: string): Promise<void> {
        this.files.set(p, data);
    }
    async append(p: string, data: string): Promise<void> {
        this.files.set(p, (this.files.get(p) ?? '') + data);
    }
    async readBinary(p: string): Promise<ArrayBuffer> {
        const v = this.bins.get(p);
        if (v === undefined) throw new Error(`ENOENT ${p}`);
        return v;
    }
    async writeBinary(p: string, buf: ArrayBuffer): Promise<void> {
        this.bins.set(p, buf);
    }
    async rename(from: string, to: string): Promise<void> {
        if (this.bins.has(from)) {
            this.bins.set(to, this.bins.get(from)!);
            this.bins.delete(from);
        } else if (this.files.has(from)) {
            this.files.set(to, this.files.get(from)!);
            this.files.delete(from);
        } else {
            throw new Error(`ENOENT ${from}`);
        }
    }
    async remove(p: string): Promise<void> {
        this.bins.delete(p);
        this.files.delete(p);
    }
    async list(dir: string): Promise<{ folders: string[]; files: string[] }> {
        const folders = new Set<string>();
        const files: string[] = [];
        const prefix = dir.endsWith('/') ? dir : dir + '/';
        for (const p of [...this.files.keys(), ...this.bins.keys()]) {
            if (!p.startsWith(prefix)) continue;
            const rest = p.slice(prefix.length);
            const slash = rest.indexOf('/');
            if (slash >= 0) folders.add(prefix + rest.slice(0, slash));
            else files.push(p);
        }
        return { folders: [...folders], files };
    }
    async stat(p: string): Promise<{ size: number; type: 'file' } | null> {
        if (this.bins.has(p)) return { size: this.bins.get(p)!.byteLength, type: 'file' };
        if (this.files.has(p)) return { size: this.files.get(p)!.length, type: 'file' };
        return null;
    }
}

function adapter(): DataAdapter {
    return new FakeAdapter() as unknown as DataAdapter;
}

const DIR = '.obsidian/plugins/seek/index';

// Deterministic tiers for a seed. `s` round-trips exactly at the record codec's
// float64 scale width (no truncation, unlike the old f32 encoding).
function tiers(seed: number): TierBytes {
    const q = new Int8Array(Q_BYTES);
    for (let i = 0; i < Q_BYTES; i++) q[i] = (((seed * 31 + i) % 255) - 127) as number;
    const sign = new Uint8Array(SIGN_BYTES);
    for (let i = 0; i < SIGN_BYTES; i++) sign[i] = (seed + i * 7) & 0xff;
    return { q, s: 0.001 + seed * 1e-5, sign };
}

// A minimal valid SidecarMeta for a device (content is irrelevant to enumeration;
// used by the clearDevice/reap-enumeration cases below).
function meta(dev: string): SidecarMeta {
    return { format: SIDECAR_FORMAT, modelId: 'm', revision: null, chunkerVersion: 1, dim: Q_BYTES, deviceId: dev, lastFullReindex: null };
}

function tiersEqual(a: TierBytes, b: TierBytes): boolean {
    if (a.s !== b.s || a.q.length !== b.q.length || a.sign.length !== b.sign.length) return false;
    for (let i = 0; i < a.q.length; i++) if (a.q[i] !== b.q[i]) return false;
    for (let i = 0; i < a.sign.length; i++) if (a.sign[i] !== b.sign[i]) return false;
    return true;
}

// Append one logical record via the production bulkAppend writer. The fixtures
// below exercise resolution semantics one record at a time; this shim keeps them
// readable while routing through the real append path (fresh-shard rotation →
// encodeRecord → writeBinaryAtomic → appendJsonlLine).
async function appendOne(a: DataAdapter, dir: string, dev: string, id: string, t: TierBytes, mtime: number) {
    const [ref] = await bulkAppend(a, dir, dev, [{ id, tiers: t, mtime }]);
    return ref;
}

// ---- codec ----

describe('record codec', () => {
    it('encode → decode round-trips byte-exact (including f64 scale)', () => {
        const t = tiers(42);
        const enc = encodeRecord(t);
        expect(enc.byteLength).toBe(VEC_BYTES);
        const dec = decodeRecord(enc.buffer as ArrayBuffer, 0);
        expect(tiersEqual(t, dec)).toBe(true);
    });

    it('decodes at a non-zero offset within a packed shard', () => {
        const a = tiers(1);
        const b = tiers(2);
        const buf = new Uint8Array(VEC_BYTES * 2);
        buf.set(encodeRecord(a), 0);
        buf.set(encodeRecord(b), VEC_BYTES);
        expect(tiersEqual(decodeRecord(buf.buffer, VEC_BYTES), b)).toBe(true);
    });

    it('isOffsetInRange guards the trailing record', () => {
        const size = VEC_BYTES * 3;
        expect(isOffsetInRange(size, VEC_BYTES * 2)).toBe(true);
        expect(isOffsetInRange(size, VEC_BYTES * 2 + 1)).toBe(false);
        expect(isOffsetInRange(size, -1)).toBe(false);
    });

    it('encodeRecord rejects wrong tier lengths', () => {
        expect(() => encodeRecord({ q: new Int8Array(8), s: 1, sign: new Uint8Array(SIGN_BYTES) })).toThrow();
    });

    it('decodeRecord asserts the stored dim — a stride mismatch fails loud (F9)', () => {
        const enc = encodeRecord(tiers(7));
        expect(() => decodeRecord(enc.buffer as ArrayBuffer, 0)).not.toThrow();        // default dim round-trips
        expect(() => decodeRecord(enc.buffer as ArrayBuffer, 0, 768)).toThrow(/dim/);  // wrong stride → refuse to mis-slice
    });

    it('decodeRecord rejects a CRC-corrupt record (GAP-1)', () => {
        const enc = encodeRecord(tiers(9));
        const corrupt = enc.slice();
        corrupt[0] ^= 0x01;   // flip one in-range bit — the silent-corruption class the CRC exists to catch
        expect(() => decodeRecord(corrupt.buffer as ArrayBuffer, 0)).toThrow(/CRC/);
    });

    it('readRecordAt resolves null (not a throw) on a CRC-corrupt shard (GAP-1)', async () => {
        const a = adapter();
        await bulkAppend(a, DIR, 'desktop-aaa', [{ id: 'c1', tiers: tiers(3), mtime: 1 }]);
        const scan = await scanJsonl(a, [jsonlPathFor(DIR, 'desktop-aaa')]);
        const entry = scan.map.get('c1')!;
        // A synced bit-flip in the shard: flip one in-range byte and re-write it.
        const path = shardPathFor(DIR, 'desktop-aaa', entry.seq);
        const buf = new Uint8Array(await a.readBinary(path));
        buf[entry.off] ^= 0x01;
        await a.writeBinary(path, buf.buffer as ArrayBuffer);
        expect(await readRecordAt(a, DIR, entry)).toBeNull();   // hydrate skips + re-embeds, never crashes
    });
});

// ---- tier fidelity: the sidecar payload must equal what IndexStore persists ----

describe('tier fidelity (IDB ↔ sidecar)', () => {
    it('encode(commitFile-derived tiers) → decode reproduces quantizeInt8 + packSignBits exactly', () => {
        // Mirror what search.ts commitFile derives from an fp32 vector, then push
        // it through the sidecar codec. Decoded tiers must equal the IDB tiers
        // byte-for-byte — that is what makes hydration a verbatim copy.
        const v = new Float32Array(Q_BYTES);
        for (let i = 0; i < Q_BYTES; i++) v[i] = Math.sin(i * 0.37) * (i % 11 === 0 ? -1 : 1); // includes near-zero + sign flips
        const q = quantizeInt8(v); // QuantVec {q, s} — the IDB rerank tier
        const sign = packSignBits(v); // Uint8Array — the IDB candidate tier (from TRUE fp32)

        const enc = encodeRecord({ q: q.q, s: q.s, sign });
        const dec = decodeRecord(enc.buffer as ArrayBuffer, 0);

        // q, sign, AND the scale are byte-identical — the scale is stored at full
        // float64 width so a hydrated peer dequantizes with the exact same scale
        // IndexedDB holds locally (no f32-truncation divergence across devices).
        expect(dec.s).toBe(q.s);
        expect([...dec.q]).toEqual([...q.q]);
        expect([...dec.sign]).toEqual([...sign]);
    });
});

// ---- single + bulk write / read ----

describe('write + read', () => {
    it('single append then readRecordAt returns the same tiers', async () => {
        const a = adapter();
        const t = tiers(7);
        const { seq, off } = await appendOne(a, DIR, 'desktop-aaa', 'c1', t, 100);
        const scan = await scanJsonl(a, [jsonlPathFor(DIR, 'desktop-aaa')]);
        const entry = scan.map.get('c1')!;
        expect(entry.seq).toBe(seq);
        expect(entry.off).toBe(off);
        const got = await readRecordAt(a, DIR, entry);
        expect(got && tiersEqual(got, t)).toBe(true);
    });

    it('bulkAppend writes contiguous offsets and reads back', async () => {
        const a = adapter();
        const recs = [0, 1, 2, 3].map(i => ({ id: `c${i}`, tiers: tiers(i), mtime: 10 + i }));
        const refs = await bulkAppend(a, DIR, 'desktop-aaa', recs);
        expect(refs.map(r => r.off)).toEqual([0, VEC_BYTES, VEC_BYTES * 2, VEC_BYTES * 3]);
        const scan = await scanJsonl(a, [jsonlPathFor(DIR, 'desktop-aaa')]);
        expect(scan.map.size).toBe(4);
        const got = await readRecordAt(a, DIR, scan.map.get('c2')!);
        expect(got && tiersEqual(got, tiers(2))).toBe(true);
    });
});

// ---- resolution semantics ----

describe('scanJsonl resolution', () => {
    it('higher mtime wins on update (same device)', async () => {
        const a = adapter();
        await appendOne(a, DIR, 'desktop-aaa', 'c1', tiers(1), 100);
        await appendOne(a, DIR, 'desktop-aaa', 'c1', tiers(2), 200);
        const scan = await scanJsonl(a, [jsonlPathFor(DIR, 'desktop-aaa')]);
        const got = await readRecordAt(a, DIR, scan.map.get('c1')!);
        expect(got && tiersEqual(got, tiers(2))).toBe(true);
    });

    it('two writers, same id: union resolves deterministically (lex-larger device)', async () => {
        const a = adapter();
        await appendOne(a, DIR, 'desktop-aaa', 'shared', tiers(1), 100);
        await appendOne(a, DIR, 'mobile-zzz', 'shared', tiers(2), 100); // equal mtime → lex-larger device wins
        const paths = await listDeviceJsonls(a, DIR);
        const scan = await scanJsonl(a, paths);
        expect(scan.map.get('shared')!.shard).toBe('mobile-zzz');
        // ...and the loser order doesn't matter: scanning reversed gives the same winner.
        const scan2 = await scanJsonl(a, [...paths].reverse());
        expect(scan2.map.get('shared')!.shard).toBe('mobile-zzz');
    });

    it('same-device tombstone hides that device’s record', async () => {
        const a = adapter();
        await appendOne(a, DIR, 'desktop-aaa', 'c1', tiers(1), 100);
        await appendTombstone(a, DIR, 'desktop-aaa', 'c1', 200);
        const scan = await scanJsonl(a, [jsonlPathFor(DIR, 'desktop-aaa')]);
        expect(scan.map.has('c1')).toBe(false);
        expect(scan.tombstoneIds.has('c1')).toBe(true);
    });

    it('a device’s tombstone does NOT suppress another device’s live record', async () => {
        const a = adapter();
        await appendOne(a, DIR, 'desktop-aaa', 'shared', tiers(1), 100);
        await appendOne(a, DIR, 'mobile-bbb', 'shared', tiers(2), 100);
        await appendTombstone(a, DIR, 'desktop-aaa', 'shared', 300); // desktop deletes its own copy
        const scan = await scanJsonl(a, await listDeviceJsonls(a, DIR));
        expect(scan.map.has('shared')).toBe(true);
        expect(scan.map.get('shared')!.shard).toBe('mobile-bbb'); // mobile's copy survives
        const got = await readRecordAt(a, DIR, scan.map.get('shared')!);
        expect(got && tiersEqual(got, tiers(2))).toBe(true);
    });
});

// ---- partial arrival ----

describe('partial arrival', () => {
    it('out-of-range record is skipped, then self-heals once the bin grows', async () => {
        const a = adapter();
        await bulkAppend(a, DIR, 'desktop-aaa', [{ id: 'c1', tiers: tiers(1), mtime: 1 }]);
        const scan = await scanJsonl(a, [jsonlPathFor(DIR, 'desktop-aaa')]);
        const entry = scan.map.get('c1')!;

        // Simulate the jsonl arriving before its shard bytes: truncate the bin.
        await a.writeBinary(shardPathFor(DIR, 'desktop-aaa', entry.seq), new ArrayBuffer(VEC_BYTES - 1));
        expect(await readRecordAt(a, DIR, entry)).toBeNull();

        // The shard finishes syncing → the same entry now resolves.
        await a.writeBinary(shardPathFor(DIR, 'desktop-aaa', entry.seq), encodeRecord(tiers(1)).buffer as ArrayBuffer);
        const got = await readRecordAt(a, DIR, entry);
        expect(got && tiersEqual(got, tiers(1))).toBe(true);
    });
});

// ---- conflict-copy hygiene ----

describe('conflict copies', () => {
    it('listDeviceJsonls ignores sync conflict copies', async () => {
        const a = adapter();
        await appendOne(a, DIR, 'desktop-aaa', 'c1', tiers(1), 1);
        // A provider-spawned conflict copy carries a space and never matches the strict glob.
        await a.write(`${DIR}/index.desktop-aaa 2.jsonl`, 'garbage\n');
        const paths = await listDeviceJsonls(a, DIR);
        expect(paths).toEqual([jsonlPathFor(DIR, 'desktop-aaa')]);
    });
});

// ---- reap enumeration: must find a device by ANY artifact, not jsonl alone ----

describe('listSidecarDeviceIds (reap enumeration)', () => {
    it('unions all four artifact types and dedups to one id per device', async () => {
        const a = adapter();
        await appendOne(a, DIR, 'desktop-aaa', 'c1', tiers(1), 1); // shards + jsonl
        await writeDeviceMeta(a, DIR, meta('desktop-aaa'));        // meta
        await a.writeBinary(bm25PathFor(DIR, 'desktop-aaa'), new ArrayBuffer(4)); // bm25
        expect(await listSidecarDeviceIds(a, DIR)).toEqual(['desktop-aaa']); // four files → one id
    });

    it('surfaces a device whose jsonl is gone but meta/bm25 linger (the leak case)', async () => {
        const a = adapter();
        // A half-cleared dead device: jsonl + shards already removed, meta + bm25 left behind
        // (clearDevice interrupted, or a sync provider dropped the jsonl deletion first).
        await writeDeviceMeta(a, DIR, meta('mobile-dead'));
        await a.writeBinary(bm25PathFor(DIR, 'mobile-dead'), new ArrayBuffer(4));
        expect(await listDeviceJsonls(a, DIR)).toEqual([]);                  // jsonl-keyed reap MISSES it (the bug)
        expect(await listSidecarDeviceIds(a, DIR)).toEqual(['mobile-dead']); // union catches it (the fix)
    });

    it('surfaces a device with only orphan shards', async () => {
        const a = adapter();
        await a.writeBinary(shardPathFor(DIR, 'desktop-orphan', 0), new ArrayBuffer(VEC_BYTES));
        expect(await listSidecarDeviceIds(a, DIR)).toEqual(['desktop-orphan']);
    });

    it('excludes conflict copies and .tmp leftovers (same strict patterns as the scanners)', async () => {
        const a = adapter();
        await appendOne(a, DIR, 'desktop-aaa', 'c1', tiers(1), 1);
        await a.write(`${DIR}/meta.desktop-aaa 2.json`, '{}');     // conflict copy (space)
        await a.write(`${DIR}/meta.desktop-bbb.json.tmp`, '{}');   // crashed atomic meta write
        await a.write(`${DIR}/bm25.desktop-ccc.json.gz.tmp`, '');  // crashed atomic bm25 write
        expect(await listSidecarDeviceIds(a, DIR)).toEqual(['desktop-aaa']);
    });

    it('empty / missing dir → []', async () => {
        expect(await listSidecarDeviceIds(adapter(), DIR)).toEqual([]);
    });
});

describe('clearDevice completeness (reap can fully reclaim a dead device)', () => {
    it('removes ALL four artifact types and leaves other devices untouched', async () => {
        const a = adapter();
        await appendOne(a, DIR, 'mobile-dead', 'c1', tiers(1), 1);   // shards + jsonl
        await writeDeviceMeta(a, DIR, meta('mobile-dead'));          // meta
        await a.writeBinary(bm25PathFor(DIR, 'mobile-dead'), new ArrayBuffer(4)); // bm25
        await appendOne(a, DIR, 'desktop-keep', 'c2', tiers(2), 1);  // a peer that must survive
        await writeDeviceMeta(a, DIR, meta('desktop-keep'));

        await clearDevice(a, DIR, 'mobile-dead');

        expect(await listSidecarDeviceIds(a, DIR)).toEqual(['desktop-keep']); // dead device fully gone
        expect(await a.exists(metaPathFor(DIR, 'mobile-dead'))).toBe(false);
        expect(await a.exists(bm25PathFor(DIR, 'mobile-dead'))).toBe(false);
        expect(await a.exists(jsonlPathFor(DIR, 'mobile-dead'))).toBe(false);
    });

    it('idempotent: a second clear of a half-cleared device is a no-op, not an error', async () => {
        const a = adapter();
        // Only meta + bm25 remain (jsonl/shards already gone) — the union re-finds it…
        await writeDeviceMeta(a, DIR, meta('mobile-dead'));
        await a.writeBinary(bm25PathFor(DIR, 'mobile-dead'), new ArrayBuffer(4));
        await clearDevice(a, DIR, 'mobile-dead');               // …and clearDevice finishes it
        await clearDevice(a, DIR, 'mobile-dead');               // re-running is harmless (exists-guarded)
        expect(await listSidecarDeviceIds(a, DIR)).toEqual([]);
    });
});

// ---- the reap DECISION: union enumeration must not over-reap ----

describe('reap decision: union enumeration × metaAccepts gate', () => {
    // Mirrors reapDeadIdentitySidecars()'s filter at the file-format layer (the real
    // method needs the whole orchestrator). Pins the airtight property the wider
    // enumeration introduces: a CURRENT-identity peer discoverable ONLY via meta (no
    // jsonl) must be KEPT — widening enumeration past jsonl must reclaim stale leftovers
    // WITHOUT ever deleting a valid peer. Self is skipped regardless of identity.
    it('keeps current-identity + self, reaps only the stale device', async () => {
        const a = adapter();
        const SELF = 'desktop-self';
        const expect_ = { modelId: 'm', revision: null, chunkerVersion: 1, dim: Q_BYTES }; // == meta()'s identity
        await writeDeviceMeta(a, DIR, meta(SELF));                                   // self: meta only
        await writeDeviceMeta(a, DIR, meta('desktop-current'));                      // current peer, meta-only (no jsonl)
        await writeDeviceMeta(a, DIR, { ...meta('mobile-stale'), modelId: 'old' }); // stale identity…
        await a.writeBinary(bm25PathFor(DIR, 'mobile-stale'), new ArrayBuffer(4));   // …with a lingering bm25

        const reaped: string[] = [];
        for (const dev of await listSidecarDeviceIds(a, DIR)) {
            if (dev === SELF) continue;                                              // never reap self
            if (metaAccepts(await readDeviceMeta(a, DIR, dev), expect_)) continue;   // current peer — keep
            await clearDevice(a, DIR, dev);
            reaped.push(dev);
        }

        expect(reaped).toEqual(['mobile-stale']);                                    // only the stale one
        expect((await listSidecarDeviceIds(a, DIR)).sort()).toEqual(['desktop-current', 'desktop-self']);
    });
});

// ---- shard rolling ----

describe('shard roll boundary', () => {
    it('bulkAppend rolls to seq 1 past the cap and cross-shard read works', async () => {
        const a = adapter();
        const n = MAX_VECTORS_PER_SHARD + 5;
        const recs = Array.from({ length: n }, (_, i) => ({ id: `c${i}`, tiers: tiers(i % 50), mtime: i }));
        const refs = await bulkAppend(a, DIR, 'desktop-aaa', recs);
        // First MAX go to seq 0, the overflow to seq 1.
        expect(refs[MAX_VECTORS_PER_SHARD - 1].seq).toBe(0);
        expect(refs[MAX_VECTORS_PER_SHARD].seq).toBe(1);
        expect(refs[MAX_VECTORS_PER_SHARD].off).toBe(0);
        const scan = await scanJsonl(a, [jsonlPathFor(DIR, 'desktop-aaa')]);
        expect(scan.map.size).toBe(n);
        // A record on the rolled shard reads back correctly.
        const last = scan.map.get(`c${n - 1}`)!;
        expect(last.seq).toBe(1);
        const got = await readRecordAt(a, DIR, last);
        expect(got && tiersEqual(got, tiers((n - 1) % 50))).toBe(true);
    });
});

// ---- crash recovery ----

describe('sweepOrphanTmpFiles', () => {
    it('deletes a stale .tmp when the real .bin exists, restores it when missing', async () => {
        const a = adapter();
        // Stale: both real and tmp present → tmp deleted.
        await a.writeBinary(`${DIR}/embeddings.desktop-aaa.0.bin`, new ArrayBuffer(VEC_BYTES));
        await a.writeBinary(`${DIR}/embeddings.desktop-aaa.0.bin.tmp`, new ArrayBuffer(VEC_BYTES));
        // Orphan: only tmp present → restored.
        await a.writeBinary(`${DIR}/embeddings.desktop-aaa.1.bin.tmp`, new ArrayBuffer(VEC_BYTES));

        const n = await sweepOrphanTmpFiles(a, DIR, 'desktop-aaa');
        expect(n).toBe(2);
        expect(await a.exists(`${DIR}/embeddings.desktop-aaa.0.bin.tmp`)).toBe(false);
        expect(await a.exists(`${DIR}/embeddings.desktop-aaa.1.bin.tmp`)).toBe(false);
        expect(await a.exists(`${DIR}/embeddings.desktop-aaa.1.bin`)).toBe(true);
    });

    it('also recovers an orphan .jsonl.tmp (restores when the real jsonl is missing)', async () => {
        const a = adapter();
        await a.write(`${DIR}/index.desktop-aaa.jsonl.tmp`, '{"id":"c1"}\n');
        const n = await sweepOrphanTmpFiles(a, DIR, 'desktop-aaa');
        expect(n).toBe(1);
        expect(await a.exists(`${DIR}/index.desktop-aaa.jsonl.tmp`)).toBe(false);
        expect(await a.exists(`${DIR}/index.desktop-aaa.jsonl`)).toBe(true);
        expect(await a.read(`${DIR}/index.desktop-aaa.jsonl`)).toBe('{"id":"c1"}\n');
    });

    it('also recovers an orphan .json.tmp meta (so an atomic writeDeviceMeta crash self-heals)', async () => {
        const a = adapter();
        await a.write(`${DIR}/meta.desktop-aaa.json.tmp`, '{"format":1}');
        const n = await sweepOrphanTmpFiles(a, DIR, 'desktop-aaa');
        expect(n).toBe(1);
        expect(await a.exists(`${DIR}/meta.desktop-aaa.json.tmp`)).toBe(false);
        expect(await a.exists(`${DIR}/meta.desktop-aaa.json`)).toBe(true);
        // A .json.tmp must not be confused with a .jsonl.tmp (distinct suffixes).
        expect(await a.read(`${DIR}/meta.desktop-aaa.json`)).toBe('{"format":1}');
    });

    it('leaves a PEER device\'s .tmp files untouched (never promotes/deletes another device\'s in-flight write)', async () => {
        const a = adapter();
        // Our own device: stale .tmp alongside real → gets swept.
        await a.writeBinary(`${DIR}/embeddings.desktop-self.0.bin`, new ArrayBuffer(VEC_BYTES));
        await a.writeBinary(`${DIR}/embeddings.desktop-self.0.bin.tmp`, new ArrayBuffer(VEC_BYTES));
        // A peer mid-write: orphan .tmp with no real file yet — must NOT be
        // rename-restored (could promote not-fully-synced bytes) or deleted.
        await a.writeBinary(`${DIR}/embeddings.mobile-peer.0.bin.tmp`, new ArrayBuffer(VEC_BYTES));
        await a.write(`${DIR}/index.mobile-peer.jsonl.tmp`, '{"id":"peer"}\n');

        const n = await sweepOrphanTmpFiles(a, DIR, 'desktop-self');
        expect(n).toBe(1); // only our own stale .tmp acted on
        expect(await a.exists(`${DIR}/embeddings.desktop-self.0.bin.tmp`)).toBe(false);
        // Peer files: completely untouched, still exactly as the peer left them.
        expect(await a.exists(`${DIR}/embeddings.mobile-peer.0.bin.tmp`)).toBe(true);
        expect(await a.exists(`${DIR}/embeddings.mobile-peer.0.bin`)).toBe(false);
        expect(await a.exists(`${DIR}/index.mobile-peer.jsonl.tmp`)).toBe(true);
        expect(await a.exists(`${DIR}/index.mobile-peer.jsonl`)).toBe(false);
    });
});

// ---- jsonl durability: atomic read+rewrite fallback when append() fails ----

describe('appendJsonlLine fallback (append unsupported)', () => {
    class NoAppendAdapter extends FakeAdapter {
        async append(): Promise<void> {
            throw new Error('append unsupported (iOS WKWebView / iCloud contention)');
        }
    }

    it('preserves prior records via tmp+rename, leaving no orphan .tmp', async () => {
        const a = new NoAppendAdapter() as unknown as DataAdapter;
        // First record creates the file (write branch). The second finds the file
        // present → append() throws → the atomic read+rewrite fallback runs.
        await appendOne(a, DIR, 'desktop-aaa', 'c1', tiers(1), 100);
        await appendOne(a, DIR, 'desktop-aaa', 'c2', tiers(2), 200);

        const scan = await scanJsonl(a, [jsonlPathFor(DIR, 'desktop-aaa')]);
        expect([...scan.map.keys()].sort()).toEqual(['c1', 'c2']);
        const got = await readRecordAt(a, DIR, scan.map.get('c2')!);
        expect(got && tiersEqual(got, tiers(2))).toBe(true);

        // The atomic rewrite renames its tmp into place — no dangling .jsonl.tmp.
        const ls = await a.list(DIR);
        expect(ls.files.some(f => f.endsWith('.jsonl.tmp'))).toBe(false);
    });
});

// ---- periodic-reconcile change signature ----

describe('sidecarDirSignature', () => {
    it('is stable when nothing changes and shifts on append / new device file', async () => {
        const a = adapter();
        await appendOne(a, DIR, 'desktop-aaa', 'c1', tiers(1), 100);
        const s1 = await sidecarDirSignature(a, DIR);
        expect(await sidecarDirSignature(a, DIR)).toBe(s1); // no-op → unchanged

        await appendOne(a, DIR, 'desktop-aaa', 'c2', tiers(2), 200); // grows the jsonl
        const s2 = await sidecarDirSignature(a, DIR);
        expect(s2).not.toBe(s1);

        await appendOne(a, DIR, 'mobile-zzz', 'c3', tiers(3), 300); // a new device file appears
        const s3 = await sidecarDirSignature(a, DIR);
        expect(s3).not.toBe(s2);
    });

    it('is empty for an absent index dir', async () => {
        const a = adapter();
        expect(await sidecarDirSignature(a, 'no/such/dir')).toBe('');
    });

    it('excludes the local device so its own appends do not move the signature (but a peer append does)', async () => {
        const a = adapter();
        await appendOne(a, DIR, 'desktop-aaa', 'c1', tiers(1), 100);  // a peer
        await appendOne(a, DIR, 'mobile-self', 'm1', tiers(2), 200);  // us
        const s1 = await sidecarDirSignature(a, DIR, 'mobile-self');

        // A catch-up burst appending to our OWN jsonl must not move the self-excluded
        // signature — otherwise the device wakes its own whole-vault reconcile.
        await appendOne(a, DIR, 'mobile-self', 'm2', tiers(3), 300);
        expect(await sidecarDirSignature(a, DIR, 'mobile-self')).toBe(s1);

        // A PEER append still must move it (real remote arrival → reconcile).
        await appendOne(a, DIR, 'desktop-aaa', 'c2', tiers(4), 400);
        expect(await sidecarDirSignature(a, DIR, 'mobile-self')).not.toBe(s1);
    });

    it('the self-exclusion stops shouldReconcileSidecar firing on an own-jsonl-only change', async () => {
        const a = adapter();
        await appendOne(a, DIR, 'desktop-aaa', 'c1', tiers(1), 100);
        await appendOne(a, DIR, 'mobile-self', 'm1', tiers(2), 200);
        const persisted = await sidecarDirSignature(a, DIR, 'mobile-self');

        await appendOne(a, DIR, 'mobile-self', 'm2', tiers(3), 300);  // our own burst
        const live = await sidecarDirSignature(a, DIR, 'mobile-self');
        // Populated store + unchanged self-excluded sig → no expensive sweep.
        expect(shouldReconcileSidecar(live, persisted, false)).toBe(false);
        // But an empty store still forces recovery regardless of the sig.
        expect(shouldReconcileSidecar(live, persisted, true)).toBe(true);
    });

    it('a peer meta rewrite alone (no jsonl append) still moves the signature', async () => {
        const a = adapter();
        await appendOne(a, DIR, 'desktop-aaa', 'c1', tiers(1), 100);
        await writeDeviceMeta(a, DIR, meta('desktop-aaa'));
        const s1 = await sidecarDirSignature(a, DIR);

        // A version-refused producer rewriting ONLY its meta (e.g. a chunkerVersion
        // bump after a full reindex, no new jsonl line yet) must be picked up —
        // this is the meta-file gap: omitting meta.<dev>.json left such an update
        // invisible until the next jsonl append. (chunkerVersion goes 1→42, not
        // 1→2, so the JSON byte length actually changes — the fake adapter's
        // stat() doesn't track mtime, only size.)
        await writeDeviceMeta(a, DIR, { ...meta('desktop-aaa'), chunkerVersion: 42 });
        const s2 = await sidecarDirSignature(a, DIR);
        expect(s2).not.toBe(s1);
    });

    it('excludes the local device\'s own meta too (self-exclusion covers meta, not just jsonl)', async () => {
        const a = adapter();
        await appendOne(a, DIR, 'desktop-aaa', 'c1', tiers(1), 100);   // a peer
        await writeDeviceMeta(a, DIR, meta('mobile-self'));            // us
        const s1 = await sidecarDirSignature(a, DIR, 'mobile-self');

        // Our own meta rewrite must not move the self-excluded signature.
        await writeDeviceMeta(a, DIR, { ...meta('mobile-self'), chunkerVersion: 42 });
        expect(await sidecarDirSignature(a, DIR, 'mobile-self')).toBe(s1);

        // A peer's meta rewrite still must move it.
        await writeDeviceMeta(a, DIR, { ...meta('desktop-aaa'), chunkerVersion: 42 });
        expect(await sidecarDirSignature(a, DIR, 'mobile-self')).not.toBe(s1);
    });
});

// ---- record shape guard (C1): valid-JSON-but-wrong-shape lines are skipped ----

describe('isValidRecord + readJsonl shape guard', () => {
    it('skips a valid-JSON line with the wrong shape, keeps surrounding records', async () => {
        const a = adapter();
        await appendOne(a, DIR, 'desktop-aaa', 'good', tiers(1), 100);
        // A torn mid-write can leave valid JSON whose shape is wrong (non-numeric off).
        await a.append(
            jsonlPathFor(DIR, 'desktop-aaa'),
            JSON.stringify({ id: 'bad', shard: 'desktop-aaa', seq: 0, off: 'NaN', mtime: 1 }) + '\n',
        );
        const scan = await scanJsonl(a, [jsonlPathFor(DIR, 'desktop-aaa')]);
        expect(scan.map.has('good')).toBe(true);
        expect(scan.map.has('bad')).toBe(false);
        expect(scan.skippedLines).toBe(1);
    });

    it('discriminates tombstone vs vector shape', () => {
        // Vector record: needs numeric seq/off.
        expect(isValidRecord({ id: 'a', shard: 'd', seq: 0, off: 0, mtime: 1 })).toBe(true);
        expect(isValidRecord({ id: 'a', shard: 'd', off: 0, mtime: 1 })).toBe(false); // missing seq
        expect(isValidRecord({ id: 'a', shard: 'd', seq: 0, off: '0', mtime: 1 })).toBe(false); // off not numeric
        // Tombstone: id/shard/mtime suffice, no seq/off required.
        expect(isValidRecord({ id: 'a', shard: 'd', tombstone: true, mtime: 1 })).toBe(true);
        // Common requirements.
        expect(isValidRecord({ shard: 'd', seq: 0, off: 0, mtime: 1 })).toBe(false); // missing id
        expect(isValidRecord(null)).toBe(false);
        expect(isValidRecord('not an object')).toBe(false);
    });
});

// ---- device meta durability (B1 atomicity + C2 validation) ----

describe('device meta', () => {
    const META: SidecarMeta = {
        format: SIDECAR_FORMAT,
        modelId: 'ml97',
        revision: null,
        chunkerVersion: 3,
        dim: 384,
        deviceId: 'desktop-aaa',
        lastFullReindex: null,
    };

    it('writeDeviceMeta is atomic — round-trips and leaves no .json.tmp orphan', async () => {
        const a = adapter();
        await writeDeviceMeta(a, DIR, META);
        const got = await readDeviceMeta(a, DIR, 'desktop-aaa');
        expect(got?.modelId).toBe('ml97');
        const ls = await a.list(DIR);
        expect(ls.files.some(f => f.endsWith('.json.tmp'))).toBe(false);
        expect(ls.files.some(f => f.endsWith('meta.desktop-aaa.json'))).toBe(true);
    });

    it('readDeviceMeta refuses a meta missing modelId (the field metaAccepts gates on)', async () => {
        const a = adapter();
        await a.write(
            `${DIR}/meta.desktop-aaa.json`,
            JSON.stringify({ format: SIDECAR_FORMAT, chunkerVersion: 3, dim: 384, deviceId: 'desktop-aaa', lastFullReindex: null }),
        );
        expect(await readDeviceMeta(a, DIR, 'desktop-aaa')).toBeNull();
    });

    it('writeDeviceMeta preserves a seqFloor the caller does not pass, and honors an explicit one', async () => {
        const a = adapter();
        await writeDeviceMeta(a, DIR, { ...META, seqFloor: 7 });
        // The per-flush F8 meta rewrite constructs a fresh object every pass
        // without knowing the floor — it must ride along, or the first flush
        // after an orphan reclaim erases it and reopens the seq-reuse hole.
        await writeDeviceMeta(a, DIR, META);
        expect((await readDeviceMeta(a, DIR, 'desktop-aaa'))?.seqFloor).toBe(7);
        // An explicit floor wins (a caller that DOES know it can move it).
        await writeDeviceMeta(a, DIR, { ...META, seqFloor: 9 });
        expect((await readDeviceMeta(a, DIR, 'desktop-aaa'))?.seqFloor).toBe(9);
        // Sanitization: a hand-edited/torn value degrades to "no floor", not NaN.
        await a.write(`${DIR}/meta.desktop-aaa.json`, JSON.stringify({ ...META, seqFloor: 'bogus' }));
        expect((await readDeviceMeta(a, DIR, 'desktop-aaa'))?.seqFloor).toBeUndefined();
    });

    it('writeDeviceMeta serializes on the dir lock, so an in-lock floor raise cannot be clobbered', async () => {
        const a = adapter();
        // The F8 race: writeDeviceMeta's preserve-read is a read-modify-write on
        // the meta file. If it ran unlocked, this interleaving — read (no floor) →
        // a fold raises the floor in-lock → write (no floor) — would erase the
        // floor at the moment it is the only guard against seq reuse. Holding the
        // lock while "the fold" writes floor 5 must therefore BLOCK the meta write
        // until release, after which the preserved floor must survive.
        let release!: () => void;
        let metaDone = false;
        const held = withDirLock(DIR, async () => {
            await new Promise<void>(r => { release = r; });
            await a.write(metaPathFor(DIR, 'desktop-aaa'), JSON.stringify({ deviceId: 'desktop-aaa', seqFloor: 5 }));
        });
        const metaWrite = writeDeviceMeta(a, DIR, META).then(() => { metaDone = true; });
        await new Promise(r => setTimeout(r, 0)); // an unlocked write would complete here
        expect(metaDone).toBe(false);             // still queued behind the held lock
        release();
        await held;
        await metaWrite;
        expect((await readDeviceMeta(a, DIR, 'desktop-aaa'))?.seqFloor).toBe(5);
    });

    it('readDeviceMeta refuses a meta missing deviceId (which routes shard/jsonl reads)', async () => {
        const a = adapter();
        await a.write(
            `${DIR}/meta.desktop-aaa.json`,
            JSON.stringify({ format: SIDECAR_FORMAT, modelId: 'ml97', chunkerVersion: 3, dim: 384, lastFullReindex: null }),
        );
        expect(await readDeviceMeta(a, DIR, 'desktop-aaa')).toBeNull();
    });

    it('metaAccepts gates on modelId / revision / format / chunker / dim', async () => {
        const a = adapter();
        await writeDeviceMeta(a, DIR, META);
        const m = await readDeviceMeta(a, DIR, 'desktop-aaa');
        expect(metaAccepts(m, { modelId: 'ml97', revision: null, chunkerVersion: 3, dim: 384 })).toBe(true);
        expect(metaAccepts(m, { modelId: 'other', revision: null, chunkerVersion: 3, dim: 384 })).toBe(false);
        expect(metaAccepts(m, { modelId: 'ml97', revision: 'deadbeefsha', chunkerVersion: 3, dim: 384 })).toBe(false);   // F10: revision mismatch refuses
        expect(metaAccepts(null, { modelId: 'ml97', revision: null, chunkerVersion: 3, dim: 384 })).toBe(false);
    });
});

// shouldReconcileSidecar gates the whole-vault hydrate sweep. The expensive case
// (reChunkLive on the mobile main thread) must run on a real change OR an empty
// store, and must be SKIPPED only when the dir is unchanged AND the index is
// already populated — the combination that caused the iOS crash-relaunch loop to
// re-chunk the entire vault on every onload once the signature is persisted.
describe('shouldReconcileSidecar', () => {
    it('unchanged signature with a populated index skips the sweep', () => {
        expect(shouldReconcileSidecar('a:1:2|b:3:4', 'a:1:2|b:3:4', false)).toBe(false);
    });

    it('changed signature always sweeps', () => {
        expect(shouldReconcileSidecar('a:1:2|b:9:9', 'a:1:2|b:3:4', false)).toBe(true);
    });

    it('first-ever load (no persisted sig) sweeps', () => {
        expect(shouldReconcileSidecar('a:1:2', null, false)).toBe(true);
    });

    it('empty store forces a sweep even when the signature matches — the eviction guard', () => {
        expect(shouldReconcileSidecar('a:1:2', 'a:1:2', true)).toBe(true);
    });

    it('empty store forces a sweep even with no persisted sig', () => {
        expect(shouldReconcileSidecar('', null, true)).toBe(true);
    });
});

// staleSidecarFormat guards compactOwnSidecar + the sidecar-commit write path
// (R2B2 adversarial finding: SIDECAR_FORMAT bumps are excluded from
// identityMatches, so nothing else forces a full reindex to clear an old-stride
// sidecar before compactDevice() would otherwise misdecode it as corrupt).
describe('staleSidecarFormat', () => {
    it('own meta at the current format is not stale', () => {
        expect(staleSidecarFormat({ format: SIDECAR_FORMAT }, SIDECAR_FORMAT)).toBe(false);
    });

    it('own meta at an older format IS stale', () => {
        expect(staleSidecarFormat({ format: SIDECAR_FORMAT - 1 }, SIDECAR_FORMAT)).toBe(true);
    });

    it('no meta at all (fresh install, nothing written yet) is not stale — nothing to misread', () => {
        expect(staleSidecarFormat(null, SIDECAR_FORMAT)).toBe(false);
    });

    it('defaults to the live SIDECAR_FORMAT when no currentFormat arg is passed', () => {
        expect(staleSidecarFormat({ format: SIDECAR_FORMAT })).toBe(false);
        expect(staleSidecarFormat({ format: SIDECAR_FORMAT - 1 })).toBe(true);
    });
});

// ---- compactDevice (R1: sidecar self-compaction) ----

describe('compactDevice', () => {
    const DEV = 'desktop-aaa';
    // Force the rewrite path past both gates so the cases below assert the mechanics.
    const FORCE = { minShardBytes: 0, minDeadRatio: 0 };
    // liveIds thunk fakes: the oracle compactDevice calls under the lock. `live(...ids)`
    // = a complete snapshot keeping exactly those ids; `incomplete` = a skipped-note
    // snapshot the compactor must refuse to delete on.
    const live = (...keep: string[]) => async () => ({ ids: new Set(keep), complete: true });
    const incomplete = async () => ({ ids: new Set<string>(), complete: false });

    // Resolve a device's own jsonl → id→entry map (single device, so this is exactly
    // what that device contributes).
    const resolve = async (a: DataAdapter, dev = DEV) =>
        (await scanJsonl(a, [jsonlPathFor(DIR, dev)])).map;

    it('drops orphaned records (id ∉ keepIds), preserves the live ones byte-exact', async () => {
        const a = adapter();
        await appendOne(a, DIR, DEV, 'c1', tiers(1), 100);
        await appendOne(a, DIR, DEV, 'c2', tiers(2), 100); // orphan — not in keepIds
        await appendOne(a, DIR, DEV, 'c3', tiers(3), 100);

        const r = await compactDevice(a, DIR, DEV, live('c1', 'c3'), FORCE);
        expect(r.compacted).toBe(true);
        expect(r.recordsBefore).toBe(3);
        expect(r.recordsAfter).toBe(2);
        expect(r.shed).toBe(0);

        const map = await resolve(a);
        expect([...map.keys()].sort()).toEqual(['c1', 'c3']);
        // Survivors still decode to their ORIGINAL tiers (verbatim byte copy + consistent
        // jsonl offsets — the post-compaction self-consistency property).
        expect(tiersEqual((await readRecordAt(a, DIR, map.get('c1')!))!, tiers(1))).toBe(true);
        expect(tiersEqual((await readRecordAt(a, DIR, map.get('c3')!))!, tiers(3))).toBe(true);
    });

    it('collapses superseded re-appends to the latest record per id', async () => {
        const a = adapter();
        await appendOne(a, DIR, DEV, 'c1', tiers(10), 100);
        await appendOne(a, DIR, DEV, 'c1', tiers(20), 200); // same id, newer → supersedes

        const r = await compactDevice(a, DIR, DEV, live('c1'), FORCE);
        expect(r.compacted).toBe(true);
        expect(r.recordsBefore).toBe(2);
        expect(r.recordsAfter).toBe(1);
        const map = await resolve(a);
        expect(tiersEqual((await readRecordAt(a, DIR, map.get('c1')!))!, tiers(20))).toBe(true); // the mtime=200 win
    });

    it('writes survivors to FRESH seqs and deletes the old shards (generational)', async () => {
        const a = adapter();
        await appendOne(a, DIR, DEV, 'c1', tiers(1), 100);
        await appendOne(a, DIR, DEV, 'c2', tiers(2), 100); // orphan
        const before = await listDeviceShards(a, DIR, DEV);
        const oldSeq = before[before.length - 1].seq;

        await compactDevice(a, DIR, DEV, live('c1'), FORCE);

        const after = await listDeviceShards(a, DIR, DEV);
        // Every surviving shard is a fresh seq strictly greater than the old max, and the
        // old shard is gone — so a peer's new jsonl can never point into an old shard.
        expect(after.every(s => s.seq > oldSeq)).toBe(true);
        expect(await a.exists(shardPathFor(DIR, DEV, oldSeq))).toBe(false);
    });

    it('reclaims a leftover pre-compaction shard even when the current jsonl has nothing dead (crash between the jsonl swap and the old-shard delete)', async () => {
        const a = adapter();
        await appendOne(a, DIR, DEV, 'c1', tiers(1), 100); // live record, referenced shard at seq 0

        // Simulate a crash mid-compaction: the jsonl swap already landed (it only
        // references seq 0), but an old shard from a PRIOR generation survives on
        // disk with NO jsonl line pointing to it at all — the delete step that
        // should have followed the swap never ran.
        await a.writeBinary(shardPathFor(DIR, DEV, 9), new ArrayBuffer(VEC_BYTES));
        expect(await listDeviceShards(a, DIR, DEV)).toHaveLength(2);

        // Nothing in the CURRENT jsonl is dead — record-count gates alone would
        // return 'nothing-dead' and never touch the unreferenced shard.
        const r = await compactDevice(a, DIR, DEV, live('c1'), FORCE);
        expect(r.reason).toBe('nothing-dead');

        // The crash-leak shard must be gone regardless of that fast path.
        const after = await listDeviceShards(a, DIR, DEV);
        expect(after.map(s => s.seq)).toEqual([0]);
        expect(await a.exists(shardPathFor(DIR, DEV, 9))).toBe(false);
        // The live record survives untouched.
        const map = await resolve(a);
        expect(tiersEqual((await readRecordAt(a, DIR, map.get('c1')!))!, tiers(1))).toBe(true);
    });

    it('below the byte floor: no-op, files left byte-identical (and never calls the oracle)', async () => {
        const a = adapter();
        await appendOne(a, DIR, DEV, 'c1', tiers(1), 100);
        await appendOne(a, DIR, DEV, 'c2', tiers(2), 100); // orphan, but tiny sidecar
        const jsonlBefore = await a.read(jsonlPathFor(DIR, DEV));
        const binBefore = await a.readBinary(shardPathFor(DIR, DEV, 0));

        // A throwing oracle proves the floor short-circuits BEFORE the re-chunk.
        let called = false;
        const r = await compactDevice(a, DIR, DEV, async () => { called = true; throw new Error('should not run'); }, { minShardBytes: 1024 * 1024 });
        expect(r).toMatchObject({ compacted: false, reason: 'below-floor' });
        expect(called).toBe(false);
        expect(await a.read(jsonlPathFor(DIR, DEV))).toBe(jsonlBefore);
        expect(await a.readBinary(shardPathFor(DIR, DEV, 0))).toEqual(binBefore);
    });

    it('below the dead-ratio threshold: no-op', async () => {
        const a = adapter();
        for (const id of ['c1', 'c2', 'c3', 'c4']) await appendOne(a, DIR, DEV, id, tiers(1), 100);
        await appendOne(a, DIR, DEV, 'c5', tiers(1), 100); // 1 orphan of 5 → deadRatio 0.2
        const r = await compactDevice(a, DIR, DEV, live('c1', 'c2', 'c3', 'c4'), { minShardBytes: 0, minDeadRatio: 0.5 });
        expect(r).toMatchObject({ compacted: false, reason: 'below-ratio', recordsBefore: 5, recordsAfter: 4 });
    });

    it('all-live, nothing dead: no-op (idempotent convergence)', async () => {
        const a = adapter();
        await appendOne(a, DIR, DEV, 'c1', tiers(1), 100);
        await appendOne(a, DIR, DEV, 'c2', tiers(2), 100);
        // First compaction drops nothing (both live) → nothing-dead.
        const r1 = await compactDevice(a, DIR, DEV, live('c1', 'c2'), FORCE);
        expect(r1).toMatchObject({ compacted: false, reason: 'nothing-dead' });
        // After a real compaction, re-running is a no-op (the second pass sees no dead).
        await appendOne(a, DIR, DEV, 'c3', tiers(3), 100); // orphan to trigger pass 1
        await compactDevice(a, DIR, DEV, live('c1', 'c2'), FORCE);
        const r2 = await compactDevice(a, DIR, DEV, live('c1', 'c2'), FORCE);
        expect(r2.compacted).toBe(false);
    });

    it('incomplete live-id snapshot → no-op, files untouched (never deletes on a bad oracle)', async () => {
        const a = adapter();
        await appendOne(a, DIR, DEV, 'c1', tiers(1), 100);
        await appendOne(a, DIR, DEV, 'c2', tiers(2), 100); // would look orphaned if we trusted the oracle
        const jsonlBefore = await a.read(jsonlPathFor(DIR, DEV));
        const binBefore = await a.readBinary(shardPathFor(DIR, DEV, 0));

        const r = await compactDevice(a, DIR, DEV, incomplete, FORCE);
        expect(r).toMatchObject({ compacted: false, reason: 'incomplete-rechunk' });
        // A skipped-note snapshot must NOT delete anything — both records survive intact.
        expect(await a.read(jsonlPathFor(DIR, DEV))).toBe(jsonlBefore);
        expect(await a.readBinary(shardPathFor(DIR, DEV, 0))).toEqual(binBefore);
    });

    it('sheds a corrupt survivor mid-rewrite instead of aborting (and counts it)', async () => {
        const a = adapter();
        await appendOne(a, DIR, DEV, 'c1', tiers(1), 100);
        const ref2 = await appendOne(a, DIR, DEV, 'c2', tiers(2), 100); // live but we corrupt it
        await appendOne(a, DIR, DEV, 'c3', tiers(3), 100);              // orphan → triggers the pass
        // Flip a byte at c2's offset so its CRC fails → decodeRecord throws on copy.
        const buf = await a.readBinary(shardPathFor(DIR, DEV, ref2.seq));
        new Uint8Array(buf)[ref2.off] ^= 0xff;
        await a.writeBinary(shardPathFor(DIR, DEV, ref2.seq), buf);

        const r = await compactDevice(a, DIR, DEV, live('c1', 'c2'), FORCE);
        expect(r.compacted).toBe(true);
        expect(r.shed).toBe(1); // c2 was unreadable → shed, not an abort
        const map = await resolve(a);
        // c1 survives; c2 was shed (corrupt = already dead-on-read); c3 dropped (orphan).
        expect([...map.keys()]).toEqual(['c1']);
        expect(tiersEqual((await readRecordAt(a, DIR, map.get('c1')!))!, tiers(1))).toBe(true);
    });

    it('all records orphaned → empty jsonl, every shard reclaimed', async () => {
        const a = adapter();
        await appendOne(a, DIR, DEV, 'c1', tiers(1), 100);
        await appendOne(a, DIR, DEV, 'c2', tiers(2), 200);
        const r = await compactDevice(a, DIR, DEV, live(), FORCE); // none live
        expect(r).toMatchObject({ compacted: true, recordsAfter: 0 });
        expect(await a.read(jsonlPathFor(DIR, DEV))).toBe('');
        expect(await listDeviceShards(a, DIR, DEV)).toHaveLength(0);
    });

    it('touches only the owning device — a peer\'s files are left intact', async () => {
        const a = adapter();
        await appendOne(a, DIR, DEV, 'c1', tiers(1), 100);
        await appendOne(a, DIR, DEV, 'c2', tiers(2), 100); // orphan on DEV
        await appendOne(a, DIR, 'desktop-bbb', 'b1', tiers(9), 100);
        const peerJsonl = await a.read(jsonlPathFor(DIR, 'desktop-bbb'));
        const peerBin = await a.readBinary(shardPathFor(DIR, 'desktop-bbb', 0));

        await compactDevice(a, DIR, DEV, live('c1'), FORCE);

        expect(await a.read(jsonlPathFor(DIR, 'desktop-bbb'))).toBe(peerJsonl);
        expect(await a.readBinary(shardPathFor(DIR, 'desktop-bbb', 0))).toEqual(peerBin);
    });

    it('deviceShardBytes sums the device\'s shard sizes', async () => {
        const a = adapter();
        await appendOne(a, DIR, DEV, 'c1', tiers(1), 100);
        await appendOne(a, DIR, DEV, 'c2', tiers(2), 100);
        expect(await deviceShardBytes(a, DIR, DEV)).toBe(2 * VEC_BYTES);
        expect(await deviceShardBytes(a, DIR, 'nobody')).toBe(0);
    });
});

// ---- append-only shards (1A) ----
// The write-amplification fix: every bulkAppend flush lands in FRESH shard
// file(s) — an existing shard is never read or rewritten. These cases pin the
// mechanism (existing bytes untouched, seq monotonicity through crash leaks),
// the reader contract (union across many small shards, cross-shard supersede),
// and the two halves of the cleanup story (crash-orphan reclaim + the
// generational compaction that coalesces small shards back into dense ones).
describe('append-only shards (1A)', () => {
    const DEV = 'desktop-aaa';

    it('each flush writes a FRESH shard and never touches existing shard bytes', async () => {
        const a = adapter();
        await bulkAppend(a, DIR, DEV, [{ id: 'a1', tiers: tiers(1), mtime: 1 }, { id: 'a2', tiers: tiers(2), mtime: 1 }]);
        const shard0 = new Uint8Array((await a.readBinary(shardPathFor(DIR, DEV, 0))).slice(0)); // snapshot
        const r2 = await bulkAppend(a, DIR, DEV, [{ id: 'b1', tiers: tiers(3), mtime: 2 }]);
        const r3 = await bulkAppend(a, DIR, DEV, [{ id: 'c1', tiers: tiers(4), mtime: 3 }, { id: 'c2', tiers: tiers(5), mtime: 3 }]);

        // Rotation: three flushes → three shards, offsets restart at 0 per shard.
        const shards = await listDeviceShards(a, DIR, DEV);
        expect(shards.map(s => s.seq)).toEqual([0, 1, 2]);
        expect(shards.map(s => s.size)).toEqual([2 * VEC_BYTES, 1 * VEC_BYTES, 2 * VEC_BYTES]);
        expect(r2[0]).toEqual({ seq: 1, off: 0 });
        expect(r3.map(r => r.off)).toEqual([0, VEC_BYTES]);
        // The 1A payoff, pinned byte-for-byte: later flushes left shard 0 alone.
        expect(new Uint8Array(await a.readBinary(shardPathFor(DIR, DEV, 0)))).toEqual(shard0);
    });

    it('readers union many small shards, and a re-append supersedes across shards', async () => {
        const a = adapter();
        for (let f = 0; f < 5; f++) {
            await bulkAppend(a, DIR, DEV, [{ id: `n${f}`, tiers: tiers(f), mtime: f }]);
        }
        await bulkAppend(a, DIR, DEV, [{ id: 'n0', tiers: tiers(99), mtime: 10 }]); // supersede n0

        const scan = await scanJsonl(a, [jsonlPathFor(DIR, DEV)]);
        expect(scan.map.size).toBe(5);
        expect(scan.map.get('n0')!.seq).toBe(5);              // latest shard wins
        for (let f = 1; f < 5; f++) {
            const got = await readRecordAt(a, DIR, scan.map.get(`n${f}`)!);
            expect(got && tiersEqual(got, tiers(f))).toBe(true);
        }
        const n0 = await readRecordAt(a, DIR, scan.map.get('n0')!);
        expect(n0 && tiersEqual(n0, tiers(99))).toBe(true);
    });

    it('a crash between the shard write and the jsonl append leaves a reclaimable orphan, and its seq is never reused', async () => {
        const a = adapter();
        const fake = a as unknown as { append(p: string, d: string): Promise<void>; write(p: string, d: string): Promise<string> };
        // Drive the crash through the PRODUCTION path: the shard lands (binary
        // tmp+rename, untouched by these patches), then every jsonl text channel
        // fails (fresh-file write AND append AND the read-modify-write fallback)
        // — bulkAppend rejects with shard 0 on disk and zero lines referencing
        // it. That is the exact post-crash disk state.
        const realAppend = fake.append.bind(fake);
        const realWrite = fake.write.bind(fake);
        const boom = async () => { throw new Error('EIO: simulated crash at jsonl append'); };
        fake.append = boom;
        fake.write = boom as unknown as typeof realWrite;
        await expect(bulkAppend(a, DIR, DEV, [{ id: 'lost', tiers: tiers(1), mtime: 1 }])).rejects.toThrow(/simulated crash/);
        fake.append = realAppend;
        fake.write = realWrite;

        expect((await listDeviceShards(a, DIR, DEV)).map(s => s.seq)).toEqual([0]);
        expect((await scanJsonl(a, [jsonlPathFor(DIR, DEV)])).map.size).toBe(0);  // no torn read

        // Next flush must NOT reuse the orphan's seq (the never-reuse invariant).
        await bulkAppend(a, DIR, DEV, [{ id: 'b1', tiers: tiers(2), mtime: 2 }]);
        expect((await listDeviceShards(a, DIR, DEV)).map(s => s.seq)).toEqual([0, 1]);

        // compactDevice's crash-leak reclaim deletes the unreferenced shard even
        // when every other gate declines (below-floor here).
        const r = await compactDevice(a, DIR, DEV, async () => ({ ids: new Set(['b1']), complete: true }));
        expect(r.compacted).toBe(false);
        expect(r.reason).toBe('below-floor');
        expect((await listDeviceShards(a, DIR, DEV)).map(s => s.seq)).toEqual([1]);
        const scan = await scanJsonl(a, [jsonlPathFor(DIR, DEV)]);
        const b1 = await readRecordAt(a, DIR, scan.map.get('b1')!);
        expect(b1 && tiersEqual(b1, tiers(2))).toBe(true);
    });

    it('compaction coalesces accumulated small shards into one dense shard and reports the census', async () => {
        const a = adapter();
        // 8 flushes re-appending the same 3 ids: 8 tiny shards, 24 raw records, 3 live.
        for (let f = 0; f < 8; f++) {
            await bulkAppend(a, DIR, DEV, [0, 1, 2].map(k => ({ id: `x${k}`, tiers: tiers(f * 10 + k), mtime: f })));
        }
        expect((await listDeviceShards(a, DIR, DEV)).length).toBe(8);

        const r = await compactDevice(a, DIR, DEV,
            async () => ({ ids: new Set(['x0', 'x1', 'x2']), complete: true }),
            { minDeadRatio: 0.5, minShardBytes: 0 });
        expect(r.compacted).toBe(true);
        expect(r).toMatchObject({ recordsBefore: 24, recordsAfter: 3, shardsBefore: 8, shardsAfter: 1, bytesAfter: 3 * VEC_BYTES });

        // One dense fresh shard (seq 8 — generational, above every old seq), old
        // shards deleted, and every surviving vector still reads back exactly.
        expect((await listDeviceShards(a, DIR, DEV)).map(s => ({ seq: s.seq, size: s.size }))).toEqual([{ seq: 8, size: 3 * VEC_BYTES }]);
        const scan = await scanJsonl(a, [jsonlPathFor(DIR, DEV)]);
        expect(scan.recordCount).toBe(3);
        for (const k of [0, 1, 2]) {
            const got = await readRecordAt(a, DIR, scan.map.get(`x${k}`)!);
            expect(got && tiersEqual(got, tiers(70 + k))).toBe(true);   // the f=7 (latest) generation
        }
    });

    it('reclaiming a max-seq crash orphan persists a seq floor, so the seq is never reused even after deletion', async () => {
        const a = adapter();
        const fake = a as unknown as { append(p: string, d: string): Promise<void>; write(p: string, d: string): Promise<string> };
        await bulkAppend(a, DIR, DEV, [{ id: 'b1', tiers: tiers(1), mtime: 1 }]);   // seq 0, referenced
        // Crash-flush: the orphan lands at seq 1 — the HIGHEST seq, which is the
        // only ordering a single crashed flush can produce (fresh shards always
        // sit above everything referenced) and the one where reclaim lowers the
        // on-disk max. The earlier crash test's ordering (live shard above the
        // orphan) cannot exercise this.
        const realAppend = fake.append.bind(fake);
        const realWrite = fake.write.bind(fake);
        const boom = async () => { throw new Error('EIO: simulated crash at jsonl append'); };
        fake.append = boom;
        fake.write = boom as unknown as typeof realWrite;
        await expect(bulkAppend(a, DIR, DEV, [{ id: 'lost', tiers: tiers(2), mtime: 2 }])).rejects.toThrow(/simulated crash/);
        fake.append = realAppend;
        fake.write = realWrite;
        expect((await listDeviceShards(a, DIR, DEV)).map(s => s.seq)).toEqual([0, 1]);

        const r = await compactDevice(a, DIR, DEV, async () => ({ ids: new Set(['b1']), complete: true }));
        expect(r.reason).toBe('below-floor');
        expect(r.bytesAfter).toBe(r.bytesBefore);   // early returns report the true (unchanged) after-state
        expect((await listDeviceShards(a, DIR, DEV)).map(s => s.seq)).toEqual([0]);  // orphan reclaimed

        // The next flush must allocate ABOVE the reclaimed seq: a peer may still
        // hold the orphan embeddings.<dev>.1.bin, and iCloud has no cross-file
        // ordering — republishing seq 1 with different bytes would let that peer
        // resolve fresh jsonl refs against stale bytes (records carry no id
        // binding, so the CRC passes and the wrong vector hydrates silently).
        await bulkAppend(a, DIR, DEV, [{ id: 'c1', tiers: tiers(3), mtime: 3 }]);
        expect((await listDeviceShards(a, DIR, DEV)).map(s => s.seq)).toEqual([0, 2]);
        const scan = await scanJsonl(a, [jsonlPathFor(DIR, DEV)]);
        const c1 = await readRecordAt(a, DIR, scan.map.get('c1')!);
        expect(c1 && tiersEqual(c1, tiers(3))).toBe(true);
    });

    it('compacting to empty (every id dead) retires the whole seq range instead of resurrecting it', async () => {
        const a = adapter();
        for (let f = 0; f < 3; f++) await bulkAppend(a, DIR, DEV, [{ id: `n${f}`, tiers: tiers(f), mtime: f }]);   // seqs 0..2
        const r = await compactDevice(a, DIR, DEV, async () => ({ ids: new Set<string>(), complete: true }), { minDeadRatio: 0.5, minShardBytes: 0 });
        expect(r).toMatchObject({ compacted: true, reason: 'done', recordsBefore: 3, recordsAfter: 0, shardsBefore: 3, shardsAfter: 0, bytesAfter: 0 });
        expect((await listDeviceShards(a, DIR, DEV)).length).toBe(0);
        // No fresh shard was written by the rewrite, so the on-disk max is gone —
        // only the persisted floor keeps the next flush off the retired seqs.
        await bulkAppend(a, DIR, DEV, [{ id: 'fresh', tiers: tiers(9), mtime: 9 }]);
        expect((await listDeviceShards(a, DIR, DEV)).map(s => s.seq)).toEqual([3]);
    });

    it('clearDevice before a rebuild preserves the seq floor in the kept meta; the reap path removes everything', async () => {
        const a = adapter();
        await writeDeviceMeta(a, DIR, meta(DEV));   // a full reindex starts with a current meta on disk
        for (let f = 0; f < 3; f++) await bulkAppend(a, DIR, DEV, [{ id: `n${f}`, tiers: tiers(f), mtime: f }]);

        await clearDevice(a, DIR, DEV, { preserveSeqFloor: true });
        expect((await listDeviceShards(a, DIR, DEV)).length).toBe(0);
        expect(await a.exists(jsonlPathFor(DIR, DEV))).toBe(false);
        // The meta survives as the floor's carrier — and because it still passes
        // metaAccepts (current identity), a peer's dead-identity reap keeps it,
        // which is what protects the floor until the rebuild's first flush.
        expect((await readDeviceMeta(a, DIR, DEV))?.seqFloor).toBe(3);
        // The rebuild's flushes land above every seq a peer might still hold.
        await bulkAppend(a, DIR, DEV, [{ id: 'rebuilt', tiers: tiers(5), mtime: 5 }]);
        expect((await listDeviceShards(a, DIR, DEV)).map(s => s.seq)).toEqual([3]);

        // Reap (no flag): total removal, floor included — a retired device id
        // never writes again, so nothing needs preserving (and a lingering meta
        // would re-surface the device every reap forever).
        await clearDevice(a, DIR, DEV);
        expect(await a.exists(metaPathFor(DIR, DEV))).toBe(false);
        expect((await listDeviceShards(a, DIR, DEV)).length).toBe(0);
    });
});

// ---- small-shard coalesce (oracle-free shard-count hygiene) ----
// The other half of 1A's cleanup story: compactDevice's byte floor is weeks away
// at per-flush accrual rates while the directory gains one small file per flush.
// coalesceSmallShards folds the small tail into dense shard(s) with no vault
// oracle — superseded duplicates and fragmentation are provable from the
// single-writer jsonl alone — and must leave dense shards byte-untouched.
describe('coalesceSmallShards', () => {
    const DEV = 'desktop-aaa';
    // 600 records × 444 B ≈ 260 KB — crosses the 256 KB default small threshold.
    const BIG_N = 600;

    async function buildMixedSidecar(a: DataAdapter) {
        // One dense shard (seq 0) from a bulk pass…
        await bulkAppend(a, DIR, DEV, Array.from({ length: BIG_N }, (_, i) => ({ id: `big${i}`, tiers: tiers(i % 90), mtime: 1 })));
        // …then per-flush smalls (seqs 1-4): a supersede INTO the dense shard's id
        // space, a new id, a supersede within the smalls, and another new id.
        await bulkAppend(a, DIR, DEV, [{ id: 'big0', tiers: tiers(91), mtime: 2 }]);
        await bulkAppend(a, DIR, DEV, [{ id: 's1', tiers: tiers(92), mtime: 3 }]);
        await bulkAppend(a, DIR, DEV, [{ id: 's1', tiers: tiers(93), mtime: 4 }]);
        await bulkAppend(a, DIR, DEV, [{ id: 's2', tiers: tiers(94), mtime: 5 }]);
    }

    it('folds the small tail into one dense shard, leaves big shards byte-untouched, and collapses superseded lines', async () => {
        const a = adapter();
        await buildMixedSidecar(a);
        const bigBytes = new Uint8Array((await a.readBinary(shardPathFor(DIR, DEV, 0))).slice(0)); // snapshot

        const r = await coalesceSmallShards(a, DIR, DEV, { minSmallShards: 3 });
        expect(r).toMatchObject({ coalesced: true, reason: 'done', smallShards: 4, shardsBefore: 5, shardsAfter: 2, bytesMoved: 3 * VEC_BYTES, shed: 0 });

        // Disk: the dense source shard + ONE fresh dense fold at a generational seq.
        expect((await listDeviceShards(a, DIR, DEV)).map(s => ({ seq: s.seq, size: s.size })))
            .toEqual([{ seq: 0, size: BIG_N * VEC_BYTES }, { seq: 5, size: 3 * VEC_BYTES }]);
        expect(new Uint8Array(await a.readBinary(shardPathFor(DIR, DEV, 0)))).toEqual(bigBytes);

        // The jsonl collapsed to exactly the live set — raw lines === resolved ids —
        // with big-shard refs untouched and moved refs pointing at the fold.
        const scan = await scanJsonl(a, [jsonlPathFor(DIR, DEV)]);
        expect(scan.map.size).toBe(BIG_N + 2);
        expect(scan.recordCount).toBe(BIG_N + 2);
        expect(scan.map.get('big1')).toMatchObject({ seq: 0, off: 1 * VEC_BYTES });
        expect(scan.map.get('big0')!.seq).toBe(5);
        expect(scan.map.get('s1')!.seq).toBe(5);
        // mtime must survive the rewrite VERBATIM on both moved and in-place
        // lines: it drives cross-device resolution (crossDeviceWins) and peers'
        // re-hydration skip — a zeroed mtime lets a peer's stale record beat the
        // fresh moved one; a bumped mtime makes peers re-hydrate the whole fold.
        expect(scan.map.get('big0')!.mtime).toBe(2);
        expect(scan.map.get('s1')!.mtime).toBe(4);
        expect(scan.map.get('big1')!.mtime).toBe(1);
        expect(scan.map.get('big0')!.dim).toBe(scan.map.get('big1')!.dim);

        // Every vector still reads back exactly, including both supersedes.
        const big0 = await readRecordAt(a, DIR, scan.map.get('big0')!);
        expect(big0 && tiersEqual(big0, tiers(91))).toBe(true);
        const s1 = await readRecordAt(a, DIR, scan.map.get('s1')!);
        expect(s1 && tiersEqual(s1, tiers(93))).toBe(true);
        const big7 = await readRecordAt(a, DIR, scan.map.get('big7')!);
        expect(big7 && tiersEqual(big7, tiers(7))).toBe(true);

        // The next flush allocates above the fold — never back into retired seqs.
        await bulkAppend(a, DIR, DEV, [{ id: 'later', tiers: tiers(9), mtime: 9 }]);
        expect((await listDeviceShards(a, DIR, DEV)).map(s => s.seq)).toEqual([0, 5, 6]);
    });

    it('below the count gate it is a listing-only no-op', async () => {
        const a = adapter();
        await bulkAppend(a, DIR, DEV, [{ id: 'x', tiers: tiers(1), mtime: 1 }]);
        await bulkAppend(a, DIR, DEV, [{ id: 'y', tiers: tiers(2), mtime: 2 }]);
        const jsonlBefore = await a.read(jsonlPathFor(DIR, DEV));
        const r = await coalesceSmallShards(a, DIR, DEV, { minSmallShards: 3 });
        expect(r).toMatchObject({ coalesced: false, reason: 'below-count', smallShards: 2, shardsBefore: 2, shardsAfter: 2 });
        expect((await listDeviceShards(a, DIR, DEV)).map(s => s.seq)).toEqual([0, 1]);
        expect(await a.read(jsonlPathFor(DIR, DEV))).toBe(jsonlBefore);
    });

    it('smalls holding zero live records are retired, not resurrected (floor covers the fold-to-nothing case)', async () => {
        const a = adapter();
        for (let f = 0; f < 3; f++) await bulkAppend(a, DIR, DEV, [{ id: 'x', tiers: tiers(f), mtime: f }]); // seqs 0-2
        await appendTombstone(a, DIR, DEV, 'x', 10); // latest for x = delete → zero live records
        const r = await coalesceSmallShards(a, DIR, DEV, { minSmallShards: 3 });
        expect(r).toMatchObject({ coalesced: true, smallShards: 3, shardsAfter: 0, bytesMoved: 0 });
        expect((await listDeviceShards(a, DIR, DEV)).length).toBe(0);
        // No dense shard was written, so only the persisted floor keeps the next
        // flush off the retired seqs (a peer may still hold shards 0-2).
        await bulkAppend(a, DIR, DEV, [{ id: 'fresh', tiers: tiers(9), mtime: 20 }]);
        expect((await listDeviceShards(a, DIR, DEV)).map(s => s.seq)).toEqual([3]);
    });

    it('a crash at the jsonl swap leaves the old state authoritative and the fold as a reclaimable orphan', async () => {
        const a = adapter();
        const fake = a as unknown as { write(p: string, d: string): Promise<void> };
        await buildMixedSidecar(a);

        // Fail every TEXT write: the dense fold shard lands (binary tmp+rename,
        // untouched), then the jsonl swap throws — the commit point is never
        // crossed, exactly a crash between the two.
        const realWrite = fake.write.bind(fake);
        fake.write = async () => { throw new Error('EIO: simulated crash at jsonl swap'); };
        await expect(coalesceSmallShards(a, DIR, DEV, { minSmallShards: 3 })).rejects.toThrow(/simulated crash/);
        fake.write = realWrite;

        // Old jsonl + old shards authoritative: everything resolves as before.
        const scan = await scanJsonl(a, [jsonlPathFor(DIR, DEV)]);
        expect(scan.map.size).toBe(BIG_N + 2);
        expect(scan.map.get('s1')!.seq).toBe(3);   // still the pre-fold ref
        const s1 = await readRecordAt(a, DIR, scan.map.get('s1')!);
        expect(s1 && tiersEqual(s1, tiers(93))).toBe(true);
        // The orphaned fold shard (seq 5) is on disk, unreferenced — the crash-leak
        // class compactDevice reclaims, floor-first, so its seq is never reused.
        expect((await listDeviceShards(a, DIR, DEV)).map(s => s.seq)).toEqual([0, 1, 2, 3, 4, 5]);
        await compactDevice(a, DIR, DEV, async () => ({ ids: new Set<string>(), complete: true }));
        expect((await listDeviceShards(a, DIR, DEV)).map(s => s.seq)).toEqual([0, 1, 2, 3, 4]);
        await bulkAppend(a, DIR, DEV, [{ id: 'later', tiers: tiers(9), mtime: 9 }]);
        expect((await listDeviceShards(a, DIR, DEV)).map(s => s.seq)).toEqual([0, 1, 2, 3, 4, 6]);
    });

    it('a mid-fold write failure rolls back landed dest shards FLOOR-FIRST, so their seqs are never reused', async () => {
        const a = adapter();
        // 16 small flushes totaling more records than one shard holds, so the fold
        // needs TWO dest shards — the only shape where a later write can fail after
        // an earlier dest shard has already landed (and possibly synced out).
        const PER = Math.ceil((MAX_VECTORS_PER_SHARD + 100) / 16);
        for (let f = 0; f < 16; f++) {
            await bulkAppend(a, DIR, DEV, Array.from({ length: PER }, (_, i) => ({ id: `n${f}-${i}`, tiers: tiers((f * PER + i) % 90), mtime: f + 1 })));
        }
        const seqsBefore = (await listDeviceShards(a, DIR, DEV)).map(s => s.seq);
        expect(seqsBefore).toEqual(Array.from({ length: 16 }, (_, i) => i));
        const jsonlBefore = await a.read(jsonlPathFor(DIR, DEV));

        // Fail the SECOND dest-shard write; the first lands and is renamed into place.
        const fake = a as unknown as { writeBinary(p: string, b: ArrayBuffer): Promise<void> };
        const realWriteBinary = fake.writeBinary.bind(fake);
        let binWrites = 0;
        fake.writeBinary = async (p: string, b: ArrayBuffer) => {
            if (++binWrites === 2) throw new Error('ENOSPC: simulated failure at dest shard 2');
            return realWriteBinary(p, b);
        };
        await expect(coalesceSmallShards(a, DIR, DEV, { smallShardBytes: 1024 * 1024, minSmallShards: 16 })).rejects.toThrow(/ENOSPC/);
        fake.writeBinary = realWriteBinary;

        // Old state authoritative: sources intact, jsonl untouched, landed dest
        // shard (seq 16) rolled back.
        expect((await listDeviceShards(a, DIR, DEV)).map(s => s.seq)).toEqual(seqsBefore);
        expect(await a.read(jsonlPathFor(DIR, DEV))).toBe(jsonlBefore);
        // The rollback burned seq 16 — it may have synced out between its rename
        // and its remove — so the floor must keep every later writer above it.
        await bulkAppend(a, DIR, DEV, [{ id: 'later', tiers: tiers(1), mtime: 99 }]);
        expect((await listDeviceShards(a, DIR, DEV)).map(s => s.seq)).toEqual([...seqsBefore, 17]);
    });
});
