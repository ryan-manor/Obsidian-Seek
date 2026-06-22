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
    sweepOrphanTmpFiles,
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

// Deterministic tiers for a seed. `s` is fround-ed so the f32 round-trip is exact.
function tiers(seed: number): TierBytes {
    const q = new Int8Array(Q_BYTES);
    for (let i = 0; i < Q_BYTES; i++) q[i] = (((seed * 31 + i) % 255) - 127) as number;
    const sign = new Uint8Array(SIGN_BYTES);
    for (let i = 0; i < SIGN_BYTES; i++) sign[i] = (seed + i * 7) & 0xff;
    return { q, s: Math.fround(0.001 + seed * 1e-5), sign };
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
// readable while routing through the real append path (pickActiveShard →
// encodeRecord → writeBinaryAtomic → appendJsonlLine).
async function appendOne(a: DataAdapter, dir: string, dev: string, id: string, t: TierBytes, mtime: number) {
    const [ref] = await bulkAppend(a, dir, dev, [{ id, tiers: t, mtime }]);
    return ref;
}

// ---- codec ----

describe('record codec', () => {
    it('encode → decode round-trips byte-exact (including f32 scale)', () => {
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

        // q and sign are byte-identical; the scale is stored f32 (it only ever
        // multiplies an int8, so f32 precision is orders below the quantization
        // error already present). So compare the scale at f32 resolution.
        expect(dec.s).toBe(Math.fround(q.s));
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

        const n = await sweepOrphanTmpFiles(a, DIR);
        expect(n).toBe(2);
        expect(await a.exists(`${DIR}/embeddings.desktop-aaa.0.bin.tmp`)).toBe(false);
        expect(await a.exists(`${DIR}/embeddings.desktop-aaa.1.bin.tmp`)).toBe(false);
        expect(await a.exists(`${DIR}/embeddings.desktop-aaa.1.bin`)).toBe(true);
    });

    it('also recovers an orphan .jsonl.tmp (restores when the real jsonl is missing)', async () => {
        const a = adapter();
        await a.write(`${DIR}/index.desktop-aaa.jsonl.tmp`, '{"id":"c1"}\n');
        const n = await sweepOrphanTmpFiles(a, DIR);
        expect(n).toBe(1);
        expect(await a.exists(`${DIR}/index.desktop-aaa.jsonl.tmp`)).toBe(false);
        expect(await a.exists(`${DIR}/index.desktop-aaa.jsonl`)).toBe(true);
        expect(await a.read(`${DIR}/index.desktop-aaa.jsonl`)).toBe('{"id":"c1"}\n');
    });

    it('also recovers an orphan .json.tmp meta (so an atomic writeDeviceMeta crash self-heals)', async () => {
        const a = adapter();
        await a.write(`${DIR}/meta.desktop-aaa.json.tmp`, '{"format":1}');
        const n = await sweepOrphanTmpFiles(a, DIR);
        expect(n).toBe(1);
        expect(await a.exists(`${DIR}/meta.desktop-aaa.json.tmp`)).toBe(false);
        expect(await a.exists(`${DIR}/meta.desktop-aaa.json`)).toBe(true);
        // A .json.tmp must not be confused with a .jsonl.tmp (distinct suffixes).
        expect(await a.read(`${DIR}/meta.desktop-aaa.json`)).toBe('{"format":1}');
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
