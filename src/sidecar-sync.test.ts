// Hydrate engine integration tests: drive hydrateFromSidecar against real sidecar
// files (an in-memory adapter) and an in-memory stand-in for IndexStore. This is
// the automated form of the Phase C "wipe IDB → hydrate → counts match" milestone,
// plus idempotency, the version gate, partial arrival, and cross-device merge.

import { describe, it, expect } from 'vitest';
import type { DataAdapter } from 'obsidian';
import type { Chunk } from './types';
import type { QuantVec } from './quant';
import { bulkAppend, shardPathFor, Q_BYTES, SIGN_BYTES, SIDECAR_FORMAT, type TierBytes } from './sidecar';
import { writeDeviceMeta } from './sidecar-meta';
import { hydrateFromSidecar, rankAcceptedProducers, probePeerAhead, type ReChunkedNote, type HydrateDeps } from './sidecar-sync';

// ---- in-memory adapter (same shape as sidecar.test.ts) ----

class FakeAdapter {
    files = new Map<string, string>();
    bins = new Map<string, ArrayBuffer>();
    async exists(p: string): Promise<boolean> {
        if (this.files.has(p) || this.bins.has(p)) return true;
        const prefix = p.endsWith('/') ? p : p + '/';
        for (const k of [...this.files.keys(), ...this.bins.keys()]) if (k.startsWith(prefix)) return true;
        return false;
    }
    async mkdir(_p: string): Promise<void> {}
    async read(p: string): Promise<string> {
        const v = this.files.get(p);
        if (v === undefined) throw new Error(`ENOENT ${p}`);
        return v;
    }
    async write(p: string, d: string): Promise<void> {
        this.files.set(p, d);
    }
    async append(p: string, d: string): Promise<void> {
        this.files.set(p, (this.files.get(p) ?? '') + d);
    }
    async readBinary(p: string): Promise<ArrayBuffer> {
        const v = this.bins.get(p);
        if (v === undefined) throw new Error(`ENOENT ${p}`);
        return v;
    }
    async writeBinary(p: string, b: ArrayBuffer): Promise<void> {
        this.bins.set(p, b);
    }
    async rename(from: string, to: string): Promise<void> {
        if (this.bins.has(from)) {
            this.bins.set(to, this.bins.get(from)!);
            this.bins.delete(from);
        } else if (this.files.has(from)) {
            this.files.set(to, this.files.get(from)!);
            this.files.delete(from);
        } else throw new Error(`ENOENT ${from}`);
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

const DIR = '.obsidian/plugins/seek/index';
const EXPECT = { modelId: 'ml97', revision: null, chunkerVersion: 3, dim: 384 };

function tiers(seed: number): TierBytes {
    const q = new Int8Array(Q_BYTES);
    for (let i = 0; i < Q_BYTES; i++) q[i] = (((seed * 17 + i) % 255) - 127) as number;
    const sign = new Uint8Array(SIGN_BYTES);
    for (let i = 0; i < SIGN_BYTES; i++) sign[i] = (seed + i) & 0xff;
    return { q, s: Math.fround(0.01 + seed * 1e-4), sign };
}

function chunk(id: string, notePath: string): Chunk {
    // The hydrate engine reads only chunk_id off the chunk; a minimal shape suffices.
    return { chunk_id: id, note_path: notePath } as unknown as Chunk;
}

// A note spec: note path → its chunk ids.
interface NoteSpec {
    path: string;
    mtime: number;
    ids: string[];
}

async function seedSidecar(a: FakeAdapter, deviceId: string, notes: NoteSpec[], seedBase = 0): Promise<void> {
    const records = notes.flatMap(n => n.ids.map((id, j) => ({ id, tiers: tiers(seedBase + j + n.ids.length), mtime: n.mtime })));
    await bulkAppend(a as unknown as DataAdapter, DIR, deviceId, records);
    await writeDeviceMeta(a as unknown as DataAdapter, DIR, {
        format: SIDECAR_FORMAT,
        modelId: EXPECT.modelId,
        revision: EXPECT.revision,
        chunkerVersion: EXPECT.chunkerVersion,
        dim: EXPECT.dim,
        deviceId,
        lastFullReindex: null,
    });
}

// Build deps backed by an in-memory "IndexStore" + a fixed live-vault re-chunk.
function makeDeps(a: FakeAdapter, live: NoteSpec[], overrides: Partial<HydrateDeps> = {}) {
    const store = new Map<string, { q: QuantVec; bin: Uint8Array }>();
    const fileRecs = new Map<string, { note_path: string; mtimeMs: number; chunk_ids: string[] }>();
    const deps: HydrateDeps = {
        adapter: a as unknown as DataAdapter,
        indexDir: DIR,
        expect: EXPECT,
        reChunk: async (): Promise<ReChunkedNote[]> => live.map(n => ({ notePath: n.path, mtimeMs: n.mtime, chunks: n.ids.map(id => chunk(id, n.path)) })),
        existingIds: async () => new Set(store.keys()),
        putQuantized: async (chunks, t) => chunks.forEach((c, i) => store.set(c.chunk_id, t[i])),
        putFileRecord: async rec => {
            fileRecs.set(rec.note_path, rec);
        },
        ...overrides,
    };
    return { deps, store, fileRecs };
}

describe('hydrateFromSidecar', () => {
    it('wipe → hydrate restores every chunk + file record (Phase C round-trip)', async () => {
        const a = new FakeAdapter();
        const notes: NoteSpec[] = [
            { path: 'a.md', mtime: 100, ids: ['a1', 'a2'] },
            { path: 'b.md', mtime: 200, ids: ['b1'] },
        ];
        await seedSidecar(a, 'desktop-aaa', notes);

        const { deps, store, fileRecs } = makeDeps(a, notes);
        const r = await hydrateFromSidecar(deps);

        expect(r.hydrated).toBe(3);
        expect(r.acceptedProducers).toBe(1);
        expect(store.size).toBe(3);
        expect(fileRecs.get('a.md')!.chunk_ids).toEqual(['a1', 'a2']);
        expect(fileRecs.get('b.md')!.chunk_ids).toEqual(['b1']);
    });

    it('is idempotent: a warm index hydrates nothing', async () => {
        const a = new FakeAdapter();
        const notes: NoteSpec[] = [{ path: 'a.md', mtime: 100, ids: ['a1', 'a2'] }];
        await seedSidecar(a, 'desktop-aaa', notes);
        const { deps, store } = makeDeps(a, notes);
        await hydrateFromSidecar(deps); // first pass fills the store
        const r = await hydrateFromSidecar(deps); // existingIds now covers everything
        expect(r.hydrated).toBe(0);
        expect(store.size).toBe(2);
    });

    it('skips the whole-vault re-chunk when no sidecar id is new to this device', async () => {
        // A peer re-append / compaction / mtime-churn flips sidecarDirSignature but
        // carries only ids we already hold. The cheap pre-gate must short-circuit
        // BEFORE the expensive whole-vault reChunk (which on mobile reads + tokenizes
        // every note on the main thread). We assert reChunk is never called.
        const a = new FakeAdapter();
        const notes: NoteSpec[] = [
            { path: 'a.md', mtime: 100, ids: ['a1', 'a2'] },
            { path: 'b.md', mtime: 200, ids: ['b1'] },
        ];
        await seedSidecar(a, 'desktop-aaa', notes);

        let reChunkCalls = 0;
        // The store already holds every sidecar id → `existing` covers the scan.
        const { deps } = makeDeps(a, notes, {
            existingIds: async () => new Set(['a1', 'a2', 'b1']),
            reChunk: async () => {
                reChunkCalls++;
                return notes.map(n => ({ notePath: n.path, mtimeMs: n.mtime, chunks: n.ids.map(id => chunk(id, n.path)) }));
            },
        });
        const r = await hydrateFromSidecar(deps);

        expect(reChunkCalls).toBe(0);        // the expensive path was skipped
        expect(r.hydrated).toBe(0);
        expect(r.scanned).toBe(3);           // the scan still ran (cheap) — 3 live ids resolved
        expect(r.acceptedProducers).toBe(1); // producer was accepted, just nothing to do
    });

    it('still re-chunks when even one sidecar id is new to this device', async () => {
        // The gate must not OVER-skip: a single missing id means a note may need
        // hydration, so the liveness oracle (reChunk) has to run.
        const a = new FakeAdapter();
        const notes: NoteSpec[] = [
            { path: 'a.md', mtime: 100, ids: ['a1', 'a2'] },
            { path: 'b.md', mtime: 200, ids: ['b1'] },
        ];
        await seedSidecar(a, 'desktop-aaa', notes);

        let reChunkCalls = 0;
        // a1, a2 already indexed; b1 is NEW (a peer embedded a note we lack).
        const { deps, store } = makeDeps(a, notes, {
            existingIds: async () => new Set(['a1', 'a2']),
            reChunk: async () => {
                reChunkCalls++;
                return notes.map(n => ({ notePath: n.path, mtimeMs: n.mtime, chunks: n.ids.map(id => chunk(id, n.path)) }));
            },
        });
        const r = await hydrateFromSidecar(deps);

        expect(reChunkCalls).toBe(1);        // one missing id is enough to re-chunk
        expect(r.hydrated).toBe(1);          // only b1 was hydrated (a.md already indexed)
        expect(store.has('b1')).toBe(true);
    });

    it('version gate refuses an incompatible producer', async () => {
        const a = new FakeAdapter();
        const notes: NoteSpec[] = [{ path: 'a.md', mtime: 100, ids: ['a1'] }];
        await seedSidecar(a, 'desktop-aaa', notes);
        // Overwrite meta with a mismatched chunker version.
        await writeDeviceMeta(a as unknown as DataAdapter, DIR, { format: SIDECAR_FORMAT, modelId: 'ml97', revision: null, chunkerVersion: 2, dim: 384, deviceId: 'desktop-aaa', lastFullReindex: null });

        let refused = '';
        let refusedMetaChunker: number | null = null;
        let refusedExpectChunker: number | null = null;
        const { deps, store } = makeDeps(a, notes, {
            onRefusedProducer: (d, meta, expect) => {
                refused = d;
                refusedMetaChunker = meta?.chunkerVersion ?? null;
                refusedExpectChunker = expect.chunkerVersion;
            },
        });
        const r = await hydrateFromSidecar(deps);
        expect(r.refusedProducers).toBe(1);
        expect(r.hydrated).toBe(0);
        expect(refused).toBe('desktop-aaa');
        // The callback carries the producer meta + this consumer's expectation so
        // the caller can report WHAT is stale (here: chunker v2 vs the expected v3).
        expect(refusedMetaChunker).toBe(2);
        expect(refusedExpectChunker).toBe(3);
        expect(store.size).toBe(0);
        // Producer is OLDER than us (v2 < v3) — we're ahead, not behind. No "update Seek".
        expect(r.peerAhead).toBe(false);
    });

    it('flags peerAhead when a refused producer is NEWER than this build (the v8≠v7 bog-down)', async () => {
        const a = new FakeAdapter();
        const notes: NoteSpec[] = [{ path: 'a.md', mtime: 100, ids: ['a1'] }];
        await seedSidecar(a, 'desktop-aaa', notes);
        // Producer at a HIGHER chunkerVersion (v4) than this consumer can read (v3).
        await writeDeviceMeta(a as unknown as DataAdapter, DIR, { format: SIDECAR_FORMAT, modelId: 'ml97', revision: null, chunkerVersion: 4, dim: 384, deviceId: 'desktop-aaa', lastFullReindex: null });

        const { deps } = makeDeps(a, notes, {});
        const r = await hydrateFromSidecar(deps);
        expect(r.refusedProducers).toBe(1);
        expect(r.hydrated).toBe(0);
        // The peer holds an index this build is too old to use → surface "update Seek"
        // (and gate the mobile re-embed) instead of silently grinding out throwaway chunks.
        expect(r.peerAhead).toBe(true);
    });

    it('skips a note whose chunk bytes have not synced yet (all-or-nothing)', async () => {
        const a = new FakeAdapter();
        const notes: NoteSpec[] = [
            { path: 'a.md', mtime: 100, ids: ['a1', 'a2'] },
            { path: 'b.md', mtime: 200, ids: ['b1'] },
        ];
        await seedSidecar(a, 'desktop-aaa', notes);
        // Truncate the shard so the LAST record's bytes are missing (partial arrival).
        const path = shardPathFor(DIR, 'desktop-aaa', 0);
        const full = await a.readBinary(path);
        await a.writeBinary(path, full.slice(0, full.byteLength - 10));

        const { deps, store, fileRecs } = makeDeps(a, notes);
        const r = await hydrateFromSidecar(deps);
        // b1 is the last record written → its note is skipped; a.md fully hydrates.
        expect(r.skippedPartialNotes).toBe(1);
        expect(store.has('a1') && store.has('a2')).toBe(true);
        expect(fileRecs.has('a.md')).toBe(true);
        expect(fileRecs.has('b.md')).toBe(false); // no file record for the skipped note → reindex will embed it
    });

    it('ignores sidecar records for notes deleted from the live vault', async () => {
        const a = new FakeAdapter();
        const seeded: NoteSpec[] = [
            { path: 'a.md', mtime: 100, ids: ['a1'] },
            { path: 'gone.md', mtime: 100, ids: ['g1', 'g2'] },
        ];
        await seedSidecar(a, 'desktop-aaa', seeded);
        // Live vault no longer has gone.md → its ids aren't reproduced.
        const live: NoteSpec[] = [{ path: 'a.md', mtime: 100, ids: ['a1'] }];
        const { deps, store } = makeDeps(a, live);
        const r = await hydrateFromSidecar(deps);
        expect(r.hydrated).toBe(1);
        expect(store.has('a1')).toBe(true);
        expect(store.has('g1')).toBe(false);
    });

    it('reports hydratedNotePaths for delta dedup (covered reported, miss omitted)', async () => {
        // Dedup uses hydratedNotePaths to decide which dirty files to skip embedding.
        const a = new FakeAdapter();
        await seedSidecar(a, 'desktop-aaa', [{ path: 'covered.md', mtime: 100, ids: ['c1', 'c2'] }]);
        // Two dirty files: one fully in the sidecar, one whose ids are NOT (a local
        // edit no peer has embedded yet → must fall through to the model).
        const dirty: NoteSpec[] = [
            { path: 'covered.md', mtime: 100, ids: ['c1', 'c2'] },
            { path: 'fresh.md', mtime: 300, ids: ['f1'] },
        ];
        const { deps } = makeDeps(a, dirty);
        const r = await hydrateFromSidecar(deps);
        expect(r.hydratedNotePaths).toEqual(['covered.md']); // fresh.md omitted → dedup keeps it for embed
        expect(r.hydrated).toBe(2);
    });

    it('inherits bg display-calibration from the freshest producer', async () => {
        const a = new FakeAdapter();
        await seedSidecar(a, 'desktop-old', [{ path: 'a.md', mtime: 100, ids: ['a1'] }]);
        await writeDeviceMeta(a as unknown as DataAdapter, DIR, {
            format: SIDECAR_FORMAT, modelId: EXPECT.modelId, revision: EXPECT.revision, chunkerVersion: EXPECT.chunkerVersion, dim: EXPECT.dim,
            deviceId: 'desktop-old', lastFullReindex: '2026-06-01T00:00:00Z', bgMean: 0.70, bgStd: 0.03,
        });
        await seedSidecar(a, 'desktop-new', [{ path: 'b.md', mtime: 200, ids: ['b1'] }]);
        await writeDeviceMeta(a as unknown as DataAdapter, DIR, {
            format: SIDECAR_FORMAT, modelId: EXPECT.modelId, revision: EXPECT.revision, chunkerVersion: EXPECT.chunkerVersion, dim: EXPECT.dim,
            deviceId: 'desktop-new', lastFullReindex: '2026-06-16T00:00:00Z', bgMean: 0.76, bgStd: 0.04,
        });
        const { deps } = makeDeps(a, [
            { path: 'a.md', mtime: 100, ids: ['a1'] },
            { path: 'b.md', mtime: 200, ids: ['b1'] },
        ]);
        const r = await hydrateFromSidecar(deps);
        expect(r.bgMean).toBeCloseTo(0.76, 6);   // newest lastFullReindex wins
        expect(r.bgStd).toBeCloseTo(0.04, 6);
    });

    it('picks the freshest producer by epoch, not lexicographic / scan order', async () => {
        const a = new FakeAdapter();
        // 'real' is genuinely newer but its NO-MILLIS ISO string sorts LEXICALLY
        // below the older producer's WITH-MILLIS string ('Z' > '.'), and a third
        // producer carries stats with a null timestamp — all three must lose to
        // the true newest by parsed epoch.
        await seedSidecar(a, 'dev-oldms', [{ path: 'a.md', mtime: 100, ids: ['a1'] }]);
        await writeDeviceMeta(a as unknown as DataAdapter, DIR, {
            format: SIDECAR_FORMAT, modelId: EXPECT.modelId, revision: EXPECT.revision, chunkerVersion: EXPECT.chunkerVersion, dim: EXPECT.dim,
            deviceId: 'dev-oldms', lastFullReindex: '2026-06-01T00:00:00.000Z', bgMean: 0.70, bgStd: 0.03,
        });
        await seedSidecar(a, 'dev-null', [{ path: 'b.md', mtime: 100, ids: ['b1'] }]);
        await writeDeviceMeta(a as unknown as DataAdapter, DIR, {
            format: SIDECAR_FORMAT, modelId: EXPECT.modelId, revision: EXPECT.revision, chunkerVersion: EXPECT.chunkerVersion, dim: EXPECT.dim,
            deviceId: 'dev-null', lastFullReindex: null, bgMean: 0.71, bgStd: 0.05,
        });
        await seedSidecar(a, 'dev-new', [{ path: 'c.md', mtime: 100, ids: ['c1'] }]);
        await writeDeviceMeta(a as unknown as DataAdapter, DIR, {
            format: SIDECAR_FORMAT, modelId: EXPECT.modelId, revision: EXPECT.revision, chunkerVersion: EXPECT.chunkerVersion, dim: EXPECT.dim,
            deviceId: 'dev-new', lastFullReindex: '2026-06-16T00:00:00Z', bgMean: 0.76, bgStd: 0.04,
        });
        const { deps } = makeDeps(a, [
            { path: 'a.md', mtime: 100, ids: ['a1'] },
            { path: 'b.md', mtime: 100, ids: ['b1'] },
            { path: 'c.md', mtime: 100, ids: ['c1'] },
        ]);
        const r = await hydrateFromSidecar(deps);
        expect(r.bgMean).toBeCloseTo(0.76, 6);   // true newest by epoch wins
        expect(r.bgStd).toBeCloseTo(0.04, 6);
    });

    it('inherits stats from a lone producer even with a null timestamp', async () => {
        const a = new FakeAdapter();
        await seedSidecar(a, 'dev-null', [{ path: 'a.md', mtime: 100, ids: ['a1'] }]);
        await writeDeviceMeta(a as unknown as DataAdapter, DIR, {
            format: SIDECAR_FORMAT, modelId: EXPECT.modelId, revision: EXPECT.revision, chunkerVersion: EXPECT.chunkerVersion, dim: EXPECT.dim,
            deviceId: 'dev-null', lastFullReindex: null, bgMean: 0.72, bgStd: 0.03,
        });
        const { deps } = makeDeps(a, [{ path: 'a.md', mtime: 100, ids: ['a1'] }]);
        const r = await hydrateFromSidecar(deps);
        expect(r.bgMean).toBeCloseTo(0.72, 6);
        expect(r.bgStd).toBeCloseTo(0.03, 6);
    });

    it('leaves bg stats undefined when no producer carries them', async () => {
        const a = new FakeAdapter();
        await seedSidecar(a, 'desktop-aaa', [{ path: 'a.md', mtime: 100, ids: ['a1'] }]);
        const { deps } = makeDeps(a, [{ path: 'a.md', mtime: 100, ids: ['a1'] }]);
        const r = await hydrateFromSidecar(deps);
        expect(r.bgMean).toBeUndefined();
        expect(r.bgStd).toBeUndefined();
    });

    it('merges two producers (cross-device hydrate)', async () => {
        const a = new FakeAdapter();
        await seedSidecar(a, 'desktop-aaa', [{ path: 'a.md', mtime: 100, ids: ['a1'] }], 0);
        await seedSidecar(a, 'mobile-bbb', [{ path: 'b.md', mtime: 200, ids: ['b1'] }], 50);
        const live: NoteSpec[] = [
            { path: 'a.md', mtime: 100, ids: ['a1'] },
            { path: 'b.md', mtime: 200, ids: ['b1'] },
        ];
        const { deps, store } = makeDeps(a, live);
        const r = await hydrateFromSidecar(deps);
        expect(r.acceptedProducers).toBe(2);
        expect(r.hydrated).toBe(2);
        expect(store.has('a1') && store.has('b1')).toBe(true);
    });
});

// Cross-device BM25 artifact (Phase 3) source selection. The consumer tries
// producers freshest-first; an incompatible chunker is refused (its ids aren't
// reproducible), and a real timestamp outranks a null one — deterministic regardless
// of file-scan order. The consumer falls THROUGH a freshest producer that has no .gz
// (e.g. mobile), so the ranked list (not a single pick) is what we assert.
describe('rankAcceptedProducers', () => {
    const setMeta = (a: FakeAdapter, deviceId: string, lastFullReindex: string | null, over: Partial<typeof EXPECT> = {}) =>
        writeDeviceMeta(a as unknown as DataAdapter, DIR, { format: SIDECAR_FORMAT, ...EXPECT, ...over, deviceId, lastFullReindex });

    it('returns an empty list when the sidecar is empty', async () => {
        const a = new FakeAdapter();
        expect(await rankAcceptedProducers(a as unknown as DataAdapter, DIR, EXPECT)).toEqual([]);
    });

    it('ranks compatible producers newest-full-reindex first', async () => {
        const a = new FakeAdapter();
        await seedSidecar(a, 'desktop-old', [{ path: 'a.md', mtime: 1, ids: ['a'] }]);
        await seedSidecar(a, 'desktop-new', [{ path: 'b.md', mtime: 1, ids: ['b'] }]);
        await setMeta(a, 'desktop-old', '2026-06-01T00:00:00.000Z');
        await setMeta(a, 'desktop-new', '2026-06-20T00:00:00.000Z');
        expect(await rankAcceptedProducers(a as unknown as DataAdapter, DIR, EXPECT)).toEqual(['desktop-new', 'desktop-old']);
    });

    it('excludes a version-incompatible producer entirely, even if it is newer', async () => {
        const a = new FakeAdapter();
        await seedSidecar(a, 'desktop-compat', [{ path: 'a.md', mtime: 1, ids: ['a'] }]);
        await seedSidecar(a, 'desktop-stale', [{ path: 'b.md', mtime: 1, ids: ['b'] }]);
        await setMeta(a, 'desktop-compat', '2026-06-01T00:00:00.000Z');
        await setMeta(a, 'desktop-stale', '2026-06-20T00:00:00.000Z', { chunkerVersion: EXPECT.chunkerVersion + 1 });
        expect(await rankAcceptedProducers(a as unknown as DataAdapter, DIR, EXPECT)).toEqual(['desktop-compat']);
    });

    it('a real timestamp outranks a null one regardless of scan order', async () => {
        const a = new FakeAdapter();
        await seedSidecar(a, 'desktop-null', [{ path: 'a.md', mtime: 1, ids: ['a'] }]); // null lastFullReindex
        await seedSidecar(a, 'desktop-dated', [{ path: 'b.md', mtime: 1, ids: ['b'] }]);
        await setMeta(a, 'desktop-dated', '2026-06-10T00:00:00.000Z');
        const ranked = await rankAcceptedProducers(a as unknown as DataAdapter, DIR, EXPECT);
        expect(ranked[0]).toBe('desktop-dated');
        expect(ranked).toContain('desktop-null');
    });

    it('keeps a gz-less freshest producer in the list so the consumer can fall through to an older one with a gz', async () => {
        // Models the real bug: a MOBILE device did the newest full reindex (writes
        // meta+jsonl, NO bm25.gz), an older DESKTOP has the gz. The ranking must put
        // mobile first (freshest) AND keep desktop, so the consumer skips the gz-less
        // mobile and loads the desktop gz instead of refitting.
        const a = new FakeAdapter();
        await seedSidecar(a, 'desktop-haz-gz', [{ path: 'a.md', mtime: 1, ids: ['a'] }]);
        await seedSidecar(a, 'mobile-no-gz', [{ path: 'b.md', mtime: 1, ids: ['b'] }]);
        await setMeta(a, 'desktop-haz-gz', '2026-06-01T00:00:00.000Z');
        await setMeta(a, 'mobile-no-gz', '2026-06-20T00:00:00.000Z'); // newer
        expect(await rankAcceptedProducers(a as unknown as DataAdapter, DIR, EXPECT)).toEqual(['mobile-no-gz', 'desktop-haz-gz']);
    });
});

describe('probePeerAhead', () => {
    // Overwrite a seeded producer's meta with a specific chunkerVersion (seedSidecar
    // registers the jsonl + a v3 meta; this re-points only the version).
    const overMeta = (a: FakeAdapter, deviceId: string, chunkerVersion: number) =>
        writeDeviceMeta(a as unknown as DataAdapter, DIR, { format: SIDECAR_FORMAT, modelId: EXPECT.modelId, revision: EXPECT.revision, chunkerVersion, dim: EXPECT.dim, deviceId, lastFullReindex: null });

    it('returns false when the sidecar is empty', async () => {
        const a = new FakeAdapter();
        expect(await probePeerAhead(a as unknown as DataAdapter, DIR, EXPECT)).toBe(false);
    });

    it('returns false when the only producer is compatible (same version)', async () => {
        const a = new FakeAdapter();
        await seedSidecar(a, 'desktop-aaa', [{ path: 'a.md', mtime: 1, ids: ['a'] }]); // chunkerVersion = EXPECT (3)
        expect(await probePeerAhead(a as unknown as DataAdapter, DIR, EXPECT)).toBe(false);
    });

    it('returns false when the only producer is OLDER (we are ahead, not behind)', async () => {
        const a = new FakeAdapter();
        await seedSidecar(a, 'desktop-aaa', [{ path: 'a.md', mtime: 1, ids: ['a'] }]);
        await overMeta(a, 'desktop-aaa', EXPECT.chunkerVersion - 1); // v2 < v3
        expect(await probePeerAhead(a as unknown as DataAdapter, DIR, EXPECT)).toBe(false);
    });

    it('returns true when a refused producer is NEWER than this build (durable peer-ahead signal)', async () => {
        const a = new FakeAdapter();
        await seedSidecar(a, 'desktop-aaa', [{ path: 'a.md', mtime: 1, ids: ['a'] }]);
        await overMeta(a, 'desktop-aaa', EXPECT.chunkerVersion + 1); // v4 > v3 — too new for this build to read
        expect(await probePeerAhead(a as unknown as DataAdapter, DIR, EXPECT)).toBe(true);
    });

    it('reports a single ahead peer even when another producer is accepted', async () => {
        const a = new FakeAdapter();
        await seedSidecar(a, 'desktop-compat', [{ path: 'a.md', mtime: 1, ids: ['a'] }]); // accepted (v3)
        await seedSidecar(a, 'desktop-ahead', [{ path: 'b.md', mtime: 1, ids: ['b'] }]);
        await overMeta(a, 'desktop-ahead', EXPECT.chunkerVersion + 1); // v4 — peer ahead
        expect(await probePeerAhead(a as unknown as DataAdapter, DIR, EXPECT)).toBe(true);
    });

    it('agrees with hydrateFromSidecar.peerAhead on the same refused-newer producer (no drift)', async () => {
        // Lock the cheap probe to the authoritative inline predicate: for an identical
        // sidecar state, the meta-only probe must equal the full hydrate path's peerAhead,
        // so the relaunch-recovered signal can never diverge from the scanned one.
        const a = new FakeAdapter();
        const notes: NoteSpec[] = [{ path: 'a.md', mtime: 100, ids: ['a1'] }];
        await seedSidecar(a, 'desktop-aaa', notes);
        await overMeta(a, 'desktop-aaa', EXPECT.chunkerVersion + 1);
        const { deps } = makeDeps(a, notes, {});
        const hydrate = await hydrateFromSidecar(deps);
        const probe = await probePeerAhead(a as unknown as DataAdapter, DIR, EXPECT);
        expect(probe).toBe(hydrate.peerAhead);
        expect(probe).toBe(true);
    });
});
