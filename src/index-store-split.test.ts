// v8 store split (docs/seek-scaling.md §B1): the `chunks` store became
// `chunk_meta` (everything except the body) + `chunk_body` (chunk_id → content).
// These tests pin the read/write contract of that split against a compact
// in-memory IDB: putBatch writes BOTH stores atomically with body text kept OUT
// of chunk_meta, listAllMeta returns metadata only, bodies come back via
// getBodiesByIds/getBodiesMap, and deleteFile clears every tier including both
// new stores. (The repo has no IDB harness and the sidecar tests use in-memory
// stand-ins; this faithful-enough fake exercises the real IndexStore methods.)

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { IndexStore } from './index-store';
import { dequantizeInt8, type QuantVec } from './quant';
import { embedInput } from './token-budget';
import type { Chunk } from './types';

// ── Minimal in-memory IDB ────────────────────────────────────────────────────
// Maps mutate synchronously; request onsuccess / transaction oncomplete fire on
// microtasks (the only async contract awaitRequest/awaitTx depend on). openCursor
// implements the lowerBound + continue(key) seek collectByKeyJump uses.
function fakeReq<T>(result: T) {
    const r = { result, onsuccess: null as null | (() => void), onerror: null } as
        { result: T; onsuccess: null | (() => void); onerror: null };
    queueMicrotask(() => r.onsuccess?.());
    return r;
}

class FakeStore {
    map = new Map<string, unknown>();
    constructor(private keyPath: string | null) {}
    private keyOf(value: unknown, key?: string): string {
        return this.keyPath ? String((value as Record<string, unknown>)[this.keyPath]) : String(key);
    }
    put(value: unknown, key?: string) { this.map.set(this.keyOf(value, key), value); return fakeReq(undefined); }
    get(key: unknown) { return fakeReq(this.map.get(String(key))); }
    getAll() { return fakeReq(Array.from(this.map.values())); }
    delete(key: unknown) { this.map.delete(String(key)); return fakeReq(undefined); }
    count() { return fakeReq(this.map.size); }
    openCursor(range?: { lower?: string }) {
        const keys = Array.from(this.map.keys()).sort();
        const r = { result: null as unknown, onsuccess: null as null | (() => void), onerror: null } as
            { result: unknown; onsuccess: null | (() => void); onerror: null };
        let pos = range?.lower != null ? keys.findIndex(k => k >= range.lower!) : 0;
        if (pos < 0) pos = keys.length;
        const fire = (): void => {
            if (pos >= keys.length) { r.result = null; r.onsuccess?.(); return; }
            const key = keys[pos];
            r.result = {
                key,
                value: this.map.get(key),
                continue: (target?: string): void => {
                    if (target == null) pos++;
                    else { pos = keys.findIndex(k => k >= target); if (pos < 0) pos = keys.length; }
                    queueMicrotask(fire);
                },
            };
            r.onsuccess?.();
        };
        queueMicrotask(fire);
        return r;
    }
}

function makeDb() {
    const stores: Record<string, FakeStore> = {
        chunk_meta: new FakeStore('chunk_id'),
        chunk_body: new FakeStore(null),
        embeddings: new FakeStore(null),
        binary: new FakeStore(null),
        files: new FakeStore('note_path'),
        meta: new FakeStore(null),
        bm25: new FakeStore(null),
    };
    const db = {
        transaction(_names: string | string[], _mode?: string) {
            const tx = {
                objectStore: (n: string) => stores[n],
                oncomplete: null as null | (() => void),
                onerror: null,
                onabort: null,
            };
            queueMicrotask(() => tx.oncomplete?.());
            return tx;
        },
        _stores: stores,
    };
    return db;
}

function attach(): { store: IndexStore; db: ReturnType<typeof makeDb> } {
    const db = makeDb();
    const store = new IndexStore();
    (store as unknown as { db: unknown }).db = db;
    return { store, db };
}

function chunk(id: string, content: string, notePath = `${id}.md`): Chunk {
    return {
        chunk_id: id, title: `T-${id}`, content, note_path: notePath,
        heading_path: [], metadata: { tags: [], aliases: [], created: null, modified: null, properties: {} },
        start_line: 0, end_line: 0,
    };
}

function vec(seed: number, d = 4): Float32Array {
    const out = new Float32Array(d);
    let x = (seed * 2654435761) >>> 0;
    for (let i = 0; i < d; i++) { x = (x * 1664525 + 1013904223) >>> 0; out[i] = (x / 0xffffffff) * 2 - 1; }
    return out;
}

beforeEach(() => {
    vi.stubGlobal('IDBKeyRange', { lowerBound: (lower: string) => ({ lower }) });
});

describe('v8 store split — chunk_meta / chunk_body read+write contract', () => {
    it('putBatch keeps body OUT of chunk_meta and IN chunk_body', async () => {
        const { store, db } = attach();
        const chunks = [chunk('a', 'alpha body text'), chunk('b', 'bravo body text')];
        await store.putBatch(chunks, [vec(1), vec(2)]);

        // chunk_meta rows carry no content; chunk_body holds it keyed by id.
        const metaRows = Array.from(db._stores.chunk_meta.map.values()) as Record<string, unknown>[];
        expect(metaRows).toHaveLength(2);
        for (const m of metaRows) expect('content' in m).toBe(false);
        expect(db._stores.chunk_body.map.get('a')).toBe('alpha body text');
        expect(db._stores.chunk_body.map.get('b')).toBe('bravo body text');
        // vector tiers written too (4-store atomicity preserved across the split).
        expect(db._stores.embeddings.map.size).toBe(2);
        expect(db._stores.binary.map.size).toBe(2);
    });

    it('listAllMeta returns metadata without content; count reports chunk_meta', async () => {
        const { store } = attach();
        await store.putBatch([chunk('a', 'x'), chunk('b', 'y'), chunk('c', 'z')], [vec(1), vec(2), vec(3)]);
        const meta = await store.listAllMeta();
        expect(meta.map(m => m.chunk_id).sort()).toEqual(['a', 'b', 'c']);
        for (const m of meta) expect((m as Record<string, unknown>).content).toBeUndefined();
        expect((await store.count()).chunks).toBe(3);
    });

    it('getBodiesByIds is input-order aligned; missing → null', async () => {
        const { store } = attach();
        await store.putBatch([chunk('a', 'AA'), chunk('b', 'BB'), chunk('c', 'CC')], [vec(1), vec(2), vec(3)]);
        expect(await store.getBodiesByIds(['c', 'a', 'zzz', 'b'])).toEqual(['CC', 'AA', null, 'BB']);
        expect(await store.getBodiesByIds([])).toEqual([]);
    });

    it('getBodiesMap returns present ids only', async () => {
        const { store } = attach();
        await store.putBatch([chunk('a', 'AA'), chunk('b', 'BB')], [vec(1), vec(2)]);
        const m = await store.getBodiesMap(['a', 'zzz', 'b']);
        expect(m.get('a')).toBe('AA');
        expect(m.get('b')).toBe('BB');
        expect(m.has('zzz')).toBe(false);
    });

    it('getEmbeddingsByIds dequantizes from the split stores', async () => {
        const { store, db } = attach();
        await store.putBatch([chunk('a', 'AA')], [vec(7)]);
        const got = await store.getEmbeddingsByIds(['a']);
        const stored = db._stores.embeddings.map.get('a') as QuantVec;
        expect(Array.from(got[0]!)).toEqual(Array.from(dequantizeInt8(stored.q, stored.s)));
    });

    it('deleteFile clears chunk_meta + chunk_body + vectors + file record', async () => {
        const { store, db } = attach();
        await store.putBatch([chunk('a', 'AA', 'N.md'), chunk('b', 'BB', 'N.md'), chunk('c', 'CC', 'Other.md')],
            [vec(1), vec(2), vec(3)]);
        await store.putFileRecord({ note_path: 'N.md', mtimeMs: 1, chunk_ids: ['a', 'b'] });
        await store.putFileRecord({ note_path: 'Other.md', mtimeMs: 1, chunk_ids: ['c'] });

        const removed = await store.deleteFile('N.md');
        expect(removed.sort()).toEqual(['a', 'b']);
        for (const s of ['chunk_meta', 'chunk_body', 'embeddings', 'binary'] as const) {
            expect(db._stores[s].map.has('a')).toBe(false);
            expect(db._stores[s].map.has('b')).toBe(false);
            expect(db._stores[s].map.has('c')).toBe(true);  // untouched note survives
        }
        expect(db._stores.files.map.has('N.md')).toBe(false);
        expect((await store.listAllMeta()).map(m => m.chunk_id)).toEqual(['c']);
    });
});

// F13 carry-over read (search.ts harvestCarryOverInto): getTiersByIds must
// reassemble each chunk's full text from the v8 split (chunk_meta + chunk_body)
// and return its RAW int8 + sign tiers. The pre-v8 implementation read the single
// legacy `chunks` store — which the makeDb fake (correctly) does NOT define, so
// these tests pin the post-split contract AND would throw on a regression to it.
describe('getTiersByIds — F13 carry-over read (v8 split)', () => {
    it('reassembles {chunk incl. content, raw q, sign} from the split stores, input-order aligned', async () => {
        const { store, db } = attach();
        await store.putBatch(
            [chunk('a', 'alpha body'), chunk('b', 'bravo body'), chunk('c', 'charlie body')],
            [vec(1), vec(2), vec(3)],
        );

        const got = await store.getTiersByIds(['c', 'a', 'b']);
        expect(got.map(t => t?.chunk.chunk_id)).toEqual(['c', 'a', 'b']);   // input-order aligned
        // Full Chunk reassembled from chunk_meta + chunk_body == the original input
        // (so embedInput(chunk) reproduces the exact carry-over key).
        expect(got[1]!.chunk).toEqual(chunk('a', 'alpha body'));
        expect(got[1]!.chunk.content).toBe('alpha body');                   // body folded back in
        // RAW tiers (no dequant): byte-identical to what embeddings/binary hold.
        expect(got[1]!.q).toEqual(db._stores.embeddings.map.get('a'));
        expect(got[1]!.sign).toEqual(db._stores.binary.map.get('a'));
    });

    it('returns null for a missing id and [] for no ids (falls through to a normal embed)', async () => {
        const { store } = attach();
        await store.putBatch([chunk('a', 'AA')], [vec(1)]);
        const got = await store.getTiersByIds(['a', 'zzz']);
        expect(got[0]!.chunk.chunk_id).toBe('a');
        expect(got[1]).toBeNull();
        expect(await store.getTiersByIds([])).toEqual([]);
    });

    it('treats an empty-content chunk as a HIT — body "" is valid, not a miss', async () => {
        // putBatch stores content ?? '' in chunk_body; embedInput re-creates the
        // bare `title\n\n` for it, so the carry-over key still matches. A === undefined
        // miss check (not a falsy one) is what keeps this a hit.
        const { store } = attach();
        await store.putBatch([chunk('e', '')], [vec(5)]);
        const [t] = await store.getTiersByIds(['e']);
        expect(t).not.toBeNull();
        expect(t!.chunk.content).toBe('');
    });

    it('round-trips denseSuffix + displayTitle (the load-bearing carry-over key) through meta+body', async () => {
        // denseSuffix is what embedInput appends after the body and what chunk_id folds
        // in (DB_VERSION 9/10 exist for it). It lives in chunk_meta via stripContent's
        // rest-spread; if a refactor dropped it from meta, the harvest key would silently
        // stop matching the re-chunked key. Pin the full reassembly AND the embedInput key.
        const { store } = attach();
        const c: Chunk = { ...chunk('d', 'body text'), denseSuffix: 'project: Seek\nstatus: shipped', displayTitle: 'D (part 1)' };
        await store.putBatch([c], [vec(9)]);
        const [t] = await store.getTiersByIds(['d']);
        expect(t!.chunk).toEqual(c);                      // denseSuffix + displayTitle survive the split
        expect(embedInput(t!.chunk)).toBe(embedInput(c)); // ⇒ the carry-over key is byte-identical
    });

    it('returns null on a torn record — chunk_meta present but the body row is gone', async () => {
        // Carry-over is all-or-nothing per file (carryOverHydrate), so a half-written
        // chunk must read as a miss and re-embed rather than reuse a bodyless chunk.
        const { store, db } = attach();
        await store.putBatch([chunk('a', 'AA')], [vec(1)]);
        db._stores.chunk_body.map.delete('a');
        expect(await store.getTiersByIds(['a'])).toEqual([null]);
    });
});
