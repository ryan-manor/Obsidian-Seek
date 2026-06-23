// getEmbeddingsByIds is the stage-2 vector fetch. It was rewritten from a
// per-id .get() loop to a single sorted cursor that JUMPS to each wanted key
// via continue(key) — the cheap shape on mobile WKWebView IDB, which is now the
// always-path there (the B2 resident-block gate disables the RAM tier on
// mobile). These tests pin that the rewrite preserves the old contract exactly:
// output aligned to INPUT order, missing ids → null, [] on empty, duplicates
// resolved. The riskiest part is the continue(key) control flow, so we drive
// the REAL method against a faithful in-memory cursor (no fake-indexeddb dep)
// and assert byte-equality against a reference per-get implementation.

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { IndexStore, classifyFileDelta, planRestoreOps, findOrphanChunkIds, isStoreClosedError, STORE_NOT_OPENED, indexDbPrefix, type StoreSnapshot } from './index-store';
import { quantizeInt8, dequantizeInt8, type QuantVec } from './quant';

// The closed-store discriminator behind the reindex storm bound: the indexer rethrows
// (aborting the whole pass) ONLY for this error, and skips just the one file otherwise.
describe('indexDbPrefix (per-plugin DB scoping for co-installed builds)', () => {
    it('the shipped id resolves to the legacy name — released build never migrates', () => {
        // open() appends `:<appId>`, so this must equal the historical
        // LEGACY_DB_NAME ('seek-index') or every public install would re-key + reindex.
        expect(indexDbPrefix('seek')).toBe('seek-index');
    });
    it('a differently-id\'d build gets a SEPARATE database prefix', () => {
        expect(indexDbPrefix('seek-prototype')).toBe('seek-prototype-index');
        // distinct from the shipped prefix → no shared DB, no cross-nuke
        expect(indexDbPrefix('seek-prototype')).not.toBe(indexDbPrefix('seek'));
    });
});

describe('isStoreClosedError (reindex-storm bound)', () => {
    it('is true for the requireDb closed-store error', () => {
        expect(isStoreClosedError(new Error(STORE_NOT_OPENED))).toBe(true);
    });
    it('is false for any other commit error (those skip only their file)', () => {
        expect(isStoreClosedError(new Error('QuotaExceededError'))).toBe(false);
        expect(isStoreClosedError(new Error('chunks.length !== vectors.length'))).toBe(false);
    });
    it('is false for a non-Error throwable', () => {
        expect(isStoreClosedError(STORE_NOT_OPENED)).toBe(false); // a bare string, not an Error
        expect(isStoreClosedError(null)).toBe(false);
        expect(isStoreClosedError(undefined)).toBe(false);
    });
});

describe('findOrphanChunkIds (referential-integrity sweep)', () => {
    it('returns chunk_ids referenced by no file record', () => {
        expect(findOrphanChunkIds(['a', 'b', 'c', 'd'], new Set(['a', 'c']))).toEqual(['b', 'd']);
    });
    it('returns [] when every chunk is referenced', () => {
        expect(findOrphanChunkIds(['a', 'b'], new Set(['a', 'b', 'x']))).toEqual([]);
    });
    it('returns all chunks when nothing is referenced (e.g. all file records gone)', () => {
        expect(findOrphanChunkIds(['a', 'b'], new Set<string>())).toEqual(['a', 'b']);
    });
    it('preserves input order for deterministic batching', () => {
        expect(findOrphanChunkIds(['z', 'a', 'm'], new Set(['a']))).toEqual(['z', 'm']);
    });
});

// ── Minimal IDB cursor, faithful to the bits getEmbeddingsByIds touches ──────
// openCursor(IDBKeyRange.lowerBound(k)) positions at the first key >= k and
// fires onsuccess; cursor.continue(key) re-fires onsuccess at the first key >=
// key (or null at the end). Keys walked ascending. queueMicrotask defers each
// fire so the real code can assign onsuccess before the first callback runs —
// exactly the IDBRequest ordering guarantee.
type OnSuccess = (() => void) | null;
interface MockCursor { key: string; value: QuantVec | undefined; continue(target?: string): void }
interface MockRequest { onsuccess: OnSuccess; onerror: (() => void) | null; result: MockCursor | null }

function makeEmbeddingsDb(records: Map<string, QuantVec>): IDBDatabase {
    const sortedKeys = Array.from(records.keys()).sort();
    function openCursor(range?: { lower?: string }): MockRequest {
        const req: MockRequest = { onsuccess: null, onerror: null, result: null };
        let pos = range?.lower != null
            ? sortedKeys.findIndex(k => k >= range.lower!)
            : 0;
        if (pos < 0) pos = sortedKeys.length;
        const fire = (): void => {
            if (pos >= sortedKeys.length) { req.result = null; req.onsuccess?.(); return; }
            const key = sortedKeys[pos];
            req.result = {
                key,
                value: records.get(key),
                continue(target?: string): void {
                    if (target == null) pos++;
                    else { pos = sortedKeys.findIndex(k => k >= target); if (pos < 0) pos = sortedKeys.length; }
                    queueMicrotask(fire);
                },
            };
            req.onsuccess?.();
        };
        queueMicrotask(fire);
        return req;
    }
    const store = { openCursor };
    const tx = { objectStore: () => store };
    return { transaction: () => tx } as unknown as IDBDatabase;
}

function attach(records: Map<string, QuantVec>): IndexStore {
    const s = new IndexStore();
    (s as unknown as { db: IDBDatabase }).db = makeEmbeddingsDb(records);
    return s;
}

// Deterministic pseudo-random unit-ish vector (no Math.random — keep stable).
function vec(seed: number, d = 8): Float32Array {
    const out = new Float32Array(d);
    let x = (seed * 2654435761) >>> 0;
    for (let i = 0; i < d; i++) { x = (x * 1664525 + 1013904223) >>> 0; out[i] = (x / 0xffffffff) * 2 - 1; }
    return out;
}

function buildStore(ids: string[]): Map<string, QuantVec> {
    const m = new Map<string, QuantVec>();
    ids.forEach((id, i) => m.set(id, quantizeInt8(vec(i + 1))));
    return m;
}

// The old per-get semantics: dequant for present ids, null for absent, in input order.
function reference(records: Map<string, QuantVec>, ids: string[]): Array<Float32Array | null> {
    return ids.map(id => {
        const qv = records.get(id);
        return qv ? dequantizeInt8(qv.q, qv.s) : null;
    });
}

function expectEqual(a: Array<Float32Array | null>, b: Array<Float32Array | null>): void {
    expect(a.length).toBe(b.length);
    for (let i = 0; i < a.length; i++) {
        const ai = a[i], bi = b[i];
        if (ai === null || bi === null) { expect(ai).toBe(bi); continue; }
        expect(Array.from(ai)).toEqual(Array.from(bi));
    }
}

beforeEach(() => {
    vi.stubGlobal('IDBKeyRange', { lowerBound: (lower: string) => ({ lower }) });
});

describe('getEmbeddingsByIds — cursor-jump parity with per-get semantics', () => {
    // content-hash-like hex keys, deliberately NOT in insertion order
    const storeIds = ['00ff', '1a2b', '3c4d', '7e8f', 'a1b2', 'c3d4', 'deca', 'ffff'];

    it('empty input → []', async () => {
        expect(await attach(buildStore(storeIds)).getEmbeddingsByIds([])).toEqual([]);
    });

    it('in-order subset', async () => {
        const recs = buildStore(storeIds);
        const ids = ['1a2b', '3c4d', 'a1b2'];
        expectEqual(await attach(recs).getEmbeddingsByIds(ids), reference(recs, ids));
    });

    it('shuffled input order is preserved on output', async () => {
        const recs = buildStore(storeIds);
        const ids = ['ffff', '00ff', 'c3d4', '1a2b'];
        expectEqual(await attach(recs).getEmbeddingsByIds(ids), reference(recs, ids));
    });

    it('gaps + absent ids (below, between, above the keyspace) → null in the right slots', async () => {
        const recs = buildStore(storeIds);
        const ids = ['00ff', 'zzzz', '7e8f', '0000', 'ffff', '5555']; // zzzz>all, 0000<all, 5555 between
        const got = await attach(recs).getEmbeddingsByIds(ids);
        expectEqual(got, reference(recs, ids));
        expect(got[1]).toBeNull();
        expect(got[3]).toBeNull();
        expect(got[5]).toBeNull();
    });

    it('duplicate ids resolve identically', async () => {
        const recs = buildStore(storeIds);
        const ids = ['a1b2', 'a1b2', '00ff', 'a1b2'];
        const got = await attach(recs).getEmbeddingsByIds(ids);
        expectEqual(got, reference(recs, ids));
        expect(Array.from(got[0]!)).toEqual(Array.from(got[1]!));
    });

    it('all ids absent → all null', async () => {
        const got = await attach(buildStore(storeIds)).getEmbeddingsByIds(['xxxx', 'yyyy']);
        expect(got).toEqual([null, null]);
    });

    it('full set, store order and reverse order', async () => {
        const recs = buildStore(storeIds);
        expectEqual(await attach(recs).getEmbeddingsByIds(storeIds), reference(recs, storeIds));
        const rev = [...storeIds].reverse();
        expectEqual(await attach(recs).getEmbeddingsByIds(rev), reference(recs, rev));
    });

    it('single present and single absent id', async () => {
        const recs = buildStore(storeIds);
        expectEqual(await attach(recs).getEmbeddingsByIds(['deca']), reference(recs, ['deca']));
        expect(await attach(recs).getEmbeddingsByIds(['nope'])).toEqual([null]);
    });
});

// classifyFileDelta is the iCloud-restamp guard: it must re-embed real edits but
// NOT a file whose mtime moved while its bytes stayed identical (the iOS sync
// churn that pinned Seek's mobile indexer to 1 fps). The two-phase 'check-bytes'
// contract keeps the content read off every clean file.
describe('classifyFileDelta', () => {
    const rec = (mtimeMs: number, contentHash?: string) => ({ mtimeMs, contentHash });

    it('never-indexed file is dirty', () => {
        expect(classifyFileDelta(undefined, 1000)).toBe('dirty');
    });

    it('unchanged mtime is clean without reading bytes', () => {
        expect(classifyFileDelta(rec(1000, 'aaa'), 1000)).toBe('clean'); // equal
        expect(classifyFileDelta(rec(1000, 'aaa'), 999)).toBe('clean');  // older (clock skew)
    });

    it('mtime advanced on a hash-bearing record asks for the bytes', () => {
        expect(classifyFileDelta(rec(1000, 'aaa'), 2000)).toBe('check-bytes');
    });

    it('mtime-only re-stamp (identical bytes) is clean — the churn fix', () => {
        expect(classifyFileDelta(rec(1000, 'aaa'), 2000, 'aaa')).toBe('clean');
    });

    it('mtime advanced AND bytes changed is dirty', () => {
        expect(classifyFileDelta(rec(1000, 'aaa'), 2000, 'bbb')).toBe('dirty');
    });

    it('legacy record without a stored hash falls back to mtime-only (dirty, backfills)', () => {
        expect(classifyFileDelta(rec(1000, undefined), 2000)).toBe('dirty');
        expect(classifyFileDelta(rec(1000, undefined), 1000)).toBe('clean'); // but only when mtime moved
    });
});

// planRestoreOps is the one subtle bit of embed-free compaction: in-line-keyed
// stores (chunk_meta→chunk_id, files→note_path) must re-put WITHOUT an explicit
// key (passing one throws DataError), while out-of-line stores must pass it back.
// The full snapshot→delete→rewrite round-trip is validated by live smoke (real
// IDB), as the rest of this store's IDB plumbing is.
describe('planRestoreOps', () => {
    it('omits the key for in-line-keyed stores and includes it for out-of-line', () => {
        const snapshot: StoreSnapshot[] = [
            { store: 'chunk_meta', inlineKey: true, records: [{ key: 'c1', value: { chunk_id: 'c1', t: 'x' } }] },
            { store: 'embeddings', inlineKey: false, records: [{ key: 'c1', value: { q: new Int8Array(2), s: 1 } }] },
        ];
        const ops = planRestoreOps(snapshot);
        expect(ops).toHaveLength(2);

        const meta = ops[0];
        expect(meta.store).toBe('chunk_meta');
        expect('key' in meta).toBe(false);          // in-line → no explicit key
        expect(meta.value).toEqual({ chunk_id: 'c1', t: 'x' });

        const emb = ops[1];
        expect(emb.store).toBe('embeddings');
        expect(emb.key).toBe('c1');                  // out-of-line → key carried back
    });

    it('preserves every record across all stores, in order', () => {
        const snapshot: StoreSnapshot[] = [
            { store: 'chunk_body', inlineKey: false, records: [
                { key: 'a', value: 'body-a' },
                { key: 'b', value: 'body-b' },
            ] },
            { store: 'bm25', inlineKey: false, records: [{ key: 'index', value: { json: 'x', stamp: 1 } }] },
        ];
        const ops = planRestoreOps(snapshot);
        expect(ops.map(o => o.store)).toEqual(['chunk_body', 'chunk_body', 'bm25']);
        expect(ops.map(o => o.key)).toEqual(['a', 'b', 'index']);
    });

    it('handles an empty snapshot', () => {
        expect(planRestoreOps([])).toEqual([]);
    });
});
