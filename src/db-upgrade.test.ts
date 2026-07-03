// DB_VERSION upgrade-path contract (audit R2 #7).
//
// The DB_VERSION history in index-store.ts documents a forced clean rebuild for
// v9->v10 (suffix cleanliness gates shift chunk ids) and v10->v11 (line spans
// move to raw-file coordinates) — but no code implemented either until
// 2026-07-02: a v9/v10 origin carried its stale stores straight through
// onupgradeneeded, while identity.ts deliberately excludes dbVersion from the
// re-embed gate on the strength of that (falsified) promise. These tests run
// the REAL openDb upgrade path on fake-indexeddb from seeded old-version
// databases and pin the drop-and-recreate behavior — so the next DB_VERSION
// bump without a matching branch goes red here instead of shipping a silent
// carry-through.

import 'fake-indexeddb/auto';
import { describe, it, expect } from 'vitest';
import { openDb, DB_VERSION } from './index-store';

// Store names duplicated as literals on purpose: the test pins the on-disk
// contract, not whatever the constants currently say.
const DATA_STORES = ['chunk_meta', 'chunk_body', 'embeddings', 'binary', 'files'];

function reqDone<T>(req: IDBRequest<T>): Promise<T> {
    return new Promise((res, rej) => {
        req.onsuccess = () => res(req.result);
        req.onerror = () => rej(req.error);
    });
}

// Build a database at `version` with the (v8+) store layout and one record in
// each data store — a stand-in for a device that last ran an older Seek.
async function seedOldDb(name: string, version: number): Promise<void> {
    const db = await new Promise<IDBDatabase>((res, rej) => {
        const req = indexedDB.open(name, version);
        req.onupgradeneeded = () => {
            const d = req.result;
            d.createObjectStore('chunk_meta', { keyPath: 'chunk_id' });
            d.createObjectStore('chunk_body');
            d.createObjectStore('embeddings');
            d.createObjectStore('binary');
            d.createObjectStore('files', { keyPath: 'note_path' });
            d.createObjectStore('meta');
            d.createObjectStore('bm25');
        };
        req.onsuccess = () => res(req.result);
        req.onerror = () => rej(req.error);
    });
    const tx = db.transaction(DATA_STORES, 'readwrite');
    tx.objectStore('chunk_meta').put({ chunk_id: 'old-id', title: 'stale' });
    tx.objectStore('chunk_body').put('stale body', 'old-id');
    tx.objectStore('embeddings').put(new Uint8Array([1, 2, 3]), 'old-id');
    tx.objectStore('binary').put(new Uint8Array([4, 5, 6]), 'old-id');
    tx.objectStore('files').put({ note_path: 'Old.md', mtimeMs: 1 });
    await new Promise<void>((res, rej) => {
        tx.oncomplete = () => res();
        tx.onerror = () => rej(tx.error);
    });
    db.close();
}

async function counts(db: IDBDatabase): Promise<Record<string, number>> {
    const out: Record<string, number> = {};
    for (const s of DATA_STORES) {
        out[s] = await reqDone(db.transaction(s, 'readonly').objectStore(s).count());
    }
    return out;
}

describe('openDb upgrade path — documented destructive rebuilds actually happen', () => {
    for (const origin of [9, 10]) {
        it(`a v${origin} origin reaches v${DB_VERSION} with EMPTY data stores (forced rebuild)`, async () => {
            const name = `seek-test-upgrade-v${origin}`;
            await seedOldDb(name, origin);
            const db = await openDb(name);
            expect(db.version).toBe(DB_VERSION);
            expect(await counts(db)).toEqual(
                Object.fromEntries(DATA_STORES.map(s => [s, 0])),
            );
            db.close();
        });
    }

    it(`a current v${DB_VERSION} database reopens with its data INTACT (no spurious drop)`, async () => {
        const name = 'seek-test-upgrade-current';
        await seedOldDb(name, DB_VERSION);
        const db = await openDb(name);
        expect(db.version).toBe(DB_VERSION);
        const c = await counts(db);
        expect(c.chunk_meta).toBe(1);
        expect(c.files).toBe(1);
        db.close();
    });
});
