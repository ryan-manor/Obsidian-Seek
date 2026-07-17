// S1 — single atomic per-file commit. putBatchQuantized now takes the file
// record into the SAME transaction (5 stores), replacing the old two-transaction
// commit (putBatchQuantized THEN putFileRecord) whose gap could strand
// record-less chunks on a mid-commit kill (the orphan class sweepOrphanChunks
// repairs). These tests pin the transaction SCOPE (the atomicity mechanism —
// asserted via a spy on db.transaction against real fake-indexeddb) and the two
// commit shapes: chunks+record together, and the record-only commit a
// fully-quarantined file makes. The Tier-2 scenario drives record-only through
// the REAL engine: a deterministically-failing embed still pins its
// failure-marker record so the file doesn't thrash dirty forever (issue #4).
import 'fake-indexeddb/auto';
import { describe, it, expect, vi, afterEach } from 'vitest';
import { IndexStore } from './index-store';
import { Scenario } from './test-harness/scenario';
import type { Chunk } from './types';
import type { QuantVec } from './quant';

function chunk(id: string, content: string, notePath = `${id}.md`): Chunk {
    return {
        chunk_id: id, title: `T-${id}`, content, note_path: notePath,
        heading_path: [], metadata: { tags: [], aliases: [], created: null, modified: null, properties: {} },
        start_line: 0, end_line: 0,
    };
}
function tier(seed: number): { q: QuantVec; bin: Uint8Array } {
    return { q: { q: new Int8Array([seed, seed + 1]), s: 0.5 }, bin: new Uint8Array([seed & 0xff]) };
}

describe('putBatchQuantized + fileRecord — one atomic transaction (Tier-1)', () => {
    const opened: IndexStore[] = [];
    const boot = async (): Promise<{ store: IndexStore; rwCalls: () => unknown[][] }> => {
        const store = new IndexStore();
        // Unique scope per test — fake-indexeddb is ONE origin-scoped global,
        // exactly like the browser (same pattern as the scenario harness).
        await store.open(`atomic-${Math.random().toString(36).slice(2)}`, 'seek-test');
        opened.push(store);
        const db = (store as unknown as { db: IDBDatabase }).db;
        const spy = vi.spyOn(db, 'transaction');
        return { store, rwCalls: () => spy.mock.calls.filter(c => c[1] === 'readwrite') };
    };
    afterEach(() => { for (const s of opened.splice(0)) s.close(); });

    it('writes chunks AND the file record in ONE readwrite tx spanning all five stores', async () => {
        const { store, rwCalls } = await boot();
        await store.putBatchQuantized(
            [chunk('a1', 'alpha'), chunk('a2', 'beta', 'a1.md')],
            [tier(1), tier(2)],
            { note_path: 'a1.md', mtimeMs: 111, chunk_ids: ['a1', 'a2'], contentHash: 'h1' },
        );
        // The mechanism: exactly one readwrite tx, scoped to all five stores — the
        // record can't trail the chunks, so a kill lands the whole file or nothing.
        expect(rwCalls()).toHaveLength(1);
        expect([...rwCalls()[0][0] as string[]].sort())
            .toEqual(['binary', 'chunk_body', 'chunk_meta', 'embeddings', 'files']);
        // And it actually landed, coherently.
        const rec = await store.getFileRecord('a1.md');
        expect(rec?.chunk_ids).toEqual(['a1', 'a2']);
        expect(rec?.contentHash).toBe('h1');
        expect((await store.listAllMeta()).map(c => c.chunk_id).sort()).toEqual(['a1', 'a2']);
    });

    it('a record with ZERO chunks (fully-quarantined file) writes just the FILES store', async () => {
        const { store, rwCalls } = await boot();
        await store.putBatchQuantized([], [], {
            note_path: 'q.md', mtimeMs: 5, chunk_ids: [], contentHash: 'hq',
            embedFailedChunks: 2, embedFailPluginVersion: 'test',
        });
        expect(rwCalls()).toHaveLength(1);
        expect(rwCalls()[0][0]).toEqual(['files']);          // no empty 4-store tx
        const rec = await store.getFileRecord('q.md');
        expect(rec?.embedFailedChunks).toBe(2);
        expect((await store.listAllMeta())).toEqual([]);
    });

    it('no chunks + no record stays a silent no-op (the pre-S1 contract)', async () => {
        const { store, rwCalls } = await boot();
        await store.putBatchQuantized([], []);
        expect(rwCalls()).toHaveLength(0);                   // not even an empty tx
    });

    it('the length-mismatch guard still throws before anything is written', async () => {
        const { store, rwCalls } = await boot();
        await expect(store.putBatchQuantized([chunk('x', 'body')], [],
            { note_path: 'x.md', mtimeMs: 1, chunk_ids: ['x'] })).rejects.toThrow(/mismatch/);
        expect(rwCalls()).toHaveLength(0);
        expect(await store.getFileRecord('x.md')).toBeUndefined();
    });
});

describe('record-only commit through the real engine (Tier-2)', () => {
    let active: Scenario | null = null;
    afterEach(async () => { await active?.teardown(); active = null; });

    // A file whose EVERY chunk deterministically fails to embed (survives batch +
    // solo retries) must still commit its failure-marker record — now via the
    // record-only branch of the atomic commit — so classifyFileDelta reads it
    // clean instead of re-reporting it dirty (and re-paying two device recycles)
    // on every reconcile forever.
    it('a fully-failed file pins its quarantine record and stops thrashing dirty', async () => {
        const s = new Scenario();
        await s.boot();
        active = s;
        s.vault.write('bad.md', 'POISON text that will never embed', 1000);
        s.vault.write('good.md', 'healthy note about sourdough starters', 1000);
        const realEmbed = s.embedder.embedBatch.bind(s.embedder);
        (s.embedder as unknown as { embedBatch: typeof realEmbed }).embedBatch = async (texts: string[], ...rest: unknown[]) => {
            if (texts.some(t => t.includes('POISON'))) throw new Error('deterministic embed failure');
            return (realEmbed as (t: string[], ...r: unknown[]) => ReturnType<typeof realEmbed>)(texts, ...rest);
        };

        const entry = await s.orch.reindexAll();
        expect(entry.filesQuarantined).toBe(1);
        expect(entry.committedFilePaths).toContain('bad.md');   // record written = real progress
        expect(entry.committedFilePaths).toContain('good.md');

        // The record-only commit landed with the marker, and zero chunks.
        const rec = await s.store.getFileRecord('bad.md');
        expect(rec?.embedFailedChunks).toBe(1);
        expect(rec?.chunk_ids).toEqual([]);
        // Quarantine holds: the next reconcile finds NOTHING dirty (no thrash).
        expect((await s.orch.computeDelta()).dirty).toEqual([]);
        // The healthy note is searchable; the failed one is invisible-not-wrong.
        expect((await s.store.count()).chunks).toBeGreaterThan(0);
    });
});
