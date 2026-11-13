// Tier-2 composed scenarios — the real orchestrator + real store (fake-indexeddb)
// + fake vault/embedder, driven by a scripted event stream. These pin emergent
// ORDERING bugs that single-decision unit tests miss by construction. See
// scenario.ts and [[Seek Testing Strategy]].
import { describe, it, expect, vi, afterEach } from 'vitest';
import { Scenario } from './scenario';

describe('Tier-2 scenario harness', () => {
    let active: Scenario | null = null;
    const boot = async (): Promise<Scenario> => {
        const s = new Scenario();
        await s.boot();
        active = s;
        return s;
    };
    afterEach(async () => { await active?.teardown(); active = null; });

    // ── Scenario 1 — the cold-build re-heal loop ────────────────────────────
    // Regression for the worst bug to date: the cold build ran the incremental
    // path, which preserves prevMeta — undefined on an empty store — so identity
    // was never stamped and the gate re-healed forever. "Fix A" (search.ts: cold
    // build with sawWholeCorpus stamps the live identity) is what this pins. A
    // unit test of the stamp decision can't see the convergence; only a composed
    // scenario can.
    it('cold build on an empty store STAMPS live identity and converges — no re-heal loop', async () => {
        const s = await boot();
        s.vault.write('a.md', 'the verge reviews the new pixel phone', 1000);
        s.vault.write('b.md', 'imogene heap concert notes and setlist', 1000);

        await s.coldStart();                                    // the path that used to run incremental + never stamp

        // (a) it indexed
        expect((await s.store.count()).chunks).toBeGreaterThan(0);

        // (b) identity STAMPED on the empty-store build — modelId AND the three
        //     identity fields the re-heal gate reads. The buggy path leaves these
        //     undefined (prevMeta on an empty store), looping the heal forever.
        const meta = await s.store.getMeta();
        expect(meta.modelId).toBe('test-model');
        expect(meta.lastIndexedAt).not.toBeNull();
        expect(meta.chunkerVersion).toBeTypeOf('number');
        expect(meta.analyzerVersion).toBeTypeOf('string');
        expect('revision' in meta).toBe(true);

        // (c) a second reconcile finds nothing dirty and does NOT re-embed
        //     (no spurious full rebuild). The buggy code re-embeds every time.
        const fingerprint = meta.lastIndexedAt;
        const spy = vi.spyOn(s.embedder, 'embedBatch');
        await s.reconcile();
        expect(spy).not.toHaveBeenCalled();
        expect((await s.store.getMeta()).lastIndexedAt).toBe(fingerprint);
    });

    // ── Scenario 2 — the iCloud mtime re-stamp ──────────────────────────────
    // The content-hash gate, composed through the REAL store: mtime moves, bytes
    // do not, and no re-embed fires. computeDelta must classify check-bytes →
    // unchanged → dirty:[].
    it('an iCloud mtime re-stamp with unchanged bytes does NOT re-embed', async () => {
        const s = await boot();
        s.vault.write('a.md', 'stable content that does not change', 1000);
        await s.coldStart();                                   // build + stamp identity (the real first-index path)

        const spy = vi.spyOn(s.embedder, 'embedBatch');
        await s.touch('a.md', 9999);                            // mtime jumps, bytes identical

        expect(spy).not.toHaveBeenCalled();                    // contentHash gate: computeDelta → dirty:[]
    });

    // The live-enqueue half of the same gate (main.ts enqueueIfDirty consults
    // classifyLiveFile on every vault 'modify'). It used to be mtime-only, so a
    // mass iCloud re-stamp queued every unchanged note into one bulk pass that
    // held the index lock ~20 s on iPhone (2026-11-10 report).
    it('classifyLiveFile: re-stamp of unchanged bytes is clean, a real edit is dirty', async () => {
        const s = await boot();
        s.vault.write('a.md', 'stable content that does not change', 1000);
        await s.coldStart();
        const live = () => s.vault.getAbstractFileByPath('a.md')!;
        const rec = async () => s.store.getFileRecord('a.md');

        expect(await s.orch.classifyLiveFile(live(), await rec())).toBe('clean');   // untouched
        s.vault.touch('a.md', 9999);
        expect(await s.orch.classifyLiveFile(live(), await rec())).toBe('clean');   // mtime-only
        s.vault.write('a.md', 'stable content that DID change', 10000);
        expect(await s.orch.classifyLiveFile(live(), await rec())).toBe('dirty');   // real edit
        s.vault.write('new.md', 'never indexed', 10000);
        expect(await s.orch.classifyLiveFile(s.vault.getAbstractFileByPath('new.md')!, undefined)).toBe('dirty');
    });

    // ── Scenario 3 — end-to-end search returns the right note rank-1 ─────────
    // Proves the harness asserts RESULTS, not just index state: the full
    // chunk→embed→store→binary-scan→int8-rerank→BM25-fuse→rank pipeline runs on
    // the real orchestrator. The fake embedder is content-derived, so a query
    // sharing tokens with one note out-scores the other on the dense channel.
    it('a query returns the topically-matching note at rank 1 (full pipeline)', async () => {
        const s = await boot();
        s.vault.write('pixel.md', 'the verge reviews the new google pixel phone camera', 1000);
        s.vault.write('music.md', 'imogene heap concert setlist and tour dates', 1000);
        await s.coldStart();

        const { results } = await s.orch.search('google pixel phone camera review', 5);
        expect(results.length).toBeGreaterThan(0);
        expect(results[0].note_path).toBe('pixel.md');
    });

    // ── Scenario 4 — a read failure mid-pass skips one file, not the batch ───
    // The carryover NotFoundError family: a file vanishes BETWEEN the directory
    // listing and its read. embedAndCommitFiles must skip just that file
    // (filesSkippedError++ / continue) and still index the rest — only a
    // closed-store error may abort the whole pass.
    it('an unreadable file mid-pass is skipped; the rest of the corpus still indexes', async () => {
        const s = await boot();
        s.vault.write('good1.md', 'first healthy note about gardening', 1000);
        s.vault.write('gone.md',  'this file disappears before it is read', 1000);
        s.vault.write('good2.md', 'second healthy note about cooking', 1000);
        s.vault.failReads.add('gone.md');                      // read throws NotFoundError

        await s.coldStart();                                   // must NOT abort on the one bad file

        const paths = new Set((await s.store.listFileRecords()).map(r => r.note_path));
        expect(paths.has('good1.md')).toBe(true);
        expect(paths.has('good2.md')).toBe(true);
        expect(paths.has('gone.md')).toBe(false);              // skipped, not indexed
    });

    // ── Scenario 4b — a persistently unreadable file does not wedge computeDelta ─
    // Scenario 4 pins the single-pass skip. Without quarantining, that skipped
    // file's record never gets written, so classifyFileDelta(undefined, mtime)
    // reports it 'dirty' again on EVERY later computeDelta call — forever. That
    // wedges reconcileIdentityInPlace in 'drained' permanently (search.ts),
    // which short-circuits periodicReconcile past sidecar reconcile / orphan
    // sweep / compaction for the rest of the session over one bad file.
    // quarantineUnreadable() backs it off after the first attempt instead.
    it('a persistently unreadable file is quarantined out of dirty after one attempt', async () => {
        const s = await boot();
        s.vault.write('good.md', 'a healthy note about hiking', 1000);
        await s.coldStart();

        s.vault.write('bad.md', 'an undownloaded iCloud placeholder', 2000);
        s.vault.failReads.add('bad.md');                       // permanently unreadable

        // First pass: attempted (and skipped) — the quarantine is set here.
        let delta = await s.orch.computeDelta();
        expect(delta.dirty).toContain('bad.md');
        await s.orch.reindexDelta(delta.dirty, delta.deleted, { embed: true });

        // Second pass: WITHOUT the fix, bad.md's dropped record makes it dirty
        // again — forever. With the fix it is quarantined and excluded.
        delta = await s.orch.computeDelta();
        expect(delta.dirty).not.toContain('bad.md');

        // A third pass, still inside the backoff window, still excludes it.
        delta = await s.orch.computeDelta();
        expect(delta.dirty).not.toContain('bad.md');
    });

    // ── Scenario 5 — a real edit re-embeds; a delete removes from index+search ─
    // The complement of Scenario 2 (changed bytes MUST re-embed) plus the
    // deleted-path drop, both composed through the real store after a cold build.
    it('editing bytes re-embeds, and deleting a note removes it from the index', async () => {
        const s = await boot();
        s.vault.write('note.md', 'original content about astronomy', 1000);
        s.vault.write('keep.md', 'unrelated note about pottery', 1000);
        await s.coldStart();

        // (a) a real edit (bytes change) DOES re-embed
        const spy = vi.spyOn(s.embedder, 'embedBatch');
        await s.edit('note.md', 'rewritten content about marine biology', 2000);
        expect(spy).toHaveBeenCalled();

        // (b) deleting the note drops its file record; the other note survives
        await s.del('note.md');
        const paths = new Set((await s.store.listFileRecords()).map(r => r.note_path));
        expect(paths.has('note.md')).toBe(false);
        expect(paths.has('keep.md')).toBe(true);
    });

    // ── Scenario 6 — synonyms v2: piped-link mining end to end ──────────────
    // The full T2 loop composed through the real pipeline: cold build →
    // ensureBm25 mines metadataCache (the fake getFileCache) → matchPhrases
    // fires on the spelled-out query → the mate reaches a short-form-only note
    // → an edit that removes the piped links retires the mate on the NEXT
    // search (applyDelta's per-path re-mine). Valve semantics ride the per-call
    // ExperimentOverride, the exact CLI synv2 path — settings stay at the
    // shipped default (OFF) throughout, pinning that the DICTIONARY still
    // builds and only consumption is gated.
    it('synonyms v2: a mined piped-link display reaches a mate-only note, and un-linking retires it', async () => {
        const s = await boot();
        // Two piped usages clear T2_MIN_COUNT. The df ceiling is REAL here
        // (5% of chunks), so pad the corpus with fillers: "qu" appears in 2
        // notes → needs ≥41 docs to stay under the ceiling.
        s.vault.write('notes.md', 'see [[Query Understanding|QU]] and [[Query Understanding|QU]] again', 1000);
        s.vault.write('qu.md', 'QU relevance work item', 1000);   // never says query/understanding
        for (let i = 0; i < 45; i++) {
            s.vault.write(`filler${i}.md`, `filler note number ${i} about topic${i}`, 1000);
        }
        await s.coldStart();

        const q = 'query understanding planning';
        const on1 = await s.orch.search(q, 10, undefined, { synonymsV2: true });
        const off1 = await s.orch.search(q, 10, undefined, { synonymsV2: false });

        // The valve gates consumption only: ON fires the phrase mate...
        expect(on1.entry.phraseMatesFired).toBe(1);
        expect(on1.entry.synonymsV2).toBe(true);
        // ...OFF writes neither telemetry field (absent ⇒ off).
        expect(off1.entry.synonymsV2).toBeUndefined();
        expect(off1.entry.phraseMatesFired).toBeUndefined();
        // And the mate is what reaches qu.md: its fused score strictly
        // improves when the phrase mate fires (BM25 0 → mate-discounted hit).
        const quOn = on1.results.find(r => r.note_path === 'qu.md')?.score ?? 0;
        const quOff = off1.results.find(r => r.note_path === 'qu.md')?.score ?? 0;
        expect(quOn).toBeGreaterThan(quOff);

        // Un-link: the edit re-mines notes.md's pairs (applyDelta touched-path
        // patch), dropping the dictionary entry — no piped links, no mate.
        await s.edit('notes.md', 'see Query Understanding twice, no links here', 2000);
        const on2 = await s.orch.search(q, 10, undefined, { synonymsV2: true });
        expect(on2.entry.phraseMatesFired).toBe(0);
    });
    // ── Scenario 7 — BM25F knobs are query-time only ─────────────────────────
    // A weight change (the seek:search per-query override) must reach
    // the very next search through the SAME BM25 cache: no reindex, no refit, and
    // the Σ UB(t) bound recomputed under the new weights (never a memoized bound
    // from the old ones).
    it('bm25f weights: a CLI override applies to the next search with no refit', async () => {
        const s = await boot();
        s.vault.write('cheer.md', '---\nplaceType: bars\nplaceLoc: "[[Austin]]"\n---\nOutdoor patio, live music most nights.', 1000);
        s.vault.write('kino.md', '# Kinokuniya Austin\nJapanese bookstore with stationery.', 1000);
        s.vault.write('austin.md', 'Austin is a city in Texas with many bars downtown.', 1000);
        for (let i = 0; i < 6; i++) s.vault.write(`f${i}.md`, `filler note ${i} about gardening`, 1000);
        await s.coldStart();
        const q = 'bars in austin';
        const cacheOf = () => (s.orch as unknown as { bm25Cache: object | null }).bm25Cache;

        const a = await s.orch.search(q, 10);
        const cache = cacheOf();
        expect(cache).not.toBeNull();
        expect(a.entry.lexicalModel).toBe('bm25f');
        expect(a.entry.bm25fKnobs).toEqual({ title: 10, aliases: 6, properties: 4, body: 1, k1: 5 });

        const b = await s.orch.search(q, 10, undefined, { bm25fKnobs: { properties: 10 } });   // CLI props=10
        expect(cacheOf()).toBe(cache);             // same index instance: nothing refit
        expect(b.entry.bm25CacheHit).toBe(true);
        expect(b.entry.bm25fKnobs?.properties).toBe(10);
        expect(b.entry.bm25Bound).not.toBe(a.entry.bm25Bound);   // UB(bar) re-derived under props 10

        const c = await s.orch.search(q, 10);      // no override: back to the default bound
        expect(c.entry.bm25Bound).toBe(a.entry.bm25Bound);
        expect(c.results.map(r => r.note_path)).toEqual(a.results.map(r => r.note_path));

        const e = await s.orch.search(q, 10, undefined, { bm25f: false });                    // CLI bm25f=false
        expect(e.entry.lexicalModel).toBe('sum');
        expect(e.entry.bm25fKnobs).toBeUndefined();
        expect(cacheOf()).toBe(cache);
    });
});
