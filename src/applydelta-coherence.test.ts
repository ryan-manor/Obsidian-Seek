// Seek scaling A1 — the row-space coupling end to end + the drift detector.
//
// applyDelta itself is a private SearchEngine method wired into reindexDelta (heavy
// deps: embedder, coordinator, vault). Rather than mock the whole engine, these
// tests drive the REAL data-path components (MultiFieldBM25.add/remove/vacuum +
// appendFrameRows/tombstoneFrameRows) through applyDelta's exact mutation sequence,
// then assert the two things that actually carry risk:
//   1. frameBm25Coherent — the runtime drift detector — accepts a coherent pair
//      and trips on every corruption mode (test 8);
//   2. an incrementally-maintained frame+BM25 equals a fresh cold rebuild over the
//      same live corpus, BY chunk_id within epsilon, including delete-only (model-
//      drift data path, test 5) and across the compaction threshold (test 4).
// reindexDelta's control flow (model-drift→empty-adds, runExclusive ordering) reuses
// unchanged machinery and is covered by reading + live smoke (PR notes).

import { describe, it, expect, vi } from 'vitest';
import {
    appendFrameRows, tombstoneFrameRows, frameBm25Coherent, pushDeltaAdds, freshDeltaAdds,
    coherenceDriftDecision, COMPACTION_TOMBSTONE_FRACTION, COHERENCE_DRIFT_COOLDOWN_MS,
    type ResidentFrame, type DeltaAdd, type RowSpaceProbe,
} from './search';
import { MultiFieldBM25 } from './bm25';
import type { Chunk, ChunkMeta } from './types';
import type { QuantVec } from './quant';

const BPV = 4;
const DIM = 8;

function chunk(id: string, content: string): Chunk {
    return {
        chunk_id: id, title: `T-${id}`, content, note_path: `${id}.md`,
        heading_path: [], metadata: { tags: [], aliases: [], created: null, modified: null, properties: {} },
        start_line: 0, end_line: 0,
    };
}
function qvOf(id: string): QuantVec {
    const seed = id.charCodeAt(id.length - 1);
    const q = new Int8Array(DIM);
    for (let i = 0; i < DIM; i++) q[i] = ((seed * 7 + i) % 127) - 63;
    return { q, s: 0.5 };
}
function binOf(id: string): Uint8Array {
    const seed = id.charCodeAt(id.length - 1);
    const b = new Uint8Array(BPV);
    for (let i = 0; i < BPV; i++) b[i] = (seed * 31 + i) & 0xff;
    return b;
}
function metaOf(c: Chunk): ChunkMeta { const { content, ...m } = c; void content; return m; }
function addOf(c: Chunk): DeltaAdd { return { chunk: c, q: qvOf(c.chunk_id), bin: binOf(c.chunk_id) }; }

// Build a coherent {frame, bm25} pair over `chunks` — exactly what ensureFrame +
// ensureBm25 produce on a cold build: row i is chunks[i] in BOTH structures.
function buildPair(chunks: Chunk[]): { frame: ResidentFrame; bm: MultiFieldBM25 } {
    const n = chunks.length;
    const activePacked = new Uint8Array(n * BPV);
    const int8 = new Int8Array(n * DIM);
    const scales = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        activePacked.set(binOf(chunks[i].chunk_id), i * BPV);
        int8.set(qvOf(chunks[i].chunk_id).q, i * DIM);
        scales[i] = qvOf(chunks[i].chunk_id).s;
    }
    const frame: ResidentFrame = {
        orderedChunks: chunks.map(metaOf),
        orderedIds: chunks.map(c => c.chunk_id),
        activePacked, bytesPerVec: BPV,
        residentInt8: int8, residentScales: scales, embDim: DIM,
        validRows: new Array<boolean>(n).fill(true),
        tombstoneCount: 0,
        generation: 1,
    };
    const bm = new MultiFieldBM25().fit(chunks, new Map(chunks.map(c => [c.chunk_id, c.content])));
    return { frame, bm };
}

// applyDelta's exact mutation core (minus the engine's generation re-stamp): the
// real functions, the real order — removes first (tombstone the captured rows),
// then adds, then await vacuum() for bound exactness.
// Mirrors production applyDelta's mutation core INCLUDING its try/catch contract:
// returns true on a clean patch, false on a mid-patch throw (the deltaFallback the
// engine routes through invalidate+rebuild). Filters already-live / within-batch
// duplicate adds through the SHARED freshDeltaAdds (NOT a copy), then feeds the SAME
// list to both sinks so the row spaces stay aligned. Runs after the removes.
async function simulateApplyDelta(frame: ResidentFrame, bm: MultiFieldBM25, adds: DeltaAdd[], removedIds: string[]): Promise<boolean> {
    try {
        const removeRows: number[] = [];
        for (const id of removedIds) { const r = bm.rowOf(id); if (r !== undefined) removeRows.push(r); bm.remove(id); }
        tombstoneFrameRows(frame, removeRows);
        const fresh = freshDeltaAdds(adds, id => bm.rowOf(id) !== undefined);
        for (const a of fresh) bm.add(a.chunk, a.chunk.content);
        appendFrameRows(frame, fresh);
        await bm.vacuum();
        return true;
    } catch {
        return false;   // applyDelta's catch → deltaFallback('exception during patch')
    }
}

function scoresById(bm: MultiFieldBM25, query: string, ids: string[]): Map<string, number> {
    const { scores } = bm.getScoresWithCoverage(query);
    const out = new Map<string, number>();
    for (const id of ids) { const r = bm.rowOf(id); if (r !== undefined) out.set(id, scores[r]); }
    return out;
}
function expectRankParity(inc: MultiFieldBM25, fresh: MultiFieldBM25, query: string, liveIds: string[]): void {
    const a = scoresById(inc, query, liveIds);
    const b = scoresById(fresh, query, liveIds);
    const matched = (m: Map<string, number>) => [...m].filter(([, s]) => s > 0).map(([id]) => id).sort();
    expect(matched(a)).toEqual(matched(b));
    for (const [id, sa] of a) expect(Math.abs(sa - (b.get(id) ?? 0))).toBeLessThan(1e-6);
}

const CORPUS = (): Chunk[] => [
    chunk('c1', 'alpha bravo charlie'),
    chunk('c2', 'alpha bravo delta'),
    chunk('c3', 'alpha echo foxtrot'),
    chunk('c4', 'bravo golf hotel'),
    chunk('c5', 'charlie india juliet'),
    chunk('c6', 'delta echo kilo'),
    chunk('c7', 'foxtrot golf lima'),
    chunk('c8', 'hotel india mike'),
];
const QUERIES = ['alpha bravo', 'charlie', 'golf hotel', 'echo delta'];

describe('frameBm25Coherent — drift detector (test 8)', () => {
    it('accepts a freshly-built coherent pair (sampled and full)', () => {
        const { frame, bm } = buildPair(CORPUS());
        expect(frameBm25Coherent(frame, bm)).toBe(true);
        expect(frameBm25Coherent(frame, bm, true)).toBe(true);
    });

    it('accepts a coherent pair after incremental add + remove', async () => {
        const { frame, bm } = buildPair(CORPUS());
        await simulateApplyDelta(frame, bm, [addOf(chunk('c9', 'november oscar'))], ['c3']);
        expect(frameBm25Coherent(frame, bm, true)).toBe(true);
    });

    it('trips on a row-count (R) mismatch', () => {
        const { frame, bm } = buildPair(CORPUS());
        frame.orderedChunks.push(metaOf(chunk('x', 'x')));   // frame R now exceeds bm.size
        frame.orderedIds.push('x'); frame.validRows.push(true);
        expect(frameBm25Coherent(frame, bm)).toBe(false);
    });

    it('trips on a live-count mismatch (tombstone not reflected in bm25)', () => {
        const { frame, bm } = buildPair(CORPUS());
        frame.validRows[2] = false; frame.tombstoneCount = 1;   // frame says 1 dead, bm25 still has all live
        expect(frameBm25Coherent(frame, bm)).toBe(false);
    });

    it('trips on an id↔row mis-join (the silent-corruption case)', () => {
        const { frame, bm } = buildPair(CORPUS());
        // Swap two orderedIds so idToIdx[orderedIds[i]] !== i — arrays stay the same
        // length (in-bounds), exactly the failure the detector exists to catch.
        [frame.orderedIds[0], frame.orderedIds[1]] = [frame.orderedIds[1], frame.orderedIds[0]];
        expect(frameBm25Coherent(frame, bm, true)).toBe(false);
    });

    it('full mode catches a deep-row mis-join the sample might skip', () => {
        const { frame, bm } = buildPair(CORPUS());
        // Corrupt a row unlikely to be in the 8-sample spread of an 8-row corpus is
        // moot here (all sampled), so prove `full` scans every row on a bigger frame.
        const big = buildPair(Array.from({ length: 50 }, (_, i) => chunk(`d${i}`, `term${i} shared`)));
        [big.frame.orderedIds[37], big.frame.orderedIds[38]] = [big.frame.orderedIds[38], big.frame.orderedIds[37]];
        expect(frameBm25Coherent(big.frame, big.bm, true)).toBe(false);
        expect(frameBm25Coherent(frame, bm)).toBe(true);   // control: untouched pair still coherent
    });
});

describe('incremental == fresh cold rebuild, by chunk_id (test 4)', () => {
    it('add + remove + change stays coherent and matches a fresh fit over the live corpus', async () => {
        const { frame, bm } = buildPair(CORPUS());
        // change c2 (remove old id, add new id) + add c9 + delete c5.
        const c2b = chunk('c2b', 'alpha bravo delta papa quebec');
        await simulateApplyDelta(frame, bm, [addOf(c2b), addOf(chunk('c9', 'november oscar alpha'))], ['c2', 'c5']);

        expect(frameBm25Coherent(frame, bm, true)).toBe(true);
        const liveIds = ['c1', 'c2b', 'c3', 'c4', 'c6', 'c7', 'c8', 'c9'];
        const liveCorpus = [
            chunk('c1', 'alpha bravo charlie'), c2b, chunk('c3', 'alpha echo foxtrot'),
            chunk('c4', 'bravo golf hotel'), chunk('c6', 'delta echo kilo'),
            chunk('c7', 'foxtrot golf lima'), chunk('c8', 'hotel india mike'),
            chunk('c9', 'november oscar alpha'),
        ];
        const fresh = new MultiFieldBM25().fit(liveCorpus, new Map(liveCorpus.map(c => [c.chunk_id, c.content])));
        for (const q of [...QUERIES, 'papa quebec', 'november alpha']) {
            expectRankParity(bm, fresh, q, liveIds);
            expect(bm.getQueryBound(q)).toBeCloseTo(fresh.getQueryBound(q), 9);
        }
        expect(bm.liveCount).toBe(liveIds.length);
    });

    it('delete-only (model-drift data path, test 5) applies and matches', async () => {
        const { frame, bm } = buildPair(CORPUS());
        await simulateApplyDelta(frame, bm, [], ['c1', 'c4']);   // adds empty (drift defers embeds)
        expect(frameBm25Coherent(frame, bm, true)).toBe(true);

        const liveIds = ['c2', 'c3', 'c5', 'c6', 'c7', 'c8'];
        const liveCorpus = CORPUS().filter(c => liveIds.includes(c.chunk_id));
        const fresh = new MultiFieldBM25().fit(liveCorpus, new Map(liveCorpus.map(c => [c.chunk_id, c.content])));
        for (const q of QUERIES) expectRankParity(bm, fresh, q, liveIds);
        expect(bm.liveCount).toBe(6);
        expect(bm.size).toBe(8);   // R unchanged — tombstone holes kept until compaction
    });

    it('crossing the tombstone threshold is detectable, and a dense rebuild (compaction) matches', async () => {
        const { frame, bm } = buildPair(CORPUS());
        // Remove 3 of 8 = 0.375 > 0.25 → compaction is due.
        await simulateApplyDelta(frame, bm, [], ['c1', 'c2', 'c3']);
        const tombFraction = frame.tombstoneCount / frame.orderedChunks.length;
        expect(tombFraction).toBeGreaterThanOrEqual(COMPACTION_TOMBSTONE_FRACTION);

        // Compaction in the engine = invalidate→warmCaches→fresh fit + dense frame.
        const liveIds = ['c4', 'c5', 'c6', 'c7', 'c8'];
        const liveCorpus = CORPUS().filter(c => liveIds.includes(c.chunk_id));
        const compacted = buildPair(liveCorpus);   // a fresh cold build over the live set
        expect(compacted.frame.tombstoneCount).toBe(0);
        expect(compacted.frame.validRows.every(Boolean)).toBe(true);
        expect(compacted.frame.orderedIds).toEqual(liveIds);   // dense, cold order
        expect(frameBm25Coherent(compacted.frame, compacted.bm, true)).toBe(true);
        // The compacted index ranks identically to the pre-compaction incremental one.
        for (const q of QUERIES) expectRankParity(bm, compacted.bm, q, liveIds);
    });
});

describe('pushDeltaAdds — both commit paths surface the same change-set shape', () => {
    it('aligns chunks[i] with tiers[i] and appends to the sink in order', () => {
        // The sidecar-hydrate path (Seek scaling A1): hydrateFromSidecar hands
        // putQuantized full chunks + {q,bin} tiers; pushDeltaAdds turns them into
        // the SAME DeltaAdd shape commitFile produces, so a dedup delta feeds
        // applyDelta incrementally instead of forcing a rebuild.
        const sink: DeltaAdd[] = [];
        const chunks = [chunk('h1', 'hydrated one'), chunk('h2', 'hydrated two')];
        const tiers = [
            { q: qvOf('h1'), bin: binOf('h1') },
            { q: qvOf('h2'), bin: binOf('h2') },
        ];
        pushDeltaAdds(sink, chunks, tiers);
        expect(sink.map(a => a.chunk.chunk_id)).toEqual(['h1', 'h2']);
        expect(sink[0].q).toBe(tiers[0].q);
        expect(sink[0].bin).toBe(tiers[0].bin);
        expect(sink[0].chunk.content).toBe('hydrated one');

        // Appends (doesn't replace) — commitFile + hydrate share one sink per delta.
        pushDeltaAdds(sink, [chunk('h3', 'three')], [{ q: qvOf('h3'), bin: binOf('h3') }]);
        expect(sink.map(a => a.chunk.chunk_id)).toEqual(['h1', 'h2', 'h3']);
    });

    it('hydrate-sourced adds process through applyDelta like any other add', async () => {
        const { frame, bm } = buildPair(CORPUS());
        // Simulate a sidecar-dedup delta: the file's stale chunk was dropped
        // (removedIds), and the same content hydrated from a peer's shard (adds
        // built by pushDeltaAdds), exactly what reindexDelta now produces.
        const hydratedSink: DeltaAdd[] = [];
        const reHydrated = chunk('c3', 'alpha echo foxtrot');   // unchanged content, same id
        pushDeltaAdds(hydratedSink, [reHydrated], [{ q: qvOf('c3'), bin: binOf('c3') }]);
        await simulateApplyDelta(frame, bm, hydratedSink, ['c3']);

        expect(frameBm25Coherent(frame, bm, true)).toBe(true);
        const allIds = CORPUS().map(c => c.chunk_id);
        const fresh = new MultiFieldBM25().fit(CORPUS(), new Map(CORPUS().map(c => [c.chunk_id, c.content])));
        for (const q of QUERIES) expectRankParity(bm, fresh, q, allIds);
        expect(bm.liveCount).toBe(allIds.length);   // c3 removed then re-added → still live
    });
});

// The 2026-06-18 mobile meltdown: a sidecar-hydrate delta surfaced a chunk_id
// already live in the in-memory BM25 as a PURE add (no matching remove, because
// IDB — which deleteFile and existingIds both read — had diverged from the cache).
// The unguarded bm.add() threw "duplicate ID", aborting applyDelta mid-patch and
// leaving frame/BM25 mis-coupled; every reconcile re-tripped it → toast+rebuild
// loop → thermal crash. freshDeltaAdds is the guard.
describe('freshDeltaAdds — duplicate-id guard', () => {
    it('drops adds whose id is already live, keeps genuinely new ones', () => {
        const live = new Set(['c1', 'c2', 'c3']);
        const adds = [addOf(chunk('c2', 'x')), addOf(chunk('c9', 'new')), addOf(chunk('c1', 'y'))];
        const fresh = freshDeltaAdds(adds, id => live.has(id));
        expect(fresh.map(a => a.chunk.chunk_id)).toEqual(['c9']);
    });

    it('drops within-batch duplicate ids, keeping the first occurrence', () => {
        const adds = [addOf(chunk('h1', 'a')), addOf(chunk('h1', 'a')), addOf(chunk('h2', 'b'))];
        const fresh = freshDeltaAdds(adds, () => false);
        expect(fresh.map(a => a.chunk.chunk_id)).toEqual(['h1', 'h2']);
    });

    it('does NOT filter an edit re-commit — its stale id is removed before the check', () => {
        // The edit path drops the stale id from bm first, so by the time the filter
        // runs the (same content-hash) id is no longer live → the re-add survives.
        const live = new Set(['c1', 'c2']);   // c3's stale row already removed
        const fresh = freshDeltaAdds([addOf(chunk('c3', 'edited'))], id => live.has(id));
        expect(fresh.map(a => a.chunk.chunk_id)).toEqual(['c3']);
    });
});

describe('hydrate duplicate-id is absorbed, not thrown (the meltdown regression)', () => {
    it('MiniSearch.add throws on a live duplicate — the raw failure the guard prevents', () => {
        const { bm } = buildPair(CORPUS());
        // c5 is already live; a raw re-add with NO remove first is exactly what the
        // unguarded loop did with a hydrate duplicate → "duplicate ID" → aborted patch.
        expect(() => bm.add(metaOf(chunk('c5', 'charlie india juliet')), 'charlie india juliet')).toThrow();
    });

    it('a pure hydrate add for an already-live id (no matching remove) stays coherent', async () => {
        const { frame, bm } = buildPair(CORPUS());
        // IDB lacked c5 so the hydrate re-surfaced it as a PURE add (removedIds empty
        // — deleteFile found nothing in IDB to drop), yet c5 is still live in `bm`.
        const sink: DeltaAdd[] = [];
        pushDeltaAdds(sink, [chunk('c5', 'charlie india juliet')], [{ q: qvOf('c5'), bin: binOf('c5') }]);
        await expect(simulateApplyDelta(frame, bm, sink, [])).resolves.toBe(true);   // applied, no throw

        expect(frameBm25Coherent(frame, bm, true)).toBe(true);
        const allIds = CORPUS().map(c => c.chunk_id);
        const fresh = new MultiFieldBM25().fit(CORPUS(), new Map(CORPUS().map(c => [c.chunk_id, c.content])));
        for (const q of QUERIES) expectRankParity(bm, fresh, q, allIds);
        expect(bm.liveCount).toBe(allIds.length);   // c5 not duplicated; nothing lost
        expect(bm.size).toBe(allIds.length);        // no phantom row appended
    });

    it('a mixed delta (live dup + genuinely-new add) applies only the new one, coherently', async () => {
        const { frame, bm } = buildPair(CORPUS());
        const sink: DeltaAdd[] = [];
        pushDeltaAdds(sink,
            [chunk('c2', 'alpha bravo delta'), chunk('c9', 'november oscar')],   // c2 live, c9 new
            [{ q: qvOf('c2'), bin: binOf('c2') }, { q: qvOf('c9'), bin: binOf('c9') }]);
        await simulateApplyDelta(frame, bm, sink, []);
        expect(frameBm25Coherent(frame, bm, true)).toBe(true);
        expect(bm.rowOf('c9')).toBeDefined();             // the new add landed
        expect(bm.liveCount).toBe(CORPUS().length + 1);   // +c9 only (c2 dup absorbed)
    });
});

// L2 — exception safety. A throw PAST the L1 filter (e.g. a vacuum/IDB hiccup)
// must degrade to the fallback (false → caller invalidates + rebuilds from IDB),
// never escape and strand a half-mutated cache. simulateApplyDelta mirrors the
// production try/catch → boolean contract.
describe('applyDelta exception safety (L2)', () => {
    it('a throw mid-patch degrades to the fallback (false), not an escape', async () => {
        const { frame, bm } = buildPair(CORPUS());
        // vacuum() runs after the adds land; force it to reject to model an
        // unexpected mid-patch failure the L1 filter cannot prevent.
        const spy = vi.spyOn(bm, 'vacuum').mockRejectedValueOnce(new Error('simulated vacuum failure'));
        const applied = await simulateApplyDelta(frame, bm, [addOf(chunk('c9', 'november oscar'))], []);
        expect(applied).toBe(false);   // deltaFallback contract — exception caught, not thrown
        spy.mockRestore();
    });

    it('a clean patch returns true (the contract the fallback is the negative of)', async () => {
        const { frame, bm } = buildPair(CORPUS());
        const applied = await simulateApplyDelta(frame, bm, [addOf(chunk('c9', 'november oscar'))], ['c3']);
        expect(applied).toBe(true);
        expect(frameBm25Coherent(frame, bm, true)).toBe(true);
    });
});

// L3 — the drift-response circuit breaker. coherenceDriftDecision is the pure core
// of onCoherenceDrift; pinning it here guards the three silent regressions that
// re-open the 2026-06-18 meltdown loop: dropping the always-invalidate, never
// suppressing the re-warm, or flipping the cooldown comparison.
describe('coherenceDriftDecision — drift backoff (L3)', () => {
    const CD = COHERENCE_DRIFT_COOLDOWN_MS;
    it('first trip (lastWarmAt = -Infinity) warms', () => {
        expect(coherenceDriftDecision(1000, -Infinity, CD)).toEqual({ invalidate: true, warm: true });
    });
    it('a re-trip inside the cooldown suppresses the warm', () => {
        expect(coherenceDriftDecision(1000 + CD - 1, 1000, CD)).toEqual({ invalidate: true, warm: false });
    });
    it('a trip exactly at the cooldown boundary warms again (>=)', () => {
        expect(coherenceDriftDecision(1000 + CD, 1000, CD)).toEqual({ invalidate: true, warm: true });
    });
    it('a trip past the cooldown warms again', () => {
        expect(coherenceDriftDecision(1000 + 5 * CD, 1000, CD)).toEqual({ invalidate: true, warm: true });
    });
    it('invalidate is ALWAYS true regardless of timing (correctness, never throttled)', () => {
        for (const [now, last] of [[0, -Infinity], [1, 1], [1, 0], [CD, 0], [CD - 1, 0]] as const) {
            expect(coherenceDriftDecision(now, last, CD).invalidate).toBe(true);
        }
    });
});

// RowSpaceProbe is the minimal surface the detector needs — assert MultiFieldBM25
// satisfies it (compile-time guard that the structural typing holds).
describe('RowSpaceProbe', () => {
    it('MultiFieldBM25 satisfies the probe shape', () => {
        const bm = buildPair(CORPUS()).bm;
        const probe: RowSpaceProbe = bm;
        expect(typeof probe.size).toBe('number');
        expect(typeof probe.liveCount).toBe('number');
        expect(probe.rowOf('c1')).toBe(0);
    });
});
