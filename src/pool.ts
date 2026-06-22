// Stage-1 candidate-pool caps (Seek Retrieval Relevance & Query §Two-Stage
// ANN → Rerank, the [!done] callout; scaling design = docs/seek-scaling.md "C").
// The union of the three arms (binary ∪ bm25 ∪ recency) is the working set the
// fp32 exact rerank touches in stage 2, so these caps size everything downstream
// — and on mobile (no resident int8 block) the union is also one IDB read per
// member (search.ts getEmbeddingsByIds), which is the binding cost the CEILINGS
// protect.
//
// ── Why scale with corpus size ───────────────────────────────────────────────
// The binary SCAN is already O(corpus) and independent of the cap (scoreAsymmetric
// touches every packed vector; the cap only sizes the bounded heap, O(corpus·log
// cap) — logarithmic, negligible). So raising a cap is nearly free on the scan
// side; the only real cost is the larger union. Meanwhile a FIXED top-200 covers a
// shrinking fraction of a growing corpus while the bm25/recency backstop stays
// fixed, so gold-in-pool erodes at scale. We therefore grow the recall arms with
// live N — but conservatively, anchored so today's vault sits exactly at the floor
// (zero change until it grows past the anchor).
//
//   FLOORS — today's validated constants. BINARY 200 is the knee: asymmetric
//     binary recovers the fp32 ceiling at N≥200 on 512-d Gemma and 320-d F2LLM
//     (bake-off Phase-9). At our ~5k corpora 200→400 = +0.000 net because the
//     bm25+recency backstop already lifts gold-in-pool to 0.991 (binary-recall
//     validation 2026-06-09) — so the floor is provably enough here, and scaling
//     only matters above current scale. BM25 100 recovers the ~9% dense-
//     unreachable gold ("exact dense caps at ~0.91 recall at any N"). RECENCY 50
//     is a coverage backstop so the newest notes are always reachable (the
//     ε-tiebreaker in ranker.ts can then surface them).
//   CEILINGS — BINARY 800 is the measured granite knee: anisotropy (mean cos
//     ~0.76) means the binary arm keeps improving to ~N=800 even at 5k, and
//     saturates past it, so capping there is the point of diminishing returns,
//     not arbitrary. BM25 400 (4× floor) tracks it. Both keep the union ≤~1150
//     even at 200k chunks → mobile IDB fetch stays bounded.
//   √N curve — the standard ANN probe-scaling heuristic. cap = floor·√(N/anchor),
//     clamped [floor, ceil]. Hits the binary ceiling at ~80k chunks.
//   RECENCY is NOT scaled — its job is "the newest note is reachable," which is
//     coverage-of-newest (a constant concern), not a fraction of the corpus.

export const POOL_ANCHOR_N = 5000;

export const POOL_FLOORS = { binary: 200, bm25: 100, recency: 50 } as const;
export const POOL_CEILINGS = { binary: 800, bm25: 400 } as const;

export interface PoolCaps {
    binary: number;
    bm25: number;
    recency: number;
}

// √N-anchored cap, clamped to [floor, ceil]. At or below the anchor the
// multiplier is ≤1, so small/current vaults sit exactly at the floor (no change
// from the pre-scaling constants). Monotonic non-decreasing in liveN.
export function scaledCap(floor: number, ceil: number, liveN: number, anchor: number = POOL_ANCHOR_N): number {
    if (liveN <= anchor) return floor;
    const scaled = Math.round(floor * Math.sqrt(liveN / anchor));
    return Math.min(ceil, Math.max(floor, scaled));
}

// The three stage-1 caps for a corpus of `liveN` live chunks (tombstones
// excluded — pass orderedChunks.length − tombstoneCount). Binary + BM25 scale by
// √N; recency is held at its floor by design.
export function poolCaps(liveN: number): PoolCaps {
    return {
        binary: scaledCap(POOL_FLOORS.binary, POOL_CEILINGS.binary, liveN),
        bm25: scaledCap(POOL_FLOORS.bm25, POOL_CEILINGS.bm25, liveN),
        recency: POOL_FLOORS.recency,
    };
}
