// Hybrid ranker. Algorithm:
//   1. dense_norm = TM2C2 theoretical norm of cosine  ((cos+1)/2)
//   2. bm25_norm  = TM2C2 theoretical norm of BM25     (/ per-query THEORETICAL
//                   bound from bm25.getQueryBound, clipped; falls back to the
//                   per-query max when the bound is 0 — see fusion.ts)
//   3. hybrid     = alpha * dense_norm + (1 - alpha) * bm25_norm
//   4. recency    = 0.5^(days_old / halfLife)   // age via the vault's recencyKey
//   5. final      = hybrid + eps * recency + title_boost   // ε-TIEBREAKER, not a blend
//
// FUSION HISTORY: through 2026-06-09 the linear blend min-max-normalized each
// channel per query, which manufactured a false 1.00 dense winner on OOV/no-
// opinion queries (the empty-note/opaque-ID bug). That was first patched with a
// lexical-only floor + an absolute-cosine confidence gate (magic 0.79/0.83
// thresholds), then replaced wholesale: an eval study (research + ~/eval-oov
// OOV slice) showed per-query min-max IS the root cause and TM2C2 theoretical
// normalization (fusion.ts) fixes it constant-free while improving aggregate
// nDCG. The gate is deleted; see [[seek-empty-stub-dense-pollution]].

import type { ChunkMeta, ScoredChunk, RecencyKeyChoice } from './types';
import { theoreticalNormDense, theoreticalNormBm25, hybridFusion, computeRecencyScore, recencyDate, titleMatchBoost } from './fusion';

// (BlendMode deleted 2026-06-11: the opt-in 'rrf' mode is gone — linear TM2C2
// is the only fusion. RRF lost the head-to-heads and discards score magnitude;
// see fusion.ts and [[Seek Rel]] §Rejected Approaches.)

export interface RankingConfig {
    alpha: number;
    recencyEpsilon: number;
    recencyHalfLifeDays: number;
    recencyKey: RecencyKeyChoice;
    createdProp: string;
    titleBoost: number;
}

export const DEFAULT_RANKING_CONFIG: RankingConfig = {
    // alpha = dense weight, on the BOUND-NORMALIZED scale (2026-06-09 fusion
    // completion: BM25 is divided by its per-query theoretical bound, not its
    // empirical max — fusion.ts theoreticalNormBm25). NOT comparable to either
    // earlier scale (min-max ~0.5–0.7, empirical-max TM2C2 0.90–0.95): the bound
    // compresses the lexical channel, so the optimum sits LOWER. 0.70–0.85 is a
    // flat plateau on BOTH the 482-q personal eval (0.8756 @ 0.70) AND the D&D
    // OOV slice (0.9364 @ 0.80) — the first scale on which one global alpha is
    // optimal across domains, which is the point of the bound (query-invariant
    // channel scale). 0.80 is the shipped joint point. The empirical-max-era
    // "saturated single-term BM25 hit" captures are handled by the norm itself
    // now (a weak best match no longer gets a forced 1.0), not by the weight.
    // The live value comes from SeekSettings.denseWeight (search.ts); this
    // default backs the eval harness + any caller that omits config.
    alpha: 0.80,
    // ε-TIEBREAKER (2026-06-11, [[Seek Rel]] §Recency Plan 2026-06-11) — the
    // additive nudge in final = hybrid + ε·recency + titleBoost. Deliberately
    // sized BELOW any meaningful score gap so it can only break genuine
    // near-ties (dated-series siblings with identical coverage boost and
    // near-identical dense vectors), where newest-first is a sane prior over
    // arbitrary frame order. It must NEVER become a lean: the 06-04 click study
    // showed 50% of episodic clicks target an instance >90d old, so any recency
    // term strong enough to reorder real score gaps loses (forced newest-first
    // within a series scored WORSE than flat relevance, MRR 0.242 vs 0.310).
    // That study is also why this replaced the old (1−rw)·hybrid + rw·recency
    // blend, whose history was 0.25 → 0.10 → parked at 0 (2026-06-08..11).
    // Ship-and-observe value, not eval-swept: the frozen eval sets are
    // single-gold known-item and therefore recency-inert by construction.
    recencyEpsilon: 0.02,
    // Smooth half-life replacing the two-stage 30d-cutoff decay, which zeroed
    // the MEDIAN episodic click target (83d old). 180d = the 06-04 operating point.
    recencyHalfLifeDays: 180,
    // Which date "recent" means — vault-global, settings-driven (SeekSettings.
    // recencyKey via search.ts). See fusion.ts recencyDate for the semantics
    // and why 'created' is the default.
    recencyKey: 'created',
    // Which frontmatter property holds the creation date — `created` is THIS
    // vault's convention, not an Obsidian builtin, so it's settings-driven
    // (SeekSettings.createdProp) for portability. Resolved read-side from the
    // indexed properties map; see the fusion.ts recencyDate ladder.
    createdProp: 'created',
    // Coverage-based known-item boost (fusion.ts titleMatchBoost): boost *
    // precision when all query tokens are in the title. 0.8 = the swept knee on
    // the 482-q old-log eval (nDCG@10 0.8143→0.8677). Was 0.025 under the old
    // exact-equality semantics — NOT comparable; do not "revert" to 0.025.
    titleBoost: 0.8,
};

export interface RankBreakdown {
    final: Float64Array;
    denseNorm: Float64Array;
    bm25Norm: Float64Array;
    hybrid: Float64Array;
    recency: Float64Array;
    titleBoost: Float64Array;
}

function safeFloat(v: number): number {
    if (Number.isNaN(v) || !Number.isFinite(v)) return 0.0;
    return v;
}

export function rank(
    chunks: ChunkMeta[],
    denseScores: Float64Array | Float32Array | number[],
    bm25Scores: Float64Array | Float32Array | number[],
    query: string,
    topK: number,
    config: RankingConfig = DEFAULT_RANKING_CONFIG,
    // Per-query theoretical BM25 bound (bm25.getQueryBound, threaded through
    // search.ts from the same MiniSearch pass that produced bm25Scores).
    // Omitted/0 → fusion falls back to the empirical per-query max.
    bm25Bound?: number,
): { results: ScoredChunk[]; breakdown: RankBreakdown } {
    const n = chunks.length;
    if (n === 0) {
        const empty = new Float64Array(0);
        return {
            results: [],
            breakdown: { final: empty, denseNorm: empty, bm25Norm: empty, hybrid: empty, recency: empty, titleBoost: empty },
        };
    }

    // ---- Lexical-only floor (belt-and-suspenders) ----
    // Title-only fallbacks for body-LESS notes (chunker.ts) embed as just their
    // title — a content-free near-neighbor. TM2C2 already stops a flat/noisy dense
    // channel from manufacturing a winner, so this floor is no longer load-bearing,
    // but flooring such chunks to the real-candidate min is cheap insurance (BM25's
    // 3.0x title boost still finds them by name). Mask a COPY so the caller's raw
    // cosines stay intact for telemetry (denseRaw).
    let denseForNorm: Float64Array | Float32Array | number[] = denseScores;
    let anyLexicalOnly = false;
    let denseRealMin = Infinity;
    for (let i = 0; i < n; i++) {
        if (chunks[i].lexicalOnly) { anyLexicalOnly = true; continue; }
        if (denseScores[i] < denseRealMin) denseRealMin = denseScores[i];
    }
    if (anyLexicalOnly) {
        const floor = isFinite(denseRealMin) ? denseRealMin : 0; // degenerate: all lexical-only
        const masked = new Float64Array(n);
        for (let i = 0; i < n; i++) masked[i] = chunks[i].lexicalOnly ? floor : denseScores[i];
        denseForNorm = masked;
    }

    // TM2C2 theoretical normalization (fusion.ts): fixed endpoints, so a no-opinion
    // dense channel stays flat and cannot out-vote a real BM25 hit. denseNorm/bm25Norm
    // are reported as ranking_signals. (The opt-in RRF blend mode was deleted
    // 2026-06-11 — linear TM2C2 is the only fusion.)
    const denseNorm = theoreticalNormDense(denseForNorm);
    const bm25Norm = theoreticalNormBm25(bm25Scores, bm25Bound);
    const hybrid = hybridFusion(denseNorm, bm25Norm, config.alpha);

    // Recency reads through the SHARED recencyDate accessor (fusion.ts) with the
    // vault-global recencyKey, the same source the S1 candidate arm (search.ts
    // topByRecency) and browseOrder use — arm ⇄ scorer ⇄ browse must agree.
    const recency = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        recency[i] = computeRecencyScore(recencyDate(chunks[i], config.recencyKey, config.createdProp), {
            halfLifeDays: config.recencyHalfLifeDays,
        });
    }

    const titleBoost = titleMatchBoost(query, chunks, config.titleBoost);

    const final = new Float64Array(n);
    for (let i = 0; i < n; i++) final[i] = hybrid[i] + config.recencyEpsilon * recency[i] + titleBoost[i];

    // Top-k by score (descending).
    const indices = Array.from({ length: n }, (_, i) => i);
    indices.sort((a, b) => final[b] - final[a]);
    const top = indices.slice(0, topK);

    const results: ScoredChunk[] = top.map(idx => ({
        ...chunks[idx],
        // Frame-lite: the frame holds metadata only, so the body is a placeholder
        // here — search.ts hydrates `content` on the ≤topK survivors before
        // snippets/render (and the modal highlight). Keep it a string, not
        // optional, so every ScoredChunk consumer stays total.
        content: '',
        score: safeFloat(final[idx]),
        ranking_signals: {
            dense: safeFloat(denseNorm[idx]),
            bm25: safeFloat(bm25Norm[idx]),
            hybrid: safeFloat(hybrid[idx]),
            recency: safeFloat(recency[idx]),
            title_boost: safeFloat(titleBoost[idx]),
            // Original cosine, before the lexical-only floor + min-max above.
            // denseScores is the untouched input array (the floor masks a copy).
            denseRaw: safeFloat(denseScores[idx]),
        },
    }));

    return {
        results,
        breakdown: { final, denseNorm, bm25Norm, hybrid, recency, titleBoost },
    };
}

// Cosine similarity for dense scores. Inputs must already be unit-normalized
// (transformers.js with normalize:true does this for us in the embedder).
export function cosineScores(
    queryVec: Float32Array,
    chunkVecs: Float32Array[],
): Float64Array {
    const dim = queryVec.length;
    const n = chunkVecs.length;
    const scores = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        const v = chunkVecs[i];
        let dot = 0;
        for (let d = 0; d < dim; d++) dot += queryVec[d] * v[d];
        scores[i] = dot;
    }
    return scores;
}
