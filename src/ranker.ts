// Hybrid ranker. Algorithm:
//   1. dense_norm = TM2C2 theoretical norm of cosine  ((cos+1)/2)
//   2. bm25_norm  = TM2C2 theoretical norm of BM25     (/ per-query THEORETICAL
//                   bound from bm25.getQueryBound, clipped; falls back to the
//                   per-query max when the bound is 0 — see fusion.ts)
//   3. hybrid     = alpha * dense_norm + (1 - alpha) * bm25_norm
//   4. recency    = 0.5^(days_old / halfLife)   // age via the vault's recencyKey
//   5. final      = hybrid + eps * recency + title_boost
//                   // ε-TIEBREAKER, not a blend
//
// FUSION HISTORY: through 2026-06-09 the linear blend min-max-normalized each
// channel per query, which manufactured a false 1.00 dense winner on OOV/no-
// opinion queries (the empty-note/opaque-ID bug). That was first patched with a
// lexical-only floor + an absolute-cosine confidence gate (magic 0.79/0.83
// thresholds), then replaced wholesale: an eval study (research + ~/seek-dnd-eval
// OOV slice) showed per-query min-max IS the root cause and TM2C2 theoretical
// normalization (fusion.ts) fixes it constant-free while improving aggregate
// nDCG. The gate is deleted; see [[seek-empty-stub-dense-pollution]].

import { DEFAULT_SETTINGS, type ChunkMeta, type ScoredChunk, type RecencyKeyChoice } from './types';
import { theoreticalNormDense, theoreticalNormBm25, hybridFusion, computeRecencyScore, recencyDate, titleMatchBoost, type TitleFuzzy } from './fusion';

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
    // Typo tolerance for the title gate (fusion.ts TitleFuzzy): per query token,
    // the edit budget it may fuzz by against title tokens. search.ts supplies it
    // (OOV-in-index tokens only, FUZZY_BY_LENGTH budget) when fuzzy is on; absent
    // = exact/prefix only, the pre-2026-11-12 behavior.
    titleFuzzy?: TitleFuzzy;
}

// ONE source of truth: the ranking defaults ARE the plugin's shipped settings
// (types.ts DEFAULT_SETTINGS — the field comments there carry each value's
// eval history). Until 2026-09-02 this object re-declared its own copy
// (titleBoost 0.8 / ε 0.02·180d / recencyKey 'created') that had drifted from
// what shipped (0.5 / 0·180d / 'modified'; 'created' is the default since
// 2026-09-02), so the harness default, the unit
// tests and the CLI help were all validating a config no install ran. Derive,
// never re-declare. The live values still come from SeekSettings per search
// (search.ts); this backs the eval harness and any caller that omits config.
export const DEFAULT_RANKING_CONFIG: RankingConfig = {
    alpha: DEFAULT_SETTINGS.denseWeight,
    recencyEpsilon: DEFAULT_SETTINGS.recencyEpsilon,
    recencyHalfLifeDays: DEFAULT_SETTINGS.recencyHalfLifeDays,
    recencyKey: DEFAULT_SETTINGS.recencyKey,
    createdProp: DEFAULT_SETTINGS.createdProp,
    titleBoost: DEFAULT_SETTINGS.navTitleBoost,
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

    const titleBoost = titleMatchBoost(query, chunks, config.titleBoost, true, config.titleFuzzy);

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
