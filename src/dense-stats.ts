// Corpus-level dense-cosine background statistics for the corpus-AGNOSTIC
// display calibration + answerability gate.
//
// The dense embedding space is strongly anisotropic (granite-r2: mean pairwise
// cosine ~0.76 on this author's corpus, corpus mean-vector norm ~0.87 on unit
// vectors). A raw cosine of 0.80 is therefore meaningless out of context — it
// could be an excellent match or pure background floor. These two scalars LOCATE
// that floor, so query time can express any cosine as a corpus-normalized
// z-score  z = (cos − mean)/std  that is unitless and transfers across vaults
// and encoders. A z-threshold shipped as a constant stays corpus-agnostic
// because the per-corpus floor is divided out here, the same way BM25 IDF is
// fit per corpus rather than baked in.
//
// These are NOT a ranking input. We measured (2026-06-16) that folding a
// background-relative rescale into fusion HURT nDCG@10 by −0.0135 (the upper
// clamp flattens genuine top matches), so the stats feed display + gate only.
// Computed once per FULL reindex (see search.ts embedAndCommitFiles); a delta
// carries the prior values forward — they are coarse corpus globals a few
// changed files barely move.

export interface DenseBgStats {
    mean: number;   // mean off-diagonal doc-doc cosine (closed form, exact)
    std: number;    // std of off-diagonal doc-doc cosine (bounded random sample)
}

// Below this many indexed vectors the background estimate is too noisy to
// calibrate against, so calibration stays OFF (raw scores shown) until a real
// corpus exists. Set from a measured noise floor, NOT a guess: subsampling N
// vectors from the granite-regime corpus 3000× and propagating (μ,σ) wobble
// into the displayed % (match_strength_noise.py, 2026-06-19) shows the noise
// is finite-corpus estimator variance (the 8k-pair σ Monte-Carlo is ~13% of
// it), decaying smoothly as ~1/√N with no cliff. At N=200 a genuine strong
// match (~90%) has a 90%-band of ±2pp and the 50% midpoint ±3.5pp — below
// display resolution; σ is also stable to its last significant digit (≤0.001)
// by N≈250. 500 was ~2–3× conservative. Floor at 200; below it the midpoint
// band approaches ~10pp, so a precise % would start to over-claim.
export const MIN_BG_SAMPLE = 200;

// Reservoir size (uniform sample of corpus vectors retained for the σ estimate)
// and number of random pairs drawn from it. 4096 vectors is a solid uniform
// sample; ~8k pairs over them gives a σ stable to the last reported digit. Both
// are a rounding-error cost against the embed pass that produced the vectors.
export const BG_RESERVOIR = 4096;
export const BG_PAIR_SAMPLE = 8000;

// μ closed form. For L2-normalized vectors the mean off-diagonal cosine is an
// EXACT function of the corpus mean vector's norm — no pair enumeration needed:
//
//     μ = (N·‖v̄‖² − 1) / (N − 1),     v̄ = (1/N) Σ vᵢ
//
// Derivation: ‖v̄‖² = (1/N²) Σᵢ Σⱼ vᵢ·vⱼ = (1/N²)(N + Σ_{i≠j} vᵢ·vⱼ). The diagonal
// contributes exactly N (unit vectors), so Σ_{i≠j} = N²‖v̄‖² − N and the
// off-diagonal mean is that over N(N−1). Verified on the live personal corpus:
// this route gives 0.75991 vs a 300k-pair sample's 0.75987.
export function bgMeanFromSum(sum: Float64Array, n: number): number {
    if (n < 2) return 0;
    let normSq = 0;
    for (let d = 0; d < sum.length; d++) {
        const mean = sum[d] / n;
        normSq += mean * mean;
    }
    return (n * normSq - 1) / (n - 1);
}

// σ from a bounded random-pair sample. The exact σ would need the full Gram
// matrix (O(N²)); the std of a few thousand random off-diagonal pairs converges
// to it within the last significant digit (test corpus: n=2000 pairs → err
// < 0.002). `rand` is injectable so tests are deterministic. Population std
// (÷ m), matching the closed-form μ's full-population semantics.
export function bgStdSampled(
    vecs: Float32Array[],
    pairs: number,
    rand: () => number = Math.random,
): number {
    const n = vecs.length;
    if (n < 2) return 0;
    let s = 0, s2 = 0, m = 0;
    for (let k = 0; k < pairs; k++) {
        const i = Math.min((rand() * n) | 0, n - 1);
        let j = Math.min((rand() * n) | 0, n - 1);
        if (i === j) j = (j + 1) % n;     // off-diagonal only
        const a = vecs[i], b = vecs[j];
        let dot = 0;
        for (let d = 0; d < a.length; d++) dot += a[d] * b[d];
        s += dot; s2 += dot * dot; m++;
    }
    if (m === 0) return 0;
    const mean = s / m;
    return Math.sqrt(Math.max(0, s2 / m - mean * mean));
}

// The whole index-time decision in one pure, testable place. Returns null when
// the corpus is below MIN_BG_SAMPLE (→ calibration disabled, raw scores shown).
export function denseBgStats(
    sum: Float64Array,
    n: number,
    reservoir: Float32Array[],
    pairs: number = BG_PAIR_SAMPLE,
    rand: () => number = Math.random,
): DenseBgStats | null {
    if (n < MIN_BG_SAMPLE) return null;
    return { mean: bgMeanFromSum(sum, n), std: bgStdSampled(reservoir, pairs, rand) };
}

// ---- Display-only confidence (query-side consumer) -------------------------
// Express a raw cosine as a 0..1 confidence RELATIVE TO THIS CORPUS's background,
// via the same z = (cos − mean)/std, squashed through a fixed logistic. DISPLAY
// ONLY — never a ranking input (folding a background rescale into fusion measured
// −0.0135 nDCG@10). The point it solves: anisotropy bunches every raw cosine into
// a ~0.04-wide band (0.74–0.80 on granite-r2), so the raw number looks identical
// for a great match and a mediocre one; dividing out the corpus background spreads
// them into a readable %. The 2026-06-16 held-out-gold study showed the ABSOLUTE
// z of a true match is corpus-dependent (personal z~3, code z~6), so this is an
// honest WITHIN-vault confidence, NOT a cross-vault answerability claim — it ships
// as a diagnostic, not a hard gate. The two constants are display-aesthetic (they
// reshape an already corpus-normalized z into a %), not tuned to any corpus.
export const CONF_Z0 = 2.0;       // z mapped to 50% — ~2σ above background
export const CONF_ZSCALE = 0.9;   // logistic softness; ±1σ ≈ 75%/25% around z0

export function calibratedConfidence(cos: number, mean: number, std: number): number {
    if (!(std > 0)) return 0;
    const z = (cos - mean) / std;
    return 1 / (1 + Math.exp(-(z - CONF_Z0) / CONF_ZSCALE));
}

// ---- Match strength: the single user-facing relevance read --------------------
// One human-interpretable 0..1 number per result, fusing the two SEPARATELY-
// calibrated arms the same way fusion weights them (denseWeight α) but with
// recency EXCLUDED — it answers "how well does this match the query," not
// "should it rank high." DISPLAY ONLY; ranking still uses the fused score incl.
// recency (folding any background rescale into fusion measured −0.0135 nDCG@10),
// so the % can and should diverge from rank order.
//
//   strength = α·pDense + (1 − α)·pLex
//
// pDense = calibratedConfidence(denseRaw, μ, σ): the corpus-relative dense
//   confidence; undefined on a sub-MIN_BG_SAMPLE / pre-stats index.
// pLex = bm25_norm ∈ [0,1]: the TM2C2 bound-normalized lexical score, already
//   self-calibrated (1.0 = perfect exact match of the query terms) — needs no
//   background fit, so it is honest at any corpus size.
//
// LINEAR, not noisy-OR. Noisy-OR saturates (a whole topical cluster reads ≥80%)
// and any bm25_norm=1.0 fuzzy false-match pins it to 100% (a junk hit on a rare
// high-IDF token reads "perfect"). The linear blend lets a low dense confidence
// VETO a lexical false-positive, so junk fuzzy hits fall to ~(1−α).
//
// Returns null when there is NO calibrated dense arm — the caller then shows the
// ordinal rank instead, never a miscalibrated number. The null check is FIRST so
// a small corpus shows rank-only UNIFORMLY (lexical-only rows included), not a
// confusing mix of % and rank in one list.
export function matchStrength(
    pDense: number | undefined,
    pLex: number,
    alpha: number,
    lexicalOnly = false,
): number | null {
    if (pDense == null) return null;   // uncalibrated corpus → caller shows rank
    // A lexical-only chunk's dense vector is just its title (often a bare date);
    // its cosine is not a meaningful semantic match, so the lexical arm is the
    // whole story. (Only reached on a calibrated corpus — see the null gate.)
    const raw = lexicalOnly ? pLex : alpha * pDense + (1 - alpha) * pLex;
    return raw < 0 ? 0 : raw > 1 ? 1 : raw;   // [0,1] guard against a bm25_norm > 1 edge
}

// Uniform fixed-size reservoir (Vitter's Algorithm R) over a stream of vectors.
// Indexing commits files newest-first (recency-ordered), so a naive "keep the
// first K" sample would be biased toward recently-edited notes; reservoir
// sampling retains each vector with equal probability regardless of arrival
// order, at O(cap) memory. Vectors are COPIED on retention so the sample can't
// pin a larger shared embed-batch buffer alive past the index pass.
export class VecReservoir {
    readonly sample: Float32Array[] = [];
    private seen = 0;

    constructor(
        private readonly cap: number,
        private readonly rand: () => number = Math.random,
    ) {}

    add(v: Float32Array): void {
        this.seen++;
        if (this.sample.length < this.cap) {
            this.sample.push(new Float32Array(v));
            return;
        }
        const j = (this.rand() * this.seen) | 0;   // 0 .. seen-1
        if (j < this.cap) this.sample[j] = new Float32Array(v);
    }
}
