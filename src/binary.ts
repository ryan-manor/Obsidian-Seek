// Sign-bit binary index + asymmetric scoring for stage-1 candidate generation.
//
// The validated form (Seek Retrieval Relevance & Query.md §Two-Stage ANN → Rerank):
//   - Pack each unit-L2 fp32 chunk vector to a Uint8Array of ceil(d/8) bytes,
//     one bit per dim, set when the dim is ≥ 0. ~32× smaller than fp32.
//   - At query time, compute `float_query · sign(doc_vec)` (asymmetric:
//     full-precision query, 1-bit doc), take top-BINARY_TOP_N as the dense arm
//     of the stage-1 union (search.ts).
//   - Recall is NOT embedder-agnostic — that earlier claim was an n=2
//     extrapolation, falsified for the shipped model. The Phase-9 bake-off saw
//     N≥200 recover the fp32 ceiling exactly on 512-d Gemma and 320-d F2LLM,
//     but granite-r2 is 384-d and CLS-pooled, hence anisotropic: mean pairwise
//     cos ≈ 0.76, ~30–60 dead sign-bits, only ~265–290 effective bits of 384
//     (binary_recall.py, ~/eval, 614 q). On granite the binary arm
//     ALONE recovers ~95.5% of dense-reachable golds at N=200; parity with
//     exact dense isn't reached until ~N=800.
//   - This costs ZERO end-to-end nDCG@10 (Δceiling +0.000 at every N from 100
//     to 800, all 3 corpora — binary_n_sweep.py) because the bm25-top-100 +
//     recency-top-50 arms backstop the dense arm: union gold-in-pool is 0.991
//     at N=200, and the residual misses don't reach top-10. So N=200 is correct
//     because the UNION backstops the gap, NOT because sign-bit recovers the
//     ceiling at 200. A bump to N=400 buys +0.000 nDCG@10 for +59% fp32 fetch —
//     not worth it; the current value already sits above the knee for the
//     backstopped path.
//   - Load-bearing assumption to watch: a gold that is dense-only (semantically
//     findable but lexically invisible to BM25) is exposed to the ~4.5% binary
//     miss with no backstop. None exist at top-10 in the current eval; re-run
//     binary_recall.py on any reindex or embedder swap to re-confirm.
//   - DO NOT binarize the query side too. Phase-9 of the bake-off tested that
//     symmetric form and it underperformed; the architectural form is rung-4
//     asymmetric for recall, fp32 exact rerank for precision.
//
// Cosine vs dot-product subtlety: this works only because the chunk vectors
// are unit-L2 normalized (transformers.js `normalize:true` does this — see
// ranker.ts:122 "Inputs must already be unit-normalized"). The sign vector
// is not unit-norm, so the asymmetric score is a *ranking* signal, not a
// calibrated similarity — magnitudes are not comparable across queries.
// That's fine for candidate gen: only the rank order matters.

import { selectTopNIndices } from './select';

// Pack a unit-L2 fp32 vector to its sign-bit representation. One bit per dim,
// LSB-first within each byte (`bit i` of byte `b` corresponds to dim `b*8+i`).
// Choosing LSB-first matches the bitmask order used in scoreAsymmetric.
// Negative values map to 0, non-negative to 1 (matches numpy `>= 0`, the
// convention used by `benchmarks/binary_candidate.py` in the eval harness).
export function packSignBits(vec: Float32Array): Uint8Array {
    const dim = vec.length;
    const bytes = (dim + 7) >> 3;
    const out = new Uint8Array(bytes);
    for (let i = 0; i < dim; i++) {
        if (vec[i] >= 0) out[i >> 3] |= 1 << (i & 7);
    }
    return out;
}

// Concatenate per-chunk packed buffers into one contiguous Uint8Array for
// cache-friendly scoring. The score loop reads bytesPerVec sequential bytes
// per doc; a single backing buffer keeps that hot path tight (vs. an array
// of small Uint8Arrays where each scored doc is a separate heap object).
export function concatPacked(packed: Uint8Array[], bytesPerVec: number): Uint8Array {
    const n = packed.length;
    const out = new Uint8Array(n * bytesPerVec);
    for (let i = 0; i < n; i++) {
        if (packed[i].length !== bytesPerVec) {
            throw new Error(`packed[${i}] is ${packed[i].length} bytes, expected ${bytesPerVec}`);
        }
        out.set(packed[i], i * bytesPerVec);
    }
    return out;
}

// Float-query · binary-doc asymmetric dot product across N docs.
//
// Math: for each doc with sign vector s where s[i] ∈ {-1, +1},
//   score = Σᵢ q[i] · s[i]
//         = (Σᵢ : sᵢ=+1 q[i]) − (Σᵢ : sᵢ=-1 q[i])
//         = 2 · (Σᵢ : sᵢ=+1 q[i]) − Σᵢ q[i]
// so per doc we need `pos` = Σ q[i] over the dims whose sign bit is set.
//
// Implementation: per-query subset-sum LUT (de-branched 2026-06-09 — see audit
// §"Does the binary stage save query time"). The query is CONSTANT across all N
// docs, so instead of re-testing 8 bits/byte per doc (the old branchy unroll),
// precompute once per query a table T[b·256 + v] = Σ q[b·8+k] over the bits k
// set in byte-value v. Then `pos` for a doc is bytesPerVec branchless table
// lookups. The table is built in one add/entry via the lowest-set-bit
// recurrence (T[v] = T[v with low bit cleared] + q[low bit]).
//
// Why: the old branchy loop read 32× less memory than fp32 but ran ~5× SLOWER
// per vector (1.86 µs/doc vs 0.368 µs/vec for fp32 cosine) — branch mispredicts,
// not bandwidth, were the bottleneck, and V8 won't vectorize a data-dependent
// branch. The LUT is branchless: measured ~4–6× faster, bringing the binary
// scan to parity with fp32 cosine while keeping the 1-bit residency win (the
// reason the index exists — fp32 lives in IDB, not RAM; see search.ts S2).
// T is bytesPerVec·256 Float64 (~96 KB for 384-d) — rebuilt per query, fits L2.
//
// `packed` is the concatenated buffer from concatPacked, length n*bytesPerVec.
// `bytesPerVec` must equal ceil(query.length / 8); any unused high bits in
// the last byte must be 0 (packSignBits guarantees this — they're never set).
export function scoreAsymmetric(
    query: Float32Array,
    packed: Uint8Array,
    n: number,
    bytesPerVec: number,
): Float64Array {
    const dim = query.length;
    if (bytesPerVec !== (dim + 7) >> 3) {
        throw new Error(`bytesPerVec ${bytesPerVec} doesn't match query dim ${dim}`);
    }
    if (packed.length !== n * bytesPerVec) {
        throw new Error(`packed length ${packed.length} ≠ n*bytesPerVec ${n * bytesPerVec}`);
    }

    let sumQ = 0;
    for (let i = 0; i < dim; i++) sumQ += query[i];

    // Build the per-byte subset-sum table once. base = b<<8 is byte position b's
    // 256-entry block; for value v, clear its lowest set bit (v & (v-1)) to reach
    // an already-filled smaller subset and add back the one weight that bit maps
    // to. Dims past `dim` in the final partial byte contribute 0 (packSignBits
    // never sets those bits, but the guard keeps the table well-defined).
    const T = new Float64Array(bytesPerVec << 8);
    for (let b = 0; b < bytesPerVec; b++) {
        const base = b << 8;
        const qOff = b << 3;
        for (let v = 1; v < 256; v++) {
            const low = v & -v;                    // isolate lowest set bit (a power of 2)
            const d = qOff + (31 - Math.clz32(low)); // dim index for that bit
            const w = d < dim ? query[d] : 0;
            T[base + v] = T[base + (v & (v - 1))] + w;
        }
    }

    const scores = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        const off = i * bytesPerVec;
        let pos = 0;
        for (let b = 0; b < bytesPerVec; b++) {
            pos += T[(b << 8) + packed[off + b]];  // branchless: one table lookup/byte
        }
        scores[i] = 2 * pos - sumQ;
    }
    return scores;
}

// Reference implementation: the original branchy bit-unroll. Kept ONLY as the
// parity oracle for binary.test.ts (it pins scoreAsymmetric's LUT to identical
// scores) and the before/after benchmark. Not used in the live search path.
// Do not "optimize" — its value is being an obviously-correct independent impl.
export function scoreAsymmetricBranchy(
    query: Float32Array,
    packed: Uint8Array,
    n: number,
    bytesPerVec: number,
): Float64Array {
    const dim = query.length;
    let sumQ = 0;
    for (let i = 0; i < dim; i++) sumQ += query[i];

    const scores = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        const off = i * bytesPerVec;
        let pos = 0;
        for (let b = 0; b < bytesPerVec; b++) {
            const byte = packed[off + b];
            if (byte === 0) continue; // skip all-negative dim blocks
            const qOff = b * 8;
            if (byte & 0x01) pos += query[qOff];
            if (byte & 0x02) pos += query[qOff + 1];
            if (byte & 0x04) pos += query[qOff + 2];
            if (byte & 0x08) pos += query[qOff + 3];
            if (byte & 0x10) pos += query[qOff + 4];
            if (byte & 0x20) pos += query[qOff + 5];
            if (byte & 0x40) pos += query[qOff + 6];
            if (byte & 0x80) pos += query[qOff + 7];
        }
        scores[i] = 2 * pos - sumQ;
    }
    return scores;
}

// Top-N indices by score (descending). Returns indices into the `scores`
// array, ordered score-descending with ascending-index tie-breaks. Used by
// stage-1 candidate gen to pick which doc IDs feed stage-2 rerank.
//
// Delegates to the bounded-heap `selectTopNIndices` (select.ts): same result
// as the old full-length-sort-then-slice — same members AND same order — but
// without allocating/sorting a corpus-length index array each keystroke. The
// binary score loop above still dominates wall-clock; this just sheds the
// short-lived full-length allocations (a mobile GC-jank source).
//
// `mask` (optional): when present, only indices `i` with `mask[i] === true`
// are eligible. This is how inline-filter pre-filtering is applied without
// shrinking the corpus the binary/BM25 indexes are built over — the index
// scores the full set, selection picks the top-N among the matching subset.
export function topNIndices(scores: Float64Array, n: number, mask?: boolean[] | null): number[] {
    const m = mask;
    return selectTopNIndices(scores.length, n, i => scores[i], m ? i => m[i] : null);
}

// Stage-1 binary candidate generation in one call: asymmetric score over the
// whole packed buffer, then top-N selection. This is the SINGLE function both
// the main-thread path and the off-thread worker (binary-worker.ts) invoke, so
// running it in a worker is a pure relocation — same inputs, same arithmetic,
// bit-identical output. (Keeping it here, importable by the worker bundle,
// rather than inlining the two steps at each call site, is what guarantees the
// worker and the synchronous fallback can never diverge.)
export function binaryCandidates(
    queryVec: Float32Array,
    packed: Uint8Array,
    n: number,
    bytesPerVec: number,
    topN: number,
    mask: boolean[] | null,
): number[] {
    return topNIndices(scoreAsymmetric(queryVec, packed, n, bytesPerVec), topN, mask);
}
