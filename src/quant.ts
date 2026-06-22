// Int8 scalar quantization (SQ8) for the fp32 exact-rerank tier.
//
// Why: the `embeddings` store held one Float32Array(d) per chunk (1536 B at
// d=384) purely to feed the stage-2 cosine rerank. That fp32 precision is
// unnecessary — the 2026-06-01 quant grid (see memory seek-quant-grid-reopen)
// measured int8 storage at ≤0.003 NDCG@10 cost on granite-r2's own vectors
// over the vault, a free 4× index shrink (1536 B → 388 B/vec) and ~4× less
// stage-2 IDB read (selectFetchMs). Binary (sign-bit) was 0.03–0.07 worse, so
// it stays the *candidate* tier (binary.ts) and int8 becomes the *rerank* tier.
//
// Why it's near-lossless here SPECIFICALLY: the stored vectors are unit-L2
// (iframe-runner sliceAndRenormalize + transformers.js normalize:true). A
// per-vector max-abs scale then resolves each component to ~1/254 of that
// vector's own span — finer than the noise floor of a 384-d cosine score.
// We do NOT use a global fixed scale: per-vector max-abs adapts to each
// vector's actual range (granite components cluster well under 1.0) for a
// 4-byte/vec cost, strictly tighter than assuming [-1, 1].
//
// Asymmetric at query time: only the STORED doc vectors are int8. The query
// vector stays fp32 (one fresh vector per search, negligible) and is cosined
// against the dequantized docs — same pattern as the binary tier's asymmetric
// float-query · sign-doc score.

export interface QuantVec {
    q: Int8Array;  // d components, each round(vᵢ / s) clamped to [-127, 127]
    s: number;     // dequant scale: vᵢ ≈ qᵢ · s  (s = max|vᵢ| / 127)
}

// Quantize a (unit-L2) fp32 vector to int8 + a per-vector scale.
//
// scale = max|vᵢ| / 127 maps the largest-magnitude component to ±127 and
// everything else proportionally; round-to-nearest keeps the quantization
// error symmetric (±s/2 per component). Int8Array stores the rounded values
// directly — JS coerces out-of-range on assignment, but round() of v/s can
// only reach ±127 by construction, so no explicit clamp is needed except the
// degenerate all-zero guard below.
export function quantizeInt8(vec: Float32Array): QuantVec {
    const dim = vec.length;
    let maxAbs = 0;
    for (let i = 0; i < dim; i++) {
        const a = vec[i] < 0 ? -vec[i] : vec[i];
        if (a > maxAbs) maxAbs = a;
    }
    const q = new Int8Array(dim);
    // Degenerate: an all-zero vector (never produced by a unit-L2 embedder, but
    // guard against div-by-zero / NaN poisoning the store). s=0 → dequant to 0.
    if (maxAbs === 0) return { q, s: 0 };
    const s = maxAbs / 127;
    const inv = 1 / s;
    for (let i = 0; i < dim; i++) {
        // Math.round(0.5) → 1 (toward +∞); for symmetric-magnitude embedding
        // components the half-integer tie bias is negligible. The result is in
        // [-127, 127] because |vᵢ| ≤ maxAbs = 127·s.
        q[i] = Math.round(vec[i] * inv);
    }
    return { q, s };
}

// Dequantize back to fp32 for the cosine rerank. The result is NOT exactly
// unit-L2 anymore (per-component error ±s/2 perturbs the norm by ~0.1%), but
// cosineScores treats inputs as unit-norm and the drift is far below the
// nDCG noise floor — re-normalizing here would cost a sqrt+divide per
// candidate for no measurable relevance gain, so we don't.
export function dequantizeInt8(q: Int8Array, s: number): Float32Array {
    const dim = q.length;
    const out = new Float32Array(dim);
    for (let i = 0; i < dim; i++) out[i] = q[i] * s;
    return out;
}
