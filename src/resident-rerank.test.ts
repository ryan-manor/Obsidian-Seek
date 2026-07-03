import { describe, it, expect } from 'vitest';
import { quantizeInt8, dequantizeInt8, type QuantVec } from './quant';
import { buildResidentRerankBlock, alignCandidate } from './search';
import { cosineScores } from './ranker';
import type { ChunkMeta } from './types';

// Deterministic LCG (matches binary.test.ts) — reproducible, no Math.random.
function lcg(seed: number) {
    let s = seed >>> 0;
    return () => {
        s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
        return s / 4294967296;
    };
}

const DIM = 384; // shipped granite-r2

// A unit-L2 fp32 vector, like the embedder produces (transformers.js normalize).
function unitVec(rnd: () => number): Float32Array {
    const v = new Float32Array(DIM);
    let norm = 0;
    for (let i = 0; i < DIM; i++) { v[i] = rnd() * 2 - 1; norm += v[i] * v[i]; }
    norm = Math.sqrt(norm) || 1;
    for (let i = 0; i < DIM; i++) v[i] /= norm;
    return v;
}

// Exactly what getEmbeddingsByIds returns for a stored record (index-store.ts).
function idbDequant(rec: QuantVec): Float32Array {
    return dequantizeInt8(rec.q, rec.s);
}

describe('buildResidentRerankBlock — resident dequant is bit-identical to the IDB path', () => {
    it('dequantizes each row byte-for-byte identically to getEmbeddingsByIds', () => {
        const rnd = lcg(11);
        const ids: string[] = [];
        const recs: QuantVec[] = [];
        const embById = new Map<string, QuantVec>();
        for (let i = 0; i < 600; i++) {
            const id = `c${i}`;
            const qv = quantizeInt8(unitVec(rnd));
            ids.push(id); recs.push(qv); embById.set(id, qv);
        }
        const block = buildResidentRerankBlock(ids, embById);
        expect(block).not.toBeNull();
        const { int8, scales, embDim } = block!;
        expect(embDim).toBe(DIM);
        expect(int8.length).toBe(ids.length * DIM);
        expect(scales.length).toBe(ids.length);
        expect(scales).toBeInstanceOf(Float64Array); // Float64, not Float32 — see below

        for (let j = 0; j < ids.length; j++) {
            const resident = dequantizeInt8(int8.subarray(j * embDim, (j + 1) * embDim), scales[j]);
            const idb = idbDequant(recs[j]);
            expect(resident.length).toBe(idb.length);
            for (let d = 0; d < DIM; d++) expect(resident[d]).toBe(idb[d]); // exact Float32 bits
        }
    });

    it('produces cosine scores identical to the IDB path for an arbitrary candidate subset', () => {
        const rnd = lcg(22);
        const ids: string[] = [];
        const recs: QuantVec[] = [];
        const embById = new Map<string, QuantVec>();
        for (let i = 0; i < 400; i++) {
            const id = `c${i}`;
            const qv = quantizeInt8(unitVec(rnd));
            ids.push(id); recs.push(qv); embById.set(id, qv);
        }
        const block = buildResidentRerankBlock(ids, embById)!;
        const { int8, scales, embDim } = block;
        const queryVec = unitVec(rnd);

        // A candidate union in arbitrary (Set-iteration-like) order.
        const candidateIndices = [7, 3, 399, 0, 128, 256, 42, 100, 5, 6];
        const residentVecs = candidateIndices.map(idx =>
            dequantizeInt8(int8.subarray(idx * embDim, (idx + 1) * embDim), scales[idx]));
        const idbVecs = candidateIndices.map(idx => idbDequant(recs[idx]));

        const a = cosineScores(queryVec, residentVecs);
        const b = cosineScores(queryVec, idbVecs);
        expect(Array.from(a)).toEqual(Array.from(b));
    });

    it('aligns block row j to orderedIds[j] regardless of map insertion order', () => {
        const rnd = lcg(33);
        const ids = ['z', 'a', 'm', 'q', 'b'];
        const vecs = ids.map(() => quantizeInt8(unitVec(rnd)));
        // Insert into the map in a DIFFERENT (reversed) order than orderedIds.
        const embById = new Map<string, QuantVec>();
        for (let i = ids.length - 1; i >= 0; i--) embById.set(ids[i], vecs[i]);

        const { int8, scales, embDim } = buildResidentRerankBlock(ids, embById)!;
        for (let j = 0; j < ids.length; j++) {
            const resident = dequantizeInt8(int8.subarray(j * embDim, (j + 1) * embDim), scales[j]);
            const expected = idbDequant(embById.get(ids[j])!); // row j must be orderedIds[j]'s vec
            for (let d = 0; d < DIM; d++) expect(resident[d]).toBe(expected[d]);
        }
    });

    it('falls back (null) when a frame row has no embedding sibling', () => {
        const embById = new Map<string, QuantVec>();
        embById.set('a', quantizeInt8(unitVec(lcg(1))));
        // 'b' deliberately absent — half-migrated/corrupted store
        expect(buildResidentRerankBlock(['a', 'b'], embById)).toBeNull();
    });

    it('falls back (null) on a dimension mismatch', () => {
        const embById = new Map<string, QuantVec>();
        embById.set('a', quantizeInt8(unitVec(lcg(2))));
        embById.set('b', quantizeInt8(new Float32Array(128).fill(0.05))); // wrong dim
        expect(buildResidentRerankBlock(['a', 'b'], embById)).toBeNull();
    });

    it('falls back (null) on empty ids or empty embeddings', () => {
        expect(buildResidentRerankBlock([], new Map())).toBeNull();
        expect(buildResidentRerankBlock(['a'], new Map())).toBeNull();
    });

    // Why scales MUST be Float64: s = max|vᵢ|/127 is a float64; the IDB path
    // dequantizes with that float64. Storing s as Float32 (Math.fround) rounds
    // it and shifts at least one dequantized component on real vectors — which
    // would silently break the bit-identical guarantee. This test fails loudly
    // if anyone "optimizes" residentScales to a Float32Array.
    it('demonstrates a Float32 scale would diverge from the IDB dequant (Float64 is load-bearing)', () => {
        const rnd = lcg(7);
        let diverged = false;
        for (let i = 0; i < 3000 && !diverged; i++) {
            const qv = quantizeInt8(unitVec(rnd));
            const idb = dequantizeInt8(qv.q, qv.s);          // float64 scale (resident & IDB)
            const f32 = dequantizeInt8(qv.q, Math.fround(qv.s)); // what a Float32Array would hold
            for (let d = 0; d < DIM; d++) if (f32[d] !== idb[d]) { diverged = true; break; }
        }
        expect(diverged).toBe(true);
    });
});

// Minimal chunk factory — only the fields alignCandidate reads/copies matter.
// Mirrors ranker.test.ts's chunk() helper.
function metaChunk(note_path: string, opts: { lexicalOnly?: boolean } = {}): ChunkMeta {
    return {
        chunk_id: note_path,
        title: note_path.replace(/\.md$/, ''),
        note_path,
        heading_path: [],
        metadata: { tags: [], aliases: [], created: null, modified: null, properties: {} },
        start_line: 1,
        end_line: 1,
        ...(opts.lexicalOnly && { lexicalOnly: true }),
    };
}

// ── alignCandidate — the S2 align loop's degrade-not-drop decision ──────────
// A candidate whose fp32 row is missing/mismatched (getEmbeddingsByIds' `null`,
// or a half-migrated/corrupted store) used to be dropped outright, even when
// BM25 ranked it first. It must instead degrade to the SAME lexical-only floor
// ranker.ts applies to body-less title-only chunks — see the callsite comment
// in search.ts.
describe('alignCandidate', () => {
    it('passes a chunk with a valid, dimension-matching fp32 row through unchanged', () => {
        const ch = metaChunk('good.md');
        const v = new Float32Array(384).fill(0.1);
        const result = alignCandidate(ch, v, 384);
        expect(result).not.toBeNull();
        expect(result!.missingFp32).toBe(false);
        expect(result!.chunk).toBe(ch); // same reference — no copy when the row is fine
    });

    it('degrades (not drops) a candidate whose fp32 row is null', () => {
        const ch = metaChunk('missing-vec.md');
        const result = alignCandidate(ch, null, 384);
        expect(result).not.toBeNull();               // NOT dropped
        expect(result!.missingFp32).toBe(true);
        expect(result!.chunk.lexicalOnly).toBe(true); // routed to the degradation floor
        expect(result!.chunk.note_path).toBe('missing-vec.md'); // still the real candidate
    });

    it('degrades a candidate whose fp32 row has the wrong dimension (corruption/half-migration)', () => {
        const ch = metaChunk('bad-dim.md');
        const v = new Float32Array(128).fill(0.1); // wrong dim vs. a 384-d query
        const result = alignCandidate(ch, v, 384);
        expect(result!.missingFp32).toBe(true);
        expect(result!.chunk.lexicalOnly).toBe(true);
    });

    it('never mutates the caller\'s chunk object when degrading', () => {
        const ch = metaChunk('shared.md'); // as if shared with orderedChunks across queries
        alignCandidate(ch, null, 384);
        expect(ch.lexicalOnly).toBeUndefined(); // the original must stay untouched
    });

    it('preserves an already-lexicalOnly chunk\'s flag when its row is also missing', () => {
        const ch = metaChunk('title-only.md', { lexicalOnly: true });
        const result = alignCandidate(ch, null, 384);
        expect(result!.chunk.lexicalOnly).toBe(true);
    });

    it('drops the candidate only when there is no chunk metadata at all', () => {
        expect(alignCandidate(undefined, new Float32Array(384), 384)).toBeNull();
        expect(alignCandidate(null, new Float32Array(384), 384)).toBeNull();
    });
});
