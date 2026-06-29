// Seek scaling A1 — resident-frame incremental maintenance.
//
// appendFrameRows / tombstoneFrameRows / buildSelectionMask are the frame half of
// applyDelta. They're pure functions over a ResidentFrame so they can be unit-
// tested without the full SearchEngine. The two things they MUST get right:
//   - the binary + metadata tiers grow in lockstep on EVERY add (even when the
//     int8 rerank block is gated off on mobile), or the single-`idx` candidate
//     join skews (GAP 5);
//   - a tombstone is excluded from EVERY selection path — and crucially the
//     filter-only browse path, whose old `!mask ||` short-circuit admitted holes
//     when no inline filter was present (GAP 4). buildSelectionMask returns a
//     DEFINED mask whenever tombstones exist, even with no matcher.

import { describe, it, expect } from 'vitest';
import {
    appendFrameRows, tombstoneFrameRows, buildSelectionMask,
    type ResidentFrame, type DeltaAdd,
} from './search';
import { topNIndices } from './binary';
import type { Chunk, ChunkMeta } from './types';
import type { QuantVec } from './quant';

const BPV = 4;   // bytes per binary vec
const DIM = 8;   // int8 dim

function chunk(id: string, notePath = `${id}.md`): Chunk {
    return {
        chunk_id: id, title: `T-${id}`, content: `body of ${id}`, note_path: notePath,
        heading_path: [], metadata: { tags: [], aliases: [], created: null, modified: null, properties: {} },
        start_line: 0, end_line: 0,
    };
}
function qv(seed: number): QuantVec {
    const q = new Int8Array(DIM);
    for (let i = 0; i < DIM; i++) q[i] = ((seed * 7 + i) % 127) - 63;
    return { q, s: 0.5 + seed * 0.01 };
}
function bin(seed: number): Uint8Array {
    const b = new Uint8Array(BPV);
    for (let i = 0; i < BPV; i++) b[i] = (seed * 31 + i) & 0xff;
    return b;
}
function add(id: string): DeltaAdd { return { chunk: chunk(id), q: qv(id.charCodeAt(0)), bin: bin(id.charCodeAt(0)) }; }

function makeFrame(ids: string[], opts: { resident: boolean }): ResidentFrame {
    const n = ids.length;
    const activePacked = new Uint8Array(n * BPV);
    for (let i = 0; i < n; i++) activePacked.set(bin(ids[i].charCodeAt(0)), i * BPV);
    let residentInt8: Int8Array | null = null;
    let residentScales: Float64Array | null = null;
    if (opts.resident) {
        residentInt8 = new Int8Array(n * DIM);
        residentScales = new Float64Array(n);
        for (let i = 0; i < n; i++) { const v = qv(ids[i].charCodeAt(0)); residentInt8.set(v.q, i * DIM); residentScales[i] = v.s; }
    }
    return {
        orderedChunks: ids.map(id => { const { content, ...m } = chunk(id); void content; return m as ChunkMeta; }),
        orderedIds: [...ids],
        activePacked,
        bytesPerVec: BPV,
        residentInt8,
        residentScales,
        embDim: opts.resident ? DIM : 0,
        validRows: new Array<boolean>(n).fill(true),
        tombstoneCount: 0,
        generation: 1,
    };
}

describe('appendFrameRows', () => {
    it('grows binary + int8 + metadata tiers in lockstep (resident enabled)', () => {
        const f = makeFrame(['a', 'b'], { resident: true });
        appendFrameRows(f, [add('c'), add('d')]);

        expect(f.orderedIds).toEqual(['a', 'b', 'c', 'd']);
        expect(f.orderedChunks.map(c => c.chunk_id)).toEqual(['a', 'b', 'c', 'd']);
        expect(f.validRows).toEqual([true, true, true, true]);
        expect(f.activePacked.length).toBe(4 * BPV);
        expect(f.residentInt8!.length).toBe(4 * DIM);
        expect(f.residentScales!.length).toBe(4);
        // Appended rows hold the new chunks' bytes verbatim (byte-identical tiers).
        expect([...f.activePacked.subarray(2 * BPV, 3 * BPV)]).toEqual([...bin('c'.charCodeAt(0))]);
        expect([...f.residentInt8!.subarray(2 * DIM, 3 * DIM)]).toEqual([...qv('c'.charCodeAt(0)).q]);
        expect(f.residentScales![2]).toBe(qv('c'.charCodeAt(0)).s);
        // Pre-existing rows are preserved across the realloc.
        expect([...f.activePacked.subarray(0, BPV)]).toEqual([...bin('a'.charCodeAt(0))]);
        // Frame holds metadata only — body text must NOT leak in.
        expect('content' in (f.orderedChunks[2] as Record<string, unknown>)).toBe(false);
    });

    it('resident-disabled (mobile): still grows binary + metadata, leaves int8 null (GAP 5)', () => {
        const f = makeFrame(['a', 'b'], { resident: false });
        appendFrameRows(f, [add('c')]);

        expect(f.orderedIds).toEqual(['a', 'b', 'c']);
        expect(f.validRows).toEqual([true, true, true]);
        expect(f.activePacked.length).toBe(3 * BPV);   // binary tier grew
        expect(f.residentInt8).toBeNull();             // int8 tier stayed off
        expect(f.residentScales).toBeNull();
    });

    it('empty add list is a no-op', () => {
        const f = makeFrame(['a'], { resident: true });
        const before = f.activePacked;
        appendFrameRows(f, []);
        expect(f.activePacked).toBe(before);
        expect(f.orderedIds).toEqual(['a']);
    });
});

describe('tombstoneFrameRows', () => {
    it('marks rows not-live and counts them; idempotent and bounds-guarded', () => {
        const f = makeFrame(['a', 'b', 'c'], { resident: true });
        tombstoneFrameRows(f, [1]);
        expect(f.validRows).toEqual([true, false, true]);
        expect(f.tombstoneCount).toBe(1);
        // Re-tombstoning the same row, and out-of-range rows, don't double-count.
        tombstoneFrameRows(f, [1, -1, 99]);
        expect(f.tombstoneCount).toBe(1);
    });
});

describe('buildSelectionMask — tombstones excluded from every path', () => {
    it('returns undefined ONLY when fully live AND unfiltered (fast path preserved)', () => {
        const f = makeFrame(['a', 'b'], { resident: true });
        expect(buildSelectionMask(f.orderedChunks, f.validRows, f.tombstoneCount, null)).toBeUndefined();
    });

    it('returns a DEFINED mask excluding the tombstone even with NO filter (GAP 4 guard)', () => {
        const f = makeFrame(['a', 'b', 'c'], { resident: true });
        tombstoneFrameRows(f, [1]);
        const mask = buildSelectionMask(f.orderedChunks, f.validRows, f.tombstoneCount, null);
        expect(mask).toEqual([true, false, true]);   // NOT undefined — the browse-path hole is closed
    });

    it('ANDs liveness with the inline-filter matcher, short-circuiting tombstoned rows', () => {
        const f = makeFrame(['a', 'b', 'c'], { resident: true });
        tombstoneFrameRows(f, [2]);
        let readTombstone = false;
        const matcher = (c: ChunkMeta) => { if (c.chunk_id === 'c') readTombstone = true; return c.chunk_id !== 'b'; };
        const mask = buildSelectionMask(f.orderedChunks, f.validRows, f.tombstoneCount, matcher);
        expect(mask).toEqual([true, false, false]);   // b filtered out, c tombstoned
        expect(readTombstone).toBe(false);            // matcher never read the stale tombstoned row
    });

    it('the mask is honored by the selection arms (tombstoned top-scorer never selected)', () => {
        const f = makeFrame(['a', 'b', 'c'], { resident: true });
        tombstoneFrameRows(f, [1]);                   // row 1 = the would-be winner
        const mask = buildSelectionMask(f.orderedChunks, f.validRows, f.tombstoneCount, null);
        const scores = new Float64Array([0.1, 0.9, 0.5]);   // row 1 has the top score
        expect(topNIndices(scores, 3, mask ?? null)).toEqual([2, 0]);   // 1 excluded despite winning
    });
});
