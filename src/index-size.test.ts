// Phase-0 index-size accounting. These pin the two pieces that carry the logic:
// the per-store byte RULE (sizeOfRow) and the RENDERER/verdict (formatIndexSizeReport).
// The IDB cursor walk that feeds them (index-store.measureSizes) is the proven
// listAllBinary idiom and isn't re-tested here — only the value-shape arithmetic is.

import { describe, it, expect } from 'vitest';
import {
    sizeOfRow,
    slackBytes,
    formatIndexSizeReport,
    type IndexSizeReport,
    type StoreSizeRow,
} from './index-size';

describe('sizeOfRow', () => {
    it('weighs a packed binary row by byteLength', () => {
        expect(sizeOfRow('bytes', new Uint8Array(48))).toBe(48);
        expect(sizeOfRow('bytes', new ArrayBuffer(16))).toBe(16);
    });

    it('weighs a quantvec as its int8 buffer plus the fp64 scale', () => {
        const v = { q: new Int8Array(384), s: 0.0123 };
        expect(sizeOfRow('quantvec', v)).toBe(384 + 8);
    });

    it('weighs utf8 by encoded byte length, not String.length', () => {
        expect(sizeOfRow('utf8', 'hello')).toBe(5);
        // a 3-byte UTF-8 char counts as 3, where String.length would say 1
        expect(sizeOfRow('utf8', '日')).toBe(3);
    });

    it('weighs json by its stringified UTF-8 bytes', () => {
        const obj = { note_path: 'A.md', chunk_ids: ['a', 'b'] };
        expect(sizeOfRow('json', obj)).toBe(new TextEncoder().encode(JSON.stringify(obj)).length);
    });

    it('weighs a bm25 record by its json blob plus the tiny stamp (string today)', () => {
        const rec = { json: 'x'.repeat(1000), stamp: { v: 11, n: 7503 } };
        const expected = 1000 + new TextEncoder().encode(JSON.stringify(rec.stamp)).length;
        expect(sizeOfRow('bm25', rec)).toBe(expected);
    });

    it('weighs a GZIPPED bm25 blob (Uint8Array) correctly — forward-compat with Phase 1', () => {
        const rec = { json: new Uint8Array(2048), stamp: null };
        // gzipped blob byteLength + JSON.stringify(null) === 'null' (4 bytes)
        expect(sizeOfRow('bm25', rec)).toBe(2048 + 4);
    });

    it('returns 0 for null / malformed rows instead of throwing', () => {
        expect(sizeOfRow('bytes', null)).toBe(0);
        expect(sizeOfRow('quantvec', {})).toBe(8); // missing q → just the scale
        expect(sizeOfRow('utf8', 12345)).toBe(0);
        expect(sizeOfRow('bm25', { stamp: 1 })).toBe(utf8('1')); // no json
    });
});

function utf8(s: string): number {
    return new TextEncoder().encode(s).length;
}

function makeReport(over: Partial<IndexSizeReport> = {}): IndexSizeReport {
    const stores: StoreSizeRow[] = [
        { store: 'bm25', label: 'BM25 inverted index', rows: 1, bytes: 20_000_000 },
        { store: 'chunk_body', label: 'chunk bodies', rows: 7503, bytes: 6_000_000 },
        { store: 'embeddings', label: 'int8 vectors', rows: 7503, bytes: 2_900_000 },
        { store: 'binary', label: 'binary sign-bits', rows: 7503, bytes: 400_000 },
    ];
    const logicalBytes = stores.reduce((s, x) => s + x.bytes, 0);
    return {
        stores,
        logicalBytes,
        indexedDbBytes: 60_000_000,
        cachesBytes: 240_000_000,
        originUsageBytes: 300_000_000,
        quotaBytes: 2_000_000_000,
        ...over,
    };
}

describe('slackBytes', () => {
    it('is physical minus logical, floored at 0', () => {
        const r = makeReport();
        expect(slackBytes(r)).toBe(60_000_000 - r.logicalBytes);
    });

    it('floors at 0 when logical exceeds the reported physical (estimate lag)', () => {
        expect(slackBytes(makeReport({ indexedDbBytes: 1_000 }))).toBe(0);
    });

    it('is null when no physical number is available', () => {
        expect(slackBytes(makeReport({ indexedDbBytes: null }))).toBeNull();
    });
});

describe('formatIndexSizeReport', () => {
    it('lists stores largest-first and shows the logical total', () => {
        const out = formatIndexSizeReport(makeReport());
        const bm25Line = out.indexOf('BM25 inverted index');
        const bodyLine = out.indexOf('chunk bodies');
        expect(bm25Line).toBeGreaterThan(-1);
        expect(bm25Line).toBeLessThan(bodyLine); // bm25 (20MB) before bodies (6MB)
        expect(out).toContain('logical total');
    });

    it('reports the physical/slack split when usageDetails is present', () => {
        const out = formatIndexSizeReport(makeReport());
        expect(out).toContain('IndexedDB physical:');
        expect(out).toContain('% LevelDB slack)');
    });

    it('verdict names both the slack and the BM25 blob when physical is present', () => {
        const out = formatIndexSizeReport(makeReport({ indexedDbBytes: 90_000_000 }));
        expect(out).toMatch(/Verdict: .* LevelDB slack \(BM25 blob is .*\)/);
    });

    it('verdict frames desktop slack as harmless + not actionable (no compaction/gzip lever)', () => {
        const out = formatIndexSizeReport(makeReport({ indexedDbBytes: 90_000_000 }));
        expect(out).toContain('not web-reclaimable');
        expect(out).toContain('does not shed physical SSTs');
        // The retired levers must not reappear as recommendations.
        expect(out).not.toMatch(/Phase 1|Phase 2|compaction\/reindex|gzip the blob/);
    });

    it('degrades to a logical-only verdict when physical is unavailable', () => {
        const out = formatIndexSizeReport(makeReport({ indexedDbBytes: null, cachesBytes: null }));
        expect(out).toContain('unavailable');
        expect(out).toMatch(/Verdict: BM25 blob is .* of .* logical/);
    });
});
