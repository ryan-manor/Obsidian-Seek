import { describe, it, expect } from 'vitest';
import { selectBucket, selectIndexBucket, selectQueryBucket, SEQ_BUCKETS, QUERY_SEQ_BUCKETS } from './iframe-runner';

describe('selectQueryBucket (query path ladder) — buckets by EXACT token count', () => {
    it('floors short queries at 8, not 32 (the whole point); smallest bucket >= tokens', () => {
        expect(selectQueryBucket(1)).toBe(8);
        expect(selectQueryBucket(3)).toBe(8);     // live p50 query
        expect(selectQueryBucket(8)).toBe(8);     // exact fit, no padding, no truncation
        expect(selectQueryBucket(9)).toBe(16);
        expect(selectQueryBucket(16)).toBe(16);
        expect(selectQueryBucket(17)).toBe(32);
    });

    it('caps at 128 — no query/dense-relevance case past it; longer truncates dense side only', () => {
        expect(QUERY_SEQ_BUCKETS).not.toContain(192);
        expect(QUERY_SEQ_BUCKETS).not.toContain(256);
        expect(QUERY_SEQ_BUCKETS).not.toContain(384);
        expect(QUERY_SEQ_BUCKETS).not.toContain(512);
        expect(selectQueryBucket(128)).toBe(128);
        expect(selectQueryBucket(129)).toBe(128); // cap: >128 truncates (BM25 unaffected)
        expect(selectQueryBucket(5000)).toBe(128);
    });

    it('leaves the index ladder (selectBucket, char-based) untouched — 32 floor, 512 cap', () => {
        const chars = (tokens: number) => Math.ceil(tokens * 4.5);
        expect(selectBucket(chars(3))).toBe(32);          // index floor unchanged
        expect(SEQ_BUCKETS).toContain(256);
        expect(SEQ_BUCKETS).toContain(512);
        expect(selectBucket(chars(5000))).toBe(512);      // index cap unchanged
    });
});

describe('selectIndexBucket (WS2.3) — index path buckets by EXACT token count', () => {
    it('smallest rung >= the real count: no under-bucketing, no padding waste', () => {
        expect(selectIndexBucket(1)).toBe(32);    // index floor stays 32 (chunks, not queries)
        expect(selectIndexBucket(32)).toBe(32);
        expect(selectIndexBucket(33)).toBe(48);   // chars/4.5 would have mis-routed dense text here
        expect(selectIndexBucket(192)).toBe(192);
        expect(selectIndexBucket(193)).toBe(256);
        expect(selectIndexBucket(512)).toBe(512); // exact fit at the cap: zero truncation
    });

    it('caps at 512 — only the oversize-title pathology ever arrives above it', () => {
        expect(selectIndexBucket(513)).toBe(512);
        expect(selectIndexBucket(5000)).toBe(512);
    });

    it('agrees with the char estimator wherever the estimate happens to be right', () => {
        // For text at exactly 4.5 chars/token the two selectors must match —
        // the WS2.3 change is exactness, not a different ladder.
        for (const tokens of [10, 40, 100, 300, 512]) {
            expect(selectIndexBucket(tokens)).toBe(selectBucket(Math.ceil(tokens * 4.5)));
        }
    });
});
