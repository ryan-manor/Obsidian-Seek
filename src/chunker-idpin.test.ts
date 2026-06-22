// Investment #4 (downgraded) — chunker id-pin.
//
// The original plan's self-hashing CHUNKER_VERSION (esbuild source-hash + build
// assert) was a NO-GO: its hash list omitted prop-normalize.ts (which feeds
// buildDenseSuffix → chunk_id), so it would false-green on its own target, and it
// mis-bucketed the v11 case. This is the SAFE half — a semantic anchor that proves
// "chunk ids did not move." When chunking changes the bytes that feed chunkIdFor
// (split boundaries, salt format, the embedded title/content shape, OR the
// frontmatter-derived dense suffix via prop-normalize), this goes RED and the dev
// consciously decides whether to bump CHUNKER_VERSION (and DB_VERSION).
//
// Pinned hash strings are characterization values: if an intentional chunker change
// moves them, re-pin them IN THE SAME COMMIT that bumps CHUNKER_VERSION — never
// silently. Note the FIXTURE carries frontmatter properties precisely so a change to
// prop-normalize / buildDenseSuffix (the surface the esbuild-hash design missed)
// trips this test.

import { describe, it, expect } from 'vitest';
import { MarkdownChunker, chunkIdFor } from './chunker';

const chunker = new MarkdownChunker();

// Frontmatter (text values fold into the dense suffix → chunk_id) + a heading + a
// short body. Kept small and deterministic so it produces a stable single chunk.
const FIXTURE = `---
tags: project
status: active
---
# Versioning Robustness

This note pins the chunker id contract so a silent id-move is caught in CI.
`;
const FIXTURE_PATH = 'Notes/Versioning Robustness.md';

describe('chunker id-pin (Investment #4, downgraded)', () => {
    it('chunkContent produces the pinned chunk_id for a fixed fixture (full pipeline incl. prop-normalize)', () => {
        const chunks = chunker.chunkContent(FIXTURE, FIXTURE_PATH);
        expect(chunks.length).toBe(1);
        expect(chunks[0].chunk_id).toBe('1eb905d44ee9d5');
    });

    it('chunkIdFor is a stable pure hash of (path, title, content, suffix)', () => {
        // Pure determinism guard — catches a cyrb53 / salt-format change independent of
        // split logic. No suffix → the pre-dense-suffix id form.
        expect(chunkIdFor('Notes/Foo.md', 'Foo', 'body text')).toBe('1324c39d06d44a');
        // With a dense suffix → the tail enters the id (a delta must re-embed on suffix change).
        expect(chunkIdFor('Notes/Foo.md', 'Foo', 'body text', 'project active')).toBe('1c8096b6b2c2b8');
    });
});
