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

// Link/URL-rich fixture (audit R2: the plain fixture above was blind to
// dense-clean edits — v8 shipped with no red test because nothing here carried
// a wikilink or URL). Covers every construct cleanDenseText rewrites: aliased
// wikilink, path#heading link, markdown link, autolink, asset embed. Pins BOTH
// the chunk_id (a dense-clean rule change moves the cleaned bytes → red) AND
// the v10 link_terms reclamation contract.
const LINK_FIXTURE = `---
tags: clippings
source: theverge
---
# Verge Clipping

Read the original at [The Verge](https://www.theverge.com/tech/some-post) — met with [[Jordan Rivera|Jordan]] to discuss it. More context in [[Notes/Projects/Seek#Status]] and the raw feed <https://feeds.example.com/rss>.

![[Pasted image 20260702.png]]
`;
const LINK_FIXTURE_PATH = 'Clippings/Verge Clipping.md';

describe('chunker id-pin (Investment #4, downgraded)', () => {
    it('chunkContent produces the pinned chunk_id for a fixed fixture (full pipeline incl. prop-normalize)', () => {
        const chunks = chunker.chunkContent(FIXTURE, FIXTURE_PATH);
        expect(chunks.length).toBe(1);
        expect(chunks[0].chunk_id).toBe('1eb905d44ee9d5');
    });

    it('pins the chunk_id AND link_terms of a link/URL-rich fixture (dense-clean edits go red here)', () => {
        const chunks = chunker.chunkContent(LINK_FIXTURE, LINK_FIXTURE_PATH);
        expect(chunks.length).toBe(1);
        expect(chunks[0].chunk_id).toBe('0e97eb738d1fdd');
        expect(chunks[0].link_terms).toBe(
            'Jordan Rivera Notes/Projects/Seek#Status https://www.theverge.com/tech/some-post ' +
            'https://feeds.example.com/rss Pasted image 20260702.png',
        );
        // The cleaned body must carry the display forms, not the raw syntax.
        expect(chunks[0].content).toContain('The Verge');
        expect(chunks[0].content).toContain('Jordan');
        expect(chunks[0].content).not.toContain('theverge.com');
    });

    it('pins the chunk_id of an inline-tag fixture (tags are metadata, NOT hashed)', () => {
        // Audit R2 #2: inline body tags join metadata.tags. The tag TEXT is body
        // content (already hashed); the extracted LIST must stay out of chunkIdFor
        // — if someone later hashes it in, this pin goes red with no other change.
        const content = `---
tags: [cooking]
---
# Weeknight Pasta

A #recipe I keep coming back to, quick enough for a #weeknight and tagged inline only.
`;
        const chunks = chunker.chunkContent(content, 'Notes/Weeknight Pasta.md');
        expect(chunks.length).toBe(1);
        expect(chunks[0].chunk_id).toBe('00da60827e5e93');
        expect(chunks[0].metadata.tags).toEqual(['cooking', 'recipe', 'weeknight']);
    });

    it('pins the chunk_id AND title of a legacy alias: fixture (aliases enter the hashed title)', () => {
        // Audit R2 #5: the singular key now reaches titleWithAliases → title/denseSuffix
        // → chunk_id. This id is NEW BY DESIGN (the pre-fix chunker ignored the key);
        // pinned so the next accidental alias-handling change is a conscious one.
        const content = `---
alias: The Saloon
---
A San Francisco bar we keep going back to for the piano and the late kitchen hours.
`;
        const chunks = chunker.chunkContent(content, 'Places/Saloon.md');
        expect(chunks.length).toBe(1);
        expect(chunks[0].title).toBe('Saloon | The Saloon');
        expect(chunks[0].chunk_id).toBe('1ba9d9643461b7');
    });

    it('chunkIdFor is a stable pure hash of (path, title, content, suffix)', () => {
        // Pure determinism guard — catches a cyrb53 / salt-format change independent of
        // split logic. No suffix → the pre-dense-suffix id form.
        expect(chunkIdFor('Notes/Foo.md', 'Foo', 'body text')).toBe('1324c39d06d44a');
        // With a dense suffix → the tail enters the id (a delta must re-embed on suffix change).
        expect(chunkIdFor('Notes/Foo.md', 'Foo', 'body text', 'project active')).toBe('1c8096b6b2c2b8');
    });
});
