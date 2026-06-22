import { describe, it, expect } from 'vitest';
import { MarkdownChunker } from './chunker';

const chunker = new MarkdownChunker();

// Whitespace-insensitive subsequence check: every char of `needle` appears in
// `haystack`, in order. hardSplit overlaps adjacent parts and trims boundary
// whitespace, so the concatenation is a whitespace-different SUPERSET of the
// input — but if any non-whitespace content was DROPPED (the S7 bug), the input
// stops being a subsequence. This is the precise anti-drop invariant.
function isSubsequenceIgnoringWs(haystack: string, needle: string): boolean {
    const h = haystack.replace(/\s+/g, '');
    const n = needle.replace(/\s+/g, '');
    let i = 0;
    for (let j = 0; j < h.length && i < n.length; j++) {
        if (h[j] === n[i]) i++;
    }
    return i === n.length;
}

describe('MarkdownChunker — lexical-only fallback flag', () => {
    // The wife's bug: a body-LESS daily note (frontmatter only) embeds as just
    // its title (a bare date), becoming a universal near-neighbor for any OOV/ID
    // query. The fallback chunk must be flagged lexicalOnly so the ranker keeps
    // it out of the dense channel.
    it('body-less note (frontmatter only) => fallback chunk is lexicalOnly', () => {
        const content = [
            '---',
            'pageType: notes',
            'created: 2026-05-26',
            'tags:',
            '  - notes',
            '---',
            '',
        ].join('\n');
        const chunks = chunker.chunkContent(content, 'Notes/2026-05-26 12.28.md');
        expect(chunks).toHaveLength(1);
        expect(chunks[0].lexicalOnly).toBe(true);
        expect(chunks[0].content).toBe('');
    });

    it('no frontmatter, empty body => lexicalOnly', () => {
        const chunks = chunker.chunkContent('', 'Notes/Empty.md');
        expect(chunks).toHaveLength(1);
        expect(chunks[0].lexicalOnly).toBe(true);
    });

    // A short-but-non-empty fallback (e.g. a one-line stub) DOES carry embeddable
    // content, so it must stay dense-eligible — the flag is absent, not false.
    it('short non-empty body (under minChunkChars) => fallback is NOT lexicalOnly', () => {
        const content = '---\npageType: person\n---\nMet Bob re: pricing.';
        const chunks = chunker.chunkContent(content, 'People/Bob.md');
        expect(chunks).toHaveLength(1);
        expect(chunks[0].lexicalOnly).toBeUndefined();
        expect(chunks[0].content.length).toBeGreaterThan(0);
    });

    it('normal multi-section note => no chunk is lexicalOnly', () => {
        const body = 'x'.repeat(80);
        const content = `# Heading\n\n${body}\n\n## Sub\n\n${body}`;
        const chunks = chunker.chunkContent(content, 'Notes/Real.md');
        expect(chunks.length).toBeGreaterThan(0);
        for (const c of chunks) expect(c.lexicalOnly).toBeUndefined();
    });
});

describe('MarkdownChunker — start_line/end_line are RAW-FILE coordinates (frontmatter included)', () => {
    // Regression for the in-note highlight + click-scroll landing INSIDE the
    // frontmatter. Pre-fix start_line was counted against the frontmatter-stripped
    // body, but its only consumers (search-modal.ts) walk the RAW note text and drive
    // the editor (which counts frontmatter as lines), so they landed fmLineCount
    // lines too early — dragging the highlight onto the note's FIRST token hit
    // instead of the matched chunk's, diverging from the snippet.
    const FM = ['---', 'pageType: notes', 'tags:', '  - notes', '---'].join('\n'); // 5 file lines

    it('shifts a section start_line past the frontmatter block', () => {
        const content = [FM, '# Overview', '', 'x'.repeat(120)].join('\n');
        const chunks = chunker.chunkContent(content, 'n.md');
        const fileLines = content.split('\n');
        expect(chunks.length).toBeGreaterThan(0);
        expect(chunks[0].start_line).toBe(6); // heading is file line 6, not body line 1
        expect(fileLines[chunks[0].start_line - 1]).toBe('# Overview');
    });

    it('no frontmatter => start_line is the unchanged file line', () => {
        const content = ['# Heading', '', 'y'.repeat(120)].join('\n');
        const chunks = chunker.chunkContent(content, 'n.md');
        expect(chunks[0].start_line).toBe(1);
        expect(content.split('\n')[chunks[0].start_line - 1]).toBe('# Heading');
    });

    it('walking start_line-1 newlines lands in the MATCHED section, excluding an earlier-section token hit', () => {
        // "recommendations" appears in the Overview (decoy) AND the later Planning
        // section (the matched chunk). The decoy must fall BEFORE the resolved
        // lineStart so the highlight window can only bind the chunk's own copy.
        const content = [
            FM,                                                              // lines 1-5
            '# Overview', '',                                                // lines 6-7
            'asset recommendations are a component of the broader system.',  // line 8  (decoy)
            '',                                                              // line 9
            '# Planning', '',                                                // lines 10-11
            'three basic pillars for providing recommendations to the user.', // line 12 (matched)
        ].join('\n');
        const chunks = chunker.chunkContent(content, 'n.md');
        const planning = chunks.find(c => c.content.includes('pillars'));
        expect(planning).toBeTruthy();
        expect(planning!.start_line).toBe(10); // "# Planning" file line

        // Resolve lineStart exactly as search-modal.ts buildMatchHighlight does.
        let lineStart = 0;
        for (let i = 0; i < planning!.start_line - 1; i++) lineStart = content.indexOf('\n', lineStart) + 1;
        const decoy = content.indexOf('recommendations');               // Overview copy
        const matched = content.indexOf('recommendations', lineStart);  // Planning copy
        expect(decoy).toBeLessThan(lineStart);                          // excluded by the window
        expect(matched).toBeGreaterThanOrEqual(lineStart);             // the one we highlight
    });
});

describe('MarkdownChunker — sub-min sections fold instead of dropping (§7)', () => {
    const big = 'x'.repeat(80); // clears minChunkChars on its own

    // The core §7 hole: a note with one healthy section silently lost its short
    // sections from BOTH indexes. They must survive, folded into a neighbour.
    it('short section before a full section folds FORWARD into it', () => {
        const content = `## Quick Note\ncall the bank\n\n## Big\n\n${big}`;
        const chunks = chunker.chunkContent(content, 'Notes/Foo.md');
        expect(chunks).toHaveLength(1);
        // body text preserved...
        expect(chunks[0].content).toContain('call the bank');
        // ...and the heading word too, so it stays lexically searchable.
        expect(chunks[0].content).toContain('Quick Note');
        expect(chunks[0].content).toContain(big);
        // resolves to the correct note regardless of the fold
        expect(chunks[0].note_path).toBe('Notes/Foo.md');
    });

    it('trailing short section folds BACKWARD into the preceding chunk', () => {
        const content = `## Big\n\n${big}\n\n## PS\nsee above`;
        const chunks = chunker.chunkContent(content, 'Notes/Bar.md');
        expect(chunks).toHaveLength(1);
        expect(chunks[0].content).toContain(big);
        expect(chunks[0].content).toContain('see above');
        expect(chunks[0].content).toContain('PS');
        // backward fold keeps the preceding section's title/attribution
        expect(chunks[0].title).toBe('Bar > Big');
    });

    // Case 1 generalized: a "napkin note" made entirely of tiny sections becomes
    // ONE real, body-bearing chunk — NOT the title-only stub that drops the body.
    it('note of only short sections => single body-bearing chunk, not lexicalOnly', () => {
        const content = `## Bank\ncall them\n\n## Rent\ndue Friday`;
        const chunks = chunker.chunkContent(content, 'Notes/Napkin.md');
        expect(chunks).toHaveLength(1);
        expect(chunks[0].lexicalOnly).toBeUndefined();
        expect(chunks[0].content).toContain('call them');
        expect(chunks[0].content).toContain('due Friday');
        expect(chunks[0].content).toContain('Bank');
        expect(chunks[0].content).toContain('Rent');
    });

    // Case 1: a note whose only body is a link is indexed by that link, not dropped.
    it('link-only note => one dense-eligible chunk carrying the link text', () => {
        const chunks = chunker.chunkContent('[[Some Project]]', 'Notes/Stub.md');
        expect(chunks).toHaveLength(1);
        expect(chunks[0].lexicalOnly).toBeUndefined();
        expect(chunks[0].content).toContain('Some Project');
        expect(chunks[0].note_path).toBe('Notes/Stub.md');
    });

    // Two healthy sections must stay separate — folding only touches sub-min ones.
    it('two full sections are not merged', () => {
        const content = `## A\n\n${big}\n\n## B\n\n${big}`;
        const chunks = chunker.chunkContent(content, 'Notes/Two.md');
        expect(chunks).toHaveLength(2);
    });
});

describe('MarkdownChunker — path-salted chunk_id (same-content twins stay distinct)', () => {
    // Two notes with the SAME basename (→ same title) and identical content but
    // different paths. Without the path salt their chunk_ids collapse to one row
    // that both file-records reference, so deleting/editing one orphans the other.
    // The salt must give them distinct ids while leaving title/content identical.
    it('identical title+content in different paths → distinct chunk_ids', () => {
        const content = `---\npageType: note\n---\n# Heading\n\n${'x'.repeat(120)}`;
        const a = chunker.chunkContent(content, 'FolderA/note.md');
        const b = chunker.chunkContent(content, 'FolderB/note.md');
        expect(a.length).toBe(b.length);
        expect(a.length).toBeGreaterThan(0);
        for (let i = 0; i < a.length; i++) {
            expect(a[i].title).toBe(b[i].title);         // same basename → same title
            expect(a[i].content).toBe(b[i].content);     // identical content
            expect(a[i].chunk_id).not.toBe(b[i].chunk_id); // ...but distinct ids
        }
    });

    // The fallback (title-only) path must be salted too — body-less daily-note
    // stubs are the classic same-title-and-(empty)-content twins.
    it('body-less twins in different paths → distinct chunk_ids', () => {
        const content = '---\npageType: note\ncreated: 2026-05-26\n---\n';
        const a = chunker.chunkContent(content, 'Daily/A.md');
        const b = chunker.chunkContent(content, 'Daily/B.md');
        expect(a).toHaveLength(1);
        expect(b).toHaveLength(1);
        expect(a[0].chunk_id).not.toBe(b[0].chunk_id);
    });

    // Salt must not break determinism: same path + same content → same id.
    it('same path + same content → identical chunk_ids (deterministic)', () => {
        const content = `# H\n\n${'y'.repeat(120)}`;
        const a = chunker.chunkContent(content, 'Notes/Same.md');
        const b = chunker.chunkContent(content, 'Notes/Same.md');
        expect(a.map(c => c.chunk_id)).toEqual(b.map(c => c.chunk_id));
    });
});

describe('MarkdownChunker — sections emit WHOLE (WS3: splitting lives in token-budget.ts)', () => {
    // Pre-WS3 the chunker char-split oversize sections at 6750 with overlap.
    // Now any size section emits as ONE chunk; enforceTokenBudget is the only
    // splitter (and the only assigner of "(part N)" displayTitles — see
    // token-budget.test.ts for those invariants).
    it('an oversize section emits as one chunk with no displayTitle', () => {
        const distinctBody = Array.from({ length: 3000 }, (_, i) => `tok${i}`).join(' ');
        const chunks = chunker.chunkContent(`## Section\n\n${distinctBody}`, 'Notes/Doc.md');
        expect(chunks).toHaveLength(1);
        expect(chunks[0].title).toBe('Doc > Section');
        expect(chunks[0].displayTitle).toBeUndefined();
        expect(isSubsequenceIgnoringWs(chunks[0].content, distinctBody)).toBe(true);
    });

    it('a 20k-char headingless blob emits whole, losing no content', () => {
        const body = `${'x'.repeat(50)} ${'y'.repeat(19000)}`;
        const chunks = chunker.chunkContent(body, 'Notes/Blob.md');
        expect(chunks).toHaveLength(1);
        expect(isSubsequenceIgnoringWs(chunks[0].content, body)).toBe(true);
    });
});

describe('MarkdownChunker — fence-aware heading detection (WS3)', () => {
    it('does not split a section at # lines inside fenced code', () => {
        const note = [
            '# Setup', '',
            'Bootstrap script.', '',
            '```bash',
            '# Install deps',
            'brew install jq', '',
            '## still a comment',
            'export FOO=1',
            '```', '',
            '## Verification', '',
            'Run the bootstrap once and check the log line output for the expected device id stamp.',
        ].join('\n');
        const chunks = chunker.chunkContent(note, 'Notes/Shell.md');
        expect(chunks.map(c => c.title)).toEqual([
            'Shell > Setup',
            'Shell > Setup > Verification', // H2 nests under the H1
        ]);
        // The fence arrives intact in the Setup section, comments included.
        expect(chunks[0].content).toContain('# Install deps');
        expect(chunks[0].content).toContain('## still a comment');
        expect(chunks[0].content).toContain('```bash');
    });

    it('treats everything after an unterminated fence as code, not sections', () => {
        const note = 'intro paragraph\n\n```text\n# not a heading\n## also not\nstill code';
        const chunks = chunker.chunkContent(note, 'Notes/Paste.md');
        expect(chunks).toHaveLength(1);
        expect(chunks[0].heading_path).toEqual([]);
    });
});

describe('MarkdownChunker — dense frontmatter suffix (2026-06-18 frontmatter-into-dense)', () => {
    // buildDenseSuffix: text-typed frontmatter values folded into the dense
    // channel, keys dropped, wikilinks→basename, aliases excluded, and
    // date/number/boolean values type-dropped. Mirrors the validated harness arm
    // fm_text_value_string (ofm_fm_valuetype.py).
    const note = [
        '---',
        'aliases:',
        '  - Pinthouse Pizza',
        'tags:',
        '  - food',
        'placeType: restaurant',
        'city: "[[Austin]]"',
        'created: 2026-06-17',
        'dateLink: "[[2026-06-17]]"',
        'rating: 5',
        'draft: false',
        '---',
        'Great patio seating downtown with wood-fired pizza and cold beer on tap.',
    ].join('\n');

    it('keeps text/link values, drops dates, numbers, booleans, and aliases', () => {
        const [chunk] = chunker.chunkContent(note, 'Notes/Places/Pinthouse.md');
        expect(chunk.denseSuffix).toBeDefined();
        // tags + placeType + city(link→basename) survive
        expect(chunk.denseSuffix).toContain('food');
        expect(chunk.denseSuffix).toContain('restaurant');
        expect(chunk.denseSuffix).toContain('Austin');
        // alias is already dense via the title — never re-injected here
        expect(chunk.title).toContain('Pinthouse Pizza');
        expect(chunk.denseSuffix).not.toContain('Pinthouse');
        // inert value types dropped (geometric CLS-mass noise on thin notes)
        expect(chunk.denseSuffix).not.toContain('2026-06-17'); // created + dateLink
        expect(chunk.denseSuffix).not.toMatch(/(^|\s)5(\s|$)/); // rating
        expect(chunk.denseSuffix).not.toContain('false');       // draft
    });

    it('omits the field entirely when no value survives the type filter', () => {
        const inert = [
            '---', 'created: 2026-06-17', 'rating: 5', 'draft: true', '---',
            'A body long enough to clear the minChunkChars gate and emit a real chunk.',
        ].join('\n');
        const [chunk] = chunker.chunkContent(inert, 'Notes/Daily/2026-06-17.md');
        expect(chunk.denseSuffix).toBeUndefined();
    });

    it('folds the suffix into chunk_id (so a delta reindex re-embeds, never strands)', () => {
        const body = 'Great patio seating downtown with wood-fired pizza and cold beer on tap.';
        const path = 'Notes/Places/Pinthouse.md';
        const [withFm] = chunker.chunkContent(`---\ncity: "[[Austin]]"\n---\n${body}`, path);
        const [without] = chunker.chunkContent(body, path);
        // identical path/title/content; the ONLY difference is the dense suffix,
        // so a differing id proves the suffix is hashed into chunk_id.
        expect(withFm.denseSuffix).toBe('Austin');
        expect(without.denseSuffix).toBeUndefined();
        expect(withFm.content).toBe(without.content);
        expect(withFm.chunk_id).not.toBe(without.chunk_id);
    });

    // ── lightweight cleanliness gates (2026-06-18, fm_cleanliness_arm.py) ──
    it('shape gate: drops URLs, asset paths, and coordinate strings by form', () => {
        const clip = [
            '---',
            'source: https://example.com/a?utm_source=x',
            'image: cover.png',
            'coordinates: 37.77, -122.41',
            'topic: photography',
            'kind: essay',
            '---',
            'A body long enough to clear the minChunkChars gate and emit a real chunk.',
        ].join('\n');
        const [chunk] = chunker.chunkContent(clip, 'Clippings/Great Article.md');
        expect(chunk.denseSuffix).toBeDefined();
        // text values survive
        expect(chunk.denseSuffix).toContain('photography');
        expect(chunk.denseSuffix).toContain('essay');
        // URL (https), asset (.png), and lat/long coordinate all dropped
        expect(chunk.denseSuffix).not.toContain('http');
        expect(chunk.denseSuffix).not.toContain('example.com');
        expect(chunk.denseSuffix).not.toContain('.png');
        expect(chunk.denseSuffix).not.toContain('37.77');
        expect(chunk.denseSuffix).not.toContain('122.41');
    });

    it('dedup gate: collapses a value repeated across keys to one (the context/tags doubling)', () => {
        const dup = [
            '---',
            'context: personal',
            'tags:',
            '  - personal',
            '  - travel',
            'places:',
            '  - "[[Tokyo]]"',
            '  - "[[Tokyo]]"',
            '---',
            'A body long enough to clear the minChunkChars gate and emit a real chunk.',
        ].join('\n');
        const [chunk] = chunker.chunkContent(dup, 'Notes/Trips/Japan.md');
        expect(chunk.denseSuffix).toBeDefined();
        const toks = chunk.denseSuffix!.split(/\s+/);
        // "personal" once despite context + a tag; "Tokyo" once despite two links
        expect(toks.filter((t) => t.toLowerCase() === 'personal')).toHaveLength(1);
        expect(toks.filter((t) => t === 'Tokyo')).toHaveLength(1);
        expect(chunk.denseSuffix).toContain('travel');
    });

    it('cap gate: truncates the joined suffix to 48 whitespace tokens', () => {
        const many = Array.from({ length: 60 }, (_, i) => `  - term${i}`).join('\n');
        const big = [
            '---', 'tags:', many, '---',
            'A body long enough to clear the minChunkChars gate and emit a real chunk.',
        ].join('\n');
        const [chunk] = chunker.chunkContent(big, 'Notes/Big.md');
        expect(chunk.denseSuffix).toBeDefined();
        expect(chunk.denseSuffix!.split(/\s+/)).toHaveLength(48);
    });
});
