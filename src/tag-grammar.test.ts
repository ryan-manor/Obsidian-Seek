import { describe, it, expect } from 'vitest';
import { extractInlineTags } from './tag-grammar';
import { MarkdownChunker } from './chunker';

describe('extractInlineTags — doc-side inline tag scan (audit R2 #2)', () => {
    it('finds a free-standing tag in prose', () => {
        expect(extractInlineTags('Cooked this from the #recipe box today.')).toEqual(['recipe']);
    });

    it('keeps hierarchy and kebab/underscore', () => {
        expect(extractInlineTags('see #meetings/1x1 and #meeting-prep and #a_b'))
            .toEqual(['meetings/1x1', 'meeting-prep', 'a_b']);
    });

    it('matches at start of body and start of line', () => {
        expect(extractInlineTags('#inbox first line\nsecond\n#followup done')).toEqual(['inbox', 'followup']);
    });

    it('dedups case-insensitively, first casing wins', () => {
        expect(extractInlineTags('#Recipe then again #recipe and #RECIPE')).toEqual(['Recipe']);
    });

    it('trailing punctuation terminates the tag', () => {
        expect(extractInlineTags('done (#done). also #wip, and #x/y.')).toEqual(['wip', 'x/y']);
        // `(#done` has no whitespace boundary before `#` — Obsidian does not
        // bind it either; `#wip,` and `#x/y.` stop at the punctuation.
        expect(extractInlineTags('done #done. #x/y.')).toEqual(['done', 'x/y']);
    });

    it('headings are not tags', () => {
        expect(extractInlineTags('# Title\n\n## Section\n\nprose')).toEqual([]);
    });

    it('URL fragments and mid-word hashes are not tags', () => {
        expect(extractInlineTags('see https://x.com/page#section and a#b')).toEqual([]);
    });

    it('purely-numeric tags are rejected (Obsidian grammar), lettered ones kept', () => {
        expect(extractInlineTags('in #1984 and #2026/06')).toEqual([]);
        expect(extractInlineTags('in #y1984')).toEqual(['y1984']);
    });

    it('unicode tags bind (#café, #日本語)', () => {
        expect(extractInlineTags('at the #café talking #日本語')).toEqual(['café', '日本語']);
    });

    it('fenced code contributes nothing (backtick and tilde fences)', () => {
        const body = [
            'prose #real',
            '',
            '```c',
            '#include <stdio.h>',
            '#define X 1',
            '```',
            '',
            '~~~',
            '#notatag',
            '~~~',
        ].join('\n');
        expect(extractInlineTags(body)).toEqual(['real']);
    });

    it('inline code spans contribute nothing', () => {
        expect(extractInlineTags('color `#fff` and `#include` but #css counts')).toEqual(['css']);
    });

    it('Obsidian %% comments contribute nothing (inline and block)', () => {
        expect(extractInlineTags('visible %%#draft%% text')).toEqual([]);
        expect(extractInlineTags('%%\n#hidden line\n%%\n\n#shown')).toEqual(['shown']);
    });

    it('an UNCLOSED %% comments out to end-of-note (Obsidian semantics)', () => {
        expect(extractInlineTags('%%\nsecret #hidden to EOF')).toEqual([]);
        expect(extractInlineTags('before #real\n\n%%\n#hidden')).toEqual(['real']);
    });

    it('stripping a code/comment span does not manufacture a tag boundary', () => {
        // Obsidian does not bind a # glued to a closing backtick or %% — a
        // space-replacement strip would (adversarial review 2026-07-02).
        expect(extractInlineTags('x `y`#z stays unbound')).toEqual([]);
        expect(extractInlineTags('a %%b%%#c stays unbound')).toEqual([]);
    });

    it('tags inside callouts and tables count (visible content)', () => {
        const body = [
            '> [!note]',
            '> tagged #callout-tag here',
            '',
            '| col |',
            '| --- |',
            '| #table-tag |',
        ].join('\n');
        expect(extractInlineTags(body)).toEqual(['callout-tag', 'table-tag']);
    });

    it('empty / hash-free body fast-paths to []', () => {
        expect(extractInlineTags('')).toEqual([]);
        expect(extractInlineTags('plain prose, nothing else')).toEqual([]);
    });
});

describe('chunker metadata.tags — frontmatter ∪ inline (audit R2 #2)', () => {
    const chunker = new MarkdownChunker();

    it('unions inline body tags with frontmatter tags, note-level on every chunk', () => {
        const content = [
            '---',
            'tags:',
            '  - recipes',
            '---',
            '# Pasta',
            'A quick #weeknight dinner. Long enough body text to clear the minimum chunk size threshold for emission.',
            '# Notes',
            'Freezes well, reheat gently in a pan with a splash of water so it does not dry out.',
        ].join('\n');
        const chunks = chunker.chunkContent(content, 'Inbox/Pasta.md');
        expect(chunks.length).toBeGreaterThan(1);
        for (const c of chunks) {
            expect(c.metadata.tags).toEqual(['recipes', 'weeknight']);
        }
    });

    it('dedups frontmatter vs inline case-insensitively, frontmatter casing wins', () => {
        const content = [
            '---',
            'tags: [Recipe]',
            '---',
            'Tagged again inline as #recipe deep in the prose of this note body here.',
        ].join('\n');
        const chunks = chunker.chunkContent(content, 'Inbox/x.md');
        expect(chunks[0].metadata.tags).toEqual(['Recipe']);
    });

    it('inline-only note gets tags without any frontmatter', () => {
        const content = 'No frontmatter at all, just prose mentioning #standup and #meetings/1x1 in passing today.';
        const chunks = chunker.chunkContent(content, 'Inbox/y.md');
        expect(chunks[0].metadata.tags).toEqual(['standup', 'meetings/1x1']);
    });
});

describe('chunker aliases — legacy singular alias: key (audit R2 #5)', () => {
    const chunker = new MarkdownChunker();
    const body = 'Body text long enough to comfortably clear the minimum chunk length gate for a single emitted chunk.';

    it('reads the legacy singular key into aliases and the chunk title', () => {
        const content = `---\nalias: The Saloon\n---\n${body}`;
        const chunks = chunker.chunkContent(content, 'Places/Saloon.md');
        expect(chunks[0].metadata.aliases).toEqual(['The Saloon']);
        expect(chunks[0].title).toBe('Saloon | The Saloon');
    });

    it('merges both keys, plural first, case-insensitive dedup', () => {
        const content = `---\naliases: [Will, Bill]\nalias: bill, Billy\n---\n${body}`;
        const chunks = chunker.chunkContent(content, 'People/William.md');
        expect(chunks[0].metadata.aliases).toEqual(['Will', 'Bill', 'Billy']);
    });

    it('aliases-only note is byte-identical to the pre-fix read (id stability)', () => {
        const content = `---\naliases: [Rapha]\n---\n${body}`;
        const chunks = chunker.chunkContent(content, 'Brands/R.md');
        expect(chunks[0].metadata.aliases).toEqual(['Rapha']);
        expect(chunks[0].title).toBe('R | Rapha');
    });

    it('duplicate entries WITHIN aliases: ride verbatim (byte-identity holds for dupes too)', () => {
        // Adversarial review 2026-07-02: an intra-list ci-dedup would have
        // moved this note's title/chunk_id despite it never using `alias:`.
        const content = `---\naliases: [Foo, foo]\n---\n${body}`;
        const chunks = chunker.chunkContent(content, 'Notes/F.md');
        expect(chunks[0].metadata.aliases).toEqual(['Foo', 'foo']);
        expect(chunks[0].title).toBe('F | Foo | foo');
    });

    it('case-variant frontmatter keys are read (Alias:/Tags: — Obsidian keys are display labels)', () => {
        const content = `---\nAlias: The Saloon\nTags: [bars]\n---\n${body}`;
        const chunks = chunker.chunkContent(content, 'Places/S.md');
        expect(chunks[0].metadata.aliases).toEqual(['The Saloon']);
        expect(chunks[0].metadata.tags).toEqual(['bars']);
    });
});
