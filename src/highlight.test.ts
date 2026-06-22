import { describe, it, expect } from 'vitest';
import { maskNonBodyText, buildHighlightRanges } from './highlight';

// A small stand-in for the real ENGLISH_STOPWORDS set (only the words used here).
const STOP = new Set(['of', 'the', 'and', 'in', 'a']);

describe('maskNonBodyText', () => {
    it('preserves total length and every newline position', () => {
        const src = '---\na: 1\n---\n# H\n\nbody [[Target|alias]] `code` text\n';
        const masked = maskNonBodyText(src);
        expect(masked.length).toBe(src.length);
        for (let i = 0; i < src.length; i++) {
            if (src[i] === '\n') expect(masked[i]).toBe('\n');
        }
    });

    it('blanks frontmatter, embeds, wikilink targets and code but keeps alias text', () => {
        const src = 'see [[Real Target|shown alias]] and ![[embed note]] and `inline code` end';
        const masked = maskNonBodyText(src);
        expect(masked).toContain('shown alias'); // visible alias kept
        expect(masked).not.toContain('Real Target'); // wikilink target blanked
        expect(masked).not.toContain('embed'); // embed transclusion blanked
        expect(masked).not.toContain('inline'); // inline code blanked
    });
});

// The real failure this guards against: a query word ("recommendations") appears
// in an intro line BEFORE the chunk's headline word ("pillars"), so scanning
// tokens in QUERY order yields DESCENDING document offsets — which Obsidian's
// eState.match (a CodeMirror RangeSet) silently mis-paints onto unrelated text.
const NOTE =
    '---\npageType: notes\n---\n' +
    '# Overview\n\n' +
    'asset recommendations are a component here.\n\n' + // "recommendations" early
    '# Section\n\n' +
    'There are three basic pillars or principles.\n';   // "pillars" late

describe('buildHighlightRanges — ordering (out-of-order drift regression)', () => {
    it('returns ranges sorted ascending even when query order is reversed', () => {
        const tokens = ['pillars', 'of', 'recommendations']; // "of" is a stopword, dropped
        const ranges = buildHighlightRanges(NOTE, tokens, 0, NOTE.length, STOP);
        expect(ranges.length).toBe(2);
        expect(ranges[0][0]).toBeLessThan(ranges[1][0]); // ascending by start offset
        expect(NOTE.slice(...ranges[0])).toBe('recommendations'); // earlier word first
        expect(NOTE.slice(...ranges[1])).toBe('pillars');
    });

    it('skips stopwords and single-char tokens', () => {
        expect(buildHighlightRanges(NOTE, ['a', 'of', 'the'], 0, NOTE.length, STOP)).toEqual([]);
    });

    it('respects the window upper bound', () => {
        const cut = NOTE.indexOf('pillars'); // window ends before "pillars"
        const ranges = buildHighlightRanges(NOTE, ['pillars', 'recommendations'], 0, cut, STOP);
        expect(ranges.length).toBe(1);
        expect(NOTE.slice(...ranges[0])).toBe('recommendations');
    });

    it('extends a stem to its inflection via \\w* and anchors on word boundaries', () => {
        const ranges = buildHighlightRanges(NOTE, ['recommendation'], 0, NOTE.length, STOP);
        expect(ranges.length).toBe(1);
        expect(NOTE.slice(...ranges[0])).toBe('recommendations');
    });

    it('never matches mid-word', () => {
        const src = 'a professional hobbyist\n';
        // "pro" is a word-start prefix → matches the whole word "professional"
        const r1 = buildHighlightRanges(src, ['pro'], 0, src.length, STOP);
        expect(src.slice(...r1[0])).toBe('professional');
        // "essional" is mid-word → no match at all
        expect(buildHighlightRanges(src, ['essional'], 0, src.length, STOP)).toEqual([]);
    });
});

describe('buildHighlightRanges — offset defence', () => {
    it('drops overlapping ranges, keeping a disjoint ascending set', () => {
        // Both tokens' first match resolves to the same "pillars" span → overlap.
        const src = 'the pillars stand\n';
        const ranges = buildHighlightRanges(src, ['pillars', 'pill'], 0, src.length, STOP);
        expect(ranges.length).toBe(1);
        expect(src.slice(...ranges[0])).toBe('pillars');
    });
});
