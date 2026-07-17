import { describe, it, expect } from 'vitest';
import { segmentSentences, buildPassageTerms, bestPassage, passageWindow, markPattern } from './passage';

// Uniform IDF — tests that don't exercise weighting use this.
const flatIdf = (): number => 0;

describe('segmentSentences', () => {
    it('splits on sentence-final punctuation followed by an uppercase start', () => {
        const text = 'First sentence here. Second sentence follows.';
        const spans = segmentSentences(text);
        expect(spans.length).toBe(2);
        expect(text.slice(spans[0].start, spans[0].end)).toBe('First sentence here.');
        expect(text.slice(spans[1].start, spans[1].end)).toBe('Second sentence follows.');
    });

    it('does not split after an abbreviation followed by lowercase (e.g., vs.)', () => {
        const spans = segmentSentences('This works e.g. when lowercase continues the thought');
        expect(spans.length).toBe(1);
    });

    it('does not split decimal numbers', () => {
        const spans = segmentSentences('Version 2.5 shipped with the fix');
        expect(spans.length).toBe(1);
    });

    it('treats every newline as a boundary (markdown structure)', () => {
        const text = 'Heading Line\nBody text under it.\n- list item one\n- list item two';
        const spans = segmentSentences(text);
        expect(spans.length).toBe(4);
        expect(text.slice(spans[0].start, spans[0].end)).toBe('Heading Line');
    });

    it('absorbs closing quotes/brackets into the sentence', () => {
        const text = 'He said "stop." Then he left.';
        const spans = segmentSentences(text);
        expect(spans.length).toBe(2);
        expect(text.slice(spans[0].start, spans[0].end)).toBe('He said "stop."');
    });

    it('returns one trimmed span for unpunctuated text and none for whitespace', () => {
        const spans = segmentSentences('  no terminal punctuation here  ');
        expect(spans.length).toBe(1);
        expect(spans[0].start).toBe(2);
        expect(segmentSentences('   \n  \n')).toEqual([]);
    });
});

describe('buildPassageTerms', () => {
    it('drops stopwords and single-char tokens, dedupes repeats', () => {
        // "not" and "the" are ENGLISH_STOPWORDS; "a" is single-char + stopword.
        const terms = buildPassageTerms('the bread not rising bread a', flatIdf);
        expect(terms.length).toBe(2); // bread, rising
    });

    it('carries raw AND processed alternates so inflections bridge both ways', () => {
        const [term] = buildPassageTerms('stories', flatIdf);
        term.re.lastIndex = 0;
        expect(term.re.test('the story begins')).toBe(true);   // processed stem "story"
        term.re.lastIndex = 0;
        expect(term.re.test('the stories begin')).toBe(true);  // raw surface form
    });

    it('maps document frequency to BM25-shaped idf (rarer → larger)', () => {
        const idfOf = (t: string): number => (t === 'quantization' ? 0.01 : 0.5);
        const terms = buildPassageTerms('quantization disk', idfOf);
        const [rare, common] = terms;
        expect(rare.idf).toBeGreaterThan(common.idf);
    });

    it('falls back to literal terms with uniform idf on an all-stopword query', () => {
        const terms = buildPassageTerms('the these', flatIdf);
        expect(terms.length).toBe(2);
        expect(terms[0].idf).toBe(terms[1].idf);
        terms[0].re.lastIndex = 0;
        expect(terms[0].re.test('the who played loud')).toBe(true);
    });
});

describe('bestPassage — the reproduced pathologies', () => {
    // Old makeSnippet anchored on the earliest raw indexOf of ANY token —
    // query "bread not rising" anchored on the "not" INSIDE "cannot".
    it('"bread not rising" ignores "cannot" and picks the sentence with real evidence', () => {
        const text =
            'We cannot say why the starter failed overnight. ' +
            'The bread is not rising in the cold kitchen.';
        const terms = buildPassageTerms('bread not rising', flatIdf);
        const best = bestPassage(text, terms);
        expect(best).not.toBeNull();
        expect(text.slice(best!.start, best!.end)).toBe('The bread is not rising in the cold kitchen.');
        // No mark may sit inside "cannot" — "not" was dropped as a stopword and
        // word boundaries stop mid-word hits.
        const cannotAt = text.indexOf('cannot');
        for (const [from, to] of best!.marks) {
            expect(to <= cannotAt || from >= cannotAt + 'cannot'.length).toBe(true);
        }
    });

    // Old makeSnippet's substring scan let "up" match INSIDE "grouped".
    it('"take up less disk space" cannot anchor "up" inside "grouped"', () => {
        const text =
            'Files are grouped by month in the archive folder. ' +
            'Compressed indexes take up less disk space on mobile.';
        const terms = buildPassageTerms('take up less disk space', flatIdf);
        const best = bestPassage(text, terms);
        expect(best).not.toBeNull();
        expect(text.slice(best!.start, best!.end)).toBe('Compressed indexes take up less disk space on mobile.');
    });

    it('returns null when no term matches (dense-only chunk → caller falls back)', () => {
        const terms = buildPassageTerms('quantization', flatIdf);
        expect(bestPassage('Nothing about that topic lives here.', terms)).toBeNull();
        expect(bestPassage('any text', [])).toBeNull();
    });

    it('weighs sentences by idf: the rare term beats the common one', () => {
        const text =
            'The disk was replaced on Tuesday afternoon quietly. ' +
            'Quantization halves the index without recall loss.';
        const idfOf = (t: string): number => (t === 'quantization' ? 0.01 : 0.5);
        const terms = buildPassageTerms('quantization disk', idfOf);
        const best = bestPassage(text, terms);
        expect(text.slice(best!.start, best!.end)).toContain('Quantization');
    });

    it('breaks ties toward the earlier passage (lead bias)', () => {
        const text =
            'Quantization shrinks the model. Filler sentence sits between. ' +
            'Quantization shrinks the model.';
        const terms = buildPassageTerms('quantization', flatIdf);
        const best = bestPassage(text, terms);
        expect(best!.start).toBe(0);
    });

    it('returns ascending marks confined to the winning sentence', () => {
        const text = 'Nothing here. The index maps index terms to index rows.';
        const terms = buildPassageTerms('index', flatIdf);
        const best = bestPassage(text, terms);
        expect(best!.marks.length).toBe(3);
        for (let i = 1; i < best!.marks.length; i++) {
            expect(best!.marks[i][0]).toBeGreaterThanOrEqual(best!.marks[i - 1][1]);
        }
        for (const [from, to] of best!.marks) {
            expect(from).toBeGreaterThanOrEqual(best!.start);
            expect(to).toBeLessThanOrEqual(best!.end);
        }
    });
});

describe('passageWindow', () => {
    it('anchors the window at the winning sentence start', () => {
        const text =
            'An opening line with no evidence at all in it. ' +
            'The bread is rising nicely today.';
        const terms = buildPassageTerms('bread rising', flatIdf);
        const { start } = passageWindow(text, terms, 200);
        expect(start).toBe(text.indexOf('The bread'));
    });

    it('falls back to the chunk head with no terms or no match', () => {
        expect(passageWindow('some text here', [], 200)).toEqual({ start: 0, end: 14 });
        const terms = buildPassageTerms('quantization', flatIdf);
        expect(passageWindow('unrelated words only', terms, 200).start).toBe(0);
    });

    it('slides into a wall-of-text sentence whose first hit falls outside the window', () => {
        const filler = 'lorem ipsum dolor sit amet consectetur adipiscing elit sed do '.repeat(5);
        const text = filler + 'quantization at the tail';   // one giant unpunctuated "sentence"
        const terms = buildPassageTerms('quantization', flatIdf);
        const { start, end } = passageWindow(text, terms, 200);
        const hit = text.indexOf('quantization');
        expect(start).toBe(hit - 40);
        expect(end - start).toBeLessThanOrEqual(200);
    });
});

describe('markPattern', () => {
    it('is null for no terms and matches every term vocabulary globally', () => {
        expect(markPattern([])).toBeNull();
        const re = markPattern(buildPassageTerms('bread rising', flatIdf))!;
        const hits = 'rising bread breads risings'.match(re);
        expect(hits).toEqual(['rising', 'bread', 'breads', 'risings']);
    });
});
