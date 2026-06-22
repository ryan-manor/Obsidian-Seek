import { describe, it, expect } from 'vitest';
import { seekTokenize, hasCjk, segmentCjkToken } from './tokenize';

// Node ≥16 ships Intl.Segmenter with full ICU, so these tests exercise the
// REAL segmentation path the Obsidian renderer (Electron/V8) uses. Exact
// segment boundaries are ICU-dictionary-dependent and may shift across ICU
// versions — tests assert structural properties (multiple terms, no giant
// run, key words present), not the full segmentation.

describe('hasCjk', () => {
    it('detects Han, kana, and Hangul; passes Latin', () => {
        expect(hasCjk('東京')).toBe(true);
        expect(hasCjk('ラーメン')).toBe(true);
        expect(hasCjk('검색')).toBe(true);
        expect(hasCjk('tokyo ramen')).toBe(false);
        expect(hasCjk('café')).toBe(false);    // diacritics are not CJK
    });
});

describe('seekTokenize — Latin path is MiniSearch-default-identical', () => {
    it('splits on whitespace and punctuation, preserves tokens verbatim', () => {
        // '+' survives: it is \p{Sm} (math symbol), not \p{P} — MiniSearch's
        // default split keeps it as a token too, and parity is the contract.
        expect(seekTokenize('Hybrid search, BM25 + dense!')).toEqual(
            ['Hybrid', 'search', 'BM25', '+', 'dense']);
    });
    it('does not fold case or touch diacritics (processTerm owns that)', () => {
        expect(seekTokenize('Café Zürich')).toEqual(['Café', 'Zürich']);
    });
});

describe('seekTokenize — CJK segmentation', () => {
    it('segments a Chinese sentence into multiple words (was ONE token)', () => {
        const terms = seekTokenize('我喜欢在瑞士的酒店');
        expect(terms.length).toBeGreaterThan(2);
        expect(Math.max(...terms.map(t => t.length))).toBeLessThan(5);
    });
    it('segments Japanese mixed kana/kanji text', () => {
        const terms = seekTokenize('東京のラーメン屋を検索する');
        expect(terms.length).toBeGreaterThan(3);
        expect(terms).toContain('東京');
        expect(terms).toContain('ラーメン');
    });
    it('handles mixed Latin+CJK tokens', () => {
        const terms = seekTokenize('GPU加速で検索');
        expect(terms.join(' ')).toContain('GPU');
        expect(terms.length).toBeGreaterThan(2);
    });
    it('keeps Korean eojeol intact or finer — never a giant merged run', () => {
        const terms = seekTokenize('검색 엔진 최적화');
        expect(terms.length).toBeGreaterThanOrEqual(3);
    });
});

describe('segmentCjkToken', () => {
    it('never returns empty for a non-empty token', () => {
        expect(segmentCjkToken('的').length).toBeGreaterThan(0);
    });
});

// ── Item 6: punctuation-glued compound tokens (additive) ──────────────────
// 2026-06-18 audit §6: query "gpt4" could not reach a doc that wrote "GPT-4".
// seekTokenize now ALSO emits the glue-stripped joined form, on both sides.
describe('seekTokenize — glue-compound joins (audit §6)', () => {
    it('emits the punctuation-stripped joined form additively', () => {
        expect(seekTokenize('GPT-4')).toEqual(['GPT', '4', 'GPT4']);
        expect(seekTokenize('co-pilot')).toEqual(['co', 'pilot', 'copilot']);
        expect(seekTokenize('granite-r2 model')).toEqual(['granite', 'r2', 'graniter2', 'model']);
        expect(seekTokenize('TCP/IP')).toEqual(['TCP', 'IP', 'TCPIP']);
        expect(seekTokenize('v2.0')).toEqual(['v2', '0', 'v20']);
    });
    it('joins a run of 3+ glued fragments ONCE (U.S.A -> USA), never pairwise', () => {
        expect(seekTokenize('U.S.A')).toEqual(['U', 'S', 'A', 'USA']);
    });
    it('a single glue-free fragment yields no redundant join token', () => {
        expect(seekTokenize('hello')).toEqual(['hello']);
    });
    it('whitespace and non-glue punctuation are real boundaries (no join)', () => {
        expect(seekTokenize('gpt 4')).toEqual(['gpt', '4']);   // space
        expect(seekTokenize('a,b')).toEqual(['a', 'b']);       // comma is not glue
    });
    it('a CJK fragment flushes the Latin run — never merged across the segmenter', () => {
        const terms = seekTokenize('東京-tokyo');
        expect(terms).toContain('tokyo');
        expect(terms.some(t => t.includes('東') && t.includes('tokyo'))).toBe(false);
    });
});

// ── Item 7: possessive 's strip ───────────────────────────────────────────
describe('seekTokenize — possessive strip (audit §7)', () => {
    it('drops a true possessive clitic (straight and curly apostrophe)', () => {
        expect(seekTokenize("ryan's")).toEqual(['ryan']);
        expect(seekTokenize("ryan's note")).toEqual(['ryan', 'note']);
        expect(seekTokenize('ryan’s note')).toEqual(['ryan', 'note']);  // U+2019
        expect(seekTokenize("boss's car")).toEqual(['boss', 'car']);
    });
    it("removes the junk bare 's' AND the junk glue-join item 6 would add", () => {
        // Pre-strip, "ryan's" tokenizes to ["ryan","s","ryans"] (bare-s junk +
        // glue-join junk via the apostrophe in GLUE_RUN); the strip kills both.
        expect(seekTokenize("ryan's")).not.toContain('s');
        expect(seekTokenize("ryan's")).not.toContain('ryans');
    });
    it('is letter-anchored — keeps a digit decade and recovers it via the join', () => {
        // "1990's" is a decade, not a possessive: 1990 kept, "1990s" recovered.
        expect(seekTokenize("1990's music")).toEqual(['1990', 's', '1990s', 'music']);
    });
    it("leaves contractions alone ('t is not the possessive 's)", () => {
        expect(seekTokenize("don't")).toContain('don');
    });
});

// ── Byte-identity: canonical tokens == the old SPACE_OR_PUNCT split ────────
// The glue-join is strictly ADDITIVE; on glue/possessive-free input the
// tokenizer must reproduce the exact MiniSearch-default split, because the
// coverage denominator + theoretical bound (bm25.ts) enumerate these tokens.
describe('seekTokenize — additive invariant (byte-identical canonical split)', () => {
    const refSplit = (t: string) => t.split(/[\n\r\p{Z}\p{P}]+/u).filter(Boolean);
    it('matches the reference split on glue-free, possessive-free input', () => {
        const cases = [
            'Hybrid search, BM25 + dense!',
            'a,b;c:d (e) [f] {g}',
            '  leading and trailing  ',
            'quote "word" end',
            'plain words only',
        ];
        for (const c of cases) expect(seekTokenize(c)).toEqual(refSplit(c));
    });
});
