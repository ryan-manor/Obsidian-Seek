// Lexical passage scoring for snippet-window selection — the sync half of the
// "better highlighting" plan ([[Seek Better Highlighting Research]] 2026-07-06).
//
// The old makeSnippet anchored the snippet window on the EARLIEST raw substring
// occurrence of ANY whitespace query token — stopwords included, no word
// boundary — so "bread not rising" anchored on the "not" inside "cannot" and
// "take up less disk space" on the "up" inside "grouped"; a dense-heavy match
// fell back to the chunk head with zero decoration. Nobody ships that: the
// industry standard (Lucene's unified highlighter) scores candidate passages
// with BM25 over sentences-as-documents and returns the best one. This module
// is that scorer, kept pure (no Obsidian imports) so it is unit-testable and so
// the sentence segmentation can be shared with the future fine-binary child-sig
// layer (tier-split gate PASSED 2026-07-06 — children reuse this segmentation).
//
// Scoring follows Lucene UH's PassageScorer: per-sentence BM25 with k1=1.2,
// b=0.75 and a CHARACTER pivot of 87 (Lucene's default — sentences are too
// short for token-length norms to behave), times a mild lead bias
// `1 + 1/ln(87 + start)` so early passages win ties (openings orient readers).
// IDF comes from the live BM25F index via termDocFraction, so the window
// chooser weighs terms exactly like the ranker that retrieved the chunk.
//
// REGEX DISCIPLINE: no lookbehind anywhere in this module — `(?<=…)` throws at
// PARSE time on iOS WKWebView before 16.4 and would take the whole plugin down
// with it (see tokenize.ts, camelCase split). The sentence segmenter is a
// forward character scan for the same reason.

import { processQueryTerm, depluralize, foldDiacritics } from './bm25';
import { seekTokenize } from './tokenize';

const K1 = 1.2;
const B = 0.75;
const PIVOT_CHARS = 87;

// IDF handed to a term whose document frequency is unavailable (index internals
// missing, or the term is OOV). When the WHOLE query degrades (no index yet),
// every term gets 1.0 and scoring falls back to uniform-weighted tf — still
// word-boundary + stopword-safe, i.e. never worse than distinct-term counting.
const DEFAULT_IDF = 1.0;

// Per-sentence cap on matches counted/marked for one term. BM25 tf saturates
// long before this; the cap only bounds pathological repetition (a log dump
// repeating one token hundreds of times) from growing the marks array.
const MAX_TF = 32;

const escapeRegExp = (s: string): string => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

export interface SentenceSpan {
    start: number;
    end: number;
}

export interface PassageTerm {
    // Word-boundary prefix matcher over LOWERCASED text: `\b(?:alt1|alt2)\w*`,
    // alternates ordered longest-first so the fuller surface form wins the
    // alternation. Global — callers set lastIndex before each exec run.
    re: RegExp;
    idf: number;
}

export interface Passage {
    start: number;   // best sentence's span (trimmed, offsets into `text`)
    end: number;
    score: number;
    marks: Array<[number, number]>;   // ascending [from,to) term hits inside the sentence
}

// Sentence segmentation: forward scan, no lookbehind (see header). Boundaries:
//   - any newline (markdown lines are structural — headings, list items,
//     paragraph breaks — and a passage window should never straddle them);
//   - a run of sentence-final punctuation [.!?…] (plus trailing closers like
//     quotes/brackets) followed by whitespace and then an uppercase letter,
//     digit, or opening quote/bracket. The next-char class check is what keeps
//     "e.g. foo" and "vs. the" from shattering mid-abbreviation — the same
//     cheap heuristic sentence BreakIterators lean on.
// Spans are whitespace-trimmed; empty spans are dropped. Deliberately NO
// min-length merging: a short heading that carries the query term SHOULD win
// the window (the snippet then extends into the text below it), and BM25's
// length norm already prices in the shortness.
const SENT_END = /[.!?…]/;
const CLOSERS = /[.!?…"'"')\]}»]/;
const OPENERS = /[\p{Lu}\p{N}"'"'([{«]/u;
const WS = /\s/;

export function segmentSentences(text: string): SentenceSpan[] {
    const spans: SentenceSpan[] = [];
    const n = text.length;
    const push = (s: number, e: number): void => {
        while (s < e && WS.test(text[s])) s++;
        while (e > s && WS.test(text[e - 1])) e--;
        if (e > s) spans.push({ start: s, end: e });
    };
    let start = 0;
    let i = 0;
    while (i < n) {
        const ch = text[i];
        if (ch === '\n') {
            push(start, i);
            i++;
            start = i;
            continue;
        }
        if (SENT_END.test(ch)) {
            let j = i + 1;
            while (j < n && CLOSERS.test(text[j])) j++;
            if (j >= n) {
                push(start, j);
                start = j;
                i = j;
                continue;
            }
            if (WS.test(text[j])) {
                let k = j;
                while (k < n && WS.test(text[k]) && text[k] !== '\n') k++;
                if (k >= n || text[k] === '\n' || OPENERS.test(text[k])) {
                    push(start, j);
                    start = k;
                    i = k;
                    continue;
                }
            }
            i = j;
            continue;
        }
        i++;
    }
    push(start, n);
    return spans;
}

// Build the query's scoring terms once per SEARCH (not per result — the same
// array serves every snippet). Tokenization is the canonical seekTokenize
// stream (derived:false — the glue-joined/camelCase recall forms never appear
// as literal words in text, so they cannot anchor a window; same contract as
// the bound/coverage enumerators, see tokenize.ts). Each token contributes two
// match alternates: its raw lowercased surface form AND its processed
// (folded+depluralized) form — prefix `\w*` alone cannot bridge both
// directions ("stories"→text "story" needs the processed stem; "story"→text
// "stories" needs the prefix), so the alternation carries both. Stopwords drop
// via processQueryTerm — the SAME stoplist BM25 indexes with, so the window
// chooser can't anchor on a word the ranker never scored. All-stopword queries
// fall back to keep-stopword processing (lowercase+fold+depluralize), mirroring
// bm25.ts's all-stopword search fallback: "the who" still anchors on "the who".
export function buildPassageTerms(
    query: string,
    idfOf: (processedTerm: string) => number,
): PassageTerm[] {
    const out: PassageTerm[] = [];
    const seen = new Set<string>();
    const add = (raw: string, processed: string): void => {
        if (seen.has(processed)) return;
        seen.add(processed);
        const alts = [...new Set([raw.toLowerCase(), processed])].filter(a => a.length >= 2);
        if (alts.length === 0) return;
        alts.sort((a, b) => b.length - a.length);
        const f = idfOf(processed);
        out.push({
            re: new RegExp(`\\b(?:${alts.map(escapeRegExp).join('|')})\\w*`, 'g'),
            idf: f > 0 ? Math.log(1 + (1 - f) / f) : DEFAULT_IDF,
        });
    };
    const tokens = seekTokenize(query, { derived: false });
    for (const raw of tokens) {
        if (raw.length < 2) continue;   // single chars would light up half the note
        const processed = processQueryTerm(raw);
        if (processed === null) continue;   // stopword
        add(raw, processed);
    }
    if (out.length === 0) {
        // All-stopword query: keep the literal terms (lowercase+fold+depluralize,
        // no stoplist) so the window can still anchor. df for stopwords is huge →
        // near-zero idf would zero the scorer, so give them the uniform default.
        for (const raw of tokens) {
            if (raw.length < 2) continue;
            add(raw, depluralize(foldDiacritics(raw.toLowerCase())));
        }
        for (const t of out) t.idf = DEFAULT_IDF;
    }
    return out;
}

// Score every sentence and return the best, or null when NO term matches
// anywhere — the caller falls back to the chunk head (dense-only matches keep
// today's behavior until the fine-binary child-sig layer lands and localizes
// them). Matching runs over a lowercased view; offsets transfer back because
// toLowerCase is length-preserving for the scripts the vault carries (the same
// bet highlight.ts already makes).
export function bestPassage(text: string, terms: PassageTerm[]): Passage | null {
    if (terms.length === 0) return null;
    const lower = text.toLowerCase();
    let best: Passage | null = null;
    for (const span of segmentSentences(text)) {
        const norm = K1 * (1 - B + B * ((span.end - span.start) / PIVOT_CHARS));
        let score = 0;
        const marks: Array<[number, number]> = [];
        for (const t of terms) {
            let tf = 0;
            t.re.lastIndex = span.start;
            let m: RegExpExecArray | null;
            while (tf < MAX_TF && (m = t.re.exec(lower)) !== null && m.index < span.end) {
                marks.push([m.index, m.index + m[0].length]);
                tf++;
            }
            if (tf > 0) score += t.idf * (tf * (K1 + 1)) / (tf + norm);
        }
        if (score <= 0) continue;
        score *= 1 + 1 / Math.log(PIVOT_CHARS + span.start);
        if (best === null || score > best.score) {
            marks.sort((a, b) => a[0] - b[0]);
            best = { start: span.start, end: span.end, score, marks };
        }
    }
    return best;
}

// The snippet window: [start, end) into `text`, `maxLen` chars anchored on the
// best passage's sentence START (a clean reading boundary — no more windows
// opening mid-word 40 chars before a match). One guard: if the best sentence is
// so long that its first term hit would fall OUTSIDE the window, slide the
// start up to 40 chars of left context before that hit (the old behavior's one
// virtue, kept for the wall-of-text case). No-match/no-term queries return the
// chunk head — byte-identical to the old fallback.
export function passageWindow(
    text: string,
    terms: PassageTerm[],
    maxLen: number,
): { start: number; end: number } {
    let start = 0;
    const best = bestPassage(text, terms);
    if (best !== null) {
        start = best.start;
        const firstMark = best.marks.length > 0 ? best.marks[0][0] : start;
        if (firstMark > start + maxLen - 20) {
            start = Math.max(start, firstMark - 40);
        }
    }
    return { start, end: Math.min(text.length, start + maxLen) };
}

// One combined global matcher over all term alternations, for decorating the
// RENDERED snippet DOM (search-modal walks text nodes with it — marks cannot be
// injected into the snippet STRING because it goes through MarkdownRenderer,
// where inserted syntax could corrupt constructs it lands inside). Null when
// there is nothing to decorate.
export function markPattern(terms: PassageTerm[]): RegExp | null {
    if (terms.length === 0) return null;
    const body = terms.map(t => t.re.source).join('|');
    return new RegExp(body, 'g');
}
