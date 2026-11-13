// Multi-field BM25 over chunks, backed by MiniSearch.
//
// History (see [[Seek Plugin - Initial Implementation Status]] Open Decision 2):
// v0.0.1 shipped a hand-rolled TS port of Tantivy's multi-field BM25. The
// hand-roll lost three things vs Tantivy: fuzzy matching, Unicode tokenization,
// and a subtly different BM25F field-length norm. The path of least resistance
// to recover most of that — without a Rust toolchain and 1.5 MB WASM sidecar —
// is MiniSearch, which is what the *original* production Python ranker's
// field-boost defaults trace back to (see `bm25.py:28-29` comment in
// [[Obsidian Semantic Search]]). So this isn't a downgrade from Tantivy
// parity; it's a return to the lineage those numbers came from.
//
// Contract preserved exactly:
//   - new MultiFieldBM25().fit(chunks) returns `this`
//   - getScores(query): Float64Array, length=chunks.length, indexed by
//     chunk position. Non-matching docs stay at 0.0.
//   - Raw (un-normalized) scores. fusion.ts's theoreticalNormBm25 is the
//     single normalization stage downstream.
//
// v0.1 shipped prefix: false, fuzzy: false to minimize score-magnitude
// drift vs the v0.0.1 baseline, pending eval. Both have since been priced
// and re-enabled as per-search toggles (the index defaults stay false;
// search.ts passes the live setting on every call):
//   - fuzzy: typo tolerance ON by default since 2026-06-09 (D&D typo eval);
//     edit distance scales by term length (≤2 exact / 3–5 = 1 / ≥6 = 2) and
//     skips CJK + digit-bearing tokens — see FUZZY_BY_LENGTH below.
//   - prefix: LAST query token only, ≥3 chars, ON by default since
//     2026-06-10 — see PREFIX_LAST_TOKEN below for the eval rationale.

import MiniSearch, { type Options } from 'minisearch';

// Content hash of the analyzer sources (bm25.ts + tokenize.ts + prop-normalize.ts
// + the MiniSearch version), injected by esbuild's `define`. It identifies the
// exact token space / scoring rules a persisted MiniSearch index was built
// under, so the persisted-index stamp (search.ts) can refuse to load a blob
// whose analyzer differs — any edit to these files changes the hash and
// auto-invalidates old blobs (→ refit). 'dev' under vitest/tsc, where the
// persist path is never exercised against a real IDB.
declare const __SEEK_ANALYZER_VERSION__: string;
export const ANALYZER_VERSION: string =
    typeof __SEEK_ANALYZER_VERSION__ !== 'undefined' ? __SEEK_ANALYZER_VERSION__ : 'dev';
import type { ChunkMeta } from './chunker';
import { seekTokenize, hasCjk } from './tokenize';
import { toDisplayForm } from './prop-normalize';
import {
    BM25F_PARAMS, Bm25fScratch, bm25fFieldB, bm25fFieldWeight, bm25fParamsKey, bm25fScoreTerm,
    type Bm25fFieldStats, type Bm25fParams,
} from './bm25f';

// Per-field BM25 term-frequency multipliers. Eval-tuned 2026-06-08 on the
// 482-q old-log personal eval (with the coverage navigational boost active):
// these were the user-facing 3/2/1.5 sliders, but once coverage is in their
// marginal effect is small (~+0.004 nDCG@10), so the sliders were removed and
// the swept values baked in here. 10/6/3 captures the gain (0.8714 vs the
// 0.8727 all=10 max) while keeping `tags` modest (3×) so noisy-tag vaults
// aren't over-weighted. (No `page_type` entry: `pageType` is not a universal
// field, so it gets no dedicated field — it rides the generic `properties`
// field by name like any other scalar prop. See ~/seek-personal-eval.)
export const DEFAULT_FIELD_BOOSTS: Record<string, number> = {
    title: 10.0,
    aliases: 6.0,
    tags: 3.0,
    // Body text. Raised 1.0 → 3.0 (2026-06-27 field-weight re-eval): the 10:1
    // title:content ratio was tuned on the pre-de-franken ("franken" double soft-AND)
    // channel; after de-franken + WAND-bound normalization the body is under-weighted.
    // Re-sweep over Example Vault + the enriched android/gaming BEIR sets (gbq4 dense held
    // fixed, coverage_pow=2) showed content=3 a robust cross-corpus win (BEIR
    // +0.005–0.012 nDCG@10, Example Vault neutral-to-+). See the re-eval note in the vault.
    content: 3.0,
    // Frontmatter property VALUES (searchableProperties setting). The field is
    // only INDEXED when the toggle is on; a boost entry for an unindexed field
    // is inert everywhere (MiniSearch ignores it, getQueryBound sees df=0).
    // 2.0 = the knee of the harness boost sweep (props_field_arm 2026-06-11):
    // captures +0.059, personal 483-q net-zero; ≥3 starts paying noise on
    // person-name/wikilink-valued props.
    properties: 2.0,
    // Section heading path (headingsField setting; same inert-when-unindexed
    // contract as properties). Harness gate (headings_field_arm 2026-06-12)
    // was a WASH with recency OFF: topical heading-only wins (+0.5) cancelled
    // by dated-series near-tie perturbation — the class the live ε-recency
    // tiebreaker re-orders. Shipped as an experimental toggle for the live
    // A/B the harness can't model. 3.0 = the sweep's best personal cell.
    headings: 3.0,
};

// Okapi-derived BM25+ params. MiniSearch's `k` is the conventional `k1`
// (term-frequency saturation point); `b` is length normalization; `d` is
// BM25+'s lower-bound delta (ensures very long documents don't score zero
// for matching terms). Note: MiniSearch's defaults are k=1.2 b=0.7 d=0.5,
// but the v0.0.1 hand-rolled implementation used b=0.75 — we set it
// explicitly here to preserve the v0.0.1 baseline.
const BM25_PARAMS = { k: 1.2, b: 0.75, d: 0.5 } as const;

// Coordination-level exponent on the coverage soft-AND. The fused lexical channel
// is `raw_bm25 · coverage^P` where coverage = (matched query terms / total) — see
// search.ts where it's applied (getScoresWithCoverage returns the raw fraction; the
// exponent is a RANKING decision, kept out of the measurement). P=1 is plain
// coordination (Lucene's old `coord`); the de-franken (0749eac) left it at 1 and
// regressed natural "[thing] in [place]" queries — a doc matching only the rare
// place token ("sf", "austin") normalized to a near-max lexical score and out-ranked
// real answers. P=2 restores the franken-era m²/T penalty magnitude as an EXPLICIT
// knob (not the old accidental ×quality squaring), hardening the partial-match
// discount so full-coverage docs win. Eval-validated 2026-06-27 (~/seek-eval-pack
// cross-corpus sweep): MyVault bars nDCG@10 0.302→0.457, captures 0.445→0.496,
// personal-eval guardrail flat-to-up (place stratum pinned 1.000), and a slight gain
// out-of-domain on BEIR android (+0.006). Rejected alternatives in the same sweep:
// length-conditional exponent (no gain — queries are short) and conditional-%
// minimum_should_match (buried "bars in sf" rank 5→69 and regressed android −0.012).
export const BM25_COVERAGE_POW = 2 as const;


// Prefix-match predicate: expand ONLY the final query token, and only when
// it's ≥3 chars. MiniSearch calls this per processed query term (stopwords
// already dropped) and, when it returns true, derives every vocab term
// extending it at weight 0.375·len(term)/len(derived), with the match
// attributed back to the source term (coverage + the ×quality multiplier
// see it). Eval-tuned 2026-06-10 (~/seek-eval-pack prefix_arm.py, WS2.3
// token-exact cache, α=0.80):
//   - last-only ≥3 (+syn layer pending): personal 0.8666→0.8730 bin nDCG@10;
//     wins are literally truncated/typeahead queries ("amster"→Amsterdam,
//     "roadmap for concep"); cap-004 "lr ac rearch" target rank 5→2.
//   - scope=all REJECTED: −0.0149 on the D&D desc stratum (common-word
//     prefix noise) and ~1813 derived terms/query on the code corpus
//     (identifier vocab fan-out) for zero relevance gain. Last-only is
//     exactly 0.0000 on both stress sets at every min-len, ~28 derived/q.
// The rationale is a typing prior: a mid-query token is a word the user
// FINISHED; only the final token is plausibly still being typed.
// Like fuzzy, derived terms score with their own idf, so they're
// numerator-only against the exact-term bound (fusion clips at 1).
export const PREFIX_MIN_CHARS_DEFAULT = 3;

// Floor-parameterized variant (2026-11-11): BM25 uses the eval-tuned 3 above;
// the title-boost gate (fusion.ts) uses 1. Below 3 chars a LONE token never
// expands (a bare "g" would fan out to every g-word); it must follow at least
// one finished token, which is what anchors precision.
export function prefixLastTokenFor(minChars: number) {
    const min = Math.max(1, Math.floor(minChars));
    return (term: string, i: number, terms: string[]): boolean =>
        i === terms.length - 1
        && term.length >= min
        && (term.length >= PREFIX_MIN_CHARS_DEFAULT || terms.length > 1);
}

export const PREFIX_LAST_TOKEN = prefixLastTokenFor(PREFIX_MIN_CHARS_DEFAULT);

// Fuzzy typo-tolerance predicate, scaled by TERM length (audit 2026-06-18 §1)
// with two exemptions. Returns the absolute max edit distance MiniSearch may
// allow when deriving fuzzy matches for `term`, or false for exact-only.
// Replaces the old flat edit-1: edit-1 on a 2-char token ("ml","ai","db","r2",
// "go") has a DENSE neighborhood (al/mr/me/…), uncontrolled expansion whose
// precision damage is invisible to aggregate nDCG but very visible to a human
// typing a short query — the audit's unifying frame.
//
// Length ladder = Lucene/Elasticsearch `fuzziness: AUTO`, the canonical typo
// model: ≤2 → exact, 3–5 → edit-1, ≥6 → edit-2 (a long high-IDF term has a
// sparse edit-2 neighborhood, so two typos still resolve to it without a noise
// explosion). The ≤2 lower bound is the actual relevance FIX; the ≥6 edit-2
// rung adds long-word typo recall.
//
// Two exemptions return false BEFORE the ladder:
//   - hasCjk: edit-1 on a dictionary-segmented CJK word-piece is a synonym
//     explosion, not a typo model (2026-06-10 CJK gate; measured Belebele ja
//     0.094 WITH fuzzy vs 0.741 without, fused 0.692 vs 0.928 — the explosion
//     single-handedly turned the segmentation win into a regression). UNCHANGED.
//   - contains a digit (audit §2): "2024" edit-1 hits 2023/2025/2014/1024 — a
//     date/version collision generator in a daily-notes vault. IDs/numbers are
//     typed precisely; a wrong digit is a different thing, not a typo. Broad
//     contains-digit (not all-digits) so v2/gpt4/k8s stay exact too.
// MiniSearch passes (term, i, terms); only `term` is needed — audit §3
// (terminal-token prefix-XOR-fuzzy) was dropped, so fuzzy and prefix stay
// independent and PREFIX_LAST_TOKEN is unchanged.
export const FUZZY_BY_LENGTH = (term: string): number | false => {
    if (hasCjk(term)) return false;
    if (/\d/.test(term)) return false;
    if (term.length <= 2) return false;
    return term.length <= 5 ? 1 : 2;
};

// Per-search overrides accepted by getScores/getScoresWithCoverage —
// MiniSearch shallow-merges these over the index's baked-in searchOptions.
export interface SearchOverrides {
    boosts?: Record<string, number>;
    fuzzy?: number | boolean | ((term: string, i: number, terms: string[]) => number | boolean);
    prefix?: boolean | ((term: string, i: number, terms: string[]) => boolean);
    // Query-side alias-dictionary expansion (synonyms.ts, default OFF). Each
    // trigger term in the query also queries its mates at `weight` — see the
    // synonym block in getScoresWithCoverage for the mechanics and the
    // semantic deltas vs the eval harness.
    synonyms?: {
        map: Map<string, string[]>;
        weight: number;
        // v2 Stage 1: per-QUERY phrase-trigger mates, resolved by the CALLER
        // (search.ts matchPhrases) because phrase detection cannot ride the
        // per-token processTerm hook — each hook call sees ONE token, and
        // adjacency is gone by then. Each entry injects `mate` at most once
        // (flushed on the first surviving query term) and credits EVERY
        // listed source slot for coverage. `sources` are in processQueryTerm
        // space and are terms the query actually contains.
        phraseMates?: Array<{ mate: string; sources: string[] }>;
    };
    // BM25F field model (bm25f.ts; settings.bm25f, default ON). Present ⇒ score
    // with BM25F under these params instead of MiniSearch's per-field sum: the
    // scores, the coverage and the Σ UB(t) bound ALL come from the BM25F scorer,
    // and `boosts` is ignored (BM25F reads `weights`). Absent ⇒ the legacy
    // per-field-sum path, byte-identical to before the flag existed (the A/B arm).
    bm25f?: Bm25fParams;
}

// MiniSearch's default derived-term weights (dist v7.2.0 defaultSearchOptions
// .weights) and fuzzy-fraction cap (maxFuzzy). Seek never overrides them, so the
// BM25F path — which runs the query specs itself — uses the same constants to
// derive prefix/fuzzy terms exactly as mini.search() would.
const MS_PREFIX_WEIGHT = 0.375;
const MS_FUZZY_WEIGHT = 0.45;
const MS_MAX_FUZZY = 6;

// MiniSearch termToQuerySpec + executeQuerySpec's fuzzy resolution (dist v7.2.0):
// the absolute max edit distance a fuzzy spec allows for `term`, 0 = exact only.
// Shared by the BM25F query loop and the fuzzy-aware bound so both derive the
// same term set.
function fuzzyMaxDist(spec: SearchOverrides['fuzzy'], term: string, i: number, terms: string[]): number {
    const fz = typeof spec === 'function' ? spec(term, i, terms) : (spec || false);
    if (!fz) return 0;
    const f = fz === true ? 0.2 : fz;
    return f < 1 ? Math.min(MS_MAX_FUZZY, Math.round(term.length * f)) : f;
}

// The slice of MiniSearch 7.2.0 internals the BM25F path reads (all private in the
// typings; verified in dist). Same defensive posture as termDocFraction /
// indexedQueryTerms: each access is checked, and a shape mismatch (a future
// MiniSearch rename) drops getScoresWithCoverage back to the legacy per-field path
// (one console warning) rather than throwing or zeroing the lexical channel.
type Postings = Map<number, Map<number, number>>;   // fieldId → shortId → tf
interface MiniInternals {
    _index: {
        get(term: string): Postings | undefined;
        atPrefix(prefix: string): Iterable<[string, Postings]>;
        fuzzyGet(term: string, maxDist: number): Map<string, [Postings, number]>;
        [Symbol.iterator](): Iterator<[string, Postings]>;
    };
    _fieldIds: Record<string, number>;
    _documentIds: Map<number, string>;
    _idToShortId: Map<string, number>;
    _documentCount: number;
    _nextId: number;
}
function popcount32(x: number): number {
    x = x - ((x >>> 1) & 0x55555555);
    x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
    return (Math.imul((x + (x >>> 4)) & 0x0f0f0f0f, 0x01010101) >>> 24);
}

function miniInternals(mini: unknown): MiniInternals | null {
    const m = mini as Partial<MiniInternals> | null;
    if (!m || !m._index || typeof m._index.get !== 'function'
        || typeof m._index.atPrefix !== 'function' || typeof m._index.fuzzyGet !== 'function'
        || !m._fieldIds || !(m._documentIds instanceof Map) || !(m._idToShortId instanceof Map)
        || typeof m._documentCount !== 'number' || typeof m._nextId !== 'number') return null;
    return m as MiniInternals;
}

// BM25F corpus state, keyed by MiniSearch SHORT doc id (the postings' key space):
// the per-field processed-token lengths (bm25f.ts UNITS) + their running sums, and
// the short-id → frame-row map. Lengths are built lazily from the postings on the
// first BM25F use after a fit/fromJSON, then maintained INCREMENTALLY by add /
// remove / removeExact (a delta never pays the full pass). The row map is cheap
// (O(live docs)) and simply rebuilt after any mutation.
interface Bm25fState {
    len: Float64Array[];        // by fieldId, indexed by short id
    lenSum: number[];           // by fieldId, over live docs
    rowOf: Int32Array;          // short id → frame row; -2 = live but unplaced (tolerant-load ghost); -1 = dead
    rowsDirty: boolean;
}

// Lucene/Elasticsearch default English stoplist (the 33-word `_english_` set).
// Matched deliberately: it's the analyzer lineage of the Anserini BM25 baseline
// the BEIR study compared against (see [[Seek notes]] Relevance, 2026-06-03 —
// Seek's lexical channel scored ~0.23-0.29 nDCG@10 on CQADupstack vs Anserini's
// ~0.38, the gap being the missing English analyzer). IDF already discounts these
// terms, so the win is modest and mostly removes match noise (stemming would be
// the larger lever); the bigger relevance lever is raising the dense fusion
// weight, tracked separately.
export const ENGLISH_STOPWORDS = new Set<string>([
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'for', 'if', 'in',
    'into', 'is', 'it', 'no', 'not', 'of', 'on', 'or', 'such', 'that', 'the',
    'their', 'then', 'there', 'these', 'they', 'this', 'to', 'was', 'will', 'with',
]);

// Exception tables for depluralize (audit 2026-06-09 §5). These are the kStem
// PRINCIPLE — only emit a stem that is a real word — applied at the scale of a
// plural-only normalizer instead of bundling a 30k-word lexicon. Each table
// fixes one verified bug class where the singular and plural landed on
// DIFFERENT stems (silent never-co-match), or a reduction collided with a
// proper name (alias->alia "Alia", lens->len "Len").
//
// A Map, NOT an object literal: keys are raw user tokens, and an object lookup
// would resolve tokens like "constructor" to Object.prototype members.
const IRREGULAR_PLURALS = new Map<string, string>([
    // Greek/Latin -is plurals: the -ses rule gave analyses->analys while the
    // -is exclusion (correctly) kept analysis intact — two different stems.
    ['analyses', 'analysis'],
    ['theses', 'thesis'],          // -ses rule gave "these", a stopword: indexed docs lost the term entirely
    ['crises', 'crisis'],
    ['diagnoses', 'diagnosis'],    // noun reading chosen over verb ("she diagnoses"); notes skew noun
    ['prognoses', 'prognosis'],
    ['hypotheses', 'hypothesis'],
    ['parentheses', 'parenthesis'],
    ['syntheses', 'synthesis'],
    ['emphases', 'emphasis'],
    ['oases', 'oasis'],
    ['neuroses', 'neurosis'],
    ['psychoses', 'psychosis'],
    // "axes"/"bases" are deliberately ABSENT: axe/axis and base/basis are both
    // live; either mapping breaks the other word, so they keep the old
    // (symmetric, self-consistent) behavior.
    ['lenses', 'lens'],            // -ses rule gave "lense"; singular guard below keeps lens (vs name "Len")
]);

// s-final singulars the generic -s rule must not strip. Their plurals already
// reduce correctly via the -ses rule (aliases->alias, biases->bias), so guarding
// the singular side restores co-match AND removes the name collisions in one
// move. We can't just add "-as" to the suffix exclusions — that would stop
// bananas/ideas/sofas from reducing.
const S_FINAL_SINGULARS = new Set<string>([
    'alias', 'bias', 'atlas', 'canvas', 'lens',
    'news', // no plural; stripping gave "new", a high-frequency cross-word collision
]);

// -ie nouns whose plural the -ies rule mangled (movies->movy vs movie). If the
// word minus its final "s" is here, strip only the "s".
const IE_NOUNS = new Set<string>([
    'movie', 'cookie', 'zombie', 'calorie', 'genie', 'pixie', 'sortie',
    'selfie', 'smoothie', 'hoodie', 'newbie', 'rookie', 'goalie', 'veggie',
    'birdie', 'foodie', 'freebie', 'indie', 'junkie', 'oldie', 'quickie',
    'talkie', 'townie', 'yuppie',
]);

// Light plural -> singular normalizer, applied symmetrically at index AND query
// time (inside processTerm) so "cuisines" matches "cuisine", "cars" matches "car".
// Deliberately NOT a full stemmer: Porter2 was tested and rejected on CQADupstack
// (net -0.0019), but this plural-only normalizer scored +0.0276 nDCG@10 on the
// DBpedia BM25 channel (2026-06-07 study). The rule set is intentionally simple;
// because it runs on both sides, even an imperfect stem (e.g. "series"->"sery")
// still matches itself. The exception tables above handle the verified cases
// where symmetry was NOT enough — singular and plural reducing to different
// stems — plus the proper-name collisions (audit 2026-06-09 §5).
export function depluralize(w: string): string {
    if (w.length <= 3) return w;
    const irregular = IRREGULAR_PLURALS.get(w);
    if (irregular !== undefined) return irregular;
    if (S_FINAL_SINGULARS.has(w)) return w;
    if (w.endsWith('ies') && w.length > 4) {
        const ie = w.slice(0, -1);                                            // movies -> movie
        if (IE_NOUNS.has(ie)) return ie;
        return w.slice(0, -3) + 'y';                                          // countries -> country
    }
    if (w.endsWith('sses')) return w.slice(0, -2);                            // classes -> class
    if ((w.endsWith('ches') || w.endsWith('shes') || w.endsWith('xes')
        || w.endsWith('zes') || w.endsWith('ses')) && w.length > 4) {
        return w.slice(0, -2);                                                // boxes->box, dishes->dish, aliases->alias
    }
    if (w.endsWith('s') && w.length > 3
        && !(w.endsWith('ss') || w.endsWith('us') || w.endsWith('is') || w.endsWith('os'))) {
        return w.slice(0, -1);                                                // cars -> car
    }
    return w;
}

// S2 per-field stopword exemption (Three-Lens S2, 2026-06-10): title and
// aliases are NAME fields — "Will", "The Saloon", "By Yu" are identifiers, not
// function words. Dropping stopwords there killed the 10×-boosted title
// channel for exactly the highest-frequency query class in a People-page
// vault (person lookup). MiniSearch passes the field name as processTerm's
// 2nd argument AT INDEX TIME (verified in dist v7.2.0: `processTerm(term,
// field)` in addDocument), so exempting these fields keeps their stopword
// terms indexed. Content/tags keep the stoplist — prose stopwords are still
// noise.
const STOPWORD_EXEMPT_FIELDS = new Set(['title', 'aliases']);

// Latin diacritic fold (audit 2026-06-18 §4): decompose to NFD and strip the
// combining marks (\p{M}) so "café"/"josé"/"zürich"/"naïve"/"pokémon"/"andrés"
// co-match their unaccented spellings — the single clearest miss for non-English
// names/loanwords, which every Seek eval corpus under-samples (so it scores
// ~0.000 on the ASCII harness and only helps the real population). Called INSIDE
// processTerm, so it is symmetric across index and query like the stoplist and
// depluralizer. GUARDED on !hasCjk: NFD DECOMPOSES precomposed Hangul into
// conjoining jamo (mangles the match key) and splits a precomposed dakuten kana
// (ガ U+30AC → カ + U+3099, so stripping the mark turns GA into KA — a different
// word); the CJK channel owns its own term space (tokenize.ts) and must never be
// folded. Marks-ONLY: stroke/bar letters (ł, ø, đ, ħ) have no canonical
// decomposition and pass through unchanged — the same miss as today, not a
// regression. Pure-ASCII-inert (NFD is a no-op on ASCII and \p{M} matches
// nothing), so every English eval term is byte-identical, and idempotent.
export function foldDiacritics(t: string): string {
    if (hasCjk(t)) return t;
    return t.normalize('NFD').replace(/\p{M}+/gu, '');
}

// MiniSearch's `processTerm` runs at BOTH index and query time, so the stoplist
// (and now the depluralizer) stay symmetric across the two with no chance of drift.
// It replaces MiniSearch's default processor (which only lowercases), so we lowercase
// here too. Stopwords are matched on the raw lowercased term BEFORE depluralization
// (mirrors the eval). Returning null drops the term from both index and query.
//
// At QUERY time MiniSearch calls this with no fieldName → stopwords drop from
// queries by default; the all-stopword fallback (keepStopwordsProcessTerm,
// applied per-call in getScoresWithCoverage) is the query-side complement —
// without it the exempt index terms would be unreachable ("will" indexed but
// never queried).
function processTerm(term: string, fieldName?: string): string | null {
    // Fold right after lowercase — BEFORE the stopword check (so an accented
    // stopword spelling like "às"/"öf" drops consistently on both sides) and
    // BEFORE depluralize (so the exception tables only ever see ASCII).
    const t = foldDiacritics(term.toLowerCase());
    if (ENGLISH_STOPWORDS.has(t)
        && !(fieldName !== undefined && STOPWORD_EXEMPT_FIELDS.has(fieldName))) {
        return null;
    }
    return depluralize(t);
}

// Query-side term processing, exported for synonyms.ts: a synonym class
// member must land in the SAME term space queries are matched in (lowercase,
// stoplist, depluralize), or trigger lookups would silently never fire.
// Stopword members drop (no field exemption): conservative — never inject a
// stopword as a mate, even though title/aliases index them (S2).
export function processQueryTerm(term: string): string | null {
    return processTerm(term);
}

// Query-side fallback processor: stoplist OFF (lowercase + depluralize only).
// Used when the WHOLE query is stopwords — mirrors fusion.ts titleMatchBoost's
// all-stopword fallback, so the two channels agree on when a stopword is a
// name. A query with ≥1 content word keeps the shipped drop behavior
// unchanged (zero effect on existing prose queries).
function keepStopwordsProcessTerm(term: string): string {
    // MUST fold too (audit §4): the index is ALWAYS built via processTerm (which
    // folds), so an all-stopword query routed through this fallback would query a
    // different term space than the index unless it folds identically.
    return depluralize(foldDiacritics(term.toLowerCase()));
}

// Tokenization is the shared seekTokenize (tokenize.ts): MiniSearch's default
// space/punct split + CJK dictionary segmentation. It is passed to MiniSearch
// as the `tokenize` option in fit() AND used to enumerate the DISTINCT
// processed query terms feeding the coverage denominator — the actual matching
// is still done by MiniSearch, through the same function, so they cannot
// drift. NOTE (corrected 2026-06-09 review): under the theoretical BOUND
// normalization the denominator does NOT cancel — that was only true of
// the old per-query max-division, where a constant-across-docs factor washed
// out. Dividing by the bound is dividing by a query-level constant the coverage
// factor is NOT part of, so 1/totalTerms scales the whole lexical channel
// against dense. Faithfulness to the indexed split IS load-bearing.

function distinctQueryTerms(query: string): Set<string> {
    const out = new Set<string>();
    for (const raw of seekTokenize(query)) {
        const t = processTerm(raw);
        if (t) out.add(t);
    }
    if (out.size > 0) return out;
    // All-stopword fallback (S2): keep the literal terms so the coverage
    // denominator matches what the fallback search actually queries.
    for (const raw of seekTokenize(query)) {
        out.add(keepStopwordsProcessTerm(raw));
    }
    return out;
}

// True when drop-mode processing leaves NOTHING of a non-empty query — the
// trigger for the all-stopword fallback across search, coverage, and bound.
function isAllStopwordQuery(query: string): boolean {
    let sawToken = false;
    for (const raw of seekTokenize(query)) {
        sawToken = true;
        if (processTerm(raw) !== null) return false;
    }
    return sawToken;
}

export function extractNoteName(chunk: ChunkMeta): string {
    let t = chunk.title;
    if (t.includes(' > ')) t = t.split(' > ')[0];
    if (t.includes(' | ')) t = t.split(' | ')[0];
    return t.trim();
}

function extractAliasesText(chunk: ChunkMeta): string {
    const aliases = chunk.metadata?.aliases;
    if (Array.isArray(aliases) && aliases.length > 0) return aliases.join(' ');
    let t = chunk.title;
    if (t.includes(' > ')) t = t.split(' > ')[0];
    const parts = t.split(' | ');
    return parts.length > 1 ? parts.slice(1).join(' ') : '';
}

function extractTagsText(chunk: ChunkMeta): string {
    const tags = chunk.metadata?.tags;
    if (Array.isArray(tags)) return tags.join(' ');
    return '';
}

// Section heading path -> plain text for the `headings` field. The WS3
// chunker is the only writer of heading_path, so this is the full ancestor
// chain ("H1 H2 H3"), per chunk — the lexical mirror of what embedInput's
// hierarchical title already gives the dense channel. Without this field,
// heading words are BM25-invisible: extractNoteName strips the path from
// `title` and the chunker excludes the heading line from section content.
// (?? [] guards chunks indexed before heading_path existed.)
export function extractHeadingsText(chunk: ChunkMeta): string {
    return (chunk.heading_path ?? []).join(' ');
}

// Pure machinery frontmatter keys — identity/UI/plumbing whose VALUES are junk
// for ANY semantic surface. Exported and SHARED with chunker's buildDenseSuffix
// so the two "leftover" surfaces (this properties field + the dense suffix) can
// never drift on what counts as machinery. Relevance Quality Audit 2026-06-29
// finding #2: text-valued keys like `icon`/`cssclasses` had been leaking into
// the dense suffix because only THIS field name-excluded them.
export const MACHINERY_KEYS = new Set([
    'icon', 'coordinates', 'cssclasses', 'cssclass', 'protected',
    'version', 'completed', 'completeddate', 'position', 'banner', 'banner_y',
]);

// Keys excluded from the searchable-properties field = machinery (above) PLUS
// keys already carried by a dedicated field (tags/aliases → own fields;
// created/modified/datelink → date filters) so they aren't double-indexed into
// this catch-all. Everything else (placeLoc, placeType, context, status,
// pageType, …) is in — the harness gate measured the generic
// everything-but-machinery posture, not a curated allowlist. `pageType` used to
// be excluded here (it had a dedicated `page_type` field); that field is gone,
// so it now flows through this catch-all by name like any other scalar prop.
// Compared lowercased against every key, scalar or list-valued alike —
// extractPropertiesText below folds a list value's items in per-element (audit
// R2 batch2 #3), so a list-valued key excluded here (e.g. a future dedicated
// field) is excluded the same way a scalar one would be.
const PROPERTY_EXCLUDE_KEYS = new Set([
    'tags', 'aliases', 'alias', 'created', 'modified', 'datelink',
    ...MACHINERY_KEYS,
]);
// Anchored BOTH ends (audit R2 batch2 #1): a bare date or ISO date+time is
// inert (dates are queryable via [key:value] filters, which read the same
// backing store untouched) but a date-PREFIXED value that carries trailing
// free text ("2026-06-29 Milan departure") is not — the old prefix-only test
// (no trailing `$`) matched on the date alone and dropped the whole value,
// silently swallowing "Milan departure" out of the lexical channel.
const PROPERTY_DATE_RE = /^\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?$/i;
const PROPERTY_NUM_RE = /^-?[\d., ]+$/;

// Property VALUES -> plain text for the `properties` field. Not a custom
// analyzer — just a normalizer in front of the standard pipeline (the field's
// terms then run through seekTokenize + processTerm like any other field).
// Wikilinks collapse to their DISPLAY form — the target basename, NOT the
// matcher-style path+alias unwrap ("[[Notes/.../Zurich|Zurich]]" -> "Zurich",
// not the "Notes Personal Places Zurich Zurich" keyword-stuffing that inflated
// a boosted field with folder tokens and doubled names; see [[Seek Index
// Processing Audit]] and toDisplayForm). Date/number-only values are dropped
// (they'd only feed the index junk terms; dates are queryable via [key:value]
// filters, which read the same backing store untouched).
export function extractPropertiesText(chunk: ChunkMeta): string {
    const props = chunk.metadata?.properties;
    if (!props) return '';
    const vals: string[] = [];
    for (const [key, raw] of Object.entries(props)) {
        if (PROPERTY_EXCLUDE_KEYS.has(key.toLowerCase())) continue;
        // List values (stored as string[] since v10 for the FILTER matcher) used
        // to be skipped here entirely, leaving them dense-suffix-only — a
        // relatedPages-style list was BM25-invisible even with a scalar sibling
        // property fully indexed (audit R2 batch2 #3). Folded in per-item through
        // the SAME normalize/type-drop gates as a scalar value, joined with the
        // rest — consistent with how the scalar half of this field is built.
        const items = Array.isArray(raw) ? raw : [raw];
        for (const item of items) {
            const v = toDisplayForm(String(item).replace(/^["']|["']$/g, ''));
            if (!v || PROPERTY_DATE_RE.test(v) || PROPERTY_NUM_RE.test(v)) continue;
            vals.push(v);
        }
    }
    return vals.join(' ');
}

// Shape of a single document handed to MiniSearch. Per-chunk granularity:
// each chunk is one document, fields that are file-level metadata (title,
// aliases, tags) are duplicated across chunks of the same note —
// same shape as the v0.0.1 hand-rolled index, by design.
interface IndexedDoc {
    chunk_id: string;
    title: string;
    aliases: string;
    tags: string;
    content: string;
    // Only populated (and only listed in `fields`) when fit() runs with
    // searchableProperties on — see fit().
    properties?: string;
    // Same contract, gated by headingsField.
    headings?: string;
}

export class MultiFieldBM25 {
    private mini: MiniSearch<IndexedDoc> | null = null;
    private idToIdx = new Map<string, number>();
    private chunkCount = 0;
    // WAND/MaxScore per-term upper bounds: processed term → max achievable EXACT
    // single-term score over the LIVE index (termUpperBound). Memoized on the
    // default-boost path and CLEARED on every mutation (fit/fromJSON/add/remove/
    // vacuum) so the bound always reflects live postings; a fresh search-based
    // recompute then honors discard() tombstones synchronously, no vacuum needed.
    private termUpperBounds = new Map<string, number>();
    // The BM25F twin of termUpperBounds: params key (bm25fParamsKey) → term → max
    // BM25F single-term score over the live index. Keyed by params VALUE so a weight
    // change (settings or a seek:search override) never reads a bound computed under
    // other weights; a few param sets are kept so an A/B can alternate without
    // thrashing. Cleared with termUpperBounds on every mutation (clearBounds).
    private bm25fUpperBounds = new Map<string, Map<string, number>>();
    // BM25F corpus state (lengths + row map), null until first BM25F use after a
    // fit/fromJSON — see Bm25fState. The scratch buffers are reused across queries.
    private bf: Bm25fState | null = null;
    private readonly bfScratch = new Bm25fScratch();
    private warnedBm25fInternals = false;
    // Cumulative 'version_conflict' warnings from MiniSearch (see the logger in
    // buildMiniOptions): a removeExact() whose reconstructed doc disagreed with
    // what was indexed. removeExact snapshots this around mini.remove() to report
    // a per-call mismatch verdict.
    private removalMismatchCount = 0;
    // Index-shape opts the live index was built with (searchableProperties /
    // headingsField). Stored so incremental add() rebuilds each doc with exactly
    // the same field set fit()/fromJSON() used — a field-set mismatch would index
    // the new chunk under a different shape than the rest. The caller (search.ts)
    // also forces a full rebuild when these settings flip, so add() never has to
    // reconcile a shape change mid-stream.
    private withProps = false;
    private withHeadings = false;
    fieldBoosts: Record<string, number>;

    constructor(fieldBoosts?: Record<string, number>) {
        this.fieldBoosts = fieldBoosts ?? DEFAULT_FIELD_BOOSTS;
    }

    // opts.searchableProperties: index frontmatter property values as a 6th
    // field (extractPropertiesText). opts.headingsField: index the section
    // heading path as a field (extractHeadingsText). Unlike fuzzy/prefix these
    // are INDEX shape changes, not per-call options — the caller's fit cache
    // must key on them (search.ts bm25CacheProps/bm25CacheHeadings), but the
    // toggles still need no embedding reindex: chunks already carry
    // metadata.properties and heading_path.
    // Bodies are supplied separately (v8 frame-lite: the resident frame holds
    // ChunkMeta, body text lives in the chunk_body store). The caller pre-fetches
    // them via IndexStore.getBodiesMap only on a (rare) refit; a missing id maps
    // to '' (same as the old `c.content ?? ''`). fromJSON below needs no bodies.
    fit(chunks: ChunkMeta[], bodies: ReadonlyMap<string, string>, opts?: { searchableProperties?: boolean; headingsField?: boolean }): this {
        const { mini, docs } = this.prepareFit(chunks, bodies, opts);
        mini.addAll(docs);
        return this;
    }

    // fit() with a thread-yield every `sliceSize` docs, for main-thread callers
    // (the cold-search/warm refit). PROVABLY identical output to fit(): MiniSearch
    // 7's addAll is a plain per-document add() loop, so slicing preserves insertion
    // order and therefore internal doc ids, postings, and toJSON byte-for-byte.
    // The instance is unusable mid-flight (partial index) — callers must assign it
    // to their cache field only AFTER this resolves, exactly like fit()'s pattern.
    async fitAsync(chunks: ChunkMeta[], bodies: ReadonlyMap<string, string>, opts: { searchableProperties?: boolean; headingsField?: boolean } | undefined, yieldFn: () => Promise<void>, sliceSize = 500): Promise<this> {
        const { mini, docs } = this.prepareFit(chunks, bodies, opts);
        for (let i = 0; i < docs.length; i += sliceSize) {
            mini.addAll(docs.slice(i, i + sliceSize));
            if (i + sliceSize < docs.length) await yieldFn();
        }
        return this;
    }

    // Shared front half of fit()/fitAsync(): reset the index and build the docs
    // in chunk order. Split out so the two paths can never drift on analyzer
    // options or doc construction. Returns the fresh MiniSearch instance (also
    // assigned to this.mini) so callers add docs without a nullability dance.
    private prepareFit(chunks: ChunkMeta[], bodies: ReadonlyMap<string, string>, opts?: { searchableProperties?: boolean; headingsField?: boolean }): { mini: MiniSearch<IndexedDoc>; docs: IndexedDoc[] } {
        this.withProps = opts?.searchableProperties === true;
        this.withHeadings = opts?.headingsField === true;
        this.mapChunkIds(chunks);
        const mini = new MiniSearch<IndexedDoc>(this.buildMiniOptions(this.withProps, this.withHeadings));
        this.mini = mini;
        this.clearBounds();   // fresh index → drop memoized per-term UBs
        this.bf = null;       // BM25F lengths rebuild lazily from the new postings
        return { mini, docs: chunks.map(c => this.buildDoc(c, bodies.get(c.chunk_id) ?? '')) };
    }

    // Build the MiniSearch document for one chunk + its body, with the field set
    // (properties/headings) the index was fit with. Shared by fit() and the
    // incremental add() so a row appended live is byte-for-byte the same document
    // a full refit would have produced for that chunk.
    private buildDoc(c: ChunkMeta, body: string): IndexedDoc {
        // link_terms: the raw substrings the v8 dense-clean dropped from this
        // chunk's section (wikilink targets, markdown-link/autolink URLs — see
        // dense-clean.ts extractLinkTerms). Folded into `content` rather than a
        // new field because pre-v8 these bytes lived IN the body at content
        // weight — this is restoration of the query↔doc analyzer symmetry
        // (`theverge.com` matching a clipping's source link), not a new lever.
        const lt = c.link_terms;
        return {
            chunk_id: c.chunk_id,
            title: extractNoteName(c),
            aliases: extractAliasesText(c),
            tags: extractTagsText(c),
            content: lt ? (body ? `${body}\n${lt}` : lt) : body,
            ...(this.withProps && { properties: extractPropertiesText(c) }),
            ...(this.withHeadings && { headings: extractHeadingsText(c) }),
        };
    }

    // Chunk-diff commit (issue #5): would two metadata snapshots of the SAME
    // content-addressed chunk produce the same indexed document? A stable
    // chunk_id pins note_path + title + content + denseSuffix, but NOT the
    // note-level metadata this doc derives fields from (an inline #tag in a
    // different section changes every chunk's tags; a wikilink-target swap
    // behind an alias changes link_terms with identical rendered content).
    // Compares via the SAME extractors buildDoc uses, so it can never drift
    // from the doc builder. Conservative on index shape: properties/headings
    // compare even when not currently indexed — a false "changed" only costs
    // an unnecessary remove+add with a reused vector, never a wrong index.
    // The content field's body half is id-pinned, so only link_terms (its
    // appended suffix — see buildDoc) needs comparing here.
    static docFieldsEqual(a: ChunkMeta, b: ChunkMeta): boolean {
        return extractNoteName(a) === extractNoteName(b)
            && extractAliasesText(a) === extractAliasesText(b)
            && extractTagsText(a) === extractTagsText(b)
            && (a.link_terms ?? '') === (b.link_terms ?? '')
            && extractPropertiesText(a) === extractPropertiesText(b)
            && extractHeadingsText(a) === extractHeadingsText(b);
    }

    // The MiniSearch constructor options — SHARED by fit() and fromJSON() so the
    // analyzer a persisted index is loaded with can never drift from the one it
    // was built with. MiniSearch.loadJSON re-supplies these from the options arg
    // (they are NOT in the serialized blob) and does NOT validate that they match
    // the postings, so this single source of truth + the search.ts stamp are the
    // only guard against loading postings under a mismatched analyzer.
    private buildMiniOptions(withProps: boolean, withHeadings: boolean): Options<IndexedDoc> {
        return {
            fields: [
                'title', 'aliases', 'tags', 'content',
                ...(withProps ? ['properties'] : []),
                ...(withHeadings ? ['headings'] : []),
            ],
            // Don't store any fields back — we don't need them in results.
            // Saves memory on a 15k-chunk vault.
            storeFields: [],
            idField: 'chunk_id',
            // Seek owns vacuum scheduling (search.ts runs it off the write mutex,
            // gated on dirtCount). MiniSearch's default auto-vacuum would fire its
            // own UNAWAITED vacuum inside discard() — invisible to the mutex and
            // to telemetry — so it must stay off. (At vault scale it never met its
            // dirtFactor≥0.1 condition anyway; this pins the behavior explicitly.)
            autoVacuum: false,
            // removeExact() removes by re-tokenizing the reconstructed original
            // doc. If reconstruction ever mismatches what was indexed, MiniSearch
            // warns per missing term with code 'version_conflict' — count it (the
            // caller's signal to fall back to a full refit) and keep the console
            // visible but capped (a whole mismatched doc would warn per term).
            logger: (level: string, message: string, code?: string) => {
                if (code === 'version_conflict') {
                    this.removalMismatchCount++;
                    if (this.removalMismatchCount <= 3) console.warn(message);
                    return;
                }
                if (level === 'warn') console.warn(message);
                else if (level === 'error') console.error(message);
                else console.info(message);
            },
            // English stopword analyzer (+ lowercasing). Applied at index AND
            // query time by MiniSearch; see ENGLISH_STOPWORDS above.
            processTerm,
            // CJK-aware tokenizer (tokenize.ts), index AND query side. Latin
            // text tokenizes exactly like MiniSearch's default; tokens with
            // CJK script chars are dictionary-segmented (Intl.Segmenter) so
            // zh/ja/ko content gets real terms instead of one giant token.
            // Wrapped so MiniSearch's (text, fieldName) call can't pass fieldName
            // into seekTokenize's opts slot — indexing always wants the full
            // recall forms (derived defaults true: glue-join + camelCase split).
            tokenize: (text: string) => seekTokenize(text),
            searchOptions: {
                bm25: BM25_PARAMS,
                boost: this.fieldBoosts,
                // Baked-in defaults stay OFF; search.ts passes the live
                // settings per call (fuzzy → edit distance 1, prefix →
                // PREFIX_LAST_TOKEN), so the cached index is config-agnostic
                // and the toggles need no reindex. See file header.
                prefix: false,
                fuzzy: false,
                // OR semantics matches the v0.0.1 behavior (any token can
                // contribute). Hard AND tightens precision but craters recall
                // (zeroed ALL lexical signal for 19% of relevant notes and LOST
                // nDCG on the 482-q eval); the precision win without that cliff is
                // the SOFT-AND coverage weight applied downstream at fusion — see
                // getScoresWithCoverage + SeekSettings.bm25Coverage.
                combineWith: 'OR',
            },
        };
    }

    // chunk_id → frame-index map + chunk count. The map places getScoresWithCoverage's
    // per-doc scores into the score array slot for each chunk. Rebuilt from the
    // SAME orderedChunks by both fit() and fromJSON(), so a loaded index scores
    // into the identical layout a freshly-fit one would.
    private mapChunkIds(chunks: ChunkMeta[]): void {
        this.chunkCount = chunks.length;
        this.idToIdx.clear();
        for (let i = 0; i < chunks.length; i++) this.idToIdx.set(chunks[i].chunk_id, i);
    }

    // Serialize the underlying MiniSearch index (postings only — the analyzer is
    // re-supplied at load via buildMiniOptions). Caller persists this string +
    // a stamp (search.ts) so a cold start can skip the ~280ms fit().
    toJSON(): string {
        if (!this.mini) throw new Error('MultiFieldBM25.toJSON() before fit()');
        return JSON.stringify(this.mini);
    }

    // Reconstruct from a persisted blob INSTEAD of fitting. The postings come
    // from `json`; the analyzer comes from buildMiniOptions (identical to fit);
    // idToIdx/chunkCount are rebuilt from `chunks`. The caller (search.ts) must
    // only invoke this when its stamp confirms the same corpus + analyzer, so
    // every chunk_id in the postings is present in `chunks` and scores land in
    // the same slots fit() would have produced — relevance-identical.
    fromJSON(json: string, chunks: ChunkMeta[], opts?: { searchableProperties?: boolean; headingsField?: boolean }): this {
        this.withProps = opts?.searchableProperties === true;
        this.withHeadings = opts?.headingsField === true;
        this.mapChunkIds(chunks);
        this.mini = MiniSearch.loadJSON<IndexedDoc>(json, this.buildMiniOptions(this.withProps, this.withHeadings));
        this.clearBounds();   // loaded index → drop any memoized UBs
        this.bf = null;       // BM25F lengths are not persisted: rebuilt from the loaded postings
        return this;
    }

    // Optional per-search overrides, all applied via MiniSearch's per-call
    // searchOptions, which it shallow-merges over the baked-in defaults
    // (verified in dist: `{ ...globalSearchOptions, ...searchOptions }`). So each
    // override replaces ONLY its own key; the index's k1/b/d, combineWith, and
    // any un-overridden option are retained. This is how the admin panel tunes
    // ranking live with no index rebuild — the cached index is config-agnostic,
    // the knobs are supplied at query time.
    //   - `boosts`: full field-boost record (title/alias/tag weights).
    //   - `fuzzy`:  edit-distance override. `1` = absolute max edit distance 1
    //     (dist: `maxDistance = fuzzy < 1 ? round(len*fuzzy) : fuzzy`); `false`
    //     keeps exact matching. The index default is fuzzy:false, so the toggle
    //     passes `1`/`false` explicitly to flip it.
    //   - `prefix`: prefix-expansion override — pass PREFIX_LAST_TOKEN (or any
    //     per-term predicate) to enable, `false` to keep exact. Index default
    //     is prefix:false, mirroring fuzzy.
    getScores(
        query: string,
        opts?: SearchOverrides,
    ): Float64Array {
        return this.getScoresWithCoverage(query, opts).scores;
    }

    // Per-query THEORETICAL upper bound on the multi-field BM25+ score:
    //   bound(q) = Σ_t UB(t)
    // where UB(t) = termUpperBound(t) is the WAND/MaxScore per-term upper bound:
    // the max achievable EXACT single-term candidate score for t over the LIVE
    // index, read from MiniSearch's OWN scorer (a single-term search). Summed over
    // the query's processed token occurrences (duplicates included, mirroring the
    // OR scoring loop). Because getScoresWithCoverage divides MiniSearch's quality
    // multiplier back out, a doc's stored score is the pure additive BM25F sum over
    // its matched terms, each ≤ that term's UB — so Σ_t UB(t) is a tight sup, and
    // for a one-term query it EQUALS the live max score (the parity test), so the
    // best doc normalizes to exactly 1.0.
    //
    // NO ×D multiplier (removed 2026-06-26). MiniSearch's search() multiplies each
    // doc's score by `quality` = the count of distinct matched terms (dist v7.2.0
    // `score: score * quality`) — a hidden term-count boost that STACKED on Seek's
    // own coverage soft-AND, an m × (m/T) = m²/T effect (the "franken" double
    // soft-AND). getScoresWithCoverage now divides that `quality` back out, so the
    // stored score is pure additive BM25F (soft-AND applied once, via coverage) and
    // the bound no longer needs ×D to stay a ceiling — ×D was the dominant multi-
    // term over-compression (a 2-term query SQUARED the inflation).
    //
    // UB(t) is DATA-DEPENDENT-given-the-index but CANDIDATE-SET-INDEPENDENT: it is
    // the corpus max for t, fixed per query term, not a function of the OTHER query
    // terms or the result set — so it stays a TM2C2-legal normalizer (the per-query
    // empirical /max over CANDIDATES is not — see theoreticalNormBm25). A fresh
    // search per term also honors discard() tombstones SYNCHRONOUSLY, so the bound
    // is exact pre-vacuum. UB is memoized per term (termUpperBounds); the cache is
    // cleared on every mutation (fit/add/remove/vacuum). If a future MiniSearch
    // change breaks single-term scoring, the one-term parity test fails loudly in CI.
    //
    // Derived-term caveat (fuzzy AND prefix): UB models EXACT matching. With either
    // expansion on, derived terms score with their own (often higher) IDF, so a doc
    // can exceed the bound — fusion clips to 1 — and a fully-OOV query (typo, or a
    // bare prefix like "rearch") has bound 0 with nonzero derived scores — fusion
    // falls back to /max for that query. Fuzzy side validated on the D&D typo slice
    // (2026-06-09); prefix side on the harness prefix arm (2026-06-10).
    //
    // BM25F (`bm25f` passed): the SAME Σ_t UB(t) over the same token occurrences, with
    // UB(t) = the max BM25F single-term score (bm25fUpperBound) — the bound measures
    // term rarity again instead of "appears in some short title" (shipped UB(`bar`)
    // 158, title-driven; BM25F k1 5: 23, near its idf·(k1+1+δ) ceiling).
    //
    // Fuzzy-aware OOV terms (`fuzzy` passed, 2026-11-12): an OOV term that fuzzy
    // derivation CAN reach is bounded by its best derived term, w·UB(t'), with
    // w = 0.45·L/(L+dist) the same weight the scorer applies. Without this a MIXED
    // typo query ("fy27 plannign") was bounded by its indexed terms alone, so every
    // doc holding fy27 + fuzzy "planning" overshot the bound, clipped to 1.0 and
    // tied, leaving dense + recency to order the whole head. In-vocab terms are
    // untouched (clean queries are bit-identical), and a fully-OOV query that fuzzy
    // reaches now gets a real bound instead of the /max fallback. The max over
    // derived terms (not their sum) keeps the "best correction" reading; a doc
    // holding several variants can still exceed it and clip, as before.
    getQueryBound(
        query: string,
        boosts?: Record<string, number>,
        bm25f?: Bm25fParams,
        fuzzy?: SearchOverrides['fuzzy'],
        prefix?: SearchOverrides['prefix'],
    ): number {
        return this.queryBoundAndTerms(query, boosts, bm25f, fuzzy, prefix).bound;
    }

    // True when the prefix spec fires for terms[i] and the index holds a longer
    // term starting with it (the scorer's atPrefix derivation would apply).
    private hasPrefixExtension(
        m: MiniInternals,
        prefix: SearchOverrides['prefix'],
        term: string, i: number, terms: string[],
    ): boolean {
        const px = typeof prefix === 'function' ? prefix(term, i, terms) : prefix === true;
        if (!px) return false;
        for (const [t] of m._index.atPrefix(term)) if (t.length > term.length) return true;
        return false;
    }

    // One pass for the bound AND the coverage denominator, so the two can never
    // disagree about which query terms are matchable: `matchable` = the distinct
    // processed terms with a positive (exact or fuzzy-reached) UB.
    private queryBoundAndTerms(
        query: string,
        boosts: Record<string, number> | undefined,
        bm25f: Bm25fParams | undefined,
        fuzzy: SearchOverrides['fuzzy'],
        prefix?: SearchOverrides['prefix'],
    ): { bound: number; matchable: Set<string> } {
        const matchable = new Set<string>();
        if (!this.mini || !query.trim()) return { bound: 0, matchable };
        const fieldBoosts = boosts ?? this.fieldBoosts;
        // All-stopword fallback (S2): bound the kept literal terms via the same
        // keep-stopword processing the fallback search uses, so e.g. "will" bounds
        // against the exempt index instead of riding the empirical-/max path.
        const allStopword = isAllStopwordQuery(query);
        const terms: string[] = [];
        for (const raw of seekTokenize(query)) {
            const term = allStopword ? keepStopwordsProcessTerm(raw) : processTerm(raw);
            if (term) terms.push(term);
        }
        const ubOf = (t: string): number => bm25f
            ? this.bm25fUpperBound(t, bm25f)
            : this.termUpperBound(t, fieldBoosts, allStopword);
        const m = fuzzy && !allStopword ? miniInternals(this.mini) : null;
        let perTermSum = 0;
        for (let i = 0; i < terms.length; i++) {
            const term = terms[i];
            let ub = ubOf(term);
            // A prefix-eligible OOV term with prefix extensions ("amster" mid-type
            // before "amsterdam") is scored mainly through those extensions, which
            // this fuzzy bound doesn't model. Leave it unbounded (and out of the
            // coverage denominator), so a lone half-typed token keeps the /max
            // fallback instead of a fuzzy-neighbour bound ("master") that clips the
            // head to ties. Matches the pre-2026-11-12 behaviour for typeahead.
            if (ub <= 0 && m && !this.hasPrefixExtension(m, prefix, term, i, terms)) {
                const maxDist = fuzzyMaxDist(fuzzy, term, i, terms);
                if (maxDist) {
                    for (const [t, [, dist]] of m._index.fuzzyGet(term, maxDist)) {
                        if (!dist) continue;
                        const w = MS_FUZZY_WEIGHT * t.length / (t.length + dist);
                        const u = w * ubOf(t);
                        if (u > ub) ub = u;
                    }
                }
            }
            if (ub <= 0) continue;            // unreachable in every field → can't score
            perTermSum += ub;                 // per-occurrence (mirrors the OR sum)
            matchable.add(term);
        }
        return { bound: perTermSum, matchable };
    }

    // Max achievable EXACT single-term candidate score for `term` over the live
    // index = the WAND/MaxScore per-term upper bound UB(t). Driven by MiniSearch's
    // OWN scorer (a single-term search, fuzzy/prefix OFF) so it needs no formula
    // replication and stays exact across MiniSearch upgrades. For a single query
    // term MiniSearch's quality multiplier is ×1, so r.score IS the pure additive
    // single-term score — directly comparable to the quality-divided scores
    // getScoresWithCoverage stores. A fresh search honors discard() tombstones, so
    // the bound is exact pre-vacuum. Memoized on the default-boost path (the
    // production call); a per-call boost override (eval only) is computed fresh.
    // Returns 0 for an OOV term (no postings → no results).
    private termUpperBound(term: string, boosts: Record<string, number>, keepStop: boolean): number {
        const mini = this.mini;
        if (!mini) return 0;
        const useCache = boosts === this.fieldBoosts;
        const key = keepStop ? ` ${term}` : term;
        if (useCache) {
            const hit = this.termUpperBounds.get(key);
            if (hit !== undefined) return hit;
        }
        let mx = 0;
        const results = mini.search(term, {
            boost: boosts,
            fuzzy: false,
            prefix: false,
            ...(keepStop ? { processTerm: keepStopwordsProcessTerm } : {}),
        });
        for (const r of results) if (r.score > mx) mx = r.score;
        if (useCache) this.termUpperBounds.set(key, mx);
        return mx;
    }

    private clearBounds(): void {
        this.termUpperBounds.clear();
        this.bm25fUpperBounds.clear();
    }

    // ── BM25F (bm25f.ts) ─────────────────────────────────────────────────────
    // Hand-rolled over MiniSearch's per-field postings rather than a MiniSearch
    // fork: the postings (term → field → doc → tf), the radix tree (prefix /
    // fuzzy derivation) and the liveness map are everything BM25F needs, and none
    // of it changes shape — the persisted blob, the analyzer and the incremental
    // add/remove contract are untouched, so the legacy path stays one flag away.

    // UB(t) under BM25F: the max BM25F single-term score for the EXACT term over the
    // live index (incl. a tolerant-load ghost, as the legacy search-based UB does).
    // Memoized per params value (see bm25fUpperBounds), cleared with the legacy UBs
    // on every mutation. 0 for an OOV term. Public for the parity test + telemetry.
    bm25fUpperBound(term: string, p: Bm25fParams = BM25F_PARAMS): number {
        const m = miniInternals(this.mini);
        if (!m) return 0;
        const pk = bm25fParamsKey(p);
        let memo = this.bm25fUpperBounds.get(pk);
        if (!memo) {
            if (this.bm25fUpperBounds.size >= 8) {   // oldest param set out
                this.bm25fUpperBounds.delete(this.bm25fUpperBounds.keys().next().value as string);
            }
            memo = new Map();
            this.bm25fUpperBounds.set(pk, memo);
        }
        const hit = memo.get(term);
        if (hit !== undefined) return hit;
        let mx = 0;
        const data = m._index.get(term);
        if (data) {
            const st = this.bm25fState(m);
            const n = bm25fScoreTerm(data, this.bm25fFieldStats(m, st, p), st.rowOf,
                m._documentCount, p.k1, p.d, this.bfScratch);
            const sc = this.bfScratch.score;
            for (let i = 0; i < n; i++) if (sc[i] > mx) mx = sc[i];
        }
        memo.set(term, mx);
        return mx;
    }

    // Per-row BM25F single-term scores for one processed term (the parity test's
    // per-term check; not on the search path). Rows without the term stay 0.
    bm25fTermScores(term: string, p: Bm25fParams = BM25F_PARAMS): Float64Array {
        const out = new Float64Array(this.chunkCount);
        const m = miniInternals(this.mini);
        const data = m?._index.get(term);
        if (!m || !data) return out;
        const st = this.bm25fState(m);
        const n = bm25fScoreTerm(data, this.bm25fFieldStats(m, st, p), st.rowOf,
            m._documentCount, p.k1, p.d, this.bfScratch);
        const { touched, score } = this.bfScratch;
        for (let i = 0; i < n; i++) {
            const row = st.rowOf[touched[i]];
            if (row >= 0) out[row] = score[i];
        }
        return out;
    }

    // Corpus statistics the BM25F scorer reads (diagnostics + the parity test):
    // N, and per field the average processed length and each row's length.
    bm25fStats(): { n: number; avgLen: Record<string, number>; lenOf(row: number, field: string): number } | null {
        const m = miniInternals(this.mini);
        if (!m) return null;
        const st = this.bm25fState(m);
        const n = m._documentCount;
        const avgLen: Record<string, number> = {};
        for (const [field, fid] of Object.entries(m._fieldIds)) avgLen[field] = n > 0 ? st.lenSum[fid] / n : 0;
        const sidOfRow = new Map<number, number>();
        for (let sid = 0; sid < st.rowOf.length; sid++) if (st.rowOf[sid] >= 0) sidOfRow.set(st.rowOf[sid], sid);
        return {
            n, avgLen,
            lenOf: (row, field) => {
                const sid = sidOfRow.get(row);
                const fid = m._fieldIds[field];
                return sid === undefined || fid === undefined ? 0 : st.len[fid][sid];
            },
        };
    }

    // Build the lazy BM25F state now (the full lengths pass + row map) instead of on
    // the first BM25F query. Optional — the state builds itself on demand — but lets
    // a caller pay it where it already pays the fit/load (search.ts ensureBm25).
    warmBm25f(): void {
        const m = miniInternals(this.mini);
        if (m) this.bm25fState(m);
    }

    private bm25fState(m: MiniInternals): Bm25fState {
        let st = this.bf;
        if (!st) {
            st = this.buildBm25fLengths(m);
            this.bf = st;
        }
        if (st.rowsDirty) this.rebuildBm25fRows(m, st);
        this.bfScratch.ensure(m._nextId);   // every kernel caller goes through here
        return st;
    }

    // The one full pass: every field's processed-token length per live doc is the
    // sum of its tf over the postings (MiniSearch's addTerm increments tf once per
    // processed token), so no body re-fetch or re-tokenization is needed — which is
    // what lets a fromJSON-loaded index (no bodies in hand) get exact lengths.
    // O(total postings); runs once per fit/load, never per delta.
    private buildBm25fLengths(m: MiniInternals): Bm25fState {
        const nf = Object.keys(m._fieldIds).length;
        const cap = m._nextId;
        const alive = new Uint8Array(cap);
        for (const sid of m._documentIds.keys()) if (sid < cap) alive[sid] = 1;
        const len: Float64Array[] = [];
        const lenSum: number[] = [];
        for (let f = 0; f < nf; f++) { len.push(new Float64Array(cap)); lenSum.push(0); }
        for (const [, data] of m._index) {
            for (const [fid, docs] of data) {
                const L = len[fid];
                if (!L) continue;
                let s = 0;
                for (const [sid, tf] of docs) {
                    if (alive[sid] !== 1) continue;
                    L[sid] += tf;
                    s += tf;
                }
                lenSum[fid] += s;
            }
        }
        return { len, lenSum, rowOf: new Int32Array(0), rowsDirty: true };
    }

    private rebuildBm25fRows(m: MiniInternals, st: Bm25fState): void {
        const cap = m._nextId;
        const rowOf = st.rowOf.length >= cap ? st.rowOf : new Int32Array(Math.max(cap, Math.ceil(st.rowOf.length * 1.5)));
        rowOf.fill(-1);
        for (const [sid, ext] of m._documentIds) {
            const row = this.idToIdx.get(ext);
            rowOf[sid] = row === undefined ? -2 : row;
        }
        st.rowOf = rowOf;
        st.rowsDirty = false;
    }

    private bm25fFieldStats(m: MiniInternals, st: Bm25fState, p: Bm25fParams): Array<Bm25fFieldStats | undefined> {
        const n = m._documentCount;
        const out: Array<Bm25fFieldStats | undefined> = [];
        for (const [field, fid] of Object.entries(m._fieldIds)) {
            const avg = n > 0 ? st.lenSum[fid] / n : 0;
            out[fid] = {
                weight: bm25fFieldWeight(p, field),
                b: bm25fFieldB(p, field),
                avg: avg || 1,                   // harness `avg_len[f] or 1.0`
                len: st.len[fid],
            };
        }
        return out;
    }

    // Incremental length maintenance (only once the lazy state exists — before
    // that, the full pass will see the change in the postings anyway).
    // bm25fForget: BEFORE the doc leaves MiniSearch (its short id is still mapped).
    private bm25fForget(id: string): void {
        const st = this.bf;
        if (!st) return;
        st.rowsDirty = true;
        const m = miniInternals(this.mini);
        const sid = m?._idToShortId.get(id);
        if (sid === undefined) return;
        for (let f = 0; f < st.len.length; f++) {
            const L = st.len[f];
            if (sid < L.length) {
                st.lenSum[f] -= L[sid];
                L[sid] = 0;
            }
        }
    }

    // bm25fLearn: AFTER mini.add(doc). Counts the doc's processed tokens per field
    // exactly as MiniSearch's add() emits postings — tokenize (seekTokenize, the
    // wrapper ignores the field) then processTerm(term, field), truthy only — so
    // the result equals what the full postings pass would compute for it.
    private bm25fLearn(id: string, doc: IndexedDoc): void {
        const st = this.bf;
        if (!st) return;
        st.rowsDirty = true;
        const m = miniInternals(this.mini);
        const sid = m?._idToShortId.get(id);
        if (!m || sid === undefined) return;
        for (let f = 0; f < st.len.length; f++) {
            if (st.len[f].length <= sid) {
                const grown = new Float64Array(Math.max(sid + 1, Math.ceil(st.len[f].length * 1.5), 64));
                grown.set(st.len[f]);
                st.len[f] = grown;
            }
        }
        for (const [field, fid] of Object.entries(m._fieldIds)) {
            const v = (doc as unknown as Record<string, unknown>)[field];
            if (v == null || !st.len[fid]) continue;
            let count = 0;
            for (const tok of seekTokenize(String(v))) if (processTerm(tok, field)) count++;
            st.len[fid][sid] = count;
            st.lenSum[fid] += count;
        }
    }

    // The BM25F query loop. Runs the query exactly as MiniSearch's executeQuery /
    // executeQuerySpec would — same tokenize + processTerm (incl. the synonym
    // wrapper that emits mates BEFORE their source), same per-term predicates
    // (termToQuerySpec), same derivation (exact, then atPrefix extensions at
    // 0.375·L/(L+0.3Δ) taking precedence over fuzzyGet matches at 0.45·L/(L+dist),
    // distance-0 skipped), same boostTerm discount on mates — but each (derived)
    // term is scored by the BM25F kernel instead of the per-field sum. A derived term
    // contributes weight × its OWN BM25F score (own postings, own doc-level IDF),
    // the harness's `add(d, w, src)`, and fills its SOURCE term's coverage slot (a
    // mate fills every source it stands in for). No `quality` multiplier is ever
    // applied, so the stored score is the pure additive sum the legacy path only
    // reaches by dividing MiniSearch's ×quality back out.
    private scoreBm25f(
        query: string,
        p: Bm25fParams,
        m: MiniInternals,
        q: {
            processTerm: (raw: string) => string | string[] | null;
            fuzzy: SearchOverrides['fuzzy'];
            prefix: SearchOverrides['prefix'];
            m2s: Map<string, string[]> | undefined;
            synWeight: number;
            totalTerms: number;
        },
        scores: Float64Array,
        coverage: Float64Array,
    ): void {
        const st = this.bm25fState(m);
        const fields = this.bm25fFieldStats(m, st, p);
        const scratch = this.bfScratch;
        const nDocs = m._documentCount;
        const rowOf = st.rowOf;

        const terms: string[] = [];
        for (const raw of seekTokenize(query)) {
            const t = q.processTerm(raw);
            if (Array.isArray(t)) { for (const x of t) if (x) terms.push(x); }
            else if (t) terms.push(t);
        }

        // Coverage slots: one per distinct SOURCE term. ≤32 ride a per-row bitmask
        // (the common case by far); any beyond spill to a per-row Set.
        const slotOf = new Map<string, number>();
        const slot = (src: string): number => {
            let s = slotOf.get(src);
            if (s === undefined) { s = slotOf.size; slotOf.set(src, s); }
            return s;
        };
        const mask = new Uint32Array(this.chunkCount);
        const overflow = new Map<number, Set<number>>();
        const hitRows: number[] = [];

        const apply = (data: Postings | undefined, weight: number, slots: number[]): void => {
            if (!data) return;
            const n = bm25fScoreTerm(data, fields, rowOf, nDocs, p.k1, p.d, scratch);
            const { touched, score } = scratch;
            for (let i = 0; i < n; i++) {
                const s = score[i];
                if (!(s > 0)) continue;
                const row = rowOf[touched[i]];
                if (row < 0) continue;            // ghost: counts in N/df, has no frame row
                if (scores[row] === 0) hitRows.push(row);
                scores[row] += weight * s;
                for (const sl of slots) {
                    if (sl < 32) mask[row] |= 1 << sl;
                    else {
                        let set = overflow.get(row);
                        if (!set) { set = new Set(); overflow.set(row, set); }
                        set.add(sl);
                    }
                }
            }
        };

        for (let i = 0; i < terms.length; i++) {
            const term = terms[i];
            // termToQuerySpec (dist v7.2.0)
            const px = typeof q.prefix === 'function' ? q.prefix(term, i, terms) : q.prefix === true;
            const termBoost = q.m2s?.has(term) ? q.synWeight : 1;
            const slots = (q.m2s?.get(term) ?? [term]).map(slot);

            apply(m._index.get(term), termBoost, slots);
            // executeQuerySpec (dist v7.2.0)
            let fuzzyMatches: Map<string, [Postings, number]> | undefined;
            const maxDist = fuzzyMaxDist(q.fuzzy, term, i, terms);
            if (maxDist) fuzzyMatches = m._index.fuzzyGet(term, maxDist);
            if (px) {
                for (const [t, data] of m._index.atPrefix(term)) {
                    const dist = t.length - term.length;
                    if (!dist) continue;                 // the exact term, scored above
                    fuzzyMatches?.delete(t);             // prefix takes precedence
                    apply(data, termBoost * MS_PREFIX_WEIGHT * t.length / (t.length + 0.3 * dist), slots);
                }
            }
            if (fuzzyMatches) {
                for (const [t, [data, dist]] of fuzzyMatches) {
                    if (!dist) continue;
                    apply(data, termBoost * MS_FUZZY_WEIGHT * t.length / (t.length + dist), slots);
                }
            }
        }

        // Coverage (soft-AND, applied ^BM25_COVERAGE_POW downstream): matched
        // distinct source terms / indexed query terms, clamped at 1 (a fuzzy-matched
        // OOV source can push the count past the indexed denominator).
        const total = q.totalTerms;
        for (const row of hitRows) {
            let cnt = popcount32(mask[row]);
            const extra = overflow.get(row);
            if (extra) cnt += extra.size;
            coverage[row] = total > 1 ? Math.min(1, cnt / total) : 1;
        }
    }

    // Fraction of docs containing `term` in ANY field (max over fields) —
    // feeds the synonym dictionary's df ceiling (synonyms.ts). Same defensive
    // internals-reach as getQueryBound; 0 when unavailable, which makes the
    // df guard a no-op rather than a false positive.
    // True when the processed `term` has postings in the index (any field). The
    // title gate's fuzzy eligibility check (search.ts → fusion.ts TitleFuzzy): only
    // an OOV token is treated as a typo. True when internals are unavailable, which
    // makes the caller fall back to exact-only (never fuzzes a real word).
    hasIndexedTerm(term: string): boolean {
        const m = miniInternals(this.mini);
        return m ? m._index.get(term) !== undefined : true;
    }

    termDocFraction(term: string): number {
        const mini = this.mini as unknown as {
            _index?: { get(term: string): Map<number, Map<number, number>> | undefined };
            _documentCount?: number;
        } | null;
        const index = mini?._index;
        const n = mini?._documentCount;
        if (!index || typeof index.get !== 'function' || typeof n !== 'number' || n <= 0) return 0;
        const fieldsData = index.get(term);
        if (!fieldsData) return 0;
        let maxDf = 0;
        for (const docs of fieldsData.values()) {
            if (docs.size > maxDf) maxDf = docs.size;
        }
        return maxDf / n;
    }

    // Coverage denominator (queryBoundAndTerms `matchable`): the distinct
    // processed query terms with a positive UB, exact or fuzzy-reached. Mirrors
    // the bound by construction: a term NO doc can match would uniformly cap
    // every doc's coverage at (T−1)/T, attenuating the whole BM25 channel vs
    // dense for one unreachable query term, contradicting the bound. Aligning the
    // two means a doc matching every MATCHABLE term keeps full lexical weight.
    // (Replaced indexedQueryTerms / bm25fIndexedTermCount 2026-11-12; UB > 0 also
    // drops a term whose every posting is a discard() tombstone.)

    // As getScores, but also returns a per-doc COVERAGE weight in [0,1]: the
    // fraction of distinct query terms the doc matched (MiniSearch's own
    // `queryTerms`, so fuzzy/prefix expansions are mapped back to the query term
    // they matched). This is the soft-AND signal — multiply raw BM25 by it BEFORE
    // normalization to discount docs that only matched part of a multi-term query
    // (SeekSettings.bm25Coverage; see search.ts). Single-term queries get weight 1
    // for every match (a no-op), so the gate is implicit. Non-matching docs stay 0
    // in both arrays. One MiniSearch pass feeds both (getScores delegates here).
    // Also returns `bound` — getQueryBound() for this query+boosts — so the one
    // MiniSearch pass hands fusion everything it needs to normalize the channel.
    getScoresWithCoverage(
        query: string,
        opts?: SearchOverrides,
    ): { scores: Float64Array; coverage: Float64Array; bound: number } {
        const scores = new Float64Array(this.chunkCount);
        const coverage = new Float64Array(this.chunkCount);
        if (!this.mini || !query.trim()) return { scores, coverage, bound: 0 };
        // BM25F (opts.bm25f) reads MiniSearch internals; if their shape ever changes
        // (a MiniSearch upgrade) it degrades to the legacy per-field path with one
        // warning instead of silently zeroing the lexical channel.
        const bfParams = opts?.bm25f;
        const bfMini = bfParams ? miniInternals(this.mini) : null;
        if (bfParams && !bfMini && !this.warnedBm25fInternals) {
            this.warnedBm25fInternals = true;
            console.warn('[seek] BM25F: MiniSearch internals unavailable; using the per-field BM25 path');
        }
        const bm25f = bfMini ? bfParams : undefined;
        // Bound + coverage denominator from one pass (queryBoundAndTerms). The
        // denominator = distinct MATCHABLE query terms: indexed, or OOV but reached
        // by fuzzy derivation (see indexedQueryTerms for why unreachable OOV terms
        // are excluded). >1 ⇒ a genuine multi-term query where partial matches
        // should be discounted; ≤1 ⇒ every match covers the whole matchable query,
        // so weight is a flat 1 (no-op). A fuzzy-reached typo now holds its own
        // slot, so "fy27 plannign" discounts a doc that only has fy27.
        const { bound, matchable } = this.queryBoundAndTerms(query, opts?.boosts, bm25f, opts?.fuzzy, opts?.prefix);
        const totalTerms = matchable.size;

        // ---- synonym expansion plumbing (per-call; dictionary from synonyms.ts)
        // Native MiniSearch path: a query-time processTerm may return an
        // ARRAY, and each element becomes its own query spec (dist v7.2.0
        // flatMaps the expansion in executeQuery). Injected mates are
        // therefore their OWN source terms — which differs from the eval
        // harness's source-attributed arm in two places, and BOTH are mapped
        // back to source semantics in the result loop below:
        //   - ×QUALITY (the double-credit the adversarial review flagged):
        //     corrected by an exact post-hoc rescale — see the loop comment
        //     for why the 2026-06-10 gate showed this is load-bearing, not
        //     cosmetic (native attribution FAILED at every weight).
        //   - COVERAGE (review fix #1): mates map back to their source term —
        //     otherwise a mate fills the coverage slot of an UNMATCHED real
        //     term and partially undoes the soft-AND: "lr roadmap" → Lr Home
        //     coverage 1.0, not 0.5.
        // Invariants:
        //   - mates are emitted BEFORE their source term, so the true last
        //     query token keeps the final index and PREFIX_LAST_TOKEN's
        //     position check can never fire on a mate;
        //   - a mate that is already an original query term is never injected
        //     (review fix #2 — it would build two specs for one term, scoring
        //     it ~1.8× against a bound that counts it once), and repeat
        //     trigger occurrences inject only once;
        //   - mates match EXACTLY (fuzzy disabled per-term below) and score
        //     at the dictionary discount via boostTerm — the harness contract.
        // The bound is untouched: getQueryBound iterates the ORIGINAL tokens,
        // so mate contributions are numerator-only and fusion clips at 1 —
        // the same contract as fuzzy and prefix derivation.
        const allStopword = isAllStopwordQuery(query);
        const synMap = opts?.synonyms?.map;
        const synWeight = opts?.synonyms?.weight ?? 1;
        // v2 phrase mates (SearchOverrides.synonyms.phraseMates): the caller
        // already resolved phrase→mate, so this layer only INJECTS. Values are
        // string[] now — a phrase mate attributes back to EVERY phrase token —
        // and the v1 per-token path stores singleton arrays. Routing phrase
        // mates through the SAME m2s inherits fuzzy exclusion, the boostTerm
        // discount, and the coverage source-mapping for free.
        const phraseMates = opts?.synonyms?.phraseMates;
        let mateToSource: Map<string, string[]> | undefined;
        let synProcessTerm: ((term: string) => string | string[] | null) | undefined;
        if ((synMap && synMap.size > 0) || (phraseMates && phraseMates.length > 0)) {
            const original = distinctQueryTerms(query);
            const baseProcess = allStopword ? keepStopwordsProcessTerm : processTerm;
            const m2s = new Map<string, string[]>();
            const pending = [...(phraseMates ?? [])];
            synProcessTerm = (raw: string) => {
                const t = baseProcess(raw);
                if (t === null) return null;
                const out: string[] = [];
                // Flush pending phrase mates on the FIRST surviving term (the
                // queue empties once): a mate emitted here can never occupy
                // terms.length-1, so PREFIX_LAST_TOKEN's position check still
                // lands on the true final query token. Same dedupe as the
                // per-token path (review fix #2): a mate the query already
                // contains would build two specs for one term.
                while (pending.length > 0) {
                    const p = pending.shift()!;
                    if (original.has(p.mate) || m2s.has(p.mate)) continue;
                    m2s.set(p.mate, p.sources);
                    out.push(p.mate);
                }
                const mates = synMap?.get(t);
                if (mates) {
                    for (const mate of mates) {
                        if (original.has(mate) || m2s.has(mate)) continue;
                        m2s.set(mate, [t]);
                        out.push(mate);
                    }
                }
                if (out.length === 0) return t;
                out.push(t);    // mates FIRST, source term LAST
                return out;
            };
            mateToSource = m2s;
        }
        const m2s = mateToSource;
        // Mates must not fuzzy-expand (they're dictionary-exact); wrap the
        // caller's fuzzy setting in a per-term predicate only when both are
        // live. The caller's setting is itself a per-term predicate
        // (FUZZY_BY_LENGTH) — delegate, never return it as a value: a function
        // is truthy and would re-enable fuzzy on the exact CJK/digit/short
        // terms the predicate exists to exclude.
        const baseFuzzy = opts?.fuzzy;
        const fuzzyOverride = m2s && baseFuzzy
            ? (term: string, i: number, terms: string[]) =>
                m2s.has(term) ? false
                : typeof baseFuzzy === 'function' ? baseFuzzy(term, i, terms)
                : baseFuzzy
            : baseFuzzy;

        // MiniSearch returns only matching docs, sorted descending. Non-
        // matches stay at the Float64Array's zero default, which matches
        // the v0.0.1 contract (every chunk gets a score; misses are 0).
        // The `cond && {key}` spreads are no-ops when the override is absent,
        // so an empty object falls through to the baked-in defaults.
        if (bm25f && bfMini) {
            this.scoreBm25f(query, bm25f, bfMini, {
                processTerm: synProcessTerm ?? (allStopword ? keepStopwordsProcessTerm : processTerm),
                fuzzy: fuzzyOverride,
                prefix: opts?.prefix,
                m2s,
                synWeight,
                totalTerms,
            }, scores, coverage);
            return { scores, coverage, bound };
        }

        const results = this.mini.search(query, {
            ...(opts?.boosts && { boost: opts.boosts }),
            ...(fuzzyOverride !== undefined && { fuzzy: fuzzyOverride }),
            ...(opts?.prefix !== undefined && { prefix: opts.prefix }),
            // All-stopword fallback (S2): an all-stopword query ("will",
            // "the who") would otherwise process to ZERO terms and return
            // nothing — while the exempt index holds exactly those terms in
            // title/aliases. Swap in the stoplist-free processor for this
            // call only; queries with ≥1 content word are untouched. The
            // synonym wrapper composes the same fallback internally.
            ...(synProcessTerm
                ? { processTerm: synProcessTerm }
                : (allStopword && { processTerm: keepStopwordsProcessTerm })),
            ...(m2s && { boostTerm: (term: string) => (m2s.has(term) ? synWeight : 1) }),
        });
        for (const r of results) {
            const idx = this.idToIdx.get(String(r.id));
            if (idx === undefined) continue;
            // r.queryTerms = the distinct query terms this doc matched —
            // fuzzy/prefix expansions are already mapped back to the source
            // query term; synonym mates are mapped back HERE (review fix #1,
            // see the plumbing comment above). With any expansion on, a query
            // term OUTSIDE the indexed denominator can still match via a
            // derived term (same exception getQueryBound documents for the
            // bound), so clamp at 1 — full coverage, never a >1 boost.
            // Divide out MiniSearch's `quality` (= |queryTerms|, dist v7.2.0
            // `score: score * quality`) to recover the pure additive BM25F sum.
            // The term-count soft-AND is then applied EXACTLY ONCE, by coverage
            // below — not MiniSearch's hidden ×m SQUARED by Seek's ×(m/T). The
            // division is exact: queryTerms is the very array `quality` was
            // computed from. (2026-06-26 de-franken — supersedes the synonym
            // source-attribution rescale, which had patched only the numerator
            // of the same double-count; coverage still maps mates → source.)
            const nativeQ = r.queryTerms.length || 1;
            scores[idx] = r.score / nativeQ;
            if (m2s) {
                // Synonym mates map back to their source term(s) so a matched
                // mate fills its source's coverage slot, not a fresh one
                // (review fix #1). A phrase mate credits EVERY phrase token —
                // matching "QU" IS matching "query understanding" — and the
                // Math.min clamp absorbs any over-credit from sources outside
                // the indexed denominator.
                const srcs = new Set<string>();
                for (const qt of r.queryTerms) {
                    const s = m2s.get(qt);
                    if (s) { for (const x of s) srcs.add(x); }
                    else srcs.add(qt);
                }
                coverage[idx] = totalTerms > 1 ? Math.min(1, srcs.size / totalTerms) : 1;
            } else {
                const matched = new Set(r.queryTerms).size;
                coverage[idx] = totalTerms > 1 ? Math.min(1, matched / totalTerms) : 1;
            }
        }
        return { scores, coverage, bound };
    }

    // ── Incremental maintenance (Seek scaling A1) ────────────────────────────
    // fit() is the cold path; add/remove/vacuum keep the SAME index alive across
    // a delta so a one-chunk edit doesn't re-fit the whole corpus. The frame
    // (search.ts) appends/tombstones its row-aligned arrays in lockstep with
    // these calls — `idToIdx[id]` MUST equal the frame row for the single-`idx`
    // candidate join to be correct, which is why add() is the sole authority for
    // row assignment (row = chunkCount) and the caller mirrors it. A runtime
    // drift detector (search.ts) re-checks that coupling and falls back to a full
    // rebuild if it ever diverges. Incremental scores are numerically ~1e-9 from
    // a fresh fit (MiniSearch's running-mean avgFieldLength differs in low bits
    // from a batch summation), not bit-identical — parity tests compare by
    // chunk_id with an epsilon.

    // Append one chunk as a new row at index chunkCount. Caller must remove() any
    // prior chunk carrying this id first (reindexDelta drops a file's stale
    // chunks before re-embedding, and content-hash ids change on edit). Returns the
    // assigned row so the caller can assert frame-row alignment.
    //
    // Ghost guard (2026-06-20): a TOLERANTLY-loaded blob (fromJSON builds idToIdx
    // from the LIVE frame, not the postings) can hold a posting under this id that
    // the live frame had already dropped — and re-adding content identical to that
    // dropped chunk yields the SAME content-derived id, colliding with the ghost
    // (which freshDeltaAdds can't filter, since idToIdx lacks the dead id). Discard a
    // GHOST first so mini.add() doesn't throw → full refit (the freeze) on the first
    // such re-add after a stale/cross-device load. The guard is narrow: it fires ONLY
    // when the id is in the postings but ABSENT from the live row space (idToIdx). A
    // LIVE duplicate (id in BOTH) is a caller-contract violation (must remove() first)
    // and still throws loudly — that is the meltdown guard, deliberately preserved.
    add(chunk: ChunkMeta, body: string): number {
        if (!this.mini) throw new Error('MultiFieldBM25.add() before fit()');
        const row = this.chunkCount;
        if (!this.idToIdx.has(chunk.chunk_id) && this.mini.has(chunk.chunk_id)) {
            this.bm25fForget(chunk.chunk_id);
            this.mini.discard(chunk.chunk_id);
        }
        const doc = this.buildDoc(chunk, body);
        this.mini.add(doc);
        this.bm25fLearn(chunk.chunk_id, doc);
        this.clearBounds();   // corpus changed → recompute UBs on next bound
        this.idToIdx.set(chunk.chunk_id, row);
        this.chunkCount = row + 1;
        return row;
    }

    // Tombstone a chunk: discard() drops it from search results but leaves its
    // postings in place (tombstoned) until vacuum(). chunkCount is UNCHANGED — the
    // row stays a hole (score 0, masked at selection by the frame's validRows), so
    // the monotonic row numbering shared with the frame never shifts. Idempotent on
    // idToIdx (a no-op if the id isn't live). The discard is wrapped because a
    // TOLERANTLY-loaded blob can carry an id in idToIdx (built from the live frame)
    // that is ABSENT from mini's postings (a chunk added AFTER the blob was written):
    // discarding a missing id is a semantic no-op (already gone), NOT the error
    // MiniSearch throws — which would otherwise abort the delta into a full refit
    // (the freeze) on the first edit/delete of a blob-missing chunk after a load.
    remove(id: string): void {
        if (!this.mini || !this.idToIdx.has(id)) return;
        this.idToIdx.delete(id);
        this.bm25fForget(id);
        try { this.mini.discard(id); } catch { /* ghost posting from a tolerant load — already absent */ }
        // Drop memoized UBs: the next getQueryBound re-searches the live index,
        // whose results already exclude the discarded doc — so the bound honors
        // this tombstone synchronously, with no vacuum (the incremental contract).
        this.clearBounds();
    }

    // Exact synchronous removal (issue #5): remove a chunk's postings IMMEDIATELY
    // by re-tokenizing its reconstructed original document, instead of discard()'s
    // tombstone-until-vacuum. Zero dirt, zero deferred cleanup — and it closes the
    // discard window where a term's postings still carry the dead doc, which
    // inflates that term's df for live docs on the FIRST search that walks the
    // tombstone late in posting order (measured: enough same-vocabulary tombstones
    // can push df past documentCount and flip BM25 idf NEGATIVE). Reconstruction
    // is exact by construction: chunk ids are content hashes, so the body fetched
    // for a removed id is byte-identical to what add() indexed, and the caller
    // passes the frame row's meta — the meta as last committed. buildDoc is the
    // single shared builder, so the remove-side doc can't drift from the add-side.
    //
    // Returns:
    //   'removed'  — clean exact removal, no residue.
    //   'mismatch' — MiniSearch warned 'version_conflict' (reconstructed doc ≠
    //                indexed doc). The doc is gone from results either way, but
    //                unknown postings may remain as self-healing dirt-like residue
    //                and field-length bookkeeping may have drifted → the caller
    //                should abandon the patch and full-refit ("slow, never wrong").
    //   'noop'     — id not live (parity with remove()'s tolerance): absent from
    //                idToIdx, or present there but missing from the postings (a
    //                tolerantly-loaded blob that predates the chunk).
    removeExact(chunk: ChunkMeta, body: string): 'removed' | 'mismatch' | 'noop' {
        const id = chunk.chunk_id;
        if (!this.mini || !this.idToIdx.has(id)) return 'noop';
        this.idToIdx.delete(id);
        this.clearBounds();
        // The blob-missing tolerance (see remove()): an id tracked by the live
        // row space but absent from a tolerantly-loaded blob's postings. Checked
        // EXPLICITLY rather than via catch — MiniSearch only throws before
        // touching postings for its two id guards; a throw anywhere later
        // (tokenizer, per-term removal) happens MID-mutation, and reporting that
        // as a clean 'noop' would let the caller commit a half-removed index.
        if (!this.mini.has(id)) {
            if (this.bf) this.bf.rowsDirty = true;   // the row left idToIdx all the same
            return 'noop';
        }
        this.bm25fForget(id);
        const before = this.removalMismatchCount;
        try {
            this.mini.remove(this.buildDoc(chunk, body));
        } catch (e) {
            console.warn('[seek] removeExact threw mid-removal — postings may be partially mutated', e);
            return 'mismatch';   // caller abandons the patch and rebuilds from IDB truth
        }
        return this.removalMismatchCount === before ? 'removed' : 'mismatch';
    }

    // Reclaim tombstoned postings left by discard()-path removals. NO LONGER on
    // any production path: removeExact() is the delta's removal primitive (zero
    // dirt), and the ghost-dirt valve (search.ts reconcileGhostDirt) reclaims
    // pathological discard accumulation via a full refit instead — MiniSearch's
    // vacuum holds a live radix-tree iterator across its timer yields, so a
    // vacuum overlapping add/remove/search-time cleanup can be invalidated
    // mid-walk, and a REJECTED vacuum never resets _currentVacuum (every later
    // vacuum on the instance then returns the same rejection). Callers must
    // guarantee NOTHING mutates or searches this index until the promise
    // settles. Kept for tests and as API completeness. (Historical note: the
    // old "vacuum before the next getQueryBound or the bound reads stale df"
    // contract died with the search-based MaxScore bound — the incremental
    // tests assert bound exactness pre-vacuum.)
    async vacuum(): Promise<void> {
        if (!this.mini) return;
        await this.mini.vacuum();
        this.clearBounds();   // postings compacted → recompute UBs
    }

    // Drift-detector / compaction surface. size = chunkCount = R (rows incl
    // tombstones); liveCount = live (non-tombstoned) rows; rowOf maps a live id to
    // its row for the sampled coherence spot-check (idToIdx[orderedIds[i]] === i).
    get size(): number { return this.chunkCount; }
    get liveCount(): number { return this.idToIdx.size; }
    rowOf(id: string): number | undefined { return this.idToIdx.get(id); }

    // Pending tombstone count (MiniSearch _dirtCount), via the same defensive
    // internals-reach as getQueryBound. 0 after a successful vacuum() — the
    // post-vacuum bound-exactness test asserts this. Returns 0 if internals are
    // unavailable (a future MiniSearch rename), matching the bound's fail-safe.
    get dirtCount(): number {
        const mini = this.mini as unknown as { _dirtCount?: number } | null;
        return typeof mini?._dirtCount === 'number' ? mini._dirtCount : 0;
    }
}
