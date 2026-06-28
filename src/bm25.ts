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
import type { Chunk, ChunkMeta } from './chunker';
import { seekTokenize, hasCjk } from './tokenize';
import { toDisplayForm } from './prop-normalize';

// Per-field BM25 term-frequency multipliers. Eval-tuned 2026-06-08 on the
// 482-q old-log personal eval (with the coverage navigational boost active):
// these were the user-facing 3/2/1.5 sliders, but once coverage is in their
// marginal effect is small (~+0.004 nDCG@10), so the sliders were removed and
// the swept values baked in here. 10/6/3 captures the gain (0.8714 vs the
// 0.8727 all=10 max) while keeping `tags` modest (3×) so noisy-tag vaults
// aren't over-weighted. `page_type` is left NEUTRAL (1×): it's not a universal
// field, so a general default shouldn't boost it. See ~/eval.
export const DEFAULT_FIELD_BOOSTS: Record<string, number> = {
    title: 10.0,
    aliases: 6.0,
    tags: 3.0,
    page_type: 1.0,
    content: 1.0,
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
// coordination (Lucene's old `coord`); the de-franken left it at 1 and
// regressed natural "[thing] in [place]" queries — a doc matching only the rare
// place token ("sf", "austin") normalized to a near-max lexical score and out-ranked
// real answers. P=2 restores the franken-era m²/T penalty magnitude as an EXPLICIT
// knob (not the old accidental ×quality squaring), hardening the partial-match
// discount so full-coverage docs win. Eval-validated 2026-06-27 (~/eval
// cross-corpus sweep): personal vault bars nDCG@10 0.302→0.457, captures 0.445→0.496,
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
// see it). Eval-tuned 2026-06-10 (~/eval prefix_arm.py, WS2.3
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
export const PREFIX_LAST_TOKEN = (term: string, i: number, terms: string[]): boolean =>
    i === terms.length - 1 && term.length >= 3;

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
    synonyms?: { map: Map<string, string[]>; weight: number };
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

function extractPageType(chunk: ChunkMeta): string {
    return chunk.metadata?.pageType ?? '';
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

// Machinery keys excluded from the searchable-properties field: identity/UI/
// plumbing, or values already carried by a dedicated field. Everything else
// (placeLoc, placeType, context, status, …) is in — the harness gate measured
// the generic everything-but-machinery posture, not a curated allowlist.
// Compared lowercased; chunker's extractProperties stores SCALARS only, so
// list-valued props (relatedPages-style) never reach this field at all.
const PROPERTY_EXCLUDE_KEYS = new Set([
    'tags', 'aliases', 'alias', 'pagetype', 'page_type', 'created', 'modified',
    'datelink', 'icon', 'coordinates', 'cssclasses', 'cssclass', 'protected',
    'version', 'completed', 'completeddate', 'position', 'banner', 'banner_y',
]);
const PROPERTY_DATE_RE = /^\d{4}-\d{2}-\d{2}/;
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
        const v = toDisplayForm(String(raw).replace(/^["']|["']$/g, ''));
        if (!v || PROPERTY_DATE_RE.test(v) || PROPERTY_NUM_RE.test(v)) continue;
        vals.push(v);
    }
    return vals.join(' ');
}

// Shape of a single document handed to MiniSearch. Per-chunk granularity:
// each chunk is one document, fields that are file-level metadata (title,
// aliases, tags, page_type) are duplicated across chunks of the same note —
// same shape as the v0.0.1 hand-rolled index, by design.
interface IndexedDoc {
    chunk_id: string;
    title: string;
    aliases: string;
    tags: string;
    page_type: string;
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
        this.withProps = opts?.searchableProperties === true;
        this.withHeadings = opts?.headingsField === true;
        this.mapChunkIds(chunks);
        this.mini = new MiniSearch<IndexedDoc>(this.buildMiniOptions(this.withProps, this.withHeadings));
        this.termUpperBounds.clear();   // fresh index → drop memoized per-term UBs

        const docs: IndexedDoc[] = chunks.map(c => this.buildDoc(c, bodies.get(c.chunk_id) ?? ''));
        // addAll is ~10x faster than per-doc add() because it defers
        // internal index rebuilds until the batch completes. At 15k chunks
        // this matters — single-add fit time was ~600 ms in early dev,
        // batched is ~80 ms.
        this.mini.addAll(docs);
        return this;
    }

    // Build the MiniSearch document for one chunk + its body, with the field set
    // (properties/headings) the index was fit with. Shared by fit() and the
    // incremental add() so a row appended live is byte-for-byte the same document
    // a full refit would have produced for that chunk.
    private buildDoc(c: ChunkMeta, body: string): IndexedDoc {
        return {
            chunk_id: c.chunk_id,
            title: extractNoteName(c),
            aliases: extractAliasesText(c),
            tags: extractTagsText(c),
            page_type: extractPageType(c),
            content: body,
            ...(this.withProps && { properties: extractPropertiesText(c) }),
            ...(this.withHeadings && { headings: extractHeadingsText(c) }),
        };
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
                'title', 'aliases', 'tags', 'page_type', 'content',
                ...(withProps ? ['properties'] : []),
                ...(withHeadings ? ['headings'] : []),
            ],
            // Don't store any fields back — we don't need them in results.
            // Saves memory on a 15k-chunk vault.
            storeFields: [],
            idField: 'chunk_id',
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
        this.termUpperBounds.clear();   // loaded index → drop any memoized UBs
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
    getQueryBound(query: string, boosts?: Record<string, number>): number {
        if (!this.mini || !query.trim()) return 0;
        const fieldBoosts = boosts ?? this.fieldBoosts;
        // All-stopword fallback (S2): bound the kept literal terms via the same
        // keep-stopword processing the fallback search uses, so e.g. "will" bounds
        // against the exempt index instead of riding the empirical-/max path.
        const allStopword = isAllStopwordQuery(query);
        let perTermSum = 0;
        for (const raw of seekTokenize(query)) {
            const term = allStopword ? keepStopwordsProcessTerm(raw) : processTerm(raw);
            if (!term) continue;
            const ub = this.termUpperBound(term, fieldBoosts, allStopword);
            if (ub <= 0) continue;            // OOV in every field → can't score
            perTermSum += ub;                 // per-occurrence (mirrors the OR sum)
        }
        return perTermSum;
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

    // Fraction of docs containing `term` in ANY field (max over fields) —
    // feeds the synonym dictionary's df ceiling (synonyms.ts). Same defensive
    // internals-reach as getQueryBound; 0 when unavailable, which makes the
    // df guard a no-op rather than a false positive.
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

    // Coverage denominator: the distinct processed query terms that are
    // INDEXED (present in MiniSearch's term index at all). Mirrors
    // getQueryBound's indexed-terms-only design (its D factor): a term absent
    // from the index can never be matched by ANY doc, so counting it would
    // uniformly cap every doc's coverage at (T−1)/T — attenuating the whole
    // BM25 channel vs dense for one OOV query term, contradicting the bound.
    // Aligning the two means a doc matching every MATCHABLE term keeps full
    // lexical weight. (482-q eval revalidation of this alignment is pending —
    // harness session in flight.) Same defensive internals-reach as
    // getQueryBound; if the internals are missing we fall back to the
    // unfiltered count — the pre-fix behavior, which only ever UNDER-credits
    // coverage, never inflates it.
    private indexedQueryTerms(query: string): Set<string> {
        const all = distinctQueryTerms(query);
        const mini = this.mini as unknown as {
            _index?: { get(term: string): unknown };
        } | null;
        const index = mini?._index;
        if (!index || typeof index.get !== 'function') return all;
        const out = new Set<string>();
        for (const t of all) {
            if (index.get(t) !== undefined) out.add(t);
        }
        return out;
    }

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
        const bound = this.getQueryBound(query, opts?.boosts);

        // Distinct INDEXED query terms = coverage denominator (see
        // indexedQueryTerms for why OOV terms are excluded). >1 ⇒ a genuine
        // multi-term query where partial matches should be discounted; ≤1 ⇒ every
        // match covers the whole matchable query, so weight is a flat 1 (no-op).
        const totalTerms = this.indexedQueryTerms(query).size;

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
        let mateToSource: Map<string, string> | undefined;
        let synProcessTerm: ((term: string) => string | string[] | null) | undefined;
        if (synMap && synMap.size > 0) {
            const original = distinctQueryTerms(query);
            const baseProcess = allStopword ? keepStopwordsProcessTerm : processTerm;
            const m2s = new Map<string, string>();
            synProcessTerm = (raw: string) => {
                const t = baseProcess(raw);
                if (t === null) return null;
                const mates = synMap.get(t);
                if (!mates) return t;
                const out: string[] = [];
                for (const mate of mates) {
                    if (original.has(mate) || m2s.has(mate)) continue;
                    m2s.set(mate, t);
                    out.push(mate);
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
                // Synonym mates map back to their source term so a matched mate
                // fills its source's coverage slot, not a fresh one (review fix #1).
                const srcs = new Set<string>();
                for (const qt of r.queryTerms) srcs.add(m2s.get(qt) ?? qt);
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
        if (!this.idToIdx.has(chunk.chunk_id) && this.mini.has(chunk.chunk_id)) this.mini.discard(chunk.chunk_id);
        this.mini.add(this.buildDoc(chunk, body));
        this.termUpperBounds.clear();   // corpus changed → recompute UBs on next bound
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
        try { this.mini.discard(id); } catch { /* ghost posting from a tolerant load — already absent */ }
        // Drop memoized UBs: the next getQueryBound re-searches the live index,
        // whose results already exclude the discarded doc — so the bound honors
        // this tombstone synchronously, with no vacuum (the incremental contract).
        this.termUpperBounds.clear();
    }

    // Reclaim tombstoned postings so `_index` sizes equal live df again — REQUIRED
    // after a remove burst before the next getQueryBound/termDocFraction, or the
    // bound reads a stale (too-large) df → idf too small → bound too small →
    // TM2C2 fused scores can exceed 1. MiniSearch's vacuum() is async (it batches
    // postings cleanup across setTimeout), so callers MUST await it inside the
    // delta's critical section before re-stamping the cache generation.
    async vacuum(): Promise<void> {
        if (!this.mini) return;
        await this.mini.vacuum();
        this.termUpperBounds.clear();   // postings compacted → recompute UBs
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
