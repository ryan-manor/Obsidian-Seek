import { describe, it, expect } from 'vitest';
import { depluralize, foldDiacritics, MultiFieldBM25, DEFAULT_FIELD_BOOSTS, PREFIX_LAST_TOKEN, FUZZY_BY_LENGTH, extractPropertiesText, extractHeadingsText, BM25_COVERAGE_POW } from './bm25';
import type { Chunk } from './types';

function makeChunk(id: string, title: string, content: string, tags: string[] = [], aliases: string[] = []): Chunk {
    return {
        chunk_id: id,
        title,
        content,
        note_path: `${id}.md`,
        heading_path: [],
        metadata: { tags, aliases, created: null, modified: null, properties: {} },
        start_line: 0,
        end_line: 0,
    };
}

// fit() takes bodies separately since the v8 frame-lite split (search.ts holds
// metadata-only frames; bodies live in the chunk_body store). Test chunks carry
// content, so derive the bodies map from them and forward — keeps every call
// site below a one-liner.
function fitBM(chunks: Chunk[], opts?: { searchableProperties?: boolean; headingsField?: boolean }): MultiFieldBM25 {
    return new MultiFieldBM25().fit(chunks, new Map(chunks.map(c => [c.chunk_id, c.content])), opts);
}

// Locks the light plural->singular normalizer behavior. Mirrors the cases
// validated in the DBpedia BM25 study (2026-06-07, +0.0276 nDCG@10). The rule
// set is intentionally simple and runs symmetrically at index + query time, so
// even imperfect stems (series->sery) match themselves — that is expected.
describe('depluralize', () => {
    it('drops a simple trailing -s', () => {
        expect(depluralize('cars')).toBe('car');
        expect(depluralize('cuisines')).toBe('cuisine');
        expect(depluralize('dishes')).toBe('dish');
    });

    it('handles -ies -> -y', () => {
        expect(depluralize('countries')).toBe('country');
        expect(depluralize('cities')).toBe('city');
    });

    it('handles -es plurals (sses/ches/shes/xes/zes/ses)', () => {
        expect(depluralize('classes')).toBe('class');
        expect(depluralize('boxes')).toBe('box');
        expect(depluralize('buses')).toBe('bus');
    });

    it('leaves non-plural -s/-us/-is/-os/-ss words alone', () => {
        expect(depluralize('virus')).toBe('virus');
        expect(depluralize('axis')).toBe('axis');
        expect(depluralize('class')).toBe('class');
        expect(depluralize('kiss')).toBe('kiss');
    });

    it('leaves already-singular words and short words alone', () => {
        expect(depluralize('cuisine')).toBe('cuisine');
        expect(depluralize('car')).toBe('car');
        expect(depluralize('is')).toBe('is');
    });

    it('is idempotent (running twice == running once)', () => {
        for (const w of ['cars', 'countries', 'classes', 'virus', 'cuisine',
            'aliases', 'lenses', 'movies', 'analyses', 'news']) {
            expect(depluralize(depluralize(w))).toBe(depluralize(w));
        }
    });

    // ---- Audit 2026-06-09 §5 bug classes -------------------------------
    // Each block pins one row of the audit table: singular and plural must
    // land on the SAME stem (co-match), and reductions must not collide with
    // proper names. Verified broken by execution before the exception tables.

    it('s-final singulars co-match their plurals (was alias->alia vs aliases->alias)', () => {
        for (const [sing, plur] of [
            ['alias', 'aliases'], ['bias', 'biases'],
            ['atlas', 'atlases'], ['canvas', 'canvases'], ['lens', 'lenses'],
        ] as const) {
            expect(depluralize(sing)).toBe(sing);
            expect(depluralize(plur)).toBe(sing);
        }
    });

    it('s-final guards kill the proper-name collisions (Alia, Len, new)', () => {
        // Names/short words pass through untouched; the guarded singulars no
        // longer reduce onto them.
        expect(depluralize('alia')).toBe('alia');
        expect(depluralize('alias')).not.toBe('alia');
        expect(depluralize('len')).toBe('len');
        expect(depluralize('lens')).not.toBe('len');
        expect(depluralize('news')).toBe('news');   // not "new"
    });

    it('-ie nouns co-match their plurals (was movies->movy vs movie)', () => {
        for (const [sing, plur] of [
            ['movie', 'movies'], ['cookie', 'cookies'],
            ['zombie', 'zombies'], ['calorie', 'calories'], ['selfie', 'selfies'],
        ] as const) {
            expect(depluralize(sing)).toBe(sing);
            expect(depluralize(plur)).toBe(sing);
        }
    });

    it('-ies common nouns still reduce to -y (regression guard for the -ie fix)', () => {
        expect(depluralize('countries')).toBe('country');
        expect(depluralize('queries')).toBe('query');
        expect(depluralize('dailies')).toBe('daily');
    });

    it('Greek/Latin -is plurals co-match their singulars (was analyses->analys)', () => {
        for (const [plur, sing] of [
            ['analyses', 'analysis'], ['theses', 'thesis'], ['crises', 'crisis'],
            ['diagnoses', 'diagnosis'], ['hypotheses', 'hypothesis'],
            ['parentheses', 'parenthesis'], ['syntheses', 'synthesis'],
        ] as const) {
            expect(depluralize(plur)).toBe(sing);
            expect(depluralize(sing)).toBe(sing);
        }
    });

    it('axes/bases are deliberately NOT mapped (axe/axis, base/basis ambiguity)', () => {
        // Either mapping would break the other word; they keep the old
        // symmetric behavior. This test documents the decision.
        expect(depluralize('axes')).toBe('axe');
        expect(depluralize('bases')).toBe('bas');   // -ses rule; imperfect but symmetric
    });

    it('is prototype-safe on object-property tokens (Map, not object literal)', () => {
        expect(depluralize('constructor')).toBe('constructor');
        expect(depluralize('hasownproperty')).toBe('hasownproperty');
        expect(depluralize('prototype')).toBe('prototype');
    });
});

// Locks the admin-panel mechanism: getScores must honor a PER-CALL field-boost
// override. If MiniSearch ignored the per-call boost (or we baked it into the
// index instead), these would fail and the settings controls would be inert.
describe('per-search field boosts', () => {
    it('applies a per-call boost override (tag-only match scales with tag boost)', () => {
        // The query term lives ONLY in the tags field, so its score is purely
        // the tag-field contribution × the tag boost.
        const idx = fitBM([
            makeChunk('a', 'Alpha', 'body text about cats', ['zzqterm']),
            makeChunk('b', 'Beta', 'unrelated body', ['other']),
        ]);
        const low = idx.getScores('zzqterm', { boosts: { ...DEFAULT_FIELD_BOOSTS, tags: 1 } });
        const high = idx.getScores('zzqterm', { boosts: { ...DEFAULT_FIELD_BOOSTS, tags: 10 } });
        expect(low[0]).toBeGreaterThan(0);          // doc 'a' matches
        expect(high[0]).toBeGreaterThan(low[0]);    // higher boost → higher score
        expect(high[1]).toBe(0);                    // doc 'b' never matches the term
    });

    it('keeps other field weights when only one boost is overridden', () => {
        // Term in the title of A and the tags of B. With a huge tag boost, the
        // tag-match doc must overtake the title-match doc — proving the override
        // changes ranking, not just magnitude.
        const idx = fitBM([
            makeChunk('a', 'zzqterm', 'plain body'),                 // title match
            makeChunk('b', 'Beta', 'plain body', ['zzqterm']),      // tag match
        ]);
        const tagHeavy = idx.getScores('zzqterm', { boosts: { ...DEFAULT_FIELD_BOOSTS, tags: 50 } });
        expect(tagHeavy[1]).toBeGreaterThan(tagHeavy[0]);
    });

    it('getScores with no opts still scores matches (default path)', () => {
        const idx = fitBM([
            makeChunk('a', 'Alpha', 'body text about cats', ['zzqterm']),
        ]);
        expect(idx.getScores('zzqterm')[0]).toBeGreaterThan(0);
    });
});

// Locks the fuzzy toggle: "on" = absolute edit distance 1. Off (the default and
// the index's baked value) must NOT match a typo; on must.
describe('fuzzy toggle', () => {
    it('fuzzy:1 matches an edit-distance-1 typo; exact does not', () => {
        const idx = fitBM([
            makeChunk('a', 'Cuisine', 'french cooking'),
        ]);
        // "cuisin" is Levenshtein distance 1 from the indexed "cuisine" (one
        // deletion). prefix:false stays off, so only fuzzy can bridge the gap.
        expect(idx.getScores('cuisin')[0]).toBe(0);                  // exact: miss
        expect(idx.getScores('cuisin', { fuzzy: false })[0]).toBe(0); // explicit off: miss
        expect(idx.getScores('cuisin', { fuzzy: 1 })[0]).toBeGreaterThan(0); // on: hit
    });

    it('fuzzy:1 does NOT match an edit-distance-2 difference', () => {
        const idx = fitBM([
            makeChunk('a', 'Cuisine', 'french cooking'),
        ]);
        // "cuyson": distance 2 from "cuisine" — beyond the dist-1 budget.
        expect(idx.getScores('cuyson', { fuzzy: 1 })[0]).toBe(0);
    });
});

// Locks the last-token prefix lever (2026-06-10 eval, settings.prefixLastToken).
// The motivating gap: "rearch" is 6 edits from "rearchitecture" — unreachable
// by exact OR fuzzy; only prefix expansion bridges it. The predicate's two
// guards (last token only, ≥3 chars) are each pinned by a test, because each
// was an eval-justified rejection (all-tokens taxed the D&D desc stratum and
// fanned out ~1813 derived terms/q on identifier-heavy corpora).
describe('prefix toggle (last token)', () => {
    it('PREFIX_LAST_TOKEN matches a strict prefix of an indexed term; default/off do not', () => {
        const idx = fitBM([
            makeChunk('a', 'Rearchitecture Meeting', 'lightroom ac plans'),
        ]);
        expect(idx.getScores('rearch')[0]).toBe(0);                          // baked default: miss
        expect(idx.getScores('rearch', { prefix: false })[0]).toBe(0);      // explicit off: miss
        expect(idx.getScores('rearch', { prefix: PREFIX_LAST_TOKEN })[0]).toBeGreaterThan(0);
    });

    it('only the LAST query token expands', () => {
        const idx = fitBM([
            makeChunk('a', 'Rearchitecture', 'design doc'),
        ]);
        // "rearch" first, real term last: "rearch" must NOT expand → doc has
        // neither exact term → no match. Reversed order puts "rearch" last → hit.
        expect(idx.getScores('rearch zzmissing', { prefix: PREFIX_LAST_TOKEN })[0]).toBe(0);
        expect(idx.getScores('zzmissing rearch', { prefix: PREFIX_LAST_TOKEN })[0]).toBeGreaterThan(0);
    });

    it('does not expand a last token under 3 chars', () => {
        const idx = fitBM([
            makeChunk('a', 'Rearchitecture', 'design doc'),
        ]);
        expect(idx.getScores('re', { prefix: PREFIX_LAST_TOKEN })[0]).toBe(0);
    });

    it('an exact match outranks a prefix-derived match (discounted weight)', () => {
        // Same term position/field; doc 'b' holds the exact query term, doc 'a'
        // only a longer extension. MiniSearch weights derived terms at
        // 0.375·len/len(derived), so the exact doc must rank first.
        const idx = fitBM([
            makeChunk('a', 'Rearchitecture', 'design doc'),
            makeChunk('b', 'Rearch', 'design doc'),
        ]);
        const s = idx.getScores('rearch', { prefix: PREFIX_LAST_TOKEN });
        expect(s[0]).toBeGreaterThan(0);
        expect(s[1]).toBeGreaterThan(s[0]);
    });

    it('coverage credits the source term for a prefix-derived match', () => {
        // Two-term query; doc 'a' matches "meeting" exactly and "rearch" only
        // via the prefix expansion. Doc 'b' puts the literal "rearch" in the
        // vocabulary so the term counts in the coverage DENOMINATOR (otherwise
        // indexedQueryTerms drops it and coverage is trivially 1). queryTerms
        // maps the derived match back to the source term, so doc 'a' must get
        // coverage 2/2 = 1, not 1/2.
        const idx = fitBM([
            makeChunk('a', 'Rearchitecture Meeting', 'lightroom ac plans'),
            makeChunk('b', 'Rearch', 'unrelated stub'),
        ]);
        const { coverage } = idx.getScoresWithCoverage('meeting rearch', { prefix: PREFIX_LAST_TOKEN });
        expect(coverage[0]).toBe(1);
    });
});

describe('synonym expansion (per-search overrides)', () => {
    // Dictionary the tests share: lr ↔ lightroom, both directions.
    const SYN = { map: new Map([['lr', ['lightroom']], ['lightroom', ['lr']]]), weight: 0.8 };

    it('a trigger term reaches a doc that only spells out the canonical form', () => {
        const idx = fitBM([
            makeChunk('a', 'Lightroom Roadmap', 'plans'),
        ]);
        const without = idx.getScores('lr roadmap');
        const withSyn = idx.getScores('lr roadmap', { synonyms: SYN });
        expect(withSyn[0]).toBeGreaterThan(without[0]);   // mate adds the lr-slot match
    });

    it('a mate-only match scores at the dictionary discount vs the exact term', () => {
        // Same field, same df, same field length — the ONLY difference is the
        // boostTerm discount, so the ratio must be exactly the weight.
        const idx = fitBM([
            makeChunk('a', 'Lightroom', 'same body'),
            makeChunk('b', 'Lr', 'same body'),
        ]);
        const s = idx.getScores('lr', { synonyms: SYN });
        expect(s[0]).toBeGreaterThan(0);
        expect(s[0] / s[1]).toBeCloseTo(0.8, 6);
    });

    it('coverage maps a mate match back to its source term (review fix #1)', () => {
        // Query "lr roadmap" (denominator 2: both indexed). Doc 'a' = the
        // content target: roadmap exact + lr via mate → coverage 1. Doc 'c' =
        // the alias-owner shape, matching lr AND lightroom but never roadmap:
        // native attribution would call that 2/2 = 1.0 and defeat the
        // soft-AND; the source mapping must keep it at 1/2.
        const idx = fitBM([
            makeChunk('a', 'Lightroom Roadmap', 'plans'),
            makeChunk('b', 'Lr', 'stub'),
            makeChunk('c', 'Lr Hub', 'lr lightroom links'),
        ]);
        const { coverage } = idx.getScoresWithCoverage('lr roadmap', { synonyms: SYN });
        expect(coverage[0]).toBe(1);
        expect(coverage[2]).toBe(0.5);
    });

    it('never injects a mate that is already a query term (review fix #2)', () => {
        // "lightroom lr": each term's only mate IS the other query term, so
        // nothing may be injected — scores must equal the no-synonyms run
        // exactly (an injected duplicate would score one term ~1.8×).
        const idx = fitBM([
            makeChunk('a', 'Lightroom', 'lr notes'),
        ]);
        const without = idx.getScores('lightroom lr');
        const withSyn = idx.getScores('lightroom lr', { synonyms: SYN });
        expect(withSyn[0]).toBe(without[0]);
    });

    it('mates-first ordering keeps PREFIX_LAST_TOKEN on the true last token', () => {
        // "lr rearch" expands to [lightroom, lr, rearch]: the mate must come
        // BEFORE its source so "rearch" keeps the final index and still
        // prefix-derives "rearchitecture". A doc reachable only through that
        // derivation proves the predicate fired on the right term.
        const idx = fitBM([
            makeChunk('a', 'Rearchitecture', 'design doc'),
        ]);
        const s = idx.getScores('lr rearch', { synonyms: SYN, prefix: PREFIX_LAST_TOKEN });
        expect(s[0]).toBeGreaterThan(0);
    });

    it('×quality double-credit is rescaled back to source attribution', () => {
        // Query "lr": doc Y matches the term AND its mate, so MiniSearch
        // multiplies its raw score by quality=2 — the double-credit the
        // native-attribution gate measured as fatal (it doubles exactly the
        // alias hub/sibling pages). The rescale divides it back out, so Y
        // must score the SUM of its parts, not twice the sum. Docs X and Z
        // isolate the parts: identical field lengths and dfs (lr in {X,Y},
        // lightroom in {Y,Z}) make the per-term scores equal across docs.
        const idx = fitBM([
            makeChunk('x', 'One', 'lr zz'),
            makeChunk('y', 'Two', 'lr lightroom'),
            makeChunk('z', 'Three', 'zz lightroom'),
        ]);
        const s = idx.getScores('lr', { synonyms: SYN });
        expect(s[1]).toBeCloseTo(s[0] + s[2], 6);
    });

    it('composes with the FUZZY_BY_LENGTH predicate (post-multilingual rebase)', () => {
        // search.ts passes fuzzy as the FUZZY_BY_LENGTH FUNCTION. The synonym
        // wrapper must DELEGATE to it per term — returning the function as a
        // value (the pre-rebase scalar assumption) is truthy and would re-enable
        // fuzzy on exactly the CJK/digit/short terms the predicate excludes.
        const idx = fitBM([
            makeChunk('a', 'A', '我 喜欢 猫'),               // cat
            makeChunk('b', 'B', '我 喜欢 狗'),               // dog (edit-1 of 猫 as a term)
            makeChunk('c', 'Lightroom', 'granite embeddings'),
        ]);
        const opts = { synonyms: SYN, fuzzy: FUZZY_BY_LENGTH };
        const cjk = idx.getScores('猫', opts);
        expect(cjk[0]).toBeGreaterThan(0);
        expect(cjk[1]).toBe(0);              // CJK stays fuzzy-excluded with synonyms active
        const latin = idx.getScores('granit', opts);
        expect(latin[2]).toBeGreaterThan(0); // Latin typo rescue still delegates through
        // Mates stay dictionary-exact: "lr" injects "lightroom", which must
        // match doc c's title exactly but never fuzzy-derive anything new —
        // same contract as the scalar-fuzzy era.
        const viaMate = idx.getScores('lr', opts);
        expect(viaMate[2]).toBeGreaterThan(0);
    });
});

// Locks the theoretical query bound (fusion's TM2C2 denominator for the BM25
// channel) — now the WAND/MaxScore tight bound Σ_t UB(t)·D, where UB(t) is the max
// achievable exact single-term score read from the LIVE index via MiniSearch's own
// scorer. The one-term parity test is load-bearing: bound == the live max score by
// construction (no replicated formula left to drift), so a MiniSearch scoring change
// surfaces here in CI; the runtime guard still falls back to empirical /max if the
// index is missing.
describe('getQueryBound', () => {
    it('equals the max single-term score for a one-term query (WAND/MaxScore tight bound)', () => {
        // The tight bound for a single indexed term is exactly that term's max
        // achievable EXACT candidate score over the corpus: UB(t)·D with D=1.
        // getQueryBound drives MiniSearch's OWN scorer, so it reproduces the live
        // max score by construction — the old analytic sup·(d+k1+1) over-stated it
        // (it summed the tf→∞ saturation a real doc can't reach).
        const idx = fitBM([
            makeChunk('a', 'Alpha', 'zzqterm xx'),
            makeChunk('b', 'Beta', 'yyother xx'),
        ]);
        // Same boosts on both sides so scores and bound are strictly comparable.
        const { scores, bound } = idx.getScoresWithCoverage('zzqterm', { boosts: DEFAULT_FIELD_BOOSTS });
        const maxScore = Math.max(...scores);
        expect(maxScore).toBeGreaterThan(0);
        expect(bound).toBeCloseTo(maxScore, 8);
    });

    it('bounds every exact-match score (sup property)', () => {
        const idx = fitBM([
            makeChunk('a', 'Switzerland', 'hotel hotel hotel in switzerland', ['travel'], ['Suisse']),
            makeChunk('b', 'Locke am Platz', 'nice hotel in zurich switzerland'),
            makeChunk('c', 'Hotel', 'hotel'),
        ]);
        for (const q of ['switzerland hotel', 'hotel', 'suisse travel switzerland']) {
            const { scores, bound } = idx.getScoresWithCoverage(q);
            expect(bound).toBeGreaterThan(0);
            for (const s of scores) expect(s).toBeLessThanOrEqual(bound + 1e-9);
        }
    });

    it('divides out MiniSearch quality so a 2-term match scores the additive sum (single soft-AND)', () => {
        // De-franken 2026-06-26: MiniSearch's search() returns score × quality
        // (quality = |distinct query terms matched|, dist v7.2.0:1291). That is a
        // hidden term-count boost that USED to stack on Seek's coverage soft-AND
        // (an m × m/T = m²/T effect) and forced the bound's ×D mirror.
        // getScoresWithCoverage now divides quality back out, so the stored score
        // is the pure additive BM25F sum and the soft-AND is applied exactly once,
        // via coverage. A doc matching BOTH terms therefore scores the SUM of its
        // single-term scores — NOT 2× the sum.
        const idx = fitBM([
            makeChunk('a', 'Switzerland', 'hotel hotel hotel in switzerland'),
            makeChunk('b', 'Beta', 'unrelated body'),
        ]);
        const sw = idx.getScores('switzerland')[0];
        const ho = idx.getScores('hotel')[0];
        const both = idx.getScores('switzerland hotel')[0];
        expect(sw).toBeGreaterThan(0);
        expect(ho).toBeGreaterThan(0);
        expect(both).toBeCloseTo(sw + ho, 8);
    });

    it('is 0 for a fully-OOV query and skips OOV terms in mixed queries', () => {
        const idx = fitBM([
            makeChunk('a', 'Alpha', 'real content here'),
        ]);
        expect(idx.getQueryBound('zzznope qqqmiss')).toBe(0);
        // Mixed: bound counts only the indexed term — same value as alone.
        expect(idx.getQueryBound('zzznope content')).toBeCloseTo(idx.getQueryBound('content'), 10);
    });

    it('scales with the per-call field-boost override (same lever as getScores)', () => {
        const idx = fitBM([
            makeChunk('a', 'Alpha', 'body', ['zzqterm']),
        ]);
        const low = idx.getQueryBound('zzqterm', { ...DEFAULT_FIELD_BOOSTS, tags: 1 });
        const high = idx.getQueryBound('zzqterm', { ...DEFAULT_FIELD_BOOSTS, tags: 10 });
        expect(low).toBeGreaterThan(0);
        expect(high).toBeCloseTo(low * 10, 8); // term only exists in tags
    });
});

// Locks the soft-AND coverage signal: per doc, the fraction of DISTINCT query
// terms it matched. This is what search.ts multiplies into raw BM25 to demote a
// single-term saturator under a multi-term query. getScores must stay identical
// to getScoresWithCoverage().scores (the delegation contract).
describe('coverage weighting (soft AND)', () => {
    it('reports full coverage for a doc matching every query term, partial for a subset', () => {
        const idx = fitBM([
            makeChunk('a', 'Switzerland', 'the country in the alps'),     // matches "switzerland" only
            makeChunk('b', 'Locke am Platz', 'nice hotel in zurich switzerland'), // matches both
            makeChunk('c', 'Hotel Boulderado', 'colorado lodging'),       // matches "hotel" only
        ]);
        const { scores, coverage } = idx.getScoresWithCoverage('switzerland hotel');
        // Two distinct query terms → full-match doc = 1.0, single-term = 0.5.
        expect(coverage[1]).toBeCloseTo(1.0, 10);   // Locke: switzerland + hotel
        expect(coverage[0]).toBeCloseTo(0.5, 10);   // Switzerland page: 1 of 2
        expect(coverage[2]).toBeCloseTo(0.5, 10);   // Boulderado: 1 of 2
        // Coverage is recall-safe: a partial match is discounted, never zeroed.
        expect(scores[0]).toBeGreaterThan(0);
    });

    it('is a no-op for single-term queries (every match covers the whole query)', () => {
        const idx = fitBM([
            makeChunk('a', 'Switzerland', 'the country'),
            makeChunk('b', 'Beta', 'unrelated'),
        ]);
        const { scores, coverage } = idx.getScoresWithCoverage('switzerland');
        expect(coverage[0]).toBeCloseTo(1.0, 10);   // matched → weight 1 (no-op)
        expect(coverage[1]).toBe(0);                // non-match stays 0
        // getScores delegates, so its scores equal the coverage-pass scores.
        expect(Array.from(idx.getScores('switzerland'))).toEqual(Array.from(scores));
    });

    it('depluralizes the coverage denominator symmetrically with BM25 ("parks" = "park")', () => {
        const idx = fitBM([
            makeChunk('a', 'Miyashita Park', 'a park in tokyo japan'),  // matches japan + park
        ]);
        // q = {japanese→… , parks→park}. "parks" depluralizes to "park" on both
        // sides, so the 2-term query is {japan-ish, park}; the doc matches "park"
        // and "japan" here → both terms covered.
        const { coverage } = idx.getScoresWithCoverage('japan parks');
        expect(coverage[0]).toBeCloseTo(1.0, 10);
    });

    // The denominator is INDEXED distinct terms, mirroring getQueryBound's D
    // (2026-06-09 review fix): an OOV query term can never be matched by any
    // doc, so it must not depress every doc's coverage — under bound-norm that
    // attenuation would NOT cancel and would re-weight the whole fusion.
    it('excludes OOV query terms from the coverage denominator (mirrors the bound)', () => {
        const idx = fitBM([
            makeChunk('a', 'Switzerland', 'the country in the alps'),
            makeChunk('b', 'Locke am Platz', 'nice hotel in zurich switzerland'),
        ]);
        // "zzznope" is indexed nowhere → denominator stays 2 (switzerland,
        // hotel), so the all-indexed-terms match keeps FULL lexical weight
        // (pre-fix it was capped at 2/3) and the partial match keeps 1/2.
        const { coverage } = idx.getScoresWithCoverage('switzerland hotel zzznope');
        expect(coverage[1]).toBeCloseTo(1.0, 10);   // Locke: both indexed terms
        expect(coverage[0]).toBeCloseTo(0.5, 10);   // Switzerland page: 1 of 2
    });

    it('treats one indexed term + OOV noise as a single-term query (flat 1)', () => {
        const idx = fitBM([
            makeChunk('a', 'Switzerland', 'the country'),
        ]);
        // Only "switzerland" is matchable → denominator 1 → the multi-term
        // discount gate doesn't open; every match covers the matchable query.
        const { coverage } = idx.getScoresWithCoverage('switzerland qqqmiss');
        expect(coverage[0]).toBeCloseTo(1.0, 10);
    });

    it('clamps coverage at 1 when fuzzy matches a term outside the indexed denominator', () => {
        const idx = fitBM([
            makeChunk('a', 'Locke am Platz', 'nice hotel in zurich switzerland'),
        ]);
        // "hotl" is OOV exactly (excluded from the denominator of 2) but
        // fuzzy-matches "hotel", so the doc matches 3 query terms — same
        // exception the bound documents for fuzzy. Coverage clamps to 1.
        const { coverage } = idx.getScoresWithCoverage('switzerland zurich hotl', { fuzzy: 1 });
        expect(coverage[0]).toBeCloseTo(1.0, 10);
    });
});

// Three-Lens S2 (2026-06-10): title/aliases are NAME fields — exempt from the
// stoplist at index time (processTerm's fieldName arg), with an all-stopword
// QUERY fallback (mirrors fusion.ts titleMatchBoost). The beneficiary class
// (a note literally titled "Will") is structurally absent from the
// click-derived eval — these tests are its gate. Harness arm verdict
// (stopname_arm.py): mode (b) free everywhere (personal −0.0003, dnd/code
// 0.0000); mode (a) "keep always" REJECTED (dnd −0.0147).
describe('S2 per-field stopword exemption (name-as-stopword)', () => {
    it('an all-stopword query finds a note literally titled with a stopword', () => {
        const idx = fitBM([
            makeChunk('a', 'Will', 'notes from coffee earlier'),
            makeChunk('b', 'Project Plan', 'we will need to plan the work'),
        ]);
        const s = idx.getScores('will');
        expect(s[0]).toBeGreaterThan(0); // the 10x-boosted title channel is alive
        expect(s[1]).toBe(0);            // prose "will" stays stopworded in content
    });

    it('the exemption covers aliases', () => {
        const idx = fitBM([
            makeChunk('a', 'William Tan', 'body text', [], ['Will']),
            makeChunk('b', 'Unrelated Note', 'body text'),
        ]);
        const s = idx.getScores('will');
        expect(s[0]).toBeGreaterThan(0);
        expect(s[1]).toBe(0);
    });

    it('queries with a content word are untouched — stopwords still drop', () => {
        // "the" is indexed in doc b's title now, but the query has content
        // words, so the fallback must NOT fire; "the" never gets queried.
        const idx = fitBM([
            makeChunk('a', 'Eames Project', 'body one'),
            makeChunk('b', 'The Daily', 'body two'),
        ]);
        const s = idx.getScores('the eames project');
        expect(s[0]).toBeGreaterThan(0);
        expect(s[1]).toBe(0); // reachable only via "the" → must not match
    });

    it('bound, coverage, and scores stay coherent on the fallback path', () => {
        const idx = fitBM([
            makeChunk('a', 'Will', 'short body here'),
            makeChunk('b', 'Will Smith', 'other body here'),
        ]);
        const { scores, coverage, bound } = idx.getScoresWithCoverage('will');
        expect(bound).toBeGreaterThan(0);             // bound covers the kept literal term
        expect(scores[0]).toBeGreaterThan(0);
        expect(scores[0]).toBeLessThanOrEqual(bound); // sup property holds on the fallback
        expect(scores[1]).toBeGreaterThan(0);
        expect(coverage[0]).toBeCloseTo(1.0, 10);     // single-term query → flat 1
    });

    it('content and tags keep the stoplist (exemption is per-field, not global)', () => {
        // Stopword appears ONLY in content/tags — must remain unindexed there.
        const idx = fitBM([
            makeChunk('a', 'Groceries', 'this is the list for the week', ['the']),
        ]);
        expect(idx.getScores('the')[0]).toBe(0);
    });
});

// CJK segmentation end-to-end through MiniSearch (2026-06-10): the tokenize
// option (tokenize.ts seekTokenize) must make unsegmented zh/ja content
// findable by sub-phrase queries — pre-fix, a CJK sentence indexed as ONE
// giant token and only an exact full-sentence query could hit it.
describe('CJK tokenization (Intl.Segmenter)', () => {
    const chunks = [
        makeChunk('zh1', 'Hotels', '我喜欢在瑞士的酒店住宿'),
        makeChunk('ja1', 'Ramen', '東京のラーメン屋を検索する'),
        makeChunk('en1', 'Plain', 'plain english body about hotels'),
    ];
    const bm = fitBM(chunks);

    it('zh sub-phrase query hits the zh doc', () => {
        const s = bm.getScores('瑞士 酒店');
        expect(s[0]).toBeGreaterThan(0);    // zh1
        expect(s[1]).toBe(0);               // ja1 unrelated
    });

    it('ja word query hits the ja doc', () => {
        const s = bm.getScores('ラーメン');
        expect(s[1]).toBeGreaterThan(0);
    });

    it('English scoring is unaffected by the tokenizer swap', () => {
        const s = bm.getScores('english hotels');
        expect(s[2]).toBeGreaterThan(0);
        expect(s[1]).toBe(0);
    });

    it('theoretical bound covers segmented CJK terms (no /max fallback)', () => {
        expect(bm.getQueryBound('瑞士 酒店')).toBeGreaterThan(0);
    });
});

// FUZZY_BY_LENGTH (audit 2026-06-18 §1/§2): typo tolerance scaled by term length
// (Lucene AUTO ladder ≤2 exact / 3–5 = 1 / ≥6 = 2), skipping CJK terms (edit-1 on
// a segmented han char = synonym explosion, fused ja 0.692-vs-0.928 on Belebele)
// and any digit-bearing token (a year/version typo is a different thing, not a
// correctable word). Replaces the old flat edit-1 FUZZY_NON_CJK.
describe('FUZZY_BY_LENGTH predicate', () => {
    it('scales edit distance by term length (≤2 exact / 3–5 edit-1 / ≥6 edit-2)', () => {
        // The Lucene AUTO ladder in bm25.ts FUZZY_BY_LENGTH: ≤2 → exact, 3–5 →
        // edit-1, ≥6 → edit-2 (CJK + digit-bearing exemptions fire before it).
        expect(FUZZY_BY_LENGTH('ml')).toBe(false);        // 2 → exact
        expect(FUZZY_BY_LENGTH('cat')).toBe(1);           // 3 → edit-1 (first fuzzy rung)
        expect(FUZZY_BY_LENGTH('data')).toBe(1);          // 4 → edit-1
        expect(FUZZY_BY_LENGTH('graph')).toBe(1);         // 5 → edit-1
        expect(FUZZY_BY_LENGTH('verge')).toBe(1);         // 5 → edit-1 (the motivating token)
        expect(FUZZY_BY_LENGTH('matrix')).toBe(2);        // 6 → edit-2 (first edit-2 rung)
        expect(FUZZY_BY_LENGTH('granite')).toBe(2);       // 7 → edit-2
        expect(FUZZY_BY_LENGTH('keywords')).toBe(2);      // 8 → edit-2
        expect(FUZZY_BY_LENGTH('amsterdam')).toBe(2);     // 9 → edit-2 (long-word, sparse nbhd)
        expect(FUZZY_BY_LENGTH('pumpernickel')).toBe(2);  // 12 → edit-2
    });

    it('exempts CJK terms (edit-1 there is a synonym explosion)', () => {
        expect(FUZZY_BY_LENGTH('手风琴')).toBe(false);
        expect(FUZZY_BY_LENGTH('ラーメン')).toBe(false);
        expect(FUZZY_BY_LENGTH('검색')).toBe(false);
    });

    it('exempts any digit-bearing token (IDs/numbers are typed exactly)', () => {
        expect(FUZZY_BY_LENGTH('2024')).toBe(false);      // year: must not edit-1 to 2023/2025
        expect(FUZZY_BY_LENGTH('gpt4')).toBe(false);      // alphanumeric ID
        expect(FUZZY_BY_LENGTH('k8s')).toBe(false);
        expect(FUZZY_BY_LENGTH('granite2')).toBe(false);  // digit beats the ≥7 edit-2 rung
    });

    it('CJK query does not fuzzy-match a different single-char word', () => {
        const chunks = [
            makeChunk('a', 'A', '我 喜欢 猫'),       // cat
            makeChunk('b', 'B', '我 喜欢 狗'),       // dog
        ];
        const bm = fitBM(chunks);
        const sGuarded = bm.getScores('猫', { fuzzy: FUZZY_BY_LENGTH });
        expect(sGuarded[0]).toBeGreaterThan(0);
        expect(sGuarded[1]).toBe(0);                 // 狗 must NOT match via edit-1
        const sUnguarded = bm.getScores('猫', { fuzzy: 1 });
        expect(sUnguarded[1]).toBeGreaterThan(0);    // documents the explosion the guard prevents
    });

    it('a long Latin typo is still rescued (6-char word gets edit-2)', () => {
        const chunks = [makeChunk('a', 'A', 'granite embeddings')];
        const bm = fitBM(chunks);
        expect(bm.getScores('granit', { fuzzy: FUZZY_BY_LENGTH })[0]).toBeGreaterThan(0);
    });

    it('a 2-char typo is NO LONGER rescued (≤2 exact — the bug the audit found)', () => {
        const chunks = [makeChunk('a', 'A', 'ml pipelines')];
        const bm = fitBM(chunks);
        expect(bm.getScores('al', { fuzzy: FUZZY_BY_LENGTH })[0]).toBe(0);     // no edit-1 on 2-char
        expect(bm.getScores('al', { fuzzy: 1 })[0]).toBeGreaterThan(0);        // flat edit-1 WOULD (old bug)
    });
});

// Searchable-properties field (2026-06-11): frontmatter VALUES as a 6th BM25
// field, gated by fit({ searchableProperties }). Harness gate: props_field_arm
// (captures +0.059 @ boost 2, personal net-zero). The normalizer is NOT a
// custom analyzer — values are unwrapped/filtered, then the standard
// tokenize/processTerm pipeline applies.
describe('searchable properties field', () => {
    function placeChunk(id: string, title: string, props: Record<string, string>): Chunk {
        const c = makeChunk(id, title, 'cheery eatery with a colorful menu');
        c.metadata.properties = props;
        return c;
    }
    const sf = placeChunk('sf', 'La Mar', { placeLoc: '[[San Francisco]]', placeType: 'restaurants' });
    const austin = placeChunk('atx', "Sap's Thai", { placeLoc: '[[Austin]]', placeType: 'restaurants', icon: 'utensils' });

    it('extractPropertiesText unwraps wikilinks and drops machinery/date/number values', () => {
        expect(extractPropertiesText(sf)).toBe('San Francisco restaurants');
        // icon is a machinery key; created/coordinates-style values are dropped
        const c = placeChunk('x', 'X', {
            placeLoc: '[[Austin|ATX]]', icon: 'utensils', created: '2026-01-30',
            rating: '4.5', context: 'personal',
        });
        // display-form indexes the target basename only — the "ATX" alias is
        // dropped (no path/alias stuffing in this boosted field).
        expect(extractPropertiesText(c)).toBe('Austin personal');
    });

    it('a date-prefixed value keeps its trailing free text (audit R2 batch2 #1)', () => {
        const c = placeChunk('trip', 'Trip', { trip: '2026-06-29 Milan departure' });
        expect(extractPropertiesText(c)).toContain('Milan');
        expect(extractPropertiesText(c)).toContain('departure');
        // a bare date (no trailing text) is still dropped
        const bare = placeChunk('bare', 'Bare', { trip: '2026-06-29' });
        expect(extractPropertiesText(bare)).toBe('');
        // a date + ISO time-of-day tail (no free text) is still dropped too
        const withTime = placeChunk('time', 'Time', { trip: '2026-06-29T14:30:00Z' });
        expect(extractPropertiesText(withTime)).toBe('');
    });

    it('list-valued properties fold into the field per-item (audit R2 batch2 #3)', () => {
        const c = makeChunk('rel', 'Project Apollo', 'launch planning notes');
        c.metadata.properties = { relatedPages: ['Mission Control', 'Ground Crew'], context: 'work' };
        expect(extractPropertiesText(c)).toBe('Mission Control Ground Crew work');
    });

    it('a list-valued property is still date/number type-filtered per item', () => {
        const c = makeChunk('rel2', 'Trip Log', 'a trip log body');
        c.metadata.properties = { dates: ['2026-06-29', 'Milan'], nums: ['4.5'] };
        expect(extractPropertiesText(c)).toBe('Milan');
    });

    it('pageType folds into the generic properties field by name (no dedicated page_type field)', () => {
        // De-specialization 2026-06-29: `pageType` (a Task-Notes / this-vault
        // convention, not an Obsidian builtin) no longer gets a dedicated
        // `page_type` BM25 field. It rides the generic properties field like any
        // other scalar key — so it's only lexically searchable when the
        // searchable-properties setting is on, while `[pageType:x]` filters and
        // the dense suffix (other code paths) keep working regardless.
        const note = placeChunk('mtg', 'Standup', { pageType: 'meeting', context: 'work' });
        expect(extractPropertiesText(note)).toBe('meeting work'); // not excluded as machinery
        expect(fitBM([note]).getScores('meeting')[0]).toBe(0);    // off → invisible to free text
        const on = fitBM([note], { searchableProperties: true });
        expect(on.getScores('meeting')[0]).toBeGreaterThan(0);    // on → searchable via properties
    });

    it('property values match plain query words only when the field is enabled', () => {
        const off = fitBM([sf, austin]);
        expect(off.getScores('francisco')[0]).toBe(0);
        const on = fitBM([sf, austin], { searchableProperties: true });
        expect(on.getScores('francisco')[0]).toBeGreaterThan(0);
        expect(on.getScores('francisco')[1]).toBe(0); // Austin note untouched
    });

    it('getQueryBound sees the properties field when enabled (df>0 path)', () => {
        const off = fitBM([sf, austin]);
        expect(off.getQueryBound('francisco')).toBe(0);
        const on = fitBM([sf, austin], { searchableProperties: true });
        expect(on.getQueryBound('francisco')).toBeGreaterThan(0);
    });

    it('analyzer pipeline applies to property terms (depluralize: restaurant ~ restaurants)', () => {
        const on = fitBM([sf, austin], { searchableProperties: true });
        expect(on.getScores('restaurant')[0]).toBeGreaterThan(0);
        expect(on.getScores('restaurant')[1]).toBeGreaterThan(0);
    });
});

// Headings field (2026-06-12): section heading path as a BM25 field, gated by
// fit({ headingsField }). Without it heading words are BM25-invisible —
// extractNoteName strips the path from `title` and the chunker drops the
// heading line from section content. Harness arm (headings_field_arm) was a
// recency-off WASH; the toggle exists for the live recency-on A/B.
describe('headings field', () => {
    function sectionChunk(id: string, title: string, headingPath: string[]): Chunk {
        const c = makeChunk(id, `${title} > ${headingPath.join(' > ')}`, 'discussion of the weekly sync points');
        c.heading_path = headingPath;
        return c;
    }
    const lr = sectionChunk('lr', 'Peter 1x1', ['Lightroom Aggregations']);
    const plain = makeChunk('plain', 'Roadmap', 'planning notes for the quarter');

    it('extractHeadingsText joins the path and survives a missing heading_path', () => {
        expect(extractHeadingsText(lr)).toBe('Lightroom Aggregations');
        const legacy = makeChunk('old', 'Old Note', 'body');
        delete (legacy as Partial<Chunk>).heading_path;
        expect(extractHeadingsText(legacy)).toBe('');
    });

    it('heading words match only when the field is enabled (BM25-invisible otherwise)', () => {
        const off = fitBM([lr, plain]);
        expect(off.getScores('lightroom')[0]).toBe(0);
        const on = fitBM([lr, plain], { headingsField: true });
        expect(on.getScores('lightroom')[0]).toBeGreaterThan(0);
        expect(on.getScores('lightroom')[1]).toBe(0); // plain note untouched
    });

    it('getQueryBound sees the headings field when enabled (df>0 path)', () => {
        const off = fitBM([lr, plain]);
        expect(off.getQueryBound('lightroom')).toBe(0);
        const on = fitBM([lr, plain], { headingsField: true });
        expect(on.getQueryBound('lightroom')).toBeGreaterThan(0);
    });

    it('analyzer pipeline applies to heading terms (depluralize: aggregation ~ aggregations)', () => {
        const on = fitBM([lr, plain], { headingsField: true });
        expect(on.getScores('aggregation')[0]).toBeGreaterThan(0);
    });

    it('composes with searchableProperties (both extra fields indexed)', () => {
        const c = sectionChunk('both', 'Trip', ['Hotels']);
        c.metadata.properties = { placeLoc: '[[Zurich]]' };
        const on = fitBM([c, plain], { searchableProperties: true, headingsField: true });
        expect(on.getScores('hotel')[0]).toBeGreaterThan(0);
        expect(on.getScores('zurich')[0]).toBeGreaterThan(0);
    });
});

// v8 frame-lite: fit() indexes the `content` field from the bodies MAP, not from
// the chunk (ChunkMeta has no content). These pin that contract directly — the
// chunk's own content is deliberately a decoy the index must ignore.
describe('fit reads body text from the bodies map (v8 frame-lite)', () => {
    it('indexes the supplied body, ignoring chunk.content', () => {
        const chunks = [makeChunk('a', 'Title A', 'DECOY'), makeChunk('b', 'Title B', 'DECOY')];
        const bodies = new Map([['a', 'pumpernickel rye'], ['b', 'sourdough']]);
        const idx = new MultiFieldBM25().fit(chunks, bodies);
        const scores = idx.getScoresWithCoverage('pumpernickel', { fuzzy: FUZZY_BY_LENGTH, prefix: PREFIX_LAST_TOKEN }).scores;
        expect(scores[0]).toBeGreaterThan(0);   // chunk a — body matched
        expect(scores[1]).toBe(0);              // chunk b — different body
        // the decoy chunk.content is never indexed
        expect(idx.getScores('decoy')[0]).toBe(0);
    });

    it('a chunk absent from the bodies map indexes an empty body', () => {
        const chunks = [makeChunk('a', 'Title A', 'DECOY')];
        const idx = new MultiFieldBM25().fit(chunks, new Map());
        expect(idx.getScoresWithCoverage('pumpernickel', { fuzzy: FUZZY_BY_LENGTH, prefix: PREFIX_LAST_TOKEN }).scores[0]).toBe(0);
    });
});

// ── Item 4: Latin diacritic folding (audit 2026-06-18 §4) ─────────────────
// foldDiacritics runs inside processTerm (symmetric index+query), so accented
// spellings co-match unaccented ones. Pure-ASCII-inert, idempotent, and CJK-
// guarded — NFD would mangle Hangul/dakuten kana (the one true landmine).
describe('foldDiacritics — Latin accent folding (audit §4)', () => {
    it('strips combining marks from Latin (decomposable) accents', () => {
        expect(foldDiacritics('café')).toBe('cafe');
        expect(foldDiacritics('zürich')).toBe('zurich');
        expect(foldDiacritics('naïve')).toBe('naive');
        expect(foldDiacritics('pokémon')).toBe('pokemon');
        expect(foldDiacritics('andrés')).toBe('andres');
        expect(foldDiacritics('José')).toBe('Jose');   // case preserved; lowercase happens upstream
    });
    it('is idempotent and symmetric (accented and unaccented fold equal)', () => {
        expect(foldDiacritics(foldDiacritics('café'))).toBe(foldDiacritics('café'));
        expect(foldDiacritics('café')).toBe(foldDiacritics('cafe'));
    });
    it('is a no-op on pure ASCII (every English eval term byte-identical)', () => {
        for (const w of ['search', 'gpt4', 'r2', 'hello123']) expect(foldDiacritics(w)).toBe(w);
    });
    it('GUARD: never folds CJK — NFD corruption avoided', () => {
        expect(foldDiacritics('한국어')).toBe('한국어');     // Hangul NOT decomposed to jamo
        expect(foldDiacritics('検索')).toBe('検索');         // Han untouched
        expect(foldDiacritics('ガ')).toBe('ガ');             // dakuten kana NOT stripped...
        expect(foldDiacritics('ガ')).not.toBe('カ');         // ...so GA never becomes KA
    });
    it('documented miss: stroke/bar letters have no NFD decomposition', () => {
        // ł, ø, đ are precomposed with no canonical decomposition, so NFD+\p{M}
        // leaves them — same as today; locked so a future char-map is deliberate.
        expect(foldDiacritics('łukasz')).toBe('łukasz');
    });
});

describe('processTerm folding — end-to-end accent co-match (audit §4)', () => {
    it('an unaccented query matches an accented indexed title and vice-versa', () => {
        const fwd = fitBM([makeChunk('a', 'Café Gitane', 'a paris bistro'),
                           makeChunk('b', 'Other Note', 'unrelated')]);
        expect(fwd.getScores('cafe')[0]).toBeGreaterThan(0);   // plain query -> accented title
        const rev = fitBM([makeChunk('a', 'Cafe Bar', 'coffee'),
                           makeChunk('b', 'Other Note', 'unrelated')]);
        expect(rev.getScores('café')[0]).toBeGreaterThan(0);   // accented query -> plain title
    });
    it('the all-stopword fallback folds too (keepStopwordsProcessTerm)', () => {
        // "às" folds to the stopword "as" -> whole query is stopwords -> fallback
        // path. The title field is stopword-exempt so "as" IS indexed; the
        // fallback MUST fold or "às" would query a term the index never built.
        const idx = fitBM([makeChunk('a', 'As', 'the band'),
                           makeChunk('b', 'Other Note', 'unrelated')]);
        expect(idx.getScores('às')[0]).toBeGreaterThan(0);
    });
});

describe('BM25_COVERAGE_POW — coordination soft-AND contract (de-franken → pow2 regression lock)', () => {
    // The soft-AND is applied ONCE, at the search.ts candidate-align site, as
    // `raw · coverage^BM25_COVERAGE_POW`. This block mirrors that exact formula to
    // lock the ranking decision: the de-franken dropped the exponent to 1 (linear
    // m/T), which let a doc matching only a rare high-IDF place token (near-max raw
    // after WAND-bound normalization, half coverage) out-rank a full-coverage
    // answer ("bars in austin" → the library beat the bars). P=2 restores the m²/T
    // penalty so full coverage wins. If a future change reverts the exponent toward
    // linear, these break — by design.
    const softAnd = (raw: number, coverage: number) => raw * Math.pow(coverage, BM25_COVERAGE_POW);

    it('the knob is 2 (changing it is a deliberate ranking decision, not a refactor)', () => {
        expect(BM25_COVERAGE_POW).toBe(2);
    });

    it('full coverage beats a higher-raw partial match — the "[thing] in [place]" fix', () => {
        const partial = softAnd(0.95, 0.5);   // rare-place-token-only: near-max raw, half coverage
        const full = softAnd(0.40, 1.0);       // real answer: lower raw, full coverage
        expect(full).toBeGreaterThan(partial); // pow2 orders them correctly
        // Linear (P=1) would have INVERTED this: 0.95·0.5 = 0.475 > 0.40·1 = 0.40.
        // That inequality is the exact regression pow2 fixes — assert it holds so
        // the test documents (and depends on) the bug it guards against.
        expect(0.95 * 0.5).toBeGreaterThan(0.40);
    });

    it('penalizes partial coverage super-linearly (m²/T, not m/T)', () => {
        expect(Math.pow(0.5, BM25_COVERAGE_POW)).toBeCloseTo(0.25, 10); // m²/T at half coverage
        expect(Math.pow(0.5, BM25_COVERAGE_POW)).toBeLessThan(0.5);     // strictly below linear m/T
    });

    it('is a no-op at full coverage — single-term / fully-covered queries stay faithful', () => {
        expect(softAnd(0.73, 1)).toBe(0.73); // 1^2 = 1, preserves the single-term-faithful invariant
    });

    it('annihilates zero coverage', () => {
        expect(softAnd(0.9, 0)).toBe(0);
    });
});

// v10 lexical reclamation: buildDoc folds chunk.link_terms into the content
// field, restoring the pre-v8 symmetry where a URL/link-target query matched
// the clipping whose CLEANED body no longer carries those bytes.
describe('link_terms fold into the content field (v10)', () => {
    it('a URL query matches ONLY via link_terms — the cleaned body alone cannot', () => {
        const clipped = makeChunk('a', 'Verge Clipping', 'Read the original at The Verge for details');
        clipped.link_terms = 'https://www.theverge.com/tech/some-post';
        const control = makeChunk('b', 'Other Clipping', 'Read the original at The Verge for details');
        const idx = fitBM([clipped, control]);

        const scores = idx.getScores('theverge.com');
        expect(scores[0]).toBeGreaterThan(0); // reclaimed: URL tokens live in content
        expect(scores[1]).toBe(0);            // identical cleaned body, no link_terms → still unfindable
    });

    it('an aliased wikilink TARGET matches via link_terms', () => {
        const meeting = makeChunk('a', 'Weekly Sync', 'met with Alex to discuss the roadmap');
        meeting.link_terms = 'Alex Goel';
        const idx = fitBM([meeting, makeChunk('b', 'Other', 'unrelated body text')]);
        expect(idx.getScores('goel')[0]).toBeGreaterThan(0);
    });

    it('a body-less chunk still indexes its link_terms', () => {
        const stub = makeChunk('a', 'Image Stub', '');
        stub.link_terms = 'Rapha Jersey.png';
        const idx = fitBM([stub]);
        expect(idx.getScores('rapha jersey')[0]).toBeGreaterThan(0);
    });
});
