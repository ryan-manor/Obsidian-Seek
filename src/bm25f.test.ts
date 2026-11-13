// BM25F scorer (bm25f.ts + the MultiFieldBM25 BM25F path). The harness parity lives
// in tests/bm25f-parity.test.ts (vault-derived fixture, kept out of src/); these are
// the synthetic invariants: the formula by hand, the bound/coverage contract, and the
// index-lifecycle paths (fromJSON, discard tombstones, incremental add/remove).
import { describe, it, expect } from 'vitest';
import { MultiFieldBM25, FUZZY_BY_LENGTH, PREFIX_LAST_TOKEN } from './bm25';
import { BM25F_PARAMS, BM25F_KNOB_DEFAULTS, bm25fIdf, bm25fParamsFrom, bm25fSaturate, type Bm25fParams } from './bm25f';
import type { Chunk } from './types';

function chunk(id: string, title: string, content: string, opts: { aliases?: string[]; props?: Record<string, string> } = {}): Chunk {
    return {
        chunk_id: id, title, content, note_path: `${id}.md`, heading_path: [],
        metadata: { tags: [], aliases: opts.aliases ?? [], created: null, modified: null, properties: opts.props ?? {} },
        start_line: 0, end_line: 0,
    };
}
const bodies = (cs: Chunk[]) => new Map(cs.map(c => [c.chunk_id, c.content]));
const fit = (cs: Chunk[]) => new MultiFieldBM25().fit(cs, bodies(cs), { searchableProperties: true, headingsField: true });
const BF = { bm25f: BM25F_PARAMS };

const CORPUS = [
    chunk('a', 'Lustre Pearl', 'A bar on the east side with a patio and live music.', { props: { placeType: 'bars' } }),
    chunk('b', 'Kinokuniya Austin', 'Bookstore with stationery and manga in Austin.'),
    chunk('c', 'Austin', 'The city. Many bars and music venues downtown.', { aliases: ['atx'] }),
    chunk('d', 'Coffee notes', 'Pour over ratios and grinder settings for a light roast.'),
    chunk('e', 'Trip plan', 'Fly into Austin, then drive east; bars on Rainey street.'),
];

describe('BM25F formula', () => {
    it('matches a hand computation: weighted pseudo-tf, one saturation, doc-level idf', () => {
        const bm = fit(CORPUS);
        const st = bm.bm25fStats()!;
        // "pearl": title of `a` only. df = 1, N = 5.
        const lenT = st.lenOf(0, 'title');               // "lustre pearl" → 2
        expect(lenT).toBe(2);
        const tft = 10 * 1 / (1 - 0.75 + 0.75 * lenT / st.avgLen.title);
        const want = bm25fIdf(5, 1) * (0.5 + 6 * tft / (5 + tft));
        expect(bm.bm25fTermScores('pearl')[0]).toBeCloseTo(want, 12);
        expect(bm25fSaturate(bm25fIdf(5, 1), tft, 5, 0.5)).toBeCloseTo(want, 12);
    });

    it('a term in several fields saturates ONCE (ceiling idf·(k1+1+δ) regardless of fields hit)', () => {
        const bm = fit(CORPUS);
        // "bar": a (content + props "bars"), c (content "bars"), e (content "bars").
        const s = bm.bm25fTermScores('bar');
        const idf = bm25fIdf(5, 3);
        for (const v of s) expect(v).toBeLessThan(idf * (5 + 1 + 0.5));
        expect(s[0]).toBeGreaterThan(s[2]);                // props weight 2 lifts `a`
    });

    it('df is document-level: a term in title AND body of one doc counts once', () => {
        const cs = [chunk('x', 'Zebra', 'zebra zebra'), chunk('y', 'Other', 'nothing here')];
        const bm = fit(cs);
        const s = bm.bm25fTermScores('zebra');
        const idf = bm25fIdf(2, 1);
        expect(s[0] / idf - 0.5).toBeGreaterThan(0);       // scored against df = 1, not 2
        expect(s[0]).toBeLessThan(idf * 6.5);
    });

    it('per-field b and custom weights are honored; boosts are ignored under BM25F', () => {
        const bm = fit(CORPUS);
        const p: Bm25fParams = { ...BM25F_PARAMS, b: { title: 0 }, bDefault: 0.75 };
        const st = bm.bm25fStats()!;
        const tft = 10 * 1 / 1;                            // b_title = 0: no length norm
        expect(bm.bm25fTermScores('pearl', p)[0]).toBeCloseTo(bm25fSaturate(bm25fIdf(5, 1), tft, 5, 0.5), 12);
        expect(st.n).toBe(5);
        const a = bm.getScoresWithCoverage('pearl', BF).scores;
        const b = bm.getScoresWithCoverage('pearl', { ...BF, boosts: { title: 1000, content: 0.001 } }).scores;
        expect(Array.from(b)).toEqual(Array.from(a));
    });
});

describe('BM25F bound + coverage contract', () => {
    it('a single-term query normalizes its best doc to exactly 1.0 (bound = UB = live max)', () => {
        const bm = fit(CORPUS);
        const { scores, bound } = bm.getScoresWithCoverage('austin', BF);
        expect(bound).toBe(bm.bm25fUpperBound('austin'));
        expect(Math.max(...scores)).toBeCloseTo(bound, 12);
    });

    it('Σ UB(t) per occurrence; OOV terms add nothing; coverage counts distinct sources', () => {
        const bm = fit(CORPUS);
        const r = bm.getScoresWithCoverage('austin bars zzqx', BF);
        expect(r.bound).toBeCloseTo(bm.bm25fUpperBound('austin') + bm.bm25fUpperBound('bar'), 12);
        expect(r.coverage[2]).toBe(1);        // c: austin + bars (zzqx is not indexed → T = 2)
        expect(r.coverage[1]).toBe(0.5);      // b: austin only
        const dup = bm.getScoresWithCoverage('austin austin', BF);
        expect(dup.bound).toBeCloseTo(2 * bm.bm25fUpperBound('austin'), 12);
        expect(dup.scores[2]).toBeCloseTo(2 * r.scores[2] - 2 * bm.bm25fTermScores('bar')[2], 9);
    });

    it('fuzzy/prefix derived terms score weight × their own BM25F score and fill the source slot', () => {
        const bm = fit(CORPUS);
        const opts = { ...BF, fuzzy: FUZZY_BY_LENGTH, prefix: PREFIX_LAST_TOKEN };
        const r = bm.getScoresWithCoverage('kinokunia', opts);       // OOV typo, edit budget 2 → kinokuniya (dist 1)
        const w = 0.45 * 10 / (10 + 1);
        expect(r.scores[1]).toBeCloseTo(w * bm.bm25fTermScores('kinokuniya')[1], 12);
        // Fuzzy-aware bound: the OOV term is bounded by its best derived term.
        expect(r.bound).toBeCloseTo(w * bm.bm25fUpperBound('kinokuniya'), 12);
        expect(bm.getScoresWithCoverage('kinokunia', BF).bound).toBe(0);   // fuzzy off: unreachable
        const p = bm.getScoresWithCoverage('east pea', opts);         // prefix pea → pearl (last token)
        const wp = 0.375 * 5 / (5 + 0.3 * 2);
        expect(p.scores[0]).toBeCloseTo(bm.bm25fTermScores('east')[0] + wp * bm.bm25fTermScores('pearl')[0], 12);
        expect(p.coverage[0]).toBe(1);
    });

    it('mixed typo query: the fuzzy-reached OOV term joins the bound and the coverage denominator', () => {
        const bm = fit(CORPUS);
        const opts = { ...BF, fuzzy: FUZZY_BY_LENGTH };
        const r = bm.getScoresWithCoverage('austin kinokunia', opts);
        const w = 0.45 * 10 / (10 + 1);
        expect(r.bound).toBeCloseTo(bm.bm25fUpperBound('austin') + w * bm.bm25fUpperBound('kinokuniya'), 12);
        expect(r.coverage[1]).toBe(1);        // b: austin + kinokuniya (fills the typo's slot)
        expect(r.coverage[2]).toBe(0.5);      // c: austin only (T = 2, not 1)
        // In-vocab terms are untouched: a clean query's bound is the exact Σ UB.
        const clean = bm.getScoresWithCoverage('austin kinokuniya', opts);
        expect(clean.bound).toBeCloseTo(bm.bm25fUpperBound('austin') + bm.bm25fUpperBound('kinokuniya'), 12);
        // An OOV term fuzzy can't reach still adds nothing.
        expect(bm.getScoresWithCoverage('austin zzqx', opts).bound).toBeCloseTo(bm.bm25fUpperBound('austin'), 12);
    });

    it('a half-typed last token with prefix extensions keeps the old bound (no fuzzy UB)', () => {
        const bm = fit(CORPUS);
        // "pear" is OOV, prefix-extends to "pearl" and is also fuzzy-reachable (dist 1).
        const typeahead = bm.getScoresWithCoverage('austin pear', { ...BF, fuzzy: FUZZY_BY_LENGTH, prefix: PREFIX_LAST_TOKEN });
        expect(typeahead.bound).toBeCloseTo(bm.bm25fUpperBound('austin'), 12);
        // Same token, prefix off: it's a plain typo, so the fuzzy bound applies.
        const typo = bm.getScoresWithCoverage('austin pear', { ...BF, fuzzy: FUZZY_BY_LENGTH });
        expect(typo.bound).toBeGreaterThan(bm.bm25fUpperBound('austin'));
        // A lone half-typed token keeps bound 0, i.e. fusion's /max fallback.
        expect(bm.getScoresWithCoverage('pear', { ...BF, fuzzy: FUZZY_BY_LENGTH, prefix: PREFIX_LAST_TOKEN }).bound).toBe(0);
    });

    it('a synonym mate scores at the discount and credits its SOURCE slot', () => {
        const bm = fit(CORPUS);
        const map = new Map([['atx', ['austin']], ['austin', ['atx']]]);
        const r = bm.getScoresWithCoverage('atx bars', { ...BF, synonyms: { map, weight: 0.8 } });
        // e has "austin" (mate of atx) + "bars": both slots filled.
        expect(r.scores[4]).toBeCloseTo(0.8 * bm.bm25fTermScores('austin')[4] + bm.bm25fTermScores('bar')[4], 12);
        expect(r.coverage[4]).toBe(1);
    });

    it('more than 32 distinct query terms spill past the bitmask without losing coverage', () => {
        const words = Array.from({ length: 40 }, (_, i) => `w${String.fromCharCode(97 + (i % 26))}${String.fromCharCode(97 + Math.floor(i / 26))}x`);
        const cs = [chunk('m', 'Many', words.slice(0, 35).join(' ')), chunk('n', 'None', 'unrelated text')];
        const bm = fit(cs);
        const r = bm.getScoresWithCoverage(words.join(' '), BF);
        expect(r.coverage[0]).toBeCloseTo(35 / 35, 12);              // 5 OOV words are not indexed: T = 35
        const r2 = bm.getScoresWithCoverage([...words.slice(0, 35), 'unrelated'].join(' '), BF);
        expect(r2.coverage[0]).toBeCloseTo(35 / 36, 12);
    });

    it('the legacy path is untouched when bm25f is absent', () => {
        const bm = fit(CORPUS);
        const legacy = bm.getScoresWithCoverage('austin bars');
        const again = new MultiFieldBM25().fit(CORPUS, bodies(CORPUS), { searchableProperties: true, headingsField: true })
            .getScoresWithCoverage('austin bars');
        expect(Array.from(legacy.scores)).toEqual(Array.from(again.scores));
        expect(legacy.bound).toBe(again.bound);
        expect(legacy.bound).not.toBeCloseTo(bm.getScoresWithCoverage('austin bars', BF).bound, 3);
    });
});

describe('BM25F across the index lifecycle', () => {
    const queries = ['austin bars', 'east pea', 'kinokunia', 'coffee'];
    const opts = { ...BF, fuzzy: FUZZY_BY_LENGTH, prefix: PREFIX_LAST_TOKEN };

    it('a fromJSON-loaded index (lengths rebuilt from postings, no bodies) scores identically to fit', () => {
        const a = fit(CORPUS);
        const b = new MultiFieldBM25().fromJSON(a.toJSON(), CORPUS, { searchableProperties: true, headingsField: true });
        for (const q of queries) {
            const x = a.getScoresWithCoverage(q, opts), y = b.getScoresWithCoverage(q, opts);
            expect(Array.from(y.scores)).toEqual(Array.from(x.scores));
            expect(Array.from(y.coverage)).toEqual(Array.from(x.coverage));
            expect(y.bound).toBe(x.bound);
        }
    });

    it('a discard() tombstone leaves N, df, lengths and the bound exactly as a fit without the doc', () => {
        for (const warmFirst of [false, true]) {
            const live = fit(CORPUS);
            if (warmFirst) live.warmBm25f();                  // incremental path vs lazy pass over tombstoned postings
            live.remove('c');
            const fresh = fit(CORPUS.filter(c => c.chunk_id !== 'c'));
            for (const q of queries) {
                const x = live.getScoresWithCoverage(q, opts), y = fresh.getScoresWithCoverage(q, opts);
                expect(x.bound).toBeCloseTo(y.bound, 12);
                for (const c of CORPUS) {
                    if (c.chunk_id === 'c') continue;
                    expect(x.scores[live.rowOf(c.chunk_id)!]).toBeCloseTo(y.scores[fresh.rowOf(c.chunk_id)!], 12);
                }
                expect(x.scores[2]).toBe(0);                  // the tombstoned row never scores
            }
        }
    });

    it('add() after the state exists keeps lengths exact (vs a fresh fit of the same rows)', () => {
        const base = CORPUS.slice(0, 3);
        const live = fit(base);
        live.warmBm25f();
        for (const c of CORPUS.slice(3)) live.add(c, c.content);
        const fresh = fit(CORPUS);
        const sl = live.bm25fStats()!, sf = fresh.bm25fStats()!;
        expect(sl.n).toBe(sf.n);
        for (const f of ['title', 'aliases', 'tags', 'content', 'properties', 'headings']) {
            expect(sl.avgLen[f]).toBeCloseTo(sf.avgLen[f], 12);
            for (let row = 0; row < CORPUS.length; row++) expect(sl.lenOf(row, f)).toBe(sf.lenOf(row, f));
        }
        for (const q of queries) {
            expect(Array.from(live.getScoresWithCoverage(q, opts).scores))
                .toEqual(Array.from(fresh.getScoresWithCoverage(q, opts).scores));
        }
    });

    it('a tolerant-load ghost (posting without a frame row) counts in N/df but is never scored', () => {
        const full = fit(CORPUS);
        const frame = CORPUS.filter(c => c.chunk_id !== 'e');   // 'e' is in the blob, not the frame
        const loaded = new MultiFieldBM25().fromJSON(full.toJSON(), frame, { searchableProperties: true, headingsField: true });
        const r = loaded.getScoresWithCoverage('austin bars', opts);
        const ref = full.getScoresWithCoverage('austin bars', opts);
        expect(r.scores.length).toBe(4);
        for (let i = 0; i < 4; i++) expect(r.scores[i]).toBeCloseTo(ref.scores[i], 12);   // same N/df as the full blob
        expect(r.bound).toBeCloseTo(ref.bound, 12);
    });
});

describe('BM25F knobs (seek:search overrides)', () => {
    it('defaults: knob defaults and BM25F_PARAMS agree (properties 4)', () => {
        expect(BM25F_KNOB_DEFAULTS).toEqual({ title: 10, aliases: 6, properties: 4, body: 1, k1: 5 });
        expect(bm25fParamsFrom({ ...BM25F_KNOB_DEFAULTS })).toBe(BM25F_PARAMS);
        expect(bm25fParamsFrom({})).toBe(BM25F_PARAMS);
    });

    it('maps body onto content/tags/headings, clamps to range, ignores garbage', () => {
        const p = bm25fParamsFrom({ title: 99, aliases: -3, properties: Number.NaN, body: 2, k1: 0 });
        expect(p.weights).toEqual({ title: 20, aliases: 0, properties: 4, content: 2, tags: 2, headings: 2 });
        expect(p.k1).toBe(0.5);
        expect(p.b).toBe(BM25F_PARAMS.b);
        expect(p.d).toBe(BM25F_PARAMS.d);
    });

    it('a weight change re-derives UB(t) and scores on the SAME index (no stale memo, no refit)', () => {
        const bm = fit(CORPUS);
        const p2 = bm25fParamsFrom({ properties: 2 });
        const ubDefault = bm.bm25fUpperBound('bar');                 // memoized under props 4
        const r4 = bm.getScoresWithCoverage('bars', BF);
        const r2 = bm.getScoresWithCoverage('bars', { bm25f: p2 });
        const fresh2 = fit(CORPUS).getScoresWithCoverage('bars', { bm25f: p2 });
        expect(r2.bound).toBe(fresh2.bound);                          // not the props-4 memo
        expect(Array.from(r2.scores)).toEqual(Array.from(fresh2.scores));
        expect(r2.bound).not.toBe(r4.bound);
        expect(bm.bm25fUpperBound('bar', p2)).toBe(fresh2.bound);
        // An equal-valued params object shares the memo; the default is untouched.
        expect(bm.bm25fUpperBound('bar', bm25fParamsFrom({ properties: 2 }))).toBe(fresh2.bound);
        expect(bm.bm25fUpperBound('bar')).toBe(ubDefault);
        expect(bm.getScoresWithCoverage('bars', BF).bound).toBe(r4.bound);
    });
});
