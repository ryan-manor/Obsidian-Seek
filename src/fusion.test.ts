import { describe, it, expect } from 'vitest';
import { hybridFusion, titleMatchBoost, titlePrefixCredit, TITLE_PREFIX_CREDIT, titleFuzzyCredit, TITLE_FUZZY_CREDIT, editDistanceWithin,theoreticalNormDense, theoreticalNormBm25, browseOrder, recencyDate, computeRecencyScore, parseDateMs } from './fusion';

describe('titleMatchBoost — token coverage (precision-scaled)', () => {
    const chunk = (note_path: string, aliases?: string[]) => ({ note_path, metadata: { aliases } });
    const B = 0.8;

    it('exact title => full boost (precision 1)', () => {
        const out = titleMatchBoost('memgraph', [chunk('Clippings/MemGraph.md')], B);
        expect(out[0]).toBeCloseTo(B, 10);
    });

    it('query is a subset of a dated title => boost scaled by precision', () => {
        // q={alex,1x1}; title tokens {alex,1x1,2026,05,19} => 2/5 = 0.4
        const out = titleMatchBoost('alex 1x1', [chunk('Notes/Alex 1x1 2026-05-19.md')], B);
        expect(out[0]).toBeCloseTo(B * (2 / 5), 10);
    });

    it('all query tokens must be present — partial coverage gets nothing', () => {
        // q={alex,workstreams}; title has alex but not workstreams => 0
        const out = titleMatchBoost('alex workstreams', [chunk('Notes/Alex 1x1 2026-05-19.md')], B);
        expect(out[0]).toBe(0);
    });

    it('word order does not matter (set membership)', () => {
        const out = titleMatchBoost('project eames', [chunk('Notes/Eames Project.md')], B);
        expect(out[0]).toBeCloseTo(B, 10); // {project,eames} == {eames,project} => precision 1
    });

    it('a longer title with the query as a subset earns less than an exact title', () => {
        const exactNote = titleMatchBoost('eames project', [chunk('Notes/Eames Project.md')], B)[0];
        const taskNote = titleMatchBoost('eames project', [chunk('Notes/Eames Project Design Review.md')], B)[0];
        expect(exactNote).toBeGreaterThan(taskNote); // precision 1 vs 2/4
        expect(taskNote).toBeCloseTo(B * (2 / 4), 10);
    });

    it('matches an alias (acronym) when the title does not contain the query', () => {
        // "aca" is not in "Creative Assistant" but is an alias
        const out = titleMatchBoost('aca', [chunk('Notes/Creative Assistant.md', ['ACA'])], B);
        expect(out[0]).toBeCloseTo(B, 10);
    });

    it('takes the best of basename and aliases', () => {
        const out = titleMatchBoost('aca', [chunk('Notes/Creative Assistant.md', ['Example Creative Assistant', 'ACA'])], B);
        expect(out[0]).toBeCloseTo(B, 10); // exact alias "ACA" wins over the 3-token alias
    });

    it('empty query => no boost', () => {
        expect(titleMatchBoost('   ', [chunk('Notes/Anything.md')], B)[0]).toBe(0);
    });

    it('depluralizes symmetrically: a plural query fires on a singular title token', () => {
        // "parks" -> "park" matches "Moores Park" {moores, park}: precision 1/2.
        // BM25 already depluralized; this closes the nav-boost asymmetry.
        const out = titleMatchBoost('parks', [chunk('Notes/Personal/Places/Moores Park.md')], B);
        expect(out[0]).toBeCloseTo(B * (1 / 2), 10);
    });

    it('depluralizes the title side too: singular query, plural title', () => {
        // "category" matches "Categories" {category} after both depluralize.
        const out = titleMatchBoost('category', [chunk('Notes/Categories.md')], B);
        expect(out[0]).toBeCloseTo(B, 10); // precision 1
    });

    // audit 2026-06-09 §6.1 — a stopword the user typed must not kill the gate.
    it('ignores a leading query stopword the title omits (§6.1)', () => {
        // q content={eames,project} (drops "the"); title {eames,project} => precision 1.
        // Before the fix this returned 0 because "the" wasn't in the title.
        const out = titleMatchBoost('the eames project', [chunk('Notes/Eames Project.md')], B);
        expect(out[0]).toBeCloseTo(B, 10);
    });

    it('drops mid-query stopwords too, denominator is the title (§6.1)', () => {
        // q content={alex,1x1} (drops "on","the"); title {alex,1x1,2026,05,19} => 2/5.
        const out = titleMatchBoost('alex on the 1x1', [chunk('Notes/Alex 1x1 2026-05-19.md')], B);
        expect(out[0]).toBeCloseTo(B * (2 / 5), 10);
    });

    it('all-stopword query falls back to literal tokens (name-as-stopword, §6.2 not regressed)', () => {
        // "will" is a stoplist word; with no content token to anchor on we keep it
        // literal so a person page titled "Will" still earns the nav boost.
        const out = titleMatchBoost('will', [chunk('People/Will.md')], B);
        expect(out[0]).toBeCloseTo(B, 10);
    });

    // Three-Lens S3 (2026-06-10): the tokenizer was ASCII-only /[a-z0-9]+/g,
    // so every non-ASCII title produced zero tokens and the boost was
    // structurally dead for "Café Gitane", "Zürich", accented people names.
    it('non-ASCII titles earn the boost (S3 Unicode tokenizer)', () => {
        const out = titleMatchBoost('café gitane', [chunk('Places/Café Gitane.md')], B);
        expect(out[0]).toBeCloseTo(B, 10); // exact match, precision 1
    });

    it('accented query tokens match accented title tokens exactly', () => {
        // q={zürich,trip} ⊆ title {zürich,trip,2025} => 2/3 precision.
        const out = titleMatchBoost('zürich trip', [chunk('Travel/Zürich Trip 2025.md')], B);
        expect(out[0]).toBeCloseTo(B * (2 / 3), 10);
    });

    it('folds diacritics — unaccented query matches an accented title (audit §4)', () => {
        // tokenSet folds via the SAME foldDiacritics processTerm uses, so the
        // title-boost keys the term space BM25 indexes: an unaccented "cafe"
        // query now earns the full boost on "Café Gitane". (Replaces the prior
        // lock test that asserted no folding — this IS that deliberate change.)
        const out = titleMatchBoost('cafe gitane', [chunk('Places/Café Gitane.md')], B);
        expect(out[0]).toBeCloseTo(B, 10); // exact match after fold, precision 1
    });

    it("strips a possessive clitic — \"ryan's\" fires on a `Ryan` title (audit 2026-06-29 #4)", () => {
        // tokenSet now tokenizes with seekTokenize(derived:false), so the
        // possessive-strip BM25 applies runs here too. Before the fix the bare
        // /[\p{L}\p{N}]+/ split left a stray "s" ("ryan","s") that failed the
        // all-query-tokens-in-title gate while BM25 stripped it and matched.
        const out = titleMatchBoost("ryan's", [chunk('People/Ryan.md')], B);
        expect(out[0]).toBeCloseTo(B, 10); // precision 1 after the clitic drops
    });

    it('a glue-joined query reaches a hyphenated title via the derived gate (audit R2 2026-07-02 #2)', () => {
        // The GATE now accepts the title's derived forms (glue-joined
        // compounds), so "gpt4" covers a `GPT-4` title ({gpt4} ⊆ {gpt,4,gpt4}).
        // The DENOMINATOR stays canonical ({gpt,4}, size 2) so this doesn't
        // reopen the dated-title dilution the derived:false denominator choice
        // guards against (audit 2026-06-29 #4) — precision is 1/2, not 1.
        const out = titleMatchBoost('gpt4', [chunk('Notes/GPT-4.md')], B);
        expect(out[0]).toBeCloseTo(B * (1 / 2), 10);
    });

    it('clamps precision at 1 when derived matches outnumber canonical title tokens', () => {
        // "Co-op" canonical {co,op} (denominator=2) but derived-matchable
        // {co,op,coop} (3). Querying with both the glued and split forms
        // ("coop co op") lets `inter` (3) exceed |t| (2) — must not exceed
        // the full boost.
        const out = titleMatchBoost('coop co op', [chunk('Notes/Co-op.md')], B);
        expect(out[0]).toBeCloseTo(B, 10);
    });

    describe('title fuzzy (typo tokens, caller-gated, partial credit)', () => {
        // Caller contract: only OOV tokens get a budget; here "plannign" is the typo.
        const budget = (tok: string) => (tok === 'plannign' ? 2 : 0);
        const sdc = chunk('Example/FY27 SDC Planning.md');

        it('a typo token fuzz-matches within its budget for 0.45·L/(L+d) credit', () => {
            expect(TITLE_FUZZY_CREDIT).toBe(0.45);
            const out = titleMatchBoost('fy27 plannign', [sdc], B, true, budget);
            expect(out[0]).toBeCloseTo(B * ((1 + titleFuzzyCredit(8, 2)) / 3), 10);
        });

        it('no budget (fuzzy off or in-vocab token) keeps the exact gate', () => {
            expect(titleMatchBoost('fy27 plannign', [sdc], B)[0]).toBe(0);
            expect(titleMatchBoost('fy27 plannign', [sdc], B, true, () => 0)[0]).toBe(0);
        });

        it('the clean spelling still outranks the typo', () => {
            const clean = titleMatchBoost('fy27 planning', [sdc], B)[0];
            const typo = titleMatchBoost('fy27 plannign', [sdc], B, true, budget)[0];
            expect(clean).toBeGreaterThan(typo);
            expect(typo).toBeGreaterThan(0);
        });

        it('beyond the budget does not match', () => {
            expect(titleMatchBoost('fy27 plnanign', [sdc], B, true, () => 1)[0]).toBe(0);
        });

        it('editDistanceWithin is plain Levenshtein with a cap', () => {
            expect(editDistanceWithin('plannign', 'planning', 2)).toBe(2);   // transposition = 2
            expect(editDistanceWithin('kinokunia', 'kinokuniya', 2)).toBe(1);
            expect(editDistanceWithin('abc', 'xyz', 1)).toBe(2);             // capped at max + 1
            expect(editDistanceWithin('same', 'same', 1)).toBe(0);
        });
    });

    describe('title prefix (typeahead final token, 1-char floor, partial credit)', () => {
        const title = chunk('Notes/Work/Evolving Infinity Search.md');
        const P = titlePrefixCredit;

        it('credit follows MiniSearch prefix weighting: 0.375 · L/(L + 0.3·d)', () => {
            expect(TITLE_PREFIX_CREDIT).toBe(0.375);
            expect(P(3, 3)).toBeCloseTo(0.375, 10);              // no unfinished tail
            expect(P(3, 7)).toBeCloseTo(0.375 * 7 / 8.2, 10);    // "car" → "carboni"
            expect(P(1, 5)).toBeLessThan(P(4, 5));               // more typed → more credit
        });

        it('a half-typed final token prefix-matches a title token for partial credit', () => {
            expect(titleMatchBoost('evolving infini', [title], B)[0]).toBeCloseTo(B * ((1 + P(6, 8)) / 3), 10);
        });

        it('an exact longer title beats a one-word prefix title ("car")', () => {
            const out = titleMatchBoost('car', [chunk("N/Cars and bikes.md"), chunk("N/Carboni's.md")], B);
            expect(out[0]).toBeCloseTo(B / 3, 10);
            expect(out[0]).toBeGreaterThan(out[1]);
            expect(out[1]).toBeGreaterThan(0);
        });

        it('uses the best (shortest) completion among title tokens', () => {
            const out = titleMatchBoost('x car', [chunk('N/X Carbonara Cart.md')], B)[0];
            expect(out).toBeCloseTo(B * ((1 + P(3, 4)) / 3), 10);
        });

        it('an exact final token outranks a prefix one', () => {
            const exact = titleMatchBoost('evolving infinity', [title], B)[0];
            const partial = titleMatchBoost('evolving infini', [title], B)[0];
            expect(exact).toBeCloseTo(B * (2 / 3), 10);
            expect(exact).toBeGreaterThan(partial);
            // "… ai": an `AI` title beats an `Aim` title of the same length.
            const out = titleMatchBoost('alex ai', [chunk('N/Alex AI.md'), chunk('N/Alex Aim.md')], B);
            expect(out[0]).toBeGreaterThan(out[1]);
            expect(out[1]).toBeGreaterThan(0);
        });

        it('a single char after a finished token prefix-matches', () => {
            const ganci = chunk('Meetings/Andrei Ganci 1x1 2026-10-13.md');
            expect(titleMatchBoost('andrei g', [ganci], B)[0]).toBeGreaterThan(0);
            expect(titleMatchBoost('andrei g', [chunk('People/Andrei Stefan.md')], B)[0]).toBe(0);
        });

        it('prefix = false is exact-only', () => {
            expect(titleMatchBoost('evolving infini', [title], B, false)[0]).toBe(0);
        });

        it('only the final token may prefix-match', () => {
            expect(titleMatchBoost('evolv infinity', [title], B)[0]).toBe(0);
        });

        it('a lone token under 3 chars never prefix-matches', () => {
            expect(titleMatchBoost('e', [title], B)[0]).toBe(0);
            expect(titleMatchBoost('ev', [title], B)[0]).toBe(0);
            expect(titleMatchBoost('evo', [title], B)[0]).toBeCloseTo(B * (P(3, 8) / 3), 10);
        });

        it('applies to aliases too', () => {
            const out = titleMatchBoost('agentic retr', [chunk('Notes/X.md', ['Agentic Retrieval Mode'])], B);
            expect(out[0]).toBeCloseTo(B * ((1 + P(4, 9)) / 3), 10);
        });
    });
});

describe('theoreticalNormDense — fixed-endpoint cosine normalization', () => {
    it('maps cosine via (cos+1)/2, clamped to [0,1]', () => {
        const out = theoreticalNormDense([1, 0, -1, 0.74, 0.86]);
        expect(out[0]).toBeCloseTo(1, 10);
        expect(out[1]).toBeCloseTo(0.5, 10);
        expect(out[2]).toBeCloseTo(0, 10);
        expect(out[3]).toBeCloseTo(0.87, 10);
        expect(out[4]).toBeCloseTo(0.93, 10);
    });
    it('a no-opinion (bunched) channel stays nearly flat — cannot manufacture a winner', () => {
        // OOV query: all cosines in the noise band ~0.74-0.80. Fixed endpoints keep
        // them bunched (~0.87-0.90), unlike min-max which stretches them to [0,1].
        const out = theoreticalNormDense([0.794, 0.752, 0.781, 0.745]);
        const spread = Math.max(...out) - Math.min(...out);
        expect(spread).toBeLessThan(0.03); // min-max would force spread = 1.0
    });
});

describe('theoreticalNormBm25 — theoretical-bound normalization', () => {
    it('divides by the theoretical bound when provided — best match is NOT forced to 1.0', () => {
        // The point of the bound: a weak best match keeps a small normalized
        // score instead of being stretched to full lexical confidence.
        const out = theoreticalNormBm25([0, 2.5, 5], 10);
        expect(out[0]).toBeCloseTo(0, 10);
        expect(out[1]).toBeCloseTo(0.25, 10);
        expect(out[2]).toBeCloseTo(0.5, 10);
    });
    it('clips scores that exceed the bound (fuzzy derived-term matches)', () => {
        const out = theoreticalNormBm25([12, 5], 10);
        expect(out[0]).toBeCloseTo(1, 10);   // exceeds → clip; ties arbitrated by dense
        expect(out[1]).toBeCloseTo(0.5, 10);
    });
    it('F1 fallback: bound 0 with nonzero scores → empirical /max (the OOV-typo rescue)', () => {
        // A fully-OOV typo query under fuzzy has bound 0 but real derived-term
        // scores. Zeroing here would kill the only channel that knows the answer
        // (34/40 vs 37/40 gold@1 on the D&D typo slice) — degrade to rank-only
        // /max instead.
        const out = theoreticalNormBm25([0, 2.5, 5], 0);
        expect(out[1]).toBeCloseTo(0.5, 10);
        expect(out[2]).toBeCloseTo(1, 10);
    });
    it('divides by the per-query max when no bound is passed (legacy callers)', () => {
        const out = theoreticalNormBm25([0, 2.5, 5]);
        expect(out[1]).toBeCloseTo(0.5, 10);
        expect(out[2]).toBeCloseTo(1, 10);
    });
    it('an all-zero channel maps to all-0 regardless of bound (keysmash case)', () => {
        expect(Array.from(theoreticalNormBm25([0, 0, 0]))).toEqual([0, 0, 0]);
        expect(Array.from(theoreticalNormBm25([0, 0, 0], 10))).toEqual([0, 0, 0]);
    });
});

describe('hybridFusion — TM2C2 blend', () => {
    it('alpha·dense + (1-alpha)·bm25 over normalized channels', () => {
        const out = hybridFusion(new Float64Array([1, 0]), new Float64Array([0, 1]), 0.9);
        expect(out[0]).toBeCloseTo(0.9, 10);
        expect(out[1]).toBeCloseTo(0.1, 10);
    });
});

describe('browseOrder — filter-only fast-path ordering (audit 2026-06-09 §1)', () => {
    const chunk = (note_path: string, created: string | null, opts: { modified?: string | null; chunk_id?: string } = {}) => ({
        note_path,
        chunk_id: opts.chunk_id ?? `${note_path}#0`,
        metadata: { created, modified: opts.modified ?? null },
    });
    const paths = (chunks: Array<{ note_path?: string }>) => chunks.map(c => c.note_path);

    it('orders by created descending, regardless of input (frame) order', () => {
        const input = [
            chunk('Notes/Old.md', '2024-01-15'),
            chunk('Notes/New.md', '2026-06-01'),
            chunk('Notes/Mid.md', '2025-03-10'),
        ];
        expect(paths(browseOrder(input))).toEqual(['Notes/New.md', 'Notes/Mid.md', 'Notes/Old.md']);
    });

    it('is deterministic: a reversed input yields the identical output order', () => {
        // The bug this replaces: rank() at recencyWeight=0 degenerated to a
        // stable sort of equal scores, i.e. the output WAS the input order.
        const input = [
            chunk('Notes/A.md', '2026-05-01'),
            chunk('Notes/B.md', '2026-04-01'),
            chunk('Notes/C.md', null),
            chunk('Notes/D.md', '2026-04-01'),
        ];
        const forward = paths(browseOrder([...input]));
        const backward = paths(browseOrder([...input].reverse()));
        expect(backward).toEqual(forward);
    });

    it('does NOT fall back to modified when created is missing (undated sorts last)', () => {
        const input = [
            chunk('Notes/ModifiedNew.md', null, { modified: '2026-01-01' }),
            chunk('Notes/CreatedOld.md', '2025-01-01'),
        ];
        expect(paths(browseOrder(input))).toEqual(['Notes/CreatedOld.md', 'Notes/ModifiedNew.md']);
    });

    it('keeps undated chunks, after all dated ones, sorted by path', () => {
        // Undated chunks matched the filter — the matched set IS the result
        // set on this path, so dropping them (as the recency ARM may) would
        // be a recall bug.
        const input = [
            chunk('Notes/Zed Undated.md', null),
            chunk('Notes/Dated.md', '2020-01-01'),
            chunk('Notes/Alpha Undated.md', null),
        ];
        expect(paths(browseOrder(input))).toEqual([
            'Notes/Dated.md',
            'Notes/Alpha Undated.md',
            'Notes/Zed Undated.md',
        ]);
    });

    it('breaks same-day ties by note_path, not input order', () => {
        // `created` is date-granular, so daily notes tie constantly; an
        // input-order tie-break would reintroduce frame order within a day.
        const a = chunk('Notes/A Meeting.md', '2026-06-09');
        const b = chunk('Notes/B Meeting.md', '2026-06-09');
        expect(paths(browseOrder([b, a]))).toEqual(['Notes/A Meeting.md', 'Notes/B Meeting.md']);
        expect(paths(browseOrder([a, b]))).toEqual(['Notes/A Meeting.md', 'Notes/B Meeting.md']);
    });

    it('breaks same-note ties by chunk_id', () => {
        const c2 = chunk('Notes/N.md', '2026-06-09', { chunk_id: 'Notes/N.md#2' });
        const c1 = chunk('Notes/N.md', '2026-06-09', { chunk_id: 'Notes/N.md#1' });
        expect(browseOrder([c2, c1]).map(c => c.chunk_id)).toEqual(['Notes/N.md#1', 'Notes/N.md#2']);
    });

    it('takes no ranking config — ordering cannot be parked by a weight change', () => {
        // Structural contract: the signature is (chunks, recencyKey?) — the key
        // is a presentation choice, not RankingConfig. fn.length counts only
        // the required params, so this also trips if someone adds a required
        // config argument. This test documents the invariant.
        expect(browseOrder.length).toBe(1);
    });

    it("key 'modified' sorts by mtime only", () => {
        const input = [
            chunk('Notes/CreatedNew.md', '2026-06-01', { modified: '2025-01-01' }), // newest created, oldest mtime
            chunk('Notes/ModifiedNew.md', '2024-01-01', { modified: '2026-01-01' }),
            chunk('Notes/OnlyCreated.md', '2025-06-01'), // no mtime → undated, last
        ];
        expect(paths(browseOrder(input, 'modified'))).toEqual([
            'Notes/ModifiedNew.md', // 2026-01-01 (mtime)
            'Notes/CreatedNew.md',  // 2025-01-01 (mtime — created date deliberately ignored)
            'Notes/OnlyCreated.md', // no mtime → undated
        ]);
    });
});

describe('recencyDate — the shared recency accessor (arm ⇄ scorer ⇄ browse)', () => {
    const c = (metadata: { created?: string | null; modified?: string | null; properties?: Record<string, string> }, note_path?: string) =>
        ({ note_path, metadata });

    it("'created' reads created only — no mtime fallback", () => {
        expect(recencyDate(c({ created: '2025-01-01', modified: '2026-01-01' }), 'created')).toBe('2025-01-01');
        expect(recencyDate(c({ created: null, modified: '2026-01-01' }), 'created')).toBeNull();
    });
    it("'modified' reads modified only — no created fallback", () => {
        expect(recencyDate(c({ created: '2025-01-01', modified: '2026-01-01' }), 'modified')).toBe('2026-01-01');
        expect(recencyDate(c({ created: '2025-01-01', modified: null }), 'modified')).toBeNull();
    });
    it('undefined chunk/metadata → null (neutral under the additive ε term)', () => {
        expect(recencyDate(undefined, 'created')).toBeNull();
        expect(recencyDate(c({}), 'modified')).toBeNull();
    });
    it('a custom createdProp reads the indexed properties map (read-side, no reindex)', () => {
        const chunk = c({ created: null, modified: '2026-01-01', properties: { 'date created': '2025-03-10' } });
        expect(recencyDate(chunk, 'created', 'date created')).toBe('2025-03-10');
    });
    it('custom prop missing → null: never substitutes `created`, a filename date, or mtime', () => {
        expect(recencyDate(c({ created: '2025-03-10', modified: '2026-01-01', properties: {} }), 'created', 'dateCreated')).toBeNull();
        expect(recencyDate(c({ properties: {} }, 'Daily/2026-06-02.md'), 'created', 'dateCreated')).toBeNull();
    });
    it('no frontmatter → null, even with a YYYY-MM-DD filename', () => {
        expect(recencyDate(c({ modified: '2026-11-07' }, 'Daily/2026-06-11.md'), 'created')).toBeNull();
    });
});

describe('computeRecencyScore — smooth half-life decay (replaced the two-stage 30d cutoff 2026-06-11)', () => {
    const NOW = Date.parse('2026-06-11T00:00:00Z');
    const daysAgo = (d: number) => new Date(NOW - d * 86_400_000).toISOString();

    it('1.0 today, 0.5 at one half-life, 0.25 at two', () => {
        expect(computeRecencyScore(daysAgo(0), { halfLifeDays: 180, referenceDateMs: NOW })).toBeCloseTo(1.0, 10);
        expect(computeRecencyScore(daysAgo(180), { halfLifeDays: 180, referenceDateMs: NOW })).toBeCloseTo(0.5, 10);
        expect(computeRecencyScore(daysAgo(360), { halfLifeDays: 180, referenceDateMs: NOW })).toBeCloseTo(0.25, 10);
    });

    it('the median episodic click target (83d old) keeps real signal — the old 30d cutoff zeroed it', () => {
        const s = computeRecencyScore(daysAgo(83), { halfLifeDays: 180, referenceDateMs: NOW });
        expect(s).toBeGreaterThan(0.7);
        expect(s).toBeLessThan(0.8);
    });

    it('never reaches zero for any dated note; undated/unparseable → 0 (neutral)', () => {
        expect(computeRecencyScore(daysAgo(3650), { halfLifeDays: 180, referenceDateMs: NOW })).toBeGreaterThan(0);
        expect(computeRecencyScore(null)).toBe(0);
        expect(computeRecencyScore('not a date')).toBe(0);
    });

    it('future dates clamp to age 0 (score 1), not >1', () => {
        expect(computeRecencyScore(daysAgo(-30), { halfLifeDays: 180, referenceDateMs: NOW })).toBeCloseTo(1.0, 10);
    });
});

describe('parseDateMs — bare-date rejects out-of-range month/day instead of normalizing (2026-07-02 review)', () => {
    it('rejects a typo\'d month (13) instead of silently rolling into next January', () => {
        expect(parseDateMs('2026-13-01')).toBeNull();
    });

    it('rejects month 00', () => {
        expect(parseDateMs('2026-00-01')).toBeNull();
    });

    it('rejects a day out of the 1-31 range', () => {
        expect(parseDateMs('2026-01-32')).toBeNull();
        expect(parseDateMs('2026-01-00')).toBeNull();
    });

    it('still accepts a real date at local midnight', () => {
        const ms = parseDateMs('2026-06-15');
        expect(ms).not.toBeNull();
        const dt = new Date(ms!);
        expect(dt.getFullYear()).toBe(2026);
        expect(dt.getMonth()).toBe(5); // 0-indexed June
        expect(dt.getDate()).toBe(15);
    });

    it('same rejection applies via the embedded-prefix fallback path', () => {
        expect(parseDateMs('2026-13-01 some note title')).toBeNull();
    });
});
