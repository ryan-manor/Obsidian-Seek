// Tests for the inline filter parser (parseQuery) and the chunk matcher
// (compileMatcher / matchesFilters).
//
// The parseQuery suite is a faithful port of the predecessor backend's
// `tests/test_query_preprocessing.py` (40 cases). The caller-merge class is
// intentionally NOT ported: Seek has no caller-provided filters — parseQuery
// only ever extracts from the raw query string. The matchesFilters suite is
// Seek-specific (the predecessor tests its filter application separately).

import { describe, it, expect } from 'vitest';
import { parseQuery, compileMatcher, matchesFilters, excludedNotePaths } from './query-parser';
import type { Chunk, ChunkMetadata, QueryFilters, FilterContext } from './types';

// A FilterContext stub: `created` is the date field (Recency ON) and the named
// keys are Number-typed. Lets the typed-value tests exercise the parser/matcher
// gates the real search path resolves from app + settings.
function ctx(numericKeys: string[] = [], dateOn = true): FilterContext {
    return {
        dateField: dateOn ? { key: 'created', createdProp: 'created' } : null,
        numericKeys: new Set(numericKeys),
    };
}

// ---- helpers ----

function makeChunk(over: Partial<Omit<Chunk, 'metadata'>> & { metadata?: Partial<ChunkMetadata> } = {}): Chunk {
    const { metadata: metaOver, ...rest } = over;
    const metadata: ChunkMetadata = {
        tags: [],
        aliases: [],
        created: null,
        modified: null,
        properties: {},
        ...(metaOver ?? {}),
    };
    return {
        chunk_id: 'c',
        title: 'T',
        content: 'body',
        note_path: 'Note.md',
        heading_path: [],
        start_line: 1,
        end_line: 2,
        ...rest,
        metadata,
    };
}

function filters(over: Partial<QueryFilters>): QueryFilters {
    return {
        tags: null,
        tagsMatchAll: false,
        frontmatter: null,
        includePaths: null,
        numeric: null,
        dateAfter: null,
        dateBefore: null,
        numericTypeMismatch: null,
        exclude: null,
        ...over,
    };
}

// =====================================================================
// parseQuery — ported from test_query_preprocessing.py
// =====================================================================

describe('parseQuery / baseline', () => {
    it('plain query passes through, no filters', () => {
        const { cleanedQuery, filters } = parseQuery('kubernetes deployment notes');
        expect(cleanedQuery).toBe('kubernetes deployment notes');
        expect(filters).toBeNull();
    });

    it('trims edges but leaves internal whitespace when nothing extracted', () => {
        const { cleanedQuery, filters } = parseQuery('   kubernetes   notes   ');
        expect(cleanedQuery).toBe('kubernetes   notes');
        expect(filters).toBeNull();
    });
});

describe('parseQuery / #tag', () => {
    it('simple hashtag → tag filter, not re-injected', () => {
        const { cleanedQuery, filters } = parseQuery('alex #meetings');
        expect(filters!.tags).toEqual(['meetings']);
        expect(cleanedQuery).toBe('alex');
    });

    it('hierarchical hashtag preserved, not flattened or re-injected', () => {
        const { cleanedQuery, filters } = parseQuery('alex #meetings/1x1');
        expect(filters!.tags).toEqual(['meetings/1x1']);
        expect(cleanedQuery).toBe('alex');
    });

    it('multiple hashtags', () => {
        const { filters } = parseQuery('project #meetings/1x1 #projects/active');
        expect(filters!.tags).toEqual(['meetings/1x1', 'projects/active']);
    });

    it('kebab-case hashtag captured whole, not split at the hyphen', () => {
        const { cleanedQuery, filters } = parseQuery('notes #meeting-prep');
        expect(filters!.tags).toEqual(['meeting-prep']);
        expect(cleanedQuery).toBe('notes'); // not "notes -prep"
    });

    it('kebab-case in a hierarchical hashtag segment', () => {
        const { filters } = parseQuery('#projects/code-review');
        expect(filters!.tags).toEqual(['projects/code-review']);
    });

    it('accented / non-Latin tags bind (Obsidian allows any letter)', () => {
        expect(parseQuery('notes #café').filters!.tags).toEqual(['café']);
        expect(parseQuery('notes #日本語').filters!.tags).toEqual(['日本語']);
        expect(parseQuery('#проект/задача').filters!.tags).toEqual(['проект/задача']);
    });

    it('emoji tag binds as one token (u flag, no surrogate split)', () => {
        expect(parseQuery('done #🎉').filters!.tags).toEqual(['🎉']);
    });

    it('a tag terminates at trailing punctuation (#café. → café)', () => {
        const { cleanedQuery, filters } = parseQuery('the #café. opened');
        expect(filters!.tags).toEqual(['café']);
        expect(cleanedQuery).toBe('the . opened');
    });
});

describe('parseQuery / tag: prefix', () => {
    it('tag:#x', () => {
        const { cleanedQuery, filters } = parseQuery('alex tag:#meetings/1x1');
        expect(filters!.tags).toEqual(['meetings/1x1']);
        expect(cleanedQuery).toBe('alex');
    });

    it('tag:x (no hash)', () => {
        const { filters } = parseQuery('alex tag:meetings/1x1');
        expect(filters!.tags).toEqual(['meetings/1x1']);
    });

    it('tag: and #tag mixed', () => {
        const { filters } = parseQuery('alex tag:#meetings #projects');
        expect(filters!.tags).toEqual(['meetings', 'projects']);
    });

    it('tag: with a kebab-case value', () => {
        const { filters } = parseQuery('alex tag:code-review');
        expect(filters!.tags).toEqual(['code-review']);
    });
});

describe('parseQuery / leading token boundary (audit R2 #9)', () => {
    it('a pasted URL fragment is not parsed as a #tag filter', () => {
        const { cleanedQuery, filters } = parseQuery('see https://example.com/page#fragment');
        expect(filters).toBeNull();
        expect(cleanedQuery).toBe('see https://example.com/page#fragment');
    });

    it('"montag:meeting" does not bind tag:meeting mid-word', () => {
        const { cleanedQuery, filters } = parseQuery('montag:meeting notes');
        expect(filters).toBeNull();
        expect(cleanedQuery).toBe('montag:meeting notes');
    });

    it('a real #tag still binds right after the mid-word non-match', () => {
        const { cleanedQuery, filters } = parseQuery('montag:meeting #meetings');
        expect(filters!.tags).toEqual(['meetings']);
        expect(cleanedQuery).toBe('montag:meeting');
    });
});

describe('parseQuery / adjacent filters with no separating whitespace (audit R2 review-2 #1)', () => {
    it('two bracket filters concatenated with zero space both bind', () => {
        const { cleanedQuery, filters } = parseQuery('[context:work][pageType:task]');
        expect(filters!.frontmatter).toEqual({ context: 'work', pageType: 'task' });
        expect(cleanedQuery).toBe('');
    });

    it('a comparison bracket directly followed by a substring bracket both bind', () => {
        const { filters } = parseQuery('[price>50][pageType:task]', ctx(['price']));
        expect(filters!.numeric).toEqual([{ key: 'price', op: '>', value: 50 }]);
        expect(filters!.frontmatter).toEqual({ pageType: 'task' });
    });

    it('a hashtag directly followed by a bracket filter both bind', () => {
        const { filters } = parseQuery('#tag[key:value]');
        expect(filters!.tags).toEqual(['tag']);
        expect(filters!.frontmatter).toEqual({ key: 'value' });
    });

    it('three bracket filters chained with zero space all bind', () => {
        const { filters } = parseQuery('[a:b][c:d][e:f]');
        expect(filters!.frontmatter).toEqual({ a: 'b', c: 'd', e: 'f' });
    });
});

describe('parseQuery / [key:value] frontmatter', () => {
    it('simple property, bracket consumed', () => {
        const { cleanedQuery, filters } = parseQuery('alex [context:work]');
        expect(filters!.frontmatter).toEqual({ context: 'work' });
        expect(cleanedQuery).toBe('alex');
    });

    it('preserves key case', () => {
        const { filters } = parseQuery('alex [Context:work]');
        expect(filters!.frontmatter).toEqual({ Context: 'work' });
    });

    it('preserves camelCase key', () => {
        const { filters } = parseQuery('alex [pageType:Daily]');
        expect(filters!.frontmatter).toEqual({ pageType: 'Daily' });
    });

    it('kebab-case frontmatter key', () => {
        const { filters } = parseQuery('alex [my-field:work]');
        expect(filters!.frontmatter).toEqual({ 'my-field': 'work' });
    });

    it('value with internal space', () => {
        const { filters } = parseQuery('notes [context:my work]');
        expect(filters!.frontmatter).toEqual({ context: 'my work' });
    });

    it('multiple properties', () => {
        const { filters } = parseQuery('[context:work] [pageType:meeting]');
        expect(filters!.frontmatter).toEqual({ context: 'work', pageType: 'meeting' });
    });

    it('quoted value keeps quotes for the matcher (exact-mode signal)', () => {
        const { filters } = parseQuery('hotels [placeLoc:"Los Angeles"]');
        expect(filters!.frontmatter).toEqual({ placeLoc: '"Los Angeles"' });
    });
});

describe('parseQuery / date filters (before:/after:)', () => {
    it('after: extracts a lower bound', () => {
        const { cleanedQuery, filters } = parseQuery('alex after:2026-04-01', ctx());
        expect(filters!.dateAfter).toBe('2026-04-01');
        expect(filters!.dateBefore).toBeNull();
        expect(filters!.frontmatter).toBeNull();
        expect(cleanedQuery).toBe('alex');
    });

    it('before: extracts an upper bound', () => {
        const { filters } = parseQuery('alex before:2026-05-01', ctx());
        expect(filters!.dateBefore).toBe('2026-05-01');
        expect(filters!.dateAfter).toBeNull();
    });

    it('after: + before: combine into a range', () => {
        const { filters } = parseQuery('after:2026-01-01 before:2026-06-28', ctx());
        expect(filters!.dateAfter).toBe('2026-01-01');
        expect(filters!.dateBefore).toBe('2026-06-28');
    });

    it('no ctx (ad-hoc) still parses before:/after: (permissive default)', () => {
        const { filters } = parseQuery('after:2026-04-01');
        expect(filters!.dateAfter).toBe('2026-04-01');
    });

    it('Recency OFF (ctx.dateField null) leaves before:/after: as plain text', () => {
        const { cleanedQuery, filters } = parseQuery('alex after:2026-04-01', ctx([], false));
        expect(filters).toBeNull();
        expect(cleanedQuery).toBe('alex after:2026-04-01');
    });

    it('the old [created:>X] bracket date syntax is gone — now a type mismatch', () => {
        // `created` is not a Number property, so the comparison is unsatisfiable
        // (D2 removed the date pseudo-keys; D3 governs the fallout).
        const { filters } = parseQuery('[created:>2026-04-01]', ctx());
        expect(filters!.dateAfter).toBeNull();
        expect(filters!.numericTypeMismatch).toEqual(['created']);
    });

    it('created without operator is frontmatter equality (substring branch)', () => {
        const { filters } = parseQuery('[created:2026-04-01]', ctx());
        expect(filters!.frontmatter).toEqual({ created: '2026-04-01' });
        expect(filters!.dateAfter).toBeNull();
        expect(filters!.dateBefore).toBeNull();
    });
});

describe('parseQuery / numeric filters', () => {
    it('colon-less comparison on a Number key', () => {
        const { cleanedQuery, filters } = parseQuery('rapha [price>50]', ctx(['price']));
        expect(filters!.numeric).toEqual([{ key: 'price', op: '>', value: 50 }]);
        expect(filters!.frontmatter).toBeNull();
        expect(cleanedQuery).toBe('rapha');
    });

    it('whitespace and colon are both tolerated', () => {
        expect(parseQuery('[price > 50]', ctx(['price'])).filters!.numeric).toEqual([{ key: 'price', op: '>', value: 50 }]);
        expect(parseQuery('[price:>50]', ctx(['price'])).filters!.numeric).toEqual([{ key: 'price', op: '>', value: 50 }]);
        expect(parseQuery('[price : > 50]', ctx(['price'])).filters!.numeric).toEqual([{ key: 'price', op: '>', value: 50 }]);
    });

    it('all three operators, value coerced to number', () => {
        expect(parseQuery('[price<200]', ctx(['price'])).filters!.numeric).toEqual([{ key: 'price', op: '<', value: 200 }]);
        expect(parseQuery('[price=160]', ctx(['price'])).filters!.numeric).toEqual([{ key: 'price', op: '=', value: 160 }]);
        expect(parseQuery('[price=50.00]', ctx(['price'])).filters!.numeric).toEqual([{ key: 'price', op: '=', value: 50 }]);
    });

    it('comparison on a non-Number key is a type mismatch, NOT substring (D3)', () => {
        const { filters } = parseQuery('[pageType<notes]', ctx(['price']));
        expect(filters!.numericTypeMismatch).toEqual(['pageType']);
        expect(filters!.numeric).toBeNull();
        expect(filters!.frontmatter).toBeNull(); // never falls back to substring
    });

    it('a non-numeric literal on a Number key is also a mismatch', () => {
        const { filters } = parseQuery('[price>abc]', ctx(['price']));
        expect(filters!.numericTypeMismatch).toEqual(['price']);
        expect(filters!.numeric).toBeNull();
    });

    it('substring [key:value] is untouched by the comparison branch', () => {
        const { filters } = parseQuery('[context:work]', ctx(['price']));
        expect(filters!.frontmatter).toEqual({ context: 'work' });
        expect(filters!.numeric).toBeNull();
        expect(filters!.numericTypeMismatch).toBeNull();
    });

    it('an operator not adjacent to the key stays a substring value', () => {
        const { filters } = parseQuery('[note:a>b]', ctx(['price']));
        expect(filters!.frontmatter).toEqual({ note: 'a>b' });
        expect(filters!.numeric).toBeNull();
    });

    it('no ctx (ad-hoc) treats any comparison key as numeric (permissive)', () => {
        const { filters } = parseQuery('[price>50]');
        expect(filters!.numeric).toEqual([{ key: 'price', op: '>', value: 50 }]);
        expect(filters!.numericTypeMismatch).toBeNull();
    });
});

describe('parseQuery / path:', () => {
    it('simple path', () => {
        const { cleanedQuery, filters } = parseQuery('alex path:Notes/Work/Meetings');
        expect(filters!.includePaths).toEqual(['Notes/Work/Meetings']);
        expect(cleanedQuery).toBe('alex');
    });

    it('path terminates at whitespace', () => {
        const { cleanedQuery, filters } = parseQuery('path:Notes/Work meetings');
        expect(filters!.includePaths).toEqual(['Notes/Work']);
        expect(cleanedQuery).toBe('meetings');
    });

    it('multiple paths', () => {
        const { filters } = parseQuery('path:Notes/A path:Notes/B');
        expect(filters!.includePaths).toEqual(['Notes/A', 'Notes/B']);
    });

    it('quoted path allows spaces (Obsidian path:"Daily notes/2022-07")', () => {
        const { cleanedQuery, filters } = parseQuery('standup path:"Daily notes/2022-07"');
        expect(filters!.includePaths).toEqual(['Daily notes/2022-07']); // quotes stripped
        expect(cleanedQuery).toBe('standup');
    });

    it('quoted path keeps the trailing glob and binds the rest as text', () => {
        const { cleanedQuery, filters } = parseQuery('path:"My Folder/*" notes');
        expect(filters!.includePaths).toEqual(['My Folder/*']);
        expect(cleanedQuery).toBe('notes');
    });
});

describe('parseQuery / combinations', () => {
    it('tag + property + path, only free text remains', () => {
        const { cleanedQuery, filters } = parseQuery('alex [context:work] #meetings/1x1 path:Notes/Work');
        expect(filters!.tags).toEqual(['meetings/1x1']);
        expect(filters!.frontmatter).toEqual({ context: 'work' });
        expect(filters!.includePaths).toEqual(['Notes/Work']);
        expect(cleanedQuery).toBe('alex');
    });

    it('date + tag', () => {
        const { filters } = parseQuery('alex after:2026-04-01 #meetings/1x1', ctx());
        expect(filters!.dateAfter).toBe('2026-04-01');
        expect(filters!.tags).toEqual(['meetings/1x1']);
    });

    it('numeric + tag + path', () => {
        const { cleanedQuery, filters } = parseQuery('rapha [price<200] #brands path:Notes/Gear', ctx(['price']));
        expect(filters!.numeric).toEqual([{ key: 'price', op: '<', value: 200 }]);
        expect(filters!.tags).toEqual(['brands']);
        expect(filters!.includePaths).toEqual(['Notes/Gear']);
        expect(cleanedQuery).toBe('rapha');
    });
});

describe('parseQuery / negation (-term)', () => {
    it('single -term → exclude token, stripped from cleaned query', () => {
        const { cleanedQuery, filters } = parseQuery('meeting -work');
        expect(filters!.exclude).toEqual(['work']);
        expect(cleanedQuery).toBe('meeting');
    });

    it('multiple -term stack', () => {
        const { cleanedQuery, filters } = parseQuery('meeting -work -meetup');
        expect(filters!.exclude).toEqual(['work', 'meetup']);
        expect(cleanedQuery).toBe('meeting');
    });

    it('negation lowercases the token', () => {
        const { filters } = parseQuery('notes -Work');
        expect(filters!.exclude).toEqual(['work']);
    });

    it('mid-word hyphen is NOT negation (covid-19 stays text)', () => {
        const { cleanedQuery, filters } = parseQuery('covid-19 cases');
        expect(filters).toBeNull();
        expect(cleanedQuery).toBe('covid-19 cases');
    });

    it('stop-word negation is a no-op (-the → no exclude, left as text)', () => {
        const { cleanedQuery, filters } = parseQuery('report -the');
        expect(filters).toBeNull();
        expect(cleanedQuery).toBe('report -the');
    });

    it('negated operator forms are dropped (no-op), never inverted to positive', () => {
        // -#tag must NOT become a positive #tag filter; deferred → dropped.
        const a = parseQuery('notes -#meetings');
        expect(a.filters).toBeNull();
        expect(a.cleanedQuery).toBe('notes');
        const b = parseQuery('notes -path:Work');
        expect(b.filters).toBeNull();
        expect(b.cleanedQuery).toBe('notes');
    });

    it('hyphenated negation splits into tokens, plus the glued compound (shared seekTokenize)', () => {
        // seekTokenize is now the shared tokenizer (audit R2 #11): hyphen is
        // glue punctuation, so "foo-bar" additively emits the joined "foobar"
        // alongside the split "foo"/"bar" — exactly as it would on the BM25
        // query side (bm25.ts distinctQueryTerms), so a doc indexed under the
        // compound form is excludable too.
        const { filters } = parseQuery('x -foo-bar');
        expect(filters!.exclude).toEqual(['foo', 'bar', 'foobar']);
    });

    it('combines with other operators', () => {
        const { cleanedQuery, filters } = parseQuery('alex #meetings -draft');
        expect(filters!.tags).toEqual(['meetings']);
        expect(filters!.exclude).toEqual(['draft']);
        expect(cleanedQuery).toBe('alex');
    });

    it('negation-only query yields empty cleaned text + exclude filter', () => {
        const { cleanedQuery, filters } = parseQuery('-work');
        expect(cleanedQuery).toBe('');
        expect(filters!.exclude).toEqual(['work']);
    });

    it('negation shares the pipeline depluralizer (audit R2 #11: -cat now folds like a query term)', () => {
        const { filters } = parseQuery('x -cat');
        expect(filters!.exclude).toEqual(['cat']);
    });

    it('negation shares the pipeline diacritic fold (-café → cafe)', () => {
        const { filters } = parseQuery('x -café');
        expect(filters!.exclude).toEqual(['cafe']);
    });
});

describe('excludedNotePaths (note-level, token match)', () => {
    // Bodies are supplied via getBody(chunkId) since the v8 frame-lite split.
    // Map by chunk_id (unique per fixture, as in prod) → the chunk's content.
    const getBody = (cs: Chunk[]) => {
        const m = new Map(cs.map(c => [c.chunk_id, c.content]));
        return (id: string): string | undefined => m.get(id);
    };

    it('excludes a note whose content contains the token', () => {
        const chunks = [
            makeChunk({ chunk_id: 'a', note_path: 'A.md', content: 'about work stuff' }),
            makeChunk({ chunk_id: 'b', note_path: 'B.md', content: 'about play stuff' }),
        ];
        const out = excludedNotePaths(chunks, ['work'], getBody(chunks));
        expect(out.has('A.md')).toBe(true);
        expect(out.has('B.md')).toBe(false);
    });

    it('token match, not substring (work ≠ workout/network)', () => {
        const chunks = [makeChunk({ note_path: 'A.md', content: 'my workout and network' })];
        expect(excludedNotePaths(chunks, ['work'], getBody(chunks)).size).toBe(0);
    });

    it('excludes whole note when ANY chunk matches (note-level)', () => {
        const chunks = [
            makeChunk({ chunk_id: 'a1', note_path: 'A.md', content: 'intro text' }),
            makeChunk({ chunk_id: 'a2', note_path: 'A.md', content: 'deadline work item' }),
        ];
        expect(excludedNotePaths(chunks, ['work'], getBody(chunks)).has('A.md')).toBe(true);
    });

    it('matches the title too', () => {
        const chunks = [makeChunk({ note_path: 'Work Log.md', title: 'Work Log', content: 'body' })];
        expect(excludedNotePaths(chunks, ['work'], getBody(chunks)).has('Work Log.md')).toBe(true);
    });

    it('any of multiple tokens triggers exclusion (OR across exclude terms)', () => {
        const chunks = [
            makeChunk({ chunk_id: 'a', note_path: 'A.md', content: 'has draft only' }),
            makeChunk({ chunk_id: 'b', note_path: 'B.md', content: 'clean note' }),
        ];
        const out = excludedNotePaths(chunks, ['work', 'draft'], getBody(chunks));
        expect(out.has('A.md')).toBe(true);
        expect(out.has('B.md')).toBe(false);
    });

    it('empty exclude list excludes nothing', () => {
        const chunks = [makeChunk({ note_path: 'A.md', content: 'work' })];
        expect(excludedNotePaths(chunks, [], getBody(chunks)).size).toBe(0);
    });

    it('-cat (end to end) suppresses a note that only contains the plural "cats" (audit R2 #11)', () => {
        const { filters } = parseQuery('meeting -cat');
        const chunks = [makeChunk({ note_path: 'A.md', content: 'I love cats' })];
        const out = excludedNotePaths(chunks, filters!.exclude!, getBody(chunks));
        expect(out.has('A.md')).toBe(true);
    });
});

describe('parseQuery / malformed → plain text', () => {
    it('unclosed bracket', () => {
        const { cleanedQuery, filters } = parseQuery('alex [unclosed');
        expect(filters).toBeNull();
        expect(cleanedQuery).toBe('alex [unclosed');
    });

    it('empty bracket value', () => {
        const { cleanedQuery, filters } = parseQuery('alex [empty:]');
        expect(filters).toBeNull();
        expect(cleanedQuery).toBe('alex [empty:]');
    });

    it('path: with no value', () => {
        const { cleanedQuery, filters } = parseQuery('alex path: meetings');
        expect(filters).toBeNull();
        expect(cleanedQuery).toBe('alex path: meetings');
    });

    it('tag: with no value', () => {
        const { cleanedQuery, filters } = parseQuery('alex tag: meetings');
        expect(filters).toBeNull();
        expect(cleanedQuery).toBe('alex tag: meetings');
    });
});

describe('parseQuery / cleaned query', () => {
    it('all structural tokens stripped, no re-injection', () => {
        const { cleanedQuery } = parseQuery('alex [context:work] path:Notes #meetings/1x1');
        expect(cleanedQuery).toBe('alex');
    });

    it('filter-only query yields empty cleaned text', () => {
        const { cleanedQuery, filters } = parseQuery('[context:work] #meetings');
        expect(cleanedQuery).toBe('');
        expect(filters!.frontmatter).toEqual({ context: 'work' });
        expect(filters!.tags).toEqual(['meetings']);
    });

    it('tagsMatchAll defaults to false (OR semantics in v1)', () => {
        const { filters } = parseQuery('#a #b');
        expect(filters!.tagsMatchAll).toBe(false);
    });
});

// =====================================================================
// matchesFilters / compileMatcher — Seek-specific filter application
// =====================================================================

describe('matchesFilters / tags (hierarchical, OR)', () => {
    it('parent filter matches child tag', () => {
        const c = makeChunk({ metadata: { tags: ['meetings/1x1'] } });
        expect(matchesFilters(c, filters({ tags: ['meetings'] }))).toBe(true);
    });

    it('exact tag matches', () => {
        const c = makeChunk({ metadata: { tags: ['projects'] } });
        expect(matchesFilters(c, filters({ tags: ['projects'] }))).toBe(true);
    });

    it('partial-segment prefix does NOT false-match (meeting vs meetings)', () => {
        const c = makeChunk({ metadata: { tags: ['meetings'] } });
        expect(matchesFilters(c, filters({ tags: ['meeting'] }))).toBe(false);
    });

    it('plural sibling does NOT match (1x1 vs 1x1s)', () => {
        const c = makeChunk({ metadata: { tags: ['meetings/1x1s'] } });
        expect(matchesFilters(c, filters({ tags: ['meetings/1x1'] }))).toBe(false);
    });

    it('OR across multiple filter tags', () => {
        const c = makeChunk({ metadata: { tags: ['projects'] } });
        expect(matchesFilters(c, filters({ tags: ['meetings', 'projects'] }))).toBe(true);
    });

    it('# prefix on chunk tag is normalized away', () => {
        const c = makeChunk({ metadata: { tags: ['#meetings'] } });
        expect(matchesFilters(c, filters({ tags: ['meetings'] }))).toBe(true);
    });

    it('tagsMatchAll = AND semantics', () => {
        const one = makeChunk({ metadata: { tags: ['a'] } });
        const both = makeChunk({ metadata: { tags: ['a', 'b'] } });
        const f = filters({ tags: ['a', 'b'], tagsMatchAll: true });
        expect(matchesFilters(one, f)).toBe(false);
        expect(matchesFilters(both, f)).toBe(true);
    });
});

describe('matchesFilters / path globs', () => {
    const c = makeChunk({ note_path: 'Notes/Work/standup.md' });

    it('trailing /* matches', () => {
        expect(matchesFilters(c, filters({ includePaths: ['Notes/Work/*'] }))).toBe(true);
    });

    it('no-wildcard pattern matches nothing (documented fnmatch quirk)', () => {
        expect(matchesFilters(c, filters({ includePaths: ['Notes/Work'] }))).toBe(false);
    });

    it('top-level glob matches', () => {
        expect(matchesFilters(c, filters({ includePaths: ['Notes/*'] }))).toBe(true);
    });

    it('*/pattern fallback matches a nested path', () => {
        const nested = makeChunk({ note_path: 'Vault/Notes/Tasks/a.md' });
        expect(matchesFilters(nested, filters({ includePaths: ['Notes/Tasks/*'] }))).toBe(true);
    });

    it('non-matching path rejected', () => {
        expect(matchesFilters(c, filters({ includePaths: ['Personal/*'] }))).toBe(false);
    });
});

describe('matchesFilters / frontmatter (Obsidian-style: substring + wikilink + quoted-exact)', () => {
    const c = makeChunk({ metadata: { properties: { context: 'work', pageType: 'meeting' } } });

    it('exact value matches', () => {
        expect(matchesFilters(c, filters({ frontmatter: { context: 'work' } }))).toBe(true);
    });

    it('case-insensitive value', () => {
        expect(matchesFilters(c, filters({ frontmatter: { context: 'Work' } }))).toBe(true);
    });

    it('non-matching value rejected', () => {
        expect(matchesFilters(c, filters({ frontmatter: { context: 'personal' } }))).toBe(false);
    });

    it('missing key rejected', () => {
        expect(matchesFilters(c, filters({ frontmatter: { status: 'open' } }))).toBe(false);
    });

    it('all keys must match (AND across frontmatter)', () => {
        expect(matchesFilters(c, filters({ frontmatter: { context: 'work', pageType: 'daily' } }))).toBe(false);
    });

    // ---- Obsidian-style substring (default, unquoted) ----
    it('substring of value matches by default', () => {
        const h = makeChunk({ metadata: { properties: { placeType: 'accommodations' } } });
        expect(matchesFilters(h, filters({ frontmatter: { placeType: 'accom' } }))).toBe(true);
    });

    it('substring that is absent still rejects', () => {
        const h = makeChunk({ metadata: { properties: { placeType: 'accommodations' } } });
        expect(matchesFilters(h, filters({ frontmatter: { placeType: 'restaurant' } }))).toBe(false);
    });

    // ---- wikilink-aware (the placeLoc gotcha) ----
    it('plain wikilink value matches unbracketed query', () => {
        const h = makeChunk({ metadata: { properties: { placeLoc: '[[Los Angeles]]' } } });
        expect(matchesFilters(h, filters({ frontmatter: { placeLoc: 'Los Angeles' } }))).toBe(true);
    });

    it('aliased wikilink matches either target or alias', () => {
        const h = makeChunk({ metadata: { properties: { when: '[[2026-05-31|May 31]]' } } });
        expect(matchesFilters(h, filters({ frontmatter: { when: '2026-05-31' } }))).toBe(true);
        expect(matchesFilters(h, filters({ frontmatter: { when: 'May 31' } }))).toBe(true);
    });

    // ---- double-quoted = whole-value exact ----
    it('quoted value forces exact match (partial rejected)', () => {
        expect(matchesFilters(c, filters({ frontmatter: { context: '"wor"' } }))).toBe(false);
        expect(matchesFilters(c, filters({ frontmatter: { context: '"work"' } }))).toBe(true);
    });

    it('quoted exact still sees through wikilinks', () => {
        const h = makeChunk({ metadata: { properties: { placeLoc: '[[Los Angeles]]' } } });
        expect(matchesFilters(h, filters({ frontmatter: { placeLoc: '"Los Angeles"' } }))).toBe(true);
    });
});

describe('matchesFilters / dates (day-inclusive, missing→reject)', () => {
    const c = makeChunk({ metadata: { created: '2026-05-16', modified: '2026-05-19' } });

    it('after: passes when on/after the bound', () => {
        expect(matchesFilters(c, filters({ dateAfter: '2026-05-01' }), ctx())).toBe(true);
        expect(matchesFilters(c, filters({ dateAfter: '2026-05-16' }), ctx())).toBe(true); // inclusive of the day itself
    });

    it('after: rejects when before the bound', () => {
        expect(matchesFilters(c, filters({ dateAfter: '2026-06-01' }), ctx())).toBe(false);
    });

    it('before: passes when on/before the bound', () => {
        expect(matchesFilters(c, filters({ dateBefore: '2026-05-20' }), ctx())).toBe(true);
        expect(matchesFilters(c, filters({ dateBefore: '2026-05-16' }), ctx())).toBe(true); // inclusive of the day itself
    });

    it('before: rejects when after the bound', () => {
        expect(matchesFilters(c, filters({ dateBefore: '2026-05-10' }), ctx())).toBe(false);
    });

    it('before: includes a same-day afternoon timestamp (the inclusivity fix)', () => {
        const afternoon = makeChunk({ metadata: { created: '2026-05-16T15:30:00' } });
        expect(matchesFilters(afternoon, filters({ dateBefore: '2026-05-16' }), ctx())).toBe(true);
    });

    it('before: includes a same-day LATE-EVENING timestamp (local-time-consistent boundary)', () => {
        // Regression for a UTC-anchored `before:` bound vs a local-time property
        // parse: an evening event would fall on the *next* UTC calendar day and
        // get rejected west of UTC even though it's still the same local day.
        // Both sides must live in the same (local) frame.
        const evening = makeChunk({ metadata: { created: '2026-05-16T23:45:00' } });
        expect(matchesFilters(evening, filters({ dateBefore: '2026-05-16' }), ctx())).toBe(true);
    });

    it('before: a bare year covers the whole year', () => {
        const dec = makeChunk({ metadata: { created: '2026-12-31' } });
        expect(matchesFilters(dec, filters({ dateBefore: '2026' }), ctx())).toBe(true);
        const next = makeChunk({ metadata: { created: '2027-01-01' } });
        expect(matchesFilters(next, filters({ dateBefore: '2026' }), ctx())).toBe(false);
    });

    it('missing date rejects a date filter', () => {
        const noDate = makeChunk({ metadata: { created: null, modified: null } });
        expect(matchesFilters(noDate, filters({ dateAfter: '2026-01-01' }), ctx())).toBe(false);
    });

    it('the date field follows ctx.dateField (modified vs created)', () => {
        const onModified: FilterContext = { dateField: { key: 'modified', createdProp: 'created' }, numericKeys: new Set() };
        // created 05-16, modified 05-19 → after:05-18 passes on modified, fails on created.
        expect(matchesFilters(c, filters({ dateAfter: '2026-05-18' }), onModified)).toBe(true);
        expect(matchesFilters(c, filters({ dateAfter: '2026-05-18' }), ctx())).toBe(false);
    });

    it('NO filename-date fallback (D4): a dated filename with no property is rejected', () => {
        const dailyNote = makeChunk({ note_path: 'Daily/2026-05-16.md', metadata: { created: null, modified: null } });
        expect(matchesFilters(dailyNote, filters({ dateAfter: '2026-01-01' }), ctx())).toBe(false);
    });
});

describe('matchesFilters / numeric (value-inclusive, missing→reject)', () => {
    const c = makeChunk({ metadata: { properties: { price: '160', importance: '3' } } });

    it('> is inclusive of the bound', () => {
        expect(matchesFilters(c, filters({ numeric: [{ key: 'price', op: '>', value: 160 }] }))).toBe(true);
        expect(matchesFilters(c, filters({ numeric: [{ key: 'price', op: '>', value: 161 }] }))).toBe(false);
    });

    it('< is inclusive of the bound', () => {
        expect(matchesFilters(c, filters({ numeric: [{ key: 'price', op: '<', value: 160 }] }))).toBe(true);
        expect(matchesFilters(c, filters({ numeric: [{ key: 'price', op: '<', value: 159 }] }))).toBe(false);
    });

    it('= is exact (post-coercion)', () => {
        expect(matchesFilters(c, filters({ numeric: [{ key: 'price', op: '=', value: 160 }] }))).toBe(true);
        expect(matchesFilters(c, filters({ numeric: [{ key: 'price', op: '=', value: 16 }] }))).toBe(false);
    });

    it('coerces a quoted / messy stored value', () => {
        const q = makeChunk({ metadata: { properties: { price: '"299.00"' } } });
        expect(matchesFilters(q, filters({ numeric: [{ key: 'price', op: '=', value: 299 }] }))).toBe(true);
        const bad = makeChunk({ metadata: { properties: { price: '197, 344, 218' } } });
        expect(matchesFilters(bad, filters({ numeric: [{ key: 'price', op: '>', value: 1 }] }))).toBe(false); // NaN → reject
    });

    it('missing key rejects', () => {
        const none = makeChunk({ metadata: { properties: {} } });
        expect(matchesFilters(none, filters({ numeric: [{ key: 'price', op: '>', value: 1 }] }))).toBe(false);
    });

    it('numericTypeMismatch makes the whole query reject everything (D3)', () => {
        const f = filters({ numericTypeMismatch: ['pageType'], tags: null });
        expect(matchesFilters(c, f)).toBe(false);
        // even a chunk that would satisfy every OTHER clause is rejected:
        const f2 = filters({ frontmatter: { price: '160' }, numericTypeMismatch: ['pageType'] });
        expect(matchesFilters(c, f2)).toBe(false);
    });
});

describe('matchesFilters / combined predicates (AND across filter types)', () => {
    const c = makeChunk({
        note_path: 'Notes/Work/q2.md',
        metadata: { tags: ['projects'], properties: { context: 'work', price: '120' }, created: '2026-05-16' },
    });

    it('all conditions satisfied', () => {
        const f = filters({
            tags: ['projects'], frontmatter: { context: 'work' }, dateAfter: '2026-05-01',
            numeric: [{ key: 'price', op: '<', value: 200 }], includePaths: ['Notes/*'],
        });
        expect(matchesFilters(c, f, ctx(['price']))).toBe(true);
    });

    it('one failing condition fails the whole match', () => {
        const f = filters({ tags: ['projects'], numeric: [{ key: 'price', op: '>', value: 200 }] });
        expect(matchesFilters(c, f, ctx(['price']))).toBe(false);
    });
});

describe('compileMatcher is reusable across chunks', () => {
    it('returns a closure applied to many chunks', () => {
        const m = compileMatcher(filters({ tags: ['meetings'] }));
        expect(m(makeChunk({ metadata: { tags: ['meetings/1x1'] } }))).toBe(true);
        expect(m(makeChunk({ metadata: { tags: ['projects'] } }))).toBe(false);
    });
});

// ── v10 query-side silent-zero fixes (audit R2 #3/#4 + case-sensitive keys) ──

describe('parseQuery / unparseable date values fall through as text (audit R2 #4)', () => {
    it('after:yesterday binds NO filter and stays searchable text', () => {
        // Pre-fix: the token was stripped from the text AND compiled to a null
        // bound — deleted from both channels. Verbatim fallthrough mirrors the
        // Recency-OFF arm.
        const { cleanedQuery, filters } = parseQuery('standup after:yesterday', ctx());
        expect(filters).toBeNull();
        expect(cleanedQuery).toBe('standup after:yesterday');
    });

    it('before:not-a-date likewise', () => {
        const { cleanedQuery, filters } = parseQuery('report before:soon', ctx());
        expect(filters).toBeNull();
        expect(cleanedQuery).toBe('report before:soon');
    });

    it('a parseable date still binds (guard is validation, not a regression)', () => {
        const { filters } = parseQuery('after:2026-04-01', ctx());
        expect(filters!.dateAfter).toBe('2026-04-01');
    });

    // 2026-07-02 review: endBoundMs's day-branch used the numeric-args Date
    // constructor, which NORMALIZES an out-of-range month/day (e.g. month 13
    // rolls into next January) instead of rejecting it like the old
    // Date.parse-based code did — so a typo'd before:/after: bound would
    // silently bind to the wrong date instead of falling through as text.
    it('before:2026-13-01 (typo month) falls through as text, does not bind', () => {
        const { cleanedQuery, filters } = parseQuery('report before:2026-13-01', ctx());
        expect(filters).toBeNull();
        expect(cleanedQuery).toBe('report before:2026-13-01');
    });

    it('after:2026-01-32 (out-of-range day) falls through as text, does not bind', () => {
        const { cleanedQuery, filters } = parseQuery('report after:2026-01-32', ctx());
        expect(filters).toBeNull();
        expect(cleanedQuery).toBe('report after:2026-01-32');
    });
});

describe('matchesFilters / case-insensitive property keys', () => {
    it('[pagetype:task] matches a stored pageType key', () => {
        const chunk = makeChunk({ metadata: { properties: { pageType: 'task' } } });
        expect(matchesFilters(chunk, filters({ frontmatter: { pagetype: 'task' } }))).toBe(true);
        expect(matchesFilters(chunk, filters({ frontmatter: { PAGETYPE: 'task' } }))).toBe(true);
    });

    it('exact-case lookup still wins the fast path', () => {
        const chunk = makeChunk({ metadata: { properties: { pageType: 'task' } } });
        expect(matchesFilters(chunk, filters({ frontmatter: { pageType: 'task' } }))).toBe(true);
    });

    it('numeric clauses resolve keys case-insensitively too', () => {
        const chunk = makeChunk({ metadata: { properties: { Price: '160' } } });
        expect(matchesFilters(chunk, filters({ numeric: [{ key: 'price', op: '=', value: 160 }] }))).toBe(true);
    });
});

describe('matchesFilters / list-valued properties (audit R2 #3)', () => {
    it('matches when ANY element matches (Obsidian list-property semantics)', () => {
        const chunk = makeChunk({ metadata: { properties: { genre: ['scifi', 'fantasy'] } } });
        expect(matchesFilters(chunk, filters({ frontmatter: { genre: 'scifi' } }))).toBe(true);
        expect(matchesFilters(chunk, filters({ frontmatter: { genre: 'fantasy' } }))).toBe(true);
        expect(matchesFilters(chunk, filters({ frontmatter: { genre: 'romance' } }))).toBe(false);
    });

    it('quoted-exact matches a whole ELEMENT, not the joined list', () => {
        const chunk = makeChunk({ metadata: { properties: { genre: ['scifi', 'fantasy'] } } });
        expect(matchesFilters(chunk, filters({ frontmatter: { genre: '"scifi"' } }))).toBe(true);
        expect(matchesFilters(chunk, filters({ frontmatter: { genre: '"sci"' } }))).toBe(false);
    });

    it('wikilink list elements stay substring-matchable through toBindForm', () => {
        const chunk = makeChunk({ metadata: { properties: { relatedPages: ['[[Notes/People/Alex Goel|Alex]]'] } } });
        expect(matchesFilters(chunk, filters({ frontmatter: { relatedpages: 'alex' } }))).toBe(true);
    });

    it('numeric clause satisfied by any list element', () => {
        const chunk = makeChunk({ metadata: { properties: { sizes: ['12', '40'] } } });
        expect(matchesFilters(chunk, filters({ numeric: [{ key: 'sizes', op: '>', value: 30 }] }))).toBe(true);
        expect(matchesFilters(chunk, filters({ numeric: [{ key: 'sizes', op: '>', value: 50 }] }))).toBe(false);
    });
});
