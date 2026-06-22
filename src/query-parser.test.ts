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
import type { Chunk, ChunkMetadata, QueryFilters } from './types';

// ---- helpers ----

function makeChunk(over: Partial<Omit<Chunk, 'metadata'>> & { metadata?: Partial<ChunkMetadata> } = {}): Chunk {
    const { metadata: metaOver, ...rest } = over;
    const metadata: ChunkMetadata = {
        tags: [],
        aliases: [],
        pageType: '',
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
        createdAfter: null,
        createdBefore: null,
        modifiedAfter: null,
        modifiedBefore: null,
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

describe('parseQuery / date filters', () => {
    it('created after', () => {
        const { filters } = parseQuery('alex [created:>2026-04-01]');
        expect(filters!.createdAfter).toBe('2026-04-01');
        expect(filters!.createdBefore).toBeNull();
        expect(filters!.frontmatter).toBeNull();
    });

    it('created before', () => {
        const { filters } = parseQuery('alex [created:<2026-05-01]');
        expect(filters!.createdBefore).toBe('2026-05-01');
    });

    it('modified after', () => {
        const { filters } = parseQuery('alex [modified:>2026-04-01]');
        expect(filters!.modifiedAfter).toBe('2026-04-01');
    });

    it('modified before', () => {
        const { filters } = parseQuery('alex [modified:<2026-05-01]');
        expect(filters!.modifiedBefore).toBe('2026-05-01');
    });

    it('date keys are case-insensitive', () => {
        const a = parseQuery('[Created:>2026-04-01]');
        expect(a.filters!.createdAfter).toBe('2026-04-01');
        expect(a.filters!.frontmatter).toBeNull();

        const b = parseQuery('[Modified:<2026-05-01]');
        expect(b.filters!.modifiedBefore).toBe('2026-05-01');
        expect(b.filters!.frontmatter).toBeNull();
    });

    it('created without operator is frontmatter equality, not a date range', () => {
        const { filters } = parseQuery('[created:2026-04-01]');
        expect(filters!.frontmatter).toEqual({ created: '2026-04-01' });
        expect(filters!.createdAfter).toBeNull();
        expect(filters!.createdBefore).toBeNull();
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
        const { filters } = parseQuery('alex [created:>2026-04-01] #meetings/1x1');
        expect(filters!.createdAfter).toBe('2026-04-01');
        expect(filters!.tags).toEqual(['meetings/1x1']);
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

    it('hyphenated negation splits into tokens (-foo-bar → foo, bar)', () => {
        const { filters } = parseQuery('x -foo-bar');
        expect(filters!.exclude).toEqual(['foo', 'bar']);
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

describe('matchesFilters / dates (missing→reject)', () => {
    const c = makeChunk({ metadata: { created: '2026-05-16', modified: '2026-05-19' } });

    it('createdAfter passes when on/after', () => {
        expect(matchesFilters(c, filters({ createdAfter: '2026-05-01' }))).toBe(true);
    });

    it('createdAfter rejects when before', () => {
        expect(matchesFilters(c, filters({ createdAfter: '2026-06-01' }))).toBe(false);
    });

    it('createdBefore passes when on/before', () => {
        expect(matchesFilters(c, filters({ createdBefore: '2026-05-20' }))).toBe(true);
    });

    it('createdBefore rejects when after', () => {
        expect(matchesFilters(c, filters({ createdBefore: '2026-05-10' }))).toBe(false);
    });

    it('missing created date rejects a created filter', () => {
        const noDate = makeChunk({ metadata: { created: null } });
        expect(matchesFilters(noDate, filters({ createdAfter: '2026-01-01' }))).toBe(false);
    });

    it('modified range', () => {
        expect(matchesFilters(c, filters({ modifiedAfter: '2026-05-18' }))).toBe(true);
        expect(matchesFilters(c, filters({ modifiedAfter: '2026-05-20' }))).toBe(false);
    });
});

describe('matchesFilters / combined predicates (AND across filter types)', () => {
    const c = makeChunk({
        note_path: 'Notes/Work/q2.md',
        metadata: { tags: ['projects'], properties: { context: 'work' }, created: '2026-05-16' },
    });

    it('all conditions satisfied', () => {
        const f = filters({ tags: ['projects'], frontmatter: { context: 'work' }, createdAfter: '2026-05-01', includePaths: ['Notes/*'] });
        expect(matchesFilters(c, f)).toBe(true);
    });

    it('one failing condition fails the whole match', () => {
        const f = filters({ tags: ['projects'], frontmatter: { context: 'personal' } });
        expect(matchesFilters(c, f)).toBe(false);
    });
});

describe('compileMatcher is reusable across chunks', () => {
    it('returns a closure applied to many chunks', () => {
        const m = compileMatcher(filters({ tags: ['meetings'] }));
        expect(m(makeChunk({ metadata: { tags: ['meetings/1x1'] } }))).toBe(true);
        expect(m(makeChunk({ metadata: { tags: ['projects'] } }))).toBe(false);
    });
});
