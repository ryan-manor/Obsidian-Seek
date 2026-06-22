import { describe, it, expect } from 'vitest';
import { SuggestEngine } from './suggest';

// Minimal fake of the bits of `App` that SuggestEngine.build() touches:
// metadataCache.getTags(), vault.getMarkdownFiles(), metadataCache.getFileCache().
function fakeApp(opts: {
    tags: Record<string, number>;
    files: Array<{ parent: string; fm?: Record<string, unknown> }>;
}) {
    const files = opts.files.map((f, i) => ({
        path: `${f.parent}/n${i}.md`,
        parent: { path: f.parent },
        __fm: f.fm,
    }));
    return {
        metadataCache: {
            getTags: () => opts.tags,
            getFileCache: (f: { __fm?: Record<string, unknown> }) => ({ frontmatter: f.__fm }),
        },
        vault: { getMarkdownFiles: () => files },
    } as unknown as Parameters<SuggestEngine['build']>[0];
}

function engine() {
    return new SuggestEngine().build(fakeApp({
        tags: {
            '#meetings': 430,
            '#meetings/1x1s': 344,
            '#meetings/interviewer': 10,
            '#claude': 147,
            '#meeting-prep': 50, // kebab-case: parser captures the hyphen — completable, lower count so it doesn't outrank `meetings` on the `mee` prefix
            '#café': 30, // accented letter — Obsidian-valid, must be completable
            '#日本語': 25, // non-Latin — Obsidian-valid, must be completable
            '#bad tag 🎉': 99, // a SPACE makes it non-bindable — must stay excluded (the emoji is fine, the space isn't)
        },
        files: [
            ...Array.from({ length: 5 }, () => ({ parent: 'Notes/Work/Meetings', fm: { pageType: 'task', context: 'work' } })),
            ...Array.from({ length: 2 }, () => ({ parent: 'Notes/Work', fm: { pageType: 'note', context: 'work', 'my-field': 'x' } })),
            { parent: 'Notes/Work/AUP AI Week 2026', fm: { context: 'personal' } }, // folder with spaces
            { parent: 'Notes/Personal', fm: { context: 'personal', completed: false } }, // boolean — must be offered (index String()s it)
        ],
    }));
}

const C = (val: string, atEnd = true) => engine().complete(val, atEnd);

describe('SuggestEngine — tags', () => {
    it('completes a tag prefix to the highest-count match', () => {
        const c = C('tag:mee');
        expect(c?.accept).toBe('tag:meetings');
        expect(c?.ghost).toBe('tings');
        expect(c?.kind).toBe('tag');
    });

    it('preserves the # sigil', () => {
        expect(C('#mee')?.accept).toBe('#meetings');
        expect(C('#mee')?.ghost).toBe('tings');
    });

    it('deepens a hierarchical tag (skips the exact already-typed value)', () => {
        const c = C('tag:meetings');
        expect(c?.accept).toBe('tag:meetings/1x1s'); // 344 beats /interviewer's 10
        expect(c?.ghost).toBe('/1x1s');
    });

    it('preserves the head before the active token', () => {
        expect(C('recipes tag:mee')?.accept).toBe('recipes tag:meetings');
        expect(C('recipes tag:mee')?.ghost).toBe('tings');
    });
});

describe('SuggestEngine — paths', () => {
    it('completes to the busiest folder and appends /* so the filter binds', () => {
        const c = C('path:Notes/W');
        expect(c?.accept).toBe('path:Notes/Work/Meetings/*'); // 5 beats Notes/Work's 2
        expect(c?.kind).toBe('path');
    });

    it('never suggests a folder containing spaces (path: is \\S+)', () => {
        // "AUP AI Week 2026" would be the only Notes/Work/A* match; it must be skipped.
        expect(C('path:Notes/Work/A')).toBeNull();
    });
});

describe('SuggestEngine — frontmatter fields', () => {
    it('completes [key:value] and closes the bracket', () => {
        const c = C('[pageType:ta');
        expect(c?.accept).toBe('[pageType:task]');
        expect(c?.ghost).toBe('sk]');
    });

    it('returns null for an unknown / non-categorical key', () => {
        expect(C('[nope:x')).toBeNull();
    });

    it('lists the busiest key for a bare [ (key menu)', () => {
        // pageType appears on 7 files, context on 9 -> context is busiest and
        // is offered first in the bare-[ key menu.
        expect(C('[')?.accept).toBe('[context:');
    });

    it('never offers a value containing [ or ] (breaks the token grammar)', () => {
        // "[status:in [progress]]" would defeat activeToken's lastIndexOf
        // bracket detection — such values must be dropped at build time.
        const e = new SuggestEngine().build(fakeApp({
            tags: {},
            files: [
                { parent: 'Notes', fm: { status: 'in [progress]' } },
                { parent: 'Notes', fm: { status: 'closed]' } },
                { parent: 'Notes', fm: { status: 'open' } },
            ],
        }));
        expect(e.complete('[status:in', true)).toBeNull();
        expect(e.complete('[status:cl', true)).toBeNull();
        expect(e.complete('[status:op', true)?.accept).toBe('[status:open]');
    });
});

describe('SuggestEngine — space-tolerant field tokens (parser allows [key: value])', () => {
    it('completes despite a space after the colon; ghost is suppressed (accept is not a literal extension)', () => {
        const c = C('[pageType: ta');
        expect(c?.accept).toBe('[pageType:task]'); // canonical, space normalized away
        expect(c?.ghost).toBe('');
    });

    it('lists value rows for a bare `[key: ` with a trailing space', () => {
        const rows = engine().listSuggestions('[pageType: ', 8);
        expect(rows.map(r => r.value)).toEqual(['pageType:task', 'pageType:note']); // busiest first
        expect(rows[0].accept).toBe('[pageType:task]');
    });

    it('offers a boolean-valued key — the index String()s it, so [completed:false] binds', () => {
        expect(C('[completed:')?.accept).toBe('[completed:false]');
    });
});

describe('SuggestEngine — non-completions', () => {
    it('does not complete bare words', () => {
        expect(C('evaluation')).toBeNull();
    });

    it('does not treat [[ as a filter', () => {
        expect(C('[[query un')).toBeNull();
    });

    it('shows nothing when the caret is not at end of input', () => {
        expect(C('tag:mee', /* atEnd */ false)).toBeNull();
    });

    it('ghost is always the accept minus the typed value', () => {
        const c = C('tag:mee')!;
        expect('tag:mee' + c.ghost).toBe(c.accept);
    });
});

describe('SuggestEngine — binding contract (only suggest what the parser captures)', () => {
    it('offers a kebab-case tag — the parser captures [\\w-]+', () => {
        // #meeting-prep binds whole now, so completing the hyphen is correct.
        expect(C('tag:meeting-')?.accept).toBe('tag:meeting-prep');
        // It's lower-count than `meetings`, so the bare `mee` prefix still
        // resolves to the busier plain tag.
        expect(C('tag:mee')?.accept).toBe('tag:meetings');
    });

    it('offers a kebab-case frontmatter key — bracket keys are [\\w-]+', () => {
        // `my-field` is on 2 files; `[my-field:` binds, so offer it.
        expect(C('[my-')?.accept).toBe('[my-field:');
    });

    it('offers accented / non-Latin tags (Obsidian allows any letter)', () => {
        expect(C('tag:caf')?.accept).toBe('tag:café');
        expect(C('tag:caf')?.ghost).toBe('é');
        expect(C('#日本')?.accept).toBe('#日本語');
    });

    it('still excludes a tag with a space (the real disqualifier, not the emoji)', () => {
        // `bad tag 🎉` can never be a `tag:`/`#` token because of the spaces.
        expect(C('tag:bad')).toBeNull();
        expect(C('#bad')).toBeNull();
    });
});

describe('SuggestEngine — wikilink values (the placeLoc capture, 2026-06-12)', () => {
    const places = () => new SuggestEngine().build(fakeApp({
        tags: {},
        files: [
            ...Array.from({ length: 3 }, () => ({ parent: 'Places', fm: { placeLoc: '[[Austin]]' } })),
            ...Array.from({ length: 2 }, () => ({ parent: 'Places', fm: { placeLoc: '[[Paris]]' } })),
            { parent: 'Places', fm: { visited: '[[2026-05-31|May 31]]' } },
            { parent: 'Places', fm: { visited: '[[2026-05-31|May 31]]' } },
        ],
    }));

    it('offers a wikilink value unwrapped — the matcher sees through [[...]], so suggest must too', () => {
        const c = places().complete('[placeLoc:Au', true);
        expect(c?.accept).toBe('[placeLoc:Austin]'); // stored "[[Austin]]", offered "Austin"
    });

    it('surfaces the KEY of a wikilink-only field in the bare-[ menu', () => {
        // Before the fix, zero values survived the bracket guard, so the key
        // itself never registered and `[placeLo` suggested nothing.
        expect(places().complete('[placeLo', true)?.accept).toBe('[placeLoc:');
    });

    it('offers an aliased wikilink as its target basename only (display-form; alias dropped, still binds by substring)', () => {
        const rows = places().listSuggestions('[visited:', 8);
        expect(rows[0]?.value).toBe('visited:2026-05-31');
    });
});

describe('SuggestEngine — case-insensitive matching (binding is case-insensitive)', () => {
    it('matches a field key typed in the wrong case; ghost is suppressed (not a literal extension)', () => {
        const e = new SuggestEngine().build(fakeApp({
            tags: {},
            files: [{ parent: 'Places', fm: { placeLoc: '[[Austin]]' } }, { parent: 'Places', fm: { placeLoc: '[[Austin]]' } }],
        }));
        const c = e.complete('[placelo', true);
        expect(c?.accept).toBe('[placeLoc:'); // canonical casing preserved in the accept
        expect(c?.ghost).toBe('');
    });

    it('matches tags and field values case-insensitively', () => {
        expect(C('tag:MEE')?.accept).toBe('tag:meetings');
        expect(C('tag:MEE')?.ghost).toBe('');
        expect(C('[pageType:TA')?.accept).toBe('[pageType:task]');
    });
});

describe('SuggestEngine — wide categorical keys (cardinality gate)', () => {
    const wide = (repeats: number) => new SuggestEngine().build(fakeApp({
        tags: {},
        files: Array.from({ length: 45 * repeats }, (_, i) => ({
            parent: 'Notes',
            fm: { city: `City${i % 45}` }, // 45 distinct values, > MAX_KEY_CARDINALITY
        })),
    }));

    it('admits a >40-distinct key whose values repeat (categorical, e.g. placeLoc)', () => {
        expect(wide(2).complete('[city:City4', true)?.accept).toBe('[city:City4]');
    });

    it('still drops a per-note-unique key (distinct ≈ total → free text)', () => {
        expect(wide(1).complete('[city:City4', true)).toBeNull();
    });
});

describe('SuggestEngine — quoted paths (spaces, Obsidian path:"…")', () => {
    it('completes a space-containing folder inside an open quote, closing it', () => {
        const c = C('path:"Notes/Work/A');
        expect(c?.accept).toBe('path:"Notes/Work/AUP AI Week 2026/*"');
        expect(c?.kind).toBe('path');
    });

    it('picks the busiest matching folder inside a quote', () => {
        expect(C('path:"Notes/W')?.accept).toBe('path:"Notes/Work/Meetings/*"'); // 5 beats Notes/Work's 2
    });
});
