import { describe, it, expect } from 'vitest';
import { MarkdownChunker, goForwardDenseParts } from './chunker';
import { embedInput } from './token-budget';

// The go-forward keyed dense channel (CHUNKER_VERSION 11, goForwardDenseParts):
// keyed `key: value` suffix, tags→BM25, machinery/date/shape offenders dropped,
// per-chunk alias + property dedup against title+body. Driven through the real
// chunkContent pipeline so peelDenseTitle is exercised against genuine titles
// (`Base | a1 | a2 > H1 > H2`), matching the validated harness goforward_text.
const chunker = new MarkdownChunker();

const NOTE = [
    '---',
    'pageType: people',
    'aliases:',
    '  - Alex',            // token already in base title "Alex Goel" -> dropped
    '  - RG',              // novel -> kept
    'person: "[[Alex Goel]]"',   // tokens all in title/body -> deduped out of suffix
    'role: Engineering Manager', // novel free-text prop -> kept, keyed
    'icon: lucide-user',   // MACHINERY key -> dropped
    'created: 2026-05-26', // ISO date value -> INERT, dropped',
    'tags:',
    '  - people',          // tags never enter the dense suffix (BM25 only)
    '---',
    '',
    '# Overview',
    'Alex Goel manages the search and discovery team across several product areas.',
    '',
    '# Notes',
    'We met to discuss ranking quality and the upcoming evaluation milestone.',
].join('\n');

describe('go-forward keyed dense channel', () => {
    const chunks = chunker.chunkContent(NOTE, 'People/Alex Goel.md');
    const overview = chunks.find(c => c.heading_path.includes('Overview'))!;

    it('emits a chunk for the Overview section', () => {
        expect(overview).toBeTruthy();
        expect(overview.title).toBe('Alex Goel | Alex | RG > Overview');
    });

    it('keys surviving properties as `key: value`', () => {
        const { suffix } = goForwardDenseParts(overview);
        expect(suffix).toContain('pageType: people');       // enum facet kept + keyed
        expect(suffix).toContain('role: Engineering Manager'); // free-text prop kept + keyed
    });

    it('drops machinery keys, ISO dates, and tags from the suffix', () => {
        const { suffix } = goForwardDenseParts(overview);
        expect(suffix).not.toMatch(/icon|lucide/i);   // machinery key gone
        expect(suffix).not.toContain('2026');         // inert ISO date gone
        expect(suffix).not.toMatch(/tags:/i);         // tags → BM25 only
    });

    it('dedups a property value whose tokens are all in title/body', () => {
        const { suffix } = goForwardDenseParts(overview);
        // `person: [[Alex Goel]]` -> "Alex Goel", already in title+body -> no group
        expect(suffix).not.toMatch(/person:/i);
    });

    it('per-chunk alias dedup: drops redundant aliases, keeps novel ones', () => {
        const { title } = goForwardDenseParts(overview);
        expect(title).toBe('Alex Goel | RG > Overview');  // "Alex" dropped, "RG" kept
    });

    it('embedInput composes denseTitle \\n\\n content \\n\\n keyed suffix', () => {
        const text = embedInput(overview);
        expect(text.startsWith('Alex Goel | RG > Overview\n\n')).toBe(true);
        expect(text).toContain('pageType: people');
        expect(text).not.toContain('| Alex |');   // deduped alias absent from dense title
    });

    it('notes without aliases leave the title untouched', () => {
        const plain = chunker.chunkContent(
            ['---', 'pageType: notes', '---', '', '# H', 'Some ordinary body text that clears the minimum chunk length gate.'].join('\n'),
            'Notes/Plain.md',
        );
        const c = plain.find(x => x.heading_path.includes('H'))!;
        const { title, suffix } = goForwardDenseParts(c);
        expect(title).toBe('Plain > H');
        expect(suffix).toBe('pageType: notes');
    });
});

describe('keyed suffix v12 fixes', () => {
    const mk = (fm: string[]) => chunker.chunkContent(
        ['---', ...fm, '---', '', '# H', 'Some ordinary body text that clears the minimum chunk length gate.'].join('\n'),
        'Notes/Plain.md',
    ).find(x => x.heading_path.includes('H'))!;

    it('treats ISO-week values as inert', () => {
        const c = mk(['pageType: notes', 'week: "[[W35, 2026]]"', 'isoWeek: 2026-W35']);
        expect(goForwardDenseParts(c).suffix).toBe('pageType: notes');
    });

    it('caps at group/value boundaries, never leaving a dangling key', () => {
        const many = Array.from({ length: 30 }, (_, i) => `  - val${i} extra${i}`);
        const c = mk(['pageType: notes', 'related:', ...many, 'owner: someone']);
        const { suffix } = goForwardDenseParts(c);
        const words = suffix.split(/\s+/).filter(Boolean);
        expect(words.length).toBeLessThanOrEqual(64);
        expect(suffix.startsWith('pageType: notes. related: val0 extra0')).toBe(true);
        expect(suffix).toMatch(/extra\d+$/);          // ends on a whole value
        expect(suffix).not.toMatch(/owner:/);          // later group past cap dropped whole
        expect(suffix).not.toMatch(/:\s*$/);
    });
});

describe('keyed suffix cap keeps the head of one overlong value', () => {
    it('truncates a single long description at word level instead of dropping it', () => {
        const long = Array.from({ length: 90 }, (_, i) => `lorem${i}`).join(' ');
        const c = chunker.chunkContent(
            ['---', 'pageType: notes', `description: ${long}`, '---', '', '# H',
             'Some ordinary body text that clears the minimum chunk length gate.'].join('\n'),
            'Notes/Plain.md',
        ).find(x => x.heading_path.includes('H'))!;
        const { suffix } = goForwardDenseParts(c);
        expect(suffix.startsWith('pageType: notes. description: lorem0 lorem1')).toBe(true);
        expect(suffix.split(/\s+/).filter(Boolean).length).toBe(64);
    });
});
