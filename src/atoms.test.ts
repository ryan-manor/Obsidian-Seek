// WS3 step zero — the constructed real-markdown set, encoded as expectations.
//
// Each fixture in src/fixtures/ is a realistic vault-style note locking ONE
// structural behavior of the atom parser (decision of record, plan §WS3):
// fences, tables, and callouts are ATOMIC — never split internally by any
// blank-line boundary — and heading detection is fence-aware. These tests are
// the gate for fence-aware chunking (plugin unit tests, NOT corpus nDCG).
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { parseAtoms, scanHeadings, type Atom } from './atoms';

const fx = (name: string): string =>
    readFileSync(new URL(`./fixtures/${name}`, import.meta.url), 'utf8');

const types = (atoms: Atom[]): string[] => atoms.map(a => a.type);

describe('parseAtoms — structural atomicity', () => {
    it('keeps a fence with internal blank lines as one atom (the motivating bug)', () => {
        const atoms = parseAtoms(fx('fence-blank-lines.md'));
        expect(types(atoms)).toEqual(['paragraph', 'fence', 'paragraph']);
        const fence = atoms[1].text;
        expect(fence.startsWith('```python')).toBe(true);
        expect(fence.endsWith('```')).toBe(true);
        // The blank lines that /\n\n+/ used to split on are still inside.
        expect(fence).toMatch(/\n\s*\n/);
        expect(fence).toContain('class Pacer');
        expect(fence).toContain('def tick');
    });

    it('keeps a tilde fence whole, preserving its info string and inner backtick fence', () => {
        const atoms = parseAtoms(fx('fence-tilde-nested.md'));
        expect(types(atoms)).toEqual(['paragraph', 'fence', 'paragraph']);
        const fence = atoms[1].text;
        expect(fence.startsWith('~~~markdown title="example"')).toBe(true);
        expect(fence.endsWith('~~~')).toBe(true);
        // The inner ```js fence is literal content, not a close of the outer fence.
        expect(fence).toContain('```js');
        expect(fence).toContain('console.log("hi");');
    });

    it('runs an unterminated fence to end of input', () => {
        const atoms = parseAtoms(fx('fence-unterminated.md'));
        expect(types(atoms)).toEqual(['paragraph', 'fence']);
        expect(atoms[1].text.startsWith('```text')).toBe(true);
        expect(atoms[1].text.endsWith('stays code.')).toBe(true);
    });

    it('keeps a GFM table as one atom', () => {
        const atoms = parseAtoms(fx('table-simple.md'));
        expect(types(atoms)).toEqual(['paragraph', 'table', 'paragraph']);
        const rows = atoms[1].text.split('\n');
        expect(rows).toHaveLength(5); // header + delimiter + 3 data rows
        expect(rows[0]).toContain('| corpus');
        expect(rows[1]).toMatch(/^\|[\s:|-]+\|$/);
    });

    it('keeps callouts whole — including bare > lines and nested quotes', () => {
        const atoms = parseAtoms(fx('callout-multiline.md'));
        expect(types(atoms)).toEqual(
            ['paragraph', 'callout', 'paragraph', 'callout', 'paragraph']);
        const warning = atoms[1].text;
        expect(warning.startsWith('> [!warning]')).toBe(true);
        // The bare ">" continuation line must not split the callout.
        expect(warning).toMatch(/\n>\n/);
        expect(warning).toContain('> > nested quote');
        expect(atoms[3].text.startsWith('> [!todo]-')).toBe(true);
    });

    it('segments a mixed section into the expected atom sequence', () => {
        const atoms = parseAtoms(fx('mixed-section.md'));
        expect(types(atoms)).toEqual(
            ['paragraph', 'fence', 'paragraph', 'table', 'callout', 'paragraph']);
    });

    it('matches the old blank-line split exactly on plain prose (regression anchor)', () => {
        const src = fx('paragraphs.md');
        const atoms = parseAtoms(src);
        const legacy = src.split(/\n\n+/).map(s => s.trim()).filter(Boolean);
        expect(atoms.every(a => a.type === 'paragraph')).toBe(true);
        expect(atoms.map(a => a.text)).toEqual(legacy);
    });

    it('loses no non-whitespace content on any fixture (round-trip invariant)', () => {
        const fixtures = [
            'fence-blank-lines.md', 'fence-hash-heading.md', 'fence-tilde-nested.md',
            'fence-unterminated.md', 'table-simple.md', 'callout-multiline.md',
            'mixed-section.md', 'paragraphs.md', 'oversize-fence.md', 'oversize-table.md',
        ];
        for (const name of fixtures) {
            const src = fx(name);
            const joined = parseAtoms(src).map(a => a.text).join('\n\n');
            // Whitespace may normalize (same loss class as the old split);
            // every other character survives in order.
            expect(joined.replace(/\s+/g, '')).toBe(src.replace(/\s+/g, ''));
            // Idempotence: re-parsing the join yields the same atoms.
            const reparsed = parseAtoms(joined);
            expect(reparsed.map(a => a.text)).toEqual(parseAtoms(src).map(a => a.text));
        }
    });

    it('handles empty and whitespace-only input', () => {
        expect(parseAtoms('')).toEqual([]);
        expect(parseAtoms('  \n\n   \n')).toEqual([]);
    });
});

describe('scanHeadings — fence awareness', () => {
    it('ignores # lines inside fences', () => {
        const lines = fx('fence-hash-heading.md').split('\n');
        const headings = scanHeadings(lines);
        expect(headings.map(h => h.text)).toEqual(['Shell Setup', 'Verification']);
        expect(headings.map(h => h.level)).toEqual([1, 2]);
    });

    it('finds no headings when the # lines sit in an unterminated fence', () => {
        const lines = fx('fence-unterminated.md').split('\n');
        expect(scanHeadings(lines)).toEqual([]);
    });

    it('reports correct line numbers outside fences', () => {
        const src = '# Top\n\n```\n# not a heading\n```\n\n## After\nbody';
        const headings = scanHeadings(src.split('\n'));
        expect(headings).toEqual([
            { lineNum: 0, level: 1, text: 'Top' },
            { lineNum: 6, level: 2, text: 'After' },
        ]);
    });
});
