import { describe, it, expect } from 'vitest';
import { sanitizeSnippet } from './snippet';

// The headline case (Inbox/Seek Dense Channel CSLS De-Hubbing.md): a $$…$$
// formula rendered as a centred MathJax block onto its own line, ballooning the
// result row. We strip the block but keep the prose around it; inline $math$ and
// a lone `$` (a price) must survive, since they render inline and never break the
// row.
describe('sanitizeSnippet — display math', () => {
    it('strips a $$…$$ block but keeps the surrounding prose', () => {
        const md = 'CSLS kills hubness.\n\n$$\\text{CSLS}(q, d) = 2\\cos(q, d) - r_k(q) - r_k(d)$$\n\nwhere r_k is the mean.';
        const out = sanitizeSnippet(md);
        expect(out).not.toContain('$$');
        expect(out).not.toContain('cos');
        expect(out).toContain('CSLS kills hubness.');
        expect(out).toContain('where r_k is the mean.');
    });

    it('strips a $$…$$ block that spans multiple lines', () => {
        const md = 'before\n\n$$\na = b\n+ c\n$$\n\nafter';
        const out = sanitizeSnippet(md);
        expect(out).not.toContain('$$');
        expect(out).not.toContain('a = b');
        expect(out).toBe('before\n\nafter');
    });

    it('strips every $$…$$ block (non-greedy, not the span between two blocks)', () => {
        const md = 'one $$a=b$$ two $$c=d$$ three';
        const out = sanitizeSnippet(md);
        expect(out).not.toContain('a=b');
        expect(out).not.toContain('c=d');
        expect(out).toContain('one');
        expect(out).toContain('two'); // the prose BETWEEN the two blocks is kept
        expect(out).toContain('three');
    });

    it('leaves inline $math$ and a lone $ (price) intact', () => {
        const md = 'Inline $x^2$ stays, and a price of $5 too.';
        expect(sanitizeSnippet(md)).toBe('Inline $x^2$ stays, and a price of $5 too.');
    });
});

// Regression guards for the rest of the function, locked in now that it lives in
// its own module.
describe('sanitizeSnippet — embeds, tables, fences', () => {
    it('strips wikilink + markdown image embeds, keeps plain links', () => {
        const md = 'see ![[cover.jpg]] and ![alt](pic.png) but [[Real Note]] and [text](https://x.com) stay';
        const out = sanitizeSnippet(md);
        expect(out).not.toContain('cover.jpg');
        expect(out).not.toContain('alt');
        expect(out).toContain('[[Real Note]]');
        expect(out).toContain('[text](https://x.com)');
    });

    it('drops GFM pipe-table lines but keeps a stray inline pipe in prose', () => {
        const md = 'intro\n| a | b |\n|---|---|\n| 1 | 2 |\nchoose A | B by hand';
        const out = sanitizeSnippet(md);
        expect(out).not.toContain('| a | b |');
        expect(out).not.toContain('|---|');
        expect(out).toContain('intro');
        expect(out).toContain('choose A | B by hand'); // prose pipe survives
    });

    it('removes code-fence markers but keeps the inner code text', () => {
        const md = 'note\n```json\n{"k": 1}\n```\nend';
        const out = sanitizeSnippet(md);
        expect(out).not.toContain('```');
        expect(out).toContain('{"k": 1}');
    });

    it('collapses leftover blank runs and trims', () => {
        const md = '\n\n$$a=b$$\n\n\n\ntext\n\n';
        expect(sanitizeSnippet(md)).toBe('text');
    });
});

// A chunk that's ENTIRELY a table or a $$…$$ formula has no surrounding prose
// to fall back on, so the normal strip-and-trim above returns ''. Rather than
// showing a blank result row for a chunk that DID match, render a flattened
// one-liner of the stripped content so it's still recognizable.
describe('sanitizeSnippet — table-only / math-only fallback', () => {
    it('flattens a table-only chunk instead of returning blank', () => {
        const md = '| Name | Age |\n|---|---|\n| Alice | 30 |\n| Bob | 40 |';
        const out = sanitizeSnippet(md);
        expect(out).not.toBe('');
        expect(out).toContain('Name');
        expect(out).toContain('Alice');
        expect(out).toContain('Bob');
        expect(out).not.toContain('|---|');
    });

    it('flattens a math-only chunk instead of returning blank', () => {
        const md = '$$\n\\text{CSLS}(q, d) = 2\\cos(q, d) - r_k(q) - r_k(d)\n$$';
        const out = sanitizeSnippet(md);
        expect(out).not.toBe('');
        expect(out).not.toContain('$$');
        expect(out).toContain('CSLS');
    });

    it('still returns empty for a chunk with no extractable content at all', () => {
        const md = '![[cover.jpg]]\n\n![alt](pic.png)';
        expect(sanitizeSnippet(md)).toBe('');
    });
});
