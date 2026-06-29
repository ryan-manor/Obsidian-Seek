// Contract tests for the dense-channel text cleaner (Seek v8). Pins the
// rendered-display wikilink rule (distinct from prop-normalize's basename rule),
// embed canonicalization, URL scheme/TLD stripping with path-word retention, and
// the fence-preservation invariant the code-retrieval eval depends on.

import { describe, it, expect } from 'vitest';
import { cleanDenseText, cleanDenseBody } from './dense-clean';

describe('cleanDenseText — wikilinks render as the reader sees them', () => {
    it('keeps the ALIAS (opposite of toDisplayForm)', () => {
        expect(cleanDenseText('met [[Alex Goel|Alex]] today')).toBe('met Alex today');
        expect(cleanDenseText('[[San Francisco|SF]] trip')).toBe('SF trip');
    });

    it('falls back to the target basename when there is no alias', () => {
        expect(cleanDenseText('see [[Austin]]')).toBe('see Austin');
        expect(cleanDenseText('[[Notes/Personal/Places/Zurich]]')).toBe('Zurich');
    });

    it('drops #heading and ^block refs', () => {
        expect(cleanDenseText('[[Project Eames#Status]]')).toBe('Project Eames');
        expect(cleanDenseText('[[Meeting Notes^abc123]]')).toBe('Meeting Notes');
    });

    it('never emits wikilink syntax', () => {
        expect(cleanDenseText('[[A/B/C|D]] and [[Foo#Bar]]')).not.toMatch(/[[\]|]/);
    });
});

describe('cleanDenseText — embeds canonicalize to display name', () => {
    it('drops image/asset embeds (no readable text)', () => {
        expect(cleanDenseText('![[Pasted image 20260628.png]]')).toBe('');
        expect(cleanDenseText('intro ![[diagram.svg]] outro')).toBe('intro outro');
        expect(cleanDenseText('![alt text](https://x.com/a.png)')).toBe('alt text');
        expect(cleanDenseText('![](https://x.com/a.png)')).toBe('');
    });

    it('keeps a note transclusion as its basename / alias', () => {
        expect(cleanDenseText('![[Some Note]]')).toBe('Some Note');
        expect(cleanDenseText('![[Some Note#Section]]')).toBe('Some Note');
        expect(cleanDenseText('![[Some Note|Shown]]')).toBe('Shown');
    });
});

describe('cleanDenseText — markdown links and bare URLs', () => {
    it('keeps link text, drops the URL', () => {
        expect(cleanDenseText('read [the verge piece](https://www.theverge.com/x)'))
            .toBe('read the verge piece');
    });

    it('strips scheme, www and TLD from a bare URL but keeps host label + path words', () => {
        expect(cleanDenseText('https://www.theverge.com/2026/tech/great-article'))
            .toBe('theverge 2026 tech great article');
    });

    it('preserves place-name words encoded in a URL path', () => {
        // the tokenize.ts Google-Maps case: the place name lives in the path
        expect(cleanDenseText('https://maps.google.com/100-198+E+5th+St+Garage/data=x'))
            .toContain('E 5th St Garage');
    });

    it('drops the query string and fragment', () => {
        expect(cleanDenseText('https://example.com/p?utm_source=hn&ref=x#frag'))
            .toBe('example p');
    });
});

describe('cleanDenseText — html and thematic breaks', () => {
    it('strips html tags, keeps inner text', () => {
        expect(cleanDenseText('<b>bold</b> and <i>italic</i>')).toBe('bold and italic');
        expect(cleanDenseText('<a href="https://x.com">anchor</a>')).toBe('anchor');
    });

    it('removes thematic-break lines', () => {
        expect(cleanDenseText('title\n\n---\n\nbody')).toBe('title\n\nbody');
    });
});

describe('query-side parity (search.ts S0.5 dense embed)', () => {
    // granite-r2 is symmetric, so the query is dense-cleaned with the SAME pass
    // the chunker applies to the doc side (audit 2026-06-29 finding #1). These
    // pin the two properties that make wrapping the dense query call safe.

    it('is a NO-OP on an ordinary plain-text query (normal queries unchanged)', () => {
        expect(cleanDenseText('kubernetes deployment notes')).toBe('kubernetes deployment notes');
        expect(cleanDenseText('alex 1x1')).toBe('alex 1x1');
        expect(cleanDenseText('café zürich trip')).toBe('café zürich trip');
    });

    it('a query carrying [[…]] / a URL cleans to the same surface form the doc side embeds', () => {
        // doc side: cleanDenseText('met [[Alex Goel|Alex]]') === 'met Alex' — the
        // query must land on the same tokens or the vectors drift.
        expect(cleanDenseText('[[Alex Goel|Alex]] 1x1')).toBe('Alex 1x1');
        expect(cleanDenseText('https://www.theverge.com/x review')).toBe('theverge x review');
    });
});

describe('cleanDenseBody — fence-aware', () => {
    it('leaves fenced code VERBATIM (urls/tags/embeds untouched)', () => {
        const code = '```js\nfetch("https://api.example.com/v1?x=1") // <not a tag>\n```';
        expect(cleanDenseBody(code)).toBe(code);
    });

    it('cleans prose around a code fence but not inside it', () => {
        const input = 'see [docs](https://docs.example.com/guide)\n\n```\ncurl https://api.x.com\n```';
        const out = cleanDenseBody(input);
        expect(out).toContain('see docs');
        expect(out).toContain('curl https://api.x.com'); // inside fence: untouched
    });

    it('cleans table and callout atoms', () => {
        expect(cleanDenseBody('> [!note] see [[Alex Goel|Alex]]')).toContain('Alex');
    });

    it('drops a paragraph that cleans to nothing', () => {
        expect(cleanDenseBody('![[cover.jpg]]\n\nreal text')).toBe('real text');
    });
});
