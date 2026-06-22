// Invariant tests for WS2.3 token-budget enforcement (token-budget.ts).
//
// The contract under test, in priority order:
//   1. BUDGET: every emitted chunk's composed embed input counts ≤ budget
//      (exception: the unsplittable oversize-title pathology, which must be
//      counted in overBudget — never silent).
//   2. NO CONTENT LOSS: the unit sequence (paragraph blocks) of the original
//      content survives re-packing exactly; hard-split loses only whitespace.
//   3. IDENTITY: untouched chunks pass through object-identical; split parts
//      get distinct path-salted chunk_ids, the same embedded `title`, and
//      display-only part numbering.
//   4. TERMINATION: adversarial tokenizers (joiner-heavy counts that break
//      the sum-of-units ≥ composed assumption) and space-less blobs converge.
//
// The fake tokenizer mirrors the two properties the packer leans on: counts
// include +2 specials per standalone call, and sub-word pieces scale with
// word length — so standalone unit sums OVERESTIMATE composed counts, same
// direction as real BPE. One test then inverts that property on purpose.

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import type { Chunk, ChunkMetadata } from './types';
import { enforceTokenBudget, embedInput, type CountTokens } from './token-budget';
import { chunkIdFor } from './chunker';
import { parseAtoms } from './atoms';

const META: ChunkMetadata = {
    tags: [], aliases: [], pageType: '', created: null, modified: null, properties: {},
};

function mkChunk(over: Partial<Chunk> & { content: string }): Chunk {
    return {
        chunk_id: 'seed',
        title: 'Note > Section',
        note_path: 'Notes/Note.md',
        heading_path: ['Section'],
        metadata: META,
        start_line: 1,
        end_line: 99,
        ...over,
    };
}

// Word-piece fake: ceil(len/4) tokens per whitespace word + 2 specials.
// Deterministic, sync-cheap, and conservative in the same direction as BPE.
function fakeCount(text: string): number {
    const words = text.split(/\s+/).filter(Boolean);
    return 2 + words.reduce((n, w) => n + Math.ceil(w.length / 4), 0);
}
const counter: CountTokens = async texts => texts.map(fakeCount);

const paras = (content: string) => content.split(/\n\n+/).map(s => s.trim()).filter(Boolean);

function makeParagraph(words: number, tag: string): string {
    return Array.from({ length: words }, (_, i) => `${tag}word${i}`).join(' ');
}

describe('enforceTokenBudget', () => {
    it('passes under-budget chunks through object-identical with exact counts', async () => {
        const chunks = [
            mkChunk({ content: 'short body one' }),
            mkChunk({ content: 'another small body', title: 'Other > Bit' }),
        ];
        const r = await enforceTokenBudget(chunks, counter, 512);
        expect(r.chunks[0]).toBe(chunks[0]);
        expect(r.chunks[1]).toBe(chunks[1]);
        expect(r.splits).toBe(0);
        expect(r.overBudget).toBe(0);
        expect(r.counts).toEqual(chunks.map(c => fakeCount(embedInput(c))));
    });

    it('re-packs an oversize chunk: every part ≤ budget, units preserved in order', async () => {
        const body = Array.from({ length: 12 }, (_, i) => makeParagraph(30, `p${i}`)).join('\n\n');
        const chunk = mkChunk({ content: body });
        expect(fakeCount(embedInput(chunk))).toBeGreaterThan(512);

        const r = await enforceTokenBudget([chunk], counter, 512);
        expect(r.splits).toBe(1);
        expect(r.overBudget).toBe(0);
        expect(r.chunks.length).toBeGreaterThan(1);
        for (let i = 0; i < r.chunks.length; i++) {
            const c = fakeCount(embedInput(r.chunks[i]));
            expect(c).toBeLessThanOrEqual(512);
            expect(r.counts[i]).toBe(c); // routing counts are the real composed counts
        }
        // No content loss, no reordering: concatenated unit streams identical.
        expect(r.chunks.flatMap(c => paras(c.content))).toEqual(paras(body));
    });

    it('split parts: same embedded title, distinct chunk_ids, display-only numbering', async () => {
        const body = Array.from({ length: 10 }, (_, i) => makeParagraph(40, `q${i}`)).join('\n\n');
        const chunk = mkChunk({ content: body });
        const r = await enforceTokenBudget([chunk], counter, 512);

        const ids = new Set(r.chunks.map(c => c.chunk_id));
        expect(ids.size).toBe(r.chunks.length);
        for (let i = 0; i < r.chunks.length; i++) {
            expect(r.chunks[i].title).toBe(chunk.title); // embedded title untouched
            expect(r.chunks[i].displayTitle).toBe(`${chunk.title} (part ${i + 1})`);
            expect(r.chunks[i].note_path).toBe(chunk.note_path);
            expect(r.chunks[i].metadata).toBe(chunk.metadata);
        }
    });

    it('hard-splits a space-less blob (JWT class) losing nothing but whitespace', async () => {
        const blob = 'A'.repeat(9000); // one unit, no spaces: fake → 2 + 2250 tokens
        const chunk = mkChunk({ content: blob });
        const r = await enforceTokenBudget([chunk], counter, 512);
        expect(r.chunks.length).toBeGreaterThan(1);
        for (const c of r.chunks) {
            expect(fakeCount(embedInput(c))).toBeLessThanOrEqual(512);
        }
        const rejoined = r.chunks.map(c => c.content).join('');
        expect(rejoined.replace(/\s+/g, '')).toBe(blob);
        expect(r.overBudget).toBe(0);
    });

    it('counts an unsplittable window-filling title instead of splitting it', async () => {
        // Title alone ~505 tokens → content budget below the packing floor.
        const hugeTitle = Array.from({ length: 503 }, (_, i) => 'aaaa'.repeat(1) + i).join(' ');
        const chunk = mkChunk({ title: hugeTitle, content: makeParagraph(200, 'b') });
        const r = await enforceTokenBudget([chunk], counter, 512);
        expect(r.overBudget).toBe(1);
        expect(r.chunks).toEqual([chunk]); // emitted unchanged, pathology surfaced
    });

    it('converges under an adversarial joiner-heavy tokenizer (verify-demote path)', async () => {
        // Composed counts EXCEED the sum of standalone unit counts: each \n\n
        // join costs 9 tokens, standalone calls pay no specials. The greedy
        // pack underestimates, so the verify loop MUST demote to hold the
        // invariant — this is the BPE-counts-aren't-additive regression test.
        const advCount = (text: string): number => {
            const words = text.split(/[ ]+/).filter(Boolean).length;
            const joins = (text.match(/\n\n/g) ?? []).length;
            return words + 9 * joins;
        };
        const adv: CountTokens = async texts => texts.map(advCount);
        const body = Array.from({ length: 40 }, (_, i) => makeParagraph(60, `r${i}`)).join('\n\n');
        const chunk = mkChunk({ content: body });

        const r = await enforceTokenBudget([chunk], adv, 512);
        for (let i = 0; i < r.chunks.length; i++) {
            expect(advCount(embedInput(r.chunks[i]))).toBeLessThanOrEqual(512);
        }
        expect(r.chunks.flatMap(c => paras(c.content))).toEqual(paras(body));
        expect(r.overBudget).toBe(0);
    });

    it('handles the empty chunk list without a counter call', async () => {
        const r = await enforceTokenBudget([], async () => { throw new Error('must not be called'); }, 512);
        expect(r).toEqual({ chunks: [], counts: [], splits: 0, overBudget: 0 });
    });
});

// WS3: units are ATOMS — fences/tables/callouts pack whole, and when one
// alone exceeds the content budget it splits structure-aware (fence-reopen
// markers, repeated table headers, ">"-line callout pieces). Fixtures are
// the constructed real-markdown set in src/fixtures/.
describe('enforceTokenBudget — WS3 structural atoms', () => {
    const fx = (name: string): string =>
        readFileSync(new URL(`./fixtures/${name}`, import.meta.url), 'utf8');

    const fenceMarkerLines = (text: string): string[] =>
        text.split('\n').filter(l => /^ {0,3}(`{3,}|~{3,})/.test(l));

    it('packs a fence whole when it fits, never splitting at its internal blank lines', async () => {
        const src = fx('fence-blank-lines.md');
        const fence = parseAtoms(src).find(a => a.type === 'fence')!.text;
        const chunk = mkChunk({ content: src });
        // Budget admits title+fence with slack, but not the whole note —
        // forces a re-pack that must keep the fence atomic.
        const budget = fakeCount(`${chunk.title}\n\n${fence}`) + 8;
        expect(fakeCount(embedInput(chunk))).toBeGreaterThan(budget);

        const r = await enforceTokenBudget([chunk], counter, budget);
        expect(r.splits).toBe(1);
        expect(r.overBudget).toBe(0);
        const withFence = r.chunks.filter(c => c.content.includes('```python'));
        expect(withFence).toHaveLength(1);
        const text = withFence[0].content;
        expect(text).toContain('class Pacer');
        expect(text).toContain('def tick');
        // The internal blank lines are still inside one part, between the markers.
        expect(text.slice(text.indexOf('```python'))).toMatch(/\n\s*\n/);
        expect(fenceMarkerLines(text)).toHaveLength(2);
    });

    it('splits an oversize fence at line boundaries, reopening the fence on every piece', async () => {
        const src = fx('oversize-fence.md');
        const fenceBody = parseAtoms(src).find(a => a.type === 'fence')!
            .text.split('\n').slice(1, -1);
        const chunk = mkChunk({ content: src });
        const r = await enforceTokenBudget([chunk], counter, 128);

        expect(r.overBudget).toBe(0);
        for (let i = 0; i < r.chunks.length; i++) {
            expect(fakeCount(embedInput(r.chunks[i]))).toBeLessThanOrEqual(128);
            // Every part is balanced: fences reopened AND closed per piece.
            const markers = fenceMarkerLines(r.chunks[i].content);
            expect(markers.length % 2).toBe(0);
            // Any piece carrying code identifies its language again.
            if (markers.length > 0) expect(markers[0]).toBe('```python');
        }
        // Every original code line survives in exactly one part, in order.
        const joined = r.chunks.map(c => c.content).join('\n');
        for (const line of fenceBody.filter(l => l.trim())) {
            const hits = r.chunks.filter(c => c.content.includes(line));
            expect(hits, `code line lost or duplicated: ${line}`).toHaveLength(1);
        }
        const idx = fenceBody.filter(l => l.trim()).map(l => joined.indexOf(l));
        expect([...idx].sort((a, b) => a - b)).toEqual(idx);
    });

    it('splits an oversize table at row boundaries, repeating header + delimiter', async () => {
        const src = fx('oversize-table.md');
        const tableLines = parseAtoms(src).find(a => a.type === 'table')!.text.split('\n');
        const [header, delim, ...dataRows] = tableLines;
        const chunk = mkChunk({ content: src });
        const r = await enforceTokenBudget([chunk], counter, 96);

        expect(r.overBudget).toBe(0);
        let partsWithRows = 0;
        for (let i = 0; i < r.chunks.length; i++) {
            expect(fakeCount(embedInput(r.chunks[i]))).toBeLessThanOrEqual(96);
            const c = r.chunks[i].content;
            if (dataRows.some(row => c.includes(row))) {
                partsWithRows++;
                // Self-describing: every row-bearing piece restates the header.
                expect(c).toContain(header);
                expect(c).toContain(delim);
            }
        }
        expect(partsWithRows).toBeGreaterThan(1); // it actually split
        // All 14 data rows survive exactly once.
        for (const row of dataRows) {
            expect(r.chunks.filter(c => c.content.includes(row)),
                `row lost or duplicated: ${row}`).toHaveLength(1);
        }
    });

    it('keeps a callout whole through a re-pack — bare ">" lines included', async () => {
        const src = fx('callout-multiline.md');
        const warning = parseAtoms(src).find(a => a.type === 'callout')!.text;
        const chunk = mkChunk({ content: src });
        const budget = fakeCount(`${chunk.title}\n\n${warning}`) + 8;
        const r = await enforceTokenBudget([chunk], counter, budget);

        const withWarning = r.chunks.filter(c => c.content.includes('[!warning]'));
        expect(withWarning).toHaveLength(1);
        // The whole callout — bare ">" continuation and nested quote — is one part.
        expect(withWarning[0].content).toContain(warning);
    });

    it('emits only balanced fences across all closed-fence fixtures at a tight budget', async () => {
        // fence-unterminated.md is excluded: an unterminated fence that needs
        // no split legitimately stays unterminated.
        const fixtures = [
            'fence-blank-lines.md', 'fence-tilde-nested.md', 'mixed-section.md',
            'oversize-fence.md', 'oversize-table.md',
        ];
        for (const name of fixtures) {
            const r = await enforceTokenBudget([mkChunk({ content: fx(name) })], counter, 128);
            for (const c of r.chunks) {
                expect(fenceMarkerLines(c.content).length % 2,
                    `unbalanced fence in a part of ${name}`).toBe(0);
            }
        }
    });
});

describe('dense suffix (frontmatter-into-dense)', () => {
    const SUFFIX = 'food restaurant Austin';

    it('embedInput appends the suffix after title + content', () => {
        const c = mkChunk({ content: 'great patio downtown', denseSuffix: SUFFIX });
        expect(embedInput(c)).toBe(`Note > Section\n\ngreat patio downtown\n\n${SUFFIX}`);
    });

    it('absent suffix leaves embedInput as the bare title\\n\\ncontent', () => {
        const c = mkChunk({ content: 'great patio downtown' });
        expect(embedInput(c)).toBe('Note > Section\n\ngreat patio downtown');
    });

    it('split parts keep the suffix, stay ≤ budget WITH it, and hash it into the id', async () => {
        // A chunk that must split, carrying a note-level suffix. The suffix is
        // reserved in the content budget (overhead), so it survives on EVERY
        // part (the ungated/all-chunks decision) instead of being truncated off
        // the at-cap part — and each part's id must equal the hash of its full
        // embedded bytes, suffix included.
        const content = [
            makeParagraph(20, 'a'), makeParagraph(20, 'b'),
            makeParagraph(20, 'c'), makeParagraph(20, 'd'),
        ].join('\n\n');
        const chunk = mkChunk({ content, denseSuffix: SUFFIX });
        const budget = 64;
        expect(fakeCount(embedInput(chunk))).toBeGreaterThan(budget); // really splits

        const r = await enforceTokenBudget([chunk], counter, budget);
        expect(r.splits).toBe(1);
        expect(r.chunks.length).toBeGreaterThan(1);
        for (const part of r.chunks) {
            // suffix carried + appended last
            expect(part.denseSuffix).toBe(SUFFIX);
            expect(embedInput(part).endsWith(`\n\n${SUFFIX}`)).toBe(true);
            // composed input (suffix included) honors the budget
            expect(fakeCount(embedInput(part))).toBeLessThanOrEqual(budget);
            // chunk_id == hash of the embedded bytes (notePath + embedInput)
            expect(part.chunk_id).toBe(
                chunkIdFor(part.note_path, part.title, part.content, SUFFIX));
        }
        // every original paragraph survived the re-pack (nothing truncated)
        const joined = r.chunks.map(c => c.content).join('\n\n');
        for (const tag of ['a', 'b', 'c', 'd']) {
            expect(joined).toContain(`${tag}word0`);
        }
    });
});
