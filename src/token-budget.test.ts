// Invariant tests for WS2.3 token-budget enforcement (token-budget.ts).
//
// The contract under test, in priority order:
//   1. BUDGET: every emitted chunk's composed embed input counts ≤ budget
//      (exception: the unsplittable oversize-title pathology, which must be
//      counted in overBudget — never silent).
//   2. NO CONTENT LOSS: every original unit (paragraph block) survives
//      re-packing, in order; hard-split loses only whitespace. Within-section
//      overlap may REPEAT a seam paragraph as the head of the next part, so the
//      invariant is "original units are an ordered subsequence of the emitted
//      stream, and the emitted stream introduces no foreign units" — not exact
//      stream equality.
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
import {
    enforceTokenBudget, embedInput, overlapSeed, collapsePadding,
    PAD_RUN_MIN, MAX_COLLAPSED_CHARS_PER_TOKEN, TOKEN_BUDGET, type CountTokens,
} from './token-budget';
import { chunkIdFor } from './chunker';
import { parseAtoms, type Atom } from './atoms';

// Original units must appear, in order, somewhere in the emitted stream —
// overlap may interleave duplicate seam paragraphs, so this is a subsequence
// check, not equality.
const isSubsequence = (sub: string[], sup: string[]) => {
    let i = 0;
    for (const x of sup) if (i < sub.length && x === sub[i]) i++;
    return i === sub.length;
};

const META: ChunkMetadata = {
    tags: [], aliases: [], created: null, modified: null, properties: {},
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
        // No content loss, no reordering, no foreign units (overlap may repeat
        // a seam paragraph, so subsequence + same-set, not exact equality).
        const flat = r.chunks.flatMap(c => paras(c.content));
        expect(isSubsequence(paras(body), flat)).toBe(true);
        expect(new Set(flat)).toEqual(new Set(paras(body)));
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
        const flat = r.chunks.flatMap(c => paras(c.content));
        expect(isSubsequence(paras(body), flat)).toBe(true);
        expect(new Set(flat)).toEqual(new Set(paras(body)));
        expect(r.overBudget).toBe(0);
    });

    it('handles the empty chunk list without a counter call', async () => {
        const r = await enforceTokenBudget([], async () => { throw new Error('must not be called'); }, 512);
        expect(r).toEqual({ chunks: [], counts: [], splits: 0, overBudget: 0 });
    });
});

describe('within-section overlap', () => {
    it('seeds each later part with the trailing paragraph of the previous part', async () => {
        // 30-word paragraphs (~60 tok each) — one fits the ~76-tok overlap cap,
        // two do not, so exactly one paragraph carries across every seam.
        const body = Array.from({ length: 12 }, (_, i) => makeParagraph(30, `p${i}`)).join('\n\n');
        const chunk = mkChunk({ content: body });
        const r = await enforceTokenBudget([chunk], counter, 512);

        expect(r.chunks.length).toBeGreaterThan(1);
        for (let k = 1; k < r.chunks.length; k++) {
            const prev = paras(r.chunks[k - 1].content);
            const here = paras(r.chunks[k].content);
            expect(here[0]).toBe(prev[prev.length - 1]); // seam paragraph repeated
        }
        // Every part still fits WITH its seed.
        for (const c of r.chunks) expect(fakeCount(embedInput(c))).toBeLessThanOrEqual(512);
        // Overlap makes parts share content, but ids stay distinct (fresh atoms differ).
        expect(new Set(r.chunks.map(c => c.chunk_id)).size).toBe(r.chunks.length);
    });

    it('overlapSeed: paragraph-only, newest-first, bounded by the limit', () => {
        const A: Atom = { type: 'paragraph', text: 'aaa' };
        const B: Atom = { type: 'paragraph', text: 'bbb' };
        const tbl: Atom = { type: 'table', text: '|h|\n|-|\n|r|' };

        // Both paragraphs fit → both, restored to document order.
        expect(overlapSeed([A, B], [1, 1], 2).atoms).toEqual([A, B]);
        // Limit fits one → newest only.
        expect(overlapSeed([A, B], [1, 1], 1).atoms).toEqual([B]);
        // No room (≤ 0) → nothing seeds.
        expect(overlapSeed([A, B], [1, 1], 0).atoms).toEqual([]);
        expect(overlapSeed([A, B], [1, 1], -5).atoms).toEqual([]);
        // A trailing non-paragraph atom is never duplicated (no table-header re-emit).
        expect(overlapSeed([A, tbl], [1, 1], 9).atoms).toEqual([]);
        // …and it halts the walk even when earlier atoms are paragraphs.
        expect(overlapSeed([A, tbl, B], [1, 1, 1], 9).atoms).toEqual([B]);
    });

    it('does not duplicate a table atom across the seam it falls on', async () => {
        // A table sized to land at a part boundary; its rows must appear once.
        const rows = Array.from({ length: 8 }, (_, i) => `| r${i}a | r${i}b |`).join('\n');
        const table = `| h1 | h2 |\n| --- | --- |\n${rows}`;
        const body = [makeParagraph(60, 'lead'), table, makeParagraph(60, 'tail0'),
            makeParagraph(60, 'tail1'), makeParagraph(60, 'tail2')].join('\n\n');
        const chunk = mkChunk({ content: body });
        const r = await enforceTokenBudget([chunk], counter, 200);

        expect(r.chunks.length).toBeGreaterThan(1);
        const tableParts = r.chunks.filter(c => c.content.includes('| h1 | h2 |'));
        expect(tableParts.length).toBe(1); // the table atom is not seeded forward
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

// Issue-#4 defenses (2026-07-06): padding collapse + count gate. The failure
// class under test: multi-MB whitespace-padded cells took the real tokenizer
// to ~1.9 GB peak per call and rode an unsplittable table atom into a single
// 51K-token overBudget chunk retried every pass. The contract:
//   1. The raw counter NEVER sees a text over the gate, nor one still
//      containing a PAD_RUN_MIN whitespace run.
//   2. Stored content is untouched — collapse is dense-channel (counts +
//      embedInput) only.
//   3. Space-less oversize blobs (the JWT/base64 class) still split, under
//      the same gate, with exact coverage.
describe('padding collapse + count gate (issue #4)', () => {
    // A counter spy that enforces contract 1 on every call and delegates to
    // the standard fake.
    function spyCounter(budget: number) {
        const gate = budget * MAX_COLLAPSED_CHARS_PER_TOKEN;
        const seen = { maxLen: 0, calls: 0 };
        const padRun = new RegExp(`[ \\t]{${PAD_RUN_MIN},}`);
        const count: CountTokens = async texts => {
            for (const t of texts) {
                seen.calls++;
                seen.maxLen = Math.max(seen.maxLen, t.length);
                expect(t.length).toBeLessThanOrEqual(gate);
                expect(padRun.test(t)).toBe(false);
            }
            return texts.map(fakeCount);
        };
        return { count, seen };
    }

    it('collapsePadding: runs under PAD_RUN_MIN survive, longer collapse, newlines untouched', () => {
        const under = `a${' '.repeat(PAD_RUN_MIN - 1)}b`;
        expect(collapsePadding(under)).toBe(under);
        expect(collapsePadding(`a${' '.repeat(PAD_RUN_MIN)}b`)).toBe('a b');
        expect(collapsePadding(`a${'\t'.repeat(PAD_RUN_MIN)}b`)).toBe('a b');
        expect(collapsePadding(`a${' \t'.repeat(PAD_RUN_MIN)}b`)).toBe('a b');
        // horizontal only: the padded line collapses, the line structure stays
        const multi = `a\n${' '.repeat(40)}\nb`;
        expect(collapsePadding(multi)).toBe('a\n \nb');
    });

    it('embedInput collapses padding and caps at the lossless char bound', () => {
        const padded = mkChunk({ content: `x${' '.repeat(100)}y` });
        expect(embedInput(padded)).toBe(`${padded.title}\n\nx y`);
        const huge = mkChunk({ content: 'y'.repeat(40_000) });
        expect(embedInput(huge).length).toBe(TOKEN_BUDGET * MAX_COLLAPSED_CHARS_PER_TOKEN);
    });

    it('poison table (whitespace-padded cells) passes through as one small-count chunk', async () => {
        // Scaled issue-#4 shape: header/delimiter rows padded so wide that the
        // pre-fix splitTableAtom header-bail would have applied, two rows an
        // order of magnitude wider still.
        const pad = (n: number) => ' '.repeat(n);
        const lines = [
            `| Title${pad(300)} | Location${pad(300)} |`,
            `| ---${pad(300)} | ---${pad(300)} |`,
        ];
        for (let i = 0; i < 2; i++) lines.push(`| Job ${i}${pad(5000)} | Remote${pad(5000)} |`);
        for (let i = 0; i < 17; i++) lines.push(`| Role ${i}${pad(300)} | Office ${i}${pad(300)} |`);
        const content = lines.join('\n');
        const chunk = mkChunk({ content });

        const { count, seen } = spyCounter(512);
        const r = await enforceTokenBudget([chunk], count, 512);

        // Collapsed, the real content is tiny: one chunk, no splits, in budget.
        expect(r.chunks.length).toBe(1);
        expect(r.splits).toBe(0);
        expect(r.overBudget).toBe(0);
        expect(r.counts[0]).toBeLessThanOrEqual(512);
        // Untouched chunks pass through object-identical; stored content keeps
        // its raw padding (display/BM25 side unaffected by the dense collapse).
        expect(r.chunks[0]).toBe(chunk);
        expect(r.chunks[0].content).toBe(content);
        expect(seen.calls).toBeGreaterThan(0);
    });

    it('space-less blob over the gate splits under the gate with exact coverage', async () => {
        const budget = 64; // gate = 4096 — small enough to exercise estimates
        const blob = 'x'.repeat(20_000);
        const chunk = mkChunk({ content: blob });

        const { count, seen } = spyCounter(budget);
        const r = await enforceTokenBudget([chunk], count, budget);

        expect(r.splits).toBe(1);
        expect(r.overBudget).toBe(0);
        expect(r.chunks.length).toBeGreaterThan(1);
        for (let i = 0; i < r.chunks.length; i++) {
            expect(fakeCount(embedInput(r.chunks[i]))).toBeLessThanOrEqual(budget);
            expect(r.counts[i]).toBeLessThanOrEqual(budget);
        }
        // 'x' has no whitespace for the cut-trim to eat and hard-split pieces
        // are too big for overlap seeding, so coverage is byte-exact.
        expect(r.chunks.map(c => c.content).join('')).toBe(blob);
        // The gate did real work: the 20K-char blob itself was never tokenized.
        expect(seen.maxLen).toBeLessThanOrEqual(budget * MAX_COLLAPSED_CHARS_PER_TOKEN);
    });

    it('padded table over budget still splits at row boundaries, header repeated', async () => {
        const pad = (n: number) => ' '.repeat(n);
        const header = ['| A | B |', '| --- | --- |'];
        const rows = Array.from({ length: 200 }, (_, i) =>
            `| item${i}${pad(50)} | value${i}${pad(50)} |`);
        const content = [...header, ...rows].join('\n');
        const chunk = mkChunk({ content });
        expect(fakeCount(collapsePadding(embedInput(chunk)))).toBeGreaterThan(512);

        const { count } = spyCounter(512);
        const r = await enforceTokenBudget([chunk], count, 512);

        expect(r.chunks.length).toBeGreaterThan(1);
        expect(r.overBudget).toBe(0);
        for (const part of r.chunks) {
            // each piece is a valid, self-describing table with RAW padding
            expect(part.content.startsWith(`${header[0]}\n${header[1]}\n`)).toBe(true);
            expect(part.content).toContain(pad(50));
        }
        // every row survives, in order, exactly once (tables never overlap-seed)
        const emitted = r.chunks.flatMap(c => c.content.split('\n').slice(2));
        expect(emitted).toEqual(rows);
    });
});
