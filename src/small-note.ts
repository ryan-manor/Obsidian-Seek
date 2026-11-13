// Small-note collapse (CHUNKER_VERSION 12).
//
// The chunker splits on every H1-H6 regardless of note size, so a note that is
// three headings with one sentence each becomes three chunks. Each one embeds
// `title > heading`, a single sentence and the full keyed suffix; all three land
// the user on the same screen, because the note is shorter than a screen. This
// pass draws such a note back down to ONE chunk.
//
// The rule, deliberately narrow:
//   - only a note the chunker emitted as 2+ chunks (single-chunk notes and
//     `.base` files are untouched),
//   - only when the WHOLE collapsed body counts <= SMALL_NOTE_TOKENS by the
//     model's own tokenizer (the count includes the 2 special tokens, so this is
//     "under 64 real tokens"),
//   - small sections inside a bigger note are left exactly as chunked today.
//
// The collapsed chunk:
//   title         the note title with aliases, no heading breadcrumb
//   heading_path  []  (so title === the peeled base in goForwardDenseParts)
//   content       every section in document order, each preceded by the heading
//                 lines it sat under, as plain text. No heading word is lost: it
//                 moves from the breadcrumb and the BM25 `headings` field into
//                 the body, the same convention the chunker's own carry buffer
//                 uses for sub-min sections ("the heading lives in the chunk
//                 content, not its title").
//
// It runs BEFORE enforceTokenBudget, at the same choke point, because it needs
// the tokenizer and because every producer path (index, sidecar hydrate, live
// re-chunk) must derive the same chunk_ids. Call sites use packChunks below
// rather than enforceTokenBudget directly.
//
// Eval (seek-eval-pack, 2026-11-11): merging at this size is inside the
// harness's resolution on personal/483q in both channels. This ships on the
// product argument above, not on a measured relevance win.
//
// Mirror: seek-eval-pack harness/small_note.py. Parity is pinned by
// src/fixtures/small-note-parity.json, which both sides read.

import type { Chunk } from './types';
import { chunkIdFor } from './chunker';
import { enforceTokenBudget, type CountTokens, type TokenBudgetResult } from './token-budget';

export const SMALL_NOTE_TOKENS = 64;

// Cheap pre-filter so the tokenizer is only asked about plausible candidates.
// 16 chars/token is far above any real text (prose runs ~4-5), so a body longer
// than this cannot be under the token limit. Part of the rule, mirrored in Python.
export const SMALL_NOTE_MAX_CHARS = SMALL_NOTE_TOKENS * 16;

// The collapsed body for one note's chunks, in order. A chunk's own heading is
// not in its `content` (it lives in title/heading_path), so each chunk is
// prefixed with the heading_path entries not already written for an earlier
// chunk. Keyed by path PREFIX, not by text, so two different "Notes" headings
// both survive while a shared parent is written once.
export function collapsedBody(chunks: Chunk[]): string {
    const written = new Set<string>();
    const pieces: string[] = [];
    for (const c of chunks) {
        const heads: string[] = [];
        for (let i = 0; i < c.heading_path.length; i++) {
            const key = c.heading_path.slice(0, i + 1).join('\u0000');
            if (!written.has(key)) {
                written.add(key);
                heads.push(c.heading_path[i]);
            }
        }
        const text = heads.length > 0 ? `${heads.join('\n')}\n${c.content}` : c.content;
        if (text) pieces.push(text);
    }
    return pieces.join('\n\n');
}

// `Base | a1 | a2 > H1 > H2` -> `Base | a1 | a2`. The chunker builds the title
// as titleWithAliases + ' > ' + heading_path.join(' > '), so the tail is exact.
function noteTitle(c: Chunk): string {
    if (c.heading_path.length === 0) return c.title;
    const tail = ` > ${c.heading_path.join(' > ')}`;
    return c.title.endsWith(tail) ? c.title.slice(0, -tail.length) : c.title;
}

function collapse(chunks: Chunk[], content: string): Chunk {
    const first = chunks[0];
    const title = noteTitle(first);
    const linkTerms = chunks.map(c => c.link_terms).filter(Boolean).join(' ');
    return {
        chunk_id: chunkIdFor(first.note_path, title, content, first.denseSuffix),
        title,
        content,
        note_path: first.note_path,
        heading_path: [],
        metadata: first.metadata,
        start_line: Math.min(...chunks.map(c => c.start_line)),
        end_line: Math.max(...chunks.map(c => c.end_line)),
        ...(first.denseSuffix && { denseSuffix: first.denseSuffix }),
        ...(linkTerms && { link_terms: linkTerms }),
    };
}

// Collapse every qualifying note in `chunks` (any number of notes, in chunker
// order). Non-qualifying chunks pass through object-identical and in place.
export async function collapseSmallNotes(chunks: Chunk[], countTokens: CountTokens): Promise<Chunk[]> {
    // Group consecutive chunks by note. The chunker emits a note's chunks
    // contiguously, and every call site passes whole notes.
    const groups: Chunk[][] = [];
    for (const c of chunks) {
        const last = groups[groups.length - 1];
        if (last && last[0].note_path === c.note_path) last.push(c);
        else groups.push([c]);
    }

    const candidates: Array<{ group: number; content: string }> = [];
    for (let g = 0; g < groups.length; g++) {
        const group = groups[g];
        if (group.length < 2 || group[0].note_path.endsWith('.base')) continue;
        const content = collapsedBody(group);
        if (content.length > SMALL_NOTE_MAX_CHARS) continue;
        candidates.push({ group: g, content });
    }
    if (candidates.length === 0) return chunks;

    const counts = await countTokens(candidates.map(c => c.content));
    const replace = new Map<number, Chunk>();
    candidates.forEach((cand, i) => {
        if (counts[i] <= SMALL_NOTE_TOKENS) replace.set(cand.group, collapse(groups[cand.group], cand.content));
    });
    if (replace.size === 0) return chunks;

    const out: Chunk[] = [];
    for (let g = 0; g < groups.length; g++) {
        const merged = replace.get(g);
        if (merged) out.push(merged);
        else out.push(...groups[g]);
    }
    return out;
}

// The producer pipeline after chunkContent: small-note collapse, then the
// 512-token budget. Every path that needs chunk_ids to match what indexing wrote
// goes through this one function.
export async function packChunks(chunks: Chunk[], countTokens: CountTokens): Promise<TokenBudgetResult> {
    return enforceTokenBudget(await collapseSmallNotes(chunks, countTokens), countTokens);
}
