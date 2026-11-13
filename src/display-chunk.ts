// Display-chunk selection: which chunk of a result note the modal SHOWS
// (snippet, breadcrumb, click-to-scroll target), decoupled from the chunk that
// RANKED the note. Display-only — runs after dedupByPath on the final list and
// never touches score, order, or ranking_signals.
//
// Why: dedupByPath keeps each note's best-FUSED chunk, and fusion is 85% dense.
// For a multi-term query the best-fused chunk is often a lateral match — a
// "Related Pages" link list, a phase table — while a sibling chunk the same
// search already retrieved carries more of the query ("infinity search prd" →
// the PRD's link list, not its Motivation section). The note's rank is right;
// its face is wrong.
//
// Rule (prototyped against the live index 2026-10-12, 6/94 results switched,
// all improvements, no lateral flips):
//   - candidates = the note's chunks among the top DISPLAY_POOL_DEPTH fused
//     chunks (already scored — no extra retrieval);
//   - coverage = distinct query terms hit in the chunk's OWN heading + body
//     (the breadcrumb ancestors and note title are shared by every sibling, so
//     they can't discriminate; the own heading must count or a "Guided
//     Creations" section loses to a passing "no guided creations?" line);
//   - switch ONLY to a chunk with STRICTLY more distinct terms; ties keep the
//     ranking chunk, and among strictly-better chunks the higher fused score
//     wins. Distinct-count, not IDF-weighted: the weighted variant traded one
//     weak match for another (13 switches, several sideways).
// Term matching reuses the snippet's PassageTerm regexes so "hit" means exactly
// what the snippet window will mark.

import type { PassageTerm } from './passage';

// Matches the prototype's candidate depth (verboseTrace's fusedTop50).
export const DISPLAY_POOL_DEPTH = 50;

export interface DisplayCandidate {
    chunk_id: string;
    heading_path: string[];
    content: string;
    score: number;
}

export function distinctTermHits(text: string, terms: PassageTerm[]): number {
    const lower = text.toLowerCase();
    let hits = 0;
    for (const t of terms) {
        t.re.lastIndex = 0;
        if (t.re.test(lower)) hits++;
    }
    return hits;
}

// The chunk's own heading text: the last heading_path segment ('' for the
// pre-first-heading chunk). Split parts share their section's heading_path.
export function ownHeading(c: { heading_path: string[] }): string {
    return c.heading_path.length > 0 ? c.heading_path[c.heading_path.length - 1] : '';
}

// `plain` projects markdown to reader text (snippetPlainText) so a term living
// only inside a URL or link target can't win the switch.
export function pickDisplayChunk<C extends DisplayCandidate>(
    rank: C,
    siblings: C[],
    terms: PassageTerm[],
    plain: (md: string) => string,
): C {
    if (terms.length === 0 || siblings.length === 0) return rank;
    const cover = (c: C): number => distinctTermHits(ownHeading(c) + '\n' + plain(c.content), terms);
    let best = rank;
    let bestHits = cover(rank);
    if (bestHits === terms.length) return rank;
    for (const c of siblings) {
        if (c.chunk_id === rank.chunk_id) continue;
        const h = cover(c);
        if (h > bestHits || (h === bestHits && best !== rank && c.score > best.score)) {
            best = c;
            bestHits = h;
        }
    }
    return best;
}
