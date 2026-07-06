// WS2.3 token-budget enforcement — the index-side correctness invariant.
//
// The dense channel embeds `title\n\ncontent` through a seq ladder capped at
// 512 tokens (SEQ_BUCKETS, iframe-runner.ts) with truncation:true. Anything
// past the truncation point is invisible to dense retrieval — measured at
// 21-26% of corpus tokens (2026-06-10, eval pack `95c08b8`), masked by the
// BM25 full-text backstop rather than absent. Two distinct defects produced
// that number, and this module + token-exact bucket routing fix one each:
//
//   1. AT-CAP loss: the chunker emits up to 6750-char (~1500-token) chunks;
//      everything past 512 actual tokens is cut no matter the bucket. Fixed
//      HERE: enforceTokenBudget re-splits any chunk whose composed embed
//      input exceeds the budget, packing paragraph units under it.
//   2. BELOW-CAP loss: chars/4.5 under-buckets dense text (code/URLs run
//      ~3.6 chars/token). Fixed in search.ts: route by exact token counts
//      (token-counts RPC) via selectIndexBucket, never the char estimate.
//
// Everything is verified against the model's own tokenizer (countTokens is
// the iframe's token-counts RPC) — estimates are only used to make the greedy
// pack cheap; every emitted part's COMPOSED input is recounted exactly. BPE
// counts aren't additive across joins, so an estimator-only version of this
// module would just rebuild the bug it deletes.
//
// Relevance posture (decision of record, 2026-06-10): this is NOT an nDCG
// play. ~512-token chunks measured flat vs shipped on the full system
// (small-chunk arm: −0.0002 personal / 0.0000 dnd / −0.0014 code), so the
// re-chunking is pre-validated relevance-safe; the gate is the invariant
// itself — dense-invisible share → 0 — checked in the eval-pack harness.
//
// Within-section overlap (2026-06-23): seam insurance for the long-section
// tail — when a super-section splits, each later part carries the prior part's
// trailing paragraph(s) so a straddling query meets a chunk with both sides.
// See OVERLAP_FRACTION / overlapSeed. Targeted, paragraph-only, capped; the
// straddle-query eval probe (gold chunk in a non-first part) is the gate.
//
// WS3 (2026-06-10): the unit boundary is the ATOM (atoms.ts) — paragraphs at
// blank lines, fences/tables/callouts whole. This module is also now the ONLY
// splitter in the pipeline (the chunker's char-based splitOversized is gone),
// so structural atomicity is enforced here: an atom over the content budget
// splits structure-aware — fences re-split at line boundaries with the fence
// REOPENED on every piece, tables repeat their header+delimiter rows, callouts
// split at ">"-line boundaries, paragraphs hard-split as before.
//
// Padding collapse + count gate (issue #4, 2026-07-06): a 5.9 MB pipe table
// whose cells were space-padded into ~2M-char lines cost the tokenizer a
// measured 1.86 GB peak / 9.3 s PER CALL (unigram lattice; padding runs
// ~127 chars/token vs ~4.5 prose, ~9.4 base64) — a renderer OOM on desktop
// (surfacing as ORT-wasm's 32-bit SafeInt "Tensor shape is too large" when
// the arena loses the race instead of V8), and far past the ~1.5 GB iOS
// jetsam line. Worse, splitTableAtom's header-bail meant the whole file was
// emitted as ONE 51K-token overBudget chunk, re-tokenized whole on every
// indexing pass — an eternal-retry churn. Two defenses, both DENSE-CHANNEL
// ONLY (stored chunk.content, chunk_ids, BM25 and snippets never change —
// code-block indentation must survive for display):
//   1. collapsePadding: horizontal-whitespace runs ≥ PAD_RUN_MIN collapse to
//      one space in every text this module counts and in embedInput itself.
//      Runs that long are alignment padding, never content — the issue-#4
//      file's real content is ~460 chars and embeds in 18 ms once collapsed.
//   2. The count gate (gatedCounter): after collapse no text packs more than
//      MAX_COLLAPSED_CHARS_PER_TOKEN chars into a token (see the constant's
//      bound argument), so a text longer than budget × that bound is over
//      budget BY CONSTRUCTION — answer with a char estimate instead of
//      tokenizing. The tokenizer never sees an input past the gate; the
//      estimate only steers oversize-split proportions, and every EMITTED
//      part is still exact-recounted by the verify loop (under the gate by
//      then, since its count must land ≤ budget to be emitted).

import type { Chunk } from './types';
import { chunkIdFor } from './chunker';
import { parseAtoms, type Atom } from './atoms';

// Batch token counter — exact counts, specials included, same tokenizer the
// forward pass uses. In production this is LocalEmbedder.tokenCounts (iframe
// RPC); tests inject a deterministic fake.
export type CountTokens = (texts: string[]) => Promise<number[]>;

// The dense embed window: SEQ_BUCKETS' last rung. Mirrored by convention
// (importing iframe-runner here would drag the template string into tests).
export const TOKEN_BUDGET = 512;

// Horizontal-whitespace runs at least this long collapse to a single space in
// the dense channel (embedInput + every gatedCounter text). 16 is past any
// legitimate code indentation level that carries dense signal and far past
// prose; runs of 15 and under pass through untouched, so normal chunks'
// embed inputs — and therefore their vectors and counts — are byte-identical
// to the pre-collapse pipeline (no CHUNKER_VERSION bump, no fleet reindex).
export const PAD_RUN_MIN = 16;
const PAD_RUN_RE = new RegExp(`[ \\t]{${PAD_RUN_MIN},}`, 'g');

// Dense-channel padding collapse. Horizontal only — newlines are never
// touched, so line structure (atom boundaries, fence/table shapes) survives.
export function collapsePadding(text: string): string {
    return text.replace(PAD_RUN_RE, ' ');
}

// Upper bound on chars-per-token for COLLAPSED text, used by the count gate
// and the embedInput char cap. The only way past ~10 chars/token with this
// tokenizer is whitespace runs (measured: padding ~127, base64 ~9.4, prose
// ~4.5); post-collapse the longest run is PAD_RUN_MIN−1 chars, and no vocab
// piece approaches 64 chars — so `len > budget × 64 ⇒ count > budget` holds
// by construction, and the first budget-many tokens of any collapsed text
// lie inside its first budget × 64 chars (what makes the embedInput cap
// lossless for the truncation window).
export const MAX_COLLAPSED_CHARS_PER_TOKEN = 64;

// Estimator for gated (over-the-gate) texts only: same 4.5 chars/token the
// bucket estimate uses (selectBucket, iframe-runner.ts). It OVERSTATES token
// counts for the sparse text that gets gated, which is the safe direction —
// hardSplitByTokens cuts proportionally smaller heads, and the verify loop
// exact-recounts every emitted part anyway.
const CHARS_PER_TOKEN_EST = 4.5;

// Wraps the raw counter so that (a) every text is collapsed before the
// tokenizer sees it and (b) texts still over the gate after collapse are
// answered with a char estimate instead of being tokenized — the estimate is
// provably > budget (gate = budget × MAX_COLLAPSED_CHARS_PER_TOKEN and
// 64 > 4.5), so callers make the same over/under decision they would with an
// exact count. This is the single choke point: enforceTokenBudget wraps once
// and every internal countTokens call (units, header, verify, packLines,
// hardSplitByTokens) inherits both protections.
function gatedCounter(countTokens: CountTokens, budget: number): CountTokens {
    const gate = budget * MAX_COLLAPSED_CHARS_PER_TOKEN;
    return async (texts: string[]): Promise<number[]> => {
        const out = new Array<number>(texts.length);
        const passIdx: number[] = [];
        const pass: string[] = [];
        for (let i = 0; i < texts.length; i++) {
            const t = collapsePadding(texts[i]);
            if (t.length > gate) {
                out[i] = Math.max(budget + 1, Math.ceil(t.length / CHARS_PER_TOKEN_EST));
            } else {
                passIdx.push(i);
                pass.push(t);
            }
        }
        if (pass.length) {
            const counts = await countTokens(pass);
            for (let j = 0; j < pass.length; j++) out[passIdx[j]] = counts[j];
        }
        return out;
    };
}

// Below this content budget (title nearly fills the window) packing would
// emit hundred-sliver chunks; treat the title as pathological instead.
const MIN_CONTENT_BUDGET = 32;

// Within-section overlap (2026-06-23). When a single heading section is too big
// for the window and splits into multiple parts, each later part is seeded with
// up to this fraction of the content budget worth of the PREVIOUS part's
// trailing paragraphs — boundary insurance so a query whose answer straddles
// the cut still meets one chunk holding both sides. This is the ONLY overlap in
// the pipeline and it lives entirely inside splitChunk, so it is structurally
// confined to over-budget sections: under-budget sections pass through
// enforceTokenBudget untouched and never overlap with their neighbours across a
// heading boundary. The deleted WS3 char-splitter overlap (chunker.ts header)
// ran in series and duplicated across arbitrary cuts — this one is
// paragraph-only and capped (overlapSeed), so the old footgun can't recur. 0
// disables it.
const OVERLAP_FRACTION = 0.15;
// Token headroom kept between the seed + first fresh atom and the budget. Under
// realistic (near-additive) BPE join costs this keeps the exact recount from
// tipping a seeded head over budget, so the verify loop won't demote it. A
// pathological tokenizer with a huge per-`\n\n` premium (≫ the granite BPE's
// 1-2 tokens) could still force a demote that emits a seed-only part — harmless
// index redundancy, never a budget breach or content loss (the budget invariant
// is enforced unconditionally by the verify loop downstream).
const OVERLAP_JOIN_SLACK = 4;

// THE embed-input composition. search.ts routing and the doc side of the
// query path both assume this exact `title\n\ncontent` shape — compose it in
// one place so the counted text and the embedded text can never drift.
export function embedInput(c: Chunk): string {
    const base = `${c.title}\n\n${c.content}`;
    // Note-level frontmatter values, dense-channel only (chunker.ts
    // buildDenseSuffix). Appended LAST so it folds into chunk_id identically
    // (chunkIdFor's tail) and is what splitChunk reserves budget for. Absent on
    // notes with no qualifying values → the bare `title\n\ncontent` as before.
    const full = c.denseSuffix ? `${base}\n\n${c.denseSuffix}` : base;
    // Collapse + hard char cap (issue #4 defenses, header comment). The cap is
    // lossless for the dense channel: post-collapse, 512 tokens always lie
    // inside the first TOKEN_BUDGET × MAX_COLLAPSED_CHARS_PER_TOKEN chars, so
    // only content already past the truncation point can be cut. It is only
    // reachable for overBudget escapes — any part the verify loop emitted
    // ≤ budget is under the cap, so its counted and embedded bytes stay
    // identical. Carry-over keys (search.ts F13) call this same function on
    // both the harvest and lookup sides within one build, so keys agree.
    return collapsePadding(full).slice(0, TOKEN_BUDGET * MAX_COLLAPSED_CHARS_PER_TOKEN);
}

export interface TokenBudgetResult {
    chunks: Chunk[];
    // Exact token count of each chunk's embed input, parallel to `chunks` —
    // search.ts routes each chunk into its seq bucket with these.
    counts: number[];
    // Input chunks that had to be re-split (telemetry).
    splits: number;
    // Outputs still over budget — only reachable via a title that (nearly)
    // fills the window by itself, which cannot be split. These truncate at
    // the cap and are the only permitted nonzero in the harness gate.
    overBudget: number;
}

export async function enforceTokenBudget(
    chunks: Chunk[],
    countTokens: CountTokens,
    budget = TOKEN_BUDGET,
): Promise<TokenBudgetResult> {
    // Single choke point for the issue-#4 defenses: every count below — here
    // and throughout splitChunk — goes through the collapsed, gated counter.
    countTokens = gatedCounter(countTokens, budget);
    const counts = chunks.length ? await countTokens(chunks.map(embedInput)) : [];
    const outChunks: Chunk[] = [];
    const outCounts: number[] = [];
    let splits = 0;
    let overBudget = 0;

    for (let i = 0; i < chunks.length; i++) {
        if (counts[i] <= budget) {
            outChunks.push(chunks[i]);
            outCounts.push(counts[i]);
            continue;
        }
        splits++;
        const r = await splitChunk(chunks[i], counts[i], countTokens, budget);
        overBudget += r.overBudget;
        outChunks.push(...r.parts);
        outCounts.push(...r.counts);
    }
    return { chunks: outChunks, counts: outCounts, splits, overBudget };
}

interface SplitResult {
    parts: Chunk[];
    counts: number[];
    overBudget: number;
}

async function splitChunk(
    chunk: Chunk,
    originalCount: number,
    countTokens: CountTokens,
    budget: number,
): Promise<SplitResult> {
    // Title overhead: tokens the title contributes to EVERY part's input.
    // Counted on `title\n\n` alone; the ±1-2-token BPE boundary slack vs the
    // composed input is absorbed by the verify loop below.
    const [titleCount] = await countTokens([`${chunk.title}\n\n`]);
    // The dense suffix (note-level frontmatter) is appended to EVERY part's embed
    // input (embedInput), so reserve it in the content budget too. Without this a
    // part packed up to the cap would have its suffix truncated off (it's last),
    // silently dropping the injection on rich/multi-chunk notes — but the ship
    // decision is ungated/all-chunks, so the suffix must survive on every part.
    // Counted on the bare `\n\n${suffix}` tail; the ±1-token BPE join slack is
    // absorbed by the verify loop below (which recounts the full composed input).
    const suffixTail = chunk.denseSuffix ? `\n\n${chunk.denseSuffix}` : '';
    const [suffixCount] = suffixTail ? await countTokens([suffixTail]) : [0];
    const overhead = titleCount + suffixCount;
    const contentBudget = budget - overhead;
    if (contentBudget < MIN_CONTENT_BUDGET) {
        // A title can't be split — emit unchanged, let the cap truncate, and
        // surface it (telemetry + harness gate). In practice zero: it takes a
        // ~500-token alias pile-up to get here.
        return { parts: [chunk], counts: [originalCount], overBudget: 1 };
    }

    // Units = atoms (WS3): paragraphs at blank lines; fences, tables and
    // callouts WHOLE. Any single atom over the content budget splits
    // structure-aware (splitAtom) before packing.
    let units: Atom[] = parseAtoms(chunk.content);
    {
        const unitCounts = await countTokens(units.map(u => u.text));
        const expanded: Atom[] = [];
        for (let i = 0; i < units.length; i++) {
            if (unitCounts[i] <= contentBudget) expanded.push(units[i]);
            else expanded.push(...await splitAtom(units[i], countTokens, contentBudget));
        }
        units = expanded;
    }

    // Greedy pack on per-unit counts. Each standalone count pays the special
    // tokens (~2) that the composed input pays only once, and the '\n\n'
    // joiner is ~1 token — so summing unit counts OVERESTIMATES the composed
    // count and the pack is conservative by construction; the verify loop
    // almost never demotes.
    const unitCounts = await countTokens(units.map(u => u.text));
    const overlapCap = Math.round(contentBudget * OVERLAP_FRACTION);
    const groups: Atom[][] = [];
    let cur: Atom[] = [];
    let curCounts: number[] = [];
    let curSum = 0;
    for (let i = 0; i < units.length; i++) {
        if (cur.length && overhead + curSum + unitCounts[i] > budget) {
            groups.push(cur);
            // Within-section overlap: seed the next group with the trailing
            // paragraph atoms of the one just closed. Bounded by overlapCap AND
            // by the room left beside the atom that triggered the break (minus a
            // join-slack margin), so under realistic BPE join costs seed + first
            // fresh atom stays under budget and the verify loop won't demote it.
            // A negative room (the triggering atom already (nearly) fills the
            // window) makes the limit ≤ 0 → overlapSeed returns empty → no
            // overlap. The seed atoms are duplicated (still owned by the closed
            // group); every original atom keeps exactly one "home" group and `i`
            // advances each iteration, so coverage and termination are unchanged.
            const room = budget - overhead - unitCounts[i] - OVERLAP_JOIN_SLACK;
            const seed = overlapSeed(cur, curCounts, Math.min(overlapCap, room));
            cur = seed.atoms;
            curCounts = seed.counts;
            curSum = seed.sum;
        }
        cur.push(units[i]);
        curCounts.push(unitCounts[i]);
        curSum += unitCounts[i];
    }
    if (cur.length) groups.push(cur);

    // Verify: recount each group's COMPOSED input exactly. Over budget with
    // >1 unit → demote the last unit into its own group (strictly smaller →
    // terminates). Over budget with 1 unit (estimator slack at the boundary)
    // → structure-aware split, tighter; if it can't shrink, accept + count
    // overBudget.
    const finalGroups: Atom[][] = [];
    const finalCounts: number[] = [];
    let overBudget = 0;
    for (const g of groups) {
        const work: Atom[][] = [g];
        while (work.length) {
            const grp = work.shift()!;
            const [c] = await countTokens([`${chunk.title}\n\n${joinAtoms(grp)}${suffixTail}`]);
            if (c <= budget) {
                finalGroups.push(grp);
                finalCounts.push(c);
                continue;
            }
            if (grp.length > 1) {
                work.unshift(grp.slice(0, -1), grp.slice(-1));
            } else {
                const pieces = await splitAtom(
                    grp[0], countTokens, Math.max(MIN_CONTENT_BUDGET, contentBudget - 8));
                if (pieces.length === 1) {
                    finalGroups.push(grp);
                    finalCounts.push(c);
                    overBudget++;
                } else {
                    work.unshift(...pieces.map(p => [p]));
                }
            }
        }
    }

    // Materialize parts. Same title (embedded/BM25-boosted/hashed), fresh
    // path-salted chunk_id per distinct content, display-only part numbering
    // — never in `title`. Line attribution stays the parent's range.
    const displayBase = chunk.displayTitle ?? chunk.title;
    const parts = finalGroups.map((g, gi): Chunk => {
        const content = joinAtoms(g);
        const part: Chunk = {
            ...chunk,
            content,
            // `...chunk` already carried denseSuffix onto the part, so embedInput
            // appends it — the id must include it too (4th arg) to stay == bytes.
            chunk_id: chunkIdFor(chunk.note_path, chunk.title, content, chunk.denseSuffix),
        };
        if (finalGroups.length > 1) part.displayTitle = `${displayBase} (part ${gi + 1})`;
        return part;
    });
    return { parts, counts: finalCounts, overBudget };
}

function joinAtoms(atoms: Atom[]): string {
    return atoms.map(a => a.text).join('\n\n');
}

// Trailing PARAGRAPH atoms of a just-closed group, newest-first up to `limit`
// tokens, returned in document order so the seeded group reads forward. Stops
// at the first non-paragraph atom — a table header, reopened fence or callout
// marker must never be duplicated into the next part — and at the first atom
// that would breach `limit`. `limit` ≤ 0 → empty (no room beside the breaking
// atom). A hard-split blob piece is type 'paragraph' but ~budget-sized, so the
// limit, not the type guard, keeps it out. Exported for direct guard tests.
export function overlapSeed(
    group: Atom[],
    counts: number[],
    limit: number,
): { atoms: Atom[]; counts: number[]; sum: number } {
    const atoms: Atom[] = [];
    const outCounts: number[] = [];
    let sum = 0;
    for (let k = group.length - 1; k >= 0; k--) {
        if (group[k].type !== 'paragraph') break;
        if (sum + counts[k] > limit) break;
        atoms.unshift(group[k]);
        outCounts.unshift(counts[k]);
        sum += counts[k];
    }
    return { atoms, counts: outCounts, sum };
}

// Structure-aware oversize split — the WS3 invariant's enforcement point.
// Returns ≥1 atoms; a return of exactly the input (length 1, same text)
// means "cannot shrink" and the caller accepts it as overBudget. Termination
// for every type: pieces strictly shrink (fewer lines / shorter text) or the
// single-piece return ends the recursion.
async function splitAtom(
    atom: Atom,
    countTokens: CountTokens,
    maxTokens: number,
): Promise<Atom[]> {
    switch (atom.type) {
        case 'fence': return splitFenceAtom(atom, countTokens, maxTokens);
        case 'table': return splitTableAtom(atom, countTokens, maxTokens);
        case 'callout': return splitLinesAtom(atom, countTokens, maxTokens);
        default: {
            const pieces = await hardSplitByTokens(atom.text, countTokens, maxTokens);
            return pieces.map((text): Atom => ({ type: 'paragraph', text }));
        }
    }
}

// Fence-reopen split (decision of record, plan §WS3): an oversize fence
// splits at LINE boundaries only, and every piece is re-wrapped — the
// original opening line (marker + info string, so syntax highlighting and
// language identity survive) plus a bare close marker. An unterminated
// fence's pieces all get closed; that normalization is the price of every
// piece being a valid, renderable fence.
async function splitFenceAtom(
    atom: Atom,
    countTokens: CountTokens,
    maxTokens: number,
): Promise<Atom[]> {
    const lines = atom.text.split('\n');
    const opening = lines[0];
    const marker = /^ {0,3}(`{3,}|~{3,})/.exec(opening)![1];
    const closeRe = /^ {0,3}(`{3,}|~{3,})[ \t]*$/;
    const lastM = lines.length > 1 ? closeRe.exec(lines[lines.length - 1]) : null;
    const hasClose = lastM !== null && lastM[1][0] === marker[0] && lastM[1].length >= marker.length;
    const body = lines.slice(1, hasClose ? -1 : undefined);
    if (body.length <= 1) return [atom]; // nothing to split at line granularity

    const [wrapperCount] = await countTokens([`${opening}\n${marker}`]);
    const lineBudget = maxTokens - wrapperCount;
    if (lineBudget < 8) return [atom]; // pathological wrapper; accept as-is

    const groups = await packLines(body, countTokens, lineBudget);
    if (groups.length <= 1) return [atom];
    return groups.map((g): Atom => ({
        type: 'fence',
        text: `${opening}\n${g.join('\n')}\n${marker}`,
    }));
}

// Table split: pieces at ROW boundaries, header + delimiter rows repeated on
// every piece so each stays a valid, self-describing GFM table.
async function splitTableAtom(
    atom: Atom,
    countTokens: CountTokens,
    maxTokens: number,
): Promise<Atom[]> {
    const lines = atom.text.split('\n');
    if (lines.length <= 3) return [atom]; // header + delim + ≤1 row
    const header = lines.slice(0, 2);
    const rows = lines.slice(2);

    const [headerCount] = await countTokens([header.join('\n')]);
    const rowBudget = maxTokens - headerCount;
    if (rowBudget < 8) return [atom];

    const groups = await packLines(rows, countTokens, rowBudget);
    if (groups.length <= 1) return [atom];
    return groups.map((g): Atom => ({
        type: 'table',
        text: [...header, ...g].join('\n'),
    }));
}

// Callout/blockquote split: line-boundary pieces, no wrapper. Each piece is
// still a run of ">" lines (a valid blockquote); only the first carries the
// [!type] marker — acceptable, the type line is presentation not content.
async function splitLinesAtom(
    atom: Atom,
    countTokens: CountTokens,
    maxTokens: number,
): Promise<Atom[]> {
    const lines = atom.text.split('\n');
    if (lines.length <= 1) return [atom];
    const groups = await packLines(lines, countTokens, maxTokens);
    if (groups.length <= 1) return [atom];
    return groups.map((g): Atom => ({ type: atom.type, text: g.join('\n') }));
}

// Greedy line packer shared by the structural splitters. A single line over
// the budget goes through hardSplitByTokens (degenerate but bounded — the
// JWT-blob class); its pieces ride as pseudo-lines.
async function packLines(
    lines: string[],
    countTokens: CountTokens,
    budget: number,
): Promise<string[][]> {
    let work = lines;
    {
        const counts = await countTokens(work);
        const expanded: string[] = [];
        for (let i = 0; i < work.length; i++) {
            if (counts[i] <= budget) expanded.push(work[i]);
            else expanded.push(...await hardSplitByTokens(work[i], countTokens, budget));
        }
        work = expanded;
    }
    const counts = await countTokens(work);
    const groups: string[][] = [];
    let cur: string[] = [];
    let curSum = 0;
    for (let i = 0; i < work.length; i++) {
        if (cur.length && curSum + counts[i] > budget) {
            groups.push(cur);
            cur = [];
            curSum = 0;
        }
        cur.push(work[i]);
        curSum += counts[i];
    }
    if (cur.length) groups.push(cur);
    return groups;
}

// Token-proportional char bisection, recount-verified. Splits at the last
// space before the proportional cut so words stay intact; falls back to a
// mid-cut on space-less text (the JWT-blob class). Every split strictly
// shrinks the piece, so it terminates on any input; a piece that cannot be
// shrunk further is returned as-is (caller decides whether that's fatal).
async function hardSplitByTokens(
    text: string,
    countTokens: CountTokens,
    maxTokens: number,
): Promise<string[]> {
    const out: string[] = [];
    const work: string[] = [text];
    let guard = text.length + 1;
    while (work.length) {
        if (guard-- <= 0) {
            out.push(...work);
            break;
        }
        const t = work.shift()!;
        const [c] = await countTokens([t]);
        if (c <= maxTokens || t.length <= 1) {
            out.push(t);
            continue;
        }
        let at = Math.max(1, Math.floor((t.length * maxTokens / c) * 0.9));
        const sp = t.lastIndexOf(' ', at);
        if (sp > 0) at = sp;
        at = Math.min(at, t.length - 1);
        const head = t.slice(0, at).trim();
        const tail = t.slice(at).trim();
        if (!head || !tail) {
            out.push(t); // degenerate (whitespace-only side) — accept as-is
            continue;
        }
        work.unshift(head, tail);
    }
    return out;
}
