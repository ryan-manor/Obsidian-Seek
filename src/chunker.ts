// Hierarchical Markdown Chunker — originally ported from
// python-backend/app/search/chunker.py, rewritten structure-aware for WS3.
//
// Strategy:
//   - Split on H1-H6 headings (fence-aware via scanHeadings — a "# comment"
//     line inside fenced code is NOT a heading), build hierarchical title
//     "Note Title > H1 > H2"
//   - Frontmatter parsed for tags / aliases / pageType / created
//   - Aliases are appended to the note title ("Note Title | Alias1 | Alias2 > H1")
//   - Sub-min sections fold into neighbours (carry buffer); title-only notes
//     get a fallback chunk
//
// WS3 change (2026-06-10): sections emit WHOLE — the char-based
// splitOversized / overlap / hardSplit path is deleted. Every production
// chunk passes through enforceTokenBudget (token-budget.ts), which re-splits
// to the 512-token embed window at ATOM boundaries (atoms.ts: fences, tables
// and callouts never split internally except at the token hard ceiling, with
// structure-aware re-wrapping). Two splitters in series meant the char
// splitter could slice a fence before the structure-aware one ever saw it,
// and its 15%-overlap duplication got repacked into arbitrary chunks. Single
// home for splitting = the invariant can actually hold. The old hardSplit
// forward-progress guarantee (JWT-blob class) lives on in token-budget.ts's
// hardSplitByTokens, which has the same strictly-shrinking guard.

import type { Chunk, ChunkMeta, ChunkMetadata } from './types';
import { scanHeadings } from './atoms';
import { toDisplayForm } from './prop-normalize';
export type { Chunk, ChunkMeta, ChunkMetadata };

const FRONTMATTER_RE = /^---\s*\n([\s\S]*?)\n---\s*\n/;

// Path-salted chunk IDs. The hash input is `note_path + title + content`
// (chunkIdFor, below), NOT just the embedded `title + content`. The salt makes
// chunk_id unique per file even when two notes share identical title-and-content.
//
// Why the salt (the bug it fixes): incremental delete (reindexDelta) deletes a
// file's chunks by the chunk_ids stored in its file-record. With content-only
// IDs, two distinct files with identical title-and-content collapsed to ONE
// chunk_id; the store (keyPath chunk_id) kept a single row that BOTH file-records
// referenced. Deleting/editing one file then dropped that shared row and orphaned
// the survivor's reference — the survivor stayed in its file-record but vanished
// from the chunk/embedding/binary stores. Salting by path gives same-content
// twins distinct ids, so each file owns its own rows. (Identical sections WITHIN
// one note still collapse — same path+title+content — which is correct: one row,
// one vector.)
//
// Accepted cost (confirmed decision, plan §"Confirmed Decisions"): every existing
// chunk_id changes, so the next reindex re-embeds the whole vault once. No
// dual-key embedding-cache scheme. A folder move now also changes the id (path
// changed) and re-embeds — the symmetric trade for correct delete semantics.
//
// chunk_id is therefore no longer 1:1 with the embedding vector (two same-content
// twins share a vector but have distinct ids); it IS 1:1 with (file, embedding).
//
// cyrb53, not SHA-256: subtle.digest is async + ~200 μs/call → ~3 s overhead
// on a 15k-chunk reindex. cyrb53 is sync, ~5 μs/call, 53-bit output. Birthday
// collisions become probable at ~95 M items; at 100k chunks the collision
// probability is ~1 in 1.8 billion. Adequate for vault-internal IDs.
export function cyrb53Hex(str: string, seed = 0): string {
    let h1 = 0xdeadbeef ^ seed;
    let h2 = 0x41c6ce57 ^ seed;
    for (let i = 0; i < str.length; i++) {
        const ch = str.charCodeAt(i);
        h1 = Math.imul(h1 ^ ch, 2654435761);
        h2 = Math.imul(h2 ^ ch, 1597334677);
    }
    h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
    h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
    // 14 hex chars = 53 bits of state. h2's low 21 bits in front, h1's 32 bits behind.
    return (2097151 & h2).toString(16).padStart(6, '0') + (h1 >>> 0).toString(16).padStart(8, '0');
}

// THE chunk_id derivation — single module-level definition so every id site
// (the three chunker sites below AND the WS2.3 token-budget packer, which
// re-derives ids for re-split parts in token-budget.ts) shares one salt
// format. notePath is prepended to the embedded `title\n\ncontent` string.
export function chunkIdFor(notePath: string, title: string, content: string, denseSuffix?: string): string {
    // denseSuffix is the note-level frontmatter values appended to the dense
    // EMBED input (token-budget.ts embedInput) — so it MUST enter the id too,
    // or a delta reindex of an mtime-unchanged note would re-embed it under the
    // old (suffix-less) id and strand the new vector. The tail mirrors
    // embedInput's `\n\n${denseSuffix}` exactly, so `${notePath}\n` + embedInput
    // is the hashed string by construction. Empty/absent suffix → no tail (id
    // identical to the pre-suffix form, only the CHUNKER_VERSION bump re-embeds).
    const tail = denseSuffix ? `\n\n${denseSuffix}` : '';
    return cyrb53Hex(`${notePath}\n${title}\n\n${content}${tail}`);
}

// Sidecar version gate: bump whenever chunking changes the bytes that feed
// chunkIdFor (split boundaries, salt format, the embedded title/content shape).
// A consumer with a different CHUNKER_VERSION can't reproduce a producer's ids
// by re-chunking, so it refuses that producer's sidecar (sidecar-meta.ts
// metaAccepts) rather than silently hydrating nothing. 3 = post-WS3
// structure-aware chunker (2026-06-10). 4 = note-level frontmatter values
// folded into the dense embed text + chunk_id (buildDenseSuffix, 2026-06-18) —
// the embedded bytes (and thus the ids) changed, so a pre-4 sidecar's vectors
// are unreproducible here and must be rejected. 5 = lightweight cleanliness
// gates on that suffix (shape/dedup/cap, 2026-06-18) — the suffix bytes shrink,
// so the ids shift again and a pre-5 sidecar is likewise rejected.
export const CHUNKER_VERSION = 5;

export interface ChunkerOptions {
    minChunkChars?: number;
}

export class MarkdownChunker {
    readonly minChunkChars: number;

    constructor(opts: ChunkerOptions = {}) {
        this.minChunkChars = opts.minChunkChars ?? 50;
    }

    chunkContent(content: string, notePath: string, noteTitle?: string, modified?: string | null): Chunk[] {
        const baseTitle = noteTitle ?? notePath.split('/').pop()!.replace(/\.md$/, '');

        // ---- Frontmatter ----
        let metadata: Record<string, unknown> = {};
        let body = content;
        // Lines occupied by the frontmatter block. start_line/end_line are counted
        // against the frontmatter-stripped `body` below (line 1 = first body line),
        // then shifted back by this count at the return site so the emitted spans are
        // RAW-FILE line numbers — see that comment for why the consumers need it.
        let fmLineCount = 0;
        const fm = FRONTMATTER_RE.exec(content);
        if (fm) {
            metadata = parseFrontmatter(fm[1]);
            body = content.slice(fm[0].length);
            for (let i = 0; i < fm[0].length; i++) if (content[i] === '\n') fmLineCount++;
        }

        const aliases = extractAliases(metadata);
        const titleWithAliases = aliases.length > 0
            ? `${baseTitle} | ${aliases.join(' | ')}`
            : baseTitle;

        // Note-level frontmatter values folded into the DENSE channel only (see
        // buildDenseSuffix). Computed once from the RAW parsed frontmatter and
        // carried on every chunk of the note (via idFor + buildChunk below), the
        // same way aliases are hoisted into every chunk title. token-budget.ts
        // embedInput appends it to the embed string; BM25 never sees it.
        const denseSuffix = buildDenseSuffix(metadata);

        const normalizedMetadata: ChunkMetadata = {
            tags: extractTags(metadata),
            aliases,
            pageType: String(metadata['pageType'] ?? ''),
            created: extractDate(metadata['created']),
            modified: modified ?? null,
            properties: extractProperties(metadata),
        };

        const lines = body.split('\n');
        // Fence-aware: a "#"-prefixed line inside fenced code is code, not a
        // section boundary (atoms.ts scanHeadings; WS3 fixture fence-hash-heading.md).
        const headings = scanHeadings(lines);

        const chunks: Chunk[] = [];

        // Path-salted chunk_id (see cyrb53Hex block above). One local binding of
        // the module-level chunkIdFor so the three id sites — buildChunk, the
        // carry backward-fold, and the title-only fallback — can never drift.
        const idFor = (title: string, content: string): string =>
            chunkIdFor(notePath, title, content, denseSuffix);

        // Build the exact chunk record. Factored out of emit() so the end-of-note
        // carry flush can re-derive a chunk_id after mutating content.
        const buildChunk = (
            content: string,
            title: string,
            headingPath: string[],
            startLine: number,
            endLine: number,
        ): Chunk => ({
            chunk_id: idFor(title, content),
            title,
            content,
            note_path: notePath,
            heading_path: headingPath,
            metadata: normalizedMetadata,
            start_line: startLine,
            end_line: endLine,
            ...(denseSuffix && { denseSuffix }),
        });

        // Carry buffer for sub-minChunkChars sections. Dropping them is the §7
        // "quiet recall hole": a note with one healthy section lost its short ones
        // from BOTH indexes, and a note of only-short sections lost its whole body
        // to the title-only fallback ("napkin notes" — prime search targets). We
        // hold a short section and fold it forward into the next full chunk; a
        // trailing remainder folds backward into the preceding chunk; a note that
        // is entirely short collapses to one note-level chunk. carryTitle/Path/Start
        // track the FIRST held section so a standalone flush keeps its attribution.
        let carry = '';
        let carryTitle = '';
        let carryHeadingPath: string[] = [];
        let carryStart = -1;

        const emit = (
            sectionContent: string,
            title: string,
            headingPath: string[],
            startLine: number,
            endLine: number,
            headingText = '',
        ) => {
            const trimmed = sectionContent.trim();
            if (trimmed.length === 0) return; // nothing to index or carry

            if (trimmed.length < this.minChunkChars) {
                // Hold it instead of dropping. Prepend the heading word so a
                // "## Bank / call them" napkin section stays findable by "Bank"
                // even after its body folds into a neighbour (whose title won't
                // carry it). The heading lives in the chunk content, not its title.
                const piece = headingText ? `${headingText}\n${trimmed}` : trimmed;
                if (carry) {
                    carry = `${carry}\n\n${piece}`;
                } else {
                    carry = piece;
                    carryTitle = title;
                    carryHeadingPath = headingPath;
                    carryStart = startLine;
                }
                return;
            }

            // Full section: fold any pending short content in front of it. The
            // carried text inherits this section's title — a minor attribution
            // smudge we accept to keep it out of a tiny standalone chunk (§8).
            let content = trimmed;
            let start = startLine;
            if (carry) {
                content = `${carry}\n\n${trimmed}`;
                start = carryStart;
                carry = '';
            }

            // Emit the section WHOLE — oversize sections are re-split at atom
            // boundaries by enforceTokenBudget (token-budget.ts), the single
            // home for splitting since WS3. `(part N)` numbering happens there
            // too (display-only, never in the embedded/hashed `title`).
            chunks.push(buildChunk(content, title, headingPath, start, endLine));
        };

        if (headings.length === 0) {
            emit(body, titleWithAliases, [], 1, lines.length);
        } else {
            const firstHeadingLine = headings[0].lineNum;
            if (firstHeadingLine > 0) {
                const preContent = lines.slice(0, firstHeadingLine).join('\n');
                emit(preContent, titleWithAliases, [], 1, firstHeadingLine);
            }

            const headingStack: Array<{ level: number; text: string }> = [];
            for (let idx = 0; idx < headings.length; idx++) {
                const { lineNum, level, text: headingText } = headings[idx];
                const endLine = idx + 1 < headings.length ? headings[idx + 1].lineNum : lines.length;

                while (headingStack.length > 0 && headingStack[headingStack.length - 1].level >= level) {
                    headingStack.pop();
                }
                headingStack.push({ level, text: headingText });

                const headingPath = headingStack.map(h => h.text);
                const title = `${titleWithAliases} > ${headingPath.join(' > ')}`;
                const sectionContent = lines.slice(lineNum + 1, endLine).join('\n');
                emit(sectionContent, title, headingPath, lineNum + 1, endLine, headingText);
            }
        }

        // Flush trailing sub-min content that never found a following full section.
        if (carry) {
            if (chunks.length > 0) {
                // Fold backward into the preceding chunk (§7 / §8: avoid a tiny
                // standalone chunk that would skew IDF and crowd the candidate pool).
                // Re-derive chunk_id since content changed; extend end_line.
                const last = chunks[chunks.length - 1];
                last.content = `${last.content}\n\n${carry}`;
                last.chunk_id = idFor(last.title, last.content);
                last.end_line = lines.length;
            } else {
                // The whole note is sub-min sections — emit it as ONE real,
                // body-bearing chunk (NOT the lexical-only title stub below: it
                // carries searchable text). This is the napkin-note case.
                chunks.push(buildChunk(carry, carryTitle, carryHeadingPath, carryStart, lines.length));
            }
            carry = '';
        }

        // Fallback: a note whose body never clears minChunkChars (e.g. a title-only
        // People/Places stub or an empty Home page) emits zero chunks above and is then
        // skipped entirely by reindexAll (search.ts: `if (fileChunks.length === 0) continue`),
        // making it absent from BOTH the dense and BM25 indexes — unretrievable even by
        // its exact name. Emit one title-only chunk so the note is still indexed via its
        // title (embedded as `${title}\n\n${content}`; BM25 boosts the title field 3.0x).
        // The minChunkChars gate stays in force for sub-sections of larger notes; this
        // only rescues notes that would otherwise vanish. Verified to recover 22/25
        // structurally-dead known-item queries in the personal eval (2026-06-05).
        if (chunks.length === 0) {
            const fallbackContent = body.trim();
            // A body-LESS note (empty fallbackContent) embeds as "<title>\n\n" —
            // a content-free vector (often just a bare date) that becomes a
            // universal near-neighbor for any OOV/ID query. Mark it lexical-only
            // so the ranker keeps it out of the dense channel; BM25's title boost
            // still makes it findable by name. A short-but-non-empty fallback DOES
            // carry embeddable content, so it stays dense-eligible (flag absent).
            // See [[seek-miss-anatomy]] (the 2026-06-05 fallback) for why the
            // fallback exists at all, and the dense-floor in ranker.ts for the
            // other half of this fix.
            const lexicalOnly = fallbackContent.length === 0 ? true : undefined;
            chunks.push({
                chunk_id: idFor(titleWithAliases, fallbackContent),
                title: titleWithAliases,
                content: fallbackContent,
                note_path: notePath,
                heading_path: [],
                metadata: normalizedMetadata,
                start_line: 1,
                end_line: lines.length,
                ...(lexicalOnly && { lexicalOnly }),
                ...(denseSuffix && { denseSuffix }),
            });
        }

        // start_line/end_line above were counted against the frontmatter-STRIPPED
        // `body` (line 1 = first body line). Shift every chunk — including the
        // carry-fold mutation and both fallbacks, which is why this is done once at
        // the single return chokepoint — into RAW-FILE coordinates: 1-based lines
        // into the on-disk note, frontmatter INCLUDED. That is the contract the
        // field's only consumers rely on (search-modal.ts): the in-note match
        // highlight walks the raw note text from `start_line`, and the click handler
        // drives editor.setCursor/scrollIntoView — and the editor counts frontmatter
        // as lines. Without this shift both land ~fmLineCount lines too early (inside
        // the frontmatter), so the highlight drifts onto the note's FIRST occurrence
        // of a query token rather than the matched chunk's, and diverges from the
        // snippet (which is built straight from chunk content). chunk_id does not
        // depend on line numbers, so this never changes chunk identities — only the
        // stored span. (DB_VERSION 10→11 forces the one coherent rebuild; the
        // mtime-only delta would otherwise leave the index mixing old body-relative
        // and new file-relative spans.)
        if (fmLineCount > 0) {
            for (const c of chunks) {
                c.start_line += fmLineCount;
                c.end_line += fmLineCount;
            }
        }
        return chunks;
    }

}

// ---- frontmatter helpers ----

// Intentionally minimal: we only need a handful of fields, and pulling in
// js-yaml would balloon the bundle by ~50 KB.
function parseFrontmatter(yaml: string): Record<string, unknown> {
    const result: Record<string, unknown> = {};
    const lines = yaml.split('\n');
    let i = 0;
    while (i < lines.length) {
        const line = lines[i];
        const m = /^([A-Za-z0-9_\-]+):\s*(.*)$/.exec(line);
        if (!m) { i++; continue; }
        const key = m[1];
        const valueStr = m[2].trim();

        if (valueStr === '') {
            // Block scalar — could be a list. Peek next lines for "  - item".
            const items: string[] = [];
            let j = i + 1;
            while (j < lines.length) {
                const next = lines[j];
                const listM = /^\s+-\s*(.+)$/.exec(next);
                if (!listM) break;
                items.push(stripQuotes(listM[1].trim()));
                j++;
            }
            result[key] = items;
            i = j;
            continue;
        }

        if (valueStr.startsWith('[') && valueStr.endsWith(']')) {
            const inner = valueStr.slice(1, -1).trim();
            result[key] = inner ? inner.split(',').map(s => stripQuotes(s.trim())) : [];
        } else {
            result[key] = stripQuotes(valueStr);
        }
        i++;
    }
    return result;
}

function stripQuotes(s: string): string {
    if ((s.startsWith('"') && s.endsWith('"')) || (s.startsWith("'") && s.endsWith("'"))) {
        return s.slice(1, -1);
    }
    return s;
}

function extractAliases(metadata: Record<string, unknown>): string[] {
    const raw = metadata['aliases'];
    if (Array.isArray(raw)) return raw.map(v => String(v).trim()).filter(Boolean);
    if (typeof raw === 'string') return raw.split(',').map(s => s.trim()).filter(Boolean);
    return [];
}

function extractTags(metadata: Record<string, unknown>): string[] {
    const raw = metadata['tags'];
    if (Array.isArray(raw)) return raw.map(v => String(v).trim()).filter(Boolean);
    if (typeof raw === 'string') return raw.split(/[,\s]+/).map(s => s.trim()).filter(Boolean);
    return [];
}

// Generic backing store for `[key:value]` inline filters: every SCALAR
// frontmatter value, string-coerced, keyed by frontmatter key. Array values
// (tags/aliases and any other list-valued props) are skipped — they have
// dedicated handling and don't participate in exact-match property filters.
// parseFrontmatter only ever emits string | string[], so the typeof guard is
// sufficient; the number/boolean arms are defensive.
function extractProperties(metadata: Record<string, unknown>): Record<string, string> {
    const props: Record<string, string> = {};
    for (const [key, value] of Object.entries(metadata)) {
        if (typeof value === 'string') props[key] = value;
        else if (typeof value === 'number' || typeof value === 'boolean') props[key] = String(value);
    }
    return props;
}

function extractDate(value: unknown): string | null {
    if (value == null) return null;
    const s = String(value);
    const m = /(\d{4}-\d{2}-\d{2})/.exec(s);
    return m ? m[1] : null;
}

// ---- dense frontmatter suffix (the 2026-06-18 frontmatter-into-dense ship) ----

// Frontmatter keys whose values are NOT folded into the dense suffix: aliases
// are already in the dense channel via titleWithAliases (chunkContent), so
// re-injecting them would double their weight. Mirrors the eval harness's
// ALIAS_KEYS exactly (ofm_fm_valuetype.py); everything else — tags, pageType,
// place links, free-text props — is fair game (tags ride here AND in their own
// BM25 field, but only this dense copy is the validated win).
const SUFFIX_SKIP_KEYS = new Set(['aliases', 'alias']);

// Inert value TYPES, dropped AFTER link-flattening so the filter is by VALUE
// shape not key name (no curation list to maintain): an ISO date (optionally
// with a trailing time), a pure number (incl. a bare year), or a boolean-ish
// word. granite ignores these SEMANTICALLY, but an ignored token still occupies
// CLS-pooling mass in a short pooled vector, so on a thin note it dilutes the
// one value that matters — and dates are often DOUBLED (`created` + a
// `[[YYYY-MM-DD]]` dateLink flatten to the same string). Tested against the
// WHOLE flattened item, so "trip 2026" (real text) survives but a bare "2026"
// does not. Byte-for-byte the validated harness INERT_RE (ofm_fm_valuetype.py).
const INERT_VALUE_RE = /^(?:\d{4}-\d{2}-\d{2}(?:[ T].*)?|-?\d+(?:\.\d+)?|true|false|yes|no|on|off)$/i;

// Shape gate: by-FORM junk the type-filter misses (frontmatter census, 2026-06-18).
// A number-list is >=2 all-numeric tokens — lat/long coordinate strings and value
// lists; a lone number is already INERT, so this only adds the multi-number case
// and never touches text ("trip 2026" keeps its word, "1.2.3" is one non-numeric
// token). URLs are 13% of suffix notes (all clippings), assets are og:image paths.
const SUFFIX_URL_RE = /https?:\/\//i;
const SUFFIX_ASSET_RE = /\.(png|jpe?g|gif|webp|svg|bmp|tiff?|pdf|mp4|mov|webm|mkv|mp3|m4a|wav|zip)\b/i;
const SUFFIX_NUM_TOK_RE = /^[+-]?\d+(?:\.\d+)?$/;
const SUFFIX_CAP_TOKENS = 48;   // ~census p90; bounds the pooling-domination tail

function isNumberList(t: string): boolean {
    const toks = t.split(/[,\s]+/).filter(Boolean);
    return toks.length >= 2 && toks.every((x) => SUFFIX_NUM_TOK_RE.test(x));
}

function isShapeJunk(t: string): boolean {
    return SUFFIX_URL_RE.test(t) || SUFFIX_ASSET_RE.test(t) || isNumberList(t);
}

// The per-note dense suffix: every frontmatter value (keys dropped), wikilinks
// flattened to their target BASENAME via toDisplayForm — the blessed display-form
// unwrap, deliberately NOT toBindForm, which keeps path + alias tokens and would
// manufacture the "Zurich Zurich" keyword-stuffing that prop-normalize.ts exists
// to prevent — aliases excluded, and date/number/boolean values type-dropped.
// Three lightweight cleanliness gates then run, all pure functions of the note
// (no corpus stats), validated in fm_cleanliness_arm.py (captures +0.0157 vs the
// type-filtered ship, sub-resolution elsewhere, OOD-inert): SHAPE drops URL/asset/
// coordinate values by form; DEDUP collapses repeated values case-insensitively
// (the 65% context/tags doubling) at the whole-value level; CAP truncates the
// joined suffix to SUFFIX_CAP_TOKENS. Dedup precedes cap so the budget is never
// spent on duplicates. Returns '' when nothing qualifies (caller: '' = no suffix).
function buildDenseSuffix(metadata: Record<string, unknown>): string {
    const parts: string[] = [];
    const seen = new Set<string>();
    for (const [key, value] of Object.entries(metadata)) {
        if (SUFFIX_SKIP_KEYS.has(key.toLowerCase())) continue;
        const items = Array.isArray(value) ? value : [value];
        for (const item of items) {
            const t = toDisplayForm(String(item)).trim();
            if (!t || INERT_VALUE_RE.test(t) || isShapeJunk(t)) continue;
            const k = t.toLowerCase();
            if (seen.has(k)) continue;            // whole-value dedup, order-preserving
            seen.add(k);
            parts.push(t);
        }
    }
    const joined = parts.join(' ');
    const toks = joined.split(/\s+/);
    return toks.length > SUFFIX_CAP_TOKENS ? toks.slice(0, SUFFIX_CAP_TOKENS).join(' ') : joined;
}
