// Hierarchical Markdown Chunker — originally ported from
// python-backend/app/search/chunker.py, rewritten structure-aware for WS3.
//
// Strategy:
//   - Split on H1-H6 headings (fence-aware via scanHeadings — a "# comment"
//     line inside fenced code is NOT a heading), build hierarchical title
//     "Note Title > H1 > H2"
//   - Frontmatter parsed for tags / aliases / created (all other scalar keys,
//     pageType included, are folded generically into metadata.properties);
//     metadata.tags also unions inline body #tags (tag-grammar.ts, audit R2 #2)
//     and aliases accepts the legacy singular `alias:` key (audit R2 #5)
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

import type { Chunk, ChunkMeta, ChunkMetadata, BaseView } from './types';
import { scanHeadings } from './atoms';
import { toDisplayForm } from './prop-normalize';
import { cleanDenseText, cleanDenseBody, extractLinkTerms, extractLinkTermsBody, ASSET_EXT_RE as SUFFIX_ASSET_RE } from './dense-clean';
import { depluralize, MACHINERY_KEYS } from './bm25';
import { seekTokenize } from './tokenize';
import { extractInlineTags } from './tag-grammar';
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
// so the ids shift again and a pre-5 sidecar is likewise rejected. 6 = .base
// files chunk per-VIEW via chunkBase (base-as-document model, 2026-06-23) — one
// base now yields a base-level chunk + one per non-generic view instead of a
// single mashed doc, so every base chunk's title/content (and id) changed and a
// pre-6 sidecar's base vectors are unreproducible. 7 = within-section overlap
// (token-budget.ts OVERLAP_FRACTION, 2026-06-23): a split super-section's later
// parts now carry the prior part's trailing paragraph(s), so every multi-part
// section's part content (and ids) changed and a pre-7 sidecar's vectors for
// those notes are unreproducible. 8 = dense-channel hygiene (2026-06-28): the
// section body and each heading are run through cleanDenseText/cleanDenseBody
// (dense-clean.ts) at the chunker — wikilink/embed syntax flattened to rendered
// display text, bare URLs stripped of scheme/TLD, HTML removed, fenced code left
// verbatim — AND the dense suffix now dedups its values against the note title's
// (depluralized) tokens (buildDenseSuffix). Both shift the embedded/indexed bytes
// (and thus the ids), so a pre-8 sidecar's vectors are unreproducible here.
// 9 = dense suffix name-excludes MACHINERY_KEYS (Relevance Quality Audit
// 2026-06-29 #2): text-valued UI keys (icon/cssclasses/cssclass/banner/…) no
// longer leak into the suffix, so a machinery-bearing note's suffix bytes (and
// id) shift and a pre-9 sidecar's vectors for those notes are unreproducible.
// 10 = lexical reclamation (2026-07-02, audit R2 #1): chunks gain link_terms —
// the raw substrings v8's dense-clean DROPPED (aliased wikilink targets,
// markdown-link/autolink URLs), BM25-folded into the content field at doc
// build. UNLIKE every bump above, the embedded bytes and chunk_ids are
// UNCHANGED (link_terms is deliberately outside chunkIdFor); the bump exists
// because stored ChunkMeta records need the new field backfilled, and a
// version mismatch is the one fleet-converging "re-run the chunker" trigger:
// desktop full-reindexes, mobile re-chunks live embed-free on hydrate. A v9
// sidecar's VECTORS would technically still be reproducible here, but the
// conservative reject costs only one desktop re-seed.
//
// PENDING (would be 11, NOT bumped yet): three fixes landed after v10 shipped
// that also move buildDenseSuffix's output or the parsed frontmatter/body —
// and thus chunkIdFor's tail or content — for affected notes: (a)
// INERT_VALUE_RE narrowed to stop swallowing free text trailing a date
// property value (audit R2 batch2 #3, chunker.ts); (b) SUFFIX_ASSET_RE
// unified onto dense-clean.ts's $-anchored ASSET_EXT_RE instead of its own
// drifted \b-bounded copy, so a property value that merely CONTAINS an asset
// extension (e.g. "trip.pdf notes") no longer gets misclassified as
// shape-junk and dropped (audit R2 batch2 #5, chunker.ts + dense-clean.ts);
// and (c) a leading BOM (U+FEFF) is now stripped before frontmatter parsing
// (audit R2 batch2 #4, chunkContent) — previously FRONTMATTER_RE's `^---`
// anchor never matched a BOM-prefixed file, so metadata stayed `{}` and body
// was the whole raw file including the frontmatter block text; for any
// BOM-prefixed note this changes metadata, body, denseSuffix, and hence
// chunk_id. All three are eval-gate-before-ship the same way v10 itself was —
// DELIBERATELY left off the live CHUNKER_VERSION until that eval runs, not an
// oversight. In the meantime this means a producer and consumer can each
// self-report chunkerVersion=10 while actually running different
// id-affecting code (a staggered-rollout risk on IDENTITY, not data safety —
// hydrateFromSidecar's id-lookup path just falls through to a normal local
// re-embed on a miss, same as any other unmatched id). Fold all three into
// this comment block when the eval runs and the version finally bumps to 11.
export const CHUNKER_VERSION = 10;

export interface ChunkerOptions {
    minChunkChars?: number;
}

export class MarkdownChunker {
    readonly minChunkChars: number;

    constructor(opts: ChunkerOptions = {}) {
        this.minChunkChars = opts.minChunkChars ?? 50;
    }

    chunkContent(content: string, notePath: string, noteTitle?: string, modified?: string | null): Chunk[] {
        // Strip a leading BOM (U+FEFF) before frontmatter parsing (audit R2
        // batch2 #4): FRONTMATTER_RE is anchored at the very start of the
        // string (^---), so a BOM-prefixed file (common from Windows editors
        // and some web clippers) never matched it and lost its frontmatter
        // entirely — tags, aliases, and every property silently vanished.
        if (content.charCodeAt(0) === 0xfeff) content = content.slice(1);
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
        const denseSuffix = buildDenseSuffix(metadata, titleWithAliases);

        const normalizedMetadata: ChunkMetadata = {
            // Frontmatter tags + inline body `#tags` (audit R2 #2): Obsidian's
            // own metadataCache indexes both, so the suggester offers inline
            // tags — the store has to carry them or those pills match nothing.
            // NOTE-level union (a tag anywhere tags the whole note, Obsidian
            // semantics), shared by every chunk via this one object. Metadata
            // only — not hashed by chunkIdFor, so ids/vectors are untouched;
            // the v10 bump backfills stored records exactly as for link_terms.
            tags: mergeTags(extractTags(metadata), extractInlineTags(body)),
            aliases,
            created: extractDate(metadata['created']),
            modified: modified ?? null,
            // pageType is NOT a dedicated field — extractProperties folds it into
            // `properties` like any other scalar key, so it reaches the searchable
            // properties field and `[pageType:x]` filters via the same generic path.
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
            linkTerms = '',
        ): Chunk => ({
            chunk_id: idFor(title, content),   // link_terms deliberately NOT hashed — ids/vectors stay stable
            title,
            content,
            note_path: notePath,
            heading_path: headingPath,
            metadata: normalizedMetadata,
            start_line: startLine,
            end_line: endLine,
            ...(denseSuffix && { denseSuffix }),
            ...(linkTerms && { link_terms: linkTerms }),
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
        let carryLinkTerms = '';
        const joinLt = (a: string, b: string) => (a && b ? `${a} ${b}` : a || b);

        const emit = (
            sectionContent: string,
            title: string,
            headingPath: string[],
            startLine: number,
            endLine: number,
            headingText = '',
            rawHeading = '',
        ) => {
            // Dense/lexical body hygiene (v8): flatten link/embed syntax, strip
            // URL/HTML noise, keep fenced code verbatim. Cleaning here (not at
            // embedInput) means the min-chunk gate and chunk_id both see the
            // cleaned bytes, and BOTH channels index them. The raw-file line span
            // (start_line/end_line) is unaffected — it is recomputed from the
            // on-disk note below, never from this string's length.
            const trimmed = cleanDenseBody(sectionContent);

            // Lexical reclamation (v10): what the clean DROPPED from this section
            // (and its raw heading line) — persisted BM25-only via link_terms so
            // `theverge.com` / `[[Alex Goel|Alex]]`-target queries still match.
            const linkTerms = joinLt(extractLinkTermsBody(sectionContent), extractLinkTerms(rawHeading));

            if (trimmed.length === 0) {
                // Nothing to index or carry as TEXT — but an image/link-only
                // section (e.g. a lone `![[Rapha Jersey.png]]`) still dropped
                // lexical material; ride it on the carry so it lands on the
                // neighbouring chunk instead of vanishing (pre-v8 the raw
                // section text was indexed).
                carryLinkTerms = joinLt(carryLinkTerms, linkTerms);
                return;
            }

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
                carryLinkTerms = joinLt(carryLinkTerms, linkTerms);
                return;
            }

            // Full section: fold any pending short content in front of it. The
            // carried text inherits this section's title — a minor attribution
            // smudge we accept to keep it out of a tiny standalone chunk (§8).
            let content = trimmed;
            let start = startLine;
            // Fold pending carry link-terms in unconditionally — they can exist
            // WITHOUT carry text (the image/link-only-section case above).
            const lt = joinLt(carryLinkTerms, linkTerms);
            carryLinkTerms = '';
            if (carry) {
                content = `${carry}\n\n${trimmed}`;
                start = carryStart;
                carry = '';
            }

            // Emit the section WHOLE — oversize sections are re-split at atom
            // boundaries by enforceTokenBudget (token-budget.ts), the single
            // home for splitting since WS3. `(part N)` numbering happens there
            // too (display-only, never in the embedded/hashed `title`).
            chunks.push(buildChunk(content, title, headingPath, start, endLine, lt));
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
                const { lineNum, level } = headings[idx];
                // Heading hygiene (v8): a heading can carry wikilinks/URLs
                // ("## Sync with [[Alex Goel|Alex]]"). Flatten to rendered
                // display text so the dense title and the BM25 headings field
                // both index "Alex", not "[[Alex Goel|Alex]]". Fall back to the
                // raw text if cleaning empties it (e.g. a heading that is only an
                // image embed), so the breadcrumb never gets a blank segment.
                const headingText = cleanDenseText(headings[idx].text) || headings[idx].text;
                const endLine = idx + 1 < headings.length ? headings[idx + 1].lineNum : lines.length;

                while (headingStack.length > 0 && headingStack[headingStack.length - 1].level >= level) {
                    headingStack.pop();
                }
                headingStack.push({ level, text: headingText });

                const headingPath = headingStack.map(h => h.text);
                const title = `${titleWithAliases} > ${headingPath.join(' > ')}`;
                const sectionContent = lines.slice(lineNum + 1, endLine).join('\n');
                emit(sectionContent, title, headingPath, lineNum + 1, endLine, headingText, headings[idx].text);
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
                if (carryLinkTerms) last.link_terms = joinLt(last.link_terms ?? '', carryLinkTerms);
            } else {
                // The whole note is sub-min sections — emit it as ONE real,
                // body-bearing chunk (NOT the lexical-only title stub below: it
                // carries searchable text). This is the napkin-note case.
                chunks.push(buildChunk(carry, carryTitle, carryHeadingPath, carryStart, lines.length, carryLinkTerms));
            }
            carry = '';
            carryLinkTerms = '';
        } else if (carryLinkTerms && chunks.length > 0) {
            // Trailing image/link-only section: no carried text, but its dropped
            // lexical material still needs a home — the preceding chunk. (With
            // zero chunks the title-only fallback below re-derives from the whole
            // body, so nothing is lost there either.)
            const last = chunks[chunks.length - 1];
            last.link_terms = joinLt(last.link_terms ?? '', carryLinkTerms);
            carryLinkTerms = '';
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
            const fallbackContent = cleanDenseBody(body);
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
            const fallbackLinkTerms = extractLinkTermsBody(body);
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
                ...(fallbackLinkTerms && { link_terms: fallbackLinkTerms }),
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

    // Chunk an Obsidian `.base` file. A base is a document whose VIEWS are its
    // sections (base-extractor.ts extractBaseDocs): each BaseView becomes one chunk
    // titled "<base> > <viewName>" with heading_path ["<viewName>"], so the view
    // name earns the 3.0x BM25 headings field and routes a semantic query to the
    // right view. The base-level entry (viewName null) is the bare "<base>" title
    // with an empty heading_path and wins bare-name queries via title-boost —
    // mirroring how a note carries a whole-note identity alongside its sections.
    // Reuses the module-level chunkIdFor (no denseSuffix: bases have no frontmatter
    // values), so a base chunk's id matches across every production site that
    // routes through chunksFor. Bases carry no note metadata and span one logical
    // line, so start/end_line = 1. Content is never empty (the base name is always
    // present), so no chunk needs the lexical-only fallback the markdown path uses.
    chunkBase(views: BaseView[], notePath: string, modified?: string | null): Chunk[] {
        const base = notePath.split('/').pop()!.replace(/\.base$/, '');
        const metadata: ChunkMetadata = {
            tags: [],
            aliases: [],
            created: null,
            modified: modified ?? null,
            properties: {},
        };
        return views.map(v => {
            const title = v.viewName ? `${base} > ${v.viewName}` : base;
            const headingPath = v.viewName ? [v.viewName] : [];
            return {
                chunk_id: chunkIdFor(notePath, title, v.content),
                title,
                content: v.content,
                note_path: notePath,
                heading_path: headingPath,
                metadata,
                start_line: 1,
                end_line: 1,
            };
        });
    }

}

// ---- frontmatter helpers ----

// Intentionally minimal: we only need a handful of fields, and bundling a full
// YAML parser would balloon the bundle by ~50 KB.
function parseFrontmatter(yaml: string): Record<string, unknown> {
    const result: Record<string, unknown> = {};
    const lines = yaml.split('\n');
    let i = 0;
    while (i < lines.length) {
        const line = lines[i];
        const m = /^([A-Za-z0-9_-]+):\s*(.*)$/.exec(line);
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

// Case-insensitive frontmatter KEY lookup. parseFrontmatter preserves the
// author's key casing while every downstream skip-list compares
// key.toLowerCase() — so an exact-case read here silently dropped `Alias:` /
// `Tags:` values from EVERY channel (skipped from properties/suffix by the
// ci skip-lists, never read as alias/tags: the same all-channels-assume-the-
// other bug class as audit R2 #5 itself, reproduced for casing — 2026-07-02
// adversarial review). Exact-case fast path first.
function lookupKeyCI(metadata: Record<string, unknown>, name: string): unknown {
    if (name in metadata) return metadata[name];
    for (const [k, v] of Object.entries(metadata)) {
        if (k.toLowerCase() === name) return v;
    }
    return undefined;
}

// Obsidian accepts BOTH the modern plural key and the legacy singular
// (audit R2 #5: `alias:` reached no ranking channel — this extractor read
// only `aliases` while every downstream skip-list excluded `alias` assuming
// this side carried it). The plural list rides VERBATIM — dupes and all,
// byte-identical to the pre-fix read, so no aliases-only note's title (and
// therefore chunk_id) moves; only the legacy singular entries dedup against
// it (case-insensitive). A note carrying `alias:` (or a case-variant key)
// gains alias tokens in titleWithAliases → its title/denseSuffix/chunk_id
// change BY DESIGN, which the pending v10 reindex absorbs.
function extractAliases(metadata: Record<string, unknown>): string[] {
    const listOf = (raw: unknown): string[] =>
        Array.isArray(raw) ? raw.map(v => String(v).trim()).filter(Boolean)
        : typeof raw === 'string' ? raw.split(',').map(s => s.trim()).filter(Boolean)
        : [];
    const out = listOf(lookupKeyCI(metadata, 'aliases'));
    const seen = new Set(out.map(a => a.toLowerCase()));
    for (const a of listOf(lookupKeyCI(metadata, 'alias'))) {
        const k = a.toLowerCase();
        if (!seen.has(k)) {
            seen.add(k);
            out.push(a);
        }
    }
    return out;
}

function extractTags(metadata: Record<string, unknown>): string[] {
    const raw = lookupKeyCI(metadata, 'tags');
    if (Array.isArray(raw)) return raw.map(v => String(v).trim()).filter(Boolean);
    if (typeof raw === 'string') return raw.split(/[,\s]+/).map(s => s.trim()).filter(Boolean);
    return [];
}

// Union frontmatter tags with inline body tags, case-insensitive first-wins
// (frontmatter casing is the author's canonical form). Keyed with the leading
// `#` stripped — frontmatter tags occasionally carry it — matching how the
// query-parser matcher normalizes both sides before comparing.
function mergeTags(frontmatter: string[], inline: string[]): string[] {
    const out: string[] = [];
    const seen = new Set<string>();
    for (const t of [...frontmatter, ...inline]) {
        const k = t.replace(/^#/, '').toLowerCase();
        if (k && !seen.has(k)) {
            seen.add(k);
            out.push(t);
        }
    }
    return out;
}

// Generic backing store for `[key:value]` inline filters: every frontmatter
// value keyed by frontmatter key — scalars string-coerced, LIST values kept as
// string[] (v10, audit R2 #3: the suggester offers list-prop pills from the
// metadata cache, and the old scalar-only store made every one of them match
// nothing). tags/aliases stay skipped — they have dedicated fields and
// operators. The matcher (query-parser lookupProp) treats a list as
// any-element-matches; the BM25 properties field also folds each list item in
// (extractPropertiesText, audit R2 batch2 #3 — list values used to be
// dense-suffix-only and BM25-invisible).
const PROPERTY_SKIP_KEYS = new Set(['tags', 'aliases', 'alias']);
function extractProperties(metadata: Record<string, unknown>): Record<string, string | string[]> {
    const props: Record<string, string | string[]> = {};
    for (const [key, value] of Object.entries(metadata)) {
        if (PROPERTY_SKIP_KEYS.has(key.toLowerCase())) continue;
        if (typeof value === 'string') props[key] = value;
        else if (typeof value === 'number' || typeof value === 'boolean') props[key] = String(value);
        else if (Array.isArray(value)) {
            const items = value.map(v => String(v).trim()).filter(Boolean);
            if (items.length) props[key] = items;
        }
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

// Frontmatter keys whose values are NOT folded into the dense suffix. Two
// name-based reasons:
//   • aliases/alias — already in the dense channel via titleWithAliases
//     (chunkContent), so re-injecting them would double their weight.
//   • MACHINERY_KEYS — identity/UI/plumbing junk, SHARED with bm25's properties
//     field. The value-shape gates below already drop the date/number/boolean/
//     url machinery, but TEXT-valued keys (`icon: lucide-book`, `cssclasses:
//     [wide-page]`) slipped through because the suffix had no name-list — this
//     closes that one gap (Relevance Quality Audit 2026-06-29 finding #2).
// Everything else — tags, pageType, place links, free-text props — stays IN
// (tags ride here AND in their own BM25 field, but only this dense copy is the
// validated win). The alias half still mirrors the eval harness ALIAS_KEYS
// (ofm_fm_valuetype.py); the machinery half is a NEW divergence the harness
// arm must adopt to stay byte-parallel.
const SUFFIX_SKIP_KEYS = new Set(['aliases', 'alias', ...MACHINERY_KEYS]);

// Inert value TYPES, dropped AFTER link-flattening so the filter is by VALUE
// shape not key name (no curation list to maintain): an ISO date (optionally
// with a trailing time), a pure number (incl. a bare year), or a boolean-ish
// word. granite ignores these SEMANTICALLY, but an ignored token still occupies
// CLS-pooling mass in a short pooled vector, so on a thin note it dilutes the
// one value that matters — and dates are often DOUBLED (`created` + a
// `[[YYYY-MM-DD]]` dateLink flatten to the same string). Tested against the
// WHOLE flattened item, so "trip 2026" (real text) survives but a bare "2026"
// does not. Byte-for-byte the validated harness INERT_RE (ofm_fm_valuetype.py).
// MODULO the date-time tail (audit R2 batch2 #1): the form above matched a
// date plus ANY trailing text ([ T].*) as inert, so a value like
// "2026-06-29 Milan departure" vanished whole — the date recognized, but the
// free text after it silently swallowed along with it, in BOTH channels (see
// bm25.ts PROPERTY_DATE_RE, same fix). Narrowed to only swallow an actual
// ISO time-of-day tail (a bare date, or date+time, is still inert either
// way — dates remain queryable via [key:value] filters); a date FOLLOWED BY
// FREE TEXT no longer matches, so the whole value (date digits and all)
// survives as ordinary searchable text instead of being dropped.
const INERT_VALUE_RE = /^(?:\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?|-?\d+(?:\.\d+)?|true|false|yes|no|on|off)$/i;

// Shape gate: by-FORM junk the type-filter misses (frontmatter census, 2026-06-18).
// A number-list is >=2 all-numeric tokens — lat/long coordinate strings and value
// lists; a lone number is already INERT, so this only adds the multi-number case
// and never touches text ("trip 2026" keeps its word, "1.2.3" is one non-numeric
// token). URLs are 13% of suffix notes (all clippings), assets are og:image paths
// (SUFFIX_ASSET_RE = dense-clean.ts's ASSET_EXT_RE, imported not copied — see
// that file for why: the two independently-defined copies had drifted, $ vs \b).
const SUFFIX_URL_RE = /https?:\/\//i;
const SUFFIX_NUM_TOK_RE = /^[+-]?\d+(?:\.\d+)?$/;
const SUFFIX_CAP_TOKENS = 48;   // ~census p90; bounds the pooling-domination tail

function isNumberList(t: string): boolean {
    const toks = t.split(/[,\s]+/).filter(Boolean);
    return toks.length >= 2 && toks.every((x) => SUFFIX_NUM_TOK_RE.test(x));
}

function isShapeJunk(t: string): boolean {
    return SUFFIX_URL_RE.test(t) || SUFFIX_ASSET_RE.test(t) || isNumberList(t);
}

// Canonical word tokens for cross-surface dedup: the shared lexical split
// (CANONICAL stream only — derived:false drops the camelCase/glue recall forms
// that would over-match), lowercased and depluralized. depluralize is the BM25
// channel's own plural normalizer (bm25.ts), so both channels agree on what
// "the same word" is — and it deliberately stops short of Porter stemming, which
// that channel's own eval rejected (-0.0019 CQADupstack) for over-conflation.
function suffixDedupTokens(s: string): string[] {
    return seekTokenize(s, { derived: false }).map(w => depluralize(w.toLowerCase()));
}

// The per-note dense suffix: every frontmatter value (keys dropped), wikilinks
// flattened to their target BASENAME via toDisplayForm — the blessed display-form
// unwrap, deliberately NOT toBindForm, which keeps path + alias tokens and would
// manufacture the "Zurich Zurich" keyword-stuffing that prop-normalize.ts exists
// to prevent — aliases excluded, and date/number/boolean values type-dropped.
// Three lightweight cleanliness gates then run, all pure functions of the note
// (no corpus stats), validated in fm_cleanliness_arm.py (captures +0.0157 vs the
// type-filtered ship, sub-resolution elsewhere, OOD-inert): SHAPE drops URL/asset/
// coordinate values by form; DEDUP now runs on the depluralized TOKEN SET (not the
// whole-value string) and is SEEDED with the note title+alias tokens — so a value
// that merely repeats a title word ("Rapha" tag on a "Rapha Order" note) or the
// pageType/tag "meeting"/"meetings" near-dup no longer re-enters the pooled
// vector (v8 cross-surface dedup); CAP truncates the joined suffix to
// SUFFIX_CAP_TOKENS. A value is skipped only when EVERY one of its tokens is
// already seen, so a multi-word phrase with any novel token survives — the "not
// over-zealous" guard (the body, which mentions these words in prose, is never
// touched here). Dedup precedes cap so the budget is never spent on duplicates.
// Returns '' when nothing qualifies (caller: '' = no suffix).
function buildDenseSuffix(metadata: Record<string, unknown>, titleText = ''): string {
    const parts: string[] = [];
    const seen = new Set<string>(suffixDedupTokens(titleText));   // cross-surface seed
    for (const [key, value] of Object.entries(metadata)) {
        if (SUFFIX_SKIP_KEYS.has(key.toLowerCase())) continue;
        const items = Array.isArray(value) ? value : [value];
        for (const item of items) {
            const t = toDisplayForm(String(item)).trim();
            if (!t || INERT_VALUE_RE.test(t) || isShapeJunk(t)) continue;
            const toks = suffixDedupTokens(t);
            if (toks.length > 0 && toks.every(tok => seen.has(tok))) continue; // fully-redundant
            for (const tok of toks) seen.add(tok);
            parts.push(t);
        }
    }
    const joined = parts.join(' ');
    const toks = joined.split(/\s+/);
    return toks.length > SUFFIX_CAP_TOKENS ? toks.slice(0, SUFFIX_CAP_TOKENS).join(' ') : joined;
}
