// Inline query-syntax parser + chunk filter matcher.
//
// A native TS port of the predecessor Python backend's proven filter layer
// (`python-backend/app/routes/search.py` preprocess_query / _apply_chunk_filters,
// with its 40-case test suite). Design rationale lives in
// [[Seek Retrieval Relevance & Query]] §Filters / §Query Processing.
//
// This module is intentionally PURE — it imports only types + a date helper,
// never `obsidian` — so the vitest suite can import it directly without the
// Obsidian runtime shim.
//
// Supported operators (v1 — "proven port only"):
//   #tag, #parent/child            -> tags (hierarchical)
//   tag:#x | tag:x                 -> tags
//   path:pattern                   -> includePaths (fnmatch-style glob)
//   [key:value]                    -> frontmatter match, Obsidian-style:
//                                     substring + case-insensitive + wikilink-aware
//   [key:"value"]                  -> frontmatter whole-value exact match (quoted)
//   [created:>YYYY-MM-DD] / [<…]   -> createdAfter / createdBefore
//   [modified:>…] / [<…]           -> modifiedAfter / modifiedBefore
//   -term                          -> exclude (note-level negation, Obsidian `-`)
// Matched tokens are STRIPPED from the text and NOT re-injected: the residual
// is pure semantic content for the embedder, and tag hierarchy is enforced by
// the filter layer, not by lossy slash->space flattening (removed 2026-05-16).
//
// Negation note: only BARE-WORD `-term` is supported. It mirrors Obsidian's
// native `-` (https://help.obsidian.md/plugins/search): excludes the whole FILE
// when the term appears in it. Matching is on Seek's own token notion (lowercase
// + the BM25 stoplist), so `-cat` excludes the word, never the substring inside
// "category"; a stop-word like `-the` is a no-op. Application is note-level and
// lives in search() (compileMatcher is metadata-only); see excludedNotePaths().
//
// Deferred (treated as plain text in v1): negated operators/phrases/groups
// (`-#tag`, `-"phrase"`, `-(a b)`, `-path:`), `file:`, phrase "…", boolean OR /
// grouping, `match-case:`, line/block/section/task, `/regex/`.

import type { ChunkMeta, QueryFilters } from './types';
import { parseDateMs } from './fusion';
import { toBindForm } from './prop-normalize';

// One character allowed inside an Obsidian tag or property key: any NON-delimiter.
// Mirrors Obsidian's own tag grammar (https://obsidian.md/help/tags) — which is
// defined by exclusion, not an allow-list, so it admits letters of every script,
// digits, AND emoji/symbols in a single rule. Excluded: whitespace, the Unicode
// General (U+2000–206F) + Supplemental (U+2E00–2E7F) Punctuation blocks, the
// ASCII punctuation that terminates a tag, and `/` (the nesting separator, which
// we match BETWEEN segments). `-` and `_` are deliberately kept. So #café,
// #日本語, #🎉, kebab `#meeting-prep`, and [my-field:…] all bind. (Obsidian also
// forbids purely-numeric tags like #1984; harmless to over-accept here — such a
// filter just matches no real tag.)
const TAG_CH = "[^\\s!-,./:-@\\[-\\^`{-~\\u2000-\\u206F\\u2E00-\\u2E7F]";
const TAG_RUN = `${TAG_CH}+(?:/${TAG_CH}+)*`; // hierarchical: parent/child/...

// Ported from search.py:25-30, then widened to Obsidian's full grammar.
// Alternatives are tried left-to-right at each position, so the bracket and
// `tag:`/`path:` prefixed forms win before a bare `#` match. Named groups + the
// `g`/`u` flags (u = codepoint-aware so astral emoji match as one unit).
//   path: takes a quoted form ("…") so paths with spaces bind — Obsidian's
//   path:"Daily notes/2022-07"; the bare \S+ form covers everything else.
const INLINE_FILTER_RE = new RegExp(
    `\\[(?<bkey>${TAG_CH}+)\\s*:\\s*(?<bop>[<>]?)\\s*(?<bval>[^\\]]+?)\\s*\\]` +
    `|tag:(?<texp>#?${TAG_RUN})` +
    `|path:(?<ipath>"[^"]+"|\\S+)` +
    `|#(?<thash>${TAG_RUN})`,
    'gu',
);

// Frontmatter keys that become top-level date filters when used with `>`/`<`.
const DATE_KEYS = new Set(['created', 'modified']);

// Bare-word negation: a `-` that STARTS a token (preceded by start-of-string or
// whitespace) and is followed by a non-space run. The lookbehind keeps mid-word
// hyphens inert ("covid-19", "well-known" never match), so only a leading dash
// is an exclusion — matching Obsidian's `-term`. Operator-like negations
// (`-#x`, `-[k:v]`, `-path:…`, `-tag:…`) are left as text in v1 (see header).
const NEGATION_RE = /(?<=^|\s)-(\S+)/g;

// Lucene/Elasticsearch `_english_` 33-word stoplist — kept in sync with
// bm25.ts ENGLISH_STOPWORDS (duplicated, not imported, to keep this module
// pure: it must import only types + a date helper so the vitest suite needs no
// MiniSearch/Obsidian shim). Negation tokenization drops these so `-the` is a
// no-op, exactly as the BM25 analyzer would treat it.
const MATCH_STOPWORDS = new Set<string>([
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'for', 'if', 'in',
    'into', 'is', 'it', 'no', 'not', 'of', 'on', 'or', 'such', 'that', 'the',
    'their', 'then', 'there', 'these', 'they', 'this', 'to', 'was', 'will', 'with',
]);

// Lowercase + split on non-(letter|number|underscore), drop empties and
// stop-words. Approximates MiniSearch's default analyzer closely enough that
// `-term` excludes the same notes BM25 would consider to contain `term`.
// Unicode-aware (\p{L}\p{N}) so non-ASCII words tokenize too.
function tokenizeForMatch(text: string): string[] {
    const out: string[] = [];
    for (const raw of text.toLowerCase().split(/[^\p{L}\p{N}_]+/u)) {
        if (raw && !MATCH_STOPWORDS.has(raw)) out.push(raw);
    }
    return out;
}

interface ParsedGroups {
    bkey?: string;
    bop?: string;
    bval?: string;
    texp?: string;
    ipath?: string;
    thash?: string;
}

/**
 * Extract inline filter syntax from a raw query.
 *
 * Returns the cleaned (operator-stripped) semantic text and the structured
 * filters. When no operators are found, `filters` is null and `cleanedQuery`
 * is the trimmed original — byte-identical search behavior to a no-parser build.
 */
export function parseQuery(raw: string): { cleanedQuery: string; filters: QueryFilters | null } {
    const tags: string[] = [];
    const frontmatter: Record<string, string> = {};
    const includePaths: string[] = [];
    const exclude: string[] = [];
    let createdAfter: string | null = null;
    let createdBefore: string | null = null;
    let modifiedAfter: string | null = null;
    let modifiedBefore: string | null = null;

    // Negation pass FIRST, so `-term` is consumed before the inline-filter
    // regex sees the residual text. A captured run that looks like a negated
    // operator (`-#x`, `-[k:v]`, `-path:…`, anything with `:`) is left in place
    // as plain text — v1 only handles bare-word exclusion (see header).
    // True once the negation pass strips a DEFERRED operator form (no exclude
    // token produced). It means the text was transformed even though no filter
    // was extracted, so the no-op early-return below must not resurrect the raw.
    let droppedDeferredNegation = false;
    const afterNegation = raw.replace(NEGATION_RE, (match: string, term: string): string => {
        // Negated operator/phrase forms (`-#tag`, `-[k:v]`, `-path:…`, anything
        // with `:`) are deferred. DROP them rather than leaving them as text —
        // otherwise the inline-filter regex below would re-parse the bare
        // `#tag`/`path:…` as a POSITIVE filter, silently inverting the intent.
        if (term.startsWith('#') || term.startsWith('[') || term.includes(':')) {
            droppedDeferredNegation = true;
            return ' ';
        }
        const toks = tokenizeForMatch(term);
        if (toks.length === 0) return match; // pure stop-word / punctuation → leave as text
        for (const t of toks) exclude.push(t);
        return ' ';
    });

    const stripped = afterNegation.replace(INLINE_FILTER_RE, (...args: unknown[]): string => {
        // With named groups, the final replace() argument is the groups object.
        const groups = args[args.length - 1] as ParsedGroups;
        const { bkey, bop, bval, texp, ipath, thash } = groups;

        if (bkey !== undefined && bval !== undefined) {
            const value = bval.trim();
            const keyLower = bkey.toLowerCase();
            if (DATE_KEYS.has(keyLower) && (bop === '<' || bop === '>')) {
                // Date pseudo-keys are case-insensitive ([Created:>X] == [created:>X]).
                if (keyLower === 'created') {
                    if (bop === '>') createdAfter = value; else createdBefore = value;
                } else {
                    if (bop === '>') modifiedAfter = value; else modifiedBefore = value;
                }
            } else {
                // Frontmatter exact-match: preserve key case so camelCase fields
                // like `pageType` resolve correctly against the metadata index.
                frontmatter[bkey] = value;
            }
            return ' ';
        }
        if (texp !== undefined) { tags.push(texp.replace(/^#/, '')); return ' '; }
        if (ipath !== undefined) {
            // Strip the optional surrounding quotes (the spaces-allowed form).
            const p = ipath.length >= 2 && ipath.startsWith('"') && ipath.endsWith('"')
                ? ipath.slice(1, -1) : ipath;
            includePaths.push(p);
            return ' ';
        }
        if (thash !== undefined) { tags.push(thash); return ' '; }
        return args[0] as string; // unreachable: some alternative always matched
    });

    const cleanedQuery = stripped.replace(/\s+/g, ' ').trim();

    const nothingExtracted =
        tags.length === 0 &&
        Object.keys(frontmatter).length === 0 &&
        includePaths.length === 0 &&
        exclude.length === 0 &&
        createdAfter === null && createdBefore === null &&
        modifiedAfter === null && modifiedBefore === null;

    if (nothingExtracted) {
        // No filters. Preserve the raw text byte-for-byte (no-parser-identical)
        // UNLESS the negation pass dropped a deferred `-operator` — then return
        // the cleaned residual so the dropped token doesn't leak back as text.
        return { cleanedQuery: droppedDeferredNegation ? cleanedQuery : raw.trim(), filters: null };
    }

    const filters: QueryFilters = {
        tags: tags.length ? tags : null,
        tagsMatchAll: false, // reserved; always OR in v1
        frontmatter: Object.keys(frontmatter).length ? frontmatter : null,
        includePaths: includePaths.length ? includePaths : null,
        createdAfter,
        createdBefore,
        modifiedAfter,
        modifiedBefore,
        exclude: exclude.length ? exclude : null,
    };
    return { cleanedQuery, filters };
}

// fnmatch-equivalent glob → anchored, case-insensitive RegExp. `*` matches any
// run (including `/`, matching Python fnmatch); `?` matches one char. All other
// regex metacharacters are escaped. Case-insensitive to mirror macOS fnmatch.
function globToRegExp(glob: string): RegExp {
    let pattern = '';
    for (const ch of glob) {
        if (ch === '*') pattern += '.*';
        else if (ch === '?') pattern += '.';
        else pattern += ch.replace(/[.+^${}()|[\]\\]/g, '\\$&');
    }
    return new RegExp(`^${pattern}$`, 'i');
}

// Normalize a frontmatter value for Obsidian-style property matching: this is
// the BIND-form surface (toBindForm) — lowercase, with `[[`, `]]`, and the
// alias `|` all flattened to spaces so the link TARGET, its PATH, and its
// ALIAS all stay substring-matchable (a stored `[[Los Angeles]]` matches a
// typed `Los Angeles`; `[[.../Places/Zurich]]` matches both `Places` and
// `Zurich`). Mirrors how Obsidian's native property search sees through
// wikilink syntax. Distinct from the DISPLAY-form used for indexing — see
// [[Seek Index Processing Audit]] / prop-normalize.ts.
function normalizePropValue(s: string): string {
    return toBindForm(s);
}

/**
 * Compile a QueryFilters into a fast per-chunk predicate. Globs, lowercased
 * filter tags, and parsed date bounds are computed ONCE here; the returned
 * closure is the hot path applied across every chunk to build the match-mask.
 * Ported from _apply_chunk_filters (search.py:565-657); frontmatter matching
 * diverges from the port toward Obsidian semantics (substring/wikilink/quoted).
 */
export function compileMatcher(f: QueryFilters): (chunk: ChunkMeta) => boolean {
    // Each include pattern matches the full path OR a `*/pattern` suffix, and
    // any one pattern matching is enough (OR across patterns).
    const pathRes = (f.includePaths ?? []).flatMap(p => [globToRegExp(p), globToRegExp(`*/${p}`)]);
    const filterTags = (f.tags ?? []).map(t => t.replace(/^#/, '').toLowerCase());
    // Obsidian-style property matching, precompiled once per query: substring +
    // case-insensitive + wikilink-aware by DEFAULT; a double-quoted value
    // (`[key:"v"]`) forces a whole-value exact match. The quote check runs on the
    // raw value before normalization so the literal `"` delimiters are visible.
    const fmMatchers = (f.frontmatter ? Object.entries(f.frontmatter) : []).map(([k, v]) => {
        const raw = String(v).trim();
        const exact = raw.length >= 2 && raw.startsWith('"') && raw.endsWith('"');
        return { key: k, expected: normalizePropValue(exact ? raw.slice(1, -1) : raw), exact };
    });
    const cAfter = f.createdAfter ? parseDateMs(f.createdAfter) : null;
    const cBefore = f.createdBefore ? parseDateMs(f.createdBefore) : null;
    const mAfter = f.modifiedAfter ? parseDateMs(f.modifiedAfter) : null;
    const mBefore = f.modifiedBefore ? parseDateMs(f.modifiedBefore) : null;

    const tagMatches = (chunkTags: string[], ft: string): boolean =>
        chunkTags.some(ct => ct === ft || ct.startsWith(ft + '/'));

    return (chunk: ChunkMeta): boolean => {
        const meta = chunk.metadata;
        const notePath = chunk.note_path ?? '';

        // Path inclusion (glob).
        if (pathRes.length > 0 && !pathRes.some(re => re.test(notePath))) return false;

        // Tag filtering (hierarchical: "meetings" matches "meetings/1x1").
        if (filterTags.length > 0) {
            const chunkTags = (meta.tags ?? []).map(t => t.replace(/^#/, '').toLowerCase());
            if (f.tagsMatchAll) {
                if (!filterTags.every(ft => tagMatches(chunkTags, ft))) return false;
            } else {
                if (!filterTags.some(ft => tagMatches(chunkTags, ft))) return false;
            }
        }

        // Frontmatter (Obsidian-style: substring + wikilink-aware by default,
        // `"quoted"` = whole-value exact). Missing key → reject.
        for (const { key, expected, exact } of fmMatchers) {
            const actual = meta.properties?.[key];
            if (actual === undefined) return false;
            const actNorm = normalizePropValue(actual);
            if (exact ? actNorm !== expected : !actNorm.includes(expected)) return false;
        }

        // Date filtering. Missing/unparseable date → reject (matches reference).
        if (cAfter !== null || cBefore !== null) {
            const dt = parseDateMs(meta.created);
            if (dt === null) return false;
            if (cAfter !== null && dt < cAfter) return false;
            if (cBefore !== null && dt > cBefore) return false;
        }
        if (mAfter !== null || mBefore !== null) {
            const dt = parseDateMs(meta.modified);
            if (dt === null) return false;
            if (mAfter !== null && dt < mAfter) return false;
            if (mBefore !== null && dt > mBefore) return false;
        }

        return true;
    };
}

/** Convenience single-shot matcher (compiles per call) — for tests/ad-hoc use.
 *  In the search hot path use compileMatcher() once and reuse the closure. */
export function matchesFilters(chunk: ChunkMeta, f: QueryFilters): boolean {
    return compileMatcher(f)(chunk);
}

/**
 * Resolve `-term` negation to the set of note paths to drop. Obsidian's `-`
 * excludes the whole FILE when a term appears in it, so this works note-level:
 * one scan over every chunk, and the moment any chunk of a note contains any
 * excluded token, that note's path is excluded (and its other chunks short-
 * circuit). Matching is on Seek's own token notion (tokenizeForMatch — lowercase
 * + stoplist), against the chunk's title + body (the text a reader sees), so
 * `-cat` drops notes with the word `cat`, never ones that merely contain
 * "category". `excludeTokens` are the already-normalized tokens from parseQuery.
 *
 * Body text is supplied via `getBody(chunkId)` (v8 frame-lite: the frame is
 * metadata-only). The caller fetches the corpus bodies — paid ONLY on negation
 * queries; the per-note short-circuit keeps multi-chunk notes cheap. Called from
 * search() to fold the result into the same selection mask the inline filters build.
 */
export function excludedNotePaths(
    chunks: ChunkMeta[],
    excludeTokens: string[],
    getBody: (chunkId: string) => string | undefined,
): Set<string> {
    const excluded = new Set<string>();
    if (excludeTokens.length === 0) return excluded;
    const terms = new Set(excludeTokens);
    for (const c of chunks) {
        const path = c.note_path ?? '';
        if (excluded.has(path)) continue; // a sibling chunk already excluded this note
        const text = `${c.title ?? ''}\n${getBody(c.chunk_id) ?? ''}`;
        for (const tok of tokenizeForMatch(text)) {
            if (terms.has(tok)) { excluded.add(path); break; }
        }
    }
    return excluded;
}
