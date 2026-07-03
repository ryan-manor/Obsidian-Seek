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
//   [key>n] [key<n] [key=n]        -> numeric comparison (colon optional), but ONLY
//                                     when `key` is a Number-typed property (ctx);
//                                     value-inclusive. Operator on a non-Number key
//                                     -> numericTypeMismatch (unsatisfiable, D3).
//   after:DATE / before:DATE       -> dateAfter / dateBefore (day-inclusive), keyed
//                                     off the Recency date field; recognized only
//                                     when Recency is ON (ctx.dateField present).
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

import type { ChunkMeta, QueryFilters, FilterContext, RecencyKeyChoice } from './types';
import { parseDateMs } from './fusion';
import { toBindForm } from './prop-normalize';

// Shared query-side tokenizer for `-term` negation (see tokenizeForMatch
// below): the same seekTokenize + processQueryTerm pipeline BM25 itself uses
// to process a query (bm25.ts distinctQueryTerms), not an ad-hoc duplicate.
import { seekTokenize } from './tokenize';
import { processQueryTerm } from './bm25';

// Tag/property-key character grammar — SHARED with the doc-side inline-tag
// scanner (tag-grammar.ts, where the full derivation comment lives): both
// sides binding the same characters is what keeps a suggested/typed `#tag`
// filter able to match what the chunker extracted (audit R2 #2).
import { TAG_CH, TAG_RUN } from './tag-grammar';

// Ported from search.py:25-30, then widened to Obsidian's full grammar.
// Alternatives are tried left-to-right at each position, so the bracket and
// `tag:`/`path:` prefixed forms win before a bare `#` match. Named groups + the
// `g`/`u` flags (u = codepoint-aware so astral emoji match as one unit).
//   path: takes a quoted form ("…") so paths with spaces bind — Obsidian's
//   path:"Daily notes/2022-07"; the bare \S+ form covers everything else.
// Two bracket alternatives, comparison tried first:
//   COMPARISON `[key>n]` `[key < n]` `[key:>n]` — an operator is REQUIRED and the
//     colon is OPTIONAL (so the readable colon-less form binds). TAG_CH excludes
//     `<`/`=`/`>`, so `key` stops cleanly before the operator. Routes to the numeric
//     path (or numericTypeMismatch) — never substring.
//   SUBSTRING `[key:value]` — a colon is REQUIRED and there is no operator. The
//     unchanged Obsidian-style frontmatter match. Reached only when the comparison
//     branch fails (no operator right after the key/colon), so `[context:work]` and
//     `[note:a>b]` (operator not adjacent) stay substring.
// after:/before: are bare prefixed date operators (like path:/tag:); a quoted form
// is accepted for symmetry though dates carry no spaces.
//
// Leading token boundary `(?:^|\s)` (audit R2 #9, same fix shape as
// NEGATION_RE below) guards only the BARE-PREFIX alternatives — `tag:`,
// `path:`, `after:`, `before:`, `#tag` — because those are the ones a
// mid-word substring can spoof: without it, a pasted URL's trailing
// `#fragment` parsed as a `#tag` filter, and typing "montag:meeting" bound a
// `tag:` filter on "meeting". Non-capturing so the named groups'
// indices/positions are untouched; `\s` (not `\s*`) consumes exactly one
// boundary char, mirrored by the callback returning a single space for a
// real match, or the untouched `args[0]` (the full match, boundary included)
// for a no-op fallthrough — either way behavior is identical to before the
// boundary was added.
//   The two `[key:value]`/`[key>n]` bracket alternatives deliberately stay
// OUTSIDE the boundary group: the leading `[` is already a self-delimiting
// signal (no mid-word-substring risk the way `tag:` or `#` have), so they
// never needed it. Originally (audit R2 #9) the boundary wrapped the WHOLE
// alternation, including the brackets — that over-application is what broke
// two filter tokens placed with zero separating whitespace (e.g.
// `[a:b][c:d]`, `#tag[key:value]`): the FIRST match consumed its boundary
// (or matched at `^` with none to give back), so the very next `[`/`#`/etc.
// sat at a position with no whitespace before it and the shared boundary
// rejected it, silently dropping every filter after the first (audit R2
// review-2 #1, verified via `parseQuery('[context:work][pageType:task]')`
// returning only the first filter). Scoping the boundary to just the
// literal-prefix alternatives restores pre-R2-#9 bracket-to-bracket (and
// prefix-to-bracket) adjacency while keeping the mid-word guard where it's
// actually needed.
const INLINE_FILTER_RE = new RegExp(
    `\\[(?<ckey>${TAG_CH}+)\\s*(?::\\s*)?(?<cop>[<>=])\\s*(?<cval>[^\\]]+?)\\s*\\]` +
    `|\\[(?<skey>${TAG_CH}+)\\s*:\\s*(?<sval>[^\\]]+?)\\s*\\]` +
    `|(?:^|\\s)(?:tag:(?<texp>#?${TAG_RUN})` +
    `|path:(?<ipath>"[^"]+"|\\S+)` +
    `|after:(?<dafter>"[^"]+"|\\S+)` +
    `|before:(?<dbefore>"[^"]+"|\\S+)` +
    `|#(?<thash>${TAG_RUN}))`,
    'gu',
);

// Coerce a stored property value to a finite number, defensively — frontmatter is
// stored as strings and real data is messy (quoted "299.00", empty, null, even
// malformed "197, 344, 218"). A non-number yields null, which the matcher treats
// as "does not match" — so a numeric comparison is robust even if a value slips
// past type detection. Mirrors parseDateMs's quote-stripping posture.
// Exported so the pill UI (query-field.ts) can validate a numeric-comparison
// value with the SAME parser this module's own comparison branch (below) and
// compileMatcher's numeric clause use — a single source of truth, so a value
// that fails to parse here also renders as the red error-pill (audit R2 #10).
export function parseNum(raw: string | null | undefined): number | null {
    const cleaned = String(raw ?? '').trim().replace(/^["']+|["']+$/g, '').trim();
    if (!cleaned) return null;
    const n = Number(cleaned);          // strict: "197, 344, 218" → NaN
    return Number.isFinite(n) ? n : null;
}

const ONE_DAY_MS = 86_400_000;

// Inclusive upper bound for `before:D`: the first instant AFTER the whole period
// D names, so everything up to and including that period is kept. Granularity
// follows what the user typed — a bare year covers the year, YYYY-MM the month,
// YYYY-MM-DD the day. LOCAL-anchored (via the Date constructor's local
// year/month/day fields, which also normalizes overflow — e.g. day 32 or
// month 12 rolls into the next period for free) to match parseDateMs's
// local-midnight treatment of bare dates, so both sides of the comparison
// live in one frame — a UTC anchor here would roll boundary-day evening
// events out of range for anyone west of UTC. Returns null when D doesn't
// parse.
function endBoundMs(d: string): number | null {
    const cleaned = String(d).trim().replace(/^["'{]+|["'}]+$/g, '').trim();
    let m = /^(\d{4})$/.exec(cleaned);
    if (m) return new Date(+m[1] + 1, 0, 1).getTime();
    m = /^(\d{4})-(\d{2})$/.exec(cleaned);
    if (m) return new Date(+m[1], +m[2], 1).getTime();
    m = /^(\d{4})-(\d{2})-(\d{2})/.exec(cleaned);
    if (m) {
        // Validate the SYNTACTIC range (month 1-12, day 1-31) before handing
        // off to the numeric-args Date constructor, which normalizes rather
        // than rejects out-of-range fields (month 13 silently rolls into next
        // January) — unlike the old Date.parse-based code this replaced. The
        // `+1` below is the deliberate exclusive-bound arithmetic (day 31 + 1
        // = 32, which legitimately rolls into next month "for free" per the
        // doc comment above) and stays untouched — only the raw typed-in
        // month/day get validated.
        const mo = +m[2], day = +m[3];
        if (mo < 1 || mo > 12 || day < 1 || day > 31) return null;
        return new Date(+m[1], mo - 1, day + 1).getTime();
    }
    const base = parseDateMs(cleaned); // a parseable datetime → +1 day from its instant
    return base === null ? null : base + ONE_DAY_MS;
}

// The date a date-FILTER reads for a chunk: the user's selected Recency field,
// then mtime — and nothing else. Deliberately NARROWER than fusion.ts recencyDate
// (the RANKING ladder), which also falls back to a filename date: parsing a date
// out of a filename is too vault-specific to be a safe FILTER behavior (D4).
// Missing both the selected property and mtime → null → the chunk is rejected.
function filterDate(
    meta: ChunkMeta['metadata'] | undefined,
    dateField: { key: RecencyKeyChoice; createdProp: string },
): number | null {
    const rawProp = dateField.createdProp === 'created' ? meta?.created : meta?.properties?.[dateField.createdProp];
    const prop = dateField.key === 'modified'
        ? meta?.modified
        : (Array.isArray(rawProp) ? rawProp[0] : rawProp);
    return parseDateMs(prop ?? meta?.modified);
}

// Default date field for the permissive path (no ctx supplied): this vault's
// `created` convention. The real search path always passes ctx.dateField.
const DEFAULT_DATE_FIELD = { key: 'created' as RecencyKeyChoice, createdProp: 'created' };

// Bare-word negation: a `-` that STARTS a token (preceded by start-of-string or
// whitespace) and is followed by a non-space run. The leading `(?:^|\s)` boundary
// keeps mid-word hyphens inert ("covid-19", "well-known" never match), so only a
// leading dash is an exclusion — matching Obsidian's `-term`. Operator-like
// negations (`-#x`, `-[k:v]`, `-path:…`, `-tag:…`) are left as text in v1 (see
// header). NON-capturing on the boundary (not a lookbehind): `(?<=…)` throws at
// parse time on iOS WKWebView before 16.4 and would take the whole module down.
// `(?:^|\s)` consumes one leading whitespace, which the callback replaces with a
// single space anyway, so behaviour is identical and the term stays group 1.
const NEGATION_RE = /(?:^|\s)-(\S+)/g;

// Negation tokenization routes through the SAME shared pipeline BM25 query
// terms use (seekTokenize, default derived:true, then processQueryTerm — see
// bm25.ts distinctQueryTerms, which follows the identical shape) instead of an
// ad-hoc fourth tokenizer. The old bare `text.toLowerCase().split(/[^\p{L}\p{N}_]+/u)`
// + a locally-duplicated stoplist had none of seekTokenize's CJK dictionary
// segmentation or processQueryTerm's diacritic-fold/depluralize — so `-cat`
// never suppressed `cats` (no depluralize) and non-Latin negated terms didn't
// fold correctly (no CJK segmentation). Reusing the real pipeline fixes both
// and retires the duplicated stoplist (audit R2 #11). This is used
// symmetrically on BOTH sides of a negation: extracting the excluded token(s)
// from `-term` below, and scanning the candidate note's title+body in
// excludedNotePaths() — so as long as both calls route through the one
// function they can never drift from each other, and now also can't drift
// from what BM25 itself would consider a match.
function tokenizeForMatch(text: string): string[] {
    const out: string[] = [];
    for (const raw of seekTokenize(text)) {
        const t = processQueryTerm(raw);
        if (t) out.push(t);
    }
    return out;
}

interface ParsedGroups {
    ckey?: string;   // comparison: [key>n]
    cop?: string;
    cval?: string;
    skey?: string;   // substring: [key:value]
    sval?: string;
    texp?: string;
    ipath?: string;
    dafter?: string;  // after:DATE
    dbefore?: string; // before:DATE
    thash?: string;
}

/**
 * Extract inline filter syntax from a raw query.
 *
 * Returns the cleaned (operator-stripped) semantic text and the structured
 * filters. When no operators are found, `filters` is null and `cleanedQuery`
 * is the trimmed original — byte-identical search behavior to a no-parser build.
 */
export function parseQuery(raw: string, ctx?: FilterContext): { cleanedQuery: string; filters: QueryFilters | null } {
    const tags: string[] = [];
    const frontmatter: Record<string, string> = {};
    const includePaths: string[] = [];
    const exclude: string[] = [];
    const numeric: Array<{ key: string; op: '<' | '>' | '='; value: number }> = [];
    const numericMismatch: string[] = [];
    let dateAfter: string | null = null;
    let dateBefore: string | null = null;

    // Type/field gates from the call site. No ctx ⇒ permissive (ad-hoc callers and
    // the vitest matcher tests): any key may compare, and bare after:/before: bind.
    // A provided ctx enforces the design's gates: numericKeys decides comparison vs
    // mismatch (D3), and a null dateField (Recency OFF) leaves after:/before: as text.
    const numericKeys = ctx?.numericKeys;
    const isNumericKey = (k: string): boolean => numericKeys ? numericKeys.has(k) : true;
    const dateEnabled = ctx === undefined || ctx.dateField !== null;
    const stripQuotes = (s: string): string =>
        s.length >= 2 && s.startsWith('"') && s.endsWith('"') ? s.slice(1, -1) : s;

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
        const { ckey, cop, cval, skey, sval, texp, ipath, dafter, dbefore, thash } = groups;

        // Comparison `[key>n]` — operator present, colon optional. Routes by the
        // key's DECLARED type (key case preserved for property lookup):
        if (ckey !== undefined && cop !== undefined && cval !== undefined) {
            const op = cop as '<' | '>' | '=';
            if (isNumericKey(ckey)) {
                const value = parseNum(cval);
                // A finite value is a real numeric filter; a non-number on a Number
                // key (`[price>abc]`) can't be honored → mismatch (0 results, D3).
                if (value !== null) numeric.push({ key: ckey, op, value });
                else numericMismatch.push(ckey);
            } else {
                // Operator on a non-Number property: unsatisfiable, never substring
                // (D3 — substring would be plausibly-but-silently wrong).
                numericMismatch.push(ckey);
            }
            return ' ';
        }
        // Substring `[key:value]` — preserve key case so camelCase fields like
        // `pageType` resolve correctly against the metadata index.
        if (skey !== undefined && sval !== undefined) {
            frontmatter[skey] = sval.trim();
            return ' ';
        }
        if (texp !== undefined) { tags.push(texp.replace(/^#/, '')); return ' '; }
        if (ipath !== undefined) {
            includePaths.push(stripQuotes(ipath));
            return ' ';
        }
        // after:/before: bind only when Recency is ON (a date field exists to key
        // off) AND the value parses as a date — validated with the SAME function
        // the compiled matcher will use, so bind and bound can't drift. An
        // unparseable value (`after:yesterday`) previously bound anyway: the
        // token was stripped from the text AND compiled to a null bound — the
        // filter silently vanished from BOTH channels. Verbatim fallthrough
        // (the Recency-OFF arm) keeps it searchable text instead. (The pill UI
        // already declines these at commit — this closes the raw-string inlets:
        // CLI, pasted text.)
        if (dafter !== undefined) {
            if (!dateEnabled) return args[0] as string;
            const v = stripQuotes(dafter).trim();
            if (parseDateMs(v) === null) return args[0] as string;
            dateAfter = v;
            return ' ';
        }
        if (dbefore !== undefined) {
            if (!dateEnabled) return args[0] as string;
            const v = stripQuotes(dbefore).trim();
            if (endBoundMs(v) === null) return args[0] as string;
            dateBefore = v;
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
        numeric.length === 0 && numericMismatch.length === 0 &&
        dateAfter === null && dateBefore === null;

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
        numeric: numeric.length ? numeric : null,
        dateAfter,
        dateBefore,
        numericTypeMismatch: numericMismatch.length ? numericMismatch : null,
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

// Case-insensitive property lookup with an exact-case fast path. Property maps
// are small (a handful of keys), so the fallback scan is cheap and only runs on
// queries that carry property filters at all.
function lookupProp(
    props: Record<string, string | string[]> | undefined,
    key: string,
): string | string[] | undefined {
    if (!props) return undefined;
    if (props[key] !== undefined) return props[key];
    const lk = key.toLowerCase();
    for (const k of Object.keys(props)) {
        if (k.toLowerCase() === lk) return props[k];
    }
    return undefined;
}

/**
 * Compile a QueryFilters into a fast per-chunk predicate. Globs, lowercased
 * filter tags, numeric clauses, and parsed date bounds are computed ONCE here;
 * the returned closure is the hot path applied across every chunk to build the
 * match-mask. Ported from _apply_chunk_filters (search.py:565-657); frontmatter
 * matching diverges from the port toward Obsidian semantics. `ctx.dateField`
 * resolves which per-chunk date the date filter reads (defaults to `created`).
 */
export function compileMatcher(f: QueryFilters, ctx?: FilterContext): (chunk: ChunkMeta) => boolean {
    // D3: a comparison on a non-Number key makes the whole query unsatisfiable —
    // every chunk is rejected, never substring-matched. Short-circuit the closure.
    if (f.numericTypeMismatch && f.numericTypeMismatch.length > 0) return () => false;

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
    const numericClauses = f.numeric ?? [];
    // Date bounds: `after:D` lower bound = start-of-period(D) (parseDateMs already
    // floors a bare year/month/day to its start); `before:D` upper bound is the
    // first instant AFTER D's whole period, so it's exclusive at the top → both
    // bounds day-INCLUSIVE. The date field is the user's Recency selection.
    const afterMs = f.dateAfter ? parseDateMs(f.dateAfter) : null;
    const beforeBoundMs = f.dateBefore ? endBoundMs(f.dateBefore) : null;
    const dateField = ctx?.dateField ?? DEFAULT_DATE_FIELD;

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
        // `"quoted"` = whole-value exact). Missing key → reject. Key resolution
        // is case-INSENSITIVE (exact case fast-path first): Obsidian property
        // names are display labels ('pageType'), and a hand-typed `[pagetype:…]`
        // used to reject every chunk with zero signal. List-valued properties
        // match if ANY element matches (Obsidian's own list-property semantics).
        for (const { key, expected, exact } of fmMatchers) {
            const actual = lookupProp(meta.properties, key);
            if (actual === undefined) return false;
            const values = Array.isArray(actual) ? actual : [actual];
            const hit = values.some(v => {
                const actNorm = normalizePropValue(v);
                return exact ? actNorm === expected : actNorm.includes(expected);
            });
            if (!hit) return false;
        }

        // Numeric comparison (value-inclusive). Missing key / non-numeric value →
        // reject (consistent with "missing field → reject"; robust to bad data).
        // Case-insensitive key resolution as above; a list value satisfies the
        // clause if ANY element does.
        for (const { key, op, value } of numericClauses) {
            const actual = lookupProp(meta.properties, key);
            const values = Array.isArray(actual) ? actual : [actual];
            const hit = values.some(v => {
                const n = parseNum(v);
                if (n === null) return false;
                return op === '>' ? n >= value : op === '<' ? n <= value : n === value;
            });
            if (!hit) return false;
        }

        // Date filtering (day-inclusive). Missing/unparseable date → reject.
        if (afterMs !== null || beforeBoundMs !== null) {
            const dt = filterDate(meta, dateField);
            if (dt === null) return false;
            if (afterMs !== null && dt < afterMs) return false;
            if (beforeBoundMs !== null && dt >= beforeBoundMs) return false;
        }

        return true;
    };
}

/** Convenience single-shot matcher (compiles per call) — for tests/ad-hoc use.
 *  In the search hot path use compileMatcher() once and reuse the closure. */
export function matchesFilters(chunk: ChunkMeta, f: QueryFilters, ctx?: FilterContext): boolean {
    return compileMatcher(f, ctx)(chunk);
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
