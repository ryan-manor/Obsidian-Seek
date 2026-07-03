// Inline filter-completion engine for the search modal.
//
// Powers the ghost-text typeahead: as the user types a filter operator, the
// top matching value is shown as a greyed suffix that Tab accepts. ONLY the
// operators that parseQuery (query-parser.ts) actually understands are
// completed — tag:/#, path:, [key:value], and the [ key-menu — so an accepted
// suggestion always produces a filter that binds. Free-text (bare-word)
// completion is deliberately NOT offered (the BM25 term index yields noisy,
// non-display-grade tokens; see Seek UX.md "Typeahead / Autosuggest").
//
// All suggestion values come from a distinct-value dictionary built from
// app.metadataCache, NOT from MiniSearch: categorical filter values are whole
// strings, and a term index shreds them (meetings/1x1s -> meetings + 1x1s).
//
// Completion is prefix-only by construction, but the prefix test is
// CASE-INSENSITIVE — the matcher binds case-insensitively, so the suggester
// must too (`[placelo` surfaces `placeLoc:`). Ghost text additionally needs
// the accept string to literally extend the typed value (you can't recolour
// already-typed text), so a case-corrected match surfaces as a dropdown row /
// Tab-accept with an empty ghost. That keeps the busiest-first progressive
// model (tag:m -> tag:meetings -> tag:meetings/1x1s, one Tab at a time).

import type { App, TFile } from 'obsidian';
import { toDisplayForm } from './prop-normalize';
import { enumerateDatePropertyNames, enumerateNumberPropertyNames } from './prop-types';
import { shouldIndexPath } from './search';
import type { SeekSettings } from './types';

export interface Completion {
    // What the input's full value becomes if accepted (head + completed token).
    accept: string;
    // The muted suffix shown after the current value: accept.slice(value.length).
    ghost: string;
    kind: 'tag' | 'path' | 'field' | 'fieldkey';
    // Human label for logging/tooltip, e.g. "meetings/1x1s · 344 notes".
    label: string;
}

// One row in the suggestion dropdown (the pill query field's menu). Extends a
// Completion — the full accept-string + ghost are still useful for ghost-text —
// with the BARE value the caller commits into a filter pill: a tag without its
// sigil ("meetings/1x1s"), a path glob without `path:`/quotes ("Work/Strategy/*"),
// or an `[key:value]` body. `count` is the backing note count, surfaced as a hint
// and used to order the menu (busiest first).
export interface SuggestItem extends Completion {
    value: string;
    count: number;
}

// Frontmatter keys never offered as [key:value] filters. Two residual reasons
// only: keys with their own operator (tags/aliases), and universal UI/identity
// machinery that is neither type- nor cardinality-detectable (cssclasses can be
// low-cardinality, so the cap won't catch it). Dates are NOT named here — they
// are excluded generically in build() via Obsidian's type registry
// (enumerateDatePropertyNames) PLUS a date-shaped-value guard, so no per-vault
// date key (created / modified / due / completedDate / dateLink / …) has to be
// hardcoded. Per-note-unique free text (week, relatedPages, …) is left to the
// cardinality cap below — categorical keys survive it, unique ones don't.
const KEY_BLOCKLIST = new Set([
    'position', 'tags', 'aliases', 'cssclass', 'cssclasses',
    'title', 'id', 'uuid', 'permalink',
]);

// A key with more distinct values than this is dropped from completion UNLESS
// its values demonstrably repeat (total ≥ 2× distinct — see build()). The cap
// alone would also drop genuinely categorical wide keys (placeLoc: 50+ cities,
// heavy repeats); the real target is per-note-unique free text, where the
// distinct count tracks the note count.
const MAX_KEY_CARDINALITY = 40;
// Skip absurdly long values — they're prose, not categories, and make terrible
// ghost suffixes.
const MAX_VALUE_LEN = 40;

// A value opening with YYYY-MM-DD is a date, never a useful categorical pick
// (dates are queried via before:/after:/recency). Mirrors bm25.ts
// PROPERTY_DATE_RE — duplicated, not imported, to keep suggest's dependency
// surface light (same rationale as TAG_CH below). This is the by-VALUE backstop
// for date fields the user hasn't typed Date in Obsidian's registry, so the
// registry check and this guard together replace the old date NAME blocklist.
const VALUE_DATE_RE = /^\d{4}-\d{2}-\d{2}/;

// What the parser (query-parser.ts) can actually capture, mirrored here so the
// completion never offers a value that wouldn't bind. TAG_CH is the same
// non-delimiter character class as INLINE_FILTER_RE's TAG_CH — duplicated, not
// imported, because query-parser.ts is intentionally obsidian-free (same reason
// the stoplist is duplicated). It admits any letter/digit/emoji plus `_`/`-`, so
// kebab/unicode/emoji tags ARE offered; only genuine non-tag values (a space, the
// punctuation blocks) are filtered out. Keep in lockstep with INLINE_FILTER_RE.
const TAG_CH = "[^\\s!-,./:-@\\[-\\^`{-~\\u2000-\\u206F\\u2E00-\\u2E7F]";
const PARSER_TAG = new RegExp(`^${TAG_CH}+(?:/${TAG_CH}+)*$`, 'u');
const PARSER_KEY = new RegExp(`^${TAG_CH}+$`, 'u');

// Every tag surface-form a single note contributes to Obsidian's vault-wide
// aggregate (metadataCache.getTags()) — inline body '#tags' (already
// position-parsed by Obsidian, no I/O needed) unioned with frontmatter
// `tags:`. Sigil-stripped + lowercased so it can be compared against
// this.tags' keys (whose canonical casing, from getTags(), may differ from
// any one file's). Used only to attribute a tag to excluded vs. included
// notes below — not a full reimplementation of chunker.ts's extractTags
// (no casing preservation / de-dup needed for a membership check).
function fileTagKeys(cacheEntry: { tags?: Array<{ tag: string }>; frontmatter?: Record<string, unknown> } | null | undefined): Set<string> {
    const out = new Set<string>();
    for (const t of cacheEntry?.tags ?? []) out.add(t.tag.replace(/^#/, '').toLowerCase());
    const raw = cacheEntry?.frontmatter?.tags;
    const list = Array.isArray(raw) ? raw.map(String) : typeof raw === 'string' ? raw.split(/[,\s]+/) : [];
    for (const t of list) {
        const s = t.trim();
        if (s) out.add(s.replace(/^#/, '').toLowerCase());
    }
    return out;
}

interface ActiveToken {
    kind: 'tag' | 'path' | 'field' | 'fieldkey';
    from: number;        // index in the query where the active token starts
    partial: string;     // the value typed so far (after the operator prefix)
    sigil?: 'tag:' | '#'; // for tags, which form the user is typing
    key?: string;        // for fields, the frontmatter key typed before the colon
    quoted?: boolean;    // for paths: inside an open `path:"…` (spaces allowed)
}

export class SuggestEngine {
    // Surface forms WITHOUT a leading '#'. e.g. "meetings/1x1s" -> 430.
    private tags = new Map<string, number>();
    // Folder paths -> file count. e.g. "Notes/Work/Meetings" -> 420.
    private folders = new Map<string, number>();
    // Frontmatter key -> (value -> count), only low-cardinality keys.
    private fields = new Map<string, Map<string, number>>();
    // Completable keys, most-populated first (drives the [ menu order).
    private fieldKeys: string[] = [];
    // Total note-count per key, retained for EVERY key seen — including the
    // high-cardinality ones dropped from `fields`. Backs the `[` menu's
    // `· N notes` hint for numeric keys like `price` (192 near-unique values →
    // excluded from `fields`, so the old fields-only count read a misleading 0).
    private keyTotals = new Map<string, number>();
    // Properties declared Number in Obsidian's type registry. Offered in the `[`
    // key-menu REGARDLESS of cardinality (the categorical gate is for value
    // completion, which numerics don't do — they complete to an operator), so a
    // high-cardinality field like `price` reappears once it's typed Number. This
    // is the direct fix for the original `[price` autocomplete gap.
    private numericKeys = new Set<string>();
    // Properties declared Date / Date & time in Obsidian's type registry,
    // lowercased. Excluded from value completion entirely — a date is per-note
    // temporal data, queried via before:/after:/recency, never a categorical
    // pick. The VALUE_DATE_RE guard in build() additionally catches date fields
    // the user hasn't typed in the registry (e.g. a bare `due:`), so neither
    // path needs the date key NAMES the blocklist used to carry.
    private dateKeys = new Set<string>();

    // Build all dictionaries from the in-memory metadata cache. Cheap enough to
    // run on every modal open (a single pass over getMarkdownFiles, no I/O), so
    // suggestions always reflect the current vault.
    //
    // `settings`, when supplied, gates every candidate on the same
    // index-membership predicate the real indexer uses (shouldIndexPath in
    // search.ts — Obsidian's "Excluded files" setting, when honorIgnoredFolders
    // is on). Without it, app.metadataCache still carries notes from
    // index-excluded folders, so a completed `[key:value]` or `tag:`/`#` pill
    // built from one would always return 0 results — the note was never
    // chunked, so the matcher can never find it. Optional (not required) so
    // existing callers keep working unfiltered; the search modal passes it.
    build(app: App, settings?: SeekSettings): this {
        this.numericKeys = enumerateNumberPropertyNames(app);
        this.dateKeys = new Set(enumerateDatePropertyNames(app).map(k => k.toLowerCase()));
        const cache = app.metadataCache as unknown as { getTags?: () => Record<string, number> };
        for (const [t, c] of Object.entries(cache.getTags?.() ?? {})) {
            const tag = t.replace(/^#/, '');
            if (PARSER_TAG.test(tag)) this.tags.set(tag, c); // skip non-bindable tags (space/emoji/non-ASCII)
        }

        // getTags() is a vault-wide aggregate with no per-file breakdown, so a
        // tag sourced ONLY from index-excluded notes can't be subtracted from it
        // directly. Instead: track which tags occur on at least one INCLUDED
        // file (below) and which files were excluded, then after the loop drop
        // any tag that only ever showed up on an excluded file. Inert (and
        // cheap — one empty-array check) when nothing is excluded, which keeps
        // the common case byte-identical to the old unfiltered behavior.
        const excludedFiles: TFile[] = [];
        const includedTagKeys = new Set<string>();

        const rawFields = new Map<string, Map<string, number>>();
        for (const f of app.vault.getMarkdownFiles()) {
            if (settings && !shouldIndexPath(app, settings, f.path)) {
                excludedFiles.push(f);
                continue;
            }

            const folder = f.parent?.path;
            if (folder && folder !== '/') this.folders.set(folder, (this.folders.get(folder) ?? 0) + 1);

            const cacheEntry = app.metadataCache.getFileCache(f);
            for (const k of fileTagKeys(cacheEntry)) includedTagKeys.add(k);

            const fm = cacheEntry?.frontmatter;
            if (!fm) continue;
            for (const [k, v] of Object.entries(fm)) {
                // Skip own-operator/machinery keys (blocklist), registry-typed
                // Date keys (by type), and non-parser-capturable keys (\w+).
                if (KEY_BLOCKLIST.has(k) || this.dateKeys.has(k.toLowerCase()) || !PARSER_KEY.test(k)) continue;
                const add = (val: unknown) => {
                    // Coerce numbers/booleans the same way the index does
                    // (extractProperties String()s them), so completed:true /
                    // version:3 are offered AND bind at match time.
                    if (typeof val === 'number' || typeof val === 'boolean') val = String(val);
                    if (typeof val !== 'string') return;
                    // Offer the DISPLAY form — the wikilink target basename,
                    // case preserved (toDisplayForm): a stored `[[Austin]]` is
                    // offered as `Austin`, and `[[Notes/.../Zurich|Zurich]]` as
                    // `Zurich`, NOT the path-stuffed "Notes Personal Places
                    // Zurich Zurich" the old matcher-style unwrap produced. The
                    // offered value still BINDS: bind-form filter matching is
                    // substring-against the full path+alias+target, so `Zurich`
                    // hits. See [[Seek Index Processing Audit]].
                    const s = toDisplayForm(val);
                    if (!s || s.length > MAX_VALUE_LEN) return;
                    // Date-shaped values are temporal data, never a useful
                    // categorical pick — drop them by VALUE shape so an untyped
                    // date field (`due:`, `date:`) stays out of the menu without
                    // being named. A key whose values are ALL dates ends up with
                    // an empty map and is never offered. (Registry-typed Date
                    // keys are already skipped above, before reaching add().)
                    if (VALUE_DATE_RE.test(s)) return;
                    // Stray bracket chars (outside wikilink syntax) break the
                    // [key:value] token grammar in the pill field (activeToken's
                    // lastIndexOf bracket detection) — never offer those.
                    if (s.includes('[') || s.includes(']')) return;
                    let m = rawFields.get(k);
                    if (!m) { m = new Map(); rawFields.set(k, m); }
                    m.set(s, (m.get(s) ?? 0) + 1);
                };
                if (Array.isArray(v)) v.forEach(add); else add(v);
            }
        }
        // Drop any tag that occurs ONLY on excluded notes — offering it would
        // promise a `tag:`/`#` pill that the matcher can never satisfy.
        if (excludedFiles.length > 0) {
            const excludedTagKeys = new Set<string>();
            for (const f of excludedFiles) {
                for (const k of fileTagKeys(app.metadataCache.getFileCache(f))) excludedTagKeys.add(k);
            }
            for (const tag of this.tags.keys()) {
                const k = tag.toLowerCase();
                if (excludedTagKeys.has(k) && !includedTagKeys.has(k)) this.tags.delete(tag);
            }
        }
        for (const [k, m] of rawFields) {
            // Categorical = values repeat. Admit a wide key when its values
            // average ≥2 notes each; a per-note-unique field (distinct ≈ total)
            // stays excluded however it's measured.
            const total = [...m.values()].reduce((a, b) => a + b, 0);
            this.keyTotals.set(k, total);
            if (m.size <= MAX_KEY_CARDINALITY || total >= 2 * m.size) this.fields.set(k, m);
        }
        const keyTotal = (k: string) => [...this.fields.get(k)!.values()].reduce((a, b) => a + b, 0);
        this.fieldKeys = [...this.fields.keys()].sort((a, b) => keyTotal(b) - keyTotal(a));
        return this;
    }

    // Is `key` a Number-typed property? Drives the numeric comparison filter
    // (`[price>50]`) — the search modal's red error-pill validation reads this,
    // and the bracket value-menu uses it to offer operator scaffolds.
    isNumericKey(key: string): boolean {
        return this.numericKeys.has(key);
    }

    // Keys offered in the `[` key-menu: the gated categorical keys (most-populated
    // first), then any Number-typed keys not already among them (alphabetical).
    // Numeric keys ride past the cardinality gate (see numericKeys above).
    private completableKeys(): string[] {
        const extra = [...this.numericKeys].filter(k => !this.fields.has(k)).sort((a, b) => a.localeCompare(b));
        return [...this.fieldKeys, ...extra];
    }

    // Note-count backing a completable key, for the menu's `· N notes` hint.
    // Reads `keyTotals` (populated for every key, categorical or not) so a
    // numeric-only key like `price` — absent from `fields` — still reports its
    // real count instead of 0. Returns 0 only for a Number-typed key the index
    // has no usable values for (registry-declared but unindexed).
    private keyCount(k: string): number {
        return this.keyTotals.get(k) ?? 0;
    }

    // Detect what filter the user is completing at the cursor (always the tail,
    // since ghost completion only applies with the caret at end-of-input).
    // Returns null for bare words and for `[[` (not a Seek operator).
    private activeToken(q: string): ActiveToken | null {
        const lastOpen = q.lastIndexOf('[');
        const lastClose = q.lastIndexOf(']');
        if (lastOpen > lastClose) {
            const inner = q.slice(lastOpen + 1);
            if (inner.startsWith('[')) return null; // `[[` — note link, not a filter
            // The parser tolerates whitespace around the colon ([pageType: task]
            // binds via \s*:\s* + .trim()), so completion must too: trim the key
            // and left-trim the partial. typedTokOf() rebuilds the canonical
            // space-free token for the prefix comparison.
            const ci = inner.indexOf(':');
            if (ci >= 0) return { kind: 'field', from: lastOpen, key: inner.slice(0, ci).trim(), partial: inner.slice(ci + 1).replace(/^\s+/, '') };
            return { kind: 'fieldkey', from: lastOpen, partial: inner.trim() };
        }
        // Open quoted path `path:"…` — the partial can contain spaces, so it
        // isn't a single \S* token. Detect it before the whitespace tokenizer.
        const pq = q.match(/path:"([^"]*)$/);
        if (pq) return { kind: 'path', from: q.length - pq[0].length, partial: pq[1], quoted: true };

        const tok = (q.match(/(\S*)$/)?.[1]) ?? '';
        const from = q.length - tok.length;
        if (tok.startsWith('tag:')) return { kind: 'tag', from, partial: tok.slice(4), sigil: 'tag:' };
        if (tok.startsWith('#')) return { kind: 'tag', from, partial: tok.slice(1), sigil: '#' };
        if (tok.startsWith('path:')) return { kind: 'path', from, partial: tok.slice(5) };
        return null; // bare word — no completion
    }

    // The typed token in canonical form, for the prefix comparison against
    // dictionary tokens. Field tokens are rebuilt space-free ("[pageType: prog"
    // → "[pageType:prog") because the parser tolerates spaces around the colon
    // but the dictionary tokens never carry them. Other kinds pass through raw.
    private typedTokOf(value: string, t: ActiveToken): string {
        if (t.kind === 'field') return `[${t.key}:${t.partial}`;
        if (t.kind === 'fieldkey') return `[${t.partial}`;
        return value.slice(t.from);
    }

    // Case-insensitive strict prefix-extension test, shared by every matcher
    // below. Case-insensitive because binding is (the matcher lowercases both
    // sides); strict (length >) so an already-complete exact match isn't
    // re-suggested — an accepted "tag:meetings" surfaces the deeper
    // "tag:meetings/1x1s" instead of itself.
    private extendsTok(tok: string, typedTok: string): boolean {
        return tok.length > typedTok.length && tok.toLowerCase().startsWith(typedTok.toLowerCase());
    }

    // Highest-count entry whose makeTok(value) prefix-extends `typedTok`.
    private bestExtension(
        entries: Iterable<[string, number]>,
        makeTok: (value: string) => string | null,
        typedTok: string,
    ): { value: string; count: number; tok: string } | null {
        let best: { value: string; count: number; tok: string } | null = null;
        for (const [value, count] of entries) {
            const tok = makeTok(value);
            if (tok === null || !this.extendsTok(tok, typedTok)) continue;
            if (!best || count > best.count) best = { value, count, tok };
        }
        return best;
    }

    // Like bestExtension, but returns the top-`limit` extensions sorted by count
    // (busiest first) — the multi-row form that backs the dropdown menu. Same
    // strict prefix rule, so every row, if accepted, binds.
    private topExtensions(
        entries: Iterable<[string, number]>,
        makeTok: (value: string) => string | null,
        typedTok: string,
        limit: number,
    ): Array<{ value: string; count: number; tok: string }> {
        const out: Array<{ value: string; count: number; tok: string }> = [];
        for (const [value, count] of entries) {
            const tok = makeTok(value);
            if (tok === null || !this.extendsTok(tok, typedTok)) continue;
            out.push({ value, count, tok });
        }
        out.sort((a, b) => b.count - a.count);
        return out.slice(0, limit);
    }

    // Top completion for the current input, or null. `atEnd` gates on the caret
    // being at end-of-input (ghost text only makes sense as a trailing suffix).
    complete(value: string, atEnd: boolean): Completion | null {
        if (!atEnd || !value) return null;
        const t = this.activeToken(value);
        if (!t) return null;
        const typedTok = this.typedTokOf(value, t);
        const head = value.slice(0, t.from);
        const build = (tok: string, kind: Completion['kind'], label: string): Completion => {
            const accept = head + tok;
            // When the typed token was canonicalized (space after the colon),
            // accept no longer literally extends value — there is no valid
            // trailing ghost, but the accept string itself still rebinds
            // correctly when applied wholesale (dropdown rows, Tab).
            return { accept, ghost: accept.startsWith(value) ? accept.slice(value.length) : '', kind, label };
        };

        switch (t.kind) {
            case 'tag': {
                const sigil = t.sigil ?? '#';
                const best = this.bestExtension(this.tags, v => sigil + v, typedTok);
                return best ? build(best.tok, 'tag', `${best.value} · ${best.count} notes`) : null;
            }
            case 'path': {
                // Append `/*` so the accepted filter binds (the matcher is an
                // anchored glob, so a bare `path:folder` wouldn't match the
                // subtree). Two forms: inside an open quote (`path:"…`) we emit a
                // quoted, space-allowing completion that closes the quote; the
                // bare form is \S+, so it must skip folders containing whitespace
                // (the user has to open a quote to complete those).
                const makeTok = t.quoted
                    ? (v: string) => `path:"${v}/*"`
                    : (v: string) => (/\s/.test(v) ? null : `path:${v}/*`);
                const best = this.bestExtension(this.folders, makeTok, typedTok);
                return best ? build(best.tok, 'path', `${best.value}/* · ${best.count} notes`) : null;
            }
            case 'field': {
                const key = t.key ?? '';
                const values = this.fields.get(key);
                if (!values) return null;
                const best = this.bestExtension(values, v => `[${key}:${v}]`, typedTok);
                return best ? build(best.tok, 'field', `${key}:${best.value} · ${best.count} notes`) : null;
            }
            case 'fieldkey': {
                // completableKeys is categorical-first (most-populated) then numeric;
                // take the first that prefix-extends, preserving that order.
                for (const k of this.completableKeys()) {
                    const tok = `[${k}:`;
                    if (this.extendsTok(tok, typedTok)) {
                        return build(tok, 'fieldkey', `${k}:`);
                    }
                }
                return null;
            }
        }
    }

    // Top-`limit` completions for the current input — the list form of
    // complete(), used to render the pill field's suggestion dropdown. Each row
    // carries the bare `value` to commit into a pill (see SuggestItem). Empty
    // when there's no active filter token (bare word, `[[`, caret not at end).
    listSuggestions(value: string, limit = 8): SuggestItem[] {
        if (!value) return [];
        const t = this.activeToken(value);
        if (!t) return [];
        const typedTok = this.typedTokOf(value, t);
        const head = value.slice(0, t.from);
        const item = (tok: string, kind: Completion['kind'], bareValue: string, count: number, label: string): SuggestItem => {
            const accept = head + tok;
            // Same no-literal-prefix ghost guard as complete() (canonicalized
            // field tokens) — rows still rebind via their accept string.
            return { accept, ghost: accept.startsWith(value) ? accept.slice(value.length) : '', kind, label, value: bareValue, count };
        };

        switch (t.kind) {
            case 'tag': {
                const sigil = t.sigil ?? '#';
                return this.topExtensions(this.tags, v => sigil + v, typedTok, limit)
                    .map(e => item(e.tok, 'tag', e.value, e.count, `${e.value} · ${e.count} notes`));
            }
            case 'path': {
                const makeTok = t.quoted
                    ? (v: string) => `path:"${v}/*"`
                    : (v: string) => (/\s/.test(v) ? null : `path:${v}/*`);
                return this.topExtensions(this.folders, makeTok, typedTok, limit)
                    .map(e => item(e.tok, 'path', `${e.value}/*`, e.count, `${e.value}/* · ${e.count} notes`));
            }
            case 'field': {
                const key = t.key ?? '';
                const values = this.fields.get(key);
                if (!values) return [];
                return this.topExtensions(values, v => `[${key}:${v}]`, typedTok, limit)
                    .map(e => item(e.tok, 'field', `${key}:${e.value}`, e.count, `${key}:${e.value} · ${e.count} notes`));
            }
            case 'fieldkey': {
                const out: SuggestItem[] = [];
                for (const k of this.completableKeys()) {
                    const tok = `[${k}:`;
                    if (this.extendsTok(tok, typedTok)) {
                        out.push(item(tok, 'fieldkey', `${k}:`, this.keyCount(k), `${k}:`));
                    }
                    if (out.length >= limit) break;
                }
                return out;
            }
        }
    }
}
