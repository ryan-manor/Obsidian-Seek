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

import type { App } from 'obsidian';
import { toDisplayForm } from './prop-normalize';

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

// Frontmatter keys never offered as [key:value] filters. Tags/aliases have
// their own operators; dates aren't categorical; the rest are Task-Notes /
// Obsidian machinery or per-note unique values that would bloat the menu.
const KEY_BLOCKLIST = new Set([
    'position', 'tags', 'aliases', 'cssclass', 'cssclasses',
    'created', 'modified', 'date', 'dateLink', 'due', 'completedDate',
    'week', 'relatedPages', 'title', 'id', 'uuid', 'permalink',
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

    // Build all dictionaries from the in-memory metadata cache. Cheap enough to
    // run on every modal open (a single pass over getMarkdownFiles, no I/O), so
    // suggestions always reflect the current vault.
    build(app: App): this {
        const cache = app.metadataCache as unknown as { getTags?: () => Record<string, number> };
        for (const [t, c] of Object.entries(cache.getTags?.() ?? {})) {
            const tag = t.replace(/^#/, '');
            if (PARSER_TAG.test(tag)) this.tags.set(tag, c); // skip non-bindable tags (space/emoji/non-ASCII)
        }

        const rawFields = new Map<string, Map<string, number>>();
        for (const f of app.vault.getMarkdownFiles()) {
            const folder = f.parent?.path;
            if (folder && folder !== '/') this.folders.set(folder, (this.folders.get(folder) ?? 0) + 1);

            const fm = app.metadataCache.getFileCache(f)?.frontmatter;
            if (!fm) continue;
            for (const [k, v] of Object.entries(fm)) {
                if (KEY_BLOCKLIST.has(k) || !PARSER_KEY.test(k)) continue; // key must be parser-capturable (\w+)
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
        for (const [k, m] of rawFields) {
            // Categorical = values repeat. Admit a wide key when its values
            // average ≥2 notes each; a per-note-unique field (distinct ≈ total)
            // stays excluded however it's measured.
            const total = [...m.values()].reduce((a, b) => a + b, 0);
            if (m.size <= MAX_KEY_CARDINALITY || total >= 2 * m.size) this.fields.set(k, m);
        }
        const keyTotal = (k: string) => [...this.fields.get(k)!.values()].reduce((a, b) => a + b, 0);
        this.fieldKeys = [...this.fields.keys()].sort((a, b) => keyTotal(b) - keyTotal(a));
        return this;
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
                // fieldKeys is already most-populated-first; take the first that
                // prefix-extends, preserving that frequency order.
                for (const k of this.fieldKeys) {
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
                for (const k of this.fieldKeys) {
                    const tok = `[${k}:`;
                    if (this.extendsTok(tok, typedTok)) {
                        const count = [...this.fields.get(k)!.values()].reduce((a, b) => a + b, 0);
                        out.push(item(tok, 'fieldkey', `${k}:`, count, `${k}:`));
                    }
                    if (out.length >= limit) break;
                }
                return out;
            }
        }
    }
}
