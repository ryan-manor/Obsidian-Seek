// PillQueryField — the search modal's query input (Component 1 of the redesign).
//
// A horizontal row of: a search glyph, a flex-wrap field holding committed
// operator PILLS followed by an uncontrolled contenteditable for free text, and
// a suggestion dropdown. Operators are acknowledged INLINE as removable pills —
// there is no separate filter-chip echo row (the pills ARE the feedback).
//
// The field is a VIEW over a query STRING: pills + free text serialize back to
// the canonical inline-filter syntax the backend already parses
// (`tag:`, `path:…/*`, `[created:>date]`), so the search pipeline
// (scheduleSearch → runSearch → parseQuery) is untouched — it still receives a
// string. `after:`/`before:` are friendly sugar that serialize to the real
// `[created:>…]` / `[created:<…]` date filters.
//
// Caret discipline (ported from the design prototype): the contenteditable is
// UNCONTROLLED — we never rewrite its text on a re-render, only on the discrete
// actions (accept ghost, commit pill, delete pill), each followed by
// "move caret to end". A naively controlled input would fight the caret.

import { Platform, setIcon } from 'obsidian';
import type { SuggestEngine } from './suggest';
import { parseDateMs } from './fusion';

export type PillOp = 'tag' | 'path' | 'after' | 'before' | 'prop';

export interface PillToken {
    op: PillOp;
    value: string;
    // Frontmatter key for `prop` pills (`[key:value]`); unused by other ops.
    key?: string;
}

// The KEYWORD operators the field auto-completes (typed as `tag:`, `path:`,
// `after:`, `before:`). These four also commit as pills; bare `#tag` and
// `[key:value]` commit as pills too (a `tag` pill and a `prop` pill
// respectively — see derive()/pendingCommit()), so every filter the backend
// understands now gets the same inline-pill treatment. Only `-term` negation
// stays free text (parseQuery handles it; it has no value to pill).
const OPS: Array<{ key: PillOp; hint: string }> = [
    { key: 'tag', hint: 'filter by #tag' },
    { key: 'path', hint: 'filter by folder' },
    { key: 'after', hint: 'created on/after a date' },
    { key: 'before', hint: 'created on/before a date' },
];
const TOK_RE = /^(tag|path|after|before):(.*)$/i;

export interface PillQueryFieldCallbacks {
    // Fired on every change (keystroke, pill commit/remove) with the full,
    // serialized query string. The modal debounces + searches on this.
    onQueryChange: (query: string) => void;
    // Move result selection by ±1 (arrow keys, when the dropdown is closed).
    onNavigate: (dir: 1 | -1) => void;
    // Open the selected result (Enter); newTab when ⌘/Ctrl was held.
    onSubmit: (newTab: boolean) => void;
    // Esc with no dropdown open → close the modal.
    onDismiss: () => void;
    // Does this tag bind to a real vault tag (exact or hierarchical parent)?
    // Drives the warn-pill state for a `tag:` that matches nothing.
    validateTag: (tag: string) => boolean;
}

// One row in the suggestion dropdown. An 'op' row completes the operator
// keyword (sets the editable to `tag:` and re-derives); a 'value' row commits a
// finished filter into a pill (`key` is set only for `prop`/`[key:value]`
// rows); a 'text' row rewrites the editable wholesale — used for the
// intermediate `[key:` key-completion, which re-derives into the value menu
// rather than committing.
type MenuItem =
    | { type: 'op'; label: string; hint: string; opKey: PillOp }
    | { type: 'value'; label: string; hint?: string; op: PillOp; value: string; key?: string }
    | { type: 'text'; label: string; hint?: string; accept: string };

interface SuggestState {
    open: boolean;
    kind: 'op' | 'value';
    items: MenuItem[];
    active: number;
    hintRow?: string;
}

// Date presets offered for after:/before:. Values are date strings Date.parse
// (parseDateMs) accepts — bare year, year-month — so the serialized
// `[created:>YYYY-MM]` binds. Built off the real clock (plugin runtime; not a
// workflow script, so `new Date()` is allowed).
function datePresets(): Array<{ value: string; label: string }> {
    const now = new Date();
    const y = now.getFullYear();
    const mm = String(now.getMonth() + 1).padStart(2, '0');
    return [
        { value: `${y}-${mm}`, label: 'this month' },
        { value: `${y}`, label: 'this year' },
        { value: `${y - 1}`, label: 'last year' },
        { value: `${y - 2}`, label: String(y - 2) },
    ];
}

function caretToEnd(el: HTMLElement): void {
    const r = document.createRange();
    r.selectNodeContents(el);
    r.collapse(false);
    const s = window.getSelection();
    if (!s) return;
    s.removeAllRanges();
    s.addRange(r);
}

// True when the caret is collapsed at offset 0 of the editable (nothing typed
// before it) — the condition under which Backspace deletes the last pill.
function caretAtStart(el: HTMLElement): boolean {
    const s = window.getSelection();
    if (!s || !s.rangeCount) return false;
    const r = s.getRangeAt(0);
    if (!r.collapsed) return false;
    const pre = r.cloneRange();
    pre.selectNodeContents(el);
    pre.setEnd(r.endContainer, r.endOffset);
    return pre.toString().length === 0;
}

export class PillQueryField {
    private tokens: PillToken[] = [];
    private ghost = '';
    private sugg: SuggestState = { open: false, kind: 'op', items: [], active: 0 };
    // Has the user DELIBERATELY engaged the open suggestion menu (moved its
    // selection via hover, or via arrows once engaged)? derive() opens an
    // 'op'-kind menu for ANY trailing alpha run that prefixes an operator
    // ("a", "be", "t", "pa"…), so an unengaged op menu must NOT hijack
    // ↑/↓/Enter — typing "what would be" then Enter should submit the query,
    // not rewrite the field to "what would before:". Value-kind menus only
    // open after an explicit `op:` prefix, so they keep the always-on arrow/
    // Enter behavior. Tab remains the always-on accept affordance for both.
    // Reset on every re-derive (refresh) and on Escape-close.
    private menuEngaged = false;

    private rootEl: HTMLElement;
    private magEl: HTMLElement;
    private fieldEl: HTMLElement;
    private editEl: HTMLElement;
    private ghostEl: HTMLElement;
    private caretEl: HTMLElement;
    private placeholderEl: HTMLElement;
    private suggEl: HTMLElement;
    private suggLis: HTMLElement[] = [];

    constructor(
        parent: HTMLElement,
        private suggester: SuggestEngine,
        private cb: PillQueryFieldCallbacks,
    ) {
        this.rootEl = parent.createDiv({ cls: 'seek-q' });

        this.magEl = this.rootEl.createDiv({ cls: 'seek-mag' });
        setIcon(this.magEl, 'search');

        this.fieldEl = this.rootEl.createDiv({ cls: 'seek-field' });

        // editwrap holds the contenteditable + the trailing ghost + the
        // placeholder overlay. Pills are inserted BEFORE it so they wrap inline
        // ahead of the editable.
        const editWrap = this.fieldEl.createSpan({ cls: 'seek-editwrap' });
        this.editEl = editWrap.createSpan({ cls: 'seek-edit' });
        this.editEl.contentEditable = 'true';
        this.editEl.spellcheck = false;
        // Label the mobile keyboard's action key "Search" (vs the meaningless
        // default ↵). Pressing it follows the iOS search-as-you-type convention:
        // dismiss the keyboard to browse the live results — see the Enter handler.
        this.editEl.setAttribute('enterkeyhint', 'search');
        this.ghostEl = editWrap.createSpan({ cls: 'seek-ghost' });
        // Synthetic caret. An empty inline contenteditable collapses to 0×0, and
        // iOS WKWebView paints no native caret into a zero-area box — so a focused
        // but empty field (e.g. right after a pill) gives no cursor and no focus
        // cue. This bar fills that gap; CSS shows it only when the field is
        // focused AND the editable is empty, suppressing the native caret in the
        // same state so they never double up (see .seek-caret in styles.css).
        this.caretEl = editWrap.createSpan({ cls: 'seek-caret' });
        this.placeholderEl = editWrap.createSpan({ cls: 'seek-ph', text: 'Search your vault…' });

        this.suggEl = this.rootEl.createEl('ul', { cls: 'seek-sugg' });
        this.suggEl.hide();
        // Keep the editable focused when interacting with the dropdown.
        this.suggEl.addEventListener('mousedown', e => e.preventDefault());

        this.editEl.addEventListener('input', () => this.refresh());
        this.editEl.addEventListener('keydown', e => this.onKeyDown(e));
        // Focus state drives the synthetic caret + the field's focus accent. Fires
        // for both user taps and the modal's programmatic focus()/blur() (the
        // mobile "scroll results → drop keyboard" gesture blurs, correctly hiding
        // the caret until the user taps back in).
        this.editEl.addEventListener('focus', () => this.rootEl.addClass('is-focused'));
        this.editEl.addEventListener('blur', () => this.rootEl.removeClass('is-focused'));

        // Clicking anywhere in the query ROW — the glyph, the field padding, the
        // empty space past the caret — should focus the editable and drop the
        // caret at the end. The listener lives on the whole .seek-q row (rootEl),
        // NOT just .seek-field: the magnifier (.seek-mag) and the row's padding
        // sit OUTSIDE .seek-field, so a tap there — the natural "reactivate the
        // search" gesture after the field blurs — was landing on dead space and
        // never re-focused (the empty contenteditable collapses to 0×0, so most
        // taps miss it otherwise). Native clicks ON existing editable text, on a
        // pill (its own button), or on a suggestion row (the dropdown keeps focus
        // via its own mousedown) are left alone. The native close button is a
        // sibling under .modal, not a descendant of .seek-q, so it never reaches
        // this handler.
        this.rootEl.addEventListener('mousedown', e => {
            const t = e.target as HTMLElement;
            if (t === this.editEl || this.editEl.contains(t)) return;
            if (t.closest('.seek-pill') || t.closest('.seek-sugg')) return;
            e.preventDefault();
            this.editEl.focus();
            caretToEnd(this.editEl);
        });

        this.renderGhost();
        this.updatePlaceholder();
        // Normalize the dropdown state (hidden) + clear any stale
        // .seek-sugg-open left on the reused .modal-content from a prior open.
        this.renderSugg();
    }

    focus(): void {
        this.editEl.focus();
        caretToEnd(this.editEl);
    }

    // Blur the editable — on mobile this dismisses the soft keyboard. Used by the
    // modal's "touch the results to scroll → drop the keyboard" gesture; a no-op
    // on desktop (nothing visible changes, the field just loses focus).
    blur(): void {
        this.editEl.blur();
    }

    // Reflects embedder readiness in the search glyph. The glyph rests dimmed
    // (model not loaded) and brightens to full strength once it's ready —
    // purely ambient, the field stays editable and queries still run while the
    // model loads (the modal holds them on modelReadyPromise). See
    // `.seek-mag` / `.seek-mag.is-ready` in styles.css for the fade.
    setModelReady(ready: boolean): void {
        this.magEl.toggleClass('is-ready', ready);
    }

    // Seed the field from a raw query string (an obsidian://seek?query= deep
    // link). The field is a VIEW over a query string (see the class header), so
    // the simplest correct seed is to drop the text into the uncontrolled
    // editable and re-derive: getQueryString() then returns it verbatim, and the
    // search pipeline re-parses inline filters (#tag, path:, [k:v], dates) from
    // the string regardless of whether they render as pills — so results match
    // typing it by hand. Pills then form as the user edits. Writing textContent
    // directly mirrors accept()/commit(); refresh() emits onQueryChange, which is
    // what schedules the search.
    setQuery(raw: string): void {
        this.tokens = [];
        this.renderPills();
        this.editEl.textContent = raw;
        caretToEnd(this.editEl);
        this.refresh();
    }

    // ---- serialization: pills + free text → canonical query string ----

    private serializeToken(t: PillToken): string {
        switch (t.op) {
            case 'tag':
                return `tag:${t.value}`;
            case 'path':
                return /\s/.test(t.value) ? `path:"${t.value}"` : `path:${t.value}`;
            case 'after':
                return `[created:>${t.value}]`;
            case 'before':
                return `[created:<${t.value}]`;
            case 'prop':
                // Round-trips any `[key:value]` parseQuery accepts: a plain
                // substring match, a `"quoted"` exact value, or a date op like
                // `created:>2026` (the operator rides along inside `value`).
                return `[${t.key}:${t.value}]`;
        }
    }

    private getQueryString(): string {
        const parts = this.tokens.map(t => this.serializeToken(t));
        const text = this.readText().trim();
        if (text) parts.push(text);
        return parts.join(' ');
    }

    private readText(): string {
        return this.editEl.textContent ?? '';
    }

    // ---- suggestion derivation ----

    private derive(text: string): { ghost: string; sugg: SuggestState } {
        const last = (text.match(/(\S*)$/)?.[1]) ?? '';

        // Property-filter stage: an open `[…` block (the `[key:value]` syntax).
        // Must run FIRST — inside a bracket the trailing word is a property
        // key/value being typed, so the op-keyword stage below must not see it
        // ("[pageType: t" is not a `tag:` summons). SuggestEngine already
        // builds the key/value dictionaries; rows rewrite the editable via
        // their accept string (free text, not a pill — the 4-op scope line
        // stands) and the completed [key:value] binds through parseQuery.
        const lastOpen = text.lastIndexOf('[');
        if (lastOpen > text.lastIndexOf(']') && !text.slice(lastOpen).startsWith('[[')) {
            const list = this.suggester.listSuggestions(text, 8)
                .filter(s => s.kind === 'field' || s.kind === 'fieldkey');
            const items: MenuItem[] = list.map(s => {
                // `fieldkey` rows are the intermediate `[key:` step: rewrite the
                // editable (free text) and re-derive into the value menu.
                if (s.kind === 'fieldkey') {
                    return { type: 'text', accept: s.accept, label: `[${s.value}`, hint: `${s.count} notes` };
                }
                // `field` rows are a complete `[key:value]` — commit a prop pill.
                // s.value is the bare `key:value`; split on the first colon so a
                // value that itself contains `:` (a time, a URL) stays intact.
                const ci = s.value.indexOf(':');
                return {
                    type: 'value', op: 'prop',
                    key: s.value.slice(0, ci), value: s.value.slice(ci + 1),
                    label: `[${s.value}]`, hint: `${s.count} notes`,
                };
            });
            return {
                ghost: list[0]?.ghost ?? '',
                sugg: { open: items.length > 0, kind: 'value', items, active: 0 },
            };
        }

        // Bare `#tag` stage: mirror `tag:` completion (same dictionary, same
        // value rows) so `#` gets the menu + ghost + pill that `tag:` has. Runs
        // before the op-keyword stage, which only sees pure-alpha runs and would
        // skip a `#`-prefixed token anyway. `##` is not a tag — let it fall
        // through. Committed rows produce a `tag` pill (op below), identical to
        // the `tag:` path, so the warn-pill + serialization are shared.
        if (last.startsWith('#') && !last.startsWith('##')) {
            const list = this.suggester.listSuggestions(text, 8).filter(s => s.kind === 'tag');
            const items: MenuItem[] = list.map(s => ({
                type: 'value', op: 'tag', value: s.value, label: `#${s.value}`, hint: `${s.count} notes`,
            }));
            return {
                ghost: list[0]?.ghost ?? '',
                sugg: { open: items.length > 0, kind: 'value', items, active: 0 },
            };
        }

        // Operator-keyword stage: a trailing alpha run that prefixes an operator.
        if (/^[a-zA-Z]+$/.test(last)) {
            const matches = OPS.filter(o => o.key.startsWith(last.toLowerCase()));
            if (matches.length) {
                return {
                    ghost: matches[0].key.slice(last.length) + ':',
                    sugg: {
                        open: true, kind: 'op', active: 0,
                        items: matches.map(o => ({ type: 'op', label: o.key + ':', hint: o.hint, opKey: o.key })),
                    },
                };
            }
        }

        // Value stage: a trailing `op:value` token.
        const tm = last.match(TOK_RE);
        if (tm) {
            const op = tm[1].toLowerCase() as PillOp;
            const val = tm[2];

            if (op === 'tag' || op === 'path') {
                const list = this.suggester.listSuggestions(text, 8).filter(s => s.kind === op);
                const items: MenuItem[] = list.map(s => ({
                    type: 'value', op, value: s.value,
                    label: op === 'tag' ? `#${s.value}` : s.value,
                    hint: `${s.count} notes`,
                }));
                return {
                    ghost: list[0]?.ghost ?? '',
                    sugg: { open: items.length > 0, kind: 'value', items, active: 0 },
                };
            }

            // after: / before: — date presets + a free-entry hint.
            const presets = datePresets().filter(p => p.value.startsWith(val));
            const items: MenuItem[] = presets.map(p => ({
                type: 'value', op, value: p.value, label: p.value, hint: p.label,
            }));
            const ghost = (val && presets[0]?.value.startsWith(val)) ? presets[0].value.slice(val.length) : '';
            return {
                ghost,
                sugg: { open: true, kind: 'value', items, active: 0, hintRow: 'type a date · YYYY, YYYY-MM or YYYY-MM-DD' },
            };
        }

        return { ghost: '', sugg: { open: false, kind: 'op', items: [], active: 0 } };
    }

    // Re-read the DOM, re-derive suggestions + ghost, and emit the new query.
    // Never writes editable text (uncontrolled discipline).
    private refresh(): void {
        // Height-explosion guard. After a delete-to-empty, WebKit/iOS leaves DOM
        // residue in the contenteditable — a stray <br>, a wrapping <div>, or a
        // trailing "\n" — which under .seek-edit's `white-space: pre-wrap` paints a
        // phantom second line: the field "explodes" even though the query is empty
        // (see Inbox/Seek mobile issues). textContent is already effectively empty
        // here, so normalize the node back to truly-empty. Assigning textContent
        // fires no 'input' event, so this can't re-enter; only a focused field's
        // caret is repositioned. The `trim() === ''` gate leaves all real input
        // alone — including mid-IME composition, whose buffer is never empty.
        if (this.editEl.firstChild && this.readText().trim() === '') {
            this.editEl.textContent = '';
            if (document.activeElement === this.editEl) caretToEnd(this.editEl);
        }
        const text = this.readText();
        const d = this.derive(text);
        this.ghost = d.ghost;
        this.sugg = d.sugg;
        // Each re-derive is a fresh menu — engagement doesn't carry across
        // keystrokes (the items under the highlight just changed).
        this.menuEngaged = false;
        this.renderGhost();
        this.renderSugg();
        this.updatePlaceholder();
        this.cb.onQueryChange(this.getQueryString());
    }

    // ---- accept / commit / remove ----

    private accept(item: MenuItem): void {
        if (item.type === 'op') {
            const text = this.readText();
            const last = (text.match(/(\S*)$/)?.[1]) ?? '';
            const base = text.slice(0, text.length - last.length);
            this.editEl.textContent = base + item.opKey + ':';
            caretToEnd(this.editEl);
            this.refresh();
        } else if (item.type === 'text') {
            // Property-filter rows replace the editable wholesale: the accept
            // string is head + canonical token, which also normalizes away any
            // space the user typed after the colon. A `[key:` row re-derives
            // into the value menu; a complete `[key:value]` stays as free text.
            this.editEl.textContent = item.accept;
            caretToEnd(this.editEl);
            this.refresh();
        } else {
            this.commit(item.op, item.value, item.key);
        }
    }

    // What pill (if any) the trailing token would commit to on Space/Enter.
    // Unifies the three committable surface forms so the keyboard handlers stay
    // identical across them:
    //   tag:/path:/after:/before:value  → that op (TOK_RE)
    //   #tag                            → a tag pill (sans sigil)
    //   [key:value]  (closed bracket)   → a prop pill (split on first colon)
    // An unclosed `[key:value` returns null — parseQuery needs the `]` to bind,
    // so it stays free text until the bracket is closed (or a menu row accepted).
    private pendingCommit(last: string): { op: PillOp; value: string; key?: string } | null {
        const tm = last.match(TOK_RE);
        if (tm && tm[2]) return { op: tm[1].toLowerCase() as PillOp, value: tm[2] };
        if (last.startsWith('#') && last.length > 1 && !last.startsWith('##')) {
            return { op: 'tag', value: last.slice(1) };
        }
        // Complete `[key:value]`: key is any non-`]:[` run, value any non-`]`
        // run, split on the FIRST colon (a value may contain `:`).
        const bm = last.match(/^\[([^\]:[]+):([^\]]*)\]$/);
        if (bm && bm[2].trim()) return { op: 'prop', key: bm[1].trim(), value: bm[2].trim() };
        return null;
    }

    // Gate on a value before it may become a pill. after:/before: values must
    // parse as a date (parseDateMs — the same parser the compiled [created:…]
    // clause runs through downstream): an unparseable value would commit a
    // confident-looking pill whose serialized clause the matcher silently
    // DROPS — an active-looking filter that filters nothing. Declining the
    // commit leaves the text in the editable instead. tag:/path: stay
    // permissive (tags get the warn-pill state, paths are normalized below).
    private canCommit(op: PillOp, value: string): boolean {
        if (op === 'after' || op === 'before') return parseDateMs(value) !== null;
        return true;
    }

    // Commit a finished `op:value` into a pill: strip the trailing token from
    // the editable, append the pill, re-render.
    private commit(op: PillOp, value: string, key?: string): void {
        if (!this.canCommit(op, value)) return;
        // Hand-typed paths get the anchored-glob form the suggestion rows
        // already emit (suggest.ts appends `/*`): globToRegExp anchors with
        // ^…$, so a bare `path:Work` would compile to a regex that can never
        // match "Work/foo.md" — a confident-looking pill that filters
        // everything out. Values that already carry a glob char are the
        // user's own pattern and pass through untouched.
        if (op === 'path' && !/[*?]/.test(value)) {
            value = value.replace(/\/$/, '') + '/*';
        }
        // Strip the partial that produced this pill. For `prop` the partial is
        // the open `[key:value…` block, whose value can hold spaces (`[place:
        // New York`), so the `\S*$` last-token rule (right for the other ops)
        // would only cut the final word — cut from the last `[` instead.
        const text = this.readText();
        let cut: number;
        if (op === 'prop') {
            const lb = text.lastIndexOf('[');
            cut = lb >= 0 ? lb : text.length;
        } else {
            const last = (text.match(/(\S*)$/)?.[1]) ?? '';
            cut = text.length - last.length;
        }
        this.editEl.textContent = text.slice(0, cut);
        caretToEnd(this.editEl);
        this.tokens.push({ op, value, key });
        this.renderPills();
        this.refresh();
    }

    private removeToken(i: number): void {
        this.tokens.splice(i, 1);
        this.renderPills();
        this.editEl.focus();
        caretToEnd(this.editEl);
        this.refresh();
    }

    // ---- keyboard ----

    private onKeyDown(e: KeyboardEvent): void {
        const text = this.readText();
        const last = (text.match(/(\S*)$/)?.[1]) ?? '';
        const openSugg = this.sugg.open && this.sugg.items.length > 0;
        const accel = e.metaKey || e.ctrlKey;

        // Arrow keys move the MENU selection only when the menu owns them:
        // always for a value-kind menu (it was summoned by an explicit `op:`
        // prefix), but for an op-kind menu only once engaged (see menuEngaged)
        // — an unengaged op menu is a passive hint over a plain word being
        // typed, and ↑/↓ keep navigating results.
        const menuOwnsArrows = openSugg && (this.sugg.kind === 'value' || this.menuEngaged);
        if (e.key === 'ArrowDown') {
            e.preventDefault();
            if (menuOwnsArrows) { this.menuEngaged = true; this.sugg.active = Math.min(this.sugg.active + 1, this.sugg.items.length - 1); this.updateActive(); }
            else this.cb.onNavigate(1);
            return;
        }
        if (e.key === 'ArrowUp') {
            e.preventDefault();
            if (menuOwnsArrows) { this.menuEngaged = true; this.sugg.active = Math.max(this.sugg.active - 1, 0); this.updateActive(); }
            else this.cb.onNavigate(-1);
            return;
        }
        if (e.key === 'Tab') {
            if (openSugg) { e.preventDefault(); this.accept(this.sugg.items[this.sugg.active]); return; }
            if (this.ghost) {
                e.preventDefault();
                this.editEl.textContent = text + this.ghost;
                caretToEnd(this.editEl);
                this.refresh();
                return;
            }
            return;
        }
        if (e.key === 'Enter') {
            // Mobile follows the iOS search-as-you-type convention: the "Search"
            // key (enterkeyhint above) DISMISSES the keyboard so the live results
            // get the full screen to browse/tap — it never opens the top hit (you
            // tap a result for that). Tear down the suggestion affordance like
            // Escape does, then drop focus. Desktop keeps the accept/commit/open
            // behavior below, where a pointer-free workflow needs Enter to act.
            if (Platform.isMobile) {
                e.preventDefault();
                this.sugg = { open: false, kind: 'op', items: [], active: 0 };
                this.ghost = '';
                this.menuEngaged = false;
                this.renderGhost();
                this.renderSugg();
                this.blur();
                return;
            }
            // Accept a highlighted suggestion — but ⌘/Ctrl+Enter always means
            // "open in new tab", so for a value row it skips straight to submit.
            // Op-kind menus additionally require engagement (menuEngaged):
            // otherwise Enter mid-sentence ("what would be↵") would rewrite the
            // field to "what would before:" instead of opening the selection.
            if (openSugg && (this.sugg.kind === 'op' ? this.menuEngaged : !accel)) {
                e.preventDefault();
                this.accept(this.sugg.items[this.sugg.active]);
                return;
            }
            const pc = this.pendingCommit(last);
            if (pc && !accel && this.canCommit(pc.op, pc.value)) {
                e.preventDefault();
                this.commit(pc.op, pc.value, pc.key);
                return;
            }
            e.preventDefault();
            this.cb.onSubmit(accel);
            return;
        }
        if (e.key === ' ') {
            // An uncommittable token (e.g. after:lastweek, or an unclosed
            // `[key:value`) lets the space type through normally — no pill, text
            // stays editable as-is.
            const pc = this.pendingCommit(last);
            if (pc && this.canCommit(pc.op, pc.value)) {
                e.preventDefault();
                this.commit(pc.op, pc.value, pc.key);
                return;
            }
            return;
        }
        if (e.key === 'Escape') {
            if (this.sugg.open) {
                e.preventDefault();
                this.sugg = { open: false, kind: 'op', items: [], active: 0 };
                this.ghost = '';
                this.menuEngaged = false;
                this.renderGhost();
                this.renderSugg();
                return;
            }
            this.cb.onDismiss();
            return;
        }
        if (e.key === 'Backspace' && caretAtStart(this.editEl) && this.tokens.length) {
            e.preventDefault();
            this.removeToken(this.tokens.length - 1);
            return;
        }
    }

    // ---- rendering ----

    private renderPills(): void {
        this.fieldEl.querySelectorAll('.seek-pill').forEach(el => el.remove());
        const editWrap = this.fieldEl.querySelector('.seek-editwrap');
        this.tokens.forEach((t, i) => {
            const pill = createSpan({ cls: `seek-pill seek-pill-${t.op}` });
            if (t.op === 'tag' && !this.cb.validateTag(t.value)) {
                pill.addClass('seek-pill-warn');
                pill.setAttr('aria-label', `No vault tag matches "${t.value}". Tag filtering is exact + hierarchical, so a sibling like "${t.value}s" won't match.`);
            }
            // A prop pill's keycap is its frontmatter key (`context:`), not the
            // internal op name; every other op labels with the op itself.
            pill.createSpan({ cls: 'seek-pill-k', text: t.op === 'prop' ? `${t.key}:` : `${t.op}:` });
            pill.createSpan({ cls: 'seek-pill-v', text: t.value });
            const x = pill.createEl('button', { cls: 'seek-pill-x', text: '×' });
            x.setAttr('aria-label', 'remove filter');
            x.addEventListener('mousedown', e => e.preventDefault());
            x.addEventListener('click', () => this.removeToken(i));
            this.fieldEl.insertBefore(pill, editWrap);
        });
    }

    private renderGhost(): void {
        this.ghostEl.setText(this.ghost);
        this.ghostEl.toggle(this.ghost.length > 0);
        // The synthetic caret only stands in when the editable has no text AND no
        // ghost preview — the one case the native caret can't render. Every text/
        // ghost mutation funnels through here (refresh, commit, backspace-remove,
        // initial build), so this is the single place to keep the flag current.
        this.rootEl.toggleClass('is-empty-edit', !this.readText() && this.ghost.length === 0);
    }

    private updatePlaceholder(): void {
        const empty = !this.readText() && this.tokens.length === 0;
        this.placeholderEl.toggle(empty);
    }

    private renderSugg(): void {
        this.suggLis = [];
        this.suggEl.empty();
        const open = this.sugg.open && this.sugg.items.length > 0;
        // Reserve modal-body height while the menu is open so a tall dropdown
        // over a short result set isn't clipped by .modal-content's
        // overflow:hidden (the dropdown is absolutely positioned, so it adds no
        // height itself). The class drives a desktop-only min-height in
        // styles.css and clears the moment the menu closes. .seek-q's parent is
        // .modal-content (PillQueryField is constructed with it as `parent`).
        this.rootEl.parentElement?.toggleClass('seek-sugg-open', open);
        if (!open) {
            this.suggEl.hide();
            return;
        }
        if (this.sugg.hintRow) {
            this.suggEl.createEl('li', { cls: 'seek-sugg-hint', text: this.sugg.hintRow });
        }
        this.sugg.items.forEach((it, i) => {
            const li = this.suggEl.createEl('li', { cls: 'seek-sugg-item' + (i === this.sugg.active ? ' is-active' : '') });
            li.createSpan({ cls: 'seek-sugg-label', text: it.label });
            if (it.hint) li.createSpan({ cls: 'seek-sugg-meta', text: it.hint });
            // Hover is a deliberate act on the menu, so it engages it — after
            // hovering, ↑/↓/Enter operate on the menu even for an op-kind one.
            li.addEventListener('mouseenter', () => { this.menuEngaged = true; this.sugg.active = i; this.updateActive(); });
            li.addEventListener('click', () => this.accept(it));
            this.suggLis.push(li);
        });
        this.suggEl.show();
    }

    private updateActive(): void {
        this.suggLis.forEach((li, i) => li.toggleClass('is-active', i === this.sugg.active));
        this.suggLis[this.sugg.active]?.scrollIntoView({ block: 'nearest' });
    }
}
