// Seek search modal. A plain Modal (not SuggestModal) with a debounced query,
// manual result rendering, and a keyboard model layered on top: arrow keys move
// a selectedIndex, Enter opens it, ⌘/Ctrl+Enter (or ⌘/Ctrl+click) opens in a
// new tab. The query field is a token/pill input (see PillQueryField) that
// serializes committed operator pills + free text back to the inline-filter
// query string the search pipeline already parses.

import { App, Modal, Notice, Platform, TFile, MarkdownView, MarkdownRenderer, Component } from 'obsidian';
import type { ScoredChunk, SearchEntry, ClickEntry, SeekSettings } from './types';
import { MATCH_STRENGTH_MIN_NOTES } from './types';
import type { SearchOrchestrator } from './search';
import type { SeekLogger } from './logger';
import { ENGLISH_STOPWORDS } from './bm25';
import { buildHighlightRanges } from './highlight';
import { sanitizeSnippet } from './snippet';
import { matchStrength } from './dense-stats';
import { SuggestEngine } from './suggest';
import { PillQueryField } from './query-field';

// Search debounce. Mobile gets a longer window: the query embed runs on the
// render thread (iframe = same event loop) and on iOS the stage-1 binary scan is
// synchronous too, so every fired search is costly. A wider debounce drops the
// count of wasted in-flight embeds while the user is still typing.
const DEBOUNCE_MS = Platform.isMobile ? 400 : 200;

// How long after the last keystroke (with results showing) we consider the query
// "settled" and signal the plugin to drain catch-up indexing — the safest mobile
// window: modal open, app provably foreground, user reading rather than typing.
// Longer than DEBOUNCE_MS so a settle only fires once the search itself has run.
const CATCHUP_SETTLE_MS = 1500;

// Pagination. The orchestrator does ALL its scoring/fusion work over the full
// candidate union regardless of topK (topK only caps the final dedup + a single
// batched body read), so fetching deep is ~free — we ask for MAX_RESULTS once.
// Rendering, however, is the jank source (one markdown render per snippet row),
// so rows are revealed a PAGE_SIZE window at a time via infinite scroll, never
// all at once. PAGE_SIZE matches the historical 10 so the first paint is
// unchanged; subsequent pages append as the user scrolls (or arrows past the
// window edge).
const PAGE_SIZE = 10;
const MAX_RESULTS = 50;
// Reveal the next page this far (px) before the sentinel scrolls fully into
// view, so the rows are already painted by the time the user reaches them.
const REVEAL_MARGIN_PX = 300;

// (sanitizeSnippet lives in ./snippet, and maskNonBodyText / escapeRegExp / the
// word-boundary range builder in ./highlight, so they can be unit-tested without
// Obsidian — see buildMatchHighlight + the snippet render in applyRow.)

// "2026-05-19" from a created date. ISO is rendered deliberately rather than a
// localized string: it matches the frontmatter shape and is unambiguous in every
// locale (no Jun/giugno/6月 drift, no DD/MM vs MM/DD ambiguity). The word
// "created" is added by the caller so the date can't be mistaken for a modified
// date. Returns '' for a missing/unparseable value so the meta line drops the
// segment.
//
// The VALUE is still parsed in the user's LOCAL zone: extractDate() normalizes
// most created values to a date-only string, which we build into a local-midnight
// Date — a bare `new Date('2026-06-19')` would be UTC midnight and roll back a day
// west of UTC (PDT → "2026-06-18"); a full timestamp falls through to native
// parsing and resolves to its local calendar day. We emit the LOCAL Y-M-D and
// never Date.toISOString(), which re-projects to UTC and would roll the day back
// for ahead-of-UTC users (JST → prior day).
function fmtCreated(iso: string | null): string {
    if (!iso) return '';
    const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(iso);
    const d = m
        ? new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3]))
        : new Date(iso);
    if (isNaN(d.getTime())) return '';
    const y = d.getFullYear();
    const mo = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    return `${y}-${mo}-${day}`;
}

// The note's display title: the file basename without extension. Deliberately
// NOT chunk.title — that's the hierarchical "Note > H1 > H2" embed string (with
// aliases appended); the heading part is shown separately as the breadcrumb.
function noteTitle(notePath: string): string {
    const base = notePath.split('/').pop() ?? notePath;
    return base.replace(/\.md$/i, '');
}

// Status of the embedder model at the time the modal opens. `ready=true` is
// the warm path (model already in memory from a prior open) and we skip all
// "loading…" UI. `ready=false` means the caller kicked off
// `ensureModelLoaded()` without awaiting and handed us the promise so the
// modal can render + accept input *during* the ~3–10 s cold start, then
// fire the pending query the moment the model resolves.
export interface ModelStatus {
    ready: boolean;
    promise: Promise<void>;
}

// A reused result row. We hold the leaf elements so reconciliation can update
// text without recreating the node, the live `data`/`rank` the row's single
// click handler reads (so the closure never goes stale), and `lastSnippet` —
// the markdown source last rendered into `snippetEl`, so an unchanged snippet
// skips the (async, comparatively expensive) markdown re-render entirely.
interface SeekResultRow {
    el: HTMLElement;
    titleEl: HTMLElement;
    breadcrumbEl: HTMLElement;
    snippetEl: HTMLElement;
    metaEl: HTMLElement;
    scoreEl: HTMLElement;
    keycapEl: HTMLElement;
    data: ScoredChunk;
    rank: number;
    lastSnippet: string;
    // The breadcrumb markdown source last rendered into `breadcrumbEl` — so an
    // unchanged heading path skips the (async) markdown re-render, mirroring
    // `lastSnippet`.
    lastCrumb: string;
}

// The index-state banner the modal renders between the query field and the results.
// The plugin supplies the copy + tone (index-notice.ts policy); the modal owns the
// "Open settings" action (where the reindex affordance lives), shown only when the spec
// asks for it (showAction) — the syncing/info state needs no button. A twin of
// IndexBannerSpec (index-notice.ts), kept separate so the UI module owns no policy.
export interface IndexBanner {
    message: string;
    tone: 'info' | 'warn';
    showAction: boolean;
}

export class SeekSearchModal extends Modal {
    private orchestrator: SearchOrchestrator;
    private logger: SeekLogger;
    // The token/pill query field (Component 1). Owns the contenteditable, the
    // committed operator pills, ghost autocomplete, and the suggestion dropdown;
    // emits the serialized query string back to us on every change.
    private field: PillQueryField | null = null;
    // Holds the vault's distinct-value dictionaries (built once on open) that
    // back the field's value suggestions.
    private suggester: SuggestEngine | null = null;
    private resultsEl: HTMLElement | null = null;
    // Fixed-position container for the version-stale banner, between the field and the
    // results. Rendered into (empty + refill) so re-renders keep their place; collapses
    // to nothing via `.seek-banner-slot:empty { display: none }` when there's no banner.
    private bannerSlot: HTMLElement | null = null;
    // Reusable result rows, reconciled in place across searches so only the
    // text that actually changed repaints — no full teardown means no flicker
    // and no scroll-reset. Indexed positionally: rows[0] is always rank 1.
    private rows: SeekResultRow[] = [];
    // Owns the lifecycle of anything MarkdownRenderer spawns while rendering
    // snippets (embeds, child renderers). Loaded in onOpen, unloaded in onClose.
    private markdownComponent: Component = new Component();
    // Snapshot of the vault's tag set (lowercased, no `#`), taken once on open
    // so a `tag:` pill can be flagged when it binds to no real vault tag.
    private vaultTagSet: Set<string> = new Set();
    private timer: number | null = null;
    private currentSearch = 0;
    // The currently displayed, ordered results — the array the keyboard model
    // indexes into via `selectedIndex`.
    private currentResults: ScoredChunk[] = [];
    // The highlighted row the keyboard model acts on. Driven by ↑/↓ (from the
    // field) and by mouse hover; clamped to the result count on every render.
    private selectedIndex = 0;
    // Infinite-scroll window: `currentResults` holds up to MAX_RESULTS fetched
    // rows, but only the first `shownCount` are materialized as DOM rows. A
    // sentinel at the list's tail, watched by `revealObserver`, grows the window
    // a page at a time as it scrolls into view. selectedIndex never exceeds
    // shownCount-1 (moveSelection reveals before crossing the edge).
    private shownCount = 0;
    private sentinelEl: HTMLElement | null = null;
    private revealObserver: IntersectionObserver | null = null;
    // Set true at the top of onClose, before any teardown. A fresh modal
    // instance is constructed on every open, so this never needs resetting —
    // it just gives in-flight async work (a query embed, checkIndexState, the
    // model-load promise) a way to recognize "the modal I'd paint into is
    // already gone" and no-op instead of writing into detached DOM, rendering
    // into the now-unloaded markdownComponent, or (via renderResults →
    // updateSentinel) constructing a fresh IntersectionObserver that nothing
    // will ever disconnect.
    private closed = false;
    // The latest SearchEntry returned by the orchestrator. Click events
    // reference its searchId so offline analysis can correlate the click back
    // to the originating query and the alternatives the user passed over.
    private latestSearchEntry: SearchEntry | null = null;
    private latestResultsShown: ScoredChunk[] = [];
    private latestSearchCompletedAt = 0;

    // Cold-start onboarding flag: set true once checkIndexState() (run on open)
    // confirms zero indexed chunks. Drives renderNoIndex() in place of the generic
    // "Type to search…" / "No notes match." copy, so a brand-new user is told to
    // build the index rather than facing a silent dead end. Stays false on an
    // unreadable store, so a still-warming index is never mislabeled "not indexed".
    private indexEmpty = false;

    // Model-load decoupling. When `modelReady` is false, `runSearch` awaits
    // `modelReadyPromise` before calling the orchestrator. The existing
    // `currentSearch` counter doubles as a "pending query" collapsing mechanism.
    private modelReady: boolean;
    private modelReadyPromise: Promise<void>;
    private modelLoadError: Error | null = null;

    // Teardown for the mobile visualViewport listeners (keyboard-aware sizing).
    // Null on desktop / when visualViewport is unavailable.
    private detachViewport: (() => void) | null = null;

    // Fires (active) on open + each keystroke and (inactive) on query-settle +
    // close. The plugin uses it to pause/trigger the catch-up drain so foreground
    // indexing never competes with the live query. Optional (absent in tests).
    private onSearchActivity?: (active: boolean) => void;

    // Reports the hard "a query embed is actually running" edge to the plugin —
    // distinct from onSearchActivity, which is keystroke-timed. Lets indexing wait
    // for the query to COMPLETE, not merely for typing to pause. Optional (absent
    // in tests). Ref-counted via `inFlight` so overlapping cold-path searches emit
    // one clean true/false pair rather than flapping.
    private onQueryInFlight?: (inFlight: boolean) => void;
    private inFlight = 0;

    // Debounce for the post-query "settled" signal (separate from the 200 ms search
    // debounce). Reset on every keystroke; fires onSearchActivity(false) on idle.
    private settleTimer: number | null = null;

    constructor(
        app: App,
        orchestrator: SearchOrchestrator,
        logger: SeekLogger,
        modelStatus: ModelStatus,
        private settings: SeekSettings,
        onSearchActivity?: (active: boolean) => void,
        onQueryInFlight?: (inFlight: boolean) => void,
        // Live thunk for the version-stale banner — null when the index is current.
        // Re-evaluated on open and after the Reindex action, so a heal that landed
        // between opens clears the banner. Optional (absent in tests / headless).
        private getIndexNotice?: () => IndexBanner | null,
        // Optional seed from a deep link (obsidian://seek?query=…). Empty for the
        // palette command. Applied in onOpen once the field exists.
        private initialQuery = '',
    ) {
        super(app);
        this.orchestrator = orchestrator;
        this.logger = logger;
        this.modelReady = modelStatus.ready;
        this.modelReadyPromise = modelStatus.promise;
        this.onSearchActivity = onSearchActivity;
        this.onQueryInFlight = onQueryInFlight;
    }

    // Arm/reset the "query settled" debounce: CATCHUP_SETTLE_MS after the last
    // keystroke (modal still open, foreground) we tell the plugin the session is
    // idle so it can drain catch-up in the safest window.
    private armSettle(): void {
        if (this.settleTimer != null) window.clearTimeout(this.settleTimer);
        this.settleTimer = window.setTimeout(() => {
            this.settleTimer = null;
            this.onSearchActivity?.(false);
        }, CATCHUP_SETTLE_MS);
    }

    onOpen(): void {
        this.modalEl.addClass('seek-modal');
        // Pin the modal high + narrow (quick-switcher feel) via the container.
        this.containerEl.addClass('seek-modal-container');
        // Mobile gets a distinct shell (full-width, keyboard-aware height) and the
        // touch gesture to dismiss the soft keyboard — see setupMobile() and the
        // `.seek-mobile` rules in styles.css. Gated so desktop is untouched.
        if (Platform.isMobile) this.modalEl.addClass('seek-mobile');
        // Tablets (iPad / Android tablet) are Platform.isMobile too, so they pick
        // up the full-bleed `.seek-mobile` shell — but edge-to-edge result rows run
        // the snippets uncomfortably wide on a big screen. `.seek-tablet` caps the
        // width back to a centred reading column (see styles.css). Same surface
        // distinction capabilityDefault() uses for the WebGPU allowlist.
        if (Platform.isTablet) this.modalEl.addClass('seek-tablet');
        // Activate the component now so MarkdownRenderer can register snippet
        // child-renderers against a loaded parent.
        this.markdownComponent.load();
        const { contentEl } = this;
        contentEl.empty();

        // Build the suggestion dictionaries once (one in-memory cache pass) and
        // the tag set used to flag non-binding `tag:` pills.
        this.suggester = new SuggestEngine().build(this.app, this.settings);
        this.vaultTagSet = this.collectVaultTags();

        // Component 1 — the token/pill query field. The 4th arg gates after:/before:
        // date filters on Recency being ON (a date field exists to key off — D4);
        // numeric-key validation (the red error pill) routes through the shared
        // suggester (SuggestEngine.isNumericKey), so no extra callback is needed.
        // Name of the date field after:/before: compare — the user's Recency
        // selection (their chosen frontmatter property, or file-modified time),
        // mirroring the Settings → Recency picker. Drives the suggestion hint so
        // it labels the real field instead of a hardcoded `created`.
        const dateFieldLabel = this.settings.recencyKey === 'modified' ? 'modified' : this.settings.createdProp;
        this.field = new PillQueryField(contentEl, this.suggester, {
            onQueryChange: q => this.scheduleSearch(q),
            onNavigate: dir => this.moveSelection(dir),
            onSubmit: newTab => this.openSelected(newTab),
            onDismiss: () => this.close(),
            validateTag: tag => this.tagBinds(tag),
        }, this.settings.recencyEpsilon > 0, dateFieldLabel);
        this.field.focus();

        // Seed from a deep link (obsidian://seek?query=…). setQuery emits the
        // field's onQueryChange, which routes through scheduleSearch — the exact
        // path a keystroke takes, so cold-start gating and inline filters behave
        // identically to typing it by hand.
        if (this.initialQuery.trim()) this.field.setQuery(this.initialQuery);

        // Version-stale banner slot, fixed between the field and the results. Empty (and
        // CSS-collapsed) unless the index was built under an older Seek version.
        this.bannerSlot = contentEl.createDiv({ cls: 'seek-banner-slot' });
        this.renderIndexBanner();

        this.resultsEl = contentEl.createDiv({ cls: 'seek-results' });
        this.renderEmpty();

        // Mobile-only wiring: keyboard-aware modal height + touch-to-dismiss.
        if (Platform.isMobile) this.setupMobile();

        // Component 3 — the footer hotkey bar. Gated on showHotkeyHints: OFF
        // gives a minimal "full results only" modal (just field + results).
        if (this.settings.showHotkeyHints) this.buildFooter(contentEl);

        // Observe the in-flight model load (no-op on the warm path). On the cold
        // path: when the model resolves, refresh the empty-state copy so the
        // resting UI doesn't lie. We deliberately don't auto-trigger a search —
        // the user's keystrokes already fired a runSearch awaiting the same
        // promise, so it proceeds on its own.
        // Ambient model-state affordance (replaces the old "model loaded"
        // toast): the search glyph rests dimmed and brightens once the embedder
        // is ready. Warm opens set it on now (synchronous → no animation); cold
        // opens fade it in when the promise resolves. Input stays live throughout.
        this.field.setModelReady(this.modelReady);
        if (!this.modelReady) {
            this.modelReadyPromise.then(() => {
                // The modal may have closed while the model was still loading
                // (e.g. the user dismissed it before the cold start finished) —
                // don't touch the (unloaded) field or paint into detached DOM.
                if (this.closed) return;
                this.modelReady = true;
                this.field?.setModelReady(true);
                if (!this.lastQuery.trim()) this.renderEmpty();
            }).catch(err => {
                if (this.closed) return;
                this.modelLoadError = err instanceof Error ? err : new Error(String(err));
                // Leave the glyph dimmed — it honestly reflects "never loaded";
                // the failure itself is surfaced as status text / a Notice.
                if (!this.lastQuery.trim()) {
                    this.renderStatus(`Model load failed: ${this.modelLoadError.message}`);
                }
            });
        }

        // Warm the search frame + BM25 the moment the modal opens, so the cold
        // IDB/frame/BM25 build overlaps the model load and the user's first
        // keystrokes instead of blocking the first query (the cold selectFetchMs /
        // binary-scan / BM25-fit cost). See warmIndex().
        this.warmIndex();

        // Cold-start onboarding: probe the store off the critical path. If nothing is
        // indexed, swap the resting copy for "your vault isn't indexed yet" guidance.
        void this.checkIndexState();

        // A live query session is now fully set up — pause any in-flight catch-up
        // drain so the foreground embed never competes with what the user is about
        // to type. Placed at the END of onOpen so a throw mid-setup can't leave the
        // flag stuck true (which would silently block all later catch-up).
        this.onSearchActivity?.(true);
    }

    // Prime the index caches off the critical path. On mobile warmCaches bails
    // until the embedder is loaded (a memory-spike guard while the ~250 MB model
    // is mid-load — see search.ts), so on the cold path we warm once the model
    // resolves; on the warm path (model already resident — the common desktop
    // case) we warm immediately. One call on every platform: the platform-specific
    // deferral lives in warmCaches, which self-guards and is a no-op once the
    // frame matches the live generation, so a redundant open never costs anything.
    private warmIndex(): void {
        const warm = () => { void this.orchestrator.warmCaches('modal-open'); };
        if (this.modelReady) warm();
        else this.modelReadyPromise.then(warm).catch(() => { /* model-load failure already surfaced */ });
    }

    onClose(): void {
        // Flip first, before any other teardown: everything below (and every
        // guard this flag feeds) assumes "closed" is already visible to any
        // async completion that races this call.
        this.closed = true;
        if (this.timer != null) window.clearTimeout(this.timer);
        if (this.settleTimer != null) { window.clearTimeout(this.settleTimer); this.settleTimer = null; }
        // Session ended — trigger the catch-up drain (the backup window to the
        // settle timer). The plugin no-ops if the app is hidden / nothing pending.
        this.onSearchActivity?.(false);
        // A modal closed mid-query (tap a result before it paints) would otherwise
        // leave the plugin's in-flight count latched > 0 and block indexing forever.
        // Force the false edge; runSearch's finally is guarded against a double end.
        if (this.inFlight !== 0) { this.inFlight = 0; this.onQueryInFlight?.(false); }
        this.detachViewport?.();
        this.detachViewport = null;
        this.markdownComponent.unload();
        this.revealObserver?.disconnect();
        this.revealObserver = null;
        this.sentinelEl = null;
        this.shownCount = 0;
        this.rows = [];
        // Drop the banner slot ref before empty() detaches it — defensive hygiene so any
        // stray renderIndexBanner call can never touch an orphaned node. (The banner is a
        // pure signpost now; it's only painted in onOpen, so this is belt-and-suspenders.)
        this.bannerSlot = null;
        this.contentEl.empty();
    }

    // Mobile-only behaviour, wired once on open. Two problems desktop never has:
    //
    //  1. The soft keyboard occludes the lower half of a `vh`-sized modal (vh is
    //     measured against the FULL screen, ignoring the keyboard), hiding the
    //     scrollable results and the footer behind it. We bind the modal's height
    //     to window.visualViewport — the area NOT covered by the keyboard — via a
    //     --seek-vvh custom property the `.seek-mobile` CSS consumes, so the
    //     footer sits just above the keyboard and the result list scrolls in the
    //     gap. The listener updates it as the keyboard shows/hides/resizes.
    //
    //  2. There's no way to dismiss the keyboard. The canonical gesture is "drag
    //     the result list to scroll → keyboard drops" (iOS UIScrollView's
    //     .onDrag). The hard part is WHEN to blur, because blurring resizes the
    //     visual viewport and the resulting reflow (via #1's listener) is a
    //     gesture-killer mid-touch: iOS cancels an in-flight overflow scroll the
    //     instant its container's geometry changes underneath it, and suppresses
    //     a click whose target moved. So blurring on `touchstart` ate the first
    //     TAP, and blurring on `touchmove` ate the first DRAG (it only dismissed;
    //     scrolling needed a second drag once geometry settled). The fix is to
    //     never blur DURING the gesture: the list is already sized to the
    //     above-keyboard gap (--seek-vvh), so it scrolls fine with the keyboard
    //     up; we just record that a drag happened (`touchmove`) and blur on
    //     `touchend`. The post-gesture reflow then grows the modal with nothing
    //     in flight to cancel. A stationary tap never sets `dragged`, so it opens
    //     a result first-try. Passive throughout — we never block the scroll.
    private setupMobile(): void {
        let dragged = false;
        this.resultsEl?.addEventListener('touchmove', () => { dragged = true; }, { passive: true });
        this.resultsEl?.addEventListener('touchend', () => {
            if (dragged) this.field?.blur();
            dragged = false;
        }, { passive: true });

        const vv = window.visualViewport;
        if (!vv) return;
        const apply = () => this.modalEl.style.setProperty('--seek-vvh', `${Math.round(vv.height)}px`);
        apply();
        vv.addEventListener('resize', apply);
        vv.addEventListener('scroll', apply);
        this.detachViewport = () => {
            vv.removeEventListener('resize', apply);
            vv.removeEventListener('scroll', apply);
        };
    }

    // The footer legend: keyboard hints on the left, esc on the right. Each
    // glyph is a <kbd> cap styled from theme variables.
    private buildFooter(parent: HTMLElement): void {
        const foot = parent.createDiv({ cls: 'seek-foot' });
        const grp = (build: (g: HTMLElement) => void): void => {
            const g = foot.createSpan({ cls: 'seek-foot-grp' });
            build(g);
        };
        const kbd = (g: HTMLElement, key: string) => g.createEl('kbd', { text: key });
        grp(g => { kbd(g, '↑'); kbd(g, '↓'); g.createSpan({ text: ' navigate' }); });
        grp(g => { kbd(g, '↵'); g.createSpan({ text: ' open' }); });
        grp(g => { kbd(g, '⌘'); kbd(g, '↵'); g.createSpan({ text: ' new tab' }); });
        grp(g => { kbd(g, 'tab'); g.createSpan({ text: ' fill autosuggest' }); });
        // Copy a shareable obsidian://seek deep-link for the CURRENT query. The
        // builder percent-encodes (so a `#tag`/`[k:v]` filter survives the URL
        // fragment delimiter) — the whole reason this exists, since a hand-typed
        // link truncates at `#`. A real action, not a hint: the click also serves
        // as the user gesture the clipboard write needs on mobile WKWebView.
        grp(g => {
            const link = g.createEl('a', { cls: 'seek-foot-link', text: '⧉ copy link' });
            link.setAttr('role', 'button');
            link.addEventListener('click', () => void this.copySearchLink());
        });
        foot.createSpan({ cls: 'seek-foot-spacer' });
        grp(g => { kbd(g, 'esc'); g.createSpan({ text: ' close' }); });
    }

    // Build + copy an obsidian://seek deep-link for the current query. `vault` is
    // included so the link reopens THIS vault; the query is percent-encoded so a
    // `#tag`/`[k:v]` filter survives the URL fragment delimiter (the handler in
    // main.ts receives it decoded). No query yet → nudge instead of a dead link.
    private async copySearchLink(): Promise<void> {
        const query = this.lastQuery.trim();
        if (!query) { new Notice('Seek: type a search first'); return; }
        const url = `obsidian://seek?vault=${encodeURIComponent(this.app.vault.getName())}&query=${encodeURIComponent(query)}`;
        try {
            await navigator.clipboard.writeText(url);
            new Notice('Seek: search link copied');
        } catch {
            // Clipboard can reject (no user gesture / locked clipboard). Surface
            // the URL so the action never silently no-ops — copy it by hand.
            new Notice(url, 8000);
        }
    }

    // The last query string emitted by the field — used by the cold-start
    // handler to tell whether a search is already pending.
    private lastQuery = '';

    private scheduleSearch(query: string, immediate = false): void {
        // A CHANGED query invalidates the keyboard selection: hover/arrows set
        // selectedIndex against the old result set, and carrying a mid-list
        // index into the new set would make Enter open whatever happens to sit
        // at the old rank (and skew click-log rank telemetry). Same-query
        // re-renders keep the selection (the clamp in renderResults handles a
        // shrunken result count).
        if (query !== this.lastQuery) this.selectedIndex = 0;
        this.lastQuery = query;
        // The user is typing → still an active session; (re)arm the settle debounce
        // so catch-up only drains once they pause. Covers both real and cleared
        // queries (clear-then-walk-away should still eventually settle + drain).
        this.onSearchActivity?.(true);
        this.armSettle();
        if (this.timer != null) window.clearTimeout(this.timer);
        if (!query.trim()) {
            // Invalidate any in-flight search, not just the debounce timer:
            // runSearch's stale check is `id !== currentSearch`, so without the
            // bump a search dispatched 200 ms ago would paint results over the
            // empty state the user just cleared to. Drop the search context too
            // so a click/capture can't reference results that aren't shown.
            this.currentSearch++;
            this.latestSearchEntry = null;
            this.renderEmpty();
            return;
        }
        const run = () => { void this.runSearch(query.trim()); };
        if (immediate) run();
        else this.timer = window.setTimeout(run, DEBOUNCE_MS);
    }

    private async runSearch(query: string): Promise<void> {
        const id = ++this.currentSearch;

        // Cold-start gate. On the warm path `modelReady` is true and we skip
        // straight to "Searching…". On the cold path we show a distinct status
        // so the user knows their query is held, not lost, then await the same
        // promise every concurrent runSearch awaits. Whichever holds the latest
        // `id` wins; the others bail on the stale check below.
        if (!this.modelReady) {
            if (this.modelLoadError) {
                this.renderStatus(`Model load failed: ${this.modelLoadError.message}`);
                return;
            }
            this.renderStatus('Loading model… your query will run as soon as it’s ready.');
            try {
                await this.modelReadyPromise;
            } catch (e) {
                if (id !== this.currentSearch || this.closed) return;
                this.renderStatus(`Model load failed: ${e instanceof Error ? e.message : String(e)}`);
                return;
            }
            // Bail if a newer query superseded us during load OR the modal was
            // closed while we waited — either way there's nothing left to paint.
            if (id !== this.currentSearch || this.closed) return;
            this.modelReady = true;
        }

        this.beginInFlight();
        try {
            this.setSearching();
            const { results, entry } = await this.orchestrator.search(query, MAX_RESULTS);
            // Stale (a newer query landed) or the modal closed mid-search — in
            // the closed case `renderResults` would otherwise paint into
            // detached DOM and (via updateSentinel) spin up a fresh
            // IntersectionObserver that onClose already ran and will never
            // disconnect.
            if (id !== this.currentSearch || this.closed) return;
            this.latestSearchEntry = entry;
            this.latestResultsShown = results;
            this.latestSearchCompletedAt = performance.now();
            this.renderResults(results);
        } catch (e) {
            if (id !== this.currentSearch || this.closed) return;
            this.renderStatus(`Error: ${e instanceof Error ? e.message : String(e)}`);
        } finally {
            this.endInFlight();
        }
    }

    // Ref-counted "a query embed is in flight" reporting. begin before the embed/
    // search await, end in its finally; the 0↔1 edges drive onQueryInFlight so the
    // plugin holds indexing until the query genuinely completes. Counted (not a
    // bool) so a cold-path search overlapping a newer one emits one clean pair.
    private beginInFlight(): void {
        if (this.inFlight++ === 0) this.onQueryInFlight?.(true);
    }
    private endInFlight(): void {
        if (this.inFlight > 0 && --this.inFlight === 0) this.onQueryInFlight?.(false);
    }

    private renderEmpty(): void {
        if (this.indexEmpty) { this.renderNoIndex(); return; }
        // Resting state (no active query, index present): collapse the lower section to
        // nothing so the modal is just the search bar (+ the version-stale banner, if
        // any). The field placeholder ("Search your vault…") already says what to do, so
        // a "Type to search…" box is redundant chrome — and a tall idle body pushed the
        // mobile keyboard-aware layout around. Active-query feedback ("No notes match.",
        // "Searching…", errors) and the empty-index onboarding still render via their own
        // paths (renderResults / renderNoIndex); only this idle placeholder is dropped.
        this.renderResting();
    }

    // Empty the results body without painting a status line — the modal collapses to the
    // query field (and the banner slot, which self-collapses when empty). clearRows()
    // already empties the container + resets the row pool; we just drop the loading dim
    // and the cached result list so a later click can't reference a stale set.
    private renderResting(): void {
        if (!this.resultsEl) return;
        this.clearRows();
        this.currentResults = [];
        this.resultsEl.removeClass('is-loading');
    }

    // Full-replace the results area with a single status line. Resets the row
    // pool + count and clears any lingering loading dim.
    private renderStatus(msg: string): void {
        if (!this.resultsEl) return;
        this.clearRows();
        this.currentResults = [];
        this.resultsEl.removeClass('is-loading');
        this.resultsEl.createDiv({ cls: 'seek-empty', text: msg });
    }

    // Off-critical-path probe of "is anything indexed". Run once on open and re-run
    // whenever we'd otherwise show the empty-index onboarding screen, so a background
    // build that populates a freshly opened (still-empty) index self-clears the screen
    // without a reopen. Re-runnable + idempotent. A null count (store unreadable
    // mid-init) leaves the flag untouched. Only the RESTING copy is repainted — a live
    // query is left to the next keystroke's search, never stomped here.
    private async checkIndexState(): Promise<void> {
        const chunks = await this.orchestrator.indexedChunkCount();
        // The modal can close while this off-critical-path probe is still in
        // flight (it's kicked off fire-and-forget from onOpen) — don't repaint
        // a resting copy into a modal that's gone.
        if (chunks == null || this.closed) return;
        const wasEmpty = this.indexEmpty;
        this.indexEmpty = chunks === 0;
        if (this.indexEmpty === wasEmpty) return;            // verdict unchanged → nothing to repaint
        if (!this.lastQuery.trim()) this.renderEmpty();      // flip (usually empty→populated) → refresh resting copy
    }

    // Cold-start onboarding state: the index is empty (fresh install, or an evicted /
    // never-synced index), so no query can return anything. Point the user at the
    // settings reindex instead of a bare "Type to search…" / "No notes match.".
    private renderNoIndex(): void {
        if (!this.resultsEl) return;
        this.clearRows();
        this.currentResults = [];
        this.resultsEl.removeClass('is-loading');
        const box = this.resultsEl.createDiv({ cls: 'seek-empty seek-noindex' });
        box.createDiv({ cls: 'seek-noindex-title', text: 'Your vault isn’t indexed yet' });
        box.createDiv({ cls: 'seek-empty-sub', text: 'Seek needs to build a search index before it can find anything.' });
        box.createEl('button', { cls: 'seek-noindex-btn mod-cta', text: 'Open Seek settings to index' })
            .addEventListener('click', () => this.openSeekSettings());
    }

    // Open Obsidian's settings straight to the Seek tab. `app.setting` isn't in the
    // public typings but is a stable runtime API (same access pattern as the
    // metadataCache.getTags() call in collectVaultTags). Close the modal first so the
    // two overlays don't stack.
    private openSeekSettings(): void {
        this.close();
        const setting = (this.app as unknown as {
            setting?: { open(): void; openTabById(id: string): void };
        }).setting;
        setting?.open();
        setting?.openTabById('seek');
    }

    // Paint (or clear) the version-stale banner from the live thunk. Empties the slot
    // first so it's idempotent: re-evaluated on each open, it shows while the index is
    // version-stale and is gone once a reindex (from Settings) heals it. CSS hides an
    // empty slot, so the common (current-index) case adds no visible chrome. The button
    // opens Seek settings (which closes this modal) — the reindex itself lives there.
    private renderIndexBanner(): void {
        if (!this.bannerSlot) return;
        this.bannerSlot.empty();
        const notice = this.getIndexNotice?.();
        if (!notice) return;
        const banner = this.bannerSlot.createDiv({ cls: 'seek-index-banner' });
        // Calm (info) variant for the "syncing from another device" state; the default
        // warning style for the stale/action-needed state.
        banner.toggleClass('is-info', notice.tone === 'info');
        banner.createSpan({ cls: 'seek-index-banner-msg', text: notice.message });
        // Only the action-needed banner carries the reindex affordance; the syncing
        // banner has nothing for the user to do, so it shows no button.
        if (notice.showAction) {
            banner.createEl('button', { cls: 'seek-index-banner-btn mod-cta', text: 'Open settings' })
                .addEventListener('click', () => this.openSeekSettings());
        }
    }

    // Loading feedback that does NOT destroy what's on screen: dim the existing
    // results in place while the next set is fetched, so a fast warm search
    // shows no flash. Only a cold search with nothing to preserve falls back to
    // a text status.
    private setSearching(): void {
        if (!this.resultsEl) return;
        if (this.rows.length > 0) this.resultsEl.addClass('is-loading');
        else this.renderSkeleton();
    }

    // Cold-search loading state: placeholder rows with a CSS shimmer, shown when
    // there are no existing results to dim. The shimmer sweep animates `transform`
    // (styles.css), so it runs on the compositor thread and keeps moving even while
    // the JS main thread is blocked by a cold index build — the freeze a static
    // "Searching…" line couldn't hide. Same treatment on every platform. Rows carry
    // `.seek-result` for layout parity (so results swap in with no height jump);
    // pointer-events are killed in CSS so the placeholders aren't hoverable.
    private renderSkeleton(): void {
        if (!this.resultsEl) return;
        this.clearRows();
        this.currentResults = [];
        this.resultsEl.removeClass('is-loading');
        const PLACEHOLDER_ROWS = 5;
        for (let i = 0; i < PLACEHOLDER_ROWS; i++) {
            const row = this.resultsEl.createDiv({ cls: 'seek-result seek-skeleton' });
            row.createDiv({ cls: 'seek-skeleton-line seek-skeleton-title' });
            row.createDiv({ cls: 'seek-skeleton-line seek-skeleton-path' });
        }
    }

    private clearRows(): void {
        this.rows = [];
        this.shownCount = 0;
        // empty() will drop the sentinel node too; unobserve + forget it first so
        // the observer isn't left holding a detached element.
        if (this.sentinelEl) this.revealObserver?.unobserve(this.sentinelEl);
        this.sentinelEl = null;
        this.resultsEl?.empty();
    }

    // Snapshot all vault tags (lowercased, leading `#` stripped). metadataCache
    // .getTags() isn't in the public typings but is a stable runtime API.
    private collectVaultTags(): Set<string> {
        const cache = this.app.metadataCache as unknown as { getTags?: () => Record<string, number> };
        const raw = cache.getTags?.() ?? {};
        return new Set(Object.keys(raw).map(t => t.replace(/^#/, '').toLowerCase()));
    }

    // Does a `tag:` filter match any real vault tag, using the SAME hierarchical
    // rule the matcher applies (exact OR a `parent/` prefix)? `meetings/1x1`
    // matches `meetings/1x1` and `meetings/1x1/child`, but NOT the sibling
    // `meetings/1x1s` — so this predicts whether the pill will bind to anything.
    private tagBinds(filterTag: string): boolean {
        const ft = filterTag.replace(/^#/, '').toLowerCase();
        for (const vt of this.vaultTagSet) {
            if (vt === ft || vt.startsWith(ft + '/')) return true;
        }
        return false;
    }

    // Reconcile the row pool against the new result set in place. Each slot is
    // reused: only changed text repaints, and only changed snippet markdown
    // re-renders. The orchestrator already deduped to one-per-note upstream.
    private renderResults(results: ScoredChunk[]): void {
        const container = this.resultsEl;
        if (!container) return;
        container.removeClass('is-loading');
        // Drop any cold-start skeleton placeholders before reusing/creating real
        // rows — ensureRow appends to the container, so leftover skeletons would
        // sit above the results. (The empty-results path's clearRows() wipes them
        // too; this covers the non-empty reuse path.)
        container.querySelectorAll(':scope > .seek-skeleton').forEach(el => el.remove());
        this.currentResults = results;

        if (results.length === 0) {
            this.clearRows();
            // An empty index can't match anything — show the onboarding state, not the
            // misleading "Try removing a filter." that assumes a populated index. Re-probe
            // off the critical path so a since-populated index drops the flag (and the
            // screen, on the next render) instead of latching "not indexed" all session.
            if (this.indexEmpty) { this.renderNoIndex(); void this.checkIndexState(); return; }
            const empty = container.createDiv({ cls: 'seek-empty' });
            empty.createDiv({ text: 'No notes match.' });
            empty.createDiv({ cls: 'seek-empty-sub', text: 'Try removing a filter.' });
            return;
        }
        // A result-bearing search is definitive proof the index is populated — clear any
        // stale cold-start flag so the onboarding screen can't reappear this session.
        this.indexEmpty = false;

        // Drop a status/empty placeholder (coming from a cold search) without
        // disturbing real rows we may be about to reuse.
        container.querySelectorAll(':scope > .seek-empty').forEach(el => el.remove());

        // Only the first page is painted now; the rest reveal on scroll. Rows are
        // reused in place by index, so re-running a query that still has ≥PAGE_SIZE
        // hits repaints those slots rather than flashing.
        this.shownCount = Math.min(PAGE_SIZE, results.length);
        for (let i = 0; i < this.shownCount; i++) {
            this.applyRow(this.ensureRow(i), results[i], i + 1);
        }
        while (this.rows.length > this.shownCount) {
            this.rows.pop()?.el.remove();
        }

        // Keep the selection in range (clamp toward the top), then paint it.
        this.selectedIndex = Math.min(Math.max(0, this.selectedIndex), this.shownCount - 1);
        this.applySelection();
        this.updateSentinel();
    }

    // Grow the rendered window. Each call appends at least one PAGE_SIZE batch
    // (more if `target` demands it, e.g. a keyboard jump), capped at the fetched
    // result count. New rows append after the sentinel; updateSentinel then moves
    // the sentinel back to the tail — all synchronous, so no intermediate paint.
    private revealMore(target = 0): void {
        if (this.shownCount >= this.currentResults.length) return;
        const next = Math.min(
            Math.max(target, this.shownCount + PAGE_SIZE),
            this.currentResults.length,
        );
        for (let i = this.shownCount; i < next; i++) {
            this.applyRow(this.ensureRow(i), this.currentResults[i], i + 1);
        }
        this.shownCount = next;
        this.updateSentinel();
    }

    // Maintain the tail sentinel that drives infinite scroll. When more rows
    // remain, ensure a sentinel exists, sits last, and is observed; when the
    // window has reached the full result set, retire it so the observer goes
    // quiet. IntersectionObserver.observe is idempotent, so re-observing the
    // same node after a re-append is a no-op.
    private updateSentinel(): void {
        const container = this.resultsEl;
        if (!container) return;
        if (this.shownCount >= this.currentResults.length) {
            if (this.sentinelEl) {
                this.revealObserver?.unobserve(this.sentinelEl);
                this.sentinelEl.remove();
                this.sentinelEl = null;
            }
            return;
        }
        if (!this.sentinelEl) {
            this.sentinelEl = container.createDiv({ cls: 'seek-load-sentinel' });
        } else {
            container.appendChild(this.sentinelEl); // keep last, after freshly appended rows
        }
        this.ensureRevealObserver();
        this.revealObserver?.observe(this.sentinelEl);
    }

    private ensureRevealObserver(): void {
        if (this.revealObserver || !this.resultsEl) return;
        // root = the scrolling results list; the bottom margin pre-loads the next
        // page before the sentinel is actually on screen. Callbacks fire async
        // (next frame), so a cascade of reveals is naturally paced one page/frame
        // until the sentinel is pushed out of the margin band — never a sync loop.
        this.revealObserver = new IntersectionObserver(
            entries => { if (entries.some(e => e.isIntersecting)) this.revealMore(); },
            { root: this.resultsEl, rootMargin: `0px 0px ${REVEAL_MARGIN_PX}px 0px` },
        );
    }

    // Return the row at slot `i`, creating its DOM scaffold on first use. The
    // click handler is bound once and reads the row's live `data`/`rank`. Hover
    // moves the selection. Clicks on a rendered link inside the snippet fall
    // through to Obsidian's own link handler.
    private ensureRow(i: number): SeekResultRow {
        const existing = this.rows[i];
        if (existing) return existing;

        const el = this.resultsEl!.createDiv({ cls: 'seek-result' });
        const top = el.createDiv({ cls: 'seek-result-top' });
        const row: SeekResultRow = {
            el,
            titleEl: top.createDiv({ cls: 'seek-result-title' }),
            // Breadcrumb is parented to `el` (not `top`) so it drops onto its own
            // line below the title instead of competing for the title's horizontal
            // space and forcing a truncating ellipsis. Created after `top` → sits
            // directly under the title, above the snippet.
            breadcrumbEl: el.createDiv({ cls: 'seek-result-path' }),
            snippetEl: el.createDiv({ cls: 'seek-result-snippet' }),
            metaEl: el.createDiv({ cls: 'seek-result-meta' }),
            scoreEl: el.createDiv({ cls: 'seek-result-score' }),
            keycapEl: el.createEl('kbd', { cls: 'seek-result-kbd', text: '↵' }),
            data: null as unknown as ScoredChunk,
            rank: i + 1,
            lastSnippet: '\0', // sentinel ≠ any real snippet so first apply renders
            lastCrumb: '\0',   // same sentinel for the breadcrumb
        };
        el.addEventListener('click', e => {
            if ((e.target as HTMLElement).closest('a')) return;
            void this.openResult(row.data, this.rows.indexOf(row) + 1, e.metaKey || e.ctrlKey);
        });
        el.addEventListener('mousemove', () => {
            const idx = this.rows.indexOf(row);
            // Hover paints the selection but must NOT scroll: applySelection's
            // scroll-into-view would shift content under the stationary cursor,
            // firing a synthetic mousemove on the next row → scroll → repeat,
            // an edge-parked feedback loop that looks like the list creeping on
            // its own. Keyboard nav still scrolls (it can target off-screen).
            if (idx !== this.selectedIndex) { this.selectedIndex = idx; this.applySelection(false); }
        });
        this.rows[i] = row;
        return row;
    }

    // Update one row's contents to a result, repainting only what changed.
    private applyRow(row: SeekResultRow, r: ScoredChunk, rank: number): void {
        row.data = r;
        row.rank = rank;

        const title = noteTitle(r.note_path);
        if (row.titleEl.textContent !== title) row.titleEl.setText(title);

        // Heading-path breadcrumb: `› Agenda › Intern pgm` (empty for a
        // whole-note / pre-heading chunk). Rendered as MARKDOWN, not plain text,
        // so a heading that is itself a link — an external `[Review…](https://…)`
        // or a `[[wikilink]]` — shows as a real Obsidian link with the box glyph +
        // underline the user's appearance settings define, exactly as on the page
        // (a bare setText left the `[text](url)` source showing). The per-segment
        // `› ` prefix doubles as a guard against a heading that starts with block
        // markdown (`1.`, `#`) being parsed as a list/heading. Skipped when the
        // path is unchanged (async render is comparatively costly).
        const crumb = (r.heading_path ?? []).map(s => `› ${s}`).join(' ');
        if (crumb !== row.lastCrumb) {
            row.lastCrumb = crumb;
            row.breadcrumbEl.empty();
            if (crumb) {
                // Fresh wrapper (as the snippet does) so a late async append from a
                // superseded row lands on a detached node, never another row's line.
                const wrap = row.breadcrumbEl.createDiv();
                MarkdownRenderer.render(this.app, crumb, wrap, r.note_path, this.markdownComponent)
                    .catch(() => wrap.setText(crumb));
            }
        }
        row.breadcrumbEl.toggle(crumb.length > 0);

        // Meta line: `created <date> · #tag #tag`. Rebuilt only when it changes.
        const created = fmtCreated(r.metadata?.created ?? null);
        const tags = r.metadata?.tags ?? [];
        const metaSig = created + '|' + tags.join(',');
        if (row.metaEl.dataset.sig !== metaSig) {
            row.metaEl.dataset.sig = metaSig;
            row.metaEl.empty();
            if (created) {
                const c = row.metaEl.createSpan({ cls: 'seek-meta-created' });
                c.createSpan({ cls: 'seek-meta-lbl', text: 'created ' });
                c.appendText(created);
            }
            if (created && tags.length) row.metaEl.createSpan({ cls: 'seek-meta-dot', text: '·' });
            for (const t of tags) row.metaEl.createSpan({ cls: 'seek-meta-tag', text: `#${t}` });
            row.metaEl.toggle(created.length > 0 || tags.length > 0);
        }

        // Score line, gated on the Display scores setting. Shows the calibrated
        // "Matching %" (the headline relevance read) plus the two ranking bonuses
        // it EXCLUDES — recency and title boost — so a rank-vs-strength divergence
        // stays legible at a glance.
        const conf = r.ranking_signals.confidence;
        const strength = matchStrength(
            conf, r.ranking_signals.bm25, this.settings.denseWeight, r.lexicalOnly);
        // A filter-only / browse query (no free text) has nothing to "match"
        // against — suppress the score entirely rather than score a non-match.
        const hasTextQuery = (this.latestSearchEntry?.cleanedQuery ?? '').trim().length > 0;
        // Below MATCH_STRENGTH_MIN_NOTES notes the dense background is too sparse to
        // calibrate (the settings toggle is disabled there too). strength == null
        // means the corpus isn't calibrated yet → hide the line, no rank fallback.
        const scoresMeaningful = this.app.vault.getMarkdownFiles().length >= MATCH_STRENGTH_MIN_NOTES;
        if (this.settings.showScores && scoresMeaningful && hasTextQuery && strength != null) {
            // Title shown as a normalized [0,1] match strength (1 = full known-item
            // title match), mirroring how recency renders its raw signal rather than
            // the weighted contribution that enters `final`. title_boost is
            // navTitleBoost·coverage (fusion.ts), so divide the configured weight
            // back out to recover coverage; the Off stage (weight 0) zeroes the
            // contribution and coverage isn't recoverable, so it reads 0.00.
            const titleWeight = this.settings.navTitleBoost;
            const titleStrength = titleWeight > 0
                ? r.ranking_signals.title_boost / titleWeight
                : 0;
            const label = `Matching ${Math.round(strength * 100)}%`
                + ` · recency ${r.ranking_signals.recency.toFixed(2)}`
                + ` · title ${titleStrength.toFixed(2)}`;
            if (row.scoreEl.textContent !== label) row.scoreEl.setText(label);
            row.scoreEl.show();
        } else {
            row.scoreEl.hide();
        }

        const snippet = sanitizeSnippet(r.snippet ?? '');
        if (snippet === row.lastSnippet) return; // unchanged — skip the markdown re-render
        row.lastSnippet = snippet;
        row.snippetEl.empty();
        row.snippetEl.toggle(snippet.length > 0);
        if (!snippet) return;

        // Render into a fresh wrapper rather than snippetEl directly: if a later
        // search supersedes this row and empty()s snippetEl, this (now detached)
        // wrapper absorbs any late async append, so results never interleave.
        const wrapper = row.snippetEl.createDiv();
        MarkdownRenderer.render(this.app, snippet, wrapper, r.note_path, this.markdownComponent)
            .catch(() => wrapper.setText(snippet));
    }

    // ---- keyboard selection model ----

    private moveSelection(dir: 1 | -1): void {
        if (this.currentResults.length === 0) return;
        const next = Math.min(Math.max(0, this.selectedIndex + dir), this.currentResults.length - 1);
        if (next === this.selectedIndex) return;
        // Arrowing into the not-yet-rendered window pulls the next page in first,
        // so keyboard users reach all MAX_RESULTS rows without touching the mouse.
        if (next >= this.shownCount) this.revealMore(next + 1);
        this.selectedIndex = next;
        this.applySelection();
    }

    // Paint the selected row (class + ↵ keycap) and, when `scroll`, keep it in
    // view. Manual scroll math against the list's scrollTop/clientHeight, NOT
    // scrollIntoView (which would also scroll the whole modal/page). Hover passes
    // scroll=false to avoid a mousemove↔scroll feedback loop (see the row's
    // mousemove handler); keyboard nav leaves it on to chase off-screen rows.
    private applySelection(scroll = true): void {
        this.rows.forEach((row, i) => {
            const sel = i === this.selectedIndex;
            row.el.toggleClass('is-selected', sel);
            row.keycapEl.toggle(sel);
        });
        if (!scroll) return;
        const list = this.resultsEl;
        const row = this.rows[this.selectedIndex];
        if (!list || !row) return;
        const rt = row.el.offsetTop;
        const rb = rt + row.el.offsetHeight;
        if (rt < list.scrollTop) list.scrollTop = rt - 6;
        else if (rb > list.scrollTop + list.clientHeight) list.scrollTop = rb - list.clientHeight + 6;
    }

    private openSelected(newTab: boolean): void {
        const r = this.currentResults[this.selectedIndex];
        if (r) void this.openResult(r, this.selectedIndex + 1, newTab);
    }

    private async openResult(r: ScoredChunk, rank: number, newTab: boolean): Promise<void> {
        // Emit click event BEFORE opening the file — the file-open switches
        // workspace state and might cancel pending work. We don't await the
        // logger write so click latency stays imperceptible.
        this.emitClick(r, rank);

        const file = this.app.vault.getAbstractFileByPath(r.note_path);
        if (!(file instanceof TFile)) return;

        // A .base is a saved query/view, not editable text. Skip the markdown
        // highlight + scroll path (buildMatchHighlight/scrollLeafToChunk assume a
        // text editor) and drive the Bases view directly: the matched view name
        // rides in heading_path (chunkBase puts it there), so we land on that exact
        // view. A base-level chunk (empty heading_path) has no viewName, so the
        // Bases view opens its default/last-used view. Mirrors the markdown path's
        // modal semantics: a background new tab keeps the modal open + focused; a
        // plain open takes the active tab and dismisses.
        if (file.extension === 'base') {
            const viewName = r.heading_path?.[r.heading_path.length - 1];
            const state: Record<string, unknown> = viewName ? { file: file.path, viewName } : { file: file.path };
            if (newTab) {
                const leaf = this.app.workspace.getLeaf('tab');
                await leaf.setViewState({ type: 'bases', active: false, state });
                this.field?.focus();
                return;
            }
            const leaf = this.app.workspace.getLeaf(false);
            await leaf.setViewState({ type: 'bases', active: true, state });
            this.close();
            return;
        }

        // Native search-style highlight of the matched terms (same transient
        // flash core Search uses), passed via ephemeral state on the open call.
        const eState = await this.buildMatchHighlight(file, r);

        if (newTab) {
            // ⌘/Ctrl+Enter or ⌘/Ctrl+click → open in a BACKGROUND new tab and
            // KEEP the modal open + focused, so the user can fan out several
            // results in one session without the modal dismissing on them.
            // `active: false` is what stops the new leaf from stealing focus
            // away from the search field.
            const leaf = this.app.workspace.getLeaf('tab');
            await leaf.openFile(file, { active: false, eState });
            this.scrollLeafToChunk(leaf, r);
            this.field?.focus();
            return;
        }

        // Plain open (Enter / click): replace the active tab and dismiss.
        const leaf = this.app.workspace.getLeaf(false);
        await leaf.openFile(file, { eState });
        this.scrollLeafToChunk(leaf, r);
        this.close();
    }

    // Build the ephemeral-state `match` payload Obsidian's view layer uses to
    // paint the transient search highlight. `match.matches` are [from,to] char
    // offsets into `match.content` (the full note text) — the same contract
    // core Search and heading-link navigation rely on (untyped in the public
    // API, hence the `Record<string,unknown>` return). Returns undefined when
    // there's nothing to highlight so the open call falls back to plain nav.
    //
    // Chunks store only start_line (no char offset), so we resolve the chunk's
    // first-line offset here and search the chunk window for the query tokens —
    // the same "earliest matching token in the chunk" logic makeSnippet uses,
    // so the highlight lands on the text the snippet already showed.
    private async buildMatchHighlight(file: TFile, r: ScoredChunk): Promise<Record<string, unknown> | undefined> {
        if (r.start_line <= 0) return undefined;
        const tokens = (this.latestSearchEntry?.cleanedQuery ?? '')
            .toLowerCase().split(/\s+/).filter(Boolean);
        if (tokens.length === 0) return undefined;

        const content = await this.app.vault.cachedRead(file);

        // Walk to the char offset of the chunk's first line (start_line is 1-based).
        let lineStart = 0;
        for (let i = 0; i < r.start_line - 1; i++) {
            const nl = content.indexOf('\n', lineStart);
            if (nl === -1) { lineStart = content.length; break; }
            lineStart = nl + 1;
        }
        // Bound the search to the chunk body (+slack for snippet trailing context)
        // so we never highlight a same-token hit elsewhere in the note.
        const windowEnd = Math.min(content.length, lineStart + r.content.length + 200);

        // Find + order the highlight ranges over a markup-masked view of the note
        // (word-boundary match per query token, stopwords/single-chars skipped via
        // the SAME ENGLISH_STOPWORDS set BM25 uses; see ./highlight). buildHighlight-
        // Ranges returns them sorted ascending & disjoint — required because eState's
        // CodeMirror RangeSet mis-paints out-of-order ranges, which is exactly what
        // happens when a later query token matches EARLIER in the note than an
        // earlier token (a term that also appears in the title or an intro line).
        const matches = buildHighlightRanges(content, tokens, lineStart, windowEnd, ENGLISH_STOPWORDS);
        if (matches.length === 0) return undefined;
        return { match: { content, matches } };
    }

    // Move the leaf's editor cursor to the matched chunk's start line and scroll
    // it into view. Works on a background (active: false) leaf too, so a fanned-
    // out new tab still lands on the right chunk.
    private scrollLeafToChunk(leaf: { view: unknown }, r: ScoredChunk): void {
        const view = leaf.view;
        if (view instanceof MarkdownView && r.start_line > 0) {
            const editor = view.editor;
            editor.setCursor({ line: Math.max(0, r.start_line - 1), ch: 0 });
            editor.scrollIntoView({
                from: { line: r.start_line - 1, ch: 0 },
                to: { line: r.end_line, ch: 0 },
            }, true);
        }
    }

    private emitClick(r: ScoredChunk, rank: number): void {
        const entry = this.latestSearchEntry;
        if (!entry) return; // stale render or no search context — drop
        const dwellMs = this.latestSearchCompletedAt > 0
            ? performance.now() - this.latestSearchCompletedAt
            : 0;
        const click: ClickEntry = {
            type: 'click',
            timestamp: new Date().toISOString(),
            searchId: entry.searchId,
            query: entry.query,
            chunk_id: r.chunk_id,
            note_path: r.note_path,
            rank,
            score: r.score,
            dense: r.ranking_signals.dense,
            bm25: r.ranking_signals.bm25,
            recency: r.ranking_signals.recency,
            title_boost: r.ranking_signals.title_boost,
            dwellMs: parseFloat(dwellMs.toFixed(0)),
            shownTop10: this.latestResultsShown.slice(0, 10).map(c => c.chunk_id),
        };
        // Fire-and-forget; click latency matters more than a guaranteed write.
        this.logger.append(click).catch(e => console.error('[seek] click log failed:', e));
    }
}
