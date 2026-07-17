// src/settings-tab.ts — Seek's Settings tab.
//
// Redesigned 2026-06-19 (see plan "Seek — Settings Tab Redesign + Default
// Ratification"): an opinionated, status-led tab in place of the old flat debug
// surface. Section order is intentional — Index leads (the one operational concern),
// relevance recedes behind a disclosure fronted by a teaching pipeline diagram, and
// the model/compute story gets a home. Section IA, top→bottom:
//   Index → Relevance → Display → Model & performance → Reset → About
//
// Built native: Obsidian `Setting` rows + a few custom DOM helpers (segmented control,
// status card, pipeline diagram, progress row), all styled from theme CSS variables in
// styles.css so the tab absorbs the user's theme + dark mode. The validated debug knobs
// (prefix / synonym / headings / coverage / properties / boosted-BM25 / sidecar toggle)
// are now silent defaults and deliberately NOT surfaced — see DEFAULT_SETTINGS + the
// rev-5 migration in types.ts/main.ts.

import { App, PluginSettingTab, Setting, Notice, setIcon } from 'obsidian';
import type SeekPlugin from './main';
import type { IndexStats, ModelStatus } from './main';
import type { AltOpenLocation, SidecarIndexLocation } from './types';
import { DEFAULT_SETTINGS, MATCH_STRENGTH_MIN_NOTES } from './types';
import {
    getBackendOverride, setBackendOverride, isWebgpuDemoted, clearWebgpuDemoted,
    type BackendChoice,
} from './platform';
import { enumerateDatePropertyNames } from './prop-types';

// Real repo/social URLs for the About footer.
const GITHUB_URL = 'https://github.com/tooape/Obsidian-Seek-prototype';
const X_URL = 'https://x.com/tooape';
const DOCS_URL = 'https://publish.obsidian.md/rmm/Seek+Documentation/About+Seek';

// The X (Twitter) logo as an inline SVG path. Obsidian's bundled Lucide no longer
// ships a `twitter`/`x` brand icon, so setIcon('twitter') rendered an empty box —
// we draw the glyph ourselves instead (see brandLink). Filled (fill=currentColor)
// so it inherits the icon button's colour like the Lucide icons do.
const X_LOGO_PATH = 'M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z';

// ISO-8601 local stamp (YYYY-MM-DD HH:MM) for the status card — the vault's date
// convention, replacing the locale "6/19/2026, 8:10:32 PM" the card showed before.
// Local components (not toISOString's UTC) so the time matches the user's clock;
// seconds are dropped as noise for a "last index" marker.
function fmtStamp(iso: string): string {
    const d = new Date(iso);
    const p = (n: number) => String(n).padStart(2, '0');
    return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}

// (The date-property picker enumerator moved to prop-types.ts, shared with the
// typed-value inline filters — see enumerateDatePropertyNames import above.)

// ---- segmented (pill) stages ------------------------------------------------------
type Stage = 'Off' | 'Default' | 'High';

// Title bonus maps the navTitleBoost scalar onto three stages. The eval-swept knee is
// 0.8 (now reachable as High); the 2026-06-19 ratification ships Default=0.5 (see
// types.ts navTitleBoost). titleStageOf() snaps any persisted/off-grid value to the
// nearest stage so a hand-tuned legacy value still selects something sensible.
const TITLE_VALUE: Record<Stage, number> = { Off: 0, Default: 0.5, High: 0.8 };
function titleStageOf(v: number): Stage {
    let best: Stage = 'Default';
    for (const stage of ['Off', 'Default', 'High'] as Stage[]) {
        if (Math.abs(TITLE_VALUE[stage] - v) < Math.abs(TITLE_VALUE[best] - v)) best = stage;
    }
    return best;
}

// Recency bonus maps {recencyEpsilon, recencyHalfLifeDays} onto three stages. Off=ε0
// (ships Off), Default=0.04·180d, High=0.1·90d (see types.ts recencyEpsilon). ε≤0 is
// Off; otherwise snap to the nearer of Default/High by ε.
//
// High's half-life SHORTENS past Default (180→90) — it does not lengthen. High shipped
// at 270d through 2026-07-16, which made it inert at the thing it advertises: ε is the
// budget, but the half-life decides how much of that budget is spent in the age band
// the query actually spans. Live-vault measurement ("brian 1x1", ~40 dated siblings,
// full-series base hybrid+title spread 0.032): the today-vs-36d recency swing is 0.0088
// at 270d, so the newest sibling sat at rank 9 of its own series; at 90d it is 0.0242 —
// enough for rank 3, not rank 1, since 0.0242 is still under the series spread (it only
// has to beat the gap to each competitor, never the full range). ε was never the
// problem; 0.1 was already enough budget.
//
// High IS a lean, deliberately — types.ts recencyEpsilon says so, and that is why it is
// opt-in rather than the default. Over the competitive 0–100d band the 90d recency range
// (~0.054) exceeds the 0.032 sibling spread, so inside a dated series date now leads
// relevance. ranker.ts's "ε must NEVER become a lean" governs the always-on DEFAULT
// tiebreaker (ε 0.02, now 0), not this opt-in stage. 270 making High *gentler* than
// Default was the incoherence.
//
// 90 is anchored, not swept: the 06-04 click study's MEDIAN episodic click target is 83d
// old (see types.ts recencyHalfLifeDays), so the decay's knee sits on the click mass.
// Don't chase shorter. What a short half-life buys is how far a 0-day note outruns the
// pool's TYPICAL AGE — against a 60d note a brand-new one gains +0.014 at 270d but +0.075
// at 30d — so the shorter it gets, the more a fresh note overtakes a hybrid deficit it
// never earned. (The term's total range over 0–730d is ~0.085–0.100 at EVERY half-life,
// so that number tells you nothing; the advantage over the pool's real age mass is the
// one that moves ranks.) Swept live on a FLAT no-opinion pool — a query nothing matches:
// purely topical through 180d, one recent intruder at 120–60d, intruder at rank 2 by 45d,
// and at 30d today's notes displace the topical results outright, which is the 30d-cutoff
// bug the smooth decay replaced. A monotonic slide, not a cliff: 90 buys the dated-series
// fix while a flat pool stays essentially topical. See [[seek-recency-halflife-high-mode]].
const RECENCY_VALUE: Record<Stage, { eps: number; hl: number }> = {
    Off: { eps: 0, hl: 180 },
    Default: { eps: 0.04, hl: 180 },
    High: { eps: 0.1, hl: 90 },
};
function recencyStageOf(eps: number): Stage {
    if (eps <= 0) return 'Off';
    return Math.abs(eps - RECENCY_VALUE.Default.eps) <= Math.abs(eps - RECENCY_VALUE.High.eps) ? 'Default' : 'High';
}

// Search strategy: Balanced (denseWeight 0.8) vs Keyword focused (0.3). Concept-focused
// (0.9) was cut. Split at the midpoint so a legacy denseWeight still resolves a side.
type Strategy = 'balanced' | 'keyword';
const STRATEGY_VALUE: Record<Strategy, number> = { balanced: 0.8, keyword: 0.3 };
function strategyOf(denseWeight: number): Strategy {
    return denseWeight <= 0.55 ? 'keyword' : 'balanced';
}

export class SeekSettingTab extends PluginSettingTab {
    // Async index/model snapshots, loaded once per tab open (guarded null→fetch→re-render).
    private stats: IndexStats | null = null;
    private modelStatus: ModelStatus | null = null;
    private loading = false;
    // UI state that must survive the synchronous display() re-renders triggered by
    // segmented picks and the reindex state machine.
    private advancedOpen = false;
    // Independent of advancedOpen (Relevance) so the Index disclosure toggles on its own.
    private indexAdvancedOpen = false;
    private reindexPhase: 'idle' | 'confirm' | 'running' = 'idle';
    private reindexDone = 0;
    private reindexTotal = 0;
    // Live-progress DOM refs, repointed on each display() so the runFullReindex
    // onProgress callback always paints the current node (robust to close/reopen).
    private progressFillEl: HTMLElement | null = null;
    private progressLabelEl: HTMLElement | null = null;
    // Transient "downloading…" flag for the model section (no byte progress available).
    private modelDownloading = false;
    // Model-delete state: two-step confirm (Delete → Cancel / Delete model) so a
    // destructive ~100 MB cache wipe can't fire on a single click, plus an in-progress
    // flag for the "Deleting…" feedback. Both reset on hide() so reopening is clean.
    private modelDeleteConfirm = false;
    private modelDeleting = false;

    constructor(app: App, private plugin: SeekPlugin) {
        super(app, plugin);
    }

    display(): void {
        // Fetch the async snapshots once; re-render when they land.
        if (!this.stats && !this.loading) void this.loadData();

        const { containerEl } = this;
        containerEl.empty();
        containerEl.addClass('seek-settings');

        this.renderIndex(containerEl);
        this.renderRelevance(containerEl);
        this.renderDisplay(containerEl);
        this.renderModel(containerEl);
        this.renderReset(containerEl);
        this.renderAbout(containerEl);
    }

    // A full display() is the simplest way to reflect cross-control dependencies (the
    // pipeline diagram reacting to strategy, the date picker enabling with recency, the
    // reindex state machine). But empty()+rebuild resets the scroll container to the top,
    // which on a long tab — especially the full-screen mobile settings — yanks the user
    // away from the control they just tapped. rerender() preserves the scroll offset
    // across the rebuild, so every segmented pick / disclosure toggle stays put. All
    // interaction-driven re-renders go through here; only Obsidian's initial display()
    // (which opens at the top anyway) calls display() directly.
    private rerender(): void {
        const scroller = this.findScroller();
        const top = scroller ? scroller.scrollTop : 0;
        this.display();
        if (scroller) scroller.scrollTop = top;
    }

    // Nearest scrollable ancestor (containerEl included). The settings scroll container
    // differs between desktop (.vertical-tab-content) and mobile, so we detect it by
    // overflow rather than hardcoding a selector. Null if nothing scrolls (short tab).
    private findScroller(): HTMLElement | null {
        let el: HTMLElement | null = this.containerEl;
        while (el) {
            if (el.scrollHeight > el.clientHeight + 1 && getComputedStyle(el).overflowY !== 'visible') return el;
            el = el.parentElement;
        }
        return null;
    }

    // Reset the per-open snapshots so the next open re-fetches fresh counts/last-index.
    hide(): void {
        this.stats = null;
        this.modelStatus = null;
        this.modelDeleteConfirm = false;
        this.resetConfirm = false;
    }

    private async loadData(): Promise<void> {
        this.loading = true;
        try {
            const [stats, modelStatus] = await Promise.all([
                this.plugin.getIndexStats(),
                this.plugin.getModelStatus(),
            ]);
            this.stats = stats;
            this.modelStatus = modelStatus;
        } finally {
            this.loading = false;
        }
        this.rerender();
    }

    private get s() { return this.plugin.settings; }
    private save = () => this.plugin.saveSettings();

    // ---- Index ---------------------------------------------------------------------
    private statusState(): 'none' | 'ok' | 'indexing' | 'error' {
        if (this.plugin.isIndexing || this.reindexPhase === 'running') return 'indexing';
        if (this.plugin.indexHealthState === 'degraded') return 'error';
        if (this.plugin.indexHealthState === 'recovering') return 'indexing';
        if (this.stats && this.stats.files === 0) return 'none';
        return 'ok'; // NOTE: 'stale' (vault edited since last index) is intentionally
                     // not derived — it needs an expensive delta scan, and the file
                     // watcher catches edits up automatically. See the plan's degradations.
    }

    private renderIndex(containerEl: HTMLElement): void {
        new Setting(containerEl).setName('Index').setHeading();

        this.renderStatusCard(containerEl);

        if (this.plugin.indexHealthState === 'degraded') {
            const warn = containerEl.createDiv({ cls: 'seek-inline-warn' });
            warn.setText('Index degraded — search still works but ranking may be off. A full reindex is recommended.');
        }

        // The reindex button + live progress bar stay outside the disclosure: it's the
        // primary action and must be visible regardless of the advanced toggle.
        this.renderReindexRow(containerEl);

        // Advanced disclosure — what to index (Bases / excluded folders) and where the
        // index lives are set-once knobs, so tuck them away like Relevance's advanced
        // section. Mirrors renderRelevance's disclosure, with its own open-state flag.
        const disc = containerEl.createDiv({ cls: 'seek-disclosure' });
        disc.createSpan({ cls: 'seek-disclosure-chev', text: this.indexAdvancedOpen ? '▾' : '▸' });
        disc.createSpan({ text: 'Advanced settings' });
        disc.onclick = () => { this.indexAdvancedOpen = !this.indexAdvancedOpen; this.rerender(); };

        if (this.indexAdvancedOpen) this.renderIndexAdvanced(containerEl);
    }

    private renderIndexAdvanced(containerEl: HTMLElement): void {
        const adv = containerEl.createDiv({ cls: 'seek-adv' });

        new Setting(adv)
            .setName('Index Base files')
            .setDesc('Include your Obsidian Bases (.base files) in the search index, so a Base shows up by its name and filters. Takes effect on the next full reindex.')
            .addToggle(t => t.setValue(this.s.indexBases).onChange(async v => { this.s.indexBases = v; await this.save(); }));

        new Setting(adv)
            .setName('Honor excluded folders')
            .setDesc("Skip files in Obsidian's Settings → Files & Links → Excluded files (e.g. Archive). Takes effect on the next full reindex.")
            .addToggle(t => t.setValue(this.s.honorIgnoredFolders).onChange(async v => { this.s.honorIgnoredFolders = v; await this.save(); }));

        // Built as DOM (intro line · two bullets · footer) rather than a setDesc() string,
        // which renders flat with no line breaks — the two location options read far more
        // clearly as a short list.
        const indexLoc = new Setting(adv).setName('Index location');
        indexLoc.descEl.createDiv({ text: 'This is where the synced index folder lives.' });
        const locList = indexLoc.descEl.createEl('ul', { cls: 'seek-desc-list' });
        const locHidden = locList.createEl('li');
        locHidden.createEl('strong', { text: 'Hidden (default): ' });
        // Literal '.obsidian', NOT vault.configDir: the sidecar index is pinned to
        // the default config folder so every device resolves the SAME synced path
        // (see main.ts sidecarConfigDir). Showing vault.configDir would misreport
        // the index location to a renamed-config user, whose index still lives here.
        locHidden.createSpan({ text: `inside the hidden .obsidian config folder.` });
        const locRoot = locList.createEl('li');
        locRoot.createEl('strong', { text: 'Vault root: ' });
        locRoot.createSpan({ text: 'a visible "Seek Index" folder will appear in your vault. Choose this only if you use Obsidian Sync with a mobile or tablet override config folder.' });
        indexLoc.descEl.createDiv({ text: 'Takes effect after reloading Seek.' });
        indexLoc.addDropdown(dd => dd
            .addOption('config', `Hidden (.obsidian, recommended)`)
            .addOption('visible', 'Vault root (Seek Index/)')
            .setValue(this.s.sidecarIndexLocation)
            .onChange(async v => {
                this.s.sidecarIndexLocation = v as SidecarIndexLocation;
                await this.save();
                new Notice('Seek: index location changed — reload Seek (or restart Obsidian) for it to take effect.', 8000);
            }));
    }

    private renderStatusCard(containerEl: HTMLElement): void {
        const card = containerEl.createDiv({ cls: 'seek-status-card' });

        const STATE: Record<string, { tone: string; label: string }> = {
            none: { tone: 'mid', label: 'No index' },
            ok: { tone: 'good', label: 'Up to date' },
            indexing: { tone: 'accent', label: 'Indexing…' },
            error: { tone: 'bad', label: 'Index error' },
        };
        const st = STATE[this.statusState()];

        const health = card.createDiv({ cls: 'seek-status-health' });
        health.createSpan({ cls: `seek-dot seek-dot-${st.tone}` });
        health.createSpan({ cls: 'seek-status-label', text: st.label });

        card.createDiv({ cls: 'seek-status-sep' });

        const metric = (value: string, label: string) => {
            const m = card.createDiv({ cls: 'seek-status-metric' });
            m.createDiv({ cls: 'seek-status-value', text: value });
            m.createDiv({ cls: 'seek-status-mlabel', text: label });
        };
        const n = (x: number) => x.toLocaleString();
        if (this.stats) {
            metric(n(this.stats.files), 'files');
            metric(n(this.stats.chunks), 'chunks');
            // Storage figures intentionally omitted from this card: index size isn't shown
            // in settings anymore (the seek:indexsize CLI still reports it for diagnostics),
            // and the model's on-disk size now lives in the Model & performance section.
            const last = card.createDiv({ cls: 'seek-status-metric seek-status-last' });
            if (this.stats.lastFullAt) {
                // Real full reindex: stamp + duration from the same run.
                const dur = this.stats.lastFullDurationMs != null
                    ? ` · ${(this.stats.lastFullDurationMs / 1000).toFixed(1)}s` : '';
                last.createDiv({ cls: 'seek-status-mlabel', text: 'last full index' });
                last.createDiv({ cls: 'seek-status-value seek-status-stamp', text: `${fmtStamp(this.stats.lastFullAt)}${dur}` });
                // A catch-up has run since the full reindex → show it faintly so the full
                // stamp is never confused with an incremental update.
                if (this.stats.lastUpdatedAt && this.stats.lastUpdatedAt > this.stats.lastFullAt) {
                    last.createDiv({ cls: 'seek-status-updated', text: `updated ${fmtStamp(this.stats.lastUpdatedAt)}` });
                }
            } else if (this.stats.lastUpdatedAt) {
                // No full reindex survives in the log — show the last update, no duration
                // (a catch-up's duration isn't meaningful to surface on its own).
                last.createDiv({ cls: 'seek-status-mlabel', text: 'last updated' });
                last.createDiv({ cls: 'seek-status-value seek-status-stamp', text: fmtStamp(this.stats.lastUpdatedAt) });
            } else {
                last.createDiv({ cls: 'seek-status-mlabel', text: 'last full index' });
                last.createDiv({ cls: 'seek-status-value seek-status-stamp', text: 'never' });
            }
        } else {
            metric('…', 'loading');
        }
    }

    private renderReindexRow(containerEl: HTMLElement): void {
        if (this.reindexPhase === 'running') {
            const row = containerEl.createDiv({ cls: 'seek-progress-row' });
            const head = row.createDiv({ cls: 'seek-progress-head' });
            head.createDiv({ cls: 'setting-item-name', text: 'Reindexing…' });
            this.progressLabelEl = head.createDiv({ cls: 'seek-progress-count' });
            const bar = row.createDiv({ cls: 'seek-progress-track' });
            this.progressFillEl = bar.createDiv({ cls: 'seek-progress-fill' });
            this.paintProgress();
            return;
        }
        this.progressFillEl = null;
        this.progressLabelEl = null;

        if (this.reindexPhase === 'confirm') {
            new Setting(containerEl)
                .setName('Full reindex')
                .setDesc("This deletes the current index and re-indexes every note. This may take a few minutes, depending on the size of your vault. Search keeps working on the old index until it's complete.")
                .addButton(b => b.setButtonText('Cancel').onClick(() => { this.reindexPhase = 'idle'; this.rerender(); }))
                .addButton(b => b.setButtonText('Delete & reindex').setWarning().onClick(() => this.startReindex()));
            this.renderReindexNote(containerEl);
            return;
        }

        // No index yet (fresh install / post-reset): there's nothing to delete, so the
        // destructive "Delete & reindex" double-confirm is just friction. Offer a single
        // non-warning click that builds the index straight away.
        if (this.statusState() === 'none') {
            new Setting(containerEl)
                .setName('Build index')
                .setDesc('Index every note so Seek can search your vault. This may take a few minutes on a large vault.')
                .addButton(b => b.setButtonText('Build index').setCta().onClick(() => this.startReindex()));
            this.renderReindexNote(containerEl);
            return;
        }

        new Setting(containerEl)
            .setName('Full reindex')
            .setDesc('Rebuild the whole search index from scratch.')
            .addButton(b => b.setButtonText('Reindex…').setWarning().onClick(() => { this.reindexPhase = 'confirm'; this.rerender(); }));
        this.renderReindexNote(containerEl);
    }

    // Building/reindexing re-embeds every note — the heaviest thing Seek does. On a
    // mobile phone that pass can be interrupted by the OS under memory pressure, so we
    // surface a standing note: run it on a computer and the phone syncs the finished
    // index embed-free. Shown on every device (it documents the limitation); tablets
    // are NOT called out — they handle the build fine.
    private renderReindexNote(containerEl: HTMLElement): void {
        containerEl.createDiv({
            cls: 'seek-hint',
            text: 'Building the index re-embeds every note and isn’t recommended on a mobile phone. Run it on a computer, and your phone will sync the finished index automatically.',
        });
    }

    private startReindex(): void {
        const md = this.app.vault.getMarkdownFiles().length;
        const bases = this.s.indexBases ? this.app.vault.getFiles().filter(f => f.extension === 'base').length : 0;
        this.reindexTotal = md + bases;
        this.reindexDone = 0;
        this.reindexPhase = 'running';
        this.rerender();

        void this.plugin.runFullReindex({
            skipConfirm: true,
            onProgress: (msg) => {
                const m = msg.match(/Indexed\s+([\d,]+)\s+files/i);
                if (m) this.reindexDone = parseInt(m[1].replace(/,/g, ''), 10);
                this.paintProgress();
            },
        }).then(() => {
            // Back to idle with a refreshed status card — that IS the "done" feedback.
            this.reindexPhase = 'idle';
            this.stats = null;
            this.rerender();
            void this.loadData();
        }).catch(() => {
            this.reindexPhase = 'idle';
            this.rerender();
        });
    }

    private paintProgress(): void {
        const pct = this.reindexTotal > 0
            ? Math.min(100, Math.round((this.reindexDone / this.reindexTotal) * 100))
            : 0;
        if (this.progressFillEl) this.progressFillEl.style.width = `${pct}%`;
        if (this.progressLabelEl) {
            this.progressLabelEl.setText(`${this.reindexDone.toLocaleString()} / ${this.reindexTotal.toLocaleString()} notes · ${pct}%`);
        }
    }

    // ---- Relevance -----------------------------------------------------------------
    private renderRelevance(containerEl: HTMLElement): void {
        new Setting(containerEl).setName('Relevance').setHeading();

        const intro = containerEl.createDiv({ cls: 'seek-rel-intro' });
        intro.createDiv({ cls: 'seek-rel-title', text: 'How Seek ranks' });
        intro.createDiv({
            cls: 'setting-item-description',
            text: 'Seek blends conceptual meaning with exact keywords, and can optionally apply bonuses for recency and exact title matching. It is strongly recommended to leave Seek in the default Balanced mode.',
        });

        this.renderPipeline(containerEl);

        containerEl.createDiv({ cls: 'seek-hint', text: 'Relevance changes apply to your next search.' });

        // Advanced disclosure
        const disc = containerEl.createDiv({ cls: 'seek-disclosure' });
        disc.createSpan({ cls: 'seek-disclosure-chev', text: this.advancedOpen ? '▾' : '▸' });
        disc.createSpan({ text: 'Advanced relevance settings' });
        disc.onclick = () => { this.advancedOpen = !this.advancedOpen; this.rerender(); };

        if (this.advancedOpen) this.renderAdvanced(containerEl);
    }

    private renderPipeline(containerEl: HTMLElement): void {
        const strategy = strategyOf(this.s.denseWeight);
        const recStage = recencyStageOf(this.s.recencyEpsilon);
        const titleStage = titleStageOf(this.s.navTitleBoost);

        const pipe = containerEl.createDiv({ cls: 'seek-pipe' });
        const box = (text: string, cls = '') => pipe.createDiv({ cls: `seek-pipe-box ${cls}`.trim(), text });
        const arrow = () => pipe.createSpan({ cls: 'seek-pipe-arrow', text: '→' });

        box('Notes');
        arrow();
        // In Balanced both branches are neutral; only Keyword-focused elevates Keyword.
        const branch = pipe.createDiv({ cls: 'seek-pipe-branch' });
        branch.createDiv({ cls: 'seek-pipe-box seek-pipe-dense', text: 'Conceptual meaning' });
        branch.createDiv({ cls: `seek-pipe-box seek-pipe-kw${strategy === 'keyword' ? ' is-elevated' : ''}`, text: 'Keyword' });
        arrow();
        box('Fusion', 'seek-pipe-fuse');
        arrow();
        // Bonuses with recency·title sub-labels: dim+strike when Off, bold when on, bolder at High.
        const bonus = pipe.createDiv({ cls: 'seek-pipe-box seek-pipe-bonus' });
        bonus.createSpan({ text: 'Bonuses' });
        const subs = bonus.createDiv({ cls: 'seek-pipe-subs' });
        const subLabel = (text: string, stage: Stage) => {
            const cls = stage === 'Off' ? 'is-off' : stage === 'High' ? 'is-high' : 'is-on';
            subs.createSpan({ cls: `seek-pipe-sub ${cls}`, text });
        };
        subLabel('recency', recStage);
        subs.createSpan({ text: ' · ' });
        subLabel('title', titleStage);
        arrow();
        box('Results');
    }

    private renderAdvanced(containerEl: HTMLElement): void {
        const adv = containerEl.createDiv({ cls: 'seek-adv' });

        // Search strategy (denseWeight)
        const strat = new Setting(adv)
            .setName('Search strategy')
            .setDesc('Balanced suits nearly everyone. Only switch to Keyword focused if you have exact terms which Balanced mode does not rank appropriately.');
        this.addSegmented(strat, ['Balanced', 'Keyword focused'],
            strategyOf(this.s.denseWeight) === 'keyword' ? 'Keyword focused' : 'Balanced',
            (pick) => {
                void (async () => {
                    this.s.denseWeight = STRATEGY_VALUE[pick === 'Keyword focused' ? 'keyword' : 'balanced'];
                    await this.save();
                    this.rerender(); // re-weight the pipeline diagram
                })();
            });

        // Fuzzy matching
        new Setting(adv)
            .setName('Fuzzy matching')
            .setDesc('Keywords will still return results even with small spelling mistakes.')
            .addToggle(t => t.setValue(this.s.fuzzyEnabled).onChange(async v => { this.s.fuzzyEnabled = v; await this.save(); }));

        // Recency bonus (3-stage) + date picker
        const recStage = recencyStageOf(this.s.recencyEpsilon);
        const rec = new Setting(adv)
            .setName('Recency bonus')
            .setDesc('Gives a score bonus to newer notes based on a selected date type property, or file modified date. This is recommended if you have episodic notes which occur regularly around the same topics, like meetings or classes.');
        this.addSegmented(rec, ['Off', 'Default', 'High'], recStage, (pick) => {
            void (async () => {
                const v = RECENCY_VALUE[pick as Stage];
                this.s.recencyEpsilon = v.eps;
                this.s.recencyHalfLifeDays = v.hl;
                await this.save();
                this.rerender(); // re-bold the pipeline "recency" sub-label + enable/disable the date picker
            })();
        });
        // Date-property picker, dimmed/disabled until a stage other than Off is chosen.
        const dateProps = enumerateDatePropertyNames(this.app);
        if (this.s.recencyKey === 'created' && !dateProps.includes(this.s.createdProp)) dateProps.unshift(this.s.createdProp);
        rec.addDropdown(dd => {
            for (const p of dateProps) dd.addOption(`prop:${p}`, `Property: ${p}`);
            dd.addOption('mtime', 'File modified time');
            dd.setValue(this.s.recencyKey === 'modified' ? 'mtime' : `prop:${this.s.createdProp}`);
            dd.onChange(async v => {
                if (v === 'mtime') { this.s.recencyKey = 'modified'; }
                else { this.s.recencyKey = 'created'; this.s.createdProp = v.slice('prop:'.length); }
                await this.save();
            });
            dd.selectEl.disabled = recStage === 'Off';
            if (recStage === 'Off') dd.selectEl.addClass('seek-dimmed');
        });

        // Title bonus (3-stage)
        const title = new Setting(adv)
            .setName('Title bonus')
            .setDesc("Gives a score bonus to notes which have matching terms in their title. This can help a note that represents an entity or topic outrank pages that merely mention it.");
        this.addSegmented(title, ['Off', 'Default', 'High'], titleStageOf(this.s.navTitleBoost), (pick) => {
            void (async () => {
                this.s.navTitleBoost = TITLE_VALUE[pick as Stage];
                await this.save();
                this.rerender(); // re-bold the pipeline "title" sub-label
            })();
        });
    }

    // ---- Display -------------------------------------------------------------------
    private renderDisplay(containerEl: HTMLElement): void {
        new Setting(containerEl).setName('Display').setHeading();

        // Per-result score line. Match strength only exists on a calibrated corpus
        // (≥ MATCH_STRENGTH_MIN_NOTES notes AND a completed full-index pass that
        // produced dense background stats), so the toggle is shown always — for
        // discoverability — but disabled with a reason until scoring is possible.
        const noteCount = this.app.vault.getMarkdownFiles().length;
        const scoringReady = noteCount >= MATCH_STRENGTH_MIN_NOTES && (this.stats?.calibrated ?? false);
        const scoresDesc = scoringReady
            ? 'Shows each result’s Matching %, recency, and title boost on the result row.'
            : noteCount < MATCH_STRENGTH_MIN_NOTES
                ? `Shows each result’s Matching %, recency, and title boost. Needs at least ${MATCH_STRENGTH_MIN_NOTES} indexed notes before scores can be calibrated.`
                : 'Shows each result’s Matching %, recency, and title boost. Available once the index finishes its first full calibration pass.';
        new Setting(containerEl)
            .setName('Display scores')
            .setDesc(scoresDesc)
            .addToggle(t => t
                .setValue(this.s.showScores)
                .setDisabled(!scoringReady)
                .onChange(async v => { this.s.showScores = v; await this.save(); }));

        new Setting(containerEl)
            .setName('Keyboard hints bar')
            .setDesc('Displays a keyboard hint bar under results in the results modal.')
            .addToggle(t => t.setValue(this.s.showHotkeyHints).onChange(async v => { this.s.showHotkeyHints = v; await this.save(); }));

        // Alt-open destination. A plain Enter/click always replaces the current
        // tab (the quick-switcher contract; not configurable) — this picks where
        // the ⌘/Ctrl alt-open fans out instead. Mobile coerces to a tab at the
        // use-site (search-modal.ts altOpenTarget), so no per-platform gating here.
        const ALT_OPEN_VALUE: Record<string, AltOpenLocation> = {
            'New tab': 'tab', 'New split': 'split', 'New window': 'window',
        };
        const ALT_OPEN_LABEL: Record<AltOpenLocation, string> = {
            tab: 'New tab', split: 'New split', window: 'New window',
        };
        const altOpen = new Setting(containerEl)
            .setName('Open results with Cmd/Ctrl in')
            .setDesc('Where a result opens when you hold Cmd/Ctrl while clicking or pressing Enter. A plain click or Enter always opens in the current tab.');
        this.addSegmented(altOpen, Object.keys(ALT_OPEN_VALUE), ALT_OPEN_LABEL[this.s.altOpenLocation], (pick) => {
            void (async () => {
                this.s.altOpenLocation = ALT_OPEN_VALUE[pick];
                await this.save();
                this.rerender(); // repaint the segmented control's active pill
            })();
        });
    }

    // ---- Model & performance -------------------------------------------------------
    private renderModel(containerEl: HTMLElement): void {
        new Setting(containerEl).setName('Model & performance').setHeading();

        // Compute backend — PER-DEVICE (localStorage), never synced. Auto / Force CPU /
        // Force WebGPU map to the platform.ts override values auto / wasm / webgpu.
        const computeLabel: Record<BackendChoice, string> = { auto: 'Auto', wasm: 'Force CPU', webgpu: 'Force WebGPU' };
        const labelToChoice: Record<string, BackendChoice> = { Auto: 'auto', 'Force CPU': 'wasm', 'Force WebGPU': 'webgpu' };
        const compute = new Setting(containerEl)
            .setName('Compute')
            .setDesc('How the embedding model runs on this device (this option is not synced to other devices). Auto uses WebGPU when available and falls back to CPU. Changing this setting is not recommended.');
        this.addSegmented(compute, ['Auto', 'Force CPU', 'Force WebGPU'], computeLabel[getBackendOverride()], (pick) => {
            setBackendOverride(labelToChoice[pick]);
            this.rerender(); // forcing WebGPU clears a prior sticky demote; reflect it
        });

        if (isWebgpuDemoted()) {
            new Setting(containerEl)
                .setName('WebGPU disabled after a crash on this device')
                .setDesc('Seek detected this device was killed by the OS during a WebGPU reindex and fell back to CPU. Reset to let Auto try WebGPU again (e.g. after an OS update).')
                .addButton(b => b.setButtonText('Reset & retry WebGPU').setWarning().onClick(() => {
                    clearWebgpuDemoted();
                    new Notice('Seek: WebGPU re-enabled on this device. Takes effect on the next model load.', 6000);
                    this.rerender();
                }));
        }

        this.renderModelStatus(containerEl);
    }

    private renderModelStatus(containerEl: HTMLElement): void {
        const ms = this.modelStatus;
        const row = new Setting(containerEl).setName('Embedding model');

        if (this.modelDownloading) {
            const desc = row.descEl;
            desc.createSpan({ cls: 'seek-spinner' });
            desc.createSpan({ text: ' Downloading… (≈100 MB — keep Obsidian open)' });
            return;
        }

        if (this.modelDeleting) {
            const desc = row.descEl;
            desc.createSpan({ cls: 'seek-spinner' });
            desc.createSpan({ text: ' Deleting model…' });
            return;
        }

        const downloaded = ms?.downloaded ?? false;
        const desc = row.descEl;
        const dot = desc.createSpan({ cls: `seek-dot seek-dot-${downloaded ? 'good' : 'mid'}` });
        dot.setCssStyles({ marginRight: '6px' });
        if (downloaded) {
            // Model on-disk size (Cache API bytes), relocated here from the index status card.
            // null on platforms that don't expose the usageDetails split (e.g. iOS) — omit it
            // there rather than render a bare dash.
            const modelMB = this.stats?.modelMB;
            const sizeText = modelMB != null ? ` · ${Math.round(modelMB)} MB` : '';
            desc.createSpan({ text: `Downloaded${sizeText} · Stored on disk.` });
            // Model id + dim on its own line below the status (a block div, not an inline
            // span) so the long repo name no longer wraps mid-sentence after "permanently.".
            if (ms) desc.createDiv({ cls: 'seek-faint seek-model-id', text: `${ms.name} · ${ms.dim}-dim` });
            // The only downloaded-state action is destructive — it frees the ~100 MB and
            // forces a re-download on the next search — so it's a red, two-step Delete
            // (Delete → Cancel / Delete model), never a single click. To re-acquire the
            // model afterward, the resting "Not downloaded" state offers Download now.
            if (this.modelDeleteConfirm) {
                row.addButton(b => b.setButtonText('Cancel').onClick(() => { this.modelDeleteConfirm = false; this.rerender(); }));
                row.addButton(b => b.setButtonText('Delete model').setWarning().onClick(() => this.deleteModel()));
            } else {
                row.addButton(b => b.setButtonText('Delete').setWarning().onClick(() => { this.modelDeleteConfirm = true; this.rerender(); }));
            }
        } else {
            desc.createSpan({ text: 'Not downloaded · the first search fetches ≈100 MB.' });
            row.addButton(b => b.setButtonText('Download now').setCta().onClick(() => this.downloadModel()));
        }
    }

    private downloadModel(): void {
        this.modelDownloading = true;
        this.rerender();
        void this.plugin.prewarmModel().finally(() => {
            this.modelDownloading = false;
            this.modelStatus = null;
            this.rerender();
            void this.loadData(); // refresh downloaded status
        });
    }

    private deleteModel(): void {
        this.modelDeleteConfirm = false;
        this.modelDeleting = true;
        this.rerender();
        void this.plugin.deleteModel().then(() => {
            new Notice('Seek: embedding model deleted. The next search re-downloads it (≈100 MB).', 6000);
        }).catch((e) => {
            new Notice(`Seek: model delete failed — ${e instanceof Error ? e.message : String(e)}`, 8000);
        }).finally(() => {
            this.modelDeleting = false;
            this.modelStatus = null;
            this.rerender();
            void this.loadData(); // refresh status → now "Not downloaded"
        });
    }

    // ---- Diagnostics + Reset --------------------------------------------------------
    private renderReset(containerEl: HTMLElement): void {
        // Diagnostics first, under its own heading, and rendered BEFORE the
        // reset-confirm early-return below so the report button is always visible
        // (it replaces the removed "Generate logging report" command). openLoggingReport
        // renders the per-device NDJSON logs into seek-report.md and opens it.
        new Setting(containerEl).setName('Diagnostics').setHeading();
        new Setting(containerEl)
            .setName('Logging report')
            .setDesc('Write a diagnostic report (seek-report.md) of indexing, searches, model loads, and any errors — generate and share it when reporting an issue. Review before sharing: it includes your recent queries and matching note paths (but not note contents).')
            .addButton(b => b.setButtonText('Generate logging report').onClick(() => void this.plugin.openLoggingReport()));

        new Setting(containerEl).setName('Reset').setHeading();
        if (this.resetConfirm) {
            new Setting(containerEl)
                .setName('Reset to defaults')
                .setDesc('Restores the default configuration for all Seek settings. Your index will not be rebuilt.')
                .addButton(b => b.setButtonText('Cancel').onClick(() => { this.resetConfirm = false; this.rerender(); }))
                .addButton(b => b.setButtonText('Reset settings').setWarning().onClick(async () => {
                    // Restore every persisted (synced) setting. Compute is per-device
                    // localStorage, not part of data.json, so it is deliberately untouched.
                    Object.assign(this.s, DEFAULT_SETTINGS);
                    await this.save();
                    this.resetConfirm = false;
                    new Notice('Seek: settings restored to defaults. Your index was not rebuilt.', 6000);
                    this.rerender();
                }));
            return;
        }
        new Setting(containerEl)
            .setName('Reset to defaults')
            .setDesc('Restore all Seek settings to their original values. Your index will not be rebuilt.')
            .addButton(b => b.setButtonText('Reset…').onClick(() => { this.resetConfirm = true; this.rerender(); }));
    }
    private resetConfirm = false;

    // ---- About ---------------------------------------------------------------------
    private renderAbout(containerEl: HTMLElement): void {
        const about = containerEl.createDiv({ cls: 'seek-about' });
        const left = about.createDiv({ cls: 'seek-about-left' });
        left.createSpan({ cls: 'seek-about-name', text: 'Seek' });
        left.createSpan({ cls: 'seek-about-ver', text: `v${this.plugin.manifest.version}` });
        left.createSpan({ cls: 'seek-about-by', text: 'by Ryan Manor' });

        const links = about.createDiv({ cls: 'seek-about-links' });
        // Lucide-named icon button (GitHub, Docs).
        const link = (href: string, icon: string, label: string) => {
            const a = links.createEl('a', { cls: 'seek-about-ic', href, attr: { 'aria-label': label, title: label } });
            setIcon(a, icon);
        };
        link(DOCS_URL, 'book-open', 'Seek Documentation');
        link(GITHUB_URL, 'github', 'Repository on GitHub');
        // X uses an inline-SVG button (no Lucide brand icon — see X_LOGO_PATH).
        this.brandLink(links, X_URL, X_LOGO_PATH, '0 0 24 24', 'On X');
    }

    // An icon-button link whose glyph is an inline SVG path rather than a Lucide
    // icon — for brand logos Obsidian's bundled Lucide doesn't (any longer) ship.
    // Built via createElementNS (SVG namespace) so it's a real <svg> the
    // `.seek-about-ic svg` rule sizes, with fill=currentColor so it tints like the
    // setIcon glyphs beside it.
    private brandLink(parent: HTMLElement, href: string, pathD: string, viewBox: string, label: string): void {
        const a = parent.createEl('a', { cls: 'seek-about-ic', href, attr: { 'aria-label': label, title: label } });
        const ns = 'http://www.w3.org/2000/svg';
        const svg = activeDocument.createElementNS(ns, 'svg');
        svg.setAttribute('viewBox', viewBox);
        svg.setAttribute('fill', 'currentColor');
        const path = activeDocument.createElementNS(ns, 'path');
        path.setAttribute('d', pathD);
        svg.appendChild(path);
        a.appendChild(svg);
    }

    // ---- shared: segmented (pill) control ------------------------------------------
    private addSegmented(setting: Setting, opts: string[], selected: string, onPick: (o: string) => void): void {
        const seg = setting.controlEl.createDiv({ cls: 'seek-seg' });
        for (const o of opts) {
            const b = seg.createEl('button', { cls: 'seek-seg-opt', text: o });
            if (o === selected) b.addClass('is-active');
            b.onclick = () => onPick(o);
        }
    }
}
