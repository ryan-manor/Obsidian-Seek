// Seek plugin entry. Three commands per v0 scope:
//   1. seek-search        — open the search modal
//   2. seek-reindex       — full reindex (nuke + rebuild)
//   3. seek-generate-log  — write seek-report.md from seek-log.ndjson
//
// Plus a headless CLI query handler (registerCliHandler), exposed only when the
// obsidian-cli bridge is present: `obsidian seek:search query="..."`. Unlike the
// palette command (which opens a modal and returns void), the CLI handler returns
// a string the bridge writes to stdout — readable text by default, JSON with
// `format=json`. See the registration in onload.
//
// Intentional non-features for v0:
//   - No settings tab (zero admin console, per task brief)
//   - No incremental reindex
//   - No sync sidecar protocol
//   - No model-cache management
//   - No MCP wrapper

import { Notice, Plugin, TFile } from 'obsidian';
import type { App } from 'obsidian';
import { LocalEmbedder, LOCAL_MODEL, LEGACY_ENGLISH_MODEL_ID, EMBEDDING_DIM } from './embedder';
import { activeModelSpec, resolveOverrideSpec, evictStaleModelCaches, deleteModelCaches, probeModelDownloaded } from './model-registry';
import { pluginIdentity, identityMatches, identityFromMeta } from './identity';
import { sweepOrphanTmpFiles } from './sidecar';
import type { SeekSettings, SidecarIndexLocation, IndexCompleteEntry, ModelDeliveryEntry } from './types';
import { DEFAULT_SETTINGS, migrateSettings } from './types';
import { IndexStore } from './index-store';
import { SeekLogger } from './logger';
import { Forensics } from './forensics';
import { SearchOrchestrator, driftRecoveryDecision } from './search';
import { SeekSearchModal } from './search-modal';
import { SeekSettingTab } from './settings-tab';
import { collectPlatformInfo, isMobilePlatform, resolveDevice, recordActiveBackend, maybeDemoteOnCrash, getBackendOverride, setBackendOverride, isWebgpuDemoted, clearWebgpuDemoted, type BackendChoice } from './platform';
import { CompositorPacer } from './pacer';
import { shouldUnloadEmbedder, type UnloadGateState } from './embedder-lifecycle';
import { drainCatchUp, CATCHUP_MAX_FILES_PER_BURST, CATCHUP_BURST_BUDGET_MS } from './catchup';
import type { LongTaskEntry, MemoryPressureEntry, StorageSnapshotEntry, EvictionSuspectedEntry, AppLocalFetchEntry } from './types';
import { TRANSFORMERS_VERSION } from './iframe-runner';

// Long-task threshold. PerformanceObserver fires for any task ≥50 ms by spec,
// but at that floor we'd flood the log. 250 ms is the rough threshold above
// which the user perceives a stutter and is also the design-doc latency
// budget for search.
const LONG_TASK_THRESHOLD_MS = 250;

// Extensions Seek indexes: markdown notes always, plus .base files (Obsidian
// Bases — YAML view definitions, indexed via a synthetic doc; see
// base-extractor.ts) when `indexBases` is on. The orchestrator's collection set
// (indexableFiles) must agree with this, so both gate on the same setting.
function isIndexableFile(f: TFile, indexBases: boolean): boolean {
    if (f.extension === 'md') return true;
    return indexBases && f.extension === 'base';
}


// Incremental-indexing debounces. Edits wait out a 5-min idle window after the
// user leaves a note (so flipping back to keep writing never triggers a flush
// mid-thought); deletes/moves apply on a short window (they're model-free and a
// dead search result is jarring). See wireIncrementalIndexing.
const IDLE_FLUSH_MS = 5 * 60 * 1000;
const STRUCT_FLUSH_MS = 1500;

// Mobile-only embedder unload (see maybeUnloadEmbedder). After this long with no
// model use in a quiescent state, tear down the iframe to release its ~240 MB
// model + ratcheted WASM heap; the next search pays one cold reload. The race
// with a pending edit flush is settled by the unload PREDICATE (an armed flush
// timer counts as `pending`), not by the relative size of these constants.
// Checked on a coarse interval — a minute of slack on a 3-minute idle is fine.
const IDLE_UNLOAD_MS = 3 * 60 * 1000;
const UNLOAD_CHECK_MS = 60 * 1000;

// A delta larger than this isn't an edit — it's a bulk import (paste, vault sync,
// git checkout). flushDirty treats it as a mini-reindex: progress is surfaced, a
// live query preempts the embed, and a cold desktop model is deferred rather than
// force-loaded for a background paste. At or below it, the single-note force path
// is unchanged (a 1-2 file embed is too short to be worth the extra machinery).
const BULK_DELTA_THRESHOLD = 50;


// DTOs returned to the settings tab by getIndexStats() / getModelStatus(). Defined
// here (not types.ts) because they describe this plugin's read API; settings-tab.ts
// imports them as types.
export interface IndexStats {
    files: number;
    chunks: number;
    // storageMB = navigator.storage.estimate().usage — the WHOLE origin (index + model
    // + every other plugin's storage); kept as a fallback. indexMB / modelMB split it via
    // the non-standard usageDetails ({ indexedDB, caches }) that Electron/Chromium exposes:
    // indexedDB ≈ the vector index, caches ≈ the transformers-cache model bytes. Both are
    // origin-shared (other plugins' IDB/Cache count too), but Seek dominates each, so the
    // split is a far more honest read than the conflated total. null when unavailable.
    storageMB: number | null;
    indexMB: number | null;
    modelMB: number | null;
    // "Last full index" is sourced from the most recent index-complete log entry whose
    // mode is 'full' — timestamp + duration come from the SAME run, so they always agree
    // (the prior code mixed meta's any-mode timestamp with the log's any-mode duration,
    // which let a 2-file catch-up's 2.0s masquerade as a full reindex). Null if no full
    // run survives in the (rotatable) log.
    lastFullAt: string | null;
    lastFullDurationMs: number | null;
    // Last index of ANY mode (incremental catch-ups included), from store meta. Shown as
    // a secondary "updated …" line when it post-dates the full reindex.
    lastUpdatedAt: string | null;
    // True once the index has a calibrated dense background (bgMean/bgStd present,
    // σ>0) — the precondition for a non-null match strength. Gates the "Display
    // scores" toggle: scores can't be shown on an uncalibrated corpus.
    calibrated: boolean;
}
export interface ModelStatus {
    downloaded: boolean;
    persisted: boolean | null;
    name: string;
    dim: number;
}

export default class SeekPlugin extends Plugin {
    private embedder = new LocalEmbedder();
    private store = new IndexStore();
    private logger!: SeekLogger;
    private orchestrator!: SearchOrchestrator;
    // Mutated in place on settings change so the orchestrator (which holds the
    // same reference) always reads current values. See types.ts SeekSettings.
    settings: SeekSettings = { ...DEFAULT_SETTINGS };

    // Promise that resolves once the model is loaded. Lazy-init: we don't
    // want to spend 250 MB of RAM on plugin startup if the user never opens
    // the search modal. The first search/reindex invocation triggers it.
    private modelLoadPromise: Promise<void> | null = null;

    // Async observer handles + global handlers we register on load and
    // explicitly tear down on unload. Without cleanup these leak into the
    // next plugin reload and we end up with duplicate logging on every hot
    // reload during development.
    private longTaskObserver: PerformanceObserver | null = null;
    private currentTaskContext: LongTaskEntry['context'] = 'idle';
    private onError: ((e: ErrorEvent) => void) | null = null;
    private onUnhandledRejection: ((e: PromiseRejectionEvent) => void) | null = null;
    private onVisibilityChange: (() => void) | null = null;
    private onPageHide: (() => void) | null = null;
    // Crash forensics (see forensics.ts). Created in onload once the vault
    // scope (appId) is known; null only during the first lines of onload.
    private forensics: Forensics | null = null;

    // Incremental indexing state (see wireIncrementalIndexing). The queues hold
    // paths touched THIS session; the durable dirty signal is always on-disk
    // mtime vs. the stored FileRecord, so the startup sweep (reconcileOnLoad) is
    // the real catch-up and these are just a live-session optimization.
    private dirtyQueue = new Set<string>();
    private deletedQueue = new Set<string>();
    private lastActiveFile: TFile | null = null;
    private idleTimer: number | null = null;   // 5-min debounce for edits
    private structTimer: number | null = null;  // short debounce for deletes/moves
    private lastModelUseAt = 0;                  // epoch ms of the last ensureModelLoaded; drives the idle-unload timer (mobile)
    private flushing = false;                    // flushDirty re-entrancy guard
    private catchUpPending = false;              // cold-mobile deferred an embed
    private catchUpRunning = false;              // runCatchUp re-entrancy guard
    // Drift auto-recovery (sibling of catch-up). The orchestrator detects persistent
    // frame/BM25 row-space drift and fires onPersistentDrift; we run a bounded,
    // embed-free recovery ladder (warm → sidecar hydrate → verify). Re-escalation is
    // suppressed per index generation (see driftRecoveryDecision); indexHealth surfaces
    // a terminal 'degraded' on the settings page when the ladder can't re-couple.
    private driftRecoveryPending = false;        // a persistent-drift escalation is queued for a safe window
    private driftRecoveryRunning = false;        // runDriftRecovery re-entrancy guard
    private lastDriftRecoveryGen = -1;           // coord generation we last escalated for; -1 = never
    private indexHealth: 'healthy' | 'recovering' | 'degraded' = 'healthy';
    get indexHealthState(): 'healthy' | 'recovering' | 'degraded' { return this.indexHealth; }
    // Fires the version-identity mismatch log + the mobile "reindex on desktop" notice
    // once per stale spell (the gate re-checks every 5 min); cleared when identity heals
    // so a future version ship reports again. See enforceIndexIdentity.
    private identityHealNotified = false;
    // True while a heal (peer rebuild / desktop reindex) is running, so the 5-min poll
    // firing mid-reindex doesn't stack a second one (reindexAllInner would queue it).
    private identityHealInFlight = false;
    // The referential-integrity orphan sweep runs once per session (Phase 3), on the
    // first healthy 5-min poll — not at boot (it taxes app-open) and not repeatedly.
    // `Done` latches only on a COMPLETED pass; `Running` guards re-entrancy so an
    // overlapping poll tick can't double-run it.
    private orphanSweepDone = false;
    private orphanSweepRunning = false;
    // Sidecar self-compaction runs once per session on the same poll (after the sweep):
    // it reclaims this device's own superseded/orphaned sidecar records — the one GC an
    // off-grid device gets, since version-bump reindex + hydrate-from-peer need sync.
    // `Done` latches on any DEFINITIVE outcome (an incomplete re-chunk leaves it open to
    // retry); `Running` guards re-entrancy.
    private sidecarCompactDone = false;
    private sidecarCompactRunning = false;

    // True while a reindex / incremental embed is running. currentTaskContext is
    // private; this is the read-only surface the settings Index status card reads
    // (on open and while polling a live reindex) to show its "Indexing…" state.
    get isIndexing(): boolean { return this.currentTaskContext === 'indexing'; }
    private searchActiveTimestamp: number | null = null;   // null = no live query session; else the ms timestamp of the last activity ping (modal open / keystroke) — pauses the catch-up drain so embedding never competes with the user's search
    private static readonly SEARCH_ACTIVE_MAX_AGE_MS = 60_000;
    private queryInFlight = false;               // modal-reported: a query embed/search is actually running right now (onQueryInFlight). Distinct from the keystroke-timed searchActive — it falls only when the query COMPLETES, so indexing waits for the query, not just for typing to pause.
    // Self-healing read of searchActive. A modal torn down without onClose (teardown
    // exception, dev hot-reload) would otherwise latch the flag true forever and
    // permanently starve catch-up (runCatchUp/drainCatchUp early-return on it, so
    // deferred cold-mobile embeds never reconcile until restart). Treat it inactive
    // past a max age; an active session re-stamps the timestamp on every keystroke.
    private get searchActive(): boolean {
        if (this.searchActiveTimestamp === null) return false;
        if (Date.now() - this.searchActiveTimestamp > SeekPlugin.SEARCH_ACTIVE_MAX_AGE_MS) {
            this.searchActiveTimestamp = null;
            return false;
        }
        return true;
    }

    // The single gate every indexing path honours: hold embeds while the user's
    // query must win the shared (iOS) thread — an active typing session (keystroke-
    // timed) OR a query embed that is actually in flight (lifecycle-timed). The
    // second term is what makes indexing wait for the query to COMPLETE rather than
    // resuming 1.5 s after the last keystroke while a slow mobile embed still runs.
    private get indexingBlocked(): boolean {
        return this.searchActive || this.queryInFlight;
    }

    async onload() {
        this.logger = new SeekLogger(this.app);
        // Sweep any pre-existing root-level seek-log/init/captures files into the
        // hidden LOG_DIR next to the index, THEN tail-truncate this device's log if it
        // has outgrown MAX_LOG_BYTES (append-only logs have no natural ceiling), THEN
        // prune any abandoned other-device / legacy logs (pruneOrphanLogs). All three are
        // fire-and-forget: new writes already target LOG_DIR, the report reads both
        // locations during the migration window, and rotation/pruning only shrink the tail
        // or drop dead files — so this blocks nothing on the load path. The steps chain in
        // order so each operates on the files at their final LOG_DIR location.
        void this.logger.migrateRootFiles()
            .then(() => this.logger.rotateIfOversize())
            .then(() => this.logger.pruneOrphanLogs())
            .catch(e => this.logger.appendError('logger-onload-maintenance', e).catch(() => {}));
        // Load persisted settings (merge over defaults so new keys appear).
        // Mutate the existing object in place — the orchestrator holds this
        // same reference.
        const raw = ((await this.loadData()) ?? {}) as Partial<SeekSettings>;
        // Rev 4 = sidecar path pinned to the literal '.obsidian'. Capture the
        // pre-migration rev HERE — the actual sidecar FILE move runs further below
        // (it needs the old active-override + new literal paths), and migrateSettings
        // is about to overwrite raw.settingsRev, so the flag must be read first.
        const migrateSidecarPath = (raw.settingsRev ?? 1) < 4;
        // Key-level schema migrations (rev 2 denseWeight rescale, rev 5 defaults
        // ratification — see migrateSettings in types.ts, where they're unit-tested).
        // Mutates `raw` and stamps settingsRev; runs BEFORE the Object.assign so the
        // migrated values win over the persisted ones rather than being overridden.
        migrateSettings(raw);
        Object.assign(this.settings, DEFAULT_SETTINGS, raw);
        await this.saveData(this.settings);
        // Scope the index DB per vault. IndexedDB is shared across every vault
        // window (one Electron origin), so an unscoped name means vault A's
        // reindex destroys vault B's index (see index-store.ts LEGACY_DB_NAME).
        // appId is Obsidian's stable per-vault id — the same key it uses for
        // its own vault-scoped localStorage; not in the public typings, hence
        // the cast. Vault name fallback keeps a (rename-fragile) scope if a
        // future Obsidian drops appId.
        const appId = (this.app as unknown as { appId?: string }).appId;
        await this.store.open(appId ?? this.app.vault.getName());

        // Crash forensics: synchronous localStorage breadcrumbs (vault-scoped
        // like the IDB name — localStorage is origin-shared across vaults on
        // mobile). bootInspect classifies an unclean previous session and we
        // promote it into the log; this is the ONLY way a jetsam kill becomes
        // visible, since async NDJSON appends die with the process.
        this.forensics = new Forensics(appId ?? this.app.vault.getName(), this.logger.deviceId, this.logger.sessionId);
        const crash = this.forensics.bootInspect();
        if (crash) {
            // Forensics: always persist the classified crash to the per-device
            // log. This is the diagnostic surface (read via the log report) — it
            // is silent by design, not a toast.
            await this.logger.append(crash);
            // Tripwire WITH a side effect: maybeDemoteOnCrash performs the sticky,
            // per-device WebGPU→WASM demotion (a localStorage write) when a mobile
            // device was killed mid-reindex in the foreground while WebGPU was the
            // active backend, so the next reindex doesn't walk into the same OS
            // kill. It MUST run on every boot — only its actionable "you're now on
            // WASM" result is surfaced. The generic "previous session ended
            // uncleanly" notice was removed: on iOS onunload never fires on an OS
            // suspend-kill, so any routine 12h+-backgrounded reopen classifies as a
            // benign `evicted-while-hidden` exit — that fired the toast on
            // essentially every mobile open for normal app lifecycle.
            if (maybeDemoteOnCrash(crash.verdict)) {
                new Notice('Seek: last session was killed mid-reindex on WebGPU — this device is now on WASM. Re-enable WebGPU in settings to retry.', 8000);
            }
        }

        // WebGPU loss diagnostics: the iframe pushes device-created / device-
        // lost / uncaptured-error events from its requestDevice hook. device-
        // lost is the only JS-visible discriminator between a GPU-process
        // death (page survives, this fires) and a WebContent kill (silence) —
        // see [[Seek Mobile WebGPU Investigation]]. Forensics beat FIRST and
        // synchronously: if the page is about to die too, the localStorage
        // write must win the race; the NDJSON append is the best-effort twin.
        this.embedder.onIframeEvent = (event) => {
            const kind = typeof event.kind === 'string' ? event.kind : 'webgpu-event';
            const detail: Record<string, number | string | boolean | null> = {};
            for (const [k, v] of Object.entries(event)) {
                if (k !== 'kind') detail[k] = v;
            }
            this.forensics?.beat(kind, detail);
            void this.logger.append({
                type: 'webgpu-event',
                timestamp: new Date().toISOString(),
                kind,
                ...detail,
            });
            // WebGPU device death (GPU process died, page survived). The in-flight
            // embedBatch would otherwise hang on the dead device until the per-RPC
            // timeout; reject it now (recoverable, NOT 'DISPOSED') so the embed
            // catch's recycle+retry rebuilds a fresh device promptly. Skipped while
            // unloading — the iframe is being torn down; don't poke it.
            if (kind === 'webgpu-device-lost' && !this.unloading) {
                this.embedder.failInflight('webgpu device lost');
            }
        };

        // One-time DB v3→v4 transition: if the user already has fp32 vectors
        // from a prior install, pack their sign-bit binary siblings in the
        // background. Steady-state this is a no-op (the count guard inside
        // returns 0 immediately). Awaited because the orchestrator's first
        // search assumes the binary index is loadable — but the actual work
        // is ~5 ms for an empty store and ~100 ms for a 5k-chunk vault.
        try {
            const packed = await this.store.backfillBinaryIfMissing();
        } catch (e) {
            // Backfill failure isn't fatal: the user can rebuild via "Full
            // reindex". Log it and continue plugin init.
            await this.logger.appendError('binary-backfill', e);
        }

        // Sidecar index dir. CRITICAL: resolved from a LITERAL config-folder
        // name, not this.manifest.dir — manifest.dir resolves against the
        // device's active Override Config Folder (vault.configDir), which is
        // per-device and never synced, so a split-config setup made producer
        // and consumer read different paths → silent zero results. See the
        // Sidecar Integration Plan §config-folder CRITICAL.
        const sidecarIndexDir = this.resolveSidecarIndexDir();
        // The pre-rev-4 path (active-override-relative). Used only to migrate an
        // existing index off it into the literal path on upgrade.
        const legacySidecarDir = this.manifest.dir ? `${this.manifest.dir}/index` : null;
        this.orchestrator = new SearchOrchestrator(this.app, this.store, this.embedder, this.logger, this.settings, this.forensics, sidecarIndexDir);
        // The orchestrator is pull-based; this is its one injected outbound edge — it
        // fires when persistent frame/BM25 drift survives the cooldown, and we drive the
        // embed-free recovery ladder from the plugin (which owns scheduling + gating).
        this.orchestrator.setPersistentDriftHandler(() => this.onPersistentDrift());
        this.addSettingTab(new SeekSettingTab(this.app, this));

        // Incremental indexing: live vault-event triggers + the startup catch-up
        // sweep. Wired here, after the orchestrator exists.
        this.wireIncrementalIndexing();
        // Sidecar restore THEN the mtime-diff sweep, sequenced (not raced): on a
        // cold/evicted device hydrate must finish populating the store before
        // reconcileOnLoad computes its delta, or computeDelta would read the empty
        // store, mark the whole vault dirty, and re-embed everything the sidecar
        // already holds. All off the load path (fire-and-forget IIFE) so plugin
        // load never blocks. sweepOrphanTmpFiles first cleans any crashed atomic
        // write. Each step is gated/no-op when the sidecar is disabled.
        void (async () => {
            let identityHandled = false;
            try {
                // One-time rev-3→4 migration: move an index written under the
                // old active-override path into the literal '.obsidian' path,
                // BEFORE hydrate reads the new location (else hydrate finds it
                // empty and re-embeds what we already have). Only in 'config'
                // mode (the literal path is the target) and only when the
                // active override actually diverges from it.
                if (this.settings.sidecarEnabled && migrateSidecarPath
                    && this.settings.sidecarIndexLocation === 'config'
                    && legacySidecarDir && legacySidecarDir !== sidecarIndexDir) {
                    await this.migrateSidecarFiles(legacySidecarDir, sidecarIndexDir);
                }
                if (this.settings.sidecarEnabled && sidecarIndexDir) await sweepOrphanTmpFiles(this.app.vault.adapter, sidecarIndexDir);
                // Version-identity gate BEFORE any catch-up: if the local index was
                // built under a different chunker/model/revision/dim than this build,
                // it is stale — hydrate from a matching peer (embed-free) / desktop-
                // reindex / mobile-wait, and skip the normal catch-up this boot (never
                // re-chunk or re-embed onto a stale-id index — the mobile jetsam path).
                identityHandled = await this.enforceIndexIdentity();
                if (!identityHandled) {
                    // Gated, not unconditional: the persisted dir-signature lets a
                    // crash-relaunch with an unchanged vault skip the whole-vault
                    // re-chunk (the iOS 1fps loop), while an empty/evicted store
                    // still forces the recovery sweep.
                    await this.orchestrator.reconcileSidecarIfChanged();
                }
            } catch (e) {
                await this.logger.appendError('sidecar-hydrate-onload', e).catch(() => {});
            }
            // Steer split-config Obsidian Sync users to the visible folder: the
            // one case the hidden literal path can't reach (a renamed config
            // folder never receives '.obsidian/' over Sync). Fire-and-forget,
            // after hydrate so a working setup is never interrupted.
            await this.maybeSteerSidecarLocation().catch(() => {});
            // Startup mtime-diff sweep — the backstop for everything changed while
            // Seek wasn't watching (external sync, edits with the plugin disabled,
            // in-session deletes lost on a crash). On desktop it loads the model +
            // re-embeds; on cold mobile it applies deletes/moves now and defers
            // edits to the first search. Skipped when the identity gate took over:
            // a rebuild already hydrated, a reindex is terminal, and a mobile wait
            // must not re-chunk a stale index.
            if (!identityHandled) await this.reconcileOnLoad();
            // The clean-launch BM25/frame warm that used to fire here ('startup') has
            // moved into ensureModelLoaded ('model-load') so it overlaps the model
            // load instead of taxing app-open. reconcileOnLoad still warms ('delta')
            // when its diff finds changes; warmCaches's `warming` guard dedups the two.
        })();

        // Periodic sidecar reconcile: remote arrivals don't fire vault events for
        // .obsidian/ dotfiles, so poll for another device's freshly-synced index
        // every 5 min. reconcileSidecarIfChanged gates on a cheap (persisted) dir
        // signature, so a warm index with no new sidecar files skips the
        // whole-vault re-chunk entirely. Cleared automatically on unload.
        this.registerInterval(window.setInterval(() => void this.periodicReconcile(), 5 * 60 * 1000));

        // Mobile: reset the WASM heap during genuine idle. Once a minute, if the
        // model hasn't been used for IDLE_UNLOAD_MS, tear it down (when quiescent);
        // the next search reloads. Desktop is excluded — the heap ratchet is a
        // mobile/iOS problem, and a cold reload on every post-idle desktop search
        // would be a latency regression for no benefit. Cleared on unload.
        if (isMobilePlatform()) {
            this.registerInterval(window.setInterval(() => {
                if (Date.now() - this.lastModelUseAt >= IDLE_UNLOAD_MS) this.maybeUnloadEmbedder('idle');
            }, UNLOAD_CHECK_MS));
        }

        // Wire global observers + handlers BEFORE we do anything else, so
        // any error in the init path itself gets logged.
        this.wireGlobalErrorHandlers();
        this.wireLongTaskObserver();
        this.wireMemoryPressureHandlers();

        // Init the iframe runtime OFF the blocking onload path. The search command
        // (registered below) no longer waits on it: an early search coalesces onto
        // this same memoized init inside embedder.load() (embedder.ts init()/load()),
        // so there is no init race. The init log entry, the diagnostics that want a
        // live iframe (platform GPU probe, app-local probe), and the failure Notice
        // all ride this continuation instead of blocking app-open. The diagnostics
        // are DEFERRED, not removed — still dogfooding.
        void this.embedder.init()
            .then(async (initEntry) => {
                await this.logger.writeInit(initEntry);
                await this.logger.append(initEntry);

                const platformInfo = await collectPlatformInfo();
                await this.logger.append(platformInfo);

                // Boot-time storage snapshot. Week-over-week drops in storageUsedMB
                // are the canary for Cache API / IDB eviction even when no cold-start
                // outlier fires. Cheap — one navigator.storage.estimate().
                await this.emitStorageSnapshot('boot');

                // `app://local/...` capability probe. Runs once per session after
                // iframe init (before any model load) so the result is available
                // regardless of whether the user ever searches.
                if (initEntry.iframeReady) {
                    this.runAppLocalProbe().catch(e => this.logger.appendError('app-local-probe', e));
                }

                // No success toast — readiness is signalled by the modal glyph
                // brightening. Only a genuine failure interrupts: a dead iframe means
                // no search will work, which is worth one toast.
                if (!initEntry.iframeReady || initEntry.error) {
                    new Notice(
                        `Seek: search engine failed to start${initEntry.error ? ` — ${initEntry.error.slice(0, 80)}` : ''}. See Settings → Seek → Generate logging report.`,
                        8000,
                    );
                }
            })
            .catch(e => this.logger.appendError('embedder-init-onload', e).catch(() => {}));

        // Auto-request persistent storage so iOS / Safari won't evict our
        // ~250 MB model cache + index under memory pressure. Best-effort.
        if (navigator.storage?.persist) {
            navigator.storage.persist().catch(e => {
                console.warn('[seek] navigator.storage.persist() failed:', e);
            });
        }

        // ---- Commands ----

        this.addCommand({
            id: 'seek-search',
            name: 'Search',
            callback: () => this.openSearchModal(),
        });

        // ---- obsidian://seek deep-link --------------------------------
        // `obsidian://seek?query=<urlencoded>[&mode=open][&vault=<name>]`.
        // registerObsidianProtocolHandler is a core Plugin API (present on
        // EVERY platform, incl. mobile — unlike the CLI bridge below), so this
        // is the mobile-safe deep-link surface. Two modes:
        //   search (default) — open the modal pre-filled + running the query
        //                      (human-in-the-loop; the modal owns cold-start).
        //   open            — headless: load the model, run the query, open the
        //                      top hit's note ("jump to my note about X").
        // `mode` (not `action`) is the discriminator because ObsidianProtocolData
        // RESERVES `action` for the protocol host ('seek') — a `&action=` param
        // would collide with it. The scheme is deliberately READ-ONLY: any web
        // page can fire an obsidian:// URL, so no write/reindex/config action is
        // ever exposed here — those stay command/CLI-only where intent is explicit.
        this.registerObsidianProtocolHandler('seek', (params) => {
            // Obsidian percent-DECODES params, so `%23`→`#` etc. arrive clean.
            // The producer must encode `#` (URL fragment delimiter AND Seek's
            // own `#tag` sigil) — the modal's "copy link" action does this.
            const query = typeof params.query === 'string' ? params.query : '';
            if (params.mode === 'open') void this.openTopResult(query);
            else this.openSearchModal(query);
        });


        // Reindex and diagnostics are intentionally NOT palette commands: a full
        // reindex nukes and re-embeds the whole vault (too destructive for a fuzzy
        // palette match), so it lives in Settings → Seek → Index behind a confirm;
        // the logging report is a Settings button (openLoggingReport). Sidecar
        // reconcile/rebuild are automatic. Search is the only command Seek adds.

        // ---- Headless CLI query handler --------------------------------
        // `obsidian seek:search query="..." [limit=N]`.
        //
        // registerCliHandler is provided by the obsidian-cli companion, not the
        // core Obsidian API typings — hence the cast and the runtime guard. On
        // mobile (no CLI bridge) the method is absent and we simply skip it.
        //
        // The contract that makes results "read out" at all: the handler RETURNS
        // a string, which the bridge pipes to stdout. addCommand callbacks return
        // void (their job is to open the modal), so they can never feed the CLI —
        // that is why `seek:seek-search` was unreachable. Output defaults to a
        // readable text list (rank/score/path/excerpt); `format=json` emits the
        // machine shape (path/title/score/excerpt), matching the predecessor
        // plugin so the same parsing works.
        const registerCliHandler = (this as unknown as {
            registerCliHandler?: (
                id: string,
                description: string,
                params: Record<string, { value?: string; description: string; required: boolean }>,
                handler: (args: Record<string, string | boolean | undefined>) => Promise<string>,
            ) => void;
        }).registerCliHandler;

        if (typeof registerCliHandler === 'function') {
            registerCliHandler.call(
                this,
                'seek:search',
                'Seek on-device semantic search (hybrid BM25 + dense embeddings + recency)',
                {
                    query: { value: '<text>', description: 'Search query (supports inline filters: #tag, tag:, path:, [k:v], dates)', required: true },
                    limit: { value: '<n>', description: 'Max results (default: 10)', required: false },
                    format: { value: 'text|json', description: 'Output format (default: text — readable list; json for programmatic use)', required: false },
                    recencyWeight: { value: '<ε>', description: 'Override recency weight ε for THIS query only (additive; default 0.02). Not persisted — for scrobbling recency configs.', required: false },
                    recencyHalflife: { value: '<days>', description: 'Override recency half-life in days for THIS query only (default 180). Not persisted.', required: false },
                },
                // Flag params arrive as the string "true"/"false", not booleans
                // (the obsidian-cli/Templater convention), so widen the type and
                // compare against both.
                async (args: Record<string, string | boolean | undefined>): Promise<string> => {
                    const query = typeof args.query === 'string' ? args.query : '';
                    const asJson = args.format === 'json';
                    // Error sink that honors the active format: JSON callers get a
                    // parseable {error}, humans get a one-liner.
                    const fail = (msg: string): string =>
                        asJson ? JSON.stringify({ error: msg, results: [] }) : `Seek error: ${msg}`;

                    if (!query) return fail('query is required');
                    if (!this.orchestrator) return fail('Seek not initialized — plugin still loading');

                    const parsedLimit = typeof args.limit === 'string' ? parseInt(args.limit, 10) : NaN;
                    const topK = Number.isFinite(parsedLimit) && parsedLimit > 0 ? parsedLimit : 10;

                    // Query-time recency overrides (scrobbling). rankConfig reads
                    // this.settings fresh per search, so mutating the live ref here
                    // takes effect for this call; restored in finally and never
                    // saved (no saveSettings) → not persisted. Safe against the
                    // shared modal/CLI settings ref because a headless call has no
                    // concurrent modal search to race it. Same validation as the UI.
                    const origRecencyEps = this.settings.recencyEpsilon;
                    const origRecencyHl = this.settings.recencyHalfLifeDays;
                    const ovEps = typeof args.recencyWeight === 'string' ? parseFloat(args.recencyWeight) : NaN;
                    const ovHl = typeof args.recencyHalflife === 'string' ? parseFloat(args.recencyHalflife) : NaN;
                    if (Number.isFinite(ovEps) && ovEps >= 0) this.settings.recencyEpsilon = ovEps;
                    if (Number.isFinite(ovHl) && ovHl > 0) this.settings.recencyHalfLifeDays = ovHl;

                    try {
                        // No modal here to overlap the cold-start, so block on the
                        // model load (3–10 s first call) before querying — otherwise
                        // the orchestrator embeds against an unloaded model.
                        await this.ensureModelLoaded();
                        const { results } = await this.orchestrator.search(query, topK);

                        // ---- format=json: programmatic / piped callers ---------
                        if (asJson) {
                            const mapped = results.map(r => {
                                const base: Record<string, unknown> = {
                                    path: r.note_path,
                                    // displayTitle carries the "(part N)" marker for
                                    // split chunks (absent otherwise); title itself is
                                    // now the clean, embedded/indexed form.
                                    title: r.displayTitle ?? r.title,
                                    score: r.score,
                                    excerpt: r.snippet ?? '',
                                };
                                return base;
                            });
                            return JSON.stringify({ results: mapped, query, count: mapped.length });
                        }

                        // ---- default: human-readable text ----------------------
                        // Minified single-line JSON wraps into an unreadable wall
                        // in a terminal. The readable form emits real newlines —
                        // one record per block: "rank  score  path", with the
                        // excerpt indented to align beneath the path.
                        if (results.length === 0) return `Seek · "${query}" · no results`;

                        const INDENT = ' '.repeat(11); // width of "NN  0.000  "
                        const lines: string[] = [
                            `Seek · "${query}" · ${results.length} result${results.length === 1 ? '' : 's'}`,
                            '',
                        ];
                        results.forEach((r, i) => {
                            lines.push(`${String(i + 1).padStart(2, ' ')}  ${r.score.toFixed(3)}  ${r.note_path}`);
                            const excerpt = (r.snippet ?? '').replace(/\s+/g, ' ').trim();
                            if (excerpt) lines.push(`${INDENT}${excerpt.length > 160 ? excerpt.slice(0, 159) + '…' : excerpt}`);
                            lines.push('');
                        });
                        return lines.join('\n').replace(/\n+$/, '');
                    } catch (err) {
                        return fail(err instanceof Error ? err.message : String(err));
                    } finally {
                        // Restore the live settings ref regardless of outcome so an
                        // override never leaks into the next query or the modal.
                        this.settings.recencyEpsilon = origRecencyEps;
                        this.settings.recencyHalfLifeDays = origRecencyHl;
                    }
                },
            );
        }
    }

    // Set true the instant onunload starts so async callbacks (the WebGPU
    // device-lost handler) can't resurrect a teardown-in-progress iframe.
    private unloading = false;

    onunload() {
        this.unloading = true;
        // First thing, synchronously: a session whose record isn't closed at
        // next boot reads as a crash. Reload/disable/quit all pass through here.
        this.forensics?.markCleanEnd();
        this.embedder.teardown();
        this.orchestrator?.dispose();
        this.store.close();
        if (this.longTaskObserver) {
            try { this.longTaskObserver.disconnect(); } catch { /* swallow */ }
            this.longTaskObserver = null;
        }
        if (this.onError) window.removeEventListener('error', this.onError);
        if (this.onUnhandledRejection) window.removeEventListener('unhandledrejection', this.onUnhandledRejection);
        if (this.onVisibilityChange) document.removeEventListener('visibilitychange', this.onVisibilityChange);
        if (this.onPageHide) window.removeEventListener('pagehide', this.onPageHide);
        if (this.idleTimer != null) window.clearTimeout(this.idleTimer);
        if (this.structTimer != null) window.clearTimeout(this.structTimer);
        // vault/workspace events and the window 'blur' DOM event are registered
        // via registerEvent/registerDomEvent, so Obsidian tears them down for us.
    }

    // ── Sidecar index location ──────────────────────────────────────────────
    // The DEFAULT config-folder name. We pin to this LITERAL string rather than
    // vault.configDir (the per-device active override) so every device resolves
    // the SAME sidecar path — the fix for the config-folder CRITICAL.
    private static readonly DEFAULT_CONFIG_DIR = '.obsidian';
    private static readonly SIDECAR_CONFIG_DIR = '.obsidian/plugins/seek/index';
    private static readonly SIDECAR_VISIBLE_DIR = 'Seek Index';

    // Hidden literal path by default; the vault-root visible folder only when a
    // split-config Obsidian Sync user opts in (see maybeSteerSidecarLocation).
    private resolveSidecarIndexDir(): string {
        return this.settings.sidecarIndexLocation === 'visible'
            ? SeekPlugin.SIDECAR_VISIBLE_DIR
            : SeekPlugin.SIDECAR_CONFIG_DIR;
    }

    // One-time rev-3→4 move of an index written under the old active-override
    // path into the literal path. Uses rename (a move) per file; idempotent and
    // non-fatal — a failed move leaves the source for the next reindex to
    // repopulate, never aborts hydrate.
    private async migrateSidecarFiles(from: string, to: string): Promise<void> {
        const adapter = this.app.vault.adapter;
        const ls = await adapter.list(from).catch(() => null);
        if (!ls || ls.files.length === 0) return; // nothing written under the old path
        if (!(await adapter.exists(to).catch(() => false))) await adapter.mkdir(to).catch(() => {});
        let moved = 0;
        for (const path of ls.files) {
            const dest = `${to}/${path.slice(path.lastIndexOf('/') + 1)}`;
            // Never clobber the new location (a prior partial migration, or this
            // device already wrote there) — the literal path is authoritative.
            if (await adapter.exists(dest).catch(() => false)) continue;
            try {
                await adapter.rename(path, dest);
                moved++;
            } catch (e) {
                await this.logger.appendError('sidecar-migrate-file', e).catch(() => {});
            }
        }
    }

    // Steer the lone unreachable case to the visible folder: Obsidian Sync + a
    // RENAMED config folder. The hidden literal '.obsidian/' is never delivered
    // to a device booting a renamed config over Sync, so embeddings can't cross.
    // iCloud/Syncthing/Dropbox carry the literal path regardless, so they never
    // trip this. One-time per device (the condition is per-device) via
    // localStorage — a uniform-config device never sees it.
    private async maybeSteerSidecarLocation(): Promise<void> {
        if (!this.settings.sidecarEnabled) return;
        if (this.settings.sidecarIndexLocation !== 'config') return; // already opted in / steered
        const configDir = this.app.vault.configDir;
        if (configDir === SeekPlugin.DEFAULT_CONFIG_DIR) return;     // uniform config — literal path syncs fine
        // Obsidian Sync writes sync.json into the active config folder; its
        // presence is the in-use signal. iCloud/Syncthing have no such file.
        const onObsidianSync = await this.app.vault.adapter.exists(`${configDir}/sync.json`).catch(() => false);
        if (!onObsidianSync) return;
        const flagKey = 'seek-sidecar-steer-shown';
        if (window.localStorage.getItem(flagKey)) return;
        window.localStorage.setItem(flagKey, '1');
        new Notice(
            `Seek: Obsidian Sync won't deliver the hidden index to a renamed config folder ("${configDir}"). ` +
            "Set Seek's index location to “Visible folder” (Settings → Seek → Sync) to sync embeddings across your devices.",
            12000,
        );
    }

    // Public pre-warm entry for the settings "Download now" button: fetches +
    // caches the ~100 MB model bytes (and loads it) so a user can pre-warm over
    // Wi-Fi before going offline, instead of search stalling on the first fetch.
    // Idempotent — delegates to the memoized loader below; a no-op if already loaded.
    async prewarmModel(): Promise<void> { await this.ensureModelLoaded(); }

    // Proactively release the embedder when it's provably safe — the only way to
    // shrink the iframe's WASM heap (WebAssembly.Memory never contracts within a
    // page, so a long mobile session ratchets toward the OOM that kills the next
    // model load). The next search/embed reloads transparently: ensureModelLoaded
    // sees `loaded` false and `modelLoadPromise` null and rebuilds (loadImpl calls
    // init() first). Nulling modelLoadPromise is load-bearing — without it,
    // ensureModelLoaded would hand back a resolved promise for a model that's gone
    // (it checks modelLoadPromise before loaded). Mirrors the manual
    // seek-unload-model command, gated by the pure shouldUnloadEmbedder predicate.
    private maybeUnloadEmbedder(reason: 'idle' | 'background'): void {
        const gate: UnloadGateState = {
            loaded: this.embedder.loaded,
            busy: this.currentTaskContext !== 'idle',
            queryActive: this.indexingBlocked,
            running: this.flushing || this.catchUpRunning || this.driftRecoveryRunning,
            pending: this.catchUpPending || this.driftRecoveryPending
                || this.dirtyQueue.size > 0 || this.deletedQueue.size > 0
                || this.idleTimer != null || this.structTimer != null,
        };
        if (!shouldUnloadEmbedder(reason, gate)) return;
        this.embedder.teardown();
        this.modelLoadPromise = null;
        const heap = (performance as unknown as { memory?: { usedJSHeapSize: number } }).memory;
        void this.logger.append({
            type: 'model-lifecycle',
            timestamp: new Date().toISOString(),
            event: 'unload',
            reason,
            heapMB: heap ? heap.usedJSHeapSize / 1e6 : null,
        }).catch(() => {});
        // Synchronous breadcrumb too — the background unload races a possible
        // jetsam kill, and the async log line above can be lost to it.
        this.forensics?.beat(reason === 'idle' ? 'model-unload-idle' : 'model-unload-bg');
    }

    private ensureModelLoaded(): Promise<void> {
        // Every search/embed path funnels through here, so this is the single
        // chokepoint that marks the model "in use" — the idle-unload timer counts
        // from it. Stamped even on the already-loaded fast path so a burst of
        // searches keeps the model resident.
        this.lastModelUseAt = Date.now();
        if (this.embedder.loaded) return Promise.resolve();
        if (this.modelLoadPromise) return this.modelLoadPromise;

        // Device selection. Desktop: WebGPU first with WASM fallback (design
        // doc: WebGPU ~4× faster on iPhone single-query historically, and
        // competitive on desktop). iOS: skip WebGPU entirely — see the iOS
        // skip rationale at the load() call below.
        this.modelLoadPromise = (async () => {
            try {
                // Warm the frame + BM25 caches in the model-load shadow. warmCaches
                // reads the store (not the model), so it overlaps the cold model load
                // and the index is hot by the time the first query runs — recovering
                // the clean-launch COLD_START miss the old onload 'startup' warm
                // covered. Self-guarded: `warming` dedups the reconcile 'delta' warm,
                // and the mobile `!loaded` guard keeps cold mobile a no-op here (the
                // model isn't loaded yet at this point).
                void this.orchestrator.warmCaches('model-load');
                // q4: equal quality (bake-off NDCG@10 Δ=0.0005) and, on the
                // pinned v4 runtime, also the fastest + lightest config (20.6
                // ch/s, ~103 MB heap vs q8 13.2 ch/s, ~380 MB). q4's viability
                // depends on the v4 ORT-Web floor — see iframe-runner.ts
                // TRANSFORMERS_VERSION / dtype-ladder comments for the full
                // history (the q4-vs-q8 gap on old ORT-Web was real, not a
                // fusion confound). 'auto' = WebGPU (q4-only ladder) with WASM
                // q4 fallback. q4f16 excluded — Gemma's LayerNorm shader fails
                // to compile on Dawn with half-precision activations (ORT #26732).
                // Phase-5: when LOCAL_MODEL.enabled, load the trimmed
                // model from the vault via an app://local resource URL
                // instead of the HF CDN. getResourcePath appends a
                // `?<ts>` cache-buster (seen in the app-local probe) —
                // strip it so transformers.js path-joins
                // `<base>/onnx/model_*.onnx` cleanly.
                const ra = this.app.vault.adapter as unknown as { getResourcePath?: (p: string) => string };
                const localBase = LOCAL_MODEL.enabled && ra.getResourcePath
                    ? ra.getResourcePath(LOCAL_MODEL.vaultRelPath).split('?')[0].replace(/\/+$/, '')
                    : undefined;
                if (LOCAL_MODEL.enabled && !localBase) {
                    await this.logger.appendError('local-model', new Error('getResourcePath unavailable; using remote MODEL_ID'));
                }
                // Active model from the registry (debug override wins, else the
                // shipped default). spec.repo is the CDN-streamed load base on the
                // remote path; spec.key is the index drift-identity. A dim mismatch
                // means EMBEDDING_DIM + DB_VERSION weren't bumped with the registry
                // — loud-log it (mis-indexing into the wrong-dim store is worse).
                const spec = activeModelSpec(this.settings);
                if (spec.dim !== EMBEDDING_DIM) {
                    await this.logger.appendError('model-registry', new Error(
                        `active model ${spec.key} dim ${spec.dim} != EMBEDDING_DIM ${EMBEDDING_DIM}; bump EMBEDDING_DIM + DB_VERSION together`));
                }
                // Device selection is per-DEVICE (platform.ts resolveDevice),
                // not a synced setting — see the NOTE in types.ts for why a
                // data.json toggle would leak across iCloud-synced devices.
                // 'auto' = WebGPU-then-WASM ladder; 'wasm' = skip WebGPU. The
                // allowlist: desktop + iPad get 'auto', iPhone + all Android get
                // 'wasm' (jetsam / immature WebView WebGPU), a per-device manual
                // override wins, and a sticky tripwire demotes any mobile device
                // that was OS-killed mid-WebGPU-reindex.
                //
                // On the 'auto' path the iframe's WebGPU attempt rewrites
                // wasmPaths to a webgpuInit-capable glue (jspi/asyncify — see
                // overrideWebkitGlueForWebgpu in iframe-runner.ts), the fix that
                // let the iPad run granite on WKWebView WebGPU (2026-06-10).
                // tx.js 4.2.0 otherwise pins anything Safari-detected (WKWebView
                // included) to the plain wasm glue compiled WITHOUT webgpuInit,
                // so without that override the webgpu EP init throws by
                // construction. On failure the load falls back to WASM on a
                // fresh module instance (tx.js's webInitChain has no rejection
                // handler — one failed load poisons the instance) and logs
                // webgpuError. ⚠️ ORT #26827 hang risk still applies on iPhone,
                // which is why iPhone stays off the 'auto' path by default.
                const requestedDevice = resolveDevice();
                // Model selection: the LOCAL_MODEL dev override wins (it's a
                // base URL, not a hub id); otherwise MODEL_ID. Both ride dtype
                // 'q4' — tx.js resolves it to onnx/model_q4.onnx (the ml97
                // repo's model_q4.onnx IS the GBQ-int4 export).
                // Bracket the load with beats: a death inside (the iPhone
                // wasm-OOM class, or a WebGPU device-init kill) gets attributed
                // to 'model-load' instead of reading as idle.
                this.forensics?.beat('model-load-start', { device: requestedDevice, model: spec.repo });
                const entry = await this.embedder.load(requestedDevice, LOCAL_MODEL.enabled ? LOCAL_MODEL.dtype : spec.dtype, localBase ?? spec.repo, LOCAL_MODEL.enabled ? null : spec.revision);
                this.forensics?.beat('model-load-done', { device: entry.actualDevice, dtype: entry.dtype, coldStartMs: Math.round(entry.coldStartMs) });
                // Stamp the backend this load actually resolved to (WebGPU can
                // fall back to WASM). Read at next boot by maybeDemoteOnCrash to
                // decide whether an indexing-crash implicates WebGPU.
                recordActiveBackend(entry.actualDevice);
                await this.logger.append(entry);
                await this.warnOnModelIndexDrift();
                // Production model delivery (remote/Cache-API path only — the
                // LOCAL_MODEL dev path loads from the vault, nothing CDN-cached):
                // (1) request Cache-API persistence (best-effort; Safari grants on
                // engagement heuristics) and (2) evict a PREVIOUS model's cached
                // bytes after a switch. Eviction is parent-side because our iframe
                // is non-sandboxed (shares the cache partition); benign if the cache
                // is absent. `cacheSeen===0` in the log is the canary that the
                // parent can't see the iframe cache (→ move eviction to an RPC).
                if (!LOCAL_MODEL.enabled) {
                    let persisted: boolean | null = null;
                    try { persisted = await navigator.storage.persist(); }
                    catch { /* private mode / unsupported */ }
                    let evicted = { seen: 0, deleted: 0 };
                    try {
                        if (typeof caches !== 'undefined') evicted = await evictStaleModelCaches(caches, spec.repo);
                    } catch (e) {
                        await this.logger.appendError('model-evict', e);
                    }
                    await this.logger.append({
                        type: 'model-delivery',
                        timestamp: new Date().toISOString(),
                        key: spec.key,
                        repo: spec.repo,
                        revision: spec.revision,
                        persisted,
                        cacheSeen: evicted.seen,
                        cacheEvicted: evicted.deleted,
                    });
                }
                // Eviction canary. The same 5 s threshold the LoadEntry's
                // checks array uses, but emitted as its own structured event
                // so the report (and future alerting) can count suspected
                // evictions without parsing free-text checklists.
                if (entry.coldStartMs >= 5000) {
                    await this.emitEvictionSuspected(entry);
                }
                // Success is signalled ambiently by the search modal's glyph
                // brightening from faint → full (PillQueryField.setModelReady),
                // not a toast — the bare device/dtype/timing line was noise on
                // every cold open. Warnings still get a toast: a degraded load
                // is worth interrupting for, and the glyph can't convey "why".
                if (!entry.pass) {
                    console.warn(`[seek] model loaded with warnings on ${entry.actualDevice} — see the logging report (Settings → Seek).`);
                }
            } catch (e) {
                this.modelLoadPromise = null; // allow retry
                await this.logger.appendError('ensureModelLoaded', e);
                throw e;
            }
        })();
        return this.modelLoadPromise;
    }

    async saveSettings(): Promise<void> {
        await this.saveData(this.settings);
    }

    // Model ↔ index drift check, run after every successful model load. The
    // index meta carries the modelId that built it (stamped by reindexAll);
    // if the loaded model differs, dense scores would be cross-model garbage —
    // tell the user to full-reindex. Pre-2026-06-10 indexes have no stamp
    // (all english-r2 by construction), so an unstamped index always reads as
    // drift against ml97 and routes to a reindex. Once per session — the
    // condition can't self-heal without a reindex, so repeating the notice is
    // pure noise.
    private modelDriftWarned = false;
    // ── Boot version-identity cascade ─────────────────────────────────────────────
    // The active replacement for warn-only drift: when the local index's stored
    // identity (chunker/model/revision/dim) differs from this build's, the index is
    // provably stale, so heal it WITHOUT user commands. Order: hydrate from a matching
    // peer (embed-free, both platforms) → else desktop auto full-reindex / mobile wait
    // (mobile never bulk-embeds = no jetsam; the 5-min poll retries until a desktop
    // publishes a current sidecar). Returns true when it took over the index (the
    // caller then skips the normal catch-up). Runs off pluginIdentity() (compiled
    // constants), so it needs no model load and can gate a cold mobile boot.
    private async enforceIndexIdentity(): Promise<boolean> {
        // The cascade self-heals via cross-device sidecar hydration (+ desktop reindex
        // fallback), so it only applies with the sidecar on. Sidecar-off keeps the
        // legacy warn-only check (warnOnModelIndexDrift, fired at model-load time).
        if (!this.settings.sidecarEnabled) return false;
        // A debug model override (testing arbitrary repos) keys the loaded model off
        // the override while the sidecar/identity machinery keys off the shipped
        // MODEL_ID — running the cascade would loop. Defer to the warn-only check.
        if (resolveOverrideSpec(this.settings)) return false;
        // A heal is already running — don't stack a second when the 5-min poll fires
        // mid-reindex. Report "handled" so the caller still skips the normal catch-up.
        if (this.identityHealInFlight) return true;
        // A reindex / cold-build / delta is mutating the index right now: its meta is
        // mid-stamp and its identity not yet written, so an identity check would race
        // it and could "heal" a build that is about to stamp itself current. Report
        // NOT handled — by the next poll the writer has finished and stamped (Fix A/C).
        // (Primary guard is periodicReconcile.isIndexBusy; this also covers the onload
        // call, where nothing is writing yet, as cheap belt-and-suspenders.)
        if (this.orchestrator.isWriting()) return false;

        const meta = await this.store.getMeta().catch(() => null);
        if (!meta) return false; // store unreadable mid-teardown — the 5-min poll rechecks
        // An empty index isn't "stale" — the normal first-build / cold-hydrate path
        // owns it. Count, NOT lastIndexedAt: a hydrate-only device holds chunks with a
        // null lastIndexedAt.
        if ((await this.store.count()).chunks === 0) return false;

        if (identityMatches(identityFromMeta(meta), pluginIdentity())) {
            this.identityHealNotified = false; // healthy — re-arm reporting for a future ship
            return false;
        }

        // ── MISMATCH: local index built under a different chunker/model/revision/dim.
        const cur = pluginIdentity();
        const firstReport = !this.identityHealNotified;
        this.identityHealNotified = true;
        if (firstReport) {
            await this.logger.appendError('index-identity-mismatch', new Error(
                `stored chunker=${meta.chunkerVersion} model=${meta.modelId} rev=${meta.revision} dim=${meta.embeddingDim}` +
                ` → build chunker=${cur.chunkerVersion} model=${cur.modelId} rev=${cur.revision} dim=${cur.dim}`)).catch(() => {});
        }

        this.identityHealInFlight = true;
        try {
            // 1. Both platforms: hydrate from a matching-identity peer first (embed-free).
            //    rebuildFromSidecar nukes+stamps+hydrates ONLY if a compatible producer
            //    exists; otherwise it returns acceptedProducers:0 without touching the index.
            const rebuilt = await this.orchestrator.rebuildFromSidecar();
            if (rebuilt && rebuilt.acceptedProducers > 0) {
                this.identityHealNotified = false; // healed (silent — background sync)
                return true;
            }

            // 1b. No matching peer. Before the destructive fallback, try the embed-free
            //     in-place heal: an index that is merely UNSTAMPED but provably current
            //     (same model+dim, files unchanged — the cold-build identity bug, PR #43)
            //     is stamped in seconds instead of a ~7-min nuke+re-embed, and an
            //     interrupted attempt resumes (re-runs the seconds-long proof) instead of
            //     restarting from zero. Both platforms reach it; mobile stays embed-free.
            //     Only a GENUINELY old index ('stale') falls through to the rebuild/wait.
            const healed = await this.orchestrator.reconcileIdentityInPlace();
            if (healed === 'stamped') {
                this.identityHealNotified = false;
                this.indexHealth = 'healthy';
                return true;
            }
            if (healed === 'drained') {
                // Healed enough to stop the destructive loop, but a few edits are deferred
                // (mobile / model not yet loaded): the index isn't fully current, so leave
                // it degraded — the catch-up drain + a later poll finish and stamp it.
                this.indexHealth = 'degraded';
                return true;
            }

            // 2. No matching peer and a genuinely old index.
            if (isMobilePlatform()) {
                // Mobile NEVER bulk-embeds (jetsam). Wait: the stale chunks are invisible
                // (content-addressed ids don't reproduce → idx===undefined skip = empty/
                // partial results, never wrong). The 5-min poll heals this the moment a
                // desktop publishes a current sidecar.
                this.indexHealth = 'degraded';
                if (firstReport) new Notice(
                    'Seek: this version needs a fresh index — reindex on a desktop and this device will sync it automatically.', 8000);
                return true;
            }
            // Desktop is the source of truth: auto full reindex, no prompt (decision #1).
            // Silent — the reindex itself shows the start + recap toast.
            // Only clear the "needs heal" report latch if the reindex actually ran (and
            // thus stamped the current identity). It can be refused if a flush grabbed
            // the write mutex during the rebuildFromSidecar await above — leave the latch
            // set so the next poll re-detects the still-stale index and re-heals.
            if (await this.runFullReindex({ skipConfirm: true })) this.identityHealNotified = false;
            return true;
        } finally {
            this.identityHealInFlight = false;
        }
    }

    // True while the index is being built or mutated — a reindex/delta under the write
    // mutex (orchestrator.isWriting, which covers the cold first-build's reindexDelta),
    // a catch-up drain, a live flush, or a drift-recovery pass. The reconcile poll
    // consults this so a long build is never reconciled / healed out from under itself
    // on a large vault (Fix C: reindex_time > poll_interval). identityHealInFlight is
    // handled separately inside enforceIndexIdentity (it reports "handled", not "busy").
    private isIndexBusy(): boolean {
        return this.catchUpRunning || this.flushing || this.driftRecoveryRunning
            || (this.orchestrator?.isWriting() ?? false);
    }

    // Periodic sidecar poll (remote sidecar arrivals fire no vault events). Re-runs the
    // identity gate FIRST so a device that was WAITING for a peer heals the moment a
    // desktop publishes a matching sidecar; otherwise the normal dir-signature-gated
    // catch-up. Wired to the 5-min interval.
    private async periodicReconcile(): Promise<void> {
        try {
            // Never poll while a build/delta is in flight: assessing identity or
            // hydrating a sidecar would race a writer whose meta is mid-stamp. The
            // next tick (build done, identity stamped) handles it. This is the fix for
            // the large-vault loop where a 415s reindex outlived the 300s poll.
            if (this.isIndexBusy()) return;
            if (await this.enforceIndexIdentity()) return;
            await this.orchestrator.reconcileSidecarIfChanged();
            // Once per session, after identity is confirmed healthy: GC orphan chunks
            // from same-version churn (a missed delete event, a file record overwritten
            // by hydrate). Embed-free set-arithmetic; if backgrounded mid-sweep it
            // aborts and the NEXT poll tick resumes it (latch only on completion).
            if (!this.orphanSweepDone && !this.orphanSweepRunning) {
                this.orphanSweepRunning = true;
                try {
                    const { removed, completed } = await this.orchestrator.sweepOrphanChunks({ shouldContinue: () => !document.hidden });
                    if (completed) this.orphanSweepDone = true;
                } finally {
                    this.orphanSweepRunning = false;
                }
            }
            // Once per session: compact this device's own sidecar if it has bloated with
            // dead records (superseded re-appends + un-tombstoned orphans). Model-free
            // byte-copy, so mobile-safe; reclaims the disk a full reindex would otherwise
            // be the only thing to free — and the only GC reaching an off-grid device.
            if (!this.sidecarCompactDone && !this.sidecarCompactRunning) {
                this.sidecarCompactRunning = true;
                try {
                    const r = await this.orchestrator.compactOwnSidecar();
                    if (r?.compacted) {
                        // A shed = this device's OWN shard was unreadable/corrupt for a
                        // still-live record → that note needs a model re-embed to be
                        // searchable again. Expected zero; surface any non-zero as a
                        // corruption breadcrumb (invisible on mobile without it).
                        if (r.shed > 0) await this.logger.appendError('sidecar-compaction-shed', new Error(`shed ${r.shed} corrupt/unreadable record(s)`)).catch(() => {});
                    } else if (r && r.reason === 'incomplete-rechunk') {
                    }
                    // Latch on any definitive verdict; an incomplete re-chunk is transient,
                    // so leave it open to retry. null = sidecar off → don't latch (it may
                    // be enabled later this session).
                    if (r && r.reason !== 'incomplete-rechunk') this.sidecarCompactDone = true;
                } catch (e) {
                    // A hard failure (disk full, write error) would otherwise re-run the
                    // whole-vault re-chunk every poll forever — log it and latch off for
                    // the session rather than spin silently.
                    await this.logger.appendError('sidecar-compaction', e).catch(() => {});
                    this.sidecarCompactDone = true;
                } finally {
                    this.sidecarCompactRunning = false;
                }
            }
        } catch (e) {
            await this.logger.appendError('periodic-reconcile', e).catch(() => {});
        }
    }

    // Generate the diagnostic report (full seek-report.json + a short seek-report.md
    // summary) and open the .md. The user-facing debug affordance, surfaced as a
    // Settings button now that the command-palette entry is gone. Errors tee to
    // console + NDJSON as usual.
    async openLoggingReport(): Promise<void> {
        try {
            const path = await this.logger.writeReport();
            const file = this.app.vault.getAbstractFileByPath(path);
            if (file instanceof TFile) await this.app.workspace.getLeaf(false).openFile(file);
            new Notice(`Seek: report written — ${path} (summary) + seek-report.json (full data)`, 6000);
        } catch (e) {
            await this.logger.appendError('generate-log', e);
            new Notice('Seek: could not write the logging report — see the developer console.', 6000);
        }
    }

    private async warnOnModelIndexDrift(): Promise<void> {
        if (this.modelDriftWarned) return;
        try {
            const meta = await this.store.getMeta();
            if (meta.lastIndexedAt === null) return;   // empty index — first reindex will stamp it
            const indexModel = meta.modelId ?? LEGACY_ENGLISH_MODEL_ID;
            if (indexModel !== this.embedder.modelId) {
                this.modelDriftWarned = true;
                new Notice(
                    'Seek: the index was built with a different embedding model. ' +
                    'Open Settings → Seek → Index and choose Reindex — until then, incremental ' +
                    'indexing is paused and semantic ranking is unreliable.',
                    15000,
                );
                await this.logger.appendError('model-index-drift', new Error(
                    `index=${indexModel} loaded=${this.embedder.modelId}`));
            }
        } catch { /* meta unavailable (store closed mid-teardown) — next load rechecks */ }
    }

    // ---- Incremental indexing ----
    //
    // Keeps the index fresh without re-embedding the vault, on cheap triggers
    // that never fire mid-edit:
    //   - Leaving a note (active-leaf-change) enqueues it IF its mtime advanced
    //     (the mtime guard skips read-only visits); a 5-min idle timer then
    //     flushes. Edits are NOT watched per-keystroke — leaving the note is the
    //     debounce.
    //   - Backgrounding (visibilitychange:hidden / pagehide / window blur) flushes
    //     immediately — the last safe write window on iOS.
    //   - Deletes/renames/creates are discrete structural events with no blur
    //     equivalent, so we watch them directly; a rename is drop-old + index-new,
    //     which makes a move into an ignored folder a soft-delete.
    //   - A BULK flush (> BULK_DELTA_THRESHOLD dirty files = a paste / vault sync /
    //     git checkout, not an edit) is a mini-reindex: progress shows on a Notice,
    //     a live query preempts the embed (shouldContinue), a cold DESKTOP model is
    //     deferred not force-loaded (like reconcileOnLoad), and the deferred/
    //     preempted remainder is reconciled by the drain. Small deltas stay a plain
    //     force-embed.
    // The catch-up drain (runCatchUp/drainCatchUp) reconciles deferred embeds —
    // cold-mobile, cold-desktop-bulk, or query-preempted — via a computeDelta diff.
    // It runs once a search SESSION ENDS (onSearchActivity, query settled / modal
    // closed, NOT per keystroke, so the foreground embed never competes with the
    // live query on the shared iOS process) AND is kicked immediately after a bulk
    // flush (a no-op when nothing's left). Startup reconciliation (reconcileOnLoad)
    // backstops anything missed while Seek wasn't running.
    private wireIncrementalIndexing(): void {
        this.lastActiveFile = this.app.workspace.getActiveFile();

        // Edits: index the note you LEAVE, not the one you arrive at.
        this.registerEvent(this.app.workspace.on('active-leaf-change', () => {
            const left = this.lastActiveFile;
            this.lastActiveFile = this.app.workspace.getActiveFile();
            if (left) void this.enqueueIfDirty(left);
        }));

        // Structural events — discrete, rare, no blur equivalent.
        this.registerEvent(this.app.vault.on('create', (f) => {
            if (f instanceof TFile && isIndexableFile(f, this.settings.indexBases)) { this.dirtyQueue.add(f.path); this.scheduleFlush(); }
        }));
        this.registerEvent(this.app.vault.on('delete', (f) => {
            if (!(f instanceof TFile) || !isIndexableFile(f, this.settings.indexBases)) return;
            this.deletedQueue.add(f.path);
            this.dirtyQueue.delete(f.path);
            this.flushStructuralSoon();
        }));
        this.registerEvent(this.app.vault.on('rename', (f, oldPath) => {
            // Drop the old path (covers plain rename, move, and move-into-ignored
            // = soft-delete) and index the new one. shouldIndex/reindexDelta decide
            // the archive/un-archive outcome by destination.
            this.deletedQueue.add(oldPath);
            this.dirtyQueue.delete(oldPath);
            if (f instanceof TFile && isIndexableFile(f, this.settings.indexBases)) this.dirtyQueue.add(f.path);
            this.flushStructuralSoon();
        }));

        // Window blur (desktop alt-tab) — same intent as visibilitychange:hidden.
        // registerDomEvent auto-tears-down on unload.
        this.registerDomEvent(window, 'blur', () => this.flushOnBackground());
    }

    // Enqueue a note for re-index only if it actually changed since we last
    // indexed it (the mtime guard) — so navigating through notes to READ them
    // never triggers an embed. One quick IDB read per note-leave.
    private async enqueueIfDirty(file: TFile | null): Promise<void> {
        if (!file || !isIndexableFile(file, this.settings.indexBases)) return;
        if (!this.orchestrator) return;
        try {
            const stored = await this.store.getFileRecord(file.path);
            if (!stored || file.stat.mtime > stored.mtimeMs) {
                this.dirtyQueue.add(file.path);
                this.scheduleFlush();
            }
        } catch (e) {
            await this.logger.appendError('enqueueIfDirty', e).catch(() => {});
        }
    }

    // 5-min idle debounce for edits: resets on every enqueue, so flipping back to
    // keep writing pushes the flush out rather than firing mid-thought.
    private scheduleFlush(): void {
        if (this.idleTimer != null) window.clearTimeout(this.idleTimer);
        this.idleTimer = window.setTimeout(() => { this.idleTimer = null; void this.flushDirty(); }, IDLE_FLUSH_MS);
    }

    // Short debounce for structural changes — model-free, so flush them soon
    // rather than waiting out the 5-min edit window (a dead result is jarring).
    private flushStructuralSoon(): void {
        if (this.structTimer != null) window.clearTimeout(this.structTimer);
        this.structTimer = window.setTimeout(() => { this.structTimer = null; void this.flushDirty(); }, STRUCT_FLUSH_MS);
    }

    // Backgrounding flush: capture the note currently being edited (it may never
    // have fired active-leaf-change) and drain immediately, before the OS can
    // reclaim the WebView.
    private flushOnBackground(): void {
        const active = this.app.workspace.getActiveFile();
        const flushed = active
            ? this.enqueueIfDirty(active).then(() => this.flushDirty())
            : this.flushDirty();
        // Mobile: once the last-safe-window flush settles, free the model before
        // iOS can jetsam-kill the backgrounded WebView. Chained AFTER the flush so
        // the unload predicate sees its settled state (not a half-armed queue);
        // any embeds the mobile flush deferred are re-found by computeDelta and
        // reloaded on the next foreground, so this never drops work.
        if (isMobilePlatform()) void flushed.then(() => this.maybeUnloadEmbedder('background')).catch(() => {});
    }

    // Drain the dirty/deleted queues through one reindexDelta. Deletes/moves
    // always apply (model-free structural phase); the embed half runs now on
    // desktop or a warm model, and defers on a cold mobile model (the edit's old
    // version stays searchable and the post-serve catch-up re-embeds it).
    private async flushDirty(): Promise<void> {
        if (this.flushing || !this.orchestrator) {
            // A timer fired while a flush was already running (the in-progress flush
            // snapshotted BEFORE these items, so they need their own cycle). Re-arm
            // one so queued edits don't wait for the next unrelated enqueue or a
            // restart. Guard on no pending timer to avoid stacking; a stale re-arm
            // is harmless — flushDirty no-ops on an empty queue.
            if (this.flushing && (this.dirtyQueue.size > 0 || this.deletedQueue.size > 0)
                && this.idleTimer == null && this.structTimer == null) {
                this.scheduleFlush();
            }
            return;
        }
        if (this.dirtyQueue.size === 0 && this.deletedQueue.size === 0) return;
        this.flushing = true;
        const orchestrator = this.orchestrator;
        try {
            const dirty = [...this.dirtyQueue];
            const deleted = [...this.deletedQueue];
            // Snapshot each dirty file's mtime so the cleanup below can tell "this
            // exact version was flushed" from "re-edited during the await". A re-edit
            // bumps the mtime and enqueueIfDirty re-adds the path; a blind delete
            // would then clobber that second edit (lost until reconcileOnLoad).
            // -1 = the file vanished (deleted/moved) — handled via deletedQueue.
            const dirtyMtimes = new Map<string, number>();
            for (const p of dirty) {
                const f = this.app.vault.getAbstractFileByPath(p);
                dirtyMtimes.set(p, f instanceof TFile ? f.stat.mtime : -1);
            }
            const bulk = dirty.length > BULK_DELTA_THRESHOLD;
            // Cold-model embed deferral. Mobile: a cold model always defers (the
            // jetsam rule). Desktop: additionally defer a BULK cold flush — a
            // background paste/sync shouldn't force a ~250 MB model load; treat it
            // like reconcileOnLoad and let the first search drive the embed. A small
            // cold delta still loads, because the user is actively in that note.
            const coldMobile = isMobilePlatform() && !this.embedder.loaded;
            const coldDesktopBulk = !isMobilePlatform() && !this.embedder.loaded && bulk;
            const deferEmbed = coldMobile || coldDesktopBulk;
            if (!deferEmbed) await this.ensureModelLoaded();

            if (bulk) {
                // Mini-reindex path. Progress goes on the same sticky Notice the
                // full reindex uses (deferred embeds have nothing to show). A live
                // query preempts the embed (shouldContinue → break within one file,
                // releasing the write mutex so the query runs); the interrupted or
                // deferred remainder keeps its old searchable chunks and stays dirty
                // (no file-record advance) for the drain to reconcile.
                const notice = deferEmbed ? null : new Notice('Seek: indexing new notes…', 0);
                try {
                    await orchestrator.reindexDelta(dirty, deleted, {
                        embed: !deferEmbed,
                        shouldContinue: () => !this.indexingBlocked,
                        onProgress: notice ? (msg) => notice.setMessage(`Seek: ${msg}`) : undefined,
                    });
                } finally {
                    notice?.hide();
                }
                // Reconcile whatever the embed left undone (deferred cold, or
                // preempted by a query). runCatchUp is self-guarding (no-op while
                // searching / hidden / model cold) and computeDelta-idempotent — a
                // fully-completed pass just costs one empty diff.
                this.catchUpPending = true;
                this.runCatchUp();
            } else {
                // Single-note fast path. Still preempt on a live OR in-flight query
                // so a just-edited note can't force a main-thread embed mid-search —
                // the foreground query must win the shared iOS thread (the note's
                // "indexing must wait for the query"). A preempted (or cold-deferred)
                // note keeps its old chunks, doesn't advance its file record, and so
                // stays dirty + pending for runCatchUp, which fires the moment the
                // query completes (onQueryInFlight(false)) — computeDelta re-finds it.
                await orchestrator.reindexDelta(dirty, deleted, {
                    embed: !deferEmbed,
                    shouldContinue: () => !this.indexingBlocked,
                });
                if ((coldMobile || this.indexingBlocked) && dirty.length > 0) this.catchUpPending = true;
                this.runCatchUp();
            }

            // Clear exactly what we snapshotted — but only dirty paths NOT re-edited
            // mid-flush (mtime unchanged since the snapshot). A re-enqueued path
            // stays for the next flush instead of being clobbered; new paths queued
            // during the await were never in `dirty`. Deletes are unconditional.
            for (const p of deleted) this.deletedQueue.delete(p);
            for (const p of dirty) {
                const f = this.app.vault.getAbstractFileByPath(p);
                const current = f instanceof TFile ? f.stat.mtime : -1;
                if (current === dirtyMtimes.get(p)) this.dirtyQueue.delete(p);
            }
        } catch (e) {
            await this.logger.appendError('flushDirty', e).catch(() => {});
        } finally {
            this.flushing = false;
        }
    }

    // Startup mtime-diff sweep. Authoritative diff of the persisted index vs. the
    // live vault — catches external sync, edits made while disabled, and deletes
    // missed by a crash. Deletes/moves apply immediately (model-free). Edits are
    // DEFERRED on every platform: forcing a cold model load at startup would break
    // the lazy-load contract (don't spend ~250 MB if the user never searches), so
    // we set catchUpPending and let the first search's model load drive the embed
    // via runCatchUp. The live-session flush (flushDirty) is where the desktop
    // force applies, since the model is typically already warm by then.
    private async reconcileOnLoad(): Promise<void> {
        if (!this.orchestrator) return;
        try {
            const { dirty, deleted } = await this.orchestrator.computeDelta();
            if (dirty.length === 0 && deleted.length === 0) return;
            await this.orchestrator.reindexDelta(dirty, deleted, { embed: false });
            if (dirty.length > 0) this.catchUpPending = true;
        } catch (e) {
            await this.logger.appendError('reconcileOnLoad', e).catch(() => {});
        }
    }

    // The modal reports its session lifecycle here: opening + each keystroke =
    // active (pauses the drain so a foreground embed never competes with the live
    // query on the shared iOS WebContent process); query-settled + modal-closed =
    // inactive, which is the trigger to drain. Gated/no-op when nothing's pending.
    private onSearchActivity(active: boolean): void {
        this.searchActiveTimestamp = active ? Date.now() : null;
        // Session settled = a safe window. Drive both deferred drains: catch-up (embeds)
        // and drift recovery (embed-free). Each is a cheap no-op unless it has pending work.
        if (!active) { this.runCatchUp(); this.runDriftRecovery(); }
    }

    // Hard query-lifecycle signal from the modal: true the moment a query embed
    // starts, false when its results paint (ref-counted modal-side across the cold
    // path). Held in indexingBlocked so a preempted/deferred reindex resumes only
    // AFTER the query completes — the note's "make indexing wait for the query".
    // The false edge is a drain trigger, exactly like onSearchActivity(false).
    private onQueryInFlight(inFlight: boolean): void {
        this.queryInFlight = inFlight;
        // Query complete = a safe window — same dual drive as onSearchActivity(false).
        if (!inFlight) { this.runCatchUp(); this.runDriftRecovery(); }
    }

    // Open the Seek search modal, optionally seeded with a query (the `seek-search`
    // command passes none; the obsidian://seek deep-link passes the URL's query).
    // Decouples modal-open from model-load: the model can take 3–10 s cold-start
    // (7.6 s on iOS first-run, per the [[Seek Model Performance]] revision trail)
    // and the input field has no reason to wait — the orchestrator only needs the
    // model at query-execution time. We start the load eagerly (overlapping the
    // user's typing latency) and hand the in-flight promise to the modal, which
    // awaits it inside `runSearch`, not `onOpen`. The .catch is the unhandled-
    // rejection guard for when nothing in the modal ever awaits it (e.g. the user
    // closes the modal before typing); errors are already logged in ensureModelLoaded.
    private openSearchModal(initialQuery = ''): void {
        this.currentTaskContext = 'search';
        try {
            const wasLoaded = this.embedder.loaded;
            const loadPromise = this.ensureModelLoaded();
            loadPromise.catch(() => { /* logged in ensureModelLoaded */ });
            new SeekSearchModal(
                this.app,
                this.orchestrator,
                this.logger,
                { ready: wasLoaded, promise: loadPromise },
                this.settings,
                (active) => this.onSearchActivity(active),
                (inFlight) => this.onQueryInFlight(inFlight),
                initialQuery,
            ).open();
        } catch (e) {
            // Synchronous failure path (rare — only if the Modal ctor or
            // ensureModelLoaded throws before returning a promise).
            this.logger.appendError('seek-search:open', e).catch(() => {});
            new Notice('Seek: search failed to open — see the developer console.');
        } finally {
            // Modal lifecycle isn't observable from here; reset to idle after open().
            this.currentTaskContext = 'idle';
        }
    }

    // Headless deep-link target (obsidian://seek?query=…&mode=open): load the
    // model, run the query, and open the top hit's note directly — no modal.
    // Mirrors the seek:search CLI handler's model-gating (cold start blocks, so a
    // Notice stands in for the modal's progress UI). An empty query falls back to
    // the normal modal so a malformed link still does something useful.
    private async openTopResult(query: string): Promise<void> {
        if (!query.trim()) { this.openSearchModal(); return; }
        if (!this.orchestrator) { new Notice('Seek: still loading — try again in a moment'); return; }
        const notice = new Notice(`Seek: searching “${query}”…`, 0);
        this.currentTaskContext = 'search';
        try {
            // No modal to overlap the cold-start, so block on the model load
            // (3–10 s first call) before querying — same as the CLI handler.
            await this.ensureModelLoaded();
            // topK=1 — we only open the single best hit.
            const { results } = await this.orchestrator.search(query, 1);
            notice.hide();
            const top = results[0];
            if (!top) { new Notice(`Seek: no results for “${query}”`); return; }
            const file = this.app.vault.getAbstractFileByPath(top.note_path);
            if (!(file instanceof TFile)) { new Notice(`Seek: top result not on disk (${top.note_path})`); return; }
            await this.app.workspace.getLeaf(false).openFile(file);
        } catch (e) {
            notice.hide();
            this.logger.appendError('seek:protocol-open', e).catch(() => {});
            new Notice('Seek: could not open the result — see the developer console.');
        } finally {
            this.currentTaskContext = 'idle';
        }
    }

    // Drain cold-mobile deferred embeds once a search session ends and the model is
    // warm. Runs the work in safety-bounded, self-chaining bursts (drainCatchUp) so
    // the foreground iOS embed can't saturate the shared process into a jetsam kill;
    // desktop runs one unbounded burst (no caps). Fire-and-forget; cheap no-op
    // unless something was deferred (catchUpPending) and we're in a safe window.
    private runCatchUp(): void {
        if (!this.catchUpPending || this.catchUpRunning || !this.embedder.loaded || !this.orchestrator) return;
        if (document.hidden || this.indexingBlocked) return;  // never start in a bad window (typing OR a query in flight)
        this.catchUpRunning = true;
        const orchestrator = this.orchestrator;
        const mobile = isMobilePlatform();
        const pacer = new CompositorPacer();
        void (async () => {
            try {
                const { pending } = await drainCatchUp({
                    computeDelta: () => orchestrator.computeDelta(),
                    reindexDelta: (d, del, opts) => orchestrator.reindexDelta(d, del, opts),
                    isHidden: () => document.hidden,
                    isSearchActive: () => this.indexingBlocked,
                    pace: () => pacer.pace(),
                    maxFiles: mobile ? CATCHUP_MAX_FILES_PER_BURST : undefined,
                    budgetMs: mobile ? CATCHUP_BURST_BUDGET_MS : undefined,
                });
                this.catchUpPending = pending;
            } catch (e) {
                await this.logger.appendError('runCatchUp', e).catch(() => {});
                this.catchUpPending = true;  // unknown state — let a later trigger retry
            } finally {
                this.catchUpRunning = false;
            }
        })();
    }

    // Persistent-drift escalation from the orchestrator (onCoherenceDrift's re-trip
    // branch). Gen-keyed suppression: only escalate once per index generation, so a
    // degraded index re-tripping drift on every keystroke doesn't re-arm recovery — a
    // later mutation (delta/reindex/invalidate/hydrate) bumps the generation and
    // re-arms it for the new state. Sets pending + drives; the actual ladder runs on
    // the next safe edge (mirrors catch-up's pending/drive split).
    private onPersistentDrift(): void {
        if (!this.orchestrator) return;
        const currentGen = this.orchestrator.currentGeneration();
        const { schedule } = driftRecoveryDecision({
            running: this.driftRecoveryRunning,
            health: this.indexHealth,
            lastRecoveryGen: this.lastDriftRecoveryGen,
            currentGen,
        });
        if (!schedule) return;
        this.lastDriftRecoveryGen = currentGen;
        this.driftRecoveryPending = true;
        this.runDriftRecovery();
    }

    // The embed-free recovery ladder, run in a gated, single-flighted scheduler (the
    // sibling of runCatchUp). Execution order is sidecar hydrate (reconcile the IDB
    // against the vault; tokenizer-only re-chunk, NO embed; null no-op when sidecar is
    // off) → warm (rebuild frame + BM25 from the reconciled IDB) → verify. NEVER reaches
    // reindexAll/embedAndCommitFiles/embedder.embed* — re-embedding is never the fix for
    // a row-space coupling bug. Unlike runCatchUp it does NOT require embedder.loaded
    // (the ladder is tokenizer-only). On exhaustion: indexHealth='degraded' +
    // console.error (NO Notice on this path), surfaced on the settings page.
    private runDriftRecovery(): void {
        if (!this.driftRecoveryPending || this.driftRecoveryRunning || !this.orchestrator) return;
        if (document.hidden || this.indexingBlocked) return;  // never start in a bad window (typing OR a query in flight)
        this.driftRecoveryRunning = true;
        this.indexHealth = 'recovering';
        const orchestrator = this.orchestrator;
        void (async () => {
            // The generation verifyCoherent actually evaluated; -1 until we reach verify.
            // A user-initiated full reindex (and any delta) runs OFF this ladder's write
            // mutex — warmCaches/verifyCoherent read the store un-serialized — so it can
            // land mid-verify, advancing the generation and OWNING the index's outcome.
            // We therefore commit our verdict + suppression key only when that generation
            // is still live (optimistic concurrency, mirroring ensureFrame's F2
            // shouldDiscardPartialFrame discard). If a mutation raced us it is
            // authoritative — its reindex left a coherent 'healthy' index, or its delta
            // re-armed a fresh drift trip — so a stale 'degraded' here would wrongly stick.
            let verifiedGen = -1;
            try {
                // Step 1 — sidecar hydrate: reconcile IDB against the live vault
                // (embed-free; null no-op when sidecar off), which itself re-warms on a
                // >0 hydrate. On the write mutex, so it can't overlap a reindex's store.close().
                await orchestrator.hydrateSidecar();
                // Step 2 — warm: co-rebuild frame + BM25 from one (reconciled) IDB
                // snapshot, fixing the transient row-space desync (on cold mobile
                // warmCaches no-ops, but the verify below rebuilds via ensureFrame/ensureBm25).
                await orchestrator.warmCaches('drift-recovery');
                // Verify the rebuild actually re-coupled the row spaces.
                verifiedGen = orchestrator.currentGeneration();
                const ok = await orchestrator.verifyCoherent();
                if (orchestrator.currentGeneration() !== verifiedGen) {
                    // Raced: a reindex/delta landed off our mutex during verify and owns
                    // the outcome. Leave health 'recovering' (invisible in settings) and
                    // the escalation suppression key untouched, so the new generation
                    // re-arms recovery on the next drift trip.
                } else {
                    this.commitDriftHealth(ok ? 'healthy' : 'degraded', verifiedGen);
                    if (ok) {
                    } else {
                        console.error('[seek] drift auto-recovery exhausted (embed-free warm + sidecar reconcile did not re-couple the frame/BM25 row space) — indexHealth=degraded; a full reindex recovers it');
                    }
                }
            } catch (e) {
                // verifyCoherent's IDB read throws if a concurrent reindex closed the
                // store mid-read. Degrade only on a GENUINE failure; if a writer raced us
                // (the generation advanced), defer — it owns the outcome, and a reindex's
                // own completion re-clears health, so a stale 'degraded' can't stick.
                if (verifiedGen < 0) {
                    // Threw before verify (e.g. a hydrate failure) — a real recovery error.
                    this.commitDriftHealth('degraded', orchestrator.currentGeneration());
                    await this.logger.appendError('runDriftRecovery', e).catch(() => {});
                } else if (orchestrator.currentGeneration() === verifiedGen) {
                    // Verify threw with the generation unchanged — a real failure
                    // (corrupt store, not a concurrent writer).
                    this.commitDriftHealth('degraded', verifiedGen);
                    await this.logger.appendError('runDriftRecovery', e).catch(() => {});
                } else {
                    // The generation advanced: a concurrent reindex/delta closed the store
                    // mid-verify and owns the outcome. Expected under the race — defer, and
                    // don't log it as an error (it isn't one).
                }
            } finally {
                this.driftRecoveryPending = false;
                this.driftRecoveryRunning = false;
            }
        })();
    }

    // Commit a terminal drift-recovery verdict: set indexHealth AND re-baseline the
    // suppression key to the SAME generation the verdict describes. Coupling them keeps
    // the two from disagreeing — re-reading currentGeneration() in a finally (as an
    // earlier cut did) could pin suppression to a later, UN-verified generation (a delta
    // that raced verify), hiding the degraded affordance for a state we never checked.
    private commitDriftHealth(health: 'healthy' | 'degraded', gen: number): void {
        this.indexHealth = health;
        this.lastDriftRecoveryGen = gen;
    }

    // Full reindex: nuke the index and re-embed every markdown file. Shared by the
    // command palette and the settings "Full reindex" button (the degraded-health
    // recovery affordance). USER-INITIATED, so unlike the automatic drift-recovery
    // ladder it is allowed to embed — the embed-free guarantee covers only the auto path.
    // opts.skipConfirm: the caller already confirmed (the settings Index section has its
    // own inline two-button confirm, so it suppresses the blocking window.confirm).
    // opts.onProgress: a second progress sink (besides the Notice) — the settings status
    // card subscribes to drive its live "N / TOTAL notes" bar. Arg-less callers (command
    // palette main.ts:496, degraded-health button) keep the confirm + Notice unchanged.
    // Returns whether a reindex actually RAN — false when refused (a build is already
    // running) or declined at the confirm, or if it threw. The identity heal uses this
    // to avoid falsely clearing its "needs heal" latch when its reindex didn't happen.
    async runFullReindex(opts?: { skipConfirm?: boolean; onProgress?: (msg: string) => void }): Promise<boolean> {
        // Refuse to STACK a reindex on a running one. The write mutex would otherwise
        // queue a second reindex behind the first (it nukes + re-embeds the whole
        // vault again the moment the first releases) — pure waste, and on a large
        // vault the second's nuke is exactly the deleteDatabase that fired versionchange
        // at an in-flight pass. The identity heal also routes through here, so this
        // doubles as a guard against a heal stacking on a manual reindex (Fix B). The
        // catch-up cold build runs under the same mutex (isWriting covers it too).
        if (this.orchestrator.isWriting()) {
            new Notice('Seek: a reindex is already running.', 4000);
            return false;
        }
        if (!opts?.skipConfirm) {
            const confirm = `This will delete the existing Seek index and re-embed every markdown file in this vault.\nProceed?`;
            if (!window.confirm(confirm)) return false;
        }

        const notice = new Notice('Seek: full reindex starting…', 0);
        this.currentTaskContext = 'indexing';
        try {
            await this.ensureModelLoaded();
            this.orchestrator.invalidateBm25Cache();
            const result = await this.orchestrator.reindexAll(msg => {
                notice.setMessage(`Seek: ${msg}`);
                opts?.onProgress?.(msg);
            });
            notice.hide();
            const summary = [
                result.pass ? '✅' : '❌',
                `${result.filesIndexed} files`,
                `${result.chunksIndexed} chunks`,
                `${(result.totalDurationMs / 1000).toFixed(1)} s`,
                `${result.chunksPerSec.toFixed(1)} ch/s`,
            ].join(' · ');
            new Notice(`Seek reindex: ${summary}`, 10000);

            // Post-reindex storage snapshot — answers "how much disk does the index
            // actually consume?" without waiting for the next platform probe (reload).
            await this.emitStorageSnapshot('post-reindex');
            // A full reindex re-couples the index from scratch (it is the terminal
            // recovery): clear any lingering degraded health so the settings affordance
            // and the suppression baseline both reset.
            this.indexHealth = 'healthy';
            return true;
        } catch (e) {
            notice.hide();
            await this.logger.appendError('seek-full-reindex', e);
            // One end-toast whether it passed or failed (the recap). Detail → console + log.
            new Notice('Seek reindex: ❌ failed — see the logging report (Settings → Seek).', 10000);
            return false;
        } finally {
            this.currentTaskContext = 'idle';
        }
    }

    // Index health snapshot for the settings Index status card. Read-only; gathers the
    // private store + logger so the settings tab needs no plugin internals. Counts come
    // from the index store, on-disk bytes from the Storage API (iOS-safe), the last-index
    // timestamp from index meta (persisted), and its duration from the newest index-complete
    // log entry (logged only). Each source is independently guarded so a single failure
    // degrades one field rather than blanking the whole card.
    async getIndexStats(): Promise<IndexStats> {
        let files = 0, chunks = 0;
        try { const c = await this.store.count(); files = c.files; chunks = c.chunks; } catch { /* index not open yet */ }
        let storageMB: number | null = null, indexMB: number | null = null, modelMB: number | null = null;
        if (navigator.storage?.estimate) {
            try {
                const est = await navigator.storage.estimate();
                storageMB = est.usage != null ? est.usage / 1e6 : null;
                // usageDetails is non-standard but present in Electron/Chromium: per-bucket
                // bytes ({ indexedDB, caches, ... }). Split the origin total into the index
                // (IndexedDB) vs the model cache (Cache API) so the card stops implying the
                // whole origin is the index.
                const details = (est as { usageDetails?: Record<string, number> }).usageDetails;
                if (details) {
                    if (typeof details.indexedDB === 'number') indexMB = details.indexedDB / 1e6;
                    if (typeof details.caches === 'number') modelMB = details.caches / 1e6;
                }
            } catch { /* unsupported */ }
        }
        // Last FULL reindex: newest index-complete entry with mode==='full'. Taking the
        // stamp AND the duration from that one entry keeps them coherent — a later
        // incremental catch-up (mode==='incremental') is skipped, so its tiny duration
        // never gets shown as if it were a full rebuild.
        let lastFullAt: string | null = null;
        let lastFullDurationMs: number | null = null;
        try {
            const entries = await this.logger.readAll();
            for (let i = entries.length - 1; i >= 0; i--) {
                if (entries[i].type !== 'index-complete') continue;
                const ic = entries[i] as IndexCompleteEntry;
                if (ic.mode !== 'full') continue;
                lastFullAt = ic.timestamp;
                lastFullDurationMs = ic.totalDurationMs;
                break;
            }
        } catch { /* log unreadable */ }
        // Last index of ANY mode (meta is rewritten on every reindex + delta) → "updated".
        // The same meta carries the dense background stats: a calibrated corpus has
        // bgMean/bgStd with σ>0 (mirrors SearchOrchestrator.getDenseBgStats).
        let lastUpdatedAt: string | null = null;
        let calibrated = false;
        try {
            const meta = await this.store.getMeta();
            lastUpdatedAt = meta.lastIndexedAt;
            calibrated = meta.bgMean != null && meta.bgStd != null && meta.bgStd > 0;
        } catch { /* meta unreadable */ }
        return { files, chunks, storageMB, indexMB, modelMB, lastFullAt, lastFullDurationMs, lastUpdatedAt, calibrated };
    }

    // Embedding-model DOWNLOAD status for the settings Model section — distinct from
    // "loaded into memory" (which the search modal signals ambiently, so settings does
    // not repeat it). Tries the parent-side Cache-API probe first; because parent
    // visibility into the iframe's cache is unproven on iPhone (the cacheSeen canary),
    // it falls back to a definitely-loaded embedder, then to the last model-delivery log
    // entry that recorded cacheSeen>0 — either implies the bytes are on disk. Never throws.
    async getModelStatus(): Promise<ModelStatus> {
        const spec = activeModelSpec(this.settings);
        const name = spec.repo.includes('/') ? spec.repo.split('/')[1] : spec.repo;
        let downloaded = false, persisted: boolean | null = null;
        if (typeof caches !== 'undefined') {
            const st = await probeModelDownloaded(caches, spec);
            downloaded = st.downloaded; persisted = st.persisted;
        }
        if (!downloaded) {
            if (this.embedder.loaded) {
                downloaded = true;
            } else {
                try {
                    const entries = await this.logger.readAll();
                    for (let i = entries.length - 1; i >= 0; i--) {
                        if (entries[i].type === 'model-delivery') {
                            if ((entries[i] as ModelDeliveryEntry).cacheSeen > 0) downloaded = true;
                            break;
                        }
                    }
                } catch { /* log unreadable */ }
            }
        }
        return { downloaded, persisted, name, dim: spec.dim };
    }

    // User-invoked "Delete model" (settings). Removes the active model's ~100 MB of
    // Cache-API bytes (the inverse of the switch-time evictStaleModelCaches), drops the
    // in-memory copy, and records a model-delivery entry with cacheSeen:0 so the three
    // sources getModelStatus() reads — Cache probe, loaded embedder, last delivery log —
    // all agree the model is gone. The next search re-downloads it. Best-effort +
    // parent-side: on iPhone, where the parent may not see the iframe's cache (the
    // cacheSeen canary), the bytes can survive until reload, same caveat as eviction.
    async deleteModel(): Promise<{ deleted: number }> {
        const spec = activeModelSpec(this.settings);
        let deleted = 0;
        try {
            if (typeof caches !== 'undefined') {
                deleted = (await deleteModelCaches(caches, spec.repo)).deleted;
            }
        } catch (e) {
            await this.logger.appendError('model-delete', e);
            throw e instanceof Error ? e : new Error(String(e));
        } finally {
            // Drop the runtime copy on ANY delete attempt — even a partial one. The iframe
            // holds the model in memory and getModelStatus() counts a loaded embedder as
            // "downloaded"; keeping it resident over a (possibly half-) deleted on-disk cache
            // would misreport state. teardown() also forces the next search to rebuild the
            // iframe and re-fetch cleanly — the intended consequence of a delete.
            this.embedder.teardown();
        }
        // Keep the log fallback honest: after a delete our cache holds 0 entries, so record
        // that — else the previous load's cacheSeen>0 would still read back as "downloaded".
        await this.logger.append({
            type: 'model-delivery',
            timestamp: new Date().toISOString(),
            key: spec.key,
            repo: spec.repo,
            revision: spec.revision,
            persisted: null,
            cacheSeen: 0,
            cacheEvicted: deleted,
        });
        return { deleted };
    }

    // ---- Global observers ----

    // Catch errors that escape the explicit try/catch sites. Without these,
    // async errors in event handlers / iframe message processing vanish.
    private wireGlobalErrorHandlers(): void {
        this.onError = (e: ErrorEvent) => {
            // Only log if the error originates from our code. The renderer
            // process gets a lot of unrelated cross-plugin noise, and we
            // don't want to claim other plugins' errors as ours.
            const src = (e.filename ?? '') + ' ' + (e.message ?? '');
            if (!/seek|transformers|webgpu/i.test(src)) return;
            this.logger.appendError('window.onerror', e.error ?? new Error(e.message)).catch(() => {});
        };
        this.onUnhandledRejection = (e: PromiseRejectionEvent) => {
            const reason = e.reason instanceof Error ? e.reason : new Error(String(e.reason));
            const stackStr = reason.stack ?? '';
            // Same filter as above. False negatives are fine; false positives
            // (logging other plugins' errors as Seek's) are worse.
            if (!/seek|transformers|webgpu/i.test(stackStr) && !/seek|transformers|webgpu/i.test(reason.message)) return;
            this.logger.appendError('unhandledrejection', reason).catch(() => {});
        };
        window.addEventListener('error', this.onError);
        window.addEventListener('unhandledrejection', this.onUnhandledRejection);
    }

    // PerformanceObserver for longtask entries. On iOS WKWebView this API
    // is supported as of iOS 16 — if absent, we silently skip (no harm).
    // Each longtask >= LONG_TASK_THRESHOLD_MS becomes a log entry tagged
    // with currentTaskContext so the report can group jank by what we
    // were doing at the time.
    private wireLongTaskObserver(): void {
        interface PolyfillObserver {
            new(cb: (list: { getEntries(): PerformanceEntry[] }) => void): PerformanceObserver;
        }
        const Ctor = (window as unknown as { PerformanceObserver?: PolyfillObserver }).PerformanceObserver;
        if (!Ctor) return;
        try {
            this.longTaskObserver = new Ctor(list => {
                for (const entry of list.getEntries()) {
                    if (entry.duration < LONG_TASK_THRESHOLD_MS) continue;
                    const attrSrc = entry as unknown as { attribution?: Array<{ name?: string }> };
                    const attribution = attrSrc.attribution?.[0]?.name ?? null;
                    const logEntry: LongTaskEntry = {
                        type: 'long-task',
                        timestamp: new Date().toISOString(),
                        durationMs: parseFloat(entry.duration.toFixed(2)),
                        startTimeMs: parseFloat(entry.startTime.toFixed(2)),
                        attribution,
                        context: this.currentTaskContext,
                    };
                    this.logger.append(logEntry).catch(() => {});
                }
            });
            this.longTaskObserver.observe({ entryTypes: ['longtask'] });
        } catch (e) {
            // entryType 'longtask' isn't supported everywhere — Safari pre-16.
            // Silently skip; the report will just have an empty long-task section.
            console.warn('[seek] longtask observer unavailable:', e);
        }
    }

    // visibilitychange + pagehide. On iOS, the WebView can be jetsam-killed
    // while backgrounded — recording state at the moment we lose foreground
    // lets us correlate "session ended abruptly" with "heap was at 240 MB".
    private wireMemoryPressureHandlers(): void {
        const emit = async (event: MemoryPressureEntry['event']) => {
            const heap = (performance as unknown as { memory?: { usedJSHeapSize: number } }).memory;
            const heapMB = heap ? heap.usedJSHeapSize / 1e6 : null;
            let storageMB: number | null = null;
            let persisted = false;
            if (navigator.storage?.estimate) {
                try {
                    const est = await navigator.storage.estimate();
                    storageMB = est.usage != null ? est.usage / 1e6 : null;
                } catch { /* swallow */ }
            }
            if (navigator.storage?.persisted) {
                try { persisted = await navigator.storage.persisted(); } catch { /* swallow */ }
            }
            const entry: MemoryPressureEntry = {
                type: 'memory-pressure',
                timestamp: new Date().toISOString(),
                event,
                heapMB,
                storageMB,
                persisted,
            };
            await this.logger.append(entry);
        };
        this.onVisibilityChange = () => {
            if (document.visibilityState === 'hidden') {
                // Forensics beat FIRST and synchronously — the async emit below
                // can be lost to a background kill; the breadcrumb can't.
                this.forensics?.beat('visibility-hidden');
                emit('visibility-hidden').catch(() => {});
                // Backgrounding is the last safe write window on iOS (the WebView
                // can be jetsam-killed). Capture the note being edited and flush
                // now, bypassing the 5-min idle debounce.
                this.flushOnBackground();
            }
            else if (document.visibilityState === 'visible') {
                this.forensics?.beat('visibility-visible');
                emit('visibility-visible').catch(() => {});
            }
        };
        this.onPageHide = () => {
            this.forensics?.beat('pagehide');
            emit('pagehide').catch(() => {});
            this.flushOnBackground();
        };
        document.addEventListener('visibilitychange', this.onVisibilityChange);
        window.addEventListener('pagehide', this.onPageHide);
    }

    // Emits an eviction-suspected event when cold-start exceeded the
    // 5 s mobile budget. Captures the storage state at the same instant
    // so a low storageUsedMB confirms the cache was actually emptied
    // (vs. a thermal-throttle false positive). Best-effort; never throws.
    private async emitEvictionSuspected(load: import('./types').LoadEntry): Promise<void> {
        let storageUsedMB: number | null = null;
        let storageQuotaMB: number | null = null;
        let persisted: boolean | null = null;
        if (navigator.storage?.estimate) {
            try {
                const est = await navigator.storage.estimate();
                storageUsedMB = est.usage != null ? est.usage / 1e6 : null;
                storageQuotaMB = est.quota != null ? est.quota / 1e6 : null;
            } catch { /* swallow */ }
        }
        if (navigator.storage?.persisted) {
            try { persisted = await navigator.storage.persisted(); } catch { /* swallow */ }
        }
        const entry: EvictionSuspectedEntry = {
            type: 'eviction-suspected',
            timestamp: new Date().toISOString(),
            coldStartMs: load.coldStartMs,
            actualDevice: load.actualDevice,
            dtype: load.dtype,
            storageUsedMB,
            storageQuotaMB,
            persisted,
        };
        await this.logger.append(entry);
    }

    // One-shot `app://local/...` capability probe. Writes a tiny file to the
    // vault, asks the iframe to fetch it via adapter.getResourcePath(), and
    // logs the result. Gates the Phase 3 model-shard streaming pattern: a
    // green probe means we can stream shards through the iframe via a
    // resource URL; a red probe means Phase 3 has to transfer bytes via
    // postMessage. See seek-dataadapter-rearchitecture-plan §Phase 1.
    private async runAppLocalProbe(): Promise<void> {
        const adapter = this.app.vault.adapter;
        // Site the probe inside the plugin's OWN folder (always present, hidden
        // under the config dir) — NOT a visible vault folder. The earlier
        // 'Documents/seek/' literal did adapter.mkdir() and left a stray, visible
        // Documents/ folder in the vault root on every load; manifest.dir is where
        // the running plugin already lives, so there's no mkdir and nothing
        // user-visible. getResourcePath resolves it identically — the capability
        // under test (iframe fetch of an app://local / capacitor:// resource URL)
        // is unchanged.
        const dir = this.manifest.dir;
        if (!dir) return; // no plugin dir → can't site the probe; skip the diagnostic
        const PROBE_PATH = `${dir}/.seek-applocal-probe`;
        const PROBE_BODY = 'seek-probe-v1';

        let url = '';
        try {
            await adapter.write(PROBE_PATH, PROBE_BODY);
            // Obsidian's adapter exposes getResourcePath on both desktop
            // (returns `app://local/...`) and Capacitor mobile (returns
            // `capacitor://localhost/...`). Either is the platform-correct
            // URL the iframe would need to use in Phase 3.
            const ra = adapter as unknown as { getResourcePath?: (p: string) => string };
            url = ra.getResourcePath ? ra.getResourcePath(PROBE_PATH) : '';
        } catch (e) {
            const entry: AppLocalFetchEntry = {
                type: 'app-local-fetch',
                timestamp: new Date().toISOString(),
                result: 'unknown',
                url,
                httpStatus: null,
                bodyMatched: null,
                error: `probe setup failed: ${e}`,
            };
            await this.logger.append(entry);
            return;
        }

        if (!url) {
            const entry: AppLocalFetchEntry = {
                type: 'app-local-fetch',
                timestamp: new Date().toISOString(),
                result: 'unknown',
                url: '',
                httpStatus: null,
                bodyMatched: null,
                error: 'adapter.getResourcePath unavailable',
            };
            await this.logger.append(entry);
            return;
        }

        const fr = await this.embedder.appLocalFetch(url);
        const bodyMatched = fr.body == null ? null : fr.body === PROBE_BODY;
        const entry: AppLocalFetchEntry = {
            type: 'app-local-fetch',
            timestamp: new Date().toISOString(),
            result: fr.ok && bodyMatched === true ? 'ok' : 'blocked',
            url,
            httpStatus: fr.status,
            bodyMatched,
            error: fr.error,
        };
        await this.logger.append(entry);
    }

    // Lightweight storage snapshot. Used after reindex; could be invoked
    // periodically in v1. Keeps the cost low compared to the full platform probe.
    private async emitStorageSnapshot(context: string): Promise<void> {
        let storageUsedMB: number | null = null;
        let storageQuotaMB: number | null = null;
        if (navigator.storage?.estimate) {
            try {
                const est = await navigator.storage.estimate();
                storageUsedMB = est.usage != null ? est.usage / 1e6 : null;
                storageQuotaMB = est.quota != null ? est.quota / 1e6 : null;
            } catch { /* swallow */ }
        }
        const heap = (performance as unknown as { memory?: { usedJSHeapSize: number } }).memory;
        const heapMB = heap ? heap.usedJSHeapSize / 1e6 : null;
        const entry: StorageSnapshotEntry = {
            type: 'storage-snapshot',
            timestamp: new Date().toISOString(),
            context,
            storageUsedMB,
            storageQuotaMB,
            heapMB,
        };
        await this.logger.append(entry);
    }
}

