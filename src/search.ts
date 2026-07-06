// SearchOrchestrator — owns the chunk-vault → embed → store → search loop.
//
// reindexAll() is intentionally the simple "nuke and rebuild" path: drop the
// IndexedDB database, walk all markdown files, chunk → embed → write. Larger
// vaults will want incremental, but the design doc explicitly defers that;
// the user asked for a "total nuke and reset" full reindex as one of the
// three v0 commands.

import type { App } from 'obsidian';
import { TFile } from 'obsidian'; // value import: reindexDelta uses `instanceof TFile`
import type { Chunk, ChunkMeta, ScoredChunk, SearchEntry, IndexCompleteEntry, IndexProgressEntry, ResetEntry, QueryFilters, FilterContext, SeekSettings, MemorySnapshot } from './types';
import { snapshotMemory, memoryDelta, distributionStats } from './types';
import { MarkdownChunker, cyrb53Hex } from './chunker';
import { cleanDenseText } from './dense-clean';
import { extractBaseDocs } from './base-extractor';
import { MultiFieldBM25, DEFAULT_FIELD_BOOSTS, PREFIX_LAST_TOKEN, FUZZY_BY_LENGTH, ANALYZER_VERSION, BM25_COVERAGE_POW } from './bm25';
import { buildSynonymMap, chunkDeclaresAlias, SYNONYM_WEIGHT, type SynonymMap } from './synonyms';
import { rank, cosineScores, DEFAULT_RANKING_CONFIG } from './ranker';
import { browseOrder, recencyDate } from './fusion';
import { IndexStore, nukeDatabase, classifyFileDelta, findOrphanChunkIds, isStoreClosedError, META_SCHEMA_VERSION, type MetaConfig, type FileRecord } from './index-store';
import { LocalEmbedder, EMBEDDING_DIM, LEGACY_ENGLISH_MODEL_ID, MODEL_ID, PLUGIN_VERSION } from './embedder';
import { SeekLogger } from './logger';
import { Forensics } from './forensics';
import { selectIndexBucket } from './iframe-runner';
import { enforceTokenBudget, embedInput, type TokenBudgetResult } from './token-budget';
import { concatPacked, topNIndices, packSignBits } from './binary';
import { selectTopNIndices } from './select';
import { poolCaps, POOL_FLOORS } from './pool';
import { BinaryScorerWorker, binaryCandidatesAsync } from './binary-scorer';
import { quantizeInt8, dequantizeInt8, type QuantVec } from './quant';
import { VecReservoir, denseBgStats, calibratedConfidence, BG_RESERVOIR, MIN_BG_SAMPLE } from './dense-stats';
import { bulkAppend, clearDevice, sidecarDirSignature, shouldReconcileSidecar, staleSidecarFormat, SIDECAR_FORMAT, bm25PathFor, writeBytesAtomic, ensureDir, listSidecarDeviceIds, compactDevice, type CompactResult, type TierBytes } from './sidecar';
import { writeDeviceMeta, readDeviceMeta, metaAccepts, expectationFor, type SidecarMeta, type MetaExpectation } from './sidecar-meta';
import { hydrateFromSidecar, rankAcceptedProducers, probePeerAhead, type ReChunkedNote, type HydrateResult, type HydrateDeps } from './sidecar-sync';
import { pluginIdentity, shouldStampLiveIdentity, identityHealEligibility } from './identity';
import { gzipString, gunzipToString, gzipAvailable } from './gzip';
import { IndexCoordinator } from './index-coordinator';
import { CompositorPacer } from './pacer';
import { isMobilePlatform, residentInt8Enabled } from './platform';
import { parseQuery, compileMatcher, excludedNotePaths } from './query-parser';
import { enumerateNumberPropertyNames } from './prop-types';

// Indexing batches via PER-BUCKET ROLLING BUFFERS (2026-06-03 redesign).
//
// The problem: naive within-file batching ran an effective batch of ~2.2
// (chunks-per-file p50=1 / p95=6) AND padded every batch to its longest
// member's seq bucket. The offline padding sim measured this at 45% efficient
// — over half of all forward compute was padding.
//
// The fix: route each chunk into a buffer keyed by ITS OWN seq bucket
// (selectIndexBucket on the chunk's EXACT token count — WS2.3 replaced the
// chars/4.5 estimate, which under-bucketed dense text and silently truncated
// below the cap), and flush a buffer the instant it reaches its per-bucket
// size (rollingBatchFor), carrying the remainder across files. Because a buffer's chunks share
// a bucket, the dispatch pads to exactly that bucket (zero cross-length waste),
// and because the flush size is FIXED and warmed, the (batch×seq) shape set
// stays inside WARMUP_BATCH_SIZES — the precondition the reverted arbitrary-
// coalescer violated (it packed groups to 7/17/22… → SafeInt overflow). Sim:
// 45%→85% efficiency, −47% forward work, dispatches 1865→~520.
//
// Smoothness: the flush size is the stutter knob, not the pacer. A dispatch is
// non-preemptible once on Metal's queue, so the worst-case stall = the largest
// dispatch's forward time. ModernBERT's global-attention layers are ~O(seq²),
// so a full batch in the 512 bucket is the longest stall by far (measured p95
// 587 ms at a flat batch of 8, 2026-06-04). So we DON'T use a flat size — we
// hold batch×seq roughly constant at a token BUDGET: big buckets flush at a
// small batch (512→3), small buckets at the cap (8). This caps the worst stall
// while leaving the cheap short-bucket dispatches full. Per-chunk compute is
// unchanged (each chunk is one independent sequence); we trade a few extra
// dispatches in the rare long buckets for shorter individual stalls. pace()
// still runs between every flush, so duty cycle stays idle-gated.
//
// ROLLING_BUDGET ≈ target batch×seq per dispatch. 1536 → {512:3, 384:4, 256:6,
// ≤192:8}. Every resulting size is in WARMUP_BATCH_SIZES [1..8]. ROLLING_MAX is
// the warmed ceiling (also mobile's thermal-friendly flush size). Lower the
// budget to cut the p95 further (more dispatches); raise it for throughput.
const ROLLING_BUDGET = 512;
const ROLLING_MAX = 8;
// WASM batch experiment CLOSED (2026-06-11): a flat batch of 4 on the CPU EP
// measured a WASH against this token-budget sizing (3.60 vs 3.83 files/s on
// the same 365-file steady segment, iPhone 15 Pro) — per-call overhead is not
// the bottleneck, and neither is padding (exact-length cut padded tokens 13%
// at identical wall time; see iframe embedBatch). Single-thread CPU forward
// is at a floor only threads (COOP/COEP, Obsidian-core) would move. Reverted
// to one shared sizing: on wasm the budget also caps the synchronous
// main-thread stall per dispatch, which batch=4 made ~4× worse for nothing.
function rollingBatchFor(bucket: number): number {
    return Math.max(1, Math.min(ROLLING_MAX, Math.round(ROLLING_BUDGET / bucket)));
}

// How often to emit a progress entry during indexing (every N committed files).
const PROGRESS_EVERY = 25;
// Time floor for the progress cadence: never let the counter sit silent
// longer than this while files are committing (see the cadence comment at
// the emit site — two healthy iPhone WASM runs were force-quit as "stalled").
const PROGRESS_MAX_SILENCE_MS = 2500;

// ---- Stage-1 candidate-gen caps (Seek Retrieval Relevance & Query §Two-Stage
// ANN → Rerank, the [!done] callout). The union of the three arms feeds the fp32
// exact rerank in stage 2. The caps now SCALE with live corpus size (√N, clamped
// to per-arm floor/ceiling) so a fixed top-200 doesn't cover a shrinking fraction
// of a growing vault — see pool.ts for the full rationale, the cost model, and
// why recency is held flat. `poolCaps(liveN)` is computed per query off the live
// chunk count; at our current ~5k scale it returns exactly the old constants, so
// behaviour is identical until the vault grows past POOL_ANCHOR_N.

// Vault-root files that are machine-generated and would otherwise pollute
// the index with constant-touch recency. The mtime on these files moves
// every time the plugin writes, which means recency=~1.0 → +0.25 lift on
// every fused score, drowning out actual content matches.
//
// This is a v0 hardcoded list rather than a settings string because we
// have zero admin console in v0. Anything that turns into a recurring
// "where did my note go" complaint should be added here.
const EXCLUDED_PATHS = new Set([
    'seek-report.md',
    'seek-report.json',
    'spike-report.md',
]);
const EXCLUDED_PREFIXES = [
    // Future-proofing: if someone runs multiple spike variants, the
    // generated reports tend to share these stems.
    'spike-init',
    'seek-init',
];

// Honor Obsidian's user-configured "Excluded files" (Settings → Files & Links).
// A user who hid a folder from Obsidian's own search/link suggestions expects
// Seek to hide it too — but vault.getMarkdownFiles() ignores that list, so we
// must filter it ourselves. The API isn't in the public typings, so reach the
// runtime defensively: prefer metadataCache.isUserIgnored() (Obsidian's own
// matcher — it handles both the folder-prefix and /regex/ filter forms and
// stays drift-free across versions), falling back to matching the raw
// userIgnoreFilters list on builds that predate that method.
function isUserIgnored(app: App, path: string): boolean {
    const mc = app.metadataCache as unknown as { isUserIgnored?: (p: string) => boolean };
    if (typeof mc.isUserIgnored === 'function') return mc.isUserIgnored(path);
    const getConfig = (app.vault as unknown as { getConfig?: (k: string) => unknown }).getConfig;
    const filters = (typeof getConfig === 'function'
        ? getConfig.call(app.vault, 'userIgnoreFilters')
        : null) as string[] | null;
    if (!Array.isArray(filters)) return false;
    return filters.some(filter => {
        // /pattern/ → regex (Obsidian's own delimiter convention).
        if (filter.length > 1 && filter.startsWith('/') && filter.endsWith('/')) {
            try { return new RegExp(filter.slice(1, -1)).test(path); } catch { return false; }
        }
        // Otherwise a folder/path prefix: match the file itself or anything under it.
        return path === filter || path.startsWith(filter.endsWith('/') ? filter : filter + '/');
    });
}

// The single index-membership predicate, exported so any code that offers a
// value SOURCED from the note (a filter pill, an autocomplete suggestion, …)
// can check whether the note it came from actually reaches the index — a
// pill built from Obsidian's raw metadataCache (which doesn't honor "Excluded
// files") can otherwise promise a result that the matcher will never return.
// SearchOrchestrator.shouldIndex delegates here so there is exactly one
// implementation to keep in sync (see the audit note in suggest.ts).
export function shouldIndexPath(app: App, settings: SeekSettings, path: string): boolean {
    if (EXCLUDED_PATHS.has(path)) return false;
    if (EXCLUDED_PREFIXES.some(p => path.startsWith(p))) return false;
    if (settings.honorIgnoredFolders && isUserIgnored(app, path)) return false;
    return true;
}

// Per-call recency override for search() — the seek:search CLI's
// recencyWeight/recencyHalflife params (main.ts). Either field may be absent
// (only one of the two CLI params given); absent means "use this.settings
// for that field". Deliberately a plain call argument, never written into
// SeekSettings: the settings object is shared by reference with the plugin
// and read by every concurrent search caller, so mutating it for an
// override's duration let a concurrent plain search transparently rank
// against someone else's in-flight override (2026-07-02 review).
export interface RecencyOverride {
    epsilon?: number;
    halfLifeDays?: number;
}

export class SearchOrchestrator {
    private app: App;
    private store: IndexStore;
    private embedder: LocalEmbedder;
    private logger: SeekLogger;
    // Crash forensics (synchronous breadcrumbs). Null in tests / when the
    // plugin couldn't create it — every use is optional-chained.
    private forensics: Forensics | null;
    private chunker = new MarkdownChunker();
    // Live settings reference (the plugin mutates the same object on settings
    // change, so the orchestrator always reads current values). See types.ts.
    private settings: SeekSettings;
    // Shared index-mutation coordination — the write mutex, the in-flight delta
    // gate, the cache-generation counter, and the sidecar location/enablement.
    // Factored out so the indexing and searching halves share exactly this state.
    // See IndexCoordinator.
    private coord: IndexCoordinator;

    // Off-thread stage-1 binary scorer (desktop only; synchronous fallback
    // everywhere). Owns its Worker; disposed on plugin unload. See binary-scorer.ts.
    private binaryWorker: BinaryScorerWorker;

    // Set once, in dispose() (plugin unload / disable). A reindex that is still
    // embedding when the plugin unloads keeps running on the microtask queue AFTER
    // onunload has already closed the store — every subsequent commit would then throw
    // "IndexStore not opened", one error per remaining file (the ~980-error storm).
    // The embed loop checks this at its top + final drain to STOP promptly instead of
    // grinding through the rest of the vault against a dead connection. Sticky: the
    // orchestrator is being torn down, so a re-enable builds a fresh instance.
    private disposed = false;

    // Paths that failed to read during an embed attempt (e.g. an undownloaded
    // iCloud placeholder that throws on every read), mapped to the epoch ms their
    // backoff expires. Without this, a persistently-unreadable file's stale
    // record gets dropped as part of the failed re-embed, so every LATER
    // computeDelta sees prev===undefined and reports it dirty again forever —
    // which wedges reconcileIdentityInPlace in 'drained' permanently (see there).
    // Quarantining it lets computeDelta skip it until the backoff elapses, and a
    // full reindex (reindexAllInner) clears the map so it is always retried then.
    private static readonly UNREADABLE_QUARANTINE_MS = 30 * 60 * 1000; // 30 min
    private unreadableQuarantine = new Map<string, number>();

    private isQuarantined(path: string): boolean {
        const until = this.unreadableQuarantine.get(path);
        return until !== undefined && Date.now() < until;
    }

    // Called wherever an indexable file's content could not be read. Logs once
    // per quarantine spell (not on every retry) so a wedged file doesn't spam.
    private quarantineUnreadable(path: string): void {
        const isNew = !this.unreadableQuarantine.has(path);
        this.unreadableQuarantine.set(path, Date.now() + SearchOrchestrator.UNREADABLE_QUARANTINE_MS);
        if (isNew) console.warn(`[seek] quarantining persistently unreadable file (will retry on backoff / next full reindex): ${path}`);
    }

    constructor(app: App, store: IndexStore, embedder: LocalEmbedder, logger: SeekLogger, settings: SeekSettings, forensics: Forensics | null = null, indexDir: string | null = null) {
        this.app = app;
        this.store = store;
        this.embedder = embedder;
        this.logger = logger;
        this.settings = settings;
        this.forensics = forensics;
        this.coord = new IndexCoordinator(indexDir, settings);
        this.binaryWorker = new BinaryScorerWorker();
    }

    // Release the off-thread scorer's Worker. Called from the plugin's onunload
    // so a reload/disable doesn't leak the worker context. Also signals any in-flight
    // reindex to stop (the embed loop polls `disposed`) so it doesn't keep committing
    // against the store onunload is about to close.
    dispose(): void {
        this.disposed = true;
        this.binaryWorker.dispose();
    }

    // The cache-generation counter (bumped on every index mutation; see
    // IndexCoordinator). Public so the plugin's drift-recovery scheduler can key
    // re-escalation suppression to it: a degraded index re-trips drift on every
    // keystroke, but the generation only advances on a real mutation (delta /
    // reindex / invalidate / hydrate), so equal generation ⇒ "nothing changed since
    // we last escalated, don't re-escalate".
    currentGeneration(): number {
        return this.coord.generation;
    }

    // True while a reindex/delta/cold-build critical section is running under the
    // write mutex. The reconcile poll consults this to avoid healing an index that
    // is still being built (a long reindex outliving the 5-min poll). See main.ts.
    isWriting(): boolean {
        return this.coord.isWriting();
    }

    // Full reindex. Drops the database, walks all markdown, re-embeds everything.
    // Serialized against deltas via the write mutex (a delta mid-nuke would throw).
    async reindexAll(onProgress?: (msg: string) => void): Promise<IndexCompleteEntry> {
        const result = await this.coord.runExclusive(() => this.reindexAllInner(onProgress));
        // Off the mutex: re-warm reads the committed index; a search arriving
        // first just does the same build itself (see warmCaches).
        void this.warmCaches('full-reindex');
        return result;
    }

    private async reindexAllInner(onProgress?: (msg: string) => void): Promise<IndexCompleteEntry> {
        const overallStart = performance.now();
        const memBefore = await snapshotMemory();

        // A full reindex re-attempts every file regardless of quarantine — clear
        // the backoff map so a file that has since become readable (e.g. an
        // iCloud placeholder that finished downloading) isn't skipped here too.
        this.unreadableQuarantine.clear();

        // Reset. Close our long-lived connection first so deleteDatabase
        // isn't blocked by us, then re-open the (fresh, empty) DB.
        const resetStart = performance.now();
        this.store.close();
        const pre = await nukeDatabase(this.store.dbName);   // per-vault DB — never another vault's
        await this.store.open();   // scope-less: reuses the name from onload's open(appId)
        await this.store.setMeta({
            embeddingDim: EMBEDDING_DIM,
            lastIndexedAt: null,
            schemaVersion: META_SCHEMA_VERSION,
            modelId: this.embedder.modelId,
        });
        const resetEntry: ResetEntry = {
            type: 'reset',
            timestamp: new Date().toISOString(),
            droppedDatabase: this.store.dbName,
            chunksDeleted: pre.chunks,
            vectorsDeleted: pre.embeddings,
            durationMs: parseFloat((performance.now() - resetStart).toFixed(2)),
            pass: true,
            checks: [
                `✅ dropped database "${this.store.dbName}"`,
                `ℹ️ removed ${pre.chunks} chunks, ${pre.embeddings} vectors, ${pre.files} file records`,
            ],
        };
        await this.logger.append(resetEntry);

        // Scan. Filter out (a) Seek's own diagnostic outputs / machine-generated
        // chatter (see EXCLUDED_PATHS) and (b) anything the user has hidden via
        // Obsidian's "Excluded files" setting (see isUserIgnored).
        const allFiles = this.indexableFiles();
        const skipped: string[] = [];
        const files = allFiles.filter(f => {
            const include = this.shouldIndex(f.path);
            if (!include) skipped.push(f.path);
            return include;
        });
        if (skipped.length > 0) {
            /* intentionally empty: skipped paths are collected above for the
               filter's side effect; no per-run action is taken on them here */
        }
        // Progressive ordering: index most-recently-modified files first. Since
        // search rebuilds BM25 in-memory and dense reads commit per file, the
        // index is queryable as it fills — recency-first means the notes a user
        // is most likely to search are searchable soonest, and on a phone that
        // may never finish a full pass, recency decides COVERAGE, not just order.
        files.sort((a, b) => b.stat.mtime - a.stat.mtime);
        await this.emitProgress('scan', 0, files.length, 0, performance.now() - overallStart);

        const result = await this.embedAndCommitFiles(files, 'full', onProgress, overallStart, memBefore);

        // Bump dataGeneration on COMPLETION. main.ts invalidates once BEFORE the
        // reindex, but a search firing during the rebuild then caches the frame /
        // binary index at that generation off the PARTIALLY-filled store; nothing
        // bumped again at the end, so a post-reindex search kept serving that
        // frozen partial frame (the frame + binary caches are generation-keyed
        // only — no chunk-count belt-and-braces like BM25 has). Invalidating here
        // forces the next search to rebuild against the complete index.
        this.invalidateBm25Cache();
        return result;
    }

    // The shared embed + commit engine behind both reindexAll (whole vault, after
    // nuke) and reindexDelta (just the changed files, after their stale chunks are
    // dropped). Per-bucket rolling-buffer batching, atomic per-file commit, the
    // ORT-Web WebGPU overflow recycle+retry, and the throughput rollups all live
    // here. Mode-dependent behavior is minimal: the post-index WebGPU pool reclaim
    // runs for 'full' only (a small delta never grows the pool to the high-water
    // mark, and a 2–4 s teardown per blur-flush would be absurd), and `mode` tags
    // the log entry. `files` is expected pre-sorted recency-first by the caller so
    // the index stays queryable as it fills.
    private async embedAndCommitFiles(
        files: TFile[],
        mode: 'full' | 'incremental',
        onProgress: ((msg: string) => void) | undefined,
        overallStart: number,
        memBefore: MemorySnapshot,
        // Per-burst budget — applied ONLY in 'incremental' mode (a full reindex is
        // always unbounded). The per-burst FILE cap is enforced by the caller
        // (reindexDelta slices its dirty list), so the engine only needs the
        // within-burst aborts: budgetMs = wall-clock ceiling for one pathological
        // huge note; shouldContinue = live abort (returns false once the app is
        // hidden / the user resumed searching). Both optional; an empty object =
        // unbounded (the existing behavior for desktop + reindexAll).
        // addsSink (Seek scaling A1): when present, commitFile pushes each
        // ACTUALLY-committed chunk's {chunk, q, bin} into it — the add half of the
        // change-set reindexDelta's incremental cache path consumes. Driven off
        // real commits (not the pre-embed file list), so a mid-burst abort yields
        // exactly the rows that landed in IDB. Absent for a full reindex (it
        // rebuilds caches from scratch).
        // storeWasEmpty: the caller's PASS-START emptiness snapshot (reindexDelta
        // captures it before phase-1 + hydration). The engine must NOT re-derive it
        // here: by the time it runs, carry-over + sidecar dedup may have already
        // committed chunks, so a fresh count() would read non-empty on a cold build
        // that hydrated some files and mis-decide sawWholeCorpus. One snapshot, one
        // source. Omitted (reindexAll) = false, harmless since mode==='full' dominates.
        budget: { budgetMs?: number; shouldContinue?: () => boolean; addsSink?: DeltaAdd[]; storeWasEmpty?: boolean } = {},
    ): Promise<IndexCompleteEntry> {
        // Per-bucket rolling-buffer embed. Each chunk lands in the buffer for
        // its own seq bucket; a buffer flushes as one warmed per-bucket-sized
        // dispatch the instant it fills, carrying the remainder across files.
        // Files commit atomically once their last chunk's vector lands. See the
        // ROLLING_BUDGET comment above for the why (45%→85% padding efficiency,
        // overflow-safe warmed shapes, budgeted per-bucket flush to cap stalls).
        let totalChunks = 0;
        let totalVectors = 0;
        let chunkMs = 0;
        let embedMs = 0;
        let commitMs = 0;
        let filesSkippedError = 0;
        let embedRecycles = 0;
        let filesCommitted = 0;
        // The paths whose file-record was ACTUALLY written (commitFile succeeded) — NOT
        // the same as the prefix of started files: an empty/below-min-chunk note or a
        // mid-list skip-error commits nothing yet still advances processedFiles, so a
        // count-based prefix over-reports. The catch-up drain shrinks its remainder by
        // exactly this set, so an un-committed file stays dirty and is retried (never
        // dropped from the work-list to spin the outer sweep). See reindexDelta.
        const committedFilePaths: string[] = [];
        let processedFiles = 0;   // files we STARTED (for the incremental burst budget + filesDeferred)
        let lastProgress = 0;
        let lastProgressAt = performance.now();
        let tokenBudgetSplits = 0;
        let tokenBudgetOverBudget = 0;
        // Embed-failure quarantine accounting (issue #4): files committed WITH a
        // failure marker (some chunks missing) and the total chunks they lost.
        // `quarantined` remembers each marker written this pass so the pass-end
        // mass-failure gate can UNWIND them: solo-retry evidence can't tell a
        // poisoned chunk from a wedged environment (device-lost storm, app
        // backgrounded mid-pass) — both retries share the same environment. The
        // discriminator is volume: deterministic content failures are rare by
        // nature (issue #4 = one file), an environmental storm hits every
        // in-flight file at once.
        let filesQuarantined = 0;
        let chunksFailedEmbed = 0;
        const quarantined: Array<{ path: string; kept: number }> = [];
        // Check lines minted before the `checks` array exists (the mass-unwind
        // runs right after the bucket drain); spliced into `checks` at build.
        const checksExtra: string[] = [];
        // Fix A (cold first-build identity + background): whether the store was EMPTY
        // before this pass committed anything, from the caller's pass-start snapshot
        // (see the `budget.storeWasEmpty` comment). A truly-empty store means every
        // chunk this pass writes is built by the CURRENT identity AND this single
        // incremental pass sees the WHOLE corpus — so it is functionally a full build.
        // `sawWholeCorpus` below lets it stamp the live identity and compute the
        // dense-cosine background, instead of copying prevMeta's absent fields as
        // `undefined` (the identity gate reads that as stale and "heals" with a spurious
        // full reindex). A NON-empty store with no identity is a LEGACY index (older
        // chunker/model): it must stay stale so the gate rebuilds it, so we gate
        // strictly on emptiness, never on "prevMeta lacks identity".
        const storeWasEmpty = budget.storeWasEmpty ?? false;
        // Corpus dense-cosine background accumulators (dense-stats.ts). Only a
        // FULL pass sees every vector, so only it produces stats; an incremental
        // pass leaves these at zero and carries the prior values forward at
        // setMeta below. bgSum is the running Σ vᵢ (→ exact closed-form μ);
        // reservoir is a uniform sample of vectors for the σ estimate.
        const bgSum = new Float64Array(EMBEDDING_DIM);
        let bgN = 0;
        const reservoir = new VecReservoir(BG_RESERVOIR);
        // Sidecar accumulator: derived (int8 + sign-bit) tiers for every committed
        // chunk, flushed in ONE bulkAppend after the embed loop (per-file appends
        // would be O(n²) read-concat-write on a growing shard). A FULL reindex
        // first drops this device's own sidecar files (clearDevice) so the run
        // REPLACES them rather than doubling; an INCREMENTAL delta legitimately
        // appends without clearing, so superseded/deleted records accumulate in
        // this device's jsonl/shards and are reclaimed only by the next full
        // reindex (there is no compaction). The growth is bounded and read-safe:
        // hydrate re-chunks the live vault and intersects ids (sidecar-sync), so
        // stale records never resolve — they only cost disk until the next full pass.
        const sidecarPending: Array<{ id: string; tiers: TierBytes; mtime: number }> = [];
        if (this.coord.sidecarOn() && mode === 'full') {
            try {
                await clearDevice(this.app.vault.adapter, this.coord.dir!, this.logger.deviceId);
            } catch (e) {
                await this.logger.appendError('sidecar-clear', e);
            }
            // Reap OTHER devices' dead-identity sidecar files (identity ≠ this build).
            // Provably useless — the same metaAccepts gate that refuses them for
            // hydration means no current-version device can use them — so this is
            // sync-safe; an un-updated device simply re-creates its own files until it
            // updates (self-resolving). Stops fleet-wide version drift from piling up.
            // Best-effort: never fail the reindex on a reap error.
            try {
                await this.reapDeadIdentitySidecars();
            } catch (e) {
                await this.logger.appendError('sidecar-reap', e);
            }
        }
        // Forensic dispatch accounting. iOS exposes no heap numbers, so the
        // memory-ceiling hypothesis is tested by CORRELATION instead: every
        // breadcrumb carries cumulative dispatches + padded tokens (batch ×
        // seq-bucket — the real GPU working-set driver, padding included). If
        // repeated deaths cluster at similar cumulative volume, that's the
        // ceiling; deaths at random volume but always hidden are background
        // kills.
        let fDispatches = 0;
        let fPaddedTokens = 0;
        this.forensics?.beat('index-start', { mode, filesTotal: files.length });

        const perFileWallMs: number[] = [];
        const chunksPerFile: number[] = [];
        const embedBatchLatencyMs: number[] = [];

        // A file in flight: its chunks scatter across bucket buffers and resolve
        // as those buffers flush. `remaining` counts unresolved chunks; the file
        // commits (or is skipped, if any chunk errored) the moment it hits 0.
        interface FileState {
            file: (typeof files)[number];
            chunks: Chunk[];
            vectors: (Float32Array | null)[];
            remaining: number;
            hadError: boolean;
            fileStart: number;
            // mtime snapshotted at READ time (not commit time). The file commits
            // many ms later, after its chunks round-trip the embedder; reading
            // file.stat.mtime at commit would record whatever the file's mtime is
            // THEN, which — if the user edited the file mid-index — is newer than
            // the content we actually embedded. computeDelta compares stored mtime
            // to live mtime, so an over-stamped record makes the next delta think
            // the edit is already indexed and silently skips it. Pin the value
            // observed when we read the bytes. (TOCTOU, search.ts read↔commit.)
            mtimeMs: number;
            // cyrb53 of the bytes we read — stored in the file-record so the next
            // computeDelta can tell a real edit from a mtime-only re-stamp.
            contentHash: string;
        }
        // A chunk waiting in a bucket buffer: where to write its vector back.
        // tokens = the chunk's exact token count (from enforceTokenBudget) —
        // on wasm the dispatch pads to the batch max, so the honest
        // paddedTokens forensics counter needs the real lengths.
        interface Pending { fs: FileState; slot: number; input: string; tokens: number; }

        // bucket → chunks awaiting a flush. Created lazily on first use.
        const buffers = new Map<number, Pending[]>();

        // One pacer for the whole run so its idle-slice budget carries across
        // dispatches: consecutive fast flushes can share a single idle window
        // instead of each paying a full yield (pacer.ts / PR #1).
        const pacer = new CompositorPacer();

        // Forensics suffix for embed-failure log contexts: enough to separate a
        // content-shaped failure (which token counts?) from an environment one
        // (which device/glue?) straight from a report — issue #4's log had
        // neither and the triage stalled on it. Counts only, never chunk text
        // (reports are shared). Goes in the CONTEXT, not the message: appendError
        // dedups on message, so a varying suffix here can't fragment the dedup key.
        const dispatchInfo = (inputs: string[], bucket: number, tokens?: number[]): string => {
            const glue = this.embedder.glue;
            return ` n=${inputs.length} bucket=${bucket} dev=${this.embedder.device}${glue ? `/${glue}` : ''}`
                + (tokens && tokens.length === inputs.length ? ` tok=[${tokens.join(',')}]` : '')
                + ` chars=[${inputs.map(t => t.length).join(',')}]`;
        };

        // Embed one warmed batch (≤ ROLLING_MAX inputs that all share a seq
        // bucket), pace after the dispatch, and recover from the ORT-Web WebGPU
        // SafeInt overflow via recycle+retry. The session poisons itself once
        // its int32 buffer accounting overflows (~2200 granite chunks, 2026-06-03
        // diagnosis); a failed dispatch is the signal to recycle (fresh device
        // resets the counter) and retry once. A second failure throws to the
        // caller. Mutates embedRecycles + embedBatchLatencyMs.
        const embedOneBatch = async (inputs: string[], bucket: number, label: string, tokens?: number[]): Promise<Float32Array[]> => {
            // Breadcrumb BEFORE the dispatch (synchronous): if this dispatch
            // kills the process, the ring's last entry says exactly which
            // batch shape died and at what cumulative volume.
            fDispatches++;
            // wasm pads to the batch max (exact-length mode, iframe embedBatch),
            // not the bucket — count what the forward pass actually sees.
            fPaddedTokens += (this.embedder.device === 'wasm' && tokens?.length === inputs.length)
                ? Math.max(...tokens) * inputs.length
                : bucket * inputs.length;
            this.forensics?.beat('index-flush', {
                bucket, n: inputs.length, dispatches: fDispatches,
                paddedTokens: fPaddedTokens, filesCommitted, chunks: totalChunks,
            });
            let result;
            try {
                result = await this.embedder.embedBatch(inputs, bucket);
            } catch (e) {
                // Intentional teardown (onunload → embedder.teardown → dispose)
                // rejects the in-flight RPC tagged 'DISPOSED'. Recycling on that
                // would rebuild a fresh iframe + reload ~250 MB into an already-
                // unloaded plugin (zombie iframe; on dev hot-reload it stacks every
                // cycle). Unwind instead. Recoverable errors (SafeInt overflow,
                // 'TIMEOUT', device-lost) fall through to recycle+retry below.
                if ((e as { code?: string } | null)?.code === 'DISPOSED') throw e;
                await this.logger.appendError(`embedBatch-recycle:${label}${dispatchInfo(inputs, bucket, tokens)}`, e);
                await this.embedder.recycle();
                embedRecycles++;
                result = await this.embedder.embedBatch(inputs, bucket);
            }
            embedBatchLatencyMs.push(result.iframeLatencyMs);
            // Pace against compositor pressure between dispatches — the rIC yield
            // keeps duty cycle capped (see "Seek System Bog-Down Diagnosis.md"
            // §PR #1). Degrades to setTimeout(0) on iOS (no rIC).
            await pacer.pace();
            return result.vectors;
        };

        // Atomic per-file commit: chunks + vectors + file-record in close
        // succession. If the plugin dies mid-commit a few chunks may persist
        // without their file-record — a minor inconsistency the next reindex
        // repairs. putBatch asserts chunks.length === vectors.length, so a
        // mis-counted distribution throws here rather than corrupting the index.
        //
        // Embed-failure quarantine (issue #4): a file with hadError commits the
        // chunks that DID embed, and its file-record carries the failure marker
        // (embedFailedChunks + the plugin version — see FileRecord). The old
        // behavior (skip the whole file, no record) left the file dirty for every
        // computeDelta, so a DETERMINISTICALLY-failing chunk re-ran the full
        // recycle cascade (two iframe teardowns + model reloads) on every
        // reconcile poll and catch-up drain, and wedged reconcileIdentityInPlace
        // in 'drained' so the identity never stamped. The marker pins the file
        // 'clean' until an edit / release bump / full reindex / peer hydrate;
        // the missing chunk is invisible-not-wrong (content-addressed ids), and
        // the healthy chunks stay searchable instead of vanishing with it.
        const commitFile = async (fs: FileState): Promise<void> => {
            const commitStart = performance.now();
            const failed = fs.hadError ? fs.vectors.reduce((n, v) => n + (v === null ? 1 : 0), 0) : 0;
            const chunks = failed === 0 ? fs.chunks : fs.chunks.filter((_, i) => fs.vectors[i] !== null);
            // Derive both tiers ONCE from the fp32 vectors: int8 rerank (QuantVec)
            // + sign-bit candidate (packed from TRUE fp32 — the fidelity invariant
            // putBatch documents). Feed IDB via putBatchQuantized so the bytes IDB
            // holds and the bytes the sidecar persists are bit-identical (no double
            // quantization, no fp32→int8 drift between the two stores).
            const fp32 = failed === 0 ? fs.vectors as Float32Array[]
                : fs.vectors.filter((v): v is Float32Array => v !== null);
            // Feed the corpus background accumulators from the committed (known-
            // good) fp32 — the index holds exactly these vectors. Cost: 384 adds
            // + a bounded reservoir insert per vector. Only consumed on a FULL
            // pass (see setMeta below); harmless to accumulate on a delta.
            for (const v of fp32) {
                bgN++;
                for (let d = 0; d < v.length; d++) bgSum[d] += v[d];
                reservoir.add(v);
            }
            const derived = fp32.map(v => ({ q: quantizeInt8(v), bin: packSignBits(v) }));
            await this.store.putBatchQuantized(chunks, derived);
            // Surface the committed rows for the incremental cache path (A1). Done
            // AFTER the IDB write so the sink only ever holds rows that truly landed.
            if (budget.addsSink) pushDeltaAdds(budget.addsSink, chunks, derived);
            if (this.coord.sidecarOn()) {
                for (let i = 0; i < chunks.length; i++) {
                    sidecarPending.push({
                        id: chunks[i].chunk_id,
                        tiers: { q: derived[i].q.q, s: derived[i].q.s, sign: derived[i].bin },
                        mtime: fs.mtimeMs,
                    });
                }
            }
            await this.store.putFileRecord({
                note_path: fs.file.path,
                // Read-time snapshot, NOT fs.file.stat.mtime (which may have
                // advanced if the file was edited during this index pass — see
                // FileState.mtimeMs). Recording the content's true mtime keeps
                // computeDelta able to detect a mid-index edit on the next pass.
                mtimeMs: fs.mtimeMs,
                chunk_ids: chunks.map(c => c.chunk_id),
                contentHash: fs.contentHash,
                ...(failed > 0 ? { embedFailedChunks: failed, embedFailPluginVersion: PLUGIN_VERSION } : {}),
            });
            commitMs += performance.now() - commitStart;
            totalChunks += chunks.length;
            totalVectors += chunks.length;
            perFileWallMs.push(performance.now() - fs.fileStart);
        };

        // Write one chunk's embedding back into its file; commit/skip the file
        // when its last chunk lands. A null vector marks an embed failure → the
        // whole file is skipped (matches the old per-file skip precision).
        const resolveChunk = async (p: Pending, vec: Float32Array | null): Promise<void> => {
            p.fs.vectors[p.slot] = vec;
            if (vec === null) p.fs.hadError = true;
            if (--p.fs.remaining > 0) return;
            const failed = p.fs.hadError ? p.fs.vectors.reduce((n, v) => n + (v === null ? 1 : 0), 0) : 0;
            if (p.fs.hadError) {
                // Quarantine, don't skip (issue #4) — commitFile writes the
                // partial chunks + the failure-marker record; see its comment.
                // Counts ride in the CONTEXT so the constant message keeps
                // deduping across files during a mass-failure storm.
                await this.logger.appendError(`indexFile-incomplete:${p.fs.file.path} failed=${failed}/${p.fs.chunks.length}`,
                    new Error('one or more chunks failed to embed — quarantined (retries on edit / new release / full reindex)'));
            }
            try {
                await commitFile(p.fs);
                // Accounting only AFTER the commit landed: a failed commit takes
                // the catch's single filesSkippedError++, so no file ever counts
                // twice, and filesQuarantined counts only records actually written
                // (its documented meaning, and the mass-unwind below relies on it).
                if (p.fs.hadError) {
                    // filesSkippedError still counts a quarantined file so the
                    // index-complete pass gate (>2% skip = fail) keeps its
                    // meaning: content IS missing.
                    filesSkippedError++;
                    filesQuarantined++;
                    chunksFailedEmbed += failed;
                    quarantined.push({ path: p.fs.file.path, kept: p.fs.chunks.length - failed });
                } else {
                    filesCommitted++;
                }
                // Real progress for the catch-up drain — quarantined files too:
                // their record is written, so they are non-dirty by the drain's
                // own criterion and must not be re-fed to the embed loop.
                committedFilePaths.push(p.fs.file.path);
            } catch (ce) {
                // A closed store (onunload, or an onversionchange from another
                // instance deleting the DB mid-reindex) makes EVERY subsequent commit
                // throw this same error. Catching it per-file logged it ~once per
                // remaining file — the ~980-error storm. Rethrow so the whole pass
                // aborts after a single error; the caller logs it once and the next
                // reindex repairs the partial. An ordinary single-file commit failure
                // still skips just that file, as before.
                if (isStoreClosedError(ce)) throw ce;
                filesSkippedError++;
                await this.logger.appendError(`commitFile:${p.fs.file.path}`, ce);
            }
        };

        // Flush up to rollingBatchFor(bucket) chunks from a bucket as one dispatch.
        // On dispatch failure (after embedOneBatch's recycle+retry rethrows),
        // isolate each chunk with a solo embed so one genuinely-bad chunk skips
        // only its own file instead of poisoning the whole batch's files.
        const flushBucket = async (bucket: number): Promise<void> => {
            const buf = buffers.get(bucket);
            if (!buf || buf.length === 0) return;
            const batch = buf.splice(0, rollingBatchFor(bucket));
            const embedStart = performance.now();
            let vectors: Float32Array[] | null = null;
            try {
                vectors = await embedOneBatch(batch.map(p => p.input), bucket, `b${bucket}`, batch.map(p => p.tokens));
            } catch (e) {
                // Intentional teardown must unwind the whole pass, never reach the
                // solo/quarantine path — a DISPOSED batch's chunks would otherwise
                // all solo-fail DISPOSED too and quarantine healthy files against
                // a store that is about to close. Mirrors embedOneBatch's own
                // DISPOSED contract.
                if ((e as { code?: string } | null)?.code === 'DISPOSED') throw e;
                await this.logger.appendError(`flushBucket:${bucket}${dispatchInfo(batch.map(p => p.input), bucket, batch.map(p => p.tokens))}`, e);
            }
            if (vectors) {
                for (let i = 0; i < batch.length; i++) await resolveChunk(batch[i], vectors[i]);
            } else {
                for (const p of batch) {
                    try {
                        const v = await embedOneBatch([p.input], bucket, `b${bucket}-solo`, [p.tokens]);
                        await resolveChunk(p, v[0]);
                    } catch (se) {
                        if ((se as { code?: string } | null)?.code === 'DISPOSED') throw se;
                        await this.logger.appendError(`soloChunk:${p.fs.file.path}${dispatchInfo([p.input], bucket, [p.tokens])}`, se);
                        await resolveChunk(p, null);
                    }
                }
            }
            embedMs += performance.now() - embedStart;
        };

        for (const file of files) {
            // Plugin unloaded mid-pass (disable / reload): stop now rather than keep
            // embedding + committing against the store onunload is about to close.
            // Applies to BOTH modes — a full reindex and the incremental cold build.
            // Any meta stamped for the partial index is still CORRECT (Fix A: current
            // identity, merely incomplete), and the next load's catch-up finishes it.
            if (this.disposed) break;
            // Per-burst budget (incremental catch-up only): stop STARTING new files
            // once the wall-clock ceiling is spent or the app went hidden / search
            // resumed (shouldContinue→false). (The file-count cap is upstream — the
            // caller slices the dirty list.) Break at loop-top, BEFORE this file's
            // chunks enter the rolling buffers — so the unconditional bucket-drain
            // below commits exactly the files already started, and un-started files
            // stay dirty (their watermark never advanced) for the drain loop to
            // re-fire. A file already in flight is never interrupted mid-commit.
            if (mode === 'incremental'
                && ((budget.budgetMs !== undefined && performance.now() - overallStart > budget.budgetMs)
                    || (budget.shouldContinue !== undefined && !budget.shouldContinue()))) {
                break;
            }
            processedFiles++;
            const fileStart = performance.now();
            // Snapshot mtime BEFORE the read so the committed file-record reflects
            // the version we actually index, even if the file is edited mid-pass
            // (TOCTOU between this read and the much-later commit).
            const mtimeMs = file.stat.mtime;
            let fileChunks: Chunk[];
            let contentHash = '';
            let content: string;
            try {
                content = await this.app.vault.cachedRead(file);
            } catch (e) {
                // Read error (e.g. an undownloaded iCloud placeholder that throws
                // on every attempt) — skip this file (it never entered a buffer).
                // Quarantine it so the NEXT computeDelta doesn't immediately
                // re-report it dirty forever (its record was already dropped by
                // the caller before this pass ran) — see the quarantine field
                // comment for why that would otherwise wedge reconcileIdentityInPlace.
                filesSkippedError++;
                await this.logger.appendError(`indexFile:${file.path}`, e);
                this.quarantineUnreadable(file.path);
                continue;
            }
            try {
                contentHash = cyrb53Hex(content);
                const modifiedIso = new Date(mtimeMs).toISOString();
                const chunkStart = performance.now();
                fileChunks = this.chunksFor(content, file.path, modifiedIso);
                chunkMs += performance.now() - chunkStart;
            } catch (e) {
                // Chunk error — skip this file (it never entered a buffer). Not
                // quarantined: this is a content/chunker problem, not a read
                // failure, so there is no reason to suppress future re-attempts.
                filesSkippedError++;
                await this.logger.appendError(`indexFile:${file.path}`, e);
                continue;
            }

            if (fileChunks.length === 0) { chunksPerFile.push(0); continue; } // empty / all below min_chunk_chars

            // WS2.3 token-budget enforcement: re-pack any chunk whose embed
            // input exceeds the 512-token window (counted by the model's own
            // tokenizer — token-counts RPC) and capture the EXACT count of
            // every final input for bucket routing below. The count RPC is
            // tokenizer-only (~ms per file) — folded into chunk time.
            let budgeted: TokenBudgetResult;
            try {
                const tbStart = performance.now();
                budgeted = await enforceTokenBudget(fileChunks, ts => this.embedder.tokenCounts(ts));
                chunkMs += performance.now() - tbStart;
            } catch (e) {
                filesSkippedError++;
                await this.logger.appendError(`tokenBudget:${file.path}`, e);
                continue;
            }
            fileChunks = budgeted.chunks;
            tokenBudgetSplits += budgeted.splits;
            tokenBudgetOverBudget += budgeted.overBudget;
            chunksPerFile.push(fileChunks.length);

            const fs: FileState = {
                file, chunks: fileChunks,
                vectors: new Array<Float32Array | null>(fileChunks.length).fill(null),
                remaining: fileChunks.length, hadError: false, fileStart,
                mtimeMs, contentHash,
            };
            for (let slot = 0; slot < fileChunks.length; slot++) {
                const c = fileChunks[slot];
                const input = embedInput(c);
                // Token-exact routing: the bucket is the smallest warmed rung
                // ≥ the input's REAL token count, so truncation cannot fire
                // (enforceTokenBudget guarantees count ≤ 512 except for the
                // counted-and-logged oversize-title pathology).
                const bucket = selectIndexBucket(budgeted.counts[slot]);
                let buf = buffers.get(bucket);
                if (!buf) { buf = []; buffers.set(bucket, buf); }
                buf.push({ fs, slot, input, tokens: budgeted.counts[slot] });
                if (buf.length >= rollingBatchFor(bucket)) await flushBucket(bucket);
            }

            // Progress is keyed to COMMITTED files (not enqueued): with rolling
            // buffers a file commits only once its rarest-bucket chunk flushes,
            // so committed count is the honest "searchable so far" signal.
            // Cadence is files-OR-time: every PROGRESS_EVERY files, or every
            // PROGRESS_MAX_SILENCE_MS as long as ANY new file committed. The
            // pure file cadence proved a UX trap twice on iPhone WASM
            // (2026-06-11): JSC tier-up makes early files ~8× slow, the
            // counter froze for 30-60 s, and the user force-quit a healthy
            // run both times. Time floor keeps the UI provably alive through
            // slow stretches at negligible cost (the mid-reindex BM25 cache
            // drop below already ran every ~1-7 s on the file cadence).
            const progressOverdue = performance.now() - lastProgressAt >= PROGRESS_MAX_SILENCE_MS;
            if (filesCommitted > lastProgress && (filesCommitted - lastProgress >= PROGRESS_EVERY || progressOverdue)) {
                lastProgress = filesCommitted;
                lastProgressAt = performance.now();
                // Make progressive fill visible: drop the caches so a search
                // mid-reindex rebuilds against the files committed so far. Full
                // reindex reads are intentionally progressive (no currentDelta
                // gate); a delta instead commits atomically and invalidates once
                // at the end, so we don't churn its caches per batch here.
                if (mode === 'full') this.invalidateBm25Cache();
                onProgress?.(`Indexed ${filesCommitted} files · ${totalChunks} chunks`);
                this.forensics?.beat('index-progress', { filesCommitted, filesTotal: files.length, chunks: totalChunks });
                await this.emitProgress('embed', filesCommitted, files.length, totalChunks, performance.now() - overallStart);
            }
        }

        // Drain every partial bucket (each remainder is 1..ROLLING_MAX-1, all
        // warmed). This is where the last chunk of most files lands and commits.
        // Skipped when disposed: the store is closing, so flushing buffered chunks
        // would just throw STORE_NOT_OPENED per bucket; the next reindex repairs.
        if (!this.disposed) for (const bucket of buffers.keys()) {
            while ((buffers.get(bucket)?.length ?? 0) > 0) await flushBucket(bucket);
        }

        // Mass-failure gate: solo-retry evidence can't distinguish a poisoned
        // chunk from a wedged ENVIRONMENT (device-lost storm, app backgrounded
        // mid-pass, model unload race) — both retries share the environment. A
        // per-release quarantine written during such a storm would silently hide
        // every affected file from search until the next release. Volume is the
        // discriminator: genuine content failures are rare (issue #4 = one
        // file), a storm hits every in-flight file. Over the cap, UNWIND the
        // markers — deleteFile drops each quarantined record + its partial
        // chunks, restoring the pre-quarantine behavior for exactly this case:
        // the files stay dirty and self-heal on the next poll once the
        // environment recovers. The cap mirrors the pass gate's skip-rate
        // threshold with an absolute floor so a single bad file in a small
        // vault still quarantines.
        const quarantineCap = Math.max(3, Math.ceil(files.length * 0.02));
        if (filesQuarantined > quarantineCap) {
            const unwoundPaths = new Set(quarantined.map(q => q.path));
            for (const q of quarantined) {
                try {
                    await this.store.deleteFile(q.path);
                    totalChunks -= q.kept;
                    totalVectors -= q.kept;
                } catch (e) {
                    if (isStoreClosedError(e)) throw e;
                    unwoundPaths.delete(q.path);   // record survived — still honestly quarantined
                    await this.logger.appendError(`quarantine-unwind:${q.path}`, e);
                }
            }
            // Unwound files have no record again → dirty by the drain's own
            // criterion → must not be reported as committed progress.
            for (let i = committedFilePaths.length - 1; i >= 0; i--) {
                if (unwoundPaths.has(committedFilePaths[i])) committedFilePaths.splice(i, 1);
            }
            filesQuarantined -= unwoundPaths.size;
            checksExtra.push(`⚠️ mass embed failure (${quarantined.length} files > cap ${quarantineCap}) — environmental, not content: unwound ${unwoundPaths.size} quarantine record(s); files stay dirty and retry on the next pass`);
        }
        onProgress?.(`Indexed ${filesCommitted} files · ${totalChunks} chunks`);
        await this.emitProgress('embed', files.length, files.length, totalChunks, performance.now() - overallStart);

        // Corpus dense-cosine background (dense-stats.ts). A FULL pass saw every
        // vector, so it recomputes from scratch (or clears the stats when the
        // corpus is below MIN_BG_SAMPLE — too small to calibrate). An INCREMENTAL
        // pass saw only the changed files, far too few to estimate a corpus
        // global, so it carries the prior full-reindex values forward unchanged.
        // Computed HERE (before the sidecar flush) so the device meta can carry it
        // to hydrate-only peers.
        const prevMeta = await this.store.getMeta();
        // A FULL pass, or a cold first-build (storeWasEmpty), saw the whole corpus, so
        // it recomputes the background from scratch AND claims the live identity. An
        // ordinary delta saw only its changed files — far too few to move a corpus
        // global — so it carries the prior values forward. (On a mobile cold build
        // each 3-file burst is empty-then-not: the first burst stamps identity but
        // yields a sub-MIN_BG_SAMPLE → null background, which peers hydrate from a
        // desktop sidecar; later bursts carry that forward. See dense-stats.ts.)
        const sawWholeCorpus = shouldStampLiveIdentity(mode, storeWasEmpty);
        const bgStats = sawWholeCorpus ? denseBgStats(bgSum, bgN, reservoir.sample) : null;
        const bgMean = sawWholeCorpus ? bgStats?.mean : prevMeta.bgMean;
        const bgStd = sawWholeCorpus ? bgStats?.std : prevMeta.bgStd;
        // One identity read, shared by the sidecar meta (producer, below) and the
        // local meta commit. A full reindex stamps the live identity; an
        // incremental preserves prevMeta's — only a full rebuild claims a new one.
        const identity = pluginIdentity();

        // Flush the sidecar once: append every committed chunk's tiers to this
        // device's own shards/jsonl and (re)write its meta for the version gate.
        // Best-effort — a sidecar failure must never fail the (already-committed)
        // IDB index; it just means cross-device/eviction durability lags a pass.
        if (this.coord.sidecarOn() && sidecarPending.length > 0) {
            const sidecarStart = performance.now();
            try {
                const adapter = this.app.vault.adapter;
                // Create the sidecar dir up front. The F8 ordering writes meta FIRST,
                // and writeDeviceMeta (→ writeTextAtomic) does not create parents — so
                // on a fresh install the meta write would ENOENT and the catch below
                // would abort the WHOLE flush before bulkAppend (which ensures the dir
                // itself) ever runs, leaving no sidecar at all. ensureDir here makes the
                // dir exist before any write, independent of which write lands first.
                await ensureDir(adapter, this.coord.dir!);
                // F8: write/refresh meta BEFORE appending shard+jsonl. These three
                // writes aren't one transaction; ordering meta first means a crash
                // between them leaves meta-without-data (metaAccepts refuses cleanly;
                // the records re-embed next pass) instead of data-without-meta — a
                // fully-written sidecar the null-meta gate refuses, un-hydratable
                // until a full reindex, defeating the eviction-recovery the sidecar
                // exists for.
                const prior = await readDeviceMeta(adapter, this.coord.dir!, this.logger.deviceId);
                // A SIDECAR_FORMAT bump alone (no chunker/model/dim change) never forces
                // a full reindex — identityMatches deliberately excludes it (identity.ts).
                // An ordinary INCREMENTAL commit landing here after such a bump would
                // otherwise write the CURRENT format into meta below while every untouched
                // note's shard bytes are still in the PRIOR record stride — a lie that
                // later misleads this device's own compactOwnSidecar (and any peer
                // hydrating from it) into decoding stale-stride bytes as corrupt. A full
                // pass already clearDevice()'d above (prior reads null there), so this
                // only fires for the incremental case: wipe first so the meta write below
                // is never untrue relative to what's actually on disk.
                if (staleSidecarFormat(prior, identity.sidecarFormat)) {
                    await clearDevice(adapter, this.coord.dir!, this.logger.deviceId);
                }
                await writeDeviceMeta(adapter, this.coord.dir!, {
                    // Canonical version slice (modelId/revision/chunkerVersion/dim)
                    // from the single identity source — NOT embedder.modelId, which
                    // is '' until a model load. format gates the cross-device protocol.
                    ...expectationFor(identity),
                    format: identity.sidecarFormat,
                    deviceId: this.logger.deviceId,
                    lastFullReindex: mode === 'full' ? new Date().toISOString() : (prior?.lastFullReindex ?? null),
                    bgMean,
                    bgStd,
                });
                await bulkAppend(adapter, this.coord.dir!, this.logger.deviceId, sidecarPending);
            } catch (e) {
                await this.logger.appendError('sidecar-commit', e);
            }
            commitMs += performance.now() - sidecarStart;
        }

        // BM25 doesn't need its own persisted index for v0 — it's rebuilt
        // in-memory at search time from the chunk store. Track the timing
        // as zero here; we'll measure it on the first search.
        const bm25Ms = 0;

        // Final commit: meta marker (bgMean/bgStd computed above, before the
        // sidecar flush, so both stores agree)
        const commitFinalStart = performance.now();
        await this.store.setMeta({
            embeddingDim: EMBEDDING_DIM,
            lastIndexedAt: new Date().toISOString(),
            schemaVersion: META_SCHEMA_VERSION,
            modelId: this.embedder.modelId,
            // Identity: a full reindex — or a cold first-build (sawWholeCorpus) —
            // stamps the live build; an ordinary incremental preserves prevMeta so a
            // delta can't falsely re-stamp a stale index as current (mirrors bgMean /
            // lastFullReindex above). Fix A: the cold build must stamp, else its
            // `undefined` identity loops the gate into a spurious heal.
            chunkerVersion: sawWholeCorpus ? identity.chunkerVersion : prevMeta.chunkerVersion,
            analyzerVersion: sawWholeCorpus ? identity.analyzerVersion : prevMeta.analyzerVersion,
            revision: sawWholeCorpus ? identity.revision : prevMeta.revision,
            bgMean,
            bgStd,
        });
        commitMs += performance.now() - commitFinalStart;

        const memAfter = await snapshotMemory();
        const totalMs = performance.now() - overallStart;
        const memD = memoryDelta(memBefore, memAfter);

        // Throughput rollups. Guard against div-by-zero in the (degenerate)
        // case where reindex completed in <1 ms — happens when the vault is
        // completely empty after the EXCLUDED_PATHS filter.
        const seconds = totalMs / 1000;
        const chunksPerSec = seconds > 0 ? totalChunks / seconds : 0;
        const filesPerSec = seconds > 0 ? files.length / seconds : 0;

        const checks: string[] = [
            `✅ indexed ${files.length} files → ${totalChunks} chunks → ${totalVectors} vectors`,
            `ℹ️ embed: ${embedMs.toFixed(0)} ms, chunk: ${chunkMs.toFixed(0)} ms, commit: ${commitMs.toFixed(0)} ms`,
            `ℹ️ total wall time: ${totalMs.toFixed(0)} ms`,
            `ℹ️ throughput: ${chunksPerSec.toFixed(1)} chunks/s, ${filesPerSec.toFixed(1)} files/s`,
        ];
        // Rolling-buffer effectiveness: dispatches = number of embedBatch
        // forward passes; effective batch = vectors / dispatches. Within-file it
        // sat at ~2.2 (p50=1 chunk/file); per-bucket rolling should approach the
        // budget-weighted mean flush size. The measurement that says it worked.
        const embedDispatches = embedBatchLatencyMs.length;
        const effectiveBatch = embedDispatches > 0 ? totalVectors / embedDispatches : 0;
        checks.push(`ℹ️ embed: ${embedDispatches} dispatches, effective batch ≈ ${effectiveBatch.toFixed(1)} (budget ${ROLLING_BUDGET}, max ${ROLLING_MAX})`);
        if (mode === 'full') {
            checks.push(bgStats
                ? `ℹ️ dense background: μ=${bgStats.mean.toFixed(4)} σ=${bgStats.std.toFixed(4)} (${bgN} vecs, calibration on)`
                : `ℹ️ dense background: ${bgN} vecs < ${MIN_BG_SAMPLE} — calibration off`);
        }
        // A handful of genuinely-malformed files skipping is tolerable; a large
        // fraction skipping is the ORT-overflow cascade (or a new systemic fault)
        // and must NOT report success — pass gates on the skip rate so it fails
        // loudly (2026-06-03: 668/2190 skipped had reported pass:true).
        const skipRate = files.length > 0 ? filesSkippedError / files.length : 0;
        const SKIP_RATE_FAIL = 0.02; // >2% of files skipped ⇒ fail the run
        if (embedRecycles > 0) checks.push(`♻️ recycled embed session ${embedRecycles}× — ORT WebGPU overflow recovery (see embedder.recycle)`);
        if (filesSkippedError > 0) {
            const tag = skipRate > SKIP_RATE_FAIL ? '❌' : '⚠️';
            checks.push(`${tag} ${filesSkippedError}/${files.length} file(s) skipped due to error (${(skipRate * 100).toFixed(1)}%) — see error log`);
        }
        if (filesQuarantined > 0) {
            checks.push(`⚠️ ${filesQuarantined} file(s) quarantined (${chunksFailedEmbed} chunk(s) failed to embed after solo retry) — healthy chunks committed + searchable; retried on edit / new release / full reindex`);
        }
        checks.push(...checksExtra);
        if (totalChunks === 0) checks.push('⚠️ no chunks produced — vault may be empty or all files below min_chunk_chars');
        // WS2.3 invariant surface: splits are normal (every >512-token chunk
        // re-packs); a nonzero overBudget means some input still truncates
        // (unsplittable window-filling title) — warn, don't fail.
        checks.push(`ℹ️ token budget: ${tokenBudgetSplits} chunk(s) re-packed to ≤512 tokens`);
        if (tokenBudgetOverBudget > 0) {
            checks.push(`⚠️ ${tokenBudgetOverBudget} embed input(s) still over the 512-token window (title alone ~fills it) — dense tail truncated for those`);
        }
        if (memD.storageDeltaMB != null) checks.push(`ℹ️ storage delta: +${memD.storageDeltaMB.toFixed(1)} MB on disk (IDB)`);

        // Reclaim the WebGPU buffer-pool high-water-mark. Indexing (plus the
        // batch-8×512 warmup grid) grows ORT-Web's off-heap GPU pool to ~1.4 GB;
        // it never shrinks on its own and would otherwise sit resident for the
        // plugin's whole query-serving life, even though queries only need the
        // batch-1 pool. A full device teardown via recycle() drops it back to the
        // ~193 MB floor (verified by toggle-and-diff, 2026-06-03); the next query
        // lazily rebuilds just the small pool it needs. Reindex is infrequent and
        // already ~3 min, so the ~2–4 s teardown is in the noise. WebGPU-only
        // (WASM has no GPU pool to reclaim); best-effort — a cleanup failure must
        // not fail an otherwise-good index, and recycle() leaves the embedder
        // loaded and usable on success regardless.
        if (mode === 'full' && this.embedder.device === 'webgpu') {
            const recycleStart = performance.now();
            try {
                await this.embedder.recycle();
                checks.push(`🧹 released WebGPU buffer pool post-index (${(performance.now() - recycleStart).toFixed(0)} ms) — reclaims the indexing high-water-mark to the query floor`);
            } catch (e) {
                await this.logger.appendError('post-index-recycle', e);
                checks.push(`⚠️ post-index GPU pool release failed (non-fatal): ${e}`);
            }
        }

        const entry: IndexCompleteEntry = {
            type: 'index-complete',
            timestamp: new Date().toISOString(),
            mode,
            dtype: this.embedder.dtype,
            embeddingDim: EMBEDDING_DIM,
            // processedFiles, not files.length: a budgeted incremental burst may
            // have started fewer than it was handed (the rest are filesDeferred).
            // Identical to files.length on a full reindex (never budget-broken).
            filesIndexed: processedFiles,
            // The files actually committed this pass (record written), so the catch-up
            // drain advances by real progress, not by count-of-started. Distinct from
            // filesIndexed (started) whenever a file is empty/skipped/budget-deferred.
            committedFilePaths,
            chunksIndexed: totalChunks,
            vectorsWritten: totalVectors,
            filesSkippedError,
            filesQuarantined,
            chunksFailedEmbed,
            filesDeferred: files.length - processedFiles,
            embedRecycles,
            tokenBudgetSplits,
            tokenBudgetOverBudget,
            chunkDurationMs: parseFloat(chunkMs.toFixed(2)),
            embedDurationMs: parseFloat(embedMs.toFixed(2)),
            bm25DurationMs: parseFloat(bm25Ms.toFixed(2)),
            commitDurationMs: parseFloat(commitMs.toFixed(2)),
            totalDurationMs: parseFloat(totalMs.toFixed(2)),
            heapDeltaMB: memD.heapDeltaMB,
            storageDeltaMB: memD.storageDeltaMB,
            chunksPerSec: parseFloat(chunksPerSec.toFixed(2)),
            filesPerSec: parseFloat(filesPerSec.toFixed(2)),
            perFileWallMs: distributionStats(perFileWallMs),
            chunksPerFile: distributionStats(chunksPerFile),
            embedBatchLatencyMs: distributionStats(embedBatchLatencyMs),
            pass: totalChunks > 0 && skipRate <= SKIP_RATE_FAIL,
            checks,
        };
        // Completion beat closes the indexing window: a death AFTER this reads
        // as idle/background eviction, not crash-while-indexing.
        this.forensics?.beat('index-complete', {
            mode, filesCommitted, chunks: totalChunks,
            dispatches: fDispatches, paddedTokens: fPaddedTokens,
        });
        await this.logger.append(entry);
        return entry;
    }

    // The single index-membership predicate, shared by full reindex (the scan
    // filter) and every incremental path (computeDelta, reindexDelta). Live
    // events and full reindex MUST agree on what's in the index, or a
    // rename-into-Archive and a full reindex would disagree. Two unconditional
    // exclusions (Seek's own machine output) plus the user-toggleable "honor
    // ignored folders" — when on, Obsidian's "Excluded files" (e.g. Archive) are
    // out-of-index, so moving a note in is a soft-delete.
    private shouldIndex(path: string): boolean {
        return shouldIndexPath(this.app, this.settings, path);
    }

    // The candidate set for every collection site (reindexAll, computeDelta, and
    // the sidecar liveness oracles reChunkLive / collectLiveIds — all of which must
    // agree on the file set or base chunk_ids drift between writer and re-deriver).
    // getMarkdownFiles() is .md-only; we additionally index .base files (Obsidian
    // Bases — saved query/view definitions) via per-view synthetic documents. The
    // watcher in main.ts gates create/rename/delete on the same two extensions.
    private indexableFiles(): TFile[] {
        const md = this.app.vault.getMarkdownFiles();
        if (!this.settings.indexBases) return md;
        const bases = this.app.vault.getFiles().filter(f => f.extension === 'base');
        return bases.length === 0 ? md : [...md, ...bases];
    }

    // Content → chunks for one file, branching by extension. A .base file isn't
    // markdown — it's a YAML view definition — so it goes through extractBaseDocs
    // (one synthetic doc per view) + chunkBase, which builds a base-level chunk
    // plus one per non-generic view (each title-boosted, dense + BM25, the view
    // name in the 3.0x headings field). Every chunk-PRODUCTION site routes through
    // here so the .md/.base split lives in one place — reChunkLive, collectLiveIds,
    // dedupViaSidecar and carryOverHydrate all call this, not chunkContent, so a
    // base chunk's id is identical wherever it is re-derived. `modifiedIso` matches
    // the chunker's `modified` param contract.
    private chunksFor(content: string, path: string, modifiedIso: string | null): Chunk[] {
        if (path.endsWith('.base')) {
            return this.chunker.chunkBase(extractBaseDocs(content, path), path, modifiedIso);
        }
        return this.chunker.chunkContent(content, path, undefined, modifiedIso);
    }

    // Diff the persisted index against the live vault — the authoritative,
    // idempotent catch-up computation used by the startup sweep and the post-serve
    // hook. `dirty` = indexable files whose mtime advanced past the stored record
    // (or were never indexed); `deleted` = previously-indexed paths now gone OR no
    // longer indexable (moved into an ignored folder, or honor-ignored toggled on)
    // — a single "not in the live indexable set" test covers both.
    async computeDelta(): Promise<{ dirty: string[]; deleted: string[] }> {
        const records = await this.store.listFileRecords();
        const stored = new Map<string, FileRecord>();
        for (const r of records) stored.set(r.note_path, r);

        const live = this.indexableFiles().filter(f => this.shouldIndex(f.path));
        const livePaths = new Set(live.map(f => f.path));

        // mtime advanced ≠ edited. An iCloud / Drive sync re-stamps a synced
        // file's mtime without changing a byte — on iOS that fires every couple
        // seconds, and keyed on mtime alone it re-embeds identical content
        // forever (each embed blocks the mobile main thread → 1 fps, and the
        // churn drives the jetsam crash-loop). classifyFileDelta confirms the
        // bytes actually changed via the stored content hash before flagging
        // dirty; we only pay the read+hash ('check-bytes') for files whose mtime
        // moved, and the hash is a sync ~5 µs cyrb53, never the embedder, so it
        // can't itself jank the UI.
        const dirty: string[] = [];
        for (const f of live) {
            // A persistently-unreadable file (quarantineUnreadable) is excluded
            // from dirty entirely while its backoff is live — otherwise its
            // dropped record makes classifyFileDelta report 'dirty' forever (see
            // the quarantine field comment), wedging every computeDelta caller.
            if (this.isQuarantined(f.path)) continue;
            const prev = stored.get(f.path);
            let decision = classifyFileDelta(prev, f.stat.mtime, undefined, PLUGIN_VERSION);
            if (decision === 'check-bytes') {
                try {
                    decision = classifyFileDelta(prev, f.stat.mtime, cyrb53Hex(await this.app.vault.cachedRead(f)), PLUGIN_VERSION);
                } catch {
                    decision = 'dirty';   // unreadable → let the embed path decide
                    this.quarantineUnreadable(f.path); // give it this one attempt, then back off
                }
            }
            if (decision === 'dirty') dirty.push(f.path);
        }
        const deleted: string[] = [];
        for (const path of stored.keys()) {
            if (!livePaths.has(path)) deleted.push(path);
        }
        return { dirty, deleted };
    }

    // ── Heal a version-mismatched index WITHOUT a full re-embed, when it is provably
    // safe. enforceIndexIdentity (main.ts) calls this AFTER the embed-free sidecar
    // peer-hydrate attempt and BEFORE the desktop full-reindex / mobile-wait fallback,
    // so both platforms reach it. It is the fix for the cold-build identity bug
    // (PR #43): a huge vault's FIRST index is built by the incremental path, which
    // never stamped chunkerVersion/revision, so the gate reads a perfectly-current
    // index as stale and the only recovery used to be a ~7-min nuke+re-embed of chunks
    // already byte-identical to what a reindex would produce — visually identical to the
    // bug, and (stamping only at the end) re-run from zero on every interruption.
    //
    //   'stale'   — a GENUINELY old index: meta carries a PRESENT, differing
    //               chunker/model/dim. The caller runs the existing full-reindex / wait.
    //   'stamped' — the index was merely UNSTAMPED but is provably current (same model +
    //               dim, every file content-unchanged): stamp the live identity in place
    //               + recompute the display background, zero re-embed, never "0 files".
    //   'drained' — unstamped + a few files drifted: caught up via the normal resumable
    //               delta. Desktop (model loaded) then falls through to the stamp; mobile
    //               / model-not-loaded defers those embeds, so we report 'drained' —
    //               healed enough to stop the destructive loop, the deferred files stay
    //               invisible (content-addressed ids), never wrong.
    //
    // SAFETY: computeDelta proves the FILES are byte-unchanged, NOT that the stored
    // chunks were produced by the current chunker/model. So the stamp arm is gated on a
    // model+dim match — that rules out a cross-model index (old english-r2 vectors
    // stamped current = wrong dense scores, the one unacceptable outcome). The mismatch
    // that triggered the heal here is, by construction, only the absent chunker/revision
    // (modelId+dim already match, or identityMatches would not have failed on them alone
    // for an otherwise-current index). chunkerVersion/revision can't be verified without
    // re-chunking (the async enforceTokenBudget step we avoid to stay embed-free on
    // mobile), so they are ASSUMED current for a model-matching unstamped index: the
    // bounded worst case is stale chunk BOUNDARIES (same real text, same current model →
    // stale, never wrong), and any later edit re-chunks that file at the live version.
    async reconcileIdentityInPlace(): Promise<'stale' | 'stamped' | 'drained'> {
        // Almost always succeeds — enforceIndexIdentity read the meta moments ago to
        // detect the mismatch. A failure here is a transient mid-teardown read; return
        // 'stale' so the caller's fallback retries. Breadcrumb it: a PERSISTENT fault
        // would otherwise silently cost a reindex on every poll with no trace.
        const meta = await this.store.getMeta().catch(e => {
            console.warn('[seek] reconcileIdentityInPlace: meta unreadable, treating as stale', e);
            return null;
        });
        if (!meta) return 'stale';

        // Gate (identity.ts): only an UNSTAMPED, current-model+dim index is safe to stamp
        // in place. A present chunkerVersion (genuinely old / stamped-but-bumped) or a
        // cross-model/dim index falls through to the existing full-reindex / wait.
        if (identityHealEligibility(meta) === 'stale') return 'stale';

        // Prove the files are byte-unchanged (embed-free AND tokenizer-free; on a stable-
        // mtime vault classifyFileDelta never even reads a file).
        let delta = await this.computeDelta();
        if (delta.dirty.length || delta.deleted.length) {
            // A few files drifted since the index was built. Catch just those up through
            // the normal resumable delta — desktop embeds (only if the model is already
            // loaded; we never force a load here, that is the caller's job), mobile stays
            // embed-free and defers. reindexDelta takes its OWN write mutex, so it is NOT
            // nested under one here.
            const embed = !isMobilePlatform() && this.embedder.loaded;
            await this.reindexDelta(delta.dirty, delta.deleted, { embed });
            delta = await this.computeDelta();
            if (delta.dirty.length || delta.deleted.length) return 'drained'; // deferred / still behind — don't stamp
        }

        // STAMP — one atomic write transaction under the write mutex (serializes against
        // a concurrent flush). Recompute the display-only dense background from the
        // stored int8 vectors (embed-free): the same accumulation a full pass does, just
        // dequantized first; the ~0.1% int8 drift is below the display tolerance and the
        // background is NOT a ranking input (dense-stats.ts). Mirrors the proven in-place
        // stamp at hydrateSidecar.
        await this.coord.runExclusive(async () => {
            const { vecs } = await this.store.listAllEmbeddings();
            const bgSum = new Float64Array(EMBEDDING_DIM);
            let bgN = 0;
            const reservoir = new VecReservoir(BG_RESERVOIR);
            for (const qv of vecs) {
                const v = dequantizeInt8(qv.q, qv.s);
                bgN++;
                for (let d = 0; d < v.length; d++) bgSum[d] += v[d];
                reservoir.add(v);
            }
            const bg = denseBgStats(bgSum, bgN, reservoir.sample);
            const id = pluginIdentity();
            const m = await this.store.getMeta();
            await this.store.setMeta({
                ...m,
                embeddingDim: EMBEDDING_DIM,
                modelId: m.modelId ?? MODEL_ID,
                chunkerVersion: id.chunkerVersion,
                analyzerVersion: id.analyzerVersion,
                revision: id.revision,
                bgMean: bg?.mean ?? m.bgMean,
                bgStd: bg?.std ?? m.bgStd,
            });
        });
        this.bgStatsGen = -1;        // invalidate the cached bg accessor (mirrors hydrateSidecar)
        this.invalidateBm25Cache();  // bump dataGeneration so the next search rebuilds against the stamped meta
        return 'stamped';
    }

    // Incremental index update. Two phases:
    //   1. Structural (no embedder): drop deleted/moved-out paths. Always runs,
    //      so deletes + move-into-ignored take effect even on a cold mobile model.
    //   2. Embed (needs a loaded model): for dirty paths that should be indexed,
    //      drop their stale chunks then re-chunk + re-embed via the shared engine.
    //      Skipped when `opts.embed` is false (cold mobile) — the OLD version stays
    //      searchable and, since the file's mtime is still ahead of its stored
    //      record, the next computeDelta re-finds it once the model is warm.
    // Returns null embedded-entry when the embed phase didn't run. Runs under the
    // write mutex (so it can't overlap a full reindex or another delta) and sets
    // `currentDelta` for its critical section so a concurrent search's frame
    // rebuild waits for full application instead of reading a half-committed delta.
    async reindexDelta(
        dirtyPaths: string[],
        deletedPaths: string[],
        // Per-burst budget for the catch-up drain: maxFiles caps how many dirty
        // files this call deletes+re-embeds (slice below — keeps deferred files
        // searchable); budgetMs/shouldContinue are the within-burst aborts passed to
        // the engine. Omitted = unbounded (desktop, flushDirty, reindexAll).
        // onProgress: surfaced for a bulk delta (a paste/sync mini-reindex) so a
        // multi-minute embed isn't silent; undefined for an ordinary single-note
        // flush. embedAndCommitFiles already calls it in 'incremental' mode.
        opts: { embed: boolean; maxFiles?: number; budgetMs?: number; shouldContinue?: () => boolean; onProgress?: (msg: string) => void },
    ): Promise<{ deletedPaths: number; deletedChunks: number; embedded: IndexCompleteEntry | null; deferredEmbed: number; sidecarHydrated: number; carriedOver: number; committedPaths: string[] }> {
        // Set inside the mutex by applyDelta; read after to gate the re-warm. A
        // successful incremental patch IS the warm, so warmCaches is skipped (it
        // would re-pay the O(N) fit the patch just avoided).
        let appliedIncrementally = false;
        const result = await this.coord.runExclusive(async () => {
            let release!: () => void;
            this.coord.currentDelta = new Promise<void>(r => { release = r; });
            try {
                // Fix A: snapshot emptiness BEFORE phase-1 drops — mirrors
                // embedAndCommitFiles. The common cold build runs an embed pass, which
                // stamps the live identity itself (read back into prevMeta below). This
                // covers the residual path where a cold build commits NOTHING via the
                // engine (every file hydrated from a peer sidecar / carried over), so
                // the meta stamp below is the ONLY one — it must still claim the live
                // identity rather than copy the empty store's `undefined` fields.
                const storeWasEmpty = (await this.store.count()).chunks === 0;
                // F13 carry-over: harvest tiers of chunks about to be removed, keyed
                // by embed text, so a move / no-op re-flush reuses the identical vector
                // instead of re-embedding. Deleted paths lose their chunks in phase 1,
                // so harvest them FIRST; only worth it when this pass will embed.
                const carryOver = new Map<string, { q: QuantVec; sign: Uint8Array }>();
                const wantCarryOver = opts.embed && this.embedder.loaded;
                if (wantCarryOver) await this.harvestCarryOverInto(carryOver, deletedPaths);

                // The change-set the incremental cache path consumes: ids removed
                // (Phase 1 deletes + Phase 2 stale-drops) and chunks committed
                // (filled by commitFile via addsSink). Both are ACTUALLY-applied
                // sets, so they stay consistent with IDB even on a mid-burst abort.
                const removedIds: string[] = [];
                const adds: DeltaAdd[] = [];
                // Phase 1: structural drops (no model).
                let deletedChunks = 0;
                for (const path of deletedPaths) {
                    const ids = await this.store.deleteFile(path);
                    deletedChunks += ids.length;
                    removedIds.push(...ids);
                }

                // Phase 2: embed dirty files (only when we can do it now).
                const indexable = dirtyPaths.filter(p => this.shouldIndex(p));
                let embedded: IndexCompleteEntry | null = null;
                let deferredEmbed = 0;
                let sidecarHydrated = 0;
                let carriedOver = 0;
                // Paths this burst actually committed (now non-dirty). Drives the
                // catch-up drain's advance-by-real-progress (see the computation after
                // the embed block, and drainCatchUp).
                let committedPaths: string[] = [];
                // Model-drift guard: if the loaded model differs from the one
                // that built the stored index (a legacy english-r2 index not
                // yet full-reindexed onto ml97), embedding dirty files
                // NOW would mix incompatible vector spaces in one index. Defer
                // the embed phase instead — old versions stay searchable, and
                // mtime keeps them dirty until the full reindex re-stamps meta.
                // A pre-stamp index (modelId undefined) is english-r2 by
                // construction — same default as main.ts warnOnModelIndexDrift.
                const metaModel = (await this.store.getMeta()).modelId ?? LEGACY_ENGLISH_MODEL_ID;
                const modelDrift = opts.embed
                    && this.embedder.loaded && metaModel !== this.embedder.modelId;
                if (opts.embed && !modelDrift && indexable.length > 0) {
                    const allDirty = indexable
                        .map(p => this.app.vault.getAbstractFileByPath(p))
                        .filter((f): f is TFile => f instanceof TFile);
                    // Per-BURST file cap (maxFiles), applied here rather than inside
                    // the engine, and recency-first: only the files this burst will
                    // actually re-embed get their stale chunks dropped — files
                    // deferred to a later burst keep their OLD searchable chunks
                    // until their turn (the drain loop's computeDelta re-finds them
                    // dirty and slices them into a subsequent burst). Deleting all
                    // dirty files up front would make every not-yet-embedded edit
                    // vanish from search for the whole multi-burst drain. undefined
                    // maxFiles (desktop / flushDirty / reindexAll) = the whole set.
                    allDirty.sort((a, b) => b.stat.mtime - a.stat.mtime);
                    const toEmbed = opts.maxFiles !== undefined ? allDirty.slice(0, opts.maxFiles) : allDirty;
                    // F13: harvest the dirty files' OWN stale tiers too (a same-path
                    // no-op re-flush reuses them; an edit's changed chunks simply miss).
                    if (wantCarryOver) await this.harvestCarryOverInto(carryOver, toEmbed.map(f => f.path));
                    // Drop each file's stale chunks before re-embedding: chunk_ids are
                    // content hashes, so edited content yields new ids and the old
                    // chunks would otherwise linger as orphans.
                    for (const f of toEmbed) removedIds.push(...await this.store.deleteFile(f.path));
                    // F13 carry-over: reuse identical vectors (moves / no-op re-flush)
                    // verbatim before falling back to sidecar dedup and the model.
                    const afterCarry = await this.carryOverHydrate(toEmbed, carryOver);
                    carriedOver = toEmbed.length - afterCarry.length;
                    // Dedup-before-embed: hydrate the files the sidecar already covers
                    // (a peer device embedded this exact content) rather than
                    // re-embedding. Stale chunks were just dropped so the engine sees
                    // each edited file as needing fill.
                    const remaining = await this.dedupViaSidecar(afterCarry, adds);
                    sidecarHydrated = afterCarry.length - remaining.length;
                    remaining.sort((a, b) => b.stat.mtime - a.stat.mtime);
                    if (remaining.length > 0) {
                        const overallStart = performance.now();
                        const memBefore = await snapshotMemory();
                        // maxFiles is already enforced by the slice above; the engine
                        // only needs the wall-clock + hidden aborts (for one huge note
                        // or a mid-burst background).
                        embedded = await this.embedAndCommitFiles(remaining, 'incremental', opts.onProgress, overallStart, memBefore,
                            // Pass the PASS-START emptiness (captured above, before carry-over
                            // + sidecar hydration committed any chunk) so the engine's identity
                            // stamp + background recompute use the same cold-build signal as the
                            // meta stamp below — not a count() taken after hydration.
                            { budgetMs: opts.budgetMs, shouldContinue: opts.shouldContinue, addsSink: adds, storeWasEmpty });
                    }
                    // Paths this burst actually committed (now non-dirty), so the catch-up
                    // drain advances by REAL progress instead of by maxFiles: carry-over
                    // reuse (toEmbed \ afterCarry) + sidecar hydration (afterCarry \
                    // remaining) + the files embedAndCommitFiles ACTUALLY committed
                    // (committedFilePaths — record written). NOT a count-based prefix of
                    // `remaining`: an empty/below-min-chunk note or a mid-list skip-error
                    // advances processedFiles without committing, so a prefix would punch a
                    // hole and drop a still-dirty file from the work-list — re-finding it
                    // dirty every sweep and spinning the outer loop. A budget-deferred file
                    // is likewise absent (stale chunks dropped, not re-embedded), so the
                    // drain retries it next burst (the searchable-during-drain invariant).
                    const afterCarrySet = new Set(afterCarry.map(f => f.path));
                    const remainingSet = new Set(remaining.map(f => f.path));
                    committedPaths = [
                        ...toEmbed.filter(f => !afterCarrySet.has(f.path)).map(f => f.path),   // carried over
                        ...afterCarry.filter(f => !remainingSet.has(f.path)).map(f => f.path), // sidecar-hydrated
                        ...(embedded?.committedFilePaths ?? []),                              // actually embedded + committed
                    ];
                } else {
                    deferredEmbed = indexable.length;
                }

                // Any store mutation must reach the in-memory caches. Prefer the
                // incremental patch (applyDelta mutates BM25 + the frame in place,
                // vacuums, and re-stamps the generation under THIS mutex); fall back
                // to a full invalidate+rebuild when the patch can't safely apply
                // (cold caches, index-shape flip, dim change, drift, or a due
                // compaction). Sidecar-hydrated chunks are in `adds` too, so a dedup
                // delta is incremental. Either way the next search sees a correct cache.
                const mutated = deletedChunks > 0 || deletedPaths.length > 0 || embedded || sidecarHydrated > 0 || carriedOver > 0;
                if (mutated) {
                    // F13 carry-over writes chunks the applyDelta change-set can't
                    // track (they bypass the model + the `adds` sink), so a delta that
                    // carried any vector over can't be patched incrementally — force
                    // the full invalidate+rebuild. The expensive re-embed is still
                    // avoided; only the cheaper in-memory cache fit is re-paid.
                    appliedIncrementally = carriedOver === 0 && await this.applyDelta(adds, removedIds);
                    if (!appliedIncrementally) this.invalidateBm25Cache();
                }
                // Stamp meta ONLY when the corpus actually changed. A no-op delta (an
                // embed:false reconcile pass, or a computeDelta that found nothing)
                // rewriting lastIndexedAt + re-storing identical meta was pure write-
                // amplification — hundreds of redundant LevelDB writes across a churny
                // mobile session, feeding the bloat ratchet. And now that the BM25 stamp
                // gate tolerates a drifted lastIndexedAt (bm25StampMatches), there is
                // nothing to keep in sync on a no-op. Preserve the existing modelId: a
                // delta embeds with the loaded model, but the BULK of the index is
                // whatever the last full reindex wrote — only reindexAll claims a new id.
                if (mutated) {
                    const prevMeta = await this.store.getMeta();
                    // Fix A: a cold first-build (storeWasEmpty) claims the live identity
                    // here. For the common embed path this is a no-op — embedAndCommitFiles
                    // already stamped it, so prevMeta reads it back — but it is the ONLY
                    // stamp for the residual hydrate/carry-only cold build, whose prevMeta
                    // would otherwise be `undefined` and re-trip the gate. A NON-empty
                    // store carries identity forward unchanged: an ordinary delta never
                    // re-stamps, and a legacy index must stay stale for the gate to heal.
                    // reindexDelta is always an incremental (delta) pass, so a cold
                    // first-build (empty store) is its only stamp-live case.
                    const live = shouldStampLiveIdentity('incremental', storeWasEmpty) ? pluginIdentity() : null;
                    await this.store.setMeta({
                        embeddingDim: EMBEDDING_DIM,
                        lastIndexedAt: new Date().toISOString(),
                        schemaVersion: META_SCHEMA_VERSION,
                        // Prefer the existing modelId (embed path set it); fall back to
                        // the live model only when the cold build wrote none (hydrate-only).
                        modelId: prevMeta.modelId ?? live?.modelId,
                        // Carry identity + corpus background forward unchanged on a delta;
                        // a cold build claims live identity (see comment above). Only a
                        // full/cold pass refits the background (see dense-stats.ts).
                        chunkerVersion: live ? live.chunkerVersion : prevMeta.chunkerVersion,
                        analyzerVersion: live ? live.analyzerVersion : prevMeta.analyzerVersion,
                        revision: live ? live.revision : prevMeta.revision,
                        bgMean: prevMeta.bgMean,
                        bgStd: prevMeta.bgStd,
                    });
                }
                // deferredEmbed: the WHOLE embed phase was skipped (cold model / model
                //   drift) → indexable.length. The drain loop reads this as its
                //   no-forward-progress / drift signal. (Budget deferral is detected
                //   instead by the shrinking dirty set on the next computeDelta, so it
                //   needs no separate return field.)
                return { deletedPaths: deletedPaths.length, deletedChunks, embedded, deferredEmbed, sidecarHydrated, carriedOver, committedPaths };
            } finally {
                release();
                this.coord.currentDelta = null;
            }
        });
        // Re-warm only when we did NOT patch incrementally (the patch is the warm).
        // Fire-and-forget, off the mutex — the moment the flush scheduler already
        // judged quiet enough for embed work, so the rebuild lands here rather than
        // on the next search. A carry-over delta always lands here (it forces the
        // non-incremental path above), so result.carriedOver gates a re-warm too.
        if (!appliedIncrementally
            && (result.deletedChunks > 0 || result.deletedPaths > 0 || result.embedded || result.sidecarHydrated > 0 || result.carriedOver > 0)) {
            void this.warmCaches('delta');
        } else if (appliedIncrementally) {
            // The incremental patch already kept the resident cache warm, so there is
            // no rebuild to do — but historically it also skipped persistBm25, leaving
            // the disk blob stale until a full rebuild. Re-persist (throttled, embed-
            // free) so the next cold start loads a near-fresh blob instead of refitting.
            this.maybePersistResidentBm25();
        }
        return result;
    }

    // Incremental cache maintenance (Seek scaling A1). Mutate the live BM25 index +
    // resident frame in place from a delta's change-set instead of nuking and
    // full-rebuilding — per-edit work drops from O(N) to O(edit size + affected
    // terms). Returns true on a clean patch; false means the caller must fall back
    // to invalidateBm25Cache() + warmCaches (a full rebuild). Correctness NEVER
    // depends on the incremental path: every "can't safely apply" condition (cold
    // caches, index-shape flip, sidecar hydrate, dim change, post-patch drift, due
    // compaction) returns false and degrades to "slow", never "wrong".
    //
    // MUST run inside reindexDelta's runExclusive critical section: it awaits
    // vacuum() and re-stamps the generation atomically, so a concurrent search
    // (which waits on coord.currentDelta in ensureFrame) sees either the old cache
    // or the fully-patched one, never a half-mutated frame.
    private async applyDelta(adds: DeltaAdd[], removedIds: string[]): Promise<boolean> {
        const frame = this.frameCache;
        const bm = this.bm25Cache;
        if (!frame || !bm) return this.deltaFallback('cold caches');
        if (frame.generation !== this.coord.generation
            || this.bm25CacheGeneration !== this.coord.generation) return this.deltaFallback('stale cache generation');
        // An index-shape flip changes the BM25 field set → must refit from scratch.
        if (this.bm25CacheProps !== this.settings.searchableProperties
            || this.bm25CacheHeadings !== (this.settings.headingsField || this.settings.boostedBm25)) {
            return this.deltaFallback('index-shape settings changed');
        }
        // Sidecar-hydrated rows ARE surfaced in `adds` now (dedupViaSidecar →
        // hydrateDeps.putQuantized → pushDeltaAdds), so a dedup delta applies
        // incrementally like any other — no sidecar-specific fallback.
        // Dim guard: a committed vector whose int8 dim differs from the resident
        // block's is a model/dim change mid-stream → rebuild. (Model drift normally
        // defers embeds so `adds` is empty; this guards a partial-stamp index.)
        if (frame.residentInt8 && adds.some(a => a.q.q.length !== frame.embDim)) {
            return this.deltaFallback('embedding dim mismatch');
        }

        // Removes first, then adds: an edit re-commits the SAME content-hash id only
        // after its stale chunk was dropped, so discard-before-add means mini.add()
        // never collides with a live duplicate. Capture each removed row BEFORE
        // bm.remove() drops it from idToIdx, so the frame tombstones the right hole.
        // Wrapped in try/catch for exception safety: a throw mid-patch leaves the
        // in-place mutation half-applied, so degrade to a full rebuild (the caller
        // invalidates the suspect caches on a false return and re-warms from IDB,
        // the source of truth) rather than letting it escape — "slow, never wrong",
        // and never the crash-loop a thrown bm.add() caused on 2026-06-18.
        // The adds that actually land after the duplicate filter (declared out here
        // so the success log reports the true row count, not the pre-filter total).
        let fresh: DeltaAdd[] = [];
        // Did this delta touch an alias-bearing note? If not, the synonym
        // dictionary is provably unchanged and its O(notes) rebuild is skipped at
        // the commit below (chunkDeclaresAlias). Tracked across BOTH removes (an
        // alias deletion/edit drops the old alias-bearing row) and adds, so an
        // alias EDIT — remove-old + add-new — trips it from either side.
        let aliasDictDirty = false;
        try {
            const removeRows: number[] = [];
            for (const id of removedIds) {
                const row = bm.rowOf(id);
                if (row !== undefined) {
                    removeRows.push(row);
                    // Read the row's metadata BEFORE tombstoneFrameRows drops it.
                    if (chunkDeclaresAlias(frame.orderedChunks[row])) aliasDictDirty = true;
                }
                bm.remove(id);
            }
            tombstoneFrameRows(frame, removeRows);
            // Drop adds whose id is already live in the row space (a hydrate-sourced
            // duplicate from an IDB↔cache divergence) or repeated within the batch;
            // see freshDeltaAdds. Runs AFTER the removes so edit re-commits survive.
            // The SAME list feeds bm.add and appendFrameRows → row spaces stay aligned.
            fresh = freshDeltaAdds(adds, id => bm.rowOf(id) !== undefined);
            if (!aliasDictDirty) aliasDictDirty = fresh.some(a => chunkDeclaresAlias(a.chunk));
            for (const a of fresh) bm.add(a.chunk, a.chunk.content ?? '');
            appendFrameRows(frame, fresh);
            // Async: reclaims tombstoned postings so getQueryBound reads exact df (the
            // TM2C2 clip-at-1 legality invariant). Awaited inside the mutex.
            await bm.vacuum();
        } catch (e) {
            // console.* is invisible on mobile (no devtools) — write the NDJSON device
            // log too, so a recurring patch-throw stays field-observable. That exact
            // telemetry channel root-caused the 2026-06-18 meltdown; a silent
            // console.error would now hide any future throw past the L1 filter.
            console.error('[seek] applyDelta threw mid-patch — dropping to full rebuild', e);
            void this.logger.appendError('applyDelta-patch', e).catch(() => {});
            return this.deltaFallback('exception during patch');
        }

        // Drift detector: verify the row-space coupling survived the patch. On ANY
        // mismatch, abandon the (suspect) patch — the caller's invalidate+rebuild
        // makes it "slow, never wrong".
        if (!frameBm25Coherent(frame, bm)) {
            console.error('[seek] applyDelta produced an incoherent frame/BM25 row space');
            return this.deltaFallback('row-space drift');
        }

        // Compaction: too many tombstone holes → rebuild densely (the amortized O(N)
        // renumber). Returning false routes that through invalidate+warmCaches.
        const n = frame.orderedChunks.length;
        if (n > 0 && frame.tombstoneCount / n >= COMPACTION_TOMBSTONE_FRACTION) {
            return this.deltaFallback(`compaction due (${frame.tombstoneCount}/${n} tombstones)`);
        }

        // Commit: bump the generation so other readers re-validate, then re-stamp
        // the patched caches to it so the next search hits them (no rebuild).
        this.coord.bumpGeneration();
        frame.generation = this.coord.generation;
        this.stampBm25Cache(bm.size);
        // Synonym dict derives ONLY from alias-bearing notes (see chunkDeclaresAlias
        // / buildClasses), so refresh it — over LIVE rows only — when the expansion
        // toggle is on AND this delta actually touched an alias. A body-only edit
        // can't change the dictionary, so it skips the O(notes) rebuild. (The
        // df-ceiling guard then rides slightly stale between alias deltas; it's a
        // coarse 5% junk filter, refreshed on the next alias-touching delta, full
        // reindex, or cold lazy build in ensureBm25.)
        if (this.settings.synonymExpansion && aliasDictDirty) {
            const liveChunks = frame.tombstoneCount === 0
                ? frame.orderedChunks
                : frame.orderedChunks.filter((_, i) => frame.validRows[i]);
            this.synonymCache = buildSynonymMap(liveChunks, t => bm.termDocFraction(t));
        }
        // persistBm25 is no longer skipped here (2026-06-20): the caller re-persists
        // this patched index after the mutex via maybePersistResidentBm25 (throttled,
        // embed-free). The old skip assumed a cold stamp would reject the drifted
        // chunkCount anyway — but the tolerant gate now LOADS it, so keeping the disk
        // blob fresh is what lets a cold relaunch skip the all-bodies refit.
        // Report fresh.length (rows that landed), not adds.length — the difference is
        // duplicates the L1 filter absorbed; surface it so the hydrate-divergence
        // signal stays visible in the very telemetry used to diagnose this bug class.
        const filtered = adds.length - fresh.length;
        if (filtered > 0) console.warn(`[seek] applyDelta absorbed ${filtered} already-live/in-batch duplicate add(s) — hydrate/cache divergence`);
        return true;
    }

    // Log why the incremental path declined + signal the caller to full-rebuild.
    // Makes the live smoke observable: every delta logs either an "applyDelta:
    // +x/-y incremental" success or an "applyDelta fallback: <reason>" line.
    private deltaFallback(reason: string): false {
        return false;
    }

    // Drift-detector trip handler for the QUERY path (the applyDelta path returns
    // false to fall back instead). Logs loudly + a user Notice, drops the caches
    // for a full rebuild, and kicks a warm — turning a silent, in-bounds row-space
    // mis-join into a visible "rebuilt from scratch" event.
    private coherenceDriftCount = 0;          // log-only diagnostic counter (trip #N)
    private lastCoherenceWarmAt = -Infinity;  // performance.now() of the last drift re-warm
    // Injected by the plugin (setPersistentDriftHandler). Fired from onCoherenceDrift's
    // re-trip branch — the orchestrator is pull-based (it owns no outbound scheduling),
    // so this mirrors the modal's onSearchActivity/onQueryInFlight injection: the plugin
    // owns the embed-free recovery scheduler (runDriftRecovery) and the indexHealth flag.
    private onPersistentDrift?: () => void;
    setPersistentDriftHandler(fn: () => void): void {
        this.onPersistentDrift = fn;
    }
    private onCoherenceDrift(where: string): void {
        this.coherenceDriftCount++;
        const now = performance.now();
        const { warm } = coherenceDriftDecision(now, this.lastCoherenceWarmAt, COHERENCE_DRIFT_COOLDOWN_MS);
        // Always drop the suspect caches — a mis-coupled frame/BM25 must never serve
        // (the cost is trivial and correctness-critical). Only the heavy re-warm and
        // the user-facing Notice are rate-limited (decision.warm).
        this.invalidateBm25Cache();
        if (!warm) {
            // Re-tripped inside the cooldown ⇒ a PERSISTENT mis-join, not a one-off.
            // Rebuilding again inline would just thrash (this turned one bad delta into
            // the 2026-06-18 mobile meltdown). The cache is invalidated; the next search
            // rebuilds it lazily via the cold path. Throttled log, no toast storm.
            console.error(`[seek] frame/BM25 drift at ${where} re-tripped within ${COHERENCE_DRIFT_COOLDOWN_MS / 1000}s (trip #${this.coherenceDriftCount}) — escalating to embed-free recovery`);
            // Hand off to the plugin's bounded, embed-free recovery ladder (sidecar
            // hydrate → warm → verify → degraded). It self-suppresses re-fires per
            // generation, so firing on every re-trip is cheap. No inline rebuild here.
            this.onPersistentDrift?.();
            return;
        }
        this.lastCoherenceWarmAt = now;
        console.error(`[seek] frame/BM25 row-space drift detected at ${where} — dropping caches for a full rebuild (trip #${this.coherenceDriftCount})`);
        void this.warmCaches('coherence-drift');
    }

    // Rebuild the IDB index from the vault-file sidecar without re-embedding —
    // the iOS-eviction / fresh-device recovery path. Runs under the write mutex
    // (can't overlap a reindex/delta), reproduces ids by re-chunking the live
    // vault, and writes only the intersection that isn't already in IDB. Returns
    // null when the sidecar is off. Idempotent: a warm index hydrates nothing.
    async hydrateSidecar(): Promise<HydrateResult | null> {
        if (!this.coord.sidecarOn()) return null;
        // Was the index empty BEFORE this hydrate? Captured outside the mutex; a race
        // that populates it concurrently only costs the identity stamp below (degrades
        // to the un-stamped path), never correctness.
        const wasEmpty = (await this.store.count()).chunks === 0;
        const result = await this.coord.runExclusive(() =>
            hydrateFromSidecar(this.hydrateDeps(() => this.reChunkLive())),
        );
        this._peerAhead = result.peerAhead; // refresh the "newer index exists" signal per scan
        // Stamp the build identity when hydrating onto a PREVIOUSLY-EMPTY index: every
        // chunk came from a metaAccepts-filtered (current-identity) producer, so the
        // index is provably at the current identity. This lets a hydrate-only device
        // (cold iOS, lastIndexedAt=null) report a current identity on its next boot —
        // without it the boot gate would needlessly nuke+rehydrate, or (no peer that
        // boot) falsely drop into the "wait for desktop" empty state. NOT stamped on a
        // non-empty hydrate: that index may still hold stale orphans, and only a nuke
        // (or the Phase-3 subtractive hydrate) clears them — claiming current identity
        // there would mask them from the gate.
        if (wasEmpty && result && result.hydrated > 0) {
            const id = pluginIdentity();
            const m = await this.store.getMeta();
            await this.store.setMeta({ ...m, modelId: m.modelId ?? MODEL_ID, chunkerVersion: id.chunkerVersion, analyzerVersion: id.analyzerVersion, revision: id.revision });
        }
        // Inherit display calibration from the producer when this device has none
        // of its own (a hydrate-only iOS device that never full-reindexed). Only
        // fill when absent — a local full reindex's stats describe THIS index more
        // faithfully and must win. Invalidate the cached accessor so the next
        // search re-reads.
        if (result && result.bgMean != null && result.bgStd != null) {
            const m = await this.store.getMeta();
            if (m.bgMean == null) {
                await this.store.setMeta({ ...m, bgMean: result.bgMean, bgStd: result.bgStd });
                this.bgStatsGen = -1;
            }
        }
        // A hydrate is a store mutation: drop caches so the next search rebuilds
        // against the restored index (mirrors reindex/delta completion).
        if (result && result.hydrated > 0) {
            this.invalidateBm25Cache();
            void this.warmCaches('hydrate');
            this.warnedStranded = false; // index is populated again — re-arm the empty-net warning
        } else if (result) {
            // Restored nothing — if the index is also empty, search will silently
            // return no results. Surface that (the net-is-gone case).
            await this.warnIfIndexStranded(result);
        }
        return result;
    }

    // Embed-free convergence: nuke this device's IDB, then hydrate it from the synced
    // sidecar. The mobile-safe counterpart to "Full reindex" — it loads the TOKENIZER
    // only (no 250 MB model, no WASM embed, no jetsam), so it rebuilds a clean index
    // in seconds instead of a hot multi-minute on-device embed. Use it when a device
    // has accumulated STALE/orphan chunks it never cleaned up (e.g. it never cleanly
    // re-chunked after a chunker change, so computeDelta — which only re-examines
    // file-watcher-flagged files — never deleted the old chunks). reChunkLive
    // reproduces only the CURRENT chunker's ids, so the orphans are simply not
    // recreated; ~all current chunks come back from the sidecar (vectors + the Phase-3
    // BM25 blob), and the few genuinely-local chunks fall to the normal gentle catch-up.
    //
    // SAFETY: refuses to nuke unless a compatible sidecar producer is actually present
    // — otherwise a not-yet-synced sidecar would leave the index empty. The device's
    // OWN shard is kept (it's a producer too), so its local-only chunks re-hydrate.
    async rebuildFromSidecar(): Promise<HydrateResult | null> {
        if (!this.coord.sidecarOn()) return null;
        const expect = expectationFor();
        const producers = await rankAcceptedProducers(this.app.vault.adapter, this.coord.dir!, expect);
        if (producers.length === 0) {
            // Nothing compatible to rebuild FROM — do NOT nuke. acceptedProducers:0
            // signals the caller to tell the user the sidecar hasn't synced yet.
            return { scanned: 0, needed: 0, hydrated: 0, skippedPartialNotes: 0, refusedProducers: 0, acceptedProducers: 0, peerAhead: false, hydratedNotePaths: [] };
        }
        // 1. Nuke + reopen under the write mutex (mirrors reindexAllInner's reset) so a
        //    delta/search can't race the close→delete→open window. modelId = the
        //    canonical sidecar model (NOT embedder.modelId, which is '' on a cold
        //    mobile embedder) so the rebuilt index's stamp matches the producer's.
        await this.coord.runExclusive(async () => {
            this.store.close();
            await nukeDatabase(this.store.dbName);
            await this.store.open();
            const id = pluginIdentity();
            await this.store.setMeta({ embeddingDim: EMBEDDING_DIM, lastIndexedAt: null, schemaVersion: META_SCHEMA_VERSION, modelId: MODEL_ID, chunkerVersion: id.chunkerVersion, analyzerVersion: id.analyzerVersion, revision: id.revision });
        });
        // 2. Drop every resident cache that referenced the nuked index, then hydrate
        //    (its own write mutex; the now-empty store forces a full reconcile).
        this.invalidateBm25Cache();
        return this.hydrateSidecar();
    }

    // Does ANY other device have a sidecar in the index dir, regardless of its identity?
    // A cheap, embed-free signal (a directory listing — no meta read, no model) that this
    // vault is multi-device. When the local index is version-stale and rebuildFromSidecar
    // found no CURRENT-identity producer, a present peer means a current index is (or will
    // be) on its way — so the UI says "syncing", not "reindex". Self is excluded: our own
    // stale sidecar is not an incoming heal. A long-dead peer's leftover sidecar is a rare,
    // benign false-positive (it only keeps the calm banner up a little longer; reapDead-
    // IdentitySidecars clears it at the next full reindex on some device).
    async peerSidecarPresent(): Promise<boolean> {
        if (!this.coord.sidecarOn()) return false;
        const ids = await listSidecarDeviceIds(this.app.vault.adapter, this.coord.dir!);
        return ids.some(id => id !== this.logger.deviceId);
    }

    // Referential-integrity sweep (Phase 3 steady-state GC): delete every chunk no
    // FILES record references — orphans from an overwritten file record (hydrate /
    // applyDelta swapping a changed note's ids) or a missed delete event. They sit
    // in chunk_meta, so they can surface stale content; this is freshness + space,
    // not just disk. Pure IDB set-arithmetic — NO re-chunk / embed / GPU — so it is
    // mobile-safe; batched under the write mutex with an optional abort so it never
    // blocks an active search for long. Re-syncs caches after any delete (orphans
    // sit in the resident frame until invalidate + warm rebuilds it). Returns the
    // count removed + whether the pass completed (a backgrounding abort resumes on
    // the next poll). Complements the identity cascade: that clears VERSION-bump
    // orphans (nuke + rehydrate), this clears SAME-version churn.
    async sweepOrphanChunks(opts: { shouldContinue?: () => boolean } = {}): Promise<{ removed: number; completed: boolean }> {
        // Snapshot all chunk ids + the referenced set TOGETHER under the write mutex.
        // A per-file commit is two transactions (putBatch THEN putFileRecord), so an
        // UNLOCKED snapshot could catch a just-written chunk before its file record
        // lands and delete the user's NEWEST edit as a false orphan. (Content-addressing
        // makes that self-healing, but it is still a real freshness hit, so close the
        // window.) The only residual — a rare re-add of the SAME content between batches
        // — reuses the same id and is itself self-healing.
        const orphans = await this.coord.runExclusive(async () => {
            const allIds = await this.store.getAllChunkIds();
            const referenced = new Set<string>();
            for (const rec of await this.store.listFileRecords()) {
                for (const id of rec.chunk_ids) referenced.add(id);
            }
            return findOrphanChunkIds(allIds, referenced);
        });
        if (orphans.length === 0) return { removed: 0, completed: true };
        const BATCH = 500;
        let deleted = 0;
        let completed = true;
        for (let i = 0; i < orphans.length; i += BATCH) {
            if (opts.shouldContinue && !opts.shouldContinue()) { completed = false; break; }
            const batch = orphans.slice(i, i + BATCH);
            await this.coord.runExclusive(() => this.store.deleteChunksByIds(batch));
            deleted += batch.length;
        }
        if (deleted > 0) {
            this.invalidateBm25Cache();
            void this.warmCaches('orphan-sweep');
        }
        return { removed: deleted, completed };
    }

    // Dead-identity sidecar reap (Phase 3 §4): at a full reindex — which republishes
    // THIS device at the current identity — remove every OTHER device's sidecar files
    // whose meta identity no longer matches this build. Provably useless (metaAccepts
    // refuses them, so no current-version device can hydrate from them), so which
    // device wrote them is irrelevant; sync-safe because an un-updated device just
    // re-creates its own until it updates. Never reaps self (just rewritten) or a
    // current-identity peer (a valid producer). A null/torn meta fails the gate and is
    // reaped too — it was unusable anyway. Returns the device count reaped.
    //
    // Enumerates by listSidecarDeviceIds (the UNION of all artifact types), NOT by
    // jsonl alone: clearDevice deletes a device's four files non-atomically and sync
    // propagates each independently, so a dead device's jsonl can already be gone
    // while its meta/bm25/shards linger. A jsonl-keyed reap would never revisit those
    // leftovers (permanent disk leak); the union re-finds a half-cleared device and
    // clearDevice — idempotent + exists-guarded — finishes it on this or a later
    // pass. The metaAccepts gate is unchanged, so a current peer mid-sync (meta
    // present, current identity) is still kept; only provably-stale or unprovable
    // (no/torn meta) devices are reaped, exactly as before.
    private async reapDeadIdentitySidecars(): Promise<number> {
        if (!this.coord.sidecarOn()) return 0;
        const dir = this.coord.dir!;
        const adapter = this.app.vault.adapter;
        const expect = expectationFor();
        let reaped = 0;
        for (const dev of await listSidecarDeviceIds(adapter, dir)) {
            if (dev === this.logger.deviceId) continue;     // never reap self
            if (metaAccepts(await readDeviceMeta(adapter, dir, dev), expect)) continue; // current peer — keep
            await clearDevice(adapter, dir, dev);
            reaped++;
        }
        return reaped;
    }

    // Post-recovery health check: rebuild the frame + BM25 from the (reconciled)
    // IDB and assert their row spaces agree. Embed-free — ensureFrame/ensureBm25
    // only read the store + fit BM25, no model touch. full=true runs the exhaustive
    // id↔row check (rare, off the hot path). True when coherent (or empty — an empty
    // index can't be incoherent); false when drift survived the rebuild (a genuinely
    // corrupt IDB → the plugin flips indexHealth to 'degraded').
    async verifyCoherent(): Promise<boolean> {
        const frame = await this.ensureFrame();
        if (!frame) return true;
        await this.ensureBm25(frame.orderedChunks);
        if (!this.bm25Cache) return true;
        return frameBm25Coherent(frame, this.bm25Cache, true);
    }

    // In-session cache of the sidecar-dir signature at the last successful
    // reconcile. Seeded from localStorage on first use (see reconcileSigKey) so
    // the gate SURVIVES reloads — without that, an iOS crash-relaunch loop reset
    // this to null every rebirth and re-chunked the entire vault on each onload.
    private lastReconcileSig: string | null = null;

    // Reconcile-signature storage key. App#saveLocalStorage vault-scopes by the
    // vault's appId, so cross-VAULT clobber is handled automatically; we KEEP
    // dbName in the key so two co-installed Seek builds (id 'seek' + 'seek-prototype')
    // in the SAME vault still don't clobber each other's gate (same appId namespace).
    private reconcileSigKey(): string {
        return `seek:reconcile-sig:${this.store.dbName}`;
    }

    private loadPersistedReconcileSig(): string | null {
        try {
            const v: unknown = this.app.loadLocalStorage(this.reconcileSigKey());
            return typeof v === 'string' ? v : null;
        }
        catch { return null; }   // unavailable (private mode/quota) → fall back to in-memory gating
    }

    private persistReconcileSig(sig: string): void {
        this.lastReconcileSig = sig;
        try { this.app.saveLocalStorage(this.reconcileSigKey(), sig); }
        catch { /* best-effort; in-memory cache still gates this session */ }
    }

    // Empty-store probe for the reconcile gate's mandatory eviction guard. A
    // count() failure is treated as empty (sweep) — the safe side: a needless
    // sweep costs time, a skipped recovery costs all search results.
    private async indexIsEmpty(): Promise<boolean> {
        try { return (await this.store.count()).files === 0; }
        catch { return true; }
    }

    // Public cold-start probe for the search modal. Returns the indexed chunk count,
    // or null when the store can't be read yet (mid-init / not open). The modal flips
    // to its "not indexed" onboarding copy only on a confirmed 0 and leaves its resting
    // copy untouched on null — so a still-warming index is never mislabeled "not
    // indexed". chunks (not files) is the true "is anything searchable" signal.
    async indexedChunkCount(): Promise<number | null> {
        try { return (await this.store.count()).chunks; }
        catch { return null; }
    }

    // Gated whole-vault reconcile, shared by BOTH the onload sweep and the 5-min
    // interval. Skips the expensive reChunkLive sweep when the sidecar dir is
    // byte-identical to the last successful reconcile AND the local index is
    // populated; always sweeps when the store is empty (iOS eviction / fresh
    // device — see shouldReconcileSidecar). The signature is persisted, so a
    // crash-relaunch with an unchanged vault no longer re-chunks for nothing.
    // The manual `seek-sidecar-reconcile` command and the drift-recovery ladder
    // deliberately bypass this and call hydrateSidecar() directly (explicit
    // user / recovery intent must never be gated).
    async reconcileSidecarIfChanged(): Promise<HydrateResult | null> {
        if (!this.coord.sidecarOn()) return null;
        const sig = await sidecarDirSignature(this.app.vault.adapter, this.coord.dir!, this.logger.deviceId);
        const prev = this.lastReconcileSig ?? this.loadPersistedReconcileSig();
        if (!shouldReconcileSidecar(sig, prev, await this.indexIsEmpty())) {
            // Signature unchanged → the (expensive) hydrate is skipped, but _peerAhead resets
            // to false on every new orchestrator and ONLY hydrateSidecar refreshes it. Recover
            // the bit with a cheap meta-only probe so the peer-ahead banner + mobile grind-stop
            // survive an app relaunch on an unchanged vault (the common iOS case) instead of
            // silently disengaging while falsely reading "healthy". No reChunk, no mutation —
            // just a producer-meta read; mirrors hydrateFromSidecar's own peerAhead predicate.
            this._peerAhead = await probePeerAhead(this.app.vault.adapter, this.coord.dir!, expectationFor());
            return null;
        }
        const result = await this.hydrateSidecar();
        this.persistReconcileSig(sig);
        return result;
    }

    // Re-chunk every live, indexable note AND .base — the liveness oracle for a
    // full hydrate. Mirrors computeDelta's file filter (indexableFiles, so bases
    // are included) AND embedAndCommitFiles' full chunk pipeline: chunksFor THEN
    // enforceTokenBudget, so chunk_ids match exactly what indexing produced —
    // including base chunks (chunksFor → chunkBase), which is what lets the phone
    // rehydrate base vectors embed-free instead of dropping them. The token-budget
    // re-split is load-bearing,
    // NOT optional: chunk_id = chunkIdFor(path, title, content), and on long
    // notes indexing splits oversize chunks into new contents → new ids. Without
    // reproducing that split here, the sidecar lookup missed every split note
    // (hydrated:0 → fell back to a full on-device re-embed → iPhone jetsam; see
    // the 2026-06-14 forensics). Loads the tokenizer ONLY (a few MB, no ~250 MB
    // model) so the hydrate stays mobile-safe.
    private async reChunkLive(): Promise<ReChunkedNote[]> {
        await this.embedder.ensureTokenizer();
        const out: ReChunkedNote[] = [];
        for (const f of this.indexableFiles().filter(f => this.shouldIndex(f.path))) {
            let content: string;
            try {
                content = await this.app.vault.cachedRead(f);
            } catch {
                continue;
            }
            let chunks = this.chunksFor(content, f.path, new Date(f.stat.mtime).toISOString());
            if (chunks.length === 0) continue;
            try {
                chunks = (await enforceTokenBudget(chunks, ts => this.embedder.tokenCounts(ts))).chunks;
            } catch (e) {
                // Tokenizer hiccup on one note — skip its hydrate (it just embeds
                // later via the catch-up); never abort the whole hydrate.
                await this.logger.appendError(`reChunkLive-tokenBudget:${f.path}`, e);
                continue;
            }
            if (chunks.length > 0) out.push({ notePath: f.path, mtimeMs: f.stat.mtime, chunks, contentHash: cyrb53Hex(content) });
        }
        return out;
    }

    // The live-vault chunk_id set + whether it is COMPLETE — the liveness oracle for
    // sidecar compaction. Mirrors reChunkLive's id pipeline (chunksFor THEN
    // enforceTokenBudget over indexableFiles, so ids match what indexing wrote for
    // both notes and bases — without the base coverage compaction would count every
    // base chunk dead and GC it). Returns only ids and,
    // critically, a completeness flag: reChunkLive SWALLOWS per-note read/tokenizer
    // errors (hydrate only skips, which is non-destructive), but compaction DELETES on
    // this oracle, so a single skipped note would wrongly drop that note's live records.
    // `complete:false` on ANY skip tells the caller to abort the compaction and retry
    // when the transient error clears. Empty notes (no chunks) are not skips — they have
    // no ids to drop. Kept separate from reChunkLive on purpose: the load-bearing hydrate
    // path stays untouched, at the cost of one extra (tokenizer-only) re-chunk per session.
    private async collectLiveIds(): Promise<{ ids: Set<string>; complete: boolean }> {
        const ids = new Set<string>();
        try {
            await this.embedder.ensureTokenizer();
        } catch (e) {
            // A tokenizer-load failure (model files mid-sync on a cold mobile launch) makes
            // the whole snapshot untrustworthy — report incomplete (transient → caller
            // retries) rather than throwing into an anonymous swallowed catch upstream.
            await this.logger.appendError('collectLiveIds-ensureTokenizer', e);
            return { ids, complete: false };
        }
        let complete = true;
        for (const f of this.indexableFiles().filter(f => this.shouldIndex(f.path))) {
            let content: string;
            try {
                content = await this.app.vault.cachedRead(f);
            } catch (e) {
                // read error (e.g. an iCloud file not yet downloaded) → snapshot incomplete →
                // unsafe to delete. Logged with the path: this is the likeliest cause of a
                // persistent 'incomplete-rechunk' skip, so it must not be silent.
                complete = false;
                await this.logger.appendError(`collectLiveIds-read:${f.path}`, e);
                continue;
            }
            let chunks = this.chunksFor(content, f.path, new Date(f.stat.mtime).toISOString());
            if (chunks.length === 0) continue; // genuinely empty note — no ids, not a skip
            try {
                chunks = (await enforceTokenBudget(chunks, ts => this.embedder.tokenCounts(ts))).chunks;
            } catch (e) {
                complete = false; // tokenizer hiccup → incomplete
                await this.logger.appendError(`collectLiveIds-tokenBudget:${f.path}`, e);
                continue;
            }
            for (const c of chunks) ids.add(c.chunk_id);
        }
        return { ids, complete };
    }

    // Compact THIS device's own sidecar (jsonl + shards) if it has accumulated enough
    // dead bytes. Model-FREE — a pure byte-copy of existing vectors — so, unlike a full
    // reindex, it is safe on mobile (same read-concat-write footprint bulkAppend already
    // runs there, no GPU/model heap → no jetsam). It is also the ONLY reclaim path for a
    // device that never reaches a desktop: a version-bump reindex and hydrate-from-a-
    // compacted-peer both need connectivity an off-grid phone/iPad may lack for months,
    // and its own edit churn would otherwise pile up unbounded. Called at most once per
    // session from periodicReconcile. Returns null when the sidecar is off.
    async compactOwnSidecar(): Promise<CompactResult | null> {
        if (!this.coord.sidecarOn()) return null;
        const adapter = this.app.vault.adapter;
        const dir = this.coord.dir!;
        const dev = this.logger.deviceId;

        // A SIDECAR_FORMAT bump is deliberately excluded from identityMatches (it
        // governs only the cross-device file protocol, not local IDB validity — see
        // identity.ts), so this device can carry a stale-format sidecar on disk
        // indefinitely: nothing forces a full reindex (the only path that clears it
        // via clearDevice) just because the format constant moved. compactDevice()
        // decodes every record with the CURRENT fixed stride unconditionally — run
        // against a genuinely stale-format shard, every record misaligns, fails its
        // CRC, and gets shed as "corrupt", permanently destroying this device's
        // pre-upgrade vector history (and leaving a peer's later hydration attempt
        // to hit the same misread). Detect the mismatch from this device's own last-
        // written meta BEFORE compacting and, instead of misreading, explicitly wipe
        // via the same primitive a full reindex uses — the sidecar self-heals from
        // normal incremental commits afterward, same as any other reap.
        const ownMeta = await readDeviceMeta(adapter, dir, dev);
        if (staleSidecarFormat(ownMeta)) {
            await clearDevice(adapter, dir, dev);
            return { compacted: false, reason: 'format-mismatch', recordsBefore: 0, recordsAfter: 0, bytesBefore: 0, bytesAfter: 0, shed: 0 };
        }

        // Off-grid means no iCloud re-upload now (it lands as a SMALLER upload on
        // reconnect), but the rewrite still costs IO — so a higher floor on mobile.
        const floor = isMobilePlatform() ? 4 * 1024 * 1024 : 2 * 1024 * 1024;
        // NOTE: no stat-only pre-gate here on purpose. deviceShardBytes(...) < floor would
        // sum ALL on-disk shard bytes, including crash-leaked orphans (shards referenced by
        // zero jsonl lines — see compactDevice's header comment), and skip calling
        // compactDevice entirely for any device whose live+orphan total stays under the
        // floor. compactDevice's own orphan reclaim runs BEFORE its internal below-floor
        // check (on live bytes only), so it must always be reachable — a caller-side
        // pre-gate on the unfiltered total would make a small, quiet device's leaked
        // shards unreclaimable forever. compactDevice is cheap to call here regardless:
        // this path runs at most once per session (see main.ts), and its own below-floor
        // check bails before the expensive live-id re-chunk.
        // The live-id oracle runs INSIDE compactDevice's dir lock so the snapshot and the
        // on-disk scan exclude concurrent appends — the drop decision needs no clock.
        const result = await compactDevice(adapter, dir, dev, () => this.collectLiveIds(), { minDeadRatio: 0.5, minShardBytes: floor });
        // Refresh the persisted reconcile sig after a self jsonl rewrite. With self now
        // excluded from sidecarDirSignature (a device's own writes don't move its own
        // signature), this compaction touches only our own jsonl so the signature is
        // already unchanged — this re-persist is belt-and-suspenders, kept so the
        // persisted sig is recomputed with the SAME selfDeviceId convention as the live
        // gate, and to stay correct if compaction ever touches a non-self artifact.
        if (result.compacted) {
            try {
                this.persistReconcileSig(await sidecarDirSignature(adapter, dir, dev));
            } catch (e) {
                await this.logger.appendError('compact-sig-refresh', e);
            }
        }
        return result;
    }

    // Devices already warned about a version-gate refusal this session, keyed by
    // `${device}:${reason}` so each distinct staleness is reported once — not on
    // every reconcile / delta flush.
    private warnedRefusals = new Set<string>();

    // True when the last sidecar scan refused a producer at a HIGHER chunkerVersion than
    // this build — another device holds a newer index this plugin is too old to read.
    // Set/cleared per scan (so updating Seek clears it on the next reconcile). The plugin
    // reads it to raise the "update Seek" banner AND, on mobile, to skip the futile local
    // re-embed (a v7 device grinding out chunks it discards the moment it updates to v8).
    private _peerAhead = false;
    get peerAhead(): boolean { return this._peerAhead; }

    // A version-gate refusal is EXPECTED and self-healing — e.g. right after a
    // CHUNKER_VERSION bump the other device's sidecar is one version behind until
    // it re-embeds. So it's a one-time warn (no stack trace, not an `error` entry):
    // the old appendError fired two red console lines + an error log on every
    // reconcile, flooding the console for a benign, transient state. Refusing the
    // producer is correct (a stale chunk_id space can't be reproduced locally) —
    // only the loudness was the bug.
    private warnRefusedProducer(dev: string, meta: SidecarMeta | null, expect: MetaExpectation): void {
        const reason = !meta
            ? 'missing/unreadable meta'
            : [
                  meta.format !== SIDECAR_FORMAT ? `format ${meta.format}≠${SIDECAR_FORMAT}` : '',
                  meta.modelId !== expect.modelId ? `model ${meta.modelId}≠${expect.modelId}` : '',
                  meta.chunkerVersion !== expect.chunkerVersion ? `chunker v${meta.chunkerVersion}≠v${expect.chunkerVersion}` : '',
                  meta.dim !== expect.dim ? `dim ${meta.dim}≠${expect.dim}` : '',
              ]
                  .filter(Boolean)
                  .join(', ');
        const key = `${dev}:${reason}`;
        if (this.warnedRefusals.has(key)) return;
        this.warnedRefusals.add(key);
        console.warn(`[seek] skipping sidecar producer ${dev} (${reason}) — its index predates this device; will hydrate once that device re-embeds`);
        // One NDJSON breadcrumb (not an error entry) so device reports still show a
        // producer was paused. Reuses SidecarHydrateEntry's open index signature.
        void this.logger.append({ type: 'sidecar-hydrate', timestamp: new Date().toISOString(), phase: 'version-gate-refused', device: dev, reason }).catch(() => {});
    }

    // Set once we've warned this session that the index is empty AND the sidecar
    // restored nothing — cleared again the moment a hydrate repopulates the index,
    // so a later strand still warns. Keeps onload + first periodic + manual
    // reconcile from each re-logging the same dead-net state.
    private warnedStranded = false;

    // The failure the sidecar exists to PREVENT: sidecar is on, yet after a
    // hydrate the IDB index is still empty → the next search returns ZERO results
    // with no error. Causes: iOS evicted the IDB and no producer has synced in, or
    // the index dir was deleted / never reached this device (the config-folder
    // split). This used to be silent — only a `producerFilesFound:0` debug line.
    // Surface it as a real console error ONCE, gated on a true-empty index (cheap
    // IDB count) so a healthy warm index never trips it. Cause is read off the
    // hydrate result so the message is actionable.
    private async warnIfIndexStranded(result: HydrateResult): Promise<void> {
        if (this.warnedStranded) return;
        const { chunks } = await this.store.count();
        if (chunks > 0) return; // index is populated — search works, nothing stranded
        this.warnedStranded = true;
        const cause =
            result.refusedProducers > 0
                ? `all ${result.refusedProducers} sidecar producer(s) were version-refused — the other device's plugin/model is out of date`
                : result.skippedPartialNotes > 0
                  ? `sidecar files are still arriving (${result.skippedPartialNotes} note(s) only partially synced)`
                  : result.acceptedProducers === 0
                    ? 'no sidecar index files were found here — deleted, or no other device has synced its index to this folder yet'
                    : 'the sidecar held nothing this device could reproduce';
        console.error(
            `[seek] index is EMPTY and the sidecar restored nothing — search will return no results. ` +
                `Cause: ${cause}. Fix: run a full reindex on this device, or let another device's sidecar sync in.`,
        );
        // Persist to NDJSON too — console.error is invisible on mobile (no
        // devtools), and mobile is exactly where eviction strands the index.
        void this.logger
            .append({
                type: 'sidecar-hydrate',
                timestamp: new Date().toISOString(),
                phase: 'index-stranded',
                cause,
                refusedProducers: result.refusedProducers,
                acceptedProducers: result.acceptedProducers,
                skippedPartialNotes: result.skippedPartialNotes,
            })
            .catch(() => {});
    }

    // Shared HydrateDeps; only the re-chunk source varies (whole vault vs the
    // dirty subset for delta dedup).
    // addsSink (Seek scaling A1): when present, putQuantized ALSO surfaces each
    // hydrated chunk into the delta change-set, so a sidecar-dedup delta can be
    // applied incrementally (applyDelta) instead of forcing a full rebuild. Only
    // the delta path (dedupViaSidecar) passes it; the standalone hydrateSidecar
    // does its own invalidate+warm, so it leaves it undefined.
    private hydrateDeps(reChunk: () => Promise<ReChunkedNote[]>, addsSink?: DeltaAdd[]): HydrateDeps {
        return {
            adapter: this.app.vault.adapter,
            indexDir: this.coord.dir!,
            expect: expectationFor(),
            reChunk,
            existingIds: async () => new Set((await this.store.listAllMeta()).map(c => c.chunk_id)),
            putQuantized: async (chunks, tiers) => {
                await this.store.putBatchQuantized(chunks, tiers);
                // After the IDB write, so the sink holds only rows that landed.
                if (addsSink) pushDeltaAdds(addsSink, chunks, tiers);
            },
            putFileRecord: rec => this.store.putFileRecord(rec),
            onRefusedProducer: (dev, meta, expect) => this.warnRefusedProducer(dev, meta, expect),
            log: (msg, detail) => {
                // Persist to the NDJSON log too — console.log is invisible on
                // mobile (no devtools), which is why hydrate outcomes never
                // surfaced in iPhone reports. Flatten arrays → counts to fit the
                // SidecarHydrateEntry scalar index signature (hydratedNotePaths
                // can be large).
                const flat: Record<string, string | number | boolean | null> = {};
                if (detail && typeof detail === 'object') {
                    for (const [k, v] of Object.entries(detail as Record<string, unknown>)) {
                        if (Array.isArray(v)) flat[`${k}Count`] = v.length;
                        else if (v == null || typeof v === 'string' || typeof v === 'number' || typeof v === 'boolean') flat[k] = v ?? null;
                        else flat[k] = String(v);
                    }
                }
                void this.logger.append({ type: 'sidecar-hydrate', timestamp: new Date().toISOString(), phase: msg, ...flat }).catch(() => {});
            },
        };
    }

    // Dedup-before-embed: hydrate the dirty files the sidecar already covers
    // (another device embedded the same content) instead of re-embedding them,
    // and return the files that still need the model. MUST be called with the
    // dirty files' STALE chunks already dropped (so existingIds doesn't report
    // the pre-edit ids as present). Pure optimization — a miss just embeds.
    private async dedupViaSidecar(files: TFile[], addsSink?: DeltaAdd[]): Promise<TFile[]> {
        if (!this.coord.sidecarOn() || files.length === 0) return files;
        // Same chunk_id-reproduction requirement as reChunkLive: apply the
        // token-budget split so split notes match the sidecar (without it this
        // dedup silently missed every long note and re-embedded it). The model
        // is loaded on this path (reindexDelta's embed half), so ensureTokenizer
        // is a no-op; the guard just makes the dependency explicit.
        await this.embedder.ensureTokenizer();
        const notes: ReChunkedNote[] = [];
        for (const f of files) {
            let content: string;
            try {
                content = await this.app.vault.cachedRead(f);
            } catch {
                continue;
            }
            let chunks = this.chunksFor(content, f.path, new Date(f.stat.mtime).toISOString());
            if (chunks.length === 0) continue;
            try {
                chunks = (await enforceTokenBudget(chunks, ts => this.embedder.tokenCounts(ts))).chunks;
            } catch (e) {
                await this.logger.appendError(`dedupViaSidecar-tokenBudget:${f.path}`, e);
                continue;
            }
            if (chunks.length > 0) notes.push({ notePath: f.path, mtimeMs: f.stat.mtime, chunks, contentHash: cyrb53Hex(content) });
        }
        if (notes.length === 0) return files;
        // Called from inside reindexDelta's runExclusive — invoke the engine
        // directly (NOT hydrateSidecar, which would re-enter the mutex). addsSink
        // (when the delta path supplies it) surfaces the hydrated chunks into the
        // change-set so applyDelta can apply them incrementally.
        const res = await hydrateFromSidecar(this.hydrateDeps(async () => notes, addsSink));
        this._peerAhead = res.peerAhead; // refresh the "newer index exists" signal per scan
        const done = new Set(res.hydratedNotePaths);
        return files.filter(f => !done.has(f.path));
    }

    // F13 carry-over: harvest the rerank/sign tiers of every chunk about to be
    // removed, keyed by EMBED TEXT (title\n\ncontent — path-independent). A move
    // re-keys identical content under a new path-salted chunk_id, so its vector is
    // unchanged; harvesting lets the embed phase reuse it verbatim. Reads the
    // affected files' stored chunks by id BEFORE they're deleted; merges into `map`
    // so the deleted-path and dirty-path harvests share one table.
    private async harvestCarryOverInto(
        map: Map<string, { q: QuantVec; sign: Uint8Array }>,
        paths: string[],
    ): Promise<void> {
        const seen = new Set<string>();
        for (const path of paths) {
            if (seen.has(path)) continue;
            seen.add(path);
            try {
                const rec = await this.store.getFileRecord(path);
                if (!rec || rec.chunk_ids.length === 0) continue;
                const tiers = await this.store.getTiersByIds(rec.chunk_ids);
                for (const t of tiers) {
                    if (t) map.set(embedInput(t.chunk), { q: t.q, sign: t.sign });
                }
            } catch (e) {
                // Carry-over is a pure optimization (reuse the identical vector on a
                // move / no-op re-flush). A harvest failure must NEVER abort the
                // delta — that would drop this edit from the index entirely. Record
                // it (forensics ring + per-device log) and fall through: this file
                // simply re-embeds normally.
                await this.logger.appendError(`carryOver-harvest:${path}`, e);
            }
        }
    }

    // F13: re-chunk each candidate and, if EVERY chunk's embed text is in the
    // carry-over map, write the chunks with their REUSED tiers (verbatim, no model
    // forward pass) and drop the file from the embed set. All-or-nothing per note,
    // mirroring dedupViaSidecar: a partially-changed file falls through to a normal
    // embed. The headline win is a folder reorg (pure moves) re-keying for free.
    private async carryOverHydrate(
        files: TFile[],
        carryOver: Map<string, { q: QuantVec; sign: Uint8Array }>,
    ): Promise<TFile[]> {
        if (carryOver.size === 0 || files.length === 0) return files;
        // The model-loaded embed path already has the tokenizer; the guard just
        // makes the chunk_id-reproduction dependency explicit (as in dedupViaSidecar).
        await this.embedder.ensureTokenizer();
        const done = new Set<string>();
        for (const f of files) {
            let content: string;
            try { content = await this.app.vault.cachedRead(f); } catch { continue; }
            let chunks = this.chunksFor(content, f.path, new Date(f.stat.mtime).toISOString());
            if (chunks.length === 0) continue;
            try {
                chunks = (await enforceTokenBudget(chunks, ts => this.embedder.tokenCounts(ts))).chunks;
            } catch (e) {
                await this.logger.appendError(`carryOver-tokenBudget:${f.path}`, e);
                continue;
            }
            if (chunks.length === 0) continue;
            const tiers = chunks.map(c => carryOver.get(embedInput(c)));
            if (tiers.some(t => t === undefined)) continue;   // not fully covered → embed normally
            await this.store.putBatchQuantized(chunks, tiers.map(t => ({ q: t!.q, bin: t!.sign })));
            await this.store.putFileRecord({ note_path: f.path, mtimeMs: f.stat.mtime, chunk_ids: chunks.map(c => c.chunk_id), contentHash: cyrb53Hex(content) });
            done.add(f.path);
        }
        return files.filter(f => !done.has(f.path));
    }

    private async emitProgress(
        phase: IndexProgressEntry['phase'],
        filesSeen: number,
        filesTotal: number,
        chunksEmitted: number,
        elapsedMs: number,
    ): Promise<void> {
        // index-progress is a high-volume per-batch firehose that's ALSO mirrored into
        // the crash-forensics breadcrumb ring (the copy that survives a jetsam kill — the
        // one that matters for crash classification). Persist it to the NDJSON only under
        // verboseTrace; index-complete still records the per-run summary unconditionally.
        if (!this.settings.verboseTrace) return;
        // Storage probe runs alongside heap so iOS gets a non-null signal too.
        const mem = await snapshotMemory();
        const entry: IndexProgressEntry = {
            type: 'index-progress',
            timestamp: new Date().toISOString(),
            phase,
            filesSeen,
            filesTotal,
            chunksEmitted,
            elapsedMs: parseFloat(elapsedMs.toFixed(2)),
            heapMB: mem.heapMB,
            storageMB: mem.storageMB,
        };
        await this.logger.append(entry);
    }

    // Resolve the per-search FilterContext from live app + settings: the set of
    // Number-typed properties (read from Obsidian's registry each search — a cheap
    // dictionary lookup) and the date field the recency-gated `before:`/`after:`
    // filters key off. dateField is null when Recency is OFF (recencyEpsilon ≤ 0),
    // which is how the parser knows to leave a typed before:/after: as plain text.
    // recencyOverride (see search()) resolves the effective epsilon locally so a
    // query-time override never has to touch this.settings to take effect here.
    // See [[Seek Typed-Value Filters Design]].
    private buildFilterContext(recencyOverride?: RecencyOverride): FilterContext {
        const epsilon = recencyOverride?.epsilon ?? this.settings.recencyEpsilon;
        const recencyOn = epsilon > 0;
        return {
            dateField: recencyOn
                ? { key: this.settings.recencyKey, createdProp: this.settings.createdProp }
                : null,
            numericKeys: enumerateNumberPropertyNames(this.app),
        };
    }

    // Search path (two-stage, v7+):
    //
    //   S0  resident frame (corpus + binary index, cached by dataGeneration —
    //       listAllChunks runs only on a cache miss, i.e. after a reindex)
    //   S1  union of three candidate gens, fed by the resident tier:
    //         a. binary-top-N — asymmetric float·sign-bit dot product
    //         b. bm25-top-M   — multi-field BM25F (cached)
    //         c. recency-top-K — newest by frontmatter `created` (mtime fallback)
    //   S2  fp32 exact rerank ONLY over the union: load fp32 vectors for the
    //       union ids, cosine, then run the shipped α-min-max-hybrid + recency
    //       + title boost over the subset
    //   S3  file-level max-aggregation dedup, snippet, log
    //
    // The S1 union — not just binary — is mandatory per the design note: BM25
    // and recency arms exist precisely to recover the ~9% dense-unreachable
    // gold that exact dense itself caps at. Skipping them would regress
    // recall vs the old all-chunks scorer.
    //
    // recencyOverride: the seek:search CLI's per-query recencyWeight/
    // recencyHalflife params (see main.ts). Resolved locally into filterCtx/
    // rankConfig below — deliberately NEVER written into this.settings.
    // this.settings is a single live object shared by every concurrent
    // caller (other CLI calls, the search modal, openTopResult), so an
    // earlier version that mutated it for the override's duration let a
    // plain concurrent search silently rank against someone else's override
    // (2026-07-02 review). Passing the override as a call-local argument
    // instead makes overlapping searches independent by construction — no
    // shared mutable state, so nothing to race or leak.
    async search(query: string, topK = 10, recencyOverride?: RecencyOverride): Promise<{ results: ScoredChunk[]; entry: SearchEntry }> {
        const t0 = performance.now();
        const searchId = `${Date.now()}-${query.slice(0, 20)}`;

        // Parse inline filter syntax (#tag / tag: / path: / [k:v] / numeric /
        // dates). `cleanedQuery` is the residual text we embed + BM25; `filters`
        // drives the candidate-selection match-mask below. A null `filters` means a
        // plain query — the mask stays undefined and the path is unchanged. The
        // FilterContext threads vault-specific facts (Number-typed props, the
        // Recency date field) into the otherwise-pure parser/matcher; same object
        // is reused by compileMatcher so parse and match agree on types/field.
        const filterCtx = this.buildFilterContext(recencyOverride);
        const { cleanedQuery, filters } = parseQuery(query, filterCtx);

        // ---- S0: resident-tier read ------------------------------------
        // listAllChunks is unavoidable: BM25 needs all chunks to build its
        // inverse index, and the result UI needs chunk text/metadata. But it's
        // pure text — no fp32 vectors traverse the cursor anymore (that was a
        // full-embeddings dequant scan, removed from the hot path).
        // ---- S0: resident frame (cached by dataGeneration) -------------
        // The frame is the corpus in binary-index order with its aligned packed
        // buffer; all three S1 arms and S2 index into it, so "candidate index i"
        // has one meaning. ensureFrame() reads the full chunk store + assembles
        // the frame only on a cache miss (reindex); a warm keystroke returns the
        // cached frame with zero IDB traffic — this is what removes the old
        // per-query listAllChunks read (2026-06-03: ~55% of warm latency).
        // Orphans (binary row without chunk sibling) are dropped inside
        // ensureFrame; they're not retrievable anyway since the UI needs text.
        const idbStart = performance.now();
        const frameWasCached = !!(this.frameCache && this.frameCache.generation === this.coord.generation);
        const frame = await this.ensureFrame();
        const idbReadMs = performance.now() - idbStart;
        // Frame and binary index are built together under one dataGeneration, so
        // a frame hit implies the binary index was served from cache too.
        const binaryCacheHitFlag = frameWasCached;

        if (!frame) {
            const entry: SearchEntry = this.emptySearchEntry(query, cleanedQuery, filters, topK, searchId, idbReadMs, performance.now() - t0);
            await this.logger.append(entry);
            return { results: [], entry };
        }

        const orderedChunks = frame.orderedChunks;
        const orderedIds = frame.orderedIds;
        const activePacked = frame.activePacked;
        const bytesPerVec = frame.bytesPerVec;
        const residentInt8 = frame.residentInt8;
        const residentScales = frame.residentScales;
        const embDim = frame.embDim;
        const frameGen = frame.generation;

        // ---- Corpus-scaled candidate-pool caps (scaling-doc C) ----------
        // liveN = rows minus incremental-delete tombstones (orderedChunks still
        // carries them until compaction), so a churned vault doesn't inflate the
        // pool. poolCaps grows binary + bm25 by √N and holds recency flat; at our
        // current scale this returns exactly the legacy 200/100/50 (see pool.ts).
        const liveN = orderedChunks.length - frame.tombstoneCount;
        const caps = poolCaps(liveN);

        // ---- Inline-filter pre-filter (match-mask) ---------------------
        // Build a boolean mask over orderedChunks. Filtering is applied at
        // candidate SELECTION (topNIndices / topByRecency below), NOT by
        // shrinking orderedChunks — that keeps the BM25 + binary indexes
        // (built over the full corpus) cache-valid across filtered queries.
        // null filters → undefined mask → byte-identical to a no-filter build.
        const matcher = filters ? compileMatcher(filters, filterCtx) : null;
        // buildSelectionMask folds frame liveness (validRows) AND the inline
        // filter into one mask, so a single downstream `mask` carries BOTH to
        // every consumer: the binary scan, the bm25/recency selection arms, the
        // filter-only browse loop, and the negation AND below. Tombstoned rows
        // (incremental deletes awaiting compaction) are excluded everywhere —
        // including the browse path, whose `!mask ||` short-circuit would
        // otherwise admit holes. undefined only when fully-live AND unfiltered
        // (the byte-identical no-filter fast path).
        let mask = buildSelectionMask(orderedChunks, frame.validRows, frame.tombstoneCount, matcher);

        // Note-level negation (`-term`). compileMatcher is metadata-only and a
        // per-chunk predicate, so it can't express "drop the whole note if any
        // chunk contains X" — we resolve that here over the full corpus and AND
        // it into the same selection mask. Obsidian's `-` is file-level, so we
        // exclude every chunk of a matched note, not just the matching chunk.
        // Folded into `mask`, it costs nothing downstream and covers both the
        // main path and the filter-only fast path below.
        if (filters?.exclude && filters.exclude.length > 0) {
            // Negation matches on body text, but the frame is metadata-only — so
            // fetch the corpus bodies on demand. Paid ONLY on a `-term` query.
            const bodyMap = await this.store.getBodiesMap(orderedIds);
            const excludedNotes = excludedNotePaths(orderedChunks, filters.exclude, id => bodyMap.get(id));
            if (excludedNotes.size > 0) {
                if (!mask) mask = new Array<boolean>(orderedChunks.length).fill(true);
                for (let i = 0; i < orderedChunks.length; i++) {
                    if (mask[i] && excludedNotes.has(orderedChunks[i].note_path ?? '')) {
                        mask[i] = false;
                    }
                }
            }
        }

        // ---- Filter-only fast path ------------------------------------
        // The query was nothing but operators (e.g. "#meetings", "path:X/*").
        // With no semantic/lexical text there is no relevance signal — this is
        // a BROWSE, ordered by the explicit recency-desc sort in browseOrder
        // (fusion.ts; keyed by the vault's recencyKey setting), NOT the ranking
        // formula. It must stay independent of rank()/RankingConfig so ranking
        // changes can never park the ordering out from under it again (audit
        // 2026-06-09 §1). Scores are honest zeros: nothing is scored here.
        if (cleanedQuery === '') {
            const matchedChunks: ChunkMeta[] = [];
            for (let i = 0; i < orderedChunks.length; i++) {
                if (!mask || mask[i]) matchedChunks.push(orderedChunks[i]);
            }
            // Inline dedup-by-note over the sorted chunks (dedupByPath wants
            // pre-scored input; building ScoredChunks for every match just to
            // throw most away would copy the whole matched set).
            const seenPaths = new Set<string>();
            const results: ScoredChunk[] = [];
            for (const c of browseOrder(matchedChunks, this.settings.recencyKey, this.settings.createdProp)) {
                if (seenPaths.has(c.note_path)) continue;
                seenPaths.add(c.note_path);
                results.push({
                    ...c,
                    content: '',   // frame-lite: hydrated below from chunk_body
                    score: 0,
                    ranking_signals: { dense: 0, bm25: 0, hybrid: 0, recency: 0, title_boost: 0, denseRaw: 0 },
                });
                if (results.length >= topK) break;
            }
            await this.hydrateBodies(results);
            for (const r of results) r.snippet = makeSnippet(r.content, '', 200);
            const entry = this.emptySearchEntry(query, cleanedQuery, filters, topK, searchId, idbReadMs, performance.now() - t0);
            entry.totalChunks = orderedChunks.length;
            entry.candidateUnionSize = matchedChunks.length;
            entry.recencyCount = matchedChunks.length;
            await this.logger.append(entry);
            return { results, entry };
        }

        // ---- S0.5: query embedding ------------------------------------
        // granite-r2 is symmetric (no query/doc prompt), so the query takes the
        // SAME pass as the doc side. As of v8 (2026-06-28) the doc side is no
        // longer raw: cleanDenseBody/cleanDenseText run in the chunker (wikilinks
        // → alias, URLs → readable words, HTML stripped), so the query must be
        // dense-cleaned too or the two vectors drift on any query carrying [[…]],
        // a URL, or markdown syntax. cleanDenseText is a no-op on a plain-text
        // query, so ordinary queries are byte-identical to before. (BM25 below
        // keeps the raw cleanedQuery: seekTokenize fragments that same syntax
        // symmetrically on both index and query side, so the lexical channel
        // needs no parallel cleaning pass.)
        const qStart = performance.now();
        const embedded = await this.embedder.embed(cleanDenseText(cleanedQuery));
        const queryVec = embedded.vector;
        const iframeEmbedMs = embedded.iframeLatencyMs;
        const queryEmbedMs = performance.now() - qStart;

        // Vector sanity: a NaN/Inf query embedding (WASM numeric fault, torn
        // model load) poisons every cosine downstream — comparisons all come
        // back false and results render in frame order scored 0, which reads as
        // silent ranking corruption rather than an error. Every dense score
        // derives from this one vector, so this single gate closes the class;
        // embed() declines to cache a non-finite vector, so a retry re-embeds.
        if (!queryVec.every(Number.isFinite)) {
            await this.logger.appendError(
                'searchQueryVectorNonFinite',
                new Error(`Query embedding contains non-finite values (dim ${queryVec.length}) — corrupt embedder output; retry the search.`),
            );
            const entry: SearchEntry = this.emptySearchEntry(query, cleanedQuery, filters, topK, searchId, idbReadMs, performance.now() - t0);
            await this.logger.append(entry);
            return { results: [], entry };
        }

        // Dim sanity: the binary index was packed at the dim of whatever the
        // last reindex wrote. If the live model now emits a different dim, the
        // asymmetric scorer will throw — surface the actionable error instead
        // of letting it crash the search.
        if (bytesPerVec !== ((queryVec.length + 7) >> 3)) {
            await this.logger.appendError(
                'searchDimMismatch',
                new Error(
                    `Binary index was packed for ${bytesPerVec * 8}-d vectors but the loaded model emits ` +
                    `${queryVec.length}-d. Run "Seek: Full reindex" to rebuild.`,
                ),
            );
            const entry: SearchEntry = this.emptySearchEntry(query, cleanedQuery, filters, topK, searchId, idbReadMs, performance.now() - t0);
            await this.logger.append(entry);
            return { results: [], entry };
        }

        // ---- S1a: binary candidate-gen (off-thread when available) ------
        // Asymmetric float·sign-bit dot product across the whole resident index.
        // The score order is what matters (asymmetric is a biased estimator of
        // cosine — see binary.ts) so we use it only to pick the top-N; the actual
        // relevance score in stage 2 comes from real cosine.
        //
        // Dispatched as a promise so the worker's O(corpus) scan OVERLAPS the
        // main-thread BM25 + recency work below; awaited just before the union.
        // binaryCandidatesAsync resolves to the IDENTICAL indices the synchronous
        // path would (the shared binaryCandidates), via the worker or its fallback.
        const binaryStart = performance.now();
        const binaryPromise = binaryCandidatesAsync(
            this.binaryWorker, frameGen, queryVec, activePacked,
            orderedChunks.length, bytesPerVec, caps.binary, mask ?? null,
        );

        // ---- S1b: BM25 candidate-gen (cached) ---------------------------
        // BM25 fits over orderedChunks (parallel to the binary array — same
        // index space) so its score array slots directly alongside the binary
        // arm. The dataGeneration cache pattern is unchanged from v0.
        const bm25Start = performance.now();
        // Cold cache → try the local persisted MiniSearch index (IDB), then the
        // cross-device sidecar artifact, before fitting. Guarded on bm25CacheValid so
        // a WARM keystroke skips the await/IDB entirely. The cross-device fallback only
        // runs when the local blob is ABSENT (a fresh/evicted device whose IDB has no
        // BM25 yet) — for a normal device with a local blob it never executes, so the
        // hot path pays zero. bm25Ms therefore times the load-or-fit, whichever ran.
        if (!this.bm25CacheValid(orderedChunks)) {
            await this.tryLoadPersistedBm25(orderedChunks);
            if (!this.bm25CacheValid(orderedChunks)) {
                await this.tryLoadCrossDeviceBm25(orderedChunks);
            }
        }
        const bm25CacheHit = await this.ensureBm25(orderedChunks);
        // Query-entry drift guard (Seek scaling A1): the frame and BM25 index are
        // about to be jointly indexed by a single row id (the candidate union
        // below), so verify that coupling holds before trusting it. A trip should
        // be impossible — applyDelta verifies + re-stamps under the delta mutex, and
        // the caches are mutated nowhere else — so on the off chance one slips
        // through, serve nothing for THIS keystroke and drop to a full rebuild
        // rather than return silently mis-joined scores; the next search hits the
        // rebuilt cache.
        if (this.bm25Cache && !frameBm25Coherent(frame, this.bm25Cache)) {
            this.onCoherenceDrift('search');
            const entry = this.emptySearchEntry(query, cleanedQuery, filters, topK, searchId, idbReadMs, performance.now() - t0);
            await this.logger.append(entry);
            return { results: [], entry };
        }
        const synEnabled = this.settings.synonymExpansion;
        // Coverage = per-doc fraction of distinct query terms matched (soft-AND).
        // Multiplied into raw BM25 just before fusion (candidate loop below) when
        // settings.bm25Coverage is on, so a doc that matched only part of a multi-
        // term query is discounted vs one that matched all of it — without hard
        // AND's recall cliff. Both arrays come from one MiniSearch pass. Candidate
        // SELECTION (bm25TopIdx) still uses the un-weighted scores, so coverage
        // re-ranks the pool without shrinking recall.
        const { scores: bm25Scores, coverage: bm25Coverage, bound: bm25Bound } = this.bm25Cache!.getScoresWithCoverage(cleanedQuery, {
            boosts: this.bm25FieldBoosts(),
            // Explicit on/off so the toggles deterministically override the
            // index's baked-in fuzzy:false / prefix:false. Fuzzy "on" = edit
            // distance scaled by term length (≤2 exact / 3–5 = 1 / ≥6 = 2),
            // skipping CJK + digit-bearing tokens — edit-1 on a 2-char token or
            // a year is uncontrolled expansion, not typo tolerance (see bm25.ts
            // FUZZY_BY_LENGTH); prefix "on" = expand only the final query token
            // at ≥3 chars (see bm25.ts PREFIX_LAST_TOKEN).
            fuzzy: this.settings.fuzzyEnabled ? FUZZY_BY_LENGTH : false,
            prefix: this.settings.prefixLastToken ? PREFIX_LAST_TOKEN : false,
            // Alias-dictionary expansion (experimental, default off): mates
            // query at the eval-tuned discount; empty dictionary = inert.
            ...(synEnabled && this.synonymCache && this.synonymCache.mates.size > 0 && {
                synonyms: { map: this.synonymCache.mates, weight: SYNONYM_WEIGHT },
            }),
        });
        const bm25TopIdx = topNIndices(bm25Scores, caps.bm25, mask);
        const bm25Ms = performance.now() - bm25Start;

        // ---- S1c: recency candidate-gen ---------------------------------
        // Sorts by frontmatter `created` (mtime fallback), descending — same
        // source the reranker scores on (see topByRecency / ranker.ts). Cheap
        // (one pass + sort) and the arm is purely for *coverage* of recent notes
        // that may not surface via
        // dense/BM25; recency *blending* in the rerank still does the heavy
        // lifting via recency_weight (see ranker.ts DEFAULT_RANKING_CONFIG).
        const recencyTopIdx = this.topByRecency(orderedChunks, caps.recency, mask);

        // Await the (possibly off-thread) binary candidates now that the main
        // thread has finished BM25 + recency. binaryMs spans the dispatch→await
        // window, so it reflects time-to-availability (incl. the overlap), not
        // raw scan cost. Identical indices to the synchronous path.
        const binaryTopIdx = await binaryPromise;
        const binaryMs = performance.now() - binaryStart;

        // ---- S1 union ---------------------------------------------------
        // Per-arm counts measure the unique contribution of each arm BEFORE
        // dedup; useful for telemetry that asks "is the recency arm actually
        // contributing anything new, or is dense already covering it?"
        const unionSet = new Set<number>();
        for (const i of binaryTopIdx) unionSet.add(i);
        const binaryCount = unionSet.size;
        for (const i of bm25TopIdx) unionSet.add(i);
        const bm25Count = unionSet.size - binaryCount;
        for (const i of recencyTopIdx) unionSet.add(i);
        const recencyCount = unionSet.size - binaryCount - bm25Count;
        const candidateIndices = Array.from(unionSet);

        // ---- S2 prep: rerank vectors for the candidate union -----------
        // The whole point of stage 1 is to avoid touching every rerank vector.
        // Prefer the RESIDENT int8 block (assembled in ensureFrame, aligned to
        // the frame index space): dequantize each candidate from RAM by its
        // frame index, dropping the per-keystroke IDB round-trip (selectFetchMs,
        // the one controllable warm cost). Bit-identical to the IDB path — same
        // {q,s} bytes, same dequantizeInt8, Float64 scale — so only the fetch
        // location changes. Falls back to getEmbeddingsByIds when the block is
        // absent (cold/empty/half-migrated index), which keeps the original
        // per-id null handling. Either way fp32Maybe[i] is the vector for
        // candidateIndices[i], so the align loop below is unchanged.
        const fetchStart = performance.now();
        let fp32Maybe: Array<Float32Array | null>;
        if (residentInt8 && residentScales) {
            fp32Maybe = candidateIndices.map(idx =>
                dequantizeInt8(residentInt8.subarray(idx * embDim, (idx + 1) * embDim), residentScales[idx]));
        } else {
            const candidateIds = candidateIndices.map(i => orderedIds[i]);
            fp32Maybe = await this.store.getEmbeddingsByIds(candidateIds);
        }
        const selectFetchMs = performance.now() - fetchStart;

        // See alignCandidate: a missing/mismatched fp32 row degrades to the
        // lexical-only floor instead of dropping the candidate. The zero vector's
        // cosine is discarded — rank() floors denseScores for any lexicalOnly
        // chunk to the real-candidate min before it's used.
        const zeroFp32 = new Float32Array(queryVec.length);
        const alignStart = performance.now();
        const candidateChunks: ChunkMeta[] = [];
        const candidateFp32: Float32Array[] = [];
        const candidateBm25: number[] = [];
        const applyCoverage = this.settings.bm25Coverage;
        for (let i = 0; i < candidateIndices.length; i++) {
            const idx = candidateIndices[i];
            const aligned = alignCandidate(orderedChunks[idx], fp32Maybe[i], queryVec.length);
            if (!aligned) continue;
            candidateChunks.push(aligned.chunk);
            candidateFp32.push(aligned.missingFp32 ? zeroFp32 : (fp32Maybe[i] as Float32Array));
            // Soft-AND: discount a partial multi-term match by its coverage^P (the
            // coordination-level penalty; see BM25_COVERAGE_POW in bm25.ts). The weight
            // is 1 for single-term queries and full-coverage docs (no-op regardless of P),
            // and never 0 for a real partial match (recall-safe). P=2 hardens the discount
            // so a rare-place-token-only match can't out-rank a full-coverage answer
            // ("bars in sf"/"bars in austin" fix). rank() then TM2C2-normalizes this.
            candidateBm25.push(applyCoverage ? bm25Scores[idx] * Math.pow(bm25Coverage[idx], BM25_COVERAGE_POW) : bm25Scores[idx]);
        }
        const alignMs = performance.now() - alignStart;

        // ---- S2 score: exact cosine over the candidate set --------------
        const cosineStart = performance.now();
        const denseScoresCand = cosineScores(queryVec, candidateFp32);
        const cosineMs = performance.now() - cosineStart;

        // ---- S2 fuse + rank ---------------------------------------------
        // Same ranker with one settings-driven lever: navTitleBoost raises the
        // exact basename/alias match weight (the 2026-06-02 study's free win —
        // the entity page outranking pages that merely mention it). Everything
        // else (min-max norm, recency decay, hybrid alpha) is the v0 pipeline.
        const rankConfig = {
            ...DEFAULT_RANKING_CONFIG,
            alpha: this.settings.denseWeight,
            titleBoost: this.settings.navTitleBoost,
            // Recency (2026-06-19): ε weight, half-life, AND the definition of
            // "recent" are all settings-driven now — read fresh from this.settings
            // every query, so a UI change takes effect on the NEXT search with no
            // reindex. ε defaults to the 0.02 tiebreaker; raise it for a deliberate
            // high-recency mode. A query-time override (recencyOverride, from the
            // seek:search CLI params) is resolved here, per-call, falling back to
            // this.settings when absent — see the search() doc comment for why
            // it's a local argument and not a this.settings mutation.
            recencyEpsilon: recencyOverride?.epsilon ?? this.settings.recencyEpsilon,
            recencyHalfLifeDays: recencyOverride?.halfLifeDays ?? this.settings.recencyHalfLifeDays,
            recencyKey: this.settings.recencyKey,
            createdProp: this.settings.createdProp,
        };
        const fusionStart = performance.now();
        // Dedup over the FULL scored candidate union, not a topK-proportional
        // slice. rank() already scores+sorts every candidate (cheap arithmetic
        // over the ≤~350-chunk stage-1 union), so returning the whole sorted
        // list lets dedupByPath reliably reach topK UNIQUE notes. The old
        // topK×3 pool starved dedup whenever one large multi-chunk note
        // dominated the head (e.g. "evaluation" → Evaluation.md's many chunks
        // filled a 9-chunk pool → only 1 doc at topK=3). dedupByPath still caps
        // the displayed rows at topK.
        const rankPoolSize = candidateChunks.length;
        const { results: rankedPool, breakdown } = rank(
            candidateChunks,
            denseScoresCand,
            new Float64Array(candidateBm25),
            cleanedQuery,
            rankPoolSize,
            rankConfig,
            // Theoretical bound from the same MiniSearch pass — fusion divides
            // by it instead of the empirical max (falls back when 0; fusion.ts).
            bm25Bound,
        );
        const fusionMs = performance.now() - fusionStart;

        // ---- S3: dedup + snippet ----------------------------------------
        const results = dedupByPath(rankedPool, topK);

        // Frame-lite: rank() set content to the '' placeholder; fetch the ≤topK
        // bodies now so snippets (and the modal's click-time highlight) have text.
        await this.hydrateBodies(results);

        // Display-only confidence: each result's raw cosine expressed relative to
        // this corpus's dense background (read once per generation; absent on a
        // pre-stats index → field stays undefined and the UI shows no conf). This
        // is the ONLY place the calibration is consumed; it never touched ranking
        // above — rankedPool was already final. Independent of hydrateBodies (it
        // reads denseRaw, a ranking signal, not the body).
        const bgStats = await this.getDenseBgStats();
        if (bgStats) {
            for (const r of results) {
                r.ranking_signals.confidence = calibratedConfidence(
                    r.ranking_signals.denseRaw, bgStats.mean, bgStats.std);
            }
        }

        const snippetStart = performance.now();
        for (const r of results) r.snippet = makeSnippet(r.content, cleanedQuery, 200);
        const snippetMs = performance.now() - snippetStart;

        // ---- Telemetry ---------------------------------------------------
        // Raw-top-5 traces are reported over the CANDIDATE set, not all chunks
        // — that's what the binary/bm25 arms actually saw. The cross-vault
        // headline "did stage 1 see this candidate at all" is the candidate
        // arm counts above; rawDenseTop5/rawBm25Top5 are for "given the
        // candidate set, what did exact rerank rank highest pre-fusion."
        const rawDenseTop5 = topKByScore(denseScoresCand, candidateChunks, 5);
        const rawBm25Top5 = topKByScore(new Float64Array(candidateBm25), candidateChunks, 5);
        // Persisted ranking-trace depth. The report only ever renders the top 10
        // (generateReport → rows.slice(0, 10)), so normal runs persist 10 — a ~5×
        // smaller search row that keeps the append-only NDJSON from ballooning.
        // verboseTrace keeps the full 50-deep tail for offline pandas/eval. The field
        // name stays `fusedTop50` for log-schema stability; it now holds ≤50 rows.
        const traceDepth = this.settings.verboseTrace ? 50 : 10;
        const fusedTop50 = rankedPool.slice(0, traceDepth).map((r, i) => ({
            chunk_id: r.chunk_id,
            note_path: r.note_path,
            rank: i + 1,
            score: r.score,
            dense: r.ranking_signals.dense,
            denseRaw: r.ranking_signals.denseRaw,
            bm25: r.ranking_signals.bm25,
            recency: r.ranking_signals.recency,
            title_boost: r.ranking_signals.title_boost,
            title: r.title,
        }));

        const totalMs = performance.now() - t0;
        const entry: SearchEntry = {
            type: 'search',
            timestamp: new Date().toISOString(),
            query, topK,
            cleanedQuery, filters,
            idbReadMs: parseFloat(idbReadMs.toFixed(2)),
            binaryMs: parseFloat(binaryMs.toFixed(2)),
            selectFetchMs: parseFloat(selectFetchMs.toFixed(2)),
            alignMs: parseFloat(alignMs.toFixed(2)),
            queryEmbedMs: parseFloat(queryEmbedMs.toFixed(2)),
            iframeEmbedMs: parseFloat(iframeEmbedMs.toFixed(2)),
            cosineMs: parseFloat(cosineMs.toFixed(2)),
            bm25Ms: parseFloat(bm25Ms.toFixed(2)),
            bm25CacheHit,
            fusionMs: parseFloat(fusionMs.toFixed(2)),
            snippetMs: parseFloat(snippetMs.toFixed(2)),
            totalMs: parseFloat(totalMs.toFixed(2)),
            totalChunks: orderedChunks.length,
            binaryTopN: caps.binary,
            bm25TopM: caps.bm25,
            recencyTopK: caps.recency,
            binaryCount,
            bm25Count,
            recencyCount,
            candidateUnionSize: candidateChunks.length,
            binaryCacheHit: binaryCacheHitFlag,
            rawDenseTop5,
            rawBm25Top5,
            fusedTop50,
            alpha: rankConfig.alpha,
            recencyWeight: rankConfig.recencyEpsilon,
            recencyKey: rankConfig.recencyKey,
            // blendMode/rrfK no longer written (RRF deleted 2026-06-11; the
            // optional SearchEntry fields remain so historical rows parse).
            bm25Coverage: applyCoverage,
            prefixLastToken: this.settings.prefixLastToken,
            synonymExpansion: synEnabled,
            searchableProperties: this.settings.searchableProperties,
            headingsField: this.settings.headingsField,
            // Theoretical BM25 bound for this query (0 = bound had no opinion →
            // fusion used the empirical-max fallback). Diagnoses weak-lexical
            // queries: max(rawBm25)/bm25Bound is the channel's confidence.
            bm25Bound: parseFloat(bm25Bound.toFixed(4)),
            searchId,
        };
        await this.logger.append(entry);

        // Catch-up is deliberately NOT triggered here. Firing a foreground embed
        // per keystroke piled embed load onto the shared iOS WebContent process at
        // the exact moment it was busiest (model warm, frame built, a query in
        // flight) → jetsam. The drain now fires from onSearchSessionEnd (query
        // settled / modal closed), wired by the plugin — see drainCatchUp.

        void breakdown;
        return { results, entry };
    }

    // Stage-1 resident binary index. Loaded from IDB once per dataGeneration
    // then cached in memory; ~64 KB per 1k chunks at d=512, so 5.5k vault ≈
    // 350 KB resident. The packed buffer is one contiguous Uint8Array for
    // cache-friendly scoring (see binary.ts:concatPacked).
    private binaryIndex: {
        ids: string[];
        packed: Uint8Array;
        bytesPerVec: number;
        generation: number;
    } | null = null;

    // Ensure the resident binary index is loaded and matches dataGeneration.
    // Returns true if served from cache, false if we had to read IDB.
    // Called at the top of every search; idempotent.
    private async ensureBinaryIndex(expectedChunkCount: number): Promise<boolean> {
        if (
            this.binaryIndex &&
            this.binaryIndex.generation === this.coord.generation
        ) {
            return true;
        }
        const { ids, packed } = await this.store.listAllBinary();
        if (ids.length === 0) {
            // No binary index yet — backfill hasn't run, or vault is empty.
            // Caller treats null as "no candidates available" and returns
            // an empty result set with the appropriate log entry.
            this.binaryIndex = null;
            return false;
        }
        const bytesPerVec = packed[0].length;
        const concat = concatPacked(packed, bytesPerVec);
        this.binaryIndex = {
            ids,
            packed: concat,
            bytesPerVec,
            generation: this.coord.generation,
        };
        // Cross-check vs the chunks store — a large divergence here is the
        // canary for a half-backfilled or partially-corrupted index.
        if (Math.abs(ids.length - expectedChunkCount) > expectedChunkCount * 0.05) {
            console.warn(
                `[seek] binary index has ${ids.length} rows vs ${expectedChunkCount} chunks ` +
                `(>5% divergence) — consider Full reindex`,
            );
        }
        return false;
    }

    // Resident unified frame: the corpus in binary-index order (orphans
    // dropped) plus its aligned packed-binary buffer. Cached per dataGeneration
    // exactly like binaryIndex/bm25Cache. The frame is query-INDEPENDENT —
    // inline filters apply as a selection mask downstream (search() line ~459),
    // never by reshaping the frame — so it stays valid across queries until a
    // reindex bumps dataGeneration. Caching it removes the per-keystroke
    // listAllChunks() full-corpus read, which the 2026-06-03 timing breakdown
    // measured as ~55% of warm-search latency (idbReadMs 42 of 76 ms). The
    // chunk text/metadata held here (~1 MB/1k chunks) is needed resident anyway
    // for the filter matcher, the recency arm, and result rendering — none of
    // which can run off ids alone.
    // See the ResidentFrame interface for the row-space contract. residentInt8 is
    // null when the embeddings store is empty/inconsistent or gated off (mobile/
    // over-budget) — stage-2 then falls back to the per-id IDB read. validRows /
    // tombstoneCount track incremental liveness (Seek scaling A1).
    private frameCache: ResidentFrame | null = null;

    // Corpus dense-cosine background (dense-stats.ts), cached by dataGeneration
    // exactly like the frame: read from persisted meta on the first query after a
    // reindex, then reused with zero IDB traffic until the next bump. Drives the
    // DISPLAY-only confidence; null on a pre-stats index (full reindex needed) or
    // a sub-MIN_BG_SAMPLE corpus → confidence simply isn't shown.
    private bgStatsCache: { mean: number; std: number } | null = null;
    private bgStatsGen = -1;

    private async getDenseBgStats(): Promise<{ mean: number; std: number } | null> {
        if (this.bgStatsGen === this.coord.generation) return this.bgStatsCache;
        const m = await this.store.getMeta();
        this.bgStatsCache = (m.bgMean != null && m.bgStd != null && m.bgStd > 0)
            ? { mean: m.bgMean, std: m.bgStd }
            : null;
        this.bgStatsGen = this.coord.generation;
        return this.bgStatsCache;
    }

    // Ensure the resident frame is built and matches dataGeneration. Returns
    // the frame, or null if there's no usable index (empty vault / no binary
    // backfill yet). On a cache miss this is the one place that reads the full
    // chunk store (listAllChunks) and assembles the binary-aligned frame; on a
    // hit it returns immediately with zero IDB traffic.
    private async ensureFrame(): Promise<ResidentFrame | null> {
        // Wait out any in-flight delta so we read the fully-applied result, not a
        // half-committed one (a multi-file delta isn't one atomic transaction).
        // Loop, not check-once: a fresh delta can start before we wake, and we
        // want to read with none in flight. The delta bumps dataGeneration on
        // completion, so the cache check below then misses and rebuilds. No-op
        // during a full reindex (currentDelta is null there) — its progressive
        // "queryable as it fills" read is intended.
        while (this.coord.currentDelta) { try { await this.coord.currentDelta; } catch { /* delta logged it */ } }
        if (this.frameCache && this.frameCache.generation === this.coord.generation) {
            return this.frameCache;
        }
        // Capture the generation we're building against. A full reindex's
        // progressive read has NO currentDelta gate (the while loop above only
        // waits out incremental deltas), so it can COMPLETE — bumping the
        // generation and nulling frameCache — during the awaits below. Without
        // a re-check before the write, this frame (assembled off a PARTIALLY
        // filled store) would be stamped with the NEW generation, and the next
        // search's generation check would accept that partial frame as fresh (F2).
        const buildGeneration = this.coord.generation;
        const chunks = await this.store.listAllMeta();
        await this.ensureBinaryIndex(chunks.length);
        if (chunks.length === 0 || !this.binaryIndex) {
            // Index advanced under us (reindex completed mid-read) — this empty/partial
            // read is stale; rebuild against the current store instead of caching it.
            if (shouldDiscardPartialFrame(buildGeneration, this.coord.generation)) return this.ensureFrame();
            this.frameCache = null;
            return null;
        }
        const chunkByIdMap = new Map<string, ChunkMeta>();
        for (const c of chunks) chunkByIdMap.set(c.chunk_id, c);

        const rawIds = this.binaryIndex.ids;
        const rawPacked = this.binaryIndex.packed;
        const bytesPerVec = this.binaryIndex.bytesPerVec;

        const orderedChunks: ChunkMeta[] = [];
        const orderedIds: string[] = [];
        // Pack the filtered binary buffer directly — one byteOffset memcpy per
        // surviving row. Pre-size assuming most rows survive; trim at the end.
        const filteredPacked = new Uint8Array(rawIds.length * bytesPerVec);
        let filteredCount = 0;
        for (let i = 0; i < rawIds.length; i++) {
            const c = chunkByIdMap.get(rawIds[i]);
            if (!c) continue;
            orderedChunks.push(c);
            orderedIds.push(rawIds[i]);
            filteredPacked.set(
                rawPacked.subarray(i * bytesPerVec, (i + 1) * bytesPerVec),
                filteredCount * bytesPerVec,
            );
            filteredCount++;
        }
        if (filteredCount < rawIds.length) {
            console.warn(`[seek] dropped ${rawIds.length - filteredCount} orphan binary rows (no chunk sibling) — index may be partially backfilled`);
        }
        const activePacked = filteredCount === rawIds.length
            ? rawPacked
            : filteredPacked.subarray(0, filteredCount * bytesPerVec);

        // Resident int8 rerank tier: read every {chunk_id → QuantVec} once and
        // assemble a RAM block aligned to orderedIds (buildResidentRerankBlock),
        // so stage-2 dequantizes candidates from RAM instead of an IDB
        // round-trip per keystroke (selectFetchMs). null → stage-2 falls back to
        // the per-id IDB read (identical behaviour, just without the speedup).
        // ~388 B/vec → ~1.8 MB at 4800 chunks, smaller than the resident chunk
        // text. Built off orderedIds (already orphan-filtered) so it can never
        // misalign with activePacked.
        // B2 memory gate: on mobile or above the resident byte budget, skip the
        // resident block entirely — don't even read the int8 tier just to throw
        // it away. Stage-2 then falls back to the per-id IDB read, which is
        // relevance-identical (see residentInt8Enabled / buildResidentRerankBlock).
        // embDim isn't known until an embedding is read, so estimate it from the
        // binary tier's bytesPerVec (= ceil(d/8)); against a 16 MB budget the
        // ±7-bit slack is negligible and biases conservative.
        let resident: ReturnType<typeof buildResidentRerankBlock> = null;
        if (residentInt8Enabled(orderedIds.length, bytesPerVec * 8)) {
            const { ids: embIds, vecs: embVecs } = await this.store.listAllEmbeddings();
            const embById = new Map<string, QuantVec>();
            for (let i = 0; i < embIds.length; i++) embById.set(embIds[i], embVecs[i]);
            resident = buildResidentRerankBlock(orderedIds, embById);
        } else {
            /* intentionally empty: resident stays null (declared above), so
               stage-2 falls back to the per-id IDB read — see comment above */
        }

        // The index advanced while we were assembling (a full reindex completed):
        // discard this stale-partial frame and rebuild against the current store
        // rather than caching it under the new generation. Recursion re-enters,
        // misses the (now stale-gen) cache, and rebuilds; bounded by the rare
        // completion bump, so it converges in one extra pass.
        if (shouldDiscardPartialFrame(buildGeneration, this.coord.generation)) return this.ensureFrame();
        this.frameCache = {
            orderedChunks, orderedIds, activePacked, bytesPerVec,
            residentInt8: resident ? resident.int8 : null,
            residentScales: resident ? resident.scales : null,
            embDim: resident ? resident.embDim : 0,
            // Cold build / compaction always yields a fully-live, dense frame:
            // every row live, no tombstones. Incremental deltas mutate these.
            validRows: new Array<boolean>(orderedChunks.length).fill(true),
            tombstoneCount: 0,
            // F2: stamp the generation we BUILT against (captured pre-await and
            // re-checked just above), not one that may have advanced during the
            // awaits — guards against caching a partial frame as fresh. Equal to
            // this.coord.generation here by the re-check guard, but explicit.
            generation: buildGeneration,
        };
        return this.frameCache;
    }

    // Ensure the BM25 cache (and the synonym dictionary derived from it) match
    // dataGeneration + the current index-shape settings. Returns true if served
    // from cache. Shared by the search hot path and warmCaches() so the two
    // can never drift on what "warm" means.
    // True if the resident BM25 cache matches the live generation + index shape.
    // searchableProperties/headingsField change the INDEX shape (extra fields),
    // not a per-call option, so flipping either must force a refit. Shared by
    // ensureBm25 (skip refit) and the search hot path (skip the persisted-load
    // IDB read on a warm keystroke).
    private bm25CacheValid(orderedChunks: ChunkMeta[]): boolean {
        return !!(
            this.bm25Cache &&
            this.bm25CacheGeneration === this.coord.generation &&
            this.bm25CacheChunkCount === orderedChunks.length &&
            this.bm25CacheProps === this.settings.searchableProperties &&
            this.bm25CacheHeadings === (this.settings.headingsField || this.settings.boostedBm25)
        );
    }

    // Stamp the resident BM25 cache after (re)building it — via fit() OR a
    // persisted load — and drop the synonym dict (it derives from this index).
    private stampBm25Cache(chunkCount: number): void {
        this.bm25CacheGeneration = this.coord.generation;
        this.bm25CacheChunkCount = chunkCount;
        this.bm25CacheProps = this.settings.searchableProperties;
        this.bm25CacheHeadings = this.settings.headingsField || this.settings.boostedBm25;
        this.synonymCache = null;
    }

    private async ensureBm25(orderedChunks: ChunkMeta[]): Promise<boolean> {
        const hit = this.bm25CacheValid(orderedChunks);
        if (!hit) {
            const propsEnabled = this.settings.searchableProperties;
            // boostedBm25 boosts the headings field to 4×, which is inert unless
            // the field is actually indexed — so the preset implies heading indexing.
            const headingsEnabled = this.settings.headingsField || this.settings.boostedBm25;
            // Frame-lite: the resident frame is metadata-only, so a (rare) refit
            // pulls bodies from the chunk_body store first. This IDB read happens
            // ONLY on a true fit miss — the persisted-load fast path
            // (tryLoadPersistedBm25) needs no bodies, and a warm keystroke skips
            // ensureBm25's refit branch entirely (bm25CacheValid hit above).
            const bodies = await this.store.getBodiesMap(orderedChunks.map(c => c.chunk_id));
            // orderedChunks is dense post-filter; BM25 indexes match the
            // unified frame 1:1. If a later refactor lets holes back in,
            // BM25 fit will throw — keep this guarantee in the loader, not
            // here.
            this.bm25Cache = new MultiFieldBM25().fit(orderedChunks, bodies,
                { searchableProperties: propsEnabled, headingsField: headingsEnabled });
            this.stampBm25Cache(orderedChunks.length);
        }
        if (this.settings.synonymExpansion && !this.synonymCache) {
            // O(notes) build, trivially cheap next to fit(); df ceiling reads
            // the freshly (re)built BM25 index. Telemetry: dictionary shape +
            // what the guards dropped — silent drops are this feature's main
            // scaling hazard (adversarial review), so they must be visible.
            this.synonymCache = buildSynonymMap(orderedChunks, t => this.bm25Cache!.termDocFraction(t));
        }
        return hit;
    }

    // Cold-start fast path: when the resident BM25 cache is cold for the live
    // generation, load a persisted MiniSearch index from IDB instead of paying the
    // ~280ms fit() (on mobile the all-bodies refit is the 18.6 s freeze, not 280ms).
    // Loads when the COMPATIBLE-corpus stamp matches (analyzer/model/dim + index
    // shape — bm25StampMatches; the corpus size/timestamp are tolerated, so a
    // stale-but-compatible blob loads and is bounded-recall-stale until the next
    // delta/catch-up reconciles it — see bm25StampMatches for the safety argument).
    // A missing blob, a never-completed index (no lastIndexedAt), a stamp mismatch,
    // or a loadJSON error all leave the cache cold so the synchronous ensureBm25()
    // below refits. Awaited only on a cold keystroke (the caller guards on
    // bm25CacheValid), so warm searches never touch IDB here.
    private async tryLoadPersistedBm25(orderedChunks: ChunkMeta[]): Promise<void> {
        try {
            const blob = await this.store.getBm25();
            if (!blob) return;
            const meta = await this.store.getMeta();
            if (!meta.lastIndexedAt) return;   // only trust a COMPLETED index
            const live = buildBm25Stamp(meta, orderedChunks.length, this.settings);
            if (!bm25StampMatches(blob.stamp, live)) return;
            this.bm25Cache = new MultiFieldBM25().fromJSON(blob.json, orderedChunks, {
                searchableProperties: this.settings.searchableProperties,
                headingsField: this.settings.headingsField || this.settings.boostedBm25,
            });
            this.stampBm25Cache(orderedChunks.length);
        } catch (e) {
            // Corrupt blob / loadJSON throw → leave cache cold, ensureBm25 refits.
            console.warn('[seek] persisted BM25 load failed (refitting)', e);
        }
    }

    // Cross-device cold-start fast path (Phase 3): when this device has NO local BM25
    // blob (a fresh install, or an iOS eviction that wiped IDB), load the gzipped BM25
    // artifact another device wrote to the synced sidecar instead of refitting over
    // all bodies (the cold-start freeze). "Sync the BM25, don't rebuild it." Only
    // reached when tryLoadPersistedBm25 left the cache cold, so a normal device with a
    // local blob never runs this — the search hot path pays nothing.
    //
    // SAFE by the same property as the tolerant local gate: the producer's chunk set
    // won't exactly equal this consumer's live frame, but bm25StampMatches gates on
    // analyzer/model/dim/shape (NOT corpus size/timestamp), and fromJSON skips any
    // posting whose id isn't live + scores absent live chunks 0 — so cross-device
    // drift is bounded recall staleness, never wrong text. metaAccepts (inside
    // rankAcceptedProducers) further refuses any producer whose chunker/model/dim this
    // device can't reproduce. We try producers FRESHEST-FIRST and skip any without a
    // loadable .gz (mobile writes meta+jsonl but no artifact — emit is desktop-only —
    // and a gz can be torn/missing), falling through to the next. Any total miss →
    // return cold → ensureBm25 refits, exactly as before Phase 3 (graceful degrade).
    //
    // The precondition is a POPULATED content-id frame (orderedChunks), NOT a local
    // full reindex: the target case is a fresh/evicted device whose frame came from a
    // sidecar HYDRATE (meta.lastIndexedAt is null there) — gating on lastIndexedAt
    // would disable the feature on exactly the device it exists for. A hydrated frame's
    // content-derived ids are what map the producer's postings, so the frame is the
    // real precondition. (The local-adopt persist below no-ops on such a device —
    // persistBm25 self-gates on lastIndexedAt — so it re-loads from the sidecar each
    // cold start; still fast, still no refit. Caching it locally is a deferred perf
    // follow-up that needs hydrate to stamp lastIndexedAt+modelId.)
    private async tryLoadCrossDeviceBm25(orderedChunks: ChunkMeta[]): Promise<void> {
        if (!this.coord.sidecarOn() || !gzipAvailable()) return;
        if (orderedChunks.length === 0) return;   // need a populated frame to map postings onto
        const dir = this.coord.dir;
        if (!dir) return;
        const adapter = this.app.vault.adapter;
        const meta = await this.store.getMeta();
        const expect = expectationFor();
        const live = buildBm25Stamp(meta, orderedChunks.length, this.settings);
        const devs = await rankAcceptedProducers(adapter, dir, expect);
        for (const dev of devs) {
            try {
                const bytes = await adapter.readBinary(bm25PathFor(dir, dev)).catch(() => null);
                if (!bytes) continue;   // this producer has no .gz (e.g. mobile) — try the next-freshest
                const rec = JSON.parse(await gunzipToString(bytes)) as { json?: unknown; stamp?: unknown };
                if (typeof rec.json !== 'string') continue;
                if (!bm25StampMatches(rec.stamp, live)) continue;   // analyzer/model/dim/shape must match
                this.bm25Cache = new MultiFieldBM25().fromJSON(rec.json, orderedChunks, {
                    searchableProperties: this.settings.searchableProperties,
                    headingsField: this.settings.headingsField || this.settings.boostedBm25,
                });
                this.stampBm25Cache(orderedChunks.length);
                // Adopt locally (no-op on a hydrate-only device; see method comment).
                void this.persistBm25(orderedChunks);
                return;
            } catch (e) {
                // Corrupt gz / parse error on THIS producer → try the next-freshest.
                console.warn(`[seek] cross-device BM25 load from ${dev} failed, trying next`, e);
            }
        }
        // No producer yielded a loadable, compatible blob → cache stays cold → ensureBm25 refits.
    }

    // Persist the warmed BM25 index so the next cold start can skip fit(). Called
    // fire-and-forget from warmCaches (the quiet post-reindex/delta moment), so
    // the toJSON serialize + IDB write stay off the search hot path. Gated on a
    // COMPLETED index (lastIndexedAt non-null) so a partial reindex's index is
    // never persisted; a failed write just means the next cold start refits.
    private async persistBm25(orderedChunks: ChunkMeta[]): Promise<void> {
        try {
            if (!this.bm25Cache) return;
            const meta = await this.store.getMeta();
            if (!meta.lastIndexedAt) return;
            const stamp = buildBm25Stamp(meta, orderedChunks.length, this.settings);
            await this.store.putBm25(this.bm25Cache.toJSON(), stamp);
        } catch (e) {
            console.warn('[seek] BM25 persist failed (cold start will refit)', e);
        }
    }

    // Producer side of the cross-device BM25 artifact (Phase 3). After a FULL reindex's
    // warm builds the complete index, publish this device's gzipped BM25 blob to the
    // synced sidecar so a fresh/evicted peer (or this device after an eviction) LOADS
    // it instead of refitting over all bodies. Desktop-only — the canonical producer:
    // mobile can't reliably build BM25 here (warmCaches bails on a cold embedder) and
    // must not write a ~0.6 MB file in a jetsam window. Best-effort: a failure never
    // touches the (already-warm) local index; the consumer just refits as before. The
    // artifact is { json: toJSON(), stamp } gzipped — the consumer validates the stamp
    // (analyzer/model/dim/shape) after gunzip, so a cross-version blob is refused.
    private async emitCrossDeviceBm25(orderedChunks: ChunkMeta[]): Promise<void> {
        if (!this.coord.sidecarOn() || isMobilePlatform() || !gzipAvailable()) return;
        const dir = this.coord.dir;
        if (!dir || !this.bm25Cache) return;
        try {
            const meta = await this.store.getMeta();
            if (!meta.lastIndexedAt) return;
            const stamp = buildBm25Stamp(meta, orderedChunks.length, this.settings);
            const payload = JSON.stringify({ json: this.bm25Cache.toJSON(), stamp });
            const gz = await gzipString(payload);
            // Create the sidecar index dir first (mirrors appendTombstone/bulkAppend):
            // adapter.writeBinary does NOT create parents, so on a fresh install the
            // first BM25 emit — which can run before any shard write made the dir —
            // would ENOENT on the `.tmp` write and silently skip the cross-device blob.
            await ensureDir(this.app.vault.adapter, dir);
            await writeBytesAtomic(this.app.vault.adapter, bm25PathFor(dir, this.logger.deviceId), gz);
        } catch (e) {
            await this.logger.appendError('emitCrossDeviceBm25', e).catch(() => {});
        }
    }

    // Throttled embed-free re-persist of the resident BM25 blob. The incremental
    // delta path patches the resident cache but historically SKIPPED persistBm25, so
    // after the first in-session delta the disk blob went stale and a cold relaunch
    // refit over all bodies (pre-2026-06-20, the 18.6 s mobile freeze). Now that the
    // tolerant cold-load accepts a slightly-stale blob, this keeps it MOSTLY fresh so
    // the residual the next reconcile/catch-up patches stays tiny. Embed-free (toJSON
    // + IDB write, no model, no body read), so it is safe to fire on cold mobile.
    //
    // Leading-edge throttle: a churny phone runs hundreds of deltas/hydrates a
    // session; a 4 MB IDB write on each would be its own write-amplification + battery
    // cost. Throttling to one write per BM25_PERSIST_THROTTLE_MS is fine precisely
    // because the tolerant gate makes the dropped writes' staleness harmless. Guards
    // on the resident frame matching the live generation + a valid BM25 cache, so it
    // never serialises a stale/half-built index; persistBm25 self-gates further.
    private lastBm25PersistMs = Number.NEGATIVE_INFINITY;
    private static readonly BM25_PERSIST_THROTTLE_MS = 30_000;
    private maybePersistResidentBm25(): void {
        const rf = this.frameCache;
        if (!rf || rf.generation !== this.coord.generation) return;
        if (!this.bm25CacheValid(rf.orderedChunks)) return;
        const now = performance.now();
        if (now - this.lastBm25PersistMs < SearchOrchestrator.BM25_PERSIST_THROTTLE_MS) return;
        this.lastBm25PersistMs = now;
        void this.persistBm25(rf.orderedChunks);
    }

    // Eager cache re-warm. Every index mutation invalidates all four query-path
    // caches (frame, binary index, BM25, synonym dict), and rebuilding them
    // lazily put the full cost on the user's NEXT SEARCH — the worst place to
    // pay it (2026-06-12 live telemetry: idbRead 140–184ms + bm25 fit 277–282ms
    // on a 4.3k-chunk vault, ~465ms of a 500ms total). Firing this after the
    // delta/reindex completes moves the rebuild onto the same quiet moment the
    // flush scheduler already chose for the embed work (note-leave + 5-min
    // idle / blur / structural debounce) — by construction the user isn't
    // mid-edit when a delta runs.
    //
    // Correctness never depends on this: it walks the SAME ensureFrame /
    // ensureBm25 fills the search path uses. A search landing mid-warm at
    // worst duplicates one build; a delta landing mid-warm bumps
    // dataGeneration, the stale build fails its generation check, and that
    // delta's own warm call (or the lazy path) rebuilds.
    //
    // Cold mobile (embedder unloaded) is special: we must NOT run the eager
    // ensureBm25 build — its getBodiesMap(ALL ids) IS the 18.6 s freeze, and
    // building multi-MB resident caches the user may never query violates the
    // lazy-load contract. BUT if the resident caches are ALREADY warm for the live
    // generation (a search/delta populated them this session), the embed-free
    // re-persist is cheap (toJSON + IDB write — no embedder, no body read) and keeps
    // the cold-start blob fresh so the NEXT relaunch loads instead of refits. So on
    // cold mobile we persist-if-resident and stand down; desktop always warms.
    private warming = false;
    async warmCaches(trigger: string): Promise<void> {
        // A warm is already in flight — safe to return. invalidateBm25Cache()
        // bumps dataGeneration BEFORE its paired warmCaches() call (see the delta /
        // hydrate / reindex sites), so the active warm sees the newer generation in
        // its loop guard below and rebuilds for it. The old bare `return` here
        // silently DROPPED a superseded warm: a delta completing mid-warm bumped the
        // generation, its warmCaches() hit this guard and bailed, and the in-flight
        // warm finished one generation stale — leaving the cache cold until the next
        // search ate the full ~280ms rebuild. That was the RACE/STALE miss class
        // (~30% of cold misses in the desktop logs).
        if (this.warming) return;
        if (isMobilePlatform() && !this.embedder.loaded) {
            // Persist-if-resident, never build (see the method comment). The throttled
            // helper guards on frameCache at the live generation + a valid BM25 cache
            // (no IDB read, no model), so a cold-mobile hydrate/delta keeps the disk
            // blob fresh without paying the all-bodies build that IS the freeze.
            this.maybePersistResidentBm25();
            return;
        }
        this.warming = true;
        try {
            // Rebuild until the cache matches the LIVE generation. ensureFrame()
            // awaits IDB, so an invalidation can land mid-build and bump
            // dataGeneration past what ensureBm25() just stamped onto
            // bm25CacheGeneration; the guard then loops once more for the new state.
            // Deltas are serialized on the write mutex and debounced by the flush
            // scheduler, so this converges in 1–2 passes under real editing.
            let warmedChunks: ChunkMeta[] | null = null;
            do {
                const frame = await this.ensureFrame();
                if (!frame) break;
                await this.ensureBm25(frame.orderedChunks);
                warmedChunks = frame.orderedChunks;
            } while (this.bm25CacheGeneration !== this.coord.generation);
            // Persist the converged, warmed BM25 index so the next cold start can
            // skip fit() (fire-and-forget — the serialize + write ride this quiet
            // moment, not the search hot path; persistBm25 self-gates on a
            // completed index and swallows its own errors).
            if (warmedChunks && this.bm25CacheGeneration === this.coord.generation) {
                void this.persistBm25(warmedChunks);
                // Producer: publish the gzipped BM25 to the cross-device sidecar on a
                // full reindex only (the authoritative complete build; desktop-only —
                // see emitCrossDeviceBm25). Deltas don't republish: the consumer's
                // tolerant load + reconcile absorbs the producer's post-reindex drift.
                if (trigger === 'full-reindex') void this.emitCrossDeviceBm25(warmedChunks);
            }
        } catch (e) {
            // Lazy rebuild on next search remains the backstop — never throw.
            console.warn('[seek] cache re-warm failed (next search rebuilds lazily)', e);
        } finally {
            this.warming = false;
        }
    }

    // Pick top-K chunks by recency descending, keyed on the vault's recencyKey
    // setting through the SHARED recencyDate accessor (fusion.ts) — the same
    // source the ranker's ε-tiebreaker and browseOrder read, so the candidate
    // ARM admits the same notes the scorer rewards. (The 06-07 lesson: a key
    // change applied to the scorer but not the arm pulls in candidates the
    // ranker then scores as old.) Skips chunks with no parseable date — this
    // arm is additive recall coverage, unlike browseOrder, which must keep
    // every matched chunk.
    // Frame-lite hydration: results are spread from metadata-only frame rows, so
    // their `content` is the '' placeholder rank()/the browse path set. Fetch the
    // ≤topK bodies from chunk_body and assign them, so makeSnippet — and the
    // modal's click-time highlight (search-modal.ts reads ScoredChunk.content) —
    // see the real text. ≤topK gets, off the per-query scoring path.
    private async hydrateBodies(results: ScoredChunk[]): Promise<void> {
        if (results.length === 0) return;
        const bodies = await this.store.getBodiesByIds(results.map(r => r.chunk_id));
        for (let i = 0; i < results.length; i++) results[i].content = bodies[i] ?? '';
    }

    private topByRecency(chunks: ChunkMeta[], k: number, mask?: boolean[] | null): number[] {
        // Parse each eligible chunk's recency date once into a parallel buffer;
        // NaN marks ineligible (filtered out by mask, or no parseable date).
        // selectTopNIndices then picks the k most recent in (date desc, index
        // asc) order — identical members and order to the old
        // build-{idx,date}-objects-then-sort, minus the per-chunk object array
        // and the corpus-length sort.
        const n = chunks.length;
        const dates = new Float64Array(n);
        for (let i = 0; i < n; i++) {
            if (mask && !mask[i]) { dates[i] = NaN; continue; }
            const raw = recencyDate(chunks[i], this.settings.recencyKey, this.settings.createdProp);
            const t = raw ? Date.parse(raw) : NaN;
            dates[i] = Number.isFinite(t) ? t : NaN;
        }
        return selectTopNIndices(n, k, i => dates[i], i => !Number.isNaN(dates[i]));
    }

    private emptySearchEntry(
        query: string,
        cleanedQuery: string,
        filters: QueryFilters | null,
        topK: number,
        searchId: string,
        idbReadMs: number,
        totalMs: number,
    ): SearchEntry {
        return {
            type: 'search',
            timestamp: new Date().toISOString(),
            query, topK,
            cleanedQuery, filters,
            idbReadMs: parseFloat(idbReadMs.toFixed(2)),
            binaryMs: 0,
            selectFetchMs: 0,
            alignMs: 0,
            queryEmbedMs: 0, iframeEmbedMs: 0,
            cosineMs: 0,
            bm25Ms: 0, bm25CacheHit: false,
            fusionMs: 0, snippetMs: 0,
            totalMs: parseFloat(totalMs.toFixed(2)),
            totalChunks: 0,
            binaryTopN: POOL_FLOORS.binary,
            bm25TopM: POOL_FLOORS.bm25,
            recencyTopK: POOL_FLOORS.recency,
            binaryCount: 0, bm25Count: 0, recencyCount: 0,
            candidateUnionSize: 0,
            binaryCacheHit: false,
            rawDenseTop5: [], rawBm25Top5: [], fusedTop50: [],
            alpha: this.settings.denseWeight,
            recencyWeight: this.settings.recencyEpsilon,
            recencyKey: this.settings.recencyKey,
            searchId,
        };
    }

    // BM25 cache. Validity is determined by:
    //   1. dataGeneration — bumped by invalidateBm25Cache() (called from reindex)
    //   2. chunk count — belt-and-braces, catches a missed invalidation
    //
    // The earlier "compare orderedChunks by reference" design always missed
    // because orderedChunks is rebuilt per call; the cache hit rate stayed at
    // 0% in production logs until the new bm25CacheHit telemetry exposed it.
    private bm25Cache: MultiFieldBM25 | null = null;
    private bm25CacheGeneration = -1;
    private bm25CacheChunkCount = -1;
    private bm25CacheProps = false;    // searchableProperties the cache was fit with
    private bm25CacheHeadings = false; // headingsField the cache was fit with

    // Alias-dictionary synonym map (synonyms.ts). Built lazily alongside the
    // BM25 cache (same dataGeneration lifecycle — it reads the same chunk set
    // AND the BM25 index's df for the junk-alias ceiling) and only when the
    // synonymExpansion setting is on; toggling the setting later builds it on
    // the next search. Nulled with the BM25 cache in invalidateBm25Cache().
    private synonymCache: SynonymMap | null = null;

    // BM25 per-field boosts. Eval-tuned constants (bm25.ts DEFAULT_FIELD_BOOSTS)
    // since the per-field sliders were removed 2026-06-08 (low marginal leverage
    // once the coverage navigational boost is in). Passed to getScores per-query;
    // the cached index is boost-agnostic so there's no reindex on change.
    private bm25FieldBoosts(): Record<string, number> {
        if (!this.settings.boostedBm25) return DEFAULT_FIELD_BOOSTS;
        // "Boosted BM25" preset: lift the name/structure fields (aliases→9,
        // headings→4) and trim noisy tags (→2); title/content/properties keep
        // their defaults. aliases/tags take effect here at
        // score time; headings needs its field indexed, which ensureBm25()
        // forces on whenever boostedBm25 is set.
        return { ...DEFAULT_FIELD_BOOSTS, aliases: 9.0, tags: 2.0, headings: 4.0 };
    }

    // Despite the name, invalidates BOTH the BM25 cache AND the resident
    // binary index — they share `dataGeneration` as the source of truth, so
    // bumping it forces both to reload on the next search. Kept under the
    // old name to avoid churn in main.ts; both consumers want exactly the
    // same trigger (the underlying chunk set changed).
    invalidateBm25Cache(): void {
        this.bm25Cache = null;
        this.bm25CacheGeneration = -1;
        this.bm25CacheChunkCount = -1;
        this.synonymCache = null;
        this.binaryIndex = null;
        // Also drop the resident frame so its ~MB of chunk text doesn't linger
        // past a reindex; the dataGeneration bump already invalidates it, this
        // just frees the memory promptly.
        this.frameCache = null;
        this.coord.bumpGeneration();
    }
}

// Assemble the resident int8 rerank block for a frame. For each chunk_id in
// `orderedIds` (frame row order, already orphan-filtered by ensureFrame), copy
// its stored int8 components into a contiguous Int8Array and its scale into a
// parallel Float64Array, so block row j ↔ orderedIds[j] ↔ activePacked row j.
//
// All-or-nothing: returns null (caller falls back to the per-id IDB read) when
// the embeddings store is empty OR any frame row lacks a same-dim embedding
// sibling. That keeps stage-2 behaviour identical to the IDB path in every
// inconsistent state (putBatch writes chunk+emb+bin atomically, so a surviving
// frame row should always have an embedding — the guard is defensive against a
// half-migrated/corrupted store) and merely faster in the consistent case.
//
// Scales are Float64 (NOT Float32): s = max|vᵢ|/127 is a float64, and stage-2
// dequantizes with dequantizeInt8(int8.subarray(...), scales[j]) — the SAME
// function getEmbeddingsByIds calls on the on-disk {q,s}. Holding s at full
// float64 precision makes that dequant bit-identical to the IDB path; a Float32
// scale would round s and could shift a dequantized component, breaking the
// relevance-identical guarantee.
// Stage-2 candidate alignment decision: whether `v` (this candidate's fp32
// row) is usable, and — if not — the degraded ChunkMeta to rank it with
// instead of dropping it. A missing/mismatched row (no chunk sibling in the
// embeddings store: a half-migrated upgrade, storage corruption, or, on
// mobile — which ALWAYS takes the per-id getEmbeddingsByIds path, never the
// resident RAM block — a chunk whose vector hasn't hydrated/embedded on this
// device yet) degrades to the SAME lexical-only floor ranker.ts already
// applies to body-less title-only chunks, rather than silently dropping a
// candidate BM25 may have ranked first. Returns null only when there is no
// chunk metadata at all (nothing to rank or render). The returned chunk is a
// COPY when degraded — the caller's orderedChunks entry is shared across
// queries and must never be mutated in place.
export function alignCandidate(
    ch: ChunkMeta | undefined | null,
    v: Float32Array | null | undefined,
    queryDim: number,
): { chunk: ChunkMeta; missingFp32: boolean } | null {
    if (!ch) return null;
    const missingFp32 = !v || v.length !== queryDim;
    return { chunk: missingFp32 ? { ...ch, lexicalOnly: true } : ch, missingFp32 };
}

export function buildResidentRerankBlock(
    orderedIds: string[],
    embById: Map<string, QuantVec>,
): { int8: Int8Array; scales: Float64Array; embDim: number } | null {
    const n = orderedIds.length;
    if (n === 0) return null;
    const first = embById.get(orderedIds[0]);
    const embDim = first ? first.q.length : 0;
    if (embDim === 0) return null;
    const int8 = new Int8Array(n * embDim);
    const scales = new Float64Array(n);
    for (let j = 0; j < n; j++) {
        const qv = embById.get(orderedIds[j]);
        if (!qv || qv.q.length !== embDim) return null;   // all-or-nothing
        int8.set(qv.q, j * embDim);
        scales[j] = qv.s;
    }
    return { int8, scales, embDim };
}

// ── Resident frame (Seek scaling A1) ─────────────────────────────────────────
// The query-time row space. All tiers are aligned row-for-row: row i is
// orderedChunks[i] / orderedIds[i] / activePacked[i*bytesPerVec…] /
// residentInt8[i*embDim…] (scale residentScales[i]) / validRows[i], and the BM25
// idToIdx maps that same id→i. A single `idx` joins all of them at search time,
// so the numbering MUST stay coherent — appendFrameRows/tombstoneFrameRows below
// mutate it in lockstep with MultiFieldBM25.add/remove, and a runtime drift
// detector (applyDelta) re-checks the coupling and falls back to a full rebuild
// if it ever diverges. tombstoneCount tracks rows whose validRows flag is false
// (deleted/edited-away, awaiting compaction).
export interface ResidentFrame {
    orderedChunks: ChunkMeta[];
    orderedIds: string[];
    activePacked: Uint8Array;
    bytesPerVec: number;
    residentInt8: Int8Array | null;
    residentScales: Float64Array | null;
    embDim: number;
    validRows: boolean[];
    tombstoneCount: number;
    generation: number;
}

// One committed chunk's data needed to append it to the live frame + BM25 index.
// reindexDelta already holds exactly this at commit time (commitFile derives
// {q, bin} from the fp32 vector; fs.chunks still carry content): the change-set
// it currently discards at invalidateBm25Cache().
export interface DeltaAdd {
    chunk: Chunk;       // full chunk: content feeds BM25's body, the rest is frame meta
    q: QuantVec;        // int8 rerank tier (resident block row)
    bin: Uint8Array;    // sign-bit binary tier (activePacked row)
}

// Append committed chunks + their derived tiers to a change-set sink. Shared by
// BOTH commit paths so they surface identically into applyDelta: commitFile
// (model-embedded, derived = quantizeInt8/packSignBits) and the sidecar hydrate
// (bytes copied from a peer's shard, tiers = {q, bin}). chunks[i] aligns with
// tiers[i].
export function pushDeltaAdds(sink: DeltaAdd[], chunks: Chunk[], tiers: { q: QuantVec; bin: Uint8Array }[]): void {
    for (let i = 0; i < chunks.length; i++) {
        sink.push({ chunk: chunks[i], q: tiers[i].q, bin: tiers[i].bin });
    }
}

// Narrow a delta's adds to those NOT already live in the BM25 row space, and drop
// any within-batch duplicate ids. THE guard for the 2026-06-18 mobile meltdown:
// hydrate-sourced adds (scaling A1) can carry a chunk_id that is already live in
// the in-memory `bm`. hydrateFromSidecar's candidate selector skips a note only
// when EVERY chunk is in IDB (existingIds reads IDB, NOT the in-memory cache), so
// after an IDB↔cache divergence (a crash / partial commit) a note re-surfaces ids
// `bm` already holds as PURE adds — with no matching remove, because the
// IDB-driven deleteFile produced none for an id IDB never had. The unguarded
// loop then called bm.add() on a live id, which MiniSearch THROWS on ("duplicate
// ID"); the throw aborted applyDelta mid-patch, left frame/BM25 mis-coupled, and
// every reconcile re-tripped it → toast + rebuild loop → thermal crash.
//
// Safe to skip because chunk_id is content-addressed (cyrb53 of path+title+body):
// a live id ⟹ a byte-identical chunk ⟹ the re-add is a pure no-op (the IDB write
// already landed in putQuantized). Caller MUST run this AFTER dropping removedIds
// from `bm`, so an edit that re-commits the same content-hash id is NOT filtered
// (its id is no longer live by then) — only genuine already-live duplicates are.
// Apply the SAME filtered list to bm.add AND appendFrameRows so the two row
// spaces stay aligned (a guard on bm.add alone would desync the frame and re-trip
// the very drift detector this prevents).
export function freshDeltaAdds(adds: DeltaAdd[], isLive: (id: string) => boolean): DeltaAdd[] {
    const out: DeltaAdd[] = [];
    const seen = new Set<string>();
    for (const a of adds) {
        const id = a.chunk.chunk_id;
        if (seen.has(id) || isLive(id)) continue;
        seen.add(id);
        out.push(a);
    }
    return out;
}

// Strip body text for the metadata-only frame (mirrors index-store.stripContent).
function frameMetaOf(c: Chunk): ChunkMeta {
    const { content, ...meta } = c;
    void content;
    return meta;
}

// Append committed chunks to a live frame IN PLACE. One realloc of the contiguous
// tiers per BURST (not per chunk): a ~R-byte copy, negligible beside the O(N)
// BM25 fit() the incremental path eliminates. The binary + metadata tiers grow
// UNCONDITIONALLY — skipping them on the resident-disabled (mobile) path would
// skew the binary scan's row space from the frame's. The int8 rerank tier grows
// only when it's live (desktop/in-budget); on mobile it stays null and stage-2
// falls back to the per-id IDB read. The block may drift slightly over the byte
// budget here — the next cold rebuild / compaction re-gates it via
// residentInt8Enabled. New rows are assigned ids in array order, matching
// MultiFieldBM25.add (row = chunkCount), so frame row === bm25 idToIdx row.
export function appendFrameRows(frame: ResidentFrame, adds: DeltaAdd[]): void {
    if (adds.length === 0) return;
    const oldRows = frame.orderedIds.length;
    const k = adds.length;
    const bpv = frame.bytesPerVec;

    const newPacked = new Uint8Array((oldRows + k) * bpv);
    newPacked.set(frame.activePacked.subarray(0, oldRows * bpv), 0);
    for (let j = 0; j < k; j++) newPacked.set(adds[j].bin, (oldRows + j) * bpv);
    frame.activePacked = newPacked;

    if (frame.residentInt8 && frame.residentScales) {
        const d = frame.embDim;
        const newInt8 = new Int8Array((oldRows + k) * d);
        newInt8.set(frame.residentInt8.subarray(0, oldRows * d), 0);
        const newScales = new Float64Array(oldRows + k);
        newScales.set(frame.residentScales.subarray(0, oldRows), 0);
        for (let j = 0; j < k; j++) {
            newInt8.set(adds[j].q.q, (oldRows + j) * d);
            newScales[oldRows + j] = adds[j].q.s;
        }
        frame.residentInt8 = newInt8;
        frame.residentScales = newScales;
    }

    for (let j = 0; j < k; j++) {
        frame.orderedChunks.push(frameMetaOf(adds[j].chunk));
        frame.orderedIds.push(adds[j].chunk.chunk_id);
        frame.validRows.push(true);
    }
}

// Tombstone rows (mark not-live). The contiguous tiers keep their bytes (holes);
// validRows masks them out at selection, browse, and recency. Idempotent and
// bounds-guarded so a stale/duplicate row can't drive tombstoneCount negative.
export function tombstoneFrameRows(frame: ResidentFrame, rows: number[]): void {
    for (const row of rows) {
        if (row >= 0 && row < frame.validRows.length && frame.validRows[row]) {
            frame.validRows[row] = false;
            frame.tombstoneCount++;
        }
    }
}

// Per-row selection mask = live rows (validRows) AND the optional inline-filter
// matcher. Returns undefined ONLY when the frame is fully live AND there's no
// matcher — the byte-identical no-filter fast path. Otherwise a defined mask,
// even with no filter, so tombstones are excluded from the filter-only browse
// path (its `!mask ||` short-circuit would otherwise admit every row, including
// holes). && short-circuits so a tombstoned row's stale ChunkMeta is never read.
export function buildSelectionMask(
    orderedChunks: ChunkMeta[],
    validRows: boolean[],
    tombstoneCount: number,
    matcher: ((c: ChunkMeta) => boolean) | null,
): boolean[] | undefined {
    if (!matcher && tombstoneCount === 0) return undefined;
    const n = orderedChunks.length;
    const mask = new Array<boolean>(n);
    for (let i = 0; i < n; i++) {
        mask[i] = validRows[i] && (matcher ? matcher(orderedChunks[i]) : true);
    }
    return mask;
}

// Compaction fires when this fraction of rows are tombstones — the amortized O(N)
// renumber that keeps the frame from growing unbounded with holes (a full rebuild
// produces a dense, fully-live frame).
export const COMPACTION_TOMBSTONE_FRACTION = 0.25;
// Rows sampled by the drift detector's id↔row spot-check (the warm-build verify path checks all).
export const COHERENCE_SAMPLES = 8;
// Circuit breaker for onCoherenceDrift: a drift that re-trips within this window of
// the last rebuild is treated as PERSISTENT (not a one-off), so the expensive
// re-warm + the user Notice are suppressed to break the thrash. The cache is still
// invalidated every trip (correctness), so the next search rebuilds it lazily once.
// Without this, one bad delta drove an unbounded toast+rebuild loop (2026-06-18).
export const COHERENCE_DRIFT_COOLDOWN_MS = 30_000;

// Pure decision for that circuit breaker, extracted so it's unit-testable without a
// live SearchOrchestrator (onCoherenceDrift is a private method with heavy deps).
// invalidate is ALWAYS true — a mis-coupled frame/BM25 must never serve, and the
// drop is cheap. warm (the O(N) rebuild + the user Notice) is allowed only once the
// cooldown since the last warm has elapsed, so a re-trip inside the window degrades
// to a lazy cold rebuild instead of a toast+rebuild storm.
export function coherenceDriftDecision(
    now: number, lastWarmAt: number, cooldownMs: number,
): { invalidate: boolean; warm: boolean } {
    return { invalidate: true, warm: now - lastWarmAt >= cooldownMs };
}

// F2 guard, extracted so ensureFrame's "don't cache a partial frame as fresh"
// invariant has a named, directly-tested home. A frame assembled at buildGen must
// be discarded (rebuilt) when the index generation advanced while we were reading:
// a full reindex completing mid-assembly would otherwise let the stale-partial
// frame be cached under the NEW generation and served as fresh. True ⇒ discard
// (the call sites re-enter ensureFrame, which converges in one extra pass).
export function shouldDiscardPartialFrame(buildGen: number, currentGen: number): boolean {
    return currentGen !== buildGen;
}

// Pure decision for the plugin's drift-recovery scheduler: escalate an embed-free
// recovery for THIS index state, or not. The suppression is generation-keyed — a
// degraded index re-trips drift on every keystroke, but currentGen only advances on
// a real index mutation, so once we've escalated for a generation we don't escalate
// again until something actually changes (a later delta/reindex/invalidate/hydrate
// bumps the generation, re-arming recovery for the new state). running short-circuits
// so a recovery already in flight is never double-scheduled. health is carried for the
// caller's UI/state but is deliberately NOT consulted here — the gen key alone decides.
export interface DriftRecoveryState {
    running: boolean;
    health: 'healthy' | 'recovering' | 'degraded';
    lastRecoveryGen: number;   // generation we last escalated for; -1 = never
    currentGen: number;
}
export function driftRecoveryDecision(s: DriftRecoveryState): { schedule: boolean } {
    if (s.running) return { schedule: false };
    if (s.currentGen === s.lastRecoveryGen) return { schedule: false };
    return { schedule: true };
}

// The BM25 surface the drift detector reads — MultiFieldBM25 satisfies it
// structurally (get size / get liveCount / rowOf). Decoupled as an interface so
// the detector is a pure, engine-free unit test target.
export interface RowSpaceProbe {
    readonly size: number;       // R: rows incl tombstones (== frame.orderedChunks.length)
    readonly liveCount: number;  // live (non-tombstoned) rows
    rowOf(id: string): number | undefined;
}

// Row-space coherence between the frame and the BM25 index — THE fragile invariant
// of the incremental path. At query time a single `idx` indexes orderedChunks[idx]
// / activePacked[idx] / residentInt8[idx] / bm25Scores[idx] together, so if their
// numbering ever drifts, search returns plausible-but-wrong scores: silent,
// in-bounds, relevance-degrading. This makes drift LOUD. O(1) structural checks
// always; a sampled (full only on the warm-build verify path) idToIdx[orderedIds[i]]===i spot-check on
// top. A false return is the trip — the caller logs + drops to a full rebuild,
// converting silent drift into a visible "rebuilt from scratch" event.
export function frameBm25Coherent(frame: ResidentFrame, probe: RowSpaceProbe, full = false): boolean {
    const n = frame.orderedChunks.length;
    // O(1): row counts + live counts must agree across both structures.
    if (probe.size !== n) return false;
    if (frame.orderedIds.length !== n || frame.validRows.length !== n) return false;
    if (probe.liveCount !== n - frame.tombstoneCount) return false;
    if (n === 0) return true;
    // id↔row spot-check over LIVE rows (tombstone holes carry no BM25 entry).
    const samples = full ? n : Math.min(COHERENCE_SAMPLES, n);
    for (let s = 0; s < samples; s++) {
        const i = full ? s : Math.floor((s * (n - 1)) / Math.max(1, samples - 1));
        if (!frame.validRows[i]) continue;
        if (probe.rowOf(frame.orderedIds[i]) !== i) return false;
    }
    return true;
}

// Identity of a persisted MiniSearch index — what must match for a stored blob
// to be loadable instead of refit. Two classes of input:
//   - analyzerVersion: a build-time content hash of the analyzer sources
//     (bm25.ts + tokenize.ts + prop-normalize.ts + MiniSearch version). Collapses
//     EVERY code/constant input that decides the token space — tokenizer,
//     processTerm, depluralize tables, field list, boosts, bm25 params,
//     combineWith — into one string. A loaded index uses its OWN postings but the
//     CURRENT analyzer (loadJSON re-supplies it and does NOT check it matches), so
//     a changed analyzer must invalidate the blob; the hash does that automatically.
//   - the runtime values a static hash can't see: model/dim (a model swap drops
//     chunks) and the two index-shape toggles props/headings — all GATED.
//   - lastIndexedAt + chunkCount: WRITTEN for diagnostics (and a possible future
//     tighter gate) but NOT gated as of 2026-06-20 — see bm25StampMatches for why
//     a stale-but-compatible blob is safe to load (content-derived ids → drift is
//     invisibility, not error). They were the churn fields that forced the freeze.
// A GATED field differing ⇒ refit (relevance-identical). generation is NOT here: it
// resets to 0 every process, so a disk blob would never match a fresh session.
export interface Bm25PersistStamp {
    analyzerVersion: string;
    modelId: string;
    embeddingDim: number;
    lastIndexedAt: string;
    chunkCount: number;
    props: boolean;
    headings: boolean;
}

export function buildBm25Stamp(meta: MetaConfig, chunkCount: number, settings: SeekSettings): Bm25PersistStamp {
    return {
        analyzerVersion: ANALYZER_VERSION,
        modelId: meta.modelId ?? LEGACY_ENGLISH_MODEL_ID,
        embeddingDim: meta.embeddingDim,
        lastIndexedAt: meta.lastIndexedAt ?? '',
        chunkCount,
        props: settings.searchableProperties,
        headings: settings.headingsField || settings.boostedBm25,
    };
}

export function bm25StampMatches(stored: unknown, live: Bm25PersistStamp): boolean {
    if (!stored || typeof stored !== 'object') return false;
    const s = stored as Partial<Bm25PersistStamp>;
    // TOLERANT GATE (2026-06-20): compare only the five CORRECTNESS-critical fields.
    // lastIndexedAt + chunkCount are deliberately NOT compared — they change on every
    // delta/hydrate, so gating on them rejected the blob on every churn event and
    // forced a cold all-bodies refit (the 18.6 s mobile freeze). Dropping them admits
    // a stale-but-compatible blob, which is SAFE by construction: chunk_id is content-
    // derived (chunkIdFor), so an edited chunk gets a NEW id, and getScoresWithCoverage
    // skips any posting whose id isn't in the live frame (bm25.ts `if (idx===undefined)
    // continue`) while a live chunk absent from the postings keeps its 0 default. So
    // staleness = edited/new chunks briefly lexical-invisible (reconciled by the next
    // delta / catch-up), NEVER wrong text or a crash. The five retained fields are
    // orthogonal to corpus size/timestamp, so a genuinely incompatible blob (changed
    // analyzer/model/dim/index-shape) still rejects in its own field. buildBm25Stamp
    // still WRITES both churn fields (diagnostics + a future tighter gate); we just
    // stop gating on them. tryLoadPersistedBm25 keeps its `meta.lastIndexedAt` presence
    // check, so a never-completed index is still never loaded.
    return s.analyzerVersion === live.analyzerVersion
        && s.modelId === live.modelId
        && s.embeddingDim === live.embeddingDim
        && s.props === live.props
        && s.headings === live.headings;
}

// File-level dedup: walk the ranked pool, keep the first (highest-scoring)
// chunk per note_path, stop at topK unique notes. Shared by the main search
// path and the filter-only fast path.
function dedupByPath(rankedPool: ScoredChunk[], topK: number): ScoredChunk[] {
    const seenPaths = new Set<string>();
    const out: ScoredChunk[] = [];
    for (const r of rankedPool) {
        if (seenPaths.has(r.note_path)) continue;
        seenPaths.add(r.note_path);
        out.push(r);
        if (out.length >= topK) break;
    }
    return out;
}

function topKByScore(scores: Float64Array, chunks: ChunkMeta[], k: number): Array<{ chunk_id: string; score: number }> {
    const indices = Array.from({ length: scores.length }, (_, i) => i);
    indices.sort((a, b) => scores[b] - scores[a]);
    return indices.slice(0, k).map(i => ({ chunk_id: chunks[i].chunk_id, score: scores[i] }));
}

// Project a chunk to the plain text a reader would see, collapsing markdown
// link/embed syntax to its display text. Run BEFORE the snippet window is
// chosen so a query term that exists only inside a URL (e.g. the `style` in
// `…/design_asset_style_ab?…`) can't drag the window into the URL and slice the
// link into unreadable fragments — we fall back to the note's opening prose
// instead. Snippet-only: the raw content is still what BM25/dense indexed, so a
// URL-only term keeps the note findable; this just governs what's shown.
//   ![[img]] / ![alt](url) → ''        (embeds carry no readable text)
//   --- / *** / ___        → ''        (thematic breaks — this vault puts one
//                                        under every H1, so top-of-note chunks
//                                        would otherwise lead with `--- …`)
//   [[target|alias]]       → alias     (Obsidian's own display rule)
//   [[target#section]]     → target
//   [text](url)            → text
function snippetPlainText(md: string): string {
    return md
        .replace(/!\[\[[^\]]*?\]\]/g, '')                       // ![[image]] embed
        .replace(/!\[[^\]]*?\]\([^)]*?\)/g, '')                 // ![alt](url) image
        .replace(/^[ \t]*([-*_])(?:[ \t]*\1){2,}[ \t]*$/gm, '') // --- / *** / ___ rule
        .replace(/\[\[([^\]]+?)\]\]/g, (_m, inner: string) =>
            inner.includes('|') ? inner.slice(inner.lastIndexOf('|') + 1) : (inner.split('#')[0] || inner))
        .replace(/\[([^\]]*?)\]\([^)]*?\)/g, '$1');  // [text](url) → text
}

function makeSnippet(content: string, query: string, maxLen: number): string {
    const text = snippetPlainText(content);
    const lower = text.toLowerCase();
    const q = query.toLowerCase().split(/\s+/).filter(Boolean);
    let best = -1;
    for (const tok of q) {
        const idx = lower.indexOf(tok);
        if (idx !== -1 && (best === -1 || idx < best)) best = idx;
    }
    const start = best === -1 ? 0 : Math.max(0, best - 40);
    const end = Math.min(text.length, start + maxLen);
    let snippet = text.slice(start, end).replace(/\s+/g, ' ').trim();
    if (start > 0) snippet = '…' + snippet;
    if (end < text.length) snippet = snippet + '…';
    return snippet;
}
