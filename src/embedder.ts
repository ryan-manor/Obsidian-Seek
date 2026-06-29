// LocalEmbedder — thin wrapper around the iframe runtime.
// Model is the onnx-community off-the-shelf PTQ export at q4. The QAT
// workstream is killed: the 2026-05-15 embedder bake-off found q4 ≈ q8 on
// this vault (fused NDCG@10 Δ=0.0005), and Phase 5 showed locally
// re-quantizing Google's QAT-bf16 weights to q4 *loses* to off-the-shelf
// PTQ-q4 by 0.0146 NDCG@10 (QAT calibration is tuned for Q8 rounding).
// Two orthogonal throughput issues, don't conflate them (the bake-off did):
//   1. The "13× regression" = the *unfused QAT graph* (7,399 nodes) vs the
//      *fused PTQ graph* (1,332 nodes, 24× MultiHeadAttention). Fixed by
//      shipping this fused PTQ build.
//   2. A separate ~2× q4-vs-q8 WebGPU gap that exists even on the fully
//      fused graph — ORT-Web's MatMulNBits M>1 kernel. Fixed NOT here but
//      by the transformers.js v4 floor (see iframe-runner.ts
//      TRANSFORMERS_VERSION). On v4, q4 is the fastest *and* lightest
//      config; PTQ alone is necessary but not sufficient.
// See Seek Model Performance.md / Seek Embedder Bake-Off Results.md.

import type { Device, RequestedDevice, Dtype, LoadEntry, InitEntry } from './types';
import { snapshotMemory, memoryDelta, LOG_SCHEMA_VERSION } from './types';
import { ACTIVE_MODEL_SPEC } from './model-registry';
import {
    IframeRunner,
    TRANSFORMERS_VERSION,
    WARMUP_BATCH_SIZES,
    SEQ_BUCKETS,
    QUERY_SEQ_BUCKETS,
    type AppLocalFetchResult,
    type RawProfile,
    type IframeEvent,
    type IframeInit,
} from './iframe-runner';

// localStorage namespace for the warmup-skip fingerprint. Bumping the prefix
// to "v2" (etc.) is a fast cache-wipe if the schema ever breaks.
const WARMUP_FP_KEY = 'seek:warmup-fingerprint:v1';

// Fingerprint composition. Any change invalidates the stored fingerprint and
// forces a one-time re-warmup that re-writes the new value. The pieces:
//   modelId      — different models compile different graphs / shaders
//   dtype        — q4 vs fp32 use different MatMulNBits / MatMul kernels
//   transformers — different ORT-Web revs emit different WGSL bodies
//   batch×seq    — the exact grid the iframe warms, mirrored via the parent
//                  exports so they can't drift
function warmupFingerprint(modelId: string, dtype: Dtype, revision: string | null): string {
    return [
        modelId,
        revision ?? 'main',   // a revision bump fetches different bytes → must re-warm
        dtype,
        TRANSFORMERS_VERSION,
        WARMUP_BATCH_SIZES.join(','),
        SEQ_BUCKETS.join(','),
        QUERY_SEQ_BUCKETS.join(','),   // query-path floors (8,16) are warmed too; grid change must re-warm
    ].join('|');
}

// localStorage is sync, fast, and reliable in the Obsidian renderer process.
// Wrapped because private-browsing modes and CSP can disable it; the failure
// mode is "warmup runs every time" which is exactly today's behavior.
function readWarmupFingerprint(): string | null {
    try { return localStorage.getItem(WARMUP_FP_KEY); }
    catch { return null; }
}
function writeWarmupFingerprint(fp: string): void {
    try { localStorage.setItem(WARMUP_FP_KEY, fp); }
    catch { /* swallow — storage unavailable; skip-cache is best-effort */ }
}

// Public result types — explicit so callers can choose between the cheap
// vector-only path and the diagnostic path that includes iframe-side timing.
export interface EmbedTimed {
    vector: Float32Array;
    iframeLatencyMs: number;
    // True when served from the query-embed LRU (no iframe work; latency ≈ 0).
    cacheHit?: boolean;
}

export interface EmbedBatchTimed {
    vectors: Float32Array[];
    iframeLatencyMs: number;
}

// ── Phase-1 remote-fetch test (REVERSIBLE) ──────────────────────────
// Points Seek at the trimmed q4 model on the HF CDN to test runtime
// fetch. REVERT: restore the onnx-community id below + LOCAL_MODEL.enabled=true.
// ── Model 2026-06-11: granite-embedding-97m-multilingual-r2, the ONLY model ──
// ml97 is english-r2's ModernBERT sibling — same 384-d, CLS pooling, no
// query/doc prompts — shipped as a self-hosted GBQ-int4 export (q4 body +
// GatherBlockQuantized int4 embedding table): 61 MB vs english-r2's 99.5 MB,
// WebGPU-native Gather (ORT ≥1.23, in the tx.js 4.2.0 pinned ort-web).
// Three-way relevance gate 2026-06-10 ([[Seek MultiLanguage]]): quantization
// free on English (±0.003), Belebele dense tax ≤0.021 (worst ko); english-r2
// kept a −2 pt code edge but collapses on non-English (Belebele dense
// 0.29–0.72 vs ml97's 0.85–0.89). The per-vault model choice that briefly
// existed (2026-06-10..11) was deleted: one model that can't be wrong for a
// vault beats a setting whose wrong value silently degrades dense ranking.
// The graph is the sentence-transformers flavor — tx.js's feature-extraction
// pipeline reads its `token_embeddings` output via the last_hidden_state??
// logits??token_embeddings fallback chain and applies CLS+normalize itself
// (verified: CLS(token_embeddings) ≡ its baked sentence_embedding, cos 1.0).
// The HF repo Seek embeds with — and the index's drift-identity stamp. Derived
// from the active registry spec so a code-level model switch (model-registry.ts
// ACTIVE_MODEL_KEY) changes this string; the model-vs-index drift check reads it
// to route every device to a full reindex. See model-registry.ts for the why.
export const MODEL_ID = ACTIVE_MODEL_SPEC.repo;

// The pinned commit sha (or null = track main) the index is built against. Stamped
// into sidecar meta + checked by the version gate so a revision change refuses
// cross-revision hydration (F10), and threaded into the transformers.js fetch so
// the model bytes are reproducible regardless of when/where a device indexes.
export const MODEL_REVISION = ACTIVE_MODEL_SPEC.revision;

// Pre-2026-06-10 indexes carry no modelId stamp in meta and were english-r2
// by construction — the drift checks (main.ts warnOnModelIndexDrift, the
// search.ts incremental guard) read this constant to classify an unstamped
// index as foreign and route the user to a full reindex.
export const LEGACY_ENGLISH_MODEL_ID = 'onnx-community/granite-embedding-small-english-r2-ONNX';

// ── Phase-5 local-model test override (vocab-trim grafted model) ─────
// When enabled, Seek loads the LOCAL trimmed+fused ONNX from the vault
// instead of streaming the stock model from the HF CDN. Reversible:
// set enabled=false (or restore main.js.bak-prelocalswap + toggle the
// plugin). FLIP q4<->fp32 = change `dtype` below:
//   'q4'   -> <vaultRelPath>/onnx/model_q4.onnx (~102 MB trim / ~197 MB stock)
//   'fp32' -> <vaultRelPath>/onnx/model.onnx    (~626 MB, relevance ceiling)
//   (vaultRelPath is the constant below — 'seek-test-model', NOT 'seek-model')
// Relevance gate: fp32 ties the q4 anchor (0.6432 vs 0.6442); q4 CPU-EP
// floor 0.6296 — this manual WebGPU test pins the real q4 number.
export const LOCAL_MODEL = {
    enabled: false,   // Phase-1: false → use remote MODEL_ID (HF CDN). REVERT: true
    vaultRelPath: 'seek-test-model',   // Q2 side-load: visible vault-root folder (iOS Files can't see .obsidian/)
    dtype: 'q4' as Dtype,   // <<< ONE-LINE SWITCH: 'q4' | 'fp32'
};
const PLUGIN_VERSION = '0.0.1';

// Output vector dimension. SINGLE SOURCE OF TRUTH = the active model spec's
// `dim` (model-registry.ts). The iframe's OUTPUT_DIM and the sidecar record
// stride (sidecar.ts Q_BYTES/SIGN_BYTES) derive from this SAME spec field, so a
// model swap can't leave them silently disagreeing (the old failure mode: write
// N-d vectors into a 384-byte stride). granite-r2 is 384-d native (NO
// Matryoshka), so the slice in iframe-runner.ts is a pass-through; a Matryoshka
// model would declare a smaller `dim` and the slice truncates to it. When you
// change the active model, bump DB_VERSION in index-store.ts so dim-incompatible
// local vectors are dropped. See dim-consistency.test.ts for the invariant.
export const EMBEDDING_DIM = ACTIVE_MODEL_SPEC.dim;

export class LocalEmbedder {
    private runner = new IframeRunner();
    private _device: Device = 'wasm';
    private _dtype: Dtype = 'q4';
    private _loaded = false;

    // Memoized iframe-init. onload fires init() un-awaited; a search that races
    // ahead coalesces onto this same promise inside load() (see init()/load()),
    // so one iframe is built and the search-before-init race can't reject with
    // 'iframe not initialized'. Nulled on a failed/not-ready init (retry) and by
    // recycle()/teardown() (the iframe they rebuild invalidates the cached entry).
    private _initPromise: Promise<InitEntry> | null = null;

    // Single-flight latch for load(). Coalesces CONCURRENT load() calls onto one
    // in-flight runner.load() so two ~250 MB model loads can't run at once — a
    // direct jetsam trigger on mobile. The window is real: ensureModelLoaded nulls
    // its modelLoadPromise on a load failure (to allow retry), and seek-unload-model
    // nulls it too, so a third caller arriving with loaded===false could otherwise
    // start a second concurrent load(). Unlike _initPromise this is CLEARED after
    // settle, not retained: load() is not idempotent across a model switch, so the
    // latch coalesces concurrency — it does not cache a result.
    private _loadPromise: Promise<LoadEntry> | null = null;

    // WebGPU device lifecycle events pushed by the iframe (device-created /
    // device-lost / uncaptured-error). Stored here, not just assigned to the
    // runner, because teardown() swaps the runner instance and the wiring
    // must survive it. recycle() reuses the runner, so it keeps the handler.
    private _onIframeEvent: ((event: IframeEvent) => void) | null = null;
    set onIframeEvent(cb: ((event: IframeEvent) => void) | null) {
        this._onIframeEvent = cb;
        this.runner.onEvent = cb;
    }

    // Last successful load() args, replayed by recycle() to rebuild the
    // iframe pipeline without re-deriving device/dtype/model selection.
    private _lastRequested: RequestedDevice = 'auto';
    private _lastReqDtype: Dtype = 'q4';
    private _lastModelId: string = MODEL_ID;
    private _lastRevision: string | null = MODEL_REVISION;

    // Query-embed LRU memo. Keyed on the exact (cleaned) query text; holds
    // vectors for the CURRENTLY-loaded pipeline only — cleared on every load()/
    // recycle() because a different model/dtype produces different vectors.
    // Map insertion order = LRU order: a hit re-inserts to the tail, inserts
    // past the cap evict the head. Single-query path only (not embedBatch —
    // indexing never repeats a chunk within a generation). The cached vector is
    // returned by reference; query vectors are treated read-only everywhere
    // (cosineScores / scoreAsymmetric / rank all read, none mutate).
    private queryEmbedCache = new Map<string, Float32Array>();
    private static readonly QUERY_EMBED_CACHE_MAX = 128;

    get device(): Device { return this._device; }
    get dtype(): Dtype { return this._dtype; }
    get loaded(): boolean { return this._loaded; }
    // Model id of the current/last pipeline — what the orchestrator stamps
    // into index meta so a later load can detect model-vs-index drift.
    get modelId(): string { return this._lastModelId; }

    // Map a runner IframeInit → loggable InitEntry. Shared by init() and recycle()
    // so the memo repopulated after a recycle is the same shape callers expect.
    private toInitEntry(r: IframeInit): InitEntry {
        return {
            type: 'init',
            timestamp: new Date().toISOString(),
            schemaVersion: LOG_SCHEMA_VERSION,
            buildTimestamp: r.buildTimestamp,
            transformersVersion: r.transformersVersion,
            cdnUrl: r.cdnUrl,
            iframeReady: r.ready,
            initMs: r.initMs,
            pluginVersion: PLUGIN_VERSION,
            error: r.error,
        };
    }

    // Memoized + idempotent. Concurrent callers (onload's un-awaited fire + a
    // racing load()) share one runner.init(), so the iframe is built exactly once.
    // The memo is RETAINED on success and NULLED on a not-ready result (runner.init
    // returns {ready:false,error} rather than throwing) so the next caller retries
    // a fresh build — mirrors ensureModelLoaded's `modelLoadPromise = null` idiom.
    init(): Promise<InitEntry> {
        if (this._initPromise) return this._initPromise;
        const p = (async (): Promise<InitEntry> => {
            const r = await this.runner.init();
            if (!r.ready) this._initPromise = null;   // don't pin a dead iframe — allow retry
            return this.toInitEntry(r);
        })().catch((e) => {
            // runner.init() swallows build errors today, but if that ever changes
            // don't leave a rejected promise pinned in the memo.
            this._initPromise = null;
            throw e;
        });
        this._initPromise = p;
        return p;
    }

    // Single-flight wrapper around loadImpl (see _loadPromise): coalesces
    // concurrent callers onto one in-flight load, then clears the latch so a
    // later model switch loads fresh (load is not idempotent across models).
    async load(requested: RequestedDevice, dtype: Dtype, modelIdOverride?: string, revision?: string | null): Promise<LoadEntry> {
        if (this._loadPromise) return this._loadPromise;
        const p = this.loadImpl(requested, dtype, modelIdOverride, revision);
        this._loadPromise = p;
        try { return await p; }
        finally { this._loadPromise = null; }
    }

    private async loadImpl(requested: RequestedDevice, dtype: Dtype, modelIdOverride?: string, revision?: string | null): Promise<LoadEntry> {
        // Self-sufficient: ensure the iframe is up before any RPC. With onload no
        // longer awaiting init(), a load() triggered by an early search coalesces
        // onto the same memoized init here instead of hitting runner.send()'s
        // 'iframe not initialized' rejection. Idempotent + memoized, so once the
        // iframe is live this is a single resolved-promise await (~0 ms).
        const initEntry = await this.init();
        if (!initEntry.iframeReady) {
            throw new Error(`iframe init failed: ${initEntry.error ?? 'unknown'}`);
        }
        const memBefore = await snapshotMemory();
        const start = performance.now();

        // Warmup-skip fingerprint. If a previous successful WebGPU load
        // wrote the same composite fingerprint to localStorage, the iframe
        // can skip the ~1 s warmup loop entirely (Dawn's disk shader cache
        // is hot for this exact grid). On any mismatch (model swap, dtype
        // flip, transformers version bump, grid change, cleared storage)
        // the iframe pays the warmup once and we re-write the fingerprint.
        const effectiveModelId = modelIdOverride ?? MODEL_ID;
        // undefined = caller didn't specify → use the active pin; explicit null = track main.
        const effectiveRevision = revision !== undefined ? revision : MODEL_REVISION;
        const expectedFp = warmupFingerprint(effectiveModelId, dtype, effectiveRevision);
        const skipWarmup = readWarmupFingerprint() === expectedFp;

        // Remember the args so recycle() can rebuild the same pipeline.
        this._lastRequested = requested;
        this._lastReqDtype = dtype;
        this._lastModelId = effectiveModelId;
        this._lastRevision = effectiveRevision;

        let result;
        try {
            result = await this.runner.load(effectiveModelId, requested, dtype, skipWarmup, effectiveRevision);
        } catch (e) {
            this._loaded = false;
            throw e;
        }

        const coldStartMs = parseFloat((performance.now() - start).toFixed(2));
        const memAfter = await snapshotMemory();

        this._device = result.device;
        this._dtype = result.dtype;
        this._loaded = true;
        this.queryEmbedCache.clear();   // vectors are pipeline-specific; new load invalidates them

        const checks: string[] = [];
        let pass = true;

        // Pass threshold is the 5 s mobile cold-start target from the design doc.
        if (coldStartMs < 5000) checks.push(`✅ cold-start ${coldStartMs.toFixed(0)} ms < 5000 ms`);
        else { pass = false; checks.push(`❌ cold-start ${coldStartMs.toFixed(0)} ms ≥ 5000 ms`); }

        if (result.webgpuAttempted && result.webgpuError) checks.push(`⚠️ WebGPU attempted: ${result.webgpuError}`);
        else if (result.webgpuAttempted && result.device === 'webgpu') checks.push(`✅ WebGPU loaded successfully (dtype=${result.dtype})`);
        // Non-null only when the WebKit glue override applied (iOS path):
        // which ort-wasm variant is actually resident matters for the memory
        // investigation (asyncify = 23.6 MB binary + eager-compile multiplier
        // vs jspi 14.5 MB — see Seek Mobile WebGPU Investigation).
        if (result.glue) checks.push(`ℹ️ webkit glue: ${result.glue}`);

        // Surface the warmup decomposition inline so the report doesn't
        // need a separate parser pass. Three cases:
        //   1. WebGPU + warmup ran: warmupMs is the cost; pre-warmup is the
        //      I/O + parse + device-init floor.
        //   2. WebGPU + warmup skipped: fingerprint cache hit; expect
        //      coldStartMs to drop by ~the previous warmupMs.
        //   3. WASM fallback: warmup doesn't apply; we say so explicitly so
        //      anyone reading the log doesn't conflate cases 2 and 3.
        if (result.warmupSkipped) {
            checks.push(`ℹ️ warmup SKIPPED (fingerprint cache hit) · cold-start ${coldStartMs.toFixed(0)} ms is read + parse + device init only`);
        } else if (result.warmupMs != null) {
            const preWarmup = coldStartMs - result.warmupMs;
            checks.push(
                `ℹ️ warmup ${result.warmupMs.toFixed(0)} ms · pre-warmup ${preWarmup.toFixed(0)} ms ` +
                `(read + parse + device init)`,
            );
        } else if (result.device === 'wasm') {
            checks.push(`ℹ️ warmup N/A (WASM path)`);
        }

        // Persist the fingerprint on any successful WebGPU load. Idempotent:
        // if we got here via a skip, the value already matches; if we ran
        // the warmup, we record that this grid is now warm. Writing on every
        // success means a future config change that lands in this branch
        // (e.g. model swap then immediate re-load) self-heals on the load
        // that *did* run the warmup, not on the load that paid for the swap.
        if (result.device === 'webgpu') {
            writeWarmupFingerprint(expectedFp);
        }

        if (result.device !== requested && requested !== 'auto') {
            pass = false;
            checks.push(`❌ requested ${requested} but used ${result.device}`);
        } else {
            checks.push(`✅ running on ${result.device} (dtype=${result.dtype})`);
        }

        const delta = memoryDelta(memBefore, memAfter);
        if (delta.heapDeltaMB != null) checks.push(`ℹ️ heap delta: +${delta.heapDeltaMB.toFixed(1)} MB`);
        if (delta.storageDeltaMB != null) checks.push(`ℹ️ storage delta: +${delta.storageDeltaMB.toFixed(1)} MB (IDB + Cache)`);
        if (delta.heapDeltaMB == null) checks.push(`ℹ️ heap unavailable (performance.memory not exposed on this platform — likely iOS WebKit)`);

        return {
            type: 'load',
            timestamp: new Date().toISOString(),
            requestedDevice: requested,
            actualDevice: result.device,
            dtype: result.dtype,
            embeddingDim: EMBEDDING_DIM,
            coldStartMs,
            warmupMs: result.warmupMs != null
                ? parseFloat(result.warmupMs.toFixed(2))
                : null,
            warmupSkipped: result.warmupSkipped,
            heapBeforeMB: memBefore.heapMB,
            heapAfterMB: memAfter.heapMB,
            heapDeltaMB: delta.heapDeltaMB,
            storageBeforeMB: memBefore.storageMB,
            storageAfterMB: memAfter.storageMB,
            storageDeltaMB: delta.storageDeltaMB,
            webgpuAttempted: result.webgpuAttempted,
            webgpuFailed: result.webgpuAttempted && result.device !== 'webgpu',
            webgpuError: result.webgpuError,
            pass,
            checks,
        };
    }

    async embed(text: string): Promise<EmbedTimed> {
        if (!this._loaded) throw new Error('Model not loaded.');
        const hit = this.queryEmbedCache.get(text);
        if (hit !== undefined) {
            this.queryEmbedCache.delete(text);     // re-insert at tail → most-recent
            this.queryEmbedCache.set(text, hit);
            return { vector: hit, iframeLatencyMs: 0, cacheHit: true };
        }
        const r = await this.runner.embed(text);
        this.queryEmbedCache.set(text, r.vector);
        if (this.queryEmbedCache.size > LocalEmbedder.QUERY_EMBED_CACHE_MAX) {
            this.queryEmbedCache.delete(this.queryEmbedCache.keys().next().value!);  // evict LRU head
        }
        return { vector: r.vector, iframeLatencyMs: r.latencyMs, cacheHit: false };
    }

    // Tear down and rebuild the ENTIRE iframe to reset ORT-Web's WebGPU state.
    // The forward path disposes its output tensor every call, and ORT's buffer
    // accounting climbs across a long reindex until a SafeInt(int32) guard
    // overflows (~2100–2200 granite chunks → "OrtRun integer overflow",
    // 2026-06-03 diagnosis); once tripped, every subsequent OrtRun throws and
    // the rest of the vault is skipped. The overflow lives at the WebGPU
    // *device* level, NOT the session level — a fresh InferenceSession on the
    // same device inherits the poisoned state (verified 2026-06-03: 667
    // session-only recycles each retried and still threw the same overflow).
    // The iframe owns the device, so we dispose the whole iframe (kills the
    // device) and rebuild it. Heavier (~2–4 s: srcdoc bootstrap + device init +
    // cached model reload, warmup skipped via the Dawn shader cache) but it
    // fires only once per ~2100 chunks. Replays the last load() args; leaves
    // _loaded true on success, false (and throws) if the rebuild fails.
    async recycle(): Promise<void> {
        this._loaded = false;
        this.runner.dispose();
        // Set the rebuild promise into the init memo BEFORE the first await so a
        // CONCURRENT embedder.init() coalesces onto it (init() returns the memo)
        // instead of racing a second runner.init() through the gap the old
        // `_initPromise = null` opened — which appends a second iframe and leaks
        // the older listener (runner.init only guards SEQUENTIAL re-entry). The
        // memo is RETAINED on success (the live rebuilt iframe), matching init().
        const rebuild = (async (): Promise<InitEntry> => {
            const init = await this.runner.init();
            if (!init.ready) {
                this._initPromise = null;   // don't pin a dead iframe — allow retry
                throw new Error(`recycle: iframe rebuild failed: ${init.error}`);
            }
            return this.toInitEntry(init);
        })();
        this._initPromise = rebuild;
        await rebuild;
        try {
            const result = await this.runner.load(
                this._lastModelId, this._lastRequested, this._lastReqDtype, /* skipWarmup */ true, this._lastRevision,
            );
            this._device = result.device;
            this._dtype = result.dtype;
            this._loaded = true;
            this.queryEmbedCache.clear();   // rebuilt pipeline → drop any memoized vectors
        } catch (e) {
            // init() succeeded but load() failed: without this the memo pins a
            // resolved "ready" entry pointing at an iframe whose model never
            // loaded, so the next init() short-circuits to a false-ready.
            this._initPromise = null;
            throw e;
        }
    }

    // bucket: explicit seq rung from token-exact routing (selectIndexBucket).
    // Optional only for back-compat; the indexer always passes it.
    async embedBatch(texts: string[], bucket?: number): Promise<EmbedBatchTimed> {
        if (!this._loaded) throw new Error('Model not loaded.');
        const r = await this.runner.embedBatch(texts, bucket);
        return { vectors: r.vectors, iframeLatencyMs: r.latencyMs };
    }

    // Exact token counts from the iframe pipeline's tokenizer (specials
    // included — the count the forward pass sees). Tokenizer-only: no GPU
    // session involvement, so no recycle interplay. Feeds token-budget
    // enforcement + bucket routing in search.ts.
    async tokenCounts(texts: string[]): Promise<number[]> {
        if (!this._loaded && !this._tokenizerLoaded) throw new Error('Neither model nor tokenizer loaded.');
        return this.runner.tokenCounts(texts);
    }

    // Tokenizer-only load for the sidecar hydrate's chunk-id reproduction. A
    // full model load already includes the tokenizer, so this no-ops when
    // _loaded. Otherwise it pulls just the vocab/merges (a few MB) — the
    // hydrate can then reproduce token-budget splits without the ~250 MB model
    // it exists to avoid on mobile. Cheap and idempotent.
    private _tokenizerLoaded = false;
    async ensureTokenizer(modelId: string = MODEL_ID, revision: string | null = MODEL_REVISION): Promise<void> {
        if (this._loaded || this._tokenizerLoaded) return;
        await this.runner.loadTokenizer(modelId, revision);
        this._tokenizerLoaded = true;
    }

    // Capability probe — see iframe-runner.ts. Exposed here so callers
    // don't have to reach into the private runner field.
    appLocalFetch(url: string): Promise<AppLocalFetchResult> {
        return this.runner.appLocalFetch(url);
    }

    // Diagnostic — runtime wall-time decomposition. See IframeRunner.profile.
    async profile(batchSizes: number[], seqBuckets: number[], reps: number): Promise<RawProfile> {
        if (!this._loaded) throw new Error('Model not loaded.');
        return this.runner.profile(batchSizes, seqBuckets, reps);
    }

    // Reject in-flight iframe RPCs (e.g. on WebGPU device-lost) WITHOUT tearing
    // down — see IframeRunner.failInflight. The embed catch then recycles+retries.
    failInflight(message: string): void {
        this.runner.failInflight(message);
    }

    teardown(): void {
        this._loaded = false;
        // save-assign-dispose: point this.runner at the fresh instance and null
        // the memos BEFORE disposing the old runner, so the invariant "this.runner
        // pairs with _initPromise" never transiently breaks for a racing init().
        // Disposing the old runner rejects its in-flight RPCs (tagged DISPOSED);
        // an in-flight load() is abandoned with it, so drop the latch too.
        const oldRunner = this.runner;
        this.runner = new IframeRunner();
        this.runner.onEvent = this._onIframeEvent;
        this._initPromise = null;
        this._loadPromise = null;
        oldRunner.dispose();
    }
}
