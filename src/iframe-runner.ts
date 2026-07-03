// Sandboxed iframe that owns the transformers.js pipeline.
//
// Why an iframe at all? Obsidian's renderer applies a strict CSP that blocks
// `import()` of remote ES modules — but iframes with srcdoc inherit a more
// permissive CSP and can dynamically import from jsdelivr. The iframe also
// quarantines the ~250 MB model heap; on plugin teardown we just unmount
// the DOM node and the runtime reclaims everything.
//
// Lifted from the embeddinggemmaiostest spike with one adjustment for v0:
//   - Model id is the onnx-community fused PTQ export, run at q4 (see
//     embedder.ts header — the QAT workstream is killed).

import type { Device, RequestedDevice, Dtype } from './types';
import { ACTIVE_MODEL_SPEC } from './model-registry';

declare const __BUILD_TS__: string;

// HARD FLOOR — do not downgrade below 4.x. v4 bundles ORT-Web 1.26-dev,
// whose WebGPU MatMulNBits kernels are what make q4 viable on this stack.
// Measured on the Personal vault (2075 files / ~3950 chunks, WebGPU, same
// fused PTQ q4 model, 2026-05-17):
//   tx.js 3.8.0  q4  →   6.7 ch/s,  heap Δ ~186 MB
//   tx.js 3.8.0  q8  →  13.2 ch/s,  heap Δ ~380 MB
//   tx.js 4.2.0  q4  →  20.6 ch/s,  heap Δ ~103 MB   ← current
// i.e. v4 q4 is ~3× faster than v3 q4 AND ~1.6× faster than v3 q8 at ~¼
// the heap. The q4 throughput penalty was never a fusion confound (op
// histograms confirmed v3 q4 was fully fused, 24× MultiHeadAttention, and
// still 6.7 ch/s) — it was ORT-Web's immature MatMulNBits/WebGPU kernel for
// M>1 (prefill), which 1.26-dev finally optimizes. Re-bench before any
// version bump; the model-load path is API-sensitive across major versions.
// 2026-06-02: bumped 3.8.0 -> 4.2.0. granite-r2 is a ModernBERT (RoPE); its
// WebGPU rotary kernel ("k_rotary/term2_mul: Can't perform binary op") is
// BROKEN on 3.8.0's ORT-Web and only works on 4.x's ORT-Web 1.26. 3.8.0 was
// pinned for iOS (v4 ORT-Web can't init WebGPU in iOS WKWebView) — under 4.2,
// iOS will fail the WebGPU load and fall back to WASM at load time (iOS is
// query-only anyway). Desktop indexing needs 4.2 for granite WebGPU.
export const TRANSFORMERS_VERSION = '4.2.0';
const CDN_URL = `https://cdn.jsdelivr.net/npm/@huggingface/transformers@${TRANSFORMERS_VERSION}`;

// Warmup grid constants — exported so the parent can compose a fingerprint
// for the localStorage skip-warmup cache. Single source of truth: defined
// here, injected into the iframe script body via JSON.stringify substitution.
// If you change either array, the next load on every install will recompute
// (the fingerprint won't match) — that's the intended behavior, no manual
// cache bust required.
//
// WARMUP_BATCH_SIZES: the indexer uses a per-bucket rolling buffer that flushes
// at a FIXED warmed size of 8 (ROLLING_BATCH in search.ts), so the only batch
// counts ever dispatched are 8 and the partial-drain remainders 1..7. We warm
// exactly that set [1..8] — no 32 (the old within-file ceiling, never dispatched
// now). All warmed on both device classes so a vault sync between desktop+mobile
// never eats a multi-second WGSL compile stall mid-reindex.
//
// SEQ_BUCKETS: padding ladder. Caps unique tensor shapes Dawn sees to O(buckets)
// instead of O(unique token lengths). Nine LOG-SPACED buckets (the offline
// padding sim, 2026-06-03: log spacing beat linear-dense 85% vs 82% at equal
// count, because chunk lengths are log-distributed). vs the old 5-bucket ladder
// this lifts rolling-buffer efficiency 72%→85% (content ÷ padded positions) for
// ~16 extra warmup shapes. Single source of truth: injected into the iframe
// template AND mirrored by the exported selectBucket() below — keep in sync.
export const WARMUP_BATCH_SIZES = [1, 2, 3, 4, 5, 6, 7, 8];
export const SEQ_BUCKETS = [32, 48, 64, 96, 128, 192, 256, 384, 512];

// QUERY_SEQ_BUCKETS: the SINGLE-embed (query) path uses its own ladder, NOT the
// index ladder above. A query is a handful of tokens (live logs 2026-06-09:
// p50=3, p90=7, max=11 — 100% landed in the old seq=32 floor, so the median
// query did ~91% of its forward on padding). So:
//   - lower floor (8, 16): a 3-token query runs an 8-wide tensor, not 32-wide.
//   - capped at 128: ~90-100 words, longer than any link/path/NL question. There
//     is no dense-relevance case past this — a single CLS-pooled vector over 100+
//     tokens averages too many concepts to discriminate (regresses to centroid),
//     and BM25 (which is NOT bucketed — see MultiFieldBM25, full query) already
//     covers the only real long-input case, a literal text paste. A >128-token
//     query truncates to 128 on the dense side only; lexical recall is intact.
// The INDEX path keeps SEQ_BUCKETS unchanged (chunks are long; 256/512 matter).
// Experiment scaffold: shrinking seq cuts compute ~linearly but per-layer WebGPU
// dispatch overhead is seq-INDEPENDENT, so the floor win must be MEASURED from
// live iframeEmbedMs, not assumed.
export const QUERY_SEQ_BUCKETS = [8, 16, 32, 48, 64, 96, 128];

// Parent-side CHAR-ESTIMATE bucket selector — legacy fallback only. The
// ceil(chars/4.5) estimate under-buckets dense text (URLs, paths, numbers,
// code run ~3.6 chars/token), silently truncating BELOW the 512 cap — the
// WS2.2/WS2.3 "dense-invisible" finding (21-26% of corpus tokens). The index
// path now routes with selectIndexBucket (exact token counts via the
// token-counts RPC) and passes the bucket explicitly to embed-batch; this
// estimator remains only as the iframe-side fallback when no bucket is given.
// Keep identical to the template copy near WEBGPU_DTYPE_LADDER.
export function selectBucket(charCount: number): number {
    const est = Math.ceil(charCount / 4.5);
    for (const b of SEQ_BUCKETS) if (est <= b) return b;
    return SEQ_BUCKETS[SEQ_BUCKETS.length - 1];
}

// Token-exact index-path bucket selector (WS2.3 — the index-side twin of
// selectQueryBucket below). Takes the EXACT token count of the embed input,
// counted by the model's own tokenizer via the token-counts RPC, so the bucket
// always holds the whole input and truncation cannot drop tokens. Only inputs
// the token-budget packer could not reduce under the cap (pathological
// >512-token titles — see token-budget.ts) ever hit the last rung oversized.
export function selectIndexBucket(tokenCount: number): number {
    for (const b of SEQ_BUCKETS) if (tokenCount <= b) return b;
    return SEQ_BUCKETS[SEQ_BUCKETS.length - 1];
}

// Query-path bucket selector (mirrored in the template). Takes the EXACT token
// count of the cleaned query — the iframe tokenizes it (see embedText), so we
// don't guess from chars like the indexer's selectBucket does. The bucket is the
// smallest one ≥ the real token length, so it always holds the whole query and
// truncation cannot drop tokens. Only a >128-token query (no dense-relevance
// case — see QUERY_SEQ_BUCKETS) hits the cap and truncates the dense side; BM25
// still sees the full query.
export function selectQueryBucket(tokenCount: number): number {
    for (const b of QUERY_SEQ_BUCKETS) if (tokenCount <= b) return b;
    return QUERY_SEQ_BUCKETS[QUERY_SEQ_BUCKETS.length - 1];
}
const IFRAME_ID = 'seek-runtime-iframe';
const READY_TIMEOUT_MS = 30_000;

// Per-RPC timeout. The iframe WebContent process can be jetsam-killed mid-RPC on
// mobile (the documented reindex-hang failure) or its WebGPU device lost; the
// child then never replies and the parent promise would hang FOREVER — which
// also strands the embed catch's recycle+retry net behind an await that never
// returns (search.ts embedOneBatch). The timeout converts a hang into a
// RECOVERABLE rejection (tagged 'TIMEOUT', distinct from teardown's 'DISPOSED')
// so the existing recovery path fires. Two tiers: embed/token RPCs settle in
// ms–seconds (60 s is a vast ceiling even for a pathological iPhone WASM batch);
// a cold load() can stream ~60 MB of model bytes from the CDN on first install,
// so it gets a far longer budget before we call it dead.
const RPC_TIMEOUT_MS = 60_000;
const LOAD_RPC_TIMEOUT_MS = 180_000;

export interface IframeInit {
    buildTimestamp: string;
    cdnUrl: string;
    transformersVersion: string;
    ready: boolean;
    error: string | null;
    // Wall time of buildIframe() (DOM create + srcdoc parse + __ready handshake).
    // 0 on the idempotent early-return path (iframe already live).
    initMs: number;
}

export interface LoadResult {
    device: Device;
    dtype: Dtype;
    coldStartMs: number;
    // Wall-time of the WebGPU warmup loop only (40 forced dispatches that
    // compile WGSL shaders for the (batch_size × seq_len) grid the indexer
    // and query path will hit). NULL when:
    //   - WASM fallback path (no warmup applies)
    //   - WebGPU warmup was SKIPPED by the parent-side fingerprint cache
    //     (warmupSkipped=true distinguishes this from the WASM case)
    // Subtract from coldStartMs to get "pre-warmup load" (Cache API read +
    // ONNX parse + WebGPU device init + first pipeline construction).
    // Measurement on desktop (2026-05-19): warmup is locked at ~1020 ms
    // across consecutive warm reloads (Dawn shader cache hot), so skipping
    // it on warm reload recovers the full ~50% of warm-reload wall time.
    warmupMs: number | null;
    // True iff the parent passed skipWarmup=true AND we took the skip path
    // (WebGPU success path only — never true for WASM).
    warmupSkipped: boolean;
    webgpuAttempted: boolean;
    webgpuError: string | null;
    // Which ort-wasm glue variant an override actually selected: 'jspi' |
    // 'asyncify' (WebKit WebGPU attempt) or 'plain' (non-WebKit wasm pin —
    // the only build carrying the CPU GatherBlockQuantized kernel). null when
    // no override applied (WebKit wasm rides tx.js's own plain pin untouched).
    // Closes the diagnostics gap where the running glue was only ever visible
    // in failure strings.
    glue: string | null;
}

// Unsolicited iframe→parent push (not an RPC reply): WebGPU device lifecycle
// events from the requestDevice prototype hook. Routed to IframeRunner.onEvent.
export type IframeEvent = Record<string, string | number | boolean | null>;

export interface EmbedResult {
    vector: Float32Array;
    latencyMs: number;
}

// Raw per-cell timing arrays from the child's unrolled path. The child stays
// dumb (returns raw samples); distributionStats + share math happen parent-
// side in TS where the existing helper lives. tokenize/forward/post are the
// unrolled decomposition; pipe is the production pipeline() total on the same
// inputs (decomposition sanity check).
export interface RawProfileCell {
    batchSize: number;
    seqBucket: number;
    reps: number;
    tokenize: number[];
    forward: number[];
    post: number[];
    pipe: number[];
}

export interface RawProfile {
    cells: RawProfileCell[];
}

export interface AppLocalFetchResult {
    // ok=true: HTTP fetch succeeded AND body matched (when expectedBody set)
    // ok=false: fetch threw, non-2xx response, or body mismatch
    ok: boolean;
    status: number | null;
    body: string | null;
    error: string | null;
}

interface Pending {
    resolve: (value: unknown) => void;
    reject: (error: Error) => void;
    // Per-RPC timeout handle, cleared when the reply lands (or on dispose).
    // number is window.setTimeout's DOM return (clearTimeout takes it).
    timer?: number;
}

// Shape of an iframe→parent postMessage payload: RPC replies plus the bootstrap
// control messages (__ready / __error / __event). event.data is structurally
// untyped (any), so we narrow it to this before reading fields.
interface IframeMsg {
    id: string;
    ok?: boolean;
    result?: unknown;
    error?: string;
    event?: IframeEvent;
}

export class IframeRunner {
    private iframe: HTMLIFrameElement | null = null;
    private pending = new Map<string, Pending>();
    private listener: ((e: MessageEvent) => void) | null = null;
    // Receiver for unsolicited '__event' pushes (WebGPU device lifecycle).
    // Survives recycle (same runner instance rebuilds the iframe); the
    // embedder re-wires it across teardown() which swaps the runner.
    onEvent: ((event: IframeEvent) => void) | null = null;

    async init(): Promise<IframeInit> {
        const result: IframeInit = {
            buildTimestamp: __BUILD_TS__,
            cdnUrl: CDN_URL,
            transformersVersion: TRANSFORMERS_VERSION,
            ready: false,
            error: null,
            initMs: 0,
        };
        // Idempotency guard for SEQUENTIAL re-entry (recycle/teardown-then-init,
        // the Unload command). buildIframe() does an unconditional
        // document.createElement(iframe) — calling init() twice on a live runner
        // would append a second #seek-runtime-iframe and leak the first listener.
        // contentWindow is non-null only after the srcdoc document attaches, so an
        // in-progress build still reads as "not live" here; CONCURRENT callers are
        // serialized one layer up by LocalEmbedder's memoized _initPromise.
        if (this.iframe?.contentWindow) {
            result.ready = true;
            return result;
        }
        const t0 = performance.now();
        try {
            await this.buildIframe();
            result.ready = true;
        } catch (e) {
            result.error = String(e);
            // A failed bootstrap (__error or the ready-timeout) still leaves the
            // iframe ELEMENT attached with a live contentWindow — buildIframe()
            // doesn't unwind. Left as-is, the idempotency guard above (and send()'s
            // 'iframe not initialized' check) would read that dead iframe as usable,
            // so a retry would short-circuit to a false-ready and load() would hang
            // posting to an iframe that never answers. dispose() unwinds to a clean
            // no-iframe state (also removing the leaked message listener) so the next
            // init() genuinely rebuilds.
            this.dispose();
        }
        result.initMs = parseFloat((performance.now() - t0).toFixed(2));
        return result;
    }

    private buildIframe(): Promise<void> {
        return new Promise((resolve, reject) => {
            const timeout = window.setTimeout(
                () => reject(new Error(`iframe ready timeout after ${READY_TIMEOUT_MS}ms`)),
                READY_TIMEOUT_MS,
            );

            this.listener = (event: MessageEvent) => {
                if (!this.iframe || event.source !== this.iframe.contentWindow) return;
                const data = event.data as IframeMsg | undefined;
                if (!data || typeof data !== 'object') return;

                if (data.id === '__ready') { window.clearTimeout(timeout); resolve(); return; }
                if (data.id === '__error') {
                    window.clearTimeout(timeout);
                    reject(new Error(`iframe bootstrap failed: ${data.error}`));
                    return;
                }
                if (data.id === '__event') {
                    // Unsolicited push, not an RPC reply. Handler errors must
                    // never poison the RPC dispatch below it.
                    try { this.onEvent?.(data.event as IframeEvent); } catch { /* swallow */ }
                    return;
                }
                const p = this.pending.get(data.id);
                if (!p) return;
                this.pending.delete(data.id);
                if (p.timer) window.clearTimeout(p.timer);
                if (data.ok) p.resolve(data.result);
                else p.reject(new Error(data.error ?? 'iframe error'));
            };
            window.addEventListener('message', this.listener);

            // Anchor the hidden compute iframe to `window.document` (the main
            // window's document), NOT activeDocument: it is display:none (no
            // popout-render benefit), must outlive any popout (anchoring it to the
            // window focused at first embed would orphan it when that popout
            // closes), and its contentWindow postMessage must reach the `window`
            // message listener bound above.
            this.iframe = window.document.createElement('iframe');
            this.iframe.id = IFRAME_ID;
            this.iframe.addClass('seek-hidden');
            // LOAD-BEARING: no `sandbox` attribute. A srcdoc iframe with no sandbox
            // inherits Obsidian's real origin (`capacitor://localhost` on iOS,
            // `app://obsidian.md` on desktop). That real origin is what lets the
            // child's transformers.js fetch() pull the model from the HF CDN —
            // HF returns `access-control-allow-origin: *`, so the cross-origin
            // request passes — and share the parent's Cache-API partition (so the
            // parent-side eviction in model-registry.ts can see the cache). Adding
            // a `sandbox` attribute would give the iframe an opaque `null` origin
            // and break BOTH on iOS. Do not add one without first moving the model
            // fetch out of the iframe (e.g. parent requestUrl → resource URL).
            window.document.body.appendChild(this.iframe);

            const childScript = buildChildScript(CDN_URL, ACTIVE_MODEL_SPEC.dim);
            this.iframe.srcdoc =
                `<!DOCTYPE html><html><body><script type="module">${childScript}</script></body></html>`;
        });
    }

    private send<T>(type: string, payload: unknown, timeoutMs: number = RPC_TIMEOUT_MS): Promise<T> {
        if (!this.iframe?.contentWindow) {
            return Promise.reject(new Error('iframe not initialized'));
        }
        const id = (crypto as { randomUUID?: () => string }).randomUUID
            ? (crypto as { randomUUID: () => string }).randomUUID()
            : `id-${Date.now()}-${Math.random()}`;
        return new Promise<T>((resolve, reject) => {
            // Backstop a dead/silent child (jetsam kill, WebGPU device loss): if
            // no reply lands in time, reject with a RECOVERABLE 'TIMEOUT' so the
            // embed catch recycles+retries instead of hanging. Cleared in the
            // message listener the instant the real reply arrives.
            const timer = window.setTimeout(() => {
                if (!this.pending.delete(id)) return;
                reject(Object.assign(
                    new Error(`iframe RPC '${type}' timed out after ${timeoutMs}ms`),
                    { code: 'TIMEOUT' },
                ));
            }, timeoutMs);
            this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject, timer });
            this.iframe!.contentWindow!.postMessage({ id, type, payload }, '*');
        });
    }

    // skipWarmup: parent decision based on the localStorage fingerprint cache
    // (see embedder.ts). If true AND the WebGPU path succeeds, the iframe
    // bypasses the 40-dispatch shader-compile loop entirely. Always pay the
    // warmup on the WASM path or on a fingerprint miss.
    load(modelId: string, device: RequestedDevice, dtype: Dtype, skipWarmup: boolean, revision?: string | null): Promise<LoadResult> {
        return this.send<LoadResult>('load', { modelId, device, dtype, skipWarmup, revision: revision ?? null }, LOAD_RPC_TIMEOUT_MS);
    }

    embed(text: string): Promise<EmbedResult> {
        return this.send<EmbedResult>('embed', { text });
    }

    // bucket: the seq-ladder rung this batch was routed into by the parent's
    // token-exact selectIndexBucket. The iframe uses it verbatim as max_length,
    // so parent routing and iframe padding agree by construction (no second
    // char-estimate derivation). Omitting it falls back to the legacy char
    // estimate inside the iframe — kept only for back-compat callers.
    embedBatch(texts: string[], bucket?: number): Promise<{ vectors: Float32Array[]; latencyMs: number }> {
        return this.send<{ vectors: Float32Array[]; latencyMs: number }>('embed-batch', { texts, bucket });
    }

    // Exact token counts (specials included — the same count the forward pass
    // sees) for each text, from the iframe pipeline's own tokenizer. Tokenizer
    // only, no model forward: cheap (~µs-ms per text) and device-independent.
    // The single source of tokenizer truth stays inside the iframe; the parent
    // never re-implements or approximates it.
    tokenCounts(texts: string[]): Promise<number[]> {
        return this.send<number[]>('token-counts', { texts });
    }

    // Load ONLY the tokenizer (vocab/merges JSON, a few MB) — no ONNX model
    // weights. Lets the sidecar hydrate reproduce the token-budget chunk splits
    // (and therefore exact chunk_ids) without the ~250 MB model load it exists
    // to avoid on mobile. Idempotent in the child; tokenCounts() then works
    // against either the full pipeline or this standalone tokenizer.
    loadTokenizer(modelId: string, revision?: string | null): Promise<{ ok: boolean; cached: boolean }> {
        return this.send<{ ok: boolean; cached: boolean }>('load-tokenizer', { modelId, revision: revision ?? null }, LOAD_RPC_TIMEOUT_MS);
    }

    // Diagnostic only. Runs the UNROLLED tokenizer→model→readback path with
    // timing boundaries across a (batchSize × seqBucket) matrix so we can see
    // where wall time actually goes on the v4 runtime. Never used in the
    // production embed path — that stays the opaque pipeline() call for
    // correctness; this is the measurement twin.
    profile(batchSizes: number[], seqBuckets: number[], reps: number): Promise<RawProfile> {
        // Diagnostic only; a full (batch × seq) matrix can run minutes — give it
        // a generous ceiling so the per-RPC timeout never trips a real profile.
        return this.send<RawProfile>('embed-profile', { batchSizes, seqBuckets, reps }, 600_000);
    }

    // Probe whether the iframe (srcdoc + sandboxed) can fetch a vault
    // resource via Obsidian's runtime URL scheme. Gates the Phase 3 shard
    // streaming pattern; see seek-dataadapter-rearchitecture-plan §Phase 1.
    appLocalFetch(url: string): Promise<AppLocalFetchResult> {
        return this.send<AppLocalFetchResult>('app-local-fetch', { url });
    }

    dispose(): void {
        if (this.listener) {
            window.removeEventListener('message', this.listener);
            this.listener = null;
        }
        if (this.iframe?.parentNode) this.iframe.parentNode.removeChild(this.iframe);
        this.iframe = null;
        // Tag the rejection 'DISPOSED' so the embed catch can distinguish an
        // intentional teardown (plugin unloading → must NOT recycle, or it
        // resurrects a zombie iframe + reloads ~250 MB into a dead plugin) from
        // a recoverable error (SafeInt overflow / TIMEOUT → recycle+retry).
        // Clear each pending timer first so a timeout can't fire post-rejection.
        for (const [, p] of this.pending) {
            if (p.timer) window.clearTimeout(p.timer);
            p.reject(Object.assign(new Error('iframe disposed'), { code: 'DISPOSED' }));
        }
        this.pending.clear();
    }

    // Reject every in-flight RPC with a RECOVERABLE error (NOT 'DISPOSED'),
    // leaving the iframe attached. Used on WebGPU device-lost: the device is
    // dead, so a hung embedBatch must reject to let the caller's recycle+retry
    // fire — but this is recovery, not teardown, so the embed catch must not see
    // 'DISPOSED' (which would unwind the reindex). The caller's recycle() then
    // replaces the now-dead-device iframe.
    failInflight(message: string): void {
        for (const [, p] of this.pending) {
            if (p.timer) window.clearTimeout(p.timer);
            p.reject(new Error(message));
        }
        this.pending.clear();
    }
}

// Exported for testing only — the string content of the RPC dispatch handler
// (e.g. the source-origin check) can't be exercised via a real srcdoc iframe
// in the node test env, so tests assert on the emitted script text instead.
export function buildChildScript(cdnUrl: string, outputDim: number): string {
    // Script body runs INSIDE the iframe. It imports transformers.js from CDN,
    // owns pipeline state, and responds to postMessage RPCs from the parent.
    // The ${...} substitutions below pin the warmup grid AND the output dimension
    // to the parent's exported constants so the localStorage fingerprint cache and
    // the vector width stay accurate by construction. NOTE: the child body is a
    // template literal in the parent — code inside it must NOT use backticks or
    // ${} (single-quote concatenation only), or the parent will eval it.
    return `
const CDN_URL = ${JSON.stringify(cdnUrl)};
const WARMUP_BATCH_SIZES = ${JSON.stringify(WARMUP_BATCH_SIZES)};
const SEQ_BUCKETS = ${JSON.stringify(SEQ_BUCKETS)};
const QUERY_SEQ_BUCKETS = ${JSON.stringify(QUERY_SEQ_BUCKETS)};
let pipeline = null;
// Standalone tokenizer (no model weights) for the sidecar hydrate's chunk-id
// reproduction. Independent of pipeline — survives recycle.
let standaloneTokenizer = null;
// EP of the loaded pipeline ('webgpu' | 'wasm'). Drives the padding mode:
// bucket-padding exists ONLY for Dawn's shape-keyed shader cache (one WGSL
// compile per tensor shape, SafeInt-overflow discipline). The CPU EP has no
// fixed-shape constraint, so on wasm we pad to the LONGEST SEQUENCE IN THE
// BATCH instead of the bucket rung — at the measured ~85% bucket efficiency
// that's ~15% of all CPU forward work spent multiplying padding, for free
// (2026-06-11 iPhone WASM run analysis).
let currentDevice = null;

// ── wasm Memory.maximum clamp (2026-06-14, WebKit-only) ──────────────────
// Probe B (experiments/lazyrelease-probe.html) confirmed the ort-wasm glue
// instantiates WebAssembly.Memory with maximum=65536 pages (4 GiB), shared. On
// a memory-pressured iPhone that 4 GiB reservation OOMs at INSTANTIATION
// (RangeError: Out of memory -> "no available backend found"), before a single
// embed runs -- observed 2026-06-14 the first time we forced WebGPU on the
// phone (the heavier asyncify glue + the 4 GiB shared max tipped a pressured
// WebContent over; the failed attempt then took the WASM fallback down too).
// Clamp the maximum so the reservation is a fraction of the ~1.5 GiB WebContent
// budget while leaving generous headroom over the working set (WebGPU mode
// keeps activations on the GPU; the wasm heap holds tokenizer + I/O staging,
// and use_ort_model_bytes_directly avoids a model copy). EXPERIMENT: if iOS
// only reserves address space this is a no-op; if it pre-charges the maximum
// against jetsam (the investigation's read) it is the fix. Risk: a growth-OOM
// mid-run if the heap genuinely needs more than the cap -> watch for a LATER
// RangeError and raise WASM_MAX_PAGES if so. WebKit-only: desktop/Electron keep
// the 4 GiB default (memory is ample there). Must run before any wasm
// instantiation; isWebKit is a hoisted function declaration below. This block
// lives inside the iframe srcdoc template literal -- no backticks here. REVERT:
// delete this block.
const WASM_MAX_PAGES = 8192; // 512 MiB (page = 64 KiB); 65536 = the 4 GiB default. 2026-06-14: 16384 (1 GiB) loaded WebGPU but the reservation ate steady-state headroom (foreground kill at ~9.5k tokens / 8 files, EARLIER than the un-capped 12-22k); 512 MiB leaves the pool more room. Raise if model load growth-OOMs; lower toward 6144 if steady-state still dies early.
if (typeof isWebKit === 'function' && isWebKit() && typeof WebAssembly !== 'undefined' && WebAssembly.Memory) {
    const OrigMemory = WebAssembly.Memory;
    const ClampedMemory = function (desc) {
        if (desc && typeof desc === 'object' && typeof desc.maximum === 'number' && desc.maximum > WASM_MAX_PAGES) {
            desc = Object.assign({}, desc, { maximum: WASM_MAX_PAGES });
        }
        return new OrigMemory(desc);
    };
    ClampedMemory.prototype = OrigMemory.prototype;
    try { WebAssembly.Memory = ClampedMemory; } catch (_) { /* frozen global -- leave default */ }
}

// ── WebGPU loss diagnostics (2026-06-11) ────────────────────────────────
// Three iPhone WebGPU-reindex deaths left ZERO OS-side forensics (no
// JetsamEvent, no crash .ips) — so the kill is likely WebKit-internal, and
// the only remaining discriminator is in-page: GPUDevice.lost resolves when
// the GPU process dies (the page SURVIVES and sees it), while a WebContent
// process kill gives JS nothing at all. Absence of a device-lost breadcrumb
// before a crash-detected verdict therefore reads as "WebContent died
// directly". ORT-Web requests its device internally — we never see the call
// — so hook GPUAdapter.prototype.requestDevice and instrument every device
// created in this realm. Events post to the parent, which writes them to
// the SYNCHRONOUS forensics ring before the async NDJSON append; the
// breadcrumb must win the race against a death that may follow within ms.
let deviceSeq = 0;
let gpuEventBudget = 12; // lifetime cap — a crash-looping device must not flood the ring
function postGpuEvent(event) {
    if (gpuEventBudget <= 0) return;
    gpuEventBudget--;
    try { window.parent.postMessage({ id: '__event', event }, '*'); } catch (_) {}
}
if (navigator.gpu && typeof GPUAdapter !== 'undefined') {
    const origRequestDevice = GPUAdapter.prototype.requestDevice;
    GPUAdapter.prototype.requestDevice = async function (desc) {
        const device = await origRequestDevice.call(this, desc);
        const seq = ++deviceSeq;
        try {
            postGpuEvent({
                kind: 'webgpu-device-created',
                deviceSeq: seq,
                // What ORT asked for — never observed before this hook.
                requiredFeatures: desc && desc.requiredFeatures ? Array.from(desc.requiredFeatures).map(String).join(',') : '',
                requiredLimitCount: desc && desc.requiredLimits ? Object.keys(desc.requiredLimits).length : 0,
            });
            device.lost.then((info) => {
                postGpuEvent({
                    kind: 'webgpu-device-lost',
                    deviceSeq: seq,
                    reason: String((info && info.reason) || 'unknown'),
                    message: String((info && info.message) || '').slice(0, 300),
                });
            }, () => {});
            device.addEventListener('uncapturederror', (e) => {
                const err = e && e.error;
                postGpuEvent({
                    kind: 'webgpu-uncaptured-error',
                    deviceSeq: seq,
                    error: String((err && err.message) || err).slice(0, 300),
                });
            });
        } catch (_) { /* diagnostics must never break the load path */ }
        return device;
    };
}

// OUTPUT_DIM is INJECTED from the parent (ACTIVE_MODEL_SPEC.dim) so it can never
// drift from embedder.EMBEDDING_DIM or the sidecar record stride. granite-r2
// outputs a 384-d CLS-pooled vector natively (NOT MRL), so sliceAndRenormalize is
// a pass-through; a Matryoshka model injects a smaller dim and the slice truncates
// to it. The embed guards below fail loud if the model's real width < OUTPUT_DIM
// (sliceAndRenormalize would otherwise silently emit a too-short vector).
const OUTPUT_DIM = ${JSON.stringify(outputDim)};

// Dtype ladder for WebGPU. tryWebgpu walks this in order and accepts the
// first dtype whose shaders compile.
//
// q4-only (q8 dropped). q4 ≈ q8 quality on this vault (fused NDCG@10
// Δ=0.0005) and, on the pinned v4 runtime, q4 is also the *fastest and
// lightest* config (20.6 ch/s, ~103 MB heap vs q8's 13.2 ch/s, ~380 MB —
// see the TRANSFORMERS_VERSION floor comment). So q8 is strictly dominated;
// no q8 rung.
//
// History, corrected: on the old 3.8.0 ORT-Web q8 genuinely *was* ~2×
// faster than q4 (13.2 vs 6.7 ch/s) — NOT a fusion confound. Op-histogram
// diff of the onnx-community protos confirmed v3 q4 was fully fused (24×
// MultiHeadAttention, 48× RotaryEmbedding, identical to q8) and still 6.7
// ch/s. The entire gap was one op: q8 decomposes to ORT-Web's golden
// MatMul + DequantizeLinear; q4 uses MatMulNBits, whose 3.8.0-era WebGPU
// kernel ran a slow generic path for M>1 (every embedding call is
// M=seq_len). v4's 1.26-dev ORT-Web optimizes that kernel — which is the
// whole reason v4 is a hard floor, not an upgrade-when-convenient.
//
// tryWebgpu prefers the model_no_gather_q4 variant when reachable (leaner
// 1333-node graph, better-coalesced memory access vs plain model_q4's
// GatherBlockQuantized embedding-table path), falling back transparently.
// fp32 last: weight size ~6× heavier (1.23 GB) but always works if it fits
// in maxBufferSize — the lifeboat rung if q4 shaders ever fail to compile.
//
// q4f16 is intentionally absent. Gemma's LayerNorm shader fails to compile
// on Dawn with half-precision activations (ORT #26732 — observed bricking a
// full reindex in May 2026 with Invalid ShaderModule errors across all files).
const WEBGPU_DTYPE_LADDER = ['q4', 'fp32'];

// SEQ_BUCKETS is injected from the parent (see template substitution above).
// Kept as a single source of truth so the parent-side fingerprint cache
// can't drift from what the iframe actually warms.
function selectQueryBucket(tokenCount) {
    for (const b of QUERY_SEQ_BUCKETS) { if (tokenCount <= b) return b; }
    return QUERY_SEQ_BUCKETS[QUERY_SEQ_BUCKETS.length - 1];
}

function selectBucket(charCount) {
    const est = Math.ceil(charCount / 4.5);
    for (const b of SEQ_BUCKETS) { if (est <= b) return b; }
    return 512;
}

// WebKit JSEP guard. ORT #26827: Safari/WebKit's WASM compiler enters an
// infinite loop through parseAndCompileOMG → GraphColoringStackAllocator
// when JSEP (WebGPU) mode is active, pinning CPU at 400%+ and growing memory
// to 14 GB+ before the process dies. Triggered once iOS 26 / visionOS 26
// expose WebGPU in WKWebView. Electron's UA also contains "Safari" because
// of its WebKit lineage, so explicitly exclude it.
function isWebKit() {
    const ua = navigator.userAgent;
    return /Safari/.test(ua) && !/Chrome/.test(ua) && !/Electron/.test(ua);
}

function sliceAndRenormalize(vec, targetDim) {
    if (vec.length <= targetDim) return vec;
    const sliced = new Float32Array(targetDim);
    for (let i = 0; i < targetDim; i++) sliced[i] = vec[i];
    let norm = 0;
    for (let i = 0; i < targetDim; i++) norm += sliced[i] * sliced[i];
    norm = Math.sqrt(norm);
    if (norm > 0) for (let i = 0; i < targetDim; i++) sliced[i] /= norm;
    return sliced;
}

// tx.js 4.2.0 serializes every session creation through a module-level
// promise chain with NO rejection handler (src/backends/onnx.js:
// webInitChain = webInitChain.then(load)). One rejected createPipeline
// poisons the chain for the module instance's lifetime: every later attempt
// SKIPS its load() and re-throws the FIRST error verbatim. That's why the
// 2026-06-10 iPad failure surfaced the raw "[webgpu] webgpuInit is not a
// function" error with no ladder/fallback wrapping — the wasm fallback never
// ran. Fix: give each load attempt a fresh module instance via a
// fragment-suffixed dynamic import. The module map keys on the full URL
// (fragment included) so '#seek-gen-2' is a distinct instance, while fetch
// strips the fragment — same HTTP cache entry, no re-download.
let importGen = 0;
async function freshTransformers(modelId) {
    importGen++;
    const mod = await import(CDN_URL + (importGen > 1 ? '#seek-gen-' + importGen : ''));
    const { pipeline: createPipeline, env, AutoTokenizer } = mod;
    // Phase-5 local-model override: a model id with a URL scheme
    // (app://local/..., capacitor://..., http(s)://) that isn't the HF
    // hub means load LOCAL weights. transformers.js then path-joins the
    // base with /onnx/model_*.onnx (plus the .onnx_data sidecar) and
    // fetch()es it from the vault resource URL instead of the HF CDN.
    const isLocalBase = modelId.includes('://') && !modelId.includes('huggingface.co');
    env.allowLocalModels = isLocalBase;
    env.allowRemoteModels = !isLocalBase;
    env.useBrowserCache = !isLocalBase;
    return { createPipeline, env, AutoTokenizer };
}

// Tokenizer-only load (no ONNX session) for the sidecar hydrate. Reuses
// freshTransformers' env wiring (CDN/local + browser cache) so the tokenizer
// files resolve identically to a full load. No webInitChain risk: AutoTokenizer
// never creates an ORT session. Idempotent — a second call no-ops.
async function loadTokenizer(modelId, revision) {
    if (standaloneTokenizer) return { ok: true, cached: true };
    const { AutoTokenizer } = await freshTransformers(modelId);
    if (!AutoTokenizer) throw new Error('transformers.js export missing AutoTokenizer — API changed?');
    standaloneTokenizer = await AutoTokenizer.from_pretrained(modelId, revision ? { revision } : {});
    return { ok: true, cached: false };
}

// tx.js 4.2.0 pins anything that detects as Safari — WKWebView included — to
// the PLAIN wasm glue (ort-wasm-simd-threaded.mjs), the one dist variant
// compiled WITHOUT the webgpuInit entry point (src/backends/onnx.js, the
// apis.IS_SAFARI wasmPaths branch). So on WebKit the webgpu EP init is
// structurally guaranteed to throw "De().webgpuInit is not a function"
// regardless of what the GPU supports — this, not a WKWebView capability
// gap, is the 2026-06-02/06-10 iOS failure. The pin only applies when
// wasmPaths is unset, so rewriting wasmPaths right after import (before the
// first createPipeline caches it via ensureWasmLoaded) restores a
// WebGPU-capable glue: jspi when the engine has JSPI (14.5 MB, no asyncify
// transform), else asyncify (23.6 MB). No-op off WebKit — everywhere else
// tx.js already picks asyncify. ⚠️ tx.js presumably pinned Safari to the
// plain glue for a reason (ORT #26827-class WASM-compile hangs are the
// suspect), so this runs only on the WebGPU attempt path, which mobile only
// reaches on the 'auto' device (iPad by default, or a forced-WebGPU override —
// see platform.ts resolveDevice; iPhone + Android stay on WASM).
function overrideWebkitGlueForWebgpu(env) {
    const wp = env.backends.onnx.wasm.wasmPaths;
    if (!wp || typeof wp !== 'object' || !wp.mjs) return null;
    if (!String(wp.mjs).includes('ort-wasm-simd-threaded.mjs')) return null; // not the Safari plain pin
    const variant = typeof WebAssembly.Suspending === 'function' ? 'jspi' : 'asyncify';
    wp.mjs = String(wp.mjs).replace('ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.' + variant + '.mjs');
    wp.wasm = String(wp.wasm).replace('ort-wasm-simd-threaded.wasm', 'ort-wasm-simd-threaded.' + variant + '.wasm');
    return variant;
}

// The mirror image, for the pure-WASM path on NON-WebKit engines. tx.js pins
// the ORT glue by ENGINE, not by requested device: WebKit gets the plain
// ort-wasm-simd-threaded build, everything else (Electron desktop, Android
// WebView) gets the asyncify WebGPU-native build — for device:'wasm' sessions
// too. Those builds do NOT carry the same CPU kernel set: the asyncify/jspi
// binaries register GatherBlockQuantized only on the webgpu EP, so a CPU-only
// session cannot place the GBQ4 model's quantized embedding-table node and
// session creation dies with "Could not find an implementation for
// GatherBlockQuantized" (r/ObsidianMD report, 2026-07-03) — the wasm fallback
// was dead-on-arrival on every Chromium machine, and with it "Force CPU" and
// all of Android. The plain build carries the full CPU kernel set (it is the
// binary every WebKit device has been running q4 on in production), so pin it
// for wasm sessions here. No-op on WebKit (already plain) and when wasmPaths
// is unset. Bonus: plain is 12.9 MB vs asyncify's 23.6 MB.
function overrideGlueForWasm(env) {
    const wp = env.backends.onnx.wasm.wasmPaths;
    if (!wp || typeof wp !== 'object' || !wp.mjs) return null;
    if (!String(wp.mjs).includes('ort-wasm-simd-threaded.asyncify.mjs')) return null; // already plain
    wp.mjs = String(wp.mjs).replace('ort-wasm-simd-threaded.asyncify.mjs', 'ort-wasm-simd-threaded.mjs');
    wp.wasm = String(wp.wasm).replace('ort-wasm-simd-threaded.asyncify.wasm', 'ort-wasm-simd-threaded.wasm');
    return 'plain';
}

async function tryWebgpu(modelId, preferredDtype, revision) {
    const order = [preferredDtype, ...WEBGPU_DTYPE_LADDER.filter(d => d !== preferredDtype)];
    const errors = [];
    for (const d of order) {
        // granite-r2's 50k vocab has no 256k-Gather hotspot, so the Gemma-era
        // model_no_gather_q4 variant doesn't exist (and isn't needed). Load the
        // stock onnx-community file per dtype: model_q4.onnx / model.onnx (fp32).
        try {
            const { createPipeline, env } = await freshTransformers(modelId);
            const glue = overrideWebkitGlueForWebgpu(env);
            try {
                // storageBufferCacheMode (2026-06-14, proven reachable via
                // experiments/lazyrelease-probe.html — ORT TraceSessionOptions
                // shows ep.webgpuExecutionProvider.storageBufferCacheMode set to
                // lazyRelease landing in the EP config_options, contradicting
                // the "extra is unreachable" prediction in Seek Mobile WebGPU
                // Investigation). ORT-Web's default bucket mode is the
                // cumulative-growth root cause of the iPhone steady-state kill:
                // a bucketed storage-buffer freelist that retains every distinct
                // size ever requested, with no global cap and no trim trigger.
                // lazyRelease frees storage buffers at onRunEnd, capping the
                // pool near working-set WITHOUT an iframe recycle. Gated to
                // (NOTE: this whole function body is inside the iframe srcdoc
                // template literal — no backticks permitted in comments here.)
                // WebKit: desktop/Electron keep bucket (memory is ample,
                // re-alloc-per-run would only cost throughput); the
                // memory-constrained iOS/iPadOS path opts into the release cost.
                // REVERT: delete the session_options branch. On-device efficacy
                // (does the iPhone now finish a reindex) is the real test, still
                // pending — reachability was the blocker, and it's cleared.
                const opts = isWebKit()
                    ? { device: 'webgpu', dtype: d, session_options: { extra: { 'ep.webgpuExecutionProvider.storageBufferCacheMode': 'lazyRelease' } } }
                    : { device: 'webgpu', dtype: d };
                if (revision) opts.revision = revision;   // F10: pin the model commit
                const p = await createPipeline('feature-extraction', modelId, opts);
                // glue is non-null only when the WebKit override rewrote the
                // Safari plain pin (jspi/asyncify) — i.e. exactly the iOS path
                // under investigation. Surfacing it on SUCCESS closes the "we
                // never log which glue actually ran" gap (it was only ever
                // recorded on failure, inside the error string).
                return { pipeline: p, dtype: d, errors, glue };
            } catch (e) {
                errors.push(d + (glue ? ' (glue:' + glue + ')' : '') + ': ' + String(e));
            }
        } catch (e) {
            errors.push(d + ': import failed: ' + String(e));
        }
    }
    throw new Error('all WebGPU dtypes failed: ' + errors.join(' | '));
}

async function loadModel(modelId, requestedDevice, requestedDtype, skipWarmup, revision) {
    const t0 = performance.now();

    let webgpuAttempted = false;
    let webgpuError = null;

    if (requestedDevice === 'webgpu' || requestedDevice === 'auto') {
        webgpuAttempted = true;
        if (!navigator.gpu) {
            webgpuError = 'navigator.gpu not present';
        } else {
            try {
                const adapter = await navigator.gpu.requestAdapter();
                if (!adapter) {
                    webgpuError = 'requestAdapter returned null';
                } else {
                    try {
                        const r = await tryWebgpu(modelId, requestedDtype, revision);
                        pipeline = r.pipeline;
                        // Warmup: one forward pass per (batch_size, seq_len_bucket) pair
                        // forces Dawn to compile all WGSL shaders upfront. Dawn encodes
                        // batch_size into the dispatch grid, so (1,64) and (2,64) are
                        // distinct compiled pipelines. [1..8] is the exact set the
                        // rolling-buffer indexer dispatches (fixed flush size 8 +
                        // partial-drain remainders 1..7); no shape outside the warmed
                        // grid is ever requested, which is what keeps the ORT-Web
                        // WebGPU pool off the SafeInt-overflow path the reverted
                        // arbitrary-coalescer hit. 8 batch sizes × 9 seq buckets =
                        // 72 passes × ~50 ms ≈ 3.6 s on first cold-start; Dawn
                        // persists to disk cache so later sessions pay only inference
                        // time. Keep WARMUP_BATCH_SIZES/SEQ_BUCKETS in sync with the
                        // indexer's ROLLING_BATCH + selectBucket (search.ts).
                        // Warmup skip: if the parent's localStorage fingerprint
                        // says we've already warmed this exact (model, dtype,
                        // transformers_version, grid) combo, skip the 40
                        // forced dispatches. On desktop measurements
                        // (2026-05-19) the warmup costs ~1020 ms locked across
                        // consecutive reloads with the Dawn shader cache hot,
                        // so this is the largest single warm-reload lever.
                        // The actual first-use compile (if any shape isn't
                        // in Dawn's cache) is paid lazily by the live call
                        // — same behavior as the try/catch around each
                        // warmup dispatch already implies.
                        let warmupMs = null;
                        if (!skipWarmup) {
                            const warmupStart = performance.now();
                            for (const n of WARMUP_BATCH_SIZES) {
                                const batch = Array(n).fill('warmup');
                                for (const bucket of SEQ_BUCKETS) {
                                    try {
                                        await pipeline(batch, {
                                            pooling: 'cls', normalize: true,
                                            padding: 'max_length', truncation: true, max_length: bucket,
                                        });
                                    } catch (_) { /* non-fatal — live call compiles on first use */ }
                                }
                            }
                            // Query path is batch=1 on QUERY_SEQ_BUCKETS; the loop
                            // above already warmed every (1 × SEQ_BUCKETS) shape, so
                            // only the query-only floors (8, 16) need compiling here.
                            for (const bucket of QUERY_SEQ_BUCKETS) {
                                if (SEQ_BUCKETS.includes(bucket)) continue;
                                try {
                                    await pipeline(['warmup'], {
                                        pooling: 'cls', normalize: true,
                                        padding: 'max_length', truncation: true, max_length: bucket,
                                    });
                                } catch (_) { /* non-fatal — live call compiles on first use */ }
                            }
                            warmupMs = performance.now() - warmupStart;
                        }
                        return {
                            device: 'webgpu', dtype: r.dtype,
                            coldStartMs: performance.now() - t0,
                            warmupMs,
                            warmupSkipped: !!skipWarmup,
                            webgpuAttempted, webgpuError: null,
                            glue: r.glue ?? null,
                        };
                    } catch (e) {
                        webgpuError = String(e);
                    }
                }
            } catch (e) {
                webgpuError = 'requestAdapter threw: ' + String(e);
            }
        }
        if (requestedDevice === 'webgpu') {
            throw new Error('WebGPU requested but failed: ' + webgpuError);
        }
    }

    // WASM fallback — always on a FRESH module instance (a failed WebGPU
    // attempt above poisoned the previous instance's webInitChain; see
    // freshTransformers). Glue: on WebKit the wasm EP rides tx.js's Safari
    // plain-glue pin (the known-good mobile path); on every OTHER engine
    // overrideGlueForWasm rewrites tx.js's asyncify pin back to the same
    // plain build, because asyncify has no CPU GatherBlockQuantized kernel
    // and a q4 session can never be created on it (see the function comment).
    //
    // Both-fail rethrow: if WebGPU was attempted and the wasm leg ALSO dies,
    // throwing only the wasm error strands the WebGPU failure cause —
    // loadModel throws before returning the LoadResult, so webgpuAttempted /
    // webgpuError never reach the log and the diagnostic report shows only
    // the terminal wasm session error, not the WebGPU init failure that
    // forced the fallback (exactly the blind spot in the r/ObsidianMD
    // report). Combine both causes so 'why did WebGPU fail' is answerable.
    const wasmFail = function (e) {
        if (webgpuAttempted && webgpuError) {
            const combined = new Error('model load failed on both paths — webgpu: ' + webgpuError + ' || wasm: ' + String(e));
            // Keep the terminal wasm throw-site stack: the child handler posts
            // e.stack to the parent, and without this the combined error's
            // stack points HERE instead of into ORT — degrading the forensic
            // channel this rethrow exists to protect.
            if (e && e.stack) combined.stack = combined.stack + '\\ncaused by (wasm): ' + e.stack;
            return combined;
        }
        return e;
    };
    // SIMD retry: transformers.js v3+ requires SIMD in sandboxed iframe /
    // WKWebView contexts; auto-detection can silently report "no available
    // backend" instead of falling back. Catch that case and retry — again on
    // a fresh instance, since the failure just poisoned this one — with SIMD
    // + multithread explicitly disabled (the conservative path the runtime
    // would have picked if detection had worked).
    let wasmGlue = null;
    try {
        const { createPipeline, env } = await freshTransformers(modelId);
        wasmGlue = overrideGlueForWasm(env);
        pipeline = await createPipeline('feature-extraction', modelId, { device: 'wasm', dtype: requestedDtype, ...(revision ? { revision } : {}) });
    } catch (e) {
        if (!String(e).includes('no available backend')) throw wasmFail(e);
        try {
            const { createPipeline, env } = await freshTransformers(modelId);
            wasmGlue = overrideGlueForWasm(env);
            env.backends.onnx.wasm.simd = false;
            env.backends.onnx.wasm.numThreads = 1;
            pipeline = await createPipeline('feature-extraction', modelId, { device: 'wasm', dtype: requestedDtype, ...(revision ? { revision } : {}) });
        } catch (e2) {
            throw wasmFail(e2);
        }
    }
    return {
        device: 'wasm', dtype: requestedDtype,
        coldStartMs: performance.now() - t0,
        // WASM path has no shader-compile warmup loop — kernels are JIT'd
        // on first call. null distinguishes "didn't run" from "ran in 0 ms".
        warmupMs: null,
        // warmupSkipped only ever true on the WebGPU success path; WASM
        // means "warmup didn't apply at all", which is different from
        // "warmup was skipped because we knew the shaders were cached".
        warmupSkipped: false,
        webgpuAttempted, webgpuError,
        glue: wasmGlue,
    };
}

async function embedText(text) {
    if (!pipeline) throw new Error('Model not loaded');
    const t0 = performance.now();
    // QUERY path (single-embed is query-only; indexing uses embedBatch below).
    // Tokenize the cleaned query and bucket off its EXACT token length on the
    // small-floor, top-trimmed query ladder (QUERY_SEQ_BUCKETS). Because the
    // bucket is ≥ the real token count, max_length never truncates — the whole
    // query is always embedded. Double-tokenization (here + inside pipeline) is
    // ~µs on a handful of tokens, negligible vs the ~20 ms forward. Fallback to
    // the char estimate only if the tokenizer accessor ever changes shape.
    let bucket;
    try {
        const dims = (await pipeline.tokenizer(text)).input_ids.dims;
        bucket = selectQueryBucket(dims[dims.length - 1]);
    } catch (_) {
        bucket = selectQueryBucket(Math.ceil(text.length / 4.5));
    }
    // wasm: padding:true on a single text = zero padding (exact length); the
    // bucket survives only as the truncation safety cap. webgpu: pad to the
    // warmed bucket shape as always.
    const output = await pipeline(text, {
        pooling: 'cls', normalize: true,
        padding: currentDevice === 'wasm' ? true : 'max_length',
        truncation: true, max_length: bucket,
    });
    // Fail loud on a model/dim misconfig: if the model's real output width is
    // SMALLER than OUTPUT_DIM, sliceAndRenormalize returns the short vector
    // unchanged (vec.length <= targetDim), which would silently corrupt the index.
    const outDim = output.dims[output.dims.length - 1];
    if (outDim < OUTPUT_DIM) throw new Error('embed: model output dim ' + outDim + ' < OUTPUT_DIM ' + OUTPUT_DIM + ' - model/dim misconfig');
    const vector = sliceAndRenormalize(output.data, OUTPUT_DIM);
    // Release the tensor's backing buffer (incl. WebGPU readback) — sliceAndRenormalize
    // already returned a fresh Float32Array, so the output is detached from the tensor.
    // See 2026-05-19 bog-down diagnosis: undisposed tensors accumulated ~600-800 MB
    // between V8 major-GC sweeps and were the iOS sustained-loop killer.
    if (typeof output.dispose === 'function') output.dispose();
    return { vector, latencyMs: performance.now() - t0 };
}

async function embedBatch(texts, explicitBucket) {
    if (!pipeline) throw new Error('Model not loaded');
    const t0 = performance.now();
    // INDEX path. The parent routes chunks into per-bucket buffers by EXACT
    // token count (token-counts RPC + selectIndexBucket) and passes the bucket
    // here, so max_length always holds every input — truncation:true below is
    // a pure safety cap that the token-budget packer keeps unreachable.
    // Fallback to the char estimate only for legacy callers that omit the
    // bucket (the estimate under-buckets dense text — see selectBucket).
    let bucket = explicitBucket;
    if (!bucket || !SEQ_BUCKETS.includes(bucket)) {
        const maxChars = texts.reduce((m, t) => Math.max(m, t.length), 0);
        bucket = selectBucket(maxChars);
    }
    // wasm: pad to the longest sequence in the batch (HF padding:true), not
    // the bucket rung — chunks in a buffer share a bucket so the intra-batch
    // spread is bounded by it, and every pad column saved is CPU work saved.
    // webgpu: 'max_length' (= the bucket) is load-bearing — Dawn's shader
    // cache and the SafeInt discipline are keyed to the warmed shape set.
    const output = await pipeline(texts, {
        pooling: 'cls', normalize: true,
        padding: currentDevice === 'wasm' ? true : 'max_length',
        truncation: true, max_length: bucket,
    });
    const dims = output.dims;
    const dim = dims[dims.length - 1];
    // Fail loud on a model/dim misconfig (see embedText): a model narrower than
    // OUTPUT_DIM would have sliceAndRenormalize emit short vectors per row.
    if (dim < OUTPUT_DIM) throw new Error('embedBatch: model output dim ' + dim + ' < OUTPUT_DIM ' + OUTPUT_DIM + ' - model/dim misconfig');
    const data = output.data;
    const vectors = [];
    for (let i = 0; i < texts.length; i++) {
        const row = new Float32Array(data.buffer, data.byteOffset + i * dim * 4, dim);
        vectors.push(sliceAndRenormalize(row, OUTPUT_DIM));
    }
    // Dispose AFTER the loop — each row above is a view into output.data.buffer,
    // not a copy; sliceAndRenormalize materializes a fresh OUTPUT_DIM-wide Float32Array per row,
    // so by the time we get here the tensor's storage is no longer referenced.
    // Releases the WebGPU readback buffer that would otherwise stay alive until V8 GC.
    if (typeof output.dispose === 'function') output.dispose();
    return { vectors, latencyMs: performance.now() - t0 };
}

// Exact token counts for the parent's index-path bucket routing and the
// token-budget packer. Counted one text at a time (no padding involved), same
// accessor the query path uses (embedText above), specials included — so
// "count <= bucket" is exactly "the forward pass sees every token". Tokenizer
// only; never touches the model, so it is safe on both WebGPU and WASM and
// adds no GPU-session pressure (no recycle interplay).
async function tokenCounts(texts) {
    // Prefer the full pipeline's tokenizer; fall back to a standalone tokenizer
    // loaded via load-tokenizer (hydrate path, no model). Same AutoTokenizer +
    // modelId either way, so counts are identical.
    const tokenizer = (pipeline && pipeline.tokenizer) || standaloneTokenizer;
    if (!tokenizer) throw new Error('no tokenizer available — load the model or call load-tokenizer first');
    const counts = [];
    for (const t of texts) {
        const enc = await tokenizer(t);
        const dims = enc.input_ids.dims;
        counts.push(dims[dims.length - 1]);
    }
    return counts;
}

// Pseudo-natural text of ~targetTokens length (chunker estimates 4.5 ch/tok).
// Timing of forward/post is shape-bound, not content-bound, but tokenizer
// cost IS content-sensitive — so use a realistic sentence, not a repeated
// single token (which would under-measure the tokenize step).
const PROFILE_SENTENCE =
    'The quarterly review covered retrieval relevance, embedding throughput, ' +
    'and the mobile memory budget; follow-ups were assigned across the team. ';
function makeProfileText(targetTokens) {
    const targetChars = Math.ceil(targetTokens * 4.5);
    let s = '';
    while (s.length < targetChars) s += PROFILE_SENTENCE;
    return s.slice(0, targetChars);
}

// Unrolled, timed twin of the embed path. Reaches into pipeline.tokenizer /
// pipeline.model (standard transformers.js FeatureExtractionPipeline surface)
// to put a clock between the stages. Deliberately does NOT dispose the output
// tensor — this harness is a NON-disposing baseline for diagnostic comparison
// against the production embedText/embedBatch paths (which DO dispose, see
// 2026-05-19 bog-down diagnosis). The heap delta the parent records around
// this run is the cost of a single embed's tensors NOT being released.
// (No backticks anywhere in this child block: it lives inside a template
// literal, so a stray backtick would terminate the script string.)
async function profileRuntime(batchSizes, seqBuckets, reps) {
    if (!pipeline) throw new Error('Model not loaded');
    const tokenizer = pipeline.tokenizer;
    const model = pipeline.model;
    if (!tokenizer || !model) {
        throw new Error('pipeline missing tokenizer/model — transformers.js API changed?');
    }
    const cells = [];
    for (const bs of batchSizes) {
        for (const bucket of seqBuckets) {
            const texts = Array(bs).fill(makeProfileText(bucket));
            const opts = {
                pooling: 'cls', normalize: true,
                padding: 'max_length', truncation: true, max_length: bucket,
            };
            const tokOpts = { padding: 'max_length', truncation: true, max_length: bucket };
            // Warm THIS exact (bs,bucket) shape once. load() already warms the
            // matrix, but a profile cell the warmup list missed would otherwise
            // fold a one-time WGSL compile into rep 0 and skew the p50.
            try { await pipeline(texts, opts); } catch (_) { /* compiled on first real use below */ }
            const tokenize = [], forward = [], post = [], pipe = [];
            for (let r = 0; r < reps; r++) {
                const a = performance.now();
                const inputs = await tokenizer(texts, tokOpts);
                const b = performance.now();
                const out = await model(inputs);
                const c = performance.now();
                // Force the GPU→CPU materialization. For a WebGPU-resident
                // tensor, touching .data is what actually pays the readback —
                // the sync stall the "per-inference serialization" hypothesis
                // points at. Measured separately from forward on purpose.
                const lhs = out.last_hidden_state || out.logits
                    || out.token_embeddings || out.sentence_embedding;
                const len = lhs && lhs.data ? lhs.data.length : 0;
                const d = performance.now();
                void len;
                // Production pipeline() total on identical inputs — the
                // decomposition sanity check (tokenize+forward+post ≈ pipe).
                const e = performance.now();
                await pipeline(texts, opts);
                const f = performance.now();
                tokenize.push(b - a);
                forward.push(c - b);
                post.push(d - c);
                pipe.push(f - e);
            }
            cells.push({ batchSize: bs, seqBucket: bucket, reps, tokenize, forward, post, pipe });
        }
    }
    return { cells };
}

// Collect the unique ArrayBuffers behind an embed result so the reply can
// TRANSFER them (zero-copy) instead of letting structured-clone deep-copy
// every vector. The reply round-trip is the heavy hop: result.vectors is
// batch x 384 x 4 bytes, paid once per dispatch (1245 dispatches @ budget 512).
//
// MUST dedup: at 384-d sliceAndRenormalize is a pass-through, so every row in
// a batch is a view into the SAME tensor buffer (embedBatch above) — adding
// that buffer once per row would transfer it N times and throw DataCloneError.
// A Set also covers the future MRL case where each sliced row owns its buffer
// (then the Set simply holds N distinct buffers). Strings (the request payload)
// aren't transferable, so this only optimizes the parent-bound direction.
function collectTransfer(result) {
    if (!result || typeof result !== 'object') return [];
    const bufs = new Set();
    if (result.vector && result.vector.buffer) bufs.add(result.vector.buffer);
    if (Array.isArray(result.vectors)) {
        for (const v of result.vectors) { if (v && v.buffer) bufs.add(v.buffer); }
    }
    return [...bufs];
}

window.addEventListener('message', async (event) => {
    // Mirror the parent-side source check (buildIframe()'s listener): only
    // dispatch RPCs that actually came from the window that embedded us.
    // Currently unreachable (no other frame holds a reference to post from),
    // but cheap to harden so a future embedding context can't slip messages
    // straight into the RPC dispatcher.
    if (event.source !== window.parent) return;
    const data = event.data;
    if (!data || typeof data !== 'object' || !data.id || !data.type) return;
    try {
        let result;
        if (data.type === 'load') {
            result = await loadModel(
                data.payload.modelId,
                data.payload.device,
                data.payload.dtype,
                data.payload.skipWarmup,
                data.payload.revision,
            );
            currentDevice = result.device;
        } else if (data.type === 'embed') {
            result = await embedText(data.payload.text);
        } else if (data.type === 'embed-batch') {
            result = await embedBatch(data.payload.texts, data.payload.bucket);
        } else if (data.type === 'token-counts') {
            result = await tokenCounts(data.payload.texts);
        } else if (data.type === 'load-tokenizer') {
            result = await loadTokenizer(data.payload.modelId, data.payload.revision);
        } else if (data.type === 'embed-profile') {
            result = await profileRuntime(
                data.payload.batchSizes, data.payload.seqBuckets, data.payload.reps);
        } else if (data.type === 'app-local-fetch') {
            // Probe — never throws to the parent; the failure cases ARE the data.
            // We want { ok: false, error } back, not a rejected RPC.
            const url = data.payload.url;
            try {
                const res = await fetch(url);
                let body = null;
                try { body = await res.text(); } catch (_) { /* body unreadable; status alone is signal */ }
                result = { ok: res.ok, status: res.status, body, error: null };
            } catch (e) {
                result = { ok: false, status: null, body: null, error: String(e) };
            }
        } else {
            throw new Error('unknown type: ' + data.type);
        }
        // Transfer the vector buffers (embed/embed-batch) so the parent gets
        // them by move, not copy. Other RPCs (load/profile/fetch) carry no
        // large buffers, so collectTransfer returns [] and this is a no-op.
        // After transfer the buffers are detached here — safe because the
        // tensor was already disposed and we never touch result again.
        window.parent.postMessage(
            { id: data.id, ok: true, result }, '*', collectTransfer(result));
    } catch (e) {
        window.parent.postMessage({
            id: data.id, ok: false,
            error: String(e),
            stack: e && e.stack ? e.stack : null,
        }, '*');
    }
});

try {
    window.parent.postMessage({ id: '__ready' }, '*');
} catch (e) {
    window.parent.postMessage({ id: '__error', error: String(e) }, '*');
}
`;
}
