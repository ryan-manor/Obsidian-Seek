// Production model delivery — the content-addressed registry of shippable
// embedding models plus the pure helpers main.ts uses to (a) pick the active
// model (honouring a debug override) and (b) evict a previous model's bytes from
// the transformers.js Cache API on a switch.
//
// WHY a registry. The ~100 MB ONNX model is fetched at runtime by transformers.js
// inside the embed iframe (marketplace plugins ship only main.js/manifest/css, so
// the model can't be bundled). transformers.js streams it from the HF CDN and
// caches it in the browser Cache API ('transformers-cache') — per-device, never
// synced, outside the vault: the Smart Connections pattern, and on iOS the only
// store that can hold 100 MB (requestUrl can't stream that on mobile, and a plugin
// can't write outside the vault sandbox). The registry adds the missing management
// layer: versioning, multi-repo support, and eviction of the old model on a switch.
//
// IDENTITY == repo. embedder.MODEL_ID is derived from the active spec's `repo` and
// remains the index's drift-identity stamp, so pointing ACTIVE_MODEL_KEY at a
// different repo fires the EXISTING reindex machinery (warnOnModelIndexDrift) for
// free — no churn to the drift / sidecar-version-gate code. `revision` is now
// threaded into the transformers.js load (createPipeline/from_pretrained both take
// a `revision` option, verified against tx.js 4.2.0) AND into the sidecar version
// gate, so a pinned commit sha makes embeddings reproducible across devices/time
// and refuses cross-revision sidecar hydration (F10). Eviction still matches on
// repo alone (a revision bump's stale bytes are reclaimed by the OS / next switch).

import type { Dtype, SeekSettings } from './types';

export interface ModelSpec {
    // Stable identity for the index drift-stamp + storage namespacing. For shipped
    // models this equals `repo`, so MODEL_ID === repo === identity and a switch
    // needs no special drift handling.
    key: string;
    // HF hub id (or a full base URL). The string handed to embedder.load() as the
    // load base on the remote path → transformers.js streams it from the CDN.
    repo: string;
    // Pinned commit sha/tag. Metadata for now (eviction + future URL pinning); not
    // yet passed to transformers.js, which resolves `main`. null = track main.
    revision: string | null;
    // Relative paths transformers.js requests under the repo root. Documentation +
    // future integrity checking; the iframe derives the actual list from dtype.
    files: string[];
    dim: number;
    dtype: Dtype;
}

// The ml97 GBQ-int4 model shipped today (mirrors the legacy embedder.MODEL_ID
// string exactly, so deriving MODEL_ID from this is a no-op for the index identity).
export const ML97_GBQ4: ModelSpec = {
    key: 'tooape/granite-embedding-97m-multilingual-r2-GBQ4-ONNX',
    repo: 'tooape/granite-embedding-97m-multilingual-r2-GBQ4-ONNX',
    // Pinned to an immutable commit sha (F10): transformers.js fetches resolve/<sha>
    // and the sidecar gate refuses cross-revision hydration, so two devices can't mix
    // vector spaces if the repo's `main` ever moves. Bump this when shipping a new
    // build (fires model-vs-index drift → full reindex). NOTE: changing it re-fetches
    // the model bytes once — the resolve URL, and thus the Cache-API key, changes.
    revision: '54db88c5667bd79b4aea24ea6027a7ef45a7bbb5',
    files: [
        'config.json',
        'tokenizer.json',
        'tokenizer_config.json',
        'special_tokens_map.json',
        'onnx/model_q4.onnx',
    ],
    dim: 384,
    dtype: 'q4',
};

export const MODEL_REGISTRY: Record<string, ModelSpec> = {
    [ML97_GBQ4.key]: ML97_GBQ4,
};

// The model Seek ships with by default. A code-level model switch = point this at
// a different registered key (and add its ModelSpec) — MODEL_ID follows, and the
// existing model-vs-index drift check routes every device to a full reindex.
export const ACTIVE_MODEL_KEY = ML97_GBQ4.key;

// The default active spec (no settings override). embedder.MODEL_ID derives from
// this so the index identity stamp stays a plain module constant.
export const ACTIVE_MODEL_SPEC = MODEL_REGISTRY[ACTIVE_MODEL_KEY];

// Build a one-off spec from the debug-only settings override (testing arbitrary
// repos). Returns null when no override is set. Identity/key = the override repo,
// so an override load drifts vs the stored index (→ reindex), same as a real swap.
export function resolveOverrideSpec(settings: SeekSettings): ModelSpec | null {
    const repo = settings.modelRepoOverride?.trim();
    if (!repo) return null;
    return {
        key: repo,
        repo,
        revision: settings.modelRevisionOverride?.trim() || null,
        files: ML97_GBQ4.files, // assume the standard transformers.js repo layout
        dim: ML97_GBQ4.dim,
        dtype: ML97_GBQ4.dtype,
    };
}

// The model to load right now: debug override wins, else the shipped default.
export function activeModelSpec(settings: SeekSettings): ModelSpec {
    return resolveOverrideSpec(settings) ?? ACTIVE_MODEL_SPEC;
}

// ---- Cache-API eviction (parent-side; orchestrated from main.ts) --------------
// transformers.js caches model files in caches.open('transformers-cache') keyed by
// the HF resolve URL `https://huggingface.co/<repo>/resolve/<rev>/<file>` (verified
// against the shipped @huggingface/transformers@4.2.0 bundle). On a model switch we
// delete every cached HF model request that is NOT the active repo, reclaiming the
// old model's ~100 MB. Pure predicate below is unit-tested; the orchestration that
// opens `caches` lives in main.ts (and is benign if the cache is absent).

export const TRANSFORMERS_CACHE_NAME = 'transformers-cache';

// True if `url` is a transformers HF model fetch for some repo OTHER than keepRepo.
// Matches the repo as a path segment (`/<repo>/resolve/`) so a repo that is a prefix
// of another (…/granite vs …/granite-2) is not mistakenly kept or evicted.
export function shouldEvictCacheUrl(url: string, keepRepo: string): boolean {
    if (!url.includes('huggingface.co') || !url.includes('/resolve/')) return false;
    return !url.includes(`/${keepRepo}/resolve/`);
}

export interface EvictionResult { seen: number; deleted: number }

// True if `url` is a transformers HF model fetch FOR `repo` — the entries a user-invoked
// "Delete model" removes. The mirror of shouldEvictCacheUrl: same HF/resolve guard and
// same path-segment match (`/<repo>/resolve/`, prefix-safe), but the predicate is
// inverted from "every repo but this one" to "exactly this one".
export function isCacheUrlForRepo(url: string, repo: string): boolean {
    if (!url.includes('huggingface.co') || !url.includes('/resolve/')) return false;
    return url.includes(`/${repo}/resolve/`);
}

// Delete the ACTIVE model's cached bytes (settings "Delete model"). The inverse of
// evictStaleModelCaches — there we drop every repo EXCEPT keepRepo; here we drop exactly
// the entries for `repo`, leaving the jsdelivr runtime and any other repo untouched.
// Best-effort + benign + parent-side, with the same iPhone visibility caveat as eviction
// (seen === 0 → the parent can't reach the iframe's cache; the bytes stay until reload).
export async function deleteModelCaches(
    caches: CacheStorage,
    repo: string,
): Promise<EvictionResult> {
    if (!(await caches.has(TRANSFORMERS_CACHE_NAME))) return { seen: 0, deleted: 0 };
    const cache = await caches.open(TRANSFORMERS_CACHE_NAME);
    const reqs = await cache.keys();
    let deleted = 0;
    for (const req of reqs) {
        if (isCacheUrlForRepo(req.url, repo)) {
            // Best-effort per entry: a single rejected delete (rare) must not abort the
            // sweep and strand the rest of the model's bytes half-removed. Worst case we
            // under-count `deleted`; the next delete / search retries the leftovers.
            try { if (await cache.delete(req)) deleted++; } catch { /* skip this key */ }
        }
    }
    return { seen: reqs.length, deleted };
}

// Open the transformers cache and delete stale-repo entries. Best-effort + benign:
// if the cache is absent (never populated, or — should not happen with our
// non-sandboxed, same-origin iframe — partitioned away from the parent) it returns
// {0,0} and the old bytes are left for the OS to reclaim. The caller logs the
// result so `seen === 0` is visible (the signal to move eviction into an iframe RPC
// if it ever shows up on a real device). Typed against the DOM `CacheStorage`; unit
// tests inject a structural fake via `as unknown as CacheStorage`.
export async function evictStaleModelCaches(
    caches: CacheStorage,
    keepRepo: string,
): Promise<EvictionResult> {
    if (!(await caches.has(TRANSFORMERS_CACHE_NAME))) return { seen: 0, deleted: 0 };
    const cache = await caches.open(TRANSFORMERS_CACHE_NAME);
    const reqs = await cache.keys();
    let deleted = 0;
    for (const req of reqs) {
        if (shouldEvictCacheUrl(req.url, keepRepo)) {
            if (await cache.delete(req)) deleted++;
        }
    }
    return { seen: reqs.length, deleted };
}

// ---- Cache-API download probe (read-only; for the settings model status) -------
export interface ModelCacheStatus {
    // The model's ~100 MB ONNX weights are present in the Cache API (vs. an aborted
    // fetch that left only the small JSON configs, or nothing). This is the
    // "Downloaded — survives reloads, evictable by iOS" state the settings tab shows,
    // distinct from "loaded into runtime memory" (which the search modal signals).
    downloaded: boolean;
    // navigator.storage persistence: true = persistent-storage granted, so the cache
    // won't be silently evicted under pressure ("stored permanently"); false = "may be
    // evicted"; null = the API is unavailable (don't render the nuance).
    persisted: boolean | null;
}

// Read-only probe of whether the active model is cached on disk. Matches the ONNX
// weight file by repo + filename FRAGMENTS in the cache keys, rather than rebuilding
// the exact resolve URL — the revision segment is `main` vs a pinned sha depending on
// the load path, and fragment-matching is robust to either (same approach as
// shouldEvictCacheUrl). Best-effort and NEVER throws: any failure (no Cache API,
// origin-partitioned cache, rejected open) resolves to { downloaded:false } so it can
// never blank the settings tab. NOTE the parent-side-visibility caveat — on a real
// iPhone the parent may not see the iframe's cache (the `cacheSeen` canary in the
// model-delivery log); the caller falls back to that log when this returns false.
export async function probeModelDownloaded(
    caches: CacheStorage,
    spec: ModelSpec,
): Promise<ModelCacheStatus> {
    let persisted: boolean | null = null;
    try {
        if (typeof navigator !== 'undefined' && navigator.storage?.persisted) {
            persisted = await navigator.storage.persisted();
        }
    } catch { /* unsupported / private mode */ }
    try {
        if (!(await caches.has(TRANSFORMERS_CACHE_NAME))) return { downloaded: false, persisted };
        const cache = await caches.open(TRANSFORMERS_CACHE_NAME);
        const reqs = await cache.keys();
        // The largest file is the ONNX weights; its presence is the download canary
        // (the small JSON configs alone = a partial/aborted fetch). Fall back to the
        // last declared file if a spec ever ships without an .onnx entry.
        const weightFile = spec.files.find(f => f.endsWith('.onnx')) ?? spec.files[spec.files.length - 1];
        const downloaded = reqs.some(r =>
            r.url.includes(`/${spec.repo}/resolve/`) && r.url.includes(weightFile));
        return { downloaded, persisted };
    } catch {
        return { downloaded: false, persisted };
    }
}
