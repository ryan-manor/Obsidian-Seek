// One-shot platform probe — logged on every plugin load so the report
// always knows what surface generated each run. Mirrors the iOS spike.

import { Platform } from 'obsidian';
import type { PlatformEntry, AdapterLimits } from './types';

// Single source of truth for the mobile-platform test. Used by the platform
// probe and by the indexer's device-adaptive embed batch ceiling, so the two
// can never disagree about what "mobile" means.
export function isMobilePlatform(): boolean {
    return Platform.isMobile;
}

// ── Per-device compute-backend selection ───────────────────────────────────
//
// WebGPU viability is a property of the DEVICE, not the vault — an iPad runs a
// granite reindex on WebGPU at desktop speed, an iPhone gets jetsam-killed mid
// reindex (~1.5 GB WebContent budget), and Android System WebView's WebGPU is
// immature/absent. But settings persisted via Plugin.saveData live in the
// vault's data.json, which iCloud (and Obsidian Sync's plugin-settings option)
// replicate to every device — so a backend toggle there would be SHARED: turn
// it on for the iPad and the iPhone inherits it on the next sync. That is the
// same shared-file trap as the per-device logs (iCloud clobbered the shared
// NDJSON) and the shared-IDB incident.
//
// So the backend choice lives in localStorage, which is per-ORIGIN
// (capacitor://localhost on iOS, the Electron origin on desktop) → per physical
// device and NEVER synced — exactly the deviceId / crash-forensics convention.

export type BackendChoice = 'auto' | 'webgpu' | 'wasm';

const OVERRIDE_KEY = 'seek-backend-override';  // BackendChoice; absent/invalid = auto
const DEMOTED_KEY = 'seek-webgpu-demoted';     // '1' once a mobile WebGPU reindex was OS-killed
const ACTIVE_KEY = 'seek-active-backend';      // 'webgpu' | 'wasm' — backend the last load resolved to

// Every localStorage access below is raw per-origin (`window.localStorage`) BY
// DESIGN — these are per-device, never-synced backend keys (see the section
// comment above). We deliberately do NOT use Obsidian's `App#saveLocalStorage`,
// which vault-scopes the value; that is exactly the shared-file trap this design
// avoids.

// User's explicit per-device override, or 'auto' (absent / unreadable).
export function getBackendOverride(): BackendChoice {
    try {
        const v = window.localStorage.getItem(OVERRIDE_KEY);
        if (v === 'webgpu' || v === 'wasm' || v === 'auto') return v;
    } catch { /* localStorage unavailable — treat as auto */ }
    return 'auto';
}

export function setBackendOverride(choice: BackendChoice): void {
    try {
        window.localStorage.setItem(OVERRIDE_KEY, choice);
        // Re-opting into WebGPU clears any sticky demote: the user is
        // explicitly asking this device to try the GPU again, so honour it.
        if (choice === 'webgpu') clearWebgpuDemoted();
    } catch { /* best-effort */ }
}

export function isWebgpuDemoted(): boolean {
    try { return window.localStorage.getItem(DEMOTED_KEY) === '1'; } catch { return false; }
}
export function clearWebgpuDemoted(): void {
    try { window.localStorage.removeItem(DEMOTED_KEY); } catch { /* best-effort */ }
}

// Stamp the backend the model actually loaded on. Read at next boot by
// maybeDemoteOnCrash to decide whether a crash implicates WebGPU.
export function recordActiveBackend(device: string): void {
    try { window.localStorage.setItem(ACTIVE_KEY, device === 'webgpu' ? 'webgpu' : 'wasm'); }
    catch { /* best-effort */ }
}

// Capability ALLOWLIST — WebGPU auto-enables ONLY on surfaces verified to
// reindex without an OS memory kill. NOT "mobile minus iPhone": anything
// unverified (every Android, an iPad we haven't characterised stays on the
// tablet branch but Android tablets do NOT) defaults to WASM until proven good.
//   - Desktop (Electron): WebGPU — the long-standing default, ~2× WASM.
//   - iPad: WebGPU — ml97 GBQ4 reindexes desktop-class on WKWebView (2026-06-10),
//     ~4 GB WebContent budget.
//   - iPhone + ALL Android: WASM — jetsam at ~1.5 GB / immature WebView WebGPU.
function capabilityDefault(): 'auto' | 'wasm' {
    if (!Platform.isMobile) return 'auto';                       // desktop
    if (Platform.isIosApp && Platform.isTablet) return 'auto';   // iPad
    return 'wasm';                                               // iPhone + all Android
}

// The device string to hand the embedder ('auto' = WebGPU-then-WASM ladder;
// 'wasm' = skip WebGPU). Override wins; else the allowlist; with a sticky
// mobile demote forcing WASM on the auto path. A 'webgpu' override maps to
// 'auto' — the iframe ladder still WASM-falls-back if the GPU can't load.
export function resolveDevice(): 'auto' | 'wasm' {
    const override = getBackendOverride();
    if (override === 'wasm') return 'wasm';
    if (override === 'webgpu') return 'auto';
    const base = capabilityDefault();
    if (base === 'auto' && Platform.isMobile && isWebgpuDemoted()) return 'wasm';
    return base;
}

// Boot-time tripwire. Demote (sticky, per-device) only when a MOBILE device was
// killed mid-reindex in the FOREGROUND (= memory-ceiling, the iPhone jetsam
// signature) WHILE WebGPU was the active backend. Never on desktop, and never
// when the kill happened on WASM (demoting WebGPU there is a no-op that would
// falsely read as a fix). Returns true if it tripped.
export function maybeDemoteOnCrash(verdict: string): boolean {
    if (verdict !== 'crash-while-indexing-foreground') return false;
    if (!Platform.isMobile) return false;
    try {
        if (window.localStorage.getItem(ACTIVE_KEY) !== 'webgpu') return false;
        window.localStorage.setItem(DEMOTED_KEY, '1');
        return true;
    } catch { return false; }
}

// Device-adaptive embed batch ceiling. See Seek Model Performance.md
// §"Larger embed batch" and its resolved [!done] callout: batch size is a
// throughput-vs-device-burden knob, not pure throughput.
//
//   - Mobile → 8. Large batches spike a phone's GPU power envelope, causing
//     thermal throttle (which *slows* the overall index) and battery drain.
//     8 is the top of the doc's 4–8 band and still 2× gentler than the old
//     fixed 16; with within-file batching + chunks/file p50=1, most mobile
//     batches are far below this ceiling anyway.
//   - Desktop → 32. The conservative floor of the doc's 32–64 band. We keep
//     within-file batching (the doc's callout endorses it — cross-file
//     pooling hit 14.2 vs 13.2 ch/s but spun the fan), so this ceiling only
//     bites the ~5% long-tail of files with >32 chunks. Widen toward 64 only
//     once the post-PTQ-swap re-bench gives a thermal/throughput baseline —
//     same "measure before widening" discipline as the seq-length cap (#2).
export function embedBatchCeiling(): number {
    return isMobilePlatform() ? 8 : 32;
}

// ── Resident int8 rerank block: memory gate (B2) ────────────────────────────
//
// buildResidentRerankBlock (search.ts) holds one contiguous block of
// n*embDim int8 bytes + n*8 float64 scale bytes = n*(embDim+8) bytes resident
// for the whole session. At 50k chunks / d=384 that single allocation is
// ~19 MB, which crowds the iOS ~50 MB-per-allocation jetsam ceiling. The block
// is a pure speedup over the per-id IDB read in getEmbeddingsByIds — stage-2
// falls back to that path BYTE-IDENTICALLY (same dequantizeInt8, Float64
// scales) when the block is absent. So gating it off costs a little latency,
// never relevance.
//
// Policy is automatic (no user setting): OFF on mobile — that's where the
// jetsam ceiling lives and where the bounded IDB fetch is the safer shape — and
// OFF on any device once the block would exceed RESIDENT_INT8_MAX_BYTES. A byte
// budget, not a raw chunk count, because embDim is model-dependent (384 today,
// was 512 before the granite swap).
export const RESIDENT_INT8_MAX_BYTES = 16 * 1024 * 1024;  // ~40k chunks @ d=384

export function residentInt8Enabled(rowCount: number, embDim: number): boolean {
    if (isMobilePlatform()) return false;
    return rowCount * (embDim + 8) <= RESIDENT_INT8_MAX_BYTES;
}

export async function collectPlatformInfo(): Promise<PlatformEntry> {
    const nav = window.navigator;
    const ua = nav.userAgent;
    const isMobile = isMobilePlatform();

    // iOS version: best-effort parse of "OS X_Y_Z" from the UA. CAVEAT: WKWebView
    // freezes/caps this token (Apple, to defeat version sniffing), so it is NOT
    // the real OS version — an iPhone on iOS 26.5 reports "18_7" here (confirmed
    // 2026-06-08, iPhone 15 Pro). Treat as a lower bound only; nothing in the load
    // path should branch on it (device selection keys off isMobilePlatform, not
    // this). A non-null WebGPU adapter (below) is a better "iOS ≥ 26" signal.
    let iosVersion: number | null = null;
    const iosM = /OS (\d+)[_.](\d+)/.exec(ua);
    if (iosM) iosVersion = parseFloat(`${iosM[1]}.${iosM[2]}`);

    // WebGPU probe — adapter creation is the right signal here; navigator.gpu
    // existing isn't sufficient on iOS WKWebView (it's defined but requestAdapter
    // can still return null pre-iOS-26).
    let gpuAvailable = false;
    let gpuAdapterDescription: string | null = null;
    let gpuIsFallbackAdapter: boolean | null = null;
    let gpuAdapterLimits: AdapterLimits | null = null;
    interface MaybeAdapter {
        // Spec: GPUAdapter.isFallbackAdapter; some engines mirror it on
        // GPUAdapterInfo instead — probe both.
        isFallbackAdapter?: boolean;
        info?: { description?: string; isFallbackAdapter?: boolean };
        requestAdapterInfo?: () => Promise<{ description?: string }>;
        limits?: {
            maxBufferSize?: number;
            maxStorageBufferBindingSize?: number;
            maxComputeWorkgroupSizeX?: number;
            maxComputeInvocationsPerWorkgroup?: number;
        };
    }
    interface MaybeGpuNav {
        gpu?: { requestAdapter: () => Promise<MaybeAdapter | null> };
    }
    const gpuNav = navigator as unknown as MaybeGpuNav;
    if (gpuNav.gpu) {
        try {
            const adapter = await gpuNav.gpu.requestAdapter();
            if (adapter) {
                gpuAvailable = true;
                // A non-null adapter can still be a SOFTWARE fallback
                // (SwiftShader-class, e.g. hardware acceleration off) that
                // ORT's WebGPU init then rejects — without this flag the
                // report's "GPU yes" reads as a contradiction of a WebGPU
                // load failure (r/ObsidianMD triage, 2026-07-03).
                if (typeof adapter.isFallbackAdapter === 'boolean') {
                    gpuIsFallbackAdapter = adapter.isFallbackAdapter;
                } else if (typeof adapter.info?.isFallbackAdapter === 'boolean') {
                    gpuIsFallbackAdapter = adapter.info.isFallbackAdapter;
                }
                try {
                    if (adapter.requestAdapterInfo) {
                        const info = await adapter.requestAdapterInfo();
                        gpuAdapterDescription = info?.description ?? null;
                    } else if (adapter.info) {
                        gpuAdapterDescription = adapter.info.description ?? null;
                    }
                } catch {
                    // Cosmetic — adapter info isn't always available (iOS 18 WKWebView).
                }
                // Adapter limits change between iOS versions and predict OOMs
                // — record what we got so the report can spot regressions
                // when transformers.js or model bumps push past the buffer cap.
                if (adapter.limits) {
                    gpuAdapterLimits = {
                        maxBufferSize: adapter.limits.maxBufferSize ?? null,
                        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize ?? null,
                        maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX ?? null,
                        maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup ?? null,
                    };
                }
            }
        } catch {
            gpuAvailable = false;
        }
    }

    // Storage estimate. Not all browsers expose `estimate()`.
    let storageUsedMB: number | null = null;
    let storageQuotaMB: number | null = null;
    let persistGranted: boolean | null = null;
    if (navigator.storage?.estimate) {
        try {
            const est = await navigator.storage.estimate();
            storageUsedMB = est.usage != null ? est.usage / 1e6 : null;
            storageQuotaMB = est.quota != null ? est.quota / 1e6 : null;
        } catch { /* swallow */ }
    }
    if (navigator.storage?.persisted) {
        try { persistGranted = await navigator.storage.persisted(); } catch { /* swallow */ }
    }

    // Memory-API availability flags. These determine which fields the report
    // will actually have data for — on iOS WebKit, `heapAvailable: false`
    // is the canary that explains why every heapDeltaMB is null.
    const heapAvailable = (performance as unknown as { memory?: unknown }).memory != null;
    const measureMemoryAvailable = typeof (performance as unknown as {
        measureUserAgentSpecificMemory?: unknown;
    }).measureUserAgentSpecificMemory === 'function';
    const crossOriginIsolated = (window as unknown as { crossOriginIsolated?: boolean }).crossOriginIsolated === true;

    return {
        type: 'platform',
        timestamp: new Date().toISOString(),
        isMobile,
        userAgent: ua,
        iosVersion,
        gpuAvailable,
        gpuAdapterDescription,
        gpuIsFallbackAdapter,
        gpuAdapterLimits,
        storageUsedMB,
        storageQuotaMB,
        persistGranted,
        heapAvailable,
        measureMemoryAvailable,
        crossOriginIsolated,
    };
}
