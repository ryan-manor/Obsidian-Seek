// EXPERIMENTAL query-only-GPU mode ('webgpu-query'): WebGPU for the query path,
// a one-time swap to wasm before any indexing batch. Pins the four contracts:
// device resolution, the swap (once, only flag+webgpu, wasm-replayed), the
// per-profile warmup fingerprint, and the child's query-profile warmup branch.

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { Platform } from 'obsidian';
import { resolveDevice, setBackendOverride, isQueryOnlyGpu, isWebgpuDemoted, maybeDemoteOnCrash, recordActiveBackend } from './platform';
import { LocalEmbedder, warmupFingerprint, type DeviceSwapInfo } from './embedder';
import { buildChildScript } from './iframe-runner';

function installLocalStorage(): void {
    let store: Record<string, string> = {};
    const ls = {
        getItem: (k: string) => (k in store ? store[k] : null),
        setItem: (k: string, v: string) => { store[k] = String(v); },
        removeItem: (k: string) => { delete store[k]; },
        clear: () => { store = {}; },
    };
    vi.stubGlobal('localStorage', ls);
    // embedder.ts reads window.localStorage for the warmup fingerprint.
    vi.stubGlobal('window', { localStorage: ls });
}

function iphone(): void {
    Platform.isMobile = true;
    Platform.isIosApp = true;
    Platform.isTablet = false;
}

beforeEach(() => {
    installLocalStorage();
    iphone();
});

describe('resolveDevice — webgpu-query', () => {
    it('iPhone default stays wasm; webgpu-query opts it into the auto (WebGPU) ladder', () => {
        expect(resolveDevice()).toBe('wasm');
        expect(isQueryOnlyGpu()).toBe(false);
        setBackendOverride('webgpu-query');
        expect(resolveDevice()).toBe('auto');
        expect(isQueryOnlyGpu()).toBe(true);
    });

    it('explicit opt-in beats (and clears) a sticky crash demote', () => {
        recordActiveBackend('webgpu');
        expect(maybeDemoteOnCrash('crash-while-indexing-foreground')).toBe(true);
        expect(isWebgpuDemoted()).toBe(true);
        setBackendOverride('webgpu-query');
        expect(isWebgpuDemoted()).toBe(false);
        expect(resolveDevice()).toBe('auto');
    });

    it('a crash after the swap restamped wasm does not demote', () => {
        setBackendOverride('webgpu-query');
        recordActiveBackend('webgpu');   // query session loaded on GPU
        recordActiveBackend('wasm');     // onDeviceSwap restamps before indexing
        expect(maybeDemoteOnCrash('crash-while-indexing-foreground')).toBe(false);
    });
});

describe('warmupFingerprint — per profile', () => {
    it('query and full profiles never share a fingerprint; full is the legacy string', () => {
        const full = warmupFingerprint('m', 'q4', 'rev');
        expect(warmupFingerprint('m', 'q4', 'rev', 'full')).toBe(full);
        const q = warmupFingerprint('m', 'q4', 'rev', 'query');
        expect(q).not.toBe(full);
        expect(q.startsWith(full)).toBe(true);
    });
});

// Runner stub: each load() resolves to the device it was asked for ('auto' →
// webgpu), records the requested device + warmup profile, and counts batches.
function swapRunner() {
    const loads: { device: string; profile: string }[] = [];
    let batches = 0;
    let disposes = 0;
    return {
        loads,
        batches: () => batches,
        disposes: () => disposes,
        dispose: () => { disposes++; },
        init: async () => ({ buildTimestamp: 't', cdnUrl: 'c', transformersVersion: '4.2.0', ready: true, error: null, initMs: 1 }),
        load: async (_m: string, device: string, _d: string, _s: boolean, _r: string | null, profile: string) => {
            loads.push({ device, profile });
            await new Promise((r) => setTimeout(r, 5));
            return {
                device: device === 'wasm' ? 'wasm' : 'webgpu', dtype: 'q4', coldStartMs: 1,
                warmupMs: null, warmupSkipped: false,
                webgpuAttempted: device !== 'wasm', webgpuError: null, glue: null,
            };
        },
        embedBatch: async (texts: string[]) => {
            batches++;
            return { vectors: texts.map(() => new Float32Array(4)), latencyMs: 1 };
        },
    };
}

function mkSwap(queryOnlyGpu: boolean) {
    const e = new LocalEmbedder();
    const fr = swapRunner();
    (e as unknown as { runner: unknown }).runner = fr;
    e.queryOnlyGpu = queryOnlyGpu;
    const swaps: DeviceSwapInfo[] = [];
    e.onDeviceSwap = (i) => swaps.push(i);
    return { e, fr, swaps };
}

describe('LocalEmbedder query-only GPU swap', () => {
    it('flag + webgpu: loads with the query profile, swaps to wasm ONCE across concurrent batches', async () => {
        const { e, fr, swaps } = mkSwap(true);
        await e.load('auto', 'q4');
        expect(e.device).toBe('webgpu');
        expect(fr.loads[0]).toEqual({ device: 'auto', profile: 'query' });

        await Promise.all([e.embedBatch(['a'], 32), e.embedBatch(['b'], 32), e.embedBatch(['c'], 32)]);
        expect(e.device).toBe('wasm');
        expect(fr.loads.map((l) => l.device)).toEqual(['auto', 'wasm']);
        expect(fr.loads[1].profile).toBe('full');   // wasm has no warmup grid
        expect(fr.disposes()).toBe(1);              // the GPU iframe was torn down
        expect(fr.batches()).toBe(3);
        expect(swaps).toHaveLength(1);
        expect(swaps[0].from).toBe('webgpu');
        expect(swaps[0].to).toBe('wasm');
        expect(swaps[0].entry.actualDevice).toBe('wasm');

        // Subsequent batches stay on wasm, and a recovery recycle replays wasm.
        await e.embedBatch(['d'], 32);
        await e.recycle();
        expect(fr.loads.map((l) => l.device)).toEqual(['auto', 'wasm', 'wasm']);
        expect(swaps).toHaveLength(1);

        // A fresh load() (idle unload → ensureModelLoaded) returns to WebGPU.
        await e.load('auto', 'q4');
        expect(e.device).toBe('webgpu');
    });

    it('flag off: a webgpu session indexes on webgpu, no swap, full profile', async () => {
        const { e, fr, swaps } = mkSwap(false);
        await e.load('auto', 'q4');
        await e.embedBatch(['a'], 32);
        expect(e.device).toBe('webgpu');
        expect(fr.loads).toEqual([{ device: 'auto', profile: 'full' }]);
        expect(swaps).toHaveLength(0);
    });

    it('flag on but already wasm (WebGPU fell back): no swap', async () => {
        const { e, fr, swaps } = mkSwap(true);
        await e.load('wasm', 'q4');
        await e.embedBatch(['a'], 32);
        expect(fr.loads.map((l) => l.device)).toEqual(['wasm']);
        expect(swaps).toHaveLength(0);
    });
});

describe('child script — query warmup profile', () => {
    it('loadModel takes the profile and branches to a batch-1 QUERY_SEQ_BUCKETS-only warmup', () => {
        const s = buildChildScript('https://example.com/cdn', 384);
        expect(s).toContain('async function loadModel(modelId, requestedDevice, requestedDtype, skipWarmup, revision, warmupProfile)');
        expect(s).toContain("if (!skipWarmup && warmupProfile === 'query')");
        expect(s).toContain("data.payload.warmupProfile === 'query' ? 'query' : 'full'");
    });
});
