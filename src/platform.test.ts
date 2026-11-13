// Per-device compute-backend resolution: the capability allowlist, the manual
// override precedence, and the crash-demote tripwire. These decide WebGPU vs
// WASM per physical device, so the never-ORT-WebGPU-on-phones/Android and
// demote-only-on-WebGPU assertions are the load-bearing ones.

import { describe, it, expect, beforeEach, vi } from 'vitest';
// Aliased to src/test-stubs/obsidian.ts via vitest.config.mts — the SAME object
// platform.ts imports, so mutating it here poses as a device class at call time.
import { Platform } from 'obsidian';

import {
    resolveDevice,
    getBackendOverride,
    setBackendOverride,
    isWebgpuDemoted,
    clearWebgpuDemoted,
    recordActiveBackend,
    maybeDemoteOnCrash,
    isUnexpectedGpuLoss,
    residentInt8Enabled,
    RESIDENT_INT8_MAX_BYTES,
} from './platform';

// Minimal in-memory localStorage (node has no DOM). Reset per test.
function installLocalStorage(): void {
    let store: Record<string, string> = {};
    vi.stubGlobal('localStorage', {
        getItem: (k: string) => (k in store ? store[k] : null),
        setItem: (k: string, v: string) => { store[k] = String(v); },
        removeItem: (k: string) => { delete store[k]; },
        clear: () => { store = {}; },
    });
}

// Pose as a device class.
function setDevice(kind: 'desktop' | 'ipad' | 'iphone' | 'android-phone' | 'android-tablet'): void {
    Platform.isMobile = kind !== 'desktop';
    Platform.isIosApp = kind === 'ipad' || kind === 'iphone';
    Platform.isTablet = kind === 'ipad' || kind === 'android-tablet';
}

beforeEach(() => {
    installLocalStorage();
    setDevice('desktop');
});

describe('capability allowlist (no override, no demote)', () => {
    it('desktop → Seek engine, ORT ladder fallback', () => {
        setDevice('desktop');
        expect(resolveDevice()).toBe('wgsl');
    });

    it('iPad → Seek engine, ORT ladder fallback', () => {
        setDevice('ipad');
        expect(resolveDevice()).toBe('wgsl');
    });

    it('iPhone → Seek engine, CPU-only fallback (never ORT WebGPU)', () => {
        setDevice('iphone');
        expect(resolveDevice()).toBe('wgsl-wasm');
    });

    it('Android phone → Seek engine, CPU-only fallback (never ORT WebGPU)', () => {
        setDevice('android-phone');
        expect(resolveDevice()).toBe('wgsl-wasm');
    });

    it('Android tablet → Seek engine, CPU-only fallback (not the iPad ORT ladder)', () => {
        setDevice('android-tablet');
        expect(resolveDevice()).toBe('wgsl-wasm');
    });
});

describe('manual override precedence', () => {
    it('force wasm overrides an allowlisted iPad', () => {
        setDevice('ipad');
        setBackendOverride('wasm');
        expect(resolveDevice()).toBe('wasm');
    });

    it('force webgpu → Seek engine with the full ORT fallback, even on iPhone', () => {
        setDevice('iphone');
        setBackendOverride('webgpu');
        expect(resolveDevice()).toBe('wgsl');
    });

    it('force webgpu turns the GPU on for Android too', () => {
        setDevice('android-phone');
        setBackendOverride('webgpu');
        expect(resolveDevice()).toBe('wgsl');
    });

    it('retired opt-ins (webgpu-query, wgsl) migrate to webgpu and persist', () => {
        for (const old of ['webgpu-query', 'wgsl']) {
            localStorage.setItem('seek-backend-override', old);
            expect(getBackendOverride()).toBe('webgpu');
            expect(localStorage.getItem('seek-backend-override')).toBe('webgpu');
        }
    });

    it('explicit auto falls back to the allowlist', () => {
        setDevice('android-tablet');
        setBackendOverride('auto');
        expect(resolveDevice()).toBe('wgsl-wasm');
    });

    it('override round-trips through localStorage', () => {
        setBackendOverride('wasm');
        expect(getBackendOverride()).toBe('wasm');
    });

    it('absent override reads as auto', () => {
        expect(getBackendOverride()).toBe('auto');
    });
});

describe('demote tripwire', () => {
    it('a demoted iPhone resolves to wasm on the auto path', () => {
        setDevice('iphone');
        recordActiveBackend('webgpu');   // Seek engine loads record 'webgpu'
        expect(maybeDemoteOnCrash('crash-while-indexing-foreground')).toBe(true);
        expect(resolveDevice()).toBe('wasm');
        setBackendOverride('webgpu');    // explicit opt-in beats (and clears) it
        expect(resolveDevice()).toBe('wgsl');
    });

    it('a demoted iPad resolves to wasm on the auto path', () => {
        setDevice('ipad');
        recordActiveBackend('webgpu');
        expect(maybeDemoteOnCrash('crash-while-indexing-foreground')).toBe(true);
        expect(isWebgpuDemoted()).toBe(true);
        expect(resolveDevice()).toBe('wasm');
    });

    it('a demoted Android phone resolves to wasm on the auto path', () => {
        setDevice('android-phone');
        recordActiveBackend('webgpu');
        expect(maybeDemoteOnCrash('crash-while-indexing-foreground')).toBe(true);
        expect(resolveDevice()).toBe('wasm');
    });

    it('does NOT demote desktop (auto path ignores the mobile-only flag)', () => {
        // Trip the flag while mobile, then resolve as desktop: demote is gated
        // on Platform.isMobile, so a desktop never self-disables.
        setDevice('ipad');
        recordActiveBackend('webgpu');
        maybeDemoteOnCrash('crash-while-indexing-foreground');
        setDevice('desktop');
        expect(resolveDevice()).toBe('wgsl');
    });

    it('forcing WebGPU clears a sticky demote', () => {
        setDevice('ipad');
        recordActiveBackend('webgpu');
        maybeDemoteOnCrash('crash-while-indexing-foreground');
        expect(isWebgpuDemoted()).toBe(true);
        setBackendOverride('webgpu');
        expect(isWebgpuDemoted()).toBe(false);
    });

    it('clearWebgpuDemoted re-enables the auto WebGPU path', () => {
        setDevice('ipad');
        recordActiveBackend('webgpu');
        maybeDemoteOnCrash('crash-while-indexing-foreground');
        clearWebgpuDemoted();
        expect(resolveDevice()).toBe('wgsl');
    });
});

describe('demote gating — never blame WASM or a non-indexing crash', () => {
    it('a WASM-reindex foreground kill does NOT demote WebGPU', () => {
        setDevice('iphone');
        recordActiveBackend('wasm');
        expect(maybeDemoteOnCrash('crash-while-indexing-foreground')).toBe(false);
        expect(isWebgpuDemoted()).toBe(false);
    });

    it('a hidden (background) indexing kill does NOT demote', () => {
        setDevice('ipad');
        recordActiveBackend('webgpu');
        expect(maybeDemoteOnCrash('crash-while-indexing-hidden')).toBe(false);
        expect(isWebgpuDemoted()).toBe(false);
    });

    it('a non-indexing crash verdict does NOT demote', () => {
        setDevice('ipad');
        recordActiveBackend('webgpu');
        expect(maybeDemoteOnCrash('evicted-while-hidden')).toBe(false);
        expect(isWebgpuDemoted()).toBe(false);
    });

    it('desktop is never demoted even with WebGPU active', () => {
        setDevice('desktop');
        recordActiveBackend('webgpu');
        expect(maybeDemoteOnCrash('crash-while-indexing-foreground')).toBe(false);
    });
});

// residentInt8Enabled keys off isMobilePlatform(), which now reads
// Platform.isMobile — so these pose as a device with setDevice(), exactly like
// the resolveDevice tests above.
describe('residentInt8Enabled — B2 resident-block memory gate', () => {
    it('mobile → always disabled, regardless of size', () => {
        setDevice('iphone');
        expect(residentInt8Enabled(10, 384)).toBe(false);
        expect(residentInt8Enabled(1, 8)).toBe(false);
    });

    it('desktop well under the byte budget → enabled', () => {
        setDevice('desktop');
        expect(residentInt8Enabled(1000, 384)).toBe(true); // 1000*392 ≈ 392 KB
    });

    it('desktop over the byte budget → disabled', () => {
        setDevice('desktop');
        const overBudget = Math.ceil(RESIDENT_INT8_MAX_BYTES / (384 + 8)) + 1;
        expect(residentInt8Enabled(overBudget, 384)).toBe(false);
    });

    it('budget tracks embDim: same row count flips with a larger model dim', () => {
        setDevice('desktop');
        const rows = 40000;
        expect(residentInt8Enabled(rows, 384)).toBe(true);  // 40000*392 = 15.68 MB ≤ 16 MB
        expect(residentInt8Enabled(rows, 512)).toBe(false); // 40000*520 = 20.8 MB > 16 MB
    });

    it('budget boundary is inclusive (≤, not <)', () => {
        setDevice('desktop');
        const embDim = 8;
        const rows = RESIDENT_INT8_MAX_BYTES / (embDim + 8); // 16 MB / 16 = 1048576, exact
        expect(Number.isInteger(rows)).toBe(true);
        expect(residentInt8Enabled(rows, embDim)).toBe(true);
        expect(residentInt8Enabled(rows + 1, embDim)).toBe(false);
    });
});

describe('isUnexpectedGpuLoss', () => {
    it('a real device loss fails in-flight RPCs', () => {
        expect(isUnexpectedGpuLoss({ kind: 'webgpu-device-lost', reason: 'unknown' })).toBe(true);
    });

    it('our own destroy() (failed Seek-engine load on the way to WASM) does not', () => {
        expect(isUnexpectedGpuLoss({ kind: 'webgpu-device-lost', reason: 'destroyed' })).toBe(false);
    });

    it('other GPU events never do', () => {
        expect(isUnexpectedGpuLoss({ kind: 'webgpu-device-created' })).toBe(false);
        expect(isUnexpectedGpuLoss({ kind: 'webgpu-uncaptured-error' })).toBe(false);
    });
});
