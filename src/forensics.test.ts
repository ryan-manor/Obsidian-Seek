// Forensics: localStorage breadcrumb ring + boot-time crash detection.
// The classification matrix is the load-bearing logic — it's what turns a
// silent iOS process death into a memory-ceiling vs background-kill verdict.

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { Forensics, classifyCrash } from './forensics';
import type { ForensicBreadcrumb } from './types';

// Minimal in-memory localStorage. node has no DOM; forensics only needs
// getItem/setItem semantics.
function installLocalStorage(): Map<string, string> {
    const map = new Map<string, string>();
    vi.stubGlobal('localStorage', {
        getItem: (k: string) => map.get(k) ?? null,
        setItem: (k: string, v: string) => { map.set(k, v); },
        removeItem: (k: string) => { map.delete(k); },
    });
    return map;
}

function beat(type: string, vis: 'visible' | 'hidden' = 'visible', detail?: ForensicBreadcrumb['detail']): ForensicBreadcrumb {
    return { t: new Date().toISOString(), type, vis, ...(detail ? { detail } : {}) };
}

describe('classifyCrash', () => {
    it('foreground death mid-indexing → memory-ceiling signature', () => {
        expect(classifyCrash([
            beat('session-start'),
            beat('index-start'),
            beat('index-flush', 'visible', { bucket: 128, n: 8 }),
        ])).toBe('crash-while-indexing-foreground');
    });

    it('hidden death mid-indexing → background-GPU-kill signature', () => {
        expect(classifyCrash([
            beat('session-start'),
            beat('index-flush'),
            beat('visibility-hidden', 'hidden'),
        ])).toBe('crash-while-indexing-hidden');
    });

    it('visibility beats do not mask indexing activity', () => {
        // hidden → visible → flush → hidden again: still indexing when killed.
        expect(classifyCrash([
            beat('index-flush'),
            beat('visibility-hidden', 'hidden'),
            beat('visibility-visible'),
            beat('index-flush'),
            beat('visibility-hidden', 'hidden'),
        ])).toBe('crash-while-indexing-hidden');
    });

    it('index-complete closes the indexing window', () => {
        expect(classifyCrash([
            beat('index-flush'),
            beat('index-complete'),
            beat('visibility-hidden', 'hidden'),
        ])).toBe('evicted-while-hidden');
    });

    it('hidden + idle → ordinary suspended-app eviction', () => {
        expect(classifyCrash([
            beat('session-start'),
            beat('model-load-done'),
            beat('visibility-hidden', 'hidden'),
        ])).toBe('evicted-while-hidden');
    });

    it('foreground death during model load → crash-foreground, attributed by last beat', () => {
        expect(classifyCrash([
            beat('session-start'),
            beat('model-load-start', 'visible', { device: 'auto' }),
        ])).toBe('crash-foreground');
    });

    it('only session-start → unknown', () => {
        expect(classifyCrash([beat('session-start')])).toBe('unknown');
        expect(classifyCrash([])).toBe('unknown');
    });

    it('webgpu lifecycle beats do not mask indexing activity', () => {
        // A device-lost mid-reindex is an observation, not activity — the
        // verdict must stay "indexing" so the kill classification is stable;
        // the device-lost discriminates via the breadcrumb tail instead.
        expect(classifyCrash([
            beat('index-start'),
            beat('index-flush', 'visible', { dispatches: 29 }),
            beat('webgpu-device-lost', 'visible', { reason: 'unknown', deviceSeq: 1 }),
        ])).toBe('crash-while-indexing-foreground');
    });

    it('webgpu beats alone (no indexing) → crash-foreground', () => {
        expect(classifyCrash([
            beat('session-start'),
            beat('model-load-start'),
            beat('webgpu-device-created', 'visible', { deviceSeq: 1 }),
        ])).toBe('crash-foreground');
    });
});

describe('Forensics', () => {
    beforeEach(() => { installLocalStorage(); });

    it('clean shutdown → next boot reports no crash', () => {
        const a = new Forensics('vault-1', 'dev-1', 'sess-1');
        expect(a.bootInspect()).toBeNull();          // no prior record
        a.beat('index-start', { filesTotal: 10 });
        a.markCleanEnd();

        const b = new Forensics('vault-1', 'dev-1', 'sess-2');
        expect(b.bootInspect()).toBeNull();          // prior ended cleanly
    });

    it('unclean end → crash-detected with dead session identity and tail', () => {
        const a = new Forensics('vault-1', 'dev-1', 'sess-1');
        a.bootInspect();
        a.beat('index-start', { mode: 'full', filesTotal: 1661 });
        a.beat('index-flush', { bucket: 128, n: 8, dispatches: 3, paddedTokens: 3072 });
        // no markCleanEnd — process "died"

        const b = new Forensics('vault-1', 'dev-1', 'sess-2');
        const crash = b.bootInspect();
        expect(crash).not.toBeNull();
        expect(crash!.deadSessionId).toBe('sess-1');
        expect(crash!.verdict).toBe('crash-while-indexing-foreground');
        expect(crash!.lastBeat!.type).toBe('index-flush');
        expect(crash!.lastBeat!.detail).toMatchObject({ dispatches: 3, paddedTokens: 3072 });
        expect(crash!.gapSeconds).not.toBeNull();
        // The dead session's tail includes its session-start and beats.
        expect(crash!.breadcrumbs.map(x => x.type)).toEqual(['session-start', 'index-start', 'index-flush']);
    });

    it('crash record is consumed: third boot is clean', () => {
        const a = new Forensics('vault-1', 'dev-1', 'sess-1');
        a.bootInspect();
        a.beat('index-flush');

        const b = new Forensics('vault-1', 'dev-1', 'sess-2');
        expect(b.bootInspect()).not.toBeNull();
        b.markCleanEnd();

        const c = new Forensics('vault-1', 'dev-1', 'sess-3');
        expect(c.bootInspect()).toBeNull();
    });

    it('vault scoping: records in different scopes do not collide', () => {
        const a = new Forensics('vault-A', 'dev-1', 'sess-A1');
        a.bootInspect();
        a.beat('index-flush');                        // vault A "dies"

        const b = new Forensics('vault-B', 'dev-1', 'sess-B1');
        expect(b.bootInspect()).toBeNull();           // vault B unaffected

        const a2 = new Forensics('vault-A', 'dev-1', 'sess-A2');
        expect(a2.bootInspect()!.deadSessionId).toBe('sess-A1');
    });

    it('ring caps at 40 — oldest beats are dropped, latest survive', () => {
        const a = new Forensics('vault-1', 'dev-1', 'sess-1');
        a.bootInspect();
        for (let i = 0; i < 60; i++) a.beat('index-flush', { dispatches: i });

        const b = new Forensics('vault-1', 'dev-1', 'sess-2');
        const crash = b.bootInspect()!;
        expect(crash.lastBeat!.detail).toMatchObject({ dispatches: 59 });
        expect(crash.breadcrumbs.length).toBe(10);    // capped tail
    });

    it('corrupt prior record never throws, is cleared, and forensics recovers', () => {
        localStorage.setItem('seek-forensics:vault-1', '{not json');
        const a = new Forensics('vault-1', 'dev-1', 'sess-1');
        expect(() => a.bootInspect()).not.toThrow();
        a.beat('index-flush');                        // sess-1 then "dies"

        // Recovery: the corrupt blob was replaced by sess-1's valid record,
        // so the next boot detects sess-1's death normally.
        const b = new Forensics('vault-1', 'dev-1', 'sess-2');
        const crash = b.bootInspect();
        expect(crash!.deadSessionId).toBe('sess-1');
    });
});
