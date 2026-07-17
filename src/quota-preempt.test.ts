// S2 (quota) + 3A (full-reindex soft preempt) + S4 (fallback tripwires) — Tier-1
// units for the QuotaExceededError classifier and the deltaFallback observability
// contract, plus Tier-2 composed scenarios: the real commit path under a full
// disk (split counter, check line, toast rate-limit, catch-up heal once space
// frees) and the shouldContinue mode contract (full PAUSES and resumes; only the
// incremental path aborts). See test-harness/scenario.ts and [[Seek Testing Strategy]].
import { describe, it, expect, vi, afterEach } from 'vitest';
import { Notice } from 'obsidian';
import { isQuotaError, isStoreClosedError } from './index-store';
import { INDEX_QUOTA_MSG } from './index-notice';
import { Scenario } from './test-harness/scenario';

// Replace ONLY Notice with a spy constructor so scenarios can assert toast
// behavior; everything else (TFile, Platform, …) stays the real test stub.
vi.mock('obsidian', async importOriginal => {
    const orig = await importOriginal<typeof import('obsidian')>();
    return { ...orig, Notice: vi.fn() };
});

describe('isQuotaError (Tier-1)', () => {
    it('matches a DOMException named QuotaExceededError', () => {
        expect(isQuotaError(new DOMException('exceeded', 'QuotaExceededError'))).toBe(true);
    });

    it('matches by name across realms — no instanceof dependence', () => {
        // An error surfaced from another realm (iframe/worker) fails
        // `instanceof DOMException` while being the same condition.
        expect(isQuotaError({ name: 'QuotaExceededError', message: 'realm-crossed' })).toBe(true);
    });

    it('rejects everything else', () => {
        expect(isQuotaError(new Error('QuotaExceededError'))).toBe(false);  // message, not name
        expect(isQuotaError(new DOMException('x', 'AbortError'))).toBe(false);
        expect(isQuotaError(null)).toBe(false);
        expect(isQuotaError(undefined)).toBe(false);
        expect(isQuotaError('QuotaExceededError')).toBe(false);
    });

    it('stays disjoint from the closed-store class (different abort semantics)', () => {
        // Closed store ⇒ rethrow + abort the whole pass; quota ⇒ per-file skip.
        // The two classifiers must never both match one error.
        const q = new DOMException('exceeded', 'QuotaExceededError');
        expect(isStoreClosedError(q)).toBe(false);
    });
});

describe('quota + preempt scenarios (Tier-2)', () => {
    let active: Scenario | null = null;
    const boot = async (): Promise<Scenario> => {
        const s = new Scenario();
        await s.boot();
        active = s;
        return s;
    };
    afterEach(async () => {
        await active?.teardown();
        active = null;
        vi.mocked(Notice).mockClear();
    });

    // ── S2: the full-disk storm ─────────────────────────────────────────────
    // Every commit rejects with the browser's quota error. The pass must count
    // the skips as quota (not generic), fail loudly with an actionable check
    // line, toast ONCE across passes, and leave the files dirty so the ordinary
    // catch-up path heals them the moment space frees — no manual reindex.
    it('a full-disk pass splits out quota skips, toasts once, and catch-up heals after space frees', async () => {
        const s = await boot();
        s.vault.write('a.md', 'alpha note about the pixel phone camera', 1000);
        s.vault.write('b.md', 'beta note about concert setlists', 1000);
        s.vault.write('c.md', 'gamma note about sourdough starters', 1000);

        const realPut = s.store.putBatchQuantized.bind(s.store);
        s.store.putBatchQuantized = async () => { throw new DOMException('exceeded', 'QuotaExceededError'); };

        const entry = await s.orch.reindexAll();
        expect(entry.filesSkippedQuota).toBe(3);
        expect(entry.filesSkippedError).toBe(3);          // quota skips still count toward the pass gate
        expect(entry.committedFilePaths).toEqual([]);
        expect(entry.pass).toBe(false);                   // 100% skip ≫ the 2% gate
        expect(entry.checks.join('\n')).toContain('storage full');

        // Exactly one toast, with the canonical copy — and a second failing pass
        // inside the rate window stays silent (catch-up bursts would otherwise storm).
        expect(Notice).toHaveBeenCalledTimes(1);
        expect(vi.mocked(Notice).mock.calls[0][0]).toBe(INDEX_QUOTA_MSG);
        await s.orch.reindexAll();
        expect(Notice).toHaveBeenCalledTimes(1);

        // Space frees: the failed files never advanced their records, so they are
        // still dirty by the drain's own criterion, and one reconcile heals all.
        s.store.putBatchQuantized = realPut;
        const { dirty } = await s.orch.computeDelta();
        expect(dirty.slice().sort()).toEqual(['a.md', 'b.md', 'c.md']);
        await s.reconcile();
        expect((await s.orch.computeDelta()).dirty).toEqual([]);
        expect((await s.store.count()).chunks).toBeGreaterThan(0);
    });

    // ── 3A: the shouldContinue mode contract, both halves ───────────────────
    // Full mode PAUSES between files while the signal is blocked, then resumes
    // and finishes everything — it must never abort (there is no catch-up that
    // re-fires a manual full reindex).
    it('a full reindex pauses for live search activity and RESUMES — commits everything (3A)', async () => {
        const s = await boot();
        s.vault.write('a.md', 'first note body with several words', 1000);
        s.vault.write('b.md', 'second note body with other words', 1000);

        // Blocked for the first two checks (the loop-top gate + the first in-wait
        // poll), clear from the third on — one real ~250 ms poll tick.
        let calls = 0;
        const shouldContinue = vi.fn(() => ++calls > 2);
        const progress: string[] = [];
        const entry = await s.orch.reindexAll(msg => progress.push(msg), { shouldContinue });

        expect(entry.committedFilePaths.slice().sort()).toEqual(['a.md', 'b.md']);
        expect(entry.filesDeferred).toBe(0);              // nothing dropped — a pause, not an abort
        expect(entry.pass).toBe(true);
        expect(shouldContinue.mock.calls.length).toBeGreaterThanOrEqual(3);  // gate + ≥1 poll + the all-clear
        // The pause explains itself — a frozen counter with no label is the shape
        // that made users force-quit healthy runs (PROGRESS_MAX_SILENCE_MS).
        expect(progress.some(m => m.includes('paused while you search'))).toBe(true);
    });

    it('the incremental path still ABORTS on shouldContinue=false — files stay dirty for catch-up', async () => {
        const s = await boot();
        s.vault.write('a.md', 'seed note so the cold build stamps identity', 1000);
        await s.coldStart();
        s.vault.write('b.md', 'new note arriving mid-session', 2000);

        const { dirty, deleted } = await s.orch.computeDelta();
        expect(dirty).toEqual(['b.md']);
        const r = await s.orch.reindexDelta(dirty, deleted, { embed: true, shouldContinue: () => false });

        expect(r.committedPaths).toEqual([]);             // burst aborted before the first file
        expect((await s.orch.computeDelta()).dirty).toEqual(['b.md']);  // still dirty — the drain re-fires it
    });

    // ── S4: fallback observability ──────────────────────────────────────────
    // The tripwire contract: every declined incremental patch surfaces its reason
    // with a per-reason session count (the "is fallback churn real?" data), and
    // still signals the full rebuild by returning false.
    it('deltaFallback records reason + session count and returns false (S4)', async () => {
        const s = await boot();
        const info = vi.spyOn(console, 'info').mockImplementation(() => {});
        try {
            const fb = (s.orch as unknown as { deltaFallback(r: string, d?: Record<string, unknown>): boolean })
                .deltaFallback.bind(s.orch);
            expect(fb('cold caches')).toBe(false);
            expect(fb('cold caches')).toBe(false);
            const msgs = info.mock.calls.map(c => String(c[0]));
            expect(msgs[0]).toContain('applyDelta fallback: cold caches');
            expect(msgs[1]).toContain('×2 this session');
            // Per-call values ride `detail`, NOT the reason key — otherwise a
            // reason like compaction (whose counts differ every occurrence) would
            // fragment the session counter into ×1 entries and never show churn.
            expect(fb('compaction due', { tombstones: 5, rows: 10 })).toBe(false);
            expect(fb('compaction due', { tombstones: 9, rows: 40 })).toBe(false);
            expect(String(info.mock.calls[3][0])).toContain('compaction due — full cache rebuild (×2 this session)');
        } finally {
            info.mockRestore();
        }
    });
});
