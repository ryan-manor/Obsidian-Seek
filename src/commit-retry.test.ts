// Commit-failure retry backoff + post-reindex re-classification (Tier-2).
// Background: 2026-11-11 iPhone full reindex — two commits aborted while the app
// was briefly backgrounded (iOS kills in-flight IDB transactions), and a 'modify'
// storm during the pass queued 4,161 notes against the empty store that were then
// rewritten for nothing. See [[Seek Optimization - Custom shaders and memory]].
import { describe, it, expect, vi, afterEach } from 'vitest';
import { Scenario } from './test-harness/scenario';
import { SearchOrchestrator } from './search';

vi.mock('obsidian', async importOriginal => {
    const orig = await importOriginal<typeof import('obsidian')>();
    return { ...orig, Notice: vi.fn() };
});

const [STEP1, STEP2, STEP3] = SearchOrchestrator.COMMIT_RETRY_BACKOFF_MS;
const abort = () => { throw new DOMException('transaction aborted', 'AbortError'); };

describe('commit-failure retry backoff (Tier-2)', () => {
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
        vi.useRealTimers();
    });

    it('a transient abort backs off ~1 min, then catch-up retries and heals', async () => {
        vi.useFakeTimers({ toFake: ['Date'] });
        vi.setSystemTime(1_000_000);
        const s = await boot();
        s.vault.write('a.md', 'alpha note about the pixel phone camera', 1000);
        s.vault.write('b.md', 'beta note about concert setlists', 1000);

        const realPut = s.store.putBatchQuantized.bind(s.store);
        s.store.putBatchQuantized = async () => abort();
        const entry = await s.orch.reindexAll();
        expect(entry.committedFilePaths).toEqual([]);
        s.store.putBatchQuantized = realPut;

        // Inside the window: held out of the diff, and a retry is scheduled.
        expect((await s.orch.computeDelta()).dirty).toEqual([]);
        expect(s.orch.nextCommitRetryAt()).toBe(1_000_000 + STEP1);

        // Window elapses: dirty again, one reconcile heals, nothing left pending.
        vi.setSystemTime(1_000_000 + STEP1 + 1);
        expect((await s.orch.computeDelta()).dirty.slice().sort()).toEqual(['a.md', 'b.md']);
        await s.reconcile();
        expect((await s.orch.computeDelta()).dirty).toEqual([]);
        expect(s.orch.nextCommitRetryAt()).toBeNull();
    });

    it('a deterministic failure walks the backoff, then parks until the note changes', async () => {
        vi.useFakeTimers({ toFake: ['Date'] });
        let now = 1_000_000;
        vi.setSystemTime(now);
        const s = await boot();
        s.vault.write('bad.md', 'a note whose commit always fails', 1000);
        const realPut = s.store.putBatchQuantized.bind(s.store);
        s.store.putBatchQuantized = async () => abort();

        await s.orch.reindexAll();                       // attempt 1
        for (const step of [STEP1, STEP2, STEP3]) {      // attempts 2..4
            now += step + 1;
            vi.setSystemTime(now);
            expect((await s.orch.computeDelta()).dirty).toEqual(['bad.md']);
            await s.reconcile();
        }
        // Backoff exhausted: parked, no retry armed, invisible even far later.
        expect(s.orch.nextCommitRetryAt()).toBeNull();
        vi.setSystemTime(now + 24 * 60 * 60_000);
        expect((await s.orch.computeDelta()).dirty).toEqual([]);

        // An edit lifts the park immediately.
        s.store.putBatchQuantized = realPut;
        s.vault.write('bad.md', 'a note whose commit always fails, now edited', 2000);
        expect((await s.orch.computeDelta()).dirty).toEqual(['bad.md']);
        await s.reconcile();
        expect((await s.orch.computeDelta()).dirty).toEqual([]);
    });

    it('a full reindex clears parked paths and retries them', async () => {
        const s = await boot();
        s.vault.write('bad.md', 'a parked note', 1000);
        const realPut = s.store.putBatchQuantized.bind(s.store);
        s.store.putBatchQuantized = async () => abort();
        await s.orch.reindexAll();
        s.store.putBatchQuantized = realPut;
        const entry = await s.orch.reindexAll();
        expect(entry.committedFilePaths).toEqual(['bad.md']);
        expect(s.orch.nextCommitRetryAt()).toBeNull();
    });

    it('quota failures are not backed off (the disk, not the note)', async () => {
        const s = await boot();
        s.vault.write('a.md', 'alpha', 1000);
        const realPut = s.store.putBatchQuantized.bind(s.store);
        s.store.putBatchQuantized = async () => { throw new DOMException('exceeded', 'QuotaExceededError'); };
        await s.orch.reindexAll();
        s.store.putBatchQuantized = realPut;
        expect(s.orch.nextCommitRetryAt()).toBeNull();
        expect((await s.orch.computeDelta()).dirty).toEqual(['a.md']);
    });
});

describe('reindexDelta re-classifies queued paths under the mutex (Tier-2)', () => {
    let active: Scenario | null = null;
    afterEach(async () => { await active?.teardown(); active = null; });

    it('drops unchanged notes queued against a stale view; keeps real edits', async () => {
        const s = new Scenario();
        await s.boot();
        active = s;
        s.vault.write('same.md', 'an unchanged note', 1000);
        s.vault.write('edited.md', 'a note that changes', 1000);
        await s.orch.reindexAll();

        s.vault.write('edited.md', 'a note that changed after its commit', 2000);
        const r = await s.orch.reindexDelta(['same.md', 'edited.md'], [], { embed: true });
        expect(r.committedPaths).toEqual(['edited.md']);
        expect((await s.orch.computeDelta()).dirty).toEqual([]);
    });
});
