// Unit tests for the mobile catch-up drain loop. The loop is the jetsam-safety
// core of the iPhone crash-loop fix: it must drain to empty across self-chaining
// budgeted bursts, preserve durable per-burst progress, abort cleanly when the app
// backgrounds or the user resumes searching, and NEVER spin (model drift / files
// that won't commit). All model-free via injected fakes.

import { describe, it, expect } from 'vitest';
import { drainCatchUp, type CatchUpDrainDeps } from './catchup';

// Build deps over a mutable dirty-set "store". reindexDelta simulates the real
// engine: it commits (removes) up to maxFiles files per call — modelling per-file
// commit + watermark advance, which is what makes the next computeDelta shrink.
interface HarnessOpts {
    initialDirty: number;
    maxFiles?: number;            // undefined = desktop-unbounded
    committedPerBurst?: (cap: number, remaining: number) => number;  // override commit behavior
    embeddedNull?: boolean;       // simulate model-drift / cold (embed phase skipped)
    deferredEmbed?: number;       // paired with embeddedNull for the drift signal
    hidden?: () => boolean;
    searchActive?: () => boolean;
    onBurst?: (burstIndex: number) => void;  // hook fired after each reindexDelta
}

function harness(o: HarnessOpts) {
    const store = { dirty: Array.from({ length: o.initialDirty }, (_, i) => `f${i}.md`) };
    let bursts = 0;
    const deps: CatchUpDrainDeps = {
        computeDelta: async () => ({ dirty: [...store.dirty], deleted: [] }),
        reindexDelta: async (_dirty, _deleted, opts) => {
            const cap = opts.maxFiles ?? store.dirty.length;   // unbounded → all
            const committed = o.committedPerBurst
                ? o.committedPerBurst(cap, store.dirty.length)
                : Math.min(cap, store.dirty.length);
            store.dirty = store.dirty.slice(committed);        // commit the first `committed`
            bursts++;
            o.onBurst?.(bursts);
            return {
                embedded: o.embeddedNull ? null : {},
                deferredEmbed: o.deferredEmbed ?? 0,
            };
        },
        isHidden: o.hidden ?? (() => false),
        isSearchActive: o.searchActive ?? (() => false),
        pace: async () => { /* immediate yield in tests */ },
        maxFiles: o.maxFiles,
    };
    return { deps, store, getBursts: () => bursts };
}

describe('drainCatchUp', () => {
    it('drains a multi-file delta to empty across budgeted bursts (convergence)', async () => {
        const h = harness({ initialDirty: 10, maxFiles: 3 });
        const { pending } = await drainCatchUp(h.deps);
        expect(pending).toBe(false);              // fully drained
        expect(h.store.dirty).toHaveLength(0);
        expect(h.getBursts()).toBe(4);            // 3+3+3+1
    });

    it('desktop (unbounded) drains in a single burst', async () => {
        const h = harness({ initialDirty: 10, maxFiles: undefined });
        const { pending } = await drainCatchUp(h.deps);
        expect(pending).toBe(false);
        expect(h.store.dirty).toHaveLength(0);
        expect(h.getBursts()).toBe(1);
    });

    it('is a no-op on an empty delta', async () => {
        const h = harness({ initialDirty: 0, maxFiles: 3 });
        const { pending } = await drainCatchUp(h.deps);
        expect(pending).toBe(false);
        expect(h.getBursts()).toBe(0);            // never touches reindexDelta
    });

    it('soft-pauses (pending) when the app backgrounds mid-drain, preserving progress', async () => {
        let hidden = false;
        const h = harness({
            initialDirty: 10,
            maxFiles: 3,
            hidden: () => hidden,
            onBurst: () => { hidden = true; },    // background right after the first commit
        });
        const { pending } = await drainCatchUp(h.deps);
        expect(pending).toBe(true);               // re-fire me later
        expect(h.getBursts()).toBe(1);            // stopped — did NOT keep embedding hidden
        expect(h.store.dirty).toHaveLength(7);    // the 3 committed are durable
    });

    it('soft-pauses when the user resumes searching mid-drain', async () => {
        let active = false;
        const h = harness({
            initialDirty: 10,
            maxFiles: 3,
            searchActive: () => active,
            onBurst: () => { active = true; },
        });
        const { pending } = await drainCatchUp(h.deps);
        expect(pending).toBe(true);
        expect(h.getBursts()).toBe(1);
        expect(h.store.dirty).toHaveLength(7);
    });

    it('does NOT start if already hidden', async () => {
        const h = harness({ initialDirty: 10, maxFiles: 3, hidden: () => true });
        const { pending } = await drainCatchUp(h.deps);
        expect(pending).toBe(true);
        expect(h.getBursts()).toBe(0);
    });

    it('drift-no-spin: stops after one burst when the embed phase cannot run', async () => {
        // Model drift: embed phase skipped (embedded=null, deferredEmbed>0), dirty
        // set never shrinks. A naive loop would re-find the same set forever.
        const h = harness({
            initialDirty: 5,
            maxFiles: 3,
            committedPerBurst: () => 0,           // nothing commits
            embeddedNull: true,
            deferredEmbed: 5,
        });
        const { pending } = await drainCatchUp(h.deps);
        expect(pending).toBe(true);
        expect(h.getBursts()).toBe(1);            // bailed immediately, no spin
    });

    it('per-file abort (shouldContinue) reflects a resumed search, not just hide', async () => {
        // The loop only enters reindexDelta when searchActive is false at the top;
        // but if the user resumes searching MID-burst, the shouldContinue passed to
        // the engine must report false so the burst aborts after its in-flight file
        // rather than competing with the live query for the whole budget.
        let active = false;
        let observed: boolean | null = null;
        const store = { dirty: ['a.md', 'b.md', 'c.md', 'd.md'] };
        const deps: CatchUpDrainDeps = {
            computeDelta: async () => ({ dirty: [...store.dirty], deleted: [] }),
            reindexDelta: async (_d, _del, opts) => {
                active = true;  // user resumes searching once the burst is underway
                observed = opts.shouldContinue ? opts.shouldContinue() : null;
                store.dirty = store.dirty.slice(1);  // commit the one in-flight file
                return { embedded: {}, deferredEmbed: 0 };
            },
            isHidden: () => false,
            isSearchActive: () => active,
            pace: async () => { /* immediate */ },
            maxFiles: 3,
        };
        const { pending } = await drainCatchUp(deps);
        expect(observed).toBe(false);   // engine abort hook saw the resumed search
        expect(pending).toBe(true);     // loop soft-paused at the next burst boundary
    });

    it('no-progress guard: stops when a burst commits nothing (persistent skip-error)', async () => {
        // embed phase RAN (embedded non-null) but committed 0 files (e.g. every file
        // skip-errors). The dirty set doesn't shrink → stop instead of spinning.
        const h = harness({
            initialDirty: 5,
            maxFiles: 3,
            committedPerBurst: () => 0,           // engine ran but committed nothing
        });
        const { pending } = await drainCatchUp(h.deps);
        expect(pending).toBe(true);
        expect(h.getBursts()).toBe(1);            // one attempt, then the progress guard halts it
    });
});
