// Unit tests for the mobile catch-up drain loop. The loop is the jetsam-safety
// core of the iPhone crash-loop fix: it must drain to empty across self-chaining
// budgeted bursts, preserve durable per-burst progress, abort cleanly when the app
// backgrounds or the user resumes searching, and NEVER spin (model drift / files
// that won't commit). All model-free via injected fakes.

import { describe, it, expect } from 'vitest';
import { drainCatchUp, type CatchUpDrainDeps } from './catchup';

// Build deps over a mutable dirty-set "store". reindexDelta simulates the real
// engine: per burst it commits up to maxFiles files of the list it was PASSED and
// reports them as committedPaths (the real engine returns the carried + hydrated +
// embedded-prefix set). It also drops them from `store.dirty` to model the watermark
// advance, so the once-per-sweep computeDelta reflects real progress. The drain
// shrinks its OWN `remaining` set by committedPaths — it no longer re-diffs per
// burst — so the harness no longer drives convergence through computeDelta.
interface HarnessOpts {
    initialDirty: number;
    maxFiles?: number;            // undefined = desktop-unbounded
    committedPerBurst?: (cap: number, remaining: number) => number;  // override commit count
    commitTail?: boolean;         // report the LAST n of the burst as committed (proves the drain
                                  // trusts committedPaths, not a prefix/position assumption)
    embeddedNull?: boolean;       // simulate model-drift / cold (embed phase skipped)
    deferredEmbed?: number;       // paired with embeddedNull for the drift signal
    hidden?: () => boolean;
    searchActive?: () => boolean;
    onBurst?: (burstIndex: number) => void;  // hook fired after each reindexDelta
}

function harness(o: HarnessOpts) {
    const store = { dirty: Array.from({ length: o.initialDirty }, (_, i) => `f${i}.md`) };
    let bursts = 0;
    let deltas = 0;
    const deps: CatchUpDrainDeps = {
        computeDelta: async () => { deltas++; return { dirty: [...store.dirty], deleted: [] }; },
        reindexDelta: async (burst, _deleted, opts) => {
            const cap = opts.maxFiles ?? burst.length;     // unbounded → all
            const n = o.committedPerBurst
                ? o.committedPerBurst(cap, burst.length)
                : Math.min(cap, burst.length);
            const committedPaths = o.commitTail ? burst.slice(burst.length - n) : burst.slice(0, n);
            const done = new Set(committedPaths);
            store.dirty = store.dirty.filter(p => !done.has(p));  // watermark advance
            bursts++;
            o.onBurst?.(bursts);
            return {
                embedded: o.embeddedNull ? null : {},
                deferredEmbed: o.deferredEmbed ?? 0,
                committedPaths,
            };
        },
        isHidden: o.hidden ?? (() => false),
        isSearchActive: o.searchActive ?? (() => false),
        pace: async () => { /* immediate yield in tests */ },
        maxFiles: o.maxFiles,
    };
    return { deps, store, getBursts: () => bursts, getDeltas: () => deltas };
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
            reindexDelta: async (burst, _del, opts) => {
                active = true;  // user resumes searching once the burst is underway
                observed = opts.shouldContinue ? opts.shouldContinue() : null;
                const committedPaths = burst.slice(0, 1);  // commit the one in-flight file
                store.dirty = store.dirty.filter(p => p !== committedPaths[0]);
                return { embedded: {}, deferredEmbed: 0, committedPaths };
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

    it('over-report backstop: bails bounded instead of spinning when committedPaths over-reports', async () => {
        // The 2026-06-29 review failure mode: the engine reports a path as committed (a
        // started-but-uncommitted file — empty/below-min-chunk note, or a skip-error —
        // that a count-of-started prefix wrongly includes) but writes NO record, so the
        // file stays dirty and computeDelta re-finds it every sweep. The inner no-progress
        // guard can't catch this (committedPaths is NON-empty), so without the cross-sweep
        // backstop the outer loop re-sweeps the whole vault forever. Model it directly:
        // reindexDelta lies (reports the burst committed) but the store NEVER shrinks.
        let deltas = 0, bursts = 0;
        const deps: CatchUpDrainDeps = {
            computeDelta: async () => { deltas++; return { dirty: ['empty.md'], deleted: [] }; },
            reindexDelta: async (burst) => {
                bursts++;
                return { embedded: {}, deferredEmbed: 0, committedPaths: [...burst] }; // claims commit; store unchanged
            },
            isHidden: () => false,
            isSearchActive: () => false,
            pace: async () => { /* immediate */ },
            maxFiles: 3,
        };
        const { pending } = await drainCatchUp(deps);
        expect(pending).toBe(true);               // bailed via the cross-sweep backstop
        expect(deltas).toBe(2);                    // sweep 1 + one re-diff that matched → STOP (not ∞)
        expect(bursts).toBe(1);                    // did not keep re-embedding the phantom file
    });

    // ---- the O(N) property: the whole point of the batch-drain restructure ----

    it('computes the delta ONCE per sweep, not once per burst (O(N), not O(N²))', async () => {
        // 100 files at maxFiles=3 is ~34 bursts. The OLD drain called the O(vault)
        // computeDelta every burst (~34 whole-vault diffs). The batched drain diffs
        // once to get the work-list and once more to confirm it drained: 2 total.
        const h = harness({ initialDirty: 100, maxFiles: 3 });
        const { pending } = await drainCatchUp(h.deps);
        expect(pending).toBe(false);
        expect(h.store.dirty).toHaveLength(0);
        expect(h.getBursts()).toBe(34);           // ceil(100 / 3) bursts still happen
        expect(h.getDeltas()).toBe(2);             // but only 2 computeDelta calls, NOT ~34
    });

    it('advances by the REPORTED committed set, not by position (no prefix assumption)', async () => {
        // The real engine commits a scattered set (carry-over + sidecar-hydrate +
        // embed-prefix), not necessarily the first N. Model that by reporting the
        // TAIL of each burst as committed; the drain must still drain every file
        // exactly once with no skips.
        const h = harness({ initialDirty: 10, maxFiles: 3, commitTail: true });
        const { pending } = await drainCatchUp(h.deps);
        expect(pending).toBe(false);
        expect(h.store.dirty).toHaveLength(0);    // every file committed, none skipped
        expect(h.getBursts()).toBe(4);            // 3+3+3+1
    });

    it('cursor = real progress: a burst that commits fewer than maxFiles retries the rest', async () => {
        // Budget defer: the engine drops stale chunks for all 3 of a slice but only
        // re-embeds 1 before the wall-clock ceiling. The deferred 2 must reappear in
        // a later burst, never be skipped past. Here every burst commits just 1.
        const h = harness({
            initialDirty: 5,
            maxFiles: 3,
            committedPerBurst: () => 1,           // only one of the (≤3) slice commits
        });
        const { pending } = await drainCatchUp(h.deps);
        expect(pending).toBe(false);
        expect(h.store.dirty).toHaveLength(0);    // all 5 drained despite per-burst defer
        expect(h.getBursts()).toBe(5);            // one file at a time, none lost
    });

    it('mid-drain churn is picked up by the next sweep, not bailed (replaces the old progress bail)', async () => {
        // iCloud restamps unrelated files mid-drain. The OLD drain compared dirty
        // counts across bursts and BAILED when a later computeDelta returned more.
        // The batched drain diffs once per sweep, so churn that arrives after a sweep
        // started is simply drained by the FOLLOWING sweep.
        let injected = false;
        const store = { dirty: ['a.md', 'b.md', 'c.md'] };
        let deltas = 0;
        const deps: CatchUpDrainDeps = {
            computeDelta: async () => {
                deltas++;
                // After the first sweep drains a/b/c, the end-of-sweep re-diff fires:
                // inject two freshly-churned files exactly once, as if edited mid-drain.
                if (!injected && store.dirty.length === 0) { injected = true; store.dirty = ['d.md', 'e.md']; }
                return { dirty: [...store.dirty], deleted: [] };
            },
            reindexDelta: async (burst, _del, opts) => {
                const committedPaths = burst.slice(0, opts.maxFiles ?? burst.length);
                const done = new Set(committedPaths);
                store.dirty = store.dirty.filter(p => !done.has(p));
                return { embedded: {}, deferredEmbed: 0, committedPaths };
            },
            isHidden: () => false,
            isSearchActive: () => false,
            pace: async () => { /* immediate */ },
            maxFiles: 3,
        };
        const { pending } = await drainCatchUp(deps);
        expect(pending).toBe(false);              // both the original and the churned files drained
        expect(store.dirty).toHaveLength(0);
        expect(deltas).toBe(3);                   // sweep1 list + churn re-diff(drains d,e) + final empty
    });

    it('applies a delete-only sweep (dirty empty, deletes pending) without bailing', async () => {
        // A pure-delete delta has no embed work, so the burst commits nothing — but
        // that is NOT a stall: the deletes still apply. The drain must run one
        // delete-only burst and then drain, not treat the empty committed set as a
        // no-progress bail.
        let applied = 0;
        const store = { deleted: ['gone.md', 'old.md'] };
        const deps: CatchUpDrainDeps = {
            computeDelta: async () => ({ dirty: [], deleted: [...store.deleted] }),
            reindexDelta: async (_burst, deleted, _opts) => {
                applied += deleted.length;
                store.deleted = [];               // deletes land → next diff is empty
                return { embedded: null, deferredEmbed: 0, committedPaths: [] };
            },
            isHidden: () => false,
            isSearchActive: () => false,
            pace: async () => { /* immediate */ },
            maxFiles: 3,
        };
        const { pending } = await drainCatchUp(deps);
        expect(pending).toBe(false);
        expect(applied).toBe(2);                  // both deletes applied
    });
});
