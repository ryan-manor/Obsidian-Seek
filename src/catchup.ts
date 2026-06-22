// Mobile catch-up drain — the jetsam-safe replacement for the old "embed the whole
// delta in the foreground after every search" behavior that crash-looped the iPhone
// (single-threaded WASM embed in the iframe shares the ~1.5 GB iOS WebContent
// process with the UI; embedding a whole delta at once saturates it → jetsam kill →
// the delta re-attempts next session forever). Self-contained + dependency-injected
// so it's a pure, model-free unit (see catchup.test.ts).

// Per-burst budget (mobile only — desktop has no jetsam ceiling and runs unbounded).
// Commit at most MAX_FILES per burst, or BUDGET_MS for one pathological huge note,
// then yield and re-check before continuing. Small enough that a single burst can't
// saturate the process; per-file commits make each burst durable, so a killed burst
// loses at most one in-flight file. The drain self-chains across bursts until the
// delta is empty, so small caps just mean "drain gently over a few yields".
export const CATCHUP_MAX_FILES_PER_BURST = 3;
export const CATCHUP_BURST_BUDGET_MS = 8000;

// Dependencies of the drain, injected so the loop has no Obsidian/model coupling.
export interface CatchUpDrainDeps {
    computeDelta: () => Promise<{ dirty: string[]; deleted: string[] }>;
    reindexDelta: (
        dirty: string[],
        deleted: string[],
        opts: { embed: boolean; maxFiles?: number; budgetMs?: number; shouldContinue?: () => boolean },
    ) => Promise<{ embedded: unknown; deferredEmbed: number }>;
    isHidden: () => boolean;        // app backgrounded — never index hidden (a separate jetsam class)
    isSearchActive: () => boolean;  // user back in a live query — don't compete
    pace: () => Promise<void>;      // inter-burst yield (CompositorPacer)
    maxFiles?: number;              // per-burst file cap (mobile); undefined = unbounded
    budgetMs?: number;              // per-burst wall-clock ceiling (mobile); undefined = unbounded
}

// Drain the incremental delta in self-chaining, safety-bounded bursts until it's
// empty — or until a stop condition (hidden / user searching / model drift / no
// forward progress) defers the rest. Per-file commits make every burst durable, so
// stopping early never loses work; `pending=true` means "re-fire me from a later
// trigger". Pure + terminating: every iteration either shrinks the dirty set or
// returns. Returns whether catch-up work remains.
export async function drainCatchUp(deps: CatchUpDrainDeps): Promise<{ pending: boolean }> {
    let prevDirty = Infinity;
    for (;;) {
        // Abort window: backgrounded, or the user reopened/resumed search.
        if (deps.isHidden() || deps.isSearchActive()) return { pending: true };
        const { dirty, deleted } = await deps.computeDelta();
        if (dirty.length === 0 && deleted.length === 0) return { pending: false };  // drained
        // No forward progress since the last burst — persistently skip-erroring or
        // un-embeddable files, or a delete that won't apply. Stop instead of
        // spinning; a later trigger (or a full reindex) retries.
        if (dirty.length >= prevDirty) return { pending: true };
        prevDirty = dirty.length;
        const r = await deps.reindexDelta(dirty, deleted, {
            embed: true,
            maxFiles: deps.maxFiles,
            budgetMs: deps.budgetMs,
            // Per-file abort: also stop the moment the user resumes searching, not
            // just on background — otherwise an in-flight burst keeps embedding (up
            // to the file/time budget) in competition with the live query, the exact
            // shared-process contention this guards against. Closes the window to at
            // most one in-flight file (same guarantee as the hidden abort).
            shouldContinue: () => !deps.isHidden() && !deps.isSearchActive(),
        });
        // Model drift: the embed phase couldn't run at all and files stay dirty.
        // Looping re-finds the same set forever — only a full reindex fixes it.
        if (r.embedded === null && r.deferredEmbed > 0) return { pending: true };
        await deps.pace();  // yield; on iOS the macrotask lets a queued visibilitychange flip isHidden
    }
}
