// Shared index-mutation coordination, factored out of SearchOrchestrator so the
// indexing and searching halves can be split apart while still sharing exactly the
// state that couples them. They share STATE, not behavior — hence a plain holder
// by composition, not a base class.
//
//   writeLock — FIFO async mutex serializing ALL index mutations (reindexAll +
//     reindexDelta) so two writers never overlap; a full reindex closes+nukes the
//     DB out from under a delta otherwise. A delta issued during a full reindex
//     queues behind it; the full reindex subsumes the changes, so the queued delta
//     is still correct (and mostly a no-op). See runExclusive.
//   currentDelta — set ONLY while a delta is mutating the store (NOT during a full
//     reindex), so ensureFrame can wait out a delta — reading the fully-applied
//     result, never a half-committed one (a multi-file delta isn't one atomic
//     transaction) — WITHOUT blocking on a full reindex, whose progressive
//     "queryable as it fills" read is intended. null when idle.
//   generation — bumped on every index mutation; the BM25 / resident-binary /
//     resident-frame caches are generation-keyed (a cache built at gen N is stale
//     once gen > N). The sole bump site is bumpGeneration(), called from
//     SearchOrchestrator.invalidateBm25Cache().

import type { SeekSettings } from './types';

export class IndexCoordinator {
    private writeLock: Promise<unknown> = Promise.resolve();
    // Public: the indexing side sets/clears it around a delta's critical section,
    // the searching side (ensureFrame) waits on it. null when idle.
    currentDelta: Promise<void> | null = null;
    private _generation = 0;
    // Count of write critical sections currently executing inside runExclusive.
    // The mutex serializes writers so this is 0 or 1 in practice, but a counter is
    // robust to nesting and reads as a simple "is the index being mutated right now?"
    // signal. The reconcile poll consults it (isWriting) to NEVER launch an identity
    // heal while a long reindex/cold-build is still running — the precise gap that
    // let a 415s reindex outlive the 300s poll and get reconciled out from under
    // itself on a large vault. See SeekPlugin.periodicReconcile / enforceIndexIdentity.
    private _activeWriters = 0;

    // indexDir: sidecar index directory (`<pluginDir>/index`), or null when the
    // plugin couldn't resolve its manifest dir / in tests. settings is the LIVE
    // reference the plugin mutates in place, so sidecarOn() always reads current
    // values without a re-wire.
    constructor(
        private indexDir: string | null,
        private settings: SeekSettings,
    ) {}

    // Cache generation. Reads are cheap; the only mutation is bumpGeneration().
    get generation(): number {
        return this._generation;
    }

    bumpGeneration(): void {
        this._generation++;
    }

    // True while any index-mutation critical section (reindexAll OR reindexDelta —
    // the cold first-build runs through the latter) is executing under the mutex.
    // The reconcile poll defers on this so a running build is never healed mid-flight.
    isWriting(): boolean {
        return this._activeWriters > 0;
    }

    // True when sidecar persistence is configured AND enabled. Read live at every
    // write/hydrate site so toggling the setting takes effect on the next op.
    sidecarOn(): boolean {
        return this.indexDir !== null && this.settings.sidecarEnabled;
    }

    // The sidecar index directory; null when unconfigured. Callers that have
    // already checked sidecarOn() assert non-null (coord.dir!).
    get dir(): string | null {
        return this.indexDir;
    }

    // FIFO async mutex for index mutations. Each caller chains on the previous op's
    // completion before running, so calls execute in arrival order and never
    // overlap. A prior op's rejection must not wedge the queue, so we swallow it
    // here (the op logged its own error).
    async runExclusive<T>(fn: () => Promise<T>): Promise<T> {
        const prev = this.writeLock;
        let release!: () => void;
        this.writeLock = new Promise<void>(r => {
            release = r;
        });
        await prev.catch(() => {
            /* prior op logged its own failure */
        });
        // Mark the critical section live only AFTER acquiring the lock, so isWriting
        // reflects the running writer, not one still queued behind another.
        this._activeWriters++;
        try {
            return await fn();
        } finally {
            this._activeWriters--;
            release();
        }
    }
}
