// Characterization tests for the index-mutation coordination invariants, factored
// out of SearchOrchestrator (which has no unit harness). These pin the behaviors a
// future Indexer/Searcher split must preserve: FIFO write-mutex serialization, the
// monotonic cache-generation counter, and live (by-reference) settings reads.

import { describe, it, expect } from 'vitest';
import type { SeekSettings } from './types';
import { IndexCoordinator } from './index-coordinator';

// sidecarOn() reads only settings.sidecarEnabled; a partial cast is sufficient and
// keeps the test independent of the (large) SeekSettings surface.
function settings(sidecarEnabled: boolean): SeekSettings {
    return { sidecarEnabled } as unknown as SeekSettings;
}

describe('IndexCoordinator: write mutex (runExclusive)', () => {
    it('serializes — no two ops overlap, and they run in FIFO arrival order', async () => {
        const coord = new IndexCoordinator(null, settings(false));
        const events: string[] = [];
        let active = 0;
        const op = (id: number) =>
            coord.runExclusive(async () => {
                events.push(`enter${id}`);
                active++;
                expect(active).toBe(1); // never more than one critical section at a time
                await Promise.resolve(); // yield: an unserialized impl would let another in here
                await new Promise(r => setTimeout(r, 1));
                active--;
                events.push(`exit${id}`);
            });
        await Promise.all([op(1), op(2), op(3)]);
        expect(events).toEqual(['enter1', 'exit1', 'enter2', 'exit2', 'enter3', 'exit3']);
    });

    it('a rejected op does not wedge the queue (the next op still runs)', async () => {
        const coord = new IndexCoordinator(null, settings(false));
        const order: string[] = [];
        const bad = coord.runExclusive(async () => {
            order.push('bad');
            throw new Error('boom');
        });
        const good = coord.runExclusive(async () => {
            order.push('good');
            return 42;
        });
        await expect(bad).rejects.toThrow('boom');
        await expect(good).resolves.toBe(42);
        expect(order).toEqual(['bad', 'good']);
    });

    it('returns the wrapped fn result', async () => {
        const coord = new IndexCoordinator(null, settings(false));
        await expect(coord.runExclusive(async () => 'ok')).resolves.toBe('ok');
    });
});

describe('IndexCoordinator: cache generation', () => {
    it('starts at 0 and only bumpGeneration advances it (the sole mutation)', () => {
        const coord = new IndexCoordinator(null, settings(false));
        expect(coord.generation).toBe(0);
        coord.bumpGeneration();
        expect(coord.generation).toBe(1);
        coord.bumpGeneration();
        expect(coord.generation).toBe(2);
    });
});

describe('IndexCoordinator: isWriting (reconcile-poll defer signal)', () => {
    // The signal that fixes the large-vault loop: a reindex/cold-build whose wall time
    // exceeds the 5-min reconcile poll must NOT be reconciled out from under itself.
    // The poll consults isWriting() and defers while any write critical section runs.
    it('is false when idle and true only INSIDE a running critical section', async () => {
        const coord = new IndexCoordinator(null, settings(false));
        expect(coord.isWriting()).toBe(false);
        let insideWasWriting = false;
        await coord.runExclusive(async () => { insideWasWriting = coord.isWriting(); });
        expect(insideWasWriting).toBe(true);   // a long reindex/cold-build reads as "writing"
        expect(coord.isWriting()).toBe(false);  // and clears the instant it finishes
    });

    it('stays true while a writer holds the lock with another queued, false after both drain', async () => {
        const coord = new IndexCoordinator(null, settings(false));
        let releaseFirst!: () => void;
        const firstHolds = new Promise<void>(r => { releaseFirst = r; });
        const first = coord.runExclusive(() => firstHolds); // holds the lock open
        const second = coord.runExclusive(async () => { /* queued behind first */ });
        await new Promise(r => setTimeout(r, 0));           // let `first` acquire the lock
        expect(coord.isWriting()).toBe(true);               // the running build registers as writing
        releaseFirst();
        await Promise.all([first, second]);
        expect(coord.isWriting()).toBe(false);              // increments/decrements balance across both
    });

    it('clears even when the critical section throws (the finally decrements)', async () => {
        const coord = new IndexCoordinator(null, settings(false));
        await expect(coord.runExclusive(async () => { throw new Error('boom'); })).rejects.toThrow('boom');
        expect(coord.isWriting()).toBe(false);
    });
});

describe('IndexCoordinator: sidecar enablement + location', () => {
    it('requires both a dir and the live setting, and reads settings BY REFERENCE', () => {
        const s = settings(false);
        const withDir = new IndexCoordinator('.obsidian/plugins/seek/index', s);
        expect(withDir.sidecarOn()).toBe(false); // setting off → off
        (s as unknown as { sidecarEnabled: boolean }).sidecarEnabled = true;
        expect(withDir.sidecarOn()).toBe(true); // live ref picks up the in-place mutation
    });

    it('is never on without a dir, even when the setting is true', () => {
        const noDir = new IndexCoordinator(null, settings(true));
        expect(noDir.sidecarOn()).toBe(false);
        expect(noDir.dir).toBeNull();
    });

    it('exposes the configured dir', () => {
        const coord = new IndexCoordinator('.obsidian/plugins/seek/index', settings(true));
        expect(coord.dir).toBe('.obsidian/plugins/seek/index');
    });
});

describe('IndexCoordinator: delta gate', () => {
    it('currentDelta defaults to null (idle)', () => {
        const coord = new IndexCoordinator(null, settings(false));
        expect(coord.currentDelta).toBeNull();
    });
});
