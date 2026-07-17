import { describe, it, expect } from 'vitest';
import { TaskContextTracker } from './task-context';

// Injectable clock: tests advance time explicitly so span boundaries are exact.
function makeClock(start = 0) {
    let t = start;
    return { now: () => t, set: (v: number) => { t = v; } };
}

describe('TaskContextTracker', () => {
    it('attributes a task inside a span to that span', () => {
        const clock = makeClock();
        const tr = new TaskContextTracker(clock.now);
        clock.set(100); tr.push('indexing');
        clock.set(900); tr.pop('indexing');
        // task ran 200..700, delivered later — still attributes correctly
        expect(tr.attribute(200, 500)).toBe('indexing');
    });

    it('attributes the FINAL task of a phase even when delivered after pop (the old-stack failure)', () => {
        const clock = makeClock();
        const tr = new TaskContextTracker(clock.now);
        clock.set(0); tr.push('indexing');
        clock.set(1000); tr.pop('indexing');
        // The observer callback fires at t=1050, after pop — the old stack
        // would have read 'idle' here. Overlap attribution recovers it.
        clock.set(1050);
        expect(tr.attribute(400, 600)).toBe('indexing');
    });

    it('returns idle for a task with no overlapping span', () => {
        const clock = makeClock();
        const tr = new TaskContextTracker(clock.now);
        clock.set(0); tr.push('search');
        clock.set(100); tr.pop('search');
        expect(tr.attribute(200, 50)).toBe('idle');
    });

    it('prefers the innermost (latest-started) span on full containment', () => {
        const clock = makeClock();
        const tr = new TaskContextTracker(clock.now);
        clock.set(0); tr.push('search');
        clock.set(100); tr.push('bm25-warm');
        clock.set(900); tr.pop('bm25-warm');
        clock.set(1000); tr.pop('search');
        // task 200..800 is inside BOTH spans — equal overlap, inner wins
        expect(tr.attribute(200, 600)).toBe('bm25-warm');
    });

    it('picks the span with the larger overlap when spans are disjoint', () => {
        const clock = makeClock();
        const tr = new TaskContextTracker(clock.now);
        clock.set(0); tr.push('model-load');
        clock.set(300); tr.pop('model-load');
        clock.set(300); tr.push('indexing');
        clock.set(1000); tr.pop('indexing');
        // task 100..900: 200ms in model-load, 600ms in indexing
        expect(tr.attribute(100, 800)).toBe('indexing');
    });

    it('treats an open span as covering now (attribution mid-phase)', () => {
        const clock = makeClock();
        const tr = new TaskContextTracker(clock.now);
        clock.set(0); tr.push('catchup');
        clock.set(5000);
        expect(tr.attribute(1000, 2000)).toBe('catchup');
    });

    it('pop closes the last open span of that context (interleaved lifetimes)', () => {
        const clock = makeClock();
        const tr = new TaskContextTracker(clock.now);
        clock.set(0); tr.push('indexing');      // reindex begins
        clock.set(100); tr.push('search');      // modal opens mid-reindex
        clock.set(200); tr.pop('search');       // search finishes first
        clock.set(1000); tr.pop('indexing');    // reindex outlives it
        expect(tr.attribute(500, 100)).toBe('indexing');
        expect(tr.attribute(120, 60)).toBe('search');
    });

    it('double-pop is harmless', () => {
        const clock = makeClock();
        const tr = new TaskContextTracker(clock.now);
        clock.set(0); tr.push('search');
        clock.set(100); tr.pop('search');
        clock.set(110); tr.pop('search');   // no open span — ignored
        expect(tr.attribute(50, 30)).toBe('search');
    });

    it('prunes old closed spans but never open ones', () => {
        const clock = makeClock();
        const tr = new TaskContextTracker(clock.now);
        clock.set(0); tr.push('indexing');   // stays open the whole test
        // flood with short closed spans far in the past to trigger pruning
        for (let i = 0; i < 70; i++) {
            clock.set(1000 + i); tr.push('search'); tr.pop('search');
        }
        clock.set(500_000);   // way past retention
        for (let i = 0; i < 70; i++) {
            clock.set(500_000 + i); tr.push('search'); tr.pop('search');
        }
        // the ancient closed spans are gone…
        expect(tr.attribute(1000, 30)).toBe('indexing');   // covered only by the open span now
        // …but the open span survived pruning
        expect(tr.attribute(499_000, 100)).toBe('indexing');
    });
});
