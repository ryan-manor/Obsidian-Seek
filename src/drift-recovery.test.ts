import { describe, it, expect } from 'vitest';
import {
    driftRecoveryDecision,
    shouldDiscardPartialFrame,
    type DriftRecoveryState,
} from './search';

// The two pure predicates behind graceful drift recovery. Both are engine-free, so
// they're unit-tested directly (no live SearchOrchestrator) — the same house style as
// coherenceDriftDecision in applydelta-coherence.test.ts. verifyCoherent's core check
// (frameBm25Coherent, full=true) is already exhaustively covered there (every
// corruption mode), so it isn't separately re-tested here.

const HEALTHS: DriftRecoveryState['health'][] = ['healthy', 'recovering', 'degraded'];

function state(p: Partial<DriftRecoveryState>): DriftRecoveryState {
    return { running: false, health: 'healthy', lastRecoveryGen: -1, currentGen: 0, ...p };
}

describe('driftRecoveryDecision', () => {
    it('never schedules while a recovery is already running (any health)', () => {
        for (const health of HEALTHS) {
            // running short-circuits before the generation check — even a brand-new
            // generation must wait for the in-flight recovery to finish.
            expect(
                driftRecoveryDecision(state({ running: true, health, lastRecoveryGen: 2, currentGen: 9 })).schedule,
            ).toBe(false);
        }
    });

    it('schedules the first escalation (lastRecoveryGen = -1)', () => {
        // Never escalated for this index ⇒ -1 can never equal a real generation.
        expect(driftRecoveryDecision(state({ lastRecoveryGen: -1, currentGen: 0 })).schedule).toBe(true);
        expect(driftRecoveryDecision(state({ lastRecoveryGen: -1, currentGen: 7 })).schedule).toBe(true);
    });

    it('suppresses re-escalation at the same generation — the per-keystroke guard', () => {
        // A degraded index re-trips drift on every keystroke, but currentGen does not
        // move, so we must NOT re-run the ladder. Health is irrelevant to this.
        expect(
            driftRecoveryDecision(state({ health: 'degraded', lastRecoveryGen: 4, currentGen: 4 })).schedule,
        ).toBe(false);
        expect(
            driftRecoveryDecision(state({ health: 'recovering', lastRecoveryGen: 4, currentGen: 4 })).schedule,
        ).toBe(false);
        // same-gen ⇒ always false, even on the (unusual) healthy-but-same-gen case.
        expect(
            driftRecoveryDecision(state({ health: 'healthy', lastRecoveryGen: 4, currentGen: 4 })).schedule,
        ).toBe(false);
    });

    it('re-arms after a degrade once the generation advances (a real mutation)', () => {
        // A later delta/reindex/invalidate/hydrate bumped the generation past the one we
        // last escalated for ⇒ recover the NEW index state.
        expect(
            driftRecoveryDecision(state({ running: false, health: 'degraded', lastRecoveryGen: 4, currentGen: 5 })).schedule,
        ).toBe(true);
        // Any mismatch re-arms — including a defensively LOWER current generation. The
        // counter is monotonic in practice, but this pins the contract so a future
        // refactor of '!==' to '>' can't silently drop the re-arm (mirrors the
        // shouldDiscardPartialFrame defensive case).
        expect(
            driftRecoveryDecision(state({ running: false, health: 'degraded', lastRecoveryGen: 5, currentGen: 4 })).schedule,
        ).toBe(true);
    });

    it('does not schedule when the generation advanced but a recovery is still running', () => {
        // running wins over the re-arm — the in-flight ladder will re-baseline on finish.
        expect(
            driftRecoveryDecision(state({ running: true, health: 'degraded', lastRecoveryGen: 4, currentGen: 5 })).schedule,
        ).toBe(false);
    });

    it('ignores health entirely — only running + generation decide', () => {
        // Same (running, lastRecoveryGen, currentGen) ⇒ same verdict across all healths.
        for (const health of HEALTHS) {
            expect(driftRecoveryDecision(state({ health, lastRecoveryGen: 3, currentGen: 3 })).schedule).toBe(false);
            expect(driftRecoveryDecision(state({ health, lastRecoveryGen: 3, currentGen: 8 })).schedule).toBe(true);
        }
    });
});

describe('shouldDiscardPartialFrame', () => {
    it('keeps a frame built at the current generation', () => {
        // build === current ⇒ nothing changed under us; cache it.
        expect(shouldDiscardPartialFrame(5, 5)).toBe(false);
        expect(shouldDiscardPartialFrame(0, 0)).toBe(false);
    });

    it('discards when the index advanced mid-build (reindex completed under us)', () => {
        // current > build ⇒ a full reindex finished while we assembled; the partial
        // frame is stale, must rebuild (the call site re-enters ensureFrame).
        expect(shouldDiscardPartialFrame(5, 6)).toBe(true);
    });

    it('discards on any mismatch, including a defensively lower current generation', () => {
        // The counter is monotonic in practice; treat current !== build as discard
        // unconditionally so a surprising regression can never cache a stale frame.
        expect(shouldDiscardPartialFrame(5, 4)).toBe(true);
    });
});
