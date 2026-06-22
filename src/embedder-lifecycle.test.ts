// shouldUnloadEmbedder gates the proactive embedder teardown. The contract that
// matters: NEVER tear down mid-flight (busy/query/running) regardless of reason,
// and only diverge on PENDING work — an idle unload stands down (a reload is
// imminent), a background unload proceeds (we're leaving; pending work reloads on
// the next foreground). These pin the iOS heap-reset behaviour.

import { describe, it, expect } from 'vitest';
import { shouldUnloadEmbedder, type UnloadGateState } from './embedder-lifecycle';

// A fully-quiescent, loaded embedder — the only state an idle unload may fire in.
const QUIESCENT: UnloadGateState = {
    loaded: true, busy: false, queryActive: false, running: false, pending: false,
};

describe('shouldUnloadEmbedder', () => {
    it('never unloads when the model is not loaded', () => {
        expect(shouldUnloadEmbedder('idle', { ...QUIESCENT, loaded: false })).toBe(false);
        expect(shouldUnloadEmbedder('background', { ...QUIESCENT, loaded: false })).toBe(false);
    });

    it('unloads when fully quiescent (both reasons)', () => {
        expect(shouldUnloadEmbedder('idle', QUIESCENT)).toBe(true);
        expect(shouldUnloadEmbedder('background', QUIESCENT)).toBe(true);
    });

    it('never interrupts in-flight work, for either reason', () => {
        for (const reason of ['idle', 'background'] as const) {
            expect(shouldUnloadEmbedder(reason, { ...QUIESCENT, busy: true })).toBe(false);
            expect(shouldUnloadEmbedder(reason, { ...QUIESCENT, queryActive: true })).toBe(false);
            expect(shouldUnloadEmbedder(reason, { ...QUIESCENT, running: true })).toBe(false);
        }
    });

    it('idle stands down on pending work; background proceeds — the key asymmetry', () => {
        const pending = { ...QUIESCENT, pending: true };
        expect(shouldUnloadEmbedder('idle', pending)).toBe(false);
        expect(shouldUnloadEmbedder('background', pending)).toBe(true);
    });

    it('running always beats pending — background still refuses mid-flight work', () => {
        expect(shouldUnloadEmbedder('background', { ...QUIESCENT, running: true, pending: true })).toBe(false);
    });
});
