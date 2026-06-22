// Unit tests for the rev-keyed settings migration (types.ts migrateSettings), applied
// to the raw persisted data.json BEFORE it's merged over DEFAULT_SETTINGS in onload.
// The rev-5 ratification (2026-06-19) flips three validated-OFF toggles ON and remaps
// two numeric defaults — critically navTitleBoost 0.8→0.5, since a persisted 0.8 would
// otherwise read as the new "High" segmented stage and silently promote every upgrader.
// A hand-tuned value must survive.

import { describe, it, expect } from 'vitest';
import { migrateSettings, type SeekSettings } from './types';

describe('migrateSettings — rev 5 defaults ratification', () => {
    it('flips synonym/headings/sidecar ON and remaps old-default numerics for a pre-rev-5 install', () => {
        const raw: Partial<SeekSettings> = {
            settingsRev: 4,
            navTitleBoost: 0.8,   // the OLD default
            recencyEpsilon: 0.02, // the OLD always-on tiebreaker
            synonymExpansion: false,
            headingsField: false,
            sidecarEnabled: false,
        };
        migrateSettings(raw);
        expect(raw.synonymExpansion).toBe(true);
        expect(raw.headingsField).toBe(true);
        expect(raw.sidecarEnabled).toBe(true);
        expect(raw.navTitleBoost).toBe(0.5); // 0.8 (old default) → 0.5 (new Default stage)
        expect(raw.recencyEpsilon).toBe(0);  // 0.02 (old tiebreaker) → 0 (ships Off)
        expect(raw.settingsRev).toBe(6);
    });

    it('preserves a hand-tuned navTitleBoost / recencyEpsilon (only the exact old default is moved)', () => {
        const raw: Partial<SeekSettings> = { settingsRev: 4, navTitleBoost: 0.7, recencyEpsilon: 0.05 };
        migrateSettings(raw);
        expect(raw.navTitleBoost).toBe(0.7);
        expect(raw.recencyEpsilon).toBe(0.05);
        expect(raw.sidecarEnabled).toBe(true); // the unconditional boolean flips still apply
        expect(raw.settingsRev).toBe(6);
    });

    it('does not re-touch the rev-5 fields of an install already at rev 5 (only advances to rev 6)', () => {
        const raw: Partial<SeekSettings> = { settingsRev: 5, navTitleBoost: 0.8, sidecarEnabled: false };
        migrateSettings(raw);
        expect(raw.navTitleBoost).toBe(0.8);    // not remapped — already past rev 5
        expect(raw.sidecarEnabled).toBe(false); // not forced — already past rev 5
        expect(raw.settingsRev).toBe(6);        // the rev-6 rename clause still advances the rev
    });

    it('gives a fresh/empty data.json (treated as rev 1) the rev-5 baseline', () => {
        const raw: Partial<SeekSettings> = {};
        migrateSettings(raw);
        expect(raw.synonymExpansion).toBe(true);
        expect(raw.headingsField).toBe(true);
        expect(raw.sidecarEnabled).toBe(true);
        expect(raw.navTitleBoost).toBe(0.5); // undefined → new default
        expect(raw.recencyEpsilon).toBe(0);
        expect(raw.settingsRev).toBe(6);
    });

    it('still applies the rev-2 denseWeight surgery for a pre-bound install (plus the rev-5 flips)', () => {
        const raw: Partial<SeekSettings> = { settingsRev: 1, denseWeight: 0.92, navTitleBoost: 0.8 };
        migrateSettings(raw);
        expect(raw.denseWeight).toBeUndefined(); // empirical-max value dropped → rev-2 default takes over
        expect(raw.navTitleBoost).toBe(0.5);     // rev-5 flips also apply (rev 1 < 5)
        expect(raw.synonymExpansion).toBe(true);
        expect(raw.settingsRev).toBe(6);
    });

    it('mutates and returns the same object (onload relies on the mutation)', () => {
        const raw: Partial<SeekSettings> = { settingsRev: 4 };
        const out = migrateSettings(raw);
        expect(out).toBe(raw);
    });
});

describe('migrateSettings — rev 6 debugMode→showScores rename', () => {
    it('carries an explicit debugMode:false from a rev-5 install into showScores:false', () => {
        // The exact regression the rev bump prevents: a user who turned the old
        // per-row score line OFF must not have it silently turned back ON.
        const raw = { settingsRev: 5, debugMode: false } as Partial<SeekSettings>;
        migrateSettings(raw);
        expect(raw.showScores).toBe(false);
        expect((raw as { debugMode?: boolean }).debugMode).toBeUndefined(); // orphan key dropped
        expect(raw.settingsRev).toBe(6);
    });

    it('carries debugMode:true into showScores:true', () => {
        const raw = { settingsRev: 5, debugMode: true } as Partial<SeekSettings>;
        migrateSettings(raw);
        expect(raw.showScores).toBe(true);
        expect(raw.settingsRev).toBe(6);
    });

    it('does not clobber an already-persisted showScores with the legacy key', () => {
        const raw = { settingsRev: 5, debugMode: true, showScores: false } as Partial<SeekSettings>;
        migrateSettings(raw);
        expect(raw.showScores).toBe(false); // explicit showScores wins over the orphan debugMode
        expect((raw as { debugMode?: boolean }).debugMode).toBeUndefined();
    });

    it('leaves showScores undefined (→ default) when neither key was persisted', () => {
        const raw: Partial<SeekSettings> = { settingsRev: 5 };
        migrateSettings(raw);
        expect(raw.showScores).toBeUndefined(); // falls through to DEFAULT_SETTINGS.showScores in onload
        expect(raw.settingsRev).toBe(6);
    });
});
