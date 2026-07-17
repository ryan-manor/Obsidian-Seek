// Unit tests for the rev-keyed settings migration (types.ts migrateSettings), applied
// to the raw persisted data.json BEFORE it's merged over DEFAULT_SETTINGS in onload.
// The rev-5 ratification (2026-06-19) flips three validated-OFF toggles ON and remaps
// two numeric defaults — critically navTitleBoost 0.8→0.5, since a persisted 0.8 would
// otherwise read as the new "High" segmented stage and silently promote every upgrader.
// A hand-tuned value must survive.

import { describe, it, expect } from 'vitest';
import { migrateSettings, DEFAULT_SETTINGS, type SeekSettings } from './types';

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
        expect(raw.settingsRev).toBe(9);
    });

    it('preserves a hand-tuned navTitleBoost / recencyEpsilon (only the exact old default is moved)', () => {
        const raw: Partial<SeekSettings> = { settingsRev: 4, navTitleBoost: 0.7, recencyEpsilon: 0.05 };
        migrateSettings(raw);
        expect(raw.navTitleBoost).toBe(0.7);
        expect(raw.recencyEpsilon).toBe(0.05);
        expect(raw.sidecarEnabled).toBe(true); // the unconditional boolean flips still apply
        expect(raw.settingsRev).toBe(9);
    });

    it('does not re-touch the rev-5 fields of an install already at rev 5 (advances to the latest rev)', () => {
        const raw: Partial<SeekSettings> = { settingsRev: 5, navTitleBoost: 0.8, sidecarEnabled: false };
        migrateSettings(raw);
        expect(raw.navTitleBoost).toBe(0.8);    // not remapped — already past rev 5
        expect(raw.sidecarEnabled).toBe(false); // not forced — already past rev 5
        expect(raw.settingsRev).toBe(9);        // later clauses (rev-6 rename, rev-7 props) still advance the rev
    });

    it('gives a fresh/empty data.json (treated as rev 1) the rev-5 baseline', () => {
        const raw: Partial<SeekSettings> = {};
        migrateSettings(raw);
        expect(raw.synonymExpansion).toBe(true);
        expect(raw.headingsField).toBe(true);
        expect(raw.sidecarEnabled).toBe(true);
        expect(raw.navTitleBoost).toBe(0.5); // undefined → new default
        expect(raw.recencyEpsilon).toBe(0);
        expect(raw.settingsRev).toBe(9);
    });

    it('still applies the rev-2 denseWeight surgery for a pre-bound install (plus the rev-5 flips)', () => {
        const raw: Partial<SeekSettings> = { settingsRev: 1, denseWeight: 0.92, navTitleBoost: 0.8 };
        migrateSettings(raw);
        expect(raw.denseWeight).toBeUndefined(); // empirical-max value dropped → rev-2 default takes over
        expect(raw.navTitleBoost).toBe(0.5);     // rev-5 flips also apply (rev 1 < 5)
        expect(raw.synonymExpansion).toBe(true);
        expect(raw.settingsRev).toBe(9);
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
        expect(raw.settingsRev).toBe(9);
    });

    it('carries debugMode:true into showScores:true', () => {
        const raw = { settingsRev: 5, debugMode: true } as Partial<SeekSettings>;
        migrateSettings(raw);
        expect(raw.showScores).toBe(true);
        expect(raw.settingsRev).toBe(9);
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
        expect(raw.settingsRev).toBe(9);
    });
});

describe('migrateSettings — rev 7 searchableProperties default ON', () => {
    it('flips a persisted searchableProperties:false ON for a pre-rev-7 install', () => {
        // The exact silent no-op the migration prevents: installs created under the old
        // default persisted `false`, which would win over the new DEFAULT_SETTINGS true.
        const raw: Partial<SeekSettings> = { settingsRev: 6, searchableProperties: false };
        migrateSettings(raw);
        expect(raw.searchableProperties).toBe(true);
        expect(raw.settingsRev).toBe(9);
    });

    it('turns it ON for an install that never persisted the key', () => {
        const raw: Partial<SeekSettings> = { settingsRev: 6 };
        migrateSettings(raw);
        expect(raw.searchableProperties).toBe(true);
        expect(raw.settingsRev).toBe(9);
    });

    it('does not re-flip an install already at rev 7 (a deliberate later OFF survives)', () => {
        const raw: Partial<SeekSettings> = { settingsRev: 7, searchableProperties: false };
        migrateSettings(raw);
        expect(raw.searchableProperties).toBe(false); // past rev 7 — the user's choice is preserved
        expect(raw.settingsRev).toBe(9);
    });
});

describe('migrateSettings — rev 8 denseWeight 0.80→0.85 re-eval', () => {
    it('moves a persisted denseWeight 0.80 (the old default) to 0.85 for a pre-rev-8 install', () => {
        // The silent no-op the migration prevents: every existing install persisted the
        // rev-2 default 0.80, which would win over the new DEFAULT_SETTINGS 0.85.
        const raw: Partial<SeekSettings> = { settingsRev: 7, denseWeight: 0.80 };
        migrateSettings(raw);
        expect(raw.denseWeight).toBe(0.85);
        expect(raw.settingsRev).toBe(9);
    });

    it('preserves a hand-tuned denseWeight (only the exact old default 0.80 is moved)', () => {
        const raw: Partial<SeekSettings> = { settingsRev: 7, denseWeight: 0.90 };
        migrateSettings(raw);
        expect(raw.denseWeight).toBe(0.90);
        expect(raw.settingsRev).toBe(9);
    });

    it('leaves an unpersisted denseWeight undefined (→ new DEFAULT_SETTINGS 0.85 in onload)', () => {
        const raw: Partial<SeekSettings> = { settingsRev: 7 };
        migrateSettings(raw);
        expect(raw.denseWeight).toBeUndefined();
        expect(raw.settingsRev).toBe(9);
    });

    it('does not re-move a deliberate later 0.80 for an install already at rev 8', () => {
        const raw: Partial<SeekSettings> = { settingsRev: 8, denseWeight: 0.80 };
        migrateSettings(raw);
        expect(raw.denseWeight).toBe(0.80); // past rev 8 — the user's choice is preserved
        expect(raw.settingsRev).toBe(9);
    });

    it('never DOWNGRADES the rev stamp of a data.json written by a newer Seek', () => {
        // Cross-device sync hazard: a rev-10+ data.json loaded by this older build
        // must keep its stamp, or the newer device re-runs its migrations on next
        // load (conditional default moves misfire on second application).
        const raw: Partial<SeekSettings> = { settingsRev: 10, denseWeight: 0.70, recencyHalfLifeDays: 270 };
        migrateSettings(raw);
        expect(raw.settingsRev).toBe(10);            // not stamped back to 9
        expect(raw.denseWeight).toBe(0.70);          // and no rev<8 migration re-fired
        expect(raw.recencyHalfLifeDays).toBe(270);   // nor the rev-9 half-life move
    });
});

describe('migrateSettings — rev 9 Recency High half-life 270→90', () => {
    it('moves a persisted 270 (the old High stage) to 90 for a pre-rev-9 install', () => {
        // The silent no-op the migration prevents: High is a value map in settings-tab.ts,
        // so a High user's fix lives entirely in their persisted data.json. recencyStageOf()
        // snaps the pill on ε alone, so ε=0.1 keeps rendering "High" while the stale 270
        // keeps ranking — the mode would look correct and stay broken.
        const raw: Partial<SeekSettings> = { settingsRev: 8, recencyEpsilon: 0.1, recencyHalfLifeDays: 270 };
        migrateSettings(raw);
        expect(raw.recencyHalfLifeDays).toBe(90);
        expect(raw.recencyEpsilon).toBe(0.1); // ε was never the problem — untouched
        expect(raw.settingsRev).toBe(9);
    });

    it('leaves Off/Default installs (180) alone — only the exact old High value moves', () => {
        const raw: Partial<SeekSettings> = { settingsRev: 8, recencyEpsilon: 0.04, recencyHalfLifeDays: 180 };
        migrateSettings(raw);
        expect(raw.recencyHalfLifeDays).toBe(180);
        expect(raw.settingsRev).toBe(9);
    });

    it('leaves an unpersisted half-life undefined (→ DEFAULT_SETTINGS 180 in onload)', () => {
        const raw: Partial<SeekSettings> = { settingsRev: 8 };
        migrateSettings(raw);
        expect(raw.recencyHalfLifeDays).toBeUndefined();
        expect(raw.settingsRev).toBe(9);
    });

    it('does not re-move a deliberate later 270 for an install already at rev 9', () => {
        const raw: Partial<SeekSettings> = { settingsRev: 9, recencyHalfLifeDays: 270 };
        migrateSettings(raw);
        expect(raw.recencyHalfLifeDays).toBe(270); // past rev 9 — the user's choice is preserved
        expect(raw.settingsRev).toBe(9);
    });

    it('leaves a pre-rev-5 install that hits BOTH the rev-5 eps clause and this one coherent', () => {
        // A raw-knob-era install can carry hl=270 with the OLD 0.02 tiebreaker eps, so a
        // single migrateSettings pass runs rev-5 (eps 0.02 -> 0, i.e. Off) AND rev-9
        // (hl 270 -> 90). The result reads oddly — Off with a High-shaped half-life — but
        // is inert by construction, and this test pins WHY: the half-life only ever reaches
        // the score through the eps product (ranker.ts `eps * recency[i]`), so eps=0 makes
        // it unobservable. topByRecency/browseOrder read recencyDate, never halfLifeDays.
        // It self-heals on the next stage pick, since the segmented control writes both
        // fields together. If a future consumer reads halfLifeDays OUTSIDE the eps product,
        // this combination becomes a real bug and this test is where it should surface.
        const raw: Partial<SeekSettings> = { settingsRev: 4, recencyEpsilon: 0.02, recencyHalfLifeDays: 270 };
        migrateSettings(raw);
        expect(raw.recencyEpsilon).toBe(0);       // rev-5: retires the always-on tiebreaker
        expect(raw.recencyHalfLifeDays).toBe(90); // rev-9: still fires — inert while eps is 0
        expect(raw.settingsRev).toBe(9);
    });
});

describe('migrateSettings — rev stamp invariant', () => {
    it('stamps exactly DEFAULT_SETTINGS.settingsRev on a fresh install', () => {
        // Couples the Math.max() target in migrateSettings to DEFAULT_SETTINGS.settingsRev.
        // Without this, bumping one and not the other leaves every other test green while
        // the two silently disagree — and the whole rev scheme rests on them matching.
        const raw: Partial<SeekSettings> = {};
        migrateSettings(raw);
        expect(raw.settingsRev).toBe(DEFAULT_SETTINGS.settingsRev);
    });
});
