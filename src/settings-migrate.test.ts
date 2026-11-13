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
        expect(raw.navTitleBoost).toBe(0.25); // 0.8 (old default) → 0.5 (rev 5) → 0.25 (rev 12 Default)
        expect(raw.recencyEpsilon).toBe(0);  // 0.02 (old tiebreaker) → 0 (ships Off)
        expect(raw.settingsRev).toBe(13);
    });

    it('preserves a hand-tuned navTitleBoost / recencyEpsilon (only the exact old default is moved)', () => {
        const raw: Partial<SeekSettings> = { settingsRev: 4, navTitleBoost: 0.7, recencyEpsilon: 0.05 };
        migrateSettings(raw);
        expect(raw.navTitleBoost).toBe(0.7);
        expect(raw.recencyEpsilon).toBe(0.05);
        expect(raw.sidecarEnabled).toBe(true); // the unconditional boolean flips still apply
        expect(raw.settingsRev).toBe(13);
    });

    it('does not re-touch the rev-5 fields of an install already at rev 5 (advances to the latest rev)', () => {
        const raw: Partial<SeekSettings> = { settingsRev: 5, navTitleBoost: 0.8, sidecarEnabled: false };
        migrateSettings(raw);
        expect(raw.navTitleBoost).toBe(0.5);    // not hit by rev 5; rev 12 maps the old High 0.8 → new High 0.5
        expect(raw.sidecarEnabled).toBe(false); // not forced — already past rev 5
        expect(raw.settingsRev).toBe(13);        // later clauses (rev-6 rename, rev-7 props) still advance the rev
    });

    it('gives a fresh/empty data.json (treated as rev 1) the rev-5 baseline', () => {
        const raw: Partial<SeekSettings> = {};
        migrateSettings(raw);
        expect(raw.synonymExpansion).toBe(true);
        expect(raw.headingsField).toBe(true);
        expect(raw.sidecarEnabled).toBe(true);
        expect(raw.navTitleBoost).toBe(0.25); // undefined → 0.5 (rev 5) → 0.25 (rev 12)
        expect(raw.recencyEpsilon).toBe(0);
        expect(raw.settingsRev).toBe(13);
    });

    it('still applies the rev-2 denseWeight surgery for a pre-bound install (plus the rev-5 flips)', () => {
        const raw: Partial<SeekSettings> = { settingsRev: 1, denseWeight: 0.92, navTitleBoost: 0.8 };
        migrateSettings(raw);
        expect(raw.denseWeight).toBeUndefined(); // empirical-max value dropped → rev-2 default takes over
        expect(raw.navTitleBoost).toBe(0.25);    // rev-5 flips also apply (rev 1 < 5), then rev 12
        expect(raw.synonymExpansion).toBe(true);
        expect(raw.settingsRev).toBe(13);
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
        expect(raw.settingsRev).toBe(13);
    });

    it('carries debugMode:true into showScores:true', () => {
        const raw = { settingsRev: 5, debugMode: true } as Partial<SeekSettings>;
        migrateSettings(raw);
        expect(raw.showScores).toBe(true);
        expect(raw.settingsRev).toBe(13);
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
        expect(raw.settingsRev).toBe(13);
    });
});

describe('migrateSettings — rev 7 searchableProperties default ON', () => {
    it('flips a persisted searchableProperties:false ON for a pre-rev-7 install', () => {
        // The exact silent no-op the migration prevents: installs created under the old
        // default persisted `false`, which would win over the new DEFAULT_SETTINGS true.
        const raw: Partial<SeekSettings> = { settingsRev: 6, searchableProperties: false };
        migrateSettings(raw);
        expect(raw.searchableProperties).toBe(true);
        expect(raw.settingsRev).toBe(13);
    });

    it('turns it ON for an install that never persisted the key', () => {
        const raw: Partial<SeekSettings> = { settingsRev: 6 };
        migrateSettings(raw);
        expect(raw.searchableProperties).toBe(true);
        expect(raw.settingsRev).toBe(13);
    });

    it('does not re-flip an install already at rev 7 (a deliberate later OFF survives)', () => {
        const raw: Partial<SeekSettings> = { settingsRev: 7, searchableProperties: false };
        migrateSettings(raw);
        expect(raw.searchableProperties).toBe(false); // past rev 7 — the user's choice is preserved
        expect(raw.settingsRev).toBe(13);
    });
});

describe('migrateSettings — rev 8 denseWeight 0.80→0.85 re-eval', () => {
    it('moves a persisted denseWeight 0.80 (the old default) to 0.85 for a pre-rev-8 install', () => {
        // The silent no-op the migration prevents: every existing install persisted the
        // rev-2 default 0.80, which would win over the new DEFAULT_SETTINGS 0.85.
        const raw: Partial<SeekSettings> = { settingsRev: 7, denseWeight: 0.80 };
        migrateSettings(raw);
        expect(raw.denseWeight).toBe(0.85);
        expect(raw.settingsRev).toBe(13);
    });

    it('preserves a hand-tuned denseWeight (only the exact old default 0.80 is moved)', () => {
        const raw: Partial<SeekSettings> = { settingsRev: 7, denseWeight: 0.90 };
        migrateSettings(raw);
        expect(raw.denseWeight).toBe(0.90);
        expect(raw.settingsRev).toBe(13);
    });

    it('leaves an unpersisted denseWeight undefined (→ new DEFAULT_SETTINGS 0.85 in onload)', () => {
        const raw: Partial<SeekSettings> = { settingsRev: 7 };
        migrateSettings(raw);
        expect(raw.denseWeight).toBeUndefined();
        expect(raw.settingsRev).toBe(13);
    });

    it('rev 13 repairs a 0.80 written by the old Balanced dropdown (rev 8-12)', () => {
        // The dropdown hard-coded Balanced = 0.8, so a 0.80 persisted after rev 8 is
        // that bug, not a deliberate choice.
        const raw: Partial<SeekSettings> = { settingsRev: 12, denseWeight: 0.80 };
        migrateSettings(raw);
        expect(raw.denseWeight).toBe(0.85);
        expect(raw.settingsRev).toBe(13);
    });

    it('rev 13 keeps hand-tuned dense weights, and repairs a 0.80 synced in under a rev-13 stamp', () => {
        const tuned: Partial<SeekSettings> = { settingsRev: 13, denseWeight: 0.75 };
        migrateSettings(tuned);
        expect(tuned.denseWeight).toBe(0.75);
        // An older build on another device picks Balanced (writes 0.8) and syncs back.
        const later: Partial<SeekSettings> = { settingsRev: 13, denseWeight: 0.80 };
        migrateSettings(later);
        expect(later.denseWeight).toBe(0.85);
        migrateSettings(later);
        expect(later.denseWeight).toBe(0.85);
    });

    it('never DOWNGRADES the rev stamp of a data.json written by a newer Seek', () => {
        // Cross-device sync hazard: a data.json stamped by a NEWER Seek (rev 14+)
        // loaded by this build must keep its stamp, or the newer device re-runs its
        // migrations on next load (conditional default moves misfire on second application).
        const raw: Partial<SeekSettings> = { settingsRev: 14, denseWeight: 0.80, recencyHalfLifeDays: 270, navTitleBoost: 0.5 };
        migrateSettings(raw);
        expect(raw.settingsRev).toBe(14);            // not stamped back to 13
        expect(raw.denseWeight).toBe(0.80);          // and no rev<8 or rev-13 dense move re-fired
        expect(raw.recencyHalfLifeDays).toBe(270);   // nor the rev-9 half-life move
        expect(raw.navTitleBoost).toBe(0.5);         // nor the rev-12 title remap
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
        expect(raw.settingsRev).toBe(13);
    });

    it('leaves Off/Default installs (180) alone — only the exact old High value moves', () => {
        const raw: Partial<SeekSettings> = { settingsRev: 8, recencyEpsilon: 0.04, recencyHalfLifeDays: 180 };
        migrateSettings(raw);
        expect(raw.recencyHalfLifeDays).toBe(180);
        expect(raw.settingsRev).toBe(13);
    });

    it('leaves an unpersisted half-life undefined (→ DEFAULT_SETTINGS 180 in onload)', () => {
        const raw: Partial<SeekSettings> = { settingsRev: 8 };
        migrateSettings(raw);
        expect(raw.recencyHalfLifeDays).toBeUndefined();
        expect(raw.settingsRev).toBe(13);
    });

    it('does not re-move a deliberate later 270 for an install already at rev 9', () => {
        const raw: Partial<SeekSettings> = { settingsRev: 9, recencyHalfLifeDays: 270 };
        migrateSettings(raw);
        expect(raw.recencyHalfLifeDays).toBe(270); // past rev 9 — the user's choice is preserved
        expect(raw.settingsRev).toBe(13);
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
        expect(raw.settingsRev).toBe(13);
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

describe('migrateSettings — synonyms v2 (no migration by design)', () => {
    it('never writes synonymsV2 (Object.assign backfills the OFF default)', () => {
        // New key, no rev bump — the altOpenLocation precedent. A data.json
        // from ANY rev must come through untouched: absent stays absent...
        const raw: Partial<SeekSettings> = { settingsRev: 4 };
        migrateSettings(raw);
        expect('synonymsV2' in raw).toBe(false);
        // ...and an explicit persisted choice survives.
        const chosen: Partial<SeekSettings> = { settingsRev: DEFAULT_SETTINGS.settingsRev, synonymsV2: true };
        migrateSettings(chosen);
        expect(chosen.synonymsV2).toBe(true);
    });

    it('ships OFF: the default is the valve closed', () => {
        expect(DEFAULT_SETTINGS.synonymsV2).toBe(false);
    });
});

describe('migrateSettings — rev 10 hub-routing removal', () => {
    it('drops the orphan hubBonus key from a pre-rev-10 install', () => {
        const raw = { settingsRev: 9, hubBonus: 0.15 } as Partial<SeekSettings> & { hubBonus?: number };
        migrateSettings(raw);
        expect('hubBonus' in raw).toBe(false);
        expect(raw.settingsRev).toBe(13);
    });

    it('does NOT touch persisted recency choices (defaults reach only unpersisted keys)', () => {
        const raw: Partial<SeekSettings> = { settingsRev: 9, recencyEpsilon: 0, recencyHalfLifeDays: 180, recencyKey: 'modified' };
        migrateSettings(raw);
        expect(raw.recencyEpsilon).toBe(0);
        expect(raw.recencyHalfLifeDays).toBe(180);
        expect(raw.recencyKey).toBe('modified');
        expect(raw.settingsRev).toBe(13);
    });
});

describe('migrateSettings — rev 11 specificity-boost removal', () => {
    it('drops the orphan specificityBoost key from a pre-rev-11 install', () => {
        const raw = { settingsRev: 10, specificityBoost: 0.05 } as Partial<SeekSettings> & { specificityBoost?: number };
        migrateSettings(raw);
        expect('specificityBoost' in raw).toBe(false);
        expect(raw.settingsRev).toBe(13);
    });

    it('drops a hand-tuned S too — the key has no consumer left', () => {
        const raw = { settingsRev: 10, specificityBoost: 0.1 } as Partial<SeekSettings> & { specificityBoost?: number };
        migrateSettings(raw);
        expect('specificityBoost' in raw).toBe(false);
    });

    it('leaves an install already at rev 11 alone', () => {
        const raw = { settingsRev: 11, denseWeight: 0.85 } as Partial<SeekSettings>;
        migrateSettings(raw);
        expect(raw.settingsRev).toBe(13);
        expect(raw.denseWeight).toBe(0.85);
    });
});

describe('migrateSettings — rev 12 title-bonus stages', () => {
    it('maps the old Default 0.5 → 0.25 and old High 0.8 → 0.5 (no chaining)', () => {
        const d: Partial<SeekSettings> = { settingsRev: 11, navTitleBoost: 0.5 };
        const h: Partial<SeekSettings> = { settingsRev: 11, navTitleBoost: 0.8 };
        migrateSettings(d); migrateSettings(h);
        expect(d.navTitleBoost).toBe(0.25);
        expect(h.navTitleBoost).toBe(0.5);
        expect(d.settingsRev).toBe(13);
    });
    it('keeps Off and hand-tuned values, and does not re-run at rev 12', () => {
        for (const v of [0, 0.3, 0.7]) {
            const raw: Partial<SeekSettings> = { settingsRev: 11, navTitleBoost: v };
            migrateSettings(raw);
            expect(raw.navTitleBoost).toBe(v);
        }
        const raw: Partial<SeekSettings> = { settingsRev: 12, navTitleBoost: 0.5 };
        migrateSettings(raw);
        expect(raw.navTitleBoost).toBe(0.5);
    });
    it('upgrades from rev 9 / 10 / 11 identically (stage values remap, unpersisted stays undefined)', () => {
        for (const rev of [9, 10, 11]) {
            const d: Partial<SeekSettings> = { settingsRev: rev, navTitleBoost: 0.5 };
            const h: Partial<SeekSettings> = { settingsRev: rev, navTitleBoost: 0.8 };
            const o: Partial<SeekSettings> = { settingsRev: rev, navTitleBoost: 0 };
            const u: Partial<SeekSettings> = { settingsRev: rev };
            for (const r of [d, h, o, u]) migrateSettings(r);
            expect(d.navTitleBoost).toBe(0.25);
            expect(h.navTitleBoost).toBe(0.5);
            expect(o.navTitleBoost).toBe(0);
            expect(u.navTitleBoost).toBeUndefined(); // → DEFAULT_SETTINGS 0.25 in onload
            expect(d.settingsRev).toBe(13);
        }
    });
    it('is idempotent: a second load does not chain the old High 0.8 → 0.5 → 0.25', () => {
        const raw: Partial<SeekSettings> = { settingsRev: 11, navTitleBoost: 0.8 };
        migrateSettings(raw);
        migrateSettings(raw);
        expect(raw.navTitleBoost).toBe(0.5);
        expect(raw.settingsRev).toBe(13);
    });
    it('ships Default = 0.25', () => {
        expect(DEFAULT_SETTINGS.navTitleBoost).toBe(0.25);
    });
});

// build/proto-bm25f: rev 13 (fix/balanced-dense-weight) composes with the removal of
// the BM25F knob settings keys. Mirrors onload: migrate the raw data.json, then
// Object.assign it over DEFAULT_SETTINGS.
describe('migrateSettings — rev 13 + BM25F knob key removal', () => {
    it('a live rev-12 data.json gets the 0.85 repair and BM25F on', () => {
        const raw: Partial<SeekSettings> = { settingsRev: 12, denseWeight: 0.8, navTitleBoost: 0.25, recencyEpsilon: 0.04 };
        const s = Object.assign({}, DEFAULT_SETTINGS, migrateSettings(raw)) as SeekSettings;
        expect(s.settingsRev).toBe(13);
        expect(s.denseWeight).toBe(0.85);
        expect(s.bm25f).toBe(true);
    });

    it('persisted BM25F knob keys are dropped, idempotently, without touching the rev', () => {
        const raw = { settingsRev: 13, bm25fTitle: 10, bm25fAliases: 6, bm25fProperties: 2, bm25fBody: 1, bm25fK1: 3, denseWeight: 0.7 } as Partial<SeekSettings>;
        const s = Object.assign({}, DEFAULT_SETTINGS, migrateSettings(raw)) as unknown as Record<string, unknown>;
        for (const k of ['bm25fTitle', 'bm25fAliases', 'bm25fProperties', 'bm25fBody', 'bm25fK1']) expect(k in s).toBe(false);
        expect(s.denseWeight).toBe(0.7);
        expect(s.settingsRev).toBe(13);
        expect(migrateSettings(raw)).toEqual(raw);   // second pass: no-op
    });
});
