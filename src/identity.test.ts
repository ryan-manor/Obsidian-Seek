// Identity foundation (Phase 1): pluginIdentity() faithfully gathers the compiled
// version constants, identityMatches() gates the LOCAL dense index on exactly the
// re-embed-invalidating fields (and ignores the ones handled elsewhere), and
// expectationFor() slices the narrower cross-device sidecar expectation.

import { describe, it, expect } from 'vitest';
import { pluginIdentity, identityMatches, identityFromMeta, shouldStampLiveIdentity, identityHealEligibility, type IndexIdentity } from './identity';
import type { MetaConfig } from './index-store';
import { DB_VERSION } from './index-store';
import { LEGACY_ENGLISH_MODEL_ID } from './embedder';
import { CHUNKER_VERSION } from './chunker';
import { ANALYZER_VERSION } from './bm25';
import { SIDECAR_FORMAT } from './sidecar';
import { MODEL_ID, MODEL_REVISION, EMBEDDING_DIM } from './embedder';
import { expectationFor, metaAccepts, type SidecarMeta } from './sidecar-meta';

describe('pluginIdentity', () => {
    it('gathers every compiled version constant verbatim', () => {
        expect(pluginIdentity()).toEqual({
            dbVersion: DB_VERSION,
            chunkerVersion: CHUNKER_VERSION,
            analyzerVersion: ANALYZER_VERSION,
            sidecarFormat: SIDECAR_FORMAT,
            modelId: MODEL_ID,
            revision: MODEL_REVISION,
            dim: EMBEDDING_DIM,
        });
    });
});

describe('identityMatches (local dense-index gate)', () => {
    const base = (): IndexIdentity => ({
        dbVersion: 11,
        chunkerVersion: 5,
        analyzerVersion: 'a1',
        sidecarFormat: 2,
        modelId: 'repo/model',
        revision: 'sha1',
        dim: 384,
    });

    it('accepts two identical identities', () => {
        expect(identityMatches(base(), base())).toBe(true);
    });

    it('rejects a change to any dense-invalidating field (forces re-embed/re-hydrate)', () => {
        const mutations: Array<(i: IndexIdentity) => void> = [
            i => { i.chunkerVersion = 6; },
            i => { i.modelId = 'repo/other'; },
            i => { i.revision = 'sha2'; },
            i => { i.revision = null; }, // pinned → track-main is a real model change
            i => { i.dim = 512; },
        ];
        for (const mutate of mutations) {
            const b = base();
            mutate(b);
            expect(identityMatches(base(), b)).toBe(false);
        }
    });

    it('IGNORES analyzerVersion / dbVersion / sidecarFormat — none is a re-embed trigger', () => {
        const mutations: Array<(i: IndexIdentity) => void> = [
            i => { i.analyzerVersion = 'a2'; }, // → BM25 refit from intact bodies, not a nuke
            i => { i.dbVersion = 12; }, // → IndexedDB onupgradeneeded handles it structurally
            i => { i.sidecarFormat = 3; }, // → cross-device gate (metaAccepts) only
        ];
        for (const mutate of mutations) {
            const b = base();
            mutate(b);
            expect(identityMatches(base(), b)).toBe(true);
        }
    });
});

describe('expectationFor (cross-device sidecar slice)', () => {
    it('slices the four cross-device fields from an identity', () => {
        const id = pluginIdentity();
        expect(expectationFor(id)).toEqual({
            modelId: id.modelId,
            revision: id.revision,
            chunkerVersion: id.chunkerVersion,
            dim: id.dim,
        });
    });

    it('defaults to the live plugin identity', () => {
        expect(expectationFor()).toEqual(expectationFor(pluginIdentity()));
    });

    it('accepts a producer meta written at the live identity', () => {
        const id = pluginIdentity();
        const meta: SidecarMeta = {
            format: id.sidecarFormat,
            modelId: id.modelId,
            revision: id.revision,
            chunkerVersion: id.chunkerVersion,
            dim: id.dim,
            deviceId: 'devA',
            lastFullReindex: null,
        };
        expect(metaAccepts(meta, expectationFor(id))).toBe(true);
    });

    it('refuses a producer one chunker version behind', () => {
        const id = pluginIdentity();
        const meta: SidecarMeta = {
            format: id.sidecarFormat,
            modelId: id.modelId,
            revision: id.revision,
            chunkerVersion: id.chunkerVersion - 1,
            dim: id.dim,
            deviceId: 'devB',
            lastFullReindex: null,
        };
        expect(metaAccepts(meta, expectationFor(id))).toBe(false);
    });
});

describe('identityFromMeta (stored index → identity, for the local gate)', () => {
    const cur = pluginIdentity();

    it('a meta stamped at the current build matches pluginIdentity', () => {
        const meta = {
            embeddingDim: cur.dim,
            lastIndexedAt: '2026-06-20T00:00:00Z',
            schemaVersion: DB_VERSION,
            modelId: cur.modelId,
            chunkerVersion: cur.chunkerVersion,
            analyzerVersion: cur.analyzerVersion,
            revision: cur.revision,
        } as MetaConfig;
        expect(identityMatches(identityFromMeta(meta), cur)).toBe(true);
    });

    it('a legacy meta (no identity fields) is forced to a mismatch → rebuild', () => {
        const meta = {
            embeddingDim: cur.dim,
            lastIndexedAt: '2026-06-01T00:00:00Z',
            schemaVersion: DB_VERSION,
            // no modelId / chunkerVersion / analyzerVersion / revision
        } as MetaConfig;
        const stored = identityFromMeta(meta);
        expect(stored.chunkerVersion).toBe(-1); // sentinel: no real version is negative
        expect(stored.modelId).toBe(LEGACY_ENGLISH_MODEL_ID);
        expect(stored.revision).toBe(null);
        expect(identityMatches(stored, cur)).toBe(false);
    });

    it('an index one chunker version behind mismatches (the orphan-pileup case)', () => {
        const meta = {
            embeddingDim: cur.dim,
            lastIndexedAt: '2026-06-19T00:00:00Z',
            schemaVersion: DB_VERSION,
            modelId: cur.modelId,
            chunkerVersion: cur.chunkerVersion - 1,
            analyzerVersion: cur.analyzerVersion,
            revision: cur.revision,
        } as MetaConfig;
        expect(identityMatches(identityFromMeta(meta), cur)).toBe(false);
    });

    it('a same-repo revision change mismatches even though modelId is unchanged', () => {
        const meta = {
            embeddingDim: cur.dim,
            lastIndexedAt: '2026-06-19T00:00:00Z',
            schemaVersion: DB_VERSION,
            modelId: cur.modelId,
            chunkerVersion: cur.chunkerVersion,
            analyzerVersion: cur.analyzerVersion,
            revision: (cur.revision ?? '') + '-different',
        } as MetaConfig;
        expect(identityMatches(identityFromMeta(meta), cur)).toBe(false);
    });
});

// The exact gap CI missed: a fresh large vault builds its first index through the
// INCREMENTAL catch-up path (reconcileOnLoad → drainCatchUp), not a `full` reindex.
// The pre-fix stamp condition (`mode === 'full'`) therefore left a populated index with
// NO identity, which identityFromMeta maps to a guaranteed mismatch → the version gate
// re-launched a full reindex on every poll (the spurious-heal loop).
describe('shouldStampLiveIdentity (cold-build stamp gate)', () => {
    it('a full reindex always claims the live identity, empty or not', () => {
        expect(shouldStampLiveIdentity('full', true)).toBe(true);
        expect(shouldStampLiveIdentity('full', false)).toBe(true);
    });

    it('a cold first-build (incremental on an EMPTY store) claims it — the fix', () => {
        expect(shouldStampLiveIdentity('incremental', true)).toBe(true);
    });

    it('an ordinary delta (incremental on a NON-empty store) carries prevMeta forward', () => {
        // Legacy-index safety: a non-empty store with no identity is an older-version
        // index that MUST stay stale so the gate rebuilds it — never self-stamp current.
        expect(shouldStampLiveIdentity('incremental', false)).toBe(false);
    });

    it('END-TO-END: a stamped cold build satisfies the gate; the buggy one loops it', () => {
        const cur = pluginIdentity();
        // Post-fix: the cold incremental build writes the live identity into meta, so
        // the very next identity poll is satisfied → no heal.
        const stamped = {
            embeddingDim: cur.dim,
            lastIndexedAt: '2026-06-21T00:00:00Z',
            schemaVersion: DB_VERSION,
            modelId: cur.modelId,
            chunkerVersion: cur.chunkerVersion,
            analyzerVersion: cur.analyzerVersion,
            revision: cur.revision,
        } as MetaConfig;
        expect(identityMatches(identityFromMeta(stamped), cur)).toBe(true);

        // Pre-fix: the cold build copied an empty store's absent identity → undefined,
        // which the gate reads as stale forever. This is the loop the fix removes.
        const buggy = {
            embeddingDim: cur.dim,
            lastIndexedAt: '2026-06-21T00:00:00Z',
            schemaVersion: DB_VERSION,
            // chunkerVersion / revision / modelId written as `undefined` by the bug
        } as MetaConfig;
        expect(identityMatches(identityFromMeta(buggy), cur)).toBe(false);
    });
});

// The embed-free in-place heal gate (reconcileIdentityInPlace's arm selection): which
// version-mismatched indexes are SAFE to stamp current (chunks already byte-identical
// to a reindex) vs which are genuinely old and must reindex. The dangerous mistake the
// gate must never make is stamping a CROSS-MODEL index current (wrong dense scores).
describe('identityHealEligibility (in-place heal gate)', () => {
    const cur = pluginIdentity();
    const base = {
        embeddingDim: cur.dim,
        lastIndexedAt: '2026-06-21T00:00:00Z',
        schemaVersion: DB_VERSION,
    };

    it('UNSTAMPED but current model+dim (the cold-build bug artifact) → eligible', () => {
        // The exact BEIR shape: modelId stamped, chunkerVersion/revision absent.
        const beir = { ...base, modelId: cur.modelId } as MetaConfig;
        expect(identityHealEligibility(beir)).toBe('eligible');
    });

    it('PRESENT chunkerVersion (a real stamped identity the gate found differing) → stale', () => {
        // Old chunker, current model — must reindex, never stamp (stale chunk space).
        const oldChunker = { ...base, modelId: cur.modelId, chunkerVersion: cur.chunkerVersion - 1 } as MetaConfig;
        expect(identityHealEligibility(oldChunker)).toBe('stale');
        // Even the CURRENT chunkerVersion present routes to 'stale' here: we only reach
        // this gate on a mismatch, so a present-and-current chunker means some OTHER field
        // (model/revision/dim) differs → genuinely needs a reindex.
        const stampedButBumped = { ...base, modelId: cur.modelId, chunkerVersion: cur.chunkerVersion } as MetaConfig;
        expect(identityHealEligibility(stampedButBumped)).toBe('stale');
    });

    it('CROSS-MODEL guard: unstamped but a different / absent modelId → stale (never stamp old vectors current)', () => {
        const wrongModel = { ...base, modelId: LEGACY_ENGLISH_MODEL_ID } as MetaConfig;
        expect(identityHealEligibility(wrongModel)).toBe('stale');
        // Pre-2026-06-10 index: no modelId at all (undefined) → not the current model → stale.
        const noModel = { ...base } as MetaConfig;
        expect(identityHealEligibility(noModel)).toBe('stale');
    });

    it('DIM guard: unstamped, current model, but a different embedding dim → stale', () => {
        const wrongDim = { ...base, embeddingDim: cur.dim + 128, modelId: cur.modelId } as MetaConfig;
        expect(identityHealEligibility(wrongDim)).toBe('stale');
    });

    it('honours an injected live identity (so the gate tracks future model/dim ships)', () => {
        const idx = { ...base, modelId: 'next-gen-model' } as MetaConfig;
        expect(identityHealEligibility(idx, { modelId: 'next-gen-model', dim: cur.dim })).toBe('eligible');
        expect(identityHealEligibility(idx, { modelId: cur.modelId, dim: cur.dim })).toBe('stale');
    });
});
