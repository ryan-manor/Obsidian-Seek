// The single source of truth for the index's *version identity* — the build
// fingerprint every "is this index still valid?" gate derives from. Shipping a
// new version is then one constant bump in one home module; no gate edits, no
// per-version migration code.
//
// Two gates consume this, and they compare DIFFERENT slices on purpose:
//
//   • Local IDB index  → identityMatches() below. The fields whose change makes
//     the stored *dense vectors* / chunk_ids unusable, so a mismatch means
//     "re-embed or re-hydrate this device".
//   • Cross-device sidecar → metaAccepts() in sidecar-meta.ts (expectation built
//     by expectationFor()). A narrower slice: a producer on a different dbVersion
//     or analyzerVersion can still hand this device reproducible chunk_ids +
//     comparable vectors, so the sidecar must hydrate across those.
//
// Conflating the two would wrongly refuse valid cross-device hydration on a
// dbVersion- or analyzer-only bump, so the SLICES stay distinct — only the
// SOURCE of the constants (pluginIdentity) is shared.

import { DB_VERSION, type MetaConfig } from './index-store';
import { CHUNKER_VERSION } from './chunker';
import { ANALYZER_VERSION } from './bm25';
import { SIDECAR_FORMAT } from './sidecar';
import { MODEL_ID, MODEL_REVISION, EMBEDDING_DIM, LEGACY_ENGLISH_MODEL_ID } from './embedder';

export interface IndexIdentity {
    dbVersion: number; // IndexedDB schema (index-store.ts) — enforced natively by IDB's onupgradeneeded
    chunkerVersion: number; // chunk content+suffix derivation (chunker.ts) — feeds both embed bytes AND chunk_id
    analyzerVersion: string; // BM25 token space / scoring (bm25.ts) — lexical-only, esbuild-injected ('dev' under vitest)
    sidecarFormat: number; // cross-device file protocol (sidecar.ts)
    modelId: string; // embedding model repo (embedder.ts)
    revision: string | null; // pinned model commit sha, or null = track main (embedder.ts)
    dim: number; // embedding dimension (embedder.ts)
}

// Read lazily (only when CALLED, never at module load) so the Phase-2
// index-store ↔ identity import cycle stays TDZ-safe: every constant is resolved
// by the time any runtime code invokes this.
export function pluginIdentity(): IndexIdentity {
    return {
        dbVersion: DB_VERSION,
        chunkerVersion: CHUNKER_VERSION,
        analyzerVersion: ANALYZER_VERSION,
        sidecarFormat: SIDECAR_FORMAT,
        modelId: MODEL_ID,
        revision: MODEL_REVISION,
        dim: EMBEDDING_DIM,
    };
}

// The LOCAL dense-index gate. Compares only the fields whose change invalidates
// the stored vectors / chunk_id space — i.e. the fields for which the only honest
// recovery is to re-embed (desktop) or re-hydrate (mobile):
//
//   chunkerVersion — chunk content+suffix feed the embed bytes and the chunk_id
//   modelId, revision — different weights ⇒ vectors are not comparable
//   dim — defensive; guards a model/dim misconfig (see main.ts EMBEDDING_DIM check)
//
// Deliberately EXCLUDED (a mismatch here is NOT a re-embed trigger):
//
//   analyzerVersion — lexical only. The BM25 blob stamp (search.ts) refits from
//     the intact chunk bodies on a mismatch, so an analyzer bump needs no nuke.
//   dbVersion — IndexedDB enforces it structurally: opening at a new DB_VERSION
//     fires onupgradeneeded, which already empties the stores.
//   sidecarFormat — governs the cross-device file protocol (metaAccepts), not the
//     validity of the local index.
export function identityMatches(a: IndexIdentity, b: IndexIdentity): boolean {
    return (
        a.chunkerVersion === b.chunkerVersion &&
        a.modelId === b.modelId &&
        a.revision === b.revision &&
        a.dim === b.dim
    );
}

// The identity a STORED index reports, for comparison against pluginIdentity() via
// identityMatches. Legacy / torn metas (written before the identity fields existed)
// map every missing field to a value that GUARANTEES a mismatch — an unstamped index
// must be treated as stale and rebuilt, never silently accepted:
//   chunkerVersion ?? -1   — no real CHUNKER_VERSION is negative
//   modelId ?? LEGACY      — a pre-2026-06-10 unstamped index can only be english-r2
//   revision ?? null       — a pinned-sha build won't match null
// (analyzerVersion / dbVersion / sidecarFormat are carried for completeness but are
// not compared by identityMatches.)
export function identityFromMeta(meta: MetaConfig): IndexIdentity {
    return {
        dbVersion: meta.schemaVersion,
        chunkerVersion: meta.chunkerVersion ?? -1,
        analyzerVersion: meta.analyzerVersion ?? '',
        sidecarFormat: SIDECAR_FORMAT,
        modelId: meta.modelId ?? LEGACY_ENGLISH_MODEL_ID,
        revision: meta.revision ?? null,
        dim: meta.embeddingDim,
    };
}

// Should an index-write pass STAMP the live build identity into its meta (and refit
// the dense-cosine background), versus carry the prior meta's fields forward unchanged?
//
//   • A FULL reindex always stamps — it rebuilt the whole index from the live build.
//   • An INCREMENTAL pass normally preserves prevMeta, so a few changed files can't
//     re-stamp a stale index as current — EXCEPT the cold first-build, where the store
//     was empty when the pass began. That pass IS the current build (every chunk it
//     writes is freshly embedded by this build) AND, as one whole-vault incremental,
//     it saw the entire corpus — so it claims the live identity + computes the
//     background, exactly like a full reindex.
//
// The emptiness gate is load-bearing and deliberately strict: a NON-empty store that
// happens to carry no identity is a LEGACY index (older chunker/model, written before
// the identity fields existed). It must stay "stale" so the version gate rebuilds it —
// stamping the live identity onto it would falsely mark old-chunker/old-model vectors
// as current. So the trigger is "store was empty", never "prevMeta lacks identity".
//
// This is the fix for the spurious-heal loop: a cold build that copied the empty meta's
// absent identity wrote `undefined`, which identityFromMeta maps to a guaranteed
// mismatch (-1 / null), so the gate re-launched a full reindex forever.
export function shouldStampLiveIdentity(mode: 'full' | 'incremental', storeWasEmpty: boolean): boolean {
    return mode === 'full' || storeWasEmpty;
}

// Is a version-mismatched index eligible for the cheap embed-free heal (stamp the
// live identity in place), or is it GENUINELY old and must take the full reindex /
// wait path? Pure gate for reconcileIdentityInPlace (search.ts) — the content proof
// (computeDelta) and the stamp I/O live there; this decides only whether stamping is
// SAFE at all, from the stored meta alone.
//
//   'stale'    — meta carries a PRESENT chunkerVersion (a real, stamped identity that
//                the gate found differing → old chunker, or model/dim bumped on a
//                stamped index), OR the stored modelId/dim ≠ the live build. The
//                modelId/dim check is the CROSS-MODEL GUARD: an unstamped index whose
//                vectors are from a different model (e.g. legacy english-r2, or an
//                absent modelId) must never be stamped current — query-vs-doc vectors
//                would be cross-model garbage, the one outcome that is WRONG not stale.
//   'eligible' — merely UNSTAMPED and from the current model+dim. The mismatch that
//                triggered the heal is, by construction, only the absent
//                chunkerVersion/revision; those can't be verified without re-chunking
//                (the embed/tokenizer cost we avoid on mobile), so they are assumed
//                current — worst case stale chunk BOUNDARIES (same text, same model →
//                stale, never wrong). The caller then proves the files are unchanged.
export function identityHealEligibility(
    meta: Pick<MetaConfig, 'chunkerVersion' | 'modelId' | 'embeddingDim'>,
    live: { modelId: string; dim: number } = { modelId: MODEL_ID, dim: EMBEDDING_DIM },
): 'stale' | 'eligible' {
    if (meta.chunkerVersion !== undefined) return 'stale';                 // present, differing → genuinely old
    if (meta.modelId !== live.modelId || meta.embeddingDim !== live.dim) return 'stale'; // cross-model guard
    return 'eligible';
}
