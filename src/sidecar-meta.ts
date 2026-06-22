// Per-device sidecar metadata + the version gate. A producer device writes its
// meta.<deviceId>.json on every full reindex / compaction; a consumer refuses to
// hydrate from any producer whose format, model, chunker version, or dimension
// differs — because a chunker mismatch means the consumer's local re-chunk won't
// reproduce the producer's ids, so it would silently load nothing and present as
// a relevance regression. Refuse loudly instead.

import type { DataAdapter } from 'obsidian';
import { metaPathFor, SIDECAR_FORMAT, writeTextAtomic } from './sidecar';
import { pluginIdentity, type IndexIdentity } from './identity';

export interface SidecarMeta {
    format: number; // SIDECAR_FORMAT at write time
    modelId: string; // embedding model id (ml97)
    revision: string | null; // pinned model commit sha (or null = track main) — F10 gate
    chunkerVersion: number; // CHUNKER_VERSION at write time
    dim: number; // embedding dimension (384)
    deviceId: string;
    lastFullReindex: string | null; // ISO timestamp, informational
    // Corpus dense-cosine background (dense-stats.ts), so a hydrate-only device
    // (iOS that never runs a full reindex) inherits the producer's display
    // calibration instead of showing none. Optional + ignored by the version gate
    // (metaAccepts) — a producer that predates these fields just yields no
    // confidence on the consumer until someone full-reindexes.
    bgMean?: number;
    bgStd?: number;
}

export interface MetaExpectation {
    modelId: string;
    revision: string | null;
    chunkerVersion: number;
    dim: number;
}

export async function readDeviceMeta(adapter: DataAdapter, indexDir: string, deviceId: string): Promise<SidecarMeta | null> {
    const path = metaPathFor(indexDir, deviceId);
    try {
        if (!(await adapter.exists(path).catch(() => false))) return null;
        const m = JSON.parse(await adapter.read(path)) as SidecarMeta;
        // Validate the fields hydrate actually depends on: a torn write that drops
        // modelId (the field metaAccepts gates on) or deviceId (which routes the
        // shard/jsonl reads) must refuse the whole device, not reach metaAccepts
        // with a malformed object.
        if (
            typeof m.format !== 'number' ||
            typeof m.chunkerVersion !== 'number' ||
            typeof m.dim !== 'number' ||
            typeof m.modelId !== 'string' ||
            typeof m.deviceId !== 'string'
        )
            return null;
        // revision is the F10 gate field; normalize a missing/legacy value to null
        // (= "track main") so metaAccepts compares cleanly. A format-2 producer
        // always writes it; this only smooths a hand-rolled or future-tolerant meta.
        m.revision = typeof m.revision === 'string' ? m.revision : null;
        return m;
    } catch {
        return null;
    }
}

export async function writeDeviceMeta(adapter: DataAdapter, indexDir: string, meta: SidecarMeta): Promise<void> {
    await writeTextAtomic(adapter, metaPathFor(indexDir, meta.deviceId), JSON.stringify(meta, null, 2));
}

// A null/missing meta is a refusal — compatibility can't be proven without it.
export function metaAccepts(meta: SidecarMeta | null, expect: MetaExpectation): boolean {
    if (!meta) return false;
    return (
        meta.format === SIDECAR_FORMAT &&
        meta.modelId === expect.modelId &&
        meta.revision === expect.revision &&
        meta.chunkerVersion === expect.chunkerVersion &&
        meta.dim === expect.dim
    );
}

// The cross-device slice of the build identity, in the exact shape metaAccepts
// compares. Centralizes the field list so the producer write + every hydrate gate
// stop hand-repeating { modelId, revision, chunkerVersion, dim } — one identity
// source (identity.ts), no drift between call sites. The slice deliberately omits
// analyzerVersion + dbVersion: two devices can differ on those and still exchange
// reproducible chunk_ids + comparable vectors, so the sidecar hydrates across them
// (see identity.ts for the full rationale).
export function expectationFor(id: IndexIdentity = pluginIdentity()): MetaExpectation {
    return { modelId: id.modelId, revision: id.revision, chunkerVersion: id.chunkerVersion, dim: id.dim };
}
