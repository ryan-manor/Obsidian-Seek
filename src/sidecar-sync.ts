// Sidecar hydrate/reconcile: rebuild the IndexedDB index from the vault-file
// sidecar WITHOUT re-embedding. This is what restores search after iOS evicts
// the IDB, and what lets a fresh device pick up another device's index.
//
// The algorithm is a pure function over injectable dependencies (HydrateDeps) so
// it can be unit-tested against fakes with no IndexedDB / vault. SearchOrchestrator
// supplies the real deps (re-chunk the live vault, read/write the store) and runs
// it under its write mutex; main.ts schedules it (load + idle + command).
//
// Liveness oracle: a chunk_id is a deterministic path-salted content hash, so the
// consumer reproduces the producer's ids by re-chunking its own copy of the vault
// and keeping only the intersection. A deleted note's ids match nothing and are
// ignored — cross-device deletes need no protocol.
//
// All-or-nothing per note: a note is hydrated only when EVERY one of its live
// chunks is available in the sidecar (present + bytes synced). A partially-synced
// note is left for a model-backed embed (computeDelta will flag it) rather than
// writing a file record that would make computeDelta think it's fully indexed.
//
// chunk_id reproduction (FIXED 2026-06-14): the oracle (reChunkLive /
// dedupViaSidecar in search.ts) runs the FULL producer pipeline — chunkContent
// THEN enforceTokenBudget — so split chunk_ids match exactly. The earlier v1
// omitted the token-budget re-split (it needs the tokenizer), which silently
// missed every note long enough to be split: on a long-note vault that was
// ~the whole corpus → hydrated:0 → full on-device re-embed → iPhone jetsam.
// The hydrate now loads the TOKENIZER ONLY (embedder.ensureTokenizer, a few MB,
// no ~250 MB model) to stay mobile-safe while reproducing ids.

import type { DataAdapter } from 'obsidian';
import type { Chunk } from './types';
import type { QuantVec } from './quant';
import {
    decodeRecord,
    deviceIdFromJsonlPath,
    isOffsetInRange,
    listDeviceJsonls,
    scanJsonl,
    shardPathFor,
    type ResolvedEntry,
} from './sidecar';
import { metaAccepts, readDeviceMeta, type MetaExpectation, type SidecarMeta } from './sidecar-meta';

// One live note re-chunked locally: the liveness oracle's unit.
export interface ReChunkedNote {
    notePath: string;
    mtimeMs: number;
    chunks: Chunk[]; // each carries chunk_id
    contentHash?: string; // cyrb53 of the raw bytes — persisted so computeDelta can skip mtime-only re-stamps
}

export interface HydrateDeps {
    adapter: DataAdapter;
    indexDir: string;
    expect: MetaExpectation; // {modelId, chunkerVersion, dim} this consumer can reproduce
    reChunk: () => Promise<ReChunkedNote[]>; // live vault → re-chunked notes
    existingIds: () => Promise<Set<string>>; // chunk_ids already in IDB (skip — idempotent)
    putQuantized: (chunks: Chunk[], tiers: { q: QuantVec; bin: Uint8Array }[]) => Promise<void>;
    putFileRecord: (rec: { note_path: string; mtimeMs: number; chunk_ids: string[]; contentHash?: string }) => Promise<void>;
    // Version-gate refusal. Carries the producer meta (null = missing/unreadable)
    // and this consumer's expectation so the consumer can log WHAT is stale
    // (e.g. "chunker v3≠v4") and throttle to once per device+reason — a chunker
    // bump otherwise refuses the same producer on every reconcile / delta flush.
    onRefusedProducer?: (deviceId: string, meta: SidecarMeta | null, expect: MetaExpectation) => void;
    log?: (msg: string, detail: unknown) => void;
    batchSize?: number; // putQuantized batch size (default 500)
}

export interface HydrateResult {
    scanned: number; // sidecar live records across accepted producers
    needed: number; // live ∩ sidecar, minus already-in-IDB (chunks)
    hydrated: number; // chunks actually written
    skippedPartialNotes: number; // notes skipped because a chunk's bytes weren't synced yet
    refusedProducers: number; // producers excluded by the version gate
    acceptedProducers: number;
    hydratedNotePaths: string[]; // notes fully hydrated (a file record was written) — drives delta dedup
    // Corpus dense-cosine background inherited from the freshest accepted producer
    // (newest lastFullReindex) — the orchestrator writes it into local meta so a
    // hydrate-only device gets display calibration. Undefined if no producer
    // carried stats.
    bgMean?: number;
    bgStd?: number;
}

// Epoch (ms) of a producer's last full reindex, for "freshest" selection. A
// null / absent / unparseable timestamp yields -Infinity so it sorts below every
// real one — the choice of which producer's calibration to inherit is then a
// well-defined max, never dependent on file-scan order.
function fullReindexEpoch(m: SidecarMeta | null): number {
    const t = m?.lastFullReindex ? Date.parse(m.lastFullReindex) : NaN;
    return Number.isNaN(t) ? -Infinity : t;
}

// Rank the version-COMPATIBLE producers freshest-first (newest full reindex). Used
// by the cross-device BM25 load (Phase 3): the consumer tries each in turn until one
// has a loadable BM25 .gz artifact — a single "freshest" pick is NOT enough because a
// producer can be the freshest reindexer yet have NO gz (mobile writes meta+jsonl but
// not the artifact — emit is desktop-only), or a torn/missing/corrupt gz. metaAccepts
// is the same gate hydrateFromSidecar uses, so a producer whose ids this device can't
// reproduce (chunker/model/dim/format mismatch) is never trusted. SELF is eligible:
// after an iOS eviction this device's OWN sidecar (which survives in the vault) is a
// valid source. Freshest-by-epoch is well-defined regardless of file-scan order (a
// null/absent timestamp sorts below every real one). Empty when the sidecar is empty
// / all refused.
export async function rankAcceptedProducers(
    adapter: DataAdapter,
    indexDir: string,
    expect: MetaExpectation,
): Promise<string[]> {
    const jsonls = await listDeviceJsonls(adapter, indexDir);
    const accepted: { dev: string; epoch: number }[] = [];
    for (const jsonl of jsonls) {
        const dev = deviceIdFromJsonlPath(jsonl);
        if (!dev) continue;
        const meta = await readDeviceMeta(adapter, indexDir, dev);
        if (!metaAccepts(meta, expect)) continue;
        accepted.push({ dev, epoch: fullReindexEpoch(meta) });
    }
    // Freshest first; listDeviceJsonls already sorted, so equal epochs stay stable.
    accepted.sort((a, b) => b.epoch - a.epoch);
    return accepted.map(a => a.dev);
}

export async function hydrateFromSidecar(deps: HydrateDeps): Promise<HydrateResult> {
    const { adapter, indexDir, expect } = deps;
    const empty: HydrateResult = { scanned: 0, needed: 0, hydrated: 0, skippedPartialNotes: 0, refusedProducers: 0, acceptedProducers: 0, hydratedNotePaths: [] };

    // 1. Version gate: keep only producers this consumer can reproduce.
    const allJsonls = await listDeviceJsonls(adapter, indexDir);
    // Early producer-file probe: producerFilesFound:0 means NO other device's
    // sidecar has reached this device's index dir yet (the iCloud-delivery gap
    // that strands an iPhone into a local re-embed). Logged before the version
    // gate so "0 files synced" is distinguishable from "files present but refused".
    deps.log?.('sidecar-hydrate-scan', {
        producerFilesFound: allJsonls.length,
        devices: allJsonls.map(j => deviceIdFromJsonlPath(j) ?? '?').join(',') || 'none',
    });
    const accepted: string[] = [];
    let refused = 0;
    // Inherit display-calibration stats from the freshest accepted producer (the
    // one whose last full reindex is newest = most representative of the corpus).
    // Compare by parsed epoch, NOT lexicographically: a missing/malformed/null
    // timestamp sorts below all real ones (a stats-bearing producer with a real
    // timestamp always wins, never just jsonl-iteration order), and mixed-width
    // ISO strings (with/without millis) can't compare backwards.
    let bgSource: SidecarMeta | null = null;
    for (const jsonl of allJsonls) {
        const dev = deviceIdFromJsonlPath(jsonl);
        if (!dev) continue;
        const meta = await readDeviceMeta(adapter, indexDir, dev);
        if (metaAccepts(meta, expect)) {
            accepted.push(jsonl);
            if (meta && meta.bgMean != null && meta.bgStd != null &&
                (bgSource == null || fullReindexEpoch(meta) > fullReindexEpoch(bgSource))) {
                bgSource = meta;
            }
        } else {
            refused++;
            deps.onRefusedProducer?.(dev, meta, expect);
        }
    }
    const bg = { bgMean: bgSource?.bgMean, bgStd: bgSource?.bgStd };
    if (accepted.length === 0) {
        deps.log?.('sidecar-hydrate', { ...empty, refusedProducers: refused });
        return { ...empty, refusedProducers: refused };
    }

    // 2. Scan accepted producers → resolved id → location map.
    const scan = await scanJsonl(adapter, accepted);

    // 3. Cheap pre-gate — skip the whole-vault re-chunk when the scan surfaced no
    //    sidecar id this device is missing. chunk_id is a content hash, so an id
    //    already in IDB already holds the exact right vector (same id ⇒ same
    //    content ⇒ same deterministic embedding) — nothing to hydrate for it. And
    //    step 5 only ever admits a candidate note that has a chunk in scan.map AND
    //    NOT in `existing`, so "no id is fresh to us" PROVES "zero candidates": the
    //    re-chunk would do whole-vault work (read + tokenize every note, on the
    //    mobile main thread) to hydrate nothing. This collapses the common no-op
    //    syncs — a peer re-embedding SHARED notes, a peer compacting its sidecar,
    //    an mtime-churn re-append — all of which flip sidecarDirSignature yet add
    //    no id new to THIS device, from a full re-chunk into a scan + a membership
    //    test. `existing` is fetched here (before reChunk) so the gate runs first;
    //    the candidate loop reuses it. NOTE: the empty-store eviction case never
    //    reaches here as a no-op — an empty IDB has nothing in `existing`, so every
    //    synced id is fresh and the full re-chunk runs (mandatory recovery).
    const existing = await deps.existingIds();
    let hasFreshId = false;
    for (const id of scan.map.keys()) {
        if (!existing.has(id)) { hasFreshId = true; break; }
    }
    if (!hasFreshId) {
        const r: HydrateResult = { ...empty, ...bg, scanned: scan.map.size, refusedProducers: refused, acceptedProducers: accepted.length };
        deps.log?.('sidecar-hydrate-skip-rechunk', { scanned: scan.map.size, reason: 'no-fresh-ids' });
        deps.log?.('sidecar-hydrate', r);
        return r;
    }

    // 4. Re-chunk the live vault (the liveness oracle). `existing` already fetched above.
    const live = await deps.reChunk();

    // 5. Select candidate notes: not already fully in IDB, and every live chunk
    //    has a resolved sidecar entry. Collect their entries for grouped reads.
    interface Candidate {
        note: ReChunkedNote;
        entries: ResolvedEntry[]; // aligned with note.chunks
    }
    const candidates: Candidate[] = [];
    let needed = 0;
    for (const note of live) {
        if (note.chunks.length === 0) continue;
        if (note.chunks.every(c => existing.has(c.chunk_id))) continue; // already indexed
        const entries: ResolvedEntry[] = [];
        let coverable = true;
        for (const c of note.chunks) {
            const e = scan.map.get(c.chunk_id);
            if (!e) {
                coverable = false; // not in the sidecar — leave to a model-backed embed
                break;
            }
            entries.push(e);
        }
        if (!coverable) continue;
        candidates.push({ note, entries });
        needed += entries.length;
    }
    if (candidates.length === 0) {
        const r: HydrateResult = { ...empty, ...bg, scanned: scan.map.size, refusedProducers: refused, acceptedProducers: accepted.length };
        deps.log?.('sidecar-hydrate', r);
        return r;
    }

    // 6. Group every needed entry by (shard, seq) so each bin is read ONCE.
    const byShard = new Map<string, ResolvedEntry[]>();
    for (const cand of candidates) {
        for (const e of cand.entries) {
            const key = `${e.shard}.${e.seq}`;
            const arr = byShard.get(key);
            if (arr) arr.push(e);
            else byShard.set(key, [e]);
        }
    }

    // 7. Read + validate + decode into a per-entry tier cache. Out-of-range or
    //    missing-bin entries decode to null (partial arrival) — their notes are
    //    skipped in step 8.
    const tierCache = new Map<string, ReturnType<typeof decodeRecord> | null>();
    const entryKey = (e: ResolvedEntry): string => `${e.shard}.${e.seq}.${e.off}`;
    for (const [, entries] of byShard) {
        const first = entries[0];
        const buf = await adapter.readBinary(shardPathFor(indexDir, first.shard, first.seq)).catch(() => null);
        for (const e of entries) {
            if (buf && isOffsetInRange(buf.byteLength, e.off)) {
                // dim mismatch or CRC failure (corrupt record) → null, skip the note.
                try { tierCache.set(entryKey(e), decodeRecord(buf, e.off, e.dim)); }
                catch { tierCache.set(entryKey(e), null); }
            } else tierCache.set(entryKey(e), null);
        }
    }

    // 8. Write each fully-covered note (all-or-nothing) in batches.
    const batchSize = deps.batchSize ?? 500;
    let pendingChunks: Chunk[] = [];
    let pendingTiers: { q: QuantVec; bin: Uint8Array }[] = [];
    const flush = async (): Promise<void> => {
        if (pendingChunks.length === 0) return;
        await deps.putQuantized(pendingChunks, pendingTiers);
        pendingChunks = [];
        pendingTiers = [];
    };
    let hydrated = 0;
    let skippedPartialNotes = 0;
    const fileRecords: Array<{ note_path: string; mtimeMs: number; chunk_ids: string[]; contentHash?: string }> = [];
    for (const cand of candidates) {
        const tiers = cand.entries.map(e => tierCache.get(entryKey(e)) ?? null);
        if (tiers.some(t => t === null)) {
            skippedPartialNotes++; // a chunk's bytes haven't synced yet — embed later, no file record
            continue;
        }
        for (let i = 0; i < cand.note.chunks.length; i++) {
            const t = tiers[i]!;
            pendingChunks.push(cand.note.chunks[i]);
            pendingTiers.push({ q: { q: t.q, s: t.s }, bin: t.sign });
            hydrated++;
            if (pendingChunks.length >= batchSize) await flush();
        }
        fileRecords.push({ note_path: cand.note.notePath, mtimeMs: cand.note.mtimeMs, chunk_ids: cand.note.chunks.map(c => c.chunk_id), contentHash: cand.note.contentHash });
    }
    await flush();
    for (const rec of fileRecords) await deps.putFileRecord(rec);

    const result: HydrateResult = {
        ...bg,
        scanned: scan.map.size,
        needed,
        hydrated,
        skippedPartialNotes,
        refusedProducers: refused,
        acceptedProducers: accepted.length,
        hydratedNotePaths: fileRecords.map(r => r.note_path),
    };
    deps.log?.('sidecar-hydrate', result);
    return result;
}
