// Sidecar: a vault-file representation of the vector index that sync providers
// (iCloud / Obsidian Sync) carry between devices and that survives WebKit's
// eviction of IndexedDB on iOS. Lifted from the validated spike at
// github.com/tooape/embeddinggemmaiostest (baseline commit c7198ce) and adapted
// for production Seek:
//   - fixed record stride [q:int8×dim | s:f64 | signbits:ceil(dim/8) | crc] — the
//     persisted DB v6 tiers verbatim, NOT fp32 (hydration is a byte copy). The
//     scale is stored at the SAME float64 width IndexedDB's `embeddings` store
//     keeps it at (quantizeInt8's raw `maxAbs / 127`, never truncated) — a
//     hydrated peer must dequantize with the bit-identical scale the producer
//     used, or the two devices' cosine scores diverge on a near-tie. The stride
//     is derived from the active model's dim (see Q_BYTES below), so a model
//     swap re-sizes it automatically; metaAccepts gates cross-dim hydration.
//   - per-device `index.<deviceId>.jsonl` — every file has exactly one writer,
//     readers union all devices' jsonls. Sync providers never merge, so a shared
//     jsonl would lose writes / spawn conflict copies (the shared-NDJSON-log
//     lesson). Per-device files make sync conflicts structurally impossible.
//   - no `.compacting` cross-device sentinel — compaction touches only the
//     owning device's own files, so the in-process dir mutex is sufficient.
//   - tombstones suppress only SAME-device records — liveness is ultimately
//     decided by the consumer's local re-chunk intersection, so device A's
//     delete must never erase device B's live copy of the same content-hash id.
//   - 4 MB shard cap (Obsidian Sync's default per-file limit is 5 MB).
//
// This module is a file-format library: it knows nothing about IndexStore or
// chunking. Its ONE model dependency is the embedding dimension
// (ACTIVE_MODEL_SPEC.dim), which sets the record stride. SidecarSync
// (sidecar-sync.ts) is the orchestration.

import type { DataAdapter } from 'obsidian';
import { ACTIVE_MODEL_SPEC } from './model-registry';

// ---- record layout (matches IndexStore DB v6 persisted tiers) ----

// Q_BYTES / SIGN_BYTES are DERIVED from the active model's embedding dimension
// (the single source — model-registry.ts), so the record stride tracks the model
// automatically and can never disagree with embedder.EMBEDDING_DIM or the iframe's
// OUTPUT_DIM. SIGN_BYTES === ceil(dim/8) matches binary.ts packSignBits() exactly.
export const Q_BYTES = ACTIVE_MODEL_SPEC.dim; // int8 rerank tier, one byte per dim
// f64 dequant scale (little-endian). Was f32 (4 B) — truncating to float32 here
// while IndexedDB's `embeddings` store kept the untruncated float64 `s` meant a
// hydrated peer dequantized with a slightly different scale than the originating
// device, producing ~1e-7 score divergence on near-ties. f64 matches IDB exactly
// (verbatim byte-copy hydration, as the module header promises) at a cost of 4
// extra bytes/record (~1% of the 440 B stride).
export const S_BYTES = 8;
export const SIGN_BYTES = (Q_BYTES + 7) >> 3; // packed sign bits for the candidate tier (ceil(dim/8))
export const CRC_BYTES = 4; // CRC-32 over [q|s|sign] — detects in-range sync bit-rot (GAP-1)
export const RECORD_PAYLOAD_BYTES = Q_BYTES + S_BYTES + SIGN_BYTES; // 440 — the CRC covers exactly this prefix
export const VEC_BYTES = RECORD_PAYLOAD_BYTES + CRC_BYTES; // 444 — fixed per-record stride
export const DIM = Q_BYTES; // logical embedding dimension

// Per-shard cap. Obsidian Sync's default per-file limit is 5 MB; staying at 4 MB
// keeps headroom and bounds iCloud whole-file re-upload cost on each append.
export const SHARD_CAP_BYTES = 4 * 1024 * 1024;
export const MAX_VECTORS_PER_SHARD = Math.floor(SHARD_CAP_BYTES / VEC_BYTES);

// Bumped when the on-disk record/meta layout changes (forces a version-gate refusal).
// 1→2: per-record CRC-32 (GAP-1) widened the record 436→440 B. Old format-1
// sidecars are refused by metaAccepts and re-hydrated on the next full reindex.
// 2→3: dequant scale f32→f64 (436/440 payload/record → 440/444) so a hydrated
// peer's scale is bit-identical to IndexedDB's, not just float32-close. Old
// format-2 sidecars are refused by metaAccepts and re-hydrated on the next full
// reindex — same self-healing path as the 1→2 bump.
export const SIDECAR_FORMAT = 3;

// One chunk's persisted tiers — the unit the sidecar stores and hydration writes back.
export interface TierBytes {
    q: Int8Array; // length Q_BYTES — int8 quantized vector
    s: number; // dequant scale (vᵢ ≈ qᵢ·s)
    sign: Uint8Array; // length SIGN_BYTES — packed sign bits from the TRUE fp32 vector
}

export interface VectorRecord {
    id: string; // chunk_id (path-salted content hash)
    dim: number;
    shard: string; // owning deviceId
    seq: number; // rotation sequence within this device's shards (0, 1, 2, ...)
    off: number; // byte offset of this record within the shard
    mtime: number;
    tombstone?: false;
}

export interface TombstoneRecord {
    id: string;
    shard: string; // owning deviceId — tombstones suppress only same-device records
    tombstone: true;
    mtime: number;
}

export type IndexRecord = VectorRecord | TombstoneRecord;

export interface ResolvedEntry {
    id: string;
    shard: string;
    seq: number;
    off: number;
    mtime: number;
    dim: number; // stored embedding dim — asserted at decode so a stride mismatch fails loud (F9)
}

export interface ScanResult {
    map: Map<string, ResolvedEntry>; // id → winning live record location
    tombstoneIds: Set<string>; // ids whose every device's latest record is a tombstone
    parseMs: number;
    recordCount: number;
    skippedLines: number;
}

export function isTombstone(r: IndexRecord): r is TombstoneRecord {
    return (r as TombstoneRecord).tombstone === true;
}

// Record-level shape guard for a parsed jsonl line. A torn mid-write can leave a
// line that is valid JSON but the wrong shape (missing off, non-numeric seq);
// without this, such a record reaches the perId map and the seq/off arithmetic in
// scanJsonl. Tombstone-discriminated: a tombstone needs only id/shard/mtime, a
// vector record additionally needs numeric seq/off. This is the record-level
// complement to the byte-level isOffsetInRange guard at the shard layer.
export function isValidRecord(r: unknown): r is IndexRecord {
    if (typeof r !== 'object' || r === null) return false;
    const o = r as Record<string, unknown>;
    if (typeof o.id !== 'string' || typeof o.shard !== 'string' || typeof o.mtime !== 'number') return false;
    if (o.tombstone === true) return true;
    return typeof o.seq === 'number' && typeof o.off === 'number';
}

// ---- 444 B record codec ----

// CRC-32 (IEEE 802.3) over a byte range. Detects an in-range bit-flip in a synced
// shard that would otherwise make the sign tier (stage-1) and the int8 tier
// (stage-2) silently disagree with no detector (GAP-1). Lazy 256-entry table;
// ~µs per 440 B payload.
let CRC32_TABLE: Uint32Array | null = null;
function crc32(bytes: Uint8Array, start: number, end: number): number {
    let table = CRC32_TABLE;
    if (!table) {
        table = new Uint32Array(256);
        for (let n = 0; n < 256; n++) {
            let c = n;
            for (let k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
            table[n] = c >>> 0;
        }
        CRC32_TABLE = table;
    }
    let crc = 0xFFFFFFFF;
    for (let i = start; i < end; i++) crc = (crc >>> 8) ^ table[(crc ^ bytes[i]) & 0xFF];
    return (crc ^ 0xFFFFFFFF) >>> 0;
}

// Pack one chunk's tiers into a 444 B record: [q:384 | s:f64LE:8 | sign:48 | crc:u32LE:4].
export function encodeRecord(t: TierBytes): Uint8Array {
    if (t.q.length !== Q_BYTES) throw new Error(`encodeRecord: q length ${t.q.length} != ${Q_BYTES}`);
    if (t.sign.length !== SIGN_BYTES) throw new Error(`encodeRecord: sign length ${t.sign.length} != ${SIGN_BYTES}`);
    const out = new Uint8Array(VEC_BYTES);
    out.set(new Uint8Array(t.q.buffer, t.q.byteOffset, Q_BYTES), 0); // int8 bytes reinterpreted as u8 — same bytes
    new DataView(out.buffer).setFloat64(Q_BYTES, t.s, true); // little-endian scale, full IDB precision
    out.set(t.sign, Q_BYTES + S_BYTES);
    // Trailing CRC-32 over [q|s|sign], little-endian (GAP-1).
    new DataView(out.buffer).setUint32(RECORD_PAYLOAD_BYTES, crc32(out, 0, RECORD_PAYLOAD_BYTES), true);
    return out;
}

// Slice a 444 B record out of a shard buffer at byte offset `off`. Caller must
// have validated isOffsetInRange first (or accept a throw on a short buffer).
// Throws on a CRC mismatch (corrupt record) or a dim≠DIM stride mismatch — both
// are "skip this record" conditions the hydrate callers catch and treat as null.
export function decodeRecord(buf: ArrayBuffer, off: number, expectedDim: number = DIM): TierBytes {
    if (expectedDim !== DIM) {
        throw new Error(`decodeRecord: record dim ${expectedDim} != ${DIM} — stride mismatch, refusing to mis-slice`);
    }
    if (!isOffsetInRange(buf.byteLength, off)) {
        throw new Error(`decodeRecord: off ${off} + ${VEC_BYTES} exceeds buffer ${buf.byteLength}`);
    }
    const bytes = new Uint8Array(buf);
    const stored = new DataView(buf).getUint32(off + RECORD_PAYLOAD_BYTES, true);
    const computed = crc32(bytes, off, off + RECORD_PAYLOAD_BYTES);
    if (stored !== computed) {
        throw new Error(`decodeRecord: CRC mismatch at off ${off} (stored ${stored}, computed ${computed}) — corrupt record`);
    }
    const q = new Int8Array(buf.slice(off, off + Q_BYTES));
    const s = new DataView(buf).getFloat64(off + Q_BYTES, true);
    const sign = new Uint8Array(buf.slice(off + Q_BYTES + S_BYTES, off + RECORD_PAYLOAD_BYTES));
    return { q, s, sign };
}

// Partial-arrival guard: a jsonl record can sync before its shard's bytes, or
// point past the synced length. Out-of-range → "not yet available", skip this
// pass, picked up on the next reconcile. Never an error.
export function isOffsetInRange(binSize: number, off: number): boolean {
    return off >= 0 && off + VEC_BYTES <= binSize;
}

// ---- path helpers ----

export function shardPathFor(indexDir: string, deviceId: string, seq: number): string {
    return `${indexDir}/embeddings.${deviceId}.${seq}.bin`;
}

export function jsonlPathFor(indexDir: string, deviceId: string): string {
    return `${indexDir}/index.${deviceId}.jsonl`;
}

export function metaPathFor(indexDir: string, deviceId: string): string {
    return `${indexDir}/meta.${deviceId}.json`;
}

// Cross-device BM25 sync artifact (Phase 3) — a gzipped, corpus-global MiniSearch
// blob a fresh/evicted device loads instead of refitting over all bodies. Per-device-
// written + freshest-picked (like the vector sidecar) so it inherits the structural
// conflict-immunity even though BM25 postings are corpus-global, not per-device. The
// `.json.gz` suffix matches NEITHER SHARD_RE nor JSONL_RE below, so the vector
// scanners AND sidecarDirSignature ignore it — a device that predates Phase 3 never
// reads it and just refits, exactly as before (graceful degrade, no format bump).
export function bm25PathFor(indexDir: string, deviceId: string): string {
    return `${indexDir}/bm25.${deviceId}.json.gz`;
}

// deviceIds are `mobile-9f3a1c08` / `desktop-2b7e54aa` — alnum + dash, never a
// dot or space. The strict patterns below therefore exclude sync conflict copies
// ("index.dev 2.jsonl", "embeddings.dev.0 2.bin"), which always carry a space.
const SHARD_RE = /^embeddings\.([A-Za-z0-9-]+)\.(\d+)\.bin$/;
const JSONL_RE = /^index\.([A-Za-z0-9-]+)\.jsonl$/;
const META_RE = /^meta\.([A-Za-z0-9-]+)\.json$/;       // matches NEITHER bm25.*.json.gz nor *.json.tmp
const BM25_RE = /^bm25\.([A-Za-z0-9-]+)\.json\.gz$/;   // matches NEITHER meta.*.json nor *.json.gz.tmp

export interface DeviceShard {
    seq: number;
    path: string;
    size: number;
}

function baseName(path: string): string {
    return path.split('/').pop() ?? '';
}

async function exists(adapter: DataAdapter, path: string): Promise<boolean> {
    return adapter.exists(path).catch(() => false);
}

// Obsidian's adapter.write/writeBinary do NOT create parent directories — ensure
// the index dir exists before the first write. Idempotent; tolerates "already exists".
export async function ensureDir(adapter: DataAdapter, dir: string): Promise<void> {
    try {
        if (!(await exists(adapter, dir))) await adapter.mkdir(dir);
    } catch {
        /* already exists or unsupported — writes below will surface a real failure */
    }
}

// List every device's index jsonl under `indexDir`, sorted, conflict copies excluded.
export async function listDeviceJsonls(adapter: DataAdapter, indexDir: string): Promise<string[]> {
    if (!(await exists(adapter, indexDir))) return [];
    const ls = await adapter.list(indexDir).catch(() => ({ folders: [] as string[], files: [] as string[] }));
    const out: string[] = [];
    for (const f of ls.files) {
        if (JSONL_RE.test(baseName(f))) out.push(f);
    }
    return out.sort();
}

// List every device's meta.<deviceId>.json under `indexDir`, sorted, conflict
// copies excluded. Mirrors listDeviceJsonls; sidecarDirSignature watches these
// too so a producer's version-gate meta update is never missed.
export async function listDeviceMetas(adapter: DataAdapter, indexDir: string): Promise<string[]> {
    if (!(await exists(adapter, indexDir))) return [];
    const ls = await adapter.list(indexDir).catch(() => ({ folders: [] as string[], files: [] as string[] }));
    const out: string[] = [];
    for (const f of ls.files) {
        if (META_RE.test(baseName(f))) out.push(f);
    }
    return out.sort();
}

// Cheap change-signature for the index dir: each device jsonl's AND meta's (path,
// size, mtime). Lets the periodic reconcile skip the expensive whole-vault
// re-chunk when no device file has changed since the last run — a handful of
// stat calls vs. re-reading and re-chunking every note. Any append (size/mtime)
// or new device file (path/count) changes the signature; a missing stat sorts to
// -1. meta.<deviceId>.json is watched alongside the jsonls (not just jsonls
// alone) because a version-refused producer's meta rewrite — e.g. a
// chunkerVersion bump after a full reindex — doesn't necessarily land in the
// same instant as its next jsonl append; omitting it left that meta update
// unnoticed until the jsonl eventually changed too.
//
// `selfDeviceId` (when supplied) excludes the LOCAL device's own jsonl/meta from
// the signature, so a device's own writes cannot wake its own reconcile — only a
// PEER change should. Without this, a mobile catch-up burst writing to
// index.<self>.jsonl flips the signature and self-triggers a whole-vault
// reChunkLive, the device's own writes endlessly waking its own sweep. Own
// writes never carry new peer data (the local index already has them), so the
// exclusion loses nothing; the empty-store branch in shouldReconcileSidecar
// still forces recovery for an evicted/fresh device regardless. The persisted
// and live signatures must both be computed with the SAME selfDeviceId for the
// comparison to hold.
export async function sidecarDirSignature(adapter: DataAdapter, indexDir: string, selfDeviceId?: string): Promise<string> {
    const jsonls = await listDeviceJsonls(adapter, indexDir);
    const metas = await listDeviceMetas(adapter, indexDir);
    const parts: string[] = [];
    for (const p of jsonls) {
        if (selfDeviceId && deviceIdFromJsonlPath(p) === selfDeviceId) continue;
        const st = await adapter.stat(p).catch(() => null);
        parts.push(`${p}:${st?.size ?? -1}:${st?.mtime ?? -1}`);
    }
    for (const p of metas) {
        if (selfDeviceId && deviceIdFromArtifactPath(p) === selfDeviceId) continue;
        const st = await adapter.stat(p).catch(() => null);
        parts.push(`${p}:${st?.size ?? -1}:${st?.mtime ?? -1}`);
    }
    return parts.join('|');
}

// Decide whether the whole-vault hydrate sweep must run, given the live sidecar
// signature, the signature persisted at the last successful reconcile, and
// whether the local index is empty. The sweep is expensive (reChunkLive reads +
// re-chunks every note on the main thread on mobile), so we skip it whenever the
// sidecar dir is byte-identical to last time. The empty-store branch is
// MANDATORY and dominates: an evicted or fresh device trivially "matches" a null
// persisted sig, but it has lost its index and MUST sweep to recover — skipping
// there is the silent-no-results failure. Pure so the gate is unit-testable.
export function shouldReconcileSidecar(
    liveSig: string,
    persistedSig: string | null,
    indexEmpty: boolean,
): boolean {
    if (indexEmpty) return true;        // eviction / fresh device — recover regardless of sig
    return liveSig !== persistedSig;    // otherwise skip only when nothing changed
}

// True when `meta` — this DEVICE'S OWN last-written sidecar meta — predates
// `currentFormat` (defaults to the live SIDECAR_FORMAT). A mismatch means the
// on-disk shard bytes for this device may still use an OLDER record stride than
// decodeRecord (fixed to the current S_BYTES/VEC_BYTES constants) expects — a
// SIDECAR_FORMAT bump is deliberately excluded from identityMatches (it governs
// only the cross-device file protocol, not local IDB validity — identity.ts), so
// nothing else forces a full reindex (the only path that clears it via
// clearDevice) just because the format constant moved. A null meta (fresh
// install, nothing written yet) is NOT stale — there is nothing on disk to
// misread. Shared by compactOwnSidecar (refuse to decode + wipe instead of
// misreading stale-stride bytes as corrupt) and the sidecar-commit write path
// (wipe before the meta write claims a format the actual bytes don't have yet)
// so the two call sites can't drift out of sync. Pure so the gate is
// unit-testable without a SearchOrchestrator harness.
export function staleSidecarFormat(meta: { format: number } | null, currentFormat: number = SIDECAR_FORMAT): boolean {
    return meta !== null && meta.format !== currentFormat;
}

// Extract the owning deviceId from an `index.<deviceId>.jsonl` path, or null if
// the basename isn't a strict device jsonl (e.g. a conflict copy).
export function deviceIdFromJsonlPath(path: string): string | null {
    const m = JSONL_RE.exec(baseName(path));
    return m ? m[1] : null;
}

// Extract the owning deviceId from ANY per-device sidecar artifact path — shard,
// jsonl, meta, or the cross-device BM25 blob — or null if the basename doesn't
// match one of the strict per-device patterns (conflict copy, unrelated file).
// Shared by every caller that needs "whose file is this", notably
// sweepOrphanTmpFiles' peer-file guard: a .tmp's REAL name (after stripping
// '.tmp') matches one of these same four patterns, so this also resolves the
// owner of an in-flight atomic write.
function deviceIdFromArtifactPath(path: string): string | null {
    const b = baseName(path);
    const m = SHARD_RE.exec(b) ?? JSONL_RE.exec(b) ?? META_RE.exec(b) ?? BM25_RE.exec(b);
    return m ? m[1] : null;
}

// Every deviceId that owns ANY sidecar artifact under `indexDir` — vector shard,
// jsonl log, version meta, OR cross-device BM25 blob — deduped to one id apiece.
// The reap/clear path enumerates by THIS union, never by jsonl alone: clearDevice
// removes a device's four file types best-effort and non-atomically, and a sync
// provider propagates each file independently, so a dead device's jsonl can vanish
// while its meta/bm25/shards linger. A jsonl-keyed reap would never revisit those
// leftovers → a permanent disk leak (they're gated out of results by metaAccepts,
// so never WRONG — but the bytes accumulate, and on iOS that pressure feeds the
// very eviction the sidecar exists to survive). Folding every artifact's deviceId
// into one set lets the next reap re-find a half-cleared device and let
// clearDevice (idempotent, exists-guarded) finish it → eventually-consistent
// cleanup, no atomic multi-file delete required. Conflict copies ("…dev 2.json")
// and .tmp leftovers are excluded by the same strict patterns the scanners use
// (sweepOrphanTmpFiles owns the .tmp class).
export async function listSidecarDeviceIds(adapter: DataAdapter, indexDir: string): Promise<string[]> {
    if (!(await exists(adapter, indexDir))) return [];
    const ls = await adapter.list(indexDir).catch(() => ({ folders: [] as string[], files: [] as string[] }));
    const ids = new Set<string>();
    for (const f of ls.files) {
        const dev = deviceIdFromArtifactPath(f);
        if (dev) ids.add(dev);
    }
    return [...ids].sort();
}

// List all shards belonging to `deviceId`, sorted by seq ascending.
export async function listDeviceShards(adapter: DataAdapter, indexDir: string, deviceId: string): Promise<DeviceShard[]> {
    if (!(await exists(adapter, indexDir))) return [];
    const ls = await adapter.list(indexDir).catch(() => ({ folders: [] as string[], files: [] as string[] }));
    const out: DeviceShard[] = [];
    for (const f of ls.files) {
        const m = SHARD_RE.exec(baseName(f));
        if (!m || m[1] !== deviceId) continue;
        const seq = parseInt(m[2], 10);
        if (!Number.isFinite(seq)) continue;
        const stat = await adapter.stat(f).catch(() => null);
        out.push({ seq, path: f, size: stat?.size ?? 0 });
    }
    out.sort((a, b) => a.seq - b.seq);
    return out;
}

interface ActiveShardPick {
    seq: number;
    path: string;
    existingSize: number;
}

// Pick the shard the next write targets, rotating to a fresh seq when the current
// active shard wouldn't fit `neededBytes`.
async function pickActiveShard(adapter: DataAdapter, indexDir: string, deviceId: string, neededBytes: number): Promise<ActiveShardPick> {
    const shards = await listDeviceShards(adapter, indexDir, deviceId);
    if (shards.length === 0) {
        return { seq: 0, path: shardPathFor(indexDir, deviceId, 0), existingSize: 0 };
    }
    const last = shards[shards.length - 1];
    if (last.size + neededBytes <= SHARD_CAP_BYTES) {
        return { seq: last.seq, path: last.path, existingSize: last.size };
    }
    return { seq: last.seq + 1, path: shardPathFor(indexDir, deviceId, last.seq + 1), existingSize: 0 };
}

// ---- JSONL reader + resolver ----

// Strips a trailing partial line (no newline) — protects against truncated writes.
async function readJsonl(adapter: DataAdapter, path: string): Promise<{ records: IndexRecord[]; skipped: number }> {
    if (!(await exists(adapter, path))) return { records: [], skipped: 0 };
    const raw = await adapter.read(path);
    if (raw.length === 0) return { records: [], skipped: 0 };
    const endsWithNewline = raw.endsWith('\n');
    const lines = raw.split('\n');
    const stop = endsWithNewline ? lines.length : lines.length - 1;
    const records: IndexRecord[] = [];
    let skipped = 0;
    for (let i = 0; i < stop; i++) {
        const line = lines[i];
        if (!line || line.trim().length === 0) continue;
        try {
            const rec: unknown = JSON.parse(line);
            if (!isValidRecord(rec)) { skipped++; continue; }
            records.push(rec);
        } catch {
            skipped++;
        }
    }
    if (!endsWithNewline && lines[lines.length - 1].trim().length > 0) skipped++;
    return { records, skipped };
}

// Within one device, which record is the latest? Higher mtime wins; at equal mtime
// a tombstone beats a live record (sticky delete), two live records break by
// seq then off, two tombstones keep the incumbent.
function winsWithinDevice(challenger: IndexRecord, incumbent: IndexRecord): boolean {
    if (challenger.mtime !== incumbent.mtime) return challenger.mtime > incumbent.mtime;
    const cT = isTombstone(challenger);
    const iT = isTombstone(incumbent);
    if (cT && !iT) return true;
    if (!cT && iT) return false;
    if (cT && iT) return false;
    const cV = challenger as VectorRecord;
    const iV = incumbent as VectorRecord;
    if ((cV.seq ?? 0) !== (iV.seq ?? 0)) return (cV.seq ?? 0) > (iV.seq ?? 0);
    return cV.off > iV.off;
}

// Across devices (both records live): higher mtime wins; at equal mtime the
// lex-larger deviceId wins — deterministic everywhere regardless of jsonl order.
function crossDeviceWins(challenger: VectorRecord, incumbent: VectorRecord): boolean {
    if (challenger.mtime !== incumbent.mtime) return challenger.mtime > incumbent.mtime;
    if (challenger.shard !== incumbent.shard) return challenger.shard > incumbent.shard;
    if ((challenger.seq ?? 0) !== (incumbent.seq ?? 0)) return (challenger.seq ?? 0) > (incumbent.seq ?? 0);
    return challenger.off > incumbent.off;
}

// Fold all devices' jsonls into one resolved id→location map. Two-level resolution:
// first the latest record per (id, device), then the cross-device winner among the
// devices whose latest is live. A tombstone removes only its own device's
// contribution; an id maps to a live record as long as ANY device still has one.
export async function scanJsonl(adapter: DataAdapter, paths: string[]): Promise<ScanResult> {
    const t0 = performance.now();
    const perId = new Map<string, Map<string, IndexRecord>>(); // id → (deviceId → latest record)
    let total = 0;
    let skipped = 0;
    for (const path of paths) {
        const { records, skipped: s } = await readJsonl(adapter, path);
        skipped += s;
        total += records.length;
        for (const r of records) {
            let byDev = perId.get(r.id);
            if (!byDev) {
                byDev = new Map();
                perId.set(r.id, byDev);
            }
            const prev = byDev.get(r.shard);
            if (!prev || winsWithinDevice(r, prev)) byDev.set(r.shard, r);
        }
    }
    const map = new Map<string, ResolvedEntry>();
    const tombstoneIds = new Set<string>();
    for (const [id, byDev] of perId) {
        let winner: VectorRecord | null = null;
        for (const rec of byDev.values()) {
            if (isTombstone(rec)) continue; // this device's latest is a delete → no contribution
            const v = rec;
            if (!winner || crossDeviceWins(v, winner)) winner = v;
        }
        if (winner) {
            map.set(id, { id, shard: winner.shard, seq: winner.seq ?? 0, off: winner.off, mtime: winner.mtime, dim: winner.dim });
        } else {
            tombstoneIds.add(id); // no device has a live copy
        }
    }
    return { map, tombstoneIds, parseMs: performance.now() - t0, recordCount: total, skippedLines: skipped };
}

// ---- read paths ----

// Read one resolved record's tiers. Returns null when the shard is missing or the
// offset is past the synced length (partial arrival) — caller skips, self-heals next pass.
export async function readRecordAt(adapter: DataAdapter, indexDir: string, entry: ResolvedEntry): Promise<TierBytes | null> {
    const path = shardPathFor(indexDir, entry.shard, entry.seq);
    const buf = await adapter.readBinary(path).catch(() => null);
    if (!buf) return null;
    if (!isOffsetInRange(buf.byteLength, entry.off)) return null;
    // dim mismatch or CRC failure (corrupt record) → skip, re-embed next pass.
    try { return decodeRecord(buf, entry.off, entry.dim); }
    catch { return null; }
}

// ---- atomic write: temp + rename ----

// Write to `${path}.tmp` then atomically swap in. A crash mid-write leaves the
// existing file intact; sweepOrphanTmpFiles recovers the .tmp on next load.
async function writeBinaryAtomic(adapter: DataAdapter, path: string, buffer: ArrayBuffer): Promise<void> {
    const tmp = path + '.tmp';
    await adapter.writeBinary(tmp, buffer);
    try {
        await adapter.rename(tmp, path);
    } catch {
        // Fallback for platforms where rename-over-existing fails (iOS Capacitor).
        if (await exists(adapter, path)) await adapter.remove(path);
        await adapter.rename(tmp, path);
    }
}

// Write text via the same temp + atomic-rename dance as writeBinaryAtomic. Used
// for the jsonl read+rewrite fallback and the device-meta write so a crash
// mid-write can't truncate the file — a reader sees the prior file or the new
// one, never a partial. Exported for sidecar-meta's writeDeviceMeta.
export async function writeTextAtomic(adapter: DataAdapter, path: string, text: string): Promise<void> {
    const tmp = path + '.tmp';
    await adapter.write(tmp, text);
    try {
        await adapter.rename(tmp, path);
    } catch {
        // Fallback for platforms where rename-over-existing fails (iOS Capacitor).
        if (await exists(adapter, path)) await adapter.remove(path);
        await adapter.rename(tmp, path);
    }
}

// Binary twin of writeTextAtomic — same write-tmp-then-rename durability, for the
// gzipped cross-device BM25 artifact (writeBinary wants an ArrayBuffer).
export async function writeBytesAtomic(adapter: DataAdapter, path: string, bytes: Uint8Array): Promise<void> {
    const tmp = path + '.tmp';
    // Hand writeBinary an exact ArrayBuffer (a Uint8Array can be a view into a larger
    // buffer; slice when it isn't a tight, zero-offset view).
    const ab = bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
        ? bytes.buffer
        : bytes.slice().buffer;
    await adapter.writeBinary(tmp, ab as ArrayBuffer);
    try {
        await adapter.rename(tmp, path);
    } catch {
        if (await exists(adapter, path)) await adapter.remove(path);
        await adapter.rename(tmp, path);
    }
}

// Sweep orphan .tmp files (.bin.tmp shards, .jsonl.tmp logs, .json.tmp meta,
// .gz.tmp BM25 artifacts) left by a crashed atomic write.
//   - real file present alongside .tmp → swap completed/never-started → delete .tmp
//   - real file missing, .tmp present → crash between remove and rename → restore .tmp
// Scoped to `deviceId`'s OWN artifacts only: every atomic write in this module is
// single-writer (one device writes only its own shard/jsonl/meta/bm25 files), so
// a .tmp owned by a PEER deviceId is that peer's own in-flight write, still being
// written or mid-sync — deleting it can erase live in-progress data, and
// rename-restoring it can promote bytes that aren't fully synced yet. A .tmp
// whose real name doesn't resolve to any known deviceId (malformed/unexpected) is
// also left alone — only a name we can positively attribute to the local device
// is acted on. Returns the number of files acted on.
export async function sweepOrphanTmpFiles(adapter: DataAdapter, root: string, deviceId: string): Promise<number> {
    if (!(await exists(adapter, root))) return 0;
    let count = 0;
    const queue: string[] = [root];
    while (queue.length > 0) {
        const dir = queue.pop()!;
        const ls = await adapter.list(dir).catch(() => null);
        if (!ls) continue;
        for (const sub of ls.folders) queue.push(sub);
        for (const f of ls.files) {
            if (!f.endsWith('.bin.tmp') && !f.endsWith('.jsonl.tmp') && !f.endsWith('.json.tmp') && !f.endsWith('.gz.tmp')) continue;
            const realPath = f.slice(0, -'.tmp'.length);
            if (deviceIdFromArtifactPath(realPath) !== deviceId) continue; // not ours — never touch a peer's in-flight write
            try {
                if (await exists(adapter, realPath)) await adapter.remove(f);
                else await adapter.rename(f, realPath);
                count++;
            } catch {
                /* best-effort cleanup */
            }
        }
    }
    return count;
}

// ---- locking (in-process only) ----

// Per-index-directory async serializer: within one plugin instance no two writers
// run at once for the same dir. With per-device single-writer files there is no
// cross-device write contention, so no file sentinel is needed.
//
// NON-GOAL: cross-process writers on the SAME physical device. deviceId is stable
// per install, so two Obsidian processes opening the same vault on one machine map
// to the same index.<deviceId> files with independent in-process locks. Since every
// write is read-concat-write-whole-shard, concurrent flushes can lose the loser's
// records (last-rename-wins on the shard) — tmp+rename still prevents a *torn* file,
// just not a lost one. This is accepted: Obsidian is single-instance per vault, and
// two distinct machines get distinct deviceIds (no collision). The cost of a
// collision is re-embed on the next reconcile, never a corrupt live index.
const dirLocks = new Map<string, Promise<unknown>>();

async function withDirLock<T>(dir: string, fn: () => Promise<T>): Promise<T> {
    const prior = dirLocks.get(dir) ?? Promise.resolve();
    let release!: () => void;
    const myCompletion = new Promise<void>(resolve => {
        release = resolve;
    });
    dirLocks.set(dir, prior.then(() => myCompletion));
    try {
        await prior;
        return await fn();
    } finally {
        release();
    }
}

async function appendJsonlLine(adapter: DataAdapter, path: string, line: string): Promise<void> {
    if (!(await exists(adapter, path))) {
        await adapter.write(path, line);
        return;
    }
    try {
        await adapter.append(path, line);
    } catch {
        // .append can fail on iOS WKWebView under iCloud contention. Read+rewrite
        // preserves prior data — via tmp+rename so a crash mid-rewrite leaves the
        // prior log intact instead of truncating it to a partial prefix.
        const existing = await adapter.read(path);
        await writeTextAtomic(adapter, path, existing + line);
    }
}

// ---- write paths (each device writes ONLY its own files) ----

// Append a tombstone (compaction hint — own device only).
export async function appendTombstone(adapter: DataAdapter, indexDir: string, deviceId: string, id: string, mtime: number): Promise<void> {
    return withDirLock(indexDir, async () => {
        await ensureDir(adapter, indexDir);
        const rec: TombstoneRecord = { id, shard: deviceId, tombstone: true, mtime };
        await appendJsonlLine(adapter, jsonlPathFor(indexDir, deviceId), JSON.stringify(rec) + '\n');
    });
}

// Bulk append. Splits across as many shards as needed when the batch would
// overflow the active shard's cap; each shard write is bounded at SHARD_CAP_BYTES.
export async function bulkAppend(
    adapter: DataAdapter,
    indexDir: string,
    deviceId: string,
    records: Array<{ id: string; tiers: TierBytes; mtime: number }>,
): Promise<Array<{ seq: number; off: number }>> {
    if (records.length === 0) return [];
    return withDirLock(indexDir, async () => {
        await ensureDir(adapter, indexDir);
        const refs: Array<{ seq: number; off: number }> = [];
        const lines: string[] = [];
        let cursor = 0;
        while (cursor < records.length) {
            const active = await pickActiveShard(adapter, indexDir, deviceId, VEC_BYTES);
            const room = SHARD_CAP_BYTES - active.existingSize;
            const maxVecs = Math.floor(room / VEC_BYTES);
            if (maxVecs <= 0) throw new Error(`no room in shard seq=${active.seq} (size=${active.existingSize})`);
            const take = Math.min(maxVecs, records.length - cursor);
            const existing = active.existingSize > 0 ? await adapter.readBinary(active.path) : new ArrayBuffer(0);
            const merged = new Uint8Array(active.existingSize + take * VEC_BYTES);
            merged.set(new Uint8Array(existing), 0);
            let localCursor = active.existingSize;
            for (let i = 0; i < take; i++) {
                const { id, tiers, mtime } = records[cursor + i];
                merged.set(encodeRecord(tiers), localCursor);
                refs.push({ seq: active.seq, off: localCursor });
                lines.push(JSON.stringify({ id, dim: DIM, shard: deviceId, seq: active.seq, off: localCursor, mtime }));
                localCursor += VEC_BYTES;
            }
            await writeBinaryAtomic(adapter, active.path, merged.buffer);
            cursor += take;
        }
        if (lines.length > 0) await appendJsonlLine(adapter, jsonlPathFor(indexDir, deviceId), lines.join('\n') + '\n');
        return refs;
    });
}

// Drop ALL of `deviceId`'s sidecar artifacts — shards, jsonl, meta, AND the bm25
// blob. Two callers: the start of a FULL reindex (so the run REPLACES this device's
// vectors rather than appending to / doubling the prior pass) and the dead-identity
// reap (which must FULLY reclaim a stale device — leaving meta/bm25 behind would let
// it re-surface every reap forever, the leak listSidecarDeviceIds exists to end).
// Other devices' files are untouched.
export async function clearDevice(adapter: DataAdapter, indexDir: string, deviceId: string): Promise<void> {
    return withDirLock(indexDir, async () => {
        if (!(await exists(adapter, indexDir))) return;
        for (const s of await listDeviceShards(adapter, indexDir, deviceId)) {
            await adapter.remove(s.path).catch(() => {});
        }
        // Remove every per-device artifact so a full reindex (or a dead-identity
        // reap) fully replaces this device: jsonl log, version meta, and the cross-
        // device BM25 blob. Previously only the shards + jsonl were cleared, so
        // meta.<id>.json + bm25.<id>.json.gz lingered (gated out as stale, but
        // accumulating across reindexes).
        for (const p of [jsonlPathFor(indexDir, deviceId), metaPathFor(indexDir, deviceId), bm25PathFor(indexDir, deviceId)]) {
            if (await exists(adapter, p)) await adapter.remove(p).catch(() => {});
        }
    });
}

// Total bytes of `deviceId`'s shards — the cheap (stat-only) pre-gate compactDevice's
// caller uses to skip the whole-vault re-chunk for a sidecar too small to be worth a
// rewrite + re-upload.
export async function deviceShardBytes(adapter: DataAdapter, indexDir: string, deviceId: string): Promise<number> {
    let bytes = 0;
    for (const s of await listDeviceShards(adapter, indexDir, deviceId)) bytes += s.size;
    return bytes;
}

export interface CompactResult {
    compacted: boolean;
    // why nothing happened (compacted:false), or 'done'. 'incomplete-rechunk' means the
    // live-id oracle reported a skipped note → unsafe to delete → retry next session.
    // 'format-mismatch' means the caller detected this device's own on-disk sidecar
    // predates the current SIDECAR_FORMAT and refused to decode it (see compactOwnSidecar).
    reason?: 'below-floor' | 'below-ratio' | 'nothing-dead' | 'incomplete-rechunk' | 'format-mismatch' | 'done';
    recordsBefore: number; // total jsonl record lines before
    recordsAfter: number;  // live records kept after
    bytesBefore: number;   // total shard bytes before
    bytesAfter: number;    // total shard bytes after
    shed: number;          // own-shard records dropped as unreadable/corrupt (expected 0)
}

// Compact `deviceId`'s OWN sidecar in place: rewrite its jsonl + shards to hold only
// its live records, reclaiming the disk that superseded re-appends (same id, newer
// mtime) and orphaned records (deleted / edited-away ids — never tombstoned in prod,
// so they resolve "live" in scanJsonl forever and only the vault re-chunk knows they
// are dead) would otherwise hold until the next FULL reindex. Single-writer + the dir
// mutex make this safe without a cross-device sentinel (see the module header).
//
// liveIds is a THUNK, invoked INSIDE the dir lock, returning the live-vault id set (the
// eviction-robust oracle hydrate uses) and whether that set is COMPLETE. Taking the
// snapshot under the same lock that serializes appends is what makes the drop decision
// exact: no record can be appended between "what is live" (the thunk) and "what is on
// disk" (the scan), so a record is dropped iff its id is genuinely absent from a current,
// complete live set — no wall-clock/mtime reasoning, no concurrent-append race. If the
// thunk reports complete:false (a note failed to read / tokenize), the oracle is
// untrustworthy and we MUST NOT delete → return 'incomplete-rechunk' and retry later.
//
// Gated: no-op under the byte floor or the dead-ratio threshold, so a lean sidecar is
// never rewritten (and re-uploaded to iCloud) for a trivial gain.
//
// Sync- AND crash-safety rests on a GENERATIONAL rewrite: survivors are written to
// FRESH shard seqs (maxSeq+1…), the new jsonl is swapped in atomically (the single
// commit point), and only THEN are the old shards deleted. A peer mid-sync that holds
// the new jsonl but not the new shard resolves a fresh seq that isn't on disk yet →
// readBinary fails → skip → self-heal; it can NEVER resolve a new offset against an
// old shard's bytes (which would decode a different chunk as this id — the one real
// correctness hole, closed by never reusing a seq/offset). A crash before the swap
// leaves the old jsonl + old shards authoritative; orphaned fresh shards are reclaimed
// by the next pass. Peak memory ≈ one source + one dest shard (~8 MB), matching bulkAppend.
export async function compactDevice(
    adapter: DataAdapter,
    indexDir: string,
    deviceId: string,
    liveIds: () => Promise<{ ids: Set<string>; complete: boolean }>,
    opts: { minDeadRatio?: number; minShardBytes?: number } = {},
): Promise<CompactResult> {
    const minDeadRatio = opts.minDeadRatio ?? 0.5;
    const minShardBytes = opts.minShardBytes ?? 2 * 1024 * 1024;
    return withDirLock(indexDir, async () => {
        const nil: CompactResult = { compacted: false, recordsBefore: 0, recordsAfter: 0, bytesBefore: 0, bytesAfter: 0, shed: 0 };
        if (!(await exists(adapter, indexDir))) return nil;

        let shardsBefore = await listDeviceShards(adapter, indexDir, deviceId);

        // Crash-leak reclaim: a shard whose seq is referenced by ZERO jsonl lines
        // (not even a dead/superseded one) can only be a leftover from a PRIOR
        // compaction that crashed between the atomic jsonl swap (the single commit
        // point, above) and the old-shard delete that follows it — the swapped-in
        // jsonl already points solely at fresh seqs, so nothing on disk can ever
        // reference the stale ones again. This must run BEFORE the below-floor /
        // below-ratio / nothing-dead gates below: those compare record counts
        // within the CURRENT (already-compacted) jsonl, see nothing dead, and
        // return early — if the leak check were gated behind them too, a crash
        // right after the swap would leak the old shards permanently (the fast
        // path can never "see" bytes the jsonl no longer mentions at all).
        const { records: rawRecords } = await readJsonl(adapter, jsonlPathFor(indexDir, deviceId));
        const referencedSeqs = new Set<number>();
        for (const r of rawRecords) if (!isTombstone(r)) referencedSeqs.add(r.seq);
        const orphanShards = shardsBefore.filter(s => !referencedSeqs.has(s.seq));
        if (orphanShards.length > 0) {
            for (const s of orphanShards) await adapter.remove(s.path).catch(() => {});
            shardsBefore = shardsBefore.filter(s => referencedSeqs.has(s.seq));
        }

        const bytesBefore = shardsBefore.reduce((sum, s) => sum + s.size, 0);
        if (bytesBefore < minShardBytes) return { ...nil, reason: 'below-floor', bytesBefore };

        // Snapshot the live-id oracle UNDER the lock (excludes concurrent appends), only
        // once past the floor (so a small sidecar never pays the re-chunk).
        const { ids: keepIds, complete } = await liveIds();
        if (!complete) return { ...nil, reason: 'incomplete-rechunk', bytesBefore };

        // This device alone → scanJsonl's cross-device step is trivial and supersedes
        // collapse to the latest record per id.
        const scan = await scanJsonl(adapter, [jsonlPathFor(indexDir, deviceId)]);
        const recordsBefore = scan.recordCount;
        const keep: ResolvedEntry[] = [];
        for (const e of scan.map.values()) if (keepIds.has(e.id)) keep.push(e);
        const base = { ...nil, recordsBefore, recordsAfter: keep.length, bytesBefore, bytesAfter: bytesBefore };
        if (recordsBefore === 0 || keep.length >= recordsBefore) return { ...base, reason: 'nothing-dead' };
        if ((recordsBefore - keep.length) / recordsBefore < minDeadRatio) return { ...base, reason: 'below-ratio' };

        // ---- generational rewrite (fresh seqs > every existing seq) ----
        const maxSeq = shardsBefore.length ? shardsBefore[shardsBefore.length - 1].seq : -1;
        keep.sort((a, b) => (a.seq - b.seq) || (a.off - b.off)); // each source shard read once
        const newLines: string[] = [];
        const newShardPaths: string[] = [];
        let shed = 0;
        let destSeq = maxSeq + 1;
        let destBuf = new Uint8Array(SHARD_CAP_BYTES);
        let destLen = 0;
        let srcSeq = -1;
        let srcBuf: ArrayBuffer | null = null;
        const flushDest = async (): Promise<void> => {
            if (destLen === 0) return;
            const path = shardPathFor(indexDir, deviceId, destSeq);
            await writeBinaryAtomic(adapter, path, destBuf.slice(0, destLen).buffer);
            newShardPaths.push(path);
            destSeq++;
            destBuf = new Uint8Array(SHARD_CAP_BYTES);
            destLen = 0;
        };
        try {
            for (const e of keep) {
                if (e.seq !== srcSeq) {
                    srcSeq = e.seq;
                    srcBuf = await adapter.readBinary(shardPathFor(indexDir, deviceId, e.seq)).catch(() => null);
                }
                // Shed a record whose source bytes are missing / out-of-range / corrupt:
                // it is already dead-on-read at hydrate (decodeRecord throws → null), so
                // dropping it costs nothing (the note re-embeds either way) and reclaims
                // its bytes. `shed` is surfaced because it should be ZERO in normal
                // operation — any non-zero is an own-shard corruption breadcrumb.
                if (!srcBuf || !isOffsetInRange(srcBuf.byteLength, e.off)) { shed++; continue; }
                try { decodeRecord(srcBuf, e.off, e.dim); } catch { shed++; continue; }
                if (destLen + VEC_BYTES > SHARD_CAP_BYTES) await flushDest();
                destBuf.set(new Uint8Array(srcBuf, e.off, VEC_BYTES), destLen);
                newLines.push(JSON.stringify({ id: e.id, dim: e.dim, shard: deviceId, seq: destSeq, off: destLen, mtime: e.mtime }));
                destLen += VEC_BYTES;
            }
            await flushDest();
        } catch (err) {
            // Roll back the fresh shards; the old jsonl + old shards remain authoritative.
            for (const p of newShardPaths) await adapter.remove(p).catch(() => {});
            throw err;
        }

        // Commit: atomic swap to the new jsonl (now pointing only at fresh shards).
        await writeTextAtomic(adapter, jsonlPathFor(indexDir, deviceId), newLines.length ? newLines.join('\n') + '\n' : '');
        // Reclaim: delete the OLD shards (seq ≤ maxSeq); fresh shards (> maxSeq) stay.
        for (const s of shardsBefore) await adapter.remove(s.path).catch(() => {});

        return { compacted: true, reason: 'done', recordsBefore, recordsAfter: newLines.length, bytesBefore, bytesAfter: newLines.length * VEC_BYTES, shed };
    });
}
