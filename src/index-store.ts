// IndexedDB-backed index store.
//
// Schema (one database, five object stores):
//   chunks      — chunk_id → Chunk (text + metadata)
//   embeddings  — chunk_id → QuantVec {q:Int8Array(d), s:number}  ← int8 exact-rerank tier
//   binary      — chunk_id → Uint8Array(ceil(d/8))  ← sign-bit candidate tier
//   files       — note_path → { mtimeMs, chunk_ids }
//   meta        — singleton config: { embeddingDim, lastIndexedAt, ... }
//
// The rerank tier is int8-quantized (SQ8, see quant.ts), NOT fp32: a free 4×
// shrink (1536 B → 388 B/vec) at ≤0.003 NDCG@10, the vectors dequant back to
// fp32 on read inside getEmbeddingsByIds so the search pipeline is unchanged.
//
// Why split chunks, embeddings, and binary? The search hot path is now:
//   1. Read ALL of `binary` once per generation (~64 KB/1k chunks, cached
//      in memory thereafter) → score asymmetric → pick top-N candidates.
//   2. Read int8 for ONLY those N chunk_ids via getEmbeddingsByIds() (dequant).
// Keeping each tier in its own store means stage 1 doesn't drag rerank bytes
// through the cursor (a full fp32-vector dequant scan, the old per-query cost)
// and stage 2 doesn't drag chunk-body bytes (the top-K text fetch).
//
// IndexedDB-on-iOS gotcha (per design doc Path A §3, validated 2026-05-13):
// never allocate a single Uint8Array > ~50 MB on iOS — jetsam kills the
// WebView with no JS error. We write embeddings one transaction per batch
// (~100 chunks), well under the limit even at 768d.

import type { Chunk, ChunkMeta } from './types';
import { packSignBits } from './binary';
import { quantizeInt8, dequantizeInt8, type QuantVec } from './quant';
import { sizeOfRow, type SizingRule, type StoreSizeRow } from './index-size';
import { ACTIVE_MODEL_SPEC } from './model-registry';

// Base IDB name. IndexedDB is ORIGIN-scoped and every Obsidian vault window
// shares the app://obsidian.md origin, so a bare constant name is ONE database
// shared by all open vaults — vault A's reindex (deleteDatabase) nukes vault
// B's index and force-closes B's connection via versionchange (live incident
// 2026-06-10: Example Vault's reindex killed ACME's fresh index + every subsequent
// ACME transaction threw "connection is closing"). The store therefore scopes
// the name per vault: `seek-index:<appId>` (open() takes the scope; appId is
// Obsidian's stable per-vault id — the same key it uses to vault-scope its own
// localStorage). The legacy unscoped DB is deleted fire-and-forget on first
// scoped open.
//
// The prefix is ALSO scoped by PLUGIN id (indexDbPrefix), so a second Seek build
// installed in the same vault — e.g. a prototype with id 'seek-prototype' — gets
// its OWN database and can't nuke/reindex the public build's index. main.ts
// passes the prefix at onload; the shipped id 'seek' resolves to 'seek-index',
// byte-identical to this legacy constant, so a released build never migrates.
const LEGACY_DB_NAME = 'seek-index';

// The per-PLUGIN IndexedDB name prefix. `seek` → `seek-index` (== LEGACY_DB_NAME,
// so the shipped build is unchanged); a differently-id'd build (`seek-prototype`)
// → a separate `seek-prototype-index` database. The vault scope (`:<appId>`) is
// appended by open(); two builds in one vault thus differ only by this prefix.
export function indexDbPrefix(pluginId: string): string {
    return `${pluginId}-index`;
}
// DB_VERSION 2 (2026-05-14): switched embedding dim 768 -> 512 (MRL slice)
// and model dtype q8 -> q4. The on-disk vectors are now shape-incompatible
// with the previous schema, so the upgrade path drops chunks/embeddings/files
// on first open. After upgrade the user sees an empty index until they run
// "Full reindex".
// DB_VERSION 3 (2026-05-14): chunk_id format changed from `${path}#chunk-N`
// to a content hash (cyrb53Hex of the embedded title+content). Old IDs are
// permanently incompatible with the new chunker output, so we drop and
// force-reindex on upgrade — same pattern as v1→v2. See chunker.ts for the
// design and the rearchitecture plan §Decisions.
// DB_VERSION 4 (2026-05-19): added `binary` store for sign-bit candidate-gen
// (see binary.ts). The fp32 store format is UNCHANGED, so this upgrade does
// NOT drop existing data — it creates the empty binary store and lazily
// backfills from fp32 on first open() (see backfillBinaryIfMissing). The
// "drop on upgrade" precedent above does not apply: only the new auxiliary
// tier needs population.
// DB_VERSION 5 (2026-06-01): embedding model swapped EmbeddingGemma (512-d,
// mean-pool) → granite-embedding-small-english-r2 (384-d, CLS-pool). The fp32
// vectors AND their derived sign-bit binary projections are dim/model-
// incompatible, so this DROPS chunks/embeddings/binary/files on upgrade —
// same precedent as v1→v2/v2→v3. User sees an empty index until "Full reindex".
// DB_VERSION 6 (2026-06-03): rerank tier fp32 Float32Array(d) → int8 QuantVec
// {q,s} (SQ8, see quant.ts). The on-disk value shape changed, so the old fp32
// records can't be read as QuantVec. In-place requantization in
// onupgradeneeded would need async cursor read+transform on the upgrade tx
// (fiddly + risky); instead we follow the established drop-and-reindex pattern
// — reindex is reliable post-recycle-fix and repopulates int8 + binary in
// lockstep via putBatch. DROPS chunks/embeddings/binary/files.
// DB_VERSION 7 (2026-06-16): ADDITIVE — new `bm25` store holding a serialized
// MiniSearch index (toJSON) + an analyzer/corpus stamp, so a cold start can
// MiniSearch.loadJSON it instead of paying the ~280ms fit(). Purely a cache:
// it's gated by the stamp and any miss falls back to fit(), so no existing
// store is touched and no reindex is forced.
// DB_VERSION 8 (2026-06-18): split the `chunks` store into `chunk_meta`
// (everything except the body) + `chunk_body` (chunk_id -> content string), so
// the resident frame can be metadata-only and the cold rebuild stops reading
// body text (see docs/seek-scaling.md §B1). Body text physically moves stores,
// so this is a clean DESTRUCTIVE bump (drop chunks/embeddings/binary/files; a
// reindex repopulates both new stores + the vector tiers in lockstep via
// putBatch) — the v4→v5 precedent, NOT a migration. The user reindexes once.
// DB_VERSION 9 (2026-06-18): note-level frontmatter VALUES are now folded into
// the dense embed text (token-budget.ts embedInput) AND into chunk_id
// (chunkIdFor's denseSuffix tail, chunker.ts buildDenseSuffix). Every chunk_id
// therefore changes, so the meta/body/embedding/binary stores (all keyed by the
// OLD ids) and the files store (which references them) are unusable — drop them
// and force one clean full re-embed, the v4→v5 precedent. The stamp-gated bm25
// cache and the meta config are left in place (the bm25 stamp won't match the
// new corpus, so it refits on its own). CHUNKER_VERSION 3→4 independently makes
// a peer's sidecar reject pre-v9 vectors. See chunker.ts buildDenseSuffix.
// DB_VERSION 10 (2026-06-18): lightweight cleanliness gates (shape/dedup/cap)
// shrink the dense suffix, so chunkIdFor's tail and every chunk_id shift again —
// same destructive drop-and-reindex as v9, paired with CHUNKER_VERSION 4→5.
// DB_VERSION 11 (2026-06-19): chunk start_line/end_line are now RAW-FILE line
// numbers (frontmatter included) instead of frontmatter-stripped body lines —
// the fix for the in-note highlight + click-scroll landing inside the frontmatter
// (and the highlight drifting to the note's first token hit) on any note with
// frontmatter (chunker.ts return site). chunk_id is independent of line numbers,
// so vectors are byte-identical and NOT re-embedded for correctness — but the
// stored spans must all migrate together (the mtime-only delta would leave a
// mixed body/file-relative index), so we force one clean rebuild here. No
// CHUNKER_VERSION bump: ids are unchanged, peers' sidecars stay valid, and the
// sidecar carries no line numbers (hydrate re-chunks the live file).
export const DB_VERSION = 11;
// Thrown by requireDb when the connection is null (closed by onunload, or by an
// onversionchange from another window/instance deleting the DB). Exported so the
// indexer's per-file commit catch can recognize it and abort the WHOLE pass fast,
// instead of logging one identical error per remaining file (the reindex storm).
export const STORE_NOT_OPENED = 'IndexStore not opened';
// Recognize the closed-store error. IndexedDB hands back a plain Error, so this is a
// message match — kept in ONE place (next to the thrower) so the indexer's "abort the
// whole pass on a closed store" check can't drift from requireDb's wording.
export function isStoreClosedError(e: unknown): boolean {
    return e instanceof Error && e.message === STORE_NOT_OPENED;
}
const STORE_CHUNKS = 'chunks';        // legacy (pre-v8) — only deleted on upgrade now
const STORE_CHUNK_META = 'chunk_meta'; // v8: Chunk minus `content`, keyed by chunk_id
const STORE_CHUNK_BODY = 'chunk_body'; // v8: chunk_id -> content string (out-of-line)
const STORE_EMBEDDINGS = 'embeddings';
const STORE_BINARY = 'binary';
const STORE_FILES = 'files';
const STORE_META = 'meta';
const STORE_BM25 = 'bm25';

const META_KEY = 'config';
const BM25_KEY = 'index';

// A persisted MiniSearch index: the toJSON string + an opaque stamp the caller
// (search.ts) uses to decide whether the blob is loadable for the live corpus +
// analyzer. The store treats `stamp` as opaque — it neither builds nor validates it.
export interface Bm25Record {
    json: string;
    stamp: unknown;
}

// One store's records captured for embed-free compaction (see `compact`). `key`
// is the primary key; `inlineKey` records whether the store derives its key from
// the value (chunk_meta→chunk_id, files→note_path) — those re-`put` WITHOUT an
// explicit key, while out-of-line stores (embeddings/binary/body/meta/bm25) must
// pass it back.
export interface StoreSnapshot {
    store: string;
    inlineKey: boolean;
    records: { key: IDBValidKey; value: unknown }[];
}

// A single re-`put` operation: `key` present ⇒ out-of-line store (pass it);
// absent ⇒ in-line store (the value carries its own key, and passing one would
// throw DataError). Pure so the key-mode branching is unit-testable without IDB.
export interface RestoreOp { store: string; value: unknown; key?: IDBValidKey }
export function planRestoreOps(snapshot: StoreSnapshot[]): RestoreOp[] {
    const ops: RestoreOp[] = [];
    for (const s of snapshot) {
        for (const r of s.records) {
            ops.push(s.inlineKey
                ? { store: s.store, value: r.value }
                : { store: s.store, value: r.value, key: r.key });
        }
    }
    return ops;
}

// The stores compaction snapshots + rewrites, with their key mode. Order is the
// write-back order; chunk_meta/files are in-line-keyed (keyPath), the rest aren't.
const COMPACTION_STORES: { store: string; inlineKey: boolean }[] = [
    { store: STORE_CHUNK_META, inlineKey: true },
    { store: STORE_CHUNK_BODY, inlineKey: false },
    { store: STORE_EMBEDDINGS, inlineKey: false },
    { store: STORE_BINARY,     inlineKey: false },
    { store: STORE_FILES,      inlineKey: true },
    { store: STORE_META,       inlineKey: false },
    { store: STORE_BM25,       inlineKey: false },
];

export interface FileRecord {
    note_path: string;
    mtimeMs: number;
    chunk_ids: string[];
    // cyrb53 of the file's raw bytes, written by every embed/hydrate commit.
    // computeDelta uses it to tell a real edit from a mtime-only re-stamp (iCloud
    // re-writes a synced file's mtime without touching its bytes — on iOS that
    // otherwise re-embeds identical content every couple seconds). Optional:
    // records written before this field shipped lack it and fall back to the
    // mtime-only path, which backfills the hash on their next re-embed.
    contentHash?: string;
}

// Referential-integrity orphan finder: chunk_ids present in the index but
// referenced by NO FILES record. Orphans are a file's chunks left behind when its
// record was overwritten (hydrate / applyDelta replacing a changed note's ids) or
// its delete event was missed; they sit in chunk_meta, so they can surface stale
// content until removed. Pure set difference, input-order-stable for deterministic
// batching; the sweep (search.ts sweepOrphanChunks) deletes the result.
export function findOrphanChunkIds(allChunkIds: string[], referenced: ReadonlySet<string>): string[] {
    return allChunkIds.filter(id => !referenced.has(id));
}

// Pure per-file delta classification — the single source of truth for the
// "is this file dirty?" rule, exported so it can be unit-tested without an
// IndexStore or Vault. computeDelta consults it twice: once on cheap metadata
// (mtime), and again with the freshly-hashed bytes only when the first call
// asks for them ('check-bytes'). This two-phase shape keeps the (potentially
// vault-wide) content read OFF every clean file while still distinguishing a
// real edit from an iCloud mtime-only re-stamp.
//   'dirty'       — (re)embed: never indexed, no stored hash (legacy → backfill),
//                   or the bytes changed.
//   'clean'       — up to date; skip.
//   'check-bytes' — mtime advanced on a hash-bearing record; the caller must read
//                   the bytes, hash them, and re-call with `liveHash`.
export type DeltaDecision = 'dirty' | 'clean' | 'check-bytes';
export function classifyFileDelta(
    prev: Pick<FileRecord, 'mtimeMs' | 'contentHash'> | undefined,
    liveMtime: number,
    liveHash?: string,
): DeltaDecision {
    if (prev === undefined) return 'dirty';            // never indexed
    if (liveMtime <= prev.mtimeMs) return 'clean';     // mtime unchanged
    if (prev.contentHash === undefined) return 'dirty'; // legacy record → re-embed backfills the hash
    if (liveHash === undefined) return 'check-bytes';  // need the bytes to decide
    return liveHash === prev.contentHash ? 'clean' : 'dirty';
}

export interface MetaConfig {
    embeddingDim: number;
    lastIndexedAt: string | null;
    schemaVersion: number;
    // Embedding model that produced the stored vectors. Optional — indexes
    // written before 2026-06-10 lack it (they are all english-r2). Used by
    // main.ts to surface "settings model ≠ index model → reindex needed";
    // never read in ranking.
    modelId?: string;
    // ── Index version identity (2026-06-20) ───────────────────────────────────
    // The build fingerprint the stored vectors + chunk_ids were produced under.
    // Optional — indexes written before this date lack them, so identityMatches
    // (identity.ts) reads them as a version mismatch and routes the index through
    // a one-time rebuild. Stamped ONLY by a full reindex / rebuild-from-sidecar;
    // a delta carries the prior values forward, because only a full rebuild may
    // honestly claim a new identity (mirrors how modelId/bgMean are preserved on
    // an incremental). See identity.ts for which fields gate a rebuild and why
    // analyzerVersion + dbVersion are deliberately excluded from that gate.
    chunkerVersion?: number;
    analyzerVersion?: string;
    revision?: string | null;
    // Corpus dense-cosine background (see dense-stats.ts). Drive the
    // corpus-AGNOSTIC display calibration + answerability gate: z = (cos − bgMean)
    // / bgStd is unitless and transfers across vaults/encoders. Computed at FULL
    // reindex only; a delta carries them forward (coarse corpus globals a few
    // changed files barely move). Absent ⇒ calibration off (corpus below
    // MIN_BG_SAMPLE, or an index built before these fields existed). NOT a
    // ranking input — folding them into fusion measured −0.0135 nDCG@10.
    bgMean?: number;
    bgStd?: number;
}

// Delete an IndexedDB database, resolving when it is gone. Extracted verbatim from
// nukeDatabase so the VersionError-recovery branch in openDb can reuse the exact
// same block-guard semantics. Bare setTimeout/clearTimeout (NOT window.*) so it also
// runs under the node test env, which has no `window`.
function deleteDbWithBlockGuard(dbName: string): Promise<void> {
    return new Promise<void>((resolve, reject) => {
        const req = indexedDB.deleteDatabase(dbName);
        let blockedTimer: ReturnType<typeof setTimeout> | null = null;
        req.onsuccess = () => {
            if (blockedTimer != null) clearTimeout(blockedTimer);
            resolve();
        };
        req.onerror = () => {
            if (blockedTimer != null) clearTimeout(blockedTimer);
            reject(req.error);
        };
        req.onblocked = () => {
            // Don't reject immediately — another tab/instance may close shortly.
            // Wait a generous interval, then fail with an actionable message.
            console.warn('[seek] deleteDatabase blocked — waiting up to 10 s for other connections to close');
            blockedTimer = setTimeout(() => {
                reject(new Error(
                    `deleteDatabase blocked: another Obsidian window/tab is holding the ${dbName} ` +
                    'IndexedDB open. Close other Obsidian windows for this vault and retry.',
                ));
            }, 10_000);
        };
    });
}

// allowRecovery: on a VersionError (the stored DB was built by a NEWER Seek with a
// higher DB_VERSION — the documented deploy-downgrade brick, seek-deploy-branch-gotcha)
// nuke the index and reopen empty instead of letting the rejection escape onload and
// brick plugin load. Bounded to ONE retry; the recursive reopen passes
// allowRecovery=false so a persistent error can't loop.
// Exported for open-recovery.test.ts — the VersionError-recovery path is only
// reachable with allowRecovery=true (IndexStore.open's default), so the test drives
// openDb directly rather than through all of open()'s post-connection setup.
export function openDb(dbName: string, allowRecovery = true): Promise<IDBDatabase> {
    return new Promise((resolve, reject) => {
        const req = indexedDB.open(dbName, DB_VERSION);
        req.onerror = () => {
            const err = req.error;
            if (allowRecovery && err instanceof DOMException && err.name === 'VersionError') {
                // Per seek-schema-bump-nuke-ok: drop + reopen; the empty-index path
                // triggers the normal first-run reindex. Match on .name only — never
                // the message, which is locale/engine-dependent.
                console.warn(`[seek] ${dbName} was built by a newer Seek — rebuilding the index`);
                deleteDbWithBlockGuard(dbName).then(
                    () => resolve(openDb(dbName, /*allowRecovery*/ false)),
                    reject,
                );
                return;
            }
            reject(err);
        };
        req.onsuccess = () => {
            const db = req.result;
            // If another window opens the DB with a higher version, the spec
            // dispatches versionchange and won't proceed until we close. The
            // alternative (ignoring it) deadlocks the schema upgrade.
            db.onversionchange = () => {
                console.warn('[seek] versionchange received — closing connection to allow schema upgrade');
                db.close();
            };
            resolve(db);
        };
        req.onupgradeneeded = (event) => {
            const db = req.result;
            const oldVersion = (event as IDBVersionChangeEvent).oldVersion;

            // v1 -> v2: vector format changed (768d Q8 -> 512d Q4). Existing
            // chunks + embeddings + files records are incompatible with the
            // new runtime, so drop them. Meta is preserved if present so the
            // upgrade can read the prior config for telemetry; setMeta will
            // overwrite it with the new dim on next reindex.
            //
            // v2 -> v3: chunk_id format changed from path-coupled to content
            // hash. The chunk store's primary key is chunk_id, so the entire
            // store is keyed by the old format and unusable. Same drop pattern.
            if (oldVersion > 0 && oldVersion < 3) {
                if (db.objectStoreNames.contains(STORE_CHUNKS)) db.deleteObjectStore(STORE_CHUNKS);
                if (db.objectStoreNames.contains(STORE_EMBEDDINGS)) db.deleteObjectStore(STORE_EMBEDDINGS);
                if (db.objectStoreNames.contains(STORE_FILES)) db.deleteObjectStore(STORE_FILES);
            }
            // v3 -> v4: ADDITIVE only. Don't touch existing stores; just
            // create the new STORE_BINARY. Backfill happens after open().

            // v4 -> v5: embedding model swap Gemma(512d,mean) -> granite-r2(384d,CLS).
            // fp32 vectors are dim-incompatible and the binary store is derived from
            // them, so drop both plus chunks+files to force a clean full reindex.
            // Meta is preserved for telemetry; setMeta overwrites it on reindex.
            //
            // v5 -> v6: rerank tier value shape fp32 Float32Array -> int8 QuantVec.
            // Old embedding records are unreadable as the new shape; binary is
            // re-derived in lockstep at reindex. Same drop set as v4->v5. The
            // `< 6` upper bound makes this branch fire for any pre-v6 origin
            // (so a v4 user upgrading straight to v6 still gets a clean drop).
            if (oldVersion >= 3 && oldVersion < 6) {
                if (db.objectStoreNames.contains(STORE_CHUNKS)) db.deleteObjectStore(STORE_CHUNKS);
                if (db.objectStoreNames.contains(STORE_EMBEDDINGS)) db.deleteObjectStore(STORE_EMBEDDINGS);
                if (db.objectStoreNames.contains(STORE_BINARY)) db.deleteObjectStore(STORE_BINARY);
                if (db.objectStoreNames.contains(STORE_FILES)) db.deleteObjectStore(STORE_FILES);
            }

            // v6/v7 -> v8: split `chunks` into `chunk_meta` + `chunk_body`. Body
            // text moves stores, so the old `chunks` records are unusable as-is;
            // drop the data stores and force a reindex (chunk_meta/chunk_body +
            // embeddings/binary repopulate in lockstep via putBatch). Origins <6
            // are already dropped by the branches above — this covers the v6/v7
            // gap. The legacy `chunks` store is removed and never recreated.
            if (oldVersion >= 6 && oldVersion < 8) {
                if (db.objectStoreNames.contains(STORE_CHUNKS)) db.deleteObjectStore(STORE_CHUNKS);
                if (db.objectStoreNames.contains(STORE_EMBEDDINGS)) db.deleteObjectStore(STORE_EMBEDDINGS);
                if (db.objectStoreNames.contains(STORE_BINARY)) db.deleteObjectStore(STORE_BINARY);
                if (db.objectStoreNames.contains(STORE_FILES)) db.deleteObjectStore(STORE_FILES);
            }

            // v8 -> v9: frontmatter values folded into the dense embed text +
            // chunk_id (chunker.ts buildDenseSuffix), so every chunk_id changes.
            // The v8 data stores are keyed by the old ids; drop them (the recreate
            // block below remakes them empty) to force one clean full re-embed.
            // bm25 (stamp-gated cache) + meta (telemetry) are intentionally kept,
            // exactly as the v6->v8 branch keeps them.
            if (oldVersion >= 8 && oldVersion < 9) {
                if (db.objectStoreNames.contains(STORE_CHUNK_META)) db.deleteObjectStore(STORE_CHUNK_META);
                if (db.objectStoreNames.contains(STORE_CHUNK_BODY)) db.deleteObjectStore(STORE_CHUNK_BODY);
                if (db.objectStoreNames.contains(STORE_EMBEDDINGS)) db.deleteObjectStore(STORE_EMBEDDINGS);
                if (db.objectStoreNames.contains(STORE_BINARY)) db.deleteObjectStore(STORE_BINARY);
                if (db.objectStoreNames.contains(STORE_FILES)) db.deleteObjectStore(STORE_FILES);
            }

            // v8: metadata and body live in separate stores. The legacy `chunks`
            // store is intentionally NOT recreated (it's dropped above for every
            // pre-v8 origin and absent on a fresh install).
            if (!db.objectStoreNames.contains(STORE_CHUNK_META)) {
                db.createObjectStore(STORE_CHUNK_META, { keyPath: 'chunk_id' });
            }
            if (!db.objectStoreNames.contains(STORE_CHUNK_BODY)) {
                db.createObjectStore(STORE_CHUNK_BODY);
            }
            if (!db.objectStoreNames.contains(STORE_EMBEDDINGS)) {
                db.createObjectStore(STORE_EMBEDDINGS);
            }
            if (!db.objectStoreNames.contains(STORE_BINARY)) {
                db.createObjectStore(STORE_BINARY);
            }
            if (!db.objectStoreNames.contains(STORE_FILES)) {
                db.createObjectStore(STORE_FILES, { keyPath: 'note_path' });
            }
            if (!db.objectStoreNames.contains(STORE_META)) {
                db.createObjectStore(STORE_META);
            }
            // v6 -> v7: ADDITIVE — persisted BM25 index cache (gated by stamp,
            // fit() fallback on any miss, so no data drop / reindex).
            if (!db.objectStoreNames.contains(STORE_BM25)) {
                db.createObjectStore(STORE_BM25);
            }
        };
    });
}

function awaitTx(tx: IDBTransaction): Promise<void> {
    return new Promise((resolve, reject) => {
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
        tx.onabort = () => reject(tx.error ?? new Error('transaction aborted'));
    });
}

function awaitRequest<T>(req: IDBRequest<T>): Promise<T> {
    return new Promise((resolve, reject) => {
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
    });
}

// Shared cursor-jump collector: walk `store` once, seeking to each wanted key
// via continue(key) in sorted order, decoding only the rows we want into a Map.
// chunk_ids are uniform content hashes, so the [min,max] of a candidate set
// spans almost the whole keyspace — a bounded-range cursor would read most of
// the store on every call; key-by-key seeking reads only the ~N wanted rows.
// The cheap shape on mobile WKWebView IDB. Shared by getEmbeddingsByIds
// (stage-2 vectors) and getBodiesByIds/getBodiesMap (lazy body text).
function collectByKeyJump<T>(
    store: IDBObjectStore,
    ids: string[],
    decode: (value: unknown) => T,
): Promise<Map<string, T>> {
    const sorted = Array.from(new Set(ids)).sort();
    const found = new Map<string, T>();
    return new Promise<Map<string, T>>((resolve, reject) => {
        if (sorted.length === 0) { resolve(found); return; }
        let i = 0;
        const cursor = store.openCursor(IDBKeyRange.lowerBound(sorted[0]));
        cursor.onsuccess = () => {
            const c = cursor.result;
            if (!c) { resolve(found); return; }
            const key = String(c.key);
            // Skip past targets that precede this row (absent from the store).
            while (i < sorted.length && sorted[i] < key) i++;
            if (i >= sorted.length) { resolve(found); return; }
            if (sorted[i] === key) {
                found.set(key, decode(c.value));
                i++;
                if (i >= sorted.length) { resolve(found); return; }
            }
            c.continue(sorted[i]);   // jump straight to the next wanted key
        };
        cursor.onerror = () => reject(cursor.error);
    });
}

// Project a Chunk to its metadata (everything except the body) for the
// chunk_meta store; the body is stored separately in chunk_body. The rest-
// capture tracks Chunk automatically as fields are added.
function stripContent(c: Chunk): ChunkMeta {
    const { content, ...meta } = c;
    void content;
    return meta;
}

export class IndexStore {
    private db: IDBDatabase | null = null;
    // Resolved per-vault DB name. Set on the first open(scope) — main.ts
    // passes the vault's appId at onload — and reused by every later
    // scope-less open() (the reset path in search.ts reindexAll).
    private _dbName: string = LEGACY_DB_NAME;
    private legacyCleanupDone = false;

    get dbName(): string { return this._dbName; }

    async open(scope?: string, dbPrefix: string = LEGACY_DB_NAME): Promise<void> {
        if (scope) this._dbName = `${dbPrefix}:${scope}`;
        this.db = await openDb(this._dbName);
        // GAP-3: openDb's onversionchange closes the connection on a cross-window
        // schema upgrade but leaves this.db pointing at the now-closed handle, so
        // every later op throws an opaque InvalidStateError. Re-wire it to also drop
        // the reference → requireDb fails cleanly ("not opened"), and a later open()
        // can rebuild. (The IndexCoordinator mutex serializes Seek's OWN writes, not
        // another vault window's versionchange.)
        const opened = this.db;
        opened.onversionchange = () => {
            console.warn('[seek] versionchange received — closing + dropping connection');
            opened.close();
            if (this.db === opened) this.db = null;
        };
        // One-time legacy cleanup: THIS build's pre-scoping shared DB (the bare,
        // appId-less prefix). Targets dbPrefix — not a hardcoded literal — so a
        // differently-id'd build only ever deletes its OWN bare legacy, never
        // another build's scoped DB. Fire-and-forget — if another window (old
        // build) still holds it, the delete stays pending until that window
        // closes; nothing here waits on it.
        if (!this.legacyCleanupDone && this._dbName !== dbPrefix) {
            this.legacyCleanupDone = true;
            try { indexedDB.deleteDatabase(dbPrefix); } catch { /* best-effort */ }
        }
    }

    close(): void {
        this.db?.close();
        this.db = null;
    }

    private requireDb(): IDBDatabase {
        if (!this.db) throw new Error(STORE_NOT_OPENED);
        return this.db;
    }

    async getMeta(): Promise<MetaConfig> {
        const db = this.requireDb();
        const tx = db.transaction(STORE_META, 'readonly');
        const store = tx.objectStore(STORE_META);
        const existing = await awaitRequest(store.get(META_KEY));
        if (existing) return existing as MetaConfig;
        return { embeddingDim: ACTIVE_MODEL_SPEC.dim, lastIndexedAt: null, schemaVersion: DB_VERSION };
    }

    async setMeta(meta: MetaConfig): Promise<void> {
        const db = this.requireDb();
        const tx = db.transaction(STORE_META, 'readwrite');
        tx.objectStore(STORE_META).put(meta, META_KEY);
        await awaitTx(tx);
    }

    // Persist the serialized MiniSearch index + its stamp (single fixed key —
    // there is one BM25 index per vault DB). Overwrites the previous blob.
    async putBm25(json: string, stamp: unknown): Promise<void> {
        const db = this.requireDb();
        const tx = db.transaction(STORE_BM25, 'readwrite');
        tx.objectStore(STORE_BM25).put({ json, stamp } satisfies Bm25Record, BM25_KEY);
        await awaitTx(tx);
    }

    // Read the persisted BM25 blob, or null if none stored yet. The caller
    // validates the stamp before MiniSearch.loadJSON; an unusable blob just
    // means the cold start refits.
    async getBm25(): Promise<Bm25Record | null> {
        const db = this.requireDb();
        const tx = db.transaction(STORE_BM25, 'readonly');
        const rec = await awaitRequest(tx.objectStore(STORE_BM25).get(BM25_KEY));
        return rec ? (rec as Bm25Record) : null;
    }

    // Bulk write: one transaction per call (matches design-doc guidance —
    // bulk writes inside a single readwrite transaction are ~10× faster
    // than one transaction per put on Safari/WKWebView).
    //
    // Writes all four stores (chunk_meta + chunk_body + int8 rerank + sign-bit
    // binary) in the same transaction so they can never diverge — every chunk
    // has its body, int8 vector, and packed binary sibling at the same chunk_id,
    // atomically. The v8 split (meta vs body) rides inside that same guarantee.
    //
    // Order matters: the sign-bit binary is packed from the TRUE fp32 vector
    // (before quantization) so the candidate tier carries the model's exact
    // signs, not int8-rounded ones. Sign is preserved under SQ8 anyway, but a
    // near-zero component can round across 0 — deriving from fp32 sidesteps that.
    async putBatch(chunks: Chunk[], vectors: Float32Array[]): Promise<void> {
        if (chunks.length === 0) return;
        if (chunks.length !== vectors.length) {
            throw new Error(`chunks/vectors length mismatch: ${chunks.length} vs ${vectors.length}`);
        }
        const db = this.requireDb();
        const tx = db.transaction([STORE_CHUNK_META, STORE_CHUNK_BODY, STORE_EMBEDDINGS, STORE_BINARY], 'readwrite');
        const metaStore = tx.objectStore(STORE_CHUNK_META);
        const bodyStore = tx.objectStore(STORE_CHUNK_BODY);
        const embStore = tx.objectStore(STORE_EMBEDDINGS);
        const binStore = tx.objectStore(STORE_BINARY);
        for (let i = 0; i < chunks.length; i++) {
            metaStore.put(stripContent(chunks[i]));
            bodyStore.put(chunks[i].content ?? '', chunks[i].chunk_id);
            binStore.put(packSignBits(vectors[i]), chunks[i].chunk_id);
            embStore.put(quantizeInt8(vectors[i]), chunks[i].chunk_id);
        }
        await awaitTx(tx);
    }

    // Hydration write path (sidecar): the (QuantVec, sign-bit) tiers already
    // exist — decoded verbatim from a sidecar shard another device (or this
    // device, pre-eviction) produced — so write them as-is with NO re-quantize.
    // Mirrors putBatch's exact 4-store atomic transaction and per-i order. The
    // fp32-sign-fidelity invariant putBatch enforces (binary packed from TRUE
    // fp32) is preserved transitively: the producer derived these bytes from its
    // fp32 vector at commit time and we copy them unchanged.
    async putBatchQuantized(chunks: Chunk[], tiers: { q: QuantVec; bin: Uint8Array }[]): Promise<void> {
        if (chunks.length === 0) return;
        if (chunks.length !== tiers.length) {
            throw new Error(`chunks/tiers length mismatch: ${chunks.length} vs ${tiers.length}`);
        }
        const db = this.requireDb();
        const tx = db.transaction([STORE_CHUNK_META, STORE_CHUNK_BODY, STORE_EMBEDDINGS, STORE_BINARY], 'readwrite');
        const metaStore = tx.objectStore(STORE_CHUNK_META);
        const bodyStore = tx.objectStore(STORE_CHUNK_BODY);
        const embStore = tx.objectStore(STORE_EMBEDDINGS);
        const binStore = tx.objectStore(STORE_BINARY);
        for (let i = 0; i < chunks.length; i++) {
            metaStore.put(stripContent(chunks[i]));
            bodyStore.put(chunks[i].content ?? '', chunks[i].chunk_id);
            binStore.put(tiers[i].bin, chunks[i].chunk_id);
            embStore.put(tiers[i].q, chunks[i].chunk_id);
        }
        await awaitTx(tx);
    }

    async putFileRecord(rec: FileRecord): Promise<void> {
        const db = this.requireDb();
        const tx = db.transaction(STORE_FILES, 'readwrite');
        tx.objectStore(STORE_FILES).put(rec);
        await awaitTx(tx);
    }

    async getFileRecord(notePath: string): Promise<FileRecord | undefined> {
        const db = this.requireDb();
        const tx = db.transaction(STORE_FILES, 'readonly');
        return await awaitRequest(tx.objectStore(STORE_FILES).get(notePath)) as FileRecord | undefined;
    }

    // Every persisted file record, for the startup mtime-diff sweep (compare
    // each record's mtimeMs against the live TFile.stat.mtime to find what
    // changed while Seek wasn't watching). Mirror of listAllChunks.
    async listFileRecords(): Promise<FileRecord[]> {
        const db = this.requireDb();
        const tx = db.transaction(STORE_FILES, 'readonly');
        return await awaitRequest(tx.objectStore(STORE_FILES).getAll()) as FileRecord[];
    }

    // Every chunk_id in the index (KEYS only — far lighter than listAllMeta's full
    // records). The referential-integrity sweep diffs this against the union of
    // FILES-record chunk_ids to find orphans.
    async getAllChunkIds(): Promise<string[]> {
        const db = this.requireDb();
        const tx = db.transaction(STORE_CHUNK_META, 'readonly');
        return await awaitRequest(tx.objectStore(STORE_CHUNK_META).getAllKeys()) as string[];
    }

    // Delete a set of chunk_ids across all four data tiers in ONE transaction. The
    // FILES store is intentionally untouched — orphans are unreferenced by
    // definition. The sweep's delete primitive; mirrors deleteFile's atomicity
    // (the tiers can't diverge) without needing a note_path.
    async deleteChunksByIds(ids: string[]): Promise<void> {
        if (ids.length === 0) return;
        const db = this.requireDb();
        const tx = db.transaction(
            [STORE_CHUNK_META, STORE_CHUNK_BODY, STORE_EMBEDDINGS, STORE_BINARY],
            'readwrite',
        );
        const metaStore = tx.objectStore(STORE_CHUNK_META);
        const bodyStore = tx.objectStore(STORE_CHUNK_BODY);
        const embStore = tx.objectStore(STORE_EMBEDDINGS);
        const binStore = tx.objectStore(STORE_BINARY);
        for (const id of ids) {
            metaStore.delete(id);
            bodyStore.delete(id);
            embStore.delete(id);
            binStore.delete(id);
        }
        await awaitTx(tx);
    }

    // Incremental delete: drop one note's chunks across all four data tiers
    // (chunk_meta + chunk_body + int8 rerank + sign-bit binary) plus its file
    // record, so the index no longer surfaces it. The inverse of putBatch +
    // putFileRecord; used by reindexDelta for deletes and the drop-half of a
    // rename/move.
    //
    // Reads the file record to learn the chunk_ids, then deletes everything in
    // ONE readwrite transaction spanning all five stores so the tiers can't
    // diverge (same atomicity guarantee putBatch gives on write). No-op (and no
    // tx) if the path was never indexed — deletes can fire for notes Seek never
    // saw (excluded folders, sub-min-chars files).
    //
    // Returns the chunk_ids removed ([] = path wasn't indexed). The ids are the
    // delete half of the change-set the incremental cache path (search.ts
    // applyDelta) tombstones; callers wanting a count take .length.
    async deleteFile(notePath: string): Promise<string[]> {
        const db = this.requireDb();
        const rec = await this.getFileRecord(notePath);
        if (!rec || rec.chunk_ids.length === 0) {
            // Still drop a stray file record with no chunks, if one exists.
            if (rec) {
                const tx = db.transaction(STORE_FILES, 'readwrite');
                tx.objectStore(STORE_FILES).delete(notePath);
                await awaitTx(tx);
            }
            return [];
        }
        const tx = db.transaction(
            [STORE_CHUNK_META, STORE_CHUNK_BODY, STORE_EMBEDDINGS, STORE_BINARY, STORE_FILES],
            'readwrite',
        );
        const metaStore = tx.objectStore(STORE_CHUNK_META);
        const bodyStore = tx.objectStore(STORE_CHUNK_BODY);
        const embStore = tx.objectStore(STORE_EMBEDDINGS);
        const binStore = tx.objectStore(STORE_BINARY);
        for (const id of rec.chunk_ids) {
            metaStore.delete(id);
            bodyStore.delete(id);
            embStore.delete(id);
            binStore.delete(id);
        }
        tx.objectStore(STORE_FILES).delete(notePath);
        await awaitTx(tx);
        return rec.chunk_ids;
    }

    // Full-corpus metadata read (v8 frame-lite): every chunk's metadata, WITHOUT
    // body text — the cold-frame-rebuild input. Replaces the old listAllChunks,
    // which dragged the whole body corpus into RAM on every cache miss; bodies
    // now come from getBodiesByIds/getBodiesMap on demand (snippets, negation,
    // BM25 refit). Reads only the small chunk_meta store.
    async listAllMeta(): Promise<ChunkMeta[]> {
        const db = this.requireDb();
        const tx = db.transaction(STORE_CHUNK_META, 'readonly');
        return await awaitRequest(tx.objectStore(STORE_CHUNK_META).getAll()) as ChunkMeta[];
    }

    // Stage-1 read: pull every chunk's packed sign-bit vector. Cheap (~64 KB
    // per 1k chunks at d=512) and intended to be cached in memory by the
    // search orchestrator across queries — see search.ts binaryIndexCache.
    // The cursor walk produces ids and per-chunk Uint8Arrays in IDB order;
    // callers concatenate them into one contiguous buffer for scoring.
    async listAllBinary(): Promise<{ ids: string[]; packed: Uint8Array[] }> {
        const db = this.requireDb();
        const tx = db.transaction(STORE_BINARY, 'readonly');
        const store = tx.objectStore(STORE_BINARY);
        const ids: string[] = [];
        const packed: Uint8Array[] = [];
        await new Promise<void>((resolve, reject) => {
            const cursor = store.openCursor();
            cursor.onsuccess = () => {
                const c = cursor.result;
                if (!c) { resolve(); return; }
                ids.push(String(c.key));
                packed.push(c.value as Uint8Array);
                c.continue();
            };
            cursor.onerror = () => reject(cursor.error);
        });
        return { ids, packed };
    }

    // Bulk read of the int8 rerank tier: every {chunk_id → QuantVec}. Mirrors
    // listAllBinary — one readonly cursor, raw {q,s} (NO dequant). Consumed once
    // per dataGeneration by ensureFrame() (search.ts) to assemble the resident
    // rerank block held in RAM, so stage-2 can dequantize candidates without a
    // per-keystroke IDB round-trip. The resident block dequantizes on demand
    // with the SAME dequantizeInt8 that getEmbeddingsByIds uses below, so the
    // two paths are bit-identical — only the byte source (RAM vs IDB) differs.
    async listAllEmbeddings(): Promise<{ ids: string[]; vecs: QuantVec[] }> {
        const db = this.requireDb();
        const tx = db.transaction(STORE_EMBEDDINGS, 'readonly');
        const store = tx.objectStore(STORE_EMBEDDINGS);
        const ids: string[] = [];
        const vecs: QuantVec[] = [];
        await new Promise<void>((resolve, reject) => {
            const cursor = store.openCursor();
            cursor.onsuccess = () => {
                const c = cursor.result;
                if (!c) { resolve(); return; }
                ids.push(String(c.key));
                vecs.push(c.value as QuantVec);
                c.continue();
            };
            cursor.onerror = () => reject(cursor.error);
        });
        return { ids, vecs };
    }

    // Phase-0 diagnostic (read-only, no schema touch): walk every store once and
    // sum the LOGICAL bytes of each row by the store's value shape (see
    // index-size.ts for the rules). One readonly cursor per store — the same
    // listAllBinary idiom — so it never holds the whole index in memory. The
    // caller (main.ts) pairs the returned logical total with
    // navigator.storage.estimate() so physical − logical = LevelDB slack.
    //
    // The legacy `chunks` store is intentionally omitted: it's deleted on the v8
    // upgrade and absent on any live index. A store missing from this DB version
    // is skipped (objectStoreNames guard) rather than throwing.
    async measureSizes(): Promise<{ stores: StoreSizeRow[]; logicalBytes: number }> {
        const db = this.requireDb();
        const plan: { store: string; label: string; rule: SizingRule }[] = [
            { store: STORE_BM25,       label: 'BM25 inverted index', rule: 'bm25' },
            { store: STORE_CHUNK_BODY, label: 'chunk bodies',        rule: 'utf8' },
            { store: STORE_CHUNK_META, label: 'chunk metadata',      rule: 'json' },
            { store: STORE_EMBEDDINGS, label: 'int8 vectors',        rule: 'quantvec' },
            { store: STORE_BINARY,     label: 'binary sign-bits',    rule: 'bytes' },
            { store: STORE_FILES,      label: 'file records',        rule: 'json' },
            { store: STORE_META,       label: 'meta/config',         rule: 'json' },
        ];
        const stores: StoreSizeRow[] = [];
        let logicalBytes = 0;
        for (const p of plan) {
            if (!db.objectStoreNames.contains(p.store)) continue;
            let rows = 0;
            let bytes = 0;
            const tx = db.transaction(p.store, 'readonly');
            const cursor = tx.objectStore(p.store).openCursor();
            await new Promise<void>((resolve, reject) => {
                cursor.onsuccess = () => {
                    const c = cursor.result;
                    if (!c) { resolve(); return; }
                    rows++;
                    bytes += sizeOfRow(p.rule, c.value);
                    c.continue();
                };
                cursor.onerror = () => reject(cursor.error);
            });
            stores.push({ store: p.store, label: p.label, rows, bytes });
            logicalBytes += bytes;
        }
        return { stores, logicalBytes };
    }

    // Embed-free compaction. RETAINED BUT UNWIRED — no UI/CLI/orchestrator surface
    // reaches it (removed 2026-06-20 after a live test disproved its premise). It is
    // kept only as a tested mechanism for a possible future iOS-gated experiment.
    //
    // Premise (DISPROVEN on desktop): snapshot → deleteDatabase → rewrite would shed
    // the LevelDB slack incremental writes accrue. Reality: Chromium backs IndexedDB
    // with LevelDB, which frees only the LOGICAL keyspace on deleteDatabase and
    // compacts dead SSTables on its own size-triggered schedule — never from web
    // code. A live run GREW the store 60 → 72 MB (the rewrite adds a fresh generation
    // and reclaims nothing). So this does NOT reclaim physical disk on desktop;
    // desktop slack is harmless (40 GB quota) and not web-fixable. iOS (WKWebView,
    // SQLite-backed) MIGHT drop the per-DB file on deleteDatabase and actually
    // reclaim — UNTESTED — which is the only reason this code survives.
    // Mechanism (data-safe, search-identical, ~0.9 s for ~21k records):
    //   1. snapshot every store into RAM (the whole logical index is ~12–18 MB)
    //   2. close + `nukeDatabase` (frees logical keyspace; physical only on iOS, if)
    //   3. reopen (recreates empty stores at DB_VERSION)
    //   4. rewrite the identical records into the fresh DB, one atomic tx
    // The logical index is byte-for-byte unchanged, so callers need NOT re-embed,
    // re-warm caches, or bump dataGeneration.
    //
    // Concurrency: the DB is CLOSED between steps 2–3, so any future caller MUST run
    // this under the orchestrator's write mutex exactly as reindexAll does — a
    // delta/search hitting requireDb() mid-window would throw.
    //
    // Crash window: a process death between deleteDatabase and the rewrite commit
    // loses the index (the RAM snapshot dies with it). It is fully recoverable —
    // a reindex on desktop, a sidecar-hydrate on mobile (the vectors live in the
    // sidecar too) — and the window is one bulk write of ~15 MB, but it is real.
    async compact(): Promise<{ records: number; stores: number }> {
        const db = this.requireDb();

        // 1. Snapshot. One readonly cursor per store, capturing primaryKey + value.
        const snapshot: StoreSnapshot[] = [];
        for (const spec of COMPACTION_STORES) {
            if (!db.objectStoreNames.contains(spec.store)) continue;
            const records: { key: IDBValidKey; value: unknown }[] = [];
            const tx = db.transaction(spec.store, 'readonly');
            const cursor = tx.objectStore(spec.store).openCursor();
            await new Promise<void>((resolve, reject) => {
                cursor.onsuccess = () => {
                    const c = cursor.result;
                    if (!c) { resolve(); return; }
                    records.push({ key: c.primaryKey, value: c.value });
                    c.continue();
                };
                cursor.onerror = () => reject(cursor.error);
            });
            snapshot.push({ store: spec.store, inlineKey: spec.inlineKey, records });
        }
        const expected = snapshot.reduce((n, s) => n + s.records.length, 0);

        // 2–3. Drop the bloated DB and reopen a fresh, empty one. Mirrors
        // reindexAllInner's reset dance (close so deleteDatabase isn't self-blocked).
        this.close();
        await nukeDatabase(this._dbName);
        await this.open();   // scope-less: reuses _dbName from the onload open(appId)

        // 4. Rewrite. One readwrite transaction over every store = atomic (a failure
        // aborts the whole rewrite, leaving the snapshot in RAM) and the fast bulk
        // path on WKWebView. In-line-keyed stores omit the explicit key.
        const fresh = this.requireDb();
        const storeNames = snapshot.map(s => s.store);
        if (storeNames.length > 0) {
            const tx = fresh.transaction(storeNames, 'readwrite');
            for (const op of planRestoreOps(snapshot)) {
                const os = tx.objectStore(op.store);
                if (op.key === undefined) os.put(op.value);
                else os.put(op.value, op.key);
            }
            await awaitTx(tx);
        }

        // Belt-and-braces: the rewrite tx is atomic, but a count mismatch would flag
        // a snapshot/key bug — surface it loudly rather than leave a thinned index.
        let restored = 0;
        if (storeNames.length > 0) {
            const tx = fresh.transaction(storeNames, 'readonly');
            const counts = await Promise.all(storeNames.map(s => awaitRequest(tx.objectStore(s).count())));
            restored = counts.reduce((a, b) => a + b, 0);
        }
        if (restored !== expected) {
            throw new Error(`compaction count mismatch: snapshot ${expected} vs restored ${restored} — run a full reindex`);
        }
        return { records: expected, stores: snapshot.length };
    }

    // Stage-2 read: pull rerank vectors for just the candidate chunk_ids the
    // binary stage picked, dequantizing int8 → fp32 so callers see plain
    // Float32Array (the search pipeline is unchanged by the storage format).
    // Returns vectors aligned to the input `ids` order; missing ids get `null`
    // (defensive — a candidate id without a stored sibling means data
    // corruption, but we'd rather skip-and-continue than throw mid-search).
    //
    // One readonly cursor that JUMPS to each wanted key (collectByKeyJump) — not
    // N independent .get()s, and not a range scan over uniform hash keys. This
    // is now the ALWAYS path on mobile (the B2 resident-block gate in search.ts
    // disables the RAM tier there). dequantizeInt8 is unchanged, so the output
    // is byte-identical to the old per-get path: aligned to input order,
    // duplicates resolved from the map, missing ids → null.
    async getEmbeddingsByIds(ids: string[]): Promise<Array<Float32Array | null>> {
        if (ids.length === 0) return [];
        const db = this.requireDb();
        const tx = db.transaction(STORE_EMBEDDINGS, 'readonly');
        const found = await collectByKeyJump(
            tx.objectStore(STORE_EMBEDDINGS),
            ids,
            v => { const rec = v as QuantVec; return dequantizeInt8(rec.q, rec.s); },
        );
        return ids.map(id => found.get(id) ?? null);
    }

    // Lazy body fetch (v8 frame-lite): the resident frame is metadata-only, so
    // body text for the ≤topK results (snippets + ScoredChunk hydration) is read
    // on demand here. Same cursor-jump shape as getEmbeddingsByIds; aligned to
    // input order, missing ids → null.
    async getBodiesByIds(ids: string[]): Promise<Array<string | null>> {
        if (ids.length === 0) return [];
        const db = this.requireDb();
        const tx = db.transaction(STORE_CHUNK_BODY, 'readonly');
        const found = await collectByKeyJump(tx.objectStore(STORE_CHUNK_BODY), ids, v => v as string);
        return ids.map(id => found.get(id) ?? null);
    }

    // Body text for a set of ids as a Map (present ids only). Used by BM25 fit()
    // and `-term` negation, which key bodies by chunk_id rather than position.
    // The bodies are transient at the call site (folded into the inverted index
    // / a token scan, not retained), so a full-corpus fetch here is acceptable on
    // the rare refit / negation path; warm search never calls it.
    async getBodiesMap(ids: string[]): Promise<Map<string, string>> {
        if (ids.length === 0) return new Map();
        const db = this.requireDb();
        const tx = db.transaction(STORE_CHUNK_BODY, 'readonly');
        return await collectByKeyJump(tx.objectStore(STORE_CHUNK_BODY), ids, v => v as string);
    }

    // Carry-over read (F13): for a set of chunk_ids, pull each chunk's text plus
    // its RAW rerank + sign tiers (NO dequant) in one readonly transaction. Used
    // to harvest the vectors of chunks about to be deleted so a move / no-op
    // re-flush can re-key the IDENTICAL vector under the new (path-salted) chunk_id
    // instead of re-running the model — re-deriving from getEmbeddingsByIds' fp32
    // would NOT be byte-identical (it dequantizes). Missing ids → null. Issues all
    // 4N gets synchronously up front so the readonly tx can't auto-commit mid-read.
    //
    // The chunk text is REASSEMBLED from the v8 split: chunk_meta (everything but
    // the body) + chunk_body (the content), exactly mirroring putBatch's write.
    // The caller keys carry-over by embedInput (title\n\ncontent\n\ndenseSuffix),
    // so the body is mandatory — a null here just means that chunk re-embeds
    // normally. (This previously read the single pre-v8 `chunks` store, which the
    // v8 upgrade deletes and never recreates; reading it threw NotFoundError and
    // aborted every warm-model carry-over delta.)
    async getTiersByIds(ids: string[]): Promise<Array<{ chunk: Chunk; q: QuantVec; sign: Uint8Array } | null>> {
        if (ids.length === 0) return [];
        const db = this.requireDb();
        const tx = db.transaction([STORE_CHUNK_META, STORE_CHUNK_BODY, STORE_EMBEDDINGS, STORE_BINARY], 'readonly');
        const metaStore = tx.objectStore(STORE_CHUNK_META);
        const bodyStore = tx.objectStore(STORE_CHUNK_BODY);
        const embStore = tx.objectStore(STORE_EMBEDDINGS);
        const binStore = tx.objectStore(STORE_BINARY);
        const metaReqs = ids.map(id => awaitRequest(metaStore.get(id)));
        const bodyReqs = ids.map(id => awaitRequest(bodyStore.get(id)));
        const qReqs = ids.map(id => awaitRequest(embStore.get(id)));
        const signReqs = ids.map(id => awaitRequest(binStore.get(id)));
        const [metas, bodies, qs, signs] = await Promise.all([
            Promise.all(metaReqs), Promise.all(bodyReqs), Promise.all(qReqs), Promise.all(signReqs),
        ]);
        return ids.map((_, i) => {
            const meta = metas[i] as ChunkMeta | undefined;
            const body = bodies[i] as string | undefined;
            const q = qs[i] as QuantVec | undefined;
            const sign = signs[i] as Uint8Array | undefined;
            // body === undefined is the only miss (an empty-content chunk stores '',
            // which is a valid body — embedInput re-creates `title\n\n` for it).
            if (!meta || body === undefined || !q || !sign) return null;
            return { chunk: { ...meta, content: body }, q, sign };
        });
    }

    // Lazy backfill: if the binary store has fewer rows than the fp32 store,
    // pack the missing ones. Runs after open() on plugin start (only does
    // real work once, after a v3→v4 upgrade where the binary store was newly
    // created against an already-populated fp32 store).
    //
    // Two-phase to sidestep IDB's read-vs-write transaction serialization:
    //   1. Drain fp32 inside one readonly cursor, packing bytes into RAM as
    //      we go. Awaiting a writeback inside the cursor would deadlock —
    //      the open cursor holds the read-tx, and the writeback can't
    //      proceed until the read-tx closes, which can't happen while the
    //      cursor is still awaited.
    //   2. After the cursor closes, write back in fixed-size batches.
    //
    // Memory ceiling on phase 1 is ~n × (id ~60 B + bytesPerVec ~64 B) =
    // ~700 KB at 5.5k chunks — well within the iOS Path-A 50 MB jetsam
    // ceiling discussed in the header comment.
    //
    // Returns the number of packed rows written (0 = no work needed, the
    // common steady-state case).
    async backfillBinaryIfMissing(onProgress?: (done: number) => void): Promise<number> {
        const db = this.requireDb();
        const counts = await this.count();
        if (counts.binary >= counts.embeddings) return 0;

        // Phase 1: drain the rerank tier → packed sign-bytes in memory. Records
        // are int8 QuantVec now, so dequant before packing (sign is preserved
        // under SQ8, but going through the same packSignBits keeps one code path).
        const pending: Array<{ id: string; bytes: Uint8Array }> = [];
        const rtx = db.transaction(STORE_EMBEDDINGS, 'readonly');
        const rstore = rtx.objectStore(STORE_EMBEDDINGS);
        await new Promise<void>((resolve, reject) => {
            const cursor = rstore.openCursor();
            cursor.onsuccess = () => {
                const c = cursor.result;
                if (!c) { resolve(); return; }
                const rec = c.value as QuantVec;
                pending.push({
                    id: String(c.key),
                    bytes: packSignBits(dequantizeInt8(rec.q, rec.s)),
                });
                c.continue();
            };
            cursor.onerror = () => reject(cursor.error);
        });

        // Phase 2: chunked writeback. Each batch is one transaction so a mid-
        // backfill crash leaves a partially-populated binary store (which
        // the next open() will resume from — the count guard at the top makes
        // backfill idempotent).
        const BACKFILL_BATCH = 200;
        let written = 0;
        for (let i = 0; i < pending.length; i += BACKFILL_BATCH) {
            const slice = pending.slice(i, i + BACKFILL_BATCH);
            const wtx = db.transaction(STORE_BINARY, 'readwrite');
            const wstore = wtx.objectStore(STORE_BINARY);
            for (const { id, bytes } of slice) wstore.put(bytes, id);
            await awaitTx(wtx);
            written += slice.length;
            onProgress?.(written);
        }
        return written;
    }

    async count(): Promise<{ chunks: number; embeddings: number; binary: number; files: number }> {
        const db = this.requireDb();
        const tx = db.transaction([STORE_CHUNK_META, STORE_EMBEDDINGS, STORE_BINARY, STORE_FILES], 'readonly');
        const [chunks, embeddings, binary, files] = await Promise.all([
            awaitRequest(tx.objectStore(STORE_CHUNK_META).count()),
            awaitRequest(tx.objectStore(STORE_EMBEDDINGS).count()),
            awaitRequest(tx.objectStore(STORE_BINARY).count()),
            awaitRequest(tx.objectStore(STORE_FILES).count()),
        ]);
        return { chunks, embeddings, binary, files };
    }
}

// Drop the entire database (used by Full Reindex). Cleaner than iterating
// stores: `indexedDB.deleteDatabase` releases all space atomically.
//
// IMPORTANT: every IDBDatabase connection to seek-index MUST be closed
// before calling this — `deleteDatabase` waits for outstanding connections,
// and if none close, `onblocked` fires forever. The orchestrator handles
// this by closing its persistent store handle before invoking nuke and
// re-opening after.
//
// Returns pre-deletion counts so the reset log entry has something to report.
export async function nukeDatabase(dbName: string): Promise<{ chunks: number; embeddings: number; binary: number; files: number }> {
    let preCount = { chunks: 0, embeddings: 0, binary: 0, files: 0 };

    // Open a transient connection, get counts, close it before deleteDatabase.
    // versionchange listener is a belt-and-braces measure: if some other tab
    // is holding the DB open, this lets us nudge it to release.
    try {
        // allowRecovery=false: keep VersionError recovery scoped to the primary
        // IndexStore.open() path. nukeDatabase is about to delete this DB anyway, so
        // a recover-then-delete here would be a pointless double-delete race.
        const db = await openDb(dbName, /*allowRecovery*/ false);
        try {
            const tx = db.transaction([STORE_CHUNK_META, STORE_EMBEDDINGS, STORE_BINARY, STORE_FILES], 'readonly');
            const [chunks, embeddings, binary, files] = await Promise.all([
                awaitRequest(tx.objectStore(STORE_CHUNK_META).count()),
                awaitRequest(tx.objectStore(STORE_EMBEDDINGS).count()),
                awaitRequest(tx.objectStore(STORE_BINARY).count()),
                awaitRequest(tx.objectStore(STORE_FILES).count()),
            ]);
            preCount = { chunks, embeddings, binary, files };
        } finally {
            db.close();
        }
    } catch {
        // DB may not exist yet — fine, counts stay zero.
    }

    await deleteDbWithBlockGuard(dbName);
    return preCount;
}
