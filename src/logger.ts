// Verbose NDJSON logger + markdown report generator.
//
// Modeled directly on the embeddinggemmaiostest spike's SpikeLogger. The
// rationale is the same: write every interesting event as a JSON line to
// a vault-root file so it's portable, diffable, and survives an Obsidian
// crash (in contrast to console.log, which is unreachable on mobile).
//
// Layout (PER-DEVICE as of schema v9). The machine-written data streams live in
// a hidden folder next to the index sidecar (LOG_DIR below) so they don't clutter
// the vault file explorer; only the human-readable report stays at the vault root
// so it remains openable in Obsidian's UI:
//   .obsidian/plugins/seek/logs/seek-log-<deviceId>.ndjson      — append-only stream of LogEntry rows
//   .obsidian/plugins/seek/logs/seek-init-<deviceId>.json       — overwritten each load with last init payload
//   .obsidian/plugins/seek/logs/seek-captures-<deviceId>.ndjson — LEGACY relevance-debug captures; no longer written, only swept in from the vault root
//   seek-report.json                                            — generated on demand: full structured diagnostic (all devices merged), the parse target
//   seek-report.md                                              — generated on demand: ~20-line human summary pointing at seek-report.json; kept at vault root so it can be opened
//   seek-log.ndjson                                             — LEGACY pre-v9 shared file; migrated into LOG_DIR, read into the
//                                                                 report (attributed to deviceId 'legacy'), never written
//
// Why per-device files: the vault is iCloud-synced, and iCloud does whole-file
// last-writer-wins sync, NOT append-merge. Two devices appending to one shared
// seek-log.ndjson silently clobber each other (a phone's run vanishes when the
// desktop's copy wins the sync). Giving each device its own file (keyed by a
// localStorage-backed deviceId) means no device ever writes another's file, so
// concurrent appends can't collide. The report reads ALL device files and merges
// by timestamp, scoping device-identity sections (Init/Platform/Loads) to the
// current sessionId so they describe the generating device, not whoever synced last.
//
// The init file is still split out because if a log ndjson is truncated mid-write,
// the last init snapshot survives for diagnostics. Same trick the iOS spike uses.

import type { App } from 'obsidian';
import type {
    LogEntry, LogMeta, InitEntry, PlatformEntry, LoadEntry,
    IndexCompleteEntry, IndexProgressEntry, SearchEntry, ResetEntry, ErrorEntry,
    LongTaskEntry, MemoryPressureEntry, StorageSnapshotEntry, DistributionStats,
    ClickEntry, EvictionSuspectedEntry, AppLocalFetchEntry, EmbedProfileEntry,
    CrashDetectedEntry,
} from './types';
import { LOG_SCHEMA_VERSION } from './types';
import { isMobilePlatform } from './platform';

// Hidden home for the machine-written data streams, alongside the index sidecar.
// Pinned to the LITERAL '.obsidian' (not vault.configDir, the per-device active
// override) for the same reason the sidecar is: every device then resolves the
// SAME directory, so iCloud syncs all devices' per-device logs into one folder
// the report can list in a single pass. Invisible in the file explorer because
// it lives under the config folder.
// Per-INSTANCE (plugin-id-scoped) so a co-installed build writes its logs into
// its OWN plugin folder, never a sibling's. id 'seek' → the historical
// '.obsidian/plugins/seek/logs', so a shipped build's log location is unchanged.
function logDirFor(pluginId: string): string { return `.obsidian/plugins/${pluginId}/logs`; }
// The report is the only shared (single-writer-at-a-time, full-overwrite) file,
// and the only one kept at the vault ROOT — it must stay a real vault file so the
// "Generate logging report" command can open it (getAbstractFileByPath only
// resolves files outside the config folder). Safe under iCloud because it's never
// appended to from two devices at once.
const REPORT_PATH = 'seek-report.md';
// The full structured diagnostic, written alongside the .md summary on each report
// generation. One JSON object (a metadata header + a flat, type-tagged `entries`
// array) — the parse target for jq / pandas; the .md is just a human glance over it.
const REPORT_JSON_PATH = 'seek-report.json';
// Per-type recency caps for the generated report (NOT the raw NDJSON, which keeps
// everything and is bounded separately by rotateIfOversize). The report is a recent-
// activity snapshot kept small enough to email + parse fast; high-volume types keep
// their most recent N, while types ABSENT here (crash-detected, init, platform, reset,
// model-*, webgpu-event) are kept in full — rare and diagnostically critical. Without
// this the report is the entire history across every device (15+ MB after two weeks,
// ~82% of it search traces). entryCount vs includedCount + the `caps` field make any
// truncation explicit — no silent capping.
const REPORT_CAPS: Record<string, number> = {
    search: 150,
    error: 300,
    'index-progress': 50,
    'index-complete': 100,
    'sidecar-hydrate': 50,
    'memory-pressure': 100,
    'long-task': 100,
    'storage-snapshot': 50,
    'app-local-fetch': 50,
    click: 100,
    load: 50,
};
// Legacy single-file log from before per-device files (schema ≤ v8). Still read
// into the report (its entries are attributed to deviceId 'legacy') so history
// isn't lost; never written to again. The basename is matched both at the vault
// root (pre-move) and inside LOG_DIR (post-move).
const LEGACY_LOG_BASE = 'seek-log.ndjson';
const LOG_PREFIX = 'seek-log-';     // per-device: seek-log-<deviceId>.ndjson
const INIT_PREFIX = 'seek-init-';   // per-device: seek-init-<deviceId>.json
const CAPTURE_PREFIX = 'seek-captures-';
// Append-only logs have no natural ceiling. On load (rotateIfOversize, after the
// root-file migration) a per-device log past MAX_LOG_BYTES is tail-truncated to the
// most recent KEEP_LOG_BYTES, snapped forward to a newline so the retained head is a
// whole JSON line. Byte budgets are approximate (a stat size in bytes is compared,
// then the string is sliced by JS length) — fine for a coarse retention cap.
const MAX_LOG_BYTES = 1 * 1024 * 1024;   // rotate once a device log passes ~1 MB
const KEEP_LOG_BYTES = 512 * 1024;       // …retaining ~the most recent 512 KB (file stays bounded at ~1 MB)
// Errors are deduped by MESSAGE (the failure identity) — NOT context+message, because
// the same fault is logged from many call sites (an "iframe not initialized" storm
// carries ~hundreds of distinct contexts but one message, so context+message would
// barely collapse it). The first occurrence is written in full (with stack); after that
// only exponential milestones (counts 2,4,8,…) are written, each carrying the running
// `repeated` total — so a chronic fault that fires thousands of times costs ~log2(N)
// rows, not N. flushErrorAggregates emits the exact final tally at report time. Keying
// and counting are memory-only (per session); the first occurrence is always persisted.
const ERROR_KEYS_MAX = 512;              // backstop: the distinct-message dedup map never grows past this
const ERROR_KEY_MAXLEN = 256;            // cap the dedup key length (messages can embed long paths/ids)
// Orphan-log GC: a log file OTHER than this device's whose newest entry is older than
// this is treated as abandoned and removed on load (rotateIfOversize only caps the
// CURRENT device's file). Age is read from content, not mtime — iCloud re-stamps mtime.
const ORPHAN_LOG_MAX_AGE_MS = 30 * 24 * 60 * 60 * 1000;   // 30 days
// localStorage key holding this install's stable device id. localStorage is
// device-local (NOT vault-synced) and survives plugin reloads — exactly the
// scope we need: desktop and phone get distinct ids even on the same vault.
const DEVICE_ID_KEY = 'seek-device-id-v1';

// crypto.randomUUID with a Math.random fallback (mirrors iframe-runner). Used
// for the device id (once) and the per-load session id.
function randId(): string {
    const c = crypto as { randomUUID?: () => string };
    if (c.randomUUID) return c.randomUUID();
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, ch => {
        const r = (Math.random() * 16) | 0;
        return (ch === 'x' ? r : (r & 0x3) | 0x8).toString(16);
    });
}

// Stable per-install device id, e.g. "mobile-9f3a1c08" / "desktop-2b7e54aa".
// The platform prefix makes log filenames + report attribution human-readable
// at a glance ("which file is the phone?") without parsing a UA.
function resolveDeviceId(): string {
    try {
        const existing = localStorage.getItem(DEVICE_ID_KEY);
        if (existing) return existing;
    } catch { /* localStorage unavailable — fall through to ephemeral id */ }
    const id = `${isMobilePlatform() ? 'mobile' : 'desktop'}-${randId().replace(/-/g, '').slice(0, 8)}`;
    try { localStorage.setItem(DEVICE_ID_KEY, id); } catch { /* best-effort */ }
    return id;
}

// The structured payload serialized to seek-report.json. A small metadata header
// plus the full merged firehose as one flat, type-tagged array — deliberately not
// pre-grouped, so a consumer filters by `entry.type` (jq / pandas) however they like.
interface ReportData {
    generated: string;
    schemaVersion: number;
    thisDevice: string;
    thisSession: string;
    entryCount: number;
    includedCount: number;
    firstTimestamp: string | null;
    lastTimestamp: string | null;
    devices: Array<{ id: string; count: number }>;
    caps: Record<string, number>;
    entries: LogEntry[];
}

export class SeekLogger {
    private app: App;
    // Stable across reloads (localStorage); identifies the physical device.
    readonly deviceId: string;
    // Fresh per plugin load; identifies this run so the report can show only
    // the current session's Init/Platform/Loads instead of a cross-device mix.
    readonly sessionId: string;
    // Per-message dedup state (memory-only, per session). The first occurrence of each
    // distinct message is always persisted, so a crash never hides an error; subsequent
    // identical messages are counted and only milestone-sampled to disk. See appendError.
    private errAgg = new Map<string, { count: number; lastWritten: number; firstTs: string; lastTs: string; lastContext: string }>();
    // Plugin-scoped per-device log directory (logDirFor) — a co-installed build
    // writes into its own folder, not a sibling's. 'seek' → the historical path.
    private readonly logDir: string;
    constructor(app: App, pluginId: string) {
        this.app = app;
        this.logDir = logDirFor(pluginId);
        this.deviceId = resolveDeviceId();
        this.sessionId = randId();
    }

    private logPath(): string { return `${this.logDir}/${LOG_PREFIX}${this.deviceId}.ndjson`; }
    private initPath(): string { return `${this.logDir}/${INIT_PREFIX}${this.deviceId}.json`; }

    // mkdir LOG_DIR if absent. Idempotent and best-effort: the parent
    // .obsidian/plugins/seek folder always exists (the plugin loads from it), so
    // this only ever creates the leaf 'logs'. Called before every first write.
    private async ensureDir(): Promise<void> {
        const adapter = this.app.vault.adapter;
        if (await adapter.exists(this.logDir).catch(() => false)) return;
        await adapter.mkdir(this.logDir).catch(() => { /* concurrent create / race — write will surface any real failure */ });
    }

    // Stamp device + session onto every outgoing entry. Centralized here so no
    // call site has to know about attribution. Constrained structurally (any
    // log-shaped row) rather than to LogEntry, so it also stamps separate-stream
    // entries like InitEntry that aren't members of the LogEntry union.
    private stamp<T extends { type: string; timestamp: string }>(entry: T): T & LogMeta {
        return { ...entry, deviceId: this.deviceId, sessionId: this.sessionId };
    }

    async append(entry: LogEntry): Promise<void> {
        const stamped = this.stamp(entry);
        const line = JSON.stringify(stamped) + '\n';
        const adapter = this.app.vault.adapter;
        await this.ensureDir();
        const path = this.logPath();
        const exists = await adapter.exists(path).catch(() => false);
        if (!exists) {
            try { await adapter.write(path, line); }
            catch (e) { console.error('[seek] log create failed:', e); }
            return;
        }
        try {
            await adapter.append(path, line);
        } catch (e) {
            // Append can fail on iOS WKWebView under iCloud contention.
            // Read-then-rewrite preserves prior data instead of clobbering.
            console.error('[seek] log append failed, falling back to read+rewrite:', e);
            try {
                const existing = await adapter.read(path);
                await adapter.write(path, existing + line);
            } catch (e2) {
                console.error('[seek] log read+rewrite also failed:', e2);
            }
        }
    }

    async writeInit(entry: InitEntry): Promise<void> {
        try {
            await this.ensureDir();
            await this.app.vault.adapter.write(this.initPath(), JSON.stringify(this.stamp(entry), null, 2));
        } catch (e) {
            console.error('[seek] init file write failed:', e);
        }
    }

    async appendError(context: string, e: unknown): Promise<void> {
        const message = e instanceof Error ? e.message : String(e);
        const stack = e instanceof Error ? (e.stack ?? null) : null;
        const ts = new Date().toISOString();
        const now = Date.now();
        // Console always fires — live dev visibility is never throttled; only the on-disk
        // NDJSON is deduped.
        console.error(`[seek] error in "${context}":`, e);
        if (stack) console.error('[seek] stack:', stack);
        // Dedup identical errors (same context+message) within ERROR_DEDUP_WINDOW_MS: the
        // first is written immediately (crash-safe — the throttle never hides an error),
        // repeats are only counted, and the count surfaces as a `repeated:N` row on the
        // next write past the window (or at report time via flushErrorAggregates).
        const key = message.slice(0, ERROR_KEY_MAXLEN);
        const agg = this.errAgg.get(key);
        if (!agg) {
            if (this.errAgg.size >= ERROR_KEYS_MAX) {
                const oldest = this.errAgg.keys().next().value;   // evict oldest (insertion order)
                if (oldest !== undefined) this.errAgg.delete(oldest);
            }
            this.errAgg.set(key, { count: 1, lastWritten: 1, firstTs: ts, lastTs: ts, lastContext: context });
            const entry: ErrorEntry = { type: 'error', timestamp: ts, context, message, stack };   // first: full, keeps stack
            await this.append(entry);
            return;
        }
        agg.count += 1;
        agg.lastTs = ts;
        agg.lastContext = context;
        // Write only at exponential milestones (count is a power of two); repeats between
        // milestones are merely counted. The running total rides along as `repeated`, and
        // stack is dropped (the first occurrence already carried it).
        if ((agg.count & (agg.count - 1)) === 0) {
            agg.lastWritten = agg.count;
            const entry: ErrorEntry = { type: 'error', timestamp: ts, context, message, stack: null, repeated: agg.count };
            await this.append(entry);
        }
    }

    // Parse one NDJSON file into entries, attributing a fallback deviceId to any
    // entry that predates v9 stamping (the legacy file, or older per-device lines).
    private parseLog(raw: string, fallbackDeviceId: string): LogEntry[] {
        const out: LogEntry[] = [];
        for (const line of raw.split('\n')) {
            if (!line.trim()) continue;
            try {
                const e = JSON.parse(line) as LogEntry;
                if (!e.deviceId) e.deviceId = fallbackDeviceId;
                out.push(e);
            } catch (err) {
                console.warn('[seek] skipping malformed log line:', line.slice(0, 200), err);
            }
        }
        return out;
    }

    // Read THIS device's log only (used where the current device's stream is all
    // that matters).
    async readAll(): Promise<LogEntry[]> {
        try {
            return this.parseLog(await this.app.vault.adapter.read(this.logPath()), this.deviceId);
        } catch {
            return [];
        }
    }

    // Read every device's log (per-device files + the legacy shared file) and
    // merge by timestamp. This is what the report reads so a phone's run is
    // visible even when generating the report on desktop — and vice versa.
    async readAllDevices(): Promise<LogEntry[]> {
        const adapter = this.app.vault.adapter;
        const all: LogEntry[] = [];

        // Always read THIS device's log + the legacy file directly, so the
        // report never depends on adapter.list() path semantics (which differ
        // across the desktop FS adapter and the mobile Capacitor one). list()
        // then ADDS any OTHER devices' logs as a best-effort bonus. We list both
        // LOG_DIR (the new home) AND the vault root: during the upgrade window a
        // not-yet-migrated device may still be writing per-device logs at root,
        // and iCloud may have synced them here before this device migrated.
        const candidates = new Set<string>([this.logPath(), `${this.logDir}/${LEGACY_LOG_BASE}`]);
        for (const dir of [this.logDir, '']) {
            try {
                const listed = await adapter.list(dir);
                for (const f of listed.files ?? []) {
                    const base = (f.split('/').pop() ?? '').replace(/^\/+/, '');
                    if (base === LEGACY_LOG_BASE || (base.startsWith(LOG_PREFIX) && base.endsWith('.ndjson'))) {
                        candidates.add(f.replace(/^\/+/, ''));
                    }
                }
            } catch { /* list unsupported/failed for this dir — current device + legacy still covered above */ }
        }

        for (const path of candidates) {
            const base = path.split('/').pop() ?? path;
            const fallback = base === LEGACY_LOG_BASE ? 'legacy' : base.slice(LOG_PREFIX.length, -'.ndjson'.length);
            try {
                all.push(...this.parseLog(await adapter.read(path), fallback));
            } catch { /* file absent / not yet downloaded from iCloud — skip */ }
        }
        all.sort((a, b) => (a.timestamp < b.timestamp ? -1 : a.timestamp > b.timestamp ? 1 : 0));
        return all;
    }

    async clear(): Promise<void> {
        const adapter = this.app.vault.adapter;
        if (await adapter.exists(this.logPath()).catch(() => false)) {
            await adapter.write(this.logPath(), '');
        }
    }

    // Tail-truncate THIS device's log if it has grown past MAX_LOG_BYTES, keeping the
    // most recent KEEP_LOG_BYTES. Best-effort and load-only (never in the append hot
    // path); per-device files mean each device only ever trims its own stream. stat
    // gives a cheap size probe so an in-bounds log is never fully read just to measure;
    // when stat is unavailable we fall back to a single startup read. The cut is
    // advanced to the next '\n' so the retained head starts on a clean line boundary —
    // and parseLog already skips any malformed line, so a worst-case slice is harmless.
    async rotateIfOversize(): Promise<void> {
        const adapter = this.app.vault.adapter;
        const path = this.logPath();
        try {
            let raw: string | null = null;
            let size: number | null = null;
            try { size = (await adapter.stat(path))?.size ?? null; } catch { /* stat unsupported — read-measure below */ }
            if (size === null) {
                raw = await adapter.read(path).catch(() => null);
                if (raw === null) return;          // file absent / unreadable
                size = raw.length;
            }
            if (size <= MAX_LOG_BYTES) return;
            if (raw === null) raw = await adapter.read(path);
            const cut = raw.length - KEEP_LOG_BYTES;
            const nl = raw.indexOf('\n', cut);
            const tail = nl >= 0 ? raw.slice(nl + 1) : raw.slice(cut);
            await adapter.write(path, tail);
        } catch (e) {
            console.error('[seek] log rotation failed:', e);
        }
    }

    // One-time move of the machine-written data streams from the vault root (where
    // pre-this-change builds wrote them) into the hidden LOG_DIR, so they stop
    // cluttering the file explorer. Idempotent and non-fatal: a file already in
    // LOG_DIR is never clobbered, and any single failed move is logged and skipped
    // rather than aborting. The report (seek-report.md) is deliberately NOT moved —
    // it stays a root vault file so it can be opened in Obsidian. Returns the count
    // moved. Safe to run on every load (it no-ops once the root is clean).
    async migrateRootFiles(): Promise<number> {
        const adapter = this.app.vault.adapter;
        const root = await adapter.list('').catch(() => null);
        if (!root) return 0;
        // Match per-device files (seek-log-<id>.ndjson) AND the bare pre-per-device
        // legacy forms (seek-log.ndjson / seek-init.json / seek-captures.ndjson),
        // which lack the trailing-dash device segment.
        const isDataFile = (base: string): boolean =>
            base === LEGACY_LOG_BASE || base === 'seek-init.json' || base === 'seek-captures.ndjson' ||
            (base.startsWith(LOG_PREFIX) && base.endsWith('.ndjson')) ||
            (base.startsWith(INIT_PREFIX) && base.endsWith('.json')) ||
            (base.startsWith(CAPTURE_PREFIX) && base.endsWith('.ndjson'));
        const toMove = (root.files ?? [])
            .map(f => f.replace(/^\/+/, ''))
            .filter(f => !f.includes('/') && isDataFile(f)); // root-level only
        if (toMove.length === 0) return 0;
        await this.ensureDir();
        let moved = 0;
        for (const src of toMove) {
            const dest = `${this.logDir}/${src}`;
            if (await adapter.exists(dest).catch(() => false)) {
                // Already present in LOG_DIR (a prior partial migration, or another
                // device synced its copy here first). Drop the stale root duplicate.
                await adapter.remove(src).catch(() => {});
                continue;
            }
            try {
                await adapter.rename(src, dest);
                moved++;
            } catch (e) {
                await this.appendError('logger-migrate-root', e).catch(() => {});
            }
        }
        return moved;
    }

    // Build the full diagnostic dataset: every device's stream merged + sorted by
    // timestamp (readAllDevices), plus a small metadata header. Serialized verbatim to
    // seek-report.json — the parse target. One flat, type-tagged `entries` array is the
    // most parse-friendly shape (filter by `.type` in jq / pandas) and needs no per-type
    // schema here; searches already carry the trimmed top-10 trace (see verboseTrace).
    // Persist the exact final tally for any message whose count has advanced past its
    // last milestone write. Called when building a report (so the artifact reflects an
    // in-flight storm) and safe anytime. Per-device: only this device's pending counts.
    async flushErrorAggregates(): Promise<void> {
        for (const [key, agg] of this.errAgg) {
            if (agg.count <= agg.lastWritten) continue;
            agg.lastWritten = agg.count;
            const summary: ErrorEntry = {
                type: 'error', timestamp: agg.lastTs,
                context: agg.lastContext, message: key,
                stack: null, repeated: agg.count,
            };
            await this.append(summary);
        }
    }

    // Reclaim abandoned log files. Active devices each cap their OWN file via
    // rotateIfOversize, but a retired device's file (and the pre-v9 legacy file) would
    // otherwise linger and sync forever. Drop any log file OTHER than this device's whose
    // most-recent entry is older than ORPHAN_LOG_MAX_AGE_MS. Age comes from file CONTENT
    // (last entry's timestamp), NOT mtime — iCloud re-stamps mtime on sync, so mtime lies.
    // Conservative: a device used within the window keeps its history, and a pruned device
    // that returns simply recreates its file.
    async pruneOrphanLogs(): Promise<void> {
        const adapter = this.app.vault.adapter;
        const mine = this.logPath();
        const listed = await adapter.list(this.logDir).catch(() => null);
        if (!listed) return;
        const now = Date.now();
        for (const path of listed.files ?? []) {
            const norm = path.replace(/^\/+/, '');
            const base = norm.split('/').pop() ?? '';
            const isLog = base === LEGACY_LOG_BASE || (base.startsWith(LOG_PREFIX) && base.endsWith('.ndjson'));
            // Legacy relevance-debug captures are never written or read anymore — pure
            // dead weight; prune them on the same age rule as abandoned device logs.
            const isCapture = base === 'seek-captures.ndjson' || (base.startsWith(CAPTURE_PREFIX) && base.endsWith('.ndjson'));
            if ((!isLog && !isCapture) || norm === mine) continue;   // never touch this device's live log
            try {
                const lastTs = this.lastTimestampOf(await adapter.read(norm));
                if (lastTs === null) continue;        // empty / unreadable — leave it
                if (now - Date.parse(lastTs) <= ORPHAN_LOG_MAX_AGE_MS) continue;
                await adapter.remove(norm);
                if (base.startsWith(LOG_PREFIX)) {    // drop the paired init sidecar too
                    const dev = base.slice(LOG_PREFIX.length, -'.ndjson'.length);
                    await adapter.remove(`${this.logDir}/${INIT_PREFIX}${dev}.json`).catch(() => {});
                }
            } catch { /* read/remove failed — skip, retry next load */ }
        }
    }

    // Last parseable line's timestamp, scanned from the end (tolerates a torn tail).
    private lastTimestampOf(raw: string): string | null {
        const lines = raw.split('\n');
        for (let i = lines.length - 1; i >= 0; i--) {
            const line = lines[i].trim();
            if (!line) continue;
            try {
                const ts = (JSON.parse(line) as { timestamp?: string }).timestamp;
                if (ts) return ts;
            } catch { /* torn/garbage line — keep scanning upward */ }
        }
        return null;
    }

    async buildReportData(): Promise<ReportData> {
        await this.flushErrorAggregates();   // surface any in-flight suppressed-error tails
        const entries = await this.readAllDevices();
        const byDevice = new Map<string, number>();
        for (const e of entries) byDevice.set(e.deviceId ?? 'legacy', (byDevice.get(e.deviceId ?? 'legacy') ?? 0) + 1);
        // Recency-cap: walk newest→oldest keeping up to REPORT_CAPS[type] of each
        // high-volume type (uncapped types kept in full). Then restore ascending order.
        const kept: LogEntry[] = [];
        const perType = new Map<string, number>();
        for (let i = entries.length - 1; i >= 0; i--) {
            const e = entries[i];
            const cap = REPORT_CAPS[e.type];
            if (cap !== undefined) {
                const n = perType.get(e.type) ?? 0;
                if (n >= cap) continue;
                perType.set(e.type, n + 1);
            }
            kept.push(e);
        }
        kept.reverse();
        // Defensively trim any historical 50-row search trace to the 10 the report uses
        // (pre-verboseTrace entries; new ones are already ≤10) so they don't bloat it.
        const trimmed = kept.map(e => {
            if (e.type !== 'search') return e;
            const s = e as SearchEntry;
            return s.fusedTop50 && s.fusedTop50.length > 10 ? { ...s, fusedTop50: s.fusedTop50.slice(0, 10) } : e;
        });
        return {
            generated: new Date().toISOString(),
            schemaVersion: LOG_SCHEMA_VERSION,
            thisDevice: this.deviceId,
            thisSession: this.sessionId,
            entryCount: entries.length,
            includedCount: trimmed.length,
            firstTimestamp: entries[0]?.timestamp ?? null,
            lastTimestamp: entries.at(-1)?.timestamp ?? null,
            devices: [...byDevice.entries()].sort((a, b) => b[1] - a[1]).map(([id, count]) => ({ id, count })),
            caps: REPORT_CAPS,
            entries: trimmed,
        };
    }

    // ~20-line human glance rendered from the already-built data (no second file read).
    // The full detail lives in seek-report.json; this surfaces the headline facts a
    // person needs before sharing, plus the privacy note, and points at the JSON.
    private summarize(d: ReportData): string {
        const lines: string[] = [];
        lines.push('# Seek Diagnostic Report');
        lines.push(`\n_Generated ${d.generated} · log schema v${d.schemaVersion}_`);
        if (d.entryCount === 0) {
            lines.push('\nNo data recorded yet. Run a search or reindex to populate the log.');
            return lines.join('\n') + '\n';
        }
        const searches = filterByType<SearchEntry>(d.entries, 'search');
        const indexes = filterByType<IndexCompleteEntry>(d.entries, 'index-complete');
        const errors = filterByType<ErrorEntry>(d.entries, 'error');
        const crashes = filterByType<CrashDetectedEntry & LogMeta>(d.entries, 'crash-detected');
        const lastInit = filterByType<InitEntry>(d.entries, 'init').at(-1);
        const lastPlatform = filterByType<PlatformEntry>(d.entries, 'platform').at(-1);

        lines.push('\n> [!warning] Review before sharing — this report includes your recent search queries and matching note paths (but **not** note contents).');
        lines.push(`\n**Full data:** \`${REPORT_JSON_PATH}\` — parse that for analysis; this \`.md\` is a human summary.`);
        lines.push('\n## At a Glance');
        lines.push(`- This device: \`${d.thisDevice}\` · session \`${d.thisSession}\``);
        lines.push(`- Events: ${d.includedCount} in report${d.includedCount < d.entryCount ? ` of ${d.entryCount} total (older high-volume entries capped — see \`caps\` in the JSON)` : ''} · ${d.firstTimestamp} → ${d.lastTimestamp}`);
        lines.push(`- Devices: ${d.devices.map(x => `\`${x.id}\` (${x.count})`).join(', ')}`);
        if (lastInit) lines.push(`- Last init: v${lastInit.pluginVersion}, iframe ${lastInit.iframeReady ? '✅' : '❌'}${lastInit.error ? ` · ⚠️ \`${lastInit.error}\`` : ''}`);
        if (lastPlatform) lines.push(`- Platform: ${lastPlatform.isMobile ? 'mobile' : 'desktop'} · GPU ${lastPlatform.gpuAvailable ? 'yes' : 'no'} · storage ${fmtMB(lastPlatform.storageUsedMB)} / ${fmtMB(lastPlatform.storageQuotaMB)}`);
        lines.push(`- Searches ${searches.length} · index runs ${indexes.length} · errors ${errors.length} · crashes ${crashes.length}`);
        if (crashes.length > 0) {
            const c = crashes[crashes.length - 1];
            lines.push('\n## ⚠️ Last Crash');
            lines.push(`- ${c.timestamp} · \`${c.deviceId ?? '?'}\` · **${c.verdict}**`);
        }
        if (errors.length > 0) {
            lines.push('\n## Recent Errors');
            for (const e of errors.slice(-5)) lines.push(`- \`${e.context}\` — ${e.message}`);
        }
        return lines.join('\n') + '\n';
    }

    // Write both report artifacts to the vault root from a single data build: the full
    // structured seek-report.json (the parse target) and a short seek-report.md human
    // summary. Returns the .md path — that's what opens in Obsidian, and it points the
    // reader at the .json.
    async writeReport(): Promise<string> {
        const adapter = this.app.vault.adapter;
        const data = await this.buildReportData();
        await adapter.write(REPORT_JSON_PATH, JSON.stringify(data, null, 2));
        await adapter.write(REPORT_PATH, this.summarize(data));
        return REPORT_PATH;
    }
}

function filterByType<T extends LogEntry>(entries: LogEntry[], type: T['type']): T[] {
    return entries.filter((e): e is T => e.type === type);
}

function fmtTs(iso: string): string { return iso.replace('T', ' ').slice(0, 19); }
function fmtMB(v: number | null): string { return v == null ? 'unknown' : `${v.toFixed(0)} MB`; }
function fmtBytes(v: number | null): string {
    if (v == null) return '—';
    if (v >= 1024 * 1024 * 1024) return `${(v / (1024 ** 3)).toFixed(1)} GB`;
    if (v >= 1024 * 1024) return `${(v / (1024 ** 2)).toFixed(0)} MB`;
    return `${v}`;
}
function distRow(d: DistributionStats): string {
    return `${d.n} | ${d.min.toFixed(1)} | ${d.p50.toFixed(1)} | ${d.mean.toFixed(1)} | ${d.p95.toFixed(1)} | ${d.max.toFixed(1)}`;
}
function pctCacheHit(searches: SearchEntry[]): string {
    if (searches.length === 0) return '—';
    const hits = searches.filter(s => s.bm25CacheHit).length;
    return ((hits / searches.length) * 100).toFixed(0);
}
