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
//   seek-report.md                                              — generated on demand ("Generate logging report"), kept at vault root so it can be opened
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
const LOG_DIR = '.obsidian/plugins/seek/logs';
// The report is the only shared (single-writer-at-a-time, full-overwrite) file,
// and the only one kept at the vault ROOT — it must stay a real vault file so the
// "Generate logging report" command can open it (getAbstractFileByPath only
// resolves files outside the config folder). Safe under iCloud because it's never
// appended to from two devices at once.
const REPORT_PATH = 'seek-report.md';
// Legacy single-file log from before per-device files (schema ≤ v8). Still read
// into the report (its entries are attributed to deviceId 'legacy') so history
// isn't lost; never written to again. The basename is matched both at the vault
// root (pre-move) and inside LOG_DIR (post-move).
const LEGACY_LOG_BASE = 'seek-log.ndjson';
const LEGACY_LOG_PATH = `${LOG_DIR}/${LEGACY_LOG_BASE}`;
const LOG_PREFIX = 'seek-log-';     // per-device: seek-log-<deviceId>.ndjson
const INIT_PREFIX = 'seek-init-';   // per-device: seek-init-<deviceId>.json
const CAPTURE_PREFIX = 'seek-captures-';
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

export class SeekLogger {
    private app: App;
    // Stable across reloads (localStorage); identifies the physical device.
    readonly deviceId: string;
    // Fresh per plugin load; identifies this run so the report can show only
    // the current session's Init/Platform/Loads instead of a cross-device mix.
    readonly sessionId: string;
    constructor(app: App) {
        this.app = app;
        this.deviceId = resolveDeviceId();
        this.sessionId = randId();
    }

    private logPath(): string { return `${LOG_DIR}/${LOG_PREFIX}${this.deviceId}.ndjson`; }
    private initPath(): string { return `${LOG_DIR}/${INIT_PREFIX}${this.deviceId}.json`; }

    // mkdir LOG_DIR if absent. Idempotent and best-effort: the parent
    // .obsidian/plugins/seek folder always exists (the plugin loads from it), so
    // this only ever creates the leaf 'logs'. Called before every first write.
    private async ensureDir(): Promise<void> {
        const adapter = this.app.vault.adapter;
        if (await adapter.exists(LOG_DIR).catch(() => false)) return;
        await adapter.mkdir(LOG_DIR).catch(() => { /* concurrent create / race — write will surface any real failure */ });
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
        // Echo to console too — useful when running with Obsidian dev tools open.
        console.log(`[seek][${entry.type}]`, stamped);
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
        const entry: ErrorEntry = {
            type: 'error',
            timestamp: new Date().toISOString(),
            context,
            message: e instanceof Error ? e.message : String(e),
            stack: e instanceof Error ? (e.stack ?? null) : null,
        };
        console.error(`[seek] error in "${context}":`, e);
        if (e instanceof Error && e.stack) console.error('[seek] stack:', e.stack);
        await this.append(entry);
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
        const candidates = new Set<string>([this.logPath(), LEGACY_LOG_PATH]);
        for (const dir of [LOG_DIR, '']) {
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
            const dest = `${LOG_DIR}/${src}`;
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
        console.log(`[seek] log migration: moved ${moved}/${toMove.length} files → ${LOG_DIR}`);
        return moved;
    }

    async generateReport(): Promise<string> {
        const entries = await this.readAllDevices();
        if (entries.length === 0) {
            return '# Seek Log Report\n\nNo data recorded yet. Run a search or reindex to populate the log.\n';
        }

        // Device-identity sections (Init / Platform / Loads) must describe the
        // device generating THIS report, not whichever device last appended to
        // the shared log. Scope them to the current session; the legacy at(-1)
        // over a cross-device merge is exactly the bug this fixes.
        const sessionEntries = entries.filter(e => e.sessionId === this.sessionId);
        const identityScope = sessionEntries.length > 0 ? sessionEntries : entries;
        const scopedToSession = sessionEntries.length > 0;

        const inits = filterByType<InitEntry>(identityScope, 'init');
        const platforms = filterByType<PlatformEntry>(identityScope, 'platform');
        const loads = filterByType<LoadEntry>(identityScope, 'load');
        const profiles = filterByType<EmbedProfileEntry>(entries, 'embed-profile');
        const progress = filterByType<IndexProgressEntry>(entries, 'index-progress');
        const completes = filterByType<IndexCompleteEntry>(entries, 'index-complete');
        const searches = filterByType<SearchEntry>(entries, 'search');
        const resets = filterByType<ResetEntry>(entries, 'reset');
        const errors = filterByType<ErrorEntry>(entries, 'error');
        const longTasks = filterByType<LongTaskEntry>(entries, 'long-task');
        const memoryPressure = filterByType<MemoryPressureEntry>(entries, 'memory-pressure');
        const storageSnaps = filterByType<StorageSnapshotEntry>(entries, 'storage-snapshot');
        const evictions = filterByType<EvictionSuspectedEntry>(entries, 'eviction-suspected');
        const probes = filterByType<AppLocalFetchEntry>(entries, 'app-local-fetch');
        const clicks = filterByType<ClickEntry>(entries, 'click');
        const crashes = filterByType<CrashDetectedEntry & import('./types').LogMeta>(entries, 'crash-detected');

        const lines: string[] = [];
        lines.push('# Seek Log Report');
        lines.push(`\n_Generated: ${new Date().toISOString()}_`);
        lines.push(`_Log entries: ${entries.length} · first: ${entries[0].timestamp} · last: ${entries[entries.length - 1].timestamp}_`);
        lines.push(`_Report schema version: ${LOG_SCHEMA_VERSION}_`);

        // ---- This Session (device attribution) ----
        // Per-device entry counts make cross-device mixing visible at a glance —
        // and confirm a phone's run actually synced into the vault.
        const byDevice = new Map<string, number>();
        for (const e of entries) byDevice.set(e.deviceId ?? 'legacy', (byDevice.get(e.deviceId ?? 'legacy') ?? 0) + 1);
        lines.push('\n## This Session');
        lines.push(`- Device: \`${this.deviceId}\``);
        lines.push(`- Session: \`${this.sessionId}\` (${sessionEntries.length} entries this session)`);
        if (!scopedToSession) {
            lines.push('- ⚠️ No entries for the current session in any log file — the Init / Platform / Loads sections below fall back to the most recent entry **across all devices** and may not describe this device.');
        }
        lines.push('\n_Entries by device (across all log files):_');
        lines.push('\n| Device | Entries |');
        lines.push('|---|---:|');
        for (const [dev, n] of [...byDevice.entries()].sort((a, b) => b[1] - a[1])) {
            lines.push(`| \`${dev}\`${dev === this.deviceId ? ' ← this device' : ''} | ${n} |`);
        }

        // ---- Crash Forensics ----
        // Promoted at boot from the synchronous localStorage breadcrumb ring
        // (forensics.ts) — the only record that survives a jetsam kill. Placed
        // near the top: a crash is the single most important fact in a report.
        if (crashes.length > 0) {
            lines.push('\n## Crash Forensics');
            lines.push(`\n${crashes.length} unclean session end(s) detected across devices. Verdict legend: \`crash-while-indexing-foreground\` = memory-ceiling signature · \`crash-while-indexing-hidden\` = iOS background-GPU kill · \`evicted-while-hidden\` = normal suspended-app eviction.`);
            lines.push('\n| When detected | Device | Verdict | Last beat | Gap |');
            lines.push('|---|---|---|---|---:|');
            for (const c of crashes.slice(-15)) {
                const lastBeat = c.lastBeat
                    ? `\`${c.lastBeat.type}\` (${c.lastBeat.vis})${c.lastBeat.detail ? ' ' + JSON.stringify(c.lastBeat.detail) : ''}`
                    : '—';
                const gap = c.gapSeconds != null ? `${Math.round(c.gapSeconds)}s` : '—';
                lines.push(`| ${c.timestamp} | \`${c.deviceId ?? '?'}\` | **${c.verdict}** | ${lastBeat} | ${gap} |`);
            }
            const lastCrash = crashes.at(-1);
            if (lastCrash && lastCrash.breadcrumbs.length > 0) {
                lines.push('\n_Most recent crash — breadcrumb tail (oldest → newest):_');
                lines.push('```');
                for (const b of lastCrash.breadcrumbs) {
                    lines.push(`${b.t} ${b.type} [${b.vis}]${b.detail ? ' ' + JSON.stringify(b.detail) : ''}`);
                }
                lines.push('```');
            }
        }

        // ---- Init ----
        const lastInit = inits.at(-1);
        if (lastInit) {
            lines.push(`\n## Last Init${scopedToSession ? ' (this session)' : ''}`);
            lines.push(`- Plugin version: \`${lastInit.pluginVersion}\``);
            lines.push(`- Build timestamp: \`${lastInit.buildTimestamp}\``);
            lines.push(`- transformers.js: \`${lastInit.transformersVersion}\``);
            lines.push(`- CDN: \`${lastInit.cdnUrl}\``);
            lines.push(`- Iframe ready: ${lastInit.iframeReady ? '✅' : '❌'}`);
            // initMs is v11+ (older logs lack it). 0 = idempotent early-return
            // (iframe already live), not a zero-cost build.
            if (typeof lastInit.initMs === 'number') {
                lines.push(`- Iframe build: ${lastInit.initMs === 0 ? 'cached (idempotent, 0 ms)' : `${lastInit.initMs} ms`}`);
            }
            if (lastInit.error) lines.push(`- Error: \`${lastInit.error}\``);
        }

        // ---- Platform ----
        const lastPlatform = platforms.at(-1);
        if (lastPlatform) {
            lines.push(`\n## Platform${scopedToSession ? ' (this session)' : ''}`);
            lines.push(`- Mobile: ${lastPlatform.isMobile}`);
            // WKWebView FREEZES the OS version in its User-Agent (Apple caps it to
            // avoid version-sniffing breakage), so this is NOT the real iOS version
            // — a phone on iOS 26 reports "18.7" here. Treat it as a lower bound.
            // A non-null GPU adapter below is a better "iOS ≥ 26" signal (the adapter
            // was null pre-26). There's no reliable JS API for the true WebView OS version.
            lines.push(`- iOS version (UA-reported, WKWebView-capped — real OS may be newer): ${lastPlatform.iosVersion ?? 'n/a'}`);
            lines.push(`- GPU available: ${lastPlatform.gpuAvailable}${lastPlatform.gpuAvailable ? ' _(adapter present ⇒ iOS ≥ 26 / modern WebKit)_' : ''}`);
            if (lastPlatform.gpuAdapterDescription) lines.push(`- GPU adapter: \`${lastPlatform.gpuAdapterDescription}\``);
            if (lastPlatform.gpuAdapterLimits) {
                const l = lastPlatform.gpuAdapterLimits;
                lines.push(`- GPU limits: maxBuffer ${fmtBytes(l.maxBufferSize)}, maxStorageBufBind ${fmtBytes(l.maxStorageBufferBindingSize)}, workgroupX ${l.maxComputeWorkgroupSizeX ?? '—'}, invocations/wg ${l.maxComputeInvocationsPerWorkgroup ?? '—'}`);
            }
            lines.push(`- Storage quota: ${fmtMB(lastPlatform.storageQuotaMB)} · used: ${fmtMB(lastPlatform.storageUsedMB)}`);
            lines.push(`- Persistent storage granted: ${lastPlatform.persistGranted ?? 'unknown'}`);
            lines.push(`- Heap API available: ${lastPlatform.heapAvailable ?? 'unknown (pre-v2 log)'} _(false → expect null heapMB everywhere; iOS WebKit doesn't expose performance.memory)_`);
            lines.push(`- measureUserAgentSpecificMemory available: ${lastPlatform.measureMemoryAvailable ?? 'unknown'} (crossOriginIsolated: ${lastPlatform.crossOriginIsolated ?? 'unknown'})`);
            lines.push(`\n<details><summary>User-Agent</summary>\n\n\`${lastPlatform.userAgent}\`\n\n</details>`);
        }

        // ---- Loads ----
        if (loads.length > 0) {
            lines.push('\n## Model Loads');
            lines.push('\n| Time | Requested | Actual | Dtype | Dim | Cold-start (ms) | Heap Δ (MB) | Storage Δ (MB) | Pass |');
            lines.push('|---|---|---|---|---:|---:|---:|---:|:---:|');
            for (const l of loads) {
                // embeddingDim added in LOG_SCHEMA_VERSION 4. Older entries
                // pre-date the field; render as "—" for back-compat.
                const dim = (l as unknown as { embeddingDim?: number }).embeddingDim;
                lines.push(
                    `| ${fmtTs(l.timestamp)} | ${l.requestedDevice} | ${l.actualDevice} | ${l.dtype}` +
                    ` | ${dim ?? '—'}` +
                    ` | ${l.coldStartMs.toFixed(0)} | ${l.heapDeltaMB?.toFixed(1) ?? '—'}` +
                    ` | ${l.storageDeltaMB?.toFixed(1) ?? '—'} | ${l.pass ? '✅' : '❌'} |`,
                );
            }
            lines.push('\n### Load Checks');
            for (const l of loads) {
                lines.push(`\n**${fmtTs(l.timestamp)} (${l.actualDevice}/${l.dtype})**`);
                for (const c of l.checks) lines.push(`- ${c}`);
            }
        }

        // ---- Runtime profile (wall-time decomposition) ----
        const lastProfile = profiles.at(-1);
        if (lastProfile) {
            lines.push('\n## Runtime Profile (wall-time decomposition)');
            lines.push(
                `\n_${fmtTs(lastProfile.timestamp)} · ${lastProfile.device}/${lastProfile.dtype}` +
                ` · transformers.js ${lastProfile.transformersVersion}` +
                ` · run ${(lastProfile.elapsedMs / 1000).toFixed(1)} s` +
                ` · heap Δ ${lastProfile.heapDeltaMB?.toFixed(1) ?? '—'} MB (non-disposing)_`,
            );
            lines.push(`\n_${lastProfile.notes}_`);
            lines.push('\n| bs × seq | tokenize p50 | forward p50 | post(readback) p50 | pipeline p50 | fwd % | tok % | /text fwd |');
            lines.push('|---|---:|---:|---:|---:|---:|---:|---:|');
            const ms = (d: DistributionStats | null) => d ? d.p50.toFixed(1) : '—';
            for (const c of lastProfile.cells) {
                lines.push(
                    `| ${c.batchSize} × ${c.seqBucket}` +
                    ` | ${ms(c.tokenizeMs)} | ${ms(c.forwardMs)} | ${ms(c.postMs)} | ${ms(c.pipelineTotalMs)}` +
                    ` | ${c.forwardSharePct ?? '—'} | ${c.tokenizeSharePct ?? '—'}` +
                    ` | ${c.perTextForwardMs ?? '—'} |`,
                );
            }
            // The decision read, spelled out so the report is self-interpreting.
            const probe = lastProfile.cells.find(c => c.batchSize === 8 && c.seqBucket === 128)
                ?? lastProfile.cells[0];
            if (probe) {
                lines.push(
                    `\n**Read (bs8×128):** forward ${probe.forwardSharePct ?? '—'}% of wall` +
                    ` · tokenize ${probe.tokenizeSharePct ?? '—'}%.` +
                    ` High forward% ⇒ GPU-forward-bound: I/O binding / worker low ROI (v4 kernel fix already captured it).` +
                    ` High tokenize% ⇒ WASM tokenizer worth it. Climbing heap Δ ⇒ the undisposed-tensor leak is real.`,
                );
            }
        }

        // ---- Index ops ----
        if (completes.length > 0 || resets.length > 0) {
            lines.push('\n## Index Operations');
            if (resets.length > 0) {
                lines.push('\n### Resets (Full Reindex)');
                lines.push('\n| Time | Chunks dropped | Vectors dropped | Duration (ms) | Pass |');
                lines.push('|---|---:|---:|---:|:---:|');
                for (const r of resets) {
                    lines.push(`| ${fmtTs(r.timestamp)} | ${r.chunksDeleted} | ${r.vectorsDeleted} | ${r.durationMs.toFixed(0)} | ${r.pass ? '✅' : '❌'} |`);
                }
            }
            if (completes.length > 0) {
                lines.push('\n### Index Builds');
                lines.push('\n| Time | Mode | Dtype/Dim | Files | Skipped | Chunks | Embed (ms) | Chunk (ms) | BM25 (ms) | Commit (ms) | Total (ms) | Chunks/s | Files/s | Heap Δ | Storage Δ | Pass |');
                lines.push('|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|');
                for (const c of completes) {
                    // dtype + embeddingDim added in LOG_SCHEMA_VERSION 4.
                    // Older entries render the cell as "—" for back-compat.
                    const dtype = (c as unknown as { dtype?: string }).dtype;
                    const dim = (c as unknown as { embeddingDim?: number }).embeddingDim;
                    const dtypeDim = dtype && dim ? `${dtype}/${dim}` : '—';
                    lines.push(
                        `| ${fmtTs(c.timestamp)} | ${c.mode} | ${dtypeDim} | ${c.filesIndexed} | ${c.filesSkippedError ?? 0} | ${c.chunksIndexed}` +
                        ` | ${c.embedDurationMs.toFixed(0)} | ${c.chunkDurationMs.toFixed(0)}` +
                        ` | ${c.bm25DurationMs.toFixed(0)} | ${c.commitDurationMs.toFixed(0)}` +
                        ` | ${c.totalDurationMs.toFixed(0)} | ${(c.chunksPerSec ?? 0).toFixed(1)} | ${(c.filesPerSec ?? 0).toFixed(1)}` +
                        ` | ${c.heapDeltaMB?.toFixed(1) ?? '—'} | ${c.storageDeltaMB?.toFixed(1) ?? '—'}` +
                        ` | ${c.pass ? '✅' : '❌'} |`,
                    );
                }

                // Distribution summary for the most recent build only —
                // the prior table's rows already capture history at the
                // aggregate level; deep stats below are most useful for
                // diagnosing the latest run.
                const last = completes.at(-1)!;
                if (last.perFileWallMs || last.chunksPerFile || last.embedBatchLatencyMs) {
                    lines.push('\n### Last Build — Distributions');
                    lines.push('\n| Metric | n | min | p50 | mean | p95 | max |');
                    lines.push('|---|---:|---:|---:|---:|---:|---:|');
                    if (last.perFileWallMs) lines.push(`| Per-file wall (ms) | ${distRow(last.perFileWallMs)} |`);
                    if (last.chunksPerFile) lines.push(`| Chunks per file | ${distRow(last.chunksPerFile)} |`);
                    if (last.embedBatchLatencyMs) lines.push(`| Embed batch latency (ms) | ${distRow(last.embedBatchLatencyMs)} |`);
                }

                lines.push('\n### Index Checks');
                for (const c of completes) {
                    lines.push(`\n**${fmtTs(c.timestamp)} — ${c.mode}**`);
                    for (const ck of c.checks) lines.push(`- ${ck}`);
                }
            }
        }

        // ---- Index Progress (last N) ----
        if (progress.length > 0) {
            const lastN = progress.slice(-50);
            lines.push('\n## Most Recent Index Progress (last 50 entries)');
            lines.push('\n| Time | Phase | Files seen / total | Chunks emitted | Elapsed (ms) | Heap (MB) | Storage (MB) |');
            lines.push('|---|---|---:|---:|---:|---:|---:|');
            for (const p of lastN) {
                lines.push(
                    `| ${fmtTs(p.timestamp)} | ${p.phase} | ${p.filesSeen} / ${p.filesTotal}` +
                    ` | ${p.chunksEmitted} | ${p.elapsedMs.toFixed(0)}` +
                    ` | ${p.heapMB?.toFixed(1) ?? '—'} | ${p.storageMB?.toFixed(1) ?? '—'} |`,
                );
            }
        }

        // ---- Search history ----
        if (searches.length > 0) {
            lines.push('\n## Search History');
            lines.push(`\n_Total queries: ${searches.length}_`);
            lines.push('\n| Time | Query | TopK | IDB | Align | Embed | Iframe | Cosine | BM25 | Cache | Fusion | Snippet | Total | Chunks |');
            lines.push('|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|');
            for (const s of searches) {
                const q = s.query.replace(/\|/g, '\\|').slice(0, 40);
                lines.push(
                    `| ${fmtTs(s.timestamp)} | \`${q}\` | ${s.topK}` +
                    ` | ${(s.idbReadMs ?? 0).toFixed(0)} | ${(s.alignMs ?? 0).toFixed(0)}` +
                    ` | ${s.queryEmbedMs.toFixed(0)} | ${(s.iframeEmbedMs ?? 0).toFixed(0)}` +
                    ` | ${(s.cosineMs ?? 0).toFixed(0)} | ${s.bm25Ms.toFixed(0)}` +
                    ` | ${s.bm25CacheHit ? '✅' : '❌'} | ${s.fusionMs.toFixed(0)}` +
                    ` | ${(s.snippetMs ?? 0).toFixed(0)} | ${s.totalMs.toFixed(0)}` +
                    ` | ${s.totalChunks} |`,
                );
            }

            // Latency stats over recent N
            const recent = searches.slice(-50);
            const totals = recent.map(s => s.totalMs).sort((a, b) => a - b);
            const p = (q: number) => totals[Math.floor((q / 100) * (totals.length - 1))]?.toFixed(0) ?? '—';
            lines.push('\n### Recent Latency (last 50 searches)');
            lines.push(`- Total p50 / p90 / p99: **${p(50)} / ${p(90)} / ${p(99)} ms**`);
            // Per-stage rollups make it obvious which stage is the bottleneck.
            const stageStats = (pick: (s: SearchEntry) => number) => {
                const sorted = recent.map(pick).sort((a, b) => a - b);
                const sp = (q: number) => sorted[Math.floor((q / 100) * (sorted.length - 1))]?.toFixed(0) ?? '—';
                return `p50 ${sp(50)} / p95 ${sp(95)}`;
            };
            lines.push(`- IDB read: ${stageStats(s => s.idbReadMs ?? 0)} ms`);
            lines.push(`- Align: ${stageStats(s => s.alignMs ?? 0)} ms`);
            lines.push(`- Embed (parent): ${stageStats(s => s.queryEmbedMs)} ms — iframe-side: ${stageStats(s => s.iframeEmbedMs ?? 0)} ms`);
            lines.push(`- Cosine: ${stageStats(s => s.cosineMs ?? 0)} ms`);
            lines.push(`- BM25: ${stageStats(s => s.bm25Ms)} ms (cache hit rate: ${pctCacheHit(recent)}%)`);
            lines.push(`- Fusion: ${stageStats(s => s.fusionMs)} ms`);
            lines.push(`- Snippet: ${stageStats(s => s.snippetMs ?? 0)} ms`);

            // Last few full breakdowns for diagnostic depth. fusedTop50 in
            // NDJSON; we render only top-10 here to keep the report readable.
            // Spreadsheet / offline analysis can read the full top-50 from
            // seek-log.ndjson directly.
            const last = searches.slice(-5);
            lines.push('\n### Last 5 Searches — Top-10 Fused Results (full top-50 in NDJSON)');
            for (const s of last) {
                // Backward compatibility: pre-schema-v3 entries had `fusedTop10`
                // and lacked `note_path` / `rank` / `searchId`. Read from the
                // new field first, fall back to the old shape, normalize so
                // the row renderer below sees a consistent structure.
                interface RowLike {
                    rank: number;
                    note_path: string;
                    title: string;
                    score: number;
                    dense: number;
                    bm25: number;
                    recency: number;
                    title_boost: number;
                }
                interface OldRow {
                    chunk_id: string;
                    title: string;
                    score: number;
                    dense: number;
                    bm25: number;
                    recency: number;
                    title_boost: number;
                }
                const legacy = s as unknown as { fusedTop10?: OldRow[] };
                const fromNew = s.fusedTop50 ?? [];
                const fromOld: RowLike[] = (legacy.fusedTop10 ?? []).map((r, i) => ({
                    rank: i + 1,
                    note_path: r.chunk_id,           // closest we have on old entries
                    title: r.title,
                    score: r.score,
                    dense: r.dense,
                    bm25: r.bm25,
                    recency: r.recency,
                    title_boost: r.title_boost,
                }));
                const rows: RowLike[] = fromNew.length > 0 ? fromNew : fromOld;
                if (rows.length === 0) continue;

                const sid = s.searchId ?? '(legacy)';
                // Inline-filter annotation (v8+). Absent on older logs → blank.
                const fp: string[] = [];
                if (s.filters?.tags) fp.push(`tags:${s.filters.tags.join(',')}`);
                if (s.filters?.includePaths) fp.push(`path:${s.filters.includePaths.join(',')}`);
                if (s.filters?.frontmatter) fp.push(Object.entries(s.filters.frontmatter).map(([k, v]) => `${k}=${v}`).join(','));
                if (s.filters?.createdAfter) fp.push(`created>${s.filters.createdAfter}`);
                if (s.filters?.createdBefore) fp.push(`created<${s.filters.createdBefore}`);
                if (s.filters?.modifiedAfter) fp.push(`modified>${s.filters.modifiedAfter}`);
                if (s.filters?.modifiedBefore) fp.push(`modified<${s.filters.modifiedBefore}`);
                const filterStr = fp.length ? ` · filters: ${fp.join(' ')}` : '';
                const cleanedStr = (s.cleanedQuery && s.cleanedQuery !== s.query) ? ` · cleaned=\`${s.cleanedQuery}\`` : '';
                const blendStr = s.blendMode === 'rrf' ? `, blend=rrf k=${s.rrfK}` : '';
                lines.push(`\n**\`${s.query}\`** _(α=${s.alpha}, recencyW=${s.recencyWeight}${blendStr}${cleanedStr}${filterStr}, searchId=\`${sid}\`)_`);
                lines.push('| Rank | Note path | Title | Final | Dense | BM25 | Recency | Title boost |');
                lines.push('|---:|---|---|---:|---:|---:|---:|---:|');
                rows.slice(0, 10).forEach(r => {
                    const title = r.title.replace(/\|/g, '\\|').slice(0, 60);
                    const path = r.note_path.replace(/\|/g, '\\|').slice(0, 50);
                    lines.push(`| ${r.rank} | \`${path}\` | ${title} | ${r.score.toFixed(3)}` +
                        ` | ${r.dense.toFixed(3)} | ${r.bm25.toFixed(3)} | ${r.recency.toFixed(3)} | ${r.title_boost.toFixed(3)} |`);
                });
            }
        }

        // ---- Click events / relevance ----
        if (clicks.length > 0) {
            lines.push('\n## Click Events / Relevance');
            lines.push(`\n_Total clicks: ${clicks.length}. CTR-style breakdown by rank-of-click:_\n`);

            // Group clicks by rank bucket. Industry convention: click-at-1
            // is the cleanest "good ranking" signal; clicks at 2–3 mean the
            // top-1 was wrong-ish but in the right neighborhood; clicks at
            // 4+ mean the ranking is meaningfully off.
            let r1 = 0, r23 = 0, r4to10 = 0, beyond10 = 0;
            for (const c of clicks) {
                if (c.rank === 1) r1++;
                else if (c.rank <= 3) r23++;
                else if (c.rank <= 10) r4to10++;
                else beyond10++;
            }
            const pct = (n: number) => clicks.length === 0 ? '—' : `${((n / clicks.length) * 100).toFixed(0)}%`;
            lines.push('| Rank bucket | Count | % of clicks |');
            lines.push('|---|---:|---:|');
            lines.push(`| Rank 1 (top hit clicked) | ${r1} | ${pct(r1)} |`);
            lines.push(`| Rank 2–3 | ${r23} | ${pct(r23)} |`);
            lines.push(`| Rank 4–10 | ${r4to10} | ${pct(r4to10)} |`);
            lines.push(`| Rank 11+ (off-screen) | ${beyond10} | ${pct(beyond10)} |`);

            // Per-click recent table (last 30) for narrative debugging.
            const recent = clicks.slice(-30);
            lines.push('\n### Recent Click Events (last 30)');
            lines.push('| Time | Query | Rank | Clicked path | Score | Dwell (ms) |');
            lines.push('|---|---|---:|---|---:|---:|');
            for (const c of recent) {
                const q = c.query.replace(/\|/g, '\\|').slice(0, 30);
                const path = c.note_path.replace(/\|/g, '\\|').slice(0, 50);
                lines.push(`| ${fmtTs(c.timestamp)} | \`${q}\` | ${c.rank} | \`${path}\` | ${c.score.toFixed(3)} | ${c.dwellMs.toFixed(0)} |`);
            }

            // Dwell time distribution — short dwell suggests "accepted top
            // result", long dwell suggests "deliberated then picked".
            const dwells = clicks.map(c => c.dwellMs).sort((a, b) => a - b);
            const pq = (q: number) => dwells[Math.floor((q / 100) * (dwells.length - 1))]?.toFixed(0) ?? '—';
            lines.push(`\n_Dwell p50 / p90: **${pq(50)} ms / ${pq(90)} ms**_`);
        }

        // ---- Long tasks ----
        if (longTasks.length > 0) {
            // Group by context for a quick "where's the jank?" view.
            const byContext = new Map<string, LongTaskEntry[]>();
            for (const t of longTasks) {
                const arr = byContext.get(t.context) ?? [];
                arr.push(t);
                byContext.set(t.context, arr);
            }
            lines.push('\n## Long Tasks (≥250 ms main-thread blocks)');
            lines.push(`\n_Total: ${longTasks.length}. Per-context breakdown:_\n`);
            lines.push('| Context | Count | Sum (ms) | Max (ms) |');
            lines.push('|---|---:|---:|---:|');
            for (const [ctx, arr] of byContext) {
                const sum = arr.reduce((s, t) => s + t.durationMs, 0);
                const max = arr.reduce((m, t) => Math.max(m, t.durationMs), 0);
                lines.push(`| ${ctx} | ${arr.length} | ${sum.toFixed(0)} | ${max.toFixed(0)} |`);
            }
            // Show the worst 20 so the user can spot specific bad actors.
            const worst = [...longTasks].sort((a, b) => b.durationMs - a.durationMs).slice(0, 20);
            lines.push('\n### Worst 20');
            lines.push('| Time | Context | Duration (ms) | Attribution |');
            lines.push('|---|---|---:|---|');
            for (const t of worst) {
                lines.push(`| ${fmtTs(t.timestamp)} | ${t.context} | ${t.durationMs.toFixed(0)} | ${t.attribution ?? '—'} |`);
            }
        }

        // ---- Memory pressure events ----
        if (memoryPressure.length > 0) {
            // Limit to the last 30 — page lifecycle events fire often.
            const lastN = memoryPressure.slice(-30);
            lines.push('\n## Memory Pressure Events');
            lines.push('\n_Captured at visibility/pagehide. Useful for correlating iOS jetsam with last-known state._\n');
            lines.push('| Time | Event | Heap (MB) | Storage (MB) | Persisted |');
            lines.push('|---|---|---:|---:|:---:|');
            for (const m of lastN) {
                lines.push(`| ${fmtTs(m.timestamp)} | ${m.event} | ${m.heapMB?.toFixed(1) ?? '—'} | ${m.storageMB?.toFixed(1) ?? '—'} | ${m.persisted ? '✅' : '❌'} |`);
            }
        }

        // ---- Storage snapshots ----
        if (storageSnaps.length > 0) {
            lines.push('\n## Storage Snapshots');
            lines.push('\n| Time | Context | Used (MB) | Quota (MB) | Heap (MB) |');
            lines.push('|---|---|---:|---:|---:|');
            for (const s of storageSnaps) {
                lines.push(`| ${fmtTs(s.timestamp)} | ${s.context} | ${s.storageUsedMB?.toFixed(1) ?? '—'} | ${s.storageQuotaMB?.toFixed(0) ?? '—'} | ${s.heapMB?.toFixed(1) ?? '—'} |`);
            }
        }

        // ---- Phase 1 telemetry: eviction-suspected + app-local probe ----
        // These two sections answer the rearchitecture plan's Phase 1 exit
        // criteria: is the model cache being evicted (count + storage drop
        // at the same timestamp), and does `app://local/...` work inside
        // the iframe (gates the Phase 3 shard streaming pattern).
        if (evictions.length > 0) {
            lines.push('\n## Eviction Suspected (cold-start ≥ 5 s)');
            lines.push(`\n_Count: ${evictions.length}. Plan exit criterion is ≥3 outliers on iOS._\n`);
            lines.push('| Time | Cold-start (ms) | Device | Dtype | Storage used (MB) | Quota (MB) | Persisted |');
            lines.push('|---|---:|---|---|---:|---:|:---:|');
            for (const e of evictions.slice(-30)) {
                lines.push(
                    `| ${fmtTs(e.timestamp)} | ${e.coldStartMs.toFixed(0)} | ${e.actualDevice} | ${e.dtype}` +
                    ` | ${e.storageUsedMB?.toFixed(1) ?? '—'} | ${e.storageQuotaMB?.toFixed(0) ?? '—'}` +
                    ` | ${e.persisted == null ? '—' : (e.persisted ? '✅' : '❌')} |`,
                );
            }
        }
        if (probes.length > 0) {
            const last = probes.at(-1)!;
            const okCount = probes.filter(p => p.result === 'ok').length;
            lines.push('\n## App-Local Fetch Probe');
            lines.push(`\n_Last result: **${last.result}** (${okCount}/${probes.length} sessions ok). Gates Phase 3 shard pattern._\n`);
            lines.push(`- URL: \`${last.url || '(no resourcePath)'}\``);
            lines.push(`- HTTP status: ${last.httpStatus ?? '—'}`);
            lines.push(`- Body matched probe content: ${last.bodyMatched == null ? '—' : (last.bodyMatched ? '✅' : '❌')}`);
            if (last.error) lines.push(`- Error: \`${last.error}\``);
        }

        // ---- Errors ----
        if (errors.length > 0) {
            lines.push('\n## Errors');
            for (const err of errors.slice(-30)) {
                lines.push(`\n**${fmtTs(err.timestamp)} — ${err.context}**`);
                lines.push('```');
                lines.push(err.message);
                lines.push('```');
                if (err.stack) {
                    lines.push(`<details><summary>Stack trace</summary>\n\n\`\`\`\n${err.stack}\n\`\`\`\n\n</details>`);
                }
            }
        }

        return lines.join('\n') + '\n';
    }

    async writeReport(): Promise<string> {
        const content = await this.generateReport();
        const adapter = this.app.vault.adapter;
        await adapter.write(REPORT_PATH, content);
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
