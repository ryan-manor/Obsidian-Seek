// Crash forensics: synchronous localStorage breadcrumbs that survive process
// death.
//
// Why this exists: the NDJSON logger's appends are async vault writes, and a
// jetsam kill doesn't wait for them. The 2026-06-11 iPhone WebGPU reindex
// death left ZERO log entries — no error, no embed tick, no visibility event;
// the next session's init was the only evidence. localStorage.setItem is
// synchronous: a beat written at the top of an embed flush is durable before
// the dispatch that might kill the process even starts.
//
// The detection trick is absence: every session writes a record that only a
// clean shutdown (plugin onunload) closes. On boot, an unclosed record from a
// previous session means the process died. The last breadcrumb's visibility
// state + what-was-running classifies the death (see CrashVerdict in types):
// foreground+indexing reads as a memory-ceiling kill, hidden+indexing as iOS
// background-GPU termination, hidden+idle as ordinary suspended-app eviction.
//
// Scope: localStorage is per-ORIGIN (capacitor://localhost on iOS), shared
// across vaults in the same app — the same trap as the shared-IDB incident.
// The storage key embeds the vault scope (appId) so two vaults' records can't
// clobber each other. The deviceId key is intentionally NOT scoped (one
// physical device = one id, already the logger's convention).

import type { ForensicBreadcrumb, CrashDetectedEntry, CrashVerdict } from './types';

// Ring capacity. Beats during a reindex arrive every flush (~100-300 ms on
// WebGPU), so 40 entries ≈ the last 4-12 s of run-up — enough to see the
// cadence (steady? stalling?) without rewriting a large blob every beat.
const RING_CAP = 40;
const KEY_PREFIX = 'seek-forensics:';

interface ForensicRecord {
    sessionId: string;
    deviceId: string;
    startedAt: string;
    cleanEnd: boolean;
    ring: ForensicBreadcrumb[];
}

// Beat types treated as "actively indexing" for the verdict. Kept as a list
// (not a prefix match) so adding beats later forces a classification decision.
const INDEXING_BEATS = new Set(['index-start', 'index-flush', 'index-progress', 'index-drain']);

export function classifyCrash(ring: ForensicBreadcrumb[]): CrashVerdict {
    const last = ring[ring.length - 1];
    if (!last || last.type === 'session-start') return 'unknown';
    // "Indexing was active" = the last real activity beat is an indexing beat
    // that isn't index-complete. Visibility beats don't overwrite activity, so
    // scan back past them. webgpu-* lifecycle beats (device-created/lost/
    // uncaptured-error) are likewise observations, not activity — a device-lost
    // fired mid-reindex must not flip the verdict away from "indexing"; it
    // shows up in the breadcrumb tail, which is where it discriminates.
    let activity: ForensicBreadcrumb | null = null;
    for (let i = ring.length - 1; i >= 0; i--) {
        const b = ring[i];
        if (b.type.startsWith('visibility-') || b.type.startsWith('webgpu-') || b.type === 'pagehide' || b.type === 'session-start') continue;
        activity = b;
        break;
    }
    const indexing = activity != null && INDEXING_BEATS.has(activity.type);
    if (indexing) {
        return last.vis === 'hidden' ? 'crash-while-indexing-hidden' : 'crash-while-indexing-foreground';
    }
    if (last.vis === 'hidden') return 'evicted-while-hidden';
    return 'crash-foreground';
}

export class Forensics {
    private key: string;
    private record: ForensicRecord;
    // localStorage can be unavailable (private mode, quota); forensics must
    // never break the plugin, so all writes are best-effort behind this flag.
    private usable = true;

    constructor(scope: string, deviceId: string, sessionId: string) {
        this.key = KEY_PREFIX + scope;
        this.record = {
            sessionId,
            deviceId,
            startedAt: new Date().toISOString(),
            cleanEnd: false,
            ring: [],
        };
    }

    // Read the PREVIOUS session's record (if any), classify it, and take over
    // the slot for this session. Returns a crash entry to log, or null when
    // the prior session ended cleanly / never existed. Call once at onload,
    // before the first beat.
    bootInspect(): CrashDetectedEntry | null {
        let prior: ForensicRecord | null = null;
        try {
            const raw = localStorage.getItem(this.key);
            if (raw) prior = JSON.parse(raw) as ForensicRecord;
        } catch {
            // Corrupt record: clear it so forensics recovers next session
            // instead of being dead forever; only an unwritable localStorage
            // disables the layer.
            try { localStorage.removeItem(this.key); } catch { this.usable = false; }
        }
        this.beat('session-start');
        if (!prior || prior.cleanEnd) return null;
        // A record from THIS device only — another device can't share an
        // origin's localStorage, but guard anyway (corrupt/legacy data).
        const ring = Array.isArray(prior.ring) ? prior.ring : [];
        const last = ring.length ? ring[ring.length - 1] : null;
        const gapSeconds = last
            ? Math.max(0, (Date.parse(this.record.startedAt) - Date.parse(last.t)) / 1000)
            : null;
        return {
            type: 'crash-detected',
            timestamp: new Date().toISOString(),
            deadSessionId: prior.sessionId ?? 'unknown',
            verdict: classifyCrash(ring),
            lastBeat: last,
            gapSeconds: gapSeconds != null && Number.isFinite(gapSeconds) ? gapSeconds : null,
            breadcrumbs: ring.slice(-10),
        };
    }

    // Synchronous by design — must complete before the caller proceeds to the
    // work that might kill the process. Never throws.
    beat(type: string, detail?: ForensicBreadcrumb['detail']): void {
        const b: ForensicBreadcrumb = {
            t: new Date().toISOString(),
            type,
            vis: typeof document !== 'undefined' && document.visibilityState === 'hidden' ? 'hidden' : 'visible',
            ...(detail ? { detail } : {}),
        };
        this.record.ring.push(b);
        if (this.record.ring.length > RING_CAP) this.record.ring.splice(0, this.record.ring.length - RING_CAP);
        this.persist();
    }

    // Plugin onunload. Reload/disable/app-quit all pass through here; a
    // session whose record still has cleanEnd=false at next boot died.
    markCleanEnd(): void {
        this.record.cleanEnd = true;
        this.persist();
    }

    private persist(): void {
        if (!this.usable) return;
        try {
            localStorage.setItem(this.key, JSON.stringify(this.record));
        } catch {
            this.usable = false;
        }
    }
}
