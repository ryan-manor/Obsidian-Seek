// Phase-0 index-size diagnostic (see "Seek Index Size & Mobile Freeze — Canonical
// Plan"). Pure, dependency-free byte accounting for the IndexedDB stores, plus a
// human-readable renderer. The IDB cursor walk that feeds this lives in
// index-store.ts (`measureSizes`); the physical/quota numbers are spliced in by
// the caller (main.ts) from navigator.storage.estimate(). Keeping the sizing
// RULES and the FORMATTER here — free of any IndexedDB handle — is what makes
// them unit-testable with plain values.
//
// The point of the tool: a hybrid index is mostly TEXT, not vectors, and it grows
// over time. It surfaces the per-store split (vectors / BM25 / chunk text) plus the
// physical−logical "slack". Measurement settled the original question: the BM25
// blob is small (~3.84 MB) and the bulk is LevelDB slack (~85%), but slack is
// HARMLESS on desktop (large quota) and NOT web-reclaimable — Chromium won't
// compact dead SSTs from web code, and a rewrite only grows the store. So this is
// now a TREND/telemetry surface, not a call-to-action: there is no in-place gzip
// and no compaction lever to pull (both retired). gzip lives only on the cross-
// device sync sidecar artifact, not in IDB.

// How a store's per-row value is weighed. Each store holds one shape:
//   bytes    — a Uint8Array / ArrayBuffer (the packed sign-bit `binary` store)
//   quantvec — { q: Int8Array, s: number } (the int8 `embeddings` rerank tier)
//   utf8     — a raw string (the `chunk_body` snippet/refit source)
//   json     — a plain object (chunk_meta / files / meta), weighed as its JSON
//   bm25     — { json, stamp } (the serialized MiniSearch index; the IDB blob is
//              plaintext JSON — gzip applies only to the cross-device sync sidecar,
//              never in IDB — but we size a Uint8Array too, defensively)
export type SizingRule = 'bytes' | 'quantvec' | 'utf8' | 'json' | 'bm25';

export interface StoreSizeRow {
    store: string;  // raw store name, e.g. 'bm25' (stable key for the verdict logic)
    label: string;  // human label, e.g. 'BM25 inverted index'
    rows: number;   // record count
    bytes: number;  // summed logical bytes for this store
}

export interface IndexSizeReport {
    stores: StoreSizeRow[];
    logicalBytes: number;            // Σ stores[].bytes — what we actually store
    // From navigator.storage.estimate(); null when the API / usageDetails bucket
    // is unavailable (older WebViews expose `usage` but not the per-bucket split).
    indexedDbBytes: number | null;   // usageDetails.indexedDB — PHYSICAL index on disk
    cachesBytes: number | null;      // usageDetails.caches — the model bytes (not the index)
    originUsageBytes: number | null; // estimate.usage — whole origin
    quotaBytes: number | null;       // estimate.quota
}

// One shared encoder — UTF-8 byte length is what LevelDB actually persists for
// text, and it counts multi-byte (CJK, emoji) correctly, unlike String.length.
const enc = new TextEncoder();
function utf8Len(s: string): number {
    return enc.encode(s).length;
}

// Weigh a single stored row by its store's shape. Defensive against null/missing
// fields (a cursor can in principle surface a malformed row) — an unweighable
// value contributes 0 rather than throwing and aborting the whole walk.
export function sizeOfRow(rule: SizingRule, value: unknown): number {
    if (value == null) return 0;
    switch (rule) {
        case 'bytes':
            if (value instanceof Uint8Array) return value.byteLength;
            if (value instanceof ArrayBuffer) return value.byteLength;
            return 0;
        case 'quantvec': {
            // { q: Int8Array(d), s: number } → the int8 buffer plus the fp64 scale.
            const q = (value as { q?: ArrayBufferView }).q;
            return (q?.byteLength ?? 0) + 8;
        }
        case 'utf8':
            return typeof value === 'string' ? utf8Len(value) : 0;
        case 'json':
            return utf8Len(JSON.stringify(value));
        case 'bm25': {
            // The MiniSearch blob dominates this store; the stamp is tiny. `json` is
            // a plaintext string in IDB (gzip applies only to the sync sidecar) — we
            // also size a Uint8Array defensively so the tool never lies.
            const r = value as { json?: unknown; stamp?: unknown };
            const blob = typeof r.json === 'string' ? utf8Len(r.json)
                : r.json instanceof Uint8Array ? r.json.byteLength
                : 0;
            return blob + utf8Len(JSON.stringify(r.stamp ?? null));
        }
    }
}

// Slack = physical IndexedDB minus what we logically store: LevelDB's uncompacted
// record versions plus its B-tree/overhead. The "worse over time" ratchet — but
// harmless on desktop (large quota) and not web-reclaimable, so it's a trend
// signal, not an action. Null when we have no physical number.
export function slackBytes(r: IndexSizeReport): number | null {
    if (r.indexedDbBytes == null) return null;
    return Math.max(0, r.indexedDbBytes - r.logicalBytes);
}

function fmtBytes(n: number): string {
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    return `${(n / (1024 * 1024)).toFixed(2)} MB`;
}

// Render the report as a monospace block: a per-store table (largest first),
// the logical total, the physical/slack split, and a one-line verdict naming the
// bigger lever. Pure string-building so both the CLI handler and the settings
// button render identical output.
export function formatIndexSizeReport(r: IndexSizeReport): string {
    const rows = [...r.stores].sort((a, b) => b.bytes - a.bytes);
    const nameW = Math.max(5, ...rows.map(x => x.label.length));
    const lines: string[] = ['Seek index size', ''];

    lines.push(`${'Store'.padEnd(nameW)}  ${'Rows'.padStart(8)}  ${'Size'.padStart(10)}  ${'%'.padStart(5)}`);
    for (const x of rows) {
        const pct = r.logicalBytes > 0 ? (100 * x.bytes / r.logicalBytes) : 0;
        lines.push(`${x.label.padEnd(nameW)}  ${x.rows.toLocaleString().padStart(8)}  ${fmtBytes(x.bytes).padStart(10)}  ${pct.toFixed(1).padStart(5)}`);
    }
    lines.push(`${'logical total'.padEnd(nameW)}  ${''.padStart(8)}  ${fmtBytes(r.logicalBytes).padStart(10)}`);
    lines.push('');

    const slack = slackBytes(r);
    if (r.indexedDbBytes != null && slack != null) {
        const slackPct = r.indexedDbBytes > 0 ? (100 * slack / r.indexedDbBytes) : 0;
        lines.push(`IndexedDB physical: ${fmtBytes(r.indexedDbBytes)}  =  logical ${fmtBytes(r.logicalBytes)} + slack ${fmtBytes(slack)} (${slackPct.toFixed(0)}% LevelDB slack)`);
    } else {
        // iOS WKWebView exposes estimate().usage/quota but NOT the per-bucket
        // usageDetails Chromium adds, so the physical IndexedDB split is unknowable
        // here. Trend the logical total + origin total across opens instead.
        lines.push('IndexedDB physical: unavailable on this platform (no usageDetails.indexedDB — iOS WKWebView).');
        lines.push('  → trend the logical total + origin total across reopens (no per-bucket split here).');
    }
    if (r.cachesBytes != null) lines.push(`Model cache:        ${fmtBytes(r.cachesBytes)}  (not part of the index)`);
    if (r.originUsageBytes != null) {
        lines.push(`Origin total:       ${fmtBytes(r.originUsageBytes)}${r.quotaBytes != null ? ` / quota ${fmtBytes(r.quotaBytes)}` : ''}`);
    }
    lines.push('');

    // Verdict: contextualize, don't prescribe. Both former "levers" are retired —
    // in-place gzip (~4% of a harmless store) and compaction (a footgun that grows
    // the store). Desktop slack is harmless + not web-reclaimable, so the numbers
    // are a trend signal. The only durable size move (cross-device) is the gzipped
    // BM25 sync sidecar, which lives on disk in the index folder, not in IDB.
    const bm25 = rows.find(x => x.store === 'bm25')?.bytes ?? 0;
    if (slack != null) {
        lines.push(`Verdict: ${fmtBytes(slack)} of the ${fmtBytes(r.indexedDbBytes ?? 0)} physical store is LevelDB slack (BM25 blob is ${fmtBytes(bm25)}). Harmless on desktop (large quota) and not web-reclaimable — trend it, don't act on it. A full reindex resets the logical floor but does not shed physical SSTs.`);
    } else {
        lines.push(`Verdict: BM25 blob is ${fmtBytes(bm25)} of ${fmtBytes(r.logicalBytes)} logical. Physical slack isn't measurable on iOS (no usageDetails) — trend the logical + origin totals across reopens.`);
    }
    return lines.join('\n');
}
