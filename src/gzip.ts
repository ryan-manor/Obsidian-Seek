// gzip helpers for the cross-device BM25 SYNC artifact (Index Size plan, Phase 3).
//
// gzip is used here ONLY for the sync payload — a fresh serialization of pure signal
// (the BM25 MiniSearch postings) that deflates ~5-8× — and NEVER for in-IDB storage.
// The in-place idea was retired: the desktop store's bulk is LevelDB SST slack gzip
// can't reclaim, and the live blob is only ~3.84 MB of a harmless store. On the wire
// there is no slack, so the deflation actually lands (3.84 MB → ~0.5-0.8 MB synced),
// and it buys the real win: a fresh/evicted device LOADS another device's BM25 index
// instead of refitting it over all bodies (the cold-start freeze).
//
// CompressionStream / DecompressionStream are present on Electron (Chromium) and
// WKWebView (iOS 16.4+; the phone is iOS 26) and on Node 18+ (so the round-trip is
// unit-tested here). gzipAvailable() lets callers degrade gracefully on an older
// WebView — a missing artifact just means the consumer refits, exactly as today.

export function gzipAvailable(): boolean {
    return typeof CompressionStream !== 'undefined' && typeof DecompressionStream !== 'undefined';
}

// Deflate a UTF-8 string to gzip bytes. Streams through a Blob → CompressionStream →
// Response so the whole thing stays off the main thread's synchronous path.
export async function gzipString(text: string): Promise<Uint8Array> {
    const stream = new Blob([text]).stream().pipeThrough(new CompressionStream('gzip'));
    const buf = await new Response(stream).arrayBuffer();
    return new Uint8Array(buf);
}

// Inflate gzip bytes back to a UTF-8 string. Throws on a corrupt/truncated stream
// (DecompressionStream rejects), which every caller wraps in try/catch so a torn
// artifact degrades to a refit rather than a broken load.
export async function gunzipToString(bytes: ArrayBuffer | Uint8Array): Promise<string> {
    const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    // `as BlobPart`: TS 5.7 splits Uint8Array<ArrayBufferLike> from the BufferSource
    // the Blob ctor wants (SharedArrayBuffer guard). The value is a plain Uint8Array
    // at runtime; the cast just bridges the over-strict lib type.
    const stream = new Blob([u8 as BlobPart]).stream().pipeThrough(new DecompressionStream('gzip'));
    return new Response(stream).text();
}
