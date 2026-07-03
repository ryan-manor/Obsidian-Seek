import { describe, it, expect, vi, afterEach } from 'vitest';
import { IframeRunner, buildChildScript } from './iframe-runner';

// F5 — per-RPC timeout. A jetsam-killed iframe child never replies; without the
// timeout the parent promise hangs forever, stranding the embed catch's
// recycle+retry (search.ts embedOneBatch). We can't build a real srcdoc iframe in
// the node test env, so inject a live-looking iframe whose contentWindow.postMessage
// is a no-op (the child never answers) and drive the timer with fake timers.
afterEach(() => { vi.useRealTimers(); });

function withDeadIframe(): IframeRunner {
    const r = new IframeRunner();
    (r as unknown as { iframe: { contentWindow: { postMessage: () => void } } }).iframe = {
        contentWindow: { postMessage: () => { /* child never replies */ } },
    };
    return r;
}

describe('IframeRunner per-RPC timeout (F5)', () => {
    it('rejects a never-answered embedBatch with a recoverable TIMEOUT error', async () => {
        vi.useFakeTimers();
        const r = withDeadIframe();
        const p = r.embedBatch(['hello']);
        // tagged TIMEOUT (not DISPOSED) so the embed catch recycles+retries.
        const assertion = expect(p).rejects.toMatchObject({ code: 'TIMEOUT' });
        await vi.advanceTimersByTimeAsync(60_001);   // RPC_TIMEOUT_MS + 1
        await assertion;
    });

    it('a load RPC uses the longer ceiling (still pending past the embed timeout)', async () => {
        vi.useFakeTimers();
        const r = withDeadIframe();
        const p = r.load('some/repo', 'wasm', 'q4', true);
        let settled = false;
        p.then(() => { settled = true; }, () => { settled = true; });
        await vi.advanceTimersByTimeAsync(60_001);   // past the 60s embed ceiling
        expect(settled).toBe(false);                 // load gets LOAD_RPC_TIMEOUT_MS (180s), not 60s
        const assertion = expect(p).rejects.toMatchObject({ code: 'TIMEOUT' });
        await vi.advanceTimersByTimeAsync(120_001);   // total > 180s
        await assertion;
    });

    it('an uninitialized runner rejects immediately, not via the timeout', async () => {
        const r = new IframeRunner();   // no iframe injected
        await expect(r.embedBatch(['x'])).rejects.toThrow(/not initialized/);
    });
});

// Iframe child's RPC dispatch runs inside a srcdoc'd module script, which we
// can't execute in the node test env — so assert on the emitted source text
// that the child rejects postMessage events not sourced from window.parent
// before it ever reaches the RPC dispatcher (mirrors the parent-side
// `event.source !== this.iframe.contentWindow` guard in buildIframe()).
describe('iframe child message handler — source check', () => {
    it('gates RPC dispatch on event.source === window.parent', () => {
        const script = buildChildScript('https://example.com/cdn', 384);
        const handlerStart = script.indexOf("addEventListener('message', async (event)");
        expect(handlerStart).toBeGreaterThan(-1);
        const guardIdx = script.indexOf('event.source !== window.parent', handlerStart);
        const dispatchIdx = script.indexOf("data.type === 'load'", handlerStart);
        expect(guardIdx).toBeGreaterThan(handlerStart);
        expect(guardIdx).toBeLessThan(dispatchIdx);
    });
});

// tx.js pins non-WebKit engines to the asyncify ORT build for EVERY device,
// but that build has no CPU GatherBlockQuantized kernel — a device:'wasm'
// session can never load the shipped GBQ4 model on it (the r/ObsidianMD
// desktop failure; also all of Android). The child must rewrite the asyncify
// pin back to the plain build on BOTH wasm attempts (initial + SIMD retry).
describe('iframe child WASM path — plain-glue pin (CPU GBQ kernel)', () => {
    it('emits the asyncify→plain wasmPaths rewrite and applies it on the wasm path', () => {
        const script = buildChildScript('https://example.com/cdn', 384);
        // The override rewrites BOTH the .mjs glue and the .wasm binary.
        expect(script).toContain("'ort-wasm-simd-threaded.asyncify.mjs', 'ort-wasm-simd-threaded.mjs'");
        expect(script).toContain("'ort-wasm-simd-threaded.asyncify.wasm', 'ort-wasm-simd-threaded.wasm'");
        // Applied on the wasm fallback: initial attempt + the 'no available
        // backend' SIMD retry each get a fresh module instance, so each must
        // re-apply the override. Match the call-site text (assignment form),
        // not the bare identifier — indexOf on the identifier alone would
        // count the function DECLARATION as the first hit and pass even if
        // the retry-path re-application were deleted.
        const first = script.indexOf('wasmGlue = overrideGlueForWasm(env)');
        const second = script.indexOf('wasmGlue = overrideGlueForWasm(env)', first + 1);
        expect(first).toBeGreaterThan(-1);
        expect(second).toBeGreaterThan(first);
    });

    it('no-ops when wasmPaths is not the asyncify pin (WebKit plain path)', () => {
        const script = buildChildScript('https://example.com/cdn', 384);
        const fnStart = script.indexOf('function overrideGlueForWasm');
        expect(fnStart).toBeGreaterThan(-1);
        const guardIdx = script.indexOf("includes('ort-wasm-simd-threaded.asyncify.mjs')", fnStart);
        const rewriteIdx = script.indexOf('wp.mjs = ', fnStart);
        expect(guardIdx).toBeGreaterThan(fnStart);
        expect(guardIdx).toBeLessThan(rewriteIdx);
    });
});

// When 'auto' falls through to WASM and WASM ALSO fails, the child must not
// throw only the terminal wasm error — that discards webgpuError before
// loadModel can return a LoadResult, so the diagnostic report shows the wasm
// session error but never WHY WebGPU fell back (the blind spot in the
// r/ObsidianMD report). Assert the emitted child preserves both causes, on
// both wasm attempts.
describe('iframe child WASM fallback — preserves the WebGPU failure cause', () => {
    it('re-throws with both webgpu and wasm error context when webgpu was attempted', () => {
        const script = buildChildScript('https://example.com/cdn', 384);
        // The combined throw fires only when a WebGPU attempt recorded an error.
        expect(script).toContain('webgpuAttempted && webgpuError');
        expect(script).toContain('model load failed on both paths');
        expect(script).toMatch(/webgpu: '\s*\+\s*webgpuError/);
        // Both the initial wasm attempt and the SIMD retry route their
        // failures through the combining helper.
        const first = script.indexOf('wasmFail(e)');
        const second = script.indexOf('wasmFail(e2)');
        expect(first).toBeGreaterThan(-1);
        expect(second).toBeGreaterThan(first);
    });
});
