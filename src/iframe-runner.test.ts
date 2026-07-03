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
