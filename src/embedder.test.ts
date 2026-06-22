import { describe, it, expect } from 'vitest';
import { LocalEmbedder } from './embedder';

// A counting stand-in for IframeRunner.embed — returns a distinct vector per
// text so we can assert reference identity on a cache hit, and counts how many
// times the (expensive, real-iframe) path was actually taken.
function fakeRunner() {
    let calls = 0;
    return {
        calls: () => calls,
        embed: async (text: string) => {
            calls++;
            return { vector: new Float32Array([text.length, calls, 0, 0]), latencyMs: 5 };
        },
    };
}

function mk() {
    const e = new LocalEmbedder();
    const fr = fakeRunner();
    (e as unknown as { runner: unknown }).runner = fr;
    (e as unknown as { _loaded: boolean })._loaded = true;
    return { e, fr };
}

// A runner stub modelling init() (with a controllable in-flight gate) + load(),
// so we can assert the search-before-init coalescing contract. The gate starts
// PENDING — call releaseInit() to let init resolve. ready toggles the init
// outcome for the retry test. A `live` flag mirrors a real iframe: load() rejects
// 'iframe not initialized' (like runner.send()) until init has resolved ready —
// so the coalescing test genuinely fails if load() doesn't await init first.
function fakeInitRunner() {
    let initCalls = 0;
    let loadCalls = 0;
    let ready = true;
    let live = false;
    let failLoad = false;
    let release!: () => void;
    const gate = new Promise<void>((res) => { release = res; });
    return {
        initCalls: () => initCalls,
        loadCalls: () => loadCalls,
        setReady: (v: boolean) => { ready = v; },
        setFailLoad: (v: boolean) => { failLoad = v; },
        releaseInit: () => release(),
        dispose: () => { live = false; },     // mirrors IframeRunner.dispose()
        init: async () => {
            initCalls++;
            await gate;
            if (ready) live = true;
            return {
                buildTimestamp: 't', cdnUrl: 'c', transformersVersion: '4.2.0',
                ready, error: ready ? null : 'boom', initMs: 1,
            };
        },
        load: async () => {
            if (!live) throw new Error('iframe not initialized');   // the real bug we're guarding
            if (failLoad) throw new Error('load boom');             // post-init load failure (ADJ-1 / retry)
            loadCalls++;
            return {
                device: 'wasm', dtype: 'q4', coldStartMs: 1,
                warmupMs: null, warmupSkipped: false,
                webgpuAttempted: false, webgpuError: null, glue: null,
            };
        },
    };
}

function mkLoad() {
    const e = new LocalEmbedder();
    const fr = fakeInitRunner();
    (e as unknown as { runner: unknown }).runner = fr;   // NOT _loaded — let load() run for real
    return { e, fr };
}

describe('iframe init coalescing (startup deferral)', () => {
    it('a load() fired before init resolves WAITS for it (no \'iframe not initialized\')', async () => {
        const { e, fr } = mkLoad();
        const initP = e.init();               // onload-style un-awaited init (gate still pending)
        const loadP = e.load('wasm', 'q4');   // search races ahead before the iframe is live
        expect(fr.initCalls()).toBe(1);       // one init kicked off

        fr.releaseInit();                      // iframe "comes up"
        // If load() did NOT await init(), the fake's load() would have thrown
        // 'iframe not initialized' (live still false) — this resolve IS the proof.
        const entry = await loadP;
        expect(entry.actualDevice).toBe('wasm');

        await initP;
        expect(fr.initCalls()).toBe(1);        // load() did NOT trigger a 2nd init — coalesced
        expect(fr.loadCalls()).toBe(1);
    });

    it('load() throws a wrapped \'iframe init failed\' when init resolves not-ready', async () => {
        const { e, fr } = mkLoad();
        fr.setReady(false);
        fr.releaseInit();
        await expect(e.load('wasm', 'q4')).rejects.toThrow(/iframe init failed/);
        expect(fr.loadCalls()).toBe(0);        // bailed before reaching runner.load()
    });

    it('concurrent init() callers share one runner.init()', async () => {
        const { e, fr } = mkLoad();
        const [a, b] = [e.init(), e.init()];
        fr.releaseInit();
        const [ra, rb] = await Promise.all([a, b]);
        expect(fr.initCalls()).toBe(1);        // memoized — single build
        expect(ra).toBe(rb);                   // same memoized entry
    });

    it('a not-ready init is not pinned; the next init() retries a fresh build', async () => {
        const { e, fr } = mkLoad();
        fr.setReady(false);
        fr.releaseInit();
        const first = await e.init();
        expect(first.iframeReady).toBe(false);
        expect(fr.initCalls()).toBe(1);

        fr.setReady(true);
        const second = await e.init();         // memo was nulled on !ready → re-invokes runner.init
        expect(second.iframeReady).toBe(true);
        expect(fr.initCalls()).toBe(2);
    });

    it('teardown() clears the memo so a fresh runner re-inits', async () => {
        const { e, fr } = mkLoad();
        fr.releaseInit();
        await e.init();
        expect((e as unknown as { _initPromise: unknown })._initPromise).not.toBeNull();
        e.teardown();
        expect((e as unknown as { _initPromise: unknown })._initPromise).toBeNull();
    });

    it('recycle() repopulates the memo (live), not leaving it null after rebuild', async () => {
        const { e, fr } = mkLoad();
        fr.releaseInit();
        await e.init();
        await e.recycle();                     // dispose + direct runner.init + repopulate
        expect((e as unknown as { _initPromise: unknown })._initPromise).not.toBeNull();
    });
});

describe('load single-flight + recycle memo (F4 / ADJ-1)', () => {
    it('concurrent load() calls share one runner.load() (F4 single-flight)', async () => {
        const { e, fr } = mkLoad();
        const [a, b] = [e.load('wasm', 'q4'), e.load('wasm', 'q4')];   // two racing loads
        fr.releaseInit();
        const [ra, rb] = await Promise.all([a, b]);
        expect(fr.loadCalls()).toBe(1);   // coalesced — not two concurrent ~250 MB model loads (jetsam)
        expect(ra).toBe(rb);              // same LoadEntry
    });

    it('the load latch clears after settle so a later load runs fresh, not cached', async () => {
        const { e, fr } = mkLoad();
        fr.releaseInit();
        await e.load('wasm', 'q4');
        expect(fr.loadCalls()).toBe(1);
        await e.load('wasm', 'q4');        // a model switch must NOT return the first memoized entry
        expect(fr.loadCalls()).toBe(2);
    });

    it('a failed load clears the latch so a retry can proceed (F4)', async () => {
        const { e, fr } = mkLoad();
        fr.setReady(false);                // init resolves not-ready → load() throws before runner.load()
        fr.releaseInit();
        await expect(e.load('wasm', 'q4')).rejects.toThrow(/iframe init failed/);
        expect((e as unknown as { _loadPromise: unknown })._loadPromise).toBeNull();
    });

    it('recycle() nulls the init memo when the post-init load fails (ADJ-1)', async () => {
        const { e, fr } = mkLoad();
        fr.releaseInit();
        await e.init();                    // memo populated, iframe live
        fr.setFailLoad(true);
        await expect(e.recycle()).rejects.toThrow(/load boom/);
        // init() succeeded inside recycle but load() failed: the memo must NOT pin a
        // resolved "ready" entry for a pipeline that never loaded.
        expect((e as unknown as { _initPromise: unknown })._initPromise).toBeNull();
    });
});

describe('query-embed LRU cache', () => {
    it('miss then hit: the second identical query skips the runner', async () => {
        const { e, fr } = mk();
        const a = await e.embed('swiss hotels');
        expect(a.cacheHit).toBe(false);
        expect(fr.calls()).toBe(1);

        const b = await e.embed('swiss hotels');
        expect(b.cacheHit).toBe(true);
        expect(b.iframeLatencyMs).toBe(0);
        expect(b.vector).toBe(a.vector);     // same reference, not re-embedded
        expect(fr.calls()).toBe(1);
    });

    it('distinct queries each miss; repeats hit', async () => {
        const { e, fr } = mk();
        await e.embed('a');
        await e.embed('b');
        await e.embed('a');                  // hit
        expect(fr.calls()).toBe(2);
    });

    it('evicts the LRU head past the cap, and access protects an entry', async () => {
        const { e, fr } = mk();
        const MAX = (LocalEmbedder as unknown as { QUERY_EMBED_CACHE_MAX: number }).QUERY_EMBED_CACHE_MAX;
        for (let i = 0; i < MAX; i++) await e.embed('q' + i);
        expect(fr.calls()).toBe(MAX);

        // Touch q0 -> moves it to the tail; the new LRU head is q1.
        expect((await e.embed('q0')).cacheHit).toBe(true);
        expect(fr.calls()).toBe(MAX);

        // Overflow by one -> evicts the head (q1), not the freshly-touched q0.
        await e.embed('qNew');
        expect(fr.calls()).toBe(MAX + 1);

        expect((await e.embed('q1')).cacheHit).toBe(false);   // q1 was evicted
        expect((await e.embed('q0')).cacheHit).toBe(true);    // q0 survived via recency
    });
});
