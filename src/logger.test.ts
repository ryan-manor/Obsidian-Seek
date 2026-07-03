// SeekLogger: per-device append reliability (concurrency-safe writes,
// opportunistic mid-session rotation) + cloned-device deviceId collision
// detection.

import { describe, it, expect, beforeEach, vi } from 'vitest';
import type { App, DataAdapter } from 'obsidian';
import { SeekLogger } from './logger';
import type { InitEntry, ErrorEntry } from './types';

// ---- in-memory DataAdapter fake (mirrors sidecar.test.ts's FakeAdapter) ----

class FakeAdapter {
    files = new Map<string, string>();

    async exists(p: string): Promise<boolean> {
        return this.files.has(p);
    }
    async mkdir(_p: string): Promise<void> {
        /* directories are implicit in this fake — no-op */
    }
    async read(p: string): Promise<string> {
        const v = this.files.get(p);
        if (v === undefined) throw new Error(`ENOENT ${p}`);
        return v;
    }
    async write(p: string, data: string): Promise<void> {
        this.files.set(p, data);
    }
    async append(p: string, data: string): Promise<void> {
        const prev = this.files.get(p);
        if (prev === undefined) throw new Error(`ENOENT ${p}`);
        this.files.set(p, prev + data);
    }
    async remove(p: string): Promise<void> {
        this.files.delete(p);
    }
    async rename(from: string, to: string): Promise<void> {
        const v = this.files.get(from);
        if (v === undefined) throw new Error(`ENOENT ${from}`);
        this.files.set(to, v);
        this.files.delete(from);
    }
    async stat(p: string): Promise<{ size: number; type: 'file' } | null> {
        const v = this.files.get(p);
        return v === undefined ? null : { size: v.length, type: 'file' };
    }
    async list(dir: string): Promise<{ folders: string[]; files: string[] }> {
        const prefix = dir.endsWith('/') || dir === '' ? dir : dir + '/';
        const files = [...this.files.keys()].filter(p => p.startsWith(prefix) && !p.slice(prefix.length).includes('/'));
        return { folders: [], files };
    }
}

function installLocalStorage(): Map<string, string> {
    const map = new Map<string, string>();
    vi.stubGlobal('localStorage', {
        getItem: (k: string) => map.get(k) ?? null,
        setItem: (k: string, v: string) => { map.set(k, v); },
        removeItem: (k: string) => { map.delete(k); },
    });
    return map;
}

function makeApp(adapter: FakeAdapter): App {
    return { vault: { adapter: adapter as unknown as DataAdapter } } as unknown as App;
}

function initEntry(): InitEntry {
    return {
        type: 'init',
        timestamp: new Date().toISOString(),
        schemaVersion: 1,
        buildTimestamp: 't',
        transformersVersion: 'x',
        cdnUrl: 'x',
        iframeReady: true,
        initMs: 0,
        pluginVersion: '0.0.0',
        error: null,
    };
}

describe('SeekLogger.append concurrency', () => {
    beforeEach(() => installLocalStorage());

    it('two concurrent first-appends to the same log file do not lose a line', async () => {
        const adapter = new FakeAdapter();
        const logger = new SeekLogger(makeApp(adapter), 'seek');
        const e1: ErrorEntry = { type: 'error', timestamp: new Date().toISOString(), context: 'a', message: 'm1', stack: null };
        const e2: ErrorEntry = { type: 'error', timestamp: new Date().toISOString(), context: 'b', message: 'm2', stack: null };
        // Fire both without awaiting the first — this is exactly the interleaving
        // that raced adapter.exists()/adapter.write() before the append queue.
        await Promise.all([logger.append(e1), logger.append(e2)]);
        const lines = (await logger.readAll());
        expect(lines).toHaveLength(2);
        const messages = lines.map(l => (l as ErrorEntry).message).sort();
        expect(messages).toEqual(['m1', 'm2']);
    });

    it('many concurrent appends are all preserved, in FIFO order', async () => {
        const adapter = new FakeAdapter();
        const logger = new SeekLogger(makeApp(adapter), 'seek');
        const n = 20;
        await Promise.all(
            Array.from({ length: n }, (_, i) =>
                logger.append({ type: 'error', timestamp: new Date().toISOString(), context: 'c', message: `m${i}`, stack: null })
            )
        );
        const lines = await logger.readAll();
        expect(lines).toHaveLength(n);
        expect(lines.map(l => (l as ErrorEntry).message)).toEqual(Array.from({ length: n }, (_, i) => `m${i}`));
    });
});

describe('SeekLogger opportunistic rotation', () => {
    beforeEach(() => installLocalStorage());

    it('rotates mid-session (no reload) once enough lines have been appended past the byte cap', async () => {
        const adapter = new FakeAdapter();
        const logger = new SeekLogger(makeApp(adapter), 'seek');
        // Pre-seed the log file directly, past MAX_LOG_BYTES, so the very first
        // opportunistic check has something to trim.
        const bigLine = 'x'.repeat(2000) + '\n';
        const path = (logger as unknown as { logPath(): string }).logPath();
        await adapter.mkdir('.obsidian/plugins/seek/logs');
        await adapter.write(path, bigLine.repeat(600)); // ~1.2 MB, past the 1 MB cap
        const before = (await adapter.stat(path))!.size;
        expect(before).toBeGreaterThan(1024 * 1024);

        // ROTATE_CHECK_INTERVAL_APPENDS lines must land before the opportunistic
        // check fires (no plugin reload in between — rotateIfOversize is never
        // called directly here).
        for (let i = 0; i < 200; i++) {
            await logger.append({ type: 'error', timestamp: new Date().toISOString(), context: 'c', message: `m${i}`, stack: null });
        }
        const after = (await adapter.stat(path))!.size;
        expect(after).toBeLessThan(before);
    });

    it('external rotateIfOversize() (main.ts onload chain, called outside appendQueue) cannot race a concurrently queued append() and drop the appended line', async () => {
        const adapter = new FakeAdapter();
        const logger = new SeekLogger(makeApp(adapter), 'seek');
        const path = (logger as unknown as { logPath(): string }).logPath();
        await adapter.mkdir('.obsidian/plugins/seek/logs');
        const bigLine = 'x'.repeat(2000) + '\n';
        await adapter.write(path, bigLine.repeat(600)); // ~1.2 MB, past the 1 MB cap, no crash line yet

        // Gate the tail-truncation write so we can freeze rotateIfOversize() mid-flight
        // (stat + read already done, write pending) — exactly the window where the old,
        // unqueued rotateIfOversize() could interleave with a concurrent append() and
        // have its stale-read-derived write clobber the just-appended line.
        let releaseWrite: () => void = () => {};
        const writeGate = new Promise<void>(res => { releaseWrite = res; });
        const realWrite = adapter.write.bind(adapter);
        let gateHit = false;
        adapter.write = async (p: string, data: string) => {
            if (p === path && !gateHit) {
                gateHit = true;
                await writeGate;
            }
            return realWrite(p, data);
        };

        // Fire-and-forget, exactly as main.ts's onload chain calls it.
        const rotatePromise = logger.rotateIfOversize();
        // Let rotateIfOversize run its stat+read and reach the gated write.
        for (let i = 0; i < 10; i++) await Promise.resolve();

        let appendResolved = false;
        const appendPromise = logger.append(
            { type: 'error', timestamp: new Date().toISOString(), context: 'crash', message: 'crash-line', stack: null }
        ).then(() => { appendResolved = true; });

        // With the fix, append() is queued behind the in-flight rotateIfOversize() and
        // must NOT resolve while its write is still gated — proving the two can never
        // interleave on the file.
        for (let i = 0; i < 10; i++) await Promise.resolve();
        expect(appendResolved).toBe(false);

        releaseWrite();
        await Promise.all([rotatePromise, appendPromise]);

        const finalContent = await adapter.read(path);
        expect(finalContent.length).toBeLessThan(1024 * 1024); // rotation actually trimmed the file
        const lines = finalContent.split('\n').filter(l => l.trim().length > 0);
        expect(lines.at(-1)).toContain('crash-line'); // appended AFTER the rotated tail, not lost
    });
});

describe('SeekLogger cloned-device deviceId collision detection', () => {
    let ls: Map<string, string>;
    beforeEach(() => { ls = installLocalStorage(); });

    it('does not false-positive on a normal single-device reload sequence', async () => {
        const adapter = new FakeAdapter();
        // First load.
        const l1 = new SeekLogger(makeApp(adapter), 'seek');
        await l1.writeInit(initEntry());
        // Second load (same device, same localStorage — deviceId persists).
        const l2 = new SeekLogger(makeApp(adapter), 'seek');
        expect(l2.deviceId).toBe(l1.deviceId);
        await l2.writeInit(initEntry());
        const errors = (await l2.readAll()).filter(e => e.type === 'error') as ErrorEntry[];
        expect(errors.some(e => e.context === 'device-clone-detected')).toBe(false);
        expect(ls.get('seek-device-id-v1')).toBe(l1.deviceId);
    });

    it('detects a foreign write on its own init file and regenerates the deviceId', async () => {
        const adapter = new FakeAdapter();
        const original = new SeekLogger(makeApp(adapter), 'seek');
        await original.writeInit(initEntry()); // gen 1

        // Simulate an iOS backup/restore: a second install clones localStorage
        // wholesale (same deviceId, same last-written-generation) onto a second
        // live device, which then writes its own next generation to the SAME
        // per-device init file (shared path because the deviceId collided).
        const clonedLs = new Map(ls);
        vi.stubGlobal('localStorage', {
            getItem: (k: string) => clonedLs.get(k) ?? null,
            setItem: (k: string, v: string) => { clonedLs.set(k, v); },
            removeItem: (k: string) => { clonedLs.delete(k); },
        });
        const clone = new SeekLogger(makeApp(adapter), 'seek');
        expect(clone.deviceId).toBe(original.deviceId); // the actual collision
        await clone.writeInit(initEntry()); // gen 2, overwrites the shared file

        // Restore the original device's localStorage view and let it write again —
        // its own locally-remembered generation (1) no longer matches the file (2).
        vi.stubGlobal('localStorage', {
            getItem: (k: string) => ls.get(k) ?? null,
            setItem: (k: string, v: string) => { ls.set(k, v); },
            removeItem: (k: string) => { ls.delete(k); },
        });
        await original.writeInit(initEntry());

        const errors = (await original.readAll()).filter(e => e.type === 'error') as ErrorEntry[];
        expect(errors.some(e => e.context === 'device-clone-detected')).toBe(true);
        // Regeneration clears the persisted id so the NEXT load mints a fresh one.
        // (this session still finishes its own writeInit under the old in-memory
        // deviceId, so the generation counter gets re-seeded to a fresh baseline —
        // it's the deviceId that's gone, not the bookkeeping.)
        expect(ls.has('seek-device-id-v1')).toBe(false);
        // Namespaced by pluginId (see logger.ts deviceGenKey doc) — both installs
        // here share pluginId 'seek', so they share this counter's key too.
        expect(ls.get('seek-device-gen-v1:seek')).toBe('1');
    });

    it('a fresh install with no prior generation never false-positives', async () => {
        const adapter = new FakeAdapter();
        const logger = new SeekLogger(makeApp(adapter), 'seek');
        await logger.writeInit(initEntry());
        const errors = (await logger.readAll()).filter(e => e.type === 'error') as ErrorEntry[];
        expect(errors.some(e => e.context === 'device-clone-detected')).toBe(false);
    });

    // R2B2 adversarial finding: two co-installed builds (e.g. seek + seek-prototype)
    // on ONE real, non-cloned device deliberately share DEVICE_ID_KEY (deviceId
    // identifies the physical device — see resolveDeviceId's doc), but each writes
    // to its OWN separate per-pluginId init file. A shared (non-namespaced)
    // generation counter would have each build's writeInit() bump the SAME counter
    // while comparing it against DIFFERENT files, false-positiving a clone collision
    // on a single legitimate machine and wiping the shared deviceId out from under
    // both builds.
    it('two co-installed builds (different pluginId, same device) never false-positive off each other', async () => {
        const adapter = new FakeAdapter();
        const proto = new SeekLogger(makeApp(adapter), 'seek-prototype');
        await proto.writeInit(initEntry());   // proto's own gen 1, proto's own init file

        const pub = new SeekLogger(makeApp(adapter), 'seek');
        expect(pub.deviceId).toBe(proto.deviceId);   // deviceId IS meant to be shared
        await pub.writeInit(initEntry());             // pub's own gen 1 (independent counter), pub's own init file

        // A later reload of proto must not see pub's write as a foreign write on
        // ITS file — they never touched the same file.
        const protoReload = new SeekLogger(makeApp(adapter), 'seek-prototype');
        await protoReload.writeInit(initEntry());

        const protoErrors = (await protoReload.readAll()).filter(e => e.type === 'error') as ErrorEntry[];
        expect(protoErrors.some(e => e.context === 'device-clone-detected')).toBe(false);
        // The shared deviceId must survive — no false regeneration.
        expect(ls.get('seek-device-id-v1')).toBe(proto.deviceId);

        const pubReload = new SeekLogger(makeApp(adapter), 'seek');
        await pubReload.writeInit(initEntry());
        const pubErrors = (await pubReload.readAll()).filter(e => e.type === 'error') as ErrorEntry[];
        expect(pubErrors.some(e => e.context === 'device-clone-detected')).toBe(false);
    });
});
