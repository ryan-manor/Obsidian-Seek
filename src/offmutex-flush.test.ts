// 1C — sidecar flush outside the write mutex. embedAndCommitFiles used to
// bulkAppend the pass's tier records (plus the meta read/write) INSIDE its
// runExclusive critical section, so searches (ensureFrame waiting on
// currentDelta) and queued deltas sat behind pure sidecar file IO. Now the
// engine only PACKAGES a SidecarFlushJob; reindexAll / reindexDelta run it after
// the mutex releases, chained FIFO, still awaited before their promise resolves.
// These tests pin the three halves of that contract against the REAL engine
// (Tier-2: real orchestrator + fake-indexeddb + fake vault/adapter/embedder):
//   1. shard writes happen with the mutex already released (isWriting() false),
//      while "pass resolved ⇒ sidecar flushed" still holds;
//   2. a queued pass's WHOLE critical section can run while the prior pass's
//      flush is still in flight — and the chained flushes land in pass order;
//   3. a flush failure logs sidecar-commit and never fails the pass.
import 'fake-indexeddb/auto';
import { describe, it, expect, afterEach } from 'vitest';
import type { App, DataAdapter } from 'obsidian';
import { IndexStore } from './index-store';
import { SearchOrchestrator } from './search';
import { DEFAULT_SETTINGS } from './types';
import { FakeVault, fakeEmbedder } from './test-harness/scenario';
import { listDeviceShards, jsonlPathFor } from './sidecar';

const INDEX_DIR = 'plugins/seek-test/index';
const DEVICE = 'test';

// Same in-memory DataAdapter shape as sidecar.test.ts — the flush path touches
// exists/mkdir/read/write/append/readBinary/writeBinary/rename/remove/list/stat.
class FakeAdapter {
    files = new Map<string, string>();
    bins = new Map<string, ArrayBuffer>();

    async exists(p: string): Promise<boolean> {
        if (this.files.has(p) || this.bins.has(p)) return true;
        const prefix = p.endsWith('/') ? p : p + '/';
        for (const k of [...this.files.keys(), ...this.bins.keys()]) if (k.startsWith(prefix)) return true;
        return false;
    }
    async mkdir(_p: string): Promise<void> { /* directories are implicit */ }
    async read(p: string): Promise<string> {
        const v = this.files.get(p);
        if (v === undefined) throw new Error(`ENOENT ${p}`);
        return v;
    }
    async write(p: string, data: string): Promise<void> { this.files.set(p, data); }
    async append(p: string, data: string): Promise<void> { this.files.set(p, (this.files.get(p) ?? '') + data); }
    async readBinary(p: string): Promise<ArrayBuffer> {
        const v = this.bins.get(p);
        if (v === undefined) throw new Error(`ENOENT ${p}`);
        return v;
    }
    async writeBinary(p: string, buf: ArrayBuffer): Promise<void> { this.bins.set(p, buf); }
    async rename(from: string, to: string): Promise<void> {
        if (this.bins.has(from)) { this.bins.set(to, this.bins.get(from)!); this.bins.delete(from); }
        else if (this.files.has(from)) { this.files.set(to, this.files.get(from)!); this.files.delete(from); }
        else throw new Error(`ENOENT ${from}`);
    }
    async remove(p: string): Promise<void> { this.bins.delete(p); this.files.delete(p); }
    async list(dir: string): Promise<{ folders: string[]; files: string[] }> {
        const folders = new Set<string>();
        const files: string[] = [];
        const prefix = dir.endsWith('/') ? dir : dir + '/';
        for (const p of [...this.files.keys(), ...this.bins.keys()]) {
            if (!p.startsWith(prefix)) continue;
            const rest = p.slice(prefix.length);
            const slash = rest.indexOf('/');
            if (slash >= 0) folders.add(prefix + rest.slice(0, slash));
            else files.push(p);
        }
        return { folders: [...folders], files };
    }
    async stat(p: string): Promise<{ size: number; type: 'file' } | null> {
        if (this.bins.has(p)) return { size: this.bins.get(p)!.byteLength, type: 'file' };
        if (this.files.has(p)) return { size: this.files.get(p)!.length, type: 'file' };
        return null;
    }
}

interface Rig {
    vault: FakeVault;
    fa: FakeAdapter;
    store: IndexStore;
    orch: SearchOrchestrator;
    errors: string[];
}

async function boot(): Promise<Rig> {
    const vault = new FakeVault();
    const fa = new FakeAdapter();
    // FakeVault ships adapter as an inert stub (scenarios run sidecar-off); this
    // rig swaps in a real fake so sidecarOn() paths have a live file surface.
    (vault as unknown as { adapter: unknown }).adapter = fa;
    const store = new IndexStore();
    // Unique scope per rig — fake-indexeddb is one origin-scoped global.
    await store.open(`omx-${Math.random().toString(36).slice(2)}`, 'seek-test');
    const app = { vault, metadataCache: { isUserIgnored: () => false } } as unknown as App;
    const errors: string[] = [];
    const logger = {
        deviceId: DEVICE,
        append: async () => {},
        appendError: async (k: string) => { errors.push(k); },
    } as never;
    const orch = new SearchOrchestrator(app, store, fakeEmbedder(), logger, structuredClone(DEFAULT_SETTINGS), null, INDEX_DIR);
    return { vault, fa, store, orch, errors };
}

// Poll helper — deterministic conditions only (IDB commit visibility), never
// wall-clock behavior.
async function until(cond: () => Promise<boolean> | boolean, what: string): Promise<void> {
    for (let i = 0; i < 400; i++) {
        if (await cond()) return;
        await new Promise(r => setTimeout(r, 5));
    }
    throw new Error(`timed out waiting for: ${what}`);
}

describe('1C — sidecar flush outside the write mutex', () => {
    let active: Rig | null = null;
    afterEach(() => { active?.orch.dispose(); active = null; });

    it('shard writes happen with the mutex released but the busy signal held; resolve ⇒ flushed still holds', async () => {
        const s = await boot();
        active = s;
        s.vault.write('a.md', 'alpha note about mountain weather patterns', 1000);
        s.vault.write('b.md', 'beta note about sourdough starter hydration', 1000);
        // The discriminating pair, probed AT each shard write: the write MUTEX
        // (coord) must already be free — that's the 1C win — while the
        // orchestrator-level busy signal must still be up, so the reconcile poll
        // and runFullReindex's stacking guard treat the in-flight flush as
        // "still indexing" (a second full pass slipping in here would nuke +
        // re-embed back-to-back and double the sidecar with litter).
        const coord = (s.orch as unknown as { coord: { isWriting(): boolean } }).coord;
        const mutexHeldDuringShardWrite: boolean[] = [];
        const busyDuringShardWrite: boolean[] = [];
        const realWriteBinary = s.fa.writeBinary.bind(s.fa);
        s.fa.writeBinary = async (p: string, buf: ArrayBuffer) => {
            if (p.includes('embeddings.')) {
                mutexHeldDuringShardWrite.push(coord.isWriting());
                busyDuringShardWrite.push(s.orch.isWriting());
            }
            return realWriteBinary(p, buf);
        };

        const entry = await s.orch.reindexAll();
        expect(entry.pass).toBe(true);
        // Durability contract unchanged: by the time reindexAll resolves, the
        // pass's records are on disk (shards + jsonl refs)…
        const shards = await listDeviceShards(s.fa as unknown as DataAdapter, INDEX_DIR, DEVICE);
        expect(shards.length).toBeGreaterThan(0);
        const jsonl = s.fa.files.get(jsonlPathFor(INDEX_DIR, DEVICE)) ?? '';
        expect(jsonl.trim().split('\n').length).toBeGreaterThan(0);
        // …and the busy signal dropped once the flush landed.
        expect(s.orch.isWriting()).toBe(false);
        // The mechanism: every shard write ran AFTER runExclusive released, yet
        // inside the busy window.
        expect(mutexHeldDuringShardWrite.length).toBeGreaterThan(0);
        expect(mutexHeldDuringShardWrite).not.toContain(true);
        expect(busyDuringShardWrite).not.toContain(false);
    });

    it('a queued pass runs its critical section while the prior flush is in flight; flushes land FIFO', async () => {
        const s = await boot();
        active = s;
        s.vault.write('a.md', 'seed corpus note about telescope collimation', 1000);
        await s.orch.reindexAll();   // cold build stamps identity; its flush completes

        // Gate the NEXT shard write: pass A's off-mutex flush blocks on it.
        let release!: () => void;
        const gate = new Promise<void>(r => { release = r; });
        let gatedOnce = false;
        const realWriteBinary = s.fa.writeBinary.bind(s.fa);
        s.fa.writeBinary = async (p: string, buf: ArrayBuffer) => {
            if (p.includes('embeddings.') && !gatedOnce) { gatedOnce = true; await gate; }
            return realWriteBinary(p, buf);
        };

        s.vault.write('a2.md', 'pass A note on winter camping stove fuel', 2000);
        const passA = s.orch.reindexDelta(['a2.md'], [], { embed: true });

        s.vault.write('b2.md', 'pass B note on fermentation temperature control', 3000);
        let passBDone = false;
        const passB = s.orch.reindexDelta(['b2.md'], [], { embed: true }).then(() => { passBDone = true; });

        // B's WHOLE critical section completes while A's flush is still gated —
        // the mutex is demonstrably free during in-flight sidecar IO. (Before 1C
        // this deadlocked the test: A held the mutex through the gated write, so
        // B's commit could never land.)
        await until(async () => (await s.store.getFileRecord('b2.md')) !== undefined, 'pass B IDB commit');
        expect(passBDone).toBe(false);   // …but B still awaits its flush, chained FIFO behind A's

        release();
        await Promise.all([passA, passB]);

        // Both flushes landed, in pass order: A's records (mtime 2000) sit at a
        // strictly lower shard seq than B's (mtime 3000).
        const lines = (s.fa.files.get(jsonlPathFor(INDEX_DIR, DEVICE)) ?? '').trim().split('\n')
            .map(l => JSON.parse(l) as { seq: number; mtime: number });
        const seqA = lines.filter(l => l.mtime === 2000).map(l => l.seq);
        const seqB = lines.filter(l => l.mtime === 3000).map(l => l.seq);
        expect(seqA.length).toBeGreaterThan(0);
        expect(seqB.length).toBeGreaterThan(0);
        expect(Math.max(...seqA)).toBeLessThan(Math.min(...seqB));
    });

    // Pins PRESERVED behavior, not a 1C discriminator: the old inline flush had
    // the same swallow-and-log catch. Kept as the regression guard that the move
    // off-mutex (and the chain's own rejection swallowing) didn't change it.
    it('a flush failure logs sidecar-commit and never fails the pass', async () => {
        const s = await boot();
        active = s;
        s.vault.write('a.md', 'note that will index fine into IDB regardless', 1000);
        const realWriteBinary = s.fa.writeBinary.bind(s.fa);
        s.fa.writeBinary = async (p: string, buf: ArrayBuffer) => {
            if (p.includes('embeddings.')) throw new Error('disk detached');
            return realWriteBinary(p, buf);
        };

        const entry = await s.orch.reindexAll();   // resolves despite the flush failing
        expect(entry.pass).toBe(true);
        expect((await s.store.count()).chunks).toBeGreaterThan(0);   // IDB commit untouched
        expect(s.errors).toContain('sidecar-commit');
        expect(await listDeviceShards(s.fa as unknown as DataAdapter, INDEX_DIR, DEVICE)).toEqual([]);
    });

    it('a job still queued at dispose() is skipped — no detached writes after unload', async () => {
        const s = await boot();
        active = s;
        s.vault.write('a.md', 'seed note about tide pool ecology surveys', 1000);
        await s.orch.reindexAll();

        // Gate pass A's flush; queue pass B behind it on the chain.
        let release!: () => void;
        const gate = new Promise<void>(r => { release = r; });
        let gatedOnce = false;
        const realWriteBinary = s.fa.writeBinary.bind(s.fa);
        s.fa.writeBinary = async (p: string, buf: ArrayBuffer) => {
            if (p.includes('embeddings.') && !gatedOnce) { gatedOnce = true; await gate; }
            return realWriteBinary(p, buf);
        };
        s.vault.write('a2.md', 'pass A note on kelp forest canopy density', 2000);
        const passA = s.orch.reindexDelta(['a2.md'], [], { embed: true });
        s.vault.write('b2.md', 'pass B note on urchin barren recovery rates', 3000);
        const passB = s.orch.reindexDelta(['b2.md'], [], { embed: true });
        // Dispose only after B's critical section committed (the engine polls
        // `disposed` and would otherwise abort B's embed loop mid-pass) — at this
        // point B's flush job is queued on the chain behind A's gated one.
        await until(async () => (await s.store.getFileRecord('b2.md')) !== undefined, 'pass B IDB commit');
        s.orch.dispose();
        release();
        await Promise.all([passA, passB]);   // both resolve; neither hangs on dispose

        // A's job had already started before dispose (harmless completion); B's
        // queued job hit the disposed fence and wrote nothing — the unload window
        // can't produce detached sidecar writes that would race a re-enabled
        // instance's fresh dir-lock map (the #91 republish invariant).
        const lines = (s.fa.files.get(jsonlPathFor(INDEX_DIR, DEVICE)) ?? '').trim().split('\n')
            .map(l => JSON.parse(l) as { mtime: number });
        expect(lines.some(l => l.mtime === 2000)).toBe(true);
        expect(lines.some(l => l.mtime === 3000)).toBe(false);
    });
});
