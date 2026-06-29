// Tier-2 scenario harness — drives the REAL SearchOrchestrator + REAL IndexStore
// (on fake-indexeddb) against fakes for the two things that aren't fakeable as
// data: the embedder (deterministic math instead of a 250 MB WASM heap) and the
// Vault (an in-memory path→{content,mtime} map instead of Obsidian's file layer).
//
// Why this exists: the unit suite pins each *decision* in isolation
// (classifyFileDelta, shouldStampLiveIdentity, …) with hand-built inputs. The
// bugs that have actually cost time were emergent ORDERING across
// computeDelta → reindexDelta → applyDelta → stamp → drain — invisible to any
// single-decision test. This harness composes the real pipeline under a scripted
// event stream so those convergence/lifecycle bugs become assertable.
//
// The dividing line (see [[Seek Testing Strategy]]): we fake the *inputs* Obsidian
// provides (file events, mtimes, vectors) — never the *behavior* it exhibits
// (WKWebView IDB cost, jetsam, iCloud races). fake-indexeddb is W3C-faithful, not
// WKWebView; the embedder is deterministic, not real WASM. The harness is honest
// about that boundary: logic under faked inputs, never platform behavior.
//
// Construction is inert under Node: BinaryScorerWorker's ctor early-returns when
// __BINARY_WORKER_SRC__ is empty (vitest), and IndexCoordinator's ctor is a field
// assignment. So the real orchestrator constructs with no source seam needed.
import 'fake-indexeddb/auto';            // installs a W3C-faithful indexedDB global
import { TFile } from 'obsidian';        // the test-stub TFile, so `instanceof TFile` holds in the index path
import { IndexStore } from '../index-store';
import { SearchOrchestrator } from '../search';
import { DEFAULT_SETTINGS } from '../types';
import type { App } from 'obsidian';
import type { LocalEmbedder } from '../embedder';

// ── fake Vault: an in-memory map is the entire Obsidian surface the index path reads ──
// search.ts touches exactly: getMarkdownFiles / getFiles / getAbstractFileByPath /
// cachedRead / adapter (sidecar only, off here) and metadataCache.isUserIgnored.
interface VFile { content: string; mtime: number; }

export class FakeVault {
    private files = new Map<string, VFile>();
    // Paths whose cachedRead throws — models a file deleted/evicted BETWEEN the
    // directory listing and the read (the carryover NotFoundError family: the
    // embed pass must skip just that file, not abort the whole batch).
    failReads = new Set<string>();

    // Driver mutators — each is the data-residue of one Obsidian Vault event:
    write(path: string, content: string, mtime: number): void { this.files.set(path, { content, mtime }); }
    touch(path: string, mtime: number): void { const f = this.files.get(path); if (f) f.mtime = mtime; } // iCloud re-stamp
    remove(path: string): void { this.files.delete(path); }

    getMarkdownFiles(): TFile[] { return this.list(p => p.endsWith('.md')); }
    getFiles(): TFile[] { return this.list(() => true); }
    getAbstractFileByPath(path: string): TFile | null {
        const f = this.files.get(path);
        return f ? this.tf(path, f) : null;
    }
    async cachedRead(file: TFile): Promise<string> {
        if (this.failReads.has(file.path)) {
            const e = new Error(`ENOENT: ${file.path}`); (e as { name: string }).name = 'NotFoundError'; throw e;
        }
        const f = this.files.get(file.path);
        if (!f) throw new Error(`cachedRead: ${file.path} not in vault`);
        return f.content;
    }
    // clearDevice / forensics reach for the adapter; sidecarOn() is false in
    // scenarios (indexDir=null), so the index path never dereferences it.
    adapter = {} as never;

    private list(pred: (p: string) => boolean): TFile[] {
        return [...this.files].filter(([p]) => pred(p)).map(([p, f]) => this.tf(p, f));
    }
    private tf(path: string, f: VFile): TFile {
        const t = new TFile();
        t.path = path;
        t.stat = { mtime: f.mtime, ctime: f.mtime, size: f.content.length };
        t.extension = path.split('.').pop() ?? '';   // `f.extension === 'base'` gate in search.ts
        return t;
    }
}

// ── fake embedder: deterministic + content-derived ──────────────────────────
// Byte-stable (re-embedding identical text yields the identical vector, so the
// content-hash gate can be exercised) AND shared tokens raise cosine (so a
// scenario can assert search RESULTS, not just index state). 384-d to match the
// store's default embeddingDim.
const DIM = 384;
export function hashVec(text: string): Float32Array {
    const v = new Float32Array(DIM);
    for (const tok of text.toLowerCase().split(/\W+/)) {
        if (!tok) continue;
        let h = 2166136261;
        for (let i = 0; i < tok.length; i++) { h ^= tok.charCodeAt(i); h = Math.imul(h, 16777619); }
        v[(h >>> 0) % DIM] += 1;
    }
    // L2-normalize, matching the real model's CLS+normalize contract (cosine = dot).
    let norm = 0;
    for (let i = 0; i < DIM; i++) norm += v[i] * v[i];
    norm = Math.sqrt(norm);
    if (norm > 0) for (let i = 0; i < DIM; i++) v[i] /= norm;
    return v;
}

export function fakeEmbedder(): LocalEmbedder {
    const e = {
        loaded: true,
        device: 'wasm',
        dtype: 'q4',
        modelId: 'test-model',
        // Match the REAL return shapes: embedBatch → { vectors, iframeLatencyMs },
        // embed → { vector, iframeLatencyMs }. Vectors aligned to inputs.
        embedBatch: async (texts: string[]) => ({ vectors: texts.map(hashVec), iframeLatencyMs: 0 }),
        embed: async (text: string) => ({ vector: hashVec(text), iframeLatencyMs: 0 }),
        tokenCounts: async (texts: string[]) => texts.map(t => t.split(/\s+/).filter(Boolean).length),
        ensureTokenizer: async () => {},
        recycle: async () => {},
        teardown: async () => {},
    };
    return e as unknown as LocalEmbedder;
}

// ── the scenario driver ─────────────────────────────────────────────────────
// Each helper mutates the vault, then runs the SAME orchestrator entrypoint the
// real Obsidian event handler runs (pinned against search.ts / main.ts):
//   • create/edit/touch/del → computeDelta() + reindexDelta() — the path
//     reconcileOnLoad / flushDirty / runCatchUp drive (main.ts).
//   • coldStart → reindexAll() — the empty-store full build (stamps identity).
export class Scenario {
    vault = new FakeVault();
    store = new IndexStore();
    embedder = fakeEmbedder();
    orch!: SearchOrchestrator;

    async boot(): Promise<void> {
        // Unique DB name per Scenario so tests don't share an origin-scoped
        // IndexedDB (fake-indexeddb is ONE global, exactly like the browser). The
        // uniqueness MUST go in `scope`, not `dbPrefix`: open() only rewrites the
        // db name when scope is truthy (`${dbPrefix}:${scope}`), so a dbPrefix-only
        // call silently keeps the default name and every scenario would collide.
        await this.store.open(`scn-${Math.random().toString(36).slice(2)}`, 'seek-test');
        const app = {
            vault: this.vault,
            metadataCache: { isUserIgnored: () => false },
        } as unknown as App;
        // The orchestrator calls append / appendError / deviceId (grep-pinned).
        const logger = { deviceId: 'test', append: async () => {}, appendError: async () => {} } as never;
        // forensics=null, indexDir=null → sidecarOn() is false (no adapter writes).
        this.orch = new SearchOrchestrator(app, this.store, this.embedder, logger, structuredClone(DEFAULT_SETTINGS));
    }

    // The incremental catch-up the live handlers run: diff persisted vs live,
    // then patch. embed=true mirrors the desktop flushDirty (model already warm).
    // Public so a scenario can re-reconcile with no vault change (convergence:
    // a second reconcile must find nothing dirty and not rebuild).
    async reconcile(embed = true): Promise<void> {
        const { dirty, deleted } = await this.orch.computeDelta();
        await this.orch.reindexDelta(dirty, deleted, { embed });
    }

    // NOTE: the incremental helpers below embed only once identity is stamped.
    // reindexDelta defers the embed phase as a model-drift guard when the store's
    // meta.modelId is unset (an empty store reads as legacy english-r2 ≠ the live
    // model), exactly as it does in production — you cannot incrementally embed
    // into an index whose identity was never claimed. So a scenario indexes the
    // initial corpus with coldStart() (the real first-index path), then uses these
    // for subsequent changes.
    create = async (p: string, body: string, t: number): Promise<void> => { this.vault.write(p, body, t); await this.reconcile(); };
    edit   = async (p: string, body: string, t: number): Promise<void> => { this.vault.write(p, body, t); await this.reconcile(); };
    touch  = async (p: string, t: number): Promise<void> => { this.vault.touch(p, t); await this.reconcile(); };
    del    = async (p: string): Promise<void> => { this.vault.remove(p); await this.reconcile(); };

    // The empty-store cold build — the path that historically ran incremental and
    // never stamped identity, re-healing forever.
    coldStart = (): Promise<unknown> => this.orch.reindexAll();

    async teardown(): Promise<void> {
        // dispose() signals any in-flight embed loop to stop. We deliberately do
        // NOT close the store: each Scenario opens a uniquely-named DB (see boot),
        // so a lingering connection can't versionchange a later one, and leaving it
        // open lets search()'s fire-and-forget warmCaches finish without throwing
        // "IndexStore not opened" against a store we yanked out from under it.
        this.orch?.dispose();
    }
}
