// Investment #2 — openDb VersionError recovery.
//
// The brick this guards against (seek-deploy-branch-gotcha): deploy a build with a
// LOWER DB_VERSION than the one that already wrote IndexedDB → indexedDB.open rejects
// with a VersionError → the rejection escapes onload (no try/catch around the store
// open) → "Failed to load plugin". openDb now catches that one error, nukes the index,
// and reopens empty (the empty-index path triggers the normal first-run reindex).
//
// Hand-rolled IDB stub — NO fake-indexeddb. We only model the surface openDb and
// nukeDatabase touch. Request handlers fire on a NATIVE promise microtask
// (Promise.resolve().then) so the stub is immune to vi.useFakeTimers() faking
// queueMicrotask, while still landing AFTER openDb's synchronous onerror/onsuccess
// assignments — exactly the real IDB ordering the recovery logic assumes.

import { describe, it, expect, vi, afterEach } from 'vitest';
import { openDb, nukeDatabase } from './index-store';

type OpenResult = 'success' | 'version-error' | 'quota-error';
type DeleteResult = 'success' | 'error' | 'blocked';

interface FakeOpts {
    open: (call: number) => OpenResult; // behaviour of the Nth indexedDB.open() (1-based)
    delete?: () => DeleteResult;        // behaviour of indexedDB.deleteDatabase()
    counts?: Record<string, number>;    // objectStore name -> count() result (nukeDatabase regression)
}

function makeFakeIndexedDB(opts: FakeOpts) {
    let openCalls = 0;
    const deleteCalls: string[] = [];
    const counts = opts.counts ?? {};

    // A fake IDBDatabase: only transaction → objectStore → count and close() are
    // reached (by nukeDatabase). openDb's onsuccess also sets .onversionchange.
    function makeDb() {
        return {
            onversionchange: null as unknown,
            close: vi.fn(),
            transaction(_stores: string[], _mode: IDBTransactionMode) {
                return {
                    objectStore(name: string) {
                        return {
                            count() {
                                const req: Record<string, unknown> = {
                                    onsuccess: null, onerror: null, result: counts[name] ?? 0,
                                };
                                void Promise.resolve().then(() => (req.onsuccess as (() => void) | null)?.());
                                return req;
                            },
                        };
                    },
                };
            },
        };
    }

    const indexedDB = {
        open(_name: string, _version: number) {
            const call = ++openCalls;
            const req: Record<string, unknown> = {
                onsuccess: null, onerror: null, onupgradeneeded: null, result: null, error: null,
            };
            void Promise.resolve().then(() => {
                const r = opts.open(call);
                if (r === 'success') {
                    req.result = makeDb();
                    (req.onsuccess as (() => void) | null)?.();
                } else if (r === 'version-error') {
                    req.error = new DOMException('the requested version is older than the existing version', 'VersionError');
                    (req.onerror as (() => void) | null)?.();
                } else {
                    req.error = new DOMException('quota exceeded', 'QuotaExceededError');
                    (req.onerror as (() => void) | null)?.();
                }
            });
            return req;
        },
        deleteDatabase(name: string) {
            deleteCalls.push(name);
            const req: Record<string, unknown> = { onsuccess: null, onerror: null, onblocked: null, error: null };
            void Promise.resolve().then(() => {
                const r = opts.delete?.() ?? 'success';
                if (r === 'success') (req.onsuccess as (() => void) | null)?.();
                else if (r === 'error') {
                    req.error = new DOMException('delete failed', 'UnknownError');
                    (req.onerror as (() => void) | null)?.();
                } else (req.onblocked as (() => void) | null)?.();
            });
            return req;
        },
    };

    return { indexedDB, deleteCalls, openCount: () => openCalls };
}

// Drain a few native-microtask hops (open onerror → deleteDatabase → onblocked) so a
// faked timer is already scheduled before we advance the clock.
async function tick(n = 3): Promise<void> {
    for (let i = 0; i < n; i++) await Promise.resolve();
}

afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
});

describe('openDb VersionError recovery (Investment #2)', () => {
    it('T1: downgrade VersionError → nuke + reopen empty (resolves to a usable db)', async () => {
        const fake = makeFakeIndexedDB({ open: c => (c === 1 ? 'version-error' : 'success') });
        vi.stubGlobal('indexedDB', fake.indexedDB);

        const db = await openDb('seek-index');

        expect(db).toBeTruthy();
        expect(fake.deleteCalls).toEqual(['seek-index']); // deleted exactly once
        expect(fake.openCount()).toBe(2);                 // original open + one reopen
    });

    it('T2: retry STILL VersionError → reject, bounded to ONE delete (no loop)', async () => {
        const fake = makeFakeIndexedDB({ open: () => 'version-error' });
        vi.stubGlobal('indexedDB', fake.indexedDB);

        await expect(openDb('seek-index')).rejects.toMatchObject({ name: 'VersionError' });
        expect(fake.deleteCalls.length).toBe(1); // recovered once; the reopen ran with allowRecovery=false
        expect(fake.openCount()).toBe(2);        // exactly two opens, not an infinite retry
    });

    it('T3: non-downgrade error (QuotaExceededError) → reject, deleteDatabase NEVER called', async () => {
        const fake = makeFakeIndexedDB({ open: () => 'quota-error' });
        vi.stubGlobal('indexedDB', fake.indexedDB);

        await expect(openDb('seek-index')).rejects.toMatchObject({ name: 'QuotaExceededError' });
        expect(fake.deleteCalls.length).toBe(0); // recovery is VersionError-only — never nuke on a transient error
    });

    it('T3b: duck-typed {name:"VersionError"} (NOT a real DOMException) → reject, never deletes', async () => {
        // The `instanceof DOMException` guard is load-bearing: a plain object that only
        // quacks like a VersionError must NOT trigger a destructive nuke. This pins the
        // exact note in the plan — match on type+name, never on a forgeable shape.
        let openCalls = 0;
        const deleteCalls: string[] = [];
        const indexedDB = {
            open() {
                openCalls++;
                const req: Record<string, unknown> = { onsuccess: null, onerror: null, result: null, error: null };
                void Promise.resolve().then(() => {
                    req.error = { name: 'VersionError' }; // duck-typed, not `new DOMException`
                    (req.onerror as (() => void) | null)?.();
                });
                return req;
            },
            deleteDatabase(n: string) {
                deleteCalls.push(n);
                const req: Record<string, unknown> = { onsuccess: null };
                void Promise.resolve().then(() => (req.onsuccess as (() => void) | null)?.());
                return req;
            },
        };
        vi.stubGlobal('indexedDB', indexedDB);

        await expect(openDb('seek-index')).rejects.toMatchObject({ name: 'VersionError' });
        expect(deleteCalls.length).toBe(0); // guard fell through to plain reject
        expect(openCalls).toBe(1);
    });

    it('T4: deleteDatabase blocked for 10s → actionable reject', async () => {
        vi.useFakeTimers();
        const fake = makeFakeIndexedDB({
            open: c => (c === 1 ? 'version-error' : 'success'),
            delete: () => 'blocked',
        });
        vi.stubGlobal('indexedDB', fake.indexedDB);

        const p = openDb('seek-index');
        p.catch(() => {});              // pre-attach: no unhandled-rejection while we pump the clock
        await tick();                   // open onerror → deleteDatabase → onblocked → schedule the 10s timer
        await vi.advanceTimersByTimeAsync(10_000);

        await expect(p).rejects.toThrow(/blocked/i);
        expect(fake.deleteCalls.length).toBe(1);
    });
});

describe('nukeDatabase shape preserved after deleteDbWithBlockGuard extraction (#2 regression)', () => {
    it('returns pre-deletion counts and deletes exactly once', async () => {
        // The extraction sits on the hot reindex path and nothing covered it before.
        // Pin the public contract: same {chunks, embeddings, binary, files} shape out,
        // exactly one deleteDatabase call.
        const fake = makeFakeIndexedDB({
            open: () => 'success',
            counts: { chunk_meta: 7, embeddings: 7, binary: 7, files: 3 },
        });
        vi.stubGlobal('indexedDB', fake.indexedDB);

        const pre = await nukeDatabase('seek-index');

        expect(pre).toEqual({ chunks: 7, embeddings: 7, binary: 7, files: 3 });
        expect(fake.deleteCalls).toEqual(['seek-index']);
    });
});
