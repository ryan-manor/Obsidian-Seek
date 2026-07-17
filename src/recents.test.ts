// Recent searches: the pure MRU logic plus the localStorage store's
// resilience posture (corrupt records recover, unusable storage goes quiet).

import { describe, it, expect } from 'vitest';
import { pushRecent, removeRecent, RecentSearches, RECENTS_CAP } from './recents';

describe('pushRecent', () => {
    it('prepends the newest query', () => {
        expect(pushRecent(['b', 'c'], 'a')).toEqual(['a', 'b', 'c']);
    });

    it('caps at RECENTS_CAP with no backfill beyond it', () => {
        const out = pushRecent(['a', 'b', 'c'], 'd');
        expect(out).toEqual(['d', 'a', 'b']);
        expect(out.length).toBe(RECENTS_CAP);
    });

    it('dedups case-insensitively, newest casing wins and moves to front', () => {
        expect(pushRecent(['x', 'Seek Plan', 'y'], 'seek plan')).toEqual(['seek plan', 'x', 'y']);
    });

    it('re-pushing the current front is a no-op apart from casing', () => {
        expect(pushRecent(['a', 'b'], 'a')).toEqual(['a', 'b']);
    });

    it('trims whitespace and ignores empty/whitespace queries', () => {
        expect(pushRecent([], '  spaced out  ')).toEqual(['spaced out']);
        expect(pushRecent(['a'], '   ')).toEqual(['a']);
        expect(pushRecent([], '')).toEqual([]);
    });

    it('serialized filter queries round-trip verbatim', () => {
        const q = 'tag:meetings [status:open] roadmap';
        expect(pushRecent([], q)).toEqual([q]);
    });
});

describe('removeRecent', () => {
    it('removes case-insensitively and leaves the rest in order', () => {
        expect(removeRecent(['A', 'b', 'C'], 'c')).toEqual(['A', 'b']);
        expect(removeRecent(['A', 'b', 'C'], 'a')).toEqual(['b', 'C']);
    });

    it('is a no-op when the query is absent', () => {
        expect(removeRecent(['a', 'b'], 'zzz')).toEqual(['a', 'b']);
    });
});

// Minimal in-memory localStorage on the test-setup `window` (node has no DOM;
// the store only needs getItem/setItem/removeItem semantics) — the same shape
// forensics.test.ts stubs.
function installLocalStorage(): Map<string, string> {
    const map = new Map<string, string>();
    (window as unknown as { localStorage: unknown }).localStorage = {
        getItem: (k: string) => map.get(k) ?? null,
        setItem: (k: string, v: string) => { map.set(k, v); },
        removeItem: (k: string) => { map.delete(k); },
    };
    return map;
}

describe('RecentSearches store', () => {
    it('push → list round-trips through localStorage, scoped by key', () => {
        const map = installLocalStorage();
        const store = new RecentSearches('seek:vaultA');
        store.push('first');
        store.push('second');
        expect(store.list()).toEqual(['second', 'first']);
        expect(map.has('seek-recents:seek:vaultA')).toBe(true);
        // A different scope reads its own (empty) slot.
        expect(new RecentSearches('seek:vaultB').list()).toEqual([]);
    });

    it('remove drops the row', () => {
        installLocalStorage();
        const store = new RecentSearches('s');
        store.push('a');
        store.push('b');
        store.remove('a');
        expect(store.list()).toEqual(['b']);
    });

    it('a corrupt record reads as empty, is cleared, and recovers', () => {
        const map = installLocalStorage();
        map.set('seek-recents:s', '{not json');
        const store = new RecentSearches('s');
        expect(store.list()).toEqual([]);
        store.push('fresh');
        expect(store.list()).toEqual(['fresh']);
    });

    it('a non-array or mixed-type record is sanitized', () => {
        const map = installLocalStorage();
        map.set('seek-recents:s', '{"nope":1}');
        expect(new RecentSearches('s').list()).toEqual([]);
        map.set('seek-recents:s', '["ok", 42, null, "also ok"]');
        expect(new RecentSearches('s').list()).toEqual(['ok', 'also ok']);
    });

    it('an unwritable localStorage goes quiet instead of throwing', () => {
        installLocalStorage();
        (window as unknown as { localStorage: { setItem: () => void } }).localStorage.setItem = () => {
            throw new Error('quota');
        };
        const store = new RecentSearches('s');
        expect(() => store.push('a')).not.toThrow();
        expect(store.list()).toEqual([]);
    });
});
