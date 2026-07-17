// Recent searches: the last few COMMITTED queries, shown in the resting modal
// (the otherwise-empty area under the query field).
//
// What counts as committed: opening a result, or closing the modal while
// results are showing (a search answered by the snippet alone) — never the
// debounced keystroke prefixes ("obsi", "obsid", …) whose searches fire while
// the user is still typing. See search-modal.ts captureRecent for the hooks.
//
// Storage is per-device localStorage, NOT data.json: data.json syncs across
// devices, and high-frequency writes there invite the iCloud clobber that
// already forced the logs per-device. The caller's scope string embeds the
// manifest id (a co-installed second Seek build keeps its own history — same
// dual-install scoping as the index DB) and the vault scope (localStorage is
// per-ORIGIN on iOS, shared across vaults — the forensics.ts trap). Same
// construction and best-effort persistence posture as Forensics.
//
// Deliberately tiny: RECENTS_CAP entries stored, so removing a row just leaves
// a shorter list — nothing backfills. No settings surface.

const KEY_PREFIX = 'seek-recents:';
export const RECENTS_CAP = 3;

// Pure MRU push: trim, drop empties, case-insensitive dedup (the newest casing
// wins), newest first, capped. Returns a new array — callers persist the result.
export function pushRecent(list: readonly string[], query: string, cap = RECENTS_CAP): string[] {
    const q = query.trim();
    if (!q) return list.slice(0, cap);
    const norm = q.toLowerCase();
    return [q, ...list.filter(e => e.toLowerCase() !== norm)].slice(0, cap);
}

// Pure removal (the row's ✕): case-insensitive. The list only shrinks.
export function removeRecent(list: readonly string[], query: string): string[] {
    const norm = query.trim().toLowerCase();
    return list.filter(e => e.toLowerCase() !== norm);
}

// localStorage-backed store. All reads/writes are best-effort behind `usable`
// (private mode, quota): recent searches are chrome, and must never break the
// modal. A corrupt record is cleared so the store recovers next write instead
// of being dead forever — mirroring Forensics.
export class RecentSearches {
    private key: string;
    private usable = true;

    constructor(scope: string) {
        this.key = KEY_PREFIX + scope;
    }

    list(): string[] {
        if (!this.usable) return [];
        try {
            const raw = window.localStorage.getItem(this.key);
            if (!raw) return [];
            const parsed: unknown = JSON.parse(raw);
            if (!Array.isArray(parsed)) return [];
            return parsed.filter((e): e is string => typeof e === 'string').slice(0, RECENTS_CAP);
        } catch {
            try { window.localStorage.removeItem(this.key); } catch { this.usable = false; }
            return [];
        }
    }

    push(query: string): void {
        this.persist(pushRecent(this.list(), query));
    }

    remove(query: string): void {
        this.persist(removeRecent(this.list(), query));
    }

    private persist(list: string[]): void {
        if (!this.usable) return;
        try {
            window.localStorage.setItem(this.key, JSON.stringify(list));
        } catch {
            this.usable = false;
        }
    }
}
