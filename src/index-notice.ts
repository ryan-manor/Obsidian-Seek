// The VISIBLE half of the version-identity gate (identity.ts is the invisible half:
// pure "is this index stale?" predicates). This is the pure copy + policy for the
// version-stale warning shown when the local index was built under an older Seek
// version than the running build — used by BOTH the toast (main.ts) and the
// search-modal banner (search-modal.ts), so the copy lives here once and can't drift.
//
// Platform-independent on purpose: the banner is a pure signpost (message + an "Open
// settings" link); the actual reindex lives behind the Settings affordance, which owns
// any platform-specific guardrails. So neither the message nor the action varies by
// device — the modal renders the same thing everywhere.
//
// Only a 'version' degradation surfaces this. The other degraded states stay silent:
//   • 'drift'  — drift-recovery exhausted; a different failure with its own settings
//                affordance. "Index change detected … reindex" would mislead.
//   • null     — e.g. the 'drained' heal (unstamped-but-current index, edits deferred):
//                not a real format change, so no banner.

export type DegradedReason = 'version' | 'drift' | null;
export type IndexHealth = 'healthy' | 'recovering' | 'degraded';

export interface IndexBannerSpec {
    message: string;
}

// One source of truth for the copy (toast + banner). Kept exported so the test asserts
// against the same string the UI shows.
export const INDEX_STALE_MSG = 'Index change detected. Search results may be inaccurate. Please reindex.';

export function indexBannerSpec(health: IndexHealth, reason: DegradedReason): IndexBannerSpec | null {
    if (health !== 'degraded' || reason !== 'version') return null;
    return { message: INDEX_STALE_MSG };
}
