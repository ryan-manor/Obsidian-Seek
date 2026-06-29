# Changelog

All notable changes to Seek are documented here. This project adheres to [Semantic Versioning](https://semver.org/).

## 1.0.0

Initial public release. Seek is a hybrid (lexical + semantic) search plugin for Obsidian, built on a quantized, sync-friendly index that stays current across devices without re-embedding on each one.

### Search & relevance
- Typed-value query filters: numeric comparison (e.g. `[price>50]`) and date ranges (`before:` / `after:`).
- Field-weight tuning from a fresh relevance evaluation — stronger body-content weighting and a higher dense-fusion weight for better-ranked results.
- Hardened the lexical coordination soft-AND so multi-term queries favor documents that match more of the query.
- Dense-channel hygiene: cleaner body and heading text, with cross-surface de-duplication before embedding.
- Converged tokenization across the surfaces that build, match, and enumerate terms, so identical text tokenizes identically everywhere.

### Query filter menu
- The `[` filter menu is keyed by property type and value shape: numeric keys show real note counts; date keys are kept out of the value menu and surfaced through `before:` / `after:` instead.
- Recency now defaults to the modified date, and the `before:` / `after:` hints name the configured date field.

### Sync & indexing
- Consent-gated reindex: a version-stale index warns and waits rather than silently rebuilding.
- A calm "syncing from another device" state when a newer index is arriving from a peer, distinct from the action-needed stale state.
- Mobile catch-up indexing is batched (O(N²) → ~O(N)) and stays stable under a large backlog.
- Peer-ahead state survives an app relaunch, and mobile no longer grinds during catch-up.

### Interface
- Search-modal keyboard, pointer, and link-handling polish; theme-proofed filter pills; the query field focuses on a dead-space click.
- The per-result score line is off by default.
