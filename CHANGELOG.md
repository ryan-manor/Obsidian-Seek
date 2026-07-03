# Changelog

All notable changes to Seek are documented here. This project adheres to [Semantic Versioning](https://semver.org/).

## 1.0.4

Search accuracy and cross-device sync reliability release.

### Changed
- Lexical (keyword) search now indexes link targets, URLs, and aliased link text that were previously stripped before reaching it. Notes are now findable by the links and sources they reference, not just the surrounding prose.
- Inline `#tags` in note body text and the legacy `alias:` frontmatter key now reach tag search and the title index, matching what `tags:`/`aliases:` already did.
- List-valued frontmatter properties are now searchable as text, not just filterable.

### Fixed
- Date filters (`before:`/`after:`) no longer silently accept an out-of-range day or month (e.g. Feb 30) and normalize it into an unintended date.
- A timezone mismatch that could shift a date filter's boundary by a day is fixed.
- Typing with an IME (e.g. Japanese, Chinese, Korean input) while a filter pill was focused could interrupt composition; fixed.
- Filter pills no longer suggest notes excluded from the index.
- Numeric property filter pills now flag a value that fails to parse instead of matching silently.
- Several low-probability sync races fixed, including a stale index surviving a database upgrade, a duplicate device identity after cloning or restoring a vault, and orphaned index data not being reclaimed on some devices.

### Internal
- Hardened startup, log rotation, and index-compaction paths against concurrent-write races.
- Hardened the internal release pipeline that produces this build.

## 1.0.3

Settings refinement. Search, indexing, and sync are unchanged.

### Changed
- Reorganized the Index settings section: the index status and the reindex button stay in view, while the set-once options (index Base files, honor excluded folders, index location) now sit under an "Advanced settings" disclosure, matching the Relevance section.
- Clarified the reindex note: building the index re-embeds every note and isn't recommended on a phone. Run it on a computer and your phone syncs the finished index automatically.

## 1.0.2

Code-quality release addressing the second round of Obsidian community plugin-review feedback. No user-visible change — search, indexing, and sync behave identically to 1.0.1.

### Internal
- Replaced the remaining lint-rule suppressions with code that satisfies Obsidian's plugin guidelines directly (member access for the per-device backend/diagnostic storage and the hidden compute frame, popout-safe globals); no `eslint-disable` comments remain in shipped code.
- Switched the dev-only YAML test dependency to `yaml`.
- Added a local reproduction of the review's lint configuration so findings are caught before submission.

## 1.0.1

Compatibility and code-quality release addressing the Obsidian community plugin review. Search behavior is unchanged — the lexical/semantic ranking is byte-identical to 1.0.0.

### Fixed
- Startup crash on iOS before 16.4: a regex feature unsupported by older WebKit prevented the plugin from loading at all on those devices.
- Popout-window support: timers and DOM access now resolve against the correct window, and the hidden background compute frame and app-visibility tracking are anchored so they survive a popout window opening or closing.
- iPad and Android tablets are now classified correctly for compute-backend selection.

### Changed
- The search command id changed from `seek-search` to `search` (Obsidian namespaces it as `seek:search`). If you bound a custom hotkey to it, rebind it once.

### Internal
- Addressed the Obsidian community plugin-review findings: Platform API for device detection, popout-safe timers/DOM, vault-scoped storage where appropriate, typed worker/iframe messages, and removed dead code. No user-visible search changes.

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
