# Changelog

All notable changes to Seek are documented here. This project adheres to [Semantic Versioning](https://semver.org/).

## 1.1.1

Indexing and sync reliability release. No reindex is needed, since the index format is unchanged.

### Changed
- **Editing a note no longer rewrites a large sync file.** Cross-device sync previously merged every change into an active shard file, so a small edit near a full 4 MB shard read and rewrote the whole file, and services like iCloud re-uploaded all of it. Each change now lands in a small fresh file, and a background pass folds accumulated small files back into dense ones, so file counts stay low at rest.
- **Searching during a full rebuild pauses indexing instead of competing with it.** A full reindex now yields between files while a query is in flight and resumes where it left off, so searches stay responsive during an initial build without cancelling any indexing work.
- **Searches no longer wait for sync files to finish writing.** At the end of an indexing pass, the sync data was written while the index was still locked, so a search issued at that moment queued behind file IO. The write now happens after the index is released.
- **Running out of storage shows one clear notice.** If the device's storage quota fills mid-index, affected files are skipped with a single "storage full" notice instead of failing quietly on every file, and they are picked up automatically once space frees.
- **A file's index entry now commits in one transaction.** A file's chunks and its bookkeeping record used to be written separately, so an interruption at the wrong moment could leave a file half-indexed. That window is closed.
- Diagnostics now record why an incremental update fell back to a full pass, to guide future tuning.

## 1.1.0

The first feature release since launch! A big thanks to everyone on the reddit thread with feedback and suggestions! No reindex is needed, since the index format is unchanged.

### Added
- **Recent searches.** The last three searches now appear in the modal's resting state, under the query field. Only searches where you opened a result, or closed the modal while results were showing are shown, and this history is not synced across devices.
- **Insert a link to a result without leaving the modal.** Shift+Enter, or Shift+click, inserts a wikilink to the highlighted result at your cursor. The link mirrors what opening the result would do: when your query strongly matches the note's title it inserts `[[Note]]`, and otherwise it links the section the match was found in, `[[Note#Section]]`. Ported from [@adrianghnguyen](https://github.com/adrianghnguyen)'s fork, thank you!
- **A setting for where Cmd/Ctrl opens a result** (Display → "Open results with Cmd/Ctrl in"): a new tab (the previous behavior, still the default), a split, or a window (desktop only). A plain click or Enter still opens in the current tab.

### Changed
- **Improvements to Snippets**
    - Seek previously anchored the snippet on the earliest raw text match of any query word including stopwords, and without respecting word boundaries. So "bread not rising" would anchor on the "not" inside "cannot". A result matched on meaning rather than wording often fell back to showing the start of the section with nothing marked. Snippets are now chosen by scoring candidate sentences and returning the best-matching window, similar to Lucene's highlighter.
- **A strong title match now opens the note at the top**
    - Queries where all search terms are in the title of a note are treated more like a note look up, rather than a passage search. Queries carrying terms beyond the title still jump to the best matched section within that note.
- **Indexing is quieter.** A large sync could leave a live-updating progress notice on screen for the entire embedding run, which could be many minutes. Indexing now shows one notice when it starts and a summary when it finishes. Live progress still streams to the inline display in settings.
- **Embedding runs off the main thread wherever WebGPU isn't in use**
    - Using a webworker on iPhone and Android, desktop with "Force CPU", and desktop when WebGPU falls back. Which keeps the interface responsive while the index builds. Results are unchanged: same model, same vectors, same throughput, only a different thread. Any failure falls back to the previous behavior.
- **The footer hint bar drops hints it can't fit** instead of overflowing the modal on narrow windows.
- **Recency "High" more strongly favors recent notes.** High now uses a 90-day half-life, so an episodic vault queried by series name ("standup", "1x1", "session") surfaces the recent entries.
- **The relevance readout no longer reports a recency score while the recency bonus is Off.** With "Show scores" enabled, the recency figure was computed and displayed even when it was being multiplied by zero and contributing nothing to the ranking.
- **A search from the CLI or an `obsidian://seek` link no longer waits for indexing to finish.** These paths didn't signal that a query was in flight, so a search could queue behind an entire indexing pass. On a cold install, a first search could wait out the full initial build. They now interrupt indexing the way a search from the modal always has.
- **A file Seek could never finish re-reading no longer makes the app unresponsive every few minutes.** Index compaction re-ran its whole-vault pass on every poll when a file was persistently unreadable — an iCloud file whose contents were never downloaded, for example. The pass now yields as it works and stops retrying after a few attempts.
- On mobile, releasing the model while idle could interrupt index compaction mid-pass, manufacturing the incomplete-pass retries above.

## 1.0.6

Indexing reliability release, prompted by a community bug report — thank you.

### Fixed
- A note dominated by enormous runs of whitespace padding (the reported case: a machine-generated Markdown table with megabytes of space-padded cells) could crash Obsidian during indexing or fail with `Tensor shape is too large`, and the file was then retried on every indexing pass without ever completing. The real cost was tokenizing the padding — gigabytes of memory for text that never survives truncation. Seek now collapses long whitespace runs before any text is measured or embedded, so these files index normally. Stored note text, snippets, and keyword search are untouched, and no reindex is needed.

### Changed
- When a chunk fails to embed deterministically, Seek now keeps the rest of the file indexed and records the failing chunk instead of retrying the whole file forever. Recorded failures are retried once per release, so a shipped fix reaches them automatically.
- Many embedding failures in a single pass (for example a lost GPU device, or the app backgrounded mid-index) are treated as an environment problem rather than a content problem: nothing is recorded as failed and the files retry normally on the next pass.

### Internal
- Embedding-failure diagnostics now record batch size, compute backend, and input sizes (counts only — never note content), so future reports of this class self-diagnose.

## 1.0.5

CPU-compute reliability release, prompted by a community bug report — thank you.

### Fixed
- On desktop and Android, loading the embedding model on the CPU path failed with `Could not find an implementation for GatherBlockQuantized`, so building the index was impossible whenever WebGPU wasn't usable (for example with hardware acceleration turned off) and when "Force CPU" was selected. Seek now loads the ONNX runtime build that includes the CPU kernels the quantized model needs. iPhone and iPad were unaffected. As a side effect, the CPU runtime download is ~10 MB smaller.
- When both WebGPU and the CPU fallback failed, the error reported only the CPU failure and discarded the reason WebGPU fell back. Both causes are now reported.

### Changed
- The diagnostic report now names the GPU adapter and flags software (fallback) adapters — "GPU yes" alone couldn't distinguish a real GPU from software rendering.
- The report now includes the last model load: which compute backend and quantization actually served it, and the fallback reason if WebGPU didn't.
- The report's version stamp now reflects the installed plugin version (it previously always read "v0.0.1").

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
