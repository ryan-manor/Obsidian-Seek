# Changelog

All notable, user-facing changes to Seek are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and Seek aims to follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> Maintainer note: this file is **hand-curated in user-facing language**, not
> generated. The public repo's git history is promotion snapshots, and the real
> feature history lives in the private development repo (which may contain private
> data), so neither can feed this file directly. When promoting, `promote.mjs`
> prints a draft from the development commits since the last promotion; curate it
> into the section below — plain language, no internal module names, PII-scrubbed.
> The matching `## [x.y.z]` section becomes the GitHub Release body.

## [Unreleased]

_Changes promoted but not yet released. Move these under a new version heading
when cutting the release (`npm version <patch|minor|major>`)._

## [0.0.1]

### Added

- On-device hybrid search: dense semantic embeddings fused with lexical BM25.
  Note content and queries never leave the machine.
- Query filters: tag, path, property, date, and term-exclusion operators with
  ghost-text completion.
- Consent-gated reindexing with a banner when the index is stale relative to the
  installed version.
- Cross-device index persistence via a per-device sidecar so the index survives
  storage eviction and syncs between devices.
