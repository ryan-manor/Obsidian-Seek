// Core types for Seek. Mirrors the verbose-logging schema from the iOS spike
// so the same generator/reader code can be reused with minimal changes.

export type Device = 'webgpu' | 'wasm';
export type RequestedDevice = Device | 'auto';
export type Dtype = 'q4f16' | 'q4' | 'q8' | 'fp32';

// Bumped whenever any LogEntry shape changes incompatibly. The report
// renderer keys off this to detect old logs and skip fields that aren't
// present. Append-only NDJSON files outlive plugin versions, so the schema
// version is the only thing keeping the report parser honest.
// v6: added EmbedProfileEntry (runtime wall-time decomposition harness —
// tokenize / forward / readback split, to settle the I/O-binding,
// worker, and WASM-tokenizer questions on the v4 runtime).
// v7: SearchEntry gained two-stage telemetry — binary candidate-gen + fp32
// exact rerank. New fields: binaryMs, selectFetchMs, candidateUnionSize,
// per-arm contribution counts (binaryCount/bm25Count/recencyCount), the
// stage-1 caps (binaryTopN/bm25TopM/recencyTopK). Existing min-max hybrid
// fields are unchanged — the rerank runs the same scorer over a subset.
// v8: SearchEntry gained inline-filter telemetry — `cleanedQuery` (residual
// text after stripping operators) and `filters` (structured QueryFilters or
// null). Additive/optional in practice; v7 logs still parse (the new fields
// are simply absent and read as undefined by the report renderer).
// v9: every entry is stamped (in logger.append) with `deviceId` (per-install,
// localStorage-backed) and `sessionId` (per plugin load). This is what lets the
// report attribute Platform/Init/Loads to the device that actually generated
// them — previously a shared iCloud-synced log made `at(-1)` cross-device.
// Each device also writes its OWN log file (seek-log-<deviceId>.ndjson) so
// concurrent appends from desktop + phone can't clobber each other under
// iCloud's whole-file last-writer-wins sync. Both fields optional → v8 logs
// (and the legacy seek-log.ndjson) still parse, attributed to deviceId 'legacy'.
// v10: crash forensics. Synchronous localStorage breadcrumbs survive process
// death (async NDJSON appends lose that race — proven by the 2026-06-11 iPhone
// reindex jetsam kill that left zero log entries); on boot, an unclosed prior
// session is promoted into the log as a crash-detected entry.
// v11: InitEntry.initMs — real iframe-build wall time. Was never measured
// (the startup-deferral tradeoff rested on an estimate). Additive/forward-only:
// older logs simply lack the field; no migration. initMs=0 means the idempotent
// init early-returned (iframe already live), not a zero-cost build.
// v12: ModelDeliveryEntry — per-load record of the active model spec, storage
// persistence, and stale-model cache eviction (production model-delivery layer).
// v13: PlatformEntry.gpuIsFallbackAdapter — distinguishes a real GPU from a
// software (SwiftShader-class) fallback adapter. Additive/forward-only. The
// r/ObsidianMD triage showed "GPU yes" is ambiguous without it: requestAdapter
// can hand back a software adapter when hardware acceleration is off, which
// then fails ORT's WebGPU init while the report still reads as GPU-capable.
// Also LoadEntry.glue — which ort-wasm glue variant actually loaded
// (previously only recoverable from the `checks` strings).
export const LOG_SCHEMA_VERSION = 13;

// ---- chunk model ----

export interface ChunkMetadata {
    tags: string[];
    aliases: string[];
    created: string | null;
    modified: string | null;
    // All frontmatter values, keyed by frontmatter key — the generic backing
    // store for `[key:value]` inline filters (context, status, pageType, …) AND
    // the searchable-properties BM25 field. Scalars are string-coerced; LIST
    // values (relatedPages-style) are kept as string[] (v10, audit R2 #3: the
    // suggester offered list-prop pills the scalar-only store could never
    // match). The matcher treats a list as any-element-matches (Obsidian's own
    // list-property semantics); the BM25 properties field folds each list item
    // in too, per-element, through the same type-drop gates as a scalar value
    // (extractPropertiesText, audit R2 batch2 #3).
    // tags/aliases are NOT duplicated here (dedicated fields + operators).
    // Populated on (re)index.
    properties: Record<string, string | string[]>;
}

// Structured filters extracted from a raw query string by parseQuery()
// (src/query-parser.ts). A null QueryFilters means the query had no inline
// operators — a plain semantic/lexical search. Mirrors the predecessor's
// Python SearchFilters; `tagsMatchAll` is reserved for a later version
// (always OR in v1).
export interface QueryFilters {
    tags: string[] | null;
    tagsMatchAll: boolean;
    frontmatter: Record<string, string> | null;
    includePaths: string[] | null;
    // Numeric comparison filters (`[price>50]` / `[price<200]` / `[price=160]`),
    // value already coerced to a finite number at parse time. Keyed off a
    // property's DECLARED type (Number), not its name — see FilterContext below
    // and [[Seek Typed-Value Filters Design]]. All operators are value-INCLUSIVE:
    // `>`/`<` keep on/above and on/below the bound, `=` is exact. Null = none.
    numeric: Array<{ key: string; op: '<' | '>' | '='; value: number }> | null;
    // Date-range filters (`after:D` / `before:D`), both day-inclusive. These read
    // the SINGLE date field the user selected for Recency (resolved per-chunk via
    // FilterContext.dateField), so they only bind when Recency is ON. Raw date
    // strings (parseDateMs-able); the inclusive bounds are computed in the matcher.
    dateAfter: string | null;
    dateBefore: string | null;
    // Keys where a comparison operator was used on a property that is NOT declared
    // Number (e.g. `[pageType<notes]`). Such a clause CANNOT be honored numerically,
    // so the whole query is unsatisfiable → the matcher returns 0 results rather
    // than silently substring-matching (Decision D3). Carried for diagnostics; the
    // search UI separately flags the offending pill red. Null = no mismatch.
    numericTypeMismatch: string[] | null;
    // Bare-word negation (`-term`, Obsidian `-` semantics). Normalized lowercase
    // tokens; a note is excluded if ANY of these appears in its title/content.
    // Unlike the metadata filters above, this is applied note-level in search()
    // (compileMatcher is metadata-only and can't see content), not in the matcher.
    exclude: string[] | null;
}

// Vault-specific facts the PURE parser/matcher (query-parser.ts) can't read for
// themselves — they import only types + a date helper, never `obsidian`. The call
// site (search.ts, which has `app` + `settings`) resolves this once and threads it
// through parseQuery()/compileMatcher(). Optional/defaulted everywhere so existing
// tests and ad-hoc callers keep working (no ctx ⇒ permissive: any key may compare,
// dates use `created`). See [[Seek Typed-Value Filters Design]] §Architecture.
export interface FilterContext {
    // The date field `before:`/`after:` target — the user's Recency selection.
    // null when Recency is OFF, which is how the parser knows to leave a typed
    // `before:`/`after:` as plain search text instead of binding a date filter.
    dateField: { key: RecencyKeyChoice; createdProp: string } | null;
    // Properties declared Number in Obsidian's type registry. A comparison on a
    // key outside this set is a type mismatch (Decision D3); a key inside it is a
    // numeric filter (and reaches the `[` autocomplete past the cardinality gate).
    numericKeys: Set<string>;
}

export interface Chunk {
    chunk_id: string;
    title: string;            // hierarchical: "Note Title > H1 > H2"
    content: string;
    note_path: string;
    heading_path: string[];
    metadata: ChunkMetadata;
    // 1-based line numbers into the RAW note file, FRONTMATTER INCLUDED — usable
    // directly against the on-disk text (editor.setCursor / scrollIntoView and a
    // raw-content offset walk for the in-note highlight; see search-modal.ts). The
    // chunker counts these against the frontmatter-stripped body and shifts them
    // back to file coordinates before emitting (chunker.ts return site).
    start_line: number;
    end_line: number;
    // Lexical-only chunk: present (true) ONLY on the title-only fallback emitted
    // for a body-LESS note (chunker.ts) — a note whose entire embed string would
    // be "<title>\n\n" with no body. Such a chunk has no semantic content to
    // offer the dense channel; its vector is just the title (often a bare date),
    // which makes it a universal near-neighbor for any content-free / OOV query
    // (e.g. an opaque ID). The ranker floors its dense score before min-max so it
    // can never be a spurious semantic hit, while BM25's 3.0x title boost keeps
    // it findable by name. Absent (undefined) for every normal chunk and for
    // short-but-non-empty fallbacks (which DO carry embeddable content).
    lexicalOnly?: boolean;
    // Human-facing title for an oversized section that was hard-split into
    // multiple parts: `"<title> (part N)"`. Present ONLY on split parts; absent
    // for every single-part chunk. The `(part N)` marker must NOT live in
    // `title`, because `title` is what gets embedded (search.ts: `title\n\ncontent`),
    // indexed in the 3.0x-boosted BM25 title field, and hashed into chunk_id —
    // a per-part suffix there poisons the dense vector and the title boost and
    // splinters otherwise-identical titles. Display surfaces read `displayTitle`
    // (falling back to `title`) so the part marker stays visible without leaking
    // into the index. Optional → pre-existing chunks read it as undefined.
    displayTitle?: string;
    // Note-level frontmatter VALUES folded into the DENSE channel only. Appended
    // to the EMBED input by token-budget.ts embedInput (`title\n\ncontent\n\n
    // denseSuffix`) and to NOTHING else — bm25.ts's `content` field stays the raw
    // body, the validated dense-only decoupling (+0.0031 dense-only vs +0.0010
    // both-channels; tags already have their own BM25 field, so injecting there
    // just dilutes). Built once per note (chunker.ts buildDenseSuffix: keys
    // dropped, wikilinks→basename, aliases excluded, date/number/boolean values
    // type-dropped) and carried on EVERY chunk of the note — the same convention
    // by which aliases are hoisted into every chunk's title. Folded into chunk_id
    // (chunkIdFor's 4th arg) so the hashed bytes still equal the embedded bytes;
    // a CHUNKER_VERSION bump rides with it. Absent (undefined) when the note has
    // no qualifying values. See [[Seek Domain Agnostic LoRA]] (locked 2026-06-18).
    denseSuffix?: string;
    // Lexical reclamation (v10, 2026-07-02): the raw substrings cleanDenseBody
    // DROPPED from this chunk's section — wikilink targets behind aliases,
    // markdown-link/autolink URLs, bare-URL scheme+TLD forms (dense-clean.ts
    // extractLinkTerms). Folded into the BM25 `content` field at doc-build time
    // (bm25.ts buildDoc) and into NOTHING else: not the embed input, not the
    // snippet body, and NOT chunk_id — ids and vectors are byte-identical with
    // or without it (the CHUNKER_VERSION 10 bump exists only to backfill this
    // field into stored records fleet-wide). Restores the pre-v8 query symmetry
    // where `theverge.com` lexically matched a clipping that links its source.
    // Absent when the section dropped nothing. Split parts share the section's
    // whole value (token-budget.ts `...chunk` spread) — a minor tf inflation on
    // oversized sections, accepted for wiring simplicity.
    link_terms?: string;
}

// One synthetic search document extracted from a `.base` file (Obsidian Bases).
// A base is modelled as a document whose VIEWS are its sections: `extractBaseDocs`
// returns one BaseView per indexable view plus a base-level entry, and the chunker
// (`chunkBase`) turns each into a Chunk that reuses the whole note pipeline (title,
// heading_path, chunk_id, dense + BM25, dedup, nav). See base-extractor.ts.
export interface BaseView {
    // The view's display name, or null for the base-level entry. Drives the chunk
    // title (`"<base> > <viewName>"` vs bare `"<base>"`) and heading_path, so the
    // view name earns the 3.0x BM25 headings field and the dense channel.
    viewName: string | null;
    // Synthetic searchable text: base name + inherited top-level filter literals +
    // this view's own filter literals + the view name, deduped. Never empty (the
    // base name is always present), so no chunk needs the lexicalOnly stub flag.
    content: string;
}

// A chunk's metadata — everything except the body text. This is what the v8
// `chunk_meta` IDB store holds and what the resident search frame keeps in RAM;
// the body lives in `chunk_body` keyed by chunk_id and is fetched lazily for
// the ≤topK results (snippets / hydration), `-term` negation, and BM25 (re)fit.
// See docs/seek-scaling.md §B1. Omit (not a hand-listed interface) so it tracks
// Chunk automatically as fields are added.
export type ChunkMeta = Omit<Chunk, 'content'>;

// User-tunable settings (persisted via Plugin.saveData). Read live by the
// orchestrator (it holds the same object ref), so changes take effect on the
// next search without a rebuild.
export interface SeekSettings {
    // Dense weight in the TM2C2 hybrid: hybrid = w·dense_norm + (1−w)·bm25_norm
    // (ranker.ts hybridFusion over TM2C2-normalized channels). w=1 is pure
    // semantic, w=0 pure lexical. Default 0.80 — the BOUND-NORMALIZED scale
    // (2026-06-09: BM25 is divided by its per-query theoretical bound, not its
    // empirical max — fusion.ts theoreticalNormBm25). NOT comparable to either
    // earlier scale (min-max ~0.5–0.7; empirical-max TM2C2 0.90–0.95): the bound
    // compresses the lexical channel, so the optimum sits lower. 0.70–0.85 is a
    // flat plateau on BOTH the 482-q personal eval and the D&D OOV slice — the
    // first scale where one global weight is optimal across domains, which is
    // the point of the bound (query-invariant channel scale). The empirical-max
    // era's "saturated single-term hit" captures are handled by the norm now
    // (a weak best match no longer gets a forced 1.0), not by this weight.
    // Single biggest relevance lever; applied per-search — no reindex. NOTE:
    // a data.json persisted before the bound switch carries an old-scale value
    // (0.92) — the settingsRev<2 migration in main.ts onload drops it so the
    // bound-norm default takes over (no manual data.json surgery needed).
    denseWeight: number;

    // Known-item boost: additive title/alias COVERAGE boost (fusion.ts
    // titleMatchBoost). Fires when every query token is in the title and scales
    // by precision (|q∩t|/|t|). The 2026-06-02 study found the dense gap is
    // largely RANKING (the entity page buried under pages that mention it);
    // coverage generalizes the old exact-match boost to the common subset case
    // ("alex 1x1" vs `Alex 1x1 2026-05-19`). The swept knee is 0.8 (knee on the
    // 482-q old-log eval, nDCG@10 0.8143→0.8677); the SHIPPED default softened to
    // 0.5 in the 2026-06-19 settings ratification (the Title-bonus "Default" stage),
    // with the 0.8 knee one click away as "High" (segmented Off=0/Default=0.5/High=0.8).
    // NOT comparable to the old exact-only 0.025; precision-scaling makes high values safe.
    navTitleBoost: number;

    // Which per-chunk date "recent" means, vault-wide — the GLOBAL definition
    // behind the recency ε-tiebreaker (ranker.ts), the S1 recency candidate arm,
    // the filter-only browse sort, AND the before:/after: date filter (all read
    // it through fusion.ts recencyDate, so they cannot disagree). 'modified'
    // (DEFAULT) = file mtime as of indexing: the ONLY date every note is
    // guaranteed to carry, so it's the only generic default we can bank on
    // across vaults — at the cost of being edit-recency, silently churned by
    // vault copies, iCloud sync, and bulk plugin edits. 'created' = a frontmatter
    // date property (createdProp, with filename-date then mtime fallback) —
    // copy-proof and matches the dated-instance semantics episodic queries want,
    // but it assumes the vault carries such a property, so it's an explicit
    // opt-in via the date-field picker, not the default. Applied per-search (no
    // reindex, both dates are already in the index). [[Seek Rel]] §Recency Plan
    // 2026-06-11. (Replaces `recencyWeight`, the parked blend weight deleted
    // 2026-06-11 with the rest of the zombie recency machinery; a stale persisted
    // value is inert.)
    recencyKey: RecencyKeyChoice;

    // Which frontmatter property holds a note's creation date. `created` is
    // THIS vault's convention, not an Obsidian builtin — other vaults use
    // `date created` (Linter), `dateCreated`, `created_at`, … so the name is a
    // setting. In the UI it is a PICKER over the vault's Date / Date & time
    // typed properties (Obsidian's property-type registry, .obsidian/types.json
    // via main.ts enumerateDatePropertyNames) — never free text, so a typo'd
    // ("craated") or non-date (tags…) choice is unselectable by construction.
    // Resolved READ-side: all scalar frontmatter is already indexed per chunk
    // (metadata.properties), so changing this needs no reindex. Notes without
    // the property fall back to a YYYY-MM-DD in the filename (daily notes /
    // dated series notes), then mtime — fusion.ts recencyDate ladder.
    createdProp: string;

    // Recency WEIGHT (ε): the magnitude of the additive recency term in
    // final = hybrid + ε·recency + titleBoost (ranker.ts). recency ∈ [0,1] from
    // the half-life decay below, so ε is literally the maximum score a
    // brand-new note can gain. The shipped DEFAULT is now 0 — recency ships Off
    // (2026-06-19 settings ratification). The Recency segmented control maps
    // Off=0 / Default=0.04·180d / High=0.1·270d; the old 0.02 always-on tiebreaker
    // is retired. A non-zero ε ("Default"/"High") is a deliberate recency lean: raising it past
    // ~0.1 lets a fresh note leapfrog a moderately-better older one. CAUTION,
    // not a free lever: the 06-04 click study found 50% of episodic clicks
    // target a note >90d old, and a recency term strong enough to reorder real
    // gaps scored WORSE (MRR 0.242 vs 0.310) — so high values trade known-item
    // precision for recency. Additive on top of uncontested relevance; applied
    // per-search (no reindex). Default 0 (Off).
    recencyEpsilon: number;

    // Recency HALF-LIFE in days: the width of the decay 0.5^(daysOld/halfLife)
    // (fusion.ts computeRecencyScore) — the age at which a note's recency signal
    // halves. This is the time CONSTANT, orthogonal to the weight above: it
    // reshapes the curve, it does not change how hard recency hits. Shipped
    // DEFAULT 180 (the 06-04 operating point — wide enough that an 83-day-old
    // median episodic target still carries ~0.73). A SHORT half-life (7–30) is
    // the other half of a "high recency" mode: it concentrates the boost on the
    // last days ("what was I just working on"); a long one spreads it gently.
    // Undated notes score 0 (neutral — never penalized). Applied per-search
    // (no reindex). Default 180.
    recencyHalfLifeDays: number;

    // (BM25 per-field boosts removed from settings 2026-06-08 — they're now
    // eval-tuned constants in bm25.ts DEFAULT_FIELD_BOOSTS. Once the coverage
    // navigational boost is in, their marginal leverage is ~+0.004 nDCG@10:
    // not worth a user knob. See [[Seek Rel]].)

    // Fuzzy lexical matching toggle. ON by default (2026-06-09): MiniSearch
    // fuzzy with an absolute max edit distance of 1 (one insertion/deletion/
    // substitution). The D&D typo eval (typo_fuzzy_eval.py) priced both sides:
    // +3–4/40 gold@1 on misspelled entity queries (typo rescue doesn't exist at
    // all without it — every norm scores 34/40 with fuzzy off), at an ns ~0.002
    // nDCG cost on clean personal queries. Interacts with the theoretical BM25
    // bound: derived terms can exceed it (fusion clips) and a fully-OOV typo
    // query has bound 0 (fusion falls back to /max) — see fusion.ts. Off =
    // exact terms only. Applied per-search (no reindex).
    fuzzyEnabled: boolean;

    // Prefix matching on the LAST query token (≥3 chars). ON by default
    // (2026-06-10): the final token of a query is the one plausibly still
    // being typed ("amster", "roadmap for concep", "lr ac rearch"), and
    // neither exact, fuzzy (edit-1), nor any scorer change can reach
    // "rearchitecture" from "rearch" — only expansion can. Eval
    // (~/seek-eval-pack prefix_arm.py, α=0.80): personal 483q
    // 0.8666→0.8730 bin nDCG@10, gold@1 +1.1pt with the wins concentrated
    // on truncated/typeahead queries; D&D and code stress sets exactly
    // unchanged; capture cap-004 target rank 5→2. Expanding ALL tokens was
    // rejected (D&D desc −0.0149, ~1813 derived terms/q on code). Derived
    // terms are numerator-only vs the theoretical bound (fusion clips),
    // same contract as fuzzy. Applied per-search (no reindex).
    prefixLastToken: boolean;

    // EXPERIMENTAL, default OFF: query-side synonym expansion from the
    // vault's own frontmatter aliases (synonyms.ts). Each note's single-token
    // name + aliases form an equivalence class ("Lr" ↔ "Lightroom"); a query
    // term in a class also queries its classmates at a discount, so "lr
    // roadmap" can reach a note that only ever spells out "Lightroom".
    // Guards: ambiguous tokens (shared by >1 class) neither trigger NOR get
    // injected (symmetric — 4 pages aliased "rohit" disable that bridge
    // entirely), and tokens matching >5% of chunks are refused (junk-alias
    // ceiling). Eval (~/seek-eval-pack, 2026-06-10): personal +0.0015 bin
    // nDCG@10 over the prefix baseline at w=0.8. The native-attribution gate
    // showed MiniSearch's raw ×quality double-credit ERASES that gain at
    // every weight (it doubles alias hub/sibling pages), so bm25.ts rescales
    // each result back to source-attribution semantics exactly — +0.0015 is
    // what ships. ON by default as of the 2026-06-19 settings ratification
    // (the always-on dogfood now IS the live smoke test; plan note "Seek
    // Synonym Expansion Plan"). Known
    // precision tax even under source semantics: the alias OWNER page
    // also climbs on expanded queries (cap-004 owner rank 5→3). English-only
    // posture: dictionary tokens run through the English analyzer. Dictionary
    // rebuilds with the BM25 cache (per dataGeneration); no reindex.
    synonymExpansion: boolean;

    // Searchable properties — index frontmatter property VALUES as a 6th BM25
    // field (boost 2.0, bm25.ts) so plain query words match structured
    // metadata: "san francisco restaurants" finds notes whose body never says
    // either but whose placeLoc/placeType do. Values are wikilink-unwrapped
    // and date/number values dropped (extractPropertiesText); machinery keys
    // (icon, cssclasses, dates, …) are excluded, list-valued props fold in
    // per-item (audit R2 batch2 #3); terms then run the standard analyzer.
    // Harness gate 2026-06-11
    // (props_field_arm): captures +0.059 at boost 2 (clean-SF 0.58→0.77,
    // "swiss hotel" crater 0.0→0.19), personal 483-q net-zero — churn is
    // symmetric on person-name/link-valued props, which is why the boost
    // stays modest. Toggle refits the BM25 cache on the next search (index
    // shape change) but needs NO embedding reindex — chunks already store
    // metadata.properties, the same backing store [key:value] filters read.
    searchableProperties: boolean;

    // Headings field — index the section heading path (chunk heading_path) as
    // a BM25 field (boost 3.0, bm25.ts). The lexical mirror of what the dense
    // channel already reads via embedInput's hierarchical title; without it
    // heading words are BM25-invisible (extractNoteName strips the path from
    // `title`, the chunker drops the heading line from section content).
    // Harness gate 2026-06-12 (headings_field_arm) was a WASH with recency
    // OFF: heading-only topical wins (+0.5 nDCG) cancelled by dated-series
    // near-tie reorders — the class the live ε-recency tiebreaker re-orders,
    // which the harness convention can't model. ON by default as of the
    // 2026-06-19 settings ratification (the live A/B is now the shipped default).
    // Same cache contract as searchableProperties: refit on toggle,
    // NO embedding reindex.
    headingsField: boolean;

    // "Boosted BM25" preset — one switch that overrides three field boosts to
    // the values explored vs Omnisearch: aliases 6→9 (Omnisearch weights an
    // alias like the title; we close most of that gap), tags 3→2 (trim noisy-
    // tag vaults), headings →4. aliases/tags are score-time (no reindex);
    // headings is index-shape, so flipping this ALSO forces the heading field
    // on in ensureBm25() — otherwise the headings boost is inert (no postings).
    // Refit-on-toggle, no embedding reindex; same contract as headingsField.
    boostedBm25: boolean;

    // BM25 coverage weighting — the SOFT-AND fix for OR's looseness. MiniSearch
    // combines query terms with OR, so a doc that saturates ONE query term (a
    // short hub/entity page whose title IS that term — the "Switzerland" country
    // page on "switzerland hotel", a person page on "alex …") can out-score a doc
    // that matches the WHOLE query. On (default) we scale each doc's raw BM25 by
    // the fraction of DISTINCT query terms it matched (|matched|/|query terms|)
    // BEFORE TM2C2 normalization, so a 1-of-2 match keeps half its lexical weight
    // and a 2-of-2 match keeps all of it. Only bites multi-term queries (single
    // term ⇒ factor 1, a no-op). Unlike hard AND it never zeroes a partial match,
    // so recall is intact (hard AND wiped ALL lexical signal for 19% of relevant
    // notes on the 482-q eval and LOST nDCG; coverage was +0.005, monotone ≥ OR
    // at every alpha — ~/seek-personal-eval/and_coverage_eval.py, 2026-06-09).
    // Applied per-search (no reindex).
    bm25Coverage: boolean;

    // (blendMode + rrfK deleted 2026-06-11: the opt-in RRF fusion is gone —
    // linear TM2C2 is the only blend. Stale persisted keys are inert.)

    // Whether incremental indexing + full reindex honor Obsidian's "Excluded
    // files" setting (metadataCache.isUserIgnored). On (default) a note in an
    // ignored folder — e.g. Archive — is treated as out-of-index: moving a note
    // INTO it is a soft-delete (its chunks are dropped), moving OUT re-indexes it.
    // Off indexes ignored folders too. Independent of EXCLUDED_PREFIXES, which
    // always excludes Seek's own machine output regardless of this flag. Applied
    // at index time (a change takes effect on the next reindex/delta, not retro-
    // actively).
    honorIgnoredFolders: boolean;

    // Whether `.base` files (Obsidian Bases — saved query/view definitions) are
    // indexed alongside markdown notes. ON (default) collects every `.base` file
    // and feeds it through extractBaseDocs → chunkBase → a base-level chunk plus
    // one per non-generic view, so a Base (and the right VIEW) surfaces in search
    // like any note section. OFF restricts the index to `.md`
    // only. Applied at index time: collection (search.ts indexableFiles) and the
    // create/rename/delete watcher (main.ts isIndexableFile) both gate on it, so
    // a change takes effect on the next reindex/delta — toggling OFF drops
    // already-indexed Bases on the next sweep, not retroactively.
    indexBases: boolean;

    // Per-result score line in the search modal. ON shows each result's
    // "Matching %" (the calibrated match strength) plus its recency and
    // title-boost bonuses. Requires a CALIBRATED corpus: the line is hidden — and
    // the "Display scores" settings toggle disabled — until the vault has at least
    // MATCH_STRENGTH_MIN_NOTES notes AND a full-corpus pass has produced dense
    // background stats (without which match strength is null). Pure presentation;
    // applies to the next time the search modal opens.
    showScores: boolean;

    // Diagnostic-only knob (no settings UI; set via data.json or an `obsidian eval`
    // settings inject). When false — the default — each search row persists only the
    // top-10 ranking trace the report actually renders; when true it keeps the full
    // 50-deep tail for offline pandas/eval. Bounds the size of the append-only log.
    verboseTrace: boolean;

    // Search-modal footer affordance. ON (default) shows the keyboard-hint bar
    // along the bottom of the modal (↑↓ navigate · ↵ open · ⌘↵ new tab · tab
    // fill autosuggest · esc close). OFF removes the whole footer for a minimal
    // "full results only" modal — just the query field and the result list.
    // Pure presentation; applies to the next time the search modal opens.
    showHotkeyHints: boolean;

    // NOTE: the compute backend (WebGPU vs WASM) is deliberately NOT a setting.
    // It is a property of the DEVICE, not the vault, and data.json syncs across
    // devices (iCloud / Obsidian Sync) — a toggle here would be shared, so the
    // iPad's WebGPU choice would leak onto the iPhone on the next sync. The
    // choice lives in per-device localStorage instead (origin-scoped, never
    // synced); see platform.ts resolveDevice / getBackendOverride. The old
    // `experimentalMobileWebgpu` boolean was removed 2026-06-12 for this reason
    // — a stale value persisted in data.json is now an ignored extra key.

    // Persist the vector index to vault files (`<pluginDir>/index/`) so it
    // survives iOS IndexedDB eviction and flows between devices via iCloud /
    // Obsidian Sync, which carry vault files but never a WebView's IDB. Desktop
    // writes a per-device sidecar that an evicted/fresh mobile device hydrates
    // WITHOUT re-embedding (re-chunk locally + copy the saved vectors). ON by
    // default as of the 2026-06-19 settings ratification (only Index location stays
    // user-facing; the sidecar seeds on the next reindex). It writes ~MBs of synced
    // index files. See sidecar.ts / sidecar-sync.ts. Obsidian Sync users must
    // also enable "Installed plugins / sync plugin files"; iCloud carries it free.
    sidecarEnabled: boolean;

    // Where the sidecar index folder lives. 'config' (default) = the LITERAL
    // '.obsidian/plugins/seek/index' — hardcoded to the default config-folder
    // name, NOT the device's active Override Config Folder (vault.configDir).
    // The CRITICAL config-folder bug was that the path resolved against the
    // active override, which is per-device and never synced: a split-config
    // setup (desktop '.obsidian' + phone '.obsidian-mobile') made producer and
    // consumer read different paths → silent zero results. The literal path is
    // identical on every device, so iCloud/Syncthing/Dropbox carry it even
    // under split config, and Obsidian Sync's "sync plugin files" carries it
    // under uniform config. 'visible' = a vault-root 'Seek Index/' folder — the
    // one location that survives Obsidian Sync + a *renamed* config folder
    // (the renamed-config device never receives '.obsidian/' over Sync), at the
    // cost of showing in the file-explorer pane. See the Sidecar Integration
    // Plan §config-folder CRITICAL. Per-device-relevant but kept in synced
    // data.json so the choice is explicit; the steer-notice only fires on the
    // device whose config is actually renamed.
    sidecarIndexLocation: SidecarIndexLocation;

    // Settings-schema revision, persisted in data.json so onload can run
    // one-time migrations. Rev 2 = the 2026-06-09 bound-norm switch: persisted
    // pre-bound denseWeight values (0.90/0.92, empirical-max scale) are on a
    // DIFFERENT scale than the bound-norm default (0.80) and must not carry
    // over — see the migration in main.ts onload. Rev 3 = sidecarEnabled added
    // (defaults to false on existing installs; no behavior change on upgrade).
    // Rev 4 = sidecarIndexLocation added + sidecar path pinned to the literal
    // '.obsidian'; the migration moves any index sitting under a non-'.obsidian'
    // active-override dir into the literal path (see main.ts onload).
    // A data.json without the key is treated as rev 1 (pre-bound).
    settingsRev: number;

    // Debug-only model override (testing arbitrary HF repos before promoting one
    // into model-registry.ts). Both optional + absent by default, so no migration
    // / settingsRev bump is needed (Object.assign backfills them as undefined).
    // When modelRepoOverride is set, activeModelSpec() loads that repo instead of
    // the shipped default; the override repo becomes the index drift-identity, so
    // switching it routes to a full reindex exactly like a real model swap.
    modelRepoOverride?: string;
    modelRevisionOverride?: string;
}

// Vault-global definition of "recent" — see SeekSettings.recencyKey above and
// fusion.ts recencyDate (the single accessor all recency consumers read through).
export type RecencyKeyChoice = 'created' | 'modified';

// Sidecar index folder placement — see SeekSettings.sidecarIndexLocation.
// 'config'  = hidden literal '.obsidian/plugins/seek/index' (default; covers
//             iCloud/Syncthing at any config naming + Obsidian Sync uniform config)
// 'visible' = vault-root 'Seek Index/' (the Obsidian-Sync-renamed-config carve-out)
export type SidecarIndexLocation = 'config' | 'visible';

export const DEFAULT_SETTINGS: SeekSettings = {
    denseWeight: 0.85,         // BOUND-NORM scale dense weight; mirrors DEFAULT_RANKING_CONFIG.alpha. Raised 0.80→0.85 (2026-06-27 re-eval): de-franken made BM25 more assertive, so a fixed α=0.80 over-weighted lexical; 0.85 is a cross-corpus win (Example Vault flat-to-+, BEIR +0.01–0.02). Migrated via rev 8. (NOT the 0.92 empirical-max point)
    navTitleBoost: 0.5,        // Title-bonus "Default" stage (segmented 0=Off / 0.5=Default / 0.8=High); softened from the 0.8 swept knee per the 2026-06-19 settings ratification — see field comment
    recencyKey: 'modified',    // global definition of "recent" (ε-tiebreaker + recency arm + browse sort + before:/after:); mtime is the only universally-present date → the generic default; 'created' (a frontmatter date prop, see createdProp) is an opt-in for true creation-recency
    createdProp: 'created',    // frontmatter property holding the creation date (vault convention; falls back to filename date, then mtime)
    recencyEpsilon: 0,         // ships Off (Recency segmented Off=ε0 / Default=ε0.04·180d / High=ε0.1·270d); was a 0.02 tiebreaker pre-2026-06-19 ratification — additive ε in final = hybrid + ε·recency + titleBoost (see field comment)
    recencyHalfLifeDays: 180,  // recency decay HALF-LIFE in days (0.5^(daysOld/HL)); 180 = 06-04 operating point, shorten (7–30) to concentrate on the last days
    fuzzyEnabled: true,        // typo tolerance ON by default (edit dist scales by term length, ≤3 exact; see bm25.ts FUZZY_BY_LENGTH); +3–4/40 gold@1 on typo'd entity queries, ns cost on clean
    prefixLastToken: true,     // last-token prefix expansion ON by default; +0.0064 personal nDCG, stress sets clean; see field comment
    synonymExpansion: true,    // ON (hidden) per the 2026-06-19 ratification; alias-dictionary query expansion (Lr↔Lightroom); BM25-dict refit, no reindex — see field comment
    searchableProperties: true, // frontmatter values as a BM25 field; ON as of 2026-06-25 — My-Vault channel eval measured +0.05 nDCG@10 (place-note recall: austin 22→3, zurich 33→7; combo_eval). Migrated on via rev 7; see field comment
    headingsField: true,       // ON (hidden) per the 2026-06-19 ratification; heading path as a BM25 field; BM25 refit, no reindex — see field comment
    boostedBm25: false,        // "Boosted BM25" preset (aliases 9 / tags 2 / headings 4); OFF — opt-in field-weight lever, implies heading indexing; see field comment
    bm25Coverage: true,        // soft-AND: scale BM25 by matched-query-term fraction (multi-term only); see field comment
    honorIgnoredFolders: true, // Archive et al. are soft-deletes by default
    indexBases: true,          // ON: index .base files (Obsidian Bases) as synthetic docs; preserves the feature's unconditional pre-toggle behavior
    showScores: false,         // OFF by default: per-result score line (Matching % · recency · title); opt-in via Display settings. (Also auto-hidden until the corpus is calibrated — ≥200 notes + full pass.) Default-only flip, no migration: installs that already persisted showScores keep their choice.
    verboseTrace: false,       // OFF: persist only the top-10 ranking trace per search (what the report shows); ON = full 50-deep tail for offline eval. Diagnostic-only, no UI
    showHotkeyHints: true,     // ON: show the modal footer keyboard-hint bar + result counter; OFF = full-results-only modal
    sidecarEnabled: true,      // ON (hidden) per the 2026-06-19 ratification; vault-file index persistence for iOS-eviction survival + cross-device sync; only Index location stays user-facing; seeds on next reindex — see field comment
    sidecarIndexLocation: 'config', // hidden literal '.obsidian/plugins/seek/index'; 'visible' = vault-root 'Seek Index/' for split-config Obsidian Sync; see field comment
    settingsRev: 8,            // current schema rev; bump alongside a migration in main.ts onload (rev 8 = 2026-06-27 denseWeight 0.80→0.85 re-eval)
};

// One-time settings migrations, keyed on the persisted settingsRev. Applied to the
// raw data.json object BEFORE it is merged over DEFAULT_SETTINGS in main.ts onload —
// without this, Object.assign(settings, DEFAULT_SETTINGS, raw) lets a stale persisted
// key silently win over a new default on every existing install. Mutates and returns
// `raw`; pure + exported so it is unit-testable without booting the plugin.
//
// (The rev-4 sidecar FILE move is NOT here — it does disk I/O and needs the plugin's
// resolved paths, so it stays in onload, gated on a `migrateSidecarPath` flag captured
// from the original settingsRev before this runs.)
export function migrateSettings(raw: Partial<SeekSettings>): Partial<SeekSettings> {
    const fromRev = raw.settingsRev ?? 1; // a data.json without the key is pre-bound (rev 1)
    // Rev 2 (2026-06-09 bound-norm switch): a denseWeight persisted on the old
    // empirical-max scale (0.90/0.92) is mis-calibrated under the theoretical-bound
    // normalization (the optimum moved to 0.80). Drop the key so the rev-2 default
    // takes over; a user who re-tunes afterwards persists a rev-2 value that survives.
    if (raw.denseWeight !== undefined && fromRev < 2) delete raw.denseWeight;
    // Rev 5 (2026-06-19 settings-redesign ratification): several validated-OFF debug
    // toggles become silent ON-defaults, navTitleBoost softens 0.8→0.5, and recency
    // ships fully Off (ε 0.02→0). The booleans flip UNCONDITIONALLY — their UI toggles
    // are removed, so the new baseline is for everyone. The two numeric defaults move
    // ONLY a user still on the exact old default, preserving any hand-tuned value.
    // CRITICAL for navTitleBoost: a persisted 0.8 (the old default) would otherwise
    // read as the new "High" segmented stage and silently promote every upgrader.
    if (fromRev < 5) {
        raw.synonymExpansion = true;
        raw.headingsField = true;
        raw.sidecarEnabled = true;
        if (raw.navTitleBoost === undefined || raw.navTitleBoost === 0.8) raw.navTitleBoost = 0.5;
        if (raw.recencyEpsilon === undefined || raw.recencyEpsilon === 0.02) raw.recencyEpsilon = 0;
    }
    // Rev 6 (2026-06-21 results-UI polish): the debug toggle `debugMode` was renamed
    // `showScores` (it now gates the calibrated "Matching %" line). Carry the user's
    // explicit choice across the rename — WITHOUT this, an upgrader who turned the old
    // per-row score line OFF would silently get it back ON (showScores falls through to
    // its `true` default, since the persisted `debugMode` key is an orphan Object.assign
    // ignores). Read the old key only when showScores wasn't already persisted, then drop
    // the orphan. Note: current installs sit at rev 5, so this MUST be a rev-6 bump — a
    // `fromRev < 5` gate would never fire for them.
    if (fromRev < 6) {
        const legacy = raw as { debugMode?: boolean };
        if (raw.showScores === undefined && legacy.debugMode !== undefined) raw.showScores = legacy.debugMode;
        delete legacy.debugMode;
    }
    // Rev 7 (2026-06-25 searchableProperties default ON): the BM25 frontmatter-values
    // field flips OFF→ON after the My-Vault channel eval measured +0.05 nDCG@10 (place-
    // note recall). Installs created under the old default persisted `false`, so a bare
    // DEFAULT_SETTINGS flip would be a silent no-op on them — migrate them. The toggle
    // still exists, so move only an install still on the old default; a deliberate post-
    // rev-7 `false` is indistinguishable from the default here (true was never persistable
    // before), which is acceptable — the new baseline is ON for everyone pre-rev-7. Refit-
    // only (search.ts bm25CacheProps mismatch → BM25 cache rebuild on next search), NO
    // embedding reindex: chunks already store metadata.properties.
    if (fromRev < 7 && (raw.searchableProperties === undefined || raw.searchableProperties === false)) {
        raw.searchableProperties = true;
    }
    // Rev 8 (2026-06-27 field-weight re-eval): denseWeight default moves 0.80 → 0.85.
    // The de-franken BM25 (more assertive lexical) made a fixed α=0.80 over-weight the
    // lexical channel; the cross-corpus re-sweep (Example Vault + enriched BEIR) put the knee
    // at ~0.85. Move ONLY an install still on the exact old default 0.80 — a hand-tuned
    // value survives, and an undefined falls through to the new DEFAULT_SETTINGS (0.85),
    // so a pre-bound install whose 0.92 the rev-2 surgery already dropped lands on 0.85
    // too. Score-time only: no reindex/refit (the dense weight is applied at fusion).
    if (fromRev < 8 && raw.denseWeight === 0.80) raw.denseWeight = 0.85;
    // Never DOWNGRADE the stamp: a data.json synced from a device running a newer
    // Seek (rev 9+) must keep its rev, or this older build stamps it back to 8 and
    // the newer device re-runs its migrations on next load (conditional default
    // moves misfire on second application).
    raw.settingsRev = Math.max(fromRev, 8);
    return raw;
}

// Vault size below which match-strength scores aren't meaningful (too few notes to
// calibrate the dense-cosine background). Both the settings Display chips toggle and
// the search-modal score read gate on this, so it lives here as one source of truth.
export const MATCH_STRENGTH_MIN_NOTES = 200;

export interface ScoredChunk extends Chunk {
    score: number;
    ranking_signals: {
        dense: number;
        bm25: number;
        hybrid: number;
        recency: number;
        title_boost: number;
        // Raw cosine similarity (query · chunk), BEFORE the lexical-only floor and
        // the per-query min-max that produce `dense`. This is the ABSOLUTE dense
        // match quality — instrumentation for the fusion confidence gate: `dense`
        // (normalized) always crowns a winner at 1.00 even for an OOV/ID query
        // where the best real cosine is weak, so the raw value is what tells us
        // "did the dense channel actually find anything." Logged in captures +
        // search telemetry so the gate's LO/HI thresholds can be set from real
        // numbers per embedding model rather than guessed.
        denseRaw: number;
        // Display-only confidence in [0,1]: denseRaw expressed relative to the
        // corpus dense-cosine background (dense-stats.ts calibratedConfidence).
        // Present only when the index carries bg stats (a full reindex on the
        // 2026-06-16+ build); never a ranking input. Rendered in the debug line.
        confidence?: number;
    };
    snippet?: string;
}

// ---- heap + memory ----

export interface HeapMB {
    mb: number | null;
    available: boolean;
}

// performance.memory is Chromium-only. On iOS WebKit and Safari this returns
// { mb: null, available: false } — every "Heap Δ" field in the report is
// blank there. See MemorySnapshot below for the iOS-friendly proxy.
export function snapshotHeap(): HeapMB {
    const mem = (performance as unknown as { memory?: { usedJSHeapSize: number } }).memory;
    if (!mem) return { mb: null, available: false };
    return { mb: mem.usedJSHeapSize / 1e6, available: true };
}

export function heapDelta(before: HeapMB, after: HeapMB): number | null {
    if (before.mb == null || after.mb == null) return null;
    return after.mb - before.mb;
}

// Full memory snapshot. `heapMB` mirrors snapshotHeap (null on iOS). `storageMB`
// reads navigator.storage.estimate().usage — counts on-disk IDB + Cache API
// bytes and works on iOS. It's not a true heap proxy but it does answer
// "did this operation actually write data to disk", which is the question
// that mattered for the iOS spike's jetsam diagnosis.
export interface MemorySnapshot {
    heapMB: number | null;
    storageMB: number | null;
    timestampMs: number;
}

export async function snapshotMemory(): Promise<MemorySnapshot> {
    const heap = snapshotHeap();
    let storageMB: number | null = null;
    if (typeof navigator !== 'undefined' && navigator.storage?.estimate) {
        try {
            const est = await navigator.storage.estimate();
            storageMB = est.usage != null ? est.usage / 1e6 : null;
        } catch { /* swallow */ }
    }
    return {
        heapMB: heap.mb,
        storageMB,
        timestampMs: performance.now(),
    };
}

export function memoryDelta(before: MemorySnapshot, after: MemorySnapshot): {
    heapDeltaMB: number | null;
    storageDeltaMB: number | null;
    elapsedMs: number;
} {
    return {
        heapDeltaMB: before.heapMB != null && after.heapMB != null ? after.heapMB - before.heapMB : null,
        storageDeltaMB: before.storageMB != null && after.storageMB != null ? after.storageMB - before.storageMB : null,
        elapsedMs: after.timestampMs - before.timestampMs,
    };
}

// ---- distribution stats ----

// Generic min/max/p50/p95 over a sample. Used to summarize per-file timings
// and chunks-per-file without bloating the log with raw arrays.
export interface DistributionStats {
    n: number;
    min: number;
    max: number;
    mean: number;
    p50: number;
    p95: number;
}

export function distributionStats(samples: number[]): DistributionStats | null {
    if (samples.length === 0) return null;
    const sorted = [...samples].sort((a, b) => a - b);
    const sum = sorted.reduce((s, v) => s + v, 0);
    const pick = (q: number) => sorted[Math.min(sorted.length - 1, Math.floor((q / 100) * (sorted.length - 1)))];
    return {
        n: sorted.length,
        min: sorted[0],
        max: sorted[sorted.length - 1],
        mean: sum / sorted.length,
        p50: pick(50),
        p95: pick(95),
    };
}

// ---- log entry types (NDJSON schema) ----

export interface InitEntry {
    type: 'init';
    timestamp: string;
    schemaVersion: number;
    buildTimestamp: string;
    transformersVersion: string;
    cdnUrl: string;
    iframeReady: boolean;
    // Wall time of the iframe build (createElement + srcdoc bootstrap + __ready
    // handshake). 0 when init early-returned because the iframe was already live
    // (recycle/teardown/unload re-entry), so a 0 here is "no build", not "instant".
    initMs: number;
    pluginVersion: string;
    error: string | null;
}

export interface AdapterLimits {
    maxBufferSize: number | null;
    maxStorageBufferBindingSize: number | null;
    maxComputeWorkgroupSizeX: number | null;
    maxComputeInvocationsPerWorkgroup: number | null;
}

export interface PlatformEntry {
    type: 'platform';
    timestamp: string;
    isMobile: boolean;
    userAgent: string;
    iosVersion: number | null;
    gpuAvailable: boolean;
    gpuAdapterDescription: string | null;
    // true = software fallback adapter (SwiftShader-class): the "GPU yes but
    // not really" case where ORT WebGPU init typically fails (e.g. hardware
    // acceleration disabled). null = attribute not exposed on this platform.
    gpuIsFallbackAdapter: boolean | null;
    gpuAdapterLimits: AdapterLimits | null;
    storageUsedMB: number | null;
    storageQuotaMB: number | null;
    persistGranted: boolean | null;
    heapAvailable: boolean;        // Whether performance.memory is exposed (false on iOS WebKit).
    measureMemoryAvailable: boolean; // Whether performance.measureUserAgentSpecificMemory exists.
    crossOriginIsolated: boolean;  // Required for measureUserAgentSpecificMemory.
}

export interface LoadEntry {
    type: 'load';
    timestamp: string;
    requestedDevice: RequestedDevice;
    actualDevice: Device;
    dtype: Dtype;
    // Output vector dimension AFTER MRL slicing (the on-the-wire / on-disk
    // shape). EmbeddingGemma's native output is 768; we slice + L2-renormalize
    // in the iframe so callers never see the full 768d.
    embeddingDim: number;
    coldStartMs: number;
    // Wall-time of the WebGPU warmup sweep (40 forced shader-compile
    // dispatches across batch × seq_bucket); included in coldStartMs. Use
    // (coldStartMs − warmupMs) to recover pre-warmup load time (Cache API
    // read + ONNX parse + WebGPU device init). NULL when:
    //   - WASM path (no warmup applies; warmupSkipped=false)
    //   - WebGPU + skipWarmup hit (fingerprint cache says shaders are warm;
    //     warmupSkipped=true)
    // The boolean disambiguates the two null cases.
    warmupMs: number | null;
    warmupSkipped: boolean;
    heapBeforeMB: number | null;
    heapAfterMB: number | null;
    heapDeltaMB: number | null;
    storageBeforeMB: number | null;
    storageAfterMB: number | null;
    storageDeltaMB: number | null;
    webgpuAttempted: boolean;
    webgpuFailed: boolean;
    webgpuError: string | null;
    // Non-null when a glue override applied: 'jspi' | 'asyncify' (WebKit
    // WebGPU) or 'plain' (non-WebKit wasm pin — the only ort-wasm build with
    // the CPU GatherBlockQuantized kernel). Previously only in `checks` text.
    glue: string | null;
    pass: boolean;
    checks: string[];
}

export interface IndexProgressEntry {
    type: 'index-progress';
    timestamp: string;
    phase: 'scan' | 'chunk' | 'embed' | 'bm25' | 'commit';
    filesSeen: number;
    filesTotal: number;
    chunksEmitted: number;
    elapsedMs: number;
    heapMB: number | null;
    storageMB: number | null;
}

export interface IndexCompleteEntry {
    type: 'index-complete';
    timestamp: string;
    mode: 'full' | 'incremental';
    // What dtype + dim were used to produce these vectors. Recorded so a
    // future model swap can detect that the on-disk index doesn't match the
    // currently loaded runtime.
    dtype: Dtype;
    embeddingDim: number;
    filesIndexed: number;          // files STARTED (== input length on a full reindex)
    // The subset of filesIndexed whose file-record was actually written (commitFile
    // succeeded). Drives the catch-up drain's forward-progress accounting — distinct from
    // filesIndexed whenever a file is empty/skipped/budget-deferred. See embedAndCommitFiles.
    committedFilePaths: string[];
    chunksIndexed: number;
    vectorsWritten: number;
    filesSkippedError: number;
    // Incremental budget: files in the input list that were never started because
    // a per-burst budget (maxFiles / budgetMs / shouldContinue) cut the pass short.
    // 0 on a full reindex (always unbounded) and on a delta that ran to completion.
    // The drain loop in main.ts reads this to decide whether to re-fire another
    // burst. Optional: absent on entries written before this field existed.
    filesDeferred?: number;
    // How many times the embed session was torn down + rebuilt mid-run to
    // recover from ORT-Web's WebGPU SafeInt overflow (embedder.recycle).
    // 0 on a healthy run; >0 means the overflow was hit and recovered.
    embedRecycles: number;
    // WS2.3 token-budget enforcement (token-budget.ts). splits = chunks whose
    // embed input exceeded the 512-token window and were re-packed under it;
    // overBudget = inputs that STILL exceed the window (unsplittable
    // ~window-filling titles only) and therefore truncate — the lone permitted
    // nonzero in the dense-invisible-share→0 invariant. Optional: absent on
    // pre-WS2.3 log entries.
    tokenBudgetSplits?: number;
    tokenBudgetOverBudget?: number;
    embedDurationMs: number;
    chunkDurationMs: number;
    bm25DurationMs: number;
    commitDurationMs: number;
    totalDurationMs: number;
    heapDeltaMB: number | null;
    storageDeltaMB: number | null;
    // Throughput rollups computed from totalDurationMs.
    chunksPerSec: number;
    filesPerSec: number;
    // Per-file wall-clock distribution (chunk+embed+commit summed per file).
    perFileWallMs: DistributionStats | null;
    chunksPerFile: DistributionStats | null;
    // Per-embed-batch latency as reported by transformers.js inside the iframe.
    embedBatchLatencyMs: DistributionStats | null;
    pass: boolean;
    checks: string[];
}

export interface SearchEntry {
    type: 'search';
    timestamp: string;
    query: string;
    topK: number;
    // Inline-filter parser output (v8+). `cleanedQuery` is the residual
    // semantic text actually embedded + BM25'd after stripping
    // #tag/tag:/path:/[k:v]/date operators; `filters` is the structured
    // extraction (null = plain query, no operators → pre-v8 behavior).
    cleanedQuery: string;
    filters: QueryFilters | null;
    // Stage timers — every one is wall-clock around a discrete pipeline phase.
    // Sum is ≤ totalMs; the residual is async scheduling overhead.
    //
    // Two-stage layout (v7+):
    //   idbReadMs        listAllChunks + listAllBinary (cached after first run)
    //   binaryMs         asymmetric float·sign-bit scoring across all chunks
    //   selectFetchMs    fp32 fetch for ONLY the union candidates (S2 prep)
    //   alignMs          building the chunk-subset → vector Map for the union
    //   cosineMs         exact cosine, but over the union — not all chunks
    //   bm25Ms / fusionMs / snippetMs unchanged
    idbReadMs: number;
    binaryMs: number;
    selectFetchMs: number;
    alignMs: number;
    queryEmbedMs: number;
    iframeEmbedMs: number;
    cosineMs: number;
    bm25Ms: number;
    bm25CacheHit: boolean;
    fusionMs: number;
    snippetMs: number;
    totalMs: number;
    totalChunks: number;
    // Two-stage candidate accounting:
    //   *TopN/M/K are the configured caps (read of the live config at search time);
    //   *Count are how many *unique* ids each arm contributed *before* union dedup;
    //   candidateUnionSize is the post-dedup size of the set fed to the S2 reranker.
    binaryTopN: number;
    bm25TopM: number;
    recencyTopK: number;
    binaryCount: number;
    bm25Count: number;
    recencyCount: number;
    candidateUnionSize: number;
    // True if the resident binary index was loaded from RAM (cache hit). False
    // means stage 1 paid an IDB cursor walk over the binary store this call.
    binaryCacheHit: boolean;
    rawDenseTop5: Array<{ chunk_id: string; score: number }>;
    rawBm25Top5: Array<{ chunk_id: string; score: number }>;
    // Ranking trace for offline analysis. Persisted depth is gated by the
    // verboseTrace setting: top-10 by default (exactly what the report renders,
    // keeping the append-only log small), or the full 50-deep tail when verboseTrace
    // is on — for spreadsheet / pandas eval over the long tail. NOTE: from v7
    // this trace is over the CANDIDATE UNION (~250), not all chunks — the
    // "rank" field is the candidate-set rank, not a global rank.
    fusedTop50: Array<{
        chunk_id: string;
        note_path: string;
        rank: number;
        score: number;
        dense: number;
        // Raw cosine before lexical-only floor + min-max (see ScoredChunk
        // ranking_signals.denseRaw) — the absolute dense match quality.
        denseRaw: number;
        bm25: number;
        recency: number;
        title_boost: number;
        title: string;
    }>;
    alpha: number;             // the dense weight in force (settings.denseWeight)
    // Since 2026-06-11 this carries the recency EPSILON (the additive tiebreaker
    // in final = hybrid + ε·recency + titleBoost). The field name is kept for
    // log-schema stability: rows before 06-08 hold the old blend weight, rows
    // 06-08..11 hold the parked 0, rows after hold ε (~0.02).
    recencyWeight: number;
    // Which date drove recency this search ('created' | 'modified'). Optional so
    // logs predating the recencyKey setting (2026-06-11) still parse; absent ⇒
    // the pre-key behavior (created with mtime fallback, same as 'created').
    recencyKey?: RecencyKeyChoice;
    // HISTORICAL (RRF deleted 2026-06-11): blend config, written only by rows
    // logged while the opt-in RRF mode existed (2026-06-09..11). Absent ⇒ linear.
    // Kept optional so those rows still parse; never written again.
    blendMode?: 'linear' | 'rrf';
    rrfK?: number;
    // Whether the soft-AND BM25 coverage weight was applied this search. Optional
    // so logs predating the lever still parse; absent ⇒ false (the old OR behavior).
    bm25Coverage?: boolean;
    // Whether last-token prefix expansion was applied this search. Optional so
    // logs predating the lever (2026-06-10) still parse; absent ⇒ false.
    prefixLastToken?: boolean;
    // Whether alias-dictionary synonym expansion was applied this search.
    // Optional so logs predating the lever still parse; absent ⇒ false.
    synonymExpansion?: boolean;
    // Whether frontmatter properties were indexed as a BM25 field this search.
    // Optional so logs predating the lever (2026-06-11) still parse; absent ⇒ false.
    searchableProperties?: boolean;
    // Whether the heading path was indexed as a BM25 field this search.
    // Optional so logs predating the lever (2026-06-12) still parse; absent ⇒ false.
    headingsField?: boolean;
    // Per-query theoretical BM25 bound (bm25.getQueryBound) the fusion divided
    // by; 0 ⇒ the bound had no opinion (fully-OOV query, or MiniSearch internals
    // unavailable) and fusion fell back to the empirical max. max(rawBm25)/bound
    // is the lexical channel's per-query confidence — the weak-lexical diagnostic.
    // Optional so logs predating the bound norm still parse.
    bm25Bound?: number;
    // Search-level identifier so click events can be correlated back to
    // the originating search row. Time-based since we don't have a UUID
    // generator hot path and the timestamp+query pair is already unique
    // in practice.
    searchId: string;
}

// Emitted when the user clicks a search result. Captures both the chosen
// chunk and the top-10 chunk_ids the user passed over, so offline analysis
// can compute "did they click rank-1 or not" CTR-style relevance signals.
// `dwellMs` is the time between search-completion and click — proxies for
// "did they actually look at the results before clicking, or accept the
// first thing they saw."
export interface ClickEntry {
    type: 'click';
    timestamp: string;
    searchId: string;       // matches SearchEntry.searchId
    query: string;
    chunk_id: string;
    note_path: string;
    rank: number;            // 1-based rank in the deduped result list shown
    score: number;
    dense: number;
    bm25: number;
    recency: number;
    title_boost: number;
    dwellMs: number;         // ms between search completion and the click
    shownTop10: string[];    // chunk_ids the user passed over (or chose from)
}

export interface ErrorEntry {
    type: 'error';
    timestamp: string;
    context: string;
    message: string;
    stack: string | null;
    // Running occurrence count carried on a milestone/flush row when errors are deduped
    // by message: how many times this message has fired this session as of this row.
    // Errors are written at exponential milestones (2,4,8,…), not one row each, so a
    // chronic fault like "iframe not initialized" costs ~log2(N) rows instead of N.
    // Absent ⇒ a first/only occurrence.
    repeated?: number;
}

export interface ResetEntry {
    type: 'reset';
    timestamp: string;
    droppedDatabase: string;
    chunksDeleted: number;
    vectorsDeleted: number;
    durationMs: number;
    pass: boolean;
    checks: string[];
}

// Long-running main-thread tasks observed via PerformanceObserver during
// indexing. iOS jank is invisible unless we record these — 250 ms is the
// rough threshold above which the user perceives a stutter.
export interface LongTaskEntry {
    type: 'long-task';
    timestamp: string;
    durationMs: number;
    startTimeMs: number;
    attribution: string | null; // PerformanceLongTaskTiming.attribution[0].name when available
    context: string;            // 'indexing' | 'search' | 'idle' — set by main.ts when wiring observer
}

// Captured on `visibilitychange` (hidden) and `pagehide`. The user-visible
// motivation is iOS jetsam: if the WebView is killed under memory pressure,
// the last memory-pressure entry tells us what state we were in when killed.
export interface MemoryPressureEntry {
    type: 'memory-pressure';
    timestamp: string;
    event: 'visibility-hidden' | 'pagehide' | 'visibility-visible';
    heapMB: number | null;
    storageMB: number | null;
    persisted: boolean;
}

// Emitted when the embedder is proactively torn down to release the iframe's
// monotonically-growing WASM heap (WebAssembly.Memory never shrinks within a
// page, so a long mobile session ratchets to the OOM ceiling). 'idle' = no model
// use for IDLE_UNLOAD_MS in a quiescent state; 'background' = mobile app
// backgrounded (free before iOS can jetsam-kill). The next search/embed
// transparently reloads via ensureModelLoaded — the FOLLOWING LoadEntry's
// coldStartMs is the reload cost this trades for the heap reset. heapMB is the JS
// heap at unload (desktop only; null on iOS, where performance.memory is absent).
export interface ModelLifecycleEntry {
    type: 'model-lifecycle';
    timestamp: string;
    event: 'unload';
    reason: 'idle' | 'background';
    heapMB: number | null;
}

// ---- crash forensics (schema v10) ----

// One synchronous localStorage breadcrumb. Deliberately tiny (the ring is
// rewritten on every beat) and self-describing enough to reconstruct what the
// process was doing when it died: visibility state at write time plus a
// per-beat detail payload (filesCommitted, dispatch counters, ...).
export interface ForensicBreadcrumb {
    t: string;                       // ISO timestamp
    type: string;                    // 'session-start' | 'visibility-hidden' | 'index-flush' | ...
    vis: 'visible' | 'hidden';       // document.visibilityState at write time
    detail?: Record<string, number | string | boolean | null>;
}

// How bootInspect() classifies an unclean prior session. The verdict is the
// whole point of the forensics layer: it discriminates the iPhone-reindex
// death hypotheses that the async log physically cannot.
//   crash-while-indexing-foreground — process died mid-reindex, app visible:
//       memory-ceiling signature (jetsam under foreground GPU/heap burst).
//   crash-while-indexing-hidden — process died mid-reindex while backgrounded:
//       iOS background-GPU termination signature.
//   evicted-while-hidden — died backgrounded and idle: ordinary iOS
//       suspended-app eviction, expected lifecycle, not a bug.
//   crash-foreground — died visible but not indexing (load burst, query, ...).
//   unknown — no breadcrumbs beyond session-start.
export type CrashVerdict =
    | 'crash-while-indexing-foreground'
    | 'crash-while-indexing-hidden'
    | 'evicted-while-hidden'
    | 'crash-foreground'
    | 'unknown';

// Logged at boot when the previous session's forensics record has no clean-end
// marker. Carries the breadcrumb tail so the report can show the last thing
// the dead process did.
export interface CrashDetectedEntry {
    type: 'crash-detected';
    timestamp: string;
    // Identity of the session that died (NOT the booting session stamped by
    // the logger — that's the session that found the body).
    deadSessionId: string;
    verdict: CrashVerdict;
    // Last breadcrumb before death + how long after it the next boot happened.
    lastBeat: ForensicBreadcrumb | null;
    gapSeconds: number | null;
    // Tail of the ring (most recent last), capped — enough to see the run-up.
    breadcrumbs: ForensicBreadcrumb[];
}

// On-demand snapshot used after reindex and at other interesting moments.
// Cheaper than the full platform probe; only captures the volatile fields.
export interface StorageSnapshotEntry {
    type: 'storage-snapshot';
    timestamp: string;
    context: string;
    storageUsedMB: number | null;
    storageQuotaMB: number | null;
    heapMB: number | null;
}

// Promoted form of the coldStartMs ≥ 5000 check that lived inside the load
// entry's `checks` array. Surfacing it as its own event lets the report
// (and any future alerting) count occurrences without parsing free-text
// checklist strings. Carries the storage state at the moment of suspected
// eviction so a low storageUsedMB at the same timestamp confirms the
// cache was actually emptied (vs. a slow disk or thermal-throttle false
// positive).
export interface EvictionSuspectedEntry {
    type: 'eviction-suspected';
    timestamp: string;
    coldStartMs: number;
    actualDevice: Device;
    dtype: Dtype;
    storageUsedMB: number | null;
    storageQuotaMB: number | null;
    persisted: boolean | null;
}

// One-shot result of the `app://local/...` capability probe. Gates the
// Phase 3 model-shard pattern: if `ok`, the iframe can stream shards from
// the vault via a resource URL; if `blocked`, Phase 3 falls back to
// transferring shard bytes through postMessage at cold start.
export interface AppLocalFetchEntry {
    type: 'app-local-fetch';
    timestamp: string;
    result: 'ok' | 'blocked' | 'unknown';
    // Full resource URL we attempted (`app://local/...` on desktop,
    // `capacitor://localhost/...` on iOS Capacitor, etc.). Recorded so
    // the report can correlate the result with the platform's URL scheme.
    url: string;
    httpStatus: number | null;
    bodyMatched: boolean | null;
    error: string | null;
}

// Emitted once per successful model load by the production model-delivery layer
// (model-registry.ts + main.ts). `key`/`repo`/`revision` identify the active spec;
// `persisted` is navigator.storage.persisted() (Cache-API durability on this
// device); `cacheSeen`/`cacheEvicted` report the parent-side eviction sweep of the
// transformers-cache — `cacheSeen === 0` is the canary that the parent can't see
// the iframe's cache partition (move eviction to an iframe RPC if it ever appears).
export interface ModelDeliveryEntry {
    type: 'model-delivery';
    timestamp: string;
    key: string;
    repo: string;
    revision: string | null;
    persisted: boolean | null;
    cacheSeen: number;
    cacheEvicted: number;
}

// ---- runtime profile (wall-time decomposition) ----
//
// One cell per (batchSize, seqBucket). The point is NOT throughput — it's
// *where the wall time goes*, because that ratio is the decision variable
// the idea-list questions all reduce to:
//   - tokenizeSharePct high  → a WASM/Rust tokenizer is worth it (else noise)
//   - forwardSharePct high   → pipeline is GPU-forward-bound; I/O binding
//                              (which only removes copies/seam stalls) has
//                              little to recover — the v4 kernel fix already
//                              captured the win
//   - forwardSharePct low / post(readback) high → serialization-bound; the
//                              "36% GPU" diagnosis still holds on v4 and the
//                              worker / readback levers matter
// `postMs` = time to materialize last_hidden_state.data (forces the WebGPU
// GPU→CPU readback) — deliberately measured separately because the readback
// sync-stall is exactly what the "per-inference serialization" hypothesis is
// about. It is NOT full pool+normalize; it is the readback boundary cost.
// `pipelineTotalMs` runs the *production* pipeline() path on the same inputs
// as a decomposition sanity check (tokenize+forward+post should ≈ it).
export interface ProfileCell {
    batchSize: number;
    seqBucket: number;
    reps: number;
    tokenizeMs: DistributionStats | null;
    forwardMs: DistributionStats | null;
    postMs: DistributionStats | null;
    pipelineTotalMs: DistributionStats | null;
    // p50-derived shares — the actual decision read.
    forwardSharePct: number | null;
    tokenizeSharePct: number | null;
    // forward p50 ÷ batchSize: per-text GPU cost, the throughput proxy.
    perTextForwardMs: number | null;
}

export interface EmbedProfileEntry {
    type: 'embed-profile';
    timestamp: string;
    schemaVersion: number;
    device: Device;
    dtype: Dtype;
    transformersVersion: string;
    cells: ProfileCell[];
    // Heap Δ across the whole non-disposing run. The profile path deliberately
    // does NOT dispose output tensors, so a climbing delta here is an early,
    // free read on the "undisposed iframe output tensor" leak hypothesis —
    // without yet committing to the disposal change itself.
    heapDeltaMB: number | null;
    elapsedMs: number;
    notes: string;
}

// Phase-5 trimmed-model smoke test (seek-phase5-smoke command). A crash-survivable
// probe: one entry per stage is disk-flushed before the next heavy step. The
// per-stage payload (loadMs, dim, norm, device, error, …) varies by stage, so it
// stays open via an index signature — this is a debug-only log, not a schema'd
// analytics event.
export interface Phase5SmokeEntry {
    type: 'phase5-smoke';
    timestamp: string;
    stage: string;
    [key: string]: string | number | boolean | undefined;
}

// WebGPU device lifecycle event relayed from the iframe's requestDevice hook
// (kind: webgpu-device-created / webgpu-device-lost / webgpu-uncaptured-error).
// device-lost is the only JS-visible discriminator between a GPU-process death
// (page survives and sees it) and a WebContent kill (total silence) — the load-
// bearing diagnostic after three iPhone reindex deaths left zero OS-side
// forensics. The same event is ALSO written synchronously to the forensics
// breadcrumb ring (which survives process death); this NDJSON twin is the
// queryable copy for sessions that lived to tell.
export interface WebgpuEventEntry {
    type: 'webgpu-event';
    timestamp: string;
    kind: string;
    [key: string]: string | number | boolean | null | undefined;
}

// Sidecar hydrate diagnostics, persisted to NDJSON so they're visible on mobile
// (the prior console.log-only hook was invisible on iOS — why hydrate outcomes
// never appeared in phone reports). `phase` = the deps.log msg: 'sidecar-hydrate-scan'
// (early producer-file probe, carries producerFilesFound + devices — the decisive
// "did the desktop sidecar reach this device" signal) or 'sidecar-hydrate' (result:
// scanned/needed/hydrated/accepted/refusedProducers). Array fields are flattened to
// `<key>Count` by the writer to fit the scalar index signature.
export interface SidecarHydrateEntry {
    type: 'sidecar-hydrate';
    timestamp: string;
    phase: string;
    [key: string]: string | number | boolean | null | undefined;
}

// Stamped onto every entry by logger.append(). Optional so pre-v9 logs (which
// predate device/session attribution) still parse — the report treats a missing
// deviceId as 'legacy' and a missing sessionId as un-scopable.
export interface LogMeta {
    deviceId?: string;
    sessionId?: string;
}

// Intersecting the union with LogMeta keeps `.type` discrimination working
// (TS distributes the intersection) while making `.deviceId` / `.sessionId`
// readable on any LogEntry without narrowing first.
export type LogEntry = (
    | InitEntry
    | PlatformEntry
    | LoadEntry
    | EmbedProfileEntry
    | IndexProgressEntry
    | IndexCompleteEntry
    | SearchEntry
    | ClickEntry
    | ErrorEntry
    | ResetEntry
    | LongTaskEntry
    | MemoryPressureEntry
    | ModelLifecycleEntry
    | CrashDetectedEntry
    | StorageSnapshotEntry
    | EvictionSuspectedEntry
    | AppLocalFetchEntry
    | ModelDeliveryEntry
    | Phase5SmokeEntry
    | WebgpuEventEntry
    | SidecarHydrateEntry
) & LogMeta;
