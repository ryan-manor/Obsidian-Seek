# Seek — Admin Controls Design Brief

> **Audience:** the design session redesigning Seek's Settings tab.
> **Purpose:** give you the relevance model in plain language, the controls worth exposing, and the new Index panel (stats + reindex) that should replace the command-palette-only workflow.
> **Scope note:** this is a *what-to-surface and how-to-frame* brief, not an implementation spec. Source references are included so you can confirm anything against the code.

---

## 1. What Seek Is (One Paragraph)

Seek is a fully on-device semantic search plugin for Obsidian. There is no server. When you index, Seek splits each note into chunks, turns each chunk into a vector with a local embedding model (granite-embedding-small-r2, runs in the browser via WebGPU/WASM), and also builds a classic keyword index (BM25). At query time it runs **both** searches, **fuses** the two ranked lists into one, applies a small **title boost** for known-item lookups, and returns the result. Everything the user can tune lives in that pipeline.

---

## 2. The Relevance Information Architecture

This is the mental model the settings UI should make legible. Every knob maps to one stage here.

```
        Your note                          Your query
            │                                   │
       ┌────▼─────┐                       ┌──────▼───────┐
       │ Chunking │                       │ Query parser │  ← inline filters (#tag, path:, [k:v], dates)
       └────┬─────┘                       └──────┬───────┘
            │ (split into passages)              │
   ┌────────┴────────┐                  ┌────────┴─────────┐
   ▼                 ▼                  ▼                  ▼
┌──────┐        ┌────────┐         ┌────────┐         ┌────────┐
│Dense │        │ BM25   │         │ Dense  │         │ BM25   │
│vector│        │keyword │  ...... │ search │         │ search │
└──────┘        └────────┘         └───┬────┘         └───┬────┘
 (index time)                          │                  │
                                       └────────┬─────────┘
                                                ▼
                                        ┌───────────────┐
                                        │    FUSION     │  ← "Dense weight" (α) + Blend mode
                                        │  α·dense +    │
                                        │  (1−α)·keyword│
                                        └───────┬───────┘
                                                ▼
                                        ┌───────────────┐
                                        │  Title boost  │  ← "Navigational boost"
                                        │ (known-item)  │
                                        └───────┬───────┘
                                                ▼
                                            Results
```

**The single most important concept to communicate to the end user: the Dense ↔ Keyword balance.** Semantic (dense) search finds notes that *mean* the same thing even with different words; keyword (BM25) search finds *exact* terms, names, and acronyms. Fusion blends them, and one slider controls the balance. Almost everything else is secondary.

```
★ Insight ─────────────────────────────────────
• The whole settings surface is really one pipeline with ~3 user-meaningful
  dials (balance, fuzzy, navigational boost). Resist the urge to give equal
  visual weight to every config field — the IA should foreground the
  Dense↔Keyword balance and let the rest recede.
• "Dense weight" is an internal name. For users, frame it as a *balance*
  with two labeled ends (Meaning ↔ Exact words), not a lone 0–1 number.
• Recency exists in the engine but is currently parked (weight forced to 0).
  Don't design a recency control yet — see §6.
─────────────────────────────────────────────────
```

---

## 3. Controls Worth Exposing (and How to Frame Them)

These already exist in the engine. The table gives the user-facing framing; the "Internal" column is the real config name so you can map back to code.

| User-facing control | Internal name | Type | Default | What it does (user words) | Presentation note |
|---|---|---|---|---|---|
| **Search balance** (Meaning ↔ Exact words) | `alpha` | slider 0–1 | **0.90** | Left = match exact keywords; right = match meaning. Default leans toward meaning. | This is the hero control. A labeled slider beats a number box. Consider tick at default. |
| **Fuzzy matching** | `fuzzy` | toggle | **off** | Tolerate one-character typos in keyword matching. | Simple on/off. Help text: "Find results even with small spelling mistakes." |
| **Navigational boost** | `titleBoost` | slider 0–1 | **0.8** | When your whole query appears in a note's title, push that note up. Helps "jump to the note I named." | Secondary. Could live under an "Advanced relevance" disclosure. |
| **Blend mode** | `blendMode` | dropdown | **Linear** | How the two result lists are combined. Linear (score-based) vs RRF (rank-based). | Power-user control. Most users never touch it. Hide under Advanced. |
| **RRF k** | `rrfK` | number | **60** | Only relevant when Blend mode = RRF. | Show **only** when blendMode === 'rrf'. Otherwise hide entirely. |
| **Honor excluded folders** | `honorIgnored` | toggle | **on** | Skip notes in Obsidian's "Excluded files" setting (e.g. Archive) when indexing. | Belongs in Index section, not Relevance. |

**Recommended grouping (the IA the user should perceive):**

- **Relevance** → Search balance (hero) · Fuzzy matching
  - *Advanced relevance* (collapsed) → Navigational boost · Blend mode · RRF k
- **Index** → status panel + Full reindex (see §4) · Honor excluded folders
- **Experimental** (collapsed) → Mobile WebGPU (see §6)

All Relevance controls apply on the **next search** — no reindexing needed. Worth saying in the UI so users feel free to experiment. The Index controls are the only ones with a heavier cost.

---

## 4. The Index Panel — Stats + Reindex (Replaces the Command)

Today, a full reindex is only reachable via the command palette (`Full reindex (nuke and rebuild)`). The ask is to **bring this into Settings** alongside live index statistics, so the user can see the state of their index and rebuild it without hunting for a command.

### Stats to display

All of these are already computable from existing data structures — nothing new needs to be measured, just surfaced:

| Stat | Source / availability | Notes for display |
|---|---|---|
| **Files indexed** | `index-store.count().files` | The headline number. |
| **Chunks** | `index-store.count().chunks` | "Passages" may read better than "chunks" for users. |
| **Embeddings** | `index-store.count().embeddings` | Usually tracks chunks; a mismatch signals an incomplete index. |
| **Index size on disk** | `navigator.storage.estimate()` (already called in `emitStorageSnapshot`) | Show human-readable (MB). |
| **Last indexed** | per-file record timestamps | "Last updated 3 min ago" style. |
| **Model** | `embedder` `MODEL_ID` (granite-embedding-small-r2) | Static, but reassures the user what's running. |
| **Embedding dimensions** | fixed at model load (384) | Optional / power-user detail. |

Storage persistence and incremental indexing already run in the background — the index updates as notes change. So the panel should read as a **health/status readout**, with full reindex as the explicit "rebuild from scratch" escape hatch, not the primary action.

### Full reindex button — behaviors to design for

- **Destructive, confirmed.** It deletes the entire index and re-embeds every note. Style it as a destructive action and keep the existing confirmation step.
- **Long-running with progress.** On a real vault this embeds thousands of chunks and takes minutes. The engine already logs progress during reindex — the UI should show live progress (e.g. "Indexed 1,240 / 4,800 passages") and a way to tell it's still working, not hung.
- **State-aware.** While a reindex runs, the button should reflect in-progress state (disabled / "Reindexing…" / optional cancel). After completion, the stats above should refresh.
- **Why a user reaches for it:** they changed chunking, switched models, or suspect the index is stale/corrupt. It is a recovery tool, so it's fine for it to be slightly buried and clearly labeled — but it must be findable in Settings, which it currently isn't.

```
★ Insight ─────────────────────────────────────
• The reindex button is the one control with real, irreversible cost and
  minutes of wall-clock. It deserves the opposite visual treatment from the
  relevance sliders: those invite play; this one demands a beat of friction.
• Pairing it with live stats is the actual UX win — users reindex blindly
  today because they have no signal about index health. The stats give them
  a reason to *not* reindex, which is usually the right call.
─────────────────────────────────────────────────
```

---

## 5. Query Syntax (Part of the Relevance Story, Optional to Surface)

Seek's query box accepts inline filters. These aren't settings, but if you design any help text, placeholder, or a "search tips" affordance, this is the vocabulary:

- `#tag` / `#parent/child` / `tag:x` — filter by tag (hierarchical: `meetings` matches `meetings/1x1`)
- `path:Folder/*` — filter by file path (glob)
- `[key:value]` — filter by frontmatter (`[context:work]`)
- `[created:>2026-01-01]`, `[modified:<...]` — date filters
- `-term` — exclude a word

This is worth a lightweight discoverability surface (placeholder text or a small "?" popover near the search box) but is **not** part of the Settings redesign per se. Flagged here so the relevance picture is complete.

---

## 6. What to Leave Alone (Important)

Designers naturally want to expose everything they find. Several things are deliberately **not** user controls — surfacing them would be a regression:

- **BM25 field boosts** (title 10×, aliases 6×, tags 3×) — eval-tuned constants, were removed from the UI on purpose (marginal effect, easy to misconfigure). Keep them hidden.
- **BM25 k1/b/d, candidate caps (200/100/50), chunk size, padding buckets, debounce timers** — internal tuning. No UI.
- **Recency weight** — the engine supports a recency blend but it's **currently forced to 0** pending a rework. Do **not** design a recency slider yet; if you want a placeholder, leave room in the Relevance group but don't wire a live control.
- **Mobile WebGPU (experimental)** — a real toggle, but genuinely experimental (can hang on iOS). Keep it in a clearly-labeled **Experimental** section, off by default, with a warning. It only takes effect on the next model load, not immediately.
- **Diagnostic commands** (profile runtime, smoke test, unload model, generate log report) — these stay in the command palette. They are developer tools, not admin controls, and don't belong in the redesigned Settings.

---

## 7. Quick Reference — Defaults at a Glance

| Control | Default | Applies |
|---|---|---|
| Search balance (`alpha`) | 0.90 (leans to meaning) | next search |
| Fuzzy matching | off | next search |
| Navigational boost (`titleBoost`) | 0.8 | next search |
| Blend mode | Linear | next search |
| RRF k | 60 (RRF mode only) | next search |
| Honor excluded folders | on | next index |
| Recency | parked (0) — no control | — |
| Mobile WebGPU | off (experimental) | next model load |

---

*Source of truth in the Seek repo: settings tab `src/main.ts` (`SeekSettingTab`), ranking config `src/ranker.ts` (`DEFAULT_RANKING_CONFIG`), fusion math `src/fusion.ts`, lexical/fuzzy `src/bm25.ts`, query syntax `src/query-parser.ts`, index counts `src/index-store.ts` (`count()`), reindex command `src/main.ts` (`seek-full-reindex`).*
