// Score fusion, recency, and title-match boost.
// Direct port of python-backend/app/search/fusion.py and the title-boost helper
// in ranker.py. Coefficients match the live Python backend's defaults.

import { depluralize, ENGLISH_STOPWORDS, foldDiacritics } from './bm25';
import { hasCjk, segmentCjkToken } from './tokenize';
import type { RecencyKeyChoice } from './types';

// (minmaxNormalize deleted 2026-06-11 with the RRF blend mode — its last
// consumer. Per-query min-max as a CHANNEL normalizer was already replaced by
// TM2C2 on 2026-06-09; see the theoreticalNorm* functions below.)

// ---- Recency (ε-tiebreaker, 2026-06-11) ------------------------------------
//
// Recency re-entered the ranking path as an ADDITIVE EPSILON TIEBREAKER
// (ranker.ts: final = hybrid + ε·recency + titleBoost), replacing both the
// blend form ((1−rw)·hybrid + rw·recency) and the two-stage linear decay
// (shallowDays/cutoffDays/shallowRate) that shipped with v0 and was parked at
// rw=0 on 2026-06-08. Why this shape (full plan: [[Seek Rel]] §Recency Plan
// 2026-06-11):
//   * The 06-04 click study killed recency as a WEIGHT: 50% of episodic clicks
//     target an instance >90d old, so any newest-lean strong enough to reorder
//     real score gaps hurts more than it helps (forced newest-first within a
//     series scored WORSE than flat relevance, MRR 0.242 vs 0.310). ε is sized
//     below meaningful score gaps, so it can only break genuine near-ties —
//     dated-series siblings with identical coverage boost and near-identical
//     dense — where newest is a sane prior over arbitrary frame order.
//   * The old 30d hard cutoff zeroed the MEDIAN episodic target (83d old); the
//     smooth half-life below keeps the whole series ordered at every age.
//
// recencyKey: which per-chunk date "recent" means — a GLOBAL vault-level
// definition set once in settings, never per-query. 'created' (default) =
// frontmatter `created`, copy-proof and matching the dated-instance semantics
// episodic queries want; falls back to mtime for un-stamped notes. 'modified'
// = file mtime as of indexing — edit-recency, for vaults that want it as an
// explicit choice (mtime is silently mass-corrupted by vault copies, iCloud
// sync, and bulk plugin edits, which is why it is not the default).

// A full YYYY-MM-DD anywhere in the basename: daily notes (`2026-06-11.md`)
// and dated series notes (`Alex 1x1 2026-05-19.md`). Deliberately requires the
// full 10-char dashed form — bare years and Jira-style `PROJ-2018` tokens are
// NOT creation dates (the false-positive class temporal.ts guarded against).
const FILENAME_DATE_RE = /(\d{4})-(\d{2})-(\d{2})/;

function filenameDate(notePath: string | undefined): string | null {
    if (!notePath) return null;
    const basename = notePath.split('/').pop() ?? '';
    const m = FILENAME_DATE_RE.exec(basename);
    if (!m) return null;
    const mo = +m[2], d = +m[3];
    if (mo < 1 || mo > 12 || d < 1 || d > 31) return null;
    return m[0];
}

// The minimal chunk shape every recency consumer can supply (full Chunks in the
// ranker/arm, the dedup-light rows on the browse path).
export interface RecencyChunk {
    note_path?: string;
    metadata?: {
        created?: string | null;
        modified?: string | null;
        properties?: Record<string, string>;
    };
}

// THE shared accessor: ranker scorer, the S1 recency candidate arm
// (search.ts topByRecency), and browseOrder below must all read the date
// through here so arm ⇄ scorer ⇄ browse can never disagree on what "recent"
// means (the 06-07 lesson: a key change applied to the scorer but not the arm
// admits the wrong candidates).
//
// The 'created' key resolves through a PORTABILITY LADDER — `created` is a
// vault convention, not an Obsidian builtin, so other vaults need fallbacks:
//   1. the configured created property (createdProp, default 'created') —
//      the canonical metadata.created field when default, else looked up in
//      metadata.properties (ALL scalar frontmatter is already indexed per
//      chunk, so a custom name is read-side: no chunker change, no reindex);
//   2. a YYYY-MM-DD in the filename — daily notes and dated series notes,
//      the zero-frontmatter convention most vaults already have;
//   3. mtime — the last resort (what Omnisearch keys everything on).
// 'modified' is mtime first, then the same ladder as its own fallback.
export function recencyDate(
    chunk: RecencyChunk | undefined,
    key: RecencyKeyChoice,
    createdProp = 'created',
): string | null {
    const meta = chunk?.metadata;
    const created = (createdProp === 'created'
        ? meta?.created
        : meta?.properties?.[createdProp] ?? meta?.created)
        ?? filenameDate(chunk?.note_path);
    if (key === 'modified') return meta?.modified ?? created ?? null;
    return created ?? meta?.modified ?? null;
}

export interface RecencyOptions {
    halfLifeDays?: number;
    referenceDateMs?: number;
}

export function parseDateMs(dateStr: string | null | undefined): number | null {
    if (!dateStr) return null;
    const cleaned = String(dateStr).trim().replace(/^["'{]+|["'}]+$/g, '').trim();
    if (!cleaned) return null;

    // Native Date.parse handles ISO 8601, "YYYY-MM-DD", and most variations.
    const direct = Date.parse(cleaned);
    if (!isNaN(direct)) return direct;

    // Fallback: pull just the YYYY-MM-DD prefix.
    const m = /(\d{4}-\d{2}-\d{2})/.exec(cleaned);
    if (m) {
        const ms = Date.parse(m[1]);
        if (!isNaN(ms)) return ms;
    }
    return null;
}

// Smooth exponential half-life decay in [0,1]: 1.0 today, 0.5 at halfLifeDays,
// 0.25 at 2×, never zero. HL=180d is the 06-04 study's operating point — wide
// enough that an 83-day-old episodic target (the MEDIAN) still carries ~0.73
// signal, instead of the flat 0 the old 30d cutoff gave it. Undated input → 0,
// which is NEUTRAL under the additive ε term (no boost, never a penalty).
export function computeRecencyScore(dateStr: string | null | undefined, opts: RecencyOptions = {}): number {
    const halfLifeDays = opts.halfLifeDays ?? 180;
    const referenceMs = opts.referenceDateMs ?? Date.now();

    const dtMs = parseDateMs(dateStr);
    if (dtMs == null) return 0.0;

    const daysOld = Math.max(0, (referenceMs - dtMs) / 86_400_000);
    if (!isFinite(daysOld)) return 0.0;

    return Math.pow(0.5, daysOld / halfLifeDays);
}

// TM2C2 — theoretical / fixed-endpoint normalization (Bruch et al., TOIS 2023,
// "An Analysis of Fusion Functions for Hybrid Retrieval"). Replaces per-query
// min-max, whose DATA-DEPENDENT denominator (the per-query max) is the actual
// root of the empty-note/OOV bug: when the dense channel has no opinion (an
// out-of-vocabulary query — opaque ID, keysmash, or an invented proper noun in
// an entity vault — where every cosine is bunched ~0.73–0.80), min-max divides
// by that tiny range and STRETCHES pure noise into a confident 1.00-vs-0 ranking
// that out-votes a real BM25 hit. Fixed endpoints leave a flat channel flat, so
// BM25 decides. The endpoints are MODEL-INDEPENDENT (no per-model/per-vault
// calibration), which is the whole point — see [[seek-empty-stub-dense-pollution]].
//
// Eval verdict (2026-06-09, ~/eval + ~/eval-oov OOV slice):
// the LESS the cosine is processed, the more bug-robust — per-query stretch is
// what manufactures false confidence, and calibrated fixed endpoints still
// stretch. The theoretical (cos+1)/2 form compresses instead: best aggregate
// (personal nDCG@10 0.8366@α0.90 vs min-max 0.8224@α0.50, +0.014), neutral on
// the entity slice, and fixes the manufactured-winner at EVERY alpha incl. 0.95,
// where min-max sinks the true match to rank 7-9. Constant-free and dominant —
// it subsumes the (now-deleted) absolute-cosine gate.

// Cosine [-1,1] -> [0,1] with the THEORETICAL endpoints (±1), clamped. Fixed, so
// a no-opinion dense channel stays near-flat and cannot manufacture a winner.
export function theoreticalNormDense(cos: Float64Array | Float32Array | number[]): Float64Array {
    const n = cos.length;
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        const v = (cos[i] + 1) / 2;
        out[i] = v < 0 ? 0 : v > 1 ? 1 : v;
    }
    return out;
}

// BM25 -> [0,1] by the per-query THEORETICAL upper bound (bm25.getQueryBound:
// Σ_t Σ_f boost_f·idf_f(t)·(k1+1+d)), completing TM2C2 on the lexical side.
//
// The previous form divided by the per-query EMPIRICAL max — the same
// data-dependent-denominator pathology TM2C2 removed from the dense side: the
// best match was forced to 1.0 whether it was a perfect multi-term title hit or
// one mediocre common term, so the channel's gain was amplified exactly when it
// was least trustworthy (the "saturated single-term hit" captures that drove
// the α 0.90→0.92 nudge). Under the bound, a weak-lexical query yields
// uniformly small bm25Norm — the channel self-attenuates — and the scale is
// query-invariant, so α means one thing everywhere. Validated 2026-06-09
// (~/eval/bound_norm_eval.py + ~/eval-oov): personal parity
// (bootstrap CI spans 0), OOV desc stratum +0.06–0.09, no manufactured-winner
// regression, and the personal/OOD α optima converge on one plateau (~0.7–0.85
// → shipped denseWeight 0.80, NOT comparable to the 0.92 empirical-max point).
//
// Two regimes need care (both validated on the D&D typo slice):
//   - Clip at 1: with fuzzy on, derived terms score with their own (often
//     higher) IDF, so a doc can exceed the exact-term bound. Ties at 1.0 are
//     arbitrated by dense — bounded, local degradation. Exact for fuzzy:false.
//   - bound==0 with nonzero scores (fully-OOV typo query under fuzzy, or
//     MiniSearch internals unavailable after an upgrade): the bound has no
//     opinion about this query, so DON'T let it manufacture silence — fall
//     back to the empirical max (rank-only normalization, the pre-bound
//     behavior). Without this, a misspelled entity query zeroes the only
//     channel that knows the answer (34/40→37/40 gold@1 on the typo slice).
// An all-zero channel still maps to all-0 (no spurious uniform vote).
export function theoreticalNormBm25(bm25: Float64Array | Float32Array | number[], bound?: number): Float64Array {
    const n = bm25.length;
    const out = new Float64Array(n);
    if (bound !== undefined && bound > 1e-10) {
        for (let i = 0; i < n; i++) {
            const v = bm25[i] / bound;
            out[i] = v > 1 ? 1 : v;
        }
        return out;
    }
    let mx = 0;
    for (let i = 0; i < n; i++) if (bm25[i] > mx) mx = bm25[i];
    if (mx < 1e-10) return out;
    for (let i = 0; i < n; i++) out[i] = bm25[i] / mx;
    return out;
}

// hybrid = alpha·dense_norm + (1-alpha)·bm25_norm over the TM2C2-normalized
// channels. alpha is the dense weight; note its scale is NOT comparable to the
// old min-max alpha — TM2C2 compresses the dense channel, so the tuned operating
// point sits higher (~0.90 vs ~0.50). See DEFAULT_RANKING_CONFIG.alpha.
export function hybridFusion(dense: Float64Array, bm25: Float64Array, alpha: number): Float64Array {
    const n = dense.length;
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        out[i] = alpha * dense[i] + (1 - alpha) * bm25[i];
    }
    return out;
}

// (weightedRrfFuse + ranksDescending deleted 2026-06-11: the opt-in RRF blend
// mode is gone — linear TM2C2 won every comparison (RRF best 0.8022 vs min-max
// linear 0.8085 on the 482-q eval, before TM2C2 then beat min-max), and a
// rank-only fusion discards the real cosine-magnitude signal. See [[Seek Rel]]
// §Rejected Approaches.)

// Title/alias known-item boost — token COVERAGE, not string equality.
//
// The original fix only fired when the query was EXACTLY a note's basename or
// alias. That misses the dominant known-item shape: the query is a SUBSET of
// the title, not equal to it — "alex 1x1" vs `Alex 1x1 2026-05-19`, "project
// atlas" vs `Atlas Project`, "graphdb" vs `GraphDB`. So we generalize:
//
//   fire only when EVERY query token appears in the title (or alias) token set
//   — "you typed a subset of the title", the known-item gate —
//   and scale the magnitude by PRECISION = |q∩t| / |t|, i.e. how much of the
//   title the query fills. A lone common word inside a long title gets a tiny
//   boost; a near-exact title gets the full `boost`. Exact match is the
//   precision==1 special case, so the old behavior is preserved at the top end.
//
// boost=0.8 is the swept default: on the 482-q old-log personal eval the `prec`
// form peaks at the knee ~0.8 (nDCG@10 0.8143→0.8677, every known-item stratum
// up, none down) and is flat above 1.0. Precision-scaling is self-limiting —
// only a near-exact title earns the full 0.8 — so a high magnitude is safe; an
// overwhelming dense+BM25 signal can still overtake a low-precision false hit.
// (Tuned by ~/eval/title_coverage.py; see [[Seek Rel]].)
//
// Tokenization: lowercase, then Unicode letter/number runs (/[\p{L}\p{N}]+/gu).
// Keeps "1x1" intact and explodes a dated title into year/month/day tokens —
// which is fine: the date tokens are matched by the date-proximity signal, not
// here. Was ASCII-only /[a-z0-9]+/g until 2026-06-10 (Three-Lens S3): every
// token of "Café Zürich" matched as empty, so the known-item boost was
// structurally DEAD for any non-ASCII title. The Unicode classes align this
// channel with MiniSearch's tokenizer, which was always Unicode-aware. (CJK
// remains out of scope as a decision — no segmentation anywhere in the stack.)
export interface TitleBoostChunk {
    note_path: string;
    metadata?: { aliases?: string[] };
}

function tokenSet(s: string, dropStopwords = false): Set<string> {
    // Depluralize each token with the SAME rule BM25 applies in processTerm, run
    // symmetrically over both the query and the title here — so a plural query
    // token matches a singular title token and vice-versa ("parks" → "park" now
    // fires the boost on a `… Park` title). Without this the nav boost silently
    // missed plurals that BM25 matched, an asymmetry between the two channels.
    //
    // dropStopwords (query side only, audit 2026-06-09 §6.1): the known-item gate
    // requires EVERY query token to appear in the title, so a single stopword the
    // user typed ("the atlas project") that the title omits killed the whole boost
    // — while BM25 strips that same stopword and matched fine. Dropping the
    // stoplist from the QUERY tokens realigns the two. Checked on the raw lowercased
    // token BEFORE depluralize, exactly as processTerm does (bm25.ts), and against
    // the SAME ENGLISH_STOPWORDS set so the analyzers can't drift apart again. The
    // TITLE side keeps stopwords (default false): we don't touch what's matchable
    // there — that's the BM25-channel name-as-stopword problem (§6.2, deferred).
    const out = new Set<string>();
    for (const m of s.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? []) {
        // CJK runs are dictionary-segmented with the SAME segmenter BM25
        // indexes with (tokenize.ts) — otherwise segmented query terms could
        // never satisfy the all-tokens-in-title gate against an unsegmented
        // CJK title set, and the boost would silently never fire on CJK. The
        // non-CJK branch folds diacritics with the SAME helper processTerm uses
        // (audit §4), so the title-boost keys the term space BM25 indexes
        // ("Café" boost fires for the query "cafe"); foldDiacritics self-guards
        // on hasCjk, but the branch only reaches it for non-CJK m anyway.
        for (const t of hasCjk(m) ? segmentCjkToken(m) : [foldDiacritics(m)]) {
            if (dropStopwords && ENGLISH_STOPWORDS.has(t)) continue;
            out.add(depluralize(t));
        }
    }
    return out;
}

// |query ∩ title| / |title| when ALL query tokens are in the title, else 0.
function coverage(qTokens: Set<string>, title: string): number {
    const t = tokenSet(title);
    if (t.size === 0) return 0;
    let inter = 0;
    for (const tok of qTokens) if (t.has(tok)) inter++;
    if (inter < qTokens.size) return 0; // require full query coverage (known-item gate)
    return inter / t.size;              // precision
}

export function titleMatchBoost(query: string, chunks: TitleBoostChunk[], boost = 0.8): Float64Array {
    const out = new Float64Array(chunks.length);
    // Gate on content tokens (stopwords dropped, §6.1). If the query is ALL
    // stopwords ("will", "the it"), fall back to the literal tokens so a note
    // literally titled "Will"/"It" still earns the boost — that name-as-stopword
    // class is the deferred §6.2 BM25-channel issue, not something to regress here.
    let qTokens = tokenSet(query, true);
    if (qTokens.size === 0) qTokens = tokenSet(query, false);
    if (qTokens.size === 0) return out;

    for (let i = 0; i < chunks.length; i++) {
        const notePath = chunks[i].note_path ?? '';
        let basename = notePath.split('/').pop() ?? '';
        if (basename.endsWith('.md')) basename = basename.slice(0, -3);

        let best = coverage(qTokens, basename);

        const aliases = chunks[i].metadata?.aliases;
        if (best < 1 && Array.isArray(aliases)) {
            for (const alias of aliases) {
                best = Math.max(best, coverage(qTokens, String(alias)));
                if (best >= 1) break;
            }
        }
        out[i] = boost * best;
    }
    return out;
}

// ---- Filter-only browse ordering ------------------------------------------
//
// Ordering for the filter-only fast path (search.ts): a query that is all
// operators (`#meetings`, `path:X/*`) has no text for dense/BM25, so there is
// no relevance signal — it's a BROWSE, and its order is an explicit
// presentation sort, never the fusion formula. (Routing it through rank()
// with zeroed score arrays coupled the ordering to the recency blend weight:
// any rw>0 was already pure recency order — the constant term can't reorder —
// and rw=0, the parked default 2026-06-08..11, collapsed to index frame order,
// an incidental mtime-as-of-last-reindex ordering that degrades as
// incremental deltas append. Audit 2026-06-09 §1.)
//
// Sort contract — invariant to ALL ranking config by construction:
//   1. the vault's recencyKey date (recencyDate accessor above) descending —
//      the same key the recency candidate arm and the ε-tiebreaker use;
//   2. undated chunks LAST, never dropped — they matched the filter, and the
//      matched set IS the result set here (unlike the recency arm, which may
//      skip undated chunks because it is additive coverage);
//   3. note_path, then chunk_id, ascending as the final tie-break — `created`
//      is date-granular frontmatter, so same-day ties are common (daily
//      notes) and an input-order tie-break would reintroduce frame order
//      within each day.

export interface BrowseChunk extends RecencyChunk {
    chunk_id?: string;
}

export function browseOrder<T extends BrowseChunk>(chunks: T[], key: RecencyKeyChoice = 'created', createdProp = 'created'): T[] {
    const keyed = chunks.map(c => ({
        c,
        ms: parseDateMs(recencyDate(c, key, createdProp)),
    }));
    keyed.sort((a, b) => {
        if (a.ms !== b.ms) {
            if (a.ms === null) return 1;
            if (b.ms === null) return -1;
            return b.ms - a.ms;
        }
        const byPath = (a.c.note_path ?? '').localeCompare(b.c.note_path ?? '');
        if (byPath !== 0) return byPath;
        return (a.c.chunk_id ?? '').localeCompare(b.c.chunk_id ?? '');
    });
    return keyed.map(k => k.c);
}
