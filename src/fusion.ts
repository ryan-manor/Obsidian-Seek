// Score fusion, recency, and title-match boost.
// Direct port of python-backend/app/search/fusion.py and the title-boost helper
// in ranker.py. Coefficients match the live Python backend's defaults.

import { depluralize, ENGLISH_STOPWORDS, foldDiacritics, prefixLastTokenFor } from './bm25';
import { hasCjk, segmentCjkToken, seekTokenize } from './tokenize';
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
// episodic queries want; un-stamped notes have no date (see recencyDate). 'modified'
// = file mtime as of indexing — edit-recency, for vaults that want it as an
// explicit choice (mtime is silently mass-corrupted by vault copies, iCloud
// sync, and bulk plugin edits, which is why it is not the default).

// The minimal chunk shape every recency consumer can supply (full Chunks in the
// ranker/arm, the dedup-light rows on the browse path).
export interface RecencyChunk {
    note_path?: string;
    metadata?: {
        created?: string | null;
        modified?: string | null;
        // string[] since v10 (list-valued props stored for the filter matcher);
        // date reads take the FIRST element of a (pathological) list value.
        properties?: Record<string, string | string[]>;
    };
}

// THE shared accessor: ranker scorer, the S1 recency candidate arm
// (search.ts topByRecency), and browseOrder below must all read the date
// through here so arm ⇄ scorer ⇄ browse can never disagree on what "recent"
// means (the 06-07 lesson: a key change applied to the scorer but not the arm
// admits the wrong candidates).
//
// STRICT since 2026-11-11: the selected date field is canonical, everywhere.
//   'modified' → file mtime as of indexing.
//   'created'  → the configured property (createdProp; the canonical
//                metadata.created field when 'created', else looked up in
//                metadata.properties — read-side, no reindex) — and NOTHING else.
// A note missing the selected field has no date: 0 recency in the ranker, not
// admitted by the recency arm, sorted after dated notes in browse, rejected by
// before:/after:. The old portability ladder (custom prop → `created` →
// filename YYYY-MM-DD → mtime) is gone: the mtime step handed un-stamped notes
// a fake "recent" date whenever a move/link-rewrite/sync touched the file
// (Notes/Ofawa.md, 2026-11-11: +0.097 at ε0.1·90d from a folder-move link
// rewrite), and the other steps silently substituted a field the user didn't
// pick. One resolver, so ranker ⇄ arm ⇄ browse ⇄ filters cannot disagree.
export function recencyDate(
    chunk: RecencyChunk | undefined,
    key: RecencyKeyChoice,
    createdProp = 'created',
): string | null {
    const meta = chunk?.metadata;
    if (key === 'modified') return meta?.modified ?? null;
    if (createdProp === 'created') return meta?.created ?? null;
    const raw = meta?.properties?.[createdProp];
    return (Array.isArray(raw) ? raw[0] : raw) ?? null;
}

export interface RecencyOptions {
    halfLifeDays?: number;
    referenceDateMs?: number;
}

// A bare "YYYY-MM-DD" (no time-of-day) → LOCAL midnight, not UTC. Matches the
// convention already used for displaying created dates (search-modal.ts
// fmtCreated): native Date.parse treats a date-ONLY ISO string as UTC
// midnight, which rolls the calendar day back a day for anyone west of UTC.
// A full timestamp (with or without an explicit offset) is unaffected — it
// already falls through to native Date.parse, which resolves those as local
// time (no offset) or the given offset, not UTC.
function localDateOnlyMs(y: string, mo: string, d: string): number | null {
    // The numeric-args Date constructor NORMALIZES out-of-range fields instead
    // of rejecting them (month 13 rolls into next January, day 32 rolls into
    // the next month) — unlike the old Date.parse-based code, which validated
    // month∈[1,12] and day∈[1,31] syntactically before falling to calendar
    // arithmetic for in-range-but-nonexistent combinations (e.g. Feb 30 →
    // rolls to Mar 2, unchanged here). Reject the syntactically-invalid case
    // so a typo'd month/day (e.g. `created: 2026-13-01`) degrades to null —
    // as documented on parseDateMs's callers — instead of silently landing on
    // a wrong date.
    const moNum = Number(mo), dNum = Number(d);
    if (moNum < 1 || moNum > 12 || dNum < 1 || dNum > 31) return null;
    const dt = new Date(Number(y), moNum - 1, dNum);
    return isNaN(dt.getTime()) ? null : dt.getTime();
}

export function parseDateMs(dateStr: string | null | undefined): number | null {
    if (!dateStr) return null;
    const cleaned = String(dateStr).trim().replace(/^["'{]+|["'}]+$/g, '').trim();
    if (!cleaned) return null;

    const dateOnly = /^(\d{4})-(\d{2})-(\d{2})$/.exec(cleaned);
    if (dateOnly) return localDateOnlyMs(dateOnly[1], dateOnly[2], dateOnly[3]);

    // Native Date.parse handles ISO 8601 timestamps (local or offset-bearing)
    // and most other variations; only the bare-date case above needs the
    // local override.
    const direct = Date.parse(cleaned);
    if (!isNaN(direct)) return direct;

    // Fallback: pull just the YYYY-MM-DD prefix, same local-midnight treatment.
    const m = /(\d{4})-(\d{2})-(\d{2})/.exec(cleaned);
    if (m) return localDateOnlyMs(m[1], m[2], m[3]);
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
// Eval verdict (2026-06-09, ~/seek-personal-eval + ~/seek-dnd-eval OOV slice):
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
// Σ_t UB(t), UB(t) = the max single-term score of t over the live index — under
// BM25F (the default) the max BM25F score, under the legacy per-field sum the max
// boosted per-field-sum score; the analytic Σ_t Σ_f boost_f·idf_f(t)·(k1+1+d) form
// this comment used to cite was retired 2026-06-26), completing TM2C2 on the
// lexical side.
//
// The previous form divided by the per-query EMPIRICAL max — the same
// data-dependent-denominator pathology TM2C2 removed from the dense side: the
// best match was forced to 1.0 whether it was a perfect multi-term title hit or
// one mediocre common term, so the channel's gain was amplified exactly when it
// was least trustworthy (the "saturated single-term hit" captures that drove
// the α 0.90→0.92 nudge). Under the bound, a weak-lexical query yields
// uniformly small bm25Norm — the channel self-attenuates — and the scale is
// query-invariant, so α means one thing everywhere. Validated 2026-06-09
// (~/seek-personal-eval/bound_norm_eval.py + ~/seek-dnd-eval): personal parity
// (bootstrap CI spans 0), OOV desc stratum +0.06–0.09, no manufactured-winner
// regression, and the personal/OOD α optima converge on one plateau (~0.7–0.85
// → shipped denseWeight 0.80, NOT comparable to the 0.92 empirical-max point).
//
// Two regimes need care (both validated on the D&D typo slice):
//   - Clip at 1: with fuzzy on, derived terms score with their own (often
//     higher) IDF, so a doc can exceed the exact-term bound. Ties at 1.0 are
//     arbitrated by dense — bounded, local degradation. Exact for fuzzy:false.
//     Since 2026-11-12 an OOV typo term that fuzzy CAN reach is bounded by its
//     best derived term (bm25.ts queryBoundAndTerms), so a mixed typo query
//     ("fy27 plannign") no longer clips its whole head to 1.0. A prefix-eligible
//     last token with prefix extensions is excluded from that, so typeahead keeps
//     the fallback below.
//   - bound==0 with nonzero scores (OOV typo fuzzy can't reach; OR a bare
//     single-token PREFIX still mid-type in typeahead — "amster" before
//     "amsterdam" is an indexed term — since the bound does not model prefix
//     extensions; OR MiniSearch internals unavailable after an upgrade):
//     the bound has no opinion about this query, so DON'T let it manufacture
//     silence — fall back to the empirical max (rank-only normalization, the
//     pre-bound behavior). The typeahead-prefix case is INTENDED, not a TM2C2
//     regression (audit 2026-06-29 #5): a half-typed prefix has no stable bound
//     to honor, so the pre-bound /max floor is the right behavior for a query
//     that isn't settled yet — the manufactured-1.0-winner TM2C2 fixes only
//     matters for committed queries. Without the fallback, a misspelled entity
//     query zeroes the only channel that knows the answer (34/40→37/40 gold@1
//     on the typo slice).
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
// eames" vs `Eames Project`, "memgraph" vs `MemGraph`. So we generalize:
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
// (Tuned by ~/seek-personal-eval/title_coverage.py; see [[Seek Rel]].)
//
// Tokenization: seekTokenize — the SAME analyzer BM25 indexes/queries with — in
// its CANONICAL stream (derived:false): possessive-strip ("ryan's"→"ryan") and
// CJK dictionary segmentation, but WITHOUT the additive glue/camelCase recall
// forms. Then per token: lowercase, fold diacritics, drop query stopwords,
// depluralize (the processTerm steps). This retires the ad-hoc THIRD tokenizer
// this channel used until 2026-06-29 (a bare /[\p{L}\p{N}]+/ run with no
// possessive-strip), under which "ryan's" carried a stray "s" that failed the
// all-in-title gate on a `Ryan` title even though BM25 stripped it and matched.
//
// derived:FALSE is deliberate and eval-gated (audit 2026-06-29 #4,
// nav_tokenizer_arm.py over the 483-q personal + 127-q dnd sets) for the
// coverage DENOMINATOR |t|: the glue/camel forms exist to widen match
// RECALL, but in a precision denominator they are noise (a dated title
// `2026-05-19` would gain its concatenation `20260519`, shrinking precision
// on exactly the dated known-item titles this boost targets). Measured:
// derived:false denominator is a strict no-op (+0.0000), derived:true
// denominator −0.0002. NOTE: the BM25 bound and coverage enumerators do NOT
// share this choice — bm25.ts distinctQueryTerms / getQueryBound tokenize with
// seekTokenize's DEFAULT derived:true, so a glue/camel recall form is its own
// query term there (counted in the coverage denominator T and bounded by its
// own UB). An earlier version of this comment claimed they take derived:false;
// they never did (2026-11-11 investigation). Collapsing derived forms into one
// slot per canonical token is a separate, open cleanup (measured ±0.006).
//
// The GATE/MEMBERSHIP test is a separate question (audit R2 2026-07-02 #2)
// and DOES use the derived stream on the title side: a query typed as one
// glued token ("gpt4") must still satisfy the gate against a hyphenated
// title ("GPT-4"), the same morphological-variant recall BM25 already gets
// via seekTokenize's default derived:true. So coverage() below
// checks membership against a DERIVED title-token set while still dividing
// by the CANONICAL |t| — loosens the gate, leaves the precision denominator
// (and therefore the eval-gated magnitude) untouched. Dated titles
// still explode into y/m/d tokens (matched by date-proximity, not here); "1x1"
// stays intact. (Was ASCII-only /[a-z0-9]+/g until 2026-06-10, which matched
// every token of "Café Zürich" as empty — non-ASCII titles couldn't fire at all.)
export interface TitleBoostChunk {
    note_path: string;
    metadata?: { aliases?: string[] };
}

// `derived` (default false) selects the CANONICAL vs derived seekTokenize
// stream — see the derived:FALSE-for-the-denominator comment above. Both the
// query and (for the coverage DENOMINATOR) the title call this with the
// default; coverage() below separately builds a derived:true set for the
// title to use ONLY as the membership gate.
function tokenSet(s: string, dropStopwords = false, derived = false): Set<string> {
    // Depluralize each token with the SAME rule BM25 applies in processTerm, run
    // symmetrically over both the query and the title here — so a plural query
    // token matches a singular title token and vice-versa ("parks" → "park" now
    // fires the boost on a `… Park` title). Without this the nav boost silently
    // missed plurals that BM25 matched, an asymmetry between the two channels.
    //
    // dropStopwords (query side only, audit 2026-06-09 §6.1): the known-item gate
    // requires EVERY query token to appear in the title, so a single stopword the
    // user typed ("the eames project") that the title omits killed the whole boost
    // — while BM25 strips that same stopword and matched fine. Dropping the
    // stoplist from the QUERY tokens realigns the two. Checked on the raw lowercased
    // token BEFORE depluralize, exactly as processTerm does (bm25.ts), and against
    // the SAME ENGLISH_STOPWORDS set so the analyzers can't drift apart again. The
    // TITLE side keeps stopwords (default false): we don't touch what's matchable
    // there — that's the BM25-channel name-as-stopword problem (§6.2, deferred).
    const out = new Set<string>();
    for (const raw of seekTokenize(s, { derived })) {
        const m = raw.toLowerCase();   // seekTokenize preserves case (camel split needs it)
        // seekTokenize already CJK-segments; the per-token branch re-applies the
        // SAME segmenter (idempotent on a single CJK piece) and, for non-CJK,
        // folds diacritics with the SAME helper processTerm uses (audit §4), so
        // the title-boost keys the exact term space BM25 indexes ("Café" boost
        // fires for the query "cafe"). foldDiacritics self-guards on hasCjk.
        for (const t of hasCjk(m) ? segmentCjkToken(m) : [foldDiacritics(m)]) {
            if (dropStopwords && ENGLISH_STOPWORDS.has(t)) continue;
            out.add(depluralize(t));
        }
    }
    return out;
}

// Title-side prefix matching (2026-10-13, made fixed behavior 2026-11-11): the
// FINAL query token may strict-prefix-match a title token from 1 char, so a
// typeahead query ("andrei g", "evolving infini") keeps its title boost while
// typing instead of dropping to 0 until the word is 3 chars long. Earlier tokens
// must still match exactly, and a lone token under 3 chars never prefix-matches
// (bm25.ts prefixLastTokenFor). Industry parity: Algolia prefixLast (prefix from
// the first char), Typesense/Meilisearch last-word prefix.
export const TITLE_PREFIX_MIN_CHARS = 1;
// A prefix hit earns PARTIAL credit, weighted the way MiniSearch weights a prefix
// hit in the BM25 channel (minisearch 7.2 index.js: prefixWeight · L/(L + 0.3·d),
// d = completion length − typed length; default prefixWeight 0.375). It still
// satisfies the all-tokens gate, but adds < 1 to the precision numerator, so an
// exact title word outranks a prefix one ("… ai" prefers an `AI` title over `Aim`),
// and a longer unfinished completion earns a little less. Industry ordering
// (Algolia Exact, Meilisearch exactness, Typesense prioritize_exact_match) tiers
// exact over prefix; Seek's signals sum rather than tier, so this mirrors the
// additive scheme our own lexical channel already uses.
//
// Why 0.375 not the earlier flat 0.5 (2026-11-11): the |title| denominator let a
// one-word prefix title beat a longer EXACT title — "car" put `Carboni's`
// (0.5/1) over `Cars and bikes` (1/3). At 0.375·7/8.2 = 0.32 the exact 3-word
// title wins. Margin is thin by design: this is a nudge, not a tier.
export const TITLE_PREFIX_CREDIT = 0.375;
export function titlePrefixCredit(typedLen: number, completionLen: number): number {
    const d = Math.max(0, completionLen - typedLen);
    return TITLE_PREFIX_CREDIT * completionLen / (completionLen + 0.3 * d);
}
const titlePrefixEligible = prefixLastTokenFor(TITLE_PREFIX_MIN_CHARS);

// Title-side fuzzy matching (2026-11-12): a query token that is a TYPO may match a
// title token within the edit budget the BM25 channel already used to reach it, so
// one misspelling ("fy27 plannign") no longer zeroes the known-item bonus for every
// note while the lexical channel happily matched "planning". The caller decides
// eligibility per token (TitleFuzzy): only tokens OOV in the lexical index, with the
// FUZZY_BY_LENGTH budget, so a real word ("plan") never fuzzes onto "play". Credit
// mirrors MiniSearch's fuzzy weight the way the prefix credit mirrors its prefix
// weight: 0.45·L/(L+d), L = title token length. It passes the all-tokens gate but
// adds < 1 to the precision numerator, so the clean spelling still outranks it.
export const TITLE_FUZZY_CREDIT = 0.45;
export function titleFuzzyCredit(titleTokLen: number, dist: number): number {
    return TITLE_FUZZY_CREDIT * titleTokLen / (titleTokLen + dist);
}
// Max edit distance a query token may fuzz by against title tokens; 0 = exact only.
export type TitleFuzzy = (tok: string) => number;

// Levenshtein distance when ≤ maxDist, else maxDist + 1 (banded, early exit).
// Plain Levenshtein, not Damerau, to match MiniSearch's fuzzyGet: the title gate
// accepts exactly what the BM25 channel could have matched.
export function editDistanceWithin(a: string, b: string, maxDist: number): number {
    const la = a.length, lb = b.length;
    if (Math.abs(la - lb) > maxDist) return maxDist + 1;
    let prev = new Array<number>(lb + 1);
    let cur = new Array<number>(lb + 1);
    for (let j = 0; j <= lb; j++) prev[j] = j;
    for (let i = 1; i <= la; i++) {
        cur[0] = i;
        let rowMin = i;
        const ca = a.charCodeAt(i - 1);
        for (let j = 1; j <= lb; j++) {
            const cost = ca === b.charCodeAt(j - 1) ? 0 : 1;
            const v = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost);
            cur[j] = v;
            if (v < rowMin) rowMin = v;
        }
        if (rowMin > maxDist) return maxDist + 1;
        const t = prev; prev = cur; cur = t;
    }
    return prev[lb] <= maxDist ? prev[lb] : maxDist + 1;
}

// Σcredit / |title| when ALL query tokens are in the title (exact, the final
// token as a prefix, or a typo token within its fuzzy budget), else 0. Exact hits
// credit 1, prefix hits titlePrefixCredit(), fuzzy hits titleFuzzyCredit().
function coverage(
    qTokens: Set<string>,
    title: string,
    prefixTok: string | null = null,
    fuzzyDist: Map<string, number> | null = null,
): number {
    const t = tokenSet(title);                // canonical — the precision DENOMINATOR
    if (t.size === 0) return 0;
    // The gate accepts the title's derived forms too (glue-joined compounds,
    // camelCase splits — tokenize.ts) so "gpt4" reaches a "GPT-4" title, the
    // same morphological-variant recall BM25 already has. |t| above stays
    // canonical, so this can't dilute precision on titles that don't need it
    // (audit R2 2026-07-02 #2).
    const matchable = tokenSet(title, false, true);
    let inter = 0;   // tokens matched — the gate
    let credit = 0;  // precision numerator — exact 1, prefix partial
    for (const tok of qTokens) {
        if (matchable.has(tok)) { inter++; credit++; continue; }
        let best = 0;
        if (tok === prefixTok) {
            // Best (shortest) completion among the title's tokens.
            for (const m of matchable) {
                if (m.startsWith(tok)) best = Math.max(best, titlePrefixCredit(tok.length, m.length));
            }
        }
        const maxDist = best > 0 ? 0 : (fuzzyDist?.get(tok) ?? 0);
        if (maxDist > 0) {
            for (const m of matchable) {
                const d = editDistanceWithin(tok, m, maxDist);
                if (d >= 1 && d <= maxDist) best = Math.max(best, titleFuzzyCredit(m.length, d));
            }
        }
        if (best > 0) { inter++; credit += best; }
    }
    if (inter < qTokens.size) return 0; // require full query coverage (known-item gate)
    // Clamped: a single derived match (e.g. "gpt4") can cover more canonical
    // title tokens than it counts toward `inter`, which would otherwise let
    // several such matches push the ratio above 1.
    return Math.min(1, credit / t.size); // precision
}

// `prefix` = false is exact-only (tests / harness probes); production always prefixes.
export function titleMatchBoost(
    query: string,
    chunks: TitleBoostChunk[],
    boost = 0.8,
    prefix = true,
    fuzzy?: TitleFuzzy,
): Float64Array {
    const out = new Float64Array(chunks.length);
    // Gate on content tokens (stopwords dropped, §6.1). If the query is ALL
    // stopwords ("will", "the it"), fall back to the literal tokens so a note
    // literally titled "Will"/"It" still earns the boost — that name-as-stopword
    // class is the deferred §6.2 BM25-channel issue, not something to regress here.
    let qTokens = tokenSet(query, true);
    if (qTokens.size === 0) qTokens = tokenSet(query, false);
    if (qTokens.size === 0) return out;

    // Independent of BM25's prefixLastToken: the title floor (1) sits below
    // BM25's (3) because only title tokens are scanned — no vocab fan-out.
    const qList = [...qTokens];
    const prefixTok = prefix && titlePrefixEligible(qList[qList.length - 1], qList.length - 1, qList)
        ? qList[qList.length - 1]
        : null;

    // Per-query fuzzy budgets, resolved once (the caller's predicate may consult
    // the lexical index). null when fuzzy is off or no token is eligible.
    let fuzzyDist: Map<string, number> | null = null;
    if (fuzzy) {
        for (const tok of qList) {
            const d = fuzzy(tok);
            if (d > 0) (fuzzyDist ??= new Map()).set(tok, d);
        }
    }

    for (let i = 0; i < chunks.length; i++) {
        const notePath = chunks[i].note_path ?? '';
        let basename = notePath.split('/').pop() ?? '';
        if (basename.endsWith('.md')) basename = basename.slice(0, -3);

        let best = coverage(qTokens, basename, prefixTok, fuzzyDist);

        const aliases = chunks[i].metadata?.aliases;
        if (best < 1 && Array.isArray(aliases)) {
            for (const alias of aliases) {
                best = Math.max(best, coverage(qTokens, String(alias), prefixTok, fuzzyDist));
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
