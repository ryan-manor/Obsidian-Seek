// Shared lexical tokenization — the CJK 80/20 (2026-06-10).
//
// MiniSearch's default tokenizer splits on whitespace + Unicode punctuation
// only, so an unsegmented CJK sentence (no word boundaries in zh/ja; Korean
// eojeol carry attached particles) indexes as ONE giant token: BM25-only on
// Belebele scored ja 0.064 / zh 0.112 vs en 0.812 ([[Seek MultiLanguage]]) —
// the lexical channel is effectively dead, and it drags fused below dense.
//
// Fix: after the MiniSearch-mirroring space/punct split, any token containing
// CJK script characters is further segmented with Intl.Segmenter (granularity
// 'word'). ICU's word-break iterator segments Han/Kana by DICTIONARY,
// independent of the locale argument, so one segmenter handles zh+ja (+ko,
// where Hangul mostly rides the existing space split). This is deliberately
// NOT per-language analysis — no language detection, no stemmers, no
// per-language stoplists; Latin-script tokens pass through byte-identical,
// so English behavior (and every English eval number) is unchanged by
// construction.
//
// Used by: bm25.ts (MiniSearch `tokenize` option — index AND query side —
// plus the coverage denominator + theoretical bound, which must enumerate the
// same terms search() scores with) and fusion.ts tokenSet (title boost; if it
// split differently, segmented query tokens could never satisfy the all-in-
// title gate on CJK titles).
//
// Fallback: if Intl.Segmenter is unavailable (very old WebView), tokens pass
// through unsegmented — exactly today's behavior, a graceful degrade.

// The delimiter char-class BODY (the chars only — no brackets, no quantifier),
// kept as ONE constant so the split regex and the capture-scan below are built
// from the same source and can never drift. The byte-identity between them is
// load-bearing (coverage denominator + theoretical bound in bm25.ts enumerate
// the SAME tokens search() scores with). Mirrors MiniSearch's SPACE_OR_PUNCTUATION
// (verified dist v7.2.0); if a MiniSearch upgrade changes its default, change
// this one constant — see the QUERY_SPLIT note in bm25.ts.
const DELIM_CLASS = '\\n\\r\\p{Z}\\p{P}';
const SPACE_OR_PUNCT = new RegExp(`[${DELIM_CLASS}]+`, 'u');

// Possessive 's, stripped from the RAW text BEFORE tokenizing (must be here, not
// in processTerm: the split shatters "ryan's" into ["ryan","s"], leaving a junk
// bare "s" — a near-stopword the stoplist does NOT carry and depluralize() won't
// drop, len<=3 short-circuit — that survives as a real indexed AND query term,
// and with the glue-join below also manufactures a junk "ryans"). Anchored on a
// LETTER + optional combining marks (so "josé's"→"josé" but "1990's" KEEPS 1990,
// a decade not a possessive) plus a right word-boundary lookahead, so only a true
// possessive clitic goes, never a word that merely ends in "s". Both straight '
// (U+0027) and curly ’ (U+2019); U+02BC (ʼ, a letter-modifier) is excluded — it
// never splits and rides transliteration. Contractions ("don't"→don,t) are left
// alone: that clitic is "'t", not "'s". seekTokenize is the shared index+query
// tokenizer, so this is symmetric by construction. (2026-06-18, [[Seek Search
// Practice Audit 2026-06-18]] item 7.)
const POSSESSIVE_S = /(\p{L}\p{M}*)['’]s(?=$|[^\p{L}\p{N}])/giu;

// Intra-word "glue" punctuation: when a run of ONLY these characters separates
// two non-empty Latin fragments, they are one compound identifier — hyphen/dash
// family (\p{Pd}: "co-pilot", "granite-r2"), period ("v2.0", "U.S.A"), forward
// slash ("TCP/IP"), connector punctuation (\p{Pc}: "my_var"), and the apostrophe
// family ("o'brien"→"obrien"). A DELIBERATELY curated subset of \p{P}, NOT all of
// it: commas/brackets/quotes are real word boundaries, so joining across them
// would manufacture noise tokens. The split itself is unchanged (still
// SPACE_OR_PUNCT); this only governs the ADDITIVE joined form below (2026-06-18,
// [[Seek Search Practice Audit 2026-06-18]] item 6: "gpt4" could not reach a doc
// that wrote "GPT-4").
const GLUE_RUN = /^[\p{Pd}._/\p{Pc}'’]+$/u;

// Single-pass scan capturing each token (group 1 = the DELIM_CLASS complement)
// together with the delimiter run that follows it (group 2), so we can tell a
// glue-joined compound ("gpt-4") from a whitespace/other-punct boundary
// ("gpt 4", "gpt,4"). Built from the SAME DELIM_CLASS as SPACE_OR_PUNCT, so
// [...text.matchAll(SCAN)].map(m=>m[1]) and text.split(SPACE_OR_PUNCT).filter(Boolean)
// enumerate byte-identical tokens (the two char classes are exact complements).
const SCAN = new RegExp(`([^${DELIM_CLASS}]+)([${DELIM_CLASS}]*)`, 'gu');

// Scripts that need dictionary segmentation. Property escapes cover the full
// blocks (incl. CJK extensions beyond the BMP) without a range zoo.
const CJK_RE = /[\p{sc=Han}\p{sc=Hiragana}\p{sc=Katakana}\p{sc=Hangul}]/u;

// One lazily-built segmenter, reused across calls (construction is the
// expensive part — it loads the ICU dictionary).
let segmenter: Intl.Segmenter | null | undefined;
function getSegmenter(): Intl.Segmenter | null {
    if (segmenter !== undefined) return segmenter;
    try {
        segmenter = typeof Intl !== 'undefined' && typeof Intl.Segmenter === 'function'
            ? new Intl.Segmenter(undefined, { granularity: 'word' })
            : null;
    } catch {
        segmenter = null;
    }
    return segmenter;
}

export function hasCjk(s: string): boolean {
    return CJK_RE.test(s);
}

// Segment one space/punct-free token that contains CJK characters into
// word-like pieces. Non-word segments (stray symbols ICU classifies as
// non-words) are dropped — same as the punctuation split drops them for
// Latin text. Returns the token unchanged when no segmenter exists.
export function segmentCjkToken(token: string): string[] {
    const seg = getSegmenter();
    if (!seg) return [token];
    const out: string[] = [];
    for (const part of seg.segment(token)) {
        if (part.isWordLike) out.push(part.segment);
    }
    // ICU can classify an entire exotic token as non-word-like; keep the
    // original rather than silently deleting it from the index.
    return out.length > 0 ? out : [token];
}

// The plugin-wide tokenizer: strip possessive 's, MiniSearch-default split, then
// CJK-aware segmentation per token, PLUS an additive punctuation-joined compound
// form.
//
// The canonical tokens are byte-identical to the old SPACE_OR_PUNCT split (the
// SCAN fragment class is its exact char-class complement), so the MiniSearch
// mirror / coverage-denominator / theoretical-bound contracts are untouched —
// the join only ADDS tokens, never removes or replaces a split token. The
// joined form is produced identically on index AND query side (one shared
// function), so query "gpt4" and a doc's "GPT-4" meet on the joined token
// "gpt4" (after processTerm lowercases both). See GLUE_RUN above.
//
// Composition rules:
//   - a joined token is emitted only for a run of ≥2 Latin fragments separated
//     by ONLY glue punctuation (one join per compound; ALL glue stripped, never
//     pairwise — bounds inflation to +1 token per compound);
//   - CJK fragments never join (the segmenter owns them) — a glued token that
//     contains CJK flushes the Latin run first, so "東京-tokyo" is never
//     re-merged into the segmented pieces;
//   - whitespace or any non-glue punctuation (comma, bracket, quote) is a real
//     boundary and flushes the run, so "a,b" and "a b" never join.
export function seekTokenize(text: string): string[] {
    const out: string[] = [];
    // Possessive 's removed up-front (item 7) so neither the canonical split nor
    // the glue-join below ever sees the junk "s" / "ryans" forms. See POSSESSIVE_S.
    const scanText = text.replace(POSSESSIVE_S, '$1');
    // Accumulates the current run of Latin fragments joined only by glue punct;
    // flushed (emitting the concatenation) at any non-glue boundary.
    let joinBuf: string[] = [];
    const flushJoin = () => {
        if (joinBuf.length >= 2) out.push(joinBuf.join(''));
        joinBuf = [];
    };
    for (const m of scanText.matchAll(SCAN)) {
        const frag = m[1];
        const delim = m[2];
        if (hasCjk(frag)) {
            flushJoin();                          // CJK never joins a Latin run
            for (const piece of segmentCjkToken(frag)) out.push(piece);
        } else {
            out.push(frag);                       // canonical token (unchanged)
            joinBuf.push(frag);
        }
        // Continue the compound only across a pure-glue delimiter; a whitespace
        // or other-punct delimiter (or end of text) ends it. (After a CJK
        // fragment joinBuf is already empty, so this is a no-op there.)
        if (delim === '' || !GLUE_RUN.test(delim)) flushJoin();
    }
    flushJoin();
    return out;
}
