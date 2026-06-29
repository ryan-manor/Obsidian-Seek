// Query-side alias-dictionary synonym expansion (v1, default OFF).
//
// Motivation (eval pack 2026-06-10, alias_expansion.py + prefix_arm.py): a
// note's frontmatter aliases declare vault-specific equivalences ("Lr" ↔
// "Lightroom") that only help the alias OWNER — doc-side alias indexing can
// never lexically reach a note that spells the canonical form out. Expanding
// the QUERY with alias classmates closes that gap: personal 483q +0.0015 bin
// nDCG@10 over the shipped prefix baseline at w=0.8 (triggered-subset +0.017
// pre-prefix), D&D/code stress sets clean (vacuously — their vaults define
// almost no aliases, which is itself the safety property: no dictionary, no
// behavior change).
//
// The dictionary is per-note but index postings are GLOBAL — every guard here
// exists because of that scope mismatch (adversarial review 2026-06-10, vault
// note "Seek Synonym Expansion Plan"):
//   - strict trigger guard: a token in >1 class never expands (4 person pages
//     aliased "rohit" → querying "rohit" must not guess which one).
//   - SYMMETRIC mate guard: an ambiguous token is never INJECTED either.
//     Without it the trigger guard leaves the reverse direction open —
//     "rsharma" → mate "rohit" would score every Rohit in the vault. Measured
//     exactly free on the personal eval (32→30 triggers, no metric motion);
//     only binds in vaults where the collision actually exists.
//   - df ceiling: ambiguity is only visible INSIDE the dictionary. A junk
//     common-word alias ("index" on one note) is unambiguous by the class
//     check yet becomes a vault-wide query-time landmine. Refuse any trigger
//     or mate matching more than SYNONYM_DF_CEILING of all chunks.
//
// English-only posture: class members tokenize through the same ASCII-leaning
// stoplist + depluralizer pipeline as queries (bm25.ts processTerm); the
// feature is untested on non-English vaults by construction.

import type { ChunkMeta } from './chunker';
import { extractNoteName, processQueryTerm } from './bm25';
import { seekTokenize } from './tokenize';

// Mate score discount, mirroring the harness arm (w=0.8 swept best on the
// personal eval; cf. MiniSearch's own fuzzy weight 0.45 / prefix 0.375).
// NOTE: the harness measured SOURCE-attributed scoring; the plugin's native
// MiniSearch path double-credits mates in the ×quality multiplier (see
// bm25.ts getScoresWithCoverage), so the gate requires a native-semantics
// re-sweep before default-ON (plan note, gate corrections).
export const SYNONYM_WEIGHT = 0.8;

// df sanity ceiling (review fix #3): a trigger or mate token matching more
// than this fraction of chunks is dropped at build time. 5% is deliberately
// loose — real alias tokens (names, acronyms, products) are rare terms; only
// genuine junk ("index", "notes") gets near it.
export const SYNONYM_DF_CEILING = 0.05;

export interface SynonymStats {
    classes: number;
    triggers: number;
    droppedAmbiguous: number;   // tokens removed as triggers by the strict guard
    droppedDf: number;          // tokens removed (trigger or mate side) by the df ceiling
}

export interface SynonymMap {
    // trigger token -> mate tokens, all in processTerm space. Mates are
    // ordered (stable build) and each mate is unambiguous (symmetric guard),
    // so a mate maps back to exactly one source class per query.
    mates: Map<string, string[]>;
    stats: SynonymStats;
}

// A class member is usable only if it survives the query pipeline as exactly
// ONE token (multi-token alias phrases are v2), ≥2 chars, not purely numeric.
//
// Tokenized with seekTokenize — the SAME analyzer the index AND query use — in its
// CANONICAL stream (derived:false): possessive-strip + CJK segmentation, but NOT
// the additive glue/camelCase recall forms (which would mis-reject a single
// camelCase alias like "MemGraph" as multi-token). This retires the last caller of
// MiniSearch.getDefault('tokenize') (audit 2026-06-29 #3), so tokenize.ts's "sole
// tokenizer" invariant is true again, and a CJK alias no longer forms a dead class
// the segmented query side can never trigger. Eval-gated as a strict no-op:
// identical synonym dictionary AND retrieval on personal/dnd/myvault
// (synonym_tokenizer_arm.py) — those vaults carry no possessive/CJK/\p{Sm} alias,
// the only members where the two tokenizers' single-token verdict diverges.
function singleToken(member: string): string | null {
    const raw = seekTokenize(member, { derived: false });
    if (raw.length !== 1) return null;
    const t = processQueryTerm(raw[0]);
    if (t === null || t.length < 2 || /^\d+$/.test(t)) return null;
    return t;
}

// Cheap, CONSERVATIVE "could this chunk's note affect the dictionary?" probe
// for the incremental rebuild gate (search.ts applyDelta). The dictionary is a
// pure function of alias-bearing notes: buildClasses keeps a class only when it
// has ≥2 single-token members, and a note's name supplies at most one, so a
// note with no frontmatter aliases NEVER forms a class. A delta that adds or
// removes no alias-bearing row therefore cannot change the dictionary.
//
// Conservative on purpose: a declared alias that tokenizes to nothing usable
// still returns true here, so the gate may over-trigger a harmless no-op
// rebuild — but it never UNDER-triggers, which would be a correctness bug
// (stale mates). Mirrors the `c.metadata?.aliases` access buildClasses uses;
// aliases is a normalized string[] (chunker extractAliases).
export function chunkDeclaresAlias(c: ChunkMeta): boolean {
    return (c.metadata?.aliases?.length ?? 0) > 0;
}

// One equivalence class per note: single-token members of {note name} +
// frontmatter aliases. Chunk metadata is file-level (duplicated across a
// note's chunks), so the first chunk per note suffices.
export function buildClasses(chunks: ChunkMeta[]): Set<string>[] {
    const seen = new Set<string>();
    const classes: Set<string>[] = [];
    for (const c of chunks) {
        if (seen.has(c.note_path)) continue;
        seen.add(c.note_path);
        const members = [extractNoteName(c), ...(c.metadata?.aliases ?? [])];
        const singles = new Set<string>();
        for (const m of members) {
            const t = singleToken(String(m));
            if (t !== null) singles.add(t);
        }
        if (singles.size >= 2) classes.push(singles);
    }
    return classes;
}

// dfFraction(term) -> fraction of chunks containing the term in any field
// (bm25.ts termDocFraction). Optional so the builder is testable standalone;
// absent means the df guard is skipped.
export function buildSynonymMap(
    chunks: ChunkMeta[],
    dfFraction?: (term: string) => number,
): SynonymMap {
    const classes = buildClasses(chunks);
    const membership = new Map<string, Set<string>[]>();
    for (const cls of classes) {
        for (const t of cls) {
            let m = membership.get(t);
            if (!m) membership.set(t, m = []);
            m.push(cls);
        }
    }
    const ambiguous = new Set<string>();
    for (const [t, clss] of membership) {
        if (clss.length > 1) ambiguous.add(t);
    }

    const stats: SynonymStats = {
        classes: classes.length, triggers: 0,
        droppedAmbiguous: ambiguous.size, droppedDf: 0,
    };
    // A term can be checked once as a trigger and again as a mate candidate of
    // other triggers — count unique dropped terms, not checks.
    const droppedDfTerms = new Set<string>();
    const overDf = (t: string): boolean => {
        if (!dfFraction) return false;
        const over = dfFraction(t) > SYNONYM_DF_CEILING;
        if (over) droppedDfTerms.add(t);
        return over;
    };

    const mates = new Map<string, string[]>();
    for (const [t, clss] of membership) {
        if (ambiguous.has(t)) continue;             // strict trigger guard
        if (overDf(t)) continue;                    // df guard, trigger side
        const out: string[] = [];
        for (const cls of clss) {
            for (const m of cls) {
                if (m === t || ambiguous.has(m)) continue;   // symmetric mate guard
                if (overDf(m)) continue;                     // df guard, mate side
                if (!out.includes(m)) out.push(m);
            }
        }
        if (out.length > 0) {
            mates.set(t, out);
            stats.triggers++;
        }
    }
    stats.droppedDf = droppedDfTerms.size;
    return { mates, stats };
}
