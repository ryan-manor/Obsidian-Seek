import { describe, it, expect } from 'vitest';
import { buildClasses, buildSynonymMap, chunkDeclaresAlias, SYNONYM_DF_CEILING } from './synonyms';
import type { Chunk } from './types';

function makeChunk(id: string, title: string, content: string, aliases: string[] = []): Chunk {
    return {
        chunk_id: id,
        title,
        content,
        note_path: `${id}.md`,
        heading_path: [],
        metadata: { tags: [], aliases, created: null, modified: null, properties: {} },
        start_line: 0,
        end_line: 0,
    };
}

describe('buildClasses', () => {
    it('builds one class per note from single-token name + aliases', () => {
        const classes = buildClasses([
            makeChunk('a', 'Lr Home', 'body', ['Lr', 'Lightroom']),
        ]);
        // "Lr Home" is two tokens → not a member; the class is the aliases.
        expect(classes).toEqual([new Set(['lr', 'lightroom'])]);
    });

    it('dedupes chunks of the same note and skips junk members', () => {
        const classes = buildClasses([
            // multi-token alias, 1-char token, and numeric token all drop;
            // class falls below 2 members → no class at all.
            makeChunk('a', 'Alpha Beta', 'x', ['lightroom home', 'q', '2026']),
            { ...makeChunk('a2', 'Alpha Beta', 'second chunk', ['lightroom home']), note_path: 'a.md' },
        ]);
        expect(classes).toEqual([]);
    });

    it('runs members through the query term pipeline (lowercase + depluralize)', () => {
        const classes = buildClasses([
            makeChunk('a', 'Recs', 'body', ['Recommendations']),
        ]);
        expect(classes).toEqual([new Set(['rec', 'recommendation'])]);
    });

    it('keeps single-word camelCase members as ONE token (audit 2026-06-29 #3: derived:false)', () => {
        // singleToken now tokenizes with seekTokenize(derived:false): "MemGraph" /
        // "GraphDB" each stay one canonical token (the additive camelCase split is
        // OFF), so both survive the single-token gate and form a class. Under
        // derived:true they would split to 3 tokens and be rejected — the dropped-
        // alias regression this convergence deliberately avoids while still
        // retiring the last MiniSearch.getDefault('tokenize') caller.
        const classes = buildClasses([
            makeChunk('a', 'MemGraph', 'body', ['GraphDB']),
        ]);
        expect(classes).toEqual([new Set(['memgraph', 'graphdb'])]);
    });
});

describe('chunkDeclaresAlias (incremental rebuild gate)', () => {
    it('is true for any chunk whose note declares ≥1 alias', () => {
        expect(chunkDeclaresAlias(makeChunk('a', 'Lr Home', 'body', ['Lr', 'Lightroom']))).toBe(true);
    });

    it('is false for a no-alias note (a body-only edit cannot change the dictionary)', () => {
        expect(chunkDeclaresAlias(makeChunk('a', 'Plain Note', 'body', []))).toBe(false);
    });

    it('CONSERVATIVE: true even when the alias tokenizes to nothing usable (safe no-op rebuild)', () => {
        // "q" (1 char) and "2026" (numeric) both drop out of buildClasses, so this
        // note forms no class — yet the gate still fires. Over-triggering a no-op
        // rebuild is acceptable; UNDER-triggering would leave mates stale.
        const c = makeChunk('a', 'Alpha Beta', 'body', ['q', '2026']);
        expect(chunkDeclaresAlias(c)).toBe(true);
        expect(buildClasses([c])).toEqual([]); // ...the rebuild it triggers does nothing
    });

    it('agrees with buildClasses on which notes can contribute (no false negatives)', () => {
        // Every note that buildClasses turns into a class MUST be flagged by the
        // gate, or an alias edit could silently skip the rebuild.
        const contributing = makeChunk('a', 'Lr Home', 'body', ['Lr', 'Lightroom']);
        expect(buildClasses([contributing])).not.toEqual([]);
        expect(chunkDeclaresAlias(contributing)).toBe(true);
    });
});

describe('buildSynonymMap', () => {
    it('maps each unambiguous token to its classmates', () => {
        const { mates, stats } = buildSynonymMap([
            makeChunk('a', 'Lr Home', 'body', ['Lr', 'Lightroom']),
        ]);
        expect(mates.get('lr')).toEqual(['lightroom']);
        expect(mates.get('lightroom')).toEqual(['lr']);
        expect(stats.classes).toBe(1);
        expect(stats.triggers).toBe(2);
    });

    it('strict guard: an ambiguous token never triggers', () => {
        // "rohit" is in both people's classes → querying it must not expand.
        const { mates, stats } = buildSynonymMap([
            makeChunk('a', 'Rohit Sharma', 'body', ['Rohit', 'rsharma']),
            makeChunk('b', 'Rohit Kumar', 'body', ['Rohit', 'rkumar']),
        ]);
        expect(mates.has('rohit')).toBe(false);
        expect(stats.droppedAmbiguous).toBe(1);
    });

    it('symmetric guard: an ambiguous token is never INJECTED as a mate either', () => {
        // The trigger guard alone leaves the reverse direction open: querying
        // the unique nickname would inject "rohit", whose postings hit every
        // Rohit in the vault. The mate set must come back empty.
        const { mates } = buildSynonymMap([
            makeChunk('a', 'Rohit Sharma', 'body', ['Rohit', 'rsharma']),
            makeChunk('b', 'Rohit Kumar', 'body', ['Rohit', 'rkumar']),
        ]);
        expect(mates.has('rsharma')).toBe(false);   // only mate was ambiguous → no entry
        expect(mates.has('rkumar')).toBe(false);
    });

    it('df ceiling refuses common-word triggers and mates', () => {
        const chunks = [makeChunk('a', 'Hub', 'body', ['hub', 'zzindex'])];
        const common = new Set(['zzindex']);
        const df = (t: string) => (common.has(t) ? SYNONYM_DF_CEILING + 0.01 : 0.001);
        const { mates, stats } = buildSynonymMap(chunks, df);
        expect(mates.has('zzindex')).toBe(false);   // trigger side refused
        expect(mates.has('hub')).toBe(false);       // its only mate was refused → no entry
        expect(stats.droppedDf).toBeGreaterThan(0);
    });

    it('droppedDf counts unique terms, not checks (trigger + mate side = 1)', () => {
        // "zzindex" is checked twice: once as a trigger, once as hub's mate
        // candidate. The telemetry must report one dropped term, not two.
        const chunks = [makeChunk('a', 'Hub', 'body', ['hub', 'zzindex'])];
        const common = new Set(['zzindex']);
        const df = (t: string) => (common.has(t) ? SYNONYM_DF_CEILING + 0.01 : 0.001);
        const { stats } = buildSynonymMap(chunks, df);
        expect(stats.droppedDf).toBe(1);
    });
});
