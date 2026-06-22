import { describe, it, expect } from 'vitest';
import { buildClasses, buildSynonymMap, SYNONYM_DF_CEILING } from './synonyms';
import type { Chunk } from './types';

function makeChunk(id: string, title: string, content: string, aliases: string[] = []): Chunk {
    return {
        chunk_id: id,
        title,
        content,
        note_path: `${id}.md`,
        heading_path: [],
        metadata: { tags: [], aliases, pageType: '', created: null, modified: null, properties: {} },
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
