import { describe, expect, it } from 'vitest';
import type { App, TFile } from 'obsidian';
import {
    buildNoteLink,
    headingSubpath,
    resolveInsertLinkAlias,
    resolveInsertLinkSubpath,
} from './insert-link';

describe('headingSubpath', () => {
    it('returns undefined for empty paths', () => {
        expect(headingSubpath([])).toBeUndefined();
        expect(headingSubpath(null)).toBeUndefined();
        expect(headingSubpath(undefined)).toBeUndefined();
    });

    it('uses the last heading segment', () => {
        expect(headingSubpath(['Agenda', 'Intern pgm'])).toBe('#Intern pgm');
        expect(headingSubpath(['Only'])).toBe('#Only');
    });
});

describe('resolveInsertLinkSubpath', () => {
    const sectionPath = ['Agenda', 'Intern pgm'];
    const base = 'Weekly Sync';

    it('links the matched section for a normal chunk-jump result', () => {
        expect(resolveInsertLinkSubpath(sectionPath, false, base)).toBe('#Intern pgm');
    });

    it('links the bare note for a title-nav result, even with a section hit', () => {
        expect(resolveInsertLinkSubpath(sectionPath, true, base)).toBe('');
    });

    it('links the bare note when the result has no heading', () => {
        expect(resolveInsertLinkSubpath([], false, base)).toBe('');
        expect(resolveInsertLinkSubpath(null, false, base)).toBe('');
    });

    it('drops a lone heading that duplicates the note title (case-insensitive)', () => {
        expect(resolveInsertLinkSubpath(['Weekly Sync'], false, base)).toBe('');
        expect(resolveInsertLinkSubpath(['weekly sync'], false, base)).toBe('');
        expect(resolveInsertLinkSubpath([' Weekly Sync '], false, base)).toBe('');
    });

    it('keeps a lone heading that differs from the note title', () => {
        expect(resolveInsertLinkSubpath(['Agenda'], false, base)).toBe('#Agenda');
    });

    it('keeps a nested last segment even when it matches the note title', () => {
        expect(resolveInsertLinkSubpath(['Intro', 'Weekly Sync'], false, base)).toBe('#Weekly Sync');
    });
});

describe('resolveInsertLinkAlias', () => {
    it('returns trimmed explicit alias for CLI', () => {
        expect(resolveInsertLinkAlias('  cli alias  ')).toBe('cli alias');
    });

    it('returns undefined when no explicit alias', () => {
        expect(resolveInsertLinkAlias(undefined)).toBeUndefined();
        expect(resolveInsertLinkAlias('')).toBeUndefined();
        expect(resolveInsertLinkAlias('   ')).toBeUndefined();
    });
});

describe('buildNoteLink', () => {
    const file = { path: 'folder/Note Title.md', extension: 'md' } as TFile;

    it('uses generateMarkdownLink when active file exists', () => {
        const app = {
            workspace: { getActiveFile: () => ({ path: 'Daily.md' }) },
            fileManager: {
                generateMarkdownLink: (
                    f: TFile,
                    source: string,
                    subpath: string,
                    alias: string,
                ) => `LINK:${f.path}:${source}:${subpath}:${alias}`,
            },
        } as unknown as App;

        expect(buildNoteLink(app, file, { subpath: '#Sec', alias: 'alias' }))
            .toBe('LINK:folder/Note Title.md:Daily.md:#Sec:alias');
    });

    it('falls back to wikilink syntax without active file', () => {
        const app = {
            workspace: { getActiveFile: () => null },
            fileManager: { generateMarkdownLink: () => 'unused' },
        } as unknown as App;

        expect(buildNoteLink(app, file)).toBe('[[Note Title]]');
        expect(buildNoteLink(app, file, { subpath: '#H', alias: 'test' }))
            .toBe('[[Note Title#H|test]]');
    });
});
