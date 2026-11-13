// Every way of opening a search-modal result (click, ⌘/Ctrl+click, tap, Enter,
// ⌘/Ctrl+Enter) must produce exactly ONE ClickEntry + ONE seek:result-open relay
// event, with the same schema, carrying `openMethod` so the eval harvest can tell
// keyboard opens from pointer opens. A double gesture (double-click, key-repeat,
// a click and an Enter on the same row) must not double-count.

import { describe, it, expect, vi } from 'vitest';
import { TFile } from 'obsidian';
import { SeekSearchModal, OpenGate, openMethodFor, DUPLICATE_OPEN_MS, type ResultOpenInfo } from './search-modal';
import type { ScoredChunk, ClickEntry } from './types';

describe('openMethodFor', () => {
    it('maps gesture + modifier + platform to a method', () => {
        expect(openMethodFor('key', false, false)).toBe('enter');
        expect(openMethodFor('key', true, false)).toBe('mod-enter');
        expect(openMethodFor('pointer', false, false)).toBe('click');
        expect(openMethodFor('pointer', true, false)).toBe('mod-click');
        expect(openMethodFor('pointer', false, true)).toBe('tap');
        // iPad with a hardware keyboard: ⌘+tap is still the alt-open gesture.
        expect(openMethodFor('pointer', true, true)).toBe('mod-click');
        // A hardware-keyboard Enter on mobile is still a keyboard open.
        expect(openMethodFor('key', false, true)).toBe('enter');
    });
});

describe('OpenGate', () => {
    it('admits one plain open per modal, then drops everything', () => {
        const g = new OpenGate();
        expect(g.admit('a', false, 0)).toBe(true);
        expect(g.admit('a', false, 5)).toBe(false);          // Enter right after a click
        expect(g.admit('b', false, 10_000)).toBe(false);     // modal is closing
        expect(g.admit('b', true, 10_000)).toBe(false);
    });

    it('lets alt-opens fan out but drops a same-row repeat inside the window', () => {
        const g = new OpenGate();
        expect(g.admit('a', true, 0)).toBe(true);
        expect(g.admit('a', true, DUPLICATE_OPEN_MS - 1)).toBe(false);  // double ⌘-click
        expect(g.admit('b', true, 100)).toBe(true);                      // different row
        expect(g.admit('a', true, 100 + DUPLICATE_OPEN_MS)).toBe(true);  // deliberate reopen
        expect(g.admit('c', false, 2000)).toBe(true);                    // then a plain open
        expect(g.admit('d', true, 2001)).toBe(false);
    });

    it('re-arms after release (plain open of a vanished file)', () => {
        const g = new OpenGate();
        expect(g.admit('a', false, 0)).toBe(true);
        g.release();
        expect(g.admit('b', false, 1)).toBe(true);
    });
});

// ---- the modal's real open path, driven with a minimal fake app ----

function chunk(id: string): ScoredChunk {
    return {
        chunk_id: id, note_path: `${id}.md`, content: 'x', score: 1,
        start_line: 0, end_line: 1, heading_path: [], title: id,
        ranking_signals: { dense: 0.5, bm25: 0.5, hybrid: 0.5, recency: 0, title_boost: 0, denseRaw: 0.5 },
    } as unknown as ScoredChunk;
}

function harness() {
    const clicks: ClickEntry[] = [];
    const relayed: ResultOpenInfo[] = [];
    const opened: string[] = [];
    const logger = { append: vi.fn(async (e: ClickEntry) => { clicks.push(e); }) };
    const settings = { altOpenLocation: 'tab', navTitleBoost: 0.5 } as never;
    const modal = new SeekSearchModal(
        {} as never, {} as never, logger as never,
        { ready: true, promise: Promise.resolve() }, settings,
        undefined, undefined, undefined, '', null,
        undefined, (info) => relayed.push(info),
    );
    const app = {
        vault: {
            getAbstractFileByPath: (p: string) => Object.assign(new TFile(), { path: p, extension: 'md' }),
        },
        workspace: {
            getLeaf: () => ({ view: null, openFile: async (f: TFile) => { opened.push(f.path); } }),
        },
    };
    const m = modal as unknown as Record<string, unknown>;
    m.app = app;
    m.latestSearchEntry = { searchId: 's1', query: 'q' };
    m.latestSearchCompletedAt = performance.now();
    m.latestResultsShown = [chunk('a'), chunk('b')];
    m.currentResults = [chunk('a'), chunk('b')];
    m.selectedIndex = 0;
    const call = (name: string, ...args: unknown[]) =>
        (m[name] as (...a: unknown[]) => unknown).apply(modal, args);
    return { modal, call, clicks, relayed, opened };
}

describe('SeekSearchModal open paths', () => {
    it('Enter on the selected row emits one event with openMethod=enter', async () => {
        const h = harness();
        h.call('openSelected', false);
        await new Promise(r => setTimeout(r, 0));
        expect(h.relayed).toHaveLength(1);
        expect(h.relayed[0]).toMatchObject({ rank: 1, newTab: false, openTarget: 'same', searchId: 's1', openMethod: 'enter' });
        expect(h.clicks).toHaveLength(1);
        expect(h.clicks[0]).toMatchObject({ rank: 1, openMethod: 'enter' });
        expect(h.opened).toEqual(['a.md']);
    });

    it('arrow-selected row + ⌘/Ctrl+Enter reports the selected rank and mod-enter', async () => {
        const h = harness();
        (h.modal as unknown as { selectedIndex: number }).selectedIndex = 1;
        h.call('openSelected', true);
        await new Promise(r => setTimeout(r, 0));
        expect(h.relayed).toHaveLength(1);
        expect(h.relayed[0]).toMatchObject({ rank: 2, newTab: true, openTarget: 'tab', openMethod: 'mod-enter' });
    });

    it('a click and an Enter on the same row count once', async () => {
        const h = harness();
        void h.call('openResult', chunk('a'), 1, false, 'click');
        h.call('openSelected', false);
        await new Promise(r => setTimeout(r, 0));
        expect(h.relayed.map(i => i.openMethod)).toEqual(['click']);
        expect(h.clicks).toHaveLength(1);
        expect(h.opened).toEqual(['a.md']);
    });

    it('click / mod-click / enter / mod-enter all emit the same schema', async () => {
        const keys: string[][] = [];
        for (const [method, newTab] of [['click', false], ['mod-click', true], ['enter', false], ['mod-enter', true], ['tap', false]] as const) {
            const h = harness();
            void h.call('openResult', chunk('a'), 1, newTab, method);
            await new Promise(r => setTimeout(r, 0));
            expect(h.relayed).toHaveLength(1);
            expect(h.relayed[0].openMethod).toBe(method);
            keys.push(Object.keys(h.relayed[0]).sort());
        }
        for (const k of keys) expect(k).toEqual(keys[0]);
    });
});
