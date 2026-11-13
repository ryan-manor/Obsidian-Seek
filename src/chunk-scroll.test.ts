import { describe, it, expect, vi } from 'vitest';
import { scrollEditorToLine, CHUNK_SCROLL_MARGIN_PX, type ScrollableEditor } from './chunk-scroll';

// Stand-in for CodeMirror's EditorView: a class whose STATIC scrollIntoView
// builds an effect, reached by the helper through `cm.constructor`.
class FakeEditorView {
    static scrollIntoView = vi.fn((pos: number, opts: unknown) => ({ kind: 'scroll', pos, opts }));
    dispatch = vi.fn();
}

function makeEditor(cm?: unknown): ScrollableEditor & { setCursor: ReturnType<typeof vi.fn>; scrollIntoView: ReturnType<typeof vi.fn> } {
    return {
        setCursor: vi.fn(),
        posToOffset: (p) => p.line * 100 + p.ch,
        scrollIntoView: vi.fn(),
        cm,
    };
}

describe('scrollEditorToLine', () => {
    it('top-aligns the chunk start line via the CM6 view, not the whole span', () => {
        FakeEditorView.scrollIntoView.mockClear();
        const cm = new FakeEditorView();
        const editor = makeEditor(cm);

        scrollEditorToLine(editor, 144);

        expect(editor.setCursor).toHaveBeenCalledWith({ line: 143, ch: 0 });
        expect(FakeEditorView.scrollIntoView).toHaveBeenCalledWith(14300, { y: 'start', yMargin: CHUNK_SCROLL_MARGIN_PX });
        expect(cm.dispatch).toHaveBeenCalledWith({ effects: { kind: 'scroll', pos: 14300, opts: { y: 'start', yMargin: CHUNK_SCROLL_MARGIN_PX } } });
        expect(editor.scrollIntoView).not.toHaveBeenCalled();
    });

    it('falls back to centring the single start line when no CM view is exposed', () => {
        const editor = makeEditor(undefined);
        scrollEditorToLine(editor, 10);
        const pos = { line: 9, ch: 0 };
        expect(editor.scrollIntoView).toHaveBeenCalledWith({ from: pos, to: pos }, true);
    });

    it('falls back when the view lacks the static scroll factory', () => {
        const editor = makeEditor({ dispatch: vi.fn() });
        scrollEditorToLine(editor, 3);
        expect(editor.scrollIntoView).toHaveBeenCalledTimes(1);
    });

    it('clamps line 0/1 to the first editor line', () => {
        const editor = makeEditor(undefined);
        scrollEditorToLine(editor, 1);
        expect(editor.setCursor).toHaveBeenCalledWith({ line: 0, ch: 0 });
    });
});
