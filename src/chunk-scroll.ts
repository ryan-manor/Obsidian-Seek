// Land a result-open on its chunk's FIRST line, top-aligned with a small margin.
//
// The previous landing scrolled the whole chunk span (start_line..end_line) into
// view with Obsidian's `editor.scrollIntoView(range, center=true)`. For a chunk
// taller than the viewport, CodeMirror centres the range's MIDPOINT, so the
// chunk's heading (the line the result's breadcrumb names) lands above the fold.
// Anchoring to start_line alone, at the top of the viewport, makes the heading
// the first thing the user sees regardless of chunk length.
//
// Obsidian's public Editor API only offers "nearest" or "center" scrolling, so
// top alignment goes through the underlying CodeMirror 6 EditorView (`editor.cm`,
// untyped in the public API). EditorView.scrollIntoView is a static effect
// factory; we reach it via `cm.constructor` rather than importing
// @codemirror/view so the plugin keeps no CodeMirror dependency. If either piece
// is missing (API drift, a non-CM editor) we fall back to centring the single
// start line, which still beats centring the whole span.

export const CHUNK_SCROLL_MARGIN_PX = 48;

interface Pos { line: number; ch: number }

export interface ScrollableEditor {
    setCursor(pos: Pos): void;
    posToOffset(pos: Pos): number;
    scrollIntoView(range: { from: Pos; to: Pos }, center?: boolean): void;
    cm?: unknown;
}

type ScrollEffectFactory = (pos: number, opts: { y: 'start'; yMargin: number }) => unknown;

function cmTopScroller(cm: unknown): ((pos: number, margin: number) => void) | null {
    if (!cm || typeof cm !== 'object') return null;
    const view = cm as { dispatch?: unknown; constructor?: { scrollIntoView?: unknown } };
    const factory = view.constructor?.scrollIntoView;
    if (typeof view.dispatch !== 'function' || typeof factory !== 'function') return null;
    const dispatch = view.dispatch.bind(cm) as (spec: { effects: unknown }) => void;
    return (pos, margin) => dispatch({ effects: (factory as ScrollEffectFactory)(pos, { y: 'start', yMargin: margin }) });
}

// `startLine` is the chunk's 1-based raw-file line (frontmatter included), the
// same coordinate system the editor uses.
export function scrollEditorToLine(editor: ScrollableEditor, startLine: number, marginPx = CHUNK_SCROLL_MARGIN_PX): void {
    const target: Pos = { line: Math.max(0, startLine - 1), ch: 0 };
    editor.setCursor(target);
    const scrollTop = cmTopScroller(editor.cm);
    if (scrollTop) {
        scrollTop(editor.posToOffset(target), marginPx);
        return;
    }
    editor.scrollIntoView({ from: target, to: target }, true);
}
