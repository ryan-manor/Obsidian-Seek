// Link insertion from the search modal and seek:insert-link CLI. Builds vault-
// respecting links via fileManager.generateMarkdownLink and inserts at the active
// editor cursor.

import { MarkdownView } from 'obsidian';
import type { App, EditorPosition, TFile } from 'obsidian';

export interface BuildNoteLinkOpts {
    subpath?: string;
    alias?: string;
}

export interface InsertLinkInEditorOpts {
    from?: EditorPosition;
    to?: EditorPosition;
}

// The modal never aliases the link with the search text — deliberately no
// search-text-as-alias mode. Auto-aliasing with the query would teach that
// prose/sentence-shaped queries are good queries (they aren't; see the query
// model in fusion.ts), so the display text stays the note's own name. An alias
// remains available as an explicit choice via the CLI's `alias=` param.
/** CLI alias: explicit `alias=` param only (default is a plain wiki link). */
export function resolveInsertLinkAlias(explicitAlias?: string | null): string | undefined {
    const a = explicitAlias?.trim();
    return a || undefined;
}

export function headingSubpath(headingPath: string[] | undefined | null): string | undefined {
    if (!headingPath?.length) return undefined;
    const last = headingPath[headingPath.length - 1]?.trim();
    if (!last) return undefined;
    return `#${last}`;
}

// The subpath mirrors what a plain click on the same row would do (search-modal
// openResult): a title-nav-gated result opens at the top of the doc, so its
// link is the bare note ([[Note]]); any other result navigates to its matched
// section, so the link carries the heading ([[Note#Section]]). Callers pass
// the same titleNav verdict they would use to open the row — one behavior,
// no setting.
//
// One redundancy is collapsed: a LONE heading that duplicates the note's own
// title is the classic "# Title" first line, and [[Note#Note]] both reads as
// chrome and lands at the top anyway — so it drops to the bare note. A nested
// path keeps its last segment even when it matches the title (that's a
// deliberate mid-doc section, and the fragment is what finds it).
export function resolveInsertLinkSubpath(
    headingPath: string[] | undefined | null,
    titleNavOpen: boolean,
    noteBasename: string,
): string {
    if (titleNavOpen) return '';
    if (
        headingPath?.length === 1 &&
        headingPath[0].trim().toLowerCase() === noteBasename.trim().toLowerCase()
    ) return '';
    return headingSubpath(headingPath) ?? '';
}

function noteBasename(file: TFile): string {
    const base = file.path.split('/').pop() ?? file.path;
    return base.replace(/\.md$/i, '');
}

export function buildNoteLink(app: App, file: TFile, opts?: BuildNoteLinkOpts): string {
    const active = app.workspace.getActiveFile();
    const subpath = opts?.subpath ?? '';
    const alias = opts?.alias;

    if (active) {
        return app.fileManager.generateMarkdownLink(
            file,
            active.path,
            subpath,
            alias ?? '',
        );
    }

    const base = noteBasename(file);
    const pathPart = subpath ? `${base}${subpath}` : base;
    if (alias) return `[[${pathPart}|${alias}]]`;
    return `[[${pathPart}]]`;
}

export function insertLinkInEditor(
    app: App,
    link: string,
    opts?: InsertLinkInEditorOpts,
): { ok: true } | { ok: false; reason: string } {
    const view = app.workspace.getActiveViewOfType(MarkdownView);
    if (!view?.editor) return { ok: false, reason: 'no active editor' };

    const editor = view.editor;
    if (opts?.from && opts?.to) {
        editor.replaceRange(link, opts.from, opts.to);
        const end = editor.offsetToPos(editor.posToOffset(opts.from) + link.length);
        editor.setCursor(end);
        return { ok: true };
    }

    const cursor = editor.getCursor();
    editor.replaceRange(link, cursor, cursor);
    const end = editor.offsetToPos(editor.posToOffset(cursor) + link.length);
    editor.setCursor(end);
    return { ok: true };
}

export function isInsertableMarkdownFile(file: TFile | null): file is TFile {
    return file != null && file.extension === 'md';
}
