// WS3 atomic-unit parser — the structure-aware boundary layer.
//
// Splitting markdown at /\n\n+/ (the pre-WS3 unit boundary in both the
// chunker's splitOversized and token-budget's splitChunk) slices fenced code
// blocks at their internal blank lines, scatters table rows, and breaks
// callouts at bare ">" continuation lines. This module replaces that with a
// line-based scan into ATOMS: paragraph, fence, table, callout. Structural
// atoms (fence/table/callout) are indivisible at any blank-line boundary —
// they split only at the token-budget hard ceiling, with structure-aware
// re-wrapping (fence-reopen markers, repeated table headers; token-budget.ts).
//
// Join contract: atoms.map(a => a.text).join('\n\n') preserves every
// non-whitespace character in order. Blank-line runs collapse to one and a
// structural atom butted directly against prose gains a separating blank
// line — the same normalization class as the old split, verified by the
// round-trip invariant in atoms.test.ts.
//
// Scope decisions (of record, plan §WS3):
//   - Lists are NOT atoms — loose lists split at blank lines exactly like
//     the old boundary (paragraphs.md fixture is the regression anchor).
//   - Callouts = any blockquote run (Obsidian callouts are blockquotes with
//     a [!type] marker; both get the same atomicity). Lazy continuation
//     (non-">" lines continuing a quote) is not honored — vault callouts are
//     written with ">" on every line.
//   - Setext headings are out of scope; the chunker has only ever split on
//     ATX headings, and scanHeadings keeps that contract (now fence-aware).

export type AtomType = 'paragraph' | 'fence' | 'table' | 'callout';

export interface Atom {
    type: AtomType;
    // Trimmed block text. Structural atoms keep their internal newlines and
    // blank lines; paragraphs never contain a blank line.
    text: string;
}

// CommonMark fence: up to 3 leading spaces, 3+ backticks or tildes, then an
// info string (which may not contain a backtick when the fence is backticks).
const FENCE_OPEN_RE = /^ {0,3}(`{3,}|~{3,})(.*)$/;
const HEADING_RE = /^(#{1,6})\s+(.+)$/;
// GFM delimiter row: cells of :?-+:? separated by pipes. Requires at least
// one pipe and one dash to avoid matching prose like ":- maybe".
const TABLE_DELIM_RE = /^ {0,3}\|?[ \t:|-]*-[ \t:|-]*\|?[ \t]*$/;
const CALLOUT_LINE_RE = /^ {0,3}>/;

interface FenceOpen {
    marker: string; // the full run of ` or ~ characters
    char: '`' | '~';
}

function fenceOpenAt(line: string): FenceOpen | null {
    const m = FENCE_OPEN_RE.exec(line);
    if (!m) return null;
    const marker = m[1];
    const char = marker[0] as '`' | '~';
    // A backtick fence's info string may not contain backticks (CommonMark);
    // such a line is inline code, not a fence open.
    if (char === '`' && m[2].includes('`')) return null;
    return { marker, char };
}

function fenceCloses(line: string, open: FenceOpen): boolean {
    const m = /^ {0,3}(`{3,}|~{3,})[ \t]*$/.exec(line);
    return m !== null && m[1][0] === open.char && m[1].length >= open.marker.length;
}

function isTableStart(line: string, next: string | undefined): boolean {
    if (next === undefined) return false;
    if (!line.includes('|') || line.trim().length === 0) return false;
    return TABLE_DELIM_RE.test(next) && next.includes('|');
}

// Parse body text into atoms. Input is section content (already free of the
// headings the chunker split on), but heading lines are tolerated as prose —
// the carry/fold path prepends heading words to short-section content.
export function parseAtoms(text: string): Atom[] {
    const lines = text.split('\n');
    const atoms: Atom[] = [];
    let para: string[] = [];

    const flushPara = () => {
        const t = para.join('\n').trim();
        if (t) atoms.push({ type: 'paragraph', text: t });
        para = [];
    };

    let i = 0;
    while (i < lines.length) {
        const line = lines[i];

        // Blank line: paragraph boundary (and a no-op between atoms).
        if (line.trim().length === 0) {
            flushPara();
            i++;
            continue;
        }

        const open = fenceOpenAt(line);
        if (open) {
            flushPara();
            let j = i + 1;
            while (j < lines.length && !fenceCloses(lines[j], open)) j++;
            // j is the closing line, or lines.length when unterminated —
            // an unterminated fence runs to end of input (CommonMark).
            const end = j < lines.length ? j + 1 : lines.length;
            atoms.push({ type: 'fence', text: lines.slice(i, end).join('\n').trim() });
            i = end;
            continue;
        }

        if (CALLOUT_LINE_RE.test(line)) {
            flushPara();
            let j = i;
            while (j < lines.length && CALLOUT_LINE_RE.test(lines[j])) j++;
            atoms.push({ type: 'callout', text: lines.slice(i, j).join('\n').trim() });
            i = j;
            continue;
        }

        if (isTableStart(line, lines[i + 1])) {
            flushPara();
            let j = i + 2; // header + delimiter consumed
            while (j < lines.length && lines[j].trim().length > 0 && lines[j].includes('|')) j++;
            atoms.push({ type: 'table', text: lines.slice(i, j).join('\n').trim() });
            i = j;
            continue;
        }

        para.push(line);
        i++;
    }
    flushPara();
    return atoms;
}

// Fence-aware ATX heading scan — the chunker's section splitter. Replaces
// the raw per-line HEADING_RE loop, which treated "# comment" lines inside
// fenced code as headings and split notes mid-fence (WS3 fixture
// fence-hash-heading.md). Callout/table state is irrelevant here: a "#" line
// inside a blockquote or table cell never matched HEADING_RE anyway.
export function scanHeadings(
    lines: string[],
): Array<{ lineNum: number; level: number; text: string }> {
    const headings: Array<{ lineNum: number; level: number; text: string }> = [];
    let fence: FenceOpen | null = null;
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        if (fence) {
            if (fenceCloses(line, fence)) fence = null;
            continue;
        }
        const open = fenceOpenAt(line);
        if (open) {
            fence = open;
            continue;
        }
        const m = HEADING_RE.exec(line);
        if (m) headings.push({ lineNum: i, level: m[1].length, text: m[2].trim() });
    }
    return headings;
}
