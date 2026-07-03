// Pure snippet sanitation, extracted from search-modal.ts so it can be
// unit-tested without Obsidian (same reasoning as ./highlight). No runtime
// imports — string in, string out.

// A line that's part of a GFM pipe table: either a normal row that's wrapped in
// outer pipes (`| a | b |`, incl. the header) or a delimiter row (`|---|:--:|`,
// with or without outer pipes). Requiring outer pipes on content rows keeps a
// stray inline pipe in prose ("A | B") from being mistaken for a table.
const TABLE_ROW_RE = /^\s*\|.*\|\s*$/;
const TABLE_DELIM_RE = /^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)+\|?\s*$/;

// Sanitize a snippet before markdown-rendering it. Notes often open with a
// banner image, a transcluded note, a table, or a $$…$$ formula — all of which
// the renderer would expand into a full-height block and blow out the result
// row. We strip:
//   • $$display math$$          → a centred MathJax block (spans lines)
//   • Obsidian wikilink embeds  `![[cover.jpg]]` / `![[file|size]]`
//   • markdown image embeds     `![alt](url)`
//   • GFM pipe-table lines      `| a | b |`, `|---|---|`
//   • code-fence markers        ```` ```json ```` / `~~~` — see below
// Plain links (`[[…]]`, `[text](url)`), inline `$math$`, and inline formatting
// are left intact so they still render inline. Leftover blank lines are
// collapsed. (The CSS row-height ceiling is the backstop for anything that still
// slips through as a block — see .seek-result-snippet in styles.css.)
//
// Code fences get only their FENCE LINES removed (not the inner code): a real
// fenced block renders as a <pre> with Obsidian's floating "copy" button, and
// the row's nowrap+overflow clip then hides the code text but NOT the
// absolutely-positioned button — so a config-dump note (body = one ```json
// block) shows up as a lone copy icon with no preview. Dropping the fence lines
// lets the inner text render as a normal clipped one-liner instead.
//
// A chunk that's ENTIRELY a table or a $$…$$ formula (no surrounding prose)
// strips down to nothing above, which would otherwise render as a blank
// result row even though it's the thing that matched. In that case fall back
// to a flattened, single-line rendering of the stripped table/math content
// itself so the row still shows something recognizable.
export function sanitizeSnippet(md: string): string {
    const mathBlocks: string[] = [];
    const noEmbeds = md
        .replace(/\$\$[\s\S]*?\$\$/g, (block) => {           // $$display math$$ (multi-line)
            mathBlocks.push(block.slice(2, -2).trim());
            return '';
        })
        .replace(/!\[\[[^\]]*?\]\]/g, '')       // ![[file]] / ![[file|size]]
        .replace(/!\[[^\]]*?\]\([^)]*?\)/g, ''); // ![alt](url)

    const tableRows: string[] = [];
    const noTables = noEmbeds
        .split('\n')
        .filter(line => {
            if (TABLE_DELIM_RE.test(line)) return false; // |---|---| — no content to keep
            if (TABLE_ROW_RE.test(line)) {
                tableRows.push(line);
                return false;
            }
            return true;
        })
        .join('\n');

    const noFences = noTables.replace(/^\s*(?:```|~~~).*$/gm, ''); // ``` / ```json / ~~~

    const result = noFences.replace(/\n{3,}/g, '\n\n').trim();
    if (result) return result;

    // Nothing left after stripping — the chunk was entirely a table and/or a
    // math block. Flatten whichever we captured into a compact one-liner.
    const fallbackParts: string[] = [];
    if (tableRows.length) {
        const flatCells = tableRows
            .map(row => row.trim().replace(/^\|/, '').replace(/\|$/, ''))
            .map(row => row.split('|').map(cell => cell.trim()).filter(Boolean).join(' · '))
            .filter(Boolean);
        if (flatCells.length) fallbackParts.push(flatCells.join('  |  '));
    }
    if (mathBlocks.length) {
        const flatMath = mathBlocks
            .map(block => block.replace(/\s+/g, ' ').trim())
            .filter(Boolean)
            .join('  ');
        if (flatMath) fallbackParts.push(flatMath);
    }

    return fallbackParts.join('  —  ');
}
