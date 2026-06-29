// Pure computation for the in-note match-highlight flash, factored out of
// search-modal.ts so it is unit-testable without Obsidian. buildMatchHighlight
// resolves the chunk window and hands the raw note text here; this module finds
// the query tokens and returns char-offset ranges ready for `eState.match`.

// Blank out the regions of a note that DON'T render as visible body text —
// frontmatter, embed transclusions, the target half of aliased wikilinks, and
// code — replacing them with spaces so the result stays byte-for-byte the same
// LENGTH as `content`. That length invariant is the whole point: offsets into
// the masked string are still valid offsets into the original `content`, which
// is the contract Obsidian's match-highlight (eState.match) relies on.
//
// Why it matters: buildMatchHighlight used to substring-search the raw markdown,
// so a query token could "match" inside a hidden region — e.g. the target path
// of `[[Public Content Recommendations|public assets]]` or an `![[…]]` embed —
// or mid-word inside a larger word. Obsidian can't anchor such an offset to any
// visible word, so it slides the highlight onto an unrelated slice of rendered
// text. Masking these regions (then matching on word boundaries) guarantees the
// offsets we hand back always point at real, visible, whole words.
//
// The visible alias text of `[[target|alias]]` is intentionally KEPT matchable;
// Obsidian silently drops an alias-interior offset rather than mis-placing it,
// so highlighting an alias hit is best-effort-safe, never wrong.
export function maskNonBodyText(content: string): string {
    const ch = content.split('');
    const blank = (start: number, end: number) => {
        for (let i = start; i < end && i < ch.length; i++) {
            if (ch[i] !== '\n') ch[i] = ' ';
        }
    };
    const fm = /^---\n[\s\S]*?\n---/.exec(content);
    if (fm) blank(0, fm[0].length);
    for (const m of content.matchAll(/!\[\[[^\]]*\]\]/g)) blank(m.index, m.index + m[0].length);
    for (const m of content.matchAll(/\[\[([^\]]*)\]\]/g)) {
        const open = m.index;
        if (open > 0 && content[open - 1] === '!') continue; // already blanked as an embed
        const inner = m[1];
        const pipe = inner.indexOf('|');
        if (pipe === -1) {
            blank(open, open + 2);                              // leading [[
            blank(open + 2 + inner.length, open + m[0].length); // trailing ]]
        } else {
            blank(open, open + 2 + pipe + 1);                   // [[target| → keep alias
            blank(open + 2 + inner.length, open + m[0].length); // trailing ]]
        }
    }
    for (const m of content.matchAll(/`[^`]*`/g)) blank(m.index, m.index + m[0].length);
    return ch.join('');
}

const escapeRegExp = (s: string): string => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

// Find the first in-window occurrence of each query token over a markup-masked,
// lowercased view of the note and return its [from,to] char offsets into the
// ORIGINAL `content` (the mask preserves length, so the offsets transfer back).
//
// `\b<tok>\w*` anchors to a word START and extends to the word end, so a token
// never lands mid-word ("of" can't match inside "professional") and a query
// stem still catches its inflection ("recommendation" → "recommendations").
// Single-char tokens and `stopwords` are skipped — the former would light up
// half the note, the latter would flash a word BM25 never indexed against (and
// is passed in as the SAME set BM25 uses so the highlight analyzer can't drift
// from the ranker).
//
// OFFSET DEFENCE — the returned ranges feed Obsidian's `eState.match`, whose
// decorations are built with a CodeMirror RangeSet. A RangeSet REQUIRES ranges
// in ascending order and silently mis-paints when they arrive out of order:
// because tokens are scanned in QUERY order, a later token can match an EARLIER
// document position than an earlier token (e.g. a term that also appears in the
// note title or an intro line that precedes the chunk body). So we (1) sort by
// start offset, (2) drop any out-of-bounds range, and (3) drop any range that
// overlaps its predecessor — the builder only ever sees ascending, in-bounds,
// disjoint ranges. Without (1) the highlight drifts onto unrelated text.
export function buildHighlightRanges(
    content: string,
    tokens: string[],
    lineStart: number,
    windowEnd: number,
    stopwords: ReadonlySet<string>,
): Array<[number, number]> {
    const masked = maskNonBodyText(content).toLowerCase();
    const raw: Array<[number, number]> = [];
    for (const tok of tokens) {
        if (tok.length < 2 || stopwords.has(tok)) continue;
        const re = new RegExp(`\\b${escapeRegExp(tok)}\\w*`, 'g');
        re.lastIndex = lineStart;
        const m = re.exec(masked);
        if (m && m.index < windowEnd) raw.push([m.index, m.index + m[0].length]);
    }

    raw.sort((a, b) => a[0] - b[0]);
    const ordered: Array<[number, number]> = [];
    for (const [from, to] of raw) {
        if (from < 0 || to > content.length || from >= to) continue; // bounds guard
        const prev = ordered[ordered.length - 1];
        if (prev && from < prev[1]) continue;                        // overlaps prev → keep disjoint
        ordered.push([from, to]);
    }
    return ordered;
}
