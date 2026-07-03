// The ONE definition of what an Obsidian tag looks like, shared by the query
// side (query-parser INLINE_FILTER_RE — `#tag` / `tag:x` filters) and the doc
// side (chunker → extractInlineTags below). Keeping both sides on the same
// character class is the point: audit R2 #2 was exactly a doc/query split —
// the suggester offered inline body tags (Obsidian's metadataCache indexes
// them) while the matcher store only ever saw frontmatter tags, so every such
// pill hard-excluded the notes it came from.

import { parseAtoms } from './atoms';

// One character allowed inside an Obsidian tag or property key: any NON-delimiter.
// Mirrors Obsidian's own tag grammar (https://obsidian.md/help/tags) — which is
// defined by exclusion, not an allow-list, so it admits letters of every script,
// digits, AND emoji/symbols in a single rule. Excluded: whitespace, the Unicode
// General (U+2000–206F) + Supplemental (U+2E00–2E7F) Punctuation blocks, the
// ASCII punctuation that terminates a tag, and `/` (the nesting separator, which
// we match BETWEEN segments). `-` and `_` are deliberately kept. So #café,
// #日本語, #🎉, kebab `#meeting-prep`, and [my-field:…] all bind. (Obsidian also
// forbids purely-numeric tags like #1984; the QUERY side over-accepts those —
// harmless, such a filter just matches no real tag — but the DOC side must
// reject them, because over-accepting there would MANUFACTURE the tag and make
// the filter match. extractInlineTags carries that guard.)
export const TAG_CH = "[^\\s!-,./:-@\\[-\\^`{-~\\u2000-\\u206F\\u2E00-\\u2E7F]";
export const TAG_RUN = `${TAG_CH}+(?:/${TAG_CH}+)*`; // hierarchical: parent/child/...

// A doc-side tag site: `#` preceded by start-of-line or whitespace. Obsidian
// only recognizes free-standing tags — `foo#bar` and URL fragments are not
// tags. Alternation instead of a lookbehind (lookbehind crashes iOS < 16.4,
// the 1.0.1 review lesson); the consumed boundary char is fine because only
// the capture group is read. `m` so `^` means line start.
const INLINE_TAG_RE = new RegExp(`(?:^|\\s)#(${TAG_RUN})`, 'gmu');
// Cheap pre-check on the RAW body (non-global so .test carries no lastIndex
// state): a candidate tag site must exist before we pay parseAtoms. Heading
// lines can't match (`# ` has whitespace after the hash; `##`'s second hash
// is excluded punctuation), so ATX-heavy notes without tags skip the scan.
// Conservative: fence/comment content can match (the strip then discards it).
const INLINE_TAG_PROBE = new RegExp(`(?:^|\\s)#${TAG_CH}`, 'mu');

// Obsidian comments are hidden in reading view and NOT tag-indexed by Obsidian.
const OBSIDIAN_COMMENT_RE = /%%[\s\S]*?%%/g;
// An UNCLOSED trailing `%%` comments out everything to end-of-note in
// Obsidian; runs after the paired strip so only a lone leftover opener matches.
const UNCLOSED_COMMENT_RE = /%%[\s\S]*$/;
// An inline code span is code, not a tag site (`#include`, `#fff`).
const INLINE_CODE_RE = /`[^`\n]*`/g;
// Stripped spans are replaced by '.' — a NON-whitespace, non-TAG_CH char — so
// the strip can neither manufacture a boundary (`x\`y\`#z`: a space here would
// let #z bind, which Obsidian does not) nor extend an adjacent tag (`.`
// terminates a TAG_RUN, unlike e.g. NUL which TAG_CH admits).
const STRIP_PLACEHOLDER = '.';

// Scan body text (frontmatter already stripped) for inline `#tags`, the way
// Obsidian's own metadataCache does: fenced code excluded (parseAtoms fence
// typing — same fence grammar the chunker splits by), comments and inline code
// excluded, purely-numeric tags rejected. Returns tags WITHOUT the leading `#`,
// original case preserved, case-insensitive first-wins dedup — the matcher
// (query-parser) lowercases both sides, and BM25's tags field wants the
// author's casing. Heading lines need no special-casing: `# Title` has
// whitespace after `#` (not a TAG_CH) and `##x`'s second `#` sits in the
// excluded ASCII-punctuation run, so neither can bind.
export function extractInlineTags(body: string): string[] {
    if (!INLINE_TAG_PROBE.test(body)) return [];
    const prose = parseAtoms(body)
        .filter(a => a.type !== 'fence')
        .map(a => a.text)
        .join('\n')
        .replace(OBSIDIAN_COMMENT_RE, STRIP_PLACEHOLDER)
        .replace(UNCLOSED_COMMENT_RE, '')
        .replace(INLINE_CODE_RE, STRIP_PLACEHOLDER);
    const out: string[] = [];
    const seen = new Set<string>();
    for (const m of prose.matchAll(INLINE_TAG_RE)) {
        const tag = m[1];
        // Obsidian forbids tags with no non-numerical character (#1984, #2026/06).
        if (!/[^0-9/]/u.test(tag)) continue;
        const key = tag.toLowerCase();
        if (!seen.has(key)) {
            seen.add(key);
            out.push(tag);
        }
    }
    return out;
}
