// Dense-channel text hygiene (Seek v8, 2026-06-28).
//
// The dense embed input is `title\n\ncontent\n\nsuffix` (token-budget.ts
// embedInput). Before v8, the `content` (section body) and the heading text
// folded into `title` were RAW markdown: inline URLs, image embeds and HTML
// residue all fed straight into the pooled granite vector. A subword embedder
// can't shatter a URL the way the BM25 delimiter class does (tokenize.ts) — it
// just spends CLS-pooling mass on `https`/`com`/og:image junk, which on a thin
// note dilutes the one value that matters.
//
// This collapses markdown link/embed syntax to the text a reader actually sees
// and strips URL/HTML noise — mirroring the GOAL the BM25 channel reaches by
// fragmentation (scheme/TLD dropped) but in a form an embedder can read. The
// chunker applies it at the source to BOTH the section body and each heading
// (decision 2026-06-28: clean once, so the cleaned bytes are what chunk_id
// hashes and what BOTH channels index — not a dense-only embedInput pass that
// would reintroduce the chunk_id staleness the denseSuffix-in-id comment
// fixes).
//
// Wikilinks use Obsidian's RENDERED-display rule ([[t|alias]] -> alias),
// deliberately NOT prop-normalize's canonical-basename rule (toDisplayForm):
// body/heading text is prose the reader sees, so the alias is the right surface
// form. The dense SUFFIX keeps toDisplayForm because a property value is an
// entity reference, not prose, and the basename is its canonical key. The two
// rules diverge on purpose.

import { parseAtoms } from './atoms';

// Image/asset extensions whose embeds carry no readable text — the file name
// ("Pasted image 20260628.png", an og:image path) is junk, so the whole embed
// drops to ''. Mirrors chunker.ts SUFFIX_ASSET_RE; kept local so dense-clean and
// chunker stay free of an import cycle (chunker imports this module).
const ASSET_EXT_RE = /\.(?:png|jpe?g|gif|webp|svg|bmp|tiff?|pdf|mp4|mov|webm|mkv|mp3|m4a|wav|zip)$/i;

// Order is load-bearing (see cleanDenseText): each EMBED form is collapsed
// before its non-embed counterpart so the leading `!` can't be orphaned, and
// markdown image before markdown link for the same reason.
const EMBED_WIKI_RE = /!\[\[([^\]]+?)\]\]/g;       // ![[target|disp]] / ![[t#sec]] / ![[img.png]]
const MD_IMAGE_RE   = /!\[([^\]]*?)\]\([^)]*?\)/g; // ![alt](url) -> alt
const WIKILINK_RE   = /\[\[([^\]]+?)\]\]/g;        // [[t|alias]] / [[t#sec]] / [[t]]
const MD_LINK_RE    = /\[([^\]]*?)\]\(([^)]*?)\)/g;// [text](url) -> text
const HTML_TAG_RE   = /<\/?[a-z][^>]*>/gi;         // <b> </b> <img …> -> ' '
const THEMATIC_RE   = /^[ \t]*([-*_])(?:[ \t]*\1){2,}[ \t]*$/gm; // --- *** ___ rule
const BARE_URL_RE   = /\bhttps?:\/\/[^\s)<>\]]+/gi;// a standalone URL run

// ![[…]] embed -> the text a reader sees. Alias wins; otherwise the target
// basename (heading/block ref dropped), UNLESS the target is an image/asset, in
// which case the embed renders a file with no readable text and drops entirely.
function embedDisplay(inner: string): string {
    const bar = inner.indexOf('|');
    if (bar !== -1) return inner.slice(bar + 1).trim();   // ![[note|disp]] -> disp
    const target = inner.split(/[#^]/, 1)[0].trim();
    if (ASSET_EXT_RE.test(target)) return '';             // ![[image.png]] -> ''
    const segs = target.split('/');
    return (segs[segs.length - 1] ?? '').trim();          // ![[Some Note]] -> Some Note
}

// [[…]] link -> rendered display (alias if present, else target basename). Case
// preserved; downstream tokenizers lowercase.
function wikiDisplay(inner: string): string {
    const bar = inner.indexOf('|');
    if (bar !== -1) return inner.slice(bar + 1).trim();   // [[t|alias]] -> alias
    const target = inner.split(/[#^]/, 1)[0];             // drop #heading / ^block
    const segs = target.split('/');
    return (segs[segs.length - 1] ?? '').trim();          // basename
}

// A bare URL -> its readable words. Strip scheme, a leading www., the host's
// final TLD label, and any ?query/#fragment; keep the host-sans-TLD label and
// the path words as spaces. This preserves the place-name-in-URL recall
// tokenize.ts documents ("…/100-198+E+5th+St+Garage" -> "E 5th St Garage")
// while dropping the scheme/TLD/query junk a subword embedder would mangle.
function cleanUrl(url: string): string {
    let s = url.replace(/^https?:\/\//i, '').replace(/^www\./i, '');
    s = s.split(/[?#]/, 1)[0];                            // drop query + fragment
    const slash = s.indexOf('/');
    let host = slash === -1 ? s : s.slice(0, slash);
    const path = slash === -1 ? '' : s.slice(slash + 1);
    host = host.replace(/\.[a-z]{2,24}$/i, '');           // drop the final TLD label
    return `${host} ${path}`.replace(/[^\p{L}\p{N}]+/gu, ' ').trim();
}

// Collapse one run of markdown text to the plain prose a reader sees. No fence
// awareness here — callers that pass multi-atom body text use cleanDenseBody,
// which routes fenced code around this. Safe on a single line (headings).
export function cleanDenseText(md: string): string {
    if (!md) return md;
    return md
        .replace(EMBED_WIKI_RE, (_m, inner: string) => embedDisplay(inner))
        .replace(MD_IMAGE_RE,   (_m, alt: string) => alt.trim())
        .replace(WIKILINK_RE,   (_m, inner: string) => wikiDisplay(inner))
        .replace(MD_LINK_RE,    (_m, text: string) => text.trim())
        .replace(HTML_TAG_RE,   ' ')
        .replace(BARE_URL_RE,   m => cleanUrl(m))
        .replace(THEMATIC_RE,   '')
        .replace(/[^\S\n]+/g, ' ')   // collapse horizontal whitespace, keep newlines
        .replace(/ ?\n ?/g, '\n')
        .replace(/\n{3,}/g, '\n\n')  // cap blank-line runs (paragraph boundaries survive)
        .trim();
}

// Fence-aware body cleaner. Code fences pass through VERBATIM — a URL or <tag>
// inside a fenced code sample is content, not noise, and the code-retrieval eval
// depends on it. Every other atom (paragraph, table, callout) is cleaned. Reuses
// parseAtoms so "what is code" has one definition (the CommonMark fence rules in
// atoms.ts), never a second drifting fence regex. Atoms rejoin on blank lines,
// which is how the chunker's own splitter (token-budget.ts) reads them back.
export function cleanDenseBody(md: string): string {
    if (!md) return md;
    return parseAtoms(md)
        .map(a => (a.type === 'fence' ? a.text : cleanDenseText(a.text)))
        .filter(t => t.length > 0)
        .join('\n\n')
        .trim();
}
