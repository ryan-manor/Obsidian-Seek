// Two-surfaces normalization for frontmatter property VALUES.
//
// Every property value has exactly two legitimate canonical forms, and the bug
// class this module exists to kill is any indexing/UI site picking the wrong
// one by accident. The "Notes Personal Places Zurich Zurith" keyword-stuffing
// incident (see [[Seek Index Processing Audit]]) was three separate sites
// (bm25 props field, autosuggest, would-be dense text) all reusing the FILTER
// matcher's unwrap — which is correct for binding and wrong for indexing.
//
//  - bind-form    : lowercased, wikilink syntax flattened to spaces so the link
//                   TARGET, its PATH, and its ALIAS are ALL substring-matchable.
//                   Wanted by [key:value] filter binding — `[placeLoc:Places]`
//                   and `[placeLoc:Zurich]` should both bind `[[.../Places/Zurich]]`.
//
//  - display-form : the canonical note NAME — target basename only; alias, path
//                   segments, and #heading/^block refs all dropped. Wanted by
//                   anything INDEXED or SHOWN (BM25 props field, autosuggest
//                   offerings, future dense-props text), where path tokens are
//                   junk that collide with real queries ("personal note...") and
//                   a doubled "Zurich Zurich" manufactures fake TF.
//
// Pick the surface by name at every call site; never hand-roll an unwrap.

// Matcher-style unwrap: `[[`, `]]`, and the alias `|` all become spaces, then
// lowercase. Path and alias and target tokens all survive — that breadth is the
// point for substring filter matching. (Identical to the former local
// normalizePropValue in query-parser.ts, which now delegates here.)
export function toBindForm(s: string): string {
    return s
        .toLowerCase()
        .replace(/\[\[|\]\]/g, ' ')
        .replace(/\|/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();
}

// Each `[[...]]` wikilink collapses to its target BASENAME — the last path
// segment, with the alias (everything after `|`) and any #heading/^block ref
// dropped. Non-wikilink text passes through untouched. Case is preserved (the
// suggester shows this verbatim; BM25's processTerm lowercases downstream).
//
//   [[Austin]]                     -> "Austin"
//   [[Notes/.../Places/Zurich|Zurith]] -> "Zurich"   (alias + path gone)
//   [[Jane Doe|Alex]]             -> "Jane Doe"     (alias dropped, name kept)
//   [[San Francisco|SF]]           -> "San Francisco" (abbrev alias dropped)
//   restaurants                    -> "restaurants"   (plain value, unchanged)
const WIKILINK_RE = /\[\[([^\]]+)\]\]/g;
export function toDisplayForm(s: string): string {
    return s
        .replace(WIKILINK_RE, (_, inner: string) => {
            const target = inner.split('|', 1)[0]; // drop alias
            const noRef = target.split(/[#^]/, 1)[0]; // drop heading/block ref
            const segs = noRef.split('/');
            return (segs[segs.length - 1] ?? '').trim(); // basename
        })
        .replace(/\s+/g, ' ')
        .trim();
}
