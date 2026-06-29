// Synthetic search documents for Obsidian `.base` files (Bases).
//
// A .base is a YAML *view definition* — a saved query plus presentation — and
// stores ZERO note content. We model a base as a DOCUMENT whose VIEWS are its
// sections (see [[Seek - Base-as-Document Model (Plan)]]): `extractBaseDocs`
// returns one BaseView per non-generic view PLUS a base-level entry, and the
// chunker (`chunkBase`) turns each into a Chunk that reuses the whole note
// pipeline. A semantic query then routes to the right VIEW ("fashion clips" →
// the Clothing view of the Clippings base), which a single mashed-together doc
// could never discriminate.
//
// The only indexable signal in a .base is:
//   - each view's NAME (becomes the chunk title + heading_path, so it earns the
//     3.0x BM25 headings field and enters the dense channel)
//   - the string LITERALS inside filter expressions ("writing", "Example Wiki",
//     "Clothing"), which name the domain a view selects
//
// Deliberately NOT indexed (chosen scope: "view name + filter literals"):
//   - filter operators / property paths (pageType ==, file.tags.contains) — grammar
//   - formula / summary bodies — pure code; we simply never read config.formulas
//     or config.summaries, so they are structurally excluded (no skip-list needed)
//   - layout config (columnSize / sort / markerIcon) — presentation; the typed
//     walk only descends `filters` trees, so it never sees these
//
// We parse with Obsidian's `parseYaml` (runtime-provided, zero bundle cost — the
// bundle-size argument for hand-scanning the chunker's frontmatter does not apply
// here, and per-view structure is awkward with a flat scan and trivial with the
// parsed tree). The expression-literal cleaner (VALUE_RE) still rejects leaked
// code: a quoted formula literal like "Quick (<30m)" carries ()<> and is dropped.

import { parseYaml } from 'obsidian';
import type { BasesConfigFile } from 'obsidian';
import type { BaseView } from './types';

export type { BaseView };

// View `name:` values that carry no domain signal — every base has a "Table".
const GENERIC_VIEW_NAMES = new Set(['table', 'view', 'list', 'untitled', 'cards', 'board', 'grid']);

const QUOTED_RE = /"([^"]+)"/g;

// A value literal worth indexing: letters/numbers/space and the handful of
// punctuation that shows up in real domain values (& / - '), nothing else.
// Expression literals ("!coordinates.isEmpty()", "Quick (<30m)", "YYYY-[W]WW")
// carry ()=<>.[] and are rejected; bare numbers are dropped as non-descriptive.
const VALUE_RE = /^[\p{L}\p{N} '&/-]{2,}$/u;

function basename(path: string): string {
    return path.split('/').pop()!.replace(/\.base$/, '');
}

// Walk a filter tree (a recursive and/or/not of expression strings) and collect
// the quoted, domain-shaped literals from its leaves into `out`. parseYaml has
// already stripped any OUTER YAML quoting, so a fully-quoted expression scalar
// ("!coordinates.isEmpty()") yields no inner `"..."` and is skipped — while a
// plain scalar (category == "product") still exposes its inner literal.
function collectFilterLiterals(filter: unknown, out: Set<string>): void {
    if (!filter) return;
    if (typeof filter === 'string') {
        let m: RegExpExecArray | null;
        QUOTED_RE.lastIndex = 0;
        while ((m = QUOTED_RE.exec(filter)) !== null) {
            const v = m[1].trim();
            if (VALUE_RE.test(v) && !/^\d+$/.test(v)) out.add(v);
        }
        return;
    }
    // `filter` is `any` from parseYaml; a hand-edited / corrupt .base can put a
    // non-string, non-object leaf (a bare number) or a non-array and/or/not branch
    // here. Guard so a malformed tree degrades to "no literals" instead of throwing
    // (`'and' in 5` and `for…of 5` both throw) — which would drop the whole file
    // and break the documented graceful-degradation contract.
    if (typeof filter !== 'object') return;
    const f = filter as Record<string, unknown>;
    const branch =
        Array.isArray(f.and) ? f.and :
        Array.isArray(f.or) ? f.or :
        Array.isArray(f.not) ? f.not : [];
    for (const sub of branch) collectFilterLiterals(sub, out);
}

// Extract the per-view search documents from a .base file's raw YAML. Returns a
// base-level entry (viewName null) followed by one entry per non-generic view.
// `content` is the deduped base name + inherited top-level literals + the view's
// own literals + the view name; it is never empty (the base name is always
// present), so the chunker never needs the lexical-only fallback. A malformed or
// empty .base degrades to a single base-level entry indexed by name.
export function extractBaseDocs(raw: string, path: string): BaseView[] {
    const base = basename(path);

    let config: BasesConfigFile;
    try {
        const parsed = parseYaml(raw);
        config = parsed && typeof parsed === 'object' ? (parsed as BasesConfigFile) : {};
    } catch {
        config = {};
    }

    // Top-level filter literals — inherited by every view AND the base-level entry.
    const topLiterals = new Set<string>();
    collectFilterLiterals(config.filters, topLiterals);

    // Generic / unnamed views carry no routing signal, but their filter literals
    // still describe the base's domain — fold them into the BASE-LEVEL entry only
    // (folding into named views would over-broaden their content). This preserves
    // the old single-doc model's "capture every filter literal somewhere".
    const genericLiterals = new Set<string>();
    const named: Array<{ name: string; literals: Set<string> }> = [];

    const views = Array.isArray(config.views) ? config.views : [];
    for (const view of views) {
        const name = (view?.name ?? '').trim();
        const literals = new Set<string>();
        collectFilterLiterals(view?.filters, literals);
        if (!name || GENERIC_VIEW_NAMES.has(name.toLowerCase())) {
            for (const l of literals) genericLiterals.add(l);
            continue;
        }
        named.push({ name, literals });
    }

    const dedupJoin = (parts: string[]): string => [...new Set(parts.filter(Boolean))].join(' ');

    const docs: BaseView[] = [];
    // Base-level entry: opens the default view, wins bare-name queries via title-boost.
    docs.push({ viewName: null, content: dedupJoin([base, ...topLiterals, ...genericLiterals]) });
    // One entry per named view: base + inherited top-level literals + own literals + name.
    for (const v of named) {
        docs.push({ viewName: v.name, content: dedupJoin([base, ...topLiterals, ...v.literals, v.name]) });
    }
    return docs;
}
