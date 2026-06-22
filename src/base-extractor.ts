// Synthetic search document for Obsidian `.base` files (Bases).
//
// A .base is a YAML *view definition* — a saved query plus presentation — and
// stores ZERO note content. The only indexable signal is therefore:
//   - the view's NAME (its filename), handled by the caller as the chunk title
//   - the string LITERALS inside its filter expressions ("writing",
//     "meetings/1x1s", "Team Wiki"), which name the domain the view selects
//   - non-generic VIEW names ("Agenda", "Map") — a "Table"/"List" is noise
//
// Deliberately NOT indexed (chosen scope: "title + filter literals"):
//   - filter operators / property paths (pageType ==, file.tags.contains) — grammar
//   - formula bodies (number(date(...)), if(priority=="high",3,...)) — pure code,
//     ~3 KB in the worst base, and a swamp of literals like "Overdue"/"Quick (<30m)"
//   - layout config (columnSize / rowHeight / sort / markerIcon) — presentation
//
// Two layers keep formula/layout noise out: (1) we skip every line under a
// column-0 `formulas:` / `summaries:` key; (2) the literal cleaner rejects any
// value carrying expression punctuation, so a leaked formula literal like
// "Quick (<30m)" is dropped anyway.
//
// We hand-scan rather than pull in a YAML parser — same bundle-size reasoning as
// chunker.ts's frontmatter parser, and we only need a flat slice of the tree.

// View `name:` values that carry no domain signal — every base has a "Table".
const GENERIC_VIEW_NAMES = new Set(['table', 'view', 'list', 'untitled', 'cards', 'board', 'grid']);

// Top-level keys whose (indented) bodies are code, not content.
const SKIP_SECTIONS = new Set(['formulas', 'summaries']);

// A column-0 `key:` line — the only place a top-level section can begin. Indented
// lines (list items, nested filters, formula bodies) never match, so they inherit
// the current section.
const TOP_KEY_RE = /^([A-Za-z][A-Za-z0-9_]*):/;
const NAME_RE = /^\s*-?\s*name:\s*(.+?)\s*$/;
const QUOTED_RE = /"([^"]+)"/g;

// A value literal worth indexing: letters/numbers/space and the handful of
// punctuation that shows up in real domain values (& / - '), nothing else.
// Expression literals ("!coordinates.isEmpty()", "Quick (<30m)", "YYYY-[W]WW")
// carry ()=<>.[] and are rejected; bare numbers are dropped as non-descriptive.
const VALUE_RE = /^[\p{L}\p{N} '&/-]{2,}$/u;

function stripQuotes(s: string): string {
    if ((s.startsWith('"') && s.endsWith('"')) || (s.startsWith("'") && s.endsWith("'"))) {
        return s.slice(1, -1);
    }
    return s;
}

function basename(path: string): string {
    return path.split('/').pop()!.replace(/\.base$/, '');
}

export interface BaseDoc {
    title: string;
    text: string;
}

// Extract the searchable document from a .base file's raw YAML. `title` is the
// basename (the caller passes it to the chunker as the note title, which the
// ranker title-boosts); `text` is the deduped filter literals + meaningful view
// names, space-joined. `text` may be empty (a base with only property filters and
// generic view names) — the chunker's title-only fallback still indexes it by name.
export function extractBaseDoc(raw: string, path: string): BaseDoc {
    const title = basename(path);
    const literals = new Set<string>();
    const viewNames = new Set<string>();

    let section = '';
    for (const line of raw.split('\n')) {
        const top = TOP_KEY_RE.exec(line);
        if (top) section = top[1];
        if (SKIP_SECTIONS.has(section)) continue;

        // A `name:` line is a view name, never a filter literal — capture and move on.
        const nameM = NAME_RE.exec(line);
        if (nameM) {
            const v = stripQuotes(nameM[1].trim());
            if (v && !GENERIC_VIEW_NAMES.has(v.toLowerCase())) viewNames.add(v);
            continue;
        }

        let m: RegExpExecArray | null;
        QUOTED_RE.lastIndex = 0;
        while ((m = QUOTED_RE.exec(line)) !== null) {
            const v = m[1].trim();
            if (VALUE_RE.test(v) && !/^\d+$/.test(v)) literals.add(v);
        }
    }

    const text = [...literals, ...viewNames].join(' ');
    return { title, text };
}
