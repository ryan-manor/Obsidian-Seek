// src/prop-types.ts — read Obsidian's property-type registry.
//
// Obsidian records each frontmatter property's declared type (Text / Number /
// Date / Date & time / …) in a registry that is NOT part of the public
// obsidian.d.ts. Both the recency date-field picker (settings tab) and the
// typed-value inline filters (numeric `[price>50]`, date `before:`/`after:`)
// key off it, so the registry access lives here once, feature-detected across
// the surfaces Obsidian has shipped it under and failing SOFT to an empty set —
// never a free-text fallback where a typo'd / mistyped property could leak in.
//
// Type detection is registry-only by design (Decision D1, "Seek Typed-Value
// Filters Design"): a property is Number iff the user declared it Number in
// Obsidian. No value-sniffing — so a `version: "3"` text field never silently
// becomes a numeric filter, at the cost of one setup step per numeric property.

import type { App } from 'obsidian';

type PropertyTypeInfo = { name?: string; type?: string; widget?: string };

// Walk the first registry surface that yields data, mapping its entries through
// `accept` (tested against the lowercased type/widget). Each source is wrapped
// in try/catch so an internal API that throws (or moves) degrades to the next,
// then to []. `info.name` is preferred over the record key so the registry's own
// canonical casing wins (the key is sometimes lowercased).
function enumeratePropertyNames(app: App, accept: (type: string) => boolean): Set<string> {
    const a = app as unknown as {
        metadataTypeManager?: {
            getAllProperties?: () => Record<string, PropertyTypeInfo>;
            types?: Record<string, PropertyTypeInfo>;
        };
        metadataCache?: { getAllPropertyInfos?: () => Record<string, PropertyTypeInfo> };
    };
    const sources: Array<() => Record<string, PropertyTypeInfo> | undefined> = [
        () => a.metadataTypeManager?.getAllProperties?.(),
        () => a.metadataTypeManager?.types,
        () => a.metadataCache?.getAllPropertyInfos?.(),
    ];
    for (const src of sources) {
        let rec: Record<string, PropertyTypeInfo> | undefined;
        try { rec = src(); } catch { continue; }
        if (!rec || Object.keys(rec).length === 0) continue;
        const out = new Set<string>();
        for (const [key, info] of Object.entries(rec)) {
            const t = (info?.type ?? info?.widget ?? '').toLowerCase();
            if (accept(t)) out.add(info?.name ?? key);
        }
        return out;
    }
    return new Set();
}

// Date / Date & time properties, sorted — backs the recency date-field picker.
export function enumerateDatePropertyNames(app: App): string[] {
    return [...enumeratePropertyNames(app, t => t === 'date' || t === 'datetime')]
        .sort((x, y) => x.localeCompare(y));
}

// Number properties — backs the numeric inline filter (`[price>50]`) type gate
// and the `[` key-menu's numeric-key completion. A Set because membership is the
// only question the parser/matcher/suggester ask.
export function enumerateNumberPropertyNames(app: App): Set<string> {
    return enumeratePropertyNames(app, t => t === 'number');
}
