import { describe, it, expect } from 'vitest';
import { extractBaseDocs } from './base-extractor';
import type { BaseView } from './types';

// Real-shape fixtures lifted from the vault's .base files.

// The worked example from the plan: top-level folder filter inherited by every
// view, five non-generic views, the default "All Clippings" view carrying no own
// filter. "fashion clips" should route to Clothing — impossible with one mashed doc.
const CLIPPINGS = `filters:
  and:
    - file.inFolder("Clippings")
views:
  - type: cards
    name: All Clippings
    order:
      - file.name
  - type: cards
    name: Products
    filters:
      and:
        - category == "product"
  - type: list
    name: Posts
    filters:
      and:
        - category == "writing"
  - type: cards
    name: Clothing
    filters:
      and:
        - category == "product"
        - subcategory.contains("Clothing")
        - read != true
  - type: list
    name: Wikis
    filters:
      and:
        - pageSource == "Team Wiki"
`;

const READING_LIST = `filters:
  or:
    - category == "writing"
    - file.tags.contains("books")
views:
  - type: table
    name: Table
    order:
      - read
      - author
    columnSize:
      file.name: 346
`;

const ALEX_1X1S = `summaries:
  Filled: values.filter(!value.isType("null")).length
views:
  - type: table
    name: Table
    filters:
      and:
        - pageType == "meetings/1x1s"
        - file.name.contains("Alex")
`;

const PLACES = `filters:
  and:
    - pageType == "places"
views:
  - type: map
    name: Map
    filters:
      and:
        - "!coordinates.isEmpty()"
  - type: table
    name: View
`;

// The pathological case: formula bodies full of literals like "Overdue",
// "Quick (<30m)", "No due date" that must NOT leak into the index.
const AGENDA = `filters:
  and:
    - file.hasTag("task")
formulas:
  priorityWeight: if(priority=="none",0,if(priority=="high",3,999))
  dueDateCategory: if(!due, "No due date", if(date(due) < today(), "Overdue", "This week"))
  timeEstimateCategory: if(timeEstimate < 30, "Quick (<30m)", "Long (>2h)")
views:
  - type: tasknotesCalendar
    name: Agenda
`;

const base = (docs: BaseView[]): BaseView => docs.find(d => d.viewName === null)!;
const view = (docs: BaseView[], name: string): BaseView | undefined => docs.find(d => d.viewName === name);

describe('extractBaseDocs', () => {
    it('emits a base-level entry plus one per non-generic view', () => {
        const docs = extractBaseDocs(CLIPPINGS, 'Bases/Clippings.base');
        expect(docs.map(d => d.viewName)).toEqual([null, 'All Clippings', 'Products', 'Posts', 'Clothing', 'Wikis']);
    });

    it('routes a view by its own filter literal AND its name (the Clothing case)', () => {
        const docs = extractBaseDocs(CLIPPINGS, 'Bases/Clippings.base');
        const clothing = view(docs, 'Clothing')!;
        expect(clothing.content).toContain('Clothing');   // view name
        expect(clothing.content).toContain('product');    // own filter literal
        expect(clothing.content).not.toContain('writing'); // a SIBLING view's literal must not bleed in
    });

    it('inherits top-level filter literals into every view and the base entry', () => {
        const docs = extractBaseDocs(CLIPPINGS, 'Bases/Clippings.base');
        for (const d of docs) expect(d.content).toContain('Clippings'); // file.inFolder("Clippings")
    });

    it('keeps each view content distinct (Posts→writing, Wikis→Team Wiki)', () => {
        const docs = extractBaseDocs(CLIPPINGS, 'Bases/Clippings.base');
        expect(view(docs, 'Posts')!.content).toContain('writing');
        expect(view(docs, 'Wikis')!.content).toContain('Team Wiki');
        expect(view(docs, 'Products')!.content).toContain('product');
    });

    it('pulls filter string literals, drops operators / property paths / layout', () => {
        const docs = extractBaseDocs(READING_LIST, 'Reading list.base');
        const c = base(docs).content;       // generic-only base → literals fold to base level
        expect(c).toContain('writing');
        expect(c).toContain('books');
        expect(c).not.toContain('category');
        expect(c).not.toContain('file.tags');
        expect(c).not.toContain('346');
    });

    it('folds a generic view\'s own literals into the base-level entry', () => {
        // ALEX's only view is a generic "Table" — its filter literals still describe
        // the base, so they land on the base-level entry, not a dropped view.
        const docs = extractBaseDocs(ALEX_1X1S, 'Alex 1x1s.base');
        expect(docs.map(d => d.viewName)).toEqual([null]);
        const c = base(docs).content;
        expect(c).toContain('meetings/1x1s');
        expect(c).toContain('Alex');
        expect(c).not.toContain('null');   // summaries never read
    });

    it('keeps meaningful view names, drops generic ones, rejects expression literals', () => {
        const docs = extractBaseDocs(PLACES, 'Places.base');
        expect(docs.map(d => d.viewName)).toEqual([null, 'Map']); // "View" (generic) dropped
        expect(base(docs).content).toContain('places');
        const map = view(docs, 'Map')!.content;
        expect(map).toContain('Map');
        expect(map).toContain('places');       // inherited
        expect(map).not.toContain('coordinates'); // "!coordinates.isEmpty()" rejected
        expect(map).not.toContain('isEmpty');
    });

    it('does not leak formula-body literals (the agenda swamp)', () => {
        const docs = extractBaseDocs(AGENDA, 'agenda.base');
        const all = docs.map(d => d.content).join(' ');
        for (const noise of ['none', 'high', 'No due date', 'Overdue', 'This week', 'Quick', 'priorityWeight']) {
            expect(all).not.toContain(noise);
        }
        expect(view(docs, 'Agenda')!.content).toContain('task');   // the one real filter literal, inherited
        expect(view(docs, 'Agenda')!.content).toContain('Agenda'); // view name, declared AFTER formulas
    });

    it('always returns a non-empty base-level entry, even for an empty/generic-only base', () => {
        const docs = extractBaseDocs('views:\n  - type: table\n    name: Table\n', 'Empty.base');
        expect(docs).toHaveLength(1);
        expect(docs[0].viewName).toBeNull();
        expect(docs[0].content).toBe('Empty');   // just the base name — never empty
    });

    it('degrades to a base-level entry on malformed YAML', () => {
        const docs = extractBaseDocs(':\n  bad: : :\n\t- nope', 'Broke.base');
        expect(docs).toHaveLength(1);
        expect(docs[0].viewName).toBeNull();
        expect(docs[0].content).toBe('Broke');
    });

    it('degrades (never throws) on structurally-odd shapes — non-array views / branches', () => {
        // Parseable YAML whose shapes violate the Bases schema (only reachable via a
        // hand-edited / corrupt .base): a non-array `views`, a null view element, and
        // a non-array and/or/not branch must all fold to a lone base-level entry, not
        // throw and lose the file.
        for (const raw of [
            'views: 5',                 // views is a scalar
            'views:\n  bad: map',       // views is a mapping
            'views:\n  - null',         // null view element
            'filters:\n  and: 7',       // branch is a scalar
            'filters: 3',               // filters is a scalar leaf
        ]) {
            const docs = extractBaseDocs(raw, 'Weird.base');
            expect(docs).toHaveLength(1);
            expect(docs[0].viewName).toBeNull();
            expect(docs[0].content).toBe('Weird');
        }
    });
});
