import { describe, it, expect } from 'vitest';
import { extractBaseDoc } from './base-extractor';

// Real-shape fixtures lifted from the vault's .base files.
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

// The pathological case: ~30 formula bodies full of literals like "Overdue",
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

describe('extractBaseDoc', () => {
    it('titles the doc by basename, minus the .base extension', () => {
        expect(extractBaseDoc(READING_LIST, 'Bases/Reading list.base').title).toBe('Reading list');
        expect(extractBaseDoc(PLACES, 'a/b/Places.base').title).toBe('Places');
    });

    it('pulls filter string literals, drops operators and property paths', () => {
        const { text } = extractBaseDoc(READING_LIST, 'Reading list.base');
        expect(text).toContain('writing');
        expect(text).toContain('books');
        expect(text).not.toContain('category');     // property path, not content
        expect(text).not.toContain('file.tags');
        expect(text).not.toContain('346');          // layout config
    });

    it('keeps multi-word and slashed literals intact', () => {
        const { text } = extractBaseDoc(ALEX_1X1S, 'Alex 1x1s.base');
        expect(text).toContain('meetings/1x1s');
        expect(text).toContain('Alex');
    });

    it('skips summaries blocks (no "null" leak)', () => {
        const { text } = extractBaseDoc(ALEX_1X1S, 'Alex 1x1s.base');
        expect(text).not.toContain('null');
    });

    it('keeps meaningful view names but drops generic ones', () => {
        const { text } = extractBaseDoc(PLACES, 'Places.base');
        expect(text).toContain('places');
        expect(text).toContain('Map');              // meaningful view name
        expect(text).not.toContain('View');         // generic
        expect(text).not.toContain('Table');        // generic
    });

    it('rejects expression literals like !coordinates.isEmpty()', () => {
        const { text } = extractBaseDoc(PLACES, 'Places.base');
        expect(text).not.toContain('coordinates');
        expect(text).not.toContain('isEmpty');
    });

    it('does not leak formula-body literals (the agenda swamp)', () => {
        const { text } = extractBaseDoc(AGENDA, 'agenda.base');
        for (const noise of ['none', 'high', 'No due date', 'Overdue', 'This week', 'Quick', 'priorityWeight']) {
            expect(text).not.toContain(noise);
        }
        expect(text).toContain('task');             // the one real filter literal
        expect(text).toContain('Agenda');           // view name, declared AFTER formulas
    });

    it('returns empty text when a base has only generic views and no literals', () => {
        const { title, text } = extractBaseDoc('views:\n  - type: table\n    name: Table\n', 'Empty.base');
        expect(title).toBe('Empty');
        expect(text).toBe('');
    });
});
