// Tests for the pure pill-validation helpers in query-field.ts. The field
// itself (PillQueryField) is a contenteditable DOM component with no jsdom
// environment configured for this suite (see vitest.config.mts) — these tests
// cover only the extracted pure logic, not rendering.

import { describe, it, expect } from 'vitest';
import { propPillNumericError } from './query-field';

describe('propPillNumericError (audit R2 #10)', () => {
    const numericKeys = new Set(['price']);
    const isNumericKey = (k: string): boolean => numericKeys.has(k);

    it('non-comparison values never error (no leading operator)', () => {
        expect(propPillNumericError('context', 'work', isNumericKey)).toBe(false);
    });

    it('a comparison on a non-Number key errors (existing D3 behavior)', () => {
        expect(propPillNumericError('pageType', '>50', isNumericKey)).toBe(true);
    });

    it('a well-formed comparison on a Number key does not error', () => {
        expect(propPillNumericError('price', '>50', isNumericKey)).toBe(false);
        expect(propPillNumericError('price', '<200', isNumericKey)).toBe(false);
        expect(propPillNumericError('price', '=50.00', isNumericKey)).toBe(false);
    });

    it('an unparseable numeric literal on a Number key errors too (the fix)', () => {
        // Previously this rendered as a healthy pill even though parseNum in
        // query-parser.ts fails on the stray comma, hard-zeroing the query.
        expect(propPillNumericError('price', '>49,99', isNumericKey)).toBe(true);
        expect(propPillNumericError('price', '>abc', isNumericKey)).toBe(true);
    });

    it('no key means no error (defensive)', () => {
        expect(propPillNumericError(undefined, '>50', isNumericKey)).toBe(false);
    });
});
