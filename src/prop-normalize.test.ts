// Parity tests for the two property-value surfaces. The whole point of naming
// bind-form and display-form is that no site re-derives an unwrap by accident
// (the "Notes Personal Places Zurich Zurich" stuffing class). These tests pin
// the contract for both forms and the invariants that distinguish them.

import { describe, it, expect } from 'vitest';
import { toBindForm, toDisplayForm } from './prop-normalize';

describe('toDisplayForm — canonical note name (basename only)', () => {
    it('passes plain scalar values through unchanged', () => {
        expect(toDisplayForm('restaurants')).toBe('restaurants');
        expect(toDisplayForm('San Francisco')).toBe('San Francisco');
    });

    it('unwraps a bare wikilink to its name', () => {
        expect(toDisplayForm('[[Austin]]')).toBe('Austin');
    });

    it('strips the folder PATH from a path-form link (the live regression)', () => {
        // The exact incident value: must NOT leak "Notes"/"Personal"/"Places".
        expect(toDisplayForm('[[Notes/Personal/Places/Zurich|Zurich]]')).toBe('Zurich');
    });

    it('drops the alias, keeping the target basename', () => {
        // Per the agreed policy: alias is display chrome; index the note name.
        expect(toDisplayForm('[[Jane Doe|Alex]]')).toBe('Jane Doe');
        expect(toDisplayForm('[[San Francisco|SF]]')).toBe('San Francisco');
    });

    it('drops #heading and ^block refs', () => {
        expect(toDisplayForm('[[Project Atlas#Status]]')).toBe('Project Atlas');
        expect(toDisplayForm('[[Meeting Notes^abc123]]')).toBe('Meeting Notes');
        expect(toDisplayForm('[[Notes/Foo/Bar|Bar#Sec]]')).toBe('Bar');
    });

    it('handles multiple links and mixed plain+link text', () => {
        expect(toDisplayForm('[[Austin]] [[Notes/Places/Round Rock|Round Rock]]'))
            .toBe('Austin Round Rock');
        expect(toDisplayForm('lunch at [[Musashino Sushi Dokoro]]'))
            .toBe('lunch at Musashino Sushi Dokoro');
    });

    it('never emits wikilink/path syntax', () => {
        for (const v of [
            '[[Notes/Personal/Places/Zurich|Zurich]]',
            '[[A/B/C|D]]',
            '[[Foo#Bar^baz]]',
        ]) {
            const out = toDisplayForm(v);
            expect(out).not.toMatch(/[[\]|/]/); // no [ ] | /
        }
    });
});

describe('toBindForm — substring-matchable filter surface', () => {
    it('lowercases and keeps PATH, target, and alias all matchable', () => {
        const out = toBindForm('[[Notes/Personal/Places/Zurich|Zurich]]');
        // breadth is the feature: every token a user might filter by survives
        expect(out).toContain('notes');
        expect(out).toContain('places');
        expect(out).toContain('zurich');
    });

    it('flattens a bare link for typed-value binding', () => {
        expect(toBindForm('[[Los Angeles]]')).toBe('los angeles');
    });
});

describe('the two surfaces diverge exactly on path/alias breadth', () => {
    const pathForm = '[[Notes/Personal/Places/Zurich|Zurich]]';
    it('bind-form keeps path tokens; display-form does not', () => {
        expect(toBindForm(pathForm)).toContain('personal');
        expect(toDisplayForm(pathForm).toLowerCase()).not.toContain('personal');
    });
});
