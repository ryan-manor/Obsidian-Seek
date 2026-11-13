import { describe, it, expect } from 'vitest';
import { WGSL_SELFTEST } from './wgsl-selftest';
import { ACTIVE_MODEL_SPEC } from './model-registry';
import { buildChildScript } from './iframe-runner';

describe('Seek engine load-time self-test', () => {
    it('reference is for the shipped model (regenerate it on a model bump)', () => {
        expect(WGSL_SELFTEST.modelId).toBe(ACTIVE_MODEL_SPEC.repo);
        expect(WGSL_SELFTEST.revision).toBe(ACTIVE_MODEL_SPEC.revision);
        expect(WGSL_SELFTEST.vec.length).toBe(ACTIVE_MODEL_SPEC.dim);
        const norm = Math.hypot(...WGSL_SELFTEST.vec);
        expect(norm).toBeCloseTo(1, 4);
    });

    it('is injected into the iframe, alongside the CPU-only fallback', () => {
        const s = buildChildScript('https://cdn.example/x.js', ACTIVE_MODEL_SPEC.dim);
        expect(s).toContain('const WGSL_SELFTEST = {');
        expect(s).toContain("requestedDevice === 'wgsl-wasm' ? 'wasm' : 'auto'");
    });
});
