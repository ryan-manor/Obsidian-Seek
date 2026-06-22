// Unit tests for the production model-delivery registry: active-spec selection
// (debug override vs shipped default) and the parent-side Cache-API eviction that
// reclaims a previous model's ~100 MB on a switch. Pure — no DOM/iframe; the Cache
// API is a structural fake injected as CacheStorage.

import { describe, it, expect } from 'vitest';
import type { SeekSettings } from './types';
import {
    ACTIVE_MODEL_SPEC,
    ML97_GBQ4,
    activeModelSpec,
    resolveOverrideSpec,
    shouldEvictCacheUrl,
    evictStaleModelCaches,
    isCacheUrlForRepo,
    deleteModelCaches,
} from './model-registry';

const ACTIVE_REPO = ML97_GBQ4.repo;
// Only modelRepoOverride / modelRevisionOverride are read; the rest is irrelevant.
const settings = (o: Partial<SeekSettings> = {}): SeekSettings => o as unknown as SeekSettings;

const hfUrl = (repo: string, file = 'onnx/model_q4.onnx') =>
    `https://huggingface.co/${repo}/resolve/main/${file}`;

describe('activeModelSpec / resolveOverrideSpec', () => {
    it('no override → shipped default spec', () => {
        expect(activeModelSpec(settings())).toBe(ACTIVE_MODEL_SPEC);
        expect(activeModelSpec(settings()).key).toBe(ML97_GBQ4.key);
    });

    it('empty / whitespace override → still the default', () => {
        expect(resolveOverrideSpec(settings({ modelRepoOverride: '' }))).toBeNull();
        expect(resolveOverrideSpec(settings({ modelRepoOverride: '   ' }))).toBeNull();
        expect(activeModelSpec(settings({ modelRepoOverride: '  ' }))).toBe(ACTIVE_MODEL_SPEC);
    });

    it('override repo wins and becomes identity (key === repo), trimmed', () => {
        const s = activeModelSpec(settings({ modelRepoOverride: '  acme/my-embed  ' }));
        expect(s.repo).toBe('acme/my-embed');
        expect(s.key).toBe('acme/my-embed');   // identity = override repo → drifts vs stored index
        expect(s.dim).toBe(ML97_GBQ4.dim);      // inherits the standard layout/dim/dtype
        expect(s.dtype).toBe(ML97_GBQ4.dtype);
    });

    it('revision override threads through (else null)', () => {
        expect(resolveOverrideSpec(settings({ modelRepoOverride: 'a/b' }))!.revision).toBeNull();
        expect(resolveOverrideSpec(settings({ modelRepoOverride: 'a/b', modelRevisionOverride: 'deadbeef' }))!.revision)
            .toBe('deadbeef');
    });
});

describe('shouldEvictCacheUrl', () => {
    it('keeps the active repo, evicts other HF repos', () => {
        expect(shouldEvictCacheUrl(hfUrl(ACTIVE_REPO), ACTIVE_REPO)).toBe(false);
        expect(shouldEvictCacheUrl(hfUrl(ACTIVE_REPO, 'config.json'), ACTIVE_REPO)).toBe(false);
        expect(shouldEvictCacheUrl(hfUrl('tooape/old-model'), ACTIVE_REPO)).toBe(true);
    });

    it('ignores non-HF / non-resolve URLs (jsdelivr runtime, etc.)', () => {
        expect(shouldEvictCacheUrl('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0', ACTIVE_REPO)).toBe(false);
        expect(shouldEvictCacheUrl('https://example.com/whatever', ACTIVE_REPO)).toBe(false);
    });

    it('matches the repo as a path segment (prefix-repo safety)', () => {
        // keepRepo is a prefix of another repo: the longer repo must still be evicted,
        // and the exact repo must still be kept.
        expect(shouldEvictCacheUrl(hfUrl('tooape/granite-2'), 'tooape/granite')).toBe(true);
        expect(shouldEvictCacheUrl(hfUrl('tooape/granite'), 'tooape/granite')).toBe(false);
    });
});

describe('isCacheUrlForRepo', () => {
    it('matches HF resolve URLs for the repo, ignores others', () => {
        expect(isCacheUrlForRepo(hfUrl(ACTIVE_REPO), ACTIVE_REPO)).toBe(true);
        expect(isCacheUrlForRepo(hfUrl(ACTIVE_REPO, 'config.json'), ACTIVE_REPO)).toBe(true);
        expect(isCacheUrlForRepo(hfUrl('tooape/other'), ACTIVE_REPO)).toBe(false);
        expect(isCacheUrlForRepo('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0', ACTIVE_REPO)).toBe(false);
    });

    it('is the exact inverse of shouldEvictCacheUrl for HF model URLs', () => {
        for (const url of [hfUrl(ACTIVE_REPO), hfUrl('tooape/granite-2'), hfUrl('tooape/granite')]) {
            expect(isCacheUrlForRepo(url, 'tooape/granite')).toBe(!shouldEvictCacheUrl(url, 'tooape/granite'));
        }
    });
});

// Structural CacheStorage fake — one named cache holding {url} request stand-ins.
function fakeCaches(urls: string[], present = true) {
    let reqs = urls.map(url => ({ url }));
    const cache = {
        keys: async () => reqs.slice(),
        delete: async (req: { url: string }) => {
            const before = reqs.length;
            reqs = reqs.filter(r => r.url !== req.url);
            return reqs.length < before;
        },
    };
    return {
        remaining: () => reqs.map(r => r.url),
        cs: {
            has: async () => present,
            open: async () => cache,
        } as unknown as CacheStorage,
    };
}

describe('evictStaleModelCaches', () => {
    it('deletes only stale-repo entries, keeps the active model + runtime', async () => {
        const f = fakeCaches([
            hfUrl(ACTIVE_REPO),
            hfUrl(ACTIVE_REPO, 'config.json'),
            hfUrl('tooape/old-model'),
            hfUrl('tooape/older-still', 'tokenizer.json'),
            'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0',
        ]);
        const res = await evictStaleModelCaches(f.cs, ACTIVE_REPO);
        expect(res.seen).toBe(5);
        expect(res.deleted).toBe(2);                 // two stale repos
        expect(f.remaining()).toContain(hfUrl(ACTIVE_REPO));
        expect(f.remaining()).toContain('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0');
        expect(f.remaining()).not.toContain(hfUrl('tooape/old-model'));
    });

    it('no-ops cleanly when the transformers cache is absent', async () => {
        const f = fakeCaches([], /* present */ false);
        expect(await evictStaleModelCaches(f.cs, ACTIVE_REPO)).toEqual({ seen: 0, deleted: 0 });
    });

    it('first-ever load (only active repo cached) deletes nothing', async () => {
        const f = fakeCaches([hfUrl(ACTIVE_REPO), hfUrl(ACTIVE_REPO, 'config.json')]);
        const res = await evictStaleModelCaches(f.cs, ACTIVE_REPO);
        expect(res).toEqual({ seen: 2, deleted: 0 });
    });
});

describe('deleteModelCaches', () => {
    it('deletes only the active repo, keeps other repos + runtime', async () => {
        const f = fakeCaches([
            hfUrl(ACTIVE_REPO),
            hfUrl(ACTIVE_REPO, 'config.json'),
            hfUrl('tooape/other-model'),
            'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0',
        ]);
        const res = await deleteModelCaches(f.cs, ACTIVE_REPO);
        expect(res.seen).toBe(4);
        expect(res.deleted).toBe(2);                            // both active-repo entries
        expect(f.remaining()).not.toContain(hfUrl(ACTIVE_REPO));
        expect(f.remaining()).toContain(hfUrl('tooape/other-model'));
        expect(f.remaining()).toContain('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0');
    });

    it('no-ops cleanly when the transformers cache is absent', async () => {
        const f = fakeCaches([], /* present */ false);
        expect(await deleteModelCaches(f.cs, ACTIVE_REPO)).toEqual({ seen: 0, deleted: 0 });
    });
});
