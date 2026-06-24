import { describe, it, expect } from 'vitest';
import { indexBannerSpec, INDEX_STALE_MSG } from './index-notice';

describe('indexBannerSpec', () => {
    it('returns null for a healthy index', () => {
        expect(indexBannerSpec('healthy', null)).toBeNull();
    });

    it('returns null while drift recovery is in progress (recovering ≠ degraded)', () => {
        expect(indexBannerSpec('recovering', 'version')).toBeNull();
    });

    it('returns null for a drift degradation — that is not a version change', () => {
        expect(indexBannerSpec('degraded', 'drift')).toBeNull();
    });

    it('returns null for a reasonless degradation (e.g. the drained heal)', () => {
        expect(indexBannerSpec('degraded', null)).toBeNull();
    });

    it('returns the shared stale message for a version degradation (any platform)', () => {
        const spec = indexBannerSpec('degraded', 'version');
        expect(spec).not.toBeNull();
        expect(spec!.message).toBe(INDEX_STALE_MSG);
    });
});
