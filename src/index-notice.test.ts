import { describe, it, expect } from 'vitest';
import { indexBannerSpec, INDEX_STALE_MSG, INDEX_SYNCING_MSG, INDEX_PEER_AHEAD_MSG } from './index-notice';

describe('indexBannerSpec', () => {
    it('returns null for a healthy index', () => {
        expect(indexBannerSpec('healthy', null)).toBeNull();
    });

    it('returns null for a drift degradation — that is not a version change', () => {
        expect(indexBannerSpec('degraded', 'drift')).toBeNull();
    });

    it('returns null for a reasonless degradation (e.g. the drained heal)', () => {
        expect(indexBannerSpec('degraded', null)).toBeNull();
    });

    it('returns null while drift recovery runs with no version reason (recovering + null)', () => {
        expect(indexBannerSpec('recovering', null)).toBeNull();
    });

    it('returns the warning stale banner (with action) for a degraded version mismatch', () => {
        const spec = indexBannerSpec('degraded', 'version');
        expect(spec).not.toBeNull();
        expect(spec!.message).toBe(INDEX_STALE_MSG);
        expect(spec!.tone).toBe('warn');
        expect(spec!.showAction).toBe(true);
    });

    it('returns the calm syncing banner (no action) only when a peer index is actually on its way (peerSyncPending=true)', () => {
        const spec = indexBannerSpec('recovering', 'version', true);
        expect(spec).not.toBeNull();
        expect(spec!.message).toBe(INDEX_SYNCING_MSG);
        expect(spec!.tone).toBe('info');
        expect(spec!.showAction).toBe(false);
    });

    // Regression (relevance/UX audit 2026-06-29): the LOCAL drift-recovery ladder sets
    // indexHealth='recovering' over a version-stale index with NO peer. The banner must
    // NOT claim "syncing from another device" then — it's a single-device vault. Driving
    // the syncing banner off peerSyncPending (not health) keeps this cell silent.
    it('stays silent during local drift recovery over a version-stale index with no peer (recovering + version, peerSyncPending=false)', () => {
        expect(indexBannerSpec('recovering', 'version', false)).toBeNull();
        expect(indexBannerSpec('recovering', 'version')).toBeNull(); // default arg = no peer
    });

    it('lets the peer signal dominate health — a degraded version index with a peer still reads as syncing', () => {
        expect(indexBannerSpec('degraded', 'version', true)?.message).toBe(INDEX_SYNCING_MSG);
    });

    it('never lets the peer signal override a non-version reason (drift stays silent)', () => {
        expect(indexBannerSpec('recovering', 'drift', true)).toBeNull();
    });

    it('returns the "update Seek" warning (no reindex button) when a peer index is newer (peer-ahead)', () => {
        const spec = indexBannerSpec('degraded', 'peer-ahead');
        expect(spec).not.toBeNull();
        expect(spec!.message).toBe(INDEX_PEER_AHEAD_MSG);
        expect(spec!.tone).toBe('warn');
        expect(spec!.showAction).toBe(false);
    });

    it('peer-ahead is independent of health (the local index matches the build)', () => {
        // Set from orchestrator.peerAhead, always paired with 'degraded' in practice, but
        // the reason alone decides the banner — so it must not depend on the health value.
        expect(indexBannerSpec('healthy', 'peer-ahead')?.message).toBe(INDEX_PEER_AHEAD_MSG);
    });
});
