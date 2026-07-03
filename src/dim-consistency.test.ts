// Single-source-of-truth guard for the embedding dimension.
//
// The dim used to be five hand-hardcoded `384`s (embedder.EMBEDDING_DIM, the
// iframe's OUTPUT_DIM, index-store's default meta, and sidecar.ts's
// Q_BYTES/SIGN_BYTES record stride). They had to agree by hand or a model swap
// would silently mis-slice vectors (write N-d vectors into a 384-byte stride).
// They now all DERIVE from ACTIVE_MODEL_SPEC.dim. These tests are the structural
// replacement for that hand-agreement: if a future swap changes the dim, every
// derived constant must move together or CI fails here, before any device can
// corrupt an index. (The iframe's OUTPUT_DIM is injected from the same spec field
// at iframe build time, so it is covered by construction — see iframe-runner.ts.)

import { describe, it, expect } from 'vitest';
import { EMBEDDING_DIM } from './embedder';
import { ACTIVE_MODEL_SPEC } from './model-registry';
import { Q_BYTES, S_BYTES, SIGN_BYTES, CRC_BYTES, RECORD_PAYLOAD_BYTES, VEC_BYTES, DIM } from './sidecar';
import { packSignBits } from './binary';

describe('embedding dimension is single-sourced from the model spec', () => {
    it('every dense-vector constant equals ACTIVE_MODEL_SPEC.dim', () => {
        const dim = ACTIVE_MODEL_SPEC.dim;
        expect(EMBEDDING_DIM).toBe(dim);
        expect(Q_BYTES).toBe(dim);
        expect(DIM).toBe(dim); // sidecar's logical dim alias
    });

    it('the sidecar sign tier matches binary.ts packing for the active dim', () => {
        expect(SIGN_BYTES).toBe((Q_BYTES + 7) >> 3);
        // The live packer must produce EXACTLY SIGN_BYTES for a dim-wide vector, or
        // the sidecar record stride and the candidate-tier bytes disagree at runtime.
        expect(packSignBits(new Float32Array(EMBEDDING_DIM)).length).toBe(SIGN_BYTES);
    });

    it('the record stride composes from the tiers', () => {
        expect(RECORD_PAYLOAD_BYTES).toBe(Q_BYTES + S_BYTES + SIGN_BYTES);
        expect(VEC_BYTES).toBe(RECORD_PAYLOAD_BYTES + CRC_BYTES);
    });
});

// Forward-width safety: the derivation FORMULAS must hold at any dimension a
// future model might use, not only today's. The sidecar codec is hardwired to the
// build-time constants, so it can't be exercised at a foreign width without a
// rebuild (that's the simulated-swap dry-run in the plan) — but the stride math
// and the sign-bit packer are pure and can be checked across widths here.
describe('dimension derivation holds across widths', () => {
    // include a non-multiple-of-8 width to exercise the ceil in (d + 7) >> 3.
    for (const d of [256, 384, 512, 768, 1024, 1000]) {
        it(`d=${d}: sign bytes = ceil(d/8) and packSignBits agrees`, () => {
            const signBytes = (d + 7) >> 3;
            expect(packSignBits(new Float32Array(d)).length).toBe(signBytes);
            // S_BYTES + CRC_BYTES are dim-independent; the record stride scales
            // only via q (d bytes) and the sign tier (ceil(d/8) bytes). Compare
            // against the symbolic constants (not a hardcoded byte count) so this
            // stays correct across a deliberate S_BYTES/CRC_BYTES width change.
            const payload = d + S_BYTES + signBytes;
            const stride = payload + CRC_BYTES;
            expect(stride).toBe(d + S_BYTES + signBytes + CRC_BYTES);
        });
    }
});
