// Production entry for the Seek engine (WGSL; compute option 'wgsl').
// Bundled to an IIFE string by esbuild.config.mjs → __WGSL_ENGINE_SRC__ and
// inlined at the top of the runtime iframe's child script, where loadWgsl()
// (src/iframe-runner.ts) picks it up from globalThis.
import { loadGraniteWeights, detachEmbedding } from './onnx-weights.js';
import { WgslEncoder } from './engine-webgpu.js';

globalThis.__seekWgsl = { loadGraniteWeights, detachEmbedding, WgslEncoder };
