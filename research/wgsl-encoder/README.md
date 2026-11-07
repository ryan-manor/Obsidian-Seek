# Investigation: faster / lighter model execution on iOS

Branch-only investigation (2026-11-07). Nothing here is wired into the default
product path. Two prototypes:

- **Prototype 2: query-only WebGPU** (in `src/`, opt-in per device).
- **Prototype 3: hand-written WebGPU (WGSL) encoder** (this folder, plus an
  experimental bench command).

## Why not Metal directly?

Seek is an Obsidian plugin, so on iOS it is JavaScript inside Obsidian's
WKWebView. Plugins cannot ship or call native code (no Swift, Metal, Core ML or
ANE). WebGPU is the only route to the GPU, and WebKit implements WebGPU on top
of Metal. Prototype 3 is as close to "using Metal directly" as a plugin can get:
our own compute shaders, with no ONNX Runtime in between.

## Prototype 2: query-only WebGPU (`webgpu-query`)

Settings → Seek → Compute → **"Search on GPU"** (experimental, per device,
stored in localStorage, never synced).

- Search embeds run on ORT-Web WebGPU. Warmup compiles only the 7 query shapes
  (batch 1 × `QUERY_SEQ_BUCKETS`) instead of ~74.
- Before any indexing batch, the session swaps to WASM. It reuses `recycle()`,
  so the iframe and the GPU device are torn down first, and two loads never run
  at once. The next fresh load (after idle unload) goes back to WebGPU.
- The active backend is restamped to `wasm` on swap, so a crash during CPU
  indexing can't trip the WebGPU demote.
- Logged: `gpuQuerySwap: {from, swapMs}` on the replacement load entry, and the
  forensics beat `gpu-query-swap`.

Risks: iPhone now takes ORT's WebKit WebGPU path, which brings the jspi/asyncify
glue, the ORT #26827 compile-hang class and the 4 GiB memory reservation (capped
at 512 MiB). The first indexing batch in a GPU session also pays a full WASM
model load.

What to measure on iPhone: `actualDevice`, `webgpuError`, `glue`,
`coldStartMs` and `warmupMs` on the load entry; per-query `iframeLatencyMs`
against "Force CPU"; `gpuQuerySwap.swapMs`; device-lost events.

## Prototype 3: hand-written WGSL encoder

| file | what |
|---|---|
| `onnx-weights.js` | ~150-line protobuf walker. Reads weights **straight from the shipped `model_q4.onnx`** (same bytes, same cache entry; no second model, no re-export). Zero-copy views. |
| `engine-cpu.js` | Plain-JS float32 reference (debug twin). |
| `engine-webgpu.js` | The engine: 4 WGSL kernels (layernorm, int4 matmul with optional fused GLU + residual, RoPE, masked attention), 97 dispatches per forward in one compute pass. |
| `ref_numpy.py`, `gen_fixtures.py` | numpy reference + onnxruntime ground truth (`fixtures.json`). |
| `test-cpu.mjs`, `run-webgpu-test.cjs`, `run-bench-child-test.cjs` | validation (Node; headless Chromium with SwiftShader WebGPU). |
| `bench-child.js` + `src/wgsl-bench.ts` | on-device bench: command palette → **"Seek: (Experimental) Benchmark WGSL encoder"**. |

### Verified here

- **Exact numerics.** Cosine vs ORT (fp32 activations) = **1.0000000** on 8
  fixtures (5–123 tokens, en/de/ja, sliding-window attention active at 123
  tokens, padded batch). The JS CPU reference matches too.
- **Production WASM is the outlier, not this engine.** The shipped graph sets
  `accuracy_level=4`, so ORT's CPU/WASM kernel quantizes activations to int8.
  That path is about 1.5e-4 off fp32 (cos 0.99982–0.99993). The WGSL engine
  matches the fp32 math, which is what ORT-WebGPU computes on desktop. So it
  sits *closer* to desktop vectors than today's iPhone path does. The index is
  unaffected.
- **Tokenizer ids:** transformers.js 4.2 `encode()` gives the same ids as the
  Python fixtures (8/8).
- **Footprint:** **20.7 MB of GPU memory** at the query sizing (128 rows):
  17.8 MB weights (int4, uploaded as-is) + 2.9 MB activations, allocated once.
  The 43 MB token-embedding table never goes to the GPU; the few rows a query
  needs are dequantized on the CPU. Parsing the model takes about 15–50 ms.
- **No per-shape compile.** Sequence length is a uniform, so the 4 pipelines
  compile once and serve every length. That removes the warmup grid, the
  fingerprint cache, padding to buckets (batch 1 runs at its exact length), and
  the shape-keyed buffer-pool growth behind the SafeInt overflow.
- **No wasm at all**, so none of: the 14–24 MB glue, the 4 GiB
  `WebAssembly.Memory` reservation, or the ORT #26827 WebKit compile hang.

### Not verified (needs the phone)

- **Speed.** SwiftShader is a CPU emulation of a GPU, so the timings here mean
  nothing. Rough expectation for a short query: about 18 MB of weights read once
  plus ~97 small dispatches in one submit, so a few ms on an A-series GPU if
  WebKit's per-dispatch overhead is small. Today's forward is about 20 ms. Treat
  this as a hypothesis until the bench runs.
- **Whether WKWebView in Obsidian exposes `navigator.gpu` on your iOS
  version,** and whether this lighter path avoids the jetsam kills that ORT's
  WebGPU path hit.

### Known tuning headroom (not done)

- The matmul runs one column per thread with 8-row tiles. Small-N layers
  (`Wo`, `Wo2`: N=384) launch only 6 workgroups, each looping over all of K.
  Splitting K, or using a subgroup/vec4 layout, would raise occupancy.
- Attention is one workgroup per (row, head) with keys in shared memory. That's
  fine for queries; for 512-token indexing chunks a tiled/flash layout would be
  better.
- The CPU embedding table (43 MB) could instead be read row by row from the
  cached Response blob, or moved to the GPU.
- fp16 (`shader-f16`) activations would halve activation bandwidth on Apple
  GPUs, but would move vectors off exact-fp32 parity. Measure before choosing.

### Running the bench on iPhone

1. Install this branch's build (`npm run build`, copy `main.js` /
   `manifest.json` / `styles.css`).
2. Search once so the model is cached, then make sure Seek isn't indexing.
3. Command palette → **Seek: (Experimental) Benchmark WGSL encoder**.
   - Runs the WGSL engine alone in its own iframe, then destroys it.
   - Then runs the production pipeline on the same 7 queries (7 reps each,
     query cache bypassed).
4. A note `Seek WGSL bench <timestamp>.md` opens with:
   - adapter info, and where the model bytes came from (cache/network)
   - cold-start breakdown and GPU MB
   - per-query WGSL vs production latency and cosine
   - batch-8 throughput

   Run it once with Compute = "Force CPU" and once with "Search on GPU" to
   compare all three paths.

### Reproducing the validation here

```sh
pip install onnx onnxruntime numpy tokenizers
# model files from huggingface.co/tooape/granite-embedding-97m-multilingual-r2-GBQ4-ONNX @ 54db88c…
python3 -I gen_fixtures.py <model_dir> fixtures.json
node test-cpu.mjs <model_dir>
node run-webgpu-test.cjs <model_dir>
node run-bench-child-test.cjs <model_dir> <node_modules/@huggingface/transformers/dist>
```
