// node test-cpu.mjs <model_dir>  — JS CPU reference vs onnxruntime fixtures.
import { readFileSync } from 'node:fs';
import { loadGraniteWeights } from './onnx-weights.js';
import { forwardCpu } from './engine-cpu.js';

const md = process.argv[2];
const config = JSON.parse(readFileSync(md + '/config.json', 'utf8'));
const t0 = performance.now();
const weights = loadGraniteWeights(readFileSync(md + '/onnx/model_q4.onnx'), config);
console.log('parsed weights in', (performance.now() - t0).toFixed(0), 'ms');
const fx = JSON.parse(readFileSync(new URL('./fixtures.json', import.meta.url), 'utf8'));
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
let worst = 1;
for (const f of fx) {
    const t = performance.now();
    const v = forwardCpu(weights, config, f.ids);
    const c32 = dot(v, f.fp32), cship = dot(v, f.shipped);
    worst = Math.min(worst, c32);
    console.log(`S=${String(f.ids.length).padStart(3)} cos(fp32 ORT)=${c32.toFixed(7)} cos(shipped ORT)=${cship.toFixed(6)} ${(performance.now() - t).toFixed(0)}ms`);
}
console.log('WORST vs fp32 ORT', worst.toFixed(7));
if (worst < 0.99999) process.exit(1);
