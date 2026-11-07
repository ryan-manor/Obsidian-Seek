// node run-bench-child-test.cjs <model_dir> <transformers_dist_dir>
// End-to-end check of the BUNDLED bench child (what the Obsidian command
// injects): HF + jsdelivr requests are routed to local files, so this runs
// offline. SwiftShader WebGPU → numerics real, timings meaningless.
const path = require('node:path'), fs = require('node:fs'), http = require('node:http');
const esbuild = require('esbuild');
const { chromium } = require('playwright');
const [modelDir, txDist] = process.argv.slice(2).map((p) => path.resolve(p));
(async () => {
  const src = (await esbuild.build({ entryPoints: [path.join(__dirname, 'bench-child.js')], bundle: true, format: 'iife', write: false, logLevel: 'silent' })).outputFiles[0].text;
  const srv = http.createServer((q, r) => { r.writeHead(200, { 'content-type': 'text/html' }); r.end('<!doctype html><body>host</body>'); }).listen(0, '127.0.0.1');
  await new Promise((r) => srv.on('listening', r));
  const browser = await chromium.launch({ args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=swiftshader', '--use-webgpu-adapter=swiftshader', '--disable-vulkan-surface'] });
  const ctx = await browser.newContext();
  await ctx.route('https://huggingface.co/**', (route) => {
    const u = new URL(route.request().url());
    const file = u.pathname.split('/resolve/')[1].split('/').slice(1).join('/');
    const p = path.join(modelDir, file);
    if (!fs.existsSync(p)) return route.fulfill({ status: 404 });
    route.fulfill({ status: 200, body: fs.readFileSync(p), headers: { 'access-control-allow-origin': '*' } });
  });
  await ctx.route('https://cdn.jsdelivr.net/**', (route) => {
    const u = new URL(route.request().url());
    const rest = u.pathname.split('/').slice(4).join('/') || 'dist/transformers.min.js';
    const p = path.join(txDist, '..', rest.startsWith('dist/') ? rest : 'dist/transformers.min.js');
    route.fulfill({ status: 200, body: fs.readFileSync(p), headers: { 'content-type': 'text/javascript', 'access-control-allow-origin': '*' } });
  });
  const page = await ctx.newPage();
  page.on('console', (m) => console.log('[page]', m.text()));
  await page.goto(`http://127.0.0.1:${srv.address().port}/`);
  const result = await page.evaluate((code) => new Promise((resolve) => {
    const f = document.createElement('iframe');
    window.addEventListener('message', (ev) => {
      if (ev.data?.type === 'wgsl-bench-ready') f.contentWindow.postMessage({ type: 'wgsl-bench-run', repo: 'tooape/granite-embedding-97m-multilingual-r2-GBQ4-ONNX', revision: '54db88c5667bd79b4aea24ea6027a7ef45a7bbb5', cdnUrl: 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0', queries: ['street photography in low light', 'Q3 roadmap'], reps: 2 }, '*');
      if (ev.data?.type === 'wgsl-bench-result') resolve(ev.data);
    });
    document.body.appendChild(f);
    f.srcdoc = '<!DOCTYPE html><html><body><script type="module">' + code + '</scr' + 'ipt></body></html>';
  }), src);
  await browser.close(); srv.close();
  if (result.error) { console.log('ERROR', result.error); process.exit(1); }
  const r = result.result;
  const fx = JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures.json'), 'utf8'));
  const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
  console.log(JSON.stringify({ adapter: r.adapter, modelSource: r.modelSource, steps: r.steps, engine: r.engine, batch8: r.batch8 }));
  for (const q of r.queries) {
    const f = fx.find((x) => x.text === q.text);
    console.log(q.text, 'tokens', q.tokens, f ? 'cos(fp32 ORT)=' + dot(q.vector, f.fp32).toFixed(7) : '(no fixture)');
  }
})();
