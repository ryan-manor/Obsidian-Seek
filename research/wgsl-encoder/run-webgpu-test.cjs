// node run-webgpu-test.cjs <model_dir> — validate the WGSL engine in headless
// Chromium (SwiftShader WebGPU: numerics are real, timings are NOT).
const http = require('node:http'), fs = require('node:fs'), path = require('node:path');
const { chromium } = require('playwright');
const modelDir = path.resolve(process.argv[2]);
const here = __dirname;
const types = { '.js': 'text/javascript', '.html': 'text/html', '.json': 'application/json' };
const srv = http.createServer((req, res) => {
  const u = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  const file = u.startsWith('/model/') ? path.join(modelDir, u.slice(7)) : path.join(here, u);
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404); return res.end(); }
    res.writeHead(200, { 'content-type': types[path.extname(file)] || 'application/octet-stream' });
    res.end(data);
  });
}).listen(0, '127.0.0.1', async () => {
  const port = srv.address().port;
  const browser = await chromium.launch({ args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=swiftshader', '--use-webgpu-adapter=swiftshader', '--disable-vulkan-surface'] });
  const page = await browser.newPage();
  page.on('console', (m) => console.log('[page]', m.text()));
  page.on('pageerror', (e) => console.log('[pageerror]', e.message));
  await page.goto(`http://127.0.0.1:${port}/test-webgpu.html`);
  const r = await page.evaluate(() => window.__result);
  await browser.close(); srv.close();
  if (r.error) process.exit(1);
  const worst = Math.min(...r.rows.map((x) => x.fp32), ...r.batchCos);
  console.log('WORST cos vs fp32 ORT:', worst.toFixed(7));
  process.exit(worst > 0.99999 ? 0 : 1);
});
