import esbuild from 'esbuild';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';

const prod = process.argv[2] === 'production';

// Content hash of the BM25 analyzer sources + the MiniSearch version. Any edit
// to tokenization / term processing / depluralize tables / field derivation —
// or a MiniSearch upgrade — changes which tokens land in the persisted index's
// postings, so this hash gates the persisted-index stamp (search.ts /
// bm25.ts ANALYZER_VERSION): a changed analyzer auto-invalidates old blobs and
// forces a refit, keeping a loaded index relevance-identical to a fresh fit.
const analyzerVersion = createHash('sha256')
    .update(readFileSync('src/bm25.ts'))
    .update(readFileSync('src/tokenize.ts'))
    .update(readFileSync('src/prop-normalize.ts'))
    .update(JSON.parse(readFileSync('node_modules/minisearch/package.json', 'utf8')).version)
    .digest('hex')
    .slice(0, 16);

// Bundle the off-thread binary scorer to a standalone IIFE string, injected into
// the main bundle via `define` (__BINARY_WORKER_SRC__). The main thread spins it
// up from a Blob URL — Obsidian plugins ship a single main.js with no sidecar
// file to load, so the worker source rides inline. It pulls in only the pure
// compute (binary.ts + select.ts), so it stays tiny and obsidian-free.
const workerBuild = await esbuild.build({
    entryPoints: ['src/binary-worker.ts'],
    bundle: true,
    format: 'iife',
    target: 'es2022',
    platform: 'browser',
    write: false,
    minify: prod,
    sourcemap: false,
    logLevel: 'silent',
});
const binaryWorkerSrc = workerBuild.outputFiles[0].text;

const context = await esbuild.context({
    entryPoints: ['src/main.ts'],
    bundle: true,
    external: ['obsidian', 'electron', 'node:*'],
    format: 'cjs',
    target: 'es2022',
    platform: 'browser',
    outfile: 'main.js',
    minify: prod,
    sourcemap: prod ? false : 'inline',
    logLevel: 'info',
    define: {
        'process.env.NODE_ENV': JSON.stringify(prod ? 'production' : 'development'),
        '__BUILD_TS__': JSON.stringify(new Date().toISOString()),
        '__SEEK_ANALYZER_VERSION__': JSON.stringify(analyzerVersion),
        '__BINARY_WORKER_SRC__': JSON.stringify(binaryWorkerSrc),
    },
});

if (prod) {
    await context.rebuild();
    await context.dispose();
    console.log('Seek: production build complete');
} else {
    await context.watch();
    console.log('Seek: dev build in watch mode');
}
