import { defineConfig } from 'vitest/config';
import { fileURLToPath } from 'node:url';

// The `obsidian` package is types-only (package.json main: "") — esbuild
// externalizes it at build time, but vitest needs a resolvable runtime module
// for the handful of files that import a runtime VALUE from it (currently just
// platform.ts → Platform). Alias it to a tiny test stub. Inert for the rest of
// the suite, which imports nothing from obsidian.
export default defineConfig({
    resolve: {
        alias: {
            obsidian: fileURLToPath(new URL('./src/test-stubs/obsidian.ts', import.meta.url)),
        },
    },
});
