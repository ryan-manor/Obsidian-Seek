import { defineConfig } from 'vitest/config';
import { fileURLToPath } from 'node:url';

// The `obsidian` package is types-only (package.json main: "") — esbuild
// externalizes it at build time, but vitest needs a resolvable runtime module
// for the handful of files that import a runtime VALUE from it (currently just
// platform.ts → Platform). Alias it to a tiny test stub. Inert for the rest of
// the suite, which imports nothing from obsidian.
export default defineConfig({
    // Provide `window`/`activeWindow` in the Node env so the plugin's
    // popout-window-safe `window.setTimeout`/`activeWindow` calls resolve under
    // test (see test-stubs/test-setup.ts). Patchable by vi.useFakeTimers().
    test: {
        setupFiles: [fileURLToPath(new URL('./src/test-stubs/test-setup.ts', import.meta.url))],
    },
    resolve: {
        alias: {
            obsidian: fileURLToPath(new URL('./src/test-stubs/obsidian.ts', import.meta.url)),
        },
    },
});
