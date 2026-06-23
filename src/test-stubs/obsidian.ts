// Test-only runtime stub for `obsidian` (the real npm package is types-only;
// esbuild externalizes it in the real build). This alias replaces `obsidian` for
// the ENTIRE vitest suite (see vitest.config.mts), so any value a test
// transitively imports must be provided here. Add new runtime exports as code
// under test reaches for them — a missing export surfaces as `undefined` at use
// (e.g. `instanceof TFile` throwing), not a clear error.
//
// Platform's device-class flags drive platform.ts's compute-backend allowlist.
// Tests mutate these to pose as different devices; the object is shared (one
// module instance).
export const Platform = {
    isMobile: false,
    isIosApp: false,
    isAndroidApp: false,
    isTablet: false,
    isPhone: false,
};

// Minimal runtime stubs for the values search.ts / main.ts read at module load
// or use with `instanceof`. Kept intentionally thin — extend as needed.
export class TFile {
    path = '';
    stat = { mtime: 0, ctime: 0, size: 0 };
}
export class Notice {
    constructor(_message?: string, _timeout?: number) {}
    setMessage(_message: string): this {
        return this;
    }
    hide(): void {}
}
export function setIcon(_el: HTMLElement, _iconId: string): void {}

// Obsidian's real `parseYaml` is a thin wrapper over js-yaml's `load`, so the
// dev-only js-yaml dependency is a faithful runtime stand-in for tests (base-
// extractor.ts is the only consumer). The real build uses Obsidian's export at
// zero bundle cost; js-yaml never ships.
import { load as loadYaml } from 'js-yaml';
export function parseYaml(yaml: string): unknown {
    return loadYaml(yaml);
}
