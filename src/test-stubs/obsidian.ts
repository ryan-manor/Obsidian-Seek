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
    // Real Obsidian TFiles carry the lowercased extension; the index path reads
    // it (search.ts filters `.base` views via `f.extension === 'base'`). Default
    // '' so existing tests that only set path/stat are unaffected.
    extension = '';
}
export class Notice {
    constructor(_message?: string, _timeout?: number) {}
    setMessage(_message: string): this {
        return this;
    }
    hide(): void {}
}
export function setIcon(_el: HTMLElement, _iconId: string): void {}

// Class stubs for the values search-modal.ts binds at MODULE LOAD (`class
// SeekSearchModal extends Modal`), so importing that module for a unit test of
// one of its pure exports (titleNavCoverage) doesn't die on `extends undefined`.
// Bodies stay empty on purpose: nothing here is exercised, and a stub with
// behaviour would invite tests that assert against the stub instead of Obsidian.
export class Component {
    onload(): void {}
    onunload(): void {}
}
export class Modal extends Component {
    constructor(_app?: unknown) {
        super();
    }
    open(): void {}
    close(): void {}
}
export class MarkdownView extends Component {}
export const MarkdownRenderer = {
    render: async (): Promise<void> => {},
};

// Obsidian's real `parseYaml` is a thin wrapper over a YAML parser, so the
// dev-only `yaml` dependency is a faithful runtime stand-in for tests (base-
// extractor.ts is the only consumer). The real build uses Obsidian's export at
// zero bundle cost; `yaml` never ships. (`yaml.parse` returns `any`, so the
// result is widened to `unknown` to match Obsidian's typed signature.)
import { parse as parseYamlImpl } from 'yaml';
export function parseYaml(yaml: string): unknown {
    return parseYamlImpl(yaml) as unknown;
}
