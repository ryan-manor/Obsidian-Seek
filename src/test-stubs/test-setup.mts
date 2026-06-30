// Vitest setup — runs once per test file, before the suite.
//
// The plugin follows Obsidian's popout-window convention (`window.setTimeout`,
// `window.clearTimeout`, `activeWindow`) so timers and globals resolve against
// the window the code is actually running in. Vitest runs in the Node
// environment, which has no `window` / `activeWindow`, so those references would
// throw under test even though the real Obsidian runtime always provides them.
//
// Alias both to the Node global object: its `setTimeout` / `clearTimeout` ARE the
// Node timers and stay patchable by `vi.useFakeTimers()` (which replaces them on
// the global), so fake-timer-driven tests keep working. Guarded by a typeof check
// so a real DOM env (jsdom) is never clobbered.
//
// This is a test-bootstrap module (deliberately `.mts`, not shipped plugin code)
// that runs only in the Node test runner where there are no popout windows — so
// it uses `globalThis` directly, the one handle to the global object before any
// `window` exists. The plugin source itself never does (it follows the
// window/activeWindow convention the Obsidian guidelines require).
const g = globalThis as unknown as Record<string, unknown>;
if (typeof g.window === 'undefined') g.window = g;
if (typeof g.activeWindow === 'undefined') g.activeWindow = g;
