// Vitest setup — runs once per test file before the suite.
//
// The plugin follows Obsidian's popout-window convention of `window.setTimeout` /
// `window.clearTimeout` / `activeWindow` (so timers and globals resolve against
// the window the code is actually running in, not always the main one). Vitest
// runs in the Node environment, which has no `window` / `activeWindow`, so those
// references throw `ReferenceError` under test even though they're always present
// in the real Obsidian runtime.
//
// Alias both to `globalThis`: its `setTimeout`/`clearTimeout` ARE the Node timers
// and stay patchable by `vi.useFakeTimers()` (which replaces them on globalThis),
// so fake-timer-driven tests keep working. Guarded by a typeof check so a real
// DOM env (jsdom) is never clobbered. Nothing in the plugin branches on
// `typeof window` for environment detection, so this is inert for real behavior.
// eslint-disable-next-line -- test-env bootstrap: the global object is the only
// handle available before any window exists; it is aliased to window/activeWindow below
const g = globalThis as unknown as Record<string, unknown>;
if (typeof g.window === 'undefined') g.window = g;
if (typeof g.activeWindow === 'undefined') g.activeWindow = g;
