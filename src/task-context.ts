// TaskContextTracker — attributes PerformanceObserver longtask entries to the
// plugin phase that was running WHEN THE TASK RAN, not when the entry was
// DELIVERED. The old stack read the top-of-stack at observer-callback time,
// but the observer delivers entries only after the task ends — so the final
// long task of any phase (pop already ran) and every task from an un-wrapped
// path logged as 'idle'. Issue #5's report had 101 long-tasks, all 'idle',
// and the triage stalled on exactly that ambiguity.
//
// Model: push()/pop() record SPANS (context + start/end on the performance.now()
// clock — the same clock longtask entry.startTime uses, both relative to the
// main window's timeOrigin). attribute() maps a task interval onto the span
// with the largest overlap; ties go to the LATEST-STARTED span so a nested
// phase (a 'bm25-warm' refit inside a 'search') wins over its parent — the
// inner span is the more specific diagnosis. No overlap at all → 'idle',
// which now genuinely means "no Seek phase was running".
//
// pop() closes the LAST OPEN span of the caller's context (not a strict stack
// discipline) so interleaved async lifetimes — a reindex outliving a modal
// open — unwind correctly, preserving the old stack's tolerance.

// The phases worth distinguishing in a report. Kept as a type (not enum) so
// LongTaskEntry.context stays a plain string in the NDJSON log.
export type TaskContext =
    | 'search'       // query path: modal keystroke → results
    | 'indexing'     // full reindex, incremental flush, or delta embed
    | 'catchup'      // deferred-embed drain (runCatchUp)
    | 'model-load'   // embedder load: fetch + wasm compile + session init + warmup
    | 'bm25-warm'    // BM25 cache build/refit (the cold-start fit)
    | 'reconcile';   // periodic/onload reconcile: sidecar scan, orphan sweep, compaction

interface Span {
    c: TaskContext;
    start: number;
    end: number | null;   // null = still open
}

// Closed spans older than this are pruned — longtask delivery lags the task by
// at most a frame or two, so anything beyond a couple minutes can never be
// needed for attribution again.
const SPAN_RETENTION_MS = 120_000;
// Hard cap on retained spans — a pathological push storm (e.g. a per-keystroke
// bug) degrades attribution rather than growing memory unboundedly.
const MAX_SPANS = 256;

export class TaskContextTracker {
    private spans: Span[] = [];
    // Injectable clock for tests; production uses performance.now() so spans
    // share the longtask entries' timebase.
    constructor(private readonly now: () => number = () => performance.now()) {}

    push(c: TaskContext): void {
        this.prune();
        this.spans.push({ c, start: this.now(), end: null });
    }

    pop(c: TaskContext): void {
        for (let i = this.spans.length - 1; i >= 0; i--) {
            const s = this.spans[i];
            if (s.c === c && s.end === null) {
                s.end = this.now();
                return;
            }
        }
        // No open span of this context — a double-pop. Harmless: ignore.
    }

    // The context of the most recently opened still-open span — the old
    // top-of-stack read. Behavioral consumers (the settings card's isIndexing,
    // the mobile unload gate's `busy`) want "what is running NOW", which is a
    // different question from attribute()'s "what was running THEN".
    current(): TaskContext | 'idle' {
        for (let i = this.spans.length - 1; i >= 0; i--) {
            if (this.spans[i].end === null) return this.spans[i].c;
        }
        return 'idle';
    }

    // Map a longtask interval to the best-overlapping span. `startMs` is the
    // entry's startTime (performance.now() timebase), `durationMs` its duration.
    attribute(startMs: number, durationMs: number): TaskContext | 'idle' {
        const taskEnd = startMs + durationMs;
        let best: Span | null = null;
        let bestOverlap = 0;
        for (const s of this.spans) {
            const spanEnd = s.end ?? Number.POSITIVE_INFINITY;   // open span covers "now"
            const overlap = Math.min(taskEnd, spanEnd) - Math.max(startMs, s.start);
            if (overlap <= 0) continue;
            if (overlap > bestOverlap || (best !== null && overlap === bestOverlap && s.start > best.start)) {
                best = s;
                bestOverlap = overlap;
            }
        }
        return best ? best.c : 'idle';
    }

    private prune(): void {
        if (this.spans.length < 64) return;   // cheap fast path for normal load
        const cutoff = this.now() - SPAN_RETENTION_MS;
        this.spans = this.spans.filter(s => s.end === null || s.end >= cutoff);
        // Still over cap after time-pruning: drop the oldest CLOSED spans.
        // Open spans are never dropped — losing one would orphan its pop.
        if (this.spans.length >= MAX_SPANS) {
            let excess = this.spans.length - MAX_SPANS + 1;
            this.spans = this.spans.filter(s => {
                if (excess > 0 && s.end !== null) { excess--; return false; }
                return true;
            });
        }
    }
}
