// BM25F lexical scoring (Robertson, Zaragoza & Taylor, CIKM 2004) — the 2026-11-11
// ship candidate `k5-props2`, shipped with properties 4 (`k5-props4`) ([[Seek Title, Bound and Coordination Investigation
// 2026-11-11]]; harness: ~/seek-eval-pack/harness/field_model_arm.py FieldModel,
// configured in round4_arm.py build_models()).
//
// WHY. MiniSearch scores each field as an independent BM25 (own IDF, own saturation,
// own d floor), multiplies by the field boost OUTSIDE, and sums — Elasticsearch
// `most_fields`. That breaks saturation: a term in title + body + properties earns
// three near-saturated scores, and the title field's per-field IDF (few titles carry
// any term) ×10 makes UB(t) — and therefore the Σ UB(t) bound — "appears in some short
// title" instead of term rarity. BM25F merges the weighted, length-normalized term
// frequencies FIRST and saturates ONCE, against one document-level IDF:
//
//     t̃f(t,d) = Σ_f w_f · tf_f(t,d) / (1 − b_f + b_f · len_f(d) / avg_f)
//     s(t,d)  = idf(t) · (δ + (k1+1) · t̃f / (k1 + t̃f))          (only where t̃f > 0)
//     idf(t)  = ln(1 + (N − df + 0.5) / (df + 0.5)),  df = docs with t in ANY field
//
// so a term's ceiling is idf·(k1+1+δ) whatever fields it hits; the weights only set
// how fast a note climbs toward it. k1 = 5 (Zaragoza TREC-13 named-page k1 ≈ 4.9):
// low k1 compresses the field weights (a 3-word title hit and one body mention land
// at 92% vs 48% of the ceiling at k1 1.2, 75% vs 18% at k1 5). Harness verdict
// (2026-11-11/12): miss sets +0.039/+0.048/+0.049, captures +0.171, faceted place
// queries 0.337 → 0.550 nDCG@10, usage −0.004..−0.009 (CI spans 0), personal flat.
//
// UNITS (all matched to the harness exactly, parity-tested in
// tests/bm25f-parity.test.ts):
//   - document = one CHUNK (the MiniSearch doc), N = live chunk count, df = live
//     chunks holding the term in any field (FieldModel._doc_idf; k5-props2 has no
//     note_of, so no note-level df).
//   - len_f(d) = the number of PROCESSED tokens the field indexed (stopwords dropped
//     except title/aliases, duplicates counted) = Σ_t tf_f(t,d) over the postings —
//     the harness's `len(tokenize(field))`. NOT MiniSearch's own `_fieldLength`,
//     which is the count of UNIQUE RAW tokens (stopwords included) and only feeds
//     the legacy per-field path. avg_f = Σ_d len_f(d) / N (empty fields count 0;
//     an all-empty field's avg 0 is read as 1, the harness's `avg_len[f] or 1.0`).
//   - δ once per term (not per field), b = 0.75 on every field.
//
// This module is the pure arithmetic; bm25.ts owns the MiniSearch internals reach
// (postings, liveness, short-id → frame-row mapping) and the query-side expansion
// (fuzzy / prefix / synonym specs), which it runs exactly as MiniSearch would.

// Field weights relative to content = 1. The harness derived them as the shipped
// per-field boosts / 3 (content 3 → 1, tags 3 → 1, headings 3 → 1) with title and
// aliases re-set to 10 and 6. Properties: 2/3 → 2 in the 2026-11-11 investigation
// (`k5-props2`), then 4 per the 2026-11-12 ml97 sweep (`k5-props4`, harness
// round4_arm build_models). A field missing here weighs 1 (harness
// `self.w.get(f, 1.0)`); a field the index doesn't carry (searchableProperties /
// headingsField off) has no postings and is inert.
export const BM25F_FIELD_WEIGHTS: Readonly<Record<string, number>> = Object.freeze({
    title: 10,
    aliases: 6,
    tags: 1,
    content: 1,
    properties: 4,
    headings: 1,
});

export interface Bm25fParams {
    k1: number;
    // Length-normalization strength: one value for every field, or per field (a
    // field absent from the record uses `bDefault`). The ship config is 0.75
    // everywhere; per-field b is kept for the title-b sweeps the literature suggests
    // (DBpedia-Entity v2 trained title b ≈ 0.05) — the harness arm supports it too.
    b: number | Readonly<Record<string, number>>;
    bDefault?: number;
    // BM25+ lower-bound delta, applied ONCE per (term, doc) after saturation — not per
    // field as MiniSearch's per-field sum does (FieldModel.score bm25f branch).
    d: number;
    weights: Readonly<Record<string, number>>;
}

// The ship config (k5-props4). bm25.ts memoizes UB(t) per params VALUE
// (bm25fParamsKey), so any params object with the same numbers shares the cache and
// a changed weight gets its own, freshly computed bounds.
export const BM25F_PARAMS: Readonly<Bm25fParams> = Object.freeze({
    k1: 5,
    b: 0.75,
    d: 0.5,
    weights: BM25F_FIELD_WEIGHTS,
});

// ---- User-tunable knobs (settings + seek:search overrides) ---------------------
// Title, aliases, properties, body and k1 are exposed; tags and headings follow body
// (they ride at content weight in the ship config, and a separate knob for each would
// be noise for a hand-tuner). All query-time: the index never sees them.
export interface Bm25fKnobs {
    title: number;
    aliases: number;
    properties: number;
    body: number;
    k1: number;
}
export const BM25F_KNOB_DEFAULTS: Readonly<Bm25fKnobs> = Object.freeze({
    title: BM25F_FIELD_WEIGHTS.title,
    aliases: BM25F_FIELD_WEIGHTS.aliases,
    properties: BM25F_FIELD_WEIGHTS.properties,
    body: BM25F_FIELD_WEIGHTS.content,
    k1: BM25F_PARAMS.k1,
});
// [min, max, step] — valid ranges for the seek:search per-query overrides; values outside are clamped.
export const BM25F_KNOB_RANGES: Readonly<Record<keyof Bm25fKnobs, readonly [number, number, number]>> = Object.freeze({
    title: [0, 20, 0.5],
    aliases: [0, 12, 0.5],
    properties: [0, 10, 0.5],
    body: [0, 4, 0.1],
    k1: [0.5, 10, 0.1],
});

// Knobs → params. Missing / non-finite values fall back to the default, the rest are
// clamped to BM25F_KNOB_RANGES. All-default knobs return BM25F_PARAMS itself.
export function bm25fParamsFrom(k: Partial<Bm25fKnobs>): Readonly<Bm25fParams> {
    const v = {} as Bm25fKnobs;
    let isDefault = true;
    for (const key of Object.keys(BM25F_KNOB_DEFAULTS) as Array<keyof Bm25fKnobs>) {
        const [lo, hi] = BM25F_KNOB_RANGES[key];
        const raw = k[key];
        const x = typeof raw === 'number' && Number.isFinite(raw) ? Math.min(hi, Math.max(lo, raw)) : BM25F_KNOB_DEFAULTS[key];
        v[key] = x;
        if (x !== BM25F_KNOB_DEFAULTS[key]) isDefault = false;
    }
    if (isDefault) return BM25F_PARAMS;
    return Object.freeze({
        k1: v.k1,
        b: BM25F_PARAMS.b,
        d: BM25F_PARAMS.d,
        weights: Object.freeze({
            title: v.title, aliases: v.aliases, properties: v.properties,
            content: v.body, tags: v.body, headings: v.body,
        }),
    });
}

// Value identity of a params object: the UB(t) memo key. Cached per object, so the
// per-term lookup does not re-serialize.
const keyCache = new WeakMap<object, string>();
export function bm25fParamsKey(p: Bm25fParams): string {
    let k = keyCache.get(p);
    if (k === undefined) {
        const w = Object.keys(p.weights).sort().map(f => `${f}=${p.weights[f]}`).join(',');
        const b = typeof p.b === 'number' ? String(p.b) : Object.keys(p.b).sort().map(f => `${f}=${(p.b as Record<string, number>)[f]}`).join(',');
        k = `k1=${p.k1}|b=${b}|bd=${p.bDefault ?? ''}|d=${p.d}|${w}`;
        keyCache.set(p, k);
    }
    return k;
}

export function bm25fFieldB(p: Bm25fParams, field: string): number {
    if (typeof p.b === 'number') return p.b;
    const v = p.b[field];
    return v !== undefined ? v : (p.bDefault ?? 0.75);
}

export function bm25fFieldWeight(p: Bm25fParams, field: string): number {
    const w = p.weights[field];
    return w !== undefined ? w : 1;
}

// Document-level IDF (BM25+ form, the same expression MiniSearch uses per field).
export function bm25fIdf(n: number, df: number): number {
    return df > 0 ? Math.log(1 + (n - df + 0.5) / (df + 0.5)) : 0;
}

// The single saturation: idf · (δ + (k1+1)·t̃f / (k1 + t̃f)). Caller guarantees t̃f > 0.
export function bm25fSaturate(idf: number, tft: number, k1: number, d: number): number {
    return idf * (d + (k1 + 1) * tft / (k1 + tft));
}

// Per-query view of the corpus statistics the per-term kernel reads, indexed by
// MiniSearch FIELD ID (the integer `_fieldIds` assigns in `fields` order).
export interface Bm25fFieldStats {
    weight: number;       // w_f
    b: number;            // b_f
    avg: number;          // avg_f (already defaulted: 0 → 1)
    len: Float64Array;    // len_f by MiniSearch short doc id
}

// Scratch buffers reused across terms and queries (sized to the short-id space),
// so the kernel allocates nothing per posting. `stamp` is an epoch marker: a short
// id is "touched for this term" iff stamp[sid] === epoch, so nothing is cleared
// between terms.
export class Bm25fScratch {
    tft = new Float64Array(0);
    stamp = new Uint32Array(0);
    touched = new Int32Array(0);
    score = new Float64Array(0);
    epoch = 0;

    ensure(size: number): void {
        if (this.tft.length >= size) return;
        const cap = Math.max(size, Math.ceil(this.tft.length * 1.5), 64);
        this.tft = new Float64Array(cap);
        this.stamp = new Uint32Array(cap);
        this.touched = new Int32Array(cap);
        this.score = new Float64Array(cap);
        this.epoch = 0;
    }

    nextEpoch(): number {
        this.epoch = (this.epoch + 1) >>> 0;
        if (this.epoch === 0) {   // wrapped: invalidate every stale stamp
            this.stamp.fill(0);
            this.epoch = 1;
        }
        return this.epoch;
    }
}

// Score ONE (exact or derived) term against every live doc holding it. `data` is the
// term's MiniSearch postings (fieldId → shortId → tf). `rowOf` is the short-id →
// frame-row map with −1 = dead: it filters tombstoned docs exactly as MiniSearch's
// termResults does (a discarded doc stays in postings until vacuum); any other value
// (a row, or −2 for a live-but-unplaced doc) is live and counts toward df. Writes
// the touched short ids to scratch.touched[0..n) and their scores to
// scratch.score[0..n) (0 where t̃f = 0, i.e. only zero-weight fields hit — counted
// in df, never scored, the harness's `nz = tft > 0`). Returns n.
export function bm25fScoreTerm(
    data: ReadonlyMap<number, ReadonlyMap<number, number>>,
    fields: ReadonlyArray<Bm25fFieldStats | undefined>,
    rowOf: Int32Array,
    nDocs: number,
    k1: number,
    d: number,
    s: Bm25fScratch,
): number {
    const epoch = s.nextEpoch();
    const { tft, stamp, touched, score } = s;
    let n = 0;
    for (const [fieldId, docs] of data) {
        const f = fields[fieldId];
        if (!f) continue;
        const { weight: w, b, avg, len } = f;
        const oneMinusB = 1 - b;
        for (const [sid, tf] of docs) {
            if (rowOf[sid] === -1) continue;
            if (stamp[sid] !== epoch) {
                stamp[sid] = epoch;
                tft[sid] = 0;
                touched[n++] = sid;
            }
            tft[sid] += w * tf / (oneMinusB + b * len[sid] / avg);
        }
    }
    const idf = bm25fIdf(nDocs, n);
    for (let i = 0; i < n; i++) {
        const t = tft[touched[i]];
        score[i] = t > 0 ? bm25fSaturate(idf, t, k1, d) : 0;
    }
    return n;
}
