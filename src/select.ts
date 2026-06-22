// Bounded top-N index selection without a full-length sort.
//
// Stage-1 candidate generation selects the top-N of each arm (binary 200, BM25
// 100, recency 50) out of the whole resident corpus (~5k+ chunks), three times
// per keystroke. The old form built a full-length index array and ran a full
// O(len log len) sort just to slice the head — three large short-lived
// allocations per keystroke, a mobile GC-jank source. This selects with a
// bounded min-heap instead: O(len log N) time, O(N) space, no len-sized array.
//
// Output is IDENTICAL (not merely set-identical) to the old
// `eligibleIndicesAscending.sort((a,b) => key[b]-key[a]).slice(0, N)`:
//   - same MEMBERS: the N items with the largest key (ties at the N-cutoff
//     resolved in favour of the lower index), and
//   - same ORDER: key descending, ties broken by ascending index.
// Order matters because the callers union these indices into a Set whose
// iteration order seeds `rank()`'s final stable-sort tie-break on
// exactly-equal fused scores; preserving order keeps that tie-break — and thus
// the whole pipeline — byte-identical, not just the candidate set.
//
// The (key desc, index asc) rule is a strict total order over distinct indices,
// so the selected set is unique and independent of scan order; the bounded heap
// is just a cheaper way to compute the same head.

// True iff item a is LESS preferred than item b under (key desc, index asc):
// a lower key, or an equal key with a higher index.
function lessPreferred(ak: number, ai: number, bk: number, bi: number): boolean {
    return ak < bk || (ak === bk && ai > bi);
}

export function selectTopNIndices(
    len: number,
    n: number,
    keyOf: (i: number) => number,
    eligible?: ((i: number) => boolean) | null,
): number[] {
    if (n <= 0 || len <= 0) return [];

    // Bounded min-heap, parallel arrays, root = the LEAST preferred kept item
    // (lowest key, highest index among ties) so a new item is admitted iff it
    // beats the root. Capacity is N; it only ever grows to min(N, eligibleCount).
    const heapIdx: number[] = [];
    const heapKey: number[] = [];

    const siftUp = (start: number): void => {
        let c = start;
        const ck = heapKey[c];
        const ci = heapIdx[c];
        while (c > 0) {
            const p = (c - 1) >> 1;
            if (lessPreferred(ck, ci, heapKey[p], heapIdx[p])) {
                heapKey[c] = heapKey[p];
                heapIdx[c] = heapIdx[p];
                c = p;
            } else break;
        }
        heapKey[c] = ck;
        heapIdx[c] = ci;
    };

    const siftDown = (start: number): void => {
        const size = heapIdx.length;
        let p = start;
        const pk = heapKey[p];
        const pi = heapIdx[p];
        for (;;) {
            let l = 2 * p + 1;
            if (l >= size) break;
            const r = l + 1;
            // Descend toward the less-preferred child (the heap's "smaller").
            if (r < size && lessPreferred(heapKey[r], heapIdx[r], heapKey[l], heapIdx[l])) l = r;
            if (lessPreferred(heapKey[l], heapIdx[l], pk, pi)) {
                heapKey[p] = heapKey[l];
                heapIdx[p] = heapIdx[l];
                p = l;
            } else break;
        }
        heapKey[p] = pk;
        heapIdx[p] = pi;
    };

    for (let i = 0; i < len; i++) {
        if (eligible && !eligible(i)) continue;
        const k = keyOf(i);
        if (heapIdx.length < n) {
            heapIdx.push(i);
            heapKey.push(k);
            siftUp(heapIdx.length - 1);
        } else if (lessPreferred(heapKey[0], heapIdx[0], k, i)) {
            // Root (current least-preferred kept item) loses to the candidate.
            heapKey[0] = k;
            heapIdx[0] = i;
            siftDown(0);
        }
    }

    // The heap holds the right SET in heap order; sort it into the canonical
    // (key desc, index asc) order so the result is order-identical to the old
    // full-sort-then-slice. Sorting ≤ N items, not the full corpus.
    heapIdx.sort((a, b) => {
        const d = keyOf(b) - keyOf(a);
        return d !== 0 ? d : a - b;
    });
    return heapIdx;
}
