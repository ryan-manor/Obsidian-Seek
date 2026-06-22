The fusion change in one screen: bound-norm replaces per-query min-max, coverage scales raw BM25.

```ts
const norm = bm25Raw / theoreticalBound(query);
const fused = alpha * dense + (1 - alpha) * norm * coverage;
```

Two operating points were considered.

| alpha | personal | dnd    |
| ----- | -------- | ------ |
| 0.90  | 0.8666   | 0.9318 |
| 0.92  | 0.8671   | 0.9301 |

> [!note]
> Effective alpha ≠ nominal alpha: BM25 norm spread is ~13× dense's.

Shipped at 0.92 after the live smoke test.
