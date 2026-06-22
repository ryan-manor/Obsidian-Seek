Full eval driver, pasted whole. The fence is far over any embed window and must split at line boundaries with the fence reopened on every piece.

```python
import json
import math
from pathlib import Path

RESULTS = Path("results")
ALPHA = 0.92

def load_queries(corpus: str) -> list[dict]:
    rows = []
    for line in (Path("queries") / f"{corpus}.jsonl").read_text().splitlines():
        rows.append(json.loads(line))
    return rows

def ndcg_at_k(ranked: list[str], gold: dict[str, float], k: int = 10) -> float:
    dcg = 0.0
    for i, doc_id in enumerate(ranked[:k]):
        rel = gold.get(doc_id, 0.0)
        dcg += (2 ** rel - 1) / math.log2(i + 2)
    ideal = sorted(gold.values(), reverse=True)[:k]
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0

def fuse(dense: dict[str, float], lexical: dict[str, float], bound: float) -> dict[str, float]:
    out = {}
    for doc_id in set(dense) | set(lexical):
        d = dense.get(doc_id, 0.0)
        l = lexical.get(doc_id, 0.0) / bound if bound > 0 else 0.0
        out[doc_id] = ALPHA * d + (1 - ALPHA) * l
    return out

def main() -> None:
    for corpus in ("personal", "dnd", "captures", "code"):
        queries = load_queries(corpus)
        scores = []
        for q in queries:
            ranked = run_query(corpus, q["text"])
            scores.append(ndcg_at_k(ranked, q["gold"]))
        mean = sum(scores) / len(scores)
        print(f"{corpus}: nDCG@10 {mean:.4f} over {len(scores)} queries")
        (RESULTS / f"{corpus}_a{ALPHA}.json").write_text(json.dumps(scores))

if __name__ == "__main__":
    main()
```

One paragraph of trailing prose so the fence is not the final atom.
