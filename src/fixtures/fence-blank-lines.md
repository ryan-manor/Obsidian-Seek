Quick recipe for the reindex throttle. The pacer yields between batches so the compositor never starves.

```python
import time


class Pacer:
    def __init__(self, budget_ms: float = 8.0):
        self.budget_ms = budget_ms


    def tick(self) -> None:
        time.sleep(self.budget_ms / 1000.0)
```

Tuning note: 8 ms keeps typing latency invisible on the M2; the iPhone needs 16.
