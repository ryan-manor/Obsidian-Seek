"""Ground-truth vectors from onnxruntime for the JS/WGSL engines.

Writes fixtures.json: per text, token ids plus the CLS-pooled L2-normalized
vector from (a) the shipped model as-is (accuracy_level=4: int8-quantized
activations, what ORT's CPU/WASM MatMulNBits does) and (b) the same weights
with accuracy_level=0 (fp32 activations, the exact math the WGSL engine does).

usage: python3 -I gen_fixtures.py <model_dir> <out.json>
"""
import sys, json, numpy as np, onnx, onnxruntime as ort
from tokenizers import Tokenizer

md, out = sys.argv[1], sys.argv[2]
tok = Tokenizer.from_file(f"{md}/tokenizer.json")
m = onnx.load(f"{md}/onnx/model_q4.onnx")
for n in m.graph.node:
    for a in n.attribute:
        if a.name == "accuracy_level": a.i = 0
s_acc0 = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
s_ship = ort.InferenceSession(f"{md}/onnx/model_q4.onnx", providers=["CPUExecutionProvider"])

texts = [
    "street photography in low light",
    "wie konfiguriere ich obsidian sync",
    "motorcycle maintenance checklist",
    "Q3 roadmap",
    "how do I tune BM25 field weights for title vs body",
    "東京の写真散歩ルート",
    "meeting notes from the search relevance review with the ranking team " * 3,
    "a " * 120,
]
fx = []
for t in texts:
    ids = tok.encode(t).ids
    feed = {"input_ids": np.array([ids], np.int64), "attention_mask": np.ones((1, len(ids)), np.int64)}
    def vec(sess):
        v = sess.run(None, feed)[0][0, 0]; return (v / np.linalg.norm(v)).tolist()
    fx.append({"text": t, "ids": ids, "fp32": vec(s_acc0), "shipped": vec(s_ship)})
json.dump(fx, open(out, "w"))
print("wrote", len(fx), "fixtures; lengths", [len(f["ids"]) for f in fx])
