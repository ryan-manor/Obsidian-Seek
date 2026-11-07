"""Numpy reference forward for granite-embedding-97m-multilingual-r2 GBQ4.

Reads the SAME model_q4.onnx Seek ships (no re-export), dequantizes the
MatMulNBits / GatherBlockQuantized weights, runs ModernBERT in float32, and
compares CLS-pooled, L2-normalized vectors against onnxruntime.

usage: python3 -I ref_numpy.py <model_dir>
"""
import sys, json, re
import numpy as np, onnx, onnxruntime as ort
from onnx import numpy_helper
from tokenizers import Tokenizer

md = sys.argv[1]
cfg = json.load(open(f"{md}/config.json"))
m = onnx.load(f"{md}/onnx/model_q4.onnx")
g = m.graph
inits = {i.name: i for i in g.initializer}

def arr(name):
    return numpy_helper.to_array(inits[name])

def deq_matmulnbits(node):
    a = {x.name: onnx.helper.get_attribute_value(x) for x in node.attribute}
    K, N, bs = a["K"], a["N"], a["block_size"]
    q = arr(node.input[1])                     # (N, K/bs, bs/2) uint8
    s = arr(node.input[2]).reshape(N, K // bs)  # (N, K/bs)
    assert len(node.input) < 4 or node.input[3] == "", "zero points unexpected"
    lo = (q & 0x0F).astype(np.int8); hi = (q >> 4).astype(np.int8)
    nib = np.stack([lo, hi], axis=-1).reshape(N, K // bs, bs)  # low nibble first
    w = (nib.astype(np.float32) - 8.0) * s[:, :, None]          # default zp = 8
    return w.reshape(N, K)                                       # y = x @ w.T

W = {}
for n in g.node:
    if n.op_type == "MatMulNBits":
        mm = re.search(r"layers\.(\d+)/(attn|mlp)/(Wqkv|Wo|Wi)", n.name)
        W[(int(mm.group(1)), mm.group(2), mm.group(3))] = deq_matmulnbits(n)
    elif n.op_type == "GatherBlockQuantized":
        q = arr(n.input[0]).astype(np.float32)   # int4 signed, unpacked by onnx
        s = arr(n.input[2])
        emb = (q.reshape(q.shape[0], -1, 32) * s[:, :, None]).reshape(q.shape)
ln = {k: arr(k) for k in inits if k.endswith("norm.weight")}

H, D, L = cfg["hidden_size"], cfg["hidden_size"] // cfg["num_attention_heads"], cfg["num_hidden_layers"]
NH = cfg["num_attention_heads"]; EPS = cfg["norm_eps"]; WIN = cfg["local_attention"] // 2

def layernorm(x, w):
    mu = x.mean(-1, keepdims=True); v = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(v + EPS) * w

def rope(x, theta):  # x: (NH, S, D), rotate_half convention
    S = x.shape[1]
    inv = 1.0 / (theta ** (np.arange(0, D, 2, dtype=np.float32) / D))
    f = np.arange(S, dtype=np.float32)[:, None] * inv[None]
    emb = np.concatenate([f, f], -1); c, s = np.cos(emb), np.sin(emb)
    x1, x2 = x[..., : D // 2], x[..., D // 2 :]
    return x * c + np.concatenate([-x2, x1], -1) * s

def forward(ids, mask):
    S = len(ids)
    x = layernorm(emb[ids], ln["0.auto_model.embeddings.norm.weight"])
    pos = np.arange(S)
    pad = np.where(mask[None, :] > 0, 0.0, -np.inf).astype(np.float32)
    win = np.where(np.abs(pos[:, None] - pos[None, :]) <= WIN, 0.0, -np.inf).astype(np.float32)
    for l in range(L):
        glob = l % cfg["global_attn_every_n_layers"] == 0
        h = x if l == 0 else layernorm(x, ln[f"0.auto_model.layers.{l}.attn_norm.weight"])
        qkv = (h @ W[(l, "attn", "Wqkv")].T).reshape(S, 3, NH, D).transpose(1, 2, 0, 3)
        theta = cfg["global_rope_theta"] if glob else cfg["local_rope_theta"]
        q, k, v = rope(qkv[0], theta), rope(qkv[1], theta), qkv[2]
        sc = q @ k.transpose(0, 2, 1) / np.sqrt(D) + pad + (0 if glob else win)
        sc = np.exp(sc - sc.max(-1, keepdims=True)); sc /= sc.sum(-1, keepdims=True)
        a = (sc @ v).transpose(1, 0, 2).reshape(S, H)
        x = x + a @ W[(l, "attn", "Wo")].T
        h = layernorm(x, ln[f"0.auto_model.layers.{l}.mlp_norm.weight"]) @ W[(l, "mlp", "Wi")].T
        inp, gate = h[:, : h.shape[1] // 2], h[:, h.shape[1] // 2 :]
        x = x + (inp / (1 + np.exp(-inp)) * gate) @ W[(l, "mlp", "Wo")].T
    x = layernorm(x, ln["0.auto_model.final_norm.weight"])
    c = x[0]; return c / np.linalg.norm(c), x

tok = Tokenizer.from_file(f"{md}/tokenizer.json")
sess = ort.InferenceSession(f"{md}/onnx/model_q4.onnx", providers=["CPUExecutionProvider"])
texts = ["street photography in low light",
         "wie konfiguriere ich obsidian sync",
         "meeting notes from the search relevance review with the ranking team " * 3,
         "x " * 150]
worst = 1.0
for t in texts:
    ids = tok.encode(t).ids
    m_ = np.ones(len(ids), dtype=np.int64)
    out = sess.run(None, {"input_ids": np.array([ids], dtype=np.int64), "attention_mask": m_[None]})
    ref = out[0][0, 0]; ref = ref / np.linalg.norm(ref)
    mine, tokens = forward(np.array(ids), m_)
    cos = float(ref @ mine); worst = min(worst, cos)
    tokcos = np.min(np.sum(out[0][0] * tokens, -1) / np.linalg.norm(out[0][0], axis=-1) / np.linalg.norm(tokens, axis=-1))
    print(f"S={len(ids):4d} cls-cos={cos:.6f} min-token-cos={tokcos:.6f} :: {t[:40]!r}")
print("WORST", worst)
