// Plain-JS float32 reference of the ml97 (ModernBERT) forward pass.
// Not a production path — it exists so the WGSL engine has a debuggable,
// line-by-line twin: if a WGSL kernel disagrees, diff intermediate activations
// against this. Mirrors ref_numpy.py exactly.

import { gatherEmbeddings } from './onnx-weights.js';

function layerNorm(x, S, H, w, eps, out = new Float32Array(S * H)) {
    for (let s = 0; s < S; s++) {
        let mu = 0;
        for (let i = 0; i < H; i++) mu += x[s * H + i];
        mu /= H;
        let v = 0;
        for (let i = 0; i < H; i++) { const d = x[s * H + i] - mu; v += d * d; }
        const r = 1 / Math.sqrt(v / H + eps);
        for (let i = 0; i < H; i++) out[s * H + i] = (x[s * H + i] - mu) * r * w[i];
    }
    return out;
}

// y[S,N] = x[S,K] · Wᵀ with MatMulNBits u4 (zp 8) blocks.
function matmulQ4(x, S, W) {
    const { q, N, K, bs } = W;
    const sv = new DataView(W.scales.buffer, W.scales.byteOffset, W.scales.byteLength);
    const nb = K / bs, y = new Float32Array(S * N);
    for (let n = 0; n < N; n++) {
        for (let b = 0; b < nb; b++) {
            const sc = sv.getFloat32((n * nb + b) * 4, true);
            const base = (n * nb + b) * (bs / 2);
            for (let s = 0; s < S; s++) {
                let acc = 0;
                const xo = s * K + b * bs;
                for (let j = 0; j < bs / 2; j++) {
                    const byte = q[base + j];
                    acc += ((byte & 15) - 8) * x[xo + 2 * j] + ((byte >> 4) - 8) * x[xo + 2 * j + 1];
                }
                y[s * N + n] += acc * sc;
            }
        }
    }
    return y;
}

export function forwardCpu(weights, config, ids) {
    const H = config.hidden_size, NH = config.num_attention_heads, D = H / NH;
    const L = config.num_hidden_layers, eps = config.norm_eps, win = config.local_attention / 2;
    const S = ids.length;
    let x = layerNorm(gatherEmbeddings(weights.emb, ids), S, H, weights.embNorm, eps);
    const ropeTables = (theta) => {
        const cos = new Float32Array(S * D / 2), sin = new Float32Array(S * D / 2);
        for (let p = 0; p < S; p++) for (let i = 0; i < D / 2; i++) {
            const f = p / Math.pow(theta, (2 * i) / D);
            cos[p * D / 2 + i] = Math.cos(f); sin[p * D / 2 + i] = Math.sin(f);
        }
        return { cos, sin };
    };
    const ropeG = ropeTables(config.global_rope_theta), ropeL = ropeTables(config.local_rope_theta);
    for (let l = 0; l < L; l++) {
        const P = weights.layers[l];
        const glob = l % config.global_attn_every_n_layers === 0;
        const h = l === 0 ? x : layerNorm(x, S, H, P.attnNorm, eps);
        const qkv = matmulQ4(h, S, P.Wqkv); // [S, 3, NH, D]
        const rt = glob ? ropeG : ropeL;
        const rot = (s, off) => { // rotate_half RoPE in place on q or k for head slice
            for (let i = 0; i < D / 2; i++) {
                const c = rt.cos[s * D / 2 + i], sn = rt.sin[s * D / 2 + i];
                const a = qkv[off + i], b = qkv[off + i + D / 2];
                qkv[off + i] = a * c - b * sn;
                qkv[off + i + D / 2] = b * c + a * sn;
            }
        };
        for (let s = 0; s < S; s++) for (let hd = 0; hd < NH; hd++) {
            rot(s, s * 3 * H + hd * D);          // q
            rot(s, s * 3 * H + H + hd * D);      // k
        }
        const attn = new Float32Array(S * H), scores = new Float32Array(S);
        const scale = 1 / Math.sqrt(D);
        for (let hd = 0; hd < NH; hd++) for (let i = 0; i < S; i++) {
            let mx = -Infinity;
            for (let j = 0; j < S; j++) {
                if (!glob && Math.abs(i - j) > win) { scores[j] = -Infinity; continue; }
                let d = 0;
                for (let e = 0; e < D; e++) d += qkv[i * 3 * H + hd * D + e] * qkv[j * 3 * H + H + hd * D + e];
                scores[j] = d * scale; if (scores[j] > mx) mx = scores[j];
            }
            let sum = 0;
            for (let j = 0; j < S; j++) { scores[j] = Math.exp(scores[j] - mx); sum += scores[j]; }
            for (let e = 0; e < D; e++) {
                let a = 0;
                for (let j = 0; j < S; j++) a += scores[j] * qkv[j * 3 * H + 2 * H + hd * D + e];
                attn[i * H + hd * D + e] = a / sum;
            }
        }
        const o = matmulQ4(attn, S, P.Wo);
        for (let i = 0; i < S * H; i++) x[i] += o[i];
        const u = matmulQ4(layerNorm(x, S, H, P.mlpNorm, eps), S, P.Wi);
        const I = P.Wi.N / 2, g = new Float32Array(S * I);
        for (let s = 0; s < S; s++) for (let i = 0; i < I; i++) {
            const a = u[s * 2 * I + i], b = u[s * 2 * I + I + i];
            g[s * I + i] = (a / (1 + Math.exp(-a))) * b;
        }
        const o2 = matmulQ4(g, S, P.Wo2);
        for (let i = 0; i < S * H; i++) x[i] += o2[i];
    }
    const cls = layerNorm(x.subarray(0, H), 1, H, weights.finalNorm, eps);
    let n = 0; for (const v of cls) n += v * v;
    n = Math.sqrt(n);
    return cls.map((v) => v / n);
}
