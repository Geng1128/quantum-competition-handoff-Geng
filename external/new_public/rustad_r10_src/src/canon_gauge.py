"""Exact projective sign gauge of the Givens angle words: every cosine uses a positive half-turn, so the two top bits
of each word agree (the J=4 streamed lookup needs this). Vectors change only by a global sign per row.

  python src/canon_gauge.py IN.npz OUT.npz
"""
import sys
import numpy as np


def canon(words, edges, brot):
    out = np.asarray(words, dtype=np.int64).copy()
    signs = np.ones((len(out), len(edges) + 1), dtype=np.int8)
    mod = 1 << brot
    for k in reversed(range(len(edges))):
        p, q = map(int, edges[k])
        x = out[:, k].copy()
        opp = signs[:, p] != signs[:, q]
        x[opp] = (-x[opp]) % mod
        flip = (x >= mod // 4) & (x < 3 * mod // 4)
        x[flip] = (x[flip] + mod // 2) % mod
        signs[flip, p] *= -1
        signs[:, q] = 1
        out[:, k] = x
    assert not np.any(((out >> (brot - 1)) ^ (out >> (brot - 2))) & 1)
    return out, signs[:, 0]


def decode(words, edges, brot):
    v = np.zeros((len(words), len(edges) + 1))
    v[:, 0] = 1
    for k, (p, q) in enumerate(edges):
        t = words[:, k] * (np.pi / (1 << (brot - 1)))
        a, b = v[:, p].copy(), v[:, q].copy()
        v[:, p] = np.cos(t) * a - np.sin(t) * b
        v[:, q] = np.sin(t) * a + np.cos(t) * b
    return v


if __name__ == "__main__":
    z = np.load(sys.argv[1])
    d = {k: np.array(z[k]) for k in z.files}
    brot, edges = int(d["brot"]), d["tree_edges"].astype(np.int64)
    w = d["angle_words"].astype(np.int64)
    g, s = canon(w, edges, brot)
    err = np.abs(decode(g, edges, brot) - s[:, None] * decode(w, edges, brot)).max()
    print("gauge max coordinate error", err, "rows flipped", int((s < 0).sum()))
    assert err < 1e-12
    R, B, N = d["U"].shape
    d["angle_words"] = g.astype(d["angle_words"].dtype)
    d["U"] = d["U"] * s[len(w) - R * B:].reshape(R, B, 1)
    for k in ("words1", "words2"):
        d.pop(k, None)
    np.savez(sys.argv[2], **d)
    print("saved", sys.argv[2])
