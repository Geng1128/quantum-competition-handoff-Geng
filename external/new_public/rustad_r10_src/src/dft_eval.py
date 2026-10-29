"""Local dfthcsos metrics using the verifier's own numpy code (no kickmix).

Words for the one-body rows are the encoded eigenvectors of h1_prime(X) of the
decoded two-body vectors, as in the starter kit's encode.py.
"""
import os
import sys
from functools import lru_cache

import numpy as np

sys.path.insert(0, os.environ.get("QCC_VERIFIERS", "data/verifiers"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dfthcsos.scoreinfra import dfthc_math as dm  # noqa: E402
from dfthcsos.scoreinfra import reference as rf  # noqa: E402

N = 76
ETA = 113
E_UPPER = rf.UPPER_ENERGY
LAMBDA_FLOOR, GAP_FLOOR = 49.0, 3.914898


@lru_cache(maxsize=1)
def ref():
    return rf.load_reference(rf.check_fcidump())


@lru_cache(maxsize=1)
def shifts():
    p = rf.load_fixed_shifts()
    S = np.asarray(dm.symmetric_from_lower(p["Hsym_lower"], N), dtype=np.float64)
    beta_1 = float(p["H1_I"]) - 0.5 * float(np.trace(S))
    return S, beta_1


def native(params):
    """(w, W) from the npz coordinates: W stored as 'W' (lowercase w), H2_I = W_offset + sum_b w."""
    w = np.asarray(params["W"], dtype=np.float64)
    Wo = np.asarray(params["H2_I"], dtype=np.float64) - w.sum(axis=1)
    return w, Wo


def tree_angles(vectors, edges):
    u = np.array(vectors, dtype=np.float64, ndmin=2)
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    theta = np.zeros((u.shape[0], N - 1))
    for i in range(N - 2, -1, -1):
        p, c = edges[i]
        theta[:, i] = np.arctan2(u[:, c], u[:, p])
        u[:, p] = np.hypot(u[:, p], u[:, c])
        u[:, c] = 0.0
    return np.mod(theta / np.pi, 2.0)


def to_words(theta_over_pi, brot):
    return np.floor(np.mod(theta_over_pi, 2.0) * (1 << (brot - 1)) + 0.5).astype(np.int64) % (1 << brot)


def decode(words, edges, brot):
    return dm.reconstruct_u_vectors(np.asarray(words) / (1 << (brot - 1)), edges)


def h1_prime(w, Wo, X):
    r = ref()
    S, beta_1 = shifts()
    h1_thc = r["oei"] - 0.5 * np.einsum("prrq->pq", r["tei"])
    return dm.build_h1_prime(h1_thc, S, beta_1, ETA, Wo, X)


def encode_all(U2, w, Wo, edges, brot):
    """Two-body words from U2 (R,B,N); one-body words from eigenvectors of h1' of the decoded X."""
    R, B, _ = U2.shape
    w2 = to_words(tree_angles(U2.reshape(-1, N), edges), brot)
    v2 = decode(w2, edges, brot).reshape(R, B, N)
    X = dm.reconstruct_x(v2, w)
    X = np.tril(X) + np.swapaxes(np.tril(X, k=-1), -1, -2)
    _, ev = np.linalg.eigh(h1_prime(w, Wo, X))
    w1 = to_words(tree_angles(ev.T, edges), brot)
    return np.vstack([w1, w2])


def evaluate(words, w, Wo, edges, brot):
    S, beta_1 = shifts()
    return dm.evaluate_dfthc_angle_words(words, w, Wo, S, beta_1, ref(), brot=brot, tree_edges=edges,
                                         energy_upper=E_UPPER)


def ok(m, margin=0.0):
    return (m["error_1"] <= 0.020 - margin and m["error_2"] <= 0.020 - margin and m["Lambda"] > LAMBDA_FLOOR
            and m["E_gap"] > GAP_FLOOR and m["gap_in_domain"])


if __name__ == "__main__":
    import time
    path, brot = sys.argv[1], int(sys.argv[2])
    z = np.load(path, allow_pickle=False)
    params = {k: z[k] for k in z.files}
    edges = params["tree_edges"].astype(np.int64)
    w, Wo = native(params)
    t = time.time()
    words = encode_all(params["U"], w, Wo, edges, brot)
    m = evaluate(words, w, Wo, edges, brot)
    print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in m.items()}, f"{time.time() - t:.1f}s")
