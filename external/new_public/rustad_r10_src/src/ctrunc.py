"""Reduce the copy count C of a factorization by per-group Gram eigen-truncation.

Group rho contributes sum_c X_rc (x) X_rc = M_rho G_rho M_rho^T with G_rho = W_rho W_rho^T (B x B), so only G matters
for error_2. Keep the top C' eigenpairs: W' = V sqrt(L) (B x C'). Offsets Wo are refit later (finish.py), so they are
set to zero here; follow with cont.py to recover error_2.

  python src/ctrunc.py IN.npz OUT.npz C_NEW
"""
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\rusta\qcc\src")
import dft_eval as de  # noqa: E402


def main(inp, out, c_new):
    z = np.load(inp, allow_pickle=False)
    p = {k: np.array(z[k]) for k in z.files}
    w, Wo = de.native(p)
    R, B, C = w.shape
    w2 = np.zeros((R, B, c_new))
    kept = []
    for r in range(R):
        G = w[r] @ w[r].T
        lam, V = np.linalg.eigh(G)
        idx = np.argsort(-np.abs(lam))[:c_new]
        assert np.all(lam[idx] > -1e-9), "Gram must be PSD"
        w2[r] = V[:, idx] * np.sqrt(np.clip(lam[idx], 0, None))
        kept.append(lam[idx].sum() / lam.clip(0).sum())
    print("kept Gram trace fraction: min %.5f mean %.5f" % (min(kept), np.mean(kept)))
    p.update(W=w2, H2_I=np.zeros((R, c_new)) + w2.sum(axis=1))
    for k in ("angle_words", "words1", "words2", "brot"):
        p.pop(k, None)
    np.savez(out, **p)
    print("saved", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
