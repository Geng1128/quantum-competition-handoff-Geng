"""Merge groups of a DF-THC factorization and compress each merged group to B_new vectors and C_new copies.

For a merged group g (a set of old groups), its exact two-body contribution is A_g^T A_g with A_g = stacked
old (C x P) weight-times-outer-product rows. Choose B_new pooled vectors by column-pivoted QR of their packed outer
products, solve the core zeta_g = Pm^+ (A_g^T A_g) Pm^+T and keep its top C_new positive eigenpairs:
W_g = V sqrt(lam) (B_new x C_new). Offsets Wo start at 0 (finish.py re-optimizes them).

  python src/grp_compress.py IN.npz OUT.npz GROUPS B_new C_new
GROUPS: e.g. "0,1;2,3;4,5;6,7;8,9;10,11,12" (old group indices per merged group)
"""
import sys

import numpy as np
import scipy.linalg as sl

N = 76


def main(inp, out, groups, b_new, c_new):
    z = np.load(inp, allow_pickle=False)
    p = {k: np.array(z[k]) for k in z.files}
    U = p["U"] / np.linalg.norm(p["U"], axis=2, keepdims=True)
    w = np.asarray(p["W"], dtype=np.float64)
    il = np.tril_indices(N)
    cw = np.where(il[0] == il[1], 1.0, np.sqrt(2.0))
    gs = [[int(i) for i in g.split(",")] for g in groups.split(";")]
    Un, Wn = [], []
    for g in gs:
        pool = np.concatenate([U[r] for r in g])  # (sum B, N)
        Pm = (pool[:, il[0]] * pool[:, il[1]] * cw).T  # (P, nb)
        rows = np.concatenate([w[r].T @ (U[r][:, il[0]] * U[r][:, il[1]] * cw) for r in g])  # (sum C, P)
        if len(pool) > b_new:
            _, _, piv = sl.qr(Pm, mode="economic", pivoting=True)
            sel = np.sort(piv[:b_new])
        else:
            sel = np.arange(len(pool))
        A = Pm[:, sel]
        Ap = np.linalg.pinv(A, rcond=1e-10)
        Y = rows @ Ap.T  # (sum C, b): target rows expressed in the selected basis (least squares)
        zeta = Y.T @ Y
        lam, V = np.linalg.eigh(zeta)
        order = np.argsort(-lam)[:c_new]
        Wg = V[:, order] * np.sqrt(np.clip(lam[order], 0, None))
        Ug = pool[sel]
        if len(Ug) < b_new:  # pad with random unit vectors carrying zero weight
            extra = np.random.default_rng(len(Un)).normal(size=(b_new - len(Ug), N))
            Ug = np.vstack([Ug, extra / np.linalg.norm(extra, axis=1, keepdims=True)])
            Wg = np.vstack([Wg, np.zeros((b_new - len(Wg), Wg.shape[1]))])
        if Wg.shape[1] < c_new:
            Wg = np.hstack([Wg, np.zeros((b_new, c_new - Wg.shape[1]))])
        tgt = rows.T @ rows
        approx = (A @ Wg[:len(sel)]) @ (A @ Wg[:len(sel)]).T
        print(f"group {g}: pool {len(pool)} -> {len(sel)} vectors, rank {c_new}, group error "
              f"{1e3 * np.linalg.norm(approx - tgt):.1f} mHa of norm {np.linalg.norm(tgt):.3f}", flush=True)
        Un.append(Ug)
        Wn.append(Wg)
    Un, Wn = np.stack(Un), np.stack(Wn)
    o = {k: p[k] for k in ("H1_I", "Hsym_lower", "tree_edges")}
    o.update(U=Un, W=Wn, H2_I=Wn.sum(axis=1))
    np.savez(out, **o)
    print("saved", out, Un.shape, Wn.shape)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
