"""Gram-preserving column rotations of a factorization's weights to lower the LCU one-norm.

Per group rho, the stacked matrix a = [w_rho; wB_rho] (B+1 rows, C columns) can be multiplied on the right by any
orthogonal C x C matrix without changing a a^T, hence without changing error_2, the absorbed one-body term
(so h1' and error_1), or sum_c Wo_c^2 (so E_gap). The two-body one-norm is 1/4 sum_c (||a_c||_1)^2, so we lower it with
exact pairwise Givens moves: for a column pair (i, j) the loss is piecewise smooth in the angle and its minimum lies at
an angle where some entry of the rotated pair is zero; all those candidate angles are checked.

  python src/gauge.py IN.npz OUT.npz [--sweeps 40]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def l1sq(a):
    return float(np.sum(np.abs(a).sum(axis=1) ** 2))


def rotate(a, sweeps=40, seed=0):
    """a: (R, B+1, C). Returns rotated copy and number of accepted moves."""
    a = a.copy()
    R, _, C = a.shape
    rng = np.random.default_rng(seed)
    accepted = 0
    for _ in range(sweeps):
        pairs = [(i, j) for i in range(C) for j in range(i + 1, C)]
        rng.shuffle(pairs)
        for i, j in pairs:
            x, y = a[:, :, i], a[:, :, j]
            cand = np.mod(np.arctan2(y, x), np.pi / 2)  # angles zeroing some entry (period pi/2 for |.| sums)
            cand = np.concatenate([np.zeros((R, 1)), cand], axis=1)
            co, si = np.cos(cand)[:, :, None], np.sin(cand)[:, :, None]
            loss = (np.abs(co * x[:, None] + si * y[:, None]).sum(2) ** 2
                    + np.abs(co * y[:, None] - si * x[:, None]).sum(2) ** 2)
            k = loss.argmin(axis=1)
            th = cand[np.arange(R), k]
            old = np.abs(x).sum(1) ** 2 + np.abs(y).sum(1) ** 2
            th = np.where(loss[np.arange(R), k] < old - 1e-12, th, 0.0)
            c, s = np.cos(th)[:, None], np.sin(th)[:, None]
            a[:, :, i], a[:, :, j] = c * x + s * y, c * y - s * x
            accepted += int(np.count_nonzero(th))
    return a, accepted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--sweeps", type=int, default=40)
    a_ = ap.parse_args()
    z = np.load(a_.inp, allow_pickle=False)
    p = {k: np.array(z[k]) for k in z.files}
    a = np.concatenate([p["W"], p["H2_I"][:, None, :]], axis=1)
    g0 = a @ a.swapaxes(1, 2)
    b, acc = rotate(a, a_.sweeps)
    drift = float(np.max(np.abs(b @ b.swapaxes(1, 2) - g0)))
    assert drift < 1e-8, drift
    print(f"one-norm term {l1sq(a):.3f} -> {l1sq(b):.3f}, accepted {acc}, Gram drift {drift:.1e}")
    p["W"], p["H2_I"] = b[:, :-1, :], b[:, -1, :]
    np.savez(a_.out, **p)
    print("saved", a_.out)


if __name__ == "__main__":
    main()
