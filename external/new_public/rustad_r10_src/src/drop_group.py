"""Drop the DF-THC group whose removal raises error_2 least (continuous), optionally resizing C.

  python src/drop_group.py IN.npz OUT.npz [--C 60]
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dft_eval as de  # noqa: E402
from qat import Problem, dev  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--C", type=int, default=0)
    a = ap.parse_args()
    z = np.load(a.inp, allow_pickle=False)
    params = {k: np.array(z[k]) for k in z.files}
    w, Wo = de.native(params)
    R, B, C = w.shape
    P = Problem(params["tree_edges"].astype(np.int64))
    il0, il1 = P.il_t
    U = torch.tensor(params["U"], dtype=torch.float64, device=dev)
    u = U / U.norm(dim=2, keepdim=True)
    M = u[:, :, il0] * u[:, :, il1] * P.cw.double()
    A = torch.einsum("rbc,rbp->rcp", torch.tensor(w, dtype=torch.float64, device=dev), M)
    G = torch.einsum("rcp,rcq->rpq", A, A)
    full = G.sum(0) - P.Tp.double()
    e = [1e3 * torch.linalg.norm(full - G[r]).item() for r in range(R)]
    drop = int(np.argmin(e))
    print("e2 without each group (mHa):", np.round(e, 2), "-> drop", drop)
    keep = [r for r in range(R) if r != drop]
    wk = w[keep]
    Cn = a.C or C
    if Cn > C:
        rng = np.random.default_rng(0)
        wk = np.concatenate([wk, 1e-4 * rng.standard_normal((R - 1, B, Cn - C))], axis=2)
    elif Cn < C:
        wk = wk[:, :, :Cn]
    out = dict(params)
    out.update(U=params["U"][keep], W=wk, H2_I=Wo[keep, :Cn] if Wo.shape[1] >= Cn else
               np.concatenate([Wo[keep], np.zeros((R - 1, Cn - C))], axis=1))
    out["H2_I"] = out["H2_I"] + wk.sum(axis=1)
    np.savez(a.out, **out)
    print("saved", a.out, wk.shape)


if __name__ == "__main__":
    main()
