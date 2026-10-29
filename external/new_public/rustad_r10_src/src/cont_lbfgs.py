"""L-BFGS continuous refit (vectors U normalized in the forward pass, weights w) on error_2 (+ optional D penalty).

  python src/cont_lbfgs.py IN.npz OUT.npz --iters 200 [--dpen 2e-5]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dft_eval as de  # noqa: E402
from qat import Problem, dev  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--inner", type=int, default=25)
    ap.add_argument("--dpen", type=float, default=0.0)
    ap.add_argument("--fp64", action="store_true")
    ap.add_argument("--shrink-group", type=int, default=-1, help="ramp an L2 penalty on this group's weights")
    ap.add_argument("--shrink-pen", default="1e-4,1e2", help="geometric ramp start,end of the shrink penalty")
    ap.add_argument("--eps", type=float, default=0.0, help="smooth |w| as sqrt(w^2 + eps^2) in the D penalty")
    a = ap.parse_args()
    z = np.load(a.inp, allow_pickle=False)
    params = {k: np.array(z[k]) for k in z.files}
    w0, Wo0 = de.native(params)
    R, B, C = w0.shape
    P = Problem(params["tree_edges"].astype(np.int64))
    dt = torch.float64 if a.fp64 else torch.float32
    Tp, cw = P.Tp.to(dt), P.cw.to(dt)
    il0, il1 = P.il_t
    U = torch.tensor(np.ascontiguousarray(params["U"]), dtype=dt, device=dev, requires_grad=True)
    w = torch.tensor(np.ascontiguousarray(w0), dtype=dt, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([U, w], lr=1, max_iter=a.inner, history_size=100, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-12, tolerance_change=1e-15)

    def e2fn():
        u = U / U.norm(dim=2, keepdim=True)
        M = u[:, :, il0] * u[:, :, il1] * cw
        A = torch.einsum("rbc,rbp->rcp", w, M).reshape(R * C, -1)
        return torch.linalg.norm(A.T @ A - Tp)

    pen_box = [0.0]

    def closure():
        opt.zero_grad()
        e = e2fn()
        aw = torch.sqrt(w * w + a.eps ** 2) if a.eps > 0 else w.abs()
        loss = e + a.dpen * 0.5 * ((aw.sum(1)) ** 2 - (w.sum(1)) ** 2).sum()
        if a.shrink_group >= 0:
            loss = loss + pen_box[0] * (w[a.shrink_group] ** 2).sum()
        loss.backward()
        return loss

    t0 = time.time()
    best = None
    p0, p1 = (float(v) for v in a.shrink_pen.split(","))
    for it in range(a.iters):
        if a.shrink_group >= 0:
            pen_box[0] = p0 * (p1 / p0) ** (it / max(1, a.iters - 1))
        opt.step(closure)
        with torch.no_grad():
            e = e2fn().item()
        if a.shrink_group >= 0:  # homotopy: keep the latest state, not the best e2
            best = e
            state = (U.detach().cpu().numpy().astype(np.float64), w.detach().cpu().numpy().astype(np.float64))
        elif best is None or e < best:
            best = e
            state = (U.detach().cpu().numpy().astype(np.float64), w.detach().cpu().numpy().astype(np.float64))
        if it % 10 == 0:
            print(f"{it:5d} e2 {1e3 * e:.4f} mHa best {1e3 * best:.4f} {time.time() - t0:.0f}s", flush=True)
    Ub, wb = state
    Ub = Ub / np.linalg.norm(Ub, axis=2, keepdims=True)
    out = dict(params)
    out.update(U=Ub, W=wb, H2_I=Wo0 + wb.sum(axis=1))
    np.savez(a.out, **out)
    print("saved", a.out, "best e2 mHa", 1e3 * best)


if __name__ == "__main__":
    main()
