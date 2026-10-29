"""Long L-BFGS continuous refit on Cartesian vectors (normalized in the forward pass) and w.

Optional homotopy group removal: --shrink-group ramps an L2 penalty on that
group's weights so the others absorb it gradually, after which the group is
cut (R -> R-1) and the remaining factorization is refit.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import dfthc_torch as dt
import givens
import refdata
from drop_group import load
from gtrain import projector_design, refit_sweeps, residual, xmat


def save(path: Path, U: torch.Tensor, w: torch.Tensor, edges: np.ndarray, brot: int) -> None:
    R, B, N = U.shape
    u = dt.normalize_rows(U.detach().reshape(-1, N)).cpu().numpy()
    theta = givens.encode_numpy(u, edges)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, theta=theta, w=w.detach().cpu().numpy(), tree_edges=edges, brot=np.array(brot))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--inner", type=int, default=25)
    parser.add_argument("--shrink-group", type=int, default=-1)
    parser.add_argument("--shrink-pen", default="1e-4,1e2", help="geometric ramp start,end")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    problem = dt.Problem.build(refdata.load())
    theta, w, edges, brot = load(args.start, problem.device)
    R, B, C = w.shape
    U = givens.decode_torch(theta, edges).reshape(R, B, -1).to(torch.float32).clone().requires_grad_(True)
    w = w.to(torch.float32).clone().requires_grad_(True)
    opt = torch.optim.LBFGS([U, w], lr=1, max_iter=args.inner, history_size=100, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-12, tolerance_change=1e-15)
    pen = [0.0]

    def err_fn():
        return residual(problem, xmat(projector_design(dt.normalize_rows(U)), w))

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = err_fn()
        if args.shrink_group >= 0:
            loss = loss + pen[0] * (w[args.shrink_group] ** 2).sum()
        loss.backward()
        return loss

    p0, p1 = (float(v) for v in args.shrink_pen.split(","))
    best = float("inf")
    t0 = time.time()
    for it in range(args.iters):
        if args.shrink_group >= 0:
            pen[0] = p0 * (p1 / p0) ** (it / max(1, args.iters - 1))
        opt.step(closure)
        with torch.no_grad():
            e = float(err_fn())
        if args.shrink_group < 0 and e < best:
            best = e
            save(args.out, U, w, edges, brot)
        if it % 10 == 0:
            gnorm = float(w[args.shrink_group].norm()) if args.shrink_group >= 0 else 0.0
            print(f"{it:5d} e2 {e:.6f} best {best:.6f} shrink |w_g| {gnorm:.3e} pen {pen[0]:.2e} {time.time() - t0:.0f}s", flush=True)
    if args.shrink_group >= 0:
        keep = [g for g in range(R) if g != args.shrink_group]
        theta = givens.encode_numpy(dt.normalize_rows(U.detach()[keep].reshape(-1, U.shape[2])).cpu().numpy(), edges)
        theta = torch.tensor(theta, dtype=torch.float64, device=problem.device)
        w64 = w.detach()[keep].to(torch.float64).clone()
        e = refit_sweeps(problem, theta, w64, edges, R - 1, B, C, 2)
        print(f"cut group {args.shrink_group}: R{R - 1} refit {e:.6f}", flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.out, theta=theta.cpu().numpy(), w=w64.cpu().numpy(), tree_edges=edges, brot=np.array(brot))


if __name__ == "__main__":
    main()
