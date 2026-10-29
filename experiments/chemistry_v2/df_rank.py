"""DF truncation bound: keep L eigen-matrices, truncate each to rank r, measure error_2."""

from __future__ import annotations

import argparse

import torch

import dfthc_torch as dt
import refdata
from gtrain import residual, top_eigenpairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, nargs="+", default=[320, 400, 512])
    parser.add_argument("--r", type=int, nargs="+", default=[8, 12, 16, 24, 32, 48, 64, 76])
    args = parser.parse_args()
    problem = dt.Problem.build(refdata.load())
    n = problem.n
    vals, vecs = top_eigenpairs(problem, max(args.L))
    Y = vecs.T.reshape(-1, n, n)
    Y = 0.5 * (Y + Y.transpose(1, 2)) * torch.sqrt(torch.clamp(vals, min=0))[:, None, None]
    mu, ev = torch.linalg.eigh(Y)  # (L, n), (L, n, n)
    order = torch.argsort(-mu.abs(), dim=1)
    for L in args.L:
        row = []
        for r in args.r:
            keep = order[:L, :r]
            m = torch.gather(mu[:L], 1, keep)
            e = torch.gather(ev[:L], 2, keep[:, None, :].expand(L, n, r))
            Yr = torch.einsum("lpk,lk,lqk->lpq", e, m, e)
            err = float(residual(problem, Yr.reshape(L, n * n)))
            row.append(f"r={r}:{err:.4f}")
        print(f"L={L}: " + "  ".join(row), flush=True)


if __name__ == "__main__":
    main()
