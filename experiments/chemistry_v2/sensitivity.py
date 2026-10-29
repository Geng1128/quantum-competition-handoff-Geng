"""How sensitive is error_2 to small angle perturbations, with and without refit?"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

import dfthc_torch as dt
import givens
import refdata
from varpro import best_rank_c, residual_norm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--brot", type=int, default=8)
    args = parser.parse_args()
    ref = refdata.load()
    problem = dt.Problem.build(ref)
    with np.load(args.seed, allow_pickle=False) as data:
        edges = np.array(data["tree_edges"]).astype(np.int64)
        U = np.array(data["U"][0])
        w_seed = torch.tensor(np.array(data["W"][0]), dtype=torch.float64, device=problem.device)
    theta = givens.encode_numpy(U, edges)
    ulp = 1.0 / (1 << (args.brot - 1))
    rng = np.random.default_rng(0)
    C = w_seed.shape[1]
    with torch.no_grad():
        U0 = torch.tensor(givens.decode_numpy(theta, edges), device=problem.device)
        w_fit, A0 = best_rank_c(problem, U0, C)
        print("seed w: |w| sum", float(w_seed.abs().sum()), "max", float(w_seed.abs().max()))
        print("fit  w: |w| sum", float(w_fit.abs().sum()), "max", float(w_fit.abs().max()))
        print("seed error", float(residual_norm(problem, A0, w_seed)), "fit error", float(residual_norm(problem, A0, w_fit)))
        for frac in (0.01, 0.05, 0.1, 0.25, 0.5):
            noise = rng.uniform(-frac, frac, theta.shape) * ulp
            Up = torch.tensor(givens.decode_numpy(theta + noise, edges), device=problem.device)
            Ap = dt.normalize_rows(Up)
            Ap = torch.einsum("bp,bq->pqb", Ap, Ap).reshape(-1, Ap.shape[0])
            e_seed_w = float(residual_norm(problem, Ap, w_seed))
            e_fixw = float(residual_norm(problem, Ap, w_fit))
            w_re, A_re = best_rank_c(problem, Up, C)
            e_refit = float(residual_norm(problem, A_re, w_re))
            print(f"noise +-{frac} ulp: seed_w {e_seed_w:.5f}  fit_w {e_fixw:.5f}  refit {e_refit:.5f}")


if __name__ == "__main__":
    main()
