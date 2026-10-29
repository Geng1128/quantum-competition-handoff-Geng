"""Variable projection for the R = 1 (THC-like) two-body fit.

With basis vectors ``u_b`` fixed, ``sum_c X_c (x) X_c = A Z A^T`` where
``A[:, b] = vec(u_b u_b^T)`` and ``Z = w w^T`` (B x B, PSD, rank <= C). The
best unconstrained symmetric ``Z`` is ``M^-1 P M^-1`` with ``M = A^T A`` and
``P = A^T T A``; a rank-C PSD truncation gives ``w``. This makes ``error_2``
a function of ``U`` alone, which is cheap to optimize and to quantize.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import dfthc_torch as dt
import givens
import refdata


def design(U: torch.Tensor) -> torch.Tensor:
    """``A`` with columns vec(u_b u_b^T): (N^2, B)."""
    u = dt.normalize_rows(U)
    return torch.einsum("bp,bq->pqb", u, u).reshape(-1, u.shape[0])


def solve_z(problem: dt.Problem, U: torch.Tensor, dtype=torch.float64) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    A = design(U).to(dtype)
    T = problem.target.to(dtype) if dtype != problem.target.dtype else problem.target
    TA = T @ A
    P = A.T @ TA
    u = dt.normalize_rows(U).to(dtype)
    M = (u @ u.T) ** 2
    Minv_P = torch.linalg.solve(M, P)
    Z = torch.linalg.solve(M, Minv_P.T)
    Z = 0.5 * (Z + Z.T)
    return Z, A, M


def truncate(Z: torch.Tensor, C: int) -> torch.Tensor:
    """Rank-C PSD factor ``w`` (B, C) with ``w w^T`` the best PSD rank-C approximation of Z."""
    vals, vecs = torch.linalg.eigh(Z)
    vals, vecs = vals.flip(0)[:C], vecs.flip(1)[:, :C]
    return vecs * torch.sqrt(torch.clamp(vals, min=0.0))


def best_rank_c(problem: dt.Problem, U: torch.Tensor, C: int, dtype=torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact best rank-C PSD fit in the metric of ``A``.

    With ``M = A^T A = L L^T`` and ``A = Q L^T``, ``||A Z A^T - T||`` equals
    ``||L^T Z L - Q^T T Q||`` plus a constant, so the optimum keeps the top-C
    positive eigenpairs of ``L^-1 P L^-T`` and maps back with ``L^-T``.
    """
    A = design(U).to(dtype)
    T = problem.target.to(dtype) if dtype != problem.target.dtype else problem.target
    P = (A.T @ (T @ A)).to(torch.float64)
    u = dt.normalize_rows(U).to(torch.float64)
    M = (u @ u.T) ** 2
    L = torch.linalg.cholesky(M)
    Y = torch.linalg.solve_triangular(L, P, upper=False)
    Y = torch.linalg.solve_triangular(L, Y.T, upper=False)
    Y = 0.5 * (Y + Y.T)
    vals, vecs = torch.linalg.eigh(Y)
    vals, vecs = vals.flip(0)[:C], vecs.flip(1)[:, :C]
    half = vecs * torch.sqrt(torch.clamp(vals, min=0.0))
    w = torch.linalg.solve_triangular(L.T, half, upper=True)
    return w, A.to(torch.float64)


def residual_norm(problem: dt.Problem, A: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    xmat = (A.to(torch.float32) @ w.to(torch.float32)).T  # (C, N^2) rows vec(X_c)
    return torch.linalg.vector_norm(xmat.T @ xmat - problem.target)


def error_for(problem: dt.Problem, U: torch.Tensor, C: int | None) -> tuple[float, torch.Tensor]:
    with torch.no_grad():
        Z, A, _ = solve_z(problem, U)
        w = truncate(Z, C or Z.shape[0])
        return float(residual_norm(problem, A, w)), w


def optimize_u(problem: dt.Problem, U0: torch.Tensor, steps: int, lr: float, C: int | None, log_every: int) -> torch.Tensor:
    """Adam on the (unnormalized) U rows, objective = exact residual of the rank-C fit."""
    U = U0.clone().requires_grad_(True)
    opt = torch.optim.Adam([U], lr=lr)
    best = (float("inf"), U0.clone())
    t0 = time.time()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        Z, A, _ = solve_z(problem, U, dtype=torch.float32)
        w = truncate(Z.to(torch.float64), C or Z.shape[0])
        err = residual_norm(problem, A, w)
        err.backward()
        opt.step()
        e = float(err)
        if e < best[0]:
            best = (e, U.detach().clone())
        if step % log_every == 0 or step == steps - 1:
            print(f"  step {step} error_2 {e:.6f} best {best[0]:.6f} {time.time() - t0:.1f}s", flush=True)
    return best[1]


def quantize_u(U: torch.Tensor, edges: np.ndarray, brot: int) -> torch.Tensor:
    theta = givens.encode_numpy(U.detach().cpu().numpy(), edges)
    q = givens.words_to_theta(givens.to_words(theta, brot), brot)
    return torch.tensor(givens.decode_numpy(q, edges), dtype=U.dtype, device=U.device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--B", type=int, nargs="+", required=True)
    parser.add_argument("--C", type=int, default=None)
    parser.add_argument("--brot", type=int, default=8)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    ref = refdata.load()
    problem = dt.Problem.build(ref)
    with np.load(args.seed, allow_pickle=False) as data:
        U_all = torch.tensor(data["U"][0], dtype=torch.float64, device=problem.device)
        w_seed = np.array(data["W"][0])
        edges = np.array(data["tree_edges"]).astype(np.int64)
    order = np.argsort(-np.linalg.norm(w_seed, axis=1))
    results = []
    for B in args.B:
        U0 = U_all[torch.tensor(np.sort(order[:B]), device=problem.device)]
        e0, _ = error_for(problem, U0, args.C)
        eq0, _ = error_for(problem, quantize_u(U0, edges, args.brot), args.C)
        print(f"B={B}: subset error {e0:.6f}, brot{args.brot}-quantized {eq0:.6f}", flush=True)
        U = optimize_u(problem, U0, args.steps, args.lr, args.C, max(1, args.steps // 6))
        e1, _ = error_for(problem, U, args.C)
        eq1, _ = error_for(problem, quantize_u(U, edges, args.brot), args.C)
        print(f"B={B}: optimized error {e1:.6f}, brot{args.brot}-quantized {eq1:.6f}", flush=True)
        results.append({"B": B, "C": args.C, "subset": e0, "subset_q": eq0, "opt": e1, "opt_q": eq1})
        np.save(args.out.parent / f"{args.out.stem}_U_B{B}.npy", U.cpu().numpy())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
