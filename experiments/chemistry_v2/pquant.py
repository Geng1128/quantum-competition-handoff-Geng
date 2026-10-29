"""Vector-wise progressive brot-bit quantization with continuous compensation.

Stage s rounds the cheapest still-continuous vectors (exact cost of rounding
each vector with w fixed) and freezes them on the lattice; the remaining
continuous vectors (Cartesian) and w are retrained with Adam, then w is refit
exactly. The final words go to dsearch.py for discrete polishing.
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
from gtrain import projector_design, refit_block, residual, xmat


def lattice(U: torch.Tensor, edges: np.ndarray, brot: int) -> torch.Tensor:
    """Nearest-word lattice vectors for rows of U: (M, N) -> (M, N)."""
    theta = givens.encode_numpy(dt.normalize_rows(U).cpu().numpy(), edges)
    q = givens.words_to_theta(givens.to_words(theta, brot), brot)
    return torch.tensor(givens.decode_numpy(q, edges), dtype=U.dtype, device=U.device)


def rounding_costs(problem, U, w, Uq) -> torch.Tensor:
    """Exact dE of replacing each vector by its lattice rounding (w fixed); shape (R, B)."""
    R, B, N = U.shape
    n2 = N * N
    A = projector_design(dt.normalize_rows(U))
    X = xmat(A, w).reshape(R, w.shape[2], -1).to(torch.float32)
    flat = X.reshape(-1, n2)
    Rm = flat.T @ flat - problem.target
    out = torch.empty(R, B, dtype=torch.float64, device=U.device)
    for r in range(R):
        u = dt.normalize_rows(U[r]).to(torch.float32)
        uq = Uq[r].to(torch.float32)
        V = (uq[:, :, None] * uq[:, None, :] - u[:, :, None] * u[:, None, :]).reshape(B, n2)
        Wr = w[r].to(torch.float32)
        Amat = Wr @ X[r]
        s = (Wr * Wr).sum(1)
        RV = V @ Rm
        vrho = (RV * Amat).sum(1)
        vRv = (RV * V).sum(1)
        vv = (V * V).sum(1)
        av = (Amat * V).sum(1)
        aa = (Amat * Amat).sum(1)
        out[r] = (4 * vrho + 2 * s * vRv + 2 * (aa * vv + av**2) + 4 * s * av * vv + s * s * vv * vv).to(torch.float64)
    return out


def refit_all(problem, U, w, sweeps=1):
    with torch.no_grad():
        A = projector_design(dt.normalize_rows(U))
        for _ in range(sweeps):
            for r in range(U.shape[0]):
                w[r] = refit_block(problem, A, w, r, w.shape[2])
        return float(residual(problem, xmat(A, w)))


def train_free(problem, U, w, frozen, steps, lr):
    """Adam on unfrozen rows of U (Cartesian) and on w."""
    U = U.clone().requires_grad_(True)
    w = w.clone().requires_grad_(True)
    opt = torch.optim.Adam([{"params": [U], "lr": lr}, {"params": [w], "lr": lr * 0.1}])
    mask = (~frozen).to(U.dtype)[:, :, None]
    best = (float("inf"), U.detach().clone(), w.detach().clone())
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        err = residual(problem, xmat(projector_design(dt.normalize_rows(U)), w))
        e = float(err.detach())
        if e < best[0]:
            best = (e, U.detach().clone(), w.detach().clone())
        err.backward()
        U.grad.mul_(mask)
        opt.step()
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=Path, required=True, help="continuous checkpoint (theta, w, tree_edges[, copy_index])")
    parser.add_argument("--brot", type=int, default=8)
    parser.add_argument("--stages", type=int, default=20)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--order", choices=("cheap", "costly"), default="cheap")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    problem = dt.Problem.build(refdata.load())
    dev = problem.device
    with np.load(args.start, allow_pickle=False) as data:
        theta = np.array(data["theta"])
        w = torch.tensor(np.array(data["w"]), dtype=torch.float64, device=dev)
        edges = np.array(data["tree_edges"]).astype(np.int64)
        extra = {"copy_index": np.array(data["copy_index"])} if "copy_index" in data.files else {}
    R, B, C = w.shape
    U = torch.tensor(givens.decode_numpy(theta, edges).reshape(R, B, 76), dtype=torch.float64, device=dev)
    frozen = torch.zeros(R, B, dtype=torch.bool, device=dev)
    err = refit_all(problem, U, w, 2)
    print(f"start R{R} B{B} C{C} continuous error_2 {err:.6f}", flush=True)
    t0 = time.time()
    log = []
    for s in range(1, args.stages + 1):
        target = int(round(R * B * (1.0 - (1.0 - s / args.stages) ** 2)))
        need = target - int(frozen.sum())
        if need > 0:
            Uq = lattice(U.reshape(R * B, 76), edges, args.brot).reshape(R, B, 76)
            cost = rounding_costs(problem, U, w, Uq)
            cost = torch.where(frozen, torch.full_like(cost, float("inf") if args.order == "cheap" else -float("inf")), cost)
            order = torch.argsort(cost.reshape(-1), descending=(args.order == "costly"))[:need]
            fr = frozen.reshape(-1).clone()
            fr[order] = True
            frozen = fr.reshape(R, B)
            U = torch.where(frozen[:, :, None], Uq, U)
        rounded = refit_all(problem, U, w, 1)
        if s < args.stages:
            e, U, w = train_free(problem, U, w, frozen, args.steps, args.lr)
        err = refit_all(problem, U, w, 1)
        print(f"stage {s}: frozen {int(frozen.sum())}/{R * B} after-round {rounded:.6f} retrained {err:.6f} ({time.time() - t0:.0f}s)", flush=True)
        log.append({"stage": s, "frozen": int(frozen.sum()), "rounded": rounded, "error_2": err})
    words = givens.to_words(givens.encode_numpy(dt.normalize_rows(U).reshape(R * B, 76).cpu().numpy(), edges), args.brot)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, words=words, w=w.cpu().numpy(), tree_edges=edges, brot=np.array(args.brot), **extra)
    args.out.with_suffix(".json").write_text(json.dumps({"log": log}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
