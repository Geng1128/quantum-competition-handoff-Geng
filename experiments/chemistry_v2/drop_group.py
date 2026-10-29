"""Drop whole groups from a verified factorization, widen C, then retrain continuously.

Start: a submission.npz (angle_words, W, tree_edges, brot) or a continuous
checkpoint (theta, w, tree_edges, brot). Each step removes the group whose
removal raises ||Rm||^2 least, refits w exactly at the new C, and runs
Cartesian Adam rounds with exact block refits between them. The continuous
checkpoint feeds aquant.py for 8-bit re-quantization.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import dfthc_torch as dt
import givens
import refdata
from gtrain import adam_cartesian, projector_design, refit_sweeps, residual, xmat

if str(refdata.VERIFIERS_DIR) not in sys.path:
    sys.path.insert(0, str(refdata.VERIFIERS_DIR))
from dfthcsos.verify import native_inputs  # noqa: E402

N = 76


def load(path: Path, device) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, int]:
    with np.load(path, allow_pickle=False) as data:
        params = {k: np.array(data[k]) for k in data.files}
    edges = params["tree_edges"].astype(np.int64)
    brot = int(params["brot"])
    if "theta" in params:
        theta, w = params["theta"], params["w"]
    else:
        w, _, _, _ = native_inputs(params)
        theta = givens.words_to_theta(params["angle_words"][N:].astype(np.int64), brot)
    return (
        torch.tensor(theta, dtype=torch.float64, device=device),
        torch.tensor(np.asarray(w), dtype=torch.float64, device=device),
        edges,
        brot,
    )


def group_removal_errors(problem, theta, w, edges) -> list[float]:
    R, B, C = w.shape
    with torch.no_grad():
        U = givens.decode_torch(theta, edges).reshape(R, B, -1)
        A = projector_design(U)
        X = xmat(A, w)
        keep = torch.ones(R, dtype=torch.bool, device=w.device)
        out = []
        for r in range(R):
            keep[r] = False
            out.append(float(residual(problem, X.reshape(R, C, -1)[keep].reshape(-1, X.shape[-1]))))
            keep[r] = True
        return out


def drop(problem, theta, w, edges, C_new: int):
    R, B, C = w.shape
    errs = group_removal_errors(problem, theta, w, edges)
    r = int(np.argmin(errs))
    keep = [g for g in range(R) if g != r]
    theta = theta.reshape(R, B, -1)[keep].reshape(-1, theta.shape[-1]).clone()
    w_new = torch.zeros((R - 1, B, C_new), dtype=w.dtype, device=w.device)
    k = min(C, C_new)
    w_new[:, :, :k] = w[keep][:, :, :k]
    err = refit_sweeps(problem, theta, w_new, edges, R - 1, B, C_new, 2)
    return theta, w_new, r, errs[r], err


def drop_vectors(problem, theta, w, edges, per_group: int):
    """Remove, in every group, the vectors with the smallest weight-row norms; refit w."""
    R, B, C = w.shape
    norms = (w * w).sum(2)  # (R, B)
    keep = torch.argsort(norms, dim=1, descending=True)[:, : B - per_group].sort(dim=1).values
    th = theta.reshape(R, B, -1)
    theta = torch.stack([th[r, keep[r]] for r in range(R)]).reshape(R * (B - per_group), -1).clone()
    w = torch.stack([w[r, keep[r]] for r in range(R)]).clone()
    err = refit_sweeps(problem, theta, w, edges, R, B - per_group, C, 2)
    return theta, w, err


def train_rounds(problem, theta, w, edges, rounds: int, steps: int, lr: float, lr_final: float):
    R, B, C = w.shape
    err = float("inf")
    for rnd in range(rounds):
        lr_r = lr * (lr_final / lr) ** (rnd / max(1, rounds - 1))
        _, theta, w = adam_cartesian(problem, theta, w, edges, R, B, C, steps, lr_r)
        err = refit_sweeps(problem, theta, w, edges, R, B, C, 1)
    return theta, w, err


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--target-R", type=int, required=True)
    parser.add_argument("--target-C", type=int, required=True)
    parser.add_argument("--target-B", type=int, default=None)
    parser.add_argument("--vec-step", type=int, default=2, help="vectors removed per group per step")
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--adam-steps", type=int, default=250)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-final", type=float, default=1e-4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    problem = dt.Problem.build(refdata.load())
    theta, w, edges, brot = load(args.start, problem.device)
    R, B, C = w.shape
    err = refit_sweeps(problem, theta, w, edges, R, B, C, 1)
    print(f"start R{R} B{B} C{C}: error_2 {err:.6f}", flush=True)
    while R > args.target_R:
        theta, w, r, raw, err = drop(problem, theta, w, edges, args.target_C)
        R, C = w.shape[0], w.shape[2]
        print(f"dropped group {r}: raw {raw:.6f} -> refit R{R} C{C} {err:.6f}", flush=True)
    if C != args.target_C:
        w_new = torch.zeros((R, B, args.target_C), dtype=w.dtype, device=w.device)
        k = min(C, args.target_C)
        w_new[:, :, :k] = w[:, :, :k]
        w, C = w_new, args.target_C
        err = refit_sweeps(problem, theta, w, edges, R, B, C, 2)
        print(f"resized to C{C}: {err:.6f}", flush=True)
    while args.target_B is not None and B > args.target_B:
        step = min(args.vec_step, B - args.target_B)
        theta, w, err = drop_vectors(problem, theta, w, edges, step)
        B = w.shape[1]
        theta, w, err2 = train_rounds(problem, theta, w, edges, 3, args.adam_steps, args.lr, args.lr * 0.3)
        print(f"B{B}: after drop {err:.6f} retrained {err2:.6f}", flush=True)
    t0 = time.time()
    log = []
    best = float("inf")
    for rnd in range(1, args.rounds + 1):
        lr = args.lr * (args.lr_final / args.lr) ** ((rnd - 1) / max(1, args.rounds - 1))
        e_adam, theta, w = adam_cartesian(problem, theta, w, edges, R, B, C, args.adam_steps, lr)
        err = refit_sweeps(problem, theta, w, edges, R, B, C, 1)
        log.append({"round": rnd, "error_2": err})
        print(f"round {rnd}: adam {e_adam:.6f} refit {err:.6f} ({time.time() - t0:.0f}s)", flush=True)
        if err < best:
            best = err
            args.out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.out, theta=theta.cpu().numpy(), w=w.cpu().numpy(), tree_edges=edges, brot=np.array(brot))
    args.out.with_suffix(".json").write_text(json.dumps({"shape": [R, B, C], "log": log}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
