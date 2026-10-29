"""Remove copies from a verified factorization, then retrain continuously.

Copies are removed by exact removal cost (one at a time per group), followed by
Cartesian Adam on the vectors and w with exact block refits between rounds.
The continuous checkpoint (theta, w, copy_index) feeds dsearch.py for 8-bit
re-quantization.
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
from dsearch import Search
from gtrain import adam_cartesian, refit_sweeps
from prune import load_start


def remove_copy_continuous(problem, theta, w, edges):
    """Drop, in every group, the copy whose removal raises ||Rm||^2 least; refit w."""
    from gtrain import projector_design, xmat

    R, B, C = w.shape
    with torch.no_grad():
        U = givens.decode_torch(theta, edges).reshape(R, B, -1)
        X = xmat(projector_design(U), w).reshape(R, C, -1).to(torch.float32)
        flat = X.reshape(-1, X.shape[-1])
        Rm = flat.T @ flat - problem.target
        keep = []
        for r in range(R):
            cost = -2 * ((X[r] @ Rm) * X[r]).sum(1) + (X[r] * X[r]).sum(1) ** 2
            drop = int(torch.argmin(cost))
            keep.append([c for c in range(C) if c != drop])
        w = torch.stack([w[r][:, keep[r]] for r in range(R)])
    refit_sweeps(problem, theta, w, edges, R, B, C - 1, 1)
    return theta, w, keep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--target-C", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--adam-steps", type=int, default=250)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-final", type=float, default=1e-4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    problem = dt.Problem.build(refdata.load())
    if args.start.suffix == ".npz" and "theta" in np.load(args.start).files:
        with np.load(args.start, allow_pickle=False) as data:
            theta = torch.tensor(np.array(data["theta"]), dtype=torch.float64, device=problem.device)
            w = torch.tensor(np.array(data["w"]), dtype=torch.float64, device=problem.device)
            edges = np.array(data["tree_edges"]).astype(np.int64)
            brot = int(data["brot"])
            base_index = np.array(data["copy_index"]) if "copy_index" in data.files else None
        R, B, C = w.shape
        copy_index = base_index if base_index is not None else np.tile(np.arange(C), (R, 1))
        while C > args.target_C:
            theta, w, keep = remove_copy_continuous(problem, theta, w, edges)
            copy_index = np.stack([copy_index[r][keep[r]] for r in range(R)])
            C -= 1
        print(f"continuous start, removed to C{C}: error_2 {refit_sweeps(problem, theta, w, edges, R, B, C, 1):.6f}", flush=True)
    else:
        words, w, edges, brot = load_start(args.start, problem.device)
        s = Search(problem, words, w, edges, brot)
        s.refit(2)
        print(f"start R{s.R} B{s.B} C{s.C} error_2 {s.error():.6f}", flush=True)
        while s.C > args.target_C:
            s.remove_copies(1)
            s.refit(1)
        print(f"after removal to C{s.C}: error_2 {s.error():.6f}", flush=True)
        R, B, C = s.R, s.B, s.C
        theta = (s.words.to(torch.float64) / float(1 << (brot - 1))).clone()
        w = s.w.clone()
        copy_index = getattr(s, "copy_index", np.tile(np.arange(C), (R, 1)))
    t0 = time.time()
    log = []
    best = (float("inf"), None, None)
    for rnd in range(1, args.rounds + 1):
        lr = args.lr * (args.lr_final / args.lr) ** ((rnd - 1) / max(1, args.rounds - 1))
        e_adam, theta, w = adam_cartesian(problem, theta, w, edges, R, B, C, args.adam_steps, lr)
        err = refit_sweeps(problem, theta, w, edges, R, B, C, 1)
        log.append({"round": rnd, "error_2": err})
        print(f"round {rnd}: adam {e_adam:.6f} refit {err:.6f} ({time.time() - t0:.0f}s)", flush=True)
        if err < best[0]:
            best = (err, theta.clone(), w.clone())
            args.out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.out, theta=best[1].cpu().numpy(), w=best[2].cpu().numpy(), tree_edges=edges,
                     brot=np.array(brot), copy_index=copy_index)
    args.out.with_suffix(".json").write_text(json.dumps({"shape": [R, B, C], "log": log}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
