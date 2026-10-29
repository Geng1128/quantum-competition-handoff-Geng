"""Quantization-aware training of R = 1 basis angles with variable projection.

Modes:
  anneal: loss = error_2 + mu(t) * mean(sin^2(pi * 2^(brot-1) * theta)), mu grows
          geometrically; all angles are rounded at the end.
  ste:    forward pass uses rounded angles, gradients flow to the latent angles
          (straight-through estimator).
Either way ``w`` is the exact best rank-C PSD fit for the current vectors.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import dfthc_torch as dt
import givens
import refdata
from varpro import best_rank_c, residual_norm


def rounded(theta: torch.Tensor, brot: int) -> torch.Tensor:
    scale = float(1 << (brot - 1))
    return torch.round(theta * scale) / scale


def error_of(problem, theta, edges, C, dtype=torch.float32):
    U = givens.decode_torch(theta, edges)
    w, A = best_rank_c(problem, U, C, dtype=dtype)
    return residual_norm(problem, A, w), w, U


def run(problem, theta0, edges, brot, C, steps, lr, mode, mu0, mu1, log_every):
    theta = theta0.clone().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    scale = float(1 << (brot - 1))
    best = (float("inf"), theta0.clone())
    t0 = time.time()
    for step in range(steps):
        frac = step / max(1, steps - 1)
        opt.zero_grad(set_to_none=True)
        if mode == "ste":
            th = theta + (rounded(theta, brot) - theta).detach()
            err, _, _ = error_of(problem, th, edges, C)
            loss = err
            e_round = float(err.detach())
        else:
            mu = mu0 * (mu1 / mu0) ** frac
            err, _, _ = error_of(problem, theta, edges, C)
            pen = torch.mean(torch.sin(math.pi * scale * theta) ** 2)
            loss = err + mu * pen
            e_round = None
        loss.backward()
        opt.step()
        if mode == "ste" and e_round < best[0]:
            best = (e_round, theta.detach().clone())
        if step % log_every == 0 or step == steps - 1:
            with torch.no_grad():
                er = float(error_of(problem, rounded(theta.detach(), brot), edges, C)[0]) if mode != "ste" else e_round
            msg = f"  step {step} err {float(err):.6f} rounded {er:.6f}"
            if mode != "ste":
                msg += f" pen {float(pen):.4f} mu {mu:.3g}"
                if er < best[0]:
                    best = (er, theta.detach().clone())
            print(msg + f" best {best[0]:.6f} {time.time() - t0:.0f}s", flush=True)
    return best[0], best[1], theta.detach().clone()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--theta-init", type=Path, default=None, help="optional .npy theta/pi to start from")
    parser.add_argument("--B", type=int, required=True)
    parser.add_argument("--C", type=int, required=True)
    parser.add_argument("--brot", type=int, default=8)
    parser.add_argument("--mode", choices=("anneal", "ste"), default="anneal")
    parser.add_argument("--pre-steps", type=int, default=300)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mu0", type=float, default=1e-4)
    parser.add_argument("--mu1", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    ref = refdata.load()
    problem = dt.Problem.build(ref)
    with np.load(args.seed, allow_pickle=False) as data:
        edges = np.array(data["tree_edges"]).astype(np.int64)
        U_seed = np.array(data["U"][0])
        w_seed = np.array(data["W"][0])
    if args.theta_init is not None:
        theta = np.load(args.theta_init)[: args.B]
    else:
        keep = np.sort(np.argsort(-np.linalg.norm(w_seed, axis=1))[: args.B])
        theta = givens.encode_numpy(U_seed[keep], edges)
    theta = torch.tensor(theta, dtype=torch.float64, device=problem.device)

    if args.pre_steps:
        print("continuous pre-training", flush=True)
        _, _, theta = run(problem, theta, edges, args.brot, args.C, args.pre_steps, args.lr, "anneal", 1e-12, 1e-12, max(1, args.pre_steps // 3))
    print(f"QAT mode={args.mode}", flush=True)
    best_err, best_theta, _ = run(problem, theta, edges, args.brot, args.C, args.steps, args.lr, args.mode, args.mu0, args.mu1, max(1, args.steps // 20))
    final = rounded(best_theta, args.brot)
    with torch.no_grad():
        err64, w, U = error_of(problem, final, edges, args.C, dtype=torch.float64)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out.with_suffix(".theta.npy"), best_theta.cpu().numpy())
    np.savez(args.out, words=givens.to_words(final.cpu().numpy(), args.brot), w=w.cpu().numpy(), U=U.cpu().numpy(), tree_edges=edges, brot=np.array(args.brot))
    summary = {"B": args.B, "C": args.C, "brot": args.brot, "mode": args.mode, "rounded_error_2": float(err64)}
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
