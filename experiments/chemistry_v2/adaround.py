"""Joint learned rounding (AdaRound-style) of all Givens angles to the brot-bit lattice.

Each angle is floor(theta * s) / s plus a learned offset h(V) / s in [0, 1],
h a rectified sigmoid; a regularizer annealed from soft to hard pushes every h
to {0, 1}, so all rounding directions are chosen together while w follows.
w is refit exactly every few hundred steps. Output words feed dsearch.py and
polish_weights.py.
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

ZETA, GAMMA = 1.1, -0.1


def h_of(V: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.sigmoid(V) * (ZETA - GAMMA) + GAMMA, 0.0, 1.0)


def v_init(frac: torch.Tensor) -> torch.Tensor:
    p = ((frac - GAMMA) / (ZETA - GAMMA)).clamp(1e-4, 1 - 1e-4)
    return torch.log(p / (1 - p))


def lattice_error(problem, base, bits, s, edges, R, B, w, sweeps=2):
    with torch.no_grad():
        U = givens.decode_torch((base + bits) / s, edges).reshape(R, B, -1)
        A = projector_design(U)
        for _ in range(sweeps):
            for r in range(R):
                w[r] = refit_block(problem, A, w, r, w.shape[2])
        return float(residual(problem, xmat(A, w)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--brot", type=int, default=8)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--w-lr", type=float, default=1e-5)
    parser.add_argument("--alpha", type=float, default=0.3, help="regularizer weight relative to start residual^2")
    parser.add_argument("--beta-start", type=float, default=20.0)
    parser.add_argument("--beta-end", type=float, default=2.0)
    parser.add_argument("--warmup", type=float, default=0.1, help="fraction of steps without regularizer")
    parser.add_argument("--refit-every", type=int, default=250)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    problem = dt.Problem.build(refdata.load())
    dev = problem.device
    with np.load(args.start, allow_pickle=False) as data:
        theta = torch.tensor(np.array(data["theta"]), dtype=torch.float64, device=dev)
        w = torch.tensor(np.array(data["w"]), dtype=torch.float64, device=dev)
        edges = np.array(data["tree_edges"]).astype(np.int64)
    R, B, C = w.shape
    s = float(1 << (args.brot - 1))
    base = torch.floor(theta * s)
    frac = theta * s - base
    nearest = (frac >= 0.5).to(theta.dtype)
    print(f"R{R} B{B} C{C}: nearest rounding {lattice_error(problem, base, nearest, s, edges, R, B, w.clone()):.6f}", flush=True)
    V = v_init(frac).requires_grad_(True)
    w = w.clone().requires_grad_(True)
    opt = torch.optim.Adam([{"params": [V], "lr": args.lr}, {"params": [w], "lr": args.w_lr}])
    res0 = None
    t0 = time.time()
    log = []
    best = (float("inf"), None, None)
    warm = int(args.warmup * args.steps)
    for step in range(args.steps + 1):
        opt.zero_grad(set_to_none=True)
        h = h_of(V)
        U = givens.decode_torch((base + h) / s, edges).reshape(R, B, -1)
        res = residual(problem, xmat(projector_design(U), w))
        if res0 is None:
            res0 = float(res.detach())
        if step < warm:
            reg = torch.zeros((), device=dev, dtype=res.dtype)
            beta = args.beta_start
        else:
            t = (step - warm) / max(1, args.steps - warm)
            beta = args.beta_start + (args.beta_end - args.beta_start) * t
            reg = (1 - (2 * h - 1).abs().pow(beta)).mean()
        loss = res**2 + args.alpha * res0**2 * reg
        loss.backward()
        opt.step()
        if step % args.refit_every == 0 or step == args.steps:
            with torch.no_grad():
                hard = (h_of(V) > 0.5).to(theta.dtype)
                soft_done = float(((h_of(V) - 0.5).abs() > 0.49).to(torch.float64).mean())
                wq = w.detach().clone()
                err = lattice_error(problem, base, hard, s, edges, R, B, wq)
                if step >= warm:
                    w.data.copy_(wq)  # exact refit at the current hard lattice point
            log.append({"step": step, "soft": float(res.detach()), "hard": err, "settled": soft_done, "beta": beta})
            print(f"step {step}: soft {float(res.detach()):.6f} hard {err:.6f} settled {soft_done:.3f} beta {beta:.1f} ({time.time() - t0:.0f}s)", flush=True)
            if err < best[0]:
                best = (err, hard.clone(), wq.clone())
                words = givens.to_words(((base + hard) / s).cpu().numpy(), args.brot)
                args.out.parent.mkdir(parents=True, exist_ok=True)
                np.savez(args.out, words=words, w=wq.cpu().numpy(), tree_edges=edges, brot=np.array(args.brot), error_2=np.array(err))
    args.out.with_suffix(".json").write_text(json.dumps({"best": best[0], "log": log}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
