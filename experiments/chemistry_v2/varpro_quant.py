"""Progressive brot-bit quantization of R = 1 basis vectors with variable projection.

The basis vectors are parametrized by Givens-tree angles. Stage s freezes a
growing fraction of the angles at their rounded ``brot``-bit values (smallest
rounding residual first) and retrains the free angles; ``w`` is always the
closed-form rank-C fit, so only ``U`` is trained.
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
from varpro import best_rank_c, residual_norm


def theta_eff(theta: torch.Tensor, frozen: torch.Tensor, brot: int) -> torch.Tensor:
    scale = float(1 << (brot - 1))
    return torch.where(frozen, torch.round(theta.detach() * scale) / scale, theta)


def fit_error(problem, theta, frozen, edges, brot, C, dtype=torch.float32):
    U = givens.decode_torch(theta_eff(theta, frozen, brot), edges)
    w, A = best_rank_c(problem, U, C, dtype=dtype)
    return residual_norm(problem, A, w), w, U


def train(problem, theta, frozen, edges, brot, C, steps, lr, tag, log_every):
    theta = theta.clone().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps), eta_min=lr * 0.1)
    best_err, best_theta = float("inf"), theta.detach().clone()
    t0 = time.time()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        err, _, _ = fit_error(problem, theta, frozen, edges, brot, C)
        e = float(err.detach())
        if e < best_err:
            best_err, best_theta = e, theta.detach().clone()
        err.backward()
        theta.grad[frozen] = 0.0
        opt.step()
        sched.step()
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"  [{tag}] step {step} error_2 {e:.6f} best {best_err:.6f} {time.time() - t0:.1f}s", flush=True)
    return best_theta, best_err


ORDER = "small"


def freeze_more(theta, frozen, fraction, brot):
    scale = float(1 << (brot - 1))
    resid = torch.abs(theta * scale - torch.round(theta * scale))
    if ORDER == "large":
        resid = -resid
    elif ORDER == "random":
        resid = torch.rand_like(resid)
    total = frozen.numel()
    need = int(round(fraction * total)) - int(frozen.sum())
    if need <= 0:
        return frozen
    flat_frozen = frozen.reshape(-1).clone()
    free_idx = torch.nonzero(~flat_frozen).squeeze(1)
    order = torch.argsort(resid.reshape(-1)[free_idx])[:need]
    flat_frozen[free_idx[order]] = True
    return flat_frozen.reshape(frozen.shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-u", type=Path, required=True, help=".npy (B, N) or factorization .npz (R=1)")
    parser.add_argument("--edges-from", type=Path, required=True)
    parser.add_argument("--B", type=int, required=True)
    parser.add_argument("--C", type=int, required=True)
    parser.add_argument("--brot", type=int, default=8)
    parser.add_argument("--pre-steps", type=int, default=600)
    parser.add_argument("--stages", type=int, default=20)
    parser.add_argument("--stage-steps", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--order", choices=("small", "large", "random"), default="small")
    parser.add_argument("--exact-basis", action="store_true", help="prepend the 76 lattice-exact e_i vectors (frozen)")
    args = parser.parse_args()
    global ORDER
    ORDER = args.order

    ref = refdata.load()
    problem = dt.Problem.build(ref)
    with np.load(args.edges_from, allow_pickle=False) as data:
        edges = np.array(data["tree_edges"]).astype(np.int64)
        seed_w = np.array(data["W"][0]) if args.seed_u.suffix == ".npz" else None
        seed_U = np.array(data["U"][0]) if args.seed_u.suffix == ".npz" else None
    if seed_U is None:
        seed_U = np.load(args.seed_u)
        order = np.arange(seed_U.shape[0])
    else:
        order = np.argsort(-np.linalg.norm(seed_w, axis=1))
    n_exact = 76 if args.exact_basis else 0
    U0 = seed_U[np.sort(order[: args.B - n_exact])]
    if n_exact:
        U0 = np.concatenate([np.eye(76), U0], axis=0)
    theta = torch.tensor(givens.encode_numpy(U0, edges), dtype=torch.float64, device=problem.device)
    frozen = torch.zeros_like(theta, dtype=torch.bool)
    if n_exact:
        scale = float(1 << (args.brot - 1))
        assert torch.allclose(theta[:n_exact] * scale, torch.round(theta[:n_exact] * scale)), "basis angles not on the lattice"
        frozen[:n_exact] = True

    log = []
    theta, err = train(problem, theta, frozen, edges, args.brot, args.C, args.pre_steps, args.lr, "continuous", max(1, args.pre_steps // 4))
    log.append({"stage": 0, "frozen": 0.0, "error_2": err})
    for s in range(1, args.stages + 1):
        # Large freezing steps first, small ones last: f(s) = 1 - (1 - s/S)^2.
        frozen = freeze_more(theta, frozen, 1.0 - (1.0 - s / args.stages) ** 2, args.brot)
        steps = args.stage_steps if s < args.stages else 0
        theta, err = train(problem, theta, frozen, edges, args.brot, args.C, steps, args.lr * 0.5, f"q{s}", 0)
        if steps == 0:
            with torch.no_grad():
                err = float(fit_error(problem, theta, frozen, edges, args.brot, args.C, dtype=torch.float64)[0])
        print(f"stage {s}: frozen {float(frozen.float().mean()):.3f} error_2 {err:.6f}", flush=True)
        log.append({"stage": s, "frozen": float(frozen.float().mean()), "error_2": err})

    with torch.no_grad():
        err64, w, U = fit_error(problem, theta, frozen, edges, args.brot, args.C, dtype=torch.float64)
    words = givens.to_words(theta_eff(theta, frozen, args.brot).cpu().numpy(), args.brot)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, words=words, w=w.cpu().numpy(), U=U.cpu().numpy(), tree_edges=edges, brot=np.array(args.brot))
    summary = {"B": args.B, "C": args.C, "brot": args.brot, "final_error_2": float(err64), "log": log}
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "log"}))


if __name__ == "__main__":
    main()
