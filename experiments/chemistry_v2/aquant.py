"""Angle-level progressive quantization (general R) with loss-aware rounding.

Each stage freezes a growing fraction of the free angles on the brot-bit
lattice, then retrains the free angles and w with Adam; w is refit exactly
between stages. With --direction loss, every candidate angle is snapped to
floor or ceil by the local quadratic model dE = g*d + h*d^2 (g the exact
gradient of the squared residual, h a diagonal Gauss-Newton estimate from
random residual probes), and angles are frozen in order of that predicted
damage. --direction nearest reproduces plain rounding.
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


def theta_eff(theta, frozen, qval):
    return torch.where(frozen, qval, theta)


def model_x(theta, frozen, qval, edges, R, B, w):
    U = givens.decode_torch(theta_eff(theta, frozen, qval), edges).reshape(R, B, -1)
    return xmat(projector_design(U), w)


def curvature(problem, theta, frozen, qval, edges, R, B, w, probes):
    """Diagonal Gauss-Newton estimate from rank-1 Hutchinson probes.

    A full random ``n²×n²`` probe makes ``X @ Z`` needlessly expensive.  A
    rank-1 probe ``v vᵀ`` gives the same type of randomized sensitivity signal
    through ``||X v||²`` while reducing the hot path to a matrix-vector product.
    """
    n2 = problem.n * problem.n
    if probes <= 0:
        return torch.zeros_like(theta)
    th = theta.clone().requires_grad_(True)
    diag = torch.zeros_like(theta)
    for _ in range(probes):
        v = torch.empty(n2, device=theta.device, dtype=torch.float32).bernoulli_(0.5).mul_(2).sub_(1)
        X = model_x(th, frozen, qval, edges, R, B, w).to(torch.float32)
        val = ((X @ v) ** 2).sum()
        (g,) = torch.autograd.grad(val, th)
        diag += g.detach() ** 2
        del v, X, val
        torch.cuda.empty_cache()
    return diag / probes


def hessian_diag(problem, theta, frozen, qval, edges, R, B, w, probes):
    """Hutchinson diagonal of the exact Hessian of ``error_2**2``.

    This is the same second-order signal used by progressive quantizers in
    model compression.  It ranks lattice moves by local curvature instead of
    the cheaper rank-one sensitivity proxy in :func:`curvature`.
    """
    if probes <= 0:
        return torch.zeros_like(theta)
    th = theta.clone().requires_grad_(True)
    err = residual(problem, model_x(th, frozen, qval, edges, R, B, w))
    (g,) = torch.autograd.grad(err ** 2, th, create_graph=True)
    diag = torch.zeros_like(theta)
    for _ in range(probes):
        v = torch.randint(0, 2, th.shape, device=th.device, dtype=torch.int64).to(th.dtype).mul_(2).sub_(1)
        (hv,) = torch.autograd.grad(g, th, grad_outputs=v, retain_graph=True)
        diag += v * hv
    return (diag / probes).detach()


def gradient(problem, theta, frozen, qval, edges, R, B, w):
    th = theta.clone().requires_grad_(True)
    err = residual(problem, model_x(th, frozen, qval, edges, R, B, w))
    (g,) = torch.autograd.grad(err**2, th)
    return g.detach()


def refit(problem, theta, frozen, qval, edges, R, B, w):
    with torch.no_grad():
        U = givens.decode_torch(theta_eff(theta, frozen, qval), edges).reshape(R, B, -1)
        A = projector_design(U)
        for r in range(R):
            w[r] = refit_block(problem, A, w, r, w.shape[2])
        return float(residual(problem, xmat(A, w)))


def train(problem, theta, frozen, qval, edges, R, B, w, steps, lr_th, lr_w, decay=1.0,
          dpen=0.0):
    theta = theta.clone().requires_grad_(True)
    w = w.clone().requires_grad_(True)
    opt = torch.optim.Adam([{"params": [theta], "lr": lr_th}, {"params": [w], "lr": lr_w}])
    # Keep the lowest physical error, while still optimizing the optional D(w)
    # regularizer in the line search.  Selecting by the penalized objective can
    # otherwise trade a large error_2 increase for a modest penalty reduction.
    best = (float("inf"), theta.detach().clone(), w.detach().clone())
    for it in range(steps):
        opt.zero_grad(set_to_none=True)
        err = residual(problem, model_x(theta, frozen, qval, edges, R, B, w))
        penalty = 0.5 * ((w.abs().sum(1)) ** 2 - (w.sum(1)) ** 2).sum() if dpen else 0.0
        objective = err + dpen * penalty if dpen else err
        e = float(err.detach())
        if e < best[0]:
            best = (e, theta.detach().clone(), w.detach().clone())
        objective.backward()
        theta.grad[frozen] = 0.0
        opt.step()
        if decay != 1.0 and steps > 0:
            # Geometric decay reaches ``decay`` times the initial lr at stage end.
            mult = decay ** (1.0 / steps)
            for group in opt.param_groups:
                group["lr"] *= mult
    return best


def train_lbfgs(problem, theta, frozen, qval, edges, R, B, w, outer_steps,
                inner_steps, dpen=0.0):
    """Compensate newly frozen angles with repeated strong-Wolfe L-BFGS steps.

    Adam is useful for the initial continuous fit, but after each quantization
    freeze the remaining problem is a smooth, low-noise least-squares problem.
    L-BFGS makes substantially larger compensation moves there.  The frozen
    mask is applied inside the closure so line-search trial points cannot
    change already committed lattice values.
    """
    theta = theta.clone().requires_grad_(True)
    w = w.clone().requires_grad_(True)
    opt = torch.optim.LBFGS(
        [theta, w], lr=1.0, max_iter=inner_steps, history_size=50,
        line_search_fn="strong_wolfe", tolerance_grad=1e-10,
        tolerance_change=1e-14,
    )
    best = (float("inf"), theta.detach().clone(), w.detach().clone())

    def closure():
        nonlocal best
        opt.zero_grad(set_to_none=True)
        err = residual(problem, model_x(theta, frozen, qval, edges, R, B, w))
        objective = err + dpen * (0.5 * ((w.abs().sum(1)) ** 2 - (w.sum(1)) ** 2).sum()) if dpen else err
        objective.backward()
        theta.grad[frozen] = 0.0
        value = float(err.detach())
        if value < best[0]:
            best = (value, theta.detach().clone(), w.detach().clone())
        return objective

    for _ in range(max(1, outer_steps)):
        opt.step(closure)
    return best


def candidates(theta, scale):
    lo = torch.floor(theta * scale) / scale
    return lo, lo + 1.0 / scale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--brot", type=int, default=8)
    parser.add_argument("--stages", type=int, default=64)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=None, help="legacy alias for --lr-th")
    parser.add_argument("--lr-th", type=float, default=3e-4)
    parser.add_argument("--lr-w", type=float, default=3e-5)
    parser.add_argument("--probes", type=int, default=2)
    parser.add_argument("--hessian", action="store_true", help="use exact Hessian Hutchinson curvature for freeze ranking")
    parser.add_argument("--power", type=float, default=3.0, help="frozen fraction 1-(1-s/S)^power")
    parser.add_argument("--fracs", default="", help="explicit cumulative frozen fractions, comma separated")
    parser.add_argument("--split", default="", help="EARLY_END,MID_END,EARLY_STEPS,MID_STEPS,LATE_STEPS")
    parser.add_argument("--decay", type=float, default=1.0, help="per-stage final/initial lr multiplier")
    parser.add_argument("--dpen", type=float, default=0.0, help="D(w) penalty during compensation")
    parser.add_argument("--fp32", action="store_true", help="use float32 for the optimization forward pass")
    parser.add_argument("--final-w", type=int, default=0, help="extra Adam steps optimizing w after the stages")
    parser.add_argument("--lbfgs", type=int, default=0, help="L-BFGS outer steps per compensation stage (0 = Adam)")
    parser.add_argument("--lbfgs-inner", type=int, default=20, help="strong-Wolfe iterations inside each L-BFGS step")
    parser.add_argument("--warmup-lbfgs", type=int, default=0,
                        help="extra L-BFGS outer steps on a resumed frozen state before advancing")
    parser.add_argument("--warmup-inner", type=int, default=30,
                        help="strong-Wolfe iterations for --warmup-lbfgs")
    parser.add_argument("--order", choices=("least", "most"), default="most")
    parser.add_argument("--direction", choices=("nearest", "loss"), default="nearest")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0, help="torch RNG seed for Hutchinson probes")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    problem = dt.Problem.build(refdata.load())
    dev = problem.device
    dtype = torch.float32 if args.fp32 else torch.float64
    with np.load(args.start, allow_pickle=False) as data:
        theta = torch.tensor(np.array(data["theta"]), dtype=dtype, device=dev)
        w = torch.tensor(np.array(data["w"]), dtype=dtype, device=dev)
        edges = np.array(data["tree_edges"]).astype(np.int64)
        extra = {"copy_index": np.array(data["copy_index"])} if "copy_index" in data.files else {}
    R, B, C = w.shape
    scale = float(1 << (args.brot - 1))
    frozen = torch.zeros_like(theta, dtype=torch.bool)
    qval = torch.zeros_like(theta)
    resume = args.out.with_suffix(".resume.npz")
    first = 1
    if resume.is_file():
        with np.load(resume) as data:
            theta = torch.tensor(data["theta"], dtype=dtype, device=dev)
            w = torch.tensor(data["w"], dtype=dtype, device=dev)
            frozen = torch.tensor(data["frozen"], device=dev)
            qval = torch.tensor(data["qval"], dtype=dtype, device=dev)
            first = int(data["stage"]) + 1
        print(f"resuming after stage {first - 1}", flush=True)
        if args.warmup_lbfgs > 0:
            _, theta, w = train_lbfgs(problem, theta, frozen, qval, edges, R, B, w,
                                       args.warmup_lbfgs, args.warmup_inner, args.dpen)
            warm_err = refit(problem, theta, frozen, qval, edges, R, B, w)
            np.savez(resume, theta=theta.cpu().numpy(), w=w.cpu().numpy(),
                     frozen=frozen.cpu().numpy(), qval=qval.cpu().numpy(), stage=first - 1)
            print(f"warmup: frozen {int(frozen.sum())}/{frozen.numel()} error_2 {warm_err:.6f}", flush=True)
    err = refit(problem, theta, frozen, qval, edges, R, B, w)
    print(f"start R{R} B{B} C{C} error_2 {err:.6f} direction {args.direction}", flush=True)
    t0 = time.time()
    log = []
    total = frozen.numel()
    fracs = [float(f) for f in args.fracs.split(",") if f.strip()] if args.fracs else None
    if fracs:
        if any(f <= 0.0 or f > 1.0 for f in fracs) or any(b <= a for a, b in zip(fracs, fracs[1:])):
            raise ValueError("--fracs must be strictly increasing values in (0, 1]")
        args.stages = len(fracs)
    split = [float(x) for x in args.split.split(",")] if args.split else None
    if split and len(split) != 5:
        raise ValueError("--split expects EARLY_END,MID_END,EARLY_STEPS,MID_STEPS,LATE_STEPS")
    lr_th = args.lr if args.lr is not None else args.lr_th
    lr_w = args.lr * 0.1 if args.lr is not None else args.lr_w
    for s in range(first, args.stages + 1):
        if fracs:
            target = int(round(total * fracs[s - 1]))
        else:
            target = total if s == args.stages else int(round(total * (1.0 - (1.0 - s / args.stages) ** args.power)))
        need = target - int(frozen.sum())
        if need > 0:
            curv = (hessian_diag(problem, theta, frozen, qval, edges, R, B, w, args.probes)
                    if args.hessian else curvature(problem, theta, frozen, qval, edges, R, B, w, args.probes))
            lo, hi = candidates(theta, scale)
            if args.direction == "loss":
                g = gradient(problem, theta, frozen, qval, edges, R, B, w)
                d_lo, d_hi = lo - theta, hi - theta
                e_lo = g * d_lo + 0.5 * curv.clamp_min(0) * d_lo**2
                e_hi = g * d_hi + 0.5 * curv.clamp_min(0) * d_hi**2
                pick = torch.where(e_lo <= e_hi, lo, hi)
                # Freeze by predicted damage. Negative values are improvements and
                # should be processed after angles that would be harmed most.
                score = torch.minimum(e_lo, e_hi).clamp_min(0)
            else:
                # ``nearest`` controls the selected lattice point, while the
                # freeze order still follows the predicted rounding impact.
                # This is the stable part of the progressive quantizer: large
                # angles are compensated while most continuous freedom remains.
                pick = torch.round(theta * scale) / scale
                d = pick - theta
                # With zero curvature probes, retain the first-order impact
                # rather than falling back to raw lattice distance.  The latter
                # ignores which angles the residual is actually sensitive to.
                g = gradient(problem, theta, frozen, qval, edges, R, B, w)
                score = (g * d).abs() + 0.5 * curv.clamp_min(0) * d ** 2
            fill = float("inf") if args.order == "least" else -float("inf")
            score = torch.where(frozen, torch.full_like(score, fill), score)
            idx = torch.argsort(score.reshape(-1), descending=(args.order == "most"))[:need]
            fr = frozen.reshape(-1).clone()
            fr[idx] = True
            qv = qval.reshape(-1).clone()
            qv[idx] = pick.reshape(-1)[idx]
            frozen, qval = fr.reshape(frozen.shape), qv.reshape(qval.shape)
        rounded = refit(problem, theta, frozen, qval, edges, R, B, w)
        if s < args.stages:
            if split:
                frac = fracs[s - 1] if fracs else s / args.stages
                steps = int(split[2] if frac <= split[0] else split[3] if frac <= split[1] else split[4])
            else:
                steps = args.steps
            if args.lbfgs > 0:
                _, theta, w = train_lbfgs(problem, theta, frozen, qval, edges, R, B, w,
                                           args.lbfgs, args.lbfgs_inner, args.dpen)
            else:
                _, theta, w = train(problem, theta, frozen, qval, edges, R, B, w, steps,
                                     lr_th, lr_w, args.decay, args.dpen)
        err = refit(problem, theta, frozen, qval, edges, R, B, w)
        log.append({"stage": s, "frozen": int(frozen.sum()), "rounded": rounded, "error_2": err})
        torch.cuda.empty_cache()
        if s % 4 == 0 or s == args.stages:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(resume, theta=theta.cpu().numpy(), w=w.cpu().numpy(), frozen=frozen.cpu().numpy(),
                     qval=qval.cpu().numpy(), stage=s)
        if s % 4 == 0 or s == args.stages:
            print(f"stage {s}: frozen {int(frozen.sum())}/{total} after-freeze {rounded:.6f} retrained {err:.6f} ({time.time() - t0:.0f}s)", flush=True)
    if args.final_w > 0:
        theta = theta.detach()
        w = w.detach().requires_grad_(True)
        opt = torch.optim.Adam([w], lr=lr_w)
        best_w, best_e = w.detach().clone(), float(residual(problem, model_x(theta, frozen, qval, edges, R, B, w)))
        for _ in range(args.final_w):
            opt.zero_grad(set_to_none=True)
            err_w = residual(problem, model_x(theta, frozen, qval, edges, R, B, w))
            err_w.backward()
            opt.step()
            if float(err_w.detach()) < best_e:
                best_e, best_w = float(err_w.detach()), w.detach().clone()
        w = best_w
        print(f"final w: error_2 {best_e:.6f}", flush=True)
    words = givens.to_words(theta_eff(theta, frozen, qval).cpu().numpy(), args.brot)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, words=words, w=w.cpu().numpy(), tree_edges=edges, brot=np.array(args.brot), **extra)
    args.out.with_suffix(".json").write_text(json.dumps({"log": log}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
