"""Prune / retrain / quantize DF-THC-SOS factorizations on the GPU.

Stage 1 (continuous): start from a seed factorization, prune basis vectors
(B) and copies (C) towards the target shape in several steps, retraining the
continuous Givens angles, ``w`` and ``W`` after each step.
Stage 2 (quantize): freeze angles to ``brot`` bits progressively (smallest
rounding residual first), retraining the free angles and weights each stage.
Stage 3: weights-only polish with all two-body angles frozen.

The objective is ``lambda_eff`` with quadratic penalties keeping
``error_2 <= e2_target`` and ``E_gap >= gap_target``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

import dfthc_torch as dt
import givens
import refdata


@dataclass
class State:
    theta: torch.Tensor  # (R*B, N-1) theta/pi
    w: torch.Tensor  # (R, B, C)
    Woff: torch.Tensor  # (R, C)
    frozen: torch.Tensor  # (R*B, N-1) bool
    edges: np.ndarray
    history: list = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.w.shape)

    def theta_eff(self, brot: int) -> torch.Tensor:
        scale = float(1 << (brot - 1))
        rounded = torch.round(self.theta.detach() * scale) / scale
        return torch.where(self.frozen, rounded, self.theta)

    def U(self, brot: int) -> torch.Tensor:
        R, B, _ = self.shape
        return givens.decode_torch(self.theta_eff(brot), self.edges).reshape(R, B, -1)


def load_seed(path: Path, device: torch.device) -> State:
    with np.load(path, allow_pickle=False) as data:
        params = {k: np.array(data[k]) for k in data.files}
    w = params["W"].astype(np.float64)
    Woff = params["H2_I"].astype(np.float64) - w.sum(axis=1)
    R, B, N = params["U"].shape
    edges = params["tree_edges"].astype(np.int64)
    theta = givens.encode_numpy(params["U"].reshape(R * B, N).astype(np.float64), edges)
    as_t = lambda a: torch.tensor(a, dtype=torch.float64, device=device)  # noqa: E731
    return State(
        theta=as_t(theta),
        w=as_t(w),
        Woff=as_t(Woff),
        frozen=torch.zeros(theta.shape, dtype=torch.bool, device=device),
        edges=edges,
    )


def prune(state: State, B_new: int, C_new: int, brot: int) -> State:
    """Keep the B_new basis vectors and C_new copies with the largest weights."""
    R, B, C = state.shape
    with torch.no_grad():
        X = dt.build_x(state.U(brot), state.w)  # (R, C, N, N)
        col_score = torch.linalg.vector_norm(X.reshape(R, C, -1), dim=-1)  # (R, C)
        keep_c = torch.argsort(col_score, dim=1, descending=True)[:, :C_new].sort(dim=1).values
        w = torch.gather(state.w, 2, keep_c[:, None, :].expand(R, B, C_new))
        Woff = torch.gather(state.Woff, 1, keep_c)
        row_score = torch.linalg.vector_norm(w, dim=2)  # (R, B)
        keep_b = torch.argsort(row_score, dim=1, descending=True)[:, :B_new].sort(dim=1).values
        w = torch.gather(w, 1, keep_b[:, :, None].expand(R, B_new, C_new))
        flat = (keep_b + torch.arange(R, device=keep_b.device)[:, None] * B).reshape(-1)
        theta = state.theta[flat]
        frozen = state.frozen[flat]
    return State(theta.clone(), w.clone(), Woff.clone(), frozen.clone(), state.edges, state.history)


@dataclass(frozen=True)
class Targets:
    e2: float
    gap: float
    weight: float = 30.0


def objective(problem: dt.Problem, state: State, brot: int, targets: Targets):
    m = dt.metrics(problem, state.U(brot), state.w, state.Woff)
    e2_violation = torch.relu(m["error_2"] - targets.e2) / 1e-3
    gap_violation = torch.relu(targets.gap - m["E_gap"]) / 1e-3
    loss = m["lambda_eff"] + targets.weight * (e2_violation**2 + gap_violation**2)
    return loss, m


def optimize(problem, state: State, brot: int, targets: Targets, steps: int, lr: float, train_theta: bool, log_every: int, tag: str) -> State:
    groups = [
        {"params": [state.w], "lr": lr * 0.01},
        {"params": [state.Woff], "lr": lr},
    ]
    if train_theta:
        groups.append({"params": [state.theta], "lr": lr})
    for g in groups:
        for p in g["params"]:
            p.requires_grad_(True)
    if not train_theta:
        state.theta.requires_grad_(False)
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.05)
    best = None
    t0 = time.time()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        loss, m = objective(problem, state, brot, targets)
        loss.backward()
        if state.theta.grad is not None:
            state.theta.grad[state.frozen] = 0.0
        opt.step()
        sched.step()
        feasible = float(m["error_2"]) <= targets.e2 and float(m["E_gap"]) >= targets.gap
        lam = float(m["lambda_eff"])
        if feasible and (best is None or lam < best[0]):
            best = (lam, state.theta.detach().clone(), state.w.detach().clone(), state.Woff.detach().clone())
        if step % log_every == 0 or step == steps - 1:
            rep = {k: round(v, 6) for k, v in dt.scalar_report(m).items() if k in ("error_2", "Lambda", "E_gap", "lambda_eff")}
            print(f"[{tag}] step {step} loss {float(loss):.5f} {rep} feasible={feasible} {time.time() - t0:.1f}s", flush=True)
    for p in (state.theta, state.w, state.Woff):
        p.requires_grad_(False)
    if best is not None:
        state.theta.copy_(best[1])
        state.w.copy_(best[2])
        state.Woff.copy_(best[3])
    state.history.append({"tag": tag, "best_lambda_eff": None if best is None else best[0], "shape": list(state.shape)})
    return state


def freeze_more(state: State, fraction: float, brot: int) -> None:
    """Freeze ``fraction`` of all angles: unfrozen ones with the smallest rounding residual first."""
    scale = float(1 << (brot - 1))
    resid = torch.abs(state.theta * scale - torch.round(state.theta * scale))
    resid = torch.where(state.frozen, torch.full_like(resid, -1.0), resid)
    total = state.frozen.numel()
    target = min(total, int(round(fraction * total)))
    need = target - int(state.frozen.sum())
    if need <= 0:
        return
    free_idx = torch.nonzero(~state.frozen.reshape(-1)).squeeze(1)
    order = torch.argsort(resid.reshape(-1)[free_idx])[:need]
    flat = state.frozen.reshape(-1)
    flat[free_idx[order]] = True


def save(state: State, brot: int, path: Path, problem: dt.Problem) -> dict:
    R, B, C = state.shape
    with torch.no_grad():
        U = state.U(brot)
        m = dt.metrics(problem, U, state.w, state.Woff)
        h1p = m["h1_prime"].cpu().numpy()
    two_body_theta = state.theta_eff(brot).cpu().numpy()
    _, vecs = np.linalg.eigh(h1p)
    one_body_theta = givens.encode_numpy(vecs.T, state.edges)
    words = np.concatenate(
        [givens.to_words(one_body_theta, brot), givens.to_words(two_body_theta, brot)], axis=0
    )
    w = state.w.cpu().numpy()
    Woff = state.Woff.cpu().numpy()
    shifts = np.load(refdata.VERIFIERS_DIR / "dfthcsos" / "FeMoCo76o113e.fixed_shifts.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        U=U.cpu().numpy(),
        W=w,
        H2_I=Woff + w.sum(axis=1),
        H1_I=np.array(shifts["H1_I"]),
        Hsym_lower=np.array(shifts["Hsym_lower"]),
        tree_edges=state.edges,
        angle_words=words,
        brot=np.array(brot),
        frozen_fraction=np.array(float(state.frozen.float().mean())),
    )
    return dt.scalar_report(m)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--B", type=int, required=True)
    parser.add_argument("--C", type=int, required=True)
    parser.add_argument("--brot", type=int, default=8)
    parser.add_argument("--prune-steps", type=int, default=6)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--quant-stages", type=int, default=16)
    parser.add_argument("--quant-steps", type=int, default=400)
    parser.add_argument("--polish-steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--e2-continuous", type=float, default=0.0185)
    parser.add_argument("--e2-final", type=float, default=0.01995)
    parser.add_argument("--gap-margin", type=float, default=3e-5)
    parser.add_argument("--log-every", type=int, default=250)
    args = parser.parse_args()

    torch.manual_seed(0)
    ref = refdata.load()
    problem = dt.Problem.build(ref)
    state = load_seed(args.seed, problem.device)
    gap_t = refdata.E_GAP_FLOOR + args.gap_margin
    R, B0, C0 = state.shape
    print("seed shape", state.shape, flush=True)

    for i in range(1, args.prune_steps + 1):
        frac = i / args.prune_steps
        B_i = round(B0 + (args.B - B0) * frac)
        C_i = round(C0 + (args.C - C0) * frac)
        state = prune(state, B_i, C_i, args.brot)
        state = optimize(problem, state, args.brot, Targets(args.e2_continuous, gap_t), args.steps, args.lr, True, args.log_every, f"prune{i}:{B_i}x{C_i}")

    for s in range(1, args.quant_stages + 1):
        freeze_more(state, s / args.quant_stages, args.brot)
        e2_t = args.e2_continuous + (args.e2_final - args.e2_continuous) * s / args.quant_stages
        state = optimize(problem, state, args.brot, Targets(e2_t, gap_t), args.quant_steps, args.lr * 0.5, True, args.log_every * 4, f"quant{s}")

    state = optimize(problem, state, args.brot, Targets(args.e2_final, gap_t), args.polish_steps, args.lr * 0.5, False, args.log_every, "polish")
    report = save(state, args.brot, args.out, problem)
    report["history"] = state.history
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
