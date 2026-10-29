"""General (R, B, C) two-body fitting: DF initialization, exact block refits, Adam on angles.

Model: ``sum_(rho c) X_rc (x) X_rc``, ``X_rc = sum_b w[r,b,c] u_rb u_rb^T``.

* Initialization (``--init eig``): eigen-matrices ``Y_l`` of the shifted target
  (``T = sum_l lam_l vec(Y_l) vec(Y_l)^T``) are dealt to the R groups, C each;
  each group's basis is pooled from the top eigenvectors of its ``Y_l``.
* Exact block refit: with the other groups fixed, group rho's ``Z = w w^T``
  is the best rank-C PSD fit of the residual in the span of its projectors.
* Adam on the Givens angles (and w) between refit sweeps.
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

import sys

if str(refdata.VERIFIERS_DIR) not in sys.path:
    sys.path.insert(0, str(refdata.VERIFIERS_DIR))


def projector_design(U: torch.Tensor) -> torch.Tensor:
    """(R, B, N) unit rows -> (R, N^2, B) columns vec(u u^T)."""
    R, B, N = U.shape
    return torch.einsum("rbp,rbq->rpqb", U, U).reshape(R, N * N, B)


def xmat(A: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """All vec(X_rc) as rows: (R*C, N^2)."""
    return torch.einsum("rkb,rbc->rck", A, w).reshape(-1, A.shape[1])


def residual(problem: dt.Problem, X: torch.Tensor) -> torch.Tensor:
    X32 = X.to(torch.float32)
    return torch.linalg.vector_norm(X32.T @ X32 - problem.target)


def refit_block(problem: dt.Problem, A: torch.Tensor, w: torch.Tensor, rho: int, C: int) -> torch.Tensor:
    """Exact best rank-C PSD Z for group ``rho`` given the others; returns new w[rho] (B, C)."""
    R = A.shape[0]
    Ar = A[rho].to(torch.float32)  # (N^2, B)
    X_all = xmat(A, w).to(torch.float32)  # (R*C, N^2)
    X_own = X_all[rho * C:(rho + 1) * C]
    # P = Ar^T (T - G_others) Ar with G_others = X_all^T X_all - X_own^T X_own
    TA = problem.target @ Ar
    P = Ar.T @ TA
    XA = X_all @ Ar
    XoA = X_own @ Ar
    P = (P - XA.T @ XA + XoA.T @ XoA).to(torch.float64)
    u_gram = (Ar.T @ Ar).to(torch.float64)
    # Ridge keeps near-collinear projectors (Gram computed in float32) from blowing up w.
    L = torch.linalg.cholesky(u_gram + 1e-6 * torch.eye(u_gram.shape[0], device=u_gram.device, dtype=u_gram.dtype))
    Y = torch.linalg.solve_triangular(L, P, upper=False)
    Y = torch.linalg.solve_triangular(L, Y.T, upper=False)
    Y = 0.5 * (Y + Y.T)
    vals, vecs = torch.linalg.eigh(Y)
    vals, vecs = vals.flip(0)[:C], vecs.flip(1)[:, :C]
    half = vecs * torch.sqrt(torch.clamp(vals, min=0.0))
    return torch.linalg.solve_triangular(L.T, half, upper=True)


def refit_sweeps(problem, theta, w, edges, R, B, C, sweeps):
    with torch.no_grad():
        U = givens.decode_torch(theta, edges).reshape(R, B, -1)
        A = projector_design(U)
        for _ in range(sweeps):
            for rho in range(R):
                w[rho] = refit_block(problem, A, w, rho, C)
        return float(residual(problem, xmat(A, w)))


EIG_CACHE = refdata.DEFAULT_CACHE.parent / "target_top_eig.npz"


def top_eigenpairs(problem: dt.Problem, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k eigenpairs of the shifted target (cached on disk, computed once in float64)."""
    if EIG_CACHE.is_file():
        with np.load(EIG_CACHE) as data:
            vals, vecs = data["vals"], data["vecs"]
    else:
        T = problem.target.to(torch.float64).cpu().numpy()
        all_vals, all_vecs = np.linalg.eigh(T)
        vals, vecs = all_vals[::-1][:1024].copy(), all_vecs[:, ::-1][:, :1024].copy()
        np.savez(EIG_CACHE, vals=vals, vecs=vecs)
    dev = problem.device
    return torch.tensor(vals[:k], dtype=torch.float64, device=dev), torch.tensor(vecs[:, :k], dtype=torch.float64, device=dev)


def eig_init(problem: dt.Problem, R: int, B: int, C: int, edges: np.ndarray, seed: int):
    """DF-style initialization from eigen-matrices of the target."""
    n = problem.n
    T = problem.target
    vals, vecs = top_eigenpairs(problem, 1024)
    L = R * C
    Y = (vecs[:, :L].T.reshape(L, n, n))
    Y = 0.5 * (Y + Y.transpose(1, 2))
    lam = torch.clamp(vals[:L], min=0.0)
    # Deal eigen-matrices round-robin so every group gets a mix of large and small ones.
    groups = [list(range(r, L, R)) for r in range(R)]
    U = torch.zeros(R, B, n, dtype=torch.float64, device=T.device)
    w = torch.zeros(R, B, C, dtype=torch.float64, device=T.device)
    for r, members in enumerate(groups):
        cands, scores = [], []
        for l in members:
            mu, ev = torch.linalg.eigh(Y[l])
            for k in range(n):
                cands.append(ev[:, k])
                scores.append(float(torch.sqrt(lam[l]) * torch.abs(mu[k])))
        order = np.argsort(-np.asarray(scores))
        chosen = []
        for idx in order:
            v = cands[idx]
            if all(abs(float(v @ c)) < 0.999 for c in chosen[-64:]):
                chosen.append(v)
            if len(chosen) == B:
                break
        U[r, : len(chosen)] = torch.stack(chosen)
        if len(chosen) < B:
            rng = np.random.default_rng(seed + r)
            U[r, len(chosen):] = torch.tensor(rng.normal(size=(B - len(chosen), n)), device=T.device)
    theta = givens.encode_numpy(U.reshape(R * B, n).cpu().numpy(), edges)
    return torch.tensor(theta, dtype=torch.float64, device=T.device), w


def npz_init(problem: dt.Problem, src: Path, B_new: int, C_new: int, edges: np.ndarray):
    """Start from a verified chemistry dir (submission.npz + _recovered_words.npz), pruned per group."""
    from dfthcsos.verify import native_inputs

    with np.load(src / "submission.npz", allow_pickle=False) as data:
        params = {k: np.array(data[k]) for k in data.files}
    w_np, _, _, _ = native_inputs(params)
    R, B, C = w_np.shape
    with np.load(src / "_recovered_words.npz") as data:
        words, brot = data["words"], int(data["brot"])
    vec = givens.decode_numpy(givens.words_to_theta(words, brot), params["tree_edges"].astype(np.int64))[76:].reshape(R, B, 76)
    keep_b = np.sort(np.argsort(-np.linalg.norm(w_np, axis=2), axis=1)[:, :B_new], axis=1)
    U = np.stack([vec[r, keep_b[r]] for r in range(R)])
    w_k = np.stack([w_np[r, keep_b[r]] for r in range(R)])
    if C_new < C:  # keep the copies with the largest weight norm
        keep_c = np.sort(np.argsort(-np.linalg.norm(w_k, axis=1), axis=1)[:, :C_new], axis=1)
        w_k = np.stack([w_k[r][:, keep_c[r]] for r in range(R)])
    theta = givens.encode_numpy(U.reshape(R * B_new, 76), edges)
    dev = problem.device
    return torch.tensor(theta, dtype=torch.float64, device=dev), torch.tensor(w_k, dtype=torch.float64, device=dev)


def adam_cartesian(problem, theta, w, edges, R, B, C, steps, lr):
    """Adam on unnormalized Cartesian rows (better conditioned than tree angles)."""
    U = givens.decode_torch(theta, edges).reshape(R, B, -1).clone().requires_grad_(True)
    w = w.clone().requires_grad_(True)
    opt = torch.optim.Adam([{"params": [U], "lr": lr}, {"params": [w], "lr": lr * 0.1}])
    best = (float("inf"), U.detach().clone(), w.detach().clone())
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        err = residual(problem, xmat(projector_design(dt.normalize_rows(U)), w))
        e = float(err.detach())
        if e < best[0]:
            best = (e, U.detach().clone(), w.detach().clone())
        err.backward()
        opt.step()
    e, U_b, w_b = best
    theta_b = givens.encode_numpy(dt.normalize_rows(U_b).reshape(R * B, -1).cpu().numpy(), edges)
    return e, torch.tensor(theta_b, dtype=torch.float64, device=theta.device), w_b


def adam_phase(problem, theta, w, edges, R, B, C, steps, lr, frozen=None):
    theta = theta.clone().requires_grad_(True)
    w = w.clone().requires_grad_(True)
    opt = torch.optim.Adam([{"params": [theta], "lr": lr}, {"params": [w], "lr": lr * 0.1}])
    best = (float("inf"), theta.detach().clone(), w.detach().clone())
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        U = givens.decode_torch(theta, edges).reshape(R, B, -1)
        err = residual(problem, xmat(projector_design(U), w))
        e = float(err.detach())
        if e < best[0]:
            best = (e, theta.detach().clone(), w.detach().clone())
        err.backward()
        if frozen is not None:
            theta.grad[frozen] = 0.0
        opt.step()
    return best


def lbfgs_phase(problem, theta, w, edges, R, B, C, iters, w_scale=0.05, frozen=None):
    """L-BFGS on error_2^2 over (theta, w / w_scale) jointly."""
    th = theta.clone().requires_grad_(True)
    wh = (w / w_scale).clone().requires_grad_(True)
    opt = torch.optim.LBFGS([th, wh], lr=1.0, max_iter=iters, history_size=100,
                            line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-15)
    state = {"best": (float("inf"), theta.clone(), w.clone())}

    def closure():
        opt.zero_grad(set_to_none=True)
        U = givens.decode_torch(th, edges).reshape(R, B, -1)
        err = residual(problem, xmat(projector_design(U), wh * w_scale))
        loss = err**2
        loss.backward()
        if frozen is not None:
            th.grad[frozen] = 0.0
        e = float(err.detach())
        if e < state["best"][0]:
            state["best"] = (e, th.detach().clone(), (wh.detach() * w_scale).clone())
        return loss

    opt.step(closure)
    return state["best"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--R", type=int, required=True)
    parser.add_argument("--B", type=int, required=True)
    parser.add_argument("--C", type=int, required=True)
    parser.add_argument("--edges-from", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--sweeps", type=int, default=2)
    parser.add_argument("--adam-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--opt", choices=("adam", "cart", "lbfgs"), default="cart")
    parser.add_argument("--lr-final", type=float, default=5e-4)
    parser.add_argument("--init-dir", type=Path, default=None, help="verified chemistry dir to prune from")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    problem = dt.Problem.build(refdata.load())
    with np.load(args.edges_from, allow_pickle=False) as data:
        edges = np.array(data["tree_edges"]).astype(np.int64)
    R, B, C = args.R, args.B, args.C
    t0 = time.time()
    if args.init_dir is not None:
        theta, w = npz_init(problem, args.init_dir, B, C, edges)
    else:
        theta, w = eig_init(problem, R, B, C, edges, args.seed)
    err = refit_sweeps(problem, theta, w, edges, R, B, C, args.sweeps)
    print(f"init refit error_2 {err:.6f} ({time.time() - t0:.0f}s)", flush=True)
    log = [{"round": 0, "error_2": err}]
    for rnd in range(1, args.rounds + 1):
        if args.opt == "adam":
            lr = args.lr * (args.lr_final / args.lr) ** ((rnd - 1) / max(1, args.rounds - 1))
            e_adam, theta, w = adam_phase(problem, theta, w, edges, R, B, C, args.adam_steps, lr)
        elif args.opt == "cart":
            lr = args.lr * (args.lr_final / args.lr) ** ((rnd - 1) / max(1, args.rounds - 1))
            e_adam, theta, w = adam_cartesian(problem, theta, w, edges, R, B, C, args.adam_steps, lr)
        else:
            e_adam, theta, w = lbfgs_phase(problem, theta, w, edges, R, B, C, args.adam_steps)
        err = refit_sweeps(problem, theta, w, edges, R, B, C, args.sweeps)
        print(f"round {rnd}: adam {e_adam:.6f} refit {err:.6f} ({time.time() - t0:.0f}s)", flush=True)
        log.append({"round": rnd, "adam": e_adam, "error_2": err})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, theta=theta.cpu().numpy(), w=w.cpu().numpy(), tree_edges=edges, shape=np.array([R, B, C]))
    args.out.with_suffix(".json").write_text(json.dumps({"shape": [R, B, C], "log": log}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
