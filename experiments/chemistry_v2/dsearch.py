"""Exact discrete local search over brot-bit angle words, with exact block refits of w.

For one vector u -> u' (one word changed by +-step), with v = vec(u'u'^T - uu^T),
a = sum_c w_c x_c over the vector's group, s = sum_c w_c^2 and the residual
matrix Rm = sum x x^T - T, the change of ||Rm||^2 is exactly

    dE = 4 v.Rm a + 2 s v.Rm v + 2(|a|^2 |v|^2 + (a.v)^2) + 4 s (a.v)|v|^2 + s^2 |v|^4

All candidate moves of a vector are scored with one GEMM; an accepted move is
a rank-3 update of Rm. Every sweep ends with exact Gauss-Seidel refits of w.
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


class Search:
    def __init__(self, problem: dt.Problem, words: np.ndarray, w: torch.Tensor, edges: np.ndarray, brot: int):
        self.p = problem
        self.edges = edges
        self.brot = brot
        self.R, self.B, self.C = w.shape
        self.n = problem.n
        self.words = torch.tensor(words, dtype=torch.int64, device=problem.device)
        self.w = w.clone()
        self.rebuild()

    def decode(self, words: torch.Tensor) -> torch.Tensor:
        theta = words.to(torch.float64) / float(1 << (self.brot - 1))
        return givens.decode_torch(theta, self.edges)

    def rebuild(self) -> None:
        self.U = self.decode(self.words).reshape(self.R, self.B, self.n)
        A = projector_design(self.U)
        self.X = xmat(A, self.w).reshape(self.R, self.C, -1).to(torch.float32)  # (R, C, N^2)
        flat = self.X.reshape(-1, self.n * self.n)
        self.Rm = flat.T @ flat - self.p.target

    def error(self) -> float:
        return float(torch.linalg.vector_norm(self.Rm))

    def refit(self, sweeps: int = 1) -> None:
        with torch.no_grad():
            A = projector_design(self.U)
            for _ in range(sweeps):
                for r in range(self.R):
                    self.w[r] = refit_block(self.p, A, self.w, r, self.C)
        self.rebuild()

    def visit(self, r: int, b: int, steps: tuple[int, ...], max_moves: int) -> int:
        """Greedy word moves on vector (r, b); returns the number of accepted moves."""
        n2 = self.n * self.n
        idx = r * self.B + b
        moves = 0
        wv = self.w[r, b].to(torch.float32)  # (C,)
        s = float(torch.dot(wv, wv))
        num_angles = self.words.shape[1]
        for _ in range(max_moves):
            a = wv @ self.X[r]  # (N^2,)
            rho = self.Rm @ a
            base = self.words[idx]
            cand = base.repeat(len(steps) * num_angles, 1)
            ar = torch.arange(num_angles, device=cand.device)
            for k, st in enumerate(steps):
                rows = ar + k * num_angles
                cand[rows, ar] = (cand[rows, ar] + st) % (1 << self.brot)
            u_new = self.decode(cand).to(torch.float32)
            u_old = self.U[r, b].to(torch.float32)
            V = (u_new[:, :, None] * u_new[:, None, :]).reshape(-1, n2) - (u_old[:, None] * u_old[None, :]).reshape(1, n2)
            RV = V @ self.Rm
            vrho = V @ rho
            vRv = (RV * V).sum(1)
            vv = (V * V).sum(1)
            av = V @ a
            aa = float(a @ a)
            dE = 4 * vrho + 2 * s * vRv + 2 * (aa * vv + av**2) + 4 * s * av * vv + s * s * vv * vv
            best = int(torch.argmin(dE))
            if float(dE[best]) >= -1e-10:
                break
            v = V[best]
            left = torch.stack([a, v, s * v], dim=1)
            right = torch.stack([v, a, v], dim=0)
            self.Rm.addmm_(left, right)
            self.X[r] += torch.outer(wv, v)
            self.words[idx] = cand[best]
            self.U[r, b] = u_new[best].to(torch.float64)
            moves += 1
        return moves

    def removal_costs(self, r: int) -> torch.Tensor:
        """Exact dE of deleting each vector of group r (its weights drop out, w otherwise fixed)."""
        n2 = self.n * self.n
        u = self.U[r].to(torch.float32)  # (B, N)
        V = -(u[:, :, None] * u[:, None, :]).reshape(self.B, n2)
        Wr = self.w[r].to(torch.float32)  # (B, C)
        Amat = Wr @ self.X[r]  # (B, N^2): a_b
        s = (Wr * Wr).sum(1)
        RV = V @ self.Rm
        vrho = (RV * Amat).sum(1)  # v.Rm a  (Rm symmetric)
        vRv = (RV * V).sum(1)
        vv = (V * V).sum(1)
        av = (Amat * V).sum(1)
        aa = (Amat * Amat).sum(1)
        return 4 * vrho + 2 * s * vRv + 2 * (aa * vv + av**2) + 4 * s * av * vv + s * s * vv * vv

    def remove_vectors(self, per_group: int) -> None:
        """Delete the ``per_group`` cheapest vectors of every group (keeps B equal across groups)."""
        keep_rows = []
        for r in range(self.R):
            cost = self.removal_costs(r)
            drop = set(torch.argsort(cost)[:per_group].tolist())
            keep_rows.append([b for b in range(self.B) if b not in drop])
        keep = torch.tensor(keep_rows, device=self.w.device)
        self.w = torch.stack([self.w[r, keep[r]] for r in range(self.R)])
        words = self.words.reshape(self.R, self.B, -1)
        self.words = torch.stack([words[r, keep[r]] for r in range(self.R)]).reshape(-1, words.shape[-1])
        self.B -= per_group
        self.rebuild()

    def remove_copies(self, per_group: int) -> None:
        """Delete the ``per_group`` copies per group whose removal raises ||Rm||^2 least."""
        keep_cols = []
        for r in range(self.R):
            Xr = self.X[r]  # (C, N^2)
            RX = Xr @ self.Rm
            cost = -2 * (RX * Xr).sum(1) + (Xr * Xr).sum(1) ** 2
            drop = set(torch.argsort(cost)[:per_group].tolist())
            keep_cols.append([c for c in range(self.C) if c not in drop])
        keep = torch.tensor(keep_cols, device=self.w.device)
        self.w = torch.stack([self.w[r][:, keep[r]] for r in range(self.R)])
        if not hasattr(self, "copy_index"):
            self.copy_index = np.tile(np.arange(self.C), (self.R, 1))
        self.copy_index = np.stack([self.copy_index[r][keep_cols[r]] for r in range(self.R)])
        self.C -= per_group
        self.rebuild()

    def sweep(self, steps: tuple[int, ...], max_moves: int, order: np.ndarray) -> int:
        total = 0
        for idx in order:
            r, b = divmod(int(idx), self.B)
            total += self.visit(r, b, steps, max_moves)
        return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=Path, required=True, help="npz with theta (continuous) or words, w, tree_edges")
    parser.add_argument("--brot", type=int, default=8)
    parser.add_argument("--sweeps", type=int, default=10)
    parser.add_argument("--max-moves", type=int, default=4)
    parser.add_argument("--steps", type=int, nargs="+", default=[1, -1])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    problem = dt.Problem.build(refdata.load())
    with np.load(args.start, allow_pickle=False) as data:
        edges = np.array(data["tree_edges"]).astype(np.int64)
        w = torch.tensor(np.array(data["w"]), dtype=torch.float64, device=problem.device)
        if "words" in data.files:
            words = np.array(data["words"]).astype(np.int64)
        else:
            words = givens.to_words(np.array(data["theta"]), args.brot)
        extra = {"copy_index": np.array(data["copy_index"])} if "copy_index" in data.files else {}
    s = Search(problem, words, w, edges, args.brot)
    print(f"rounded start error_2 {s.error():.6f}", flush=True)
    s.refit(2)
    print(f"after refit error_2 {s.error():.6f}", flush=True)
    rng = np.random.default_rng(args.seed)
    log = []
    t0 = time.time()
    best_err = s.error()
    for sw in range(1, args.sweeps + 1):
        order = rng.permutation(s.R * s.B)
        moves = s.sweep(tuple(args.steps), args.max_moves, order)
        drift = s.error()
        s.refit(1)
        err = s.error()
        print(f"sweep {sw}: moves {moves} incremental {drift:.6f} refit {err:.6f} ({time.time() - t0:.0f}s)", flush=True)
        log.append({"sweep": sw, "moves": moves, "error_2": err})
        if err < best_err:
            best_err = err
            args.out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.out, words=s.words.cpu().numpy(), w=s.w.cpu().numpy(), tree_edges=edges, brot=np.array(args.brot), **extra)
        if moves == 0:
            break
    args.out.with_suffix(".json").write_text(json.dumps({"best_error_2": best_err, "log": log}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
