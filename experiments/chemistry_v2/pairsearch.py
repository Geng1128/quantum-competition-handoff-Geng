"""Joint two-angle discrete search for the chemistry tail.

The ordinary dsearch changes one Givens word at a time.  This pass also tests
two adjacent tree edges on the same vector, which can be useful when two
single-angle moves are individually uphill but jointly downhill.
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
from dsearch import Search
from gtrain import projector_design, refit_block, xmat


class PairSearch(Search):
    def __init__(self, *args, anchor_k: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        self.anchor_k = anchor_k
        self.angle_neighbors = self._angle_neighbors(self.edges)

    @staticmethod
    def _angle_neighbors(edges: np.ndarray) -> list[list[int]]:
        out = [[] for _ in range(len(edges))]
        for i in range(len(edges)):
            for j in range(i):
                if len(set(edges[i]) & set(edges[j])):
                    out[i].append(j)
                    out[j].append(i)
        return out

    def _delta_for_words(self, r: int, b: int, candidates: torch.Tensor) -> torch.Tensor:
        """Return exact one-vector residual changes for candidate words."""
        n2 = self.n * self.n
        idx = r * self.B + b
        u_new = self.decode(candidates).to(torch.float32)
        u_old = self.U[r, b].to(torch.float32)
        V = (u_new[:, :, None] * u_new[:, None, :]).reshape(-1, n2)
        V -= (u_old[None, :, None] * u_old[None, None, :]).reshape(1, n2)
        wv = self.w[r, b].to(torch.float32)
        s = float(torch.dot(wv, wv))
        a = wv @ self.X[r]
        rho = self.Rm @ a
        RV = V @ self.Rm
        vrho = V @ rho
        vRv = (RV * V).sum(1)
        vv = (V * V).sum(1)
        av = V @ a
        aa = float(a @ a)
        return 4 * vrho + 2 * s * vRv + 2 * (aa * vv + av**2) + 4 * s * av * vv + s * s * vv * vv

    def best_move(self, r: int, b: int) -> tuple[float, torch.Tensor] | None:
        idx = r * self.B + b
        n_angles = self.words.shape[1]
        base = self.words[idx]
        changes: list[tuple[int, int, int]] = []
        for angle in range(n_angles):
            changes.extend([(angle, 1, 1), (angle, -1, 1)])
        single_words = base.repeat(len(changes), 1)
        for row, (angle, step, _) in enumerate(changes):
            single_words[row, angle] = (base[angle] + step) % (1 << self.brot)
        single_dE = self._delta_for_words(r, b, single_words)
        k = min(self.anchor_k, len(changes))
        anchors = torch.argsort(single_dE)[:k].tolist()

        pairs: set[tuple[int, int, int, int]] = set()
        for anchor_row in anchors:
            a, sa, _ = changes[anchor_row]
            for c in self.angle_neighbors[a]:
                for sb in (1, -1):
                    x, y = sorted((a, c))
                    sx, sy = (sa, sb) if a == x else (sb, sa)
                    pairs.add((x, y, sx, sy))
        pair_words = base.repeat(len(pairs), 1) if pairs else base.new_empty((0, n_angles))
        for row, (a, b2, sa, sb) in enumerate(sorted(pairs)):
            pair_words[row, a] = (base[a] + sa) % (1 << self.brot)
            pair_words[row, b2] = (base[b2] + sb) % (1 << self.brot)
        pair_dE = self._delta_for_words(r, b, pair_words) if pairs else base.new_empty((0,), dtype=torch.float32)

        all_dE = torch.cat((single_dE, pair_dE))
        if not len(all_dE):
            return None
        best = int(torch.argmin(all_dE))
        if float(all_dE[best]) >= -1e-10:
            return None
        if best < len(changes):
            return float(all_dE[best]), single_words[best]
        return float(all_dE[best]), pair_words[best - len(changes)]

    def visit_pair(self, r: int, b: int) -> bool:
        move = self.best_move(r, b)
        if move is None:
            return False
        _, new_word = move
        idx = r * self.B + b
        u_new = self.decode(new_word[None, :])[0].to(torch.float32)
        u_old = self.U[r, b].to(torch.float32)
        v = (u_new[:, None] * u_new[None, :]).reshape(-1) - (u_old[:, None] * u_old[None, :]).reshape(-1)
        wv = self.w[r, b].to(torch.float32)
        s = float(torch.dot(wv, wv))
        a = wv @ self.X[r]
        left = torch.stack([a, v, s * v], dim=1)
        right = torch.stack([v, a, v], dim=0)
        self.Rm.addmm_(left, right)
        self.X[r] += torch.outer(wv, v)
        self.words[idx] = new_word
        self.U[r, b] = u_new.to(torch.float64)
        return True

    def pair_sweep(self, order: np.ndarray) -> int:
        moves = 0
        with torch.no_grad():
            for idx in order:
                r, b = divmod(int(idx), self.B)
                moves += int(self.visit_pair(r, b))
        return moves


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--brot", type=int, default=8)
    parser.add_argument("--sweeps", type=int, default=2)
    parser.add_argument("--anchor-k", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    problem = dt.Problem.build(refdata.load())
    with np.load(args.start, allow_pickle=False) as data:
        edges = np.array(data["tree_edges"]).astype(np.int64)
        w = torch.tensor(np.array(data["w"]), dtype=torch.float64, device=problem.device)
        words = np.array(data["words"]).astype(np.int64)
        extra = {"copy_index": np.array(data["copy_index"])} if "copy_index" in data.files else {}
    search = PairSearch(problem, words, w, edges, args.brot, anchor_k=args.anchor_k)
    print(f"pair start error_2 {search.error():.9f}", flush=True)
    with torch.no_grad():
        search.refit(1)
    print(f"after refit error_2 {search.error():.9f}", flush=True)
    rng = np.random.default_rng(args.seed)
    best_err = search.error()
    log = []
    t0 = time.time()
    for sweep in range(1, args.sweeps + 1):
        order = rng.permutation(search.R * search.B)
        moves = search.pair_sweep(order)
        search.refit(1)
        err = search.error()
        print(f"pair sweep {sweep}: moves {moves} error_2 {err:.9f} ({time.time() - t0:.0f}s)", flush=True)
        log.append({"sweep": sweep, "moves": moves, "error_2": err})
        if err < best_err:
            best_err = err
            args.out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.out, words=search.words.cpu().numpy(), w=search.w.cpu().numpy(), tree_edges=edges, brot=np.array(args.brot), **extra)
        if moves == 0:
            break
    args.out.with_suffix(".json").write_text(json.dumps({"best_error_2": best_err, "log": log}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
