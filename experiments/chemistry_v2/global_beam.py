"""Experimental global low-rank beam search over several discrete angle moves.

For a vector move, the DF-THC residual update is a rank-3 matrix update.  For
several vectors we add the same rank-3 terms plus the exact within-group cross
terms.  This evaluates multi-angle combinations without rebuilding the full
model for every candidate.  The current implementation is a research probe;
the full scan has not produced a verified candidate and can exceed memory.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch

import dfthc_torch as dt
import refdata
from dsearch import Search


class GlobalBeam(Search):
    def candidate_moves(self, r: int, b: int, steps: tuple[int, ...], top_per: int):
        idx = r * self.B + b
        n_angles = self.words.shape[1]
        base = self.words[idx]
        changes = [(angle, step) for angle in range(n_angles) for step in steps]
        candidates = base.repeat(len(changes), 1)
        for row, (angle, step) in enumerate(changes):
            candidates[row, angle] = (base[angle] + step) % (1 << self.brot)

        n2 = self.n * self.n
        u_new = self.decode(candidates).to(torch.float32)
        u_old = self.U[r, b].to(torch.float32)
        V = (u_new[:, :, None] * u_new[:, None, :]).reshape(-1, n2)
        V -= (u_old[None, :, None] * u_old[None, None, :]).reshape(1, n2)
        wv = self.w[r, b].to(torch.float32)
        s = float(torch.dot(wv, wv))
        a = wv @ self.X[r]
        rho = self.Rm @ a
        RV = V @ self.Rm
        dE = 4 * (V @ rho) + 2 * s * ((RV * V).sum(1))
        vv = (V * V).sum(1)
        av = V @ a
        aa = float(a @ a)
        dE += 2 * (aa * vv + av**2) + 4 * s * av * vv + s * s * vv * vv
        order = torch.argsort(dE)[:top_per]
        out = []
        for j in order.tolist():
            v = V[j]
            aj = a
            D = torch.outer(aj, v) + torch.outer(v, aj) + s * torch.outer(v, v)
            out.append({
                "r": r, "b": b, "idx": idx, "dE": float(dE[j]),
                "word": candidates[j].clone(), "v": v, "a": aj,
                "s": s, "D": D,
            })
        return out

    def combo_delta(self, moves):
        D = torch.zeros_like(self.Rm)
        for move in moves:
            D += move["D"]
        for left, right in itertools.combinations(moves, 2):
            if left["r"] != right["r"]:
                continue
            t = float(torch.dot(self.w[left["r"], left["b"]].to(torch.float32),
                                self.w[right["r"], right["b"]].to(torch.float32)))
            D += t * (torch.outer(left["v"], right["v"]) + torch.outer(right["v"], left["v"]))
        return D

    def combo_cost(self, moves) -> float:
        D = self.combo_delta(moves)
        return float(2 * (self.Rm * D).sum() + (D * D).sum())

    def apply_combo(self, moves) -> None:
        D = self.combo_delta(moves)
        self.Rm += D
        by_group: dict[int, list[dict]] = {}
        for move in moves:
            by_group.setdefault(move["r"], []).append(move)
        for r, group_moves in by_group.items():
            for move in group_moves:
                self.X[r] += torch.outer(self.w[r, move["b"]].to(torch.float32), move["v"])
                self.words[move["idx"]] = move["word"]
                u_new = self.decode(move["word"][None, :])[0].to(torch.float64)
                self.U[r, move["b"]] = u_new

    def best_block(self, order: np.ndarray, steps: tuple[int, ...], top_per: int,
                   top_global: int, triple_top: int):
        all_moves = []
        with torch.no_grad():
            for idx in order:
                r, b = divmod(int(idx), self.B)
                all_moves.extend(self.candidate_moves(r, b, steps, top_per))
                # Keep only the global shortlist so discarded rank-3 tensors do
                # not accumulate on the GPU across all 1,144 vectors.
                if len(all_moves) > 2 * top_global:
                    all_moves.sort(key=lambda item: item["dE"])
                    all_moves = all_moves[:top_global]
        all_moves.sort(key=lambda item: item["dE"])
        all_moves = all_moves[:top_global]
        best = (float("inf"), [])
        for move in all_moves:
            if move["dE"] < best[0]:
                best = (move["dE"], [move])
        for left, right in itertools.combinations(all_moves, 2):
            if left["idx"] == right["idx"]:
                continue
            cost = self.combo_cost((left, right))
            if cost < best[0]:
                best = (cost, [left, right])
        small = all_moves[:triple_top]
        for combo in itertools.combinations(small, 3):
            if len({m["idx"] for m in combo}) < 3:
                continue
            cost = self.combo_cost(combo)
            if cost < best[0]:
                best = (cost, list(combo))
        return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--top-per", type=int, default=3)
    parser.add_argument("--top-global", type=int, default=80)
    parser.add_argument("--triple-top", type=int, default=16)
    parser.add_argument("--steps", type=int, nargs="+", default=[1, -1])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    problem = dt.Problem.build(refdata.load())
    with np.load(args.start, allow_pickle=False) as data:
        words = np.array(data["words"]).astype(np.int64)
        w = torch.tensor(np.array(data["w"]), dtype=torch.float64, device=problem.device)
        edges = np.array(data["tree_edges"]).astype(np.int64)
        extra = {"copy_index": np.array(data["copy_index"])} if "copy_index" in data.files else {}
        brot = int(data["brot"]) if "brot" in data.files else 8
    search = GlobalBeam(problem, words, w, edges, brot)
    print(f"global beam start error_2 {search.error():.9f}", flush=True)
    search.refit(2)
    print(f"after refit error_2 {search.error():.9f}", flush=True)
    rng = np.random.default_rng(args.seed)
    log = []
    best_err = search.error()
    t0 = time.time()
    order = np.arange(search.R * search.B)
    for rnd in range(1, args.rounds + 1):
        rng.shuffle(order)
        cost, moves = search.best_block(order, tuple(args.steps), args.top_per, args.top_global, args.triple_top)
        if cost >= -1e-10:
            print(f"round {rnd}: no negative block ({cost:.3g})", flush=True)
            break
        search.apply_combo(moves)
        search.refit(1)
        err = search.error()
        print(f"round {rnd}: block_size {len(moves)} predicted {cost:.9f} error_2 {err:.9f} ({time.time() - t0:.0f}s)", flush=True)
        log.append({"round": rnd, "block_size": len(moves), "predicted_dE": cost, "error_2": err})
        if err < best_err:
            best_err = err
            args.out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.out, words=search.words.cpu().numpy(), w=search.w.cpu().numpy(), tree_edges=edges, brot=np.array(brot), **extra)
    args.out.with_suffix(".json").write_text(json.dumps({"best_error_2": best_err, "log": log}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
