"""Turn a two-body word checkpoint into a complete chemistry factorization.

1. U from the fixed two-body words; w from the checkpoint; W offsets from the
   source factorization's kept copies.
2. Polish (w, W) for lambda_eff under error_2 / E_gap constraints.
3. One-body rows: brot-bit words of the h1' eigenvectors, then greedy +-1 word
   moves until error_1 is under its target.
4. Write submission.npz and the full (76 + R*B, 75) angle-word table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

import dfthc_torch as dt
import givens
import refdata

if str(refdata.VERIFIERS_DIR) not in sys.path:
    sys.path.insert(0, str(refdata.VERIFIERS_DIR))
from dfthcsos.verify import native_inputs  # noqa: E402


def polish(problem, U, w, Woff, steps, lr, e2_t, gap_t, scale=1e-3):
    w = w.clone().requires_grad_(True)
    Woff = Woff.clone().requires_grad_(True)
    opt = torch.optim.Adam([{"params": [w], "lr": lr * 0.01}, {"params": [Woff], "lr": lr}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.02)
    best = None
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        m = dt.metrics(problem, U, w, Woff)
        pen = (torch.relu(m["error_2"] - e2_t) / scale) ** 2 + (torch.relu(gap_t - m["E_gap"]) / scale) ** 2
        loss = m["lambda_eff"] + pen
        ok = float(m["error_2"]) <= e2_t and float(m["E_gap"]) >= gap_t
        lam = float(m["lambda_eff"].detach())
        if ok and (best is None or lam < best[0]):
            best = (lam, w.detach().clone(), Woff.detach().clone())
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0:
            print(f"  polish {step}: lam {lam:.4f} e2 {float(m['error_2']):.6f} gap {float(m['E_gap']):.5f} best {best[0] if best else None}", flush=True)
    return best


def one_body_words(h1p: np.ndarray, edges: np.ndarray, brot: int, e1_target: float, sweeps: int = 20):
    """Quantized eigenvectors of h1', then greedy +-1 word moves to reduce error_1."""
    vals, vecs = np.linalg.eigh(h1p)
    words = givens.to_words(givens.encode_numpy(vecs.T, edges), brot)
    dev = "cuda"
    H = torch.tensor(h1p, dtype=torch.float64, device=dev)
    lam = torch.tensor(vals, dtype=torch.float64, device=dev)

    def decode(wd):
        return torch.tensor(givens.decode_numpy(givens.words_to_theta(wd, brot), edges), dtype=torch.float64, device=dev)

    V = decode(words)
    assign = linear_sum_assignment(-np.abs(V.cpu().numpy() @ vecs))[1]
    lam_rows = lam[torch.tensor(assign, device=dev)]

    def e1_of(Vm):
        return torch.linalg.vector_norm(torch.einsum("j,jp,jq->pq", lam_rows, Vm, Vm) - H)

    e1 = float(e1_of(V))
    print(f"  one-body rounded error_1 {e1:.6f}", flush=True)
    mod = 1 << brot
    for sw in range(sweeps):
        if e1 <= e1_target:
            break
        improved = False
        for j in range(V.shape[0]):
            base = words[j]
            cands = np.repeat(base[None], 150, axis=0)
            for t in range(75):
                cands[t, t] = (cands[t, t] + 1) % mod
                cands[75 + t, t] = (cands[75 + t, t] - 1) % mod
            U_c = decode(cands)  # (150, N)
            M = torch.einsum("j,jp,jq->pq", lam_rows, V, V) - H
            M = M - lam_rows[j] * torch.outer(V[j], V[j])
            trial = M[None] + lam_rows[j] * U_c[:, :, None] * U_c[:, None, :]
            errs = torch.linalg.vector_norm(trial.reshape(150, -1), dim=1)
            k = int(torch.argmin(errs))
            if float(errs[k]) < e1 - 1e-12:
                words[j] = cands[k]
                V[j] = U_c[k]
                e1 = float(errs[k])
                improved = True
        print(f"  one-body sweep {sw}: error_1 {e1:.6f}", flush=True)
        if not improved:
            break
    return words, e1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="source chemistry dir (submission.npz) for W offsets / shifts")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--e2", type=float, default=0.01996)
    parser.add_argument("--e1", type=float, default=0.0198)
    parser.add_argument("--gap-margin", type=float, default=3e-5)
    args = parser.parse_args()

    problem = dt.Problem.build(refdata.load())
    dev = problem.device
    with np.load(args.checkpoint, allow_pickle=False) as data:
        words2 = np.array(data["words"]).astype(np.int64)
        w = np.array(data["w"])
        edges = np.array(data["tree_edges"]).astype(np.int64)
        brot = int(data["brot"])
        copy_index = np.array(data["copy_index"]) if "copy_index" in data.files else None
    with np.load(args.source / "submission.npz", allow_pickle=False) as data:
        src = {k: np.array(data[k]) for k in data.files}
    _, W_src, _, _ = native_inputs(src)
    R, B, C = w.shape
    Woff = W_src if copy_index is None else np.stack([W_src[r, copy_index[r]] for r in range(R)])
    U = torch.tensor(givens.decode_numpy(givens.words_to_theta(words2, brot), edges).reshape(R, B, 76), device=dev)
    w_t = torch.tensor(w, dtype=torch.float64, device=dev)
    W_t = torch.tensor(Woff, dtype=torch.float64, device=dev)
    with torch.no_grad():
        start = dt.scalar_report(dt.metrics(problem, U, w_t, W_t))
    print("start", json.dumps(start), flush=True)
    best = polish(problem, U, w_t, W_t, args.steps, args.lr, args.e2, refdata.E_GAP_FLOOR + args.gap_margin)
    if best is None:
        raise SystemExit("polish found no feasible point")
    lam, w_b, W_b = best
    with torch.no_grad():
        m = dt.metrics(problem, U, w_b, W_b)
    words1, e1 = one_body_words(m["h1_prime"].cpu().numpy(), edges, brot, args.e1)
    if e1 > 0.02:
        raise SystemExit(f"error_1 {e1} above the limit")
    args.out.mkdir(parents=True, exist_ok=True)
    w_np, W_np = w_b.cpu().numpy(), W_b.cpu().numpy()
    all_words = np.concatenate([words1, words2], axis=0)
    np.savez(
        args.out / "submission.npz",
        U=U.cpu().numpy(), W=w_np, H2_I=W_np + w_np.sum(axis=1),
        H1_I=src["H1_I"], Hsym_lower=src["Hsym_lower"], tree_edges=edges,
    )
    np.save(args.out / "words.npy", all_words)
    report = {"start": start, "final": {**dt.scalar_report(m), "error_1": e1}, "shape": [R, B, C], "brot": brot}
    (args.out / "finalize_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report["final"], indent=1))


if __name__ == "__main__":
    main()
