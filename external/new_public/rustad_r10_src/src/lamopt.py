"""Joint (w, Wo) optimization of lambda_eff with the two-body words fixed, error_2 kept below a cap.

lambda_eff^2 = E_gap (2 Lambda - E_gap): E_gap depends on the negative eigenvalues of h1'(w, Wo) and on sum Wo^2,
Lambda on |eig h1'| and the weight one-norm. Words are fixed, so error_2 depends on w only. Afterwards run
finish.py (one-body words for error_1) and gauge.py.

  python src/lamopt.py IN.npz OUT.npz [--steps 3000 --cap 0.0193]
"""
import argparse
import sys
import time

import numpy as np
import torch

sys.path.insert(0, r"C:\Users\rusta\qcc\src")
import dft_eval as de  # noqa: E402
from qat import Problem, decode, dev, DT, N  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--cap", type=float, default=0.0193)
    ap.add_argument("--gap-margin", type=float, default=0.003)
    ap.add_argument("--e2-tol", type=float, default=2e-4, help="feasible if error_2 <= cap + e2_tol")
    ap.add_argument("--gap-feas", type=float, default=1e-3, help="feasible if E_gap > floor + gap_feas")
    ap.add_argument("--lr-w", type=float, default=2e-5)
    ap.add_argument("--lr-o", type=float, default=2e-4)
    ap.add_argument("--mu", type=float, default=1e6)
    ap.add_argument("--log", type=int, default=100)
    ap.add_argument("--decay", type=float, default=1.0, help="lr multiplier reached at the last step (exponential)")
    a = ap.parse_args()
    z = np.load(a.inp, allow_pickle=False)
    params = {k: np.array(z[k]) for k in z.files}
    brot = int(params["brot"])
    edges = params["tree_edges"].astype(np.int64)
    w0, Wo0 = de.native(params)
    R, B, C = w0.shape
    words2 = params["words2"] if "words2" in params else params["angle_words"][N:]
    P = Problem(edges)
    v = decode(torch.tensor(words2, dtype=DT, device=dev) / (1 << (brot - 1)), P.edges).reshape(R, B, N)
    M = v[:, :, P.il_t[0]] * v[:, :, P.il_t[1]] * P.cw
    w = torch.tensor(w0, dtype=DT, device=dev, requires_grad=True)
    Wo = torch.tensor(Wo0, dtype=DT, device=dev, requires_grad=True)

    def metrics():
        A = torch.einsum("rbc,rbp->rcp", w, M).reshape(R * C, -1)
        e2 = torch.linalg.norm(A.T @ A - P.Tp)
        X = torch.einsum("rbp,rbq,rbc->rcpq", v, v, w)
        h1p = P.base1 - torch.einsum("rc,rcpq->pq", Wo, X)
        lam = torch.linalg.eigvalsh(h1p)
        lam1 = lam.abs().sum()
        wB = Wo + w.sum(dim=1)
        Lam = lam1 + 0.25 * ((w.abs().sum(dim=1) + wB.abs()) ** 2).sum()
        gap = de.E_UPPER - (P.const_sos - (lam1 - lam.sum()) - 0.5 * (Wo ** 2).sum())
        return e2, gap, Lam, torch.sqrt(gap * (2 * Lam - gap))

    opt = torch.optim.Adam([{"params": [w], "lr": a.lr_w}, {"params": [Wo], "lr": a.lr_o}])
    best, best_state = None, None
    t0 = time.time()
    for s in range(a.steps + 1):
        e2, gap, Lam, leff = metrics()
        feas = e2.item() <= a.cap + a.e2_tol and gap.item() > de.GAP_FLOOR + a.gap_feas and Lam.item() > 49.5
        if feas and (best is None or leff.item() < best):
            best = leff.item()
            best_state = (w.detach().cpu().numpy().copy(), Wo.detach().cpu().numpy().copy())
        if s % a.log == 0:
            print(f"{s:5d} leff {leff.item():.4f} gap {gap.item():.6f} Lam {Lam.item():.3f} e2 {1e3 * e2.item():.3f} "
                  f"best {best} {time.time() - t0:.0f}s", flush=True)
        if s == a.steps:
            break
        loss = torch.log(leff) + a.mu * (torch.relu(e2 - a.cap) ** 2
                                         + torch.relu(de.GAP_FLOOR + a.gap_margin - gap) ** 2
                                         + torch.relu(49.5 - Lam) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        for g, lr0 in zip(opt.param_groups, (a.lr_w, a.lr_o)):
            g["lr"] = lr0 * a.decay ** ((s + 1) / a.steps)
    wb, Wob = best_state if best_state is not None else (w0, Wo0)
    out = dict(params)
    out.update(W=wb, H2_I=Wob + wb.sum(axis=1), words2=words2)
    out.pop("words1", None)
    np.savez(a.out, **out)
    print("saved", a.out, "best leff", best)


if __name__ == "__main__":
    main()
