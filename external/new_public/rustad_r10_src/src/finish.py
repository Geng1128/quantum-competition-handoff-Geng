"""Stage 2 of the brot-k refit: Wo for lambda_eff, then one-body words for error_1 (two-body words and w fixed).

(1) L-BFGS-B on the offsets Wo for the exact lambda_eff with gap/Lambda floor penalties, using analytic eigenvalue
    derivatives (after chemistry_refine.optimize_effective_norm by Ozer / anazemcev / Kozykorpesh, Apache-2.0).
    error_2 does not depend on Wo.
(2) One-body rows: nearest-rounded eigenvectors of h1', then exact greedy +-1 word flips (GPU) on
    ||sum_j lam_j q_j q_j^T - h1'||_F until error_1 is below target.

  python src/finish.py IN.npz OUT.npz
IN must hold words2 and brot (flip.py / pq.py output).
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch
from scipy.optimize import minimize, linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dft_eval as de  # noqa: E402
from qat import decode, dev, DT, N  # noqa: E402


def effective_norm(u, w, Wo0, gap_margin=0.002, maxiter=400, kappa=0.0):
    r, _, c = w.shape
    S, beta = de.shifts()
    ref = de.ref()
    h1_thc = ref["oei"] - 0.5 * np.einsum("prrq->pq", ref["tei"])
    constant = h1_thc - 0.5 * de.ETA * S + beta * np.eye(N)
    energy = de.E_UPPER - ref["ecore"] + de.ETA * beta
    sum_w, l1_w = w.sum(axis=1), np.abs(w).sum(axis=1)
    floor = de.GAP_FLOOR + gap_margin

    def quantities(flat):
        off = flat.reshape(r, c)
        comb = np.einsum("rc,rbc->rb", off, w)
        h1 = constant - np.einsum("rb,rbp,rbq->pq", comb, u, u, optimize=True)
        vals, vecs = np.linalg.eigh(h1)
        sgn = (vecs * np.sign(vals)) @ vecs.T
        con = np.einsum("rbp,pq,rbq->rb", u, sgn, u, optimize=True)
        gap = float(energy + np.abs(vals).sum() - vals.sum() + 0.5 * np.sum(off ** 2))
        wB = off + sum_w
        lam = float(np.abs(vals).sum() + 0.25 * np.sum((l1_w + np.abs(wB)) ** 2))
        g_gap = off + np.einsum("rb,rbc->rc", 1.0 - con, w)
        g_lam = -np.einsum("rb,rbc->rc", con, w) + 0.5 * (l1_w + np.abs(wB)) * np.sign(wB)
        # d/dWo of ||h1'||_F^2 = -2 <h1', X_rc>
        quantities.fro2 = float((vals ** 2).sum())
        quantities.g_fro2 = -2 * np.einsum("rb,rbp,pq,rbq->rb", np.ones_like(comb), u, h1, u, optimize=True)
        return gap, lam, g_gap, g_lam

    def objective(flat):
        gap, lam, gg, gl = quantities(flat)
        rad = gap * (2.0 * lam - gap)
        if rad <= 0:
            return 1e12 + abs(rad), np.zeros_like(flat)
        loss = 0.5 * math.log(rad) + kappa * quantities.fro2
        grad = ((lam - gap) * gg + gap * gl) / rad + kappa * np.einsum("rb,rbc->rc", quantities.g_fro2, w)
        if gap < floor:
            loss += 1e5 * (floor - gap) ** 2
            grad -= 2e5 * (floor - gap) * gg
        if lam < 49.001:
            loss += 1e5 * (49.001 - lam) ** 2
            grad -= 2e5 * (49.001 - lam) * gl
        return loss, grad.ravel()

    g0, l0, _, _ = quantities(Wo0.ravel())
    res = minimize(objective, Wo0.ravel(), jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "maxcor": 12, "ftol": 1e-14, "gtol": 1e-9, "maxls": 40})
    off = res.x.reshape(r, c)
    g1, l1, _, _ = quantities(off.ravel())
    print(f"  ||h1'||_F^2 = {quantities.fro2:.4f}")
    print(f"effective norm: gap {g0:.4f}->{g1:.4f} Lambda {l0:.3f}->{l1:.3f} "
          f"leff {math.sqrt(g0 * (2 * l0 - g0)):.4f}->{math.sqrt(g1 * (2 * l1 - g1)):.4f} it {res.nit}", flush=True)
    return off, constant


def one_body_flips(h1p_np, edges, brot, target, max_sweeps=1000, words1=None):
    scale, mod = 1 << (brot - 1), 1 << brot
    ed = [tuple(map(int, e)) for e in edges]
    h1p = torch.tensor(h1p_np, dtype=DT, device=dev)
    lamv, vec = torch.linalg.eigh(h1p)
    if words1 is None:
        words1 = de.to_words(de.tree_angles(vec.T.cpu().numpy(), edges), brot)
    words1 = torch.tensor(words1, dtype=torch.int64, device=dev)
    q_of = lambda w_: decode(w_.to(DT) / scale, ed)  # noqa: E731
    q1 = q_of(words1)
    ov = (q1 @ vec).abs().cpu().numpy()
    ri, ci = linear_sum_assignment(-ov)
    lj = lamv[torch.tensor(ci[np.argsort(ri)], device=dev)]
    D1 = torch.einsum("j,jp,jq->pq", lj, q1, q1) - h1p
    T = N - 1
    idx = torch.arange(T, device=dev)
    for sweep in range(max_sweeps):
        e1 = torch.linalg.norm(D1).item()
        if e1 < target:
            break
        steps = (1, -1, 2, -2)
        nw = words1[:, None, None, :].repeat(1, len(steps), T, 1)
        for si, st in enumerate(steps):
            nw[:, si, idx, idx] = (nw[:, si, idx, idx] + st) % mod
        qn = q_of(nw.reshape(-1, T)).reshape(N, len(steps) * T, N)
        # unit vectors: <D1, qq^T - pp^T> = q'D1q - p'D1p and ||qq^T - pp^T||^2 = 2 - 2 (q.p)^2
        l = lj[:, None]
        qDq = torch.einsum("jkp,pq,jkq->jk", qn, D1, qn)
        pDp = torch.einsum("jp,pq,jq->j", q1, D1, q1)[:, None]
        qp = torch.einsum("jkp,jp->jk", qn, q1)
        dlt = 2 * l * (qDq - pDp) + l ** 2 * (2 - 2 * qp ** 2)
        best, arg = dlt.min(dim=1)
        applied = 0
        for j in torch.argsort(best).tolist():
            if best[j].item() >= 0:
                break
            k = int(arg[j])
            t, dr = k % T, steps[k // T]
            nwr = words1[j].clone()
            nwr[t] = (nwr[t] + dr) % mod
            qj = q_of(nwr[None])[0]
            xj = torch.outer(qj, qj) - torch.outer(q1[j], q1[j])
            d = (2 * lj[j] * (D1 * xj).sum() + lj[j] ** 2 * (xj * xj).sum()).item()
            if d < 0:
                D1 += lj[j] * xj
                words1[j], q1[j] = nwr, qj
                applied += 1
        if sweep % 10 == 0:
            print(f"  one-body sweep {sweep}: e1 {1e3 * torch.linalg.norm(D1).item():.3f} mHa ({applied} flips)", flush=True)
        if applied == 0:
            break
    print(f"  one-body final e1 {1e3 * torch.linalg.norm(D1).item():.3f} mHa", flush=True)
    return words1.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--target", type=float, default=0.0192)
    ap.add_argument("--gap-margin", type=float, default=0.002)
    ap.add_argument("--kappa", type=float, default=0.0)
    a = ap.parse_args()
    t0 = time.time()
    z = np.load(a.inp, allow_pickle=False)
    params = {k: np.array(z[k]) for k in z.files}
    brot = int(params["brot"])
    edges = params["tree_edges"].astype(np.int64)
    w, Wo0 = de.native(params)
    R, B, C = w.shape
    words2 = params["words2"] if "words2" in params else params["angle_words"][N:]
    u = de.decode(words2, edges, brot).reshape(R, B, N)
    Wo, constant = effective_norm(u, w, Wo0, a.gap_margin, kappa=a.kappa)
    X = np.einsum("rbp,rbq,rbc->rcpq", u, u, w)
    h1p = constant - np.einsum("rc,rcpq->pq", Wo, X)
    words1 = one_body_flips(h1p, edges, brot, a.target)
    words = np.vstack([words1, words2])
    m = de.evaluate(words, w, Wo, edges, brot)
    print("verifier-math:", {k: round(float(v), 6) for k, v in m.items()}, "OK" if de.ok(m) else "FAIL",
          f"{time.time() - t0:.0f}s")
    out = dict(params)
    out.update(H2_I=Wo + w.sum(axis=1), words1=words1, words2=words2, angle_words=words, brot=np.array(brot))
    np.savez(a.out, **out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
