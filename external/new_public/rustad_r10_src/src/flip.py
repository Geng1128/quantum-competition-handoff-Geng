"""Exact greedy word-flip search for error_2 at a fixed brot, alternating with L-BFGS refits of w.

The residual D = A^T A - T (packed, P = N(N+1)/2) is kept exactly. Changing one two-body vector u_rb changes its
packed outer product by x, which updates D by x s^T + s x^T + k x x^T with s = sum_c w_rbc a_rc and k = sum_c w_rbc^2,
so the exact change of ||D||^2 of any candidate costs one D x product.

  python src/flip.py IN.npz OUT.npz --brot 7 --rounds 20
IN may be a factorization npz (U, W, H2_I, tree_edges) or an earlier OUT (adds words2).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dft_eval as de  # noqa: E402
from qat import Problem, decode, dev, DT, N  # noqa: E402


class State:
    def __init__(self, P, words, w, brot):
        self.P, self.brot = P, brot
        self.R, self.B, self.C = w.shape
        self.words = torch.tensor(words, dtype=torch.int64, device=dev)  # (RB, N-1)
        self.w = torch.tensor(w, dtype=DT, device=dev)
        self.per_row = 1
        self.refresh()

    def packed(self, vecs):
        return vecs[:, self.P.il_t[0]] * vecs[:, self.P.il_t[1]] * self.P.cw

    def vecs(self, words):
        return decode(words.to(DT) / (1 << (self.brot - 1)), self.P.edges)

    def refresh(self):
        R, B, C = self.R, self.B, self.C
        self.M = self.packed(self.vecs(self.words)).reshape(R, B, -1)  # (R,B,P)
        self.A = torch.einsum("rbc,rbp->rcp", self.w, self.M)  # (R,C,P)
        Af = self.A.reshape(R * C, -1)
        self.D = Af.T @ Af - self.P.Tp
        self.S = torch.einsum("rbc,rcp->rbp", self.w, self.A)  # (R,B,P)
        self.kap = (self.w ** 2).sum(dim=2)  # (R,B)

    def e2(self):
        return torch.linalg.norm(self.D).item()

    def deltas(self, s, kap, x):
        """Exact change of ||D||^2 for candidate updates x (K,P) of rows with s (K,P), kap (K,)."""
        Dx = x @ self.D  # D symmetric
        sDx = (s * Dx).sum(1)
        xDx = (x * Dx).sum(1)
        xx = (x * x).sum(1)
        ss = (s * s).sum(1)
        xs = (x * s).sum(1)
        inner = 2 * sDx + kap * xDx
        norm2 = 2 * (xx * ss + xs ** 2) + kap ** 2 * xx ** 2 + 4 * kap * xx * xs
        return 2 * inner + norm2

    def apply(self, r, b, new_words_row, x):
        s = self.S[r, b].clone()
        k = self.kap[r, b]
        self.D += torch.outer(x, s) + torch.outer(s, x) + k * torch.outer(x, x)
        self.words[r * self.B + b] = new_words_row
        self.M[r, b] += x
        self.A[r] += self.w[r, b][:, None] * x[None, :]
        self.S[r] += (self.w[r] @ self.w[r, b])[:, None] * x[None, :]

    def sweep(self, rows_per_chunk=8, max_apply=None):
        """Evaluate all +-1 moves of every word; apply improving ones greedily with exact re-checks."""
        RB, T = self.words.shape
        mod = 1 << self.brot
        cands = []  # (delta, row, t, dir)
        for r0 in range(0, RB, rows_per_chunk):
            rows = torch.arange(r0, min(RB, r0 + rows_per_chunk), device=dev)
            K = len(rows)
            base = self.words[rows]  # (K,T)
            nw = base[:, None, None, :].repeat(1, 2, T, 1)  # (K,2,T,T)
            idx = torch.arange(T, device=dev)
            nw[:, 0, idx, idx] = (nw[:, 0, idx, idx] + 1) % mod
            nw[:, 1, idx, idx] = (nw[:, 1, idx, idx] - 1) % mod
            nwf = nw.reshape(-1, T)
            mnew = self.packed(self.vecs(nwf)).reshape(K, 2 * T, -1)
            r_idx, b_idx = rows // self.B, rows % self.B
            x = mnew - self.M[r_idx, b_idx][:, None, :]
            s = self.S[r_idx, b_idx][:, None, :].expand(-1, 2 * T, -1)
            kap = self.kap[r_idx, b_idx][:, None].expand(-1, 2 * T)
            d = self.deltas(s.reshape(-1, s.shape[-1]), kap.reshape(-1), x.reshape(-1, x.shape[-1])).reshape(K, 2 * T)
            vals, args = torch.topk(-d, self.per_row, dim=1)
            vals, args = (-vals).cpu().numpy(), args.cpu().numpy()
            for k in range(K):
                for v, ag in zip(vals[k], args[k]):
                    if v < 0:
                        cands.append((float(v), int(rows[k]), int(ag) % T, 1 if int(ag) < T else -1))
        cands.sort()
        applied = 0
        for _, row, t, dr in cands:
            if max_apply and applied >= max_apply:
                break
            r, b = row // self.B, row % self.B
            nwr = self.words[row].clone()
            nwr[t] = (nwr[t] + dr) % mod
            mnew = self.packed(self.vecs(nwr[None]))[0]
            x = mnew - self.M[r, b]
            d = self.deltas(self.S[r, b][None], self.kap[r, b][None], x[None])[0].item()
            if d < 0:
                self.apply(r, b, nwr, x)
                applied += 1
        return applied, len(cands)

    def refit_w(self, iters=10):
        R, B, C = self.R, self.B, self.C
        M = self.M.detach()
        w = self.w.clone().requires_grad_(True)
        opt = torch.optim.LBFGS([w], lr=1, max_iter=20, history_size=50, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            A = torch.einsum("rbc,rbp->rcp", w, M).reshape(R * C, -1)
            e = torch.linalg.norm(A.T @ A - self.P.Tp)
            e.backward()
            return e
        for _ in range(iters):
            opt.step(closure)
        self.w = w.detach()
        self.refresh()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--brot", type=int, default=7)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--refit", type=int, default=5)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--per-row", type=int, default=1)
    a = ap.parse_args()
    z = np.load(a.inp, allow_pickle=False)
    params = {k: np.array(z[k]) for k in z.files}
    edges = params["tree_edges"].astype(np.int64)
    w0, _ = de.native(params)
    if "words2" in params and int(params.get("brot", -1)) == a.brot:
        words = params["words2"]
    else:
        words = de.to_words(de.tree_angles(params["U"].reshape(-1, N), edges), a.brot)
    P = Problem(edges)
    st = State(P, words, w0, a.brot)
    st.per_row = a.per_row
    print(f"start e2 {1e3 * st.e2():.3f} mHa", flush=True)
    t0 = time.time()
    for rd in range(a.rounds):
        n, nc = st.sweep(a.chunk)
        e_flip = st.e2()
        st.refit_w(a.refit)
        print(f"round {rd}: applied {n}/{nc} flips -> e2 {1e3 * e_flip:.3f}, after refit {1e3 * st.e2():.3f} mHa "
              f"{time.time() - t0:.0f}s", flush=True)
        save(a.out, params, st, edges)
        if n == 0:
            break


def save(path, params, st, edges):
    out = dict(params)
    R, B, C = st.R, st.B, st.C
    words2 = st.words.cpu().numpy()
    w = st.w.cpu().numpy()
    _, Wo = de.native(params)
    out.update(U=de.decode(words2, edges, st.brot).reshape(R, B, N), W=w, H2_I=Wo + w.sum(axis=1), words2=words2,
               brot=np.array(st.brot))
    np.savez(path, **out)


if __name__ == "__main__":
    main()
