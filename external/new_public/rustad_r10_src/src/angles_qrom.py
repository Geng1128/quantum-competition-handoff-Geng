"""ANGLE_k circuits for any brot / batch size J: signed half-open gauge + four-address banked QROM.

Generalizes gauge_qrom.build (Ahmed Berat Ozer, Apache-2.0; uses banked_lab by Kozykorpesh / Sumit Chongder /
Kirill Tymchishin / anazemcev, Apache-2.0) from (brot=10, J=3) to any (brot, J) that fits in 128 qubits:
selection n + words brot*J + bank 3*(brot-1)*J + 5 scratch + 1 <= 128.

Half-open gauge: every word has bit(brot-1) == bit(brot-2), so only brot-1 bits are loaded and the top bit is copied.
"""
from __future__ import annotations

import numpy as np


def canonical_words(words, tree_edges, brot):
    """Re-encode decoded vectors in the signed half-open gauge (exact, up to a per-row sign)."""
    import sys
    from dfthcsos.scoreinfra.dfthc_math import reconstruct_u_vectors
    u = reconstruct_u_vectors(np.asarray(words) / (1 << (brot - 1)), tree_edges)
    a = u.copy()
    angles = np.empty(np.shape(words), dtype=np.float64)
    for j, (p, q) in reversed(list(enumerate(tree_edges))):
        norm = np.hypot(a[:, p], a[:, q])
        sign = np.where(np.abs(a[:, p]) < 1e-12, np.where(a[:, q] > 0, -1, 1), np.where(a[:, p] < 0, -1, 1))
        angles[:, j] = np.where(norm < 1e-12, 0, np.arctan2(a[:, q] * sign, a[:, p] * sign))
        a[:, p] = np.where(norm < 1e-12, 0, norm * sign)
        a[:, q] = 0
    out = np.rint(angles * ((1 << (brot - 1)) / np.pi)).astype(np.int64) % (1 << brot)
    assert not np.any(((out >> (brot - 1)) ^ (out >> (brot - 2))) & 1), "half-open gauge failed"
    dec = reconstruct_u_vectors(out / (1 << (brot - 1)), tree_edges)
    sign = np.where((u * dec).sum(axis=1) < 0, -1, 1)
    drift = float(np.max(np.abs(dec * sign[:, None] - u)))
    if drift > 1e-9:
        raise ArithmeticError(f"Givens gauge drift {drift}")
    return out


def batch_deltas(words, J):
    from math import ceil
    words = np.asarray(words, dtype=np.int64)
    nv, na = words.shape
    padded = np.zeros((nv, ceil(na / J) * J), dtype=np.int64)
    padded[:, :na] = words
    batches = padded.reshape(nv, -1, J).transpose(1, 0, 2)
    result = batches.copy()
    result[1:] ^= batches[:-1]
    return result  # (K, A, J)


def max_batch(brot, num_addresses, addr=4):
    """Largest J that fits 128 qubits: 4-address gauge banks, or 2-address rank-reduced banks (gauge saves a column)."""
    n = (num_addresses - 1).bit_length()
    if addr == 4:
        need = lambda J: n + brot * J + 3 * (brot - 1) * J + 6  # noqa: E731
    else:
        need = lambda J: n + brot * J + (brot - 1) * J + (n - 2)  # noqa: E731
    J = 1
    while need(J + 1) <= 128:
        J += 1
    return J


def build2(delta, brot, return_report=False):
    """Two-address banked QROM (banked_lab.build_qroam2, linear mode: bank = GF(2) rank of the difference columns)."""
    import banked_lab as B
    c, rep = B.build_qroam2(np.asarray(delta, dtype=np.int64), brot, mode="linear", return_report=True)
    assert c.num_qubits <= 128, rep
    return (c, rep) if return_report else c


def build(delta, brot, return_report=False):
    import kickmix as km
    import banked_lab as B
    from gauge_qrom import pebbled_init, xor_and
    delta = np.asarray(delta, dtype=np.int64)
    A0, J = delta.shape
    pad = (-A0) % 4
    if pad:  # addresses beyond the legal range are never queried; pad with zeros (still in the gauge)
        delta = np.vstack([delta, np.zeros((pad, J), dtype=np.int64)])
    A = len(delta)
    top, nxt = brot - 1, brot - 2
    assert not np.any(((delta >> top) ^ (delta >> nxt)) & 1)
    lo = (1 << (brot - 1)) - 1
    W = brot * J
    BW = (brot - 1) * J
    H = A // 4
    n = (A0 - 1).bit_length()
    packed = [B.pack(row, brot) for row in delta]
    compact = lambda v: sum(((v >> (j * brot)) & lo) << (j * (brot - 1)) for j in range(J))  # noqa: E731
    expand = lambda v: sum((((v >> ((brot - 1) * j)) & lo) | (((v >> ((brot - 1) * j + brot - 2)) & 1) << (brot - 1)))  # noqa: E731
                           << (brot * j) for j in range(J))
    ds = [[], [], []]
    base = []
    for h in range(H):
        v = packed[4 * h:4 * h + 4]
        base.append(v[0])
        ds[0].append(compact(v[0] ^ v[1]))
        ds[1].append(compact(v[0] ^ v[2]))
        ds[2].append(compact(v[0] ^ v[1] ^ v[2] ^ v[3]))
    restored = [base[h] ^ expand((ds[0][h] if a & 1 else 0) ^ (ds[1][h] if a & 2 else 0) ^ (ds[2][h] if (a & 3) == 3 else 0))
                for h in range(H) for a in range(4)]
    assert restored == packed
    values = [base[h] | (ds[0][h] << W) | (ds[1][h] << (W + BW)) | (ds[2][h] << (W + 2 * BW)) for h in range(H)]
    hb = n - 2
    b = km.CircuitBuilder()
    sel = b.create_quantum_register(n, "selection")
    tgt = [q for j in range(J) for q in b.create_quantum_register(brot, f"word_{j}")]
    bank = list(b.alloc_qubits(3 * BW))
    scratch = b.alloc_qubits(5)
    completed, stats = pebbled_init(b, values, sel[2:], tgt + bank, scratch)  # truncated ANF over the H legal rows
    b.free(scratch)
    both = b.alloc_qubits(1)[0]
    b.init_and(sel[0], sel[1], both)
    for s, ctl in enumerate((sel[0], sel[1], both)):
        for j in range(J):
            for k in range(brot - 1):
                ts = [tgt[j * brot + k]] + ([tgt[j * brot + brot - 1]] if k == brot - 2 else [])
                xor_and(b, ctl, bank[s * BW + j * (brot - 1) + k], ts)
    b.del_and(sel[0], sel[1], both)
    b.free(both)
    table = [v >> W for v in completed]  # completed covers all 2**hb high addresses
    erase = B.del_lookup_anf(b, table=table, address=sel[2:], target=bank, low_bits=min(5, hb - 1))
    b.free(km.array(bank))
    c = b.finish_circuit()
    report = {"num_qubits": c.num_qubits, "static_toffoli": int(c.operation_statistics().num_toffoli_like),
              "init": stats, "cleanup": erase}
    assert c.num_qubits <= 128, report
    return (c, report) if return_report else c
