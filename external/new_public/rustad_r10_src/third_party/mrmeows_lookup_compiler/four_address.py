# Adapted from marculera/dfthc-merge-pipeline, dfthc_refit/fourqrom.py.
# Source: https://www.kaggle.com/datasets/marculera/dfthc-merge-pipeline
# Licensed Apache-2.0 per the dataset and its README.
# Original paired-combine and erasure implementations are retained.
# Modification: replace the high-address unary/Gray loader with exact prefix
# ANF synthesis and direct-to-target leaves; parameterize word precision below.
# Attribution details are preserved in FOUR_ADDRESS_NOTICE.md.
"""Four-address ANGLE backend with paired combine and exact ANF loading.

For three ten-bit signed-gauge words, decompose each group of four rows
into a base and three nine-bit-per-word difference functions.  Load the
base and the 81 difference-bank bits through a prefix-Mobius circuit on
the high address.  Combine the two low address bits using 41 paired
Toffolis plus one AND, then measurement-erase the bank with an exact
phase lookup on valid addresses.

At D=876: H=219 high addresses, eight high-address bits, 210 nonlinear
prefix monomials, 42 combine Toffolis, and 23 phase-erasure Toffolis.
This is at most 275 Toffolis and 126 qubits per circuit.  The XOR action
holds for arbitrary initial word registers.  Measurement controls are
used solely for exact phase correction; no Toffoli is classically
conditioned to reduce its measured count.

The paired-combine and erasure code is adapted from the Apache-2.0
source identified in the file header and FOUR_ADDRESS_NOTICE.md.
"""
from __future__ import annotations

import numpy as np
import kickmix as km

BROT, J, BANK_BITS = 10, 3, 9


def _bits(value: int) -> list[int]:
    out, i = [], 0
    while value:
        if value & 1:
            out.append(i)
        value >>= 1
        i += 1
    return out


def _erase(b: km.CircuitBuilder, bank: list, addr, table: list[int]) -> None:
    """Measurement-based erase of `bank` = table[h] (h < len(table)), like kickmix's del_lookup,
    but the phase lookup only covers the rows that exist.

    X-measure the bank, then flip the phase of address h by P[h] = parity(m & table[h]). The low k
    address bits are unfolded into 2**k one-hot qubits u (reusing the measured bank qubits); a
    unary iteration over the high bits applies, per entry v, Z on the parity of the u_l with
    P[v 2**k + l] = 1 (classically conditioned CNOTs into one accumulator, then CZ or CCZ).
    """
    H, hb, n = len(table), len(addr), len(bank)
    m = b.alloc_bits(n)
    b.hmr(km.array(bank), m)
    P = b.alloc_bits(H)
    b.bit_store(P, False)
    for j in range(n):
        hs = [P[h] for h in range(H) if table[h] >> j & 1]
        if hs:
            b.push_condition(m[j])
            b.cx(True, km.array(hs))
            b.pop_condition()

    k = min(range(1, hb + 1), key=lambda k: (1 << k) - k - 1 + -(-H >> k))
    V = -(-H >> k)
    u, acc, pool = bank[:1 << k], bank[1 << k], list(bank[(1 << k) + 1:])

    def split(i: int, forward: bool) -> None:
        """u_l -> (u_l & ~a_i, u_l & a_i) for l < 2**i. The new halves XOR to a_i, so the last
        one is a_i ^ (the others) by CNOTs: 2**i - 1 Toffolis; the reverse is measurement-based."""
        s, last = 1 << i, (1 << i) - 1

        def fill_last() -> None:  # u[last + s] ^= a_i ^ (u[l + s] for l < last)
            for q in [addr[i]] + [u[l + s] for l in range(last)]:
                b.cx(q, u[last + s])

        if forward:
            for l in range(last):
                b.init_and(u[l], addr[i], u[l + s])
                b.cx(u[l + s], u[l])
            fill_last()
            b.cx(u[last + s], u[last])
        else:
            b.cx(u[last + s], u[last])
            fill_last()
            for l in reversed(range(last)):
                b.cx(u[l + s], u[l])
                b.del_and(u[l], addr[i], u[l + s])

    b.x(u[0])
    for i in range(k):
        split(i, True)

    def parity(v: int) -> None:  # acc ^= XOR of u_l over P[v 2**k + l]
        for l in range(1 << k):
            h = (v << k) + l
            if h < H:
                b.push_condition(P[h])
                b.cx(u[l], acc)
                b.pop_condition()

    hi = addr[k:]

    def leaf(ctrl, v: int, bit) -> None:
        parity(v)
        b.z(acc) if ctrl is None else b.cz(ctrl, acc)
        if v + 1 < V:  # the difference to entry v + 1 under ctrl & bit
            parity(v + 1)
            b.cz(bit, acc) if ctrl is None else b.ccz(ctrl, bit, acc)
            parity(v + 1)
        parity(v)

    def phases(ctrl, bit: int, offset: int) -> None:
        if bit == 0:
            leaf(ctrl, offset, hi[0])
            return
        half = 1 << bit
        if offset + half >= V:
            phases(ctrl, bit - 1, offset)
            return
        if ctrl is None:  # the root: the address bit itself is the control
            b.x(hi[bit])
            phases(hi[bit], bit - 1, offset)
            b.x(hi[bit])
            phases(hi[bit], bit - 1, offset + half)
            return
        t = pool.pop()
        b.x(hi[bit])
        b.init_and(ctrl, hi[bit], t)
        b.x(hi[bit])
        phases(t, bit - 1, offset)
        b.cx(ctrl, t)
        phases(t, bit - 1, offset + half)
        b.del_and(ctrl, hi[bit], t)
        pool.append(t)

    if len(hi):
        phases(None, len(hi) - 1, 0)
    else:
        parity(0)
        b.z(acc)
        parity(0)

    for i in reversed(range(k)):
        split(i, False)
    b.x(u[0])
    b.free(m)
    b.free(P)


def _paired_combine(d1: np.ndarray, d2: np.ndarray, d3: np.ndarray):
    """The bank contents and Toffolis of the paired combine.

    Each of the 27 bit slots s (word w, bit k < 9) needs q_s = l0 a_s + l1 b_s + L c_s over GF(2),
    with L = l0 l1 and (a, b, c) the slot's bits of (D1, D2, D3). A Toffoli whose controls are a
    selection bit XORed onto a bank wire, (L + x)(l0 + y) = L (1 + y) + l0 x + x y, yields two
    useful terms at once; x y depends on h only and is folded into the loaded word. Pairing slots
    i, j covers their six terms with three Toffolis (exhaustively checked over GF(2)):

        S  -> i and j:  (L + a_i)(l0 + ~c_j)
        Pi -> i:        (L + b_i)(l1 + ~(c_i ^ c_j))
        Pj -> j:        (l0 + b_j)(l1 + a_i ^ a_j), plus a CNOT of L

    An odd last slot m takes (L + a_m)(l0 + ~c_m) and l1 b_m. For 27 slots: 13 * 3 + 2 = 41
    Toffolis instead of 81, in the same 81 bank wires.

    Returns the bank values (H,) (bit i = wire i), the word-table corrections (H,) (bit s = slot s),
    and the Toffolis as (control, control, target slots, slots that also get a CNOT of L); a
    control is (selection, wire): selection 'L', 0, 1 or None, wire an index or None.
    """
    slots = d1.shape[1] * BANK_BITS
    pack = lambda d: sum((d[:, i].astype(np.int64) & ((1 << BANK_BITS) - 1)) << (BANK_BITS * i) for i in range(d.shape[1]))  # noqa: E731
    a, bb, c = pack(d1), pack(d2), pack(d3)
    bit = lambda v, s: (v >> s) & 1  # noqa: E731
    wires, corr, toffolis = [], np.zeros(len(a), np.int64), []

    def add(sel_x, x, sel_y, y, targets, cnot_l=()):
        """A Toffoli on (sel_x + wire x)(sel_y + wire y); x or y None means the selection bit alone."""
        nonlocal corr
        cx = (sel_x, None) if x is None else (sel_x, len(wires))
        if x is not None:
            wires.append(x)
        cy = (sel_y, None) if y is None else (sel_y, len(wires))
        if y is not None:
            wires.append(y)
        if x is not None and y is not None:  # the h-only term x y, at l0 = l1 = 0
            for s in targets:
                corr ^= (x & y) << s
        toffolis.append((cx, cy, list(targets), list(cnot_l)))

    for i in range(0, slots - 1, 2):
        j = i + 1
        add('L', bit(a, i), 0, 1 ^ bit(c, j), (i, j))
        add('L', bit(bb, i), 1, 1 ^ bit(c, i) ^ bit(c, j), (i,))
        add(0, bit(bb, j), 1, bit(a, i) ^ bit(a, j), (j,), cnot_l=(j,))
    if slots % 2:
        m = slots - 1
        add('L', bit(a, m), 0, 1 ^ bit(c, m), (m,))
        add(1, None, None, bit(bb, m), (m,))
    assert len(wires) == 3 * slots
    bits = np.stack(wires, axis=1)  # (H, 81): wire values per high address, as Python ints (> 64 bits)
    bank = [sum(1 << int(i) for i in np.flatnonzero(row)) for row in bits]
    return bank, corr, toffolis


def build_circuit(delta: np.ndarray, n_sel: int) -> km.Circuit:
    """The XOR oracle |a>|z> -> |a>|z ^ f(a)> for the (A, 3) word table `delta` (signed gauge)."""
    delta = np.asarray(delta, dtype=np.int64)
    A, j = delta.shape
    assert j == J and A % 4 == 0 and A <= (1 << n_sel)
    assert not np.any(((delta >> (BROT - 1)) ^ (delta >> (BROT - 2))) & 1), 'words must be in the signed gauge'
    H, hb = A // 4, n_sel - 2
    assert hb >= 4, "needs at least 4 high address bits (A > 32)"

    def pack(row) -> int:  # three full 10-bit words
        return sum(int(v) << (BROT * i) for i, v in enumerate(row))

    def spread(slot_bits: int) -> int:  # slot (w, k) -> word bit 10 w + k, and slot (w, 8) also -> bit 10 w + 9
        out = 0
        for s in _bits(slot_bits):
            w, k = divmod(s, BANK_BITS)
            out |= (1 << (BROT * w + k)) | ((1 << (BROT * w + BROT - 1)) if k == BANK_BITS - 1 else 0)
        return out

    q = delta.reshape(H, 4, J)
    d1, d2, d3 = q[:, 0] ^ q[:, 1], q[:, 0] ^ q[:, 2], q[:, 0] ^ q[:, 1] ^ q[:, 2] ^ q[:, 3]
    bank_table, corr, toffolis = _paired_combine(d1, d2, d3)
    W = BROT * J
    # Each word starts at f(4h) plus the combine's h-only terms, so the combine leaves exactly f(4h + l).
    table = [(pack(q[h, 0]) ^ spread(int(corr[h]))) | (bank_table[h] << W) for h in range(H)]

    b = km.CircuitBuilder()
    sel = b.create_quantum_register(n_sel, 'selection')
    words = [b.create_quantum_register(BROT, f'word_{i}') for i in range(J)]
    z = [qq for w in words for qq in w]
    bank = list(b.alloc_qubits(3 * J * BANK_BITS))
    outputs = z + bank
    addr = sel[2:]

    def xor_from(ctrl, value: int) -> None:
        ts = [outputs[i] for i in _bits(value)]
        if ts:
            b.cx(ctrl, km.array(ts))

    def xor_and(c1, c2, targets: list) -> None:
        """targets ^= c1 & c2 with one Toffoli: fan the first target out to the rest around it."""
        if not targets:
            return
        first, rest = targets[0], targets[1:]
        for t in rest:
            b.cx(first, t)
        b.ccx(c1, c2, first)
        for t in reversed(rest):
            b.cx(first, t)

    # Modified exact loader: prefix Mobius interpolation, with each nonlinear
    # high-address monomial computed once and leaves applied directly to targets.
    coeff = list(table)
    for bit in range(hb):
        for mask in range(H):
            if mask & (1 << bit):
                coeff[mask] ^= coeff[mask ^ (1 << bit)]
    check = list(coeff)
    for bit in range(hb):
        for mask in range(H):
            if mask & (1 << bit):
                check[mask] ^= check[mask ^ (1 << bit)]
    assert check == table
    needed = {m for m, value in enumerate(coeff) if value}
    for mask in tuple(needed):
        while mask:
            mask ^= 1 << (mask.bit_length() - 1)
            if mask:
                needed.add(mask)
    children = {mask: [] for mask in needed if mask}
    for mask in sorted(needed):
        if mask.bit_count() >= 2:
            parent = mask ^ (1 << (mask.bit_length() - 1))
            children[parent].append(mask)
    xor_from(True, coeff[0])

    def visit(mask, control):
        xor_from(control, coeff[mask])
        for child in children.get(mask, ()):
            bit = child.bit_length() - 1
            if not children[child]:
                xor_and(control, addr[bit], [outputs[i] for i in _bits(coeff[child])])
            else:
                temp = b.alloc_qubits(1)[0]
                b.init_and(control, addr[bit], temp)
                visit(child, temp)
                b.del_and(control, addr[bit], temp)
                b.free(temp)
    for bit in range(hb):
        if (1 << bit) in needed:
            visit(1 << bit, addr[bit])

    both = b.alloc_qubits(1)[0]
    b.init_and(sel[0], sel[1], both)
    selection = {'L': both, 0: sel[0], 1: sel[1]}

    def slot_targets(slots) -> list:
        out = []
        for s in slots:
            w, k = divmod(s, BANK_BITS)
            out += [z[BROT * w + k]] + ([z[BROT * w + BROT - 1]] if k == BANK_BITS - 1 else [])
        return out

    for cx, cy, slots, cnot_l in toffolis:
        shifted = [(selection[s], bank[i]) for s, i in (cx, cy) if s is not None and i is not None]
        for s, wire in shifted:  # the control becomes selection + wire
            b.cx(s, wire)
        controls = [bank[i] if i is not None else selection[s] for s, i in (cx, cy)]
        xor_and(controls[0], controls[1], slot_targets(slots))
        for s, wire in shifted:
            b.cx(s, wire)
        for t in slot_targets(cnot_l):
            b.cx(both, t)
    b.del_and(sel[0], sel[1], both)
    b.free(both)

    _erase(b, bank, addr, bank_table)
    b.free(km.array(bank))
    circuit = b.finish_circuit()
    assert circuit.num_qubits <= 128, circuit.num_qubits
    return circuit


def batch_deltas(words: np.ndarray, j: int = J) -> np.ndarray:
    """f_k = batch k-1 XOR batch k of the (A, 75) word table, zero padded to a multiple of j."""
    words = np.asarray(words, dtype=np.int64)
    A, n = words.shape
    padded = np.zeros((A, -(-n // j) * j), dtype=np.int64)
    padded[:, :n] = words
    batches = padded.reshape(A, -1, j).transpose(1, 0, 2)
    out = batches.copy()
    out[1:] ^= batches[:-1]
    return out


def build_family(words: np.ndarray) -> list[km.Circuit]:
    A = len(words)
    n_sel = max(1, (A - 1).bit_length())
    return [build_circuit(d, n_sel) for d in batch_deltas(words)]


def check(circuit: km.Circuit, delta: np.ndarray, repeats: int = 4, seed: int = 0) -> float:
    """Simulate every address with random word contents; return Toffolis per shot."""
    A = len(delta)
    n_sel = max(1, (A - 1).bit_length())
    a = np.tile(np.arange(A), repeats)
    rng = np.random.default_rng(seed)
    z0 = rng.integers(0, 1 << BROT, size=(len(a), J))
    sim = km.Simulator(batch_size=len(a), seed=seed, counting_operations=True)
    sim.use_same_registers_as(circuit)
    sim.write_across_shots('selection', ((a[None, :] >> np.arange(n_sel)[:, None]) & 1).astype(bool))
    for j in range(J):
        sim.write_across_shots(f'word_{j}', ((z0[:, j][None, :] >> np.arange(BROT)[:, None]) & 1).astype(bool))
    sim.do(circuit)
    got = np.stack([(np.int64(1) << np.arange(BROT)) @ sim.read_across_shots(f'word_{j}').astype(np.int64)
                    for j in range(J)], axis=1)
    assert np.array_equal(got, z0 ^ delta[a]), 'wrong XOR'
    assert not np.any(sim.read_phase_flipped_across_shots()), 'uncorrected phase'
    sel = (np.int64(1) << np.arange(n_sel)) @ sim.read_across_shots('selection').astype(np.int64)
    assert np.array_equal(sel, a), 'selection changed'
    return sim.operation_statistics().num_toffoli_like / len(a)
