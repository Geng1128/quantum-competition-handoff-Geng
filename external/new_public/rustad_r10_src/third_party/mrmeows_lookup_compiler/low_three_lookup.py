"""Exact eight-address lookup using retained monomials and affine products.

This is a low-three-address specialization.  It uses seven Toffolis for
each pair of independent output bits and four for an unpaired output bit.
All high-address functions are free affine forms in a retained prefix-ANF
basis.  No bank qubits are allocated.  The arbitrary incoming target XOR
contract is preserved; high basis changes are reversed after every product.

The three-product identity for the first three low monomials is adapted
from the Apache-2.0 paired-combine source credited in FOUR_ADDRESS_NOTICE.md.
The remaining four low monomials are handled in two pairs per output.
Measurement-conditioned gates are restricted to Clifford phase correction
inside Kickmix's del_and; no Toffoli is measurement conditioned.
"""
from __future__ import annotations

import numpy as np
import kickmix as km

from factor_affine_controls import xor_affine_product


def mobius(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.uint8).copy()
    for bit in range((len(out) - 1).bit_length()):
        for mask in range(len(out)):
            if mask & (1 << bit):
                out[mask] ^= out[mask ^ (1 << bit)]
    return out


def batch_deltas(words: np.ndarray, batch_size: int) -> np.ndarray:
    rows, angles = words.shape
    padded = np.zeros((rows, ((angles + batch_size - 1) // batch_size) * batch_size),
                      dtype=np.int64)
    padded[:, :angles] = words
    batches = padded.reshape(rows, -1, batch_size).transpose(1, 0, 2)
    delta = batches.copy()
    delta[1:] ^= batches[:-1]
    return delta


def build_circuit(delta: np.ndarray, *, brot: int = 11):
    """Build an exact XOR oracle for signed-gauge words of unchanged precision.

    For D rows and J words, set H=ceil(D/8), m=(brot-1)J.  The universal
    Toffoli bound is H-1-ceil(log2(H)) + 4 + ceil(7m/2).  The width is
    ceil(log2(D)) + brot*J + H-1-ceil(log2(H)) + 4.  Degenerate table
    coefficients can lower the native count; they cannot increase it.
    """
    delta = np.asarray(delta, dtype=np.int64)
    if brot < 3 or brot > 30 or delta.ndim != 2 or not len(delta):
        raise ValueError("invalid precision or word table")
    if np.any(delta < 0) or np.any(delta >= (1 << brot)):
        raise ValueError("angle word outside precision range")
    if np.any(((delta >> (brot - 1)) ^ (delta >> (brot - 2))) & 1):
        raise ValueError("exact signed gauge is required")
    count, word_count = delta.shape
    n_sel = (count - 1).bit_length()
    if n_sel < 4:
        raise ValueError("this implementation requires at least four address bits")
    high_rows = (count + 7) // 8
    padded = np.zeros((8 * high_rows, word_count), dtype=np.int64)
    padded[:count] = delta
    bits = np.stack([(padded[:, word] >> bit) & 1
                     for word in range(word_count) for bit in range(brot - 1)],
                    axis=1).astype(np.uint8)
    # Perform only the three-low-bit Mobius transform.  Each resulting
    # coefficient is a truth table on the high address.
    coefficients = bits.reshape(high_rows, 8, -1).copy()
    for bit in range(3):
        for mask in range(8):
            if mask & (1 << bit):
                coefficients[:, mask] ^= coefficients[:, mask ^ (1 << bit)]

    builder = km.CircuitBuilder()
    selection = builder.create_quantum_register(n_sel, "selection")
    words = [builder.create_quantum_register(brot, f"word_{j}")
             for j in range(word_count)]
    target_groups = [[words[word][bit]] +
                     ([words[word][brot - 1]] if bit == brot - 2 else [])
                     for word in range(word_count) for bit in range(brot - 1)]

    high_basis = {0: True}
    high_products = []
    for mask in range(1, high_rows):
        bit = mask.bit_length() - 1
        if mask.bit_count() == 1:
            high_basis[mask] = selection[3 + bit]
        else:
            parent = mask ^ (1 << bit)
            target = builder.alloc_qubits(1)[0]
            builder.init_and(high_basis[parent], selection[3 + bit], target)
            high_basis[mask] = target
            high_products.append((high_basis[parent], selection[3 + bit], target))
    low_basis = {0: True}
    low_products = []
    for mask in range(1, 8):
        bit = mask.bit_length() - 1
        if mask.bit_count() == 1:
            low_basis[mask] = selection[bit]
        else:
            parent = mask ^ (1 << bit)
            target = builder.alloc_qubits(1)[0]
            builder.init_and(low_basis[parent], selection[bit], target)
            low_basis[mask] = target
            low_products.append((low_basis[parent], selection[bit], target))

    basis = [high_basis[mask] for mask in range(1, high_rows)]
    basis += [low_basis[mask] for mask in range(1, 8)]
    zero = np.zeros(high_rows, dtype=np.uint8)
    corrected = coefficients[:, 0].copy()

    def high_form(values):
        terms = mobius(values)
        mask = sum(1 << (int(k) - 1) for k in np.flatnonzero(terms[1:]) + 1)
        return mask, bool(terms[0])

    def add_product(low1, high1, low2, high2, slots):
        # All high-only products are absorbed into the base truth tables.
        for slot in slots:
            corrected[:, slot] ^= high1 & high2
        mask1, constant1 = high_form(high1)
        mask2, constant2 = high_form(high2)
        if low1:
            mask1 ^= 1 << (high_rows - 1 + low1 - 1)
        if low2:
            mask2 ^= 1 << (high_rows - 1 + low2 - 1)
        xor_affine_product(builder, basis, mask1, mask2,
                           [q for slot in slots for q in target_groups[slot]],
                           constant1, constant2)

    aa, bb, cc = coefficients[:, 1], coefficients[:, 2], coefficients[:, 3]
    slots = len(target_groups)
    for i in range(0, slots - 1, 2):
        j = i + 1
        add_product(3, aa[:, i], 1, 1 ^ cc[:, j], (i, j))
        add_product(3, bb[:, i], 2, 1 ^ cc[:, i] ^ cc[:, j], (i,))
        add_product(1, bb[:, j], 2, aa[:, i] ^ aa[:, j], (j,))
        builder.cx(low_basis[3], km.array(target_groups[j]))
    if slots % 2:
        i = slots - 1
        add_product(3, aa[:, i], 1, 1 ^ cc[:, i], (i,))
        add_product(2, zero, 0, bb[:, i], (i,))

    for slot in range(slots):
        for left, right in ((4, 5), (6, 7)):
            # (M_left + f_right)(M_right + f_left) plus M_union and
            # f_left*f_right equals f_left*M_left + f_right*M_right.
            add_product(left, coefficients[:, right, slot],
                        right, coefficients[:, left, slot], (slot,))
            builder.cx(low_basis[left | right], km.array(target_groups[slot]))

    high_coefficients = mobius(corrected)
    for mask, row in enumerate(high_coefficients):
        targets = [q for slot in np.flatnonzero(row) for q in target_groups[int(slot)]]
        if targets:
            builder.cx(high_basis[mask], km.array(targets))

    for left, right, target in reversed(low_products + high_products):
        # Both sets depend only on their own address group.  Order within
        # each set is reversed so every parent monomial remains available.
        builder.del_and(left, right, target)
        builder.free(target)
    circuit = builder.finish_circuit()
    expected_bound = high_rows - 1 - (high_rows - 1).bit_length() + 4 + (7 * slots + 1) // 2
    if int(circuit.operation_statistics().num_toffoli_like) > expected_bound:
        raise AssertionError("universal Toffoli bound violated")
    return circuit


def check(circuit, delta, *, brot=11, repeats=4, seed=127813):
    """Check every valid address with arbitrary targets, phase and scratch."""
    count, word_count = delta.shape
    n_sel = (count - 1).bit_length()
    addresses = np.tile(np.arange(count), repeats)
    rng = np.random.default_rng(seed)
    initial = rng.integers(0, 1 << brot, size=(len(addresses), word_count))
    if repeats >= 2:
        initial[:count] = 0
        initial[count:2*count] = (1 << brot) - 1
    sim = km.Simulator(batch_size=len(addresses), seed=seed, counting_operations=True)
    sim.use_same_registers_as(circuit)
    sim.write_across_shots("selection", ((addresses[None, :] >> np.arange(n_sel)[:, None]) & 1).astype(bool))
    for j in range(word_count):
        sim.write_across_shots(f"word_{j}", ((initial[:, j][None, :] >> np.arange(brot)[:, None]) & 1).astype(bool))
    powers = np.int64(1) << np.arange(brot)
    for application in range(2):
        sim.do(circuit)
        got = np.stack([powers @ sim.read_across_shots(f"word_{j}").astype(np.int64)
                        for j in range(word_count)], axis=1)
        expected = initial ^ delta[addresses] if application == 0 else initial
        assert np.array_equal(got, expected), "incorrect arbitrary-target XOR"
        assert not np.any(sim.read_phase_flipped_across_shots()), "incorrect phase"
        decoded = (np.int64(1) << np.arange(n_sel)) @ sim.read_across_shots("selection").astype(np.int64)
        assert np.array_equal(decoded, addresses), "selection changed"
        total = sim.read_across_shots(km.array([km.q(i) for i in range(circuit.num_qubits)]))
        assert not np.any(total[n_sel + brot * word_count:]), "scratch not returned to zero"
    return float(sim.operation_statistics().num_toffoli_like) / len(addresses) / 2
