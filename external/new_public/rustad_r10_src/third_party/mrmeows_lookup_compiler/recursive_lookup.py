"""Exact nested four-address lookup with retained high-address monomials.

The paired-combine identity is adapted from the Apache-2.0 implementation
credited in ../FOUR_ADDRESS_NOTICE.md. This implementation applies it twice,
retains a common high-address basis, and erases intermediate banks by Clifford
phase correction against the retained basis. It does not condition Toffolis.

Targets may hold arbitrary bit strings. Ten-bit signed-gauge words retain all
ten output bits; the two redundant high bits are toggled together.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import kickmix as km


@dataclass
class Plan:
    base: np.ndarray
    bank: np.ndarray
    operations: list


def paired_plan(values: np.ndarray) -> Plan:
    """Compile four-address bit tables into corrected bases and bank controls."""
    values = np.asarray(values, dtype=np.uint8)
    if values.ndim != 2 or len(values) % 4 or np.any(values > 1):
        raise ValueError('paired_plan needs a Boolean (4H,m) table')
    h, slots = len(values) // 4, values.shape[1]
    q = values.reshape(h, 4, slots)
    aa = q[:, 0] ^ q[:, 1]
    bb = q[:, 0] ^ q[:, 2]
    cc = q[:, 0] ^ q[:, 1] ^ q[:, 2] ^ q[:, 3]
    corrected = q[:, 0].copy()
    wires, operations = [], []

    def add(sx, x, sy, y, targets, cnot_l=()):
        cx = (sx, None if x is None else len(wires))
        if x is not None:
            wires.append(x)
        cy = (sy, None if y is None else len(wires))
        if y is not None:
            wires.append(y)
        if x is not None and y is not None:
            for t in targets:
                corrected[:, t] ^= x & y
        operations.append((cx, cy, tuple(targets), tuple(cnot_l)))

    for i in range(0, slots - 1, 2):
        j = i + 1
        add('L', aa[:, i], 0, 1 ^ cc[:, j], (i, j))
        add('L', bb[:, i], 1, 1 ^ cc[:, i] ^ cc[:, j], (i,))
        add(0, bb[:, j], 1, aa[:, i] ^ aa[:, j], (j,), (j,))
    if slots % 2:
        i = slots - 1
        add('L', aa[:, i], 0, 1 ^ cc[:, i], (i,))
        add(1, None, None, bb[:, i], (i,))
    return Plan(corrected, np.stack(wires, axis=1), operations)


def mobius(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.uint8).copy()
    for bit in range((len(out) - 1).bit_length()):
        for mask in range(len(out)):
            if mask & (1 << bit):
                out[mask] ^= out[mask ^ (1 << bit)]
    return out


def xor_product(builder, control1, control2, targets):
    """Toggle every arbitrary target by one product, with one Toffoli."""
    if not targets:
        return
    first, *rest = targets
    for target in rest:
        builder.cx(first, target)
    builder.ccx(control1, control2, first)
    for target in reversed(rest):
        builder.cx(first, target)


def load_linear(builder, values, target_groups, high_basis):
    coefficients = mobius(values)
    for mask, row in enumerate(coefficients):
        targets = [q for j in np.flatnonzero(row) for q in target_groups[int(j)]]
        if targets:
            builder.cx(high_basis[mask], km.array(targets))


def combine(builder, plan, bank, target_groups, low, low_product):
    selection = {0: low[0], 1: low[1], 'L': low_product}
    for left, right, slots, extra_slots in plan.operations:
        shifted = [(selection[s], bank[i]) for s, i in (left, right)
                   if s is not None and i is not None]
        for ctrl, target in shifted:
            builder.cx(ctrl, target)
        controls = [bank[i] if i is not None else selection[s]
                    for s, i in (left, right)]
        xor_product(builder, *controls,
                    [q for slot in slots for q in target_groups[slot]])
        for ctrl, target in reversed(shifted):
            builder.cx(ctrl, target)
        for slot in extra_slots:
            builder.cx(low_product, km.array(target_groups[slot]))


def combine_affine_high(builder, plan, target_groups, high_basis, low, low_product):
    """Apply the combine without materializing any high-only bank qubits."""
    from factor_affine_controls import xor_affine_product
    high_count = len(high_basis) - 1
    basis = [high_basis[m] for m in range(1, len(high_basis))]
    basis += [low[0], low[1], low_product]
    low_indices = {0: high_count, 1: high_count + 1, 'L': high_count + 2}
    coefficients = mobius(plan.bank)
    masks = [sum(1 << (int(m) - 1) for m in np.flatnonzero(coefficients[1:, j]) + 1)
             for j in range(plan.bank.shape[1])]
    constants = [bool(value) for value in coefficients[0]]

    def form(control):
        selector, wire = control
        mask = masks[wire] if wire is not None else 0
        constant = constants[wire] if wire is not None else False
        if selector is not None:
            mask ^= 1 << low_indices[selector]
        return mask, constant

    for left, right, slots, extra_slots in plan.operations:
        a, ca = form(left)
        b, cb = form(right)
        xor_affine_product(builder, basis, a, b,
                           [q for slot in slots for q in target_groups[slot]], ca, cb)
        for slot in extra_slots:
            builder.cx(low_product, km.array(target_groups[slot]))


def erase_functions(builder, bank, values, high_basis, low_basis=None):
    """Erase functions of high monomials, optionally times two-low-bit monomials.

    Every phase correction is Z or CZ, possibly controlled by a measurement.
    It is exact for every measurement record, including constant functions.
    """
    measurements = builder.alloc_bits(len(bank))
    builder.hmr(km.array(bank), measurements)
    coefficients = mobius(values)
    for j, measured in enumerate(measurements):
        masks = np.flatnonzero(coefficients[:, j])
        if not len(masks):
            continue
        builder.push_condition(measured)
        for raw_mask in masks:
            mask = int(raw_mask)
            if low_basis is None:
                builder.z(high_basis[mask])
            else:
                builder.cz(high_basis[mask >> 2], low_basis[mask & 3])
        builder.pop_condition()
    builder.free(measurements)


def build_circuit(delta: np.ndarray, *, inner_mode: str = 'affine') -> Any:
    """Build an exact XOR oracle for D-by-J ten-bit signed-gauge words."""
    if inner_mode not in ('bank', 'affine'):
        raise ValueError('inner_mode must be bank or affine')
    delta = np.asarray(delta, dtype=np.int64)
    if delta.ndim != 2 or not len(delta) or np.any(delta < 0) or np.any(delta >= 1024):
        raise ValueError('invalid ten-bit word table')
    if np.any(((delta >> 9) ^ (delta >> 8)) & 1):
        raise ValueError('exact signed gauge is required')
    count, words_per_batch = delta.shape
    n_sel = (count - 1).bit_length()
    if n_sel < 5:
        raise ValueError('this implementation needs at least five address bits')
    high_rows = (count + 15) // 16
    padded = np.zeros((16 * high_rows, words_per_batch), dtype=np.int64)
    padded[:count] = delta
    bits = np.stack([(padded[:, word] >> bit) & 1
                     for word in range(words_per_batch) for bit in range(9)], axis=1).astype(np.uint8)
    builder = km.CircuitBuilder()
    selection = builder.create_quantum_register(n_sel, 'selection')
    words = [builder.create_quantum_register(10, f'word_{i}')
             for i in range(words_per_batch)]
    output_groups = [[words[word][bit]] + ([words[word][9]] if bit == 8 else [])
                     for word in range(words_per_batch) for bit in range(9)]

    high_basis = {0: True}
    products = []
    for mask in range(1, high_rows):
        bit = mask.bit_length() - 1
        if mask.bit_count() == 1:
            high_basis[mask] = selection[4 + bit]
        else:
            parent = mask ^ (1 << bit)
            target = builder.alloc_qubits(1)[0]
            builder.init_and(high_basis[parent], selection[4 + bit], target)
            high_basis[mask] = target
            products.append((high_basis[parent], selection[4 + bit], target))
    low_products = builder.alloc_qubits(2)
    builder.init_and(selection[0], selection[1], low_products[0])
    builder.init_and(selection[2], selection[3], low_products[1])
    v_basis = {0: True, 1: selection[2], 2: selection[3], 3: low_products[1]}

    def load_inner(values, targets):
        for inner_start in range(0, len(targets), 2):
            inner_groups = targets[inner_start:inner_start + 2]
            inner = paired_plan(values[:, inner_start:inner_start + 2])
            load_linear(builder, inner.base, inner_groups, high_basis)
            if inner_mode == 'affine':
                combine_affine_high(builder, inner, inner_groups, high_basis,
                                    selection[2:4], low_products[1])
            else:
                inner_bank = list(builder.alloc_qubits(inner.bank.shape[1]))
                load_linear(builder, inner.bank, [[q] for q in inner_bank], high_basis)
                combine(builder, inner, inner_bank, inner_groups,
                        selection[2:4], low_products[1])
                erase_functions(builder, inner_bank, inner.bank, high_basis)
                builder.free(km.array(inner_bank))

    for start in range(0, len(output_groups), 2):
        groups = output_groups[start:start + 2]
        outer = paired_plan(bits[:, start:start + 2])
        if len(groups) == 1:
            # Pair the one base with the odd combine's third bank function.
            # This avoids two separate odd inner calls (one extra Toffoli).
            # The last outer product needs only that retained third wire.
            retained = builder.alloc_qubits(1)[0]
            load_inner(np.concatenate((outer.base, outer.bank[:, 2:3]), axis=1),
                       groups + [[retained]])
            first_bank = list(builder.alloc_qubits(2))
            load_inner(outer.bank[:, :2], [[q] for q in first_bank])
            first = Plan(outer.base, outer.bank[:, :2], [outer.operations[0]])
            combine(builder, first, first_bank, groups, selection[:2], low_products[0])
            erase_functions(builder, first_bank, outer.bank[:, :2], high_basis, v_basis)
            builder.free(km.array(first_bank))
            left, right, targets, extras = outer.operations[1]
            mapped = [(selector, None if wire is None else 0)
                      for selector, wire in (left, right)]
            last = Plan(outer.base, outer.bank[:, 2:3],
                        [(mapped[0], mapped[1], targets, extras)])
            combine(builder, last, [retained], groups, selection[:2], low_products[0])
            erase_functions(builder, [retained], outer.bank[:, 2:3], high_basis, v_basis)
            builder.free(retained)
            continue
        load_inner(outer.base, groups)
        # Each bank function is used by only one outer product. Two wires
        # suffice: load a product's controls, use them, erase, and reuse.
        for left, right, targets, cnot_targets in outer.operations:
            wire_indices = [wire for _, wire in (left, right) if wire is not None]
            assert len(wire_indices) == len(set(wire_indices))
            values = outer.bank[:, wire_indices]
            bank = list(builder.alloc_qubits(len(wire_indices)))
            remap = {old: new for new, old in enumerate(wire_indices)}
            mapped = [(selector, None if wire is None else remap[wire])
                      for selector, wire in (left, right)]
            load_inner(values, [[q] for q in bank])
            piece = Plan(outer.base, values, [(mapped[0], mapped[1], targets, cnot_targets)])
            combine(builder, piece, bank, groups, selection[:2], low_products[0])
            erase_functions(builder, bank, values, high_basis, v_basis)
            builder.free(km.array(bank))

    for j in reversed(range(2)):
        builder.del_and(selection[2*j], selection[2*j+1], low_products[j])
    builder.free(low_products)
    for left, right, target in reversed(products):
        builder.del_and(left, right, target)
        builder.free(target)
    return builder.finish_circuit()


def batch_deltas(words: np.ndarray, batch_size: int) -> np.ndarray:
    rows, angles = words.shape
    padded = np.zeros((rows, ((angles + batch_size - 1) // batch_size) * batch_size), dtype=np.int64)
    padded[:, :angles] = words
    batches = padded.reshape(rows, -1, batch_size).transpose(1, 0, 2)
    delta = batches.copy()
    delta[1:] ^= batches[:-1]
    return delta


def check(circuit, delta, repeats=4, seed=92813):
    count, batch_size = delta.shape
    n_sel = (count - 1).bit_length()
    addresses = np.tile(np.arange(count), repeats)
    rng = np.random.default_rng(seed)
    initial = rng.integers(0, 1024, size=(len(addresses), batch_size))
    sim = km.Simulator(batch_size=len(addresses), seed=seed, counting_operations=True)
    sim.use_same_registers_as(circuit)
    sim.write_across_shots('selection', ((addresses[None, :] >> np.arange(n_sel)[:, None]) & 1).astype(bool))
    for j in range(batch_size):
        sim.write_across_shots(f'word_{j}', ((initial[:, j][None, :] >> np.arange(10)[:, None]) & 1).astype(bool))
    powers = np.int64(1) << np.arange(10)
    for application in range(2):
        sim.do(circuit)
        got = np.stack([powers @ sim.read_across_shots(f'word_{j}').astype(np.int64)
                        for j in range(batch_size)], axis=1)
        expected = initial ^ delta[addresses] if application == 0 else initial
        assert np.array_equal(got, expected), 'incorrect XOR oracle'
        assert not np.any(sim.read_phase_flipped_across_shots()), 'incorrect phase'
        decoded = (np.int64(1) << np.arange(n_sel)) @ sim.read_across_shots('selection').astype(np.int64)
        assert np.array_equal(decoded, addresses), 'selection changed'
    return float(sim.operation_statistics().num_toffoli_like) / len(addresses) / 2
