"""Truncated ANF bank loading and split ANF erasure for exact ANGLES XOR.

The contiguous prefix of1988 high addresses uses1976 nonlinear monomials
versus1986 unary products. All legal rows are certified unchanged. A precise
polynomial extension defines the other60 rows and is used for cleanup too.

Own follow-up: eleven-bit measurement correction uses 83 Toffoli instead of88,
using a split Boolean algebraic normal form and measurement-erased products.

SPDX-License-Identifier: Apache-2.0

Banked QROM construction adapted from Kozykorpesh / Sumit Chongder:
https://www.kaggle.com/code/kozykappa/quantum-ai-compact-arithmetic-banked-qrom?scriptVersionId=361433617
https://www.kaggle.com/code/sumitchongder/notebook2f196e646b?scriptVersionId=361357149
Read through Kirill Tym's public quantum-circuits-qrom-tower42 v10 source.

Our change: allocate only nonzero difference columns. Optionally conjugate the
target by a reversible CNOT change of basis, reducing the difference bank to
its GF(2) column rank. The same change is applied to BOTH lookup tables. Thus
the input target can be arbitrary: T^-1(T z XOR T f(a)) = z XOR f(a).
No coefficient, angle word, address, or measurement branch is approximated.

Numeric/profile operations work without kickmix. Circuit building/testing
requires the competition's Linux kickmix package.
"""
from __future__ import annotations
from math import ceil
import numpy as np


def batch_deltas(words, J=5):
    words = np.asarray(words, dtype=np.int64)
    nv, na = words.shape
    padded = np.zeros((nv, ceil(na / J) * J), dtype=np.int64)
    padded[:, :na] = words
    batches = padded.reshape(nv, -1, J).transpose(1, 0, 2)
    result = batches.copy()
    result[1:] ^= batches[:-1]
    return result


def pack(row, brot):
    return sum(int(value) << (j * brot) for j, value in enumerate(row))


def _change_basis(values, operations):
    result = list(values)
    for source, target in operations:
        result = [value ^ (((value >> source) & 1) << target)
                  for value in result]
    return result


def difference_tables(delta, brot=10, mode="zero"):
    """Return packed d0, compact dd, and exact reversible output transforms."""
    if mode not in {"baseline", "zero", "linear"}:
        raise ValueError("mode must be baseline, zero, or linear")
    delta = np.asarray(delta, dtype=np.int64)
    if delta.ndim != 2 or len(delta) < 4:
        raise ValueError("expected at least four address rows")
    if np.any(delta < 0) or np.any(delta >= (1 << brot)):
        raise ValueError("angle word out of range")
    A, J = delta.shape
    width = J * brot
    d0 = [pack(row, brot) for row in delta[::2]]
    d1 = [pack(delta[2 * h + 1], brot) if 2 * h + 1 < A else value
          for h, value in enumerate(d0)]
    dd = [a ^ b for a, b in zip(d0, d1)]
    paired = delta[::2].copy()
    paired[:len(delta[1::2])] ^= delta[1::2]
    if A & 1:
        paired[-1] = 0
    bit_columns = ((paired[:, :, None] >> np.arange(brot)) & 1).reshape(H := len(d0), width).T
    packed_columns = np.packbits(bit_columns.astype(np.uint8), axis=1, bitorder="little")
    columns = [int.from_bytes(column.tobytes(), "little") for column in packed_columns]
    active = [bit for bit, column in enumerate(columns) if column]
    unique = len(set(columns) - {0})
    basis = {}
    operations = []
    reduced = columns.copy()
    for bit, original in enumerate(columns):
        column = original
        relation = 0
        while column:
            pivot = column.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = (column, relation ^ (1 << bit))
                break
            value, combination = basis[pivot]
            column ^= value
            relation ^= combination
        if not column:
            # Independent coordinates retain their original values. A dependent
            # coordinate is zeroed by XORing only independent coordinates into it.
            # In particular, identical columns need just one CNOT each way.
            operations.extend((source, bit) for source in range(width)
                              if (relation >> source) & 1)
            reduced[bit] = 0
    rank = len(basis)
    if mode == "linear":
        d0 = _change_basis(d0, operations)
        transformed_dd = _change_basis(dd, operations)
        active = [bit for bit, column in enumerate(reduced) if column]
    else:
        transformed_dd = dd
        operations = []
        if mode == "baseline":
            active = list(range(width))
    compact = [sum(((value >> bit) & 1) << target
                   for target, bit in enumerate(active))
               for value in transformed_dd]
    # A complete integer certificate of the table identity for every legal a.
    reconstructed = [sum(((value >> target) & 1) << bit
                         for target, bit in enumerate(active)) for value in compact]
    restored = _change_basis(
        [d0[a // 2] ^ (reconstructed[a // 2] if a & 1 else 0)
         for a in range(A)], reversed(operations))
    assert restored == [pack(row, brot) for row in delta]
    return d0, compact, active, operations, {
        "rows": A, "width": width, "active_columns": sum(bool(c) for c in columns),
        "unique_nonzero_columns": unique, "gf2_rank": rank,
        "bank_width": len(active), "basis_cnot_count": 2 * len(operations),
        "integer_table_certificate": "PASS", "mode": mode,
    }


def mobius_transform(table):
    """Packed-output Boolean truth table <-> ANF coefficient table over GF(2)."""
    result = list(map(int, table))
    if not result or len(result) & (len(result) - 1):
        raise ValueError("truth-table length must be a power of two")
    n = (len(result) - 1).bit_length()
    for bit in range(n):
        step = 1 << bit
        for start in range(0, len(result), 2 * step):
            for offset in range(step):
                result[start + step + offset] ^= result[start + offset]
    return result


def del_lookup_anf(builder, *, table, address, target, low_bits=None):
    """Clear table[address] with exact conditional ANF phase correction.

    `target` must contain the given lookup value and remain allocated. Extra
    scratch is allocated only when the reset target is too narrow. Clean target
    wires are reused internally; all target and extra scratch finish at zero.
    """
    import kickmix as km
    address = list(address)
    target = list(target)
    n = len(address)
    if len(table) != 1 << n:
        raise ValueError("lookup table must cover the complete address space")
    if n < 2:
        raise ValueError("split ANF cleanup expects at least two address bits")
    low_bits = n // 2 if low_bits is None else low_bits
    if not 0 < low_bits < n:
        raise ValueError("both address halves must be nonempty")
    if any(value < 0 or value >= (1 << len(target)) for value in table):
        raise ValueError("lookup value does not fit the target")
    coefficients = mobius_transform(table)
    assert mobius_transform(coefficients) == list(table), "exact ANF identity failed"
    measurements = builder.alloc_bits(len(target))
    builder.hmr(km.array(target), measurements)
    parity = builder.alloc_bits(1)[0]

    low_count = 1 << low_bits
    high_bits = n - low_bits
    retained_masks = [mask for mask in range(1, low_count) if mask.bit_count() > 1]
    scratch_count = len(retained_masks) + high_bits - 1
    extra = builder.alloc_qubits(max(0, scratch_count - len(target)))
    scratch = target + list(extra)
    low = {1 << bit: address[bit] for bit in range(low_bits)}
    for mask, qubit in zip(retained_masks, scratch):
        bit = mask.bit_length() - 1
        parent = mask ^ (1 << bit)
        builder.init_and(low[parent], address[bit], qubit)
        low[mask] = qubit
    carries = scratch[len(retained_masks):scratch_count]

    def correct(high_mask, high_qubit):
        for low_mask in range(low_count):
            coefficient = coefficients[low_mask | (high_mask << low_bits)]
            if not coefficient:
                continue
            builder.bit_store(parity, False)
            while coefficient:
                bit = (coefficient & -coefficient).bit_length() - 1
                builder.cx(measurements[bit], parity)
                coefficient &= coefficient - 1
            if not low_mask and not high_mask:
                builder.neg_if(parity)
                continue
            builder.push_condition(parity)
            if not low_mask:
                builder.z(high_qubit)
            elif not high_mask:
                builder.z(low[low_mask])
            else:
                builder.cz(low[low_mask], high_qubit)
            builder.pop_condition()

    def walk(mask, highest_bit, control):
        correct(mask, control)
        for bit in range(highest_bit + 1, high_bits):
            child_mask = mask | (1 << bit)
            if not mask:
                walk(child_mask, bit, address[low_bits + bit])
            else:
                child = carries[mask.bit_count() - 1]
                builder.init_and(control, address[low_bits + bit], child)
                walk(child_mask, bit, child)
                builder.del_and(control, address[low_bits + bit], child)

    walk(0, -1, None)
    for mask in reversed(retained_masks):
        bit = mask.bit_length() - 1
        parent = mask ^ (1 << bit)
        builder.del_and(low[parent], address[bit], low[mask])
    builder.free(extra)
    return {"address_bits": n, "low_bits": low_bits, "high_bits": high_bits,
            "retained_products": len(retained_masks), "traversal_carries": high_bits - 1,
            "anf_toffoli": (1 << low_bits) - low_bits - 1 + (1 << high_bits) - high_bits - 1,
            "scratch_count": scratch_count, "anf_table_identity": "PASS"}


def init_banked_truncated_anf(builder, *, d0, dd, address, target, bank, scratch):
    """XOR d0[h] into target, dd[h] into bank; return the completed dd table.

    Requires `mobius_transform` in the surrounding module (anf_cleanup.py).
    Legal h ranges from zero through len(d0)-1. `scratch` is clean and remains
    clean. All target bits may start arbitrary; bank initialization is the
    caller's usual clean lookup bank.
    """
    import kickmix as km
    address, target, bank, scratch = map(list, (address, target, bank, scratch))
    H, n, W = len(d0), len(address), len(target)
    if len(dd) != H or H > (1 << n) or H <= (1 << (n - 1)):
        raise ValueError("expected a nonempty prefix using every address bit")
    if len(scratch) < n - 1:
        raise ValueError("need n-1 clean traversal carry wires")
    original = [int(first) | (int(second) << W) for first, second in zip(d0, dd)]
    coefficients = mobius_transform(original + [0] * ((1 << n) - H))
    coefficients[H:] = [0] * ((1 << n) - H)
    completed = mobius_transform(coefficients)
    assert completed[:H] == original, "truncated ANF changed a legal table row"
    assert mobius_transform(completed) == coefficients
    outputs = target + bank

    def load(mask, control):
        value = coefficients[mask]
        targets = [outputs[bit] for bit in range(len(outputs)) if (value >> bit) & 1]
        if targets:
            if control is None:
                builder.x(km.array(targets))
            else:
                builder.cx(control, km.array(targets))

    def walk(mask, highest_bit, control):
        load(mask, control)
        for bit in range(highest_bit + 1, n):
            child_mask = mask | (1 << bit)
            if child_mask >= H:
                continue
            if not mask:
                walk(child_mask, bit, address[bit])
            else:
                child = scratch[mask.bit_count() - 1]
                builder.init_and(control, address[bit], child)
                walk(child_mask, bit, child)
                builder.del_and(control, address[bit], child)

    walk(0, -1, None)
    return [value >> W for value in completed], {
        "legal_high_addresses": H, "address_bits": n,
        "nonlinear_products": H - 1 - n,
        "previous_unary_products": H - 2,
        "exact_legal_table_identity": "PASS",
        "extension_mobius_identity": "PASS",
        "completed_high_addresses": (1 << n) - H,
        "max_required_carries": n - 1,
    }


def build_qroam2(delta, brot=10, *, mode="zero", return_report=False):
    """Exact banked XOR oracle; same public register contract as upstream."""
    import kickmix as km
    delta = np.asarray(delta, dtype=np.int64)
    A, J = delta.shape
    nbits = (A - 1).bit_length()
    H = (A + 1) // 2
    hbits = nbits - 1
    W = J * brot
    d0, dd, active, transforms, report = difference_tables(delta, brot, mode)
    b = km.CircuitBuilder()
    sel = b.create_quantum_register(nbits, "selection")
    words = [b.create_quantum_register(brot, f"word_{j}") for j in range(J)]
    tgt = [q for word in words for q in word]
    for source, target in transforms:
        b.cx(tgt[source], tgt[target])
    B = b.alloc_qubits(len(active))
    hreg = sel[1:nbits]
    anc = b.alloc_qubits(hbits - 1)
    completed_dd, load_report = init_banked_truncated_anf(
        b, d0=d0, dd=dd, address=hreg, target=tgt, bank=B, scratch=anc)
    report["anf_bank_load"] = load_report
    b.free(anc)
    for bank_bit, target in enumerate(active):
        b.ccx(sel[0], B[bank_bit], tgt[target])
    if active:
        table = completed_dd
        report["anf_cleanup"] = del_lookup_anf(b, table=table, address=hreg, target=B)
    b.free(B)
    for source, target in reversed(transforms):
        b.cx(tgt[source], tgt[target])
    circuit = b.finish_circuit()
    report.update(num_qubits=circuit.num_qubits,
                  static_toffoli=int(circuit.operation_statistics().num_toffoli_like))
    return (circuit, report) if return_report else circuit


def test_random_targets(circuits, deltas, *, brot=10, repeats=8, seed=20261023):
    """Every legal address, random dirty targets, HMR phase and scratch checks.

    This is simulation evidence; the exhaustive integer identity is separately
    certified above. This test is not a proof over all measurement branches.
    The full official verifier remains the acceptance check for the family.
    """
    import kickmix as km
    from dfthcsos.scoreinfra.circuits import register_layout, _check_state
    rng = np.random.default_rng(seed)
    layout = register_layout(circuits)
    A = len(deltas[0])
    addresses = np.tile(np.arange(A, dtype=np.int64), repeats)
    address_bits = ((addresses[None, :] >> np.arange(len(layout.selection))[:, None]) & 1).astype(bool)
    sim = km.Simulator(batch_size=len(addresses), debug_print=False,
                       seed=seed, counting_operations=True)
    sim.use_same_registers_as(circuits[0])
    place_values = np.int64(1) << np.arange(brot)
    toffolis = []
    for k, (circuit, delta) in enumerate(zip(circuits, deltas)):
        sim.clear_for_shot()
        sim.write_across_shots(layout.selection, address_bits)
        dirty = rng.integers(0, 1 << brot, size=(len(addresses), len(layout.words)), dtype=np.int64)
        for j, word in enumerate(layout.words):
            sim.write_across_shots(word, ((dirty[:, j][None, :] >> np.arange(brot)[:, None]) & 1).astype(bool))
        before = sim.operation_statistics().num_toffoli_like
        sim.do(circuit)
        _check_state(sim, circuit, layout, address_bits, f"random-target ANGLE_{k}")
        result = np.stack([place_values @ sim.read_across_shots(word).astype(np.int64)
                           for word in layout.words], axis=1)
        assert np.array_equal(result, dirty ^ delta[addresses]), f"ANGLE_{k} wrong XOR"
        toffolis.append((sim.operation_statistics().num_toffoli_like - before) / len(addresses))
    return {"status": "PASS", "addresses": A, "repeats": repeats,
            "shots_per_circuit": len(addresses), "num_circuits": len(circuits),
            "random_target_xor": "PASS", "unchanged_selection": "PASS",
            "clean_workspace": "PASS", "zero_observed_phase_errors": True,
            "toffoli_per_circuit": toffolis, "toffoli_per_pass": sum(toffolis),
            "full_official_verification": "PENDING", "seed": seed}

