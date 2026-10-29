"""One-Toffoli products of two affine forms, with no allocated workspace.

Given disjoint basis wires x and arbitrary target wires z, this emits

    |x>|z> -> |x>|z XOR (a(x) AND b(x))>,

with the product XORed into every target. Basis wires may be arbitrary,
correlated, or entangled with other registers. They need not be zero or
independent classical functions. Only reversible X/CNOT basis changes and
one CCX are used; all basis changes are undone. Dependent affine forms use
only Clifford gates.
"""
from __future__ import annotations

from typing import Any, Iterable


def _indices(mask: int):
    while mask:
        low = mask & -mask
        yield low.bit_length()-1
        mask ^= low


def xor_affine_product(builder: Any, basis_qubits: Iterable,
                       mask1: int, mask2: int, target_qubits: Iterable,
                       const1: bool = False, const2: bool = False) -> int:
    """Emit an affine-product XOR and return its unconditional Toffoli count.

    ``mask1`` and ``mask2`` index ``basis_qubits``. Constants are XORed into
    the forms. No targets may overlap the basis, and wire lists must not
    contain duplicates. Zero targets are a no-op. The return value is 0
    for dependent/constant forms and 1 for two independent linear parts.

    Proof: for independent masks, conjugate by a reversible CNOT network
    sending the two linear forms to distinct pivot wires, add their
    affine constants, apply CCX, then undo the conjugation. Multi-target
    fanout before and after the CCX cancels arbitrary initial target bits.
    For equal masks, the two forms are equal (product equals the form) or
    complementary (product is zero). Constant forms reduce immediately.
    """
    basis, targets = list(basis_qubits), list(target_qubits)
    if not isinstance(mask1, int) or not isinstance(mask2, int):
        raise TypeError("affine masks must be Python integers")
    if min(mask1, mask2) < 0 or (mask1 | mask2) >> len(basis):
        raise ValueError("affine mask lies outside the supplied basis")
    if len(set(basis)) != len(basis) or len(set(targets)) != len(targets):
        raise ValueError("basis and target wire lists must contain distinct wires")
    if set(basis).intersection(targets):
        raise ValueError("affine basis wires must be disjoint from all targets")
    if not targets:
        return 0
    c1, c2 = bool(const1), bool(const2)

    def xor_form(mask: int, constant: bool) -> None:
        for target in targets:
            if constant:
                builder.x(target)
            for i in _indices(mask):
                builder.cx(basis[i], target)

    if not mask1:
        if c1:
            xor_form(mask2, c2)
        return 0
    if not mask2:
        if c2:
            xor_form(mask1, c1)
        return 0
    if mask1 == mask2:
        if c1 == c2:
            xor_form(mask1, c1)
        return 0

    # First materialize mask1 in its least-significant pivot. Express mask2
    # in the resulting basis before selecting its distinct second pivot.
    pbit = mask1 & -mask1
    p = pbit.bit_length()-1
    first_controls = list(_indices(mask1 ^ pbit))
    second_mask = mask2 ^ ((mask1 ^ pbit) if mask2 & pbit else 0)
    remaining = second_mask & ~pbit
    if not remaining:
        raise AssertionError("independent masks did not leave a second pivot")
    qbit = remaining & -remaining
    q = qbit.bit_length()-1
    second_controls = list(_indices(second_mask ^ qbit))

    for i in first_controls:
        builder.cx(basis[i], basis[p])
    for i in second_controls:
        builder.cx(basis[i], basis[q])
    if c1:
        builder.x(basis[p])
    if c2:
        builder.x(basis[q])

    first, rest = targets[0], targets[1:]
    for target in rest:
        builder.cx(first, target)
    builder.ccx(basis[p], basis[q], first)
    for target in reversed(rest):
        builder.cx(first, target)

    if c2:
        builder.x(basis[q])
    if c1:
        builder.x(basis[p])
    for i in reversed(second_controls):
        builder.cx(basis[i], basis[q])
    for i in reversed(first_controls):
        builder.cx(basis[i], basis[p])
    return 1
