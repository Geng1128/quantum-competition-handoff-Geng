#!/usr/bin/env python3
"""Exact, space-bounded ANGLE compilation for the dfthcsos contract.

This compiles the *existing integer words*, without refitting, requantizing,
changing the tree, or changing any Hamiltonian coefficient.  Its circuit is an
XOR oracle for every incoming target, not only targets encountered in the
benchmark's forward traversal.

Proof of the construction
-------------------------
The address prefix 0 <= a < D is closed under taking bitwise subsets.  The
monomials M_s(a) = product(a_i for i in s), s < D, consequently form a basis
for all Boolean functions on this domain.  The prefix Mobius transform gives
the coefficients, independently for every output bit.  DFS visits each used
nonlinear monomial once.  Each internal node is a temporary AND, erased by
HMR followed by its matching conditional CZ.  This cancels the measurement
phase for *each* outcome.  Leaves may fan a Toffoli directly into an arbitrary
target via CNOT conjugation, eliminating their temporary qubit.

Thus CCX <= D - 1 - ceil(log2 D); all CCX are unconditional.  No promise is
made about unspecified addresses a >= D.  Constant and linear terms use
only X/CX.  The chemistry errors and normalization are unchanged if the
recovered integer words agree.

Examples:
  python compiler.py --submission submission.npz --angle-words words.npy \
      --brot 10 --batch-size 11 --out candidate
  python compiler.py --submission old/submission.npz --source-circuits old \
      --verifier-dir official/verifiers/dfthcsos --out candidate

Requires NumPy and the competition-compatible native Kickmix module.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from math import ceil
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np


SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_words(path: Path) -> np.ndarray:
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.ndarray):
        words = loaded
    else:
        with loaded as archive:
            if "angle_words" not in archive.files:
                raise ValueError(f"{path} does not contain an angle_words array")
            words = archive["angle_words"].copy()
    if words.ndim != 2 or not np.issubdtype(words.dtype, np.integer):
        raise ValueError("angle_words must be a two-dimensional integer array")
    if words.size == 0 or np.any(words < 0):
        raise ValueError("angle_words must be nonempty and nonnegative")
    if np.max(words) > np.iinfo(np.int64).max:
        raise ValueError("angle_words does not fit int64")
    return np.asarray(words, dtype=np.int64)


def submission_shape(path: Path) -> tuple[int, int, int, int]:
    required = {"U", "W", "H2_I", "H1_I", "Hsym_lower", "tree_edges"}
    with np.load(path, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path}: missing submission keys {sorted(missing)}")
        u, w = archive["U"], archive["W"]
        if u.ndim != 3 or w.ndim != 3 or u.shape[:2] != w.shape[:2]:
            raise ValueError("U must have shape (R,B,N) and W shape (R,B,C)")
        r, b, n = map(int, u.shape)
        c = int(w.shape[2])
        if min(r, b, c, n) < 1 or n < 2:
            raise ValueError("invalid factorization dimensions")
        if archive["tree_edges"].shape != (n - 1, 2):
            raise ValueError("tree_edges must have shape (N-1,2)")
        if archive["H2_I"].shape != (r, c):
            raise ValueError("H2_I must have shape (R,C)")
    return r, b, c, n


def load_official_circuits(verifier_dir: Path) -> Any:
    """Load the supplied verifier's circuits.py, without modifying its code."""
    path = verifier_dir.resolve() / "scoreinfra" / "circuits.py"
    if not path.is_file():
        raise FileNotFoundError(f"official circuit-contract module not found: {path}")
    name = "_dfthcsos_contract_" + hashlib.sha256(str(path).encode()).hexdigest()[:12]
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def batch_deltas(words: np.ndarray, batch_size: int) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    d, a = words.shape
    k = ceil(a / batch_size)
    padded = np.zeros((d, k * batch_size), dtype=np.int64)
    padded[:, :a] = words
    batches = padded.reshape(d, k, batch_size).transpose(1, 0, 2)
    deltas = batches.copy()
    deltas[1:] ^= batches[:-1]
    return deltas


def prefix_mobius(table: np.ndarray) -> np.ndarray:
    """A self-inverse transform on the lower ideal {0,...,D-1}."""
    out = table.copy()
    d = len(out)
    indices = np.arange(d, dtype=np.int64)
    for bit in range((d - 1).bit_length()):
        selected = indices[(indices & (1 << bit)) != 0]
        out[selected] ^= out[selected ^ (1 << bit)]
    return out


@dataclass
class Plan:
    coefficients: np.ndarray
    children: dict[int, tuple[int, ...]]
    used: set[int]
    scratch_qubits: int
    nonlinear_nodes: int
    direct_leaves: bool


def plan_prefix(delta: np.ndarray, *, direct_leaves: bool = True) -> Plan:
    coefficients = prefix_mobius(delta)
    # This verifies the actual finite interpolation, not a float approximation.
    if not np.array_equal(prefix_mobius(coefficients), delta):
        raise AssertionError("prefix Mobius round trip failed")
    d = len(delta)
    used = set(int(x) for x in np.flatnonzero(np.any(coefficients != 0, axis=1)))
    # Every nonzero monomial needs its prefix ancestors as intermediate ANDs.
    for mask in tuple(used):
        while mask:
            mask ^= 1 << (mask.bit_length() - 1)
            if mask:
                used.add(mask)
    child_lists: dict[int, list[int]] = {s: [] for s in used if s}
    for mask in sorted(used):
        if mask.bit_count() >= 2:
            parent = mask ^ (1 << (mask.bit_length() - 1))
            child_lists[parent].append(mask)
    children = {s: tuple(v) for s, v in child_lists.items()}
    nonlinear = [s for s in used if s.bit_count() >= 2]
    stored = [s for s in nonlinear if not direct_leaves or children[s]]
    scratch = max((s.bit_count() - 1 for s in stored), default=0)
    return Plan(coefficients, children, used, scratch, len(nonlinear), direct_leaves)


def build_angle(delta: np.ndarray, brot: int, *, direct_leaves: bool = True) -> tuple[Any, dict]:
    import kickmix as km

    if not 1 <= brot <= 62:
        raise ValueError("brot must be between 1 and 62")
    if delta.ndim != 2 or len(delta) < 2:
        raise ValueError("delta must have shape (D,J) with D >= 2")
    if np.any(delta < 0) or np.any(delta >= (1 << brot)):
        raise ValueError(f"angle words must lie in [0,2**{brot})")
    d, j = map(int, delta.shape)
    n = (d - 1).bit_length()
    plan = plan_prefix(delta, direct_leaves=direct_leaves)
    width = n + j * brot + plan.scratch_qubits
    if width > 128:
        raise ValueError(f"ANGLE requires {width} qubits, exceeding 128 (address={n}, words={j*brot}, scratch={plan.scratch_qubits})")

    builder = km.CircuitBuilder()
    address = builder.create_quantum_register(n, "selection")
    word_regs = [builder.create_quantum_register(brot, f"word_{k}") for k in range(j)]
    scratch = builder.alloc_qubits(plan.scratch_qubits) if plan.scratch_qubits else []

    def targets(mask: int) -> list[Any]:
        result = []
        for column, word in enumerate(plan.coefficients[mask]):
            bits = int(word)
            while bits:
                low = bits & -bits
                result.append(word_regs[column][low.bit_length() - 1])
                bits ^= low
        return result

    for target in targets(0):
        builder.x(target)

    def visit(mask: int, control: Any) -> None:
        for target in targets(mask):
            builder.cx(control, target)
        for child in plan.children.get(mask, ()):
            bit = child.bit_length() - 1
            child_targets = targets(child)
            if direct_leaves and not plan.children[child]:
                # L^-1 CCX L fans one product into an arbitrary dirty target
                # vector.  Every bit of the initial target is restored except
                # for the requested XOR of the product.
                pivot, *rest = child_targets
                for target in rest:
                    builder.cx(pivot, target)
                builder.ccx(control, address[bit], pivot)
                for target in rest:
                    builder.cx(pivot, target)
            else:
                temp = scratch[child.bit_count() - 2]
                builder.init_and(control, address[bit], temp)
                visit(child, temp)
                builder.del_and(control, address[bit], temp)

    for bit in range(n):
        mask = 1 << bit
        if mask in plan.used:
            visit(mask, address[bit])
    circuit = builder.finish_circuit()
    native_count = int(circuit.operation_statistics().num_toffoli_like)
    if native_count != plan.nonlinear_nodes:
        raise AssertionError(f"native count {native_count} != planned {plan.nonlinear_nodes}")
    if int(circuit.num_qubits) > 128:
        raise AssertionError("native circuit exceeds the qubit cap")
    return circuit, {
        "num_qubits": int(circuit.num_qubits),
        "selection_qubits": n,
        "word_qubits": j * brot,
        "scratch_qubits": plan.scratch_qubits,
        "toffoli_count": native_count,
        "num_instructions": len(circuit),
        "nonlinear_nodes_before_pruning": d - 1 - n,
        "direct_target_leaves": direct_leaves,
        "mobius_round_trip": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--submission", type=Path, required=True, help="factorization NPZ; copied byte for byte to the new output directory")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--angle-words", type=Path, help="explicit .npy table or .npz containing angle_words")
    source.add_argument("--source-circuits", type=Path, help="existing ANGLE family to recover through the official verifier")
    parser.add_argument("--verifier-dir", type=Path, help="official dfthcsos directory containing scoreinfra/circuits.py")
    parser.add_argument("--brot", type=int, help="required for --angle-words; must match recovered precision for --source-circuits")
    parser.add_argument("--batch-size", type=int, help="words per ANGLE; required for --angle-words, otherwise defaults to source layout")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--format", choices=("kmx", "kmb"), default="kmb")
    parser.add_argument("--no-direct-leaves", action="store_true", help="use a clean temporary AND even for leaf monomials")
    parser.add_argument("--overwrite", action="store_true", help="replace generated files in an existing distinct output directory")
    args = parser.parse_args()
    r, b, c, n = submission_shape(args.submission)
    d = n + r * b
    source_details: dict[str, Any]
    if args.source_circuits is not None:
        if args.verifier_dir is None:
            parser.error("--source-circuits requires --verifier-dir")
        official = load_official_circuits(args.verifier_dir)
        source_family = official.load_circuits(args.source_circuits)
        words, layout, old_count = official.recover_words_from_circuits(
            source_family, num_vectors=d, num_angles=n - 1,
            shots_per_address=64, seed=317849,
        )
        if args.brot is not None and args.brot != layout.brot:
            parser.error("this exact compiler does not change recovered word precision")
        brot = layout.brot
        batch_size = args.batch_size if args.batch_size is not None else layout.batch_size
        source_details = {
            "kind": "recovered_circuits", "path": str(args.source_circuits.resolve()),
            "source_brot": layout.brot, "source_batch_size": layout.batch_size,
            "source_angle_toffoli_forward_and_reverse": float(old_count),
            "recovery_shots_per_address": 64,
            "verifier_circuits_sha256": sha256_file(args.verifier_dir / "scoreinfra" / "circuits.py"),
        }
    else:
        if args.brot is None or args.batch_size is None:
            parser.error("--angle-words requires both --brot and --batch-size")
        words = load_words(args.angle_words)
        brot, batch_size = args.brot, args.batch_size
        source_details = {"kind": "explicit_words", "path": str(args.angle_words.resolve()), "sha256": sha256_file(args.angle_words)}
    if words.shape != (d, n - 1):
        raise ValueError(f"word shape {words.shape} does not match factorization {(d,n-1)}")
    if not 1 <= brot <= 62 or np.any(words >= (1 << brot)):
        raise ValueError("angle words are outside the specified precision range")
    out = args.out.resolve()
    protected = {args.submission.resolve().parent}
    if args.source_circuits is not None:
        protected.add(args.source_circuits.resolve())
    if out in protected:
        raise ValueError("output must be distinct from the source submission/circuit directory")
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{out} is nonempty; use a new directory or --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    # Build every circuit before changing any generated family on disk.
    built = [build_angle(delta, brot, direct_leaves=not args.no_direct_leaves)
             for delta in batch_deltas(words, batch_size)]
    reports = []
    for k, (circuit, detail) in enumerate(built):
        path = out / f"angle_k_{k}.{args.format}"
        circuit.to_file(path, format=args.format)
        reports.append({"k": k, **detail, "file": path.name, "sha256": sha256_file(path)})
    if args.overwrite:
        keep = {row["file"] for row in reports}
        for path in out.glob("angle_k_*.*"):
            if path.suffix in (".kmx", ".kmb") and path.name not in keep:
                path.unlink()
    shutil.copyfile(args.submission, out / "submission.npz")
    np.save(out / "angle_words.npy", words, allow_pickle=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "method": "exact_prefix_anf",
        "factorization_shape": {"R": r, "B": b, "C": c, "N": n},
        "num_vectors": d, "num_angles": n - 1,
        "brot": brot, "batch_size": batch_size, "num_circuits": len(built),
        "source": source_details,
        "submission_sha256": sha256_file(out / "submission.npz"),
        "submission_copied_byte_for_byte": sha256_file(args.submission) == sha256_file(out / "submission.npz"),
        "angle_words_sha256": sha256_file(out / "angle_words.npy"),
        "angle_toffoli_forward_and_reverse": 2 * sum(row["toffoli_count"] for row in reports),
        "max_angle_qubits": max(row["num_qubits"] for row in reports),
        "circuits": reports,
        "chemistry_status": "not evaluated by compiler; run the unchanged official verifier",
        "lookup_status": "algebraic construction and interpolation checked; run validate_lookup.py for native simulation",
    }
    (out / "compile_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
