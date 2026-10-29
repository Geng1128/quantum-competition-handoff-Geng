#!/usr/bin/env python3
"""Exact four-address compilation with a prefix-ANF bank loader.

For D=876 ten-bit words, J=3, every circuit costs at most 275 Toffolis
and 126 qubits.  Twenty-five circuits load the same 75 tree angles.
The only optional word change is an exact INTEGER projective gauge:
negation and addition of pi, with no rounding or precision change.

Run this against actual private words or recover them from actual circuits:

  python compile_four_address.py --submission old/submission.npz \
    --source-circuits old --verifier-dir official/verifiers/dfthcsos \
    --out improved

It does not claim chemical acceptance. Run run_verifier.py on its output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from compiler import load_words, load_official_circuits, submission_shape, sha256_file


def decode_tree(words: np.ndarray, edges: np.ndarray, brot: int) -> np.ndarray:
    n = len(edges) + 1
    result = np.zeros((len(words), n), dtype=np.float64)
    result[:, 0] = 1
    for t, (p, q) in enumerate(edges):
        theta = words[:, t] * (np.pi / (1 << (brot - 1)))
        cp, cq = result[:, p].copy(), result[:, q].copy()
        co, si = np.cos(theta), np.sin(theta)
        result[:, p] = co * cp - si * cq
        result[:, q] = si * cp + co * cq
    return result


def validate_tree(edges: np.ndarray) -> None:
    n = len(edges) + 1
    seen = {0}
    if edges.shape != (n - 1, 2) or not np.issubdtype(edges.dtype, np.integer):
        raise ValueError("tree_edges must be an integer (N-1,2) array")
    for p, q in edges:
        if int(p) not in seen or int(q) in seen or not 0 <= int(q) < n:
            raise ValueError("tree must introduce each new child once in parent-before-child order, rooted at 0")
        seen.add(int(q))


def signed_gauge(words: np.ndarray, edges: np.ndarray, brot: int) -> tuple[np.ndarray, np.ndarray]:
    """Return signed-gauge words and their exact overall vector signs.

    In reverse tree order, preserve a diagonal sign map between original
    and transformed partial vectors.  Negation flips sine; adding pi flips
    both sine and cosine.  The predecessor child's amplitude is zero, so
    its predecessor sign is irrelevant.  At the end only the root sign
    remains.  Thus new_u = root_sign * original_u exactly, over the reals.

    The half-open signed interval includes -pi/2 and excludes +pi/2.
    In particular, this handles quarter-turn and zero-component cases
    with integers; there is no atan2, clipping, or numerical rerounding.
    """
    validate_tree(edges)
    if brot < 2 or words.shape[1] != len(edges):
        raise ValueError("invalid precision or tree/word dimensions")
    modulus = 1 << brot
    if np.any(words < 0) or np.any(words >= modulus):
        raise ValueError("words outside precision range")
    half, quarter = modulus // 2, modulus // 4
    out = np.asarray(words, dtype=np.int64).copy()
    signs = np.ones((len(words), len(edges) + 1), dtype=np.int8)
    for t in reversed(range(len(edges))):
        p, q = edges[t]
        angle = out[:, t]
        angle = np.where(signs[:, p] != signs[:, q], (-angle) % modulus, angle)
        flip_parent = (angle >= quarter) & (angle < 3 * quarter)
        angle = np.where(flip_parent, (angle + half) % modulus, angle)
        out[:, t] = angle
        signs[flip_parent, p] *= -1
        signs[:, q] = 1
    if np.any(((out >> (brot - 1)) ^ (out >> (brot - 2))) & 1):
        raise AssertionError("signed-gauge invariant failed")
    return out, signs[:, 0].copy()


def build_family(words: np.ndarray, brot: int = 10, batch_size: int = 3):
    """Actual native circuits; conservative defaults target the supplied c29b shape."""
    import four_address as four
    if brot != 10 or batch_size != 3:
        raise ValueError("this implementation is validated for brot=10,J=3 only")
    if len(words) % 4:
        raise ValueError("four-address backend currently requires D divisible by four")
    if np.any(((words >> 9) ^ (words >> 8)) & 1):
        raise ValueError("backend requires exact signed-gauge words")
    n_sel = (len(words) - 1).bit_length()
    family = [four.build_circuit(delta, n_sel) for delta in four.batch_deltas(words)]
    if any(c.num_qubits > 128 for c in family):
        raise AssertionError("circuit exceeds the 128-qubit cap")
    return family


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--submission", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--angle-words", type=Path)
    source.add_argument("--source-circuits", type=Path)
    parser.add_argument("--verifier-dir", type=Path)
    parser.add_argument("--brot", type=int, help="required with explicit words; actual recovered precision is checked")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--format", choices=("kmb", "kmx"), default="kmb")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    r, b, c, n = submission_shape(args.submission)
    d = n + r * b
    with np.load(args.submission, allow_pickle=False) as archive:
        edges = archive["tree_edges"].copy()
    if args.source_circuits is not None:
        if args.verifier_dir is None:
            parser.error("source recovery requires --verifier-dir")
        contract = load_official_circuits(args.verifier_dir)
        old = contract.load_circuits(args.source_circuits)
        original, layout, old_t = contract.recover_words_from_circuits(
            old, num_vectors=d, num_angles=n - 1, shots_per_address=64, seed=317849,
        )
        brot = layout.brot
        if args.brot is not None and args.brot != brot:
            parser.error("requested precision differs from recovered precision")
        source_info = {"kind": "recovered_circuits", "path": str(args.source_circuits.resolve()),
                       "original_J": layout.batch_size, "original_angle_toffoli": float(old_t)}
    else:
        if args.brot is None:
            parser.error("explicit words require --brot")
        original, brot = load_words(args.angle_words), args.brot
        source_info = {"kind": "explicit_words", "path": str(args.angle_words.resolve()),
                       "sha256": sha256_file(args.angle_words)}
    if brot != 10:
        raise ValueError("this four-address implementation requires actual ten-bit angle words; it will not requantize")
    if original.shape != (d, n - 1):
        raise ValueError(f"word shape {original.shape} differs from {(d,n-1)}")
    converted, signs = signed_gauge(original, edges, brot)
    old_u, new_u = decode_tree(original, edges, brot), decode_tree(converted, edges, brot)
    vector_residual = float(np.max(np.abs(new_u - signs[:, None] * old_u)))
    if vector_residual > 1e-12:
        raise AssertionError(f"integer gauge vector check failed: {vector_residual}")
    family = build_family(converted)
    out = args.out.resolve()
    protected = {args.submission.resolve().parent}
    if args.source_circuits is not None:
        protected.add(args.source_circuits.resolve())
    if out in protected:
        raise ValueError("output directory must be distinct from all source directories")
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{out} is nonempty; use a fresh directory or --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, circuit in enumerate(family):
        path = out / f"angle_k_{k}.{args.format}"
        circuit.to_file(path, format=args.format)
        rows.append({"k": k, "file": path.name, "sha256": sha256_file(path),
                     "num_qubits": int(circuit.num_qubits),
                     "toffoli_count": int(circuit.operation_statistics().num_toffoli_like),
                     "num_instructions": len(circuit)})
    if args.overwrite:
        keep = {row["file"] for row in rows}
        for path in out.glob("angle_k_*.*"):
            if path.suffix in (".kmb", ".kmx") and path.name not in keep:
                path.unlink()
    shutil.copyfile(args.submission, out / "submission.npz")
    np.save(out / "angle_words.npy", converted, allow_pickle=False)
    np.save(out / "source_angle_words.npy", original, allow_pickle=False)
    np.save(out / "gauge_root_signs.npy", signs, allow_pickle=False)
    manifest = {
        "method": "four_address_paired_combine_prefix_anf",
        "factorization_shape": {"R": r, "B": b, "C": c, "N": n},
        "num_vectors": d, "brot": brot, "batch_size": 3,
        "num_circuits": len(family), "source": source_info,
        "submission_sha256": sha256_file(out / "submission.npz"),
        "submission_copied_byte_for_byte": sha256_file(args.submission) == sha256_file(out / "submission.npz"),
        "gauge": {"integer_negation_and_half_turn_only": True,
                  "rounding_or_clipping": False,
                  "changed_words": int(np.count_nonzero(original != converted)),
                  "max_signed_vector_float_residual": vector_residual,
                  "projectors_preserved_exactly_by_algebra": True},
        "angle_words_sha256": sha256_file(out / "angle_words.npy"),
        "angle_toffoli_forward_and_reverse": 2 * sum(row["toffoli_count"] for row in rows),
        "max_angle_qubits": max(row["num_qubits"] for row in rows),
        "circuits": rows,
        "chemistry_status": "not evaluated by compiler; run unchanged official verifier",
        "attribution": "paired combine and bank erasure adapted from marculera/dfthc-merge-pipeline, Apache-2.0; see FOUR_ADDRESS_NOTICE.md",
    }
    (out / "compile_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
