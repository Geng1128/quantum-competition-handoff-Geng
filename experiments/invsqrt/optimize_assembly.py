"""Conservatively cancel self-inverse gates separated only by operations on disjoint qubits."""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import shlex


ARITY = {"X": 1, "CX": 2, "CCX": 3, "Z": 1, "CZ": 2, "CCZ": 3, "SWAP": 2, "CSWAP": 3}
# Noncanonical IDs such as q00 may alias q0; leave their entire instruction as a barrier.
WIRE = re.compile(r"q(?:0|[1-9][0-9]*)\Z")
COUNT_SCOPE = (
    "Static counts of removed instructions, including conditional regions; "
    "not executed Toffoli counts, verifier resource cost, or a Kaggle score."
)
RULE = (
    "Cancel identical self-inverse gates only when every remaining intervening operation "
    "acts on a disjoint set of qubits. Conditions, measurements, resets, classical bit "
    "operations, register declarations, metadata and unknown instructions are barriers."
)


def _classify_line(line: str) -> tuple[str, tuple[str, ...] | None, frozenset[str]]:
    try:
        tokens = shlex.split(line, comments=True, posix=True)
    except ValueError:
        return "barrier", None, frozenset()
    if not tokens:
        return "ignore", None, frozenset()
    operation, operands = tokens[0], tokens[1:]
    if operation in ARITY and len(operands) == ARITY[operation]:
        if len(set(operands)) == len(operands) and all(WIRE.fullmatch(wire) for wire in operands):
            return "involution", tuple(tokens), frozenset(operands)
    if operation == "Z_POW" and len(operands) == 2 and WIRE.fullmatch(operands[0]):
        try:
            Fraction(operands[1])
        except (ValueError, ZeroDivisionError):
            return "barrier", None, frozenset()
        return "unitary", None, frozenset([operands[0]])
    return "barrier", None, frozenset()


def optimize_text(original: str) -> tuple[str, dict]:
    """Return unchanged surviving lines plus a static report; do not approximate any angle."""
    lines = original.splitlines(keepends=True)
    surviving: list[str | None] = list(lines)
    parsed = [_classify_line(line) for line in lines]
    removed: Counter[str] = Counter()
    pairs = []
    for current, (kind, gate, wires) in enumerate(parsed):
        if kind != "involution":
            continue
        for previous in range(current - 1, -1, -1):
            if surviving[previous] is None:
                continue
            previous_kind, previous_gate, previous_wires = parsed[previous]
            if previous_kind == "barrier":
                break
            if previous_wires & wires:
                if previous_kind == "involution" and previous_gate == gate:
                    surviving[previous] = surviving[current] = None
                    removed[gate[0]] += 2
                    pairs.append({
                        "operation": gate[0], "wires": list(gate[1:]),
                        "input_lines": [previous + 1, current + 1],
                    })
                break
    candidate = "".join(line for line in surviving if line is not None)
    return candidate, {
        "schema_version": 1,
        "rule": RULE,
        "input_sha256": hashlib.sha256(original.encode("utf-8")).hexdigest(),
        "candidate_sha256": hashlib.sha256(candidate.encode("utf-8")).hexdigest(),
        "input_lines": len(lines),
        "candidate_lines": len(candidate.splitlines()),
        "removed_static_gate_counts": dict(sorted(removed.items())),
        "removed_static_gate_count": sum(removed.values()),
        "removed_static_toffoli_like": sum(removed[gate] for gate in ("CCX", "CCZ", "CSWAP")),
        "cancelled_pairs": pairs,
        "count_scope": COUNT_SCOPE,
        "official_verification_status": "NOT_RUN",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = [path.resolve() for path in (args.input, args.output, args.report)]
    if len(set(paths)) != 3:
        raise ValueError("Input, output and report must use distinct paths.")
    original = args.input.read_bytes().decode("utf-8")
    candidate, report = optimize_text(original)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(candidate.encode("utf-8"))
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        key: report[key] for key in (
            "input_sha256", "candidate_sha256", "removed_static_gate_counts",
            "removed_static_toffoli_like", "official_verification_status", "count_scope",
        )
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
