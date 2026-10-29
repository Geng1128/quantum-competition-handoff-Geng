from fractions import Fraction
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import shlex

import pytest


SCRIPT = Path(__file__).parents[1] / "experiments/invsqrt/optimize_assembly.py"
spec = importlib.util.spec_from_file_location("optimize_assembly", SCRIPT)
optimizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(optimizer)


def evaluate(body, bits):
    state = list(bits)
    phase = Fraction(0)
    for line in body.splitlines():
        tokens = shlex.split(line, comments=True)
        if not tokens:
            continue
        operation = tokens[0]
        wires = [int(wire[1:]) for wire in tokens[1:] if wire.startswith("q")]
        if operation == "X":
            state[wires[0]] ^= 1
        elif operation in ("CX", "CCX"):
            if all(state[wire] for wire in wires[:-1]):
                state[wires[-1]] ^= 1
        elif operation in ("Z", "CZ", "CCZ"):
            if all(state[wire] for wire in wires):
                phase += 1
        elif operation == "Z_POW":
            if state[wires[0]]:
                phase += Fraction(tokens[2])
        elif operation in ("SWAP", "CSWAP"):
            if operation == "SWAP" or state[wires[0]]:
                first, second = wires[-2:]
                state[first], state[second] = state[second], state[first]
        else:
            raise AssertionError(f"Unsupported test oracle instruction: {operation}")
    return tuple(state), phase % 2


@pytest.mark.parametrize("gate", [
    "X q0", "CX q0 q1", "CCX q0 q1 q2", "Z q0", "CZ q0 q1",
    "CCZ q0 q1 q2", "SWAP q0 q1", "CSWAP q0 q1 q2",
])
def test_cancel_involution_preserves_every_basis_state_and_phase(gate):
    original = gate + "\nZ_POW q3 0.125\n" + gate + "\n"
    candidate, report = optimizer.optimize_text(original)
    assert candidate == "Z_POW q3 0.125\n"
    assert report["removed_static_gate_counts"] == {gate.split()[0]: 2}
    for bits in itertools.product((0, 1), repeat=4):
        assert evaluate(original, bits) == evaluate(candidate, bits)


def test_nested_cancellations_and_disjoint_interleaving():
    original = "CCX q0 q1 q2\nX q3\nZ q3\nZ q3\nX q3\nCCX q0 q1 q2\n"
    candidate, report = optimizer.optimize_text(original)
    assert candidate == ""
    assert report["removed_static_toffoli_like"] == 2
    for bits in itertools.product((0, 1), repeat=4):
        assert evaluate(original, bits) == evaluate(candidate, bits)


@pytest.mark.parametrize("barrier", [
    "PUSH_CONDITION if b0", "POP_CONDITION", "HMR q3 b0", "R q3",
    "BIT_INVERT b0", "BIT_STORE0 b0", "BIT_STORE1 b0", "INFO_TAG marker",
    "REGISTER r0 \"x\"", "APPEND_TO_REGISTER q3 r0", "UNKNOWN q3",
    "X q3 if b0", "Z_POW q3 0.125 if b0",
])
def test_barriers_block_cancellation_even_on_unrelated_qubits(barrier):
    original = f"CCX q0 q1 q2\n{barrier}\nCCX q0 q1 q2\n"
    candidate, report = optimizer.optimize_text(original)
    assert candidate == original
    assert report["removed_static_toffoli_like"] == 0


@pytest.mark.parametrize("middle", ["X q0", "CX q1 q3", "Z_POW q2 0.125", "CSWAP q2 q3 q4"])
def test_shared_qubit_blocks_cancellation(middle):
    original = f"CCX q0 q1 q2\n{middle}\nCCX q0 q1 q2\n"
    assert optimizer.optimize_text(original)[0] == original


def test_condition_boundaries_preserved_and_never_crossed():
    original = "X q0\nPUSH_CONDITION if b0\nX q0\nPOP_CONDITION\nX q0\n"
    assert optimizer.optimize_text(original)[0] == original
    inside = "PUSH_CONDITION if b0\nCCX q0 q1 q2\nCCX q0 q1 q2\nPOP_CONDITION\n"
    assert optimizer.optimize_text(inside)[0] == "PUSH_CONDITION if b0\nPOP_CONDITION\n"


def test_unmodified_lines_remain_byte_exact():
    original = '# preserve 中文\r\nCCX q0 q1 q2  # first\r\n\r\n  Z_POW q3 0.000125   # untouched\r\nCCX q0 q1 q2 # second\r\n'
    candidate, _ = optimizer.optimize_text(original)
    assert candidate == '# preserve 中文\r\n\r\n  Z_POW q3 0.000125   # untouched\r\n'


def test_angles_never_cancel_or_change():
    original = "Z_POW q0 0.25\nZ_POW q0 0.25\n"
    assert optimizer.optimize_text(original)[0] == original


def test_malformed_or_repeated_wire_gate_is_barrier():
    original = "X q0\nCCX q3 q3 q4\nX q0\n"
    assert optimizer.optimize_text(original)[0] == original


@pytest.mark.parametrize("original", [
    "X q0\nZ_POW q00 0.5\nX q0\n",
    "X q1\nZ_POW q01 0.5\nX q1\n",
    "CX q0 q1\nZ q01\nCX q0 q1\n",
])
def test_leading_zero_wire_aliases_are_barriers(original):
    candidate, report = optimizer.optimize_text(original)
    assert candidate == original
    assert report["removed_static_gate_count"] == 0
    for bits in itertools.product((0, 1), repeat=2):
        assert evaluate(original, bits) == evaluate(candidate, bits)


def test_file_cli_preserves_input_and_reports_hashes(tmp_path):
    input_path = tmp_path / "baseline.kmx"
    original = b"CCX q0 q1 q2\nX q3\nCCX q0 q1 q2\n"
    input_path.write_bytes(original)
    output_path = tmp_path / "candidate.kmx"
    report_path = tmp_path / "manifest.json"
    assert optimizer.main(["--input", str(input_path), "--output", str(output_path), "--report", str(report_path)]) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert input_path.read_bytes() == original
    assert output_path.read_bytes() == b"X q3\n"
    assert report["input_sha256"] == hashlib.sha256(original).hexdigest()
    assert report["candidate_sha256"] == hashlib.sha256(b"X q3\n").hexdigest()
    assert report["removed_static_toffoli_like"] == 2
    assert "not" in report["count_scope"]
    assert report["cancelled_pairs"][0]["input_lines"] == [1, 3]


def test_cli_rejects_overwriting_input(tmp_path):
    source = tmp_path / "baseline.kmx"
    source.write_text("X q0\nX q0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="distinct"):
        optimizer.main(["--input", str(source), "--output", str(source), "--report", str(tmp_path / "report.json")])
