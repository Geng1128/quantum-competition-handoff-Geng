"""Extract the published invsqrt assembly without executing or modifying notebook code."""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import shlex


MAGIC = "%%writefile src/phuc_exp222/invsqrt.kmx"
SOURCE_URL = "https://www.kaggle.com/code/anazemcev/quantum-circuits-joint-norm-and-compact-dqi?scriptVersionId=362074479"
SCRIPT_VERSION_ID = 362074479
WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_NOTEBOOK = WORKSPACE / "external/notebooks/public/quantum-circuits-joint-norm-and-compact-dqi.ipynb"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "baseline"
DECIMAL = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")
WIRE = re.compile(r"q([0-9]+)\Z")
DIRECTIVES = {"REGISTER", "APPEND_TO_REGISTER"}
OPERATIONS = {
    "X", "CX", "CCX", "Z", "CZ", "Z_POW", "CSWAP", "R", "HMR",
    "PUSH_CONDITION", "POP_CONDITION", "INFO_TAG", "BIT_INVERT",
}
COUNT_SCOPE = (
    "Static text operation counts, including every conditional branch. "
    "These are not executed gate counts, verifier resource costs, or a Kaggle score."
)


def analyze_assembly(body: str) -> dict:
    """Index actual decimal Z_POW syntax using exact arithmetic in half-turn units."""
    operation_counts: Counter[str] = Counter()
    directive_counts: Counter[str] = Counter()
    phases = []
    phase_counts = {"clifford": 0, "t": 0, "synthesis": 0}
    maximum_wire = -1
    for line_number, raw_line in enumerate(body.splitlines(), 1):
        tokens = shlex.split(raw_line, comments=True, posix=True)
        if not tokens:
            continue
        operation = tokens[0]
        if operation not in DIRECTIVES | OPERATIONS:
            raise ValueError(f"Unhandled assembly operation at line {line_number}: {operation}")
        (directive_counts if operation in DIRECTIVES else operation_counts)[operation] += 1
        operands = [] if operation == "REGISTER" else tokens[1:]
        for operand in operands:
            match = WIRE.fullmatch(operand)
            if match:
                maximum_wire = max(maximum_wire, int(match[1]))
        if operation != "Z_POW":
            continue
        if len(tokens) != 3 or not WIRE.fullmatch(tokens[1]) or not DECIMAL.fullmatch(tokens[2]):
            raise ValueError(f"Unhandled Z_POW syntax at assembly line {line_number}: {raw_line}")
        angle_text = tokens[2]
        angle = Fraction(angle_text)
        classification = (
            "clifford" if angle % Fraction(1, 2) == 0
            else "t" if angle % Fraction(1, 4) == 0
            else "synthesis"
        )
        phase_counts[classification] += 1
        phases.append({
            "assembly_line": line_number,
            "cell_line": line_number + 1,
            "target_wire": tokens[1],
            "angle_text": angle_text,
            "angle_unit": "half_turns",
            "half_turns_exact": str(angle),
            "classification": classification,
            "source_line": raw_line,
        })
    return {
        "body_lines": len(body.splitlines()),
        "static_operation_counts": dict(sorted(operation_counts.items())),
        "static_directive_counts": dict(sorted(directive_counts.items())),
        "static_operation_count": sum(operation_counts.values()),
        "num_qubits_from_max_wire": maximum_wire + 1,
        "phase_counts": phase_counts,
        "phase_sites": phases,
        "count_scope": COUNT_SCOPE,
    }


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def extract_notebook(
    notebook_path: Path, output_dir: Path,
    source_url: str = SOURCE_URL, script_version_id: int = SCRIPT_VERSION_ID,
) -> dict:
    """Write the unique target cell's exact body, phase index, and deterministic manifest."""
    notebook_path = Path(notebook_path)
    notebook_bytes = notebook_path.read_bytes()
    notebook = json.loads(notebook_bytes)
    matches = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        source = source if isinstance(source, str) else "".join(source)
        for line_number, line in enumerate(source.splitlines(keepends=True), 1):
            if line.rstrip("\r\n") == MAGIC:
                matches.append((index, line_number, source))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one code-cell target writefile magic, found {len(matches)}.")
    cell_index, magic_line, source = matches[0]
    if magic_line != 1:
        raise ValueError("Target writefile magic must be the code cell header.")
    source_lines = source.splitlines(keepends=True)
    body = "".join(source_lines[1:])
    body_bytes = body.encode("utf-8")
    analysis = analyze_assembly(body)
    body_hash = hashlib.sha256(body_bytes).hexdigest()
    phase_document = {
        "schema_version": 1,
        "body_sha256": body_hash,
        "positions": {"assembly_line": "1-based body line", "cell_line": "1-based cell line, including magic"},
        "angle_unit": "half_turns",
        "phase_sites": analysis.pop("phase_sites"),
    }
    phase_bytes = _json_bytes(phase_document)
    manifest = {
        "schema_version": 1,
        "source_url": source_url,
        "script_version_id": script_version_id,
        "notebook_filename": notebook_path.name,
        "notebook_sha256": hashlib.sha256(notebook_bytes).hexdigest(),
        "cell_index": cell_index,
        "cell_source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "magic": MAGIC,
        "body_sha256": body_hash,
        "body_bytes": len(body_bytes),
        "phase_sites_sha256": hashlib.sha256(phase_bytes).hexdigest(),
        **analysis,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in {
        "invsqrt.kmx": body_bytes,
        "phase_sites.json": phase_bytes,
        "manifest.json": _json_bytes(manifest),
    }.items():
        (output_dir / name).write_bytes(content)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-url", default=SOURCE_URL)
    parser.add_argument("--script-version-id", type=int, default=SCRIPT_VERSION_ID)
    args = parser.parse_args(argv)
    manifest = extract_notebook(args.notebook, args.output_dir, args.source_url, args.script_version_id)
    print(json.dumps({
        "body_sha256": manifest["body_sha256"], "body_lines": manifest["body_lines"],
        "num_qubits_from_max_wire": manifest["num_qubits_from_max_wire"],
        "phase_counts": manifest["phase_counts"], "count_scope": manifest["count_scope"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
