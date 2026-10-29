import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "experiments/invsqrt/extract_assembly.py"
spec = importlib.util.spec_from_file_location("extract_assembly", SCRIPT)
extractor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extractor)

MAGIC = "%%writefile src/phuc_exp222/invsqrt.kmx"


def write_notebook(path, cells):
    path.write_text(json.dumps({"cells": cells}, ensure_ascii=False), encoding="utf-8")


def code(source):
    return {"cell_type": "code", "source": source.splitlines(keepends=True)}


@pytest.mark.parametrize("cells", [[], [code("print('no magic')\n")], [
    {"cell_type": "markdown", "source": [MAGIC + "\nZ_POW q0 0.25\n"]}
]])
def test_missing_code_magic_rejected(tmp_path, cells):
    notebook = tmp_path / "source.ipynb"
    write_notebook(notebook, cells)
    with pytest.raises(ValueError, match="exactly one"):
        extractor.extract_notebook(notebook, tmp_path / "output")
    assert not (tmp_path / "output/invsqrt.kmx").exists()


@pytest.mark.parametrize("cells", [
    [code(MAGIC + "\nX q0\n"), code(MAGIC + "\nZ q0\n")],
    [code(MAGIC + "\nX q0\n" + MAGIC + "\nZ q0\n")],
])
def test_multiple_target_magics_rejected(tmp_path, cells):
    notebook = tmp_path / "source.ipynb"
    write_notebook(notebook, cells)
    with pytest.raises(ValueError, match="exactly one"):
        extractor.extract_notebook(notebook, tmp_path / "output")


def test_roundtrip_preserves_body_exactly_and_is_deterministic(tmp_path):
    body = '# 中文注释\r\nAPPEND_TO_REGISTER q0 r0\r\nREGISTER r0 "x"\r\nZ_POW q0 0.25 # keep comment\r\n'
    notebook = tmp_path / "source.ipynb"
    source = MAGIC + "\r\n" + body
    write_notebook(notebook, [code("print('ignored')\n"), code(source)])
    output = tmp_path / "output"
    manifest = extractor.extract_notebook(notebook, output)
    assert (output / "invsqrt.kmx").read_bytes() == body.encode("utf-8")
    assert manifest["body_sha256"] == hashlib.sha256(body.encode("utf-8")).hexdigest()
    assert manifest["cell_source_sha256"] == hashlib.sha256(source.encode("utf-8")).hexdigest()
    assert manifest["notebook_sha256"] == hashlib.sha256(notebook.read_bytes()).hexdigest()
    assert manifest["cell_index"] == 1
    first = {path.name: path.read_bytes() for path in output.iterdir()}
    extractor.extract_notebook(notebook, output)
    assert first == {path.name: path.read_bytes() for path in output.iterdir()}


@pytest.mark.parametrize("angle, classification, exact", [
    ("0.25", "t", "1/4"), ("0.5", "clifford", "1/2"),
    ("1.0", "clifford", "1"), ("-0.25", "t", "-1/4"),
    ("2.25", "t", "9/4"), ("0", "clifford", "0"),
    ("1e-3", "synthesis", "1/1000"),
    ("0.1249999974869483985485951738070298233651556074619293212890625", "synthesis", None),
])
def test_phase_classification_is_exact(angle, classification, exact):
    report = extractor.analyze_assembly(f"Z_POW q8 {angle}\n")
    site = report["phase_sites"][0]
    assert site["angle_text"] == angle
    assert site["classification"] == classification
    assert site["angle_unit"] == "half_turns"
    assert site["target_wire"] == "q8"
    if exact is not None:
        assert site["half_turns_exact"] == exact


def test_comments_do_not_create_phases_or_qubits():
    body = '# Z_POW q999 0.123\n\nZ_POW q7 0.25 # Z_POW q888 0.001\nCX q0 q7\n'
    report = extractor.analyze_assembly(body)
    assert report["phase_counts"] == {"clifford": 0, "t": 1, "synthesis": 0}
    assert report["static_operation_counts"] == {"CX": 1, "Z_POW": 1}
    assert report["num_qubits_from_max_wire"] == 8
    assert report["phase_sites"][0]["assembly_line"] == 3
    assert report["phase_sites"][0]["cell_line"] == 4


@pytest.mark.parametrize("body", ["Z_POW q0 1/4\n", "Z_POW q0 nan\n", "Z_POW q0 0.25 if b0\n", "Z_POW 0.25 q0\n"])
def test_unhandled_phase_syntax_is_rejected(body):
    with pytest.raises(ValueError, match="Z_POW"):
        extractor.analyze_assembly(body)


def test_magic_must_be_cell_header(tmp_path):
    notebook = tmp_path / "source.ipynb"
    write_notebook(notebook, [code("print('not a cell magic')\n" + MAGIC + "\nX q0\n")])
    with pytest.raises(ValueError, match="header"):
        extractor.extract_notebook(notebook, tmp_path / "output")
