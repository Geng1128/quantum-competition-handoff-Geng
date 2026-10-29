"""Checks for a fresh, pinned public portfolio reproduction notebook."""

import ast
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    script = ROOT / "scripts/prepare_public_refresh.py"
    assert script.exists(), "Public refresh generator is missing."
    spec = importlib.util.spec_from_file_location("public_refresh", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_notebook_runs_all_three_fresh_and_stays_private(tmp_path):
    module = load_module()
    result = module.prepare_job(ROOT, tmp_path / "job", owner="gengsr")
    metadata = json.loads(
        (tmp_path / "job/kernel-metadata.json").read_text(encoding="utf-8")
    )
    notebook = json.loads(
        (tmp_path / "job" / metadata["code_file"]).read_text(encoding="utf-8")
    )
    assert metadata["is_private"] is True
    assert not metadata["enable_internet"] and not metadata["enable_gpu"]
    assert metadata["kernel_sources"] == [module.SOURCE_KERNEL]
    assert result["uploaded"] is False
    assert result["archive_sha256"] == module.EXPECTED_ARCHIVE
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse(cell["source"])
    execution = notebook["cells"][-1]["source"]
    assert '"--tasks", *TASKS' in execution
    assert "complete_submission_verification" in execution
    assert "cached_results_used" in execution
    assert "subprocess.run" in execution
    assert "competitions submit" not in execution


def test_archive_inspection_rejects_traversal(tmp_path):
    module = load_module()
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../escape.kmb", b"x")
    with pytest.raises(ValueError):
        module.inspect_archive(path)
