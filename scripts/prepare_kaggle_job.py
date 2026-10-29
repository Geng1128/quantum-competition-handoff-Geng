"""Build a private, CPU-only Kaggle notebook for two fresh full verifications."""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import re
from pathlib import Path
from typing import Any

COMPETITION = "google-verifiable-quantum-circuit-compilation"
PUBLIC_KERNEL = "anazemcev/quantum-circuits-joint-norm-and-compact-dqi/3"
PUBLIC_ZIP_SHA256 = "ede2fb264c13bd663dcc945a60f005a8008752d98a4483a979ae5287fc972e2f"
DEFAULT_SLUG = "quantum-verification-workbench"
DEFAULT_SALT = "quantum-verification-workbench-20261025-v1"


def code_cell(source: str) -> dict[str, Any]:
    """Make an ordinary Python cell with a stable identifier."""
    source = source.strip() + "\n"
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": hashlib.sha256(source.encode()).hexdigest()[:12],
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def validate_notebook(notebook: dict[str, Any]) -> None:
    """Check the generated JSON and compile every executable cell locally."""
    if notebook.get("nbformat") != 4 or notebook.get("nbformat_minor") != 5:
        raise ValueError("The generated notebook must use nbformat 4.5.")
    cells = notebook.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("The notebook has no cells.")
    identifiers: set[str] = set()
    for number, cell in enumerate(cells):
        if cell["cell_type"] != "code" or not isinstance(cell["source"], list):
            raise ValueError(f"Cell {number} is not a Python code cell.")
        if cell["id"] in identifiers:
            raise ValueError("Notebook cell identifiers must be unique.")
        identifiers.add(cell["id"])
        source = "".join(cell["source"])
        compile(source, f"generated_cell_{number}", "exec")
    json.dumps(notebook, allow_nan=False)


def build_notebook(
    runner: bytes,
    metric: bytes,
    metric_metadata: dict[str, Any],
    salt: str,
) -> dict[str, Any]:
    """Embed source artifacts; the public notebook supplies only mounted files."""
    ast.parse(runner.decode("utf-8-sig"), filename="verify_candidate.py")
    metric_document = json.loads(metric)
    if not isinstance(metric_document.get("cells"), list):
        raise ValueError("The official metric is not a notebook.")
    payloads = {
        "verify_candidate.py": base64.b64encode(runner).decode("ascii"),
        "official_metric.ipynb": base64.b64encode(metric).decode("ascii"),
    }
    expected_hashes = {
        "verify_candidate.py": hashlib.sha256(runner).hexdigest(),
        "official_metric.ipynb": hashlib.sha256(metric).hexdigest(),
    }
    setup = f'''
from __future__ import annotations
import base64
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import stat
import subprocess
import sys
import time
import zipfile

os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
TOOLS = WORK / "tools"
RUNS = WORK / "runs"
TOOLS.mkdir(parents=True, exist_ok=True)
RUNS.mkdir(parents=True, exist_ok=True)
SALT = {salt!r}
TASKS = ("invsqrt", "dqi", "dfthcsos")
EXPECTED_PUBLIC_ZIP = {PUBLIC_ZIP_SHA256!r}
PINNED_PUBLIC_KERNEL = {PUBLIC_KERNEL!r}
PINNED_IMAGE = {metric_metadata['docker_image']!r}
SOURCE_HASHES = {expected_hashes!r}
EMBEDDED_SOURCES = {payloads!r}

def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def save_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\\n", encoding="utf-8")
    temporary.replace(path)

for filename, payload in EMBEDDED_SOURCES.items():
    content = base64.b64decode(payload, validate=True)
    if hashlib.sha256(content).hexdigest() != SOURCE_HASHES[filename]:
        raise ValueError("Embedded source digest mismatch: " + filename)
    destination = TOOLS / filename
    destination.write_bytes(content)
    if digest(destination) != SOURCE_HASHES[filename]:
        raise ValueError("Written source digest mismatch: " + filename)

ENVIRONMENT = {{
    "python": sys.version,
    "platform": platform.platform(),
    "packages": {{name: importlib.metadata.version(name) for name in ("numpy", "scipy")}},
    "docker_image": PINNED_IMAGE,
    "docker_image_pinning_type": "original",
    "cpu_only": True,
    "internet_enabled": False,
    "local_salt": SALT,
    "verification_mode": "fresh_full",
    "source_sha256": SOURCE_HASHES,
}}
save_json(RUNS / "environment.json", ENVIRONMENT)
print("Embedded runner and official Metric verified. CPU, offline, fresh full checks.", flush=True)
'''
    locate_and_extract = '''
known_roots = [
    INPUT / "competitions/google-verifiable-quantum-circuit-compilation",
    INPUT / "google-verifiable-quantum-circuit-compilation",
]
data_roots = {p.resolve() for p in known_roots if (p / "verifiers/invsqrt/verify_submission_invsqrt.py").is_file()}
if not data_roots:
    data_roots = {
        p.parents[2].resolve()
        for p in INPUT.rglob("verify_submission_invsqrt.py")
        if (p.parents[2] / "sample_submission").is_dir()
        and (p.parents[2] / "verifiers/dqi/verify_submission_dqi.py").is_file()
    }
if len(data_roots) != 1:
    raise RuntimeError(f"Expected one official competition data root, found {len(data_roots)}.")
DATA_ROOT = next(iter(data_roots))
OFFICIAL_SUBMISSION = DATA_ROOT / "sample_submission"

public_archives = [p for p in INPUT.rglob("submission.zip") if digest(p) == EXPECTED_PUBLIC_ZIP]
if len(public_archives) != 1:
    raise RuntimeError(f"Expected one SHA-256-pinned public V3 archive, found {len(public_archives)}.")
PUBLIC_ARCHIVE = public_archives[0]
PUBLIC_SUBMISSION = WORK / "candidates/public_baseline"

def extract_submission(archive_path, destination):
    """Extract a bounded file-only submission, rejecting traversal and symlinks."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Use a fresh Run All session: " + str(destination))
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        if not 4 <= len(members) <= 128:
            raise ValueError("Unexpected submission archive member count.")
        if sum(item.file_size for item in members) > 4 * 2**30:
            raise ValueError("Submission archive exceeds the uncompressed size budget.")
        names = set()
        angle_indices = []
        for item in members:
            name = item.filename
            parts = name.split("/")
            mode = item.external_attr >> 16
            if (not name or "\\\\" in name or ":" in name or name.startswith("/")
                or any(part in ("", ".", "..") for part in parts)
                or item.is_dir() or stat.S_IFMT(mode) not in (0, stat.S_IFREG)
                or item.flag_bits & 1 or name in names):
                raise ValueError("Unsafe or duplicate archive member: " + name)
            names.add(name)
            if name in {"invsqrt.kmb", "dqi.kmb", "dfthcsos/submission.npz"}:
                continue
            prefix, suffix = "dfthcsos/angle_k_", ".kmb"
            if not name.startswith(prefix) or not name.endswith(suffix):
                raise ValueError("Unexpected archive member: " + name)
            index = name[len(prefix):-len(suffix)]
            if not index.isdigit() or str(int(index)) != index:
                raise ValueError("Invalid ANGLE index: " + name)
            angle_indices.append(int(index))
        if not {"invsqrt.kmb", "dqi.kmb", "dfthcsos/submission.npz"} <= names:
            raise ValueError("The archive is missing required submission files.")
        if not angle_indices or sorted(angle_indices) != list(range(len(angle_indices))):
            raise ValueError("ANGLE circuits must have consecutive indices starting at zero.")
        destination.mkdir(parents=True)
        root = destination.resolve()
        extracted = []
        for item in members:
            target = destination.joinpath(*PurePosixPath(item.filename).parts)
            if not target.resolve().is_relative_to(root):
                raise ValueError("Archive member escapes the destination.")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(item) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=2**20)
            extracted.append({"path": item.filename, "sha256": digest(target), "bytes": target.stat().st_size})
    return extracted

public_members = extract_submission(PUBLIC_ARCHIVE, PUBLIC_SUBMISSION)
official_members = [
    {"path": str(p.relative_to(OFFICIAL_SUBMISSION)), "sha256": digest(p), "bytes": p.stat().st_size}
    for p in sorted(OFFICIAL_SUBMISSION.rglob("*")) if p.is_file()
]
INPUT_MANIFEST = {
    "competition": "google-verifiable-quantum-circuit-compilation",
    "official_data_root": str(DATA_ROOT),
    "official_submission": official_members,
    "public_kernel": PINNED_PUBLIC_KERNEL,
    "public_archive": str(PUBLIC_ARCHIVE),
    "public_archive_sha256": digest(PUBLIC_ARCHIVE),
    "public_submission": public_members,
    "embedded_sources": SOURCE_HASHES,
    "reuse_prior_verification_receipts": False,
}
save_json(RUNS / "input_manifest.json", INPUT_MANIFEST)
print(f"Pinned public archive extracted: {len(public_members)} files. Official root: {DATA_ROOT}", flush=True)
'''
    execution = '''
def verify_fresh(name, submission):
    output = RUNS / name
    if output.exists():
        raise FileExistsError("Use a fresh Run All session: " + str(output))
    command = [sys.executable, "-u", str(TOOLS / "verify_candidate.py"),
               "--data-root", str(DATA_ROOT), "--submission-dir", str(submission),
               "--output-dir", str(output), "--salt", SALT,
               "--metric", str(TOOLS / "official_metric.ipynb"), "--tasks", *TASKS]
    started = time.monotonic()
    log_path = RUNS / (name + ".console.log")
    execution = {"name": name, "status": "RUNNING", "verification_mode": "fresh_full",
                 "submission": str(submission), "output": str(output), "log": str(log_path)}
    receipt_path = RUNS / (name + ".execution.json")
    save_json(receipt_path, execution)
    print(f"Starting {name}: all three official full verifiers, fresh processes.", flush=True)
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                     timeout=3 * 60 * 60, check=False)
        execution["returncode"] = process.returncode
        result_path = output / "results.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else None
        execution["result_status"] = result.get("status") if result else None
        passed = (process.returncode == 0 and result is not None
                  and result.get("status") == "PASS"
                  and result.get("verification_mode") == "fresh_full"
                  and result.get("complete_submission_verification") is True
                  and set(result.get("tasks", {})) == set(TASKS)
                  and all(result["tasks"][task].get("status") == "PASS" for task in TASKS))
        execution["status"] = "PASS" if passed else "FAIL"
        execution["seconds"] = time.monotonic() - started
        save_json(receipt_path, execution)
        if not passed:
            save_json(WORK / "verification_failure.json", execution)
            raise RuntimeError(f"{name} failed fresh full verification; inspect {output} and {log_path}.")
        print(f"{name}: fresh full PASS in {execution['seconds']:.1f}s", flush=True)
        return result
    except (subprocess.TimeoutExpired, OSError, ValueError) as error:
        execution.update(status="FAIL", seconds=time.monotonic() - started,
                         error=f"{type(error).__name__}: {error}")
        save_json(receipt_path, execution)
        save_json(WORK / "verification_failure.json", execution)
        raise
'''
    comparison = '''
rows = {}
for task in TASKS:
    baseline = official_result["tasks"][task]
    candidate = public_result["tasks"][task]
    values = [baseline["cost"], candidate["cost"]]
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("Non-positive or non-finite task cost: " + task)
    rows[task] = {
        "official_cost": baseline["cost"], "public_cost": candidate["cost"],
        "improvement_ratio": baseline["cost"] / candidate["cost"],
        "official_qubits": baseline["num_qubits"], "public_qubits": candidate["num_qubits"],
        "official_shots": baseline["verified_shots"], "public_shots": candidate["verified_shots"],
    }
geometric_mean = math.exp(sum(math.log(row["improvement_ratio"]) for row in rows.values()) / len(TASKS))
COMPARISON = {
    "status": "PASS", "verification_mode": "fresh_full", "tasks": rows,
    "geometric_mean_improvement": geometric_mean,
    "local_score": geometric_mean, "observed_kaggle_score": None,
    "local_salt": SALT, "public_kernel": PINNED_PUBLIC_KERNEL,
    "public_archive_sha256": EXPECTED_PUBLIC_ZIP,
    "official_results_sha256": digest(RUNS / "official_baseline/results.json"),
    "public_results_sha256": digest(RUNS / "public_baseline/results.json"),
    "environment_sha256": digest(RUNS / "environment.json"),
    "input_manifest_sha256": digest(RUNS / "input_manifest.json"),
}
save_json(WORK / "comparison.json", COMPARISON)
print(json.dumps(COMPARISON, indent=2, allow_nan=False), flush=True)
'''
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": [
            code_cell(setup),
            code_cell(locate_and_extract),
            code_cell(execution),
            code_cell('official_result = verify_fresh("official_baseline", OFFICIAL_SUBMISSION)'),
            code_cell('public_result = verify_fresh("public_baseline", PUBLIC_SUBMISSION)'),
            code_cell(comparison),
        ],
    }
    validate_notebook(notebook)
    return notebook


def prepare_job(project_root: Path, output_dir: Path, owner: str, salt: str) -> dict[str, Any]:
    """Write a reviewable notebook and metadata without invoking Kaggle."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", owner):
        raise ValueError("Invalid Kaggle owner name.")
    if len(salt) < 32:
        raise ValueError("Use a local development salt of at least 32 characters.")
    runner_path = project_root / "scripts/verify_candidate.py"
    metric_dir = project_root / "external/notebooks/metric"
    metric_path = metric_dir / "google-quantum-ai-metric.ipynb"
    metadata_path = metric_dir / "kernel-metadata.json"
    metric_metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    image = metric_metadata.get("docker_image")
    if not isinstance(image, str) or not re.fullmatch(r"gcr\.io/kaggle-images/python@sha256:[0-9a-f]{64}", image):
        raise ValueError("The official Metric metadata has no immutable Python image digest.")
    notebook = build_notebook(runner_path.read_bytes(), metric_path.read_bytes(), metric_metadata, salt)
    code_filename = DEFAULT_SLUG + ".ipynb"
    metadata = {
        "id": f"{owner}/{DEFAULT_SLUG}",
        "title": "Quantum Verification Workbench",
        "code_file": code_filename,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [],
        "kernel_sources": [PUBLIC_KERNEL],
        "competition_sources": [COMPETITION],
        "model_sources": [],
        "docker_image": image,
        "docker_image_pinning_type": "original",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, value in ((code_filename, notebook), ("kernel-metadata.json", metadata)):
        path = output_dir / filename
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if filename.endswith(".ipynb"):
            validate_notebook(parsed)
        elif parsed != metadata:
            raise ValueError("Written kernel metadata differs from the generated configuration.")
    return {
        "status": "PREPARED",
        "directory": str(output_dir.resolve()),
        "kernel_id": metadata["id"],
        "docker_image": image,
        "public_kernel_source": PUBLIC_KERNEL,
        "notebook_cells": len(notebook["cells"]),
        "uploaded": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--owner", default="gengsr")
    parser.add_argument("--salt", default=DEFAULT_SALT)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir or project_root / "kaggle" / DEFAULT_SLUG
    result = prepare_job(project_root, output_dir.resolve(), args.owner, args.salt)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
