"""Prepare a separate private offline CPU job for the fixed-word chemistry batch experiment."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import re


SLUG = "quantum-chemistry-batch-sweep"
SALT = "quantum-verification-workbench-20261025-v1"
PUBLIC_NOTEBOOK_SHA256 = "0ee01c1a037c4a1f340d243768a7059b5a7414ad375a8a1384cb41501b61b10e"
SOURCE_URL = "https://www.kaggle.com/code/anazemcev/quantum-circuits-joint-norm-and-compact-dqi?scriptVersionId=362074479"


def compiler_sources(notebook_path: Path) -> dict[str, bytes]:
    """Extract only the two pinned writefile bodies; execute none of the public notebook."""
    notebook_bytes = Path(notebook_path).read_bytes()
    if hashlib.sha256(notebook_bytes).hexdigest() != PUBLIC_NOTEBOOK_SHA256:
        raise ValueError("Public V3 notebook source hash differs from the reviewed source.")
    notebook = json.loads(notebook_bytes)
    sources = {}
    for index, filename in ((5, "learned_qrom.py"), (6, "learned_factor_compile.py")):
        cell = notebook["cells"][index]
        source = cell["source"]
        source = source if isinstance(source, str) else "".join(source)
        lines = source.splitlines(keepends=True)
        if cell["cell_type"] != "code" or not lines or lines[0].rstrip("\r\n") != "%%writefile src/" + filename:
            raise ValueError("Pinned compiler source cell differs: " + filename)
        body = "".join(lines[1:]).encode("utf-8")
        compile(body, filename, "exec")
        sources[filename] = body
    return sources


def _base_builder():
    path = Path(__file__).with_name("prepare_kaggle_job.py")
    spec = importlib.util.spec_from_file_location("chemistry_base_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_job(project_root: Path, output_dir: Path, owner: str = "gengsr", salt: str = SALT) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", owner) or not isinstance(salt, str) or len(salt) < 32:
        raise ValueError("Invalid owner or local salt.")
    project_root, output_dir = Path(project_root), Path(output_dir)
    base = _base_builder()
    metric_dir = project_root / "external/notebooks/metric"
    metric_metadata = json.loads((metric_dir / "kernel-metadata.json").read_text(encoding="utf-8-sig"))
    image = metric_metadata.get("docker_image")
    if not isinstance(image, str) or not re.fullmatch(r"gcr\.io/kaggle-images/python@sha256:[0-9a-f]{64}", image):
        raise ValueError("Official metric image must use an immutable digest.")
    notebook = base.build_notebook(
        (project_root / "scripts/verify_candidate.py").read_bytes(),
        (metric_dir / "google-quantum-ai-metric.ipynb").read_bytes(), metric_metadata, salt,
    )
    sources = compiler_sources(project_root / "external/notebooks/public/quantum-circuits-joint-norm-and-compact-dqi.ipynb")
    sources["batch_sweep.py"] = (project_root / "experiments/chemistry/batch_sweep.py").read_bytes()
    for filename, source in sources.items():
        compile(source, filename, "exec")
    payloads = {filename: base64.b64encode(content).decode("ascii") for filename, content in sources.items()}
    hashes = {filename: hashlib.sha256(content).hexdigest() for filename, content in sources.items()}
    experiment_setup = f'''
EXPERIMENT_PAYLOADS = {payloads!r}
EXPERIMENT_HASHES = {hashes!r}
for filename, payload in EXPERIMENT_PAYLOADS.items():
    content = base64.b64decode(payload, validate=True)
    if hashlib.sha256(content).hexdigest() != EXPERIMENT_HASHES[filename]:
        raise ValueError("Experiment source digest mismatch: " + filename)
    target = TOOLS / filename
    target.write_bytes(content)
    if digest(target) != EXPERIMENT_HASHES[filename]:
        raise ValueError("Written experiment source digest mismatch: " + filename)
ENVIRONMENT.update(experiment_scope="chemistry_only", experiment_sources=EXPERIMENT_HASHES,
                   public_source_url={SOURCE_URL!r}, public_notebook_sha256={PUBLIC_NOTEBOOK_SHA256!r},
                   batching_domain=[1, 2, 3], preserve_angle_words=True,
                   preserve_submission_npz_bytes=True, numerical_fitting=False)
save_json(RUNS / "environment.json", ENVIRONMENT)
print("Pinned compiler bodies restored unchanged; chemistry-only J=1/2/3 sweep.", flush=True)
'''
    execution = '''
command = [sys.executable, "-u", str(TOOLS / "batch_sweep.py"),
           "--data-root", str(DATA_ROOT), "--public-submission", str(PUBLIC_SUBMISSION),
           "--output-dir", str(WORK / "chemistry_batch_sweep"),
           "--tools-dir", str(TOOLS), "--salt", SALT]
print("Starting full baseline chemistry recovery, exact batch rebuilds, then at most one candidate verification.", flush=True)
subprocess.run(command, check=True, timeout=3 * 60 * 60)
summary_path = WORK / "chemistry_batch_sweep/sweep.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
if summary.get("status") not in {"NO_IMPROVEMENT", "CHEMISTRY_IMPROVEMENT_VERIFIED"}:
    raise RuntimeError("Chemistry sweep did not finish successfully: " + str(summary.get("status")))
save_json(WORK / "chemistry_sweep_result.json", summary)
print(json.dumps({"status": summary["status"], "winner": summary.get("winner")}, indent=2, allow_nan=False), flush=True)
'''
    notebook["cells"] = notebook["cells"][:2] + [base.code_cell(experiment_setup), base.code_cell(execution)]
    base.validate_notebook(notebook)
    metadata = {
        "id": f"{owner}/{SLUG}", "title": "Quantum Chemistry Batch Sweep",
        "code_file": SLUG + ".ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
        "dataset_sources": [], "kernel_sources": [base.PUBLIC_KERNEL],
        "competition_sources": [base.COMPETITION], "model_sources": [],
        "docker_image": image, "docker_image_pinning_type": "original",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, document in ((metadata["code_file"], notebook), ("kernel-metadata.json", metadata)):
        (output_dir / filename).write_text(json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return {"status": "PREPARED", "kernel_id": metadata["id"], "directory": str(output_dir.resolve()),
            "uploaded": False, "scope": "chemistry_only", "batches": [1, 2, 3],
            "docker_image": image, "experiment_source_sha256": hashes}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--owner", default="gengsr")
    parser.add_argument("--salt", default=SALT)
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    result = prepare_job(root, args.output_dir or root / "kaggle" / SLUG, args.owner, args.salt)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
