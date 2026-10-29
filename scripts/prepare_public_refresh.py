"""Prepare an offline, private all-three fresh verification of a pinned public ZIP.

No public notebook code or cached PASS receipt is executed. Reuses our audited
verification-workbench setup/extraction cells, with new source and archive pins.
Preparing this notebook neither uploads it nor submits to the competition.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import re
import stat
import zipfile


SOURCE_KERNEL = "alexchilton/toffolis-go-18-3-schur-dqi-at-883-qubits"
EXPECTED_ARCHIVE = "cf7517ee9985352279fc43b8805fb5a0696732f7f70107579c604344ee2deb60"
SLUG = "quantum-public-refresh-20261027"
REMOTE_SLUG = "quantum-public-refresh-fresh-full-verification"
COMPETITION = "google-verifiable-quantum-circuit-compilation"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inspect_archive(path: Path) -> list[dict]:
    """Stream-hash only an exact, bounded standard submission layout."""
    records = []
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if (
            not 4 <= len(members) <= 128
            or sum(m.file_size for m in members) > 4 * 2**30
        ):
            raise ValueError("Unexpected archive size/member count.")
        names = set()
        angles = []
        for item in members:
            name = item.filename
            mode = stat.S_IFMT(item.external_attr >> 16)
            valid = name in ("invsqrt.kmb", "dqi.kmb", "dfthcsos/submission.npz")
            match = re.fullmatch(r"dfthcsos/angle_k_(0|[1-9][0-9]*)\.kmb", name)
            if match:
                angles.append(int(match[1]))
                valid = True
            if (
                not valid
                or name in names
                or item.is_dir()
                or mode not in (0, stat.S_IFREG)
                or item.flag_bits & 1
            ):
                raise ValueError("Unsafe/duplicate/unexpected member: " + name)
            names.add(name)
            with archive.open(item) as stream:
                sha = hashlib.file_digest(stream, "sha256").hexdigest()
            records.append(dict(path=name, bytes=item.file_size, sha256=sha))
        if (
            not {"invsqrt.kmb", "dqi.kmb", "dfthcsos/submission.npz"} <= names
            or not angles
            or sorted(angles) != list(range(len(angles)))
        ):
            raise ValueError("Missing submission files or nonconsecutive ANGLE family.")
    return records


def _replace_assignment(source: str, name: str, value: object) -> str:
    lines = source.splitlines(keepends=True)
    matches = [i for i, line in enumerate(lines) if line.startswith(name + " = ")]
    if len(matches) != 1:
        raise ValueError("Unexpected setup assignment: " + name)
    lines[matches[0]] = name + " = " + repr(value) + "\n"
    return "".join(lines)


def prepare_job(project_root: Path, output_dir: Path, owner: str = "gengsr") -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", owner):
        raise ValueError("Invalid owner.")
    root = project_root.resolve()
    archive = root / "external/20261027-schur-output/submission.zip"
    if digest(archive) != EXPECTED_ARCHIVE:
        raise ValueError("Public archive changed; review new bytes before use.")
    members = inspect_archive(archive)
    template_folder = root / "kaggle/quantum-portfolio-update"
    notebook = json.loads(
        (template_folder / "quantum-portfolio-update.ipynb").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (template_folder / "kernel-metadata.json").read_text(encoding="utf-8")
    )
    notebook["cells"] = notebook["cells"][:2]
    setup = notebook["cells"][0]["source"]
    locate = notebook["cells"][1]["source"]
    if isinstance(setup, list) and all(isinstance(line, str) for line in setup):
        setup = "".join(setup)
    if isinstance(locate, list) and all(isinstance(line, str) for line in locate):
        locate = "".join(locate)
    if not isinstance(setup, str) or not isinstance(locate, str):
        raise ValueError("Unexpected template cell source format.")
    sources = {
        "verify_candidate.py": (root / "scripts/verify_candidate.py").read_bytes(),
        "official_metric.ipynb": (
            root / "external/notebooks/metric/google-quantum-ai-metric.ipynb"
        ).read_bytes(),
    }
    for name, value in {
        "EXPECTED_PUBLIC_ZIP": EXPECTED_ARCHIVE,
        "PINNED_PUBLIC_KERNEL": SOURCE_KERNEL,
        "SOURCE_HASHES": {
            key: hashlib.sha256(value).hexdigest() for key, value in sources.items()
        },
        "EMBEDDED_SOURCES": {
            key: base64.b64encode(value).decode() for key, value in sources.items()
        },
    }.items():
        setup = _replace_assignment(setup, name, value)
    locate = locate.replace("pinned public V3 archive", "pinned public refresh archive")
    notebook["cells"][0]["source"] = setup
    notebook["cells"][1]["source"] = locate
    previous = json.loads(
        (root / "runs/portfolio_update_v1/portfolio_result.json").read_text()
    )
    refs = previous["reference_cost_source"]["official_costs"]
    backend = previous["current_backend_sha256"]
    execution = f"""EXPECTED_MEMBERS = {members!r}
EXPECTED_BACKEND = {backend!r}
REFERENCE_COSTS = {refs!r}
OLD_LOCAL_SCORE = 10.531117694808444
if public_members != EXPECTED_MEMBERS:
    raise ValueError("Extracted public members differ from locally inspected bytes.")
CURRENT_BACKEND = {{name: digest(DATA_ROOT / name) for name in EXPECTED_BACKEND}}
if CURRENT_BACKEND != EXPECTED_BACKEND:
    raise ValueError("Official backend changed; review before using historical references.")
save_json(RUNS / "backend_manifest.json", CURRENT_BACKEND)
output = RUNS / "fresh_public_refresh"
command = [sys.executable, "-u", str(TOOLS / "verify_candidate.py"),
           "--data-root", str(DATA_ROOT), "--submission-dir", str(PUBLIC_SUBMISSION),
           "--output-dir", str(output), "--salt", SALT,
           "--metric", str(TOOLS / "official_metric.ipynb"), "--tasks", *TASKS]
print("Running all three fresh full verifiers. Public cached PASS values are not used.", flush=True)
with (RUNS / "fresh_public_refresh.console.log").open("w", encoding="utf-8") as log:
    process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=10800, check=False)
if process.returncode:
    raise RuntimeError("Fresh full verification failed; no submission is exported.")
fresh = json.loads((output / "results.json").read_text())
if fresh.get("status") != "PASS" or not fresh.get("complete_submission_verification") or fresh.get("cached_results_used") is not False:
    raise ValueError("Incomplete or reused verification.")
if set(fresh["tasks"]) != set(TASKS):
    raise ValueError("A task is missing.")
observed = {{record["path"]: record["sha256"] for task in TASKS for record in fresh["tasks"][task]["files"]}}
expected = {{record["path"]: record["sha256"] for record in EXPECTED_MEMBERS}}
if observed != expected or any(not fresh["tasks"][task].get("fresh_verification") for task in TASKS):
    raise ValueError("Verified files or fresh-verification evidence differ.")
costs = {{task: float(fresh["tasks"][task]["cost"]) for task in TASKS}}
score = math.prod(REFERENCE_COSTS[task] / costs[task] for task in TASKS) ** (1/3)
result = {{"status": "IMPROVEMENT_VERIFIED_FRESH_FULL" if score > OLD_LOCAL_SCORE else "VERIFIED_NO_IMPROVEMENT",
          "source_kernel": PINNED_PUBLIC_KERNEL, "source_archive_sha256": EXPECTED_PUBLIC_ZIP,
          "old_local_score": OLD_LOCAL_SCORE, "new_local_score": score,
          "local_score_multiplier": score / OLD_LOCAL_SCORE,
          "observed_kaggle_score": None,
          "score_scope": "Local salted full verification against historical reference costs; not a Kaggle score.",
          "verification_mode": "all_three_fresh_full", "cached_results_used": False,
          "current_backend_sha256": CURRENT_BACKEND, "salt": SALT,
          "fresh_result_sha256": digest(output / "results.json"), "tasks": fresh["tasks"]}}
if score > OLD_LOCAL_SCORE:
    shutil.copyfile(PUBLIC_ARCHIVE, WORK / "submission.zip")
    if digest(WORK / "submission.zip") != EXPECTED_PUBLIC_ZIP:
        raise ValueError("Exported ZIP differs from the freshly checked archive.")
    result["submission_zip_sha256"] = EXPECTED_PUBLIC_ZIP
save_json(WORK / "public_refresh_result.json", result)
(WORK / "SOURCE_NOTICE.txt").write_text(
    "Public circuit reproduction; no authorship of inherited circuits is claimed.\\n"
    "Portfolio: https://www.kaggle.com/code/" + PINNED_PUBLIC_KERNEL + "\\n"
    "invsqrt/chemistry: mrmeows QCC open circuits and credited upstream authors.\\n"
    "DQI: jday96314 QuantumAI Schur decoder; Alex Chilton 883Q reverse synthesis.\\n"
    "Retain original Apache-2.0 licenses and upstream notices on redistribution.\\n"
    "Our work: independent pinned all-three fresh full verification.\\n", encoding="utf-8")
print(json.dumps({{key: result[key] for key in ("status", "new_local_score", "local_score_multiplier")}}, indent=2), flush=True)
"""
    notebook["cells"].append(
        {
            "cell_type": "code",
            "execution_count": None,
            "outputs": [],
            "metadata": {},
            "source": execution,
        }
    )
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse(cell["source"])
    metadata.update(
        id=f"{owner}/{REMOTE_SLUG}",
        title="Quantum Public Refresh | Fresh Full Verification",
        code_file=SLUG + ".ipynb",
        is_private=True,
        enable_gpu=False,
        enable_tpu=False,
        enable_internet=False,
        kernel_sources=[SOURCE_KERNEL],
        dataset_sources=[],
        model_sources=[],
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in [
        (metadata["code_file"], notebook),
        ("kernel-metadata.json", metadata),
    ]:
        (output_dir / name).write_text(
            json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
        )
    return dict(
        status="PREPARED",
        kernel_id=metadata["id"],
        directory=str(output_dir.resolve()),
        archive_sha256=EXPECTED_ARCHIVE,
        member_count=len(members),
        uploaded=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--owner", default="gengsr")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_job(
                args.project_root,
                args.output_dir or args.project_root / "kaggle" / SLUG,
                args.owner,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
