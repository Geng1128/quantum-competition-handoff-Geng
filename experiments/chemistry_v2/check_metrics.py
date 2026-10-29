"""Cross-check torch metrics against the official numpy implementation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import dfthc_torch as dt
import refdata

if str(refdata.VERIFIERS_DIR) not in sys.path:
    sys.path.insert(0, str(refdata.VERIFIERS_DIR))
from dfthcsos.scoreinfra.dfthc_math import evaluate_dfthc_angle_words, evaluate_dfthc_vectors  # noqa: E402
from dfthcsos.verify import native_inputs  # noqa: E402


def official_ref(ref: refdata.Reference) -> dict:
    return {"oei": ref.oei, "tei": ref.tei, "electrons": ref.electrons, "ecore": ref.ecore}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=Path)
    args = parser.parse_args()
    ref = refdata.load()
    with np.load(args.npz, allow_pickle=False) as data:
        params = {k: np.array(data[k]) for k in data.files}
    w, W, _, _ = native_inputs(params)
    U = params["U"].astype(np.float64)
    R, B, N = U.shape

    problem = dt.Problem.build(ref)
    dev = problem.device
    m = dt.metrics(
        problem,
        torch.tensor(U, dtype=torch.float64, device=dev),
        torch.tensor(w, dtype=torch.float64, device=dev),
        torch.tensor(W, dtype=torch.float64, device=dev),
    )
    torch_report = dt.scalar_report(m)

    h1p = m["h1_prime"].detach().cpu().numpy()
    _, vecs = np.linalg.eigh(h1p)
    u_all = np.concatenate([vecs.T, U.reshape(R * B, N)], axis=0)
    off = evaluate_dfthc_vectors(u_all, w, W, ref.S, ref.beta_1, official_ref(ref), energy_upper=refdata.UPPER_ENERGY)
    out = {"torch_continuous": torch_report, "official_continuous": off}
    if "angle_words" in params:
        brot = int(np.max(params["angle_words"])).bit_length()
        brot = int(params["brot"]) if "brot" in params else brot
        out["official_quantized"] = evaluate_dfthc_angle_words(
            params["angle_words"].astype(np.int64), w, W, ref.S, ref.beta_1, official_ref(ref),
            brot=brot, tree_edges=params["tree_edges"].astype(np.int64), energy_upper=refdata.UPPER_ENERGY,
        )
        out["brot"] = brot
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
