"""Build dfthcsos ANGLE circuits from a factorization npz and run the official verifier (local).

  python src/build_dft_local.py FACT.npz OUTDIR [--addr 2] [--J 0]
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [os.environ.get("QCC_VERIFIERS", "data/verifiers"), os.path.dirname(os.path.abspath(__file__))]
import angles_qrom as AQ  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("fact")
ap.add_argument("out")
ap.add_argument("--addr", type=int, default=2)
ap.add_argument("--J", type=int, default=0)
a = ap.parse_args()
out = Path(a.out)
shutil.rmtree(out, ignore_errors=True)
out.mkdir(parents=True)
z = np.load(a.fact, allow_pickle=False)
params = {k: np.array(z[k]) for k in z.files}
brot = int(params["brot"])
edges = params["tree_edges"].astype(np.int64)
words = AQ.canonical_words(params["angle_words"], edges, brot)
J = a.J or AQ.max_batch(brot, len(words), a.addr)
for k, d in enumerate(AQ.batch_deltas(words, J)):
    c = (AQ.build if a.addr == 4 else AQ.build2)(d, brot)
    c.to_file(str(out / f"angle_k_{k}.kmb"), format="kmb")
np.savez(out / "submission.npz", **{k: params[k] for k in ("U", "W", "H2_I", "H1_I", "Hsym_lower", "tree_edges")})
from dfthcsos.verify import verify  # noqa: E402
res = verify(str(out))
print(json.dumps({k: res[k] for k in ("status", "score", "lambda_eff", "toffoli_count", "num_qubits")}, default=str))
