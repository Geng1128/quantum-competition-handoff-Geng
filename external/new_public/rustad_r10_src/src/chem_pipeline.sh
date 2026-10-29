#!/bin/bash
# finish -> gauge -> lamopt -> finish -> gauge -> local build+verify.  usage: chem_pipeline.sh IN.npz TAG
set -e
cd "${QCC_ROOT:-.}"
IN=$1; T=$2
python src/finish.py $IN runs/${T}_fin.npz --kappa 0 | grep -E "effective|final e1|verifier"
python src/gauge.py runs/${T}_fin.npz runs/${T}_g.npz --sweeps 60 | tail -1
python src/lamopt.py runs/${T}_g.npz runs/${T}_lam.npz --steps 1500 --lr-w 5e-7 --lr-o 2e-6 --log 500 --cap $3 | tail -1
python src/finish.py runs/${T}_lam.npz runs/${T}_lamfin.npz --kappa 0 | grep -E "effective|final e1|verifier"
python src/gauge.py runs/${T}_lamfin.npz runs/${T}_final.npz --sweeps 60 | tail -1
python src/build_dft_local.py runs/${T}_final.npz builds/dft_${T} | tail -1
