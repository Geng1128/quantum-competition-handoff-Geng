#!/bin/bash
# R10/B129/C44 recipe (version 5). Inputs: the C44 continuous refit (ctrunc.py 48 -> 44 + cont_lbfgs.py, e2 13.31 mHa).
FR=0.0005,0.001,0.0015,0.002,0.003,0.004,0.005,0.0075,0.01,0.015,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.135,0.15,0.17,0.19,0.21,0.24,0.27,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0
python src/pq.py cont_c44.npz q_c44.npz --brot 8 --fp32 --fracs $FR --split 0.05,0.3,12000,9000,2000   --lr-th 2e-4 --lr-w 5e-5 --dpen 6e-5 --final-w 4000 --gn --decay 0.2 --probes 8
python src/finish.py q_c44.npz f0.npz --kappa 0
python src/gauge.py f0.npz g0.npz --sweeps 60
python src/lamopt.py g0.npz lam.npz --steps 20000 --lr-w 5e-7 --lr-o 2e-6 --decay 0.01 --cap 0.01998   --e2-tol 1e-5 --gap-margin 2e-6 --gap-feas 5e-7
python src/finish.py lam.npz fin.npz --kappa 0
python src/gauge.py fin.npz final.npz --sweeps 60
python src/canon_gauge.py final.npz final_canon.npz
