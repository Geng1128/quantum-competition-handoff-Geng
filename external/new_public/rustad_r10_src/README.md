# DF-THC R10 merged-group factorization (dfthcsos, up to 9.09x)

Chemistry factorization and training sources for the Google Quantum AI Verifiable Quantum Circuit Compilation
competition. Used by the notebook `michaelrustad/vqcc-r10-merged-group-chemistry-12-87-lb`.

Official `dfthcsos` verifier on the circuits built from this file: **PASS, cost 5.59893e8 = 5.993x reference**
(R = 10, B = 129, C = 60, brot 8, lambda_eff 48.031, 26,078 Toffoli, 447 qubits, error_2 19.65 mHa, 11 ANGLE circuits).

## Files

| path | content |
|---|---|
| `fact/dfthc_r10_b129_c60_brot8.npz` | `U (10,129,76)`, `W (10,129,60)`, `H2_I (10,60)`, `H1_I`, `Hsym_lower`, `tree_edges (75,2)`, `brot = 8`, `angle_words (1366,75)` (76 one-body + 1290 two-body Givens words). SHA-256 `27dda66f38c60821665cfc74e51d1bd2bfd2a7f0d902dbaffcdc5b26543dadad` |
| `src/build_dft_local.py` | builds the ANGLE circuits (2-address banked QROM, J = 7 words per circuit) and runs the official verifier |
| `src/angles_qrom.py` | signed half-open gauge + banked-QROM ANGLE_k circuits for any (brot, J) |
| `src/banked_lab.py`, `src/gauge_qrom.py` | upstream Apache-2.0 banked QROM / gauge code (credits in their headers) |
| `src/grp_compress.py` | merge DF-THC groups and recompress |
| `src/cont.py`, `src/cont_lbfgs.py` | continuous refit (Adam / L-BFGS) with the D(w) penalty |
| `src/pq.py` | progressive quantization of the Givens words with Adam compensation |
| `src/flip.py` | exact greedy bit-flip polish |
| `src/finish.py` | one-body (Wo) effective-norm refit + one-body flips |
| `src/gauge.py` | Gram-preserving column rotations |
| `src/lamopt.py` | joint w / Wo lambda_eff descent under an error_2 cap |
| `src/chem_pipeline.sh` | finish -> gauge -> lamopt -> finish -> gauge -> build + verify |
| `src/dft_eval.py`, `src/qat.py` | exact metric wrappers and the packed two-body target |

Set `QCC_VERIFIERS` to the competition `verifiers` directory. The training scripts use PyTorch (CUDA).

## Recipe

1. Start from Ahmed Berat Ozer's public (26, 60, 30) factorization.
2. Merge pairs of groups into (13, 129, 60). Each merged group contains its two parents exactly
   (block weights), and B >= 129 halves the inner QROAM temporary register at brot 8.
3. Drop the weakest group with a long L-BFGS continuous refit after each drop (13 -> 12 -> 11 -> 10;
   R10 continuous error_2 10.26 mHa).
4. Progressive quantization to brot 8: 64 stages x 400 Adam steps (error_2 19.56 mHa).
5. `chem_pipeline.sh` with error_2 cap 19.85 mHa: lambda_eff 48.03.
6. Build 11 ANGLE circuits with the 2-address banked QROM.

## Version 2: R10/B129/C46 (8.440x with the exact tensor lookup compiler)

`fact/dfthc_r10_b129_c46_brot8.npz` (SHA-256 `4c34e1174b38371ff37ad750eccc004b6e29489d79a615d641eac839f170704a`):
(R, B, C) = (10, 129, 46), brot 8, error_2 19.93 mHa, lambda_eff 48.204. Built with mrmeows' exact tensor lookup
compiler (J = 5, 15 ANGLE circuits) the official verifier gives **PASS, cost 3.97565e8 = 8.440x** (T 19,136, Q 431).
This chemistry scored **20.05029** on the public leaderboard together with mrmeows' 63-qubit invsqrt and the public
883-qubit Schur DQI circuit; see the notebook `michaelrustad/vqcc-r10-c46-chemistry-20-05-lb`.

Recipe changes since version 1:

1. Copy truncation: `src/ctrunc.py` reduces C 60 -> 48 -> 46 (per-group Gram eigen-truncation), each followed by a
   long L-BFGS refit (`src/cont_lbfgs.py`, D penalty 2e-5); continuous error_2 12.81 mHa at C = 46.
2. **Hybrid quantization**: freezing the highest-impact angles first costs the most, and short Adam compensation never
   recovers it. `src/pq_lb.py` freezes the first 10% of angles in fine steps (0.5, 1, 2, 3.5, 5, 7.5, 10%) with long
   L-BFGS compensation between freezes, then `src/pq.py --resume STATE` freezes the rest in 64 fine Adam stages.
   On R10/C60 this lowers the quantized error_2 from 19.56 to 17.86 mHa; it is what makes C = 46 (and R9/C72) fit.
3. `src/qat.py` decodes the Givens tree in node-disjoint layers (7 batched layers, exact, about 4.7x faster).

The ANGLE circuits for version 2 are compiled at run time by the notebook from
[mrmeows/qcc-1375-open-circuits](https://www.kaggle.com/datasets/mrmeows/qcc-1375-open-circuits)
(`alias_compiler/support/generic_lookup.py`, Apache-2.0; four-address loader lineage marculera); they are not part of our code: the unchanged Apache-2.0 compiler files are vendored in
`third_party/mrmeows_lookup_compiler/` (with LICENSE and NOTICE) because later versions of that dataset moved them.

## Version 4: R10/B129/C46 at lambda_eff 48.127, four-word lookups (9.094x, LB 21.41)

`fact/dfthc_r10_b129_c46_lam48127_canon.npz` (SHA-256 `e5ee2f8c9ab25cd045a0ad4d707594e4d3b1e64e829669a4d544765d79fb13e9`):
the same C46 factorization after a longer `src/lamopt.py` run (error_2 cap 19.96 mHa, `--e2-tol 1e-5 --gap-margin 2e-6
--gap-feas 5e-7`), `src/finish.py` and `src/gauge.py` (lambda_eff 48.204 -> 48.127, error_2 19.97 mHa), then the exact
sign gauge `src/canon_gauge.py` (two top bits of every word agree; metrics unchanged). Compiled with mrmeows'
four-word streamed lookup and GF(2) product-sharing planner (J = 4, 19 circuits, 274-275 Toffolis each, vendored
unchanged with its Apache-2.0 licence in `third_party/mrmeows_streamed_planner/`), the official verifier gives
**PASS, cost 3.69005e8 = 9.094x** (T 18,126, Q 423). With mrmeows' 63-qubit invsqrt and SomeJoint's confluent DQI
decoder this scored **21.40999**; see the notebook `michaelrustad/vqcc-r10-c46-four-word-chemistry-21-41-lb`.

## Version 5: R10/B129/C44 at lambda_eff 47.549 (9.576x, LB 21.98)

`fact/dfthc_r10_b129_c44_lam47549_canon.npz` (SHA-256 `6918e6c3b252b02dc15cf83e082553d35e0f13ce635fe9bdafe06daed3658606`): (R, B, C) = (10, 129, 44), brot 8,
error_2 19.99 mHa, Lambda 290.57, lambda_eff 47.549, canonical sign gauge. Recipe in `src/chain_c44.sh`:

1. Copy truncation 46 -> 44 (`src/ctrunc.py`) with a 6000-iteration L-BFGS refit (continuous error_2 13.31 mHa).
2. Quantization (`src/pq.py`) with 36 explicit freeze fractions, Gauss-Newton impact ranking of the angles to freeze
   (`--gn`), learning-rate decay within each stage, best-iterate keeping, long stages
   (`--split 0.05,0.3,12000,9000,2000`) and a stronger D(w) penalty during compensation (`--dpen 6e-5`):
   error_2 19.86 mHa. The D(w) penalty is what lowers Lambda (295 -> 290.6) and therefore lambda_eff; with the
   default 2e-5 the same schedule ends at lambda_eff 47.89-47.92, and above 6e-5 error_2 no longer fits under 20 mHa.
3. finish -> gauge -> `src/lamopt.py --decay 0.01` (exponential learning-rate decay) -> finish -> gauge -> canon_gauge.

Compiled with the five-word (J = 5) lookup used in Ahmed Berat Ozer's public v9 portfolio (15 circuits), the official
verifier gives **PASS, cost 3.50414e8 = 9.576x** (T 17,139, Q 430). With the public 63-qubit invsqrt and 877-qubit
DQI circuits of that portfolio this scored **21.97849** on the public leaderboard.

## Credits and terms

- Factorization lineage: the public (65, 60, 30) DF-THC factorization and
  [Ahmed Berat Ozer's DF-THC Group Compression](https://www.kaggle.com/datasets/ahmedberatozer/df-thc-group-compression-factorization)
  (26, 60, 30) file. This data is derived from the competition reference data and that dataset; their terms apply.
- Banked QROM / ANF eraser: Kozykorpesh, Sumit Chongder, Kirill Tymchishin, anazemcev (Apache-2.0), via
  Ahmed Berat Ozer's `gauge_qrom` (Apache-2.0).
- Effective-norm and gauge ideas: the public chemistry refinement notebooks; reimplemented here.
- All code in `src/` written for this dataset is Apache-2.0 (see `LICENSE`).
