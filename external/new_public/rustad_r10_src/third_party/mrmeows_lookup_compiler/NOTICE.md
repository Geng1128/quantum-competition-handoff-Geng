# Third-party code: mrmeows exact lookup compiler (Apache-2.0)

These files are copied unchanged from version 1 of the dataset
[mrmeows/qcc-1375-open-circuits](https://www.kaggle.com/datasets/mrmeows/qcc-1375-open-circuits)
(`source/chemistry_research/alias_compiler/support/*.py` and `source/chemistry_four_address/*.py`), released by
mrmeows under the Apache License 2.0 (see LICENSE). The four-address loader lineage (paired combine and bank erasure in `four_address.py`, adapted by mrmeows) is
marculera's [dfthc-merge-pipeline](https://www.kaggle.com/datasets/marculera/dfthc-merge-pipeline) (`dfthc_refit/fourqrom.py`,
Apache-2.0); file headers that mention FOUR_ADDRESS_NOTICE.md refer to that attribution. They are included so that the notebook
`michaelrustad/vqcc-r10-c46-chemistry-20-05-lb` keeps working after that dataset's later versions moved them.
Entry point: `generic_lookup.py --submission FACT.npz --brot 8 --batch-size 5 --lowbits 6 --count 99 --out DIR`.
