# Pro's 897-qubit decoder with the 658-Toffoli jet

This is the decoder folder of our second decoder kit, unchanged, with one more file:

- **The kit's decoder** (`decode.py`, `exp/`, `playbook/`): our 883-qubit decoder before Ahmed Berat Özer's narrowings, with the 658-Toffoli jet in `exp/jet24/pro_jet.py`.
- **`final_decoder/`:** GPT-6 Astra Pro's three files, unchanged: `decode.py`, `conversion.py` and `synthesis.py`. They import the kit's modules.
- **`build_897.py`:** the build script.

Build with `python build_897.py OUTPUT.kmb`, with Kickmix on the path. It writes `dqi.kmb`, SHA-256 `b3354f60e459dcebe67910363e96498f8bb0aa7f699bc016ef962015c3732fd6`, and `dqi.build.json` with its counts. It takes about a minute and 5 GB.
