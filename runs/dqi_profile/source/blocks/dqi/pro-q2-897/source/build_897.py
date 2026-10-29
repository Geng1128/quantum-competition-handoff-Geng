"""Builds Pro's q2 decoder with the 658-Toffoli jet: python build_897.py OUTPUT.kmb, with Kickmix on the path.

It writes OUTPUT.kmb, SHA-256 b3354f60e459dcebe67910363e96498f8bb0aa7f699bc016ef962015c3732fd6, and OUTPUT.build.json
with its counts. It builds the way pro/q2-decoder/local/official.sh does: the entry final_decoder/decode.py, with
Kickmix as the kit's builder module circ, and the registers S0 to S69 and E0 to E4094 named in the file.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import kickmix as km

sys.modules['circ'] = km
HERE = Path(__file__).resolve().parent
target = Path(sys.argv[1])
start = time.time()
spec = importlib.util.spec_from_file_location('decoder_under_test', HERE / 'final_decoder/decode.py')
module = importlib.util.module_from_spec(spec)
sys.modules['decoder_under_test'] = module
spec.loader.exec_module(module)
b = km.CircuitBuilder()
S = [b.alloc_qubits(12) for _ in range(70)]
E = [b.alloc_bits(12) for _ in range(4095)]
module.decode(b, S, E)
metadata = []
for i in range(70):
    metadata.append(f'REGISTER r{i} "S{i}"')
    metadata.extend(f'APPEND_TO_REGISTER q{12 * i + k} r{i}' for k in range(12))
for j in range(4095):
    metadata.append(f'REGISTER r{70 + j} "E{j}"')
    metadata.extend(f'APPEND_TO_REGISTER b{12 * j + k} r{70 + j}' for k in range(12))
c = km.Circuit('\n'.join(metadata)) + b.finish_circuit()
c.to_file(str(target), format='kmb')
with open(target, 'rb') as f:
    digest = hashlib.file_digest(f, 'sha256').hexdigest()
record = dict(sha256=digest, Q=c.num_qubits, T_static=c.operation_statistics().num_toffoli_like, num_ops=len(c),
              num_bits=c.num_bits, seconds=round(time.time() - start, 1))
target.with_suffix('.build.json').write_text(json.dumps(record, indent=1) + '\n')
print(json.dumps(record), flush=True)
