#!/usr/bin/env python3
"""Organizer-only coherent prefix lookup with restored dirty output helpers.

The source table is required explicitly. This file imports no fitted data or
previous decomposition certificate. It only emits X, CX and CCX gates.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mobius(table):
    result = np.asarray(table, dtype=np.int64).copy()
    ids = np.arange(len(result))
    for bit in range((len(result)-1).bit_length()):
        rows = ids[(ids & (1 << bit)) != 0]
        result[rows] ^= result[rows ^ (1 << bit)]
    return result


def delta_batch(words, size, k):
    def part(index):
        result = np.zeros((len(words), size), dtype=np.int64)
        if index >= 0:
            start = index * size
            width = max(0, min(size, words.shape[1] - start))
            result[:, :width] = words[:, start:start+width]
        return result
    return part(k) ^ part(k-1)


def build(delta, brot, clean_cap):
    import kickmix as km
    count, size = delta.shape
    n = (count-1).bit_length()
    coefficients = mobius(delta)
    assert np.array_equal(mobius(coefficients), delta)
    used = set(map(int, np.flatnonzero(np.any(coefficients != 0, axis=1))))
    for mask in tuple(used):
        while mask:
            mask ^= 1 << (mask.bit_length()-1)
            if mask:
                used.add(mask)
    children = {mask: [] for mask in used if mask}
    for mask in sorted(used):
        if mask.bit_count() > 1:
            children[mask ^ (1 << (mask.bit_length()-1))].append(mask)

    builder = km.CircuitBuilder()
    qubits = list(builder.create_quantum_register(n, 'selection'))
    for j in range(size):
        qubits += list(builder.create_quantum_register(brot, f'word_{j}'))
    target_ids = list(range(n, n + brot*size))
    scratch_start = len(qubits)
    qubits += list(builder.alloc_qubits(clean_cap))
    if len(qubits) > 128:
        raise ValueError(f'layout requires {len(qubits)} qubits')
    trace = []
    fallback = []

    def gate(name, *ids):
        trace.append((name, *ids))
        getattr(builder, name.lower())(*[qubits[x] for x in ids])

    def targets(mask):
        return [n + j*brot + bit for j in range(size)
                for bit in range(brot) if int(coefficients[mask,j]) >> bit & 1]

    def multi_x(controls, target):
        k = len(controls)
        if k <= 2:
            gate(('X', 'CX', 'CCX')[k], *controls, target)
            return
        dirty = [q for q in target_ids if q != target][:k-2]
        if len(dirty) != k-2:
            raise ValueError('not enough distinct borrowed output bits')
        # B = U CCX(c0,c1,d0) U^-1 toggles the last dirty bit by
        # product(c0,...,c[k-2]). T B T B then toggles only the target.
        # There are 4*k-8 CCX gates, and every dirty bit is restored.
        def bridge():
            for i in reversed(range(1,k-2)):
                gate('CCX', dirty[i-1], controls[i+1], dirty[i])
            gate('CCX', controls[0], controls[1], dirty[0])
            for i in range(1,k-2):
                gate('CCX', dirty[i-1], controls[i+1], dirty[i])
        gate('CCX', dirty[-1], controls[-1], target)
        bridge()
        gate('CCX', dirty[-1], controls[-1], target)
        bridge()

    def fan_product(controls, outputs):
        if not outputs:
            return
        pivot, *others = outputs
        for q in others:
            gate('CX', pivot, q)
        multi_x(controls, pivot)
        for q in others:
            gate('CX', pivot, q)

    def direct_subtree(mask, ancestor, control):
        outputs = targets(mask)
        if outputs:
            rest = mask ^ ancestor
            controls = [control] + [bit for bit in range(n) if rest >> bit & 1]
            fan_product(controls, outputs)
            fallback.append({'mask': mask, 'controls':len(controls),
                             'ccx': 1 if len(controls)==2 else 4*len(controls)-8})
        for child in children[mask]:
            direct_subtree(child, ancestor, control)

    def visit(mask, control):
        for q in targets(mask):
            gate('CX', control, q)
        for child in children[mask]:
            new_bit = child.bit_length()-1
            if not children[child]:
                fan_product([control, new_bit], targets(child))
            elif child.bit_count()-1 <= clean_cap:
                temporary = scratch_start + child.bit_count()-2
                gate('CCX', control, new_bit, temporary)
                visit(child, temporary)
                gate('CCX', control, new_bit, temporary)
            else:
                direct_subtree(child, mask, control)

    for q in targets(0):
        gate('X', q)
    for bit in range(n):
        if 1 << bit in used:
            visit(1 << bit, bit)
    circuit = builder.finish_circuit()
    num_ccx = sum(t[0] == 'CCX' for t in trace)
    assert int(circuit.operation_statistics().num_toffoli_like) == num_ccx
    return circuit, trace, {'num_qubits':int(circuit.num_qubits),
        'toffoli_count':num_ccx, 'clean_scratch':clean_cap,
        'nonlinear_anf_terms':sum(m.bit_count()>1 for m in used),
        'fallback_products':fallback, 'gates':len(trace)}


def symbolic_check(trace, delta, brot, num_qubits):
    """Exact address-packed affine labels cover every arbitrary target state."""
    count, size = delta.shape
    n = (count-1).bit_length()
    all_addresses = (1 << count)-1
    def packed(bits):
        return int.from_bytes(np.packbits(np.asarray(bits,dtype=np.uint8),
                                         bitorder='little').tobytes(), 'little')
    addresses = np.arange(count, dtype=np.int64)
    states = [{-1:packed(addresses >> bit & 1)} for bit in range(n)]
    states += [{i:all_addresses} for i in range(size*brot)]
    states += [{} for _ in range(num_qubits-len(states))]
    def add(target, source):
        for label, bits in source.items():
            value = states[target].get(label,0) ^ bits
            if value:
                states[target][label] = value
            else:
                states[target].pop(label,None)
    for instruction in trace:
        name, *q = instruction
        if name == 'X':
            add(q[0], {-1:all_addresses})
        elif name == 'CX':
            add(q[1], states[q[0]])
        elif name == 'CCX':
            left, right = states[q[0]], states[q[1]]
            if any(k >= 0 for k in left):
                left, right = right, left
            if any(k >= 0 for k in left):
                raise AssertionError('unexpected target-quadratic intermediate')
            control = left.get(-1,0)
            add(q[2], {key:value & control for key,value in right.items()})
        else:
            raise AssertionError(f'non-classical gate {name}')
    for bit in range(n):
        assert states[bit] == {-1:packed(addresses >> bit & 1)}
    for j in range(size):
        for bit in range(brot):
            expected = {j*brot+bit:all_addresses}
            value = packed(delta[:,j] >> bit & 1)
            if value:
                expected[-1] = value
            assert states[n+j*brot+bit] == expected
    assert all(not s for s in states[n+size*brot:])
    return {'status':'PASS', 'all_valid_addresses':count,
            'arbitrary_target_bits':size*brot,
            'target_domain':f'all 2**{size*brot} assignments',
            'all_clean_scratch_restored':True,
            'all_selection_bits_restored':True,
            'zero_phase':'Only unconditional X/CX/CCX permutation gates'}


def native_check(circuit, delta, brot, verifier, repeats=2):
    import kickmix as km
    path = Path(verifier)/'scoreinfra/circuits.py'
    spec = importlib.util.spec_from_file_location('_organizer_lookup_contract',path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    layout = module.register_layout([circuit])
    count = len(delta)
    addresses = np.tile(np.arange(count,dtype=np.int64),repeats)
    rng = np.random.default_rng(902113)
    initial = rng.integers(0,1 << brot,size=(len(addresses),delta.shape[1]))
    bits = ((addresses[None,:] >> np.arange(len(layout.selection))[:,None]) & 1).astype(bool)
    sim = km.Simulator(batch_size=len(addresses),debug_print=False,seed=902113,
                       counting_operations=True)
    sim.use_same_registers_as(circuit)
    sim.write_across_shots(layout.selection,bits)
    for j,reg in enumerate(layout.words):
        sim.write_across_shots(reg,((initial[:,j][None,:] >> np.arange(brot)[:,None]) & 1).astype(bool))
    sim.do(circuit)
    module._check_state(sim,circuit,layout,bits,'organizer borrowed-prefix probe')
    powers = np.int64(1) << np.arange(brot)
    actual = np.stack([powers @ sim.read_across_shots(reg).astype(np.int64)
                       for reg in layout.words],axis=1)
    assert np.array_equal(actual, initial ^ delta[addresses])
    assert sim.operation_statistics().num_toffoli_like == len(addresses)*circuit.operation_statistics().num_toffoli_like
    return {'status':'PASS','actual_native_arbitrary_target_shots':len(addresses),
            'scratch_phase_selection_checked':True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--angle-words',type=Path,required=True)
    p.add_argument('--source-factorization',type=Path,required=True)
    p.add_argument('--brot',type=int,default=12)
    p.add_argument('--batch-size',type=int,default=9)
    p.add_argument('--k',type=int,default=0)
    p.add_argument('--clean-cap',type=int,default=8)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--verifier-dir',type=Path,default=Path('/tmp/chem_verifiers/dfthcsos'))
    a = p.parse_args()
    start = time.time()
    if a.out.exists():
        raise FileExistsError('refusing to overwrite a probe directory')
    words = np.load(a.angle_words,allow_pickle=False)
    if isinstance(words,np.lib.npyio.NpzFile):
        words = words['angle_words'].copy()
    if np.any(words < 0) or np.any(words >= 1 << a.brot):
        raise ValueError('word outside the selected precision')
    delta = delta_batch(words,a.batch_size,a.k)
    circuit, trace, report = build(delta,a.brot,a.clean_cap)
    report['symbolic_oracle'] = symbolic_check(trace,delta,a.brot,circuit.num_qubits)
    report['native_oracle'] = native_check(circuit,delta,a.brot,a.verifier_dir)
    a.out.mkdir(parents=True)
    cp = a.out/f'angle_k_{a.k}.kmb'
    circuit.to_file(cp,format='kmb')
    (a.out/f'angle_k_{a.k}.kmx').write_text(str(circuit))
    np.save(a.out/'batch_delta.npy',delta)
    report.update({'scope':'ORGANIZER_ONLY_EXACT_CIRCUIT_COMPONENT',
        'full_family_compiled':False,'chemistry_acceptance_claimed':False,
        'brot':a.brot,'batch_size':a.batch_size,'k':a.k,
        'source_factorization':str(a.source_factorization.resolve()),
        'source_factorization_sha256':sha(a.source_factorization),
        'angle_words':str(a.angle_words.resolve()),'angle_words_sha256':sha(a.angle_words),
        'compiler_sha256':sha(__file__),'native_sha256':sha(cp),
        'seconds':time.time()-start})
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
