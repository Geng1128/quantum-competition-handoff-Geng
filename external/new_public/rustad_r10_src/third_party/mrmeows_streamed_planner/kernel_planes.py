#!/usr/bin/env python3
"""Exact multi-output product sharing on a single organizer-only fixture.

The search chooses a small subspace of the coefficient tensor's exact kernel.
Rank-two product templates are admitted only when BOTH of their coefficient
columns annihilate those relations. A greedy paired basis search is accepted
only after exact reconstruction of every tensor coefficient. No randomized
rank test or chemistry approximation is used as an acceptance criterion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np

import rank_parity as core


def packed(row):
    return sum(int(value) << j for j,value in enumerate(row))


def tensor_column(direction,lowform,m):
    return sum(direction << (m*j) for j in core.set_bits(lowform))


def low_planes():
    planes = set()
    def add(a,b):
        first,second,_ = sorted((a,b,a^b))
        planes.add((first,second))
    for shift in (0,3):
        for a in range(1,8):
            for b in range(a+1,8):
                add(a << shift,b << shift)
    for a in range(1,8):
        for b in range(1,8):
            add(a,b << 3)
        add(a,1 << 6)
        add(a << 3,1 << 6)
    return sorted(planes)


def reduce_row(value,pivots):
    while value:
        p = value.bit_length()-1
        if p not in pivots:
            break
        value ^= pivots[p]
    return value


def append_row(value,pivots):
    value = reduce_row(value,pivots)
    if not value:
        return False
    pivots[value.bit_length()-1] = value
    return True


def select_relations(rows,width,dimension,rng):
    kernel = core.nullspace(rows,width)
    if len(kernel) < dimension:
        raise ValueError('requested kernel dimension exceeds the actual nullity')
    chosen = []
    span = {}
    while len(chosen) < dimension:
        mask = rng.getrandbits(len(kernel))
        value = 0
        for j in core.set_bits(mask):
            value ^= kernel[j]
        if append_row(value,span):
            assert all((value & row).bit_count()%2 == 0 for row in rows)
            chosen.append(value)
    return chosen,len(kernel)


def paired_basis(templates,target_rank,seed,trials):
    rng = random.Random(seed)
    best = 0
    reports = []
    for trial in range(trials):
        order = list(range(len(templates)))
        rng.shuffle(order)
        span = {}
        chosen = []
        for i in order:
            a,b,*_ = templates[i]
            ar = reduce_row(a,span)
            if not ar:
                continue
            p = ar.bit_length()-1
            span[p] = ar
            br = reduce_row(b,span)
            if not br:
                del span[p]
                continue
            span[br.bit_length()-1] = br
            chosen.append(i)
            if len(span) == target_rank:
                reports.append({'trial':trial,'rank':len(span)})
                return chosen,reports
        best = max(best,len(span))
        reports.append({'trial':trial,'rank':len(span)})
    return None,reports


def plan(delta,brot,dimensions=(15,15),seed=0,pair_trials=16,samples=12):
    core.require_redundant_sign_bit(delta,brot)
    ac = core.coefficients(delta,brot,3)
    tensor = ac[1:,1:,:]
    H,L,m = tensor.shape
    if L != 7:
        raise ValueError('this bounded planner requires exactly three low address bits')
    F = tensor.reshape(H,L*m)
    rows = [packed(row) for row in F]
    rng = random.Random(seed)
    relations = []
    nullities = []
    for group,dimension in enumerate(dimensions):
        block_rows = [packed(row) for row in F[:,3*m*group:3*m*(group+1)]]
        chosen,nullity = select_relations(block_rows,3*m,dimension,rng)
        relations += [value << (3*m*group) for value in chosen]
        nullities.append(nullity)
    assert all((row & relation).bit_count()%2 == 0 for row in rows for relation in relations)
    mask = (1 << m)-1
    chunks = [[(r >> (j*m)) & mask for j in range(L)] for r in relations]
    templates = []
    plane_dimensions = []
    for ell1,ell2 in low_planes():
        equations = []
        for row in chunks:
            for ell in (ell1,ell2):
                value = 0
                for j in core.set_bits(ell):
                    value ^= row[j]
                if value:
                    equations.append(value)
        V = core.nullspace(equations,m)
        plane_dimensions.append(len(V))
        directions = set(V)
        for _ in range(samples):
            choice = rng.getrandbits(len(V))
            value = 0
            for j in core.set_bits(choice):
                value ^= V[j]
            if value:
                directions.add(value)
        for direction in sorted(directions):
            a = tensor_column(direction,ell1,m)
            b = tensor_column(direction,ell2,m)
            assert all((a&r).bit_count()%2 == 0 and (b&r).bit_count()%2 == 0 for r in relations)
            templates.append((a,b,direction,ell1,ell2))
    all_span = {}
    for a,b,*_ in templates:
        append_row(a,all_span)
        append_row(b,all_span)
    target_rank = L*m-sum(dimensions)
    contains_target = all(not reduce_row(row,all_span) for row in rows)
    report = {'H':H,'L':L,'O':m,'kernel_dimensions':list(dimensions),
        'kernel_nullities':nullities,'kernel_relations':relations,
        'num_templates':len(templates),'template_span_rank':len(all_span),
        'desired_span_rank':target_rank,'contains_target':contains_target,
        'plane_output_nullities':plane_dimensions,'seed':seed}
    if len(all_span) != target_rank or not contains_target:
        return None,report
    indices,attempts = paired_basis(templates,target_rank,seed+1001,pair_trials)
    report['paired_basis_attempts'] = attempts
    if indices is None:
        return None,report
    chosen = [templates[i] for i in indices]
    basis = core.BinaryBasis()
    for a,b,*_ in chosen:
        assert basis.add(a) and basis.add(b)
    solutions = [basis.solve(row) for row in rows]
    assert all(value is not None for value in solutions)
    terms = []
    reconstructed = [0]*H
    for i,(a,b,direction,ell1,ell2) in enumerate(chosen):
        h1 = sum(((value >> (2*i))&1) << h for h,value in enumerate(solutions))
        h2 = sum(((value >> (2*i+1))&1) << h for h,value in enumerate(solutions))
        terms.append({'target_direction':direction,'factors':[(h1,ell1),(h2,ell2)]})
        for h in core.set_bits(h1):
            reconstructed[h] ^= a
        for h in core.set_bits(h2):
            reconstructed[h] ^= b
    assert reconstructed == rows
    report.update({'selected_pairs':len(chosen),'exact_selected_binary_rank':len(basis.vectors),
        'exact_tensor_reconstruction':True,'selected_template_indices':indices})
    return (ac,terms),report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture-dir',type=Path,required=True)
    p.add_argument('--dimensions',type=int,nargs=2,default=(15,15))
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--pair-trials',type=int,default=16)
    p.add_argument('--samples',type=int,default=12)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--verifier-dir',type=Path,default=Path('/tmp/chem_verifiers/dfthcsos'))
    a = p.parse_args()
    start = time.time()
    if a.out.exists():
        raise FileExistsError('refusing to overwrite an experiment')
    source = json.loads((a.fixture_dir/'report.json').read_text())
    if source['scope'] != 'ORGANIZER_SUBSET_CIRCUIT_FIXTURE_ONLY':
        raise ValueError('this bounded experiment requires the organizer fixture')
    delta = np.load(a.fixture_dir/'batch_delta.npy',allow_pickle=False)
    result,planning = plan(delta,source['brot'],tuple(a.dimensions),a.seed,a.pair_trials,a.samples)
    a.out.mkdir(parents=True)
    if result is None:
        report={'status':'NO_EXACT_PAIRED_PLAN','planning':planning,'seconds':time.time()-start}
        (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k!='planning'}))
        print(json.dumps({k:v for k,v in planning.items() if k not in ['kernel_relations','plane_output_nullities']}))
        return
    ac,terms = result
    circuit,trace,report = core.build(delta,source['brot'],3,ac,terms)
    report['exact_symbolic_oracle'] = core.symbolic_check(trace,delta,source['brot'],circuit.num_qubits)
    report['native_oracle'] = core.native_check(circuit,delta,source['brot'],a.verifier_dir)
    cp = a.out/'angle_k_0.kmb'
    circuit.to_file(cp,format='kmb')
    (a.out/'angle_k_0.kmx').write_text(str(circuit))
    (a.out/'certificate.json').write_text(json.dumps({'planning':planning,'terms':terms},indent=2)+'\n')
    report.update({'status':'PASS','scope':'ORGANIZER_SUBSET_CIRCUIT_FIXTURE_ONLY',
        'chemistry_acceptance_claimed':False,'full_family_compiled':False,
        'D':len(delta),'brot':source['brot'],'batch_size':source['batch_size'],
        'planning':planning,'fixture_source_report_sha256':core.sha(a.fixture_dir/'report.json'),
        'fixture_delta_sha256':core.sha(a.fixture_dir/'batch_delta.npy'),
        'planner_sha256':core.sha(__file__),'emitter_sha256':core.sha(core.__file__),
        'proof_helper_sha256':core.sha(Path(__file__).with_name('borrowed_prefix.py')),
        'native_sha256':core.sha(cp),'seconds':time.time()-start})
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='planning'},indent=2))
    print(json.dumps({k:v for k,v in planning.items() if k not in ['kernel_relations','plane_output_nullities']},indent=2))


if __name__=='__main__':
    main()
