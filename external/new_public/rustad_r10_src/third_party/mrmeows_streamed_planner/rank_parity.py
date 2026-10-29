#!/usr/bin/env python3
"""Independent coherent lookup via an even-rank output-function basis.

All emitted gates are unconditional X/CX/CCX. No decomposition certificate or
candidate data is imported. The optional organizer subset is a circuit fixture,
not a chemistry submission. A rank-one matrix count is never reported as the
native paired-product count.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from borrowed_prefix import delta_batch, native_check, symbolic_check


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def set_bits(value):
    while value:
        bit = (value & -value).bit_length()-1
        yield bit
        value &= value-1


def mobius_axis(values, axis):
    out = np.swapaxes(values.copy(),axis,0)
    ids = np.arange(len(out))
    for bit in range((len(out)-1).bit_length()):
        rows = ids[(ids & (1 << bit)) != 0]
        out[rows] ^= out[rows ^ (1 << bit)]
    return np.swapaxes(out,0,axis)


def canonical_gauge(words, edges, brot):
    """Exact integer projective gauge; all cosines use a positive half-turn."""
    out = np.asarray(words,dtype=np.int64).copy()
    signs = np.ones((len(words),len(edges)+1),dtype=np.int8)
    modulus = 1 << brot
    for k in reversed(range(len(edges))):
        p,q = map(int,edges[k])
        x = out[:,k].copy()
        opposite = signs[:,p] != signs[:,q]
        x[opposite] = (-x[opposite]) % modulus
        flip = (x >= modulus//4) & (x < 3*modulus//4)
        x[flip] = (x[flip]+modulus//2) % modulus
        signs[flip,p] *= -1
        signs[:,q] = 1
        out[:,k] = x
    assert not np.any(((out >> (brot-1)) ^ (out >> (brot-2))) & 1)
    return out, signs[:,0]


def decode(words,edges,brot):
    v = np.zeros((len(words),len(edges)+1))
    v[:,0] = 1
    for k,(p,q) in enumerate(edges):
        theta = words[:,k]*(np.pi/(1 << (brot-1)))
        left,right = v[:,p].copy(),v[:,q].copy()
        v[:,p] = np.cos(theta)*left - np.sin(theta)*right
        v[:,q] = np.sin(theta)*left + np.cos(theta)*right
    return v


class BinaryBasis:
    def __init__(self):
        self.pivots = {}
        self.vectors = []

    def add(self,value):
        original = value
        coordinates = 1 << len(self.vectors)
        while value:
            p = value.bit_length()-1
            if p not in self.pivots:
                self.pivots[p] = (value,coordinates)
                self.vectors.append(original)
                return True
            v,c = self.pivots[p]
            value ^= v
            coordinates ^= c
        return False

    def solve(self,value):
        coordinates = 0
        while value:
            p = value.bit_length()-1
            if p not in self.pivots:
                return None
            v,c = self.pivots[p]
            value ^= v
            coordinates ^= c
        return coordinates


def nullspace(rows, width):
    pivots = {}
    for value in rows:
        value = int(value)
        while value:
            p = value.bit_length()-1
            if p not in pivots:
                pivots[p] = value
                if len(pivots) == width:
                    return []
                break
            value ^= pivots[p]
    result = []
    for j in range(width):
        if j not in pivots:
            value = 1 << j
            for p,row in sorted(pivots.items()):
                if (row & value).bit_count() & 1:
                    value ^= 1 << p
            result.append(value)
    return result


def factor_matrix(columns):
    basis = BinaryBasis()
    for col in columns:
        basis.add(col)
    lows = [0]*len(basis.vectors)
    for k,col in enumerate(columns):
        coordinates = basis.solve(col)
        assert coordinates is not None
        for bit in set_bits(coordinates):
            lows[bit] ^= 1 << k
    return list(zip(basis.vectors,lows))


def require_redundant_sign_bit(delta,brot):
    """Reject raw tables that cannot be represented by b-1 Boolean functions.

    Every physical word bit remains an arbitrary target. This precondition is
    about the requested XOR values, not an assumption about target contents.
    The caller must explicitly apply a legal projective gauge beforehand.
    """
    if not isinstance(brot,(int,np.integer)) or not 2 <= brot <= 62:
        raise ValueError('brot must be an integer between 2 and 62')
    table = np.asarray(delta)
    if (table.ndim != 2 or min(table.shape,default=0) < 1 or
            not np.issubdtype(table.dtype,np.integer)):
        raise ValueError('delta must be a nonempty two-dimensional integer table')
    if np.any(table < 0) or np.any(table >= (1 << brot)):
        raise ValueError('delta word outside the selected precision')
    table = table.astype(np.int64,copy=False)
    if np.any(((table >> (brot-1)) ^ (table >> (brot-2))) & 1):
        raise ValueError('redundant sign-bit precondition failed: the two highest '
                         'bits of every requested XOR word must be equal')
    return table


def coefficients(delta,brot,lowbits):
    delta = require_redundant_sign_bit(delta,brot)
    count,J = delta.shape
    low = 1 << lowbits
    high = (count+low-1)//low
    padded = np.zeros((low*high,J),dtype=np.int64)
    padded[:count] = delta
    require_redundant_sign_bit(padded,brot)
    bits = np.stack([(padded[:,j] >> bit)&1 for j in range(J)
                     for bit in range(brot-1)],axis=1).astype(np.uint8)
    truth = bits.reshape(high,low,-1)
    result = mobius_axis(mobius_axis(truth,0),1)
    assert np.array_equal(mobius_axis(mobius_axis(result,1),0),truth)
    return result


def plan(delta,brot,lowbits,max_kernels=32767):
    ac = coefficients(delta,brot,lowbits)
    tensor = ac[1:,1:,:]
    H,L,O = tensor.shape
    columns = [[sum(int(tensor[h,k,j]) << h for h in range(H))
                for k in range(L)] for j in range(O)]
    original_rank = [len(factor_matrix(cols)) for cols in columns]
    chosen = BinaryBasis()
    certificates = []
    # Preserve existing already-even or unusually-low rank directions first.
    for j,r in sorted(enumerate(original_rank),key=lambda item:(item[1]+1)//2):
        if r < min(H,L) or r % 2 == 0:
            if chosen.add(1 << j):
                certificates.append({'direction':1 << j,'kernel':None,'rank':r})
    packed_low = [[sum(int(tensor[h,k,j]) << j for j in range(O))
                   for h in range(H)] for k in range(L)]
    current = [0]*H
    previous = 0
    searched = 0
    for counter in range(1,min((1 << L)-1,max_kernels)+1):
        mask = counter ^ (counter >> 1)
        changed = (mask ^ previous).bit_length()-1
        current = [a ^ b for a,b in zip(current,packed_low[changed])]
        previous = mask
        searched += 1
        for direction in nullspace(current,O):
            if chosen.solve(direction) is not None:
                continue
            mixture = [0]*L
            for j in set_bits(direction):
                mixture = [a ^ b for a,b in zip(mixture,columns[j])]
            rank = len(factor_matrix(mixture))
            # A rank drop must actually cross a paired-product threshold.
            if (rank+1)//2 < (min(H,L)+1)//2:
                assert chosen.add(direction)
                certificates.append({'direction':direction,'kernel':mask,'rank':rank})
        if len(chosen.vectors) == O:
            break
    for j in range(O):
        if chosen.add(1 << j):
            certificates.append({'direction':1 << j,'kernel':None,
                                 'rank':original_rank[j]})
    assert len(chosen.vectors) == O
    output_directions = [0]*O
    for j in range(O):
        inverse_column = chosen.solve(1 << j)
        assert inverse_column is not None
        for i in set_bits(inverse_column):
            output_directions[i] |= 1 << j
    terms = []
    reconstructed = [[0]*L for _ in range(O)]
    products = 0
    for i,vector in enumerate(chosen.vectors):
        mixture = [0]*L
        for j in set_bits(vector):
            mixture = [a ^ b for a,b in zip(mixture,columns[j])]
        factors = factor_matrix(mixture)
        assert len(factors) == certificates[i]['rank']
        terms.append({'input_output_basis':vector,
                      'target_direction':output_directions[i],
                      'factors':factors})
        products += (len(factors)+1)//2
        for j in set_bits(output_directions[i]):
            reconstructed[j] = [a ^ b for a,b in zip(reconstructed[j],mixture)]
    assert reconstructed == columns
    return ac,terms,{'H':H,'L':L,'O':O,'searched_nonzero_low_kernels':searched,
        'original_matrix_ranks':original_rank,'output_basis_certificates':certificates,
        'untransformed_paired_products':sum((r+1)//2 for r in original_rank),
        'transformed_paired_products':products,'exact_mixed_tensor_reconstruction':True}


def build(delta,brot,lowbits,ac,terms):
    import kickmix as km
    delta = require_redundant_sign_bit(delta,brot)
    D,J = delta.shape
    n = (D-1).bit_length()
    A,B,O = ac.shape
    H,L = A-1,B-1
    builder = km.CircuitBuilder()
    qubits = list(builder.create_quantum_register(n,'selection'))
    for j in range(J):
        qubits += list(builder.create_quantum_register(brot,f'word_{j}'))
    trace = []
    def gate(name,*ids):
        trace.append((name,*ids))
        getattr(builder,name.lower())(*[qubits[i] for i in ids])
    cache = []
    def cached_basis(size,offset):
        result = {0:None}
        for mask in range(1,size):
            bit = mask.bit_length()-1
            if mask.bit_count()==1:
                result[mask] = offset+bit
            else:
                parent = mask ^ (1 << bit)
                q = len(qubits)
                qubits.extend(builder.alloc_qubits(1))
                result[mask] = q
                triple = (result[parent],offset+bit,q)
                gate('CCX',*triple)
                cache.append(triple)
        return result
    high = cached_basis(A,lowbits)
    low = cached_basis(B,0)
    if len(qubits)>128:
        raise ValueError(f'cache layout needs {len(qubits)} qubits')
    basis = [high[i] for i in range(1,A)]+[low[i] for i in range(1,B)]
    groups = [[n+j*brot+bit] + ([n+j*brot+brot-1] if bit==brot-2 else [])
              for j in range(J) for bit in range(brot-1)]
    def target_bits(direction):
        return [q for j in set_bits(direction) for q in groups[j]]
    def linear_output(control,direction):
        for q in target_bits(direction):
            gate('CX',control,q)
    def product(a,b,direction):
        if not a or not b or not direction:
            return
        if a == b:
            for i in set_bits(a):
                linear_output(basis[i],direction)
            return
        p = (a & -a).bit_length()-1
        first = list(set_bits(a ^ (1 << p)))
        for i in first:
            gate('CX',basis[i],basis[p])
        if b >> p & 1:
            b ^= a ^ (1 << p)
        choices = b & ~(1 << p)
        assert choices
        q = (choices & -choices).bit_length()-1
        second = list(set_bits(b ^ (1 << q)))
        for i in second:
            gate('CX',basis[i],basis[q])
        pivot,*others = target_bits(direction)
        for out in others:
            gate('CX',pivot,out)
        gate('CCX',basis[p],basis[q],pivot)
        for out in others:
            gate('CX',pivot,out)
        for i in reversed(second):
            gate('CX',basis[i],basis[q])
        for i in reversed(first):
            gate('CX',basis[i],basis[p])
    high_only = [sum(int(ac[h,0,j]) << j for j in range(O)) for h in range(A)]
    low_only = [sum(int(ac[0,l,j]) << j for j in range(O)) for l in range(B)]
    low_only[0] = 0
    def correction(destination,left,right,direction):
        for a in set_bits(left):
            for b in set_bits(right):
                index = (a+1)|(b+1)
                if index < len(destination):
                    destination[index] ^= direction
    for term in terms:
        pairs = term['factors']
        direction = term['target_direction']
        for i in range(0,len(pairs),2):
            h1,l1 = pairs[i]
            if i+1 == len(pairs):
                product(h1,l1 << H,direction)
            else:
                h2,l2 = pairs[i+1]
                # (h1+l2)(h2+l1) + h1*h2 + l1*l2
                # equals h1*l1 + h2*l2 over GF(2).
                product(h1 ^ (l2 << H),h2 ^ (l1 << H),direction)
                correction(high_only,h1,h2,direction)
                correction(low_only,l1,l2,direction)
    for q in target_bits(high_only[0]):
        gate('X',q)
    for mask in range(1,A):
        linear_output(high[mask],high_only[mask])
    for mask in range(1,B):
        linear_output(low[mask],low_only[mask])
    for triple in reversed(cache):
        gate('CCX',*triple)
    circuit = builder.finish_circuit()
    native = int(circuit.operation_statistics().num_toffoli_like)
    assert native == sum(t[0]=='CCX' for t in trace)
    return circuit,trace,{'num_qubits':int(circuit.num_qubits),
        'toffoli_count':native,'clean_cached_functions':len(cache),
        'coherent_cache_compute_and_cleanup_toffolis':2*len(cache),
        'gates':len(trace)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--angle-words',type=Path,required=True)
    p.add_argument('--tree',type=Path,required=True)
    p.add_argument('--source-factorization',type=Path,required=True)
    p.add_argument('--input-brot',type=int,default=12)
    p.add_argument('--brot',type=int,default=8)
    p.add_argument('--rows',type=int,default=700)
    p.add_argument('--batch-size',type=int,default=5)
    p.add_argument('--lowbits',type=int,default=4)
    p.add_argument('--k',type=int,default=0)
    p.add_argument('--max-kernels',type=int,default=32767)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--verifier-dir',type=Path,default=Path('/tmp/chem_verifiers/dfthcsos'))
    a = p.parse_args()
    start = time.time()
    if a.out.exists():
        raise FileExistsError('refusing to overwrite previous results')
    original = np.load(a.angle_words,allow_pickle=False)[:a.rows].copy()
    edges = np.load(a.tree,allow_pickle=False)
    if a.input_brot < a.brot:
        raise ValueError('input precision must be at least output precision')
    shift = a.input_brot-a.brot
    rounded = ((original+(1 << (shift-1))) >> shift) % (1 << a.brot) if shift else original
    words,signs = canonical_gauge(rounded,edges,a.brot)
    gauge_error = float(np.max(np.abs(decode(words,edges,a.brot)-
                                     signs[:,None]*decode(rounded,edges,a.brot))))
    assert gauge_error < 1e-12
    delta = delta_batch(words,a.batch_size,a.k)
    ac,terms,planning = plan(delta,a.brot,a.lowbits,a.max_kernels)
    circuit,trace,report = build(delta,a.brot,a.lowbits,ac,terms)
    report['exact_symbolic_oracle'] = symbolic_check(trace,delta,a.brot,circuit.num_qubits)
    report['native_oracle'] = native_check(circuit,delta,a.brot,a.verifier_dir)
    a.out.mkdir(parents=True)
    np.save(a.out/'fixture_angle_words.npy',words)
    np.save(a.out/'batch_delta.npy',delta)
    np.save(a.out/'anf_coefficients.npy',ac)
    (a.out/'certificate.json').write_text(json.dumps({'plan':planning,'terms':terms},indent=2)+'\n')
    cp = a.out/f'angle_k_{a.k}.kmb'
    circuit.to_file(cp,format='kmb')
    (a.out/f'angle_k_{a.k}.kmx').write_text(str(circuit))
    report.update({'scope':'ORGANIZER_SUBSET_CIRCUIT_FIXTURE_ONLY',
        'chemistry_acceptance_claimed':False,'full_family_compiled':False,
        'D':len(words),'brot':a.brot,'batch_size':a.batch_size,'k':a.k,
        'lowbits':a.lowbits,'planning':planning,'gauge_max_coordinate_error':gauge_error,
        'source_factorization_sha256':sha(a.source_factorization),
        'source_words_sha256':sha(a.angle_words),'source_tree_sha256':sha(a.tree),
        'fixture_words_sha256':sha(a.out/'fixture_angle_words.npy'),
        'compiler_sha256':sha(__file__),'proof_helper_sha256':sha(Path(__file__).with_name('borrowed_prefix.py')),
        'native_sha256':sha(cp),'seconds':time.time()-start})
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='planning'},indent=2))
    print(json.dumps({k:v for k,v in planning.items() if k not in ['output_basis_certificates','original_matrix_ranks']},indent=2))


if __name__ == '__main__':
    main()
