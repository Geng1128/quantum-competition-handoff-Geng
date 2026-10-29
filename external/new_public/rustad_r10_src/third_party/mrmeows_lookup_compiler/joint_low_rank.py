"""Joint six-output synthesis by rank-one matrix-space bases over GF(2).

Every rank-one low/output feature has a high-only coefficient. Pair two
features with the same output direction into one affine-product Toffoli;
the high-only and low-only cross products are absorbed into free lookups.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sys
import random
import numpy as np

SCRIPTS=Path(__file__).resolve().parents[1]/'new_lookup/scripts'
sys.path.insert(0,str(SCRIPTS))
import kickmix as km
from recursive_lookup import mobius, batch_deltas
from factor_affine_controls import xor_affine_product

def parity(v): return int(v).bit_count()&1

def packed_rows(a):
    return [int.from_bytes(np.packbits(r.astype(np.uint8),bitorder='little').tobytes(),'little') for r in a]

def rowbasis(rows):
    piv={}
    for row in rows:
        value=int(row)
        while value:
            p=value.bit_length()-1
            if p in piv: value^=piv[p]
            else: piv[p]=value;break
    return piv

def nullspace(rows,n):
    # RREF using the highest set position as each pivot.
    piv=rowbasis(rows)
    for p in sorted(piv):
        for q in list(piv):
            if q>p and ((piv[q]>>p)&1):piv[q]^=piv[p]
    out=[]
    for f in range(n):
        if f in piv:continue
        value=1<<f
        for p,row in piv.items():
            if (row>>f)&1:value^=1<<p
        assert all(parity(value&r)==0 for r in rows)
        out.append(value)
    return out

def rank1(low,out,m):
    return sum(out << (l*m) for l in range(15) if (low>>l)&1)

def reduce_row(v,piv):
    while v:
        p=v.bit_length()-1
        if p not in piv:return v
        v^=piv[p]
    return 0

@dataclass
class Feature:
    low: int
    out: int
    vector: int

def feature_basis(matrix,m, codim_cap=13, annihilator_seed=None):
    """Find rank-one rows grouped in even counts by output direction."""
    rows=packed_rows(matrix)
    rank=len(rowbasis(rows));relations=nullspace(rows,15*m)
    if len(relations)>codim_cap:
        if annihilator_seed is None:
            relations=relations[:codim_cap]
        else:
            rng=random.Random(annihilator_seed);chosen=[];independent={}
            while len(chosen)<codim_cap:
                combo=rng.getrandbits(len(relations));value=0
                for j,rel in enumerate(relations):
                    if (combo>>j)&1:value^=rel
                reduced=reduce_row(value,independent)
                if reduced:
                    chosen.append(value);independent[reduced.bit_length()-1]=reduced
            relations=chosen
    span_rank=15*m-len(relations)
    choices={}
    for out in range(1,1<<m):
        constraints=[sum(parity((rel>>(l*m))&out)<<l for l in range(15)) for rel in relations]
        lows=nullspace(constraints,15)
        if lows:
            shift=out%len(lows)
            lows=lows[shift:]+lows[:shift]
        choices[out]=[Feature(lo,out,rank1(lo,out,m)) for lo in lows]
    # Add an even number of independent features whenever possible. The
    # final odd rank requires exactly one unpaired feature.
    piv={}; selected=[]
    while len(piv)<span_rank:
        best=None
        for out,features in choices.items():
            trial=piv.copy();novel=[]
            for feature in features:
                reduced=reduce_row(feature.vector,trial)
                if reduced:
                    trial[reduced.bit_length()-1]=reduced
                    novel.append(feature)
            n=len(novel)
            if not n:continue
            remaining=span_rank-len(piv)
            take=n if n==remaining else n-(n&1)
            if not take:take=1
            score=(take%2==0,take,n,-out)
            if best is None or score>best[0]:best=(score,novel[:take])
        if best is None:
            raise ValueError(f'rank-one span exhausted at {len(piv)} of {span_rank}, m={m}')
        for f in best[1]:
            reduced=reduce_row(f.vector,piv)
            assert reduced
            piv[reduced.bit_length()-1]=reduced
            selected.append(f)
    # Re-eliminate while tracking selected feature combinations.
    tracked={}
    for j,f in enumerate(selected):
        value,combo=f.vector,1<<j
        while value:
            p=value.bit_length()-1
            if p not in tracked:
                tracked[p]=(value,combo);break
            v,c=tracked[p];value^=v;combo^=c
    coefficients=[]
    for row in rows:
        value,combo=row,0
        while value:
            p=value.bit_length()-1
            if p not in tracked:raise AssertionError('target row not in rank-one span')
            v,c=tracked[p];value^=v;combo^=c
        coefficients.append(combo)
    return selected,coefficients,{'rank':rank,'span_rank':span_rank,'relations':len(relations),'features':len(selected),
        'annihilator_seed':annihilator_seed,
        'products':sum((sum(f.out==out for f in selected)+1)//2 for out in set(f.out for f in selected))}

def groups_for_slots(slots,group_size):
    # Groups six have 90 coefficient columns against at most 79 independent
    # high nonconstant functions. Smaller residue uses the same exact basis.
    return [(i,min(i+group_size,slots)) for i in range(0,slots,group_size)]

def analyze(delta, group_size=6, codim_cap=13):
    count,words=delta.shape;h=(count+15)//16
    padded=np.zeros((16*h,words),dtype=np.int64);padded[:count]=delta
    bits=np.stack([(padded[:,w]>>bit)&1 for w in range(words) for bit in range(9)],axis=1).astype(np.uint8)
    coeff=bits.reshape(h,16,-1).copy()
    for bit in range(4):
        for mask in range(16):
            if mask&(1<<bit):coeff[:,mask]^=coeff[:,mask^(1<<bit)]
    allcoeff=mobius(coeff)
    plans=[]
    for start,end in groups_for_slots(bits.shape[1],group_size):
        target=allcoeff[1:,1:,start:end].reshape(h-1,-1)
        best=None
        for seed in [None]+list(range(16)):
            try:
                result=feature_basis(target,end-start,codim_cap,seed)
            except ValueError:
                continue
            if best is None or result[2]['products']<best[2]['products']:best=result
            if result[2]['products']==(result[2]['span_rank']+1)//2:break
        if best is None:
            best=feature_basis(target,end-start,codim_cap-1,None)
        features,highcoeff,stats=best
        plans.append((start,end,features,highcoeff,stats))
    return bits,coeff,allcoeff,plans

def build_circuit(delta, group_size=6, codim_cap=13):
    delta=np.asarray(delta,dtype=np.int64)
    count,word_count=delta.shape;n=(count-1).bit_length();h=(count+15)//16
    assert not np.any(((delta>>9)^(delta>>8))&1)
    bits,coeff,allcoeff,plans=analyze(delta,group_size,codim_cap)
    builder=km.CircuitBuilder();sel=builder.create_quantum_register(n,'selection')
    words=[builder.create_quantum_register(10,f'word_{j}') for j in range(word_count)]
    outputs=[[words[w][bit]]+([words[w][9]] if bit==8 else []) for w in range(word_count) for bit in range(9)]
    high={0:True};low={0:True};products=[]
    for basis,size,offset in [(high,h,4),(low,16,0)]:
        for mask in range(1,size):
            bit=mask.bit_length()-1
            if mask.bit_count()==1:basis[mask]=sel[offset+bit]
            else:
                parent=mask^(1<<bit);q=builder.alloc_qubits(1)[0]
                builder.init_and(basis[parent],sel[offset+bit],q)
                basis[mask]=q;products.append((basis[parent],sel[offset+bit],q))
    basis=[high[k] for k in range(1,h)]+[low[k] for k in range(1,16)]
    highonly=coeff[:,0].copy()
    lowonly=allcoeff[0].copy();lowonly[0]=0
    lowtruth=lowonly.copy()
    for bit in range(4):
        for mask in range(16):
            if mask&(1<<bit):lowtruth[mask]^=lowtruth[mask^(1<<bit)]
    for start,end,features,highcoeff,stats in plans:
        hforms=[]
        for j,f in enumerate(features):
            mask=sum(((v>>j)&1)<<(row) for row,v in enumerate(highcoeff))
            # coefficient bit row corresponds to high monomial row+1
            values=np.array([0]+[((v>>j)&1) for v in highcoeff],dtype=np.uint8)
            truth=mobius(values)
            ltruth=np.array([parity(f.low&sum((1<<(k-1)) for k in range(1,16) if (k&x)==k)) for x in range(16)],dtype=np.uint8)
            hforms.append((mask,truth,ltruth))
        for out in sorted(set(f.out for f in features)):
            ids=[j for j,f in enumerate(features) if f.out==out]
            targets=[q for j in range(end-start) if (out>>j)&1 for q in outputs[start+j]]
            for off in range(0,len(ids),2):
                j=ids[off];fa,ta,la=hforms[j]
                if off+1==len(ids):
                    xor_affine_product(builder,basis,fa,features[j].low<<(h-1),targets)
                else:
                    k=ids[off+1];fb,tb,lb=hforms[k]
                    xor_affine_product(builder,basis,fa^(features[k].low<<(h-1)),
                                       fb^(features[j].low<<(h-1)),targets)
                    for slot in range(end-start):
                        if(out>>slot)&1:
                            highonly[:,start+slot]^=ta&tb
                            lowtruth[:,start+slot]^=la&lb
    hc=mobius(highonly);lc=mobius(lowtruth)
    for table,bases in [(hc,high),(lc,low)]:
        for mask,row in enumerate(table):
            targets=[q for j in np.flatnonzero(row) for q in outputs[int(j)]]
            if targets:builder.cx(bases[mask],km.array(targets))
    for a,b,q in reversed(products):builder.del_and(a,b,q);builder.free(q)
    c=builder.finish_circuit()
    return c,[p[-1] for p in plans]

if __name__=='__main__':
    import json,time
    w=np.load(Path(__file__).resolve().parents[1]/'better_lookup_verified/dfthcsos/submission.npz')['angle_words']
    d=batch_deltas(w,3)[0]
    t=time.monotonic();c,stats=build_circuit(d)
    from low_three_lookup import check
    native=check(c,d,brot=10,repeats=4)
    print(json.dumps({'stats':stats,'T':native,'Q':c.num_qubits,'seconds':time.monotonic()-t},indent=2))
