"""Exact joint-output synthesis with reversed six/four-bit address spaces."""
from __future__ import annotations
from pathlib import Path
import random, sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'new_lookup/scripts'))
import kickmix as km
import joint_low_rank as base
from recursive_lookup import mobius, batch_deltas
from factor_affine_controls import xor_affine_product


from joint_low_rank import Feature, packed_rows, rowbasis, nullspace, reduce_row, parity


def feature_basis(matrix,m, codim_cap=13, annihilator_seed=None, nlow=13):
    """Find rank-one rows grouped in even counts by output direction."""
    rows=packed_rows(matrix)
    rank=len(rowbasis(rows));relations=nullspace(rows,nlow*m)
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
    span_rank=nlow*m-len(relations)
    choices={}
    for out in range(1,1<<m):
        constraints=[sum(parity((rel>>(l*m))&out)<<l for l in range(nlow)) for rel in relations]
        lows=nullspace(constraints,nlow)
        if lows:
            shift=out%len(lows)
            lows=lows[shift:]+lows[:shift]
        choices[out]=[Feature(lo,out,sum(out << (l*m) for l in range(nlow) if (lo>>l)&1)) for lo in lows]
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


def prepare(delta, lowbits=6):
    count,words=delta.shape
    nA=1<<lowbits; nB=(count+nA-1)//nA
    padded=np.zeros((nB*nA,words),dtype=np.int64);padded[:count]=delta
    bits=np.stack([(padded[:,w]>>bit)&1 for w in range(words) for bit in range(9)],axis=1).astype(np.uint8)
    truth=bits.reshape(nB,nA,-1).transpose(1,0,2).copy()
    bcoeff=mobius(truth.transpose(1,0,2)).transpose(1,0,2).copy()
    return truth,bcoeff,mobius(bcoeff)


def analyze(delta, lowbits=6, group_size=6, codim_cap=12, trials=16, plan_records=None):
    truth,bcoeff,allcoeff=prepare(delta,lowbits)
    nA,nB,slots=truth.shape
    plans=[]
    for start in range(0,slots,group_size):
        end=min(start+group_size,slots)
        target=allcoeff[1:,1:,start:end].reshape(nA-1,-1)
        if plan_records is not None:
            rec=plan_records[len(plans)]
            features,coeff,stats=feature_basis(target,end-start,rec['relations'],
                                               rec['annihilator_seed'],nlow=nB-1)
            assert stats['products']==rec['products']
            plans.append((start,end,features,coeff,stats))
            continue
        best=None
        for codim in range(codim_cap,-1,-1):
            for seed in [None]+list(range(trials)):
                try:result=feature_basis(target,end-start,codim,seed,nlow=nB-1)
                except ValueError:continue
                if best is None or result[2]['products']<best[2]['products']:best=result
                if result[2]['products']==(result[2]['span_rank']+1)//2:break
            if best is not None:break
        if best is None:raise AssertionError('No basis found')
        features,coeff,stats=best
        plans.append((start,end,features,coeff,stats))
    return truth,bcoeff,allcoeff,plans


def build_circuit(delta, lowbits=6, group_size=6, codim_cap=12, trials=16, plan_records=None):
    delta=np.asarray(delta,dtype=np.int64)
    count,words=delta.shape;n=(count-1).bit_length()
    assert not np.any(((delta>>9)^(delta>>8))&1)
    truth,bcoeff,allcoeff,plans=analyze(delta,lowbits,group_size,codim_cap,trials,plan_records)
    nA,nB,_=truth.shape
    builder=km.CircuitBuilder();selection=builder.create_quantum_register(n,'selection')
    registers=[builder.create_quantum_register(10,f'word_{j}') for j in range(words)]
    outputs=[[registers[w][bit]]+([registers[w][9]] if bit==8 else [])
             for w in range(words) for bit in range(9)]
    A={0:True};B={0:True};products=[]
    for basis,size,offset in [(A,nA,0),(B,nB,lowbits)]:
        for mask in range(1,size):
            bit=mask.bit_length()-1
            if mask.bit_count()==1:basis[mask]=selection[offset+bit]
            else:
                parent=mask^(1<<bit);q=builder.alloc_qubits(1)[0]
                builder.init_and(basis[parent],selection[offset+bit],q)
                basis[mask]=q;products.append((basis[parent],selection[offset+bit],q))
    basis=[A[k] for k in range(1,nA)]+[B[k] for k in range(1,nB)]
    Aonly=bcoeff[:,0].copy()
    Bonly=allcoeff[0].copy();Bonly[0]=0
    Btruth=mobius(Bonly)
    Bmonomials=[sum(1<<(k-1) for k in range(1,nB) if (k&x)==k) for x in range(nB)]
    for start,end,features,coeff,stats in plans:
        forms=[]
        for j,feature in enumerate(features):
            mask=sum(((v>>j)&1)<<row for row,v in enumerate(coeff))
            av=mobius(np.array([0]+[((v>>j)&1) for v in coeff],dtype=np.uint8))
            bv=np.array([base.parity(feature.low&x) for x in Bmonomials],dtype=np.uint8)
            forms.append((mask,av,bv))
        for out in sorted(set(f.out for f in features)):
            ids=[j for j,f in enumerate(features) if f.out==out]
            targets=[q for j in range(end-start) if (out>>j)&1 for q in outputs[start+j]]
            for off in range(0,len(ids),2):
                j=ids[off];fa,ta,la=forms[j]
                if off+1==len(ids):
                    xor_affine_product(builder,basis,fa,features[j].low<<(nA-1),targets)
                else:
                    k=ids[off+1];fb,tb,lb=forms[k]
                    xor_affine_product(builder,basis,fa^(features[k].low<<(nA-1)),
                                       fb^(features[j].low<<(nA-1)),targets)
                    for slot in range(end-start):
                        if (out>>slot)&1:
                            Aonly[:,start+slot]^=ta&tb
                            Btruth[:,start+slot]^=la&lb
    for table,bases in [(mobius(Aonly),A),(mobius(Btruth),B)]:
        for mask,row in enumerate(table):
            targets=[q for j in np.flatnonzero(row) for q in outputs[int(j)]]
            if targets:builder.cx(bases[mask],km.array(targets))
    for a,b,q in reversed(products):builder.del_and(a,b,q);builder.free(q)
    circuit=builder.finish_circuit()
    return circuit,[p[-1] for p in plans]


if __name__=='__main__':
    import json,time
    from low_three_lookup import check
    root=Path(__file__).resolve().parents[1]
    words=np.load(root/'push_synthesis/c29b_joint_j5_optimized/angle_words.npy')
    delta=batch_deltas(words,5)[0]
    t=time.monotonic();c,stats=build_circuit(delta)
    native=check(c,delta,brot=10,repeats=4)
    print(json.dumps({'T':native,'Q':c.num_qubits,'stats':stats,'seconds':time.monotonic()-t},indent=2))
