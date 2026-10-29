"""Exact four-word oracle with a two-stage, seven-qubit low-function cache.

All target bits may be arbitrary. Never borrow targets or modify input data.
Only measurement-conditioned Clifford phase corrections are emitted.
"""
from __future__ import annotations
import numpy as np
import kickmix as km
from rank_parity import set_bits, require_redundant_sign_bit


def build_streamed(delta, ac, stage_terms, *, brot=8):
    delta=require_redundant_sign_bit(delta,brot)
    D,J=delta.shape
    if J != 4 or brot != 8:
        raise ValueError('This layout requires four eight-bit words')
    A,B,O=ac.shape
    if B!=16 or O!=28:
        raise ValueError('Expected four low selection bits and 28 independent outputs')
    n=(D-1).bit_length(); H=A-1
    builder=km.CircuitBuilder()
    qs=list(builder.create_quantum_register(n,'selection'))
    for j in range(J): qs+=list(builder.create_quantum_register(brot,f'word_{j}'))
    emitted=0
    def gate(name,*ids):
        nonlocal emitted
        getattr(builder,name.lower())(*[qs[i] for i in ids])
        emitted += (name=='CCX')
    high={0:None}; high_cache=[]
    for mask in range(1,A):
        bit=mask.bit_length()-1
        if mask.bit_count()==1: high[mask]=4+bit
        else:
            parent=mask^(1<<bit); q=len(qs);qs.extend(builder.alloc_qubits(1))
            high[mask]=q;triple=(high[parent],4+bit,q)
            gate('CCX',*triple);high_cache.append(triple)
    slots=[]
    for _ in range(7):
        slots.append(len(qs));qs.extend(builder.alloc_qubits(1))
    if len(qs)>128:raise ValueError(f'Layout requires {len(qs)} qubits')
    low={1<<bit:bit for bit in range(4)};low_triples={}
    def create(mask,slot,parent=None):
        if parent is None:
            bit=mask.bit_length()-1;parent=mask^(1<<bit)
        else:bit=(mask^parent).bit_length()-1
        triple=(low[parent],bit,slot);gate('CCX',*triple)
        low[mask]=slot;low_triples[mask]=triple
    def erase(mask):
        a,b,q=low_triples.pop(mask)
        builder.del_and(qs[a],qs[b],qs[q])
        del low[mask]
        return q
    for mask,q in zip([3,5,6,7,9,10,11],slots):
        create(mask,q,parent=6 if mask==7 else None)
    groups=[[n+j*brot+b]+([n+j*brot+brot-1] if b==brot-2 else [])
            for j in range(J) for b in range(brot-1)]
    def targets(direction):return [q for bit in set_bits(direction) for q in groups[bit]]
    def linear(control,direction):
        for q in targets(direction):gate('CX',control,q)
    def product(left,right,direction,basis):
        if not left or not right or not direction:return
        if left==right:
            for i in set_bits(left):linear(basis[i],direction)
            return
        p=(left&-left).bit_length()-1
        first=list(set_bits(left^(1<<p)))
        for i in first:gate('CX',basis[i],basis[p])
        if (right>>p)&1:right^=left^(1<<p)
        choices=right&~(1<<p)
        if not choices:raise AssertionError('Dependent controls')
        q=(choices&-choices).bit_length()-1
        second=list(set_bits(right^(1<<q)))
        for i in second:gate('CX',basis[i],basis[q])
        pivot,*others=targets(direction)
        for out in others:gate('CX',pivot,out)
        gate('CCX',basis[p],basis[q],pivot)
        for out in others:gate('CX',pivot,out)
        for i in reversed(second):gate('CX',basis[i],basis[q])
        for i in reversed(first):gate('CX',basis[i],basis[p])
    high_only=[sum(int(ac[h,0,j])<<j for j in range(O)) for h in range(A)]
    low_only=[sum(int(ac[0,l,j])<<j for j in range(O)) for l in range(B)]
    low_only[0]=0
    def correction(dest,left,right,direction,indices):
        for a in set_bits(left):
            for b in set_bits(right):
                index=indices[a]|indices[b]
                if index<len(dest):dest[index]^=direction
    # Compute same-side corrections before any cache is erased. This is
    # classical circuit construction, not a measurement-dependent operation.
    stages=[list(range(1,12)),list(range(12,16))]
    for indices,terms in zip(stages,stage_terms):
        for term in terms:
            fs=term['factors'];d=term['target_direction']
            for i in range(0,len(fs)-1,2):
                h1,l1=fs[i];h2,l2=fs[i+1]
                correction(high_only,h1,h2,d,list(range(1,A)))
                correction(low_only,l1,l2,d,indices)
    def emit_stage(indices,terms):
        basis=[high[h] for h in range(1,A)]+[low[l] for l in indices]
        for term in terms:
            fs=term['factors'];d=term['target_direction']
            for i in range(0,len(fs),2):
                h1,l1=fs[i]
                if i+1==len(fs):product(h1,l1<<H,d,basis)
                else:
                    h2,l2=fs[i+1]
                    product(h1^(l2<<H),h2^(l1<<H),d,basis)
        for l in indices:linear(low[l],low_only[l])
    emit_stage(stages[0],stage_terms[0])
    # 7 uses parent 6 rather than 3, so the latter can be erased here.
    freed=[erase(mask) for mask in [11,10,9,3]]
    for mask,q in zip([12,13,14,15],freed):create(mask,q)
    emit_stage(stages[1],stage_terms[1])
    for q in targets(high_only[0]):gate('X',q)
    for h in range(1,A):linear(high[h],high_only[h])
    for mask in [15,14,13,12,7,6,5]:erase(mask)
    for a,b,q in reversed(high_cache):builder.del_and(qs[a],qs[b],qs[q])
    c=builder.finish_circuit()
    native=int(c.operation_statistics().num_toffoli_like)
    assert native==emitted
    return c,{'T':native,'Q':int(c.num_qubits),'high_cache_qubits':len(high_cache),
              'peak_low_cache_qubits':7,'low_ANDs_computed':11,
              'stage_products':[sum((len(t['factors'])+1)//2 for t in terms) for terms in stage_terms]}
