"""Three-low-direction kernel cycles for exact J4 bilinear synthesis."""
from pathlib import Path
import sys,json,time,random,itertools
import numpy as np
import rank_parity as core
import kernel_planes as binary

def quotient_selection(templates,target,L,seed,trials=64):
    W0={};coordinate=[];spaces=[{} for _ in range(L)]
    for i,(a,b,v,ell1,ell2) in enumerate(templates):
        if ell1.bit_count()==ell2.bit_count()==1:
            left=ell1.bit_length()-1;right=ell2.bit_length()-1
            coordinate.append((i,left,right,v))
            binary.append_row(a,W0);binary.append_row(b,W0)
            binary.append_row(v,spaces[left]);binary.append_row(v,spaces[right])
    pivots=sorted(W0.items(),reverse=True)
    def quotient(v):
        for p,row in pivots:
            if v>>p&1:v^=row
        return v
    mixed=[]
    for i,(a,b,v,ell1,ell2) in enumerate(templates):
        ar,br=quotient(a),quotient(b)
        if ar and not br and ell2.bit_count()==1:mixed.append((i,ar,ell2.bit_length()-1,v))
        elif br and not ar and ell1.bit_count()==1:mixed.append((i,br,ell1.bit_length()-1,v))
    qrank=target-len(W0);dims=[len(x) for x in spaces];rng=random.Random(seed);attempts=[]
    for trial in range(trials):
        order=mixed.copy();rng.shuffle(order);qspan={};local=[{} for _ in range(L)];chosen=[]
        for i,q,vertex,v in order:
            qr=binary.reduce_row(q,qspan);vr=binary.reduce_row(v,local[vertex])
            if qr and vr:
                qspan[qr.bit_length()-1]=qr;local[vertex][vr.bit_length()-1]=vr;chosen.append(i)
        if len(qspan)!=qrank:
            attempts.append({'trial':trial,'quotient_rank':len(qspan),'required':qrank});continue
        pool=coordinate.copy();rng.shuffle(pool)
        while 2*len(chosen)<target:
            deficit=[dims[j]-len(local[j]) for j in range(L)];best=None
            for i,left,right,v in pool:
                if not deficit[left] or not deficit[right]:continue
                lr=binary.reduce_row(v,local[left]);rr=binary.reduce_row(v,local[right])
                if not lr or not rr:continue
                score=(deficit[left]+deficit[right],min(deficit[left],deficit[right]))
                if best is None or score>best[0]:best=(score,i,left,right,lr,rr)
            if best is None:break
            _,i,left,right,lr,rr=best;local[left][lr.bit_length()-1]=lr;local[right][rr.bit_length()-1]=rr;chosen.append(i)
        span={}
        for i in chosen:
            binary.append_row(templates[i][0],span);binary.append_row(templates[i][1],span)
        attempts.append({'trial':trial,'quotient_rank':len(qspan),'rank':len(span),'pairs':len(chosen)})
        if 2*len(chosen)==len(span)==target:return chosen,{'W0':len(W0),'qrank':qrank,'attempts':attempts}
    return None,{'W0':len(W0),'qrank':qrank,'attempts':attempts}

def plan(delta,extra=15,seed=0):
    ac=core.coefficients(delta,8,4); tensor=ac[1:,1:,:];H,L,m=tensor.shape
    assert (L,m)==(15,28)
    rows=[binary.packed(row) for row in tensor.reshape(H,L*m)]
    rng=random.Random(seed);relations=[];nullities=[]
    chosen_extra=set(rng.sample(range(L),extra))
    for first in range(L):
        vs=[(first+i)%L for i in range(3)]
        block=np.concatenate([tensor[:,v,:] for v in vs],axis=1)
        selected,nullity=binary.select_relations([binary.packed(row) for row in block],3*m,3+(first in chosen_extra),rng)
        nullities.append(nullity)
        for rel in selected:
            relations.append(sum(((rel>>(j*m))&((1<<m)-1))<<(v*m) for j,v in enumerate(vs)))
    rb={}
    for r in relations:binary.append_row(r,rb)
    assert all(not ((row&r).bit_count()&1) for row in rows for r in relations)
    chunks=[[(r>>(j*m))&((1<<m)-1) for j in range(L)] for r in relations]
    planes=[(1<<i,1<<j) for i in range(L) for j in range(i+1,L)]
    for step in [1,2]:
        planes.extend(((1<<i)|(1<<((i+step)%L)),1<<k) for i in range(L) for k in range(L) if k not in [i,(i+step)%L])
    templates=[];dims=[]
    for a,b in planes:
        eq=[]
        for cs in chunks:
            for ell in [a,b]:
                value=0
                for j in core.set_bits(ell):value^=cs[j]
                if value:eq.append(value)
        V=core.nullspace(eq,m);dims.append(len(V))
        directions=set(V)
        if len(V)<=6:
            for code in range(1,1<<len(V)):
                direction=0
                for j in core.set_bits(code):direction^=V[j]
                directions.add(direction)
        for v in sorted(directions):
            if v:
                u=binary.tensor_column(v,a,m);w=binary.tensor_column(v,b,m)
                templates.append((u,w,v,a,b))
    span={}
    for a,b,*_ in templates:binary.append_row(a,span);binary.append_row(b,span)
    target=L*m-len(rb)
    report={'extra':extra,'seed':seed,'nullities':nullities,'relations':len(rb),'target_rank':target,'span_rank':len(span),'templates':len(templates),'plane_dim_hist':{i:dims.count(i) for i in sorted(set(dims))},'contains_target':all(not binary.reduce_row(row,span) for row in rows),'T_lower_if_paired':84+(target+1)//2}
    if len(span)!=target or not report['contains_target'] or target%2:return None,report
    selected,attempts=quotient_selection(templates,target,L,seed+7001)
    report['paired_attempts']=attempts
    if selected is None:return None,report
    basis=core.BinaryBasis()
    for i in selected:
        a,b,*_=templates[i];assert basis.add(a) and basis.add(b)
    solutions=[basis.solve(row) for row in rows];assert all(s is not None for s in solutions)
    terms=[];reconstructed=[0]*H
    for j,i in enumerate(selected):
        a,b,v,ell1,ell2=templates[i]
        h1=sum(((s>>(2*j))&1)<<h for h,s in enumerate(solutions));h2=sum(((s>>(2*j+1))&1)<<h for h,s in enumerate(solutions))
        terms.append({'target_direction':v,'factors':[(h1,ell1),(h2,ell2)]})
        for h in core.set_bits(h1):reconstructed[h]^=a
        for h in core.set_bits(h2):reconstructed[h]^=b
    assert reconstructed==rows
    report.update({'paired_products':len(selected),'T_HMR':84+len(selected),'Q':127,'exact_tensor':True})
    return (ac,terms),report
