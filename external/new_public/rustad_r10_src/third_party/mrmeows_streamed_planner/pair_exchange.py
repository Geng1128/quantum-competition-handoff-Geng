"""Exact local-exchange augmentation for a paired binary basis.

Greedy paired selection can stall two dimensions below the full space.
Remove one selected pair, then insert two pairs independent in the quotient.
Every augmentation is checked using exact GF(2) coordinates.
"""
from __future__ import annotations
import random
import rank_parity as core
import kernel_planes as binary


def paired_exchange(templates,target,seed=0,trials=16):
    rng=random.Random(seed);best=[];trace=[]
    for trial in range(trials):
        order=list(range(len(templates)));rng.shuffle(order)
        span={};selected=[]
        for i in order:
            a,b=templates[i][:2];ar=binary.reduce_row(a,span)
            if not ar:continue
            pivot=ar.bit_length()-1;span[pivot]=ar
            br=binary.reduce_row(b,span)
            if not br:del span[pivot];continue
            span[br.bit_length()-1]=br;selected.append(i)
            if len(span)==target:return selected,{'method':'exchange','trace':trace,'greedy_trial':trial}
        if len(selected)>len(best):best=selected
    selected=best
    for iteration in range(target//2):
        rank=2*len(selected)
        if rank==target:return selected,{'method':'exchange','trace':trace}
        basis=core.BinaryBasis()
        for i in selected:
            assert basis.add(templates[i][0]) and basis.add(templates[i][1])
        for a,b,*_ in templates:
            basis.add(a);basis.add(b)
            if len(basis.vectors)==target:break
        if len(basis.vectors)!=target:raise ValueError('Incomplete template span')
        coords=[(basis.solve(t[0]),basis.solve(t[1])) for t in templates]
        remove_order=list(range(len(selected)));rng.shuffle(remove_order)
        found=None
        for removed in remove_order:
            shift=2*removed;planes={}
            for j,(a,b) in enumerate(coords):
                ar=((a>>shift)&3)|((a>>rank)<<2)
                br=((b>>shift)&3)|((b>>rank)<<2)
                if not ar or not br or ar==br:continue
                plane=tuple(sorted((ar,br,ar^br)))
                if plane in planes:continue
                for other,k in planes.items():
                    if not any(v in other for v in plane):
                        found=(removed,j,k);break
                if found:break
                planes[plane]=j
            if found:break
        if not found:
            trace.append({'rank':rank,'augmentation':False});return None,{'method':'exchange','trace':trace}
        removed,j,k=found
        selected=[i for pos,i in enumerate(selected) if pos!=removed]+[j,k]
        span={}
        for i in selected:
            assert binary.append_row(templates[i][0],span) and binary.append_row(templates[i][1],span)
        assert len(span)==rank+2
        trace.append({'rank_before':rank,'rank_after':len(span),'removed_position':removed,'inserted':[j,k]})
    return None,{'method':'exchange','trace':trace}
