"""Signed Givens gauge, four-address QROM, and selective AND pebbling.

New implementation, Apache-2.0. Uses the attributed banked_lab ANF eraser.
Ten-bit half-open signed angles have bit9=bit8. Three nine-bit differences
per word encode four low-address cases. Three words require 81 bank wires;
selection12 + public30 + bank81 + five clean scratch = 128 qubits.
"""
from functools import lru_cache
import numpy as np
import kickmix as km
import banked_lab as B

def canonicalize(context,data,brot=10):
    import chemistry_refine as C
    out=C.copy_data(data)
    u=context.math.reconstruct_u_vectors(data['angle_words']/(1<<(brot-1)),data['tree_edges'])
    a=u.copy();angles=np.empty_like(data['angle_words'],dtype=np.float64)
    # Numerical zero branches arise from sin(pi), cos(pi/2). Their angles
    # carry no physical information; explicitly remove only <1e-12 mass.
    for j,(p,q) in reversed(list(enumerate(data['tree_edges']))):
        norm=np.hypot(a[:,p],a[:,q])
        sign=np.where(np.abs(a[:,p])<1e-12,np.where(a[:,q]>0,-1,1),np.where(a[:,p]<0,-1,1))
        angles[:,j]=np.where(norm<1e-12,0,np.arctan2(a[:,q]*sign,a[:,p]*sign))
        a[:,p]=np.where(norm<1e-12,0,norm*sign);a[:,q]=0
    words=np.rint(angles*((1<<(brot-1))/np.pi)).astype(np.int64)%(1<<brot)
    assert not np.any(((words>>(brot-1))^(words>>(brot-2)))&1), 'half-open gauge failed'
    decoded=context.math.reconstruct_u_vectors(words/(1<<(brot-1)),data['tree_edges'])
    sign=np.where((u*decoded).sum(axis=1)<0,-1,1)
    drift=float(np.max(np.abs(decoded*sign[:,None]-u)))
    if drift>1e-11:raise ArithmeticError(f'Givens gauge drift {drift}')
    out['angle_words']=words;out['U']=decoded[u.shape[1]:].reshape(data['U'].shape)
    return out,{'max_component_drift_up_to_row_sign':drift,'redundant_bit_identity':'bit9 = bit8','brot':brot}

def xor_and(b,left,right,targets):
    targets=list(targets)
    if not targets:return
    q=targets[0]
    for t in targets[1:]:b.cx(q,t)
    b.ccx(left,right,q)
    for t in reversed(targets[1:]):b.cx(q,t)

def pebbled_init(b,table,address,outputs,scratch):
    address,outputs,scratch=map(list,(address,outputs,scratch))
    H,n=len(table),len(address)
    coefficients=B.mobius_transform(list(table)+[0]*((1<<n)-H))
    coefficients[H:]=[0]*((1<<n)-H)
    completed=B.mobius_transform(coefficients)
    assert completed[:H]==table
    stats={'monomial_ands':0,'recomputed_ands':0,'pebble_sites':0,'direct_leaf_ands':0,'scratch':len(scratch)}
    @lru_cache(None)
    def children(mask,last):return tuple((mask|(1<<j),j) for j in range(last+1,n) if (mask|(1<<j))<H)
    @lru_cache(None)
    def need(mask,last):
        return max([1+need(c,j) for c,j in children(mask,last) if children(c,j)]+[0])
    def targets(mask):
        v=coefficients[mask]
        return [outputs[j] for j in range(len(outputs)) if (v>>j)&1]
    def walk(mask,last,control,path,free):
        erased=[]
        if mask.bit_count()==5 and need(mask,last)>len(free):
            m=need(mask,last)-len(free)
            assert 0<m<=len(path)-1
            erased=path[:m]
            for a,c,t in reversed(erased):b.del_and(a,c,t);free.append(t)
            stats['pebble_sites']+=1
        ts=targets(mask)
        if ts:
            if control is None:b.x(km.array(ts))
            else:b.cx(control,km.array(ts))
        for child,j in children(mask,last):
            if not mask:
                walk(child,j,address[j],path,free)
            elif not children(child,j):
                ts=targets(child)
                if ts:xor_and(b,control,address[j],ts);stats['direct_leaf_ands']+=1
            else:
                assert free,('scratch exhausted',mask,j,len(path))
                t=free.pop();b.init_and(control,address[j],t);stats['monomial_ands']+=1
                walk(child,j,t,path+[(control,address[j],t)],free)
                b.del_and(control,address[j],t);free.append(t)
        for a,c,t in erased:
            free.remove(t);b.init_and(a,c,t);stats['recomputed_ands']+=1
    walk(0,-1,None,[],scratch.copy())
    return completed,stats

def build(delta,brot=10,return_report=False):
    delta=np.asarray(delta,dtype=np.int64);A,J=delta.shape
    assert brot==10 and J==3 and A%4==0
    assert not np.any(((delta>>9)^(delta>>8))&1)
    W=10*J;BW=9*J;H=A//4;n=(A-1).bit_length()
    packed=[B.pack(row,10) for row in delta]
    ds=[[],[],[]];base=[]
    compact=lambda row:sum(((row>>(j*10))&511)<<(j*9) for j in range(J))
    for h in range(H):
        v=packed[4*h:4*h+4];base.append(v[0])
        ds[0].append(compact(v[0]^v[1]));ds[1].append(compact(v[0]^v[2]))
        ds[2].append(compact(v[0]^v[1]^v[2]^v[3]))
    def expand(v):
        return sum((((v>>(9*j))&511)|(((v>>(9*j+8))&1)<<9))<<(10*j) for j in range(J))
    restored=[base[h]^expand((ds[0][h] if a&1 else 0)^(ds[1][h] if a&2 else 0)^(ds[2][h] if (a&3)==3 else 0)) for h in range(H) for a in range(4)]
    assert restored==packed
    values=[base[h]|(ds[0][h]<<W)|(ds[1][h]<<(W+BW))|(ds[2][h]<<(W+2*BW)) for h in range(H)]
    b=km.CircuitBuilder();sel=b.create_quantum_register(n,'selection')
    tgt=[q for j in range(J) for q in b.create_quantum_register(10,f'word_{j}')]
    bank=list(b.alloc_qubits(3*BW));scratch=b.alloc_qubits(5)
    completed,stats=pebbled_init(b,values,sel[2:],tgt+bank,scratch);b.free(scratch)
    both=b.alloc_qubits(1)[0];b.init_and(sel[0],sel[1],both)
    for s,ctl in enumerate((sel[0],sel[1],both)):
        for j in range(J):
            for k in range(9):
                ts=[tgt[j*10+k]]+([tgt[j*10+9]] if k==8 else [])
                xor_and(b,ctl,bank[s*BW+j*9+k],ts)
    b.del_and(sel[0],sel[1],both);b.free(both)
    erase=B.del_lookup_anf(b,table=[v>>W for v in completed],address=sel[2:],target=bank,low_bits=5)
    b.free(km.array(bank));c=b.finish_circuit()
    report={'num_qubits':c.num_qubits,'static_toffoli':int(c.operation_statistics().num_toffoli_like),'table_identity':'PASS','init':stats,'cleanup':erase}
    assert c.num_qubits<=128
    return (c,report) if return_report else c
