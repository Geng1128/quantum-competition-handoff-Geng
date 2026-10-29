"""Explicit 20-AND GF64 inverse XOR circuit with exact measurement cleanup.

All masks below describe physical linear forms. Prefix source order is x0..5,
six raw norm rank-products, then the three inverse-norm bits. Constant is bit0
of a form mask; physical source i is bit(i+1).
"""
try:
    from .baseline import *
except ImportError:
    from baseline import *
from contextlib import contextmanager
import json

DAG=json.loads(Path(__file__).with_name('optimized.json').read_text())

@contextmanager
def expose_pair(b,registers,left,right):
    u,v=left>>1,right>>1;p=next(bits(u));ops=[]
    for j in bits(u^(1<<p)):b.cx(registers[j],registers[p]);ops.append((j,p))
    if v>>p&1:v^=u^(1<<p)
    q=next(bits(v&~(1<<p)))
    for j in bits(v^(1<<q)):b.cx(registers[j],registers[q]);ops.append((j,q))
    if left&1:b.x(registers[p])
    if right&1:b.x(registers[q])
    yield registers[p],registers[q]
    if right&1:b.x(registers[q])
    if left&1:b.x(registers[p])
    for j,k in reversed(ops):b.cx(registers[j],registers[k])

def form_xor(b,registers,mask,target):
    if mask&1:b.x(target)
    for i in bits(mask>>1):b.cx(registers[i],target)

def norm_forms(K):
    forms=[0]*3;f=field(3)
    for i,v in enumerate(square_columns(3)):
        for j in bits(v):forms[j]^=1<<(1+i)
        for j in bits(f.mul(K,v)):forms[j]^=1<<(1+3+i)
    for i,(_,v)in enumerate(tensor(3)):
        for j in bits(v):forms[j]^=1<<(1+6+i)
    return forms

def inverse3_virtual(b,registers,forms):
    out=b.alloc_qubits(3)
    for mask,vector in enumerate(inverse3_anf()):
        if not vector:continue
        ix=list(bits(mask))
        if len(ix)==1:
            for j in bits(vector):form_xor(b,registers,forms[ix[0]],out[j])
        else:
            assert len(ix)==2
            with expose_pair(b,registers,forms[ix[0]],forms[ix[1]])as(p,q):rankone_xor(b,p,q,out,vector)
    return out

def phase_inverse3_virtual(b,registers,forms,mask):
    for monomial,vector in enumerate(inverse3_anf()):
        if not vector:continue
        ix=list(bits(monomial))
        if len(ix)==1:
            assert not(forms[ix[0]]&1)
            for i in bits(forms[ix[0]]>>1):
                for j in bits(vector):b.cz(registers[i],mask[j])
        else:
            with expose_pair(b,registers,forms[ix[0]],forms[ix[1]])as(p,q):
                for j in bits(vector):b.ccz(p,q,mask[j])

def inverse6_xor20(b,x,z=None,keep_norm=False):
    """z ^= inverse(x), inv0=0; 20T,21 total Q for input6+output6.

    Arbitrary target is supported. If omitted, allocate clean six-bit target.
    Exactly six norm ANDs, three inverse3 ANDs, eleven final ANDs; cleanup0T.
    """
    assert len(x)==6
    to,frm,K=tower(6);linear_inplace(b,x,to)
    if z is not None:linear_inplace(b,z,to)
    g=b.alloc_qubits(6)
    for i,(mask,_)in enumerate(tensor(3)):
        with expose_pair(b,x,mask<<1,mask<<4)as(p,q):b.init_and(p,q,g[i])
    prefix=list(x)+list(g);forms=norm_forms(K)
    w=inverse3_virtual(b,prefix,forms);source=prefix+list(w)
    if z is None:z=b.alloc_qubits(6)
    for j,expression in enumerate(DAG['outputs']):form_xor(b,source,expression&65535,z[j])
    for i,(left,right)in enumerate(DAG['gates']):
        vector=sum(((expression>>(16+i))&1)<<j for j,expression in enumerate(DAG['outputs']))
        assert vector
        with expose_pair(b,source,left,right)as(p,q):rankone_xor(b,p,q,z,vector)
    wm=b.alloc_bits(3);b.hmr(w,wm);b.free(w)
    phase_inverse3_virtual(b,prefix,forms,wm);b.free(wm)
    kept=None
    if keep_norm:
        kept=b.alloc_qubits(3)
        for j,form in enumerate(forms):form_xor(b,prefix,form,kept[j])
    for i,(mask,_)in reversed(list(enumerate(tensor(3)))):
        with expose_pair(b,x,mask<<1,mask<<4)as(p,q):b.del_and(p,q,g[i])
    b.free(g);linear_inplace(b,x,frm);linear_inplace(b,z,frm)
    return (z,kept)if keep_norm else z

def inverse_initialized(b,x,target=None):
    """No-scratch recursive initialized inverse using the20T base at n6."""
    n=len(x)
    if n==3:return inverse3_noscratch(b,x,target)
    if n==6:return inverse6_xor20(b,x,target)
    k=n//2;to,frm,K=tower(n);linear_inplace(b,x,to)
    norm=b.alloc_qubits(k);f=field(k)
    linear_xor(b,x[:k],norm,square_columns(k))
    linear_xor(b,x[k:],norm,[f.mul(K,v)for v in square_columns(k)])
    mul_xor_noscratch(b,x[:k],x[k:],norm)
    invnorm=inverse_initialized(b,norm)
    ar._erase_norm(b,x,norm,K);b.free(norm)
    if target is None:target=b.alloc_qubits(n)
    b.cx(x[k:],x[:k]);mul_xor_noscratch(b,x[:k],invnorm,target[:k]);mul_xor_noscratch(b,x[k:],invnorm,target[k:]);b.cx(x[k:],x[:k])
    ar._erase_norm(b,target,invnorm,K);b.free(invnorm)
    linear_inplace(b,x,frm);linear_inplace(b,target,frm)
    return target

def norm_xor(b,x,z,K):
    k=len(z);f=field(k)
    linear_xor(b,x[:k],z,square_columns(k))
    linear_xor(b,x[k:],z,[f.mul(K,v)for v in square_columns(k)])
    mul_xor_noscratch(b,x[:k],x[k:],z)

def inverse12_keep_norm(b,x):
    """65T inverse12 keeping its6-bit norm; used by closed reverse24."""
    assert len(x)==12
    to,frm,K=tower(12);linear_inplace(b,x,to)
    N=b.alloc_qubits(6);norm_xor(b,x,N,K);w=inverse6_xor20(b,N)
    target=b.alloc_qubits(12);b.cx(x[6:],x[:6])
    mul_xor_noscratch(b,x[:6],w,target[:6]);mul_xor_noscratch(b,x[6:],w,target[6:]);b.cx(x[6:],x[:6])
    ar._erase_norm(b,target,w,K);b.free(w)
    linear_inplace(b,x,frm);linear_inplace(b,target,frm)
    return target,N

def inverse_inplace_forward20(b,x):
    target=inverse_initialized(b,x);mask=b.alloc_bits(len(x))
    b.hmr(x,mask);b.swap(x,target);b.free(target);return mask

def inverse_inplace_reverse20(b,x,mask):
    """Closed reverse12/24 at71/215T,21/42 private qubits."""
    n=len(x);assert n in(12,24);k=n//2
    to,frm,K=tower(n);linear_inplace(b,x,to)
    N=b.alloc_qubits(k);norm_xor(b,x,N,K)
    t,D=inverse12_keep_norm(b,N)if n==24 else inverse6_xor20(b,N,keep_norm=True)
    out=b.alloc_qubits(k);mul_xor_noscratch(b,x[k:],t,out)
    erase_product(b,out,N,x[k:]);mul_xor_noscratch(b,x[:k],t,x[k:]);b.cx(out,x[k:])
    b.cx(out,x[k:]);erase_product(b,x[k:],N,x[:k]);b.cx(out,x[k:])
    b.swap(x[:k],x[k:]);b.swap(x[k:],out);b.free(out)
    ar._erase_inverse_with_norm(b,t,N,D);b.free(N)
    tok,frmk,Kk=tower(k);linear_inplace(b,t,tok)
    u=b.alloc_qubits(k//2);norm_xor(b,t,u,Kk);erase_inverse(b,u,D);b.free(D)
    ar._erase_norm(b,t,u,Kk);b.free(u);linear_inplace(b,t,frmk)
    ar._erase_norm(b,x,t,K);b.free(t);linear_inplace(b,x,frm)
    b.cz(x,mask);b.free(mask)

def inverse24_forward20(b,x,K=None):
    cols,frm,_=extension_columns(K);linear_inplace(b,x,cols)
    mask=inverse_inplace_forward20(b,x);linear_inplace(b,x,frm);return mask

def inverse24_reverse20(b,x,mask,K=None):
    cols,frm,_=extension_columns(K);linear_inplace(b,x,cols)
    inverse_inplace_reverse20(b,x,mask);linear_inplace(b,x,frm)
