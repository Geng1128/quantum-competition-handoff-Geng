"""Lead's finite-field and reversible linear utilities (GF12 word bank)."""
from functools import lru_cache
from pathlib import Path
import sys
import circ as km

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'exp/arithmetic'))
import field as ar
gf=km.GF2Field(12)
K=2048

def mul(a,b):return gf.mul(a,b)
def inv(a):return gf.invert(a) if a else 0
def power(a,n):return gf.pow(a,n)
def xor(xs):
    out=0
    for x in xs:out^=x
    return out
def trim(p):
    p=list(p)
    while len(p)>1 and not p[-1]:p.pop()
    return p
def add(a,b):return trim([(a[i] if i<len(a) else 0)^(b[i] if i<len(b) else 0) for i in range(max(len(a),len(b)))])
def pmul(a,b):
    c=[0]*(len(a)+len(b)-1)
    for i,x in enumerate(a):
        for j,y in enumerate(b):c[i+j]^=mul(x,y)
    return trim(c)
def divmodp(a,b):
    a=trim(a);out=[0]*max(1,len(a)-len(b)+1)
    assert b[-1]
    bi=inv(b[-1])
    for k in range(len(a)-len(b),-1,-1):
        t=mul(a[k+len(b)-1],bi);out[k]=t
        for j,v in enumerate(b):a[k+j]^=mul(t,v)
    return trim(out),trim(a)
def emul(x,y,k=K):
    a,b=x&4095,x>>12;c,d=y&4095,y>>12;bd=mul(b,d)
    return (mul(a,c)^mul(k,bd))|((mul(a,d)^mul(b,c)^bd)<<12)
def einv(x,k=K):
    a,b=x&4095,x>>12;t=inv(mul(a,a)^mul(a,b)^mul(k,mul(b,b)))
    return mul(a^b,t)|(mul(b,t)<<12)
def eev(p,x):
    v=0
    for a in reversed(p):v=emul(v,x)^a
    return v

@lru_cache(None)
def constcols(c):return tuple(mul(c,1<<i) for i in range(12))
def cx(b,x,z,c=1):
    if c==1:b.cx(x,z)
    elif c:ar.linear_xor(b,x,z,constcols(c))
def scale(b,x,c):
    if c!=1:ar.linear_inplace(b,x,constcols(c))
def scalepair(b,x,c):
    scale(b,x[:12],c);scale(b,x[12:],c)
def combine(words):return km.array([q for w in words for q in w])
def wordlist(reg):return [km.array(list(reg)[i:i+12]) for i in range(0,len(reg),12)]

@lru_cache(None)
def matrix_ops(rows):
    """Elementary row operations reducing an invertible GF12 matrix to I."""
    a=[list(r) for r in rows];n=len(a);ops=[]
    assert all(len(r)==n for r in a)
    for j in range(n):
        p=next(i for i in range(j,n) if a[i][j])
        if p!=j:a[p],a[j]=a[j],a[p];ops.append(('swap',p,j,0))
        v=inv(a[j][j])
        if v!=1:a[j]=[mul(x,v) for x in a[j]];ops.append(('scale',j,0,v))
        for i in range(n):
            if i!=j and a[i][j]:
                v=a[i][j];a[i]=[x^mul(y,v) for x,y in zip(a[i],a[j])];ops.append(('add',j,i,v))
    return tuple(ops)

def linear_words(b,words,rows,reverse=False):
    ops=matrix_ops(tuple(tuple(r) for r in rows))
    for op,i,j,v in (ops if reverse else reversed(ops)):
        if op=='swap':b.swap(words[i],words[j])
        elif op=='scale':scale(b,words[i],v if reverse else inv(v))
        else:cx(b,words[i],words[j],v)

def zero_hold(b,x):
    """Return a held all-zero ladder, len(x)-1 Toffolis."""
    b.x(x);lad=b.alloc_qubits(len(x)-1)
    b.init_and(x[0],x[1],lad[0])
    for j in range(2,len(x)):b.init_and(lad[j-2],x[j],lad[j-1])
    b.x(x)
    return lad
def zero_free(b,x,lad):
    b.x(x)
    for j in reversed(range(2,len(x))):b.del_and(lad[j-2],x[j],lad[j-1])
    b.del_and(x[0],x[1],lad[0]);b.free(lad);b.x(x)

def reclaim(b,regs):
    """Reclaim specified known-free physical wires, without new wire IDs."""
    wanted={str(q) for reg in regs for q in reg}
    available=b.alloc_qubits(b.finish_circuit().num_qubits-b.num_allocated_qubits)
    have={str(q) for q in available}
    assert wanted<=have,(wanted-have)
    b.free(km.array([q for q in available if str(q) not in wanted]))

def extmul_xor(b,x,y,z):
    """126T product in explicit a+b*y basis; one scratch bit."""
    a,c=x[:12],x[12:];d,e=y[:12],y[12:]
    b.cx(z[:12],z[12:]);ar.mul_xor(b,a,d,z[:12]);b.cx(z[:12],z[12:])
    scale(b,z[:12],inv(K))
    ar.mul_xor(b,c,e,z[:12])
    scale(b,z[:12],K)
    b.cx(c,a);b.cx(e,d);ar.mul_xor(b,a,d,z[12:]);b.cx(e,d);b.cx(c,a)

def phase_extmul(b,x,y,mask):
    for i in range(24):
        for j in range(24):
            v=emul(1<<i,1<<j)
            for k in ar.bits(v):b.ccz(x[i],y[j],mask[k])

def write(s,qs,vals):
    import numpy as np
    vals=np.asarray(vals,dtype=np.uint64)
    s.write_across_shots(km.array(qs),((vals[None,:]>>np.arange(len(qs),dtype=np.uint64)[:,None])&1).astype(bool))
def read(s,qs):
    import numpy as np
    return (s.read_across_shots(km.array(qs)).astype(np.uint64)*(np.uint64(1)<<np.arange(len(qs),dtype=np.uint64))[:,None]).sum(axis=0)
