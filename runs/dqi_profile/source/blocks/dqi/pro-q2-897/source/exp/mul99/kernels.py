"""GF24 multiplication: GF16 degree6, 11 evaluations times rank9 =99.

Separate experimental implementation; does not edit the stable arithmetic API.
"""
from functools import lru_cache
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'exp/arithmetic'))
import field as ar
from field import *


def extension_tensor(n,k,d,subterms):
    f=field(n);sub=field(k);gen=f.pow(f.primitive_element,(2**n-1)//(2**k-1))
    def ev(v,p):
        out=0
        for i in reversed(range(k+1)):out=f.mul(out,p)^((v>>i)&1)
        return out
    root=next(f.pow(gen,i) for i in range(1,2**k) if ev(sub.modulus,f.pow(gen,i))==0)
    embed=[f.pow(root,i) for i in range(k)]
    cols=[f.mul(v,f.pow(f.primitive_element,j)) for j in range(d) for v in embed]
    inv=inverse_columns(cols);masks=[]
    for point in list(range(2*d-2))+[None]:
        values=[]
        for co in inv:
            if point is None:value=(co>>(k*(d-1)))&((1<<k)-1)
            else:
                value=0
                for j in reversed(range(d)):value=sub.mul(value,point)^((co>>(k*j))&((1<<k)-1))
            values.append(value)
        for mask,_ in subterms:
            masks.append(sum(((v&mask).bit_count()%2)<<j for j,v in enumerate(values)))
    equations=[]
    for i in range(n):
        for j in range(i,n):
            row=sum((((m>>i)&1)&((m>>j)&1))<<r for r,m in enumerate(masks))
            equations.append((row,f.mul(1<<i,1<<j)))
    vectors=ar._solve_output_vectors(equations,len(masks));assert all(vectors)
    return tuple(zip(masks,vectors))


@lru_cache(None)
def tensor4():return extension_tensor(4,2,2,tensor(2))


@lru_cache(None)
def tensor99():
    out=extension_tensor(24,4,6,tensor4());assert len(out)==99
    return out


def multiplier(b,x,y,z,terms):
    assert not(set(map(str,x))&set(map(str,y)) or set(map(str,x))&set(map(str,z)) or set(map(str,y))&set(map(str,z)))
    for mask,v in terms:
        ix=list(bits(mask));iz=list(bits(v));p=ix[0];t=iz[0]
        for i in ix[1:]:b.cx(x[i],x[p]);b.cx(y[i],y[p])
        for j in iz[1:]:b.cx(z[t],z[j])
        b.ccx(x[p],y[p],z[t])
        for j in reversed(iz[1:]):b.cx(z[t],z[j])
        for i in reversed(ix[1:]):b.cx(y[i],y[p]);b.cx(x[i],x[p])


def mul_xor99_zero(b,x,y,z):return multiplier(b,x,y,z,tensor99())


@lru_cache(None)
def tensor99_extension(K=None):
    cols,invcols,_=extension_columns(K)
    return tuple((sum(((mask&col).bit_count()%2)<<j for j,col in enumerate(cols)),apply(invcols,v))
                 for mask,v in tensor99())


def mul24_xor99_zero(b,x,y,z,K=None):return multiplier(b,x,y,z,tensor99_extension(K))


def phase_product_quantum99(b,x,y,mask):
    for m,v in tensor99():
        ix=list(bits(m));iz=list(bits(v));p=ix[0];t=iz[0]
        for i in ix[1:]:b.cx(x[i],x[p]);b.cx(y[i],y[p])
        for j in iz[1:]:b.cx(mask[j],mask[t])
        b.ccz(x[p],y[p],mask[t])
        for j in reversed(iz[1:]):b.cx(mask[j],mask[t])
        for i in reversed(ix[1:]):b.cx(y[i],y[p]);b.cx(x[i],x[p])


# Stable public spellings: no zero-target promise is required.
mul_xor99=mul_xor99_zero
mul24_xor99=mul24_xor99_zero
