"""In-place structured GF12 kernels using exp/arithmetic primitives."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'arithmetic'))
import field as ar
from hankel_check import mul,inv,mi,power
from functools import lru_cache

def ccols(c):return [mul(c,1<<i) for i in range(12)]
def cx(b,x,y,c=1):
    if c==1:b.cx(x,y)
    elif c:ar.linear_xor(b,x,y,ccols(c))
def scale_const(b,x,c):
    if c!=1:ar.linear_inplace(b,x,ccols(c))

def expose(b,words,coeffs):
    p=next(i for i,c in enumerate(coeffs) if c);c=coeffs[p]
    scale_const(b,words[p],c)
    for i,a in enumerate(coeffs):
        if i!=p:cx(b,words[i],words[p],a)
    return words[p],p

def unexpose(b,words,coeffs,p):
    for i in reversed(range(len(coeffs))):
        if i!=p:cx(b,words[i],words[p],coeffs[i])
    scale_const(b,words[p],inv(coeffs[p]))

def rank_update(b,aw,ac,bw,bc,tw,tc):
    x,ap=expose(b,aw,ac);y,bp=expose(b,bw,bc)
    p=next(i for i,c in enumerate(tc) if c);v=tc[p];vi=inv(v)
    for i,c in enumerate(tc):
        if i!=p:cx(b,tw[p],tw[i],mul(c,vi))
    scale_const(b,tw[p],vi)
    ar.mul_xor(b,x,y,tw[p])
    scale_const(b,tw[p],v)
    for i in reversed(range(len(tc))):
        if i!=p:cx(b,tw[p],tw[i],mul(tc[i],vi))
    unexpose(b,bw,bc,bp);unexpose(b,aw,ac,ap)

@lru_cache(None)
def vandermonde(n):
    V=[[power(t,j) for j in range(n)] for t in range(n)]
    return V,mi(V)

@lru_cache(None)
def cross_terms(k,m):
    n=k+m-1;V,W=vandermonde(n)
    return [(V[t][:k][::-1],[W[j][t] for j in range(n)],V[t][:m]) for t in range(n)]

def tri(b,q,x,reverse=False):
    # q=(q1,...,q_(n-1)), q0=1 implicit.
    n=len(x)
    if n==1:return
    k=n//2;m=n-k
    if not reverse:tri(b,q[:m-1],x[k:])
    else:tri(b,q[:k-1],x[:k],True)
    ts=cross_terms(k,m)
    for ac,bc,tc in (reversed(ts) if reverse else ts):rank_update(b,x[:k],ac,q[:n-1],bc,x[k:],tc)
    if not reverse:tri(b,q[:k-1],x[:k])
    else:tri(b,q[:m-1],x[k:],True)

def hankel(b,B,A,reverse=False):
    # H_ij=B_(i+j+1), implicit monic B_degree=len(A).
    assert len(B)==len(A)
    q=list(reversed(B[1:]))
    if reverse:
        for i in range(len(A)//2):b.swap(A[i],A[-1-i])
        tri(b,q,A,True)
    else:
        tri(b,q,A)
        for i in range(len(A)//2):b.swap(A[i],A[-1-i])

def convolution_xor(b,a,c,out,reverse=False):
    # Coefficients 0..len(out)-1 of a*c; supports short output truncation.
    N=len(a)+len(c)-1;V,W=vandermonde(N)
    terms=range(N-1,-1,-1) if reverse else range(N)
    for t in terms:
        tc=[W[j][t] for j in range(len(out))]
        if any(tc):rank_update(b,a,V[t][:len(a)],c,V[t][:len(c)],out,tc)

def convolution_transpose_left_xor(b,source,c,out,reverse=False):
    # out[j] += sum_i source[i]*c[i-j], transpose of outpoly*c truncated to source.
    N=len(out)+len(c)-1;V,W=vandermonde(N)
    terms=range(N-1,-1,-1) if reverse else range(N)
    for t in terms:
        ac=[W[i][t] for i in range(len(source))]
        if any(ac):rank_update(b,source,ac,c,V[t][:len(c)],out,V[t][:len(out)])
