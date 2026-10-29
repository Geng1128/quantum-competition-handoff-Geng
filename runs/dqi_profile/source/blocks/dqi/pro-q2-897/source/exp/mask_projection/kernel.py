"""Classical mask projection onto the70 syndrome coordinates,0 Toffolis.

Interpolate a degree<=70 polynomial F with F(0)=0 through70 selected masks
in the trace-dual field basis. Pay Tr(sum S_k F_(k+1)) directly, subtract
F's evaluations from every mask, and omit the70 now-zero scan positions.
An additive FFT avoids a dense4095x70 classical CNOT transform.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from exp.decoder.common import *

SELECTED=tuple(range(70))

def ev(poly,x):
    y=0
    for a in reversed(poly):y=mul(y,x)^a
    return y

def tr(x):
    y=0
    for _ in range(12):y^=x;x=mul(x,x)
    assert y in(0,1)
    return y

@lru_cache(None)
def trace_columns():
    return tuple(sum(tr(mul(1<<i,1<<j))<<i for i in range(12))for j in range(12))

@lru_cache(None)
def subspace_polynomials():
    out=[[0,1]]
    for j in range(12):
        s=out[-1];d=ev(s,1<<j);assert d
        z=[0]*(2*len(s)-1)
        for i,c in enumerate(s):z[2*i]^=mul(c,c);z[i]^=mul(d,c)
        out.append(z)
    return out

@lru_cache(None)
def novel_basis(active):
    ss=subspace_polynomials();out=[]
    for i in range(active):
        p=[1]
        for j in range(i.bit_length()):
            if i>>j&1:p=pmul(p,ss[j])
        assert len(p)==i+1 and p[-1]==1
        out.append(tuple(p))
    return tuple(out)

def to_novel(b,words,active,reverse=False):
    basis=novel_basis(active)
    for i in(range(1,active)if reverse else range(active-1,0,-1)):
        for j,c in enumerate(basis[i][:-1]):cx(b,words[i],words[j],c)

def additive_fft(b,words,m=12,offset=0,active=71,reverse=False):
    """Novel-basis coefficients -> values at offset+span(1,...,2^(m-1)).

    Only the first active coefficients may be nonzero on entry. The inverse
    restores the unused classical wires tozero. Output order is numeric field
    representation, not primitive-element exponent order.
    """
    assert len(words)==1<<m and 0<active<=len(words)
    if not m:return
    h=1<<(m-1);lo,hi=words[:h],words[h:];high=max(0,active-h)
    nextactive=min(active,h);ss=subspace_polynomials()[m-1]
    c=ev(ss,offset);d=ev(ss,1<<(m-1));assert d
    if reverse:
        additive_fft(b,lo,m-1,offset,nextactive,True)
        additive_fft(b,hi,m-1,offset^(1<<(m-1)),nextactive,True)
    for j in range(nextactive):
        if j>=high:b.cx(lo[j],hi[j])
        elif not reverse:
            cx(b,hi[j],lo[j],c);scale(b,hi[j],d);b.cx(lo[j],hi[j])
        else:
            b.cx(lo[j],hi[j]);scale(b,hi[j],inv(d));cx(b,hi[j],lo[j],c)
    if not reverse:
        additive_fft(b,lo,m-1,offset,nextactive)
        additive_fft(b,hi,m-1,offset^(1<<(m-1)),nextactive)

@lru_cache(None)
def interpolation_matrix():
    alpha=gf.primitive_element
    return tuple(tuple(power(power(alpha,j),k+1)for k in range(70))for j in SELECTED)

def residual_masks(b,E,F):
    beta=1;T=trace_columns()
    for mask in E:
        ar.linear_xor(b,F[beta],mask,T);beta=mul(beta,gf.primitive_element)
    assert beta==1

def project(b,S,E):
    assert len(S)==70 and len(E)==4095
    F=[b.alloc_bits(12)for _ in range(4096)]
    b.bit_store(combine(F),False)
    Ti=ar.inverse_columns(trace_columns())
    for k,j in enumerate(SELECTED):
        b.cx(E[j],F[k+1]);ar.linear_inplace(b,F[k+1],Ti)
    linear_words(b,F[1:71],interpolation_matrix(),reverse=True)
    for s,f in zip(S,F[1:71]):
        ar.linear_inplace(b,f,trace_columns());b.cz(s,f);ar.linear_inplace(b,f,Ti)
    to_novel(b,F,71);additive_fft(b,F)
    residual_masks(b,E,F)
    return F

def restore(b,E,F):
    residual_masks(b,E,F)
    # F is classical deterministic working memory, not quantum garbage.
    for word in F:b.free(word)
