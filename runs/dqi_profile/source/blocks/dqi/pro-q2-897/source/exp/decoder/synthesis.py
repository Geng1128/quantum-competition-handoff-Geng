"""Head-absorption synthesis with a complete deferred measurement transcript.

The rational interpolation chart is exact on valid heads without extension
poles. Zero evaluation factors are replaced by one in the remainder map, so
arbitrary heads are preserved; that repair can change the decoded rational
function after an extension pole. The top-coefficient promises are separate.
"""
from common import *
from extract import constants
import division as dv
from lowwidth import divide_lowwidth375
sys.path.insert(0,str(ROOT/'exp/convert'))
from kernels import rank_update,vandermonde
import kernels as ck
from exp.noscratch.kernels import mul_xor_noscratch
from convert import inverse_context
from safe_bridge import divide_zero_bridge

def tail_recur(b,heads,p,q,d,reverse=False):
    for a,z in (reversed(heads) if reverse else heads):
        if reverse:b.swap(p,q);ar.mul_xor(b,z,q,p)
        ar.mul_xor(b,p,q,d)
        b.cx(z,a);ar.linear_inplace(b,q,ar.square_columns(12));ar.mul_xor(b,a,q,d)
        ar.linear_inplace(b,q,[power(1<<j,2048) for j in range(12)]);b.cx(z,a)
        if not reverse:ar.mul_xor(b,z,q,p);b.swap(p,q)

@lru_cache(None)
def quotient_terms(n,kap):
    """floor(l*A/q): A0 does not contribute; n rank-one products."""
    V,W=vandermonde(n);out=[]
    for t in range(n):
        polynomial=[0]+[W[j][t] for j in range(n)]
        quotient,_=divmodp(polynomial,[kap,1,1]);quotient+= [0]*(n-1-len(quotient))
        out.append(([1,t],V[t][:n-1],quotient))
    return out

def quotient_xor(b,head,A,B,kap,reverse=False):
    assert len(B)==len(A)-1
    terms=quotient_terms(len(A),kap)
    for ac,bc,tc in (reversed(terms) if reverse else terms):
        if any(tc):rank_update(b,head,ac,A[1:],bc,B,tc)

def full_product_xor(b,head,A,target):
    n=len(A)+1;V,W=vandermonde(n)
    for t in range(n):rank_update_noanc(b,head,[1,t],A,V[t][:len(A)],target,[W[j][t] for j in range(n)])

def rank_update_noanc(b,aw,ac,bw,bc,tw,tc):
    x,ap=ck.expose(b,aw,ac);y,bp=ck.expose(b,bw,bc)
    p=next(i for i,c in enumerate(tc) if c);v=tc[p];vi=inv(v)
    for i,c in enumerate(tc):
        if i!=p:cx(b,tw[p],tw[i],mul(c,vi))
    scale(b,tw[p],vi);mul_xor_noscratch(b,x,y,tw[p]);scale(b,tw[p],v)
    for i in reversed(range(len(tc))):
        if i!=p:cx(b,tw[p],tw[i],mul(tc[i],vi))
    ck.unexpose(b,bw,bc,bp);ck.unexpose(b,aw,ac,ap)

@lru_cache(None)
def embedding_matrix(n,kap):
    # Input [remainder0,remainder1,quotient0,...] -> q*quotient+remainder.
    rows=[[0]*n for _ in range(n)];rows[0][0]=1;rows[1][1]=1
    for j in range(n-2):
        for k,c in enumerate([kap,1,1]):rows[j+k][j+2]=c
    return tuple(tuple(r) for r in rows)

@lru_cache(None)
def evaluation_coefficients(n,c):
    g=c|(1<<12);out=[1]
    for j in range(1,n):out.append(emul(out[-1],g))
    return tuple(out)

def expose_eval(b,A,c,reverse=False):
    es=evaluation_coefficients(len(A),c)
    if reverse:
        for i in reversed(range(2,len(A))):cx(b,A[i],A[1],es[i]>>12)
        for i in reversed(range(2,len(A))):cx(b,A[i],A[0],es[i]&4095)
        cx(b,A[1],A[0],c)
    else:
        cx(b,A[1],A[0],c)
        for i in range(2,len(A)):cx(b,A[i],A[0],es[i]&4095)
        for i in range(2,len(A)):cx(b,A[i],A[1],es[i]>>12)
    return combine(A[:2])

def absorb(b,head,A,B,c,kap,reverse=False,transcript=None):
    out=head+B
    if not reverse:
        value=expose_eval(b,A,c);flag=dv.zero_flag(b,value);expose_eval(b,A,c,True)
        # Zero-safe factor: A(gamma)=0 gets factor1. Adding1 to A changes
        # l*A only in its remainder, so the quotient map is unchanged.
        b.cx(flag,A[0][0])
        spare=[b.alloc_qubits(12),b.alloc_qubits(12)];target=spare+B
        linear_words(b,target,embedding_matrix(len(target),kap))
        full_product_xor(b,head,A,target)
        b.cx(flag,A[0][0])
        transcript=b.alloc_bits(24);b.hmr(combine(head),transcript)
        for h,z in zip(head,spare):b.swap(h,z);b.free(z)
        flagmask=b.alloc_bits(1);b.hmr(flag,flagmask[0]);b.free(flag)
        return out,A,(transcript,flagmask)
    linear_words(b,out,embedding_matrix(len(out),kap),True)
    value=expose_eval(b,A,c)
    cx(b,head[1],head[0],c)
    headmask,flagmask=transcript
    divide_zero_bridge(b,combine(head),value,flagmask[0]);b.free(flagmask)
    cx(b,head[1],head[0],c)
    b.cz(combine(head),headmask);b.free(headmask)
    expose_eval(b,A,c,True)
    quotient_xor(b,head,A,B,kap,True)
    return out,A,transcript

def rotate(b,words,control,reverse=False):
    # multiply by x using a top-zero cyclic register, controlled by even chart.
    for i in (range(1,len(words)) if reverse else reversed(range(1,len(words)))):
        b.cswap(control,words[i-1],words[i])

def flag_zero(b,q,flag):
    ladder=zero_hold(b,q);b.cx(ladder[-1],flag);zero_free(b,q,ladder)

def tail_scale_forward(b,p,q,d):
    with inverse_context(b,q) as t:
        out=b.alloc_qubits(12);mul_xor_noscratch(b,d,t,out)
        mask=b.alloc_bits(12);b.hmr(d,mask);b.swap(d,out);b.free(out)
    return mask

def tail_scale_reverse(b,p,q,d,mask):
    with inverse_context(b,q) as t:
        out=b.alloc_qubits(12);mul_xor_noscratch(b,d,q,out)
        newer=b.alloc_bits(12);b.hmr(d,newer);b.swap(d,out);b.free(out)
        ar.phase_product(b,d,t,newer);b.free(newer)
    b.cz(d,mask);b.free(mask)

def synthesize(b,words,keep_leading=True):
    """Return A,B, transcript. Exactly70datawords after forward completion."""
    h=len(words)//2;assert h%2==1
    cs,ks,_,_=constants(h);heads=[words[2*i:2*i+2] for i in range(h)]
    p,q,d=[b.alloc_qubits(12) for _ in range(3)];b.x(q[0])
    tail_recur(b,heads,p,q,d)
    dmask=tail_scale_forward(b,p,q,d)
    # Uniform tail: when q=0,d=0, [d/q,p]/q is x times the old even
    # chart [p]/0. Every transfer is polynomial-linear, so the required
    # padding x is already present; no parity bit or outer shift is needed.
    A=[d,p];B=[q];history=[]
    for i in reversed(range(h)):
        oldA,oldB=A,B
        A,B,m=absorb(b,heads[i],A,B,cs[i],ks[i])
        history.append((heads[i],oldA,oldB,cs[i],ks[i],m))
    # Highest A coefficient=0 and highest B coefficient=1 are determinant
    # identities. The second A coefficient can be nonzero on a bad tail.
    extra=[A[h],A[h+1],B[h]];b.x(B[h][0])
    freed=extra[1:] if keep_leading else extra
    for reg in freed:b.free(reg)
    transcript=dict(h=h,heads=heads,p=p,q=q,d=d,dmask=dmask,history=history,A=A,B=B,extra=extra,freed=freed,keep_leading=keep_leading)
    return A[:h],B[:h],transcript

def unsynthesize(b,transcript):
    t=transcript;h=t['h'];A=t['A'];B=t['B']
    reclaim(b,t['freed'])
    b.x(B[h][0])
    for head,oldA,oldB,c,kap,mask in reversed(t['history']):absorb(b,head,oldA,oldB,c,kap,True,mask)
    p,q,d=t['p'],t['q'],t['d']
    tail_scale_reverse(b,p,q,d,t['dmask'])
    tail_recur(b,t['heads'],p,q,d,True);b.x(q[0])
    b.free(p);b.free(q);b.free(d)
