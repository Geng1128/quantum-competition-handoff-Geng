"""Partial-q-gauge synthesis. Same public A,B,transcript API as decoder.

70 input field words plus one rolling d field word and one boundary flag.
Only lower A/B coefficients are stored; implicit high coefficients are aliases
of earlier q-prefix slots. extra[0] is the physical A[h] zero on valid heads.
No edits to the live decoder; local measurement transcript is fully closed.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'exp/decoder'))
from common import *
from extract import constants
from exp.decoder import synthesis as base
from exp.partial_gauge import prefix as pg
from exp.partial_gauge.scalar import divide117
from exp.partial_gauge.tail import divide_safe138,multiply_safe165,controlled_linear_xor
from exp.mul99.schedules import divide_zero_bridge414
import division as dv
ck=base.ck

def xword(b,w,v):
    for j in ar.bits(v):b.x(w[j])

def add_alias(b,src,targets,coeffs,virtual_zero=None,constant=0):
    if src is None:
        for z,c in zip(targets,coeffs):xword(b,z,mul(constant,c))
    else:
        for z,c in zip(targets,coeffs):cx(b,src,z,c)
        if virtual_zero is not None:controlled_linear_xor(b,virtual_zero,src,targets,coeffs)

def eval_implicit(b,A,qi,c,reverse=False):
    if len(A)==1:
        assert qi is not None
        return base.expose_eval(b,A+[qi],c,reverse)
    v=base.evaluation_coefficients(len(A)+1,c)[-1]
    if reverse:
        add_alias(b,qi,A[:2],[v&4095,v>>12],constant=1)
        return base.expose_eval(b,A,c,True)
    out=base.expose_eval(b,A,c)
    add_alias(b,qi,A[:2],[v&4095,v>>12],constant=1);return out

def rank_affine(b,head,ac,words,bc,target,tc,constant=0):
    if not any(tc):return
    x,ap=ck.expose(b,head,ac);y,bp=ck.expose(b,words,bc)
    xword(b,y,constant)
    p=next(i for i,c in enumerate(tc) if c);v=tc[p];vi=inv(v)
    for i,c in enumerate(tc):
        if i!=p:cx(b,target[p],target[i],mul(c,vi))
    scale(b,target[p],vi);ar.mul_xor0(b,x,y,target[p]);scale(b,target[p],v)
    for i in reversed(range(len(tc))):
        if i!=p:cx(b,target[p],target[i],mul(tc[i],vi))
    xword(b,y,constant);ck.unexpose(b,words,bc,bp);ck.unexpose(b,head,ac,ap)

def projected_product(b,head,A,qi,target,kap):
    k=len(A)-1;n=k+3;V,W=base.vandermonde(n)
    words=A+([qi] if qi is not None else [])
    for t in range(n):
        tc=[W[j][t] for j in range(k+2)];top=W[k+2][t]
        tc[k]^=mul(kap,top);tc[k+1]^=top
        rank_affine(b,head,[1,t],words,V[t][:len(words)],target,tc,V[t][k+1] if qi is None else 0)

def truncated_quotient(b,head,A,qi,B,kap):
    k=len(B)
    if not k:return
    words=A[1:]+([qi] if qi is not None else [])
    for ac,bc,tc in reversed(base.quotient_terms(k+2,kap)):
        rank_affine(b,head,ac,words,bc[:len(words)],B,tc[:k],bc[k] if qi is None else 0)

def prefix_encode(b,heads,i,F,d=None):
    a,x=heads[i]
    if not i:
        if d is not None:b.cx(a,d);b.cx(x,d)
        return None
    q=heads[i-1][1]
    def after():
        ar.linear_inplace(b,q,ar.square_columns(12));ar.mul_xor0(b,a,q,d)
        ar.linear_inplace(b,q,[power(1<<j,2048) for j in range(12)]);ar.mul_xor0(b,x,q,d)
    callback=after if d is not None else None
    if i==1:return pg.boundary_forward(b,x,q,F,callback)
    return pg.forward(b,x,heads[i-2][1],q,F if i==2 else None,callback)

def prefix_decode(b,heads,i,F,m,d=None):
    a,x=heads[i]
    if not i:
        if d is not None:b.cx(a,d);b.cx(x,d)
        return
    q=heads[i-1][1]
    if d is not None:
        ar.mul_xor0(b,x,q,d)
        if i==1:b.ccx(F,q,d)
    if i==1:pg.boundary_reverse(b,x,q,F,m)
    else:pg.reverse_compact(b,x,heads[i-2][1],q,m,F if i==2 else None)
    if d is not None:
        ar.linear_inplace(b,q,ar.square_columns(12));ar.mul_xor0(b,a,q,d)
        ar.linear_inplace(b,q,[power(1<<j,2048) for j in range(12)])

def p_alias(heads,i,F):
    return (None,None,0) if i==0 else (None,None,1) if i==1 else (heads[i-2][1],F if i==2 else None,0)

def absorb(b,head,A,B,qi,p,virtual,pconst,c,kap,reverse=False,transcript=None):
    k=len(B);assert len(A)==k+1;out=head+B
    if not reverse:
        value=eval_implicit(b,A,qi,c);flag=dv.zero_flag(b,value);eval_implicit(b,A,qi,c,True)
        b.cx(flag,A[0][0]);spare=[b.alloc_qubits(12),b.alloc_qubits(12)];target=spare+B
        linear_words(b,target,base.embedding_matrix(k+2,kap));projected_product(b,head,A,qi,target,kap)
        b.cx(flag,A[0][0]);cs=[0]*(k+2);cs[k]=kap;cs[k+1]=1
        add_alias(b,p,target,cs,virtual,pconst)
        hm=b.alloc_bits(24);b.hmr(combine(head),hm)
        for h,z in zip(head,spare):b.swap(h,z);b.free(z)
        fm=b.alloc_bits(1);b.hmr(flag,fm[0]);b.free(flag)
        return out,A,(hm,fm)
    linear_words(b,out,base.embedding_matrix(k+2,kap),True)
    quo,rem=divmodp([0]*(k+2)+[1],[kap,1,1]);coeffs=[rem[j] if j<len(rem) else 0 for j in range(2)]+[quo[j] if j<len(quo) else 0 for j in range(k)]
    add_alias(b,p,out,coeffs,virtual,pconst)
    value=eval_implicit(b,A,qi,c);cx(b,head[1],head[0],c);hm,fm=transcript
    divide_zero_bridge414(b,combine(head),value,K=K,extra_zero_mask=fm[0]);b.free(fm)
    cx(b,head[1],head[0],c);b.cz(combine(head),hm);b.free(hm)
    eval_implicit(b,A,qi,c,True);truncated_quotient(b,head,A,qi,B,kap)
    return out,A,transcript

def synthesize(b,words,keep_leading=True):
    h=len(words)//2;assert h>=3 and h%2==1
    cs,ks,_,_=constants(h);heads=[words[2*i:2*i+2] for i in range(h)]
    d=b.alloc_qubits(12);F=pg.zero_compact(b,heads[0][1]);masks=[]
    for i in range(h):masks.append(prefix_encode(b,heads,i,F,d))
    divide_safe138(b,d,heads[-1][1]);A=[d];B=[];history=[]
    for i in reversed(range(h)):
        prefix_decode(b,heads,i,F,masks[i]);qi=heads[i-1][1] if i else None;p,virtual,pc=p_alias(heads,i,F)
        oldA,oldB=A,B;A,B,m=absorb(b,heads[i],A,B,qi,p,virtual,pc,cs[i],ks[i]);history.append((i,oldA,oldB,qi,p,virtual,pc,cs[i],ks[i],m))
    extra=[A[h]];freed=[] if keep_leading else extra
    for w in freed:b.free(w)
    fmask=b.alloc_bits(1);b.hmr(F,fmask[0]);b.free(F)
    t=dict(fmask=fmask,kind='partial_q_gauge_noflag',h=h,heads=heads,d=d,F=F,history=history,A=A,B=B,extra=extra,freed=freed,keep_leading=keep_leading)
    return A[:h],B,t

def unsynthesize(b,t):
    if t['freed']:reclaim(b,t['freed'])
    heads=t['heads'];F=t['F'];masks=[None]*t['h']
    for i,A,B,qi,p,virtual,pc,c,kap,m in reversed(t['history']):
        absorb(b,heads[i],A,B,qi,p,virtual,pc,c,kap,True,m);masks[i]=prefix_encode(b,heads,i,F)
        if i==0:
            reclaim(b,[[F]]);lad=base.zero_hold(b,heads[0][1]);b.cx(lad[-1],F);base.zero_free(b,heads[0][1],lad)
            b.cz(F,t['fmask'][0]);b.free(t['fmask'])
    multiply_safe165(b,t['d'],heads[-1][1])
    for i in reversed(range(t['h'])):prefix_decode(b,heads,i,F,masks[i],t['d'])
    pg.zero_compact_free(b,heads[0][1],F);b.free(t['d'])
