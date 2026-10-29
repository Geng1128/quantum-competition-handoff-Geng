"""CRT joint remainder/Taylor fold. Live BEST modules are read only.

The factor flag remains live across quotient restoration, so its existing
completion can be shared without another zero test or controlled word bank.
The polynomial and zero-factor completion match BEST for arbitrary heads.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from exp.partial_gauge import synthesis_noflag as old
from exp.mul99.kernels import mul24_xor99
from exp.mul99.schedules import divide_lowwidth369
from exp.fold_joint.divide import divide_zero371
from exp.decoder.common import *
base=old.base;pg=old.pg;ck=old.ck;dv=old.dv

def epow(c,k):
    ans=1;g=c|4096
    for _ in range(k):ans=emul(ans,g)
    return ans
def escale(b,R,c):ar.linear_inplace(b,combine(R),[emul(1<<i,c)for i in range(24)])

def q_times_low(b,B,kap,reverse=False):
    n=len(B)
    if not n:return
    rows=[[kap if i==j else 1 if i-j in(1,2)else 0 for j in range(n)]for i in range(n)]
    linear_words(b,B,rows,reverse)

def low_product(b,head,A,B):
    """l*A mod x^k, k low Acoeffs; rank0/1/k+1 at k0/1/>=2."""
    k=len(B)
    if not k:return
    V,W=base.vandermonde(k+1)
    for t in range(k+1):
        old.rank_affine(b,head,[1,t],A[:k],V[t][:k],B,[W[j][t]for j in range(k)])

def R_to_tops(b,R,B,c,reverse=False):
    """CRT(E-remainder, lowTaylor)->(topcoeffs, unchangedlowTaylor)."""
    k=len(B);powers=base.evaluation_coefficients(k,c)
    if reverse:
        cx(b,R[1],R[0],c);escale(b,R,epow(c,k))
        for word,v in zip(B,powers):cx(b,word,R[0],v&4095);cx(b,word,R[1],v>>12)
    else:
        for word,v in zip(B,powers):cx(b,word,R[0],v&4095);cx(b,word,R[1],v>>12)
        escale(b,R,einv(epow(c,k)));cx(b,R[1],R[0],c)

def divide414_callback(b,x,y,K,extra_zero_mask,after):
    F=dv.zero_flag(b,y);b.cx(F,y[0]);divide_lowwidth369(b,x,y,K);b.cx(F,y[0])
    after(F)
    mask=b.alloc_bits(1);b.hmr(F,mask[0]);b.free(F);b.cx(extra_zero_mask,mask[0])
    dv.phase_zero(b,y,mask[0]);b.free(mask)

def absorb(b,head,A,B,qi,p,virtual,pconst,c,kap,reverse=False,transcript=None,wide=False):
    k=len(B);assert len(A)==k+1;out=B+head
    if not reverse:
        value=old.eval_implicit(b,A,qi,c);flag=dv.zero_flag(b,value)
        spare=[b.alloc_qubits(12),b.alloc_qubits(12)];b.cx(flag,value[0]);cx(b,head[1],head[0],c)
        mul24_xor99(b,combine(head),value,combine(spare),K)
        cx(b,head[1],head[0],c);b.cx(flag,value[0]);old.eval_implicit(b,A,qi,c,True)
        q_times_low(b,B,kap);b.cx(flag,A[0][0]);low_product(b,head,A,B);b.cx(flag,A[0][0])
        v=epow(c,k+2);old.add_alias(b,p,spare,[v&4095,v>>12],virtual,pconst)
        R_to_tops(b,spare,B,c)
        hm=b.alloc_bits(24);b.hmr(combine(head),hm)
        for h,z in zip(head,spare):b.swap(h,z);b.free(z)
        fm=b.alloc_bits(1);b.hmr(flag,fm[0]);b.free(flag)
        return out,A,(hm,fm)
    R_to_tops(b,head,B,c,True)
    v=epow(c,k+2);old.add_alias(b,p,head,[v&4095,v>>12],virtual,pconst)
    value=old.eval_implicit(b,A,qi,c);hm,fm=transcript
    def after(F):
        cx(b,head[1],head[0],c);old.eval_implicit(b,A,qi,c,True)
        b.cx(F,A[0][0]);low_product(b,head,A,B);b.cx(F,A[0][0]);q_times_low(b,B,kap,True)
        old.eval_implicit(b,A,qi,c);cx(b,head[1],head[0],c)
    if wide:divide_zero371(b,combine(head),value,K=K,extra_zero_mask=fm[0],after=after)
    else:divide414_callback(b,combine(head),value,K,fm[0],after)
    b.free(fm)
    cx(b,head[1],head[0],c);b.cz(combine(head),hm);b.free(hm)
    old.eval_implicit(b,A,qi,c,True)
    return out,A,transcript

def synthesize(b,words,keep_leading=True,wide=False):
    h=len(words)//2;assert h>=3 and h%2==1
    cs,ks,_,_=old.constants(h);heads=[words[2*i:2*i+2]for i in range(h)]
    d=b.alloc_qubits(12);F=pg.zero_compact(b,heads[0][1]);masks=[]
    for i in range(h):masks.append(old.prefix_encode(b,heads,i,F,d))
    old.divide_safe138(b,d,heads[-1][1]);A=[d];B=[];hist=[]
    for i in reversed(range(h)):
        old.prefix_decode(b,heads,i,F,masks[i]);qi=heads[i-1][1]if i else None;p,v,pc=old.p_alias(heads,i,F)
        oldA,oldB=A,B;A,B,t=absorb(b,heads[i],A,B,qi,p,v,pc,cs[i],ks[i],wide=wide)
        hist.append((i,oldA,oldB,qi,p,v,pc,cs[i],ks[i],t))
    extra=[A[h]];freed=[]if keep_leading else extra
    for word in freed:b.free(word)
    fmask=b.alloc_bits(1);b.hmr(F,fmask[0]);b.free(F)
    t=dict(h=h,heads=heads,d=d,F=F,history=hist,A=A,B=B,extra=extra,freed=freed,keep_leading=keep_leading,fmask=fmask,wide=wide)
    return A[:h],B,t

def unsynthesize(b,t):
    if t['freed']:reclaim(b,t['freed'])
    heads=t['heads'];F=t['F'];masks=[None]*t['h']
    for i,A,B,qi,p,v,pc,c,kap,token in reversed(t['history']):
        absorb(b,heads[i],A,B,qi,p,v,pc,c,kap,True,token,t['wide']);masks[i]=old.prefix_encode(b,heads,i,F)
        if i==0:
            reclaim(b,[[F]]);lad=base.zero_hold(b,heads[0][1]);b.cx(lad[-1],F);base.zero_free(b,heads[0][1],lad)
            b.cz(F,t['fmask'][0]);b.free(t['fmask'])
    old.multiply_safe165(b,t['d'],heads[-1][1])
    for i in reversed(range(t['h'])):old.prefix_decode(b,heads,i,F,masks[i],t['d'])
    pg.zero_compact_free(b,heads[0][1],F);b.free(t['d'])
