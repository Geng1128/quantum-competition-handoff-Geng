"""Tail d aliases raw a0; quadratic companion action closes outer fold.

No additional persistent field word. Inputs and returned A/B use70 words.
All maps are total: factors zero are replaced by factor1 consistently.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from exp.fold_joint.kernel import *
from exp.tail_alias.divide import divide_zero369
from types import FunctionType
# Private binding: imported CRT arithmetic modules are never mutated.
_absorb=absorb
absorb=FunctionType(_absorb.__code__,dict(_absorb.__globals__,divide_zero371=divide_zero369),
    'absorb',_absorb.__defaults__,_absorb.__closure__)

def expose_quotient(b,C,D):
    """Top C pair -> quotient l of q*C by monic D, h>=3; involution."""
    b.cx(C[-1],C[-2]);ar.mul_xor0(b,C[-1],D[-1],C[-2])

def outer(b,C,D,c,kap,reverse=False,transcript=None,wide=True):
    """C <- q*C mod (D+[D(gamma)=0]), D monic degree h unchanged.

    D lists its h low coefficients, C its h coefficients. Constant completion
    leaves quotient l unaffected for h>=3. CRT output is lowerC then top pair.
    """
    h=len(C);assert len(D)==h and h>=3;head=C[-2:];low=C[:-2]
    if not reverse:
        expose_quotient(b,C,D)
        value=old.eval_implicit(b,D,None,c);F=dv.zero_flag(b,value)
        spare=[b.alloc_qubits(12),b.alloc_qubits(12)]
        b.cx(F,value[0]);cx(b,head[1],head[0],c)
        mul24_xor99(b,combine(head),value,combine(spare),K)
        cx(b,head[1],head[0],c);b.cx(F,value[0]);old.eval_implicit(b,D,None,c,True)
        q_times_low(b,low,kap);b.cx(F,D[0][0]);low_product(b,head,D,low);b.cx(F,D[0][0])
        R_to_tops(b,spare,low,c)
        hm=b.alloc_bits(24);b.hmr(combine(head),hm)
        for w,z in zip(head,spare):b.swap(w,z);b.free(z)
        fm=b.alloc_bits(1);b.hmr(F,fm[0]);b.free(F)
        return hm,fm
    R_to_tops(b,head,low,c,True)
    value=old.eval_implicit(b,D,None,c);hm,fm=transcript
    def after(F):
        cx(b,head[1],head[0],c);old.eval_implicit(b,D,None,c,True)
        b.cx(F,D[0][0]);low_product(b,head,D,low);b.cx(F,D[0][0]);q_times_low(b,low,kap,True)
        old.eval_implicit(b,D,None,c);cx(b,head[1],head[0],c)
    if wide:divide_zero369(b,combine(head),value,K=K,extra_zero_mask=fm[0],after=after)
    else:divide414_callback(b,combine(head),value,K,fm[0],after)
    b.free(fm);cx(b,head[1],head[0],c);b.cz(combine(head),hm);b.free(hm)
    old.eval_implicit(b,D,None,c,True);expose_quotient(b,C,D)
    return transcript

def synthesize(b,words,wide=True):
    h=len(words)//2;assert h>=3 and h%2==1
    cs,ks,_,_=old.constants(h);heads=[words[2*i:2*i+2]for i in range(h)]
    d=heads[0][0];F=pg.zero_compact(b,heads[0][1]);b.cx(heads[0][1],d);masks=[None]*h
    for i in range(1,h):masks[i]=old.prefix_encode(b,heads,i,F,d)
    old.divide_safe138(b,d,heads[-1][1]);A=[d];B=[];hist=[]
    for i in reversed(range(1,h)):
        old.prefix_decode(b,heads,i,F,masks[i]);qi=heads[i-1][1];p,v,pc=old.p_alias(heads,i,F)
        oldA,oldB=A,B;A,B,t=absorb(b,heads[i],A,B,qi,p,v,pc,cs[i],ks[i],wide=wide)
        hist.append((i,oldA,oldB,qi,p,v,pc,cs[i],ks[i],t))
    C=B+[heads[0][1]];ot=outer(b,C,A,cs[0],ks[0],wide=wide)
    fm=b.alloc_bits(1);b.hmr(F,fm[0]);b.free(F)
    return C,A,dict(h=h,heads=heads,d=d,F=F,history=hist,A=C,B=A,outer=ot,wide=wide,fmask=fm)

def unsynthesize(b,t):
    h=t['h'];heads=t['heads'];F=t['F'];cs,ks,_,_=old.constants(h)
    outer(b,t['A'],t['B'],cs[0],ks[0],True,t['outer'],t['wide'])
    reclaim(b,[[F]]);lad=base.zero_hold(b,heads[0][1]);b.cx(lad[-1],F);base.zero_free(b,heads[0][1],lad)
    b.cz(F,t['fmask'][0]);b.free(t['fmask']);masks=[None]*h
    for i,A,B,qi,p,v,pc,c,kap,token in reversed(t['history']):
        absorb(b,heads[i],A,B,qi,p,v,pc,c,kap,True,token,t['wide'])
        masks[i]=old.prefix_encode(b,heads,i,F)
    old.multiply_safe165(b,t['d'],heads[-1][1])
    for i in reversed(range(1,h)):old.prefix_decode(b,heads,i,F,masks[i],t['d'])
    b.cx(heads[0][1],t['d']);pg.zero_compact_free(b,heads[0][1],F)
