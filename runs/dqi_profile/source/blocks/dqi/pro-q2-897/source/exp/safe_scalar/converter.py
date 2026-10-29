# Correct residues on coprime degree profiles with gaps at most two.
# Zero-clipped scalar permutations preserve restoration on unsupported profiles.
"""Guarded history-free multi-h2 Bezout lookahead, with borrowed monic-root word.

On proper coprime half-degree charts with drops1or2, parent gap flags are
recomputed from the deliberatelyzero child quotient metadata. No persistent
quantum history register. Crosspass token is classical and phaseclosed after
reverse around a diagonal oracle. See RESTORATION.md for the all-input
restoration argument; larger gaps may still produce wrong decoded phases.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'convert'))
from convert import ar,frob
from repair import rotate
from kernels import hankel,convolution_xor
from lowwidth import phase_inverse36
from contextlib import contextmanager

def scale(b,x,a,ai,divide=True):
    out=b.alloc_qubits(12)
    ar.mul_xor0(b,x,ai if divide else a,out)
    ar.erase_product(b,a if divide else ai,out,x)
    b.swap(x,out);b.free(out)

def runops(b,ops,a,ai,undo=False):
    for op in (reversed(ops)if undo else ops):
        if op[0]=='s':scale(b,op[1],a,ai,op[2]^undo)
        elif op[0]=='m':ar.mul_xor0(b,op[1],op[2],op[3])
        elif op[0]=='c':convolution_xor(b,op[1],op[2],op[3],undo)
        else:raise ValueError(op[0])

def zero_toggle(b,a,target):
    # Holdthe11ANDladderuntiltheflagcopy, thenuncomputeallforfree.
    for q in a:b.x(q)
    ladder=list(b.alloc_qubits(len(a)-1));b.init_and(a[0],a[1],ladder[0])
    for i in range(1,len(ladder)):b.init_and(ladder[i-1],a[i+1],ladder[i])
    b.cx(ladder[-1],target)
    for i in reversed(range(len(ladder))):
        if i:b.del_and(ladder[i-1],a[i+1],ladder[i])
        else:b.del_and(a[0],a[1],ladder[0])
    b.free(ladder)
    for q in a:b.x(q)

from exp.inverse12_escape.zero1 import inverse_zero,consume,erase,inverse_apply_safe

@contextmanager
def q_inverse(b,a,m,reverse,phase):
    ai,F=inverse_zero(b,a)
    if reverse:consume(b,ai,F,phase['q'].pop(m))
    b.cx(F,a[0]);b.cx(F,ai[0]);yield ai;b.cx(F,ai[0]);b.cx(F,a[0])
    token=erase(b,a,ai,F,not reverse)
    if not reverse:phase['q'][m]=token

def inverse_apply(b,a,action,incoming=None,defer=False):
    return inverse_apply_safe(b,a,action,incoming,defer)

def pre(b,f,g,payloads,undo=False,incoming=None,defer=False):
    m=len(f)-1;n=m-1;a=g[-1];q0=f[-2];q1=f[-1]
    def action(ai):
        ops=[('s',v[-1],True)for v in payloads]
        ops +=[('m',g[i],v[-1],v[i])for v in payloads for i in range(n)]
        ops +=[('s',q1,True),('m',q1,g[-2],q0),('s',q0,True),('c',g[:-1],[q0,q1],f[:-2])]
        runops(b,ops,a,ai,undo)
    return inverse_apply(b,a,action,incoming,defer)

def canonical_permute(b,f,g,F,reverse=False):
    # [G0,F0,G1,F1,...,G_(m-2),F_(m-2),G_(m-1)] leftcyclic.
    cyc=[]
    for i in range(len(g)-1):cyc +=[g[i],f[i]]
    cyc +=[g[-1]]
    seq=range(len(cyc)-2,-1,-1) if reverse else range(len(cyc)-1)
    for i in seq:b.cswap(F,cyc[i],cyc[i+1])

def masked_product(b,F,x,c,target):
    u=b.alloc_qubits(12)
    for j in range(12):b.init_and(F,x[j],u[j])
    ar.mul_xor0(b,u,c,target)
    mask=b.alloc_bits(12);b.hmr(u,mask);b.free(u)
    for j in range(12):b.ccz(F,x[j],mask[j])
    b.free(mask)

def fresh_zero_flag(b,a):
    F=b.alloc_qubits(1)[0];zero_toggle(b,a,F);return F

def clear_flag(b,a,F):zero_toggle(b,a,F);b.free(F)

def q_post(b,q,payloads,H,m,parent,reverse=False,phase=None):
    if parent is None:
        with q_inverse(b,q,m,reverse,phase) as qi:
            for v in payloads:scale(b,v[-1],q,qi,not reverse)
        return
    tops,p,c=parent;F=fresh_zero_flag(b,q)
    b.cswap(F,q,p)
    if not reverse:
        for v,t in zip(payloads,tops):b.cswap(F,v[-1],t)
    with q_inverse(b,q,m,reverse,phase) as qi:
        if not reverse:
            for v in payloads:scale(b,v[-1],q,qi,True)
            for v,t in zip(payloads,tops):masked_product(b,F,v[-1],c,t)
        else:
            for v,t in zip(payloads,tops):masked_product(b,F,v[-1],c,t)
            for v in payloads:scale(b,v[-1],q,qi,False)
    if reverse:
        for v,t in zip(payloads,tops):b.cswap(F,v[-1],t)
    b.cswap(F,q,p);clear_flag(b,q,F)

def post_look(b,f,g,payloads,H,parent,reverse=False,incoming=None,defer=False,phase=None):
    m=len(f)-1;n=m-1;a=g[-1];q0=f[-2];q1=f[-1]
    def dots():
        for v in payloads:
            for i in range(n):ar.mul_xor0(b,g[i],v[i],v[-1])
    def last(undo,inc=None,delay=False):
        def action(ai):
            ops=[('s',v[-1],True)for v in payloads]
            ops +=[('c',g[:-1],[q0,q1],f[:-2]),('s',q0,False),('m',q1,g[-2],q0),('s',q1,False)]
            runops(b,ops,a,ai,undo)
        return inverse_apply(b,a,action,inc,delay)
    if not reverse:q_post(b,q1,payloads,H,m,parent,phase=phase);dots();return last(False,incoming,defer)
    else:
        carry=last(True,incoming,defer);dots();q_post(b,q1,payloads,H,m,parent,True,phase);return carry

def base(b,f,g,payloads,H,parent,reverse=False,phase=None):
    a=g[0];q=f[-1]
    def first(undo,inc=None,delay=False):
        def action(ai):runops(b,[('s',v[0],True)for v in payloads]+[('s',q,True)],a,ai,undo)
        return inverse_apply(b,a,action,inc,delay)
    def last(undo,inc=None,delay=False):
        def action(ai):runops(b,[('s',v[0],True)for v in payloads]+[('s',q,False)],a,ai,undo)
        return inverse_apply(b,a,action,inc,delay)
    if not reverse:
        carry=first(False,delay=True);q_post(b,q,payloads,H,1,parent,phase=phase);phase['a'][1]=last(False,carry,True)
    else:
        carry=last(True,phase['a'].pop(1),True);q_post(b,q,payloads,H,1,parent,True,phase);first(True,carry)

def look_solve(b,f,g,payloads,H=None,parent=None,reverse=False,phase=None):
    m=len(f)-1
    if m==1:return base(b,f,g,payloads,H,parent,reverse,phase)
    F=fresh_zero_flag(b,g[-1]);rotate(b,g,F)
    if not reverse:carry=pre(b,f,g,payloads,defer=True)
    else:carry=post_look(b,f,g,payloads,H,parent,True,incoming=phase['a'].pop(m),defer=True,phase=phase)
    canonical_permute(b,f,g,F);clear_flag(b,g[-1],F)
    childparent=([v[-1]for v in payloads],f[-1],f[-2])
    look_solve(b,g,f[:-2],[v[:-1]for v in payloads],None,childparent,reverse,phase)
    F=fresh_zero_flag(b,g[-1]);canonical_permute(b,f,g,F,True)
    if not reverse:phase['a'][m]=post_look(b,f,g,payloads,H,parent,incoming=carry,defer=True,phase=phase)
    else:pre(b,f,g,payloads,True,incoming=carry)
    rotate(b,g,F,True);clear_flag(b,g[-1],F)

def bez_lookahead(b,O,E,payloads,reverse=False,history_scratch12=None,phase=None):
    def reduce():
        for i in range(len(O)):ar.mul_xor0(b,E[-1],O[i],E[i])
    reduce()
    lead=list(b.alloc_qubits(12))if history_scratch12 is None else list(history_scratch12)
    assert len(lead)==12
    b.x(lead[0])
    look_solve(b,O+[lead],E[:-1],payloads,reverse=reverse,phase=phase)
    b.x(lead[0])
    if history_scratch12 is None:b.free(lead)
    reduce()

def convert_safe(b,A,B,reverse=False,history_scratch12=None,phase_token=None):
    """Optional12zero wires borrowed as monic rootleadingword; returnedzero."""
    phase={'a':{},'q':{}}if phase_token is None else phase_token
    assert not reverse or phase_token is not None
    E=B[::2];O=B[1::2];u=A[1::2];w=A[:34:2];top=A[-1]
    if reverse:hankel(b,B,A,True)
    frob(b,B,2048);frob(b,A,2048)
    if reverse:
        for i in range(17):b.swap(A[2*i],A[2*i+1])
        for i in range(17):ar.mul_xor0(b,O[i],u[i],top)
        bez_lookahead(b,O,E,[u,w],True,history_scratch12,phase)
        for i in range(17):ar.mul_xor0(b,top,O[i],w[i])
    else:
        for i in range(17):ar.mul_xor0(b,top,O[i],w[i])
        bez_lookahead(b,O,E,[u,w],False,history_scratch12,phase)
        for i in range(17):ar.mul_xor0(b,O[i],u[i],top)
        for i in range(17):b.swap(A[2*i],A[2*i+1])
    frob(b,A,2);frob(b,B,2)
    if not reverse:hankel(b,B,A)

    if reverse:assert not phase['a']and not phase['q']
    return phase if not reverse else None
