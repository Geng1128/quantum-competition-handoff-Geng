"""Nine homogeneous external chains, a trailing jet, and exact phase replay.

The only new dependencies are final_decoder/synthesis.py and conversion.py.
Every other import is an unchanged decoderkit module.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parent)]
import decode as baseline
from exp.decoder.common import *
from exp.confluent_extract.model import synthetic
from exp.inverse12_escape.joint import inverse24_forward190,inverse24_reverse209
from exp.jet24.pro_jet import jet_forward,jet_reverse
from exp.inverse19_joint.kernel import inverse_initialized
from exp.fold_cached_norm.phase34 import phase_inverse34
from exp.convert_short_hankel.phase34 import _private as guarded
from final_decoder import synthesis as folds
from final_decoder.conversion import convert_safe,clear_flag


def geometry():
    groups=[];qs=[];P=[0,0,1]
    for i in range(17):
        c=2*i;kap=mul(c,c)^c^K
        groups.append(dict(zero=False,c=c,kap=kap,gamma=c|4096,
                           count=2,head=2*i,word=4*i,t=1))
        qs.extend([(kap,1)]*2)
        for _ in range(2):P=pmul(P,[kap,1,1])
    groups.append(dict(zero=True,c=0,kap=0,gamma=0,
                       count=1,head=34,word=68,t=0))
    qs.append((0,0));matrix=[]
    for g in groups[:-1]:
        Q=P
        for _ in range(2):
            Q,r=synthetic(Q,g['gamma']);assert r==0
            q=Q+[0]*(70-len(Q))
            matrix.extend([tuple(x&4095 for x in q),tuple(x>>12 for x in q)])
    matrix.extend([tuple(P[1:]+[0]*(71-len(P))),
                   tuple(P[2:]+[0]*(72-len(P)))])
    return groups,qs,P,matrix


groups,qs,P,matrix=geometry()
cs=[g['c']for g in groups[:-1]for _ in range(2)]
ks=[kap for kap,_ in qs[:-1]]
Pscan,rem=divmodp(P,[ks[0],1,1]);assert rem==[0]
corrections=[];c=0
for kap in ks:corrections.append(c);c^=inv(kap)
assert c==0


def square12(b,x,reverse=False):
    ar.linear_inplace(b,x,[power(1<<j,2048 if reverse else 2)for j in range(12)])


def scale_word(b,x,f,fi):
    out=b.alloc_qubits(12);ar.mul_xor0(b,x,f,out)
    ar.erase_product(b,out,fi,x);b.swap(x,out);b.free(out)


def masked_update(b,x,f,F):
    """Return the phase mask of F*x_old, after x += F*x_old*(f+1)."""
    u=b.alloc_qubits(12)
    for a,z in zip(x,u):b.init_and(F,a,z)
    b.x(f[0]);ar.mul_xor0(b,u,f,x);b.x(f[0])
    m=b.alloc_bits(12);b.hmr(u,m);b.free(u);return m


def phase_masked_product(b,x,f,F,m):
    t=b.alloc_qubits(1)[0];field=ar.field(12)
    for i,a in enumerate(x):
        b.init_and(F,a,t)
        for j,c in enumerate(f):
            for k in ar.bits(field.mul(1<<i,1<<j)):b.ccz(t,c,m[k])
        b.del_and(F,a,t)
    b.free(t);b.free(m)


def reverse_masked_multiply(b,x,f,fi,F,m):
    fresh=masked_update(b,x,fi,F)
    for a,bit in zip(x,m):b.ccz(F,a,bit)
    b.free(m);phase_masked_product(b,x,f,F,fresh)


def deferred_multiply(b,x,f):
    out=b.alloc_qubits(12);ar.mul_xor0(b,x,f,out)
    m=b.alloc_bits(12);b.hmr(x,m);b.swap(x,out);b.free(out);return m


def reverse_deferred_multiply(b,x,f,fi,m):
    fresh=deferred_multiply(b,x,fi)
    b.cz(x,m);b.free(m);ar.phase_product(b,x,f,fresh);b.free(fresh)


def controlled_inverse(b,x,F,reverse=False,token=None):
    xi=inverse_initialized(b,x)
    if reverse:b.cz(xi,token);b.free(token)
    b.cswap(F,x,xi)
    m=b.alloc_bits(12);b.hmr(xi,m);b.free(xi)
    if reverse:phase_inverse34(b,x,m);b.free(m)
    else:return m


def term_correction_with_inverse(b,p,q,d,u,v,pole,qi,reverse=False,token=None,Z=None):
    if reverse:
        Z=guarded.fresh_zero_flag(b,d);b.cz(Z,token[0]);b.free(token)
    F=b.alloc_qubits(1)[0];b.cx(Z,F);b.ccx(Z,pole,F)
    def add_product(x,target):
        w=b.alloc_qubits(12);ar.mul_xor0(b,x,qi,w)
        b.ccx(F,w,target);ar.erase_product(b,x,qi,w);b.free(w)
    if reverse:add_product(v,u);add_product(p,v)
    else:add_product(p,v);add_product(v,u)
    b.ccx(Z,pole,F);b.cx(Z,F);b.free(F)
    if reverse:return Z
    zm=b.alloc_bits(1);b.hmr(Z,zm[0]);b.free(Z);return zm


def tail_finish(b,p,q,d,u,v,pole,reverse=False,token=None,Z=None):
    """Share q's inverse0 across correction and clipped pole products."""
    if reverse:b.cswap(pole,v,d);b.cswap(pole,u,v)
    qi,Zq=guarded.inverse_zero(b,q)
    if reverse:guarded.consume(b,qi,Zq,token['inverse'])
    if not reverse:tt=term_correction_with_inverse(b,p,q,d,u,v,pole,qi,Z=Z)
    b.cx(Zq,q[0]);b.cx(Zq,qi[0])
    if reverse:
        reverse_masked_multiply(b,d,qi,q,pole,token['d'])
        reverse_masked_multiply(b,v,q,qi,pole,token['v'])
    else:
        vm=masked_update(b,v,q,pole);dm=masked_update(b,d,qi,pole)
    b.cx(Zq,qi[0]);b.cx(Zq,q[0])
    if reverse:Z=term_correction_with_inverse(b,p,q,d,u,v,pole,qi,True,token['term'])
    qi_token=guarded.erase(b,q,qi,Zq,not reverse)
    if reverse:return Z
    if not reverse:
        b.cswap(pole,u,v);b.cswap(pole,v,d)
        return dict(inverse=qi_token,term=tt,v=vm,d=dm)


def tail_inplace(b,p,q,d,u,v,pole,reverse=False,token=None):
    """(d,u,v) <-> (delta,N0,N1), with p,q the implicit leaders."""
    if not reverse:
        ar.mul_xor0(b,q,v,p)
        square12(b,q);ar.mul_xor0(b,q,u,d);square12(b,q,True)
        masks=[]
        def divide(pi):
            b.x(pole);masks.append(masked_update(b,d,pi,pole));b.x(pole)
        tp=guarded.inverse_apply(b,p,divide,defer=True);tp=(tp,masks[0]);ar.mul_xor0(b,q,v,p)
        def multiply(di,m):
            b.x(pole);vm=masked_update(b,v,d,pole);b.x(pole)
            ar.mul_xor0(b,q,u,v);um=deferred_multiply(b,u,d)
            return vm,um
        Z=guarded.fresh_zero_flag(b,d);b.cx(Z,d[0]);td=multiply(None,None);b.cx(Z,d[0])
        tf=tail_finish(b,p,q,d,u,v,pole,Z=Z)
        return dict(p=tp,d=td,finish=tf)
    Z=tail_finish(b,p,q,d,u,v,pole,True,token['finish'])
    def divide(di,m):
        vm,um=m
        reverse_deferred_multiply(b,u,d,di,um);ar.mul_xor0(b,q,u,v)
        b.x(pole);reverse_masked_multiply(b,v,d,di,pole,vm);b.x(pole)
    di=inverse_initialized(b,d);b.cx(Z,d[0]);b.cx(Z,di[0])
    divide(di,token['d']);b.cx(Z,di[0]);b.cx(Z,d[0])
    guarded.erase(b,d,di,Z,False);ar.mul_xor0(b,q,v,p)
    def multiply(pi):
        b.x(pole);reverse_masked_multiply(b,d,pi,p,pole,token['p'][1]);b.x(pole)
    guarded.inverse_apply(b,p,multiply,incoming=token['p'][0]);ar.mul_xor0(b,q,v,p)
    square12(b,q);ar.mul_xor0(b,q,u,d);square12(b,q,True)


def synthesize(b,words,pole):
    heads=[words[2*i:2*i+2]for i in range(34)];u,v=words[-2:]
    d,border=heads[0];r=b.alloc_qubits(12)
    b.cx(border,r);b.cx(border,d);b.cx(r,border);b.x(border[0])
    for i in range(1,34):
        qi,target=(r,border)if i%2 else(border,r)
        ar.mul_xor0(b,heads[i][1],qi,target)
        folds.add_tail(b,heads[i],qi,d,bool(i%2))
    ar.mul_xor0(b,r,border,d);tt=tail_inplace(b,r,border,d,u,v,pole)
    # The exact chart remains live through the diagonal middle stages.
    A,B,hist=[u,v],[d],[]
    for i in reversed(range(1,34)):
        aa,bb=A,B
        if i%2:
            ar.mul_xor0(b,heads[i][1],r,border);qi,p=r,border
            A,B,t=folds.sg['absorb'](b,heads[i],A,B,qi,p,None,0,cs[i],ks[i],wide=False)
        else:
            qi,p=border,r
            A,B,t=folds.sg['absorb_even'](b,heads[i],A,B,qi,r,cs[i],ks[i])
        hist.append((i,aa,bb,qi,p,t))
    b.x(border[0]);b.free(border);C=B+[r]
    def undo():
        folds.ug['reclaim'](b,[border]);b.x(border[0])
        for i,aa,bb,qi,p,t in reversed(hist):
            if i%2:
                folds.ug['absorb'](b,heads[i],aa,bb,qi,p,None,0,cs[i],ks[i],True,t,False)
                ar.mul_xor0(b,heads[i][1],r,border)
            else:folds.ug['absorb_even'](b,heads[i],aa,bb,qi,r,cs[i],ks[i],True,t)
        tail_inplace(b,r,border,d,u,v,pole,True,tt);ar.mul_xor0(b,r,border,d)
        for i in reversed(range(1,34)):
            qi,target=(r,border)if i%2 else(border,r)
            folds.add_tail(b,heads[i],qi,d,bool(i%2));ar.mul_xor0(b,heads[i][1],qi,target)
        b.cx(r,d);b.x(border[0]);b.cx(r,border);b.cx(border,r);b.free(r)
    return C,A,undo


def step(b,head,N,D,L,kap,c,reverse=False):
    a,h=head
    if reverse:scale(b,N,inv(kap));b.swap(N,D);ar.mul_xor0(b,a,D,N);scale(b,L,inv(kap))
    cx(b,a,h,c);square12(b,D);ar.mul_xor0(b,h,D,L);square12(b,D,True);cx(b,a,h,c)
    if not reverse:scale(b,L,kap);ar.mul_xor0(b,a,D,N);b.swap(N,D);scale(b,N,kap)


def first_step(b,head,N,D,L,kap):
    b.cx(head[0],N);b.swap(N,D);scale(b,N,kap)
    b.cx(head[1],L);scale(b,L,kap)


def scalar_step(b,head,N,D,kap,reverse=False,first=False):
    if reverse:scale(b,N,inv(kap));b.swap(N,D)
    if first:b.cx(head[0],N)
    else:ar.mul_xor0(b,head[0],D,N)
    if not reverse:b.swap(N,D);scale(b,N,kap)


def normal_form(b,N,D,L):
    """Normalize by the guaranteed nonzero D; defer only its inverse mask."""
    di=inverse_initialized(b,D);scale_word(b,N,di,D)
    square12(b,D);square12(b,di);scale_word(b,L,di,D)
    square12(b,di,True);square12(b,D,True)
    m=b.alloc_bits(12);b.hmr(di,m);b.free(di);return m


def extraction(b,words,reverse=False,token=None):
    heads=[words[2*i:2*i+2]for i in range(34)];N,L=words[-2:]
    if not reverse:
        ext=external_extraction(b,words);D=b.alloc_qubits(12);b.x(D[0])
        first_step(b,heads[0],N,D,L,ks[0])
        for head,kap,c in zip(heads[1:],ks[1:],corrections[1:]):step(b,head,N,D,L,kap,c)
        pole=b.alloc_qubits(1)[0];guarded.zero_toggle(b,D,pole);b.cswap(pole,N,D)
        norm=normal_form(b,N,D,L);ci=controlled_inverse(b,L,pole)
        gauge=b.alloc_bits(12);b.hmr(D,gauge);b.free(D)
        return dict(external=ext,pole=pole,normalization=norm,gauge_mask=gauge,chart_inverse=ci)
    pole=token['pole'];controlled_inverse(b,L,pole,True,token['chart_inverse'])
    D=b.alloc_qubits(12);b.x(D[0]);b.cswap(pole,N,D)
    for head,kap,c in reversed(list(zip(heads,ks,corrections))[1:]):step(b,head,N,D,L,kap,c,True)
    scale(b,N,inv(ks[0]));b.swap(N,D);scale(b,L,inv(ks[0]))
    extra_z_mask=normal_form(b,N,D,L)
    b.cx(heads[0][0],N);b.cx(heads[0][1],L)
    b.cz(D,token['normalization']);b.free(token['normalization'])
    inverse_mask=b.alloc_bits(12);b.hmr(D,inverse_mask);b.free(D)
    z_mask=token['gauge_mask'];b.cx(extra_z_mask,z_mask);b.free(extra_z_mask)
    D=b.alloc_qubits(12);b.x(D[0]);scalar_step(b,heads[0],N,D,ks[0],first=True)
    for head,kap in zip(heads[1:],ks[1:]):scalar_step(b,head,N,D,kap)
    b.cswap(pole,N,D);b.cz(D,z_mask);b.free(z_mask)
    phase_inverse34(b,D,inverse_mask);b.free(inverse_mask);b.cswap(pole,N,D)
    guarded.zero_toggle(b,D,pole)
    for head,kap in reversed(list(zip(heads[1:],ks[1:]))):scalar_step(b,head,N,D,kap,True)
    scalar_step(b,heads[0],N,D,ks[0],True,first=True)
    b.x(D[0]);b.free(D);b.free(pole);external_extraction(b,words,True,token['external'])


def decode(b,S,E):
    with b.mark('mask_projection'):mt=baseline.project(b,S,E)
    with b.mark('initial'):linear_words(b,S,matrix)
    with b.mark('extract'):et=extraction(b,S)
    with b.mark('synthesis'):A,B,undo=synthesize(b,S,et['pole'])
    with b.mark('convert'):ct=convert_safe(b,A,B,monic_root=True)
    with b.mark('scan'):baseline.scan(b,A,B,E,skip_points=baseline.SELECTED,node_polynomial=Pscan)
    with b.mark('unconvert'):convert_safe(b,A,B,True,phase_token=ct)
    with b.mark('unsynthesis'):undo()
    with b.mark('unextract'):extraction(b,S,True,et)
    with b.mark('uninitial'):linear_words(b,S,matrix,True)
    with b.mark('restore_masks'):baseline.restore(b,E,mt)


from functools import lru_cache
from exp.mul99.kernels import mul24_xor99
from exp.mul99.schedules import multiply_forward99
from exp.arithmetic.division import phase_product24,zero_flag
from exp.noscratch.schedules import norm_xor_noscratch,inverse_stream_noscratch
from exp.noscratch.kernels import mul_xor_noscratch
from exp.mul99.phase36 import phase_inverse36_pivot

EXTERNAL_HOMOGENEOUS=frozenset(range(8,17))

@lru_cache(None)
def external_square_columns(reverse=False):
    def epow(x,n):
        out=1
        while n:
            if n&1:out=emul(out,x)
            x=emul(x,x);n>>=1
        return out
    return tuple(epow(1<<i,1<<23 if reverse else 2)for i in range(24))


def external_square24(b,x,reverse=False):ar.linear_inplace(b,x,external_square_columns(reverse))


def external_mul_head(b,head,D,N,ct):
    a,h=head;cx(b,h,a,ct)
    mul24_xor99(b,combine(head),D,N,K)
    cx(b,h,a,ct)


def external_corrected_product(b,head,D,L,ct,c):
    a,h=head
    if not c:
        ar.mul_xor0(b,h,D[:12],L[:12]);ar.mul_xor0(b,h,D[12:],L[12:])
    else:
        scale(b,a,c);cx(b,h,a,mul(c,ct)^1);scale(b,h,c)
        mul24_xor99(b,combine(head),D,L,K)
        scale(b,h,inv(c));cx(b,h,a,mul(c,ct)^1);scale(b,a,inv(c))


def external_chain_step(b,head,N,D,L,q,ct,c,reverse=False,first=False):
    if reverse:
        scalepair(b,N,inv(q));b.swap(N,D);external_mul_head(b,head,D,N,ct);scalepair(b,L,inv(q))
    if first and not reverse:
        b.cx(head[1],L[:12]);scalepair(b,L,q)
        a,h=head;cx(b,h,a,ct);b.cx(combine(head),N);cx(b,h,a,ct)
        b.swap(N,D);scalepair(b,N,q)
        return
    external_square24(b,D);external_corrected_product(b,head,D,L,ct,c);external_square24(b,D,True)
    if not reverse:
        scalepair(b,L,q);external_mul_head(b,head,D,N,ct);b.swap(N,D);scalepair(b,N,q)


def external_scalar_step(b,head,N,D,q,ct,reverse=False,first=False):
    if reverse:scalepair(b,N,inv(q));b.swap(N,D)
    if first:
        a,h=head;cx(b,h,a,ct);b.cx(combine(head),N);cx(b,h,a,ct)
    else:external_mul_head(b,head,D,N,ct)
    if not reverse:b.swap(N,D);scalepair(b,N,q)


def external_ordinary_node(b,words,gi,reverse=False,token=None):
    g=groups[gi];u=combine(words[g['word']:g['word']+2]);v=combine(words[g['word']+2:g['word']+4])
    history=[]if not reverse else list(token)
    for h in reversed(range(2*gi))if reverse else range(2*gi):
        a,d=words[2*h:2*h+2];q=ks[h]^g['kap']
        def translate():
            b.cx(a,u[:12]);cx(b,d,u[:12],g['c']);b.cx(d,u[12:]);b.cx(d,v[:12])
        if not reverse:
            translate();history.append(jet_forward(b,u,v));scalepair(b,v,q);b.cx(u,v);scalepair(b,u,q)
        else:
            scalepair(b,u,inv(q));b.cx(u,v);scalepair(b,v,inv(q));jet_reverse(b,u,v,history.pop());translate()
    if reverse:assert not history
    return history


def external_normalize_paired(b,N,D,L,reverse=False,token=None):
    b.cx(D[12:],D[:12]);nm=multiply_forward99(b,N,D,K)
    external_square24(b,D);lm=multiply_forward99(b,L,D,K);external_square24(b,D,True)
    b.cx(D[12:],D[:12])
    u=b.alloc_qubits(12);norm_xor_noscratch(b,D,u,K)
    if reverse:b.cz(u,token);b.free(token)
    m=inverse_stream_noscratch(b,u);ar._phase_norm(b,D,m,K);b.free(m)
    masks=[]
    for word,squared in [(N,False),(L,True)]:
        if squared:square12(b,u)
        row=[]
        for off in (0,12):
            half=word[off:off+12];out=b.alloc_qubits(12)
            mul_xor_noscratch(b,half,u,out);hm=b.alloc_bits(12)
            b.hmr(half,hm);b.swap(half,out);b.free(out);row.append(hm)
        masks.append(row)
        if squared:square12(b,u,True)
    if reverse:um=u
    else:
        um=b.alloc_bits(12);b.hmr(u,um);b.free(u)
    norm=b.alloc_qubits(12);norm_xor_noscratch(b,D,norm,K)
    for word,row,squared in [(N,masks[0],False),(L,masks[1],True)]:
        if squared:square12(b,norm)
        for off,hm in zip((0,12),row):
            ar.phase_product(b,word[off:off+12],norm,hm);b.free(hm)
        if squared:square12(b,norm,True)
    if not reverse:ar._erase_norm(b,D,norm,K);b.free(norm)
    phase_product24(b,N,D,nm,K);b.free(nm)
    external_square24(b,D);phase_product24(b,L,D,lm,K);b.free(lm);external_square24(b,D,True)
    return (norm,um) if reverse else um


def external_inverse_phase_norm_pair(b,x,mask,reciprocal,norm):
    b.cx(x[12:],x[:12])
    ar.phase_product(b,x[:12],reciprocal,mask[:12]);ar.phase_product(b,x[12:],reciprocal,mask[12:])
    b.cx(x[12:],x[:12])
    im=b.alloc_bits(12);b.hmr(reciprocal,im);b.free(reciprocal)
    phase_inverse36_pivot(b,norm,im);b.free(im)
    nm=b.alloc_bits(12);b.hmr(norm,nm);b.free(norm)
    ar._phase_norm(b,x,nm,K);b.free(nm)


def external_node_chain(b,words,gi,reverse=False,token=None,normalizer=None):
    g=groups[gi];N=combine(words[g['word']:g['word']+2]);L=combine(words[g['word']+2:g['word']+4])
    heads=[words[2*i:2*i+2]for i in range(2*gi)]
    params=[];c=0
    for i in range(2*gi):
        q=ks[i]^g['kap'];assert q
        params.append((q,c));c^=inv(q)
    assert c==0
    if not reverse:
        D=b.alloc_qubits(24);b.x(D[0])
        for i,(head,(q,c))in enumerate(zip(heads,params)):
            external_chain_step(b,head,N,D,L,q,g['c'],c,first=i==0)
        pole=zero_flag(b,D);b.cswap(pole,N,D)
        norm=external_normalize_paired(b,N,D,L)
        gauge=b.alloc_bits(24);b.hmr(D,gauge);b.free(D)
        return dict(pole=pole,gauge=gauge,norm=norm)
    pole=token['pole'];D=b.alloc_qubits(24);b.x(D[0]);b.cswap(pole,N,D)
    for head,(q,c)in reversed(list(zip(heads,params))[1:]):external_chain_step(b,head,N,D,L,q,g['c'],c,True)
    q0=params[0][0];scalepair(b,N,inv(q0));b.swap(N,D);scalepair(b,L,inv(q0))
    norm=external_normalize_paired(b,N,D,L,True,token['norm'])
    a,h=heads[0];cx(b,h,a,g['c']);b.cx(combine(heads[0]),N);cx(b,h,a,g['c']);b.cx(h,L[:12])
    invmask=b.alloc_bits(24);b.hmr(D,invmask);b.free(D)
    D=b.alloc_qubits(24);b.x(D[0])
    for i,(head,(q,c))in enumerate(zip(heads,params)):external_scalar_step(b,head,N,D,params[i][0],g['c'],first=i==0)
    b.cswap(pole,N,D);b.cz(D,token['gauge']);b.free(token['gauge'])
    reciprocal,direct=norm
    external_inverse_phase_norm_pair(b,D,invmask,reciprocal,direct);b.free(invmask);b.cswap(pole,N,D)
    clear_flag(b,D,pole)
    for i in reversed(range(len(heads))):external_scalar_step(b,heads[i],N,D,params[i][0],g['c'],True,first=i==0)
    b.x(D[0]);b.free(D)


def external_extraction(b,words,reverse=False,token=None):
    history=[]if not reverse else list(token)
    for gi in reversed(range(17))if reverse else range(17):
        g=groups[gi];u=combine(words[g['word']:g['word']+2]);v=combine(words[g['word']+2:g['word']+4])
        a,d=words[g['word']:g['word']+2]
        if not reverse:
            nt=external_node_chain(b,words,gi,normalizer=None)if gi in EXTERNAL_HOMOGENEOUS else external_ordinary_node(b,words,gi)
            cx(b,d,a,g['c']);b.cx(d,v[:12]);local=inverse24_forward190(b,v,K)
            cx(b,words[g['word']+3],words[g['word']+2],g['c'])
            history.append((nt,local))
        else:
            nt,local=history.pop();cx(b,words[g['word']+3],words[g['word']+2],g['c'])
            inverse24_reverse209(b,v,local,K);b.cx(d,v[:12]);cx(b,d,a,g['c'])
            if gi in EXTERNAL_HOMOGENEOUS:external_node_chain(b,words,gi,True,nt,None)
            else:external_ordinary_node(b,words,gi,True,nt)
    assert not reverse or not history
    return history
