"""Correlated inverse-norm cache lowers the paired I12 reverse to67T/Q33.
The initialized inverse is unchanged64T/Q31; no <=63 isolated claim.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from exp.inverse19_joint.kernel import *
from exp.inverse12_escape.zero1 import erase_raw_cache

def inverse6_keep_inner19(b,x):
    """Preserve x; return inv0(x) and norm(inv0(x))3,19T/Q19."""
    to,frm,K=tower(6);linear_inplace(b,x,to);g=b.alloc_qubits(6)
    for i,(l,r)in enumerate(basic.raw_forms19()):
        with ref.expose_pair(b,x,l,r)as(p,q):b.init_and(p,q,g[i])
    prefix=list(x)+list(g);forms=ref.norm_forms(K);w=ref.inverse3_virtual(b,prefix,forms)
    linear_inplace(b,g,LAY['raw_columns']);R,E=g[:4],g[4:];z=list(E)+list(b.alloc_qubits(4));source=list(x)+list(R)+list(w)
    for j,e in enumerate(LAY['outputs']):ref.form_xor(b,source,e&65535,z[j])
    for i,(l,r)in enumerate(LAY['gates']):
        v=sum(((e>>(16+i))&1)<<j for j,e in enumerate(LAY['outputs']))
        with ref.expose_pair(b,source,l,r)as(p,q):rankone_xor(b,p,q,z,v)
    erase_raw_cache(b,x,R,LAY['raw_rows'][:4]);linear_inplace(b,z,LAY['output_inverse_columns'])
    linear_inplace(b,x,frm);linear_inplace(b,z,frm);return z,w

def inverse12_reverse67(b,x,mask):
    assert len(x)==12;to,frm,K=tower(12);linear_inplace(b,x,to)
    N=b.alloc_qubits(6);norm_xor_noscratch(b,x,N,K);t,v=inverse6_keep_inner19(b,N);out=b.alloc_qubits(6)
    mul_xor_noscratch(b,x[6:],t,out);erase_product(b,out,N,x[6:]);mul_xor_noscratch(b,x[:6],t,x[6:]);b.cx(out,x[6:])
    b.cx(out,x[6:]);erase_product(b,x[6:],N,x[:6]);b.cx(out,x[6:]);b.swap(x[:6],x[6:]);b.swap(x[6:],out);b.free(out)
    D=inverse3_noscratch(b,v);ar._erase_inverse_with_norm(b,t,N,D);b.free(N);erase_inverse(b,v,D);b.free(D)
    to6,frm6,K6=tower(6);linear_inplace(b,t,to6);ar._erase_norm(b,t,v,K6);b.free(v);linear_inplace(b,t,frm6)
    ar._erase_norm(b,x,t,K);b.free(t);linear_inplace(b,x,frm);b.cz(x,mask);b.free(mask)

inverse12_forward64=inverse_inplace_forward20

def phase_inverse6_quantum19(b,u,L):
    """Phase L dot inv0(u), bothquantum6;19T,9private, exactclosedHMR."""
    to,frm,K=tower(6);linear_inplace(b,u,to);transpose=[sum(((frm[i]>>j)&1)<<i for i in range(6))for j in range(6)];linear_inplace(b,L,transpose)
    g=b.alloc_qubits(6)
    for i,(l,r)in enumerate(basic.raw_forms19()):
        with ref.expose_pair(b,u,l,r)as(p,q):b.init_and(p,q,g[i])
    prefix=list(u)+list(g);forms=ref.norm_forms(K);w=ref.inverse3_virtual(b,prefix,forms);source=prefix+list(w)
    for j,e in enumerate(basic.DAG19['outputs']):
        if e&1:b.z(L[j])
        for r in bits((e&65535)>>1):b.cz(L[j],source[r])
    for i,(l,r)in enumerate(basic.DAG19['gates']):
        v=sum(((e>>(16+i))&1)<<j for j,e in enumerate(basic.DAG19['outputs']));ix=list(bits(v));piv=ix[0]
        for j in ix[1:]:b.cx(L[j],L[piv])
        with ref.expose_pair(b,source,l,r)as(p,q):b.ccz(p,q,L[piv])
        for j in reversed(ix[1:]):b.cx(L[j],L[piv])
    m=b.alloc_bits(3);b.hmr(w,m);b.free(w);ref.phase_inverse3_virtual(b,prefix,forms,m);b.free(m)
    for i,(l,r)in reversed(list(enumerate(basic.raw_forms19()))):
        with ref.expose_pair(b,u,l,r)as(p,q):b.del_and(p,q,g[i])
    b.free(g);linear_inplace(b,L,inverse_columns(transpose));linear_inplace(b,u,frm)

def inverse12_keep_inverse_norm64(b,x):
    """Preserve x12, return t=inv(x),u=norm(t)6,64T;Q31."""
    to,frm,K=tower(12);linear_inplace(b,x,to);D=b.alloc_qubits(6);norm_xor_noscratch(b,x,D,K);u=inverse6_init19(b,D);ar._erase_norm(b,x,D,K);b.free(D)
    t=b.alloc_qubits(12);b.cx(x[6:],x[:6]);mul_xor_noscratch(b,x[:6],u,t[:6]);mul_xor_noscratch(b,x[6:],u,t[6:]);b.cx(x[6:],x[:6]);linear_inplace(b,x,frm);linear_inplace(b,t,frm);return t,u

def erase_norm_via_inverse_cache19(b,t,N,u):
    """Erase N=inv0(t),u=norm(t)6;19T vsoldDcachecleanup24T.

    Joint phase mask for N is linear in t and classical HMR outcomes; evaluate
    inverse6 with that quantum mask, then erase u quadratically as norm(t).
    """
    to,frm,K=tower(12);linear_inplace(b,t,to);linear_inplace(b,N,to);m=b.alloc_bits(12);b.hmr(N,m);b.free(N);L=b.alloc_qubits(6);f=field(6)
    def masks():
        for j in range(6):
            for i in range(6):
                for r in bits(f.mul(1<<i,1<<j)):
                    b.ccx(m[r],t[i],L[j]);b.ccx(m[r],t[i+6],L[j]);b.ccx(m[r+6],t[i+6],L[j])
    masks();phase_inverse6_quantum19(b,u,L);masks();b.free(L);b.free(m);ar._erase_norm(b,t,u,K);b.free(u);linear_inplace(b,t,frm)

def inverse24_reverse209(b,x,mask,K=None):
    cols,back,_=extension_columns(K);linear_inplace(b,x,cols);to,frm,K24=tower(24);linear_inplace(b,x,to)
    N=b.alloc_qubits(12);norm_xor_noscratch(b,x,N,K24);t,u=inverse12_keep_inverse_norm64(b,N);out=b.alloc_qubits(12)
    mul_xor_noscratch(b,x[12:],t,out);erase_product(b,out,N,x[12:]);mul_xor_noscratch(b,x[:12],t,x[12:]);b.cx(out,x[12:]);b.cx(out,x[12:]);erase_product(b,x[12:],N,x[:12]);b.cx(out,x[12:]);b.swap(x[:12],x[12:]);b.swap(x[12:],out);b.free(out)
    erase_norm_via_inverse_cache19(b,t,N,u);ar._erase_norm(b,x,t,K24);b.free(t);linear_inplace(b,x,frm);b.cz(x,mask);b.free(mask);linear_inplace(b,x,back)

inverse24_forward190=basic.inverse24_forward19
