"""Completed dual-number inversion reference; allinput, explicitextension basis.
J(u,v)=(inv0u,v*(inv0u+[u0])²). Forward290T; closedpair686T/49private.
No persistent quantum cache; return only two classical masks.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from exp.inverse12_escape import joint as base
from exp.inverse12_escape import zero1
from exp.inverse12_escape.joint import *
from exp.mul99.kernels import mul_xor99_zero,phase_product_quantum99
from exp.arithmetic.lowwidth import classical_transpose,phase_product_quantum


def phase_zero_virtual(b,prefix,forms,w,mask):
    """zero(prefix norm) phase,0T while its norm3 andinverse3 forms live."""
    b.z(mask);source=list(prefix)+list(w);f=field(3)
    for i in range(3):
        for j in range(3):
            if f.mul(1<<i,1<<j)&1:
                with ref.expose_pair(b,source,forms[i],1<<(1+len(prefix)+j))as(p,q):b.ccz(p,q,mask)


def phase_outputs_classical(b,source,outputs,gates,m):
    for j,e in enumerate(outputs):
        if e&1:b.z(m[j])
        for r in bits((e&65535)>>1):b.cz(source[r],m[j])
    cb=b.alloc_bits(1)[0];b.bit_store(cb,False)
    for i,(l,r)in enumerate(gates):
        js=[j for j,e in enumerate(outputs)if e>>(16+i)&1]
        for j in js:b.cx(m[j],cb)
        with ref.expose_pair(b,source,l,r)as(p,q):b.ccz(p,q,cb)
        for j in reversed(js):b.cx(m[j],cb)
    b.free(cb)


def phase_inverse12_quantum64(b,x,L,zero_mask=None):
    """Phase L.inv0(x),quantumL12,64T withpaid-innerprefixretained."""
    to12,frm12,K12=tower(12);linear_inplace(b,x,to12);T=[sum(((frm12[i]>>j)&1)<<i for i in range(12))for j in range(12)];linear_inplace(b,L,T)
    D=b.alloc_qubits(6);norm_xor_noscratch(b,x,D,K12);to6,frm6,K6=tower(6);linear_inplace(b,D,to6);g=b.alloc_qubits(6)
    for i,(l,r)in enumerate(basic.raw_forms19()):
        with ref.expose_pair(b,D,l,r)as(p,q):b.init_and(p,q,g[i])
    prefix=list(D)+list(g);forms=ref.norm_forms(K6);w=ref.inverse3_virtual(b,prefix,forms);source=prefix+list(w);out=b.alloc_qubits(6)
    for j,e in enumerate(basic.DAG19['outputs']):ref.form_xor(b,source,e&65535,out[j])
    for i,(l,r)in enumerate(basic.DAG19['gates']):
        v=sum(((e>>(16+i))&1)<<j for j,e in enumerate(basic.DAG19['outputs']))
        with ref.expose_pair(b,source,l,r)as(p,q):rankone_xor(b,p,q,out,v)
    linear_inplace(b,out,frm6);b.cx(x[6:],x[:6]);phase_product_quantum(b,x[:6],out,L[:6]);phase_product_quantum(b,x[6:],out,L[6:]);b.cx(x[6:],x[:6])
    if zero_mask is not None:phase_zero_virtual(b,prefix,forms,w,zero_mask)
    m=b.alloc_bits(6);b.hmr(out,m);b.free(out);mt=classical_transpose(b,m,frm6);b.free(m);phase_outputs_classical(b,source,basic.DAG19['outputs'],basic.DAG19['gates'],mt);b.free(mt)
    wm=b.alloc_bits(3);b.hmr(w,wm);b.free(w);ref.phase_inverse3_virtual(b,prefix,forms,wm);b.free(wm)
    for i,(l,r)in reversed(list(enumerate(basic.raw_forms19()))):
        with ref.expose_pair(b,D,l,r)as(p,q):b.del_and(p,q,g[i])
    b.free(g);linear_inplace(b,D,frm6);ar._erase_norm(b,x,D,K12);b.free(D);linear_inplace(b,L,inverse_columns(T));linear_inplace(b,x,frm12)


def phase_inverse24_classical106(b,x,mask,zero_mask=None):
    """Phase mask.inv0(x)+zero_mask*[x0],106T/45private."""
    to,frm,K=tower(24);linear_inplace(b,x,to);mt=classical_transpose(b,mask,frm);N=b.alloc_qubits(12);norm_xor_noscratch(b,x,N,K);L=b.alloc_qubits(12);f=field(12)
    def coeff():
        for j in range(12):
            for i in range(12):
                for r in bits(f.mul(1<<i,1<<j)):
                    b.ccx(mt[r],x[i],L[j]);b.ccx(mt[r],x[i+12],L[j]);b.ccx(mt[r+12],x[i+12],L[j])
    coeff();phase_inverse12_quantum64(b,N,L,zero_mask);coeff();b.free(L);ar._erase_norm(b,x,N,K);b.free(N);b.free(mt);linear_inplace(b,x,frm)


def inverse24_initialized_zero191(b,x):
    to,frm,K=tower(24);linear_inplace(b,x,to);N=b.alloc_qubits(12);norm_xor_noscratch(b,x,N,K);t,F=zero1.inverse_zero(b,N);ar._erase_norm(b,x,N,K);b.free(N)
    z=b.alloc_qubits(24);b.cx(x[12:],x[:12]);mul_xor_noscratch(b,x[:12],t,z[:12]);mul_xor_noscratch(b,x[12:],t,z[12:]);b.cx(x[12:],x[:12]);ar._erase_norm(b,z,t,K);b.free(t);linear_inplace(b,x,frm);linear_inplace(b,z,frm);return z,F


def safe_square(b,x,F,undo=False):
    sq=square_columns(24)
    if undo:linear_inplace(b,x,inverse_columns(sq));b.cx(F,x[0])
    else:b.cx(F,x[0]);linear_inplace(b,x,sq)


def forward_canonical(b,u,v):
    z,F=inverse24_initialized_zero191(b,u);out=b.alloc_qubits(24);safe_square(b,z,F);mul_xor99_zero(b,v,z,out);safe_square(b,z,F,True)
    m=b.alloc_bits(24);b.hmr(v,m);safe_square(b,u,F);phase_product_quantum99(b,out,u,m);safe_square(b,u,F,True);b.free(m);b.swap(v,out);b.free(out)
    mu=b.alloc_bits(24);fz=b.alloc_bits(1);b.hmr(u,mu);b.swap(u,z);b.free(z);b.hmr(F,fz[0]);b.free(F);return mu,fz


def jet_forward(b,u,v,K=None):
    """290T/49private;returns classical(mu24,zero1),noquantumcache."""
    cols,frm,_=extension_columns(K);linear_inplace(b,u,cols);linear_inplace(b,v,cols);token=forward_canonical(b,u,v);linear_inplace(b,u,frm);linear_inplace(b,v,frm);return token


def jet_reverse(b,u,v,token,K=None):
    """396T/49private closedreverse ofjet_forward;106T inversephase."""
    cols,frm,_=extension_columns(K);linear_inplace(b,u,cols);linear_inplace(b,v,cols);mu2,fz2=forward_canonical(b,u,v);mu,fz=token;b.cz(u,mu);b.free(mu);b.cx(fz,fz2);b.free(fz);phase_inverse24_classical106(b,u,mu2,fz2[0]);b.free(mu2);b.free(fz2);linear_inplace(b,u,frm);linear_inplace(b,v,frm)
