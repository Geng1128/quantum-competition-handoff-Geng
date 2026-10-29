"""39-private paired GF24 inverse via a three-bit norm cache.

Forward191T is unchanged iso20. Reverse220T adds five Toffolis while
reducing reverse private workspace42->39. Explicit transcript inclinv0.
No stable module is edited or monkeypatched.
"""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from exp.inverse_search import inverse20_iso as iso
from exp.inverse_search import inverse20 as ref
from exp.inverse_search.inverse20_iso import *
from exp.arithmetic.lowwidth import classical_transpose
PART=json.loads(Path(__file__).with_name('partial5.json').read_text())
FULL=json.loads((ROOT/'exp/inverse_search/optimized.json').read_text())

def coefficient_xor(b,t,nm,e,anc):
    """anc ^=dot(nm,conjugate(t)*e), e GF64 public, t tower12."""
    f6=field(6)
    for i in range(6):
        v=f6.mul(e,1<<i)
        for r in bits(v):
            b.ccx(nm[r],t[i],anc);b.ccx(nm[r],t[i+6],anc);b.ccx(nm[r+6],t[i+6],anc)

def phase_partial_inverse6(b,u,t,nm,cm):
    """14T: quantum-mask upper3(inv(u)), classical-mask lower3(inv(u)).

    u canonicalGF64, t alreadytower12; nm12 classical, cm3classical.
    Upper/lower refer to the standardGF8 tower6, not polynomialGF64 bits.
    All u/norm/inverse prefix state and the one parity ancilla are restored.
    """
    to,frm,K=tower(6);linear_inplace(b,u,to);g=b.alloc_qubits(6)
    for i,(mask,_) in enumerate(tensor(3)):
        with ref.expose_pair(b,u,mask<<1,mask<<4)as(p,q):b.init_and(p,q,g[i])
    prefix=list(u)+list(g);forms=ref.norm_forms(K);w=ref.inverse3_virtual(b,prefix,forms);source=prefix+list(w)
    # Retained lower3 cache's ordinary classical HMR correction costs0T.
    for j,e in enumerate(FULL['outputs'][:3]):
        assert not(e&1)
        for r in bits(e&65535):b.cz(source[r-1],cm[j])
    cb=b.alloc_bits(1)[0];b.bit_store(cb,False)
    for i,(left,right)in enumerate(FULL['gates']):
        js=[j for j,e in enumerate(FULL['outputs'][:3]) if e>>(16+i)&1]
        if not js:continue
        for j in js:b.cx(cm[j],cb)
        with ref.expose_pair(b,source,left,right)as(p,q):b.ccz(p,q,cb)
        for j in reversed(js):b.cx(cm[j],cb)
    b.free(cb)
    # Exactly five products span the three missing output functions.
    anc=b.alloc_qubits(1)[0]
    def coeff(v):coefficient_xor(b,t,nm,apply(frm,v<<3),anc)
    for r in range(16):
        v=sum(((e>>r)&1)<<j for j,e in enumerate(PART['outputs']))
        if not v:continue
        coeff(v)
        if r:b.cz(anc,source[r-1])
        else:b.z(anc)
        coeff(v)
    for i,(left,right)in enumerate(PART['gates']):
        v=sum(((e>>(16+i))&1)<<j for j,e in enumerate(PART['outputs']))
        assert v
        coeff(v)
        with ref.expose_pair(b,source,left,right)as(p,q):b.ccz(p,q,anc)
        coeff(v)
    b.free(anc)
    wm=b.alloc_bits(3);b.hmr(w,wm);b.free(w);ref.phase_inverse3_virtual(b,prefix,forms,wm);b.free(wm)
    for i,(mask,_)in reversed(list(enumerate(tensor(3)))):
        with ref.expose_pair(b,u,mask<<1,mask<<4)as(p,q):b.del_and(p,q,g[i])
    b.free(g);linear_inplace(b,u,frm)

def compress_D(b,N,D):
    """Retain lower3 tower6 coordinates, eraseupper3 with quadratic norm phase."""
    to6,frm6,K6=tower(6);linear_inplace(b,D,to6);m=b.alloc_bits(3);b.hmr(D[3:],m);b.free(D[3:])
    mt=b.alloc_bits(6);b.bit_store(mt,False);b.cx(m,mt[3:]);canon=classical_transpose(b,mt,to6);b.free(mt);b.free(m)
    to12,frm12,K12=tower(12);linear_inplace(b,N,to12);ar._phase_norm(b,N,canon,K12);linear_inplace(b,N,frm12);b.free(canon)
    return D[:3]

def erase_N_cache(b,t,N,C):
    """Joint cleanup29T: norm(t)15 +sharedprefix9 +fivequantumCCZ."""
    to12,frm12,K12=tower(12);linear_inplace(b,t,to12);linear_inplace(b,N,to12)
    nm=b.alloc_bits(12);b.hmr(N,nm);b.free(N);frm6=tower(6)[1];f6=field(6)
    for j in range(3):
        for i in range(6):
            for r in bits(f6.mul(frm6[j],1<<i)):
                b.ccz(C[j],t[i],nm[r]);b.ccz(C[j],t[i+6],nm[r]);b.ccz(C[j],t[i+6],nm[r+6])
    cm=b.alloc_bits(3);b.hmr(C,cm);b.free(C)
    u=b.alloc_qubits(6);iso.norm_xor(b,t,u,K12);phase_partial_inverse6(b,u,t,nm,cm);b.free(nm);b.free(cm)
    ar._erase_norm(b,t,u,K12);b.free(u);linear_inplace(b,t,frm12)

def inverse_inplace_reverse39(b,x,mask):
    assert len(x)==24;k=12;to,frm,K=tower(24);linear_inplace(b,x,to)
    N=b.alloc_qubits(12);iso.norm_xor(b,x,N,K);t,D=iso.inverse12_keep_norm(b,N);C=compress_D(b,N,D)
    out=b.alloc_qubits(12);mul_xor_noscratch(b,x[12:],t,out)
    erase_product(b,out,N,x[12:]);mul_xor_noscratch(b,x[:12],t,x[12:]);b.cx(out,x[12:])
    b.cx(out,x[12:]);erase_product(b,x[12:],N,x[:12]);b.cx(out,x[12:])
    b.swap(x[:12],x[12:]);b.swap(x[12:],out);b.free(out)
    erase_N_cache(b,t,N,C);ar._erase_norm(b,x,t,K);b.free(t);linear_inplace(b,x,frm);b.cz(x,mask);b.free(mask)

def inverse24_forward39(b,x,K=None):return iso.inverse24_forward20(b,x,K)

def inverse24_reverse39(b,x,mask,K=None):
    cols,frm,_=extension_columns(K);linear_inplace(b,x,cols);inverse_inplace_reverse39(b,x,mask);linear_inplace(b,x,frm)
