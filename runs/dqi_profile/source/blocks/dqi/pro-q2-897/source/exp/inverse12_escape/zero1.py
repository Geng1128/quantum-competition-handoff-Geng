"""65T/Q31 zero-aware I12 from a one-AND correlated zero output.
Not an improvement to isolated64T I12 inversion; no live module is changed.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from exp.inverse19_joint.kernel import *
from types import FunctionType
ZLAY=json.loads(Path(__file__).with_name('zero_layout.json').read_text())

def erase_raw_cache(b,x,cache,rows):
    m=b.alloc_bits(len(cache));b.hmr(cache,m);b.free(cache)
    for i,(l,r)in enumerate(basic.raw_forms19()):
        with ref.expose_pair(b,x,l,r)as(p,q):
            for j,row in enumerate(rows):
                if row>>i&1:b.ccz(p,q,m[j])
    b.free(m)

def inverse6_zero20(b,x):
    to,frm,K=tower(6);linear_inplace(b,x,to);g=b.alloc_qubits(6)
    for i,(l,r)in enumerate(basic.raw_forms19()):
        with ref.expose_pair(b,x,l,r)as(p,q):b.init_and(p,q,g[i])
    prefix=list(x)+list(g);forms=ref.norm_forms(K);w=ref.inverse3_virtual(b,prefix,forms)
    linear_inplace(b,g,LAY['raw_columns']);R,E=g[:4],g[4:];z=list(E)+list(b.alloc_qubits(4));source=list(x)+list(R)+list(w)
    for j,e in enumerate(LAY['outputs']):ref.form_xor(b,source,e&65535,z[j])
    for i,(l,r)in enumerate(LAY['gates']):
        v=sum(((e>>(16+i))&1)<<j for j,e in enumerate(LAY['outputs']))
        with ref.expose_pair(b,source,l,r)as(p,q):rankone_xor(b,p,q,z,v)
    # One norm coordinate lies in the4D live raw-control span. Keep it, erase3.
    linear_inplace(b,R,ZLAY['cache_columns']);erase_raw_cache(b,x,R[1:],ZLAY['raw_cache_rows'][1:])
    source=list(x)+[R[0]]+list(w);left=1|(ZLAY['left_w']<<8);right=1^ZLAY['input_form']^(1<<7)^(ZLAY['right_w']<<8)
    F=b.alloc_qubits(1)[0]
    with ref.expose_pair(b,source,left,right)as(p,q):b.init_and(p,q,F)
    erase_raw_cache(b,x,R[:1],ZLAY['raw_cache_rows'][:1]);linear_inplace(b,z,LAY['output_inverse_columns']);ar._erase_norm(b,z,w,K);b.free(w)
    linear_inplace(b,x,frm);linear_inplace(b,z,frm);return z,F

def inverse_zero(b,x):
    to,frm,K=tower(12);linear_inplace(b,x,to);N=b.alloc_qubits(6);norm_xor_noscratch(b,x,N,K)
    t,F=inverse6_zero20(b,N);ar._erase_norm(b,x,N,K);b.free(N)
    out=b.alloc_qubits(12);b.cx(x[6:],x[:6]);mul_xor_noscratch(b,x[:6],t,out[:6]);mul_xor_noscratch(b,x[6:],t,out[6:]);b.cx(x[6:],x[:6])
    ar._erase_norm(b,out,t,K);b.free(t);linear_inplace(b,x,frm);linear_inplace(b,out,frm);return out,F

from exp.inverse19_joint import kernel as parentkernel
_old=parentkernel.inverse_apply_safe
inverse_apply_safe=FunctionType(_old.__code__,globals(),'inverse_apply_safe',_old.__defaults__,_old.__closure__)
