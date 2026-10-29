"""New19-AND GF64 inverse at norm isotopy6; I24 pair190/219T,39private.

No imported module is changed. The six-bit XOR inverse supports arbitrary
output targets; initialized totalQ21 is wider than iso20, but nested I24 peaks
remain36forward/39reverse because the larger subcall is not the global peak.
"""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from exp.inverse_search import inverse20 as ref
from exp.inverse_search import inverse20_iso as iso
from exp.inverse_search.inverse20_iso import *
from exp.inverse_width.kernel import compress_D,erase_N_cache
DAG19=json.loads(Path(__file__).with_name('inverse19.json').read_text())

def raw_forms19():
    f=field(3);scale=6;inv=f.invert(scale);out=[]
    for mask,v in tensor(3):
        left=sum(((mask&f.mul(scale,1<<i)).bit_count()&1)<<(1+i)for i in range(3))
        right=sum(((mask&f.mul(inv,1<<i)).bit_count()&1)<<(4+i)for i in range(3));out.append((left,right))
    return out

def inverse6_xor19(b,x,z=None,keep_norm=False):
    assert len(x)==6;to,frm,K=tower(6);linear_inplace(b,x,to)
    if z is not None:linear_inplace(b,z,to)
    g=b.alloc_qubits(6)
    for i,(left,right)in enumerate(raw_forms19()):
        with ref.expose_pair(b,x,left,right)as(p,q):b.init_and(p,q,g[i])
    prefix=list(x)+list(g);forms=ref.norm_forms(K);w=ref.inverse3_virtual(b,prefix,forms);source=prefix+list(w)
    if z is None:z=b.alloc_qubits(6)
    for j,e in enumerate(DAG19['outputs']):ref.form_xor(b,source,e&65535,z[j])
    for i,(left,right)in enumerate(DAG19['gates']):
        v=sum(((e>>(16+i))&1)<<j for j,e in enumerate(DAG19['outputs']));assert v
        with ref.expose_pair(b,source,left,right)as(p,q):rankone_xor(b,p,q,z,v)
    wm=b.alloc_bits(3);b.hmr(w,wm);b.free(w);ref.phase_inverse3_virtual(b,prefix,forms,wm);b.free(wm)
    kept=None
    if keep_norm:
        kept=b.alloc_qubits(3)
        for j,f in enumerate(forms):ref.form_xor(b,prefix,f,kept[j])
    for i,(left,right)in reversed(list(enumerate(raw_forms19()))):
        with ref.expose_pair(b,x,left,right)as(p,q):b.del_and(p,q,g[i])
    b.free(g);linear_inplace(b,x,frm);linear_inplace(b,z,frm)
    return (z,kept)if keep_norm else z

# Rebind only this module's private copies of the recursive schedules.
from types import FunctionType
inverse6_xor20=inverse6_xor19
for _name in('inverse_initialized','inverse12_keep_norm','inverse_inplace_forward20'):
    _old=getattr(ref,_name);globals()[_name]=FunctionType(_old.__code__,globals(),_name,_old.__defaults__,_old.__closure__)

def inverse24_forward19(b,x,K=None):
    cols,frm,_=extension_columns(K);linear_inplace(b,x,cols)
    mask=inverse_inplace_forward20(b,x);linear_inplace(b,x,frm);return mask

def inverse_inplace_reverse19(b,x,mask):
    to,frm,K=tower(24);linear_inplace(b,x,to);N=b.alloc_qubits(12);iso.norm_xor(b,x,N,K)
    t,D=inverse12_keep_norm(b,N);C=compress_D(b,N,D);out=b.alloc_qubits(12);mul_xor_noscratch(b,x[12:],t,out)
    erase_product(b,out,N,x[12:]);mul_xor_noscratch(b,x[:12],t,x[12:]);b.cx(out,x[12:])
    b.cx(out,x[12:]);erase_product(b,x[12:],N,x[:12]);b.cx(out,x[12:])
    b.swap(x[:12],x[12:]);b.swap(x[12:],out);b.free(out)
    erase_N_cache(b,t,N,C);ar._erase_norm(b,x,t,K);b.free(t);linear_inplace(b,x,frm);b.cz(x,mask);b.free(mask)

def inverse24_reverse19(b,x,mask,K=None):
    cols,frm,_=extension_columns(K);linear_inplace(b,x,cols);inverse_inplace_reverse19(b,x,mask);linear_inplace(b,x,frm)
