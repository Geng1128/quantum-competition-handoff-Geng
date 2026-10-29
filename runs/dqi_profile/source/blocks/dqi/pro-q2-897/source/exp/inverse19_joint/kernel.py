"""Width-audited19AND inverse substitutions; stable imports remain unchanged."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from exp.inverse_width_subspaces import kernel19 as basic
from exp.inverse_width_subspaces.kernel19 import *
from exp.safe_scalar import kernel as safe
from exp.safe_scalar.phase import phase_inverse36_zero
from exp.noscratch.schedules import norm_xor_noscratch
LAY=json.loads(Path(__file__).with_name('layout.json').read_text())

def inverse6_init19(b,x,with_zero=False):
    """Fresh inverse6:19T/Q19, or inverse+zero21T/Q19; preserve x.

    Reuse two redundant raw norm directions as output. Four other norm
    directions remain until all final products are added. Compute zero only
    AFTER those four wires are erased, using the still-live inverse norm3.
    """
    to,frm,K=tower(6);linear_inplace(b,x,to);g=b.alloc_qubits(6)
    for i,(l,r)in enumerate(basic.raw_forms19()):
        with ref.expose_pair(b,x,l,r)as(p,q):b.init_and(p,q,g[i])
    prefix=list(x)+list(g);forms=ref.norm_forms(K);w=ref.inverse3_virtual(b,prefix,forms)
    linear_inplace(b,g,LAY['raw_columns']);R,E=g[:4],g[4:]
    z=list(E)+list(b.alloc_qubits(4));source=list(x)+list(R)+list(w)
    for j,e in enumerate(LAY['outputs']):ref.form_xor(b,source,e&65535,z[j])
    for i,(l,r)in enumerate(LAY['gates']):
        v=sum(((e>>(16+i))&1)<<j for j,e in enumerate(LAY['outputs']));assert v
        with ref.expose_pair(b,source,l,r)as(p,q):rankone_xor(b,p,q,z,v)
    # Four retained raw products have only quadratic phase corrections.
    rm=b.alloc_bits(4);b.hmr(R,rm);b.free(R)
    for i,(l,r)in enumerate(basic.raw_forms19()):
        with ref.expose_pair(b,x,l,r)as(p,q):
            for j,row in enumerate(LAY['raw_rows'][:4]):
                if row>>i&1:b.ccz(p,q,rm[j])
    b.free(rm);linear_inplace(b,z,LAY['output_inverse_columns'])
    F=None
    if with_zero:
        b.x(w);u=b.alloc_qubits(1)[0];F=b.alloc_qubits(1)[0]
        b.init_and(w[0],w[1],u);b.init_and(u,w[2],F);b.del_and(w[0],w[1],u);b.free(u);b.x(w)
    ar._erase_norm(b,z,w,K);b.free(w);linear_inplace(b,x,frm);linear_inplace(b,z,frm)
    return(z,F)if with_zero else z

def inverse6_xor19(b,x,z=None,keep_norm=False):
    if z is None and not keep_norm:return inverse6_init19(b,x)
    return basic.inverse6_xor19(b,x,z,keep_norm)

# Local recursive schedules, no mutation of imported globals.
from types import FunctionType
inverse6_xor20=inverse6_xor19
for _name in('inverse_initialized','inverse12_keep_norm','inverse_inplace_forward20','inverse_inplace_reverse20'):
    _old=getattr(ref,_name);globals()[_name]=FunctionType(_old.__code__,globals(),_name,_old.__defaults__,_old.__closure__)

def inverse_zero(b,x):
    """GF12 inverse0 +zero:66T/Q31, preserve input; closure unchanged36T."""
    to,frm,K=tower(12);linear_inplace(b,x,to);N=b.alloc_qubits(6);norm_xor_noscratch(b,x,N,K)
    t,F=inverse6_init19(b,N,True);ar._erase_norm(b,x,N,K);b.free(N)
    out=b.alloc_qubits(12);b.cx(x[6:],x[:6]);mul_xor_noscratch(b,x[:6],t,out[:6]);mul_xor_noscratch(b,x[6:],t,out[6:]);b.cx(x[6:],x[:6])
    ar._erase_norm(b,out,t,K);b.free(t);linear_inplace(b,x,frm);linear_inplace(b,out,frm);return out,F

consume=safe.consume;erase=safe.erase

def inverse_apply_safe(b,a,action,incoming=None,defer=False):
    ai,F=inverse_zero(b,a);consume(b,ai,F,incoming)
    b.cx(F,a[0]);b.cx(F,ai[0]);action(ai);b.cx(F,ai[0]);b.cx(F,a[0])
    return erase(b,a,ai,F,defer)

from exp.arithmetic.lowwidth import phase_inverse_generic6,phase_product_quantum

def inverse12_xor73(b,x,z,base6=inverse6_init19):
    """Arbitrary-target XOR inverse12:73T/Q43, preserve x and target phase.

    Initialized inverse schedules are NOT arbitrary-target oracles: they erase
    t using norm(output), which is wrong for a preexisting target. Keep N and
    close t by a9T phase inverse instead.
    """
    to,frm,K=tower(12);linear_inplace(b,x,to);linear_inplace(b,z,to)
    N=b.alloc_qubits(6);norm_xor_noscratch(b,x,N,K);t=base6(b,N)
    b.cx(x[6:],x[:6]);mul_xor_noscratch(b,x[:6],t,z[:6]);mul_xor_noscratch(b,x[6:],t,z[6:]);b.cx(x[6:],x[:6])
    m=b.alloc_bits(6);b.hmr(t,m);b.free(t);phase_inverse_generic6(b,N,m);b.free(m)
    ar._erase_norm(b,x,N,K);b.free(N);linear_inplace(b,x,frm);linear_inplace(b,z,frm)
