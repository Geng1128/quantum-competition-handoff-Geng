"""Zero-aware inverse12: expose zero cheaply at the bottom tower norm."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from exp.inverse_search.inverse20_iso import *
from exp.noscratch.schedules import norm_xor_noscratch
from exp.safe_scalar.phase import phase_inverse36_zero

def inverse_zero(b,x):
    """Preserve x, return inverse0(x) and [x=0],67T with clean scratch."""
    to,frm,K=tower(12);linear_inplace(b,x,to)
    N=b.alloc_qubits(6);norm_xor_noscratch(b,x,N,K)
    t,D=inverse6_xor20(b,N,keep_norm=True)
    b.x(D);u=b.alloc_qubits(1)[0];F=b.alloc_qubits(1)[0]
    b.init_and(D[0],D[1],u);b.init_and(u,D[2],F);b.del_and(D[0],D[1],u);b.free(u);b.x(D)
    to6,frm6,K6=tower(6);linear_inplace(b,N,to6)
    ar._erase_norm(b,N,D,K6);b.free(D);linear_inplace(b,N,frm6)
    ar._erase_norm(b,x,N,K);b.free(N)
    out=b.alloc_qubits(12);b.cx(x[6:],x[:6])
    mul_xor_noscratch(b,x[:6],t,out[:6]);mul_xor_noscratch(b,x[6:],t,out[6:]);b.cx(x[6:],x[:6])
    ar._erase_norm(b,out,t,K);b.free(t)
    linear_inplace(b,x,frm);linear_inplace(b,out,frm)
    return out,F

def consume(b,ai,F,token):
    if token is not None:
        m,z=token;b.cz(ai,m);b.cz(F,z[0]);b.free(m);b.free(z)

def erase(b,a,ai,F,defer):
    m=b.alloc_bits(12);z=b.alloc_bits(1)
    b.hmr(ai,m);b.hmr(F,z[0]);b.free(ai);b.free(F)
    if defer:return m,z
    phase_inverse36_zero(b,a,m,z[0]);b.free(m);b.free(z)

def inverse_apply_safe(b,a,action,incoming=None,defer=False):
    ai,F=inverse_zero(b,a);consume(b,ai,F,incoming)
    b.cx(F,a[0]);b.cx(F,ai[0]);action(ai);b.cx(F,ai[0]);b.cx(F,a[0])
    return erase(b,a,ai,F,defer)
