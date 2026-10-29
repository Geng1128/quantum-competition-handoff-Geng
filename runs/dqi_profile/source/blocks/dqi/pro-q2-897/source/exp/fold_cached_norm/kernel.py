"""Spend spare global width on a retained norm inside quadratic fold division."""
from pathlib import Path
import sys
from types import FunctionType
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from exp.mul99.schedules import multiply_forward99,phase_inverse36_pivot
from exp.noscratch.schedules import norm_xor_noscratch
from exp.noscratch.kernels import mul_xor_noscratch
from exp.inverse12_escape.joint import inverse_initialized
from exp.arithmetic.division import phase_product24
from exp.arithmetic.field import erase_product
from exp.zero_padding_synthesis import kernel as prior
from exp.inverse12_escape.zero1 import inverse_zero
from exp.fold_cached_norm.phase34 import phase_inverse34
from exp.inverse12_escape.zero1 import inverse6_zero20
from exp.phase24_tradeoff.phase6small import phase_inverse6_classical6
from exp.alternating_gauge import scalar as scalar_prior

def divide325(b,x,y,K=None):
    """x/=y, preserve y,325 executed T/36 private; y!=0 or x=0 promise.

    Inner synthesis explicitly clips its factor before this call. The norm
    stays live through both base-field output scales, allowing immediate
    old-half phase repayment and avoiding its42T reconstruction.
    """
    assert K is not None
    b.cx(y[12:],y[:12]);old=multiply_forward99(b,x,y,K);b.cx(y[12:],y[:12])
    N=b.alloc_qubits(12);norm_xor_noscratch(b,y,N,K);u=inverse_initialized(b,N)
    for off in (0,12):
        half=x[off:off+12];out=b.alloc_qubits(12)
        mul_xor_noscratch(b,half,u,out);erase_product(b,out,N,half)
        b.swap(half,out);b.free(out)
    mask=b.alloc_bits(12);b.hmr(u,mask);b.free(u)
    phase_inverse36_pivot(b,N,mask);b.free(mask)
    prior.ar._erase_norm(b,y,N,K);b.free(N)
    phase_product24(b,x,y,old,K);b.free(old)

def bind(fn,**changes):
    return FunctionType(fn.__code__,dict(fn.__globals__,**changes),fn.__name__,fn.__defaults__,fn.__closure__)

divide323=bind(divide325,phase_inverse36_pivot=phase_inverse34)
divide323.__doc__='Unit factor division:323T/36private; retained norm and34T/21private inverse phase.'

def divide_zero324(b,x,y,K=None,extra_zero_mask=None,after=None):
    """Identity-at-zero division,324T/49 private, all HMR phases closed."""
    assert K is not None
    N=b.alloc_qubits(12);norm_xor_noscratch(b,y,N,K);u,F=inverse_zero(b,N)
    b.cx(y[12:],y[:12]);b.cx(F,y[0]);old=multiply_forward99(b,x,y,K)
    b.cx(F,y[0]);b.cx(y[12:],y[:12]);b.cx(F,u[0])
    for off in (0,12):
        half=x[off:off+12];out=b.alloc_qubits(12);mul_xor_noscratch(b,half,u,out)
        b.cx(F,N[0]);erase_product(b,out,N,half);b.cx(F,N[0])
        b.swap(half,out);b.free(out)
    b.cx(F,u[0]);um=b.alloc_bits(12);b.hmr(u,um);b.free(u)
    b.cx(F,y[0]);phase_product24(b,x,y,old,K);b.cx(F,y[0]);b.free(old)
    if after is not None:after(F)
    fm=b.alloc_bits(1);b.hmr(F,fm[0]);b.free(F)
    if extra_zero_mask is not None:b.cx(extra_zero_mask,fm[0])
    phase_inverse34(b,N,um,fm[0]);b.free(um);b.free(fm)
    prior.ar._erase_norm(b,y,N,K);b.free(N)

absorb=bind(prior.absorb_odd,divide_lowwidth369=divide323)
absorb_even=bind(prior.absorb_even,divide_lowwidth369=divide323)
outer=bind(prior.outer,divide_zero369=divide_zero324)

def _inverse6_zero(b,x,with_zero):
    assert with_zero
    return inverse6_zero20(b,x)

divide_zero113=bind(scalar_prior.divide_zero117,inverse6_init19=_inverse6_zero,phase_inverse_generic6=phase_inverse6_classical6)
multiply_zero141=bind(scalar_prior.multiply_zero144,inverse_zero=inverse_zero,phase_inverse36_zero=phase_inverse34)
prefix_reverse137=bind(scalar_prior.prefix_reverse141,divide_zero117=divide_zero113)
odd_decode=bind(prior.synthesize.__globals__['odd_decode'],prefix_reverse141=prefix_reverse137)
synthesize=bind(prior.synthesize,absorb=absorb,absorb_even=absorb_even,outer=outer,odd_decode=odd_decode,divide_zero117=divide_zero113)
unsynthesize=bind(prior.unsynthesize,absorb=absorb,absorb_even=absorb_even,outer=outer,odd_decode=odd_decode,multiply_zero144=multiply_zero141)
