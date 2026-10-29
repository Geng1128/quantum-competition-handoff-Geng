"""Explicit compositions; no patching or edits of existing arithmetic modules."""
try:
    from .kernels import *
except ImportError:
    from kernels import *
sys.path.insert(0,str(ROOT))
from exp.noscratch.schedules import norm_xor_noscratch,inverse_stream_noscratch
from exp.noscratch.phase39 import phase_inverse39
from exp.noscratch.kernels import mul_xor_noscratch
from division import phase_product24,zero_flag,phase_zero
try:
    from .phase36 import phase_inverse36_pivot
except ImportError:
    from phase36 import phase_inverse36_pivot


def multiply_forward99(b,x,y,K=None):
    """x*=y:99T/24scratch, old-x phase deferred. Promise x=0 when y=0."""
    z=b.alloc_qubits(24);mul24_xor99(b,x,y,z,K)
    m=b.alloc_bits(24);b.hmr(x,m);b.swap(x,z);b.free(z);return m


def divide_lowwidth372(b,x,y,K=None):
    """x/=y, preserves y:372T/24scratch. Promise x=0 when y=0."""
    return _divide_lowwidth(b,x,y,K,phase_inverse39)


def divide_lowwidth369(b,x,y,K=None):
    """Same division, <=369 EXECUTED T /24scratch; static count501T."""
    return _divide_lowwidth(b,x,y,K,phase_inverse36_pivot)


def _divide_lowwidth(b,x,y,K,phase_kernel):
    if K is None:K=extension_columns()[2]
    b.cx(y[12:],y[:12]);oldmask=multiply_forward99(b,x,y,K);b.cx(y[12:],y[:12])
    u=b.alloc_qubits(12);norm_xor_noscratch(b,y,u,K)
    nm=inverse_stream_noscratch(b,u);ar._phase_norm(b,y,nm,K);b.free(nm)
    masks=[]
    for offset in(0,12):
        half=x[offset:offset+12];out=b.alloc_qubits(12);mul_xor_noscratch(b,half,u,out)
        m=b.alloc_bits(12);b.hmr(half,m);masks.append(m);b.swap(half,out);b.free(out)
    um=b.alloc_bits(12);b.hmr(u,um);b.free(u)
    N=b.alloc_qubits(12);norm_xor_noscratch(b,y,N,K)
    for offset,m in zip((0,12),masks):phase_product(b,x[offset:offset+12],N,m);b.free(m)
    phase_kernel(b,N,um);b.free(um)
    ar._erase_norm(b,y,N,K);b.free(N)
    phase_product24(b,x,y,oldmask,K);b.free(oldmask)


def divide_zero_bridge417(b,x,y,K=None,extra_zero_mask=None):
    """x/=y if y!=0, else identity:417T/25scratch, all inputs exact."""
    flag=zero_flag(b,y);b.cx(flag,y[0])
    divide_lowwidth372(b,x,y,K)
    b.cx(flag,y[0])
    mask=b.alloc_bits(1);b.hmr(flag,mask[0]);b.free(flag)
    if extra_zero_mask is not None:b.cx(extra_zero_mask,mask[0])
    phase_zero(b,y,mask[0]);b.free(mask)


def divide_zero_bridge414(b,x,y,K=None,extra_zero_mask=None):
    """All-input exact zero bridge: <=414 EXECUTED T /25scratch."""
    flag=zero_flag(b,y);b.cx(flag,y[0])
    divide_lowwidth369(b,x,y,K)
    b.cx(flag,y[0])
    mask=b.alloc_bits(1);b.hmr(flag,mask[0]);b.free(flag)
    if extra_zero_mask is not None:b.cx(extra_zero_mask,mask[0])
    phase_zero(b,y,mask[0]);b.free(mask)


def multiply_zero_forward99(b,x,y,K=None):
    """x*=y if y!=0, else identity:146T/25scratch; deferred masks."""
    flag=zero_flag(b,y)
    z=b.alloc_qubits(24);mul24_xor99(b,x,y,z,K);b.ccx(flag,x,z)
    old=b.alloc_bits(24);b.hmr(x,old);b.swap(x,z);b.free(z)
    mask=b.alloc_bits(1);b.hmr(flag,mask[0]);b.free(flag)
    return old,mask[0]
