"""Closed valid-tail scale pair; q=0 requires d=0 throughout."""
try:
    from .scalar import *
except ImportError:
    from scalar import *
from lowwidth import phase_inverse36

def multiply_closed144(b,x,q):
    """x*=q, 144T/19private, q0 requires x0; all HMR phases closed."""
    out=b.alloc_qubits(12);mul_xor0(b,x,q,out);m=b.alloc_bits(12);b.hmr(x,m);b.swap(x,out);b.free(out)
    iq=inverse(b,q);phase_product(b,x,iq,m);b.free(m)
    im=b.alloc_bits(12);b.hmr(iq,im);b.free(iq);phase_inverse36(b,q,im);b.free(im)

def divide_safe138(b,x,q):
    """All-input x/=q when q!=0, identity whenq0;138T/20private."""
    try:from .prefix import zero_compact,zero_compact_free
    except ImportError:from prefix import zero_compact,zero_compact_free
    f=zero_compact(b,q);b.cx(f,q[0]);divide117(b,x,q);b.cx(f,q[0]);zero_compact_free(b,q,f)

def multiply_safe165(b,x,q):
    """Inverse all-input tail map,165T/20private, no deferred phase."""
    try:from .prefix import zero_compact,zero_compact_free
    except ImportError:from prefix import zero_compact,zero_compact_free
    f=zero_compact(b,q);b.cx(f,q[0]);multiply_closed144(b,x,q);b.cx(f,q[0]);zero_compact_free(b,q,f)

def controlled_linear_xor(b,F,x,targets,coeffs):
    """targets[j]^=F*coeffs[j]*x,12T,0scratch; GF12 coefficients classical."""
    f=field(12);p=next(i for i,v in enumerate(coeffs) if v);v=coeffs[p];vi=f.invert(v)
    for i,c in enumerate(coeffs):
        if i!=p:linear_xor(b,targets[p],targets[i],[f.mul(f.mul(c,vi),1<<j) for j in range(12)])
    linear_inplace(b,targets[p],[f.mul(vi,1<<j) for j in range(12)])
    b.ccx(F,x,targets[p]);linear_inplace(b,targets[p],[f.mul(v,1<<j) for j in range(12)])
    for i in reversed(range(len(coeffs))):
        if i!=p:linear_xor(b,targets[p],targets[i],[f.mul(f.mul(coeffs[i],vi),1<<j) for j in range(12)])
