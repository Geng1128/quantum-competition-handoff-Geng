"""Reference compositions of the plugins; no changes to arithmetic modules."""
try:
    from .phase39 import *
except ImportError:
    from phase39 import *
from division import phase_product24


def norm_xor_noscratch(b,x,z,K):
    k=len(z);f=field(k)
    linear_xor(b,x[:k],z,square_columns(k))
    linear_xor(b,x[k:],z,[f.mul(K,v)for v in square_columns(k)])
    mul_xor_noscratch(b,x[:k],x[k:],z)


def inverse_stream_noscratch(b,x):
    """Same streamed inverse T3/6/12/24=3/21/66/192, n scratch."""
    n=len(x)
    if n==3:
        y=inverse3_noscratch(b,x);mask=b.alloc_bits(n)
        b.hmr(x,mask);b.swap(x,y);b.free(y);return mask
    k=n//2;to,frm,K=tower(n);linear_inplace(b,x,to)
    N=b.alloc_qubits(k);norm_xor_noscratch(b,x,N,K)
    nm=inverse_stream_noscratch(b,N);ar._phase_norm(b,x,nm,K);b.free(nm)
    outb=b.alloc_qubits(k);mul_xor_noscratch(b,x[k:],N,outb)
    raw=b.alloc_bits(n);b.hmr(x[k:],raw[k:])
    mul_xor_noscratch(b,x[:k],N,x[k:]);b.cx(outb,x[k:]);b.hmr(x[:k],raw[:k])
    b.swap(x[:k],x[k:]);b.swap(x[k:],outb);b.free(outb)
    ar._erase_norm(b,x,N,K);b.free(N);linear_inplace(b,x,frm)
    mask=classical_transpose(b,raw,to);b.free(raw);return mask


def multiply_forward_noscratch(b,x,y,K=None):
    z=b.alloc_qubits(24);mul24_xor_noscratch(b,x,y,z,K)
    m=b.alloc_bits(24);b.hmr(x,m);b.swap(x,z);b.free(z);return m


def divide_lowwidth378(b,x,y,K=None):
    """378T/24 scratch, promise x=0 when y=0; preserves y."""
    if K is None:K=extension_columns()[2]
    b.cx(y[12:],y[:12]);oldmask=multiply_forward_noscratch(b,x,y,K);b.cx(y[12:],y[:12])
    u=b.alloc_qubits(12);norm_xor_noscratch(b,y,u,K)
    nm=inverse_stream_noscratch(b,u);ar._phase_norm(b,y,nm,K);b.free(nm)
    masks=[]
    for offset in(0,12):
        half=x[offset:offset+12];out=b.alloc_qubits(12);mul_xor_noscratch(b,half,u,out)
        m=b.alloc_bits(12);b.hmr(half,m);masks.append(m);b.swap(half,out);b.free(out)
    um=b.alloc_bits(12);b.hmr(u,um);b.free(u)
    N=b.alloc_qubits(12);norm_xor_noscratch(b,y,N,K)
    for offset,m in zip((0,12),masks):phase_product(b,x[offset:offset+12],N,m);b.free(m)
    phase_inverse39(b,N,um);b.free(um)
    ar._erase_norm(b,y,N,K);b.free(N)
    phase_product24(b,x,y,oldmask,K);b.free(oldmask)
