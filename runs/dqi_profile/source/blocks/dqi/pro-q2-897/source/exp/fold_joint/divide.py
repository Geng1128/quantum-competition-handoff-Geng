"""Zero-aware streamed factor division24, no edits to arithmetic modules."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from exp.mul99.schedules import *
from exp.safe_scalar.phase import phase_inverse36_zero
from exp.noscratch.kernels import inverse3_noscratch
from exp.arithmetic.lowwidth import classical_transpose

def inverse_stream_zero(b,x):
    """Overwrite12-bit x with inv0(x); expose zero via bottom norm,68T.

    Same recursive streamed schedule as old66T inverse, with2AND flag only
    after deepest inverse's old-input wires are erased. Peak12+1 scratch.
    """
    n=len(x)
    if n==3:
        y=inverse3_noscratch(b,x);mask=b.alloc_bits(3);b.hmr(x,mask);b.swap(x,y);b.free(y)
        b.x(x);u=b.alloc_qubits(1)[0];F=b.alloc_qubits(1)[0]
        b.init_and(x[0],x[1],u);b.init_and(u,x[2],F);b.del_and(x[0],x[1],u);b.free(u);b.x(x)
        return mask,F
    k=n//2;to,frm,K=tower(n);linear_inplace(b,x,to)
    N=b.alloc_qubits(k);norm_xor_noscratch(b,x,N,K)
    nm,F=inverse_stream_zero(b,N);ar._phase_norm(b,x,nm,K);b.free(nm)
    outb=b.alloc_qubits(k);mul_xor_noscratch(b,x[k:],N,outb)
    raw=b.alloc_bits(n);b.hmr(x[k:],raw[k:])
    mul_xor_noscratch(b,x[:k],N,x[k:]);b.cx(outb,x[k:]);b.hmr(x[:k],raw[:k])
    b.swap(x[:k],x[k:]);b.swap(x[k:],outb);b.free(outb)
    ar._erase_norm(b,x,N,K);b.free(N);linear_inplace(b,x,frm)
    mask=classical_transpose(b,raw,to);b.free(raw);return mask,F

def divide_zero371(b,x,y,K=None,extra_zero_mask=None,after=None):
    """x/=y for y!=0, identity when y0. <=371T,37 private, all-input safe.

    Norm's zero flag costs2T in streamed inverse, not a23-bit factor test.
    Its phase is fused into the existing36T inverse-norm phase correction.
    WIDTH OBSTRUCTION: u12 remains live across a24-bit product target, +F.
    """
    if K is None:K=extension_columns()[2]
    u=b.alloc_qubits(12);norm_xor_noscratch(b,y,u,K)
    nm,F=inverse_stream_zero(b,u);ar._phase_norm(b,y,nm,K);b.free(nm)
    b.cx(y[12:],y[:12]);b.cx(F,y[0]);oldmask=multiply_forward99(b,x,y,K);b.cx(F,y[0]);b.cx(y[12:],y[:12])
    b.cx(F,u[0]);masks=[]
    for off in(0,12):
        half=x[off:off+12];out=b.alloc_qubits(12);mul_xor_noscratch(b,half,u,out)
        mask=b.alloc_bits(12);b.hmr(half,mask);masks.append(mask);b.swap(half,out);b.free(out)
    b.cx(F,u[0]);um=b.alloc_bits(12);b.hmr(u,um);b.free(u)
    N=b.alloc_qubits(12);norm_xor_noscratch(b,y,N,K)
    b.cx(F,N[0])
    for off,mask in zip((0,12),masks):phase_product(b,x[off:off+12],N,mask);b.free(mask)
    b.cx(F,N[0]);b.cx(F,y[0]);phase_product24(b,x,y,oldmask,K);b.cx(F,y[0]);b.free(oldmask)
    if after is not None:after(F)
    fm=b.alloc_bits(1);b.hmr(F,fm[0]);b.free(F)
    if extra_zero_mask is not None:b.cx(extra_zero_mask,fm[0])
    phase_inverse36_zero(b,N,um,fm[0]);b.free(um);b.free(fm)
    ar._erase_norm(b,y,N,K);b.free(N)
