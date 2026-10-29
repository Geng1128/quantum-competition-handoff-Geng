"""21T correlated inverse6 XOR seed, arbitrary target, clean private state."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from exp.noscratch.kernels import *

def inverse6_xor21(b,x,z):
    assert len(x)==len(z)==6
    to,frm,K=tower(6);linear_inplace(b,x,to);linear_inplace(b,z,to)
    N=b.alloc_qubits(3);f=field(3)
    linear_xor(b,x[:3],N,square_columns(3))
    linear_xor(b,x[3:],N,[f.mul(K,v)for v in square_columns(3)])
    mul_xor_noscratch(b,x[:3],x[3:],N)
    w=inverse3_noscratch(b,N)
    b.cx(x[3:],x[:3]);mul_xor_noscratch(b,x[:3],w,z[:3]);b.cx(x[3:],x[:3])
    mul_xor_noscratch(b,x[3:],w,z[3:])
    erase_inverse(b,N,w);b.free(w);ar._erase_norm(b,x,N,K);b.free(N)
    linear_inplace(b,x,frm);linear_inplace(b,z,frm)
