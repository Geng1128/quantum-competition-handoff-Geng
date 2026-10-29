"""Retained-norm GF12 division:117T,18 private, complete local HMR phase."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'exp/arithmetic'))
from field import *
import field as ar
from lowwidth import phase_inverse_generic6

def divide117(b,x,y):
    """x/=y, y preserved; y=0 requires x=0. 117T/18 private qubits."""
    f=field(12);to,frm,K=tower(12);conj=[f.frobenius(1<<i,6) for i in range(12)]
    linear_inplace(b,y,conj);out=b.alloc_qubits(12);mul_xor0(b,x,y,out)
    old=b.alloc_bits(12);b.hmr(x,old);b.swap(x,out);b.free(out);linear_inplace(b,y,conj)
    N=b.alloc_qubits(6);linear_inplace(b,y,to);ar._norm_xor(b,y,N,K);linear_inplace(b,y,frm)
    u=inverse(b,N);linear_inplace(b,x,to)
    for off in(0,6):
        half=x[off:off+6];out=b.alloc_qubits(6);mul_xor0(b,half,u,out)
        m=b.alloc_bits(6);b.hmr(half,m);b.swap(half,out);b.free(out);phase_product(b,half,N,m);b.free(m)
    um=b.alloc_bits(6);b.hmr(u,um);b.free(u);phase_inverse_generic6(b,N,um);b.free(um)
    linear_inplace(b,y,to);ar._erase_norm(b,y,N,K);linear_inplace(b,y,frm);b.free(N)
    linear_inplace(b,x,frm);phase_product(b,x,y,old);b.free(old)
