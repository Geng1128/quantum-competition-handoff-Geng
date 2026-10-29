"""25-private zero-aware scalar context; same117T as promised old divide."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from exp.tail_alias.kernel import *
from exp.inverse19_joint.kernel import inverse6_init19,inverse_zero
from exp.arithmetic.lowwidth import phase_inverse_generic6
from exp.safe_scalar.phase import phase_inverse36_zero
from exp.noscratch.schedules import norm_xor_noscratch


def divide_zero117(b,x,y,before=None,after=None):
 """All-input x/=y or x when y0;117T/25private; hooks see F=[y0]."""
 to,frm,kk=ar.tower(12);N=b.alloc_qubits(6);ar.linear_inplace(b,y,to);norm_xor_noscratch(b,y,N,kk);ar.linear_inplace(b,y,frm)
 u,F=inverse6_init19(b,N,True)
 if before is not None:before(F)
 conj=[ar.field(12).frobenius(1<<i,6)for i in range(12)]
 ar.linear_inplace(b,y,conj);b.cx(F,y[0]);out=b.alloc_qubits(12);ar.mul_xor0(b,x,y,out)
 oldm=b.alloc_bits(12);b.hmr(x,oldm);b.swap(x,out);b.free(out);b.cx(F,y[0]);ar.linear_inplace(b,y,conj)
 ar.linear_inplace(b,x,to);b.cx(F,u[0]);b.cx(F,N[0])
 for off in(0,6):
  half=x[off:off+6];out=b.alloc_qubits(6);ar.mul_xor0(b,half,u,out)
  m=b.alloc_bits(6);b.hmr(half,m);b.swap(half,out);b.free(out);ar.phase_product(b,half,N,m);b.free(m)
 b.cx(F,N[0]);b.cx(F,u[0]);ar.linear_inplace(b,x,frm)
 b.cx(F,y[0]);ar.phase_product(b,x,y,oldm);b.cx(F,y[0]);b.free(oldm)
 if after is not None:after(F)
 fm=b.alloc_bits(1);b.hmr(F,fm[0]);b.free(F);b.z(fm[0]);zm=b.alloc_bits(6);b.bit_store(zm,False);b.cx(fm[0],zm[0]);ar.phase_product(b,N,u,zm);b.free(zm);b.free(fm)
 um=b.alloc_bits(6);b.hmr(u,um);b.free(u);phase_inverse_generic6(b,N,um);b.free(um)
 ar.linear_inplace(b,y,to);ar._erase_norm(b,y,N,kk);ar.linear_inplace(b,y,frm);b.free(N)

def multiply_zero144(b,x,y):
 """Inverse of divide_zero117,144T/25private, no deferred transcript."""
 u,F=inverse_zero(b,y);b.cx(F,y[0]);out=b.alloc_qubits(12);ar.mul_xor0(b,x,y,out)
 m=b.alloc_bits(12);b.hmr(x,m);b.swap(x,out);b.free(out);b.cx(F,y[0]);b.cx(F,u[0]);ar.phase_product(b,x,u,m);b.cx(F,u[0]);b.free(m)
 um=b.alloc_bits(12);fm=b.alloc_bits(1);b.hmr(u,um);b.hmr(F,fm[0]);b.free(u);b.free(F)
 phase_inverse36_zero(b,y,um,fm[0]);b.free(um);b.free(fm)

def prefix_reverse141(b,x,p,q,m):
 """Inverse oldpg.forward:141T/25private, including zero-q swap chart."""
 def before(F):b.cswap(F,x,p);b.cx(p,x);b.ccx(F,p,x)
 def after(F):
  b.cz(x,m)
  for j in range(12):b.ccz(F,x[j],m[j]);b.ccz(F,p[j],m[j])
  b.free(m)
 divide_zero117(b,x,q,before,after)
