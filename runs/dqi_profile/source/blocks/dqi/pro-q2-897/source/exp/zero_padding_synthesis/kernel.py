"""One-flag Taylor folds with common x-power padding on valid heads."""
from pathlib import Path
import sys
from types import FunctionType
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from exp.tail_alias.kernel import *
from exp.alternating_gauge import kernel as alt
from exp.alternating_gauge.even_fold import full_product,full_quotient
from exp.confluent_synthesis import kernel as ca


def zero_finish(b,value,F,fm):
 m=b.alloc_bits(1);b.hmr(F,m[0]);b.free(F);b.cx(fm[0],m[0]);b.free(fm)
 dv.phase_zero(b,value,m[0]);b.free(m)


def absorb_odd(b,head,A,B,Q,p,virtual,pconst,c,kap,reverse=False,transcript=None,wide=False):
 k=len(B);assert len(A)==k+1 and virtual is None and pconst==0
 out=B+head;v=epow(c,k+2);co=[v&4095,v>>12]
 if not reverse:
  value=old.eval_implicit(b,A,Q,c);F=dv.zero_flag(b,value)
  spare=[b.alloc_qubits(12),b.alloc_qubits(12)];b.cx(F,value[0]);cx(b,head[1],head[0],c)
  mul24_xor99(b,combine(head),value,combine(spare),K)
  cx(b,head[1],head[0],c);b.cx(F,value[0]);old.eval_implicit(b,A,Q,c,True)
  q_times_low(b,B,kap);b.cx(F,A[0][0]);low_product(b,head,A,B);b.cx(F,A[0][0])
  old.add_alias(b,p,spare,co);old.controlled_linear_xor(b,F,p,spare,co);R_to_tops(b,spare,B,c)
  hm=b.alloc_bits(24);b.hmr(combine(head),hm)
  for x,z in zip(head,spare):b.swap(x,z);b.free(z)
  fm=b.alloc_bits(1);b.hmr(F,fm[0]);b.free(F)
  return out,A,(hm,fm)
 R_to_tops(b,head,B,c,True);old.add_alias(b,p,head,co)
 value=old.eval_implicit(b,A,Q,c);F=dv.zero_flag(b,value);hm,fm=transcript
 old.controlled_linear_xor(b,F,p,head,co);b.cx(F,value[0]);divide_lowwidth369(b,combine(head),value,K);b.cx(F,value[0])
 cx(b,head[1],head[0],c);old.eval_implicit(b,A,Q,c,True)
 b.cx(F,A[0][0]);low_product(b,head,A,B);b.cx(F,A[0][0]);q_times_low(b,B,kap,True)
 value=old.eval_implicit(b,A,Q,c);zero_finish(b,value,F,fm);old.eval_implicit(b,A,Q,c,True)
 b.cz(combine(head),hm);b.free(hm)
 return out,A,transcript


def absorb_even(b,head,A,B,Q,r,c,kap,reverse=False,transcript=None):
 k=len(B);out=head+B;target=out+[r]
 if not reverse:
  value=old.eval_implicit(b,A,Q,c);F=dv.zero_flag(b,value);old.eval_implicit(b,A,Q,c,True)
  b.cx(F,A[0][0]);spare=[b.alloc_qubits(12),b.alloc_qubits(12)];full=spare+B+[r]
  linear_words(b,full,base.embedding_matrix(k+3,kap));full_product(b,head,A,Q,full);b.cx(F,A[0][0])
  old.controlled_linear_xor(b,F,r,full[-3:-1],[kap,1])
  hm=b.alloc_bits(24);b.hmr(combine(head),hm)
  for x,z in zip(head,spare):b.swap(x,z);b.free(z)
  fm=b.alloc_bits(1);b.hmr(F,fm[0]);b.free(F)
  return out,A,(hm,fm)
 value=old.eval_implicit(b,A,Q,c);F=dv.zero_flag(b,value);old.eval_implicit(b,A,Q,c,True)
 old.controlled_linear_xor(b,F,r,out[-2:],[kap,1]);linear_words(b,target,base.embedding_matrix(k+3,kap),True)
 value=old.eval_implicit(b,A,Q,c);cx(b,head[1],head[0],c);hm,fm=transcript
 b.cx(F,value[0]);divide_lowwidth369(b,combine(head),value,K);b.cx(F,value[0]);cx(b,head[1],head[0],c)
 b.cz(combine(head),hm);b.free(hm);old.eval_implicit(b,A,Q,c,True)
 full_quotient(b,head,A,Q,B,r,kap)
 value=old.eval_implicit(b,A,Q,c);zero_finish(b,value,F,fm);old.eval_implicit(b,A,Q,c,True)
 return out,A,transcript

synthesize=FunctionType(alt.synthesize.__code__,dict(alt.synthesize.__globals__,old=ca.old,absorb=absorb_odd,absorb_even=absorb_even),'synthesize',alt.synthesize.__defaults__,alt.synthesize.__closure__)
unsynthesize=FunctionType(alt.unsynthesize.__code__,dict(alt.unsynthesize.__globals__,old=ca.old,absorb=absorb_odd,absorb_even=absorb_even),'unsynthesize',alt.unsynthesize.__defaults__,alt.unsynthesize.__closure__)
