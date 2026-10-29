"""Full product/quotient jointly updates explicit odd continuant r, no shear."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from exp.tail_alias.kernel import *
from exp.mul99.schedules import divide_zero_bridge414

def full_product(b,head,A,qi,target):
 n=len(target);V,W=base.vandermonde(n)
 for t in range(n):old.rank_affine(b,head,[1,t],A+[qi],V[t][:len(A)+1],target,[W[j][t]for j in range(n)])

def full_quotient(b,head,A,qi,B,r,kap):
 k=len(B)
 if not k:ar.mul_xor0(b,head[1],qi,r);return
 words=A[1:]+[qi]
 for ac,bc,tc in reversed(base.quotient_terms(k+2,kap)):
  old.rank_affine(b,head,ac,words,bc,B+[r],tc)

def absorb_even(b,head,A,B,qi,r,c,kap,reverse=False,transcript=None):
 """r=qnext->qprev in forward, inverse restores qnext; qi disjoint r."""
 k=len(B);out=head+B;target=out+[r]
 if not reverse:
  value=old.eval_implicit(b,A,qi,c);F=dv.zero_flag(b,value);old.eval_implicit(b,A,qi,c,True)
  b.cx(F,A[0][0]);spare=[b.alloc_qubits(12),b.alloc_qubits(12)];full=spare+B+[r]
  linear_words(b,full,base.embedding_matrix(k+3,kap));full_product(b,head,A,qi,full);b.cx(F,A[0][0])
  hm=b.alloc_bits(24);b.hmr(combine(head),hm)
  for w,z in zip(head,spare):b.swap(w,z);b.free(z)
  fm=b.alloc_bits(1);b.hmr(F,fm[0]);b.free(F)
  return out,A,(hm,fm)
 linear_words(b,target,base.embedding_matrix(k+3,kap),True)
 value=old.eval_implicit(b,A,qi,c);cx(b,head[1],head[0],c);hm,fm=transcript
 divide_zero_bridge414(b,combine(head),value,K=K,extra_zero_mask=fm[0]);b.free(fm)
 cx(b,head[1],head[0],c);b.cz(combine(head),hm);b.free(hm);old.eval_implicit(b,A,qi,c,True)
 full_quotient(b,head,A,qi,B,r,kap)
 return out,A,transcript
