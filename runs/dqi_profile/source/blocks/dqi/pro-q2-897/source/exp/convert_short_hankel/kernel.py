"""Drop-in guarded CONVERT with exact short-H boundary.

Default preserves the stable arbitrary borrowed-history completion. Optional
monic_root=True requires the supplied history word initially zero; if absent
it is allocated zero. B/payloads otherwise arbitrary, including all gaps.
Reverse infers the option from its forward phase token.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from exp.safe_scalar import converter as old


def _root_ops(b,ops,a,ai,undo=False):
 for op in (reversed(ops)if undo else ops):
  if op[0]=='copy':b.x(op[1][0]);b.cx(ai,op[1])
  else:old.runops(b,[op],a,ai,undo)


def _pre_root(b,f,g,payloads,undo=False,incoming=None,defer=False):
 m=len(f)-1;n=m-1;a=g[-1];q0,q1=f[-2:]
 def action(ai):
  ops=[('s',v[-1],True)for v in payloads]
  ops +=[('m',g[i],v[-1],v[i])for v in payloads for i in range(n)]
  ops +=[('copy',q1),('m',q1,g[-2],q0),('s',q0,True),('c',g[:-1],[q0,q1],f[:-2])]
  _root_ops(b,ops,a,ai,undo)
 return old.inverse_apply(b,a,action,incoming,defer)


def _post_root(b,f,g,payloads,reverse=False,incoming=None,defer=False):
 n=len(g)-1;a=g[-1];q0,q1=f[-2:]
 def action(ai):
  # q1=1/(a+[a0]), so division by q1 is multiplication by the
  # already clipped a in this same context, even on scalar-zero charts.
  ops=[('s',v[-1],False)for v in payloads]
  ops +=[('m',g[i],v[i],v[-1])for v in payloads for i in range(n)]
  ops +=[('s',v[-1],True)for v in payloads]
  ops +=[('c',g[:-1],[q0,q1],f[:-2]),('s',q0,False),('m',q1,g[-2],q0),('copy',q1)]
  _root_ops(b,ops,a,ai,reverse)
 return old.inverse_apply(b,a,action,incoming,defer)


def _look_root(b,f,g,payloads,reverse,phase):
 m=len(f)-1;F=old.fresh_zero_flag(b,g[-1]);old.rotate(b,g,F)
 if not reverse:carry=_pre_root(b,f,g,payloads,defer=True)
 else:carry=_post_root(b,f,g,payloads,True,phase['a'].pop(m),True)
 old.canonical_permute(b,f,g,F);old.clear_flag(b,g[-1],F)
 parent=([v[-1]for v in payloads],f[-1],f[-2])
 old.look_solve(b,g,f[:-2],[v[:-1]for v in payloads],None,parent,reverse,phase)
 F=old.fresh_zero_flag(b,g[-1]);old.canonical_permute(b,f,g,F,True)
 if not reverse:phase['a'][m]=_post_root(b,f,g,payloads,False,carry,True)
 else:_pre_root(b,f,g,payloads,True,carry)
 old.rotate(b,g,F,True);old.clear_flag(b,g[-1],F)


def _bez_monic(b,O,E,payloads,reverse,history_scratch12,phase):
 def reduce():
  for i in range(len(O)):old.ar.mul_xor0(b,E[-1],O[i],E[i])
 reduce();lead=list(b.alloc_qubits(12))if history_scratch12 is None else list(history_scratch12)
 b.x(lead[0]);_look_root(b,O+[lead],E[:-1],payloads,reverse,phase);b.x(lead[0])
 if history_scratch12 is None:b.free(lead)
 reduce()


def convert_safe(b,A,B,reverse=False,history_scratch12=None,phase_token=None,*,monic_root=False):
 assert len(A)==len(B)==35
 phase={'a':{},'q':{},'_short_monic':bool(monic_root)}if phase_token is None else phase_token
 assert not reverse or phase_token is not None
 use_monic=phase['_short_monic']
 # Frobenius first on BOTH directions: reverse input is the completed C.
 old.frob(b,B,2048);old.frob(b,A,2048)
 E=B[::2];O=B[1::2];u=A[1::2];w=A[:34:2];top=A[-1]
 def bez(rev):
  if use_monic:_bez_monic(b,O,E,[u,w],rev,history_scratch12,phase)
  else:old.bez_lookahead(b,O,E,[u,w],rev,history_scratch12,phase)
 if not reverse:
  for i in range(17):old.ar.mul_xor0(b,top,O[i],w[i])
  bez(False)
  for i in range(17):old.ar.mul_xor0(b,E[i+1],w[i],top)
  for i in range(17):b.swap(A[2*i],A[2*i+1])
  old.hankel(b,B[1:],A[:34])
  for i in reversed(range(34)):b.swap(A[i],A[i+1])
 else:
  for i in range(34):b.swap(A[i],A[i+1])
  old.hankel(b,B[1:],A[:34],True)
  for i in range(17):b.swap(A[2*i],A[2*i+1])
  for i in range(17):old.ar.mul_xor0(b,E[i+1],w[i],top)
  bez(True)
  for i in range(17):old.ar.mul_xor0(b,top,O[i],w[i])
 old.frob(b,A,2);old.frob(b,B,2)
 if reverse:assert not phase['a']and not phase['q']
 return phase if not reverse else None
