"""Half-density continuant prefix: odd encoded heads, one rolling odd q.

Raw b0 becomes a reversible physical q0=1 border; r carries its old value.
The border is freed after inner folds and r becomes the numerator top word.
No persistent quantum zero flag or second leading-coefficient bank is used.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from exp.tail_alias.kernel import *
from exp.alternating_gauge.even_fold import absorb_even
from exp.alternating_gauge.scalar import prefix_reverse141,divide_zero117,multiply_zero144

def add_tail_a(b,a,q,d):
 ar.linear_inplace(b,q,ar.square_columns(12));ar.mul_xor0(b,a,q,d)
 ar.linear_inplace(b,q,[power(1<<j,2048)for j in range(12)])

def odd_encode(b,heads,i,r,d=None):
 a,x=heads[i];p=heads[i-2][1]if i>1 else heads[0][1]
 def after():
  add_tail_a(b,a,r,d);ar.mul_xor0(b,x,r,d)
 return pg.forward(b,x,p,r,after_product=after if d is not None else None)

def odd_decode(b,heads,i,r,m,d=None):
 a,x=heads[i];p=heads[i-2][1]if i>1 else heads[0][1]
 if d is not None:ar.mul_xor0(b,x,r,d)
 prefix_reverse141(b,x,p,r,m)
 if d is not None:add_tail_a(b,a,r,d)

def even_step(b,heads,i,r,d=None,reverse=False):
 a,x=heads[i];q=heads[i-1][1]
 if d is not None and reverse:
  add_tail_a(b,a,q,d);ar.mul_xor0(b,r,q,d)
 ar.mul_xor0(b,x,q,r)
 if d is not None and not reverse:
  add_tail_a(b,a,q,d);ar.mul_xor0(b,r,q,d)

def synthesize(b,words):
 h=len(words)//2;assert h>=3 and h%2==1;cs,ks,_,_=old.constants(h)
 heads=[words[2*i:2*i+2]for i in range(h)];d=heads[0][0];border=heads[0][1]
 r=b.alloc_qubits(12);b.cx(border,r);b.cx(border,d);b.cx(r,border);b.x(border[0]);masks=[None]*h
 for i in range(1,h):
  if i%2:masks[i]=odd_encode(b,heads,i,r,d)
  else:even_step(b,heads,i,r,d)
 divide_zero117(b,d,r);A=[d];B=[];hist=[]
 for i in reversed(range(1,h)):
  if i%2:
   odd_decode(b,heads,i,r,masks[i]);qi=r;p=heads[i-2][1]if i>1 else border
  else:
   qi=heads[i-1][1];p=r
  aa,bb=A,B
  if i%2:A,B,token=absorb(b,heads[i],A,B,qi,p,None,0,cs[i],ks[i],wide=False)
  else:A,B,token=absorb_even(b,heads[i],A,B,qi,r,cs[i],ks[i])
  hist.append((i,aa,bb,qi,p,cs[i],ks[i],token))
 b.x(border[0]);b.free(border);C=B+[r];ot=outer(b,C,A,cs[0],ks[0],wide=True)
 return C,A,dict(h=h,heads=heads,d=d,r=r,border=border,history=hist,A=C,B=A,outer=ot)

def unsynthesize(b,t):
 h=t['h'];heads=t['heads'];r=t['r'];cs,ks,_,_=old.constants(h)
 outer(b,t['A'],t['B'],cs[0],ks[0],True,t['outer'],True)
 reclaim(b,[t['border']]);b.x(t['border'][0]);masks=[None]*h
 for i,A,B,qi,p,c,kap,token in reversed(t['history']):
  if i%2:
   absorb(b,heads[i],A,B,qi,p,None,0,c,kap,True,token,False);masks[i]=odd_encode(b,heads,i,r)
  else:absorb_even(b,heads[i],A,B,qi,r,c,kap,True,token)
 multiply_zero144(b,t['d'],r)
 for i in reversed(range(1,h)):
  if i%2:odd_decode(b,heads,i,r,masks[i],t['d'])
  else:even_step(b,heads,i,r,t['d'],True)
 b.cx(r,t['d']);b.x(t['border'][0]);b.cx(r,t['border']);b.cx(t['border'],r);b.free(r)
