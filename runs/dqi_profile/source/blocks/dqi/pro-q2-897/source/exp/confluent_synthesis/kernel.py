"""Read-only constants adapter for frozen alternating-gauge synthesis."""
from pathlib import Path
import sys
from functools import lru_cache
from types import FunctionType,SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from exp.alternating_gauge import kernel as base
from exp.decoder.common import mul,K,pmul
@lru_cache(None)
def constants(h=35):
 assert h>=3 and h%2==1
 cs=[0]+[2*j for j in range(1,(h+1)//2)for _ in range(2)]
 ks=[mul(c,c)^c^K for c in cs];P=[1]
 for k in ks:P=pmul(P,[k,1,1])
 return cs,ks,P,()
old=SimpleNamespace(**dict(vars(base.old),constants=constants))
synthesize=FunctionType(base.synthesize.__code__,dict(base.synthesize.__globals__,old=old),'synthesize',base.synthesize.__defaults__,base.synthesize.__closure__)
unsynthesize=FunctionType(base.unsynthesize.__code__,dict(base.unsynthesize.__globals__,old=old),'unsynthesize',base.unsynthesize.__defaults__,base.unsynthesize.__closure__)
