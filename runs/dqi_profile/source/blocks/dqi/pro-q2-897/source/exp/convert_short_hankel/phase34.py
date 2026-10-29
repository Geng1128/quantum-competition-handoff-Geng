"""Private34T phase substitution; never mutates stable module globals.

Same API as kernel.convert_safe. Saving2T per ultimate inverse closure;
phase helper21private fits below the existing39-private converter peak.
"""
from types import FunctionType,SimpleNamespace
from contextlib import contextmanager
from exp.safe_scalar import converter as stable
from exp.fold_cached_norm.phase34 import phase_inverse34
from exp.convert_short_hankel import kernel as short


def clone(fn,scope,name=None):
 out=FunctionType(fn.__code__,scope,name or fn.__name__,fn.__defaults__,fn.__closure__)
 out.__kwdefaults__=None if fn.__kwdefaults__ is None else dict(fn.__kwdefaults__)
 return out

# Erase/inverse_apply_safe are imported closures with different globals;
# clone the chain explicitly before rebinding the converter namespace.
_erase=clone(stable.erase,dict(stable.erase.__globals__,phase_inverse36_zero=phase_inverse34))
_apply=clone(stable.inverse_apply_safe,dict(stable.inverse_apply_safe.__globals__,erase=_erase))
_scope=dict(vars(stable),erase=_erase,inverse_apply_safe=_apply)
_qgen=clone(stable.q_inverse.__wrapped__,_scope,'q_inverse')
_scope['q_inverse']=contextmanager(_qgen)
for _name in ['scale','runops','zero_toggle','inverse_apply','pre','canonical_permute','masked_product','fresh_zero_flag','clear_flag','q_post','post_look','base','look_solve','bez_lookahead']:
 _scope[_name]=clone(getattr(stable,_name),_scope,_name)
_private=SimpleNamespace(**_scope)

# Clone short-H and optional monic-root functions against that private graph.
_short_scope=dict(vars(short),old=_private)
for _name in ['_root_ops','_pre_root','_post_root','_look_root','_bez_monic','convert_safe']:
 _short_scope[_name]=clone(getattr(short,_name),_short_scope,_name)
convert_safe=_short_scope['convert_safe']
