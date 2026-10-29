"""Reuse Euclid gap predicates with a sparse quantum flag bank.

KEEP_FULL holds nine flags through the diagonal middle stages. KEEP_CHILD
also holds level3 through its child recursion; that flag is measured at
forward return and reconstructed with its matching phase at reverse entry.
Other levels stream masks between identical adjacent predicates. Every
final flag measurement is repaid by its exact zero-predicate phase.
"""
from pathlib import Path
import sys
from types import SimpleNamespace
from contextlib import contextmanager
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from exp.convert_short_hankel import phase34 as base
from types import FunctionType
from exp.fold_cached_norm.phase34 import phase_inverse34
from exp.decoder.common import ar

def clone(fn,scope):
    out=FunctionType(fn.__code__,scope,fn.__name__,fn.__defaults__,fn.__closure__)
    out.__kwdefaults__=fn.__kwdefaults__
    return out

def clear_flag(b,a,F):
    m=b.alloc_bits(1);b.hmr(F,m[0]);b.free(F)
    b.push_condition(m[0]);b.x(a)
    nodes=b.alloc_qubits(len(a)-2)
    b.init_and(a[0],a[1],nodes[0])
    for i in range(1,len(nodes)):b.init_and(nodes[i-1],a[i+1],nodes[i])
    b.ccz(nodes[-1],a[-1],m[0])
    for i in reversed(range(len(nodes))):
        if i:b.del_and(nodes[i-1],a[i+1],nodes[i])
        else:b.del_and(a[0],a[1],nodes[0])
    b.free(nodes);b.x(a);b.pop_condition();b.free(m)

def hold_flag(b,F):
    m=b.alloc_bits(1);b.hmr(F,m[0]);b.free(F);return m

def consume_flag(b,F,m):
    b.cz(F,m[0]);b.free(m)

KEEP_FULL=frozenset((2,4,6,8,10,12,14,16,17))
KEEP_CHILD=KEEP_FULL|{3}


def q_post(b,q,payloads,H,m,parent,reverse=False,phase=None):
    if parent is None:
        with q_inverse(b,q,m,reverse,phase) as qi:
            for v in payloads:scale(b,v[-1],q,qi,not reverse)
        return
    tops,p,c=parent
    if reverse or m+1 in KEEP_CHILD:F=phase['given_gap'].pop(m+1)
    else:
        F=fresh_zero_flag(b,q)
        consume_flag(b,F,phase['incoming_gap'].pop(m+1))
    b.cswap(F,q,p)
    if not reverse:
        for v,t in zip(payloads,tops):b.cswap(F,v[-1],t)
    with q_inverse(b,q,m,reverse,phase) as qi:
        if not reverse:
            for v in payloads:scale(b,v[-1],q,qi,True)
            for v,t in zip(payloads,tops):masked_product(b,F,v[-1],c,t)
        else:
            for v,t in zip(payloads,tops):masked_product(b,F,v[-1],c,t)
            for v in payloads:scale(b,v[-1],q,qi,False)
    if reverse:
        for v,t in zip(payloads,tops):b.cswap(F,v[-1],t)
    b.cswap(F,q,p)
    if reverse and m+1 not in KEEP_CHILD:phase['return_mask'][m+1]=hold_flag(b,F)
    else:phase['returned_gap'][m+1]=F


def look_solve(b,f,g,payloads,H=None,parent=None,reverse=False,phase=None):
    m=len(f)-1
    if m==1:return base_case(b,f,g,payloads,H,parent,reverse,phase)
    keep=m in KEEP_CHILD;full=m in KEEP_FULL
    F=phase['held_gap'].pop(m) if reverse and full else fresh_zero_flag(b,g[-1])
    if reverse and not full:consume_flag(b,F,phase['gap_flags'].pop(m))
    rotate(b,g,F)
    if not reverse:carry=pre(b,f,g,payloads,defer=True)
    else:carry=post_look(b,f,g,payloads,H,parent,True,incoming=phase['a'].pop(m),defer=True,phase=phase)
    canonical_permute(b,f,g,F)
    if reverse or keep:phase['given_gap'][m]=F
    else:phase['incoming_gap'][m]=hold_flag(b,F)
    childparent=([v[-1]for v in payloads],f[-1],f[-2])
    look_solve(b,g,f[:-2],[v[:-1]for v in payloads],None,childparent,reverse,phase)
    if reverse and not keep:
        F=fresh_zero_flag(b,g[-1]);consume_flag(b,F,phase['return_mask'].pop(m))
    else:F=phase['returned_gap'].pop(m)
    canonical_permute(b,f,g,F,True)
    if not reverse:phase['a'][m]=post_look(b,f,g,payloads,H,parent,incoming=carry,defer=True,phase=phase)
    else:pre(b,f,g,payloads,True,incoming=carry)
    rotate(b,g,F,True)
    if reverse:clear_flag(b,g[-1],F)
    elif full:phase['held_gap'][m]=F
    else:phase['gap_flags'][m]=hold_flag(b,F)


def erase_product_zero(b, a, ai, F):
    mask = b.alloc_bits(1)
    b.hmr(F, mask[0]); b.free(F)
    b.z(mask[0])
    word = b.alloc_bits(12)
    b.bit_store(word, False); b.cx(mask[0], word[0])
    ar.phase_product(b, a, ai, word)
    b.free(word); b.free(mask)


def cached_root_action(b, f, g, payloads, ai, Z, post, undo):
    a = g[-1]; q0, q1 = f[-2:]
    n = len(g)-1
    if post:
        operations = [('s', v[-1], False) for v in payloads]
        operations += [('m', g[i], v[i], v[-1]) for v in payloads for i in range(n)]
        operations += [('s', v[-1], True) for v in payloads]
        operations += [('c', g[:-1], [q0, q1], f[:-2]),
                       ('s', q0, False), ('m', q1, g[-2], q0)]
    else:
        operations = [('s', v[-1], True) for v in payloads]
        operations += [('m', g[i], v[-1], v[i]) for v in payloads for i in range(n)]
        operations += [('m', q1, g[-2], q0), ('s', q0, True),
                       ('c', g[:-1], [q0, q1], f[:-2])]
    b.cx(Z, a[0])
    old.runops(b, operations, a, ai, undo)
    b.cx(Z, a[0])


def look_root_cached(b, O, g, payloads, reverse, phase):
    m = len(O); keep = m in KEEP_CHILD; full = m in KEEP_FULL
    F = phase['held_gap'].pop(m) if reverse and full else old.fresh_zero_flag(b, g[-1])
    if reverse and not full:
        consume_flag(b, F, phase['gap_flags'].pop(m))
    old.rotate(b, g, F)
    a = g[-1]
    ai, Z = old.inverse_zero(b, a)
    b.cx(Z, ai[0])
    if reverse:
        mask = phase.pop('_root_inverse_mask')
        b.cz(ai, mask); b.free(mask)
    f = list(O) + [ai]
    cached_root_action(b, f, g, payloads, ai, Z, post=reverse, undo=reverse)
    erase_product_zero(b, a, ai, Z)
    old.canonical_permute(b, f, g, F)
    if reverse or keep:
        phase['given_gap'][m] = F
    else:
        phase['incoming_gap'][m] = hold_flag(b, F)
    parent = ([v[-1] for v in payloads], f[-1], f[-2])
    old.look_solve(b, g, f[:-2], [v[:-1] for v in payloads], None,
                   parent, reverse, phase)
    if reverse and not keep:
        F = old.fresh_zero_flag(b, g[-1])
        consume_flag(b, F, phase['return_mask'].pop(m))
    else:
        F = phase['returned_gap'].pop(m)
    old.canonical_permute(b, f, g, F, True)
    Z = old.fresh_zero_flag(b, a)
    cached_root_action(b, f, g, payloads, ai, Z, post=not reverse, undo=reverse)
    erase_product_zero(b, a, ai, Z)
    mask = b.alloc_bits(12)
    b.hmr(ai, mask); b.free(ai)
    if reverse:
        phase_inverse34(b, a, mask, zero_mask=mask[0]); b.free(mask)
    else:
        phase['_root_inverse_mask'] = mask
    old.rotate(b, g, F, True)
    if reverse:
        clear_flag(b, g[-1], F)
    elif full:
        phase['held_gap'][m] = F
    else:
        phase['gap_flags'][m] = hold_flag(b, F)


def _bez_monic(b, O, E, payloads, reverse, history_scratch12, phase):
    def reduce():
        for i in range(len(O)):
            ar.mul_xor0(b, E[-1], O[i], E[i])
    reduce()
    look_root_cached(b, O, E[:-1], payloads, reverse, phase)
    reduce()



scope = dict(vars(base._private), KEEP_CHILD=KEEP_CHILD, KEEP_FULL=KEEP_FULL, hold_flag=hold_flag,
             consume_flag=consume_flag, clear_flag=clear_flag)
for name in ['scale', 'runops', 'zero_toggle', 'inverse_apply', 'pre',
             'canonical_permute', 'masked_product', 'fresh_zero_flag',
             'post_look', 'base', 'bez_lookahead']:
    scope[name] = clone(getattr(base._private, name), scope)
scope['base_case'] = scope['base']
scope['q_inverse'] = contextmanager(clone(base._private.q_inverse.__wrapped__, scope))
scope['q_post'] = clone(q_post, scope)
scope['look_solve'] = clone(look_solve, scope)
short = dict(base._short_scope, KEEP_CHILD=KEEP_CHILD, KEEP_FULL=KEEP_FULL, old=SimpleNamespace(**scope),
             hold_flag=hold_flag, consume_flag=consume_flag, clear_flag=clear_flag,
             ar=ar, erase_product_zero=erase_product_zero,
             phase_inverse34=phase_inverse34)
for name, fn in [('cached_root_action', cached_root_action),
                 ('look_root_cached', look_root_cached), ('_bez_monic', _bez_monic)]:
    short[name] = clone(fn, short)
_core_convert = clone(base._short_scope['convert_safe'], short)


def convert_safe(b, A, B, reverse=False, history_scratch12=None,
                 phase_token=None, *, monic_root=False):
    keys = ['gap_flags', 'given_gap', 'incoming_gap', 'return_mask',
            'returned_gap', 'held_gap']
    if phase_token is None:
        phase_token = {'a': {}, 'q': {}, '_short_monic': bool(monic_root)}
        for key in keys:
            phase_token[key] = {}
    result = _core_convert(b, A, B, reverse, history_scratch12, phase_token,
                          monic_root=monic_root)
    if reverse:
        assert all(not phase_token[key] for key in keys)
        assert '_root_inverse_mask' not in phase_token
    return result
