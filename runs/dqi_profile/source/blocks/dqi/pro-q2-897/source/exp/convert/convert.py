"""Reversible fixed-generic-chart half-degree CONVERT.

Public convert(builder,A[35] words,B[35] words,reverse=False).
A,B hold GF12 polynomial coefficients; B35=1 implicit.
Safe divisors make roundtrip exact even on degree gaps; forward decode is only
proved for generic Euclid degrees. Runtime gap2 routing is not yet implemented.
"""
from contextlib import contextmanager
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from kernels import ar,cx,convolution_xor,hankel
from lowwidth import phase_inverse36

def zero_flag(b,a):
    for q in a:b.x(q)
    nodes=[];left=a[0]
    for q in a[1:]:
        out=b.alloc_qubits(1)[0];b.init_and(left,q,out);nodes.append((left,q,out));left=out
    for x,y,z in reversed(nodes[:-1]):b.del_and(x,y,z);b.free(z)
    for q in a:b.x(q)
    return nodes[-1][2]

def erase_zero_flag(b,a,flag):
    for q in a:b.x(q)
    nodes=[];left=a[0]
    for q in a[1:-1]:
        out=b.alloc_qubits(1)[0];b.init_and(left,q,out);nodes.append((left,q,out));left=out
    b.del_and(left,a[-1],flag);b.free(flag)
    for x,y,z in reversed(nodes):b.del_and(x,y,z);b.free(z)
    for q in a:b.x(q)

@contextmanager
def inverse_context(b,a):
    flag=zero_flag(b,a);b.cx(flag,a[0])
    ai=ar.inverse(b,a)
    yield ai
    mask=b.alloc_bits(12);b.hmr(ai,mask);b.free(ai)
    phase_inverse36(b,a,mask);b.free(mask)
    b.cx(flag,a[0]);erase_zero_flag(b,a,flag)

def scale(b,x,a,ai,divide=True):
    out=b.alloc_qubits(12);ar.mul_xor(b,x,ai if divide else a,out)
    ar.erase_product(b,a if divide else ai,out,x)
    b.swap(x,out);b.free(out)

def runops(b,ops,a,ai,undo=False):
    for op in (reversed(ops) if undo else ops):
        if op[0]=='s':scale(b,op[1],a,ai,op[2]^undo)
        elif op[0]=='m':ar.mul_xor(b,op[1],op[2],op[3])
        elif op[0]=='c':convolution_xor(b,op[1],op[2],op[3],undo)
        else:raise ValueError(op[0])

def pre(b,f,g,payloads,undo=False):
    m=len(f)-1;n=m-1;a=g[-1];q0=f[-2]
    with inverse_context(b,a) as ai:
        q1=ai if f[-1] is None else f[-1]
        ops=[('s',v[-1],True) for v in payloads]
        ops += [('m',g[i],v[-1],v[i]) for v in payloads for i in range(n)]
        if f[-1] is not None:ops += [('s',q1,True)]
        ops += [('m',q1,g[-2],q0),('s',q0,True),('c',g[:-1],[q0,q1],f[:-2])]
        runops(b,ops,a,ai,undo)

def post(b,f,g,payloads,undo=False):
    n=len(g)-1;a=g[-1];q0=f[-2]
    def dots():
        for v in payloads:
            for i in range(n):ar.mul_xor(b,g[i],v[i],v[-1])
    if f[-1] is None:
        with inverse_context(b,a) as ai:
            q1=ai
            ops=[('s',v[-1],False) for v in payloads]
            ops += [('m',g[i],v[i],v[-1]) for v in payloads for i in range(n)]
            ops += [('s',v[-1],True) for v in payloads]
            ops += [('c',g[:-1],[q0,q1],f[:-2]),('s',q0,False),('m',q1,g[-2],q0)]
            runops(b,ops,a,ai,undo)
    else:
        q1=f[-1]
        def qscale(reverse):
            with inverse_context(b,q1) as qi:
                for v in payloads:scale(b,v[-1],q1,qi,not reverse)
        def last(reverse):
            with inverse_context(b,a) as ai:
                ops=[('s',v[-1],True) for v in payloads]
                ops += [('c',g[:-1],[q0,q1],f[:-2]),('s',q0,False),('m',q1,g[-2],q0),('s',q1,False)]
                runops(b,ops,a,ai,reverse)
        if not undo:qscale(False);dots();last(False)
        else:last(True);dots();qscale(True)

def raw_solve(b,f,g,payloads,reverse=False):
    m=len(f)-1
    assert len(g)==m and all(len(v)==m for v in payloads)
    if m==1:
        divisors=[p for p in (f[-1],g[-1]) if p is not None]
        for a in (reversed(divisors) if reverse else divisors):
            with inverse_context(b,a) as ai:
                for v in payloads:scale(b,v[0],a,ai,not reverse)
        return
    low=[v[:-1] for v in payloads]
    if not reverse:
        pre(b,f,g,payloads);raw_solve(b,g,f[:-2],low);post(b,f,g,payloads)
    else:
        post(b,f,g,payloads,True);raw_solve(b,g,f[:-2],low,True);pre(b,f,g,payloads,True)

def bez_solve(b,O,E,payloads,reverse=False):
    def reduce():
        for i in range(len(O)):ar.mul_xor(b,E[-1],O[i],E[i])
    reduce();raw_solve(b,O+[None],E[:-1],payloads,reverse);reduce()

def frob(b,words,power):
    f=ar.field(12);cols=[f.pow(1<<i,power) for i in range(12)]
    for x in words:ar.linear_inplace(b,x,cols)

def convert(b,A,B,reverse=False):
    assert len(A)==len(B)==35
    E=B[::2];O=B[1::2];u=A[1::2];w=A[:34:2];top=A[-1]
    if reverse:hankel(b,B,A,True)
    frob(b,B,2048);frob(b,A,2048)
    if reverse:
        for i in range(17):b.swap(A[2*i],A[2*i+1])
        for i in range(17):ar.mul_xor(b,O[i],u[i],top)
        bez_solve(b,O,E,[u,w],True)
        for i in range(17):ar.mul_xor(b,top,O[i],w[i])
    else:
        for i in range(17):ar.mul_xor(b,top,O[i],w[i])
        bez_solve(b,O,E,[u,w])
        for i in range(17):ar.mul_xor(b,O[i],u[i],top)
        for i in range(17):b.swap(A[2*i],A[2*i+1])
    frob(b,A,2);frob(b,B,2)
    if not reverse:hankel(b,B,A)

def raw_solve_blocked(b,f,g,payloads,block_size=2,reverse=False):
    """Closed checkpointed scalar schedule grouped into quotient blocks.

    This is the no-speedup control construction: at most2*block_size quotient
    metadata words remain in already consumed f/g coefficients; no matrix or
    evaluation bank is materialized. Product fusion is deliberately absent.
    """
    frames=[]
    for _ in range(block_size):
        if len(f)==2:break
        frames.append((f,g,payloads))
        if not reverse:pre(b,f,g,payloads)
        else:post(b,f,g,payloads,True)
        m=len(f)-1;f,g=g,f[:-2];payloads=[v[:-1]for v in payloads]
    if len(f)==2:raw_solve(b,f,g,payloads,reverse)
    else:raw_solve_blocked(b,f,g,payloads,block_size,reverse)
    for f,g,payloads in reversed(frames):
        if not reverse:post(b,f,g,payloads)
        else:pre(b,f,g,payloads,True)
