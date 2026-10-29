"""Closed local q-prefix step with the historical swap zero chart.

Inputs (b,p,q) -> (q*b+p,p,q) when q!=0, or (p,b,q) whenq=0.
Forward65T, reverse152T using divide117; pair217T,29private peak.
The reverse is exact for all (b,p,q), not only adjacent-nonzero continuants.
"""
try:
    from .scalar import *
except ImportError:
    from scalar import *

def zero_hold(b,x):
    b.x(x);lad=b.alloc_qubits(len(x)-1);b.init_and(x[0],x[1],lad[0])
    for j in range(2,len(x)):b.init_and(lad[j-2],x[j],lad[j-1])
    b.x(x);return lad

def zero_free(b,x,lad):
    b.x(x)
    for j in reversed(range(2,len(x))):b.del_and(lad[j-2],x[j],lad[j-1])
    b.del_and(x[0],x[1],lad[0]);b.free(lad);b.x(x)

def forward(b,x,p,q,virtual_zero=None,after_product=None):
    out=b.alloc_qubits(12);mul_xor0(b,x,q,out);b.cx(p,out)
    if virtual_zero is not None:b.ccx(virtual_zero,p,out)
    lad=zero_hold(b,q);b.cswap(lad[-1],x,p)
    m=b.alloc_bits(12);b.hmr(x,m);b.swap(x,out);b.free(out)
    if after_product is not None:after_product()
    zero_free(b,q,lad);return m

def reverse(b,x,p,q,m):
    lad=zero_hold(b,q);f=lad[-1];b.cswap(f,x,p);b.cx(p,x);b.ccx(f,p,x)
    b.cx(f,q[0]);divide117(b,x,q);b.cx(f,q[0])
    b.cz(x,m)
    for j in range(12):b.ccz(f,x[j],m[j]);b.ccz(f,p[j],m[j])
    b.free(m);zero_free(b,q,lad)

def zero_compact(b,x):
    lad=zero_hold(b,x);b.x(x)
    for j in reversed(range(2,len(x)-1)):b.del_and(lad[j-2],x[j],lad[j-1]);b.free(lad[j-1])
    b.del_and(x[0],x[1],lad[0]);b.free(lad[0]);b.x(x)
    return lad[-1]

def zero_compact_free(b,x,f):
    b.x(x);lad=b.alloc_qubits(len(x)-2);b.init_and(x[0],x[1],lad[0])
    for j in range(2,len(x)-1):b.init_and(lad[j-2],x[j],lad[j-1])
    b.del_and(lad[-1],x[-1],f);b.free(f)
    for j in reversed(range(2,len(x)-1)):b.del_and(lad[j-2],x[j],lad[j-1])
    b.del_and(x[0],x[1],lad[0]);b.free(lad);b.x(x)

def reverse_compact(b,x,p,q,m,virtual_zero=None):
    """162T reverse,19private; paired with forward gives227T/23private."""
    f=zero_compact(b,q);b.cswap(f,x,p);b.cx(p,x);b.ccx(f,p,x)
    if virtual_zero is not None:b.ccx(virtual_zero,p,x)
    b.cx(f,q[0]);divide117(b,x,q);b.cx(f,q[0]);b.cz(x,m)
    for j in range(12):b.ccz(f,x[j],m[j]);b.ccz(f,p[j],m[j])
    b.free(m);zero_compact_free(b,q,f)

def boundary_forward(b,x,q,F,after_product=None):
    """i=1 special chart; F=zero(original q1),66T,12private."""
    b.cx(F,q[0]);out=b.alloc_qubits(12);mul_xor0(b,x,q,out)
    m=b.alloc_bits(12);b.hmr(x,m);b.swap(x,out);b.free(out);b.cx(F,q[0]);b.x(x[0])
    if after_product is not None:after_product()
    b.ccx(F,x,q);b.cx(F,q[0]);b.ccx(F,q,x)
    return m

def boundary_reverse(b,x,q,F,m):
    """129T,18private; exact inverse, closes boundary_forward's mask."""
    b.cswap(F,q,x);b.cx(F,q[0]);b.x(x[0]);b.cx(F,x[0])
    b.cx(F,q[0]);divide117(b,x,q);b.cx(F,q[0]);b.cz(x,m);b.free(m)
