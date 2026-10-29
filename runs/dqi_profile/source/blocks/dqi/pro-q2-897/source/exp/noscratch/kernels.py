"""Scratch-free rankone bilinear accumulation; no coremodule edits."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'exp/arithmetic'))
import field as ar
from field import *

def rankone_xor(b,x,y,z,vector):
    """z^=vector*(xANDy), arbitraryz, zeroancilla, oneCCX.

    x,y arebits; vector is nonzero bitmaskoverz. Inputsdisjointfromtarget.
    """
    out=list(bits(vector));p=out[0]
    for j in out[1:]:b.cx(z[p],z[j])
    b.ccx(x,y,z[p])
    for j in reversed(out[1:]):b.cx(z[p],z[j])

def tensor_xor_noscratch(b,x,y,z,terms):
    assert len(x)==len(y)==len(z)
    assert not(set(map(str,x))&set(map(str,y))or set(map(str,x))&set(map(str,z))or set(map(str,y))&set(map(str,z)))
    for mask,vector in terms:
        ix=list(bits(mask));p=ix[0]
        for j in ix[1:]:b.cx(x[j],x[p]);b.cx(y[j],y[p])
        rankone_xor(b,x[p],y[p],z,vector)
        for j in reversed(ix[1:]):b.cx(y[j],y[p]);b.cx(x[j],x[p])

def mul_xor_noscratch(b,x,y,z):
    """DisjointGF2^n productaccumulate; n2/3/6/12, rank3/6/15/42,0scratch."""
    tensor_xor_noscratch(b,x,y,z,tensor(len(x)))

def mul24_xor_noscratch(b,x,y,z,K=None):
    """ExplicitquadraticGF24 productaccumulate, arbitrarytarget,105T/0scratch."""
    assert len(x)==24;tensor_xor_noscratch(b,x,y,z,ar.tensor24_extension(K))

def inverse3_xor_noscratch(b,x,z):
    """z^=inverseGF8(x), inverse0=0, arbitrarytarget,3T/0scratch."""
    assert len(x)==len(z)==3 and not(set(map(str,x))&set(map(str,z)))
    for mask,vector in enumerate(inverse3_anf()):
        if not vector:continue
        ix=list(bits(mask))
        if len(ix)==1:
            for j in bits(vector):b.cx(x[ix[0]],z[j])
        elif len(ix)==2:rankone_xor(b,x[ix[0]],x[ix[1]],z,vector)
        else:raise ValueError((mask,vector))

def inverse3_noscratch(b,x,target=None):
    """Freshcleantargetinverse, or XORintogiventarget,3T/0privatequbits."""
    if target is None:target=b.alloc_qubits(3)
    inverse3_xor_noscratch(b,x,target);return target
