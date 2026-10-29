"""Exact45045-Toffoli Chien/residue phase scan,11workspacequbits.

A,B:35GF12 words each, degree<35 numerator and monicdegree35 denominator.
E:4095classical12-bit masks, indexed beta=alpha**j. Bothbanks and E restored.
A is already residue-converted; phase is dot(Ej,A(beta)/(beta*P(beta)))
when B(beta)=0, and0otherwise. P is the node polynomial from extract.constants.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'exp/decoder'))
from common import *
from extract import constants


def scan(b,A,B,E,*,points=4095,skip_points=(),node_polynomial=None):
    if len(A)!=35 or len(B)!=35 or len(E)<points:raise ValueError('need35+35words and one maskperpoint')
    if points!=4095:raise ValueError('fullcycle4095points required for bank restoration')
    alpha=gf.primitive_element;P=constants()[2]if node_polynomial is None else node_polynomial
    powers=[power(alpha,k) for k in range(35)]
    skip_points=frozenset(skip_points)
    assert all(0<=j<4095 for j in skip_points)
    beta=1;lead=1;leadstep=power(alpha,35);m=b.alloc_bits(12)
    for j in range(4095):
        # The original coefficientk now holds Ak*beta**k (likewiseB).
        if j not in skip_points:
            for k in range(1,35):b.cx(A[k],A[0]);b.cx(B[k],B[0])
            b.gf2_iadd(lead,target=B[0])
            ladder=zero_hold(b,B[0]);flag=ladder[-1]
            pv=0
            for v in reversed(P):pv=mul(pv,beta)^v
            c=inv(mul(beta,pv));assert c
            # Classically apply transpose of multiplicationc to maskEj.
            b.bit_store(m,False)
            for t in range(12):
                for i in ar.bits(mul(c,1<<t)):b.cx(E[j][i],m[t])
            b.ccz(flag,A[0],m)
            zero_free(b,B[0],ladder)
            b.gf2_iadd(lead,target=B[0])
            for k in range(1,35):b.cx(A[k],A[0]);b.cx(B[k],B[0])
        for k in range(1,35):scale(b,A[k],powers[k]);scale(b,B[k],powers[k])
        beta=mul(beta,alpha);lead=mul(lead,leadstep)
    assert beta==lead==1
    b.free(m)
