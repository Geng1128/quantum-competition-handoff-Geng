"""Inverse GF4096 phase with twelve private qubits, 39 Toffolis.

Only imports core modules; this plugin never patches or edits them.
"""
try:
    from .kernels import *
except ImportError:
    from kernels import *
from lowwidth import classical_transpose


def norm_bit_xor(b,x,z,K,bit=0):
    """XOR one GF8 norm coordinate into arbitrary single-qubit z, 3T."""
    assert len(x)==6
    a,c=x[:3],x[3:];f=field(3)
    for j,v in enumerate(square_columns(3)):
        if v>>bit&1:b.cx(a[j],z)
        if f.mul(K,v)>>bit&1:b.cx(c[j],z)
    for mask,vector in tensor(3):
        if not(vector>>bit&1):continue
        ix=list(bits(mask));p=ix[0]
        for j in ix[1:]:b.cx(a[j],a[p]);b.cx(c[j],c[p])
        b.ccx(a[p],c[p],z)
        for j in reversed(ix[1:]):b.cx(c[j],c[p]);b.cx(a[j],a[p])


def phase_inverse39(b,x,mask):
    """Apply (-1)^dot(mask,inverse(x)); inv(0)=0, x unchanged.

    E[0] is HMR-erased and its quadratic norm phase repaid before the
    mask-parity qubit is allocated. Once that qubit is freed, E[0] is
    reconstructed with 3 CCX, enabling the ordinary inverse erasure.
    Peak D6 + E2 + w3 + parity1 = 12 private quantum wires.
    """
    assert len(x)==len(mask)==12
    to12,frm12,K12=tower(12);linear_inplace(b,x,to12)
    mt=classical_transpose(b,mask,frm12)
    D=b.alloc_qubits(6);ar._norm_xor(b,x,D,K12)
    to6,frm6,K6=tower(6);linear_inplace(b,D,to6)
    E=b.alloc_qubits(3);ar._norm_xor(b,D,E,K6)
    w=inverse3_noscratch(b,E)
    em=b.alloc_bits(3);b.bit_store(em,False);b.hmr(E[:1],em[:1])
    ar._phase_norm(b,D,em,K6);b.free(em);b.free(E[:1])
    anc=b.alloc_qubits(1)[0];f6=field(6)
    def coefficient(h):
        for j in range(6):
            for r in bits(f6.mul(h,1<<j)):
                b.ccx(mt[r],x[j],anc)
                b.ccx(mt[r],x[j+6],anc)
                b.ccx(mt[r+6],x[j+6],anc)
    for segment in (0,1):
        if segment==0:b.cx(D[3:],D[:3])
        controls=D[:3] if segment==0 else D[3:]
        for m,v in tensor(3):
            h=apply(frm6,v<<(3*segment));ix=list(bits(m));p=ix[0]
            coefficient(h)
            for j in ix[1:]:b.cx(controls[j],controls[p]);b.cx(w[j],w[p])
            b.ccz(controls[p],w[p],anc)
            for j in reversed(ix[1:]):b.cx(w[j],w[p]);b.cx(controls[j],controls[p])
            coefficient(h)
        if segment==0:b.cx(D[3:],D[:3])
    b.free(anc)
    e0=b.alloc_qubits(1);norm_bit_xor(b,D,e0[0],K6)
    E=[e0[0],E[1],E[2]]
    erase_inverse(b,E,w);b.free(w)
    ar._erase_norm(b,D,E,K6);b.free(E);linear_inplace(b,D,frm6)
    ar._erase_norm(b,x,D,K12);b.free(D)
    linear_inplace(b,x,frm12);b.free(mt)
