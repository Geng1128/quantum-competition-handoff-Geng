"""Inverse12 phase with12 scratch and at most36 EXECUTED Toffolis.

The coefficient mask is classical. A classical first-nonzero pivot exposes
its parity in an existing input bit, avoiding the13th quantum wire.
"""
try:
    from exp.mul99.kernels import *
except ImportError:
    from exp.mul99.kernels import *
sys.path.insert(0,str(ROOT))
from exp.noscratch.kernels import inverse3_noscratch
from lowwidth import classical_transpose


def phase_classical_parity(b,a,c,x,co):
    """CCZ(a,c,parity(co*x)), zero quantum scratch, <=1 executed CCZ."""
    first=b.alloc_bits(len(x));b.bit_store(first,False)
    seen=b.alloc_bits(1)[0];b.bit_store(seen,False)
    for j in range(len(x)):
        b.x(seen);b.push_condition(seen)
        b.bit_store(first[j],True,control=co[j])
        b.pop_condition();b.x(seen)
        b.bit_store(seen,True,control=co[j])
    for j in range(len(x)):
        b.push_condition(first[j])
        for k in range(len(x)):
            if k!=j:b.ccx(co[k],x[k],x[j])
        b.ccz(a,c,x[j])
        for k in reversed(range(len(x))):
            if k!=j:b.ccx(co[k],x[k],x[j])
        b.pop_condition()
    b.free(seen);b.free(first)


def phase_inverse36_zero(b,x,mask,zero_mask):
    """Phase mask dot inverse0(x): <=36 mean T /12 quantum scratch.

    Static count168T due to12 mutually exclusive classical branches per CCZ.
    No quantum input controls branch selection; all branches use HMR-mask bits.
    """
    assert len(x)==len(mask)==12
    to12,frm12,K12=tower(12);linear_inplace(b,x,to12)
    mt=classical_transpose(b,mask,frm12)
    D=b.alloc_qubits(6);ar._norm_xor(b,x,D,K12)
    to6,frm6,K6=tower(6);linear_inplace(b,D,to6)
    E=b.alloc_qubits(3);ar._norm_xor(b,D,E,K6);w=inverse3_noscratch(b,E)
    b.z(zero_mask);zm=b.alloc_bits(3);b.bit_store(zm,False);b.cx(zero_mask,zm[0])
    phase_product(b,E,w,zm);b.free(zm)
    f6=field(6)
    for segment in(0,1):
        if segment==0:b.cx(D[3:],D[:3])
        controls=D[:3] if segment==0 else D[3:]
        for m,v in tensor(3):
            h=apply(frm6,v<<(3*segment));co=b.alloc_bits(12);b.bit_store(co,False)
            for j in range(6):
                for r in bits(f6.mul(h,1<<j)):
                    b.cx(mt[r],co[j]);b.cx(mt[r],co[j+6]);b.cx(mt[r+6],co[j+6])
            ix=list(bits(m));p=ix[0]
            for j in ix[1:]:b.cx(controls[j],controls[p]);b.cx(w[j],w[p])
            phase_classical_parity(b,controls[p],w[p],x,co)
            for j in reversed(ix[1:]):b.cx(w[j],w[p]);b.cx(controls[j],controls[p])
            b.free(co)
        if segment==0:b.cx(D[3:],D[:3])
    erase_inverse(b,E,w);b.free(w)
    ar._erase_norm(b,D,E,K6);b.free(E);linear_inplace(b,D,frm6)
    ar._erase_norm(b,x,D,K12);b.free(D);linear_inplace(b,x,frm12);b.free(mt)
