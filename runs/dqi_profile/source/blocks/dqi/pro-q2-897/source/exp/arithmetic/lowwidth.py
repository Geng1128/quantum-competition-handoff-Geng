"""Streamed towerinversion and25-scratch preserved-factor division candidates."""
import field as ar
from field import *
from division import mul24_xor,phase_product24,multiply_forward


def classical_transpose(b,mask,columns):
    out=b.alloc_bits(len(columns));b.bit_store(out,False)
    for i,c in enumerate(columns):
        for j in bits(c):b.cx(mask[j],out[i])
    return out


def inverse_stream_forward(b,x):
    """Initialized in-place inverse with deferredoldinputphase, W(n)=n+1.

    T3=3,T6=21,T12=66,T24=192. Transcriptisintheoriginalpolynomialbasis.
    Allinternalnorm transcriptsarefixedquadratically beforeparentinputchanges.
    """
    n=len(x)
    if n==3:
        y=inverse(b,x);mask=b.alloc_bits(n);b.hmr(x,mask);b.swap(x,y);b.free(y);return mask
    k=n//2;to,frm,K=tower(n);linear_inplace(b,x,to)
    N=b.alloc_qubits(k);ar._norm_xor(b,x,N,K)
    nm=inverse_stream_forward(b,N);ar._phase_norm(b,x,nm,K);b.free(nm)
    outb=b.alloc_qubits(k);mul_xor(b,x[k:],N,outb)
    raw=b.alloc_bits(n);b.hmr(x[k:],raw[k:])
    mul_xor(b,x[:k],N,x[k:]);b.cx(outb,x[k:]);b.hmr(x[:k],raw[:k])
    b.swap(x[:k],x[k:]);b.swap(x[k:],outb);b.free(outb)
    ar._erase_norm(b,x,N,K);b.free(N)
    linear_inplace(b,x,frm)
    mask=classical_transpose(b,raw,to);b.free(raw);return mask


def phase_product_quantum(b,x,y,mask):
    """Phase dot(mask,x*y) with QUANTUMmask: rank(n)CCZ,0scratch."""
    n=len(x);assert len(y)==len(mask)==n
    for m,v in tensor(n):
        ix=list(bits(m));iz=list(bits(v));i=ix[0];j=iz[0]
        for k in ix[1:]:b.cx(x[k],x[i]);b.cx(y[k],y[i])
        for k in iz[1:]:b.cx(mask[k],mask[j])
        b.ccz(x[i],y[i],mask[j])
        for k in reversed(iz[1:]):b.cx(mask[k],mask[j])
        for k in reversed(ix[1:]):b.cx(y[k],y[i]);b.cx(x[k],x[i])


def phase_inverse_generic6(b,x,mask):
    """9T inversephase on6inputbits,7scratch; maskclassical6."""
    to,frm,K=tower(6);linear_inplace(b,x,to)
    mt=classical_transpose(b,mask,frm)
    N=b.alloc_qubits(3);ar._norm_xor(b,x,N,K);u=inverse(b,N)
    b.cx(x[3:],x[:3]);phase_product(b,x[:3],u,mt[:3]);phase_product(b,x[3:],u,mt[3:]);b.cx(x[3:],x[:3])
    erase_inverse(b,N,u);b.free(u);ar._erase_norm(b,x,N,K);b.free(N)
    linear_inplace(b,x,frm);b.free(mt)


def phase_inverse_low12(b,x,mask):
    """60T phaseofinverse12 using13scratch beyondinput12,maskclassical."""
    to,frm,K=tower(12);linear_inplace(b,x,to)
    mt=classical_transpose(b,mask,frm)
    N=b.alloc_qubits(6);ar._norm_xor(b,x,N,K)
    nm=inverse_stream_forward(b,N);ar._phase_norm(b,x,nm,K);b.free(nm)
    b.cx(x[6:],x[:6]);phase_product(b,x[:6],N,mt[:6]);phase_product(b,x[6:],N,mt[6:]);b.cx(x[6:],x[:6])
    um=b.alloc_bits(6);b.hmr(N,um);b.free(N)
    N=b.alloc_qubits(6);ar._norm_xor(b,x,N,K);phase_inverse_generic6(b,N,um);b.free(um)
    ar._erase_norm(b,x,N,K);b.free(N);linear_inplace(b,x,frm);b.free(mt)


def fix_scaled_half_masks(b,head,factor,masks,K):
    """Phase sum_j dot(maskj,headj*norm(factor)):42CCZ,12scratch.

    Thetwoheadhalves shareonequantum12-bittransposedmask, hence42not84CCZ.
    """
    L=b.alloc_qubits(12)
    def coefficient_mask():
        for i in range(12):
            for half in range(2):
                for j in range(12):
                    for r in bits(field(12).mul(1<<i,1<<j)):
                        b.ccx(masks[half][r],head[12*half+j],L[i])
    coefficient_mask()
    phase_linear(b,factor[:12],L,square_columns(12))
    phase_linear(b,factor[12:],L,[field(12).mul(K,v) for v in square_columns(12)])
    phase_product_quantum(b,factor[:12],factor[12:],L)
    coefficient_mask();b.free(L)


def divide_lowwidth(b,x,y,K=None):
    """441T division x/=y,preservingy,25workspace; promise x0wheny0."""
    if K is None:K=extension_columns()[2]
    b.cx(y[12:],y[:12]);oldmask=multiply_forward(b,x,y,K);b.cx(y[12:],y[:12])
    u=b.alloc_qubits(12);ar._norm_xor(b,y,u,K)
    nm=inverse_stream_forward(b,u);ar._phase_norm(b,y,nm,K);b.free(nm)
    masks=[]
    for offset in(0,12):
        half=x[offset:offset+12];out=b.alloc_qubits(12);mul_xor(b,half,u,out)
        m=b.alloc_bits(12);b.hmr(half,m);masks.append(m);b.swap(half,out);b.free(out)
    fix_scaled_half_masks(b,x,y,masks,K)
    for m in masks:b.free(m)
    um=b.alloc_bits(12);b.hmr(u,um);b.free(u)
    N=b.alloc_qubits(12);ar._norm_xor(b,y,N,K);phase_inverse_low12(b,N,um);b.free(um)
    ar._erase_norm(b,y,N,K);b.free(N)
    phase_product24(b,x,y,oldmask,K);b.free(oldmask)


def phase_inverse36(b,x,mask):
    """Inverse12phase:36Toffolis,13scratch beyondinput12.

    Norm6cost15; nestednorm3+inverse3cost9; twoGF3trilinearphasescost12CCZ.
    Theirthirdquantumcontroliscomputedoneparityatatimefromxandclassicalmask,
    so no6-qubittransposed-maskbank isneeded.
    """
    to12,frm12,K12=tower(12);linear_inplace(b,x,to12)
    mt=classical_transpose(b,mask,frm12)
    D=b.alloc_qubits(6);ar._norm_xor(b,x,D,K12)
    to6,frm6,K6=tower(6);linear_inplace(b,D,to6)
    E=b.alloc_qubits(3);ar._norm_xor(b,D,E,K6);w=inverse(b,E)
    anc=b.alloc_qubits(1)[0];f6=field(6)
    def coefficient(h):
        for j in range(6):
            for r in bits(f6.mul(h,1<<j)):
                b.ccx(mt[r],x[j],anc);b.ccx(mt[r],x[j+6],anc);b.ccx(mt[r+6],x[j+6],anc)
    for segment in(0,1):
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
    b.free(anc);erase_inverse(b,E,w);b.free(w)
    ar._erase_norm(b,D,E,K6);b.free(E);linear_inplace(b,D,frm6)
    ar._erase_norm(b,x,D,K12);b.free(D);linear_inplace(b,x,frm12);b.free(mt)


def divide_lowwidth375(b,x,y,K=None):
    """375T preserved-factor division at25scratch,promisedx0wheny0."""
    if K is None:K=extension_columns()[2]
    b.cx(y[12:],y[:12]);oldmask=multiply_forward(b,x,y,K);b.cx(y[12:],y[:12])
    u=b.alloc_qubits(12);ar._norm_xor(b,y,u,K)
    nm=inverse_stream_forward(b,u);ar._phase_norm(b,y,nm,K);b.free(nm)
    masks=[]
    for offset in(0,12):
        half=x[offset:offset+12];out=b.alloc_qubits(12);mul_xor(b,half,u,out)
        m=b.alloc_bits(12);b.hmr(half,m);masks.append(m);b.swap(half,out);b.free(out)
    um=b.alloc_bits(12);b.hmr(u,um);b.free(u)
    N=b.alloc_qubits(12);ar._norm_xor(b,y,N,K)
    for offset,m in zip((0,12),masks):phase_product(b,x[offset:offset+12],N,m);b.free(m)
    phase_inverse36(b,N,um);b.free(um)
    ar._erase_norm(b,y,N,K);b.free(N)
    phase_product24(b,x,y,oldmask,K);b.free(oldmask)
