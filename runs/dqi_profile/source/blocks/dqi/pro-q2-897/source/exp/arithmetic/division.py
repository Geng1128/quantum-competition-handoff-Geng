"""Width-bounded GF24 scalar multiplication/division reference schedules.

All24-bit words use explicit a+b*y, y²+y=K. This file is separate from the
verified primitive API while the scalar-normalization direction is measured.
"""
from field import *


def mul24_xor126(b,x,y,z,K=None):
    """z ^=x*y,126Toffolis,1scratch; inputs/target disjoint."""
    if K is None:K=extension_columns()[2]
    f=field(12);a=x[:12];c=x[12:];d=y[:12];e=y[12:];za=z[:12];zb=z[12:]
    # p=a*d contributes to both output halves.
    b.cx(za,zb);mul_xor(b,a,d,za);b.cx(za,zb)
    # q=c*e contributes Kq to the first half only.
    b.gf2_imul(f.invert(K),target=km.array(za),field=f);mul_xor(b,c,e,za);b.gf2_imul(K,target=km.array(za),field=f)
    # r=(a+c)(d+e) contributes to the second half.
    b.cx(c,a);b.cx(e,d);mul_xor(b,a,d,zb);b.cx(e,d);b.cx(c,a)



def mul24_xor(b,x,y,z,K=None):
    """Default105T/1scratch product; old126Tcomparison ismul24_xor126."""
    mul24_xor105(b,x,y,z,K)


def phase_product24(b,x,y,mask,K=None):
    if K is None:K=extension_columns()[2]
    a=x[:12];c=x[12:];d=y[:12];e=y[12:];ma=mask[:12];mb=mask[12:]
    phase_product(b,a,d,ma)
    mk=b.alloc_bits(12);b.bit_store(mk,False)
    # Transpose constant multiplication K into the classical phase mask.
    for i in range(12):
        for j in bits(field(12).mul(K,1<<i)):b.cx(ma[j],mk[i])
    phase_product(b,c,e,mk);b.free(mk)
    phase_product(b,a,e,mb);phase_product(b,c,d,mb);phase_product(b,c,e,mb)


def multiply_forward(b,x,y,K=None):
    """105T forward x*=y, returns deferred oldx phase mask.

    For y=0 this is correct only on valid subspace x=0. It is not an all-input
    identity-at-zero permutation; caller must enforce that promise or add flag.
    """
    z=b.alloc_qubits(24);mul24_xor(b,x,y,z,K)
    m=b.alloc_bits(24);b.hmr(x,m);b.swap(x,z);b.free(z);return m


def divide_preserving_factor(b,x,y,K=None):
    """513T division x/=y preserving y,43workspace; valid x=0 when y=0."""
    ym=inverse24_inplace_forward(b,y,K)
    z=b.alloc_qubits(24);mul24_xor(b,x,y,z,K)
    xm=b.alloc_bits(24);b.hmr(x,xm);b.swap(x,z);b.free(z)
    inverse24_inplace_reverse(b,y,ym,K)
    phase_product24(b,x,y,xm,K);b.free(xm)


def multiply_reverse(b,x,y,transcript,K=None):
    divide_preserving_factor(b,x,y,K)
    b.cz(x,transcript);b.free(transcript)


def divide_via_norm(b,x,y,K=None,retain_norm=True):
    """Division x/=y:321T/43workspace or342T/37workspace.

    y=0 requires promisedx=0. This all-valid-state exactness includes0 but does
    NOT implement identity on arbitraryx at y=0.
    """
    import field as ar
    if K is None:K=extension_columns()[2]
    b.cx(y[12:],y[:12]);oldmask=multiply_forward(b,x,y,K);b.cx(y[12:],y[:12])
    N=b.alloc_qubits(12);ar._norm_xor(b,y,N,K)
    if retain_norm:u,D=ar._inverse_keep_norm(b,N)
    else:u=inverse(b,N)
    for offset in (0,12):
        half=x[offset:offset+12];out=b.alloc_qubits(12);mul_xor(b,half,u,out)
        erase_product(b,out,N,half);b.swap(half,out);b.free(out)
    if retain_norm:
        to,frm,K12=tower(12);linear_inplace(b,u,to)
        v=b.alloc_qubits(6);ar._norm_xor(b,u,v,K12);linear_inplace(b,u,frm)
        ar._erase_inverse_with_norm(b,N,u,v);b.free(u)
        erase_inverse(b,D,v);b.free(v)
        linear_inplace(b,N,to);ar._erase_norm(b,N,D,K12);b.free(D);linear_inplace(b,N,frm)
    else:erase_inverse_releasing(b,N,u)
    ar._erase_norm(b,y,N,K);b.free(N)
    phase_product24(b,x,y,oldmask,K);b.free(oldmask)


def multiply_reverse_via_norm(b,x,y,transcript,K=None,retain_norm=True):
    divide_via_norm(b,x,y,K,retain_norm=retain_norm)
    b.cz(x,transcript);b.free(transcript)


def zero_flag(b,x):
    """Compute [x=0], n-1ANDs; returnoneflag, clearallinternalANDs."""
    n=len(x);b.x(x);ladder=b.alloc_qubits(n-1)
    b.init_and(x[0],x[1],ladder[0])
    for i in range(1,n-1):b.init_and(ladder[i-1],x[i+1],ladder[i])
    for i in reversed(range(n-2)):
        if i==0:b.del_and(x[0],x[1],ladder[0])
        else:b.del_and(ladder[i-1],x[i+1],ladder[i])
    b.x(x);b.free(ladder[:-1]);return ladder[-1]


def phase_zero(b,y,mask,head=None,headmask=None):
    """Applyzero(y)*(mask+dot(head,headmask));22T+1CCZ for y24."""
    n=len(y);b.x(y);ladder=b.alloc_qubits(n-2)
    b.init_and(y[0],y[1],ladder[0])
    for i in range(1,n-2):b.init_and(ladder[i-1],y[i+1],ladder[i])
    b.ccz(ladder[-1],y[-1],mask)
    if head is not None:
        q=b.alloc_qubits(1)[0]
        for xi,mi in zip(head,headmask):b.ccx(mi,xi,q)
        b.ccz(ladder[-1],y[-1],q)
        for xi,mi in reversed(list(zip(head,headmask))):b.ccx(mi,xi,q)
        b.free(q)
    for i in reversed(range(n-2)):
        if i==0:b.del_and(y[0],y[1],ladder[0])
        else:b.del_and(ladder[i-1],y[i+1],ladder[i])
    b.free(ladder);b.x(y)


def multiply_zero_forward(b,x,y,K=None):
    """152T multiplication, identityat y0. Deferredhead+zero transcripts."""
    flag=zero_flag(b,y);z=b.alloc_qubits(24);mul24_xor(b,x,y,z,K);b.ccx(flag,x,z)
    oldmask=b.alloc_bits(24);b.hmr(x,oldmask);b.swap(x,z);b.free(z)
    fm=b.alloc_bits(1);b.hmr(flag,fm[0]);b.free(flag)
    return oldmask,fm[0]


def divide_zero(b,x,y,K=None,extra_zero_mask=None):
    """379T exact division x/=y, identityat y0,43workspace."""
    import field as ar
    if K is None:K=extension_columns()[2]
    N=b.alloc_qubits(12);ar._norm_xor(b,y,N,K);flag=zero_flag(b,N)
    b.cx(y[12:],y[:12]);z=b.alloc_qubits(24);mul24_xor(b,x,y,z,K);b.ccx(flag,x,z);b.cx(y[12:],y[:12])
    oldmask=b.alloc_bits(24);b.hmr(x,oldmask);b.swap(x,z);b.free(z)
    b.cx(flag,N[0]);fm=b.alloc_bits(1);b.hmr(flag,fm[0]);b.free(flag)
    u,D=ar._inverse_keep_norm(b,N)
    for offset in (0,12):
        half=x[offset:offset+12];out=b.alloc_qubits(12);mul_xor(b,half,u,out)
        erase_product(b,out,N,half);b.swap(half,out);b.free(out)
    to,frm,K12=tower(12);linear_inplace(b,u,to);v=b.alloc_qubits(6);ar._norm_xor(b,u,v,K12);linear_inplace(b,u,frm)
    ar._erase_inverse_with_norm(b,N,u,v);b.free(u);erase_inverse(b,D,v);b.free(v)
    linear_inplace(b,N,to);ar._erase_norm(b,N,D,K12);b.free(D);linear_inplace(b,N,frm)
    nm=b.alloc_bits(12);b.hmr(N,nm);b.free(N);ar._phase_norm(b,y,nm,K)
    comb=b.alloc_bits(1);b.bit_store(comb,False);b.cx(fm[0],comb[0]);b.cx(nm[0],comb[0])
    if extra_zero_mask is not None:b.cx(extra_zero_mask,comb[0])
    phase_product24(b,x,y,oldmask,K);phase_zero(b,y,comb[0],x,oldmask)
    b.free(oldmask);b.free(fm);b.free(nm);b.free(comb)


def multiply_zero_reverse(b,x,y,transcript,K=None):
    oldmask,fm=transcript;divide_zero(b,x,y,K,extra_zero_mask=fm)
    b.cz(x,oldmask);b.free(oldmask);b.free(fm)



def erase_inverse_releasing(b,x,y):
    """Phase-only inverse erasure; measure/free target BEFORE recomputation.

    Same9/45/153T recursion as field.erase_inverse, but permits early reuse
    of y's qubits. This helper owns and frees y.
    """
    import field as ar
    n=len(x)
    if n==3:
        mask=b.alloc_bits(3);b.hmr(y,mask);b.free(y)
        for im,v in enumerate(inverse3_anf()):
            ix=list(bits(im))
            for j in bits(v):
                if len(ix)==1:b.cz(x[ix[0]],mask[j])
                elif len(ix)==2:b.ccz(x[ix[0]],x[ix[1]],mask[j])
        b.free(mask);return
    k=n//2;to,frm,K=tower(n);linear_inplace(b,y,to)
    mask=b.alloc_bits(n);b.hmr(y,mask);b.free(y)
    linear_inplace(b,x,to);norm=b.alloc_qubits(k);ar._norm_xor(b,x,norm,K);invnorm=inverse(b,norm)
    b.cx(x[k:],x[:k]);phase_product(b,x[:k],invnorm,mask[:k]);phase_product(b,x[k:],invnorm,mask[k:]);b.cx(x[k:],x[:k]);b.free(mask)
    erase_inverse_releasing(b,norm,invnorm)
    ar._erase_norm(b,x,norm,K);b.free(norm);linear_inplace(b,x,frm)
