"""Low-width GF(2^m) arithmetic, standard Kickmix polynomial basis.

Public: mul_xor(builder,x,y,z), inverse(builder,x,target=None),
linear_xor(builder, x, z, columns), linear_inplace(builder,x,columns),
phase_product(builder,x,y,mask), erase_linear/erase_product/erase_inverse.
All array arguments can be lists or km.array. Arithmetic preserves inputs.
`inverse` returns a fresh clean output, 0 maps to 0. Default GF2Field basis.
"""
from functools import lru_cache
import circ as km


def bits(x):
    while x:
        t=x&-x; yield t.bit_length()-1; x^=t


def apply(columns, value):
    out=0
    for i in bits(value): out ^= columns[i]
    return out


def inverse_columns(columns):
    n=len(columns)
    rows=[sum(((c>>i)&1)<<j for j,c in enumerate(columns)) | (1<<(n+i)) for i in range(n)]
    for j in range(n):
        p=next(i for i in range(j,n) if rows[i]>>j&1)
        rows[j],rows[p]=rows[p],rows[j]
        for i in range(n):
            if i!=j and rows[i]>>j&1: rows[i]^=rows[j]
    return [sum(((rows[i]>>(n+j))&1)<<i for i in range(n)) for j in range(n)]


def linear_xor(b,x,z,columns):
    for i,c in enumerate(columns):
        for j in bits(c): b.cx(x[i],z[j])


def linear_inplace(b,x,columns):
    """Invertible GF2 map, no ancillary qubits. columns[j] = image(1<<j)."""
    n=len(x); rows=[sum(((c>>i)&1)<<j for j,c in enumerate(columns)) for i in range(n)]; ops=[]
    for j in range(n):
        p=next(i for i in range(j,n) if rows[i]>>j&1)
        if p!=j: rows[p],rows[j]=rows[j],rows[p]; ops.append(('swap',p,j))
        for i in range(n):
            if i!=j and rows[i]>>j&1: rows[i]^=rows[j];ops.append(('cx',j,i))
    for op,i,j in reversed(ops): getattr(b,op)(x[i],x[j])


def phase_linear(b,x,mask,columns):
    for i,c in enumerate(columns):
        for j in bits(c): b.cz(x[i],mask[j])


def erase_linear(b,x,z,columns):
    mask=b.alloc_bits(len(z)); b.hmr(z,mask); phase_linear(b,x,mask,columns);b.free(mask)


@lru_cache(None)
def field(n): return km.GF2Field(n)


@lru_cache(None)
def tensor(n):
    """Symmetric multiplication tensor (linear mask, output vector), rank3/6/15/42."""
    f=field(n)
    if n==2: masks=[1,2,3]
    elif n==3: masks=[1,7,4,5,6,3]
    elif n in (6,12,24):
        k,d={6:(2,3),12:(3,4),24:(6,4)}[n]
        sub=field(k); power=(2**n-1)//(2**k-1)
        gen=f.pow(f.primitive_element,power)
        def evalpoly(v,p):
            q=0
            for i in reversed(range(k+1)):q=f.mul(q,p)^((v>>i)&1)
            return q
        embroot=next(f.pow(gen,i) for i in range(1,2**k) if evalpoly(sub.modulus,f.pow(gen,i))==0)
        embed=[f.pow(embroot,i) for i in range(k)]
        theta=f.primitive_element
        cols=[f.mul(embed[i],f.pow(theta,j)) for j in range(d) for i in range(k)]
        inv=inverse_columns(cols)
        lowmasks=[m for m,_ in tensor(k)]
        masks=[]
        for point in list(range(2*d-2))+[None]:
            outcols=[]
            for extbit in range(n):
                coord=inv[extbit]
                if point is None: value=(coord>>(k*(d-1)))&((1<<k)-1)
                else:
                    value=0
                    for j in reversed(range(d)): value=sub.mul(value,point)^((coord>>(k*j))&((1<<k)-1))
                outcols.append(value)
            for lm in lowmasks:masks.append(sum(((v&lm).bit_count()%2)<<i for i,v in enumerate(outcols)))
    else: raise ValueError(n)
    # Each basis-pair equation determines output vector of rank terms.
    rank=len(masks); piv={}
    for i in range(n):
        for j in range(i,n):
            row=sum(((m>>i&1)&(m>>j&1))<<r for r,m in enumerate(masks)); rhs=f.mul(1<<i,1<<j)
            while row:
                p=(row&-row).bit_length()-1
                if p in piv:row^=piv[p][0];rhs^=piv[p][1]
                else:piv[p]=(row,rhs);break
            if row==0: assert rhs==0,(n,i,j,rhs)
    out=[0]*rank
    for p,(row,rhs) in sorted(piv.items(),reverse=True):
        for j in bits(row^(1<<p)):rhs^=out[j]
        out[p]=rhs
    assert all(out)
    return tuple(zip(masks,out))


def parity_into(b,x,mask,t):
    for i in bits(mask):b.cx(x[i],t)


def mul_xor(b,x,y,z):
    """Accumulate product, 1 scratch when inputs disjoint; 3 if overlapping.

    Linear forms are computed temporarily in a chosen input pivot, restored
    after each rank-one term. The output must be disjoint from both inputs.
    """
    n=len(x);assert len(y)==len(z)==n
    overlap=bool(set(map(str,x)) & set(map(str,y)))
    if overlap:
        a,c,p=b.alloc_qubits(3)
        for mask,v in tensor(n):
            parity_into(b,x,mask,a);parity_into(b,y,mask,c)
            b.init_and(a,c,p)
            for i in bits(v):b.cx(p,z[i])
            b.del_and(a,c,p)
            parity_into(b,y,mask,c);parity_into(b,x,mask,a)
        b.free([a,c,p])
    else:
        p=b.alloc_qubits(1)[0]
        for mask,v in tensor(n):
            ix=list(bits(mask));pivot=ix[0];a=x[pivot];c=y[pivot]
            for i in ix[1:]:b.cx(x[i],a);b.cx(y[i],c)
            b.init_and(a,c,p)
            for i in bits(v):b.cx(p,z[i])
            b.del_and(a,c,p)
            for i in reversed(ix[1:]):b.cx(y[i],c);b.cx(x[i],a)
        b.free(p)


def phase_product(b,x,y,mask):
    """Exact phase dot(mask,x*y), zero Toffolis, no quantum workspace."""
    f=field(len(x))
    for i in range(len(x)):
        for j in range(len(y)):
            v=f.mul(1<<i,1<<j)
            for k in bits(v):b.ccz(x[i],y[j],mask[k])


def erase_product(b,x,y,z):
    mask=b.alloc_bits(len(z));b.hmr(z,mask);phase_product(b,x,y,mask);b.free(mask)


@lru_cache(None)
def tower(n):
    """Coordinates a+b*y, y²+y=K; returns to/from tower and K."""
    k=n//2;f=field(n);sub=field(k)
    gen=f.pow(f.primitive_element,(2**n-1)//(2**k-1))
    def ev(v,p):
        out=0
        for i in reversed(range(k+1)):out=f.mul(out,p)^((v>>i)&1)
        return out
    root=next(f.pow(gen,i) for i in range(1,2**k) if ev(sub.modulus,f.pow(gen,i))==0)
    embed=[f.pow(root,i) for i in range(k)]
    # Any trace-relative-one element has minimal polynomial y²+y+K.
    y=next(v for v in range(2**n) if f.pow(v,2**k)^v==1)
    cols=embed+[f.mul(v,y) for v in embed];inv=inverse_columns(cols)
    kval=apply(inv,f.square(y)^y);assert kval<2**k
    return tuple(inv),tuple(cols),kval


def square_columns(n):return [field(n).square(1<<i) for i in range(n)]


def _norm_xor(b,x,z,kval):
    """x already in tower coordinates. Norm = a²+ab+K b²."""
    k=len(z);f=field(k);a=x[:k];c=x[k:]
    linear_xor(b,a,z,square_columns(k));linear_xor(b,c,z,[f.mul(kval,v) for v in square_columns(k)])
    mul_xor(b,a,c,z)


def _phase_norm(b,x,mask,kval):
    k=len(mask);f=field(k);a=x[:k];c=x[k:]
    phase_linear(b,a,mask,square_columns(k));phase_linear(b,c,mask,[f.mul(kval,v) for v in square_columns(k)])
    phase_product(b,a,c,mask)


def _erase_norm(b,x,z,kval):
    mask=b.alloc_bits(len(z));b.hmr(z,mask);_phase_norm(b,x,mask,kval);b.free(mask)


@lru_cache(None)
def inverse3_anf():
    f=field(3);a=[0 if x==0 else f.invert(x) for x in range(8)]
    for i in range(3):
        for s in range(8):
            if s>>i&1:a[s]^=a[s^(1<<i)]
    assert a[7]==0
    return tuple(a)


def inverse(b,x,target=None):
    """Clean out-of-place inverse: I3=3,I6=21,I12=66,I24=192 Toffolis.

    Target must be initially zero when passed. Allocating target externally can
    increase peak width: omit target for just-in-time recursive allocation.
    """
    n=len(x)
    if n==3:
        if target is None:target=b.alloc_qubits(3)
        anc=b.alloc_qubits(1)[0]
        for mask,v in enumerate(inverse3_anf()):
            ix=list(bits(mask))
            if len(ix)==1:
                for j in bits(v):b.cx(x[ix[0]],target[j])
            elif len(ix)==2:
                b.init_and(x[ix[0]],x[ix[1]],anc)
                for j in bits(v):b.cx(anc,target[j])
                b.del_and(x[ix[0]],x[ix[1]],anc)
        b.free(anc);return target
    k=n//2;to,frm,K=tower(n)
    linear_inplace(b,x,to)
    norm=b.alloc_qubits(k);_norm_xor(b,x,norm,K)
    invnorm=inverse(b,norm)
    _erase_norm(b,x,norm,K);b.free(norm)
    if target is None:target=b.alloc_qubits(n)
    # x conjugate is (a+b,b), modify temporarily in-place.
    a=x[:k];c=x[k:];b.cx(c,a)
    mul_xor(b,a,invnorm,target[:k]);mul_xor(b,c,invnorm,target[k:]);b.cx(c,a)
    # norm(target) = invnorm, including zero.
    _erase_norm(b,target,invnorm,K);b.free(invnorm)
    linear_inplace(b,x,frm);linear_inplace(b,target,frm)
    return target


def inverse_inplace_forward(b,x):
    """Forward 192T (n24), classical n-bit deferred input phase transcript."""
    y=inverse(b,x);mask=b.alloc_bits(len(x));b.hmr(x,mask);b.swap(x,y);b.free(y)
    return mask


def erase_inverse(b,x,y):
    """Erase y=inv(x) with a closed phase-only inverse recomputation.

    Generic fallback costs Ierase3=0, Ierase6=9, Ierase12=45,
    Ierase24=153. It is NOT the claimed 30T inverse12 erasure.
    """
    n=len(x)
    if n==3:
        mask=b.alloc_bits(3);b.hmr(y,mask)
        for im,v in enumerate(inverse3_anf()):
            ix=list(bits(im))
            for j in bits(v):
                if len(ix)==1:b.cz(x[ix[0]],mask[j])
                elif len(ix)==2:b.ccz(x[ix[0]],x[ix[1]],mask[j])
        b.free(mask);return
    k=n//2;to,frm,K=tower(n);linear_inplace(b,x,to);linear_inplace(b,y,to)
    norm=b.alloc_qubits(k);_norm_xor(b,x,norm,K);invnorm=inverse(b,norm)
    mask=b.alloc_bits(n);b.hmr(y,mask)
    b.cx(x[k:],x[:k]);phase_product(b,x[:k],invnorm,mask[:k]);phase_product(b,x[k:],invnorm,mask[k:]);b.cx(x[k:],x[:k]);b.free(mask)
    erase_inverse(b,norm,invnorm);b.free(invnorm)
    _erase_norm(b,x,norm,K);b.free(norm);linear_inplace(b,x,frm)


def inverse_inplace_reverse_fallback(b,x,transcript):
    """Close forward transcript; exact345T reverse at degree24, width61.

    Kept as a correctness reference, NOT competitive with the target222T.
    """
    y=inverse(b,x)
    b.cz(y,transcript);b.free(transcript)
    erase_inverse(b,y,x)
    b.swap(x,y);b.free(y)


def _inverse_keep_norm(b,x):
    """Initialized inverse with its half-size input norm kept quantum-live.

    Cost unchanged from inverse(). Caller must eventually erase kept norm.
    """
    n=len(x);k=n//2;to,frm,K=tower(n);linear_inplace(b,x,to)
    norm=b.alloc_qubits(k);_norm_xor(b,x,norm,K)
    invnorm=inverse(b,norm)
    target=b.alloc_qubits(n)
    b.cx(x[k:],x[:k]);mul_xor(b,x[:k],invnorm,target[:k]);mul_xor(b,x[k:],invnorm,target[k:]);b.cx(x[k:],x[:k])
    _erase_norm(b,target,invnorm,K);b.free(invnorm)
    linear_inplace(b,x,frm);linear_inplace(b,target,frm)
    return target,norm


def _erase_inverse_with_norm(b,x,y,norm_y):
    """Erase y=inv(x) when norm(y) is retained. Zero Toffolis.

    y=conjugate(x)*norm(y), so the phase fix is bilinear.
    Both x and y arrive in polynomial basis; norm_y in half-field basis.
    """
    n=len(x);k=n//2;to,frm,K=tower(n)
    linear_inplace(b,x,to);linear_inplace(b,y,to)
    b.cx(x[k:],x[:k]);erase_product(b,x[:k],norm_y,y[:k]);erase_product(b,x[k:],norm_y,y[k:]);b.cx(x[k:],x[:k])
    linear_inplace(b,x,frm)


def inverse_inplace_reverse(b,x,transcript):
    """Close inverse_inplace_forward: degree24 reverse216T, peak67 qubits.

    Retain norm(norm(input))6 during inverse12. Stream the two output halves
    so retained N12+t12+D6 add at most43 ancillas to input24. The final D
    cleanup costs norm(t)15 + inverse6 phase9 =24 additional Toffolis.
    """
    n=len(x);assert n in (12,24)
    k=n//2;to,frm,K=tower(n);linear_inplace(b,x,to)
    N=b.alloc_qubits(k);_norm_xor(b,x,N,K)
    t,D=_inverse_keep_norm(b,N)
    outb=b.alloc_qubits(k);mul_xor(b,x[k:],t,outb)
    erase_product(b,outb,N,x[k:])
    mul_xor(b,x[:k],t,x[k:]);b.cx(outb,x[k:])
    b.cx(outb,x[k:]);erase_product(b,x[k:],N,x[:k]);b.cx(outb,x[k:])
    b.swap(x[:k],x[k:]);b.swap(x[k:],outb);b.free(outb)
    # N=inv(t), and norm(N)=D was retained inside inverse(N).
    _erase_inverse_with_norm(b,t,N,D);b.free(N)
    tok,frmk,Kk=tower(k);linear_inplace(b,t,tok)
    u=b.alloc_qubits(k//2);_norm_xor(b,t,u,Kk)
    erase_inverse(b,u,D);b.free(D)
    _erase_norm(b,t,u,Kk);b.free(u);linear_inplace(b,t,frmk)
    _erase_norm(b,x,t,K);b.free(t)
    linear_inplace(b,x,frm)
    b.cz(x,transcript);b.free(transcript)


@lru_cache(None)
def extension_columns(K=None):
    """Map base12 pair a+b*y to standard GF24 polynomial basis.

    Uses base GF12 modulus0x1053. K is any GF12 trace-one element, y²+y=K.
    Omitting K picks the smallest trace-one word, as in BRIEF.md.
    Return (to_polynomial_columns, from_polynomial_columns, K).
    """
    f=field(12)
    def trace(v):
        out=0
        for i in range(12):out^=f.frobenius(v,i)
        return out
    if K is None:K=next(v for v in range(4096) if trace(v)==1)
    if not 0<=K<4096 or trace(K)!=1:raise ValueError('K must have absolute trace1')
    _,frm,K0=tower(24)
    c=next(v for v in range(4096) if f.square(v)^v==K^K0)
    F=field(24);y=frm[12]^apply(frm[:12],c)
    cols=tuple(frm[:12])+tuple(F.mul(v,y) for v in frm[:12])
    return cols,tuple(inverse_columns(cols)),K


def inverse24_inplace_forward(b,x,K=None):
    """192T forward inversion in explicit a+b*y coordinates; returns transcript."""
    cols,invcols,_=extension_columns(K);linear_inplace(b,x,cols)
    transcript=inverse_inplace_forward(b,x);linear_inplace(b,x,invcols)
    return transcript


def inverse24_inplace_reverse(b,x,transcript,K=None):
    """216T closed reverse,43 workspace, in explicit a+b*y coordinates."""
    cols,invcols,_=extension_columns(K);linear_inplace(b,x,cols)
    inverse_inplace_reverse(b,x,transcript);linear_inplace(b,x,invcols)


def _solve_output_vectors(rows,rank):
    piv={}
    for row,rhs in rows:
        while row:
            p=(row&-row).bit_length()-1
            if p in piv:row^=piv[p][0];rhs^=piv[p][1]
            else:piv[p]=(row,rhs);break
        if row==0:assert rhs==0,'tensor does not span function'
    out=[0]*rank
    for p,(row,rhs) in sorted(piv.items(),reverse=True):
        for j in bits(row^(1<<p)):rhs^=out[j]
        out[p]=rhs
    return out


@lru_cache(None)
def cube4_tensor():
    """Four ANDs plus linear gates for GF16 cube."""
    f=field(4);pairs=[(i,j) for i in range(4) for j in range(i+1,4)]
    quads=[sum(((f.pow((1<<i)^(1<<j),3)^f.pow(1<<i,3)^f.pow(1<<j,3))>>k&1)<<p for p,(i,j) in enumerate(pairs)) for k in range(4)]
    span={0}
    for q in quads:span|={v^q for v in list(span)}
    selected=[];piv={}
    for a in range(1,16):
        for c in range(a+1,16):
            q=sum((((a>>i&1)&(c>>j&1))^((a>>j&1)&(c>>i&1)))<<p for p,(i,j) in enumerate(pairs))
            if q not in span:continue
            rem=q
            for p,v in sorted(piv.items()):
                if rem>>p&1:rem^=v
            if rem:
                p=(rem&-rem).bit_length()-1;piv[p]=rem;selected.append((a,c))
    assert len(selected)==4
    rows=[]
    for i,j in pairs:
        row=sum((((a>>i&1)&(c>>j&1))^((a>>j&1)&(c>>i&1)))<<p for p,(a,c) in enumerate(selected))
        rhs=f.pow((1<<i)^(1<<j),3)^f.pow(1<<i,3)^f.pow(1<<j,3);rows.append((row,rhs))
    out=_solve_output_vectors(rows,len(selected))
    return tuple((a,c,v) for (a,c),v in zip(selected,out))


@lru_cache(None)
def cube12_tensor():
    """Seven GF16 cubic evaluations, each four ANDs: rank28 quadratic map."""
    f=field(12);sub=field(4);gen=f.pow(f.primitive_element,273)
    def ev(v,p):
        out=0
        for i in reversed(range(5)):out=f.mul(out,p)^((v>>i)&1)
        return out
    root=next(f.pow(gen,i) for i in range(1,16) if ev(sub.modulus,f.pow(gen,i))==0)
    embed=[f.pow(root,i) for i in range(4)]
    coords=inverse_columns([f.mul(v,f.pow(f.primitive_element,j)) for j in range(3) for v in embed])
    terms=[]
    for point in range(7):
        vals=[]
        for co in coords:
            v=0
            for j in reversed(range(3)):v=sub.mul(v,point)^((co>>(4*j))&15)
            vals.append(v)
        for a,c,_ in cube4_tensor():
            A=sum(((v&a).bit_count()%2)<<i for i,v in enumerate(vals));C=sum(((v&c).bit_count()%2)<<i for i,v in enumerate(vals));terms.append((A,C))
    rows=[]
    for i in range(12):
        for j in range(i+1,12):
            row=sum((((a>>i&1)&(c>>j&1))^((a>>j&1)&(c>>i&1)))<<p for p,(a,c) in enumerate(terms))
            rhs=f.pow((1<<i)^(1<<j),3)^f.pow(1<<i,3)^f.pow(1<<j,3);rows.append((row,rhs))
    out=_solve_output_vectors(rows,len(terms));assert all(out)
    linear=[]
    for i in range(12):
        v=f.pow(1<<i,3)
        for (a,c),o in zip(terms,out):
            if a>>i&1 and c>>i&1:v^=o
        linear.append(v)
    return tuple((a,c,v) for (a,c),v in zip(terms,out)),tuple(linear)


def cube_xor(b,x,z):
    """GF12 cube accumulator z ^=x³,28Toffolis and3workspace."""
    assert len(x)==len(z)==12
    terms,linear=cube12_tensor();linear_xor(b,x,z,linear)
    a,c,p=b.alloc_qubits(3)
    for A,C,v in terms:
        parity_into(b,x,A,a);parity_into(b,x,C,c);b.init_and(a,c,p)
        for i in bits(v):b.cx(p,z[i])
        b.del_and(a,c,p);parity_into(b,x,C,c);parity_into(b,x,A,a)
    b.free([a,c,p])



@lru_cache(None)
def tensor24_extension(K=None):
    """105rank multiplicationtensor in explicit a+b*y, y²+y=K basis."""
    cols,invcols,_=extension_columns(K)
    return tuple((sum(((m&v).bit_count()%2)<<i for i,v in enumerate(cols)),apply(invcols,out)) for m,out in tensor(24))


def mul24_xor105(b,x,y,z,K=None):
    """Arbitrarytarget GF24 product in explicit quadraticbasis:105T,1scratch.

    No default126T multiplicationcaller ischanged; this is anoptional newAPI.
    Allthree24-qubit registers mustbedisjoint.
    """
    assert len(x)==len(y)==len(z)==24
    assert not (set(map(str,x))&set(map(str,y)) or set(map(str,x))&set(map(str,z)) or set(map(str,y))&set(map(str,z)))
    p=b.alloc_qubits(1)[0]
    for mask,v in tensor24_extension(K):
        ix=list(bits(mask));pivot=ix[0];a=x[pivot];c=y[pivot]
        for i in ix[1:]:b.cx(x[i],a);b.cx(y[i],c)
        b.init_and(a,c,p)
        for i in bits(v):b.cx(p,z[i])
        b.del_and(a,c,p)
        for i in reversed(ix[1:]):b.cx(y[i],c);b.cx(x[i],a)
    b.free(p)


def mul_xor0(b,x,y,z):
    """Standardpolynomialbasis arbitrarytarget product withZEROworkspace.

    Disjointx,y,z required. Conjugateeachoutputrankvectorintoonepivot,
    applyCCXdirectlythere,thenundooutputconjugation. Same42/105Tcounts.
    """
    n=len(x);assert len(y)==len(z)==n
    assert not(set(map(str,x))&set(map(str,y)) or set(map(str,x))&set(map(str,z)) or set(map(str,y))&set(map(str,z)))
    for mask,v in tensor(n):
        ix=list(bits(mask));iz=list(bits(v));p=ix[0];t=iz[0]
        for i in ix[1:]:b.cx(x[i],x[p]);b.cx(y[i],y[p])
        for j in iz[1:]:b.cx(z[t],z[j])
        b.ccx(x[p],y[p],z[t])
        for j in reversed(iz[1:]):b.cx(z[t],z[j])
        for i in reversed(ix[1:]):b.cx(y[i],y[p]);b.cx(x[i],x[p])


def mul24_xor105_zero(b,x,y,z,K=None):
    """Explicita+b*y-basis105T multiplication withZEROprivatequbits."""
    assert len(x)==len(y)==len(z)==24
    assert not(set(map(str,x))&set(map(str,y)) or set(map(str,x))&set(map(str,z)) or set(map(str,y))&set(map(str,z)))
    for mask,v in tensor24_extension(K):
        ix=list(bits(mask));iz=list(bits(v));p=ix[0];t=iz[0]
        for i in ix[1:]:b.cx(x[i],x[p]);b.cx(y[i],y[p])
        for j in iz[1:]:b.cx(z[t],z[j])
        b.ccx(x[p],y[p],z[t])
        for j in reversed(iz[1:]):b.cx(z[t],z[j])
        for i in reversed(ix[1:]):b.cx(y[i],y[p]);b.cx(x[i],x[p])
