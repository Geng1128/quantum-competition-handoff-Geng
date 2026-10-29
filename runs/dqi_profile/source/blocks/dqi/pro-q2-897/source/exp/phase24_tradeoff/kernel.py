"""Inverse24 classical-mask phase tradeoff; frozen jet source is read-only."""
from exp.jet24.kernel import *
from exp.mul99.phase36 import phase_classical_parity


def rank(rows):
    p={}
    for v in rows:
        while v:
            j=v.bit_length()-1
            if j in p:v^=p[j]
            else:p[j]=v;break
    return len(p)


def cols_from_rows(rows):return[sum(((r>>j)&1)<<i for i,r in enumerate(rows))for j in range(len(rows))]


def layout():
    rows=[]
    for l,r in LAY['gates']:
        for v in(l>>1,r>>1):
            if rank(rows+[v])>len(rows):rows.append(v)
    assert len(rows)==12
    pivot=next(j for j in range(6)if rank(rows+[1<<j])==13);rows.append(1<<pivot);cols=cols_from_rows(rows);inv=inverse_columns(cols)
    def form(e):return(e&1)|sum((((((e>>1)&65535)&inv[j]).bit_count()&1)<<(j+1))for j in range(13))
    outputs=[(e&~65535)|form(e&65535)for e in LAY['outputs']];gates=[tuple(form(e)for e in g)for g in LAY['gates']]
    assert all(not(e>>13&1)for g in gates for e in g)
    v=sum(((e>>13)&1)<<j for j,e in enumerate(outputs));assert rank([1,2,v])==3
    rest=[]
    for j in range(2,6):
        if rank([1,2,v]+rest+[1<<j])>3+len(rest):rest.append(1<<j)
    B=[1,2]+rest+[v];assert len(B)==6;U=inverse_columns(B)
    def outputmix(values):return[sum(0 for _ in[])for j in[]]
    newoutputs=[]
    for j in range(6):
        e=0
        for i in range(6):
            if U[i]>>j&1:e^=outputs[i]
        newoutputs.append(e)
    assert[(e>>13)&1 for e in newoutputs]==[0,0,0,0,0,1]
    return dict(pivot=pivot,source_columns=cols,source_inverse=inv,output_B=B,outputs=newoutputs,gates=gates)

L30=layout()


def norm_bit_xor(b,N,target,K12,to6,j):
    """One tower6 component ofnorm12(N),6CCX andnoquantumscratch.
    N is12towercoordinates; outputcanonical6isthenmappedbyto6.
    """
    f=field(6)
    for i in range(6):
        if apply(to6,f.square(1<<i))>>j&1:b.cx(N[i],target)
        if apply(to6,f.mul(K12,f.square(1<<i)))>>j&1:b.cx(N[i+6],target)
        row=sum(((apply(to6,f.mul(1<<i,1<<k))>>j)&1)<<k for k in range(6));ix=list(bits(row));p=ix[0]
        for k in ix[1:]:b.cx(N[6+k],N[6+p])
        b.ccx(N[i],N[6+p],target)
        for k in reversed(ix[1:]):b.cx(N[6+k],N[6+p])


def virtual_mask_phase(b,x24,mask24,N12,out6,frm12):
    """30 executedCCZ,0scratch; L12 virtual inx24 andclassicalmask."""
    f=field(12)
    for segment in(0,1):
        if segment==0:b.cx(N12[6:],N12[:6])
        xx=N12[:6]if segment==0 else N12[6:]
        for m,v in tensor(6):
            h=apply(frm12,v<<(6*segment));co=b.alloc_bits(24);b.bit_store(co,False)
            for i in range(12):
                for r in bits(f.mul(h,1<<i)):
                    b.cx(mask24[r],co[i]);b.cx(mask24[r],co[i+12]);b.cx(mask24[r+12],co[i+12])
            ix=list(bits(m));p=ix[0]
            for k in ix[1:]:b.cx(xx[k],xx[p]);b.cx(out6[k],out6[p])
            phase_classical_parity(b,xx[p],out6[p],x24,co)
            for k in reversed(ix[1:]):b.cx(out6[k],out6[p]);b.cx(xx[k],xx[p])
            b.free(co)
        if segment==0:b.cx(N12[6:],N12[:6])


def phase_inverse24_canonical118(b,x,mask,zero_mask=None):
    """<=118 executedT/30private, inverse0phase; inputcanonical24."""
    to24,frm24,K24=tower(24);linear_inplace(b,x,to24);mt=classical_transpose(b,mask,frm24)
    N=b.alloc_qubits(12);norm_xor_noscratch(b,x,N,K24);to12,frm12,K12=tower(12);linear_inplace(b,N,to12)
    D=b.alloc_qubits(6);norm_xor_noscratch(b,N,D,K12);to6,frm6,K6=tower(6);linear_inplace(b,D,to6);g=b.alloc_qubits(6)
    for i,(l,r)in enumerate(basic.raw_forms19()):
        with ref.expose_pair(b,D,l,r)as(p,q):b.init_and(p,q,g[i])
    prefix=list(D)+list(g);forms=ref.norm_forms(K6);w=ref.inverse3_virtual(b,prefix,forms)
    linear_inplace(b,g,LAY['raw_columns']);R,E=g[:4],g[4:];source=list(D)+list(R)+list(w);linear_inplace(b,source,L30['source_columns']);C=source[:12];z=list(E)+list(b.alloc_qubits(3))+source[12:]
    for j,e in enumerate(L30['outputs']):ref.form_xor(b,C,e&8191,z[j])
    for i,(l,r)in enumerate(L30['gates']):
        v=sum(((e>>(16+i))&1)<<j for j,e in enumerate(L30['outputs']))
        with ref.expose_pair(b,C,l,r)as(p,q):rankone_xor(b,p,q,z,v)
    linear_inplace(b,z,L30['output_B']);linear_inplace(b,z,LAY['output_inverse_columns']);linear_inplace(b,z,frm6)
    virtual_mask_phase(b,x,mt,N,z,frm12);b.free(mt)
    # OutputHMR first frees6 before restoring the one originalD coordinate.
    zm=b.alloc_bits(6);b.hmr(z,zm);b.free(z);ym=classical_transpose(b,zm,frm6);b.free(zm);mm=classical_transpose(b,ym,LAY['output_inverse_columns']);b.free(ym)
    dj=b.alloc_qubits(1)[0];norm_bit_xor(b,N,dj,K12,to6,L30['pivot']);S=list(C)+[dj];linear_inplace(b,S,L30['source_inverse']);DD=S[:6];RR=S[6:10];ww=S[10:]
    phase_outputs_classical(b,S,LAY['outputs'],LAY['gates'],mm)
    for i,(l,r)in enumerate(basic.raw_forms19()):
        with ref.expose_pair(b,DD,l,r)as(p,q):
            for j,row in enumerate(LAY['raw_rows'][4:]):
                if row>>i&1:b.ccz(p,q,mm[j])
    b.free(mm)
    EE=b.alloc_qubits(3);norm_xor_noscratch(b,DD,EE,K6)
    if zero_mask is not None:
        b.z(zero_mask)
        for i in range(3):
            for j in range(3):
                if field(3).mul(1<<i,1<<j)&1:b.ccz(EE[i],ww[j],zero_mask)
    erase_inverse(b,EE,ww);b.free(ww);ar._erase_norm(b,DD,EE,K6);b.free(EE);zero1.erase_raw_cache(b,DD,RR,LAY['raw_rows'][:4]);linear_inplace(b,DD,frm6)
    dm=b.alloc_bits(6);b.hmr(DD,dm);b.free(DD);ar._phase_norm(b,N,dm,K12);b.free(dm);linear_inplace(b,N,frm12);ar._erase_norm(b,x,N,K24);b.free(N);linear_inplace(b,x,frm24)


def phase_inverse24_extension118(b,x,mask,K=None,zero_mask=None):
    cols,frm,_=extension_columns(K);linear_inplace(b,x,cols);mt=classical_transpose(b,mask,frm);phase_inverse24_canonical118(b,x,mt,zero_mask);b.free(mt);linear_inplace(b,x,frm)
