"""Classical-mask inverse6 phase:<=6executedT/5scratch; exact64pointSLP.
Classicalmask-dependent constantfieldscaling maps63nonzero components toone.
"""
from exp.phase24_tradeoff.kernel import *
import json
P6=json.loads(Path(__file__).with_name('phase6_six_search.json').read_text())['candidate']

def classical_equal(b,x,value):
    chain=b.alloc_bits(len(x)+1);b.bit_store(chain,False);b.x(chain[0])
    for j,q in enumerate(x):
        if not(value>>j&1):b.x(q)
        b.push_condition(chain[j]);b.bit_store(chain[j+1],True,control=q);b.pop_condition()
        if not(value>>j&1):b.x(q)
    b.free(chain[:-1]);return chain[-1]


def fixed_phase6(b,x,do_inverse=True,zero_mask=None):
    to,frm,K=tower(6);linear_inplace(b,x,to);ng=len(P6['raw_forms']);g=b.alloc_qubits(ng)
    for i,(l,r)in enumerate(P6['raw_forms']):
        with ref.expose_pair(b,x,l,r)as(p,q):b.init_and(p,q,g[i])
    w=P6['w'];base=[]
    for j in range(ng):
        if rank(base+[w,1<<j])==len(base)+2:base.append(1<<j)
    B=base+[w];assert len(B)==ng;iv=inverse_columns(B);linear_inplace(b,g,iv)
    def form(v):
        return(v&63)|sum((((v>>6)&co).bit_count()&1)<<(6+j)for j,co in enumerate(B))
    p,q=form(P6['p']),form(P6['q']);assert not(((p|q)>>(6+ng-1))&1);source=list(x)+list(g[:ng-1])
    with ref.expose_pair(b,source,p<<1,q<<1)as(a,c):b.ccx(a,c,g[ng-1])
    linear_inplace(b,g,B);S=list(x)+list(g)
    for term in(P6['terms']if do_inverse else[]):
        if len(term)==1:b.z(S[term[0]])
        else:b.cz(S[term[0]],S[term[1]])
    if zero_mask is not None:
        b.z(zero_mask);b.push_condition(zero_mask)
        for term in P6['zero_terms']:
            if len(term)==1:b.z(S[term[0]])
            else:b.cz(S[term[0]],S[term[1]])
        b.pop_condition()
    linear_inplace(b,g,iv);hm=b.alloc_bits(1)[0];b.hmr(g[ng-1],hm)
    with ref.expose_pair(b,source,p<<1,q<<1)as(a,c):b.ccz(a,c,hm)
    rows=[sum(((iv[i]>>j)&1)<<i for i in range(ng))for j in range(ng)]
    for i,(l,r)in enumerate(P6['raw_forms']):
        if rows[ng-1]>>i&1:
            with ref.expose_pair(b,x,l,r)as(a,c):b.ccz(a,c,hm)
    b.free(hm);b.free(g[ng-1]);mm=b.alloc_bits(ng-1);b.hmr(g[:ng-1],mm);b.free(g[:ng-1])
    for i,(l,r)in enumerate(P6['raw_forms']):
        with ref.expose_pair(b,x,l,r)as(a,c):
            for j in range(ng-1):
                if rows[j]>>i&1:b.ccz(a,c,mm[j])
    b.free(mm);linear_inplace(b,x,frm)


def mask_factors():
    f=field(6);out={}
    for s in range(1,64):
        si=f.invert(s);m=sum(((f.mul(si,1<<j)>>P6['output_bit'])&1)<<j for j in range(6));assert m not in out;out[m]=s
    return out
FACTORS=mask_factors()


def phase_inverse6_classical6(b,x,mask,zero_mask=None):
    """Exactmask·inv0(x),0→0,<=6executedT/5scratch,arbitraryclassicalmask."""
    for m,s in sorted(FACTORS.items()):
        eq=classical_equal(b,mask,m);b.push_condition(eq);co=[field(6).mul(s,1<<j)for j in range(6)];linear_inplace(b,x,co);fixed_phase6(b,x,zero_mask=zero_mask);linear_inplace(b,x,inverse_columns(co));b.pop_condition();b.free(eq)

    if zero_mask is not None:
        eq=classical_equal(b,mask,0);b.push_condition(eq);b.push_condition(zero_mask);fixed_phase6(b,x,do_inverse=False,zero_mask=zero_mask);b.pop_condition();b.pop_condition();b.free(eq)
