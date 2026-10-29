"""Width-preserving20T GF64 inverse using the lambda2 norm isotopy.

Initialized output reuses three redundant norm directions as output wires.
No core modules are edited or monkeypatched.
"""
try:
    from . import inverse20 as base
    from .inverse20 import *
except ImportError:
    import inverse20 as base
    from inverse20 import *
DAG=json.loads(Path(__file__).with_name('optimized_isotopy2.json').read_text())
EXTRA_ROWS=[11,52,34]
NORM_ROWS=[42,49,54]
assert [(e>>7)&63 for e in DAG['outputs']]==[11,0,52,34,52,52]
assert all(not((l|r)&(63<<7))for l,r in DAG['gates'])

def raw_forms():
    f=field(3);scale=2;inv=f.invert(scale);forms=[]
    for mask,_ in tensor(3):
        left=sum(((mask&f.mul(scale,1<<i)).bit_count()&1)<<(1+i)for i in range(3))
        right=sum(((mask&f.mul(inv,1<<i)).bit_count()&1)<<(4+i)for i in range(3))
        forms.append((left,right))
    return forms

def xw_form(mask):return(mask&127)|((mask>>6)&(7<<7))

def erase_extras(b,x,E):
    masks=b.alloc_bits(3);b.hmr(E,masks);b.free(E)
    for i,(left,right)in enumerate(raw_forms()):
        with expose_pair(b,x,left,right)as(p,q):
            for j,row in enumerate(EXTRA_ROWS):
                if row>>i&1:b.ccz(p,q,masks[j])
    b.free(masks)

def inverse6_xor20(b,x,z=None,keep_norm=False):
    """20T exact inverse6: fresh-output Q15; arbitrary-target Q18.

    keep_norm=True returns(input_norm3) too, initialized Q18. All private
    erasures are explicit0T quadratic phase fixes. Zero maps to zero.
    """
    assert len(x)==6;fresh=z is None
    to,frm,K=tower(6);linear_inplace(b,x,to)
    if not fresh:linear_inplace(b,z,to)
    g=b.alloc_qubits(6)
    for i,(left,right)in enumerate(raw_forms()):
        with expose_pair(b,x,left,right)as(p,q):b.init_and(p,q,g[i])
    rows=NORM_ROWS+EXTRA_ROWS
    cols=[sum(((r>>j)&1)<<i for i,r in enumerate(rows))for j in range(6)]
    linear_inplace(b,g,cols);N,E=g[:3],g[3:]
    for j,form in enumerate(norm_forms(K)):form_xor(b,x,form&127,N[j])
    if not fresh:
        b.cx(E[0],z[0]);b.cx(E[1],[z[2],z[4],z[5]]);b.cx(E[2],z[3])
        erase_extras(b,x,E)
    w=inverse3_noscratch(b,N)
    if fresh:
        if not keep_norm:ar._erase_norm(b,x,N,K);b.free(N)
        extra=b.alloc_qubits(3)
        # Fixed physical layout: these data wires are now output coordinates.
        z=[E[0],extra[0],E[1],E[2],extra[1],extra[2]]
        b.cx(z[2],[z[4],z[5]])
    source=list(x)+list(w)
    for j,e in enumerate(DAG['outputs']):form_xor(b,source,xw_form(e),z[j])
    for i,(left,right)in enumerate(DAG['gates']):
        vector=sum(((e>>(16+i))&1)<<j for j,e in enumerate(DAG['outputs']))
        with expose_pair(b,source,xw_form(left),xw_form(right))as(p,q):rankone_xor(b,p,q,z,vector)
    if fresh:
        ar._erase_norm(b,z,w,K);b.free(w)
    else:
        erase_inverse(b,N,w);b.free(w)
        if not keep_norm:ar._erase_norm(b,x,N,K);b.free(N)
    linear_inplace(b,x,frm);linear_inplace(b,z,frm)
    return(z,N)if keep_norm else z

# These higher-level schedules are unchanged structurally. Rebind their private
# globals to this plugin's inverse6 implementation; no imported module changes.
from types import FunctionType
for _name in ('inverse_initialized','inverse12_keep_norm','inverse_inplace_forward20',
              'inverse_inplace_reverse20','inverse24_forward20','inverse24_reverse20'):
    _original=getattr(base,_name)
    globals()[_name]=FunctionType(_original.__code__,globals(),_name,_original.__defaults__,_original.__closure__)
