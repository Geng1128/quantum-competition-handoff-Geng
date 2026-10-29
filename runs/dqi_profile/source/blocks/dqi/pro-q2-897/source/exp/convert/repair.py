"""Adaptive one-gap Barnett repair for raw Bezout solve.

Separate fromdefaultgenericconvert.py. Firstleading-zeroisrepairedbymultiplygbyx
andprecompanionMx. More/largegapsandrepairwithf(0)=0decodeincorrectlybut
forward/reversearestilllocallyphase-correctpermutations. No deferredtranscripts.
"""
from contextlib import contextmanager
from convert import ar,zero_flag,erase_zero_flag,scale,pre,post,raw_solve,frob
from lowwidth import phase_inverse36
from kernels import hankel

@contextmanager
def nonzero_inverse(b,a):
    ai=ar.inverse(b,a)
    yield ai
    mask=b.alloc_bits(12);b.hmr(ai,mask);b.free(ai);phase_inverse36(b,a,mask);b.free(mask)

@contextmanager
def safe_word(b,a):
    flag=zero_flag(b,a);b.cx(flag,a[0])
    yield
    b.cx(flag,a[0]);erase_zero_flag(b,a,flag)

def rotate(b,v,F,reverse=False):
    for j in (range(1,len(v)) if reverse else range(len(v)-1,0,-1)):
        b.cswap(F,v[j],v[j-1])

def coefficient_update(b,f,u,target):
    if f[-1] is None:b.x(f[0][0])
    else:b.cx(f[-1],f[0])
    ar.mul_xor(b,f[0],u,target)
    if f[-1] is None:b.x(f[0][0])
    else:b.cx(f[-1],f[0])

def companion(b,f,payloads,F,reverse=False):
    """Conditionallymultiplyeachpayloadbyx modf, oritsinverse.

    f0 andleadingaresafenedonlyduringthispermutation, soalwaysinvertible.
    Masku=F*lowest/bbeforeupdates; afterupdatesu=F*lowest/f0.
    """
    lead=f[-1]
    @contextmanager
    def lead_safe():
        if lead is None:yield
        else:
            with safe_word(b,lead):yield
    @contextmanager
    def invctx(a):
        if a is None:yield None
        else:
            with nonzero_inverse(b,a) as ai:yield ai
    with safe_word(b,f[0]),lead_safe():
        if not reverse:
            for v in payloads:rotate(b,v,F)
        masks=[]
        divisor=f[0] if reverse else lead
        with invctx(divisor) as di:
            for v in payloads:
                u=b.alloc_qubits(12)
                for j in range(12):b.init_and(F,v[0][j],u[j])
                if divisor is not None:scale(b,u,divisor,di,True)
                if not reverse:
                    for i in range(1,len(v)):ar.mul_xor(b,f[i],u,v[i])
                    coefficient_update(b,f,u,v[0])
                else:
                    coefficient_update(b,f,u,v[0])
                    for i in reversed(range(1,len(v))):ar.mul_xor(b,f[i],u,v[i])
                mask=b.alloc_bits(12);b.hmr(u,mask);b.free(u);masks.append(mask)
        recovery=lead if reverse else f[0]
        with invctx(recovery) as ri:
            for v,mask in zip(payloads,masks):
                if recovery is None:
                    for j in range(12):b.ccz(F,v[0][j],mask[j])
                else:
                    tmp=b.alloc_qubits(12);ar.mul_xor(b,v[0],ri,tmp)
                    for j in range(12):b.ccz(F,tmp[j],mask[j])
                    ar.erase_product(b,v[0],ri,tmp);b.free(tmp)
                b.free(mask)
        if reverse:
            for v in payloads:rotate(b,v,F,True)

def equal_code(b,H,code):
    for i,q in enumerate(H):
        if code>>i&1:b.x(q)
    flag=zero_flag(b,H)
    for i,q in enumerate(H):
        if code>>i&1:b.x(q)
    return flag

def erase_equal(b,H,code,flag):
    for i,q in enumerate(H):
        if code>>i&1:b.x(q)
    erase_zero_flag(b,H,flag)
    for i,q in enumerate(H):
        if code>>i&1:b.x(q)

def history_arm(b,H,a,code):
    hz=zero_flag(b,H);az=zero_flag(b,a);F=b.alloc_qubits(1)[0];b.ccx(hz,az,F)
    erase_zero_flag(b,a,az);erase_zero_flag(b,H,hz)
    for i,q in enumerate(H):
        if code>>i&1:b.cx(F,q)
    return F

def history_clear_flag(b,H,code,F):
    eq=equal_code(b,H,code);b.cx(eq,F);erase_equal(b,H,code,eq);b.free(F)

def history_finish(b,H,a,code,F):
    for i,q in enumerate(H):
        if code>>i&1:b.cx(F,q)
    hz=zero_flag(b,H);az=zero_flag(b,a);b.ccx(hz,az,F)
    erase_zero_flag(b,a,az);erase_zero_flag(b,H,hz);b.free(F)

def repair_solve(b,f,g,payloads,H,reverse=False):
    m=len(f)-1
    if m==1:return raw_solve(b,f,g,payloads,reverse)
    F=history_arm(b,H,g[-1],m)
    rotate(b,g,F)
    if not reverse:companion(b,f,payloads,F)
    history_clear_flag(b,H,m,F)
    low=[v[:-1]for v in payloads]
    if not reverse:
        pre(b,f,g,payloads);repair_solve(b,g,f[:-2],low,H);post(b,f,g,payloads)
    else:
        post(b,f,g,payloads,True);repair_solve(b,g,f[:-2],low,H,True);pre(b,f,g,payloads,True)
    F=equal_code(b,H,m)
    if reverse:companion(b,f,payloads,F,True)
    rotate(b,g,F,True)
    history_finish(b,H,g[-1],m,F)

def bez_repair(b,O,E,payloads,reverse=False):
    def reduce():
        for i in range(len(O)):ar.mul_xor(b,E[-1],O[i],E[i])
    reduce();H=list(b.alloc_qubits(5));repair_solve(b,O+[None],E[:-1],payloads,H,reverse);b.free(H);reduce()

def convert_repair(b,A,B,reverse=False):
    assert len(A)==len(B)==35
    E=B[::2];O=B[1::2];u=A[1::2];w=A[:34:2];top=A[-1]
    if reverse:hankel(b,B,A,True)
    frob(b,B,2048);frob(b,A,2048)
    if reverse:
        for i in range(17):b.swap(A[2*i],A[2*i+1])
        for i in range(17):ar.mul_xor(b,O[i],u[i],top)
        bez_repair(b,O,E,[u,w],True)
        for i in range(17):ar.mul_xor(b,top,O[i],w[i])
    else:
        for i in range(17):ar.mul_xor(b,top,O[i],w[i])
        bez_repair(b,O,E,[u,w])
        for i in range(17):ar.mul_xor(b,O[i],u[i],top)
        for i in range(17):b.swap(A[2*i],A[2*i+1])
    frob(b,A,2);frob(b,B,2)
    if not reverse:hankel(b,B,A)
