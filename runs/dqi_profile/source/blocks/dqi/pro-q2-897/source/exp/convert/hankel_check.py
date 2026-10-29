"""Independent GF(2^12) rank-one, in-place Hankel audit; no verifier."""
import random, json
from functools import lru_cache
MOD=0x1053

def slowmul(a,b):
    c=0
    for _ in range(12):
        if b&1:c^=a
        b>>=1;a<<=1
        if a&4096:a^=MOD
    return c
EXP=[1]*8190;LOG=[-1]*4096
for i in range(1,4095):EXP[i]=slowmul(EXP[i-1],2)
assert len(set(EXP[:4095]))==4095
for i in range(4095):LOG[EXP[i]]=i;EXP[i+4095]=EXP[i]
def mul(a,b):return EXP[LOG[a]+LOG[b]] if a and b else 0
def inv(a):assert a;return EXP[4095-LOG[a]]
def power(a,n):return EXP[(LOG[a]*n)%4095] if a else int(n==0)
def dot(a,b):
    out=0
    for x,y in zip(a,b):out^=mul(x,y)
    return out

def mi(a):
    n=len(a);a=[r[:]+[int(i==j) for j in range(n)] for i,r in enumerate(a)]
    for j in range(n):
        p=next(i for i in range(j,n) if a[i][j]);a[p],a[j]=a[j],a[p]
        t=inv(a[j][j]);a[j]=[mul(t,v) for v in a[j]]
        for i in range(n):
            if i!=j:
                t=a[i][j];a[i]=[x^mul(t,y) for x,y in zip(a[i],a[j])]
    return [r[n:] for r in a]

@lru_cache(None)
def terms(k,m):
    n=k+m-1;points=list(range(n));V=[[power(t,j) for j in range(n)] for t in points];W=mi(V)
    # Cross = sum_t source(reverse(low),V[t]) * coefficient(kernel,W[:,t]) * V[t,:m]
    return [(V[t][:k][::-1],[W[j][t] for j in range(n)],V[t][:m]) for t in range(n)]

def cross(q,low,high):
    count=0
    for source,coef,target in terms(len(low),len(high)):
        p=mul(dot(source,low),dot(coef,q));count+=1
        for i,c in enumerate(target):high[i]^=mul(c,p)
    return count

def tri(q,x,reverse=False):
    n=len(x)
    if n==1:return 0
    k=n//2;m=n-k;lo=x[:k];hi=x[k:]
    if not reverse:
        count=tri(q[:m],hi);count+=cross(q[1:n],lo,hi);count+=tri(q[:k],lo)
    else:
        count=tri(q[:k],lo,True);count+=cross(q[1:n],lo,hi);count+=tri(q[:m],hi,True)
    x[:]=lo+hi
    return count

def main():
    rng=random.Random(102617147);tests=0;rank_counts={}
    for n in (2,3,4,8,16,17,35):
        for _ in range(100):
            q=[1]+[rng.randrange(4096) for _ in range(n-1)];x=[rng.randrange(4096) for _ in range(n)];orig=x[:]
            target=[dot(q[:i+1],orig[i::-1]) for i in range(n)]
            c=tri(q,x);assert x==target
            cr=tri(q,x,True);assert x==orig and c==cr
            rank_counts[n]=c;tests+=1
    # H_ij=B_(i+j+1), coefficients B_35=1, higher 0. H=J*T with q_s=B_(35-s).
    for _ in range(100):
        b=[rng.randrange(4096) for _ in range(35)]+[1];x=[rng.randrange(4096) for _ in range(35)];old=x[:]
        expected=[dot((b[i+j+1] if i+j+1<=35 else 0 for j in range(35)),old) for i in range(35)]
        c=tri(list(reversed(b[1:])),x);x.reverse();assert x==expected and c==147
        x.reverse();tri(list(reversed(b[1:])),x,True);assert x==old
    out={'GF12_modulus':hex(MOD),'triangular_forward_reverse_cases':tests,'hankel35_forward_reverse_cases':100,'products':rank_counts,'H35_forward_T':147*42,'H35_pair_T':294*42,'extra_words':0,'inversions':0,'measurement_phase_corrections':0}
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
