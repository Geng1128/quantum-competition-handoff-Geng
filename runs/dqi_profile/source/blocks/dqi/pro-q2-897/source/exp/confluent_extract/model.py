"""Singleton-first confluent quadratic-node map and zero-safe jet EXTRACT.

All polynomial/rational references are classical. No decoder/verifier claim.
"""
from functools import lru_cache
from pathlib import Path
import sys,json,random,time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'playbook/exp/cf_check'))
from audit import Field
import audit as cf_audit
f=Field(12,0x1053);assert f.K==2048

def eadd(a,b):return a^b
def esquare(a):return f.emul(a,a)
def ehead(a,b,c):return (a^f.mul(b,c))|(b<<12)
def coords(u,c):
    b=u>>12;return (u&4095)^f.mul(b,c),b
def get(bank,j):return bank[j]|(bank[j+1]<<12)
def put(bank,j,x):bank[j]=x&4095;bank[j+1]=x>>12

def synthetic(poly,gamma):
    assert len(poly)>=2
    Q=[0]*(len(poly)-1);Q[-1]=poly[-1]
    for i in reversed(range(len(Q)-1)):Q[i]=poly[i+1]^f.emul(gamma,Q[i+1])
    return Q,poly[0]^f.emul(gamma,Q[0])

@lru_cache(None)
def constants(h=35):
    assert h%2 and h>=1;d=(h-1)//2
    groups=[dict(c=0,kappa=f.K,gamma=4096,head=0,word=0,double=False)]
    headcs=[0]
    for j in range(1,d+1):
        c=2*j;groups.append(dict(c=c,kappa=f.mul(c,c)^c^f.K,gamma=c|4096,head=2*j-1,word=4*j-2,double=True));headcs.extend([c,c])
    kappas=[f.mul(c,c)^c^f.K for c in headcs];P=[1]
    for kap in kappas:P=f.pmul(P,[kap,1,1])
    matrix=[]
    for group in groups:
        Q,rem=synthetic(P,group['gamma']);assert rem==0
        matrix.append([q&4095 for q in Q]);matrix.append([q>>12 for q in Q])
        if group['double']:
            QQ,rem=synthetic(Q,group['gamma']);assert rem==0
            QQ=QQ+[0]
            matrix.append([q&4095 for q in QQ]);matrix.append([q>>12 for q in QQ])
    assert len(matrix)==2*h and all(len(r)==2*h for r in matrix)
    return tuple(headcs),tuple(kappas),tuple(P),tuple(tuple(r)for r in matrix),tuple(groups)

@lru_cache(None)
def inverse_matrix(h=35):
    M=constants(h)[3];n=len(M);a=[list(row)+[int(i==j)for j in range(n)]for i,row in enumerate(M)]
    for j in range(n):
        p=next(i for i in range(j,n)if a[i][j]);a[j],a[p]=a[p],a[j]
        iv=f.inv(a[j][j]);a[j]=[f.mul(v,iv)for v in a[j]]
        for i in range(n):
            if i!=j and a[i][j]:
                q=a[i][j];a[i]=[x^f.mul(q,y)for x,y in zip(a[i],a[j])]
    assert all(a[i][j]==int(i==j)for i in range(n)for j in range(n))
    return tuple(tuple(row[n:])for row in a)

def matrix_apply(M,s):
    out=[]
    for row in M:
        v=0
        for a,x in zip(row,s):v^=f.mul(a,x)
        out.append(v)
    return out

def initial(s,h=35,reverse=False):return matrix_apply(inverse_matrix(h)if reverse else constants(h)[3],s)

@lru_cache(None)
def plan(h=35):
    groups=constants(h)[4];ops=[]
    for i,g in enumerate(groups):
        ops.append(dict(kind='coordinates',head=g['head'],c=g['c']))
        for target in groups[i+1:]:ops.append(dict(kind='jet',head=g['head'],target=target['word'],c=target['c'],q=g['kappa']^target['kappa']))
        if g['double']:
            ops.append(dict(kind='local',head=g['head'],c=g['c']))
            for target in groups[i+1:]:ops.append(dict(kind='jet',head=g['head']+1,target=target['word'],c=target['c'],q=g['kappa']^target['kappa']))
    assert sum(op['kind']=='jet'for op in ops)==((h-1)//2)**2
    assert sum(op['kind']=='local'for op in ops)==(h-1)//2
    return tuple(ops)

def J(u,v):
    z=int(u==0);ui=f.einv(u);return ui,f.emul(v,esquare(ui^z))

def extract(bank,h=35,reverse=False,events=None):
    bank=list(bank);assert len(bank)==2*h
    for op in (reversed(plan(h))if reverse else plan(h)):
        if op['kind']=='coordinates':bank[2*op['head']]^=f.mul(op['c'],bank[2*op['head']+1]);continue
        if op['kind']=='local':
            i=2*op['head'];j=i+2
            if not reverse:
                u=get(bank,j)^bank[i+1]
                if events is not None and not u:events.append(dict(kind='local_zero_inverse',head=op['head']+1))
                a,b=coords(f.einv(u),op['c']);bank[j:j+2]=[a,b]
            else:
                a,b=bank[j:j+2];put(bank,j,f.einv(ehead(a,b,op['c']))^bank[i+1])
            continue
        i=2*op['head'];j=op['target'];a,b=bank[i:i+2];u=get(bank,j);v=get(bank,j+2);q=op['q'];assert q
        if not reverse:
            u^=ehead(a,b,op['c']);v^=b
            if events is not None and not u:events.append(dict(kind='jet_zero_value',head=op['head'],target=j//2,derivative_nonzero=bool(v)))
            u,v=J(u,v);v=u^f.escale(v,q);u=f.escale(u,q)
        else:
            u=f.escale(u,f.inv(q));v=f.escale(v^u,f.inv(q));u,v=J(u,v);u^=ehead(a,b,op['c']);v^=b
        put(bank,j,u);put(bank,j+2,v)
    return bank

def pair_from_residues(roots,weights):
    D=[1]
    for root in roots:D=f.pmul(D,[root,1])
    N=[0]
    for root,w in zip(roots,weights):
        Q,rem=f.pdiv(D,[root,1]);assert rem==[0];N=f.add(N,f.scale(Q,w))
    return N,D

def instance(h,roots,weights):
    P=constants(h)[2];syndrome=[0]*(2*h);errors=[]
    for root,w in zip(roots,weights):
        pb=f.ev(P,root);assert pb and not pb>>12
        start=f.div(w,pb);errors.append(f.div(start,root));v=start
        for j in range(2*h):syndrome[j]^=v;v=f.mul(v,root)
    N,D=pair_from_residues(roots,weights)
    return dict(h=h,roots=roots,residues=weights,errors=errors,syndrome=syndrome,N=N,D=D)

def direct_jets(roots,weights,h=35):
    out=[]
    for group in constants(h)[4]:
        u=v=0
        for root,w in zip(roots,weights):
            t=f.einv(group['gamma']^root);u^=f.escale(t,w);v^=f.escale(esquare(t),w)
        out.extend([u&4095,u>>12])
        if group['double']:out.extend([v&4095,v>>12])
    return out

def polynomial_oracle(N,D,h=35):
    cs,ks,*_=constants(h);heads=[];term=None
    for i,(c,kap)in enumerate(zip(cs,ks)):
        gamma=c|4096;de=f.ev(D,gamma)
        if not de:return dict(heads=heads,termination=term,pole=i)
        u=f.emul(f.ev(N,gamma),f.einv(de));a,b=coords(u,c);heads.extend([a,b])
        H=f.add(N,f.pmul([a,b],D));Q,rem=f.pdiv(H,[kap,1,1]);assert rem==[0]
        if H==[0]:term=i;heads.extend([0]*(2*h-len(heads)));break
        N,D=D,Q
    return dict(heads=heads,termination=term,pole=None)

def trial(case):
    h=case['h'];s=case['syndrome'];jet=initial(s,h);assert jet==direct_jets(case['roots'],case['residues'],h)
    assert initial(jet,h,True)==s
    events=[];heads=extract(jet,h,events=events);assert extract(heads,h,True)==jet
    reference=polynomial_oracle(case['N'],case['D'],h)
    match=heads==reference['heads']if reference['pole']is None else None
    raw_ratio=None;raw_padding=None;raw_squarefree=None
    if reference['pole']is None:
        assert match
        pairs=[heads[i:i+2]for i in range(0,len(heads),2)]
        A,B,tail=cf_audit.synth(f,constants(h)[1],pairs)
        raw_ratio=f.pmul(A,case['D'])==f.pmul(B,case['N']);assert raw_ratio
        expected=case['D'][:]
        for i in range(len(case['roots'])+1,h,2):expected=f.pmul(expected,[constants(h)[1][i],1,1])
        if not len(case['roots'])%2:expected=[0]+expected
        raw_padding=B==expected;assert raw_padding
        raw_squarefree=f.gcd(B,f.derivative(B))==[1];assert raw_squarefree
        assert len(B)==h+1 and B[-1]==1
    else:assert heads[:len(reference['heads'])]==reference['heads']
    return dict(h=h,weight=len(case['roots']),oracle_pole=reference['pole'],termination=reference['termination'],
        heads_match=match,jet_zero_events=sum(e['kind']=='jet_zero_value'for e in events),
        nonzero_derivative_zero_events=sum(e['kind']=='jet_zero_value'and e['derivative_nonzero']for e in events),
        local_zero_inverses=sum(e['kind']=='local_zero_inverse'for e in events),roundtrip=True,
        raw_synthesis_ratio=raw_ratio,raw_synthesis_padding=raw_padding,raw_synthesis_squarefree=raw_squarefree)

def main(repeats=4):
    start=time.time();rng=random.Random(202610270901);tests=[];arbitrary=[]
    for h in[3,9,35]:
        inverse_matrix(h)
        for weight in range(h+1):
            for repeat in range(repeats):
                roots=rng.sample(range(1,4096),weight);weights=[rng.randrange(1,4096)for _ in roots]
                tests.append(trial(instance(h,roots,weights)))
        for j in range(300):
            bank=[rng.randrange(4096)for _ in range(2*h)]
            if j%3==0:
                for group in constants(h)[4][1:]:bank[group['word']]=bank[group['word']+1]=0
            if j%3==1:bank=[0 if rng.randrange(2)else v for v in bank]
            assert extract(extract(bank,h),h,True)==bank
        arbitrary.append(dict(h=h,cases=300,includes_zero_values_and_zero_jets=True))
        print('h',h,'done',flush=True)
    peer=json.loads((ROOT/'exp/confluent_synthesis/fixtures.json').read_text());compared=0
    for fixture in peer:
        h=fixture['h'];case=instance(h,fixture['roots'],fixture['weights']);heads=extract(initial(case['syndrome'],h),h)
        assert heads==[v for head in fixture['heads']for v in head];compared+=1
        A,B,_=cf_audit.synth(f,constants(h)[1],[heads[i:i+2]for i in range(0,len(heads),2)])
        assert A==fixture['rawA']and B==fixture['rawB']
    result=dict(tests=tests,arbitrary=arbitrary,peer_fixture_heads_compared=compared,
        matrix_ranks={h:2*h for h in[3,9,35]},counts={h:dict(jets=sum(o['kind']=='jet'for o in plan(h)),local_inverses=sum(o['kind']=='local'for o in plan(h)))for h in[3,9,35]},
        seconds=time.time()-start,meaning='Classical exactness and fixed-linear map; no full decoder/verifier')
    print(json.dumps(dict(cases=len(tests),heads_bad=sum(t['heads_match']is False for t in tests),poles=sum(t['oracle_pole']is not None for t in tests),
        peer_fixture_heads_compared=compared,arbitrary=arbitrary,seconds=result['seconds']),indent=2))
    Path(__file__).with_name('results.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
