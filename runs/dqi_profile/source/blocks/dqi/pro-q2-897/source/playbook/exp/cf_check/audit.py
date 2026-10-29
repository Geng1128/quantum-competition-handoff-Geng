"""Standalone classical finite-field EXTRACT/SYNTHESIZE audit (no circuits/verifier)."""
import argparse, json, random, time
from collections import Counter

class Field:
    def __init__(self, m, modulus):
        self.m=m; self.n=1<<m; self.mask=self.n-1
        self.exp=[]; self.log=[-1]*self.n; x=1
        for i in range(self.n-1):
            assert self.log[x]<0
            self.log[x]=i; self.exp.append(x)
            x<<=1
            if x&self.n:x^=modulus
        assert x==1
        self.exp*=2
        self.K=next(x for x in range(1,self.n) if self.trace(x)==1)
    def mul(self,a,b):return self.exp[self.log[a]+self.log[b]] if a and b else 0
    def inv(self,a):return self.exp[self.n-1-self.log[a]] if a else 0
    def div(self,a,b):return self.mul(a,self.inv(b))
    def trace(self,a):
        t=0
        for _ in range(self.m):t^=a;a=self.mul(a,a)
        return t
    def emul(self,u,v):
        a=u&self.mask;b=u>>self.m;c=v&self.mask;d=v>>self.m
        bd=self.mul(b,d)
        return (self.mul(a,c)^self.mul(self.K,bd)) | ((self.mul(a,d)^self.mul(b,c)^bd)<<self.m)
    def einv(self,u):
        a=u&self.mask;b=u>>self.m
        ni=self.inv(self.mul(a,a)^self.mul(a,b)^self.mul(self.K,self.mul(b,b)))
        return self.mul(a^b,ni)|(self.mul(b,ni)<<self.m)
    def escale(self,u,a):return self.mul(u&self.mask,a)|(self.mul(u>>self.m,a)<<self.m)
    def ev(self,p,u):
        out=0
        for a in reversed(p):out=self.emul(out,u)^a
        return out
    def trim(self,p):
        while len(p)>1 and not p[-1]:p.pop()
        return p
    def add(self,a,b):
        c=[0]*max(len(a),len(b))
        for i,x in enumerate(a):c[i]^=x
        for i,x in enumerate(b):c[i]^=x
        return self.trim(c)
    def pmul(self,a,b):
        c=[0]*(len(a)+len(b)-1)
        for i,x in enumerate(a):
            for j,y in enumerate(b):c[i+j]^=self.mul(x,y)
        return self.trim(c)
    def scale(self,p,c):return self.trim([self.mul(x,c) for x in p])
    def pdiv(self,a,b):
        a=a.copy();out=[0]*max(1,len(a)-len(b)+1);bi=self.inv(b[-1])
        assert b!=[0]
        while a!=[0] and len(a)>=len(b):
            d=len(a)-len(b);v=self.mul(a[-1],bi);out[d]=v
            for i,x in enumerate(b):a[i+d]^=self.mul(v,x)
            self.trim(a)
        return self.trim(out),a
    def gcd(self,a,b):
        while b!=[0]:a,b=b,self.pdiv(a,b)[1]
        return self.scale(a,self.inv(a[-1]))
    def derivative(self,p):return self.trim([p[i] if i%2 else 0 for i in range(1,len(p))] or [0])

def data(f, cs, betas, ws):
    kappas=[f.mul(c,c)^c^f.K for c in cs]
    nodes=[c|f.n for c in cs]
    D=[1]
    for b in betas:D=f.pmul(D,[b,1])
    N=[0]
    for b,w in zip(betas,ws):
        quotient,rem=f.pdiv(D,[b,1]);assert rem==[0]
        N=f.add(N,f.scale(quotient,w))
    vals=[f.emul(f.ev(N,g),f.einv(f.ev(D,g))) for g in nodes]
    return kappas,nodes,N,D,vals

def extract(f,cs,kappas,vals):
    vals=vals.copy();heads=[];zeros=[]
    for i,c in enumerate(cs):
        b=vals[i]>>f.m;a=(vals[i]&f.mask)^f.mul(b,c);heads.append((a,b))
        for j in range(i+1,len(cs)):
            u=vals[j]^a^f.mul(b,cs[j])^(b<<f.m)
            if not u:zeros.append((i,j))
            vals[j]=f.escale(f.einv(u),kappas[i]^kappas[j])
    return heads,zeros

def synth(f,kappas,heads):
    p,q,d=0,1,0
    for a,b in heads:p,q,d=q,p^f.mul(b,q),d^f.mul(p,q)^f.mul(a^b,f.mul(q,q))
    tail=(p,q,d)
    if q:A,B=[f.div(d,q),p],[q]
    else:A,B=[p],[0]
    for (a,b),kappa in reversed(list(zip(heads,kappas))):
        A,B=f.add(f.pmul([a,b],A),f.pmul([kappa,1,1],B)),A
    if not q:A,B=[0]+A,[0]+B
    return f.trim(A),f.trim(B),tail

def rational_audit(f,kappas,nodes,N,D,heads):
    """Follow exact rational transforms until the first genuine pole or termination."""
    first_poles=[];termination=None
    for i,(a,b) in enumerate(heads):
        if f.ev(D,nodes[i])==0:return {'head_pole':i,'poles':first_poles}
        assert f.ev(N,nodes[i])==f.emul(a^f.mul(b,nodes[i]&f.mask)^(b<<f.m),f.ev(D,nodes[i]))
        H=f.add(N,f.pmul([a,b],D))
        if H==[0]:termination=i;break
        H,rem=f.pdiv(H,[kappas[i],1,1]);assert rem==[0]
        poles=[j for j in range(i+1,len(nodes)) if f.ev(H,nodes[j])==0]
        if poles:return {'first_nonterminal_zero_step':i,'poles':poles,'residual_degree':len(H)-1}
        N,D=D,H
    return {'termination':termination,'poles':first_poles}

def validate(m,modulus,h,repeats,seed):
    f=Field(m,modulus);rng=random.Random(seed);cs=list(range(0,2*h,2));assert max(cs)<f.n
    results=Counter();examples=[]
    for t in range(h+1):
        for trial in range(repeats):
            betas=rng.sample(range(1,f.n),t);ws=[rng.randrange(1,f.n) for _ in betas]
            kappas,nodes,N,D,vals=data(f,cs,betas,ws)
            heads,zeros=extract(f,cs,kappas,vals);A,B,tail=synth(f,kappas,heads)
            ra=rational_audit(f,kappas,nodes,N,D,heads)
            correct=f.pmul(A,D)==f.pmul(B,N)
            sqfree=f.gcd(B,f.derivative(B))==[1]
            degree=len(B)-1
            result={'t':t,'trial':trial,'correct':correct,'degree':degree,'monic':B[-1]==1,'squarefree':sqfree,'tail':tail,'rational':ra,'zeros':len(zeros)}
            results['cases']+=1;results['correct']+=correct;results['monic']+=B[-1]==1;results['degree_h']+=degree==h;results['squarefree']+=sqfree
            if 'termination' in ra and ra['termination'] is not None:
                term=ra['termination'];results['terminated']+=1
                assert all(a==b==0 for a,b in heads[term+1:])
                results['later_heads_zero']+=1
            if not correct or not sqfree or degree!=h or not result['monic']:
                if len(examples)<12:examples.append(dict(result,betas=betas,ws=ws,heads=heads,A=A,B=B))
            if correct:
                pad,rem=f.pdiv(B,D);assert rem==[0]
                expected=D.copy()
                for j in range(t+1,h,2):expected=f.pmul(expected,[kappas[j],1,1])
                if not t%2:expected=[0]+expected
                assert expected==B,('padding index pattern',t,B,expected)
                allowed=[kappas[i] for i in range(h)]
                if not tail[1]:
                    pad,rem=f.pdiv(pad,[0,1]);assert rem==[0]
                for k in allowed:
                    quotient,rem=f.pdiv(pad,[k,1,1])
                    if rem==[0]:pad=quotient
                assert pad==[1],pad
                assert all(f.ev(B,b)!=0 for b in range(1,f.n) if b not in betas)
    return {'m':m,'K':f.K,'h':h,'repeats_per_weight':repeats,'counts':dict(results),'examples':examples}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--m',type=int,default=12);parser.add_argument('--h',type=int,default=35);parser.add_argument('--repeats',type=int,default=3);parser.add_argument('--seed',type=int,default=1)
    args=parser.parse_args();mods={3:0b1011,4:0b10011,6:0b1000011,12:0x1053}
    start=time.time();result=validate(args.m,mods[args.m],args.h,args.repeats,args.seed);result['seconds']=time.time()-start
    print(json.dumps(result,indent=2))
