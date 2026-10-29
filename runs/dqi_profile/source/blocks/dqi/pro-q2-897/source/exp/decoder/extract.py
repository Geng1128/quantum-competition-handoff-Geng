"""In-place conjugate-node initial map and triangular EXTRACT."""
from functools import lru_cache
from common import *
from exp.inverse12_escape.joint import inverse24_forward190 as inverse24_forward20,inverse24_reverse209 as inverse24_reverse20

@lru_cache(None)
def constants(h=35):
    cs=list(range(0,2*h,2));ks=[mul(c,c)^c^K for c in cs];P=[1]
    for k in ks:P=pmul(P,[k,1,1])
    matrix=[]
    for c in cs:
        g=c|(1<<12);Q=[0]*(2*h);Q[-1]=1
        for j in reversed(range(2*h-1)):Q[j]=P[j+1]^emul(g,Q[j+1])
        assert P[0]^emul(g,Q[0])==0
        matrix.append(tuple(v&4095 for v in Q));matrix.append(tuple(v>>12 for v in Q))
    return cs,ks,P,tuple(matrix)

def initial(b,words,reverse=False):
    linear_words(b,words,constants(len(words)//2)[3],reverse=reverse)

def extract(b,words,reverse=False,transcript=None):
    h=len(words)//2;cs,ks,_,_=constants(h);bank=[combine(words[2*j:2*j+2]) for j in range(h)]
    if not reverse:
        history=[]
        for i in range(h):
            cx(b,words[2*i+1],words[2*i],cs[i])
            for j in range(i+1,h):
                b.cx(words[2*i],words[2*j]);cx(b,words[2*i+1],words[2*j],cs[j]);b.cx(words[2*i+1],words[2*j+1])
                history.append(inverse24_forward20(b,bank[j],K))
                scalepair(b,bank[j],ks[i]^ks[j])
        return history
    history=list(transcript)
    for i in reversed(range(h)):
        for j in reversed(range(i+1,h)):
            scalepair(b,bank[j],inv(ks[i]^ks[j]))
            inverse24_reverse20(b,bank[j],history.pop(),K)
            b.cx(words[2*i+1],words[2*j+1]);cx(b,words[2*i+1],words[2*j],cs[j]);b.cx(words[2*i],words[2*j])
        cx(b,words[2*i+1],words[2*i],cs[i])
    assert not history
