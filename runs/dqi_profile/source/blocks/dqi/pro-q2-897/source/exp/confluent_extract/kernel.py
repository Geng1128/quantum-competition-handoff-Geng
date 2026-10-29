"""Private fixed Hermite map and injectable singleton-first jet adapter.

No stable decoder modules are edited. Pass the arithmetic teammate's jet
forward/reverse kernels; both operate on two24-bit data registers and return
or consume classical transcripts, with no retained quantum cache.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from exp.decoder.common import combine,cx,scalepair,linear_words,K
from exp.inverse12_escape.joint import inverse24_forward190,inverse24_reverse209
try:
    from .model import constants,plan
except ImportError:
    from model import constants,plan

def initial(b,words,reverse=False):
    assert len(words)%2==0
    linear_words(b,words,constants(len(words)//2)[3],reverse)

def translate(b,words,head,target,c):
    a,z=words[2*head:2*head+2]
    b.cx(a,words[target]);cx(b,z,words[target],c);b.cx(z,words[target+1]);b.cx(z,words[target+2])

def extract(b,words,jet_forward,jet_reverse,reverse=False,transcript=None,
            inverse_forward=inverse24_forward190,inverse_reverse=inverse24_reverse209):
    """Return/consume (kind,token) history; output head pairs alias input bank.

    jet_forward(b,u24,d24,K) -> token
    jet_reverse(b,u24,d24,token,K) restores both data and closes all phases.
    """
    h=len(words)//2;history=[]if not reverse else list(transcript)
    for op in (reversed(plan(h))if reverse else plan(h)):
        if op['kind']=='coordinates':cx(b,words[2*op['head']+1],words[2*op['head']],op['c']);continue
        if op['kind']=='local':
            i=2*op['head'];j=i+2;x=combine(words[j:j+2])
            if not reverse:
                b.cx(words[i+1],words[j]);history.append(('local',inverse_forward(b,x,K)))
                cx(b,words[j+1],words[j],op['c'])
            else:
                cx(b,words[j+1],words[j],op['c']);kind,t=history.pop();assert kind=='local'
                inverse_reverse(b,x,t,K);b.cx(words[i+1],words[j])
            continue
        j=op['target'];u=combine(words[j:j+2]);d=combine(words[j+2:j+4]);q=op['q']
        if not reverse:
            translate(b,words,op['head'],j,op['c']);history.append(('jet',jet_forward(b,u,d,K)))
            scalepair(b,d,q);b.cx(u,d);scalepair(b,u,q)
        else:
            from exp.decoder.common import inv
            scalepair(b,u,inv(q));b.cx(u,d);scalepair(b,d,inv(q))
            kind,t=history.pop();assert kind=='jet';jet_reverse(b,u,d,t,K)
            translate(b,words,op['head'],j,op['c'])
    if reverse:assert not history
    return history
