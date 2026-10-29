"""Exact all-address, arbitrary-target, all-HMR-outcome symbolic audit."""
import numpy as np

def packed(bits):
    return int.from_bytes(np.packbits(np.asarray(bits,np.uint8),bitorder='little').tobytes(),'little')

def audit(text,delta,nq):
    D,J=delta.shape;n=(D-1).bit_length();m=8*J
    full=(1<<D)-1;values=[0]*nq;symbols=[0]*nq
    for i in range(n):values[i]=packed((np.arange(D)>>i)&1)
    for i in range(m):symbols[n+i]=1<<i
    addresses=values[:n].copy();phase=[];classical={};ccx=0
    for line in text.splitlines():
        fields=line.split()
        if not fields:continue
        name=fields[0]
        if name in ['REGISTER','APPEND_TO_REGISTER']:continue
        condition=None
        if 'if' in fields:
            k=fields.index('if');condition=fields[k+1];fields=fields[:k]
        ids=[int(q[1:]) for q in fields[1:] if q.startswith('q')]
        if name not in ['CZ','Z','NEG']:assert condition is None,('conditioned nonphase gate',line)
        if name=='R':
            q,=ids;assert values[q]==symbols[q]==0,('discarding information',line)
        elif name=='X':
            q,=ids;values[q]^=full
        elif name=='CX':
            a,b=ids;values[b]^=values[a];symbols[b]^=symbols[a]
        elif name=='CCX':
            a,b,c=ids;assert symbols[a]==symbols[b]==0,('target-dependent nonlinear control',line)
            values[c]^=values[a]&values[b];ccx+=1
        elif name=='HMR':
            q,=ids;assert symbols[q]==0,('measuring unknown target',line)
            cb=next(b for b in fields[1:] if b.startswith('b'))
            classical[cb]=len(phase);phase.append(values[q]);values[q]=0
        elif name in ['CZ','Z','NEG']:
            assert condition in classical,('unexpected unconditioned phase',line)
            assert not any(symbols[q] for q in ids)
            correction=full
            for q in ids:correction&=values[q]
            phase[classical[condition]]^=correction
        else:raise AssertionError(('unhandled instruction',line))
    assert values[:n]==addresses and not any(symbols[:n]),'address changed'
    for j in range(J):
        for b in range(8):
            q=n+8*j+b
            assert values[q]==packed((delta[:,j]>>b)&1),('wrong delta',j,b)
            assert symbols[q]==1<<(8*j+b),('wrong target linear map',j,b)
    assert not any(values[n+m:]) and not any(symbols[n+m:]),'dirty scratch'
    assert not any(phase),'measurement-dependent phase'
    return {'status':'PASS','all_addresses':D,'arbitrary_target_bits':m,
            'all_target_assignments':f'2^{m}','independent_HMR_outcomes':len(phase),
            'all_HMR_assignments':f'2^{len(phase)}','zero_phase_for_every_outcome':True,
            'all_scratch_zero':True,'CCX_unconditional':True,'CCX':ccx,'Q':nq}
