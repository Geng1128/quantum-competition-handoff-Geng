"""Zero-safe division by temporarily replacing a zero factor by one."""
from common import *
import division as dv
from exp.mul99.schedules import divide_zero_bridge414

def divide_zero_bridge(b,x,y,extra_zero_mask=None):
    return divide_zero_bridge414(b,x,y,K=K,extra_zero_mask=extra_zero_mask)

if __name__=='__main__':
    import numpy as np,json
    rng=np.random.default_rng(18171026);n=16384
    xv=rng.integers(0,1<<24,n,dtype=np.uint64);yv=rng.integers(0,1<<24,n,dtype=np.uint64);yv[:4096]=0
    b=km.CircuitBuilder();x=b.alloc_qubits(24);y=b.alloc_qubits(24)
    old,fm=dv.multiply_zero_forward(b,x,y,K);mm=b.alloc_bits(24);b.cz(x,mm);b.cz(x[2],x[13])
    divide_zero_bridge(b,x,y,fm);b.free(fm);b.cz(x,old);b.free(old)
    c=b.finish_circuit();s=km.Simulator(batch_size=n,seed=188182);ms=rng.integers(0,1<<24,n,dtype=np.uint64)
    write(s,x,xv);write(s,y,yv);write(s,mm,ms);s.do(c)
    assert np.array_equal(read(s,x),xv) and np.array_equal(read(s,y),yv)
    z=[emul(int(a),int(bb)) if bb else int(a) for a,bb in zip(xv,yv)]
    want=np.array([bool((a&int(m)).bit_count()%2)^bool(((a>>2)&1)&((a>>13)&1)) for a,m in zip(z,ms)])
    assert np.array_equal(want,s.read_phase_flipped_across_shots())
    for q in range(48,c.num_qubits):assert not s.read_across_shots(km.q(q)).any(),q
    r=dict(shots=n,pair_T=c.operation_statistics().num_toffoli_like,division_T=c.operation_statistics().num_toffoli_like-152,scratch=c.num_qubits-48,value_errors=0,phase_errors=0)
    print(json.dumps(r,indent=2));(ROOT/'exp/decoder/safe_bridge.json').write_text(json.dumps(r,indent=2)+'\n')
