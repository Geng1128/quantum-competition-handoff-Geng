"""Bro8/bro10 exact tensor compiler; no chemistry or approximate word changes."""
from __future__ import annotations
from pathlib import Path
import argparse,hashlib,json,shutil,sys,time
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'push_synthesis'),str(ROOT/'new_lookup/scripts')]
from joint_partition import feature_basis,mobius,batch_deltas,km,xor_affine_product
from joint_low_rank import packed_rows,rowbasis
from compile_four_address import signed_gauge,decode_tree
from low_three_lookup import check


def prepare(delta,brot,lowbits):
    count,J=delta.shape;A=1<<lowbits;B=(count+A-1)//A
    pad=np.zeros((A*B,J),np.int64);pad[:count]=delta
    bits=np.stack([(pad[:,j]>>b)&1 for j in range(J) for b in range(brot-1)],axis=1).astype(np.uint8)
    truth=bits.reshape(B,A,-1).transpose(1,0,2).copy()
    bc=mobius(truth.transpose(1,0,2)).transpose(1,0,2).copy()
    return truth,bc,mobius(bc)


def decompose(delta,brot=8,lowbits=6,group_size=5,cap=17,trials=32):
    truth,bc,ac=prepare(delta,brot,lowbits);A,B,O=truth.shape;terms=[];stats=[]
    for start in range(0,O,group_size):
        end=min(start+group_size,O);m=end-start;target=ac[1:,1:,start:end].reshape(A-1,-1);best=None
        for cd in range(cap,-1,-1):
            for seed in [None]+list(range(trials)):
                try:result=feature_basis(target,m,cd,seed,nlow=B-1)
                except ValueError:continue
                if best is None or result[2]['products']<best[2]['products']:best=result
                if result[2]['products']==(result[2]['span_rank']+1)//2:break
            if best is not None:break
        if best is None:raise AssertionError('No exact feature basis')
        fs,hc,st=best;stats.append({'start':start,'end':end,**st})
        for j,f in enumerate(fs):
            a=sum(((v>>j)&1)<<h for h,v in enumerate(hc));terms.append((a,f.low,f.out<<start))
    return terms,stats


def build_circuit(delta,terms,brot=8,lowbits=6):
    truth,bc,ac=prepare(delta,brot,lowbits);A,B,O=truth.shape;count,J=delta.shape
    targetrows=packed_rows(ac[1:,1:,:].reshape(A-1,-1));actual=[0]*(A-1)
    for a,b,c in terms:
        lo=sum(c<<(l*O) for l in range(B-1) if(b>>l)&1)
        for h in range(A-1):
            if(a>>h)&1:actual[h]^=lo
    assert actual==targetrows
    cachecost=sum(k.bit_count()>1 for k in range(1,A))+sum(k.bit_count()>1 for k in range(1,B))
    n=(count-1).bit_length();assert n+J*brot+cachecost<=128,('cap',n+J*brot+cachecost)
    cb=km.CircuitBuilder();sel=cb.create_quantum_register(n,'selection');regs=[cb.create_quantum_register(brot,f'word_{j}') for j in range(J)]
    outs=[[regs[j][bit]]+([regs[j][brot-1]] if bit==brot-2 else []) for j in range(J) for bit in range(brot-1)]
    X={0:True};Y={0:True};cache=[]
    for bases,size,offset in [(X,A,0),(Y,B,lowbits)]:
        for mask in range(1,size):
            bit=mask.bit_length()-1
            if mask.bit_count()==1:bases[mask]=sel[offset+bit]
            else:
                par=mask^(1<<bit);q=cb.alloc_qubits(1)[0];cb.init_and(bases[par],sel[offset+bit],q);bases[mask]=q;cache.append((bases[par],sel[offset+bit],q))
    basis=[X[k] for k in range(1,A)]+[Y[k] for k in range(1,B)]
    xonly=bc[:,0].copy();yonly=ac[0].copy();yonly[0]=0;ytruth=mobius(yonly);groups=defaultdict(list)
    for a,b,c in terms:groups[c].append((a,b))
    predicted=cachecost
    for c,pairs in sorted(groups.items()):
        targets=[q for j in range(O) if(c>>j)&1 for q in outs[j]]
        for off in range(0,len(pairs),2):
            a,b=pairs[off]
            if off+1==len(pairs):predicted+=xor_affine_product(cb,basis,a,b<<(A-1),targets)
            else:
                aa,bb=pairs[off+1];predicted+=xor_affine_product(cb,basis,a^(bb<<(A-1)),aa^(b<<(A-1)),targets)
                av=mobius(np.array([0]+[(a>>h)&1 for h in range(A-1)],np.uint8));aav=mobius(np.array([0]+[(aa>>h)&1 for h in range(A-1)],np.uint8))
                bv=mobius(np.array([0]+[(b>>l)&1 for l in range(B-1)],np.uint8));bbv=mobius(np.array([0]+[(bb>>l)&1 for l in range(B-1)],np.uint8))
                for j in range(O):
                    if(c>>j)&1:xonly[:,j]^=av&aav;ytruth[:,j]^=bv&bbv
    for table,bases in [(mobius(xonly),X),(mobius(ytruth),Y)]:
        for mask,row in enumerate(table):
            targets=[q for j in np.flatnonzero(row) for q in outs[int(j)]]
            if targets:cb.cx(bases[mask],km.array(targets))
    for a,b,q in reversed(cache):cb.del_and(a,b,q);cb.free(q)
    return cb.finish_circuit(),predicted


def main():
    p=argparse.ArgumentParser();p.add_argument('--submission',type=Path,required=True);p.add_argument('--words',type=Path);p.add_argument('--brot',type=int,default=8);p.add_argument('--batch-size',type=int,default=5);p.add_argument('--lowbits',type=int,default=6);p.add_argument('--group-size',type=int,default=5);p.add_argument('--cap',type=int,default=17);p.add_argument('--trials',type=int,default=32);p.add_argument('--count',type=int,default=1);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    src=np.load(a.submission);original=np.load(a.words) if a.words else src['angle_words'];words,signs=signed_gauge(original,src['tree_edges'],a.brot)
    err=float(np.max(np.abs(decode_tree(words,src['tree_edges'],a.brot)-signs[:,None]*decode_tree(original,src['tree_edges'],a.brot))));assert err<1e-12
    a.out.mkdir(parents=True,exist_ok=True);reports=[];start=time.monotonic()
    for k,delta in enumerate(batch_deltas(words,a.batch_size)[:a.count]):
        terms,stats=decompose(delta,a.brot,a.lowbits,a.group_size,a.cap,a.trials);c,predicted=build_circuit(delta,terms,a.brot,a.lowbits)
        native=check(c,delta,brot=a.brot,repeats=4,seed=54124+k);assert native==predicted
        path=a.out/f'angle_k_{k}.kmb';c.to_file(path,format='kmb');rec={'circuit':k,'T':native,'Q':int(c.num_qubits),'stats':stats,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()};reports.append(rec);print(json.dumps(rec),flush=True)
    shutil.copy2(a.submission,a.out/'submission.npz');np.save(a.out/'angle_words.npy',words);np.save(a.out/'source_angle_words.npy',original);np.save(a.out/'gauge_root_signs.npy',signs)
    report={'schema':'generic_brot_tensor_compiler_v1','source':str(a.submission),'brot':a.brot,'batch_size':a.batch_size,'lowbits':a.lowbits,'source_gauge_max_residual':err,'circuits':reports,'complete_family':len(reports)==len(batch_deltas(words,a.batch_size)),'chemistry_verified':False,'seconds':time.monotonic()-start}
    (a.out/'compile_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
