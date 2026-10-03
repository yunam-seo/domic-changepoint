"""Section 6.2: Holevo partitioning on a marginal-only scale change.

X and Y independent throughout; the scale of X is multiplied by s in {1, 1.3, 1.6, 2} at t = 300;
n = 600, 100 replicates per s; segment costs centered by 5 row-permuted copies. The penalty beta is
the smallest value on the grid 0.5..25 at which <= 5% of the no-change (s = 1) replicates yield
any break.

Run: python 00_SRC/run_pelt_specificity.py
Writes 04_DAOU/EXPERIMENT/pelt_specificity/{results.json, records_reps.csv}.
"""
import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import numpy as np, json, os
OUT=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'04_DAOU','EXPERIMENT','pelt_specificity')
os.makedirs(OUT,exist_ok=True)
from multiprocessing import Pool
from dots.domi import ranks01, unit_rff
from dots.encode import MomentCache
from dots import pelt as P
N,D,TAU=600,8,300
GRID=np.arange(0,N+1,10)
def cost(x,y,rep):
    FX=unit_rff(ranks01(x[:,None]),D,2026); FY=unit_rff(ranks01(y[:,None]),D,2027)
    J=np.einsum("ti,tj->tij",FX,FY).reshape(N,D*D)
    C=P.cost_matrix_vn(MomentCache(J),GRID); Cp=np.zeros_like(C)
    rng=np.random.default_rng(7000+rep)
    for _ in range(5): Cp+=P.cost_matrix_vn(MomentCache(J[rng.permutation(N)]),GRID)
    return C-Cp/5
def gen(rep,s):
    rng=np.random.default_rng(30000+rep)
    x=rng.standard_normal(N); y=rng.standard_normal(N); x[TAU:]*=s          # X,Y independent always
    return x,y
def _job(a):
    rep,s=a; x,y=gen(rep,s); return cost(x,y,rep)
if __name__=="__main__":
    reps=100
    with Pool(14,maxtasksperchild=10) as pool:
        Cs={s:pool.map(_job,[(i,s) for i in range(reps)],chunksize=1) for s in (1.0,1.3,1.6,2.0)}
    betas=list(np.linspace(0.5,25,50))
    beta=next((b for b in betas if np.mean([len(P.pelt_from_costs(C,b))>0 for C in Cs[1.0]])<=0.05),betas[-1])
    print(f"beta calibrated on the no-change null: {beta:.2f} (null any-break rate "
          f"{np.mean([len(P.pelt_from_costs(C,beta))>0 for C in Cs[1.0]]):.3f})")
    print(f"{'s':>5} {'P(any break)':>13} {'P(within 30 of tau)':>21} {'mean k':>7}")
    out={}; rec=[]
    for s in (1.0,1.3,1.6,2.0):
        anyb=[];near=[];ks=[]
        for ri,C in enumerate(Cs[s]):
            loc=[int(GRID[c]) for c in P.pelt_from_costs(C,beta)]
            anyb.append(len(loc)>0); near.append(any(abs(l-TAU)<=30 for l in loc)); ks.append(len(loc))
            rec.append(dict(level=s,rep=ri,any=int(anyb[-1]),near=int(near[-1]),k=ks[-1]))
        out[s]=dict(any=float(np.mean(anyb)),near=float(np.mean(near)),mean_k=float(np.mean(ks)))
        print(f"{s:>5} {np.mean(anyb):>13.3f} {np.mean(near):>21.3f} {np.mean(ks):>7.2f}")
    import csv as _csv
    with open(os.path.join(OUT,'records_reps.csv'),'w',newline='') as f:
        w=_csv.DictWriter(f,fieldnames=['level','rep','any','near','k']); w.writeheader(); w.writerows(rec)
    json.dump(dict(beta=float(beta),out={str(k):v for k,v in out.items()}),
              open(os.path.join(OUT,'results.json'),'w'),indent=1)
