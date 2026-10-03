"""Is the marginal-scale diagnostic of run_e8_marginal.py valid under serial dependence? (Section 4.7)
It uses block permutation, so it should hold its level; this script measures it.
Null: AR(1) margins (phi=0.6), NO scale change anywhere. Alternative: scale x8 at the midpoint.
200 replicates per design (data seeds 50000 + rep, permutation seeds 60000 + rep), K = 99.

Run:    python 00_SRC/run_marginal_validity.py | tee 04_DAOU/EXPERIMENT/marginal_validity/output.txt
Prints the two rejection rates (the stored output.txt is this printed output) and writes the
per-replicate p-values to 04_DAOU/EXPERIMENT/marginal_validity/records.csv."""
import os
import sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool
from dots.hourly import block_perm_order
from dots.perm import ge  # noqa: E402
UNIT=336; N=UNIT*52; K=99; SUPER=3   # 52 two-week intervals: the +/- 1 year stage-two window
def curve(v):
    e=np.arange(0,len(v)+1,UNIT)
    if e[-1]!=len(v): e=np.append(e,len(v))
    B=len(e)-1
    s1=np.array([np.nansum(v[e[k]:e[k+1]]) for k in range(B)])
    s2=np.array([np.nansum(v[e[k]:e[k+1]]**2) for k in range(B)])
    cnt=np.array([e[k+1]-e[k] for k in range(B)])
    return s1,s2,cnt
def stat(s1,s2,cnt,lo=3):
    B=len(cnt); c1=np.concatenate([[0],np.cumsum(s1)]); c2=np.concatenate([[0],np.cumsum(s2)]); cn=np.concatenate([[0],np.cumsum(cnt)])
    n=cn[B]; ts=np.arange(lo,B-lo+1); out=np.empty(len(ts))
    for i,t in enumerate(ts):
        nl,nr=cn[t],n-cn[t]
        vl=c2[t]/nl-(c1[t]/nl)**2; vr=(c2[B]-c2[t])/nr-((c1[B]-c1[t])/nr)**2
        out[i]=np.sqrt(nl*nr/n)*abs(np.log(max(vl,1e-12))-np.log(max(vr,1e-12)))/2
    return ts,out
def gen(rep,scale):
    rng=np.random.default_rng(50000+rep); phi=0.6
    e=rng.standard_normal(N); x=np.empty(N); x[0]=e[0]
    for t in range(1,N): x[t]=phi*x[t-1]+e[t]
    if scale!=1: x=x.copy(); x[N//2:]*=scale
    return x
def job(a):
    rep,scale=a; x=gen(rep,scale)
    s1,s2,cnt=curve(x); ts,obs=stat(s1,s2,cnt)
    rng=np.random.default_rng(60000+rep); R=np.empty((K,len(ts)))
    for k in range(K):
        o=block_perm_order(len(cnt),SUPER,rng); R[k]=stat(s1[o],s2[o],cnt[o])[1]
    A=np.vstack([obs[None,:],R]); mu,sd=A.mean(0),A.std(0)+1e-12
    T=[float(np.nanmax((A[i]-mu)/sd)) for i in range(K+1)]
    return (1+sum(ge(t, T[0]) for t in T[1:]))/(K+1)
if __name__=="__main__":
    out=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"04_DAOU","EXPERIMENT","marginal_validity")
    os.makedirs(out,exist_ok=True)
    rows=[]
    with Pool(12) as p:
        for scale,lab in ((1,"H0 (no scale change)"),(8,"H1 (x8 at midpoint)")):
            ps=p.map(job,[(r,scale) for r in range(200)],chunksize=2)
            rows+=[(scale,r,x) for r,x in enumerate(ps)]
            print(f"  {lab:24s}: reject@0.05 = {np.mean([x<=0.05 for x in ps]):.3f}")
    with open(os.path.join(out,"records.csv"),"w") as f:
        f.write("scale,rep,p\n"+"".join(f"{a},{r},{x!r}\n" for a,r,x in rows))
