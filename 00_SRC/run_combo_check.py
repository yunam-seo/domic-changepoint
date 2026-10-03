#!/usr/bin/env python
"""Combined statistic T = max(studentized DOMI, studentized Spearman), calibrated by the SAME
pair permutation (P1 applies verbatim -> finite-sample valid). Power in both regimes (D1
monotone correlation, D2 nonlinear dependence) and the 2021-22 stock-bond window
(Section 4.5; Supplementary Section B.18). The window row here is a K = 49 pair-permutation preview;
the reported window results are those of run_combined_block_seeds.py."""
import os, sys, csv, json
import numpy as np
from multiprocessing import Pool
SRC=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(SRC); sys.path.insert(0,SRC)
from dots.synth import generate
from dots.domi import DOMIContext, domi_stats, DEP_BASELINES
from dots.perm import ge  # noqa: E402

K=49

def stats_all(X,Y,w=60):
    c=DOMIContext(X,Y,w,D=8,seed=2026,n_perm=0)
    return {"DOMI":domi_stats(c,"global")["DOMI-diff"],"SP":DEP_BASELINES["Spearman-diff"](c,"global")}

def test(X,Y,seedkey):
    obs=stats_all(X,Y); rng=np.random.default_rng(seedkey)
    R={m:[] for m in obs}
    for k in range(K):
        idx=rng.permutation(len(X)); c=stats_all(X[idx],Y[idx])
        for m in c: R[m].append(c[m])
    T={}; 
    for m in obs:
        A=np.vstack([obs[m][None,:]]+ [r[None,:] for r in R[m]])
        mu,sd=A.mean(0),A.std(0)+1e-12
        T[m]=[np.max((A[i]-mu)/sd) for i in range(K+1)]
    combo=[max(T["DOMI"][i],T["SP"][i]) for i in range(K+1)]
    out={}
    for name,Ts in [("DOMI",T["DOMI"]),("SP",T["SP"]),("COMBO",combo)]:
        out[name]=(1+sum(ge(t, Ts[0]) for t in Ts[1:]))/(K+1)
    return out

def job(a):
    scen,li,rep,null=a
    smp=generate(scen,{"D1":[0.2,0.35,0.5],"D2":[0.5,0.7,0.9]}[scen][li],rep,null=null,base_seed=20260830,level_idx=li)
    Z=smp["Z"]; return test(Z[:,[0]],Z[:,[1]],[20260830,rep,int(null),li])

if __name__=="__main__":
    rows=[]; rec=[]
    with Pool(24,maxtasksperchild=10) as pool:
        for scen,li in [("D1",1),("D2",1),("D2",2)]:
            nulls=pool.map(job,[(scen,li,r,True) for r in range(100)],chunksize=1)
            alts=pool.map(job,[(scen,li,r,False) for r in range(100)],chunksize=1)
            for tag,ds in (("null",nulls),("alt",alts)):
                rec+=[dict(scen=scen,li=li,side=tag,rep=i,**{m:d[m] for m in ("DOMI","SP","COMBO")})
                      for i,d in enumerate(ds)]
            for m in ("DOMI","SP","COMBO"):
                fa=np.mean([d[m]<=0.05 for d in nulls]); pw=np.mean([d[m]<=0.05 for d in alts])
                rows.append(dict(scen=scen,li=li,method=m,fpr=float(fa),power=float(pw)))
                print(f"{scen} li={li} {m}: FPR={fa:.2f} power={pw:.2f}",flush=True)
    # finance window
    from run_e6_finance import load, align_pair
    spx,tnx=load("yahoo_gspc.csv"),load("yahoo_tnx.csv")
    dates,x,y=align_pair(spx,tnx,("logret","diff"))
    sel=[i for i,d in enumerate(dates) if "20210101"<=d<="20221231"]
    r=test(x[sel][:,None],y[sel][:,None],[20260830,999])
    print("SPX-TNX 2021-22:",{m:round(v,3) for m,v in r.items()},flush=True)
    rows.append(dict(scen="SPX-TNX-2021-22",li=-1,method="pvals",fpr=-1,power=-1,extra=json.dumps(r)))
    with open(os.path.join(ROOT,"04_DAOU","EXPERIMENT","theory","records_combo_reps.csv"),"w",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=["scen","li","side","rep","DOMI","SP","COMBO"]); wr.writeheader(); wr.writerows(rec)
    with open(os.path.join(ROOT,"04_DAOU","EXPERIMENT","theory","combo_check.csv"),"w",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=["scen","li","method","fpr","power","extra"]); wr.writeheader()
        for rr in rows: wr.writerow({**dict(extra=""),**rr})
    print("combo check done")
