#!/usr/bin/env python
"""Monthly aggregation of the finance pairs: exchangeability diagnostic and, where the
diagnostic no longer rejects, pair-permutation-calibrated Holevo-partitioning segmentation —
completing the resolution ladder (daily -> weekly -> monthly).
Writes 04_DAOU/EXPERIMENT/e6/results_monthly.json
"""
import json, os, sys
import numpy as np
SRC=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(SRC); sys.path.insert(0,SRC)
from run_e6_finance import load, align_pair
from dots.segtests import pelt_segment
from run_e6a_block import pelt_segment_block
from run_exch_diag import exch_diag
OUT=os.path.join(ROOT,"04_DAOU","EXPERIMENT","e6")

def msum(x,k=21):
    m=len(x)//k
    return x[:m*k].reshape(m,k).sum(1)

def main():
    spx,tnx=load("yahoo_gspc.csv"),load("yahoo_tnx.csv")
    ksp,fx=load("yahoo_kospi.csv"),load("yahoo_usdkrw.csv")
    pairs={"SPX-TNX":align_pair(spx,tnx,("logret","diff")),
           "KOSPI-USDKRW":align_pair(ksp,fx,("logret","logret"))}
    res={}
    betas=list(np.linspace(0.5,12,24))
    for name,(dates,x,y) in pairs.items():
        xm,ym=msum(x),msum(y)
        mdates=dates[::21][:len(xm)]
        d=exch_diag(xm,ym,L=12,seed=4000+len(name))   # fixed per-pair seed
        entry=dict(n_months=len(xm),diag_decision=d["decision"],diag_pvalues=d["pvalues"],b_hat=d["b_hat"])
        import dots.segtests as _SEG
        _SEG.PERM_TAG=f"e6c_{name}"
        if d["decision"]=="pair":
            obs,fa=pelt_segment(xm,ym,step=2,betas=betas)
            beta=next((b for b in betas if fa[b]<=0.05),betas[-1])
            entry.update(calibration="pair",beta=beta,fa=fa[beta],
                         cps_dates=[mdates[min(c,len(mdates)-1)] for c in obs[beta]])
        else:
            obs,fa=pelt_segment_block(xm,ym,step=2,betas=betas,block=d["b_hat"])
            beta=next((b for b in betas if fa[b]<=0.05),betas[-1])
            entry.update(calibration=f"block{d['b_hat']}",beta=beta,fa=fa[beta],
                         cps_dates=[mdates[min(c,len(mdates)-1)] for c in obs[beta]])
        res[name]=entry
        print(f"{name}: n={len(xm)} diag={d['decision']} pvals={d['pvalues']} b_hat={d['b_hat']} "
              f"-> {entry.get('calibration')} beta={entry.get('beta'):.1f} fa={entry.get('fa'):.2f} CPs {entry.get('cps_dates')}",flush=True)
    import dots.segtests as SEG
    SEG.flush_perm_log(os.path.join(OUT,"records_perm_monthly.csv"))
    json.dump(res,open(os.path.join(OUT,"results_monthly.json"),"w"),indent=1,default=str)
    print("saved results_monthly.json")

if __name__=="__main__": main()
