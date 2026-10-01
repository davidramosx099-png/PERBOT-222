import csv, math
import numpy as np
from pathlib import Path

def load_csv(path):
    ts=[]; bid=[]; ask=[]
    with Path(path).open(newline="",encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ts.append(int(r["timestamp"])); bid.append(float(r["bidPrice"])); ask.append(float(r["askPrice"]))
    return np.asarray(ts,dtype=np.int64),np.asarray(bid),np.asarray(ask)

def forward_return(mid,horizon_ms):
    n=len(mid); out=np.full(n,np.nan); j=0
    for i in range(n):
        target=ts_global[i]+horizon_ms
        if j<i: j=i
        while j<n and ts_global[j]<target: j+=1
        if j<n: out[i]=mid[j]/mid[i]-1.0
    return out

def audit(path):
    global ts_global
    ts,bid,ask=load_csv(path); ts_global=ts; mid=(bid+ask)/2
    print("N",len(ts),"duration_s",(ts[-1]-ts[0])/1000)
    for sec in (1,3,5,10,30,60):
        r=forward_return(mid,sec*1000)
        valid=r[~np.isnan(r)]
        print(f"H={sec:>2}s valid={len(valid)} mean_bps={np.mean(valid)*1e4:.3f} median_bps={np.median(valid)*1e4:.3f} up={np.mean(valid>0):.4f}")
    sp=ask-bid
    print("spread_bps median/max",np.median(sp/mid)*1e4,np.max(sp/mid)*1e4)

if __name__=="__main__": audit("data/xau_test/xauusd-tick-2026-09-29-2026-09-30.csv")