import csv
import numpy as np
from pathlib import Path

def load(path):
    with Path(path).open(newline="", encoding="utf-8") as f: r=list(csv.DictReader(f))
    ts=np.array([int(x["timestamp"]) for x in r],dtype=np.int64)
    bid=np.array([float(x["bidPrice"]) for x in r]); ask=np.array([float(x["askPrice"]) for x in r])
    return ts,bid,ask

def main():
    ts,bid,ask=load("data/xau_test/xauusd-tick-2026-09-29-2026-09-30.csv")
    mid=(bid+ask)/2; d=np.diff(mid); signed=np.sign(d)
    cut=int(len(ts)*.7)
    print("70/30 chronological quantile test; train bins fixed, TEST untouched")
    for lb in (250,500,1000,2000):
        pressure=np.convolve(signed,np.ones(lb)/lb,mode="valid")
        activity=np.convolve(np.abs(d),np.ones(lb)/lb,mode="valid")
        for h in (10000,30000,60000):
            X=[]; Y=[]
            for k in range(len(pressure)):
                i=k+lb
                if i>=len(ts): break
                j=np.searchsorted(ts,ts[i]+h)
                if j>=len(ts): break
                X.append((pressure[k],activity[k],i))
                Y.append((mid[j]/mid[i]-1)*1e4)
            X=np.asarray(X); Y=np.asarray(Y)
            train=X[:,2]<cut
            pa=np.quantile(X[train,0],[.2,.4,.6,.8])
            aa=np.quantile(X[train,1],[.2,.4,.6,.8])
            print(f"lb={lb} h={h//1000}s")
            for q in range(5):
                lo=-np.inf if q==0 else pa[q-1]; hi=np.inf if q==4 else pa[q]
                m=(X[:,0]>=lo)&(X[:,0]<hi)&(~train)
                if m.sum(): print(f"  TEST pressure_q{q+1}: n={m.sum()} mean_bps={Y[m].mean():.3f}")
            for q in range(5):
                lo=-np.inf if q==0 else aa[q-1]; hi=np.inf if q==4 else aa[q]
                m=(X[:,1]>=lo)&(X[:,1]<hi)&(~train)
                if m.sum(): print(f"  TEST activity_q{q+1}: n={m.sum()} mean_bps={Y[m].mean():.3f}")

if __name__=="__main__": main()
