import csv
import numpy as np
from pathlib import Path

def load(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        r=list(csv.DictReader(f))
    ts=np.array([int(x["timestamp"]) for x in r],dtype=np.int64)
    bid=np.array([float(x["bidPrice"]) for x in r])
    ask=np.array([float(x["askPrice"]) for x in r])
    return ts,bid,ask

def main():
    ts,bid,ask=load("data/xau_test/xauusd-tick-2026-09-29-2026-09-30.csv")
    cut=int(len(ts)*.7)
    spread_med=np.median(ask[:cut]-bid[:cut])
    print("TRAIN 70%; quote-aware BUY return; signed pressure + acceleration")
    for lb in (50,100,250,500,1000,2000):
        d=np.diff((bid+ask)/2)
        s=np.sign(d)
        p=np.convolve(s,np.ones(lb)/lb,mode="valid")
        half=max(10,lb//4)
        fast=np.convolve(s,np.ones(half)/half,mode="valid")
        for h in (1000,3000,5000,10000):
            vals=[]
            for k,x in enumerate(p):
                i=k+lb
                if i>=cut or i<half: continue
                f=fast[i-half]
                if x<=0.10 or f<=x or (ask[i]-bid[i])>2*spread_med:
                    continue
                j=np.searchsorted(ts,ts[i]+h)
                if j<len(ts):
                    vals.append((bid[j]-ask[i])/ask[i])
            if len(vals)>=100:
                a=np.asarray(vals)
                print(f"lb={lb} h={h//1000}s n={len(a)} mean_net_bps={a.mean()*1e4:.3f} median={np.median(a)*1e4:.3f} win={np.mean(a>0):.3f}")

if __name__=="__main__":
    main()
