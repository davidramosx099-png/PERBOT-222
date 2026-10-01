import csv
import numpy as np
from pathlib import Path

def load(path):
    with Path(path).open(newline="", encoding="utf-8") as f: r=list(csv.DictReader(f))
    ts=np.array([int(x["timestamp"]) for x in r],dtype=np.int64)
    bid=np.array([float(x["bidPrice"]) for x in r]); ask=np.array([float(x["askPrice"]) for x in r])
    return ts,bid,ask

def rank_corr(x,y):
    a=np.argsort(np.argsort(x)); b=np.argsort(np.argsort(y))
    return np.corrcoef(a,b)[0,1]

def main():
    ts,bid,ask=load("data/xau_test/xauusd-tick-2026-09-29-2026-09-30.csv")
    mid=(bid+ask)/2; d=np.diff(mid); signed=np.sign(d)
    cut=int(len(ts)*.7)
    print("TRAIN 70%; feature/forward-return rank correlation; no trading rule")
    for lb in (50,100,250,500,1000,2000):
        feats={
            "pressure":np.convolve(signed,np.ones(lb)/lb,mode="valid"),
            "abs_move":np.convolve(np.abs(d),np.ones(lb)/lb,mode="valid"),
        }
        for h in (1000,3000,5000,10000,30000,60000):
            target=[]
            idx=[]
            for k in range(len(feats["pressure"])):
                i=k+lb
                if i>=cut: break
                j=np.searchsorted(ts,ts[i]+h)
                if j<len(ts):
                    target.append((mid[j]/mid[i]-1)*1e4); idx.append(k)
            y=np.asarray(target)
            if len(y)<1000: continue
            for name,xall in feats.items():
                x=xall[:len(y)]
                print(f"lb={lb} h={h//1000}s feature={name} rho={rank_corr(x,y):.5f}")
if __name__=="__main__": main()
