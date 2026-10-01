import csv
import numpy as np
from pathlib import Path

def load(path):
    with Path(path).open(newline="",encoding="utf-8") as f:r=list(csv.DictReader(f))
    ts=np.array([int(x["timestamp"]) for x in r],dtype=np.int64)
    bid=np.array([float(x["bidPrice"]) for x in r]); ask=np.array([float(x["askPrice"]) for x in r])
    return ts,bid,ask

def main():
    files=sorted(Path("data/xau_test").glob("*.csv"))
    if not files: raise SystemExit("no CSV files")
    for path in files:
        ts,bid,ask=load(path); mid=(bid+ask)/2; d=np.diff(mid); signed=np.sign(d); cut=int(len(ts)*.7)
        spread_med=np.median((ask-bid)[:cut])
        lb=1000; h=60000
        p=np.convolve(signed,np.ones(lb)/lb,mode="valid")
        a=np.convolve(np.abs(d),np.ones(lb)/lb,mode="valid")
        vals=[]
        for k in range(len(p)):
            i=k+lb
            if i>=cut or a[k] <= np.quantile(a[:max(1,cut-lb)],.8): continue
            j=np.searchsorted(ts,ts[i]+h)
            if j<len(ts): vals.append((bid[j]-ask[i])/ask[i]*1e4)
        print(path.name,"N",len(ts),"spread_med_bps",spread_med/mid[cut//2]*1e4,
              "high_activity_60s_quote_net_mean_bps",np.mean(vals) if vals else None,
              "n",len(vals))
if __name__=="__main__":main()
