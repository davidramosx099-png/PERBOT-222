import csv
import numpy as np
from pathlib import Path

LB=1000
H=60_000
MIN_ROWS=100_000
COOLDOWN_MS=60_000

def load(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    ts=np.array([int(r["timestamp"]) for r in rows],dtype=np.int64)
    bid=np.array([float(r["bidPrice"]) for r in rows])
    ask=np.array([float(r["askPrice"]) for r in rows])
    return ts,bid,ask

def features(ts,bid,ask):
    mid=(bid+ask)/2.0
    d=np.diff(mid)
    s=np.sign(d)
    kernel=np.ones(LB)/LB
    pressure=np.convolve(s,kernel,mode="valid")
    activity=np.convolve(np.abs(d),kernel,mode="valid")
    return mid,np.convolve(s,kernel,mode="valid"),activity

def evaluate(ts,bid,ask,pressure,activity,p_q,a_q):
    vals=[]
    last_entry=-10**30
    for k in range(len(pressure)):
        i=k+LB
        if activity[k] < a_q or pressure[k] > p_q:
            continue
        if ts[i]-last_entry < COOLDOWN_MS:
            continue
        j=np.searchsorted(ts,ts[i]+H)
        if j>=len(ts):
            continue
        vals.append((bid[j]-ask[i])/ask[i]*1e4)
        last_entry=ts[i]
    return np.array(vals,dtype=float)

def main():
    files=sorted(Path("data/multi").glob("*.csv"))
    valid=[]
    for f in files:
        ts,bid,ask=load(f)
        if len(ts)>=MIN_ROWS:
            valid.append((f,ts,bid,ask))
        else:
            print("SKIP",f.name,"rows",len(ts))
    if len(valid)<4:
        raise SystemExit("need at least 4 full active days")
    print("VALID_DAYS",len(valid))
    for cut in range(3,len(valid)):
        train=valid[:cut]
        test=valid[cut]
        train_a=[]; train_p=[]
        for _,ts,bid,ask in train:
            _,p,a=features(ts,bid,ask)
            train_a.append(a); train_p.append(p)
        aq=np.quantile(np.concatenate(train_a),.80)
        pq=np.quantile(np.concatenate(train_p),.20)
        ts,bid,ask=test[1],test[2],test[3]
        _,p,a=features(ts,bid,ask)
        vals=evaluate(ts,bid,ask,p,a,pq,aq)
        spread=np.median((ask-bid)/((ask+bid)/2)*1e4)
        print("TRAIN_DAYS",cut,"TEST",test[0].name,
              "activity_q80",round(aq,6),"pressure_q20",round(pq,6),
              "spread_med_bps",round(float(spread),3),
              "signals",len(vals),
              "mean_net_bps",round(float(np.mean(vals)),4) if len(vals) else None,
              "median_net_bps",round(float(np.median(vals)),4) if len(vals) else None,
              "win",round(float(np.mean(vals>0)),4) if len(vals) else None)

if __name__=="__main__":
    main()
