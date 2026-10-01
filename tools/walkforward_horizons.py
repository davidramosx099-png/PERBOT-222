import csv
import numpy as np
from pathlib import Path

LB=1000
HORIZONS=(10_000,30_000,60_000)
MIN_ROWS=100_000
COOLDOWN_MS=60_000

def load(p):
    with Path(p).open(newline="",encoding="utf-8") as f:
        rd=csv.DictReader(f); rows=[]
        for x in rd: rows.append((int(x["timestamp"]),float(x["bidPrice"]),float(x["askPrice"])))
    a=np.asarray(rows,dtype=float)
    return a[:,0].astype(np.int64),a[:,1],a[:,2]

def feat(bid,ask):
    d=np.diff((bid+ask)/2)
    k=np.ones(LB)/LB
    return np.convolve(np.sign(d),k,mode="valid"),np.convolve(np.abs(d),k,mode="valid")

def run(ts,bid,ask,p,a,pq,aq,h):
    vals=[]; last=None
    for k in range(len(p)):
        i=k+LB
        if a[k]<aq or p[k]>pq: continue
        cur=int(ts[i])
        if last is not None and cur-last<COOLDOWN_MS: continue
        j=np.searchsorted(ts,cur+h)
        if j>=len(ts): continue
        vals.append((bid[j]-ask[i])/ask[i]*1e4); last=cur
    return np.asarray(vals)

def main():
    fs=[]
    for f in sorted(Path("data/multi").glob("*.csv")):
        ts,bid,ask=load(f)
        if len(ts)>=MIN_ROWS: fs.append((f,ts,bid,ask))
    print("VALID_DAYS",len(fs))
    allres={h:[] for h in HORIZONS}
    for cut in range(3,len(fs)):
        train=fs[:cut]; test=fs[cut]
        aa=[]; pp=[]
        for _,ts,bid,ask in train:
            p,a=feat(bid,ask); pp.append(p); aa.append(a)
        aq=float(np.quantile(np.concatenate(aa),.80))
        pq=float(np.quantile(np.concatenate(pp),.20))
        ts,bid,ask=test[1:]
        p,a=feat(bid,ask)
        for h in HORIZONS:
            v=run(ts,bid,ask,p,a,pq,aq,h); allres[h].append(v)
            print(test[0].name,"H",h//1000,"n",len(v),
                  "mean",round(float(np.mean(v)),4) if len(v) else None,
                  "win",round(float(np.mean(v>0)),4) if len(v) else None)
    print("AGGREGATE_OOS")
    for h,arr in allres.items():
        x=np.concatenate([v for v in arr if len(v)])
        print("H",h//1000,"N",len(x),"mean",round(float(np.mean(x)),4),
              "median",round(float(np.median(x)),4),"win",round(float(np.mean(x>0)),4))
if __name__=="__main__": main()
