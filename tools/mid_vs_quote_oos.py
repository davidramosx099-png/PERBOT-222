import csv
import numpy as np
from pathlib import Path

LB=(20,50,100,250,500,1000,2000)
H=(10_000,30_000,60_000)
MIN=100_000
COOLDOWN=60_000

def load(p):
    with Path(p).open(newline="",encoding="utf-8") as f:
        r=csv.DictReader(f)
        rows=[(int(x["timestamp"]),float(x["bidPrice"]),float(x["askPrice"])) for x in r]
    if not rows:return None
    a=np.asarray(rows,float)
    return a[:,0].astype(np.int64),a[:,1],a[:,2]

def feat(ts,bid,ask,lb):
    mid=(bid+ask)/2
    d=np.diff(mid)
    pressure=np.convolve(np.sign(d),np.ones(lb)/lb,mode="valid")
    activity=np.convolve(np.abs(d),np.ones(lb)/lb,mode="valid")
    return mid,pressure,activity

def run(ts,bid,ask,mid,pressure,activity,h,cut,pq,aq):
    gross=[];quote=[];last=None
    for k in range(len(pressure)):
        if pressure[k] > pq or activity[k] < aq:
            continue
        i=k+cut
        cur=int(ts[i])
        if last is not None and cur-last<COOLDOWN:
            continue
        j=np.searchsorted(ts,cur+h)
        if j>=len(ts):continue
        gross.append((mid[j]/mid[i]-1)*1e4)
        quote.append((bid[j]-ask[i])/ask[i]*1e4)
        last=cur
    return np.asarray(gross),np.asarray(quote)

def main():
    fs=[]
    for f in sorted(Path("data/multi").glob("*.csv")):
        x=load(f)
        if x is not None and len(x[0])>=MIN:fs.append((f,*x))
    print("VALID_DAYS",len(fs))
    for lb in LB:
        for h in H:
            gross=[];quote=[]
            for oi in range(3,len(fs)):
                train_p=[];train_a=[]
                for item in fs[max(0,oi-3):oi]:
                    ts,bid,ask=item[1:]
                    _,p,a=feat(ts,bid,ask,lb)
                    train_p.append(p);train_a.append(a)
                pq=float(np.quantile(np.concatenate(train_p),.20))
                aq=float(np.quantile(np.concatenate(train_a),.80))
                ts,bid,ask=fs[oi][1:]
                mid,p,a=feat(ts,bid,ask,lb)
                g,q=run(ts,bid,ask,mid,p,a,h,lb,pq,aq)
                if len(g):
                    gross.extend(g);quote.extend(q)
            if gross:
                g=np.asarray(gross);q=np.asarray(quote)
                print("DECOMP","lb",lb,"h",h//1000,"N",len(g),
                      "gross_mean",round(float(g.mean()),4),
                      "quote_mean",round(float(q.mean()),4),
                      "cost_gap",round(float((g-q).mean()),4),
                      "gross_win",round(float(np.mean(g>0)),4),
                      "quote_win",round(float(np.mean(q>0)),4))
if __name__=="__main__":
    main()
