import csv
import numpy as np
from pathlib import Path

LB=(100,250,500,1000)
H=(30_000,60_000)
SPREAD_Q=(.20,.40,.60)
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
    p=np.convolve(np.sign(d),np.ones(lb)/lb,mode="valid")
    a=np.convolve(np.abs(d),np.ones(lb),mode="valid")
    return mid,p,a

def run(ts,bid,ask,mid,p,a,h,cut,pq,aq,sq):
    g=[];q=[];last=None
    for k in range(len(p)):
        if p[k]>=pq or a[k]<aq: continue
        i=k+cut
        if (ask[i]-bid[i])/mid[i] > sq: continue
        cur=int(ts[i])
        if last is not None and cur-last<COOLDOWN: continue
        j=np.searchsorted(ts,cur+h)
        if j>=len(ts): continue
        g.append((mid[j]/mid[i]-1)*1e4)
        q.append((bid[j]-ask[i])/ask[i]*1e4)
        last=cur
    return np.asarray(g),np.asarray(q)

def main():
    fs=[]
    for f in sorted(Path("data/multi").glob("*.csv")):
        x=load(f)
        if x is not None and len(x[0])>=MIN: fs.append((f,*x))
    print("VALID_DAYS",len(fs))
    for lb in LB:
        for h in H:
            for sq in SPREAD_Q:
                g=[];q=[]
                for oi in range(3,len(fs)):
                    tp=[];ta=[];tspr=[]
                    for item in fs[oi-3:oi]:
                        ts,bid,ask=item[1:]
                        mid,p,a=feat(ts,bid,ask,lb)
                        tp.append(p);ta.append(a);tspr.append((ask[lb:]-bid[lb:])/mid[lb:]*1e4)
                    pq=float(np.quantile(np.concatenate(tp),.20))
                    aq=float(np.quantile(np.concatenate(ta),.80))
                    spread_q=float(np.quantile(np.concatenate(tspr),sq))
                    ts,bid,ask=fs[oi][1:]
                    mid,p,a=feat(ts,bid,ask,lb)
                    gg,qq=run(ts,bid,ask,mid,p,a,h,lb,pq,aq,spread_q)
                    if len(gg):g.extend(gg);q.extend(qq)
                if g:
                    x=np.asarray(g);y=np.asarray(q)
                    print("SPREAD","lb",lb,"h",h//1000,"q",sq,"N",len(x),
                          "gross",round(float(x.mean()),4),
                          "quote",round(float(y.mean()),4),
                          "win",round(float(np.mean(y>0)),4),
                          "spread_bps",round(float(spread_q),4))
if __name__=="__main__":main()
