import csv
import numpy as np
from pathlib import Path

LB=(20,50,100,250,500)
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
    ret=np.convolve(d,np.ones(lb),mode="valid")/mid[lb:]*1e4
    return mid,ret

def run(ts,bid,ask,mid,ret,h,cut,lo,hi,side):
    g=[];q=[];last=None
    for k,x in enumerate(ret):
        if side=="down" and x>=lo:continue
        if side=="up" and x<=hi:continue
        i=k+cut;cur=int(ts[i])
        if last is not None and cur-last<COOLDOWN:continue
        j=np.searchsorted(ts,cur+h)
        if j>=len(ts):continue
        # Evaluate continuation in the direction of the event.
        if side=="up":
            g.append((mid[j]/mid[i]-1)*1e4)
            q.append((bid[j]-ask[i])/ask[i]*1e4)
        else:
            g.append((mid[i]/mid[j]-1)*1e4)
            q.append((ask[i]-bid[j])/bid[j]*1e4)
        last=cur
    return np.asarray(g),np.asarray(q)

def main():
    fs=[]
    for f in sorted(Path("data/multi").glob("*.csv")):
        x=load(f)
        if x is not None and len(x[0])>=MIN:fs.append((f,*x))
    print("VALID_DAYS",len(fs))
    for lb in LB:
        for h in H:
            for side in ("up","down"):
                g=[];q=[]
                for oi in range(3,len(fs)):
                    tr=[]
                    for item in fs[oi-3:oi]:
                        ts,bid,ask=item[1:]
                        _,r=feat(ts,bid,ask,lb);tr.append(r)
                    z=np.concatenate(tr)
                    lo=float(np.quantile(z,.20));hi=float(np.quantile(z,.80))
                    ts,bid,ask=fs[oi][1:]
                    mid,r=feat(ts,bid,ask,lb)
                    gg,qq=run(ts,bid,ask,mid,r,h,lb,lo,hi,side)
                    if len(gg):g.extend(gg);q.extend(qq)
                if g:
                    x=np.asarray(g);y=np.asarray(q)
                    print("EVENT","lb",lb,"h",h//1000,"side",side,"N",len(x),
                          "gross",round(float(x.mean()),4),
                          "quote",round(float(y.mean()),4),
                          "win",round(float(np.mean(y>0)),4))
if __name__=="__main__":main()
