import csv
import numpy as np
from pathlib import Path

LB=(20,50,100,250,500,1000)
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
    p=np.convolve(np.sign(d),np.ones(lb)/lb,mode="valid")
    a=np.convolve(np.abs(d),np.ones(lb)/lb,mode="valid")
    r=np.convolve(d,np.ones(lb),mode="valid")/mid[lb:]*1e4
    return mid,p,a,r

def run(ts,bid,ask,mid,p,a,r,h,cut,p_lo,a_hi,r_lo):
    gross=[];quote=[];last=None
    for k in range(len(p)):
        # Pre-registered state: strong negative pressure + high activity,
        # optionally conditioned on unusually large negative return.
        if p[k]>=p_lo or a[k]<a_hi or r[k]>=r_lo:
            continue
        i=k+cut
        cur=int(ts[i])
        if last is not None and cur-last<COOLDOWN:continue
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
                tp=[];ta=[];tr=[]
                for item in fs[max(0,oi-3):oi]:
                    ts,bid,ask=item[1:]
                    _,p,a,r=feat(ts,bid,ask,lb)
                    tp.append(p);ta.append(a);tr.append(r)
                p_lo=float(np.quantile(np.concatenate(tp),.20))
                a_hi=float(np.quantile(np.concatenate(ta),.80))
                r_lo=float(np.quantile(np.concatenate(tr),.20))
                ts,bid,ask=fs[oi][1:]
                mid,p,a,r=feat(ts,bid,ask,lb)
                g,q=run(ts,bid,ask,mid,p,a,r,h,lb,p_lo,a_hi,r_lo)
                if len(g):gross.extend(g);quote.extend(q)
            if gross:
                g=np.asarray(gross);q=np.asarray(quote)
                print("STATE","lb",lb,"h",h//1000,"N",len(g),
                      "gross",round(float(g.mean()),4),
                      "quote",round(float(q.mean()),4),
                      "win",round(float(np.mean(g>0)),4))
if __name__=="__main__":main()
