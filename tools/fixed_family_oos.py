import csv
import numpy as np
from pathlib import Path

LB=(50,100,250,500,1000,2000)
H=(10_000,30_000,60_000)
MIN=100_000
COOLDOWN=60_000

def load(p):
    with Path(p).open(newline="",encoding="utf-8") as f:
        r=csv.DictReader(f); rows=[(int(x["timestamp"]),float(x["bidPrice"]),float(x["askPrice"])) for x in r]
    if not rows:return None
    a=np.asarray(rows,float)
    return a[:,0].astype(np.int64),a[:,1],a[:,2]

def feat(ts,bid,ask,lb):
    d=np.diff((bid+ask)/2); k=np.ones(lb)/lb
    return np.convolve(np.sign(d),k,"valid"),np.convolve(np.abs(d),k,"valid")

def eval_day(ts,bid,ask,p,a,pq,aq,h,kind):
    out=[]; last=None
    for k in range(len(p)):
        i=k+LB_CUR
        if kind=="contra": ok=p[k]<=pq
        elif kind=="momentum": ok=p[k]>=pq
        else: ok=True
        ok=ok and a[k]>=aq
        if not ok:continue
        cur=int(ts[i])
        if last is not None and cur-last<COOLDOWN:continue
        j=np.searchsorted(ts,cur+h)
        if j>=len(ts):continue
        out.append((bid[j]-ask[i])/ask[i]*1e4); last=cur
    return np.asarray(out)

def main():
    global LB_CUR
    fs=[]
    for f in sorted(Path("data/multi").glob("*.csv")):
        x=load(f)
        if x is not None and len(x[0])>=MIN:fs.append((f,*x))
    print("VALID_DAYS",len(fs))
    # Pre-registered families: no daily model selection. Each family is evaluated across every OOS day.
    for lb in (250,500,1000,2000):
        LB_CUR=lb
        for h in H:
            allv=[]
            for _,ts,bid,ask in fs[3:]:
                p,a=feat(ts,bid,ask,lb)
                # threshold is defined from the immediately preceding 3 days only
                idx=fs.index((_,ts,bid,ask)) if False else 0
            # use fixed empirical thresholds pooled from first 3 valid sessions
            pp=[];aa=[]
            for _,ts,bid,ask in fs[:3]:
                p,a=feat(ts,bid,ask,lb);pp.append(p);aa.append(a)
            pq=float(np.quantile(np.concatenate(pp),.20)); aq=float(np.quantile(np.concatenate(aa),.80))
            for kind in ("contra","momentum"):
                vals=[]
                for item in fs[3:]:
                    ts,bid,ask=item[1:];p,a=feat(ts,bid,ask,lb)
                    v=eval_day(ts,bid,ask,p,a,pq,aq,h,kind)
                    if len(v):vals.extend(v)
                if vals:
                    vals=np.asarray(vals)
                    print("FIXED","lb",lb,"h",h//1000,"kind",kind,"N",len(vals),
                          "mean",round(float(vals.mean()),4),"median",round(float(np.median(vals)),4),
                          "win",round(float(np.mean(vals>0)),4))
if __name__=="__main__":main()
