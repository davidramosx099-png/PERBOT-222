import csv
import numpy as np
from pathlib import Path

LB=(20,50,100,250,500)
H=(10_000,30_000,60_000)
MIN=100_000
COOLDOWN=60_000

def load(p):
    with Path(p).open(newline="",encoding="utf-8") as f:
        r=csv.DictReader(f); rows=[(int(x["timestamp"]),float(x["bidPrice"]),float(x["askPrice"])) for x in r]
    if not rows:return None
    a=np.asarray(rows,float);return a[:,0].astype(np.int64),a[:,1],a[:,2]

def feat(ts,bid,ask,lb):
    mid=(bid+ask)/2; d=np.diff(mid); k=np.ones(lb)/lb
    return mid,d,np.convolve(d,k,"valid"),np.convolve(np.abs(d),k,"valid")

def run(ts,bid,ask,mid,d,net,activity,h,cut):
    out=[];last=None
    for k in range(len(net)):
        i=k+cut
        if activity[k] < np.quantile(activity[:max(1000,int(len(activity)*.7))],.8):continue
        # mean-reversion after an unusually large negative move
        if net[k] >= -np.quantile(np.abs(net[:max(1000,int(len(net)*.7))]),.8):continue
        cur=int(ts[i])
        if last is not None and cur-last<COOLDOWN:continue
        j=np.searchsorted(ts,cur+h)
        if j>=len(ts):continue
        out.append((bid[j]-ask[i])/ask[i]*1e4);last=cur
    return np.asarray(out)

def main():
    fs=[]
    for f in sorted(Path("data/multi").glob("*.csv")):
        x=load(f)
        if x is not None and len(x[0])>=MIN:fs.append((f,*x))
    print("VALID_DAYS",len(fs))
    for lb in LB:
        allv={h:[] for h in H}
        for oi in range(3,len(fs)):
            # thresholds from preceding 3 sessions only
            ns=[];aa=[]
            for item in fs[max(0,oi-3):oi]:
                ts,bid,ask=item[1:];_,_,n,a=feat(ts,bid,ask,lb);ns.append(n);aa.append(a)
            nq=float(np.quantile(np.concatenate(ns),.20))
            aq=float(np.quantile(np.concatenate(aa),.80))
            ts,bid,ask=fs[oi][1:]
            mid,d,n,a=feat(ts,bid,ask,lb)
            for h in H:
                v=run(ts,bid,ask,mid,d,n,a,h,lb)
                if len(v):allv[h].extend(v)
        for h,v in allv.items():
            if v:
                x=np.asarray(v)
                print("REVERSAL","lb",lb,"h",h//1000,"N",len(x),
                      "mean",round(float(x.mean()),4),"median",round(float(np.median(x)),4),
                      "win",round(float(np.mean(x>0)),4))
if __name__=="__main__":main()
