import csv
import numpy as np
from pathlib import Path

LOOKBACKS=(50,100,250,500,1000,2000)
HORIZONS=(10_000,30_000,60_000)
MIN_ROWS=100_000
COOLDOWN_MS=60_000

def load(p):
    with Path(p).open(newline="",encoding="utf-8") as f:
        r=csv.DictReader(f); rows=[]
        for x in r:
            rows.append((int(x["timestamp"]),float(x["bidPrice"]),float(x["askPrice"])))
    if not rows: return np.empty(0,dtype=np.int64),np.empty(0),np.empty(0)
    a=np.asarray(rows,dtype=float)
    return a[:,0].astype(np.int64),a[:,1],a[:,2]

def feat(ts,bid,ask,lb):
    d=np.diff((bid+ask)/2)
    k=np.ones(lb)/lb
    p=np.convolve(np.sign(d),k,mode="valid")
    a=np.convolve(np.abs(d),k,mode="valid")
    return p,a

def score(ts,bid,ask,p,a,pq,aq,h,mode):
    vals=[]; last=None
    for k in range(len(p)):
        i=k+mode[0]
        if mode[1]=="contrarian":
            ok=(p[k]<=pq)
        elif mode[1]=="momentum":
            ok=(p[k]>=pq)
        else:
            ok=True
        if mode[2]=="high": ok=ok and a[k]>=aq
        elif mode[2]=="low": ok=ok and a[k]<=aq
        if not ok: continue
        cur=int(ts[i])
        if last is not None and cur-last<COOLDOWN_MS: continue
        j=np.searchsorted(ts,cur+h)
        if j>=len(ts): continue
        vals.append((bid[j]-ask[i])/ask[i]*1e4); last=cur
    if len(vals)<8: return np.nan,len(vals)
    return float(np.mean(vals)),len(vals)

def daily_features(item,lb):
    ts,bid,ask=item
    p,a=feat(ts,bid,ask)
    return ts,bid,ask,p,a

def main():
    files=[]
    for f in sorted(Path("data/multi").glob("*.csv")):
        x=load(f)
        if len(x[0])>=MIN_ROWS: files.append((f,*x))
    print("VALID_DAYS",len(files))
    cache={}
    for di,item in enumerate(files):
        cache[di]={}
        ts,bid,ask=item[1:]
        for lb in LOOKBACKS:
            p,a=feat(ts,bid,ask,lb)
            cache[di][lb]=(p,a)
    results=[]
    for oi in range(4,len(files)):
        prior=files[:oi]
        split=max(2,int(len(prior)*0.6))
        fit_idx=list(range(split)); val_idx=list(range(split,oi))
        candidates=[]
        for lb in LOOKBACKS:
            fit_arrays=[cache[d][lb] for d in fit_idx]
            pp=[x[0] for x in fit_arrays]; aa=[x[1] for x in fit_arrays]
            aq=float(np.quantile(np.concatenate(aa),.80))
            for h in HORIZONS:
                for mode_name in ("contrarian","momentum","neutral"):
                    pq=float(np.quantile(np.concatenate(pp),.20)) if mode_name=="contrarian" else float(np.quantile(np.concatenate(pp),.80))
                    for act in ("high","low","all"):
                        mode=(lb,mode_name,act)
                        valv=[]
                        for d in val_idx:
                            item=files[d]; ts,bid,ask=item[1:]
                            p,a=cache[d][lb]
                            v,n=score(ts,bid,ask,p,a,pq,aq,h,mode)
                            if np.isfinite(v): valv.extend([v]*n)
                        if valv:
                            candidates.append((float(np.mean(valv)),mode,h,lb,len(valv)))
        candidates.sort(key=lambda z:z[0],reverse=True)
        best=candidates[0]
        meanv,mode,h,lb,_=best
        aa=[]; pp=[]
        for d in range(oi):
            p,a=cache[d][lb]; pp.append(p); aa.append(a)
        aq=float(np.quantile(np.concatenate(aa),.80))
        pq=float(np.quantile(np.concatenate(pp),.20)) if mode[1]=="contrarian" else float(np.quantile(np.concatenate(pp),.80))
        item=files[oi]; ts,bid,ask=item[1:]
        p,a=cache[oi][lb]
        v,n=score(ts,bid,ask,p,a,pq,aq,h,mode)
        print("OOS",item[0].name,"SELECTED","lb",lb,"h",h//1000,"mode",mode[1],"activity",mode[2],
              "VAL",round(meanv,4),"N",n,"OOS",round(v,4) if np.isfinite(v) else None)
        if np.isfinite(v): results.extend([v]*n)
    if results:
        print("AGGREGATE_SELECTED_OOS","N",len(results),"mean",round(float(np.mean(results)),4),
              "median",round(float(np.median(results)),4),"win",round(float(np.mean(np.asarray(results)>0)),4))

if __name__=="__main__": main()
