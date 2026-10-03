import csv
import numpy as np
from pathlib import Path

LB=(500,1000,2000); H=(30_000,60_000); RQ=(0.90,0.95); PQ=(0.05,0.10,0.15,0.20)
TP=(0.5,1.0,1.5,2.0); SL=(1.0,1.5,2.0,3.0); MIN_ROWS=100_000; COOLDOWN=60_000

def load(p):
    with Path(p).open(newline="",encoding="utf-8") as f:
        r=csv.DictReader(f); rows=[(int(x["timestamp"]),float(x["bidPrice"]),float(x["askPrice"])) for x in r]
    if not rows:return None
    a=np.asarray(rows,float); return a[:,0].astype(np.int64),a[:,1],a[:,2]

def feat(ts,bid,ask,lb):
    mid=(bid+ask)/2; dt=np.diff(ts).astype(float); d=np.diff(mid)
    sec=np.maximum(np.convolve(dt,np.ones(lb),mode="valid"),1.0)
    rate=1000.0*lb/sec; move=np.convolve(d,np.ones(lb),mode="valid")
    absm=np.convolve(np.abs(d),np.ones(lb),mode="valid")
    return rate,move/np.maximum(absm,1e-12)

def thresholds(train,lb,rq,pq):
    rr=[]; pp=[]
    for ts,bid,ask in train:
        r,p=feat(ts,bid,ask,lb); rr.append(r); pp.append(p)
    return float(np.quantile(np.concatenate(rr),rq)),float(np.quantile(np.concatenate(pp),pq))

def run(ts,bid,ask,rate,pressure,h,lb,rc,pc,tp,sl):
    out=[]; last=None
    for k in range(len(rate)):
        if rate[k]<rc or pressure[k]>pc: continue
        i=k+lb; cur=int(ts[i])
        if last is not None and cur-last<COOLDOWN: continue
        end=np.searchsorted(ts,cur+h)
        if end>=len(ts): continue
        entry=ask[i]; tp_px=entry*(1+tp/1e4); sl_px=entry*(1-sl/1e4); exit_b=bid[end]
        for j in range(i+1,end+1):
            if bid[j]>=tp_px or bid[j]<=sl_px:
                exit_b=bid[j]; break
        out.append((exit_b/entry-1)*1e4); last=cur
    return out

def eval_causal(fs,c,start,end):
    lb,h,rq,pq,tp,sl=c; vals=[]; day_ns=[]
    for oi in range(start,end):
        tr=fs[oi-3:oi]
        rc,pc=thresholds(tr,lb,rq,pq)
        ts,bid,ask=fs[oi]; r,p=feat(ts,bid,ask,lb)
        z=run(ts,bid,ask,r,p,h,lb,rc,pc,tp,sl)
        vals.extend(z); day_ns.append(len(z))
    return vals,day_ns

def main():
    fs=[]
    for f in sorted(Path("data/multi").glob("*.csv")):
        x=load(f)
        if x is not None and len(x[0])>=MIN_ROWS: fs.append(x)
    print("VALID_DAYS",len(fs),flush=True)
    if len(fs)<12: raise SystemExit("not enough valid sessions")
    # Freeze selection using first 8 valid sessions only.
    dev_end=8
    candidates=[(lb,h,rq,pq,tp,sl) for lb in LB for h in H for rq in RQ for pq in PQ for tp in TP for sl in SL]
    scores=[]
    for c in candidates:
        vals,_=eval_causal(fs,c,3,dev_end)
        if vals:scores.append((float(np.mean(vals)),c,len(vals)))
    scores.sort(key=lambda x:x[0],reverse=True); chosen=scores[0]; c=chosen[1]
    print("FROZEN_SELECTION","lb",c[0],"h",c[1]//1000,"rate_q",c[2],"press_q",c[3],"tp",c[4],"sl",c[5],"DEV_N",chosen[2],"DEV_MEAN_BPS",round(chosen[0],4),flush=True)
    final,day_ns=eval_causal(fs,c,dev_end,len(fs))
    a=np.asarray(final,float) if final else np.asarray([],float)
    print("EXPANDED_ROLLING","N",len(a),"MEAN_BPS",round(float(a.mean()),4) if len(a) else "nan","MEDIAN_BPS",round(float(np.median(a)),4) if len(a) else "nan","WIN",round(float(np.mean(a>0)),4) if len(a) else "nan","DAYS",len(day_ns),"DAY_N",day_ns,flush=True)

if __name__=="__main__": main()
