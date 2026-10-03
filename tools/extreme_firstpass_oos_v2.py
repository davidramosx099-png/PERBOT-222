import csv
import numpy as np
from pathlib import Path

LB=(500,1000,2000)
H=(30_000,60_000)
RQ=(0.90,0.95)
PQ=(0.05,0.10,0.15,0.20)
TP=(0.5,1.0,1.5,2.0)
SL=(1.0,1.5,2.0,3.0)
MIN_ROWS=100_000
COOLDOWN=60_000

def load(p):
    with Path(p).open(newline="",encoding="utf-8") as f:
        r=csv.DictReader(f)
        rows=[(int(x["timestamp"]),float(x["bidPrice"]),float(x["askPrice"])) for x in r]
    if not rows:return None
    a=np.asarray(rows,float)
    return a[:,0].astype(np.int64),a[:,1],a[:,2]

def feat(ts,bid,ask,lb):
    mid=(bid+ask)/2.0
    dt=np.diff(ts).astype(float)
    d=np.diff(mid)
    sec=np.maximum(np.convolve(dt,np.ones(lb),mode="valid"),1.0)
    rate=1000.0*lb/sec
    move=np.convolve(d,np.ones(lb),mode="valid")
    absm=np.convolve(np.abs(d),np.ones(lb),mode="valid")
    pressure=move/np.maximum(absm,1e-12)
    return rate,pressure

def run(ts,bid,ask,rate,pressure,h,lb,rc,pc,tp,sl):
    res=[]; last=None
    for k in range(len(rate)):
        if rate[k]<rc or pressure[k]>pc: continue
        i=k+lb; cur=int(ts[i])
        if last is not None and cur-last<COOLDOWN: continue
        end=np.searchsorted(ts,cur+h)
        if end>=len(ts): continue
        entry=ask[i]
        tp_price=entry*(1.0+tp/1e4)
        sl_price=entry*(1.0-sl/1e4)
        outcome=0
        exit_b=bid[end]
        for j in range(i+1,end+1):
            if bid[j]>=tp_price:
                outcome=1; exit_b=bid[j]; break
            if bid[j]<=sl_price:
                outcome=-1; exit_b=bid[j]; break
        realized=(exit_b/entry-1.0)*1e4
        res.append((outcome,realized))
        last=cur
    return res

def main():
    fs=[]
    for f in sorted(Path("data/multi").glob("*.csv")):
        x=load(f)
        if x is not None and len(x[0])>=MIN_ROWS: fs.append((f,*x))
    print("VALID_DAYS",len(fs),flush=True)
    print("PROTOCOL prior-3-days thresholds | first passage | executable quote-aware BUY",flush=True)
    for lb in LB:
      for h in H:
       for rq in RQ:
        for pq in PQ:
         for tp in TP:
          for sl in SL:
           allr=[]
           for oi in range(3,len(fs)):
            tr=[]
            for item in fs[oi-3:oi]:
             ts,bid,ask=item[1:]
             rate,pressure=feat(ts,bid,ask,lb)
             tr.append((rate,pressure))
            rates=np.concatenate([z[0] for z in tr])
            press=np.concatenate([z[1] for z in tr])
            rc=float(np.quantile(rates,rq)); pc=float(np.quantile(press,pq))
            ts,bid,ask=fs[oi][1:]
            rate,pressure=feat(ts,bid,ask,lb)
            z=run(ts,bid,ask,rate,pressure,h,lb,rc,pc,tp,sl)
            allr.extend(z)
           if allr:
            z=np.asarray(allr,float)
            outcomes=z[:,0]
            realized=z[:,1]
            print("FIRSTPASS","lb",lb,"h",h//1000,"rate_q",rq,"press_q",pq,
                  "tp",tp,"sl",sl,"N",len(z),
                  "hit",round(float(np.mean(outcomes==1)),4),
                  "stop",round(float(np.mean(outcomes==-1)),4),
                  "timeout",round(float(np.mean(outcomes==0)),4),
                  "realized_bps",round(float(realized.mean()),4),
                  "median_bps",round(float(np.median(realized)),4),flush=True))
if __name__=="__main__": main()
