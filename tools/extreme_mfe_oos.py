import csv
import numpy as np
from pathlib import Path

LB=(500,1000,2000)
H=(30_000,60_000)
RQ=(0.90,0.95)
PQ=(0.05,0.10,0.15,0.20)
TARGET=(0.5,1.0,1.5,2.0)
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
    return mid,rate,pressure

def run(ts,bid,ask,mid,rate,pressure,h,lb,rc,pc):
    out=[]
    last=None
    for k in range(len(rate)):
        if rate[k]<rc or pressure[k]>pc: continue
        i=k+lb
        cur=int(ts[i])
        if last is not None and cur-last<COOLDOWN: continue
        end=np.searchsorted(ts,cur+h)
        if end>=len(ts): continue
        future_bid=bid[i+1:end+1]
        if len(future_bid)==0: continue
        entry=ask[i]
        mfe=(np.max(future_bid)/entry-1.0)*1e4
        endpoint=(bid[end]/entry-1.0)*1e4
        out.append((mfe,endpoint))
        last=cur
    return np.asarray(out)

def main():
    fs=[]
    for f in sorted(Path("data/multi").glob("*.csv")):
        x=load(f)
        if x is not None and len(x[0])>=MIN_ROWS: fs.append((f,*x))
    print("VALID_DAYS",len(fs))
    print("PROTOCOL prior-3-days thresholds | extreme arrival+negative pressure | quote-aware MFE")
    for lb in LB:
      for h in H:
       for rq in RQ:
        for pq in PQ:
          vals=[]
          for oi in range(3,len(fs)):
            train=[]
            for item in fs[oi-3:oi]:
              ts,bid,ask=item[1:]
              _,rate,pressure=feat(ts,bid,ask,lb)
              train.append((rate,pressure))
            rates=np.concatenate([z[0] for z in train])
            press=np.concatenate([z[1] for z in train])
            rc=float(np.quantile(rates,rq))
            pc=float(np.quantile(press,pq))
            ts,bid,ask=fs[oi][1:]
            mid,rate,pressure=feat(ts,bid,ask,lb)
            z=run(ts,bid,ask,mid,rate,pressure,h,lb,rc,pc)
            if len(z): vals.extend(z.tolist())
          if vals:
            z=np.asarray(vals)
            line=["MFE","lb",lb,"h",h//1000,"rate_q",rq,"press_q",pq,"N",len(z),
                  "mfe_mean",round(float(z[:,0].mean()),4),
                  "mfe_med",round(float(np.median(z[:,0])),4),
                  "endpoint",round(float(z[:,1].mean()),4)]
            for t in TARGET:
              line += [f"hit_{t}",round(float(np.mean(z[:,0]>=t)),4)]
            print(*line)
if __name__=="__main__": main()
