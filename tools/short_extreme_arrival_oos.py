import csv
import numpy as np
from pathlib import Path

LB=(250,500,1000,2000); H=(30_000,60_000)
Q_RATE=(0.80,0.90,0.95); Q_PRESS=(0.80,0.85,0.90,0.95)
MIN_ROWS=100_000; COOLDOWN=60_000

def load(p):
    with Path(p).open(newline="",encoding="utf-8") as f:
        r=csv.DictReader(f); rows=[(int(x["timestamp"]),float(x["bidPrice"]),float(x["askPrice"])) for x in r]
    if not rows:return None
    a=np.asarray(rows,float); return a[:,0].astype(np.int64),a[:,1],a[:,2]

def feat(ts,bid,ask,lb):
    mid=(bid+ask)/2; dt=np.diff(ts).astype(float); d=np.diff(mid)
    sec=np.maximum(np.convolve(dt,np.ones(lb),mode="valid"),1.0)
    rate=1000.0*lb/sec; move=np.convolve(d,np.ones(lb),mode="valid")
    pressure=move/np.maximum(np.convolve(np.abs(d),np.ones(lb),mode="valid"),1e-12)
    return mid,rate,pressure

def run(ts,bid,ask,mid,rate,pressure,h,lb,rc,pc):
    gross=[]; quote=[]; last=None
    for k in range(len(rate)):
        if rate[k]<rc or pressure[k]<pc: continue
        i=k+lb; cur=int(ts[i])
        if last is not None and cur-last<COOLDOWN: continue
        j=np.searchsorted(ts,cur+h)
        if j>=len(ts): continue
        gross.append((mid[i]/mid[j]-1.0)*1e4)
        quote.append((bid[i]-ask[j])/bid[i]*1e4)
        last=cur
    return np.asarray(gross),np.asarray(quote)

def main():
    fs=[]
    for f in sorted(Path("data/multi").glob("*.csv")):
        x=load(f)
        if x is not None and len(x[0])>=MIN_ROWS: fs.append((f,*x))
    print("VALID_DAYS",len(fs))
    print("PROTOCOL prior-3-days thresholds | high arrival + extreme positive pressure | quote-aware SELL")
    for lb in LB:
        for h in H:
            for rq in Q_RATE:
                for pq in Q_PRESS:
                    ga=[];qa=[]
                    for oi in range(3,len(fs)):
                        train=[feat(item[1],item[2],item[3],lb)[1:] for item in fs[oi-3:oi]]
                        rc=float(np.quantile(np.concatenate([x[0] for x in train]),rq))
                        pc=float(np.quantile(np.concatenate([x[1] for x in train]),pq))
                        ts,bid,ask=fs[oi][1:]; mid,rate,pressure=feat(ts,bid,ask,lb)
                        g,q=run(ts,bid,ask,mid,rate,pressure,h,lb,rc,pc)
                        if len(g): ga.extend(g);qa.extend(q)
                    if qa:
                        g=np.asarray(ga);q=np.asarray(qa)
                        print("SHORT_EXTREME","lb",lb,"h",h//1000,"rate_q",rq,"press_q",pq,"N",len(q),"gross",round(float(g.mean()),4),"quote",round(float(q.mean()),4),"win",round(float(np.mean(q>0)),4))
if __name__=="__main__": main()
