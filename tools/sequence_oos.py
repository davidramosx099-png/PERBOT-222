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
    s=np.sign(d)
    ret=np.convolve(d,np.ones(lb),mode="valid")
    ab=np.convolve(np.abs(d),np.ones(lb),mode="valid")
    # Number of sign changes in the last lb ticks.
    changes=np.convolve((s[1:]!=s[:-1]).astype(float),np.ones(lb-1),mode="valid")
    # Length of the current same-sign streak, capped at lb.
    streak=np.zeros(len(s),float)
    run=0.0
    prev=0.0
    for i,x in enumerate(s):
        if x==0:
            run=0.0
        elif x==prev:
            run+=1.0
        else:
            run=1.0
        streak[i]=run
        prev=x
    return mid,d,ret,ab,changes,streak[lb-1:]

def run(ts,bid,ask,featv,h,cut,lo,hi,mode):
    out=[];last=None
    for k,x in enumerate(featv):
        i=k+cut
        if mode=="low" and x>lo: continue
        if mode=="high" and x<hi: continue
        cur=int(ts[i])
        if last is not None and cur-last<COOLDOWN: continue
        j=np.searchsorted(ts,cur+h)
        if j>=len(ts): continue
        out.append((bid[j]-ask[i])/ask[i]*1e4)
        last=cur
    return np.asarray(out)

def main():
    fs=[]
    for f in sorted(Path("data/multi").glob("*.csv")):
        x=load(f)
        if x is not None and len(x[0])>=MIN: fs.append((f,*x))
    print("VALID_DAYS",len(fs))
    for lb in LB:
        # Fixed, pre-registered sequence states:
        # 1) low sign-change rate = persistent sequence
        # 2) high sign-change rate = alternating sequence
        # 3) short current streak = no persistence requirement, control state
        for feature in ("changes","streak"):
            allv={h:[] for h in H}
            for oi in range(3,len(fs)):
                train=[]
                for item in fs[max(0,oi-3):oi]:
                    ts,bid,ask=item[1:]
                    vals=feat(ts,bid,ask,lb)
                    train.append(vals[4] if feature=="changes" else vals[5])
                z=np.concatenate(train)
                lo=float(np.quantile(z,.20)); hi=float(np.quantile(z,.80))
                ts,bid,ask=fs[oi][1:]
                vals=feat(ts,bid,ask,lb)
                x=vals[4] if feature=="changes" else vals[5]
                for h in H:
                    # For streak, only the high state is economically directional;
                    # for changes, evaluate both tails without choosing a winner.
                    modes=("low","high") if feature=="changes" else ("high",)
                    for mode in modes:
                        v=run(ts,bid,ask,x,h,lb,lo,hi,mode)
                        if len(v): allv[h].extend(v)
            for h,v in allv.items():
                if v:
                    x=np.asarray(v)
                    print("SEQUENCE","lb",lb,"feature",feature,"h",h//1000,
                          "N",len(x),"mean",round(float(x.mean()),4),
                          "median",round(float(np.median(x)),4),
                          "win",round(float(np.mean(x>0)),4))
if __name__=="__main__": main()
