import csv
import numpy as np
from pathlib import Path

def load(path):
 with Path(path).open(newline="",encoding="utf-8") as f:
  r=list(csv.DictReader(f))
 ts=np.array([int(x["timestamp"]) for x in r],dtype=np.int64)
 bid=np.array([float(x["bidPrice"]) for x in r]); ask=np.array([float(x["askPrice"]) for x in r])
 return ts,(bid+ask)/2,ask-bid

def eval_rule(ts,mid,spread,lookback,horizon_ms,threshold):
 n=len(mid); cut=int(n*.7); entry=[]; ret=[]
 med_spread=np.median(spread[:cut])
 for i in range(lookback,cut):
  j=np.searchsorted(ts,ts[i]+horizon_ms,side="left")
  if j>=n: break
  momentum=mid[i]/mid[i-lookback]-1
  if momentum>threshold and spread[i] <= med_spread*2:
   entry.append(i); ret.append((mid[j]-mid[i]-spread[i])/mid[i])
 r=np.array(ret)
 if len(r)==0:return (0,float("nan"),float("nan"))
 return len(r),float(np.mean(r)*1e4),float(np.mean(r>0))

def main():
 ts,mid,spread=load("data/xau_test/xauusd-tick-2026-09-29-2026-09-30.csv")
 print("TRAIN/TEST split: 70/30 chronological; results shown on TRAIN only for rule discovery")
 for lb in (50,100,250,500,1000,2000):
  for h in (1000,3000,5000,10000):
   for th in (0.00002,0.00005,0.0001,0.0002):
    n,mean,win=eval_rule(ts,mid,spread,lb,h,th)
    if n>=100 and mean>0: print(f"lb={lb} h={h//1000}s th={th:.5f} n={n} mean_net_bps={mean:.3f} win={win:.3f}")
if __name__=="__main__": main()
