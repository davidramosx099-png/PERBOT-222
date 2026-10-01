import csv,sys
from pathlib import Path
p=Path("data/xau_test/xauusd-tick-2026-09-29-2026-09-30.csv")
with p.open(newline="",encoding="utf-8") as f:
 r=csv.DictReader(f); rows=list(r)
print("ROWS",len(rows)); print("COLUMNS",r.fieldnames)
for k in ("timestamp","askPrice","bidPrice"):
 vals=[x[k] for x in rows if x.get(k)]
 print(k,"FIRST",vals[0],"LAST",vals[-1])
b=[float(x["bidPrice"]) for x in rows]; a=[float(x["askPrice"]) for x in rows]
s=[x-y for x,y in zip(a,b)]
print("BID_MIN",min(b),"BID_MAX",max(b))
print("SPREAD_MIN",min(s),"SPREAD_MAX",max(s),"SPREAD_MEDIAN",sorted(s)[len(s)//2])
print("NONPOSITIVE_SPREAD",sum(v<0 for v in s))
t=[int(float(x["timestamp"])) for x in rows]
print("TIMESTAMP_NONMONOTONIC",sum(y<x for x,y in zip(t,t[1:])))
print("DUPLICATE_TIMESTAMPS",sum(y==x for x,y in zip(t,t[1:])))