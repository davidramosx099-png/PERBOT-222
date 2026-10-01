import csv
import subprocess
from datetime import date, timedelta
from pathlib import Path

def run(start: date, days: int, out: str = "data/multi"):
    Path(out).mkdir(parents=True, exist_ok=True)
    for k in range(days):
        d=start+timedelta(days=k)
        e=d+timedelta(days=1)
        cmd=["npx","dukascopy-node","-i","xauusd","-from",d.isoformat(),
             "-to",e.isoformat(),"-t","tick","-f","csv","-dir",out]
        print("DOWNLOADING",d.isoformat(),e.isoformat())
        subprocess.run(cmd,check=True)
    files=sorted(Path(out).glob("*.csv"))
    print("FILES",len(files))
    for p in files:
        with p.open(newline="",encoding="utf-8") as f:
            n=sum(1 for _ in f)-1
        print(p.name,"rows",n)

if __name__=="__main__":
    run(date(2026,9,20),7)
