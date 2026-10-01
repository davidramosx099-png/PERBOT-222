import csv
import sys
from datetime import date,timedelta
from pathlib import Path
import subprocess

def main():
    start=date.fromisoformat(sys.argv[1]) if len(sys.argv)>1 else date(2026,9,7)
    days=int(sys.argv[2]) if len(sys.argv)>2 else 10
    out=sys.argv[3] if len(sys.argv)>3 else "data/multi"
    Path(out).mkdir(parents=True,exist_ok=True)
    for k in range(days):
        d=start+timedelta(days=k); e=d+timedelta(days=1)
        subprocess.run(["npx","dukascopy-node","-i","xauusd","-from",d.isoformat(),
                        "-to",e.isoformat(),"-t","tick","-f","csv","-dir",out],check=True)
        p=Path(out)/f"xauusd-tick-{d.isoformat()}-{e.isoformat()}.csv"
        if p.exists():
            with p.open(encoding="utf-8") as f: n=max(0,sum(1 for _ in f)-1)
            print(d.isoformat(),"rows",n)
if __name__=="__main__": main()
