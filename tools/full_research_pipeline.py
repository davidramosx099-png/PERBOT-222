from __future__ import annotations
import subprocess, sys, time
from pathlib import Path

SCRIPTS=[
"tools/audit_csv.py",
"tools/baseline_numpy.py",
"tools/conditional_baseline.py",
"tools/microstructure_baseline.py",
"tools/acceleration_baseline.py",
"tools/information_baseline.py",
"tools/quantile_oos.py",
"tools/mid_vs_quote_oos.py",
"tools/state_grid_oos.py",
"tools/spread_filter_oos.py",
"tools/event_move_oos.py",
"tools/volatility_regime_oos.py",
"tools/arrival_rate_oos.py",
"tools/arrival_direction_oos.py",
"tools/extreme_arrival_direction_oos.py",
"tools/extreme_mfe_oos.py",
"tools/extreme_firstpass_oos_v2.py",
"tools/frozen_firstpass_holdout.py",
"tools/frozen_rolling_holdout.py",
"tools/expanded_rolling_holdout.py",
]
def main():
    root=Path.cwd()
    report=[]
    for script in SCRIPTS:
        p=root/script
        if not p.exists():
            report.append(f"SKIP {script}: missing")
            continue
        t=time.time()
        print(f"\n=== START {script} ===",flush=True)
        r=subprocess.run([sys.executable,str(p)],text=True,capture_output=True)
        elapsed=time.time()-t
        print(r.stdout,flush=True)
        if r.stderr: print(r.stderr,flush=True)
        status="PASS" if r.returncode==0 else "FAIL"
        report.append(f"{status} {script} rc={r.returncode} seconds={elapsed:.1f}")
        print(f"=== END {script}: {status} ===",flush=True)
    print("\n=== PIPELINE SUMMARY ===",flush=True)
    for x in report: print(x,flush=True)
if __name__=="__main__": main()
