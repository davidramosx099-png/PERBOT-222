import csv
from perbot222.csv_import import import_csv
from perbot222.ingest import TickStore

def test_import_tick_csv(tmp_path):
    p=tmp_path/"xau.csv"
    with p.open("w", newline="") as f:
        w=csv.writer(f); w.writerow(["timestamp","askPrice","bidPrice","askVolume","bidVolume"]); w.writerow([1000,4176.02,4176.00,1,2])
    store=TickStore(tmp_path/"ticks.csv")
    assert import_csv(p,store)==1
    assert store.read_recent(1)[0].bid==4176.0