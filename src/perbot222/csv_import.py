from __future__ import annotations
import csv
from pathlib import Path
from .ingest import Tick, TickStore

def import_csv(path: str | Path, store: TickStore) -> int:
    count = 0
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"timestamp", "askPrice", "bidPrice"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(f"missing columns: {sorted(required - set(reader.fieldnames or []))}")
        for row in reader:
            ts = int(float(row["timestamp"]))
            ask = float(row["askPrice"])
            bid = float(row["bidPrice"])
            av = row.get("askVolume")
            bv = row.get("bidVolume")
            volume = None
            if av not in (None, "") or bv not in (None, ""):
                volume = (float(av or 0) + float(bv or 0)) / 2.0
            store.append(Tick(timestamp_ns=ts * 1_000_000, bid=bid, ask=ask, volume=volume))
            count += 1
    return count