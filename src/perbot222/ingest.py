from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Tick:
    timestamp_ns: int
    bid: float
    ask: float
    volume: float | None = None


class TickStore:
    """Append-only CSV tick store with strict validation and no pandas dependency."""

    header = ("timestamp_ns", "bid", "ask", "volume")

    def __init__(self, path: str | Path):
        self.path = Path(path)

    @staticmethod
    def validate(tick: Tick) -> Tick:
        if tick.timestamp_ns <= 0:
            raise ValueError("timestamp_ns must be positive")
        if tick.bid <= 0 or tick.ask <= 0:
            raise ValueError("bid and ask must be positive")
        if tick.ask < tick.bid:
            raise ValueError("ask < bid")
        if tick.volume is not None and tick.volume < 0:
            raise ValueError("volume must be non-negative")
        return tick

    def append(self, tick: Tick) -> None:
        self.validate(tick)
        new_file = not self.path.exists() or self.path.stat().st_size == 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(self.header)
            writer.writerow((tick.timestamp_ns, tick.bid, tick.ask, "" if tick.volume is None else tick.volume))

    def read_recent(self, limit: int = 1000) -> list[Tick]:
        if limit < 1:
            raise ValueError("limit must be positive")
        if not self.path.exists():
            return []
        with self.path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        rows = rows[-limit:]
        out: list[Tick] = []
        for row in rows:
            out.append(Tick(
                timestamp_ns=int(row["timestamp_ns"]),
                bid=float(row["bid"]),
                ask=float(row["ask"]),
                volume=None if row["volume"] == "" else float(row["volume"]),
            ))
        return out


def ticks_to_arrays(ticks: Iterable[Tick]):
    import numpy as np

    items = list(ticks)
    if not items:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
        )
    return (
        np.asarray([x.timestamp_ns for x in items], dtype=np.int64),
        np.asarray([x.bid for x in items], dtype=float),
        np.asarray([x.ask for x in items], dtype=float),
    )
