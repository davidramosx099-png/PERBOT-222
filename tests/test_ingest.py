from pathlib import Path

import pytest

from perbot222.ingest import Tick, TickStore


def test_store_roundtrip(tmp_path: Path):
    store = TickStore(tmp_path / "ticks.csv")
    store.append(Tick(1_700_000_000_000_000_000, 2000.0, 2000.02))
    store.append(Tick(1_700_000_000_100_000_000, 2000.01, 2000.03, 2.0))
    ticks = store.read_recent(10)
    assert len(ticks) == 2
    assert ticks[-1].bid == 2000.01
    assert ticks[-1].volume == 2.0


def test_rejects_bad_spread(tmp_path: Path):
    store = TickStore(tmp_path / "ticks.csv")
    with pytest.raises(ValueError):
        store.append(Tick(1_700_000_000_000_000_000, 2000.1, 2000.0))
