from __future__ import annotations
import lzma
import struct
import urllib.request
from datetime import datetime, timezone, timedelta
from .ingest import Tick, TickStore

BASE = "https://datafeed.dukascopy.com/datafeed"\n\n# Direct HTTP archive access is legacy and may return 503. Current Dukascopy\n# historical export uses S3 Requester Pays; keep this module only as a decoder\n# until authenticated S3 ingestion is configured.\nXAUUSD_POINT_VALUE = None

def decode_bi5_daily(payload: bytes, day_start_ms: int, point_value: float) -> list[Tick]:
    raw = lzma.decompress(payload)
    if len(raw) % 20: raise ValueError("invalid BI5 payload length")
    out = []
    for i in range(0, len(raw), 20):
        ms, ask_i, bid_i, ask_v, bid_v = struct.unpack(">IIIff", raw[i:i+20])
        out.append(Tick(timestamp_ns=(day_start_ms + ms)*1_000_000, bid=bid_i/point_value, ask=ask_i/point_value, volume=float((ask_v+bid_v)/2.0)))
    return out

def download_day(symbol: str, day: datetime, point_value: float, timeout: int = 30) -> list[Tick]:
    day = day.astimezone(timezone.utc)
    url = f"{BASE}/{symbol.upper()}/{day:%Y}/{day.month-1:02d}/{day.day:02d}_ticks.bi5"
    req = urllib.request.Request(url, headers={"User-Agent":"PERBOT-222/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r: payload = r.read()
    day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
    return decode_bi5_daily(payload, int(day_start.timestamp())*1000, point_value)

def download_range(start: datetime, end: datetime, store: TickStore, point_value: float) -> tuple[int,int]:
    cur=start.astimezone(timezone.utc).replace(hour=0,minute=0,second=0,microsecond=0); end=end.astimezone(timezone.utc).replace(hour=0,minute=0,second=0,microsecond=0)
    total=missing=0
    while cur<=end:
        try:
            ticks=download_day("XAUUSD",cur,point_value)
            for t in ticks: store.append(t)
            total+=len(ticks)
        except Exception as exc:
            if getattr(exc,"code",None)==404: missing+=1
            else: raise
        cur+=timedelta(days=1)
    return total,missing