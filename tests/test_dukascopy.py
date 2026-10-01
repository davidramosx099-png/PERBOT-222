import lzma, struct
from perbot222.dukascopy import decode_bi5_daily

def test_decode_daily_xau_shape():
    raw=struct.pack(">IIIff",123,4176123,4176000,1.0,1.2)
    ticks=decode_bi5_daily(lzma.compress(raw),1_000_000_000,1000)
    assert len(ticks)==1 and ticks[0].bid==4176.0 and ticks[0].ask==4176.123