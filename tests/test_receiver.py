import json
import threading
import urllib.request

from perbot222.receiver import serve


def test_receiver_accepts_tick(tmp_path):
    path = tmp_path / "ticks.csv"

    # Bind an ephemeral port by constructing the server indirectly through a
    # short-lived thread is intentionally avoided here; test the handler logic
    # through the public endpoint in a subprocess-like thread.
    from http.server import ThreadingHTTPServer
    from perbot222.receiver import TickHandler
    from perbot222.ingest import TickStore

    TickHandler.store = TickStore(path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), TickHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        payload = json.dumps({"bid": 2000.0, "ask": 2000.02}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/tick",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2) as response:
            assert response.status == 200
        assert len(TickStore(path).read_recent(10)) == 1
    finally:
        server.shutdown()
        server.server_close()
