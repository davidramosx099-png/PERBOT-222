from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from time import time_ns

from .ingest import Tick, TickStore


class TickHandler(BaseHTTPRequestHandler):
    store: TickStore

    def do_POST(self):
        if self.path != "/tick":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 16_384:
                raise ValueError("invalid body length")
            body = json.loads(self.rfile.read(length))
            ts = int(body.get("timestamp_ns", time_ns()))
            tick = Tick(
                timestamp_ns=ts,
                bid=float(body["bid"]),
                ask=float(body["ask"]),
                volume=None if body.get("volume") is None else float(body["volume"]),
            )
            self.store.append(tick)
            payload = json.dumps({"ok": True, "timestamp_ns": ts}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.send_error(400, str(exc))

    def log_message(self, fmt, *args):
        return


def serve(host: str = "127.0.0.1", port: int = 8765, path: str = "data/ticks.csv"):
    store = TickStore(path)
    TickHandler.store = store
    server = ThreadingHTTPServer((host, port), TickHandler)
    print(f"PERBOT-222 tick receiver listening on http://{host}:{port}/tick")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
