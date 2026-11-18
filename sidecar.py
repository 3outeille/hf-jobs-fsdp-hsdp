"""The replica's adapter cache on :8100, stdlib only: PUT /adapters/<name> (tar body) -> /tmp/loras/<name>, DELETE the
same path, GET /health. A tar is extracted to a temp dir and renamed into place, so a reader sees all of it or none.
"""
import argparse, io, shutil, tarfile, tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path("/tmp/loras")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def reply(self, status, body=b""):
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def name(self):
        parts = [part for part in self.path.split("?")[0].split("/") if part]
        return parts[1] if len(parts) == 2 and parts[0] == "adapters" else None

    def do_PUT(self):
        name = self.name()
        if name is None:
            return self.reply(400)
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        staging = Path(tempfile.mkdtemp(dir=ROOT, prefix=".stage-"))
        try:
            with tarfile.open(fileobj=io.BytesIO(body), mode="r:*") as archive:
                archive.extractall(staging, filter="data")
        except (tarfile.TarError, ValueError):
            shutil.rmtree(staging, ignore_errors=True)
            return self.reply(400)
        target = ROOT / name
        shutil.rmtree(target, ignore_errors=True)
        staging.rename(target)
        print(f"[sidecar] put {name} {len(body)} B -> {target}", flush=True)
        self.reply(200, b'{"ok": true}')

    def do_DELETE(self):
        name = self.name()
        shutil.rmtree(ROOT / name, ignore_errors=True) if name else None
        print(f"[sidecar] delete {name}", flush=True)
        self.reply(204 if name else 400)

    def do_GET(self):
        self.reply(200, b'{"ok": true}') if self.path.startswith("/health") else self.reply(404)


ROOT.mkdir(parents=True, exist_ok=True)
parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8100)
print(f"[sidecar] :{parser.parse_args().port} -> {ROOT}", flush=True)
ThreadingHTTPServer(("0.0.0.0", parser.parse_args().port), Handler).serve_forever()
