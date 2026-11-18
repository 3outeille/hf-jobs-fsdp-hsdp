"""The gateway and the sidecar against stub replicas: routing, the adapter push, all-or-nothing, broadcast.
    python3 -m pytest tests/test_gateway.py -q        # an HF cpu-basic Job
"""
import json, os, socket, subprocess, sys, threading, time, urllib.error, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
PY = sys.executable
LORAS = Path("/tmp/loras")

def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]

def wait(url, timeout=30):
    """True once the URL answers; False if it never does (a replica that is down answers never)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1).read()
            return True
        except OSError:
            time.sleep(0.05)
    return False

def post(port, path, payload):
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()

def make_adapter(name):
    path = Path("/tmp") / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text("{}")
    (path / "adapter_model.safetensors").write_bytes(b"weights")
    return path

class Stub(ThreadingHTTPServer):
    """A vLLM replica: records every call, answers TRL's endpoints, can refuse the load."""
    daemon_threads = True
    def __init__(self, load_error=None):
        super().__init__(("127.0.0.1", free_port()), self.handler_class())
        self.calls, self.load_error = [], load_error
        threading.Thread(target=self.serve_forever, daemon=True).start()
    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_address[1]}"
    def handler_class(self):
        stub = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def run(self):
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                stub.calls.append((self.command, self.path.split("?")[0], body))
                if self.path.startswith("/v1/load_lora_adapter"):
                    if stub.load_error:
                        return self.reply(stub.load_error, b"no")
                    lora_path = json.loads(body)["lora_path"]
                    assert lora_path.startswith("/tmp/loras/"), f"lora_path is the replica's: {lora_path}"
                if self.path.startswith("/health"):
                    return self.reply(200, b"ok")
                return self.reply(200, b'{"choices":[{"text":"\\boxed{4}"}]}')
            def reply(self, status, payload=b"{}"):
                self.send_response(status)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            do_GET = do_POST = do_PUT = do_DELETE = run
        return Handler

class Stack:
    """Two stub replicas, two real sidecars, one real gateway; extra env for the gateway as keyword arguments."""
    def __init__(self, stubs=None, **env):
        self.stubs = stubs or [Stub(), Stub()]
        self.sidecars = []
        for _ in self.stubs:
            port = free_port()
            self.sidecars.append((port, subprocess.Popen([PY, str(HERE / "sidecar.py"), "--port", str(port)],
                                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)))
            assert wait(f"http://127.0.0.1:{port}/health"), "the sidecar never answered"
        self.port = free_port()
        gateway_env = {**os.environ, "GATEWAY_PORT": str(self.port),
                       "UPSTREAM_URLS": ",".join(s.url for s in self.stubs),
                       "SIDECAR_URLS": ",".join(f"http://127.0.0.1:{port}" for port, _ in self.sidecars), **env}
        self.gateway = subprocess.Popen([PY, str(HERE / "gateway.py")], env=gateway_env,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert wait(f"http://127.0.0.1:{self.port}/health"), "the gateway never answered"
    def close(self):
        self.gateway.kill()
        for _, process in self.sidecars:
            process.kill()
        for stub in self.stubs:
            stub.shutdown()

def test_one_prompt_keeps_its_eight_rollouts_on_one_replica():
    stack = Stack()
    try:
        prompt = [{"role": "user", "content": "1+3?"}]
        for _ in range(8):
            assert post(stack.port, "/v1/completions", {"prompt": prompt})[0] == 200
        assert len([s for s in stack.stubs if any(c[1] == "/v1/completions" for c in s.calls)]) == 1
    finally:
        stack.close()

def test_the_load_tars_to_every_sidecar_and_rewrites_the_path():
    stack = Stack()
    try:
        status, body = post(stack.port, "/v1/load_lora_adapter",
                            {"lora_name": "v1", "lora_path": str(make_adapter("adapter-v1"))})
        assert status == 200, body
        assert (LORAS / "v1" / "adapter_model.safetensors").read_bytes() == b"weights"
        for stub in stack.stubs:
            loads = [c for c in stub.calls if c[1] == "/v1/load_lora_adapter"]
            assert len(loads) == 1 and json.loads(loads[0][2])["lora_path"] == "/tmp/loras/v1"
    finally:
        stack.close()

def test_all_or_nothing_when_a_replica_refuses_the_load():
    stack = Stack()
    try:
        stack.stubs[0].load_error = 500
        status, _ = post(stack.port, "/v1/load_lora_adapter",
                         {"lora_name": "v2", "lora_path": str(make_adapter("adapter-v2"))})
        assert status == 502
        assert any(c[1] == "/v1/unload_lora_adapter" for c in stack.stubs[0].calls), "a loaded replica is unloaded"
        assert not (LORAS / "v2").exists(), "the pushed bytes are deleted again"
    finally:
        stack.close()

def test_broadcast_and_health():
    stack = Stack()
    try:
        for path in ("/pause", "/resume", "/reset_prefix_cache", "/v1/unload_lora_adapter"):
            assert post(stack.port, path, {"lora_name": "v1"})[0] == 200, path
            assert all(any(c[1] == path for c in stub.calls) for stub in stack.stubs), path
        stack.stubs[0].shutdown()
        time.sleep(0.5)
        assert wait(f"http://127.0.0.1:{stack.port}/health", timeout=5) is False, "/health must be 503 if a replica is down"
    finally:
        stack.close()
