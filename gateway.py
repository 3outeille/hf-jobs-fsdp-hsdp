"""The trainer's server, one hop from TRL: fans generation out to the vLLM replicas and ships the adapter to all of them.

Runs on :8000 in the trainer Job's rank 0 and replaces the blog's `lora_proxy.py` and our old `adapter_gateway.py`. TRL's
async GRPO path (`trl/experimental/async_grpo/vllm_client.py` at the pinned SHA) calls exactly these:

    GET  /health                      ok only when every replica is ok
    GET  /v1/models, /server_info, /get_world_size    one replica
    POST /v1/completions              generation: the eight rollouts of a prompt stay on one replica
    POST /v1/load_lora_adapter        tar the directory -> PUT to every sidecar -> load on every replica, all-or-nothing
    POST /v1/unload_lora_adapter, /pause, /resume, /reset_prefix_cache    broadcast
    anything else                     least-loaded replica (the weight-transfer endpoints of the merged path)

Replicas are addressed by their network alias with the group prefix (`nwa-<group>-vllm1`), which is what a Job sees inside
its network group; the sidecar of a replica is the same host on :8100.
"""

import asyncio
import hashlib
import io
import json
import os
import tarfile
import time
from pathlib import Path

from aiohttp import ClientSession, ClientTimeout, TCPConnector, web

HOP_BY_HOP = {"host", "connection", "keep-alive", "trailers", "transfer-encoding", "upgrade", "content-length"}
BROADCAST = {"/v1/unload_lora_adapter", "/pause", "/resume", "/reset_prefix_cache"}
ONE_REPLICA = {"/v1/models", "/server_info", "/get_world_size"}
LOAD = "/v1/load_lora_adapter"
IMBALANCE = int(os.environ.get("PROXY_IMBALANCE", "8"))
# TRL keeps MAX_INFLIGHT=384 rollouts in flight: aiohttp's default connector (100 connections) would cap generation, so
# the pool is unlimited and the timeouts are objects (a bare float is a deprecated total timeout).
TIMEOUT = ClientTimeout(total=float(os.environ.get("UPSTREAM_TIMEOUT_S", "1800")), connect=30)


def config(env=os.environ):
    upstreams = [u.strip().rstrip("/") for u in env.get("UPSTREAM_URLS", "").split(",") if u.strip()]
    if not upstreams:
        # Inside a Job, a replica is reachable by its alias under the group's prefix (`nwa-<group>-vllm1`); the platform
        # exports that prefix to every Job in the group.
        # The platform's prefix already ends with '-' (`nwa-<group>-`): no second dash, or the name does not resolve.
        prefix = env.get("HF_NETWORK_GROUP_PREFIX", "")
        upstreams = [f"http://{prefix}vllm{i}:8000" for i in range(1, int(env.get("REPLICAS", "3")) + 1)]
    if not upstreams or "None" in upstreams[0]:
        raise SystemExit("no replica URLs: set UPSTREAM_URLS, or run inside the network group")
    sidecars = [f"http://{u.split('://', 1)[1].rsplit(':', 1)[0]}:{env.get('SIDECAR_PORT', '8100')}"
                for u in upstreams]
    explicit = [u.strip().rstrip("/") for u in env.get("SIDECAR_URLS", "").split(",") if u.strip()]
    sidecars = explicit or sidecars
    return {"upstreams": upstreams, "sidecars": sidecars, "port": int(env.get("GATEWAY_PORT", "8000")),
            "push_timeout": float(env.get("ADAPTER_PUSH_TIMEOUT_S", "120"))}


def tar_dir(path):
    """One tar of an adapter directory, in memory: it is a few MB (`r=1` on four projections of 48 layers)."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for entry in sorted(Path(path).rglob("*")):
            if entry.is_file():
                archive.add(entry, arcname=entry.relative_to(path).as_posix())
    return buffer.getvalue()


async def push_one(session, url, name, body, timeout):
    async with session.put(f"{url}/adapters/{name}", data=body, timeout=timeout) as response:
        return response.status


def make_app(cfg):
    app = web.Application(client_max_size=1 << 30)
    app["cfg"] = cfg
    app["inflight"] = {url: 0 for url in cfg["upstreams"]}

    async def on_startup(app):
        app["session"] = ClientSession(connector=TCPConnector(limit=0), timeout=TIMEOUT)

    async def on_cleanup(app):
        await app["session"].close()

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    def replica_for(prompt):
        """The eight rollouts of one prompt have to reach one replica (its KV cache is the prefill it already did): hash
        the prompt, and only fall back to the least loaded when that replica is more than IMBALANCE requests ahead."""
        blob = json.dumps(prompt, sort_keys=True).encode()
        chosen = cfg["upstreams"][int(hashlib.sha256(blob).hexdigest(), 16) % len(cfg["upstreams"])]
        lightest = min(cfg["upstreams"], key=lambda url: app["inflight"][url])
        return lightest if app["inflight"][chosen] - app["inflight"][lightest] > IMBALANCE else chosen

    async def proxy(url, request, body=None, path=None):
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
        async with app["session"].request(request.method, url + (path or request.path_qs), data=body,
                                          headers=headers, timeout=TIMEOUT) as response:
            payload = await response.read()
            return web.Response(status=response.status, body=payload,
                                headers={k: v for k, v in response.headers.items() if k.lower() not in HOP_BY_HOP})

    async def health(request):
        async def one(url):
            try:
                async with app["session"].get(f"{url}/health", timeout=ClientTimeout(total=5)) as response:
                    return response.status == 200
            except Exception:  # noqa: BLE001 - an unreachable replica is not healthy
                return False
        results = await asyncio.gather(*(one(url) for url in cfg["upstreams"]))
        if all(results):
            return web.Response(text="ok")
        return web.Response(status=503, text=f"{list(results).count(False)}/{len(results)} replicas unhealthy")

    async def completions(request):
        body = await request.read()
        prompt = json.loads(body).get("prompt", "") if body else ""
        url = replica_for(prompt)
        app["inflight"][url] += 1
        try:
            return await proxy(url, request, body)
        finally:
            app["inflight"][url] -= 1

    async def load(request):
        body = await request.read()
        try:
            payload = json.loads(body)
            name, path = payload["lora_name"], payload["lora_path"]
        except (ValueError, KeyError):
            return web.Response(status=400, text="gateway: lora_name and lora_path are required")
        started = time.perf_counter()
        try:
            blob = await asyncio.to_thread(tar_dir, path)
        except OSError as error:
            return web.Response(status=400, text=f"gateway: cannot read {path}: {error}")
        pushed = await asyncio.gather(*(push_one(app["session"], url, name, blob, cfg["push_timeout"])
                                        for url in cfg["sidecars"]), return_exceptions=True)
        ok = [status == 200 for status in pushed if not isinstance(status, Exception)]
        if not all(ok) or len(ok) != len(cfg["sidecars"]):
            await asyncio.gather(*(delete_one(app["session"], url, name) for url in cfg["sidecars"]))
            return web.Response(status=502, text=f"gateway: the adapter reached {ok.count(True)}/{len(cfg['sidecars'])} "
                                                 f"sidecars; nothing loaded")
        seconds = time.perf_counter() - started
        # vLLM reads `lora_path` itself, so it has to be the replica's local path, not the trainer's directory.
        payload["lora_path"] = f"/tmp/loras/{name}"
        loads = await asyncio.gather(*(proxy(url, request, json.dumps(payload).encode()) for url in cfg["upstreams"]),
                                     return_exceptions=True)
        statuses = [item.status if isinstance(item, web.Response) else 502 for item in loads]
        if not all(status == 200 for status in statuses):
            # The rollback is an unload, not a second load: the replicas that did load it drop the adapter again.
            await asyncio.gather(*(proxy(url, request, json.dumps({"lora_name": name}).encode(),
                                         path="/v1/unload_lora_adapter") for url in cfg["upstreams"]),
                                 return_exceptions=True)
            await asyncio.gather(*(delete_one(app["session"], url, name) for url in cfg["sidecars"]))
            return web.Response(status=502, text=f"gateway: load failed on {statuses.count(200)}/{len(statuses)} replicas")
        print(f"[gateway] {name}: {len(blob)} B to {len(cfg['sidecars'])} sidecars in {seconds * 1e3:.0f} ms, loaded on "
              f"{len(statuses)} replicas", flush=True)
        return web.Response(text=json.dumps({"ok": True, "bytes": len(blob), "push_s": round(seconds, 3)}),
                            content_type="application/json")

    async def broadcast(request):
        body = await request.read()
        results = await asyncio.gather(*(proxy(url, request, body) for url in cfg["upstreams"]),
                                       return_exceptions=True)
        statuses = [item.status if isinstance(item, web.Response) else 502 for item in results]
        return web.Response(status=200 if all(s == 200 for s in statuses) else 502,
                            text=f"{statuses.count(200)}/{len(statuses)} replicas answered")

    async def any_replica(request):
        body = await request.read() if request.can_read_body else None
        url = min(cfg["upstreams"], key=lambda item: app["inflight"][item])
        return await proxy(url, request, body)

    app.router.add_get("/health", health)
    app.router.add_post("/v1/completions", completions)
    app.router.add_post(LOAD, load)
    for path in BROADCAST:
        app.router.add_post(path, broadcast)
    for path in ONE_REPLICA:
        app.router.add_get(path, lambda request, path=path: proxy(cfg["upstreams"][0], request))
    app.router.add_route("*", "/{tail:.*}", any_replica)
    return app


async def delete_one(session, url, name):
    try:
        async with session.delete(f"{url}/adapters/{name}", timeout=ClientTimeout(total=30)) as response:
            return response.status
    except Exception:  # noqa: BLE001 - the delete is best-effort by design
        return None


def main():
    cfg = config()
    print(f"[gateway] :{cfg['port']} -> {len(cfg['upstreams'])} replicas, sidecars on {cfg['sidecars'][0]}", flush=True)
    web.run_app(make_app(cfg), port=cfg["port"], print=None)


if __name__ == "__main__":
    main()
