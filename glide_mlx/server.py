"""HTTP server on Apple Silicon: POST /v1/systemone and GET /health, same request and response format as upstream glide/server.py.

    python -m glide_mlx.server --model ~/local-llms/glide/mlx-q8 --port 8790

One MLX model per process. MLX work (loading and every forward pass) runs on one dedicated worker thread, so requests are
scored one at a time, the questions of a request in turn.
Environment (defaults in brackets): GLIDE_MODEL, GLIDE_MODEL_NAME [glide-v1.2-mlx-<bits>], GLIDE_MAX_LEN [40000],
GLIDE_API_KEY (if set, requests must send "Authorization: Bearer <key>"), GLIDE_VISION [1],
GLIDE_REQUEST_TIMEOUT_S [600] (HTTP 504 after that; the forward pass itself still finishes),
GLIDE_MIN_PIXELS / GLIDE_MAX_PIXELS (read by images.py, as upstream).
"""
from __future__ import annotations

import argparse
import hmac
import concurrent.futures
import json
import logging
import os
import time

from .prompt import RequestError

log = logging.getLogger("glide_mlx.server")


def default_name(path):
    try:
        q = json.load(open(os.path.join(path, "config.json"))).get("quantization")
    except OSError:
        q = None
    return f"glide-v1.2-mlx-{q['bits']}bit" if q else "glide-v1.2-mlx-bf16"


def create_app(model=None, model_path=None):
    from fastapi import FastAPI, Header
    from fastapi.responses import JSONResponse

    path = model_path or os.environ.get("GLIDE_MODEL", "")
    api_key = os.environ.get("GLIDE_API_KEY", "")
    timeout_s = float(os.environ.get("GLIDE_REQUEST_TIMEOUT_S", "600"))
    ctx = {"model": model, "stats": {"requests": 0, "busy_s": 0.0, "started": time.time()}}
    worker = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="glide-mlx")   # MLX streams are per thread

    def load():
        if ctx["model"] is None:
            from .model import GlideMLX
            if not path:
                raise RuntimeError("set GLIDE_MODEL or --model")
            ctx["model"] = GlideMLX.from_pretrained(path, vision=os.environ.get("GLIDE_VISION", "1") != "0",
                                                    max_len=int(os.environ.get("GLIDE_MAX_LEN", "40000")),
                                                    name=os.environ.get("GLIDE_MODEL_NAME") or default_name(path))
            log.info("loaded %s in %.0f s", path, ctx["model"].load_s)

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(_app):
        worker.submit(load).result()
        yield
        worker.shutdown(wait=False)

    app = FastAPI(title="GLiDE (MLX)", lifespan=lifespan)

    def error(status, message):
        return JSONResponse(status_code=status, content={"error": message})

    @app.get("/health")
    def health():
        m = ctx["model"]
        if m is None:
            return error(503, "loading")
        return {"status": "ok", "model": m.name, "engine": "mlx", "vision": m.vision, "max_len": m.max_len,
                "stats": dict(ctx["stats"])}

    @app.post("/v1/systemone")
    def systemone(body: dict, authorization: str = Header(default="")):
        if api_key and not hmac.compare_digest(authorization.encode(), f"Bearer {api_key}".encode()):
            return error(401, "unauthorized")
        if "state" not in body:
            return error(422, "state is required")
        if not isinstance(body.get("questions"), dict):
            return error(422, "questions must be an object mapping keys to questions")
        t0 = time.perf_counter()
        req = {"state": body["state"], "questions": body["questions"]}

        def job():
            t1 = time.perf_counter()
            try:
                return ctx["model"].decide_batch([req])[0]
            except Exception as e:  # noqa: BLE001
                log.exception("request failed")
                return e
            finally:
                ctx["stats"]["requests"] += 1
                ctx["stats"]["busy_s"] += time.perf_counter() - t1

        try:
            r = worker.submit(job).result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            return error(504, "request timed out")
        if isinstance(r, RequestError):
            return error(422, str(r))
        if isinstance(r, Exception):
            return error(500, f"engine error: {type(r).__name__}")
        return {**r, "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}

    return app


def main():
    ap = argparse.ArgumentParser(description="GLiDE on MLX (POST /v1/systemone, GET /health)")
    ap.add_argument("--model", default=os.environ.get("GLIDE_MODEL", ""))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8790)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import uvicorn
    uvicorn.run(create_app(model_path=args.model), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
