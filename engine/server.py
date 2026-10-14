"""OpenAI-compatible HTTP server for Aero.

Implements the parts an OpenAI client actually needs:

    GET  /v1/models                 list available models
    POST /v1/chat/completions       chat, with stream=true (SSE) and false
    POST /v1/completions            legacy text completion
    GET  /health                    liveness

Stdlib only -- no framework, no dependencies. Streaming uses Server-Sent Events
exactly as OpenAI does (`data: {...}\\n\\n`, terminated by `data: [DONE]`), so
existing clients work unchanged.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple

from .runtime import GenParams, ModelRuntime


class _Ctx:
    """Shared server context."""

    def __init__(self, runtime: ModelRuntime, model_id: str, devices: str,
                 verbose: bool = False) -> None:
        self.runtime = runtime
        self.model_id = model_id
        self.devices = devices
        self.verbose = verbose
        self.started = time.time()
        self.requests = 0
        self.lock = threading.Lock()

    def next_id(self) -> str:
        return "chatcmpl-" + uuid.uuid4().hex[:24]


def _now() -> int:
    return int(time.time())


def _prompt_tokens(runtime: ModelRuntime, messages: List[Dict[str, str]]) -> int:
    try:
        return len(runtime.tokenize(runtime.apply_chat_template(messages)))
    except Exception:  # noqa: BLE001
        return sum(len(str(m.get("content", "")).split()) for m in messages)


def _apply_stop(text: str, stop: Tuple[str, ...]) -> Tuple[str, bool]:
    """Truncate at the earliest stop sequence. Returns (text, hit)."""
    if not stop:
        return text, False
    cut = None
    for s in stop:
        if not s:
            continue
        i = text.find(s)
        if i >= 0 and (cut is None or i < cut):
            cut = i
    if cut is None:
        return text, False
    return text[:cut], True


def make_handler(ctx: _Ctx):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        # ---- plumbing ----------------------------------------------------
        def log_message(self, fmt, *args):  # quiet unless --verbose
            if ctx.verbose:
                super().log_message(fmt, *args)

        def _json(self, obj: Any, code: int = 200) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, message: str, code: int = 400, etype: str = "invalid_request_error") -> None:
            self._json({"error": {"message": message, "type": etype, "code": code}}, code)

        def _read_body(self) -> Optional[dict]:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0:
                self._error("empty request body")
                return None
            try:
                return json.loads(self.rfile.read(n))
            except json.JSONDecodeError as e:
                self._error(f"invalid JSON: {e}")
                return None

        # ---- routes ------------------------------------------------------
        def do_GET(self):  # noqa: N802
            path = self.path.split("?")[0].rstrip("/")
            if path in ("/health", "/v1/health"):
                self._json({"status": "ok", "uptime_s": round(time.time() - ctx.started, 1),
                            "devices": ctx.devices, "runtime": type(ctx.runtime).__name__})
            elif path == "/v1/models":
                self._json({"object": "list", "data": [{
                    "id": ctx.model_id, "object": "model", "created": _now(),
                    "owned_by": "aero",
                }]})
            else:
                self._error(f"unknown path {path}", 404, "not_found")

        def do_POST(self):  # noqa: N802
            path = self.path.split("?")[0].rstrip("/")
            body = self._read_body()
            if body is None:
                return
            if path == "/v1/chat/completions":
                self._chat(body)
            elif path == "/v1/completions":
                self._completion(body)
            else:
                self._error(f"unknown path {path}", 404, "not_found")

        # ---- handlers ----------------------------------------------------
        def _messages(self, body: dict) -> Optional[List[Dict[str, str]]]:
            msgs = body.get("messages")
            if not isinstance(msgs, list) or not msgs:
                self._error("'messages' must be a non-empty array")
                return None
            out = []
            for m in msgs:
                if not isinstance(m, dict):
                    self._error("each message must be an object")
                    return None
                out.append({"role": str(m.get("role", "user")),
                            "content": str(m.get("content", ""))})
            return out

        def _chat(self, body: dict) -> None:
            messages = self._messages(body)
            if messages is None:
                return
            params = GenParams.from_openai(body)
            stream = bool(body.get("stream", False))
            model = str(body.get("model") or ctx.model_id)
            cid = ctx.next_id()
            ptoks = _prompt_tokens(ctx.runtime, messages)

            with ctx.lock:
                ctx.requests += 1

            if stream:
                self._chat_stream(cid, model, messages, params)
            else:
                text = "".join(ctx.runtime.generate(messages, params))
                text, hit = _apply_stop(text, params.stop)
                ctoks = len(ctx.runtime.tokenize(text))
                self._json({
                    "id": cid, "object": "chat.completion", "created": _now(), "model": model,
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": text}}],
                    "usage": {"prompt_tokens": ptoks, "completion_tokens": ctoks,
                              "total_tokens": ptoks + ctoks},
                })

        def _chat_stream(self, cid: str, model: str,
                         messages: List[Dict[str, str]], params: GenParams) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            # Close after the body: the stream has no Content-Length, so this is
            # what guarantees every client sees a clean end after [DONE].
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

            def chunk(delta: dict, finish=None):
                return {"id": cid, "object": "chat.completion.chunk", "created": _now(),
                        "model": model,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

            def emit(obj):
                self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
                self.wfile.flush()

            emit(chunk({"role": "assistant", "content": ""}))
            acc = ""
            try:
                for piece in ctx.runtime.generate(messages, params):
                    acc += piece
                    acc, hit = _apply_stop(acc, params.stop)
                    emit(chunk({"content": piece}))
                    if hit:
                        break
            except Exception as e:  # noqa: BLE001
                emit({"error": {"message": str(e), "type": "server_error"}})
            emit(chunk({}, finish="stop"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def _completion(self, body: dict) -> None:
            prompt = body.get("prompt")
            if isinstance(prompt, list):
                prompt = prompt[0] if prompt else ""
            if not isinstance(prompt, str):
                self._error("'prompt' must be a string")
                return
            params = GenParams.from_openai(body)
            model = str(body.get("model") or ctx.model_id)
            text = "".join(ctx.runtime.generate([{"role": "user", "content": prompt}], params))
            text, _ = _apply_stop(text, params.stop)
            self._json({
                "id": "cmpl-" + uuid.uuid4().hex[:24], "object": "text_completion",
                "created": _now(), "model": model,
                "choices": [{"index": 0, "text": text, "finish_reason": "stop",
                             "logprobs": None}],
                "usage": {"prompt_tokens": len(ctx.runtime.tokenize(prompt)),
                          "completion_tokens": len(ctx.runtime.tokenize(text)),
                          "total_tokens": 0},
            })

    return Handler


def build_server(runtime: ModelRuntime, model_id: str = "aero",
                 host: str = "127.0.0.1", port: int = 8000,
                 devices: str = "", verbose: bool = False) -> ThreadingHTTPServer:
    ctx = _Ctx(runtime, model_id, devices, verbose)
    handler = make_handler(ctx)
    return ThreadingHTTPServer((host, port), handler)


def serve(runtime: ModelRuntime, model_id: str = "aero",
          host: str = "127.0.0.1", port: int = 8000,
          devices: str = "", verbose: bool = False) -> None:
    httpd = build_server(runtime, model_id, host, port, devices, verbose)
    print(f"aero server listening on http://{host}:{port}")
    print(f"  model   : {model_id}")
    print(f"  runtime : {type(runtime).__name__}")
    if devices:
        print(f"  devices : {devices}")
    print(f"  endpoints: GET /v1/models · POST /v1/chat/completions · GET /health")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        httpd.server_close()


def main(argv=None) -> int:
    import argparse
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from . import build_engine, format_limits
    from .runtime import make_runtime

    p = argparse.ArgumentParser(prog="aero-serve",
                                description="OpenAI-compatible server for Aero.")
    p.add_argument("--runtime", default="echo", choices=["echo", "gguf"])
    p.add_argument("--model", default="aero", help="model id or GGUF path")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--verbose", action="store_true", help="log every request")
    p.add_argument("--limits", action="store_true", help="print device limits and exit")
    for name in ("cpu", "gpu", "ane"):
        p.add_argument(f"--use-{name}", type=int, default=None, metavar="N")
    a = p.parse_args(argv)

    if a.limits:
        print(format_limits())
        return 0

    try:
        engine = build_engine(use_cpu=a.use_cpu, use_gpu=a.use_gpu, use_ane=a.use_ane)
    except Exception as e:  # noqa: BLE001
        print(f"error: {e}", file=sys.stderr)
        return 2

    runtime = make_runtime(a.runtime)
    try:
        runtime.load(a.model)
    except NotImplementedError as e:
        print(f"error: {e}", file=sys.stderr)
        return 3

    serve(runtime, model_id=a.model, host=a.host, port=a.port,
          devices=str(engine.plan), verbose=a.verbose)
    return 0


if __name__ == "__main__":
    import sys as _sys

    _sys.exit(main())
