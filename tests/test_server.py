#!/usr/bin/env python3
"""Aero serving-layer self-test: OpenAI API + CLI chat.

Exercises the HTTP surface with a real socket (no mocking) and drives the CLI
chat as a subprocess, so both are tested the way they are actually used.

Run:  .venv-cml/bin/python tests/test_server.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from engine.runtime import EchoRuntime, GenParams, make_runtime  # noqa: E402
from engine.server import build_server  # noqa: E402

CHECKS = 0
FAILS = 0


def check(cond: bool, what: str) -> None:
    global CHECKS, FAILS
    CHECKS += 1
    if cond:
        print(f"  ok   {what}")
    else:
        FAILS += 1
        print(f"  FAIL {what}")


def get(url: str):
    with urllib.request.urlopen(url, timeout=20) as r:
        return r.status, json.loads(r.read())


def post(url: str, body: dict):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def post_raw(url: str, payload: bytes):
    req = urllib.request.Request(url, data=payload,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def post_stream(url: str, body: dict):
    """Return (chunks, saw_done, reconstructed_text)."""
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    chunks = []
    saw_done = False
    with urllib.request.urlopen(req, timeout=60) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                saw_done = True
                break
            chunks.append(json.loads(payload))
    text = "".join(c["choices"][0].get("delta", {}).get("content") or "" for c in chunks)
    return chunks, saw_done, text


def main() -> int:
    print("=== aero serving-layer self-test ===\n")

    # ------------------------------------------------------------- runtime
    print("[1] runtime layer")
    rt = make_runtime("echo")
    check(isinstance(rt, EchoRuntime), "make_runtime('echo') returns EchoRuntime")
    try:
        make_runtime("nope")
        check(False, "unknown runtime is rejected")
    except ValueError:
        check(True, "unknown runtime is rejected")
    rt.load("test-model")
    p = GenParams.from_openai({"max_tokens": 7, "temperature": 0.1, "stop": "END"})
    check(p.max_tokens == 7 and abs(p.temperature - 0.1) < 1e-9, "GenParams maps max_tokens/temperature")
    check(p.stop == ("END",), "a string 'stop' becomes a 1-tuple")
    p2 = GenParams.from_openai({"stop": ["a", "b"]})
    check(p2.stop == ("a", "b"), "a list 'stop' is preserved")
    check(GenParams.from_openai({}).max_tokens == 256, "max_tokens defaults to 256")

    # -------------------------------------------------------------- server
    rt2 = make_runtime("echo")
    rt2.load("aero-test")
    httpd = build_server(rt2, model_id="aero-test", devices="gpu=1", port=0)
    port = httpd.server_address[1]
    base = f"http://127.0.0.1:{port}"
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"\n[2] http server on {base}")

    # /v1/models
    st, body = get(base + "/v1/models")
    check(st == 200, "GET /v1/models -> 200")
    check(body.get("object") == "list", "models payload has object=list")
    check(body["data"][0]["id"] == "aero-test", "model id is reported")
    check(body["data"][0]["object"] == "model", "entry has object=model")

    # /health
    st, body = get(base + "/health")
    check(st == 200 and body.get("status") == "ok", "GET /health -> ok")

    # non-streaming chat
    st, body = post(base + "/v1/chat/completions", {
        "model": "aero-test",
        "messages": [{"role": "user", "content": "hello there world"}],
    })
    check(st == 200, "POST /v1/chat/completions -> 200")
    check(body.get("object") == "chat.completion", "object=chat.completion")
    check(body["id"].startswith("chatcmpl-"), "id has the chatcmpl- prefix")
    ch = body["choices"][0]
    check(ch["message"]["role"] == "assistant", "choice message role=assistant")
    check(isinstance(ch["message"]["content"], str) and ch["message"]["content"], "content is non-empty")
    check(ch["finish_reason"] == "stop", "finish_reason=stop")
    check("3 word" in ch["message"]["content"], "runtime saw the 3-word prompt")
    u = body["usage"]
    check(u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"],
          "usage total == prompt + completion")
    check(u["prompt_tokens"] > 0, "prompt_tokens > 0")

    # streaming chat
    chunks, done, text = post_stream(base + "/v1/chat/completions", {
        "model": "aero-test",
        "messages": [{"role": "user", "content": "hello there world"}],
        "stream": True,
    })
    check(done, "stream terminated with data: [DONE]")
    check(len(chunks) >= 3, f"stream emitted multiple chunks ({len(chunks)})")
    check(chunks[0]["choices"][0]["delta"].get("role") == "assistant",
          "first chunk carries the assistant role")
    check(all(c.get("object") == "chat.completion.chunk" for c in chunks),
          "every chunk has object=chat.completion.chunk")
    check(chunks[-1]["choices"][0]["finish_reason"] == "stop",
          "last chunk has finish_reason=stop")
    check(text == ch["message"]["content"],
          "streamed text == non-streamed text (same generation, same result)")

    # stop sequences
    st, body = post(base + "/v1/chat/completions", {
        "model": "aero-test",
        "messages": [{"role": "user", "content": "one two"}],
        "stop": ["tok2"],
    })
    check("tok2" not in body["choices"][0]["message"]["content"],
          "stop sequence is honoured (output truncated)")

    # legacy completions
    st, body = post(base + "/v1/completions", {"model": "aero-test", "prompt": "a b c"})
    check(st == 200 and body.get("object") == "text_completion", "POST /v1/completions works")
    check(body["choices"][0]["text"], "completion text non-empty")

    # errors
    st, _ = post(base + "/v1/chat/completions", {"model": "x"})
    check(st == 400, "missing 'messages' -> 400")
    st, raw = post_raw(base + "/v1/chat/completions", b"{not json")
    check(st == 400, "malformed JSON -> 400")
    err = json.loads(raw)
    check("error" in err and "message" in err["error"], "error payload is OpenAI-shaped")
    try:
        urllib.request.urlopen(base + "/nope", timeout=10)
        check(False, "unknown path -> 404")
    except urllib.error.HTTPError as e:
        check(e.code == 404, "unknown path -> 404")

    # concurrency: the server must not serialise independent requests
    results = []

    def hit():
        s, b = post(base + "/v1/chat/completions", {
            "model": "aero-test", "messages": [{"role": "user", "content": "x"}]})
        results.append(s)

    threads = [threading.Thread(target=hit) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    check(len(results) == 6 and all(s == 200 for s in results),
          "6 concurrent requests all succeeded")

    httpd.shutdown()
    httpd.server_close()

    # ----------------------------------------------------------------- cli
    print("\n[3] CLI chat (subprocess, piped stdin)")
    script = "/devices\nhello world\n/stats\n/exit\n"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "engine", "chat.py"),
                        "--runtime", "echo", "--model", "cli-test"],
                       input=script, capture_output=True, text=True, timeout=120)
    out = r.stdout
    check(r.returncode == 0, f"CLI exits 0 (got {r.returncode})")
    check("device-split inference" in out, "banner printed")
    check("gpu=1" in out, "default device allocation shown (gpu=1)")
    check("stream(s)" in out, "stream count shown")
    check("you> " in out, "prompt rendered")
    check("aero> " in out, "response rendered")
    check("echo:cli-test" in out, "echo runtime answered")
    check("turns   : 1" in out, "one turn recorded")
    check("session: 1 turn" in out, "session summary printed on exit")
    check("/devices" in out and "plan:" in out, "/devices printed the allocation")
    check("no weights loaded yet" in out,
          "/devices says why there are no row spans yet (honest, not silent)")
    if r.returncode != 0:
        print("    stderr:", r.stderr[:400])

    # CLI device flags reach the engine
    r2 = subprocess.run([sys.executable, os.path.join(ROOT, "engine", "chat.py"),
                         "--runtime", "echo", "--use-cpu=2", "--use-gpu=0"],
                        input="/exit\n", capture_output=True, text=True, timeout=120)
    check("cpu=2" in r2.stdout, "CLI honours --use-cpu=2 --use-gpu=0")

    # The gguf runtime IS implemented now, so a missing file must fail as a
    # missing file (loudly, non-zero) rather than as "not implemented". The
    # contract being tested is unchanged: no silent fake, ever.
    r3 = subprocess.run([sys.executable, os.path.join(ROOT, "engine", "chat.py"),
                         "--runtime", "gguf", "--model", "definitely-not-here.gguf"],
                        input="", capture_output=True, text=True, timeout=180)
    out3 = (r3.stdout + r3.stderr)
    check(r3.returncode != 0,
          f"gguf runtime with a missing model exits non-zero (got {r3.returncode})")
    check("not implemented" not in out3.lower(),
          "gguf runtime is no longer reported as unimplemented (P4 landed)")
    check("error" in out3.lower(),
          "the failure is reported as an error -- no silent fake")

    # ...and a REAL model must actually run. This is the check that would have
    # caught the previous stub: it asserts tokens come out, not that a message
    # says "not implemented".
    # A real model run costs ~9 minutes (numpy decode), so it is OPT-IN: the
    # default gate must stay fast or nobody runs it. AERO_REAL_SERVER=1 enables.
    real = ("/Volumes/Backup/lmstudio/models/empero-ai/"
            "Qwen3.8-4B-Distill-GGUF/Qwen3.8-4B-Q4_K_M.gguf")
    if os.environ.get("AERO_REAL_SERVER") and os.path.exists(real):
        r4 = subprocess.run(
            [sys.executable, os.path.join(ROOT, "engine", "chat.py"),
             "--runtime", "gguf", "--model", real, "--context", "32",
             "--max-ram", "6GB"],
            input="Hello\n/exit\n", capture_output=True, text=True, timeout=900)
        out4 = r4.stdout + r4.stderr
        check(r4.returncode == 0, f"real gguf run exits 0 (got {r4.returncode})")
        check("GgufRuntime" in out4,
              "the banner names the real runtime class (GgufRuntime)")
        check("echo" not in out4.lower().split("runtime")[-1][:40],
              "...and it is NOT the echo stub")
        check("not implemented" not in out4.lower(),
              "nothing claims the runtime is unimplemented")
        # a real run must show the token budget it is working to
        check("tokens" in out4.lower(), "the token budget is shown up front")
    elif not os.environ.get("AERO_REAL_SERVER"):
        print("       (skipped real gguf run: set AERO_REAL_SERVER=1; it takes "
              "~9 min)")
    else:
        print("       (skipped real gguf run: model not present)")

    print(f"\n=== {CHECKS} checks, {FAILS} failures ===")
    return 0 if FAILS == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
