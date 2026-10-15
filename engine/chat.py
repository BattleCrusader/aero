#!/usr/bin/env python3
"""aero chat -- command-line chat interface.

    ./aero-chat                                  # default runtime
    ./aero-chat --runtime gguf --model <path>    # real weights, real forward pass
    ./aero-chat --use-cpu=2 --use-gpu=1          # pick the device allocation
    ./aero-chat --server http://127.0.0.1:8000   # talk to a running aero server

Commands inside the REPL: /help /reset /stats /devices /exit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import Device, build_engine, format_limits  # noqa: E402
from engine.runtime import GenParams, accepts_kwarg, make_runtime  # noqa: E402

BANNER = r"""
   _   ___ ___  ___
  /_\ | __| _ \/ _ \     device-split inference
 / _ \| _||   / (_) |
/_/ \_\___|_|_\\___/
"""


def _stream_local(runtime, messages, params, show_tokens: bool):
    """Print a local runtime response as it streams."""
    t0 = time.perf_counter()
    pieces = []
    for piece in runtime.generate(messages, params):
        pieces.append(piece)
        sys.stdout.write(piece)
        sys.stdout.flush()
    dt = time.perf_counter() - t0
    text = "".join(pieces)
    if show_tokens:
        n = len(runtime.tokenize(text))
        rate = (n / dt) if dt > 0 else 0.0
        print(f"\n  [{n} tok, {dt:.2f}s, {rate:.1f} tok/s]")
    return text


def _stream_remote(base: str, model: str, messages, params, show_tokens: bool):
    """Stream from an OpenAI-compatible /v1/chat/completions endpoint."""
    body = json.dumps({
        "model": model, "messages": messages, "stream": True,
        "max_tokens": params.max_tokens, "temperature": params.temperature,
        "top_p": params.top_p, "stop": list(params.stop) or None,
    }).encode()
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", data=body,
                                headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    acc = []
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if "error" in obj:
                    print(f"\n  server error: {obj['error']}", file=sys.stderr)
                    break
                for ch in obj.get("choices", []):
                    piece = ch.get("delta", {}).get("content") or ""
                    if piece:
                        acc.append(piece)
                        sys.stdout.write(piece)
                        sys.stdout.flush()
    except urllib.error.HTTPError as e:
        print(f"\n  HTTP {e.code}: {e.read()[:300]!r}", file=sys.stderr)
    except urllib.error.URLError as e:
        print(f"\n  cannot reach {base}: {e}", file=sys.stderr)
    dt = time.perf_counter() - t0
    text = "".join(acc)
    if show_tokens:
        n = len(text.split())
        print(f"\n  [{n} word(s), {dt:.2f}s]")
    return text


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aero-chat", description="Command-line chat for Aero.")
    p.add_argument("--runtime", default="echo", choices=["echo", "gguf"],
                   help="model runtime (default echo; 'gguf' runs real weights)")
    p.add_argument("--max-ram", default=None,
                   help="hard cap on process footprint, e.g. 8, 8G, 8GB "
                        "(the load is REFUSED up front if the plan exceeds it)")
    p.add_argument("--context", type=int, default=4096,
                   help="KV cache capacity in positions (default 4096)")
    p.add_argument("--block-rows", type=int, default=64,
                   help="rows dequantised per matmul block (default 64)")
    p.add_argument("--kv-type-k", default=None, choices=["fp32", "int8", "int4"],
                   help="K cache precision (overrides --kv-type for K only). "
                        "MEASURED: K tolerates far more compression than V -- "
                        "at equal bytes K4/V8 is 9.6x more accurate than K8/V4.")
    p.add_argument("--kv-type-v", default=None, choices=["fp32", "int8", "int4"],
                   help="V cache precision (overrides --kv-type for V only). "
                        "V is the sensitive side; give it the bits.")
    p.add_argument("--kv-type", default="fp32", choices=["fp32", "int8", "int4"],
                   help="KV cache precision. MEASURED at 262144 ctx vs the "
                        "18.00G fp32 baseline: int8 = 4.57G with 4/4 tokens "
                        "identical to fp32 (0.93%% logit error); int4 = 2.32G "
                        "but 3/4 tokens and 32%% logit error. Default fp32.")
    p.add_argument("--model", default="aero", help="model id or GGUF path")
    p.add_argument("--server", default=None, metavar="URL",
                   help="talk to a running aero server instead of a local runtime")
    p.add_argument("--system", default=None, help="system prompt")
    p.add_argument("--max-tokens", type=int, default=None,
                   help="tokens per reply (default 256, or 8 for a real runtime "
                        "-- numpy decode is ~30 s/token on this box)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed for sampling. Only observable at "
                        "--temperature > 0: greedy decoding ignores it.")
    p.add_argument("--no-stream", action="store_true", help="disable streaming")
    p.add_argument("--show-tokens", action="store_true", help="print token counts and rate")
    p.add_argument("--limits", action="store_true", help="print device limits and exit")
    for name in ("cpu", "gpu", "ane"):
        p.add_argument(f"--use-{name}", type=int, default=None, metavar="N",
                       help=f"{name} units 0..limit (see --limits)")
    return p


HELP = """commands:
  /help      this text
  /reset     clear the conversation history
  /stats     runtime and session statistics
  /devices   the device allocation in use
  /exit      leave (also /quit, Ctrl-D)
"""


def main(argv=None) -> int:
    # Behave like a normal Unix tool when stdout is a pipe that closes early
    # (e.g. `aero-chat | head`): die on SIGPIPE instead of raising
    # BrokenPipeError and printing "Exception ignored".
    try:
        import signal
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    except (ImportError, AttributeError, ValueError):
        pass

    args = build_parser().parse_args(argv)

    if args.limits:
        print(format_limits())
        return 0

    # device allocation (resolved and reported; the runtime uses it once the
    # real compute path exists)
    try:
        engine = build_engine(use_cpu=args.use_cpu, use_gpu=args.use_gpu,
                              use_ane=args.use_ane)
    except Exception as e:  # noqa: BLE001
        print(f"error: {e}", file=sys.stderr)
        return 2
    devices = str(engine.plan)
    if engine.plan.notice():
        print(f"note: {engine.plan.notice()}")

    runtime = None
    if not args.server:
        # Options the real runtime needs; make_runtime() keeps its no-arg
        # signature for the echo stub, so they are applied only when accepted.
        opts = {}
        # THE PLAN GOES WITH THE RUNTIME, or the flags die here. --use-cpu reaches
        # compute as runtime.threads -> aero_geom.threads -> aero::matmul, and an
        # unspecified count means "use the machine's cores", not "use one".
        if accepts_kwarg(args.runtime, "device_plan"):
            opts["device_plan"] = engine.plan
        for key in ("max_ram", "context", "block_rows"):
            if hasattr(args, key) and getattr(args, key) is not None:
                opts[key] = getattr(args, key)
        if getattr(args, "kv_type", None):
            opts["kv_dtype"] = args.kv_type
        if getattr(args, "kv_type_k", None):
            opts["k_dtype"] = args.kv_type_k
        if getattr(args, "kv_type_v", None):
            opts["v_dtype"] = args.kv_type_v
        try:
            runtime = make_runtime(args.runtime, **opts)
        except TypeError:
            runtime = make_runtime(args.runtime)
        try:
            runtime.load(args.model)
        except NotImplementedError as e:
            # no silent fake: an unimplemented path says so and stops
            print(f"error: {e}", file=sys.stderr)
            return 3
        except MemoryError as e:
            # the predictive gate refused the load; that is the guard working
            print(f"error: {e}", file=sys.stderr)
            return 4
        except Exception as e:
            print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
            return 5

    # A real runtime decodes at seconds-per-token in numpy, so the OpenAI-default
    # 256 tokens would mean HOURS per reply with no feedback. Default it down and
    # say so, rather than appearing to hang.
    if args.max_tokens is None:
        args.max_tokens = 8 if (not args.server and args.runtime == "gguf") else 256

    print(BANNER)
    print(f"  model    : {args.model}")
    print(f"  runtime  : {'remote ' + args.server if args.server else type(runtime).__name__}")
    print(f"  devices  : {devices}  ({engine.plan.total_streams} stream(s))")
    print(f"  commands : /help /reset /stats /devices /exit")
    if not args.server and args.runtime == "echo":
        print("\n  NOTE: the 'echo' runtime is a DEV stub, not a language model.")
    if not args.server and args.runtime == "gguf":
        print(f"  tokens   : {args.max_tokens} per reply "
              f"(numpy decode is slow; raise with --max-tokens)")
    print()

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    turns = 0
    total_chars = 0
    try:
        while True:
            try:
                line = input("you> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            line = line.strip()
            if not line:
                continue

            if line.startswith("/"):
                cmd = line.split()[0].lower()
                if cmd in ("/exit", "/quit"):
                    break
                if cmd == "/help":
                    print(HELP)
                    continue
                if cmd == "/reset":
                    messages = []
                    if args.system:
                        messages.append({"role": "system", "content": args.system})
                    turns = 0
                    total_chars = 0
                    print("  history cleared")
                    continue
                if cmd == "/stats":
                    print(f"  runtime : {type(runtime).__name__ if runtime else 'remote'}")
                    print(f"  model   : {args.model}")
                    print(f"  devices : {devices}")
                    print(f"  turns   : {turns}")
                    print(f"  chars   : {total_chars}")
                    if runtime is not None:
                        # stats() is a method; printing the bound method would
                        # show a memory address instead of the actual numbers.
                        st = runtime.stats() if callable(runtime.stats) else runtime.stats
                        for k, v in st.items():
                            if isinstance(v, int) and v > 1024:
                                print(f"  {k:14s}: {v/2**20:.1f} MiB")
                            else:
                                print(f"  {k:14s}: {v}")
                    continue
                if cmd == "/devices":
                    print(engine.describe())
                    continue
                print(f"  unknown command {cmd}; try /help")
                continue

            messages.append({"role": "user", "content": line})
            params = GenParams(max_tokens=args.max_tokens,
                               temperature=args.temperature, seed=args.seed)

            sys.stdout.write("aero> ")
            sys.stdout.flush()
            if args.server:
                reply = _stream_remote(args.server, args.model, messages, params,
                                       args.show_tokens)
            else:
                reply = _stream_local(runtime, messages, params, args.show_tokens)
            if not reply.endswith("\n"):
                print()

            messages.append({"role": "assistant", "content": reply})
            turns += 1
            total_chars += len(reply)
            print()
    finally:
        if runtime is not None:
            runtime.close()

    print(f"session: {turns} turn(s), {total_chars} char(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
