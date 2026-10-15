#!/usr/bin/env python3
"""aero -- device-count CLI.

    aero --limits                          show detected device limits
    aero --plan                            show the resolved allocation
    aero --use-gpu=0 --use-ane=16 --bench  run a benchmark on that allocation

Flags (counts, not booleans -- 0 disables a device):
    --use-cpu=0..CPU_CORE_LIMIT   default 0
    --use-gpu=0..GPU_CORE_LIMIT   default 1
    --use-ane=0..ANE_CORE_LIMIT   default 0
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from engine import (DEFAULT_COUNTS, Device, DeviceCountOutOfRange,  # noqa: E402
                    NoDevicesSelected, build_engine, format_limits)
from engine.limits import all_limits  # noqa: E402


def own_footprint_bytes() -> int:
    """phys_footprint via /usr/bin/footprint -- includes GPU memory, unlike rss."""
    try:
        out = subprocess.run(["/usr/bin/footprint", "-p", str(os.getpid())],
                             capture_output=True, text=True, timeout=20, check=False)
        for line in out.stdout.splitlines():
            at = line.find("Footprint:")
            if at >= 0:
                parts = line[at:].split()
                if len(parts) >= 3:
                    value = float(parts[1])
                    unit = parts[2]
                    mult = {"KB": 1024, "MB": 1024 ** 2, "GB": 1024 ** 3}.get(unit, 1)
                    return int(value * mult)
    except Exception:  # noqa: BLE001
        pass
    return 0


def parse_max_ram(text: str) -> int:
    """Mirror of the C++ guard's parser: bare number = GiB, K/M/G binary, KB/MB/GB decimal."""
    t = text.strip().replace(" ", "")
    i = 0
    dot = False
    while i < len(t):
        c = t[i]
        if c.isdigit():
            i += 1
            continue
        if c == "." and not dot:
            dot = True
            i += 1
            continue
        break
    num, suf = t[:i], t[i:].lower()
    value = float(num)
    if suf == "":
        mult = 1.0 if (not dot and num.isdigit() and value >= 1048576.0) else 1024 ** 3
    elif suf in ("b", "bytes"):
        mult = 1.0
    elif suf in ("k", "kib"):
        mult = 1024.0
    elif suf in ("m", "mib"):
        mult = 1024.0 ** 2
    elif suf in ("g", "gib"):
        mult = 1024.0 ** 3
    elif suf == "kb":
        mult = 1000.0
    elif suf == "mb":
        mult = 1e6
    elif suf == "gb":
        mult = 1e9
    else:
        raise ValueError(f"unrecognised size suffix in {text!r}")
    return int(value * mult)


def human(n: int) -> str:
    v = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if v < 1024 or unit == "TiB":
            return f"{v:.2f} {unit}"
        v /= 1024
    return f"{v:.2f} TiB"


def build_parser() -> argparse.ArgumentParser:
    lim = all_limits()
    p = argparse.ArgumentParser(
        prog="aero",
        description="Run a weight matvec across CPU / GPU / ANE in any combination.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"detected limits:\n{format_limits(lim)}\n\n"
               f"defaults: cpu={DEFAULT_COUNTS['cpu']} gpu={DEFAULT_COUNTS['gpu']} "
               f"ane={DEFAULT_COUNTS['ane']}",
    )
    for name in ("cpu", "gpu", "ane"):
        L = lim[name]
        p.add_argument(
            f"--use-{name}", type=int, default=None, metavar="N",
            help=f"{name} units to use, 0..{L.limit} "
                 f"(default {DEFAULT_COUNTS[name]}); {'detected' if L.detected else 'documented'}",
        )
    p.add_argument("--limits", action="store_true", help="print detected limits and exit")
    p.add_argument("--plan", action="store_true", help="print the resolved allocation and exit")
    p.add_argument("--bench", type=int, default=0, metavar="N",
                   help="run N matvec iterations and report time")
    p.add_argument("--rows", type=int, default=4096, help="weight matrix rows (default 4096)")
    p.add_argument("--cols", type=int, default=1024, help="weight matrix cols (default 1024)")
    p.add_argument("--no-coreml", action="store_true",
                   help="use NumPy for every device instead of CoreML")
    p.add_argument("--max-ram", default=None, metavar="SIZE",
                   help="refuse if the process footprint would exceed SIZE (e.g. 8, 8G, 8GB)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.limits:
        print(format_limits())
        return 0

    try:
        eng = build_engine(use_cpu=args.use_cpu, use_gpu=args.use_gpu, use_ane=args.use_ane,
                           use_coreml=not args.no_coreml)
    except DeviceCountOutOfRange as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except NoDevicesSelected as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    plan = eng.plan
    print(f"allocation: {plan}  ({plan.total_streams} stream(s))")
    # A flag that is accepted and ignored is worse than one that errors: it makes the
    # user believe they changed something. A device in the plan with no kernel behind
    # it is therefore said out loud, immediately, not left to be inferred.
    _note = plan.notice()
    if _note:
        print(f"note: {_note}")

    cap = None
    if args.max_ram:
        try:
            cap = parse_max_ram(args.max_ram)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        print(f"--max-ram   {args.max_ram} -> {cap} bytes ({human(cap)})")

    if args.plan:
        print(eng.describe())
        return 0

    if not args.bench:
        print("(nothing to do; pass --bench N or --plan)")
        return 0

    weights = (np.random.default_rng(0).standard_normal((args.rows, args.cols)) * 0.05
               ).astype(np.float32)
    print(f"building weight matrix {args.rows}x{args.cols} "
          f"({args.rows * args.cols * 4 / 1e6:.0f} MB fp32)")
    eng.build(weights)
    print(eng.describe())

    x = np.random.default_rng(1).standard_normal((args.cols,)).astype(np.float32)

    eng.matvec(x, repeat=1)                      # warm up
    y, stats = eng.matvec(x, repeat=args.bench)

    print(f"\nbench: {args.bench} iteration(s) in {stats.seconds:.3f}s"
          f"  -> {stats.seconds / args.bench * 1000:.2f} ms/matvec")
    print(f"calls per device: {stats.calls_by_device}")
    print(f"devices used: {eng.devices_used()}")

    fp = own_footprint_bytes()
    if fp:
        print(f"footprint: {human(fp)}")
        if cap and fp > cap:
            print(f"REFUSED: footprint {human(fp)} exceeds --max-ram {human(cap)}",
                  file=sys.stderr)
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
