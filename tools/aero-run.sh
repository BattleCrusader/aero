#!/bin/bash
# aero-run: run an aero python probe UNDER a hard memory cap.
#
# WHY THIS EXISTS: a probe that loads a GGUF directly allocates several GB, and
# running two of them (or one while a subagent runs another) OOMs a 16 GB box.
# The engine has a --max-ram guard, but a quick diagnostic script does not, and
# that is exactly the script that gets run carelessly. So this wrapper makes the
# cap PHYSICAL rather than advisory:
#
#   1. phys_footprint watchdog: polls the child's macOS phys_footprint (which,
#      unlike `ps -o rss`, includes GPU/Metal memory) every 250 ms and SIGKILLs
#      the whole process group when it exceeds the cap.
#   2. A global lock: only ONE aero model process may run at a time. Concurrent
#      model loads are the actual OOM mechanism; a lock removes the possibility
#      instead of relying on discipline.
#   3. nice -n 20 on the child so the machine stays responsive while it runs.
#
# usage:  tools/aero-run.sh <script.py> [args...]
#         AERO_CAP_MB=6144 tools/aero-run.sh probe.py
set -uo pipefail

CAP_MB="${AERO_CAP_MB:-6144}"          # 6 GiB default: half the box
LOCK="/tmp/aero-model.lock"
POLL=0.25

SCRIPT="${1:?usage: aero-run.sh <script.py> [args...]}"
shift || true
# `shift` removed the script from $@, so remember it rather than re-testing $1
# (which is now the first ARGUMENT, or empty -- that empty is what silently ran
# `nice -n 20` with no command at all).
FIRST="$SCRIPT"

# ---- global single-model lock -------------------------------------------------
# macOS has no flock(1), so this uses mkdir, which is atomic on POSIX. Only ONE
# aero model process may run at a time: concurrent model loads are the actual OOM
# mechanism, so the lock removes the possibility rather than relying on care.
LOCKDIR="/tmp/aero-model.lock.d"
waited=0
while ! mkdir "$LOCKDIR" 2>/dev/null; do
  # stale lock from a killed run?
  if [ -f "$LOCKDIR/pid" ] && ! kill -0 "$(cat "$LOCKDIR/pid" 2>/dev/null)" 2>/dev/null; then
    echo "aero-run: removing stale lock (pid $(cat "$LOCKDIR/pid" 2>/dev/null) is gone)" >&2
    rm -rf "$LOCKDIR"
    continue
  fi
  [ "$waited" -eq 0 ] && echo "aero-run: another aero model process holds the lock; waiting" >&2
  sleep 2; waited=$((waited + 2))
  if [ "$waited" -ge 1800 ]; then
    echo "aero-run: gave up waiting for $LOCKDIR" >&2; exit 2
  fi
done
echo $$ > "$LOCKDIR/pid"
cleanup() { rm -rf "$LOCKDIR"; }
trap 'cleanup; kill -TERM -"$CHILD" 2>/dev/null; kill -KILL -"$CHILD" 2>/dev/null; exit 130' INT TERM
trap cleanup EXIT

# ---- launch under the watchdog -----------------------------------------------
# nice runs a COMMAND; a bare .py path is not executable, so invoke the
# interpreter explicitly when the first argument is a Python file.
if [ "${FIRST##*.}" = "py" ]; then
  PY="${AERO_PY:-.venv-cml/bin/python}"
  nice -n 20 "$PY" "$FIRST" "$@" &
else
  nice -n 20 "$FIRST" "$@" &
fi
CHILD=$!

peak=0
status=0
while kill -0 "$CHILD" 2>/dev/null; do
  # Footprint of the child in MB. /usr/bin/footprint prints a "Footprint:" line
  # (matching engine/cli.py's own_footprint_bytes, which is what the tested guard
  # uses) -- grepping for "phys_footprint" in its output matches nothing and
  # silently reports 0 forever, which would make this watchdog useless.
  # line looks like:  bash [123]: 64-bit    Footprint: 1169 KB (16384 bytes per page)
  # so awk must emit fields 5 and 6 (value, unit) -- 2 and 3 are the pid and
  # the word "64-bit".
  fp=$(/usr/bin/footprint -p "$CHILD" 2>/dev/null \
       | awk '/Footprint:/ {print $5, $6; exit}')
  set -- $fp
  val="${1:-}"; unit="${2:-}"
  case "$val" in
    ''|*[!0-9.]*) : ;;
    *)  case "$unit" in
          KB) mb=$(awk -v v="$val" 'BEGIN{printf "%d", v/1024}') ;;
          MB) mb=$(awk -v v="$val" 'BEGIN{printf "%d", v}') ;;
          GB) mb=$(awk -v v="$val" 'BEGIN{printf "%d", v*1024}') ;;
          *)  mb=0 ;;
        esac
        [ "$mb" -gt "$peak" ] && peak=$mb
        if [ "$mb" -gt "$CAP_MB" ]; then
          echo "aero-run: KILLED — footprint ${mb} MB exceeded AERO_CAP_MB=${CAP_MB}" >&2
          kill -KILL -"$CHILD" 2>/dev/null
          status=137
          break
        fi ;;
  esac
  sleep "$POLL"
done
wait "$CHILD" 2>/dev/null
rc=$?
[ "$status" -eq 137 ] || status=$rc
echo "aero-run: exit=$status peak_phys_footprint=${peak} MB (cap ${CAP_MB} MB)" >&2
exit "$status"
