"""Run a Python script under a hard 3 GB address-space limit.

macOS does not allow bash's ulimit -v to be changed, so the limit is applied
from inside the interpreter before the heavy imports happen.  A watchdog thread
also aborts the process if resident memory approaches the cap.
"""

from __future__ import annotations

import os
import resource
import runpy
import sys
import threading

LIMIT_BYTES = int(os.environ.get("SLEEVE_MEM_LIMIT_GB", "3")) * 1024**3
WARN_BYTES = int(LIMIT_BYTES * 0.97)
POLL_SECONDS = 0.1


def _apply_limit() -> None:
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    for pair in ((LIMIT_BYTES, hard), (LIMIT_BYTES, LIMIT_BYTES)):
        try:
            resource.setrlimit(resource.RLIMIT_AS, pair)
            print(f"[memlimit] RLIMIT_AS set to {pair[0] / 1024**3:.1f} GiB", file=sys.stderr)
            return
        except (ValueError, OSError) as exc:  # pragma: no cover - platform dependent
            last = exc
    try:
        resource.setrlimit(resource.RLIMIT_DATA, (LIMIT_BYTES, LIMIT_BYTES))
        print(f"[memlimit] RLIMIT_DATA set to {LIMIT_BYTES / 1024**3:.1f} GiB", file=sys.stderr)
        return
    except (ValueError, OSError) as exc:
        print(
            f"[memlimit] RLIMIT_AS/RLIMIT_DATA not enforceable (AS: {last}; DATA: {exc}); "
            "relying on the RSS watchdog",
            file=sys.stderr,
        )


def _current_rss() -> int:
    try:
        import psutil  # type: ignore

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _watch() -> None:
    while True:
        rss = _current_rss()
        if rss >= WARN_BYTES:
            print(
                f"[memlimit] aborting: RSS {rss / 1024**3:.2f} GiB exceeds the limit",
                file=sys.stderr,
                flush=True,
            )
            os._exit(9)
        threading.Event().wait(POLL_SECONDS)


def main() -> None:
    _apply_limit()
    if len(sys.argv) < 2:
        raise SystemExit("usage: pyrun.py <script.py> [args...]")
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    threading.Thread(target=_watch, daemon=True).start()
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
