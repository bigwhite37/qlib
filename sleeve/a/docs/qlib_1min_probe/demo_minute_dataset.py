"""Minimal, runnable demo: point Qlib at a 1-minute dataset and use its expressions.

Run it from the repository root, through the memory wrapper:

    cd /Users/shuzhenyi/code/python/qlib
    scripts/run_python_3gb.sh sleeve/a/docs/qlib_1min_probe/demo_minute_dataset.py

There is no Qlib patch anywhere in this file -- this is stock, unmodified Qlib.
The only requirement is that the dataset on disk uses Qlib's native layout:

    <root>/calendars/1min.txt                 # one bar timestamp per line
    <root>/instruments/all.txt                # SYMBOL<TAB>start<TAB>end
    <root>/features/sh600519/close.1min.bin   # float32: [start_index, v0, v1, ...]

The demo dataset was built from the warehouse parquet by build_probe.py.

Two Qlib gotchas this file exists to demonstrate
------------------------------------------------
1. `python path/to/script.py` puts the *script's* directory on sys.path[0], not
   the cwd -- so a bare `import qlib` silently picks up the pip-installed Qlib
   instead of this checkout. We force the checkout onto sys.path explicitly.
2. `D.features` fans out over `C.kernels` worker processes via joblib's
   multiprocessing backend. With the "spawn" start method every worker
   re-imports this module, so any top-level code without an
   `if __name__ == "__main__":` guard re-executes in every child and the script
   forks itself into a storm. Everything below lives inside `main()`.
"""

from __future__ import annotations

import sys
from pathlib import Path

# --- 1. force this checkout, not the pip-installed qlib ---------------------
QLIB_HOME = Path("/Users/shuzhenyi/code/python/qlib")
if str(QLIB_HOME) not in sys.path:
    sys.path.insert(0, str(QLIB_HOME))

ROOT = "/tmp/qlib1m_probe/cn_data_1min"
START, END = "2026-09-01 09:30:00", "2026-09-03 14:59:00"


def main() -> int:
    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D

    # The high-frequency operators live in qlib.contrib.ops.high_freq and are NOT
    # in the stock OpsList, so they must be handed to qlib.init explicitly.
    from qlib.contrib.ops.high_freq import (
        BFillNan,
        Cut,
        Date,
        DayCumsum,
        DayLast,
        FFillNan,
        IsInf,
        IsNull,
        Select,
    )

    assert Path(qlib.__file__).is_relative_to(QLIB_HOME), (
        f"wrong qlib: {qlib.__file__} (expected a path under {QLIB_HOME})"
    )
    print(f"qlib: {qlib.__file__}")

    qlib.init(
        provider_uri={"1min": ROOT},  # <-- the whole trick: freq -> path
        region=REG_CN,
        custom_ops=[DayCumsum, DayLast, FFillNan, BFillNan, Date, Select, IsNull, IsInf, Cut],
        expression_cache=None,
        dataset_cache=None,
    )

    # --- 2. the minute calendar --------------------------------------------
    cal = D.calendar(start_time=START, end_time=END, freq="1min")
    print(f"\ncalendar : {len(cal)} bars, {cal[0]} .. {cal[-1]}")

    # --- 3. the universe ---------------------------------------------------
    insts = D.list_instruments(
        D.instruments(market="all"), start_time=START, end_time=END, freq="1min", as_list=True
    )
    print(f"universe : {insts}")

    # --- 4. expressions, evaluated at freq=1min ----------------------------
    fields = [
        "$close",
        "$volume",
        "Ref($close, 1)",
        "Mean($close, 20)",
        "Std($close, 20)",
        "DayLast($close)",
        "DayCumsum($volume, '9:30', '14:59')",
    ]
    df = D.features(insts, fields, start_time=START, end_time=END, freq="1min")
    print(f"\nD.features shape = {df.shape}")
    print(df.loc["SH600519"].head(3).to_string())
    print("...")
    print(df.loc["SH600519"].tail(1).to_string())

    # --- 5. self-check against the daily facts -----------------------------
    sub = df.loc["SH600519"]
    tail = sub.loc[sub.index.normalize() == sub.index.normalize().max()]
    day_last = tail["DayLast($close)"].iloc[-1]
    day_cumvol = tail["DayCumsum($volume, '9:30', '14:59')"].iloc[-1]
    print(
        f"\nself-check on {tail.index[-1]:%Y-%m-%d}: "
        f"DayLast($close)={day_last:.2f}   DayCumsum($volume)={day_cumvol:.0f}"
    )

    # --- 6. the usual Dataset/Handler path works too -----------------------
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP

    handler = DataHandlerLP(
        instruments="all",
        start_time=START,
        end_time=END,
        data_loader={
            "class": "QlibDataLoader",
            "kwargs": {
                "config": (["$close", "Mean($close, 20)"], ["close", "ma20"]),
                "swap_level": False,
                "freq": "1min",
            },
        },
        infer_processors=[],
        learn_processors=[],
    )
    panel = DatasetH(handler=handler, segments={"all": (START, END)}).prepare("all")
    print(f"\nDatasetH panel shape = {panel.shape}")
    print(panel.head(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
