"""Exercise qlib's stock expression engine against the 1min probe dataset.

No qlib source is modified here: this answers "does the unmodified expression
engine already work at freq=1min, and what has to be registered/added on top".
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import qlib
from qlib.constant import REG_CN
from qlib.data import D
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

PROBE = Path("/tmp/qlib1m_probe/cn_data_1min")
RAW = pd.read_parquet("/tmp/qlib1m_probe/raw_minutes.parquet")

START = "2026-09-01 09:30:00"
END = "2026-09-16 14:59:00"

HIGH_FREQ_OPS = [DayCumsum, DayLast, FFillNan, BFillNan, Date, Select, IsNull, IsInf, Cut]


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    qlib.init(
        provider_uri={"1min": str(PROBE)},
        region=REG_CN,
        custom_ops=HIGH_FREQ_OPS,
        expression_cache=None,
        dataset_cache=None,
    )
    print("qlib initialised on", PROBE)

    section("1. calendar served at freq=1min")
    cal = D.calendar(start_time=START, end_time=END, freq="1min")
    print("bars:", len(cal), "first:", cal[0], "last:", cal[-1])
    per_day = pd.Series(1, index=pd.DatetimeIndex(cal)).resample("1D").sum()
    print("bars per day:", per_day[per_day > 0].to_dict())

    section("2. native expressions evaluated by the stock engine")
    fields = [
        "$close",
        "$open",
        "Ref($close, 1)",
        "Mean($close, 5)",
        "Std($close, 20)",
        "Delta($close, 1)",
        "Corr($close, $volume, 10)",
        "DayLast($close)",
        "DayCumsum($volume, '9:30', '14:59')",
        "$vwap",
        "$amount",
    ]
    df = D.features(["SH600519"], fields, start_time=START, end_time=END, freq="1min")
    print("shape:", df.shape)
    print(df.head(4).to_string())
    print("...")
    print(df.tail(2).to_string())

    section("3. correctness vs pandas ground truth (SH600519)")
    sub = RAW[RAW["symbol_id"] == 1600519].sort_values("dt")
    raw_close = pd.Series(
        sub["close_i"].to_numpy() / 1e4, index=pd.DatetimeIndex(sub["dt"]), name="close"
    )
    raw_vol = pd.Series(
        sub["volume"].to_numpy().astype("float64"), index=pd.DatetimeIndex(sub["dt"])
    )
    got = df.loc["SH600519"]

    checks = {
        "$close": (got["$close"], raw_close),
        "Ref($close,1)": (got["Ref($close, 1)"], raw_close.shift(1)),
        "Mean($close,5)": (got["Mean($close, 5)"], raw_close.rolling(5).mean()),
        "Delta($close,1)": (got["Delta($close, 1)"], raw_close.diff(1)),
        "DayLast($close)": (
            got["DayLast($close)"],
            raw_close.groupby(raw_close.index.normalize()).transform("last"),
        ),
        "DayCumsum($volume)": (
            got["DayCumsum($volume, '9:30', '14:59')"],
            raw_vol.groupby(raw_vol.index.normalize()).cumsum(),
        ),
    }
    ok = True
    for name, (a, b) in checks.items():
        aligned = pd.concat([a.rename("got"), b.rename("want")], axis=1).dropna()
        if aligned.empty:
            print(f"  {name:<24} EMPTY -> cannot compare")
            ok = False
            continue
        diff = (aligned["got"] - aligned["want"]).abs()
        # float32 storage in qlib bins vs float64 reference
        worst = float(diff.max())
        rel = worst / max(float(aligned["want"].abs().max()), 1e-12)
        status = "OK " if rel < 1e-5 else "FAIL"
        if rel >= 1e-5:
            ok = False
        print(f"  {name:<24} {status}  n={len(aligned):<6} max_abs_diff={worst:.3e} rel={rel:.2e}")

    section("4. expression spanning a session boundary / overnight gap")
    # Ref at the first bar of a day must reach back into the *previous* day
    first_bars = got.index[got.index.time == pd.Timestamp("09:30").time()][1:]
    sample = first_bars[0]
    prev = got["Ref($close, 1)"].loc[sample]
    print(f"  bar {sample}  Ref($close,1)={prev}")
    print("  -> the 1min calendar is one flat axis, so Ref/Mean silently cross the")
    print("     lunch break and the overnight gap; there is no session-aware rolling.")

    section("5. what is NOT registered by default")
    from qlib.data.ops import Operators

    for name in ["Ref", "Mean", "Sum", "Corr", "DayLast", "DayCumsum", "Cut", "Select", "Date"]:
        try:
            Operators.__getattr__(name)
            builtin = True
        except AttributeError:
            builtin = False
        print(f"  {name:<12} reachable={builtin}")

    print("\nRESULT:", "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
