"""Second probe: 1min expression *semantics* and *scale* in stock qlib.

Part A pins down what the built-in rolling operators actually do across the
lunch break and the overnight gap, and checks which high-frequency operators are
registered out of the box.

Part B builds a ~200-symbol / 12-day 1min dataset straight from the parquet and
times a realistic cross-sectional ``D.features`` call, plus reports the on-disk
cost of qlib's native binary layout.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import duckdb
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

WAREHOUSE = Path("/Volumes/lexar_4t/code/data/warehouse")
BARS_1M = WAREHOUSE / "silver" / "bars_1m_raw"
DIM = (
    WAREHOUSE
    / "checkpoints"
    / "instrument-clock-repair-20260914.eZTGRK"
    / "dim_instrument_before.parquet"
)
SMALL = Path("/tmp/qlib1m_probe/cn_data_1min")
BIG = Path("/tmp/qlib1m_probe/cn_data_1min_scale")
RAW = pd.read_parquet("/tmp/qlib1m_probe/raw_minutes.parquet")

START = "2026-09-01 09:30:00"
END = "2026-09-16 14:59:00"
N_SYMBOLS = 200

HF_OPS = [DayCumsum, DayLast, FFillNan, BFillNan, Date, Select, IsNull, IsInf, Cut]


def con():
    c = duckdb.connect(":memory:")
    c.execute("SET memory_limit='2GB'")
    c.execute("SET threads=2")
    c.execute("SET temp_directory='/tmp/ddb_scratch'")
    c.execute("SET preserve_insertion_order=false")
    return c


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# --------------------------------------------------------------------------- A
def part_a():
    qlib.init(
        provider_uri={"1min": str(SMALL)},
        region=REG_CN,
        custom_ops=HF_OPS,
        expression_cache=None,
        dataset_cache=None,
    )
    section("A1. which high-frequency operators are in the stock OpsList?")
    from qlib.data.ops import OpsList

    builtin = {c.__name__ for c in OpsList}
    for name in ["Ref", "Mean", "Std", "Corr", "Delta", "DayLast", "DayCumsum", "Cut", "Select", "Date"]:
        print(f"  {name:<12} in qlib.data.ops.OpsList = {name in builtin}")
    print("  -> the Day*/Cut/Select/Date family lives in qlib.contrib.ops.high_freq and")
    print("     only exists if the caller passes custom_ops=[...] to qlib.init().")

    section("A2. rolling operators on the flat 1min axis")
    df = D.features(
        ["SH600519"],
        ["$close", "Ref($close, 1)", "Ref($close, 2)", "Mean($close, 240)"],
        start_time="2026-09-01 09:30:00",
        end_time="2026-09-03 14:59:00",
        freq="1min",
    ).loc["SH600519"]
    probe = [
        ("2026-09-01 11:29:00", "last bar before lunch"),
        ("2026-09-01 13:00:00", "first bar after lunch"),
        ("2026-09-02 09:30:00", "first bar of next day"),
        ("2026-09-02 09:31:00", "second bar of next day"),
    ]
    for ts, note in probe:
        ts = pd.Timestamp(ts)
        r = df.loc[ts]
        print(
            f"  {ts}  ({note:<26}) $close={r['$close']:>9.2f} "
            f"Ref1={r['Ref($close, 1)']:>9.2f} Ref2={r['Ref($close, 2)']:>9.2f}"
        )
    print("  09:30 on 09-02 Ref1 equals 2026-09-01 14:59 close  -> Ref crosses the overnight gap")
    print("  13:00 on 09-01 Ref1 equals 2026-09-01 11:29 close  -> Ref crosses the lunch break")
    print("  => there is NO session-aware or day-anchored rolling in stock qlib.")

    section("A3. DayCumsum with a partial session window")
    df2 = D.features(
        ["SH600519"],
        [
            "$volume",
            "DayCumsum($volume, '9:30', '14:59')",
            "DayCumsum($volume, '9:30', '10:00')",
        ],
        start_time="2026-09-01 09:30:00",
        end_time="2026-09-01 14:59:00",
        freq="1min",
    ).loc["SH600519"]
    for ts in ["2026-09-01 09:30:00", "2026-09-01 10:00:00", "2026-09-01 10:01:00", "2026-09-01 14:59:00"]:
        r = df2.loc[pd.Timestamp(ts)]
        print(
            f"  {ts}  $volume={r['$volume']:>8.0f}  full-day-cum={r['DayCumsum($volume, \'9:30\', \'14:59\')']:>9.0f}"
            f"  window-9:30-10:00-cum={r['DayCumsum($volume, \'9:30\', \'10:00\')']:>9.0f}"
        )
    print("  -> after the window closes the operator returns 0, not the frozen total.")

    section("A4. DayLast / Date as day-boundary anchors")
    df3 = D.features(
        ["SH600519"],
        ["$close", "DayLast($close)", "DayLast(Ref($close, 240))", "Date($close)"],
        start_time="2026-09-02 09:30:00",
        end_time="2026-09-02 09:33:00",
        freq="1min",
    ).loc["SH600519"]
    print(df3.to_string())


# --------------------------------------------------------------------------- B
def build_scale():
    section(f"B1. building a {N_SYMBOLS}-symbol scale dataset from parquet")
    c = con()
    t0 = time.time()
    ids = c.execute(
        f"""
        SELECT DISTINCT symbol_id FROM read_parquet(
            '{BARS_1M}/year=2026/month=09/exchange=SH/bucket=*/*.parquet')
        ORDER BY symbol_id LIMIT {N_SYMBOLS}
        """
    ).df()["symbol_id"].tolist()
    inst = c.execute(
        f"SELECT symbol_id, qlib_symbol FROM read_parquet('{DIM}') "
        f"WHERE symbol_id IN ({','.join(str(int(i)) for i in ids)})"
    ).df()
    minutes = c.execute(
        f"""
        SELECT symbol_id, trade_date, minute_slot, open_i, high_i, low_i, close_i,
               volume, amount_i
        FROM read_parquet('{BARS_1M}/year=2026/month=09/exchange=SH/bucket=*/*.parquet')
        WHERE symbol_id IN ({','.join(str(int(i)) for i in ids)})
          AND trade_date BETWEEN 20260901 AND 20260916
        ORDER BY symbol_id, trade_date, minute_slot
        """
    ).df()
    c.close()
    print(f"  parquet read: {len(minutes):,} rows for {len(inst)} symbols in {time.time() - t0:.1f}s")

    minutes["dt"] = (
        pd.to_datetime(minutes["trade_date"].astype(str), format="%Y%m%d")
        + pd.to_timedelta((minutes["minute_slot"] - 1) // 60, unit="h")
        + pd.to_timedelta((minutes["minute_slot"] - 1) % 60, unit="m")
    )
    cal = pd.DatetimeIndex(sorted(minutes["dt"].unique()))
    pos = {t: i for i, t in enumerate(cal)}

    if BIG.exists():
        shutil.rmtree(BIG)
    (BIG / "calendars").mkdir(parents=True)
    (BIG / "instruments").mkdir(parents=True)
    (BIG / "calendars" / "1min.txt").write_text(
        "\n".join(t.strftime("%Y-%m-%d %H:%M:%S") for t in cal) + "\n"
    )

    lines, sizes = [], []
    for _, row in inst.iterrows():
        sub = minutes[minutes["symbol_id"] == row["symbol_id"]].sort_values("dt")
        if sub.empty:
            continue
        fdir = BIG / "features" / row["qlib_symbol"].lower()
        fdir.mkdir(parents=True, exist_ok=True)
        si = pos[sub["dt"].iloc[0]]
        exp = cal[si : si + len(sub)]
        if not np.array_equal(exp.values, sub["dt"].values):
            continue  # non-contiguous -> skip for this probe
        vol = sub["volume"].to_numpy().astype("float64")
        amt = sub["amount_i"].to_numpy().astype("float64")
        fv = {
            "open": sub["open_i"].to_numpy() / 1e4,
            "high": sub["high_i"].to_numpy() / 1e4,
            "low": sub["low_i"].to_numpy() / 1e4,
            "close": sub["close_i"].to_numpy() / 1e4,
            "volume": vol,
            "amount": amt / 1000.0,
        }
        for name, vals in fv.items():
            p = fdir / f"{name}.1min.bin"
            buf = np.empty(len(vals) + 1, dtype="<f4")
            buf[0] = float(si)
            buf[1:] = vals
            with p.open("wb") as fp:
                buf.tofile(fp)
            sizes.append(p.stat().st_size)
        lines.append(f"{row['qlib_symbol']}\t{sub['dt'].min():%Y-%m-%d}\t{sub['dt'].max():%Y-%m-%d}")
    (BIG / "instruments" / "all.txt").write_text("\n".join(lines) + "\n")
    total = sum(p.stat().st_size for p in BIG.rglob("*.bin"))
    nfiles = sum(1 for _ in BIG.rglob("*.bin"))
    print(f"  wrote {len(lines)} symbols, {nfiles} .bin files, {total / 1e6:.1f} MB")
    print(f"  bytes per (symbol, field): {np.mean(sizes):,.0f} for {len(cal)} bars")
    print(f"  => bytes per value: {np.mean(sizes) / (len(cal) + 1):.2f} (qlib stores float32)")


def part_b():
    build_scale()
    section("B2. timing a realistic cross-sectional D.features at freq=1min")
    qlib.init(
        provider_uri={"1min": str(BIG)},
        region=REG_CN,
        custom_ops=HF_OPS,
        expression_cache=None,
        dataset_cache=None,
    )
    cal = D.calendar(start_time=START, end_time=END, freq="1min")
    print(f"  calendar bars: {len(cal):,}")
    insts = sorted(
        p.name.upper() for p in (BIG / "features").iterdir() if p.is_dir()
    )
    print(f"  instruments: {len(insts)}")

    for fields, label in [
        (["$close"], "1 raw field"),
        (["$close", "$open", "$high", "$low", "$volume"], "5 raw fields"),
        (["$close", "Mean($close, 20)", "Std($close, 20)"], "1 raw + 2 rolling(20)"),
    ]:
        t0 = time.time()
        df = D.features(insts, fields, start_time=START, end_time=END, freq="1min")
        dt = time.time() - t0
        cells = df.shape[0] * df.shape[1]
        print(
            f"  {label:<24} shape={df.shape}  {dt:6.2f}s  "
            f"{df.shape[0] / dt:,.0f} rows/s  {cells / dt / 1e6:.2f} Mcells/s"
        )
        del df


if __name__ == "__main__":
    import sys

    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "a"):
        part_a()
    if which in ("all", "b"):
        part_b()
    print("\n[memlimit] probe done")
