"""Validate an exported Qlib 1min dataset against the warehouse parquet.

Reads the exported dataset through Qlib's stock expression engine and checks the
results against the raw parquet read independently with DuckDB.

    scripts/run_python_3gb.sh sleeve/a/scripts/validate_qlib_1min.py \
        --data /tmp/qlib_1min_out --warehouse /Volumes/lexar_4t/code/data/warehouse
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

QLIB_HOME = Path("/Users/shuzhenyi/code/python/qlib")
if str(QLIB_HOME) not in sys.path:
    sys.path.insert(0, str(QLIB_HOME))


def con(limit="2GB", threads=2):
    c = duckdb.connect(":memory:")
    c.execute(f"SET memory_limit='{limit}'")
    c.execute(f"SET threads={threads}")
    c.execute("SET temp_directory='/tmp/ddb_scratch'")
    c.execute("SET preserve_insertion_order=false")
    return c


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--warehouse", default="/Volumes/lexar_4t/code/data/warehouse")
    p.add_argument("--symbols", default=None, help="comma separated; default = 3 from all.txt")
    p.add_argument("--tol", type=float, default=1e-3, help="relative tolerance (float32 storage)")
    p.add_argument("--kernels", type=int, default=None,
                   help="worker processes; 1 keeps tracebacks readable")
    args = p.parse_args()

    import qlib
    from qlib.constant import REG_CN
    from qlib.data import D
    from qlib.contrib.ops.high_freq import (
        BFillNan, Cut, Date, DayCumsum, DayLast, FFillNan, IsInf, IsNull, Select,
    )

    assert Path(qlib.__file__).is_relative_to(QLIB_HOME), f"wrong qlib: {qlib.__file__}"

    root = Path(args.data)
    all_txt = (root / "instruments" / "all.txt").read_text().strip().splitlines()
    avail = [ln.split("\t")[0] for ln in all_txt]
    syms = (
        [s.strip().upper() for s in args.symbols.split(",")]
        if args.symbols
        else avail[:3]
    )
    print(f"dataset : {root}")
    print(f"symbols : {len(avail)} available, validating {syms}")

    cal = pd.read_csv(root / "calendars" / "1min.txt", header=None)[0]
    start, end = cal.iloc[0], cal.iloc[-1]
    print(f"calendar: {len(cal)} bars, {start} .. {end}")

    qlib.init(
        provider_uri={"1min": str(root)},
        region=REG_CN,
        custom_ops=[DayCumsum, DayLast, FFillNan, BFillNan, Date, Select, IsNull, IsInf, Cut],
        expression_cache=None,
        dataset_cache=None,
        **({"kernels": args.kernels} if args.kernels else {}),
    )

    fields = [
        "$close", "$open", "$high", "$low", "$volume", "$amount", "$vwap",
        "DayLast($close)",
        "DayCumsum($volume, '9:30', '14:59')",
        "Ref($close, 1)",
    ]
    df = D.features(syms, fields, start_time=start, end_time=end, freq="1min")
    print(f"D.features -> {df.shape}")

    # ---- ground truth straight from parquet -------------------------------
    dim = (Path(args.warehouse) / "checkpoints"
           / "instrument-clock-repair-20260914.eZTGRK" / "dim_instrument_before.parquet")
    c = con()
    ids = c.execute(
        f"SELECT symbol_id, qlib_symbol FROM read_parquet('{dim}') "
        f"WHERE qlib_symbol IN ({','.join(repr(s) for s in syms)})"
    ).df()
    id_list = ",".join(str(int(i)) for i in ids["symbol_id"])
    raw = c.execute(
        f"""
        SELECT r.symbol_id, i.qlib_symbol AS sym, r.trade_date, r.minute_slot,
               r.open_i/1e4 AS open, r.high_i/1e4 AS high, r.low_i/1e4 AS low,
               r.close_i/1e4 AS close, r.volume, r.amount_i
        FROM read_parquet('{args.warehouse}/silver/bars_1m_raw/year=*/month=*/exchange=*/bucket=*/*.parquet',
                          hive_partitioning=true) r
        JOIN ids i ON i.symbol_id = r.symbol_id
        WHERE r.symbol_id IN ({id_list})
          AND r.trade_date BETWEEN {start[:4]}{start[5:7]}{start[8:10]}
                               AND {end[:4]}{end[5:7]}{end[8:10]}
        """
    ).df()
    c.close()
    print(f"parquet  -> {raw.shape} rows")

    # ---- compare ----------------------------------------------------------
    checks: list[tuple[str, float, float]] = []

    for sym in syms:
        sub = raw[raw["sym"] == sym].copy()
        if sub.empty:
            print(f"  {sym}: NO PARQUET DATA")
            continue
        sub["dt"] = (
            pd.to_datetime(sub["trade_date"].astype(str), format="%Y%m%d")
            + pd.to_timedelta((sub["minute_slot"] - 1) // 60, unit="h")
            + pd.to_timedelta((sub["minute_slot"] - 1) % 60, unit="m")
        )
        sub = sub.sort_values("dt")
        got = df.loc[sym]

        # per-bar raw fields
        for qf, rawcol in [("$close", "close"), ("$open", "open"), ("$high", "high"),
                           ("$low", "low"), ("$volume", "volume")]:
            want = pd.Series(sub[rawcol].to_numpy(), index=pd.DatetimeIndex(sub["dt"]))
            j = pd.concat([got[qf].rename("g"), want.rename("w")], axis=1).dropna()
            if j.empty:
                checks.append((f"{sym} {qf}", np.inf, 0))
                continue
            rel = float((j["g"] - j["w"]).abs().max() / max(j["w"].abs().max(), 1e-12))
            checks.append((f"{sym} {qf}", rel, len(j)))

        # qlib $amount = amount_i / 1000
        want_amt = pd.Series(sub["amount_i"].to_numpy() / 1000.0, index=pd.DatetimeIndex(sub["dt"]))
        j = pd.concat([got["$amount"].rename("g"), want_amt.rename("w")], axis=1).dropna()
        rel = float((j["g"] - j["w"]).abs().max() / max(j["w"].abs().max(), 1e-12)) if len(j) else np.inf
        checks.append((f"{sym} $amount", rel, len(j)))

        # vwap = amount_i / (volume*100)
        with np.errstate(invalid="ignore", divide="ignore"):
            want_vwap = np.where(sub["volume"].to_numpy() > 0,
                                 sub["amount_i"].to_numpy() / (sub["volume"].to_numpy() * 100.0), np.nan)
        want_vwap = pd.Series(want_vwap, index=pd.DatetimeIndex(sub["dt"]))
        j = pd.concat([got["$vwap"].rename("g"), want_vwap.rename("w")], axis=1).dropna()
        rel = float((j["g"] - j["w"]).abs().max() / max(j["w"].abs().max(), 1e-12)) if len(j) else np.inf
        checks.append((f"{sym} $vwap", rel, len(j)))

        # daily aggregates: DayLast / DayCumsum must equal the parquet day totals
        daily = sub.groupby(sub["dt"].dt.normalize()).agg(
            close=("close", "last"), volume=("volume", "sum"))
        g = got.copy()
        g["day"] = g.index.normalize()
        gl = g.groupby("day")["DayLast($close)"].last()
        gv = g.groupby("day")["DayCumsum($volume, '9:30', '14:59')"].last()
        j = pd.concat([gl.rename("g"), daily["close"].rename("w")], axis=1).dropna()
        rel = float((j["g"] - j["w"]).abs().max() / max(j["w"].abs().max(), 1e-12)) if len(j) else np.inf
        checks.append((f"{sym} DayLast($close)==day close", rel, len(j)))
        j = pd.concat([gv.rename("g"), daily["volume"].rename("w")], axis=1).dropna()
        rel = float((j["g"] - j["w"]).abs().max() / max(j["w"].abs().max(), 1e-12)) if len(j) else np.inf
        checks.append((f"{sym} DayCumsum($volume)==day volume", rel, len(j)))

    print("\n" + "=" * 78)
    ok = True
    for name, rel, n in checks:
        good = rel <= args.tol
        ok &= good
        print(f"  {'OK  ' if good else 'FAIL'} {name:<45} n={n:<6} rel={rel:.3e}")
    print("=" * 78)
    print("RESULT:", "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
