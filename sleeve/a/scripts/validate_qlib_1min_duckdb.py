#!/usr/bin/env python
"""Contract test for the DuckDB minute provider (route B of the 1min plan).

Reads the minute dataset **through Qlib's own expression engine** and checks it
against the warehouse parquet read independently with DuckDB.  The assertions are
the ones frozen in ``docs/qlib_1min_integration_plan.md`` §P0:

* ``DayLast($close)``    == the day's 15:00 bar close (raw CNY)
* ``DayCumsum($volume)`` == the day's summed hands
* ``DayCumsum($amount)`` == the day's summed amount (thousand CNY, Qlib's unit)
* every covered day has exactly 240 bars (no half days in 2020-2026)
* bridge: ``DayLast($close) * factor == daily close`` and
  ``DayCumsum($volume) == daily volume * factor`` against the day database

Every DuckDB connection carries the mandatory ``memory_limit='2GB'``.

Usage::

    scripts/run_python_3gb.sh sleeve/a/scripts/validate_qlib_1min_duckdb.py \
        --provider /tmp/qlib_1min_provider/minute_provider.duckdb \
        --symbols SH600519,SZ000001 --start 2025-02-10 --end 2025-02-14
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

WAREHOUSE_DEFAULT = "/Volumes/lexar_4t/code/data/warehouse"
DAILY_DEFAULT = "/Users/shuzhenyi/code/data/qlib_tmp_correct/qlib_tmp_correct_reindexed.duckdb"
DIM_DEFAULT = (
    "{warehouse}/checkpoints/instrument-clock-repair-20260914.eZTGRK/dim_instrument_before.parquet"
)
MEMORY_LIMIT = "2GB"
TEMP_DIRECTORY = "/tmp/ddb_scratch"
BARS_PER_DAY = 240


def connect():
    connection = duckdb.connect(":memory:")
    connection.execute(f"SET memory_limit='{MEMORY_LIMIT}'")
    connection.execute("SET threads=2")
    connection.execute(f"SET temp_directory='{TEMP_DIRECTORY}'")
    connection.execute(f"SET max_temp_directory_size='{MEMORY_LIMIT}'")
    connection.execute("SET preserve_insertion_order=false")
    return connection


def connect_db(path: str):
    """Read-only connection to a DuckDB database file, still capped at 2 GB."""

    connection = duckdb.connect(path, read_only=True)
    connection.execute(f"SET memory_limit='{MEMORY_LIMIT}'")
    connection.execute("SET threads=2")
    connection.execute(f"SET temp_directory='{TEMP_DIRECTORY}'")
    connection.execute("SET preserve_insertion_order=false")
    return connection


def warehouse_daily(connection, bars: Path, symbol_id: int, exchange: str, start_i: int, end_i: int):
    """Independent DuckDB aggregation of the same days, straight from parquet."""

    bucket = int(symbol_id) % 8
    pattern = f"{bars}/year=*/month=*/exchange={exchange}/bucket={bucket:02d}/*.parquet"
    return connection.execute(
        f"""
        SELECT trade_date,
               arg_max(close_i, minute_slot)/1e4 AS ref_close,
               sum(volume) AS ref_volume,
               sum(amount_i)/1000 AS ref_amount,
               count(*) AS ref_bars
        FROM read_parquet('{pattern}', hive_partitioning=true)
        WHERE symbol_id = ? AND trade_date BETWEEN ? AND ?
        GROUP BY trade_date ORDER BY trade_date
        """,
        [int(symbol_id), start_i, end_i],
    ).df()


def daily_reference(connection, symbols: list[str], start_i: int, end_i: int):
    placeholders = ",".join("?" * len(symbols))
    return connection.execute(
        f"""
        SELECT qlib_symbol, trade_date, close, volume, factor
        FROM (
            SELECT qlib_symbol, trade_date, close, volume, factor,
                   row_number() OVER (
                       PARTITION BY qlib_symbol, trade_date
                       ORDER BY CASE row_kind WHEN 'final' THEN 3 WHEN 'intraday' THEN 2
                                              WHEN 'historical' THEN 1 ELSE 0 END DESC,
                            batch_id DESC
                   ) AS rn
            FROM qlib_daily_features
            WHERE qlib_symbol IN ({placeholders}) AND trade_date BETWEEN ? AND ?
              AND row_status = 'active'
        )
        WHERE rn = 1
        """,
        [*symbols, start_i, end_i],
    ).df()


def report(label: str, actual: np.ndarray, expected: np.ndarray, tolerance: float) -> int:
    scale = np.maximum(np.abs(expected.astype(float)), 1e-9)
    error = float(np.max(np.abs(actual.astype(float) - expected.astype(float)) / scale))
    ok = error <= tolerance
    print(f"{'OK  ' if ok else 'FAIL'} {label:44s} n={len(actual):3d} rel={error:.3e}")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--provider", required=True)
    parser.add_argument("--warehouse", default=WAREHOUSE_DEFAULT)
    parser.add_argument("--daily", default=DAILY_DEFAULT)
    parser.add_argument("--dim-instrument", default=None)
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--tol", type=float, default=1e-4, help="relative tolerance (float32)")
    args = parser.parse_args()

    import qlib
    from qlib.data import D

    if Path(qlib.__file__).resolve().parent != (QLIB_HOME / "qlib").resolve():
        raise SystemExit(f"loaded the wrong qlib: {qlib.__file__}")
    qlib.init(
        provider_uri={"day": args.daily, "1min": args.provider},
        region="cn",
        kernels=1,
        expression_cache=None,
        dataset_cache=None,
    )

    symbols = [name.strip().upper() for name in args.symbols.split(",") if name.strip()]
    start_i = int(pd.Timestamp(args.start).strftime("%Y%m%d"))
    end_i = int(pd.Timestamp(args.end).strftime("%Y%m%d"))
    # Qlib resolves start/end through the **minute** calendar, so a bare date as
    # `end_time` means 00:00 and silently drops that day's bars.  Pin the session bounds.
    start_time = f"{pd.Timestamp(args.start):%Y-%m-%d} 09:30:00"
    end_time = f"{pd.Timestamp(args.end):%Y-%m-%d} 14:59:00"
    bars = Path(args.warehouse) / "silver" / "bars_1m_raw"
    dim_path = args.dim_instrument or DIM_DEFAULT.format(warehouse=args.warehouse)

    connection = connect()
    try:
        mapping = connection.execute(
            f"SELECT symbol_id, qlib_symbol, exchange FROM read_parquet('{dim_path}') "
            f"WHERE qlib_symbol IN ({','.join('?' * len(symbols))})",
            symbols,
        ).df()
    finally:
        connection.close()
    daily_connection = connect_db(args.daily)
    try:
        reference = daily_reference(daily_connection, symbols, start_i, end_i).rename(
            columns={"close": "ref_daily_close", "volume": "ref_daily_volume"}
        )
    finally:
        daily_connection.close()
    if mapping.empty:
        raise SystemExit("dim_instrument does not know these symbols")

    print("qlib     :", qlib.__file__)
    print("provider :", args.provider)
    print("calendar :", len(D.calendar(freq="1min")), "bars")
    print("window   :", pd.Timestamp(args.start).date(), "..", pd.Timestamp(args.end).date())

    fields = ["DayLast($close)", "DayCumsum($volume)", "DayCumsum($amount)"]
    failures = 0
    for row in mapping.itertuples():
        frame = D.features(
            [row.qlib_symbol], fields, start_time=start_time, end_time=end_time, freq="1min"
        )
        if frame.empty:
            print(f"FAIL {row.qlib_symbol}: no minute bars in the window")
            failures += 1
            continue
        frame.columns = ["day_close", "day_volume", "day_amount"]
        bars_frame = frame.reset_index()
        bars_frame["trade_date"] = bars_frame["datetime"].dt.strftime("%Y%m%d").astype(int)
        # One row per (instrument, day): the last bar of the day carries the frozen
        # DayLast / DayCumsum values.
        last = (
            bars_frame.sort_values("datetime")
            .groupby(["instrument", "trade_date"], as_index=False)
            .tail(1)
        )
        last["instrument"] = last["instrument"].astype(str)
        connection = connect()
        try:
            raw = warehouse_daily(
                connection, bars, int(row.symbol_id), row.exchange, start_i, end_i
            )
        finally:
            connection.close()
        merged = last.merge(raw, on="trade_date", how="outer", indicator=True)
        # A suspended day legitimately has no warehouse bar while Qlib still returns the
        # calendar position; the opposite direction (warehouse bars missing from Qlib) is
        # a real read/calendar bug.
        if (merged["_merge"] == "right_only").any():
            missing = merged.loc[merged["_merge"] == "right_only", "trade_date"].tolist()
            print(f"FAIL {row.qlib_symbol}: warehouse days missing from Qlib: {missing[:5]}")
            failures += 1
            continue
        merged = merged.loc[merged["_merge"] == "both"]
        failures += report(
            f"{row.qlib_symbol} DayLast($close)==15:00 close",
            merged["day_close"].to_numpy(),
            merged["ref_close"].to_numpy(),
            args.tol,
        )
        failures += report(
            f"{row.qlib_symbol} DayCumsum($volume)==Σvolume",
            merged["day_volume"].to_numpy(),
            merged["ref_volume"].to_numpy(),
            args.tol,
        )
        failures += report(
            f"{row.qlib_symbol} DayCumsum($amount)==Σamount/1e3",
            merged["day_amount"].to_numpy(),
            merged["ref_amount"].to_numpy(),
            args.tol,
        )
        if not (merged["ref_bars"] == BARS_PER_DAY).all():
            print(f"FAIL {row.qlib_symbol}: a day does not have {BARS_PER_DAY} bars")
            failures += 1
        # Bridge to the day database; raw = adjusted / factor.
        bridge = merged.merge(
            reference.loc[reference["qlib_symbol"] == row.qlib_symbol],
            on="trade_date",
            how="inner",
        )
        if bridge.empty:
            print(f"WARN {row.qlib_symbol}: no daily rows to bridge")
            continue
        failures += report(
            f"{row.qlib_symbol} DayLast*factor==daily close",
            bridge["day_close"].to_numpy() * bridge["factor"].to_numpy(),
            bridge["ref_daily_close"].to_numpy(),
            args.tol,
        )
        failures += report(
            f"{row.qlib_symbol} DayCumsum==daily volume*factor",
            bridge["day_volume"].to_numpy(),
            bridge["ref_daily_volume"].to_numpy() * bridge["factor"].to_numpy(),
            args.tol,
        )

    print("ALL CHECKS PASSED" if failures == 0 else f"{failures} CHECK(S) FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
