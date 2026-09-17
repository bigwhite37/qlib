#!/usr/bin/env python
"""Build a Qlib **minute** provider database that reads the warehouse via DuckDB.

This is route B of ``docs/qlib_1min_integration_plan.md``: instead of exporting a
55 GB copy of the warehouse into Qlib ``.bin`` files, the provider database only
holds the calendar, the instrument spans and a description of where the minute
bars live.  ``qlib.data.storage.duckdb_storage.DuckDBFeatureStorage`` then reads
the parquet on demand, pruned by ``year/month/exchange/bucket``.

Written layout (all tables inside ``--out``)::

    qlib_minute_calendar(trade_index, trade_date, trade_date_text, minute_slot, freq)
    v_qlib_instruments_latest(qlib_symbol, start_date, end_date)
    qlib_minute_sources(field, source_expr, parquet_root, symbol_table, price_unit)

Minute labels follow Qlib's own convention: the warehouse ``minute_slot`` is the
bar's **closing** instant, so the Qlib label is ``minute_slot - 1``
(09:30..11:29 + 13:00..14:59 == 240 bars/day).

Every DuckDB connection in this script -- the read-only warehouse scans and the
provider database write -- is opened with ``memory_limit='2GB'``.

Usage::

    scripts/run_python_3gb.sh sleeve/a/scripts/build_qlib_1min_provider.py \
        --out /Volumes/lexar_4t/code/data/qlib/sleeve/a/minute_provider.duckdb \
        --start 2025-01-02 --end 2025-12-31
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

QLIB_HOME = Path("/Users/shuzhenyi/code/python/qlib")
if str(QLIB_HOME) not in sys.path:
    sys.path.insert(0, str(QLIB_HOME))

WAREHOUSE_DEFAULT = "/Volumes/lexar_4t/code/data/warehouse"
DIM_INSTRUMENT_DEFAULT = (
    "{warehouse}/checkpoints/instrument-clock-repair-20260914.eZTGRK/dim_instrument_before.parquet"
)
MEMORY_LIMIT = "2GB"
TEMP_DIRECTORY = "/tmp/ddb_scratch"

MORNING_SLOTS = (571, 690)
AFTERNOON_SLOTS = (781, 900)
SLOT_OFFSET_MINUTES = 1
BARS_PER_DAY = 240

#: qlib field -> SQL over the warehouse columns.  Units are the ones verified in
#: ``docs/qlib_1min_integration_plan.md``: ``*_i`` are 1e-4 CNY, ``volume`` is in
#: hands, ``amount_i`` is in CNY (the schema doc claiming cents is wrong).
SOURCE_EXPRESSIONS = {
    "open": ("CAST(open_i AS DOUBLE)/1e4", 1e-4),
    "high": ("CAST(high_i AS DOUBLE)/1e4", 1e-4),
    "low": ("CAST(low_i AS DOUBLE)/1e4", 1e-4),
    "close": ("CAST(close_i AS DOUBLE)/1e4", 1e-4),
    "volume": ("CAST(volume AS DOUBLE)", 1.0),
    "amount": ("CAST(amount_i AS DOUBLE)/1000", 1000.0),
    "vwap": (
        "CASE WHEN volume > 0 THEN CAST(amount_i AS DOUBLE)/ (CAST(volume AS DOUBLE)*100) END",
        1.0,
    ),
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def connect(path: str | None = None, read_only: bool = False):
    """Open DuckDB with the project's mandatory 2 GB limit."""

    target = str(path) if path else ":memory:"
    connection = duckdb.connect(target, read_only=read_only)
    connection.execute(f"SET memory_limit='{MEMORY_LIMIT}'")
    connection.execute("SET threads=2")
    connection.execute(f"SET temp_directory='{TEMP_DIRECTORY}'")
    connection.execute(f"SET max_temp_directory_size='{MEMORY_LIMIT}'")
    connection.execute("SET preserve_insertion_order=false")
    applied = connection.execute("SELECT current_setting('memory_limit')").fetchone()[0]
    if _size_in_bytes(applied) > _size_in_bytes(MEMORY_LIMIT):
        raise RuntimeError(f"DuckDB memory limit is {applied}, expected at most {MEMORY_LIMIT}")
    return connection


_SIZE_UNITS = {"B": 1, "KB": 10**3, "MB": 10**6, "GB": 10**9,
               "KIB": 1024, "MIB": 1024**2, "GIB": 1024**3}


def _size_in_bytes(text: str) -> float:
    """DuckDB renders ``2GB`` as ``1.8 GiB``; compare byte values, not strings."""

    compact = "".join(str(text).split())
    match = __import__("re").fullmatch(r"([0-9]*\.?[0-9]+)([A-Za-z]+)", compact)
    if match is None:
        raise ValueError(f"cannot parse DuckDB size {text!r}")
    return float(match.group(1)) * _SIZE_UNITS[match.group(2).upper()]


def minute_labels() -> list[int]:
    slots = list(range(MORNING_SLOTS[0], MORNING_SLOTS[1] + 1)) + list(
        range(AFTERNOON_SLOTS[0], AFTERNOON_SLOTS[1] + 1)
    )
    return [slot - SLOT_OFFSET_MINUTES for slot in slots]


def build_calendar(connection, bars: Path, start_i: int, end_i: int, exchanges: list[str]) -> pd.DataFrame:
    exchange_sql = ",".join("'" + name + "'" for name in exchanges)
    days = [
        int(row[0])
        for row in connection.execute(
            f"SELECT DISTINCT trade_date FROM read_parquet('{bars}/year=*/month=*/"
            f"exchange=*/bucket=*/*.parquet', hive_partitioning=true) "
            f"WHERE exchange IN ({exchange_sql}) AND trade_date BETWEEN {start_i} AND {end_i} "
            f"ORDER BY trade_date"
        ).fetchall()
    ]
    if not days:
        raise SystemExit("no trading days in the requested range")
    labels = minute_labels()
    slots = [label + SLOT_OFFSET_MINUTES for label in labels]
    frames = []
    for trade_date in days:
        day = pd.Timestamp(str(trade_date))
        stamps = day + pd.to_timedelta([label // 60 for label in labels], unit="h") + pd.to_timedelta(
            [label % 60 for label in labels], unit="m"
        )
        frames.append(
            pd.DataFrame(
                {
                    "trade_date": trade_date,
                    "trade_date_text": stamps.strftime("%Y-%m-%d %H:%M:%S"),
                    "minute_slot": slots,
                }
            )
        )
    calendar = pd.concat(frames, ignore_index=True)
    calendar.insert(0, "trade_index", range(len(calendar)))
    calendar["freq"] = "1min"
    log(f"calendar: {len(days)} trading days, {len(calendar)} bars")
    return calendar


def build_instruments(connection, dim_path: str, exchanges: list[str]) -> pd.DataFrame:
    frame = connection.execute(
        f"SELECT symbol_id, qlib_symbol, exchange, list_date, delist_date FROM "
        f"read_parquet('{dim_path}') WHERE qlib_symbol IS NOT NULL"
    ).df()
    frame = frame[frame["exchange"].isin(exchanges)]
    frame = frame[frame["qlib_symbol"].str.match(r"^(SH6[08]\d{4}|SZ(?:00|30)\d{4}|BJ\d{6})$")]
    if frame.empty:
        raise SystemExit("dim_instrument has no matching A-share symbols")
    frame = frame.drop_duplicates("qlib_symbol").sort_values("qlib_symbol")
    log(f"instruments: {len(frame)} symbols")
    return frame[["qlib_symbol", "list_date", "delist_date"]].rename(
        columns={"list_date": "start_date", "delist_date": "end_date"}
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--warehouse", default=WAREHOUSE_DEFAULT)
    parser.add_argument("--dim-instrument", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--exchanges", default="SH,SZ")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    exchanges = [name.strip().upper() for name in args.exchanges.split(",") if name.strip()]
    start_i = int(pd.Timestamp(args.start).strftime("%Y%m%d"))
    end_i = int(pd.Timestamp(args.end).strftime("%Y%m%d"))
    bars = Path(args.warehouse) / "silver" / "bars_1m_raw"
    dim_path = args.dim_instrument or DIM_INSTRUMENT_DEFAULT.format(warehouse=args.warehouse)
    out = Path(args.out)
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} exists; pass --overwrite")
        out.unlink()
    out.parent.mkdir(parents=True, exist_ok=True)

    reader = connect()
    try:
        calendar = build_calendar(reader, bars, start_i, end_i, exchanges)
        instruments = build_instruments(reader, dim_path, exchanges)
    finally:
        reader.close()

    writer = connect(str(out))
    try:
        writer.register("calendar_rows", calendar)
        writer.register("instrument_rows", instruments)
        writer.execute(
            "CREATE TABLE qlib_minute_calendar AS SELECT trade_index, trade_date, "
            "trade_date_text, minute_slot, freq FROM calendar_rows"
        )
        writer.execute(
            "CREATE TABLE v_qlib_instruments_latest AS SELECT qlib_symbol, start_date, end_date "
            "FROM instrument_rows"
        )
        sources = pd.DataFrame(
            [
                {
                    "field": field,
                    "source_expr": expression,
                    "parquet_root": str(bars),
                    "symbol_table": str(dim_path),
                    "price_unit": unit,
                }
                for field, (expression, unit) in SOURCE_EXPRESSIONS.items()
            ]
        )
        writer.register("source_rows", sources)
        writer.execute(
            "CREATE TABLE qlib_minute_sources AS SELECT field, source_expr, parquet_root, "
            "symbol_table, price_unit FROM source_rows"
        )
        writer.execute("CHECKPOINT")
    finally:
        writer.close()

    log(f"provider written: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    log("fields: " + ", ".join(SOURCE_EXPRESSIONS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
