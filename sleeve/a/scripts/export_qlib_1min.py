#!/usr/bin/env python
"""Export the warehouse 1-minute parquet into a Qlib-native dataset.

Reads ``silver/bars_1m_raw`` read-only and writes::

    <out>/calendars/1min.txt                    one bar timestamp per line
    <out>/calendars/day.txt                     one trading day per line
    <out>/instruments/all.txt                   SYMBOL<TAB>start<TAB>end
    <out>/features/<sym_lower>/<field>.1min.bin float32: [start_index, v0, v1, ...]
    <out>/manifest.json                         provenance + counts

Contract notes (all verified against the warehouse, see
``docs/qlib_1min_integration_plan.md``):

* ``minute_slot`` is ``HH*60+MM`` of the bar's **closing** instant
  (571=09:31 .. 690=11:30, 781=13:01 .. 900=15:00). Qlib's 1min label is
  ``minute_slot - 1`` -> 09:30..11:29 + 13:00..14:59, exactly 240 bars/day.
* ``*_i`` prices are scaled by 1e4 -> CNY.
* ``amount_i`` is in **CNY** (the schema doc claiming "cents" is wrong)
  -> qlib ``$amount`` = ``amount_i / 1000``.
* ``volume`` is in **hands** (100 shares) -> ``vwap = amount_i / (volume * 100)``.

Two things that are easy to get wrong and are handled here explicitly:

1. A Qlib ``.bin`` is ``float32`` **with the start calendar index as its first
   element**. Writing only the values makes Qlib read the first price as the
   index and every series shifts by one bar.
2. A symbol's bars are spread over many partitions (one per month), so bins must
   be *merged* across partitions, not rewritten. We accumulate one
   (exchange, bucket, year) group in memory and merge it into the bins, which
   bounds peak memory to roughly one year of one bucket.

Every DuckDB connection is capped: the warehouse holds an 18.7-billion-row
catalog table and an uncapped scan will take the machine down.

Usage::

    scripts/run_python_3gb.sh sleeve/a/scripts/export_qlib_1min.py \
        --out /tmp/qlib_1min_out --start 2026-09-01 --end 2026-09-16 \
        --exchanges SH --limit 50
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

WAREHOUSE_DEFAULT = "/Volumes/lexar_4t/code/data/warehouse"
DIM_INSTRUMENT_DEFAULT = (
    "{warehouse}/checkpoints/instrument-clock-repair-20260914.eZTGRK/dim_instrument_before.parquet"
)
FACTOR_DB_DEFAULT = "/Users/shuzhenyi/code/data/qlib_tmp_correct/qlib_tmp_correct_reindexed.duckdb"

MEANINGFUL_FIELDS = ("open", "high", "low", "close", "volume", "amount", "vwap", "factor")

SLOT_OFFSET_MINUTES = 1  # qlib label = minute_slot - 1 minute
MORNING_SLOTS = (571, 690)  # 09:31 .. 11:30
AFTERNOON_SLOTS = (781, 900)  # 13:01 .. 15:00
BARS_PER_DAY = 240

# bars_1m_raw holds stocks only. dim_instrument also carries indices (SH000300),
# ETFs and funds, which sort *before* the stocks alphabetically and have no
# minute bars at all -- so an unfiltered alphabetical universe silently selects
# symbols with zero data.
A_SHARE_RE = re.compile(r"^(SH6[08]\d{4}|SZ(?:00|30)\d{4}|BJ\d{6})$")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def connect(memory_limit: str, threads: int, temp_dir: str, path: str | None = None):
    con = duckdb.connect(path, read_only=bool(path)) if path else duckdb.connect(":memory:")
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET threads={int(threads)}")
    con.execute(f"SET temp_directory='{temp_dir}'")
    con.execute("SET preserve_insertion_order=false")
    return con


def write_bin(path: Path, start_index: int, values: np.ndarray) -> None:
    """Write a qlib feature bin: little-endian float32 [start_index, v0, v1, ...].

    The dtype must be pinned. Concatenating a Python int with a float32 array
    upcasts the buffer to float64 and qlib then reads every other word.
    """
    buf = np.empty(len(values) + 1, dtype="<f4")
    buf[0] = float(start_index)
    buf[1:] = values
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fp:
        buf.tofile(fp)


def merge_into_bin(path: Path, start_index: int, values: np.ndarray) -> None:
    """Overlay ``values`` at ``start_index`` onto an existing bin (or create it)."""
    if path.exists():
        old = np.fromfile(path, dtype="<f4")
        old_start, old_vals = int(old[0]), old[1:]
    else:
        old_start, old_vals = start_index, np.empty(0, dtype="<f4")

    new_start = min(old_start, start_index)
    new_end = max(old_start + len(old_vals), start_index + len(values))
    merged = np.full(new_end - new_start, np.nan, dtype="float64")
    if len(old_vals):
        merged[old_start - new_start : old_start - new_start + len(old_vals)] = old_vals
    merged[start_index - new_start : start_index - new_start + len(values)] = values
    write_bin(path, new_start, merged)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--warehouse", default=WAREHOUSE_DEFAULT)
    p.add_argument("--dim-instrument", default=None)
    p.add_argument("--factor-db", default=FACTOR_DB_DEFAULT)
    p.add_argument("--out", required=True)
    p.add_argument("--start", required=True, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--end", required=True, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--exchanges", default="SH,SZ")
    p.add_argument("--symbols", default=None, help="comma-separated qlib symbols; default = all in range")
    p.add_argument("--limit", type=int, default=None, help="cap the total number of symbols (global)")
    p.add_argument("--include-non-stock", action="store_true",
                   help="keep indices/ETFs/funds too (bars_1m_raw only stores stocks, "
                        "so these normally have no data)")
    p.add_argument("--fields", default="open,high,low,close,volume,amount,vwap")
    p.add_argument("--adjust", action="store_true",
                   help="NOT IMPLEMENTED YET; the tool exits immediately if set")
    p.add_argument("--memory-limit", default="2GB")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--temp-dir", default="/tmp/ddb_scratch")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.adjust:
        raise SystemExit(
            "--adjust is not implemented yet. The daily factor lives in the catalog / "
            "qlib-daily DB and applying it rewrites every price bin, so it belongs in a "
            "separate pass after the raw export has been verified. Export raw for now."
        )

    args.exchanges = [e.strip().upper() for e in args.exchanges.split(",") if e.strip()]
    args.fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    bad = set(args.fields) - set(MEANINGFUL_FIELDS)
    if bad:
        raise SystemExit(f"unknown fields: {sorted(bad)}; allowed: {MEANINGFUL_FIELDS}")

    start_i = int(pd.Timestamp(args.start).strftime("%Y%m%d"))
    end_i = int(pd.Timestamp(args.end).strftime("%Y%m%d"))
    bars = Path(args.warehouse) / "silver" / "bars_1m_raw"
    dim_path = args.dim_instrument or DIM_INSTRUMENT_DEFAULT.format(warehouse=args.warehouse)

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"{out} exists and is not empty; pass --overwrite")
        shutil.rmtree(out)
    for sub in ("calendars", "instruments", "features"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    log(f"warehouse : {args.warehouse}")
    log(f"range     : {args.start} .. {args.end}  exchanges={args.exchanges}")
    log(f"fields    : {args.fields}  adjust={args.adjust}")

    # ---- universe ---------------------------------------------------------
    con = connect(args.memory_limit, args.threads, args.temp_dir)
    try:
        inst = con.execute(
            f"SELECT symbol_id, qlib_symbol, exchange FROM read_parquet('{dim_path}') "
            f"WHERE qlib_symbol IS NOT NULL"
        ).df()
        inst = inst[inst["exchange"].isin(args.exchanges)]
        if not args.include_non_stock:
            inst = inst[inst["qlib_symbol"].str.match(A_SHARE_RE)]
        if args.symbols:
            keep = {s.strip().upper() for s in args.symbols.split(",")}
            inst = inst[inst["qlib_symbol"].isin(keep)]
        else:
            inst = inst.sort_values("qlib_symbol")
            if args.limit:
                inst = inst.head(args.limit)
        if inst.empty:
            raise SystemExit("no instruments selected")
        log(f"universe  : {len(inst)} symbols")

        # ---- calendar -----------------------------------------------------
        ex_sql = ",".join("'" + e + "'" for e in args.exchanges)
        days = [r[0] for r in con.execute(
            f"SELECT DISTINCT trade_date FROM read_parquet('{bars}/year=*/month=*/"
            f"exchange=*/bucket=*/*.parquet', hive_partitioning=true) "
            f"WHERE exchange IN ({ex_sql}) AND trade_date BETWEEN {start_i} AND {end_i} "
            f"ORDER BY trade_date"
        ).fetchall()]
        if not days:
            raise SystemExit("no trading days in range")

        # ---- calendar files ----------------------------------------------
        # 09:30..11:29 and 13:00..14:59 == warehouse slots 571..690 / 781..900
        # minus one minute (the slot is the bar's closing instant).
        mins = np.concatenate([
            np.arange(MORNING_SLOTS[0], MORNING_SLOTS[1] + 1),
            np.arange(AFTERNOON_SLOTS[0], AFTERNOON_SLOTS[1] + 1),
        ]) - SLOT_OFFSET_MINUTES
        day_ts = pd.to_datetime([str(d) for d in days], format="%Y%m%d")
        stamps = (
            day_ts.values.repeat(len(mins))
            + pd.to_timedelta(np.tile(mins // 60, len(days)), unit="h").values
            + pd.to_timedelta(np.tile(mins % 60, len(days)), unit="m").values
        )
        (out / "calendars" / "1min.txt").write_text(
            "\n".join(pd.Timestamp(t).strftime("%Y-%m-%d %H:%M:%S") for t in stamps) + "\n"
        )
        (out / "calendars" / "day.txt").write_text(
            "\n".join(day_ts.strftime("%Y-%m-%d")) + "\n"
        )
        log(f"calendar  : {len(stamps)} bars over {len(days)} trading days")

        # ---- stream partitions -------------------------------------------
        # bucket == symbol_id % 8, so a symbol lives in exactly one
        # (exchange, bucket). We accumulate one (exchange, bucket, year) group
        # at a time and merge it into the bins.
        groups: dict[tuple[str, int], dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
        for year_dir in sorted(bars.glob("year=*")):
            year = int(year_dir.name.split("=")[1])
            for month_dir in sorted(year_dir.glob("month=*")):
                month = int(month_dir.name.split("=")[1])
                for exch_dir in sorted(month_dir.glob("exchange=*")):
                    exch = exch_dir.name.split("=")[1]
                    if exch not in args.exchanges:
                        continue
                    for bucket_dir in sorted(exch_dir.glob("bucket=*")):
                        groups[(exch, int(bucket_dir.name.split("=")[1]))][year].append(month)

        slot_min = np.concatenate([
            np.arange(MORNING_SLOTS[0], MORNING_SLOTS[1] + 1),
            np.arange(AFTERNOON_SLOTS[0], AFTERNOON_SLOTS[1] + 1),
        ])
        cal_len = len(days) * BARS_PER_DAY
        want_ids = set(inst["symbol_id"].astype("int64"))
        id2sym = dict(zip(inst["symbol_id"].astype("int64"), inst["qlib_symbol"]))
        cal_pos = {d: i * BARS_PER_DAY for i, d in enumerate(days)}

        t0 = time.time()
        total_rows = 0
        written: dict[str, dict] = defaultdict(lambda: {"first": None, "last": None, "bars": 0})
        log(f"partitions: {len(groups)} (exchange,bucket) groups, "
            f"{sum(len(ms) for y in groups.values() for ms in y.values())} (group,year,month) units")

        for (exch, bucket), years in sorted(groups.items()):
            for year, months in sorted(years.items()):
                if year < int(args.start[:4]) or year > int(args.end[:4]):
                    continue
                acc: dict[str, dict[str, np.ndarray]] = {}
                for month in sorted(months):
                    glob = str(bars / f"year={year}" / f"month={month:02d}"
                               / f"exchange={exch}" / f"bucket={bucket:02d}" / "*.parquet")
                    df = con.execute(
                        f"SELECT symbol_id, trade_date, minute_slot, open_i, high_i, low_i, "
                        f"close_i, volume, amount_i FROM read_parquet('{glob}') "
                        f"WHERE trade_date BETWEEN {start_i} AND {end_i}"
                    ).df()
                    if df.empty:
                        continue
                    df = df[df["symbol_id"].astype("int64").isin(want_ids)]
                    if df.empty:
                        continue
                    total_rows += len(df)
                    df = df.reset_index(drop=True)

                    pos = (
                        df["trade_date"].map(cal_pos).to_numpy()
                        + np.searchsorted(slot_min, df["minute_slot"].to_numpy())
                    )
                    vol = df["volume"].to_numpy().astype("float64")
                    amt = df["amount_i"].to_numpy().astype("float64")
                    vals = {
                        "open": df["open_i"].to_numpy() / 1e4,
                        "high": df["high_i"].to_numpy() / 1e4,
                        "low": df["low_i"].to_numpy() / 1e4,
                        "close": df["close_i"].to_numpy() / 1e4,
                        "volume": vol,
                        "amount": amt / 1000.0,
                        "vwap": np.where(vol > 0, amt / np.maximum(vol * 100.0, 1e-9), np.nan),
                        "factor": np.full(len(df), np.nan),
                    }
                    syms = df["symbol_id"].astype("int64").map(id2sym).to_numpy()
                    for sym in pd.unique(syms):
                        m = syms == sym
                        if sym not in acc:
                            acc[sym] = {
                                f: np.full(cal_len, np.nan, dtype="float64") for f in args.fields
                            }
                        p_ = pos[m]
                        for f in args.fields:
                            acc[sym][f][p_] = vals[f][m]
                        rec = written[sym]
                        rec["first"] = int(p_.min()) if rec["first"] is None else min(rec["first"], int(p_.min()))
                        rec["last"] = int(p_.max()) if rec["last"] is None else max(rec["last"], int(p_.max()))
                        rec["bars"] += int(m.sum())

                # flush this (exchange, bucket, year)
                for sym, bufs in acc.items():
                    fdir = out / "features" / sym.lower()
                    for f in args.fields:
                        merge_into_bin(fdir / f"{f}.1min.bin", 0, bufs[f])
                if acc:
                    log(f"  {exch}/bucket={bucket:02d}/{year}: {len(acc)} symbols flushed "
                        f"(rows so far {total_rows:,}, {time.time() - t0:.0f}s)")
    finally:
        con.close()

    # ---- instruments ------------------------------------------------------
    # Spans come from observed coverage. dim_instrument.status/delist_date is
    # deliberately not trusted here (see the manifest note).
    stamps = pd.read_csv(out / "calendars" / "1min.txt", header=None)[0]
    lines = []
    for sym, rec in sorted(written.items()):
        lines.append(
            f"{sym}\t{stamps.iloc[rec['first']][:10]}\t{stamps.iloc[rec['last']][:10]}"
        )
    (out / "instruments" / "all.txt").write_text("\n".join(lines) + "\n")

    n_bins = sum(1 for _ in (out / "features").rglob("*.bin"))
    size = sum(f.stat().st_size for f in (out / "features").rglob("*.bin"))
    manifest = {
        "warehouse": args.warehouse,
        "bars_root": str(bars),
        "dim_instrument": dim_path,
        "range": [args.start, args.end],
        "exchanges": args.exchanges,
        "fields": args.fields,
        "adjust": False,
        "trading_days": len(days),
        "calendar_bars": cal_len,
        "slot_offset_minutes": SLOT_OFFSET_MINUTES,
        "bars_per_day": BARS_PER_DAY,
        "symbols": len(written),
        "minute_rows_read": int(total_rows),
        "bin_files": n_bins,
        "bin_bytes": size,
        "elapsed_seconds": round(time.time() - t0, 1),
        "note": (
            "instruments spans are derived from observed data coverage, not from "
            "dim_instrument.status/delist_date; the only dim_instrument snapshot "
            "reachable without the catalog is a repair intermediate state"
        ),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log("done")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
