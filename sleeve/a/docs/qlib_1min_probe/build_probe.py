"""Build a tiny qlib-native 1min dataset from the warehouse parquet (READ-ONLY).

Probe goals
-----------
1. Confirm we can read 1-minute bars out of ``silver/bars_1m_raw`` parquet with a
   hard DuckDB memory cap and Hive partition pruning (no ``catalog.duckdb``).
2. Confirm the ``minute_slot`` -> qlib minute-label mapping (slot_minutes - 1).
3. Materialise qlib's native on-disk layout (calendars/1min.txt, instruments/all.txt,
   features/<sym>/<field>.1min.bin) so the stock expression engine can be exercised
   without writing any new qlib code.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

WAREHOUSE = Path("/Volumes/lexar_4t/code/data/warehouse")
BARS_1M = WAREHOUSE / "silver" / "bars_1m_raw"
DIM_INSTRUMENT = (
    WAREHOUSE
    / "checkpoints"
    / "instrument-clock-repair-20260914.eZTGRK"
    / "dim_instrument_before.parquet"
)
OUT = Path("/tmp/qlib1m_probe/cn_data_1min")

SYMBOLS = ["SH600519", "SZ000001"]
START_DATE = 20260901
END_DATE = 20260916

PRICE_COLS = ["open", "high", "low", "close", "vwap"]
# qlib minute calendar: 09:30..11:29 and 13:00..14:59, 240 bars/day.
# warehouse minute_slot is HH*60+MM of the bar's *closing* instant
# (571=09:31 ... 690=11:30, 781=13:01 ... 900=15:00), so qlib_label = slot - 1.
SLOT_OFFSET_MINUTES = 1


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("SET memory_limit='2GB'")
    con.execute("SET threads=2")
    con.execute("SET temp_directory='/tmp/ddb_scratch'")
    con.execute("SET preserve_insertion_order=false")
    return con


def load_instruments() -> pd.DataFrame:
    con = connect()
    df = con.execute(
        f"SELECT symbol_id, symbol, qlib_symbol, exchange, list_date, delist_date, status "
        f"FROM read_parquet('{DIM_INSTRUMENT}') WHERE qlib_symbol IN "
        f"({','.join(repr(s) for s in SYMBOLS)})"
    ).df()
    con.close()
    return df


def load_minutes(symbol_ids: list[int]) -> pd.DataFrame:
    """Read only the partitions that can hold these symbols (bucket = symbol_id % 8)."""
    buckets = sorted({sid % 8 for sid in symbol_ids})
    globs = [
        str(BARS_1M / "year=2026" / "month=09" / "exchange=*" / f"bucket={b:02d}" / "*.parquet")
        for b in buckets
    ]
    con = connect()
    df = con.execute(
        f"""
        SELECT symbol_id, trade_date, minute_slot, open_i, high_i, low_i, close_i,
               volume, amount_i
        FROM read_parquet({globs!r})
        WHERE symbol_id IN ({','.join(str(int(s)) for s in symbol_ids)})
          AND trade_date BETWEEN {START_DATE} AND {END_DATE}
        ORDER BY symbol_id, trade_date, minute_slot
        """
    ).df()
    con.close()
    return df


def qlib_minute_labels(trade_dates, slots) -> pd.DatetimeIndex:
    minutes = slots.to_numpy(dtype="int32") - SLOT_OFFSET_MINUTES
    dates = pd.to_datetime(trade_dates.astype(str), format="%Y%m%d")
    return pd.DatetimeIndex(
        dates
        + pd.to_timedelta(minutes // 60, unit="h")
        + pd.to_timedelta(minutes % 60, unit="m")
    )


def write_bin(path: Path, start_index: int, values: np.ndarray) -> None:
    """qlib feature bins are little-endian float32: [start_index, v0, v1, ...].

    The dtype must be pinned explicitly -- concatenating a Python int with a
    float32 array silently upcasts the whole buffer to float64 and qlib then
    reads every other 4-byte word as garbage.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = np.empty(len(values) + 1, dtype="<f4")
    payload[0] = float(start_index)
    payload[1:] = values
    with path.open("wb") as fp:
        payload.tofile(fp)


def main() -> int:
    inst = load_instruments()
    print("instruments:\n", inst.to_string(index=False))
    if len(inst) != len(SYMBOLS):
        print("!! did not resolve every requested symbol", file=sys.stderr)
        return 1

    frames = load_minutes(inst["symbol_id"].tolist())
    print(f"\nminute rows loaded: {len(frames)}")
    print(frames.head(3).to_string(index=False))

    # ---- unit reconciliation: pin down what _i / volume / amount_i mean ----
    print("\n--- unit reconciliation (SH600519, 2026-09-16) ---")
    day = frames[
        (frames["symbol_id"] == 1600519) & (frames["trade_date"] == 20260916)
    ].sort_values("minute_slot")
    px = day["close_i"].to_numpy() / 1e4
    vol = day["volume"].to_numpy().astype("float64")
    amt = day["amount_i"].to_numpy().astype("float64")
    print(f"  bars                  : {len(day)}")
    print(f"  close_i/1e4 (CNY)     : first={px[0]:.2f} (slot {day['minute_slot'].iloc[0]}, "
          f"= daily open_raw 1273.93)  last={px[-1]:.2f} (slot {day['minute_slot'].iloc[-1]}, "
          f"= daily close_raw 1258.00)")
    print(f"  sum(volume)           : {vol.sum():.0f}  (= daily volume 26235)")
    print(f"  sum(amount_i)         : {amt.sum():.0f}  (= daily amount 3307926407)")
    print(f"  implied turnover      : sum(amount_i) = {amt.sum():.0f} CNY "
          f"(600519 turns over ~1e9-1e10 CNY/day -> amount_i is YUAN, not cents)")
    print(f"  implied vwap          : sum(amount_i)/(sum(volume)*100) = "
          f"{amt.sum() / (vol.sum() * 100):.2f} CNY/share")
    print(f"  qlib daily amount     : 3307926.407  -> amount_i/1000 (thousand CNY)")
    print(f"  qlib daily volume     : 104999.03    -> raw_volume/factor "
          f"({vol.sum():.0f}/0.24985944 = {vol.sum() / 0.2498594431434518:.2f})")
    print(f"  qlib daily close      : 314.3232     -> raw_close*factor "
          f"(1258.00*0.24985944 = {1258.0 * 0.2498594431434518:.4f})")

    # ---- qlib calendar: every distinct bar timestamp, ascending ----
    labels = qlib_minute_labels(frames["trade_date"], frames["minute_slot"])
    frames = frames.assign(dt=labels)
    calendar = pd.DatetimeIndex(sorted(frames["dt"].unique()))
    cal_pos = {ts: i for i, ts in enumerate(calendar)}
    print(f"\ncalendar bars: {len(calendar)}  days: {calendar.normalize().nunique()}")
    print("per-day bar counts:", calendar.normalize().value_counts().to_dict())

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "calendars").mkdir(exist_ok=True)
    (OUT / "calendars" / "1min.txt").write_text(
        "\n".join(ts.strftime("%Y-%m-%d %H:%M:%S") for ts in calendar) + "\n"
    )

    # ---- instruments/all.txt : SYMBOL \t start \t end ----
    (OUT / "instruments").mkdir(exist_ok=True)
    lines = []
    for _, row in inst.iterrows():
        sub = frames[frames["symbol_id"] == row["symbol_id"]]
        lines.append(
            f"{row['qlib_symbol']}\t{sub['dt'].min():%Y-%m-%d}\t{sub['dt'].max():%Y-%m-%d}"
        )
    (OUT / "instruments" / "all.txt").write_text("\n".join(lines) + "\n")

    # ---- features/<sym>/<field>.1min.bin ----
    summary = {}
    for _, row in inst.iterrows():
        sym = row["qlib_symbol"]
        sub = frames[frames["symbol_id"] == row["symbol_id"]].sort_values("dt")
        start_index = cal_pos[sub["dt"].iloc[0]]
        # qlib needs a contiguous run from start_index; our probe window is contiguous
        expected = calendar[start_index : start_index + len(sub)]
        if not np.array_equal(expected.values, sub["dt"].values):
            print(f"!! {sym}: bars are not contiguous in the calendar", file=sys.stderr)
            return 1
        fdir = OUT / "features" / sym.lower()
        vol = sub["volume"].to_numpy().astype("float64")
        amt = sub["amount_i"].to_numpy().astype("float64")
        fields = {
            "open": sub["open_i"].to_numpy() / 1e4,
            "high": sub["high_i"].to_numpy() / 1e4,
            "low": sub["low_i"].to_numpy() / 1e4,
            "close": sub["close_i"].to_numpy() / 1e4,
            "volume": vol,
            # amount_i is in CNY; qlib's daily convention stores amount/1000
            "amount": amt / 1000.0,
            # volume is in hands (100 shares); vwap_raw = CNY / shares
            "vwap": np.where(vol > 0, amt / np.maximum(vol * 100.0, 1e-9), np.nan),
        }
        for name, values in fields.items():
            write_bin(fdir / f"{name}.1min.bin", start_index, np.asarray(values, dtype="float64"))
        summary[sym] = {
            "start_index": int(start_index),
            "bars": int(len(sub)),
            "first_dt": str(sub["dt"].iloc[0]),
            "last_dt": str(sub["dt"].iloc[-1]),
        }

    print("\nwritten:", json.dumps(summary, indent=2))
    frames.to_parquet("/tmp/qlib1m_probe/raw_minutes.parquet")
    print("\nOK ->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
