"""DuckDB data access for the sleeve/a strategy.

The database is opened exclusively through Qlib's local DuckDB backend so that
the configured memory limit (2GB by default, SLEEVE_DUCKDB_MEMORY_LIMIT to override) is set for every connection.  Bulk reads are done
with explicit SQL (still through that connection) because the built-in
``FileFeatureStorage`` equivalent would issue one query per symbol and field,
which is too slow for a 6,000-symbol A-share panel.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .bootstrap import ensure_local_qlib
from .config import Config

# A-share stock codes.  Funds, bonds and B shares share the same Qlib symbol
# namespace (SH5xxxxx/SH9xxxxx, SZ1xxxxx, ...) and must not enter the universe.
A_SHARE_REGEX = r"^(SH6[08][0-9]{4}|SZ(?:00|30)[0-9]{4}|BJ[0-9]{6})$"
A_SHARE_RE = re.compile(A_SHARE_REGEX)

# vwap is present and populated in this DuckDB (verified by scripts/audit_schema.py
# and scripts/audit_vwap_ic.py), so it is loaded and used as a feature source
# rather than dropped.
FIELD_NAMES = ("open", "high", "low", "close", "volume", "amount", "factor", "vwap")

# Columns needed for the strategy.  ``vwap`` is intentionally not required:
# the design notes VWAP may be absent and Alpha158's VWAP features must be
# dropped in that case.
REQUIRED_FIELDS = ("high", "low", "close", "volume", "amount", "factor")


class DataError(RuntimeError):
    pass


def parse_date_int(value: Any) -> int:
    if isinstance(value, (int, np.integer)):
        text = str(int(value))
        if len(text) == 8:
            return int(text)
    return int(pd.Timestamp(value).strftime("%Y%m%d"))


def date_ints_to_index(values: Iterable[Any]) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime([str(int(v)) for v in values]))


def _memory_setting_to_bytes(text: str) -> float:
    parts = str(text).strip().split()
    if not parts:
        return float("inf")
    value = float(parts[0])
    unit = parts[1].upper() if len(parts) > 1 else "B"
    factor = {
        "B": 1,
        "KB": 1024,
        "KIB": 1024,
        "MB": 1024**2,
        "MIB": 1024**2,
        "GB": 1024**3,
        "GIB": 1024**3,
        "TB": 1024**4,
        "TIB": 1024**4,
    }.get(unit)
    if factor is None:
        return float("inf")
    return value * factor


@dataclass
class PanelData:
    """In-memory (float32) A-share daily panel.

    All matrices have shape ``(n_dates, n_symbols)`` and use the adjusted Qlib
    price/volume conventions.  ``raw_close`` is recovered as
    ``close / factor`` (the design document's stated identity), and raw volume
    in shares is ``volume * factor * 100`` (Qlib volume is Tushare hands
    divided by factor).
    """

    dates: pd.DatetimeIndex
    date_ints: np.ndarray
    trade_indices: np.ndarray
    symbols: List[str]
    fields: Dict[str, np.ndarray]
    valid: np.ndarray
    instrument_meta: pd.DataFrame
    db_path: str = ""
    source_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_dates(self) -> int:
        return len(self.dates)

    @property
    def n_symbols(self) -> int:
        return len(self.symbols)

    def field(self, name: str) -> np.ndarray:
        if name not in self.fields:
            raise KeyError(f"Unknown panel field {name!r}; available={sorted(self.fields)}")
        return self.fields[name]

    def df(self, name: str) -> pd.DataFrame:
        return pd.DataFrame(self.field(name), index=self.dates, columns=self.symbols)

    def raw_close(self) -> np.ndarray:
        close = self.field("close")
        factor = self.field("factor")
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = close / factor
        raw[~np.isfinite(raw)] = np.nan
        return raw.astype(np.float32)

    def raw_volume_shares(self) -> np.ndarray:
        volume = self.field("volume")
        factor = self.field("factor")
        raw = volume * factor * 100.0
        raw[~np.isfinite(raw)] = np.nan
        return raw.astype(np.float32)

    def amount_yuan(self) -> np.ndarray:
        # The corrected DuckDB contract stores amount in thousand yuan
        # (Tushare ``amount_k``); it must not be multiplied by the factor.
        amount = self.field("amount")
        return (amount * 1000.0).astype(np.float32)

    def mark_price(self) -> np.ndarray:
        """Raw close forward-filled over suspensions, for NAV marking."""

        raw = self.raw_close()
        df = pd.DataFrame(raw, index=self.dates).ffill()
        return df.to_numpy(dtype=np.float32)


def _panel_cache_key(cfg: Config, stat: os.stat_result) -> str:
    payload = {
        "db": str(Path(cfg.data.db_path).resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "start": cfg.data.start_date,
        "end": cfg.data.end_date,
        "regex": A_SHARE_REGEX,
        "version": 2,  # 2: adds the vwap field
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]


class DuckDBPanelLoader:
    def __init__(self, cfg: Config, use_cache: bool = True):
        self.cfg = cfg
        self.use_cache = use_cache
        self._con = None

    # ------------------------------------------------------------------
    # connection / schema
    # ------------------------------------------------------------------
    def _connection(self):
        if self._con is not None:
            return self._con
        ensure_local_qlib(self.cfg.data.qlib_home)
        from qlib.data.storage.duckdb_storage import (  # noqa: WPS433
            DEFAULT_MEMORY_LIMIT,
            DEFAULT_TEMP_DIRECTORY,
            get_duckdb_connection,
        )

        db_path = Path(self.cfg.data.db_path).expanduser()
        if not db_path.exists():
            raise DataError(f"DuckDB provider not found: {db_path}")
        # Use Qlib's default temp directory (and therefore its exact
        # connection configuration).  DuckDB refuses two connections to the
        # same file with different settings, and Qlib itself may already hold
        # a default connection after ``qlib.init``.
        temp_dir = Path(DEFAULT_TEMP_DIRECTORY).expanduser()
        temp_dir.mkdir(parents=True, exist_ok=True)
        memory_limit = self.cfg.data.memory_limit or DEFAULT_MEMORY_LIMIT
        con = get_duckdb_connection(
            db_path,
            read_only=True,
            memory_limit=memory_limit,
            temp_directory=str(temp_dir),
            use_cache=False,
        )
        setting = con.execute("SELECT current_setting('memory_limit')").fetchone()[0]
        limit_bytes = _memory_setting_to_bytes(setting)
        if limit_bytes > 2.05 * 1024**3:
            con.close()
            raise DataError(f"DuckDB memory limit was not applied as required: {setting!r}")
        con.execute("SET preserve_insertion_order=false")
        self._con = con
        return con

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None

    def __enter__(self) -> "DuckDBPanelLoader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # schema / audit
    # ------------------------------------------------------------------
    def audit(self) -> Dict[str, Any]:
        con = self._connection()
        tables = {
            row[0]
            for row in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
            ).fetchall()
        }
        required = {"qlib_calendar", "qlib_instruments", "qlib_daily_features", "v_qlib_daily_features_runtime"}
        missing = sorted(required - tables)
        if missing:
            raise DataError(f"DuckDB is missing required objects: {missing}")
        info: Dict[str, Any] = {
            "db_path": self.cfg.data.db_path,
            "memory_limit": con.execute("SELECT current_setting('memory_limit')").fetchone()[0],
            "tables": sorted(tables),
        }
        cal = con.execute(
            "SELECT COUNT(*) AS rows, MIN(trade_date) AS start, MAX(trade_date) AS end "
            "FROM v_qlib_calendar_latest"
        ).fetchone()
        info.update({"calendar_rows": int(cal[0]), "calendar_start": int(cal[1]), "calendar_end": int(cal[2])})
        inst = con.execute(
            "SELECT COUNT(*) AS rows, COUNT(DISTINCT qlib_symbol) AS symbols FROM v_qlib_instruments_latest"
        ).fetchone()
        info.update({"instrument_rows": int(inst[0]), "instrument_symbols": int(inst[1])})
        dup = con.execute(
            """
            SELECT COUNT(*) FROM (
              SELECT qlib_symbol, trade_date, COUNT(*) AS c
              FROM v_qlib_daily_features_runtime
              GROUP BY qlib_symbol, trade_date
              HAVING COUNT(*) > 1
            ) AS d
            """
        ).fetchone()[0]
        info["runtime_duplicate_keys"] = int(dup)
        if dup:
            raise DataError(f"Runtime feature view returns duplicate (symbol,date) keys: {dup}")
        return info

    # ------------------------------------------------------------------
    # metadata
    # ------------------------------------------------------------------
    def load_instruments(self) -> pd.DataFrame:
        con = self._connection()
        rows = con.execute(
            """
            SELECT qlib_symbol, start_date, end_date
            FROM v_qlib_instruments_latest
            WHERE regexp_matches(qlib_symbol, ?)
            ORDER BY qlib_symbol
            """,
            [A_SHARE_REGEX],
        ).fetchall()
        if not rows:
            raise DataError("No A-share instruments matched the universe regex")
        frame = pd.DataFrame(rows, columns=["symbol", "start_date", "end_date"]).set_index("symbol", drop=False)
        frame["start_date"] = frame["start_date"].astype(int)
        frame["end_date"] = frame["end_date"].astype(int)
        frame["board"] = [infer_board(s) for s in frame["symbol"]]
        return frame

    def load_calendar(self) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
        con = self._connection()
        rows = con.execute(
            """
            SELECT trade_index, trade_date, trade_date_text
            FROM v_qlib_calendar_latest
            WHERE is_open AND trade_date BETWEEN ? AND ?
            ORDER BY trade_index
            """,
            [parse_date_int(self.cfg.data.start_date), parse_date_int(self.cfg.data.end_date)],
        ).fetchall()
        if not rows:
            raise DataError("Calendar query returned no trading days")
        indices = np.asarray([int(r[0]) for r in rows], dtype=np.int64)
        date_ints = np.asarray([int(r[1]) for r in rows], dtype=np.int64)
        dates = pd.DatetimeIndex(pd.to_datetime([str(r[2]) for r in rows]))
        if np.any(np.diff(indices) != 1):
            raise DataError("Trading calendar is not contiguous in trade_index within the requested range")
        if np.any(np.diff(date_ints) <= 0):
            raise DataError("Trading dates are not strictly increasing")
        return dates, date_ints, indices

    # ------------------------------------------------------------------
    # panel loading
    # ------------------------------------------------------------------
    def load(self, refresh_cache: bool = False) -> PanelData:
        db_path = Path(self.cfg.data.db_path).expanduser()
        stat = db_path.stat()
        cache_dir = Path(self.cfg.data.cache_dir).expanduser() if self.cfg.data.cache_dir else None
        key = _panel_cache_key(self.cfg, stat)
        if self.use_cache and cache_dir is not None and not refresh_cache:
            cached = self._load_cache(cache_dir, key)
            if cached is not None:
                cached.source_info["cache"] = "hit"
                return cached

        with self as loader:
            audit = loader.audit()
            instruments = loader.load_instruments()
            dates, date_ints, trade_indices = loader.load_calendar()
            fields, valid = loader._load_fields(instruments, dates, date_ints, trade_indices)
            panel = PanelData(
                dates=dates,
                date_ints=date_ints,
                trade_indices=trade_indices,
                symbols=instruments["symbol"].tolist(),
                fields=fields,
                valid=valid,
                instrument_meta=instruments,
                db_path=str(db_path),
                source_info={"audit": audit, "cache": "miss"},
            )
        if self.use_cache and cache_dir is not None:
            self._save_cache(panel, cache_dir, key)
        return panel

    def _load_fields(
        self,
        instruments: pd.DataFrame,
        dates: pd.DatetimeIndex,
        date_ints: np.ndarray,
        trade_indices: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        con = self._connection()
        symbols = instruments["symbol"].tolist()
        n_dates, n_symbols = len(dates), len(symbols)
        fields = {name: np.full((n_dates, n_symbols), np.nan, dtype=np.float32) for name in FIELD_NAMES}

        # Registering the universe as a temporary table keeps the feature scan
        # bounded to ~6k symbols and avoids expensive regex evaluation per row.
        con.execute("DROP TABLE IF EXISTS _sleeve_a_univ")
        con.execute("CREATE TEMP TABLE _sleeve_a_univ(sid INTEGER, qlib_symbol VARCHAR)")
        con.executemany("INSERT INTO _sleeve_a_univ VALUES (?, ?)", list(enumerate(symbols)))
        try:
            sql = """
                WITH ranked AS (
                    SELECT
                        f.trade_index, u.sid,
                        f.open, f.high, f.low, f.close,
                        f.volume, f.amount, f.factor, f.vwap,
                        row_number() OVER (
                            PARTITION BY f.qlib_symbol, f.trade_date
                            ORDER BY CASE f.row_kind
                                WHEN 'final' THEN 3
                                WHEN 'intraday' THEN 2
                                WHEN 'historical' THEN 1
                                ELSE 0 END DESC,
                                f.batch_id DESC
                        ) AS rn
                    FROM qlib_daily_features AS f
                    JOIN _sleeve_a_univ AS u ON f.qlib_symbol = u.qlib_symbol
                    WHERE f.row_status = 'active'
                      AND f.trade_index >= ?
                      AND f.trade_index <= ?
                )
                SELECT trade_index, sid, open, high, low, close, volume, amount, factor, vwap
                FROM ranked WHERE rn = 1
                ORDER BY trade_index, sid
            """
            chunk = 100
            first_index = int(trade_indices[0])
            for offset in range(0, n_dates, chunk):
                lo = int(trade_indices[offset])
                hi = int(trade_indices[min(offset + chunk, n_dates) - 1])
                result = con.execute(sql, [lo, hi]).fetchnumpy()
                if not result or len(result.get("trade_index", [])) == 0:
                    continue
                tpos = result["trade_index"].astype(np.int64) - first_index
                jpos = result["sid"].astype(np.int64)
                for name in FIELD_NAMES:
                    values = result[name]
                    arr = np.asarray(values, dtype=np.float64)
                    fields[name][tpos, jpos] = arr.astype(np.float32)
        finally:
            con.execute("DROP TABLE IF EXISTS _sleeve_a_univ")

        valid = np.isfinite(fields["close"]) & (fields["close"] > 0)
        valid &= np.isfinite(fields["high"]) & (fields["high"] > 0)
        valid &= np.isfinite(fields["low"]) & (fields["low"] > 0)
        valid &= np.isfinite(fields["factor"]) & (fields["factor"] > 0)
        valid &= np.isfinite(fields["volume"]) & (fields["volume"] > 0)
        valid &= fields["high"] >= fields["low"]
        valid &= fields["high"] >= fields["close"]
        valid &= fields["low"] <= fields["close"]
        if self.cfg.universe.require_positive_volume:
            valid &= fields["volume"] > 0

        # Instrument life-cycle mask.
        life = np.zeros((n_dates, n_symbols), dtype=bool)
        for j, symbol in enumerate(symbols):
            start = int(instruments.iloc[j]["start_date"])
            end = int(instruments.iloc[j]["end_date"])
            left = int(np.searchsorted(date_ints, start, side="left"))
            right = int(np.searchsorted(date_ints, end, side="right"))
            if right > left:
                life[left:right, j] = True
        valid &= life
        return fields, valid

    # ------------------------------------------------------------------
    # cache
    # ------------------------------------------------------------------
    def _load_cache(self, cache_dir: Path, key: str) -> Optional[PanelData]:
        meta_path = cache_dir / f"panel_{key}.json"
        if not meta_path.exists():
            return None
        try:
            with meta_path.open("r", encoding="utf-8") as fh:
                meta = json.load(fh)
            fields = {}
            for name in meta["fields"]:
                path = cache_dir / f"panel_{key}_{name}.npy"
                fields[name] = np.load(path, mmap_mode="r")
            valid = np.load(cache_dir / f"panel_{key}_valid.npy", mmap_mode="r")
            dates = pd.DatetimeIndex(pd.to_datetime(meta["dates"]))
            instruments = pd.DataFrame(meta["instruments"]).set_index("symbol", drop=False)
            return PanelData(
                dates=dates,
                date_ints=np.asarray(meta["date_ints"], dtype=np.int64),
                trade_indices=np.asarray(meta["trade_indices"], dtype=np.int64),
                symbols=list(meta["symbols"]),
                fields=fields,
                valid=valid,
                instrument_meta=instruments,
                db_path=meta["db_path"],
                source_info={"cache": "hit", "audit": meta.get("audit", {})},
            )
        except Exception:
            return None

    def _save_cache(self, panel: PanelData, cache_dir: Path, key: str) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir = cache_dir / f".tmp_panel_{key}"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)
        try:
            for name, arr in panel.fields.items():
                np.save(tmp_dir / f"panel_{key}_{name}.npy", arr)
            np.save(tmp_dir / f"panel_{key}_valid.npy", panel.valid)
            meta = {
                "key": key,
                "db_path": panel.db_path,
                "dates": [d.strftime("%Y-%m-%d") for d in panel.dates],
                "date_ints": panel.date_ints.tolist(),
                "trade_indices": panel.trade_indices.tolist(),
                "symbols": panel.symbols,
                "fields": list(panel.fields),
                "instruments": panel.instrument_meta.reset_index(drop=True).to_dict(orient="records"),
                "audit": panel.source_info.get("audit", {}),
            }
            with (tmp_dir / f"panel_{key}.json").open("w", encoding="utf-8") as fh:
                json.dump(meta, fh, ensure_ascii=False)
            # Atomic-ish move: remove any stale files, then move new ones in.
            for path in tmp_dir.iterdir():
                target = cache_dir / path.name
                if target.exists():
                    target.unlink()
                path.replace(target)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def infer_board(symbol: str) -> str:
    if symbol.startswith("SH60"):
        return "sh_main"
    if symbol.startswith("SH68"):
        return "star"
    if symbol.startswith("SZ30"):
        return "chinext"
    if symbol.startswith("SZ00"):
        return "sz_main"
    if symbol.startswith("BJ"):
        return "bse"
    return "unknown"
