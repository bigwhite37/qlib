# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""DuckDB-backed storage for Qlib.

This module implements Qlib's :class:`~qlib.data.storage.storage.CalendarStorage`,
:class:`~qlib.data.storage.storage.InstrumentStorage` and
:class:`~qlib.data.storage.storage.FeatureStorage` interfaces on top of the
DuckDB database layout used by the local data pipeline.

The intended database schema is
- ``qlib_calendar``: one row per trading day (``trade_index`` / ``trade_date``)
- ``v_qlib_instruments_latest``: active instruments
- ``qlib_daily_features``: long table of daily features

Each DuckDB connection created by this module is configured with a strict
memory limit (``2GB`` by default).  A temporary directory is also configured so
that a read-only database does not try to spill next to the database file.

Basic usage is intentionally the same as Qlib's native file storage::

    import qlib
    from qlib.constant import REG_CN

    qlib.init(provider_uri="/path/to/qlib_data.duckdb", region=REG_CN)

The provider backend is auto-selected from the ``.duckdb`` suffix (or it can be
configured explicitly with ``DuckDBFeatureStorage`` etc. in the provider
``backend`` config).
"""

from __future__ import annotations

import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union

import duckdb
import numpy as np
import pandas as pd

from qlib.config import C
from qlib.data.storage.storage import CalendarStorage, FeatureStorage, InstrumentStorage, CalVT, InstKT, InstVT
from qlib.log import get_module_logger
from qlib.utils.resam import resam_calendar
from qlib.utils.time import Freq

logger = get_module_logger("duckdb_storage")

# Suffixes which identify a DuckDB database file.
DUCKDB_SUFFIXES = (".duckdb", ".ddb", ".db")

# The user-facing memory limit required by this project.
# Every DuckDB connection opened through this module is capped.  The default is
# 2 GB and can be overridden with SLEEVE_DUCKDB_MEMORY_LIMIT.
DEFAULT_MEMORY_LIMIT = os.environ.get("SLEEVE_DUCKDB_MEMORY_LIMIT", "2GB")

# A fallback temporary directory.  It can be overridden with the
# ``temp_directory`` keyword.  We use a process-wide sub-directory so that
# connections in the same process share DuckDB's temp files naturally.
DEFAULT_TEMP_DIRECTORY = str(Path(tempfile.gettempdir()) / "qlib_duckdb")

# Thread-local connection cache.  Every (path, process, settings) tuple gets one
# connection.  Keeping the connection alive avoids opening the database for each
# single expression read.
_CONNECTION_STATE = threading.local()

# Feature columns which can be read from ``qlib_daily_features``.  Keep this
# allow-list explicit; it prevents SQL injection through the ``field`` argument.
_FEATURE_COLUMNS = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "amount": "amount",
    "vwap": "vwap",
    "factor": "factor",
    "change": "change",
    "adjclose": "adjclose",
}

# Extra columns that *some* databases add to ``qlib_daily_features`` (the ETF PIT
# dataset exposes its decision-time and fund facts this way).  Listing them here
# makes them addressable as Qlib expressions and lets callers validate field
# names statically, but presence is still checked against the real table in
# ``_field_column`` below -- a database without these columns degrades to an
# empty series instead of emitting a query that is guaranteed to fail.
_FEATURE_COLUMNS.update(
    {
        "close_1449": "close_1449",
        "open_0930": "open_0930",
        "high_1449": "high_1449",
        "low_1449": "low_1449",
        "vwap_1449": "vwap_1449",
        "amount_1449": "amount_1449",
        "amount_last_30m": "amount_last_30m",
        "intraday_vol_1449": "intraday_vol_1449",
        "iopv": "iopv",
        "nav": "nav",
        "premium": "premium",
        "share_10k": "share_10k",
        # ETF 因子研究需要「原始价 / 参考净值」的显式拆分：折溢价必须用原始价，
        # 不能用复权价除以单位净值（见 docs/chatgpt/003_etf_factor_study.md §二.4）。
        "raw_close": "raw_close",
        # L1：与实际下单窗口一致的成交额中位数（过去 20 日）。
        "amount_1450_med20": "amount_1450_med20",
        # T3 的公告可见性：``nav_pit`` 是"截至该交易日**已公告**"的最近一笔单位净值，
        # 与 ``nav``（按净值所属日对齐）不同——后者含未公告的前视。
        "nav_pit": "nav_pit",
        "nav_ref_lag_days": "nav_ref_lag_days",
        # 满窗口校验：该 ETF 截至当日的有效日线记录数。
        "valid_history_days": "valid_history_days",
    }
)

#: Actual column set per ``(database, feature table)``.  Resolving a field would
#: otherwise cost one ``information_schema`` query per expression leaf, and Qlib
#: builds a fresh storage object for every ``(instrument, field)`` pair.
_TABLE_COLUMN_CACHE: Dict[Tuple[str, str], frozenset] = {}

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?$")

# Minute frequencies need their own tables: a 1-minute calendar carries one label
# per bar (not per day) and can therefore never be derived from ``qlib_calendar``.
# ``resam_calendar`` refuses a day -> minute resample outright, so without this
# redirection every minute query dies in ``DuckDBCalendarStorage.data``.
MINUTE_TABLES = {
    "calendar": "qlib_minute_calendar",
    "instrument": "v_qlib_instruments_latest",
    "feature": "qlib_minute_features",
}
DAILY_TABLES = {
    "calendar": "qlib_calendar",
    "instrument": "v_qlib_instruments_latest",
    "feature": "qlib_daily_features",
}


def is_minute_freq(freq) -> bool:
    """Whether ``freq`` is a minute-or-finer frequency."""

    try:
        return Freq(freq).base == Freq.NORM_FREQ_MINUTE
    except Exception:  # pragma: no cover - unknown freq strings are not minute
        return False


#: Optional table describing where minute bars physically live.  When a minute
#: provider database carries this table, ``DuckDBFeatureStorage`` reads the
#: warehouse parquet directly through DuckDB instead of materialising a Qlib
#: ``.bin`` copy of every (symbol, field).
MINUTE_SOURCE_TABLE = "qlib_minute_sources"

#: A stored expression is executed as SQL, so it must be a plain arithmetic
#: expression over warehouse columns.  The table is written by our own builder,
#: but the check keeps a hand-edited provider database from turning into an
#: arbitrary SQL channel.
_MINUTE_EXPR_RE = re.compile(r"^[A-Za-z0-9_+\-*/().,'<>=! ]+$")

# Column names a stored expression may reference.
_MINUTE_ALLOWED_COLUMNS = frozenset(
    {
        "symbol_id",
        "trade_date",
        "minute_slot",
        "open_i",
        "high_i",
        "low_i",
        "close_i",
        "volume",
        "amount_i",
    }
)


#: SQL tokens a stored expression is allowed to use besides plain arithmetic.
_MINUTE_SQL_TOKENS = frozenset(
    {"cast", "as", "double", "float", "int", "case", "when", "then", "else", "end", "null"}
)
_MINUTE_NUMBER_RE = re.compile(r"\b[0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?\b", re.IGNORECASE)


def check_minute_expression(expression: str) -> str:
    """Validate a stored minute expression before it is used as SQL.

    Only arithmetic over an allow-listed set of warehouse columns plus ``CAST``/``CASE``
    is accepted; comments and statement separators are rejected outright.
    """

    text = " ".join(str(expression).split())
    if not text or _MINUTE_EXPR_RE.match(text) is None:
        raise ValueError(f"Unsafe minute source expression: {expression!r}")
    if "--" in text or "/*" in text or ";" in text:
        raise ValueError(f"Minute source expression contains a comment or separator: {expression!r}")
    without_numbers = _MINUTE_NUMBER_RE.sub(" ", text)
    identifiers = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", without_numbers))
    unknown = {
        name for name in identifiers if name.lower() not in _MINUTE_SQL_TOKENS
    } - _MINUTE_ALLOWED_COLUMNS
    if unknown:
        raise ValueError(f"Minute source expression references unknown columns: {sorted(unknown)}")
    return text


def is_duckdb_uri(uri) -> bool:
    """Return whether ``uri`` (a path, dict of paths, etc.) points to DuckDB.

    This helper is intentionally dependency-light so it can be used from
    :class:`qlib.data.data.ProviderBackendMixin` to choose the default backend.
    """

    if uri is None:
        return False
    if isinstance(uri, Mapping):
        return any(is_duckdb_uri(v) for v in uri.values())
    try:
        return Path(str(uri)).expanduser().suffix.lower() in DUCKDB_SUFFIXES
    except (TypeError, ValueError):
        return False


def _normalise_db_path(path: Union[str, Path]) -> Path:
    return Path(str(path)).expanduser().resolve()


def _extract_duckdb_uri(provider_uri, freq: str) -> Path:
    """Extract a DuckDB path from a Qlib ``provider_uri`` value."""

    if not isinstance(provider_uri, Mapping):
        return _normalise_db_path(provider_uri)

    # provider_uri of Qlib is often a mapping from freq to path.  The
    # ``__DEFAULT_FREQ`` key is used when the user provides one string.
    default_key = getattr(C, "DEFAULT_FREQ", "__DEFAULT_FREQ")
    candidates = [str(freq), default_key]
    for key in candidates:
        if key in provider_uri:
            return _normalise_db_path(provider_uri[key])
    if len(provider_uri) == 1:
        return _normalise_db_path(next(iter(provider_uri.values())))
    raise ValueError(f"Can not find a DuckDB path for freq={freq!r} from provider_uri={provider_uri!r}")


def _prepare_connection_paths(db_path, temp_directory):
    db_path = str(_normalise_db_path(db_path))
    temp_directory = str(_normalise_db_path(temp_directory)) if temp_directory else None
    if temp_directory is not None:
        Path(temp_directory).mkdir(parents=True, exist_ok=True)
    return db_path, temp_directory


def _connection_config(memory_limit, temp_directory, max_temp_directory_size, threads):
    config = {"memory_limit": memory_limit}
    if temp_directory is not None:
        config["temp_directory"] = temp_directory
        # Bounding temporary files is a useful complement to the memory-bound
        # requirement.  If the caller doesn't set it, use the memory limit.
        config["max_temp_directory_size"] = max_temp_directory_size or memory_limit
    if threads is not None:
        config["threads"] = int(threads)
    return config


def _open_connection(
    db_path: Union[str, Path],
    read_only: bool = True,
    memory_limit: str = DEFAULT_MEMORY_LIMIT,
    temp_directory: Optional[str] = DEFAULT_TEMP_DIRECTORY,
    max_temp_directory_size: Optional[str] = None,
    threads: Optional[int] = None,
):
    db_path, temp_directory = _prepare_connection_paths(db_path, temp_directory)
    config = _connection_config(memory_limit, temp_directory, max_temp_directory_size, threads)
    connection = duckdb.connect(db_path, read_only=bool(read_only), config=config)
    try:
        # Avoiding insertion-order preservation significantly lowers the memory
        # required by larger reads and is also faster for our aggregate reads.
        connection.execute("SET preserve_insertion_order=false")
    except Exception:  # pragma: no cover - only fails on very old DuckDB
        logger.debug("Could not disable `preserve_insertion_order` on this DuckDB version")
    return connection


def _get_connection(
    db_path: Union[str, Path],
    read_only: bool = True,
    memory_limit: str = DEFAULT_MEMORY_LIMIT,
    temp_directory: Optional[str] = DEFAULT_TEMP_DIRECTORY,
    max_temp_directory_size: Optional[str] = None,
    threads: Optional[int] = None,
):
    """Get one cached DuckDB connection per process/thread/settings combination."""

    db_path, temp_directory = _prepare_connection_paths(db_path, temp_directory)
    key = (os.getpid(), db_path, bool(read_only), memory_limit, temp_directory, max_temp_directory_size, threads)
    connections = getattr(_CONNECTION_STATE, "connections", None)
    if connections is None:
        connections = {}
        _CONNECTION_STATE.connections = connections

    connection = connections.get(key)
    if connection is not None:
        return connection

    connection = _open_connection(
        db_path,
        read_only=read_only,
        memory_limit=memory_limit,
        temp_directory=temp_directory,
        max_temp_directory_size=max_temp_directory_size,
        threads=threads,
    )
    connections[key] = connection
    return connection


def get_duckdb_connection(
    db_path: Union[str, Path],
    read_only: bool = True,
    memory_limit: str = DEFAULT_MEMORY_LIMIT,
    temp_directory: Optional[str] = DEFAULT_TEMP_DIRECTORY,
    max_temp_directory_size: Optional[str] = None,
    threads: Optional[int] = None,
    use_cache: bool = False,
):
    """Return a DuckDB connection with a strict memory limit.

    By default a fresh connection is returned; the caller owns it and should
    close it when finished.  If ``use_cache=True`` the process/thread-local
    storage connection is returned instead (do not close it in that case).

    All arguments except ``db_path`` share the storage defaults, notably
    ``memory_limit='3GB'``.  DuckDB requires every connection to the same file
    in one process to use the same configuration, so callers should stick to
    these defaults when mixing this helper with Qlib's storage classes.
    """

    if use_cache:
        return _get_connection(
            db_path,
            read_only=read_only,
            memory_limit=memory_limit,
            temp_directory=temp_directory,
            max_temp_directory_size=max_temp_directory_size,
            threads=threads,
        )
    return _open_connection(
        db_path,
        read_only=read_only,
        memory_limit=memory_limit,
        temp_directory=temp_directory,
        max_temp_directory_size=max_temp_directory_size,
        threads=threads,
    )


class DuckDBStorageMixin:
    """Common configuration/read helpers for DuckDB storage classes."""

    def _setup_duckdb(
        self,
        provider_uri=None,
        db_path=None,
        memory_limit: str = DEFAULT_MEMORY_LIMIT,
        temp_directory: Optional[str] = DEFAULT_TEMP_DIRECTORY,
        read_only: bool = True,
        threads: Optional[int] = None,
        max_temp_directory_size: Optional[str] = None,
        calendar_table: Optional[str] = None,
        instrument_table: Optional[str] = None,
        feature_table: Optional[str] = None,
    ) -> None:
        if db_path is not None:
            self._db_path = _normalise_db_path(db_path)
        elif provider_uri is not None:
            self._db_path = _extract_duckdb_uri(provider_uri, str(self.freq))
        else:
            try:
                resolved = C.dpm.get_data_uri(self.freq)
            except Exception:
                resolved = C.get("provider_uri", "")
            self._db_path = _extract_duckdb_uri(resolved, str(self.freq))

        if not self._db_path.exists():
            raise ValueError(f"DuckDB storage does not exist: {self._db_path}")

        if not is_duckdb_uri(self._db_path):
            logger.warning(
                "The configured provider_uri %s does not have a DuckDB suffix; "
                "it will still be opened as a DuckDB database.",
                self._db_path,
            )

        self._duckdb_memory_limit = memory_limit
        self._duckdb_temp_directory = temp_directory
        self._duckdb_read_only = bool(read_only)
        self._duckdb_threads = threads
        self._duckdb_max_temp_directory_size = max_temp_directory_size
        # Table names default by frequency: a minute provider keeps its own
        # calendar/feature tables in its own database file.  Callers may still
        # pin explicit names.
        tables = MINUTE_TABLES if is_minute_freq(self.freq) else DAILY_TABLES
        self._calendar_table = self._validate_identifier(
            calendar_table or tables["calendar"], "calendar_table"
        )
        self._instrument_table = self._validate_identifier(
            instrument_table or tables["instrument"], "instrument_table"
        )
        self._feature_table = self._validate_identifier(
            feature_table or tables["feature"], "feature_table"
        )
        self._minute_freq = is_minute_freq(self.freq)

    @staticmethod
    def _validate_identifier(identifier: str, name: str) -> str:
        if not isinstance(identifier, str) or not _IDENTIFIER_RE.match(identifier):
            raise ValueError(f"Invalid DuckDB {name}: {identifier!r}")
        return identifier

    @property
    def _connection(self):
        return _get_connection(
            self._db_path,
            read_only=self._duckdb_read_only,
            memory_limit=self._duckdb_memory_limit,
            temp_directory=self._duckdb_temp_directory,
            max_temp_directory_size=self._duckdb_max_temp_directory_size,
            threads=self._duckdb_threads,
        )

    @property
    def db_path(self) -> Path:
        return self._db_path

    @property
    def connection(self):
        """Return this storage's cached DuckDB connection.

        The connection is mostly an implementation detail, but exposing it
        makes it easy to verify settings such as ``memory_limit``.
        """

        return self._connection

    @property
    def memory_limit(self) -> str:
        return self._duckdb_memory_limit

    def _fetchall(self, sql: str, parameters: Optional[list] = None):
        return self._connection.execute(sql, parameters or []).fetchall()

    def _fetchone(self, sql: str, parameters: Optional[list] = None):
        return self._connection.execute(sql, parameters or []).fetchone()

    def _check_read_only(self, operation: str):
        # Only read paths are implemented for DuckDB storage.  Keeping the
        # write interface explicit is safer than silently doing nothing when a
        # writable connection was requested.
        raise NotImplementedError(
            f"{self.__class__.__name__}.{operation} is not supported by the DuckDB storage. "
            "The DuckDB backend currently provides read-only access for Qlib."
        )


class DuckDBCalendarStorage(DuckDBStorageMixin, CalendarStorage):
    """Read Qlib's trading calendar from a DuckDB database."""

    def __init__(
        self,
        freq: str,
        future: bool,
        provider_uri=None,
        db_path=None,
        memory_limit: str = DEFAULT_MEMORY_LIMIT,
        temp_directory: Optional[str] = DEFAULT_TEMP_DIRECTORY,
        read_only: bool = True,
        threads: Optional[int] = None,
        max_temp_directory_size: Optional[str] = None,
        calendar_table: Optional[str] = None,
        **kwargs,
    ):
        super(DuckDBCalendarStorage, self).__init__(freq, future, **kwargs)
        self._setup_duckdb(
            provider_uri=provider_uri,
            db_path=db_path,
            memory_limit=memory_limit,
            temp_directory=temp_directory,
            read_only=read_only,
            threads=threads,
            max_temp_directory_size=max_temp_directory_size,
            calendar_table=calendar_table,
        )

    @property
    def storage_name(self) -> str:
        return "calendar"

    @property
    def data(self) -> List[CalVT]:
        table = self._calendar_table
        rows = self._fetchall(f"SELECT trade_date_text, trade_date FROM {table} ORDER BY trade_index")
        values: List[CalVT] = []
        for trade_date_text, trade_date in rows:
            if trade_date_text is not None:
                values.append(trade_date_text)
            else:
                values.append(str(trade_date))
        if Freq(self.freq) == Freq("day") and self.future and len(values) > 0:
            # Qlib's backtest calendar needs a right endpoint for the last
            # trading day (it calls `get_step_time` and looks at index + 1).
            # DuckDB has no separate future-calendar table, so when the caller
            # requests `future=True` we provide the next business day as a
            # boundary.  It is not used as a tradable day for a backtest ending
            # on the last covered date, only as the upper time bound.
            last_value = pd.Timestamp(values[-1])
            next_business_day = pd.bdate_range(last_value + pd.Timedelta(days=1), periods=1)
            if len(next_business_day) == 1:
                values.append(next_business_day[0])

        if self._minute_freq:
            # A minute calendar table stores its own bar labels.  Deriving them
            # from the day calendar is impossible (`resam_calendar` requires a
            # minute raw calendar), so read them as-is and refuse to guess.
            self._validate_minute_calendar(values)
        elif Freq(self.freq) != Freq("day"):
            values = resam_calendar(
                np.array(list(map(pd.Timestamp, values))), "day", self.freq, C.get("region", "cn")
            ).tolist()
        return values

    def _validate_minute_calendar(self, values: List[CalVT]) -> None:
        """Refuse a minute calendar whose rows carry no intraday resolution.

        Silently resampling or truncating here would shift every bar label, so an
        accidentally-configured day calendar must fail loudly.
        """

        if not values:
            return
        stamps = pd.to_datetime(list(values))
        if bool((stamps == stamps.normalize()).all()):
            raise ValueError(
                f"{self._calendar_table} holds day-level labels but freq={self.freq!r} was requested; "
                "point the minute provider at a database whose calendar carries minute labels"
            )

    # Calendar is read-only in this implementation.  The methods below keep the
    # same signatures as `CalendarStorage` so callers receive an explicit,
    # meaningful error instead of a silent no-op.
    def clear(self) -> None:
        self._check_read_only("clear")

    def extend(self, iterable: Iterable[CalVT]) -> None:
        self._check_read_only("extend")

    def index(self, value: CalVT) -> int:
        return self.data.index(value)

    def insert(self, index: int, value: CalVT) -> None:
        self._check_read_only("insert")

    def remove(self, value: CalVT) -> None:
        self._check_read_only("remove")

    def __setitem__(self, i, value) -> None:
        self._check_read_only("__setitem__")

    def __delitem__(self, i) -> None:
        self._check_read_only("__delitem__")

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self) -> int:
        return len(self.data)


class DuckDBInstrumentStorage(DuckDBStorageMixin, InstrumentStorage):
    """Read active instruments from a DuckDB database.

    The supplied database does not keep separate universe membership files
    (e.g. ``csi300``).  The storage therefore supports the two useful cases:

    - ``market`` is ``all``/``cn`` or a Qlib market group name -> return all
      active instruments.
    - ``market`` starts with an exchange prefix such as ``SH``/``SZ`` -> return
      the active instruments of that exchange.
    """

    _EXCHANGE_ALIASES = {
        "SH": "SH",
        "SSE": "SH",
        "SS": "SH",
        "SZ": "SZ",
        "SZSE": "SZ",
        "BSE": "BJ",
        "BJ": "BJ",
    }

    def __init__(
        self,
        market: str,
        freq: str,
        provider_uri=None,
        db_path=None,
        memory_limit: str = DEFAULT_MEMORY_LIMIT,
        temp_directory: Optional[str] = DEFAULT_TEMP_DIRECTORY,
        read_only: bool = True,
        threads: Optional[int] = None,
        max_temp_directory_size: Optional[str] = None,
        instrument_table: Optional[str] = None,
        **kwargs,
    ):
        super(DuckDBInstrumentStorage, self).__init__(market, freq, **kwargs)
        self._setup_duckdb(
            provider_uri=provider_uri,
            db_path=db_path,
            memory_limit=memory_limit,
            temp_directory=temp_directory,
            read_only=read_only,
            threads=threads,
            max_temp_directory_size=max_temp_directory_size,
            instrument_table=instrument_table,
        )

    @property
    def storage_name(self) -> str:
        return "instrument"

    def _market_filter(self) -> Tuple[str, List]:
        market = "" if self.market is None else str(self.market).strip()
        upper_market = market.upper()
        if upper_market in {"", "ALL", "CN", "CHINA", "STOCK", "STOCKS"}:
            return "", []
        exchange = self._EXCHANGE_ALIASES.get(upper_market, upper_market)
        if exchange in {"SH", "SZ", "BJ"}:
            return "qlib_symbol LIKE ?", [f"{exchange}%"]
        # Membership lists such as csi300/csi500 are not part of this database.
        # Return the whole universe and make the fallback visible to the user.
        logger.warning(
            "DuckDB instrument storage cannot resolve market %r (it has no membership table); "
            "returning all active instruments instead.",
            self.market,
        )
        return "", []

    @property
    def data(self) -> Dict[InstKT, InstVT]:
        table = self._instrument_table
        where_sql, params = self._market_filter()
        where = f" WHERE {where_sql}" if where_sql else ""
        rows = self._fetchall(
            f"SELECT qlib_symbol, start_date, end_date FROM {table}{where} ORDER BY qlib_symbol, start_date",
            params,
        )

        instruments: Dict[InstKT, InstVT] = {}
        for symbol, start_date, end_date in rows:
            start = pd.Timestamp(str(int(start_date)))
            end = pd.Timestamp(str(int(end_date)))
            instruments.setdefault(symbol, []).append((start, end))
        return instruments

    def clear(self) -> None:
        self._check_read_only("clear")

    def update(self, *args, **kwargs) -> None:
        self._check_read_only("update")

    def __setitem__(self, k: InstKT, v: InstVT) -> None:
        self._check_read_only("__setitem__")

    def __delitem__(self, k: InstKT) -> None:
        self._check_read_only("__delitem__")

    def __getitem__(self, k: InstKT) -> InstVT:
        return self.data[k]

    def __len__(self) -> int:
        return len(self.data)


class MinuteSource:
    """Where one minute field comes from, resolved from ``qlib_minute_sources``."""

    def __init__(self, field, expression, parquet_root, symbol_table, price_unit):
        self.field = field
        self.expression = check_minute_expression(expression)
        self.parquet_root = Path(str(parquet_root))
        self.symbol_table = Path(str(symbol_table))
        self.price_unit = float(price_unit)


class MinuteCalendar:
    """Position -> (trade_date, minute_slot) mapping of a minute provider."""

    def __init__(self, trade_dates, slots):
        self.trade_dates = np.asarray(trade_dates, dtype=np.int64)
        self.slots = np.asarray(slots, dtype=np.int64)
        if len(self.trade_dates) != len(self.slots):
            raise ValueError("Minute calendar columns have different lengths")
        # 240 bars per day in a fixed order: remember each day's first position and
        # each slot's offset inside a day, so mapping back is O(1) per bar.
        self.day_first_position = {}
        for position, day in enumerate(self.trade_dates):
            self.day_first_position.setdefault(int(day), position)
        self.slot_index = {}
        order = []
        last_day = None
        for position, (day, slot) in enumerate(zip(self.trade_dates, self.slots)):
            if day != last_day:
                order = []
                last_day = day
            if int(slot) in self.slot_index:
                continue
            self.slot_index[int(slot)] = len(order)
            order.append(int(slot))
        self.bars_per_day = len(order)

    def position(self, trade_date: int, slot: int):
        first = self.day_first_position.get(int(trade_date))
        index = self.slot_index.get(int(slot))
        if first is None or index is None:
            return None
        position = first + index
        if position >= len(self.trade_dates):
            return None
        return position

    def days(self) -> List[int]:
        return sorted(self.day_first_position)


class DuckDBFeatureStorage(DuckDBStorageMixin, FeatureStorage):
    """Read a Qlib feature series from ``qlib_daily_features``.

    ``qlib_daily_features`` is a long table which can contain several versions
    of one symbol/day row (``historical`` < ``intraday`` < ``final``, then the
    highest ``batch_id``).  We select the latest version with a small,
    symbol-and-index-filtered window query, so the 2GB memory limit is respected
    even if the full table is much larger than memory.
    """

    _ROW_KIND_PRIORITY = (
        "CASE row_kind "
        "WHEN 'final' THEN 3 "
        "WHEN 'intraday' THEN 2 "
        "WHEN 'historical' THEN 1 "
        "ELSE 0 END"
    )

    # Metadata caches.  Qlib builds one storage object per (instrument, field), so
    # re-reading the provider schema each time would dominate the cost.
    _MINUTE_SOURCE_CACHE: Dict[tuple, Optional["MinuteSource"]] = {}
    _MINUTE_CALENDAR_CACHE: Dict[str, Optional["MinuteCalendar"]] = {}
    _MINUTE_SYMBOL_CACHE: Dict[tuple, Optional[tuple]] = {}
    _MINUTE_FILE_CACHE: Dict[tuple, List[str]] = {}

    def __init__(
        self,
        instrument: str,
        field: str,
        freq: str,
        provider_uri=None,
        db_path=None,
        memory_limit: str = DEFAULT_MEMORY_LIMIT,
        temp_directory: Optional[str] = DEFAULT_TEMP_DIRECTORY,
        read_only: bool = True,
        threads: Optional[int] = None,
        max_temp_directory_size: Optional[str] = None,
        feature_table: Optional[str] = None,
        **kwargs,
    ):
        super(DuckDBFeatureStorage, self).__init__(instrument, field, freq, **kwargs)
        self._setup_duckdb(
            provider_uri=provider_uri,
            db_path=db_path,
            memory_limit=memory_limit,
            temp_directory=temp_directory,
            read_only=read_only,
            threads=threads,
            max_temp_directory_size=max_temp_directory_size,
            feature_table=feature_table,
        )

    @property
    def storage_name(self) -> str:
        return "feature"

    def _present_columns(self) -> frozenset:
        """Real column set of the feature table, cached per (database, table)."""

        key = (str(self.db_path), self._feature_table)
        cached = _TABLE_COLUMN_CACHE.get(key)
        if cached is None:
            try:
                rows = self._fetchall(
                    "SELECT column_name FROM information_schema.columns WHERE table_name = ?",
                    [self._feature_table],
                )
                cached = frozenset(str(row[0]).lower() for row in rows)
            except Exception:  # pragma: no cover - missing table / permissions
                cached = frozenset()
            _TABLE_COLUMN_CACHE[key] = cached
        return cached

    @property
    def _field_column(self) -> Optional[str]:
        field = str(self.field).strip().lower()
        if field.startswith("$"):
            field = field[1:]
        column = _FEATURE_COLUMNS.get(field)
        if column is None:
            return None
        present = self._present_columns()
        # Empty means the schema could not be read (older database, missing
        # table); keep the historical behaviour and do not second-guess it.
        if present and column not in present:
            return None
        return column

    def _field_sql(self) -> Optional[str]:
        column = self._field_column
        if column is None:
            return None
        return f"CAST({column} AS DOUBLE)"

    @property
    def minute_source(self) -> Optional["MinuteSource"]:
        """Resolve (once per provider/field) where the minute field comes from."""

        if not self._minute_freq:
            return None
        column = self._field_column
        if column is None:
            return None
        key = (str(self.db_path), column)
        if key not in DuckDBFeatureStorage._MINUTE_SOURCE_CACHE:
            source = None
            tables = {
                row[0]
                for row in self._fetchall(
                    "SELECT table_name FROM information_schema.tables WHERE table_name = ?",
                    [MINUTE_SOURCE_TABLE],
                )
            }
            if tables:
                rows = self._fetchall(
                    f"SELECT field, source_expr, parquet_root, symbol_table, price_unit "
                    f"FROM {MINUTE_SOURCE_TABLE} WHERE field = ?",
                    [column],
                )
                if rows:
                    source = MinuteSource(*rows[0])
            DuckDBFeatureStorage._MINUTE_SOURCE_CACHE[key] = source
        return DuckDBFeatureStorage._MINUTE_SOURCE_CACHE[key]

    @property
    def minute_calendar(self) -> Optional["MinuteCalendar"]:
        if not self._minute_freq:
            return None
        key = str(self.db_path)
        if key not in DuckDBFeatureStorage._MINUTE_CALENDAR_CACHE:
            calendar = None
            try:
                rows = self._fetchall(
                    f"SELECT trade_date, minute_slot FROM {self._calendar_table} "
                    "ORDER BY trade_index"
                )
            except Exception:  # pragma: no cover - provider without a minute calendar
                rows = []
            if rows:
                calendar = MinuteCalendar(
                    [int(row[0]) for row in rows], [int(row[1]) for row in rows]
                )
            DuckDBFeatureStorage._MINUTE_CALENDAR_CACHE[key] = calendar
        return DuckDBFeatureStorage._MINUTE_CALENDAR_CACHE[key]

    @property
    def minute_symbol(self):
        """``(symbol_id, exchange)`` of this instrument, from the declared map."""

        source = self.minute_source
        if source is None:
            return None
        key = (str(source.symbol_table), self.instrument)
        if key not in DuckDBFeatureStorage._MINUTE_SYMBOL_CACHE:
            rows = self._fetchall(
                f"SELECT symbol_id, exchange FROM read_parquet('{source.symbol_table}') "
                "WHERE qlib_symbol = ?",
                [self.instrument],
            )
            DuckDBFeatureStorage._MINUTE_SYMBOL_CACHE[key] = (
                (int(rows[0][0]), str(rows[0][1])) if rows else None
            )
        return DuckDBFeatureStorage._MINUTE_SYMBOL_CACHE[key]

    def minute_files(self, days: List[int], exchange: str, symbol_id: int) -> List[str]:
        """Parquet partitions covering ``days`` for one symbol, pruned by bucket."""

        bucket = int(symbol_id) % 8
        key = (str(self.minute_source.parquet_root), exchange, bucket)
        catalogue = DuckDBFeatureStorage._MINUTE_FILE_CACHE.get(key)
        if catalogue is None:
            catalogue = {}
            root = self.minute_source.parquet_root
            for year_dir in sorted(root.glob("year=*")):
                for month_dir in sorted(year_dir.glob("month=*")):
                    directory = month_dir / f"exchange={exchange}" / f"bucket={bucket:02d}"
                    if directory.is_dir():
                        files = sorted(str(path) for path in directory.glob("*.parquet"))
                        if files:
                            catalogue[(int(year_dir.name.split("=")[1]),
                                       int(month_dir.name.split("=")[1]))] = files
            DuckDBFeatureStorage._MINUTE_FILE_CACHE[key] = catalogue
        months = {(int(str(day)[:4]), int(str(day)[4:6])) for day in days}
        selected = []
        for month in sorted(months):
            selected.extend(catalogue.get(month, ()))
        return selected

    @property
    def start_index(self) -> Optional[int]:
        if self._minute_freq:
            calendar = self.minute_calendar
            return None if calendar is None else 0
        row = self._fetchone(
            f"SELECT MIN(trade_index) FROM {self._feature_table} "
            "WHERE qlib_symbol = ? AND row_status = 'active'",
            [self.instrument],
        )
        return None if row is None or row[0] is None else int(row[0])

    @property
    def end_index(self) -> Optional[int]:
        if self._minute_freq:
            calendar = self.minute_calendar
            return None if calendar is None else len(calendar.trade_dates) - 1
        row = self._fetchone(
            f"SELECT MAX(trade_index) FROM {self._feature_table} "
            "WHERE qlib_symbol = ? AND row_status = 'active'",
            [self.instrument],
        )
        return None if row is None or row[0] is None else int(row[0])

    @property
    def data(self) -> pd.Series:
        return self[:]

    def clear(self) -> None:
        self._check_read_only("clear")

    def write(self, data_array, index: int = None) -> None:
        self._check_read_only("write")

    def _empty_series(self) -> pd.Series:
        return pd.Series(dtype=np.float32)

    def _read_range(self, start: int, end: int) -> pd.Series:
        """Read ``[start, end]`` (calendar index, both closed) for one symbol."""

        if start is None or end is None or end < start:
            return self._empty_series()
        if self._minute_freq and self.minute_source is not None:
            return self._read_minute_range(int(start), int(end))
        column = self._field_sql()
        if column is None:
            logger.debug("Feature %r is not present in DuckDB table %s", self.field, self._feature_table)
            return self._empty_series()

        sql = f"""
            SELECT trade_index, value
            FROM (
                SELECT
                    trade_index,
                    {column} AS value,
                    row_number() OVER (
                        PARTITION BY trade_date
                        ORDER BY {self._ROW_KIND_PRIORITY} DESC, batch_id DESC
                    ) AS rn
                FROM {self._feature_table}
                WHERE qlib_symbol = ?
                  AND row_status = 'active'
                  AND trade_index >= ?
                  AND trade_index <= ?
            ) AS feature_versions
            WHERE rn = 1
            ORDER BY trade_index
        """
        rows = self._fetchall(sql, [self.instrument, int(start), int(end)])
        if not rows:
            return self._empty_series()

        actual_index = pd.Index([int(row[0]) for row in rows], dtype="int64")
        series = pd.Series(
            [float(row[1]) if row[1] is not None else np.nan for row in rows], index=actual_index, dtype=np.float32
        )

        # qlib's file storage returns a contiguous series for the requested
        # range (missing dates are represented with NaN).  Mimic that behavior
        # so callers can rely on absolute calendar indexing.
        expected_index = pd.RangeIndex(int(start), int(end) + 1)
        if len(series) != len(expected_index):
            series = series.reindex(expected_index)
        return series

    def _read_minute_range(self, start: int, end: int) -> pd.Series:
        """Read minute bars for one symbol straight from the warehouse parquet.

        One DuckDB query per (instrument, field, window).  Partition pruning is by
        ``year/month/exchange/bucket`` with ``bucket = symbol_id % 8``, so a symbol
        never triggers a scan of the whole warehouse.  The connection carries the
        module's memory limit (2GB by default).
        """

        source = self.minute_source
        calendar = self.minute_calendar
        symbol = self.minute_symbol
        if source is None or calendar is None or symbol is None:
            return self._empty_series()
        symbol_id, exchange = symbol
        expected_index = pd.RangeIndex(start, end + 1)
        days = sorted({int(day) for day in calendar.trade_dates[start : end + 1]})
        if not days:
            return self._empty_series()
        files = self.minute_files(days, exchange, symbol_id)
        if not files:
            return self._empty_series()
        placeholders = ",".join("?" for _ in days)
        sql = f"""
            SELECT trade_date, minute_slot, {source.expression} AS value
            FROM read_parquet(?, hive_partitioning=true)
            WHERE symbol_id = ? AND trade_date IN ({placeholders})
            ORDER BY trade_date, minute_slot
        """
        rows = self._connection.execute(sql, [files, symbol_id, *days]).fetchall()
        if not rows:
            return self._empty_series()
        positions, values = [], []
        for trade_date, slot, value in rows:
            position = calendar.position(int(trade_date), int(slot))
            if position is None or position < start or position > end:
                continue
            positions.append(position)
            values.append(np.nan if value is None else float(value))
        series = pd.Series(values, index=pd.Index(positions, dtype="int64"), dtype=np.float32)
        return series.reindex(expected_index)

    def __getitem__(self, i):
        storage_start = self.start_index
        storage_end = self.end_index
        if storage_start is None or storage_end is None:
            if isinstance(i, int):
                return None, None
            if isinstance(i, slice):
                return self._empty_series()
            raise TypeError(f"type(i) = {type(i)}")

        if isinstance(i, int):
            if i < storage_start:
                raise IndexError(f"{i}: start index is {storage_start}")
            if i > storage_end:
                return None, None
            series = self._read_range(i, i)
            if series.empty:
                return i, float("nan")
            return i, float(series.iloc[0])

        if isinstance(i, slice):
            start = storage_start if i.start is None else int(i.start)
            stop = storage_end + 1 if i.stop is None else int(i.stop)
            if stop <= start:
                return self._empty_series()
            start = max(start, storage_start)
            end = min(stop - 1, storage_end)
            if end < start:
                return self._empty_series()
            return self._read_range(start, end)

        raise TypeError(f"type(i) = {type(i)}")

    def __len__(self) -> int:
        row = self._fetchone(
            f"SELECT COUNT(DISTINCT trade_date) FROM {self._feature_table} "
            "WHERE qlib_symbol = ? AND row_status = 'active'",
            [self.instrument],
        )
        return 0 if row is None or row[0] is None else int(row[0])


__all__ = [
    "DUCKDB_SUFFIXES",
    "DEFAULT_MEMORY_LIMIT",
    "DEFAULT_TEMP_DIRECTORY",
    "MINUTE_SOURCE_TABLE",
    "MinuteSource",
    "MinuteCalendar",
    "check_minute_expression",
    "is_duckdb_uri",
    "is_minute_freq",
    "get_duckdb_connection",
    "DuckDBStorageMixin",
    "DuckDBCalendarStorage",
    "DuckDBInstrumentStorage",
    "DuckDBFeatureStorage",
]
