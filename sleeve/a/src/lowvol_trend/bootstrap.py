"""Bootstrap the *local* Qlib checkout instead of any pip-installed copy."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Dict

from .config import Config, DEFAULT_QLIB_HOME


class QlibBootstrapError(RuntimeError):
    pass


def ensure_local_qlib(qlib_home: str = DEFAULT_QLIB_HOME) -> Any:
    """Import Qlib from ``qlib_home`` and fail if another installation wins.

    The project requirement explicitly says the pip-installed Qlib must not be
    used.  We therefore prepend the local checkout to ``sys.path`` and, when a
    different ``qlib`` module was already imported, remove it before retrying.
    """

    home = Path(qlib_home).expanduser().resolve()
    init_file = home / "qlib" / "__init__.py"
    if not init_file.exists():
        raise QlibBootstrapError(f"Local qlib checkout not found: {init_file}")

    home_str = str(home)
    if home_str not in sys.path:
        sys.path.insert(0, home_str)

    existing = sys.modules.get("qlib")
    if existing is not None:
        existing_file = Path(getattr(existing, "__file__", "") or "").resolve()
        if home not in existing_file.parents:
            # Drop the imported package (and its submodules) and import again.
            for name in list(sys.modules):
                if name == "qlib" or name.startswith("qlib."):
                    del sys.modules[name]
            importlib.invalidate_caches()

    import qlib  # noqa: WPS433 (deliberate late import)

    qlib_file = Path(qlib.__file__).resolve()
    if home not in qlib_file.parents:
        raise QlibBootstrapError(
            f"Imported qlib from {qlib_file}, expected a module below {home}. "
            "Refusing to continue with the pip-installed Qlib."
        )
    return qlib


def init_local_qlib(cfg: Config) -> Dict[str, Any]:
    """Initialise Qlib against the DuckDB provider and return diagnostics."""

    from qlib.constant import REG_CN  # local import after path bootstrap

    qlib = ensure_local_qlib(cfg.data.qlib_home)
    qlib.init(
        provider_uri=cfg.data.db_path,
        region=REG_CN,
        expression_cache=None,
        dataset_cache=None,
        # DuckDB connections are thread-local and must not be inherited by
        # multiprocessing children (which would deadlock or conflict on the
        # same database file).  Qlib supports a threading joblib backend.
        joblib_backend="threading",
    )
    from qlib.data import D

    calendar = D.calendar(freq="day", start_time=cfg.data.start_date, end_time=cfg.data.end_date)
    return {
        "qlib_file": str(Path(qlib.__file__).resolve()),
        "provider_uri": cfg.data.db_path,
        "calendar_len": int(len(calendar)),
        "calendar_first": str(calendar[0]) if len(calendar) else None,
        "calendar_last": str(calendar[-1]) if len(calendar) else None,
    }
