#!/usr/bin/env python3
"""Build forward-return labels for the cached V2 matrix rows.

The ranking model has so far been trained on the realised policy net return,
which is truncated by the exit policy and therefore noisy.  This script adds the
plain (cross-sectionally demeaned) forward raw return so the model can be trained
on the alpha directly, while the win model keeps its policy-consistent label.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lowvol_trend.bootstrap import ensure_local_qlib, init_local_qlib  # noqa: E402

ensure_local_qlib()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lowvol_trend.config import load_config  # noqa: E402
from lowvol_trend.data import DuckDBPanelLoader  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="/Volumes/lexar_4t/code/data/qlib/sleeve/a/cache/v2m_tp5h40")
    parser.add_argument("--horizons", default="10,20")
    args = parser.parse_args()
    prefix = Path(args.prefix)
    cfg = load_config(str(ROOT / "configs" / "v0.yaml"))
    init_local_qlib(cfg)
    with DuckDBPanelLoader(cfg, use_cache=True) as loader:
        panel = loader.load(refresh_cache=False)
    with np.load(prefix.with_name(prefix.name + "_labels.npz")) as handle:
        signal_index = handle["signal_index"].astype(np.int64)
        symbol_index = handle["symbol_index"].astype(np.int64)
        entry_filled = handle["entry_filled"]
    raw = np.asarray(panel.raw_close(), dtype=np.float64)
    n_dates = raw.shape[0]
    out = {}
    for horizon in [int(h) for h in args.horizons.split(",")]:
        fwd = np.full(len(signal_index), np.nan, dtype=np.float32)
        ok = (signal_index + horizon) < n_dates
        idx = signal_index[ok]
        jdx = symbol_index[ok]
        with np.errstate(all="ignore"):
            values = raw[idx + horizon, jdx] / raw[idx, jdx] - 1.0
        values[~np.isfinite(values)] = np.nan
        fwd[ok] = values.astype(np.float32)
        # cross-sectional demeaning by signal date
        frame = pd.DataFrame({"d": signal_index, "v": fwd})
        mean = frame.groupby("d")["v"].transform("mean").to_numpy()
        out["fwd" + str(horizon)] = (fwd - mean).astype(np.float32)
        print(f"[fwd] h={horizon} finite {np.isfinite(out['fwd' + str(horizon)]).sum()} of {len(fwd)}", flush=True)
    out["entry_filled"] = entry_filled
    target = prefix.with_name(prefix.name + "_fwd.npz")
    np.savez(target, **out)
    print("[fwd] saved", target)


if __name__ == "__main__":
    main()
