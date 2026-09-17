#!/usr/bin/env python3
"""Audit the frozen buy-price cap: what does it reject, and what did those names do?

docs/chats_002.md section 4.1 requires this check explicitly: the 3% cap is kept
to avoid changing too many execution conditions at once, and the trades it
refuses must be inspected separately rather than assumed harmless.
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
    parser.add_argument("--run", default="sleeve/a/output/v2_g4")
    parser.add_argument("--horizons", default="5,20")
    parser.add_argument("--output", default="/Volumes/lexar_4t/code/data/qlib/sleeve/a/output/v2_research/price_cap_audit.csv")
    args = parser.parse_args()
    cfg = load_config(str(ROOT / "configs" / "v0.yaml"))
    init_local_qlib(cfg)
    orders = pd.read_csv(Path(ROOT) / args.run / "orders.csv")
    buys = orders[orders["side"] == "BUY"].copy()
    print("buy orders:", len(buys), "statuses:", buys["status"].value_counts().to_dict())
    with DuckDBPanelLoader(cfg, use_cache=True) as loader:
        panel = loader.load(refresh_cache=False)
    raw = panel.raw_close()
    date_to_index = {d: i for i, d in enumerate(panel.dates)}
    symbol_to_j = {s: j for j, s in enumerate(panel.symbols)}
    rows = []
    for horizon in [int(h) for h in args.horizons.split(",")]:
        for status, frame in buys.groupby("status"):
            values = []
            for record in frame.itertuples():
                t = date_to_index.get(pd.Timestamp(record.decision_date))
                j = symbol_to_j.get(record.symbol)
                if t is None or j is None or t + horizon >= raw.shape[0]:
                    continue
                p0 = raw[t, j]
                p1 = raw[t + horizon, j]
                if np.isfinite(p0) and np.isfinite(p1) and p0 > 0:
                    values.append(p1 / p0 - 1.0)
            if not values:
                continue
            values = np.asarray(values)
            rows.append(
                {
                    "horizon": horizon,
                    "status": status,
                    "n": len(values),
                    "mean_return": float(values.mean()),
                    "median_return": float(np.median(values)),
                    "win_rate": float((values > 0).mean()),
                }
            )
            print(
                f"[cap] h={horizon:2d} {status:12s} n={len(values):4d} mean={values.mean():+.4f} "
                f"median={np.median(values):+.4f} win={(values > 0).mean():.4f}"
            )
    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
