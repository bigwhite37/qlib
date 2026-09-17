#!/usr/bin/env python3
"""IC of panel features against the *policy* labels (not raw forward returns).

Decides whether the model should learn the policy outcome directly or whether
the exit policy destroys the predictability that exists in forward returns.
"""

from __future__ import annotations

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
from lowvol_trend.features import build_features  # noqa: E402
from lowvol_trend.v2 import compute_v2_market_state, v2_base_mask  # noqa: E402

sys.path.insert(0, str(ROOT))
from scripts.v2_fast_probe import FeatureBank  # noqa: E402


def main() -> None:
    out = Path("/Volumes/lexar_4t/code/data/qlib/sleeve/a/output/v2_research")
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(ROOT / "configs" / "v0.yaml"))
    init_local_qlib(cfg)
    with DuckDBPanelLoader(cfg, use_cache=True) as loader:
        panel = loader.load(refresh_cache=False)
    features = build_features(panel, cfg)
    base = v2_base_mask(features, cfg)
    market = compute_v2_market_state(features, base, cfg)
    bank = FeatureBank(features, market)
    labels = pd.read_parquet(
        "/Volumes/lexar_4t/code/data/qlib/sleeve/a/cache/v2_labels.parquet",
        columns=["signal_date", "signal_index", "instrument", "status", "policy_return_net", "policy_win"],
    )
    labels = labels[labels["status"] == "closed"].reset_index(drop=True)
    labels["signal_date"] = pd.to_datetime(labels["signal_date"])
    symbol_index = {s: j for j, s in enumerate(panel.symbols)}
    labels["j"] = labels["instrument"].map(symbol_index)
    labels = labels[labels["j"].notna()].copy()
    labels["j"] = labels["j"].astype(np.int64)
    labels = labels[labels["signal_date"] >= "2016-01-01"].reset_index(drop=True)
    t = labels["signal_index"].to_numpy(dtype=np.int64)
    j = labels["j"].to_numpy(dtype=np.int64)
    print("[lab-ic] rows", len(labels), flush=True)
    X = bank.matrix(t, j)
    names = bank.names
    frame = pd.DataFrame({"date": labels["signal_date"].to_numpy()})
    y_ret = labels["policy_return_net"].to_numpy(dtype=np.float64)
    y_win = labels["policy_win"].to_numpy(dtype=np.float64)
    # cross-sectional demeaning of the return label
    y_rel = y_ret - pd.Series(y_ret).groupby(frame["date"]).transform("mean").to_numpy()
    rows = []
    for k, name in enumerate(names):
        col = X[:, k]
        tmp = pd.DataFrame({"d": frame["date"].to_numpy(), "x": col, "yr": y_ret, "yw": y_win, "yrl": y_rel})
        tmp = tmp[np.isfinite(tmp["x"])]
        if len(tmp) < 10000:
            continue
        g = tmp.groupby("d")
        ic_ret = g.apply(lambda s: s["x"].rank().corr(s["yr"].rank()), include_groups=False)
        ic_rel = g.apply(lambda s: s["x"].rank().corr(s["yrl"].rank()), include_groups=False)
        ic_win = g.apply(lambda s: s["x"].rank().corr(s["yw"].rank()), include_groups=False)
        rows.append(
            {
                "feature": name,
                "ic_policy_ret": float(ic_ret.mean()),
                "t_ret": float(ic_ret.mean() / ic_ret.std() * np.sqrt(len(ic_ret))) if ic_ret.std() > 0 else np.nan,
                "ic_policy_rel": float(ic_rel.mean()),
                "t_rel": float(ic_rel.mean() / ic_rel.std() * np.sqrt(len(ic_rel))) if ic_rel.std() > 0 else np.nan,
                "ic_policy_win": float(ic_win.mean()),
                "t_win": float(ic_win.mean() / ic_win.std() * np.sqrt(len(ic_win))) if ic_win.std() > 0 else np.nan,
            }
        )
        print(
            f"[lab-ic] {name:16s} ic_ret={rows[-1]['ic_policy_ret']:+.4f} (t={rows[-1]['t_ret']:+.1f}) "
            f"ic_rel={rows[-1]['ic_policy_rel']:+.4f} (t={rows[-1]['t_rel']:+.1f}) "
            f"ic_win={rows[-1]['ic_policy_win']:+.4f} (t={rows[-1]['t_win']:+.1f})",
            flush=True,
        )
    table = pd.DataFrame(rows).sort_values("ic_policy_rel")
    table.to_csv(out / "label_ic_scan.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
