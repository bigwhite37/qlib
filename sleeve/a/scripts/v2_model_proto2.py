#!/usr/bin/env python3
"""Quick model-driven V2 prototype using simple panel features + ranks.

This is a fast feasibility test before spending time on the full Alpha158
rolling pipeline:
* train a return and a win model on 2016-2018 matured policy labels;
* validate on 2019H1, calibrate win on 2019H2;
* predict all V2 base rows in 2020;
* run the same Qlib account/execution stack with several selection rules.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lowvol_trend.bootstrap import ensure_local_qlib, init_local_qlib  # noqa: E402

ensure_local_qlib()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import lightgbm as lgb  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from lowvol_trend.config import Config, load_config  # noqa: E402
from lowvol_trend.data import DuckDBPanelLoader  # noqa: E402
from lowvol_trend.features import FeatureStore, build_features, row_pct_rank  # noqa: E402
from lowvol_trend.v2 import compute_v2_market_state, v2_base_mask  # noqa: E402
from scripts.research_variants import run_variant  # noqa: E402


def build_matrix(features: FeatureStore, market, t_idx, j_idx):
    close = pd.DataFrame(features.arr("close"), index=features.dates, columns=features.symbols)
    r5 = close.pct_change(5, fill_method=None).to_numpy(np.float32)
    r20 = close.pct_change(20, fill_method=None).to_numpy(np.float32)
    r60 = close.pct_change(60, fill_method=None).to_numpy(np.float32)
    proxy = pd.Series(market.proxy, index=features.dates)
    m5 = (proxy / proxy.shift(5) - 1).to_numpy(np.float32)
    m20 = (proxy / proxy.shift(20) - 1).to_numpy(np.float32)
    m60 = (proxy / proxy.shift(60) - 1).to_numpy(np.float32)
    stock = {
        "ret1": features.arr("ret1")[t_idx, j_idx],
        "ret5": r5[t_idx, j_idx],
        "ret20": r20[t_idx, j_idx],
        "ret60": r60[t_idx, j_idx],
        "rel5": r5[t_idx, j_idx] - m5[t_idx],
        "rel20": r20[t_idx, j_idx] - m20[t_idx],
        "rel60": r60[t_idx, j_idx] - m60[t_idx],
        "vol60": features.arr("vol60")[t_idx, j_idx],
        "downvol20": features.arr("downvol20")[t_idx, j_idx],
        "atr_ratio": features.arr("atr20")[t_idx, j_idx]
        / np.where(features.arr("close")[t_idx, j_idx] > 0, features.arr("close")[t_idx, j_idx], np.nan),
        "dist_high_atr": features.arr("dist_high20")[t_idx, j_idx]
        / np.where(features.arr("atr20")[t_idx, j_idx] > 0, features.arr("atr20")[t_idx, j_idx], np.nan),
        "rs60": features.arr("rs60")[t_idx, j_idx],
        "trend_quality": features.arr("trend_quality60")[t_idx, j_idx],
        "amount_rank": features.arr("amount_rank")[t_idx, j_idx],
        "raw_price": features.arr("raw_close")[t_idx, j_idx],
        "valid_count": features.arr("valid_count")[t_idx, j_idx],
    }
    df = pd.DataFrame(stock)
    df["signal_date"] = features.dates[t_idx]
    # Cross-sectional ranks of stock features inside the candidate frame.
    for col in list(stock):
        df[col + "_rk"] = df.groupby("signal_date")[col].rank(pct=True)
    market_cols = {
        "mkt_ret5": m5[t_idx],
        "mkt_ret20": m20[t_idx],
        "mkt_ret60": m60[t_idx],
        "mkt_b20": market.breadth_20[t_idx],
        "mkt_b60": market.breadth_60[t_idx],
        "mkt_q": market.drop_diffusion[t_idx],
        "mkt_cap": market.effective_cap[t_idx],
        "mkt_rawcap": market.raw_cap[t_idx],
        "mkt_recovery": market.recovery_phase[t_idx],
    }
    for k, v in market_cols.items():
        df[k] = v
    feature_cols = [c for c in df.columns if c != "signal_date"]
    return df, feature_cols


def main() -> None:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    cfg = load_config(str(ROOT / "configs" / "v0.yaml"))
    init_local_qlib(cfg)
    with DuckDBPanelLoader(cfg, use_cache=True) as loader:
        panel = loader.load(refresh_cache=False)
    print("[v2-model] features ...", flush=True)
    features = build_features(panel, cfg)
    base = v2_base_mask(features, cfg)
    market = compute_v2_market_state(features, base, cfg)
    labels = pd.read_parquet(ROOT / "cache" / "v2_labels.parquet")
    labels = labels[labels["status"] == "closed"].copy()
    labels["signal_date"] = pd.to_datetime(labels["signal_date"])
    labels["label_end_time"] = pd.to_datetime(labels["label_end_time"])
    symbol_index = {s: j for j, s in enumerate(panel.symbols)}

    # Training feature matrices only for rows with closed labels.
    def rows_masked(start, end, label_limit):
        m = labels["signal_date"].between(start, end)
        if label_limit is not None:
            m &= labels["label_end_time"] < pd.Timestamp(label_limit)
        return labels[m].reset_index(drop=True)

    parts: Dict[str, tuple] = {}
    for name, (start, end, limit) in {
        "train": ("2016-01-01", "2018-12-31", "2019-01-01"),
        "valid": ("2019-01-01", "2019-06-30", "2019-07-01"),
        "calib": ("2019-07-01", "2019-12-31", "2020-01-01"),
    }.items():
        rows = rows_masked(start, end, limit)
        t = rows["signal_index"].to_numpy(np.int64)
        j = rows["instrument"].map(symbol_index).to_numpy(np.int64)
        df, cols = build_matrix(features, market, t, j)
        df["policy_return_net"] = rows["policy_return_net"].to_numpy(np.float32)
        df["policy_win"] = rows["policy_win"].to_numpy(np.float32)
        parts[name] = (df, cols)
        print(f"[v2-model] {name} rows={len(df)}", flush=True)

    feature_cols = parts["train"][1]
    Xtr = parts["train"][0][feature_cols].to_numpy(np.float32)
    Xva = parts["valid"][0][feature_cols].to_numpy(np.float32)
    Xca = parts["calib"][0][feature_cols].to_numpy(np.float32)
    ytr_ret = parts["train"][0]["policy_return_net"].to_numpy(np.float32)
    yva_ret = parts["valid"][0]["policy_return_net"].to_numpy(np.float32)
    ytr_win = parts["train"][0]["policy_win"].to_numpy(np.int8)
    yva_win = parts["valid"][0]["policy_win"].to_numpy(np.int8)
    params = dict(num_leaves=15, max_depth=4, learning_rate=0.03, min_data_in_leaf=500, verbosity=-1, seed=42)
    print("[v2-model] train return model", flush=True)
    ret_model = lgb.train(dict(params, objective="mse"), lgb.Dataset(Xtr, label=ytr_ret), num_boost_round=300, valid_sets=[lgb.Dataset(Xva, label=yva_ret)], callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
    print("[v2-model] train win model", flush=True)
    win_model = lgb.train(dict(params, objective="binary"), lgb.Dataset(Xtr, label=ytr_win), num_boost_round=300, valid_sets=[lgb.Dataset(Xva, label=yva_win)], callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
    raw_ca = win_model.predict(Xca)
    calib = LogisticRegression().fit(raw_ca.reshape(-1, 1), parts["calib"][0]["policy_win"].to_numpy(np.int8))
    print("[v2-model] best iters", ret_model.best_iteration, win_model.best_iteration, flush=True)

    # Predict all base rows in 2020.
    t_all, j_all = np.nonzero(base & (features.dates.year == 2020)[:, None])
    df_test, _ = build_matrix(features, market, t_all, j_all)
    pred_ret = ret_model.predict(df_test[feature_cols].to_numpy(np.float32))
    raw = win_model.predict(df_test[feature_cols].to_numpy(np.float32))
    pred_win = calib.predict_proba(raw.reshape(-1, 1))[:, 1]
    pr = np.full(base.shape, np.nan, np.float32)
    pw = np.full(base.shape, np.nan, np.float32)
    pr[t_all, j_all] = pred_ret
    pw[t_all, j_all] = pred_win
    print("[v2-model] predict rows", len(df_test), "ret range", pred_ret.min(), pred_ret.max(), "win range", pred_win.min(), pred_win.max(), flush=True)

    def mutate(c: Config):
        c.strategy.profit_target_pct = 0.0
        c.strategy.exit_trend_ma20 = False
        c.strategy.exit_trend_ma60 = False
        c.strategy.exit_rank = False
        c.strategy.stop_atr_mult = 2.0
        c.strategy.stop_min = 0.04
        c.strategy.stop_max = 0.08
        c.strategy.max_hold_days = 20
        c.strategy.use_market_timing = True

    rules = [
        ("top10pct_ret", lambda: _top_rule(pr, 0.10)),
        ("top20pct_ret", lambda: _top_rule(pr, 0.20)),
        ("top10pct_ret_p45", lambda: _top_rule(pr, 0.10) & (pw >= 0.45)),
        ("top20pct_ret_p45", lambda: _top_rule(pr, 0.20) & (pw >= 0.45)),
        ("p50", lambda: (pw >= 0.50)),
        ("ret_positive", lambda: (pr >= 0.0)),
    ]
    for tag, fn in rules:
        sig = base & np.isfinite(pr) & fn()
        if sig.sum() == 0:
            print(f"[v2-model] rule {tag}: no signals", flush=True)
            continue
        arrays = dict(features.arrays)
        arrays["entry_signal"] = sig
        arrays["entry_score"] = np.where(sig, pr, np.nan).astype(np.float32)
        fv = FeatureStore(panel=features.panel, cfg=features.cfg, market=features.market, arrays=arrays, diagnostics=dict(features.diagnostics))
        r = run_variant(fv, cfg, "2020-01-02", "2020-12-31", mutate)
        print(
            f"[v2-model] rule {tag}: signals={int(sig.sum())} win={r['win']:.4f} exp={r['expectancy']:.4f} "
            f"cagr={r['cagr']:.4f} vol={r['vol']:.4f} mdd={r['mdd']:.4f} rounds={r['rounds']}",
            flush=True,
        )


def _top_rule(pr: np.ndarray, q: float) -> np.ndarray:
    out = np.zeros_like(pr, dtype=bool)
    for t in range(pr.shape[0]):
        row = pr[t]
        finite = np.isfinite(row)
        if finite.sum() == 0:
            continue
        thr = np.nanquantile(row, 1.0 - q)
        out[t] = finite & (row >= thr)
    return out


if __name__ == "__main__":
    main()
