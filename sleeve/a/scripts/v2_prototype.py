#!/usr/bin/env python3
"""Prototype the V2 two-model pipeline on one time split.

Split used here (for a quick signal test):
    train 2016-2018, valid 2019H1, calibrate 2019H2, test 2020.
Features are built on demand from the panel and discarded; only the small
prediction tables are saved, because the full feature matrix is large.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lowvol_trend.bootstrap import ensure_local_qlib, init_local_qlib  # noqa: E402

ensure_local_qlib()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lowvol_trend.alpha158_features import Alpha158Features, alpha158_feature_names  # noqa: E402
from lowvol_trend.config import load_config  # noqa: E402
from lowvol_trend.data import DuckDBPanelLoader  # noqa: E402
from lowvol_trend.features import build_features  # noqa: E402
from lowvol_trend.v2 import compute_v2_market_state, v2_base_mask  # noqa: E402


def build_extra(features, market, t_idx, j_idx):
    close = pd.DataFrame(features.arr("close"), index=features.dates, columns=features.symbols)
    ret1 = features.arr("ret1")
    ret5 = close.pct_change(5, fill_method=None).to_numpy(dtype=np.float32)
    ret20 = close.pct_change(20, fill_method=None).to_numpy(dtype=np.float32)
    ret60 = close.pct_change(60, fill_method=None).to_numpy(dtype=np.float32)
    proxy = pd.Series(market.proxy, index=features.dates)
    m_ret1 = proxy.pct_change(fill_method=None).to_numpy(dtype=np.float32)
    m_ret5 = (proxy / proxy.shift(5) - 1.0).to_numpy(dtype=np.float32)
    m_ret20 = (proxy / proxy.shift(20) - 1.0).to_numpy(dtype=np.float32)
    m_ret60 = (proxy / proxy.shift(60) - 1.0).to_numpy(dtype=np.float32)
    return {
        "ret1": ret1[t_idx, j_idx],
        "ret5": ret5[t_idx, j_idx],
        "ret20": ret20[t_idx, j_idx],
        "ret60": ret60[t_idx, j_idx],
        "rel_ret5": ret5[t_idx, j_idx] - m_ret5[t_idx],
        "rel_ret20": ret20[t_idx, j_idx] - m_ret20[t_idx],
        "rel_ret60": ret60[t_idx, j_idx] - m_ret60[t_idx],
        "market_proxy_ret1": m_ret1[t_idx],
        "market_proxy_ret5": m_ret5[t_idx],
        "market_proxy_ret20": m_ret20[t_idx],
        "market_proxy_ret60": m_ret60[t_idx],
        "market_breadth20": market.breadth_20[t_idx],
        "market_breadth60": market.breadth_60[t_idx],
        "market_drop_diffusion": market.drop_diffusion[t_idx],
        "market_effective_cap": market.effective_cap[t_idx],
        "market_raw_cap": market.raw_cap[t_idx],
        "market_recovery_phase": market.recovery_phase[t_idx],
        "amount_rank": features.arr("amount_rank")[t_idx, j_idx],
        "raw_price": features.arr("raw_close")[t_idx, j_idx],
        "valid_count": features.arr("valid_count")[t_idx, j_idx],
    }


def make_features(features, market, rows: pd.DataFrame, symbol_index) -> Tuple[np.ndarray, list]:
    t_idx = rows["signal_index"].to_numpy(dtype=np.int64)
    j_idx = rows["instrument"].map(symbol_index).to_numpy(dtype=np.int64)
    alpha = Alpha158Features(features).compute(t_idx, j_idx)
    extra = build_extra(features, market, t_idx, j_idx)
    names = list(alpha.keys()) + list(extra.keys())
    matrix = np.column_stack([alpha[k] for k in alpha] + [extra[k] for k in extra]).astype(np.float32)
    return matrix, names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default=str(ROOT / "cache" / "v2_labels.parquet"))
    parser.add_argument("--output", default=str(ROOT / "cache" / "v2_prototype_predictions.parquet"))
    args = parser.parse_args()
    cfg = load_config(str(ROOT / "configs" / "v0.yaml"))
    init_local_qlib(cfg)
    with DuckDBPanelLoader(cfg, use_cache=True) as loader:
        panel = loader.load(refresh_cache=False)
    print("[proto] building features ...", flush=True)
    features = build_features(panel, cfg)
    base = v2_base_mask(features, cfg)
    market = compute_v2_market_state(features, base, cfg)
    labels = pd.read_parquet(args.labels)
    symbol_index = {s: j for j, s in enumerate(panel.symbols)}
    labels = labels[labels["status"] == "closed"].copy()
    labels["signal_date"] = pd.to_datetime(labels["signal_date"])
    labels["label_end_time"] = pd.to_datetime(labels["label_end_time"])

    segments = {
        "train": ("2016-01-01", "2018-12-31", "2019-07-01"),
        "valid": ("2019-01-01", "2019-06-30", "2019-07-01"),
        "calib": ("2019-07-01", "2019-12-31", "2020-01-01"),
        "test": ("2020-01-01", "2020-12-31", None),
    }
    matrices: Dict[str, Tuple[np.ndarray, pd.DataFrame]] = {}
    for name, (start, end, maturity_limit) in segments.items():
        mask = labels["signal_date"].between(start, end)
        if maturity_limit is not None:
            mask &= labels["label_end_time"] < pd.Timestamp(maturity_limit)
        rows = labels[mask].reset_index(drop=True)
        if rows.empty:
            raise RuntimeError(f"empty segment {name}")
        print(f"[proto] {name} rows={len(rows)} dates={rows['signal_date'].min()}..{rows['signal_date'].max()}", flush=True)
        t0 = time.time()
        matrix, names = make_features(features, market, rows, symbol_index)
        print(f"[proto] {name} features {matrix.shape} in {time.time()-t0:.1f}s", flush=True)
        matrices[name] = (matrix, rows)

    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression

    params = dict(
        objective="mse",
        num_leaves=15,
        max_depth=4,
        learning_rate=0.03,
        min_data_in_leaf=500,
        verbosity=-1,
        seed=42,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
    )
    X_train, r_train = matrices["train"]
    X_valid, r_valid = matrices["valid"]
    y_return_train = r_train["policy_return_net"].to_numpy(dtype=np.float32)
    y_return_valid = r_valid["policy_return_net"].to_numpy(dtype=np.float32)
    print("[proto] training return model ...", flush=True)
    ret_model = lgb.train(
        params,
        lgb.Dataset(X_train, label=y_return_train),
        num_boost_round=500,
        valid_sets=[lgb.Dataset(X_valid, label=y_return_valid)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    X_cal, r_cal = matrices["calib"]
    X_test, r_test = matrices["test"]
    ret_cal = ret_model.predict(X_cal)
    ret_test = ret_model.predict(X_test)

    params_bin = dict(params)
    params_bin["objective"] = "binary"
    y_win_train = r_train["policy_win"].to_numpy(dtype=np.int16)
    y_win_valid = r_valid["policy_win"].to_numpy(dtype=np.int16)
    print("[proto] training win model ...", flush=True)
    win_model = lgb.train(
        params_bin,
        lgb.Dataset(X_train, label=y_win_train),
        num_boost_round=500,
        valid_sets=[lgb.Dataset(X_valid, label=y_win_valid)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )
    raw_cal = win_model.predict(X_cal)
    raw_test = win_model.predict(X_test)
    calibrator = LogisticRegression()
    calibrator.fit(raw_cal.reshape(-1, 1), r_cal["policy_win"].to_numpy(dtype=np.int16))
    p_cal = calibrator.predict_proba(raw_cal.reshape(-1, 1))[:, 1]
    p_test = calibrator.predict_proba(raw_test.reshape(-1, 1))[:, 1]

    test = r_test.copy()
    test["pred_return"] = ret_test
    test["pred_win_raw"] = raw_test
    test["pred_win_cal"] = p_test
    test["selected"] = (test["pred_return"] >= 0.01) & (test["pred_win_cal"] >= 0.70)
    print("[proto] test rows", len(test), "selected", int(test["selected"].sum()), flush=True)
    print(
        "[proto] selected win=%.3f ret=%.4f | all test win=%.3f ret=%.4f"
        % (
            test.loc[test["selected"], "policy_win"].mean(),
            test.loc[test["selected"], "policy_return_net"].mean(),
            test["policy_win"].mean(),
            test["policy_return_net"].mean(),
        ),
        flush=True,
    )
    # Calibration quality on test (the calibrated probability is fitted on 2019H2
    # but evaluated here for information only).
    for lo, hi in [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]:
        sub = test[(test["pred_win_cal"] >= lo) & (test["pred_win_cal"] < hi)]
        if len(sub):
            print(f"[proto] bucket {lo:.2f}-{hi:.2f}: n={len(sub)} hit={sub['policy_win'].mean():.3f} ret={sub['policy_return_net'].mean():.4f}")
    out = test[
        ["signal_date", "instrument", "pred_return", "pred_win_raw", "pred_win_cal", "selected", "policy_return_net", "policy_win"]
    ]
    out.to_parquet(args.output, index=False)
    print("[proto] saved", args.output, flush=True)


if __name__ == "__main__":
    main()
