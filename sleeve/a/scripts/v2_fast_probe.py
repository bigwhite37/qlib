#!/usr/bin/env python3
"""Fast probe: can a LightGBM ranking model create policy-level edge?

Train on 2016-2018 policy labels with a compact feature set, predict a test
year, then push the model's top-ranked subsets through the real policy
simulator.  This is the decisive experiment before investing in the full
Alpha158 rolling pipeline.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

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

from lowvol_trend.config import load_config  # noqa: E402
from lowvol_trend.data import DuckDBPanelLoader  # noqa: E402
from lowvol_trend.features import build_features  # noqa: E402
from lowvol_trend.v2 import compute_v2_market_state, v2_base_mask  # noqa: E402
from lowvol_trend.v2_policy import ExitPolicy, policy_stats, simulate_entries  # noqa: E402


class FeatureBank:
    def __init__(self, features, market):
        self.features = features
        self.market = market
        panel = features.panel
        close = pd.DataFrame(features.arr("close"), index=panel.dates, columns=panel.symbols, copy=False)
        volume = pd.DataFrame(panel.field("volume"), index=panel.dates, columns=panel.symbols, copy=False)
        high = pd.DataFrame(panel.field("high"), index=panel.dates, columns=panel.symbols, copy=False)
        low = pd.DataFrame(panel.field("low"), index=panel.dates, columns=panel.symbols, copy=False)
        proxy = pd.Series(market.proxy, index=panel.dates)
        self.stock: Dict[str, np.ndarray] = {}
        for n in (1, 3, 5, 10, 20, 60, 120):
            self.stock[f"ret{n}"] = close.pct_change(n, fill_method=None).to_numpy(dtype=np.float32)
        m5 = (proxy / proxy.shift(5) - 1.0).to_numpy(dtype=np.float32)
        m20 = (proxy / proxy.shift(20) - 1.0).to_numpy(dtype=np.float32)
        m60 = (proxy / proxy.shift(60) - 1.0).to_numpy(dtype=np.float32)
        self.stock["rel5"] = self.stock["ret5"] - m5[:, None]
        self.stock["rel20"] = self.stock["ret20"] - m20[:, None]
        self.stock["rel60"] = self.stock["ret60"] - m60[:, None]
        self.stock["dist_ma5"] = (close / close.rolling(5, min_periods=5).mean() - 1.0).to_numpy(dtype=np.float32)
        self.stock["dist_ma20"] = (close / close.rolling(20, min_periods=20).mean() - 1.0).to_numpy(dtype=np.float32)
        self.stock["dist_ma60"] = (close / close.rolling(60, min_periods=60).mean() - 1.0).to_numpy(dtype=np.float32)
        self.stock["atr_ratio"] = (features.arr("atr20") / np.where(features.arr("close") > 0, features.arr("close"), np.nan)).astype(np.float32)
        self.stock["vol20"] = close.pct_change(fill_method=None).rolling(20, min_periods=20).std().to_numpy(dtype=np.float32)
        self.stock["vol60"] = features.arr("vol60")
        self.stock["downvol20"] = features.arr("downvol20")
        self.stock["amt_rank"] = features.arr("amount_rank")
        self.stock["vol_surge"] = (volume / volume.rolling(20, min_periods=20).mean()).to_numpy(dtype=np.float32)
        hi20 = close.rolling(20, min_periods=20).max()
        lo20 = close.rolling(20, min_periods=20).min()
        self.stock["pos_range20"] = ((close - lo20) / (hi20 - lo20)).to_numpy(dtype=np.float32)
        self.stock["dist_hi60"] = (close / close.rolling(60, min_periods=60).max() - 1.0).to_numpy(dtype=np.float32)
        self.stock["dist_hi250"] = (close / close.rolling(250, min_periods=120).max() - 1.0).to_numpy(dtype=np.float32)
        ret1 = close.pct_change(fill_method=None)
        self.stock["up_ratio20"] = (ret1 > 0).rolling(20, min_periods=20).mean().to_numpy(dtype=np.float32)
        self.stock["amp20"] = ((high - low) / close).rolling(20, min_periods=20).mean().to_numpy(dtype=np.float32)
        self.stock["trend_quality"] = features.arr("trend_quality60")
        self.stock["rs60"] = features.arr("rs60")
        self.stock["skew20"] = ret1.rolling(20, min_periods=20).skew().to_numpy(dtype=np.float32)
        self.stock["log_amount"] = np.log(np.clip(panel.amount_yuan().astype(np.float64), 1.0, None)).astype(np.float32)
        self.market_feats = {
            "mkt_ret5": m5,
            "mkt_ret20": m20,
            "mkt_ret60": m60,
            "mkt_b20": market.breadth_20.astype(np.float32),
            "mkt_b60": market.breadth_60.astype(np.float32),
            "mkt_q": market.drop_diffusion.astype(np.float32),
            "mkt_cap": market.effective_cap.astype(np.float32),
            "mkt_recovery": market.recovery_phase.astype(np.float32),
        }
        self.names = list(self.stock) + list(self.market_feats)

    def matrix(self, t: np.ndarray, j: np.ndarray) -> np.ndarray:
        cols = [self.stock[k][t, j] for k in self.stock]
        cols += [np.asarray(v, dtype=np.float32)[t] for v in self.market_feats.values()]
        return np.column_stack(cols).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-start", default="2016-01-01")
    parser.add_argument("--train-end", default="2018-12-31")
    parser.add_argument("--valid-start", default="2019-01-01")
    parser.add_argument("--valid-end", default="2019-06-30")
    parser.add_argument("--calib-start", default="2019-07-01")
    parser.add_argument("--calib-end", default="2019-12-31")
    parser.add_argument("--test-start", default="2020-01-01")
    parser.add_argument("--test-end", default="2020-12-31")
    parser.add_argument("--labels", default="/Volumes/lexar_4t/code/data/qlib/sleeve/a/cache/v2_labels.parquet")
    parser.add_argument("--policy", default="chosen")
    parser.add_argument("--output", default="/Volumes/lexar_4t/code/data/qlib/sleeve/a/output/v2_probe")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(ROOT / "configs" / "v0.yaml"))
    init_local_qlib(cfg)
    with DuckDBPanelLoader(cfg, use_cache=True) as loader:
        panel = loader.load(refresh_cache=False)
    features = build_features(panel, cfg)
    base = v2_base_mask(features, cfg)
    market = compute_v2_market_state(features, base, cfg)
    bank = FeatureBank(features, market)
    print("[probe] features ready:", len(bank.names), flush=True)

    labels = pd.read_parquet(
        args.labels, columns=["signal_date", "signal_index", "instrument", "status", "policy_return_net", "policy_win"]
    )
    labels = labels[labels["status"] == "closed"].reset_index(drop=True)
    labels["signal_date"] = pd.to_datetime(labels["signal_date"])
    labels["label_end_time"] = labels["signal_date"]  # placeholder, maturity handled by segment bounds
    symbol_index = {s: j for j, s in enumerate(panel.symbols)}
    labels["j"] = labels["instrument"].map(symbol_index)
    labels = labels[labels["j"].notna()].copy()
    labels["j"] = labels["j"].astype(np.int64)

    def segment(start, end):
        m = labels["signal_date"].between(start, end)
        return labels[m].reset_index(drop=True)

    tr = segment(args.train_start, args.train_end)
    va = segment(args.valid_start, args.valid_end)
    ca = segment(args.calib_start, args.calib_end)
    print(f"[probe] train={len(tr)} valid={len(va)} calib={len(ca)}", flush=True)
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression

    t0 = time.time()
    Xtr = bank.matrix(tr["signal_index"].to_numpy(), tr["j"].to_numpy())
    Xva = bank.matrix(va["signal_index"].to_numpy(), va["j"].to_numpy())
    Xca = bank.matrix(ca["signal_index"].to_numpy(), ca["j"].to_numpy())
    print(f"[probe] matrices built in {time.time()-t0:.1f}s", flush=True)
    params = dict(
        num_leaves=31,
        max_depth=5,
        learning_rate=0.03,
        min_data_in_leaf=200,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        verbosity=-1,
        seed=42,
    )
    ytr = tr["policy_return_net"].to_numpy(dtype=np.float32)
    yva = va["policy_return_net"].to_numpy(dtype=np.float32)
    ret_model = lgb.train(
        dict(params, objective="mse"),
        lgb.Dataset(Xtr, label=ytr),
        num_boost_round=400,
        valid_sets=[lgb.Dataset(Xva, label=yva)],
        callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)],
    )
    btr = tr["policy_win"].to_numpy(dtype=np.int8)
    bva = va["policy_win"].to_numpy(dtype=np.int8)
    win_model = lgb.train(
        dict(params, objective="binary"),
        lgb.Dataset(Xtr, label=btr),
        num_boost_round=400,
        valid_sets=[lgb.Dataset(Xva, label=bva)],
        callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)],
    )
    print("[probe] best iters", ret_model.best_iteration, win_model.best_iteration, flush=True)
    raw_ca = win_model.predict(Xca).reshape(-1, 1)
    cal = LogisticRegression(max_iter=1000)
    yca = ca["policy_win"].to_numpy(dtype=np.int8)
    cal.fit(raw_ca, yca)
    p_ca = cal.predict_proba(raw_ca)[:, 1]
    print("[probe] calib base rate", float(yca.mean()), "pred mean", float(p_ca.mean()), flush=True)

    # test rows: all base rows in the test window
    t_lo = int(np.searchsorted(panel.dates, pd.Timestamp(args.test_start)))
    t_hi = int(np.searchsorted(panel.dates, pd.Timestamp(args.test_end), side="right"))
    tt, jj = np.nonzero(base[t_lo:t_hi])
    tt = tt + t_lo
    Xte = bank.matrix(tt, jj)
    p_ret = ret_model.predict(Xte)
    p_win_raw = win_model.predict(Xte).reshape(-1, 1)
    p_win = cal.predict_proba(p_win_raw)[:, 1]
    print(f"[probe] test rows={len(tt)} ret range=({p_ret.min():.4f},{p_ret.max():.4f}) win range=({p_win.min():.3f},{p_win.max():.3f})", flush=True)
    pred = pd.DataFrame({"t": tt, "j": jj, "p_ret": p_ret, "p_win": p_win, "p_win_raw": p_win_raw[:, 0]})
    pred["date"] = panel.dates[tt]
    # cross-sectional quantile of predicted return
    pred["q_ret"] = pred.groupby("date")["p_ret"].rank(pct=True)
    pred["q_win"] = pred.groupby("date")["p_win"].rank(pct=True)
    pred.to_parquet(out / "predictions.parquet", index=False)

    # informational: how do predictions relate to realised labels on overlapping rows
    lbl = labels[labels["signal_date"].between(args.test_start, args.test_end)]
    join = pred.merge(
        lbl[["signal_index", "j", "policy_return_net", "policy_win"]], left_on=["t", "j"], right_on=["signal_index", "j"], how="inner"
    )
    if len(join):
        for lo, hi in [(0.0, 0.02), (0.02, 0.10), (0.10, 0.30), (0.30, 0.70), (0.70, 1.01)]:
            sel = join[(join["q_ret"] >= 1.0 - hi) & (join["q_ret"] < 1.0 - lo)]
            if len(sel):
                print(
                    f"[probe] pred-ret bucket {lo:.2f}-{hi:.2f}: n={len(sel)} realised win={sel['policy_win'].mean():.4f} "
                    f"ret={sel['policy_return_net'].mean():+.4f}",
                    flush=True,
                )

    policies = {
        "tp5_stop_trail20": ExitPolicy(profit_target_pct=0.05),
        "tp3_h10": ExitPolicy(profit_target_pct=0.03, max_hold_days=10),
        "tp8_h30": ExitPolicy(profit_target_pct=0.08, max_hold_days=30),
        "nostop_h10": ExitPolicy(use_stop=False, use_trailing=False, max_hold_days=10),
        "tp5_nostop_h20": ExitPolicy(use_stop=False, use_trailing=False, profit_target_pct=0.05, max_hold_days=20),
    }
    extreme = market.effective_state == "extreme"
    rows: List[dict] = []
    for label, qcol, q in [
        ("top1_ret", "q_ret", 0.99),
        ("top5_ret", "q_ret", 0.95),
        ("top10_ret", "q_ret", 0.90),
        ("top20_ret", "q_ret", 0.80),
        ("top5_win", "q_win", 0.95),
        ("top10_win", "q_win", 0.90),
        ("all_base", None, None),
    ]:
        mask = np.zeros_like(base)
        if qcol is None:
            mask[t_lo:t_hi] = base[t_lo:t_hi]
        else:
            sel = pred[pred[qcol] >= q]
            mask[sel["t"].to_numpy(), sel["j"].to_numpy()] = True
        for pname, policy in policies.items():
            frame = simulate_entries(features, cfg, mask, policy, market_extreme=extreme, start_index=t_lo, end_index=t_hi)
            stats = policy_stats(frame)
            stats.update({"entry": label, "policy": pname})
            rows.append(stats)
            print(
                f"[probe] {label:12s} {pname:18s} n={stats['n']:6d} win={stats.get('win', float('nan')):.3f} "
                f"ret={stats.get('ret', float('nan')):+.4f} hold={stats.get('hold', float('nan')):.1f}",
                flush=True,
            )
    pd.DataFrame(rows).to_csv(out / "probe_results.csv", index=False)


if __name__ == "__main__":
    main()
