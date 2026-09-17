#!/usr/bin/env python3
"""Rolling quarterly V2 predictions, written to stay inside a 3 GB budget.

Protocol (docs/chats_002.md section 8.2) for every target quarter:

    train = previous 36 months (labels matured before the validation window)
    valid = previous 6 months (early stopping on cross-sectional rank IC / AUC)
    calib = the 6 months before that (probability calibration only)
    target = the quarter itself

The cached feature matrix is read block by block (never memory-mapped as a
whole) and the label table is loaded without its object columns, so peak RSS
stays well under the cap.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lowvol_trend.bootstrap import ensure_local_qlib  # noqa: E402

ensure_local_qlib()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

NPY_HEADER = 128
LABEL_COLUMNS = [
    "signal_date",
    "signal_index",
    "symbol_index",
    "entry_filled",
    "net_return",
    "exit_date",
    "matrix_row",
]


class MatrixReader:
    """Read feature rows from the per-chunk part files written by the builder."""

    def __init__(self, base_dir: Path, meta: dict):
        self.parts = [
            (int(part["start_row"]), int(part["end_row"]), base_dir / part["file"]) for part in meta["parts"]
        ]
        self.n_cols = int(meta["n_features"])

    def read(self, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        out = np.empty((len(rows), self.n_cols), dtype=np.float32)
        order = np.argsort(rows, kind="stable")
        sorted_rows = rows[order]
        cursor = 0
        for start_row, end_row, path in self.parts:
            left = int(np.searchsorted(sorted_rows, start_row, side="left"))
            right = int(np.searchsorted(sorted_rows, end_row, side="left"))
            if right <= left:
                continue
            block = np.load(path)
            local = sorted_rows[left:right] - start_row
            out[order[left:right]] = block[local]
            del block
            cursor = right
        if cursor != len(rows):
            missing = len(rows) - cursor
            raise RuntimeError(f"{missing} rows are outside the cached feature parts")
        return out


def relevance_buckets(values: np.ndarray, groups: np.ndarray, levels: int) -> np.ndarray:
    """Cross-sectional relevance buckets in [0, levels - 1] for lambdarank."""

    pct = pd.Series(values).groupby(pd.Series(groups)).rank(pct=True).to_numpy()
    return np.clip((pct * levels).astype(np.int32), 0, levels - 1)


def quarters(start: str, end: str) -> List[pd.Period]:
    begin = pd.Period(pd.Timestamp(start), freq="Q")
    last = pd.Period(pd.Timestamp(end), freq="Q")
    out: List[pd.Period] = []
    period = begin
    while period <= last:
        out.append(period)
        period += 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="/Volumes/lexar_4t/code/data/qlib/sleeve/a/cache/v2m_tp5")
    parser.add_argument("--output", default="/Volumes/lexar_4t/code/data/qlib/sleeve/a/cache/v2m_tp5_predictions.parquet")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-09-14")
    parser.add_argument("--train-months", type=int, default=36)
    parser.add_argument("--valid-months", type=int, default=6)
    parser.add_argument("--calib-months", type=int, default=6)
    parser.add_argument("--max-rounds", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--max-train-rows", type=int, default=700_000)
    parser.add_argument("--max-valid-rows", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank-label", default="policy", choices=["policy", "fwd10", "fwd20"],
                        help="target of the ranking model")
    # Model-shape switches.  The first release of this builder always early
    # stopped the return model on a 6-month validation rank IC, which in 9 of 31
    # quarters selected iteration 1..3 out of 300 - i.e. the "model" that the
    # account blends against was often a single split.  These switches make the
    # training length and the objective explicit so both can be measured.
    parser.add_argument("--objective", default="mse", choices=["mse", "lambdarank"],
                        help="return-model objective; lambdarank optimises the ordering directly")
    parser.add_argument("--fixed-rounds", type=int, default=0,
                        help="train exactly this many rounds and skip early stopping (0 = early stop)")
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--min-data-in-leaf", type=int, default=300)
    parser.add_argument("--feature-fraction", type=float, default=0.7)
    parser.add_argument("--relevance-levels", type=int, default=5,
                        help="lambdarank: number of cross-sectional relevance buckets")
    parser.add_argument("--truncation", type=int, default=200,
                        help="lambdarank: only the top N per date contribute to the gradient")
    args = parser.parse_args()
    prefix = Path(args.prefix)
    meta = json.loads(prefix.with_name(prefix.name + "_features.json").read_text(encoding="utf-8"))
    names: List[str] = meta["feature_names"]
    n_cols = len(names)
    reader = MatrixReader(prefix.parent, meta)
    t0 = time.time()
    label_path = prefix.with_name(prefix.name + "_labels.npz")
    with np.load(label_path) as handle:
        dates_all = handle["dates"]
        signal_index_all = handle["signal_index"].astype(np.int64)
        symbol_index_all = handle["symbol_index"].astype(np.int64)
        entry_filled_all = handle["entry_filled"]
        exit_fill_all = handle["exit_fill_index"].astype(np.int64)
        net_return_all = handle["net_return"].astype(np.float32)
    usable = entry_filled_all & np.isfinite(net_return_all)
    matrix_row = np.flatnonzero(usable).astype(np.int64)
    signal_date = dates_all[signal_index_all[usable]]
    exit_date = np.full(len(matrix_row), np.datetime64("NaT"), dtype="datetime64[ns]")
    has_exit = exit_fill_all[usable] >= 0
    exit_date[has_exit] = dates_all[exit_fill_all[usable][has_exit]]
    net_return = net_return_all[usable]
    symbol_index = symbol_index_all[usable]
    signal_index = signal_index_all[usable]
    # the design defines the win label as 1[net return > 0]
    win = (net_return > 0).astype(np.float32)
    n_rows = len(matrix_row)
    fwd_rank = None
    if args.rank_label != "policy":
        with np.load(prefix.with_name(prefix.name + "_fwd.npz")) as handle:
            fwd_all = handle[args.rank_label].astype(np.float32)
        fwd_rank = fwd_all[matrix_row]
        print(f"[pred] ranking label = {args.rank_label}, finite {np.isfinite(fwd_rank).sum()}", flush=True)
    del dates_all, signal_index_all, symbol_index_all, entry_filled_all, exit_fill_all, net_return_all, usable
    print(f"[pred] labels {n_rows} usable rows loaded in {time.time()-t0:.1f}s", flush=True)
    print(f"[pred] usable label rows {n_rows}", flush=True)

    # trailing base-universe mean policy return, matured at least 60 days back
    unique_dates = np.sort(np.unique(signal_date))
    date_pos = pd.Series(np.arange(len(unique_dates)), index=unique_dates)
    rows_pos = date_pos.reindex(signal_date).to_numpy(dtype=np.int64)
    order = np.argsort(rows_pos, kind="stable")
    sorted_pos = rows_pos[order]
    sorted_ret = net_return[order].astype(np.float64)
    cum = np.concatenate([[0.0], np.cumsum(sorted_ret)])
    counts = np.concatenate([[0], np.bincount(sorted_pos, minlength=len(unique_dates))])
    start_off = np.concatenate([[0], np.cumsum(counts)])
    trailing = np.full(len(unique_dates), np.nan)
    for i in range(len(unique_dates)):
        lo, hi = max(0, i - 180), max(0, i - 60)
        a, b = start_off[lo], start_off[hi]
        if b > a:
            trailing[i] = (cum[b] - cum[a]) / (b - a)
    trailing = pd.Series(trailing, index=unique_dates).ffill().fillna(0.0)
    del order, sorted_pos, sorted_ret, cum, counts, start_off
    print("[pred] trailing base mean ready", flush=True)

    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression

    params = dict(
        num_leaves=args.num_leaves,
        max_depth=5,
        learning_rate=args.learning_rate,
        min_data_in_leaf=args.min_data_in_leaf,
        feature_fraction=args.feature_fraction,
        bagging_fraction=0.8,
        bagging_freq=1,
        verbosity=-1,
        seed=args.seed,
        num_threads=6,
        max_bin=127,
    )
    preds: List[pd.DataFrame] = []
    metrics: List[dict] = []
    for q in quarters(args.start, args.end):
        q_start = q.start_time
        q_end = min(q.end_time, pd.Timestamp(args.end))
        v_start = q_start - pd.DateOffset(months=args.valid_months)
        c_start = v_start - pd.DateOffset(months=args.calib_months)
        t_start = c_start - pd.DateOffset(months=args.train_months)
        train_mask = (signal_date >= np.datetime64(t_start)) & (signal_date < np.datetime64(c_start)) & (
            exit_date < np.datetime64(v_start)
        )
        valid_mask = (signal_date >= np.datetime64(v_start)) & (signal_date < np.datetime64(q_start)) & (
            exit_date < np.datetime64(q_start)
        )
        calib_mask = (signal_date >= np.datetime64(c_start)) & (signal_date < np.datetime64(v_start)) & (
            exit_date < np.datetime64(q_start)
        )
        target_mask = (signal_date >= np.datetime64(q_start)) & (signal_date <= np.datetime64(q_end))
        tr_rows = np.flatnonzero(train_mask)
        va_rows = np.flatnonzero(valid_mask)
        ca_rows = np.flatnonzero(calib_mask)
        te_rows = np.flatnonzero(target_mask)
        if len(tr_rows) < 5000 or len(te_rows) == 0:
            print(f"[pred] {q}: skipped train={len(tr_rows)} target={len(te_rows)}", flush=True)
            continue
        rng = np.random.default_rng(args.seed + q.ordinal)
        if len(tr_rows) > args.max_train_rows:
            tr_rows = np.sort(rng.choice(tr_rows, size=args.max_train_rows, replace=False))
        if len(va_rows) > args.max_valid_rows:
            va_rows = np.sort(rng.choice(va_rows, size=args.max_valid_rows, replace=False))
        Xtr = reader.read(matrix_row[tr_rows])
        Xva = reader.read(matrix_row[va_rows])
        Xca = reader.read(matrix_row[ca_rows])
        Xte = reader.read(matrix_row[te_rows])
        if fwd_rank is None:
            y_tr = net_return[tr_rows].astype(np.float64)
            y_va = net_return[va_rows].astype(np.float64)
            g_tr = signal_date[tr_rows]
            g_va = signal_date[va_rows]
            y_rel = (y_tr - pd.Series(y_tr).groupby(g_tr).transform("mean").to_numpy()).astype(np.float32)
            va_rel = (y_va - pd.Series(y_va).groupby(g_va).transform("mean").to_numpy()).astype(np.float32)
        else:
            # the forward label is already cross-sectionally demeaned, so it can
            # be used directly as the relative-return target
            y_rel = fwd_rank[tr_rows]
            va_rel = fwd_rank[va_rows]
            g_va = signal_date[va_rows]
        va_rel_rank = pd.Series(va_rel).groupby(pd.Series(g_va)).rank().to_numpy()
        groups = pd.Series(g_va)

        def rank_ic(preds_in, dataset):
            frame = pd.DataFrame({"p": preds_in})
            ic = frame.groupby(groups).apply(
                lambda s: s["p"].rank().corr(pd.Series(va_rel_rank[s.index])), include_groups=False
            )
            return "rank_ic", float(ic.mean()), True

        if args.objective == "lambdarank":
            # Relevance buckets from the cross-sectional rank of the training
            # target, with the rows grouped by signal date (LightGBM requires
            # the data sorted by group).
            rel_tr = relevance_buckets(y_rel, g_tr, args.relevance_levels)
            rel_va = relevance_buckets(va_rel, g_va, args.relevance_levels)
            order_tr = np.argsort(g_tr, kind="stable")
            order_va = np.argsort(g_va, kind="stable")
            _, sizes_tr = np.unique(g_tr[order_tr], return_counts=True)
            _, sizes_va = np.unique(g_va[order_va], return_counts=True)
            train_set = lgb.Dataset(Xtr[order_tr], label=rel_tr[order_tr], group=sizes_tr)
            valid_set = lgb.Dataset(
                Xva[order_va], label=rel_va[order_va], group=sizes_va, reference=train_set
            )
            del order_tr, order_va, rel_tr, rel_va
            ret_model = lgb.train(
                dict(
                    params,
                    objective="lambdarank",
                    lambdarank_truncation_level=args.truncation,
                    metric="ndcg",
                    ndcg_eval_at=[10, 50],
                ),
                train_set,
                num_boost_round=args.fixed_rounds or args.max_rounds,
                valid_sets=[valid_set],
                callbacks=(
                    [lgb.log_evaluation(0)]
                    if args.fixed_rounds
                    else [lgb.early_stopping(args.patience, verbose=False), lgb.log_evaluation(0)]
                ),
            )
            del train_set, valid_set
        elif args.fixed_rounds:
            ret_model = lgb.train(
                dict(params, objective="mse"),
                lgb.Dataset(Xtr, label=y_rel),
                num_boost_round=args.fixed_rounds,
                callbacks=[lgb.log_evaluation(0)],
            )
        else:
            ret_model = lgb.train(
                dict(params, objective="mse"),
                lgb.Dataset(Xtr, label=y_rel),
                num_boost_round=args.max_rounds,
                valid_sets=[lgb.Dataset(Xva, label=va_rel)],
                feval=rank_ic,
                callbacks=[lgb.early_stopping(args.patience, verbose=False), lgb.log_evaluation(0)],
            )
        win_model = lgb.train(
            dict(params, objective="binary"),
            lgb.Dataset(Xtr, label=win[tr_rows]),
            num_boost_round=args.max_rounds,
            valid_sets=[lgb.Dataset(Xva, label=win[va_rows])],
            callbacks=[lgb.early_stopping(args.patience, verbose=False), lgb.log_evaluation(0)],
        )
        ca_win = win[ca_rows].astype(np.int8)
        raw_ca = win_model.predict(Xca).reshape(-1, 1)
        if len(np.unique(ca_win)) > 1:
            calibrator = LogisticRegression(max_iter=2000).fit(raw_ca, ca_win)
            p_win = calibrator.predict_proba(win_model.predict(Xte).reshape(-1, 1))[:, 1]
        else:
            p_win = np.full(len(te_rows), float(ca_win.mean()) if len(ca_win) else np.nan)
        pred_rel = ret_model.predict(Xte).astype(np.float32)
        te_dates = signal_date[te_rows]
        base_mean = trailing.reindex(pd.DatetimeIndex(te_dates)).to_numpy(dtype=np.float32)
        frame = pd.DataFrame(
            {
                "datetime": te_dates,
                "symbol_index": symbol_index[te_rows],
                "pred_rel": pred_rel,
                "pred_win": p_win.astype(np.float32),
                "pred_return": (pred_rel + base_mean).astype(np.float32),
            }
        )
        frame["q_pred_rel"] = frame.groupby("datetime")["pred_rel"].rank(pct=True).astype(np.float32)
        realised = net_return[te_rows]
        realised_win = win[te_rows]
        top = frame["q_pred_rel"].to_numpy() >= 0.9
        bottom = frame["q_pred_rel"].to_numpy() <= 0.1
        metrics.append(
            {
                "quarter": str(q),
                "n_train": len(tr_rows),
                "n_valid": len(va_rows),
                "n_calib": len(ca_rows),
                "n_target": len(te_rows),
                "best_iter_ret": int(ret_model.best_iteration or ret_model.current_iteration() or 0),
                "best_iter_win": int(win_model.best_iteration or 0),
                "target_win": float(realised_win.mean()),
                "target_net": float(realised.mean()),
                "top_decile_win": float(realised_win[top].mean()) if top.any() else float("nan"),
                "top_decile_net": float(realised[top].mean()) if top.any() else float("nan"),
                "bottom_decile_win": float(realised_win[bottom].mean()) if bottom.any() else float("nan"),
                "bottom_decile_net": float(realised[bottom].mean()) if bottom.any() else float("nan"),
            }
        )
        print(
            f"[pred] {q} train={len(tr_rows)} target={len(te_rows)} iters=({ret_model.best_iteration},{win_model.best_iteration}) "
            f"top10 win={metrics[-1]['top_decile_win']:.3f} net={metrics[-1]['top_decile_net']:+.4f} | "
            f"bot10 win={metrics[-1]['bottom_decile_win']:.3f} net={metrics[-1]['bottom_decile_net']:+.4f} | "
            f"all win={metrics[-1]['target_win']:.3f} net={metrics[-1]['target_net']:+.4f}",
            flush=True,
        )
        preds.append(frame)
        del Xtr, Xva, Xca, Xte, ret_model, win_model
    out = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()
    out.to_parquet(args.output, index=False)
    pd.DataFrame(metrics).to_csv(Path(args.output).with_suffix(".quarters.csv"), index=False)
    print("[pred] saved", args.output, out.shape, flush=True)


if __name__ == "__main__":
    main()
