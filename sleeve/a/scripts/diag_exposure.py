#!/usr/bin/env python3
"""Exposure diagnostics: is the book actually invested, and what caps it?

Reads constraints_daily.csv / daily_decisions.csv / market_daily.csv from one
or more run directories and reports the realized exposure profile.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main(names) -> None:
    rows = []
    for name in names:
        d = ROOT / "output" / name
        cons = pd.read_csv(d / "constraints_daily.csv", parse_dates=["datetime"])
        dec = pd.read_csv(d / "daily_decisions.csv", parse_dates=["decision_date"])
        mkt = pd.read_csv(d / "market_daily.csv")
        mkt.columns = [c if c else "date" for c in mkt.columns]
        mkt = mkt.rename(columns={mkt.columns[0]: "date"})
        cons = cons[cons.datetime >= "2019-01-02"]
        dec = dec[dec.decision_date >= "2019-01-02"]
        gross = cons.gross_exposure.to_numpy()
        npos = cons.n_positions.to_numpy()
        target = dec.target_gross.to_numpy()
        # on days the strategy wanted more than it held, it was capacity limited
        join = dec[["decision_date", "target_gross", "market_cap", "n_new"]].merge(
            cons, left_on="decision_date", right_on="datetime", how="inner"
        )
        rows.append(
            {
                "run": name,
                "gross_mean": gross.mean(),
                "gross_median": np.median(gross),
                "gross_p10": np.percentile(gross, 10),
                "gross_max": gross.max(),
                "npos_mean": npos.mean(),
                "npos_max": npos.max(),
                "days_flat": float((npos == 0).mean()),
                "days_full": float((gross > 0.9 * gross.max()).mean()) if gross.max() else 0.0,
                "target_mean": target.mean(),
                "entries_per_day": dec.n_new.mean(),
                "state_mix": {
                    s: float(v) for s, v in mkt.effective_state.value_counts(normalize=True).items()
                },
                "cap_mean": mkt.effective_cap[mkt.index >= 0].mean(),
            }
        )
        del join
    table = pd.DataFrame(rows)
    print(table.round(4).to_string(index=False))
    for r in rows:
        print(r["run"], "state mix:", {k: round(v, 3) for k, v in r["state_mix"].items()})


if __name__ == "__main__":
    main(sys.argv[1:] or ["v2_g4"])
