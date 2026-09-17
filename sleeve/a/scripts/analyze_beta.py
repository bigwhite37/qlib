#!/usr/bin/env python3
"""How much of the account's risk is market beta, and what would a hedge leave?

If most of the account's volatility is market exposure rather than stock
selection, then the strategy is a long-beta strategy wearing a low-volatility
costume, and the design fix is a market-neutral construction rather than more
entry-signal work.  This regresses each run's daily account return on the
market proxy the run itself recorded (market_daily.csv) and reports the beta,
the R-squared, and the residual (idiosyncratic) volatility.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def analyse(run: str) -> dict:
    d = ROOT / "output" / run
    nav = pd.read_csv(d / "nav.csv")
    date_col = "datetime" if "datetime" in nav.columns else nav.columns[0]
    nav[date_col] = pd.to_datetime(nav[date_col])
    acc = nav.set_index(date_col)["nav"].pct_change()

    mkt = pd.read_csv(d / "market_daily.csv")
    mkt = mkt.rename(columns={mkt.columns[0]: "date"})
    mkt["date"] = pd.to_datetime(mkt["date"])
    proxy = mkt.set_index("date")["proxy"].astype(float)
    mret = proxy.pct_change()

    joined = pd.concat([acc.rename("acc"), mret.rename("mkt")], axis=1).dropna()
    joined = joined[joined.index >= "2019-01-02"]
    x = joined["mkt"].to_numpy()
    y = joined["acc"].to_numpy()
    beta, alpha = np.polyfit(x, y, 1)
    resid = y - (alpha + beta * x)
    r2 = 1.0 - resid.var() / y.var()
    ann = np.sqrt(252.0)
    return {
        "run": run,
        "n": len(joined),
        "beta": beta,
        "alpha_ann": alpha * 252.0,
        "R2": r2,
        "acc_vol": y.std() * ann,
        "mkt_vol": x.std() * ann,
        "idio_vol": resid.std() * ann,
        "mkt_share_of_var": beta ** 2 * x.var() / y.var(),
        "ann_ret": (1.0 + y).prod() ** (252.0 / len(y)) - 1.0,
    }


def main(names) -> None:
    rows = [analyse(name) for name in names]
    table = pd.DataFrame(rows)
    table["IR"] = table.ann_ret / table.acc_vol
    table["hedged_IR"] = table.alpha_ann / table.idio_vol
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:] or ["v2_v_c2_42"])
