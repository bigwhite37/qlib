#!/usr/bin/env python3
"""Do independent sub-books combine into a higher information ratio?

Every configuration in this project is the *same* signal (combo3 blended 50/50 with
the rolling model); what differs is the exit policy, the exposure rule and the
blend weight.  This script takes the recorded daily NAV of each configuration
family, averages it over the nine model seeds (which is itself a first level of
ensembling), and then measures:

  * each family's own CAGR / vol / max drawdown / IR,
  * the correlation matrix of their daily returns,
  * the information ratio of equal-weight combinations.

If sub-book diversification is the missing multiplier, combinations must beat
their best member on IR.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEEDS = ["42", "7", "2024", "11", "23", "101", "777", "31337", "5"]

# name -> run-name template with a {seed} placeholder (seed 42 uses the historical name)
FAMILIES = {
    "c2_tp7.5_h120_vt055": ("v2_v_c2_{seed}", None),
    "c3_tp6_h120_vt055": ("v2_v_c3_{seed}", "v2_v_c3_42"),
    "c1_tp9_h90_vt055": ("v2_v_c1_{seed}", "v2_v_c1_42"),
    "h120_tp3.5_h120_vt055": ("v2_f_h120_{seed}", None),
    "h90_tp3.5_h90_vt055": ("v2_f_h90_{seed}", None),
    "z1_vt14_tp7.5_h120": ("v2_z1_{seed}", None),
    "r120_vt14_tp3.5_h120": ("v2_g_r120_{seed}", None),
    "m120_vt09_tp7.5_h120": ("v2_g_m120_{seed}", None),
    "g4_ladder_tp3.5_h45": ("v2_sd_g4_{seed}", "v2_g4"),
    "notm_tp3.5_h45": ("v2_sa_notm_{seed}", None),
    "y4_compliance_ladder": ("v2_sa_y4_{seed}", "v2_y4"),
    "sc_d_notm_vt11": ("v2_sc_d_{seed}", None),
    "sd2_h_gross062": ("v2_sd2_h_{seed}", None),
    "b35_blend035": ("v2_b35_{seed}", None),
    "b65_blend065": ("v2_b65_{seed}", None),
}


def load_run(name: str) -> "pd.Series | None":
    path = ROOT / "output" / name / "nav.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    col = "datetime" if "datetime" in frame.columns else frame.columns[0]
    frame[col] = pd.to_datetime(frame[col])
    series = frame.set_index(col)["nav"].astype(float)
    series = series[series.index >= "2019-01-02"]
    return series.pct_change().dropna()


def family_series(template: str, seed42: "str | None"):
    parts = []
    for seed in SEEDS:
        name = seed42 if (seed == "42" and seed42) else template.format(seed=seed)
        series = load_run(name)
        if series is not None:
            parts.append(series.rename(seed))
    if not parts:
        return None
    frame = pd.concat(parts, axis=1)
    return frame.mean(axis=1)


def stats(returns: "pd.Series") -> dict:
    ann = np.sqrt(252.0)
    years = len(returns) / 252.0
    total = float((1.0 + returns).prod())
    cagr = total ** (1.0 / years) - 1.0
    vol = float(returns.std() * ann)
    nav = (1.0 + returns).cumprod()
    mdd = float((nav / nav.cummax() - 1.0).min())
    return {
        "cagr": cagr,
        "vol": vol,
        "mdd": mdd,
        "ir": cagr / vol if vol else float("nan"),
        "ratio": abs(mdd) / vol if vol else float("nan"),
    }


def main() -> None:
    series = {}
    for name, (template, seed42) in FAMILIES.items():
        got = family_series(template, seed42)
        if got is not None:
            series[name] = got
    frame = pd.DataFrame(series).dropna()
    rows = [{"family": name, **stats(frame[name])} for name in frame.columns]
    table = pd.DataFrame(rows).set_index("family").sort_values("ir", ascending=False)
    print("== sub-book properties (seed-mean daily returns) ==")
    print(table.round(4).to_string())

    print()
    print("== correlation of daily returns ==")
    corr = frame.corr()
    print(corr.round(2).to_string())

    print()
    print("== equal-weight combinations ==")
    combos = []
    names = list(frame.columns)
    for k in (2, 3, 4):
        for combo in itertools.combinations(names, k):
            blended = frame[list(combo)].mean(axis=1)
            entry = stats(blended)
            entry["members"] = " + ".join(combo)
            entry["k"] = k
            entry["mean_pairwise_corr"] = float(
                np.mean([corr.loc[a, b] for a, b in itertools.combinations(combo, 2)])
            )
            combos.append(entry)
    combos_table = pd.DataFrame(combos).sort_values("ir", ascending=False)
    print("best 12 combinations by IR:")
    print(combos_table.head(12).round(4).to_string(index=False))
    print()
    best_single = table.iloc[0]
    print("best single family IR: %.4f (%s)" % (best_single.ir, table.index[0]))
    print("best combination IR : %.4f" % combos_table.iloc[0].ir)


if __name__ == "__main__":
    main()
