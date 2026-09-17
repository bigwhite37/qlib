#!/usr/bin/env python3
"""Write "partner ranker" files so a third opinion can join the entry consensus.

The entry score is currently 0.5 * rank(model) + 0.5 * rank(combo3).  A rank blend
is not a portfolio blend: it selects names that both rankers agree on, which is why
it is far stronger than either member (the members' own daily returns correlate
0.96).  If agreement is the mechanism, a third, different opinion should sharpen it.

Each partner file has the same schema the account runner expects for a second
prediction file: datetime / symbol_index / pred_rel.  pred_rel is written negated
for factors where LOW values are good (the same convention combo3 uses), so that a
high rank always means a better candidate.
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
from lowvol_trend.v2 import v2_base_mask  # noqa: E402
from lowvol_trend.v2_data import build_v2_features  # noqa: E402
from lowvol_trend.v2_strategy import RuleFrames  # noqa: E402

OUT = Path("/Volumes/lexar_4t/code/data/qlib/sleeve/a/cache/partners")

# name -> (RuleFrames component, negate so that high rank = good)
PARTNERS = {
    "ivol60": ("ivol60", True),
    "maxret20": ("maxret20", True),
    "mom20_reversal": ("mom20", True),
    "dd252": ("dd252", True),
    "semidev20": ("semidev20", True),
    "skew60": ("skew60", True),
    "pos_days20": ("pos_days20", True),
    "beta60": ("beta60", True),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(str(ROOT / "configs" / "v0.yaml"))
    init_local_qlib(cfg)
    with DuckDBPanelLoader(cfg, use_cache=True) as loader:
        panel = loader.load(refresh_cache=False)
    features = build_v2_features(panel, cfg)
    base = v2_base_mask(features, cfg)
    dates = features.dates
    symbols = np.asarray(features.symbols)
    frames = RuleFrames(features, base)
    if not hasattr(frames, "mom20"):
        pass

    n_dates, n_symbols = base.shape
    for name, (component, negate) in PARTNERS.items():
        try:
            values = frames.values(component)
        except Exception as exc:  # pragma: no cover - diagnostics
            print("[partner] %-16s FAILED: %s" % (name, exc), flush=True)
            continue
        if component == "mom20":
            # RuleFrames implements mom60/dist_ma120 but not mom20; fall back to
            # the 20-day reversal computed from the close frame.
            close = frames.close
            values = (close / close.shift(20) - 1.0).to_numpy(dtype=np.float32)
        mask = base & np.isfinite(values)
        t_idx, j_idx = np.nonzero(mask)
        rel = values[t_idx, j_idx].astype(np.float32)
        if negate:
            rel = -rel
        frame = pd.DataFrame(
            {
                "datetime": dates.to_numpy()[t_idx],
                "symbol_index": j_idx.astype(np.int64),
                "pred_rel": rel,
                "pred_win": np.float32(1.0),
                "pred_return": np.float32(0.0),
            }
        )
        path = OUT / ("partner_%s.parquet" % name)
        frame.to_parquet(path, index=False)
        print("[partner] %-16s rows %9d -> %s" % (name, len(frame), path.name), flush=True)
        del frame, rel, values
    del frames
    print("[partner] done", flush=True)


if __name__ == "__main__":
    main()
