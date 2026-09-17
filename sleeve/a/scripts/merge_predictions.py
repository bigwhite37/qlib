#!/usr/bin/env python3
"""Average several rolling-model prediction files into one rank ensemble.

Each input file carries a per-date cross-sectional percentile rank of the
model's relative-return prediction.  Averaging those ranks (rather than the raw
scores) keeps the ensemble scale-free, which is what the account consumes: the
V2 ranking only ever uses the within-date ordering.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

KEYS = ["datetime", "symbol_index"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    frames = []
    for path in args.inputs:
        frame = pd.read_parquet(path, columns=KEYS + ["pred_rel", "pred_win", "pred_return"])
        frame["model_rank"] = frame.groupby("datetime")["pred_rel"].rank(pct=True)
        frames.append(frame.set_index(KEYS)[["model_rank", "pred_win", "pred_return"]])
        print("[merge] read", Path(path).name, len(frame), flush=True)

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.add(frame, fill_value=0.0)
    merged = merged / float(len(frames))
    merged = merged.reset_index()
    merged["pred_rel"] = merged["model_rank"].astype("float32")
    merged["pred_win"] = merged["pred_win"].astype("float32")
    merged["pred_return"] = merged["pred_return"].astype("float32")
    merged = merged.drop(columns=["model_rank"])
    merged.to_parquet(args.output, index=False)
    print("[merge] wrote", args.output, merged.shape, flush=True)


if __name__ == "__main__":
    main()
