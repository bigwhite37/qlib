#!/usr/bin/env bash
# Round-24b: extend the exit grid around the new optimum.
#
# The first grid found tp 9% / hold 90 days at 11.85% CAGR, 9.82% volatility and
# an information ratio of 1.21 - the best risk-adjusted result in the project.
# This extends the target upward and the hold downward to bracket the optimum.
set -u
cd /Users/shuzhenyi/code/python/qlib
P=sleeve/a/scripts/pyrun.py
R=sleeve/a/scripts/run_qrun_v2.py
C=/Volumes/lexar_4t/code/data/qlib/sleeve/a/cache
L=/Volumes/lexar_4t/code/data/qlib/sleeve/a/output/v2_research
PR=$C/v2m_tp5h40_predictions.parquet
BASE="--composite combo3 --max-new 1 --max-weight 0.08 --max-positions 12 --risk-per-trade 0.016 --extreme-exit 0 --extreme-liquidates 0 --drawdown-tiers 0.08:0.7,0.12:0.45 --rank-blend 0.5 --use-market-timing 0 --vol-target 0.055"

run () {
  local name="$1"; shift
  python $P $R --predictions $PR --start 2019-01-02 --end 2026-09-14 \
    --output sleeve/a/output/$name --recorder $name --experiment sleeve_a_v2 \
    $BASE "$@" > $L/run_$name.log 2>&1
  echo -n "$name "
  grep -E 'cagr_ge_25pct|annual_vol_le_10pct|max_drawdown_ge_minus10pct|round_trip_win_rate' $L/run_$name.log \
    | tr -s ' ' | cut -d' ' -f3 | tr '\n' ' '
  echo
}

run v2_gt_tp0075_h60 --profit-target 0.075 --hold 60
run v2_gt_tp0120_h60 --profit-target 0.12 --hold 60
run v2_gt_tp0150_h60 --profit-target 0.15 --hold 60
run v2_gt_tp0075_h90 --profit-target 0.075 --hold 90
run v2_gt_tp0120_h90 --profit-target 0.12 --hold 90
run v2_gt_tp0150_h90 --profit-target 0.15 --hold 90
run v2_gt_tp0075_h120 --profit-target 0.075 --hold 120
run v2_gt_tp0120_h120 --profit-target 0.12 --hold 120
run v2_gt_tp0150_h120 --profit-target 0.15 --hold 120
echo ALLDONE
