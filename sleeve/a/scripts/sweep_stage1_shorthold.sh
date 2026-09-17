#!/usr/bin/env bash
# Stage-1 round 6: the short-hold corner of the exit-policy grid.
#
# Work item 2 says to optimise holding period x breadth x cost jointly.  Breadth is
# closed, but the exit-policy grid of round 24 only covered hold >= 60 days, and the
# older hold sweeps only covered hold >= 45.  The short-hold corner - where
# IR ~ IC*sqrt(BR) gains most from breadth of bets while fees scale with turnover -
# has never been searched under the current policy.  Twelve cells on one seed.
set -u
cd /Users/shuzhenyi/code/python/qlib
R=sleeve/a/scripts/run_qrun_v2.py
C=/Volumes/lexar_4t/code/data/qlib/sleeve/a/cache
L=/Volumes/lexar_4t/code/data/qlib/sleeve/a/output/v2_research
PR=$C/v2m_tp5h40_predictions.parquet
BASE="--composite combo3 --max-new 1 --max-weight 0.08 --max-positions 12 --risk-per-trade 0.016 --extreme-exit 0 --extreme-liquidates 0 --drawdown-tiers 0.08:0.7,0.12:0.45 --use-market-timing 0 --vol-target 0.14 --rank-blend 0.5"

run () {
  local name="$1"; shift
  scripts/run_python_3gb.sh $R --predictions $PR --start 2019-01-02 --end 2026-09-14 \
    --output sleeve/a/output/$name --recorder $name --experiment sleeve_a_v2 \
    $BASE "$@" > $L/run_$name.log 2>&1
  echo -n "$name "
  grep -E 'cagr_ge_25pct|annual_vol_le_10pct|max_drawdown_ge_minus10pct|round_trip_win_rate' $L/run_$name.log \
    | tr -s ' ' | cut -d' ' -f3 | tr '\n' ' '
  echo
}

run v2_sh_tp0020_h10 --profit-target 0.02 --hold 10
run v2_sh_tp0030_h10 --profit-target 0.03 --hold 10
run v2_sh_tp0040_h10 --profit-target 0.04 --hold 10
run v2_sh_tp0050_h10 --profit-target 0.05 --hold 10
run v2_sh_tp0020_h20 --profit-target 0.02 --hold 20
run v2_sh_tp0030_h20 --profit-target 0.03 --hold 20
run v2_sh_tp0040_h20 --profit-target 0.04 --hold 20
run v2_sh_tp0050_h20 --profit-target 0.05 --hold 20
run v2_sh_tp0020_h30 --profit-target 0.02 --hold 30
run v2_sh_tp0030_h30 --profit-target 0.03 --hold 30
run v2_sh_tp0040_h30 --profit-target 0.04 --hold 30
run v2_sh_tp0050_h30 --profit-target 0.05 --hold 30
echo ALLDONE
