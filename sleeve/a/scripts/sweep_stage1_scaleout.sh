#!/usr/bin/env bash
# Stage-1 round 8: scale-out - the last untested exit design.
#
# Exits are the only place this project has ever found a gain (rounds 23-24: target
# 3.5% -> 7.5% and cap 45 -> 120 days were worth +2.5pp of CAGR and +8pp of win
# rate).  Scale-out is the standard remaining exit design: bank part of the
# position at the first target and let the rest run to a second one.  A zero
# control must reproduce the incumbent bit for bit.
set -u
cd /Users/shuzhenyi/code/python/qlib
R=sleeve/a/scripts/run_qrun_v2.py
C=/Volumes/lexar_4t/code/data/qlib/sleeve/a/cache
L=/Volumes/lexar_4t/code/data/qlib/sleeve/a/output/v2_research
PR=$C/v2m_tp5h40_predictions.parquet
BASE="--composite combo3 --max-new 1 --max-weight 0.08 --max-positions 12 --risk-per-trade 0.016 --extreme-exit 0 --extreme-liquidates 0 --drawdown-tiers 0.08:0.7,0.12:0.45 --use-market-timing 0 --vol-target 0.14 --profit-target 0.075 --hold 120 --rank-blend 0.5"

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

run v2_so_off   --scale-out 0
run v2_so_h_t15 --scale-out 0.5 --profit-target-2 0.15
run v2_so_h_t25 --scale-out 0.5 --profit-target-2 0.25
run v2_so_h_t40 --scale-out 0.5 --profit-target-2 0.40
run v2_so_t_t25 --scale-out 0.34 --profit-target-2 0.25
run v2_so_q_t25 --scale-out 0.75 --profit-target-2 0.25
echo ALLDONE
