#!/usr/bin/env bash
# Stage-1 round 4b: risk-balanced re-weighting of the held sleeve.
#
# Work item 3 of the target prompt.  New positions are sized inversely to
# volatility, but existing positions are never re-sized, so the book drifts and its
# risk ends up dominated by whichever names have run.  This pulls every held weight
# toward the inverse-vol target at the same total gross, within the per-name caps.
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

run v2_rp_off  --risk-parity 0
run v2_rp_s25  --risk-parity 1 --risk-parity-strength 0.25
run v2_rp_s50  --risk-parity 1 --risk-parity-strength 0.50
run v2_rp_s100 --risk-parity 1 --risk-parity-strength 1.00
echo ALLDONE
