#!/usr/bin/env bash
# Stage-1 round 4: release the slot when a holding falls out of favour.
#
# The book is slot-constrained (11.73 of 12 slots full while 3044 candidates pass
# the gate every day), so the one structural lever left is *which* names occupy the
# slots.  The rank exit - leave a position once its blended score drops out of the
# top quantile - exists in the config (default True!) and is implemented in the
# older portfolio.evaluate_exits, but the V2 live path never read it, so it has
# been silently inert in all ~520 recorded runs.  It is implemented in the live
# path here and tested against a zero control.
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

run v2_rk_off      --exit-rank 0
run v2_rk_q50      --exit-rank 1 --rank-exit-pct 0.50 --rank-exit-confirm 2 --rank-exit-min-hold 3
run v2_rk_q30      --exit-rank 1 --rank-exit-pct 0.30 --rank-exit-confirm 2 --rank-exit-min-hold 3
run v2_rk_q70      --exit-rank 1 --rank-exit-pct 0.70 --rank-exit-confirm 2 --rank-exit-min-hold 3
run v2_rk_q30c5    --exit-rank 1 --rank-exit-pct 0.30 --rank-exit-confirm 5 --rank-exit-min-hold 5
run v2_rk_q50c10   --exit-rank 1 --rank-exit-pct 0.50 --rank-exit-confirm 10 --rank-exit-min-hold 10
echo ALLDONE
