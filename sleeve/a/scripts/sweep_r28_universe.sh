#!/usr/bin/env bash
# Round-28: re-open the universe as a design axis.
#
# The strategy lives in small, illiquid, below-MA60 names.  The liquidity floor
# was swept once under the old exit policy (round 18: 0.10 -> 13.4% CAGR,
# 0.35 -> 4.6%), and never re-opened since the exit redesign.  This is the first
# test of whether a different *universe* (a different design, not a parameter)
# produces a better information ratio.
set -u
cd /Users/shuzhenyi/code/python/qlib
P=sleeve/a/scripts/pyrun.py
R=sleeve/a/scripts/run_qrun_v2.py
C=/Volumes/lexar_4t/code/data/qlib/sleeve/a/cache
L=/Volumes/lexar_4t/code/data/qlib/sleeve/a/output/v2_research
PR=$C/v2m_tp5h40_predictions.parquet
BASE="--composite combo3 --max-new 1 --max-weight 0.08 --max-positions 12 --risk-per-trade 0.016 --extreme-exit 0 --extreme-liquidates 0 --drawdown-tiers 0.08:0.7,0.12:0.45 --rank-blend 0.5 --use-market-timing 0 --vol-target 0.055 --profit-target 0.075 --hold 120"

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

run v2_un20 --min-amount-rank 0.20
run v2_un35 --min-amount-rank 0.35
run v2_un50 --min-amount-rank 0.50
run v2_un65 --min-amount-rank 0.65
run v2_un80 --min-amount-rank 0.80
echo ALLDONE
