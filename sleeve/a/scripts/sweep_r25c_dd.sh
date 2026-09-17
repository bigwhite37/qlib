#!/usr/bin/env bash
# Round-25c: re-tune the account drawdown overlay under the new policy.
#
# The 0.08/0.12 tiers were chosen under the old exit policy.  With c2 the
# realised drawdown is -13.5% to -16%, so the overlay is active and its shape
# matters; the contract's -10% floor is the remaining check that is far away.
set -u
cd /Users/shuzhenyi/code/python/qlib
P=sleeve/a/scripts/pyrun.py
R=sleeve/a/scripts/run_qrun_v2.py
C=/Volumes/lexar_4t/code/data/qlib/sleeve/a/cache
L=/Volumes/lexar_4t/code/data/qlib/sleeve/a/output/v2_research
PR=$C/v2m_tp5h40_predictions.parquet
BASE="--composite combo3 --max-new 1 --max-weight 0.08 --max-positions 12 --risk-per-trade 0.016 --extreme-exit 0 --extreme-liquidates 0 --rank-blend 0.5 --use-market-timing 0 --vol-target 0.055 --profit-target 0.075 --hold 120"

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

run v2_y_d1 --drawdown-tiers 0.06:0.6,0.10:0.35
run v2_y_d2 --drawdown-tiers 0.10:0.7,0.15:0.4
run v2_y_d3 --drawdown-tiers 0.08:0.6,0.13:0.3
run v2_y_d4 --drawdown-tiers 0.05:0.5,0.08:0.25,0.12:0.1
run v2_y_d5 --risk-trim-mode close_weakest
echo ALLDONE
