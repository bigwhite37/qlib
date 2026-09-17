#!/usr/bin/env bash
# Round-26: risk-consistent exits (design change).
#
# The profit target is a flat percentage, so it means different things for
# different names: 7.5% is half a daily standard deviation for a quiet name and a
# fifth for a volatile one.  This sweep expresses the target in multiples of the
# position's own ATR at entry, which is a change to the strategy design rather
# than a parameter value.  Base is c2; the ATR multiples are chosen so that the
# average target lands near 7.5%.
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

run v2_a_atr2   --profit-atr-mult 2.0
run v2_a_atr3   --profit-atr-mult 3.0
run v2_a_atr4   --profit-atr-mult 4.0
run v2_a_atr6   --profit-atr-mult 6.0
run v2_a_atr3f5 --profit-atr-mult 3.0 --profit-atr-floor 0.05
run v2_a_atr4c15 --profit-atr-mult 4.0 --profit-atr-cap 0.15
echo ALLDONE
