#!/usr/bin/env bash
# Stage-1 round 2d: vary the measure that actually carries the strategy.
#
# The replicated attribution says the composite's engine is the liquidity
# component (removing it costs 4.6-9.2pp of CAGR on 0/9 seeds) and that low
# volatility contributes little.  So the object worth varying is the liquidity
# measure itself.  Single-seed screen first; the survivors get nine seeds.
set -u
cd /Users/shuzhenyi/code/python/qlib
R=sleeve/a/scripts/run_qrun_v2.py
C=/Volumes/lexar_4t/code/data/qlib/sleeve/a/cache
L=/Volumes/lexar_4t/code/data/qlib/sleeve/a/output/v2_research
PR=$C/v2m_tp5h40_predictions.parquet
BASE="--max-new 1 --max-weight 0.08 --max-positions 12 --risk-per-trade 0.016 --extreme-exit 0 --extreme-liquidates 0 --drawdown-tiers 0.08:0.7,0.12:0.45 --use-market-timing 0 --vol-target 0.14 --profit-target 0.075 --hold 120"

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

run v2_liq_amt20   --composite solo_amt   --rank-blend 0.2
run v2_liq_amt5    --composite solo_amt5  --rank-blend 0.2
run v2_liq_amt60   --composite solo_amt60 --rank-blend 0.2
run v2_liq_ratio   --composite solo_amtratio --rank-blend 0.2
run v2_liq_amihud  --composite solo_amihud   --rank-blend 0.2
run v2_liq_amt5ma  --composite pair_amt5_ma  --rank-blend 0.2
run v2_liq_ratiov  --composite pair_amt_ratio_vol --rank-blend 0.2
echo ALLDONE
