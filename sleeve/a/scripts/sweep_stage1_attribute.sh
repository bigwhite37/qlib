#!/usr/bin/env bash
# Stage-1 round 2: attribute the engine.
#
# Stage-1 round 1 showed the rolling model contributes only ~5% of the
# information ratio and that the composite-only arm is seed-independent (IR
# 0.903, bit-identical across four seeds).  So the composite IS the strategy, and
# its three components have never been attributed.  These runs rank by a single
# component, then by each pair, with no model in the loop at all - which also
# makes every number exact rather than a seed draw.
set -u
cd /Users/shuzhenyi/code/python/qlib
R=sleeve/a/scripts/run_qrun_v2.py
C=/Volumes/lexar_4t/code/data/qlib/sleeve/a/cache
L=/Volumes/lexar_4t/code/data/qlib/sleeve/a/output/v2_research
PR=$C/v2m_tp5h40_predictions.parquet
BASE="--max-new 1 --max-weight 0.08 --max-positions 12 --risk-per-trade 0.016 --extreme-exit 0 --extreme-liquidates 0 --drawdown-tiers 0.08:0.7,0.12:0.45 --use-market-timing 0 --vol-target 0.14 --profit-target 0.075 --hold 120 --rank-blend 0 --composite-only 1"

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

run v2_att_combo3   --composite combo3
run v2_att_solovol  --composite solo_vol
run v2_att_soloamt  --composite solo_amt
run v2_att_soloma   --composite solo_ma
run v2_att_pvamt    --composite pair_vol_amt
run v2_att_pvma     --composite pair_vol_ma
run v2_att_pamtma   --composite pair_amt_ma
echo ALLDONE
