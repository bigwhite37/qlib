#!/usr/bin/env bash
# Round-1b: correct the ranker ablation.
#
# With --rank-blend 0.0 the blend branch is skipped entirely and the entry score
# stays the RAW model prediction, so the historical "composite-only" arm never
# used the composite at all.  --composite-only makes --rank-blend 0 mean "rank by
# the rule composite".  This run measures the true three-arm ablation.
set -u
cd /Users/shuzhenyi/code/python/qlib
R=sleeve/a/scripts/run_qrun_v2.py
C=/Volumes/lexar_4t/code/data/qlib/sleeve/a/cache
L=/Volumes/lexar_4t/code/data/qlib/sleeve/a/output/v2_research
PR=$C/v2m_tp5h40_predictions.parquet
BASE="--composite combo3 --max-new 1 --max-weight 0.08 --max-positions 12 --risk-per-trade 0.016 --extreme-exit 0 --extreme-liquidates 0 --drawdown-tiers 0.08:0.7,0.12:0.45 --use-market-timing 0 --vol-target 0.14 --profit-target 0.075 --hold 120"

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

run v2_ab_blend50      --rank-blend 0.5
run v2_ab_composite    --rank-blend 0.0 --composite-only 1
run v2_ab_model        --rank-blend 1.0
run v2_ab_rawmodel     --rank-blend 0.0
run v2_ab_ruleonly     --no-predictions 1
run v2_ab_b3515        --rank-blend 0.35
echo ALLDONE
