#!/usr/bin/env bash
# Stage-1 round 1d: is the consensus advantage robust across seeds?
#
# The corrected ablation shows the rank consensus at IR 1.295 against 0.941 for the
# composite alone - but that is one seed, and seed 42 is the lucky one for this
# pipeline.  If the composite-only arm also averages ~0.95 across seeds, the
# consensus is adding nothing on average and the single-seed gap is a draw.
set -u
cd /Users/shuzhenyi/code/python/qlib
R=sleeve/a/scripts/run_qrun_v2.py
C=/Volumes/lexar_4t/code/data/qlib/sleeve/a/cache
L=/Volumes/lexar_4t/code/data/qlib/sleeve/a/output/v2_research
BASE="--composite combo3 --max-new 1 --max-weight 0.08 --max-positions 12 --risk-per-trade 0.016 --extreme-exit 0 --extreme-liquidates 0 --drawdown-tiers 0.08:0.7,0.12:0.45 --use-market-timing 0 --vol-target 0.14 --profit-target 0.075 --hold 120"

run () {
  local name="$1"; local pred="$2"; shift 2
  scripts/run_python_3gb.sh $R --predictions $pred --start 2019-01-02 --end 2026-09-14 \
    --output sleeve/a/output/$name --recorder $name --experiment sleeve_a_v2 \
    $BASE "$@" > $L/run_$name.log 2>&1
  echo -n "$name "
  grep -E 'cagr_ge_25pct|annual_vol_le_10pct|max_drawdown_ge_minus10pct|round_trip_win_rate' $L/run_$name.log \
    | tr -s ' ' | cut -d' ' -f3 | tr '\n' ' '
  echo
}

  run v2_abl_comp_7 $C/v2m_tp5h40_predictions_s7.parquet --rank-blend 0.0 --composite-only 1
  run v2_abl_model_7 $C/v2m_tp5h40_predictions_s7.parquet --rank-blend 1.0
  run v2_abl_comp_101 $C/v2m_tp5h40_predictions_s101.parquet --rank-blend 0.0 --composite-only 1
  run v2_abl_model_101 $C/v2m_tp5h40_predictions_s101.parquet --rank-blend 1.0
  run v2_abl_comp_23 $C/v2m_tp5h40_predictions_s23.parquet --rank-blend 0.0 --composite-only 1
  run v2_abl_model_23 $C/v2m_tp5h40_predictions_s23.parquet --rank-blend 1.0
echo ALLDONE
