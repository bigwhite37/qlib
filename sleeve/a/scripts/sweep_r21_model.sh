#!/usr/bin/env bash
# Round-21: does a properly trained rolling model improve the blend?
#
# The first release of build_v2_predictions.py early-stopped the return model on
# a 6-month validation rank IC, which selected iteration 1 in 9 of 31 quarters
# and a median of 7 - the "model" the account blends against was often a single
# split.  These runs swap in models trained for a fixed number of rounds, and a
# lambdarank model as a control.
set -u
cd /Users/shuzhenyi/code/python/qlib
P=sleeve/a/scripts/pyrun.py
R=sleeve/a/scripts/run_qrun_v2.py
C=/Volumes/lexar_4t/code/data/qlib/sleeve/a/cache
L=/Volumes/lexar_4t/code/data/qlib/sleeve/a/output/v2_research
BASE="--composite combo3 --vol-target 0.14 --profit-target 0.035 --hold 45 --max-new 1 --max-weight 0.08 --max-positions 12 --risk-per-trade 0.016 --extreme-exit 0 --extreme-liquidates 0 --drawdown-tiers 0.08:0.7,0.12:0.45 --rank-blend 0.5"

run () {
  local name="$1"; local pred="$2"; shift 2
  python $P $R --predictions $pred --start 2019-01-02 --end 2026-09-14 \
    --output sleeve/a/output/$name --recorder $name --experiment sleeve_a_v2 \
    $BASE "$@" > $L/run_$name.log 2>&1
  echo -n "$name "
  grep -E 'cagr_ge_25pct|annual_vol_le_10pct|max_drawdown_ge_minus10pct|round_trip_win_rate' $L/run_$name.log \
    | tr -s ' ' | cut -d' ' -f3 | tr '\n' ' '
  echo
}

run v2_m21_mse150 $C/v2m_tp5h40_predictions_mse150.parquet
run v2_m21_mse400 $C/v2m_tp5h40_predictions_mse400.parquet
run v2_m21_ll150  $C/v2m_tp5h40_predictions_ll150.parquet
run v2_m21_mse150b40 $C/v2m_tp5h40_predictions_mse150.parquet --rank-blend 0.4
run v2_m21_mse150b60 $C/v2m_tp5h40_predictions_mse150.parquet --rank-blend 0.6
echo ALLDONE
