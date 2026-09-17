#!/usr/bin/env bash
# Round-21c: seed noise, seed averaging, and the validation window.
#
# The random seed is an irrelevant knob - it cannot change the strategy's
# economics - so the spread of account results across seeds is a second, fully
# independent measurement of the local noise floor.  The three-seed rank
# ensemble then tests whether averaging that noise away helps.
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

run v2_m21_seed7    $C/v2m_tp5h40_predictions_s7.parquet
run v2_m21_seed2024 $C/v2m_tp5h40_predictions_s2024.parquet
run v2_m21_seed3   $C/v2m_tp5h40_predictions_seed3.parquet
run v2_m21_valid3  $C/v2m_tp5h40_predictions_v3.parquet
run v2_m21_valid12 $C/v2m_tp5h40_predictions_v12.parquet
echo ALLDONE
