#!/usr/bin/env bash
# Round-24c: validate the exit-grid optima on all nine model seeds.
#
# The grid (one model) says the frozen 3.5% profit target was far too small: at
# the contract's risk budget a 9% target with a 90-day cap earns 11.85% CAGR at
# 9.82% volatility (IR 1.21), against 7.36% for the old 3.5% target.  Three
# settings are validated here paired on the same nine models.
set -u
cd /Users/shuzhenyi/code/python/qlib
P=sleeve/a/scripts/pyrun.py
R=sleeve/a/scripts/run_qrun_v2.py
C=/Volumes/lexar_4t/code/data/qlib/sleeve/a/cache
L=/Volumes/lexar_4t/code/data/qlib/sleeve/a/output/v2_research
BASE="--composite combo3 --max-new 1 --max-weight 0.08 --max-positions 12 --risk-per-trade 0.016 --extreme-exit 0 --extreme-liquidates 0 --drawdown-tiers 0.08:0.7,0.12:0.45 --rank-blend 0.5 --use-market-timing 0 --vol-target 0.055"

run () {
  local name="$1"; local pred="$2"; local cfg="$3"
  python $P $R --predictions $pred --start 2019-01-02 --end 2026-09-14 \
    --output sleeve/a/output/$name --recorder $name --experiment sleeve_a_v2 \
    $cfg > $L/run_$name.log 2>&1
  echo -n "$name "
  grep -E 'cagr_ge_25pct|annual_vol_le_10pct|max_drawdown_ge_minus10pct|round_trip_win_rate' $L/run_$name.log \
    | tr -s ' ' | cut -d' ' -f3 | tr '\n' ' '
  echo
}

for S in 42 7 2024 11 23 101 777 31337 5; do
  P9=$C/v2m_tp5h40_predictions_s$S.parquet
  [ "$S" = "42" ] && P9=$C/v2m_tp5h40_predictions.parquet
  run v2_v_c1_$S $P9 "${BASE} --profit-target 0.09 --hold 90"
  run v2_v_c2_$S $P9 "${BASE} --profit-target 0.075 --hold 120"
  run v2_v_c3_$S $P9 "${BASE} --profit-target 0.06 --hold 120"
done
echo ALLDONE
