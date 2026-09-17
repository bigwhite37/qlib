#!/usr/bin/env bash
# Stage-1 round 2b: the composite attribution needs replications.
#
# With --composite-only every configuration produces exactly ONE path, and the
# component attribution showed absurd non-additivity (pairs far worse than either
# single, IR from 0.087 to 0.969), which is the signature of path noise rather
# than of signal.  The fix is to give each composite NINE replications: keep the
# composite in charge (weight 0.8) and let the model provide a small per-seed
# perturbation (weight 0.2), then compare the four candidates paired by seed.
set -u
cd /Users/shuzhenyi/code/python/qlib
R=sleeve/a/scripts/run_qrun_v2.py
C=/Volumes/lexar_4t/code/data/qlib/sleeve/a/cache
L=/Volumes/lexar_4t/code/data/qlib/sleeve/a/output/v2_research
BASE="--max-new 1 --max-weight 0.08 --max-positions 12 --risk-per-trade 0.016 --extreme-exit 0 --extreme-liquidates 0 --drawdown-tiers 0.08:0.7,0.12:0.45 --use-market-timing 0 --vol-target 0.14 --profit-target 0.075 --hold 120"

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

for S in 42 7 2024 11 23 101 777 31337 5; do
  P9=$C/v2m_tp5h40_predictions_s$S.parquet
  [ "$S" = "42" ] && P9=$C/v2m_tp5h40_predictions.parquet
  run v2_ca_combo3_$S $P9 --composite combo3 --rank-blend 0.2
done
for S in 42 7 2024 11 23 101 777 31337 5; do
  P9=$C/v2m_tp5h40_predictions_s$S.parquet
  [ "$S" = "42" ] && P9=$C/v2m_tp5h40_predictions.parquet
  run v2_ca_soloamt_$S $P9 --composite solo_amt --rank-blend 0.2
done
for S in 42 7 2024 11 23 101 777 31337 5; do
  P9=$C/v2m_tp5h40_predictions_s$S.parquet
  [ "$S" = "42" ] && P9=$C/v2m_tp5h40_predictions.parquet
  run v2_ca_solovol_$S $P9 --composite solo_vol --rank-blend 0.2
done
for S in 42 7 2024 11 23 101 777 31337 5; do
  P9=$C/v2m_tp5h40_predictions_s$S.parquet
  [ "$S" = "42" ] && P9=$C/v2m_tp5h40_predictions.parquet
  run v2_ca_pairvolamt_$S $P9 --composite pair_vol_amt --rank-blend 0.2
done
echo ALLDONE
