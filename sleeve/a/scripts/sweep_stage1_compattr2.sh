#!/usr/bin/env bash
# Stage-1 round 2c: finish the replicated component attribution.
#
# The single-run attribution showed absurd non-additivity, and replication already
# overturned one of its claims (solo_amt looked 0.07 IR better than combo3 and is
# in fact a tie).  The remaining cells - the pairs containing dist_ma60 and the
# solo MA60 arm - are replicated here so the whole 7-cell table is measured the
# same way.
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
  run v2_ca_pairamtma_$S $P9 --composite pair_amt_ma --rank-blend 0.2
done
for S in 42 7 2024 11 23 101 777 31337 5; do
  P9=$C/v2m_tp5h40_predictions_s$S.parquet
  [ "$S" = "42" ] && P9=$C/v2m_tp5h40_predictions.parquet
  run v2_ca_pairvolma_$S $P9 --composite pair_vol_ma --rank-blend 0.2
done
for S in 42 7 2024 11 23 101 777 31337 5; do
  P9=$C/v2m_tp5h40_predictions_s$S.parquet
  [ "$S" = "42" ] && P9=$C/v2m_tp5h40_predictions.parquet
  run v2_ca_soloma_$S $P9 --composite solo_ma --rank-blend 0.2
done
echo ALLDONE
