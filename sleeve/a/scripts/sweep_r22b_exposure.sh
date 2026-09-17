#!/usr/bin/env bash
# Round-22b: can we keep the ladder-off configuration's higher return and win
# rate while paying less volatility?  Three candidate exposure rules, each run
# on all nine model seeds so the comparison is paired.
set -u
cd /Users/shuzhenyi/code/python/qlib
P=sleeve/a/scripts/pyrun.py
R=sleeve/a/scripts/run_qrun_v2.py
C=/Volumes/lexar_4t/code/data/qlib/sleeve/a/cache
L=/Volumes/lexar_4t/code/data/qlib/sleeve/a/output/v2_research
BASE="--composite combo3 --profit-target 0.035 --hold 45 --max-new 1 --max-weight 0.08 --max-positions 12 --risk-per-trade 0.016 --extreme-exit 0 --extreme-liquidates 0 --drawdown-tiers 0.08:0.7,0.12:0.45 --rank-blend 0.5 --vol-target 0.14"

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
  # A: ladder off, gross scaled down to match the headline's volatility
  run v2_sb_a_$S $P9 "$BASE --use-market-timing 0 --gross 0.75"
  # B: mild ladder - keeps the extreme-state protection, removes most of the
  #    weak-market de-risking that costs the return
  run v2_sb_b_$S $P9 "$BASE --state-caps weak:0.70,neutral:0.85,strong:0.95"
  # C: between B and the design ladder
  run v2_sb_c_$S $P9 "$BASE --state-caps weak:0.50,neutral:0.75,strong:0.92"
done
echo ALLDONE
