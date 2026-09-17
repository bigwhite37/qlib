#!/usr/bin/env bash
# Round-22d: the ladder-off configuration is starved by the market ladder and
# protected only by the account drawdown tiers.  These runs test whether a
# *stronger* drawdown overlay, keyed on the account's own NAV rather than on
# the market state, buys the drawdown back without paying the return.
set -u
cd /Users/shuzhenyi/code/python/qlib
P=sleeve/a/scripts/pyrun.py
R=sleeve/a/scripts/run_qrun_v2.py
C=/Volumes/lexar_4t/code/data/qlib/sleeve/a/cache
L=/Volumes/lexar_4t/code/data/qlib/sleeve/a/output/v2_research
BASE="--composite combo3 --profit-target 0.035 --hold 45 --max-new 1 --max-weight 0.08 --max-positions 12 --risk-per-trade 0.016 --extreme-exit 0 --extreme-liquidates 0 --rank-blend 0.5 --vol-target 0.14 --use-market-timing 0"

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
  # F: ladder off, wider drawdown tiers (cut later, cut harder)
  run v2_sd2_f_$S $P9 "$BASE --drawdown-tiers 0.10:0.7,0.16:0.45"
  # G: ladder off, tighter drawdown tiers (cut earlier)
  run v2_sd2_g_$S $P9 "$BASE --drawdown-tiers 0.06:0.65,0.10:0.40"
  # H: ladder off with a gross cap that actually binds, to trace the frontier
  run v2_sd2_h_$S $P9 "$BASE --drawdown-tiers 0.08:0.7,0.12:0.45 --gross 0.62"
done
echo ALLDONE
