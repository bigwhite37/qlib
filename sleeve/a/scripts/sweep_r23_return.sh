#!/usr/bin/env bash
# Round-23, step 4: does the long hold also help at the return end?
#
# At the compliance exposure, moving the hold cap from 45 to 90/120 days raised
# the seed-mean win rate from ~70% to 78% and left the CAGR unchanged.  If the
# same holds at the return end of the frontier, the return headline improves
# too.  Two exposure points, both with a 120-day hold, on all nine seeds.
set -u
cd /Users/shuzhenyi/code/python/qlib
P=sleeve/a/scripts/pyrun.py
R=sleeve/a/scripts/run_qrun_v2.py
C=/Volumes/lexar_4t/code/data/qlib/sleeve/a/cache
L=/Volumes/lexar_4t/code/data/qlib/sleeve/a/output/v2_research
BASE="--composite combo3 --profit-target 0.035 --max-new 1 --max-weight 0.08 --max-positions 12 --risk-per-trade 0.016 --extreme-exit 0 --extreme-liquidates 0 --drawdown-tiers 0.08:0.7,0.12:0.45 --rank-blend 0.5 --use-market-timing 0 --hold 120"

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
  # return end of the frontier, 120-day hold
  run v2_g_r120_$S $P9 "$BASE --vol-target 0.14"
  # middle of the frontier, 120-day hold
  run v2_g_m120_$S $P9 "$BASE --vol-target 0.09"
done
echo ALLDONE
