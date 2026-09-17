#!/usr/bin/env bash
# Round-21d: the headline configuration and the compliance configuration, each
# run on every model seed, so both can be reported as a distribution rather than
# as a single lucky path.
set -u
cd /Users/shuzhenyi/code/python/qlib
P=sleeve/a/scripts/pyrun.py
R=sleeve/a/scripts/run_qrun_v2.py
C=/Volumes/lexar_4t/code/data/qlib/sleeve/a/cache
L=/Volumes/lexar_4t/code/data/qlib/sleeve/a/output/v2_research
G4="--composite combo3 --vol-target 0.14 --profit-target 0.035 --hold 45 --max-new 1 --max-weight 0.08 --max-positions 12 --risk-per-trade 0.016 --extreme-exit 0 --extreme-liquidates 0 --drawdown-tiers 0.08:0.7,0.12:0.45 --rank-blend 0.5"
Y4="--composite combo3 --vol-target 0.14 --profit-target 0.03 --hold 120 --max-new 1 --max-weight 0.08 --max-positions 12 --risk-per-trade 0.016 --extreme-exit 0 --extreme-liquidates 0 --drawdown-tiers 0.08:0.7,0.12:0.45 --rank-blend 0.5 --cap-scale 0.8"

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

for S in 11 23 101 777 31337 5; do
  run v2_sd_g4_$S  $C/v2m_tp5h40_predictions_s$S.parquet "$G4"
done
for S in 11 23 101 777 31337 5; do
  run v2_sd_y4_$S  $C/v2m_tp5h40_predictions_s$S.parquet "$Y4"
done
echo ALLDONE
