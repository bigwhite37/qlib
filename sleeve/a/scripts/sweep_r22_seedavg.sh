#!/usr/bin/env bash
# Round-22: seed-averaged evaluation of the headline alternatives.
#
# The seed distribution measured in round 21 has a standard deviation of 2.2pp
# of CAGR, which is larger than most of the differences the project has been
# selecting on.  Every configuration below is therefore run on all nine model
# seeds and compared on its MEAN, not on its best draw.
set -u
cd /Users/shuzhenyi/code/python/qlib
P=sleeve/a/scripts/pyrun.py
R=sleeve/a/scripts/run_qrun_v2.py
C=/Volumes/lexar_4t/code/data/qlib/sleeve/a/cache
L=/Volumes/lexar_4t/code/data/qlib/sleeve/a/output/v2_research
G4="--composite combo3 --vol-target 0.14 --profit-target 0.035 --hold 45 --max-new 1 --max-weight 0.08 --max-positions 12 --risk-per-trade 0.016 --extreme-exit 0 --extreme-liquidates 0 --drawdown-tiers 0.08:0.7,0.12:0.45 --rank-blend 0.5"

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
  # market ladder off: the best single-run candidate (14.60%)
  run v2_sa_notm_$S $P9 "$G4 --use-market-timing 0"
  # headline without the drawdown tiers: the second best single run (14.53%)
  run v2_sa_m1b_$S  $P9 "--composite combo3 --vol-target 0.14 --profit-target 0.035 --hold 45 --max-new 1 --max-weight 0.08 --max-positions 12 --risk-per-trade 0.016 --extreme-exit 0 --extreme-liquidates 0 --rank-blend 0.5"
  # correlation-aware sizing: the best single-run information ratio (1.141)
  run v2_sa_cs1_$S  $P9 "$G4 --corr-sizing 1 --corr-sizing-strength 0.5 --corr-sizing-base 0.3 --corr-sizing-floor 0.5"
  # the compliance configuration, completing the nine-seed panel
  run v2_sa_y4_$S   $P9 "--composite combo3 --vol-target 0.14 --profit-target 0.03 --hold 120 --max-new 1 --max-weight 0.08 --max-positions 12 --risk-per-trade 0.016 --extreme-exit 0 --extreme-liquidates 0 --drawdown-tiers 0.08:0.7,0.12:0.45 --rank-blend 0.5 --cap-scale 0.8"
done
echo ALLDONE
