#!/usr/bin/env bash
# Round-25b: re-tune the model/rule blend under the new policy.
#
# The 0.5 blend weight was chosen a priori in round 8 and has never been
# re-validated - and every other entry-side parameter was tuned under a 3.5%
# profit target and a 45-day hold, both of which rounds 23-24 replaced with 7.5%
# and 120 days.  The blend is the single largest effect in the project, so it is
# tested here paired on all nine models against the c2 setting of 0.5.
set -u
cd /Users/shuzhenyi/code/python/qlib
P=sleeve/a/scripts/pyrun.py
R=sleeve/a/scripts/run_qrun_v2.py
C=/Volumes/lexar_4t/code/data/qlib/sleeve/a/cache
L=/Volumes/lexar_4t/code/data/qlib/sleeve/a/output/v2_research
BASE="--composite combo3 --max-new 1 --max-weight 0.08 --max-positions 12 --risk-per-trade 0.016 --extreme-exit 0 --extreme-liquidates 0 --drawdown-tiers 0.08:0.7,0.12:0.45 --use-market-timing 0 --vol-target 0.055 --profit-target 0.075 --hold 120"

run () {
  local name="$1"; local pred="$2"; local cfg="$3"
  python $P $R --predictions $pred --start 2019-01-02 --end 2026-09-14 \
    --output sleeve/a/output/$name --recorder $name --experiment sleeve_a_v2 \
    $BASE $cfg > $L/run_$name.log 2>&1
  echo -n "$name "
  grep -E 'cagr_ge_25pct|annual_vol_le_10pct|max_drawdown_ge_minus10pct|round_trip_win_rate' $L/run_$name.log \
    | tr -s ' ' | cut -d' ' -f3 | tr '\n' ' '
  echo
}

for S in 42 7 2024 11 23 101 777 31337 5; do
  P9=$C/v2m_tp5h40_predictions_s$S.parquet
  [ "$S" = "42" ] && P9=$C/v2m_tp5h40_predictions.parquet
  run v2_b35_$S $P9 "--rank-blend 0.35"
  run v2_b65_$S $P9 "--rank-blend 0.65"
done
echo ALLDONE
