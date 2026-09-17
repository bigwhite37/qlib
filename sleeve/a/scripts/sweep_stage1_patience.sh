#!/usr/bin/env bash
# Stage-1 round 5b: time-series patience on the entry side.
#
# The absolute threshold (--entry-min-score) was a silent no-op: the maximum of a
# percentile-rank average over ~3000 names is always high.  The meaningful version
# is relative - only open a slot on days whose BEST candidate is above a given
# percentile of the trailing 252-day distribution of daily best candidates.
set -u
cd /Users/shuzhenyi/code/python/qlib
R=sleeve/a/scripts/run_qrun_v2.py
C=/Volumes/lexar_4t/code/data/qlib/sleeve/a/cache
L=/Volumes/lexar_4t/code/data/qlib/sleeve/a/output/v2_research
PR=$C/v2m_tp5h40_predictions.parquet
BASE="--composite combo3 --max-new 1 --max-weight 0.08 --max-positions 12 --risk-per-trade 0.016 --extreme-exit 0 --extreme-liquidates 0 --drawdown-tiers 0.08:0.7,0.12:0.45 --use-market-timing 0 --vol-target 0.14 --profit-target 0.075 --hold 120 --rank-blend 0.5"

run () {
  local name="$1"; shift
  scripts/run_python_3gb.sh $R --predictions $PR --start 2019-01-02 --end 2026-09-14 \
    --output sleeve/a/output/$name --recorder $name --experiment sleeve_a_v2 \
    $BASE "$@" > $L/run_$name.log 2>&1
  echo -n "$name "
  grep -E 'cagr_ge_25pct|annual_vol_le_10pct|max_drawdown_ge_minus10pct|round_trip_win_rate' $L/run_$name.log \
    | tr -s ' ' | cut -d' ' -f3 | tr '\n' ' '
  grep -o '\[patience\][^(]*' $L/run_$name.log | head -1
  echo
}

run v2_pa_off  --entry-quality-pct 0.0
run v2_pa_50   --entry-quality-pct 0.50
run v2_pa_70   --entry-quality-pct 0.70
run v2_pa_90   --entry-quality-pct 0.90
run v2_pa_70w120 --entry-quality-pct 0.70 --entry-quality-window 120
echo ALLDONE
