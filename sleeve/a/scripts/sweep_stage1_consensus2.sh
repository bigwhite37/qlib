#!/usr/bin/env bash
# Stage-1 round 5: hard consensus and entry patience.
#
# Two mechanisms that follow directly from the round-1 and round-3 findings:
#   * the entry score is an AVERAGE of two rankers, which lets a name ranked first
#     by one and 500th by the other win the day - a hard consensus takes the
#     elementwise minimum instead;
#   * the book is slot-constrained, so a freed slot is filled with whatever that
#     day offers - patience skips the day when the best candidate is weak.
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
  echo
}

run v2_hc_off     --blend-mode mean --entry-min-score 0
run v2_hc_min     --blend-mode min
run v2_hc_min_50  --blend-mode min --entry-min-score 0.50
run v2_hc_min_60  --blend-mode min --entry-min-score 0.60
run v2_hc_mean_50 --blend-mode mean --entry-min-score 0.50
run v2_hc_mean_60 --blend-mode mean --entry-min-score 0.60
run v2_hc_mean_40 --blend-mode mean --entry-min-score 0.40
echo ALLDONE
