#!/usr/bin/env bash
# Round-18d: the local noise floor of the headline.
#
# Every run here is the v2_g4 configuration with one parameter moved by about
# 1-2 per cent of its own value - far too little to change the strategy's
# economics, but enough to occasionally reorder two candidates or move a fill.
# The spread of the resulting CAGRs is therefore a measurement of how much of
# the headline is path, not edge.
set -u
cd /Users/shuzhenyi/code/python/qlib
P=sleeve/a/scripts/pyrun.py
R=sleeve/a/scripts/run_qrun_v2.py
PR=/Volumes/lexar_4t/code/data/qlib/sleeve/a/cache/v2m_tp5h40_predictions.parquet
L=/Volumes/lexar_4t/code/data/qlib/sleeve/a/output/v2_research
C="--composite combo3 --vol-target 0.14 --profit-target 0.035 --hold 45 --max-new 1 --max-weight 0.08 --max-positions 12 --risk-per-trade 0.016 --extreme-exit 0 --extreme-liquidates 0 --drawdown-tiers 0.08:0.7,0.12:0.45 --rank-blend 0.5"

run () {
  local name="$1"; shift
  python $P $R --predictions $PR --start 2019-01-02 --end 2026-09-14 \
    --output sleeve/a/output/$name --recorder $name --experiment sleeve_a_v2 \
    $C "$@" > $L/run_$name.log 2>&1
  echo -n "$name "
  grep -E 'cagr_ge_25pct|annual_vol_le_10pct|max_drawdown_ge_minus10pct|round_trip_win_rate' $L/run_$name.log \
    | tr -s ' ' | cut -d' ' -f3 | tr '\n' ' '
  echo
}

run v2_j_blend49 --rank-blend 0.49
run v2_j_blend51 --rank-blend 0.51
run v2_j_ma199   --min-amount-rank 0.199
run v2_j_ma201   --min-amount-rank 0.201
run v2_j_prem29  --buy-premium 0.029
run v2_j_prem31  --buy-premium 0.031
run v2_j_vw59    --vol-window 59
run v2_j_vw61    --vol-window 61
run v2_j_risk159 --risk-per-trade 0.0159
run v2_j_risk161 --risk-per-trade 0.0161
echo ALLDONE
