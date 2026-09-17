#!/usr/bin/env bash
# Round-24: two-dimensional exit grid at the compliance exposure.
#
# Round 23 showed the hold cap is the strongest lever in the project (45 -> 120
# days is worth +8.3pp of win rate at no return cost) and that a 4.5% profit
# target was best at a 45-day cap.  The two interact, so this grid searches them
# jointly, with the risk budget held at the contract's ceiling.
set -u
cd /Users/shuzhenyi/code/python/qlib
P=sleeve/a/scripts/pyrun.py
R=sleeve/a/scripts/run_qrun_v2.py
C=/Volumes/lexar_4t/code/data/qlib/sleeve/a/cache
L=/Volumes/lexar_4t/code/data/qlib/sleeve/a/output/v2_research
PR=$C/v2m_tp5h40_predictions.parquet
BASE="--composite combo3 --max-new 1 --max-weight 0.08 --max-positions 12 --risk-per-trade 0.016 --extreme-exit 0 --extreme-liquidates 0 --drawdown-tiers 0.08:0.7,0.12:0.45 --rank-blend 0.5 --use-market-timing 0 --vol-target 0.055"

run () {
  local name="$1"; shift
  python $P $R --predictions $PR --start 2019-01-02 --end 2026-09-14 \
    --output sleeve/a/output/$name --recorder $name --experiment sleeve_a_v2 \
    $BASE "$@" > $L/run_$name.log 2>&1
  echo -n "$name "
  grep -E 'cagr_ge_25pct|annual_vol_le_10pct|max_drawdown_ge_minus10pct|round_trip_win_rate' $L/run_$name.log \
    | tr -s ' ' | cut -d' ' -f3 | tr '\n' ' '
  echo
}

run v2_gt_tp0030_h90 --profit-target 0.03 --hold 90
run v2_gt_tp0045_h90 --profit-target 0.045 --hold 90
run v2_gt_tp0060_h90 --profit-target 0.06 --hold 90
run v2_gt_tp0090_h90 --profit-target 0.09 --hold 90
run v2_gt_tp1000_h90 --profit-target 1 --hold 90
run v2_gt_tp0030_h120 --profit-target 0.03 --hold 120
run v2_gt_tp0045_h120 --profit-target 0.045 --hold 120
run v2_gt_tp0060_h120 --profit-target 0.06 --hold 120
run v2_gt_tp0090_h120 --profit-target 0.09 --hold 120
run v2_gt_tp1000_h120 --profit-target 1 --hold 120
run v2_gt_tp0030_h200 --profit-target 0.03 --hold 200
run v2_gt_tp0045_h200 --profit-target 0.045 --hold 200
run v2_gt_tp0060_h200 --profit-target 0.06 --hold 200
run v2_gt_tp0090_h200 --profit-target 0.09 --hold 200
run v2_gt_tp1000_h200 --profit-target 1 --hold 200
echo ALLDONE
