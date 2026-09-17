#!/usr/bin/env bash
# Round-23, step 1: calibrate the exposure of the ladder-off configuration.
#
# Round 22 found that with the market ladder off the book is entry limited
# (mean gross 0.703 against a 0.95 cap), so neither --gross 0.75 nor
# --vol-target 0.11 moved realised volatility.  This sweep traces the response
# of realised volatility to the two remaining knobs - the portfolio volatility
# target (which only rescales weights and does not block entries) and the
# single-name weight cap - so the ladder-off strategy can be compared with the
# compliance configuration at the same realised risk.
set -u
cd /Users/shuzhenyi/code/python/qlib
P=sleeve/a/scripts/pyrun.py
R=sleeve/a/scripts/run_qrun_v2.py
C=/Volumes/lexar_4t/code/data/qlib/sleeve/a/cache
L=/Volumes/lexar_4t/code/data/qlib/sleeve/a/output/v2_research
PR=$C/v2m_tp5h40_predictions.parquet
BASE="--composite combo3 --profit-target 0.035 --hold 45 --max-new 1 --max-weight 0.08 --max-positions 12 --risk-per-trade 0.016 --extreme-exit 0 --extreme-liquidates 0 --drawdown-tiers 0.08:0.7,0.12:0.45 --rank-blend 0.5 --use-market-timing 0"

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

run v2_cal_vt12 --vol-target 0.12
run v2_cal_vt10 --vol-target 0.10
run v2_cal_vt085 --vol-target 0.085
run v2_cal_vt07 --vol-target 0.07
run v2_cal_vt055 --vol-target 0.055
run v2_cal_mw05 --vol-target 0.14 --max-weight 0.05
echo ALLDONE
