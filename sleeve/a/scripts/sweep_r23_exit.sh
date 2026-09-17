#!/usr/bin/env bash
# Round-23, step 2: re-tune the exit policy at the compliance exposure.
#
# The exposure calibration shows that at ~10% realised volatility the ladder-off
# configuration and the design ladder produce the same return (7.4%), so the
# ladder-off gain is leverage rather than alpha.  The two objectives the task
# names first - win rate and CAGR - therefore have to be improved at a FIXED
# risk budget.  This sweep holds the risk budget at the contract's ceiling
# (vol-target 0.055 -> ~10% realised volatility) and moves the exit policy.
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

run v2_e_tp20 --profit-target 0.02 --hold 45
run v2_e_tp25 --profit-target 0.025 --hold 45
run v2_e_tp30 --profit-target 0.03 --hold 45
run v2_e_tp45 --profit-target 0.045 --hold 45
run v2_e_tp60 --profit-target 0.06 --hold 45
run v2_e_h60  --profit-target 0.035 --hold 60
run v2_e_h90  --profit-target 0.035 --hold 90
run v2_e_tp60h90 --profit-target 0.06 --hold 90
echo ALLDONE
