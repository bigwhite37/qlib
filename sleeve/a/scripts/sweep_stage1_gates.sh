#!/usr/bin/env bash
# Stage-1 round 3c: the entry quality gate under the new policy.
#
# The diagnostic says the book is slot-constrained: 11.73 of 12 slots are full on
# average while ~3000 candidates pass the gate every day.  So a new position is
# bought whenever a slot frees, regardless of how good that day's best candidate
# is.  The predicted-win and expected-return gates exist but have been pinned at
# non-binding values (p_min 0, mu_min -1) since the policy changed.
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

run v2_gate_p50 --p-min 0.50
run v2_gate_p60 --p-min 0.60
run v2_gate_p70 --p-min 0.70
run v2_gate_p80 --p-min 0.80
run v2_gate_mu0  --mu-min 0.0
run v2_gate_mu2  --mu-min 0.02
echo ALLDONE
