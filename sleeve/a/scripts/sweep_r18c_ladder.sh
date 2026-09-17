#!/usr/bin/env bash
# Round-18c sweep: is the market ladder's value in its LEVELS or in its TIMING?
# The exposure diagnostic shows the account sits at its ladder cap almost every
# day (mean cap 0.587, mean gross 0.565), so the ladder - not the volatility
# target - is what sets exposure.  Each run below replaces the per-state caps,
# leaving the state machine untouched.
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

# constant exposure at exactly the ladder's mean cap (0.587): the ladder's
# timing contribution, isolated from its average level
run v2_lflat58 --state-caps weak:0.587,neutral:0.587,strong:0.587
# constant exposure at other levels, to trace the risk/return line
run v2_lflat50 --state-caps weak:0.50,neutral:0.50,strong:0.50
run v2_lflat68 --state-caps weak:0.68,neutral:0.68,strong:0.68
# a flatter ladder at the same average level: same mean exposure, less variation
run v2_lsoft   --state-caps weak:0.50,neutral:0.62,strong:0.66
# an amplified ladder at a similar average level: more variation
run v2_lamp    --state-caps weak:0.25,neutral:0.72,strong:0.90
run v2_lamp2   --state-caps weak:0.15,neutral:0.80,strong:0.95
# inverted ladder: is the state ordering informative at all?
run v2_linv    --state-caps weak:0.85,neutral:0.60,strong:0.35
echo ALLDONE
