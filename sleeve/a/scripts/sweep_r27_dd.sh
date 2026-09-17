#!/usr/bin/env bash
# Round-27: can the drawdown be bought down by cutting exposure or going to cash?
#
# The question is whether a stronger de-risking rule improves the
# drawdown-to-volatility ratio, which is what caps the contract-feasible CAGR.
# Two new mechanisms are tested against the current overlay:
#   * a hard cut to cash on an account drawdown (needs --extreme-liquidates 1 to
#     actually sell), which is self-locking under an all-time peak NAV;
#   * the same rule measured against a ROLLING peak (--dd-peak-window), which
#     releases itself after the window and so can actually re-enter.
set -u
cd /Users/shuzhenyi/code/python/qlib
P=sleeve/a/scripts/pyrun.py
R=sleeve/a/scripts/run_qrun_v2.py
C=/Volumes/lexar_4t/code/data/qlib/sleeve/a/cache
L=/Volumes/lexar_4t/code/data/qlib/sleeve/a/output/v2_research
PR=$C/v2m_tp5h40_predictions.parquet
BASE="--composite combo3 --max-new 1 --max-weight 0.08 --max-positions 12 --risk-per-trade 0.016 --extreme-exit 0 --rank-blend 0.5 --use-market-timing 0 --vol-target 0.055 --profit-target 0.075 --hold 120"

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

# control: the c2 overlay (0.08:0.7, 0.12:0.45) with the all-time peak
run v2_dd_ctrl --drawdown-tiers 0.08:0.7,0.12:0.45
# tighter proportional trims (already tested in round 25, repeated for the table)
run v2_dd_tight --drawdown-tiers 0.06:0.6,0.10:0.35
# hard cut to cash on the account drawdown, all-time peak (expected to lock out)
run v2_dd_cash --drawdown-tiers 0.06:0.5,0.09:0.0 --extreme-liquidates 1
# same, but the drawdown is measured against a 60-day rolling NAV high
run v2_dd_cash60 --drawdown-tiers 0.06:0.5,0.09:0.0 --extreme-liquidates 1 --dd-peak-window 60
# a slower, deeper hard stop with a 120-day rolling peak
run v2_dd_cash120 --drawdown-tiers 0.08:0.4,0.12:0.0 --extreme-liquidates 1 --dd-peak-window 120
# the current tiers, but on a rolling peak
run v2_dd_roll60 --drawdown-tiers 0.08:0.7,0.12:0.45 --dd-peak-window 60
echo ALLDONE
