#!/usr/bin/env bash
# Round-25: reopen the loss-cutting dimension under the new policy regime.
#
# Stops, trailing stops, stale-loser exits and profit locks were all rejected in
# rounds 2-4 - but that was under a 3.5% profit target and a 45-day hold cap,
# which round 23/24 have since replaced with 7.5% and 120 days.  A stop and a
# profit target are the same kind of object, so the old verdict does not carry
# over.  One model first, then the survivors on all nine.
set -u
cd /Users/shuzhenyi/code/python/qlib
P=sleeve/a/scripts/pyrun.py
R=sleeve/a/scripts/run_qrun_v2.py
C=/Volumes/lexar_4t/code/data/qlib/sleeve/a/cache
L=/Volumes/lexar_4t/code/data/qlib/sleeve/a/output/v2_research
PR=$C/v2m_tp5h40_predictions.parquet
BASE="--composite combo3 --max-new 1 --max-weight 0.08 --max-positions 12 --risk-per-trade 0.016 --extreme-exit 0 --extreme-liquidates 0 --drawdown-tiers 0.08:0.7,0.12:0.45 --rank-blend 0.5 --use-market-timing 0 --vol-target 0.055 --profit-target 0.075 --hold 120"

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

run v2_x_s1 --use-stop 1 --stop-atr-mult 3 --stop-min 0.08 --stop-max 0.15
run v2_x_s2 --use-stop 1 --stop-atr-mult 4 --stop-min 0.12 --stop-max 0.20
run v2_x_s3 --use-stop 1 --stop-atr-mult 3 --stop-min 0.08 --stop-max 0.15 --trailing 1
run v2_x_s4 --stale-days 30
run v2_x_s5 --stale-days 45
run v2_x_s6 --profit-lock 0.5
run v2_x_s7 --use-stop 1 --stop-atr-mult 2 --stop-min 0.05 --stop-max 0.10
run v2_x_s8 --use-stop 1 --stop-atr-mult 3 --stop-min 0.08 --stop-max 0.15 --trailing 1 --profit-lock 0.5
echo ALLDONE
