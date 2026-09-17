#!/usr/bin/env bash
# Round-18b sweep: fine resolution on the three axes that were only ever
# sampled coarsely - the model/rule blend weight, the portfolio volatility
# estimation window, and the liquidity floor of the base universe.
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

# blend weight, resolved at 0.05 instead of 0.5
run v2_rb30 --rank-blend 0.30
run v2_rb40 --rank-blend 0.40
run v2_rb45 --rank-blend 0.45
run v2_rb55 --rank-blend 0.55
run v2_rb60 --rank-blend 0.60
run v2_rb65 --rank-blend 0.65
run v2_rb70 --rank-blend 0.70
run v2_rb80 --rank-blend 0.80
# portfolio volatility estimation window (default 60)
run v2_vw40  --vol-window 40
run v2_vw90  --vol-window 90
run v2_vw120 --vol-window 120
# liquidity floor of the V2 base universe (default 0.20 amount rank)
run v2_ma10 --min-amount-rank 0.10
run v2_ma35 --min-amount-rank 0.35
run v2_ma50 --min-amount-rank 0.50
echo ALLDONE
