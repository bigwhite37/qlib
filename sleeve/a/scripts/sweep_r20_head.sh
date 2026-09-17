#!/usr/bin/env bash
# Round-20: head re-ranking.
#
# Round 18 established that the account only ever buys from the very top of the
# blended list (broadening the book destroys IR), and round 19 that the head,
# not the average of the ranking, is what the composite has to get right.  This
# sweep applies a second ranking stage to the top K candidates only, choosing
# the daily buy on a single well-understood dimension.
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

run v2_hr3_vol   --head-rerank 3  --head-key vol60
run v2_hr5_vol   --head-rerank 5  --head-key vol60
run v2_hr3_amt   --head-rerank 3  --head-key amount20_yuan
run v2_hr5_amt   --head-rerank 5  --head-key amount20_yuan
run v2_hr3_ma60  --head-rerank 3  --head-key ma60
run v2_hr3_atr   --head-rerank 3  --head-key atr20
run v2_hr3_ret   --head-rerank 3  --head-key ret1 --head-sign -1
run v2_hr5_ma60  --head-rerank 5  --head-key ma60
echo ALLDONE
