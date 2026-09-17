#!/usr/bin/env bash
# Round-19 sweep: the classic low-risk / lottery anomalies, tested at the
# account level (which is the level that has four times contradicted the IC
# scans).  Each run is v2_g4 with the composite's vol20 component replaced.
set -u
cd /Users/shuzhenyi/code/python/qlib
P=sleeve/a/scripts/pyrun.py
R=sleeve/a/scripts/run_qrun_v2.py
PR=/Volumes/lexar_4t/code/data/qlib/sleeve/a/cache/v2m_tp5h40_predictions.parquet
L=/Volumes/lexar_4t/code/data/qlib/sleeve/a/output/v2_research
BASE="--vol-target 0.14 --profit-target 0.035 --hold 45 --max-new 1 --max-weight 0.08 --max-positions 12 --risk-per-trade 0.016 --extreme-exit 0 --extreme-liquidates 0 --drawdown-tiers 0.08:0.7,0.12:0.45 --rank-blend 0.5"

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

run v2_r19beta --composite r19beta
run v2_r19ivol --composite r19ivol
run v2_r19semi --composite r19semi
run v2_r19skew --composite r19skew
run v2_r19dd   --composite r19dd
run v2_r19max  --composite r19max
run v2_r19pos  --composite r19pos
run v2_r19ac1  --composite r19ac1
run v2_r19mix  --composite r19mix
echo ALLDONE
