#!/usr/bin/env bash
# Round-18 sweep: portfolio breadth, entry rate, and the sizing/timing layer.
# Every run is the v2_g4 headline configuration with exactly one axis changed,
# so any difference is attributable to that axis.  All runs share the same
# prebuilt rolling-model predictions, engine, fees and execution model.
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
  echo "== $name =="
  grep -A8 'acceptance passed' $L/run_$name.log | head -9
}

# control: the headline configuration, re-run to prove reproducibility
run v2_g4b
# breadth: same gross capacity, more names (max-weight ~ 1/N)
run v2_bp16  --max-positions 16 --max-weight 0.07
run v2_bp20  --max-positions 20 --max-weight 0.055
run v2_bp24  --max-positions 24 --max-weight 0.045
# breadth with a matching entry rate (N names, ~20d average hold needs N/20 entries/day)
run v2_bp16n2 --max-positions 16 --max-weight 0.07  --max-new 2
run v2_bp20n2 --max-positions 20 --max-weight 0.055 --max-new 2
run v2_bp20n3 --max-positions 20 --max-weight 0.055 --max-new 3
# the sizing / timing layer on its own
run v2_vt0    --use-vol-target 0
run v2_vt20   --vol-target 0.20
run v2_notm   --use-market-timing 0
echo ALLDONE
