#!/usr/bin/env bash
# Stage-1 round 7: resolve the board-composition lead with a bigger sample.
#
# The STAR/Beijing exclusion showed +0.96pp of CAGR (paired t = 1.19 over nine seeds)
# - the only positive sign in six rounds, but short of the protocol's 8/9 bar.  Six
# further model seeds are added here, taking the paired sample to fifteen, so the
# effect can be either confirmed or killed.
set -u
cd /Users/shuzhenyi/code/python/qlib
R=sleeve/a/scripts/run_qrun_v2.py
C=/Volumes/lexar_4t/code/data/qlib/sleeve/a/cache
L=/Volumes/lexar_4t/code/data/qlib/sleeve/a/output/v2_research
BASE="--composite combo3 --max-new 1 --max-weight 0.08 --max-positions 12 --risk-per-trade 0.016 --extreme-exit 0 --extreme-liquidates 0 --drawdown-tiers 0.08:0.7,0.12:0.45 --use-market-timing 0 --vol-target 0.14 --profit-target 0.075 --hold 120 --rank-blend 0.5"

run () {
  local name="$1"; local pred="$2"; shift 2
  scripts/run_python_3gb.sh $R --predictions $pred --start 2019-01-02 --end 2026-09-14 \
    --output sleeve/a/output/$name --recorder $name --experiment sleeve_a_v2 \
    $BASE "$@" > $L/run_$name.log 2>&1
  echo -n "$name "
  grep -E 'cagr_ge_25pct|annual_vol_le_10pct|max_drawdown_ge_minus10pct|round_trip_win_rate' $L/run_$name.log \
    | tr -s ' ' | cut -d' ' -f3 | tr '\n' ' '
  echo
}

run v2_x9_z1_2025     $C/v2m_tp5h40_predictions_s2025.parquet
run v2_x9_nostar_2025 $C/v2m_tp5h40_predictions_s2025.parquet --exclude-boards star,bse
run v2_x9_z1_31     $C/v2m_tp5h40_predictions_s31.parquet
run v2_x9_nostar_31 $C/v2m_tp5h40_predictions_s31.parquet --exclude-boards star,bse
run v2_x9_z1_137     $C/v2m_tp5h40_predictions_s137.parquet
run v2_x9_nostar_137 $C/v2m_tp5h40_predictions_s137.parquet --exclude-boards star,bse
run v2_x9_z1_404     $C/v2m_tp5h40_predictions_s404.parquet
run v2_x9_nostar_404 $C/v2m_tp5h40_predictions_s404.parquet --exclude-boards star,bse
run v2_x9_z1_909     $C/v2m_tp5h40_predictions_s909.parquet
run v2_x9_nostar_909 $C/v2m_tp5h40_predictions_s909.parquet --exclude-boards star,bse
run v2_x9_z1_1234     $C/v2m_tp5h40_predictions_s1234.parquet
run v2_x9_nostar_1234 $C/v2m_tp5h40_predictions_s1234.parquet --exclude-boards star,bse
echo ALLDONE
