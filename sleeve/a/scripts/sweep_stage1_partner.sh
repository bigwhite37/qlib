#!/usr/bin/env bash
# Round-1c: a third opinion only helps if it is structurally different.
#
# Round 1 found that the rank consensus (IR 1.295) beats both the best single arm
# (0.941) and the 50/50 RETURN blend of the same two arms (0.935) - so the blend is
# a selection effect, not portfolio diversification.  The two arms correlate only
# 0.67.  Adding rule-factor partners (ivol, maxret, dd252, ...) all hurt because
# they are the same *kind* of opinion as combo3.  The next candidates are models
# trained on different labels, which are the same kind as the model arm but on a
# different target.
set -u
cd /Users/shuzhenyi/code/python/qlib
R=sleeve/a/scripts/run_qrun_v2.py
C=/Volumes/lexar_4t/code/data/qlib/sleeve/a/cache
L=/Volumes/lexar_4t/code/data/qlib/sleeve/a/output/v2_research
PR=$C/v2m_tp5h40_predictions.parquet
BASE="--composite combo3 --max-new 1 --max-weight 0.08 --max-positions 12 --risk-per-trade 0.016 --extreme-exit 0 --extreme-liquidates 0 --drawdown-tiers 0.08:0.7,0.12:0.45 --use-market-timing 0 --vol-target 0.14 --profit-target 0.075 --hold 120"

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

# partner = a model trained on a different label
run v2_p_fwd20_333 $C/v2m_tp5h40_predictions_fwd20.parquet
run v2_p_fwd20_204 --predictions2 $C/v2m_tp5h40_predictions_fwd20.parquet --rank-blend 0.4 --blend2 0.2
run v2_p_t60_333   --predictions2 $C/v2m_tp5h40_predictions_t60.parquet --rank-blend 0.33 --blend2 0.33
run v2_p_t60_204   --predictions2 $C/v2m_tp5h40_predictions_t60.parquet --rank-blend 0.4 --blend2 0.2
# partner = the same model on a different seed family (weakly different opinion)
run v2_p_seed_336  --predictions2 $C/v2m_tp5h40_predictions_s101.parquet --rank-blend 0.33 --blend2 0.33
# keep the composite at half the weight and split the rest between two model views
run v2_p_t60_2515  --predictions2 $C/v2m_tp5h40_predictions_t60.parquet --rank-blend 0.25 --blend2 0.15
echo ALLDONE
