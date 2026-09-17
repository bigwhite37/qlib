#!/usr/bin/env bash
# Round-1 (stage-1 objective): does a third opinion sharpen the entry consensus?
#
# The entry score is 0.5*rank(model) + 0.5*rank(combo3).  A rank blend does not
# combine two return streams - it selects names both rankers agree on, which is
# why it is so much stronger than either member (whose own daily returns
# correlate 0.96).  If agreement is the mechanism, a third, different opinion
# should sharpen it further.  Seven partner rankers are tested, each given 20% of
# the score with the model at 40% and combo3 at 40%.
set -u
cd /Users/shuzhenyi/code/python/qlib
P=sleeve/a/scripts/pyrun.py
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

# control: the two-way consensus (model 50 / composite 50)
run v2_cs_control --rank-blend 0.5
run v2_cs_ivol60 --rank-blend 0.4 --blend2 0.2 --predictions2 $C/partners/partner_ivol60.parquet
run v2_cs_maxret20 --rank-blend 0.4 --blend2 0.2 --predictions2 $C/partners/partner_maxret20.parquet
run v2_cs_dd252 --rank-blend 0.4 --blend2 0.2 --predictions2 $C/partners/partner_dd252.parquet
run v2_cs_semidev20 --rank-blend 0.4 --blend2 0.2 --predictions2 $C/partners/partner_semidev20.parquet
run v2_cs_skew60 --rank-blend 0.4 --blend2 0.2 --predictions2 $C/partners/partner_skew60.parquet
run v2_cs_posdays20 --rank-blend 0.4 --blend2 0.2 --predictions2 $C/partners/partner_pos_days20.parquet
run v2_cs_beta60 --rank-blend 0.4 --blend2 0.2 --predictions2 $C/partners/partner_beta60.parquet
echo ALLDONE
