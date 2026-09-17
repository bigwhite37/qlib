# sleeve/a: policy-consistent V2 strategy (market-state constrained, two models)

This directory implements the designs in `docs/chats_001.md` (V0/V1) and
`docs/chats_002.md` (V2). All commands run from the repository root
`/Users/shuzhenyi/code/python/qlib` and force the local Qlib checkout onto
`sys.path`; the pip-installed Qlib is never used.

## Memory policy (mandatory)

Every Python run goes through `scripts/run_python_3gb.sh`, which enforces a hard
3 GiB resident-memory budget. macOS cannot apply `ulimit -v`, so the wrapper sets
RLIMIT_AS where the platform allows it and otherwise kills the interpreter from an
RSS watchdog. DuckDB connections carry `memory_limit = '2GB'` (from
`SLEEVE_DUCKDB_MEMORY_LIMIT`, which the wrapper exports) and the adapter verifies
the setting after opening.

```bash
scripts/run_python_3gb.sh sleeve/a/scripts/<script>.py [args]
scripts/run_python_3gb.sh -c "import qlib; print(qlib.__version__)"
```

(The older `sleeve/a/scripts/pyrun.py` launcher still exists and is used by the
sweep shell scripts; both enforce the same ceiling.)

## Final result after eight stage-1 rounds

**The contract's CAGR requirement is not reachable within the stated constraints.**
Fifteen design mechanisms were implemented and tested, each with a control that
reproduced the incumbent bit for bit, and the seed-averaged information ratio stayed
in the band **0.94-1.00** throughout, against stage 1's 1.30 and the contract's 2.5.
What is delivered, on all nine model seeds: win rate **74.8-80.5%** (>= 70% met),
volatility **9.67%** (<= 10% met), both frequency floors met, and **4 of 7 contract
checks on every seed**; CAGR **9.16%** at the compliant exposure and **13.76%** at
the return end, positive in every calendar year. See
`reports/v2_acceptance_report.md` section 11 for the per-seed matrix and section 12
for the stage statement.

## V2 pipeline (2019-01-02 .. 2026-09-14)

```bash
# 1) policy labels + compact feature matrix for every base-universe row
python sleeve/a/scripts/pyrun.py sleeve/a/scripts/build_v2_matrix.py \
  --profit-target 0.05 --stop none --hold 20 \
  --output-prefix /Volumes/lexar_4t/code/data/qlib/sleeve/a/cache/v2m_tp5

# 2) rolling quarterly models (36m train / 6m early stop / 6m calibration)
python sleeve/a/scripts/pyrun.py sleeve/a/scripts/build_v2_predictions.py

# 3) account backtest through Qlib's engine with the frozen V2 exit policy
python sleeve/a/scripts/pyrun.py sleeve/a/scripts/run_qrun_v2.py \
  --start 2019-01-02 --end 2026-09-14 --output sleeve/a/output/v2_final \
  --p-min 0.55 --profit-target 0.05 --use-stop 1 \
  --stop-atr-mult 0.0 --stop-min 0.08 --stop-max 0.08 --hold 20

# 4) research harnesses (entry x exit grids, IC scans, policy frontier, sweeps)
python sleeve/a/scripts/pyrun.py sleeve/a/scripts/research_v2_sweep.py --stage gates

# 5) acceptance report + tests
python sleeve/a/scripts/pyrun.py sleeve/a/scripts/build_report_v2.py
python -m pytest sleeve/a/tests -q
```

## What V2 changed relative to V0/V1

| Area | V0/V1 | V2 |
|---|---|---|
| Entry | hard rule gates (trend + RS + 1-3 ATR pullback + MA5 reclaim) | two LightGBM models: expected net policy return (ranking) and calibrated win probability (gate) |
| Labels | fixed 10-day direction | the realised outcome of the same exit policy the account runs |
| Exit policy | 2.5 ATR stop, MA/rank exits, 20-day cap | one shared ExitPolicy used by labels, simulator and live strategy |
| Market | four-state ladder with a weak-market repair gate | V2 ladder with an explicit extreme-risk recovery protocol (no per-name repair gate) |
| Risk | market x vol x drawdown multipliers | market cap x portfolio vol target only |
| Evaluation | final NAV curve | A/B/C/D attribution, opportunity-coverage table, exit-reason decomposition |

## Implemented with Qlib (not re-implemented)

| Need | Qlib component |
|---|---|
| DuckDB calendar / instruments / features | `qlib/data/storage/duckdb_storage.py` (3 GB limit verified at open) |
| Account, matching, fees | `SimulatorExecutor` (parallel buys/sells) and an `Exchange` subclass with A-share limit rules plus frozen max-buy prices |
| Strategy skeleton | `qlib.contrib.strategy.signal_strategy.WeightStrategyBase` + `OrderGenerator` subclass |
| Models | LightGBM with the same core hyper-parameters as Qlib's `LGBModel`, plus a rank-IC early-stopping callback |
| Observability | `qlib.workflow.R` (params, metrics, report/positions/orders/fills/round-trips/coverage/market-state artifacts), `risk_analysis(mode='product')`, `indicator_analysis` |
| Risk model | `qlib.model.riskmodel.shrink.ShrinkCovEstimator` for the portfolio volatility estimate |

The only V2-specific data code is `v2_data.build_v2_features`, a lean builder
that computes exactly the arrays V2 reads; it is unit-tested against the full V0
builder for identical values.

## Results (see reports/v2_acceptance_report.md)

Headline run `output/v2_g4` (3.5% profit target, no stop, 45-day cap,
model/rule blend 0.5, 14% vol target, **<=12 names** as the design specifies,
<=1 new/day, **8% single-name weight cap**, no forced liquidation on the extreme
market state, soft drawdown tiers):

| metric | value | contract |
|---|---:|---|
| CAGR (after fees) | **+13.39%** | >= 25% |
| annualised volatility | 11.82% | <= 10% |
| maximum drawdown | **-15.00%** | >= -10% |
| round-trip win rate | **70.29%** | >= 70% (PASS) |
| complete quarters | 11 of 30 negative | 0 negative |
| holding frequency | 97.62% / 90.48% (252d / 63d minima) | >= 60% / >= 40% (both PASS) |
| round trips / average trade / profit factor | 956 / +1.76% / 1.84 | - |
| information ratio | 1.13 | - |

Positive in every calendar year: 2019 +10.7%, 2020 +1.6%, 2021 +24.2%, 2022 +10.4%,
2023 +11.1%, 2024 +12.8%, 2025 +14.8%, 2026 +18.6% (to 14 September).

Three of the seven contract checks pass.

> **Correction (round 21/22): the table above is one draw, not the expected result.**
> The rolling model's random seed cannot change the strategy's economics, yet nine
> seeds of the identical pipeline produce CAGR from **6.94% to 13.39%** - a mean of
> **9.93% with a standard deviation of 2.22pp** - while realised volatility moves by
> only 0.22pp (11.84% +/- 0.22pp) and the win rate by 1.56pp (68.92% +/- 1.56pp).
> The 13.39% headline is the maximum of those nine draws, +1.6 sd above their mean.
> The same nine-model panel through the compliance configuration gives
> 7.40% +/- 2.18pp CAGR at 10.19% +/- 0.28pp volatility, with only 2 of 7 seeds
> actually under the contract's 10% ceiling.
>
> Read every "best run" in this project as the maximum of a distribution with a
> ~2.2pp standard deviation: **differences between single runs below about 4.4pp of
> CAGR (2 sd) are draws, not results**. The table's real content is therefore that
> the strategy earns **about 10% a year at about 12% volatility (IR 0.84)**, and
> about 7% at a volatility that reliably clears the 10% ceiling.

> **Correction (round 22): the market ladder is removed from the headline.**
> Because single runs turned out to be draws, every headline alternative was re-run
> on all nine model seeds and compared *paired* (same nine models, one flag
> different). Removing the V2 market ladder is worth **+2.19pp of CAGR (t = 4.16,
> 8 of 9 seeds)** and **+3.36pp of win rate (t = 9.63, 9 of 9 seeds)** - the largest
> and most significant effect in the project, and the only one that improves the
> contract's two named objectives at once:
>
> | configuration (nine-seed mean) | CAGR | vol | max DD | win | seeds with win >= 70% |
> |---|---:|---:|---:|---:|---:|
> | `v2_g4` design ladder (previous headline) | 9.93% | 11.84% | -17.90% | 68.92% | 4/9 |
> | **`v2_notm` ladder off** | **12.12%** | 15.00% | -26.57% | **72.28%** | **8/9** |
> | `v2_sd2_f_*` ladder off, wider drawdown tiers | **12.39%** | 15.75% | -28.29% | **72.57%** | **8/9** |
> | `v2_y4` compliance (cap-scale 0.80) | 7.26% | **10.13%** | **-15.59%** | 71.45% | 8/10 |
>
> The cost is 3.2pp of volatility and 8.7pp of drawdown. The ladder is a leftover
> from the V0/V1 trend design: for this mean-reversion book the weak-market state
> (36.7% of all days) is where the edge is, and the ladder under-invests there. It
> is kept available (`--use-market-timing`, `--state-caps`) and the compliance
> configuration still uses it.
>
> **Correction (round 23): the ladder-off gain is exposure, not alpha - but the
> hold cap is worth 8 points of win rate.** Scaled to the contract's 10% volatility
> ceiling the ladder-off configuration ties the design ladder (9.97% vol, 7.36%
> CAGR against 9.88% / 7.44%), so round 22's +2.19pp was the extra exposure. Re-tuning
> the exit policy at that fixed risk budget then found the project's largest single
> effect: extending the hold cap from 45 to **120 days** raises the win rate by
> **+8.26pp (t = 27.8, 9 of 9 seeds)** for a CAGR difference that is not statistically
> detectable. The frozen 45-day cap was force-closing low-volatility names before the
> 3.5% profit target could be reached, and those forced exits were the losing trades.
>
> The recommended compliant configuration is therefore `h120` (market ladder off,
> volatility target 0.055, 120-day hold):
>
> | configuration (nine-seed mean) | CAGR | vol | max DD | win | win >= 70% | vol <= 10% |
> |---|---:|---:|---:|---:|---:|---:|
> | `r120` ladder off, vt 0.14, hold 120 | **11.22%** | 14.51% | -23.73% | **80.54%** | 9/9 | 0/9 |
> | `m120` ladder off, vt 0.09, hold 120 | 9.69% | 12.79% | -21.80% | **80.06%** | 9/9 | 0/9 |
> | **`h120` ladder off, vt 0.055, hold 120** | 6.62% | **9.65%** | **-15.85%** | **78.22%** | **9/9** | **9/9** |
> | `y4` design ladder, cap-scale 0.80 (previous compliance point) | 7.24% | 10.16% | -15.68% | 71.54% | 7/9 | 3/9 |
>
> `h120` beats `v2_y4` paired by **+6.68pp of win rate (t = 18.35)** and 0.52pp of
> volatility at an indistinguishable CAGR, and it meets **both** shape checks on
> every seed. The win-rate target is now met with a 7-10 point margin. The CAGR
> target is not: at the 10% ceiling the strategy earns 6.6%, and the contract needs
> an information ratio of 2.5 against the 0.69-0.89 measured here.

> **Correction (round 24): the frozen 3.5% profit target was the last big mistake,
> and the headline changes to `c2`.** Round 23 found the hold cap by accident;
> round 24 searched the exit policy properly (profit target x hold cap) with the
> risk budget pinned at the contract's ceiling. The 3.5% target - set in round 2 on
> a much weaker signal and never re-tested since - sits at the *bottom* of the
> range. Raising it to **7.5%** and holding to **120 days** is worth **+2.54pp of
> CAGR (t = 5.66, 9 of 9 seeds)** for 3.45pp of win rate, which still leaves the win
> rate at 74.8% (9 of 9 seeds over 70%).
>
> `c2` = `--use-market-timing 0 --vol-target 0.055 --profit-target 0.075 --hold 120`
> (nine-seed means):
>
> | configuration | CAGR | vol | max DD | win | IR | vol <= 10% | win >= 70% | checks |
> |---|---:|---:|---:|---:|---:|---:|---:|---:|
> | **`c2` target 7.5%, hold 120** | **9.16%** | **9.67%** | **-15.92%** | **74.76%** | **0.948** | **9/9** | **9/9** | **4 on 9/9** |
> | `c1` target 9.0%, hold 90 | 9.59% | 9.95% | -16.78% | 68.27% | 0.964 | 5/9 | 3/9 | 2-4 |
> | `c3` target 6.0%, hold 120 | 8.00% | 9.68% | -15.88% | 76.36% | 0.826 | 9/9 | 9/9 | 4 on 9/9 |
> | `v2_g4` (headline rounds 8-21) | 9.93% | 11.84% | -17.90% | 68.92% | 0.844 | 0/9 | 4/9 | 3 on 4/9 |
>
> Paired on the same nine models, `c2` against the old headline `v2_g4`: CAGR -0.76pp
> (t = -0.96, not significant), **win rate +5.85pp (t = 10.30, 9 of 9)**, volatility
> **-2.17pp**, drawdown **+1.98pp**. The same return with two points less risk and six
> points more win rate, and - unlike the old headline - it passes the volatility and
> win-rate checks on **every** seed.
>
> Year by year (`c2`, seed 42): +11.49% / +0.06% / +21.20% / +3.30% / +17.02% /
> +7.22% / +13.56% / +10.40%, total +119.68%, maximum drawdown -13.45%, win rate
> 73.53% over 442 round trips, positive in all eight calendar years.
>
> The compliant information ratio has moved from 0.685 to **0.948** over two rounds.
> The CAGR requirement is still the one that is out of reach: 9.2% against 25%, which
> needs IR 2.5.
>
> **Robustness (round 25).** Because rounds 23-24 replaced the two constants that
> everything else had been tuned around, the inherited parameters were re-opened
> under the new policy. All three come back the same way, so `c2` is a local optimum
> on every axis tested: **loss cutting is still rejected** (ATR stops 5-20%, trailing
> stops, stale-loser exits and profit locks each cost 1.2-7.5pp of CAGR *and* 6-20pp of
> win rate - a stop turns recoverable trades into realised losses); the **50/50 blend
> is confirmed** (0.65 is significantly worse on both CAGR and win rate, 0.35 is
> neutral); and the **account drawdown overlay is inert** at this exposure (two
> alternative tier sets reproduce `c2` bit for bit, because the volatility target
> already holds gross near 0.5).
>
> **Design changes (round 26).** With design changes authorised, one was tested and
> rejected - expressing the profit target in multiples of the position's own ATR
> instead of a flat percentage loses on every axis (the quiet names this strategy
> selects end up exiting at tiny gains, raising fee drag) - and the round-23/24 exit
> redesign was applied to the return end of the frontier. The result is a frontier of
> two points at the **same information ratio of about 0.95**:
>
> | configuration (nine-seed mean) | CAGR | vol | max DD | win | IR | checks on every seed |
> |---|---:|---:|---:|---:|---:|---|
> | `c2` ladder off, vt 0.055, tp 7.5%, hold 120 | 9.16% | **9.67%** | **-15.92%** | 74.76% | 0.948 | 4 of 7 |
> | `z1` ladder off, vt 0.14, tp 7.5%, hold 120 | **13.76%** | 14.58% | -23.18% | **75.39%** | 0.945 | 3 of 7 |
> | `v2_g4` (headline rounds 8-21) | 9.93% | 11.84% | -17.90% | 68.92% | 0.844 | 3 of 7 on 4/9 seeds |
>
> `z1` against `v2_g4` paired on the same nine models: **CAGR +3.83pp (t = 3.83,
> 8 of 9 seeds)** and **win rate +6.47pp (t = 8.07, 9 of 9)**. `z1` also dominates
> `notm`: win rate +3.11pp (t = 4.26, 9 of 9), volatility -0.43pp, drawdown +3.39pp,
> CAGR +1.64pp (not significant). The redesign lifted the whole frontier rather than
> tilting it - the same risk budget now buys about 1.8pp more CAGR and 6pp more win
> rate than before rounds 23-26.

> **Correction (round 4).** The round-3 headline (`v2_m2`, +11.40% CAGR, 70.01%
> win) was produced with a **bypassed position budget**: the V2 shortcut added in
> round 1 sat in front of the `max_positions` check and skipped it, so the book
> grew to 22 names against a configured cap of 6. The check is now enforced and
> covered by two regression tests; the honest headline within the design's
> 12-name cap is `v2_r1` above. Enforcing the cap costs 0.8pp of CAGR.

For reference V0 produced CAGR -4.16% with a 32.0% win rate, V1 -1.41% with
34.8%, and the first V2 configuration +1.00% with 51.4%.

Round-2 changes (CAGR +1.00% -> +10.32%):

1. **Exit policy.** An 8% stop costs about 4x the net alpha per holding day
   versus no stop with a 60-day cap, because it sells into volatility rather
   than into decay (`output/v2_research/rank_compare.csv`).
2. **Ranking signal.** A 50/50 blend of the rolling model with the causal rule
   composite (low volatility + low liquidity + distance below MA60) beats either
   alone; the model alone gives only +4.5% CAGR.
3. **Position sizing.** Six positions at up to 20% of NAV instead of twelve at
   8%: the same gross exposure with larger tickets, which cuts the minimum
   commission paid per fill.

Round-3 changes (CAGR +10.32% -> +11.40%, win rate 51.9% -> 70.0%):

4. **Minimum trim size.** Partial trims below 2% of NAV are skipped; the run had
   been paying the 5 CNY minimum commission on 2,927 shaves worth 39% of all
   fees. Total fee drag fell from 8.18% to 5.62% of NAV per year.
5. **No forced liquidation on the extreme market state.** The design's rule
   ("market extreme -> sell at the next close") systematically sold at the
   bottom: removing it lifted CAGR from +11.9% to +12.9% and the win rate from
   54.0% to 61.6%. New entries are still blocked while the cap is 0%.
6. **Smaller profit target.** 2.5% instead of 6%: the win rate rose from 61.6%
   to 68.5% for about 1.4pp of CAGR.
7. **Soft drawdown tiers** (70% exposure beyond an 8% drawdown, 45% beyond 12%)
   took the win rate to 70.0% and held the frequency checks above their floors.

Round-4/5 changes:

8. **Position budget enforced** (regression fix, see the correction above): the
   design's 12-name cap now binds, costing 0.8pp of CAGR and covered by two tests.
9. **Forward-return ranking label tested and rejected**: training the ranking model
   on the plain 20-day forward return rather than the policy return gives a
   top-decile alpha of +0.37pp against +0.44pp, so the signal, not the label, is
   the limit (`v2m_tp5h40_predictions_fwd20.quarters.csv`).
10. **Profit target and holding period swept**: removing the target entirely
    collapses the win rate to 46-48%; raising it from 2.5% to 3.5% with a 45-day
    cap lifts CAGR from +10.60% to +13.56% and IR from 0.89 to 1.07 while the win
    rate falls from 70.3% to 65.4%. The two contract objectives trade off directly
    through this parameter, so both ends of the frontier are reported
    (`v2_r1`, `v2_u2`, `v2_u4`, `v2_t7`).

Round-6 changes:

11. **A-share microstructure features built, tested and rejected for the account.**
    Seventeen candidate features (limit-up/down counts, overnight vs intraday
    decomposition, Parkinson range volatility, Amihud illiquidity, up/down volume
    ratio, turnover acceleration) were screened: the strongest reach rank IC -0.104
    (parkinson20) and -0.087 with t = -55 (limit_up_20), well beyond the original
    feature set. Adding the top sixteen raises the model's top-decile alpha from
    +0.44pp to +0.50pp, **but the account gets worse**: +8.13% and +12.39% CAGR
    against +10.60% and +13.56% for the same configurations without them. The
    extended matrix is kept as a research artefact (`v2m_x`), not used for
    the headline.
12. **Signal application made memory-lean** (searchsorted over the calendar instead
    of a per-row dict lookup), removing ~300 MB of peak memory so the extended runs
    no longer trip the 3 GB watchdog.

Round-7 changes:

13. **The design's win-probability entry gate tested again and rejected.** Under the
    current setup a 0.50/0.55 threshold on the calibrated win probability costs
    4-8pp of CAGR and drops the 63-day holding frequency to 0: it removes more
    opportunity than the hit rate it buys.
14. **Market timing quantified.** Removing the ladder lifts CAGR from +10.60% to
    +11.31% but raises volatility from 11.90% to 15.30% and drawdown from -18.53%
    to -25.08%. The ladder is kept.
15. **Execution pressure matrix completed** (report section 7c): friction 5bp
    +9.80%, 20bp +7.69%, one-day signal delay +9.54%, 1,000,000 CNY account
    +2.26% (the 5bp participation cap starves the book). The 5bp row landing below
    the 10bp row shows that ~1pp differences between nearby configurations are path
    noise - which is why the report reports a frontier rather than a single winner.

Round-8 changes (CAGR +10.60% -> +13.39%, drawdown -18.53% -> -15.00%, IR 0.89 -> 1.13):

16. **Single-name weight cap tightened from 20% to 8%.** This is the largest single
    improvement since round 3. A 20% cap let one new position take a very large slice
    whenever the book was thin; the resulting volatility estimate then forced the
    whole book smaller. With an 8% cap the book stays even, the estimator allows more
    gross, and both return and drawdown improve (the sweep is 0.20 -> 0.12 -> 0.08:
    +10.60% / +12.25% / +13.39% CAGR).
17. **Profit target 3.5% with a 45-day cap** re-tuned around the new cap
    (3.0%/45d +12.67%, 3.5%/60d +8.90%, 4.0%/45d +11.63%).
18. **Correlation filter rejected** (`v2_f1`): rejecting candidates correlated
    with current holdings drops CAGR to +3.81% and the win rate to 44.5%, because an
    illiquidity-driven signal naturally selects correlated names.

Round-9 changes:

19. **Rule composite rebuilt from the stronger features and rejected**
    (`combo3p`, `combo4l`, `combo5l`): CAGR +8.23%, +8.51%, +6.80%
    against +13.39% for the original combo3. Together with round 6 this is a
    three-times-repeated result: the features with the highest cross-sectional IC
    (Parkinson range volatility, limit-up counts, 5-day maximum return) make the
    account *worse*, through the model and through the rule composite alike.
    A bucket's average forward return is not the same as a name surviving a 3.5%
    target and a 45-day cap under the frozen execution rules.
20. **Headline reproduction verified bit-identical** after the composite refactor
    (`v2_g4b` matches `v2_g4` to every recorded digit).
21. **Buy-price-cap audit added** (`scripts/audit_price_cap.py`), the check the design
    asks for in section 4.1: of 959 buy orders, 508 filled, 449 partially filled and
    2 rejected, so the 3% cap is not costing measurable opportunity.

Round-10 changes:

22. **Weak-market cap swept** (35% -> 50% -> 70%). A 50% cap gives a higher return and
    hit rate (+14.53% CAGR, 72.75% win) but a worse risk profile (13.38% vol, -18.31%
    drawdown) and a lower IR (1.09 versus 1.13), so the design's 35% is kept for the
    headline; 70% is clearly worse on every risk measure. `v2_m1b` is reported on the
    frontier as the higher-return, higher-risk alternative.
23. **Overlay overlap measured**: the market ladder binds on 36% of days, the
    volatility target on 41%, and the drawdown tiers almost never bind alone - with a
    50% weak cap the tiers become irrelevant altogether (`v2_m3b` == `v2_m4b`).

Round-11 changes:

24. **Every loss-cutting mechanism rejected, and the harvest side tested too.** A
    stale-loser exit (close a position still below entry after 15/20/25/30 days) hurts
    at every horizon (+10.64% to +12.12% against +13.39%), and a profit lock (give back
    at most 1-3% from the running maximum once the target is armed) also hurts
    (+7.55% to +12.24%). Together with the price stops this is the fourth independent
    confirmation that the reversal alpha recovers after drawdown and must be held.
25. **Ranker ablation completed** - the strongest single result in the project:
    rule composite alone +4.71% (IR 0.36), model alone +5.16% (IR 0.41), the 50/50 blend
    +13.39% (IR 1.13). Combining the two rankers nearly triples either one and cuts the
    drawdown by ten points.
26. **Consolidated experiment log written** (`reports/v2_experiment_log.md`): every
    mechanism tested across eleven rounds with its run id, headline numbers and verdict.

Round-12 changes:

27. **Rank ensemble with a second model rejected** (`v2_r4e`): CAGR +10.00% against
    +13.39%. The forward-return model's own alpha is lower, so it dilutes the blend, and
    it doubles prediction memory (three of four ensemble runs tripped the watchdog).
28. **Composite component weights swept** (low-vol 2x, illiquidity 2x, half reversal,
    half illiquidity): +5.40% to +10.36% against +13.39% for equal weights. Equal
    weighting is a clear local optimum.
29. **60-month training window rejected**: top-decile alpha +0.32pp against +0.44pp for
    the design's 36 months.
30. **Universe liquidity floor swept** (0.02 / 0.20 / 0.35): +8.23% / +13.39% / +4.55% CAGR
    with IR 0.72 / 1.13 / 0.37. The design's 0.20 floor is a local optimum, and the alpha
    cannot be separated from the illiquid tail by filtering the universe.

Round-14 changes:

31. **Faster volatility reaction rejected**: a 20-day portfolio volatility window gives
    CAGR +12.97% with drawdown -18.06% (worse than the 60-day window).
32. **Market-volatility exposure brake rejected** (`v2_v3g`, `v2_v4g`):
    scaling the gross cap by a market-vol budget cut the mean exposure to 0.91/0.85 with a
    minimum of 0.24/0.19 yet left the drawdown at -15.1%/-16.8% while the CAGR fell to
    +9.9%. Cutting exposure after volatility rises only locks in the loss.
33. **Drawdown structure quantified across 109 runs**: the drawdown/volatility ratio has a
    mean of 1.70 and a median of 1.61, and the headline `v2_g4` has the lowest ratio
    found (1.27); CAGR correlates -0.74 with it. The -10% drawdown floor therefore implies
    about 7.9% realized volatility and a CAGR near 8.9% regardless of the overlay used.

Round-15 changes:

34. **The volatility target does not control volatility.** Cutting it from 0.14 to 0.115 moved
    realized volatility only from 11.82% to 11.66%: the book is entry-rate limited (one new
    name per day, 12-name cap) and never reaches the vol-scaled target. **Scaling the
    market-ladder gross caps is the effective control** - gross 56.5% -> 43.8% and realized
    volatility 11.82% -> 9.25% as the scale goes from 1.00 to 0.70.
35. **First four-check configuration** (`v2_y4`, cap scale 0.80 + 3.0% target + 120-day cap):
    volatility 9.88% and win rate 70.66% together with both holding-frequency floors, at
    CAGR +7.44% and drawdown -14.86%. `v2_z1` also passes four checks with the best hit
    rate measured (71.74%).
36. **Buy-premium cap confirmed non-binding**: tightening it from the design's 3% to 2% / 1%
    costs CAGR (+11.42% / +8.93%).
37. **Memory hardening**: the microstructure features are now opt-in in the lean builder and
    the runner releases everything the market state, base mask and composite no longer need,
    which removed the watchdog aborts that were plaguing recent sweeps.

Round-16 changes:

38. **The constrained frontier is stable.** Four independent parameterisations at the 10%
    volatility cap and the 70% win gate all land at CAGR 6.9-7.5%: `v2_y4` (+7.44%), `v2_z1`
    (+6.94%), `v2_aa3` (+7.45%, 8 names at 12%). Relaxing either gate (two entries per day,
    12% single-name weight, 16 names) pushes volatility back over 10% and loses the check.
39. **Final position recorded** in `reports/v2_experiment_log.md` section 9 and
    `reports/v2_evidence_index.md`: two points summarise the frontier - `v2_g4` maximises
    return (+13.39% CAGR, IR 1.13, 70.29% win) and `v2_y4` maximises contract compliance
    (+7.44% CAGR, 9.88% volatility, 70.66% win, four of seven checks).

After thirteen rounds every axis that can be varied without changing the data itself has
been swept, and the headline `v2_g4` is the best of 100+ recorded runs. See
`reports/v2_experiment_log.md` for the full list with verdicts.

The contract is still not met, and section 8 of the report explains why with the
measurements behind it: costs run 6-11% of NAV per year at the 100k scale (3.8%/yr
at 1M), the model adds only about +0.45pp per trade, and the best configuration
reaches IR 0.83, while 25% CAGR inside a 10% volatility cap needs IR 2.5.

## Layout

```text
sleeve/a/
├── configs/                    frozen V0/V2 configuration
├── docs/chats_001.md           V0 design contract
├── docs/chats_002.md           V2 design contract
├── src/lowvol_trend/
│   ├── v2.py                   base universe, market ladder, policy labels
│   ├── v2_policy.py            the shared ExitPolicy + entry/exit simulator
│   ├── v2_data.py              lean 3 GB feature builder
│   ├── v2_strategy.py          V2 strategy, gates, Qlib account wiring
│   ├── qlib_backtest.py        Exchange / Quote / OrderGenerator / WeightStrategy
│   ├── ledger.py, metrics.py   round trips, NAV/frequency/acceptance metrics
│   └── ...                     V0/V1 modules kept for comparison runs
├── scripts/                    pyrun.py launcher + build/run/research/report
├── tests/                      pure-rule, no-look-ahead, builder-equivalence tests
└── reports/                    acceptance reports
```

## Large directories

`cache/`, `output/` and `mlruns/` are symlinks to
`/Volumes/lexar_4t/code/data/qlib/sleeve/a/{cache,output,mlruns}`, so the
repository keeps only code and reports.

## Known limitations

- Daily bars cannot reconstruct ST/delisting status, closing-auction queues or
  intraday fills; limit-up/limit-down and suspension blocks are modelled.
- Dividend cash is not credited; adjusted-return figures are reported alongside
  raw ones where it matters.
- The most recent quarter's labels are not fully matured, so the last quarter is
  excluded from label statistics.
- The profit-target exit fills at the next close rather than intraday, so part
  of the move is given back; this is deliberately conservative.
