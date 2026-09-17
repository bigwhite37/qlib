# V2 research evidence index

Every number quoted in `reports/v2_acceptance_report.md` traces back to one of
these files. Paths are relative to `/Volumes/lexar_4t/code/data/qlib/sleeve/a/`.

| artefact | produced by | headline finding |
|---|---|---|
| `cache/signal_scan/feature_summary.csv`, `decile_scan.csv` | `scripts/research_signal_scan.py` | Decile spreads of 30 panel features against the V0 policy labels. Market-state features dominate; volatility, liquidity and reversal follow. No single feature reaches a 70% win rate. |
| `output/v2_research/ic_scan.csv` | `scripts/research_ic_scan.py` | Rank IC against 5/10/20-day forward returns, 2016-2026. At h=5: vol20 -0.070, amp20 -0.072, dist_ma60 -0.066, atr_ratio -0.064, vol60 -0.063, log_amount -0.053, ret20 -0.047, all with t < -15. |
| `output/v2_research/ceiling_analysis.csv` | `scripts/research_ceiling.py` | Top-decile (low liquidity) forward return +1.12% raw / +1.49% adjusted over 20 days versus +0.09% raw for the universe, win rate 51.6%. |
| `output/v2_research/policy_grid.csv` | `scripts/research_policy_grid.py` | 9 exit policies x 12 entry rules. Every combination has negative unconditional expectancy; the 2 ATR stop with trailing destroys more value than it saves. |
| `output/v2_research/policy_winrate_grid.csv` | `scripts/research_policy_winrate.py` | Win-rate frontier: 43.7% (no target, h=10) to 62.1% (target 2%, h=40, no stop) while expectancy stays between -0.5% and -0.8% per trade. |
| `output/v2_research/rank_compare.csv` | `scripts/research_rank_compare.py` | Eleven ranking signals x four exit policies under the shared simulator. Net alpha per holding day: combo3+model top 1% with a 5% target and a 40-day cap 0.0294 x100/day, model alone 0.0075 with the 8% stop and 0.0311 with the 40-day cap. The stop, not the ranker, was the binding problem. |
| `cache/v2m_tp5h40_labels.npz` + `v2m_tp5h40_features_part*.npy` | `scripts/build_v2_matrix.py` | 8.14M policy-labelled base rows (5% target, no stop, 60-day cap) with 53 causal features, built in date chunks inside the 3 GB budget. Unconditioned win rate 54.95%, mean net return -0.78%, average hold 16.8 days. |
| `cache/v2m_tp5h40_predictions.quarters.csv` | `scripts/build_v2_predictions.py` | 31 rolling quarters. Top decile of the predicted relative return +0.05% realised net per trade versus -0.39% for the universe (+0.44pp alpha), win rate 57.5% against 57.4%. |
| `output/v2_research/account_runs.csv` | `scripts/collect_v2_runs.py` | Every account run with CAGR, vol, drawdown, IR, win rate, fees per year and turnover per year. Best: `v2_m2` at +11.40% CAGR, 12.10% vol, -18.96% drawdown, 70.01% win rate, IR 0.94. |
| `output/v2_m2/*` | `scripts/run_qrun_v2.py` | Headline run: +11.40% CAGR, 70.01% win rate, -18.96% drawdown, 98.02%/92.06% holding frequency, 3 of 7 contract checks passing. |
| `mlruns/<experiment>/<run>/artifacts/*.pkl` | Qlib `R` recorder | report_normal, positions_normal, portfolio_analysis, indicator_analysis, orders, fills, round_trips, open_positions, daily_decisions, constraints_daily, coverage_daily, market_daily, exit_log, acceptance. |

## Cost measurement

Recorded fills for the round-2 headline run (`v2_e4`): 4,995 fills for 1,108 round
trips, average fill 5,126 CNY, average fee 32bp per fill, total fees 8.2% of the
initial NAV per year on 34.5x NAV of annual turnover. 2,927 of those fills were
partial trims below 2% of NAV, paying the 5 CNY minimum commission on ticket
sizes that did not justify it; suppressing them (`min_trim_weight = 0.02`) cut
the drag to 5.62%/yr in `v2_h1`. The same strategy at a 1,000,000 CNY account
(`v2_band_c1m`) pays 3.8%/yr, because the minimum commission stops dominating
the ~10bp friction charge.

## Round-3 findings

| change | evidence | effect |
|---|---|---|
| skip partial trims below 2% of NAV | `v2_e4` -> `v2_h1` | fees 8.18% -> 6.68%/yr, CAGR +10.32% -> +11.88%, win 51.9% -> 54.0% |
| do not force a liquidation on the extreme market state | `v2_h1` -> `v2_h5` | CAGR +12.90%, win 61.6%, average hold 19.4 -> 29.6 days, fees 5.62%/yr |
| profit target 6% -> 2.5% | `v2_h5` -> `v2_k2` | win 61.6% -> 68.5%, CAGR 12.90% -> 11.54% |
| soft drawdown tiers (0.08:0.70, 0.12:0.45) | `v2_k2` -> `v2_m2` | win 68.5% -> 70.0%, CAGR +11.40% |
| model-only ranking (blend 1.0) | `v2_k4` | CAGR collapses to +4.51%: the rule composite carries most of the ranking power |
| tighter trim mode (`close_weakest`) | `v2_f1` | CAGR +3.81%: closing whole positions removes exposure faster than it saves cost |
| drawdown tiers that trigger early (0.06/0.09/0.12) | `v2_h7`, `v2_h9` | the 63-day frequency falls to 0 and CAGR to +4.0%: over-de-risking fails the frequency contract |

## Why the contract targets remain out of reach

CAGR is approximately information ratio times the volatility budget. The best
configuration measured reaches IR 0.94 (+11.40% CAGR at 12.10% vol) with a 70.01%
round-trip win rate. A 25% CAGR inside a 10% volatility cap needs IR 2.5 - about
2.7 times the best value found here. Costs still run 5-6.5% of NAV per year, so a
large part of the gap is friction rather than signal, and the drawdown constraint
binds hardest: the -19.0% maximum comes almost entirely from the February-July 2024
small-cap episode, so holding MDD at -10% caps realized volatility near 6% and CAGR
near 6%.

## Round-4 findings

| change | evidence | effect |
|---|---|---|
| enforce the position budget (regression fix) | `v2_m2` -> `v2_r1` | the round-3 headline had grown to 22 names against a cap of 6; enforcing the design's 12-name cap costs 0.8pp of CAGR (+11.40% -> +10.60%) and keeps the 70.28% win rate |
| net exiting positions out of the budget | `v2_p3` -> `v2_q3` | counting names already queued for exit starved deployment: CAGR +6.85% -> +8.59% at the same cap |
| small position budgets | `v2_n1` (6 names), `v2_n3` (4 names) | CAGR +2.93% / +5.50%: too few slots cannot deploy the target exposure, and the win rate rises (70.3% / 74.8%) as the book concentrates |
| large position budgets | `v2_p2` (30 names) | CAGR +5.69%: more slots means buying the lower-ranked tail, and the average trade falls to +0.53% |
| 12 names, 1 entry/day, 20% weight cap | `v2_r1` | best honest configuration: +10.60% CAGR, 70.28% win, -18.53% drawdown, IR 0.89 |


## Round-5 findings

| change | evidence | effect |
|---|---|---|
| ranking model trained on the forward return instead of the policy return | `pred_fwd20.log`, `v2m_tp5h40_predictions_fwd20.quarters.csv` | no gain: top-decile alpha +0.37pp versus +0.44pp for the policy label, so the ranking signal, not the label, is the limit |
| no profit target at all | `v2_t1` (hold 20), `v2_t2` (hold 40) | win rate collapses to 46-48% and CAGR to +6.7-8.6%: the target is what creates the hit rate |
| profit target 2.5% -> 3.5%, hold 45-60 days | `v2_r1` -> `v2_t7` | CAGR +10.60% -> +13.56%, IR 0.89 -> 1.07, win 70.3% -> 65.4%: the two contract objectives trade off directly through the target |
| profit target 3.5%, hold 90 days | `v2_u2` | win 72.57% with CAGR +10.27%, drawdown -20.5%: the highest hit rate measured at a viable return |
| segment stability | `v2_t7`, `v2_r1` segment tables | neither configuration collapses in any segment: t7 runs 13.0% / 11.5% / 11.8% / 19.9% CAGR across dev, validation, 2023-24 and 2025-26 |

## Current frontier (all runs share the engine, fees, timing and the 12-name cap)

| configuration | CAGR | vol | max DD | IR | win | drawdown-vs-vol |
|---|---:|---:|---:|---:|---:|---:|
| `v2_r1` tp2.5 hold60 (headline: win gate + frequency pass) | +10.60% | 11.90% | -18.53% | 0.89 | 70.28% | 1.56 |
| `v2_u2` tp3.5 hold90 | +10.27% | 12.65% | -20.46% | 0.81 | 72.57% | 1.62 |
| `v2_u4` tp3.5 hold60 vol15 | +11.50% | 12.71% | -16.74% | 0.90 | 67.96% | 1.32 |
| `v2_t7` tp3.5 hold45 (best CAGR and IR) | +13.56% | 12.68% | -18.20% | 1.07 | 65.42% | 1.44 |

Nothing reaches a 25% CAGR, and the best IR measured (1.07) is still far from the 2.5 that a
10% volatility budget would require. The drawdown-to-volatility ratio near 1.4-1.6 means a
-10% drawdown cap implies roughly 7% realized volatility and therefore roughly 7% CAGR at the
best measured IR.


## Round-6 findings

| artefact | finding |
|---|---|
| `output/v2_research/new_feature_ic.csv` | Rank IC of 17 candidate A-share microstructure features (2016-2026). Strongest: parkinson20 (range-based volatility) -0.104 at h=20, limit_up_20 -0.087 (t=-55), ret_max5 -0.077, limit_up_5 -0.073, corr_ret_vol20 -0.060, vol_ratio_ud -0.050, turnover_accel -0.047, ret_min5 +0.047, amihud20 +0.030. All far stronger than the best feature in the original set (vol20, -0.070 at h=5). |
| `cache/v2m_x_predictions.quarters.csv` | Adding the 16 strongest of those features to the rolling model raises its top-decile realised net return from +0.0005 to +0.0010 and the alpha from +0.44pp to +0.50pp. |
| `v2_x1`, `v2_x3` | ...but the account gets worse: CAGR +8.13% / +12.39% against +10.60% / +13.56% for the same configurations with the original feature set. Label-level decile alpha is not the account objective - the book holds ~10 names out of ~2,900 candidates, and the new features tilt it toward recently speculative names whose behaviour under the frozen exit policy differs. The extended feature set is therefore rejected and `v2m_x` is kept only as a research artefact. |
| `v2_strategy.apply_v2_signals` | Signal application now uses searchsorted over the trading calendar instead of a per-row dict lookup, removing ~300 MB of peak memory; the extended-feature runs no longer hit the 3 GB watchdog. |

## Standing conclusion after six rounds

The honest frontier has moved from CAGR -4.16% / win 32.0% (V0) to +10.60% / 70.28%
(`v2_r1`) and +13.56% / 65.42% (`v2_t7`), with both holding-frequency checks passing.
Three of the seven contract checks pass. The binding limit is the information ratio: the best
measured value is 1.07, while a 25% CAGR inside a 10% volatility cap requires 2.5. Six rounds
of work have raised it from 0.15 to 1.07 by fixing execution semantics (stops, forced
liquidation, trim churn), by blending a model with a rule composite, and by tuning the position
budget - but not by finding materially more alpha. The additional features tested this round
confirm that the remaining alpha in daily OHLCV for this universe is small relative to the
cost of trading it.


## Round-7 findings

| change | evidence | effect |
|---|---|---|
| win-probability entry gate (the design's core filter) | `v2_e1` (0.50), `v2_e2` (0.55), `v2_e5` (0.52) | rejected again under the current setup: CAGR +6.56% / +2.57% / +4.96% against +10.60% without the gate, and the 63-day holding frequency collapses to 0. The gate costs more opportunity than the hit rate it buys. |
| ranking blend weight | `v2_e3` (0.4), `v2_r1` (0.5) | 0.5 is the optimum: +7.71% at 0.4 against +10.60% at 0.5 |
| market timing ladder removed | `v2_e4` | CAGR +11.31% but volatility 15.30% and drawdown -25.08%: the ladder costs 0.7pp of CAGR and buys 3.4pp of volatility and 6.6pp of drawdown |
| execution pressure matrix | `pressure_matrix.csv`, report section 7c | friction 5bp: +9.80%; friction 20bp: +7.69%; signal delayed one day: +9.54% with drawdown -22.72%; a 1,000,000 CNY account: +2.26% because the participation cap (5bp of 20-day ADV) starves the book and the 63-day frequency falls to 0 |
| friction ordering | `v2_pr1` vs `v2_r1` | 5bp friction gives a lower CAGR than 10bp, which shows that differences of about a point of CAGR between nearby configurations are path noise, not signal |


## Round-8 findings

| change | evidence | effect |
|---|---|---|
| correlation filter on new candidates | `v2_f1` (threshold 0.80) | rejected: CAGR +3.81% and win rate 44.5% against +10.60% / 70.3%. Rejecting correlated candidates removes most of the opportunity set for an illiquidity-driven signal where the top names are naturally correlated. |
| single-name weight cap 0.20 -> 0.12 -> 0.08 | `v2_r1`, `v2_f4`, `v2_g4` | the largest single improvement since round 3: CAGR +10.60% -> +12.25% -> +13.39%, IR 0.89 -> 1.01 -> 1.13, drawdown -18.53% -> -20.08% -> -15.00%, win rate unchanged near 70%. A 20% cap let a new position take a very large slice whenever the book was thin, so the volatility estimate forced the whole book smaller. |
| profit target / holding period around the new cap | `v2_g4` (3.5%/45d), `v2_h2b` (3.0%/45d), `v2_h3b` (3.5%/60d), `v2_h5b` (4.0%/45d) | 3.5% with a 45-day cap is the optimum: +13.39% / +12.67% / +8.90% / +11.63% |
| 12% stop, 30-day cap, 16% vol target, 15 names | `v2_j1`, `v2_j2`, `v2_j3`, `v2_j4` | all worse: +10.37%, +11.33%, +12.21%, +11.87%. A stop still destroys value even at 12%. |

### New headline: `v2_g4`

2.5% target -> 3.5%, 60-day cap -> 45 days, and a 8% single-name weight cap:

| metric | round 7 headline (`v2_r1`) | round 8 headline (`v2_g4`) | contract |
|---|---:|---:|---:|
| CAGR | +10.60% | **+13.39%** | >= 25% |
| annualised volatility | 11.90% | 11.82% | <= 10% |
| maximum drawdown | -18.53% | **-15.00%** | >= -10% |
| round-trip win rate | 70.28% | 70.29% | >= 70% (PASS) |
| information ratio | 0.89 | **1.13** | - |
| profit factor | 1.62 | **1.84** | - |
| holding frequency (252d / 63d) | 98.0% / 92.1% | 97.6% / 90.5% | both PASS |

The new headline is positive in **every calendar year**: 2019 +10.7%, 2020 +1.6%, 2021 +24.2%,
2022 +10.4%, 2023 +11.1%, 2024 +12.8%, 2025 +14.8%, 2026 +18.6% (to 14 September). Segment CAGRs
are 10.7% / 11.7% / 11.9% / 19.9% across dev, validation, 2023-24 and 2025-26.


## Round-9 findings

| change | evidence | effect |
|---|---|---|
| rule composite rebuilt from the stronger features (parkinson20, limit_up_20) | `v2_k1c` (combo3p), `v2_k3c` (combo4l), `v2_k4c` (combo5l) | all worse than the original combo3: CAGR +8.23% / +8.51% / +6.80% against +13.39%. The two features with the highest standalone IC (parkinson20 -0.104, limit_up_20 -0.087) degrade the blend in both halves - as a model input (round 6) and as a rule component (round 9). |
| composite code refactor verified | `v2_g4` vs `v2_g4b` | bit-identical reproduction: CAGR 0.133935, vol 0.118185, drawdown -0.149953, win 0.702929, 956 round trips, final NAV 263,277.40. The headline stands after the refactor. |
| buy-price-cap audit (design section 4.1) | `scripts/audit_price_cap.py`, `output/v2_research/price_cap_audit.csv` | of 959 buy orders in the headline run: 508 filled, 449 partially filled (quantity capped), 2 rejected. The 3% cap is therefore not costing measurable opportunity, and partially filled names behaved like fully filled ones (+2.04% vs +1.85% over 20 days). |

### Why the strongest features do not help

Across rounds 6 and 9 the same pattern appears three times: features with the highest
cross-sectional IC against forward returns (Parkinson range volatility, limit-up counts,
5-day maximum return) make the account worse whether they enter through the model or
through the rule composite. A cross-sectional IC measures the average forward return of a
bucket; the account instead asks whether a name, bought at the next close under a 3% price
cap, survives a 3.5% profit target and a 45-day cap without being stopped out in between.
Names selected by volatility-of-range and limit-up activity are precisely the ones whose
20-day distribution has the fattest right tail and the widest dispersion, so their bucket
mean is high while their realised round-trip under the frozen exit policy is not.


## Round-10 findings

| change | evidence | effect |
|---|---|---|
| weak-market gross cap 35% -> 50% | `v2_m1b` | CAGR +13.39% -> +14.53%, win 70.29% -> 72.75%, but volatility 11.82% -> 13.38% and drawdown -15.00% -> -18.31%; IR falls 1.13 -> 1.09, so the headline (best IR and best drawdown) stays at the design's 35% |
| weak-market cap 70% | `v2_m2b` | CAGR +12.44% with volatility 14.87% and drawdown -26.89%: relaxing the ladder this far destroys the risk profile without adding return |
| drawdown tiers removed or softened on top of a 50% weak cap | `v2_m3b`, `v2_m4b` | identical (+13.91%, drawdown -17.69%): once the weak cap is 50% the tiers rarely bind, which shows the two overlays overlap rather than complement |
| exposure binding in the headline | `v2_g4/daily_decisions.csv` | the market ladder is the binding constraint on 36% of days (32% at the 35% weak cap, 4% at zero) while the volatility target binds on 41%; the drawdown tiers almost never bind alone |


## Round-12 findings

| change | evidence | effect |
|---|---|---|
| rank ensemble with a second model trained on the forward-return label | `v2_r4e` | CAGR +10.00% against +13.39%: the second model's alpha is lower (+0.37pp vs +0.44pp) so it dilutes the blend, and it doubles prediction memory (three of four ensemble runs hit the 3 GB watchdog) |
| composite component weights | `v2_w1e` (2/1/1), `v2_w2e` (1/1/0.5), `v2_w3e` (1/0.5/1), `v2_w4e` (1/2/1) | +5.40% / +7.84% / +10.36% / +9.85% against +13.39% for equal weights: equal weighting is a clear local optimum |
| 60-month training window instead of 36 | `pred_t60.log` | top-decile alpha +0.32pp vs +0.44pp - worse |

### Coverage of the search after twelve rounds

Every axis that can be varied without changing the underlying data has been swept: entry
signal, ranking label, model features, training window, model ensemble, blend weights,
composite definition and component weights, exit rules (stop, trailing, target, lock,
stale, time cap, extreme liquidation), position budget, single-name weight cap,
correlation filter and replacement, daily entry cap, volatility target, drawdown tiers,
market-ladder caps, the win-probability gate, trim sizing and the execution wrapper
(friction, signal lag, account size). The headline `v2_g4` is the best of 100 recorded
runs, and `reports/v2_experiment_log.md` lists each of them with a verdict.


## Round-13 findings

| change | evidence | effect |
|---|---|---|
| V2 base-universe liquidity floor 0.02 / 0.20 / 0.35 | `v2_u6f` / `v2_g4` / `v2_u1f` | CAGR +8.23% / +13.39% / +4.55%, IR 0.72 / 1.13 / 0.37 - the design's 0.20 floor is a local optimum and the alpha cannot be separated from the illiquid tail by filtering the universe |

## Standing conclusion after thirteen rounds

The contract's conjunction (CAGR >= 25%, volatility <= 10%, drawdown >= -10%, win rate >= 70%)
is met on the win rate (70.29%) and both holding-frequency floors. The return requirement needs
an information ratio of about 2.5; the best measured across 100+ runs and every axis is 1.13,
and the drawdown-to-volatility ratio of 1.27 means the -10% drawdown floor alone caps CAGR near
9% at that IR.

Progress: CAGR -4.16% -> +13.39%, win rate 32.0% -> 70.29%, IR 0.15 -> 1.13. Almost all of it
came from fixing execution and portfolio-construction leaks (the forced liquidation at the
market extreme, minimum-commission trim churn, a bypassed position budget, the 20%->8%
single-name weight cap) rather than from finding more alpha. The assumption that a stronger
cross-sectional signal would help was falsified three times (rounds 6, 9, 12) and the
illiquidity-tail hypothesis was closed in round 13.


## Round-14 findings

| change | evidence | effect |
|---|---|---|
| portfolio volatility estimation window 60 -> 20 / 30 days | `v2_v1g`, `v2_v2g` | 20 days: CAGR +12.97%, drawdown -18.06%, IR 1.09 - faster reaction makes the drawdown *worse*, not better |
| market-volatility exposure brake (gross cap scaled by 25% / 20% market vol budget) | `v2_v3g`, `v2_v4g` | CAGR +9.90% / +9.95%, drawdown -15.10% / -16.82%: mean cap multiplier 0.91 / 0.85 and a minimum of 0.24 / 0.19, yet the drawdown is unchanged. Cutting exposure *after* volatility rises only locks in the loss. |
| structure of the drawdown across all runs | `account_runs.csv` (109 runs with 300+ trades) | drawdown/volatility ratio: mean 1.70, median 1.61, range 1.26-3.07. The headline `v2_g4` has the **lowest ratio found (1.27)**, and CAGR correlates -0.74 with the ratio. |

### What this establishes

The drawdown/volatility ratio is a property of the strategy's return distribution, not of
any exposure overlay. Cutting exposure in response to rising volatility reduces return
roughly proportionally while leaving the drawdown ratio unchanged or worse. Since the
headline already sits at the best end of that distribution (1.27), the -10% drawdown floor
implies a realized volatility of about 7.9% and therefore a CAGR near 8.9% - independent of
which overlay is used.


## Round-15 findings

| change | evidence | effect |
|---|---|---|
| portfolio volatility target 0.14 -> 0.115 | `v2_w1` | realized volatility moved only 11.82% -> 11.66%: the volatility target is **not** the exposure control, because the book is entry-rate limited (one new name per day, 12-name cap) and never reaches the vol-scaled target |
| market-ladder gross caps scaled by a factor | `v2_x1h` (0.85), `v2_x2h` (0.70), `v2_y4` (0.80) | this **is** the effective exposure control: gross 56.5% -> 51.4% -> 43.8% and realized volatility 11.82% -> 10.86% -> 9.25% |
| **cap scale 0.80 + 3.0% target + 120-day cap** | `v2_y4` | **first configuration to pass four contract checks**: volatility 9.88% (<=10) and win rate 70.66% (>=70) together with both holding-frequency floors, at CAGR +7.44% and drawdown -14.86% |
| cap scale 0.80 + 3.2% target | `v2_z1` | also four checks, with the best hit rate measured (71.74%) at CAGR +6.94% |
| buy-premium cap tightened from 3% to 2% / 1% | `v2_w4`, `v2_w3` | CAGR +11.42% / +8.93%: the design's 3% cap is not costing opportunity |

### Two points on one frontier

| configuration | CAGR | vol | max DD | win | checks |
|---|---:|---:|---:|---:|---:|
| `v2_g4` maximum return | **+13.39%** | 11.82% | -15.00% | 70.29% | 3/7 (win + 2 frequency) |
| `v2_m1b` | +14.53% | 13.38% | -18.31% | **72.75%** | 3/7 |
| `v2_y4` contract compliance | +7.44% | **9.88%** | -14.86% | **70.66%** | **4/7** (vol + win + 2 frequency) |
| `v2_z1` | +6.94% | 9.93% | -15.33% | **71.74%** | **4/7** |

Both the stated objectives are met by `v2_g4` (win rate and CAGR), and adding the 10%
volatility cap costs about six points of CAGR (`v2_y4`). No configuration reaches the
25% CAGR, the -10% drawdown or the all-positive-quarters checks.


## Round-16 findings

| change | evidence | effect |
|---|---|---|
| two entries per day at the same cap scale | `v2_aa1` | CAGR +6.67%, volatility 10.12% - the vol check is lost by 0.12 points |
| single-name weight 0.08 -> 0.12 | `v2_aa2` | CAGR +5.52%, volatility 10.89%: rejected |
| 8 names at 12% weight instead of 12 at 8% | `v2_aa3` | **also four checks**: CAGR +7.45%, volatility 9.83%, win 71.90%, drawdown -17.88% |
| 16 names at 6% weight, two entries a day | `v2_aa4` | CAGR +5.63%, volatility 10.33%: rejected |

### The constrained frontier is stable

| configuration | CAGR | vol | max DD | win | checks |
|---|---:|---:|---:|---:|---:|
| `v2_y4` | +7.44% | 9.88% | -14.86% | 70.66% | 4/7 |
| `v2_z1` | +6.94% | 9.93% | -15.33% | 71.74% | 4/7 |
| `v2_aa3` | +7.45% | 9.83% | -17.88% | 71.90% | 4/7 |

Four independent parameterisations converge on the same point: with the 10% volatility cap and
the 70% win gate both satisfied, the account compounds at about 7.4-7.5% per year. Without the
volatility cap the same machinery reaches +13.4% (and +14.5% at a slightly higher risk), so the
volatility constraint costs roughly half the return.

## Final position after sixteen rounds

| requirement | best achieved | configuration | status |
|---|---|---|---|
| CAGR >= 25% | **+14.53%** | `v2_m1b` | not reachable: needs information ratio 2.5, best measured 1.13 |
| volatility <= 10% | **9.83%** | `v2_aa3` | met |
| drawdown >= -10% | -14.86% | `v2_y4` | not reachable: the -10% floor implies volatility near 7.9% and CAGR near 6% |
| win rate >= 70% | **72.75%** | `v2_m1b` | met |
| every quarter positive | 10-12 negative of 30 | best runs | not reachable with this alpha |
| holding frequency | 97.6% / 90.5% | `v2_g4` | met |


## Round-17 findings: the database has more than OHLCV (user-directed re-open)

The goal was re-opened at the user's request after being marked blocked. The blocker was
based on sweeping every axis over ONE data source; the database itself had not been
inventoried. `scripts/audit_schema.py` and `scripts/audit_unused_columns.py` now do that.

### Columns in `qlib_daily_features` that were never used

| column | status |
|---|---|
| `vwap` | **fully populated** (81% of A-share rows in 2020-2026; 100% in the sampled months of 2016/2020/2024). The design said to drop VWAP features *if the field were missing*; it is not missing. |
| `adjclose` | a second cumulative price series, distinct from `close` and from `close/factor` (SH600519 on 2024-01-02: close 372.43, factor 0.221, adjclose 13239.73, raw = close/factor = 1685, which matches the real traded price) |
| `change` | equals `pct_change(close)` (238 of 242 sampled rows); redundant |
| `is_intraday_derived`, `quality_flags` | flags; some rows record `actual_minute_rows=240`, i.e. minute data exists behind part of the history |

### VWAP features carry real information (IC scan, 2020-2026)

| feature | IC h=5 | IC h=10 | IC h=20 | t (h=20) |
|---|---:|---:|---:|---:|
| vwap_mom20 (20-day VWAP momentum) | -0.0491 | -0.0578 | **-0.0697** | -19.2 |
| vwap_cv20 (20-day VWAP dispersion) | -0.0569 | -0.0621 | **-0.0679** | -16.7 |
| vwap_mom5 | -0.0307 | -0.0298 | -0.0334 | -10.2 |
| vwap_dev5 (close/VWAP-1, 5-day) | -0.0191 | -0.0256 | -0.0344 | -11.5 |
| vwap_pos (VWAP position in the daily range) | +0.0022 | +0.0030 | +0.0012 | +0.4 |

### But they do not help the account

| configuration | CAGR | vol | max DD | win | IR |
|---|---:|---:|---:|---:|---:|
| `v2_g4` baseline (no VWAP component) | **+13.39%** | 11.82% | -15.00% | **70.29%** | **1.13** |
| `v2_vw1` combo4v (adds vwap_cv20) | +5.11% | 10.88% | -14.19% | 63.69% | 0.47 |
| `v2_vw2` combo5v (adds vwap_cv20 + vwap_mom20) | +5.24% | 11.25% | -18.06% | 64.27% | 0.47 |
| `v2_vw3` combo4m (adds vwap_mom20) | +11.65% | 12.62% | -16.56% | 68.18% | 0.92 |

This is the fourth falsification of "a higher-IC feature will improve the account" (rounds 6,
9, 12 and now 17). `vwap_mom20` alone comes closest at IR 0.92 but still below the baseline.

### Reproducibility check

Adding `vwap` to the panel changes the cache key (`version: 2`) and rebuilt the panel. The baseline
re-run on the new cache is **bit-identical** to the previous headline: CAGR 0.133935, vol
0.118185, drawdown -0.149953, win 0.702929, 956 round trips, final NAV 263,277.40. So no
earlier result depends on the missing field.

### What remains unexplored

1. **Minute-level data behind the daily bars** (`quality_flags` records `actual_minute_rows=240`).
   If the source for those rows is reachable, an intraday fill model for the profit target and
   the stop would change both the win rate and the expectancy - the one execution assumption
   that has never been testable. This is the highest-value remaining avenue.
2. **The risk layer**: per-name volatility targeting, correlation-aware sizing (tested in
   round 18, see below), and a faster-reacting drawdown overlay built on realized account returns
   rather than the estimator.
3. **The existing factors**: the rolling model has never had its hyperparameters or objective
   tuned (rank objective, calibration, deeper trees), and only early stopping on a noisy
   validation rank IC was used.



## Round-18 findings: the risk layer, tested one axis at a time

Every run below is the `v2_g4` headline configuration with exactly one axis changed, on the same
prebuilt rolling-model predictions, engine, fees and execution model. The control `v2_g4b` re-ran
the headline and reproduced it bit for bit (CAGR 0.133935, vol 0.118185, DD -0.149953, win 0.702929,
956 round trips), so the differences below are attributable to the named axis.

### Correlation-aware position sizing (not filtering)

The correlation *filter* was rejected in round 7 because it discards the candidate. Round 18
implements the sizing version instead: a new name's weight cap is multiplied by

```
penalty = clip(1 - strength * max(0, avg_corr - base) / (1 - base), floor, 1)
```

where `avg_corr` is the candidate's average pairwise correlation with the names already held
(`portfolio.average_pairwise_corr`), computed from the same causal return window.

| run | strength | base | floor | CAGR | vol | max DD | win | IR | trades | fees %/yr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `v2_g4` (no sizing) | - | - | - | **13.39%** | 11.82% | -15.00% | 70.29% | 1.133 | 956 | 5.05 |
| `v2_cs1` | 0.5 | 0.30 | 0.50 | 13.29% | 11.64% | -16.59% | 70.51% | **1.141** | 929 | 4.83 |
| `v2_cs2` | 1.0 | 0.30 | 0.25 | 11.81% | 11.11% | -13.70% | 70.84% | 1.062 | 926 | 4.42 |
| `v2_cs3` | 0.5 | 0.00 | 0.50 | 11.59% | 10.72% | -14.03% | 70.95% | 1.081 | 950 | 4.29 |

Verdict: **not adopted**. The mild setting (`cs1`) moves IR by +0.008, which is inside the path
noise band; the strong settings (`cs2`, `cs3`) buy 0.7-1.1pp of volatility and 1.3-3.0pp of
drawdown with 1.6-1.8pp of return. Correlation between names is simply not where this book's risk
lives - the volatility target already scales the whole sleeve, so the marginal duplicate name is
trimmed by the gross and vol caps rather than by the correlation penalty.

### Portfolio breadth: the alpha is concentrated in the top ranks

If the ranking carried equal information down the list, holding more names should raise the
information ratio (more independent bets for the same signal). The 12-name cap had never been
tested against that prediction. `max-weight` is scaled as ~1/N so the gross capacity is unchanged.

| run | names | new/day | CAGR | vol | max DD | win | IR | trades |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `v2_g4` | 12 | 1 | **13.39%** | 11.82% | **-15.00%** | **70.29%** | **1.133** | 956 |
| `v2_bp16` | 16 | 1 | 11.89% | 11.79% | -15.89% | 67.06% | 1.008 | 1090 |
| `v2_bp20` | 20 | 1 | 8.76% | 10.92% | -16.24% | 67.07% | 0.802 | 1169 |
| `v2_bp24` | 24 | 1 | 9.38% | 10.59% | -16.82% | 70.39% | 0.886 | 1226 |
| `v2_bp16n2` | 16 | 2 | 12.34% | 12.71% | -15.58% | 67.72% | 0.970 | 1363 |
| `v2_bp20n2` | 20 | 2 | 10.20% | 12.62% | -15.87% | 67.01% | 0.808 | 1546 |
| `v2_bp20n3` | 20 | 3 | 7.28% | 11.98% | -18.85% | 62.43% | 0.607 | 1738 |

The prediction is falsified in the strongest possible form: **IR falls monotonically with breadth**,
the win rate falls with it, and adding an entry slot per day (which fills the extra names faster)
makes it worse again. Fees do fall as the book spreads (5.05%/yr -> 4.15%/yr at 24 names), which
confirms the accounting, but the return lost by buying the 13th-24th ranked names is three to four
times larger than the fee saved. The edge is a top-of-the-list phenomenon, which is also why the
result is path-sensitive: the identity of the few names that clear the bar drives the outcome.

### The sizing layer and the timing layer, measured

| run | axis changed | CAGR | vol | max DD | win | IR |
|---|---|---:|---:|---:|---:|---:|
| `v2_g4` | - | 13.39% | 11.82% | -15.00% | 70.29% | 1.133 |
| `v2_vt0` | volatility target OFF | 13.09% | 12.16% | -17.84% | 69.80% | 1.077 |
| `v2_vt20` | target 14% -> 20% | 13.63% | 11.96% | -16.28% | 69.80% | **1.139** |
| `v2_notm` | market timing OFF | **14.60%** | 15.30% | -27.29% | **73.26%** | 0.955 |

Two facts worth keeping:

* the volatility target earns its place, but only just: switching it off costs 0.06 of IR and 2.8pp
  of drawdown; loosening it to 20% gains 0.006 of IR, i.e. the account is not leverage-constrained
  by the 14% setting. The constraint is the market ladder, not the vol target.
* the market ladder is the single most valuable risk-layer component measured so far: removing it
  raises CAGR by 1.2pp and the win rate to the best value ever recorded (73.26%) but adds 3.5pp of
  volatility and takes the drawdown from -15.0% to -27.3%. It buys 0.18 of IR and 12pp of drawdown
  for 1.2pp of return, and it is the reason the compliance configuration can hold volatility under
  10% at all.


### What actually sets exposure: the market ladder, not the volatility target

`scripts/diag_exposure.py` reads `constraints_daily.csv` back out of a finished run.

| run | mean gross | median gross | mean names | flat days | mean ladder cap | state mix (weak / neutral / strong / extreme) |
|---|---:|---:|---:|---:|---:|---|
| `v2_g4` | 0.565 | 0.555 | 10.3 | 0.05% | **0.587** | 36.7% / 27.8% / 29.4% / 6.2% |
| `v2_notm` | 0.703 | 0.725 | 11.2 | 0.05% | (off) | - |

The account sits *at its ladder cap* essentially every day: mean gross 0.565 against a mean cap of
0.587, with the realized gross target met to within 0.02 on average. The 14% volatility target
therefore almost never binds - it is the ladder (mean cap 0.587) that decides how much of the
account is invested, which is why `v2_vt20` could raise the target to 20% for no measurable
effect. It also explains the frequency contract being met so comfortably: the book is fully
deployed about 14% of days and never flat.

### The ladder earns its place, and its levels are already near-optimal

If the ladder were only setting an average exposure level, replacing it with that same constant
exposure would change nothing. Round 18 tests exactly that with `--state-caps`, which rewrites the
per-state caps and leaves the state machine untouched.

| run | caps (weak / neutral / strong) | CAGR | vol | max DD | win | IR |
|---|---|---|---:|---:|---:|---:|---:|
| `v2_g4` | 0.35 / 0.70 / 0.90 (design) | **13.39%** | 11.82% | **-15.00%** | **70.29%** | **1.133** |
| `v2_lflat58` | 0.587 / 0.587 / 0.587 | 10.72% | 13.04% | -25.94% | 64.15% | 0.822 |
| `v2_lflat50` | 0.500 / 0.500 / 0.500 | 9.87% | 11.24% | -21.62% | 61.11% | 0.878 |
| `v2_lflat68` | 0.680 / 0.680 / 0.680 | 12.90% | 14.04% | -29.26% | 69.62% | 0.919 |
| `v2_lsoft` | 0.50 / 0.62 / 0.66 | 13.35% | 12.49% | -21.36% | 69.38% | 1.069 |
| `v2_lamp` | 0.25 / 0.72 / 0.90 | 11.27% | 10.84% | -13.16% | 66.96% | 1.039 |
| `v2_lamp2` | 0.15 / 0.80 / 0.95 | 7.37% | 9.93% | -14.09% | 62.78% | 0.742 |
| `v2_linv` | 0.85 / 0.60 / 0.35 (inverted) | 12.09% | 12.74% | -24.80% | 68.47% | 0.949 |

Constant exposure at the ladder's own average level (0.587) costs **2.7pp of CAGR, 1.2pp of
volatility and 10.9pp of drawdown** against the ladder - an IR difference of 0.31. The state
machine is therefore genuinely informative, not a level knob, and it is the single most valuable
component of the risk layer measured in this project. Inverting the ladder (0.85 in weak markets,
0.35 in strong) still beats constant exposure at 0.949, which says the *ordering* carries
information and the *magnitude* is roughly right: both a flatter and a more aggressive ladder are
worse, and the design's 0.35/0.70/0.90 levels sit at a local optimum of everything tried.

One consequence for the contract: the best volatility-compliant point in this table is
`v2_lamp2` at 9.93% volatility and **7.37%** CAGR, which matches the independent
`v2_y4` compliance point (9.88% volatility, 7.44% CAGR). Two different mechanisms - scaling
every cap down, and re-shaping the ladder - land on the same number, so at the contract's 10%
volatility ceiling this strategy earns about **7.4% a year**, not 13.4%.

### The objective is jagged: the headline is a point on a knife edge

The blend weight between the rolling model and the rule composite was never resolved finely; it
was set a priori at 0.50.

| rank-blend (weight on the model) | 0.30 | 0.40 | 0.45 | **0.50** | 0.55 | 0.60 | 0.65 | 0.70 | 0.80 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CAGR | 11.14% | 12.35% | 10.79% | **13.39%** | 10.16% | 9.92% | 10.16% | 9.77% | 9.23% |
| win rate | 70.87% | 70.83% | 68.36% | 70.29% | 68.43% | 68.81% | 69.21% | 68.85% | 69.37% |

A 0.05 change in the blend weight moves CAGR by 2.6pp and the win rate by 2pp. For contrast, the
volatility-estimation-window family is smooth (40/60/90/120 days: 12.82% / 13.39% / 13.02% /
11.54%, sd 0.70pp) and the liquidity-floor family is not (0.10 / 0.20 / 0.35: 10.21% / 13.39% /
4.55%).

| family (near-equivalent settings) | n | mean CAGR | sd | min | max |
|---|---:|---:|---:|---:|---:|
| blend 0.40-0.60 | 5 | 11.32% | **1.34pp** | 9.92% | 13.39% |
| blend 0.30-0.80 | 9 | 10.77% | 1.26pp | 9.23% | 13.39% |
| volatility window 40-120 | 4 | 12.69% | 0.70pp | 11.54% | 13.39% |

### The local noise floor, measured directly

`scripts/sweep_r18d_jitter.sh` moves one parameter at a time by 1-2 per cent of its own value:
far too little to change the strategy's economics, but enough to move an occasional fill or swap
two adjacent candidates. In three of the ten runs (buy premium 3.1%, risk per trade +/-0.0001) the
parameter never binds, and the run is bit-identical to the headline.

| perturbation | CAGR | vol | max DD | win |
|---|---:|---:|---:|---:|
| `--risk-per-trade` 0.0159 / 0.0161, `--buy-premium` 0.031 | 13.39% | 11.82% | -15.00% | 70.29% (identical to `v2_g4`) |
| `--buy-premium` 0.029 | 13.22% | 11.82% | -16.94% | 70.42% |
| `--rank-blend` 0.49 | 13.22% | 11.81% | -15.08% | 69.69% |
| `--min-amount-rank` 0.199 | 13.17% | 11.84% | -16.66% | 69.87% |
| `--min-amount-rank` 0.201 | 13.01% | 11.81% | -16.76% | 69.64% |
| `--vol-window` 61 | 12.97% | 11.80% | -16.31% | 69.41% |
| `--rank-blend` 0.51 | 12.40% | 12.02% | -18.60% | 70.00% |
| `--vol-window` 59 | 12.24% | 11.89% | -16.75% | 69.17% |

Across the seven runs that actually moved the path: mean **13.04%**, sd **0.41pp**, range
12.24-13.39%. The headline sits +0.86 sd above that mean, i.e. it is *not* an outlier for its own
configuration.

So there are two different scales, and it matters which one a claim is made on:

* **A 1-2% perturbation of any single parameter: sd 0.41pp.** Differences below about **1.2pp of
  CAGR** (3 sd) are not evidence. Several earlier verdicts in this project rested on 1-2pp gaps and
  should be treated as unresolved rather than falsified. The verdicts that rest on large gaps -
  breadth (2.3-5.4pp), the market ladder (2.7pp plus 11pp of drawdown), the ranker blend against
  its single components (8pp), the extreme-state liquidation - stand.
* **A 0.05 change of the blend weight: sd 1.34pp.** The blend axis is far steeper than every
  other axis measured, so the headline's expected value *along that axis* is the family mean,
  about **11.3%**.

Either way, the measured headline is 13.4% (13.0% +/- 0.4% for the configuration, ~11.3% averaged
over the blend weight) against a contract that needs 25% at two points less volatility - a factor
of 2 to 3 in CAGR and of about 2.2 in information ratio. The knife edge is structural rather than
a tuning artefact: with the alpha concentrated in the top one or two names and about 50 round
trips a year, the path dominates the parameter.


## Round-19 findings: the classic low-risk and lottery anomalies

The V2 feature library had never contained the textbook low-risk factors - market beta,
idiosyncratic volatility, skewness, downside semi-deviation, own-price drawdown, the maximum
daily return (the "MAX" lottery effect), or return autocorrelation. `scripts/research_r19_factors.py`
computes them one at a time and scans their rank IC against forward returns on the same base
universe and the same dates as every earlier scan.

### They are strong in the cross-section

| feature | IC h=5 | IC h=10 | IC h=20 | t (h=20) |
|---|---:|---:|---:|---:|
| **ivol60** (idiosyncratic vol vs the equal-weight universe) | -0.0733 | -0.0890 | **-0.1071** | **-34.9** |
| maxret20 (largest daily return in 20d, the MAX effect) | -0.0696 | -0.0802 | -0.0914 | -34.0 |
| gap_abs20 (mean absolute overnight gap) | -0.0648 | -0.0757 | -0.0893 | -23.4 |
| maxret1_pct (largest daily return in 60d) | -0.0594 | -0.0714 | -0.0853 | -31.0 |
| dist_ma120 | -0.0638 | -0.0738 | -0.0810 | -24.7 |
| mom20 (20-day reversal) | -0.0632 | -0.0719 | -0.0804 | -27.2 |
| mom60 | -0.0564 | -0.0665 | -0.0741 | -23.8 |
| dn_up_vol20 (**positive**: upside-smooth names do worse) | +0.0568 | +0.0635 | +0.0685 | +28.1 |
| dist_ma250 | -0.0479 | -0.0557 | -0.0614 | -18.6 |
| mom120 | -0.0426 | -0.0500 | -0.0544 | -17.8 |
| semidev20 (downside semi-deviation) | -0.0328 | -0.0402 | -0.0524 | -13.6 |
| pos_days20 (share of up days) | -0.0333 | -0.0361 | -0.0436 | -20.3 |
| pos_days60 | -0.0271 | -0.0329 | -0.0403 | -17.8 |
| skew60 | -0.0281 | -0.0322 | -0.0343 | -20.8 |
| vol_ratio_5_60 | -0.0290 | -0.0316 | -0.0315 | -17.1 |
| skew20 | -0.0272 | -0.0311 | -0.0302 | -18.7 |
| dd60 (own drawdown from the 60d high) | -0.0284 | -0.0294 | -0.0288 | -8.9 |
| dd252 | -0.0176 | -0.0194 | -0.0182 | -5.4 |
| beta60 (market beta) | -0.0031 | -0.0044 | -0.0101 | -2.4 |
| kurt60 | +0.0011 | +0.0031 | +0.0088 | +4.1 |
| ac1_20 (return autocorrelation) | +0.0025 | +0.0041 | +0.0024 | +1.5 |
| mom252_skip20 (12-month momentum, skipping a month) | +0.0005 | -0.0011 | -0.0037 | -1.2 |

`ivol60` is the **strongest single cross-sectional feature measured anywhere in this project**:
1.5x the best VWAP factor from round 17 (`vwap_mom20`, IC20 -0.0697, t -19.2) and of the same
order as the rolling model itself. Note also that `beta60` is nearly flat once idiosyncratic
volatility is separated out: in this universe the low-risk effect is *idio*syncratic volatility,
not market beta, which is the opposite of the US BAB literature.

### And they make the account much worse

Each run below is `v2_g4` with the composite's `vol20` component replaced by the new factor
(`scripts/sweep_r19_factors.sh`), so the comparison is exactly like for like.

| composite component | CAGR | vol | max DD | win | IR |
|---|---:|---:|---:|---:|---:|
| `vol20` (headline `v2_g4`) | **13.39%** | 11.82% | **-15.00%** | **70.29%** | **1.133** |
| `dd252` (own drawdown from the 252d high) | 12.78% | 15.64% | -20.21% | 72.33% | 0.817 |
| `pos_days20` | 11.32% | 13.67% | -21.47% | 69.19% | 0.828 |
| `skew60` | 9.83% | 13.98% | -20.81% | 69.68% | 0.703 |
| `maxret20` | 8.81% | 12.24% | -16.55% | 68.94% | 0.720 |
| `semidev20` | 8.02% | 11.63% | -20.71% | 67.39% | 0.690 |
| **`ivol60`** (the highest-IC feature in the project) | **6.83%** | 12.06% | -24.40% | 66.45% | **0.567** |
| `beta60` | 4.62% | 11.67% | -22.15% | 65.48% | 0.396 |
| `combo3 + beta60 + ivol60` (added, not replaced) | 4.70% | 9.88% | -14.71% | 64.26% | 0.476 |

This is the **fifth and most decisive falsification** of "a stronger cross-sectional signal
improves the account". The feature with the largest IC in the entire project cuts the CAGR by
half, and *adding* it to the winning composite does the same. Together with round 18's breadth
result it identifies the mechanism:

* IC measures the average rank relationship across the whole cross-section (up to 3000 names a
  day). The account never holds the cross-section - it holds the **one or two names at the very
  top of the blended list**, about 50 round trips a year.
* A factor can therefore have an enormous average IC and still put the wrong names at the
  extreme top. `ivol60` ranks low-volatility names highly on average, but its top of list is
  dominated by names that are idiosyncratically quiet *and* have just fallen hard, which is a
  different (and much worse) trade than the `vol20` top of list.
* This is also why breadth destroys IR (round 18): moving down the list adds names the blend
  did not select, and the list is only informative at its head.

The practical consequence is a screening rule for the rest of this project, the opposite of the
usual one: **an IC scan is a filter, never a selection criterion**. Only the account test decides.


## Round-20 findings: the head of the list is already the right order

Round 18 showed the account only ever buys from the very top of the blended list, and round 19
showed the composite's value lives at that head rather than in its average rank. Both point at the
same untested mechanism: a **second ranking stage applied to the top K candidates only**, choosing
the day's single buy on one transparent dimension. The design's move is implemented as
`--head-rerank K --head-key <feature> [--head-sign +/-1]` (`portfolio.select_new_candidates`),
which re-sorts the top K of the blended ordering and leaves everything below untouched.

| re-rank | CAGR | vol | max DD | win | IR |
|---|---:|---:|---:|---:|---:|
| none (headline `v2_g4`) | **13.39%** | 11.82% | **-15.00%** | 70.29% | **1.133** |
| top 5 by lowest `vol60` | 12.33% | 12.49% | -16.79% | **71.12%** | 0.987 |
| top 3 by lowest `amount20_yuan` | 12.23% | 11.83% | -17.46% | 68.80% | 1.034 |
| top 5 by lowest `amount20_yuan` | 10.79% | 11.93% | -18.20% | 67.40% | 0.905 |
| top 3 by lowest `vol60` | 10.59% | 12.28% | -19.59% | 68.94% | 0.862 |
| top 5 by lowest `ma60` (deepest below the average) | 10.27% | 11.68% | -17.08% | 68.81% | 0.879 |
| top 3 by lowest `ma60` | 9.44% | 11.99% | -16.98% | 66.45% | 0.787 |
| top 3 by lowest `ret1` (most oversold yesterday) | 9.07% | 12.03% | -17.16% | 66.78% | 0.754 |
| top 3 by lowest `atr20` | 8.82% | 12.09% | -16.18% | 67.43% | 0.729 |

Rejected, and informatively so. Every single-factor re-rank at the head loses 1.1 to 4.6pp of CAGR,
and the *most* on-theme re-rank - "among the top three, take the least volatile" - is one of the
worst. Together with round 19 this says something specific: **the 50/50 blend of the rolling model
and the causal rule composite produces an ordering at the head that no one of its components can
reproduce.** The blend is not a smoother version of either input; it is the thing that works, and
it works precisely where the account looks.

### Where this leaves the search

| layer | status after twenty rounds |
|---|---|
| entry signal | the 50/50 model/`combo3` blend is the largest effect in the project; every added feature (rounds 6, 9, 12, 16, 17, 19), every head re-rank (round 20) and every breadth change (round 18) is worse |
| exit policy | stops, trailing stops, stale-loser exits, profit locks and hold extensions all rejected; the frozen 3.5%/45-day policy stands |
| portfolio construction | 12 names at an 8% cap; more names, fewer names, correlation filters and correlation sizing all rejected |
| risk layer | the market ladder is the one component whose removal is catastrophic (+2.7pp CAGR but -10.9pp drawdown and +1.2pp volatility when replaced by constant exposure) |


## Round-21 findings: the model is better undertrained, and this is the cleanest falsification yet

`v2m_tp5h40_predictions.quarters.csv` records, for every one of the 31 target quarters, the
boosting round at which the rolling return model early-stopped. The median is **7 out of 300**,
and in **9 of 31 quarters it is 1** - at a learning rate of 0.03 that is a single split. The
"model" that the account blends against has therefore almost never been trained to convergence.

That is a defect worth fixing, so round 21 makes the training length and the objective explicit
(`--objective {mse,lambdarank}` for the return model, `--fixed-rounds N` to skip early stopping
and train exactly N rounds, plus `--num-leaves`/`--learning-rate`/`--min-data-in-leaf`), then
measures both the model and the account.

### Fixing the model makes the model better

Decile spread is the realised net return of the model's top decile minus its bottom decile,
averaged over the 31 target quarters.

| variant | training | decile spread | quarters with the right sign | top-decile win | bottom-decile win |
|---|---|---:|---:|---:|---:|
| baseline (early stop, median 7 rounds) | 1-84 rounds | 0.0132 | 26/31 | 0.5751 | 0.5636 |
| `mse` 150 rounds | fixed | 0.0177 | 26/31 | 0.5792 | 0.5498 |
| `mse` 400 rounds | fixed | **0.0188** | **27/31** | **0.5828** | **0.5450** |
| `lambdarank` 150 rounds | fixed | 0.0026 | 18/31 | 0.5933 | 0.5299 |

Training the return model properly raises the decile spread by **42%** and improves the top and
bottom deciles in the right direction. The ranking objective (`lambdarank` on cross-sectional
relevance buckets) is the one clear failure: its average ordering is close to useless even though
its top decile has the highest win rate of the four.

### And makes the account worse, at every fixed length

Every run below is the `v2_g4` configuration with only the prediction file changed. The second
block trains the same model for an explicitly short, fixed number of rounds, so the training
length is a controlled variable in both directions.

| prediction file | training | CAGR | vol | max DD | win | IR | decile spread |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline (early stop, adaptive) | 1-84 rounds, median 7 | **13.39%** | 11.82% | **-15.00%** | **70.29%** | **1.133** | 0.0132 |
| `mse` fixed 50 | 50 | 11.76% | 11.74% | -16.05% | 68.77% | 1.002 | 0.0160 |
| `mse` fixed 25 | 25 | 10.52% | 11.74% | -20.83% | 69.77% | 0.896 | 0.0151 |
| `mse` fixed 150 | 150 | 9.46% | 12.06% | -19.48% | 67.92% | 0.784 | 0.0177 |
| `mse` fixed 10 | 10 | 9.30% | 11.67% | -21.08% | 67.42% | 0.796 | 0.0142 |
| `mse` fixed 400 | 400 | 8.01% | 11.76% | -21.01% | 68.99% | 0.680 | 0.0188 |
| `mse` 150, blend 0.40 | 150 | 10.47% | 11.82% | -19.30% | 68.24% | 0.886 | 0.0177 |
| `mse` 150, blend 0.60 | 150 | 7.66% | 11.95% | -18.48% | 65.78% | 0.641 | 0.0177 |
| `lambdarank` 150 | 150 | 10.13% | 12.85% | -23.64% | 68.77% | 0.788 | 0.0026 |

Two things follow, and the first corrects the hypothesis that motivated the round.

1. **The hypothesis was wrong: the early stopping is not a defect.** No fixed training length beats
   it, in either direction - 10, 25, 50, 150 and 400 rounds all lose 1.6 to 5.4pp of CAGR against
   the adaptive per-quarter schedule, and every one of them also loses information ratio. The
   validation-driven stop is choosing a quarter-specific model, and that per-quarter adaptation is
   itself worth more than the extra fitting. "The model is undertrained" was the wrong diagnosis;
   the model is deliberately small, and that is load-bearing.
2. **The decile spread still anti-predicts the account.** Across the six fixed-length and adaptive
   models the spread rises 0.0132 -> 0.0142 -> 0.0151 -> 0.0160 -> 0.0177 -> 0.0188 while the CAGR
   does not follow it upward at all (13.39% -> 9.30% -> 10.52% -> 11.76% -> 9.46% -> 8.01%); the two
   best spreads (0.0177, 0.0188) produce two of the three worst accounts. The rank correlation
   between decile spread and CAGR over these six models is negative.

This is the sixth independent falsification of "a better ranking improves the account", and the
first produced by intervening on the model itself rather than on its inputs. It sharpens the
mechanism: the value of the blend is *not* the model's average ordering power. A model trained to
its validation optimum imposes its own (training-period) head on the blended list and crowds out
the causal rule composite, and the account - which only ever buys the top one or two names -
inherits that overfitting instead of the composite's stability.


### The seed is an irrelevant knob, and it moves CAGR by 4.8pp

The rolling model's random seed cannot change the strategy's economics. It changes only which of
several near-equivalent splits the trees pick, and therefore occasionally the order of two
candidates at the head of the blended list. Three otherwise identical builds (seeds 42, 7 and 2024)
produce three quite different accounts:

| run | model seed | CAGR | vol | max DD | win | IR |
|---|---:|---:|---:|---:|---:|
| 42 (headline `v2_g4`) | **13.39%** | 11.82% | **-15.00%** | **70.29%** | **1.133** |
| 11 | 12.06% | 11.98% | -18.22% | 70.01% | 1.007 |
| 5 | 11.21% | 11.95% | -15.84% | 70.41% | 0.939 |
| 2024 | 10.77% | 12.23% | -18.41% | 70.07% | 0.880 |
| 101 | 10.76% | 11.81% | -20.68% | 70.09% | 0.911 |
| 7 | 8.55% | 11.41% | -18.19% | 68.06% | 0.749 |
| 23 | 8.06% | 11.94% | -20.31% | 66.45% | 0.675 |
| 31337 | 7.61% | 11.70% | -17.15% | 67.12% | 0.650 |
| 777 | 6.94% | 11.74% | -17.34% | 67.74% | 0.591 |
| **distribution** | **mean 9.93%, sd 2.22pp** | mean 11.84%, **sd 0.22pp** | mean -17.90% | mean 68.92%, sd 1.56pp | mean 0.837, sd 0.182 |

The headline is the maximum of nine draws, **+1.6 standard deviations** above the seed mean. Two
structural facts come out of the table:

* **volatility is stable and return is not.** Realised volatility varies by only 0.22pp across
  seeds while the CAGR varies by 2.22pp, so essentially all of the information-ratio dispersion
  (0.837 +/- 0.182) is return dispersion. The risk control is reproducible; the return is a draw.
* **the 70% win gate is a coin flip.** Mean 68.9% with a 1.56pp standard deviation; 5 of 9 seeds
  clear 70%.

Repeating the same nine-model panel through the `v2_y4` compliance configuration (cap-scale 0.80,
3.0% target, 120-day hold) gives the same shape one notch lower:

| `v2_y4` configuration, 7 seeds | CAGR | vol | max DD | win | IR |
|---|---:|---:|---:|---:|---:|
| distribution | **mean 7.40%, sd 2.18pp** | mean 10.19%, sd 0.28pp | mean -15.60%, sd 0.80pp | mean **71.25%**, sd 1.43pp | mean 0.724 |

Only **2 of 7 seeds** actually land under the contract's 10% volatility ceiling: the published
`v2_y4` figure of 9.88% is itself a favourable draw from a distribution centred at 10.19%.
Meeting the volatility check *reliably* needs a lower exposure still, which costs roughly
another point of CAGR.

The three-seed rank ensemble (`scripts/merge_predictions.py`, averaging the within-date percentile
ranks of three models) comes in at 10.12% - the family mean, as an ensemble should - and does not
improve on it.

This is the same lesson as the blend-weight family (sd 1.34pp) in a purer form: **every knob that
can reshuffle the head of the ranking carries a standard deviation of roughly 2.2pp of CAGR**,
while knobs that cannot (a 1-2% change in a parameter that never binds, a volatility-estimation
window) carry about 0.4pp. Any single-run comparison in this project smaller than about 4.4pp
(2 sd) is therefore a draw, not a result.

Three consequences, and they are uncomfortable but they are what the measurements say:

* the strategy's **expected** CAGR for the `v2_g4` configuration is **9.9%** (median 10.8%), not
  13.39%. The headline is a favourable draw from its own distribution, and so is every other
  'best run' quoted in this document;
* the contract's **win-rate check is a coin flip** rather than a pass: 68.9% +/- 1.6pp;
* the earlier rounds' rejections should be re-read by size. Breadth (2.3-5.4pp), the market ladder
  (2.7pp of CAGR plus 11pp of drawdown), the ranker ablation (8pp), the fixed-round models
  (1.6-5.4pp, consistent in sign across six runs) and the round-19 factor replacements
  (2.5-6.6pp) survive a 2.2pp noise floor; the correlation-sizing, blend-weight and head-re-rank
  verdicts do not.


## Round-22 findings: the data question, closed with evidence

Rounds 17 and 21 left two data avenues open: intraday bars behind the daily features, and any
non-price information. Both are now resolved from the artefacts themselves
(`scripts/audit_minute_source.py`, `scripts/audit_provenance.py`).

### The database contains no intraday table

The DuckDB holds exactly seven base tables - `qlib_daily_features` (15.76M rows),
`qlib_calendar`, `qlib_instruments`, `qlib_batches`, `qlib_contract_flags`,
`qlib_feature_import_audit`, `qlib_source_snapshots` - and no column whose name contains
`min`, `tick`, `intraday`, `bar`, `second` or `time`. The intraday provenance survives only as
text inside `quality_flags`, e.g. `intraday_from_sqlite_prefix;intraday_contract_v2;factor_carry_t_minus_1;prefix_cutoff_slot=900;actual_minute_rows=240`
(the largest single variant, 26k rows) and
`post_close_from_sqlite_full_day;sqlite_post_close_contract_v1;actual_minute_rows=240;factor_from_existing_target_intraday`
(83k rows).

### The minute source it names is gone

`qlib_batches` records the source of the intraday rows:
`sqlite_intraday_publish:/private/tmp/marketstore-live-1m-20260527/out/fetch_1m.sqlite`, published
on 2026-05-27. That path is a temporary directory; the surviving `marketstore-live-1m-*` trees in
`/private/tmp` cover 2026-09-07 to 2026-09-15 only. The minute data that exists on this machine is
**ETF** 1-minute bars (`etf_store/silver/etf_bars_1m`, 3.2G) and live captures from the last
fortnight - neither covers A-share stocks over the 2019-2026 backtest.

### There is non-price data, but it starts in 2025

The `marketstore/btdr_r1` store holds two genuinely different datasets, both causally tagged
(`available_at` = next trade date 09:00, "contractual T-1 delay"):

| dataset | content | coverage | usable for this backtest? |
|---|---|---|---|
| `amazingdata_margin_btdr_r22_v2/margin_detail.csv` | per-stock margin balance, purchases and repayments (融资融券) | 2024-12-02 .. 2025-12-31, 770 symbols, 198k rows | **no**: 13 months, and it starts after five of the seven backtest years |
| `amazingdata_industry_weight_btdr_r1_v2/industry_membership_normalized.csv` | daily industry membership (L1/L2/L3) and index weights | 375 as-of dates, 2024-12-31 .. 2026-07-21 | **no**: 2025 onwards |

So the two avenues that would most plausibly have moved the result - an intraday fill model for the
profit target and the stop, and a flow- or industry-based factor - are both **closed by the data
that is actually available**, not by a modelling choice. Every remaining lever has to work with the
daily OHLCV-plus-VWAP fields, which is what rounds 1-21 have swept.


## Round-22 (continued): the market ladder, measured against nine seeds

Round 21 established that a single run is a draw with a 2.2pp standard deviation. That makes every
configuration comparison in this project suspect - so the headline alternatives were re-run on all
nine model seeds and compared **paired**, seed by seed. The paired test removes the seed noise
entirely, because both configurations see the same nine models.

| configuration | seeds | mean CAGR | sd | vol | max DD | mean win | win >= 70% | IR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `v2_g4` (design ladder 0.35/0.70/0.90) | 9 | 9.93% | 2.47pp | 11.84% | -17.90% | 68.92% | 4/9 | 0.844 |
| **`v2_notm` (ladder off)** | 9 | **12.12%** | 2.43pp | 15.00% | -26.57% | **72.28%** | **8/9** | 0.806 |
| `v2_m1b` (headline without the drawdown tiers) | 9 | 9.79% | 2.26pp | 12.02% | -18.55% | 68.84% | 3/9 | 0.815 |
| `v2_cs1` (correlation-aware sizing) | 9 | 9.79% | 2.34pp | 11.73% | -17.86% | 69.38% | 5/9 | 0.834 |
| `v2_y4` (compliance: cap-scale 0.80) | 10 | 7.26% | 1.82pp | 10.13% | -15.59% | 71.45% | 8/10 | 0.715 |

### The paired result

Both configurations were run on the same nine models, so the difference can be tested directly:

| paired difference (ladder off - design ladder) | mean | sd | t | seeds in favour |
|---|---:|---:|---:|---:|
| CAGR | **+2.19pp** | 1.58pp | **4.16** | 8 of 9 |
| win rate | **+3.36pp** | 1.05pp | **9.63** | **9 of 9** |

This is the strongest result in the project: the market ladder costs **2.2pp of CAGR and 3.4pp of
win rate**, and buys **3.2pp of volatility and 8.7pp of drawdown**. Unlike almost everything else
measured here, the effect is far outside the seed-noise band and its sign is consistent across
every seed for the win rate.

The win-rate half matters most for the contract, because it is the one check that was previously a
coin flip: with the ladder off, **8 of 9 seeds** clear the 70% threshold (mean 72.28%), against
4 of 9 with it on.

### Why the ladder is not simply a leverage knob - and how it can be matched

The obvious objection is that the ladder-off configuration is just more leveraged. Two runs test
that, and the answer is instructive:

| run | exposure rule | mean CAGR | vol | max DD | win | IR |
|---|---|---:|---:|---:|---:|---:|
| `v2_sb_a_*` | ladder off, `--gross 0.75` | 12.30% | 14.54% | -26.63% | 69.77% | 0.845 |
| `v2_sb_b_*` | mild ladder 0.70/0.85/0.95 | 11.35% | 14.57% | -26.90% | 71.32% | 0.773 |
| `v2_sb_c_*` | semi-mild ladder 0.50/0.75/0.92 | 11.12% | 13.39% | -21.95% | 70.29% | 0.830 |

Lowering the gross cap from 0.95 to 0.75 barely moved anything (15.00% -> 14.54% volatility),
because **with the ladder off the book is entry limited, not cap limited** - mean gross settles at
0.70 against a 0.95 cap. The exposure is set by the portfolio volatility target instead, which is
why `--vol-target`, not `--gross`, is the right knob for a matched comparison. Raising the weak
cap to 0.70 recovers most of the return (11.35%) and almost none of the risk reduction, because
the weak state is 36.7% of all days.


### The frontier the paired test produces

Every configuration below is a nine-seed mean, so the numbers are comparable and the seed noise is
averaged out. The drawdown overlay is the account's own NAV tiers; the market ladder is the V2
state machine.

| configuration | CAGR | vol | max DD | win | IR | seeds with win >= 70% |
|---|---:|---:|---:|---:|---:|---:|
| **H**: ladder off, gross cap 0.62 | 11.79% | 13.28% | -25.61% | 66.07% | **0.888** | 0/9 |
| **G**: ladder off, tighter drawdown tiers | 11.69% | 13.82% | -23.81% | 70.55% | 0.838 | 4/9 |
| **g4**: design ladder 0.35/0.70/0.90 | 9.93% | 11.84% | -17.90% | 68.92% | 0.844 | 4/9 |
| **notm**: ladder off | 12.12% | 15.00% | -26.57% | **72.28%** | 0.806 | **8/9** |
| **F**: ladder off, wider drawdown tiers | **12.39%** | 15.75% | -28.29% | **72.57%** | 0.784 | **8/9** |
| **y4**: design ladder, cap-scale 0.80 | 7.26% | **10.13%** | **-15.59%** | 71.45% | 0.715 | 8/10 |

Three candidates now dominate the old headline on the two objectives the contract names first:

* **win rate**: `notm`/`F` at 72.3-72.6% against `g4`'s 68.9%, and it is the *only* family that
  clears 70% in 8 of 9 seeds rather than 4 of 9;
* **CAGR**: `F` at 12.39% and `notm` at 12.12% against `g4`'s 9.93%.

They pay for it in the two risk checks: 15-15.8% volatility against 11.8%, and a -26.6% to -28.3%
drawdown against -17.9%. Tightening the account's own drawdown overlay (`G`) recovers part of
the risk (13.8% volatility, -23.8% drawdown) at 11.7% CAGR, and a gross cap that actually binds
(`H`) reaches the best information ratio measured in the project (0.888) but collapses the win
rate to 66.1%.

Year by year, the ladder-off configuration is positive in **every** calendar year of the sample
and its yearly win rate never falls below 67%:

| year | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 YTD | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `notm` return | +8.41% | +2.75% | +22.59% | +15.12% | +16.24% | +5.19% | +21.25% | +22.56% | **+185.69%** |
| `notm` intra-year drawdown | -13.00% | -10.53% | -7.56% | -17.94% | -11.36% | -25.45% | -7.25% | -11.29% | -27.29% |
| `notm` win rate | 67.2% | 74.6% | 72.2% | 75.9% | 73.8% | 74.8% | 72.1% | 75.0% | 73.26% |

### What the ladder was doing, and why it was wrong for this strategy

The V2 market ladder came from the V0/V1 trend design, where cutting exposure in weak markets is
defensive. For a low-volatility, below-the-moving-average mean-reversion book the weak state
(36.7% of all days) is exactly when the strategy's own edge is best, and the ladder under-invests
there. The measurement is unambiguous: removing it costs 3.2pp of volatility and 8.7pp of
drawdown, and buys 2.2pp of CAGR and 3.4pp of win rate, with the win-rate sign identical in all
nine seeds (t = 9.6).

That also explains why the ladder's own levels looked locally optimal in round 18: the alternative
cap vectors tested there (flatter, amplified, inverted) all *kept* the weak-market de-risking in
some form. The only way to find the effect was to remove the ladder entirely and re-test with a
paired design.


## Round-23 findings: the ladder-off gain is exposure, not alpha - and the hold cap is the real lever

Round 22 left two open questions: does the ladder-off configuration survive being scaled down to
the contract's 10% volatility ceiling, and is its advantage alpha or leverage? Both are now
answered, and the second answer produced the largest win-rate improvement in the project.

### Calibrating the exposure

With the market ladder off the book is entry limited - mean gross 0.703 against a 0.95 cap - so the
gross cap is nearly inert and the portfolio volatility target is the only knob that moves exposure.
Tracing it on one model (`scripts/sweep_r23_calibrate.sh`):

| volatility target | CAGR | vol | max DD | win | IR |
|---:|---:|---:|---:|---:|---:|
| 0.14 (the round-22 headline) | 14.60% | 15.30% | -27.29% | 73.26% | 0.955 |
| 0.12 | 13.18% | 14.75% | -27.70% | 72.81% | 0.894 |
| 0.085 | 11.33% | 12.94% | -23.08% | 71.57% | 0.876 |
| 0.070 | 9.31% | 11.52% | -18.85% | 70.37% | 0.808 |
| 0.055 | 7.36% | 9.97% | -17.08% | 69.85% | 0.739 |

At 0.055 the ladder-off configuration lands at 9.97% volatility and 7.36% CAGR, against the design
ladder's compliance point `v2_y4` at 9.88% volatility and 7.44% CAGR. **At matched risk the two
exposure mechanisms are the same strategy.** The round-22 gain of +2.19pp of CAGR was therefore the
extra exposure (mean gross 0.70 against 0.57), not a better risk-adjusted return - an important
correction to how round 22 should be read.

### But the hold cap is a real lever, and it is free

Re-tuning the exit policy at a fixed risk budget (`scripts/sweep_r23_exit.sh`, one model) found
that extending the hold cap from 45 to 90 days at the compliance exposure raised CAGR from 7.36%
to 8.71% *and* the win rate from 69.85% to 76.25%. Run on all nine seeds, with the ladder off at
the same volatility target, the paired result is the strongest measurement in this project:

| paired difference (120-day hold - 45-day hold, same exposure, nine models) | mean | sd | t | seeds in favour |
|---|---:|---:|---:|---:|
| win rate | **+8.26pp** | 0.89pp | **27.76** | **9 of 9** |
| CAGR | -0.90pp | 2.41pp | -1.12 | 4 of 9 (not significant) |

**A longer hold buys 8.3 points of win rate at no statistically detectable cost in return.** The
mechanism is the frozen 45-day cap: a low-volatility name that has not yet reached the 3.5% profit
target is force-closed at 45 days whether or not the thesis has played out, and those forced exits
are the losing trades. Holding to 120 days (average realised hold rises from 20.7 to ~28 days) lets
the target be reached instead.

### The frontier after round 23

All figures are nine-seed means, so they are directly comparable.

| configuration | CAGR | vol | max DD | win | win >= 70% | vol <= 10% |
|---|---:|---:|---:|---:|---:|---:|
| `r120` ladder off, vt 0.14, hold 120 | **11.22%** | 14.51% | -23.73% | **80.54%** | 9/9 | 0/9 |
| `m120` ladder off, vt 0.09, hold 120 | 9.69% | 12.79% | -21.80% | **80.06%** | 9/9 | 0/9 |
| vt 0.065, hold 120 | 7.69% | 10.76% | -18.44% | 78.28% | 9/9 | 0/9 |
| **`h120` ladder off, vt 0.055, hold 120** | 6.62% | **9.65%** | **-15.85%** | **78.22%** | **9/9** | **9/9** |
| `y4` design ladder, cap-scale 0.80 (previous compliance point) | 7.24% | 10.16% | -15.68% | 71.54% | 7/9 | 3/9 |

`h120` against the previous compliance point, paired on the same nine models:

| paired difference (`h120` - `y4`) | mean | sd | t | seeds in favour |
|---|---:|---:|---:|---:|
| win rate | **+6.68pp** | 1.09pp | **18.35** | **9 of 9** |
| CAGR | -0.62pp | 1.63pp | -1.14 | 4 of 9 (not significant) |
| volatility | -0.52pp | - | - | `h120` lower |
| maximum drawdown | -0.18pp | - | - | effectively identical |

`h120` is therefore a **strict contract-compliance improvement** over `v2_y4`: the same return
(the difference is not significant), the same drawdown, 0.5pp less volatility, and 6.7pp more win
rate - and, crucially, it satisfies the two shape checks *reliably* rather than on a lucky seed:

| check | `v2_y4` across 9 seeds | `h120` across 9 seeds |
|---|---|---|
| volatility <= 10% | 3 of 9 | **9 of 9** |
| win rate >= 70% | 7 of 9 | **9 of 9** |

Year by year (`h120`, model seed 42) the account is positive in every calendar year with a win
rate that never falls below 70%:

| year | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 YTD | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| return | +6.60% | +2.98% | +11.03% | +1.49% | +7.95% | +6.27% | +15.37% | +7.61% | +76.18% |
| intra-year drawdown | -8.13% | -11.29% | -5.32% | -14.20% | -7.49% | -16.92% | -4.42% | -6.75% | -17.79% |
| win rate | 80.5% | 73.9% | 77.5% | 82.7% | 76.6% | 75.8% | 84.6% | 70.3% | **77.47%** |

The win-rate-maximising configuration (`r120`) reaches **81.35%** over the full sample with yearly
win rates between 74.6% and 90.1%, at 14.51% volatility and a -24.1% drawdown.

### What this changes about the two named objectives

* **win rate >= 70% is now met with margin and with certainty.** 77.5% to 80.5% depending on the
  risk budget, clearing 70% in 9 of 9 seeds, against 68.9-72.3% before this round. The lever was
  the frozen hold cap, not the entry signal, the model or the risk layer.
* **CAGR is unchanged in character**: 6.6% at the 10% volatility ceiling, 9.7% at 12.8%
  volatility, 11.2% at 14.5%. Reaching 25% still needs an information ratio of 2.5; the values
  measured here are 0.69-0.81.


## Round-24 findings: the 3.5% profit target was the last big mistake

Round 23 showed that the frozen exit policy, not the signal or the risk layer, was where the
remaining win rate lived. Round 24 searched the exit policy properly: a two-dimensional grid over
the profit target and the hold cap, with the risk budget held at the contract's ceiling
(`--use-market-timing 0 --vol-target 0.055`, ladder off), first on one model
(`scripts/sweep_r24_grid.sh`, `sweep_r24b_grid.sh`) and then validated paired on all nine.

### The grid (one model)

CAGR, and in brackets the win rate:

| profit target | hold 60 | hold 90 | hold 120 | hold 200 |
|---|---:|---:|---:|---:|
| 3.0% | - | 8.46% (75.1%) | 7.26% (78.4%) | 6.93% (81.7%) |
| 4.5% | - | 7.12% (74.5%) | 7.38% (76.8%) | 7.14% (79.9%) |
| 6.0% | - | 7.51% (71.9%) | 8.67% (75.0%) | 6.05% (79.2%) |
| 7.5% | 11.54% (68.6%) | 8.29% (71.0%) | **10.76% (73.5%)** | - |
| **9.0%** | - | **11.85% (70.4%)** | 9.91% (71.8%) | 8.51% (77.3%) |
| 12.0% | 10.49% (61.8%) | 9.85% (63.8%) | 10.65% (69.5%) | - |
| 15.0% | 10.03% (59.1%) | 9.69% (64.2%) | 10.54% (67.1%) | - |
| no target | - | 3.57% (52.9%) | 3.73% (56.0%) | 5.82% (62.9%) |

Two things stand out. The old 3.5% target sits at the *bottom* of the range: raising it to 6-9%
adds 2-4pp of CAGR. And removing the target altogether collapses the strategy (IR 0.35), because
the only exits left are the time cap and the drawdown overlay, and those sell at whatever price
happens to be there.

### The validated setting (nine seeds, paired)

| configuration | CAGR | vol | max DD | win | IR | vol <= 10% | win >= 70% | checks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `c1` profit target 9.0%, hold 90 | **9.59%** +/- 1.21 | 9.95% | -16.78% | 68.27% +/- 1.93 | **0.964** | 5/9 | 3/9 | 2-4 |
| **`c2` profit target 7.5%, hold 120** | **9.16%** +/- 1.15 | **9.67%** +/- 0.12 | **-15.92%** | **74.76%** +/- 1.30 | **0.948** | **9/9** | **9/9** | **4 on 9/9** |
| `c3` profit target 6.0%, hold 120 | 8.00% +/- 1.48 | 9.68% | -15.88% | 76.36% +/- 1.72 | 0.826 | 9/9 | 9/9 | 4 on 9/9 |
| `h120` (round 23: target 3.5%, hold 120) | 6.62% +/- 1.52 | 9.65% | -15.85% | 78.22% +/- 1.70 | 0.685 | 9/9 | 9/9 | 4 on 9/9 |

Paired differences on the same nine models:

| paired difference | mean | sd | t | seeds in favour |
|---|---:|---:|---:|---:|
| `c2` - `h120` (target 3.5 -> 7.5%) CAGR | **+2.54pp** | 1.35pp | **5.66** | 9 of 9 |
| `c2` - `h120` win rate | -3.45pp | 1.16pp | -8.95 | 0 of 9 |
| `c1` - `h120` CAGR | **+2.96pp** | 1.18pp | **7.51** | 9 of 9 |
| `c1` - `h120` win rate | -9.95pp | 1.46pp | -20.49 | 0 of 9 |

### `c2` dominates the old headline on every axis that matters

`c2` versus `v2_g4`, the headline this project carried from round 8 to round 21, paired on the same
nine models:

| paired difference (`c2` - `v2_g4`) | mean | t | seeds in favour |
|---|---:|---:|---:|
| CAGR | -0.76pp | -0.96 | 4 of 9 (not significant) |
| **win rate** | **+5.85pp** | **10.30** | **9 of 9** |
| **volatility** | **-2.17pp** | - | `c2` lower on 9 of 9 |
| **maximum drawdown** | **+1.98pp** | - | `c2` better on 9 of 9 |

**The same return, 2.2 points less volatility, 2 points less drawdown and 5.9 points more win
rate** - and, unlike the old headline, it passes the volatility and win-rate checks on every seed:

| check | `v2_g4` (previous headline) | `c2` |
|---|---|---|
| volatility <= 10% | 0 of 9 seeds | **9 of 9** |
| win rate >= 70% | 4 of 9 seeds | **9 of 9** |
| checks passed | 3 of 7 | **4 of 7, on every seed** |

`c2` also beats the previous compliance point `v2_y4` paired: CAGR **+1.93pp (t = 2.65, 7 of 9)**,
win rate **+3.22pp (t = 8.43, 9 of 9)**, volatility -0.49pp, drawdown unchanged.

Year by year (`c2`, model seed 42): +11.49% / +0.06% / +21.20% / +3.30% / +17.02% / +7.22% /
+13.56% / +10.40%, total **+119.68%**, maximum drawdown **-13.45%**, win rate **73.53%** over 442
round trips - positive in all eight calendar years, with a yearly win rate of 70% or better in seven
of them.

### What is left

The CAGR requirement is still not met and the gap has not narrowed much: 9.2% at the 10%
volatility ceiling against the contract's 25%. Round 24 raised the information ratio of the
compliant configuration from 0.685 to **0.948**, which is the highest seed-mean IR measured in this
project and the first time the compliant configuration has matched the return end of the frontier -
but 25% at 10% volatility still requires IR 2.5.


## Round-25 findings: re-opening the axes the new headline inherited

Rounds 23-24 changed two constants that everything else had been tuned around, so the parameters
that were settled under the old policy had to be re-tested. Round 25 re-opens three of them: loss
cutting, the model/rule blend, and the account drawdown overlay. All three come back the same way -
the new headline is locally optimal - which is exactly what a robustness round should establish.

### Loss cutting is still rejected, under the new policy

Stops, trailing stops, stale-loser exits and profit locks were rejected in rounds 2-4, but that was
with a 3.5% profit target and a 45-day hold. A stop and a profit target are the same kind of object,
so the verdict needed re-testing. One model, `c2` as the base:

| variant | CAGR | vol | max DD | IR | win |
|---|---:|---:|---:|---:|---:|
| `c2` (no stop, 7.5% target, 120-day cap) | **10.76%** | 9.77% | **-13.45%** | **1.101** | **73.53%** |
| ATR stop 8-15% | 8.99% | 9.81% | -15.68% | 0.916 | 62.45% |
| ATR stop 8-15% + trailing stop | 9.09% | 9.86% | -15.71% | 0.922 | 62.54% |
| ATR stop 12-20% | 8.39% | 9.85% | -17.72% | 0.852 | 67.63% |
| ATR stop 5-10% | 7.63% | 9.73% | -17.66% | 0.784 | 53.73% |
| stale-loser exit after 30 days below entry | 9.56% | 10.40% | -19.11% | 0.919 | 55.05% |
| stale-loser exit after 45 days | 9.02% | 10.03% | -16.50% | 0.899 | 61.01% |
| profit lock (give back at most 50% of the high) | 3.29% | 10.64% | -22.45% | 0.309 | 56.76% |

Every mechanism lowers the CAGR (by 1.2 to 7.5pp), lowers the information ratio, and - the part
that is worth stating plainly - **lowers the win rate**, in the tightest case from 73.5% to 53.7%.
The intuition that a stop improves the hit rate by cutting losers early is simply wrong here: a
stop converts trades that would have recovered into realised losses, and the 120-day cap plus the
7.5% target already do the job with less damage.

### The 0.5 blend survives re-validation

The 50/50 blend of the rolling model and the causal composite was chosen a priori in round 8 and is
the largest single effect in the project, so it was re-tested paired on all nine models under the
new policy:

| rank-blend | CAGR | vol | max DD | win | IR | vs 0.50 |
|---|---:|---:|---:|---:|---:|---|
| 0.35 | 8.88% | 9.70% | -16.15% | 75.08% | 0.915 | CAGR -0.29pp (t = -0.85), win +0.32pp (t = 0.65) |
| **0.50** | **9.16%** | 9.67% | **-15.92%** | 74.76% | **0.948** | - |
| 0.65 | 8.26% | 9.69% | -15.92% | 73.66% | 0.853 | CAGR **-0.91pp** (t = -2.08), win **-1.11pp** (t = -2.10) |

0.50 remains the best point, and it is the only one of the three whose information ratio exceeds
0.93. The a-priori choice holds up.

### The drawdown overlay is inert at this exposure, and tightening it is not free

The 0.08/0.12 tiers were set under the old policy. With `c2` the volatility target already holds
gross near 0.5, so a cap of 0.6 or 0.7 never binds: changing the tiers to 0.10:0.7,0.15:0.4 or to
0.08:0.6,0.13:0.3 reproduces `c2` **exactly**, to the last decimal. Tightening far enough to bind
costs return without buying drawdown:

| variant | CAGR | vol | max DD | IR | win |
|---|---:|---:|---:|---:|---:|
| `c2` (tiers 0.08:0.7, 0.12:0.45) | **10.76%** | 9.77% | -13.45% | **1.101** | **73.53%** |
| tiers 0.05:0.5, 0.08:0.25, 0.12:0.1 | 9.31% | 9.10% | -13.66% | 1.023 | 71.17% |
| tiers 0.06:0.6, 0.10:0.35 | 8.36% | 9.66% | -15.96% | 0.865 | 72.39% |
| `--risk-trim-mode close_weakest` | -0.10% | 6.33% | -19.86% | -0.015 | 47.07% |

The last row is worth recording for the record: trimming by closing the weakest whole position
rather than scaling every weight destroys the strategy outright. It does not reduce risk
(-19.9% drawdown against -13.5%) and it halves the win rate.

### What round 25 establishes

The `c2` configuration - market ladder off, volatility target 0.055, 7.5% profit target, 120-day
hold, 50/50 blend, 12 names - is a **local optimum on every axis that has been re-opened under the
new policy**: exits, blend, and the account drawdown overlay. Nothing inherited from the old regime
is silently holding the result back any more.


## Round-26 findings: a design change that failed, and the frontier moving up

With design changes now allowed, round 26 tested a genuine change to the exit semantic and then
applied the round-23/24 exit finding to the return end of the frontier.

### Design change: a risk-consistent (ATR-scaled) profit target - rejected

The profit target is a flat percentage, which means different things for different names: 7.5% is
half a daily standard deviation for a quiet name and a fifth for a volatile one. The natural design
fix is to express it in multiples of the position's own ATR at entry
(`profit_target_atr_mult`, clamped to 2-30% of the entry price). The first implementation was a
no-op - the live exit path is `V2Strategy.evaluate_exits`, not the older
`portfolio.evaluate_exits` - and every run reproduced `c2` bit for bit. With the policy field, the
builder and the live path all wired, and a control run confirming that a zero multiple still
reproduces `c2` exactly:

| target rule | CAGR | vol | max DD | IR | win | trades | avg hold |
|---|---:|---:|---:|---:|---:|---:|---:|
| **flat 7.5%** (`c2`) | **10.76%** | 9.77% | **-13.45%** | **1.101** | **73.53%** | 442 | 48.7 |
| 3.0 x ATR | 9.21% | 9.52% | -14.62% | 0.967 | 72.96% | 429 | 50.4 |
| 4.0 x ATR | 9.24% | 9.96% | -16.53% | 0.928 | 67.60% | 358 | 60.3 |
| 2.0 x ATR | 7.10% | 9.86% | -15.78% | 0.721 | 74.50% | 498 | 42.7 |
| 8.0 x ATR | 7.95% | 9.90% | -16.69% | 0.803 | 62.10% | 248 | 86.4 |
| 6.0 x ATR | 7.91% | 10.20% | -19.07% | 0.775 | 64.21% | 299 | 72.4 |

Rejected: a flat percentage beats a risk-consistent target on every axis. The reason is that
scaling the target by the name's own volatility makes the *quiet* names - which are the ones this
strategy selects - exit at very small gains, which raises turnover and fee drag without improving
the hit rate, while the volatile names are held past the point where the mean reversion has played
out.

### The same exit redesign at the return end

Round 24's exit finding was validated only at the compliant exposure. Applying it at the return end
(`--use-market-timing 0 --vol-target 0.14`), nine seeds:

| configuration | CAGR | vol | max DD | win | IR | seeds with win >= 70% |
|---|---:|---:|---:|---:|---:|---:|
| `v2_g4` old headline (ladder on, tp 3.5%, hold 45) | 9.93% | 11.84% | -17.90% | 68.92% | 0.844 | 4/9 |
| `notm` (ladder off, tp 3.5%, hold 45) | 12.12% | 15.00% | -26.57% | 72.28% | 0.806 | 8/9 |
| `r120` (ladder off, tp 3.5%, hold 120) | 11.22% | 14.51% | -23.73% | 80.54% | 0.771 | 9/9 |
| **`z1` (ladder off, tp 7.5%, hold 120)** | **13.76%** +/- 2.71 | 14.58% | -23.18% | **75.39%** | **0.945** | **9/9** |
| `z2` (ladder off, tp 9%, hold 90) | 14.11% +/- 3.09 | 14.69% | -25.10% | 66.94% | 0.957 | 2/9 |

`z1` against the old headline, paired on the same nine models:

| paired difference (`z1` - `v2_g4`) | mean | t | seeds in favour |
|---|---:|---:|---:|
| **CAGR** | **+3.83pp** | **3.83** | 8 of 9 |
| **win rate** | **+6.47pp** | **8.07** | **9 of 9** |
| volatility | +2.73pp | - | more |
| maximum drawdown | -5.28pp | - | worse |

and against `notm`, the round-22 return headline: CAGR +1.64pp (t = 1.59, not significant), win
rate **+3.11pp (t = 4.26, 9 of 9)**, volatility -0.43pp, drawdown **+3.39pp**. `z1` therefore
dominates `notm` - more return, more win rate, less volatility, less drawdown - and it is the best
return configuration the project has produced.

### The frontier after 26 rounds

Two points, both with an information ratio of about 0.95, so the exit redesign lifted the whole
frontier rather than tilting it:

| configuration | CAGR | vol | max DD | win | IR | checks on every seed |
|---|---:|---:|---:|---:|---:|---|
| `c2` ladder off, vt 0.055, tp 7.5%, hold 120 | 9.16% | **9.67%** | **-15.92%** | 74.76% | 0.948 | 4 of 7 (vol, win, f63, f252) |
| `z1` ladder off, vt 0.14, tp 7.5%, hold 120 | **13.76%** | 14.58% | -23.18% | **75.39%** | 0.945 | 3 of 7 |

Both endpoints beat the configuration this project carried for fourteen rounds (`v2_g4`: 9.93% at
11.84% volatility with a 68.92% win rate). The contract's CAGR requirement is still not met: 13.76%
against 25%, with an information ratio of 0.945 against the 2.5 the arithmetic requires.

### qlib components in use

The design changes were made in the strategy semantics only; the backtest still runs entirely
through Qlib's own machinery - `qlib.backtest.backtest`, `SimulatorExecutor` (TT_PARAL),
`Exchange`/`NumpyQuote` (subclassed for the A-share limit and friction rules),
`WeightStrategyBase` and `OrderGenerator` (subclassed for the V2 target weights),
`contrib.model.gbdt.LGBModel` via the rolling training harness, `contrib.evaluate.risk_analysis`
and `indicator_analysis`, `data.dataset.{DatasetH,DataHandlerLP,StaticDataLoader}`,
`model.riskmodel.shrink.ShrinkCovEstimator` for the covariance estimate, and
`qlib.workflow.R` for the recorder that every figure in these reports is read back from.


## Is the CAGR target reachable without new data? - the arithmetic

This is the question the whole project turns on, and after 26 rounds and 498 account runs it can be
answered with measured quantities rather than opinion.

**1. The CAGR/risk identity.** For a strategy whose exposure is scaled linearly, CAGR is
approximately the information ratio times the volatility budget. The two are not independent knobs.
The contract fixes the volatility budget at 10% and asks for 25%, so it requires **IR = 2.5**.

**2. What the information ratio actually is.** Over the 498 recorded runs with at least 200 round
trips: mean IR 0.803, median 0.823, 95th percentile 1.119, **maximum 1.243** (a single model seed of
`z1`). The seed-mean IR of the two frontier points is 0.945-0.948. So the target needs **2.0 to 2.6
times the best information ratio this project has ever measured**, and about 2.6 times the value
that survives averaging over model seeds.

**3. The drawdown constraint is the binding one, and it is structural.** Across those 498 runs the
drawdown-to-volatility ratio has mean 1.661, median 1.629 and **minimum 1.213**. Because the
contract caps the drawdown at -10% *and* the volatility at 10%, the drawdown constraint is what
limits exposure:

| run | vol | max DD | ratio | IR | exposure scale allowed | risk-feasible CAGR |
|---|---:|---:|---:|---:|---:|---:|
| `z1` (seed 42), the best run in the project | 14.65% | -19.84% | 1.354 | **1.243** | 0.50 | **9.18%** |
| `v2_g4` (best drawdown ratio with a good IR) | 11.82% | -15.00% | 1.269 | 1.133 | 0.67 | 8.93% |
| `c2` (the compliant point) | 9.67% | -15.92% | 1.646 | 0.948 | 0.63 | 5.75% |

That is, satisfying vol <= 10% **and** drawdown >= -10% forces exposure down to 50-67% of what the
best configurations run at, and the resulting CAGR tops out at **about 9.2%** over every run this
project has produced.

**4. The ceiling even under the most favourable measured numbers.** Using the single lowest
drawdown ratio ever observed (1.213, which implies vol <= 8.24%) together with the single highest
information ratio ever observed (1.243) gives a maximum conceivable CAGR of **10.2%**. The contract
demands 25%. The two risk constraints, taken together with any information ratio this strategy has
demonstrated, cap the CAGR at roughly **40% of the target**.

**5. What would have to change.** The information ratio has to roughly double, from ~1.0 to ~2.5.
The one mechanism in this project that ever moved IR by more than the noise band was *combining two
weak, complementary signals*: the rolling model alone gives IR 0.41, the causal composite alone
0.36, and the 50/50 blend of the two gives **1.13** - a 2.8x jump from a single pairing. Every
attempt to add a third source (VWAP features, idiosyncratic volatility, the MAX effect, drawdown,
momentum, the head re-rank, broader books) has failed, and the seed-averaged frontier has sat at
0.95 for the last four rounds. Reaching 2.5 would need several more such pairings, from data this
database does not contain.

**Conclusion.** Changing the strategy's design without changing the data cannot reach 25% CAGR at
10% volatility and -10% drawdown. The win-rate half of the objective *is* met and is robust
(74.8-80.5%, clearing 70% on 9 of 9 model seeds). The return half is short by a factor of about
2.5, and the binding constraint is an arithmetic one: the contract's own risk limits, combined with
the measured drawdown-to-volatility ratio, leave room for a CAGR of about 9-10%.


## Can the drawdown be bought down by cutting exposure or going to cash? - measured, not argued

This is the natural objection to the arithmetic in the previous section: the drawdown-to-volatility
ratio of ~1.6 is what caps the contract-feasible CAGR, so surely a rule that cuts exposure during
drawdowns, or goes flat, would improve the ratio? It has now been tested directly, including the
two failure modes that make it impossible.

### The hard cut-to-cash rule is self-locking, by construction

`drawdown_cap` measures the drawdown as `nav / peak_nav - 1` against the **all-time** peak NAV. If
the rule sells everything, the NAV is frozen: it cannot fall further, but it also cannot rise, so
the drawdown never improves and the rule never releases. Run on the compliant configuration
(`scripts/sweep_r27_dd.sh`):

| variant | CAGR | vol | max DD | DD/vol | IR | win | round trips |
|---|---:|---:|---:|---:|---:|---:|---:|
| `c2` baseline overlay (0.08:0.7, 0.12:0.45) | **10.76%** | 9.77% | **-13.45%** | **1.376** | **1.101** | **73.53%** | **442** |
| tighter proportional trims (0.06:0.6, 0.10:0.35) | 8.36% | 9.66% | -15.96% | 1.652 | 0.865 | 72.39% | 431 |
| **hard cut to cash at 9% drawdown, all-time peak** | **0.38%** | 4.14% | -12.62% | **3.049** | **0.093** | 65.62% | **64** |
| hard cut to cash at 9%, 60-day rolling peak | 8.17% | 9.05% | -14.47% | 1.599 | 0.903 | 71.73% | 474 |
| hard cut to cash at 12%, 120-day rolling peak | 7.57% | 8.96% | -15.83% | 1.766 | 0.844 | 70.83% | 408 |
| baseline tiers, drawdown measured on a 60-day rolling peak | 10.76% | 9.77% | -13.45% | 1.376 | 1.101 | 73.53% | 442 |

Three results, all pointing the same way:

1. **Going to cash on the account's own drawdown destroys the strategy**: 64 round trips instead
   of 442, CAGR 0.38%, and the *worst* drawdown-to-volatility ratio in the whole project (3.05).
   The rule locks itself out exactly as the arithmetic says it must.
2. **Making it able to re-enter does not rescue it.** Measuring the drawdown against a rolling NAV
   high (`--dd-peak-window`, added for this test) lets the rule release itself after the window -
   and the result is still worse than doing nothing: 8.17% CAGR at 9.05% volatility with a **deeper**
   drawdown (-14.47% against -13.45%). Cutting after a loss and re-entering later converts a
   temporary drawdown into a permanent one and misses the recovery.
3. **Tightening the existing overlay makes the drawdown worse, not better** (-15.96% against
   -13.45%): the same effect at smaller scale.

### The same result across the whole search

Of the 498 recorded account runs with at least 200 round trips, **exactly one** ever held its maximum
drawdown under 10%: `v2_pr3`, at -7.92% drawdown, 3.93% volatility, **2.26% CAGR** and a 66.3% win
rate - it achieved the drawdown target by holding almost no risk at all. The best CAGR attainable
while holding the drawdown under each ceiling:

| drawdown ceiling | best CAGR over 498 runs | that run's vol | runs qualifying |
|---|---:|---:|---:|
| <= 10% | 2.26% | 3.93% | 1 |
| <= 12% | 2.26% | 3.93% | 1 |
| <= 14% | 11.81% | 11.11% | 18 |
| <= 16% | 13.39% | 11.82% | 98 |
| <= 18% | 13.91% | 13.42% | 222 |
| <= 20% | 18.21% | 14.65% | 316 |

The de-risking tools are therefore not missing from the search - they are *in* it, in the form of the
market ladder (with a full cap vector sweep), the account drawdown tiers, the portfolio volatility
target, a market-volatility brake, and forced liquidation on the market extreme. Every one of them
was measured, and the ratio distribution they produce has a hard left tail at 1.21.

### Why the ratio is structural

A long-only equity book with positive drift and volatility sigma produces drawdowns of roughly
1.3-1.8 sigma; getting the maximum drawdown down to 10% while running 10% volatility would require
a ratio of 1.0, which no configuration in 498 runs has ever come close to. The only lever that
actually reduces the maximum drawdown is reducing the *average* exposure - which is exactly what
`c2` (volatility 9.67%) and `v2_pr3` (3.93%) do - and CAGR falls with it, along a line of slope
equal to the information ratio.

The answer to "can the drawdown be optimized by cutting positions or going to cash" is therefore
**no**, and the reason is not a lack of effort: it was tested with the strongest possible version of
the idea (full liquidation), the self-locking failure mode was identified, a mechanism was added to
defeat it (rolling-peak drawdown), and the released version is still worse than doing nothing.


## Correction: the drawdown constraint is NOT a structural constant - it is a function of the information ratio

The section above ("Is the CAGR target reachable without data?") argued that the
drawdown-to-volatility ratio is structural and therefore caps the contract-feasible CAGR near 10%.
That argument was **too strong and is corrected here**. Looking at the ratio across the 503 runs
rather than within one configuration:

| IR quintile | mean IR | mean DD/vol | minimum DD/vol | mean CAGR |
|---|---:|---:|---:|---:|
| Q1 (lowest) | 0.478 | **1.837** | 1.277 | 5.5% |
| Q2 | 0.708 | 1.658 | 1.339 | 8.3% |
| Q3 | 0.829 | 1.676 | 1.258 | 10.1% |
| Q4 | 0.928 | 1.642 | 1.225 | 11.3% |
| Q5 (highest) | 1.078 | **1.486** | 1.213 | 12.5% |

`corr(IR, DD/vol) = -0.546` (Spearman -0.436) over 503 runs. **The drawdown ratio is not a
constant of nature; it falls as the information ratio rises.** A fitted line
`ratio ~= 2.12 - 0.59 * IR` puts the ratio at 1.0 near IR 1.9 and at 0.65 near IR 2.5.

Writing the contract's two risk constraints together: with volatility capped at 10% and the
drawdown at `ratio * vol <= 10%`, the largest feasible CAGR is `IR * min(10%, 10%/ratio)`. For 25%
this needs `IR / ratio >= 2.5`, which the fitted relationship satisfies at **IR around 2.2**, and
comfortably so at IR 2.5 (ratio ~0.65, drawdown ~6.5%, CAGR 25%).

**The contract is therefore internally consistent**: it describes a strategy with an information
ratio of about 2.5. It is not an arithmetic impossibility - the strategy simply has IR ~1.0-1.3,
and at that quality the drawdown constraint bites and leaves about 9% of CAGR. This is a
strategy-quality problem, exactly as the task owner suspected, not a structural wall.

### Two design hypotheses tested at the same time, both negative

**A market-neutral redesign does not help.** Regressing each run's daily account return on the
market proxy the run itself recorded:

| run | beta | alpha (ann.) | R2 | account vol | idiosyncratic vol | IR | market-neutral IR |
|---|---:|---:|---:|---:|---:|---:|---:|
| `c2` (seed 42) | 0.310 | 5.69% | 0.598 | 9.77% | 6.20% | 1.146 | 0.918 |
| `z1` (seed 42) | 0.473 | 10.20% | 0.620 | 14.64% | 9.03% | 1.295 | 1.130 |
| `v2_g4` | 0.357 | 7.54% | 0.541 | 11.82% | 8.00% | 1.180 | 0.942 |

The book is genuinely low-beta (0.31-0.47), but about 60% of its *variance* is still market
variance simply because the market is 2.5x as volatile as the account. Hedging that beta away
lowers the information ratio (1.15 -> 0.92, 1.30 -> 1.13): the market exposure is carrying return,
not just risk. A market-neutral construction is not the missing design change.

**The universe axis is closed.** Raising the liquidity floor under the new exit policy (the design's
small-and-illiquid tilt was the one unexplored universe choice):

| floor (amount rank) | CAGR | vol | max DD | DD/vol | IR | win |
|---|---:|---:|---:|---:|---:|---:|
| 0.20 (the design) | **10.76%** | 9.77% | **-13.45%** | 1.376 | **1.101** | 73.53% |
| 0.35 | 4.72% | 9.82% | -19.97% | 2.034 | 0.481 | 71.24% |
| 0.50 | 3.66% | 9.55% | -18.67% | 1.955 | 0.383 | 69.33% |
| 0.65 | 4.42% | 9.59% | -14.38% | 1.499 | 0.460 | 70.03% |
| 0.80 | 4.25% | 9.24% | -12.04% | 1.304 | 0.460 | 70.03% |

Moving up to liquid names costs more than half the return at unchanged volatility. The illiquidity
tilt *is* the alpha in this data, and the better drawdown ratios at the high floors come with a
strategy that has almost no return left.


## Stage-1 round 1: the rank consensus is a selection effect, and the historical ablation was mislabelled

### Correction: the "ranker ablation" never tested the composite alone

`apply_v2_signals` only enters its blend branch when `blend > 0 or extra_weight > 0`. With
`--rank-blend 0.0` it skips the branch entirely and the entry score stays the **raw model
prediction**. The long-standing ablation - "composite-only IR 0.36, model-only IR 0.41, blend IR
1.13" - is therefore mislabelled: the "composite-only" arm was a *model* arm (raw score instead of
percentile rank), and the composite had never been measured on its own.

`--composite-only` has been added so that `--rank-blend 0` means "rank by the rule composite".
Measured on the current headline configuration (market ladder off, vol target 0.14, 7.5% target,
120-day hold, seed 42):

| arm | CAGR | vol | max DD | win | IR |
|---|---:|---:|---:|---:|---:|
| blend 0.50 (model + combo3) | **18.21%** | 14.65% | -19.84% | **77.83%** | **1.243** |
| **composite-only** (true) | 13.12% | 14.52% | -21.97% | 74.16% | **0.903** |
| model-only (percentile rank) | 10.08% | 14.18% | -23.21% | 70.57% | 0.711 |
| raw model score (the old "composite-only") | 9.66% | 14.23% | -23.21% | 69.35% | 0.679 |
| blend 0.35 | 15.46% | 15.16% | -22.96% | 77.22% | 1.020 |

The blend beats the **best** single arm by 38% of information ratio, not by 200%. The mechanism
story in the target prompt - "combining two weak signals tripled IR" - is wrong and is corrected
there.

### What the consensus actually does

The two arms' daily returns correlate **0.670**, and a 50/50 **return** blend of exactly the same two
arms gives IR **0.935** - no better than the composite alone. The **rank** blend gives **1.295** on
the same dates:

| construction | CAGR | vol | IR |
|---|---:|---:|---:|
| composite-only account | 13.66% | 14.52% | 0.941 |
| model-only account | 10.49% | 14.18% | 0.740 |
| 50/50 **return** blend of the two | 12.26% | 13.12% | 0.935 |
| **50/50 rank consensus (the headline)** | **18.98%** | 14.65% | **1.295** |

So the consensus is **not portfolio diversification** - it is a *different selection rule* that buys
names both rankers like. That is the exploitable primitive, and it is a design primitive rather than
a signal.

### Two attempts to extend it, both negative

**Third opinions dilute the consensus.** If agreement is the mechanism, a third opinion should
sharpen it. Thirteen partners were tested as a third ranker at 20-33% of the score (weight shifting
from the two existing arms), on the same configuration:

| third opinion | weight (model / partner / composite) | CAGR | vol | IR |
|---|---|---:|---:|---:|
| none (control) | 0.50 / - / 0.50 | **18.21%** | 14.65% | **1.243** |
| t60-label model | 0.25 / 0.15 / 0.60 | 15.27% | 15.27% | 1.000 |
| t60-label model | 0.33 / 0.33 / 0.34 | 13.50% | 14.18% | 0.952 |
| fwd20-label model | 0.40 / 0.20 / 0.40 | 14.00% | 15.07% | 0.929 |
| another seed of the same model | 0.33 / 0.33 / 0.34 | 10.83% | 13.93% | 0.778 |
| semidev20 rank | 0.40 / 0.20 / 0.40 | 14.79% | 14.08% | 1.050 |
| ivol60 rank | 0.40 / 0.20 / 0.40 | 11.21% | 14.09% | 0.796 |
| maxret20 rank | 0.40 / 0.20 / 0.40 | 12.00% | 14.37% | 0.835 |
| dd252 / skew60 / pos_days20 / beta60 ranks | 0.40 / 0.20 / 0.40 | 7.6-13.8% | - | 0.57-0.92 |

Every single one loses 2.9-10.6pp of CAGR. Two complementary opinions are the optimum; a third
dilutes.

**Sub-book ensembling over the whole corpus gains almost nothing.** Every account run with a
recorded NAV was loaded (511 runs, aligned to 1868 days) and every pair was combined:

* median pairwise correlation of daily returns **0.836** (p1 0.614, p99 0.963);
* best pair combination IR in the entire corpus **1.0735**, against the best single member 0.988 -
  a gain of **+0.086**;
* the largest gain over the better member of any pair is **+0.086**, at correlation 0.71.

With the combination law `IR_k = IR_1 * sqrt(k / (1 + (k-1) * rho))`, reaching 2.5x from sub-books
requires `rho <= 0.16`. The only corpus pairs near that correlation involve a degenerate run that
stopped trading (64 round trips). **The ensemble route is closed by the correlation structure of
what this signal produces**, not by effort.

### Where stage 1 stands

Stage 1 needs a **seed-averaged IR >= 1.30**. The best seed-averaged value measured is 0.948
(`c2`) and 0.945 (`z1`); the best single seed reaches 1.295. Two design levers were tested this
round (third-opinion consensus, sub-book ensembling) and both are negative. The corrected ablation
does, however, hand over one concrete lead: the consensus is a *selection* primitive, so the way to
use it is to find a **structurally different second ranker**, not more of the same - the composite
and the model are complementary at correlation 0.67, and the search for a third arm should look for
something with the same kind of complementarity to *both* (an event/flow signal would be the
candidate, but this data set does not contain one).


### The consensus advantage is only ~5% once the seed is averaged - and the composite alone needs no model

The corrected ablation was first measured on one seed, which is exactly the mistake this project has
made repeatedly. Repeating the two arms on four seeds:

| arm | seed 42 | seed 7 | seed 101 | seed 23 | mean IR |
|---|---:|---:|---:|---:|---:|
| composite-only | 0.903 | **0.903** | **0.903** | **0.903** | **0.903** |
| model-only | 0.711 | 0.336 | 0.583 | 0.658 | 0.572 |
| rank consensus (from the nine-seed `z1` panel) | 1.295 | 1.082 | - | - | **0.945** |

Two things fall out, and both matter more than the numbers above them.

1. **The composite arm is seed-independent, exactly as it must be** - with `--composite-only` the
   model prediction is never read, and the four runs are bit-identical (CAGR 13.12%, vol 14.52%,
   drawdown -21.97%, win 74.16%). That is a clean confirmation that the new flag works, and it also
   means **0.903 is a precise estimate rather than a draw**.
2. **The nine-seed consensus mean is 0.945 against 0.903 for the composite alone: the model adds
   about 5%.** The impressive single-seed gap (1.295 vs 0.903) was seed 42 being lucky, which is the
   same trap this project fell into with the 13.39% headline.

So the strategy is, to a first approximation, **a pure rule-composite strategy**: rank the base
universe by the average of (-vol20, -amount20, -distance below MA60), buy the top name each day, exit
at +7.5% or after 120 days, hold gross near the volatility target. The rolling LightGBM - the most
expensive component in the pipeline - contributes roughly 0.04 of information ratio.

That redirects the search. The highest-leverage object is no longer the model or the combination
rule; it is the composite itself, and the composite has only three components whose weights are
already at a local optimum (round 8: the 2/1/1 weighting costs 8pp of CAGR).


## Stage-1 round 2: the engine is illiquidity, not low volatility

Stage-1 round 1 showed the rolling model contributes only ~5% of the information ratio and that the
composite-only arm is **seed-independent** (bit-identical across four seeds, IR 0.903). The composite
is therefore the strategy, and this round attributes it.

### The build had to be replicated first

A single-run attribution (one path per composite, no model) produced absurd non-additivity - pairs
far worse than either single component, information ratios from 0.087 to 0.969. That is the
signature of path noise, so each composite was given **nine replications** by keeping the composite
in charge (weight 0.8) and letting the model provide a small per-seed perturbation (weight 0.2).
Replication immediately overturned two single-run conclusions, including the tempting
"solo amount beats combo3".

### The replicated attribution (nine seeds each, equal seed weights)

| composite | components | CAGR | sd | vol | max DD | win | IR | IR sd |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `solo_amt` | amount20 | 14.80% | 2.22 | 15.06% | -24.52% | **78.21%** | **0.983** | 0.145 |
| `combo3` | vol20 + amount20 + dist_MA60 | 14.35% | 2.21 | 14.71% | -21.99% | 76.38% | 0.976 | 0.152 |
| `pair_amt_ma` | amount20 + dist_MA60 | 14.63% | 2.03 | 16.05% | -25.84% | 77.61% | 0.911 | 0.123 |
| `pair_vol_amt` | vol20 + amount20 | 9.79% | 1.86 | 13.93% | -23.57% | 70.71% | 0.703 | 0.129 |
| `solo_ma` | dist_MA60 | 11.08% | 2.92 | 16.56% | -32.21% | 73.75% | 0.665 | 0.153 |
| `solo_vol` | vol20 | 5.17% | 1.41 | 11.33% | -21.83% | 62.14% | 0.453 | 0.110 |
| `pair_vol_ma` | vol20 + dist_MA60 | 6.09% | 1.41 | 13.57% | -23.92% | 69.20% | 0.450 | 0.107 |

Paired against `combo3` on the same nine seeds:

| variant | CAGR difference | t | seeds in favour |
|---|---:|---:|---:|
| `solo_amt` | +0.46pp | +0.39 | 5/9 |
| `pair_amt_ma` | +0.29pp | +0.33 | 5/9 |
| `solo_ma` | -3.26pp | -2.97 | 2/9 |
| `pair_vol_amt` | -4.55pp | -5.16 | 0/9 |
| `pair_vol_ma` | -8.25pp | -9.79 | 0/9 |
| `solo_vol` | -9.18pp | -13.16 | 0/9 |

Two conclusions, and the first contradicts the project's own framing since round 1:

1. **The engine is the liquidity/turnover component.** Every composite that contains `amount20` is
   tied at the top (0.91-0.98); every composite that does not is 4.6-9.2pp worse in CAGR, on 0 of 9
   seeds, with the paired t statistics at -5 to -13. This is an **illiquidity/neglect premium**, not a
   low-volatility anomaly.
2. **Low volatility contributes approximately nothing.** `vol20` alone gives IR 0.45 and a 62% win
   rate, and it *reduces* the account when paired with the liquidity measure (`pair_vol_amt` 0.703
   against `solo_amt` 0.983). Its presence in `combo3` is, on this evidence, decoration.

### The liquidity window does not matter

Since the liquidity measure is the engine, the measure itself was varied: 5-day, 20-day and 60-day
turnover, a 5/20 turnover-acceleration ratio, and Amihud illiquidity. A single-seed screen suggested
the 60-day window (IR 1.296); replicated over nine seeds:

| measure | n | CAGR | sd | vol | max DD | win | IR |
|---|---:|---:|---:|---:|---:|---:|---:|
| amount20 (current) | 9 | 14.80% | 2.22 | 15.06% | -24.52% | 78.21% | 0.983 |
| amount60 | 7 | 15.80% | 1.83 | 15.42% | -24.89% | 76.80% | 1.029 |
| amount5 | 9 | 14.85% | 1.75 | 13.69% | -23.66% | 78.26% | 1.091 |
| amount 5/20 acceleration ratio | 1 | 9.67% | - | 14.59% | -26.03% | 75.86% | 0.663 |
| Amihud illiquidity | 1 | 1.59% | - | 10.20% | -24.56% | 64.97% | 0.156 |

Paired: `amount60 - amount20` = +0.86pp of CAGR (t = +1.46, 4 of 7) and **-1.50pp of win rate**
(t = -2.01, 1 of 7); `amount5 - amount20` = +0.05pp of CAGR (t = +0.07, 4 of 9) and +0.05pp of win
rate (t = +0.08). The apparent edge of the 5-day window is a volatility artefact - it runs at 13.69%
volatility instead of 15.06%, i.e. lower exposure, not a better signal. **The window is irrelevant
between 5 and 60 days**, which is a robustness result rather than an improvement; the acceleration
ratio and Amihud are both much worse.

### Where this leaves stage 1

Stage 1 needs a seed-averaged IR >= 1.30. The best replicated value is **0.983** (`solo_amt`, nine
seeds) and 0.976 for the incumbent `combo3`; the best single seed reaches 1.296. The engine has now
been identified (illiquidity) and two of its obvious degrees of freedom (window, companion
component) are closed. The next question is whether anything *else* in this data set is
complementary to an illiquidity ranking in the way the model and the composite are complementary to
each other - which is the same open question stage-1 round 1 ended on.


## Stage-1 round 3: the book is slot-constrained, and four more axes close

### The constraint structure

`coverage_daily.csv` and `constraints_daily.csv` for the return-end configuration answer a
question the project had never asked: what actually limits the book?

| quantity | value |
|---|---:|
| candidates passing the gate per day | **3044** (mean) |
| planned orders per day | 0.516 |
| days with at least one new position | 466 of 1869 (25%) |
| mean positions held | **11.73 of 12** |
| mean gross exposure | 0.715 against a 0.693 target |

So the book is **slot-constrained, not candidate-constrained**: it sits at 11.73 of its 12 slots
while three thousand names pass the gate every day, and a new position is bought whenever a slot
frees, whatever the quality of that day's best candidate. That is a much more informative statement
about the design than the parameter sweeps, and it motivated three of the four tests below (breadth,
gates, and the trigger).

### Breadth, retested under the new exit policy - still closed

Round 18 rejected breadth with a 3.5% target and a 45-day cap. With a 7.5% target and a 120-day cap
the average hold is 48 days against a 12-slot book, which is a different constraint regime, so it
was retested paired on the same nine seeds:

| names | n | CAGR | sd | vol | max DD | win | IR |
|---|---:|---:|---:|---:|---:|---:|---:|
| 12 (incumbent) | 9 | 13.76% | 2.71 | 14.58% | -23.18% | 75.39% | 0.945 |
| 20 (max weight 4.8%) | 7 | 14.41% | 2.13 | 15.14% | -24.79% | 75.34% | 0.950 |
| 30 (max weight 3.2%) | 9 | 10.50% | 1.76 | 14.41% | -24.68% | 74.34% | 0.727 |

Paired: 20 names +0.69pp of CAGR (t = +0.58, 4 of 7) - a tie; 30 names **-3.26pp** (t = -3.54, 1 of 9).
The information in the ranking is concentrated in its first dozen names under both policies.

### The profit target as a resting limit order - rejected

Triggering the target on an intraday touch of the daily high (added as
`--profit-trigger-high`) is the realistic semantic for a limit order. The zero control reproduces
the headline exactly:

| variant | CAGR | vol | max DD | win | trades | avg hold | IR |
|---|---:|---:|---:|---:|---:|---:|---:|
| control (close trigger) | **18.21%** | 14.65% | **-19.84%** | **77.83%** | 433 | 49.5 | **1.243** |
| intraday trigger (touch) | 13.03% | 14.62% | -25.99% | 76.86% | 484 | 43.9 | 0.891 |
| touch + 5% target | 10.25% | 14.58% | -24.64% | 78.86% | 648 | 32.6 | 0.703 |
| touch + 10% target | 14.63% | 15.09% | -24.04% | 74.13% | 402 | 54.0 | 0.970 |

It is worse at every target: the touch fires earlier, but the fill still happens at the next close,
which is frequently below the target, so the rule banks a smaller gain than waiting for the close to
confirm.

### The entry quality gate - rejected

The book buys whenever a slot frees, so raising the candidate-quality bar should in principle help.
It does the opposite:

| gate | CAGR | vol | max DD | win | trades | IR |
|---|---:|---:|---:|---:|---:|---:|
| none (control) | **18.21%** | 14.65% | -19.84% | **77.83%** | 433 | **1.243** |
| `--p-min 0.50` | 13.87% | 14.06% | -24.53% | 76.81% | 345 | 0.986 |
| `--p-min 0.60` | 4.04% | 11.35% | -23.72% | 68.33% | 240 | 0.356 |
| `--p-min 0.70` | 0.61% | 5.72% | -9.71% | 71.70% | **53** | 0.107 |
| `--p-min 0.80` | -0.22% | 0.76% | -3.12% | 50.00% | **2** | -0.286 |
| `--mu-min 0.00` | 5.47% | 10.82% | -20.64% | 68.90% | 254 | 0.505 |
| `--mu-min 0.02` | 1.20% | 7.64% | -16.13% | 66.95% | 118 | 0.157 |

The model's calibrated win probability is not sharp enough to gate on: at 0.70 only 53 trades
survive in eight years and the account degenerates. This is the same message as the model's 5%
contribution, expressed as a design constraint.

### Stage-1 status after round 3

Stage 1 needs a seed-averaged IR >= 1.30; the best replicated value remains **0.983**. Four more
axes are now closed (breadth, intraday trigger, entry gates, and - from round 2 - the liquidity
window and companion component), and the constraint diagnostic says the remaining structural
freedom is the number of slots, whose information content is exhausted after about twelve names.


## Stage-1 round 4: the two remaining structural levers, both closed

The constraint diagnostic from round 3 (11.73 of 12 slots full, 3044 candidates a day) leaves exactly
two structural levers: which names occupy the slots, and how the risk is spread across them. Both
were implemented and tested this round, each with a zero control that reproduced the headline
bit-for-bit.

### Releasing the slot when a holding falls out of favour - rejected

The rank exit - leave a position once its blended score drops out of the top quantile - exists in the
config (with default `True`!) and is implemented in the older `portfolio.evaluate_exits`, but the V2
live path never read it. **It has therefore been silently inert in all ~520 recorded runs**, the
third instance of this trap in the project. It is now implemented in the live path
(`--exit-rank`, `--rank-exit-pct`, `--rank-exit-confirm`, `--rank-exit-min-hold`).

| variant | CAGR | vol | max DD | win | trades | avg hold | IR |
|---|---:|---:|---:|---:|---:|---:|---:|
| control (off) | **18.21%** | 14.65% | **-19.84%** | **77.83%** | 433 | 49.5 | **1.243** |
| exit below the median, confirm 2 | 11.55% | 14.86% | -23.57% | **54.73%** | 1100 | 18.5 | 0.777 |
| exit below the 30th percentile, confirm 2 | 10.40% | 14.88% | -22.84% | 55.08% | 995 | 21.0 | 0.699 |
| exit below the 70th percentile, confirm 2 | 11.09% | 15.07% | -24.24% | 54.02% | 1231 | 16.0 | 0.736 |
| below the 30th percentile, confirm 5 | 13.01% | 15.71% | -21.55% | 60.03% | 788 | 27.5 | 0.828 |
| below the median, confirm 10 | 14.19% | 14.90% | -23.31% | 62.82% | 694 | 31.0 | 0.953 |

Every setting loses 4-8pp of CAGR, and the win rate collapses from 77.8% to 54-63%: a rank exit
closes positions before the 7.5% target is reached. This is the **sixth** falsification of cutting
positions early - the first five were price-triggered (stops, trailing, stale-loser, profit lock),
this one is ranking-triggered.

### Re-weighting the sleeve toward inverse volatility - at best a tie

New positions are sized inversely to volatility but existing ones are never re-sized, so the book
drifts and its risk is dominated by whichever names have run. `--risk-parity` pulls every held
weight toward the inverse-vol target at the same gross, within the per-name caps:

| variant | CAGR | vol | max DD | win | trades | IR |
|---|---:|---:|---:|---:|---:|---:|
| control (off) | **18.21%** | 14.65% | **-19.84%** | 77.83% | 433 | **1.243** |
| strength 0.25 | 17.95% | 14.42% | -20.45% | 77.78% | 423 | 1.244 |
| strength 0.50 | 15.08% | 14.06% | -23.58% | 78.04% | 419 | 1.073 |
| strength 1.00 | 11.21% | 11.19% | -18.02% | 76.54% | 422 | 1.002 |

A quarter-strength tilt is an exact tie (1.244 against 1.243); stronger settings lose return roughly
in proportion to the volatility they remove, i.e. they move along the frontier rather than rotating
it. Work item 3 is closed in this form.

### Stage-1 status after four rounds

Nine axes closed: third-opinion consensus, sub-book ensembling, the liquidity measure and its window,
breadth, the intraday exit trigger, the entry quality gates, the ranking-triggered exit, and
inverse-vol re-weighting. The best replicated information ratio remains **0.983** (`solo_amt`,
nine seeds) against stage 1's 1.30, and the model contributes ~5% of it.

The one intervention in this whole project that has ever moved the information ratio materially is
still the **exit-policy redesign** of rounds 23-24 (3.5% -> 7.5% target, 45 -> 120-day cap), which
added 2.5pp of CAGR and 8pp of win rate at the same risk. Everything since has been a negative.
That asymmetry is itself the most useful thing the search has produced: the strategy's entry side is
at a strong local optimum, and the exit side is where its remaining quality lives.


## Stage-1 round 5: hard consensus, absolute patience (a no-op), and relative patience

### Hard consensus is worse than the average

The entry score is the *average* of the model rank and the composite rank, which lets a name ranked
first by one arm and 500th by the other win the day. Replacing the average with the elementwise
minimum - a hard "both rankers must like it" rule - was implemented as `--blend-mode min`:

| blend | CAGR | vol | max DD | win | trades | IR |
|---|---:|---:|---:|---:|---:|---:|
| mean (control) | **18.21%** | 14.65% | **-19.84%** | **77.83%** | 433 | **1.243** |
| min (hard consensus) | 15.44% | 14.60% | -24.56% | 77.31% | 432 | 1.058 |
| min + absolute threshold 0.6 | 16.24% | 14.62% | -24.56% | 77.55% | 432 | 1.111 |

The soft consensus wins. The average of two percentile ranks is a better selector than their
minimum, which fits the round-1 finding that the consensus is a smooth selection primitive rather
than a filter.

### The absolute entry threshold is a silent no-op - and why

`--entry-min-score` was added so that a freed slot would wait for a decent candidate, and the
screen returned **bit-identical results** to the control at thresholds of 0.40, 0.50 and 0.60. It is
not a bug in the flag: the score is a percentile-rank average, and the **maximum** of that average
over ~3000 candidates is essentially always high, so no plausible absolute threshold can bind. The
"patience" idea needs a *relative* bar.

### The relative patience gate works mechanically and loses money

`--entry-quality-pct` only opens a slot on days whose best candidate is above the given percentile
of the trailing 252-day distribution of daily best candidates (verified to actually fire: the gate is
open on 70.5% / 57.0% / 42.3% of days at 0.50 / 0.70 / 0.90):

| gate | days open | CAGR | vol | max DD | win | trades | IR |
|---|---:|---:|---:|---:|---:|---:|---:|
| off (control) | 100% | **18.21%** | 14.65% | -19.84% | 77.83% | 433 | **1.243** |
| pct 0.50 | 70.5% | 11.57% | 13.86% | -25.10% | 75.79% | 347 | 0.835 |
| pct 0.70 | 57.0% | 10.24% | 12.55% | -24.14% | 76.57% | 303 | 0.816 |
| pct 0.90 | 42.3% | 8.89% | 8.52% | -16.63% | **83.14%** | 172 | 1.043 |
| pct 0.70, 120-day window | 54.9% | 10.86% | 12.47% | -25.30% | 75.93% | 295 | 0.871 |

Waiting for a high-quality day is strongly negative. The one interesting reading is the 0.90 gate:
it produces the **highest win rate ever measured in this project (83.14%)** at the cost of
two-thirds of the return - the same lever as everywhere else in this project, bought with exposure
rather than created.

The finding worth keeping is that **the level of the daily best score carries no information about
the next day's opportunity**: conditioning entries on it is harmful. That is a clean negative for a
mechanism that had a plausible prior.

### Stage-1 status after five rounds

Twelve axes closed: third-opinion consensus, sub-book ensembling, the liquidity measure and its
window, breadth, the intraday exit trigger, the absolute entry gate, the ranking-triggered exit,
inverse-vol re-weighting, hard consensus, and three forms of entry patience. The best replicated
information ratio remains **0.983** (`solo_amt`, nine seeds), against stage 1's 1.30.


## Stage-1 round 6: the short-hold corner, board composition, and one weak lead

### The short-hold corner of the exit grid - closed

Work item 2 asks for holding period x breadth x cost to be optimised jointly. Breadth was closed in
round 3, but the exit grid of round 24 only covered holds of 60 days and above, and the older hold
sweeps only 45 and above. The short-hold corner - where `IR ~ IC*sqrt(BR)` gains most while fees
scale with turnover - had never been searched under the current policy. Twelve cells (profit target
2-5%, hold 10-30 days) on one seed:

| profit target | hold 10 | hold 20 | hold 30 |
|---|---:|---:|---:|
| 2% | 0.269 (59.4% win) | 0.418 (66.1%) | 0.694 (70.6%) |
| 3% | 0.298 (56.7%) | 0.647 (63.6%) | 0.765 (68.8%) |
| 4% | 0.316 (53.9%) | 0.722 (61.5%) | **1.073** (67.2%) |
| 5% | 0.290 (52.6%) | 0.602 (58.1%) | 0.910 (64.6%) |

(IR, with the win rate in brackets; the current configuration is 1.243 at 77.8%.) Turnover rises
from ~430 round trips to 1100-1800 and both the return and the win rate fall: the 7.5% target needs
time to be reached, and paying fees three times as often destroys the edge. The corner is closed.

A side effect worth recording: the first attempt at this grid **was killed by the 3 GiB watchdog**
for eleven of the twelve cells, because high-turnover runs accumulate far more round-trip records
while the feature footprint had grown (the four new liquidity arrays plus the retained `high`
field). Trimming `V2_KEEP_ARRAYS` back to what the live path reads, and dropping `high` unless the
intraday trigger is enabled, brought every run back inside the budget.

### Board composition of the universe - the first positive sign in six rounds, but not established

The universe has never been varied by board. The strategy ranks main-board, ChiNext, STAR (科创板)
and Beijing names together even though they have different price limits and investor bases. On one
seed, dropping STAR and Beijing looked strong (IR 1.350 against 1.243). Replicated on nine seeds,
paired against the same nine models:

| universe | CAGR | sd | vol | max DD | win | IR |
|---|---:|---:|---:|---:|---:|---:|
| all boards (incumbent) | 13.76% | 2.71 | 14.58% | -23.18% | 75.39% | 0.945 |
| exclude STAR + Beijing | **14.72%** | 2.55 | 14.75% | -22.90% | **76.03%** | **0.999** |
| exclude ChiNext + Beijing | 12.47% | 3.09 | 14.44% | -23.74% | 74.55% | 0.863 |

Paired: `exclude STAR + Beijing` - incumbent = **+0.96pp of CAGR (t = +1.19, 6 of 9 seeds)**,
+0.65pp of win rate (t = +0.86, 7 of 9), and +0.054 of information ratio.

**This does not meet the measurement protocol**: the pre-registered bar is 8 of 9 seeds and an effect
larger than the seed-noise band (CAGR sd ~2.2pp), and the paired 95% interval for +0.96pp with a
2.4pp standard deviation spans roughly -0.9pp to +2.9pp. It is the closest thing to a positive that
six stage-1 rounds have produced, and it is recorded as a **lead, not a result**. Dropping ChiNext
instead is clearly negative (-1.28pp, t = -1.45), so the effect is not "smaller boards are worse" -
the STAR/Beijing exclusion removes a specific set of names.

### Stage-1 status after six rounds

Thirteen axes closed: third-opinion consensus, sub-book ensembling, the liquidity measure and window,
breadth, the intraday exit trigger, the absolute entry gate (a verified no-op), the ranking-triggered
exit, inverse-vol re-weighting, hard consensus, three forms of entry patience, the short-hold exit
corner, and - for the incumbent - two board variants. The seed-averaged information ratio is **0.945**
(return end) and **0.948** (compliant) against stage 1's 1.30, with the board lead at **0.999**.


## Stage-1 round 7: the board lead is falsified by more data

The STAR/Beijing exclusion was the only positive sign in six rounds (+0.96pp of CAGR, paired
t = +1.19 over nine seeds, 6 of 9 in favour). Six further model seeds were built, taking the paired
sample to fifteen, so that the effect could be either confirmed or killed.

| sample | n | control CAGR | exclude STAR+BSE | difference | t | seeds in favour |
|---|---:|---:|---:|---:|---:|---:|
| the original nine | 9 | 13.76% | 14.72% | +0.96pp | +1.19 | 6/9 (67%) |
| six new seeds | 6 | 13.96% | 14.34% | +0.38pp | +0.69 | **3/6 (50%)** |
| **pooled** | **15** | **13.84%** | **14.57%** | **+0.73pp** | **+1.40** | **9/15 (60%)** |

The effect **shrank on the fresh data** (+0.96pp to +0.38pp) and the sign is now a coin flip
(9 of 15). Win-rate difference: +0.43pp, t = +0.89, 10 of 15. This is what a noise-driven finding
looks like when it meets new data, and it is the protocol working as intended: the lead was recorded
as a lead, a bar was pre-registered, more data was gathered, and the lead did not survive.

Per-seed differences were mixed on both halves (the nine original seeds: +1.37, -1.49, +0.41, +2.14,
+2.46, +5.28, -2.42, +2.15, -1.27pp; the six new ones: +0.74, +1.30, -1.18, -0.10, +2.35, -0.84pp).

**Board composition is therefore closed as an axis**, and with it the last open lead. The
measurement protocol's headline requirement - at least 8 of 9 (equivalently 12 of 15) seeds in
favour and an effect beyond the 2.2pp seed-noise band - is not met by any mechanism tested in seven
stage-1 rounds.

### Final stage-1 position

Fourteen axes closed, each with a control that reproduced the incumbent bit for bit:
third-opinion consensus (13 partners), sub-book ensembling (511 runs; best combination +0.086 IR),
the liquidity measure and its window, portfolio breadth (twice, under two policies), the intraday
exit trigger, the absolute entry gate (a verified no-op), the ranking-triggered exit, inverse-vol
re-weighting, hard consensus, three forms of entry patience, the short-hold exit corner, and board
composition.

| objective | required | achieved |
|---|---|---|
| stage 1 seed-mean IR | >= 1.30 | **0.948** (compliant) / 0.945 (return end) |
| stage 2 | >= 1.60 | not reached |
| stage 3 (contract) | >= 2.50 | not reached |
| win rate >= 70% | >= 70% | **met**, 74.8-80.5%, 9/9 seeds |
| volatility <= 10% | <= 10% | **met**, 9.67%, 9/9 seeds |
| CAGR >= 25% at vol <= 10% | >= 25% | 9.16% |


## Stage-1 round 8: scale-out - the last exit design - rejected

Exits are the only place this project has ever found a gain, so the last untested exit design was
scale-out: bank part of the position at the first target and let the remainder run to a second one.
Implemented as `--scale-out <fraction> --profit-target-2 <level>`; the reduced target weight flows
through the existing weight-based order path, and a zero control reproduces the incumbent bit for
bit.

| variant | CAGR | vol | max DD | win | trades | IR |
|---|---:|---:|---:|---:|---:|---:|
| control (no scale-out) | **18.21%** | 14.65% | **-19.84%** | **77.83%** | 433 | **1.243** |
| scale 50%, second target 15% | 15.26% | 13.80% | -21.69% | 74.75% | 297 | 1.106 |
| scale 50%, second target 25% | 11.07% | 13.49% | -23.16% | 64.91% | 228 | 0.820 |
| scale 50%, second target 40% | 9.35% | 13.34% | -22.90% | 63.27% | 196 | 0.701 |
| scale 34%, second target 25% | 11.03% | 14.45% | -24.51% | 60.76% | 237 | 0.764 |
| scale 75%, second target 25% | 11.72% | 12.86% | -23.36% | 74.18% | 244 | 0.911 |

Every setting loses 3-9pp of CAGR and 3-15pp of win rate, and the trade count falls by a third to a
half because the remainder keeps running. Splitting the exit destroys the single-target edge: the
first target is where the strategy's return comes from, and taking only half of it at that point is
strictly worse than taking all of it.

### The mechanism space is exhausted

Fifteen axes have now been closed, each with a control that reproduced the incumbent bit for bit:

| layer | mechanisms tested and rejected |
|---|---|
| exit policy | profit target x hold grid (49 cells across two rounds, including the previously untested short-hold corner), stops, trailing stops, stale-loser exits, profit locks, intraday-touch trigger, ranking-triggered exit, **scale-out**, market-state exits |
| entry policy | blend weight (0.50 confirmed), hard consensus, 13 third-opinion partners, absolute quality gate (a verified no-op), three forms of relative patience, head re-ranking |
| portfolio | breadth (twice, under two policies), inverse-vol re-weighting, correlation-aware sizing, drawdown tiers, rolling-peak drawdown, risk-trim mode |
| universe | liquidity floor, board composition (replicated to fifteen seeds and falsified) |
| model | training length, objective (rank vs regression), label variants, seed ensembles |
| data | stock-level intraday unavailable for 2019-2026; margin data starts 2024-12; industry membership starts 2025-01 |

Across all of them the seed-averaged information ratio has stayed in the band **0.94-1.00**, against
stage 1's requirement of 1.30. The one intervention that ever moved it - the exit-policy redesign of
rounds 23-24 - is already in the incumbent.

