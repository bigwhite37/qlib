# V2 experiment log: every mechanism tested, and its verdict

All runs use the same Qlib account engine, the same DuckDB panel, the same fee
schedule, the same T+1 close execution and the same 3 GB memory cap. Each row
changes one thing against the headline; the numbers come from `output/v2_*/summary.json`
and the complete table is in `output/v2_research/account_runs.csv`.

Headline: `v2_g4` - CAGR +13.39%, volatility 11.82%, drawdown -15.00%, win rate 70.29%,
IR 1.13, 956 round trips, fees 5.05%/yr.

## 1. Entry signal

| mechanism | run | CAGR | win | verdict |
|---|---|---:|---:|---|
| rule composite only (blend 0.0) | `v2_rule0` | +4.71% | 65.7% | alone it is weak (IR 0.36) |
| model only (blend 1.0) | `v2_model1` | +5.16% | 66.6% | alone it is weak (IR 0.41) |
| **50/50 model + composite blend** | `v2_g4` | **+13.39%** | **70.3%** | **kept - nearly three times either component alone (IR 1.13)** |
| blend weight 0.4 / 0.6 | `v2_e3` / `v2_e5` | +7.71% / +4.96% | 66.8% / 67.7% | rejected |
| win-probability gate >= 0.50 / 0.55 | `v2_e1` / `v2_e2` | +6.56% / +2.57% | 65.1% / 65.9% | rejected; frequency collapses to 0 |
| forward-return ranking label | `v2m_tp5h40_predictions_fwd20` | +8.13% | 66.1% | rejected (alpha +0.37pp vs +0.44pp) |
| 16 extra microstructure features | `v2_x1`, `v2_x3` | +8.13% / +12.39% | 66.1% / 63.2% | rejected despite a higher label-level IC |
| composite rebuilt on parkinson / limit-up | `v2_k1c`, `v2_k3c`, `v2_k4c` | +8.23% / +8.51% / +6.80% | 66.4% / 66.2% / 62.3% | rejected |

## 2. Exit policy

| mechanism | run | CAGR | win | verdict |
|---|---|---:|---:|---|
| 2 ATR stop + trailing (design's V0 policy) | `p_stop_trail20` grid | -0.86%/trade | 38% | rejected |
| fixed 8% / 10% / 12% stop | `v2_r1` family, `v2_j1` | +10.37% at 12% | 67.1% | rejected at every level tested |
| market-extreme forced liquidation | `v2_h1` -> `v2_h5` | +11.88% -> +12.90% | 54.0% -> 61.6% | **removed** - it sells at the bottom |
| no profit target at all | `v2_t1`, `v2_t2` | +8.62% / +6.70% | 48.1% / 46.3% | rejected - the target creates the hit rate |
| profit target 2.5 / 3.0 / 3.5 / 4.0% | `v2_r1` / `v2_h2b` / `v2_g4` / `v2_h5b` | +10.60% / +12.67% / **+13.39%** / +11.63% | 70.3% / 70.4% / 70.3% / 69.2% | 3.5% kept |
| holding cap 30 / 45 / 60 / 90 days | `v2_j2` / `v2_g4` / `v2_h3b` / `v2_u1` | +11.33% / **+13.39%** / +8.90% / +7.13% | 64.6% / 70.3% / 71.9% / 68.2% | 45 days kept |
| stale-loser exit after 15 / 20 / 25 / 30 days | `v2_n1d`..`v2_n4d` | +10.64% / +9.14% / +9.26% / +12.12% | 57.3% / 60.6% / 62.3% / 64.7% | rejected - losers recover often enough |
| profit lock 1.0 / 1.5 / 2.0 / 3.0% | `v2_p1d`..`v2_p4d` | +11.30% / n-a / +12.24% / +7.55% | 65.6% / n-a / 63.7% / 58.8% | rejected - higher average win, lower hit rate |

## 3. Portfolio construction

| mechanism | run | CAGR | win | verdict |
|---|---|---:|---:|---|
| single-name weight cap 0.20 / 0.12 / 0.08 | `v2_r1` / `v2_f4` / `v2_g4` | +10.60% / +12.25% / **+13.39%** | 70.3% / 71.2% / 70.3% | **0.08 kept** - the largest single gain after round 3 |
| position budget 4 / 6 / 12 / 20 / 30 names | `v2_n3` / `v2_n1` / `v2_r1` / `v2_p3` / `v2_p2` | +5.50% / +2.93% / +10.60% / +6.85% / +5.69% | 74.8% / 70.3% / 70.3% / 71.3% / 64.2% | 12 names (the design's value) is empirically best |
| correlation filter 0.80 / 0.60 | `v2_f1`, `v2_f2` | +3.81% | 44.5% | rejected - an illiquidity signal is naturally correlated |
| correlation replacement | `v2_f3` | - | - | rejected |
| volatility target 0.08 / 0.10 / 0.14 / 0.16 | `v2_l3` / `v2_band_a` / `v2_g4` / `v2_j3` | +6.50% / +0.91% / **+13.39%** / +12.21% | 66.5% / 46.5% / 70.3% / 68.7% | 0.14 kept |
| drawdown tiers none / soft / hard | `v2_m3b` / `v2_g4` / `v2_h7` | +13.91% / +13.39% / +11.62% | 72.0% / 70.3% / 59.9% | soft tiers kept for the win rate |
| weak-market cap 35 / 50 / 70% | `v2_g4` / `v2_m1b` / `v2_m2b` | +13.39% / +14.53% / +12.44% | 70.3% / 72.8% / 70.5% | 35% kept (best IR 1.13) |
| market ladder removed | `v2_e4` | +11.31% | 70.7% | rejected - volatility 15.3%, drawdown -25.1% |
| daily new-position cap 1 / 2 / 3 | `v2_g4` / `v2_h4b` / `v2_n7` | +13.39% / +12.09% / +4.07% | 70.3% / 68.7% / 69.4% | 1/day kept |

## 4. Execution and accounting

| mechanism | run | CAGR | win | verdict |
|---|---|---:|---:|---|
| minimum trim size (skip partial trims under 2% of NAV) | `v2_e4` -> `v2_h1` | +10.32% -> +11.88% | 51.9% -> 54.0% | **kept** - 39% of all fees were 5-CNY-minimum shaves |
| whole-position risk trims instead of proportional | `v2_f1` (round 3) | +3.81% | 44.5% | rejected |
| no-trade bands on volatility / gross rebalancing | `v2_band_a` vs `v2_a` | +0.91% vs -0.97% | 46.5% vs 45.5% | **kept** |
| friction 5 / 10 / 20 bp per side | `v2_pr1` / `v2_g4` / `v2_pr2` | +9.80% / +10.60% / +7.69% (at the round-7 config) | 70.2% / 70.3% / 67.8% | reported as pressure |
| signal delayed one day | `v2_pr4` | +9.54% | 61.6% | reported as pressure |
| 1,000,000 CNY account | `v2_pr3` | +2.26% | 66.3% | reported as pressure - the 5bp participation cap starves the book |
| buy-price cap audit | `scripts/audit_price_cap.py` | - | - | 2 of 959 buy orders rejected: immaterial |

## 5. What the search establishes

1. **Loss cutting does not work here.** Four independent mechanisms (an 8%, 10% and
   12% price stop, and a stale-time exit at four horizons) all reduce both return and
   hit rate. The alpha is a reversal/illiquidity premium that recovers after drawdown.
2. **The profit target is the hit-rate engine, not a return cap.** Removing it drops
   the win rate to 46-48%; replacing it with a profit lock raises the average win but
   loses more hit rate than it gains in return.
3. **Portfolio construction mattered more than the signal.** The two largest gains
   after round 3 came from a minimum trim size (a fee leak) and from tightening the
   single-name weight cap from 20% to 8% (a risk-estimation leak), not from new alpha.
4. **Cross-sectional IC is not the account objective.** Features with the strongest
   forward-return IC (Parkinson range volatility, limit-up counts) make the account
   worse through both the model and the rule composite, because their bucket mean is
   driven by a right tail that the frozen exit policy does not capture.
5. **The residual gap is the information ratio.** 1.13 measured against 2.5 required
   for 25% CAGR inside a 10% volatility cap, and the drawdown-to-volatility ratio near
   1.3 means the -10% drawdown floor alone caps CAGR near 9%.

## 6. Ranker ablation (the strongest single result in the project)

| ranker | CAGR | volatility | max DD | IR | win |
|---|---:|---:|---:|---:|---:|
| rule composite only | +4.71% | 13.16% | -24.62% | 0.36 | 65.71% |
| model only | +5.16% | 12.47% | -24.86% | 0.41 | 66.63% |
| **50/50 blend** | **+13.39%** | **11.82%** | **-15.00%** | **1.13** | **70.29%** |

With every other setting identical, combining the two rankers roughly triples the
CAGR of either one and cuts its drawdown by 10 points. The two signals are not
substitutes: the rolling model learns short-horizon patterns from 53 features while
the composite encodes a slow illiquidity/volatility/reversal tilt, and the names
they agree on behave differently from the names either prefers alone. This also
explains why every attempt to *replace* the composite with a better-IC feature set
(rounds 6 and 9) failed while the blend itself kept improving.

## 7. Round-12 additions (all rejected)

| mechanism | run | result | verdict |
|---|---|---|---|
| rank ensemble with a second model (forward-return label) | `v2_r4e` (0.25/0.25 blend) | CAGR +10.00% vs +13.39% | rejected: the second model's own alpha is lower (+0.37pp vs +0.44pp), so it dilutes the blend; it also doubles the prediction-matrix memory |
| composite component weights (low-vol 2x, illiquidity 2x, half reversal, half illiquidity) | `v2_w1e`..`v2_w4e` | +5.40% / +9.85% / +7.84% / +10.36% against +13.39% for equal weights | rejected: equal weighting of the three components is a clear local optimum |
| 60-month training window instead of the design's 36 | `pred_t60.log` | top-decile alpha +0.32pp vs +0.44pp | rejected: the longer window is worse, not better |

With these, every axis that can be varied without changing the data itself has been
swept: entry signal, ranking label, model features, training window, model ensemble,
blend weights, composite definition and component weights, exit rules (stop, trailing,
target, lock, stale, time cap, extreme liquidation), position budget, single-name weight
cap, correlation filter and replacement, daily entry cap, volatility target, drawdown
tiers, market-ladder caps, the win-probability gate, trim sizing, and the execution
wrapper (friction, lag, account size). The headline `v2_g4` is the best of 100 recorded runs.


## Round-13 findings

| change | evidence | effect |
|---|---|---|
| V2 base-universe liquidity floor 0.02 | `v2_u6f` | CAGR +8.23%, IR 0.72 against +13.39% / 1.13 at the design's 0.20 floor |
| V2 base-universe liquidity floor 0.35 | `v2_u1f` | CAGR +4.55%, IR 0.37: filtering out the less liquid part of the market removes most of the alpha |
| V2 base-universe liquidity floor 0.50 / 0.65 / 0.10 | `v2_u2f`, `v2_u3f`, `v2_u5f` | watchdog aborts; the surviving points bracket the optimum |

The design's 0.20 floor sits at a local optimum: both a looser and a stricter liquidity
requirement are worse. Together with rounds 6 and 9 this closes the "is the alpha just the
illiquidity tail?" question - it is, and the tail cannot be filtered away without losing the
return.

## Standing conclusion after thirteen rounds

The contract asks for CAGR >= 25% with volatility <= 10%, drawdown >= -10% and a 70% win
rate simultaneously. The win rate is met (70.29%) and both holding-frequency floors are met,
but the return requirement needs an information ratio of about 2.5 and the best value
measured across 100+ runs and every axis is 1.13. The drawdown-to-volatility ratio of 1.27
means the -10% drawdown floor alone caps CAGR near 9% at that IR.

Progress across the project: CAGR -4.16% -> +13.39%, win rate 32.0% -> 70.29%, IR 0.15 ->
1.13. Almost all of it came from fixing execution and portfolio-construction leaks rather
than from finding more alpha: the forced liquidation at the market extreme, the minimum-
commission trim churn, the bypassed position budget, and the 20%-to-8% single-name weight
cap. The last untested assumption - that a stronger cross-sectional signal would help - was
falsified three times (rounds 6, 9 and 12).


## 8. Round-14 additions: risk-shape overlays (all rejected)

| mechanism | run | result | verdict |
|---|---|---|---|
| portfolio volatility window 60 -> 20 days | `v2_v1g` | CAGR +12.97%, drawdown -18.06%, IR 1.09 | rejected: faster reaction makes the drawdown worse |
| market-volatility exposure brake, 25% / 20% budget | `v2_v3g`, `v2_v4g` | CAGR +9.90% / +9.95%; drawdown -15.10% / -16.82% | rejected: exposure fell (mean 0.91 / 0.85, minimum 0.24 / 0.19) without reducing the drawdown |

### The drawdown ratio is structural

Across the 109 recorded runs with 300+ round trips, the drawdown-to-volatility ratio has a
mean of 1.70 and a median of 1.61, ranging from 1.26 to 3.07. The headline `v2_g4` holds
the **lowest ratio measured (1.27)**, and CAGR correlates -0.74 with the ratio, i.e. the
configurations with the smoothest drawdown shape earn the least. The -10% drawdown floor
therefore implies realized volatility near 7.9% and a CAGR near 8.9% no matter which overlay
is used - the constraint cannot be engineered away, only paid for in return.


## 9. Final position after sixteen rounds

| requirement | best achieved | configuration | status |
|---|---|---|---|
| CAGR >= 25% | **+14.53%** | `v2_m1b` | not reachable: needs an information ratio of 2.5; best measured 1.13 |
| volatility <= 10% | **9.83%** | `v2_aa3` | met |
| drawdown >= -10% | **-14.86%** | `v2_y4` | not reachable: the floor implies about 7.9% volatility and about 6% CAGR at the measured drawdown ratio |
| win rate >= 70% | **72.75%** | `v2_m1b` | met |
| every complete quarter positive | 10-12 negative of 30 | all runs | not reachable with this alpha |
| holding frequency (252d / 63d) | **97.6% / 90.5%** | `v2_g4` | met |

Two points summarise the frontier: **`v2_g4`** maximises return (+13.39% CAGR, 11.82%
volatility, -15.00% drawdown, 70.29% win, IR 1.13) and **`v2_y4`** maximises contract
compliance (+7.44% CAGR, 9.88% volatility, -14.86% drawdown, 70.66% win, four of seven checks).



## 10. Round-18 additions: the risk layer, one axis at a time

All runs are `v2_g4` with a single axis changed (control `v2_g4b` reproduced the headline exactly).

| mechanism | run | result | verdict |
|---|---|---|---|
| correlation-aware *sizing* (mild: strength 0.5, base 0.30, floor 0.50) | `v2_cs1` | CAGR +13.29%, vol 11.64%, DD -16.59%, win 70.51%, IR 1.141 | rejected: +0.008 IR is inside the path-noise band |
| correlation-aware sizing (strength 1.0, floor 0.25) | `v2_cs2` | CAGR +11.81%, vol 11.11%, DD -13.70%, win 70.84%, IR 1.062 | rejected: pays 1.6pp of return for 0.7pp of volatility |
| correlation-aware sizing (base 0.00, floor 0.50) | `v2_cs3` | CAGR +11.59%, vol 10.72%, DD -14.03%, win 70.95%, IR 1.081 | rejected: same trade, worse |
| portfolio breadth 12 -> 16 / 20 / 24 names | `v2_bp16`, `v2_bp20`, `v2_bp24` | IR 1.008 / 0.802 / 0.886; win 67% / 67% / 70% | rejected, and the breadth hypothesis is falsified: IR **falls** monotonically, so the alpha lives in the top few ranks and the next-best names are noise |
| breadth + faster fill (max-new 2 / 3) | `v2_bp16n2`, `v2_bp20n2`, `v2_bp20n3` | IR 0.970 / 0.808 / 0.607; win 67.7% / 67.0% / 62.4% | rejected: filling the extra slots faster makes it worse again |
| portfolio volatility target switched off | `v2_vt0` | CAGR +13.09%, vol 12.16%, DD -17.84%, IR 1.077 | kept: the target is worth +0.056 IR and 2.8pp of drawdown |
| volatility target 14% -> 20% | `v2_vt20` | CAGR +13.63%, vol 11.96%, DD -16.28%, win 69.80%, IR 1.139 | neutral: the account is not leverage-limited by the 14% setting |
| market ladder switched off | `v2_notm` | CAGR +14.60%, vol 15.30%, DD **-27.29%**, win **73.26%**, IR 0.955 | kept ON: worth +0.18 IR and 12pp of drawdown for 1.2pp of return |

The breadth result is the most useful negative result of the round. It removes the last
plausible "free" route to a higher information ratio - buying more of the ranking - and it
explains the path sensitivity visible elsewhere in the search: if only the top two or three
names matter, then any change that reorders the top of the list moves the whole account.



## 11. Round-19 additions: new factors, and the rule that replaces IC screening

The V2 feature library never contained the textbook low-risk or lottery anomalies. Round 19 adds
them (`scripts/research_r19_factors.py`) and tests each one at the account level by replacing
`vol20` inside the winning composite (`scripts/sweep_r19_factors.sh`).

| mechanism | run | result | verdict |
|---|---|---|---|
| idiosyncratic volatility (`ivol60`), IC20 **-0.107** (t -34.9), the highest IC measured in this project | `v2_r19ivol` | CAGR +6.83%, vol 12.06%, DD -24.40%, win 66.45%, IR 0.567 | **rejected**: the strongest feature in the project halves the return |
| market beta (`beta60`), IC20 -0.010 | `v2_r19beta` | CAGR +4.62%, IR 0.396 | rejected |
| downside semi-deviation (`semidev20`), IC20 -0.052 | `v2_r19semi` | CAGR +8.02%, IR 0.690 | rejected |
| skewness (`skew60`), IC20 -0.034 | `v2_r19skew` | CAGR +9.83%, IR 0.703 | rejected |
| own drawdown from the 252d high (`dd252`), IC20 -0.018 | `v2_r19dd` | CAGR +12.78%, vol 15.64%, **win 72.33%**, IR 0.817 | rejected: the best win rate of the round, at 3.8pp more volatility |
| maximum daily return (`maxret20`), IC20 -0.091 | `v2_r19max` | CAGR +8.81%, IR 0.720 | rejected |
| share of up days (`pos_days20`), IC20 -0.044 | `v2_r19pos` | CAGR +11.32%, IR 0.828 | rejected |
| `combo3` **plus** beta60 and ivol60 | `v2_r19mix` | CAGR +4.70%, vol 9.88%, IR 0.476 | rejected |

The IC scan also produced, for the record, the audit table that rounds 6, 9, 12, 17 and 19 have
all failed to convert: `maxret20` -0.091 (t -34.0), `gap_abs20` -0.089, `mom20` -0.080,
`dist_ma120` -0.081, `dn_up_vol20` **+0.069** (the only large positive sign: names whose upside
volatility is large relative to their downside volatility do *worse*), `dd60` -0.029,
`kurt60` +0.009, `ac1_20` +0.002, `mom252_skip20` -0.004.

### The screening rule this establishes

Five independent times now, a feature with a *higher* cross-sectional IC has made the account
worse - most decisively here, where the highest-IC feature in the project halves the CAGR. The
mechanism is visible in the round-18 breadth result: IC averages over a cross-section of up to
3000 names, while the account holds the one or two names at the very top of the blended list. A
factor can have a huge average IC and still put the wrong names at the head.

**An IC scan is a filter, never a selection criterion. Only the account test decides.**


## 12. Round-18c/d additions: how much of the result is the risk layer, and how much is path

| mechanism | run | result | verdict |
|---|---|---|---|
| constant exposure at the ladder's own mean (0.587) | `v2_lflat58` | CAGR +10.72%, vol 13.04%, DD -25.94%, win 64.15%, IR 0.822 | the ladder is worth **+2.7pp CAGR, -1.2pp vol, +10.9pp drawdown, +0.31 IR** over the same average exposure |
| flatter ladder (0.50/0.62/0.66) | `v2_lsoft` | CAGR +13.35%, vol 12.49%, DD -21.36%, IR 1.069 | rejected: same return, worse risk |
| amplified ladder (0.25/0.72/0.90) | `v2_lamp` | CAGR +11.27%, vol 10.84%, DD -13.16%, IR 1.039 | rejected, but the best volatility-compliant point found by re-shaping (9.93% vol at `v2_lamp2`) |
| inverted ladder (0.85/0.60/0.35) | `v2_linv` | CAGR +12.09%, vol 12.74%, DD -24.80%, IR 0.949 | rejected: the *ordering* carries information, the design's ordering is the right one |
| ten 1-2% single-parameter perturbations | `v2_j_*` | mean 13.04%, **sd 0.41pp**; three runs bit-identical to the headline | measured local noise floor; differences below ~1.2pp are not evidence |

Three conclusions close the round:

1. **The risk layer is the productive layer.** The market ladder is the only component measured
   in this project whose removal costs more than the noise band *and* whose every perturbation is
   worse than the design. Constant exposure, a flatter ladder, a more aggressive ladder and an
   inverted ladder all lose; the design's 0.35/0.70/0.90 sits at a local optimum.
2. **The signal layer is exhausted for this data set.** Five IC-driven rounds and every exit
   policy variant have been rejected; the ranker blend of a rolling LightGBM with the causal
   `combo3` rule remains the single largest effect in the project and nothing added to it has
   survived.
3. **At the contract's 10% volatility ceiling the strategy earns about 7.4% a year.** Two
   independent mechanisms - scaling every gross cap by 0.80 (`v2_y4`, 9.88% vol, +7.44%) and
   re-shaping the ladder (`v2_lamp2`, 9.93% vol, +7.37%) - agree to within 0.1pp. The contract
   asks for 25%.



## 13. Round-20 additions: head re-ranking (rejected)

Rounds 18 and 19 both point at the head of the ranking: the account buys only the top one or two
names, and the composite matters through what it puts there. Round 20 therefore added a second
ranking stage over the top K candidates only (`--head-rerank K --head-key <feature> --head-sign`),
choosing the day's single buy on one transparent dimension.

| mechanism | run | result | verdict |
|---|---|---|---|
| re-rank top 5 by lowest `vol60` | `v2_hr5_vol` | CAGR +12.33%, vol 12.49%, **win 71.12%**, IR 0.987 | rejected |
| re-rank top 3 by lowest `amount20_yuan` | `v2_hr3_amt` | CAGR +12.23%, IR 1.034 | rejected |
| re-rank top 5 by lowest `amount20_yuan` | `v2_hr5_amt` | CAGR +10.79%, IR 0.905 | rejected |
| re-rank top 3 by lowest `vol60` | `v2_hr3_vol` | CAGR +10.59%, IR 0.862 | rejected |
| re-rank top 5 / top 3 by lowest `ma60` | `v2_hr5_ma60`, `v2_hr3_ma60` | CAGR +10.27% / +9.44% | rejected |
| re-rank top 3 by lowest `ret1` | `v2_hr3_ret` | CAGR +9.07%, IR 0.754 | rejected |
| re-rank top 3 by lowest `atr20` | `v2_hr3_atr` | CAGR +8.82%, IR 0.729 | rejected |

Every variant loses 1.1-4.6pp of CAGR against the untouched ordering, and the most on-theme
variant - "among the top three, take the least volatile" - is among the worst. The 50/50 blend
therefore produces a head ordering that none of its own components can reproduce; the blend is
the mechanism, not a smoothed version of its inputs.

**Verified after every round-18/19/20 code change**: `v2_g4c3` reproduces the headline exactly
(CAGR 0.13393459533504304, vol 0.1181851719289933, drawdown -0.14995347183337682, win
0.702928870292887), and `python sleeve/a/scripts/pyrun.py sleeve/a/scripts/run_tests.py` passes
27 tests.



## 14. Round-21 additions: the model is deliberately small, and the seed is not a detail

The rolling return model early-stops after a median of 7 boosting rounds out of 300 (and after
1 round in 9 of 31 quarters). Round 21 treated that as a defect and made the training length and
objective explicit (`--objective {mse,lambdarank}`, `--fixed-rounds N`, `--num-leaves`,
`--learning-rate`, `--min-data-in-leaf`).

| mechanism | run | result | verdict |
|---|---|---|---|
| `mse`, fixed 400 rounds | `v2_m21_mse400` | decile spread 0.0188 (best of all models), CAGR +8.01%, IR 0.680 | rejected: the best model makes one of the worst accounts |
| `mse`, fixed 150 rounds | `v2_m21_mse150` | spread 0.0177, CAGR +9.46%, IR 0.784 | rejected |
| `mse`, fixed 50 / 25 / 10 rounds | `v2_m21_r50` / `_r25` / `_r10` | CAGR +11.76% / +10.52% / +9.30% | rejected: no fixed length in either direction beats the adaptive stop |
| `lambdarank` (cross-sectional relevance buckets), 150 rounds | `v2_m21_ll150` | spread 0.0026, CAGR +10.13% | rejected: the ranking objective is the worst model of the four |
| validation window 3 / 12 months instead of 6 | `v2_m21_valid3` / `_valid12` | CAGR +10.05% / +11.01% | rejected |
| three-seed rank ensemble | `v2_m21_seed3` | CAGR +10.12%, IR 0.848 | rejected: lands on the seed mean |
| nine model seeds through the `v2_g4` configuration | `v2_sd_g4_*` | **CAGR mean 9.93%, sd 2.22pp**, range 6.94-13.39%; vol mean 11.84%, **sd 0.22pp**; win mean 68.92%, sd 1.56pp | the headline is the maximum of nine draws |
| seven model seeds through the `v2_y4` compliance configuration | `v2_sd_y4_*` | CAGR mean 7.40%, sd 2.18pp; vol mean 10.19%, sd 0.28pp; win mean 71.25%, sd 1.43pp | only 2 of 7 seeds actually clear the 10% volatility ceiling |

Two conclusions:

1. **"The model is undertrained" was the wrong diagnosis.** No fixed training length beats the
   adaptive per-quarter early stop, in either direction, and the model's own decile spread rises
   monotonically (0.0132 -> 0.0142 -> 0.0151 -> 0.0160 -> 0.0177 -> 0.0188) while the account does
   not follow it. This is the sixth falsification of "a better ranking improves the account", and
   the first produced by intervening on the model itself.
2. **The strategy's expected performance is materially below its headline.** Nine seeds of an
   identical pipeline span 6.94% to 13.39% CAGR with a standard deviation of 2.22pp, while realised
   volatility moves by only 0.22pp. The honest expectation for the `v2_g4` configuration is
   **9.9% CAGR at 11.8% volatility (IR 0.84)**, and for a reliably sub-10%-volatility configuration
   about **7%**. Every "best run" quoted anywhere in this project should be read as the maximum of
   a distribution with a 2.2pp standard deviation.



## 15. Round-22 additions: the market ladder is removed from the headline

Round 21 measured the seed noise (2.2pp of CAGR) and showed that single-run comparisons in this
project were not decisive. Round 22 therefore re-ran the headline alternatives on **all nine model
seeds** and compared them **paired** - same nine models, one flag different - which cancels the
seed noise instead of averaging over it.

| mechanism | seed-mean CAGR | vol | max DD | win | IR | verdict |
|---|---:|---:|---:|---:|---:|---|
| design ladder 0.35/0.70/0.90 (`v2_g4`) | 9.93% | 11.84% | -17.90% | 68.92% | 0.844 | superseded |
| **ladder off (`v2_notm`)** | **12.12%** | 15.00% | -26.57% | **72.28%** | 0.806 | **adopted as the return/win headline** |
| ladder off, wider drawdown tiers (`v2_sd2_f_*`) | **12.39%** | 15.75% | -28.29% | **72.57%** | 0.784 | best CAGR and win rate measured |
| ladder off, tighter drawdown tiers (`v2_sd2_g_*`) | 11.69% | 13.82% | -23.81% | 70.55% | 0.838 | the balanced ladder-off point |
| ladder off, gross cap 0.62 (`v2_sd2_h_*`) | 11.79% | 13.28% | -25.61% | 66.07% | **0.888** | best IR in the project, win rate fails |
| headline without the drawdown tiers (`v2_m1b`) | 9.79% | 12.02% | -18.55% | 68.84% | 0.815 | rejected |
| correlation-aware sizing (`v2_cs1`) | 9.79% | 11.73% | -17.86% | 69.38% | 0.834 | rejected |
| compliance cap-scale 0.80 (`v2_y4`) | 7.26% | 10.13% | -15.59% | 71.45% | 0.715 | kept as the compliance point |

Paired difference, ladder off minus design ladder, over the same nine models:

| paired difference | mean | sd | t | seeds in favour |
|---|---:|---:|---:|---:|
| CAGR | **+2.19pp** | 1.58pp | **4.16** | 8 of 9 |
| win rate | **+3.36pp** | 1.05pp | **9.63** | **9 of 9** |

This is the largest and most significant effect measured in this project, and the first one that
moves **both** objectives in the contract's first sentence - CAGR and win rate - in the right
direction at the same time measured across seeds rather than on one lucky path.

The interpretation is that the V2 market ladder is a leftover from the V0/V1 trend design. For a
low-volatility, below-the-moving-average mean-reversion book, the "weak market" state (36.7% of
all days) is where the edge is, and the ladder systematically under-invests exactly there. Its
round-18 appearance of being locally optimal was an artefact of only ever testing cap vectors that
kept the weak-market de-risking in some form.

**Cost of the change**: volatility 11.84% -> 15.00% and drawdown -17.90% -> -26.57%. The ladder is
therefore kept available (`--use-market-timing`, `--state-caps`) and the compliance
configuration still uses it; what changed is which configuration is called the headline.



## 16. Round-23 additions: the hold cap was costing 8 points of win rate

Round 22 delivered the ladder-off headline; round 23 asked whether that gain survives being scaled
to the contract's 10% volatility ceiling, and re-tuned the exit policy at a fixed risk budget.

| mechanism | seed-mean result | verdict |
|---|---|---|
| ladder off, volatility target lowered to reach the 10% ceiling | at vt 0.055: 7.36% CAGR, 9.97% vol, 69.85% win | the round-22 gain was **exposure, not alpha**: at matched risk the ladder-on and ladder-off mechanisms tie |
| **hold cap 45 -> 120 days, same exposure** | win rate **+8.26pp** (t = **27.76**, 9 of 9 seeds), CAGR -0.90pp (t = -1.12, not significant) | **adopted**: the largest and most significant effect in the project, and it costs nothing measurable in return |
| hold cap 45 -> 90 days | win rate +4.20pp (t = 9.93, 9 of 9) | also works, less so |
| profit target 0.02 / 0.025 / 0.045 / 0.06 at the compliance exposure | CAGR 4.67% / 6.50% / 8.04% / 6.58%; win 69.8% / 71.2% / 67.9% / 62.3% | rejected: the 3.5% target remains best |
| ladder off + 120-day hold at three volatility targets | vt 0.14: 11.22% CAGR / 14.51% vol / **80.54%** win; vt 0.09: 9.69% / 12.79% / 80.06%; vt 0.055: 6.62% / **9.65%** / 78.22% | the new frontier |
| `h120` (vt 0.055, hold 120) against `v2_y4` | win **+6.68pp** (t = 18.35, 9 of 9), CAGR -0.62pp (not significant), volatility -0.52pp, drawdown unchanged | **`h120` replaces `v2_y4` as the compliance configuration** |

The mechanism behind the hold result is the frozen policy itself: a low-volatility name that has
not reached the 3.5% profit target is force-closed at 45 days whether or not the thesis played out,
and those forced exits are the losing round trips. Holding to 120 days (average realised hold rises
from 20.7 to about 28 days) lets the target be reached instead. Note that this is the *opposite* of
what the earlier y-family sweep suggested - that sweep changed the hold cap while the market ladder
was still cutting exposure in weak markets, which confounded the two effects.

**Reliability, not just level.** The two shape checks are now met on every seed rather than on a
lucky one:

| check | `v2_y4` (9 seeds) | `h120` (9 seeds) |
|---|---|---|
| volatility <= 10% | 3 of 9 | **9 of 9** |
| win rate >= 70% | 7 of 9 | **9 of 9** |
| mean CAGR | 7.24% | 6.62% (difference not significant) |
| mean win rate | 71.54% | **78.22%** |



## 17. Round-24 additions: the profit target was the last big mistake

Round 23 found the hold cap; round 24 searched the exit policy jointly (profit target x hold cap)
with the risk budget pinned at the contract's ceiling, then validated the optima paired on all nine
models.

| mechanism | seed-mean result | verdict |
|---|---|---|
| profit target 3.5% -> **7.5%**, hold 120 | **+2.54pp CAGR** (t = 5.66, 9 of 9) for -3.45pp win rate (still 74.8%, 9 of 9 seeds over 70%) | **adopted as the headline** |
| profit target -> 9.0%, hold 90 | +2.96pp CAGR (t = 7.51, 9 of 9) but win rate falls to 68.3% and fails 6 of 9 seeds | rejected: breaks the win-rate check |
| profit target -> 6.0%, hold 120 | +1.38pp CAGR, win 76.4%, 9 of 9 on both checks | kept as the conservative variant |
| profit target -> 12% / 15% | CAGR flat, win 62-70% | rejected |
| profit target removed entirely | IR collapses to 0.35, win 53-63% | rejected |
| hold cap 200 days | CAGR below hold 120 at every target | rejected |

`c2` (profit target 7.5%, hold 120, market ladder off, volatility target 0.055) is the new
headline for the compliant objective, and it **dominates the previous headline `v2_g4` paired on
the same nine models**:

| paired difference (`c2` - `v2_g4`) | mean | t | seeds in favour |
|---|---:|---:|---:|
| CAGR | -0.76pp | -0.96 | 4 of 9 (not significant) |
| win rate | **+5.85pp** | **10.30** | 9 of 9 |
| volatility | **-2.17pp** | - | 9 of 9 lower |
| maximum drawdown | **+1.98pp** | - | 9 of 9 better |

with 4 of 7 contract checks passed on **every** seed, against 3 of 7 on 4 of 9 seeds for `v2_g4`.
Against the previous compliance point `v2_y4`: CAGR +1.93pp (t = 2.65), win rate +3.22pp
(t = 8.43), volatility -0.49pp.

The lesson is the same one round 23 taught, one level down: the exit policy was set in round 2 on a
much weaker signal, and every later round re-tuned the *entry* while leaving the exit fixed. Two
frozen constants - a 3.5% profit target and a 45-day hold cap - were together costing about
8.4pp of CAGR and 11pp of win rate at the contract's risk budget.



## 18. Round-25 additions: the new headline survives every axis that was re-opened

Rounds 23-24 replaced two constants (3.5% profit target, 45-day hold) that everything else had been
tuned around, so the inherited parameters had to be re-tested. Round 25 re-opened three.

| mechanism | seed / panel | result | verdict |
|---|---|---|---|
| ATR stops 5-20%, trailing stop, stale-loser exits, profit lock | one model | every variant cuts CAGR by 1.2-7.5pp, cuts IR, and **cuts the win rate** (73.5% -> 53.7% at the tightest) | **rejected again**: the round 2-4 verdict carries over to the new policy, and the reason is now visible - a stop converts recoverable trades into realised losses |
| model/rule blend 0.35 and 0.65 | nine seeds, paired | 0.35: CAGR -0.29pp (t = -0.85), win +0.32pp (t = 0.65); 0.65: CAGR **-0.91pp** (t = -2.08), win **-1.11pp** (t = -2.10) | **0.50 confirmed**: the a-priori 50/50 blend remains optimal |
| account drawdown tiers | one model | 0.10:0.7,0.15:0.4 and 0.08:0.6,0.13:0.3 reproduce `c2` **bit for bit** (the volatility target already holds gross near 0.5, so the caps never bind); 0.06:0.6,0.10:0.35 costs 2.4pp of CAGR without improving the drawdown | kept as is |
| `--risk-trim-mode close_weakest` | one model | CAGR -0.10%, win 47.07%, drawdown -19.86% | rejected outright |

The configuration `c2` is therefore a local optimum on every axis re-opened under the new policy.
Combined with rounds 19-24 this closes the search: the entry signal (five falsifications), the exit
policy (now optimised properly), the portfolio shape, the risk layer, the model, and the data
(intraday and non-price sources are unavailable for the backtest window).



## 19. Round-26 additions: design changes, and the frontier moves up

The task owner authorised changing the strategy's design as long as the available data fields stay
the same, so round 26 tested a real change to the exit semantic and then applied the round-23/24
exit finding to the return end.

| mechanism | panel | result | verdict |
|---|---|---|---|
| **design change**: profit target expressed in multiples of the position's own ATR at entry, clamped to 2-30% of entry price | one model, plus a zero multiple control | flat 7.5% gives 10.76% CAGR / IR 1.101 / 73.5% win; 3x ATR 9.21% / 0.967 / 73.0%; 4x ATR 9.24% / 0.928 / 67.6%; 2x ATR 7.10% / 0.721 / 74.5%; 6-8x ATR worse still | **rejected**: a flat percentage beats a risk-consistent target on every axis - scaling by the name's own volatility makes the quiet names (which this strategy selects) exit at tiny gains, raising fee drag without improving the hit rate |
| the new exit policy (target 7.5%, hold 120) at the **return end** (`z1`, vt 0.14) | nine seeds, paired | **13.76% CAGR** +/- 2.71, 14.58% vol, -23.18% drawdown, **75.39% win**, IR 0.945 | **adopted as the return headline** |
| the same at target 9% / hold 90 (`z2`) | nine seeds | 14.11% CAGR but win rate 66.94%, only 2 of 9 seeds over 70% | rejected: breaks the win-rate check |

`z1` against the old headline `v2_g4`, paired on the same nine models: **CAGR +3.83pp (t = 3.83,
8 of 9 seeds)** and **win rate +6.47pp (t = 8.07, 9 of 9)**, at 2.73pp more volatility and 5.28pp
more drawdown. Against `notm`, the round-22 return headline, `z1` dominates: CAGR +1.64pp
(t = 1.59, not significant), win rate **+3.11pp (t = 4.26, 9 of 9)**, volatility -0.43pp, drawdown
**+3.39pp**.

The frontier is now two points with the same information ratio of about 0.95:

| configuration | CAGR | vol | max DD | win | IR |
|---|---:|---:|---:|---:|---:|
| `c2` ladder off, vt 0.055, tp 7.5%, hold 120 | 9.16% | **9.67%** | **-15.92%** | 74.76% | 0.948 |
| `z1` ladder off, vt 0.14, tp 7.5%, hold 120 | **13.76%** | 14.58% | -23.18% | **75.39%** | 0.945 |
| `v2_g4` (headline rounds 8-21) | 9.93% | 11.84% | -17.90% | 68.92% | 0.844 |

The redesign therefore lifted the whole frontier rather than tilting it: the same risk budget now
buys about 1.8pp more CAGR and 6pp more win rate than it did before rounds 23-26.

One process note worth recording: the first ATR implementation was a silent no-op because the live
exit path is `V2Strategy.evaluate_exits`, not the older `portfolio.evaluate_exits` that the flag was
added to - every run reproduced `c2` bit for bit. It was caught by a control run. A zero-multiple
control is now part of the exit-policy sweeps.



## 20. Stage-1 round 1: corrected ablation, and two negative extensions of the consensus

| mechanism | sample | result | verdict |
|---|---|---|---|
| **correction**: `--rank-blend 0.0` skipped the blend branch, so the historical "composite-only" arm (IR 0.36) was actually the raw model score | one seed | true composite-only IR **0.903-0.941**, model-only 0.711-0.740, blend 1.243-1.295 | the documentation was wrong; `--composite-only` now exists |
| rank consensus vs return blend of the same two arms | one seed | composite 0.941, model 0.740, 50/50 **return** blend 0.935, 50/50 **rank** consensus **1.295** | the consensus is a **selection** effect, not portfolio diversification |
| third-opinion consensus, 13 partners (rule factors + alternative-label models + extra seeds) at 20-33% weight | one seed | every partner loses 2.9-10.6pp of CAGR; best third opinion IR 1.000 against the control 1.243 | rejected: two complementary opinions are the optimum |
| sub-book ensembling over the whole corpus | 511 runs, all pairs | median pairwise correlation 0.836; best pair IR 1.0735 vs best member 0.988 (**+0.086**); reaching 2.5x would need rho <= 0.16 | rejected: the ensemble route is closed by the correlation structure |

The corrected picture is that the project's one large effect - the rank consensus - is a **selection
primitive** (buy what two complementary rankers both like), not a diversification primitive. The
composite and the model are complementary at a return correlation of 0.67; the search for a third
arm should look for something complementary to *both* in the same way, and nothing in this data set
is.



## 21. Stage-1 round 2: the engine is illiquidity, not low volatility

The composite is the strategy (the model adds ~5% of IR and the composite-only arm is
seed-independent), so the composite was attributed. Every composite was given **nine replications**
- composite at weight 0.8, model at 0.2, so each cell has nine paths - because the first
single-path attribution produced information ratios from 0.087 to 0.969 with pairs far worse than
either single component, which is the signature of path noise rather than of signal. Replication
overturned two of its conclusions, including a tempting "solo amount beats combo3".

| mechanism | sample | result | verdict |
|---|---|---|---|
| component attribution: amount20 only / vol20 only / dist_MA60 only / the three pairs / combo3 | nine seeds each | every composite containing amount20 is tied at the top (IR 0.91-0.98); removing amount20 costs **4.6-9.2pp of CAGR on 0 of 9 seeds** (paired t -5.2 to -13.2); `vol20` alone gives IR 0.45 and 62% win | **the engine is an illiquidity/neglect premium**; `combo3`, `solo_amt` and `pair_amt_ma` are statistically tied (t 0.33-0.39) |
| the liquidity measure itself: 5-day, 20-day, 60-day turnover, 5/20 acceleration ratio, Amihud | nine seeds (five cells) / one seed (two) | 5/20/60 are tied paired (`amount5 - amount20` +0.05pp CAGR, t = 0.07); the 5-day variant's higher IR (1.091 vs 0.983) is a volatility artefact - it runs at 13.69% instead of 15.06% | the window does not matter; the acceleration ratio (0.663) and Amihud (0.156) are far worse |

Two consequences:

1. **The project's framing was wrong.** This has been described as a "low-volatility mean-reversion"
   book since round 1. The attribution says the low-volatility component contributes approximately
   nothing: it is an **illiquidity/neglect** strategy, which is also why its capacity is so small
   (the 1M-account run starves) and why the illiquidity tilt was worth more than half the return
   when the liquidity floor was raised.
2. **Stage 1 is not reached and the engine has no obvious lever left.** The best replicated
   information ratio is **0.983** (`solo_amt`, nine seeds) against the incumbent 0.976
   (`combo3`); stage 1 needs 1.30. Window and companion component are both closed.

Control check: after adding the new liquidity features to the builder the headline configuration
reproduces bit-identically (CAGR 0.18208367646362533, vol 0.14648497961999804, drawdown
-0.19836662929520532, win 0.7782909930715936, 433 round trips), and the 27 tests pass.



## 22. Stage-1 round 3: slot-constrained, and four more axes close

The constraint diagnostic is the useful result of the round: the book holds **11.73 of its 12
slots** while **3044 candidates pass the gate every day**, so the strategy is **slot-constrained,
not candidate-constrained**. A new position is bought whenever a slot frees, whatever the quality of
that day's best candidate. That framed three of the four tests.

| mechanism | sample | result | verdict |
|---|---|---|---|
| breadth, retested under the new policy (20 and 30 names at matched gross capacity) | nine seeds, paired | 20 names +0.69pp CAGR (t = 0.58, 4/7) = tie; 30 names **-3.26pp** (t = -3.54, 1/9) | closed again: the ranking's information is exhausted after ~12 names |
| profit target triggered on an intraday touch of the high (`--profit-trigger-high`) | one seed plus a zero control | 13.03% CAGR / IR 0.891 against the control's 18.21% / 1.243; at 5% and 10% targets also worse | rejected: the touch fires earlier but the fill is at the next close, which is often below the target |
| entry quality gates (`--p-min` 0.5-0.8, `--mu-min` 0.0-0.02) | one seed | `--p-min 0.70` leaves **53 trades in eight years**; 0.80 leaves 2; every setting is worse | rejected: the model's calibrated win probability is not sharp enough to gate on - the same message as its 5% contribution |
| constraint structure (new diagnostic) | one run | 3044 candidates/day, 0.52 orders/day, 466 entry days of 1869, 11.73/12 slots full, gross 0.715 against a 0.693 target | the design's binding constraint is the slot count, and its information content is exhausted |

Six axes have now been closed in three stage-1 rounds: the third-opinion consensus, sub-book
ensembling, the liquidity measure and its window, breadth, the intraday exit trigger, and the entry
gates. The best replicated information ratio remains **0.983** (`solo_amt`, nine seeds) against
stage 1's 1.30. Control reproduction after this round's code changes
(`--profit-trigger-high`, keeping the `high` field): the flag-off run is bit-identical to the
headline (18.21% / 14.65% / -19.84% / 77.83% / 433 trades), and the 27 tests pass.



## 23. Stage-1 round 4: the last two structural levers, both closed

The round-3 diagnostic (11.73 of 12 slots full while 3044 candidates pass the gate daily) leaves two
structural levers: which names occupy the slots, and how risk is spread across them.

| mechanism | sample | result | verdict |
|---|---|---|---|
| **rank exit in the live path** (`--exit-rank`): leave a holding once its blended score drops out of the top quantile | one seed, six settings, zero control | control 18.21% / IR 1.243; every setting loses 4-8pp of CAGR and the win rate falls from 77.83% to 54-63% | rejected - the **sixth** falsification of cutting positions early, and the first with a *ranking* trigger rather than a price trigger |
| **inverse-vol re-weighting of the held sleeve** (`--risk-parity`), work item 3 | one seed, three strengths, zero control | strength 0.25 is an exact tie (IR 1.244 vs 1.243); 0.5 and 1.0 lose return in proportion to the volatility they remove (1.073, 1.002) | closed: it moves along the frontier instead of rotating it |

Two process notes matter as much as the results:

* The rank exit was **silently inert in all ~520 recorded runs**: the config default is
  `exit_rank = True` and the mechanism exists in `portfolio.evaluate_exits`, but the V2 live path
  (`V2Strategy.evaluate_exits`) never read it. This is the third instance of that trap (the ATR
  target and the `--rank-blend 0` composite arm were the others), and it is why every exit-side
  change in this project now carries a zero control.
* Both zero controls reproduced the headline bit-for-bit, so the new flags are verified rather than
  assumed.

Nine axes are now closed across four stage-1 rounds. The best replicated information ratio remains
**0.983** (`solo_amt`, nine seeds) against stage 1's 1.30. The only intervention in the entire
project that has materially moved the information ratio is the **exit-policy redesign** of rounds
23-24 (target 3.5% -> 7.5%, cap 45 -> 120 days), worth +2.5pp of CAGR and +8pp of win rate at equal
risk; the entry side has been a wall of negatives.



## 24. Stage-1 round 5: hard consensus, patience (twice), and the acceptance matrix

| mechanism | sample | result | verdict |
|---|---|---|---|
| **hard consensus** (`--blend-mode min`): require both rankers to like the name instead of averaging their ranks | one seed, zero control | 15.44% CAGR / IR 1.058 (min) and 16.24% / 1.111 (min + threshold 0.6) against the control's 18.21% / 1.243 | rejected: the **soft** consensus (average) is the better selector |
| **absolute entry threshold** (`--entry-min-score` 0.40/0.50/0.60) | one seed | **bit-identical to the control** at every setting | a verified **silent no-op**: the maximum of a percentile-rank average over ~3000 names is always high, so no absolute threshold can bind |
| **relative patience** (`--entry-quality-pct`): only open a slot on days whose best candidate beats a trailing percentile | one seed, four settings | the gate verifiably fires (open on 70.5% / 57.0% / 42.3% of days) and every setting loses: 11.57% / IR 0.835, 10.24% / 0.816, 8.89% / 1.043, 10.86% / 0.871 | rejected: the level of the daily best score carries **no information** about the next day's opportunity; the 0.90 gate does produce the highest win rate ever measured here (83.14%) but at two-thirds less return |

Twelve axes are now closed across five stage-1 rounds, each with a reproduced control. Together with
the earlier rounds the measured information ratio has stayed in the **0.90-1.00 band across twelve
consecutive independent interventions**, and the only mechanism that ever moved it remains the exit
policy redesign of rounds 23-24.

**Deliverable produced this round**: `scripts/acceptance_matrix.py` builds the per-seed acceptance
matrix straight from the recorders. `c2` passes **4 of 7 checks on all nine seeds** (volatility,
win rate, and both frequency floors); `z1` passes **3 of 7 on all nine seeds**. Neither stage 1
(seed-mean IR >= 1.30) nor stage 2 (>= 1.60) is reached; the achieved seed-mean information ratio is
**0.948** at the compliant exposure and **0.945** at the return end.



## 25. Stage-1 round 7: the board lead is killed by more data

The STAR/Beijing exclusion was the only positive sign in six rounds (+0.96pp of CAGR, paired
t = +1.19, 6 of 9 seeds). Six further model seeds were built to take the paired sample to fifteen.

| sample | n | control | exclude STAR+BSE | difference | t | seeds in favour |
|---|---:|---:|---:|---:|---:|---:|
| original nine | 9 | 13.76% | 14.72% | +0.96pp | +1.19 | 6/9 |
| six new seeds | 6 | 13.96% | 14.34% | +0.38pp | +0.69 | 3/6 |
| **pooled** | **15** | **13.84%** | **14.57%** | **+0.73pp** | **+1.40** | **9/15** |

The effect shrank on fresh data and the sign became a coin flip, so **board composition is closed**
and with it the last open lead. Fourteen axes are now closed across seven stage-1 rounds, every one
with a control that reproduced the incumbent bit for bit.

Final position: seed-mean information ratio **0.948** (compliant) and **0.945** (return end) against
stage 1's 1.30, stage 2's 1.60 and stage 3's 2.50. The contract's win-rate and volatility checks are
met on every seed (74.8-80.5% and 9.67%); the CAGR check is not, at 9.16% against 25%.



## 26. Stage-1 round 8: scale-out rejected, and the mechanism space is exhausted

| mechanism | sample | result | verdict |
|---|---|---|---|
| **scale-out** (`--scale-out <fraction> --profit-target-2 <level>`): bank part of the position at the first target and let the remainder run to a second | one seed, five settings, zero control | control 18.21% / IR 1.243 / 77.83% win (bit-identical); every setting loses 3-9pp of CAGR and 3-15pp of win rate (best 15.26% / 1.106) | rejected - the first target is where the strategy's return comes from, and taking half of it there is strictly worse |

**Fifteen axes closed across eight rounds**, each with a control that reproduced the incumbent bit
for bit: exit policy (the full target x hold grid including the short-hold corner, stops, trailing
stops, stale-loser exits, profit locks, the intraday-touch trigger, the ranking-triggered exit,
scale-out, market-state exits), entry policy (blend weight, hard consensus, 13 third-opinion
partners, an absolute gate that verified as a no-op, three forms of relative patience, head
re-ranking), portfolio construction (breadth twice, inverse-vol re-weighting, correlation sizing,
drawdown tiers, rolling-peak drawdown, risk-trim mode), universe (liquidity floor, board
composition), model (training length, objective, labels, seed ensembles), and data availability.

The seed-averaged information ratio has stayed in the **0.94-1.00** band through all of it, against
stage 1's 1.30. The one intervention that ever moved it - the exit-policy redesign of rounds 23-24 -
is already inside the incumbent, and the deliverable statement in
`reports/v2_acceptance_report.md` section 12 records which stage was reached and why the remaining
routes lie outside the stated constraints.

