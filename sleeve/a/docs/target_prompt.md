# Target prompt: raise the information ratio of the sleeve/a strategy

## 1. Where we are (verified, 28 rounds / ~520 account runs)

The strategy is a policy-consistent, low-volatility mean-reversion book: 12 names, 1 entry per day,
market ladder off, 7.5% profit target, 120-day hold, 50/50 blend of a rolling LightGBM and the
causal `combo3` rule composite. Everything is measured through Qlib's engine with Qlib's recorder.

Two frontier points, both nine-seed means (nine independently trained model seeds of an otherwise
identical pipeline), so the numbers are not single lucky draws:

| configuration | CAGR | vol | max DD | win rate | IR | contract checks |
|---|---:|---:|---:|---:|---:|---|
| `c2` compliant: `--use-market-timing 0 --vol-target 0.055 --profit-target 0.075 --hold 120` | 9.16% | 9.67% | -15.92% | 74.76% | 0.948 | 4/7 on 9/9 seeds |
| `z1` return end: `--use-market-timing 0 --vol-target 0.14 --profit-target 0.075 --hold 120` | 13.76% | 14.58% | -23.18% | 75.39% | 0.945 | 3/7 |

Already met and robust: **win rate >= 70%** (74.8-80.5%, clearing the bar on 9 of 9 seeds) and
**volatility <= 10%** at the compliant point (9.67%, 9 of 9 seeds).

**Not met: CAGR >= 25%.** The reason is now established and it is not a drawdown-mechanics problem:

* `CAGR ~= IR x volatility budget`, so 25% at 10% volatility needs **IR = 2.5**. The best
  seed-mean IR measured is 0.948 and the best single run ever recorded is 1.243.
* The drawdown-to-volatility ratio is **not** a structural constant: across 503 runs
  `corr(IR, DD/vol) = -0.55`, with a fitted line `DD/vol ~= 2.15 - 0.61 x IR`. Feeding that back
  into the contract's two risk limits (vol <= 10% and DD >= -10%) gives a feasible CAGR of 6.5% at
  IR 1.0, 20% at IR 2.0 and **25% at IR 2.5**. The contract is internally consistent; it describes a
  strategy with information ratio ~2.5. The gap is strategy quality, not arithmetic.

## 2. Objective

**Raise the seed-averaged information ratio of the strategy.** Everything else follows from it.

| stage | seed-mean IR | implies CAGR at vol <= 10% | additional requirements |
|---|---:|---:|---|
| current | 0.95 | 9.2% | win >= 70%, vol <= 10% |
| **stage 1** | **>= 1.30** | **>= 13%** | win >= 70%, max DD >= -16% |
| **stage 2** | **>= 1.60** | **>= 16%** | win >= 70%, max DD >= -13% |
| **stage 3 (contract)** | **>= 2.50** | **>= 25%** | win >= 70%, vol <= 10%, max DD >= -10% |

Secondary target at the return end: with volatility around 15%, **CAGR >= 20%** and win rate
**>= 75%** (today: 13.76% / 75.39%).

Every stage must hold on **at least 8 of 9 model seeds**, not on the best seed.

## 3. Work items, in priority order

1. **Find a structurally different second ranker (the consensus primitive).** CORRECTED after
   measurement: the entry score is a **rank consensus**, and it beats both of its members and their
   *return* blend (composite-only IR 0.941, model-only 0.740, 50/50 return blend 0.935, 50/50 rank
   consensus **1.295** on the same dates). It is a *selection* effect, not diversification: the two
   arms' daily returns correlate 0.67 but the consensus picks names both rankers like. Thirteen
   third-opinion partners (rule factors and alternative-label models) were tested and every one
   dilutes the consensus, and a corpus-wide scan of 511 runs shows sub-book ensembling gains at most
   **+0.086 IR** (median pairwise correlation 0.836; reaching 2.5x would need rho <= 0.16). The open
   question is therefore not "how do we combine accounts" but "what second opinion is complementary
   to *both* the model and the rule composite in the way they are complementary to each other".
   An event/flow-style ranker would be the natural candidate; this data set has no such field.
2. **Joint design of holding period x breadth x cost.** `IR ~= IC x sqrt(BR)` while fees scale with
   turnover; the 100k account pays ~5%/yr at 25x NAV turnover. Optimise the triple jointly instead of
   sweeping the hold cap alone.
3. **Dynamic re-weighting and true risk parity.** Today existing positions are never re-sized and new
   ones use inverse-vol plus a weight cap. Use Qlib's `ShrinkCovEstimator` (already imported) for a
   covariance-based risk-parity target, and allow re-adding to winners.
4. **(Optional, no new data)** Make the profit target a resting limit order: trigger on the daily
   `high` and fill at the target price instead of waiting for the close. This is the realistic
   semantic for a limit exit and it is testable with the fields already in the panel.

## 4. Closed directions - do not re-open without new evidence

Loss cutting (ATR stops, trailing stops, stale-loser exits, profit locks: five falsifications, and
they lower the win rate too); going to cash or trimming on the account's own drawdown (self-locking
under an all-time peak; the released rolling-peak version is still worse than doing nothing);
tightening the drawdown tiers (makes the drawdown deeper); the universe liquidity floor (raising it
costs more than half the return); portfolio breadth (IR falls monotonically with more names);
re-tuning the rank-blend weight (0.50 confirmed paired on nine seeds); correlation-aware sizing;
head re-ranking; and swapping `vol20` for VWAP, idiosyncratic volatility, the MAX effect, skewness or
drawdown features. An IC scan is a filter, never a selection criterion - five times now, the
highest-IC feature has made the account worse.

## 5. Environment and constraints (mandatory)

**Every Python file and every Python snippet must be started through the memory wrapper:**

```bash
scripts/run_python_3gb.sh sleeve/a/scripts/run_qrun_v2.py --start 2019-01-02 --end 2026-09-14 ...
scripts/run_python_3gb.sh sleeve/a/scripts/run_tests.py
scripts/run_python_3gb.sh -c "import qlib; print(qlib.__version__)"
echo "print(1)" | scripts/run_python_3gb.sh -
```

It enforces a 3 GiB resident-memory ceiling on the interpreter (macOS refuses RLIMIT_AS, so it is
enforced by an RSS watchdog that kills the process when the ceiling is crossed) and it exports
`SLEEVE_DUCKDB_MEMORY_LIMIT=2GB`.

**Every DuckDB connection must be capped at 2 GB.** The Qlib storage adapter reads
`SLEEVE_DUCKDB_MEMORY_LIMIT` (default `2GB`) and asserts the setting after opening; any direct
`duckdb.connect(...)` must pass `config={"memory_limit": "2GB"}`.

Other standing constraints:

* Use the Qlib checkout at `/Users/shuzhenyi/code/python/qlib`; never the pip-installed Qlib.
* Prefer Qlib's own components: `SimulatorExecutor`, `Exchange`/`Quote`, `WeightStrategyBase`,
  `OrderGenerator`, `contrib.model.gbdt.LGBModel`, `model.riskmodel.shrink.ShrinkCovEstimator`,
  `contrib.evaluate.risk_analysis`/`indicator_analysis`, `workflow.R`.
* Wire Qlib's backtest observability into every run (the `R` recorder); every reported number must be
  read back from a recorded artefact, never from a narrative.
* Any directory that may exceed 100 MB goes under `/Volumes/lexar_4t/code/data/qlib/sleeve/a`.
* Do not add new data sources; the available fields are the daily OHLCV + VWAP panel in
  `/Users/shuzhenyi/code/data/qlib_tmp_correct/qlib_tmp_correct_reindexed.duckdb`.

## 6. Measurement protocol (a result only counts if it passes all of these)

1. **Nine model seeds minimum**, reported as mean +/- sd; comparisons between configurations must be
   **paired** on the same seeds.
2. **Beat the control on >= 8 of 9 seeds**, and the effect must exceed the seed-noise band
   (sd ~2.2pp of CAGR, ~1.6pp of win rate).
3. **Control reproduction**: after any code change, a control run of the current headline must
   reproduce bit-identically (CAGR 0.13393459533504304 for `v2_g4`, or the recorded values for the
   current headline). An exit-policy change must include a zero/off control.
4. **No silent no-ops**: verify that a new flag actually changes behaviour before drawing conclusions
   from it.
5. Run `scripts/run_python_3gb.sh sleeve/a/scripts/run_tests.py` (27 tests) before reporting.
6. Update `reports/v2_evidence_index.md`, `reports/v2_experiment_log.md`,
   `reports/v2_acceptance_report.md` and `README.md` with the verdicts - including negative ones.

## 7. Deliverables

* A configuration that reaches at least stage 1, with its seed-averaged CAGR / vol / max DD / win
  rate / IR and its per-year breakdown.
* The sub-book correlation matrix and the combined-IR measurement for the ensemble.
* An updated acceptance report against the `docs/chats_002.md` contract, with the pass/fail per
  check and per seed.
* A written statement of which stage was reached and, if stage 3 was not reached, the measured
  information ratio that was achieved and the evidence for why it is the ceiling of this data set.
