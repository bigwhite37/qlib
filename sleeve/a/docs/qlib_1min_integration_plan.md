# Qlib 接入 1 分钟 A 股行情：现状核查与改造方案

> 核查方式：对 `/Volumes/lexar_4t/code/data/warehouse` 全程**只读**访问。
> 所有结论标注为「实测」的都有可复现的脚本或 SQL；标注为「未核」的没有验证，不要当结论用。
> 复现脚本见 `sleeve/a/docs/qlib_1min_probe/`。

---

## 0. 结论先行

三条：

1. **qlib 的表达式引擎不需要改核心就能在 1min 上跑通。** 这不是推断——我用仓库里的 parquet 造了一份 qlib 原生 1min 数据集，`$close / Ref / Mean / Std / Delta / Corr / DayLast / DayCumsum / $vwap / $amount` 全部与 pandas 真值一致（float32 精度内）。**零 qlib 代码改动。**

2. **真正的缺口在接入层和语义层，不在引擎层。**
   - 接入层：日历、instruments、复权、增量。已有的 `duckdb_storage.py` 只认日频；`DuckDBCalendarStorage` 的分钟分支**今天必然抛异常**。
   - 语义层：`Ref/Mean` 在 1min 平轴上会**跨午休和隔夜**；高频算子**不在默认 `OpsList`**；`DayCumsum` 窗口关闭后返回 0 而不是冻结值。这些是"原生支持"要补的东西。

3. **路线 A 已经落地并验收通过。** exporter 写好了，全市场 5194 标的 × 12 交易日跑通，
   qlib 表达式逐字段与 parquet 对账全部通过（相对误差 ~4e-8，即 float32 精度）。
   全历史外推约 **57 GB / 1–2 小时**。
   - `sleeve/a/scripts/export_qlib_1min.py` — parquet → qlib 原生 1min 布局
   - `sleeve/a/scripts/validate_qlib_1min.py` — 用 qlib 表达式回归对账 parquet

4. **路线 B 也已落地并验收通过（本轮新增，见 §8）。** 不导出任何行情副本：provider 库里只有
   日历、证券表和"字段从哪来"的描述，`D.features(..., freq='1min')` 直接按
   `year/month/exchange/bucket` 剪枝读 warehouse parquet。
   - `sleeve/a/scripts/build_qlib_1min_provider.py` — 生成 provider 库（DuckDB，2 GB 上限）
   - `sleeve/a/scripts/validate_qlib_1min_duckdb.py` — P0 契约测试，`ALL CHECKS PASSED`
   - 全市场 5463 标的 × 2 字段 × 2 日（245 万行）实测 **24.4 s**，RSS 峰值 < 3 GiB
   - 高频算子（`DayLast/DayCumsum/...`）已按"配了 1min 就自动可用"注册，不再需要手写 `custom_ops`

---

## 1. 仓库现状（只读核查）

### 1.1 存储形态

| 层 | 路径 | 内容 |
|---|---|---|
| 事实数据 | `silver/bars_1m_raw/year=Y/month=M/exchange=E/bucket=NN/part-*.parquet` | Hive 分区，zstd |
| 控制平面 | `meta/catalog.duckdb` (57.8 GiB) | 维表 + `bars_1m_raw_catalog` 冗余副本 |
| 已有 qlib 导出 | `marketstore/exports/qlib_daily/qlib_data/cn_data/` | **仅日频** |
| 已有 qlib 库 | `warehouse/qlib/qlib.duckdb` → `qlib_tmp_correct_reindexed.duckdb` (1.2 GB) | **仅日频**，sleeve/a 现在走这条 |

**实测（parquet 文件元数据）**：1456 个文件、**1,847,849,342 行**、17 GB。
分区键 `year/month/exchange/bucket`，**`bucket = symbol_id % 8`**（每个桶目录恰好 1 个 parquet 文件）。

**parquet 列**（`DESCRIBE`，实测）：

```
symbol_id BIGINT, trade_date INTEGER, minute_slot SMALLINT,
open_i BIGINT, high_i BIGINT, low_i BIGINT, close_i BIGINT,
volume BIGINT, amount_i BIGINT,
bucket VARCHAR, exchange VARCHAR, month VARCHAR, year BIGINT   -- Hive 分区列
```

**注意：parquet 里没有 `symbol` 列**，只有 `symbol_id`，必须 join `dim_instrument`。
`catalog.duckdb` 里的 `bars_1m_raw_catalog` 是同构副本 + 冗余的 `symbol/exchange/batch_id/source_file` 列（18.76 亿行，比 parquet 多 2778 万行，口径存疑——未核）。

### 1.2 `minute_slot` 语义（实测）

`minute_slot = HH*60 + MM`，取的是该 bar 的**收盘时刻**：

| 时段 | minute_slot | 时钟 | 根数 |
|---|---|---|---|
| 早盘 | 571 … 690 | 09:31 … 11:30 | 120 |
| 午盘 | 781 … 900 | 13:01 … 15:00 | 120 |

**qlib 1min label = `minute_slot` 减 1 分钟** → 09:30…11:29 + 13:00…14:59，正好 240 根，
与 `qlib/utils/time.py:get_min_cal(region='cn')` 的产出一字不差。

这个对齐是整件事的地基，`sleeve/a/docs/qlib_1min_probe/build_probe.py` 里有断言可复现。

### 1.3 单位（实测，SH600519 / 2026-09-16）

| 仓库列 | 实际单位 | 换算 | 对应 qlib 日频列 |
|---|---|---|---|
| `open_i/high_i/low_i/close_i` | 0.0001 元 | `/1e4` → 元 | `× factor` |
| `volume` | **手**（100 股） | 直接可用 | `/ factor` |
| `amount_i` | **元** | `/1000` | qlib `$amount` |
| `vwap` | **无此列** | `amount_i / (volume × 100)` | `× factor` |
| `factor` | 按日（`adjust_factor_daily`） | 日内常数 | 直接乘 |

> `docs/generated/db-schema.md` 写 `amount_i | INT64 | monetary amount in cents`（分）——**实测是元**。
> 600519 当日 `Σ amount_i = 3,307,926,407`，按"分"算是 3308 万元，与其成交额量级差 1000 倍；按"元"算是 33.08 亿元，正确。

对账证据（全部实测）：

```
Σ volume(240 根)            = 26235    == bars_1d_raw.volume        ✓
Σ amount_i(240 根)          = 3307926407 == bars_1d_raw.amount      ✓
open(slot 571)              = 1273.93  == bars_1d_raw.open_raw      ✓
close(slot 900)             = 1258.00  == bars_1d_raw.close_raw     ✓
vwap = amount_i/(volume×100) = 1260.88                              ✓
qlib daily close  314.3232  = 1258.00 × 0.2498594431434518          ✓
qlib daily volume 104999.03 = 26235   / 0.2498594431434518          ✓
qlib daily amount 3307926.407 = 3307926407 / 1000                   ✓
```

**复权口径**：1min raw 不回写复权价，用 `adjust_factor_daily` 按日广播——因为日内无除权事件，factor 日内是常数，**日内收益率完全不受影响**，跨日连续。这个口径是对的。

### 1.4 覆盖与数据质量（实测）

- **2020-01-02 … 2026-09-16，1627 个交易日**
- **每个标的每个交易日恰好 240 根**。对 SH `bucket=00` 全历史 425,636 个 symbol-day 做 `GROUP BY bars` 分布检查，结果**只有 240 这一档** —— 没有半日市、没有缺 bar、没有停牌补零。
  这一条很重要：它意味着 qlib 硬编码的 240 根日历是**精确**的，也意味着 `DayCumsum` 内部 `assert len(df) == 240 // granularity` 不会炸。
- 交易所：SH/SZ 全期；**BJ 只有 2025 年起**（20 个分区）。

### 1.5 已有的 qlib 产物（都要注意"仅日频"）

`marketstore/exports/qlib_daily/qlib_data/cn_data/`：

```
calendars/day.txt              # 每行一个日期
instruments/all.txt            # SYMBOL\tstart\tend
features/sh600592/close.day.bin  # float32: [start_index, v0, v1, ...]
```

这正是 qlib 原生格式，也证明了"导出"这条路在你们仓库里已经跑通过一次——**只是没做 1min**。

---

## 2. Qlib 侧现状

### 2.1 已经具备的（不要重造）

| 能力 | 位置 |
|---|---|
| 1min 是一等频率 | `Freq`、`get_min_cal()`、`resam_calendar`、`HIGH_FREQ_CONFIG` |
| 表达式按 freq 透传 | `Expression.load(..., freq)` → `LocalExpressionProvider` → `FeatureD.feature(..., freq)` |
| 高频算子库 | `qlib/contrib/ops/high_freq.py`：`DayLast / DayCumsum / FFillNan / BFillNan / Date / Select / IsNull / IsInf / Cut` |
| 高频 handler | `qlib/contrib/data/highfreq_handler.py`：`HighFreqHandler / HighFreqGeneralHandler / HighFreqBacktestHandler / HighFreqOrderHandler` |
| RL 集成 | `qlib/rl/data/integration.py` 已按 1min/5min/day 三档注册 |
| 回测/执行 | `Exchange(freq='1min')`、`NestedExecutor`、`DayCumsum` 量约束、`examples/nested_decision_execution/` |
| 完整范例 | `examples/highfreq/` |

### 2.2 缺口与坑（全部实测）

**① `DuckDBCalendarStorage` 的分钟分支今天必然抛异常 —— 已修（§8.1）**

原实现按 `Freq(self.freq) != Freq("day")` 走 `resam_calendar(..., "day", self.freq, ...)`：

```python
if freq_sam.base == Freq.NORM_FREQ_MINUTE:
    if freq_raw.base != Freq.NORM_FREQ_MINUTE:
        raise ValueError("when sampling minute calendar, freq of raw calendar must be minute or min")
```

传 `"day"` 进去必然 `ValueError`。**这条路径从来没被走通过。** 现在分钟频率改为直接读自己的
`qlib_minute_calendar`，并用 `_validate_minute_calendar` 拒绝把日级标签当成分钟日历。

**② `DuckDBFeatureStorage` 没有分钟维度**

写死 `qlib_daily_features` 表 + `trade_index`（日频日历下标），列名白名单是 `open/high/low/close/volume/amount/vwap/factor/...`，没有任何分钟键。

**③ 高频算子不在默认 `OpsList`（实测）—— 已修（§8.2）**

```
Ref / Mean / Std / Corr / Delta   in qlib.data.ops.OpsList = True
DayLast / DayCumsum / Cut / Select / Date  in OpsList = False
```

原来必须显式 `qlib.init(custom_ops=[DayLast, DayCumsum, ...])`。现在 `register_all_ops` 检测到
provider 里配了分钟频率就自动注册 `qlib.contrib.ops.high_freq.HIGH_FREQ_OPS`，日线配置不受影响。

**④ `Ref/Mean` 在 1min 平轴上跨午休和隔夜（实测）**

| 时点 | `$close` | `Ref($close,1)` | `Ref($close,2)` |
|---|---:|---:|---:|
| 2026-09-01 11:29 | 1301.35 | 1300.46 | 1301.27 |
| 2026-09-01 **13:00** | 1299.00 | **1301.35** ← 11:29 收盘 | 1300.46 |
| 2026-09-02 **09:30** | 1299.80 | **1299.56** ← 昨日 14:59 | 1299.50 |

没有 session-aware、也没有 day-anchored 的 rolling。做日内因子时这是最容易踩的坑。

**⑤ `DayCumsum` 窗口关闭后返回 0，不是冻结值（实测）**

```
2026-09-01 10:00  $volume=241  DayCumsum($volume,'9:30','10:00')=7798
2026-09-01 10:01  $volume=224  DayCumsum($volume,'9:30','10:00')=0     ← 不是 7798
2026-09-01 14:59  $volume=266  DayCumsum($volume,'9:30','10:00')=0
```

**⑥ `DayLast` 是日内前视算子（实测）**

09:30 就返回当日收盘（1301.87）。这是**设计如此**——用于做标签和归一化基准——但必须配 `Ref(DayLast(...), 240)` 才是"昨收"，也就是 `HighFreqHandler` 里的标准写法。文档没写清楚，容易误用。

**⑦ `DayCumsum` 内部断言与半日市不兼容**

`assert len(df) == 240 // self.data_granularity`。你们数据恰好没有半日市（1.4 节实测），所以安全；但这是个隐含耦合。

---

## 3. 实测验证

### 3.1 表达式在 1min 上端到端跑通（零 qlib 代码改动）

造了 2 标的 × 12 交易日 × 240 根的 qlib 原生 1min 数据集，`qlib.init(provider_uri={'1min': ...})`：

```
  $close            OK   n=2880  max_abs_diff=5.859e-05  rel=4.38e-08
  Ref($close,1)     OK   n=2879  max_abs_diff=5.859e-05  rel=4.38e-08
  Mean($close,5)    OK   n=2876  max_abs_diff=1.035e-04  rel=7.74e-08
  DayLast($close)   OK   n=2880  max_abs_diff=5.859e-05  rel=4.41e-08
  DayCumsum($volume) OK  n=2880  max_abs_diff=0.000e+00  rel=0.00e+00
```

（`Delta($close,1)` 显示的 1.34e-05 是 float32 相减的抵消误差，不是 bug。）

**交叉验证**：`DayLast($close)` 当日末值 `= 1258.0 = bars_1d_raw.close_raw`；
`DayCumsum($volume)` 当日末值 `= 26235 = bars_1d_raw.volume`。
**1min 表达式与日频事实表自洽。**

可运行的完整例子：`sleeve/a/docs/qlib_1min_probe/demo_minute_dataset.py`

```bash
cd /Users/shuzhenyi/code/python/qlib
scripts/run_python_3gb.sh sleeve/a/docs/qlib_1min_probe/demo_minute_dataset.py
```

```python
qlib.init(provider_uri={"1min": ROOT}, region="cn",
          custom_ops=[DayLast, DayCumsum, FFillNan, BFillNan, Date, Select, IsNull, IsInf, Cut])
df = D.features(insts, ["$close", "Mean($close, 20)", "DayCumsum($volume, '9:30', '14:59')"],
                start_time=START, end_time=END, freq="1min")
```

**也就是说：只要能落出一份 qlib 原生布局的 1min 数据集，今天就能用表达式，一行 qlib 代码都不用改。**
缺的不是引擎能力，是**从 warehouse parquet 生成那份数据集的 exporter**（P3 路线 A，~2 人日）。

### 3.1b 两个实操坑（我自己踩了，写脚本必看）

跑 `sleeve/a/docs/qlib_1min_probe/demo_minute_dataset.py` 时踩到两个，都会静默出问题：

**① `python path/to/script.py` 会加载 pip 安装的 qlib，而不是这个 checkout**

Python 把**脚本所在目录**放进 `sys.path[0]`，不是 cwd。所以
`python sleeve/a/docs/.../demo.py` 里的裸 `import qlib` 会解析到
`/Users/shuzhenyi/miniconda3/lib/python3.12/site-packages/qlib/`。
（用 `-c "..."` 时 `sys.path[0]` 才是 cwd，所以我早先的探针没暴露这个问题。）

脚本里必须显式把 checkout 插到 `sys.path` 前面，并断言 `qlib.__file__` 在 checkout 下。
sleeve/a 的 `bootstrap.ensure_local_qlib()` 就是干这个的。

**② 脚本必须有 `if __name__ == "__main__":` 保护**

`D.features` 会通过 joblib 的 **multiprocessing（spawn）** 后端按 `C.kernels` 起工作进程。
spawn 的子进程会**重新 import 主模块**——如果模块顶层有可执行代码（`qlib.init`、`D.features` 调用），
每个子进程都会再跑一遍并再起一批子进程，变成进程风暴，表现为脚本卡死 + CPU 打满。

我第一版 demo 就是顶层直接调的，直接把机器跑挂了几分钟。**所有 1min 脚本都要包成 `main()`。**

### 3.3 Exporter 已实现并通过验收

```bash
cd /Users/shuzhenyi/code/python/qlib

# 导出（默认只取 A 股股票，因为 bars_1m_raw 里只有股票）
scripts/run_python_3gb.sh sleeve/a/scripts/export_qlib_1min.py \
    --out /tmp/qlib_1min_full --start 2026-09-01 --end 2026-09-16 --exchanges SH,SZ --overwrite

# 用 qlib 表达式回归对账 parquet
scripts/run_python_3gb.sh sleeve/a/scripts/validate_qlib_1min.py \
    --data /tmp/qlib_1min_full --symbols SH600519,SZ000001,SH601318
```

**全市场实测（5194 标的 × 12 交易日）**

| 指标 | 值 |
|---|---|
| 读取 parquet 行数 | 14,943,600 |
| 写出 bin 文件 | 36,358（7 字段 × 5194 标的） |
| 写出体积 | 419 MB |
| 耗时 | **56.5 s** |

**验收结果：全字段 ALL CHECKS PASSED**

```
OK   SH600519 $close   n=2880  rel=4.377e-08
OK   SH600519 $volume  n=2880  rel=0.000e+00
OK   SH600519 $vwap    n=2856  rel=4.535e-08
OK   SH600519 DayLast($close)==day close       n=12  rel=4.406e-08
OK   SH600519 DayCumsum($volume)==day volume   n=12  rel=0.000e+00
```

（`$vwap` 的 n=2856 是 24 根零成交 bar 产生 NaN，被 dropna 排除，符合预期。）

**全历史外推**：1627 交易日 × 240 根 = 390,480 根/标的
→ 约 **2.0e9 cells/字段 × 4 bytes × 7 字段 ≈ 57 GB**，耗时 **1–2 小时**量级。

**实现里踩到并修掉的三个坑**（都会静默产出错误数据，不会报错）：

1. **bin 必须带 `start_index` 头**。第一版 `write_bin` 只写了值，qlib 就把第一个价格当成
   起始索引，整条序列错位一根 —— 表现是 `$close` 差 1.8%、`$volume` 全 NaN。
2. **bin 要跨分区合并，不能重写**。一个标的的分钟数据分散在每个月分区里，
   早期版本每个月都整体覆盖一次，最后只剩最后一个分区。
3. **`--limit` 的universe 必须过滤非股票**。`dim_instrument` 里 `SH000300`（沪深300指数）
   按字母序排在 `SH600000` 前面，而 `bars_1m_raw` 只存股票 ——
   不加过滤会选出一批**完全没有数据**的标的，导出 0 行。

### 3.2 规模与性能（200 标的 × 2880 根 × 5 字段）

| 读取方式 | 数据量 | 耗时 |
|---|---|---|
| **parquet 直读**（DuckDB，2GB 上限，2 线程） | 576,000 行 | **0.1 s** |
| qlib 文件后端，1 字段 | 576,000 cells | 2.75 s |
| qlib 文件后端，5 字段 | 2,880,000 cells | 2.88 s |
| qlib 文件后端，1 原始 + 2 个 rolling(20) | 1,728,000 cells | 5.13 s |

- **跨截面读 parquet 比 qlib 文件后端快约 27×**
- qlib bin 开销：**4.00 bytes/值**（float32，无压缩）
- 全量外推：1627 日 × 240 根 ≈ 390,600 根/标的；5000 标的 ≈ **19.5 亿 cells/字段**
  - qlib 文件布局：**≈ 7.8 GB/字段** → 7 字段 ≈ **55 GB**
  - 按 ~1.0 Mcells/s 全量物化 ≈ **33 分钟/字段**

**这组数字直接决定架构**：qlib 的文件后端是"每 (标的, 字段) 一个文件"的随机访问模式，
全市场全历史需要 5000×7 = 35,000 次文件打开。这也是为什么路线 B 有吸引力。

---

## 4. 需要做的工作

### P0 — 数据契约（0.5 人日）

把第 1 节的所有映射固化成 repo 内的 spec + **契约测试**：

- `minute_slot → qlib label`（减 1 分钟）
- 单位换算表
- 复权口径（`× adjust_factor_daily`）
- 缺失 bar 的语义（NaN 还是 ffill）——你们数据不缺 bar，但**未来可能缺**，要定死
- 契约测试要能直接对 parquet 跑，断言：
  `DayLast($close) == bars_1d_raw.close_raw`、`DayCumsum($volume) == bars_1d_raw.volume`

### P1 — 交易日历（1 人日）

- 生成 `calendars/1min.txt`。因为实测**没有半日市**，最稳的做法是
  「日频日历 × `get_min_cal(region='cn')` 展开 240 根」。
- 同时产出 `calendars/day.txt`（qlib 的日频对齐和回测要用）。
- 若要更通用（防未来半日市），改用 `trading_calendar` 的
  `session_open_time / session_close_time / midday_break_* / is_half_day` 生成。
- **验收**：`len(D.calendar(freq='1min')) == 交易日数 × 240`。

### P2 — instruments（0.5 人日）

- `instruments/all.txt`：`dim_instrument` 的 `qlib_symbol` + `list_date` + `delist_date`。
- 市场分组：`all / csi300 / csi500 / csi1000`（`D.instruments(market=...)` 要用）。
- ⚠️ **`dim_instrument` 权威源在 `catalog.duckdb`**。我在 checkpoints 下读到的两份 parquet 是
  **repair 中间态**——把 600519、000001 都标成 `delisted / delist_date=20260911`，明显不是终态，不能直接用。

### 实施状态（§8 有细节）

| 工作包 | 状态 |
|---|---|
| P0 数据契约 + 契约测试 | ✅ `validate_qlib_1min_duckdb.py`（`DayLast/DayCumsum` 对账 + 日线桥接 + 240 根断言） |
| P1 交易日历 | ✅ `build_qlib_1min_provider.py` 写 `qlib_minute_calendar`（日频日历 × 240 根） |
| P2 instruments | ✅ 同一脚本写 `v_qlib_instruments_latest` |
| P3 特征存储 | ✅ 路线 A（bin 导出）与路线 B（DuckDB 直读）都可用；本轮把路线 B 做成默认的 qlib 原生路径 |
| P4.1 算子自动注册 | ✅ `register_all_ops` 按 provider 频率注册 |
| P4.2 语义算子 | 部分：新增 `PrevDayLast` / `DayOpen` / `BarOfDay`；`SessionRef/SessionMean/SessionCumsum` 见 §8.4 的说明 |
| P4.3 文档 | ✅ 本节与 §8 |
| P5 缓存与性能 | 部分：provider 路由已有元数据缓存与分区文件目录缓存；`DiskExpressionCache` 未针对分钟专门配置 |
| P6 回测/执行接入 | 未做（`$factor0`、`$paused_num`、分钟涨跌停仍缺） |
| P7 验收 | ✅ 见 `validate_qlib_1min_duckdb.py` 与 §8.3 的实测数字 |

### P3 — 特征存储（二选一）

**路线 A：导出成 qlib 原生 bin —— ✅ 已实现（见 §3.3）**

`sleeve/a/scripts/export_qlib_1min.py`，`parquet → calendars + instruments + features/<sym>/<field>.1min.bin`。

- 优点：**零 qlib 代码改动**，已端到端验收；`DiskExpressionCache` 直接可用；
  回测/RL 链路全部复用 `examples/highfreq`。
- 代价：全历史 ~57 GB；需要增量更新逻辑；与仓库双份数据。
- **未做**：复权（`--adjust` 目前直接报错退出，故意如此）。裸价导出已验证，
  复权应当是独立的一遍，因为它会改写每个价格 bin，必须单独对账。
- **未做**：全历史增量刷新（现在是全量重建）。

**路线 B：1min 的 DuckDB/Parquet 存储后端（推荐的长期解）**

在已有的 `qlib/data/storage/duckdb_storage.py` 基础上扩展：

1. 修 `DuckDBCalendarStorage` 的分钟分支（现在必抛，见 2.2①）
2. 让 `DuckDB{Calendar,Instrument,Feature}Storage` 变成 **freq-aware**（按 freq 选表/列）
3. 新增 1min 特征读取：按 `(qlib_symbol, trade_date, minute_slot)` 聚簇，或
   直接 `read_parquet` + `bucket= symbol_id % 8` 分区裁剪
4. 复权在读取层乘 `adjust_factor_daily`（日内常数）
5. **难点**：`LocalFeatureProvider.feature()` 是**每 (instrument, field) 一次调用**，
   5000 标的 × N 字段 = 5000N 次查询。必须靠 (a) 表按 symbol 聚簇 + zone map、
   (b) expression cache 落盘、(c) 或给 provider 加批量取数路径 来解。

- 工作量：**5-8 人日**

**路线 C（混合，最务实）**：路线 A 的导出器，但只导出**热字段**（`close/volume/vwap/factor`），
冷字段按需从 parquet 取。磁盘和速度都能接受。

### P4 — 表达式原生支持（2-4 人日）★ 你明确要的部分

分两层：

**P4.1 注册面（0.5 人日）**

把 `contrib/ops/high_freq.py` 的算子从"用户自己传 `custom_ops`"变成"配了 1min 就自动可用"。
在 `register_all_ops(C)` 里判断当前 provider 是否含分钟频率，是则把 `TOpsList`/高频算子并入默认注册集。
纯配置改动，风险低。

**P4.2 语义面（主要工作量）**

新增 1min 原生算子，全部要正确处理 `get_extended_window_size`（否则 `D.features` 边界截断会算错）：

| 算子 | 语义 | 为什么要 |
|---|---|---|
| `SessionRef(x, n)` | 按日内 bar 序号回看，跨日时截断为 NaN 或对齐到前一日同分钟 | 修 2.2④ |
| `SessionMean/Std/Max/Min(x, n)` | 日内窗口，跨日重置 | 日内因子 |
| `DayOpen / DayHigh / DayLow / DayVWAP` | 日内锚点，对标已有的 `DayLast` | 日内归一化基准 |
| `PrevDayLast(x)` | `DayLast(Ref(x, 240))` 的显式封装，**消除前视陷阱** | 修 2.2⑥ |
| `BarOfDay()` | 当日第几根 bar，对标 `Date` | 定位 |
| `SessionCumsum(x, start, end, freeze=True)` | 修 2.2⑤，窗口关闭后返回冻结值 | 量约束/回测 |

**建议：不要改 `Ref/Mean` 的默认语义**，只新增显式 `Session*` 算子。
理由：`examples/highfreq`、`HighFreqHandler`、RL 集成都依赖现在的平轴语义，改了会静默破坏既有代码。

**P4.3 文档（0.5 人日）**

`examples/highfreq` 有范例但没算子文档。补一份
`docs/advanced/minute_expression.rst`，把 2.2 的六个坑写进去。

### P5 — 缓存与性能（1-2 人日）

- `HIGH_FREQ_CONFIG` 已默认 `DiskExpressionCache`，但缓存目录是
  `C.dpm.get_data_uri(freq)/features_cache`。provider_uri 指向 parquet 目录时要显式给可写目录。
- 并行：`C.kernels`；**1min 建议 `maxtasksperchild=1`**（配置注释里已经这么写了）。
- `DiskDatasetCache` 对 1min 面板会非常大，建议只开 (instrument, field) 级缓存。

### P6 — 回测 / 执行接入（2-3 人日）

- `Exchange(freq='1min')` + `NestedExecutor`，`deal_price` 用 `$vwap0` / `$close0`。
- `HighFreqBacktestHandler` 需要两个你们仓库里**没有现成列**的东西：
  - `$factor0` —— 当日复权因子广播（可从 `adjust_factor_daily` 派生）
  - `$paused_num` —— 停牌标记（要从 `security_status` 派生）
- 涨跌停：`security_price_limit_daily` 有日频，但 1min 上要按分钟判可成交，需要新增派生。
- 量约束：`DayCumsum($volume, ...)` 可用，但注意 2.2⑤ 的语义坑。

### P7 — 验收（1 人日）

- **一致性**：1min 聚合到日 == `bars_1d_raw` == qlib 日频 `$close/$volume/$amount`
- **表达式**：P0 的契约测试
- **端到端**：一个 `examples/highfreq` 风格的最小 workflow 跑通

### 工作量汇总

| 范围 | 人日 |
|---|---|
| 路线 A + P4.1（**让 1min 表达式原生可用**） | **4-5** |
| 路线 A + P4 完整 + P5 | 7-9 |
| 路线 B + P4 完整 + P5 + P6 + P7（生产级） | **12-18** |

---

## 5. 需要你拍板的决策点

1. **路线选择**：A（导出 55 GB，零代码，4-5 人日）/ B（写后端，无冗余，12-18 人日）/ C（混合）
2. **复权口径**：1min 是否也做前复权（`raw × 当日 factor`）？我倾向"是"，与现有日频口径一致。
3. **起点**：2020 起全历史（55 GB）还是只导最近 N 年？
4. **跨午休/隔夜的 rolling 语义**：默认重置，还是默认连续？我倾向**不改默认**，只加显式 `Session*` 算子。
5. **BJ 交易所**（只有 2025+）是否纳入？
6. **与现有日频 qlib 库的关系**：1min 和 `qlib_tmp_correct_reindexed.duckdb` 是否共用一个
   `provider_uri` 字典（`{"day": ..., "1min": ...}`）？两者必须共用同一份 `dim_instrument` 口径。

---

## 6. 未核 / 风险

| 项 | 状态 |
|---|---|
| `dim_instrument` 权威源 | **未核** —— 只读到 repair 中间态 parquet，live 版在 `catalog.duckdb` |
| 逐年行数拆分 | **未核** —— 中途失去卷读权限（见 §7） |
| `bars_1m_intraday_snapshot` 语义 | **未核** —— 决定"盘中实时 1min"能否也用 qlib |
| `security_status` / 停牌 / 涨跌停在 1min 上的表达 | **未核** |
| `bars_1m_raw_catalog` 比 parquet 多 2778 万行 | **未核** —— 口径差异需要解释 |
| 半日市 | 实测 2020-2026 无；**未来若有**，`DayCumsum` 会断言失败，日历也要改 |

---

## 7. 附：本次核查中的环境事故（需要你处理）

1. **内存**：核查早期我有几条 DuckDB CLI 查询没有设内存上限，且直接扫了
   `catalog.duckdb` 里 18.7 亿行的 `bars_1m_raw_catalog` 实体表；同时我派出的一个后台
   subagent 也在无内存上限的情况下对该文件做了统计与计时查询。后果是
   `warehouse/meta/catalog.duckdb.tmp/` 下留下 **33 GB 孤儿溢写文件**
   （`duckdb_temp_storage_*.tmp`，时间戳 20:42–20:43，无进程持有）。
   已按你的确认删除。`catalog.duckdb` 本身未变：仍是 62,093,799,424 字节、mtime 9月16 15:57、无 `.wal`，
   所有连接均为 `-readonly`。
   **后续纪律**：所有 DuckDB 连接强制 `SET memory_limit='2GB'; SET threads=2; SET temp_directory=...`；
   优先读 parquet 分区而不是 catalog 大表；先 `EXPLAIN` 再跑。

2. **卷权限丢失（未解决）**：核查过程中 `/Volumes/lexar_4t` 变成
   **元数据可读（`stat` 正常）但文件 `open()` 返回 EPERM**，对目录 `ls` 也 EPERM。
   这是 macOS 对可移动卷的 TCC 权限被回收的典型表现，不是 DSH 沙箱（本会话是 danger-full-access）。
   发生时间与上面那次 `rm -rf` 接近，但我**无法确定因果**。
   当前状态下**我已完全无法读取该卷上的任何数据**。恢复办法大概是
   系统设置 → 隐私与安全性 → 文件与文件夹 / 可移动卷，给宿主 App 授权后重新挂载。

   本文档第 1、3 节的所有数据都是在权限丢失**之前**采集的，结论不受影响。

---

## 8. 本轮实施记录（路线 B：DuckDB 直读）

> 本节记录实际落地的代码与实测结果。第 1–7 节保留原始核查结论，冲突处以本节为准。

### 8.1 存储层改动（`qlib/data/storage/duckdb_storage.py`）

1. **表名按频率解析**：分钟频率读 `qlib_minute_calendar` / `qlib_minute_features`，
   日频维持 `qlib_calendar` / `qlib_daily_features`。
2. **分钟日历不再从日线 resample**：直接读自己的标签，并用 `_validate_minute_calendar`
   在"日历行全是日级时间戳"时直接报错，而不是悄悄把日标签当成分钟标签。
3. **新增 `qlib_minute_sources` 表**描述每个字段从哪里来：

   ```
   qlib_minute_sources(field, source_expr, parquet_root, symbol_table, price_unit)
   ```

   `DuckDBFeatureStorage` 在分钟频率下若查到该表，就按
   `year/month/exchange/bucket`（`bucket = symbol_id % 8`）剪枝读 parquet，一次查询取回
   一个 (标的, 字段, 窗口) 序列。`source_expr` 会先过 `check_minute_expression`：
   只允许算术 + `CAST`/`CASE`，标识符必须在白名单列内，出现注释或语句分隔符直接拒绝。
4. 元数据（source、日历、symbol_id、分区文件目录）按 provider 路径做进程内缓存，
   避免"每个 (标的, 字段) 都重读一遍 schema"。

### 8.2 算子层改动

* `qlib/contrib/ops/high_freq.py` 新增 `HIGH_FREQ_OPS` 列表，并补三个语义算子：
  - `PrevDayLast(x)`：上一交易日收盘。`Ref(x, 1)` 会跨夜，裸 `DayLast(x)` 是**当日**收盘
    （前视），这个算子把安全写法显式化。
  - `DayOpen(x)`：当日第一个值。
  - `BarOfDay(x)`：当日第几根 bar。
* `qlib/data/ops.py::register_all_ops` 在 provider 里存在分钟频率时自动注册 `HIGH_FREQ_OPS`。

### 8.3 验收（实测）

```bash
cd /Users/shuzhenyi/code/python/qlib

# 1) 生成 provider（只读 warehouse；写库同样锁 2 GB）
scripts/run_python_3gb.sh sleeve/a/scripts/build_qlib_1min_provider.py \
    --out /tmp/qlib_1min_provider/minute_provider.duckdb \
    --start 2025-02-07 --end 2025-03-31 --overwrite

# 2) P0 契约测试（qlib 表达式 vs DuckDB 手算）
scripts/run_python_3gb.sh sleeve/a/scripts/validate_qlib_1min_duckdb.py \
    --provider /tmp/qlib_1min_provider/minute_provider.duckdb \
    --symbols SH600519,SZ000001,SH601318,SZ300750 --start 2025-02-10 --end 2025-02-21
```

契约测试结果（`ALL CHECKS PASSED`，40 项）：

| 检查 | 实测相对误差 |
|---|---|
| `DayLast($close)` == 当日 15:00 收盘 | ~4e-8（float32 精度） |
| `DayCumsum($volume)` == Σ volume | 0 |
| `DayCumsum($amount)` == Σ amount/1e3 | ~6e-7 |
| `DayLast($close) × factor` == 日线复权收盘 | ~1e-7 |
| `DayCumsum($volume)` == 日线 volume × factor | ~2e-6 |
| 每个覆盖日恰好 240 根 | 通过 |

吞吐（`D.features(..., freq='1min')`，2 个字段，2 个交易日窗口）：

| 标的数 | 行数 | 耗时 | 每标的 |
|---|---|---|---|
| 10 | 4,320 | 0.08 s | 8.2 ms |
| 50 | 23,520 | 0.17 s | 3.3 ms |
| 200 | 85,920 | 0.57 s | 2.9 ms |
| 5463（全市场） | 2,453,280 | **24.4 s** | 4.5 ms |

对比第 2 节对路线 B 的悲观估计（"5000 标的 × N 字段 = 5000N 次查询，必须靠聚簇/zone map/
缓存"）：bucket 分区剪枝 + parquet 统计信息已经足够，**不需要**额外的批量取数通道。

### 8.4 未做与注意事项

* **`start_time`/`end_time` 是分钟标签**。`end_time="2025-02-21"` 会被解析成 00:00，
  那一天一根 bar 都取不到。要用 `"2025-02-21 14:59:00"`。`factor_lab/minute_data.py` 传的是
  会话轴上的精确时间戳，不受影响。
* **`$factor` 不在分钟 parquet 里**：复权因子是日频属性。分钟 provider 不提供该字段，
  需要时由调用方从日线库按日广播（factor_lab 就是这么做的）。
* **`SessionRef/SessionMean/SessionStd/SessionCumsum` 没有新增**。它们在 flat 分钟轴上需要
  显式会话边界，而 `factor_lab` 已经用"会话轴 + 哨兵 bar"在结构上保证不跨界；再加一层算子
  会在没有消费者的情况下扩大 qlib 的公开面。要做的话应当连同会话元数据一起设计。
* **P6 回测/执行接入仍未做**：`$factor0`、`$paused_num`、分钟级涨跌停都还缺，
  `NestedExecutor` 与 `HighFreqBacktestHandler` 的接线未验证。
* **半日市**：provider 的日历按 240 根/日展开。实测 2020-2026 没有半日市；将来若出现，
  `DayCumsum` 的内部断言会失败，需要连同日历一起改。
* provider 库只描述数据位置，**不锁数据**。warehouse 分区被重写后必须重建 provider。
