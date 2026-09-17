# AGENTS.md

本仓库（`/Users/shuzhenyi/code/python/qlib`）的开发注意事项。**动手前先读完这一页。**

---

## 1. Python 一律用限内存脚本包裹执行

不要直接 `python xxx.py`。所有 Python 文件、`-c` 片段、管道输入，都必须走：

```bash
scripts/run_python_3gb.sh <script.py> [args...]
scripts/run_python_3gb.sh -c "import qlib; print(qlib.__version__)"
echo "print(1)" | scripts/run_python_3gb.sh -
```

脚本默认施加 **3 GiB 常驻内存上限**（macOS 无法对解释器生效 `ulimit -v`，所以由 wrapper
轮询子进程 RSS，超限直接 kill -9 并返回 137）。

- 改上限：`SLEEVE_MEM_LIMIT_GB=6 scripts/run_python_3gb.sh ...`
- 换解释器（例如装包到别的 conda env）但保留内存约束：`SLEEVE_PYTHON=/path/to/python scripts/run_python_3gb.sh ...`

**为什么必须这样**：本机内存有限，一次失控的 pandas/DuckDB 全表扫描就会把整机拖进 swap，
影响的不只是你自己的任务。事故已经发生过一次（扫 18.7 亿行的表，留下 33 GB 孤儿溢写文件）。

### 附带的硬性要求：脚本必须有 `__main__` 保护

`qlib.data.D.features` 会按 `C.kernels` 通过 joblib 的 **multiprocessing(spawn)** 后端起工作进程。
spawn 的子进程会**重新 import 主模块**——如果模块顶层有可执行代码（`qlib.init`、`D.features` 调用），
每个子进程都会再跑一遍并再起一批子进程，变成**进程风暴**，表现为卡死 + CPU 打满。

```python
def main() -> int:
    ...
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

调试时加 `--kernels 1`（或 `qlib.init(kernels=1)`）可以让 traceback 保持可读——
默认情况下 worker 里的异常会被 joblib 吞掉，只留一个空 `AssertionError`。

---

## 2. DuckDB 连接必须设 2 GB 内存上限

**每一条** DuckDB 连接（CLI 和 Python 都算）都要显式设上限。DuckDB 默认吃物理内存的 ~80%
并跑满所有核心，对本地数据仓库是灾难。

### Python

```python
import duckdb

con = duckdb.connect(":memory:")          # 或 duckdb.connect(path, read_only=True)
con.execute("SET memory_limit='2GB'")
con.execute("SET threads=2")
con.execute("SET temp_directory='/tmp/ddb_scratch'")
con.execute("SET preserve_insertion_order=false")
```

- `read_only=True` 是读仓库的默认姿势；只读连接**不会**写主库文件，但**会**写溢写文件，
  所以要显式指定 `temp_directory`。
- 不设 `temp_directory`，DuckDB 会把溢写文件放在 `<db>.tmp/`——即数据库旁边。
  进程被 kill 时这些文件**不会**被清理，会留下几十 GB 垃圾。
- `SET preserve_insertion_order=false` 显著降低大读取的内存占用。

### CLI

```bash
duckdb -readonly /path/to/db.duckdb -c "
SET memory_limit='2GB'; SET threads=2; SET temp_directory='/tmp/ddb_scratch';
SELECT ...;"
```

### 读数据仓库时的纪律

数据仓库在 `/Volumes/lexar_4t/code/data/warehouse`：

- **优先读 parquet 分区，不要碰 `meta/catalog.duckdb` 里的大表。**
  `bars_1m_raw_catalog` 是 18.7 亿行的实体表；对应的 parquet 在
  `silver/bars_1m_raw/year=*/month=*/exchange=*/bucket=*/`，Hive 分区可以按 year/month 裁剪，
  成本差几个数量级。
- 先 `EXPLAIN` 再决定跑不跑。
- 不要在 18 亿行的表上跑无过滤的 `COUNT(*)` / `DISTINCT` / `SUM` / `SUMMARIZE`。
- 任何查询都要有 `LIMIT` 或分区谓词。

---

## 3. 系统里可能装着另一份（旧的）qlib

**pip 安装的 qlib 和本目录的代码是两份东西**，而且很容易静默用错。

```bash
python -c "import qlib; print(qlib.__file__)"
# 期望：/Users/shuzhenyi/code/python/qlib/qlib/__init__.py
# 危险：/Users/shuzhenyi/miniconda3/lib/python3.12/site-packages/qlib/__init__.py
```

**两份的版本号是一样的**（都报 `0.1.dev2066`），所以 `qlib.__version__` 区分不出来，
只能看 `qlib.__file__`。本目录是带本地改动的 git checkout（例如 `data/storage/duckdb_storage.py`），
site-packages 那份是旧的安装副本。

### 什么时候会踩到

`python path/to/script.py` 时，Python 把**脚本所在目录**放进 `sys.path[0]`，**不是 cwd**。
所以从仓库根目录运行 `sleeve/a/scripts/foo.py`，里面裸 `import qlib` 会解析到 site-packages 那份。
（用 `-c "..."` 时 `sys.path[0]` 才是 cwd，所以 `-c` 不容易暴露这个问题。）

### 怎么写才安全

脚本里显式把本 checkout 插到 `sys.path` 最前面，并断言：

```python
import sys
from pathlib import Path

QLIB_HOME = Path("/Users/shuzhenyi/code/python/qlib")
if str(QLIB_HOME) not in sys.path:
    sys.path.insert(0, str(QLIB_HOME))

import qlib
assert Path(qlib.__file__).is_relative_to(QLIB_HOME), f"wrong qlib: {qlib.__file__}"
```

`sleeve/a/src/lowvol_trend/bootstrap.py` 的 `ensure_local_qlib()` 已经实现了这套逻辑，优先复用它。

### 后果

两份代码有差异时，你会看到**改了本目录的代码却不生效**，或者
**traceback 指向 site-packages 的行号**。看到这两类现象，第一件事就是确认 `qlib.__file__`。

---

## 4. 跑长任务

- 用后台任务 + 轮询，不要用一个前台调用干等；定期输出进度并 `flush=True`。
- 用 `timeout` 包一层，配合 wrapper 的内存上限。
- 输出重定向到文件再读。**不要** `... | tail`——管道整体被 SIGTERM 时 `tail` 可能来不及 flush，
  你会看到"没有任何输出"，误以为脚本没跑。

---

## 5. 相关文档

- 1min 数据接入方案：`sleeve/a/docs/qlib_1min_integration_plan.md`
- 1min exporter / 验证脚本：`sleeve/a/scripts/export_qlib_1min.py`、`sleeve/a/scripts/validate_qlib_1min.py`
- 内存策略原始说明：`sleeve/a/README.md`
