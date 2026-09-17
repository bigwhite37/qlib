#!/usr/bin/env python3
"""ETF 1-minute dataset inventory + stock-universe survivorship check."""

from __future__ import annotations

import glob

import duckdb


def main() -> None:
    files = sorted(glob.glob("/Volumes/lexar_4t/code/data/etf_1m_ad_v1/bars/year=*/month=*/exchange=*/*.parquet"))
    print("total bar files:", len(files))
    con = duckdb.connect(config={"memory_limit": "2GB"})
    info = con.execute(
        "select count(*) as n_rows, count(distinct etf_code) as n_syms, "
        "min(trade_date) as lo, max(trade_date) as hi from read_parquet(?)",
        [files],
    ).df()
    print(info.to_string())
    year_counts = con.execute(
        "select year, count(*) as n_rows, count(distinct etf_code) as n_syms "
        "from read_parquet(?) group by 1 order by 1",
        [files],
    ).df()
    print(year_counts.to_string(index=False))
    syms = con.execute(
        "select distinct etf_code from read_parquet(?) order by 1", [files]
    ).df()
    print("universe (%d):" % len(syms), syms.etf_code.tolist())
    minutes = con.execute(
        "select count(distinct minute) as n_minutes, min(minute) as first_min, max(minute) as last_min "
        "from read_parquet(?)",
        [files],
    ).df()
    print(minutes.to_string(index=False))
    con.close()

    con = duckdb.connect(
        "/Users/shuzhenyi/code/data/qlib_tmp_correct/qlib_tmp_correct_reindexed.duckdb",
        read_only=True,
        config={"memory_limit": "2GB"},
    )
    print()
    print("== stock instruments ==")
    print(
        con.execute(
            "select count(*) as n_rows, count(distinct qlib_symbol) as n_syms, "
            "sum(case when end_date < 20260914 then 1 else 0 end) as ended_before_today, "
            "min(start_date) as lo, max(end_date) as hi from qlib_instruments"
        ).df().to_string(index=False)
    )
    con.close()


if __name__ == "__main__":
    main()
