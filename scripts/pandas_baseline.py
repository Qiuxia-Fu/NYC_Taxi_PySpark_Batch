"""
Pandas baseline: 跟 scripts/spark_pipeline.py 做同样的事,用来做 benchmark 对照。

设计说明:朴素写法会把3个月数据先 pd.concat() 成一个大 DataFrame 再处理,
这样整个数据集必须同时放进内存——内存不够的机器上会被 OOM kill。
这里采用更稳健的做法:按月处理,每个月读入->join->过滤->写盘->释放,
只在内存里保留很小的局部聚合结果,内存占用始终只有"一个月的数据量"。
"""
import gc
import glob
import json
import os
import time

import pandas as pd

RAW_DIR = "data/raw"
SILVER_PANDAS_DIR = "data/silver_pandas"
BENCH_DIR = "benchmarks"

# 和 Spark 版本的 TRIP_SCHEMA 对应:显式指定 dtype,而且专门用 float32/category
# 而不是 pandas 默认的 float64/object——这是 pandas 版本能把内存占用压下来的关键手段(Spark 的列式存储、谓词下推是引擎自动帮你做的,pandas 要自己管)。
DTYPES = {
    "VendorID": "category",
    "passenger_count": "float32",
    "trip_distance": "float32",
    "RatecodeID": "category",
    "store_and_fwd_flag": "category",
    "PULocationID": "int32",
    "DOLocationID": "int32",
    "payment_type": "category",
    "fare_amount": "float32",
    "extra": "float32",
    "mta_tax": "float32",
    "tip_amount": "float32",
    "tolls_amount": "float32",
    "improvement_surcharge": "float32",
    "total_amount": "float32",
    "congestion_surcharge": "float32",
}
PARSE_DATES = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]

# 跟 spark_pipeline.py 里的 RATECODE_ROWS 保持完全一致,确保两边结果可比
RATECODE_ROWS = [
    (1, "Standard rate"),
    (2, "JFK"),
    (3, "Newark"),
    (4, "Nassau or Westchester"),
    (5, "Negotiated fare"),
    (6, "Group ride"),
]


def month_from_filename(path: str) -> str:
    base = os.path.basename(path)
    return base.replace("yellow_tripdata_", "").replace(".csv.gz", "")


def main():
    os.makedirs(BENCH_DIR, exist_ok=True)
    t_start = time.time()

    # 维度表只需要读一次,循环外面处理
    zone_lookup = pd.read_csv(os.path.join(RAW_DIR, "taxi_zone_lookup.csv"))
    pu_zone = zone_lookup.rename(columns={
        "LocationID": "PULocationID", "Borough": "pickup_borough", "Zone": "pickup_zone",
    })[["PULocationID", "pickup_borough", "pickup_zone"]]
    do_zone = zone_lookup.rename(columns={
        "LocationID": "DOLocationID", "Borough": "dropoff_borough", "Zone": "dropoff_zone",
    })[["DOLocationID", "dropoff_borough", "dropoff_zone"]]
    ratecode_lookup = pd.DataFrame(RATECODE_ROWS, columns=[
                                   "RatecodeID", "rate_code_desc"])
    ratecode_lookup["RatecodeID"] = ratecode_lookup["RatecodeID"].astype(
        "category")

    files = sorted(glob.glob(os.path.join(
        RAW_DIR, "yellow_tripdata_*.csv.gz")))

    read_elapsed = join_elapsed = filter_elapsed = write_elapsed = 0.0
    bronze_rows = silver_rows = 0
    partial_aggs = []

    for f in files:
        ym = month_from_filename(f)
        year, month = (int(x) for x in ym.split("-"))

        t0 = time.time()
        df = pd.read_csv(
            f, dtype=DTYPES, parse_dates=PARSE_DATES, compression="gzip")
        read_elapsed += time.time() - t0
        month_bronze = len(df)
        bronze_rows += month_bronze
        print(f"[read] {ym}: {month_bronze:,} rows")

        t0 = time.time()
        df = df.merge(pu_zone, on="PULocationID", how="left")
        df = df.merge(do_zone, on="DOLocationID", how="left")
        df = df.merge(ratecode_lookup, on="RatecodeID", how="left")
        join_elapsed += time.time() - t0

        # 和 Silver 层完全一样的数据质量过滤规则,保证两边 apple-to-apple
        t0 = time.time()
        mask = (
            (df["trip_distance"] > 0)
            & (df["fare_amount"] >= 0)
            & (df["total_amount"] > 0)
            & (df["passenger_count"] > 0)
            & df["tpep_pickup_datetime"].notna()
            & df["tpep_dropoff_datetime"].notna()
            & (df["tpep_dropoff_datetime"] > df["tpep_pickup_datetime"])
        )
        df = df[mask].copy()
        filter_elapsed += time.time() - t0
        month_silver = len(df)
        silver_rows += month_silver
        print(f"[filter] {ym}: {month_silver:,} rows remain")

        t0 = time.time()
        part_dir = os.path.join(
            SILVER_PANDAS_DIR, f"pickup_year={year}", f"pickup_month={month}")
        os.makedirs(part_dir, exist_ok=True)
        df.to_parquet(os.path.join(part_dir, "part-0.parquet"),
                      engine="pyarrow", index=False)
        write_elapsed += time.time() - t0

        # 只保留这个月的小聚合结果,原始的几百万行 df 马上释放
        partial = (
            df.groupby("pickup_borough", observed=True)
            .agg(
                trip_count=("fare_amount", "size"),
                fare_sum=("fare_amount", "sum"),
                distance_sum=("trip_distance", "sum"),
                tip_sum=("tip_amount", "sum"),
            )
            .reset_index()
        )
        partial["pickup_year"] = year
        partial["pickup_month"] = month
        partial_aggs.append(partial)

        del df
        gc.collect()

    # 把每个月的小聚合结果合并成最终的 Gold 对照表
    t0 = time.time()
    all_partials = pd.concat(partial_aggs, ignore_index=True)
    summary = (
        all_partials.groupby(
            ["pickup_year", "pickup_month", "pickup_borough"], observed=True)
        .agg(trip_count=("trip_count", "sum"),
             fare_sum=("fare_sum", "sum"),
             distance_sum=("distance_sum", "sum"),
             tip_sum=("tip_sum", "sum"))
        .reset_index()
    )
    summary["avg_fare"] = (summary["fare_sum"] /
                           summary["trip_count"]).round(2)
    summary["avg_distance_miles"] = (
        summary["distance_sum"] / summary["trip_count"]).round(2)
    summary["avg_tip"] = (summary["tip_sum"] / summary["trip_count"]).round(2)
    summary = summary.drop(columns=["fare_sum", "distance_sum", "tip_sum"])
    summary = summary.sort_values(
        ["pickup_year", "pickup_month", "trip_count"], ascending=[True, True, False]
    )
    summary.to_csv(os.path.join(
        BENCH_DIR, "borough_month_summary_pandas.csv"), index=False)
    agg_elapsed = time.time() - t0

    total_elapsed = time.time() - t_start

    result = {
        "engine": "pandas (chunked month-by-month)",
        "pandas_version": pd.__version__,
        "read_seconds": round(read_elapsed, 2),
        "join_seconds": round(join_elapsed, 2),
        "filter_seconds": round(filter_elapsed, 2),
        "write_seconds": round(write_elapsed, 2),
        "aggregation_seconds": round(agg_elapsed, 2),
        "total_seconds": round(total_elapsed, 2),
        "bronze_rows": bronze_rows,
        "silver_rows": silver_rows,
        "rows_dropped_by_quality_filter": bronze_rows - silver_rows,
    }
    with open(os.path.join(BENCH_DIR, "pandas_timing.json"), "w") as fh:
        json.dump(result, fh, indent=2)

    print("=== pandas (chunked) baseline summary ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
