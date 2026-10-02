from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, IntegerType, DoubleType, StringType, TimestampType
)
from pyspark.sql import functions as F
import glob
import os
import time
import json

# 显式定义schema，对应CSV的18列
TRIP_SCHEMA = StructType([
    StructField("VendorID", IntegerType(), True),
    StructField("tpep_pickup_datetime", TimestampType(), True),
    StructField("tpep_dropoff_datetime", TimestampType(), True),
    StructField("passenger_count", IntegerType(), True),
    StructField("trip_distance", DoubleType(), True),
    StructField("RatecodeID", IntegerType(), True),
    StructField("store_and_fwd_flag", StringType(), True),
    StructField("PULocationID", IntegerType(), True),
    StructField("DOLocationID", IntegerType(), True),
    StructField("payment_type", IntegerType(), True),
    StructField("fare_amount", DoubleType(), True),
    StructField("extra", DoubleType(), True),
    StructField("mta_tax", DoubleType(), True),
    StructField("tip_amount", DoubleType(), True),
    StructField("tolls_amount", DoubleType(), True),
    StructField("improvement_surcharge", DoubleType(), True),
    StructField("total_amount", DoubleType(), True),
    StructField("congestion_surcharge", DoubleType(), True),
])

spark = (
    SparkSession.builder
    .appName("nyc-taxi-batch-pipeline")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

t_start = time.time()

RAW_DIR = "data/raw"
BRONZE_DIR = "data/bronze"


def month_from_filename(path: str) -> str:
    # yellow_tripdata_2019-01.csv.gz -> "2019-01"
    base = os.path.basename(path)
    return base.replace("yellow_tripdata_", "").replace(".csv.gz", "")


files = sorted(glob.glob(os.path.join(RAW_DIR, "yellow_tripdata_*.csv.gz")))

t0 = time.time()
for f in files:
    ym = month_from_filename(f)          # "2019-01"
    year, month = ym.split("-")

    df = (
        spark.read.csv(f, header=True, schema=TRIP_SCHEMA)
        .withColumn("pickup_year", F.lit(int(year)))
        .withColumn("pickup_month", F.lit(int(month)))
    )

    out_path = os.path.join(BRONZE_DIR, f"yellow_tripdata_{ym}")
    df.write.mode("overwrite").parquet(out_path)
    print(f"[bronze] {ym}: write {df.count():,} rows -> {out_path}")
bronze_elapsed = time.time() - t0


BRONZE_GLOB = "data/bronze/yellow_tripdata_*"

t0 = time.time()
bronze = spark.read.parquet(BRONZE_GLOB)

# 维度表1:zone lookup(265行)
zone_lookup = (
    spark.read.csv("data/raw/taxi_zone_lookup.csv",
                   header=True, inferSchema=True)
    .select("LocationID", "Borough", "Zone", "service_zone")
)

# PULocationID和DOLocationID都指向同一张zone表,所以要分别重命名、join两次
pu_zone = zone_lookup.select(
    F.col("LocationID").alias("PULocationID"),
    F.col("Borough").alias("pickup_borough"),
    F.col("Zone").alias("pickup_zone"),
)
do_zone = zone_lookup.select(
    F.col("LocationID").alias("DOLocationID"),
    F.col("Borough").alias("dropoff_borough"),
    F.col("Zone").alias("dropoff_zone"),
)

# 维度表2:ratecode lookup(官方只有6种,手写一个小表,不用额外下载文件)
RATECODE_ROWS = [
    (1, "Standard rate"),
    (2, "JFK"),
    (3, "Newark"),
    (4, "Nassau or Westchester"),
    (5, "Negotiated fare"),
    (6, "Group ride"),
]
ratecode_lookup = spark.createDataFrame(
    RATECODE_ROWS, ["RatecodeID", "rate_code_desc"])

enriched = (
    bronze
    .join(F.broadcast(pu_zone), on="PULocationID", how="left")
    .join(F.broadcast(do_zone), on="DOLocationID", how="left")
    .join(F.broadcast(ratecode_lookup), on="RatecodeID", how="left")
)

enriched.select(
    "PULocationID", "pickup_borough", "pickup_zone",
    "DOLocationID", "dropoff_borough", "dropoff_zone",
    "RatecodeID", "rate_code_desc",
).show(5, truncate=False)


cleaned = enriched.filter(
    (F.col("trip_distance") > 0)
    & (F.col("fare_amount") >= 0)
    & (F.col("total_amount") > 0)
    & (F.col("passenger_count") > 0)
    & F.col("tpep_pickup_datetime").isNotNull()
    & F.col("tpep_dropoff_datetime").isNotNull()
    & (F.col("tpep_dropoff_datetime") > F.col("tpep_pickup_datetime"))
)

silver_path = "data/silver/yellow_tripdata"
(
    cleaned.write
    .mode("overwrite")
    .partitionBy("pickup_year", "pickup_month")
    .parquet(silver_path)
)

bronze_count = bronze.count()
silver_count = cleaned.count()
print(f"[silver] bronze rows: {bronze_count:,}")
print(f"[silver] silver rows(filtered): {silver_count:,}")
print(f"[silver] filtered out: {bronze_count - silver_count:,} rows")

silver_elapsed = time.time() - t0

t0 = time.time()
summary = (
    cleaned.groupBy("pickup_year", "pickup_month", "pickup_borough")
    .agg(
        F.count("*").alias("trip_count"),
        F.round(F.avg("fare_amount"), 2).alias("avg_fare"),
        F.round(F.avg("trip_distance"), 2).alias("avg_distance_miles"),
        F.round(F.avg("tip_amount"), 2).alias("avg_tip"),
    )
    .orderBy("pickup_year", "pickup_month", F.desc("trip_count"))
)

summary.show(25, truncate=False)

summary_path = "benchmarks/borough_month_summary"
summary.coalesce(1).write.mode("overwrite").option(
    "header", True).csv(summary_path)
gold_elapsed = time.time() - t0

total_elapsed = time.time() - t_start

result = {
    "engine": "pyspark",
    "spark_version": spark.version,
    "master": "local[*]",
    "bronze_ingestion_seconds": round(bronze_elapsed, 2),
    "silver_build_seconds": round(silver_elapsed, 2),
    "gold_aggregation_seconds": round(gold_elapsed, 2),
    "total_seconds": round(total_elapsed, 2),
    "bronze_rows": bronze_count,
    "silver_rows": silver_count,
    "rows_dropped_by_quality_filter": bronze_count - silver_count,
}
with open("benchmarks/pyspark_timing.json", "w") as fh:
    json.dump(result, fh, indent=2)

print(json.dumps(result, indent=2))

spark.stop()
