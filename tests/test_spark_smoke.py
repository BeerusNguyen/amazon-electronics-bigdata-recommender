from pyspark.sql import SparkSession


def main():
    spark = (SparkSession.builder.master("local[2]")
             .appName("amazon-electronics-smoke")
             .getOrCreate())
    try:
        spark.sparkContext.setLogLevel("ERROR")
        df = spark.createDataFrame(
            [("U001", "P001", 5.0), ("U002", "P002", 4.0)],
            ["user_id", "product_id", "rating"],
        )
        assert df.count() == 2
        print("Spark version:", spark.version)
        print("Smoke test: PASS")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
