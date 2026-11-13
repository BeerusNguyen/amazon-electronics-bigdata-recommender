"""Reader dung chung cho metadata (T06 profile, T12 clean).

Chi khai bao 6 truong can dung. Cac truong khac (rank - kieu hon hop list/str, details - dict
thay doi, imageURL, also_buy, ...) bi Spark bo qua nen khong gay loi kieu du lieu.

Quyet dinh chon truong (dua tren khao sat 5000 dong dau, xem docs):
  giu  : asin, title, brand, main_cat, category, price (price chi de PROFILE, khong dua vao ban clean)
  bo   : rank, details, tech1/tech2, fit, similar_item, date, feature, description,
         also_buy, also_view, imageURL, imageURLHighRes
"""
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import types as T

META_FIELDS = [
    T.StructField("asin", T.StringType(), True),
    T.StructField("title", T.StringType(), True),
    T.StructField("brand", T.StringType(), True),
    T.StructField("main_cat", T.StringType(), True),
    T.StructField("category", T.ArrayType(T.StringType()), True),
    T.StructField("price", T.StringType(), True),
]
META_SCHEMA = T.StructType(META_FIELDS)
CORRUPT_COL = "_corrupt_record"


def read_metadata(spark: SparkSession, path: str, with_corrupt: bool = False) -> DataFrame:
    """Doc meta_Electronics.json.gz. with_corrupt=True them cot _corrupt_record de dem dong hong."""
    fields = META_FIELDS + ([T.StructField(CORRUPT_COL, T.StringType(), True)] if with_corrupt else [])
    reader = spark.read.schema(T.StructType(fields)).option("mode", "PERMISSIVE")
    if with_corrupt:
        reader = reader.option("columnNameOfCorruptRecord", CORRUPT_COL)
    return reader.json(path)
