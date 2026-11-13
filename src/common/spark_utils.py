"""Tien ich Spark dung chung: tao SparkSession va ghi Parquet (co duong du phong cho Windows)."""
import logging
import os
import re
import shutil
from pathlib import Path

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

logger = logging.getLogger("spark_utils")


def get_spark(app_name: str = "amazon-electronics", shuffle_partitions: int = 64) -> SparkSession:
    """Cau hinh doc tu bien moi truong:
      SPARK_DRIVER_MEMORY : bo nho JVM, dang <so><k|m|g|t> (vd 4g, 6g, 512m). Mac dinh 4g. KHONG dung '6gb'.
      SPARK_MASTER        : so core song song: local[*] (mac dinh), local[4], local[2]...
    Bo nho JVM duoc CHIA cho cac task chay dong thoi, nen nhieu core + it RAM de gay OutOfMemoryError.
    Gia tri sai dinh dang bi bao ro rang (thay vi loi JAVA_GATEWAY_EXITED kho hieu).
    """
    memory = os.environ.get("SPARK_DRIVER_MEMORY", "4g").strip()
    master = os.environ.get("SPARK_MASTER", "local[*]").strip()
    if not re.fullmatch(r"\d+[kKmMgGtT]", memory):
        raise ValueError(f"SPARK_DRIVER_MEMORY='{memory}' khong hop le. Dung dang 4g, 6g hoac 512m (khong co 'b'). "
                         f"PowerShell: $env:SPARK_DRIVER_MEMORY='4g'  |  bo han: Remove-Item Env:SPARK_DRIVER_MEMORY")
    if not re.fullmatch(r"local(\[(\*|\d+)\])?", master):
        raise ValueError(f"SPARK_MASTER='{master}' khong hop le. Dung local[*], local[4] hoac local[2]. "
                         f"PowerShell: $env:SPARK_MASTER='local[4]'  |  bo han: Remove-Item Env:SPARK_MASTER")
    try:
        return (
            SparkSession.builder.master(master).appName(app_name)
            .config("spark.driver.memory", memory)
            .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")   # toPandas() tiet kiem bo nho/nhanh hon
            .getOrCreate()
        )
    except Exception as exc:
        raise RuntimeError(
            f"Khong khoi dong duoc Spark (master={master}, driver memory={memory}): {str(exc).splitlines()[0][:200]}\n"
            "Kiem tra: (1) 'java -version' chay duoc va JAVA_HOME dung; (2) con tien trinh java.exe cu dang chiem RAM "
            "(PowerShell: Get-Process java | Stop-Process); (3) dat SPARK_DRIVER_MEMORY nho hon RAM trong, vi du 3g; "
            "(4) cuon len tren terminal de doc dong 'Error occurred during initialization of VM' (neu co)."
        ) from exc


def storage_level() -> StorageLevel:
    """Muc cache cho cac buoc trung gian (dem so dong sau tung quy tac clean).
      SPARK_STORAGE_LEVEL=MEMORY_AND_DISK (mac dinh) : uu tien RAM, tran xuong ddia
      SPARK_STORAGE_LEVEL=DISK_ONLY                  : chi dung dia - khong chiem RAM, hop voi may dang dung RAM cho viec khac
    """
    name = os.environ.get("SPARK_STORAGE_LEVEL", "MEMORY_AND_DISK").strip().upper()
    if name not in ("MEMORY_AND_DISK", "DISK_ONLY"):
        raise ValueError(f"SPARK_STORAGE_LEVEL='{name}' khong hop le. Dung MEMORY_AND_DISK hoac DISK_ONLY.")
    return getattr(StorageLevel, name)


def write_parquet(df: DataFrame, path: str, writer: str = "auto", num_files: int = None) -> str:
    """Ghi DataFrame ra thu muc Parquet. Tra ve writer thuc su da dung.

    writer="spark"  : dung Spark/Hadoop (Linux, WSL2, GCP).
    writer="pandas" : ghi bang pyarrow, khong can Hadoop/winutils. Neu num_files > 1, du lieu duoc chia
                      thanh num_files phan va moi phan duoc toPandas() rieng => khong phai nap het vao RAM.
    writer="auto"   : thu Spark truoc; loi (thuong la Hadoop tren Windows) thi chuyen sang pandas.
    """
    if writer in ("spark", "auto"):
        try:
            out = df.coalesce(num_files) if num_files else df
            out.write.mode("overwrite").parquet(path)
            return "spark"
        except Exception as exc:  # Py4JJavaError, ...
            if writer == "spark":
                raise
            logger.warning("Ghi Parquet bang Spark that bai (%s). Chuyen sang pandas/pyarrow.",
                           str(exc).splitlines()[0][:150])
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    k = max(1, num_files or 1)
    written = 0
    if k > 1:
        parts = df.repartition(k).persist(storage_level())
        try:
            for i in range(k):
                chunk = parts.where(F.spark_partition_id() == i).toPandas()
                if len(chunk):
                    chunk.to_parquet(os.path.join(path, f"part-{i:05d}.parquet"), index=False)
                    written += 1
        finally:
            parts.unpersist()
    if written == 0:
        df.toPandas().to_parquet(os.path.join(path, "part-00000.parquet"), index=False)
    return "pandas"


def parquet_files(path: str):
    """Danh sach file .parquet trong thu muc local (bo qua file an/_SUCCESS). Nem loi ro rang neu khong co."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Khong thay duong dan Parquet: {path} (thu muc hien tai: {os.getcwd()})")
    if p.is_file():
        return [p]
    files = sorted(f for f in p.rglob("*.parquet") if not f.name.startswith((".", "_")))
    if not files:
        raise FileNotFoundError(f"Thu muc {path} khong co file .parquet nao: {sorted(x.name for x in p.iterdir())[:10]}")
    return files


def read_parquet(spark: SparkSession, path: str) -> DataFrame:
    """Doc Parquet tu THU MUC (tat ca cac file part-*.parquet) hoac tu duong dan gs://...

    Voi duong dan local, tu liet ke cac file .parquet bang Python roi dua danh sach file cho Spark
    (khong phu thuoc viec Hadoop liet ke thu muc - hay loi tren Windows).
    Neu nhan vao 1 FILE: van doc, nhung canh bao vi chi co MOT PHAN du lieu.
    """
    if "://" in path:
        return spark.read.parquet(path)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Khong thay duong dan Parquet: {path} (thu muc hien tai: {os.getcwd()})")
    if p.is_file():
        logger.warning("Dang doc 1 FILE Parquet (%s), khong phai ca thu muc: chi co MOT PHAN du lieu. "
                       "Hay truyen duong dan THU MUC chua tat ca cac part.", path)
        return spark.read.parquet(str(p))
    files = sorted(f for f in p.rglob("*.parquet") if not f.name.startswith((".", "_")))
    if not files:
        raise FileNotFoundError(f"Thu muc {path} khong co file .parquet nao: {sorted(x.name for x in p.iterdir())[:10]}")
    logger.info("Doc %d file Parquet tu %s", len(files), path)
    return spark.read.parquet(*[str(f) for f in files])