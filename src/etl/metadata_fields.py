"""KHUNG KHAI BAO CAC TRUONG METADATA - noi DUY NHAT can sua khi them/bot truong.

Moi truong la 1 dong FieldSpec trong danh sach FIELDS (cuoi file). Khai bao xong thi cac buoc sau TU DONG ap dung:
  read_metadata    : them truong vao schema doc JSON
  clean_metadata   : lam sach + (tuy chon) gop duplicate/bien the + dien null; them cot vao Parquet clean
  profile_metadata : do blank/null/HTML cua truong
  write_parquet    : schema ky vong khi kiem tra Parquet
  join_metadata    : cot di vao bang ratings_enriched va bao cao coverage

===================== CACH THEM MOT TRUONG =====================
1) Truong chuoi don gian (vd "date"):
       FieldSpec("date"),                     # tu dong: bo the HTML, giai ma &amp;, gop khoang trang, rong -> null

2) Truong danh sach chuoi (vd "description", "feature"):
       FieldSpec("description", dtype=T.ArrayType(T.StringType())),

3) Truong co luat lam sach rieng: viet ham Column -> Column roi truyen vao cleaner=
       def clean_model(c: Column) -> Column:
           x = clean_text(c)                                   # tai dung buoc co ban
           return F.upper(F.regexp_replace(x, r"\\s+", ""))      # vd: "ab 12" -> "AB12"
       FieldSpec("model", cleaner=clean_model),

4) Gop bien the viet khac nhau (giong brand: SONY/Sony/sony -> Sony):
       FieldSpec("main_cat", canonicalize_variants=True),

5) Dien gia tri khi van con null sau khi lam sach va gop tu ban ghi trung ASIN:
       FieldSpec("brand", ..., fill_null="Unknown"),

6) Chi doc de profile, KHONG dua vao ban clean:
       FieldSpec("price", keep=False),

Cac tuy chon khac: completeness=True (truong nay tinh vao do "day du" khi chon ban ghi tot nhat cho ASIN trung),
merge_duplicates=False (khong dien tu ban ghi trung), keep_raw=True (giu them cot <ten>_raw = gia tri goc).

LUU Y: truong co KIEU HON HOP trong file goc (vd "rank" luc la list luc la str, "details" la dict doi cau truc)
khong khai bao truc tiep duoc. Dung scripts/peek_jsonl.py de xem kieu truoc khi them.
Sau khi doi FIELDS: chay lai profile_metadata.py -> clean_metadata.py -> write_parquet.py (schema Parquet doi).
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

KEY = "asin"

# ----------------------------------------------------------------------------------------------
# Ham lam sach dung chung (Column -> Column)
# ----------------------------------------------------------------------------------------------
_ENTITIES = [("&nbsp;", " "), ("&quot;", '"'), ("&#39;", "'"), ("&lt;", "<"), ("&gt;", ">"), ("&amp;", "&")]


def clean_text(c: Column) -> Column:
    """Bo the HTML, giai ma entity co ban, gop khoang trang; chuoi rong -> null."""
    x = F.regexp_replace(c, r"<[^>]*>", " ")
    for ent, rep in _ENTITIES:          # &amp; de cuoi cung de tranh giai ma 2 lan
        x = F.regexp_replace(x, ent, rep)
    x = F.trim(F.regexp_replace(x, r"\s+", " "))
    return F.when(x == "", F.lit(None)).otherwise(x)


def clean_text_array(c: Column) -> Column:
    """Lam sach tung phan tu cua mang chuoi, bo phan tu rong; null -> mang rong."""
    arr = F.coalesce(c, F.array().cast("array<string>"))
    return F.filter(F.transform(arr, clean_text), lambda x: x.isNotNull())


# --- brand ---------------------------------------------------------------------------------------
# Gia tri "giu cho" duoc coi la null. Chinh sua danh sach nay sau khi xem
# results/metadata_profile.json -> removed_by_cleaning_top10 cua brand.
BRAND_PLACEHOLDERS = ("n/a", "na", "none", "null", "unknown", "unbranded", "no brand", "not available", "tbd", "?", "-", "--")
_EDGE = r"[\s\-\u2013\u2014_.,;:'\"`*/\\|]+"     # dau cau/khoang trang o hai dau chuoi


def clean_brand(c: Column) -> Column:
    x = clean_text(c)
    x = F.regexp_replace(x, r"(?i)^visit\s+(?:the\s+|amazon'?s\s+)?(.+?)\s+(?:store|page)$", "$1")  # "Visit the Sony Store"
    x = F.regexp_replace(x, r"(?i)^brand\s*:\s*(.+)$", "$1")                                          # "Brand: Sony"
    x = F.regexp_replace(x, "[\u00ae\u2122\u00a9\u2120]", "")                                          # ® ™ © ℠
    x = F.regexp_replace(x, "^" + _EDGE, "")
    x = F.regexp_replace(x, _EDGE + "$", "")
    x = F.trim(F.regexp_replace(x, r"\s+", " "))
    return F.when((x == "") | F.lower(x).isin(*BRAND_PLACEHOLDERS), F.lit(None)).otherwise(x)


# ----------------------------------------------------------------------------------------------
# Khai bao truong
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class FieldSpec:
    name: str
    dtype: T.DataType = field(default_factory=T.StringType)
    cleaner: Optional[Callable[[Column], Column]] = None   # None -> clean_text / clean_text_array theo kieu
    keep: bool = True                  # False: chi doc de profile
    completeness: bool = False         # tinh vao do day du khi chon ban ghi tot nhat cho ASIN trung
    merge_duplicates: bool = True      # dien gia tri thieu tu cac ban ghi trung ASIN
    keep_raw: bool = False             # them cot <name>_raw (chi cho truong chuoi)
    canonicalize_variants: bool = False  # gop bien the viet khac nhau (chi cho truong chuoi)
    fill_null: Optional[str] = None    # dien gia tri nay neu van thieu (chi cho truong chuoi)

    @property
    def is_array(self) -> bool:
        return isinstance(self.dtype, T.ArrayType)

    def clean(self, c: Column) -> Column:
        if self.cleaner is not None:
            return self.cleaner(c)
        return clean_text_array(c) if self.is_array else clean_text(c)

    def is_missing(self, c: Column) -> Column:
        """Cot dieu kien: gia tri null / chuoi rong / mang rong."""
        return (c.isNull() | (F.size(c) == 0)) if self.is_array else (c.isNull() | (F.trim(c) == ""))


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  THEM / SUA TRUONG O DAY  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
FIELDS = [
    FieldSpec("title", completeness=True),
    FieldSpec("brand", cleaner=clean_brand, completeness=True, keep_raw=True, canonicalize_variants=True),
    FieldSpec("main_cat"),
    FieldSpec("category", dtype=T.ArrayType(T.StringType()), completeness=True),
    FieldSpec("price", keep=False),     # 65% rong, khong dung lam feature: chi profile
]

# Cot dan xuat (tinh tu cac cot da lam sach): (ten, kieu, ham tao Column, cac truong bat buoc phai co)
DERIVED = [
    ("category_path", T.StringType(),
     lambda: F.when(F.size("category") > 0, F.concat_ws(" > ", "category")), ("category",)),
]
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


def kept_fields():
    return [s for s in FIELDS if s.keep]


def read_struct_fields():
    """StructField cho schema doc JSON (asin + moi truong khai bao, ke ca keep=False)."""
    return [T.StructField(KEY, T.StringType(), True)] + [T.StructField(s.name, s.dtype, True) for s in FIELDS]


def active_derived(field_names):
    """Cac cot dan xuat co du truong nguon trong field_names."""
    names = set(field_names)
    return [(n, dt, b) for n, dt, b, req in DERIVED if set(req) <= names]


def output_columns(specs=None):
    """Danh sach (ten cot, kieu dang chuoi) theo DUNG THU TU cua Parquet clean."""
    specs = kept_fields() if specs is None else specs
    cols = [(KEY, "string")]
    for s in specs:
        cols.append((s.name, s.dtype.simpleString()))
        if s.keep_raw:
            cols.append((f"{s.name}_raw", "string"))
    cols += [(n, dt.simpleString()) for n, dt, _ in active_derived(s.name for s in specs)]
    return cols


# ----------------------------------------------------------------------------------------------
# Gop bien the viet khac nhau cua cung mot gia tri (muc DATASET, khong phai muc dong)
# ----------------------------------------------------------------------------------------------
def variant_key(c: Column) -> Column:
    """Khoa so sanh: chu thuong, ky tu khong phai chu/so -> khoang trang. 'D-Link' == 'D-LINK' == 'd link'."""
    k = F.trim(F.regexp_replace(F.lower(c), r"[^\p{L}\p{N}]+", " "))
    return F.when(k == "", F.lower(c)).otherwise(k)


def canonicalize_variants(df: DataFrame, col: str):
    """Moi nhom bien the (cung variant_key) duoc thay bang dang XUAT HIEN NHIEU NHAT.
    Hoa luc: uu tien dang khong viet HOA toan bo, roi dang ngan hon, roi theo thu tu chu cai.
    Tra ve (df moi, stats)."""
    key = f"_key_{col}"
    keyed = df.withColumn(key, variant_key(F.col(col)))
    counts = keyed.where(F.col(col).isNotNull()).groupBy(key, col).count()
    w = Window.partitionBy(key).orderBy(F.col("count").desc(),
                                        (F.col(col) == F.upper(F.col(col))).cast("int"),
                                        F.length(col), F.col(col))
    canon = (counts.withColumn("_rn", F.row_number().over(w)).where("_rn = 1")
             .select(key, F.col(col).alias("_canon")).persist())
    out = (keyed.join(canon, key, "left")
           .withColumn(col, F.coalesce(F.col("_canon"), F.col(col))).drop(key, "_canon"))

    joined = counts.join(canon, key)
    rows_changed = joined.where(F.col(col) != F.col("_canon")).agg(F.sum("count")).first()[0] or 0
    groups = (joined.groupBy(key)
              .agg(F.sum("count").alias("total"), F.first("_canon").alias("canonical"),
                   F.collect_list(F.struct(F.col(col).alias("value"), "count")).alias("variants"),
                   F.count(F.lit(1)).alias("n_variants"))
              .where("n_variants > 1"))
    examples = [dict(canonical=r["canonical"], total_rows=int(r["total"]),
                     variants=[dict(value=v["value"], count=int(v["count"])) for v in r["variants"]])
                for r in groups.orderBy(F.desc("total")).limit(10).collect()]
    stats = dict(field=col, distinct_before=counts.count(), distinct_after=canon.count(),
                 rows_changed=int(rows_changed), variant_groups=groups.count(), examples=examples)
    return out, stats