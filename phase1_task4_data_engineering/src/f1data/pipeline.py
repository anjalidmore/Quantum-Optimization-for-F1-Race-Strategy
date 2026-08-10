"""
f1data.pipeline
===============

The data-engineering pipeline: load → clean → transform, with a data-quality
audit recorded at every step so the process is transparent and reportable.

Stages
------
1. **Load** — read the CSVs, validating that required columns are present, and
   coerce the Ergast ``"\\N"`` sentinel to proper NaN.
2. **Clean** — remove duplicate rows, coerce dtypes, parse ``m:ss.mmm`` lap-time
   strings to seconds, normalise inconsistent categorical spellings, and impute
   or drop missing values by column kind.
3. **Detect outliers** — flag numeric outliers via the IQR rule (and report,
   optionally cap, them).
4. **Encode** — label/one-hot encode categoricals for downstream ML.
5. **Scale** — standardise numeric features.

Each stage appends a :class:`StepReport` to a :class:`CleaningReport`, giving a
full before/after audit trail.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .schemas import ColKind, TableSchema

_NULL_SENTINELS = (r"\N", "\\N", "", "NaN", "nan", "None")
_LAPTIME_RE = re.compile(r"^\s*(?:(\d+):)?(\d+(?:\.\d+)?)\s*$")


@dataclass
class StepReport:
    """A record of what a single pipeline step did."""

    step: str
    detail: str
    rows_before: int
    rows_after: int
    extra: Dict[str, object] = field(default_factory=dict)


@dataclass
class CleaningReport:
    """The full audit trail for cleaning one table."""

    table: str
    steps: List[StepReport] = field(default_factory=list)

    def add(self, step: str, detail: str, before: int, after: int, **extra) -> None:
        self.steps.append(StepReport(step, detail, before, after, dict(extra)))


def parse_laptime_to_seconds(value: object) -> float:
    """
    Convert an Ergast-style ``m:ss.mmm`` (or plain seconds) string to float
    seconds. Returns NaN for unparseable / missing values.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")
    s = str(value).strip()
    if s in _NULL_SENTINELS:
        return float("nan")
    m = _LAPTIME_RE.match(s)
    if not m:
        try:
            return float(s)
        except ValueError:
            return float("nan")
    minutes = float(m.group(1)) if m.group(1) else 0.0
    seconds = float(m.group(2))
    return minutes * 60.0 + seconds


def load_csv(path: str | Path, schema: TableSchema,
             require_all_columns: bool = False) -> pd.DataFrame:
    """
    Load a CSV, converting Ergast null sentinels to NaN and validating columns.

    If ``require_all_columns`` is True, a missing schema column raises; otherwise
    only a warning-worthy subset is required (the primary key).
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    # Convert sentinels to NaN uniformly.
    df = df.replace(list(_NULL_SENTINELS), np.nan)

    present = set(df.columns)
    expected = set(schema.column_names())
    missing = expected - present
    if require_all_columns and missing:
        raise ValueError(f"{schema.name}: missing columns {sorted(missing)}")
    missing_pk = set(schema.primary_key) - present
    if missing_pk:
        raise ValueError(f"{schema.name}: missing primary-key columns {sorted(missing_pk)}")
    return df


def clean_table(df: pd.DataFrame, schema: TableSchema,
                cap_outliers: bool = False) -> Tuple[pd.DataFrame, CleaningReport]:
    """
    Run the clean stage on ``df`` given its ``schema``.

    Returns the cleaned frame and a :class:`CleaningReport`.
    """
    report = CleaningReport(table=schema.name)
    kinds = {c.name: c.kind for c in schema.columns}

    # --- 1. Deduplicate ---------------------------------------------------
    before = len(df)
    df = df.drop_duplicates(ignore_index=True)
    report.add("deduplicate", "Removed exact duplicate rows.", before, len(df),
               removed=before - len(df))

    # --- 2. Type coercion + lap-time parsing ------------------------------
    for col in df.columns:
        kind = kinds.get(col)
        if kind is ColKind.NUMERIC:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif kind is ColKind.TIME:
            # Duration/lap-time strings -> seconds (float).
            df[col] = df[col].map(parse_laptime_to_seconds)
        elif kind is ColKind.DATE:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    report.add("coerce_types", "Coerced numeric/time/date columns; parsed lap "
               "times to seconds.", len(df), len(df))

    # --- 3. Normalise categorical spellings -------------------------------
    cat_cols = [c for c, k in kinds.items() if k is ColKind.CATEGORICAL and c in df.columns]
    for col in cat_cols:
        df[col] = df[col].astype("string").str.strip().str.upper()
    report.add("normalise_categoricals",
               f"Trimmed/upper-cased {len(cat_cols)} categorical column(s).",
               len(df), len(df), columns=cat_cols)

    # --- 4. Missing-value handling ----------------------------------------
    na_before = int(df.isna().sum().sum())
    for col in df.columns:
        kind = kinds.get(col)
        if col not in df.columns:
            continue
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            continue
        if kind is ColKind.NUMERIC or kind is ColKind.TIME:
            median = df[col].median()
            df[col] = df[col].fillna(median)
        elif kind is ColKind.CATEGORICAL:
            mode = df[col].mode(dropna=True)
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "UNKNOWN")
        # IDs / dates with missing values are left as NaN (not imputable safely).
    na_after = int(df.isna().sum().sum())
    report.add("impute_missing",
               "Imputed numeric/time by median, categorical by mode.",
               len(df), len(df), na_before=na_before, na_after=na_after)

    # --- 5. Outlier detection (IQR) ---------------------------------------
    numeric_cols = [c for c, k in kinds.items()
                    if k in (ColKind.NUMERIC, ColKind.TIME) and c in df.columns]
    outlier_counts: Dict[str, int] = {}
    for col in numeric_cols:
        series = df[col].astype(float)
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr <= 0:
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (series < lo) | (series > hi)
        outlier_counts[col] = int(mask.sum())
        if cap_outliers and mask.any():
            # Ensure the column is float so numeric bounds can be written back
            # (an int64 column cannot hold a fractional cap value).
            df[col] = series
            df.loc[mask & (series < lo), col] = lo
            df.loc[mask & (series > hi), col] = hi
    report.add("detect_outliers",
               ("Capped" if cap_outliers else "Flagged") + " IQR outliers.",
               len(df), len(df), outliers=outlier_counts)

    return df, report


@dataclass
class EncodingResult:
    """Result of encoding + scaling a feature frame."""

    frame: pd.DataFrame
    label_mappings: Dict[str, Dict[str, int]]
    scaled_columns: List[str]


def encode_and_scale(df: pd.DataFrame, schema: TableSchema,
                     one_hot_max_cardinality: int = 12) -> EncodingResult:
    """
    Label-encode (or one-hot for low cardinality) categoricals and standardise
    numeric columns. IDs and text columns are left untouched.
    """
    kinds = {c.name: c.kind for c in schema.columns}
    out = df.copy()
    label_mappings: Dict[str, Dict[str, int]] = {}

    cat_cols = [c for c, k in kinds.items()
                if k is ColKind.CATEGORICAL and c in out.columns]
    for col in cat_cols:
        values = out[col].astype("string").fillna("UNKNOWN")
        cardinality = values.nunique()
        if cardinality <= one_hot_max_cardinality:
            dummies = pd.get_dummies(values, prefix=col)
            out = pd.concat([out.drop(columns=[col]), dummies], axis=1)
        else:
            categories = sorted(values.unique())
            mapping = {v: i for i, v in enumerate(categories)}
            out[col] = values.map(mapping).astype(int)
            label_mappings[col] = mapping

    numeric_cols = [c for c, k in kinds.items()
                    if k in (ColKind.NUMERIC, ColKind.TIME) and c in out.columns]
    scaled_cols: List[str] = []
    if numeric_cols:
        scaler = StandardScaler()
        valid = out[numeric_cols].astype(float)
        # Only scale columns with non-zero variance.
        varying = [c for c in numeric_cols if valid[c].std(ddof=0) > 0]
        if varying:
            out[varying] = scaler.fit_transform(valid[varying])
            scaled_cols = varying

    return EncodingResult(out, label_mappings, scaled_cols)
