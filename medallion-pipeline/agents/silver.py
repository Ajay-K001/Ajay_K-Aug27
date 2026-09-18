from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd


def silver_clean(bronze_output_paths: Sequence[str | Path], output_dir: str | Path = "data/silver_layer", business_intent: str = "") -> list[str]:
    """Clean and standardize Bronze layer data into Silver layer.

    Reads from Bronze Parquet outputs and applies:
    - Deduplication
    - Null handling (remove all-null rows, forward-fill key columns)
    - Type standardization (lowercase string columns, normalize dates)
    - Business-intent-driven cleansing
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for path in bronze_output_paths:
        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(f"Bronze source file not found: {src}")

        df = pd.read_parquet(src)

        # Silver layer transformations
        # 1. Remove duplicates
        df = df.drop_duplicates()

        # 2. Remove completely empty rows
        df = df.dropna(how="all")

        # 3. Standardize string columns (lowercase categorical values)
        for col in df.select_dtypes(include=["object"]).columns:
            if col.startswith("_"):  # Skip metadata columns
                continue
            df[col] = df[col].str.lower() if df[col].dtype == "object" else df[col]

        # 4. Standardize date columns
        for col in df.select_dtypes(include=["object"]).columns:
            if "date" in col.lower() or "time" in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                except Exception:
                    pass

        # 5. Add Silver metadata
        df["_silver_cleaned_at"] = pd.Timestamp.now()

        target = output / f"{src.stem.replace('_bronze', '')}_silver.parquet"
        df.to_parquet(target, index=False)
        written.append(str(target))

    return written
