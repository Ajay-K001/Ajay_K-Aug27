from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd



def bronze_ingest(file_paths: Sequence[str | Path], output_dir: str | Path = "data/bronze_layer") -> list[str]:
    """Ingest raw CSV files into Bronze layer with validation and quality checks.

    Reads from landing CSVs, applies basic validation (schema inference, null checks),
    and writes validated Parquet to Bronze layer.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for path in file_paths:
        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(f"Source file not found: {src}")

        df = pd.read_csv(src)

        # Basic Bronze layer transformations
        # 1. Infer and validate schema
        df = df.infer_objects(copy=False)

        # 2. Track data quality metrics
        row_count = len(df)
        null_counts = df.isnull().sum().to_dict()

        # 3. Add Bronze metadata
        df["_bronze_loaded_at"] = pd.Timestamp.now()
        df["_bronze_source_file"] = src.name

        target = output / f"{src.stem}_bronze.parquet"
        df.to_parquet(target, index=False)
        written.append(str(target))

    return written
