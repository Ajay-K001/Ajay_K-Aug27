from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd



def bronze_ingest(file_paths: Sequence[str | Path], output_dir: str | Path = "data/bronze_layer") -> list[str]:
    """Ingest CSV files and create Bronze-layer Parquet placeholders in a simple workflow."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for path in file_paths:
        src = Path(path)
        df = pd.read_csv(src)
        target = output / f"{src.stem}_bronze.parquet"
        df.to_parquet(target, index=False)
        written.append(str(target))
    return written
