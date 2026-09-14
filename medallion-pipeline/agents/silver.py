from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd


def silver_clean(file_paths: Sequence[str | Path], output_dir: str | Path = "data/silver_layer") -> list[str]:
    """Create a simple Silver-cleansed parquet side-effect using pandas."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for path in file_paths:
        src = Path(path)
        df = pd.read_csv(src)
        df = df.drop_duplicates().dropna(how="all")
        target = output / f"{src.stem}_silver.parquet"
        df.to_parquet(target, index=False)
        written.append(str(target))
    return written
