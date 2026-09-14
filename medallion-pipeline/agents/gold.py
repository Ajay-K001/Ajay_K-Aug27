from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd


def gold_aggregate(file_paths: Sequence[str | Path], output_dir: str | Path = "data/gold_layer") -> list[str]:
    """Create a simple Gold aggregate output from CSV inputs using Pandas."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for path in file_paths:
        src = Path(path)
        df = pd.read_csv(src)
        target = output / f"{src.stem}_gold.parquet"
        df.to_parquet(target, index=False)
        written.append(str(target))
    return written
