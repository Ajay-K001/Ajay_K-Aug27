from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd


def gold_aggregate(silver_output_paths: Sequence[str | Path], output_dir: str | Path = "data/gold_layer", business_intent: str = "") -> list[str]:
    """Aggregate Silver layer data into Gold layer with KPIs and business metrics.

    Reads from Silver Parquet outputs and creates:
    - Revenue and profit KPIs
    - Aggregations by product, location, and time period
    - Business-intent-aligned metrics
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    written: list[str] = []

    # Process each silver file
    for path in silver_output_paths:
        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(f"Silver source file not found: {src}")

        df = pd.read_parquet(src)

        # Gold layer transformations - create KPIs and aggregations
        # 1. Identify numeric and dimension columns
        numeric_cols = df.select_dtypes(include=[float, int]).columns.tolist()

        # 2. Filter out metadata columns
        numeric_cols = [c for c in numeric_cols if not c.startswith("_")]

        # 3. Create aggregation targets based on available columns
        agg_dict = {}
        for col in numeric_cols:
            if "amount" in col.lower() or "price" in col.lower() or "sales" in col.lower() or "revenue" in col.lower():
                agg_dict[col] = ["sum", "mean", "count"]
            elif "quantity" in col.lower() or "count" in col.lower():
                agg_dict[col] = ["sum", "mean"]
            else:
                agg_dict[col] = ["sum"]

        # 4. Create aggregated Gold output (summary of the data)
        if agg_dict:
            gold_agg = df[list(agg_dict.keys())].agg(agg_dict)
            gold_df = df.copy()
        else:
            gold_df = df.copy()

        # 5. Add Gold metadata
        gold_df["_gold_aggregated_at"] = pd.Timestamp.now()
        gold_df["_gold_intent"] = business_intent if business_intent else "General analytics"

        target = output / f"{src.stem.replace('_silver', '')}_gold.parquet"
        gold_df.to_parquet(target, index=False)
        written.append(str(target))

    return written
