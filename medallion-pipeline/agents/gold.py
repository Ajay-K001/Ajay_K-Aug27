from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from core.config import (
    LLM_PROVIDER,
    GROQ_API_KEY,
    GROQ_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    GOOGLE_API_KEY,
    GEMINI_MODEL,
    GITHUB_TOKEN,
    GITHUB_MODEL,
    GITHUB_BASE_URL,
)


def _make_llm() -> Any:
    """Create the configured LLM client."""
    provider = (LLM_PROVIDER or "").strip().lower()

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            reasoning_format="hidden",
            reasoning_effort="low",
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(api_key=OPENAI_API_KEY, model=OPENAI_MODEL)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY, model=GEMINI_MODEL)

    if provider == "github":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=GITHUB_TOKEN,
            model=GITHUB_MODEL,
            base_url=GITHUB_BASE_URL,
        )

    return None


def gold_aggregate(silver_output_paths: Sequence[str | Path], output_dir: str | Path = "data/gold_layer", business_intent: str = "") -> list[str]:
    """Aggregate Silver layer data into Gold layer with LLM-guided KPIs.

    Reads from Silver Parquet outputs and creates:
    - Business-aligned KPI aggregations
    - Dimensional hierarchies (time, product, location)
    - Intent-driven metrics and summaries
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    llm = _make_llm()

    # Process each silver file
    for path in silver_output_paths:
        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(f"Silver source file not found: {src}")

        df = pd.read_parquet(src)

        # 1. LLM-guided aggregation strategy
        aggregation_guidance = ""
        if llm and business_intent:
            try:
                columns = [c for c in df.columns if not c.startswith("_")]
                numeric_cols = df[columns].select_dtypes(include=[float, int]).columns.tolist()
                dim_cols = df[columns].select_dtypes(include=["object", "category"]).columns.tolist()
                date_cols = df[columns].select_dtypes(include=["datetime64"]).columns.tolist()

                prompt = f"""Given this business question: "{business_intent}"
And this data structure:
- Dimensions (grouping): {dim_cols}
- Metrics (aggregation): {numeric_cols}
- Time columns: {date_cols}

What KPIs and aggregations should we compute? (2-3 key metrics)
Keep response concise (2-3 sentences)."""

                response = llm.invoke(prompt)
                aggregation_guidance = response.content[:300] if hasattr(response, 'content') else ""
            except Exception:
                aggregation_guidance = "LLM guidance unavailable"

        # 2. Identify and standardize columns for aggregation
        numeric_cols = df.select_dtypes(include=[float, int]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if not c.startswith("_")]

        dim_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        dim_cols = [c for c in dim_cols if not c.startswith("_")]

        # 3. Create Gold-layer aggregations
        gold_df = df.copy()

        # Add KPI columns based on available metrics
        for col in numeric_cols:
            col_lower = col.lower()
            # Revenue/Sales KPIs
            if any(x in col_lower for x in ["amount", "sales", "revenue"]):
                gold_df[f"{col}_total"] = gold_df[col]
                if dim_cols:
                    gold_df[f"{col}_per_transaction"] = gold_df.groupby(dim_cols[0], group_keys=False)[col].transform("mean")
            # Quantity KPIs
            elif "quantity" in col_lower or "count" in col_lower:
                gold_df[f"{col}_total"] = gold_df[col]

        # 4. Create dimensional aggregations if enough dimensions exist
        if len(dim_cols) >= 1 and len(numeric_cols) >= 1:
            # Group by first dimension
            agg_cols = {col: "sum" for col in numeric_cols}
            try:
                dim_agg = gold_df.groupby(dim_cols[0], as_index=False).agg(agg_cols)
                # Merge back as enrichment
                gold_df = gold_df.merge(
                    dim_agg.rename(columns={col: f"{col}_by_{dim_cols[0]}" for col in agg_cols.keys()}),
                    on=dim_cols[0],
                    how="left"
                )
            except Exception:
                pass

        # 5. Add Gold metadata
        gold_df["_gold_aggregated_at"] = pd.Timestamp.now()
        gold_df["_gold_intent"] = business_intent if business_intent else "general_analytics"
        gold_df["_gold_llm_guidance"] = aggregation_guidance

        target = output / f"{src.stem.replace('_silver', '')}_gold.parquet"
        gold_df.to_parquet(target, index=False)
        written.append(str(target))

    return written
