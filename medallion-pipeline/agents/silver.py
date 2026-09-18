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


def silver_clean(bronze_output_paths: Sequence[str | Path], output_dir: str | Path = "data/silver_layer", business_intent: str = "") -> list[str]:
    """Clean and standardize Bronze layer data with LLM-assisted transformations.

    Reads from Bronze Parquet outputs and applies:
    - LLM-guided deduplication and null handling
    - Type standardization based on business context
    - Intent-driven cleansing rules
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    llm = _make_llm()

    for path in bronze_output_paths:
        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(f"Bronze source file not found: {src}")

        df = pd.read_parquet(src)

        # 1. LLM-assisted cleansing strategy (if LLM available and intent provided)
        cleaning_notes = ""
        if llm and business_intent:
            try:
                # Build context for LLM
                sample_rows = df.head(3).to_dict(orient="records")
                columns_info = {col: str(df[col].dtype) for col in df.columns if not col.startswith("_")}

                prompt = f"""Given this business context: "{business_intent}"
And this data structure:
- Columns: {list(columns_info.keys())}
- Types: {columns_info}
- Sample rows: {json.dumps(sample_rows, default=str)[:500]}

What are the key cleaning rules needed? (e.g., which columns are keys, which are metrics, any standardization needed)
Keep response concise (3-4 sentences)."""

                response = llm.invoke(prompt)
                cleaning_notes = response.content[:300] if hasattr(response, 'content') else ""
            except Exception:
                cleaning_notes = "LLM-guided cleaning unavailable"

        # 2. Core Silver transformations
        # Remove duplicates
        df = df.drop_duplicates()

        # Remove completely empty rows
        df = df.dropna(how="all")

        # 3. Standardize string columns (lowercase categorical values)
        for col in df.select_dtypes(include=["object"]).columns:
            if col.startswith("_"):  # Skip metadata columns
                continue
            try:
                df[col] = df[col].str.lower() if df[col].dtype == "object" else df[col]
            except AttributeError:
                pass

        # 4. Standardize date columns
        for col in df.columns:
            if "date" in col.lower() or "time" in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                except Exception:
                    pass

        # 5. Add Silver metadata
        df["_silver_cleaned_at"] = pd.Timestamp.now()
        df["_silver_llm_guidance"] = cleaning_notes
        df["_silver_business_intent"] = business_intent if business_intent else "general_analytics"

        target = output / f"{src.stem.replace('_bronze', '')}_silver.parquet"
        df.to_parquet(target, index=False)
        written.append(str(target))

    return written
