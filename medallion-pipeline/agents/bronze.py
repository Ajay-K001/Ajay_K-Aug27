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



def bronze_ingest(file_paths: Sequence[str | Path], output_dir: str | Path = "data/bronze_layer") -> list[str]:
    """Ingest raw CSV files into Bronze layer with LLM-assisted validation.

    Uses LLM to analyze schema quality, null patterns, and data anomalies,
    then applies Bronze-layer standardization and metadata tracking.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    llm = _make_llm()

    for path in file_paths:
        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(f"Source file not found: {src}")

        df = pd.read_csv(src)

        # 1. LLM-assisted schema validation (if LLM available)
        if llm:
            try:
                schema_summary = {
                    "filename": src.name,
                    "columns": list(df.columns),
                    "dtypes": {col: str(df[col].dtype) for col in df.columns},
                    "null_counts": df.isnull().sum().to_dict(),
                    "row_count": len(df),
                }

                prompt = f"""Analyze this CSV schema and provide a brief validation summary:
{json.dumps(schema_summary, indent=2)}

Is this data ready for Bronze ingestion? Note any critical issues (data type mismatches, excessive nulls, etc).
Keep response concise (2-3 sentences)."""

                response = llm.invoke(prompt)
                validation_note = response.content[:200] if hasattr(response, 'content') else ""
            except Exception:
                validation_note = "LLM validation unavailable"
        else:
            validation_note = "No LLM configured"

        # 2. Infer and validate schema
        df = df.infer_objects(copy=False)

        # 3. Track data quality metrics
        row_count = len(df)
        null_counts = df.isnull().sum().to_dict()

        # 4. Add Bronze metadata
        df["_bronze_loaded_at"] = pd.Timestamp.now()
        df["_bronze_source_file"] = src.name
        df["_bronze_llm_validation"] = validation_note
        df["_bronze_row_count"] = row_count

        target = output / f"{src.stem}_bronze.parquet"
        df.to_parquet(target, index=False)
        written.append(str(target))

    return written
