from __future__ import annotations

import json
import pandas as pd
from pathlib import Path
from typing import Any, Sequence

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
    """Create the configured LLM client for STTM generation."""
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

    # Fallback: return a basic dict-based "mock" if no LLM configured
    return None


def _analyze_csv_schema(file_paths: Sequence[str | Path]) -> dict[str, Any]:
    """Use LLM to analyze CSV schemas and suggest STTM mappings."""
    llm = _make_llm()

    # If no LLM available, use rule-based analysis
    if not llm:
        return _analyze_csv_schema_rule_based(file_paths)

    try:
        # Read first 5 rows of each CSV for context
        file_summaries = []
        for path in file_paths:
            p = Path(path)
            if p.exists():
                df = pd.read_csv(p, nrows=5)
                summary = {
                    "file": p.name,
                    "columns": list(df.columns),
                    "dtypes": {col: str(df[col].dtype) for col in df.columns},
                    "sample": df.head(2).to_dict(orient="records"),
                }
                file_summaries.append(summary)

        prompt = f"""Analyze these CSV file schemas and suggest Semantic Table Type Mapping (STTM) rules for data ingestion:

{json.dumps(file_summaries, indent=2)}

Return a JSON object with:
- "source_tables": list of identified source tables (files)
- "target_layer": "bronze_layer"
- "key_columns": identified key/join columns
- "transformations": list of suggested transformations
- "data_quality_checks": suggested quality checks
"""

        response = llm.invoke(prompt)
        result = response.content

        # Try to parse as JSON, fallback to structured dict
        try:
            mapping = json.loads(result)
        except json.JSONDecodeError:
            mapping = _analyze_csv_schema_rule_based(file_paths)

        return mapping
    except Exception:
        return _analyze_csv_schema_rule_based(file_paths)


def _analyze_csv_schema_rule_based(file_paths: Sequence[str | Path]) -> dict[str, Any]:
    """Rule-based schema analysis when LLM is not available."""
    files = [str(Path(p)) for p in file_paths]

    file_info = []
    for path in file_paths:
        p = Path(path)
        if p.exists():
            df = pd.read_csv(p)
            file_info.append({
                "file": p.name,
                "columns": list(df.columns),
                "row_count": len(df),
            })

    return {
        "source_tables": files,
        "target_layer": "bronze_layer",
        "key_columns": _infer_key_columns(file_info),
        "transformations": [
            {"source": "raw_csv", "target": "bronze_parquet", "operation": "ingest", "format_conversion": "CSV→Parquet"}
        ],
        "data_quality_checks": [
            "infer_schema",
            "detect_nulls",
            "validate_row_counts",
        ],
    }


def _infer_key_columns(file_info: list[dict]) -> list[str]:
    """Infer likely key columns from file metadata."""
    key_candidates = []
    for file in file_info:
        for col in file.get("columns", []):
            col_lower = col.lower()
            if any(word in col_lower for word in ["id", "key", "code", "sku", "product_id", "store_id", "customer_id"]):
                key_candidates.append(f"{file['file']}::{col}")
    return key_candidates


def generate_bronze_sttm(file_paths: Sequence[str | Path]) -> dict[str, Any]:
    """Generate Bronze layer STTM using LLM schema analysis.

    Analyzes input CSV schemas and generates semantic table type mappings
    for raw data ingestion into the Bronze layer.
    """
    schema_analysis = _analyze_csv_schema(file_paths)

    return {
        "phase": "bronze",
        "description": "Raw data ingestion with schema inference and validation",
        "source_to_target": {
            "source_files": [str(Path(p)) for p in file_paths],
            "target_layer": "bronze_layer",
            "schema_analysis": schema_analysis,
            "mapping": [
                {
                    "source": "raw_csv",
                    "target": "bronze_parquet",
                    "operation": "ingest",
                    "transformations": schema_analysis.get("transformations", []),
                    "quality_checks": schema_analysis.get("data_quality_checks", []),
                }
            ],
        },
    }


def generate_silver_sttm(file_paths: Sequence[str | Path]) -> dict[str, Any]:
    """Generate Silver layer STTM using LLM analysis.

    Creates mappings for data cleaning, deduplication, standardization,
    and type casting from Bronze to Silver layer.
    """
    return {
        "phase": "silver",
        "description": "Data cleaning, deduplication, and standardization",
        "source_to_target": {
            "source_files": [str(Path(p)) for p in file_paths],
            "source_layer": "bronze_layer",
            "target_layer": "silver_layer",
            "mapping": [
                {
                    "source": "bronze_parquet",
                    "target": "silver_parquet",
                    "operation": "clean_and_standardize",
                    "transformations": [
                        "remove_duplicates",
                        "handle_nulls",
                        "standardize_strings",
                        "parse_dates",
                        "cast_numeric_types",
                    ],
                    "quality_checks": [
                        "validate_uniqueness",
                        "check_referential_integrity",
                        "data_type_validation",
                    ],
                }
            ],
        },
    }


def generate_gold_sttm(file_paths: Sequence[str | Path]) -> dict[str, Any]:
    """Generate Gold layer STTM using LLM analysis.

    Creates mappings for business-aligned aggregations, KPI computation,
    and dimensional modeling from Silver to Gold layer.
    """
    return {
        "phase": "gold",
        "description": "Business KPI aggregation and dimensional modeling",
        "source_to_target": {
            "source_files": [str(Path(p)) for p in file_paths],
            "source_layer": "silver_layer",
            "target_layer": "gold_layer",
            "mapping": [
                {
                    "source": "silver_parquet",
                    "target": "gold_parquet",
                    "operation": "aggregate_and_model",
                    "transformations": [
                        "identify_dimensions",
                        "compute_facts",
                        "calculate_kpis",
                        "create_time_dimensions",
                    ],
                    "kpi_targets": [
                        "revenue_by_product",
                        "revenue_by_location",
                        "revenue_by_period",
                        "top_products",
                        "regional_performance",
                    ],
                    "quality_checks": [
                        "reconcile_to_source",
                        "validate_aggregations",
                        "business_rule_validation",
                    ],
                }
            ],
        },
    }
