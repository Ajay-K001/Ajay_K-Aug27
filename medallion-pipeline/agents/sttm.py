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


def _parse_llm_json(content: str) -> dict:
    """Strip markdown code fences and parse JSON from LLM response."""
    content = content.strip()
    if content.startswith("```"):
        parts = content.split("```")
        content = parts[1] if len(parts) > 1 else content
        if content.startswith("json"):
            content = content[4:]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {}


def _analyze_csv_schema(file_paths: Sequence[str | Path]) -> dict[str, Any]:
    """Use LLM to analyze CSV schemas and suggest Bronze STTM mappings."""
    llm = _make_llm()

    if not llm:
        return _analyze_csv_schema_rule_based(file_paths)

    try:
        file_summaries = []
        for path in file_paths:
            p = Path(path)
            if p.exists():
                df = pd.read_csv(p, nrows=5)
                file_summaries.append({
                    "file": p.name,
                    "columns": list(df.columns),
                    "dtypes": {col: str(df[col].dtype) for col in df.columns},
                    "sample": df.head(2).to_dict(orient="records"),
                })

        prompt = f"""Analyze these CSV file schemas and suggest Semantic Table Type Mapping (STTM) rules for Bronze layer ingestion:

{json.dumps(file_summaries, indent=2, default=str)[:2000]}

Return a JSON object with:
- "source_tables": list of identified source tables (file names)
- "target_layer": "bronze_layer"
- "key_columns": identified key/join columns (e.g. "sales_data.csv::transaction_id")
- "transformations": list of suggested transformations
- "data_quality_checks": suggested quality checks
"""
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        mapping = _parse_llm_json(content)
        if not mapping:
            mapping = _analyze_csv_schema_rule_based(file_paths)
        return mapping
    except Exception:
        return _analyze_csv_schema_rule_based(file_paths)


def _analyze_csv_schema_rule_based(file_paths: Sequence[str | Path]) -> dict[str, Any]:
    files = [str(Path(p)) for p in file_paths]
    file_info = []
    for path in file_paths:
        p = Path(path)
        if p.exists():
            df = pd.read_csv(p)
            file_info.append({"file": p.name, "columns": list(df.columns), "row_count": len(df)})

    return {
        "source_tables": files,
        "target_layer": "bronze_layer",
        "key_columns": _infer_key_columns(file_info),
        "transformations": [
            {"source": "raw_csv", "target": "bronze_parquet", "operation": "ingest", "format_conversion": "CSV→Parquet"}
        ],
        "data_quality_checks": ["infer_schema", "detect_nulls", "validate_row_counts"],
    }


def _infer_key_columns(file_info: list[dict]) -> list[str]:
    key_candidates = []
    for file in file_info:
        for col in file.get("columns", []):
            col_lower = col.lower()
            if any(word in col_lower for word in ["id", "key", "code", "sku"]):
                key_candidates.append(f"{file['file']}::{col}")
    return key_candidates


def generate_bronze_sttm(file_paths: Sequence[str | Path]) -> dict[str, Any]:
    """Generate Bronze layer STTM using LLM CSV schema analysis."""
    schema_analysis = _analyze_csv_schema(file_paths)

    return {
        "phase": "bronze",
        "description": "Raw data ingestion with LLM schema inference and validation",
        "source_to_target": {
            "source_files": [str(Path(p)) for p in file_paths],
            "target_layer": "bronze_layer",
            "llm_schema_analysis": schema_analysis,
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


def generate_silver_sttm(
    bronze_paths: Sequence[str | Path],
    business_intent: str = "",
) -> dict[str, Any]:
    """Generate Silver layer STTM using LLM analysis of actual Bronze Parquet schema."""
    llm = _make_llm()

    # Read actual Bronze Parquet schema
    file_summaries = []
    for path in bronze_paths:
        p = Path(path)
        if p.exists():
            try:
                df = pd.read_parquet(p)
                data_cols = [c for c in df.columns if not c.startswith("_")]
                numeric_cols = df[data_cols].select_dtypes(include=[float, int]).columns.tolist()
                string_cols = df[data_cols].select_dtypes(include=["object"]).columns.tolist()
                date_cols = [c for c in data_cols if "date" in c.lower() or "time" in c.lower()]
                file_summaries.append({
                    "file": p.name,
                    "columns": data_cols,
                    "numeric_columns": numeric_cols,
                    "string_columns": string_cols,
                    "date_columns": date_cols,
                    "row_count": len(df),
                    "null_counts": {col: int(df[col].isnull().sum()) for col in data_cols[:10]},
                })
            except Exception:
                pass

    llm_analysis: dict[str, Any] = {}
    if llm and file_summaries:
        try:
            prompt = f"""Analyze these Bronze layer Parquet schemas and generate Silver transformation rules:

Business Question: "{business_intent}"

Bronze Schema:
{json.dumps(file_summaries, indent=2, default=str)[:2000]}

Return valid JSON with:
- "key_columns": list of primary/join key column names
- "transformations": list of specific cleaning operations (dedup, date parsing, type casting, string normalization)
- "columns_to_standardize": string columns needing normalization
- "date_columns": date columns to parse to datetime
- "numeric_columns": numeric columns to cast/validate
- "data_quality_checks": quality validations to run
- "business_context": how the business question affects cleaning decisions"""

            response = llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            llm_analysis = _parse_llm_json(content)
        except Exception:
            llm_analysis = {}

    transformations = llm_analysis.get("transformations", [
        "remove_duplicates",
        "handle_nulls",
        "standardize_strings",
        "parse_dates",
        "cast_numeric_types",
    ])

    return {
        "phase": "silver",
        "description": "LLM-guided data cleaning, deduplication, and standardization",
        "business_intent": business_intent,
        "source_to_target": {
            "source_files": [str(Path(p)) for p in bronze_paths],
            "source_layer": "bronze_layer",
            "target_layer": "silver_layer",
            "llm_analysis": llm_analysis,
            "mapping": [
                {
                    "source": "bronze_parquet",
                    "target": "silver_parquet",
                    "operation": "clean_and_standardize",
                    "transformations": transformations,
                    "key_columns": llm_analysis.get("key_columns", []),
                    "columns_to_standardize": llm_analysis.get("columns_to_standardize", []),
                    "date_columns": llm_analysis.get("date_columns", []),
                    "quality_checks": llm_analysis.get("data_quality_checks", [
                        "validate_uniqueness",
                        "check_referential_integrity",
                        "data_type_validation",
                    ]),
                }
            ],
        },
    }


def generate_gold_sttm(
    silver_paths: Sequence[str | Path],
    business_intent: str = "",
) -> dict[str, Any]:
    """Generate Gold layer STTM using LLM analysis of actual Silver Parquet schema."""
    llm = _make_llm()

    # Read actual Silver Parquet schema
    file_summaries = []
    for path in silver_paths:
        p = Path(path)
        if p.exists():
            try:
                df = pd.read_parquet(p)
                data_cols = [c for c in df.columns if not c.startswith("_")]
                numeric_cols = df[data_cols].select_dtypes(include=[float, int]).columns.tolist()
                dim_cols = df[data_cols].select_dtypes(include=["object", "category"]).columns.tolist()
                date_cols = df[data_cols].select_dtypes(include=["datetime64"]).columns.tolist()
                file_summaries.append({
                    "file": p.name,
                    "numeric_columns": numeric_cols,
                    "dimension_columns": dim_cols,
                    "date_columns": date_cols,
                    "row_count": len(df),
                })
            except Exception:
                pass

    llm_analysis: dict[str, Any] = {}
    if llm and file_summaries:
        try:
            prompt = f"""Analyze these Silver layer schemas and design Gold layer KPI aggregations:

Business Question: "{business_intent}"

Silver Schema:
{json.dumps(file_summaries, indent=2, default=str)[:2000]}

Return valid JSON with:
- "kpi_metrics": list of specific KPIs to compute (e.g., "revenue_by_product", "monthly_sales_trend", "top_stores_by_revenue")
- "dimensions": list of grouping dimensions (e.g., "product_id", "region", "year", "month")
- "transformations": list of aggregation operations to apply
- "data_quality_checks": validation checks on aggregated data
- "business_context": how the business question maps to these KPIs"""

            response = llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            llm_analysis = _parse_llm_json(content)
        except Exception:
            llm_analysis = {}

    kpi_targets = llm_analysis.get("kpi_metrics", [
        "revenue_by_product",
        "revenue_by_location",
        "revenue_by_period",
        "top_products",
        "regional_performance",
    ])

    return {
        "phase": "gold",
        "description": "LLM-guided business KPI aggregation and dimensional modeling",
        "business_intent": business_intent,
        "source_to_target": {
            "source_files": [str(Path(p)) for p in silver_paths],
            "source_layer": "silver_layer",
            "target_layer": "gold_layer",
            "llm_analysis": llm_analysis,
            "mapping": [
                {
                    "source": "silver_parquet",
                    "target": "gold_parquet",
                    "operation": "aggregate_and_model",
                    "transformations": llm_analysis.get("transformations", [
                        "identify_dimensions",
                        "compute_revenue_kpis",
                        "calculate_time_based_aggregations",
                        "rank_top_products",
                        "compute_regional_performance",
                    ]),
                    "kpi_targets": kpi_targets,
                    "dimensions": llm_analysis.get("dimensions", []),
                    "quality_checks": llm_analysis.get("data_quality_checks", [
                        "reconcile_to_source",
                        "validate_aggregations",
                        "business_rule_validation",
                    ]),
                }
            ],
        },
    }
