from __future__ import annotations

import json
import csv
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from core.config import (
    LLM_PROVIDER, GROQ_API_KEY, GROQ_MODEL,
    OPENAI_API_KEY, OPENAI_MODEL,
    GOOGLE_API_KEY, GEMINI_MODEL,
    GITHUB_TOKEN, GITHUB_MODEL, GITHUB_BASE_URL,
)


def _make_llm() -> Any:
    provider = (LLM_PROVIDER or "").strip().lower()
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, reasoning_format="hidden", reasoning_effort="low")
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(api_key=OPENAI_API_KEY, model=OPENAI_MODEL)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY, model=GEMINI_MODEL)
    if provider == "github":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(api_key=GITHUB_TOKEN, model=GITHUB_MODEL, base_url=GITHUB_BASE_URL)
    return None


def _parse_llm_json(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        parts = content.split("```")
        content = parts[1] if len(parts) > 1 else content
        if content.startswith("json"):
            content = content[4:]
    try:
        return json.loads(content.strip())
    except json.JSONDecodeError:
        return {}


def _infer_dtype(series: pd.Series) -> str:
    dtype = str(series.dtype)
    if "int" in dtype:
        return "integer"
    if "float" in dtype:
        return "float"
    if "datetime" in dtype:
        return "datetime"
    if "bool" in dtype:
        return "boolean"
    return "string"


def _default_transform(col: str, dtype: str) -> str:
    col_l = col.lower()
    if any(k in col_l for k in ["name", "category", "region", "city", "state", "segment", "method"]):
        return "trim"
    if any(k in col_l for k in ["date", "time"]):
        return "parse_date"
    if any(k in col_l for k in ["amount", "price", "revenue", "profit"]):
        return "cast_float"
    if any(k in col_l for k in ["quantity", "count", "qty"]):
        return "cast_integer"
    return "keep"


def save_sttm_csv(sttm: dict, path: str | Path) -> None:
    """Write the STTM rules list to a CSV file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rules = sttm.get("rules", [])
    if not rules:
        p.write_text("source_file,source_column,target_column,data_type,transformation_logic\n", encoding="utf-8")
        return
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["source_file", "source_column", "target_column", "data_type", "transformation_logic"])
    writer.writeheader()
    writer.writerows(rules)
    p.write_text(buf.getvalue(), encoding="utf-8")


def generate_bronze_sttm(file_paths: Sequence[str | Path], business_intent: str = "") -> dict[str, Any]:
    """Generate Bronze STTM with Objectives, Rules (column-level CSV mapping), and Notes."""
    llm = _make_llm()

    # Build column-level rules from CSV schemas
    rules: list[dict] = []
    file_summaries = []
    for path in file_paths:
        p = Path(path)
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p, nrows=10)
            for col in df.columns:
                rules.append({
                    "source_file": p.name,
                    "source_column": col,
                    "target_column": col,
                    "data_type": _infer_dtype(df[col]),
                    "transformation_logic": _default_transform(col, _infer_dtype(df[col])),
                })
            file_summaries.append({
                "file": p.name,
                "columns": list(df.columns),
                "dtypes": {c: _infer_dtype(df[c]) for c in df.columns},
                "sample_nulls": {c: int(df[c].isnull().sum()) for c in df.columns},
                "row_count": len(pd.read_csv(p)),
            })
        except Exception:
            pass

    # Default objectives and notes
    objectives = [
        f"Load raw data from {len(file_paths)} CSV source(s) into the Bronze layer with minimal transformation.",
        "Preserve all columns including potential join keys (product_id, store_id) for downstream Silver/Gold joins.",
        "Apply light-touch ingestion: whitespace trimming on strings, basic type casting only.",
        "Keep all rows including nulls and duplicates; Bronze is the raw historical record.",
        "Validate row counts match source CSV after ingestion and flag any load failures.",
    ]
    notes = [
        "Bronze stores raw data as-is; data quality issues will be addressed in Silver.",
        "All source columns are preserved to ensure no information is lost at ingestion.",
        "Metadata columns (_bronze_loaded_at, _bronze_source_file) added for lineage tracking.",
    ]

    if llm:
        try:
            prompt = f"""You are a data engineer generating a Semantic Table Type Mapping (STTM) for Bronze ingestion.

Business intent: "{business_intent}"
Source files: {json.dumps(file_summaries, indent=2, default=str)[:2000]}

Return JSON with exactly these keys:
- "objectives": list of 5-6 specific ingestion objectives as strings
- "notes": list of 4-6 data quality observations as strings
- "column_overrides": list of {{source_file, source_column, transformation_logic}} overrides (only non-default transforms)

Keep responses concise."""
            response = llm.invoke(prompt)
            parsed = _parse_llm_json(response.content if hasattr(response, "content") else str(response))
            if parsed.get("objectives"):
                objectives = parsed["objectives"]
            if parsed.get("notes"):
                notes = parsed["notes"]
            # Apply any LLM column overrides
            override_map = {(o["source_file"], o["source_column"]): o["transformation_logic"]
                            for o in parsed.get("column_overrides", [])
                            if "source_file" in o and "source_column" in o}
            for rule in rules:
                key = (rule["source_file"], rule["source_column"])
                if key in override_map:
                    rule["transformation_logic"] = override_map[key]
        except Exception:
            pass

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    log: list[str] = [
        f"[{ts}] Bronze STTM generation started",
        f"[{ts}] LLM provider: {(LLM_PROVIDER or 'none').upper()}",
        f"[{ts}] Source files analysed: {len(file_summaries)}",
    ]
    for fs in file_summaries:
        log.append(f"[{ts}]   • {fs['file']} — {fs['row_count']} rows, {len(fs['columns'])} columns")
        null_cols = [c for c, n in fs.get('sample_nulls', {}).items() if n > 0]
        if null_cols:
            log.append(f"[{ts}]     Nulls detected in: {', '.join(null_cols)}")
    log.append(f"[{ts}] Column rules generated: {len(rules)}")
    log.append(f"[{ts}] LLM objectives/notes: {'applied' if llm else 'skipped (no LLM)'}")
    log.append(f"[{ts}] Bronze STTM ready for HITL approval")

    return {
        "phase": "bronze",
        "business_intent": business_intent,
        "objectives": objectives,
        "rules": rules,
        "notes": notes,
        "log": log,
    }


def generate_silver_sttm(
    bronze_paths: Sequence[str | Path],
    business_intent: str = "",
) -> dict[str, Any]:
    """Generate Silver STTM by reading actual Bronze Parquet schemas."""
    llm = _make_llm()

    rules: list[dict] = []
    file_summaries = []
    for path in bronze_paths:
        p = Path(path)
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p)
            data_cols = [c for c in df.columns if not c.startswith("_")]
            for col in data_cols:
                rules.append({
                    "source_file": p.name,
                    "source_column": col,
                    "target_column": col,
                    "data_type": _infer_dtype(df[col]),
                    "transformation_logic": _default_transform(col, _infer_dtype(df[col])),
                })
            null_counts = {c: int(df[c].isnull().sum()) for c in data_cols[:15]}
            file_summaries.append({
                "file": p.name,
                "columns": data_cols,
                "dtypes": {c: _infer_dtype(df[c]) for c in data_cols},
                "null_counts": null_counts,
                "row_count": len(df),
                "duplicate_count": int(df.duplicated().sum()),
            })
        except Exception:
            pass

    objectives = [
        "Deduplicate records using primary key columns identified in the Bronze schema.",
        "Standardise string columns: strip whitespace, normalise capitalisation (title case for names, upper for codes).",
        "Parse and validate date columns to ISO 8601 datetime format.",
        "Cast numeric columns (amount, price, quantity) to correct float/integer types.",
        f"Apply business-intent-driven filtering aligned to: '{business_intent}'.",
        "Add Silver metadata columns (_silver_cleaned_at, _silver_business_intent) for lineage.",
    ]
    notes = [
        "Silver retains only STTM-approved columns; all others are dropped.",
        "Null handling: flag high-null columns, fill or drop based on column criticality.",
        "Referential integrity: rows with missing join keys (product_id, store_id) are flagged.",
        "Category and text fields are normalised to prevent groupby mismatches in Gold.",
    ]

    if llm:
        try:
            prompt = f"""You are a data engineer generating a Silver layer STTM.

Business intent: "{business_intent}"
Bronze Parquet schemas: {json.dumps(file_summaries, indent=2, default=str)[:2000]}

Return JSON with exactly these keys:
- "objectives": list of 5-6 specific Silver cleaning objectives
- "notes": list of 4-5 data quality observations
- "column_overrides": list of {{source_file, source_column, transformation_logic}} for non-default transforms

Focus on: deduplication, date parsing, type casting, null handling, string normalisation."""
            response = llm.invoke(prompt)
            parsed = _parse_llm_json(response.content if hasattr(response, "content") else str(response))
            if parsed.get("objectives"):
                objectives = parsed["objectives"]
            if parsed.get("notes"):
                notes = parsed["notes"]
            override_map = {(o["source_file"], o["source_column"]): o["transformation_logic"]
                            for o in parsed.get("column_overrides", [])
                            if "source_file" in o and "source_column" in o}
            for rule in rules:
                key = (rule["source_file"], rule["source_column"])
                if key in override_map:
                    rule["transformation_logic"] = override_map[key]
        except Exception:
            pass

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    log: list[str] = [
        f"[{ts}] Silver STTM generation started",
        f"[{ts}] LLM provider: {(LLM_PROVIDER or 'none').upper()}",
        f"[{ts}] Bronze Parquet files read: {len(file_summaries)}",
    ]
    for fs in file_summaries:
        log.append(f"[{ts}]   • {fs['file']} — {fs['row_count']} rows, {fs.get('duplicate_count', 0)} duplicates")
        nulls = {c: n for c, n in fs.get('null_counts', {}).items() if n > 0}
        if nulls:
            log.append(f"[{ts}]     Null counts: " + ", ".join(f"{c}={n}" for c, n in list(nulls.items())[:6]))
    log.append(f"[{ts}] Cleaning rules generated: {len(rules)}")
    log.append(f"[{ts}] LLM cleaning objectives: {'applied' if llm else 'skipped (no LLM)'}")
    log.append(f"[{ts}] Planned transforms: dedup, date-parse, type-cast, title-case, null-fill")
    log.append(f"[{ts}] Silver STTM ready for HITL approval")

    return {
        "phase": "silver",
        "business_intent": business_intent,
        "objectives": objectives,
        "rules": rules,
        "notes": notes,
        "log": log,
    }


def generate_gold_sttm(
    silver_paths: Sequence[str | Path],
    business_intent: str = "",
) -> dict[str, Any]:
    """Generate Gold STTM by reading actual Silver Parquet schemas."""
    llm = _make_llm()

    rules: list[dict] = []
    file_summaries = []
    for path in silver_paths:
        p = Path(path)
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p)
            data_cols = [c for c in df.columns if not c.startswith("_")]
            numeric_cols = df[data_cols].select_dtypes(include=["number"]).columns.tolist()
            dim_cols = df[data_cols].select_dtypes(include=["object", "category"]).columns.tolist()
            for col in data_cols:
                is_numeric = col in numeric_cols
                transform = f"SUM({col})" if is_numeric and any(k in col.lower() for k in ["amount", "revenue", "price", "total"]) \
                    else f"COUNT({col})" if is_numeric and any(k in col.lower() for k in ["quantity", "count"]) \
                    else "GROUP_BY" if col in dim_cols and any(k in col.lower() for k in ["id", "name", "region", "category", "year", "month"]) \
                    else "keep"
                rules.append({
                    "source_file": p.name,
                    "source_column": col,
                    "target_column": col,
                    "data_type": _infer_dtype(df[col]),
                    "transformation_logic": transform,
                })
            file_summaries.append({
                "file": p.name,
                "numeric_columns": numeric_cols,
                "dimension_columns": dim_cols,
                "row_count": len(df),
            })
        except Exception:
            pass

    objectives = [
        f"Answer the business question: '{business_intent}'.",
        "Join Silver tables on shared _id keys (product_id, store_id) to create an enriched fact table.",
        "Aggregate revenue metrics: SUM(total_amount) by product, region, and time period.",
        "Compute KPI tables: revenue_by_product, revenue_by_region, revenue_by_period.",
        "Inject pk_gold_id and Gold metadata for lineage and auditability.",
        "Write one Parquet file per target KPI table for efficient downstream querying.",
    ]
    notes = [
        "Gold layer uses DuckDB for SQL-based joins and aggregations.",
        "All dimension columns (product_name, region, city) are resolved via Silver joins.",
        "Time dimensions (_year, _month) are extracted from transaction_date for trend analysis.",
        "Business intent drives which KPI tables are prioritised.",
    ]

    if llm:
        try:
            prompt = f"""You are a data engineer generating a Gold layer STTM.

Business intent: "{business_intent}"
Silver schemas: {json.dumps(file_summaries, indent=2, default=str)[:2000]}

Return JSON with exactly these keys:
- "objectives": list of 5-6 Gold aggregation objectives
- "notes": list of 4-5 notes on the aggregation approach
- "column_overrides": list of {{source_file, source_column, transformation_logic}} for KPI columns

Focus on: dimensional joins, revenue aggregations, KPI table design."""
            response = llm.invoke(prompt)
            parsed = _parse_llm_json(response.content if hasattr(response, "content") else str(response))
            if parsed.get("objectives"):
                objectives = parsed["objectives"]
            if parsed.get("notes"):
                notes = parsed["notes"]
            override_map = {(o["source_file"], o["source_column"]): o["transformation_logic"]
                            for o in parsed.get("column_overrides", [])
                            if "source_file" in o and "source_column" in o}
            for rule in rules:
                key = (rule["source_file"], rule["source_column"])
                if key in override_map:
                    rule["transformation_logic"] = override_map[key]
        except Exception:
            pass

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    log: list[str] = [
        f"[{ts}] Gold STTM generation started",
        f"[{ts}] LLM provider: {(LLM_PROVIDER or 'none').upper()}",
        f"[{ts}] Silver Parquet files read: {len(file_summaries)}",
    ]
    for fs in file_summaries:
        log.append(f"[{ts}]   • {fs['file']} — {fs['row_count']} rows")
        if fs.get("numeric_columns"):
            log.append(f"[{ts}]     Numeric cols (to aggregate): {', '.join(fs['numeric_columns'][:6])}")
        if fs.get("dimension_columns"):
            log.append(f"[{ts}]     Dimension cols (to GROUP BY): {', '.join(fs['dimension_columns'][:6])}")
    log.append(f"[{ts}] KPI aggregation rules generated: {len(rules)}")
    log.append(f"[{ts}] LLM Gold objectives: {'applied' if llm else 'skipped (no LLM)'}")
    log.append(f"[{ts}] Planned KPIs: revenue_by_product, revenue_by_region, revenue_by_period")
    log.append(f"[{ts}] Gold STTM ready for HITL approval")

    return {
        "phase": "gold",
        "business_intent": business_intent,
        "objectives": objectives,
        "rules": rules,
        "notes": notes,
        "log": log,
    }
